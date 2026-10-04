#!/usr/bin/env python3
"""Predeclared actor feature-unit ablation; defaults to no-gradient preflight."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import run_procedural_transfer as physical
from run_patient_learning import FrozenSequence, frozen_population_rollouts, preserve_source, write_json
from resectionlab.evaluation import freeze_candidates
from resectionlab.learning import (MaskedPatientPolicy, TrainingConfig, load_policy,
    policy_hash, rollout_policy, train_patient_policy)
from resectionlab.worlds import FrozenDecisionModel, content_hash

STUDY_ID = 'procedural-native-feature-units-v1'
DECLARATION_PATH = Path('manifests/experiments') / f'{STUDY_ID}.json'
DECLARATION_HASH = 'sha256:b19aac85f9241e37b98ae53d539ad923f3258bf42a106d67e520f37abcf6a237'
SCOPE = 'procedural_native_feature_units_development'
PROFILES = ('RAW', 'FEATURE_UNITS')
SEEDS = (11, 23, 47)


def assert_declaration(value: dict) -> None:
    payload = dict(value)
    supplied = payload.pop('declaration_content_hash', None)
    if supplied != DECLARATION_HASH or content_hash(payload) != supplied:
        raise ValueError('Feature-unit declaration differs from committed preregistration')


def load_declaration(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert_declaration(value)
    return value


def study_source_snapshot(root: Path = ROOT) -> dict:
    snapshot = physical.transfer_source_snapshot(root)
    name = 'scripts/run_native_feature_units.py'
    snapshot['numerical_runtime_sha256'][name] = snapshot['file_sha256'][name]
    snapshot['numerical_runtime_content_hash'] = content_hash(snapshot['numerical_runtime_sha256'])
    return snapshot


def assert_source_unchanged(snapshot: dict, root: Path = ROOT) -> None:
    if study_source_snapshot(root)['numerical_runtime_content_hash'] != snapshot['numerical_runtime_content_hash']:
        raise ValueError('SOURCE_CHANGED_DURING_RUN: preserve this attempt')


def preserve_inputs(declaration: dict, reference: dict, destination: Path, root: Path = ROOT) -> None:
    physical.preserve_declaration_inputs(reference, destination, root)
    path = destination / DECLARATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((root / DECLARATION_PATH).read_bytes())
    physical._equal(load_declaration(path), declaration, 'preserved feature declaration')


def operation_id(operation: dict) -> str:
    return ':'.join(str(operation[key]) for key in ('phase', 'profile', 'seed') if key in operation)


def planned_learning_ids() -> list[str]:
    return [f'{phase}:{profile}:{seed}' for seed in SEEDS
            for phase, profile in (('scratch', 'RAW'), ('scratch', 'FEATURE_UNITS'),
                                   ('adapted', 'RAW'), ('adapted', 'FEATURE_UNITS'))]


def policy_diagnostics(policy: Any, observation: Any) -> dict:
    """Observed actor conditioning on a permitted optimization state; no updates."""
    with torch.no_grad():
        actions = torch.as_tensor(observation.action_features.copy(), dtype=torch.float32)
        state = torch.as_tensor(observation.state_features.copy(), dtype=torch.float32)
        inputs = torch.cat((policy.actor_inputs(actions), state.expand(len(actions), -1)), dim=1)
        preactivation = policy.actor[0](inputs)
        activation = torch.tanh(preactivation)
        logits, value = policy(observation)
        probabilities = torch.softmax(logits, dim=0)
        positive = probabilities[probabilities > 0]
        return {'action_ids': list(observation.action_ids), 'action_probabilities': probabilities.tolist(),
            'entropy_nats': float(-(positive * positive.log()).sum()), 'critic_physical_value': float(value),
            'actor_preactivation_min': float(preactivation.min()), 'actor_preactivation_max': float(preactivation.max()),
            'actor_tanh_min': float(activation.min()), 'actor_tanh_max': float(activation.max()),
            'actor_tanh_fraction_abs_above_0_95': float((activation.abs() > .95).float().mean()),
            'diagnostic_scope': 'initial permitted optimization state; no transitions or gradients'}


def paired_initialization(base: Any, declaration: dict, world_seed: int) -> list[dict]:
    """Construction only: identical trainable tensors, unchanged critic inputs."""
    from resectionlab.learning import trainable_parameter_hash
    observation = base.clone().reset(world_seed)
    rows = []
    for seed in (101, *SEEDS):
        policies = {}
        for profile in PROFILES:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                policies[profile] = MaskedPatientPolicy(15, 6,
                    declaration['unchanged_learning']['hidden_features'], input_profile=profile)
            if policies[profile].input_profile.fingerprint != declaration['profiles'][profile]['profile_content_hash']:
                raise ValueError('Runtime input profile differs from the declared semantic fingerprint')
        raw, scaled = (policies[name] for name in PROFILES)
        with torch.no_grad():
            raw_logits, raw_value = raw(observation)
            scaled_logits, scaled_value = scaled(observation)
        if trainable_parameter_hash(raw) != trainable_parameter_hash(scaled):
            raise ValueError('Paired profiles do not start from identical trainable tensors')
        if not torch.equal(raw_value, scaled_value):
            raise ValueError('Feature scaling changed the initial critic prediction')
        rows.append({'seed': seed, 'trainable_parameter_hash': trainable_parameter_hash(raw),
            'initial_critic_value': float(raw_value),
            'input_profile_hashes': {name: policy.input_profile.fingerprint for name, policy in policies.items()},
            'policy_hashes': {name: policy_hash(policy) for name, policy in policies.items()},
            'initial_action_probabilities': {'RAW': torch.softmax(raw_logits, dim=0).tolist(),
                'FEATURE_UNITS': torch.softmax(scaled_logits, dim=0).tolist()},
            'diagnostics': {name: policy_diagnostics(policy, observation) for name, policy in policies.items()},
            'optimization_world_seed': world_seed, 'gradient_steps': 0})
    return rows


def preflight(base: Any, case: Any, target: Any, members: tuple,
        declaration: dict, reference: dict, output: Path, cancelled: Callable[[], bool]) -> tuple[Any, list[dict]]:
    from resectionlab.procedural_learning import validate_procedural_target_worlds
    physical.assert_declaration(reference)
    physical.assert_declared_sources(reference)
    physical._equal(reference['declaration_content_hash'],
        declaration['reference_design']['declaration_content_hash'], 'inherited physical declaration')
    panels = physical.assert_declared_target(case, base, reference)
    config = TrainingConfig(**declaration['budgets']['online_scratch_and_adapted_each_seed'])
    if config.max_episode_steps != base.config.max_steps + 1:
        raise ValueError('Policy and search action horizons differ')
    checks = []
    for profile in PROFILES:
        for seed in SEEDS:
            receipt = validate_procedural_target_worlds(target, base, panels.optimization,
                panels.selection, replace(config, seed=seed), input_profile=profile, study_id=STUDY_ID)
            checks.append({'input_profile': profile, 'seed': seed, 'receipt': receipt})
    paired = paired_initialization(base, declaration, panels.optimization.seeds[0])
    write_json(output / 'public-world-preflight.json', {'checks': checks,
        'paired_initialization': paired, 'gradient_steps': 0,
        'before_offline_pretraining': True, 'final_worlds_used': False})
    physical.procedural_preflight(members, reference, output, cancelled)
    return panels, paired


def cold_arm(base: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    arm = base.fresh()
    physical._equal(physical.model_inventory(arm), physical.model_inventory(base), 'cold arm inventory')
    return arm, time.perf_counter() - started


def learning_arm(base: Any, panels: Any, target: Any, config: TrainingConfig,
        operation: dict, output: Path, shared: dict, initial_hashes: dict,
        cancelled: Callable[[], bool]) -> tuple[dict, list[FrozenSequence]]:
    from resectionlab.learning import trainable_parameter_hash
    from resectionlab.procedural_learning import train_procedural_adapted_policy
    phase, profile, seed = (operation[key] for key in ('phase', 'profile', 'seed'))
    started = time.perf_counter()
    arm, preparation_seconds = cold_arm(base)
    folder = output / f'{profile.lower()}-{phase}-{seed}'
    settings = dict(config=replace(config, seed=seed), output_dir=folder,
        cancelled=cancelled, input_profile=profile)
    context_before = physical.execution_context()
    call_started = time.perf_counter()
    result = (train_patient_policy(arm.clone, panels.optimization, panels.selection, **settings)
        if phase == 'scratch' else train_procedural_adapted_policy(arm.clone,
            panels.optimization, panels.selection, checkpoint=shared[profile]['path'],
            target=target, study_id=STUDY_ID, **settings))
    call_seconds = time.perf_counter() - call_started
    detailed = json.loads((folder / 'result.json').read_text())
    contract = json.loads((folder / 'contract.json').read_text())
    record = {**asdict(result), 'phase': phase, 'input_profile': profile, 'seed': seed,
        'operation_id': operation_id(operation), 'preparation_seconds': preparation_seconds,
        'trainer_call_seconds': call_seconds, 'execution_context_before': context_before,
        'execution_context_after': physical.execution_context(), 'algorithm': contract['algorithm'],
        'learner_wall_budget_overshoot_seconds': max(0., result.elapsed_seconds - config.max_wall_seconds),
        'actor_parameters_changed': detailed['actor_parameters_changed'],
        'initial_actor_hash': detailed['initial_actor_hash'], 'latest_actor_hash': detailed['latest_actor_hash'],
        'selected_is_initial': result.selected_checkpoint_hash == result.initial_checkpoint_hash,
        'candidate_extraction_seconds': None, 'complete_arm_seconds_before_independent_audit': None,
        'input_profile_hash': detailed['input_profile_hash'],
        'selection_seconds': detailed['selection_seconds'],
        'post_budget_checkpoint_and_reporting_residual_seconds': max(0., call_seconds-result.elapsed_seconds-(result.initialization_seconds or 0.)),
        'post_budget_residual_scope': 'full trainer call minus its learner and initialization clocks; reporting/checkpoint overhead, not isolated export latency'}
    completed_panels = detailed.get('selection_history', [])
    panel_seconds = [row['panel_elapsed_seconds'] for row in completed_panels]
    initial_panel_seconds = panel_seconds[0] if panel_seconds else None
    record.update(initial_selection_seconds=initial_panel_seconds,
        later_and_partial_selection_seconds=(detailed['selection_seconds']-initial_panel_seconds
            if initial_panel_seconds is not None else None),
        incomplete_selection_seconds=max(0., detailed['selection_seconds']-sum(panel_seconds)),
        optimization_and_bookkeeping_seconds=max(0., result.elapsed_seconds-detailed['selection_seconds']),
        final_checkpoint_export_seconds=detailed['final_checkpoint_export_seconds'])
    # Preserve the raw attempt before checking whether it has an eligible selection.
    write_json(folder / 'arm-record.json', record)
    if contract['algorithm'] != 'masked_reinforce_state_value_v2':
        raise ValueError('Learning algorithm differs from declared feature ablation')
    if result.status in {'failed', 'cancelled'}:
        raise RuntimeError(f'{operation_id(operation)} did not finish: {result.status}')
    physical.require_complete_selection(result, detailed.get('selection_history', []), len(panels.selection.seeds))
    if phase == 'adapted' and result.initial_checkpoint_hash != shared[profile]['policy_hash']:
        raise ValueError('Adaptation did not start from its unchanged profile-specific shared policy')
    extraction_started = time.perf_counter()
    initial = load_policy(folder / 'initial.pt', expected_input_profile=profile)
    if policy_hash(initial) != result.initial_checkpoint_hash:
        raise ValueError('Saved initial policy disagrees with the training record')
    initial_trainable_hash = trainable_parameter_hash(initial)
    if phase == 'scratch' and initial_trainable_hash != initial_hashes[seed]:
        raise ValueError('Actual scratch initialization differs from paired zero-gradient preflight')
    record['initial_trainable_parameter_hash'] = initial_trainable_hash
    diagnostic_observation = arm.clone().reset(panels.optimization.seeds[0])
    record['initial_policy_diagnostics'] = policy_diagnostics(initial, diagnostic_observation)
    mode = 'PATIENT_SCRATCH_RL' if phase == 'scratch' else physical.ADAPTED
    checkpoints = [(f'{profile}:{mode}:{seed}', 'checkpoint.pt', result.selected_checkpoint_hash)]
    if phase == 'scratch':
        checkpoints.append((f'{profile}:INITIAL:{seed}', 'initial.pt', result.initial_checkpoint_hash))
    candidates = []
    for label, filename, expected in checkpoints:
        policy = load_policy(folder / filename, expected_input_profile=profile)
        if policy_hash(policy) != expected:
            raise ValueError('Saved policy differs from reported profile-specific checkpoint')
        if ':INITIAL:' not in label:
            record['selected_policy_diagnostics'] = policy_diagnostics(policy, diagnostic_observation)
        rollout = rollout_policy(policy, arm.clone(), seed=panels.selection.seeds[0],
            max_steps=config.max_episode_steps, expected_model_hash=base.decision_model_hash, interrupt=cancelled)
        candidates.append(FrozenSequence(label, base.case_hash, rollout.actions,
            'INITIAL_POLICY' if ':INITIAL:' in label else mode, expected,
            shared_checkpoint_hash=shared[profile]['policy_hash'] if phase == 'adapted' else None))
    record['candidate_extraction_seconds'] = time.perf_counter() - extraction_started
    record['complete_arm_seconds_before_independent_audit'] = time.perf_counter() - started
    write_json(folder / 'arm-record.json', record)
    return record, candidates


def run_feature_study(base: Any, case: Any, target: Any, members: tuple,
        output: Path, declaration: dict, reference: dict, *, execute: bool = False,
        cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Exact ordered study. Test fixtures require pytest-only validator substitution."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    status = {'run_id': f'feature-units-development-{uuid.uuid4()}', 'scope': SCOPE,
        'status': 'running', 'declaration_hash': DECLARATION_HASH, 'final_worlds_used': False,
        'training_requested': execute, 'planned_learning_runs': planned_learning_ids(),
        'completed_learning_runs': [], 'planned_offline_runs': list(PROFILES), 'completed_offline_runs': [],
        'planned_frozen_runs': list(PROFILES), 'completed_frozen_runs': [], 'expected_candidate_count': 23,
        'completed_operations': [], 'attempted_operations': []}
    write_json(output / 'status.json', status)
    try:
        assert_declaration(declaration)
        snapshot = study_source_snapshot()
        write_json(output / 'source.json', snapshot)
        preserve_source(snapshot, output / 'source-snapshot')
        preserve_inputs(declaration, reference, output / 'source-snapshot')
        panels, paired = preflight(base, case, target, members, declaration, reference, output, cancelled)
        assert_source_unchanged(snapshot)
        preflight_seconds = time.perf_counter() - started
        if not execute:
            status.update(status='preflight_passed_no_training', training_executed=False,
                preflight_seconds=preflight_seconds, total_seconds=time.perf_counter() - started)
            write_json(output / 'status.json', status)
            return status
        from resectionlab.procedural_learning import (train_procedural_native_policy,
            validate_procedural_checkpoint, load_frozen_procedural_policy)
        config = TrainingConfig(**declaration['budgets']['online_scratch_and_adapted_each_seed'])
        offline_config = TrainingConfig(**declaration['budgets']['offline_training'])
        initial_hashes = {row['seed']: row['trainable_parameter_hash'] for row in paired}
        model = FrozenDecisionModel.create(case_hash=base.case_hash,
            geometry={'simulation_hash': base.decision_model_hash},
            objectives={**asdict(base.config.reward), 'partial_contact_weight': base.partial_contact_weight},
            tools=[asdict(tool) for tool in base.native_config.tools],
            world_generator=base.config.world_generator.to_dict(),
            action_primitives={'native_adapter': physical.NATIVE_ADAPTER_VERSION,
                'inventory': physical.model_inventory(base)})
        write_json(output / 'manifest.json', {'declaration': declaration, 'reference_hash': physical.DECLARATION_HASH,
            'target': asdict(target), 'world_partitions': panels.to_dict(), 'decision_model': model.to_dict(),
            'source_hash': snapshot['numerical_runtime_content_hash'], 'final_worlds_used': False})
        pretrained, shared, frozen_records, learned, searches = {}, {}, {}, [], []
        frozen_policies = {}
        candidates = [FrozenSequence('STOP', base.case_hash, ('STOP',), 'STOP')]
        for operation in declaration['execution_order']:
            assert_source_unchanged(snapshot)
            if cancelled():
                raise InterruptedError('Feature-unit study cancelled')
            op_id = operation_id(operation)
            status['current_operation'] = op_id
            status['attempted_operations'].append(op_id)
            write_json(output / 'status.json', status)
            phase, profile = operation['phase'], operation.get('profile')
            phase_started = time.perf_counter()
            if phase == 'offline_pretraining':
                context_before = physical.execution_context()
                result = train_procedural_native_policy(members, excluded_targets=(target,),
                    config=replace(offline_config, seed=operation['seed']),
                    output_dir=output / f'pretraining-{profile.lower()}', cancelled=cancelled,
                    input_profile=profile, study_id=STUDY_ID)
                pretrained[profile] = {**result, 'full_pretraining_call_seconds': time.perf_counter() - phase_started,
                    'execution_context_before': context_before, 'execution_context_after': physical.execution_context()}
                write_json(output / 'pretraining.json', pretrained)
                if result.get('status') != 'completed' or not result.get('checkpoint_path'):
                    raise RuntimeError(f'{profile} pretraining did not export a qualified checkpoint')
                initial = load_policy(output / f'pretraining-{profile.lower()}/training/initial.pt', expected_input_profile=profile)
                from resectionlab.learning import trainable_parameter_hash
                if trainable_parameter_hash(initial) != initial_hashes[operation['seed']]:
                    raise ValueError('Offline initialization differs from paired preflight')
                status['completed_offline_runs'].append(profile)
            elif phase == 'geometry_only_baselines':
                for method in ('GREEDY', 'SEARCH'):
                    method_started = time.perf_counter()
                    arm, preparation_seconds = cold_arm(base)
                    before = physical.execution_context()
                    options = declaration['budgets']['search']
                    result = (physical.native_greedy_search(arm, max_wall_seconds=options['max_wall_seconds'])
                        if method == 'GREEDY' else physical.native_beam_search(arm, **options))
                    searches.append({'method': method, **asdict(result), 'shared_baseline_id': method,
                        'reused_by_profiles': list(PROFILES), 'preparation_seconds': preparation_seconds,
                        'complete_arm_seconds_before_independent_audit': time.perf_counter() - method_started,
                        'execution_context_before': before, 'execution_context_after': physical.execution_context()})
                    candidates.append(FrozenSequence(method, base.case_hash, result.actions, method))
                write_json(output / 'search.json', searches)
            elif phase == 'frozen_selection':
                validation = dict(target=target, simulator=base, hidden_features=config.hidden_features,
                    input_profile=profile, study_id=STUDY_ID)
                initialization_started = time.perf_counter()
                checkpoint = Path(pretrained[profile]['checkpoint_path'])
                details = validate_procedural_checkpoint(checkpoint, **validation)
                saved = output / f'{profile.lower()}-procedural-source.pt'
                saved.write_bytes(checkpoint.read_bytes())
                physical._equal(validate_procedural_checkpoint(saved, **validation), details, 'copied profile checkpoint')
                policy = load_frozen_procedural_policy(saved, **validation)
                frozen_policies[profile] = policy
                shared[profile] = {**details, 'path': str(saved),
                    'initialization_seconds': time.perf_counter() - initialization_started}
                write_json(output / 'shared-provenance.json', shared)
                arm, preparation_seconds = cold_arm(base)
                before = physical.execution_context()
                frozen_config = replace(config,
                    max_wall_seconds=declaration['budgets']['frozen']['max_wall_seconds_for_complete_selection_panel'])
                record, actions = frozen_population_rollouts(arm.clone, policy, panels.selection,
                    frozen_config, cancelled=cancelled)
                record.update(input_profile=profile, preparation_seconds=preparation_seconds,
                    shared_checkpoint_initialization_seconds=shared[profile]['initialization_seconds'],
                    complete_arm_seconds_before_independent_audit=time.perf_counter() - phase_started,
                    execution_context_before=before, execution_context_after=physical.execution_context(),
                    elapsed_seconds_scope='selection panel only; cold setup and shared initialization separately measured')
                record['policy_diagnostics'] = policy_diagnostics(policy, arm.clone().reset(panels.optimization.seeds[0]))
                record['complete_arm_seconds_before_independent_audit'] = time.perf_counter() - phase_started
                frozen_records[profile] = record
                write_json(output / 'frozen.json', frozen_records)
                if not record['selection_panel_complete']:
                    raise RuntimeError(f'{profile} frozen checkpoint has no complete selection panel')
                candidates.append(FrozenSequence(f'{profile}:{physical.FROZEN}', base.case_hash, actions,
                    physical.FROZEN, details['policy_hash'], shared_checkpoint_hash=details['policy_hash']))
                status['completed_frozen_runs'].append(profile)
            elif phase in ('scratch', 'adapted'):
                try:
                    record, new_candidates = learning_arm(base, panels, target, config, operation,
                        output, shared, initial_hashes, cancelled)
                except Exception:
                    folder = output / f'{profile.lower()}-{phase}-{operation["seed"]}'
                    receipt = folder / 'arm-record.json'
                    learned.append(json.loads(receipt.read_text()) if receipt.exists() else {
                        'operation_id': op_id, 'phase': phase, 'input_profile': profile,
                        'seed': operation['seed'], 'status': 'failed_before_complete_result', 'output_dir': str(folder)})
                    write_json(output / 'training.json', learned)
                    raise
                learned.append(record)
                write_json(output / 'training.json', learned)
                candidates.extend(new_candidates)
                status['completed_learning_runs'].append(op_id)
            else:
                raise ValueError(f'Unknown declared phase: {phase}')
            assert_source_unchanged(snapshot)
            status['completed_operations'].append(op_id)
            write_json(output / 'status.json', status)
        for profile in PROFILES:
            if (policy_hash(frozen_policies[profile]) != shared[profile]['policy_hash']
                    or hashlib.sha256(Path(shared[profile]['path']).read_bytes()).hexdigest() != shared[profile]['checkpoint_file_sha256']):
                raise ValueError('A shared profile checkpoint changed during adaptation')
        if len(candidates) != declaration['expected_denominators']['retained_candidates_including_baselines_and_initials']:
            raise ValueError('Candidate denominator differs from preregistration')
        freeze = freeze_candidates(candidates, model, panels.selection,
            'declared shared baselines; profile-specific initial, frozen and selection-best actors; earliest tie',
            optimization_manifest=panels.optimization)
        write_json(output / 'candidate-freeze.json', {**freeze.to_dict(),
            'candidates': [asdict(candidate) for candidate in candidates], 'input_profiles': declaration['profiles'],
            'candidate_profiles': {candidate.plan_id: {
                'profile_id': candidate.plan_id.split(':')[0],
                'profile_content_hash': declaration['profiles'][candidate.plan_id.split(':')[0]]['profile_content_hash'],
                'full_policy_hash': candidate.selected_checkpoint_hash}
                for candidate in candidates if candidate.plan_id.split(':')[0] in PROFILES}})
        audit_dir = output / f'development-geometry-{uuid.uuid4()}'
        audit_dir.mkdir()
        write_json(audit_dir / 'manifest.json', {'candidate_freeze_hash': freeze.fingerprint,
            'final_worlds_used': False, 'development_run_id': status['run_id']})
        audit = physical.validate_frozen_candidates(base, case, candidates,
            panels.selection.seeds[0], audit_dir, cancelled)
        assert_source_unchanged(snapshot)
        status.update(status='invalidated' if audit['rejected_candidate_ids'] else 'completed',
            validation=audit, geometry_validation_run_id=audit_dir.name, training_executed=True,
            preflight_seconds=preflight_seconds, total_seconds=time.perf_counter() - started)
        write_json(output / 'summary.json', {**status, 'pretraining': pretrained, 'search': searches,
            'frozen': frozen_records, 'learning': learned, 'shared': shared,
            'source_hash': snapshot['numerical_runtime_content_hash'], 'clinical_deficit_probability': None})
        write_json(output / 'status.json', status)
        return status
    except Exception as exc:
        status.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed',
            exception=type(exc).__name__, reason=str(exc), total_seconds=time.perf_counter() - started,
            unfinished_learning_runs=[x for x in status['planned_learning_runs'] if x not in status['completed_learning_runs']],
            unfinished_offline_runs=[x for x in PROFILES if x not in status['completed_offline_runs']],
            unfinished_frozen_runs=[x for x in PROFILES if x not in status['completed_frozen_runs']])
        write_json(output / 'status.json', status)
        raise


def worker(output: Path, bundle: Path, *, execute: bool, cancelled: Callable[[], bool]) -> dict:
    from resectionlab.imaging import load_case
    from resectionlab.procedural_learning import TransferTarget, make_native_procedural_fixture
    started = time.perf_counter()
    declaration = load_declaration(ROOT / DECLARATION_PATH)
    reference = physical.load_declaration(ROOT / declaration['reference_design']['path'])
    physical.assert_declared_sources(reference)
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != reference['target']['bundle_sha256']:
        raise ValueError('Source bundle differs from declared patient bytes')
    case = load_case(bundle)
    base = physical.make_native_patient_simulator(case, candidate_count=4, max_steps=3,
        max_actions=7, cancelled=cancelled)
    identity = reference['target']
    target = TransferTarget(case.semantic_hash, case.planning_hash, identity['group_id'],
        tuple(identity['aliases']), identity['source_kind'])
    members = tuple(make_native_procedural_fixture(target, input_profile='RAW', study_id=STUDY_ID))
    write_json(output / 'worker-preparation.json', {'case_and_fixture_seconds': time.perf_counter() - started,
        'source': study_source_snapshot(), 'human_pretraining_patients': 0,
        'patient_derived_development_cases': 1, 'final_worlds_used': False})
    return run_feature_study(base, case, target, members, output / 'study', declaration,
        reference, execute=execute, cancelled=cancelled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-bundle', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--frozen-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    output = args.output.resolve()
    if args.frozen_worker:
        worker(output, args.case_bundle.resolve(), execute=args.execute, cancelled=stopped.is_set)
        return
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {'status': 'running', 'study_id': STUDY_ID, 'training_requested': args.execute,
        'final_worlds_used': False, 'declaration_hash': DECLARATION_HASH}
    write_json(output / 'experiment-status.json', status)
    try:
        declaration = load_declaration(ROOT / DECLARATION_PATH)
        reference = physical.load_declaration(ROOT / declaration['reference_design']['path'])
        physical.assert_declared_sources(reference)
        snapshot = study_source_snapshot()
        write_json(output / 'launch-source.json', snapshot)
        frozen = output / 'frozen-source'
        preserve_source(snapshot, frozen)
        preserve_inputs(declaration, reference, frozen)
        source = args.case_bundle or ROOT / reference['target']['bundle_path']
        saved = output / 'source-case.ressectionlab'
        saved.write_bytes(source.read_bytes())
        if hashlib.sha256(saved.read_bytes()).hexdigest() != reference['target']['bundle_sha256']:
            raise ValueError('Preserved source bundle differs from declaration')
        assert_source_unchanged(snapshot)
        command = [sys.executable, str(frozen / 'scripts/run_native_feature_units.py'),
            '--output', str(output), '--case-bundle', str(saved), '--frozen-worker']
        if args.execute:
            command.append('--execute')
        process = subprocess.Popen(command, cwd=frozen)
        interruption_sent = False
        while process.poll() is None:
            if stopped.is_set() and not interruption_sent:
                process.terminate()
                interruption_sent = True
            time.sleep(.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        result = json.loads((output / 'study/status.json').read_text())
        status.update(status=result['status'], total_seconds_including_source_freeze=time.perf_counter() - started)
    except Exception as exc:
        status.update(status='cancelled' if stopped.is_set() else 'failed', exception=type(exc).__name__,
            reason=str(exc), total_seconds_including_source_freeze=time.perf_counter() - started)
        write_json(output / 'experiment-status.json', status)
        raise
    write_json(output / 'experiment-status.json', status)
    print(f"{status['status']}: {output / 'study/status.json'}")


if __name__ == '__main__':
    main()
