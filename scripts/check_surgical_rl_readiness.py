#!/usr/bin/env python3
"""Bounded scripted readiness checks. No training, policy selection or optimizer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_cusa_v3 import default_profile as cusa_default_profile
from resectionlab.surgical_observed_policy import candidates
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_env import SurgicalRLEnv
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_skill_options import build_skill_options
from resectionlab.surgical_skills import SkillExecutor, SkillTask

SCHEMA = 'resectionlab.surgical-readiness-check.v1'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Budget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.deadline = self.started + seconds

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('Readiness wall-time budget exhausted')

    def remaining(self):
        self.check()
        return self.deadline - time.perf_counter()

    def timed(self, function):
        self.check()
        start = time.perf_counter()
        result = function()
        self.check()
        return result, time.perf_counter() - start


def summarize(samples):
    return {'repeats': len(samples), 'median_seconds': float(np.median(samples)),
            'p95_seconds': float(np.quantile(samples, .95)),
            'min_seconds': min(samples), 'max_seconds': max(samples)}


def status_counts(statuses):
    counts = Counter(statuses)
    require(not set(counts) - {'accepted', 'partial', 'rejected'}, 'Unknown primitive outcome status')
    return {name: counts[name] for name in ('accepted', 'partial', 'rejected')}


def repeat(budget, function, count=10):
    times = [budget.timed(function)[1] for _ in range(count)]
    return summarize(times)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path, budget):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            budget.check()
            digest.update(block)
    return digest.hexdigest()


def pose(z=-5., *, x=9.):
    return {'position': [x, 9., z], 'quaternion': [0., 0., 0., 1.]}


def make_surrogate(*, rescue=True, causal_cusa=True):
    """Declared generated coupon, not a patient or a vascular anatomy estimate."""
    labels = np.zeros((24, 24, 20), np.uint16)
    labels[8:11, 8:11, 8:11] = 1
    properties = {'tissuePhysicsV2': default_profile()}
    if causal_cusa:
        properties['cusaV3'] = cusa_default_profile()
    if rescue:
        properties['bleedingSources'] = [{
            'id': 'declared-preinjured-fixture',
            'cell': int(np.ravel_multi_index((9, 9, 9), labels.shape)),
            'rateMm3PerSecond': .1, 'coagulationExposure': .02, 'patent': False}]
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'Synthetic target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated skill fixture'}, properties)
    task = SkillTask('resection', 'preinjured_rescue' if rescue else 'prevention')
    env = SurgicalRLEnv(scene, config=task.config(
        require_critical_coverage=False, max_steps=64, max_seconds=60.))
    return env, task


def script_requests():
    # Explicit fixture endpoints. No route is computed from hidden source cells.
    return [
        ('inspect', {}),
        ('approach', {'instrument': 'bipolar_forceps', 'pose': pose(),
                      'parameters': {'radiusMm': 1.}}),
        ('coagulate', {'pose': pose(8.5), 'parameters': {'power': 2.}, 'duration': .2}),
        ('withdraw', {}),
        ('approach', {'instrument': 'cusa', 'pose': pose(5.9), 'parameters': {'radiusMm': 1.5}}),
        *[('resect', {'pose': pose(11.), 'duration': seconds}) for seconds in (4.9, 5., 5.)],
        ('withdraw', {}),
        ('approach', {'instrument': 'frazier_suction', 'pose': pose(),
                      'parameters': {'radiusMm': 1.5}}),
        ('aspirate', {'pose': pose(9.), 'parameters': {'flowMm3PerSecond': 100.}, 'duration': .2}),
        ('stop', {}),
    ]


def declared_skill_requests(requests):
    """Declare this script's routes and both hands' controls before execution."""
    options = [{'id': f'request-{index:02d}', 'skill': name,
                'slot': 'left', **values} for index, (name, values) in enumerate(requests)]
    for skill in ('release', 'withdraw'):
        for slot in ('left', 'right'):
            if not any(row['skill'] == skill and row.get('slot') == slot for row in options):
                options.append({'id': f'control-{skill}-{slot}', 'skill': skill, 'slot': slot})
    if not any(row['skill'] == 'stop' for row in options):
        options.append({'id': 'control-stop', 'skill': 'stop'})
    return options


def execute_requests(env, task, requests, budget, *, capture_prefix=False, skill_options=False,
                     declared_options=None, request_offset=0):
    start = time.perf_counter()
    executor = SkillExecutor(env, task)
    option_requests = (declared_options if declared_options is not None else declared_skill_requests(requests)) if skill_options else None
    require(type(request_offset) is int and request_offset >= 0, 'Request offset must be a nonnegative integer')
    setup_seconds = time.perf_counter() - start
    rows = []
    decision_times = []
    resection_progression = []
    prefix = None
    prefix_native = None
    resume_request_index = None
    prefix_pending = None
    for index, (name, values) in enumerate(requests):
        decision_started = time.perf_counter()
        if skill_options:
            # Each decision starts with a new observation and constructs ALL
            # declared options. Selection is this fixed script's ID lookup;
            # there is no actor, ranking, feasibility oracle or hidden search.
            observation, observation_seconds = budget.timed(env._observe)
            options, candidate_seconds = budget.timed(lambda: build_skill_options(
                observation, option_requests, task=task,
                max_action_seconds=env.config.max_action_seconds))
            selected = next(row for row in options['options'] if row['id'] == f'request-{index + request_offset:02d}')
            require(selected['status'] == 'nominal_proposal',
                    'Prescribed request was nominally refused: ' + str(selected['reason']))
        receipt, seconds = budget.timed(lambda: executor.execute(
            name, check=budget.check, max_wall_seconds=budget.remaining(), **values))
        if skill_options:
            require(selected['plan']['actions'] == receipt['plan']['actions']
                    and selected['plan']['observationId'] == receipt['plan']['observationId'],
                    'Fresh declared proposal differs from public executor plan')
            decision_times.append({'request_id': selected['id'],
                'request_set_id': options['requestSetId'], 'observation_id': options['observationId'],
                'candidate_count': len(options['options']),
                'nominal_count': sum(row['status'] == 'nominal_proposal' for row in options['options']),
                'observation_seconds': observation_seconds, 'candidate_seconds': candidate_seconds,
                'execution_seconds': seconds, 'whole_decision_seconds': time.perf_counter() - decision_started})
        work_partial = (name == 'resect' and receipt['status'] == 'partial'
                        and receipt['transitions'][-1]['reasonCode'] == 'TISSUE_ASPIRATION_IN_PROGRESS')
        require(receipt['status'] in ('completed', 'episode_finished') or work_partial,
                'Scripted skill did not complete: ' + name + ' / ' + receipt['status'])
        rows.append({'skill': name, 'wall_seconds': seconds, 'receipt': receipt})
        if name == 'resect':
            resection_progression.append(env.measurements['tumor_removed_mm3'])
        if capture_prefix and name == 'resect' and prefix is None:
            require(work_partial, 'First causal CUSA request must retain genuine partial work')
            prefix_pending = sum(env.engine.state['cusaV3']['progressMm3'].values())
            require(prefix_pending > 0 and env.engine.state['cusaV3']['progressExact'], 'Snapshot prefix lacks nonzero exact pending CUSA work')
            prefix = json.loads(json.dumps(env.snapshot(), allow_nan=False))
            prefix_native = json.loads(json.dumps(env.engine.snapshot(), allow_nan=False))
            resume_request_index = index + 1
    elapsed = time.perf_counter() - start
    statuses = status_counts(transition['status'] for row in rows
                             for transition in row['receipt']['transitions'])
    result = {'wall_seconds': elapsed, 'executor_and_declarations_setup_seconds': setup_seconds,
            'skill_decisions': len(rows), 'skills': rows,
            'resection_stroke_progression_mm3': resection_progression,
            'work_progress_partial_primitives': sum(transition['reasonCode'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
                for row in rows for transition in row['receipt']['transitions']),
            'resume_request_index': resume_request_index, 'prefix_pending_capacity_mm3': prefix_pending,
            'primitive_status_counts': statuses,
            'primitive_attempts': sum(statuses.values())}
    if skill_options:
        result['skill_options'] = {
            'schema': 'resectionlab.surgical-skill-options.v1',
            'candidate_calls': len(decision_times), 'policy_executions': 0, 'actor_inference': 0,
            'selection': 'predeclared script request ID; no scoring or ranking',
            'decisions': decision_times,
            'observation': summarize([row['observation_seconds'] for row in decision_times]),
            'candidate_preparation': summarize([row['candidate_seconds'] for row in decision_times]),
            'execution': summarize([row['execution_seconds'] for row in decision_times]),
            'whole_decision': summarize([row['whole_decision_seconds'] for row in decision_times]),
            'decisions_per_episode_second': len(rows) / elapsed,
            'scope': 'Short generated prescribed script; all fresh observations/options/public execution included. '
                     'Episode time also includes executor/declaration setup, bookkeeping and optional prefix snapshot capture. '
                     'Not sustained patient or actor throughput; p95 uses a small sample.'}
    return result, prefix, prefix_native


def conservation(env, source_hash):
    state = env.engine.state
    present = sum(state['fluid'].values()) + sum(s.get('reservoirMm3', 0.) for s in state['sources'])
    fluid_residual = state['fluidAddedMm3'] - present - state['fluidRemovedMm3'] - state['fluidEscapedMm3']
    debris_generated = state.get('drillingV2', {}).get('removedVolumeMm3', 0.)
    debris_residual = debris_generated - state['debrisMm3'] - state.get('debrisRemovedMm3', 0.) - state.get('debrisEscapedMm3', 0.)
    cusa = state['cusaV3']
    pending = sum(cusa['progressMm3'].values())
    cusa_residual = cusa['inputCapacityMm3'] - cusa['aspiratedSolidMm3'] - pending - cusa['unusedCapacityMm3'] - cusa['discardedCapacityMm3']
    require(abs(fluid_residual) <= 1e-9, 'Fluid conservation differs')
    require(abs(debris_residual) <= 1e-9, 'Debris conservation differs')
    require(abs(cusa_residual) <= 1e-9, 'CUSA input/aspirated/pending/unused/discarded capacity differs')
    require(cusa['aspiratedSolidMm3'] == 27., 'Causal CUSA did not aspirate the entire unchanged target')
    require(cusa['inputCapacityMm3'] <= 2. * cusa['effectiveActivationSeconds'] + 1e-9, 'CUSA exceeded declared time capacity')
    require(array_digest(env.scene.labels) == source_hash, 'Generated source labels changed')
    require(len(state['removed']) == len(set(state['removed'])), 'Duplicate removed source identities')
    return {'fluid_residual_mm3': fluid_residual, 'debris_residual_mm3': debris_residual,
            'fluid_generated_mm3': state['fluidAddedMm3'], 'fluid_aspirated_mm3': state['fluidRemovedMm3'],
            'fluid_present_mm3': present, 'fluid_escaped_mm3': state['fluidEscapedMm3'],
            'debris_generated_mm3': debris_generated, 'debris_generation_exercised': debris_generated > 0,
            'cusa_capacity_residual_mm3': cusa_residual, 'cusa_capacity_ledger': cusa,
            'source_labels_unchanged': True,
            'unique_removed_source_cells': len(state['removed'])}


def run_seed(seed, output, budget, *, skill_options=False):
    (env, task), construction_seconds = budget.timed(make_surrogate)
    source_hash = array_digest(env.scene.labels)
    (observation, _), reset_seconds = budget.timed(lambda: env.reset(seed=seed))
    first = {key: array_digest(value) for key, value in observation.items()}
    (observation, _), repeat_reset_seconds = budget.timed(lambda: env.reset(seed=seed))
    require(first == {key: array_digest(value) for key, value in observation.items()}, 'Seeded reset differs')
    require(env.observation_space.contains(observation), 'Reset observation outside declared space')
    observation_times = repeat(budget, env._observe)
    declarations = declared_skill_requests(script_requests()) if skill_options else None
    if skill_options:
        candidate_times = repeat(budget, lambda: build_skill_options(
            observation, declarations, task=task, max_action_seconds=env.config.max_action_seconds))
    else:
        candidate_times = repeat(budget, lambda: candidates(observation))
        features, options = candidates(observation)
        require(len(options) == len(features) == 9, 'Candidate observation interface differs')
    require(not env.attempts, 'Read-only candidate preparation executed an action')
    try:
        SkillExecutor(env, SkillTask('resection', 'prevention'))
        raise AssertionError('Preinjured source accepted as prevention')
    except ValueError as error:
        require('preinjured_rescue' in str(error), 'Prevention failed for an unrelated reason')
    episode, prefix, prefix_native = execute_requests(env, task, script_requests(), budget,
        capture_prefix=True, skill_options=skill_options, declared_options=declarations)
    require(len(episode['resection_stroke_progression_mm3']) == 3
            and 0 < episode['resection_stroke_progression_mm3'][0] < 27.
            and episode['resection_stroke_progression_mm3'][-1] == 27.,
            'Causal declared-tip script did not complete the unchanged target over fixed repeated strokes')
    require(env.terminated and not env.truncated and env._info()['success'], 'Generated task did not succeed')
    require(env._cleanup_state()['pending'] == [], 'Generated success left cleanup pending')
    require(env.measurements['tumor_removed_mm3'] == 27., 'Generated target volume differs')
    require(env.engine.metrics()['collisions'] == 0, 'Causal script recorded a genuine collision')
    checks = conservation(env, source_hash)
    native_record = json.loads(json.dumps(env.engine.snapshot(), allow_nan=False))
    native, native_prefix_seconds = budget.timed(lambda: SurgicalEngine.restore(env.scene, prefix_native))
    require(native.state['cusaV3'] == prefix_native['checkpoint']['state']['cusaV3'], 'Native JSON prefix pending work differs')
    native_tail_attempts = env.attempts[len(prefix['attempts']):]
    def resume_native():
        for item in native_tail_attempts:
            outcome = native.step(item['action'], check=budget.check)
            require(outcome['status'] == item['status'] and outcome.get('reason') == item['reason']
                    and native.state_id == item['stateId'], 'Native sealed tail differs')
    _, native_tail_seconds = budget.timed(resume_native)
    native_seconds = native_prefix_seconds + native_tail_seconds
    require(native.state_id == env.engine.state_id and native.state == env.engine.state, 'Native endpoint replay differs')
    resumed, resumed_task = make_surrogate()
    _, restore_seconds = budget.timed(lambda: resumed.restore(prefix))
    require(resumed.engine.state['cusaV3'] == prefix_native['checkpoint']['state']['cusaV3'], 'RL JSON prefix pending work differs')
    resumed_episode, _, _ = execute_requests(resumed, resumed_task,
        script_requests()[episode['resume_request_index']:], budget, skill_options=skill_options,
        declared_options=declarations, request_offset=episode['resume_request_index'])
    final = json.loads(json.dumps(env.snapshot(), allow_nan=False))
    require(resumed.snapshot() == final, 'RL JSON prefix resume differs from uninterrupted episode')
    write_json(output / f'seed-{seed}-native-snapshot.json', native_record)
    write_json(output / f'seed-{seed}-native-prefix.json', prefix_native)
    write_json(output / f'seed-{seed}-rl-prefix.json', prefix)
    write_json(output / f'seed-{seed}-rl-final.json', final)
    return {'seed': seed, 'source_kind': 'generated coupon, not patient evidence',
            'scene_id': env.scene.identity, 'environment_id': env.environment_id,
            'action_contract': env.action_contract,
            'cusa_profile': env.scene.properties['cusaV3'],
            'construction_seconds': construction_seconds, 'reset_seconds': reset_seconds,
            'same_seed_reset_seconds': repeat_reset_seconds, 'seeded_reset_equal': True,
            'rescue_declaration_enforced': True,
            'observation': observation_times, 'candidate_preparation_only': candidate_times,
            'candidate_interface': 'declared nominal skills' if skill_options else 'legacy nine raw left-slot choices',
            'candidate_policy_decisions': 0, 'episode': episode,
            'reset_plus_script_seconds': reset_seconds + episode['wall_seconds'],
            'native_replay_seconds': native_seconds, 'native_replayed_primitives': len(native.actions),
            'native_prefix_restore_seconds': native_prefix_seconds, 'native_tail_seconds': native_tail_seconds,
            'nonzero_pending_prefix_native_and_rl_equal': True,
            'native_replay_status_counts': status_counts(item['status'] for item in native.history),
            'rl_prefix_restore_seconds': restore_seconds, 'rl_prefix_replayed_primitives': len(prefix['attempts']),
            'rl_prefix_replay_status_counts': status_counts(item['status'] for item in prefix['attempts']),
            'resumed_episode': resumed_episode, 'snapshot_resume_equal': True,
            'endpoint_state_id': native.state_id, 'metrics': env.measurements,
            'engine_collision_count': env.engine.metrics()['collisions'],
            'cleanup': env._cleanup_state(), 'conservation': checks}


def run_control_probes(budget):
    env, task = make_surrogate(rescue=False)
    env.reset(seed=17)
    executor = SkillExecutor(env, task)
    before, attempts, actions = env.engine.state_id, len(env.attempts), len(env.engine.actions)
    calls = 0
    class InjectedCancellation(Exception):
        pass
    def cancel():
        nonlocal calls
        budget.check()
        calls += 1
        if calls >= 12:
            raise InjectedCancellation('Intentional interruption inside checked movement')
    try:
        executor.execute('approach', pose=pose(7.), check=cancel, max_wall_seconds=budget.remaining())
        raise AssertionError('Interrupted action unexpectedly committed')
    except InjectedCancellation:
        pass
    require(env.engine.state_id == before and len(env.attempts) == attempts and len(env.engine.actions) == actions,
            'Interrupted current primitive committed state or history')
    partial, partial_seconds = budget.timed(lambda: executor.execute(
        'approach', pose=pose(9.), check=budget.check, max_wall_seconds=budget.remaining()))
    require(partial['status'] in ('partial', 'rejected'), 'Intact tissue approach unexpectedly completed')
    _, rejected_seconds = budget.timed(lambda: env.step_action(
        {'slot': 'left', 'operation': 'inspect', 'duration': 6.}, check=budget.check))
    require(env.attempts[-1]['status'] == 'rejected', 'Excess duration not rejected')
    return {'interrupted_primitive_no_commit': True, 'interruption_callback_calls': calls,
            'cancelled_primitives_without_committed_outcome': 1,
            'partial_probe_seconds': partial_seconds, 'partial_probe_receipt': partial,
            'rejected_probe_seconds': rejected_seconds,
            'primitive_status_counts': status_counts(item['status'] for item in env.attempts),
            'skill_decisions': 1, 'raw_validation_probe_decisions': 1}


def run_tip_probes(budget):
    # Geometry-only variants remain a separate legacy control, with no CUSA
    # removal actions. The causal profile deliberately fixes CUSA at 1.5 mm.
    env, _ = make_surrogate(rescue=False, causal_cusa=False)
    env.reset(seed=17)
    requests = [
        ({'operation': 'exchange', 'instrument': 'cusa'}, None),
        ({'operation': 'inspect', 'parameters': {'radiusMm': 12.}}, 'INSTRUMENT_RADIUS_EXCEEDS_CATALOG'),
        ({'operation': 'inspect', 'parameters': {'radiusMm': .5}}, None),
        ({'operation': 'move', 'pose': pose()}, None),
        ({'operation': 'inspect', 'parameters': {'radiusMm': 1.}}, 'WITHDRAW_BEFORE_RADIUS_CHANGE'),
        ({'operation': 'inspect', 'parameters': {'radiusMm': .25}}, 'WITHDRAW_BEFORE_RADIUS_CHANGE'),
        ({'operation': 'inspect', 'parameters': {'radiusMm': .5}}, None),
        ({'operation': 'withdraw'}, None),
        ({'operation': 'inspect', 'parameters': {'radiusMm': 1.5}}, None),
    ]
    rows = []
    for values, rejection in requests:
        before = env.engine.state_id
        action = {'slot': 'left', 'duration': .1, **values}
        result, seconds = budget.timed(lambda: env.step_action(action, check=budget.check))
        outcome = result[4]['outcome']
        require(outcome['status'] == ('rejected' if rejection else 'accepted'), 'Tip contract probe outcome differs')
        if rejection:
            require(outcome['reasonCode'] == rejection and env.engine.state_id == before, 'Rejected tip change altered physical state')
        rows.append({'action': action, 'status': outcome['status'], 'reasonCode': outcome.get('reasonCode'), 'wall_seconds': seconds})
    return {'catalog_cap_enforced': True, 'inserted_growth_and_shrink_blocked': True,
            'scope': 'legacy geometry-only variant controls; no material removal; causal CUSA profile uses fixed 1.5 mm tip',
            'same_radius_respec_allowed': True, 'withdrawn_variant_selection_allowed': True,
            'primitive_status_counts': status_counts(row['status'] for row in rows),
            'raw_probe_decisions': len(rows), 'probes': rows}


def patient_reset_only(path, protection, budget):
    sources = [path] + ([protection, protection.with_suffix('.npz')] if protection else [])
    hashes = {str(p): file_hash(p, budget) for p in sources}
    env, load_seconds = budget.timed(lambda: SurgicalRLEnv.from_patient(path, protection_path=protection))
    (observation, _), reset_seconds = budget.timed(lambda: env.reset(seed=17))
    labels_hash = array_digest(env.scene.labels)
    observation_times = repeat(budget, env._observe)
    require(env.observation_space.contains(observation), 'Patient observation outside declared space')
    require(not env.attempts and not env.engine.actions, 'Patient reset benchmark executed physics actions')
    require(array_digest(env.scene.labels) == labels_hash, 'Patient source labels changed')
    require(hashes == {str(p): file_hash(p, budget) for p in sources}, 'Patient/protection source bytes changed')
    return {'source': str(path), 'protection': str(protection) if protection else None,
            'source_hashes_sha256': hashes, 'source_bytes_unchanged': True,
            'scene_id': env.scene.identity, 'grid_shape': list(env.scene.shape),
            'cusa_profile': env.scene.properties.get('cusaV3'),
            'load_and_construct_seconds': load_seconds, 'reset_seconds': reset_seconds,
            'observation': observation_times, 'physics_actions': 0, 'skill_decisions': 0,
            'primitive_status_counts': status_counts(()),
            'policy_decisions': 0, 'scope': 'load/reset/observation only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory below this checkout testcases/outputs/')
    parser.add_argument('--max-wall-seconds', type=float, default=30., help='Whole-check deadline in (0, 60] seconds')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43], help='One to three deterministic smoke seeds')
    parser.add_argument('--skill-options', action='store_true',
                        help='Build all declared nominal skill options before each prescribed decision; no policy')
    parser.add_argument('--patient', type=Path, help='Optional existing .ressectionlab; reset/observe only')
    parser.add_argument('--protection', type=Path, help='Optional matching protection JSON for --patient')
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 60:
        parser.error('--max-wall-seconds must be in (0, 60]')
    if not 1 <= len(args.seeds) <= 3 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error('--seeds requires one to three distinct uint32 seeds')
    if args.protection and not args.patient:
        parser.error('--protection requires --patient')
    if args.patient and not args.patient.is_file():
        parser.error('--patient must name an existing workspace file')
    if args.protection and (not args.protection.is_file() or not args.protection.with_suffix('.npz').is_file()):
        parser.error('--protection requires its matching existing JSON and NPZ files')
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'testcases/outputs').resolve()):
        parser.error('--output must be below this checkout testcases/outputs/')
    if output.exists():
        parser.error('--output must be new; preserve previous receipts')
    if not hasattr(signal, 'setitimer'):
        parser.error('This bounded local runner requires POSIX setitimer (macOS/Linux)')
    output.mkdir(parents=True)
    report = {'schema': SCHEMA, 'no_training': True, 'optimizer_updates': 0,
              'policy_executions': 0, 'seeds': args.seeds, 'max_wall_seconds': args.max_wall_seconds,
              'candidate_interface': 'declared nominal skills' if args.skill_options else 'legacy nine raw left-slot choices',
              'generated_runs': [], 'gates': {
                  name: {'status': 'open', 'reason': reason} for name, reason in {
                      'patient_skill_throughput': 'No patient skills executed by this check.',
                      'end_to_end_10hz': 'Short generated script timing does not establish sustained patient/actor throughput.',
                      'full_patient_task_feasibility': 'Generated coupon success does not establish full-patient feasibility.',
                      'held_out_generalization': 'Seeds repeat one generated geometry, not independent patients.',
                      'anatomical_calibration': 'No reviewed anatomy or physics calibration supplied by this check.',
                      'clinical_validation': 'Uncalibrated research simulator; no clinical validation.',
                      'bounded_duration_cusa_mechanics': 'Generated opt-in v3 contact/time capacity is an uncalibrated surrogate; original patient profile is not upgraded or qualified.',
                      'patient_reset_observation': 'Optional patient reset benchmark not requested.'}.items()}}
    measured_gates = ('generated_scripted_task', 'native_snapshot_identity', 'rl_snapshot_resume_identity',
                      'generated_conservation', 'interrupted_primitive_no_commit', 'catalog_tip_geometry',
                      'generated_cusa_contact_time_budget', 'nonzero_pending_cusa_json_resume')
    report['gates'].update({name: {'status': 'open', 'reason': 'Required generated check not yet completed.'}
                            for name in measured_gates})
    budget = Budget(args.max_wall_seconds)
    def timeout(*_):
        raise TimeoutError('Readiness wall-time budget exhausted')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    exit_code = 0
    try:
        signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
        for seed in args.seeds:
            report['generated_runs'].append(run_seed(seed, output, budget, skill_options=args.skill_options))
        report['control_probes'], seconds = budget.timed(lambda: run_control_probes(budget))
        report['control_probes']['total_seconds'] = seconds
        report['tip_probes'], seconds = budget.timed(lambda: run_tip_probes(budget))
        report['tip_probes']['total_seconds'] = seconds
        endpoints = {run['endpoint_state_id'] for run in report['generated_runs']}
        require(len(endpoints) == 1, 'Deterministic generated endpoints differ across seeds')
        report['deterministic_endpoint_across_seeds'] = True
        for name in measured_gates:
            report['gates'][name] = {'status': 'passed', 'scope': 'generated engineering fixtures only'}
        if args.patient:
            report['patient_reset_only'] = patient_reset_only(args.patient.resolve(), args.protection.resolve() if args.protection else None, budget)
            report['gates']['patient_reset_observation'] = {'status': 'passed', 'scope': 'source-preserving reset/observation, zero physics actions'}
        report['status'] = 'passed_measured_checks_with_open_gates'
    except Exception as error:
        report['status'] = 'incomplete' if isinstance(error, TimeoutError) else 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        exit_code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        report['wall_seconds'] = time.perf_counter() - budget.started
        write_json(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                      'wall_seconds': report['wall_seconds'], 'no_training': True,
                      'optimizer_updates': 0}, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
