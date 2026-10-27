#!/usr/bin/env python3
"""Bounded same-case high-level CUSA BC/PPO and prespecified DEVELOPMENT transfer.

Only fresh source-bound native skill options are executed. The actor receives
published observations, while source sectors, previews, option receipts and
native snapshots remain privileged audit data. Both DEVELOPMENT sources and
all failed gates remain in the result denominator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import scipy
import torch

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.patient_sector_learning import (ACTION_IDS, MAX_NATIVE_ACTIONS,
    OBSERVATION_SCHEMA, SCHEMA, DEVELOPMENT_ACCESS_METHOD, OptionActor, OptionCritic, actor_choice, fit_bc,
    fixed_choice, frozen_actor, model_binding, parameter_hash, ppo_update,
    prepare_case, rollout, search_control, seal_actor, sha256)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_verified_cache import implementation_fingerprint
from resectionlab.workspace_bundle import load_workspace, save_workspace


SEEDS = (17, 29, 43)
PPO_UPDATES = 2
PPO_EPISODES_PER_UPDATE = 2
MAX_SUITE_SECONDS = 900.
MAX_SUITE_PHYSICAL_CALLS = 3600
MAX_ALLOWED_SUITE_PHYSICAL_CALLS = 6000
PPO_SOURCE_SCHEDULES = ('train-only', 'paired-development')


def ppo_episode_plan(schedule):
    """Declare all sampled source assignments before any patient physics."""
    if schedule not in PPO_SOURCE_SCHEDULES:
        raise ValueError('Unknown PPO source schedule')
    if PPO_UPDATES != 2 or PPO_EPISODES_PER_UPDATE != 2:
        raise ValueError('Fixed paired DEVELOPMENT PPO protocol changed')
    return [{'update': update, 'episode': episode,
             'sourceKey': ('transfer' if schedule == 'paired-development' and
                           episode == 1 else 'train')}
            for update in range(PPO_UPDATES)
            for episode in range(PPO_EPISODES_PER_UPDATE)]


def optimization_role(stage, source_key, schedule):
    if stage not in ('bc', 'ppo') or source_key not in ('train', 'transfer') or \
            schedule not in PPO_SOURCE_SCHEDULES:
        raise ValueError('Unknown DEVELOPMENT optimization role')
    if stage == 'bc':
        return 'DEV_BC_TEACHER' if source_key == 'train' else 'DEV_EVALUATION_ONLY'
    if source_key == 'train' or schedule == 'paired-development':
        return 'DEV_TRAINED_FOR_PPO'
    return 'DEV_FROZEN_TRANSFER_EVALUATION'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def producer_identity():
    engine = implementation_fingerprint()
    if engine is None:
        raise ValueError('Authoritative surgical engine fingerprint unavailable')
    from resectionlab import patient_sector_learning as policy
    from resectionlab import surgical_sector_planning as skill
    from resectionlab import surgical_rl_env as sensors
    return {'runnerSourceSha256': sha256(__file__),
            'policySourceSha256': sha256(policy.__file__),
            'skillSourceSha256': sha256(skill.__file__),
            'sensorSourceSha256': sha256(sensors.__file__),
            'authoritativeEngineFingerprint': engine,
            'runtime': {'python': sys.version, 'cacheTag': sys.implementation.cache_tag,
                'platform': platform.platform(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'torch': str(torch.__version__)}}


def development_declaration(source):
    """Check the same saved patient/revision against its earlier source role."""
    case, artifacts, _ = load_workspace(source)
    adapter = case.metadata.get('surgical_adapter', {})
    manifest = adapter.get('sourceManifest', {})
    patient = adapter.get('sourcePatient')
    series = manifest.get('series', ())
    revisions = {row.get('source_mirror_revision') for row in series}
    if (not isinstance(patient, str) or not patient or
            manifest.get('patient_id') != patient or not series or
            any(row.get('patient_id') != patient for row in series) or
            len(revisions) != 1 or None in revisions or
            artifacts.get('surgical_toolbox', {}).get('planning', {}).get('task') != 'anatomy-access'):
        raise ValueError('Source lacks a consistent predeclared DEVELOPMENT patient/revision')
    matches = []
    for path in sorted((ROOT/'testcases/downloaded-datasets').glob(
            '*/development-declaration.json')):
        declaration = json.loads(path.read_text())
        if patient not in declaration.get('cases', ()):
            continue
        if (declaration.get('schema') !=
                'resectionlab.fresh-source-development-declaration.v1' or
                declaration.get('role') != 'DEVELOPMENT' or
                declaration.get('mirror_revision') != next(iter(revisions)) or
                not declaration.get('selection_rule') or
                not declaration.get('created_at')):
            raise ValueError('Matching source has a conflicting DEVELOPMENT declaration')
        matches.append((path, declaration))
    if len(matches) != 1:
        raise ValueError('Source has no unique pre-pixel DEVELOPMENT declaration')
    path, declaration = matches[0]
    return {'role': 'DEVELOPMENT', 'patientId': patient,
            'mirrorRevision': declaration['mirror_revision'],
            'selectionRule': declaration['selection_rule'],
            'declarationPath': str(path.resolve()),
            'declarationSha256': sha256(path)}


def _physical_calls(report):
    cost = report['cost']
    return cost['previewCalls']+cost['committedCalls']+cost['independentReplayCalls']


def _reports(search):
    return [item['report'] for item in search['variants']]


def _source_validation_calls(prepared):
    # Saved comparison validation checks source content/seals. It does not
    # replay all retained access branches. Only the active intact restore ran.
    _, artifacts, _ = load_workspace(prepared.source)
    record = artifacts['surgical_toolbox']
    return {'actualReplayCalls': len(record['branches'][record['branchId']]['actions']),
            'savedAccessBranchActionInventory': sum(
                len(record['branches'][row['branchId']]['actions'])
                for row in record['planning']['alternatives'])}


def export_native(prepared, trial, model, role, output, *, seed, declaration,
                  optimizer_updates, ppo_source_schedule='train-only',
                  optimization_source_bindings=None):
    """Save the actual actor-selected history; reopen and check its native state."""
    case, artifacts, workspace = load_workspace(prepared.source)
    artifacts = copy.deepcopy(artifacts)
    record = artifacts['surgical_toolbox']
    actor = trial['report']
    branch_id = 'learned-sector-'+model['stage']+'-'+role+'-seed'+str(seed)
    if branch_id in record['branches']:
        raise ValueError('Derived native strategy branch already exists')
    record['branches'][branch_id] = trial['engine'].snapshot()
    record.setdefault('parents', {})[branch_id] = {
        'branchId': record['branchId'], 'position': 0}
    if model['stage'] not in ('bc', 'bc-initialized-ppo'):
        raise ValueError('Unknown exported checkpoint stage')
    source_use = optimization_role(
        'bc' if model['stage'] == 'bc' else 'ppo', role, ppo_source_schedule)
    optimization_bindings = (optimization_source_bindings if
                             optimization_source_bindings is not None else
                             {'train': model['sourceBinding']})
    if (set(optimization_bindings) !=
            ({'train', 'transfer'} if model['stage'] == 'bc-initialized-ppo' and
              ppo_source_schedule == 'paired-development' else {'train'}) or
            optimization_bindings['train'] != model['sourceBinding']):
        raise ValueError('Export optimization source bindings differ')
    evaluation_role = (f'paired-development-{model["stage"]}-{role}-evaluation'
                       if ppo_source_schedule == 'paired-development' else role)
    metadata = {'schema': SCHEMA, 'branchId': branch_id,
        'sourceRole': 'DEVELOPMENT', 'evaluationRole': evaluation_role,
        'sourceRoleDetail': source_use,
        'optimizationRole': source_use,
        'ppoSourceSchedule': ppo_source_schedule,
        'evaluationNote': (
            'The internal transfer source key names a second declared DEVELOPMENT '
            'case assigned to PPO optimization; neither case is held out.'
            if ppo_source_schedule == 'paired-development' else
            'Same-case declared DEVELOPMENT evaluation.' if role == 'train' else
            'Prespecified second DEVELOPMENT case; not a held-out assessment.'),
        'trainingSourceBinding': model['sourceBinding'],
        'trainingOptimizationSourceBindings': copy.deepcopy(optimization_bindings),
        'executionSourceArchiveSha256': prepared.source_sha256,
        'executionSourceSceneId': prepared.scene.identity,
        'executionAccessComparisonSeal': prepared.access_comparison_seal,
        'executionAccessPlanSeal': prepared.access['planSeal'],
        'executionAccessMethod': prepared.access_method,
        'sourceRoleDeclaration': declaration,
        'checkpointFileSha256': model['fileSha256'],
        'actorSha256': model['actorSha256'],
        'trainingReceiptSha256': model['trainingReceiptSha256'],
        'observationSchema': OBSERVATION_SCHEMA, 'actionIds': list(ACTION_IDS),
        'optimizerUpdates': optimizer_updates,
        'status': actor['status'], 'taskGateMet': actor['taskGateMet'],
        'cleanupComplete': actor['cleanupComplete'],
        'deltaTargetMm3': actor['deltaTargetMm3'],
        'deltaNonTargetMm3': actor['deltaNonTargetMm3'],
        'totalNonTargetMm3': actor['totalNonTargetMm3'],
        'perSectorRemovedMm3': actor['perSectorRemovedMm3'],
        'highLevelDecisions': [row['optionId'] for row in actor['decisions']
                               if row['boundary'] != 'forced_cleanup'],
        'policyDecisions': copy.deepcopy(actor['decisions']),
        'executionCost': copy.deepcopy(actor['cost']),
        'nativeActions': actor['nativeActions'], 'prefixActions': 0,
        'finalStateId': actor['finalStateId'],
        'exactNativeReplay': actor['exactNativeReplay'],
        'promotionStatus': ('no-clinical-promotion-development-only' if actor['taskGateMet']
                            else 'not-promoted-unpassed-development-episode'),
        'limitations': [
            ('Both declared DEVELOPMENT cases are assigned to PPO updates; '
             'paired PPO checkpoints have DEV_TRAINED_FOR_PPO exposure to both; '
             'no held-out or generalization claim'
             if ppo_source_schedule == 'paired-development' else
             'Two declared DEVELOPMENT cases; no held-out or clinical claim'),
                        'Skull and complete critical anatomy remain unsupported',
                        'Fixed native skill library; actor only chooses high-level options']}
    record['learnedSectorExecutions'] = [metadata]
    artifacts['patient_sector_learning'] = metadata
    destination = output/(role+'-'+model['stage']+'-seed'+str(seed)+'.ressectionlab')
    if workspace is None:
        save_case(case, destination, artifacts=artifacts)
    else:
        save_workspace(case, destination, sources=workspace['sources'],
            imaging=workspace['imagingState'], episode=workspace['episode'],
            selection=workspace['selection'], originals=workspace['originals'],
            artifacts=artifacts)
    reopened_case, reopened_artifacts, _ = load_workspace(destination)
    saved = reopened_artifacts['surgical_toolbox']['branches'][branch_id]
    reopened = SurgicalEngine.restore(prepared.scene, saved)
    if (reopened_case.semantic_hash != case.semantic_hash or
            reopened.state_id != trial['engine'].state_id or
            reopened.actions != trial['engine'].actions or
            reopened_artifacts['patient_sector_learning'] != metadata):
        raise ValueError('Derived frozen native strategy changed after save/reopen')
    return {'workspacePath': str(destination.resolve()),
            'fileSha256': sha256(destination), 'branchId': branch_id,
            'finalStateId': reopened.state_id, 'reopenedExact': True,
            'metadata': metadata, 'nativeReplayCalls': len(reopened.actions)}


def _suite_check(started, reports, *, source_replays=0, exported_replays=0,
                 max_physical_calls=MAX_SUITE_PHYSICAL_CALLS):
    elapsed = time.perf_counter()-started
    calls = (sum(_physical_calls(row) for row in reports)+
             source_replays+exported_replays)
    if elapsed > MAX_SUITE_SECONDS:
        raise TimeoutError('Hierarchical learning suite wall budget exhausted')
    if calls > max_physical_calls:
        raise TimeoutError('Hierarchical learning suite native-call budget exhausted')
    return {'wallSeconds': elapsed, 'physicalCalls': calls,
            'maxWallSeconds': MAX_SUITE_SECONDS,
            'maxPhysicalCalls': max_physical_calls}


def run(train_source, transfer_source, output, *, fit=False,
        train_access_method=DEVELOPMENT_ACCESS_METHOD,
        transfer_access_method=DEVELOPMENT_ACCESS_METHOD,
        max_suite_physical_calls=MAX_SUITE_PHYSICAL_CALLS,
        ppo_source_schedule='train-only'):
    if (type(max_suite_physical_calls) is not int or
            not 1 <= max_suite_physical_calls <= MAX_ALLOWED_SUITE_PHYSICAL_CALLS):
        raise ValueError('Suite native-call cap must be an integer in 1..6000')
    if (not isinstance(train_access_method, str) or not train_access_method or
            not isinstance(transfer_access_method, str) or not transfer_access_method):
        raise ValueError('Both selected access methods must be nonempty strings')
    episode_plan = ppo_episode_plan(ppo_source_schedule)
    output = Path(output).resolve()
    if output.exists() or 'testcases' not in output.parts:
        raise ValueError('Choose a new ignored testcases/ output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {'schema': SCHEMA, 'status': 'running',
        'mode': 'fit' if fit else 'profile',
        'scope': (
            '0007 and 0005 declared DEVELOPMENT; 0007 BC teacher; one episode '
            'from each case per PPO update is preassigned; profile performs no fit; '
            'no held-out or generalization claim'
            if ppo_source_schedule == 'paired-development' and not fit else
            '0007 and 0005 declared DEVELOPMENT; 0007 BC teacher; both used for '
            'paired PPO optimization; no held-out or generalization claim'
            if ppo_source_schedule == 'paired-development' else
            '0007 DEVELOPMENT same-case fit; prespecified 0005 DEVELOPMENT frozen transfer'),
        'sourceRole': 'DEVELOPMENT', 'seeds': SEEDS,
        'ppoSourceSchedule': ppo_source_schedule,
        'ppoEpisodePlan': episode_plan,
        'ppoScheduleExecuted': False,
        'plannedOptimizationRoles': {
            label: {'bc': optimization_role('bc', label, ppo_source_schedule),
                    'ppo': optimization_role('ppo', label, ppo_source_schedule)}
            for label in ('train', 'transfer')},
        'requestedAccessMethods': {'train': train_access_method,
                                   'transfer': transfer_access_method},
        'requestedMaxSuitePhysicalCalls': max_suite_physical_calls,
        'ppoProtocol': {'updates': PPO_UPDATES,
                        'sampledEpisodesPerUpdate': PPO_EPISODES_PER_UPDATE,
                        'finalCheckpointOnly': True},
        'taskGate': {'additionalTargetMm3': 30., 'eachSectorMm3': 10.,
                     'additionalNonTargetMm3': 20., 'totalNonTargetMm3': 200.,
                     'minAdmittedSectorOptions': 2,
                     'maxNativeActions': MAX_NATIVE_ACTIONS},
        'sources': {}, 'controls': {}, 'seedRuns': []}
    reports = []
    source_replays = exported_replays = 0
    try:
        prepared = {}
        for label, source, access_method in (
                ('train', train_source, train_access_method),
                ('transfer', transfer_source, transfer_access_method)):
            declaration = development_declaration(source)
            case = prepare_case(source, access_method=access_method)
            prepared[label] = (case, declaration)
            source_validation = _source_validation_calls(case)
            validation_replays = source_validation['actualReplayCalls']
            source_replays += validation_replays
            receipt['sources'][label] = {'path': str(case.source),
                'fileSha256': case.source_sha256, 'caseId': case.case_id,
                'sceneId': case.scene.identity,
                'accessPlanSeal': case.access['planSeal'],
                'accessMethod': case.access_method,
                'plannedOptimizationRoles': receipt['plannedOptimizationRoles'][label],
                'sectorCellSha256': {name: value['sourceCellSha256']
                                     for name, value in case.sector_evidence.items()},
                'roleDeclaration': declaration,
                'sourceValidationReplayCalls': validation_replays,
                'savedAccessBranchActionInventory':
                    source_validation['savedAccessBranchActionInventory'],
                'accessComparisonContentValidated': True}
        producer = producer_identity()
        receipt['producerIdentity'] = producer
        searches = {}
        for label, (case, _) in prepared.items():
            script = rollout(case, fixed_choice(('corridor:left','near:left',
                'far:left','STOP')), method='fixed-near-first-left')
            reports.append(script['report'])
            search = search_control(case)
            reports.extend(_reports(search))
            searches[label] = search
            receipt['controls'][label] = {'scripted': script['report'],
                'boundedSearch': {'variants': _reports(search),
                    'selectionRule': search['selectionRule'],
                    'teacherMethod': (search['teacher']['report']['method']
                                      if search['teacher'] else None)}}
            _suite_check(started, reports, source_replays=source_replays,
                         exported_replays=exported_replays,
                         max_physical_calls=max_suite_physical_calls)
        teacher = searches['train']['teacher']
        if not teacher or not teacher['report']['taskGateMet']:
            raise ValueError('0007 has no successful checked high-level BC demonstration')
        # The 0005 failed comparison is retained. It never becomes a positive label.
        receipt['deterministicFeasibility'] = {
            'trainSuccess': True,
            'transferSuccess': searches['transfer']['teacher'] is not None,
            'transferFailureRetained': searches['transfer']['teacher'] is None}
        if not fit:
            receipt['status'] = 'profiled'
            return receipt
        train, train_declaration = prepared['train']
        transfer, transfer_declaration = prepared['transfer']
        for seed in SEEDS:
            run_receipt = {'seed': seed, 'untrained': {}, 'bc': {}, 'ppo': {},
                           'ppoSampledEpisodeAssignments': [], 'failures': []}
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                untrained = OptionActor(); untrained.eval()
            run_receipt['untrainedActorSha256'] = parameter_hash(untrained)
            for label, (case, _) in prepared.items():
                trial = rollout(case, actor_choice(untrained),
                                method='untrained-same-architecture')
                reports.append(trial['report'])
                run_receipt['untrained'][label] = trial['report']
            try:
                actor, bc_fit = fit_bc(teacher, seed=seed)
            except Exception as error:
                run_receipt['failures'].append({
                    'stage': 'bc_fit', 'type': type(error).__name__,
                    'message': str(error)})
                run_receipt['bc']['status'] = 'fit_failed'
                run_receipt['ppo']['status'] = 'not_started_bc_fit_failed'
                receipt['seedRuns'].append(run_receipt)
                continue
            bc_training = {'sourceBinding': model_binding(train),
                'producerIdentity': producer, 'bcFit': bc_fit,
                'optimizationSourceRoles': {'train': 'DEV_BC_TEACHER',
                                            'transfer': 'DEV_EVALUATION_ONLY'},
                'demonstration': teacher['report'],
                'demonstrationNativeCalls': _physical_calls(teacher['report']),
                'role': 'same-case DEVELOPMENT supervised high-level option fit'}
            bc_model = seal_actor(actor, output/('bc-seed'+str(seed)+'.pt'),
                stage='bc', prepared=train, producer=producer,
                training_receipt=bc_training)
            bc_actor = frozen_actor(bc_model['path'], expected_producer=producer)
            run_receipt['bc'] = {'fit': bc_fit, 'model': bc_model, 'evaluations': {}}
            for label, (case, declaration) in prepared.items():
                trial = rollout(case, actor_choice(bc_actor), method='frozen-bc-live')
                reports.append(trial['report'])
                derived = export_native(case, trial, bc_model, label, output,
                    seed=seed, declaration=declaration, optimizer_updates=0,
                    ppo_source_schedule=ppo_source_schedule)
                exported_replays += derived['nativeReplayCalls']
                run_receipt['bc']['evaluations'][label] = {'report': trial['report'],
                                                           'derived': derived}
            if not run_receipt['bc']['evaluations']['train']['report']['taskGateMet']:
                run_receipt['failures'].append('Frozen BC missed the 0007 live gate; PPO not started')
                run_receipt['ppo']['status'] = 'not_started_bc_live_gate_failed'
                receipt['seedRuns'].append(run_receipt)
                _suite_check(started, reports, source_replays=source_replays,
                             exported_replays=exported_replays,
                             max_physical_calls=max_suite_physical_calls)
                continue
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                critic = OptionCritic(); critic.eval()
            generator = torch.Generator().manual_seed(seed)
            updates = []
            sampled_reports = []
            ppo_update_failure = None
            for update in range(PPO_UPDATES):
                episodes = []
                for episode in range(PPO_EPISODES_PER_UPDATE):
                    assignment = episode_plan[update*PPO_EPISODES_PER_UPDATE+episode]
                    source_key = assignment['sourceKey']
                    sample_case, _ = prepared[source_key]
                    trial = rollout(sample_case, actor_choice(actor, sample=True,
                        generator=generator), method=f'ppo-sampled-{update}-{episode}')
                    episodes.append(trial)
                    sampled_reports.append(trial['report'])
                    run_receipt['ppoSampledEpisodeAssignments'].append({
                        **assignment, 'sourceArchiveSha256': sample_case.source_sha256,
                        'optimizationRole': optimization_role(
                            'ppo', source_key, ppo_source_schedule)})
                    reports.append(trial['report'])
                    _suite_check(started, reports, source_replays=source_replays,
                                 exported_replays=exported_replays,
                                 max_physical_calls=max_suite_physical_calls)
                try:
                    updates.append(ppo_update(actor, critic, episodes))
                except Exception as error:
                    ppo_update_failure = {'stage': 'ppo_update',
                        'update': update, 'type': type(error).__name__,
                        'message': str(error)}
                    break
            if ppo_update_failure is not None:
                run_receipt['failures'].append(ppo_update_failure)
                run_receipt['ppo'] = {'status': 'update_failed',
                    'updates': updates, 'sampledEpisodes': sampled_reports}
                receipt['seedRuns'].append(run_receipt)
                continue
            ppo_training = {'sourceBinding': model_binding(train),
                'producerIdentity': producer,
                'bcCheckpointSha256': bc_model['fileSha256'],
                'bcActorSha256': bc_model['actorSha256'],
                'ppoSourceSchedule': ppo_source_schedule,
                'ppoEpisodePlan': episode_plan,
                'sampledEpisodeAssignments': run_receipt['ppoSampledEpisodeAssignments'],
                'optimizationSourceBindings': {
                    label: model_binding(prepared[label][0])
                    for label in (('train', 'transfer') if
                                  ppo_source_schedule == 'paired-development' else ('train',))},
                'optimizationSourceRoles': {
                    label: optimization_role('ppo', label, ppo_source_schedule)
                    for label in (('train', 'transfer') if
                                  ppo_source_schedule == 'paired-development' else ('train',))},
                'ppoUpdates': updates, 'sampledEpisodes': sampled_reports,
                'actualPublicOptionDecisions': sum(len(row['decisions']) for row in sampled_reports),
                'role': ('paired DEVELOPMENT BC-initialized PPO high-level option fit'
                         if ppo_source_schedule == 'paired-development' else
                         'same-case DEVELOPMENT BC-initialized PPO high-level option fit')}
            ppo_model = seal_actor(actor, output/('ppo-seed'+str(seed)+'.pt'),
                stage='bc-initialized-ppo', prepared=train, producer=producer,
                training_receipt=ppo_training)
            ppo_actor = frozen_actor(ppo_model['path'], expected_producer=producer)
            run_receipt['ppo'] = {'status': 'evaluated', 'updates': updates,
                                  'sampledEpisodes': sampled_reports,
                                  'model': ppo_model, 'evaluations': {}}
            for label, (case, declaration) in prepared.items():
                trial = rollout(case, actor_choice(ppo_actor),
                                method='frozen-ppo-live')
                reports.append(trial['report'])
                derived = export_native(case, trial, ppo_model, label, output,
                    seed=seed, declaration=declaration,
                    optimizer_updates=len(updates),
                    ppo_source_schedule=ppo_source_schedule,
                    optimization_source_bindings=ppo_training['optimizationSourceBindings'])
                exported_replays += derived['nativeReplayCalls']
                run_receipt['ppo']['evaluations'][label] = {'report': trial['report'],
                                                            'derived': derived}
            receipt['seedRuns'].append(run_receipt)
            _suite_check(started, reports, source_replays=source_replays,
                         exported_replays=exported_replays,
                         max_physical_calls=max_suite_physical_calls)
        receipt['ppoScheduleExecuted'] = all(
            len(row['ppoSampledEpisodeAssignments']) == len(episode_plan)
            for row in receipt['seedRuns'])
        receipt['status'] = 'saved'
        return receipt
    except BaseException as error:
        receipt['status'] = 'failed'
        receipt['failure'] = {'type': type(error).__name__, 'message': str(error),
                              'traceback': traceback.format_exc(limit=8)}
        raise
    finally:
        rollout_calls = sum(_physical_calls(row) for row in reports)
        receipt['cost'] = {'wallSeconds': time.perf_counter()-started,
            'physicalCalls': rollout_calls+source_replays+exported_replays,
            'optionRolloutCalls': rollout_calls,
            'sourceValidationReplayCalls': source_replays,
            'exportedWorkspaceReplayCalls': exported_replays,
            'maxWallSeconds': MAX_SUITE_SECONDS,
            'maxPhysicalCalls': max_suite_physical_calls,
            'reportCount': len(reports)}
        receipt['sourcesUnchanged'] = {label: sha256(row['path']) == row['fileSha256']
            for label, row in receipt['sources'].items()}
        write_json(output/'receipt.json', receipt)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-source', type=Path, required=True)
    parser.add_argument('--transfer-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-role', choices=['DEVELOPMENT'], required=True)
    parser.add_argument('--train-access-method', default=DEVELOPMENT_ACCESS_METHOD,
        help='Saved anatomy-access method for the 0007 fit source')
    parser.add_argument('--transfer-access-method', default=DEVELOPMENT_ACCESS_METHOD,
        help='Saved anatomy-access method for the 0005 frozen transfer source')
    parser.add_argument('--max-suite-physical-calls', type=int,
        default=MAX_SUITE_PHYSICAL_CALLS,
        help='Aggregate native-call cap, default 3600 and hard maximum 6000')
    parser.add_argument('--ppo-source-schedule', choices=PPO_SOURCE_SCHEDULES,
        default='train-only',
        help='Fixed PPO episode sources: two 0007 episodes per update, or one 0007 then one 0005 DEVELOPMENT episode')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fit', action='store_true',
        help='Fit fixed-seed BC/PPO only after the source implementation is frozen')
    mode.add_argument('--profile', action='store_true',
        help='Run non-fitting deterministic controls only (the default)')
    return parser


def main():
    args = build_parser().parse_args()
    result = run(args.train_source, args.transfer_source, args.output, fit=args.fit,
        train_access_method=args.train_access_method,
        transfer_access_method=args.transfer_access_method,
        max_suite_physical_calls=args.max_suite_physical_calls,
        ppo_source_schedule=args.ppo_source_schedule)
    print(json.dumps({'status': result['status'], 'mode': result['mode'],
        'cost': result['cost'], 'seeds': result['seeds']}))


if __name__ == '__main__':
    main()
