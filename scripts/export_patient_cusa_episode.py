#!/usr/bin/env python3
"""Export cold-replayed, unpassed CUSA episodes as inspectable native branches.

This exporter fits no actor and changes no source. It refuses a successful
episode: the package is explicitly an ABSTAINED/INCOMPLETE DEVELOPMENT record,
never a promoted checkpoint or a claim that the task gate passed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.patient_cusa_learning import (
    ACTION_IDS, Budget, CONTROLLER_REVISION, TARGET_GOAL_MM3,
    MIN_ADMITTED_ACTIONS, actor_choice, execute, prepare, sha256,
)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_planning import validate_saved_comparison
from resectionlab.workspace_bundle import load_workspace, save_workspace
from run_patient_cusa_learning import frozen_actor, physics_audit, producer_identity

SCHEMA = 'resectionlab.patient-cusa-unpassed-export.v1'
_ATTEMPT_FIELDS = ('durationSeconds', 'previewStatus', 'reason', 'refusal',
    'previewStateId', 'previewTargetRemovedMm3', 'previewNonTargetRemovedMm3',
    'commitStatus', 'commitStateId')
_EVENT_FIELDS = ('observationHash', 'featureHash', 'actionId', 'actorAction',
    'action', 'previewStatus', 'status', 'reason', 'engineReason', 'stateId',
    'controllerRecoveryApplied')
_EPISODE_FIELDS = ('status', 'stopReason', 'deltaTargetMm3',
    'deltaNonTargetMm3', 'admittedNonStopActions', 'taskGateMet',
    'cleanupComplete', 'finalStateId', 'sourceStateId', 'nativeActions',
    'controllerRevision')


def require_unpassed(episode):
    if episode.get('status') not in ('abstained', 'incomplete') or episode.get('taskGateMet'):
        raise ValueError('Unpassed exporter refuses successful or task-gate-passing episodes')


def _stable_attempt(attempt):
    return {key: attempt.get(key) for key in _ATTEMPT_FIELDS}


def _stable_event(event):
    return {**{key: event.get(key) for key in _EVENT_FIELDS},
            'attempts': [_stable_attempt(item) for item in event.get('attempts', [])]}


def compare_episode(recorded, actual, prefix_position):
    """Check all published decisions, preview outcomes and committed states."""
    require_unpassed(recorded)
    require_unpassed(actual['receipt'])
    fresh = actual['receipt']
    for key in _EPISODE_FIELDS:
        if recorded.get(key) != fresh.get(key):
            raise ValueError('Frozen episode changed: ' + key)
    for section in ('decisions', 'cleanup'):
        left = [_stable_event(row) for row in recorded[section]]
        right = [_stable_event(row) for row in fresh[section]]
        if left != right:
            raise ValueError('Frozen episode changed: ' + section)
    committed = [row['stateId'] for section in ('decisions', 'cleanup')
                 for row in fresh[section] if row['status'] in ('accepted', 'partial')]
    states = [row['stateAfter'] for row in actual['engine'].history[prefix_position:]]
    if states != committed or len(states) != fresh['nativeActions']:
        raise ValueError('Frozen episode committed action states differ')
    if actual['engine'].state_id != fresh['finalStateId']:
        raise ValueError('Frozen episode final native state differs')
    return {'allDecisionAndCleanupEventsMatch': True,
            'allCommittedActionStatesMatch': True,
            'committedStateIds': states, 'finalStateId': actual['engine'].state_id}


def failure_reasons(episode):
    require_unpassed(episode)
    reasons = []
    if episode['deltaTargetMm3'] < TARGET_GOAL_MM3:
        reasons.append('target_progress_below_declared_30_mm3_goal')
    if episode['admittedNonStopActions'] < MIN_ADMITTED_ACTIONS:
        reasons.append('fewer_than_two_admitted_non_stop_actions')
    if not episode['cleanupComplete']:
        reasons.append('checked_cleanup_incomplete')
    if episode['status'] == 'incomplete':
        reasons.append('episode_incomplete_' + str(episode['stopReason']))
    if not reasons:
        reasons.append('declared_task_gate_unpassed')
    return reasons


def _check_binding(recorded, prepared):
    if recorded['source']['sha256'] != prepared['binding']['sourceSha256'] or not recorded['source']['unchanged']:
        raise ValueError('Frozen receipt source archive identity differs')
    for key, value in prepared['binding'].items():
        if key != 'sourcePath' and recorded['prefix'].get(key) != value:
            raise ValueError('Frozen receipt source prefix differs: ' + key)
    if recorded['controllerRevision'] != CONTROLLER_REVISION or recorded['task']['role'] != 'DEVELOPMENT':
        raise ValueError('Frozen receipt controller or DEVELOPMENT role differs')
    if recorded['physicsAudit'] != physics_audit(prepared['scene']):
        raise ValueError('Frozen receipt physics audit differs')
    if recorded['producerIdentity'] != producer_identity():
        raise ValueError('Frozen receipt producer identity differs')


def _load_and_replay(prepared, receipt_path, checkpoint_path, budget):
    recorded = json.loads(Path(receipt_path).read_text())
    if recorded.get('status') != 'evaluated' or not isinstance(recorded.get('execution'), dict):
        raise ValueError('Expected one previously evaluated frozen episode')
    require_unpassed(recorded['execution'])
    _check_binding(recorded, prepared)
    checkpoint_path = Path(checkpoint_path)
    if sha256(checkpoint_path) != recorded['model']['fileSha256']:
        raise ValueError('Frozen checkpoint file SHA differs from evaluation receipt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['actorSha256'] != recorded['model']['actorSha256'] or
            checkpoint['stage'] != recorded['model']['stage'] or
            checkpoint['controllerRevision'] != CONTROLLER_REVISION or
            checkpoint['producerIdentity'] != recorded['producerIdentity'] or
            semantic_digest(checkpoint['trainingReceipt']) != recorded['model']['trainingReceiptSha256']):
        raise ValueError('Frozen checkpoint does not match recorded producer and fit')
    actor = frozen_actor(checkpoint_path, checkpoint['actorSha256'], scene=prepared['scene'])
    episode = execute(prepared, actor_choice(actor), budget=budget,
                      method='frozen-transfer-live')
    comparison = compare_episode(recorded['execution'], episode,
                                 prepared['binding']['position'])
    stage = 'ppo' if checkpoint['stage'] == 'bc-initialized-ppo' else 'bc'
    return {'stage':stage, 'episode':episode, 'recorded':recorded,
            'checkpoint':checkpoint, 'checkpointPath':str(checkpoint_path.resolve()),
            'checkpointSha256':sha256(checkpoint_path),
            'receiptPath':str(Path(receipt_path).resolve()),
            'receiptSha256':sha256(receipt_path), 'comparison':comparison}


def _save_unpassed(prepared, verified, destination, budget):
    artifacts = copy.deepcopy(prepared['artifacts'])
    record = artifacts['surgical_toolbox']
    entries = []
    for item in verified:
        episode = item['episode']['receipt']
        actor_sha = item['checkpoint']['actorSha256']
        branch_id = 'learned-cusa-' + item['stage'] + '-unpassed-' + actor_sha[:10]
        if branch_id in record['branches']:
            raise ValueError('Unpassed learned branch already exists')
        record['branches'][branch_id] = item['episode']['engine'].snapshot()
        record.setdefault('parents', {})[branch_id] = {
            'branchId':prepared['binding']['branchId'],
            'position':prepared['binding']['position']}
        entries.append({'schema':item['checkpoint']['schema'],
            'controllerRevision':CONTROLLER_REVISION,
            'exportSchema':SCHEMA, 'method':item['stage'],
            'branchId':branch_id, 'parentBranchId':prepared['binding']['branchId'],
            'parentPosition':prepared['binding']['position'],
            'sourceStateId':prepared['binding']['stateId'],
            'sourceSceneId':prepared['binding']['sceneId'],
            'sourceArchiveSha256':prepared['binding']['sourceSha256'],
            'modelSha256':actor_sha,
            'checkpointFileSha256':item['checkpointSha256'],
            'trainingReceiptSha256':semantic_digest(item['checkpoint']['trainingReceipt']),
            'producerIdentitySha256':semantic_digest(item['checkpoint']['producerIdentity']),
            'fitPhysicsAudit':item['checkpoint']['fitPhysicsAudit'],
            'fitSourceSceneId':item['checkpoint']['sourceBinding']['sceneId'],
            'fitPrefixStateId':item['checkpoint']['sourceBinding']['stateId'],
            'observationSchema':item['checkpoint']['observationSchema'],
            'actionIds':ACTION_IDS,
            'status':episode['status'], 'taskGateMet':False,
            'promotionStatus':'not-promoted-unpassed-development-episode',
            'failureReasons':failure_reasons(episode),
            'stopReason':episode['stopReason'],
            'deltaTargetMm3':episode['deltaTargetMm3'],
            'deltaNonTargetMm3':episode['deltaNonTargetMm3'],
            'admittedNonStopActions':episode['admittedNonStopActions'],
            'cleanupComplete':episode['cleanupComplete'],
            'finalStateId':episode['finalStateId'],
            'actorDecisions':[row['actionId'] for row in episode['decisions']],
            'recoveryAttempts':[
                {'actionId':row.get('actionId', row.get('action', {}).get('operation')),
                 'attempts':row['attempts']}
                for section in ('decisions','cleanup') for row in episode[section]
                if row.get('attempts') and len(row['attempts']) > 1],
            'optimizerUpdates':len(item['checkpoint']['trainingReceipt'].get('optimizerUpdates', [])),
            'limitations':['Unpassed fixed 30 mm3 / two-stroke task gate',
                           'DEVELOPMENT episode; no clinical or generalization claim',
                           'Source enhancing tumor annotation and unreviewed brain estimate']})
    record.setdefault('learnedExecutions', []).extend(entries)
    record.setdefault('learnedExecution', entries[0])
    validate_saved_comparison(prepared['scene'], record, record['planning'])
    case = prepared['case']
    workspace = prepared['workspace']
    if workspace is None:
        save_case(case, destination, artifacts=artifacts)
    else:
        save_workspace(case, destination, sources=workspace['sources'],
            imaging=workspace['imagingState'], episode=workspace['episode'],
            selection=workspace['selection'], originals=workspace['originals'],
            artifacts=artifacts)
    reopened_case, reopened_artifacts, _ = load_workspace(destination)
    reopened_record = reopened_artifacts['surgical_toolbox']
    validate_saved_comparison(prepared['scene'], reopened_record, reopened_record['planning'])
    if reopened_case.semantic_hash != case.semantic_hash:
        raise ValueError('Native unpassed archive changed source case content')
    for item, entry in zip(verified, entries):
        snapshot = reopened_record['branches'][entry['branchId']]
        budget.reserve(len(snapshot['actions']))
        replay = SurgicalEngine.restore(prepared['scene'], snapshot)
        if (replay.state_id != item['episode']['engine'].state_id or
                replay.state != item['episode']['engine'].state or
                entry['status'] not in ('abstained','incomplete') or
                entry['taskGateMet']):
            raise ValueError('Reopened unpassed branch changed or acquired a positive label')
    return {'path':str(destination.resolve()), 'sha256':sha256(destination),
            'branches':entries, 'reopenedExact':True}


def run(source, branch, episodes, output, *, source_role='DEVELOPMENT'):
    if source_role != 'DEVELOPMENT':
        raise ValueError('Unpassed export requires explicit DEVELOPMENT role')
    if not episodes or len(episodes) > 2:
        raise ValueError('Supply one or two frozen result/checkpoint pairs')
    output = Path(output)
    budget = Budget(max_calls=200)
    receipt = {'schema':SCHEMA, 'status':'running',
        'source':{'path':str(Path(source).resolve()),'sha256':sha256(source)},
        'role':source_role, 'controllerRevision':CONTROLLER_REVISION,
        'physicalTransitionCap':200, 'episodes':[]}
    try:
        prepared = prepare(source, branch, budget=budget)
        receipt['sourcePrefix'] = prepared['binding']
        verified = []
        for receipt_path, checkpoint_path in episodes:
            item = _load_and_replay(prepared, receipt_path, checkpoint_path, budget)
            verified.append(item)
            receipt['episodes'].append({'stage':item['stage'],
                'originalReceiptPath':item['receiptPath'],
                'originalReceiptSha256':item['receiptSha256'],
                'checkpointPath':item['checkpointPath'],
                'checkpointSha256':item['checkpointSha256'],
                'comparison':item['comparison'],
                'status':item['episode']['receipt']['status'],
                'failureReasons':failure_reasons(item['episode']['receipt'])})
        if len({item['stage'] for item in verified}) != len(verified):
            raise ValueError('Choose distinct frozen BC/PPO stages')
        receipt['derived'] = _save_unpassed(prepared, verified,
            output / 'unpassed-learned-continuations.ressectionlab', budget)
        if sha256(source) != receipt['source']['sha256']:
            raise ValueError('Original source archive changed during unpassed export')
        receipt['status'] = 'saved-unpassed'
        return receipt
    except BaseException as error:
        receipt['status'] = 'failed'
        receipt['failure'] = {'type':type(error).__name__, 'message':str(error),
                              'traceback':traceback.format_exc(limit=8)}
        raise
    finally:
        receipt['budget'] = budget.report()
        receipt['source']['unchanged'] = sha256(source) == receipt['source']['sha256']
        output.mkdir(parents=True, exist_ok=True)
        (output / 'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--branch',required=True)
    parser.add_argument('--episode',nargs=2,action='append',required=True,
        metavar=('RECEIPT','CHECKPOINT'),help='Repeat for distinct frozen BC and PPO results')
    parser.add_argument('--source-role',choices=['DEVELOPMENT'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if 'testcases' not in output.parts or output.exists():
        parser.error('Choose a new output directory under ignored testcases/')
    result = run(args.source,args.branch,args.episode,output,source_role=args.source_role)
    print(json.dumps({'status':result['status'],
        'branches':[row['branchId'] for row in result['derived']['branches']],
        'budget':result['budget']}))


if __name__ == '__main__':
    main()
