#!/usr/bin/env python3
"""Export one bounded joint access/sector comparison from an intact source.

The original source archive and its classic branches remain untouched.
Every new branch is replayed after native save/reopen through the planner's
shared public-transition meter. Outputs belong under ignored testcases/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_joint_planning import validate_joint_report
from resectionlab.surgical_planning import (
    validate_modeled_outcome_binding, validate_saved_comparison)
from resectionlab.surgical_sector_planning import validate_saved_sector_comparison
from resectionlab.surgical_transition_meter import TransitionMeter
from resectionlab.workspace_bundle import load_workspace, save_workspace

SCHEMA = 'resectionlab.joint-anatomy-strategy-export.v1'
ROOT = Path(__file__).resolve().parents[1]
TESTCASES = (ROOT/'testcases').resolve()
MAX_ACTIONS = 48
MAX_CALLS = 1500
MAX_SECONDS = 120.
MAX_NEW_BRANCHES = 6


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(case, scene):
    """Store both semantic identities and exact source array shapes/content."""
    return {'caseHash': case.semantic_hash, 'planningHash': case.planning_hash,
            'mri': {'shape': list(case.mri.shape), 'sha256': array_digest(case.mri)},
            'compartments': {name: {'shape': list(data.shape), 'sha256': array_digest(data)}
                             for name, data in sorted(case.compartments.items())},
            'affineSha256': array_digest(case.affine),
            'sceneId': scene.identity, 'sceneShape': list(scene.labels.shape),
            'sceneLabelsSha256': array_digest(scene.labels)}


def _source_record(record, scene):
    if (not isinstance(record, dict) or not isinstance(record.get('branches'), dict) or
            not record['branches'] or 'jointPlanning' in record or
            record.get('branchId') not in record['branches'] or
            record.get('position') != 0 or
            record['branches'][record['branchId']].get('actions') != []):
        raise ValueError('Joint comparison requires an intact, unmodified classic source')
    access, sectors = record.get('planning'), record.get('sectorPlanning')
    if not isinstance(access, dict) or not isinstance(sectors, dict):
        raise ValueError('Classic access and sector receipts are required')
    validate_saved_comparison(scene, record, access)
    validate_saved_sector_comparison(scene, record, sectors)
    if sectors['sourceAccessComparisonSeal'] != access['comparisonSeal']:
        raise ValueError('Classic comparison source seal differs')
    return {'branchId': record['branchId'], 'branchCount': len(record['branches']),
            'branchMapHash': semantic_digest(record['branches']),
            'parentMapHash': semantic_digest(record.get('parents', {})),
            'accessComparisonSeal': access['comparisonSeal'],
            'sectorComparisonSeal': sectors['comparisonSeal']}


def _joint_documents(result, scene, intact):
    if not isinstance(result, dict) or not {
            'accessBranches', 'sectorBranches', 'failedBranches', 'report', 'meter'} <= set(result):
        raise ValueError('Joint planner result is incomplete')
    if not isinstance(result['meter'], TransitionMeter):
        raise ValueError('Joint planner did not return its shared native transition meter')
    if (any(not isinstance(result[key], list) for key in
            ('accessBranches', 'sectorBranches', 'failedBranches')) or
            sum(len(result[key]) for key in
                ('accessBranches', 'sectorBranches', 'failedBranches')) > MAX_NEW_BRANCHES):
        raise ValueError('Joint export exceeds six retained native branches')
    report = result['report']
    if (not isinstance(report, dict) or
            report.get('schema') != 'resectionlab.joint-anatomy-strategy.v1' or
            report.get('sourceSceneId') != scene.identity or
            report.get('startStateId') != intact.state_id or
            not isinstance(report.get('geometry'), dict) or
            not isinstance(report.get('trials'), list) or
            not isinstance(report.get('failedExecutions'), list) or
            not isinstance(report.get('cost'), dict) or
            report.get('status') != 'joint_strategies_compared' or
            report.get('executionRule') !=
                'one far-first left-CUSA sector strategy per source-selected route' or
            report.get('optimizerUpdates') != 0):
        raise ValueError('Joint report source or evidence fields differ')
    access, sectors = report.get('accessComparison'), report.get('sectorComparison')
    if (not isinstance(access, dict) or not isinstance(sectors, dict) or
            access.get('comparisonSeal') is None or
            sectors.get('comparisonSeal') is None or
            sectors.get('sourceAccessComparisonSeal') != access['comparisonSeal']):
        raise ValueError('Joint access/sector comparison seal differs')
    seal = report.get('jointSeal')
    if seal != semantic_digest({key: value for key, value in report.items() if key != 'jointSeal'}):
        raise ValueError('Joint report seal differs')
    if report['failedExecutions'] != [receipt for _, receipt in result['failedBranches']]:
        raise ValueError('Joint sealed failed executions differ from native retention')
    return access, sectors, report


def _install_joint(record, result, access, sectors):
    # The desktop bridge and export CLI share the same checked graph install.
    from resectionlab.surgical_joint_planning import install_joint_comparison
    return install_joint_comparison(record, result, access, sectors)


def _validate_pair(scene, record, original_access, original_sectors):
    """Validate active and archived pairs against the same retained branches."""
    validate_saved_comparison(scene, record, record['planning'])
    validate_saved_sector_comparison(scene, record, record['sectorPlanning'])
    old_view = {**record, 'planning': original_access}
    validate_saved_comparison(scene, old_view, original_access)
    validate_saved_sector_comparison(scene, old_view, original_sectors)
    if (record['sectorPlanningHistory'][-1]['sourceAccessComparison'] != original_access or
            record['sectorPlanningHistory'][-1]['comparison'] != original_sectors or
            record['sectorSourceAccessComparison'] != record['planning']):
        raise ValueError('Classic history or active sector source binding differs')


def _validate_failed_receipts(scene, record, report, start_state_id):
    """Bind every sealed failed execution to its retained native branch."""
    rows = report['failedExecutions']
    ids = [row.get('branchId') for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Joint failed execution branch identities are repeated')
    for row in rows:
        branch_id = row.get('branchId')
        snapshot = record['branches'].get(branch_id)
        if (not isinstance(snapshot, dict) or
                row.get('schema') != 'resectionlab.joint-anatomy-failure.v1' or
                row.get('retentionRole') != 'stopped_native_branch_not_admitted' or
                row.get('startStateId') != start_state_id or
                row.get('finalStateId') != snapshot.get('stateId') or
                row.get('nativeActions') != len(snapshot.get('actions', [])) or
                snapshot.get('sceneId') != scene.identity or
                type(row.get('cleanupComplete')) is not bool or
                type(row.get('cleanupPending')) is not bool or
                row['cleanupComplete'] is row['cleanupPending'] or
                type(row.get('verification', {}).get('exactNativeReplay')) is not bool):
            raise ValueError('Joint failed execution source/branch identity differs')
        validate_modeled_outcome_binding(scene, snapshot,
            row.get('modeledOutcomes'), start_state_id)
    return ids


def _replay_exported(meter, intact, saved, created, *, check=lambda: None):
    """Count every public reopen replay on the planner's original shared meter."""
    if len(created) > MAX_NEW_BRANCHES:
        raise ValueError('Joint export exceeds six retained native branches')
    required = sum(len(saved['branches'][key]['actions']) for key in created)
    if meter.remaining < required:
        raise ValueError('Joint meter lacks reserved calls for every saved branch replay')
    original = meter.wrap(intact)
    replayed = []
    with meter.phase('export-reopen-replay'):
        for branch_id, row in created.items():
            check()
            snapshot = saved['branches'][branch_id]
            candidate = row['candidate']
            if (snapshot['stateId'] != row['stateId'] or
                    semantic_digest(snapshot['actions']) != row['actionsHash'] or
                    len(candidate.history) != len(snapshot['actions'])):
                raise ValueError('Reopened joint branch changed before native replay')
            replay = original.fresh()
            for action, history in zip(snapshot['actions'], candidate.history):
                check()
                result = replay.step(action)
                if (result['status'] != history['status'] or
                        result.get('reason') != history.get('reason') or
                        replay.state_id != history['stateAfter']):
                    raise ValueError('Reopened joint native transition differs')
            if (replay.state_id != snapshot['stateId'] or
                    replay.state != candidate.state or
                    replay.actions != snapshot['actions']):
                raise ValueError('Reopened joint native endpoint differs')
            replayed.append({'branchId': branch_id, 'kind': row['kind'],
                             'stateId': replay.state_id,
                             'nativeActions': len(replay.actions),
                             'actionsHash': row['actionsHash']})
    if meter.report()['byPhase'].get('export-reopen-replay', {}).get('physicalCalls') != required:
        raise ValueError('Joint export replay physical-call inventory differs')
    return replayed


def run(source, output, *, max_native_actions=MAX_ACTIONS,
        max_physical_calls=MAX_CALLS, max_wall_seconds=MAX_SECONDS):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output.exists() or not output.is_relative_to(TESTCASES) or not source.is_file() or
            type(max_native_actions) is not int or max_native_actions != MAX_ACTIONS or
            type(max_physical_calls) is not int or not 352 < max_physical_calls <= MAX_CALLS or
            type(max_wall_seconds) not in (int, float) or
            not 0 < max_wall_seconds <= MAX_SECONDS):
        raise ValueError('Use an existing source and a new ignored testcases/ output within declared limits')
    started = time.monotonic()
    def check():
        if time.monotonic()-started > max_wall_seconds:
            raise TimeoutError('Joint comparison end-to-end wall budget exhausted')
    source_sha = _sha(source)
    output.mkdir(parents=True)
    receipt = {'schema': SCHEMA, 'source': str(source), 'sourceSha256': source_sha,
               'limits': {'maxNativeActions': max_native_actions,
                          'maxPhysicalCalls': max_physical_calls,
                          'maxWallSeconds': max_wall_seconds}, 'status': 'running'}
    try:
        check()
        case, artifacts, workspace = load_workspace(source)
        check()
        scene = load_environment(source)
        check()
        original_record = artifacts.get('surgical_toolbox')
        original = _source_record(original_record, scene)
        check()
        source_identity = _source_identity(case, scene)
        intact = SurgicalEngine.restore(scene,
            original_record['branches'][original_record['branchId']])
        if intact.actions or intact.state_id != original_record['branches'][original_record['branchId']]['stateId']:
            raise ValueError('Source active branch does not restore at intact state')
        from resectionlab.surgical_joint_planning import compare_joint_strategies
        result = compare_joint_strategies(intact, acknowledge_unknowns=True,
            max_native_actions=max_native_actions,
            max_physical_calls=max_physical_calls,
            max_wall_seconds=max_wall_seconds, check=check)
        check()
        access, sectors, report = _joint_documents(result, scene, intact)
        meter = result['meter']
        planning_calls = meter.calls
        if report['cost'].get('physicalCalls') != planning_calls:
            raise ValueError('Sealed joint planning cost differs from shared meter')
        record = copy.deepcopy(original_record)
        created = _install_joint(record, result, access, sectors)
        _validate_failed_receipts(scene, record, report, intact.state_id)
        _validate_pair(scene, record, original_record['planning'],
                       original_record['sectorPlanning'])
        check()
        artifacts = copy.deepcopy(artifacts)
        record['jointPlanning'] = copy.deepcopy(report)
        validate_joint_report(scene, record, record['jointPlanning'])
        check()
        artifacts['surgical_toolbox'] = record
        destination = output/'joint-comparison.ressectionlab'
        check()
        if workspace is None:
            from resectionlab.imaging import save_case
            save_case(case, destination, artifacts=artifacts)
        else:
            save_workspace(case, destination, sources=workspace['sources'],
                imaging=workspace['imagingState'], episode=workspace['episode'],
                selection=workspace['selection'], originals=workspace['originals'],
                artifacts=artifacts)
        check()
        reopened_case, reopened_artifacts, _ = load_workspace(destination)
        check()
        reopened_scene = load_environment(destination)
        check()
        saved = reopened_artifacts['surgical_toolbox']
        if (_source_identity(reopened_case, reopened_scene) != source_identity or
                {key: saved['branches'][key] for key in original_record['branches']} !=
                    original_record['branches'] or
                {key: saved.get('parents', {}).get(key)
                 for key in original_record.get('parents', {})} != original_record.get('parents', {}) or
                len(saved['branches']) != len(original_record['branches'])+len(created) or
                saved.get('jointPlanning') != report):
            raise ValueError('Reopened joint case, classic branch inventory or report differs')
        _validate_pair(reopened_scene, saved, original_record['planning'],
                       original_record['sectorPlanning'])
        _validate_failed_receipts(reopened_scene, saved, saved['jointPlanning'],
                                  intact.state_id)
        validate_joint_report(reopened_scene, saved, saved['jointPlanning'])
        check()
        replayed = _replay_exported(meter, intact, saved, created, check=check)
        check()
        metering = meter.report()
        if (metering['physicalCalls'] > max_physical_calls or
                metering['physicalCalls'] != planning_calls+sum(row['nativeActions'] for row in replayed)):
            raise ValueError('Joint comparison total native call accounting differs')
        if _sha(source) != source_sha:
            raise ValueError('Original joint source bytes changed')
        check()
        saved_sha = _sha(destination)
        check()
        receipt.update(status='saved', savedWorkspace=str(destination),
            savedSha256=saved_sha, sourcePreserved=True,
            sourceIdentity=source_identity, originalSource=original,
            jointSeal=report['jointSeal'], jointStatus=report['status'],
            accessComparisonSeal=access['comparisonSeal'],
            sectorComparisonSeal=sectors['comparisonSeal'],
            newBranches=replayed,
            nativePlanningWallSeconds=report['wallSeconds'],
            nativePlanningPhysicalCalls=planning_calls,
            nativeExportReplayPhysicalCalls=sum(row['nativeActions'] for row in replayed),
            totalNativePhysicalCalls=metering['physicalCalls'],
            physicalCallInventory=metering)
    except Exception as error:
        receipt.update(status='failed', errorType=type(error).__name__,
                       error=str(error), sourcePreserved=_sha(source) == source_sha)
        raise
    finally:
        receipt['wallSeconds'] = time.monotonic()-started
        receipt['endToEndWallSeconds'] = receipt['wallSeconds']
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2,
            allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-native-actions', type=int, default=MAX_ACTIONS)
    parser.add_argument('--max-physical-calls', type=int, default=MAX_CALLS)
    parser.add_argument('--max-wall-seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args()
    result = run(args.case, args.output, max_native_actions=args.max_native_actions,
                 max_physical_calls=args.max_physical_calls,
                 max_wall_seconds=args.max_wall_seconds)
    print(json.dumps({'status': result['status'], 'savedWorkspace': result['savedWorkspace'],
        'savedSha256': result['savedSha256'],
        'totalNativePhysicalCalls': result['totalNativePhysicalCalls']},
        sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
