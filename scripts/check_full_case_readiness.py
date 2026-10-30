#!/usr/bin/env python3
"""Bounded source-backed bridge readiness, protection and cold-replay check.

Reuses the three prepared local cases. No downloads, full episodes, planning,
policy execution, fitting or training. All generated evidence stays in testcases.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import grid_binding, load_protection
from resectionlab.workspace_bundle import load_workspace

SCHEMA = 'resectionlab.full-case-readiness-source-check.v1'
CASES = ('meningioma-sub-06', 'UCSF-PDGM-0004', 'meningioma-sub-13')


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_arrays(case):
    return {'image': array_digest(case.mri), 'affine': array_digest(case.affine),
            'compartments': {k: array_digest(v) for k, v in case.compartments.items()},
            'sourceCompartments': {k: array_digest(v) for k, v in case.source_compartments.items()},
            'sourceRefs': semantic_digest([ref.to_dict() for ref in case.source_refs])}


class Check:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def call(self, session, operation, values):
        start = time.perf_counter()
        row = {'operation': operation, 'command': values.get('command')}
        try:
            result = session.execute(operation, values, _Request(operation), lambda *_: None)
            row['status'] = 'returned'
            return result
        except Exception as error:
            row.update(status='refused', errorType=type(error).__name__, message=str(error),
                       code=getattr(error, 'code', None))
            raise
        finally:
            row['seconds'] = time.perf_counter() - start
            self.calls.append(row)
            print(json.dumps(row), flush=True)

    def surgical(self, session, key, command, current=None, **values):
        body = {'caseHash': key, 'command': command, **values}
        if current is not None:
            body['expectedStateId'] = current['view']['stateId']
        return self.call(session, 'surgicalRehearsal', body)

    def refusal(self, session, key, current, **values):
        entry = session.cases[key]
        before = entry.artifacts
        before_digest = semantic_digest(thaw_json(before))
        cached = session.surgical_sessions[key]
        engine_id = cached[1].state_id
        try:
            self.surgical(session, key, values.pop('command', 'attachProtection'), current, **values)
        except (ValueError, RuntimeError) as error:
            require(entry.artifacts is before, 'Refused operation replaced source artifacts')
            require(semantic_digest(thaw_json(entry.artifacts)) == before_digest,
                    'Refused operation mutated source artifacts')
            require(session.surgical_sessions[key] is cached and cached[1].state_id == engine_id,
                    'Refused operation changed the admitted engine')
            return {'errorType': type(error).__name__, 'message': str(error),
                    'code': getattr(error, 'code', None), 'artifactsAndEngineUnchanged': True}
        raise AssertionError('Expected source/protection refusal')

    def case(self, name):
        output = self.output / name
        output.mkdir()
        path = ROOT / 'testcases/cases' / (name + '-standard-v2.ressectionlab')
        protection_path = ROOT / 'testcases/environments' / name / 'rl-protection.json'
        input_hashes = {str(p): sha(p) for p in (path, protection_path, protection_path.with_suffix('.npz'))}
        scene = load_environment(path)
        protection = load_protection(scene, protection_path, strict=True)
        inventory = {'case': str(path), 'sceneId': scene.identity, 'sourceGridBinding': grid_binding(scene),
                     'shape': list(scene.shape), 'regions': scene.regions,
                     'explicitProtection': str(protection_path), 'structures': len(protection.annotations),
                     'protectedSourceCells': len(protection.cells), 'missingRequired': list(protection.missing),
                     'protectionReview': sorted({a.review_status for a in protection.annotations}),
                     'inputHashes': input_hashes, 'sourceReuse': 'Existing local files reused without copies or downloads'}
        del protection, scene
        session = BridgeSession(output / 'transfers', max_cases=1)
        try:
            key = self.call(session, 'loadCase', {'path': str(path)})['caseHash']
            original_arrays = source_arrays(session.cases[key].case)
            current = self.surgical(session, key, 'inspect')
            if current is None:
                current = self.surgical(session, key, 'create')
            require('fullCaseReadiness' in current, 'Bridge response omitted readiness')
            original = copy.deepcopy(thaw_json(session.cases[key].artifacts)['surgical_toolbox'])
            original_branch = current['branchId']
            before_readiness = current['fullCaseReadiness']
            require(before_readiness['status'] == 'blocked'
                    and 'attachProtection' in before_readiness['availableActions']
                    and {'primary', 'vascular'}.isdisjoint(before_readiness['availableActions']),
                    'Unbound source did not expose the expected protection recovery')
            require(any(row['code'] == 'missing-protection' for row in before_readiness['blockers']),
                    'Unbound source omitted the missing-protection blocker')
            wrong_name = 'UCSF-PDGM-0004' if name == 'meningioma-sub-06' else 'meningioma-sub-06'
            wrong = ROOT / 'testcases/environments' / wrong_name / 'rl-protection.json'
            refusals = {
                'missingEvidenceFullCase': self.refusal(session, key, current,
                    command='fullCase', scenario='primary', maxActions=1),
                'missingEvidence': self.refusal(session, key, current,
                    protectionPath=str(output / 'missing-protection.json')),
                'incompatibleSource': self.refusal(session, key, current, protectionPath=str(wrong))}
            current = self.surgical(session, key, 'attachProtection', current,
                                    protectionPath=str(protection_path))
            attached = current['fullCaseReadiness']
            record = thaw_json(session.cases[key].artifacts)['surgical_toolbox']
            require(record['branches'][original_branch] == original['branches'][original_branch],
                    'Protection attachment modified the original branch')
            require(current['branchId'] != original_branch, 'Protection attachment did not preserve a separate branch')
            strict = record['branches'][current['branchId']]['protectionContract']
            require(strict['config']['require_critical_coverage'] is True, 'Attachment did not bind strict coverage')
            require(strict['sourceBinding'] == inventory['sourceGridBinding'], 'Attachment changed source binding')
            result = {'inventory': inventory, 'caseHash': key, 'beforeAttachment': before_readiness,
                      'afterAttachment': attached, 'refusals': refusals,
                      'originalBranchPreserved': True, 'strictContractHash': strict['contentHash']}
            if name == 'meningioma-sub-13':
                require('primary' not in attached['availableActions'] and 'vascular' not in attached['availableActions'],
                        'Missing-target source admitted full rehearsal')
                result['missingTargetRefusal'] = self.refusal(session, key, current,
                    command='fullCase', scenario='primary', maxActions=1)
            else:
                require({'primary', 'vascular'} <= set(attached['availableActions']),
                        'Matching source evidence did not admit supported research scenarios')
                current = self.surgical(session, key, 'fullCase', current, scenario='primary', maxActions=1)
                require(current['position'] <= 1, 'Bounded slice exceeded one committed action')
                result['slice'] = {'scenario': 'primary', 'maxActions': 1, 'position': current['position'],
                                   'branchId': current['branchId'], 'stateId': current['view']['stateId'],
                                   'director': current['fullCase'], 'readiness': current['fullCaseReadiness']}
                result['continuedAttachmentRefusal'] = self.refusal(session, key, current,
                    protectionPath=str(protection_path)) if current['position'] else None
            record = copy.deepcopy(thaw_json(session.cases[key].artifacts)['surgical_toolbox'])
            saved = output / (name + '-bounded-readiness.ressectionlab')
            self.call(session, 'saveCase', {'caseHash': key, 'path': str(saved)})
            expected_state = current['view']['stateId']
            branch = current['branchId']
            expected_readiness = current['fullCaseReadiness']
        finally:
            session.close()
            del session
            gc.collect()
        saved_case, saved_artifacts, _ = load_workspace(saved)
        require(source_arrays(saved_case) == original_arrays, 'Save changed patient source arrays/references')
        require(saved_case.semantic_hash == key, 'Save changed patient semantic identity')
        require(saved_artifacts['surgical_toolbox'] == record, 'Save changed branch/director/protection records')
        del saved_case, saved_artifacts
        gc.collect()
        # A fresh cache directory plus a fresh BridgeSession forces independent
        # public-action reconstruction, including contract and checkpoint checks.
        from resectionlab import surgical_verified_cache
        old_cache = surgical_verified_cache.DEFAULT_DIRECTORY
        surgical_verified_cache.DEFAULT_DIRECTORY = output / 'cold-verified-cache'
        cold = BridgeSession(output / 'cold-transfers', max_cases=1)
        replayed = []
        step_code = SurgicalEngine.step.__code__
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code is step_code:
                replayed.append(copy.deepcopy(frame.f_locals['action']))
        require(sys.getprofile() is None, 'Run checker without an existing Python profiler')
        try:
            cold_key = self.call(cold, 'loadCase', {'path': str(saved)})['caseHash']
            sys.setprofile(profile)
            try:
                reopened = self.surgical(cold, cold_key, 'inspect', branchId=branch)
            finally:
                sys.setprofile(None)
            require(cold_key == key and reopened['view']['stateId'] == expected_state,
                    'Cold canonical inspect changed source/state identity')
            require(source_arrays(cold.cases[cold_key].case) == original_arrays,
                    'Cold load changed patient source arrays/references')
            restored = thaw_json(cold.cases[cold_key].artifacts)['surgical_toolbox']
            snapshot = record['branches'][branch]
            require(restored['branches'][branch] == snapshot, 'Cold canonical inspect changed saved snapshot')
            require(restored['branches'][branch]['protectionContract'] == strict,
                    'Cold canonical inspect changed strict protection contract')
            require(reopened['fullCaseReadiness'] == expected_readiness, 'Cold readiness differs')
            require(replayed == snapshot['actions'], 'Cold inspect did not canonically replay exactly the saved actions')
            result['saveAndColdReopen'] = {'path': str(saved), 'sha256': sha(saved),
                'stateId': expected_state, 'branchId': branch, 'canonicalPublicActionsReplayed': len(replayed),
                'sourceArraysAndReferencesEqual': True, 'strictContractEqual': True,
                'actionLogAndCheckpointEqual': True, 'readinessEqual': True}
        finally:
            sys.setprofile(None)
            cold.close()
            surgical_verified_cache.DEFAULT_DIRECTORY = old_cache
        require(all(sha(Path(p)) == digest for p, digest in input_hashes.items()),
                'Verification changed original source/evidence bytes')
        result['originalInputsUnchanged'] = True
        write_json(output / 'receipt.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'testcases/outputs/full-case-readiness-20261030')
    parser.add_argument('--case', action='append', choices=CASES, help='Defaults to all three prepared cases')
    parser.add_argument('--max-wall-seconds', type=int, default=300)
    args = parser.parse_args()
    require(1 <= args.max_wall_seconds <= 900, 'Wall budget must be within 1..900 seconds')
    output = args.output.expanduser().resolve()
    require(output.is_relative_to((ROOT / 'testcases').resolve()), 'Generated app evidence must stay under testcases/')
    require(not output.exists(), 'Output already exists; preserve prior provenance and choose a distinct --output')
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'schema': SCHEMA, 'createdUtc': datetime.now(timezone.utc).isoformat(),
              'noTraining': True, 'optimizerUpdates': 0, 'downloads': 0, 'policyExecutions': 0,
              'fullEpisodes': 0, 'maxActionsPerSourceSlice': 1, 'maxWallSeconds': args.max_wall_seconds,
              'output': str(output), 'cases': [], 'checkerSha256': sha(Path(__file__)),
              'implementationHashes': {str(p.relative_to(ROOT)): sha(p) for p in
                  sorted((ROOT / 'src/resectionlab').glob('*.py'))},
              'scope': 'Source-bound research integration only; no completed resection, physiological validation or clinical qualification.'}
    check = Check(output)
    def timeout(*_):
        raise TimeoutError('Bounded source integration wall-time budget exhausted')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
    try:
        for name in dict.fromkeys(args.case or CASES):
            report['cases'].append(check.case(name))
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', errorType=type(error).__name__, message=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        report.update(wallSeconds=time.perf_counter() - started, calls=check.calls)
        write_json(output / 'receipt.json', report)
        print(json.dumps({'status': report['status'], 'receipt': str(output / 'receipt.json'),
                          'wallSeconds': report['wallSeconds']}), flush=True)


if __name__ == '__main__':
    main()
