#!/usr/bin/env python3
"""Reproduce one source-bound planning interaction through the desktop bridge.

Reuses the local acquired case and its recorded exposure. No image acquisition,
training, physiology, additional tissue removal or continuum fitting is run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.workspace_bundle import load_workspace
from verify_interactive_geometry import verify as independent_geometry, verify_constraint_change_and_refusal

SOURCE = ROOT / 'testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(case):
    return {'mri': array_digest(case.mri),
            'compartments': {key: array_digest(value) for key, value in case.compartments.items()}}


def check(output, source=SOURCE):
    output, source = output.resolve(), source.resolve()
    require(output.is_relative_to(ROOT/'testcases'), 'Generated app data belongs under testcases/')
    require(not output.exists(), 'Preserve previous evidence; choose a new output directory')
    output.mkdir(parents=True)
    tick = time.perf_counter()
    report = {'schema': 'resectionlab.interactive-planning-integration.v1', 'source': str(source),
              'output': str(output), 'calls': [], 'checks': [], 'generations': [], 'passed': False}
    session = None
    try:
        original_hash = _sha(source)
        case, artifacts, _ = load_workspace(source)
        original = artifacts['surgical_toolbox']
        original_branches = copy.deepcopy(original['branches'])
        original_arrays = _arrays(case)
        report.update(sourceSha256=original_hash, caseHash=case.semantic_hash)
        session = BridgeSession(output/'transfers')

        def call(operation, **args):
            started = time.perf_counter()
            row = {'operation': operation, 'command': args.get('command')}
            try:
                result = session.execute(operation, args, _Request(operation), lambda *_: None)
                return result
            except Exception as error:
                row['error'] = str(error)
                raise
            finally:
                row['seconds'] = time.perf_counter()-started
                report['calls'].append(row)
                print(json.dumps(row), flush=True)

        loaded = call('loadCase', path=str(source))
        key = loaded['caseHash']
        require(key == case.semantic_hash, 'Source case identity changed')
        state = call('surgicalRehearsal', caseHash=key, command='inspect', branchId='original-exposure')
        require(state['position'] == 4, 'Use the recorded four-action exposure checkpoint')
        baseline = state['view']['stateId']
        defaults = state['interactivePlanningAvailability']['defaults']
        require(state['interactivePlanningAvailability']['available'], 'Documented post-exposure source is unavailable')

        def record():
            return thaw_json(session._get_case(key).artifacts)['surgical_toolbox']

        def preserved():
            now = record()
            require(all(now['branches'][name] == value for name, value in original_branches.items()),
                    'An original native branch was modified')
            require(_sha(source) == original_hash, 'Original acquired workspace bytes changed')

        def generate(name, constraints):
            nonlocal state
            state = call('surgicalRehearsal', caseHash=key, command='generateApproaches',
                expectedStateId=state['view']['stateId'], constraints=constraints, acknowledgeResearchUnknowns=True)
            generated = state['interactivePlanning']
            require(generated['current'] and generated['startStateId'] == baseline, 'Generated plan is not current at its bound checkpoint')
            require(state['view']['stateId'] == baseline, 'Planning changed the live physical state')
            (output/(name+'.json')).write_text(json.dumps(generated, indent=2, allow_nan=False)+'\n')
            report['generations'].append({'name': name, 'reportSeal': generated['reportSeal'],
                'status': generated['status'], 'routeCount': len(generated['routes']),
                'constraints': generated['constraints'], 'rejections': generated['rejections']})
            preserved()
            return generated

        base = generate('01-base-approaches', defaults)
        require(len(base['routes']) >= 2, 'At least two genuinely distinct approaches are required on this source')
        started = time.perf_counter()
        independent = independent_geometry(base, source, branch='original-exposure')
        independent['wallSeconds'] = time.perf_counter()-started
        require(independent['passed'], 'Independent continuous full-tool geometry check failed')
        (output/'independent-geometry.json').write_text(json.dumps(independent, indent=2, allow_nan=False)+'\n')
        report['independent'] = independent
        report['checks'].append('New routes pass a separately authored continuous whole-shaft/working-volume and withdrawal checker')

        # Every displayed route must be executable through unchanged public
        # actions, including its separately checked withdrawal.
        report['previews'] = []
        for route in base['routes']:
            before = semantic_digest(record())
            preview = call('surgicalRehearsal', caseHash=key, command='previewApproach',
                expectedStateId=baseline, expectedPlanSeal=base['reportSeal'], candidateId=route['id'])
            require(preview['view']['stateId'] == baseline and preview['changed'] is False, 'Preview committed its candidate')
            require(semantic_digest(record()) == before, 'Preview changed saved workspace evidence')
            receipt = preview['approachPreview']['receipt']
            require(receipt['exactPhysicsPreserved'], 'Preview changed native physics')
            report['previews'].append(receipt)
        report['checks'].append('All retained candidates execute exact public insertion and withdrawal; previews preserve live state/artifacts')

        selected = base['routes'][0]
        state = call('surgicalRehearsal', caseHash=key, command='applyApproach', expectedStateId=baseline,
            expectedPlanSeal=base['reportSeal'], candidateId=selected['id'])
        applied_branch = state['branchId']
        applied_snapshot = copy.deepcopy(record()['branches'][applied_branch])
        require(not state['interactivePlanning']['current'], 'Installed approach must invalidate the baseline result')
        require(state['view']['stateId'] == report['previews'][0]['approachStateId'], 'Apply differs from exact preview')
        state = call('surgicalRehearsal', caseHash=key, command='step', expectedStateId=state['view']['stateId'],
            action={'slot': 'left', 'operation': 'inspect', 'duration': .1})
        require(state['branchId'] != applied_branch and record()['branches'][applied_branch] == applied_snapshot,
                'Editing an installed plan did not preserve its checked branch')
        # Return through the normal canonical branch selector, retaining the
        # applied branch and its editable child.
        state = call('surgicalRehearsal', caseHash=key, command='selectBranch', expectedStateId=state['view']['stateId'],
            branchId=base['sourceBranchId'])
        require(state['view']['stateId'] == baseline and state['interactivePlanning']['current'], 'Exact source checkpoint did not revalidate')
        report['checks'].append('Install creates a normal replayable branch; later edits fork and exact baseline return revalidates')

        narrow = copy.deepcopy(defaults)
        narrow['entry']['centerMm'] = copy.deepcopy(base['routes'][-1]['entryPose']['position'])
        narrow['entry']['radiusMm'] = .5
        changed = generate('02-restricted-entry', narrow)
        require(len(changed['routes']) < len(base['routes']), 'Restricting the entry patch did not meaningfully change eligibility')
        require(changed['rejections'] or changed['status'] != base['status'], 'Eligibility change lacks an explanation')
        report['checks'].append('A restricted entry patch reduces admissible distinct routes and retains explicit exclusion reasons')

        protected = copy.deepcopy(defaults)
        protected['nonTargetClearanceMm'] = .5
        blocked = generate('03-mapped-tissue-clearance', protected)
        require(not blocked['routes'] and any(row['reason'] == 'NON_TARGET_CLEARANCE' for row in blocked['rejections']),
                'Mapped non-target protection margin did not explain the blocked approaches')
        report['checks'].append('A source-supported 0.5 mm non-target margin refuses the default candidates with explicit geometric conflicts')

        missing = copy.deepcopy(defaults)
        missing['requiredEvidence'] = ['functional']
        refused = generate('04-required-unknown-functional', missing)
        require(not refused['routes'], 'Missing required functional evidence produced an apparent recommendation')
        require(refused['rejections'], 'Evidence refusal needs an explicit explanation')
        report['checks'].append('Required unavailable functional evidence prevents a recommendation')
        report['independentDecision'] = verify_constraint_change_and_refusal(base, changed, refused)
        require(report['independentDecision']['passed'], 'Independent constraint/refusal audit differs')

        # Superseded report and stale checkpoint calls must fail atomically.
        for changes in [{'expectedPlanSeal': base['reportSeal']}, {'expectedStateId': 'sha256:'+'0'*64}]:
            before = semantic_digest(record())
            args = {'caseHash': key, 'command': 'applyApproach', 'expectedStateId': baseline,
                    'expectedPlanSeal': refused['reportSeal'], 'candidateId': selected['id'], **changes}
            try:
                call('surgicalRehearsal', **args)
            except ValueError as error:
                report.setdefault('rejections', []).append(str(error))
            else:
                raise AssertionError('Stale/superseded approach request was admitted')
            require(before == semantic_digest(record()), 'Rejected request mutated the archive')

        ready = generate('05-demo-ready', defaults)
        saved = output/'interactive-planning.ressectionlab'
        call('saveCase', caseHash=key, path=str(saved), overwrite=False)
        saved_case, saved_artifacts, _ = load_workspace(saved)
        require(_arrays(saved_case) == original_arrays, 'Save altered acquired MRI or source segmentations')
        saved_record = copy.deepcopy(saved_artifacts['surgical_toolbox'])
        session.close()
        session = BridgeSession(output/'cold-transfers')
        reopened = call('loadCase', path=str(saved))
        require(reopened['caseHash'] == key, 'Cold reopened source identity changed')
        state = call('surgicalRehearsal', caseHash=key, command='inspect')
        require(state['interactivePlanning'] == ready and state['view']['stateId'] == baseline,
                'Cold canonical reopen did not preserve the current report and checkpoint')
        require(record()['interactiveApproaches'] == saved_record['interactiveApproaches'], 'Saved revisions changed')
        state = call('surgicalRehearsal', caseHash=key, command='selectBranch',
            expectedStateId=state['view']['stateId'], branchId=applied_branch)
        require(state['view']['stateId'] == applied_snapshot['stateId'], 'Installed approach did not replay exactly after cold reopen')
        require(not state['interactivePlanning']['current'], 'Historical plan unexpectedly became current during replay')
        preserved()
        report['checks'].append('Native save/cold canonical reopen preserve acquired arrays, all report revisions and exact installed-branch replay')
        report.update(saved=str(saved), sourcePreserved=True, passed=True,
            additionalTissueRemovalMm3=0., continuumCandidateSolves=0, optimizerUpdates=0,
            limits=['Authored post-exposure cavity; historical 4,940 mm³ non-target removal remains.',
                    'Inactive CUSA boundary approach only; no target removal is performed.',
                    'Missing skull, vessels, function and shift remain unknown; no clinical safety admission.'])
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        if session is not None:
            session.close()
        report['wallSeconds'] = time.perf_counter()-tick
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.output, args.source)
    print(json.dumps({'passed': result['passed'], 'saved': result['saved'], 'wallSeconds': result['wallSeconds']}))
