#!/usr/bin/env python3
"""Source-backed policy-editor integration over retained checked vectors only.

Uses the public BridgeSession command, native save and a fresh session reopen.
It does not regenerate the three-world comparison, plan or execute new actions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.simulator_qualification import file_sha256
from resectionlab.surgical_consequences import compare_samples
from resectionlab.workspace_bundle import load_workspace

SOURCE = ROOT / 'testcases/cases/UCSF-PDGM-0007-consequence-comparison.ressectionlab'
DEFAULT_OUTPUT = ROOT / 'testcases/outputs/consequence-editor-20261030/bridge'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_arrays(case):
    return {'mri': array_digest(case.mri),
            'compartments': {k: array_digest(v) for k, v in case.compartments.items()}}


def check(output, source=SOURCE):
    output, source = output.resolve(), source.resolve()
    require(output.is_relative_to(ROOT / 'testcases'), 'Evidence belongs under ignored testcases/')
    require(not output.exists(), 'Preserve previous evidence; use a new output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'schema': 'resectionlab.consequence-editor-integration.v1', 'input': str(source),
              'output': str(output), 'calls': [], 'checks': [], 'revisions': [], 'passed': False,
              'newPhysicalActions': 0, 'optimizerUpdates': 0, 'newAcquiredDatasetBytes': 0}
    session = None
    try:
        original_bytes = file_sha256(source)
        original_case, artifacts, _ = load_workspace(source)
        arrays = source_arrays(original_case)
        original = copy.deepcopy(artifacts['surgical_toolbox'])
        comparison = original['consequenceComparison']
        frozen = {k: copy.deepcopy(original[k]) for k in ('consequenceComparison', 'branches')}
        baseline_id = comparison['baselineBranchId']
        baseline = original['branches'][baseline_id]
        require(baseline_id == 'retraction-start' and len(baseline['actions']) == 15,
                'Use the retained complete common checkpoint')
        require(len(comparison['samples']) == 9 and len(comparison['scenarios']) == 3,
                'Retain exactly the original three-world comparison')
        report['sourceSha256'] = original_bytes
        report['caseHash'] = original_case.semantic_hash
        report['comparisonSeal'] = comparison['comparisonSeal']
        report['frozenRecordHash'] = semantic_digest(frozen)
        report['baseline'] = {'branchId': baseline_id, 'position': 15, 'stateId': baseline['stateId']}
        report['samples'] = [{'branchId': row['branchId'], 'scenarioId': row['scenarioId'],
                              'additionalTargetMm3': row['vector']['dimensions']['targetRemovedMm3']['incremental'],
                              'cumulativeNonTargetMm3': row['vector']['dimensions']['nonTargetRemovedMm3']['cumulative'],
                              'residualTargetMm3': row['vector']['residualTargetMm3']}
                             for row in comparison['samples']]
        session = BridgeSession(output / 'transfers')

        def call(operation, **args):
            tick = time.perf_counter()
            try:
                result = session.execute(operation, args, _Request(operation), lambda *_: None)
            except Exception as error:
                report['calls'].append({'operation': operation, 'command': args.get('command'),
                                        'seconds': time.perf_counter()-tick, 'error': str(error)})
                raise
            report['calls'].append({'operation': operation, 'command': args.get('command'),
                                    'seconds': time.perf_counter()-tick})
            print(json.dumps(report['calls'][-1]), flush=True)
            return result

        loaded = call('loadCase', path=str(source))
        key = loaded['caseHash']
        require(key == original_case.semantic_hash, 'Native source identity differs')
        state = call('surgicalRehearsal', caseHash=key, command='inspect', branchId=baseline_id)
        require(state['view']['stateId'] == baseline['stateId'], 'Canonical baseline replay differs')

        def record():
            return thaw_json(session._get_case(key).artifacts)['surgical_toolbox']

        def unchanged():
            now = record()
            require(all(now[k] == value for k, value in frozen.items()),
                    'Policy comparison or replay changed frozen comparison/declaration/native snapshots')
            require(file_sha256(source) == original_bytes, 'Source archive bytes changed')

        def reject_without_mutation(**changes):
            before = semantic_digest(record())
            request = {'caseHash': key, 'command': 'reevaluateConsequences',
                       'expectedStateId': state['view']['stateId'],
                       'expectedComparisonSeal': comparison['comparisonSeal'], 'policyInput': base_policy}
            request.update(changes)
            try:
                call('surgicalRehearsal', **request)
            except Exception as error:
                report.setdefault('rejections', []).append({'changes': changes, 'error': str(error)})
            else:
                raise AssertionError('Invalid/stale request was accepted')
            require(semantic_digest(record()) == before, 'Rejected request mutated saved artifacts')
            inspected = call('surgicalRehearsal', caseHash=key, command='inspect')
            require(inspected['view']['stateId'] == state['view']['stateId'], 'Rejected request changed engine state')
            unchanged()

        base_policy = {'minimumAdditionalTargetMm3': 8., 'maximumCumulativeNonTargetMm3': 4940.,
                       'requiredMetrics': [], 'maxBrainStrainSquaredMm3S': None,
                       'maxBrainStressSquaredPa2Mm3S': None}
        require(state['consequenceAnalysis']['available'], 'Exact common checkpoint should allow assessment')
        reject_without_mutation(expectedStateId='sha256:'+'0'*64)
        reject_without_mutation(expectedComparisonSeal='sha256:'+'0'*64)
        for name, overrides, feasible in [
            ('admit-two-continuations', {}, ['short-hold', 'sustained-hold']),
            ('minimum-nine-excludes-all', {'minimumAdditionalTargetMm3': 9.}, []),
            ('whole-access-budget-excludes-all', {'maximumCumulativeNonTargetMm3': 4939.}, []),
            ('unknown-motor-selects-stop', {'requiredMetrics': ['motorEncounterMm3']}, []),
        ]:
            policy = {**base_policy, **overrides}
            prior = copy.deepcopy(record().get('consequenceAnalyses', []))
            state = call('surgicalRehearsal', caseHash=key, command='reevaluateConsequences',
                         expectedStateId=state['view']['stateId'],
                         expectedComparisonSeal=comparison['comparisonSeal'], policyInput=policy)
            analysis = state['consequenceAnalysis']
            latest = analysis['latest']; decision = latest['decision']
            require(analysis['available'] and analysis['current'], 'Assessment is not current at common checkpoint')
            require(latest['policyInput'] == policy, 'Editor policy changed')
            expected = compare_samples(comparison['samples'], latest['policy'],
                stop_branch=comparison['stopBranchId'],
                scenario_ids=[s['id'] for s in comparison['scenarios']],
                branch_ids=comparison['alternativeBranchIds'])
            require(decision == expected, 'Public decision differs from shared comparison algebra')
            require(sorted(decision['feasibleContinuations']) == sorted(feasible), name+' has incorrect eligibility')
            require(latest['policy']['objectives'] == ['targetRemovedMm3','nonTargetRemovedMm3','elapsedSeconds'],
                    'Editor objectives changed or introduced scalar weights')
            require(decision['selectedBranchId'] == ('short-hold' if feasible else comparison['stopBranchId']),
                    name+' selects unexpected branch')
            require(decision['taskSatisfied'] == bool(feasible), 'STOP must retain unsatisfied target task')
            require(decision['stopExecutable'], 'STOP cleanup is not executable')
            require(state['view']['stateId'] == baseline['stateId'] and state['branchId'] == baseline_id,
                    'Policy assessment executed a new action or selected a different branch')
            revisions = record()['consequenceAnalyses']
            require(revisions[:-1] == prior and revisions[-1] == latest,
                    'New revision changed earlier analysis history')
            require(latest['revision'] == len(revisions) and analysis['revisionCount'] == len(revisions),
                    'Analysis revision sequence differs')
            unchanged()
            report['revisions'].append({'name': name, 'analysis': latest})
        report['checks'].append('Policy edits preserve source bytes, original declaration/comparison and all six native branches')
        unknown = state['consequenceAnalysis']['latest']['decision']
        require(all(any(v.get('metric') == 'motorEncounterMm3' and v['reason'] == 'unknown_required_consequence'
                        for v in unknown['violations'][b]) for b in comparison['alternativeBranchIds']),
                'Unknown required motor evidence was treated as zero')
        saved = output / 'policy-at-common-checkpoint.ressectionlab'
        call('saveCase', caseHash=key, path=str(saved), overwrite=False)
        saved_case, saved_artifacts, _ = load_workspace(saved)
        require(source_arrays(saved_case) == arrays, 'Native save changed MRI or source segmentation arrays')
        saved_record = copy.deepcopy(saved_artifacts['surgical_toolbox'])
        require(all(saved_record[k] == value for k, value in frozen.items()), 'Native save changed original programs')
        saved_analysis = copy.deepcopy(state['consequenceAnalysis'])
        session.close(); session = BridgeSession(output / 'cold-transfers')
        reopened = call('loadCase', path=str(saved)); require(reopened['caseHash'] == key, 'Cold source identity differs')
        state = call('surgicalRehearsal', caseHash=key, command='inspect')
        require(state['consequenceAnalysis'] == saved_analysis, 'Cold replay did not reproduce saved policy, bindings and decision')
        require(record()['consequenceAnalyses'] == saved_record['consequenceAnalyses'], 'Cold reopen changed analysis history')
        report['checks'].append('Native save and fresh BridgeSession canonical reopen reproduce all four sealed analysis revisions and current decision')
        for branch in comparison['alternativeBranchIds']:
            state = call('surgicalRehearsal', caseHash=key, command='selectBranch', branchId=branch,
                         expectedStateId=state['view']['stateId'])
            expected = original['branches'][branch]
            require(state['view']['stateId'] == expected['stateId'] and state['position'] == len(expected['actions']),
                    'Normal branch selection did not open exact canonical replay endpoint')
            analysis = state['consequenceAnalysis']
            require(not analysis['available'] and not analysis['current'] and analysis['latest'] == saved_analysis['latest'],
                    'Endpoint replay silently rebound or removed the baseline assessment')
            unchanged()
        reject_without_mutation()
        report['checks'].append('All three alternative endpoints open through normal verified replay; saved baseline analysis stays historical')
        report['saved'] = str(saved)
        report['sourcePreserved'] = file_sha256(source) == original_bytes
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        if session is not None:
            session.close()
        report['wallSeconds'] = time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': report['passed'], 'report': str(output/'report.json'),
                      'wallSeconds': report['wallSeconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--case', type=Path, default=SOURCE)
    args = parser.parse_args()
    check(args.output, args.case)
