"""Ordinary source-qualified planning must charge kerf cleanup and preserve the dirty outcome."""
from resectionlab.surgical_cranial_planning import compare_cranial_strategies
from resectionlab.surgical_scene import from_case
from scripts.demonstrate_generated_kerf_cleanup import DIRTY_STATE, square_source


def test_same_square_can_finish_or_stop_before_access_at_the_shorter_horizon():
    case, record = square_source()
    base = from_case(case, record['mapping'])
    short = compare_cranial_strategies(base, record, planning_horizon=3)['report']
    stopped = next(row for row in short['alternatives'] if row['branchId'] == short['selectedBranchId'])
    assert short['decision'] == 'STOP' and stopped['nativeActions'] == 0
    assert stopped['endpointAssessment']['classification'] == 'valid-stop-with-residual'
    assert stopped['goalAssessment']['residualOriginalTargetMm3'] == 27.
    assert stopped['ledger']['wholeNonTargetRemovedMm3'] == stopped['ledger']['modeledElapsedSeconds'] == 0.
    assert short['cost']['physicalCalls'] > 0

    complete = compare_cranial_strategies(base, record)['report']
    assert complete['decision'] == 'COMPLETED_FROZEN_TASK'
    selected = next(p for p in complete['alternatives'] if p['branchId'] == complete['selectedBranchId'])
    actual = selected['endpointAssessment']
    assert actual['completedTask'] and actual['validWithdrawal'] and actual['completeOriginalTargetRemoval']
    assert not actual['pending'] and actual['debrisMm3'] == actual['fluidMm3'] == 0.
    assert selected['ledger']['wholeNonTargetRemovedMm3'] == 334.
    assert selected['ledger']['workNonTargetRemovedMm3'] == selected['ledger']['unauthorizedRemovedSourceCells'] == 0
    assert selected['nativeHistory'][-1]['action']['operation'] == 'withdraw'
    dirty = next(p for p in complete['selectionProofs'] if p['snapshot']['stateId'] == DIRTY_STATE)
    assert len(dirty['nativeHistory']) == 119
    candidate = next(c for c in complete['selectionEvidence']['candidates'] if c['finalStateId'] == DIRTY_STATE)
    assert not candidate['eligible'] and candidate['endpointAssessment']['pending'] == ['debris']
    assert candidate['endpointAssessment']['debrisMm3'] == 236.
    assert candidate['endpointAssessment']['progress']['originalTargetTreatmentMm3'] == 27.
    assert record['position'] == 0 and set(record['branches']) == {'main'}
