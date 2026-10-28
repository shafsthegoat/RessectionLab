"""Native precision entry and original-baseline continuation contracts."""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_sector_planning import _sector_cells
from resectionlab.surgical_staged_planning import assess_task, larger_task, source_zones
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter
from resectionlab.surgical_precision_reentry import (
    continue_precision_reentry, execute_precision_reentry, install_precision_continuation,
    propose_precision_reentries, validate_precision_report)


def coupon():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[10, 10, 7:15] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 2, 'name': 'generated enhancing target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated narrow precision entry coupon'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    intact = engine.state_id
    assert engine.step({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2})['status'] == 'accepted'
    pose = lambda z: {'position': [10., 10., z], 'quaternion': [0., 0., 0., 1.]}
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
        'entryPose': pose(6.), 'surfacePose': pose(2.), 'outsidePose': pose(-6.),
        'inwardRasUnit': [0., 0., 1.], 'pathLengthMm': 4., 'corridorRadiusMm': 1.,
        'accessCompleteness': 'unsupported_skull_access_brain_corridor'}
    baseline = {'targetRemovedMm3': 0., 'nonTargetRemovedMm3': 0.,
                'removedCells': [], 'nativeActions': 1}
    task = larger_task(scene)
    zones, _ = source_zones(scene)
    sectors, _ = _sector_cells(scene, route)
    outcome = assess_task(engine, task, zones, {}, exact_replay=True,
        baseline=baseline, admitted_strokes=0, legacy_sectors=sectors)
    row = {'branchId': 'generated-source', 'sourceMethod': 'coupon', 'sourceAccessPlanSeal': 'coupon-plan',
        'route': route, 'hand': 'left', 'sourceStateId': intact, 'finalStateId': engine.state_id,
        'status': 'unpassed', 'nativeActions': len(engine.actions), 'approachBaseline': baseline,
        'stages': [], 'outcome': outcome, 'exactNativeReplay': True}
    report = {'schema': 'resectionlab.staged-anatomy-planning.v1',
        'sourceSceneId': scene.identity, 'sourceStateId': intact, 'sourceRole': 'DEVELOPMENT',
        'optimizerUpdates': 0, 'task': task,
        'cost': {'physicalCalls': 0, 'previewCalls': 0, 'directStepCalls': 0, 'maxPhysicalCalls': 600},
        'wallSeconds': 0., 'sourceAccessPlans': [{'method': 'coupon', 'planSeal': 'coupon-plan',
            'sourceSceneId': scene.identity, 'startStateId': intact, 'routeSeal': semantic_digest(route)}],
        'alternatives': [row]}
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branchId': row['branchId'], 'position': len(engine.actions),
        'branches': {row['branchId']: engine.snapshot()},
        'timeline': {row['branchId']: copy.deepcopy(engine.history)}, 'stagedPlanning': report}
    return engine, route, record


def test_complete_native_precision_entry_and_exit_preserve_original_source_and_failed_gates():
    engine, _, record = coupon()
    before = copy.deepcopy(record)
    result = continue_precision_reentry(engine, record, acknowledge_unknowns=True)
    assert record == before and engine.snapshot() == before['branches']['generated-source']
    assert result['report']['newStageCount'] >= 1
    assert result['report']['additionalTargetMm3'] > 0
    assert result['report']['additionalNonTargetMm3'] == 0
    assert not result['report']['outcome']['taskGateMet']
    assert 'wholeTarget' in result['report']['outcome']['failedGates']
    assert result['report']['originalBaselineNonTargetMm3'] == 0
    install_precision_continuation(record, result, engine.scene)
    for key in ('branches', 'timeline'):
        assert record[key]['generated-source'] == before[key]['generated-source']
    assert record['stagedPlanning'] == before['stagedPlanning']
    assert validate_precision_report(engine.scene, record, result['report'])['verified']
    with pytest.raises(ValueError, match='already contains'):
        continue_precision_reentry(result['engine'], record, acknowledge_unknowns=True)


def test_no_endpoint_ancestry_or_cursor_shortcuts_before_native_calls(monkeypatch):
    engine, _, record = coupon()
    monkeypatch.setattr(SurgicalEngine, 'step', lambda *_a, **_k: pytest.fail('No native call before binding'))
    changed = copy.deepcopy(record)
    changed['position'] = 0
    with pytest.raises(ValueError, match='complete saved'):
        continue_precision_reentry(engine, changed, acknowledge_unknowns=True)
    other = engine.clone()
    other.actions.append(copy.deepcopy(other.actions[-1]))
    assert other.state_id == engine.state_id
    with pytest.raises(ValueError, match='complete saved'):
        continue_precision_reentry(other, record, acknowledge_unknowns=True)
    changed = copy.deepcopy(record)
    changed['timeline']['generated-source'][0]['reason'] = 'forged'
    with pytest.raises(ValueError, match='history differs'):
        continue_precision_reentry(engine, changed, acknowledge_unknowns=True)


def test_geometry_scoring_never_transitions_and_rejects_resealed_forged_candidate(monkeypatch):
    engine, route, _ = coupon()
    limits = SkillLimits(240, 600, max_additional_non_target_mm3=20., baseline_non_target_mm3=0.)
    meter = TransitionMeter(600)
    before = engine.snapshot()
    monkeypatch.setattr(SurgicalEngine, 'step', lambda *_a, **_k: pytest.fail('Scoring cannot transition'))
    monkeypatch.setattr(SurgicalEngine, 'preview', lambda *_a, **_k: pytest.fail('Scoring cannot preview'))
    candidates, selection = propose_precision_reentries(engine, route, limits=limits)
    assert candidates and selection['rankedEligible'] > 0 and engine.snapshot() == before
    forged = copy.deepcopy(candidates[0])
    forged['endpoint']['position'][0] += 10
    forged['id'] = 'precision-reentry-'+semantic_digest({k: v for k, v in forged.items() if k != 'id'})
    with pytest.raises(ValueError, match='canonical'):
        execute_precision_reentry(engine, forged, route, limits=limits, meter=meter)
    assert meter.calls == 0


@pytest.mark.parametrize('unexpected', [False, True])
def test_failed_or_interrupted_exit_retains_speculative_prefix_and_clean_stop(monkeypatch, unexpected):
    import resectionlab.surgical_precision_reentry as precision
    engine, _, record = coupon()
    original = engine.snapshot()
    def refused_exit(work, *_args, **_kwargs):
        assert work.metrics()['targetRemovedMm3'] > 0
        if unexpected:
            raise RuntimeError('generated interruption after committed entry')
        return work.clone(), {'status': 'refused', 'reason': 'GENERATED_EXIT_UNAVAILABLE'}
    monkeypatch.setattr(precision, 'find_checked_exit', refused_exit)
    result = continue_precision_reentry(engine, record, acknowledge_unknowns=True)
    report = result['report']
    assert report['newStageCount'] == 0 and result['engine'].snapshot() == original
    assert report['additionalTargetMm3'] == 0
    assert report['outcome']['configurationTargetRemovedMm3'] == {}
    assert report['stoppedExecutions']
    assert any(row['nativeSnapshot']['checkpoint']['state']['removed']
               for row in report['stoppedExecutions'])
    assert report['cost']['physicalCalls'] > 2*len(original['actions'])
    assert report['stopReason'].startswith('OPTIONAL_CONTROLLER_ERROR' if unexpected else 'NO_ADMITTED')
    install_precision_continuation(record, result, engine.scene)
    assert record['branches']['generated-source'] == original
