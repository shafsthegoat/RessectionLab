"""Generated coupons only: precision rejects must precede public native calls."""
import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter
import resectionlab.surgical_precision_reentry as precision


ROOT = Path(__file__).resolve()
while not (ROOT / 'src/resectionlab').is_dir():
    ROOT = ROOT.parent


def coupon():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[9:14, 9:14, 8:17] = 1
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'tumor', 'material': 'tumor', 'target': True}],
        {'kind': 'generated precision validation coupon'}, {})
    engine = SurgicalEngine(standardize(scene))
    scene = engine.scene
    assert engine.step({'slot': 'left', 'operation': 'exchange',
                        'instrument': 'cusa', 'duration': .1})['status'] == 'accepted'
    pose = lambda z: {'position': [11., 11., z], 'quaternion': [0., 0., 0., 1.]}
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
             'entryPose': pose(8.), 'surfacePose': pose(3.), 'outsidePose': pose(-5.),
             'inwardRasUnit': [0., 0., 1.], 'pathLengthMm': 5.}
    limits = SkillLimits(240, 600, max_additional_non_target_mm3=20.,
                         baseline_non_target_mm3=0.)
    return engine, route, limits


def prohibit_native(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Read-only precision validation reached a public native transition')
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)


def test_precision_proposals_are_deterministic_read_only_and_state_bound(monkeypatch):
    engine, route, limits = coupon()
    snapshot, original_route = engine.snapshot(), copy.deepcopy(route)
    prohibit_native(monkeypatch)
    candidates, report = precision.propose_precision_reentries(engine, route, limits=limits)
    assert candidates
    assert (candidates, report) == precision.propose_precision_reentries(engine, route, limits=limits)
    assert engine.snapshot() == snapshot and route == original_route
    assert len(candidates) <= 2
    for candidate in candidates:
        assert candidate['sourceStateId'] == snapshot['stateId']
        assert candidate['sourceRouteSeal'] == semantic_digest(route)
        assert candidate['id'] == 'precision-reentry-' + semantic_digest(
            {key: value for key, value in candidate.items() if key != 'id'})


@pytest.mark.parametrize('change', [
    'scene', 'path_type', 'zero_axis', 'nan_axis', 'infinite_entry',
    'nonunit_quaternion', 'wrong_orientation', 'wrong_length', 'inside_outside',
    'nonfinite_length', 'boolean_length', 'dirty_left', 'dirty_right', 'hold',
])
def test_precision_source_rejects_before_native_calls(monkeypatch, change):
    engine, route, limits = coupon()
    if change == 'scene': route['sourceSceneId'] = 'different-source'
    elif change == 'path_type': route['pathType'] = 'unsupported'
    elif change == 'zero_axis': route['inwardRasUnit'] = [0., 0., 0.]
    elif change == 'nan_axis': route['inwardRasUnit'][2] = float('nan')
    elif change == 'infinite_entry': route['entryPose']['position'][0] = float('inf')
    elif change == 'nonunit_quaternion': route['entryPose']['quaternion'] = [0., 0., 0., 2.]
    elif change == 'wrong_orientation': route['entryPose']['quaternion'] = [1., 0., 0., 0.]
    elif change == 'wrong_length': route['pathLengthMm'] += 1.
    elif change == 'inside_outside': route['outsidePose']['position'][2] = 4.
    elif change == 'nonfinite_length': route['pathLengthMm'] = float('nan')
    elif change == 'boolean_length': route['pathLengthMm'] = True
    elif change == 'dirty_left': engine.state['slots']['left']['inserted'] = True
    elif change == 'dirty_right': engine.state['slots']['right']['active'] = True
    elif change == 'hold': engine.state['holds']['synthetic'] = {}
    prohibit_native(monkeypatch)
    with pytest.raises(ValueError):
        precision.propose_precision_reentries(engine, route, limits=limits)


@pytest.mark.parametrize('change', ['coordinate', 'source_state', 'route', 'id', 'contact', 'extra_key'])
def test_forged_candidate_even_with_updated_id_is_refused_without_calls(monkeypatch, change):
    engine, route, limits = coupon()
    candidate = copy.deepcopy(precision.propose_precision_reentries(engine, route, limits=limits)[0][0])
    if change == 'coordinate': candidate['endpoint']['position'][0] += .01
    elif change == 'source_state': candidate['sourceStateId'] = 'different-state'
    elif change == 'route': candidate['sourceRouteSeal'] = 'different-route'
    elif change == 'id': candidate['id'] = 'different-id'
    elif change == 'contact': candidate['sourceNewBrainMm3'] += 1.
    else: candidate['injected'] = True
    if change != 'id':
        candidate['id'] = 'precision-reentry-' + semantic_digest(
            {key: value for key, value in candidate.items() if key != 'id'})
    meter = TransitionMeter(600)
    snapshot = engine.snapshot()
    prohibit_native(monkeypatch)
    with pytest.raises(ValueError, match='canonical'):
        precision.execute_precision_reentry(engine, candidate, route, limits=limits, meter=meter)
    assert meter.calls == 0 and engine.snapshot() == snapshot


@pytest.mark.parametrize('kind', ['whole', 'continuation'])
def test_original_approach_caps_cannot_be_bypassed(monkeypatch, kind):
    engine, route, limits = coupon()
    monkeypatch.setattr(engine, 'metrics', lambda: {
        'nonTargetRemovedMm3': 201. if kind == 'whole' else 21.})
    prohibit_native(monkeypatch)
    with pytest.raises(ValueError, match='approach budget'):
        precision.propose_precision_reentries(engine, route, limits=limits)


def stopped_report(monkeypatch):
    # Reuse the small, actually executed native staged-report coupon. No patient data.
    helper = runpy.run_path(str(ROOT / 'tests/test_surgical_staged_planning.py'))
    helper['_native_report'].__globals__['SurgicalScene'] = lambda *a, **k: standardize(SurgicalScene(*a, **k))
    scene, record, report = helper['_native_report']()
    row = report['alternatives'][0]
    row['hand'] = 'left'
    route = row['route']
    route['outsidePose'] = {'position': [6., 6., -5.], 'quaternion': [0., 0., 0., 1.]}
    report['sourceAccessPlans'][0]['routeSeal'] = semantic_digest(route)
    helper['_seal_report'](report)
    record.update(stagedPlanning=report, branchId=row['branchId'], position=row['nativeActions'])
    engine = SurgicalEngine.restore(scene, record['branches'][row['branchId']])
    record['timeline'][row['branchId']] = copy.deepcopy(engine.history)
    monkeypatch.setattr(precision, 'propose_precision_reentries', lambda *_a, **_k: ([], {}))
    result = precision.continue_precision_reentry(engine, record, acknowledge_unknowns=True)
    precision.install_precision_continuation(record, result, scene)
    return scene, record, result['report']


def test_precision_validator_rejects_resealed_checkpoint_identity_corruption(monkeypatch):
    scene, record, report = stopped_report(monkeypatch)
    snapshot = record['branches'][report['sourceBranchId']]
    snapshot['checkpoint']['state']['elapsedSeconds'] += 1.
    report['sourceProgramSeal'] = semantic_digest(snapshot)
    report['comparisonSeal'] = semantic_digest({key: value for key, value in report.items()
                                               if key != 'comparisonSeal'})
    prohibit_native(monkeypatch)
    with pytest.raises(ValueError):
        precision.validate_precision_report(scene, record, report)


@pytest.mark.parametrize('change', ['cursor', 'program', 'history', 'branch'])
def test_precision_selected_source_must_be_complete_and_exact(monkeypatch, change):
    scene, record, report = stopped_report(monkeypatch)
    engine = SurgicalEngine.restore(scene, record['branches'][report['sourceBranchId']])
    del record['precisionPlanning']
    if change == 'cursor': record['position'] -= 1
    elif change == 'program':
        # A no-op exchange preserves state but creates a different native ancestry.
        engine.actions.append(copy.deepcopy(engine.actions[0]))
    elif change == 'history': engine.history[-1]['reason'] = 'forged history field'
    else: record['branchId'] = 'unrelated-branch'
    prohibit_native(monkeypatch)
    with pytest.raises(ValueError):
        precision.continue_precision_reentry(engine, record, acknowledge_unknowns=True)


@pytest.mark.parametrize('change', ['source_checkpoint', 'entry_brain', 'native_cost'])
def test_admitted_report_rejects_resealed_source_or_evidence_corruption(monkeypatch, change):
    helper = runpy.run_path(str(ROOT / 'tests/test_surgical_precision_reentry.py'))
    engine, _, record = helper['coupon']()
    result = precision.continue_precision_reentry(engine, record, acknowledge_unknowns=True)
    assert result['report']['newStageCount'] > 0
    precision.install_precision_continuation(record, result, engine.scene)
    report = result['report']
    if change == 'source_checkpoint':
        source = record['branches'][report['sourceBranchId']]
        source['checkpoint']['state']['elapsedSeconds'] += 1.
        report['sourceProgramSeal'] = semantic_digest(source)
    elif change == 'entry_brain':
        report['stages'][0]['entryBrainIncrementMm3'] += 1.
    else:
        for key in ('physicalCalls', 'previewCalls', 'directStepCalls'):
            report['cost'][key] = 0
    report['comparisonSeal'] = semantic_digest({key: value for key, value in report.items()
                                               if key != 'comparisonSeal'})
    prohibit_native(monkeypatch)
    with pytest.raises(ValueError):
        precision.validate_precision_report(engine.scene, record, report)
