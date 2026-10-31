"""Falsification controls for the independent literal-geometry source auditor."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_surgical_anatomical_encounter import context_case, context_report, qualified, prepared

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_anatomical_encounter.py'
spec = importlib.util.spec_from_file_location('independent_anatomical_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def slot(position, instrument='cusa', quaternion=(0., 0., 0., 1.)):
    return {'position': list(position), 'quaternion': list(quaternion), 'instrument': instrument,
            'parameters': {'radiusMm': 1.5, 'depthMm': 2.}}


def footprint(cells, removed=(), *, shape=(32, 32, 64), affine=None, margin=0.):
    mask = np.zeros(shape, bool)
    for cell in cells:
        mask[cell] = True
    return audit.IndependentFootprint(mask, np.eye(4) if affine is None else affine, removed,
                                     {'marginMm': margin, 'maxNearMm3': 1000.})


def test_literal_cusa_sphere_has_no_shoulder_or_shortened_shaft():
    tool = slot([10., 10., 45.])
    shaft, tip = audit.literal_capsules(tool)
    assert np.array_equal(shaft[0], [10., 10., 5.])
    assert np.array_equal(shaft[1], tip[0]) and np.array_equal(tip[0], tip[1])
    assert shaft[2] == .5 and tip[2] == 1.5
    # An invented 2mm rear working capsule hits this 0.25mm source box; the
    # actual spherical tip and thin shaft both miss it.
    affine = np.diag([.25, .25, .25, 1.])
    tracker = footprint([(46, 40, 175)], shape=(64, 64, 190), affine=affine)
    tracker.add(audit.literal_capsules(tool), ())
    assert tracker.receipt(())['nearSourceCells'] == []
    tracker.add([(np.array([10., 10., 43.]), np.array([10., 10., 45.]), 1.5, 'working')], ())
    assert tracker.receipt(())['nearMm3'] == pytest.approx(.25 ** 3)


def test_shaft_omission_is_detected_away_from_working_tip():
    tracker = footprint([(10, 10, 15)])
    capsules = audit.literal_capsules(slot([10., 10., 45.]))
    tracker.add(capsules[1:], ())
    assert tracker.receipt(())['nearMm3'] == 0.
    tracker.add(capsules[:1], ())
    result = tracker.receipt(())
    assert result['nearMm3'] == result['shaftNearMm3'] == 1.
    assert result['workingNearMm3'] == 0.


def test_stationary_other_hand_is_counted_independently():
    before = {'left': slot([5., 5., 45.]), 'right': slot([20., 20., 45.], 'frazier_suction')}
    after = copy.deepcopy(before)
    after['left']['position'][2] = 46.
    tracker = footprint([(20, 20, 15)])
    tracker.observe({'slot': 'left', 'operation': 'move'}, before, after, ())
    assert tracker.receipt(())['shaftNearMm3'] == 1.
    omitted = footprint([(20, 20, 15)])
    omitted.add(audit.axial_capsules(before['left'], after['left']), ())
    assert omitted.receipt(())['nearMm3'] == 0.


def test_withdrawal_cannot_be_omitted_from_union():
    before = {'left': slot([10., 10., 45.]), 'right': slot([25., 25., 45.], 'frazier_suction')}
    after = copy.deepcopy(before)
    after['left']['position'][2] = 5.
    tracker = footprint([(10, 10, 2)])
    tracker.add(audit.literal_capsules(before['left']), ())
    assert tracker.receipt(())['nearMm3'] == 0.
    tracker.observe({'slot': 'left', 'operation': 'withdraw'}, before, after, ())
    assert tracker.receipt(())['nearMm3'] == 1.


def test_historical_removal_and_new_removal_are_not_retained_encounter():
    shape = (32, 32, 64)
    removed = int(np.ravel_multi_index((10, 10, 15), shape))
    retained = int(np.ravel_multi_index((10, 10, 16), shape))
    tracker = footprint([(10, 10, 15), (10, 10, 16)], [removed])
    tracker.add(audit.literal_capsules(slot([10., 10., 45.])), [removed])
    result = tracker.receipt([removed])
    assert result['baselineRemovedContextMm3'] == result['baselineRetainedContextMm3'] == 1.
    assert result['newRemovedContextMm3'] == 0.
    assert result['nearSourceCells'] == [retained]
    assert result['nearPositionsMm'] == [[10., 10., 16.]]
    assert tracker.receipt([removed, retained])['newRemovedContextMm3'] == 1.


@pytest.mark.parametrize('instrument', ['cusa', 'frazier_suction'])
def test_union_is_partition_and_reversal_invariant(instrument):
    cells = [(10, 10, z) for z in (3, 15, 25, 46)]
    start, middle, end = (slot([10., 10., z], instrument) for z in (8., 25., 45.))
    whole = footprint(cells)
    whole.add(audit.axial_capsules(start, end), ())
    pieces = footprint(cells)
    for a, b in ((start, middle), (middle, end), (end, start)):
        pieces.add(audit.axial_capsules(a, b), ())
    assert whole.receipt(()) == pieces.receipt(())


def test_rotated_reflected_orthogonal_boxes_and_geometric_margin():
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler('xyz', [20., 30., 10.], degrees=True).as_matrix() @ np.diag([-.5, .75, 1.])
    affine[:3, 3] = [50., -20., 10.]
    center = affine[:3, :3] @ [10., 10., 15.] + affine[:3, 3]
    quaternion = Rotation.align_vectors([[0., 0., 1.]], [[0., 0., 1.]])[0].as_quat()
    tool = slot(center, quaternion=quaternion)
    tracker = footprint([(10, 10, 15)], affine=affine)
    tracker.add(audit.literal_capsules(tool), ())
    assert tracker.receipt(())['workingNearMm3'] == pytest.approx(.375)
    assert np.allclose(tracker.receipt(())['nearPositionsMm'], [center])
    near = footprint([(10, 10, 15)], margin=2.)
    near.add(audit.literal_capsules(slot([13., 10., 15.])), ())
    assert near.receipt(())['nearMm3'] == 1.


def test_shear_and_nonaxial_or_nonmonotone_motion_are_refused():
    affine = np.eye(4)
    affine[0, 1] = .1
    with pytest.raises(ValueError, match='sheared'):
        footprint([(1, 1, 1)], affine=affine)
    start, end = slot([10., 10., 20.]), slot([11., 10., 30.])
    with pytest.raises(ValueError, match='collinear'):
        audit.axial_capsules(start, end)
    before = {'left': slot([10., 10., 20.]), 'right': slot([25., 25., 45.], 'frazier_suction')}
    trace = []
    previous = copy.deepcopy(before)
    for z in (25., 23., 30.):
        current = copy.deepcopy(previous)
        current['left']['position'][2] = z
        trace.append({'hand': 'left', 'operation': 'move', 'before': previous, 'after': current})
        previous = current
    with pytest.raises(ValueError, match='nonmonotone'):
        audit.certify_trace({'slot': 'left', 'operation': 'move'}, before, previous, trace)


def test_transaction_clones_detach_trace_owner():
    engine = object.__new__(audit.CaptureEngine)
    engine.state = {}; engine.actions = []; engine.history = []; engine._audit_trace = []
    candidate = engine.clone()
    candidate._audit_trace.append({'uncommitted': True})
    assert engine._audit_trace == []


def test_tangent_and_just_outside_use_declared_linear_boundary_tolerance():
    for displacement, expected in ((0., 1.), (-.5e-9, 1.), (-2e-9, 0.)):
        tracker = footprint([(12, 10, 45)])
        tracker.add(audit.literal_capsules(slot([10. + displacement, 10., 45.])), ())
        assert tracker.receipt(())['nearMm3'] == expected
        assert audit.BOUNDARY_TOLERANCE_MM == 1e-9


def test_auditor_does_not_import_planner_or_encounter_predictor():
    import ast
    parsed = ast.parse(PATH.read_text())
    modules = [n.module for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)]
    assert not any(m and ('surgical_two_tool' in m or 'surgical_anatomical_encounter' in m) for m in modules)


def test_generated_native_prefixes_match_independent_source_boxes(context_case, context_report):
    _, engine, _, _, _ = context_case
    mask, affine = audit.load_source_context(engine, context_report)
    row = context_report['publicPlan']['alternatives'][1]
    trial, tracker, prefixes = audit.replay_program(engine, mask, affine,
        context_report['publicPlan']['constraints']['encounter'], row['actionProgram'], row)
    assert tracker.receipt(trial.state['removed']) == row['anatomicalEncounter']
    assert len(prefixes) == len(row['actionProgram']) and tracker.box_tests > 0
    changed = copy.deepcopy(row)
    changed['contextPrefixHashes'][0] = 'sha256:invented-footprint'
    with pytest.raises(ValueError, match='anatomical prefix'):
        audit.replay_program(engine, mask, affine, context_report['publicPlan']['constraints']['encounter'],
                             changed['actionProgram'], changed)


@pytest.mark.parametrize('field', ['review', 'original', 'label', 'public-mask'])
def test_generated_source_binding_cannot_be_resealed_to_other_evidence(context_case, context_report, field):
    _, engine, _, _, _ = context_case
    changed = copy.deepcopy(context_report)
    binding = changed['contextBinding']
    if field == 'review': binding['reviewStatus'] = 'reviewed'
    elif field == 'original': binding['original']['sha256'] = '0' * 64
    elif field == 'label': binding['labelValue'] = 17
    else: changed['publicPlan']['publicObservation']['anatomicalContext']['maskHash'] = 'invented-mask'
    binding['bindingSeal'] = audit.semantic_digest({k: v for k, v in binding.items() if k != 'bindingSeal'})
    with pytest.raises(ValueError, match='binding fields|public source context'):
        audit.load_source_context(engine, changed)
