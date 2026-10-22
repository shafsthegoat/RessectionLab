"""Generated contact coupons for an explicitly uncalibrated optional law."""
import copy
from fractions import Fraction
import json

import numpy as np
import pytest

from resectionlab.surgical_cusa_v3 import (SCHEMA, apply, contact_limited_pose,
    default_profile, initialize, metrics, reconcile_removed, validate,
    validate_parameters)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def fixture(points=((9, 9, 9),), *, rate=2., enabled=True, material='tumor', spacing=(1., 1., 1.)):
    labels = np.zeros((20, 20, 20), np.uint16)
    for point in points:
        labels[point] = 1
    p = default_profile()
    p['capacityMm3PerSecond'] = rate
    scene = SurgicalScene(labels, np.diag([*spacing, 1.]),
        [{'id': 1, 'name': 'Generated contact coupon', 'material': material,
          'target': material == 'tumor'}],
        {'kind': 'generated CUSA contact coupon; no acquired anatomy'},
        {'cusaV3': p} if enabled else {})
    engine = SurgicalEngine(scene)
    initialize(engine)
    cells = list(map(int, np.flatnonzero(scene.labels)))
    return engine, cells


def contact(engine, cells, seconds, **values):
    position = values.pop('position', [9., 9., 9.])
    active = values.pop('active', True)
    return apply(engine, cells, seconds, {'radiusMm': 1.5, 'power': 1., **values},
        position=position, active=active)


def conserved(engine):
    m = metrics(engine)
    assert m['cusaInputCapacityMm3'] == pytest.approx(
        m['cusaAspiratedSolidMm3'] + m['cusaPendingCapacityMm3'] +
        m['cusaUnusedCapacityMm3'] + m['cusaDiscardedCapacityMm3'], abs=1e-12)
    assert set(engine.state['cusaV3']['progressMm3']) == set(engine.state['cusaV3']['progressExact'])
    return m


def test_absent_property_preserves_legacy_state_and_profile_is_complete_explicit():
    engine, _ = fixture(enabled=False)
    before = engine.snapshot()
    initialize(engine)
    assert reconcile_removed(engine) == 0.
    assert engine.snapshot() == before and 'cusaV3' not in engine.state
    assert validate(engine.scene) is None
    assert validate(default_profile())['schema'] == SCHEMA
    for change in ({'tipRadiusMm': 12.}, {'shaftRadiusMm': 1.},
                   {'capacityMm3PerSecond': 0.}, {'maxActivation': 10.},
                   {'provenance': {'kind': 'calibrated', 'description': 'unsupported'}}):
        with pytest.raises(ValueError):
            validate({**default_profile(), **change})
    incomplete = default_profile()
    incomplete.pop('capacityMm3PerSecond')
    with pytest.raises(ValueError, match='COMPLETE_EXPLICIT'):
        validate(incomplete)


@pytest.mark.parametrize('radius', [.1, 1., 3., 12., 1.500000001])
def test_growing_or_shrinking_catalog_radius_is_rejected(radius):
    engine, cells = fixture()
    before = copy.deepcopy(engine.state)
    with pytest.raises(ValueError, match='FIXED_CATALOG_GEOMETRY'):
        contact(engine, cells, 1., radiusMm=radius)
    assert engine.state == before


@pytest.mark.parametrize('values', [{'power': 1.1}, {'power': -1.}, {'power': True},
    {'duty': 1.1}, {'duty': -1.}, {'shaftRadiusMm': 2.}])
def test_activation_and_duty_are_bounded_without_silent_clamping(values):
    with pytest.raises(ValueError):
        validate_parameters(values)


def test_inactive_zero_activation_zero_duty_and_no_contact_cannot_remove():
    engine, cells = fixture()
    for values in ({'active': False}, {'power': 0.}, {'duty': 0.}):
        assert contact(engine, cells, 10., **values) == []
    assert not engine.state['removed'] and not engine.state['cusaV3']['progressMm3']
    assert conserved(engine)['cusaInputCapacityMm3'] == 0.
    contact(engine, cells, .25, position=[9., 9., 2.])
    m = conserved(engine)
    assert not engine.state['removed'] and m['cusaUnusedCapacityMm3'] == .5
    assert engine.state['fluidRemovedMm3'] == 0.


def test_duration_accumulates_capacity_before_whole_source_voxel_is_aspirated():
    engine, cells = fixture()
    contact(engine, cells, .125)
    assert not engine.state['removed']
    assert conserved(engine)['cusaPendingCapacityMm3'] == .25
    contact(engine, cells, .125, power=.5)
    assert conserved(engine)['cusaPendingCapacityMm3'] == .375
    assert engine.state['cusaV3']['contactSeconds'][str(cells[0])] == .25
    contact(engine, cells, .3125)
    assert engine.state['removed'] == cells
    assert conserved(engine)['cusaAspiratedSolidMm3'] == 1.
    assert engine.state['cusaV3']['progressMm3'] == {}
    assert engine.state['cusaV3']['contactSeconds'] == {}
    assert engine.state['debrisMm3'] == engine.state['fluidRemovedMm3'] == 0.
    json.dumps(engine.state, allow_nan=False)


def test_total_capacity_is_shared_nearest_contact_then_stable_source_id():
    engine, cells = fixture(((8, 9, 9), (10, 9, 9)))
    # Equal physical distances: source ID resolves the tie, independent of input.
    events = contact(engine, list(reversed(cells)) * 2, .75)
    assert engine.state['removed'] == [min(cells)]
    assert engine.state['cusaV3']['progressMm3'] == {str(max(cells)): .5}
    assert events[0]['inputCapacityMm3'] == 1.5
    assert conserved(engine)['cusaAspiratedSolidMm3'] == 1.
    other, other_cells = fixture(((8, 9, 9), (10, 9, 9)))
    contact(other, other_cells, .5, position=[10., 9., 9.])
    assert other.state['removed'] == [max(other_cells)]


def test_more_contact_cells_cannot_multiply_one_interval_capacity():
    points = tuple((x, y, z) for x in (8, 9) for y in (8, 9) for z in (8, 9))
    engine, cells = fixture(points)
    contact(engine, cells, .125)
    m = conserved(engine)
    assert not engine.state['removed']
    assert m['cusaPendingCapacityMm3'] == m['cusaInputCapacityMm3'] == .25


def test_timestep_partition_keeps_same_cleared_cells_and_volume_ledgers():
    whole, cells = fixture(rate=1.)
    divided, divided_cells = fixture(rate=1.)
    contact(whole, cells, 1.)
    for _ in range(10):
        contact(divided, divided_cells, .1)
    assert whole.state['removed'] == divided.state['removed'] == cells
    a, b = conserved(whole), conserved(divided)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_rounded_display_progress_cannot_force_full_completion():
    engine, cells = fixture(rate=1.)
    almost = float(np.nextafter(1., 0.))
    gap = 1. - almost
    contact(engine, cells, almost)
    contact(engine, cells, gap * .75)
    state = engine.state['cusaV3']
    # The float display rounds to 1, but exact pending capacity is still below 1.
    assert state['progressMm3'][str(cells[0])] == 1.
    assert Fraction(*state['progressExact'][str(cells[0])]) < 1
    assert not engine.state['removed']
    contact(engine, cells, gap * .25)
    assert engine.state['removed'] == cells
    conserved(engine)


def test_swept_remote_ids_are_filtered_and_tip_position_is_explicit():
    engine, cells = fixture(((9, 9, 9), (9, 9, 15)))
    with pytest.raises(ValueError, match='EXPLICIT_TIP_POSITION'):
        apply(engine, cells, 1., {})
    contact(engine, cells, 2.)
    assert engine.state['removed'] == [min(cells)]
    assert conserved(engine)['cusaUnusedCapacityMm3'] == 3.
    with pytest.raises(ValueError, match='SOURCE_CELL'):
        contact(engine, [True], 1.)


def test_pending_work_removed_by_another_tool_is_discarded_capacity_not_mass():
    engine, cells = fixture()
    contact(engine, cells, .125)
    engine.state['removed'] = cells[:]
    assert reconcile_removed(engine) == .25
    m = conserved(engine)
    assert m['cusaDiscardedCapacityMm3'] == .25
    assert m['cusaAspiratedSolidMm3'] == 0.
    assert not engine.state['cusaV3']['progressMm3']
    assert not engine.state['cusaV3']['contactSeconds']


def test_contact_limited_pose_stops_at_retained_tissue_until_clearance():
    engine, cells = fixture()
    before = copy.deepcopy(engine.state)
    start = np.array([9., 9., 2.])
    requested = np.array([9., 9., 12.])
    limited, contacts = contact_limited_pose(engine, start, requested)
    assert limited[2] < requested[2] and contacts == cells
    assert engine.state == before
    still, _ = contact_limited_pose(engine, limited, requested)
    np.testing.assert_array_equal(still, limited)
    contact(engine, cells, .5, position=limited)
    resumed, contacts = contact_limited_pose(engine, limited, requested)
    np.testing.assert_array_equal(resumed, requested)
    assert contacts == []


def test_protected_membership_has_no_partial_capacity_or_removal():
    engine, cells = fixture()
    engine._protected_ids = frozenset(cells)
    before = copy.deepcopy(engine.state)
    with pytest.raises(ValueError, match='PROTECTED_STRUCTURE'):
        contact(engine, cells, 1.)
    assert engine.state == before


def test_affine_voxel_volume_sets_clearance_threshold_and_json_pending_state_resumes():
    engine, cells = fixture(spacing=(.5, .5, 2.))
    position = engine.scene.points(cells)[0]
    assert engine.scene.voxel_mm3 == .5
    contact(engine, cells, .125, position=position)
    assert not engine.state['removed']
    assert conserved(engine)['cusaPendingCapacityMm3'] == .25
    engine.state = json.loads(json.dumps(engine.state, allow_nan=False))
    initialize(engine)
    contact(engine, cells, .125, position=position)
    assert engine.state['removed'] == cells
    assert conserved(engine)['cusaAspiratedSolidMm3'] == .5
