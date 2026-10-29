"""Posture coordinates, physical units, bounded mechanics and mass invariants."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest

from resectionlab.surgical_position import (apply_position, validate_position, describe_position,
    gravity_forces_n, activate_exposed_patch, relax_fluid, GRAVITY_M_PER_S2, PA_PER_MMHG)


def engine(shape=(11, 11, 11), affine=None):
    affine = np.eye(4) if affine is None else affine
    scene = SimpleNamespace(shape=shape, affine=affine, labels=np.ones(shape, np.uint8))
    state = {'removed': [], 'holds': {}, 'fluid': {}, 'debris': {}, 'cutEdges': [], 'transectedEdges': []}
    e = SimpleNamespace(scene=scene, state=state, by_label={1: {'material': 'brain'}, 2: {'material': 'bone'}})
    def occupancy():
        a = e.scene.labels.ravel().copy(); a[e.state['removed']] = 0; return a
    e.occupancy = occupancy
    return e


def cell(e, ijk): return int(np.ravel_multi_index(ijk, e.scene.shape))


@pytest.mark.parametrize('preset,direction', [('supine', [0, -1, 0]), ('prone', [0, 1, 0]),
    ('left_lateral', [-1, 0, 0]), ('right_lateral', [1, 0, 0])])
def test_preset_gravity_is_physical_ras(preset, direction):
    p = validate_position({'preset': preset}); result = describe_position(p)
    np.testing.assert_allclose(result['gravityRasMPerS2'], np.array(direction) * GRAVITY_M_PER_S2, atol=1e-12)
    for field in ('bodyRasToRoom', 'headRasToRoom'):
        matrix = np.array(result[field]); np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.)


def test_head_rotation_changes_tissue_gravity_without_camera_or_body_motion():
    neutral = describe_position(validate_position({'preset': 'supine'}))
    turned = describe_position(validate_position({'preset': 'supine', 'headRotationRasDeg': [0, 0, 90]}))
    np.testing.assert_allclose(turned['gravityRasMPerS2'], [-GRAVITY_M_PER_S2, 0, 0], atol=1e-12)
    assert neutral['bodyRasToRoom'] == turned['bodyRasToRoom']
    assert turned['siteHeightAboveHeartEstimateMm'] == neutral['siteHeightAboveHeartEstimateMm']


def test_positive_tilt_elevates_site_and_hydrostatic_offset_has_correct_units_and_sign():
    result = describe_position(validate_position({'preset': 'semisitting', 'bodyTiltDeg': 30,
        'heartToSiteRasMm': [0, 0, 300], 'fluidDensityKgPerM3': 1060}))
    assert result['siteHeightAboveHeartEstimateMm'] == pytest.approx(150.)
    expected = -1060 * GRAVITY_M_PER_S2 * .150
    assert result['hydrostaticSiteMinusHeartPa'] == pytest.approx(expected)
    assert result['hydrostaticSiteMinusHeartMmHg'] == pytest.approx(expected / PA_PER_MMHG)
    assert result['heightIsMeasured'] is False
    lower = describe_position(validate_position({'preset': 'supine', 'siteHeightAboveHeartMm': -100}))
    assert lower['hydrostaticSiteMinusHeartPa'] == pytest.approx(1060 * GRAVITY_M_PER_S2 * .1)


def test_estimated_vertical_height_override_retains_provenance_and_sources():
    state = {}; result = apply_position(state, {'preset': 'prone', 'siteHeightAboveHeartMm': 200,
        'heightEstimateSource': 'operator estimate, no heart segmentation'})
    assert result['siteHeightAboveHeartEstimateMm'] == 200
    assert result['heightEstimateSource'] == 'operator estimate, no heart segmentation'
    assert len(result['sources']) >= 3 and all(s['url'].startswith('https://') for s in result['sources'])
    assert 'perfusion' in result['hydrostaticModel']
    result['sources'].clear()
    assert describe_position(state['patientPosition'])['sources']


@pytest.mark.parametrize('changes', [{'preset': 'invalid'}, {'bodyTiltDeg': 90}, {'bodyTiltDeg': True},
    {'headRotationRasDeg': [0, float('nan'), 0]}, {'headRotationRasDeg': [0, 0]},
    {'fluidDensityKgPerM3': 0}, {'heartToSiteRasMm': [0, 0, 3000]}, {'siteHeightAboveHeartMm': float('inf')},
    {'heightEstimateSource': ''}, {'modelVersion': 2}, {'modelVersion': True}, {'camera': [0, 0, 0]}])
def test_invalid_position_rejected_before_state_mutation(changes):
    e = engine(); apply_position(e.state, {'preset': 'supine'}); before = copy.deepcopy(e.state)
    with pytest.raises(ValueError, match='INVALID_PATIENT_POSITION'):
        apply_position(e.state, {'preset': 'supine', **changes})
    assert e.state == before


def test_absent_position_never_initializes_fields_or_changes_legacy_state():
    e = engine(); before = copy.deepcopy(e.state)
    assert activate_exposed_patch(e) == [] and relax_fluid(e, .01) == []
    np.testing.assert_array_equal(gravity_forces_n(e, [1, 2]), np.zeros((2, 3)))
    assert e.state == before


def test_body_force_uses_kg_m3_mm3_to_m3_and_not_grid_axis_or_translation():
    affine = np.array([[0., -3., 0, 1234], [2., 0, 0, -4321], [0, 0, 4., 89], [0, 0, 0, 1.]])
    e = engine(affine=affine); apply_position(e.state, {'preset': 'supine'})
    force = gravity_forces_n(e, [0, 10])
    np.testing.assert_allclose(force, [[0, -1040 * 24e-9 * GRAVITY_M_PER_S2, 0]] * 2)


def test_exposed_patch_is_local_limited_and_respects_removed_bone_and_cut_interfaces():
    e = engine(); removed = cell(e, (5, 5, 5)); seed = cell(e, (6, 5, 5))
    apply_position(e.state, {'preset': 'supine'}); e.state['removed'] = [removed]
    e.scene.labels[7, 5, 5] = 2
    blocked = cell(e, (6, 6, 5))
    e.state['cutEdges'] = [sorted((seed, n)) for n in [blocked]]
    active = activate_exposed_patch(e, radius_mm=1, max_nodes=7, exposed_cells=[seed])
    assert seed in active and removed not in active and blocked not in active
    assert cell(e, (7, 5, 5)) not in active and len(active) <= 7
    for c in active:
        assert np.linalg.norm(np.array(np.unravel_index(c, e.scene.shape)) - [6, 5, 5]) <= 1
    assert e.state['deformationModel']['anchors']


def test_intact_anatomy_and_natural_image_edge_do_not_create_mechanics_patch():
    e = engine(); e.scene.labels[0, :, :] = 0
    apply_position(e.state, {'preset': 'semisitting'})
    assert activate_exposed_patch(e) == []
    assert 'deformationModel' not in e.state


def test_patch_repeat_is_deterministic_and_never_exceeds_node_budget():
    e = engine(); apply_position(e.state, {'preset': 'park_bench'})
    e.state['removed'] = [cell(e, (5, 5, 5))]; other = copy.deepcopy(e)
    a = activate_exposed_patch(e, radius_mm=4, max_nodes=32)
    b = activate_exposed_patch(other, radius_mm=4, max_nodes=32)
    assert a == b and len(a) == 32
    activate_exposed_patch(e, radius_mm=4, max_nodes=32)
    assert e.state['deformationModel']['active'] == a


def test_unchanged_patch_uses_bounded_cache_without_new_graph_work(monkeypatch):
    import resectionlab.surgical_position as position
    e = engine(); apply_position(e.state, {'preset': 'supine'})
    e.state['removed'] = [cell(e, (5, 5, 5))]
    active = activate_exposed_patch(e, max_nodes=32)
    def unexpected_graph_work(*args): raise AssertionError('cached patch traversed graph')
    monkeypatch.setattr(position, '_neighbors', unexpected_graph_work)
    assert activate_exposed_patch(e, max_nodes=32) == active


def test_downhill_subcapacity_fluid_moves_one_edge_with_conservative_debris():
    e = engine(shape=(5, 5, 5)); e.scene.labels[:] = 0
    apply_position(e.state, {'preset': 'supine'})
    origin = cell(e, (2, 3, 2)); below = cell(e, (2, 2, 2)); twice = cell(e, (2, 1, 2))
    e.state['fluid'] = {str(origin): .8}; e.state['debris'] = {str(origin): .2}
    transfers = relax_fluid(e, .125)
    assert len(transfers) == 1 and transfers[0]['fromCell'] == origin and transfers[0]['toCell'] == below
    assert transfers[0]['fluidMm3'] == pytest.approx(.4) and transfers[0]['debrisMm3'] == pytest.approx(.1)
    assert sum(e.state['fluid'].values()) == pytest.approx(.8)
    assert sum(e.state['debris'].values()) == pytest.approx(.2)
    assert e.state['fluid'][str(origin)] == pytest.approx(.4)
    assert str(twice) not in e.state['fluid']


def test_prone_fluid_direction_reverses_and_solid_cells_block_flow():
    e = engine(shape=(5, 5, 5)); e.scene.labels[:] = 0
    origin = cell(e, (2, 2, 2)); above = cell(e, (2, 3, 2))
    e.scene.labels.flat[above] = 2
    apply_position(e.state, {'preset': 'prone'}); e.state['fluid'] = {str(origin): .8}
    assert relax_fluid(e, .25) == []
    e.scene.labels.flat[above] = 0; relax_fluid(e, .25)
    assert e.state['fluid'][str(above)] == pytest.approx(.8)


def test_fluid_uses_physical_affine_and_preserves_boundary_mass():
    affine = np.array([[0, -1., 0, 0], [1., 0, 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]])
    e = engine(shape=(4, 4, 4), affine=affine); e.scene.labels[:] = 0
    apply_position(e.state, {'preset': 'supine'})
    origin = cell(e, (1, 2, 2)); below = cell(e, (0, 2, 2))
    e.state['fluid'] = {str(origin): .75}; relax_fluid(e, .25)
    assert e.state['fluid'][str(below)] == .75
    relax_fluid(e, .25)
    assert sum(e.state['fluid'].values()) == .75


def test_fluid_snapshot_prevents_multihop_and_overfill_and_is_replay_deterministic():
    e = engine(shape=(5, 5, 5)); e.scene.labels[:] = 0
    apply_position(e.state, {'preset': 'semisitting', 'bodyTiltDeg': 45})
    origin = cell(e, (2, 3, 3)); lower = cell(e, (2, 2, 3))
    e.state['fluid'] = {str(origin): 1., str(lower): .9}; e.state['debris'] = {str(origin): .1}
    replay = copy.deepcopy(e)
    for _ in range(10):
        assert relax_fluid(e, .01) == relax_fluid(replay, .01)
        assert all(v <= 1 + 1e-12 for v in e.state['fluid'].values())
    assert e.state == replay.state
    assert sum(e.state['fluid'].values()) == pytest.approx(1.9)
    assert sum(e.state['debris'].values()) == pytest.approx(.1)


def test_zero_fluid_entries_do_not_starve_bounded_active_sources():
    e = engine(shape=(4, 4, 4)); e.scene.labels[:] = 0
    apply_position(e.state, {'preset': 'supine'})
    origin = cell(e, (2, 2, 2)); below = cell(e, (2, 1, 2))
    e.state['fluid'] = {str(c): 0. for c in range(origin)}
    e.state['fluid'][str(origin)] = 1.
    relax_fluid(e, .25, max_cells=1)
    assert e.state['fluid'][str(below)] == 1.
    assert str(origin) not in e.state['fluid']
