"""Public-action contracts for explicitly selected revised research tools.

Coupons verify mechanism semantics and conservative cell/fluid accounting;
they do not calibrate device geometry, tissue properties or clinical effects.
"""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene


def coupon(material='tumor', points=((8, 8, 10),), *, properties=None,
           configured=True, obstacles=()):
    labels = np.zeros((32, 32, 24), np.uint16)
    for point in points:
        labels[point] = 1
    for point in obstacles:
        labels[point] = 2
    regions = [{'id': 1, 'name': 'Engineering coupon', 'material': material,
                'target': material == 'tumor'}]
    if obstacles:
        regions.append({'id': 2, 'name': 'Retained obstacle', 'material': 'bone'})
    props = {'tissuePhysicsV2': default_profile(), 'attachments': [],
             **copy.deepcopy(properties or {})}
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), regions,
        {'kind': 'generated revised-tool mechanism coupon'}, props))
    if configured:
        accepted(engine, 'configure', duration=.1)
    return engine


def pose(z, x=8., y=8.):
    return {'position': [x, y, z], 'quaternion': [0., 0., 0., 1.]}


def action(operation, *, slot='left', duration=.2, **kwargs):
    return {'slot': slot, 'operation': operation, 'duration': duration, **kwargs}


def accepted(engine, operation, **kwargs):
    result = engine.step(action(operation, **kwargs))
    assert result['status'] == 'accepted', result
    return result


def prepare(engine, tool, *, jaw=0., radius=1.5, **parameters):
    accepted(engine, 'exchange', instrument=tool, duration=.1)
    accepted(engine, 'place', pose=pose(5.), parameters={
        'radiusMm': radius, 'jawOpeningMm': jaw, **parameters}, duration=.1)


def cell(engine, point):
    return int(np.ravel_multi_index(point, engine.scene.shape))


def test_configure_is_explicit_replayable_and_preserves_source():
    e = coupon(configured=False)
    source = e.scene.labels.copy()
    properties = copy.deepcopy(e.scene.properties)
    assert 'toolMechanicsRevision' not in e.state and 'cusaV3' not in e.state
    accepted(e, 'configure', duration=.1)
    assert e.state['toolMechanicsRevision'] == 1 and 'cusaV3' in e.state
    np.testing.assert_array_equal(e.scene.labels, source)
    assert e.scene.properties == properties
    assert e.replay(e.actions).state_id == e.state_id
    assert SurgicalEngine.restore(e.scene, e.snapshot()).state_id == e.state_id


@pytest.mark.parametrize('occupied_hand,operation', [
    ('left', 'move'), ('right', 'move'), ('left', 'activate'), ('right', 'activate')])
def test_configure_requires_both_hands_withdrawn_and_inactive(occupied_hand, operation):
    e = coupon(configured=False)
    if operation == 'move':
        accepted(e, 'move', slot=occupied_hand, pose=pose(3., x=2.), duration=.1)
    else:
        accepted(e, 'activate', slot=occupied_hand, duration=.1)
    before = e.state_id
    result = e.step(action('configure', duration=.1))
    assert result['status'] == 'rejected'
    assert result['reason'] == 'WITHDRAW_BOTH_TOOLS_BEFORE_SETUP'
    assert e.state_id == before and 'toolMechanicsRevision' not in e.state


def test_no_configure_keeps_legacy_cusa_and_soft_scissors_replay():
    cusa = coupon(configured=False)
    prepare(cusa, 'cusa')
    accepted(cusa, 'apply', pose=pose(8.8), parameters={'power': 0., 'duty': 0.})
    assert cusa.state['removed'] == [cell(cusa, (8, 8, 10))]
    assert 'toolMechanicsRevision' not in cusa.state and 'cusaV3' not in cusa.state
    assert cusa.replay(cusa.actions).state_id == cusa.state_id
    scissors = coupon('soft_tissue', ((8, 7, 10), (8, 8, 10), (8, 9, 10)),
                       configured=False)
    prepare(scissors, 'micro_scissors')
    accepted(scissors, 'apply', pose=pose(9.8))
    assert not scissors.state['cutEdges'] and not scissors.state['transectedEdges']
    assert scissors.replay(scissors.actions).state_id == scissors.state_id


@pytest.mark.parametrize('parameters', [{'power': 0.}, {'duty': 0.}])
def test_revised_cusa_zero_activation_cannot_erase_or_advance(parameters):
    e = coupon()
    prepare(e, 'cusa')
    result = e.step(action('apply', pose=pose(12.), parameters=parameters))
    assert result['status'] == 'partial'
    assert result['reason'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
    assert not e.state['removed'] and not e.state['cusaV3']['progressMm3']
    assert e.state['slots']['left']['position'][2] < 10.
    assert e.state['cusaV3']['inputCapacityMm3'] == 0.


def test_revised_cusa_pending_contact_then_clearance_and_preview_parity():
    e = coupon()
    prepare(e, 'cusa')
    stroke = action('apply', pose=pose(12.), duration=.05)
    before = e.state_id
    preview = e.preview(stroke)
    assert e.state_id == before
    result = e.step(stroke)
    assert result['stateId'] == preview['stateId']
    assert result['status'] == 'partial' and not e.state['removed']
    assert sum(e.state['cusaV3']['progressMm3'].values()) > 0.
    for _ in range(10):
        result = e.step(action('apply', pose=pose(12.), duration=.1))
        if e.state['removed']:
            break
    assert result['status'] == 'accepted'
    assert e.state['removed'] == [cell(e, (8, 8, 10))]
    state = e.state['cusaV3']
    assert state['aspiratedSolidMm3'] == 1.
    assert state['inputCapacityMm3'] == pytest.approx(
        state['aspiratedSolidMm3'] + sum(state['progressMm3'].values())
        + state['unusedCapacityMm3'] + state['discardedCapacityMm3'])


@pytest.mark.parametrize('zero', [{'rpm': 0.}, {'forceN': 0.}, {'duty': 0.}])
def test_drill_zero_work_stalls_at_bone_without_debris(zero):
    e = coupon('bone')
    prepare(e, 'cranial_drill')
    result = e.step(action('apply', pose=pose(12.), parameters=zero))
    assert result['status'] == 'partial' and result['reason'] == 'BONE_CUTTING_IN_PROGRESS'
    assert not e.state['removed'] and not e.state['debris']
    assert e.state['drillingV2']['inputEnergyJ'] == 0.


def test_drill_positive_work_retains_partial_then_releases_conserved_debris():
    e = coupon('bone')
    prepare(e, 'cranial_drill')
    result = e.step(action('apply', pose=pose(12.), duration=.1,
                           parameters={'rpm': 1000., 'forceN': .1}))
    assert result['status'] == 'partial' and not e.state['removed']
    assert sum(e.state['drillingV2']['workJ'].values()) > 0.
    accepted(e, 'apply', pose=pose(12.), duration=1.,
             parameters={'rpm': 40000., 'forceN': 1.})
    assert len(e.state['removed']) == 1 and e.state['debrisMm3'] == 1.
    d = e.state['drillingV2']
    assert d['inputEnergyJ'] == pytest.approx(d['cuttingWorkJ'] + d['unusedEnergyJ'])


@pytest.mark.parametrize('tool,material', [('scalpel', 'soft_tissue'),
    ('dural_scissors', 'membrane'), ('micro_scissors', 'soft_tissue'),
    ('micro_scissors', 'tumor')])
def test_closed_cutting_tools_sever_bonds_and_retain_source_cells(tool, material):
    e = coupon(material, ((8, 7, 10), (8, 8, 10), (8, 9, 10)))
    prepare(e, tool)
    accepted(e, 'apply', pose=pose(9.8))
    assert e.state['cutEdges'] and not e.state['removed']
    assert all(e.scene.labels.flat[i] == 1 for edge in e.state['cutEdges'] for i in edge)


@pytest.mark.parametrize('tool,material', [('dural_scissors', 'membrane'),
    ('micro_scissors', 'soft_tissue'), ('micro_scissors', 'tumor'),
    ('micro_scissors', 'vessel')])
def test_open_scissors_have_no_cutting_effect(tool, material):
    e = coupon(material, ((8, 7, 10), (8, 8, 10), (8, 9, 10)))
    prepare(e, tool, jaw=2., forceN=0.)
    accepted(e, 'apply', pose=pose(9.8))
    assert not e.state['cutEdges'] and not e.state['transectedEdges']
    assert not e.state['removed']


def test_micro_scissors_vessel_cut_creates_wound_without_solid_removal():
    e = coupon('vessel', ((8, 7, 10), (8, 8, 10), (8, 9, 10)))
    prepare(e, 'micro_scissors', forceN=0.)
    accepted(e, 'apply', pose=pose(9.8))
    assert e.state['transectedEdges'] and e.state['vesselsV2']['wounds']
    assert not e.state['removed']
    v = e.state['vesselsV2']
    assert v['initialMm3'] + v['refilledMm3'] == pytest.approx(
        v['remainingMm3'] + v['generatedMm3'])


@pytest.mark.parametrize('tool,kind,other', [
    ('periosteal_elevator', 'periosteal', 'soft'), ('dissector', 'soft', 'periosteal')])
def test_separation_tools_release_only_declared_attachment_kind(tool, kind, other):
    point = (8, 8, 10)
    identifier = int(np.ravel_multi_index(point, (32, 32, 24)))
    e = coupon('soft_tissue', properties={'attachments': [
        {'id': 'matching', 'kind': kind, 'cells': [identifier]},
        {'id': 'other', 'kind': other, 'cells': [identifier]}]})
    prepare(e, tool)
    accepted(e, 'apply', pose=pose(9.8))
    assert e.state['separatedAttachments'] == ['matching']
    assert not e.state['removed']


def test_craniotome_removes_kerf_and_accounts_debris():
    e = coupon('bone', ((8, 7, 10), (8, 8, 10), (8, 9, 10)))
    prepare(e, 'craniotome')
    accepted(e, 'apply', pose=pose(9.8))
    assert e.state['removed'] == [cell(e, (8, 8, 10))]
    assert e.state['debrisMm3'] == e.scene.voxel_mm3
    assert len(e.state['flapComponents']) == 2


def test_retractor_displaces_retained_tissue_and_release_relaxes():
    points = tuple((x, y, 10) for x in range(7, 10) for y in range(7, 10))
    e = coupon('soft_tissue', points)
    prepare(e, 'brain_retractor', radius=2.)
    accepted(e, 'hold', pose=pose(9.8))
    accepted(e, 'move', pose=pose(9.8, x=8.3), duration=.4)
    before = max(np.linalg.norm(v) for v in e.state['deformation'].values())
    assert before > 0. and not e.state['removed']
    accepted(e, 'release', duration=.2)
    after = max(np.linalg.norm(v) for v in e.state['deformation'].values())
    assert after < before and not e.state['holds']


@pytest.mark.parametrize('jaw,power,expect_heat', [(2., 1., False),
    (0., 0., False), (0., 1., True)])
def test_bipolar_requires_closed_local_contact_and_positive_power(jaw, power, expect_heat):
    e = coupon('vessel', ((8, 8, 10), (9, 8, 10)))
    prepare(e, 'bipolar_forceps', jaw=jaw, forceN=0., power=power)
    accepted(e, 'apply', pose=pose(9.8))
    energy = sum(e.state['thermal'].values())
    assert (energy > 0.) == expect_heat
    # Adjacent vessel center is in the spherical tip but outside the narrow
    # two-electrode contact convention. It must receive no energy proxy.
    assert e.state['thermal'].get(str(cell(e, (9, 8, 10))), 0.) == 0.
    assert not e.state['removed']


@pytest.mark.parametrize('tool', ['frazier_suction', 'irrigation'])
def test_field_tools_zero_flow_have_no_material_or_fluid_effect(tool):
    e = coupon('soft_tissue')
    prepare(e, tool, flowMm3PerSecond=0., forceN=0.)
    accepted(e, 'apply', pose=pose(7.5))
    assert e.state['fluidAddedMm3'] == 0. and e.state['fluidRemovedMm3'] == 0.
    assert not e.state['removed'] and not e.state['debris']


def test_irrigation_and_suction_manage_field_without_removing_tissue():
    e = coupon('soft_tissue')
    prepare(e, 'irrigation', flowMm3PerSecond=4., depthMm=1.)
    accepted(e, 'apply', pose=pose(6.), duration=1.)
    assert e.state['fluidAddedMm3'] == pytest.approx(4.)
    accepted(e, 'withdraw')
    prepare(e, 'frazier_suction', flowMm3PerSecond=10., radius=3.)
    accepted(e, 'apply', pose=pose(6.), duration=1.)
    assert e.state['fluidRemovedMm3'] > 0.
    assert e.state['fluidAddedMm3'] == pytest.approx(e.state['fluidRemovedMm3']
        + e.state['fluidEscapedMm3'] + sum(e.state['fluid'].values()))
    assert not e.state['removed']


@pytest.mark.parametrize('tool,probe', [('cortical_probe', 'cortical'),
    ('subcortical_probe', 'subcortical')])
def test_mapping_tools_report_only_supplied_matching_evidence(tool, probe):
    identifier = int(np.ravel_multi_index((8, 8, 10), (32, 32, 24)))
    evidence = {'id': 'known-site', 'cell': identifier, 'probe': probe,
                'result': 'synthetic response', 'uncertainty': 'coupon only'}
    e = coupon('soft_tissue', properties={'functionalEvidence': [evidence]})
    prepare(e, tool)
    accepted(e, 'apply', pose=pose(9.8), parameters={
        'amplitudeMa': 2., 'frequencyHz': 60., 'pulseWidthUs': 100.})
    accepted(e, 'inspect')
    assert e.state['observations'][-1]['evidence'] == [evidence]
    assert not e.state['thermal'] and not e.state['removed']


def test_forceps_open_jaws_do_not_grasp():
    e = coupon()
    prepare(e, 'tumor_forceps', jaw=2.)
    accepted(e, 'grasp', pose=pose(9.8))
    assert not e.state['holds'] and not e.state['removed']


@pytest.mark.parametrize('duration', [.05, .2, 1.])
def test_forceps_preserve_in_field_then_checked_move_and_single_extraction(duration):
    # Distant anatomy defines an exterior beyond z=5; moving the fragment from
    # z=10 to z=8 is free transport but still in the source anatomical field.
    e = coupon(obstacles=((20, 20, 5), (20, 20, 15)))
    prepare(e, 'tumor_forceps')
    accepted(e, 'grasp', pose=pose(9.8))
    tumor = cell(e, (8, 8, 10))
    before = e.state_id
    result = e.step(action('extract', duration=duration))
    assert result['status'] == 'rejected'
    assert result['reason'] == 'MOVE_FRAGMENT_THROUGH_EXIT_BEFORE_EXTRACTION'
    assert e.state_id == before and not e.state['removed']
    accepted(e, 'move', pose=pose(7.8))
    before = e.state_id
    result = e.step(action('extract', duration=duration))
    assert result['status'] == 'rejected'
    assert result['reason'] == 'FRAGMENT_NOT_AT_SOURCE_EXTERIOR_EXIT'
    assert e.state_id == before and not e.state['removed']
    assert e.occupancy()[cell(e, (8, 8, 8))] == tumor + 1
    accepted(e, 'move', pose=pose(3.8))
    request = action('extract', duration=duration)
    before = e.state_id
    preview = e.preview(request)
    assert e.state_id == before
    result = e.step(request)
    assert result['status'] == 'accepted' and result['stateId'] == preview['stateId']
    assert e.state['removed'] == [tumor] and not e.state['holds']
    assert e.state['evacuatedForcepsMm3'] == 1.
    assert sum(event['kind'] == 'detached_extraction' for event in result['events']) == 1
    assert e.replay(e.actions).state_id == e.state_id


@pytest.mark.parametrize('breakable', [False, True])
def test_forceps_never_auto_break_declared_attachment(breakable):
    identifier = int(np.ravel_multi_index((8, 8, 10), (32, 32, 24)))
    attachment = {'id': 'intact', 'kind': 'tumor', 'cells': [identifier],
                  'breakable': breakable}
    e = coupon(properties={'attachments': [attachment]})
    prepare(e, 'tumor_forceps')
    accepted(e, 'grasp', pose=pose(9.8))
    for request in [action('move', pose=pose(7.8)), action('extract')]:
        before = e.state_id
        result = e.step(request)
        assert result['status'] == 'rejected', result
        assert result['reason'].startswith('SEPARATE_ATTACHMENTS_BEFORE_')
        assert e.state_id == before and not e.state['separatedAttachments']
        assert not e.state['removed']


def test_forceps_checked_transport_stops_before_retained_obstacle():
    e = coupon(obstacles=((12, 8, 10),))
    prepare(e, 'tumor_forceps')
    accepted(e, 'grasp', pose=pose(9.8))
    result = e.step(action('move', pose=pose(9.8, x=14.), duration=1.))
    assert result['status'] in ('partial', 'rejected')
    assert result['reason'] in ('DISPLACEMENT_COLLISION', 'UNSUPPORTED_MATERIAL',
                                'RETAINED_TISSUE_COLLISION')
    assert not e.state['removed'] and e.state['holds']['left']['cells']
    assert e.state['slots']['left']['position'][0] < 12.
    assert e.occupancy()[cell(e, (12, 8, 10))] == cell(e, (12, 8, 10)) + 1


def test_forceps_release_retains_displaced_fragment_then_regrasp_accumulates_transport():
    e = coupon(obstacles=((20, 20, 5), (20, 20, 15)))
    prepare(e, 'tumor_forceps')
    accepted(e, 'grasp', pose=pose(9.8))
    accepted(e, 'move', pose=pose(7.8))
    source = cell(e, (8, 8, 10))
    destination = cell(e, (8, 8, 8))
    accepted(e, 'release')
    assert not e.state['holds'] and not e.state['removed']
    assert e.occupancy()[source] == 0
    assert e.occupancy()[destination] == source + 1
    assert [source, destination] in e.view()['moved']
    accepted(e, 'inspect', duration=1.)
    assert e.occupancy()[destination] == source + 1
    resumed = SurgicalEngine.restore(e.scene, e.snapshot())
    assert resumed.state_id == e.state_id
    accepted(e, 'grasp')
    assert e.state['holds']['left']['cells'] == [source]
    assert not e.state.get('detachedFragments')
    accepted(e, 'move', pose=pose(3.8))
    assert e.occupancy()[cell(e, (8, 8, 4))] == source + 1
    accepted(e, 'extract', duration=1.)
    assert e.state['removed'] == [source]
    assert e.state['evacuatedForcepsMm3'] == 1.
    assert e.replay(e.actions).state_id == e.state_id
    for request in e.actions[len(resumed.actions):]:
        assert resumed.step(request)['status'] == 'accepted'
    assert resumed.state_id == e.state_id


def test_forceps_transfers_active_lattice_source_to_one_rigid_owner_and_keeps_release():
    e = coupon()
    prepare(e, 'brain_retractor')
    accepted(e, 'hold', pose=pose(9.8))
    accepted(e, 'move', pose=pose(9.8, x=8.3))
    accepted(e, 'release')
    accepted(e, 'withdraw')
    prepare(e, 'tumor_forceps')
    source = cell(e, (8, 8, 10))
    mapped_before = dict(e.view()['moved'])[source]
    # Retraction leaves the snapped source at z=11. Rigid purchase must
    # preserve that destination instead of adding a second source copy.
    assert mapped_before == cell(e, (8, 8, 11))
    accepted(e, 'grasp', pose=pose(10.8))
    assert np.count_nonzero(np.asarray(e.occupancy()) == source + 1) == 1
    assert e.occupancy()[mapped_before] == source + 1
    assert str(source) not in e.state['deformation']
    assert source not in e.state['deformationModel']['active']
    accepted(e, 'move', pose=pose(8.8))
    destination = cell(e, (8, 8, 9))
    assert e.occupancy()[destination] == source + 1
    assert np.count_nonzero(np.asarray(e.occupancy()) == source + 1) == 1
    assert [pair for pair in e.view()['moved'] if pair[0] == source] == [[source, destination]]
    accepted(e, 'release')
    accepted(e, 'inspect', duration=1.)
    assert e.occupancy()[destination] == source + 1
    assert np.count_nonzero(np.asarray(e.occupancy()) == source + 1) == 1
    assert str(source) not in e.state['deformation']
    accepted(e, 'grasp')
    accepted(e, 'move', pose=pose(5.8))
    accepted(e, 'extract')
    assert e.state['removed'] == [source] and e.state['evacuatedForcepsMm3'] == 1.
    assert e.replay(e.actions).state_id == e.state_id


@pytest.mark.parametrize('tool,material,operation,field', [
    ('tumor_forceps', 'tumor', 'grasp', 'holds'),
    ('micro_scissors', 'soft_tissue', 'apply', 'cutEdges'),
    ('scalpel', 'soft_tissue', 'apply', 'cutEdges'),
    ('brain_retractor', 'soft_tissue', 'hold', 'holds'),
    ('craniotome', 'bone', 'apply', 'removed')])
def test_revised_working_parts_do_not_reach_ahead_with_old_tip_sphere(tool, material, operation, field):
    points = ((8, 7, 10), (8, 8, 10), (8, 9, 10))
    e = coupon(material, points)
    prepare(e, tool)
    # The old 1.5-mm sphere reaches the z=10 voxel from z=8.8, while the
    # actual distal jaw/blade capsules stop before the z=9.5 cell face.
    accepted(e, operation, pose=pose(8.8))
    assert not e.state[field] and not e.state['removed']


def test_stationary_active_forceps_do_not_grasp_ahead_of_jaw_capsule():
    e = coupon()
    prepare(e, 'tumor_forceps')
    accepted(e, 'apply', pose=pose(8.8))
    accepted(e, 'activate')
    accepted(e, 'inspect', slot='right', duration=1.)
    assert not e.state['holds'] and not e.state['removed']
    # The same instrument does grasp after its recorded working part reaches
    # the tissue, so the negative assertion is a contact bound, not disablement.
    accepted(e, 'move', pose=pose(9.8))
    assert e.state['holds']['left']['cells'] == [cell(e, (8, 8, 10))]


@pytest.mark.parametrize('blocked', [False, True])
def test_jaw_articulation_observed_frames_retain_actual_aperture_and_canonical_replay(tmp_path, blocked):
    from resectionlab.surgical_frame_replay import SurgicalFrameCache
    from resectionlab.surgical_replay_cache import SurgicalReplayCache

    e = coupon(obstacles=((11, 8, 5),) if blocked else ())
    prepare(e, 'tumor_forceps')
    position = len(e.actions)
    root_pose = e.state['slots']['left']['position'][:]
    result = e.step(action('apply', parameters={'jawOpeningMm': 8.}, duration=1.2))
    actual = e.state['slots']['left']['parameters']['jawOpeningMm']
    if blocked:
        assert result['status'] == 'partial' and result['reason'] == 'UNSUPPORTED_MATERIAL'
        assert 0. < actual < 8.
    else:
        assert result['status'] == 'accepted' and actual == 8.
    assert e.state['slots']['left']['position'] == root_pose
    saved = e.snapshot()
    frames = SurgicalFrameCache(directory=tmp_path)
    replay = SurgicalReplayCache()
    _, _, first = frames.seek(e, saved, position, 0, replay)
    apertures = []
    for index in range(first['frameCount']):
        observed, _, info = frames.seek(e, saved, position, index, replay)
        slot = observed.observe()['slots']['left']
        jaw = slot['parameters']['jawOpeningMm']
        apertures.append(jaw)
        assert slot['position'] == root_pose
        working = [part for part in slot['contactGeometry'] if part['part'] == 'working']
        assert sorted(part['a'][0] for part in working) == pytest.approx([-jaw / 2, jaw / 2])
        assert info['frameStateId'] == observed.state_id
    assert len(apertures) > 2
    assert apertures == sorted(apertures) and apertures[0] == 0.
    assert apertures[-1] == actual
    assert any(0. < jaw < actual for jaw in apertures)
    assert observed.snapshot() == saved
    assert e.snapshot() == saved and e.replay(e.actions).state_id == e.state_id


def test_forceps_nonrigid_purchase_rejects_without_erasing_lattice_state():
    e = coupon(points=tuple((x, 8, 10) for x in range(5, 16)))
    prepare(e, 'brain_retractor', radius=.6)
    accepted(e, 'hold', pose=pose(9.4))
    accepted(e, 'move', pose=pose(9.4, x=10.5), duration=1.)
    # These short, legal control actions preserve the just-created strain;
    # otherwise its small local patch relaxes before forceps purchase.
    accepted(e, 'release', duration=.001)
    accepted(e, 'withdraw', duration=.001)
    accepted(e, 'exchange', instrument='tumor_forceps', duration=.001)
    accepted(e, 'place', pose=pose(5.), duration=.001,
             parameters={'radiusMm': 1.5, 'jawOpeningMm': 2.})
    accepted(e, 'apply', pose=pose(9.8), duration=.001)
    assert len({tuple(np.rint(v).astype(int)) for v in e.state['deformation'].values()}) > 1
    lattice = copy.deepcopy(e.state['deformation'])
    result = e.step(action('grasp', duration=.001, parameters={'jawOpeningMm': 0.}))
    # Open-jaw interpolation is an accepted prefix; purchase first becomes
    # possible during closure, when the unsupported nonrigid transform stops it.
    assert result['status'] == 'partial'
    assert result['reason'] == 'NONRIGID_FRAGMENT_GRASP_UNSUPPORTED'
    assert .5 < e.state['slots']['left']['parameters']['jawOpeningMm'] < 2.
    assert e.state['deformation'] == lattice and not e.state['holds']
    accepted(e, 'inspect', duration=.001, parameters={'jawOpeningMm': 0.})
    assert e.state['slots']['left']['parameters']['jawOpeningMm'] == pytest.approx(0., abs=1e-12)
    before = e.state_id
    lattice = copy.deepcopy(e.state['deformation'])
    result = e.step(action('grasp', duration=.001))
    assert result['status'] == 'rejected'
    assert result['reason'] == 'NONRIGID_FRAGMENT_GRASP_UNSUPPORTED'
    assert e.state_id == before and e.state['deformation'] == lattice
    assert not e.state['holds'] and not e.state['removed']
    assert e.replay(e.actions).state_id == e.state_id


@pytest.mark.parametrize('tool,probe', [('cortical_probe', 'subcortical'),
    ('subcortical_probe', 'cortical')])
def test_mapping_tools_do_not_convert_other_probe_evidence_into_negative_mapping(tool, probe):
    identifier = int(np.ravel_multi_index((8, 8, 10), (32, 32, 24)))
    e = coupon('soft_tissue', properties={'functionalEvidence': [
        {'id': 'other-probe-site', 'cell': identifier, 'probe': probe,
         'result': 'synthetic response', 'uncertainty': 'coupon only'}]})
    prepare(e, tool)
    accepted(e, 'apply', pose=pose(9.8))
    result = e.state['observations'][-1]
    assert result['result'] == 'unknown' and result['evidence'] == []
    assert result['coverage'] == 'unknown'


@pytest.mark.parametrize('tool,material', [('cusa', 'bone'),
    ('cranial_drill', 'tumor'), ('dural_scissors', 'bone'),
    ('micro_scissors', 'bone'), ('tumor_forceps', 'soft_tissue')])
def test_incompatible_material_stops_revised_tool_before_removal(tool, material):
    e = coupon(material)
    prepare(e, tool)
    result = e.step(action('apply', pose=pose(10.)))
    assert result['status'] in ('partial', 'rejected')
    assert result['reason'] == 'UNSUPPORTED_MATERIAL'
    assert not e.state['removed'] and not e.state['holds']
    assert not e.state['cutEdges'] and not e.state['transectedEdges']
