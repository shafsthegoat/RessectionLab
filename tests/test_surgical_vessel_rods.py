import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix

from resectionlab.core import CaseData, SourceRef
from resectionlab import surgical_deformation_v2 as tissue
from resectionlab import surgical_glioma_materials as materials
from resectionlab import surgical_vessel_rods as rods
from resectionlab import surgical_vessel_surface as surfaces
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import from_case
from resectionlab import native_backend, surgical_wall_sweep, surgical_vessel_vertex_motion as vertex_motion
from resectionlab.surgical_wall_sweep_reference import WallSweepReference


def _fixture(*, guarded=False, vertex_transport=False, section_mechanics=False, conditioned=False,
             section_overrides=None, vessel_contact=False, parallel=False, soft_contact=False, soft_bank=False):
    shape = (35, 35, 30)
    spacing = .4
    affine = np.diag([spacing, spacing, spacing, 1.])
    grid = np.moveaxis(np.indices(shape), 0, -1)
    sites = [(i, 15, 15) for i in range(11, 24)]
    other = [(i, 15, 19) for i in range(11, 24)] if parallel else [(i, 23, 15) for i in range(11, 14)]
    support = {sites[i]: (sites[i][0], 15, 17) for i in (0, 6, 12)}
    if parallel:
        support.update({other[i]: (other[i][0], 15, 21) for i in (0, 6, 12)})
    radius = .77 if parallel else .66
    distances = np.minimum(*(np.linalg.norm(grid - np.clip(grid, chain[0], chain[-1]), axis=-1)
                             for chain in (sites, other)))
    scores = np.clip(.5 + radius - distances * spacing, 0., 1.).astype(np.float32)
    vessels = scores >= .5
    brain = np.zeros(shape, dtype=bool)
    for cell in support.values():
        brain[cell] = True
    if soft_bank:
        brain[11:24, 13:18, 17:21] = True
    props = materials.generated_vascular_properties(shape, sites, other, support, 'native-score-rod',
                                                    vessel_motion_model=rods.MODEL)
    props[materials.PROPERTY][rods.GEOMETRY] = {
        'source': 'Analytic score-bound capsule',
        'sites': {str(np.ravel_multi_index(cell, shape)): [*(np.asarray(cell) * spacing), radius]
                  for cell in sites + other},
    }
    if guarded:
        props[materials.PROPERTY][surgical_wall_sweep.PROPERTY] = surgical_wall_sweep.POLICY
    if vertex_transport:
        props[materials.PROPERTY][vertex_motion.PROPERTY] = vertex_motion.POLICY
    if section_mechanics:
        section = rods.section_profile(vessel_contact=vessel_contact, soft_contact=soft_contact)
        if section_overrides:
            section.update(section_overrides)
        props[materials.PROPERTY][rods.SECTION_MECHANICS] = section
    case = CaseData('native-score-rod', scores, {'brain': brain, 'vessels': vessels}, affine,
                    (SourceRef('fixture', 'synthetic://native-score-rod', provenance='simulated'),),
                    metadata={'anatomy_layers': {'brain': {'kind': 'brain'}, 'vessels': {'kind': 'vessels'}},
                              **({surfaces.GEOMETRY_PROPERTY: surfaces.CONDITIONED_GEOMETRY} if conditioned else {}),
                              'surgical_properties': {'tissuePhysicsV2': default_profile(), **props}},
                    vascular_probabilities={'vessels': scores})
    return SurgicalEngine(from_case(case))


def _action(operation, *, z=None, dt=.1, **fields):
    return {
        'slot': 'left', 'operation': operation, 'duration': dt,
        **({'pose': {'position': [6.8, 6., z], 'quaternion': [0, 0, 0, 1]}} if z is not None else {}),
        **fields,
    }


@pytest.mark.parametrize(('guarded', 'vertex_transport', 'section_mechanics'),
                         [(False, False, False), (True, False, False), (True, True, False), (True, True, True)])
def test_native_wall_contact_tracks_passive_dissection_and_verified_replay(guarded, vertex_transport, section_mechanics):
    engine = _fixture(guarded=guarded, vertex_transport=vertex_transport, section_mechanics=section_mechanics,
                      conditioned=section_mechanics)
    actions = [_action('exchange', instrument='dissector'), _action('place', z=4.),
               _action('move', z=5.05, dt=.4)]
    for action in actions:
        receipt = engine.step(action)
        assert receipt['status'] == 'accepted', receipt
    view = engine.view()[rods.FIELD]
    assert view['geometryModeByLabel'] == {'2': 'native-probability-wall'}
    assert view['contactedCells']['left']
    movements = np.array([np.subtract(row['positionMm'], row['restPositionMm']) for row in view['sites']])
    peak = np.linalg.norm(movements, axis=1).max()
    assert .05 < peak < .7
    assert np.count_nonzero(np.linalg.norm(movements, axis=1) > .005) >= 3
    assert any(row['radiusMm'] < row['restRadiusMm'] for row in view['sites'])
    # Engine contact uses the exact displayed native-wall vertex positions.
    frame = rods._accepted_frames(engine, rods.definitions(engine))[0]
    wall, offsets, *_ = frame
    shifts = wall.vertex_motion.offsets if vertex_transport else offsets[wall.source.owners]
    vertices = wall.source.vertices + shifts
    close = np.flatnonzero((np.abs(vertices[:, 0] - 6.8) < .3) & (vertices[:, 2] < 6.))
    assert len(close)
    clip = close[np.argmin(vertices[close, 2])]
    vertex = vertices[clip]
    owner = int(wall.source.cells[wall.source.owners[clip]])
    if vertex_transport:
        packet = view['vertexDisplacementsByLabel']['2']
        assert packet['geometryBinding'] == wall.source.geometry_binding
        assert packet['offsetsMm'][packet['indices'].index(clip)] == shifts[clip].tolist()
        assert packet['vertexCount'] == len(wall.source.vertices)
        assert not np.array_equal(shifts, offsets[wall.source.owners])
    else:
        assert view['surfaceDisplacements'][str(owner)] == offsets[wall.source.owners[clip]].tolist()
    assert owner in engine.touched(vertex, vertex, 0.)
    assert owner not in engine.touched(vertex - [0., 0., .12], vertex - [0., 0., .12], 0.)
    clone = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert clone.view()[rods.FIELD] == view
    withdraw = _action('withdraw', z=4., dt=.35)
    receipt = engine.step(withdraw)
    assert receipt['status'] == 'accepted', receipt
    assert clone.step(withdraw) == receipt
    released = engine.view()[rods.FIELD]
    assert max(np.linalg.norm(np.subtract(row['positionMm'], row['restPositionMm'])) for row in released['sites']) < peak


def test_support_removal_changes_motion_and_cut_removes_the_bond():
    engine = _fixture()
    profile = engine.scene.properties[materials.PROPERTY]
    mid_site, center_support = profile['vesselAttachments'][1]
    for action in (_action('exchange', instrument='dissector'), _action('place', z=4.),
                   _action('move', z=5.05, dt=.35)):
        assert engine.step(action)['status'] == 'accepted'
    before = engine.view()[rods.FIELD]
    position = lambda view: next(s['positionMm'] for s in view['sites'] if s['sourceCell'] == mid_site)
    assert center_support in next(s['activeSupportCells'] for s in before['sites'] if s['sourceCell'] == mid_site)
    engine.state['removed'] = [center_support]
    tissue.tick(engine, .15)
    after = engine.view()[rods.FIELD]
    assert next(s['activeSupportCells'] for s in after['sites'] if s['sourceCell'] == mid_site) == []
    assert .01 < np.linalg.norm(np.subtract(position(before), position(after))) < .3
    assert len(after['segments']) == len(before['segments'])
    cut = engine.scene.properties['vascularGraphV1']['edges'][0]['sourceFace']
    from resectionlab import surgical_vascular_graph as graph

    graph.damage(engine, [cut])
    tissue.tick(engine, .1)
    split = engine.view()[rods.FIELD]
    assert {s['sourceCell'] for s in split['sites']} == {s['sourceCell'] for s in before['sites']}
    assert len(split['segments']) == len(after['segments']) - 1
    assert cut not in [s['sourceCells'] for s in split['segments']]
    assert split['geometryModeByLabel'] == {'2': 'edited-native-wall'}
    edited = split['editedWallsByLabel']['2']
    triangles = np.array(edited['triangles'])
    pairs = np.sort(np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, ::2])), axis=1)
    _, rows = np.unique(pairs, axis=0, return_counts=True)
    assert np.all(rows == 2)
    # The newly opened cut is also open to a zero-radius collision witness.
    cut_center = np.mean(engine.scene.points(cut), axis=0)
    assert not engine.touched(cut_center, cut_center, 0.)


def test_section_mechanics_responds_to_loaded_support_release_and_drives_lumen_flow():
    from resectionlab import surgical_vascular_graph as graph

    engine = _fixture(guarded=True, vertex_transport=True, section_mechanics=True, conditioned=True)
    mid_site, support = engine.scene.properties[materials.PROPERTY]['vesselAttachments'][1]

    def site(subject):
        return next(row for row in subject.view()[rods.FIELD]['sites'] if row['sourceCell'] == mid_site)

    def measure(subject):
        row = site(subject)
        delta = np.asarray(row['positionMm']) - row['restPositionMm']
        return float(np.linalg.norm(delta)), row['radiusMm']

    unloaded = engine.clone()
    unloaded.state['removed'] = [support]
    rods.tick(unloaded, .15)
    assert site(unloaded)['activeSupportCells'] == []
    assert measure(unloaded) == (0., site(unloaded)['restRadiusMm'])

    for action in (_action('exchange', instrument='dissector'), _action('place', z=4.),
                   _action('move', z=5.05, dt=.4)):
        assert engine.step(action)['status'] == 'accepted'
    assert engine.view()[rods.FIELD]['sectionModel'] == rods.SECTION_MODEL
    loaded_delta, loaded_radius = measure(engine)
    assert .1 < loaded_delta < .4
    assert .9 * site(engine)['restRadiusMm'] < loaded_radius < site(engine)['restRadiusMm']
    row = engine.state[graph.PROPERTY]['edges']['artery']
    assert row['acceptedLumenResistanceRatio'] > row['stretchResistanceRatio'] >= 1.
    edge = engine.scene.properties[graph.PROPERTY]['edges'][0]
    assert graph._edge_conductance(edge, row) < edge['conductanceMm3PerPaS']

    held, released = engine.clone(), engine.clone()
    released.state['removed'] = [support]
    for subject in (held, released):
        rods.tick(subject, .15)
    assert site(held)['activeSupportCells'] == [support]
    assert site(released)['activeSupportCells'] == []
    assert .02 < measure(released)[0] - measure(held)[0] < .12

    restored = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert restored.view()[rods.FIELD] == engine.view()[rods.FIELD]
    withdraw = _action('withdraw', z=4., dt=.35)
    assert restored.step(withdraw) == engine.step(withdraw)
    recoil, recovered_radius = measure(restored)
    assert recoil < loaded_delta
    assert loaded_radius < recovered_radius < site(engine)['restRadiusMm']
    assert restored.state[graph.PROPERTY]['edges']['artery']['acceptedLumenResistanceRatio'] < row['acceptedLumenResistanceRatio']


def test_section_profile_requires_vertex_bound_swept_wall():
    with pytest.raises(ValueError, match='VESSEL_SECTION_MECHANICS_REQUIRES_BOUNDED_SWEPT_SOURCE_GEOMETRY'):
        _fixture(guarded=True, section_mechanics=True)


def test_sliding_tool_moves_contact_tangentially_with_bounded_friction():
    shifts = []
    for friction in (0., .15):
        engine = _fixture(guarded=True, vertex_transport=True, section_mechanics=True, conditioned=True,
                          section_overrides={'frictionCoefficient': friction})
        slide = _action('move', z=5.05, dt=.18)
        slide['pose']['position'][0] += .6
        for action in (_action('exchange', instrument='dissector'), _action('place', z=4.),
                       _action('move', z=5.05, dt=.4), slide):
            assert engine.step(action)['status'] == 'accepted'
        source, _ = engine.scene.properties[materials.PROPERTY]['vesselAttachments'][1]
        site = next(row for row in engine.view()[rods.FIELD]['sites'] if row['sourceCell'] == source)
        shifts.append(np.asarray(site['positionMm']) - site['restPositionMm'])
    assert .01 < shifts[1][0] - shifts[0][0] < .05
    assert abs(shifts[1][2] - shifts[0][2]) < .03


def test_section_stiffness_sensitivity_under_the_same_tool_path():
    excursions = []
    for modulus in (1.8, 3., 5.5):
        moduli = {**rods.section_profile()['axialModulusMPa'], 'artery': modulus}
        engine = _fixture(guarded=True, vertex_transport=True, section_mechanics=True, conditioned=True,
                          section_overrides={'axialModulusMPa': moduli})
        for action in (_action('exchange', instrument='dissector'), _action('place', z=4.),
                       _action('move', z=5.05, dt=.4)):
            assert engine.step(action)['status'] == 'accepted'
        definition = rods.definitions(engine)
        position, _ = rods._current(engine, definition)
        excursions.append(np.linalg.norm(position - definition.rest, axis=1).max())
    assert excursions[0] > excursions[1] > excursions[2]
    assert excursions[2] < .92 * excursions[0]


def test_native_inter_vessel_contact_transfers_load_and_replays_without_crossing():
    control, resolved = (_fixture(guarded=True, vertex_transport=True, section_mechanics=True,
                                  conditioned=True, parallel=True, vessel_contact=enabled)
                         for enabled in (False, True))
    expected_binding = control.scene._vessel_source_walls[0].geometry_binding
    assert resolved.scene._vessel_source_walls[0].geometry_binding == expected_binding
    for engine in (control, resolved):
        for action in (_action('exchange', instrument='dissector'), _action('place', z=4.)):
            assert engine.step(action)['status'] == 'accepted'
        definition = rods.definitions(engine)
        position, radius = rods._current(engine, definition)
        np.testing.assert_array_equal(position, definition.rest)
        np.testing.assert_array_equal(radius, definition.radius)

    advance = _action('move', z=4.9, dt=.4)
    original = control.snapshot()
    rejected = control.step(advance)
    assert (rejected['status'], rejected['reasonCode']) == ('rejected', 'VESSEL_WALL_SWEEP_UNSAFE')
    assert control.snapshot() == original

    definition = rods.definitions(resolved)
    second = definition.rest[:, 2] > 7.
    second_cells = set(np.asarray(definition.cells)[second])

    def passive_excursion(engine):
        frames = rods._accepted_frames(engine, definition)
        assert surgical_wall_sweep.is_clear(engine, frames, frames, static=True)
        view = engine.view()[rods.FIELD]
        assert view['sectionModel'] == rods.SECTION_CONTACT_MODEL
        assert view['vertexDisplacementsByLabel']['2']['geometryBinding'] == expected_binding
        assert not second_cells.intersection(view['contactedCells'].get('left', ()))
        positions, radii = rods._current(engine, definition)
        assert np.min(radii / definition.radius) >= 1. - rods.MAX_COMPRESSION
        return float((positions - definition.rest)[second, 2].max())

    assert resolved.step(advance)['status'] == 'accepted'
    first_contact = passive_excursion(resolved)
    assert .10 < first_contact < .18
    assert resolved.step(_action('move', z=5.05, dt=.26))['status'] == 'accepted'
    loaded = passive_excursion(resolved)
    assert first_contact < loaded < .30

    replay = SurgicalEngine.restore(resolved.scene, json.loads(json.dumps(resolved.snapshot())), force_verify=True)
    assert replay.view()[rods.FIELD] == resolved.view()[rods.FIELD]
    for action in (_action('withdraw', z=4., dt=.35), _action('inspect', dt=.5)):
        receipt = resolved.step(action)
        assert receipt['status'] == 'accepted'
        assert replay.step(action) == receipt
    assert replay.view()[rods.FIELD] == resolved.view()[rods.FIELD]
    assert 0. <= passive_excursion(resolved) < .25 * loaded


def test_native_vessel_contact_pushes_and_releases_retained_soft_tissue():
    baseline, coupled = (_fixture(guarded=True, vertex_transport=True, section_mechanics=True,
                                  conditioned=True, vessel_contact=True, soft_contact=enabled, soft_bank=True)
                         for enabled in (False, True))
    assert np.array_equal(baseline.scene.labels, coupled.scene.labels)
    active_vessel = rods.definitions(coupled)
    held_supports = {cell for _, cell in active_vessel.supports}

    def tissue_motion(engine):
        view = engine.view()
        support = np.asarray([view['deformation'][str(cell)] for cell in view['deformationModel']['active']
                              if cell not in held_supports and str(cell) in view['deformation']]).reshape(-1, 3)
        return np.max(support[:, 2], initial=0.)

    for engine in (baseline, coupled):
        for action in (_action('exchange', instrument='dissector'), _action('place', z=4.)):
            assert engine.step(action)['status'] == 'accepted'
        assert tissue_motion(engine) == 0.
        assert engine.step(_action('move', z=5.05, dt=.4))['status'] == 'accepted'
    first = tissue_motion(coupled)
    assert tissue_motion(baseline) == 0.
    assert .015 < first < .035
    center_cell = coupled.scene.properties[materials.PROPERTY]['vesselAttachments'][1][0]
    center = active_vessel.index[center_cell]
    assert rods._current(coupled, active_vessel)[0][center, 2] < rods._current(baseline, active_vessel)[0][center, 2]
    assert coupled.step(_action('move', z=5.15, dt=.25))['status'] == 'accepted'
    loaded = tissue_motion(coupled)
    assert first < loaded < .065
    frames = rods._accepted_frames(coupled, active_vessel)
    assert surgical_wall_sweep.is_clear(coupled, frames, frames, static=True)
    assert coupled.view()[rods.FIELD]['sectionModel'] == rods.SECTION_TISSUE_CONTACT_MODEL

    replay = SurgicalEngine.restore(coupled.scene, json.loads(json.dumps(coupled.snapshot())), force_verify=True)
    assert replay.state['deformation'] == coupled.state['deformation']
    for action in (_action('withdraw', z=4., dt=.35), _action('inspect', dt=.8)):
        result = coupled.step(action)
        assert result['status'] == 'accepted'
        assert replay.step(action) == result
    assert replay.state['deformation'] == coupled.state['deformation']
    assert 0. < tissue_motion(coupled) < .35 * loaded


def test_removed_positive_reextracts_native_wall_without_erasing_untouched_triangles():
    engine = _fixture()
    wall = rods._accepted_frames(engine, rods.definitions(engine))[0][0]
    retired = int(wall.source.cells[0])
    engine.state['removed'] = [retired]
    assert not engine.touched(*([engine.scene.points([retired])[0]] * 2), 0.)
    view = engine.view()[rods.FIELD]
    assert view['geometryModeByLabel'] == {'2': 'edited-native-wall'}
    edited = view['editedWallsByLabel']['2']
    assert retired not in edited['sourceCells']
    baseline = set(map(tuple, wall.source.vertices.round(7)))
    updated = np.asarray(edited['positionsIJK']) @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
    assert len(baseline.intersection(map(tuple, updated.round(7)))) > .95 * len(baseline)


@pytest.mark.parametrize(('guarded', 'vertex_transport', 'section_mechanics'),
                         [(False, False, False), (True, False, False), (True, True, False), (True, True, True)])
def test_public_scissors_transect_score_wall_and_replay_the_same_cap(guarded, vertex_transport, section_mechanics):
    engine = _fixture(guarded=guarded, vertex_transport=vertex_transport, section_mechanics=section_mechanics,
                      conditioned=section_mechanics)
    q = Rotation.from_rotvec([0, 0, np.pi / 2]).as_quat().tolist()
    place = _action('place', z=4., parameters={'depthMm': 1.1, 'jawOpeningMm': .2})
    cut = _action('apply', z=5.2, dt=.5)
    place['pose'] = {'position': [7., 6., 4.], 'quaternion': q}
    cut['pose'] = {'position': [7., 6., 5.2], 'quaternion': q}
    for action in (_action('configure', mechanicsRevision=2), _action('exchange', instrument='micro_scissors'),
                   place, cut):
        receipt = engine.step(action)
        assert receipt['status'] == 'accepted', receipt
    assert any(event['kind'] == 'strand_transection' for event in receipt['events'])
    wall = engine.view()[rods.FIELD]['editedWallsByLabel']['2']
    assert not wall.get('cuboidSourceCells')
    if vertex_transport:
        packet = engine.view()[rods.FIELD]['vertexDisplacementsByLabel']['2']
        assert packet['geometryBinding'] == wall['identity']
        assert packet['vertexCount'] == len(wall['positionsIJK'])
    restored = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert restored.view()[rods.FIELD]['editedWallsByLabel']['2'] == wall


def test_vertex_transport_requires_the_transactional_wall_sweep():
    with pytest.raises(ValueError, match='VESSEL_VERTEX_TRANSPORT_REQUIRES_SWEPT_NATIVE_WALL'):
        _fixture(vertex_transport=True)


def test_short_edge_transport_pools_motion_and_leaves_protected_components_separate():
    matrix = csr_matrix([[1., 0.], [0., 1.], [.5, .5], [1., 0.], [0., 1.]])
    edges = np.array([[0, 1], [1, 2], [3, 4]])
    protected = np.array([False, False, False, True, False])
    translation, radial = vertex_motion._pool_subresolution_edges((matrix, 2. * matrix), edges, protected, 5)
    expected = [[.5, .5], [.5, .5], [.5, .5], [1., 0.], [0., 1.]]
    np.testing.assert_array_equal(translation.toarray(), expected)
    np.testing.assert_array_equal(radial.toarray(), 2. * np.asarray(expected))


def test_wall_sweep_rejects_crossing_between_clear_endpoints_and_midpoint():
    vertices = np.array([[-.10, -.10, 0.], [.10, -.10, 0.], [0., .10, 0.],
                         [-.025, -.025, .004], [.025, -.025, .004], [0., .025, .004]])
    faces = np.arange(6, dtype=np.int64).reshape(2, 3)
    rest = vertices[faces]
    start = np.zeros((6, 3))
    crossing = start.copy()
    crossing[3:, 2] = -.016
    near_miss_start = start.copy()
    near_miss_start[3:, 0] = .3
    delayed_crossing = crossing * .75
    reference = WallSweepReference(rest, faces, faces, 6)
    checkers = [reference.clear]
    if native_backend.wall_sweep_available():
        source = native_backend.make_wall_sweep(rest, faces, faces, 6)

        def native_clear(prior, candidate, *, static=False):
            return native_backend.wall_sweep_clear(source, prior, candidate, static)

        checkers.append(native_clear)
    for clear in checkers:
        for pose in (start, crossing * .5, crossing):
            assert clear(pose, pose, static=True)
        assert not clear(start, crossing)
        assert not clear(start, delayed_crossing)
        assert clear(near_miss_start, crossing)


def test_local_sweep_rejects_foldover_of_edge_neighbors_with_clear_endpoints():
    # The faces use a common edge. Their opposite corners pass through the
    # same side of the shared plane inside the step, at a noncollapsing pose.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., .5, 0.], [0., 1., .5]])
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    clear = surgical_wall_sweep._local_face_guard(vertices[faces], faces, faces)
    start = np.zeros_like(vertices)
    end = start.copy()
    end[3, 2] = -2.
    assert clear(start, start, True)
    assert clear(end, end, True)
    assert not clear(start, end, False)
    assert clear(start, .2 * end, False)


def test_local_sweep_rejects_excessive_facet_compression():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]])
    clear = surgical_wall_sweep._local_face_guard(vertices[faces], faces, faces)
    start = np.zeros_like(vertices)
    end = start.copy()
    end[2, 1] = -.85
    assert not clear(start, end, False)
    assert clear(start, .5 * end, False)


def test_concave_cut_cap_retains_all_ring_edges_and_requires_the_vessel_axis_inside():
    points = np.asarray([[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [.4, 2., 0.],
                         [.4, .5, 0.], [-.4, .5, 0.], [-.4, 2., 0.], [-2., 2., 0.]])
    ring = list(range(len(points)))
    axis = np.asarray([0., 0., 1.])
    triangles = surfaces._concave_cap(points, ring, points, axis)
    assert len(triangles) == len(ring) - 2
    edges = {(a, b) for row in triangles for a, b in zip(row, (*row[1:], row[0]))}
    assert all((b, a) in edges for a, b in zip(ring, ring[1:] + ring[:1]))
    assert surfaces._concave_cap(points, ring, points - [0., 1., 0.], axis) is None


def test_wall_sweep_rolls_back_the_whole_stroke_after_a_later_microstep(monkeypatch):
    engine = _fixture(guarded=True)
    for action in (_action('exchange', instrument='dissector'), _action('place', z=4.)):
        assert engine.step(action)['status'] == 'accepted'
    previous = engine.snapshot()
    require_clear = surgical_wall_sweep.require_clear
    attempted_times = set()

    def simulate_late_crossing(candidate, prior, proposed, *, static=False):
        if not static:
            attempted_times.add(candidate.state['elapsedSeconds'])
            if len(attempted_times) == 3:
                raise ValueError('VESSEL_WALL_SWEEP_UNSAFE: synthetic crossing')
        require_clear(candidate, prior, proposed, static=static)

    monkeypatch.setattr(surgical_wall_sweep, 'require_clear', simulate_late_crossing)
    receipt = engine.step(_action('move', z=5.05, dt=.4))
    assert len(attempted_times) == 3
    assert receipt['status'] == 'rejected'
    assert receipt['reasonCode'] == 'VESSEL_WALL_SWEEP_UNSAFE'
    assert engine.snapshot() == previous


def test_unresolved_cut_falls_back_only_in_its_source_component():
    engine = _fixture()
    scene = engine.scene
    cells = [int(np.ravel_multi_index(ijk, scene.shape)) for ijk in ((17, 14, 15), (17, 15, 15))]
    engine.state['cutEdges'] = [cells]
    view = engine.view()[rods.FIELD]
    source = view['editedWallsByLabel']['2']
    assert source['cuboidSourceCells']
    assert np.all(np.unravel_index(source['cuboidSourceCells'], scene.shape)[1] < 20)
    assert np.all(np.unravel_index(source['sourceCells'], scene.shape)[1] > 20)
    assert all(c in engine.touched(p, p, 0.) for c, p in zip(cells, scene.points(cells)))


def test_standard_environment_preserves_native_wall_contact():
    from resectionlab.surgical_environment import standardize

    original = _fixture()
    engine = SurgicalEngine(standardize(original.scene))
    wall = original.scene._vessel_source_walls[0]
    contact = wall.vertices[0]
    assert engine.view()[rods.FIELD]['geometryModeByLabel'] == {'2': 'native-probability-wall'}
    assert engine.touched(contact, contact, 0.) == original.touched(contact, contact, 0.)


def test_segment_triangle_distance_includes_crossings_face_normals_and_edge_grazes():
    plane = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    for a, b, expected in (([.25, .25, -1.], [.25, .25, 1.], 0.),
                           ([.25, .25, .2], [.3, .3, .2], .2),
                           ([.6, .6, -1.], [.6, .6, 1.], np.sqrt(.02))):
        distances, difference = surfaces._triangle_distances(np.array(a), np.array(b), plane)
        assert distances[0] == pytest.approx(expected)
        assert np.linalg.norm(difference[0]) == pytest.approx(expected)
