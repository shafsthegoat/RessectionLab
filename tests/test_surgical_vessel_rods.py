import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import CaseData, SourceRef
from resectionlab import surgical_deformation_v2 as tissue
from resectionlab import surgical_glioma_materials as materials
from resectionlab import surgical_vessel_rods as rods
from resectionlab import surgical_vessel_surface as surfaces
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import from_case
from resectionlab import native_backend, surgical_wall_sweep
from resectionlab.surgical_wall_sweep_reference import WallSweepReference


def _fixture(*, guarded=False):
    shape = (35, 35, 30)
    spacing = .4
    affine = np.diag([spacing, spacing, spacing, 1.])
    grid = np.moveaxis(np.indices(shape), 0, -1)
    sites = [(i, 15, 15) for i in range(11, 24)]
    other = [(i, 23, 15) for i in range(11, 14)]
    support = {sites[i]: (sites[i][0], 15, 17) for i in (0, 6, 12)}
    radius = .66
    distances = np.minimum(*(np.linalg.norm(grid - np.clip(grid, chain[0], chain[-1]), axis=-1)
                             for chain in (sites, other)))
    scores = np.clip(.5 + radius - distances * spacing, 0., 1.).astype(np.float32)
    vessels = scores >= .5
    brain = np.zeros(shape, dtype=bool)
    for cell in support.values():
        brain[cell] = True
    props = materials.generated_vascular_properties(shape, sites, other, support, 'native-score-rod',
                                                    vessel_motion_model=rods.MODEL)
    props[materials.PROPERTY][rods.GEOMETRY] = {
        'source': 'Analytic score-bound capsule',
        'sites': {str(np.ravel_multi_index(cell, shape)): [*(np.asarray(cell) * spacing), radius]
                  for cell in sites + other},
    }
    if guarded:
        props[materials.PROPERTY][surgical_wall_sweep.PROPERTY] = surgical_wall_sweep.POLICY
    case = CaseData('native-score-rod', scores, {'brain': brain, 'vessels': vessels}, affine,
                    (SourceRef('fixture', 'synthetic://native-score-rod', provenance='simulated'),),
                    metadata={'anatomy_layers': {'brain': {'kind': 'brain'}, 'vessels': {'kind': 'vessels'}},
                              'surgical_properties': {'tissuePhysicsV2': default_profile(), **props}},
                    vascular_probabilities={'vessels': scores})
    return SurgicalEngine(from_case(case))


def _action(operation, *, z=None, dt=.1, **fields):
    return {
        'slot': 'left', 'operation': operation, 'duration': dt,
        **({'pose': {'position': [6.8, 6., z], 'quaternion': [0, 0, 0, 1]}} if z is not None else {}),
        **fields,
    }


@pytest.mark.parametrize('guarded', [False, True])
def test_native_wall_contact_tracks_passive_dissection_and_verified_replay(guarded):
    engine = _fixture(guarded=guarded)
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
    vertices = wall.source.vertices + offsets[wall.source.owners]
    close = np.flatnonzero((np.abs(vertices[:, 0] - 6.8) < .3) & (vertices[:, 2] < 6.))
    assert len(close)
    clip = close[np.argmin(vertices[close, 2])]
    vertex = vertices[clip]
    owner = int(wall.source.cells[wall.source.owners[clip]])
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


@pytest.mark.parametrize('guarded', [False, True])
def test_public_scissors_transect_score_wall_and_replay_the_same_cap(guarded):
    engine = _fixture(guarded=guarded)
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
    restored = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert restored.view()[rods.FIELD]['editedWallsByLabel']['2'] == wall


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
