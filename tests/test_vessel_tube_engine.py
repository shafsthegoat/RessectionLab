from dataclasses import replace

import numpy as np
import pytest

from resectionlab import surgical_vessel_rods as rods, surgical_vessel_surface as surfaces, vessel_tubes
from resectionlab.core import thaw_json
from resectionlab.imaging import load_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_rl_protection import GuardConfig
from resectionlab.surgical_scene import from_case
from test_surgical_patient import add_source_brain, native_wall_fixture

LAYERS = {'brain': 'brain', 'vessels': 'vessels'}


def branched_wall_fixture(tmp_path):
    """The native-wall fixture vessel with a side branch leaving it at RAS (0, 0, 1)."""
    import nibabel as nib
    from resectionlab.patient_anatomy import PatientImagingCollection

    shape = (65, 65, 65)
    affine = np.diag([-.5, .5, .5, 1.])
    affine[:3, 3] = [16., -16., -16.]
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    trunk = np.broadcast_to(np.hypot(y - 32, z - 34), shape)
    branch = np.where(y >= 32, np.hypot(x - 32, z - 34), np.hypot(np.hypot(x - 32, z - 34), y - 32))
    scores = np.clip(.5 + np.maximum(2.45 - trunk, 1.8 - branch) / 4, 0, 1).astype(np.float32)
    brain = np.broadcast_to((x > 4) & (z > 8) & (z <= 32), shape).astype(np.uint8)
    study = PatientImagingCollection.create(tmp_path / 'native-wall', 'native-wall-branch')
    for series, modality, values in (('t1', 'T1', brain + np.broadcast_to(z / 64, shape)), ('tof', 'TOF', scores)):
        image = nib.Nifti1Image(values.astype(np.float32), affine)
        image.header.set_xyzt_units('mm')
        path = tmp_path / f'{series}.nii.gz'
        nib.save(image, path)
        study.add_series(path, series_id=series, modality=modality, timepoint='preoperative')
    study.set_registration('tof', np.eye(4), status='reviewed', method='explicit test registration')
    add_source_brain(study, tmp_path, brain)
    study.add_layer((scores >= .5).astype(np.uint8), affine, layer_id='vessels', source_series='tof', kind='vessels',
                    role='context', color='#aa2222', probability=scores)
    study.save()
    return study
CENTER = [0., 0., 1.]


@pytest.fixture(scope='module')
def tube_case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('tube-wall')
    study, _, _ = native_wall_fixture(tmp)
    report = prepare_surgical_environment(study.root, tmp / 'tube', layer_ids=LAYERS, native_wall_center=CENTER,
                                          vessel_geometry=surfaces.TUBE_GEOMETRY)
    return study, report


def pose(position):
    return {'position': position, 'quaternion': [1., 0., 0., 0.]}


def _inside(point, triangles):
    axis = np.array([1., .072934, .039701])
    v0, v1, v2 = triangles.transpose(1, 0, 2)
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(axis, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-15
    inv = 1. / np.where(ok, det, 1.)
    t = point - v0
    u = np.einsum('ij,ij->i', t, p) * inv
    q = np.cross(t, e1)
    v = q @ axis * inv
    s = np.einsum('ij,ij->i', e2, q) * inv
    return np.count_nonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (s > 0)) % 2 == 1


def clearances(engine):
    """Working-axis and shell clearance (mm) of each tool against the displaced wall."""
    definitions = rods.definitions(engine)
    _, _, bonds, _ = rods._topology(engine, definitions)
    position, radius = rods._current(engine, definitions)
    result = []
    for frame in surfaces.frames(engine, definitions, position, radius, bonds):
        rest, _, bindings, offsets = surfaces.sweep_mesh(frame)
        triangles = rest + offsets[bindings]
        for _, a, b, calibre, _ in rods._work_capsules(engine):
            distance = float(surfaces._triangle_distances(a, b, triangles)[0].min())
            inside = any(_inside(p, triangles) for p in a + np.linspace(0., 1., 9)[:, None] * (b - a))
            axis = -distance if inside else distance
            result.append((axis, axis - calibre))
    return result


def test_tube_wall_rods_and_tags_come_from_one_stored_model(tube_case):
    _, report = tube_case
    case = load_case(report['casePath'])
    receipt = report['nativeWall']
    assert receipt['geometryPolicy'] == surfaces.TUBE_GEOMETRY == case.metadata[surfaces.GEOMETRY_PROPERTY]
    stored = thaw_json(case.metadata[surfaces.TUBE_MODELS]['vessels'])
    assert receipt['tubeModelDigest'] == surfaces.tube_model_digest(stored)
    model = vessel_tubes.TubeModel.from_dict(stored)
    scene = from_case(case)
    wall = scene._vessel_source_walls[0]
    assert wall.geometry_policy == surfaces.TUBE_GEOMETRY and wall.geometry_binding == receipt['geometryBinding']
    # The contact and display wall lies on the zero level of the stored field
    # (marching cubes on a 0.25 mm grid; float32 positions).
    value, _ = model.evaluate(wall.vertices)
    assert np.median(np.abs(value)) < .005 and np.abs(value).max() < .05
    audit = vessel_tubes.audit(wall.vertices, wall.triangles)
    assert audit['open_edges'] == audit['self_intersections'] == audit['non_manifold_vertices'] == 0
    # Rod sites and calibres are the stored model's.
    sites = scene.properties['gliomaMaterialsV1'][rods.GEOMETRY]['sites']
    expected = vessel_tubes.rod_sites(model, [int(cell) for cell in sites])
    assert {int(cell): site for cell, site in sites.items()} == pytest.approx(expected)
    # The same field gives the soft tags.
    origin, tags = vessel_tubes.tag_volume(model)
    mask = case.compartments['vessels'][tuple(slice(o, o + n) for o, n in zip(origin, tags.shape))]
    inside = tags >= .5
    assert 2 * np.sum(inside & mask) / (inside.sum() + mask.sum()) > .95
    # A changed stored model no longer matches the accepted wall.
    tampered = thaw_json(case.metadata)
    tampered[surfaces.TUBE_MODELS]['vessels']['edgeRadii'][0][0] += .1
    with pytest.raises(ValueError, match='Saved patient native wall differs'):
        from_case(replace(case, metadata=tampered))


def test_tube_preparation_is_reproducible_and_explicit(tube_case, tmp_path):
    study, report = tube_case
    again = prepare_surgical_environment(study.root, tmp_path / 'again', layer_ids=LAYERS, native_wall_center=CENTER,
                                         vessel_geometry=surfaces.TUBE_GEOMETRY)
    for key in ('geometryBinding', 'tubeModelDigest'):
        assert again['nativeWall'][key] == report['nativeWall'][key]
    default = prepare_surgical_environment(study.root, tmp_path / 'default', layer_ids=LAYERS, native_wall_center=CENTER)
    assert default['nativeWall']['geometryPolicy'] == surfaces.ISOLATED_SOURCE_GEOMETRY
    assert surfaces.TUBE_MODELS not in load_case(default['casePath']).metadata
    with pytest.raises(ValueError, match='vessel geometry option'):
        prepare_surgical_environment(study.root, tmp_path / 'unknown', layer_ids=LAYERS, native_wall_center=CENTER,
                                     vessel_geometry='centerline-tubes-v0')


STROKES = {
    'press': ([0., 0., 6.], .15, [[0., 0., z] for z in (4., 3., 2.5, 2., 1.5, 1.)]),
    'fast': ([0., 0., 6.], .05, [[0., 0., 1.]]),
    'glancing': ([0., -4., 2.6], .15, [[0., y, 2.6] for y in (-2., -1., 0., 1., 2., 4.)]),
    'axial': ([-6., 0., 3.5], .3, [[-6., 0., 2.8]] + [[x, 0., 2.8] for x in (-4., -2., 0., 2., 4.)]),
    'junction': ([.8, .8, 4.], .4, [[.8, .8, z] for z in (3.4, 3., 2.8, 2.6, 2.4, 2.2, 2.)]),
}


@pytest.fixture(scope='module')
def branched_case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('tube-branch')
    return prepare_surgical_environment(branched_wall_fixture(tmp).root, tmp / 'tube', layer_ids=LAYERS,
                                        native_wall_center=CENTER, vessel_geometry=surfaces.TUBE_GEOMETRY)


def displaced_wall_audit(engine):
    definitions = rods.definitions(engine)
    _, _, bonds, _ = rods._topology(engine, definitions)
    position, radius = rods._current(engine, definitions)
    for frame in surfaces.frames(engine, definitions, position, radius, bonds):
        yield vessel_tubes.audit(rods._wall_vertices(frame), surfaces.sweep_mesh(frame)[1])


@pytest.mark.parametrize('stroke', sorted(STROKES))
def test_a_tool_driven_into_a_tube_vessel_displaces_it_and_never_enters_the_wall(tube_case, branched_case, stroke):
    report = branched_case if stroke == 'junction' else tube_case[1]
    engine = SurgicalEngine(load_environment(report['casePath']), guard_config=GuardConfig(require_critical_coverage=False))
    start, duration, targets = STROKES[stroke]
    for action in ({'operation': 'exchange', 'instrument': 'dissector', 'duration': .08},
                   {'operation': 'place', 'pose': pose(start), 'duration': .08}):
        assert engine.step({**action, 'slot': 'left'})['status'] == 'accepted'
    statuses = []
    for target in targets:
        before = list(engine.state['slots']['left']['position'])
        result = engine.step({'operation': 'move', 'pose': pose(target), 'duration': duration, 'slot': 'left'})
        statuses.append(result['status'])
        if result['status'] == 'rejected':
            # A stopped stroke leaves the tool where it was.
            assert engine.state['slots']['left']['position'] == before
        for axis, shell in clearances(engine):
            assert axis > 0., (stroke, target, axis)
            # The 1 mm working shell stays out of the displayed and collision wall.
            assert shell >= -rods.SHELL_TOLERANCE_MM, (stroke, target, shell)
        for audit in displaced_wall_audit(engine):
            assert audit['self_intersections'] == audit['open_edges'] == audit['non_manifold_vertices'] == 0
    moved = max((np.linalg.norm(v) for v in engine.state['deformation'].values()), default=0.)
    if stroke != 'fast':
        assert moved > .05, (stroke, moved)
    if stroke in ('fast', 'glancing', 'junction'):
        # Where the wall cannot clear the shell within its limits the stroke stops short.
        assert {'partial', 'rejected'} & set(statuses)


def test_tube_wall_reaches_the_desktop_with_tags_and_replays_after_reopen(tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request

    study, _, _ = native_wall_fixture(tmp_path)

    def call(session, operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)

    def surface(payload):
        layer = next(layer for layer in payload['compartments'] if layer['name'] == 'vessels')
        receipt = payload['metadata']['surgical_adapter']['nativeWall']
        wall = layer['vesselSurface']
        assert receipt['geometryPolicy'] == surfaces.TUBE_GEOMETRY
        assert ':'.join(wall[key]['sha256'] for key in ('positions', 'indices', 'sourceCells')) == receipt['geometryBinding']
        assert 'scoreLighting' not in wall
        return wall, receipt

    session = BridgeSession(tmp_path / 'transfers')
    try:
        opened = call(session, 'anatomyOpen', {'path': str(study.root / 'patient-anatomy.json')})
        payload = call(session, 'prepareSurgicalEnvironment', {
            'collectionId': opened['collectionId'], 'snapshotId': opened['document']['snapshot_id'],
            'layers': LAYERS, 'nativeWallCenter': CENTER, 'vesselGeometry': surfaces.TUBE_GEOMETRY})
        wall, receipt = surface(payload)
        tags = np.fromfile(wall['vesselTags']['path'], '<f4').reshape(payload['shape'])
        assert tags.min() >= 0 and tags.max() == 1 and np.any((tags > 0) & (tags < 1))
        classes = np.fromfile(wall['vesselClasses']['path'], 'u1')
        # The local wall model carries no named labels, so every vertex is the artery/unclassified colour.
        assert len(classes) == wall['positions']['shape'][0] and not classes.any()
        case_args = {'caseHash': payload['caseHash']}
        state = call(session, 'surgicalRehearsal', {**case_args, 'command': 'inspect'})
        for action in ({'operation': 'exchange', 'instrument': 'dissector', 'duration': .08},
                       {'operation': 'place', 'pose': pose([0., 0., 6.]), 'duration': .08},
                       {'operation': 'move', 'pose': pose([0., 0., 2.5]), 'duration': .4}):
            state = call(session, 'surgicalRehearsal', {**case_args, 'command': 'step',
                                                        'expectedStateId': state['view']['stateId'],
                                                        'action': {**action, 'slot': 'left'}})
        motion = state['view']['vesselMechanics']
        assert max(np.linalg.norm(np.subtract(site['positionMm'], site['restPositionMm'])) for site in motion['sites']) > .05
        assert motion['contactedCells']['left']
        assert motion['vertexDisplacementsByLabel']['2']['geometryBinding'] == receipt['geometryBinding']
        destination = tmp_path / 'saved.ressectionlab'
        call(session, 'saveCase', {**case_args, 'path': str(destination), 'workspace': {}, 'overwrite': False})
    finally:
        session.close()
    cold = BridgeSession(tmp_path / 'cold-transfers')
    try:
        reopened = call(cold, 'loadCase', {'path': str(destination)})
        assert surface(reopened)[1]['geometryBinding'] == receipt['geometryBinding']
        replay = call(cold, 'surgicalRehearsal', {**case_args, 'command': 'inspect', 'branchId': state['branchId']})
        assert replay['view']['stateId'] == state['view']['stateId']
        assert replay['view']['vesselMechanics'] == motion
    finally:
        cold.close()


def _no_refit(*args, **kwargs):
    raise AssertionError('an edit or replay must not refit the tube model')


def test_an_edit_cuts_the_stored_field_and_never_refits(tube_case, monkeypatch):
    import json

    _, report = tube_case
    engine = SurgicalEngine(load_environment(report['casePath']), guard_config=GuardConfig(require_critical_coverage=False))
    # A tool rests on the vessel 9 mm from the edit.
    for action in ({'operation': 'exchange', 'instrument': 'dissector', 'duration': .08},
                   {'operation': 'place', 'pose': pose([6., 0., 4.]), 'duration': .08},
                   {'operation': 'move', 'pose': pose([6., 0., 3.1]), 'duration': .3}):
        assert engine.step({**action, 'slot': 'left'})['status'] == 'accepted'
    wall = engine.scene._vessel_source_walls[0]
    model = wall.tube_model
    points = engine.scene.points(wall.cells)
    removed = [int(c) for c, p in zip(wall.cells, points) if abs(p[0] + 3.) <= .5 and p[2] > 1.6]
    monkeypatch.setattr(vessel_tubes, 'extract', _no_refit)
    engine.state['removed'] = removed
    view = engine.view()[rods.FIELD]
    edited = view['editedWallsByLabel'][str(wall.label)]
    assert edited['tubeRemovedCells'] == sorted(removed)
    cut = surfaces.removed_tube(wall, engine.scene.shape, sorted(removed))
    assert edited['tubeModelDigest'] == surfaces.tube_model_digest(surfaces.stored_tube_model(cut))
    assert len(edited['vesselClasses']) == len(edited['positionsIJK'])
    # Geometry away from the cut is exactly the stored wall.
    rest = wall.positions_ijk.astype(np.float32)
    kept = {tuple(row) for row in np.asarray(edited['positionsIJK'], np.float32).tolist()}
    centre = np.linalg.solve(engine.scene.affine[:3, :3], np.array([-3., 0., 1.]) - engine.scene.affine[:3, 3])
    far = np.linalg.norm((rest - centre) * engine.scene.spacing, axis=1) > 8.5
    assert far.sum() > 100 and all(tuple(row) in kept for row in rest[far].tolist())
    # The resting tool keeps its shell out of the edited wall, and the removed segment is no longer tagged.
    rods.require_working_axes_clear(engine)
    assert all(shell >= -rods.SHELL_TOLERANCE_MM for _, shell in clearances(engine))
    origin, before = vessel_tubes.tag_volume(model)
    _, after = vessel_tubes.tag_volume(cut)
    local = np.column_stack(np.unravel_index(removed, engine.scene.shape)) - origin
    assert np.all(after[tuple(local.T)] < .5) and np.mean(before[tuple(local.T)] >= .5) > .5
    # Cold reopen: the saved case, the same actions and the same edit record give the same state and
    # wall from the stored model alone (extract stays disabled). Vessel-cell removal has no public
    # action in a local wall rehearsal, so the record is applied as the engine state it lives in.
    snapshot = json.loads(json.dumps(engine.snapshot()))
    cold = SurgicalEngine(load_environment(report['casePath']), guard_config=GuardConfig(require_critical_coverage=False))
    for action in snapshot['actions']:
        assert cold.step(action)['status'] == 'accepted'
    cold.state['removed'] = list(snapshot['checkpoint']['state']['removed'])
    assert cold.state_id == engine.state_id
    assert cold.view()[rods.FIELD]['editedWallsByLabel'] == view['editedWallsByLabel']


def test_a_full_transection_leaves_two_capped_stumps(tube_case):
    _, report = tube_case
    case = load_case(report['casePath'])
    model, mask = surfaces.case_tube_model(case, 'vessels'), case.compartments['vessels']
    cells = np.flatnonzero(mask)
    x = (np.column_stack(np.unravel_index(cells, mask.shape)) @ case.affine[:3, :3].T + case.affine[:3, 3])[:, 0]
    slab = cells[np.abs(x - 3.) <= .5]
    edited = mask.copy()
    edited.flat[slab] = False
    whole = surfaces._tube_wall(model, mask)
    cut = surfaces._tube_wall(model.with_removed(slab, surfaces.open_faces(cells, mask.shape, slab)), edited)
    world = lambda ijk: ijk.astype(float) @ case.affine[:3, :3].T + case.affine[:3, 3]
    before, after = vessel_tubes.audit(world(whole[0]), whole[1]), vessel_tubes.audit(world(cut[0]), cut[1])
    assert after['components'] == before['components'] + 1
    for key in ('open_edges', 'non_manifold_edges', 'inconsistent_edges', 'non_manifold_vertices', 'self_intersections'):
        assert after[key] == 0, (key, after)
    assert np.all(np.abs(x[np.isin(cells, slab)] - 3.) <= .5)
    # No wall vertex lies inside the removed slab.
    assert not np.any(np.abs(world(cut[0])[:, 0] - 3.) < .5 - 1e-6)


def _two_tools(report):
    engine = SurgicalEngine(load_environment(report['casePath']), guard_config=GuardConfig(require_critical_coverage=False))
    for slot, y in (('left', -3.5), ('right', 3.5)):
        for action in ({'operation': 'exchange', 'instrument': 'dissector', 'duration': .08},
                       {'operation': 'place', 'pose': pose([0., y, 6.]), 'duration': .08},
                       {'operation': 'move', 'pose': pose([0., y, 1.6]), 'duration': .3}):
            assert engine.step({**action, 'slot': slot})['status'] == 'accepted'
    return engine


def test_a_vessel_pinched_between_two_tools_stops_without_entering_either_shell(tube_case):
    engine = _two_tools(tube_case[1])
    statuses, lateral = [], []
    for y in (2.4, 2.1, 1.9, 1.7, 1.5):
        for slot, side in (('left', -1.), ('right', 1.)):
            statuses.append(engine.step({'operation': 'move', 'pose': pose([0., side * y, 1.6]), 'duration': .15,
                                         'slot': slot})['status'])
            for axis, shell in clearances(engine):
                assert axis > 0. and shell >= -rods.SHELL_TOLERANCE_MM, (slot, y, shell)
            definitions = rods.definitions(engine)
            position, _ = rods._current(engine, definitions)
            middle = np.argmin(np.abs(definitions.rest[:, 0]) + np.abs(definitions.rest[:, 1]))
            lateral.append(float(position[middle, 1] - definitions.rest[middle, 1]))
    # Squeezed from both sides the vessel cannot clear both shells: the tools stop instead.
    assert {'partial', 'rejected'} & set(statuses)
    # No oscillation: the trapped vessel stays near its rest line.
    assert max(map(abs, lateral)) < .3 and np.count_nonzero(np.diff(np.sign(np.round(lateral, 3)))) <= 2


def test_a_failed_shell_push_is_charged_to_the_tool_that_caused_it(tube_case):
    engine = SurgicalEngine(load_environment(tube_case[1]['casePath']), guard_config=GuardConfig(require_critical_coverage=False))
    definitions = rods.definitions(engine)
    retained, _, bonds, _ = rods._topology(engine, definitions)
    position, radius = rods._current(engine, definitions)
    clear = ('left', np.array([0., -3.5, 1.6]), np.array([0., -3.5, 8.]), 1., .1)
    inside = ('right', np.array([0., 1.6, 1.6]), np.array([0., 1.6, 8.]), 1., .1)
    moved, hands = rods._project_shell(engine, definitions, position, radius, bonds, retained, (clear, inside))
    assert hands == ['right'] and np.abs(moved - position).max() > 0
    assert rods._project_shell(engine, definitions, position, radius, bonds, retained, (clear,))[1] == []


def test_the_tube_option_needs_shell_contact_and_its_metadata_its_policy(tube_case, tmp_path):
    from resectionlab.surgical_patient import DEFAULT_PHYSICS, prepare_surgical_patient

    study, report = tube_case
    with pytest.raises(ValueError, match='requires V2 tissue physics'):
        prepare_surgical_patient(study.root, tmp_path / 'v1', layer_ids=LAYERS, native_wall_center=CENTER,
                                 require_complete=False, tissue_physics=DEFAULT_PHYSICS,
                                 vessel_geometry=surfaces.TUBE_GEOMETRY)
    case = load_case(report['casePath'])
    foreign = thaw_json(case.metadata)
    foreign[surfaces.GEOMETRY_PROPERTY] = foreign['surgical_adapter']['nativeWall']['geometryPolicy'] = surfaces.ISOLATED_SOURCE_GEOMETRY
    del foreign['surgical_adapter']['nativeWall']['tubeModelDigest']
    with pytest.raises(ValueError, match='Saved patient native wall differs'):
        from_case(replace(case, metadata=foreign))
