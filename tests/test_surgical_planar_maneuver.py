"""Canonical integration checks on a generated engineering coupon only."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import surgical_planar_contact as contact
from resectionlab import surgical_planar_maneuver as maneuver
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def action(index):
    return {'slot': 'left', 'operation': 'bladeManeuver',
            'maneuverPhase': maneuver.PHASES[index], 'duration': maneuver.DURATIONS[index]}


def run(engine, count=4):
    for index in range(count):
        result = engine.step(action(index))
        assert result['status'] == 'accepted', result
    return engine


class ObservedEngine(SurgicalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed = []

    def clone(self):
        candidate = super().clone()
        candidate.observed = copy.deepcopy(self.observed)
        return candidate

    def _substep(self, *args):
        events = super()._substep(*args)
        self.observed.append(copy.deepcopy(self.state))
        return events


def test_every_native_increment_matches_preceding_kernel_and_visible_geometry():
    engine = ObservedEngine(maneuver.engineering_scene())
    kernel = maneuver.model(engine)
    state = contact.ContactState(kernel.identity, kernel.initial_state().displacement_mm,
                                 kernel.initial_state().memory_mm, -.1, 0)
    expected = []
    for index in range(4):
        begin = state.offset_mm
        count = round(maneuver.DURATIONS[index] / maneuver.DT)
        for tick in range(count):
            offset = begin + (maneuver.OFFSETS[index] - begin) * (tick + 1) / count
            commanded = (kernel.blade.origin_mm + offset * kernel.blade.inward_normal
                         + kernel.blade.size_mm[1] / 2 * kernel.blade.axis_v)
            physical_offset = float((commanded - kernel.blade.origin_mm
                - kernel.blade.size_mm[1] / 2 * kernel.blade.axis_v) @ kernel.blade.inward_normal)
            solved = kernel.step(state, offset_mm=physical_offset, dt_s=maneuver.DT)
            state = solved['state']
            expected.append((state, solved['receipt']))
        assert engine.step(action(index))['status'] == 'accepted'
    assert len(engine.observed) == len(expected) == 28
    for captured, (direct, receipt) in zip(engine.observed, expected):
        raw = captured[maneuver.PROPERTY]['kernel']
        np.testing.assert_array_equal(raw['displacementMm'], direct.displacement_mm)
        np.testing.assert_array_equal(raw['memoryMm'], direct.memory_mm)
        assert raw['steps'] == direct.steps
        assert raw['offsetMm'] == direct.offset_mm
        assert captured[maneuver.PROPERTY]['receipt']['normalReactionN'] == receipt['normalReactionN']
        assert 'wallSeconds' not in captured[maneuver.PROPERTY]['receipt']
        shell = engine.clone(); shell.state = captured
        visible = shell.view()['planarBladeManeuver']
        np.testing.assert_array_equal(visible['contactSurface']['positionsMm'],
                                      kernel.mesh.nodes + direct.displacement_mm.reshape(-1, 3))
        slot = captured['slots']['left']
        geometry = slot['planarBladeGeometry']
        rotation = Rotation.from_quat(slot['quaternion'])
        center = np.asarray(slot['position']) + rotation.apply(geometry['centerLocalMm'])
        np.testing.assert_allclose(center, kernel.blade.origin_mm + direct.offset_mm * kernel.blade.inward_normal
                                   - maneuver.THICKNESS / 2 * kernel.blade.inward_normal, atol=1e-12)
        assert captured['removed'] == [] and captured['holds'] == {}


def test_preview_is_detached_and_commit_matches_without_other_clock_effects():
    engine = SurgicalEngine(maneuver.engineering_scene())
    before = engine.snapshot(); history = copy.deepcopy(engine.history)
    right = copy.deepcopy(engine.state['slots']['right'])
    static = {key: copy.deepcopy(engine.state[key]) for key in
              ('fluid', 'thermal', 'sources', 'vesselsV2', 'drillingV2', 'contact', 'debris')}
    for index in range(4):
        current = engine.snapshot()
        preview = engine.preview(action(index))
        assert engine.snapshot() == current
        committed = engine.step(action(index))
        assert committed['stateId'] == preview['stateId']
        assert committed['events'] == preview['events']
        assert engine.state['slots']['right'] == right
        assert all(engine.state[key] == value for key, value in static.items())
    assert before['actions'] == [] and history == []
    assert engine.state['elapsedSeconds'] == pytest.approx(sum(maneuver.DURATIONS))
    assert [e['kind'] for e in engine.state[maneuver.PROPERTY]['contactEvents']] == [
        'blade_contact_acquired', 'blade_contact_released']
    preview['view']['planarBladeManeuver']['details']['declaration']['cells'].clear()
    assert engine.scene.properties[maneuver.PROPERTY]['cells']


@pytest.mark.parametrize('bad', [
    {'slot': 'left', 'operation': 'inspect', 'duration': .2},
    {'slot': 'right', 'operation': 'activate', 'duration': .2},
    {'slot': 'left', 'operation': 'bladeManeuver', 'maneuverPhase': 'hold', 'duration': .2},
    {**action(0), 'duration': .2},
    {**action(0), 'parameters': {}},
])
def test_refused_actions_are_atomic_before_and_during_maneuver(bad):
    engine = SurgicalEngine(maneuver.engineering_scene())
    for started in (False, True):
        if started: run(engine, 1)
        before = engine.snapshot(); history = copy.deepcopy(engine.history)
        result = engine.step(bad)
        assert result['status'] == 'rejected'
        assert result['elapsedSeconds'] == 0.
        assert engine.snapshot() == before and engine.history == history


@pytest.mark.parametrize('mode', ['solve', 'after_increment', 'finalize'])
def test_cancellation_and_failed_finalization_preserve_complete_committed_owner(monkeypatch, mode):
    from resectionlab import surgical_effects
    engine = run(ObservedEngine(maneuver.engineering_scene()), 1)
    before = engine.snapshot(); history = copy.deepcopy(engine.history)
    observations = copy.deepcopy(engine.observed)
    class Cancelled(Exception): pass
    if mode == 'solve':
        original = contact.PlanarContact.step
        def fail(self, state, **kwargs):
            original(self, state, **kwargs)
            raise Cancelled('after a real solve')
        monkeypatch.setattr(contact.PlanarContact, 'step', fail)
        callback = None
    elif mode == 'after_increment':
        calls = 0
        def callback():
            nonlocal calls
            calls += 1
            if calls == 20: raise Cancelled('inside native transition')
    else:
        def fail(candidate):
            candidate.state['fluid']['0'] = 7.
            raise Cancelled('during finalization')
        monkeypatch.setattr(surgical_effects, 'finalize_action', fail)
        callback = None
    with pytest.raises(Cancelled): engine.step(action(1), check=callback)
    assert engine.snapshot() == before and engine.history == history
    assert engine.observed == observations


def test_rejected_late_collision_rolls_back_earlier_accepted_increments(monkeypatch):
    engine = run(ObservedEngine(maneuver.engineering_scene()), 1)
    before = engine.snapshot(); history = copy.deepcopy(engine.history)
    observations = copy.deepcopy(engine.observed)
    original = maneuver._collision
    calls = 0
    def fail_late(*args):
        nonlocal calls
        calls += 1
        original(*args)
        if calls == 5: raise ValueError('GENERATED_LATE_COLLISION')
    monkeypatch.setattr(maneuver, '_collision', fail_late)
    result = engine.step(action(1))
    assert result['status'] == 'rejected' and result['reason'] == 'GENERATED_LATE_COLLISION'
    assert calls == 5
    assert engine.snapshot() == before and engine.history == history and engine.observed == observations


def test_restore_branch_and_checkpoint_tampering():
    engine = run(SurgicalEngine(maneuver.engineering_scene()))
    saved = engine.snapshot()
    restored = SurgicalEngine.restore(engine.scene, saved)
    assert restored.snapshot() == saved and restored.history == engine.history
    for position in range(5):
        branch = engine.branch(position)
        assert len(branch.actions) == position
        for index in range(position, 4): assert branch.step(action(index))['status'] == 'accepted'
        assert branch.snapshot() == saved and branch.history == engine.history
    changed = copy.deepcopy(saved)
    changed['checkpoint']['state'][maneuver.PROPERTY]['kernel']['memoryMm'][0][0] += .001
    with pytest.raises(ValueError, match='Checkpoint differs'): SurgicalEngine.restore(engine.scene, changed)
    props = copy.deepcopy(engine.scene.properties)
    props[maneuver.PROPERTY]['sourceGeometryHash'] = 'unbound'
    with pytest.raises(ValueError, match='SOURCE_BINDING'):
        SurgicalScene(engine.scene.labels, engine.scene.affine, engine.scene.regions, engine.scene.source, props)


def test_verified_cache_and_observed_frames_keep_exact_canonical_boundaries(tmp_path):
    from resectionlab.surgical_verified_cache import VerifiedSurgicalCache
    from resectionlab.surgical_frame_replay import SurgicalFrameCache
    from resectionlab.surgical_replay_cache import SurgicalReplayCache
    engine = run(SurgicalEngine(maneuver.engineering_scene()))
    saved = engine.snapshot(); history = copy.deepcopy(engine.history)
    cache = VerifiedSurgicalCache(directory=tmp_path / 'verified')
    canonical = SurgicalEngine.restore(engine.scene, saved, use_cache=True, force_verify=True, verified_cache=cache)
    assert canonical.snapshot() == saved
    cached = SurgicalEngine.restore(engine.scene, saved, use_cache=True, verified_cache=cache)
    assert cached.snapshot() == saved and cache.hits >= 1
    frames = SurgicalFrameCache(directory=tmp_path / 'frames'); replay = SurgicalReplayCache()
    for position in range(4):
        initial, _, info = frames.seek(engine, saved, position, 0, replay, verified_history=history)
        assert initial.snapshot() == engine.branch(position).snapshot()
        for index in range(info['frameCount']):
            observed, _, metadata = frames.seek(engine, saved, position, index, replay, verified_history=history)
            assert observed.view()['planarBladeManeuver']['available']
            if metadata['isActionBoundary'] and index == info['frameCount'] - 1:
                assert observed.snapshot() == engine.branch(position + 1).snapshot()
    assert engine.snapshot() == saved and engine.history == history


def test_exact_boundary_facets_refuse_a_warped_bilinear_q1_face():
    engine = SurgicalEngine(maneuver.engineering_scene()); kernel = maneuver.model(engine)
    raw = {'modelId': kernel.identity, 'displacementMm': np.zeros(len(kernel.free)).tolist(),
           'memoryMm': np.zeros((len(kernel.times), len(kernel.free))).tolist(), 'offsetMm': 0., 'steps': 1}
    # One free exterior corner lifts out of the top-face plane. Its Q1 face
    # interior is bilinear and cannot be represented by two planar triangles.
    top = np.flatnonzero(np.isclose(kernel.mesh.nodes[:, 2], 4.))
    raw['displacementMm'][3 * int(top[0]) + 2] = .005
    with pytest.raises(ValueError, match='NONPLANAR_RENDERED_Q1_FACE'):
        maneuver._surface(kernel, {'kernel': raw})


def test_swept_pad_cannot_tunnel_through_material_between_clear_endpoints():
    engine = SurgicalEngine(maneuver.engineering_scene()); kernel = maneuver.model(engine)
    zero = kernel.initial_state().displacement_mm
    # A deliberately oversized diagnostic translation has clear endpoint pads
    # on opposite sides of the intact specimen, but its swept pad crosses it.
    maneuver._collision(engine, kernel, zero, -1.)
    maneuver._collision(engine, kernel, zero, 5.)
    with pytest.raises(ValueError, match='PAD_COLLISION_OR_UNRESOLVED_Q1_GEOMETRY'):
        maneuver._collision(engine, kernel, zero, 5., zero, -1.)


def test_loaded_reference_footprint_migration_refuses_real_poisson_expansion():
    source = maneuver.engineering_scene(); props = copy.deepcopy(source.properties)
    props['brainContinuumV1']['materials']['1']['poissonRatio'] = .3
    # Material assumptions are a separate declaration binding from source geometry.
    props[maneuver.PROPERTY]['materialProfileHash'] = semantic_digest(props['brainContinuumV1'])
    scene = SurgicalScene(source.labels, source.affine, source.regions, source.source, props)
    engine = run(SurgicalEngine(scene), 1); before = engine.snapshot()
    result = engine.step(action(1))
    assert result['status'] == 'rejected'
    assert 'FOOTPRINT_MIGRATION_UNSUPPORTED' in result['reason']
    assert engine.snapshot() == before


def test_observed_intermediate_frames_are_actual_accepted_kernel_states(tmp_path):
    from resectionlab.surgical_frame_replay import SurgicalFrameCache
    from resectionlab.surgical_replay_cache import SurgicalReplayCache
    observed = run(ObservedEngine(maneuver.engineering_scene()))
    source = run(SurgicalEngine(observed.scene)); saved = source.snapshot()
    accepted = []
    for captured in observed.observed:
        shell = source.clone(); shell.state = captured
        accepted.append(shell.state_id)
    replay = SurgicalReplayCache(); frames = SurgicalFrameCache(directory=tmp_path / 'frames')
    admission = frames.admit(source, saved, replay, verified_history=source.history)
    middle = 0
    for position in range(4):
        _, _, first = frames.seek(source, admission.snapshot, position, 0, replay,
                                 verified_history=admission.history, admission=admission)
        for index in range(first['frameCount']):
            shell, _, metadata = frames.seek(source, admission.snapshot, position, index, replay,
                                            verified_history=admission.history, admission=admission)
            if not metadata['isActionBoundary']:
                middle += 1
                assert shell.state_id in accepted
                raw = shell.state[maneuver.PROPERTY]['kernel']
                np.testing.assert_array_equal(shell.view()['planarBladeManeuver']['contactSurface']['positionsMm'],
                    maneuver.model(shell).mesh.nodes + np.asarray(raw['displacementMm']).reshape(-1, 3))
    assert middle >= 3


def test_global_rigid_frame_change_preserves_forces_geometry_and_maneuver():
    original = maneuver.engineering_scene(); old = original.properties[maneuver.PROPERTY]
    transform = Rotation.from_euler('xyz', [.31, -.42, .27]); shift = np.array([9., -12., 6.])
    affine = original.affine.copy()
    affine[:3, :3] = transform.as_matrix() @ affine[:3, :3]
    affine[:3, 3] = transform.apply(affine[:3, 3]) + shift
    props = copy.deepcopy(original.properties); props.pop(maneuver.PROPERTY)
    base = SurgicalScene(original.labels, affine, original.regions, original.source, props)
    b = old['blade']
    blade = contact.PlanarBlade.create(transform.apply(b['originMm']) + shift,
        transform.apply(b['axisU']), transform.apply(b['axisV']), transform.apply(b['inwardNormal']), b['sizeMm'])
    props[maneuver.PROPERTY] = maneuver.declaration(base, old['cells'], old['contactFaces'], old['supportFaces'], blade)
    rotated = SurgicalScene(base.labels, base.affine, base.regions, base.source, props)
    first, second = SurgicalEngine(original), SurgicalEngine(rotated)
    for index in range(4):
        assert first.step(action(index))['status'] == 'accepted'
        result = second.step(action(index))
        assert result['status'] == 'accepted', result
        source_view, rotated_view = first.view()['planarBladeManeuver'], second.view()['planarBladeManeuver']
        np.testing.assert_allclose(rotated_view['contactSurface']['positionsMm'],
            transform.apply(source_view['contactSurface']['positionsMm']) + shift, atol=1e-12)
        np.testing.assert_allclose(rotated_view['details']['lastReceipt']['reactionOnBladeN'],
            transform.apply(source_view['details']['lastReceipt']['reactionOnBladeN']), atol=1e-12)
        assert rotated_view['contactSurface']['triangles'] == source_view['contactSurface']['triangles']


def test_material_change_requires_redeclared_material_binding():
    scene = maneuver.engineering_scene(); props = copy.deepcopy(scene.properties)
    props['brainContinuumV1']['materials']['1']['shearPa'] *= 2
    with pytest.raises(ValueError, match='FIXED_DOMAIN_REQUIRED'):
        SurgicalScene(scene.labels, scene.affine, scene.regions, scene.source, props)
