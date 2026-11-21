"""Cross-component safety and replay checks using the actual generated walkthrough anatomy."""
import copy

import pytest
import numpy as np

from scripts.generated_glioma_walkthrough_case import build_glioma_case
from resectionlab import native_backend
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


@pytest.fixture(scope='module')
def anatomy():
    _, scene, landmarks = build_glioma_case()
    return scene, landmarks


def action(operation, **kwargs):
    return {'slot': 'left', 'operation': operation, 'duration': .1, **kwargs}


def pose(position):
    return {'position': list(position), 'quaternion': [1., 0., 0., 0.]}


def test_acquired_provenance_refuses_generated_exposure(anatomy):
    scene, _ = anatomy
    marked = SurgicalScene(scene.labels, scene.affine, scene.regions,
                           {**scene.source, 'patientId': 'unadmitted-source-check'}, scene.properties)
    with pytest.raises(ValueError, match='OPERATIVE_EXPOSURE_REQUIRES_GENERATED_SOURCE'):
        SurgicalEngine(marked)


def test_intact_opening_refuses_direct_placement_and_swept_access(anatomy):
    scene, landmarks = anatomy
    engine = SurgicalEngine(scene)
    assert engine.step(action('exchange', instrument='cusa'))['status'] == 'accepted'
    before = engine.snapshot()
    request = action('place', pose=pose(landmarks['targetRasMm']))
    for result in [engine.preview(request, include_view=False), engine.step(request)]:
        assert result['status'] == 'rejected'
    assert engine.snapshot() == before

    outside = list(landmarks['entryCenterRasMm'])
    outside[2] += 16
    assert engine.step(action('place', pose=pose(outside)))['status'] == 'accepted'
    request = action('move', pose=pose(landmarks['targetRasMm']))
    initial = engine.state_id
    predicted = engine.preview(request, include_view=False)
    assert engine.state_id == initial
    outcome = engine.step(request)
    assert outcome['status'] in ('partial', 'rejected')
    assert predicted['stateId'] == outcome['stateId']
    assert engine.state['slots']['left']['position'] != landmarks['targetRasMm']
    assert not engine.state['removed']
    assert SurgicalEngine.restore(scene, engine.snapshot(), force_verify=True).snapshot() == engine.snapshot()


def test_active_aspirator_cannot_remove_intact_scalp(anatomy):
    scene, landmarks = anatomy
    engine = SurgicalEngine(scene)
    center = np.asarray(landmarks['entryCenterRasMm'], dtype=float)
    exterior = center + [0., 0., 16.]
    assert engine.step(action('exchange', instrument='cusa'))['status'] == 'accepted'
    assert engine.step(action('place', pose=pose(exterior)))['status'] == 'accepted'
    outcome = engine.step(action('apply', pose=pose(center-[0., 0., 2.])))
    assert outcome['status'] in ('rejected', 'partial')
    assert outcome['reasonCode'] == 'OPERATIVE_DEEP_INSTRUMENT_ON_CRANIAL_LAYER'
    assert not engine.state['removed']
    assert engine.metrics()['targetRemovedMm3'] == engine.metrics()['nonTargetRemovedMm3'] == 0


def test_exposure_replays_in_python_and_rejects_corrupted_checkpoint(anatomy):
    scene, _ = anatomy
    engine = SurgicalEngine(scene)
    if native_backend.cpp_enabled():
        from resectionlab.surgical_native_engine import create
        assert create(engine) == (None, 'advanced-operative-profile')
    count = native_backend.backend_identity()['counters']['residentActionCalls']
    command = action('inspect')
    assert engine.step(command)['status'] == 'accepted'
    assert engine.__dict__.get('_resident') is None
    assert native_backend.backend_identity()['counters']['residentActionCalls'] == count
    snapshot = engine.snapshot()
    assert SurgicalEngine.restore(scene, snapshot, use_cache=False, force_verify=True).state_id == engine.state_id

    altered = copy.deepcopy(snapshot)
    altered['checkpoint']['state']['elapsedSeconds'] += 1.
    with pytest.raises(ValueError, match='Checkpoint'):
        SurgicalEngine.restore(scene, altered, use_cache=False, force_verify=True)


def test_native_file_guard_exits_on_cancel_and_detects_changed_source(tmp_path, monkeypatch):
    if not native_backend.cpp_enabled():
        pytest.skip('compiled numerical kernels unavailable')
    source = tmp_path / 'verified-source'
    source.write_bytes(b'original data')
    monkeypatch.setitem(native_backend._files, source, (native_backend._stat(source), native_backend._digest(source)))

    def interrupted():
        raise InterruptedError('cancelled')

    with pytest.raises(InterruptedError):
        native_backend.operative_python_substep(interrupted)
    assert getattr(native_backend._execution, 'depth', 0) == 0

    def replace_source():
        source.write_bytes(b'changed data at a candidate boundary')
        # The inner floating-point kernels still check the module bindings.
        native_backend.cpp_enabled()

    with pytest.raises(native_backend.NativeIntegrityError):
        native_backend.operative_python_substep(replace_source)
    assert getattr(native_backend._execution, 'depth', 0) == 0


def test_carried_stay_uses_accepted_pose_and_cannot_support_tissue(anatomy):
    from resectionlab import surgical_operative_exposure as exposure
    scene, _ = anatomy
    engine = SurgicalEngine(scene)
    source = scene.properties[exposure.PROPERTY]['bodies']['scalp']['gripCell']
    position = scene.points([source])[0]
    exposure._place_object(engine, 'scalp', 'retaining-stay', position, cell=source)
    obj = engine.state[exposure.PROPERTY]['objects'][-1]
    assert exposure._available(engine, 'scalp', 'retaining-stay') is obj
    attached = exposure._object_position(engine, obj)
    accepted = attached + [3., 2., 4.]
    for status, hand in [('carried', 'right'), ('retrieved', None)]:
        obj.update(status=status, heldBy=hand, positionMm=accepted.tolist())
        assert exposure._available(engine, 'scalp', 'retaining-stay') is None
        assert np.array_equal(exposure._object_position(engine, obj), accepted)
        described = exposure.summary(engine)['objects'][-1]
        assert described['sourceCell'] == source
        assert described['positionMm'] == accepted.tolist()


def test_carried_objects_remain_in_closure_inventory(anatomy):
    from resectionlab import surgical_operative_exposure as exposure
    scene, _ = anatomy
    engine = SurgicalEngine(scene)
    state = engine.state[exposure.PROPERTY]
    config = scene.properties[exposure.PROPERTY]['bodies']
    cargo = {'id': 'pad', 'kind': 'cottonoid', 'body': 'field', 'status': 'carried', 'heldBy': 'left'}
    state['objects'].append(cargo)
    state['bodies']['dura']['status'] = 'supported'
    grip = config['dura']['gripCell']
    point = exposure.source_position(engine, 'dura', grip)
    with pytest.raises(ValueError, match='OPERATIVE_REMOVE_COUNTED_COTTONOID'):
        exposure._grasp(engine, 'right', 'dura', point, np.array([1., 0., 0., 0.]), [grip])
    cargo.update(kind='raney-clip', body='scalp')
    state['bodies']['scalp']['status'] = 'seated'
    anchor = config['scalp']['anchorCells'][0]
    with pytest.raises(ValueError, match='OPERATIVE_REMOVE_SCALP_CLIP_BEFORE_CLOSURE'):
        exposure._anchors(engine, 'needle_driver', 'scalp', 'right', scene.points([anchor])[0], [anchor])
