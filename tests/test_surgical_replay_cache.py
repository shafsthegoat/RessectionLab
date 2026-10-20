"""Generated cursor controls, checked against the independent full replay."""
import copy
import json

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_scene import SurgicalScene


def engine():
    labels = np.zeros((12, 12, 12), np.uint16)
    labels[3:6, 3:6, 3:6] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'control', 'material': 'tumor', 'target': True}], {}, {})
    result = SurgicalEngine(scene)
    for _ in range(6):
        assert result.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
    return result


def test_hot_seek_exact_effect_state_branch_isolation_and_no_repeat_simulation(monkeypatch):
    full = engine()
    snapshot = full.snapshot()
    cache = SurgicalReplayCache()
    for position in (0, 3, 6, 2, 5):
        actual = cache.seek(full, snapshot, position)
        reference = full.branch(position)
        assert actual.state == reference.state
        assert actual.history == reference.history
        assert actual.state_id == reference.state_id
    target = cache.seek(full, snapshot, 3)
    target.state['observations'].append({'tampering': True})
    target.actions[0]['duration'] = 999
    target.history[0]['stateAfter'] = 'tampered'
    expected = full.branch(3)
    monkeypatch.setattr(SurgicalEngine, 'step', lambda *_: pytest.fail('revisited state re-executed physics'))
    actual = cache.seek(full, snapshot, 3)
    assert actual.state == expected.state
    assert actual.actions == expected.actions
    assert actual.history == expected.history


def test_changed_archive_and_source_require_fresh_admission():
    full = engine()
    cache = SurgicalReplayCache()
    snapshot = full.snapshot()
    cache.seek(full, snapshot, 0)
    corrupted = copy.deepcopy(snapshot)
    corrupted['actions'][0]['duration'] = .2
    with pytest.raises(ValueError, match='identity differs'):
        cache.seek(full, corrupted, 3)
    corrupted = copy.deepcopy(snapshot)
    corrupted['checkpoint']['state']['fluidEscapedMm3'] = 7
    with pytest.raises(ValueError, match='Checkpoint differs'):
        cache.seek(full, corrupted, 3)
    corrupted = copy.deepcopy(snapshot)
    corrupted['protectionContract'] = {'contentHash': 'changed-contract'}
    with pytest.raises((ValueError, KeyError)):
        cache.seek(full, corrupted, 3)
    changed = engine()
    changed.scene.identity = 'different-source'
    with pytest.raises(ValueError, match='Source scene changed'):
        cache.seek(changed, snapshot, 3)


def test_on_demand_checkpoints_and_archive_memory_plateau():
    full = engine()
    cache = SurgicalReplayCache(max_checkpoints=3, max_archives=2, max_bytes=200_000)
    for _ in range(3):
        for position in (0, 1, 2, 3, 4, 5, 6):
            actual = cache.seek(full, full.snapshot(), position)
            assert actual.state_id == full.branch(position).state_id
            stats = cache.statistics
            assert stats['ownedBytes'] <= stats['maxBytes']
            assert stats['checkpoints'] <= 3
    tiny = SurgicalReplayCache(max_bytes=1)
    assert tiny.seek(full, full.snapshot(), 2).state == full.branch(2).state
    assert tiny.statistics['ownedBytes'] == 0


def test_cancel_between_replayed_actions_never_publishes_partial_checkpoint():
    full = engine()
    cache = SurgicalReplayCache()
    snapshot = full.snapshot()
    cache.seek(full, snapshot, 0)
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        cache.seek(full, snapshot, 4, check=check)
    assert cache.seek(full, snapshot, 4).state == full.branch(4).state


def test_repeated_cancelled_distinct_archives_stay_bounded_without_mutating_authority():
    full = engine()
    cache = SurgicalReplayCache(max_archives=1, max_checkpoints=1, max_bytes=200_000)
    for _ in range(12):
        assert full.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
        snapshot = full.snapshot()
        before_state, before_actions, before_history = copy.deepcopy((full.state, full.actions, full.history))
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            # The new endpoint has been admitted, but the requested zero cursor
            # has not yet passed its publication check or gained a checkpoint.
            if calls == 3:
                raise RuntimeError('cancel after archive admission')
        with pytest.raises(RuntimeError, match='cancel after archive admission'):
            cache.seek(full, snapshot, 0, check=check)
        stats = cache.statistics
        assert stats['archives'] <= 1
        assert stats['checkpoints'] <= stats['maxCheckpoints']
        assert stats['ownedBytes'] <= stats['maxBytes']
        assert full.state == before_state and full.actions == before_actions and full.history == before_history
    assert cache.seek(full, full.snapshot(), 0).state == full.branch(0).state


def test_source_arrays_cannot_be_mutated_or_made_writeable():
    source = engine().scene
    for array in (source.labels, source.affine, source.inverse, source.spacing):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 123


@pytest.mark.parametrize('field', ['regions', 'source', 'properties', 'labels', 'affine', 'voxel_mm3'])
def test_unversioned_scene_changes_cannot_reuse_cached_acceptance(field):
    full = engine()
    cache = SurgicalReplayCache()
    snapshot = full.snapshot()
    cache.seek(full, snapshot, 0)
    if field == 'regions':
        full.scene.regions[0]['material'] = 'bone'
    elif field in ('source', 'properties'):
        getattr(full.scene, field)['changed'] = True
    elif field == 'voxel_mm3':
        full.scene.voxel_mm3 *= 2
    else:
        setattr(full.scene, field, getattr(full.scene, field).copy())
    with pytest.raises(ValueError, match='Source scene changed'):
        cache.seek(full, snapshot, 3)


def test_branch_archives_share_one_pure_runtime_but_never_effect_state():
    full = engine()
    cache = SurgicalReplayCache(max_checkpoints=8, max_archives=4)
    first = cache.seek(full, full.snapshot(), 3)
    # The independent reference creates another grid. Admission must not retain
    # one patient grid and one cover LRU per archived branch.
    separate = SurgicalEngine.restore(full.scene, full.snapshot())
    assert separate.grid is not first.grid
    assert separate.step({'slot': 'left', 'operation': 'inspect', 'duration': .2})['status'] == 'accepted'
    second = cache.seek(separate, separate.snapshot(), 3)
    assert second.grid is first.grid
    assert second._cell_cover_cache is first._cell_cover_cache
    assert second.state is not first.state
    assert cache.statistics['sharedGeometryGrids'] == 1


def test_bridge_warm_cursor_uses_detached_state_and_verified_view_hash_without_redundant_clone(tmp_path, monkeypatch):
    from resectionlab.core import thaw_json
    from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
    session = BridgeSession(tmp_path / 'transfers')
    def call(body, request=None):
        return session.execute('surgicalRehearsal', body, request or _Request('cursor'), lambda *_: None)
    try:
        case = session.execute('createSurgicalFixture', {}, _Request('fixture'), lambda *_: None)
        key = case['caseHash']
        result = call({'caseHash': key, 'command': 'create'})
        for _ in range(3):
            result = call({'caseHash': key, 'command': 'step', 'expectedStateId': result['view']['stateId'],
                           'action': {'slot': 'left', 'operation': 'inspect', 'duration': .1}})
        full = session.surgical_sessions[key][1]
        reference = {position: json.dumps(full.branch(position).view(), sort_keys=True, separators=(',', ':')).encode()
                     for position in (0, 1, 3)}
        for position in (0, 3, 1):
            result = call({'caseHash': key, 'command': 'seek', 'position': position, 'expectedStateId': result['view']['stateId']})
            assert json.dumps(result['view'], sort_keys=True, separators=(',', ':')).encode() == reference[position]
        getter = SurgicalEngine.state_id.fget
        hashes = []
        monkeypatch.setattr(SurgicalEngine, 'state_id', property(lambda engine: hashes.append(id(engine)) or getter(engine)))
        original_clone = SurgicalEngine.clone
        clones = []
        monkeypatch.setattr(SurgicalEngine, 'clone', lambda engine: clones.append(id(engine)) or original_clone(engine))
        result = call({'caseHash': key, 'command': 'seek', 'position': 1, 'expectedStateId': result['view']['stateId']})
        assert len(hashes) == 2, 'retain expected-state and complete reconstructed-state checks'
        assert clones == [], 'a hot cursor must not clone the engine it only reads'
        assert json.dumps(result['view'], sort_keys=True, separators=(',', ':')).encode() == reference[1]
        record = thaw_json(session.cases[key].artifacts)['surgical_toolbox']
        assert result['view']['stateId'] == record['timeline'][record['branchId']][0]['stateAfter']
        authority, artifact = session.surgical_sessions[key][1], session.cases[key].artifacts
        before = copy.deepcopy(authority.state)
        request = _Request('cancel-cursor')
        original_substep = SurgicalEngine._substep
        def interrupt(engine, *args):
            events = original_substep(engine, *args)
            request.cancel()
            return events
        monkeypatch.setattr(SurgicalEngine, '_substep', interrupt)
        with pytest.raises(BridgeError, match='cancelled'):
            call({'caseHash': key, 'command': 'seek', 'position': 2, 'expectedStateId': result['view']['stateId']}, request)
        assert session.surgical_sessions[key][1] is authority and authority.state == before
        assert session.cases[key].artifacts is artifact
        with pytest.raises(BridgeError) as stale:
            call({'caseHash': key, 'command': 'seek', 'position': 0, 'expectedStateId': 'untrusted-client-state'})
        assert stale.value.code == 'SURGICAL_STATE_STALE'
    finally:
        session.close()


def test_owned_frozen_artifact_admission_reuses_only_the_exact_installed_object(tmp_path, monkeypatch):
    from types import MappingProxyType
    import resectionlab.surgical_bridge as bridge
    from resectionlab.core import freeze_json, thaw_json
    from resectionlab.desktop_bridge import BridgeSession, _Request
    session = BridgeSession(tmp_path / 'transfers')
    def call(body):
        return session.execute('surgicalRehearsal', body, _Request('artifact-admission'), lambda *_: None)
    try:
        case = session.execute('createSurgicalFixture', {}, _Request('fixture'), lambda *_: None)
        key = case['caseHash']
        initial = call({'caseHash': key, 'command': 'create'})
        entry = session.cases[key]
        installed = entry.artifacts
        assert isinstance(installed, MappingProxyType)
        snapshot = installed['surgical_toolbox']['branches']['main']
        assert isinstance(snapshot, MappingProxyType) and isinstance(snapshot['actions'], tuple)
        with pytest.raises(TypeError):
            snapshot['checkpoint']['state']['elapsedSeconds'] = 999
        with pytest.raises(TypeError):
            snapshot['checkpoint']['state']['slots']['left']['position'][0] = 999
        restore = SurgicalEngine.restore.__func__
        restored = []
        monkeypatch.setattr(SurgicalEngine, 'restore', classmethod(lambda cls, *a, **kw: restored.append(a[1]) or restore(cls, *a, **kw)))
        artifact_hashes = []
        digest = bridge.semantic_digest
        monkeypatch.setattr(bridge, 'semantic_digest', lambda value: artifact_hashes.append(value) or digest(value))
        same = call({'caseHash': key, 'command': 'inspect'})
        assert same['view'] == initial['view'] and restored == []
        assert artifact_hashes == [], 'owned frozen bundle artifacts need no repeated raw/record hashes'
        assert session.surgical_sessions[key][3] is entry.artifacts
        pristine = thaw_json(entry.artifacts)
        # An identical but separately frozen object has no prior admission.
        entry.artifacts = freeze_json(pristine)
        copied = call({'caseHash': key, 'command': 'inspect'})
        assert len(restored) == 1 and copied['view'] == initial['view']
        authority = session.surgical_sessions[key][1]
        corrupt = copy.deepcopy(pristine)
        corrupt['surgical_toolbox']['branches']['main']['checkpoint']['state']['fluidEscapedMm3'] = 999
        entry.artifacts = freeze_json(corrupt)
        with pytest.raises(ValueError, match='Checkpoint differs'):
            call({'caseHash': key, 'command': 'inspect'})
        assert session.surgical_sessions[key][1] is authority
        source_changed = copy.deepcopy(pristine)
        source_changed['surgical_toolbox']['branches']['main']['sceneId'] = 'different-source'
        entry.artifacts = freeze_json(source_changed)
        with pytest.raises(ValueError, match='Saved scene'):
            call({'caseHash': key, 'command': 'inspect'})
        assert session.surgical_sessions[key][1] is authority
    finally:
        session.close()


def test_previous_scene_selection_never_reuses_same_object_hot_admission(tmp_path, monkeypatch):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    session = BridgeSession(tmp_path / 'transfers')
    def call(body):
        return session.execute('surgicalRehearsal', body, _Request('scene-exchange'), lambda *_: None)
    try:
        case = session.execute('createSurgicalFixture', {}, _Request('fixture'), lambda *_: None)
        key = case['caseHash']
        original = call({'caseHash': key, 'command': 'create'})
        mapping = copy.deepcopy(original['view']['scene']['regions'])
        mapping[0]['target'] = not mapping[0]['target']
        next_scene = call({'caseHash': key, 'command': 'create', 'mapping': mapping})
        assert next_scene['view']['scene']['sceneId'] != original['view']['scene']['sceneId']
        restore = SurgicalEngine.restore.__func__
        restored = []
        monkeypatch.setattr(SurgicalEngine, 'restore', classmethod(lambda cls, *a, **kw: restored.append(a[1]) or restore(cls, *a, **kw)))
        selected = call({'caseHash': key, 'command': 'selectScene', 'sceneIndex': 0,
                         'expectedStateId': next_scene['view']['stateId']})
        assert len(restored) == 1
        assert selected['view'] == original['view']
    finally:
        session.close()


@pytest.mark.parametrize('material,tool,operations', [
    ('bone', 'cranial_drill', [('apply', [8., 8., 10.], .3)]),
    ('vessel', 'micro_scissors', [('apply', [8., 8., 7.], 1.), ('inspect', None, .4)]),
    ('soft_tissue', 'brain_retractor', [('hold', [8., 8., 7.], .2), ('move', [8.4, 8., 7.], .4), ('release', None, .01)]),
])
def test_irreversible_v2_effects_and_timed_relaxation_match_full_reconstruction(material, tool, operations):
    from resectionlab.surgical_physics_v2 import default_profile
    labels = np.zeros((24, 24, 20), np.uint16)
    if material == 'soft_tissue':
        labels[6:11, 6:11, 8] = 1
    else:
        labels[8, 7:10, 8] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'control', 'material': material}], {},
                          {'tissuePhysicsV2': default_profile()})
    full = SurgicalEngine(scene)
    actions = [{'slot': 'left', 'operation': 'exchange', 'instrument': tool, 'duration': .2},
               {'slot': 'left', 'operation': 'place', 'duration': .2,
                'pose': {'position': [8., 8., 4.], 'quaternion': [0., 0., 0., 1.]},
                'parameters': {'radiusMm': 1.5, 'jawOpeningMm': 0.}}]
    for operation, point, duration in operations:
        actions.append({'slot': 'left', 'operation': operation, 'duration': duration,
            **({'pose': {'position': point, 'quaternion': [0., 0., 0., 1.]}} if point else {}),
            **({'parameters': {'rpm': 1000., 'forceN': .1, 'duty': 1.}} if material == 'bone' else {})})
    for action in actions:
        assert full.step(action)['status'] in ('accepted', 'partial')
    if material == 'bone':
        assert full.state['drillingV2']['workJ']
    elif material == 'vessel':
        assert full.state['vesselsV2']['generatedMm3'] > 0
    else:
        assert full.state['deformation']
    cache = SurgicalReplayCache()
    for position in [0, len(actions) - 1, len(actions), 1, len(actions) - 1, len(actions)]:
        actual = cache.seek(full, full.snapshot(), position)
        reference = full.branch(position)
        assert actual.state == reference.state
        assert actual.view() == reference.view()
        assert actual.state_id == reference.state_id
