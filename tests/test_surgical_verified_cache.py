"""Authenticated disposable cache never replaces independent native replay."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache


def scene():
    labels = np.zeros((8, 8, 8), np.uint16)
    labels[3:5, 3:5, 3:5] = 1
    return SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'fixture', 'material': 'tumor', 'target': True}],
                         {'kind': 'explicit-engineering-fixture'}, {})


def engine(count=4):
    result = SurgicalEngine(scene())
    for i in range(count):
        assert result.step({'slot': 'left', 'operation': 'inspect', 'duration': .1 + i / 100})['status'] == 'accepted'
    return result


def test_only_complete_independent_replay_seals_and_cached_states_are_detached(tmp_path):
    full = engine()
    record = full.snapshot()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    assert SurgicalEngine.restore(full.scene, record, verified_cache=cache).snapshot() == record
    assert not cache.directory.exists(), 'default restoration remains independent'
    restored = SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    assert cache.publications == 1 and cache.hits == 0
    assert restored.snapshot() == record and restored.history == full.history
    assert (cache.directory.stat().st_mode & 0o777) == 0o700
    assert ((cache.directory / 'key').stat().st_mode & 0o777) == 0o600
    cached = SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    assert cache.hits == 1 and cached.snapshot() == record
    cached.state['observations'].append({'untrusted': True})
    cached.history[0]['stateAfter'] = 'untrusted'
    for position in range(len(full.actions) + 1):
        actual, timeline = cache.restore(SurgicalEngine, full.scene, record, position=position, include_history=True)
        reference = full.branch(position)
        assert actual.snapshot() == reference.snapshot()
        assert actual.history == reference.history and timeline == full.history
    owner = SurgicalReplayCache(verified_cache=cache)
    assert owner.seek(full, record, 2).snapshot() == full.branch(2).snapshot()
    cursor, timeline = owner.seek(full, record, 2, include_history=True)
    timeline[0]['stateAfter'] = 'caller-owned-timeline'
    assert cursor.history[0]['stateAfter'] == full.history[0]['stateAfter']
    assert cache.statistics['ownedMemoryCacheBytes'] == 0


def test_fresh_process_reads_only_matching_signed_implementation(tmp_path):
    full = engine()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    record = full.snapshot()
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(record))
    script = '''
import json, sys, numpy as np
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache
labels=np.zeros((8,8,8),np.uint16);labels[3:5,3:5,3:5]=1
s=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'fixture','material':'tumor','target':True}],{'kind':'explicit-engineering-fixture'},{})
r=json.load(open(sys.argv[1]));c=VerifiedSurgicalCache(sys.argv[2])
e=SurgicalEngine.restore(s,r,use_cache=True,verified_cache=c)
print(json.dumps({'stateId':e.state_id,'hits':c.hits,'publications':c.publications,'snapshotEqual':e.snapshot()==r}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    out = subprocess.run([sys.executable, '-c', script, str(path), str(cache.directory)],
                         env=env, capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result == {'stateId': full.state_id, 'hits': 1, 'publications': 0, 'snapshotEqual': True}


@pytest.mark.parametrize('target', ['manifest.json.gz', 'manifest.mac', '2.json.gz'])
def test_tampered_cache_files_fall_back_to_full_verification(tmp_path, target):
    full = engine()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    record = full.snapshot()
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    entry = next(cache.directory.glob('entry-*'))
    path = entry / target
    data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1; path.write_bytes(data)
    position = 2 if target == '2.json.gz' else len(full.actions)
    assert cache.restore(SurgicalEngine, full.scene, record, position=position) is None
    restored = SurgicalEngine.restore(full.scene, record, use_cache=True, force_verify=True, verified_cache=cache)
    assert restored.snapshot() == record and cache.publications == 2


@pytest.mark.parametrize('field', ['checkpoint', 'action'])
def test_native_tampering_is_not_admitted_or_sealed(tmp_path, field):
    full = engine()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    record = full.snapshot()
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    corrupt = copy.deepcopy(record)
    if field == 'checkpoint':
        corrupt['checkpoint']['state']['fluidAddedMm3'] = 999
    else:
        corrupt['actions'][0]['duration'] = .9
    with pytest.raises(ValueError, match='Checkpoint|identity'):
        SurgicalEngine.restore(full.scene, corrupt, use_cache=True, verified_cache=cache)
    assert cache.publications == 1
    assert not list(cache.directory.glob('.pending-*'))


def test_source_mutation_cannot_reuse_a_signed_entry(tmp_path):
    full = engine()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    record = full.snapshot()
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    full.scene.properties['changed'] = True
    with pytest.raises(ValueError, match='Source scene changed'):
        SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    assert cache.hits == 0


def test_insecure_key_permissions_disable_cache_admission(tmp_path):
    full = engine()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    record = full.snapshot()
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    (cache.directory / 'key').chmod(0o644)
    result = SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    assert result.snapshot() == record
    assert cache.hits == 0 and cache.publications == 1


def test_runtime_changes_and_force_verify_do_not_skip_public_replay(tmp_path, monkeypatch):
    full = engine()
    record = full.snapshot()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    assert SurgicalEngine.restore(full.scene, record, use_cache=True, force_verify=True, verified_cache=cache).snapshot() == record
    assert cache.publications == 2 and cache.hits == 0
    original = SurgicalEngine.step
    calls = []
    def changed(self, action, **kwargs):
        calls.append(action)
        return original(self, action, **kwargs)
    monkeypatch.setattr(SurgicalEngine, 'step', changed)
    restored = SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    assert restored.snapshot() == record and len(calls) == len(full.actions)
    assert cache.hits == 0 and cache.publications == 2


def test_cancellation_never_publishes_a_partial_seal_and_cache_hits_still_check(tmp_path):
    full = engine()
    record = full.snapshot()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    checks = 0
    failure = ValueError('cancelled canonical replay')
    def check():
        nonlocal checks
        checks += 1
        if checks == 12:
            raise failure
    with pytest.raises(ValueError) as caught:
        SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache, check=check)
    assert caught.value is failure
    assert cache.publications == 0 and not list(cache.directory.glob('entry-*'))
    assert not list(cache.directory.glob('.pending-*'))
    SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache)
    def cancelled():
        raise InterruptedError('cancelled cached read')
    with pytest.raises(InterruptedError):
        SurgicalEngine.restore(full.scene, record, use_cache=True, verified_cache=cache, check=cancelled)


def test_disk_budget_is_bounded_and_oversized_admission_retains_normal_replay(tmp_path):
    cache = VerifiedSurgicalCache(tmp_path / 'tiny', max_disk_bytes=100)
    full = engine()
    assert SurgicalEngine.restore(full.scene, full.snapshot(), use_cache=True, verified_cache=cache).state_id == full.state_id
    assert cache.publications == 0 and cache.statistics['diskBytes'] <= 100
    bounded = VerifiedSurgicalCache(tmp_path / 'bounded', max_disk_bytes=6000)
    for count in range(2, 9):
        full = engine(count)
        assert SurgicalEngine.restore(full.scene, full.snapshot(), use_cache=True, verified_cache=bounded).state_id == full.state_id
        assert bounded.statistics['diskBytes'] <= bounded.max_disk_bytes
    assert bounded.statistics['ownedMemoryCacheBytes'] == 0


@pytest.mark.parametrize('material,tool,operations', [
    ('bone', 'cranial_drill', [('apply', [8., 8., 10.], .3)]),
    ('vessel', 'micro_scissors', [('apply', [8., 8., 7.], 1.), ('inspect', None, .4)]),
    ('soft_tissue', 'brain_retractor', [('hold', [8., 8., 7.], .2), ('move', [8.4, 8., 7.], .4), ('release', None, .01)]),
])
def test_signed_v2_state_continues_with_identical_physics_after_reconstruction(tmp_path, material, tool, operations):
    from resectionlab.surgical_physics_v2 import default_profile
    labels = np.zeros((24, 24, 20), np.uint16)
    if material == 'soft_tissue':
        labels[6:11, 6:11, 8] = 1
    else:
        labels[8, 7:10, 8] = 1
    source = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'fixture', 'material': material}], {},
                           {'tissuePhysicsV2': default_profile()})
    full = SurgicalEngine(source)
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
    record = full.snapshot()
    cache = VerifiedSurgicalCache(tmp_path / material)
    SurgicalEngine.restore(source, record, use_cache=True, verified_cache=cache)
    cached = SurgicalEngine.restore(source, record, use_cache=True, verified_cache=cache)
    assert cache.hits == 1 and cached.snapshot() == record
    reference = full.clone()
    next_action = {'slot': 'right', 'operation': 'inspect', 'duration': .3}
    assert cached.step(next_action) == reference.step(next_action)
    assert cached.snapshot() == reference.snapshot()
