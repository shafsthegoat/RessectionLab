"""``rl-ledger``: the batched near-margin union equals per-capsule ``capsule_hits``."""
import numpy as np
import pytest

from resectionlab import accel_rl, acceleration, surgical_rl_protection as rp
from resectionlab.surgical_rl_env import RLConfig, SurgicalRLEnv
from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene

REGIONS = [dict(id=1, name='Engineering target', material='tumor', target=True),
           dict(id=2, name='brain', material='soft_tissue')]


def _scene(n=24):
    labels = np.full((n, n, n), 2, np.uint16)
    labels[8:12, 8:12, 8:12] = 1
    return SurgicalScene(labels, np.eye(4), REGIONS, {'kind': 'engineering-control', 'provenance': 'synthetic'}, {})


def _protection(scene, centre=(16, 16, 16), radius=5):
    g = np.indices(scene.shape)
    mask = (((g - np.array(centre)[:, None, None, None]) ** 2).sum(0) < radius ** 2) & (scene.labels == 2)
    return ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask, provenance='synthetic')],
                         clearance_mm=1.5)


def _points(protection):
    _, _, _, index, _, _, _, _ = rp._GEOMETRY[protection]
    return rp._INDEXES[index][0].data


def _reference(protection, capsules, positions):
    found = set()
    for a, b, radius in capsules:
        found.update(map(int, protection.capsule_hits(a, b, radius, positions=positions)))
    return found


@pytest.mark.parametrize('seed', range(30))
def test_capsule_union_matches_reference(seed):
    rng = np.random.default_rng(seed)
    protection = _protection(_scene())
    positions = _points(protection) + rng.normal(0, .2, _points(protection).shape)
    capsules = []
    for _ in range(int(rng.integers(1, 9))):
        a = rng.uniform(4, 20, 3)
        b = a if rng.random() < .2 else a + rng.normal(0, 4, 3)  # Includes zero-length capsules.
        capsules.append((a, b, float(rng.uniform(0, 4))))
    assert accel_rl.capsule_union(protection, capsules, positions) == _reference(protection, capsules, positions)


def test_threshold_boundary_cells_defer_to_the_reference(monkeypatch):
    protection = _protection(_scene())
    positions = _points(protection).copy()
    _, _, source_radius, *_ = rp._GEOMETRY[protection]
    a = positions[0] + [0., 0., -8.]
    b = positions[0] + [0., 0., -3.]
    # Cell 0 sits on the reference threshold, 3 mm beyond the capsule end.
    radius = 3. - protection.clearance_mm - source_radius - 1e-9
    assert radius > 0
    calls = []
    real = ProtectionSet.capsule_hits
    monkeypatch.setattr(ProtectionSet, 'capsule_hits', lambda self, *args, **kw: calls.append(1) or real(self, *args, **kw))
    monkeypatch.setattr(accel_rl, '_stock', lambda protection: True)
    capsules = [(a, b, radius), (a + 30, b + 30, 1.)]
    expected = set()
    for start, end, r in capsules:
        expected.update(map(int, real(protection, start, end, r, positions=positions)))
    calls.clear()
    assert accel_rl.capsule_union(protection, capsules, positions) == expected
    assert calls == [1]  # Only the ambiguous capsule was re-evaluated.


def test_unusual_inputs_return_none_for_the_reference():
    protection = _protection(_scene())
    positions = _points(protection)
    capsule = [(np.zeros(3), np.ones(3), 1.)]
    assert accel_rl.capsule_union(protection, capsule, None) is None
    assert accel_rl.capsule_union(protection, capsule, positions[:-1]) is None
    assert accel_rl.capsule_union(protection, [(np.zeros(3), np.full(3, np.nan), 1.)], positions) is None
    assert accel_rl.capsule_union(protection, [(np.zeros(3), np.ones(3), -1.)], positions) is None

    class Instrumented(ProtectionSet):
        def capsule_hits(self, *args, **kwargs):
            return super().capsule_hits(*args, **kwargs)

    protection.__class__ = Instrumented
    assert accel_rl.capsule_union(protection, capsule, positions) is None


def test_rl_steps_are_identical_with_the_batched_ledger():
    scene = _scene()
    protection = _protection(scene, centre=(10, 10, 16), radius=3)
    config = RLConfig(max_seconds=600., max_healthy_removed_mm3=1e9, max_vessel_damage_mm2=1e9,
                      max_bleeding_mm3=1e9)
    setup = [dict(slot='left', operation='exchange', instrument='cusa', duration=.1),
             dict(slot='left', operation='place', duration=.1, parameters={'radiusMm': .5},
                  pose={'position': [9., 9., -5.], 'quaternion': [0., 0., 0., 1.]})]
    action = dict(slot='left', operation='apply', duration=.2,
                  pose={'position': [9., 9., 9.], 'quaternion': [0., 0., 0., 1.]})
    results = []
    for features in ((), ('rl-ledger',)):
        with acceleration.override(*features):
            env = SurgicalRLEnv(scene, protection=protection, config=config)
            env.reset(seed=0)
            for command in setup:
                assert env.engine.step(command)['status'] == 'accepted'
            event = env.engine.step(action)
            results.append((event['status'], env.engine.rl_near_exposure, sorted(env.engine.rl_strain_peaks.items()),
                            sorted(map(int, env.engine.state['removed']))))
    assert results[0] == results[1]
