"""Independent generated controls for native behavior, never patient evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _require_extension():
    if importlib.util.find_spec('resectionlab._geometry_cpp') is None:
        pytest.skip('Build the C++ extension to run independent native acceptance')


def _child(mode, program):
    environment = dict(os.environ, RESSECTIONLAB_NUMERICAL_BACKEND=mode,
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                            env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr[-6000:]
    return json.loads(result.stdout)


RL_TRACE = r'''
import hashlib
import json
import numpy as np
from resectionlab import native_backend
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet

def bits(value):
    if isinstance(value, np.ndarray):
        return ['array', value.dtype.str, list(value.shape), value.tobytes().hex()]
    if isinstance(value, (float, np.floating)):
        return ['float', float(value).hex()]
    if isinstance(value, np.generic):
        return bits(value.item())
    if isinstance(value, dict):
        return {str(key): bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [bits(item) for item in value]
    return value

def digest(value):
    return hashlib.sha256(json.dumps(bits(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def command(operation='inspect', **fields):
    return {'slot': 'left', 'operation': operation, 'duration': .1, **fields}

def pose(x, z):
    return {'position': [x, 9., z], 'quaternion': [0., 0., 0., 1.]}

labels = np.zeros((24, 24, 20), np.uint16)
labels[8:11, 8:11, 8:11] = 1
scene = SurgicalScene(labels, np.eye(4), [{'id': 1,
    'name': 'Generated independent RL target', 'material': 'tumor', 'target': True}],
    {'kind': 'generated independent software acceptance control'},
    {'tissuePhysicsV2': default_profile()})

def row(environment, result):
    observation, reward, terminated, truncated, info = result
    assert environment.observation_space.contains(observation)
    return {'observation': {key: digest(value) for key, value in observation.items()},
        'reward': float(reward).hex(), 'terminated': terminated, 'truncated': truncated,
        'info': digest(info), 'engine': digest(environment.engine.snapshot()),
        'episode': digest(environment.snapshot()), 'status': info['outcome']['status']}

success = SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False))
initial = success.reset(seed=318)
rows = [{'initial': digest(initial)}]
actions = [
    command('place', pose={'position': [9., 9., 9.], 'quaternion': [0., 0., 0., 0.]}),
    command('exchange', instrument='cusa'),
    command('place', pose=pose(8., -5.), parameters={'radiusMm': 1.5}),
    command('apply', pose=pose(8., 11.), duration=1.),
    command('withdraw', duration=1.),
    command('place', pose=pose(10., -5.)),
    command('apply', pose=pose(10., 11.), duration=1.),
    command('withdraw', duration=1.)]
for action in actions:
    rows.append(row(success, success.step_action(action)))
assert rows[1]['status'] == 'rejected'
assert rows[-1]['terminated'] and not rows[-1]['truncated']
assert success.termination_reason == 'target_fraction_reached'
assert success.attempts[-1]['terms']['tumor_removal'] == 0
assert success.attempts[-2]['terms']['tumor_removal'] > 0
restored = SurgicalRLEnv(scene, config=success.config)
restored.restore(json.loads(json.dumps(success.snapshot(), allow_nan=False)))
assert digest(restored.snapshot()) == digest(success.snapshot())
assert digest(restored._observe()) == digest(success._observe())
rows.append({'restored': digest(restored.engine.snapshot())})

truncated = SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False, max_steps=2))
truncated.reset(seed=318)
rows.append(row(truncated, truncated.step_action(command())))
rows.append(row(truncated, truncated.step_action(command())))
assert rows[-1]['truncated'] and not rows[-1]['terminated']
assert truncated.termination_reason == 'attempt_limit'

mask = np.zeros(scene.shape, bool)
mask[9, 9, 8] = True
annotation = ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
    provenance='Generated overlap-exclusion control; no acquired anatomy')
protection = ProtectionSet(scene, [annotation], clearance_mm=1., strict=True)
guarded = SurgicalRLEnv(scene, protection=protection)
guarded.reset(seed=318)
for action in [command('exchange', instrument='cusa'),
        command('place', pose=pose(9., -5.), parameters={'radiusMm': .5}),
        command('apply', pose=pose(9., 9.)), command('withdraw')]:
    rows.append(row(guarded, guarded.step_action(action)))
assert int(np.flatnonzero(mask)[0]) not in guarded.engine.state['removed']
assert guarded.measurements['protected_damage_mm3'] == 0
assert any(item['status'] in ('rejected', 'partial') for item in guarded.attempts)
print(json.dumps({'rows': rows, 'backend': native_backend.backend_identity()}, allow_nan=False))
'''


def test_generated_rl_observation_reward_termination_and_restore_match_exactly():
    _require_extension()
    pytest.importorskip('gymnasium')
    reference = _child('reference', RL_TRACE)
    compiled = _child('cpp', RL_TRACE)
    assert compiled['rows'] == reference['rows']
    assert reference['backend']['selected'] == 'reference'
    assert reference['backend']['counters']['bindingCalls'] == 0
    assert compiled['backend']['selected'] == 'cpp'
    assert compiled['backend']['counters']['bindingCalls'] > 0


def test_orthogonal_capsules_match_independent_scalar_box_oracle():
    _require_extension()
    from resectionlab.evaluation import segment_box_distance_sq
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices_batch

    rotation = np.array([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    affine = np.eye(4)
    spacing = np.array([1.7, .6, 2.1])
    affine[:3, :3] = rotation @ np.diag(spacing)
    affine[:3, 3] = [8., -3., 11.]
    scene = GeometryScene(np.zeros((7, 6, 5), bool), affine)
    cells = np.array(list(np.ndindex(scene.shape)), dtype=np.int64)
    lower, upper = (cells - .5) * spacing, (cells + .5) * spacing
    rng = np.random.default_rng(741)
    queries = []
    expected = []
    for start, end, radius in [([2.2, 1.2, 2.], [2.2, 1.2, 2.], .23),
            ([-2., 1.2, 2.], [15., 1.2, 2.], .23),
            *[(rng.uniform(-2, 12, 3), rng.uniform(-2, 12, 3), .31) for _ in range(9)]]:
        start, end = np.asarray(start), np.asarray(end)
        queries.append((rotation @ start + affine[:3, 3],
                        rotation @ end + affine[:3, 3], radius))
        distances = np.array([segment_box_distance_sq(start, end, lo, hi)
                              for lo, hi in zip(lower, upper)])
        # These generated controls stay away from safety thresholds; grazing
        # controls below compare directly with the retained exact predicate.
        assert np.min(np.abs(np.sqrt(distances) - radius)) > 1e-5
        expected.append(cells[np.sqrt(distances) <= radius])
    actual = capsule_voxel_indices_batch(scene, queries)
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected):
        np.testing.assert_array_equal(observed, wanted)
        assert not observed.flags.writeable


def test_grazing_degenerate_and_sheared_ordered_covers_match_reference_in_fresh_processes():
    _require_extension()
    program = r'''
import hashlib, json
import numpy as np
from resectionlab.geometry import GeometryScene, capsule_voxel_indices_batch
from resectionlab import native_backend
results = []
for affine in (np.eye(4), np.diag([2.3, .7, 1.2, 1.]),
        np.array([[1., .6, 0., 0.], [0., 1., .3, 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])):
    scene = GeometryScene(np.zeros((9, 8, 7), bool), affine)
    queries = []
    for radius in (0., .2, np.float32(.2), np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.)):
        for start, end in (([2., 2., 2.], [2., 2., 2.]),
                ([2.5, 2., 2.], [2.5, 2., 2.]), ([2.5, 2.5, 2.5], [2.5, 2.5, 2.5]),
                ([-2., 3., 3.], [12., 3., 3.]), ([-1., -1., -1.], [7., 6., 5.]),
                ([100., 100., 100.], [100., 100., 100.])):
            queries.append((scene.voxel_to_world(start), scene.voxel_to_world(end), radius))
    for cells in capsule_voxel_indices_batch(scene, queries):
        results.append([cells.dtype.str, list(cells.shape), hashlib.sha256(cells.tobytes()).hexdigest()])
print(json.dumps({'results': results, 'backend': native_backend.backend_identity()}))
'''
    compiled, reference = _child('cpp', program), _child('reference', program)
    assert compiled['results'] == reference['results']
    assert compiled['backend']['counters']['bindingCalls'] > 0
    assert compiled['backend']['counters']['maxBatchItems'] > 1


def test_epsilon_degeneracy_branch_rounding_cannot_change_corner_contacts():
    _require_extension()
    program = r'''
import json
import numpy as np
from resectionlab import geometry
eps = geometry._EPS
affine = np.eye(4)
affine[:3, 3] = .5
scene = geometry.GeometryScene(np.zeros((3, 3, 3), bool), affine)
start = -np.ones(3) * (1.25 * eps / np.sqrt(3.))
rng = np.random.default_rng(124)
queries = []
for _ in range(10000):
    direction = rng.uniform(.1, 1., 3)
    direction /= np.linalg.norm(direction)
    end = start + direction * eps
    delta = end - start
    scalar_sq = sum(float(value) * float(value) for value in delta)
    dot_sq = float(delta @ delta)
    if (scalar_sq > eps * eps) != (dot_sq > eps * eps):
        queries.append((start.copy(), end, 0.))
        if len(queries) == 32:
            break
assert queries, 'This runtime provided no differing dot/sum degeneracy controls'
outputs = geometry.capsule_voxel_indices_batch(scene, queries)
print(json.dumps({'queries': len(queries), 'cells': [value.tolist() for value in outputs]}))
'''
    assert _child('cpp', program) == _child('reference', program)


@pytest.mark.parametrize('points,distance', [
    ([[0., 0., 0.], [0., 0., 0.], [3., 4., 0.], [3., 4., 0.]], 5.),
    ([[0., 0., 0.], [0., 0., 5.], [2., 0., 0.], [2., 0., 5.]], 2.),
    ([[-2., 0., 0.], [2., 0., 0.], [0., -2., 3.], [0., 2., 3.]], 3.),
    ([[-2., 0., 0.], [2., 0., 0.], [0., -2., 0.], [0., 2., 0.]], 0.),
    ([[-4e-10, 0., 0.], [4e-10, 0., 0.], [0., -10., 0.], [0., 10., 0.]], 0.),
])
def test_segment_collision_predicate_preserves_analytic_grazing_boundaries(points, distance):
    _require_extension()
    from resectionlab.geometry import segments_within_distance

    points = np.asarray(points)
    for order in ([0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 3, 2], [2, 3, 0, 1]):
        segments = points[order]
        if distance:
            assert not segments_within_distance(*segments, np.nextafter(distance, 0.))
        assert segments_within_distance(*segments, distance)
        assert segments_within_distance(*segments, np.nextafter(distance, np.inf))


def _binding_inputs():
    radius, eps = .2, 1e-9
    cell_radius = float(np.sqrt(3.) / 2)
    query = np.array([[0.] * 12 + [radius + eps, radius + cell_radius + eps]], np.float64)
    spans = np.array([[0, 0, 0, 3, 3, 3, 0, 64]], np.int64)
    return query, spans, np.eye(4), np.ones(3), cell_radius, eps


@pytest.mark.parametrize('invalid', ['float32', 'byteorder', 'strides', 'shape', 'span-dtype',
    'negative-count', 'negative-offset', 'out-of-bounds', 'oversized', 'nonfinite', 'spacing'])
def test_raw_binding_rejects_invalid_buffers_without_mutating_inputs(invalid):
    _require_extension()
    from resectionlab import _geometry_cpp

    arguments = list(_binding_inputs())
    if invalid == 'float32':
        arguments[0] = arguments[0].astype(np.float32)
    elif invalid == 'byteorder':
        arguments[0] = arguments[0].astype('>f8')
    elif invalid == 'strides':
        arguments[0] = np.repeat(arguments[0], 2, axis=1)[:, ::2]
    elif invalid == 'shape':
        arguments[0] = arguments[0].reshape(2, 7)
    elif invalid == 'span-dtype':
        arguments[1] = arguments[1].astype(np.int32)
    elif invalid == 'negative-count':
        arguments[1][0, 7] = -1
    elif invalid == 'negative-offset':
        arguments[1][0, 6] = -1
    elif invalid == 'out-of-bounds':
        arguments[1][0, 6] = 64
    elif invalid == 'oversized':
        arguments[1][0, 3:6] = 100
        arguments[1][0, 7] = 32769
    elif invalid == 'nonfinite':
        arguments[0][0, 0] = np.nan
    elif invalid == 'spacing':
        arguments[3][0] = -1.
    original = [value.tobytes() for value in arguments[:4]]
    with pytest.raises((ValueError, TypeError)):
        _geometry_cpp.capsule_chunk(*arguments)
    assert [value.tobytes() for value in arguments[:4]] == original


def test_raw_binding_returns_owned_outputs_and_accepts_readonly_inputs():
    _require_extension()
    from resectionlab import _geometry_cpp

    arguments = _binding_inputs()
    for value in arguments[:4]:
        value.setflags(write=False)
    outputs = _geometry_cpp.capsule_chunk(*arguments)
    assert len(outputs) == 1
    hits, ambiguous = outputs[0]
    np.testing.assert_array_equal(hits, [[0, 0, 0]])
    assert ambiguous.shape == (0, 3)
    assert hits.dtype == ambiguous.dtype == np.dtype(np.int64)
    for output in (hits, ambiguous):
        assert output.flags.owndata
        assert not any(np.shares_memory(output, value) for value in arguments[:4])


def test_preloaded_python_entrypoint_cannot_claim_a_compiled_backend():
    _require_extension()
    program = r'''
import json
from resectionlab import _geometry_cpp
_geometry_cpp.capsule_chunk = lambda *args: ()
try:
    from resectionlab import native_backend
except RuntimeError as error:
    print(json.dumps({'refused': True, 'error': str(error)}))
else:
    print(json.dumps({'refused': False, 'backend': native_backend.backend_identity()}))
'''
    assert _child('cpp', program)['refused']


def test_native_integrity_failure_permanently_invalidates_a_prefix(monkeypatch):
    _require_extension()
    from resectionlab import native_backend
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_prefix_replay import SurgicalPrefixReplay, PrefixInvalidated
    from resectionlab.surgical_scene import SurgicalScene

    labels = np.ones((2, 2, 2), np.uint16)
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Generated prefix control',
        'material': 'tumor', 'target': True}], {'kind': 'generated independent control'}, {})
    state = {'failed': False}
    original = native_backend.strict_live_identity
    def identity():
        if state['failed']:
            raise native_backend.NativeIntegrityError('independent loaded-binary drift control')
        return original()
    identity.__module__ = native_backend.__name__
    monkeypatch.setattr(native_backend, 'strict_live_identity', identity)
    prefix = SurgicalPrefixReplay(scene, SurgicalEngine(scene).snapshot())
    state['failed'] = True
    with pytest.raises(PrefixInvalidated):
        prefix.seek(0)
    state['failed'] = False
    with pytest.raises(PrefixInvalidated):
        prefix.seek(0)


def test_native_execution_counters_do_not_invalidate_implementation_proofs():
    _require_extension()
    from resectionlab import geometry, native_backend
    from resectionlab.surgical_observed_cache import light_implementation_seal
    from resectionlab.surgical_proof_cache import implementation_epoch
    from resectionlab.surgical_verified_cache import implementation_fingerprint

    if not native_backend.cpp_enabled():
        pytest.skip('Select cpp or auto to verify compiled proof stability')
    implementation_epoch()  # Complete the proof domain's existing lazy imports.
    functions = (implementation_fingerprint, light_implementation_seal, implementation_epoch)
    before = tuple(function() for function in functions)
    assert all(value is not None for value in before)
    identity = native_backend.strict_live_identity()
    calls = native_backend.backend_identity()['counters']['bindingCalls']
    scene = geometry.GeometryScene(np.zeros((3, 3, 3), bool), np.eye(4))
    geometry.capsule_voxel_indices(scene, [0., 0., 0.], [2., 2., 2.], .1)
    assert native_backend.backend_identity()['counters']['bindingCalls'] > calls
    assert native_backend.strict_live_identity() == identity
    assert tuple(function() for function in functions) == before


def test_largest_frozen_batch_cancels_within_budget_without_publishing(record_property):
    _require_extension()
    from resectionlab import geometry, native_backend
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene

    if not native_backend.cpp_enabled():
        pytest.skip('Select cpp or auto to measure native cancellation')
    manifest = json.loads((ROOT / 'manifests/experiments/cpp-acceleration-v1.json').read_text())
    contract = manifest['nativeCancellation']
    shape = tuple(contract['largestGeneratedQueryShape'])
    items = contract['maximumBatchItems']
    labels = np.zeros(shape, np.uint16)
    labels[tuple(size // 2 for size in shape)] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Generated cancellation target',
        'material': 'tumor', 'target': True}], {'kind': 'generated independent cancellation control'}, {})
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    # Distinct keys prevent cache deduplication. Each query traverses the full
    # frozen grid; cancellation occurs while one bounded native chunk executes.
    queries = tuple(([index * 1e-6, 0., 0.], [float(size - 1) for size in shape], 40.)
                    for index in range(items))
    latencies, chunks = [], []

    class Cancelled(Exception):
        pass

    for _ in range(5):
        requested = threading.Event()
        outcome = {}
        counters_before = native_backend.backend_identity()['counters']['geometryBindingCalls']
        def check():
            if requested.is_set():
                raise Cancelled('independent cancellation requested during native work')
        def work():
            try:
                engine._cell_cover_cache.query_many(engine.grid, queries,
                    kernel=geometry.capsule_voxel_indices, check=check)
                outcome['published'] = True
            except BaseException as error:
                outcome.update(error=error, stopped=time.perf_counter())
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        deadline = time.perf_counter() + 5.
        while worker.is_alive() and native_backend.backend_identity()['counters']['geometryBindingCalls'] == counters_before:
            if time.perf_counter() >= deadline:
                break
            requested.wait(.0001)
        requested_at = time.perf_counter()
        requested.set()
        worker.join(timeout=2.)
        assert not worker.is_alive(), 'Cancelled bounded native work did not terminate'
        assert isinstance(outcome.get('error'), Cancelled), outcome
        counters = native_backend.backend_identity()['counters']
        assert counters['geometryBindingCalls'] > counters_before, 'Cancellation preceded native computation'
        latencies.append(outcome['stopped'] - requested_at)
        chunks.append(counters['geometryBindingCalls'] - counters_before)
        assert 0 <= latencies[-1] <= contract['latencyBudgetSeconds'], latencies
        assert counters['maxPrimitiveCandidates'] <= contract['maximumPrimitiveCandidatesPerChunk']
        assert counters['maxBatchItems'] == items
        assert engine._cell_cover_cache.stats()['entries'] == 0
        assert engine.snapshot() == before
    record_property('nativeCancellation', json.dumps({'shape': shape, 'batchItems': items,
        'primitiveCap': contract['maximumPrimitiveCandidatesPerChunk'],
        'maxPrimitiveCandidates': counters['maxPrimitiveCandidates'],
        'budgetSeconds': contract['latencyBudgetSeconds'], 'latencySeconds': latencies,
        'bindingChunksBeforeStop': chunks, 'acceptedStateUnchanged': True,
        'publishedCoverEntries': engine._cell_cover_cache.stats()['entries']}))
