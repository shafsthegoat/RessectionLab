"""Native ordered predicate screening, checked ownership and atomic publication."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from resectionlab import native_backend

native = pytest.importorskip('resectionlab._geometry_cpp')

EPS = 1e-9


def sequential(points, thresholds, function=native.segment_predicate):
    for index, threshold in enumerate(thresholds):
        result = function(points[index], float(threshold), EPS)
        if result:
            return index, result, index + 1
    return -1, 0, len(points)


def point_rows(count=3):
    points = np.zeros((count, 4, 3), np.float64)
    points[:, 2:, 0] = 5.
    return points, np.ones(count)


@pytest.mark.parametrize('kind', ['point', 'point-segment', 'parallel', 'skew', 'tiny', 'grazing', 'mixed'])
def test_ordered_batch_screen_matches_each_scalar_predicate(kind):
    rng = np.random.default_rng(1173)
    points = rng.uniform(-30., 30., (64, 4, 3))
    if kind == 'point':
        points[:, 1] = points[:, 0]; points[:, 3] = points[:, 2]
    elif kind == 'point-segment':
        points[:, 1] = points[:, 0]
    elif kind == 'parallel':
        points[:, 1] = points[:, 0]; points[:, 1, 2] += 10.
        points[:, 3] = points[:, 2]; points[:, 3, 2] += 20.
    elif kind == 'tiny':
        points[:, 1] = points[:, 0] + rng.uniform(-1e-9, 1e-9, (64, 3))
    elif kind == 'grazing':
        points[:] = [[0., 0., 0.], [0., 0., 0.], [3., 4., 0.], [3., 4., 0.]]
    thresholds = rng.uniform(0., 15., 64)
    if kind == 'grazing':
        thresholds[:] = 5.
        thresholds[::3] = np.nextafter(5., 0.)
        thresholds[1::3] = np.nextafter(5., np.inf)
    for size in (1, 5, 8, 16, 64):
        for offset in (0, 64 - size):
            selected = points[offset:offset + size].copy(), thresholds[offset:offset + size].copy()
            assert native.segment_predicate_batch(*selected, EPS) == sequential(*selected)


def test_empty_and_all_false_exact_row_accounting():
    assert native.segment_predicate_batch(np.empty((0, 4, 3)), np.empty(0), EPS) == (-1, 0, 0)
    points, thresholds = point_rows(64)
    assert native.segment_predicate_batch(points, thresholds, EPS) == (-1, 0, 64)


@pytest.mark.parametrize('first_status', [1, -1])
@pytest.mark.parametrize('late_invalid', ['endpoint-nan', 'threshold-nan', 'threshold-negative', 'overflow'])
def test_later_bad_rows_are_not_validated_or_computed_after_first_stop(first_status, late_invalid):
    points, thresholds = point_rows(3)
    thresholds[0] = 6. if first_status == 1 else 5.
    if late_invalid == 'endpoint-nan':
        points[1, 0, 0] = np.nan
    elif late_invalid == 'threshold-nan':
        thresholds[1] = np.nan
    elif late_invalid == 'threshold-negative':
        thresholds[1] = -1.
    else:
        points[1, 2:] = 1e308
    before = points.tobytes(), thresholds.tobytes()
    assert native.segment_predicate_batch(points, thresholds, EPS) == (0, first_status, 1)
    assert native.segment_predicate_batch(points, thresholds, EPS) == sequential(points, thresholds)
    assert before == (points.tobytes(), thresholds.tobytes())
    thresholds[0] = 1.
    error_type = RuntimeError if late_invalid == 'overflow' else ValueError
    with pytest.raises(error_type):
        native.segment_predicate_batch(points, thresholds, EPS)


@pytest.mark.parametrize('invalid', ['dtype', 'byteorder', 'strides', 'shape', 'threshold-dtype',
    'threshold-strides', 'threshold-shape', 'count', 'endpoint-nan', 'threshold-nan',
    'threshold-negative', 'eps-zero', 'eps-nan', 'overflow'])
def test_invalid_consumed_data_or_full_buffer_contract_raises_without_mutation(invalid):
    points, thresholds = point_rows(2)
    eps = EPS
    if invalid == 'dtype': points = points.astype(np.float32)
    elif invalid == 'byteorder': points = points.astype('>f8')
    elif invalid == 'strides': points = np.repeat(points, 2, axis=2)[:, :, ::2]
    elif invalid == 'shape': points = points.reshape(2, 3, 4)
    elif invalid == 'threshold-dtype': thresholds = thresholds.astype(np.float32)
    elif invalid == 'threshold-strides': thresholds = np.ones(4)[::2]
    elif invalid == 'threshold-shape': thresholds = thresholds[:, None]
    elif invalid == 'count': points, thresholds = point_rows(65)
    elif invalid == 'endpoint-nan': points[0, 0, 0] = np.nan
    elif invalid == 'threshold-nan': thresholds[0] = np.nan
    elif invalid == 'threshold-negative': thresholds[0] = -1.
    elif invalid == 'eps-zero': eps = 0.
    elif invalid == 'eps-nan': eps = np.nan
    elif invalid == 'overflow': points[0, 2:] = 1e308
    before = points.tobytes(), thresholds.tobytes()
    errors = (RuntimeError,) if invalid == 'overflow' else (ValueError, TypeError)
    with pytest.raises(errors): native.segment_predicate_batch(points, thresholds, eps)
    assert before == (points.tobytes(), thresholds.tobytes())


def test_readonly_aliases_and_returned_owned_immutable_values():
    points = np.zeros((2, 4, 3), np.float64)
    thresholds = points.ravel()[:2]
    points.setflags(write=False); thresholds.setflags(write=False)
    result = native.segment_predicate_batch(points, thresholds, EPS)
    assert result == (0, -1, 1)
    assert type(result) is tuple and all(type(item) is int for item in result)
    points.setflags(write=True); points[:] = 100.
    assert result == (0, -1, 1)


def test_ambiguity_stops_before_a_later_hit_and_remains_reference_authority():
    points, thresholds = point_rows(3)
    thresholds[:] = [1., 5., 6.]
    assert native.segment_predicate_batch(points, thresholds, EPS) == (1, -1, 2)


@pytest.mark.parametrize('invalid, message', [
    ('endpoint', 'native geometry metadata must be finite'),
    ('threshold', 'invalid segment predicate threshold/epsilon'),
    ('overflow', 'native segment predicate overflow')])
def test_consumed_numerical_errors_match_existing_scalar(invalid, message):
    points, thresholds = point_rows(2)
    if invalid == 'endpoint': points[0, 0, 0] = np.nan
    elif invalid == 'threshold': thresholds[0] = np.nan
    else: points[0, 2:] = 1e308
    for operation in (lambda: native.segment_predicate(points[0], float(thresholds[0]), EPS),
                      lambda: native.segment_predicate_batch(points, thresholds, EPS)):
        with pytest.raises((ValueError, RuntimeError)) as error:
            operation()
        assert str(error.value) == message


def require_selected_cpp():
    if not native_backend.cpp_enabled():
        pytest.skip('Select cpp or auto for the checked native adapter')


def test_adapter_has_separate_scalar_batch_and_capsule_counters(monkeypatch):
    require_selected_cpp()
    points, thresholds = point_rows(3); thresholds[1] = 5.
    native_backend.reset_counters()
    calls = []
    original = native_backend._check_live
    def checked():
        calls.append(1)
        original()
    with monkeypatch.context() as patch:
        patch.setattr(native_backend, '_check_live', checked)
        assert native_backend.try_segment_batch(points, thresholds, EPS) == (1, -1, 2)
    assert len(calls) == 2
    counters = native_backend.backend_identity()['counters']
    assert counters['bindingCalls'] == counters['geometryBindingCalls'] == 1
    assert counters['segmentBatchCalls'] == 1
    assert counters['segmentBatchItems'] == 3 and counters['segmentBatchExamined'] == 2
    assert counters['segmentPredicateCalls'] == counters['segmentPredicateRefinements'] == 0
    assert counters['batchCalls'] == counters['batchItems'] == 0


@pytest.mark.parametrize('eps', [0., -1., float('nan'), float('inf'), np.float64(EPS), 1, None])
def test_adapter_declines_invalid_or_non_builtin_float_eps_without_touching_buffers(eps):
    require_selected_cpp()
    native_backend.reset_counters()
    assert native_backend.try_segment_batch(object(), object(), eps) is None
    assert native_backend.backend_identity()['counters']['bindingCalls'] == 0


def test_adapter_declines_custom_float_eps_without_invoking_hooks():
    require_selected_cpp()
    class Float(float):
        def __float__(self): raise AssertionError('custom float conversion')
        def __le__(self, other): raise AssertionError('custom float comparison')
    assert native_backend.try_segment_batch(object(), object(), Float(EPS)) is None


def test_adapter_propagates_arithmetic_failure_without_retry(monkeypatch):
    require_selected_cpp()
    points, thresholds = point_rows(2); points[0, 2:] = 1e308
    native_backend.reset_counters()
    checks = []
    original = native_backend._check_live
    def checked():
        checks.append(1)
        original()
    with monkeypatch.context() as patch:
        patch.setattr(native_backend, '_check_live', checked)
        with pytest.raises(RuntimeError, match='native segment predicate overflow'):
            native_backend.try_segment_batch(points, thresholds, EPS)
    assert checks == [1]
    assert native_backend.backend_identity()['counters']['bindingCalls'] == 1


def test_adapter_checks_integrity_after_computation_before_return(monkeypatch):
    require_selected_cpp()
    points, thresholds = point_rows(2)
    calls = []
    original = native_backend._check_live
    def checked():
        calls.append(1)
        if len(calls) == 2:
            raise native_backend.NativeIntegrityError('changed during computation')
        original()
    with monkeypatch.context() as patch:
        patch.setattr(native_backend, '_check_live', checked)
        with pytest.raises(native_backend.NativeIntegrityError, match='changed during computation'):
            native_backend.try_segment_batch(points, thresholds, EPS)
    assert calls == [1, 1]


def test_adapter_rejects_replaced_batch_entrypoint_before_execution(monkeypatch):
    require_selected_cpp()
    points, thresholds = point_rows(2)
    native_backend.reset_counters()
    with monkeypatch.context() as patch:
        patch.setattr(native, 'segment_predicate_batch', lambda *args: (-1, 0, 2))
        with pytest.raises(native_backend.NativeIntegrityError):
            native_backend.try_segment_batch(points, thresholds, EPS)
    assert native_backend.backend_identity()['counters']['bindingCalls'] == 0


def child(mode, program):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, RESSECTIONLAB_NUMERICAL_BACKEND=mode, PYTHONPATH=str(root / 'src'))
    result = subprocess.run([sys.executable, '-c', program], env=environment, cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_reference_adapter_has_no_native_batch_invocations():
    result = child('reference', """
import json
from resectionlab import native_backend as backend
assert backend.try_segment_batch(object(), object(), 1e-9) is None
print(json.dumps(backend.backend_identity()))
""")
    assert result['selected'] == 'reference'
    assert result['counters']['bindingCalls'] == result['counters']['segmentBatchCalls'] == 0


def test_preloaded_replacement_of_batch_entrypoint_fails_admission():
    result = child('cpp', """
import json
from resectionlab import _geometry_cpp
_geometry_cpp.segment_predicate_batch = lambda *args: (-1, 0, 0)
try:
    from resectionlab import native_backend
except RuntimeError as error:
    print(json.dumps({'refused': True, 'error': str(error)}))
else:
    print(json.dumps({'refused': False}))
""")
    assert result['refused']
