"""Opt-in C++ stroke and surface-audit kernels equal their Python references."""
import numpy as np
import pytest
from skimage.measure import marching_cubes

from resectionlab import _accel_loader, accel_geometry, acceleration, native_resection, surface_fairing

native = _accel_loader.module()
needs_native = pytest.mark.skipif(native is None, reason='python -m resectionlab.build_accel_native not run')


def _reference(name, *args, **kwargs):
    with acceleration.override():
        return getattr(native_resection, name)(*args, **kwargs)


def _accelerated(name, *args, **kwargs):
    with acceleration.override('stroke'):
        return getattr(native_resection, name)(*args, **kwargs)


# --- stroke -----------------------------------------------------------------

def _stroke_case(seed, shape=(14, 12, 10)):
    rng = np.random.default_rng(seed)
    remaining = rng.random(shape) < rng.uniform(.35, .85)
    free = ~remaining & (rng.random(shape) < rng.uniform(0., .6))
    cells = np.argwhere(remaining)
    take = rng.choice(len(cells), size=min(len(cells), int(rng.integers(1, 120))), replace=False)
    return rng, remaining, free, cells[take].astype(np.int64)


@needs_native
@pytest.mark.parametrize('seed', range(40))
def test_connected_surface_cells_match_reference(seed):
    _, _, free, candidates = _stroke_case(seed)
    expected = _reference('_connected_surface_cells', candidates, free)
    actual = _accelerated('_connected_surface_cells', candidates, free)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    frozen = native_resection._frozen(free, bool)  # Committed masks are immutable views.
    np.testing.assert_array_equal(_accelerated('_connected_surface_cells', candidates, frozen), expected)


@needs_native
@pytest.mark.parametrize('seed', range(40))
@pytest.mark.parametrize('with_domain', [False, True])
def test_extend_connected_free_matches_reference_including_refusals(seed, with_domain):
    rng, remaining, free, cells = _stroke_case(seed)
    remaining[tuple(cells.T)] = False
    domain = (rng.random(remaining.shape) < .97) | free if with_domain else None
    if with_domain:
        domain[tuple(cells[: len(cells) // 2].T)] = True
    outcomes = []
    for run in (_reference, _accelerated):
        mask = free.copy()
        try:
            run('_extend_connected_free', cells, remaining, mask, domain)
            error = None
        except ValueError as exc:
            error = str(exc)
        outcomes.append((mask, error))
    (expected, expected_error), (actual, actual_error) = outcomes
    assert actual_error == expected_error
    np.testing.assert_array_equal(actual, expected)  # Partial updates match too.


@needs_native
def test_stroke_kernels_defer_on_inputs_they_do_not_accept():
    _, remaining, free, cells = _stroke_case(3)
    assert accel_geometry.connected_surface_cells(cells.astype(float), free) is None
    assert accel_geometry.connected_surface_cells(cells, free.astype(np.uint8)) is None
    assert accel_geometry.connected_surface_cells(cells - 20, free) is None
    assert not accel_geometry.extend_connected_free(cells, remaining, free[:, ::-1], None)
    assert not accel_geometry.extend_connected_free(cells, remaining, native_resection._frozen(free, bool), None)
    assert not accel_geometry.extend_connected_free(cells, remaining.astype(np.uint8), free.copy(), None)


def test_stroke_hooks_fall_back_without_the_extension(monkeypatch):
    _, remaining, free, cells = _stroke_case(5)
    expected = _reference('_connected_surface_cells', cells, free)
    monkeypatch.setattr(_accel_loader, 'module', lambda: None)
    np.testing.assert_array_equal(_accelerated('_connected_surface_cells', cells, free), expected)


# --- surface-audit -----------------------------------------------------------

def _surface(seed, size=18):
    rng = np.random.default_rng(seed)
    grid = np.indices((size,) * 3).astype(float)
    centre = (size - 1) / 2 + rng.uniform(-.5, .5, 3)
    field = sum(((grid[i] - centre[i]) / (size * rng.uniform(.25, .4))) ** 2 for i in range(3))
    field += rng.standard_normal(field.shape) * .05
    points, faces, _, _ = marching_cubes(field, 1.)
    return rng, points.astype(float), faces.astype(np.int64)


def _audit(enabled, *args):
    with acceleration.override(*(['surface-audit'] if enabled else [])):
        return surface_fairing._intersection_vertices(*args)


@needs_native
def test_projection_order_self_check_selects_one_order():
    assert accel_geometry.projection_order() in (0, 1, 2)


@needs_native
@pytest.mark.parametrize('seed', range(16))
@pytest.mark.parametrize('amplitude', [.05, .4, 1.5])
def test_intersection_vertices_match_reference(seed, amplitude):
    rng, source, faces = _surface(seed)
    candidate = source + rng.standard_normal(source.shape) * amplitude
    changed = None if seed % 2 else rng.random(len(source)) < .5
    expected = _audit(False, source, candidate, faces, lambda: False, changed)
    actual = _audit(True, source, candidate, faces, lambda: False, changed)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if amplitude >= .4:
        assert len(expected)  # The comparison exercised real contacts.


@needs_native
def test_intersection_audit_cancels_on_the_reference_cadence():
    rng, source, faces = _surface(7, size=24)
    candidate = source + rng.standard_normal(source.shape) * .2
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) == 2

    for enabled in (False, True):
        calls.clear()
        with pytest.raises(InterruptedError, match='Surface intersection audit cancelled'):
            _audit(enabled, source, candidate, faces, cancelled)
        assert len(calls) == 2
    counts = []
    for enabled in (False, True):
        calls.clear()
        _audit(enabled, source, candidate, faces, lambda: calls.append(1) and False)
        counts.append(len(calls))
    assert counts[0] == counts[1] == -(-len(faces) // 512)


@needs_native
def test_fair_surface_is_bit_identical_with_the_kernel():
    grid = np.indices((24, 24, 24))
    mask = ((grid - 11.5) ** 2 / np.array([60., 40., 30.])[:, None, None, None]).sum(0) < 1
    mask[3:6, 10:14, 8:12] = True
    points, faces, _, _ = marching_cubes(mask.astype(float), .5)
    affine = np.diag([1.2, 1.2, 4., 1.])
    results = []
    for enabled in (False, True):
        with acceleration.override(*(['surface-audit'] if enabled else [])):
            results.append(surface_fairing.fair_surface(points, faces, affine, 'brain', native=True))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


@needs_native
def test_audit_defers_on_inputs_it_does_not_accept():
    rng, source, faces = _surface(1)
    candidate = source + .3
    assert accel_geometry.intersection_vertices(source.astype(np.float32), candidate, faces, lambda: False) is None
    bad = candidate.copy(); bad[0, 0] = np.nan
    assert accel_geometry.intersection_vertices(source, bad, faces, lambda: False) is None
    assert accel_geometry.intersection_vertices(source, candidate, faces - len(source), lambda: False) is None
