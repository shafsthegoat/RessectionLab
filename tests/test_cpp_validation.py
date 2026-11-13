"""Generated complete validation checks, including the original custom-input path."""
import gc

import numpy as np
import pytest

from resectionlab import native_backend as backend
from resectionlab import surgical_craniotomy as cranial
from resectionlab.core import immutable_array
from resectionlab.surgical_scene import SurgicalScene


@pytest.fixture
def compiled():
    if backend._module is None:
        pytest.skip('compiled validation required')
    return backend._module


def outcome(function):
    try:
        return 'return', function()
    except Exception as error:
        return type(error).__name__, str(error)


def cell_case():
    scene = object.__new__(SurgicalScene)
    scene.labels = immutable_array(np.array([1, 1, 1, 2, 0, 0, 0, 0], np.uint16).reshape(2, 2, 2))
    scene.regions = [{'id': 1, 'material': 'bone'}, {'id': 2, 'material': 'tumor'}]
    scene.properties = {'boneAnchors': [2]}
    raw = {'windowCells': [0, 1], 'flapCells': [0], 'kerfCells': [1],
           'retainedSkullCells': [2], 'boneAnchorCells': [2]}
    return scene, raw


def literal_cells_partition(raw, scene):
    parsed = [cranial._cells(raw[name], scene.labels.size, name, empty=name == 'boneAnchorCells')
              for name in ('windowCells', 'flapCells', 'kerfCells', 'retainedSkullCells', 'boneAnchorCells')]
    window, flap, kerf, retained, anchors = map(set, parsed)
    bone = set(map(int, np.flatnonzero(np.isin(scene.labels, [r['id'] for r in scene.regions if r['material'] == 'bone']))))
    if flap & kerf or retained & (flap | kerf) or flap | kerf | retained != bone:
        cranial._invalid('F/K/R must partition all categorical source bone')
    if not (flap | kerf) <= window <= bone or not anchors <= retained:
        cranial._invalid('window must contain F union K; anchors must belong to retained skull')
    if not set(scene.properties.get('boneAnchors', ())) <= anchors:
        cranial._invalid('declared source bone anchors cannot be omitted or reassigned')
    return parsed


def native_cells(compiled, raw, scene, marker):
    return compiled.cranial_cells_partition(raw, scene, (SurgicalScene, np.ndarray),
        cranial._CELL_NAMESPACE, cranial._validation_builtins.__dict__, cranial._CELL_BINDINGS, marker)


@pytest.mark.parametrize('field,values', [
    (None, None), ('windowCells', (0, 1)), ('windowCells', []), ('windowCells', [1, 0]),
    ('windowCells', [0, 0, 1]), ('windowCells', [-1]), ('windowCells', [10**100]),
    ('flapCells', [0, 1]), ('retainedSkullCells', [1, 2]), ('retainedSkullCells', [2, 3]),
    ('windowCells', [0]), ('windowCells', [0, 1, 3]), ('boneAnchorCells', []),
    ('boneAnchorCells', [0]),
])
def test_complete_cell_partition_matches_reference(compiled, field, values):
    scene, raw = cell_case()
    if field:
        raw[field] = values
    expected = outcome(lambda: literal_cells_partition(raw, scene))
    marker = object()
    actual = outcome(lambda: native_cells(compiled, raw, scene, marker))
    assert actual == expected
    if actual[0] == 'return' and field == 'windowCells':
        assert actual[1][0] is values


def test_randomized_partition_and_ordering_cases(compiled):
    random = np.random.default_rng(412)
    for _ in range(150):
        scene, raw = cell_case()
        for field in raw:
            raw[field] = sorted(random.choice(8, size=int(random.integers(0, 5)), replace=False).tolist())
        assert outcome(lambda: native_cells(compiled, raw, scene, None)) == outcome(
            lambda: literal_cells_partition(raw, scene))


def test_custom_cell_inputs_and_patched_numpy_stay_on_reference_path(compiled, monkeypatch):
    scene, raw = cell_case()
    class Integer(int):
        pass
    raw['flapCells'] = [Integer(0)]
    marker = object()
    assert native_cells(compiled, raw, scene, marker) is marker
    monkeypatch.setattr(np, 'isin', lambda *args, **kwargs: None)
    assert not cranial._native_cells_supported()


def test_source_anchor_omission_and_big_integer_match(compiled):
    for anchors in ([1], [-1], [10**100], []):
        scene, raw = cell_case()
        scene.properties['boneAnchors'] = anchors
        assert outcome(lambda: native_cells(compiled, raw, scene, None)) == outcome(
            lambda: literal_cells_partition(raw, scene))


def test_modified_numpy_keyword_defaults_disable_native_cells(compiled, monkeypatch):
    implementation = np.isin._implementation
    assert cranial._native_cells_supported()
    monkeypatch.setattr(implementation, '__kwdefaults__', dict(implementation.__kwdefaults__))
    implementation.__kwdefaults__['kind'] = 'sort'
    assert not cranial._native_cells_supported()


def test_larger_source_scan_keeps_compiled_numpy_path(compiled):
    scene, raw = cell_case()
    scene.labels = immutable_array(np.zeros((1, 1, 1048577), np.uint16))
    marker = object()
    assert native_cells(compiled, raw, scene, marker) is marker


def test_output_allocation_gc_cannot_invalidate_cell_snapshots(compiled):
    scene, raw = cell_case()
    expected = literal_cells_partition(raw, scene)
    marker = object()
    function = compiled.cranial_cells_partition
    arguments = (raw, scene, (SurgicalScene, np.ndarray), cranial._CELL_NAMESPACE,
                 cranial._validation_builtins.__dict__, cranial._CELL_BINDINGS, marker)
    assert function(*arguments) == expected  # Warm array API and materialize the scene dictionary.
    groups = tuple(raw.values())
    events = []
    class Collected:
        def __del__(self):
            events.append('collected')
            for group in groups:
                group.clear()
            raw.clear()
            scene.regions.clear()
            scene.properties.clear()
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    gc.disable()
    gc.collect()
    cycle = Collected()
    cycle.self = cycle
    del cycle
    try:
        gc.set_threshold(1, 1000000, 1000000)
        gc.enable()
        actual = function(*arguments)
        assert events == ['collected']
        assert actual == expected
        assert raw == {} and all(group == [] for group in groups)
    finally:
        gc.set_threshold(*thresholds)
        if not enabled:
            gc.disable()
