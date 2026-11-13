"""Exact engine graph/memo contracts for the bounded native copier."""
import copy
import gc
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from resectionlab import native_backend
from resectionlab import surgical_engine as engine


@pytest.fixture
def native():
    return pytest.importorskip('resectionlab._geometry_cpp')


def test_native_preserves_aliases_cycles_atomic_keys_order_and_float_bits(native):
    row = [1, -0., float('nan'), float('inf'), float('-inf'), True, None, 'é😀', 10**200]
    mapping = {None: row, False: row, 2: -0., 2.5: float('inf'), 'last': row}
    source = {'row': row, 'mapping': mapping, 'again': mapping}
    source['cycle'] = [source]
    memo = {}
    copied = native.copy_engine_graph(source, memo, object())
    assert list(copied) == list(source)
    assert list(copied['mapping']) == list(mapping)
    assert copied['mapping'] is copied['again']
    assert copied['row'] is copied['mapping'][None] is copied['mapping'][False]
    assert copied['cycle'][0] is copied
    assert copied['row'] is not row and copied['mapping'] is not mapping
    assert memo[id(source)] is copied and memo[id(row)] is copied['row']
    for index in (1, 2, 3, 4):
        assert struct.pack('d', copied['row'][index]) == struct.pack('d', row[index])
    copied['row'].append('caller-owned')
    assert len(row) == 9


def test_existing_memo_values_and_component_exemption_take_precedence(native):
    class Components(list):
        def __iter__(self):
            raise AssertionError('memo-exempt value was traversed')

    component = Components([object()])
    source = {'flapComponents': component, 'alias': component, 'removed': [1]}
    memo = {id(component): component}
    copied = native.copy_engine_graph(source, memo, object())
    assert copied['flapComponents'] is copied['alias'] is component
    assert copied['removed'] == [1] and copied['removed'] is not source['removed']
    assert memo[id(component)] is component


@pytest.mark.parametrize('unusual', [object(), (1, 2), np.int64(2), np.array([1.])])
def test_unsupported_graph_leaves_original_memo_and_input_untouched(native, unusual):
    exempt, seed = [1], object()
    source = {'valid': [1, {'value': -0.}], 'unusual': unusual}
    source['cycle'] = source
    memo = {id(exempt): seed}
    before = dict(memo)
    marker = object()
    assert native.copy_engine_graph(source, memo, marker) is marker
    assert list(memo) == list(before)
    assert all(memo[key] is value for key, value in before.items())
    assert source['valid'] == [1, {'value': -0.}] and source['cycle'] is source
    assert source['unusual'] is unusual


def test_unsupported_hooks_and_memo_keys_are_not_invoked(native):
    calls = []

    class Custom:
        def __deepcopy__(self, memo):
            calls.append('deepcopy')
            return {'custom': True}

        def __hash__(self):
            calls.append('hash')
            return 7

        def __eq__(self, other):
            calls.append('eq')
            return False

    custom = Custom()
    value = {custom: 1}
    memo = {custom: object()}
    calls.clear()
    marker = object()
    assert native.copy_engine_graph({'leaf': custom}, {}, marker) is marker
    assert native.copy_engine_graph(value, {}, marker) is marker
    assert native.copy_engine_graph([1], memo, marker) is marker
    assert calls == []


def test_bounds_are_unsupported_before_memo_publication(native):
    marker = object()
    deep = []
    for _ in range(native.MAX_ENGINE_COPY_DEPTH + 1):
        deep = [deep]
    for source in (deep, [0] * native.MAX_ENGINE_COPY_ITEMS):
        seed = object()
        memo = {11: seed}
        assert native.copy_engine_graph(source, memo, marker) is marker
        assert memo == {11: seed}


@pytest.mark.parametrize('container', [list, dict])
def test_gc_finalizer_can_clear_source_and_inspect_tracked_lists(native, container):
    gc.collect()
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    gc.disable()
    source = ([[index] for index in range(1000)] if container is list else
              {str(index): [index] for index in range(1000)})
    calls = []

    class Mutator:
        def __del__(self):
            for tracked in gc.get_objects():
                if type(tracked) is list:
                    tuple(tracked)
            calls.append(True)
            source.clear()

    doomed = Mutator()
    doomed.cycle = doomed
    del doomed
    gc.set_threshold(gc.get_count()[0] + 20, 1000000, 1000000)
    gc.enable()
    try:
        marker = object()
        copied = native.copy_engine_graph(source, {}, marker)
        assert calls == [True]
        assert copied is not marker and type(copied) is container
        assert source == container()
    finally:
        gc.set_threshold(*thresholds)
        if not enabled:
            gc.disable()


def test_production_fallback_calls_custom_deepcopy_once(monkeypatch):
    calls = []

    class Custom:
        def __deepcopy__(self, memo):
            calls.append('deepcopy')
            return {'detached': [1]}

    source = {'valid': [1], 'custom': Custom()}
    copied = engine._copy_engine_data(source)
    assert calls == ['deepcopy']
    assert copied == {'valid': [1], 'custom': {'detached': [1]}}
    assert copied['valid'] is not source['valid']


def test_native_integrity_failure_is_not_retried(monkeypatch):
    def failed(*args):
        raise native_backend.NativeIntegrityError('changed graph copier')

    def unexpected(*args):
        raise AssertionError('native error entered reference copier')

    monkeypatch.setattr(engine, 'native_engine_copy', failed)
    monkeypatch.setattr(engine, '_copy_engine_data_reference', unexpected)
    with pytest.raises(native_backend.NativeIntegrityError, match='changed graph copier'):
        engine._copy_engine_data({'valid': [1]})


def test_checkpoint_retains_original_deepcopy_hook(monkeypatch):
    calls = []
    original = copy.deepcopy
    source = {'removed': [1], 'debris': {}, 'nested': [2]}

    def installed(value, memo):
        calls.append(value)
        return original(value, memo)

    monkeypatch.setattr(copy, 'deepcopy', installed)
    assert engine._microstep_checkpoint(source) == source
    assert calls[0] is source


@pytest.mark.parametrize('kind', [dict, list, float, int, bool, str, type(None)])
def test_checkpoint_retains_mutable_deepcopy_dispatch(monkeypatch, kind):
    calls = []

    def installed(value, memo):
        calls.append(value)
        return 'installed-copy-hook'

    source = {'removed': [], 'debris': {}, 'nested': [1, 2., True, None, 'leaf']}
    monkeypatch.setitem(copy._deepcopy_dispatch, kind, installed)
    expected = copy.deepcopy(source, {id(source['removed']): source['removed'].copy(),
                                     id(source['debris']): source['debris'].copy()})
    calls.clear()
    assert engine._microstep_checkpoint(source) == expected
    assert calls


def test_checkpoint_retains_in_place_dispatch_code_change(monkeypatch):
    function = copy._deepcopy_dispatch[dict]

    def installed(value, memo, deepcopy=None):
        return {'installed-copy-hook': True}

    monkeypatch.setattr(function, '__code__', installed.__code__)
    source = {'removed': [], 'debris': {}, 'nested': [1]}
    assert engine._microstep_checkpoint(source) == {'installed-copy-hook': True}


@pytest.mark.parametrize('setup, expected', [
    ('', {'removed': [], 'debris': {}, 'nested': [1]}),
    ("def installed(value, memo, deepcopy=None):\n return {'preimport-hook': True}\n"
     "copy._deepcopy_dispatch[dict].__code__ = installed.__code__", {'preimport-hook': True}),
    ("copy._deepcopy_dispatch[dict].__defaults__ = (lambda value, memo: 'preimport-default',)",
     {'preimport-default': 'preimport-default'}),
    ("def unavailable(name):\n raise ImportError('no stock loader code')\n"
     "copy.__spec__.loader.get_code = unavailable", {'removed': [], 'debris': {}, 'nested': [1]}),
])
def test_fresh_import_qualifies_stock_code_and_retains_preimport_hooks(setup, expected):
    program = "import copy, json\nfrom scipy.spatial.transform import Rotation, Slerp\n" + setup + "\n" + '''
from resectionlab import surgical_engine as engine
source = {'removed': [], 'debris': {}, 'nested': [1]}
print(json.dumps({'stock': engine._engine_copy_genuine,
                  'checkpoint': engine._microstep_checkpoint(source)}))
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', program], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src')), text=True, capture_output=True, timeout=30)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result['stock'] is (not setup)
    assert result['checkpoint'] == expected
