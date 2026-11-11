"""Differential checks for the native JSON detach boundary."""
import importlib
import gc
import json
import math
from types import MappingProxyType

import numpy as np
import pytest

from resectionlab import core


@pytest.fixture
def native():
    return pytest.importorskip("resectionlab._geometry_cpp")


@pytest.mark.parametrize("value", [
    None, True, False, 0, -17, 10 ** 200, "é😀\x00\ud800", -0.0,
    math.nextafter(0.0, 1.0), 1.7976931348623157e308, [], {},
    {"z": [None, True, 1, -0.0], "a": {"unicode": "é😀"}},
])
def test_builtin_copy_matches_reference_bytes_and_types(native, value):
    expected = core._copy_builtin_json(value)
    copied = native.copy_builtin_json(value, core._JSON_COPY_FALLBACK)
    assert type(copied) is type(expected)
    assert json.dumps(copied, allow_nan=False) == json.dumps(expected, allow_nan=False)
    if type(value) is float:
        assert math.copysign(1.0, copied) == math.copysign(1.0, value)
    if type(value) is dict:
        assert list(copied) == list(value)


def test_copy_detaches_containers_without_preserving_aliases(native):
    shared = {"values": [1, -0.0]}
    source = {"first": shared, "second": shared}
    copied = native.copy_builtin_json(source, core._JSON_COPY_FALLBACK)
    assert copied == source
    assert copied is not source
    assert copied["first"] is not shared
    assert copied["first"] is not copied["second"]
    assert copied["first"]["values"] is not shared["values"]
    copied["first"]["values"].append(3)
    assert source == {"first": {"values": [1, -0.0]}, "second": {"values": [1, -0.0]}}


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), float("-inf"), object(), 1j,
    {1: 2}, (1, 2), MappingProxyType({"frozen": (1, 2)}),
    np.int64(2), np.float64(2), np.array([1.0]),
])
def test_unsupported_returns_the_supplied_marker(native, value):
    marker = object()
    assert native.copy_builtin_json(value, marker) is marker
    assert core._copy_builtin_json(value) is core._JSON_COPY_FALLBACK


def test_subclass_callbacks_are_not_invoked(native):
    class UnusualList(list):
        def __iter__(self):
            raise AssertionError("subclass iteration")

    class UnusualDict(dict):
        def items(self):
            raise AssertionError("subclass items")

    class UnusualStr(str):
        pass

    class UnusualInt(int):
        pass

    class UnusualFloat(float):
        pass

    marker = object()
    for value in (UnusualList([1]), UnusualDict(a=1), UnusualStr("leaf"),
                  UnusualInt(1), UnusualFloat(1), {UnusualStr("key"): 1}):
        assert native.copy_builtin_json(value, marker) is marker


def test_depth_bound_and_cycles_match_original(native):
    marker = object()
    for depth in (0, 30, 31, 32, 33):
        for value in (0, [], [0], {"key": 0}):
            expected = core._copy_builtin_json(value, depth)
            copied = native.copy_builtin_json(value, marker, depth)
            if expected is core._JSON_COPY_FALLBACK:
                assert copied is marker
            else:
                assert copied == expected
    source = []
    source.append(source)
    assert native.copy_builtin_json(source, marker) is marker
    assert len(source) == 1 and source[0] is source
    # Negative depths are unsupported at the native boundary, keeping recursion bounded.
    assert native.copy_builtin_json(source, marker, -1) is marker


@pytest.mark.parametrize("container", [list, dict])
def test_gc_finalizer_can_mutate_source_during_copy(native, container):
    gc.collect()
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    gc.disable()
    source = ([[index] for index in range(2000)] if container is list else
              {str(index): [index] for index in range(2000)})
    calls = []

    class Mutator:
        def __del__(self):
            calls.append(True)
            source.clear()

    doomed = Mutator()
    doomed.cycle = doomed
    del doomed
    # Let native container allocations trigger collection after entering the copy.
    gc.set_threshold(gc.get_count()[0] + 20, 1000000, 1000000)
    gc.enable()
    try:
        marker = object()
        copied = native.copy_builtin_json(source, marker)
        assert calls == [True]
        assert copied is not marker
        assert type(copied) is container
        assert source == container()
        json.dumps(copied, allow_nan=False)
    finally:
        gc.set_threshold(*thresholds)
        if not enabled:
            gc.disable()


def test_digest_uses_native_copy_before_reference(native, monkeypatch):
    source = {"z": [-0.0, 2], "a": True}
    expected = core.semantic_digest(source)
    monkeypatch.setattr(core, "native_json_copy", native.copy_builtin_json)

    def unexpected_reference(*_args):
        raise AssertionError("supported JSON entered reference copier")

    monkeypatch.setattr(core, "_copy_builtin_json", unexpected_reference)
    assert core.semantic_digest(source) == expected


def test_digest_propagates_native_errors_without_retry(monkeypatch):
    backend = importlib.import_module("resectionlab.native_backend")

    def failed_copy(*_args):
        raise backend.NativeIntegrityError("changed binary")

    def unexpected_reference(*_args):
        raise AssertionError("native failure entered reference copier")

    monkeypatch.setattr(core, "native_json_copy", failed_copy)
    monkeypatch.setattr(core, "_copy_builtin_json", unexpected_reference)
    with pytest.raises(backend.NativeIntegrityError, match="changed binary"):
        core.semantic_digest({"a": [1]})
