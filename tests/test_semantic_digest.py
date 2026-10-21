"""Digest parity against the literal pre-fast-path metadata normalization."""
from collections.abc import Mapping
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
import pytest

from resectionlab import core


def legacy_freeze(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata cannot contain nonfinite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({key: legacy_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(legacy_freeze(item) for item in value)
    raise TypeError(f"metadata must be JSON-compatible; got {type(value).__name__}")


def legacy_thaw(value):
    if isinstance(value, Mapping):
        return {key: legacy_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [legacy_thaw(item) for item in value]
    return value


def legacy_bytes(value):
    return json.dumps(legacy_thaw(legacy_freeze(value)), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def legacy_digest(value):
    return "sha256:" + hashlib.sha256(legacy_bytes(value)).hexdigest()


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class StrSubclass(str):
    pass


@pytest.mark.parametrize("value", [
    None, True, False, 0, -17, 10 ** 200, "é😀\x00", -0.0,
    math.nextafter(0.0, 1.0), 1.7976931348623157e308,
    {"z": [None, True, 1, -0.0], "a": {"unicode": "é😀"}},
    (1, {"tuple": (2, 3)}), MappingProxyType({"frozen": (1, 2)}),
    {"i": np.int64(2), "f": np.float64(-0.0)},
    DictSubclass({"x": 1}), ListSubclass([1, 2]),
    {"a": IntSubclass(2), "b": FloatSubclass(.5), "c": StrSubclass("é")},
    {StrSubclass("key"): [1]},
])
def test_digest_and_exact_encoded_bytes_match_legacy(value, monkeypatch):
    expected_bytes = legacy_bytes(value)
    expected_digest = legacy_digest(value)
    dumps = json.dumps
    encoded = []

    def capture(*args, **kwargs):
        result = dumps(*args, **kwargs)
        encoded.append(result.encode("utf-8"))
        return result

    monkeypatch.setattr(core.json, "dumps", capture)
    assert core.semantic_digest(value) == expected_digest
    assert encoded == [expected_bytes]


@pytest.mark.parametrize("value", [{1: 2}, float("nan"), float("inf"),
                                     np.array([1.]), object(), 10 ** 5000],
                         ids=["key", "nan", "inf", "array", "object", "huge-int"])
def test_legacy_error_type_and_message_are_preserved(value):
    with pytest.raises((ValueError, TypeError)) as expected:
        legacy_digest(value)
    with pytest.raises(type(expected.value)) as actual:
        core.semantic_digest(value)
    assert str(actual.value) == str(expected.value)


def test_plain_graph_is_detached_and_each_call_fresh(monkeypatch):
    shared = {"values": [1, -0.0]}
    value = {"a": shared, "b": shared}
    expected = legacy_digest(value)
    dumped = []
    dumps = json.dumps

    def mutate_then_dump(copied, **kwargs):
        dumped.append(copied)
        # This occurs after normalization, immediately before actual encoding.
        shared["values"].append(float("nan"))
        value[3] = object()
        assert copied is not value
        assert copied["a"] is not shared
        assert copied["a"]["values"] is not shared["values"]
        return dumps(copied, **kwargs)

    monkeypatch.setattr(core.json, "dumps", mutate_then_dump)
    assert core.semantic_digest(value) == expected
    assert dumped[0]["a"] == dumped[0]["b"] == {"values": [1, -0.0]}
    monkeypatch.setattr(core.json, "dumps", dumps)
    with pytest.raises(ValueError, match="keys must be strings"):
        core.semantic_digest(value)
    del value[3]
    shared["values"].pop()
    shared["values"][0] = 2
    assert core.semantic_digest(value) == legacy_digest(value) != expected


def test_fast_graph_avoids_freeze_but_fallback_starts_at_original(monkeypatch):
    freeze = core.freeze_json
    calls = []

    def record(value):
        calls.append(value)
        return freeze(value)

    monkeypatch.setattr(core, "freeze_json", record)
    assert core.semantic_digest({"plain": [1, 2.]}) == legacy_digest({"plain": [1, 2.]})
    assert calls == []
    exotic = {"plain": [1], "tuple": (2, 3)}
    assert core.semantic_digest(exotic) == legacy_digest(exotic)
    assert calls[0] is exotic


def test_unsupported_mapping_callbacks_are_not_speculatively_invoked():
    class CountedMapping(Mapping):
        def __init__(self):
            self.calls = []

        def __iter__(self):
            self.calls.append("iter")
            return iter(["leaf"])

        def __len__(self):
            self.calls.append("len")
            return 1

        def __getitem__(self, key):
            self.calls.append(("get", key))
            return [1, 2.]

    legacy = CountedMapping()
    candidate = CountedMapping()
    assert core.semantic_digest({"custom": candidate}) == legacy_digest({"custom": legacy})
    assert candidate.calls == legacy.calls


def test_deep_and_cyclic_graphs_use_original_validation(monkeypatch):
    freeze = core.freeze_json
    roots = []

    def record(value):
        roots.append(value)
        return freeze(value)

    deep = 0
    for _ in range(70):
        deep = [deep]
    monkeypatch.setattr(core, "freeze_json", record)
    assert core.semantic_digest(deep) == legacy_digest(deep)
    assert roots[0] is deep
    cycle = []
    cycle.append(cycle)
    with pytest.raises(RecursionError):
        legacy_digest(cycle)
    roots.clear()
    with pytest.raises(RecursionError):
        core.semantic_digest(cycle)
    assert roots[0] is cycle
