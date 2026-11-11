"""Canonical native hashing stays byte-for-byte compatible with stdlib JSON."""
import hashlib
import json
import math
import random
import struct
import sys
from types import MappingProxyType

import numpy as np
import pytest


@pytest.fixture
def native():
    return pytest.importorskip("resectionlab._geometry_cpp")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def assert_canonical(native, value, **kwargs):
    expected = canonical_bytes(value)
    observed = []

    def sha256(payload):
        assert type(payload) is bytes
        observed.append(payload)
        return hashlib.sha256(payload)

    marker = object()
    result = native.canonical_json_digest(value, sha256, marker, **kwargs)
    assert result == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert observed == [expected]


@pytest.mark.parametrize("value", [
    None, True, False, 0, -17, -(2**63), 2**63 - 1, -(2**63) - 1, 2**63,
    10**200, -(10**200), -0.0, math.nextafter(0.0, 1.0),
    1.7976931348623157e308, [], {},
    "".join(chr(index) for index in range(256)) + "é😀\ud800\udfff\u2028",
    {"😀": "astral", "\ue000": "BMP", "\ud800": "surrogate", "a": [], "": None},
    {"z": [None, True, 1, -0.0], "a": {"unicode": "é😀"}},
])
def test_canonical_bytes_and_digest_match_stdlib(native, value):
    assert_canonical(native, value)


def test_generated_float_bits_and_nested_json_match_stdlib(native):
    rng = random.Random(4713)
    values = [0.0, -0.0, 1.0, -1.0, 1e-7, 1e-6, 1e15, 1e16]
    for _ in range(2048):
        value = struct.unpack("!d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        if math.isfinite(value):
            values.append(value)
    assert_canonical(native, {"numbers": values, "ints": [rng.getrandbits(100) for _ in range(64)]})


def test_owned_tuple_and_aliasing_are_encoded_without_changing_source(native):
    shared = {"z": [-0.0], "a": "é😀"}
    value = {"last": shared, "first": (shared, 10**100)}
    marker = object()
    assert native.canonical_json_digest(value, hashlib.sha256, marker) is marker
    assert_canonical(native, value, allow_tuple=True, max_depth=64)
    assert value["first"][0] is value["last"] is shared


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), float("-inf"), object(), 1j, {1: 2},
    (1, 2), MappingProxyType({"frozen": (1, 2)}), np.int64(2), np.float64(2),
    np.array([1.0]), {"first": [1, 2], "last": object()},
])
def test_unsupported_graph_returns_marker_without_hashing(native, value):
    marker = object()

    def unexpected_hash(*_args):
        raise AssertionError("unsupported graph was published")

    assert native.canonical_json_digest(value, unexpected_hash, marker) is marker


def test_subclass_hooks_are_not_invoked(native):
    class UnusualList(list):
        def __iter__(self):
            raise AssertionError("subclass iteration")

    class UnusualDict(dict):
        def items(self):
            raise AssertionError("subclass items")

    class UnusualStr(str):
        def __lt__(self, other):
            raise AssertionError("subclass comparison")

    class UnusualInt(int):
        def __str__(self):
            raise AssertionError("subclass formatting")

    class UnusualFloat(float):
        def __repr__(self):
            raise AssertionError("subclass formatting")

    marker = object()
    for value in (UnusualList([1]), UnusualDict(a=1), UnusualStr("leaf"),
                  UnusualInt(1), UnusualFloat(1), {UnusualStr("key"): 1}):
        assert native.canonical_json_digest(value, hashlib.sha256, marker) is marker


def test_cycles_depth_and_work_caps_keep_reference_fallback(native):
    marker = object()
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict
    deep = []
    for _ in range(32):
        deep = [deep]
    for value in (cyclic_list, cyclic_dict, deep, [None] * 65537, "x" * (4 * 1024 * 1024)):
        assert native.canonical_json_digest(value, hashlib.sha256, marker) is marker
    assert cyclic_list[0] is cyclic_list and cyclic_dict["self"] is cyclic_dict
    assert_canonical(native, deep, max_depth=64)
    for depth in (-1, 0, 257):
        assert native.canonical_json_digest({}, hashlib.sha256, marker, max_depth=depth) is marker


def test_hash_callback_observes_completed_snapshot(native):
    value = {"nested": [1, -0.0], "unicode": "é😀"}
    expected = canonical_bytes(value)

    def sha256(payload):
        value.clear()
        assert payload == expected
        return hashlib.sha256(payload)

    result = native.canonical_json_digest(value, sha256, object())
    assert result == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert value == {}


def test_hash_callback_failure_is_not_retried(native):
    class HashFailure(RuntimeError):
        pass

    def failed_hash(_payload):
        raise HashFailure("hash callback")

    with pytest.raises(HashFailure, match="hash callback"):
        native.canonical_json_digest({"a": [1]}, failed_hash, object())


def test_public_hashes_use_fused_encoder(native):
    from resectionlab import native_backend as backend
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_engine import _owned_digest
    if backend._module is None:
        pytest.skip('native dispatch required')
    backend.reset_counters()
    value = {'a': [1, -0., 'é😀'], 'b': True}
    expected = 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()
    assert semantic_digest(value) == _owned_digest(value) == expected
    assert backend._counters['canonicalDigestCalls'] == 2
    assert backend._counters['canonicalDigestFallbackCalls'] == 0
    assert backend._counters['jsonCopyCalls'] == 0


@pytest.mark.parametrize('target', ['dumps', 'encode', 'c_factory_call'])
def test_public_hashes_preserve_replaced_json_callbacks(native, monkeypatch, target):
    import _json
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_engine import _owned_digest
    owner, name = {'dumps': (json, 'dumps'), 'encode': (json.JSONEncoder, 'encode'),
                   'c_factory_call': (_json.make_encoder, '__call__')}[target]
    original = getattr(owner, name)
    calls = []
    def changed(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, name, changed)
    value = {'a': [1, -0., 'é😀']}
    expected = 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()
    for function in (semantic_digest, _owned_digest):
        calls.clear()
        assert function(value) == expected
        assert calls == [name]


def test_public_hashes_preserve_changed_json_defaults(native, monkeypatch):
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_engine import _owned_digest
    defaults = dict(json.dumps.__kwdefaults__, ensure_ascii=False)
    monkeypatch.setattr(json.dumps, '__kwdefaults__', defaults)
    value = {'text': 'é😀'}
    expected = 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()
    assert semantic_digest(value) == _owned_digest(value) == expected


def test_giant_integer_uses_reference_even_with_decimal_limit_disabled(native):
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        marker = object()
        assert native.canonical_json_digest(10**10000, hashlib.sha256, marker) is marker
    finally:
        sys.set_int_max_str_digits(previous_limit)


def function_guard(function):
    return (function, function.__code__, function.__defaults__, function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()))


def test_guard_detects_bindings_code_defaults_and_keyword_mutations(native):
    def function(value=17, *, enabled=True):
        return value, enabled

    sentinel = object()
    namespace = {"present": sentinel}
    bindings = ((namespace, "present", sentinel, True), (namespace, "absent", None, False))
    guard = (bindings, (function_guard(function),))
    assert_canonical(native, {"a": [1]}, guard=guard)
    marker = object()

    def fallback():
        assert native.canonical_json_digest({"a": [1]}, hashlib.sha256, marker, guard=guard) is marker

    namespace["present"] = object()
    fallback()
    namespace["present"] = sentinel
    namespace["absent"] = None
    fallback()
    del namespace["absent"]
    function.__kwdefaults__["enabled"] = False
    fallback()
    function.__kwdefaults__["enabled"] = True
    old_defaults = function.__defaults__
    function.__defaults__ = (18,)
    fallback()
    function.__defaults__ = old_defaults
    function.__code__ = (lambda: None).__code__
    fallback()


def test_guard_rejects_custom_mapping_keys_without_equality_hooks(native):
    class Key:
        def __hash__(self):
            return hash("missing")

        def __eq__(self, other):
            raise AssertionError("custom key equality")

    namespace = {Key(): None}
    guard = (((namespace, "missing", None, False),), ())
    marker = object()
    assert native.canonical_json_digest({}, hashlib.sha256, marker, guard=guard) is marker


def test_guard_checks_encoder_type_slots_without_custom_getters(native):
    class Encoder:
        encode = object()

    guard = (((Encoder, "encode", Encoder.encode, True),), ())
    assert_canonical(native, {}, guard=guard)
    marker = object()
    Encoder.__getattribute__ = lambda *_args: (_ for _ in ()).throw(AssertionError("getter"))
    assert native.canonical_json_digest({}, hashlib.sha256, marker, guard=guard) is marker

    class Meta(type):
        def __getattribute__(self, name):
            raise AssertionError("metaclass getter")

    class UnusualEncoder(metaclass=Meta):
        pass

    guard = (((UnusualEncoder, "encode", None, False),), ())
    assert native.canonical_json_digest({}, hashlib.sha256, marker, guard=guard) is marker


@pytest.mark.parametrize("guard", [False, (), ((),), ([], ()), (((),), ()), ((), ((None,) * 5,))])
def test_malformed_guard_falls_back(native, guard):
    marker = object()
    assert native.canonical_json_digest({}, hashlib.sha256, marker, guard=guard) is marker
