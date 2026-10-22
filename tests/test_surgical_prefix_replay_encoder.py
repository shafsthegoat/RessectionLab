"""Byte-level bounded serialization checks; no engine actions or learning."""
import json
import math
import random
import zlib

import pytest

from resectionlab.surgical_prefix_replay import PrefixBudgetExceeded, _encode
import resectionlab.surgical_prefix_replay as prefix


LIMIT = 8 * 1024**2


def reference(value):
    return ''.join(json.JSONEncoder(separators=(',', ':'), allow_nan=False).iterencode(value)).encode()


def encoded(value, **options):
    return _encode(value, compressed_limit=options.pop('compressed_limit', LIMIT),
                   raw_limit=options.pop('raw_limit', LIMIT), **options)


@pytest.mark.parametrize('value', [
    {}, [], (), None, True, False, 0, -0.0, 5.e-324, -1.7976931348623157e308,
    {'z': [0, -1, 2**63, 2**64, -2**65, True, None, -0.0], 'a': 'after z'},
    {None: 'nil', False: 'false', 17: 'integer', .125: 'float', 'z': 'last'},
    {'ascii': '\b\t\n\f\r"\\', 'unicode': '\u2028\U0001f9e0\ud800\udfff'},
    [list(range(513)), tuple(range(1025)), [0.0, -0.0, 1.e-300, 1.e300] * 300],
])
def test_exact_compact_json_bytes_and_decoded_values(value):
    raw = zlib.decompress(encoded(value))
    assert raw == reference(value)
    assert json.loads(raw) == json.loads(reference(value))
    if value == -0.0 and type(value) is float:
        assert math.copysign(1., json.loads(raw)) == -1.


def test_large_flat_numeric_batches_use_bounded_c_encoder(monkeypatch):
    value = [-(2**64 - 1), 2**64 - 1, -1.7976931348623157e308,
             5.e-324, -0.0, True, False, None] * 1000
    expected = reference(value)
    calls = []
    original = json.dumps
    def checked(batch, *args, **kwargs):
        assert type(batch) in (list, tuple) and 0 < len(batch) <= 512
        raw = original(batch, *args, **kwargs)
        assert len(raw.encode()) < 13 * 1024
        calls.append(len(batch))
        return raw
    monkeypatch.setattr(prefix.json, 'dumps', checked)
    assert zlib.decompress(encoded(value)) == expected
    assert sum(calls) == len(value) and len(calls) == math.ceil(len(value) / 512)


def test_many_small_topology_subtrees_batch_without_exceeding_c_fragment_bound(monkeypatch):
    value = {'flapComponents': [
        {'cells': [i, i + 1, i + 2], 'separable': None if i % 3 else True}
        for i in range(5000)],
        'unicode': [{'name': '\U0001f9e0' * 200, 'value': i} for i in range(50)]}
    expected = reference(value)
    original = json.dumps
    fragments = []
    def bounded(item, *args, **kwargs):
        if type(item) in (dict, list, tuple):
            assert len(item) <= 512
        text = original(item, *args, **kwargs)
        assert len(text.encode()) <= 16384
        fragments.append(len(text))
        return text
    monkeypatch.setattr(prefix.json, 'dumps', bounded)
    assert zlib.decompress(encoded(value)) == expected
    assert 1 < len(fragments) < 1000


def test_giant_strings_escape_in_bounded_pieces_and_refuse_raw_overflow(monkeypatch):
    value = {'\U0001f9e0' * 3000: '\U0001f600\ud800"\\\n' * 12000}
    expected = reference(value)
    calls = []
    original = json.encoder.encode_basestring_ascii
    def bounded(text):
        assert str.__len__(text) <= 1024
        result = original(text)
        assert len(result) <= 12 * 1024 + 2
        calls.append(str.__len__(text))
        return result
    monkeypatch.setattr(prefix.json.encoder, 'encode_basestring_ascii', bounded)
    assert zlib.decompress(encoded(value)) == expected
    assert max(calls) == 1024
    with pytest.raises(PrefixBudgetExceeded, match='Decoded JSON'):
        encoded(value, raw_limit=100)


class CustomString(str):
    def __getitem__(self, key):
        raise AssertionError('The standard encoder ignores custom string slicing')
    def __len__(self):
        raise AssertionError('The standard encoder ignores custom string length')
    def __str__(self):
        return 'overridden string'


class CustomInt(int):
    def __repr__(self):
        return 'overridden integer'


class CustomFloat(float):
    def __repr__(self):
        return 'overridden float'


class CustomList(list):
    def __iter__(self):
        return iter([CustomInt(3), CustomFloat(-0.0), CustomString('real string')])


class CustomDict(dict):
    def items(self):
        return reversed(list(super().items()))


def test_custom_container_and_scalar_fallback_preserves_standard_behavior():
    value = CustomDict({'a': CustomList([1, 2]), CustomString('b'): CustomString('\U0001f600' * 3000),
                        CustomInt(11): CustomFloat(1.25)})
    assert zlib.decompress(encoded(value)) == reference(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'),
    [1] * 600 + [float('nan')], {'x': object()}, {'x': {1, 2}}, {('invalid',): None}])
def test_nan_and_unsupported_values_keep_standard_exception_behavior(value):
    with pytest.raises((ValueError, TypeError)) as expected:
        reference(value)
    with pytest.raises(type(expected.value)) as actual:
        encoded(value)
    assert str(actual.value) == str(expected.value)


def test_circular_checks_distinguish_shared_values_from_active_ancestry():
    value = [1, 2, 3]
    assert zlib.decompress(encoded({'a': value, 'b': value})) == reference({'a': value, 'b': value})
    value.append(value)
    with pytest.raises(ValueError, match='Circular reference detected'):
        encoded(value)
    dictionary = {}
    dictionary['self'] = dictionary
    with pytest.raises(ValueError, match='Circular reference detected'):
        encoded(dictionary)


def test_raw_compressed_and_growth_callback_caps_are_checked_before_append():
    value = {'cells': list(range(10000)), 'text': '\U0001f600' * 4000}
    expected = reference(value)
    blob = encoded(value)
    assert encoded(value, raw_limit=len(expected), compressed_limit=len(blob)) == blob
    with pytest.raises(PrefixBudgetExceeded, match='Decoded JSON'):
        encoded(value, raw_limit=len(expected) - 1)
    observed = []
    def growth(size):
        assert size <= len(blob) - 1
        observed.append(size)
    with pytest.raises(PrefixBudgetExceeded, match='Compressed state'):
        encoded(value, compressed_limit=len(blob) - 1, on_growth=growth)
    assert observed == sorted(observed)
    error = RuntimeError('interrupted serialization')
    def interrupt(_):
        raise error
    with pytest.raises(RuntimeError) as actual:
        encoded(value, on_growth=interrupt)
    assert actual.value is error
    assert zlib.decompress(encoded(value)) == expected


def test_giant_integer_cannot_allocate_a_conversion_before_raw_budget_failure(monkeypatch):
    integer = 1 << 1000000
    def unexpected(*args, **kwargs):
        raise AssertionError('An impossible giant scalar reached JSON conversion')
    monkeypatch.setattr(prefix.json.JSONEncoder, 'iterencode', unexpected)
    monkeypatch.setattr(prefix.json, 'dumps', unexpected)
    for value in ([integer], {integer: 0}):
        with pytest.raises(PrefixBudgetExceeded, match='Decoded JSON'):
            encoded(value, raw_limit=64)


def test_nested_generated_json_preserves_all_delimiters_and_insertion_order():
    rng = random.Random(3741)
    leaves = [None, True, False, -0.0, 1.e-300, 0, -17, 2**67, '\U0001f9e0\n"\\']
    def make(depth):
        if not depth or rng.random() < .3:
            return rng.choice(leaves)
        if rng.random() < .5:
            return [make(depth - 1) for _ in range(rng.randrange(8))]
        return {str(i): make(depth - 1) for i in rng.sample(range(10), rng.randrange(8))}
    for _ in range(40):
        value = make(4)
        assert zlib.decompress(encoded(value)) == reference(value)
