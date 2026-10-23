"""A True result must imply the ordinary stock ASCII encoder fits."""
import copy
import json
import math
import random
import sys

import pytest

from resectionlab.json_budget import builtin_json_fits, MAX_DEPTH, MAX_NODES, MAX_INT_BITS


def stock(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii')


@pytest.mark.parametrize('value', [None, False, True, 0, -1, 10**100, 0., -0.,
    sys.float_info.max, -sys.float_info.max, sys.float_info.min, 5e-324,
    '', 'ascii', '\n\t\r\b\f"\\', 'é', '漢字', '😀', '\ud800', '\udfff', '\ud800\udfff',
    [], (), {}, [None, True, 1, '😀'], {'a': (1, 2), '😀': ['\ud800']},
    {None: 'null key', False: 'bool key', 7: 'int key', 1.5: 'float key', 's': 'string key'}])
def test_true_is_a_stock_serializability_and_byte_upper_bound_proof(value):
    before = copy.deepcopy(value)
    actual = stock(value)
    for limit in (0, max(0, len(actual)-1), len(actual), len(actual)+1, len(actual)*12+128):
        admitted = builtin_json_fits(value, limit)
        assert type(admitted) is bool
        if admitted:
            assert len(actual) <= limit
        if limit < len(actual):
            assert not admitted
    assert builtin_json_fits(value, len(actual)*12+128)
    assert value == before


def test_exact_punctuation_and_scalar_thresholds_and_deliberate_string_float_slack():
    for value, size in ((None, 4), (True, 4), (False, 5), (-12345, 6),
                        ([], 2), ({}, 2), ((), 2), ([1, 2], 5), ({7: 8}, 7)):
        assert len(stock(value)) == size
        assert not builtin_json_fits(value, size-1)
        assert builtin_json_fits(value, size)
    assert builtin_json_fits('😀', 14) and len(stock('😀')) == 14
    assert not builtin_json_fits('a', 3)  # False is unknown, not too large.
    assert len(stock('a')) == 3 and builtin_json_fits('a', 14)
    assert len(stock(-0.)) == 4 and not builtin_json_fits(-0., 4)
    assert builtin_json_fits(-0., 32)


def test_key_coercions_use_quoted_stock_json_tokens():
    for key in (None, False, True, -17, 10**50, -0., 5e-324, sys.float_info.max, '😀', '\ud800'):
        value = {key: None}
        actual = stock(value)
        assert not builtin_json_fits(value, len(actual)-1)
        assert builtin_json_fits(value, len(actual)*12+128)
    for key in ((1, 2), object()):
        value = {key: 1}
        assert not builtin_json_fits(value, 10**6)
        with pytest.raises(TypeError):
            stock(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_and_keys_leave_stock_error_handling_intact(value):
    for document in (value, [value], {value: 1}, {'value': value}):
        assert not builtin_json_fits(document, 10**6)
        with pytest.raises(ValueError):
            stock(document)


def test_aliases_are_charged_each_time_but_only_active_path_cycles_decline():
    shared = [1, 2, 3]
    value = [shared, shared, {'again': shared}]
    assert builtin_json_fits(value, 10**6)
    actual = len(stock(value))
    assert not builtin_json_fits(value, actual-1)
    empty = []
    aliases = [empty] * 1000
    assert builtin_json_fits(aliases, len(stock(aliases)))
    assert not builtin_json_fits(aliases, len(stock(aliases))-1)
    cycle = []; cycle.append(cycle)
    indirect = {}; indirect['cycle'] = [indirect]
    for document in (cycle, indirect):
        assert not builtin_json_fits(document, 10**6)
        with pytest.raises(ValueError, match='Circular reference'):
            stock(document)


def test_depth_and_node_limits_are_bounded_and_do_not_mutate_the_graph():
    value = 0
    for _ in range(MAX_DEPTH):
        value = [value]
    assert builtin_json_fits(value, 10**6)
    deeper = [value]
    assert not builtin_json_fits(deeper, 10**6)
    assert len(stock(deeper)) < 10**6  # Decline remains a legitimate fallback.
    values = [None] * (MAX_NODES-1)
    assert builtin_json_fits(values, 8*MAX_NODES)
    values.append(None)
    assert not builtin_json_fits(values, 8*MAX_NODES)
    assert len(values) == MAX_NODES and all(item is None for item in values)


def test_integer_digit_limit_errors_decline_without_changing_global_limit():
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        value = 10**1000
        for document in (value, {value: 1}, [value]):
            assert not builtin_json_fits(document, 10**6)
            with pytest.raises(ValueError):
                stock(document)
        assert sys.get_int_max_str_digits() == 640
    finally:
        sys.set_int_max_str_digits(previous)


def test_large_integers_decline_before_decimal_conversion_even_with_unlimited_digits():
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        edge = 1 << (MAX_INT_BITS-1)
        assert builtin_json_fits(edge, len(stock(edge)))
        huge = 1 << (MAX_INT_BITS+10000)
        for value in (huge, -huge, {huge: None}, [-huge]):
            assert not builtin_json_fits(value, 10**9)
        assert sys.get_int_max_str_digits() == 0
    finally:
        sys.set_int_max_str_digits(previous)


def test_custom_subclasses_and_metaclasses_are_declined_without_callbacks():
    calls = []
    def touched(*args, **kwargs):
        calls.append('callback')
        raise AssertionError('Proof must never invoke custom hooks')
    class CustomStr(str):
        __str__ = __len__ = __iter__ = touched
    class CustomInt(int):
        __str__ = __repr__ = __int__ = touched
    class CustomFloat(float):
        __str__ = __repr__ = __float__ = touched
    class CustomList(list):
        __iter__ = __len__ = __getitem__ = touched
    class CustomTuple(tuple):
        __iter__ = __len__ = __getitem__ = touched
    class CustomDict(dict):
        items = __iter__ = __len__ = __getitem__ = touched
    class Meta(type):
        __eq__ = __hash__ = touched
    class Foreign(metaclass=Meta):
        __str__ = __repr__ = __iter__ = __len__ = touched
    for document in (CustomStr('x'), CustomInt(1), CustomFloat(1.), CustomList([1]),
                     CustomTuple((1,)), CustomDict(a=1), Foreign(), {'k': CustomList()}):
        assert not builtin_json_fits(document, 10**6)
    for key in (CustomStr('x'), CustomInt(2), CustomFloat(3.)):
        assert not builtin_json_fits({key: 1}, 10**6)
    for limit in (None, True, -1, 1., '100', CustomInt(10), Foreign()):
        assert not builtin_json_fits({}, limit)
    assert calls == []


def test_generated_builtin_graphs_agree_with_the_stock_oracle_for_every_true_result():
    rng = random.Random(192013)
    scalars = [None, True, False, -0., 0., 1.5, 5e-324, sys.float_info.max,
               -sys.float_info.max, 0, -1, 12345678901234567890, '', 'path', '😀', '\ud800', '"\\\n']
    keys = [None, True, -12, 1.25, 'name', '漢字', '\udfff']
    def graph(depth):
        if depth == 0 or rng.randrange(3) == 0:
            return rng.choice(scalars)
        kind = rng.randrange(3)
        if kind == 2:
            return {rng.choice(keys): graph(depth-1) for _ in range(rng.randrange(7))}
        children = [graph(depth-1) for _ in range(rng.randrange(7))]
        return children if kind == 0 else tuple(children)
    for _ in range(250):
        value = graph(5); before = copy.deepcopy(value); actual = len(stock(value))
        for limit in (max(0, actual-1), actual, actual+1, actual*12+128):
            fits = builtin_json_fits(value, limit)
            if fits:
                assert actual <= limit
        assert builtin_json_fits(value, actual*12+128)
        assert value == before
