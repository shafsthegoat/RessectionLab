"""Prefix budget acceptance preserves the original encoder fallback contract."""
import json
import math
import subprocess
import sys
import textwrap

import pytest

from resectionlab import desktop_bridge as bridge


def check(value, limit):
    return bridge._require_json_budget(value, limit, 'TEST_LIMIT', 'payload too big', _builtin_fast=True)


def legacy(value, limit):
    return bridge._require_json_budget(value, limit, 'TEST_LIMIT', 'payload too big')


@pytest.mark.parametrize('value', [None, True, -123, -0., 5e-324, 'é😀\ud800',
    {'fields': [1, None, '😀'], 7: {'nested': 2.5}, False: ()}, [[1, 2]]*3])
def test_prefix_fast_check_and_stock_boundary_decision_match(value):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    for limit in (0, len(encoded)-1, len(encoded), len(encoded)+1, 32*1024**2):
        for operation in (check, legacy):
            if len(encoded) > limit:
                with pytest.raises(bridge.BridgeError, match='payload too big') as error:
                    operation(value, limit)
                assert error.value.code == 'TEST_LIMIT'
            else:
                assert operation(value, limit) is None


def test_default_bridge_path_stays_streaming_and_prefix_shortcut_is_qualified(monkeypatch):
    from resectionlab import json_budget
    assert bridge._stock_budget_encoder()
    calls=[]
    original=json_budget.builtin_json_fits
    def proof(value, limit):
        calls.append(value)
        return original(value, limit)
    monkeypatch.setattr(json_budget, 'builtin_json_fits', proof)
    assert legacy({'example': 1}, 10**6) is None and calls == []
    assert check({'example': 1}, 10**6) is None and calls == [{'example': 1}]


@pytest.mark.parametrize('kind', ['class', 'factory', 'iterencode', 'init', 'default', 'ascii', 'make', 'indent', 'ensure_ascii', 'circular', 'infinity', 'builtin_float'])
def test_custom_encoder_paths_keep_the_original_callbacks_or_errors(monkeypatch, kind):
    calls=[]
    original=json.JSONEncoder
    def fail(*args, **kwargs):
        calls.append(kind)
        raise RuntimeError('custom encoder reached')
    if kind=='class':
        class Encoder(original):
            def iterencode(self,*args,**kwargs):return fail()
        monkeypatch.setattr(json,'JSONEncoder',Encoder)
    elif kind=='factory':monkeypatch.setattr(json,'JSONEncoder',fail)
    elif kind in ('iterencode','init','default'):
        monkeypatch.setattr(original, {'init':'__init__'}.get(kind,kind), fail)
    elif kind=='ascii':monkeypatch.setattr(json.encoder,'encode_basestring_ascii',fail)
    elif kind=='make':monkeypatch.setattr(json.encoder,'_make_iterencode',fail)
    elif kind in ('indent','ensure_ascii','circular'):
        key={'circular':'check_circular'}.get(kind,kind)
        monkeypatch.setitem(original.__init__.__kwdefaults__,key, {'indent':' '*100,'ensure_ascii':False,'circular':False}[kind])
    elif kind=='infinity':monkeypatch.setattr(json.encoder,'INFINITY',123.)
    else:monkeypatch.setattr(json.encoder,'float',lambda *args:fail(),raising=False)
    assert bridge._stock_budget_encoder() is False
    value={'content':'é😀'} if kind!='default' else object()
    def outcome(operation):
        try:return ('ok',operation(value,50))
        except Exception as error:return (type(error),str(error),getattr(error,'code',None))
    assert outcome(check)==outcome(legacy)
    if kind in ('class','factory','iterencode','init','default','ascii','make'):assert calls


def test_nonfinite_and_cycles_preserve_stock_errors_even_with_foreign_math_hook(monkeypatch):
    monkeypatch.setattr(math,'isfinite',lambda value: True)
    cycle=[];cycle.append(cycle)
    for value in [math.nan, math.inf, {math.inf: 1}, cycle, {(1,2): 3}]:
        def result(operation):
            try:operation(value,10**6)
            except Exception as error:return type(error),str(error)
            raise AssertionError('Invalid value admitted')
        assert result(check)==result(legacy)


def test_low_recursion_headroom_keeps_stream_fallback(monkeypatch):
    monkeypatch.setattr(sys,'getrecursionlimit',lambda: 128)
    assert not bridge._stock_budget_encoder()
    assert check({'valid': 1},10000) is None


@pytest.mark.parametrize('setup', [
    "json.JSONEncoder.__init__.__kwdefaults__['indent']=' '*100",
    "json.JSONEncoder=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('cold custom encoder'))",
    "json.encoder.encode_basestring_ascii=lambda value: (_ for _ in ()).throw(RuntimeError('cold custom scalar'))",
])
def test_preimport_custom_encoders_are_not_mistaken_for_stock(setup):
    script='import json, importlib\nfrom resectionlab import desktop_bridge as b\n'+setup+'\n'+textwrap.dedent('''
        importlib.reload(b)
        assert not b._stock_budget_encoder()
        def outcome(fast):
            try:b._require_json_budget({'a':'b'},20,'LIMIT','too big',_builtin_fast=fast)
            except Exception as error:return type(error).__name__,str(error)
            return 'ok'
        assert outcome(True)==outcome(False)
    ''')
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr


def test_replaced_builtin_bool_declines_without_undercharging_integer():
    import builtins
    original=builtins.bool
    refused=False
    try:
        builtins.bool=int
        qualified=bridge._stock_budget_encoder()
        try:check(1234567,4)
        except bridge.BridgeError as error:refused=error.code=='TEST_LIMIT'
    finally:
        builtins.bool=original
    assert not qualified and refused
