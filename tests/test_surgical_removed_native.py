"""Caller dispatch admission and unchanged rollback/occupancy ownership.

The optional C entry's contracts are covered with an explicitly selected build
in test_surgical_idle_native. These tests use a spy at that API boundary only.
"""
import ast
import builtins
from pathlib import Path

import pytest

from resectionlab import surgical_engine as engine
from resectionlab import surgical_idle_native as adapter


def isolated_admission(globals_kind=dict, builtins_kind=dict, builtin_capture=None, alias_capture=None):
    # Re-execute the actual helper/captures with copied caller builtins. This
    # covers exec/module loaders whose function lookup differs from builtins.
    namespace = globals_kind(engine.__dict__)
    namespace['__builtins__'] = builtins_kind(builtins.__dict__)
    if alias_capture:
        namespace.update(alias_capture)
    nodes = []
    for node in ast.parse(Path(engine.__file__).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_try_removed_native', '_removed_known_c_function'):
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = [part.id for target in node.targets for part in ast.walk(target)
                     if isinstance(part, ast.Name)]
            if names and all(name.startswith('_removed_') for name in names):
                nodes.append(node)
    code = compile(ast.Module(body=nodes, type_ignores=[]), 'isolated-removed-admission', 'exec')
    overrides = {} if builtin_capture is None else builtin_capture
    original = {name: builtins.__dict__[name] for name in overrides}
    run = exec
    # Compilation/pytest machinery stays outside the brief actual-builtin
    # replacement. Captures and provenance evaluation invoke no unknown input.
    try:
        for name, value in overrides.items():
            builtins.__dict__[name] = value
            namespace['__builtins__'][name] = value
        run(code, namespace)
    finally:
        for name, value in original.items():
            builtins.__dict__[name] = value
    # Frozen pre-native predicate, with its original caller dispatch ordering.
    exec('def legacy(value):\n'
         '    return all(map(_strict_int_is, map(type, value), _strict_int_repeat(int)))\n'
         'def candidate(value):\n'
         '    return _try_removed_native(value) is True or '
         'all(map(_strict_int_is, map(type, value), _strict_int_repeat(int)))\n', namespace)
    return namespace


def test_small_and_nonexact_ledgers_skip_native_api_entirely(monkeypatch):
    def forbidden(value):
        raise AssertionError('Small/custom ledger reached optional identity/native API')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    class Removed(list):
        def __len__(self): raise AssertionError('No custom length lookup')
        def __iter__(self): raise AssertionError('No custom iteration')
    for value in ([], [0] * 1024, [0] * 16384, [0] * 32767,
                  Removed([0] * 32768), (0,) * 32768, None):
        assert engine._try_removed_native(value) is None
    saved = engine._microstep_checkpoint({'removed': [0] * 32767})
    assert saved['removed'] == [0] * 32767


def test_large_exact_list_calls_api_once_with_original_object(monkeypatch):
    value = [0] * 32768
    calls = []
    def predicate(actual):
        calls.append(actual)
        return True
    monkeypatch.setattr(adapter, 'try_exact_int_list', predicate)
    assert engine._removed_captures_genuine is True  # Includes real heap-type itertools.repeat.
    assert engine._try_removed_native(value) is True
    assert len(calls) == 1 and calls[0] is value


@pytest.mark.parametrize('name', ['all', 'map', 'type', 'int', 'list', '__import__',
                                '_strict_int_is', '_strict_int_repeat'])
@pytest.mark.parametrize('mode', ['false', 'error'])
def test_precapture_replacements_retain_frozen_result_and_exception_dispatch(monkeypatch, name, mode):
    calls = []
    def changed(*args):
        calls.append(name)
        if mode == 'error':
            raise RuntimeError('pre-capture ' + name)
        if name in ('map', '_strict_int_repeat'):
            return iter((False,))
        return False
    def forbidden(value): raise AssertionError('Unknown pre-capture dispatch admitted native')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    if name.startswith('_strict_'):
        namespace = isolated_admission(alias_capture={name: changed})
    else:
        namespace = isolated_admission(builtin_capture={name: changed})
    assert namespace['_removed_captures_genuine'] is False
    assert calls == []  # Never call replacements merely to validate captures.
    def outcome(function):
        calls.clear()
        try:
            result = function([0] * 32768)
        except Exception as error:
            result = (type(error), str(error))
        return result, tuple(calls)
    assert outcome(namespace['candidate']) == outcome(namespace['legacy'])


def test_precapture_operator_false_result_is_not_bypassed_by_native_true(monkeypatch):
    import operator
    monkeypatch.setattr(operator, 'is_', lambda *args: False)
    namespace = isolated_admission(alias_capture={'_strict_int_is': operator.is_})
    calls = []
    def native_true(value):
        calls.append(value)
        return True
    monkeypatch.setattr(adapter, 'try_exact_int_list', native_true)
    assert namespace['legacy']([0] * 32768) is False
    assert namespace['candidate']([0] * 32768) is False
    assert calls == []


def test_precapture_len_replacement_is_not_called_by_exact_list_length_admission(monkeypatch):
    def forbidden(*args): raise AssertionError('Admission invoked mutable builtin len')
    namespace = isolated_admission(builtin_capture={'len': forbidden})
    monkeypatch.setattr(adapter, 'try_exact_int_list', lambda value: True)
    assert namespace['_removed_captures_genuine'] is True
    assert namespace['_try_removed_native']([0] * 32768) is True


@pytest.mark.parametrize('name', ['all', 'map', '_strict_int_is', '_strict_int_repeat'])
def test_precapture_other_real_c_functions_and_types_do_not_pass_name_provenance(monkeypatch, name):
    import itertools
    import operator
    alternatives = {'all': any, 'map': filter, '_strict_int_is': operator.eq,
                    '_strict_int_repeat': itertools.count}
    if name.startswith('_strict_'):
        namespace = isolated_admission(alias_capture={name: alternatives[name]})
    else:
        namespace = isolated_admission(builtin_capture={name: alternatives[name]})
    def forbidden(value): raise AssertionError('Different C builtin admitted native')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    assert namespace['_removed_captures_genuine'] is False
    assert namespace['_try_removed_native']([0] * 32768) is None


@pytest.mark.parametrize('name', ['type', 'int', 'all', 'map', 'list', '__import__', '_strict_int_is', '_strict_int_repeat'])
def test_replaced_caller_globals_decline_without_invoking_replacement(monkeypatch, name):
    namespace = isolated_admission()
    def forbidden(*args): raise AssertionError('Admission invoked altered caller dispatch')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    namespace[name] = forbidden
    assert namespace['_try_removed_native']([0] * 32768) is None


@pytest.mark.parametrize('name', ['type', 'int', 'all', 'map', 'list', '__import__'])
def test_actual_copied_caller_builtins_mutation_declines_native(monkeypatch, name):
    namespace = isolated_admission()
    mapping = namespace['__builtins__']
    assert namespace['_try_removed_native'].__builtins__ is mapping
    assert namespace['_removed_caller_builtins'] is mapping
    assert mapping is not builtins.__dict__
    def forbidden(*args): raise AssertionError('Changed builtin or optional API was invoked')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    mapping[name] = forbidden
    assert namespace['_try_removed_native']([0] * 32768) is None


@pytest.mark.parametrize('where', ['globals', 'builtins'])
@pytest.mark.parametrize('name', ['type', 'int', 'all', 'map', 'list', '__import__', '_strict_int_is', '_strict_int_repeat'])
def test_fallback_keeps_frozen_predicate_dispatch_and_errors(monkeypatch, where, name):
    if where == 'builtins' and name.startswith('_strict_'):
        pytest.skip('Strict aliases are module globals, not builtin lookups')
    namespace = isolated_admission()
    calls = []
    def changed(*args):
        calls.append(name)
        raise RuntimeError('original dispatch ' + name)
    def forbidden(value): raise AssertionError('Altered caller admitted native')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    mapping = namespace if where == 'globals' else namespace['__builtins__']
    # int is a value passed to repeat, so replace its identity without making a
    # new callable assertion; the frozen predicate then returns False.
    mapping[name] = object() if name == 'int' else changed
    def outcome(function):
        calls.clear()
        try:
            result = function([0] * 32768)
        except Exception as error:
            result = (type(error), str(error))
        return result, tuple(calls)
    assert outcome(namespace['candidate']) == outcome(namespace['legacy'])


def test_real_builtin_mapping_replacement_is_checked_without_mutable_type_dispatch(monkeypatch):
    def forbidden(*args): raise AssertionError('Mutable builtin dispatch invoked')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    value = [0] * 32768
    mapping = engine._try_removed_native.__builtins__
    # Do not run pytest/context-manager machinery while shared builtins differ.
    for name in ('type', 'int', 'all', 'map', 'list', '__import__'):
        original = mapping[name]
        try:
            mapping[name] = forbidden
            answer = engine._try_removed_native(value)
        finally:
            mapping[name] = original
        assert answer is None


@pytest.mark.parametrize('where', ['globals', 'builtins'])
def test_custom_caller_mapping_declines_before_get_callback(monkeypatch, where):
    class Mapping(dict):
        def get(self, *args): raise AssertionError('Custom mapping lookup invoked')
    namespace = isolated_admission(**{where + '_kind': Mapping})
    def forbidden(value): raise AssertionError('Custom caller mapping admitted native')
    monkeypatch.setattr(adapter, 'try_exact_int_list', forbidden)
    assert namespace['_try_removed_native']([0] * 32768) is None


def test_unconfigured_full_and_light_seals_accept_helper_without_dictionary_defaults():
    from resectionlab.surgical_verified_cache import implementation_fingerprint
    from resectionlab.surgical_observed_cache import light_implementation_seal
    assert engine._try_removed_native.__defaults__ is None
    assert engine._try_removed_native.__kwdefaults__ is None
    first = implementation_fingerprint()
    assert first is not None and implementation_fingerprint() == first
    first = light_implementation_seal()
    assert light_implementation_seal() == first
