"""Reuse of exact-int qualification must preserve mutable public ledger semantics."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from resectionlab import surgical_engine as module
from resectionlab.surgical_engine import SurgicalEngine
from test_surgical_occupancy_key import source, LegacyKeyEngine, assert_ownership, legacy_key
from test_surgical_removed_native import isolated_admission


def model(values=(1560, 2160, 2760)):
    result = SurgicalEngine(source())
    result.state['removed'] = list(values)
    result.occupancy()
    return result


def test_identity_certificate_survives_equal_lists_but_recertifies_distinct_equal_int_objects():
    engine = model(); proof = engine._removed_ids_proof; owner = engine.occupancy()
    assert engine.occupancy() is owner and engine._removed_ids_proof is proof
    engine.state['removed'] = list(proof)
    assert engine.occupancy() is owner and engine._removed_ids_proof is proof
    engine.state['removed'] = [int(str(v)) for v in proof]
    assert all(a is not b for a, b in zip(engine.state['removed'], proof))
    assert engine.occupancy() is owner
    refreshed = engine._removed_ids_proof
    assert refreshed is not proof and all(a is b for a, b in zip(refreshed, engine.state['removed']))


def test_aliased_append_pop_reverse_slice_and_same_length_replacement_match_json_reference():
    actual, reference = model(), LegacyKeyEngine(source())
    reference.state = copy.deepcopy(actual.state)
    aliases = [actual.state['removed'], reference.state['removed']]
    for mutate in (lambda x: x.append(3300), lambda x: x.pop(), lambda x: x.reverse(),
                   lambda x: x.__setitem__(1, 3300), lambda x: x.__setitem__(slice(None), [4200, 3600, 3000])):
        for values in aliases: mutate(values)
        assert_ownership(actual, reference)
        assert actual._removed_ids_proof == tuple(actual.state['removed'])


@pytest.mark.parametrize('value', [True, 1., np.int64(1)])
def test_equal_valued_nonint_edit_cannot_inherit_integer_qualification(value):
    engine = model([0, 1, 2]); engine.state['removed'][1] = value
    engine.occupancy()
    assert engine._occupancy_cache[0] == legacy_key(engine.state)
    assert engine._removed_ids_proof is None
    engine.state['removed'][1] = 1; engine.occupancy()
    assert engine._removed_ids_proof == (0, 1, 2)


def test_integer_subclass_hooks_are_not_used_by_identity_scan(monkeypatch):
    events = []
    class Integer(int):
        def __eq__(self, other): events.append('equality'); raise AssertionError('equality')
        def __hash__(self): events.append('hash'); raise AssertionError('hash')
    engine = model([0, 1, 2]); engine.state['removed'][1] = Integer(1)
    original = module._owned_digest
    def boundary(value):
        if 'removed' in value: raise RuntimeError('legacy boundary')
        return original(value)
    monkeypatch.setattr(module, '_owned_digest', boundary)
    with pytest.raises(RuntimeError, match='legacy boundary'): engine.occupancy()
    assert events == [] and engine._removed_ids_proof is None


def test_changes_digest_can_mutate_removed_without_moving_legacy_key_construction(monkeypatch):
    engine = model([0, 1, 2]); proof = engine._removed_ids_proof
    engine._occupancy_cache = None
    original = module._owned_digest
    def digest(value):
        if 'removed' not in value: engine.state['removed'][1] = True
        return original(value)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_owned_digest', digest)
        engine.occupancy()
    # Original predicate saw ints, then the changes callback edited the list.
    assert engine._occupancy_cache[0][0] == (0, True, 2)
    assert type(engine._occupancy_cache[0][0][1]) is bool
    assert engine._removed_ids_proof is proof
    engine.occupancy()
    assert type(engine._occupancy_cache[0]) is str and engine._removed_ids_proof is None


def test_self_restoring_custom_predicate_cannot_certify_a_future_integer_proof(monkeypatch):
    engine = model([0, 1, 2]); engine.state['removed'][1] = True
    original_all = all
    def custom_all(values):
        monkeypatch.delattr(module, 'all')
        return True
    monkeypatch.setattr(module, 'all', custom_all, raising=False)
    engine.occupancy()
    assert engine._removed_ids_proof is None
    engine.occupancy()
    assert type(engine._occupancy_cache[0]) is str
    assert original_all(type(v) is int for v in engine.state['removed']) is False


def isolated_occupancy(**kwargs):
    namespace = isolated_admission(**kwargs)
    tree = ast.parse(Path(module.__file__).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_removed_snapshot_dispatch', '_reuse_removed_snapshot')]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SurgicalEngine')
    nodes.append(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'occupancy'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'isolated-occupancy-proof', 'exec'), namespace)
    # Frozen old admission and key construction, with the same caller mapping.
    exec('''def frozen_key(self):
    removed = self.state['removed']
    changes = {'holds': self.state['holds']}
    if isinstance(removed, list) and ((_try_removed_native(removed) is True or
            all(map(_strict_int_is, map(type, removed), _strict_int_repeat(int))))
            if type(removed) is list else all(type(v) is int for v in removed)):
        change_key = _owned_digest(changes)
        return (tuple(removed), change_key)
    return _owned_digest({'removed': removed, **changes})
''', namespace)
    return namespace


@pytest.mark.parametrize('where', ['globals', 'builtins'])
@pytest.mark.parametrize('name', ['type', 'int', 'all', 'map', 'list', 'tuple', '__import__',
                                '_strict_int_is', '_strict_int_repeat'])
def test_changed_caller_dispatch_keeps_frozen_key_and_exception_order(where, name):
    if where == 'builtins' and name.startswith('_strict'): pytest.skip('Module alias only')
    namespace = isolated_occupancy(); engine = model([0, 1, 2]); events = []
    def changed(*args): events.append(name); raise RuntimeError('changed '+name)
    mapping = namespace if where == 'globals' else namespace['__builtins__']
    mapping[name] = object() if name == 'int' else changed
    def outcome(function):
        events.clear()
        try: result = function(engine)
        except Exception as error: return ('error', type(error), str(error)), tuple(events)
        return ('key', engine._occupancy_cache[0] if function is namespace['occupancy'] else result), tuple(events)
    expected = outcome(namespace['frozen_key'])
    observed = outcome(namespace['occupancy'])
    assert observed == expected
    assert engine._removed_ids_proof is None


@pytest.mark.parametrize('name', ['type', 'int', 'all', 'map', 'list', 'tuple', '__import__'])
def test_precapture_replacements_do_not_admit_existing_proof(name):
    def changed(*args): return False
    namespace = isolated_occupancy(builtin_capture={name: changed})
    engine = model()
    assert namespace['_removed_snapshot_dispatch'](engine.state['removed']) is False


def test_native_selection_still_runs_on_every_lookup_and_does_not_publish_custom_proof(monkeypatch):
    engine = model([0, 1, 2]); calls = []
    def selected(value): calls.append(value); return True
    monkeypatch.setattr(module, '_try_removed_native', selected)
    owner = engine.occupancy()
    assert engine.occupancy() is owner
    assert calls == [engine.state['removed'], engine.state['removed']]
    assert engine._removed_ids_proof is None


def test_clone_restore_and_rollback_keep_independent_ledgers_with_bounded_proof():
    engine = model(); proof = engine._removed_ids_proof
    clone = engine.clone()
    assert clone._removed_ids_proof is proof and clone.state['removed'] is not engine.state['removed']
    clone.state['removed'][0] = 100
    assert clone.occupancy() is not engine.occupancy()
    assert engine._removed_ids_proof is proof and clone._removed_ids_proof is not proof
    intact = SurgicalEngine(engine.scene); intact.occupancy()
    restored = SurgicalEngine.restore(intact.scene, intact.snapshot())
    assert restored._removed_ids_proof is None
    restored.occupancy(); assert restored._removed_ids_proof == ()
    state = module._microstep_checkpoint(engine.state)
    state['removed'][0] = 100; engine.state = state; engine.occupancy()
    assert engine._removed_ids_proof[0] == 100 and proof[0] == 1560
    engine.state['removed'] = [0]*32768; engine.occupancy()
    assert len(engine._removed_ids_proof) == 32768
    engine.state['removed'].append(0); engine.occupancy()
    assert engine._removed_ids_proof is None


@pytest.mark.parametrize('where', ['globals', 'builtins'])
def test_custom_caller_mapping_declines_before_any_get_hook(where):
    class Mapping(dict):
        def get(self, *args): raise AssertionError('custom get')
    namespace = isolated_occupancy(**{where+'_kind': Mapping})
    assert namespace['_removed_snapshot_dispatch']([1, 2, 3]) is False


def test_replaced_len_is_not_added_to_the_legacy_dispatch():
    namespace = isolated_occupancy()
    def forbidden(*args): raise AssertionError('added len dispatch')
    namespace['len'] = forbidden; namespace['__builtins__']['len'] = forbidden
    engine = model()
    owner = namespace['occupancy'](engine)
    assert namespace['occupancy'](engine) is owner
    assert engine._removed_ids_proof is not None


def test_list_subclass_keeps_generator_error_order_even_after_a_proof_exists():
    engine = model(); events = []
    class Removed(list):
        def __len__(self): raise AssertionError('custom length was not in old predicate')
        def __iter__(self):
            events.append('iter'); yield 1560
            events.append('next'); raise RuntimeError('iteration boundary')
    engine.state['removed'] = Removed([1560, 2160, 2760])
    with pytest.raises(RuntimeError, match='iteration boundary'): engine.occupancy()
    assert events == ['iter', 'next'] and engine._removed_ids_proof is None
