"""Reusable strategy verdicts must not hide changed implementation/dispatch."""
import copy

import pytest

from resectionlab import surgical_strategy_workspace as workspace
from resectionlab import surgical_two_tool as planning
from resectionlab import surgical_task_admission as task
from resectionlab import surgical_patient
from test_surgical_two_tool import qualified, comparison, prepared


def record_for(case, engine, source, comparison):
    record = copy.deepcopy(source)
    workspace.install_report(engine, record, case.semantic_hash, comparison)
    workspace._VALIDATED.clear()
    return record


def test_collecting_genuine_stage_owners_keeps_empty_workspace_proof_epoch_stable():
    import gc
    import numpy as np
    from resectionlab.generated_cpp_fixture import joint_source
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab import surgical_proof_cache as proofs
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        # The cancelled forest can leave genuine private witnesses in a cycle.
        # Collect their owners without replacing code, constants or inputs.
        garbage = [joint_source()]
        garbage.append(garbage)
        del garbage
        scene = SurgicalScene(np.zeros((4, 4, 4), np.uint16), np.eye(4), [], {}, {})
        epoch = proofs.implementation_epoch()
        assert epoch is not None
        with proofs.proof_scope(scene, {}):
            assert gc.collect() > 0
            assert proofs.implementation_epoch() == epoch
        assert proofs.implementation_epoch() == epoch
    finally:
        if enabled:
            gc.enable()


def test_warm_workspace_does_not_hide_changed_report_validator(qualified, comparison, monkeypatch):
    case, engine, source, _ = qualified
    record = record_for(case, engine, source, comparison)
    workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    def changed(*args, **kwargs): raise RuntimeError('changed report implementation')
    monkeypatch.setattr(workspace, 'validate_report', changed)
    with pytest.raises(RuntimeError, match='changed report implementation'):
        workspace.validate_workspace(engine.scene, record, case.semantic_hash)


def test_non_json_baseline_cannot_borrow_equal_json_warm_authority(qualified, comparison, monkeypatch):
    case, engine, source, _ = qualified
    record = record_for(case, engine, source, comparison)
    workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    baseline = record['branches'][record['strategyComparisons'][0]['sourceBranchId']]
    baseline['actions'] = tuple(baseline['actions'])
    original = workspace.SurgicalEngine.restore.__func__
    def observed(cls, scene, value, **kwargs):
        if type(value['actions']) is tuple: raise RuntimeError('original tuple dispatch')
        return original(cls, scene, value, **kwargs)
    monkeypatch.setattr(workspace.SurgicalEngine, 'restore', classmethod(observed))
    with pytest.raises(RuntimeError, match='original tuple dispatch'):
        workspace.validate_workspace(engine.scene, record, case.semantic_hash)


def test_warm_workspace_rechecks_custom_scene_dispatch(qualified, comparison, monkeypatch):
    case, engine, source, _ = qualified
    record = record_for(case, engine, source, comparison)
    workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    def changed(*args, **kwargs): raise RuntimeError('custom scene point transform')
    object.__setattr__(engine.scene, 'points', changed)
    try:
        with pytest.raises(RuntimeError, match='custom scene point transform'):
            workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    finally:
        object.__delattr__(engine.scene, 'points')


def test_warm_report_does_not_hide_changed_public_compiler(qualified, comparison, monkeypatch):
    engine = qualified[1]
    planning.validate_report(engine, comparison)
    def changed(*args, **kwargs): raise RuntimeError('changed public compiler')
    monkeypatch.setattr(planning, '_compile', changed)
    with pytest.raises(RuntimeError, match='changed public compiler'):
        planning.validate_report(engine, comparison)


def test_warm_source_bytes_do_not_hide_changed_resampler(qualified, monkeypatch):
    engine = qualified[1]; value = task.admission(engine)
    task.validate_admission(engine, value, require_baseline=False)
    def changed(*args, **kwargs): raise RuntimeError('changed source resampler')
    monkeypatch.setattr(surgical_patient, '_resample', changed)
    with pytest.raises(RuntimeError, match='changed source resampler'):
        task.validate_admission(engine, value, require_baseline=False)


def test_stock_json_workspace_still_reuses_complete_verdicts(qualified, comparison):
    case, engine, source, _ = qualified
    record = record_for(case, engine, source, comparison)
    from resectionlab import surgical_proof_cache as proofs
    assert proofs._json(record) and proofs._stock_scene(engine.scene, None)
    assert proofs.implementation_epoch() is not None
    workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    assert len(workspace._VALIDATED) == 1
    from resectionlab.surgical_proof_cache import proof_scope
    with proof_scope(engine.scene, record):
        key = (engine.scene.identity, planning.semantic_digest(record['branches'][record['strategyComparisons'][0]['sourceBranchId']]),
               planning.semantic_digest(record['strategyComparisons'][0]))
        assert key in workspace._VALIDATED
    workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    assert workspace.baseline_cache_stats()['restores'] == 0


def test_transient_custom_implementation_cannot_read_or_publish_then_restore(qualified, monkeypatch):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); scene = qualified[1].scene
    with proof_scope(scene, {'owned': True}): cache['old'] = True
    with pytest.raises(ValueError, match='STRATEGY_PROOF'):
        with proof_scope(scene, {'owned': True}):
            original = planning._compile
            with monkeypatch.context() as patch:
                patch.setattr(planning, '_compile', lambda *args, **kwargs: None)
                with proof_scope(scene, {'nested': True}):
                    assert 'old' not in cache
                    cache['new'] = True
            assert planning._compile is original
    assert not cache


def test_callback_scene_override_taints_hits_and_final_return(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); scene = qualified[1].scene
    with proof_scope(scene, {}): cache['old'] = True
    try:
        with pytest.raises(ValueError, match='STRATEGY_PROOF'):
            with proof_scope(scene, {}):
                object.__setattr__(scene, 'points', lambda *args: None)
                assert 'old' not in cache
                object.__delattr__(scene, 'points')
    finally:
        if 'points' in vars(scene): object.__delattr__(scene, 'points')
    assert not cache


def test_closed_inherited_context_has_no_authority_or_retained_owners(qualified, monkeypatch):
    import contextvars
    from resectionlab import surgical_proof_cache as proofs
    cache = proofs.ProofCache(); scene = qualified[1].scene
    with proofs.proof_scope(scene, {'owned': True}):
        owner = proofs._ACTIVE.get(); cache['old'] = True
        inherited = contextvars.copy_context()
    assert owner.closed and owner.scene is owner.engine is owner.inputs is owner.parent is None
    assert inherited.run(lambda: 'old' not in cache)
    with monkeypatch.context() as patch:
        patch.setattr(planning, '_compile', lambda *args, **kwargs: None)
        def later():
            with proofs.proof_scope(scene, {'owned': True}): assert 'old' not in cache
        inherited.run(later)


def test_original_input_mutation_is_detected_after_cached_verdict(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); scene = qualified[1].scene; dto = {'value': 1}
    with proof_scope(scene, dto): cache['old'] = True
    with pytest.raises(ValueError, match='STRATEGY_PROOF'):
        with proof_scope(scene, dto):
            assert 'old' in cache
            dto['value'] = 2
    assert not cache


def test_unknown_closure_private_owner_is_neither_copied_nor_retained(qualified, monkeypatch):
    import weakref,gc
    from resectionlab.surgical_proof_cache import implementation_epoch
    class PrivateOwner:
        def __deepcopy__(self, memo): pytest.fail('private evaluator copied')
        def __repr__(self): pytest.fail('private evaluator formatted')
    def make_callback():
        owner = PrivateOwner()
        def callback(): return owner
        callback.__module__ = planning.__name__
        return callback, weakref.ref(owner)
    callback, reference = make_callback()
    with monkeypatch.context() as patch:
        patch.setattr(planning, '_compile', callback)
        assert implementation_epoch() is None
    del callback; gc.collect()
    assert reference() is None


def test_inplace_compiler_code_change_invalidates_warm_report(qualified, comparison, monkeypatch):
    engine = qualified[1]; planning.validate_report(engine, comparison)
    def changed(*args, **kwargs): raise RuntimeError('in-place compiler change')
    with monkeypatch.context() as patch:
        patch.setattr(planning._compile, '__code__', changed.__code__)
        with pytest.raises(RuntimeError, match='in-place compiler change'):
            planning.validate_report(engine, comparison)


def test_nested_non_json_dto_never_uses_or_publishes_authority(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); scene = qualified[1].scene
    with proof_scope(scene, {'plain': [1, 2]}): cache['old'] = True
    with proof_scope(scene, {'plain': [1, 2]}):
        with proof_scope(scene, {'plain': (1, 2)}):
            assert 'old' not in cache
            cache['custom'] = True
        assert 'old' in cache and 'custom' not in cache


def test_scene_private_evaluator_attributes_are_not_inspected_or_retained(qualified, comparison):
    import weakref,gc
    engine = qualified[1]
    class PrivateOwner:
        def __deepcopy__(self, memo): pytest.fail('private owner copied')
        def __repr__(self): pytest.fail('private owner formatted')
    owner = PrivateOwner(); reference = weakref.ref(owner)
    engine._private_evaluation_truth = owner
    try:
        assert planning.validate_report(engine, comparison)
    finally:
        del engine._private_evaluation_truth, owner
    gc.collect(); assert reference() is None


def test_replaced_equal_frame_owner_taints_reuse_and_cannot_be_restored_away(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); scene = qualified[1].scene; original = scene.affine
    with proof_scope(scene, {}): cache['old'] = True
    try:
        with pytest.raises(ValueError, match='STRATEGY_PROOF'):
            with proof_scope(scene, {}):
                scene.affine = scene.affine.copy()
                with pytest.raises(ValueError, match='Source scene changed'): scene.require_cache_source()
                assert 'old' not in cache
                scene.affine = original
    finally: scene.affine = original
    assert not cache


def test_changed_live_engine_binding_prevents_successful_return(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); engine = qualified[1]; elapsed = engine.state['elapsedSeconds']
    try:
        with pytest.raises(ValueError, match='STRATEGY_PROOF'):
            with proof_scope(engine.scene, {}, engine=engine):
                cache['owned'] = True
                engine.state['elapsedSeconds'] += .1
    finally: engine.state['elapsedSeconds'] = elapsed
    assert not cache


def test_transient_engine_change_taints_cache_hits_even_after_restoration(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    cache = ProofCache(); engine = qualified[1]; elapsed = engine.state['elapsedSeconds']
    with proof_scope(engine.scene, {}, engine=engine): cache['old'] = True
    try:
        with pytest.raises(ValueError, match='STRATEGY_PROOF'):
            with proof_scope(engine.scene, {}, engine=engine):
                engine.state['elapsedSeconds'] += .1
                assert 'old' not in cache
                engine.state['elapsedSeconds'] = elapsed
    finally: engine.state['elapsedSeconds'] = elapsed
    assert not cache


def test_verdict_cache_accepts_only_bounded_primitive_authority_metadata(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    class PrivateOwner:
        def __hash__(self): pytest.fail('private owner hashed')
        def __eq__(self, other): pytest.fail('private owner compared')
        def __repr__(self): pytest.fail('private owner formatted')
    owner = PrivateOwner(); cache = ProofCache()
    with proof_scope(qualified[1].scene, {}):
        assert owner not in cache
        cache[owner] = True; cache['private'] = owner
        cache['too-large'] = ('x',)*4097
        cache['valid'] = (('source-path', 1, 2, 3, 4, 5),)
        assert len(cache) == 1 and cache['valid'][0][0] == 'source-path'


def test_json_certificate_detects_nested_alias_edits_and_equal_noninteger_replacement(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    scene = qualified[1].scene; cache = ProofCache()
    for edit in (lambda values: values.__setitem__(1, True), lambda values: values.append(3),
                 lambda values: values.reverse()):
        values = [0, 1, 2]; dto = {'first': values, 'alias': values}
        with proof_scope(scene, dto): cache['old'] = True
        with pytest.raises(ValueError, match='STRATEGY_PROOF'):
            with proof_scope(scene, dto):
                edit(values); assert 'old' not in cache
        assert not cache


def test_equal_json_replacement_keeps_original_validation_semantics(qualified):
    from resectionlab.surgical_proof_cache import ProofCache, proof_scope
    dto = {'value': [3001, 3002]}; cache = ProofCache()
    with proof_scope(qualified[1].scene, dto):
        cache['old'] = True
        dto['value'] = [int(str(v)) for v in dto['value']]
        assert 'old' in cache


@pytest.mark.parametrize('name', ['all', 'any', 'type', 'len', 'map', 'int', 'list', 'tuple', '__import__'])
def test_actual_builtin_replacement_declines_before_epoch_and_metadata_dispatch(qualified, name):
    import builtins
    from resectionlab import surgical_proof_cache as proofs
    cache = proofs.ProofCache()
    with proofs.proof_scope(qualified[1].scene, {}): cache['old'] = True
    calls = []
    def changed(*args, **kwargs):
        calls.append(name)
        raise AssertionError('Changed builtin was invoked by proof admission')
    original = builtins.__dict__[name]
    try:
        builtins.__dict__[name] = changed
        epoch = proofs.implementation_epoch()
        hit = 'old' in cache
        cache['new'] = True
    finally:
        builtins.__dict__[name] = original
    assert epoch is None and hit is False and not calls
    assert len(cache) == 1


def test_replaced_all_cannot_admit_equal_non_json_input_or_publish(qualified):
    import builtins
    from resectionlab import surgical_proof_cache as proofs
    cache = proofs.ProofCache(); scene = qualified[1].scene
    with proofs.proof_scope(scene, {'value': [1]}): cache['old'] = True
    original = builtins.all
    try:
        builtins.all = lambda iterable: True
        with proofs.proof_scope(scene, {'value': (1,)}):
            allowed = proofs._ACTIVE.get().allowed
            hit = 'old' in cache
            cache['new'] = True
    finally:
        builtins.all = original
    assert allowed is False and hit is False and len(cache) == 1


@pytest.mark.parametrize('where', ['globals', 'copied_builtins'])
def test_contributing_function_effective_lookup_cannot_borrow_warm_proof(qualified, monkeypatch, where):
    from types import FunctionType
    from resectionlab import surgical_proof_cache as proofs
    scene = qualified[1].scene; cache = proofs.ProofCache()
    with proofs.proof_scope(scene, {}): cache['old'] = True
    original = planning._compile
    namespace = dict(original.__globals__)
    namespace['__builtins__'] = dict(original.__builtins__)
    target = namespace if where == 'globals' else namespace['__builtins__']
    target['all'] = lambda iterable: True
    changed = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    with monkeypatch.context() as patch:
        patch.setattr(planning, '_compile', changed)
        assert proofs.implementation_epoch() is None
        with proofs.proof_scope(scene, {}):
            assert 'old' not in cache
            cache['new'] = True
    assert len(cache) == 1
    assert proofs.implementation_epoch() is not None


def test_proof_helper_copied_builtin_mapping_is_checked_before_use(qualified, monkeypatch):
    from types import FunctionType
    from resectionlab import surgical_proof_cache as proofs
    original = proofs._json
    namespace = dict(original.__globals__)
    namespace['__builtins__'] = dict(original.__builtins__)
    namespace['__builtins__']['type'] = lambda value: dict
    changed = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    with monkeypatch.context() as patch:
        patch.setattr(proofs, '_json', changed)
        assert proofs.implementation_epoch() is None
    assert proofs.implementation_epoch() is not None


@pytest.mark.parametrize('name', ['all', 'type', 'len', 'map', 'tuple', '__import__'])
def test_preimport_builtin_replacements_are_not_blessed_as_originals(name):
    import ast
    import builtins
    from pathlib import Path
    from resectionlab import surgical_proof_cache as proofs
    nodes = []
    for node in ast.parse(Path(proofs.__file__).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_genuine_builtins', '_dispatch', '_proof_dispatch'):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and
                target.id.startswith(('_intrinsic_', '_builtin_', '_proof_globals')) for target in node.targets):
            nodes.append(node)
    code = compile(ast.Module(body=nodes, type_ignores=[]), 'isolated-proof-builtin-captures', 'exec')
    namespace = {'_builtins': builtins, '__builtins__': dict(builtins.__dict__),
                 '__name__': proofs.__name__}
    run = exec
    original = builtins.__dict__[name]
    def changed(*args, **kwargs): raise AssertionError('Pre-import builtin invoked')
    try:
        builtins.__dict__[name] = changed
        namespace['__builtins__'][name] = changed
        run(code, namespace)
        genuine = namespace['_builtin_captures_genuine']
        dispatch = namespace['_proof_dispatch']()
    finally:
        builtins.__dict__[name] = original
    assert genuine is False and dispatch is False


def test_observed_transient_builtin_change_taints_scope_after_restoration(qualified):
    import builtins
    from resectionlab import surgical_proof_cache as proofs
    cache = proofs.ProofCache(); scene = qualified[1].scene
    with proofs.proof_scope(scene, {}): cache['old'] = True
    with pytest.raises(ValueError, match='STRATEGY_PROOF'):
        with proofs.proof_scope(scene, {}):
            original = builtins.all
            try:
                builtins.all = lambda iterable: True
                hit = 'old' in cache
                cache['new'] = True
            finally:
                builtins.all = original
            assert hit is False
    assert not cache
