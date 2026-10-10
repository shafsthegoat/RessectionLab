"""No-bulk regression for distinct native-launch and read-only comparison scopes."""
import ast
from pathlib import Path
from types import ModuleType
import inspect
import sys

import pytest
from scripts import mechanics_hbe_v5_native_comparison as native


def calls(function):
    tree=ast.parse(inspect.getsource(function))
    return [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]


def test_native_entry_uses_read_only_verifiers_not_native_release_audit():
    entry=calls(native.compare_native_rows)
    assert entry.count('remaining.validate_prior_chain')==2
    assert entry.count('supplement.verify_exact')==2
    assert 'remaining.validate_release' not in entry
    assert not any(name.endswith('audit_imports') for name in entry)
    # These are actual unchanged function bodies, not mocked chain call graphs.
    chain=calls(native.remaining.validate_prior_chain)
    assert 'admission.verify_exact' in chain
    assert not any(name.endswith(('audit_imports','validate_release')) for name in chain)
    inspection=calls(native.remaining.inspect_readout)
    assert not any(name.endswith('audit_imports') for name in inspection)


def test_exact_supplement_scopes_historical_import_audit_explicitly():
    tree=ast.parse(inspect.getsource(native.supplement.verify_exact))
    invocations=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                 and ast.unparse(node.func)=='replay.validate_original']
    assert len(invocations)==1
    keyword={item.arg:ast.literal_eval(item.value) for item in invocations[0].keywords
             if item.arg=='audit_loaded_imports'}
    assert keyword=={'audit_loaded_imports':False}
    original=ast.parse(inspect.getsource(native.supplement.replay.validate_original))
    conditions=[node for node in ast.walk(original) if isinstance(node,ast.If)
                and ast.unparse(node.test)=='audit_loaded_imports']
    assert len(conditions)==1
    conditional_calls=[ast.unparse(node.func) for node in ast.walk(conditions[0]) if isinstance(node,ast.Call)]
    assert conditional_calls==['audit_imports']
    audit_calls=[node for node in ast.walk(original) if isinstance(node,ast.Call)
                 and ast.unparse(node.func)=='audit_imports']
    assert len(audit_calls)==1


def test_frozen_twenty_source_launch_audit_rejects_comparator_without_allowlist_change(tmp_path,monkeypatch):
    frozen=tuple(native.remaining.SOURCE_PATHS)
    assert len(frozen)==20
    allowed=ModuleType('scripts.mechanics_hbe_v5_remaining_one_shot')
    allowed.__file__=str(tmp_path/'scripts/mechanics_hbe_v5_remaining_one_shot.py')
    new=ModuleType('scripts.mechanics_hbe_v5_native_comparison')
    new.__file__=str(tmp_path/'scripts/mechanics_hbe_v5_native_comparison.py')
    modules={allowed.__name__:allowed}
    monkeypatch.setattr(sys,'modules',modules)
    native.remaining.audit_imports(root=tmp_path)
    modules[new.__name__]=new
    with pytest.raises(ValueError,match='Unbound executing scripts import'):
        native.remaining.audit_imports(root=tmp_path)
    assert tuple(native.remaining.SOURCE_PATHS)==frozen
