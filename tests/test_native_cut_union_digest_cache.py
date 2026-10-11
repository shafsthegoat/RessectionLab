"""Generated controls for exact ledger-digest memoization only."""
from pathlib import Path
import hashlib
import importlib.util
import sys

import numpy as np
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT/'src'))


def load(name, path):
    spec = importlib.util.spec_from_file_location('resectionlab.'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    baseline = ROOT/'artifacts/native-cut-union-digest-cache-v1/baseline/native_cut_union.py'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == 'bd74877ae7448a7acc42d83f837e1933bb73564d45efc356929178f9dc5ba241'
    from resectionlab import native_cut_union as current
    return load('_digest_baseline', baseline), current


def test_hits_still_validate_public_mutations_and_spec(modules, monkeypatch):
    old, new = modules
    calls=[]; original=new.semantic_digest
    monkeypatch.setattr(new, 'semantic_digest', lambda value: (calls.append(1), original(value))[1])
    c=new.PreviewCutCoverage(new.CutUnionSpec(),{},np.eye(4));calls.clear()
    initial=c.digest
    assert c.digest==initial and len(calls)==1
    c.cells[(1,2,3)]=1
    assert c.digest==old.ledger_digest(old.CutUnionSpec(),c.cells) and len(calls)==2
    assert c.digest==new.ledger_digest(c.spec,c.cells)
    c.cells[(1,2,3)]=True  # bool==1 must not turn this into an undetected hit.
    with pytest.raises(ValueError):c.digest
    c.cells[(1,2,3)]=1
    c.spec=new.CutUnionSpec(max_depth=2)
    assert c.digest==old.ledger_digest(old.CutUnionSpec(max_depth=2),c.cells)
    del c.cells[(1,2,3)]
    assert c.digest==old.ledger_digest(old.CutUnionSpec(max_depth=2),{})


def test_complementary_cover_prune_probe_and_cap_keep_exact_records(modules):
    old,new=modules
    a=old.PreviewCutCoverage(old.CutUnionSpec(),{},np.eye(4))
    b=new.PreviewCutCoverage(new.CutUnionSpec(),{},np.eye(4))
    cell=np.array([[2,2,3]]);empty=np.empty((0,3),int)
    free=np.ones((5,5,7),bool);free[2,2,3]=False
    for x in (1.65,2.35):
        for z in (2.,2.5,3.,3.6):
            args=(cell,empty,free,np.array([x,2,z-2.]),np.array([x,2,z]),.65)
            ar=a.cover(*args);br=b.cover(*args)
            assert ar==br and a.cells==b.cells
            aa=a.complete(cell);bb=b.complete(cell)
            assert np.array_equal(aa,bb)
            assert a.finish_step(ar,aa)==b.finish_step(br,bb)
            assert a.empty_step()==b.empty_step()
    assert not a.cells and not b.cells
    assert a.new_leaf_count==b.new_leaf_count==512
    assert a.leaf_tests==b.leaf_tests and a.witness_rows==b.witness_rows
    # Fresh preview reset, private clone-like copies, and failed preview state.
    for module in (old,new):
        original={(2,2,3):1}
        bounded=module.PreviewCutCoverage(module.CutUnionSpec(max_leaf_tests_per_preview=1),original,np.eye(4))
        before=bounded.digest
        with pytest.raises(module.CutUnionLimit):bounded.cover(cell,empty,free,np.array([1.65,2,1.]),np.array([1.65,2,3.6]),.65)
        assert bounded.digest==before and original=={(2,2,3):1}
        sibling=module.PreviewCutCoverage(bounded.spec,bounded.freeze(),np.eye(4))
        sibling.cells.clear()
        assert bounded.digest==before and sibling.digest!=before
