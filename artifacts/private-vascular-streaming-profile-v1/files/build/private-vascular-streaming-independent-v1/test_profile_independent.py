"""Worker boundary controls only; never call the realistic-size kernel."""
from pathlib import Path
from types import SimpleNamespace
import builtins
import importlib.util
import json
import os
import stat
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('streaming_profile_independent_boundary',ROOT/'build/private-vascular-streaming-preparation-v1/profile_generated.py')
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)


def test_read_cap_survives_underreported_stat_size(tmp_path,monkeypatch):
    path=tmp_path/'grow.json';path.write_bytes(b'x'*33)
    monkeypatch.setattr(p.os,'fstat',lambda fd:SimpleNamespace(st_mode=stat.S_IFREG,st_size=0))
    with pytest.raises(ValueError,match='Bounded source/release read exceeded'):
        p.read_regular(path,32)


def test_valid_release_without_fresh_cache_guard_refuses_before_numpy(tmp_path,monkeypatch):
    pins,_=p.source_pins()
    release=tmp_path/'release.json'
    release.write_text(json.dumps({'scope':p.SCOPE,'source_hashes':pins,'root_release':True}))
    output=tmp_path/'never-created'
    monkeypatch.setattr(sys,'argv',['profile','--release',str(release),'--output-directory',str(output)])
    monkeypatch.setattr(sys,'dont_write_bytecode',True)
    monkeypatch.setattr(sys,'pycache_prefix',None)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name!='numpy','No scientific import for refused cache guard'
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(SystemExit,match='pycache_prefix'):
        p.main()
    assert not output.exists()


def test_inventory_parent_traversal_is_refused(monkeypatch):
    actual=p.read_regular
    entries={'src/resectionlab/../outside.py':'a'*64,'src/resectionlab/evaluation.py':'b'*64}
    def changed(path,cap=4*1024**2):
        if Path(path).name=='repository-source-pins.json':
            return json.dumps(entries).encode()
        return actual(path,cap)
    monkeypatch.setattr(p,'read_regular',changed)
    with pytest.raises(ValueError,match='Source inventory path refused'):
        p.source_pins()


def test_correct_file_label_with_wrong_loader_origin_is_refused(tmp_path,monkeypatch):
    wrong=tmp_path/'shadow.py';wrong.write_text('# generated loader origin\n')
    module=SimpleNamespace(__file__=str(ROOT/'src/resectionlab/evaluation.py'),
                           __spec__=SimpleNamespace(origin=str(wrong)))
    monkeypatch.setitem(sys.modules,'resectionlab.evaluation',module)
    with pytest.raises(ValueError,match='Loaded repository origin mismatch'):
        p.verify_loaded_repository_origins()
