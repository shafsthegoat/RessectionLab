"""Static extraction and mocked orchestration only; no C++ execution or patch command."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
from scripts import febio_accelerate_runtime as d

HERE=d.ROOT/'artifacts/febio-accelerate-lifecycle-v1'


def test_literal_methods_and_csc_fragment_authenticity():
    source=(HERE/'patched-AccelerateSparseSolver.cpp').read_text()
    manifest=json.loads((HERE/'extraction.json').read_bytes())
    assert hashlib.sha256(source.encode()).hexdigest()==manifest['source_sha256']==d.PATCHED_SHA
    fragments=[]
    for item in manifest['extracted_methods']:
        a=source.index(item['signature']); start=source.index('{',a); depth=1; i=start+1
        while depth:
            depth+=(source[i]=='{')-(source[i]=='}'); i+=1
        if item['signature'].startswith('class '):i+=1
        part=source[a:i]+'\n'
        assert len(part.encode())==item['bytes']
        assert hashlib.sha256(part.encode()).hexdigest()==item['sha256']
        fragments.append(part)
    assert '\n'.join(fragments)==(HERE/'extracted-methods.inc').read_text()
    legacy=(d.ROOT/'artifacts/febio-accelerate-matrix-init-v1/patched-AccelerateSparseSolver.cpp').read_text()
    a=source.index('    if (imp->m_n == 0) return LinearSolver::PreProcess();')
    b=source.index('    if (__builtin_available',a)
    assert source[a:b] in legacy
    initialization=(d.ROOT/'artifacts/febio-accelerate-matrix-init-v1/matrix-initialization-fragment.cpp.txt').read_text()
    assert initialization in source
    assert 'SparseRefactor(' not in source


def test_prospective_controls_pin_all_files_but_cannot_build(monkeypatch):
    monkeypatch.setattr(d.subprocess,'run',lambda *a,**k:pytest.fail('No command allowed'))
    declaration=json.loads((HERE/'control-declaration.json').read_bytes())
    for item in declaration['pins'].values():d.pin(item)
    spec,v1=d.load_declaration(d.rt.sha(d.DECLARATION),require_controls=False)
    prospective=json.loads((d.ROOT/'artifacts/febio-accelerate-csc-runtime-preparation-v3/prospective-runtime-declaration.json').read_bytes())
    assert prospective['adapter_control_validation'] is None
    with pytest.raises(ValueError,match='adapter controls'):d.require_adapter_controls(prospective)
    assert v1['patch']['file']['sha256']==d.PATCH_SHA
    assert v1['tools']==json.loads((d.ROOT/'manifests/experiments/febio-accelerate-csc-runtime-v1.json').read_bytes())['tools']


@pytest.mark.parametrize('success',[True,False])
def test_multihunk_patch_has_one_pinned_attempt_no_retry(tmp_path,monkeypatch,success):
    source_text=(d.OLD_PREFIX/'source/NumCore/AccelerateSparseSolver.cpp').read_bytes()
    spec,v1=d.load_declaration(d.rt.sha(d.DECLARATION),require_controls=False)
    prefix=tmp_path/'new';output=tmp_path/'out';output.mkdir()
    monkeypatch.setattr(d,'PREFIX',prefix)
    def extract(stream,destination,**kwargs):
        (destination/'NumCore').mkdir(parents=True)
        (destination/'NumCore/AccelerateSparseSolver.cpp').write_bytes(source_text)
        return {'mock':True}
    monkeypatch.setattr(d.rt,'extract_tar',extract)
    monkeypatch.setattr(d,'verify_source',lambda *a,**kw:{'mock':True})
    calls=[]
    def patched(argv,**kwargs):
        calls.append(argv)
        assert argv[:5]==['/usr/bin/patch','--batch','--forward','--fuzz=0','-p1']
        assert kwargs['timeout']==15 and kwargs['cwd']==prefix/'source'
        if success:(prefix/'source/NumCore/AccelerateSparseSolver.cpp').write_bytes((HERE/'patched-AccelerateSparseSolver.cpp').read_bytes())
        return subprocess.CompletedProcess(argv,0 if success else 1)
    monkeypatch.setattr(d.subprocess,'run',patched)
    if success:d.prepare_source(v1,output)
    else:
        with pytest.raises(ValueError,match='patch application failed'):d.prepare_source(v1,output)
    assert len(calls)==1 and (prefix/'patch-attempt.json').exists()
    with pytest.raises(FileExistsError):d.prepare_source(v1,output)
    assert len(calls)==1


def test_control_driver_declared_caps_and_defined_mock_scope():
    spec=importlib.util.spec_from_file_location('adapter_control',HERE/'adapter-control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    declaration=module.declaration(module.sha(module.DECLARATION))
    assert declaration['caps']=={'wall_seconds':100,'process_group_sampled_rss_bytes':1024**3,
        'per_fixture_compile_seconds':40,'per_fixture_run_seconds':5,'attempts':1,'automatic_retry':False}
    assert declaration['expected_stdout']==module.EXPECTED_LOGS
    fixture=(HERE/'lifecycle-control.cpp').read_text()
    assert fixture.index('#include <Accelerate/Accelerate.h>')<fixture.index('#define SparseFactor MockSparseFactor')
    assert '#include "extracted-methods.inc"' in fixture
    assert 'if(numeric_mode==Numeric::failed_empty)' in fixture
    assert 'Numeric::failed_symbolic_only' in fixture and 'Numeric::failed_storage' in fixture
    assert 'assert(released[0] && released[1])' in fixture
