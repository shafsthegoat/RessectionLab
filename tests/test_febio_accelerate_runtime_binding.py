"""Final declaration joins only; no configure/compiler/runtime execution."""
import copy
import json
from pathlib import Path

import pytest
from scripts import febio_accelerate_runtime as d


def test_final_declaration_accepts_saved_independent_controls_only(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'OUTPUT',tmp_path/'not-created')
    monkeypatch.setattr(d.subprocess,'run',lambda *a,**k:pytest.fail('No command allowed'))
    spec, effective=d.load_declaration(d.rt.sha(d.DECLARATION))
    assert spec['adapter_control_validation'] is not None
    assert effective['patch']['patched_sha256']==d.PATCHED_SHA
    assert not d.OUTPUT.exists()
    historical=d.ROOT/'artifacts/febio-accelerate-csc-runtime-preparation-v3/prospective-runtime-declaration.json'
    assert d.rt.sha(historical)=='4f24c624808a7077e2a02a6350e85efd2b86d407180c1a5599e4e456d94dcedd'
    with pytest.raises(ValueError,match='adapter controls'):
        d.require_adapter_controls(json.loads(historical.read_bytes()))


def graph(tmp_path):
    spec=json.loads(d.DECLARATION.read_bytes())
    originals=spec['adapter_control_validation']
    records={key:json.loads((d.ROOT/value['path']).read_bytes()) for key,value in originals.items()}
    def write(key):
        path=tmp_path/(key+'.json');path.write_text(json.dumps(records[key]))
        return {'path':str(path),'sha256':d.rt.sha(path),'bytes':path.stat().st_size}
    def seal():
        bindings={key:write(key) for key in ('declaration','result','supervision')}
        records['acceptance'].update(declaration_sha256=bindings['declaration']['sha256'],
            result_sha256=bindings['result']['sha256'],supervision_sha256=bindings['supervision']['sha256'])
        # The declared source/control bytes, not JSON whitespace, are unchanged.
        records['result']['declaration_sha256']=bindings['declaration']['sha256']
        bindings['result']=write('result')
        records['acceptance']['result_sha256']=bindings['result']['sha256']
        bindings['acceptance']=write('acceptance')
        records['independent_review'].update(result_sha256=bindings['result']['sha256'],
            acceptance_sha256=bindings['acceptance']['sha256'],supervision_sha256=bindings['supervision']['sha256'])
        bindings['independent_review']=write('independent_review')
        result=copy.deepcopy(spec);result['adapter_control_validation']=bindings
        result['prospective_adapter_controls']=bindings['declaration']
        return result
    return records,seal


@pytest.mark.parametrize('target,field,value',[
    ('acceptance','status','failed_no_retry'),
    ('result','patched_source_sha256','0'*64),
    ('result','sdk_factorization_executed',True),
    ('supervision','exit_code',1),
    ('supervision','wall_cap_seconds',101),
    ('independent_review','status','preparation_only'),
])
def test_resealed_record_graph_cannot_hide_semantic_failure(tmp_path,target,field,value):
    records,seal=graph(tmp_path)
    d.require_adapter_controls(seal())
    records[target][field]=value
    with pytest.raises(ValueError,match='accepted repaired-only evidence'):
        d.require_adapter_controls(seal())
