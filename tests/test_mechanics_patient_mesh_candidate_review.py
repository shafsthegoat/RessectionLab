"""Independent pure/mocked preparation checks; no patient or native API access."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_independent_review',ROOT/'scripts/mechanics_patient_mesh_candidate.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
CONFIG=json.loads((ROOT/'manifests/experiments/resect-case4-patient-mesh-graded-v2.json').read_text())


def arrays(n=10,e=1):
    X=np.arange(n*3,dtype=np.float64).reshape(n,3)*.00001
    E=np.tile(np.arange(10,dtype=np.int64),(e,1))
    origin={'gmsh_node_ids':list(range(101,101+n)),'gmsh_element_ids':list(range(3001,3001+e)),
            'gmsh_to_febio_permutation':list(range(10)),'discrete_patches':1}
    return X,E,origin


def test_full_maximum_packet_keeps_every_array_and_counts_headers_and_manifest(tmp_path):
    X,E,origin=arrays(10000,20000)
    # Include nonfinite geometry as diagnostic evidence, never sanitize it.
    X[123,2]=np.nan
    result=v.preserve_diagnostic(tmp_path/'packet',X,E,origin,{'fixture':'not patient'},CONFIG['diagnostic_limits'])
    assert result['complete'] and not result['candidate_accepted']
    manifest=json.loads((tmp_path/'packet/manifest.json').read_text())
    paths=list((tmp_path/'packet').iterdir())
    assert len(paths)==5 and sum(p.stat().st_size for p in paths)==result['total_bytes']<=2**21
    assert result['total_bytes']>X.nbytes+E.nbytes+8*(len(X)+len(E))
    for name,expected in [('nodes_m.npy',X),('tet10_indices.npy',E),('gmsh_node_ids.npy',origin['gmsh_node_ids']),('gmsh_element_ids.npy',origin['gmsh_element_ids'])]:
        np.testing.assert_equal(np.load(tmp_path/'packet'/name,allow_pickle=False),expected)
        assert v.sha(tmp_path/'packet'/name)==manifest['files'][name]['sha256']
    assert manifest['diagnostic_complete'] and manifest['candidate_accepted'] is False


@pytest.mark.parametrize('gate',['nodes','elements','bytes','metadata'])
def test_omission_is_complete_and_explicit_without_truncation(tmp_path,gate):
    X,E,origin=arrays();limits=copy.deepcopy(CONFIG['diagnostic_limits']);context={}
    if gate=='nodes':limits['maximum_nodes']=9
    elif gate=='elements':limits['maximum_elements']=0
    elif gate=='bytes':limits['maximum_total_bytes']=X.nbytes+E.nbytes+8*(len(X)+len(E))
    else:context={'too_large':'x'*8192}
    result=v.preserve_diagnostic(tmp_path/'packet',X,E,origin,context,limits)
    assert result['status']=='omitted' and result['complete'] is False
    assert result['counts']=={'nodes':10,'tet10_elements':1}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('failure',['array_short_write','same_length_corrupt_write','marker_rename'])
def test_failed_packet_never_has_a_complete_marker_and_keeps_partial_evidence(tmp_path,monkeypatch,failure):
    X,E,origin=arrays()
    if failure in ('array_short_write','same_length_corrupt_write'):
        real=np.lib.format.write_array;calls=[]
        def short(stream,*args,**kwargs):
            calls.append(1)
            if failure=='array_short_write' and len(calls)==2:stream.write(b'partial')
            else:
                real(stream,*args,**kwargs)
                if failure=='same_length_corrupt_write' and len(calls)==1:
                    end=stream.tell();stream.seek(end-8);stream.write(b'\x00'*8)
        monkeypatch.setattr(np.lib.format,'write_array',short)
    else:
        real=Path.rename
        def reject_marker(path,destination):
            if path.name=='manifest.pending':raise OSError('injected marker commit failure')
            return real(path,destination)
        monkeypatch.setattr(Path,'rename',reject_marker)
    result=v.preserve_diagnostic(tmp_path/'packet',X,E,origin,{},CONFIG['diagnostic_limits'])
    assert result['status']=='write_failed' and result['complete'] is False
    assert not (tmp_path/'packet').exists()
    assert (tmp_path/'packet.partial').is_dir()
    assert list((tmp_path/'packet.partial').iterdir())
    assert not list(tmp_path.rglob('manifest.json'))
    with pytest.raises(FileExistsError):
        v.preserve_diagnostic(tmp_path/'packet',X,E,origin,{},CONFIG['diagnostic_limits'])


def test_count_rejection_preserves_full_packet_and_stops_before_quality_or_distance(tmp_path,monkeypatch):
    X,E,origin=arrays(6001,1)
    source=np.array([[0.,0,0],[.01,0,0],[0,.01,0],[0,0,.01]])
    calls=[]
    def generate(_gmsh,vertices,faces,config,charge):
        calls.append(config['candidate_id']);charge();return X,E,origin
    monkeypatch.setattr(v,'generate_one',generate)
    monkeypatch.setattr(v.BASE,'validate_tet10',lambda *_:pytest.fail('Count-rejected mesh cannot enter quality'))
    result=v.assess_candidate(None,source,v.BASE.FACES,tmp_path/'attempt',
        distance_factory=lambda *_:pytest.fail('No count-rejected distance'),config=CONFIG)
    assert len(calls)==result['native_generation_calls']==1
    assert result['status']=='failed_or_incomplete' and result['solver_admitted'] is False
    assert result['diagnostic']['complete'] and result['returned_nodes']==6001
    np.testing.assert_array_equal(np.load(tmp_path/'attempt/diagnostic/nodes_m.npy'),X)
    assert 'quality' not in result and 'source_to_mesh' not in result
    with pytest.raises(FileExistsError):v.assess_candidate(None,source,v.BASE.FACES,tmp_path/'attempt',distance_factory=None,config=CONFIG)
    assert len(calls)==1


def test_graded_field_is_only_a_requested_target_and_cannot_override_post_gates():
    fields={};options={};background=[]
    def add(kind):
        key=len(fields)+1;fields[key]={'kind':kind};return key
    field=SimpleNamespace(add=add,setNumbers=lambda k,n,x:fields[k].__setitem__(n,list(x)),
        setNumber=lambda k,n,x:fields[k].__setitem__(n,x),setAsBackgroundMesh=background.append)
    gmsh=SimpleNamespace(model=SimpleNamespace(mesh=SimpleNamespace(field=field)),
                         option=SimpleNamespace(setNumber=lambda n,x:options.__setitem__(n,x)))
    v.configure_profile(gmsh,[2,19],CONFIG['size_profile'])
    assert fields[1]=={'kind':'Distance','SurfacesList':[2,19],'Sampling':100}
    assert fields[2]=={'kind':'Threshold','InField':1,'SizeMin':.012,'SizeMax':.024,
        'DistMin':.002,'DistMax':.024,'Sigmoid':0,'StopAtDistMax':0}
    assert background==[2] and options=={'Mesh.MeshSizeMin':.012,'Mesh.MeshSizeMax':.024}
    old=json.loads((ROOT/v.BASE.MANIFEST).read_text())
    assert CONFIG['surface_fidelity']==old['surface_fidelity'] and CONFIG['quality']==old['quality']
    assert CONFIG['gmsh_options']==old['gmsh_options']
    assert CONFIG['caps']['maximum_generations']==1 and CONFIG['caps']['retries']==0
    assert CONFIG['solver_admission']['authorized'] is False and CONFIG['execution_release']['authorized'] is False
    assert CONFIG['solver_admission']['three_worst_case_skyline_value_arrays_bytes']>3*2**30
