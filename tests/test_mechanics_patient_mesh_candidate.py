"""Analytic software controls only; all mesher/locator calls are mocked."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('graded_mesh', ROOT/'scripts/mechanics_patient_mesh_candidate.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
CONFIG = v.declaration()


def analytic_tet():
    corners = .01*np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    nodes = np.vstack((corners, corners[v.BASE.EDGES].mean(axis=1)))
    cells = np.arange(10,dtype=np.int64).reshape(1,10)
    origin = {'gmsh_node_ids':list(range(11,21)), 'gmsh_element_ids':[81],
              'gmsh_to_febio_permutation':[0,1,2,3,4,5,6,7,9,8], 'discrete_patches':1}
    return nodes, cells, origin


def test_complete_diagnostic_roundtrip_all_arrays_and_hashes(tmp_path):
    nodes,cells,origin=analytic_tet(); limits=CONFIG['diagnostic_limits']
    report=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{'scope':'analytic_control'},limits)
    assert report['status']=='complete' and report['complete'] and not report['candidate_accepted']
    folder=Path(report['directory']); manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['diagnostic_complete'] and not manifest['candidate_accepted']
    expected={'nodes_m.npy':nodes,'tet10_indices.npy':cells,
              'gmsh_node_ids.npy':np.array(origin['gmsh_node_ids']),
              'gmsh_element_ids.npy':np.array(origin['gmsh_element_ids'])}
    for name,array in expected.items():
        np.testing.assert_array_equal(np.load(folder/name,allow_pickle=False),array)
        assert manifest['files'][name]['sha256']==v.sha(folder/name)
        assert manifest['files'][name]['bytes']==(folder/name).stat().st_size
    assert report['total_bytes']==sum(p.stat().st_size for p in folder.iterdir())
    assert report['total_bytes']<=limits['maximum_total_bytes']
    assert manifest['gmsh_to_febio_permutation']==origin['gmsh_to_febio_permutation']
    assert not (tmp_path/'diagnostic.partial').exists()


def test_header_ids_and_metadata_reservation_in_byte_gate(tmp_path):
    nodes,cells,origin=analytic_tet();limits=copy.deepcopy(CONFIG['diagnostic_limits'])
    limits['maximum_total_bytes']=nodes.nbytes+cells.nbytes
    report=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},limits)
    assert report['reason']=='diagnostic_byte_cap' and not report['complete']
    assert report['reserved_total_bytes']>limits['maximum_total_bytes']
    assert not list(tmp_path.iterdir())


def test_exact_reserved_budget_saves_without_unbounded_archive_overhead(tmp_path):
    nodes,cells,origin=analytic_tet();limits=copy.deepcopy(CONFIG['diagnostic_limits'])
    limits['maximum_total_bytes']=32*len(nodes)+88*len(cells)+4*128+limits['maximum_metadata_bytes']
    report=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},limits)
    assert report['complete'] and report['reserved_total_bytes']==limits['maximum_total_bytes']
    assert report['total_bytes']<=limits['maximum_total_bytes']


def test_count_limit_never_truncates_or_writes(tmp_path):
    nodes,cells,origin=analytic_tet();limits=copy.deepcopy(CONFIG['diagnostic_limits'])
    limits['maximum_nodes']=len(nodes)-1
    result=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},limits)
    assert result['counts']=={'nodes':10,'tet10_elements':1}
    assert result['reason']=='diagnostic_count_cap' and not result['complete']
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('kind',['ids','permutation','dtype','metadata'])
def test_incomplete_representation_or_metadata_is_omitted_before_writes(tmp_path,kind):
    nodes,cells,origin=analytic_tet();context={}
    if kind=='ids':origin['gmsh_node_ids']=origin['gmsh_node_ids'][:-1]
    if kind=='permutation':origin['gmsh_to_febio_permutation']=[0]*10
    if kind=='dtype':nodes=nodes.astype(np.float32)
    if kind=='metadata':context={'oversized':'x'*20000}
    result=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,context,CONFIG['diagnostic_limits'])
    assert not result['complete'] and result['status']=='omitted'
    assert not list(tmp_path.iterdir())


def test_short_array_write_retains_partial_without_complete_marker(tmp_path,monkeypatch):
    nodes,cells,origin=analytic_tet()
    monkeypatch.setattr(v.np.lib.format,'write_array',lambda stream,*args,**kwargs:stream.write(b'short'))
    report=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},CONFIG['diagnostic_limits'])
    assert report['status']=='write_failed' and not report['complete']
    assert (tmp_path/'diagnostic.partial/nodes_m.npy').read_bytes()==b'short'
    assert not (tmp_path/'diagnostic').exists()
    assert not (tmp_path/'diagnostic.partial/manifest.json').exists()


def test_same_size_corrupted_write_cannot_become_complete(tmp_path,monkeypatch):
    nodes,cells,origin=analytic_tet();write=v.np.lib.format.write_array
    def corrupt(stream,array,**kwargs):
        changed=array.copy();changed.flat[0]+=1
        write(stream,changed,**kwargs)
    monkeypatch.setattr(v.np.lib.format,'write_array',corrupt)
    report=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},CONFIG['diagnostic_limits'])
    assert report['status']=='write_failed' and not report['complete']
    assert not (tmp_path/'diagnostic').exists()
    assert not (tmp_path/'diagnostic.partial/manifest.json').exists()


def test_failed_atomic_rename_retains_partial_and_existing_attempt_is_untouched(tmp_path,monkeypatch):
    nodes,cells,origin=analytic_tet();rename=Path.rename
    def refuse(path,target):
        if path.name=='diagnostic.partial':raise OSError('Injected rename failure')
        return rename(path,target)
    monkeypatch.setattr(Path,'rename',refuse)
    report=v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},CONFIG['diagnostic_limits'])
    assert report['status']=='write_failed' and not (tmp_path/'diagnostic').exists()
    assert not (tmp_path/'diagnostic.partial/manifest.json').exists()
    before={p.name:p.read_bytes() for p in (tmp_path/'diagnostic.partial').iterdir()}
    with pytest.raises(FileExistsError):
        v.preserve_diagnostic(tmp_path/'diagnostic',nodes,cells,origin,{},CONFIG['diagnostic_limits'])
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'diagnostic.partial').iterdir()}


def install_mock_generation(monkeypatch,node_count=10):
    nodes,cells,origin=analytic_tet()
    if node_count>10:
        nodes=np.vstack((nodes,np.zeros((node_count-10,3))))
        origin['gmsh_node_ids']=list(range(1,node_count+1))
    calls=[]
    def generate(_api,_X,_F,_config,charge):
        charge();calls.append('one');return nodes,cells,origin
    monkeypatch.setattr(v,'generate_one',generate)
    return calls


def test_overeligibility_mesh_saved_completely_then_rejected_without_validation(tmp_path,monkeypatch):
    calls=install_mock_generation(monkeypatch,6001)
    monkeypatch.setattr(v.BASE,'validate_tet10',lambda *_:pytest.fail('No validation beyond count rejection'))
    corners=analytic_tet()[0][:4]
    r=v.assess_candidate(None,corners,v.BASE.FACES,tmp_path/'attempt',
        distance_factory=lambda *_:pytest.fail('No native locator'),config=copy.deepcopy(CONFIG))
    assert calls==['one'] and r['native_generation_calls']==1 and r['retries']==0
    assert r['status']=='failed_or_incomplete' and r['diagnostic']['complete']
    assert np.load(tmp_path/'attempt/diagnostic/nodes_m.npy').shape==(6001,3)
    assert 'quality' not in r and not r['solver_admitted'] and r['solver_calls']==0
    before=(tmp_path/'attempt/result.json').read_bytes()
    with pytest.raises(FileExistsError):
        v.assess_candidate(None,corners,v.BASE.FACES,tmp_path/'attempt',distance_factory=None)
    assert calls==['one'] and (tmp_path/'attempt/result.json').read_bytes()==before


def test_no_diagnostic_completeness_means_no_validation_continuation(tmp_path,monkeypatch):
    calls=install_mock_generation(monkeypatch)
    config=copy.deepcopy(CONFIG);config['diagnostic_limits']['maximum_total_bytes']=1
    monkeypatch.setattr(v.BASE,'validate_tet10',lambda *_:pytest.fail('No continuation'))
    r=v.assess_candidate(None,analytic_tet()[0][:4],v.BASE.FACES,tmp_path/'attempt',distance_factory=None,config=config)
    assert calls==['one'] and r['status']=='failed_or_incomplete'
    assert r['diagnostic']['reason']=='diagnostic_byte_cap' and 'quality' not in r


@pytest.mark.parametrize('distance',[0.,.003])
def test_independent_fidelity_and_no_solver_admission(tmp_path,monkeypatch,distance):
    calls=install_mock_generation(monkeypatch)
    r=v.assess_candidate(None,analytic_tet()[0][:4],v.BASE.FACES,tmp_path/'attempt',
        distance_factory=lambda *_:(lambda q:np.full(len(q),distance)),config=copy.deepcopy(CONFIG))
    assert calls==['one'] and r['diagnostic']['complete']
    assert r['status']==('failed_or_incomplete' if distance else 'geometry_candidate_passed_no_solver_authorization')
    assert r['source_to_mesh']['maximum_sample_distance_m']==distance
    assert 'relative_volume_error' in r and not r['solver_admitted']


def test_exact_prospective_size_field_calls_without_mesher():
    operations=[];values={};fields=iter([21,22])
    api=SimpleNamespace(model=SimpleNamespace(mesh=SimpleNamespace(field=SimpleNamespace(
        add=lambda kind:(operations.append(('add',kind)),next(fields))[1],
        setNumbers=lambda tag,key,val:values.__setitem__((tag,key),val),
        setNumber=lambda tag,key,val:values.__setitem__((tag,key),val),
        setAsBackgroundMesh=lambda tag:operations.append(('background',tag))))),
        option=SimpleNamespace(setNumber=lambda key,val:values.__setitem__(key,val)))
    v.configure_profile(api,[2,7,19],CONFIG['size_profile'])
    assert operations==[('add','Distance'),('add','Threshold'),('background',22)]
    assert values[(21,'SurfacesList')]==[2,7,19] and values[(21,'Sampling')]==100
    for key,val in [('InField',21),('SizeMin',.012),('SizeMax',.024),('DistMin',.002),('DistMax',.024)]:
        assert values[(22,key)]==val
    assert values['Mesh.MeshSizeMin']==.012 and values['Mesh.MeshSizeMax']==.024


def test_original_fixed_fidelity_and_quality_gates_unchanged():
    prior=json.loads((ROOT/'manifests/experiments/resect-case4-patient-mesh-v1.json').read_text())
    assert CONFIG['surface_fidelity']==prior['surface_fidelity']
    assert CONFIG['quality']==prior['quality'] and CONFIG['gmsh_options']==prior['gmsh_options']
    assert CONFIG['eligibility']['maximum_nodes']==6000 and CONFIG['eligibility']['maximum_elements']==6000
    assert CONFIG['caps']['maximum_generations']==1 and CONFIG['caps']['retries']==0
    assert not CONFIG['execution_release']['authorized'] and not CONFIG['solver_admission']['authorized']
    assert CONFIG['solver_admission']['three_worst_case_skyline_value_arrays_bytes']>3*1024**3
