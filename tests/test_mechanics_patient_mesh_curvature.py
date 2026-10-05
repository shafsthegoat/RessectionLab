"""Preparation controls: fake mesher and analytic tet only, no native calls/data."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

run=load('scripts/mechanics_patient_mesh_curvature_run.py','curvature_run_control')
v=run.HELPER
old=v.SHARED
caller=run.SHARED
CONFIG=v.declaration()


def tet():
    corners=np.array([[0.,0,0],[.01,0,0],[0,.01,0],[0,0,.01]])
    return np.vstack([corners,corners[v.BASE.EDGES].mean(axis=1)])


def fake_gmsh():
    events=[];options={};fields={};ids=iter([31,32]);nodes=tet()
    def option(name,value):options[name]=value;events.append(('option',name,value))
    mesh=SimpleNamespace(
        field=SimpleNamespace(add=lambda kind:next(ids),
            setNumbers=lambda tag,name,value:fields.__setitem__((tag,name),value),
            setNumber=lambda tag,name,value:fields.__setitem__((tag,name),value),
            setAsBackgroundMesh=lambda tag:events.append(('background',tag))),
        addNodes=lambda *a:events.append(('nodes',)),addElementsByType=lambda *a:events.append(('triangles',)),
        classifySurfaces=lambda *a:events.append(('classify',a,dict(options))),
        createGeometry=lambda:events.append(('parametrize',)),
        generate=lambda dim:events.append(('generate',dim,dict(options))),
        getElements=lambda dim:([11],[np.array([91])],[np.arange(1,11)]),
        getElementProperties=lambda kind:('tet10',3,2,10,(nodes/.01).ravel(),4),
        getNodes=lambda:(np.arange(1,11),nodes.ravel(),[]))
    api=SimpleNamespace(option=SimpleNamespace(setNumber=option),clear=lambda:events.append(('clear',)),
        model=SimpleNamespace(add=lambda name:events.append(('model',name)),mesh=mesh,
            addDiscreteEntity=lambda dim:7,getEntities=lambda dim:[(2,7)],
            geo=SimpleNamespace(addSurfaceLoop=lambda tags:8,addVolume=lambda loops:9,
                                synchronize=lambda:events.append(('synchronize',)))))
    return api,events,fields


def test_only_declared_profile_and_count_changes_preserve_physical_gates():
    previous=old.declaration()
    for key in ['source_surface_binding','source_frame','size_profile','caps','quality','surface_fidelity']:
        assert CONFIG[key]==previous[key]
    modified={key for key in CONFIG['gmsh_options'] if CONFIG['gmsh_options'][key]!=previous['gmsh_options'][key]}
    assert modified=={'Mesh.MeshSizeFromCurvature'}
    assert CONFIG['gmsh_options']['Mesh.MeshSizeFromCurvature']==24
    assert CONFIG['global_size_profile']['minimum_m']==.003
    assert CONFIG['eligibility']['maximum_nodes']==64000
    assert CONFIG['eligibility']['maximum_elements']==80000
    assert not CONFIG['execution_release']['authorized'] and not CONFIG['solver_admission']['authorized']
    assert len(run.SPEC.closure)==10
    assert run.SPEC.output_bytes==32*1024**2 and run.SPEC.file_bytes==8*1024**2


@pytest.mark.parametrize('key,value',[
    ('global_size_profile',{'minimum_m':.002}),('caps',{'retries':1}),
    ('quality',{'minimum_mean_ratio':.01}),('surface_fidelity',{'maximum_distance_m':.003}),
    ('gmsh_options',{'Mesh.SecondOrderLinear':0}),('eligibility',{'maximum_nodes':64001}),
    ('diagnostic_limits',{'maximum_elements':100001}),('output_limits',{'maximum_total_bytes':64*1024**2}),
    ('solver_admission',{'authorized':True})])
def test_any_unreviewed_setting_fails_before_mesher(tmp_path,key,value):
    changed=copy.deepcopy(CONFIG);changed[key].update(value)
    with pytest.raises(ValueError,match='exact declared'):
        v.assess_candidate(None,None,None,tmp_path/'no',distance_factory=None,config=changed)
    assert not (tmp_path/'no').exists()


def test_actual_shared_generation_flow_floor_last_and_charge_before_single_generate():
    api,events,fields=fake_gmsh()
    X,E,origin=v.generate_one(api,tet()[:4],v.BASE.FACES,CONFIG,lambda:events.append(('charge',)))
    generate=next(x for x in events if x[0]=='generate')
    assert generate[2]['Mesh.MeshSizeMin']==.003 and generate[2]['Mesh.MeshSizeMax']==.024
    assert generate[2]['Mesh.SecondOrderLinear']==1 and generate[2]['Mesh.MeshSizeFromCurvature']==24
    classification=next(x for x in events if x[0]=='classify')
    assert classification[1]==(np.pi,True,True,np.pi) and classification[2]['Mesh.MeshSizeFromCurvature']==24
    floor=[x[2] for x in events if x[:2]==('option','Mesh.MeshSizeMin')]
    assert floor==[.012,.003]
    i=events.index(generate);assert events[i-1]==('charge',)
    assert sum(x[0]=='generate' for x in events)==1
    assert fields[(32,'SizeMin')]==.012 and fields[(32,'SizeMax')]==.024
    assert fields[(31,'Sampling')]==100
    np.testing.assert_array_equal(X,tet());np.testing.assert_array_equal(E,[range(10)])
    assert origin['gmsh_element_ids']==[91]


def test_shared_default_after_new_generation_remains_12mm_and_curvature_off():
    api,events,_=fake_gmsh()
    old.generate_one(api,tet()[:4],v.BASE.FACES,old.declaration(),lambda:None)
    generate=next(x for x in events if x[0]=='generate')
    assert generate[2]['Mesh.MeshSizeMin']==.012 and generate[2]['Mesh.MeshSizeFromCurvature']==0
    assert caller.execution_spec().file_bytes==4*1024**2
    assert caller.execution_spec().output_bytes==8*1024**2
    assert caller.execution_spec().entrypoint=='scripts/mechanics_patient_mesh_candidate_run.py'


@pytest.mark.parametrize('distance,passes',[(0.,True),(.003,False)])
def test_assessment_runs_unchanged_gates_keeps_complete_packet_and_no_solver(tmp_path,distance,passes):
    api,events,_=fake_gmsh()
    r=v.assess_candidate(api,tet()[:4],v.BASE.FACES,tmp_path/'attempt',
        distance_factory=lambda *_:lambda q:np.full(len(q),distance))
    assert (r['status']=='geometry_candidate_passed_no_solver_authorization') is passes
    assert r['native_generation_calls']==1 and r['solver_calls']==0 and not r['solver_admitted']
    assert r['diagnostic']['complete']
    assert r['source_to_mesh']['maximum_sample_distance_m']==distance
    metadata=json.loads((tmp_path/'attempt/diagnostic/manifest.json').read_text())
    assert metadata['context']['candidate_id']==CONFIG['candidate_id']
    assert not metadata['candidate_accepted']


def test_packet_upper_bound_includes_headers_ids_metadata_without_allocating_maximum():
    limits=CONFIG['diagnostic_limits'];N=limits['maximum_nodes'];E=limits['maximum_elements']
    # Exact NPY v1 header lengths from the authoritative serializer, using a
    # zero-stride descriptor only; no 11 MB allocation or packet write needed.
    arrays=[np.broadcast_to(np.float64(0),(N,3)),np.broadcast_to(np.int64(0),(E,10)),
            np.broadcast_to(np.int64(0),(N,)),np.broadcast_to(np.int64(0),(E,))]
    sizes=[len(old.array_header(a))+a.nbytes for a in arrays]
    assert sum(sizes)+limits['maximum_metadata_bytes']==11_368_704
    assert sum(sizes)+limits['maximum_metadata_bytes']<16*1024**2
    assert max(sizes)==8_000_128 and max(sizes)<run.SPEC.file_bytes


def test_over_eligibility_still_saves_complete_available_packet_then_rejects(tmp_path,monkeypatch):
    X=np.zeros((64001,3));X[:10]=tet();E=np.arange(10,dtype=np.int64).reshape(1,10)
    origin={'gmsh_node_ids':list(range(1,len(X)+1)),'gmsh_element_ids':[1],
            'gmsh_to_febio_permutation':list(range(10)),'discrete_patches':1}
    def generate(*args):args[-1]();return X,E,origin
    monkeypatch.setattr(v,'generate_one',generate)
    monkeypatch.setattr(v.BASE,'validate_tet10',lambda *_:pytest.fail('No quality check after count refusal'))
    r=v.assess_candidate(None,tet()[:4],v.BASE.FACES,tmp_path/'attempt',distance_factory=None)
    assert r['status']=='failed_or_incomplete' and r['diagnostic']['complete']
    assert r['returned_nodes']==64001 and r['native_generation_calls']==1
    assert np.load(tmp_path/'attempt/diagnostic/nodes_m.npy',allow_pickle=False).shape==(64001,3)


def test_explicit_output_limits_do_not_mutate_defaults(tmp_path):
    (tmp_path/'large').write_bytes(b'x'*(4*1024**2+1))
    assert caller.output_usage(tmp_path,spec=run.SPEC)['bytes']==4*1024**2+1
    with pytest.raises(ValueError):caller.output_usage(tmp_path)
    assert caller.FILE_BYTES==4*1024**2


def test_supervisor_forwards_exact_caps_without_leaking_output_observer(tmp_path):
    observed=[]
    def rss(*a,**k):return 1,[]
    runtime=SimpleNamespace(process_group_rss=rss)
    def supervise(command,output,**kw):
        observed.append(kw);runtime.process_group_rss(1,timeout_seconds=.1)
        return {'status':'completed'}
    runtime.supervise=supervise
    receipt,disk=caller.supervised(runtime,['no-native'],tmp_path,{},CONFIG['caps'],spec=run.SPEC)
    assert observed[0]['seconds']==180 and observed[0]['rss_bytes']==3*1024**3
    assert disk['aggregate_bytes_cap']==32*1024**2 and disk['per_file_bytes_cap']==8*1024**2
    assert runtime.process_group_rss is rss


def test_explicit_entrypoint_and_spec_forwarded_to_worker_not_old_cli(tmp_path,monkeypatch):
    inputs={'opaque':'hash'};seen=[]
    monkeypatch.setattr(caller,'released',lambda *a,**kw:(CONFIG,None,None,inputs))
    monkeypatch.setattr(caller,'unchanged',lambda x:{k:True for k in x})
    environment={key:'1' for key in ['OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}
    environment['OMP_DYNAMIC']='FALSE'
    runtime=SimpleNamespace(private_environment=lambda _:dict(environment),declaration=lambda:{})
    monkeypatch.setattr(caller,'module',lambda *_:runtime)
    def supervised(*args,**kwargs):
        seen.append((args,kwargs));return {'status':'failed_or_incomplete'},{'error':None}
    monkeypatch.setattr(caller,'supervised',supervised)
    assert caller.launch('not-read',tmp_path/'attempt',spec=run.SPEC)==1
    args,kwargs=seen[0]
    assert args[1][2]==str(ROOT/run.SPEC.entrypoint) and args[1][3]=='worker'
    assert args[3]['OMP_DYNAMIC']=='FALSE' and args[3]['VTK_SMP_IMPLEMENTATION_TYPE']=='Sequential'
    assert kwargs['spec']==run.SPEC
    before=(tmp_path/'attempt/acceptance.json').read_bytes()
    with pytest.raises(FileExistsError):caller.launch('not-read',tmp_path/'attempt',spec=run.SPEC)
    assert (tmp_path/'attempt/acceptance.json').read_bytes()==before


def temporary_release(tmp_path,monkeypatch):
    archive=tmp_path/'archive';repository=tmp_path/'repository';repository.mkdir();archive.mkdir()
    attempt=tmp_path/'attempt';config=copy.deepcopy(CONFIG)
    surface=repository/config['source_surface_binding']['path'];surface.parent.mkdir(parents=True)
    surface.write_bytes(b'OPAQUE ANALYTIC TEST TOKEN, NEVER DECODED')
    # Fixed production config pins are exercised by the previous controls; here
    # a validator spy accepts only this exact synthetic metadata substitution.
    config['source_surface_binding']['sha256']=caller.sha(surface)
    evidence=repository/'evidence.json';evidence.write_text('{"mock":true}')
    config['evidence_bindings']=[{'path':'evidence.json','sha256':caller.sha(evidence)}]
    checks=[]
    def validate(value):assert value==config;checks.append('validated')
    spec=run.SPEC._replace(validate_config=validate)
    first={'surface_fidelity':config['surface_fidelity'],'quality':config['quality'],'input_bindings':[]}
    expected={}
    for name in spec.closure:
        p=archive/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(first if name==caller.V1 else config if name==spec.candidate_manifest else {'source':name}))
        expected[name]=p.read_bytes()
    contexts=[]
    for role in spec.context_roles:
        p=repository/(role+'.json');p.write_text(json.dumps({'status':'failed_or_incomplete',
            'output_sha256':{'native-surface.npz':caller.sha(surface)}} if role=='prior_mesh_worker' else {'mock':True}))
        contexts.append({'role':role,'path':str(p),'sha256':caller.sha(p)})
    release={'schema':spec.version,'authorized':True,'source_directory':str(archive),
        'repository_directory':str(repository),'source_commit':'1'*40,'attempt_directory':str(attempt),
        'clinical_validation':False,'anatomical_registration_accepted':False,'solver_authorized':False,
        'reuse_saved_native_surface_only':True,'context_bindings':contexts,
        'source_sha256':{name:caller.sha(archive/name) for name in spec.closure}}
    path=tmp_path/'release.json';path.write_text(json.dumps(release))
    monkeypatch.setattr(caller,'ROOT',archive)
    monkeypatch.setattr(caller.subprocess,'run',lambda args,**kw:SimpleNamespace(stdout=expected[args[-1].split(':',1)[1]]))
    return path,attempt,spec,evidence,checks


def test_exact_ten_file_archive_and_preceding_evidence_rehashed_before_worker(tmp_path,monkeypatch):
    path,attempt,spec,evidence,checks=temporary_release(tmp_path,monkeypatch)
    config,first,surface,bound=caller.released(path,attempt,spec=spec)
    assert checks==['validated'] and len(spec.closure)==10
    assert str(evidence) in bound and str(surface) in bound and not attempt.exists()
    evidence.write_text('changed')
    with pytest.raises(ValueError,match='input/runtime/context'):
        caller.released(path,attempt,spec=spec)


def test_new_spec_requires_separate_authorization_before_input_access(tmp_path,monkeypatch):
    release=tmp_path/'release.json';release.write_text(json.dumps({'schema':run.SPEC.version,'authorized':False}))
    monkeypatch.setattr(caller,'sha',lambda *_:pytest.fail('No hash before release'))
    with pytest.raises(ValueError,match='Separate'):
        caller.released(release,tmp_path/'attempt',spec=run.SPEC)


def test_new_per_file_bound_rejects_without_reading_payload(tmp_path):
    with (tmp_path/'sparse').open('wb') as stream:stream.truncate(run.SPEC.file_bytes+1)
    with pytest.raises(ValueError,match='cap exceeded'):
        caller.output_usage(tmp_path,spec=run.SPEC)
