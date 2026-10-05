"""Independent caller controls: metadata and mocks; no native APIs or source arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_caller_review',ROOT/'scripts/mechanics_patient_mesh_candidate_run.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def fake_launch(tmp_path,monkeypatch,*,tight):
    marker=tmp_path/'input.json';marker.write_text('{}')
    bound={str(marker):v.sha(marker)}
    caps={'aggregate_seconds':180,'process_group_rss_bytes':3*2**30}
    monkeypatch.setattr(v,'released',lambda *_:({'caps':caps},None,None,bound))
    env={k:'1' for k in ['OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}
    env['OMP_DYNAMIC']='FALSE'
    runtime=SimpleNamespace(declaration=lambda:{},private_environment=lambda _:dict(env))
    monkeypatch.setattr(v,'module',lambda *_:runtime)
    if tight=='bytes':monkeypatch.setattr(v,'OUTPUT_BYTES',4096)
    else:monkeypatch.setattr(v,'OUTPUT_FILES',3)
    def supervise(_runtime,command,output,environment,given_caps):
        assert given_caps==caps and command[3]=='worker'
        target=output/'candidate/diagnostic/manifest.json';target.parent.mkdir(parents=True);target.write_text('{}')
        record={'status':'completed_geometry_only','candidate_status':'geometry_candidate_passed_no_solver_authorization',
            'native_generation_calls':1,'solver_calls':0,'candidate_output_sha256':{'candidate/diagnostic/manifest.json':v.sha(target)}}
        (output/'worker.json').write_text(json.dumps(record))
        pad=output/'padding.log';pad.write_bytes(b'x')
        if tight=='bytes':
            size=v.OUTPUT_BYTES-v.output_usage(output)['bytes']-16
            with pad.open('ab') as stream:stream.write(b'x'*size)
        assert v.output_usage(output)['bytes']<=v.OUTPUT_BYTES
        return {'status':'completed'}, {'error':None}
    monkeypatch.setattr(v,'supervised',supervise)
    return tmp_path/'attempt'


@pytest.mark.parametrize('tight',['bytes','files'])
def test_final_parent_receipt_cannot_turn_an_overcap_attempt_into_success(tmp_path,monkeypatch,tight):
    output=fake_launch(tmp_path,monkeypatch,tight=tight)
    assert v.launch('mock-release',output)==1
    result=json.loads((output/'acceptance.json').read_text())
    assert result['status']=='failed_or_incomplete'


def test_output_guard_rejects_symlinks_without_reading_their_targets(tmp_path):
    (tmp_path/'link').symlink_to(tmp_path/'unopened-target')
    with pytest.raises(ValueError,match='symlink'):v.output_usage(tmp_path)


def test_disk_observer_failure_reaches_group_supervisor_and_restores_observer(tmp_path,monkeypatch):
    called=[]
    def original(_pgid,*,timeout_seconds):called.append(1);return 123,[]
    runtime=SimpleNamespace(process_group_rss=original)
    def supervisor(_command,_output,**kwargs):
        assert kwargs['seconds']==180 and kwargs['rss_bytes']==3*2**30
        with pytest.raises(ValueError,match='forced disk cap'):
            runtime.process_group_rss(5,timeout_seconds=1)
        return {'status':'failed_or_incomplete','kill_reason':'supervision_exception'}
    runtime.supervise=supervisor
    def fail(_):raise ValueError('forced disk cap')
    monkeypatch.setattr(v,'output_usage',fail)
    result,audit=v.supervised(runtime,[],tmp_path,{}, {'aggregate_seconds':180,'process_group_rss_bytes':3*2**30})
    assert result['status']=='failed_or_incomplete' and audit['error']=='forced disk cap'
    assert runtime.process_group_rss is original and not called


def release_fixture(tmp_path,monkeypatch):
    archive=tmp_path/'archive';repository=tmp_path/'repository';archive.mkdir();repository.mkdir()
    monkeypatch.setattr(v,'ROOT',archive)
    source_surface=repository/'saved-surface.fixture';source_surface.write_text('ANALYTIC METADATA BYTES ONLY')
    first={'input_bindings':[], 'surface_fidelity':{'maximum_distance_m':.002,'maximum_relative_volume_error':.03},'quality':{'minimum_mean_ratio':.05}}
    candidate={'caps':{'maximum_generations':1,'retries':0},'source_surface_binding':{'path':'saved-surface.fixture','sha256':v.sha(source_surface)},
               'surface_fidelity':first['surface_fidelity'],'quality':first['quality']}
    for name in v.CLOSURE:
        path=archive/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(first if name==v.V1 else candidate if name==v.V2 else {'fixture_only':name}))
    contexts=[]
    for role in v.CONTEXT_ROLES:
        path=tmp_path/(role+'.json')
        path.write_text(json.dumps({'status':'failed_or_incomplete','output_sha256':{'native-surface.npz':v.sha(source_surface)}} if role=='prior_mesh_worker' else {'fixture_only':role}))
        contexts.append({'role':role,'path':str(path),'sha256':v.sha(path)})
    release={'schema':v.VERSION,'authorized':True,'source_directory':str(archive),'repository_directory':str(repository),
             'source_commit':'b'*40,'attempt_directory':str(tmp_path/'attempt'),'clinical_validation':False,
             'anatomical_registration_accepted':False,'solver_authorized':False,'reuse_saved_native_surface_only':True,
             'source_sha256':{name:v.sha(archive/name) for name in v.CLOSURE},'context_bindings':contexts}
    path=tmp_path/'release.json'
    monkeypatch.setattr(v.subprocess,'run',lambda command,**_:SimpleNamespace(stdout=(archive/command[-1].split(':',1)[1]).read_bytes()))
    return path,release,archive,source_surface


@pytest.mark.parametrize('corrupt',['none','archive','context','ancestry','different_surface','solver','reextract','missing_closure'])
def test_exact_archive_context_and_saved_surface_ancestry_are_required(tmp_path,monkeypatch,corrupt):
    path,release,archive,surface=release_fixture(tmp_path,monkeypatch)
    if corrupt=='archive':(archive/'scripts/mechanics_patient_mesh_candidate.py').write_text('modified')
    elif corrupt=='context':Path(release['context_bindings'][0]['path']).write_text('modified')
    elif corrupt=='ancestry':
        item=next(x for x in release['context_bindings'] if x['role']=='prior_mesh_worker')
        target=Path(item['path']);target.write_text(json.dumps({'status':'completed_geometry_only','output_sha256':{'native-surface.npz':v.sha(surface)}}));item['sha256']=v.sha(target)
    elif corrupt=='different_surface':surface.write_text('changed')
    elif corrupt=='solver':release['solver_authorized']=True
    elif corrupt=='reextract':release['reuse_saved_native_surface_only']=False
    elif corrupt=='missing_closure':release['source_sha256'].pop(v.V1)
    path.write_text(json.dumps(release))
    if corrupt=='none':
        _,_,returned,bound=v.released(path,tmp_path/'attempt')
        assert returned==surface and len(bound)==14
    else:
        with pytest.raises(ValueError):v.released(path,tmp_path/'attempt')


@pytest.mark.parametrize('case',['success','unexpected_array','solver_admission'])
def test_worker_checks_saved_array_keys_and_hard_file_limit_without_native_execution(tmp_path,monkeypatch,case):
    import numpy as np
    import sys
    calls=[];output=tmp_path/'attempt';output.mkdir()
    marker=tmp_path/'bound.json';marker.write_text('{}')
    surface=tmp_path/'surface.fixture'
    first={'package_versions':{},'gmsh_runtime':{'module_path':'/mock/gmsh.py','library_path':'/mock/libgmsh.dylib','version':'4.15.2'}}
    monkeypatch.setattr(v,'released',lambda *_:({},first,surface,{str(marker):v.sha(marker)}))
    def limit(kind,values):
        assert kind==v.resource.RLIMIT_FSIZE and values==(4*2**20,4*2**20);calls.append('hard_limit')
    monkeypatch.setattr(v.resource,'setrlimit',limit)
    class Archive:
        files=['vertices_m','triangles']+(['unexpected'] if case=='unexpected_array' else [])
        def __enter__(self):return self
        def __exit__(self,*_):calls.append('archive_closed')
        def __getitem__(self,key):return 'analytic-'+key
    def load_saved(path,**kwargs):
        assert path==surface and kwargs=={'allow_pickle':False} and calls==['hard_limit']
        calls.append('saved_surface_only');return Archive()
    monkeypatch.setattr(np,'load',load_saved)
    def assessed(_gmsh,vertices,faces,directory,**kwargs):
        assert vertices=='analytic-vertices_m' and faces=='analytic-triangles'
        assert 'archive_closed' in calls
        calls.append('assess_once');directory.mkdir()
        (directory/'result.json').write_text('{}')
        return {'status':'geometry_candidate_passed_no_solver_authorization','native_generation_calls':1,
                'solver_calls':0,'solver_admitted':case=='solver_admission'}
    helper=SimpleNamespace(assess_candidate=assessed,BASE=SimpleNamespace(vtk_distance_function=lambda *_:None))
    gmsh=SimpleNamespace(__file__='/mock/gmsh.py',lib=SimpleNamespace(_name='/mock/libgmsh.dylib'),__version__='4.15.2',
        initialize=lambda *args,**kwargs:calls.append('mock_initialize'),finalize=lambda:calls.append('mock_finalize'))
    def fake_module(path,name):return helper if name=='released_graded_mesh' else gmsh
    monkeypatch.setattr(v,'module',fake_module)
    monkeypatch.delitem(sys.modules,'gmsh',raising=False)
    monkeypatch.setitem(sys.modules,'vtkmodules.vtkCommonCore',SimpleNamespace(vtkSMPTools=SimpleNamespace(SetBackend=lambda _:True,GetBackend=lambda:'Sequential')))
    code=v.worker('mock-release',output)
    result=json.loads((output/'worker.json').read_text())
    assert result['per_file_hard_limit_bytes']==4*2**20 and result['solver_calls']==0
    assert result['MRI_reextracted'] is False and result['B_or_V_access'] is False
    if case=='success':assert code==0 and calls.count('assess_once')==1
    else:assert code==1 and result['status']=='failed_or_incomplete'
    if case=='unexpected_array':
        assert 'mock_initialize' not in calls and 'assess_once' not in calls
        assert result['source_arrays_reused'] is False
    else:assert calls.count('mock_finalize')==1
