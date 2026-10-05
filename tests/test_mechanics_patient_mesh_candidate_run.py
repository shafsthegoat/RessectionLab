"""Launcher controls with temporary text files and mocked supervisors only."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_caller',ROOT/'scripts/mechanics_patient_mesh_candidate_run.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def fake_release(tmp_path,monkeypatch):
    archive=tmp_path/'archive';repo=tmp_path/'repository';repo.mkdir();archive.mkdir()
    attempt=tmp_path/'attempt'
    source=repo/'surface.npz';source.write_bytes(b'ANALYTIC TEXT BINDING ONLY - NEVER DECODE')
    dependency=repo/'runtime';dependency.write_text('mock')
    first={'surface_fidelity':{'maximum_distance_m':.002},'quality':{'minimum_mean_ratio':.05},
           'input_bindings':[{'path':str(dependency),'sha256':v.sha(dependency)}]}
    candidate={'caps':{'maximum_generations':1,'retries':0},'surface_fidelity':first['surface_fidelity'],
               'quality':first['quality'],'source_surface_binding':{'path':'surface.npz','sha256':v.sha(source)}}
    hashes={}
    for name in v.CLOSURE:
        p=archive/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(first if name==v.V1 else candidate if name==v.V2 else {'mock':name}))
        hashes[name]=v.sha(p)
    contexts=[]
    for role in sorted(v.CONTEXT_ROLES):
        p=repo/(role+'.json')
        p.write_text(json.dumps({'status':'failed_or_incomplete','output_sha256':{'native-surface.npz':v.sha(source)}} if role=='prior_mesh_worker' else {'scope':'mock'}))
        contexts.append({'role':role,'path':str(p),'sha256':v.sha(p)})
    release={'schema':v.VERSION,'authorized':True,'source_directory':str(archive),'repository_directory':str(repo),
             'source_commit':'a'*40,'attempt_directory':str(attempt),'clinical_validation':False,
             'anatomical_registration_accepted':False,'solver_authorized':False,'reuse_saved_native_surface_only':True,
             'source_sha256':hashes,'context_bindings':contexts}
    path=tmp_path/'release.json';path.write_text(json.dumps(release))
    monkeypatch.setattr(v,'ROOT',archive)
    expected={name:(archive/name).read_bytes() for name in v.CLOSURE}
    monkeypatch.setattr(v.subprocess,'run',lambda argv,**kwargs:SimpleNamespace(stdout=expected[argv[-1].split(':',1)[1]]))
    return path,attempt,release,source


def test_exact_source_archive_and_saved_surface_are_bound_without_decoding(tmp_path,monkeypatch):
    path,attempt,release,source=fake_release(tmp_path,monkeypatch)
    candidate,first,surface,bindings=v.released(path,attempt)
    assert surface==source and str(source) in bindings
    assert len(release['source_sha256'])==7
    assert all(v.unchanged(bindings).values())
    assert not attempt.exists()


@pytest.mark.parametrize('damage',['archive','surface','context','commit','path','solver','gates'])
def test_release_changes_fail_before_worker(tmp_path,monkeypatch,damage):
    path,attempt,r,source=fake_release(tmp_path,monkeypatch)
    if damage=='archive':(v.ROOT/'scripts/mechanics_patient_mesh.py').write_text('changed')
    if damage=='surface':source.write_text('changed')
    if damage=='context':Path(r['context_bindings'][0]['path']).write_text('changed')
    if damage=='commit':r['source_commit']='a49e70a'
    if damage=='path':attempt=tmp_path/'another'
    if damage=='solver':r['solver_authorized']=True
    if damage=='gates':r['source_sha256'].pop(v.V1)
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):v.released(path,attempt)


def test_no_release_fails_before_any_input_hash(tmp_path,monkeypatch):
    path=tmp_path/'release.json';path.write_text(json.dumps({'schema':v.VERSION,'authorized':False}))
    monkeypatch.setattr(v,'sha',lambda _:pytest.fail('No input access without release'))
    with pytest.raises(ValueError,match='Separate'):v.released(path,tmp_path/'attempt')


@pytest.mark.parametrize('kind',['bytes','files','depth','symlink'])
def test_output_guards_are_bounded_and_do_not_follow_links(tmp_path,monkeypatch,kind):
    folder=tmp_path/'output';folder.mkdir()
    if kind=='bytes':
        monkeypatch.setattr(v,'OUTPUT_BYTES',4);(folder/'one').write_bytes(b'12345')
    if kind=='files':
        monkeypatch.setattr(v,'OUTPUT_FILES',1)
        (folder/'one').touch();(folder/'two').touch()
    if kind=='depth':(folder/'a/b/c/d/e').mkdir(parents=True)
    if kind=='symlink':(folder/'escape').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):v.output_usage(folder)


def test_output_guard_uses_original_rss_observer_and_restores_it(tmp_path):
    calls=[]
    def observer(pid,*,timeout_seconds):calls.append((pid,timeout_seconds));return 1,[]
    runtime=SimpleNamespace(process_group_rss=observer)
    def supervise(*args,**kwargs):
        assert runtime.process_group_rss is not observer
        runtime.process_group_rss(23,timeout_seconds=.2)
        return {'status':'completed'}
    runtime.supervise=supervise
    receipt,guard=v.supervised(runtime,['mock'],tmp_path,{}, {'aggregate_seconds':180,'process_group_rss_bytes':3*1024**3})
    assert calls==[(23,.2)] and runtime.process_group_rss is observer
    assert guard['observations']==1 and guard['error'] is None


def test_normal_atomic_publication_does_not_raise_false_output_failure(tmp_path,monkeypatch):
    temporary=tmp_path/'result.tmp';temporary.write_text('complete')
    original=Path.stat; published_once=[]
    def published(path,*args,**kwargs):
        if path==temporary and kwargs.get('follow_symlinks') is False and not published_once:
            published_once.append(True)
            temporary.rename(tmp_path/'result.json')
            raise FileNotFoundError(2,'Atomic publication between listing and stat')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',published)
    assert v.output_usage(tmp_path)['files']==0
    assert v.output_usage(tmp_path)['bytes']==8


def test_output_failure_reaches_supervisor_observer_kill_path(tmp_path,monkeypatch):
    (tmp_path/'over').write_bytes(b'12345');monkeypatch.setattr(v,'OUTPUT_BYTES',4)
    observer=lambda *args,**kwargs:pytest.fail('No RSS observation after output rejection')
    runtime=SimpleNamespace(process_group_rss=observer)
    def supervise(*args,**kwargs):
        try:runtime.process_group_rss(23,timeout_seconds=.2)
        except ValueError as error:return {'status':'failed_or_incomplete','kill_reason':'supervision_exception','error':str(error)}
        pytest.fail('Observer must reject')
    runtime.supervise=supervise
    receipt,guard=v.supervised(runtime,['mock'],tmp_path,{}, {'aggregate_seconds':180,'process_group_rss_bytes':3*1024**3})
    assert receipt['kill_reason']=='supervision_exception' and guard['error']
    assert runtime.process_group_rss is observer


def test_repeated_internal_worker_preserves_receipt_before_release_or_limits(tmp_path,monkeypatch):
    receipt=tmp_path/'worker.json';receipt.write_text('original terminal record')
    monkeypatch.setattr(v,'released',lambda *_:pytest.fail('No repeated release read'))
    monkeypatch.setattr(v.resource,'setrlimit',lambda *_:pytest.fail('No repeated process limit mutation'))
    with pytest.raises(FileExistsError):v.worker('unused',tmp_path)
    assert receipt.read_text()=='original terminal record'


def test_failed_preflight_records_failure_and_never_launches(tmp_path,monkeypatch):
    def fail(*_):raise ValueError('Unreleased')
    monkeypatch.setattr(v,'released',fail)
    monkeypatch.setattr(v,'module',lambda *_:pytest.fail('No native/supervisor import'))
    assert v.launch('unused',tmp_path/'attempt')==1
    record=json.loads((tmp_path/'attempt/acceptance.json').read_text())
    assert not record['worker_started'] and not record['solver_authorized']
    with pytest.raises(FileExistsError):v.launch('unused',tmp_path/'attempt')
