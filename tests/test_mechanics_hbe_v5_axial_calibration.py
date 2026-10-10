"""Generated controls only: the real HBE archive and all patient inputs are forbidden."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parents[2] if STAGE.name=='stage' else STAGE
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('axial_core_test',STAGE/'scripts/mechanics_hbe_v5_axial_calibration.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
spec2=importlib.util.spec_from_file_location('axial_runner_test',STAGE/'scripts/mechanics_hbe_v5_axial_calibration_experiment.py')
runner=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(runner)
from scripts import mechanics_hbe_evaluation as evaluation
from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_branch_calibration_v4 as coordinates


@pytest.fixture(autouse=True)
def blocked_sources(monkeypatch):
    original=Path.open
    def opened(path,*args,**kwargs):
        absolute=path.resolve()
        if absolute.is_relative_to(ROOT/'data') or absolute.is_relative_to(ROOT/'outputs'):
            raise AssertionError('real measured/patient/native source access forbidden')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)


def curves(scale=2.5):
    observed,reference,refs={},{},{}
    for b,sign in (('compression',-1),('tension',1)):
        x=[sign*.0007*i/30 for i in range(1,31)]
        r=[sign*.0007*i/120 for i in range(121)]
        body=('displacement,force\n'+''.join(f'{a:.17g},{scale*50*a:.17g}\n' for a in x)).encode()
        identity=hashlib.sha256(body).hexdigest()
        observed[b]=evaluation.curve(b,x,[scale*50*a for a in x],source_sha256=identity)
        reference[b]=evaluation.curve(b,r,[50*a for a in r])
        refs[b]={'member':{'sha256':identity},'coordinate_sha256':coordinates.coordinate_hash(x)}
    return observed,reference,{'references':refs}


def test_known_scale_and_descriptive_terminal():
    obs,refs,decl=curves();calls=[]
    fit,metrics=core.fit_terminal(obs,refs,decl,on_fit=lambda:calls.append(1))
    assert fit['scale']==pytest.approx(2.5) and calls==[1]
    assert fit['offset_or_sign_fitted'] is False and fit['held_out_values_used'] is False
    assert metrics['physical_validation_pass'] is None and metrics['independent_donor_validation'] is False
    assert metrics['equal_branch_RMSE']<1e-15


@pytest.mark.parametrize('mutation',['hash','rows','coordinate','endpoint','branch'])
def test_reject_before_fit(mutation):
    obs,refs,decl=curves();calls=[];c=obs['compression']
    if mutation=='hash':obs['compression']=c._replace(source_sha256='0'*64)
    elif mutation=='rows':obs['compression']=c._replace(coordinate=c.coordinate[:-1],response=c.response[:-1])
    elif mutation=='coordinate':decl['references']['compression']['coordinate_sha256']='0'*64
    elif mutation=='endpoint':refs['compression']=refs['compression']._replace(coordinate=refs['compression'].coordinate[:-1],response=refs['compression'].response[:-1])
    else:obs['compression']=c._replace(branch='tension')
    with pytest.raises(ValueError):core.fit_terminal(obs,refs,decl,on_fit=lambda:calls.append(1))
    assert calls==[]


@pytest.mark.parametrize('scale',[0.,-2.])
def test_nonpositive_scale_refuses_without_clipping(scale):
    obs,refs,decl=curves(scale);calls=[]
    with pytest.raises(ValueError,match='Positive'):core.fit_terminal(obs,refs,decl,on_fit=lambda:calls.append(1))
    assert calls==[1]


def test_unidentified_reference_refuses():
    obs,refs,decl=curves()
    refs={b:c._replace(response=tuple(0. for _ in c.response)) for b,c in refs.items()}
    with pytest.raises(ValueError,match='denominator'):core.fit_terminal(obs,refs,decl)


def fixture_study(tmp_path,monkeypatch,*,bad_header=False,authorized=True):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(core.canonical(value))
        return {'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    schema={'delimiter':',','header':['displacement','force'],'coordinate_column':0,
            'response_column':1,'coordinate_unit':'m','response_unit':'N'}
    members=[]
    archive_path=tmp_path/'generated.zip'
    with zipfile.ZipFile(archive_path,'w') as archive:
        for b,s in (('compression',-1),('tension',1)):
            head='wrong,force' if bad_header and b=='compression' else 'displacement,force'
            body=(head+'\n'+''.join(f'{s*i*.00002},{s*i*.001}\n' for i in range(1,31))).encode()
            name=f'HBE_01/HBE_01_03/{b}_c3.csv';archive.writestr(name,body)
            info=archive.getinfo(name);members.append({'path':name,'bytes':len(body),'crc32':f'{info.CRC:08x}'})
        archive.writestr('HBE_01/HBE_01_03/torsion_l1_c3_neg.csv','protected-sentinel')
    roles={'source':{'archive_path':'generated.zip','archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                     'archive_bytes':archive_path.stat().st_size},'calibration':{'members':members},
           'held_out_validation':{'members':[{'path':'HBE_01/HBE_01_03/torsion_l1_c3_neg.csv'}]}}
    role_binding=save('roles.json',roles);protocol_binding=save('protocol.json',{'roles':role_binding})
    declaration={'schema':'hbe-v5-axial-fit-declaration-v1','evidence':{'protocol':protocol_binding},
        'caps':deepcopy(core.CAPS),'authorized_held_out_members':[],'native_calls':0,'fit_calls':1,
        'csv_schemas':{b:deepcopy(schema) for b in core.AXIAL}}
    db=save(core.DECLARATION,declaration);monkeypatch.setattr(core,'DECLARATION_SHA256',db['sha256'])
    release={'schema':'hbe-v5-axial-fit-release-v1','execution_released':authorized,'declaration':db,
        'output_directory':core.OUTPUT,'caps':deepcopy(core.CAPS),'source_commit':'0'*40,'source_bindings':{},
        'runtime':{},'independent_source_review':None}
    normalized={**release,'execution_released':False,'source_commit':None,'independent_source_review':None}
    review=save('review.json',{'schema':'hbe-v5-axial-fit-source-review-v1','decision':'GO',
                            'release_candidate_sha256':core.sha(core.canonical(normalized))})
    release['independent_source_review']=review;rb=save('release.json',release)
    study=core.make_study(tmp_path,rb,release,declaration,recheck=lambda:None)
    return study,declaration,release,rb,save


def test_existing_reader_reads_exact_pair_once(tmp_path,monkeypatch):
    study,decl,*_=fixture_study(tmp_path,monkeypatch);reads=[];original=zipfile.ZipFile.read
    def read(z,info,*args,**kwargs):
        reads.append(info.filename if hasattr(info,'filename') else info)
        assert 'torsion' not in reads[-1];return original(z,info,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'read',read)
    assert set(study.read_calibration(decl['csv_schemas']))==set(core.AXIAL)
    assert len(reads)==2
    with pytest.raises(ValueError,match='consumed'):study.read_calibration(decl['csv_schemas'])
    events=study._events();assert [x['phase'] for x in events]==['calibration_attempt','calibration_completed']
    assert len(reads)==2


def test_partial_exposure_consumes_attempt(tmp_path,monkeypatch):
    study,decl,*_=fixture_study(tmp_path,monkeypatch,bad_header=True)
    with pytest.raises(ValueError,match='header'):study.read_calibration(decl['csv_schemas'])
    assert [x['phase'] for x in study._events()]==['calibration_attempt']
    with pytest.raises(ValueError,match='consumed'):study.read_calibration(decl['csv_schemas'])


def test_false_release_never_opens_archive(tmp_path,monkeypatch):
    study,decl,*_=fixture_study(tmp_path,monkeypatch,authorized=False)
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**k:pytest.fail('archive opened'))
    with pytest.raises(ValueError,match='root release'):study.read_calibration(decl['csv_schemas'])
    assert not study.ledger_path.exists()


@pytest.mark.parametrize('method',['freeze_predictions','evaluate_held_out','direct_selected'])
def test_all_torque_paths_refuse(tmp_path,monkeypatch,method):
    study,decl,*_=fixture_study(tmp_path,monkeypatch)
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**k:pytest.fail('archive opened'))
    with pytest.raises(ValueError):
        if method=='direct_selected':
            study._read_selected(study.roles,study.roles['held_out_validation']['members'],('torsion_neg',),{},'held_out')
        else:getattr(study,method)()
    assert not study.ledger_path.exists()


def test_review_label_without_exact_candidate_refuses(tmp_path,monkeypatch):
    study,decl,release,rb,save=fixture_study(tmp_path,monkeypatch)
    release['independent_source_review']=save('fake-review.json',{'schema':'hbe-v5-axial-fit-source-review-v1',
         'decision':'GO','release_candidate_sha256':'0'*64})
    rb=save('release.json',release)
    with pytest.raises(ValueError,match='exact candidate'):core.check_release(tmp_path,rb,executing=True)


def test_changed_release_refuses_before_read(tmp_path,monkeypatch):
    study,decl,release,rb,save=fixture_study(tmp_path,monkeypatch)
    release['caps']['fit_calls']=2;save('release.json',release)
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**k:pytest.fail('archive opened'))
    with pytest.raises(ValueError):study.read_calibration(decl['csv_schemas'])


def test_output_exclusive_and_bound(tmp_path):
    path=tmp_path/'result.json'
    with pytest.raises(ValueError,match='cap'):core.exclusive_json(path,{'long':'x'*1000},maximum=32)
    assert not path.exists()
    core.exclusive_json(path,{'a':1})
    with pytest.raises(FileExistsError):core.exclusive_json(path,{'a':2})


def fake_evidence(tmp_path):
    # Static declared coordinates are public metadata; all response/probe values below are invented.
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    prior=json.loads((ROOT/'manifests/experiments/hbe-01-03-branch-calibration-v4.json').read_text())
    study=json.loads((ROOT/'manifests/experiments/hbe-01-03-branch-calibration-v5.json').read_text())
    def save(name,obj):
        path=tmp_path/name
        path.write_bytes(core.canonical(obj) if not isinstance(obj,bytes) else obj)
        return {'path':name,'sha256':core.sha(path.read_bytes())}
    roles={'calibration':{'members':[{'path':b} for b in core.AXIAL]}}
    bindings={'roles':save('roles.json',roles)}
    bindings['protocol']=save('protocol.json',{'roles':bindings['roles']})
    bindings['v3_state']=save('old-state.json',{'status':'failed_or_incomplete','native_calls':0,'held_out_responses_accessed':False})
    bindings['v3_publication']=save('old-publication.json',{'accepted':False})
    bindings['v3_access_ledger']=save('old-access.jsonl',b'{"phase":"calibration_attempt"}\n{"phase":"calibration_completed"}\n')
    bindings['v4']=save('v4.json',prior);bindings['v5']=save('v5.json',study)
    refs={};manifest={'rows':{}}
    for branch,n in (('compression',36),('tension',24)):
        rid=f'{branch}:N{n}:S120:reference';schedule=v5.schedule(study,prior,rid)
        row={'load_coordinate_full_m':schedule['full_coordinates_m'],'applied_force_N':[50*x for x in schedule['full_coordinates_m']],
             'probe_displacements_m':[[[0.,0.,0.] for j in range(75)] for i in range(121)],
             'criteria_max_ratio':{'generated':0.},'solver':{'passed':True},'full_energy_work':{'passed':True},
             'native_energy_work':{'passed':True},'minimum_sampled_J':1.,'minimum_logged_J':1.,'numerical_passed':True}
        rb=save(branch+'.json',row);manifest['rows'][rid]={'readout':rb}
        refs[branch]={'run_id':rid,'readout':rb,'member':prior['coordinates'][branch]['member']}
    bindings['manifest']=save('manifest.json',manifest)
    comparison={'schema':'hbe-v5-native-numerical-comparison-v1','comparison_manifest':bindings['manifest'],
        'numerical_qualification_passed':True,'native_output_admitted':True,'physical_validation_pass':None,
        'calibration_released':False,'measured_response_accessed':False}
    bindings['comparison']=save('comparison.json',comparison)
    receipt={'status':'completed_numerical_comparison','comparison_sha256':bindings['comparison']['sha256'],
       'cleanup':{'contained':True,'direct_child_reaped':True,'remaining_members':[],'errors':[]}}
    bindings['comparison_receipt']=save('receipt.json',receipt)
    summary={'comparison_sha256':bindings['comparison']['sha256'],'manifest_sha256':bindings['manifest']['sha256'],
      'receipt_sha256':bindings['comparison_receipt']['sha256'],'numerical_qualification_passed':True,'physical_validation_pass':None}
    bindings['independent_summary']=save('summary.json',summary)
    audit={'status':'PASS','files_pre_post_identical':{bindings[k]['path']:{'sha256':bindings[k]['sha256']}
         for k in ('comparison','manifest')},'numerical_qualification_passed':True}
    bindings['independent_audit']=save('audit.json',audit);bindings['independent_report']=save('report.txt',b'Generated review fixture only\n')
    return {'evidence':bindings,'references':refs,'permitted_members':list(core.AXIAL)},save


def test_generated_numerical_admission(tmp_path):
    decl,_=fake_evidence(tmp_path);records,refs=core.admit_evidence(tmp_path,decl)
    assert set(refs)==set(core.AXIAL) and all(len(c.coordinate)==121 for c in refs.values())


@pytest.mark.parametrize('kind',['numerical_fail','provenance','audit','cleanup','reference','ledger'])
def test_bad_admission_refuses_before_member_access(tmp_path,kind,monkeypatch):
    decl,save=fake_evidence(tmp_path)
    def change(key,**changes):
        old=core.read_binding(tmp_path,decl['evidence'][key]);old.update(changes)
        decl['evidence'][key]=save(decl['evidence'][key]['path'],old)
    if kind=='numerical_fail':change('comparison',numerical_qualification_passed=False)
    elif kind=='provenance':change('comparison',native_output_admitted=False)
    elif kind=='audit':change('independent_audit',status='HOLD')
    elif kind=='cleanup':change('comparison_receipt',cleanup={'contained':False})
    elif kind=='reference':decl['references']['compression']['readout']=decl['references']['tension']['readout']
    else:decl['evidence']['v3_access_ledger']=save('old-access.jsonl',b'{"phase":"held_out_attempt"}\n')
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**k:pytest.fail('archive opened'))
    with pytest.raises((ValueError,KeyError)):core.admit_evidence(tmp_path,decl)


@pytest.mark.parametrize('mode',['success','semantic_fail','stage_fail','cleanup_fail','stage_raise'])
def test_parent_checks_semantics_and_cleanup(tmp_path,monkeypatch,mode):
    from scripts import mechanics_hbe_v5_remaining_one_shot as stages
    from launchers import hbe_v5_ordinal9_continuation_v1 as owned
    release={'execution_released':True,'declaration':{'path':'decl.json','sha256':'1'*64},'source_commit':'a'*40}
    rb={'path':'release.json','sha256':'2'*64};calls=[]
    class Owner:
        def __init__(self):calls.append('owned')
        def cleanup(self,*args):
            calls.append('cleanup')
            return {'contained':mode!='cleanup_fail','direct_child_reaped':True,
                    'remaining_members':[],'errors':[],'fallback_used':False,'exit_code':0}
    monkeypatch.setattr(owned,'OwnedStage',Owner)
    monkeypatch.setattr(core,'audit_loaded',lambda *a:None)
    monkeypatch.setattr(core,'verify_sources',lambda *a,**k:None)
    declaration={'evidence':{},'references':{}}
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,declaration))
    monkeypatch.setattr(sys,'pycache_prefix',str(tmp_path/'build/unused-parent-cache'))
    def stage(kind,command,out,record,**kwargs):
        assert kind=='readout' and '-S' in command and '-B' in command
        assert 0<kwargs['wall_cap']<=60 and kwargs['rss_cap']==512*1024**2
        if mode=='stage_raise':raise RuntimeError('generated observer failure')
        fit_sha=core.exclusive_json(out/'fit.json',{'generated':True})
        metric_sha=core.exclusive_json(out/'calibration-metrics.json',{'physical_validation_pass':None})
        state={'status':'failed_axial_calibration_attempt' if mode=='semantic_fail' else core.DONE,
               'fit_calls':1,'native_calls':0,'mesher_calls':0,'held_out_member_reads':0,
               'calibration_responses_accessed':True,'freeze_saved':False,'physical_validation_pass':None,'release':rb,
               'fit':{'path':core.OUTPUT+'/fit.json','sha256':fit_sha},
               'calibration_metrics':{'path':core.OUTPUT+'/calibration-metrics.json','sha256':metric_sha}}
        core.exclusive_json(out/'state.json',state)
        (out/'access.jsonl').write_text('{"phase":"calibration_attempt"}\n{"phase":"calibration_completed"}\n')
        core.exclusive_json(out/'receipt.json',{'generated_supervision':True})
        (out/'readout-console.txt').write_bytes(b'')
        return {'status':'failed_or_incomplete' if mode=='stage_fail' else 'completed_within_caps','exit_code':0}
    monkeypatch.setattr(stages,'supervise_stage',stage)
    result=runner.execute(tmp_path,core,rb,release,declaration)
    assert result['status']==(core.DONE if mode=='success' else 'failed_axial_calibration_attempt')
    assert calls==['owned','cleanup'] and (tmp_path/core.OUTPUT/'terminal.json').exists()
    with pytest.raises(FileExistsError):runner.execute(tmp_path,core,rb,release,declaration)


def test_parent_false_release_does_not_reserve(tmp_path):
    with pytest.raises(ValueError,match='root release'):runner.execute(tmp_path,core,{}, {'execution_released':False},{})
    assert not (tmp_path/core.OUTPUT).exists()


def test_owned_child_eperm_fallback_and_reap(monkeypatch):
    from launchers import hbe_v5_ordinal9_continuation_v1 as owned
    class Child:
        pid=99999999
        live=True
        killed=False
        reaped=False
        def poll(self):return None if self.live else -9
        def kill(self):self.killed=True;self.live=False
        def wait(self,timeout):self.reaped=True;return -9
    child=Child();owner=owned.OwnedStage();owner.process=child
    monkeypatch.setattr(owned.os,'killpg',lambda *a:(_ for _ in ()).throw(PermissionError('generated EPERM')))
    def observer(pid,**kwargs):return (1,[pid]) if child.live else (0,[])
    record=owner.cleanup(observer,owned.time.monotonic()+2)
    assert child.killed and child.reaped and record['contained'] and record['direct_child_reaped']
    assert 'killpg:PermissionError' in record['errors'] # Parent retains failure; no success laundering.


@pytest.mark.parametrize('kind',['fifo','symlink','too_large','empty','bad_hash','changed'])
def test_bootstrap_regular_read_refusals(tmp_path,monkeypatch,kind):
    import os
    path=tmp_path/'input'
    if kind=='fifo':os.mkfifo(path)
    elif kind=='symlink':
        (tmp_path/'target').write_bytes(b'ok');path.symlink_to(tmp_path/'target')
    else:path.write_bytes(b'' if kind=='empty' else b'valid')
    if kind=='changed':
        original=runner.os.fdopen
        class Changing:
            def __init__(self,*a,**k):self.stream=original(*a,**k)
            def __enter__(self):return self
            def __exit__(self,*a):self.stream.close()
            def read(self,n):
                raw=self.stream.read(n);path.write_bytes(b'altered');return raw
        monkeypatch.setattr(runner.os,'fdopen',Changing)
    with pytest.raises((ValueError,OSError)):
        runner.regular_bytes(path,2 if kind=='too_large' else 32,
                             expected='0'*64 if kind=='bad_hash' else None)


def test_bootstrap_compiles_only_verified_bytes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    (tmp_path/'scripts').mkdir();(tmp_path/'site').mkdir()
    source=b"marker='verified'\ndef check_release(*a,**k):return RELEASE,{}\ndef verify_sources(*a,**k):pass\n"
    path=tmp_path/runner.CORE;path.write_bytes(source)
    (tmp_path/runner.RUNNER).write_bytes(b'generated runner')
    release={'source_bindings':{runner.CORE:core.sha(source),runner.RUNNER:core.sha(b'generated runner')},
             'execution_released':False,'runtime':{'site_packages':[str(tmp_path/'site')]}}
    raw=core.canonical(release);(tmp_path/'release.json').write_bytes(raw)
    original=runner.regular_bytes;reads=[]
    def read(p,*a,**k):
        data=original(p,*a,**k)
        if Path(p)==path:
            reads.append(1);path.write_bytes(b"marker='changed'\n")
        return data
    monkeypatch.setattr(runner,'regular_bytes',read)
    monkeypatch.setattr(runner,'__file__',str(tmp_path/runner.RUNNER))
    monkeypatch.setattr(sys,'flags',SimpleNamespace(isolated=True,no_site=True))
    monkeypatch.setattr(sys,'dont_write_bytecode',True)
    monkeypatch.setattr(sys,'pycache_prefix',str(tmp_path/'build/fresh'))
    # Inject only the release object into the generated test module before exec.
    original_module=runner.importlib.util.module_from_spec
    def module(spec):
        value=original_module(spec);value.RELEASE=release;return value
    monkeypatch.setattr(runner.importlib.util,'module_from_spec',module)
    monkeypatch.setattr(sys,'path',list(sys.path))
    checked,*_=runner.bootstrap(tmp_path,'release.json',core.sha(raw),worker=False)
    assert checked.marker=='verified' and reads==[1]


def test_cache_lexical_escape_and_existing_path_refuse(tmp_path):
    for path in (tmp_path/'build/../outside',tmp_path/'build/unused/../../outside'):
        with pytest.raises(ValueError,match='cache'):runner.check_cache(tmp_path,path)
    exists=tmp_path/'build/existing';exists.mkdir(parents=True)
    with pytest.raises(ValueError,match='cache'):runner.check_cache(tmp_path,exists)


@pytest.mark.parametrize('mode',['fallback','cleanup_exit','late_fit','late_extra','persisted_terminal',
    'persisted_publication','late_cache','late_authority','publication_deadline'])
def test_late_acceptance_gaps_refuse(tmp_path,monkeypatch,mode):
    from scripts import mechanics_hbe_v5_remaining_one_shot as stages
    from launchers import hbe_v5_ordinal9_continuation_v1 as owned
    release={'execution_released':True,'declaration':{'path':'decl.json','sha256':'1'*64},'source_commit':'a'*40}
    rb={'path':'release.json','sha256':'2'*64};declaration={'evidence':{},'references':{}}
    class Owner:
        def cleanup(self,*a):return dict(contained=True,direct_child_reaped=True,remaining_members=[],errors=[],
            fallback_used=mode=='fallback',exit_code=9 if mode=='cleanup_exit' else 0)
    monkeypatch.setattr(owned,'OwnedStage',Owner)
    monkeypatch.setattr(core,'audit_loaded',lambda *a:None)
    monkeypatch.setattr(core,'verify_sources',lambda *a,**k:None)
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,declaration))
    monkeypatch.setattr(sys,'pycache_prefix',str(tmp_path/'build/unused-parent-cache'))
    original=core.exclusive_json;clock=runner.time.monotonic;deadline_shift=[0.]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock()+deadline_shift[0])
    def save(path,value,**kw):
        if path.name=='terminal.json':
            if mode=='late_fit':(path.parent/'fit.json').write_text('{"tampered":true}\n')
            if mode=='late_extra':(path.parent/'unbound.json').write_text('{"extra":true}\n')
            if mode=='persisted_terminal':value={**value,'fit_calls':999}
        if path.name=='publication.json' and value.get('accepted') is True:
            if mode=='persisted_publication':value={**value,'outputs':{}}
            if mode=='late_cache':(path.parent/'unused-worker-pycache').mkdir()
            if mode=='late_authority':monkeypatch.setattr(core,'check_release',lambda *a,**k:({},{}))
            if mode=='publication_deadline':deadline_shift[0]=100.
        return original(path,value,**kw)
    monkeypatch.setattr(core,'exclusive_json',save)
    def stage(kind,command,out,record,**kw):
        fit=core.exclusive_json(out/'fit.json',{'generated':True})
        met=core.exclusive_json(out/'calibration-metrics.json',{'physical_validation_pass':None})
        state={'status':core.DONE,'fit_calls':1,'native_calls':0,'mesher_calls':0,'held_out_member_reads':0,
            'calibration_responses_accessed':True,'freeze_saved':False,'physical_validation_pass':None,'release':rb,
            'fit':{'path':core.OUTPUT+'/fit.json','sha256':fit},
            'calibration_metrics':{'path':core.OUTPUT+'/calibration-metrics.json','sha256':met}}
        core.exclusive_json(out/'state.json',state)
        core.exclusive_json(out/'receipt.json',{'generated_supervision':True})
        (out/'readout-console.txt').write_bytes(b'')
        (out/'access.jsonl').write_text('{"phase":"calibration_attempt"}\n{"phase":"calibration_completed"}\n')
        return {'status':'completed_within_caps','exit_code':0}
    monkeypatch.setattr(stages,'supervise_stage',stage)
    result=runner.execute(tmp_path,core,rb,release,declaration)
    assert result['status']=='failed_axial_calibration_attempt'
    out=tmp_path/core.OUTPUT
    assert (out/'publication.json').exists()
    publication=json.loads((out/'publication.json').read_text())
    assert publication['accepted'] is False or (out/'finalization-failure.json').exists()


@pytest.mark.parametrize('digest',[None,'','invalid',True,'missing'])
def test_missing_source_digest_refuses_before_source_or_compile(tmp_path,monkeypatch,digest):
    from types import SimpleNamespace
    release={'source_bindings':{runner.CORE:digest,runner.RUNNER:'1'*64}}
    if digest=='missing':del release['source_bindings'][runner.CORE]
    raw=core.canonical(release);(tmp_path/'release.json').write_bytes(raw)
    original=runner.regular_bytes
    def read(path,*a,**k):
        assert Path(path).name=='release.json','unverified source read attempted'
        return original(path,*a,**k)
    monkeypatch.setattr(runner,'regular_bytes',read)
    monkeypatch.setattr(sys,'flags',SimpleNamespace(isolated=True,no_site=True))
    monkeypatch.setattr(sys,'dont_write_bytecode',True)
    monkeypatch.setattr(sys,'pycache_prefix',str(tmp_path/'build/fresh'))
    with pytest.raises(ValueError,match='source hashes'):
        runner.bootstrap(tmp_path,'release.json',core.sha(raw),worker=False)
