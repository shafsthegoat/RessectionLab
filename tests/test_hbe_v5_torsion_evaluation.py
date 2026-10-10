"""Generated fixtures only. Run after root releases the current resource slot."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('HBE_TORSION_SOURCE_ROOT',ROOT))


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,SOURCE/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


core=load('torsion_test_core','mechanics_hbe_v5_torsion_evaluation.py')
launcher=load('torsion_test_launcher','mechanics_hbe_v5_torsion_evaluation_experiment.py')


@pytest.fixture(autouse=True)
def no_process(monkeypatch):
    import subprocess
    import socket
    def forbidden(*a,**k):raise AssertionError('No process/network in generated tests')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    # Keep the class intact: ssl.SSLSocket subclasses it during source imports.
    for name in ('connect','connect_ex','bind','listen','accept','sendto'):
        monkeypatch.setattr(socket.socket,name,forbidden)
    for name in ('create_connection','getaddrinfo'):
        monkeypatch.setattr(socket,name,forbidden)


def test_network_guard_preserves_socket_class_and_refuses_entry():
    import socket
    import ssl
    assert isinstance(socket.socket,type) and issubclass(ssl.SSLSocket,socket.socket)
    for name in ('connect','connect_ex','bind','listen','accept','sendto'):
        with pytest.raises(AssertionError,match='No process/network'):
            getattr(socket.socket,name)(None)
    for name in ('create_connection','getaddrinfo'):
        with pytest.raises(AssertionError,match='No process/network'):
            getattr(socket,name)()


def saved(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    raw=core.canonical(value);path.write_bytes(raw)
    return {'path':name,'sha256':core.sha(raw)}


@pytest.fixture
def tiny_study(tmp_path,monkeypatch):
    from scripts import mechanics_hbe_access as access
    archive=tmp_path/'generated.zip';members=[]
    with zipfile.ZipFile(archive,'w') as output:
        for branch in core.TORSION:
            sign=-1 if branch=='torsion_neg' else 1
            data=f'0,0\n{sign*.05},{sign*.003}\n{sign*.1},{sign*.006}\n'.encode()
            output.writestr(branch+'.csv',data)
            members.append({'path':branch+'.csv','bytes':len(data),
                'crc32':f'{zipfile.crc32(data):08x}'})
    raw=archive.read_bytes()
    roles={'source':{'archive_path':'generated.zip','archive_sha256':core.sha(raw),'archive_bytes':len(raw)},
           'held_out_validation':{'members':members},'calibration':{'members':[]}}
    role_binding=saved(tmp_path,'roles.json',roles)
    protocol=saved(tmp_path,'protocol.json',{'roles':role_binding})
    schema={'delimiter':',','header':None,'coordinate_column':0,'response_column':1,
            'coordinate_unit':'rad','response_unit':'Nm'}
    declaration={'protocol':protocol,'roles':role_binding,'csv_schemas':{b:dict(schema) for b in core.TORSION}}
    release={'upstream':{'generated':'fixture only'}};rb={'path':'release.json','sha256':'a'*64}
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,declaration))
    study=core.make_study(tmp_path,rb,release,declaration,recheck=lambda:None)
    return SimpleNamespace(root=tmp_path,study=study,roles=roles,decl=declaration,release=release,rb=rb)


def test_preserved_upstream_then_two_members_once(tiny_study):
    f=tiny_study;s=f.study
    s._audit('upstream_freeze_admitted',[],upstream=f.release['upstream'])
    observed=s._read_selected(f.roles,f.roles['held_out_validation']['members'],core.TORSION,
                             f.decl['csv_schemas'],'held_out')
    assert set(observed)==set(core.TORSION)
    assert [r['phase'] for r in s._events()]==['upstream_freeze_admitted','held_out_attempt','held_out_completed']
    with pytest.raises(ValueError,match='consumed'):
        s._read_selected(f.roles,f.roles['held_out_validation']['members'],core.TORSION,
                         f.decl['csv_schemas'],'held_out')
    core.validate_terminal_ledger(f.root,s._events(),f.rb,f.release,f.decl,
        {'member_sha256':{b:observed[b].source_sha256 for b in core.TORSION}})


def test_without_freeze_no_archive_open(tiny_study,monkeypatch):
    f=tiny_study
    with pytest.raises(ValueError,match='consumed'):
        f.study._read_selected(f.roles,f.roles['held_out_validation']['members'],core.TORSION,
            f.decl['csv_schemas'],'held_out')
    assert not f.study.ledger_path.exists()


@pytest.mark.parametrize('change',['members','branches','phase','schema'])
def test_narrow_member_and_schema_scope(tiny_study,change):
    f=tiny_study;s=f.study;s._audit('upstream_freeze_admitted',[],upstream=f.release['upstream'])
    members=f.roles['held_out_validation']['members'];branches=core.TORSION;phase='held_out'
    schemas=f.decl['csv_schemas']
    if change=='members':members=members[::-1]
    if change=='branches':branches=('compression','tension')
    if change=='phase':phase='calibration'
    if change=='schema':schemas={b:dict(row,header=['angle','torque']) for b,row in schemas.items()}
    with pytest.raises(ValueError):s._read_selected(f.roles,members,branches,schemas,phase)
    assert len(s._events())==1


def test_parse_failure_retains_attempt_and_blocks_retry(tiny_study,monkeypatch):
    from scripts import mechanics_hbe_access as access
    f=tiny_study;s=f.study;s._audit('upstream_freeze_admitted',[],upstream=f.release['upstream'])
    def fail(*a,**k):raise ValueError('generated header mismatch')
    monkeypatch.setattr(access,'parse_member_csv',fail)
    with pytest.raises(ValueError,match='header mismatch'):
        s._read_selected(f.roles,f.roles['held_out_validation']['members'],core.TORSION,
            f.decl['csv_schemas'],'held_out')
    assert [r['phase'] for r in s._events()]==['upstream_freeze_admitted','held_out_attempt']
    with pytest.raises(ValueError,match='consumed'):
        s._read_selected(f.roles,f.roles['held_out_validation']['members'],core.TORSION,
            f.decl['csv_schemas'],'held_out')


def test_old_actions_refuse(tiny_study):
    for action in ('read_calibration','freeze_predictions','evaluate_held_out'):
        with pytest.raises(ValueError):getattr(tiny_study.study,action)()


def test_metrics_unchanged_and_descriptive():
    from scripts import mechanics_hbe_evaluation as evaluation
    pair={b:evaluation.curve(b,[0,(-.1 if b.endswith('neg') else .1)],
                            [0,(-.01 if b.endswith('neg') else .01)]) for b in core.TORSION}
    result=evaluation.paired_metrics(pair,pair,characteristic_response_scale=core.MU*.004**3)
    assert result['equal_branch_RMSE']==0 and result['physical_validation_pass'] is None
    assert result['empirical_tolerance'] is None and not result['independent_donor_validation']
    bad=dict(pair,torsion_pos=evaluation.curve('torsion_pos',[0,.11],[0,.01]))
    with pytest.raises(ValueError,match='no extrapolation'):
        evaluation.paired_metrics(bad,pair,characteristic_response_scale=core.MU*.004**3)


@pytest.fixture
def upstream(tmp_path):
    prefix='outputs/mechanics/hbe-v5-fixed-fit-confirmation-v1/attempt-01/'
    protocol={'path':'protocol.json','sha256':'1'*64};roles={'path':'roles.json','sha256':'2'*64}
    rb={'path':'upstream-release.json','sha256':'3'*64}
    old_ledger=saved(tmp_path,'old-access.jsonl',{'generated':'old ledger remains unchanged'})
    prediction={'schema':'hbe-torsion-prediction-v1','reference':{},'fitted':{}}
    for b in core.TORSION:
        sign=-1 if b.endswith('neg') else 1
        x=[sign*.183434625*i/120 for i in range(121)];x[-1]=sign*.183434625
        y=[v*.001 for v in x]
        prediction['reference'][b]={'coordinate':x,'response':y}
        prediction['fitted'][b]={'coordinate':x,'response':[v*core.SCALE for v in y],'response_unit':'Nm'}
    bindings={'predictions':saved(tmp_path,prefix+'predictions.json',prediction)}
    numeric={'passed':True,'mu_Pa':core.MU,'fixed_fit_sha256':core.FIT_SHA,
             'branches':{'compression':{'passed':True},'tension':{'passed':True}}}
    bindings['numerical']=saved(tmp_path,prefix+'numerical.json',numeric)
    freeze={'schema':'hbe-v5-fixed-parameter-prediction-freeze-v1','release':rb,'roles':roles,'protocol':protocol,
        'fixed_fit':{'fit':{'path':'do-not-read-fit.json','sha256':core.FIT_SHA}},'original_fit_ledger':old_ledger,
        'mu_Pa':core.MU,'scale':core.SCALE,'refit_calls':0,'native_calls':2,'mesher_calls':0,
        'measured_member_reads':0,'held_out_member_reads':0,'held_out_access_released':False,
        'patient_tool_mechanics_admitted':False,'physical_validation_pass':None,**bindings}
    bindings['freeze']=saved(tmp_path,prefix+'freeze.json',freeze)
    bindings['continuation_ledger']=saved(tmp_path,prefix+'continuation-ledger.json',{
        'phase':'freeze_saved','freeze':bindings['freeze'],'previous_fit_ledger':old_ledger,
        'measured_member_reads':0,'refit_calls':0})
    clean={'contained':True,'direct_child_reaped':True,'remaining_members':[],
           'fallback_used':False,'exit_code':0,'errors':[]}
    terminal={'status':'completed_fitted_confirmation_and_prediction_freeze','native_calls':2,
              'release':rb,'freeze':bindings['freeze'],
              'stages':{s:{'cleanup':dict(clean)} for s in ('prepare','compression','tension','readout')}}
    bindings['terminal']=saved(tmp_path,prefix+'terminal.json',terminal)
    published={'schema':'hbe-v5-fixed-fit-publication-v1','accepted':True,'physical_validation_pass':None,
        'release':rb,'outputs':{Path(v['path']).name:{'bytes':(tmp_path/v['path']).stat().st_size,
            'sha256':v['sha256']} for v in bindings.values()}}
    bindings['publication']=saved(tmp_path,prefix+'publication.json',published)
    review={'schema':'hbe-v5-fixed-fit-heldout-admission-v1','decision':'GO',**bindings,
        'fixed_fit_sha256':core.FIT_SHA,'mu_Pa':core.MU,'scale':core.SCALE,
        'fitted_checks':{'compression':'PASS','tension':'PASS'},'held_out_member_reads':0,
        'measured_member_reads':0,'states_per_native_run':121,'native_calls':2,'refit_calls':0,
        'held_out_access_released':False,'physical_validation_pass':None,'release':rb,
        'scope':'accepted saved fixed-fit confirmation and pre-reveal prediction freeze evidence only'}
    bindings['independent_review']=saved(tmp_path,'review.json',review)
    return SimpleNamespace(root=tmp_path,release={'upstream':bindings},decl={'roles':roles,'protocol':protocol})


def test_generated_upstream_admission_never_needs_fit_body(upstream):
    records,curves=core.admit_upstream(upstream.root,upstream.release,upstream.decl)
    assert all(len(curves[b].coordinate)==121 for b in core.TORSION)
    assert not (upstream.root/'do-not-read-fit.json').exists()


@pytest.mark.parametrize('name',['freeze','predictions','numerical','publication','terminal','continuation_ledger'])
def test_modified_upstream_bytes_refuse(upstream,name):
    path=upstream.root/upstream.release['upstream'][name]['path'];path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):core.admit_upstream(upstream.root,upstream.release,upstream.decl)


def test_failure_marker_refuses(upstream):
    (upstream.root/Path(upstream.release['upstream']['publication']['path']).parent/'finalization-failure.json').write_text('{}')
    with pytest.raises(ValueError,match='failed upstream'):core.admit_upstream(upstream.root,upstream.release,upstream.decl)


def test_wrong_review_binding_refuses(upstream):
    binding=upstream.release['upstream']['independent_review']
    obj=json.loads((upstream.root/binding['path']).read_bytes());obj['freeze']['sha256']='f'*64
    upstream.release['upstream']['independent_review']=saved(upstream.root,binding['path'],obj)
    with pytest.raises(ValueError,match='independently accepted'):core.admit_upstream(upstream.root,upstream.release,upstream.decl)


@pytest.mark.parametrize('change',['old_shape','access_permission','scope','failed_check','read_count','wrong_release'])
def test_admission_is_exact_evidence_not_reader_permission(upstream,change):
    binding=upstream.release['upstream']['independent_review']
    obj=json.loads((upstream.root/binding['path']).read_bytes())
    if change=='old_shape':
        obj['bindings']={key:obj.pop(key) for key in core.UPSTREAM_NAMES}
        obj['both_fitted_confirmations_passed']=True
    elif change=='access_permission':obj['held_out_access_released']=True
    elif change=='scope':obj['scope']='physical validation and clinical force approval'
    elif change=='failed_check':obj['fitted_checks']['tension']='FAIL'
    elif change=='read_count':obj['held_out_member_reads']=1
    elif change=='wrong_release':obj['release']['sha256']='0'*64
    upstream.release['upstream']['independent_review']=saved(upstream.root,binding['path'],obj)
    with pytest.raises(ValueError,match='independently accepted'):
        core.admit_upstream(upstream.root,upstream.release,upstream.decl)


def test_false_execution_before_directory(monkeypatch,tmp_path):
    fake=SimpleNamespace(need=core.need,OUTPUT='attempt')
    with pytest.raises(ValueError,match='release required'):
        launcher.execute(tmp_path,fake,{}, {'execution_released':False},{})
    assert not (tmp_path/'attempt').exists()


def test_exact_output_inventory_refuses_extra(tmp_path):
    (tmp_path/'allowed').write_text('{}');(tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError,match='inventory'):
        launcher.output_snapshot(core,tmp_path,{'allowed'})


@pytest.mark.parametrize('mode',['success','observer_raise','timeout','cleanup_eperm','unreaped',
    'late_output','late_source','late_publication','elapsed_after_exit0'])
def test_fake_parent_wiring_and_durable_failure(tmp_path,monkeypatch,mode):
    import sys
    from scripts import mechanics_hbe_v5_remaining_one_shot as stages
    from launchers import hbe_v5_ordinal9_continuation_v1 as owned
    rb={'path':'release.json','sha256':'2'*64}
    release={'execution_released':True,'source_commit':'a'*40,
             'declaration':{'path':'declaration.json','sha256':'1'*64},'upstream':{'fixture':True}}
    roles={'held_out_validation':{'members':[{'path':b+'.csv'} for b in core.TORSION]}}
    declaration={'roles':saved(tmp_path,'roles.json',roles),'protocol':{'path':'protocol.json','sha256':'3'*64}}
    clean={'contained':True,'direct_child_reaped':True,'remaining_members':[],'errors':[],
           'fallback_used':False,'exit_code':0}
    if mode=='cleanup_eperm':clean.update(errors=['killpg:PermissionError'],fallback_used=True)
    if mode=='unreaped':clean.update(contained=False,direct_child_reaped=False,remaining_members=[999999])
    cleanup_calls=[]
    class Owner:
        def cleanup(self,*a):cleanup_calls.append(True);return dict(clean)
    monkeypatch.setattr(owned,'OwnedStage',Owner)
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,declaration))
    monkeypatch.setattr(core,'verify_sources',lambda *a,**k:None)
    monkeypatch.setattr(core,'audit_loaded',lambda *a,**k:None)
    monkeypatch.setattr(core,'admit_upstream',lambda *a,**k:None)
    monkeypatch.setattr(sys,'pycache_prefix',str(tmp_path/'build/unused-parent-cache'))
    original=core.exclusive_json;clock=launcher.time.monotonic;shift=[0.]
    monkeypatch.setattr(launcher.time,'monotonic',lambda:clock()+shift[0])
    def save(path,value,**kwargs):
        if path.name=='terminal.json' and mode=='late_output':
            (path.parent/'held-out-metrics.json').write_text('{"tampered":true}\n')
        if path.name=='publication.json' and value.get('accepted') is True:
            if mode=='late_source':monkeypatch.setattr(core,'check_release',lambda *a,**k:({},{}))
            if mode=='late_publication':value={**value,'outputs':{}}
            if mode=='elapsed_after_exit0':shift[0]=100.
        return original(path,value,**kwargs)
    monkeypatch.setattr(core,'exclusive_json',save)
    def stage(kind,command,out,record,**kwargs):
        assert kind=='readout' and '--worker' in command and '-S' in command and '-B' in command
        assert 0<kwargs['wall_cap']<=60 and kwargs['rss_cap']==512*1024**2
        if mode=='observer_raise':raise RuntimeError('generated observer failure')
        metric=original(out/'held-out-metrics.json',{'physical_validation_pass':None,'empirical_tolerance':None,
            'independent_donor_validation':False,'branches':{b:{} for b in core.TORSION}})
        hashes={b:('4' if b.endswith('neg') else '5')*64 for b in core.TORSION}
        observed=original(out/'observed-torsion.json',{'schema':'hbe-held-out-torsion-curves-v1',
            'release':rb,'curves':{b:{'source_sha256':hashes[b]} for b in core.TORSION}})
        original(out/'state.json',{'status':core.DONE,'fit_calls':0,'native_calls':0,'mesher_calls':0,
            'held_out_member_reads':2,'held_out_responses_accessed':True,'new_freeze_saved':False,
            'physical_validation_pass':None,'empirical_tolerance':None,'release':rb,'upstream':release['upstream'],
            'held_out_metrics':{'path':core.OUTPUT+'/held-out-metrics.json','sha256':metric},
            'observed_torsion':{'path':core.OUTPUT+'/observed-torsion.json','sha256':observed},'member_sha256':hashes})
        events=[]
        for i,phase in enumerate(('upstream_freeze_admitted','held_out_attempt','held_out_completed')):
            row={'sequence':i,'phase':phase,'protocol_sha256':declaration['protocol']['sha256'],
                'release_sha256':rb['sha256'],'members':[] if i==0 else [b+'.csv' for b in core.TORSION]}
            if i==0:row['upstream']=release['upstream']
            if i==2:row['member_sha256']=hashes
            events.append(json.dumps(row)+'\n')
        (out/'access.jsonl').write_text(''.join(events));(out/'readout-console.txt').write_bytes(b'')
        original(out/'receipt.json',{'generated':True})
        return {'status':'failed_or_incomplete' if mode=='timeout' else 'completed_within_caps','exit_code':0}
    monkeypatch.setattr(stages,'supervise_stage',stage)
    result=launcher.execute(tmp_path,core,rb,release,declaration)
    assert cleanup_calls==[True]
    out=tmp_path/core.OUTPUT
    assert (out/'terminal.json').is_file() and (out/'publication.json').is_file()
    publication=json.loads((out/'publication.json').read_text())
    if mode=='success':assert result['status']==core.DONE and publication['accepted'] is True
    else:
        assert result['status']==core.FAILED
        assert publication['accepted'] is False or (out/'finalization-failure.json').is_file()
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,declaration))
    with pytest.raises(FileExistsError):launcher.execute(tmp_path,core,rb,release,declaration)


def test_owned_child_eperm_uses_fallback_but_never_clean_success(monkeypatch):
    from launchers import hbe_v5_ordinal9_continuation_v1 as owned
    class Child:
        pid=99999999;live=True;killed=False;reaped=False
        def poll(self):return None if self.live else -9
        def kill(self):self.killed=True;self.live=False
        def wait(self,timeout):self.reaped=True;return -9
    child=Child();owner=owned.OwnedStage();owner.process=child
    def eperm(*a):raise PermissionError('generated EPERM')
    monkeypatch.setattr(owned.os,'killpg',eperm)
    result=owner.cleanup(lambda pid,**k:(1,[pid]) if child.live else (0,[]),time.monotonic()+2)
    assert child.killed and child.reaped and result['direct_child_reaped']
    assert result['fallback_used'] and 'killpg:PermissionError' in result['errors']
