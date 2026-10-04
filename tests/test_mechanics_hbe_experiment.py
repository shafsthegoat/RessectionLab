"""Orchestration-only temporary fixtures: no tissue data or FEBio execution."""
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from scripts import mechanics_hbe_experiment as x


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(x.access.canonical_json(value))
    return {'path': path.name, 'sha256': x.runtime.sha(path)}


def test_child_timeout_is_terminal_and_reaped(tmp_path):
    pidfile = tmp_path/'pid.txt'
    code = f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(10)'
    with pytest.raises(subprocess.TimeoutExpired):
        x.solve([sys.executable, '-c', code], tmp_path, .1)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    receipt=json.loads((tmp_path/'solver-execution.json').read_text())
    assert receipt['timed_out'] and receipt['exit_code']==-9 and receipt['cleanup']['reaped']
    assert receipt['elapsed_seconds']>0 and receipt['pid']==pid


def test_nonzero_solver_never_becomes_success(tmp_path):
    with pytest.raises(RuntimeError, match='exited 7'):
        x.solve([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path, 2)
    assert (tmp_path/'console.txt').is_file()
    receipt=json.loads((tmp_path/'solver-execution.json').read_text())
    assert receipt['exit_code']==7 and receipt['timed_out'] is False and receipt['cleanup']['reaped']


@pytest.mark.parametrize('kind', ['total', 'active', 'observer', 'receipt'])
def test_output_caps_fail_closed_without_signaling_real_groups(tmp_path, monkeypatch, kind):
    active = tmp_path/'run';active.mkdir();(active/'payload').write_bytes(b'x'*20)
    kills = []
    watch = x.OutputWatch(tmp_path, tmp_path/'watch.json',
        {'generated_output_bytes': 10 if kind=='total' else 10000,
         'each_active_run_output_bytes': 10 if kind=='active' else 10000}, kill=lambda:kills.append(True))
    watch.active = active
    if kind=='observer':
        monkeypatch.setattr(x.meshing, 'tree_bytes', lambda _: (_ for _ in ()).throw(OSError('unreadable')))
    if kind=='receipt':
        monkeypatch.setattr(x.runtime, 'write_json', lambda *_: (_ for _ in ()).throw(OSError('full disk')))
    with pytest.raises(RuntimeError):watch.sample()
    assert kills==[True]
    if kind!='receipt':assert json.loads((tmp_path/'watch.json').read_text())['status']=='failed'


def test_output_watcher_refuses_unrelated_group(tmp_path, monkeypatch):
    monkeypatch.setattr(x.os, 'getpid', lambda:101)
    monkeypatch.setattr(x.os, 'getpgrp', lambda:99)
    watch = x.OutputWatch(tmp_path, tmp_path/'watch.json',
        {'generated_output_bytes':1000, 'each_active_run_output_bytes':1000})
    with pytest.raises(RuntimeError, match='supervised session leader'):
        watch.__enter__()


def test_real_output_watchdog_kills_only_its_supervised_group(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();active=raw/'active';active.mkdir()
    code=('from pathlib import Path\nimport time\n'
          'from scripts.mechanics_hbe_experiment import OutputWatch\n'
          f'with OutputWatch(Path({str(raw)!r}),Path({str(raw/"watch.json")!r}),'
          '{"generated_output_bytes":100000,"each_active_run_output_bytes":500}) as watch:\n'
          f' watch.active=Path({str(active)!r})\n'
          f' Path({str(active/"payload")!r}).write_bytes(b"x"*600)\n'
          ' time.sleep(10)\n')
    result=x.runtime.supervise([sys.executable,'-c',code],tmp_path/'supervision',cwd=x.ROOT,
        environment=dict(os.environ),seconds=3,rss_bytes=512*1024**2)
    assert result['status']!='completed'
    assert result['exit_code']==-9
    assert json.loads((raw/'watch.json').read_text())['reason']=='active_output_cap'
    assert os.getpid()!=result['pid']


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path/'helper.py';source.write_text('value = 1\n')
    driver = tmp_path/'driver.py';driver.write_text('value = 2\n')
    monkeypatch.setattr(x, 'SOURCE_MODULES', {'helper':SimpleNamespace(__file__=str(source))})
    monkeypatch.setattr(x, '__file__', str(driver))
    sources = {'helper':x.binding(tmp_path,source), 'orchestrator':x.binding(tmp_path,driver)}
    archive_path = tmp_path/'source.tar'
    with tarfile.open(archive_path,'w') as archive:
        for path in (source,driver):archive.add(path,arcname='scripts/'+path.name)
    archive = x.binding(tmp_path,archive_path)
    return sources,archive


def test_source_archive_and_loaded_origin_both_bound(tmp_path,monkeypatch):
    sources,archive=source_fixture(tmp_path,monkeypatch)
    x.source_inventory(tmp_path,sources,archive,archive['sha256'])
    alien=tmp_path/'alien.py';alien.write_text('value = 1\n')
    monkeypatch.setattr(x, 'SOURCE_MODULES', {'helper':SimpleNamespace(__file__=str(alien))})
    with pytest.raises(ValueError,match='origin'):
        x.source_inventory(tmp_path,sources,archive,archive['sha256'])


def test_changed_source_cannot_borrow_original_archive(tmp_path,monkeypatch):
    sources,archive=source_fixture(tmp_path,monkeypatch)
    (tmp_path/'helper.py').write_text('value = 999\n')
    sources['helper']=x.binding(tmp_path,tmp_path/'helper.py')
    with pytest.raises(ValueError,match='committed archive'):
        x.source_inventory(tmp_path,sources,archive,archive['sha256'])


def fixture(tmp_path, monkeypatch, *, fail_run=None, bad_comparison=False, freeze_failure=False):
    """Clearly synthetic state-machine doubles; no production evidence receipts."""
    directory=tmp_path/'experiment';directory.mkdir()
    mesh=tmp_path/'mesh.json';mesh.write_text('{}')
    plan={'protocol':{'budgets':{'aggregate_specimen_seconds':900,'each_solver_seconds':90}},
          'protocol_binding':{'path':'protocol.json','sha256':'a'*64},
          'release_binding':{'path':'release.json','sha256':'b'*64},
          'inputs':{},'raw_root':str(tmp_path),'experiment':str(directory),
          'source_bindings':{key:{} for key in ('physics','primitive_parser','mesh_deck','curve_evaluator','readout')},
          'cases':{key:{} for key,v in x.access.expected_runs().items() if v[-1]!='fitted'},
          'meshes':{'12':x.binding(tmp_path,mesh)},'csv_schemas':{mode:{} for mode in x.BRANCHES}}
    events=[]
    class Watch:
        def __init__(self,*_):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
    monkeypatch.setattr(x,'OutputWatch',Watch)
    def run_case(root,plan,run_id,inputs,mu,state,watch,deadline):
        events.append(run_id)
        state['solver_invocations']+=1
        if run_id==fail_run:
            state['runs'][run_id]['status']='failed'
            raise RuntimeError('deliberate analytic orchestration failure')
        branch,N,steps,role=x.access.expected_runs()[run_id]
        sign=-1 if branch in ('compression','torsion_neg') else 1
        coordinate=[0.,sign*.0005] if not branch.startswith('torsion') else [0.,sign*.1]
        row={'branch':branch,'mesh_N':N,'steps':steps,'mu_Pa':mu,'frame_count':steps+1,
             'load_coordinate':coordinate,'applied_force_N':[0.,sign*.01*mu/1000],
             'applied_torque_Nm':[0.,sign*.001*mu/1000],'criteria':{}}
        state['runs'][run_id]['status']='passed_individual_numerical_checks'
        return row,{}
    monkeypatch.setattr(x,'run_case',run_case)
    def build(runs,caches,**kwargs):
        events.append('compare20' if kwargs['fitted_mu_Pa'] is not None else 'compare18')
        metric={'actual':2. if bad_comparison else 0.,'limit':1.,'units':'synthetic_test_ratio'}
        return {'runs':dict(runs),**{group:{'synthetic':{'criterion':metric}} for group in ('mesh','step','scale','fitted_confirmation')}}
    monkeypatch.setattr(x.readout,'build_numerical_evidence',build)
    def check(report,**kwargs):
        for group in ('mesh','step','scale','fitted_confirmation'):
            for row in report[group].values():
                for metric in row.values():x.access._metric_record(metric)
    monkeypatch.setattr(x.access,'check_numeric_report',check,raising=False)
    class Study:
        def __init__(self,*_,**__):pass
        def read_calibration(self,schemas):
            events.append('calibration')
            return {mode:x.evaluation.curve(mode,[0.,sign*.0005],[0.,sign*.02])
                    for mode,sign in [('compression',-1),('tension',1)]}
        def freeze_predictions(self,**kwargs):
            events.append('freeze')
            if freeze_failure:raise ValueError('deliberate freeze rejection')
            for name in ('fit_binding','prediction_binding','numerical_binding'):
                x.access.verify_binding(tmp_path,kwargs[name])
            return x.saved(tmp_path, tmp_path/kwargs['output_path'], {'synthetic_test_freeze':True})
        def evaluate_held_out(self,**kwargs):
            events.append('holdout')
            assert (directory/'parameter-prediction-freeze.json').is_file()
            state=json.loads((directory/'state.json').read_text())
            assert state['status']=='parameters_and_predictions_frozen_before_holdout'
            return {'physical_validation_pass':None,'synthetic_test_only':True}
    monkeypatch.setattr(x.access,'ReleasedStudy',Study)
    monkeypatch.setattr(x.meshing,'specimen_deck',lambda *args:('<synthetic-test-deck/>',{}))
    monkeypatch.setattr(x,'check_fitted_deck',lambda *args:None)
    test_deck=tmp_path/'fixture.feb';test_deck.write_text('<synthetic-test-deck/>')
    for mode in ('compression','tension'):
        plan['cases'][f'{mode}:N12:S120:reference']['deck']=x.binding(tmp_path,test_deck)
    return plan,events


def test_fixed_twenty_then_durable_freeze_precedes_holdout(tmp_path,monkeypatch):
    plan,events=fixture(tmp_path,monkeypatch)
    result=x.experiment_worker(tmp_path,plan)
    assert result['status']=='completed_descriptive_one_specimen_comparison'
    assert result['solver_invocations']==20
    runs=[v for v in events if ':N' in v]
    assert runs==list(x.access.expected_runs())
    assert events.index('compare18')<events.index('calibration')<events.index('compression:N12:S120:fitted')
    assert events.index('compare20')<events.index('freeze')<events.index('holdout')


@pytest.mark.parametrize('fail_run', ['compression:N4:S60:reference','torsion_pos:N12:S120:double_mu','tension:N12:S120:fitted'])
def test_first_failure_preserved_and_all_later_cases_stay_unexecuted(tmp_path,monkeypatch,fail_run):
    plan,events=fixture(tmp_path,monkeypatch,fail_run=fail_run)
    result=x.experiment_worker(tmp_path,plan)
    assert result['status']=='failed_or_incomplete'
    assert 'holdout' not in events and 'freeze' not in events
    ids=list(x.access.expected_runs());index=ids.index(fail_run)
    assert result['solver_invocations']==index+1
    assert all(result['runs'][key]['status']=='not_executed' for key in ids[index+1:])
    assert json.loads((tmp_path/'experiment/state.json').read_text())['error']['message'].startswith('deliberate')


def test_bad_reference_comparison_keeps_calibration_sealed(tmp_path,monkeypatch):
    plan,events=fixture(tmp_path,monkeypatch,bad_comparison=True)
    result=x.experiment_worker(tmp_path,plan)
    assert result['status']=='failed_or_incomplete' and result['solver_invocations']==18
    assert 'calibration' not in events and 'holdout' not in events
    assert (tmp_path/'experiment/reference-numerical.json').is_file()


def test_failed_freeze_never_reads_torque(tmp_path,monkeypatch):
    plan,events=fixture(tmp_path,monkeypatch,freeze_failure=True)
    result=x.experiment_worker(tmp_path,plan)
    assert result['status']=='failed_or_incomplete' and result['solver_invocations']==20
    assert 'freeze' in events and 'holdout' not in events


def test_exclusive_attempt_marker_blocks_second_launch_before_supervision(tmp_path,monkeypatch):
    plan={'raw_root':str(tmp_path),'experiment':str(tmp_path/'experiment')}
    monkeypatch.setattr(x,'preflight',lambda *_:plan)
    (tmp_path/'.specimen-experiment-started.json').write_text('{}')
    monkeypatch.setattr(x.runtime,'supervise',lambda *a,**k:pytest.fail('No retry may launch'))
    with pytest.raises(FileExistsError):x.launch(tmp_path,{}, {})


@pytest.mark.parametrize('failure', ['solver', 'primitive', 'allocation'])
def test_case_retains_actual_stage_and_stops_without_relabeling_failure(tmp_path,monkeypatch,failure):
    prepared=tmp_path/'prepared';prepared.mkdir()
    records={}
    for name,filename in [('deck','specimen.feb'),('loading','loading.json'),('mesh','mesh.json')]:
        path=prepared/filename;path.write_text('synthetic test input')
        records[name]=x.binding(tmp_path,path)
    executable=tmp_path/'test-executable';executable.write_text('never executed in this fixture')
    experiment=tmp_path/'experiment';experiment.mkdir()
    plan={'experiment':str(experiment),'executable':str(executable),'runtime_identity':{},
          'protocol':{'budgets':{'each_solver_seconds':90}},'protocol_binding':{'sha256':'a'*64},
          'inputs':{str(executable):x.runtime.sha(executable)}}
    run_id='compression:N4:S60:reference'
    state={'solver_invocations':20 if failure=='allocation' else 0,'runs':{run_id:{'status':'not_executed'}}}
    calls=[]
    def fake_solver(command,directory,seconds):
        calls.append(command)
        if failure=='solver':
            (directory/'solver.log').write_text('partial solver failure output retained')
            raise subprocess.TimeoutExpired(command,seconds)
        for filename in ('nodes.log','elements.log','solver.log'):
            (directory/filename).write_text('synthetic finite primitive placeholder in unit fixture only')
        return {'exit_code':0,'elapsed_seconds':.01,'timed_out':False}
    monkeypatch.setattr(x,'solve',fake_solver)
    def rejected_readout(root,bindings,**kwargs):
        for value in bindings.values():x.access.verify_binding(root,value)
        return {'passed':False,'synthetic_test_only':True},None
    monkeypatch.setattr(x.readout,'read_run',rejected_readout)
    with pytest.raises((subprocess.TimeoutExpired,ValueError)):
        x.run_case(tmp_path,plan,run_id,records,1000.,state,SimpleNamespace(active=None),time.monotonic()+10)
    row=json.loads((experiment/'state.json').read_text())['runs'][run_id]
    assert row['status']=='failed'
    assert ('solver.log' in row['retained_files'])==(failure!='allocation')
    assert ('execution.json' in row['retained_files'])==(failure=='primitive')
    assert len(calls)==(0 if failure=='allocation' else 1)
    assert state['solver_invocations']==(20 if failure=='allocation' else 1)


def test_real_readout_interface_from_analytic_logs_through_case_receipt(tmp_path,monkeypatch):
    # The external solver is replaced by an explicitly analytical log generator;
    # the production parser, physics readout and binding assembly run unchanged.
    path=Path(__file__).with_name('test_mechanics_hbe_readout.py')
    spec=importlib.util.spec_from_file_location('analytical_log_fixture',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    prepared=tmp_path/'prepared';prepared.mkdir()
    records=helper.fixture(prepared)
    records={key:{'path':'prepared/'+value['path'],'sha256':value['sha256']} for key,value in records.items()}
    executable=tmp_path/'test-executable';executable.write_text('not a FEBio runtime; never launched')
    experiment=tmp_path/'experiment';experiment.mkdir()
    plan={'experiment':str(experiment),'executable':str(executable),'runtime_identity':{},
          'protocol':{'budgets':{'each_solver_seconds':90}},'protocol_binding':{'sha256':'1'*64},
          'inputs':{str(executable):x.runtime.sha(executable)}}
    run_id='tension:N4:S60:reference';state={'solver_invocations':0,'runs':{run_id:{'status':'not_executed'}}}
    def analytical_output(command,directory,seconds):
        for key in ('nodes','elements','solver'):
            (directory/(key+'.log')).write_bytes(x.access.local_path(tmp_path,records[key]['path']).read_bytes())
        return {'exit_code':0,'elapsed_seconds':.01,'timed_out':False}
    monkeypatch.setattr(x,'solve',analytical_output)
    receipt,cache=x.run_case(tmp_path,plan,run_id,{key:records[key] for key in ('mesh','deck','loading')},
                             1000.,state,SimpleNamespace(active=None),time.monotonic()+10)
    assert receipt['passed'] and receipt['frame_count']==61 and cache is None
    execution=x.access.verify_binding(tmp_path,receipt['execution_binding'],read_json=True)
    assert execution['primitive_bindings']==receipt['primitive_bindings']
    assert execution['cwd']==state['runs'][run_id]['directory']
    assert execution['run_id']==run_id and execution['protocol_sha256']=='1'*64
    assert state['runs'][run_id]['status']=='passed_individual_numerical_checks'


def scalar_deck(mu):
    return (f'<febio_spec><Control><solver><min_residual>{(1e-10*mu*.004**2)**2:.17g}</min_residual>'
            '<dtol>1e-8</dtol></solver></Control><Material><material>'
            f'<c1>{2*mu:.17g}</c1><k>{149*mu/3:.17g}</k><m1>2</m1>'
            '</material></Material><LoadData><load_controller><points><pt>0,0</pt><pt>1,1</pt>'
            '</points></load_controller></LoadData><Mesh><Elements><elem id="1">1,2,3,4,5,6,7,8</elem>'
            '</Elements></Mesh></febio_spec>')


def test_fitted_xml_allows_only_three_exact_declared_scalars():
    x.check_fitted_deck(scalar_deck(1000.),scalar_deck(123.456),123.456)


@pytest.mark.parametrize('before,after', [
    ('<m1>2</m1>','<m1>3</m1>'),
    ('<dtol>1e-8</dtol>','<dtol>1e-6</dtol>'),
    ('<pt>0,0</pt><pt>1,1</pt>','<pt>1,1</pt><pt>0,0</pt>'),
    ('1,2,3,4,5,6,7,8','2,1,3,4,5,6,7,8'),
    ('<c1>246.91200000000001</c1>','<c1>247</c1>'),
])
def test_fitted_xml_rejects_other_physics_or_order_changes(before,after):
    candidate=scalar_deck(123.456)
    assert before in candidate
    with pytest.raises(ValueError):x.check_fitted_deck(scalar_deck(1000.),candidate.replace(before,after),123.456)
