"""Launcher orchestration tests only; never start a process or load FEBio."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mpc_launch',ROOT/'scripts/mechanics_patient_constraints_run.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def complete_result(inputs):
    return {'status':'completed','solver_invocations':3,'no_retry':True,
            'cases':[{'case':case,'status':'passed','checker_passed':True} for case in v.CASES],
            'inputs_after':{name:{'unchanged':True} for name in inputs}}


def worker_fixture(tmp_path,monkeypatch,checker_pass=True,remove_input=False):
    source=tmp_path/'archive';source.mkdir();output=tmp_path/'output';output.mkdir()
    marker=source/'bound';marker.write_text('fixed')
    decks=source/v.DECKS;decks.mkdir(parents=True)
    for case in v.CASES:(decks/(case+'.feb')).write_text('fixed '+case)
    base={'source_directory':str(source),'release_path':'mock-release','executable':'NEVER_EXECUTE',
          'input_hashes':{str(marker):v.sha(marker)},'attempt_directory':str(output)}
    v.write(output/'execution-baseline.json',base);v.write(output/'results.json',v.initial_result())
    monkeypatch.setattr(v,'ROOT',source);monkeypatch.setattr(v,'baseline',lambda _:copy.deepcopy(base))
    checker=SimpleNamespace(__file__=str(source/'checker.py'),np=SimpleNamespace(__version__='mock'),
                            patch=SimpleNamespace(read_bounded_text=lambda p:Path(p).read_text()),
                            check_outputs=lambda *args:{'passed':checker_pass})
    monkeypatch.setattr(v,'load',lambda *args:checker)
    invocations=[]
    def run(command,*,cwd,stdout,stderr,timeout):
        invocations.append(command);name=Path(command[4]).stem
        for suffix in ('nodes.log','elements.log','log'):(Path(cwd)/(name+'.'+suffix)).write_text('mock primitive')
        stdout.write('Selecting linear solver skyline\n')
        if remove_input:marker.unlink()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(v.subprocess,'run',run)
    return source,output,base,invocations


def test_unreleased_template_refuses_before_any_git_or_solver_call(tmp_path,monkeypatch):
    release=tmp_path/'release.json';release.write_text(json.dumps({'schema':v.VERSION+'-release','authorized':False}))
    monkeypatch.setattr(v.subprocess,'run',lambda *a,**k:pytest.fail('No process allowed'))
    with pytest.raises(ValueError,match='released'):v.baseline(release)


def test_complete_cross_record_consistency_is_required():
    after={'a':{'unchanged':True}};result=complete_result(after);supervision={'status':'completed','exit_code':0}
    assert v.accept(supervision,result,after)
    variants=[]
    for key,value in [('status','failed'),('solver_invocations',2),('no_retry',False),('inputs_after',{})]:
        altered=copy.deepcopy(result);altered[key]=value;variants.append(altered)
    for cases in ([1,2,3],list(reversed(result['cases'])),result['cases'][:2]):
        altered=copy.deepcopy(result);altered['cases']=cases;variants.append(altered)
    assert not any(v.accept(supervision,item,after) for item in variants)
    assert not v.accept({'status':'failed_or_incomplete','exit_code':0},result,after)
    assert not v.accept(supervision,result,{'different':{'unchanged':True}})


def test_three_mocked_cases_are_executed_once_in_declared_order(tmp_path,monkeypatch):
    _,output,_,commands=worker_fixture(tmp_path,monkeypatch)
    assert v.worker(output)==0 and len(commands)==3
    result=json.loads((output/'results.json').read_text())
    assert [row['case'] for row in result['cases']]==list(v.CASES)
    assert all(row['status']=='passed' for row in result['cases'])


def test_first_checker_failure_preserves_two_unexecuted_cases(tmp_path,monkeypatch):
    _,output,_,commands=worker_fixture(tmp_path,monkeypatch,checker_pass=False)
    assert v.worker(output)==1 and len(commands)==1
    result=json.loads((output/'results.json').read_text())
    assert [row['status'] for row in result['cases']]==['failed','not_executed','not_executed']
    assert (output/v.CASES[0]/'checked.json').is_file()
    assert not (output/v.CASES[1]).exists()


def test_final_missing_input_still_writes_failed_receipt(tmp_path,monkeypatch):
    _,output,base,commands=worker_fixture(tmp_path,monkeypatch,remove_input=True)
    assert v.worker(output)==1
    result=json.loads((output/'results.json').read_text())
    row=result['inputs_after'][next(iter(base['input_hashes']))]
    assert not row['unchanged'] and 'FileNotFoundError' in row['error']


def test_worker_must_revalidate_release_before_first_mocked_case(tmp_path,monkeypatch):
    _,output,_,commands=worker_fixture(tmp_path,monkeypatch)
    def reject(_):raise ValueError('source mismatch')
    monkeypatch.setattr(v,'baseline',reject)
    assert v.worker(output)==1 and not commands
    result=json.loads((output/'results.json').read_text())
    assert 'source mismatch' in result['worker_error']
    assert all(row['status']=='not_executed' for row in result['cases'])


def test_parent_does_not_accept_success_payload_after_cap_failure(tmp_path,monkeypatch):
    marker=tmp_path/'source';marker.write_text('fixed');output=tmp_path/'output'
    base={'input_hashes':{str(marker):v.sha(marker)},'attempt_directory':str(output)}
    monkeypatch.setattr(v,'baseline',lambda _:base)
    def supervise(command,directory,**kwargs):
        assert kwargs['seconds']==60 and kwargs['rss_bytes']==3*1024**3
        v.write(output/'results.json',complete_result(base['input_hashes']))
        return {'status':'failed_or_incomplete','exit_code':0,'kill_reason':'wall_cap'}
    runtime=SimpleNamespace(private_environment=lambda _: {},declaration=lambda:{'caps':{'thread_environment':{}}},supervise=supervise)
    monkeypatch.setattr(v,'load',lambda *args:runtime)
    assert v.launch('mock-release',output)==1
    assert json.loads((output/'execution.json').read_text())['status']=='failed_or_incomplete'


def test_fresh_output_gate_preserves_existing_attempt(tmp_path,monkeypatch):
    output=tmp_path/'output';output.mkdir();marker=output/'original';marker.write_text('keep')
    monkeypatch.setattr(v,'baseline',lambda _:pytest.fail('No second attempt'))
    with pytest.raises(FileExistsError):v.launch('unused',output)
    assert marker.read_text()=='keep'


def test_release_cannot_launch_into_a_second_fresh_output_path(tmp_path,monkeypatch):
    requested=tmp_path/'wrong';released=tmp_path/'only_attempt'
    monkeypatch.setattr(v,'baseline',lambda _:{'attempt_directory':str(released),'input_hashes':{}})
    monkeypatch.setattr(v,'load',lambda *a:pytest.fail('Cannot launch another attempt'))
    assert v.launch('mock-release',requested)==1
    receipt=json.loads((requested/'execution.json').read_text())
    assert not receipt['worker_started'] and 'single released attempt' in receipt['error']
