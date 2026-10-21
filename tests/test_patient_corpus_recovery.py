"""Recovery controls use copied JSON and fake workers, never source payloads."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from resectionlab import patient_corpus as pc
from resectionlab.core import semantic_digest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corpus_recovery',ROOT/'scripts/qualify_patient_corpus_recovery.py')
rc=importlib.util.module_from_spec(spec);spec.loader.exec_module(rc)


def forbidden(*args,**kwargs):raise AssertionError('A real payload or worker was opened')


def copy_metadata(root,*,launch=False):
    document,plan=rc.declaration(ROOT)
    paths={row['path'] for row in plan['metadata_evidence']}|{rc.DECLARATION_PATH,
        document['prior_failure']['path'],document['prior_failure']['supervision_path'],
        pc.EXECUTION_OUTPUT+'/launch-admission.json'}|{row['path'] for row in document['frozen_receipts']}
    for path in paths:
        assert path.endswith('.json')
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
    prepared=rc.prepare(root)
    if launch:pc._atomic_json(root/rc.OUTPUT/'launch-admission.json',rc.expected_launch(prepared))
    return document,plan


def worker_environment(monkeypatch):
    monkeypatch.setenv('RESSECTIONLAB_CORPUS_SUPERVISOR_PID',str(os.getppid()))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):monkeypatch.setenv(key,'2')


def fake_engine(monkeypatch,root,document,plan):
    """Populate JSON-only stage results to test dispatch and cached-only limits."""
    calls=[]
    def binding(base,candidate,current_plan):
        assert current_plan==plan
        return rc.receipt_index(root,document)[candidate['patient_group']]['record']['input_binding']
    def qualify(base,output,current_plan,*,limit):
        index=len(calls);calls.append(limit);stage=document['stages'][index]
        assert limit==stage['maximum_new_decodes'] and Path(output)==root/pc.EXECUTION_OUTPUT
        if stage['new_patient']:
            candidate=next(row for row in plan['candidates'] if row['patient_group']==stage['new_patient'])
            stub_binding={'plan_hash':plan['plan_hash'],'candidate':candidate,'generated_metadata_control':True}
            record={'patient_group':stage['new_patient'],'status':'qualified_with_limitations','input_binding':stub_binding,
                'input_binding_hash':semantic_digest(stub_binding),'generated_metadata_control':True}
            record['receipt_hash']=semantic_digest(record)
            pc._atomic_json(output/'receipts'/f'generated-stage-{index}.json',record)
        return {'counts':{'qualified_existing_people':stage['expected_qualified'],'pending':12-stage['expected_qualified'],'rejected':0},
            'rejections':[],'new_qualification_receipts':limit,'reused_qualification_receipts':stage['expected_reused']}
    monkeypatch.setattr(pc,'_input_binding',binding);monkeypatch.setattr(pc,'_run_qualification',qualify)
    return calls


def safe_sample():return {'rss':1,'available':10*1024**3,'elapsed':0,'output_bytes':1,'free_disk':100*1024**3}


def test_prepare_reads_metadata_and_code_only_and_resumes_without_launch(tmp_path,monkeypatch):
    document,plan=copy_metadata(tmp_path);before=(tmp_path/rc.OUTPUT/'prepared-recovery.json').read_bytes()
    real_open=Path.open
    def permitted(path,*args,**kwargs):
        assert path.suffix in {'.json','.py'},str(path)
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',permitted);monkeypatch.setattr(pc,'_input_binding',forbidden)
    monkeypatch.setattr(pc,'load_workspace',forbidden);monkeypatch.setattr(rc.subprocess,'Popen',forbidden)
    prepared=rc.prepare(tmp_path)
    assert before==(tmp_path/rc.OUTPUT/'prepared-recovery.json').read_bytes()
    assert len(plan['candidates'])==12 and len(document['frozen_receipts'])==10
    assert [s['maximum_new_decodes'] for s in document['stages']]==[1,1,0]
    assert prepared['source_payloads_opened'] is prepared['actual_launch_authorized'] is False
    with pytest.raises(pc.QualificationError) as error:rc.run(tmp_path)
    assert error.value.code=='RECOVERY_NOT_RELEASED'
    assert not (tmp_path/rc.OUTPUT/'started.json').exists()


@pytest.mark.parametrize('target',['declaration','receipt','failure','launch','prepared'])
def test_changed_frozen_protocol_or_receipt_refuses_before_worker(tmp_path,monkeypatch,target):
    document,_=copy_metadata(tmp_path,launch=True)
    path={'declaration':rc.DECLARATION_PATH,'receipt':document['frozen_receipts'][0]['path'],
        'failure':document['prior_failure']['path'],'launch':rc.OUTPUT+'/launch-admission.json',
        'prepared':rc.OUTPUT+'/prepared-recovery.json'}[target]
    record=json.loads((tmp_path/path).read_text());record['unauthorized_change']=True
    (tmp_path/path).write_text(json.dumps(record))
    monkeypatch.setattr(pc,'_input_binding',forbidden);monkeypatch.setattr(rc.subprocess,'Popen',forbidden)
    with pytest.raises(pc.QualificationError):rc.run(tmp_path)
    assert not (tmp_path/rc.OUTPUT/'started.json').exists()


def test_recovery_cannot_redirect_output_into_original_attempt(tmp_path):
    copy_metadata(tmp_path)
    path=tmp_path/rc.OUTPUT;path.rename(tmp_path/'moved');path.symlink_to(tmp_path/pc.EXECUTION_OUTPUT,target_is_directory=True)
    with pytest.raises(pc.QualificationError) as error:rc.prepare(tmp_path)
    assert error.value.code=='OUTPUT_SCOPE'


def test_generated_worker_stages_are_exactly_one_one_zero_decodes(tmp_path,monkeypatch):
    document,plan=copy_metadata(tmp_path,launch=True);worker_environment(monkeypatch)
    calls=fake_engine(monkeypatch,tmp_path,document,plan)
    before={r['path']:rc.digest(tmp_path/r['path']) for r in document['frozen_receipts']}
    for index in range(3):rc.run_worker(tmp_path,index)
    assert calls==[1,1,0]
    records=[rc.read_record(tmp_path/rc.OUTPUT/f'stage-{i}.json') for i in range(3)]
    assert [r['result']['counts']['qualified_existing_people'] for r in records]==[11,12,12]
    assert [r['final_cached_verification'] for r in records]==[False,False,True]
    assert records[0]['complete_twelve_person_catalog'] is False
    assert all(rc.digest(tmp_path/path)==expected for path,expected in before.items())


def test_changed_cached_binding_stops_before_any_new_decode(tmp_path,monkeypatch):
    copy_metadata(tmp_path,launch=True);worker_environment(monkeypatch)
    monkeypatch.setattr(pc,'_input_binding',lambda *args:{'changed_source':True})
    monkeypatch.setattr(pc,'_run_qualification',forbidden)
    with pytest.raises(pc.QualificationError) as error:rc.run_worker(tmp_path,0)
    assert error.value.code=='CACHED_BINDING_CHANGED'
    assert not (tmp_path/rc.OUTPUT/'stage-0.json').exists()


@pytest.mark.parametrize('index',[-1,1,2,3])
def test_worker_cannot_skip_repeat_or_expand_fixed_stages(tmp_path,monkeypatch,index):
    copy_metadata(tmp_path,launch=True);worker_environment(monkeypatch)
    monkeypatch.setattr(pc,'_input_binding',forbidden)
    with pytest.raises(pc.QualificationError) as error:rc.run_worker(tmp_path,index)
    assert error.value.code=='STAGE_SCOPE'


def test_new_source_rejection_stops_worker_before_stage_completion(tmp_path,monkeypatch):
    document,_=copy_metadata(tmp_path,launch=True);worker_environment(monkeypatch)
    monkeypatch.setattr(pc,'_input_binding',lambda root,candidate,plan:rc.receipt_index(root,document)[candidate['patient_group']]['record']['input_binding'])
    monkeypatch.setattr(pc,'_run_qualification',lambda *args,**kwargs:{'rejections':[{'failure_code':'SOURCE_HASH'}],'counts':{'rejected':1}})
    with pytest.raises(pc.QualificationError) as error:rc.run_worker(tmp_path,0)
    assert error.value.code=='SOURCE_REJECTION' and not (tmp_path/rc.OUTPUT/'stage-0.json').exists()


def test_supervisor_uses_three_fresh_sequential_children_and_no_repeat(tmp_path,monkeypatch):
    document,plan=copy_metadata(tmp_path,launch=True);worker_environment(monkeypatch)
    calls=fake_engine(monkeypatch,tmp_path,document,plan);children=[]
    monkeypatch.setattr(rc,'sample_resources',lambda *args:safe_sample())
    class Worker:
        returncode=0
        def __init__(self,pid):self.pid=pid
        def poll(self):return 0
    def popen(command,**kwargs):
        assert not children or children[-1].poll()==0
        assert kwargs['start_new_session'] and kwargs['env']['OMP_NUM_THREADS']=='2'
        index=int(command[-1]);rc.run_worker(tmp_path,index)
        worker=Worker(987654320+index);children.append(worker);return worker
    monkeypatch.setattr(rc.subprocess,'Popen',popen)
    result=rc.run(tmp_path)
    assert result['status']=='completed' and result['final_cached_verification']
    assert calls==[1,1,0] and len({p.pid for p in children})==3
    with pytest.raises(pc.QualificationError) as error:rc.run(tmp_path)
    assert error.value.code=='RETRY_NOT_RELEASED' and len(children)==3


@pytest.mark.parametrize('fault',['rss','time','worker','interrupt'])
def test_supervisor_stops_first_failure_or_interrupt_without_next_worker(tmp_path,monkeypatch,fault):
    copy_metadata(tmp_path,launch=True);launched=[];killed=[]
    class Worker:
        pid=987654321;returncode=None
        def poll(self):
            if fault=='worker':self.returncode=1
            return self.returncode
        def wait(self,timeout):self.returncode=-9;return -9
    worker=Worker()
    def popen(*args,**kwargs):launched.append(args);return worker
    def sample(*args):
        result=safe_sample()
        if len(args)==5:
            if fault=='rss':result['rss']=1610612737
            if fault=='time':result['elapsed']=601
            if fault=='interrupt':raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(rc.subprocess,'Popen',popen);monkeypatch.setattr(rc,'sample_resources',sample)
    monkeypatch.setattr(pc.os,'killpg',lambda *args:killed.append(args));monkeypatch.setattr(pc,'_input_binding',forbidden)
    with pytest.raises(KeyboardInterrupt if fault=='interrupt' else pc.QualificationError):rc.run(tmp_path)
    record=rc.read_record(tmp_path/rc.OUTPUT/'supervision.json')
    assert record['status']=='failed' and not record['complete_twelve_person_catalog']
    assert record['completed_stages']==[] and len(launched)==1 and record['automatic_retries']==0
    assert len(killed)==(0 if fault=='worker' else 1)
    with pytest.raises(pc.QualificationError) as error:rc.run(tmp_path)
    assert error.value.code=='RETRY_NOT_RELEASED' and len(launched)==1


def test_output_budget_combines_prior_core_metadata_with_recovery_metadata(tmp_path,monkeypatch):
    document,_=copy_metadata(tmp_path)
    monkeypatch.setattr(pc,'_execution_sample',lambda *args:{**safe_sample(),'output_bytes':7})
    expected=pc._execution_directory_bytes(tmp_path/pc.EXECUTION_OUTPUT)
    assert rc.sample_resources(None,tmp_path,document,0)['output_bytes']==expected+7
