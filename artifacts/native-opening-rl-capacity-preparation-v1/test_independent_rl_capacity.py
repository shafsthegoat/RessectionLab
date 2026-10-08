"""Saved-record/scalar/orchestration checks only; no task, forward or gradient."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import run_native_opening_rl_capacity as runner
FROZEN={
 'scripts/run_native_opening_rl_capacity.py':'8391c317dd7cf305747bde3c62de3f0b67d0c8b69e0ff0b196924c63f92e39e0',
 'tests/test_native_opening_rl_capacity.py':'4803c66efceea4de7f368e1f351f5dc6e6e992b489d278670714dc65a118e906',
 'manifests/experiments/native-opening-rl-capacity-v1.json':'14df1a1fbb9bb8187d97bc7dcf99cacf923a9e5e9f4182693de0e1d01a5e1285',
 'docs/native-opening-rl-capacity.md':'8cce154f0d415938ccb9102d6c5bd1004058123ede8680b4716756013c42d78b'}
def record():return json.loads((ROOT/'manifests/experiments/native-opening-rl-capacity-v1.json').read_text())
def saved_readout():return json.loads((ROOT/'artifacts/native-opening-bc-capacity-v1/result.json').read_text())['readouts']['0']

def test_exact_freeze_and_authenticated_original_inputs():
    for name,expected in FROZEN.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    decl=record();runner.validate(decl)
    assert len(decl['source_sha256'])==33 and len(decl['input_sha256'])==81
    bundle,tree=runner.load_evaluation_tree(decl)
    assert len(tree['routes'])==16 and len(tree['original_episodes'])==64 and len(tree['original_updates'])==16
    assert decl['lineage']['real_patient_count']==0 and decl['lineage']['human_roles_opened']==[]

def test_saved_initial_probabilities_reproduce_scalar_expectation_without_forward():
    _,tree=runner.load_evaluation_tree(record())
    result=runner.expected_metrics(saved_readout(),tree)
    assert result['nominal_expected_return']==pytest.approx(-.2754633654,abs=1e-10)
    assert result['generated_any_target_probability']==pytest.approx(.235496,abs=1e-6)
    assert result['terminal_probability_sum']==pytest.approx(1.,abs=1e-12)

@pytest.mark.parametrize('mode',['STOP','teacher'])
def test_degenerate_probability_tree_has_exact_known_return(mode):
    bundle,tree=runner.load_evaluation_tree(record())
    rows=[]
    for key,ids in tree['state_actions'].items():
        action='STOP' if mode=='STOP' else bundle['rows'][key]['selected_action_id']
        rows.append({'observation_hash':key,'action_ids':ids,'probabilities':[float(a==action) for a in ids]})
    result=runner.expected_metrics({'states':rows},tree)
    assert result['terminal_probability_sum']==pytest.approx(1.,abs=1e-12)
    assert result['nominal_expected_return']==pytest.approx(0. if mode=='STOP' else 1.1,abs=1e-12)
    assert result['generated_full_target_probability']==float(mode=='teacher')
    assert result['expected_normal_mm3']==(0. if mode=='STOP' else 4.)

@pytest.mark.parametrize('field,value',[('observation_hash','wrong'),('action_mask',[False]),('logits',[99.]),('value',99.)])
def test_first16_full_numerical_signature_rejects_changes(field,value):
    _,tree=runner.load_evaluation_tree(record())
    report=copy.deepcopy(tree['original_episodes']['rl-00-0.json'])
    report['decisions'][0][field]=value
    with pytest.raises(ValueError,match='sampled observation/action/reward'):
        runner.verify_original_episode(1,0,report,tree)

def test_existing_output_preserved_and_validation_does_not_execute(monkeypatch,tmp_path):
    marker=tmp_path/'retained.json';marker.write_text('preserve\n')
    monkeypatch.setattr(runner,'worker',lambda *a,**kw:pytest.fail('Worker forbidden'))
    monkeypatch.setattr(runner.io,'supervise_worker',lambda *a,**kw:pytest.fail('Supervisor forbidden'))
    arguments=[runner.SCRIPT,'--declaration',str(ROOT/'manifests/experiments/native-opening-rl-capacity-v1.json')]
    monkeypatch.setattr(sys,'argv',arguments);runner.main()
    monkeypatch.setattr(sys,'argv',arguments+['--execute','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='new output directory'):runner.main()
    assert marker.read_text()=='preserve\n'

def stub_worker(monkeypatch,tmp_path,*,reject=None,assessment_offset=0.):
    import torch
    import resectionlab.native_spatial_task as tasks
    import resectionlab.spatial_policy as policy
    import resectionlab.spatial_policy_diagnostics as diagnostics
    from resectionlab.data_policy import GeneratedDevelopmentContext
    bundle,tree=runner.load_evaluation_tree(record())
    # Vary only privileged assessment outputs, keeping actor observations and
    # sampled old histories fixed. Nothing executes these nominal rewards.
    tree=copy.deepcopy(tree)
    for route in tree['routes']:route['nominal_return']+=assessment_offset
    base=SimpleNamespace(case=SimpleNamespace(source_hash='sha256:'+'1'*64),decision_model_hash='sha256:'+'2'*64)
    counts={'optimizer':0,'episodes':0,'loss':0,'gradient':0};fresh=[]
    class Profiler:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def snapshot(self):return {}
    class Model:
        architecture_hash='orchestration-stub-only'
        def parameters(self):return ()
        def state_dict(self):return {}
        def architecture_record(self):return {}
    class Adam:
        def __init__(self,*a,**kw):counts['optimizer']+=1
    def guarded(base,model,output,name,guard,**kw):
        if name=='initial':
            original=json.loads((runner.SAVED/'initial.json').read_text())
            accepted=reject!='initial'
        else:
            index=counts['episodes'];counts['episodes']+=1
            original=tree['original_episodes'][f'rl-00-{index}.json']
            accepted=reject!=index
            assert kw['mode']=='sample' and isinstance(kw['generator'],torch.Generator)
        report=copy.deepcopy(original)
        report['independent_evaluation']['accepted']=accepted
        report['guarded_online_seconds']=.01
        report['independent_evaluation_seconds']=.001
        runner.io.write_json(output/(name+'-cost.json'),{'online':{'native_preview_entries':12}})
        episode=[SimpleNamespace(terminated=i==len(report['decisions'])-1) for i in range(len(report['decisions']))]
        if name!='initial':fresh.append(episode)
        return episode,report,None
    def loss(model,batch,**kw):
        counts['loss']+=1
        assert len(batch)==len(fresh)==4 and all(a is b for a,b in zip(batch,fresh))
        assert kw['gamma']==1. and kw['entropy_weight']==.01 and kw['value_weight']==.5
        assert set(kw)=={'gamma','entropy_weight','value_weight','learning_context'}
        raise RuntimeError('CONFIRMED_FRESH_ONLY_LOSS_INPUTS')
    def gradient(*a,**kw):
        counts['gradient']+=1;pytest.fail('No actual or stub update requested')
    monkeypatch.setattr(runner,'validate',lambda _:None)
    monkeypatch.setattr(runner,'load_evaluation_tree',lambda _:(bundle,tree))
    monkeypatch.setattr(tasks,'make_native_opening_task',lambda **kw:base)
    monkeypatch.setattr(diagnostics,'NativePreviewProfiler',Profiler)
    monkeypatch.setattr(runner.evaluation,'reconstruct_teacher',lambda *a:(('evaluation-only-sentinel',),{}))
    monkeypatch.setattr(runner.evaluation,'readout',lambda *a:copy.deepcopy(saved_readout()))
    monkeypatch.setattr(policy,'SpatialPolicy',lambda *a:Model())
    monkeypatch.setattr(policy,'parameter_hash',lambda _:runner.INITIAL_HASH)
    monkeypatch.setattr(policy,'reinforce_loss',loss)
    monkeypatch.setattr(policy,'gradient_step',gradient)
    monkeypatch.setattr(torch.optim,'Adam',Adam)
    monkeypatch.setattr(torch,'save',lambda *a,**kw:None)
    monkeypatch.setattr(GeneratedDevelopmentContext,'require_task',lambda *a:None)
    monkeypatch.setattr(runner.prior,'guarded_episode',guarded)
    return counts

def test_failed_initial_audit_stops_before_optimizer(monkeypatch,tmp_path):
    counts=stub_worker(monkeypatch,tmp_path,reject='initial')
    with pytest.raises(ValueError,match='inference audit refused'):
        runner.worker(record(),tmp_path,declaration_sha256='3'*64)
    saved=json.loads((tmp_path/'result.json').read_text())
    assert counts=={'optimizer':0,'episodes':0,'loss':0,'gradient':0}
    assert saved['updates']==0 and saved['status']=='failed'
    assert saved['methods']['initial']['outcomes'] is None and saved['methods']['RL256']['outcomes'] is None

def test_failed_second_collection_preserves_first_and_refuses_loss(monkeypatch,tmp_path):
    counts=stub_worker(monkeypatch,tmp_path,reject=1)
    with pytest.raises(ValueError,match='Complete independently audited'):
        runner.worker(record(),tmp_path,declaration_sha256='3'*64)
    saved=json.loads((tmp_path/'result.json').read_text())
    assert counts=={'optimizer':1,'episodes':2,'loss':0,'gradient':0}
    assert saved['updates']==0 and saved['completed_training_episodes']==1
    active=saved['active_update']['episodes']
    assert active[0]['status']=='complete' and active[0]['outcomes'] is not None
    assert active[1]['status']=='failed' and active[1]['outcomes'] is None
    assert all(r['status']=='not_started' and r['outcomes'] is None for r in active[2:])

@pytest.mark.parametrize('assessment_offset',[0.,1000.])
def test_privileged_assessment_does_not_enter_fresh_loss(monkeypatch,tmp_path,assessment_offset):
    counts=stub_worker(monkeypatch,tmp_path,assessment_offset=assessment_offset)
    with pytest.raises(RuntimeError,match='CONFIRMED_FRESH_ONLY_LOSS_INPUTS'):
        runner.worker(record(),tmp_path,declaration_sha256='3'*64)
    saved=json.loads((tmp_path/'result.json').read_text())
    assert counts=={'optimizer':1,'episodes':4,'loss':1,'gradient':0}
    assert saved['updates']==0 and saved['completed_training_episodes']==4 and saved['training_transitions']==7
    assert all(e['status']=='complete' for e in saved['active_update']['episodes'])
    assert saved['methods']['RL256']['outcomes'] is None
