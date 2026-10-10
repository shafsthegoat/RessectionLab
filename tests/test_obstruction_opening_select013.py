"""Fabricated public-manifest/native/checkpoint contracts; no acquired data.

Generated checkpoint endpoint metadata tests reload interfaces, not training.
No optimizer is constructed or stepped. Historical parity is optional only when
its separately pinned source copy is absent in a fresh checkout.
"""
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
from resectionlab.core import semantic_digest, thaw_json
from resectionlab import patient_planning_admission as admission
from resectionlab import public_patient_factory as factory
from resectionlab import patient_select_inference as inference
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_spatial_task import SUPPLIED_TUMOR_UNION_OCCUPANCY as UNION
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
from resectionlab.patient_planning_cohort_spec import TRAIN
import test_paired_train_occupancy as helper
import test_obstruction_opening_learning_admission as training


def lineage(p, method):
    return {'version':'frozen-TRAIN-checkpoint-lineage-v1', 'checkpoint_sha256':'a'*64,
        'method':method, 'training_release_sha256':'b'*64,
        'learning_protocol_hash':semantic_digest(p), 'initial_parameter_hash':'sha256:'+'c'*64,
        'parameter_hash':'sha256:'+'d'*64, 'architecture_hash':'sha256:'+'e'*64,
        'completed_updates':p['updates_per_method'],
        'training_context_hashes':{'ReMIND:'+s[-3:]:'sha256:'+'f'*64 for s in TRAIN},
        'public_target_context_variant':TARGET_CONTEXT, 'optimizer_updates_on_SELECT':0}


def build(path, inputs, p, *, checkpoint_lineage=None, **changes):
    path.mkdir()
    config=NominalCavityProposalConfig(**thaw_json(p['cohort_execution']['proposal_config']))
    limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':10000,
        'max_policy_forwards':96,'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,
        'search':thaw_json(p['cohort_execution']['search'])}
    options=dict(learning_protocol_hash=semantic_digest(p),proposal_config=config,
        public_target_context_variant=TARGET_CONTEXT,occupancy_condition=UNION,
        occupancy_inference_protocol=p,expected_role='SELECT',
        checkpoint_lineage=checkpoint_lineage or lineage(p,'IL' if p['updates_per_method']==64 else 'RL'))
    options.update(changes)
    return factory.prepare_public_source(path,{'limits':limits},'b'*64,None,lambda *a,**k:None,
        **inputs,**options)


@pytest.mark.parametrize('updates',[8,64])
def test_select013_exact_declared_union_context_no_training(tmp_path,updates):
    inputs,_=helper.fixture(tmp_path,subject='ReMIND-013',role='SELECT');p=training.protocol(updates)
    built=build(tmp_path/'out',inputs,p); task,context=training.admit(built,inputs)
    record=inference.require_select_context(context)
    assert record['execution_kind']==admission.UNION_SELECT_EXECUTION
    assert record['comparison_scope']=='same_declared_union_world_SELECT013_frozen_inference_only'
    assert record['max_optimizer_updates']==0 and record['policy_comparison_permitted'] is True
    assert record['derived_occupancy_anatomically_validated'] is False
    assert record['checkpoint_lineage']['completed_updates']==updates
    with pytest.raises(ValueError,match='TRAIN_gradient'):context.require_training()
    context.require_task(task.fresh());context.require_task(task.planning_clone())
    context.require_observations([task.observation()])
    assert task.step('STOP').terminated
    again=build(tmp_path/'again',inputs,p)
    assert built[0].source_hash==again[0].source_hash and built[1:]==again[1:]
    assert built[0].public_target_context_variant==TARGET_CONTEXT


@pytest.mark.parametrize('subject',('ReMIND-037','ReMIND-067',*TRAIN,'ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'))
def test_other_subjects_refused_before_payload(tmp_path,monkeypatch,subject):
    inputs,_=helper.fixture(tmp_path,subject=subject,role='SELECT')
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('must refuse before public payload'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,training.protocol())


@pytest.mark.parametrize('change',('condition','partial','training_flag','role','private_payload','unknown_coverage'))
def test_factory_refuses_wrong_condition_before_payload(tmp_path,monkeypatch,change):
    inputs,_=helper.fixture(tmp_path,subject='ReMIND-013',role='SELECT');p=training.protocol()
    options={}
    if change=='condition':options['occupancy_condition']='raw_cerebrum_baseline'
    elif change=='partial':options['occupancy_condition']=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY
    elif change=='training_flag':options['occupancy_learning_protocol']=p
    elif change=='role':options['expected_role']='TRAIN'
    else:
        path=inputs['public_manifest_path'];record=json.loads(path.read_text())
        if change=='private_payload':record['input_files']['ventricles']={'path':'never-read'}
        else:record['public_support_domain_fully_covered']=False
        path.write_text(json.dumps(record));inputs['public_manifest_sha256']=factory.sha(path)
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('must refuse before public payload'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,p,**options)


@pytest.mark.parametrize('change',('hash','horizon','config','initializer','updates','forwards','condition','private','QC','endpoint','method','protocol','search','variant'))
def test_direct_admission_rejects_changed_condition_before_inventory(tmp_path,monkeypatch,change):
    inputs,_=helper.fixture(tmp_path,subject='ReMIND-013',role='SELECT')
    built=build(tmp_path/'out',inputs,training.protocol());case,args=helper.contract(built,inputs);args=copy.deepcopy(args)
    plan=args['protocol']
    if change=='hash':plan['learning_protocol_hash']='sha256:'+'0'*64
    elif change=='horizon':plan['max_steps']=6
    elif change=='config':case=replace(case,proposal_config=replace(case.proposal_config,obstruction_opening=False))
    elif change=='initializer':plan['initialization']='fresh_seeded_shared_initialization'
    elif change=='updates':plan['max_optimizer_updates']=1
    elif change=='forwards':plan['max_policy_forwards']=0
    elif change=='condition':plan['occupancy_condition']='raw_cerebrum_baseline'
    elif change=='private':plan['private_reference_used']=True
    elif change=='QC':args['qc_receipt']['derived_occupancy_anatomically_validated']=True
    elif change=='endpoint':plan['checkpoint_lineage']['completed_updates']=7
    elif change=='method':plan['checkpoint_lineage']['method']='IL'
    elif change=='protocol':plan['occupancy_inference_protocol']['private_training_reward']=True
    elif change=='search':plan['search']['seconds']=299
    else:case=replace(case,public_target_context_variant=None,public_target_domain=None)
    helper.rebind(args)
    monkeypatch.setattr(admission,'NativeSpatialTask',lambda *a,**k:pytest.fail('must refuse before inventory'))
    with pytest.raises(ValueError):admission.make_patient_planning_task(case,**args)


@pytest.mark.parametrize('condition',[helper.RAW,helper.UNION])
def test_old_raw_and_search_records_have_exact_default_parity(tmp_path,monkeypatch,condition):
    repo=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
    baseline=Path(os.environ.get('SELECT013_BASELINE_DIR',repo/'build/obstruction-opening-select013-v1/baseline'))
    if not baseline.is_dir():pytest.skip('separately pinned historical source comparison unavailable')
    old={}
    for name in ('public_patient_factory','patient_planning_admission'):
        spec=importlib.util.spec_from_file_location('resectionlab._before_select013_'+name,baseline/(name+'.py'))
        module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module);old[name]=module
    inputs,_=helper.fixture(tmp_path)
    current=helper.build(tmp_path/'current',inputs,condition=condition)
    prior=helper.build(tmp_path/'prior',inputs,condition=condition,module=old['public_patient_factory'])
    assert current[0].source_hash==prior[0].source_hash and current[1:]==prior[1:]
    task,context=training.admit(current,inputs)
    case,args=helper.contract(prior,inputs);old_task,old_context=old['patient_planning_admission'].make_patient_planning_task(case,**args)
    assert context.record()==old_context.record()
    assert task.observation().fingerprint==old_task.observation().fingerprint
    assert task.candidate_inventory()==old_task.candidate_inventory()
    assert {p.name:p.read_bytes() for p in (tmp_path/'current').iterdir()}=={p.name:p.read_bytes() for p in (tmp_path/'prior').iterdir()}


def test_saved_training_protocol_default_round_trips():
    training.test_saved_prior_protocols_keep_exact_round_trip()


@pytest.mark.parametrize('updates',[8,64])
def test_generated_exact_union_checkpoint_reload_and_complete_inference(tmp_path,monkeypatch,updates):
    import torch
    from resectionlab.patient_planning_learning import common_patient_policies
    from resectionlab.patient_planning_cohort_io import OutputBudget,_save_checkpoint
    from resectionlab.spatial_policy import parameter_hash
    torch.set_num_threads(1)
    p=training.protocol(updates);method='IL' if updates==64 else 'RL';contexts=[]
    for subject in TRAIN:
        folder=tmp_path/subject;folder.mkdir();inputs,_=helper.fixture(folder,subject=subject)
        _,context=training.admit(training.build(folder/'source',inputs,p),inputs);contexts.append(context)
    il,rl=common_patient_policies(contexts,p);policy=il if method=='IL' else rl;initial=parameter_hash(policy)
    with torch.no_grad():
        for parameter in policy.parameters():parameter.zero_()
        policy.stop[-1].bias.fill_(5.)
    # Interface fixture: metadata endpoint is deliberately constructed, not trained.
    saved=_save_checkpoint(policy,method=method,contexts=contexts,protocol=p,updates=updates,
        initial_hash=initial,output=OutputBudget(tmp_path,16*1024**2),limits={'checkpoint_bytes':8*1024**2})
    pins={c.patient_group:c.fingerprint for c in contexts}
    checkpoint=inference.load_select_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=p,expected_context_hashes=pins,expected_method=method,training_release_sha256='b'*64)
    inputs,_=helper.fixture(tmp_path,subject='ReMIND-013',role='SELECT')
    task,context=training.admit(build(tmp_path/'SELECT',inputs,p,checkpoint_lineage=checkpoint.lineage_record()),inputs)
    checkpoint.require_task(task,context)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('no SELECT optimizer'))
    out=tmp_path/'greedy';out.mkdir()
    trace=inference.collect_select_greedy(task,context,checkpoint,output=out,guard=lambda:None)
    result=inference.seal_and_replay_select(task,context,trace,checkpoint,output=out,guard=lambda:None)
    assert [t.action_id for t in trace.transitions]==['STOP']
    assert result['independent_geometry']['accepted'] is True
    assert result['select_inference']['optimizer_updates_on_SELECT']==0
    assert result['select_inference']['checkpoint_lineage']['completed_updates']==updates
    assert all(not param.requires_grad and param.grad is None for param in checkpoint.policy.parameters())
    changed=context.record();changed.pop('occupancy_condition')
    wrong=admission.PatientPlanningContext(admission.freeze_json(changed),semantic_digest(changed))
    with pytest.raises(ValueError,match='frozen TRAIN condition'):checkpoint.require_task(task,wrong)
    with pytest.raises(ValueError):context.require_training()
    # Native greedy has its own honest completion record, and never uses weights.
    monkeypatch.setattr(checkpoint.policy,'forward',lambda *a,**k:pytest.fail('SEARCH cannot forward'))
    actions,accounting=task.observed_greedy_search(seconds=5)
    assert 'call_cap_reached' not in accounting
    out=tmp_path/'search';out.mkdir()
    trace=inference.collect_select_search(task,context,checkpoint,actions=actions,accounting=accounting,output=out,guard=lambda:None)
    result=inference.seal_and_replay_select(task,context,trace,checkpoint,output=out,guard=lambda:None)
    assert result['independent_geometry']['accepted'] is True and trace.method=='SEARCH'
    assert trace.behavior_parameter_hash is None and result['plan']['learning_updates']==0
    assert json.loads((out/'search-return.json').read_text())['accounting']==accounting


@pytest.mark.parametrize('change',('incomplete','one_step','count','scores','winner','time','horizon','return'))
def test_native_greedy_accounting_refuses_false_completion(change):
    from types import SimpleNamespace
    record={'method':'observed_greedy','complete':True,
        'objective_source':'permitted_nominal_target_and_frozen_geometric_costs',
        'initial_steps':0,'max_steps':24,'model_transition_calls':1,
        'native_replay_required':True,'global_optimality_proven':False,
        'planning_seconds':.1,'time_budget_seconds':5,'estimated_incremental_return':0.,
        'evaluated_nonstop_actions':1,'decisions':[{'step':0,'source_state_hash':'sha256:'+'0'*64,
            'legal_nonstop_actions':1,'scored_nonstop_actions':1,'all_current_legal_actions_scored':True,
            'scores':[{'action_id':'STOP','reward':0.},{'action_id':'move','reward':-.1}],
            'selected_action_id':'STOP'}]}
    if change=='incomplete':record['complete']=False
    elif change=='one_step':record['model_transition_calls']=0
    elif change=='count':record['evaluated_nonstop_actions']=2
    elif change=='scores':record['decisions'][0]['all_current_legal_actions_scored']=False
    elif change=='winner':record['decisions'][0]['scores'][1]['reward']=1.
    elif change=='time':record['planning_seconds']=6.
    elif change=='horizon':record['max_steps']=6
    else:record['estimated_incremental_return']=1.
    with pytest.raises(ValueError):inference._require_native_greedy_completion(SimpleNamespace(max_steps=24),('STOP',),record)


def test_generated_positive_native_greedy_motion_then_STOP_replays(tmp_path,monkeypatch):
    import numpy as np
    from resectionlab.core import array_digest
    from test_patient_select_inference import checkpoint_fixture,generated_case
    from test_patient_planning_admission import rebind
    checkpoint,_,_,p,_,_=checkpoint_fixture(tmp_path)
    case,args=generated_case('ReMIND-013','SELECT',p,checkpoint.lineage_record())
    # One exposed off-axis cell is the entire generated objective; after its
    # complete removal every further motion has strictly negative native cost.
    target=np.zeros(case.observed_support.shape,np.float32);target[5,5,1]=1.
    case=replace(case,nominal_target=target,reference_target=target)
    args['source_binding'].update(source_hash=case.source_hash,target_array_hash=array_digest(target))
    rebind(args);task,context=admission.make_patient_planning_task(case,**args)
    actions,accounting=task.observed_greedy_search(seconds=5)
    assert len(actions)==2 and actions[0]!='STOP' and actions[1]=='STOP'
    assert accounting['estimated_incremental_return']>0
    monkeypatch.setattr(checkpoint.policy,'forward',lambda *a,**k:pytest.fail('no SEARCH policy call'))
    destination=tmp_path/'native-greedy';destination.mkdir()
    trace=inference.collect_select_search(task,context,checkpoint,actions=actions,accounting=accounting,
        output=destination,guard=lambda:None)
    result=inference.seal_and_replay_select(task,context,trace,checkpoint,output=destination,guard=lambda:None)
    assert result['independent_geometry']['accepted'] is True
    assert trace.transitions[0].reward>0 and trace.transitions[1].reward==0.
    assert result['plan']['parameter_hash'] is None and result['plan']['learning_updates']==0
    assert task.metrics()['steps']==0
    # A validly shaped complete record still cannot claim another state/reward.
    for field in ('source_state_hash','reward'):
        altered=copy.deepcopy(accounting)
        if field=='source_state_hash':altered['decisions'][0][field]='sha256:'+'0'*64
        else:
            winner=next(r for r in altered['decisions'][0]['scores'] if r['action_id']==actions[0])
            winner['reward']+=1.;altered['estimated_incremental_return']+=1.
        out=tmp_path/field;out.mkdir()
        with pytest.raises(ValueError,match='state or legal inventory|selected reward'):
            inference.collect_select_search(task,context,checkpoint,actions=actions,accounting=altered,
                output=out,guard=lambda:None)
        assert not (out/'complete-trace.json').exists()
