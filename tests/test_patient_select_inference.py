"""Generated weight-format/native controls only; source prepared, not patient runs.

Checkpoint fixtures contain deliberately constructed tensors and claimed endpoint
metadata solely to exercise loader interfaces. They are not evidence of training.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from resectionlab.core import array_digest, freeze_json, semantic_digest, thaw_json
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.native_proposals import NominalCavityProposalConfig, SUPPLIED_GOAL_REGION
from resectionlab.observed_search import observed_beam_search
from resectionlab.patient_planning_admission import (PatientPlanningContext,
    COHORT_SHA256, SELECT_INITIALIZATION, _public_observation_binding,
    make_patient_planning_task, validate_select_checkpoint_lineage)
from resectionlab.patient_planning_cohort_io import (TRAIN, OutputBudget, _save_checkpoint,
    cohort_learning_protocol)
from resectionlab.patient_planning_learning import (PatientTrainSession,
    admit_collected_trace, common_patient_policies)
from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol
from resectionlab.patient_select_inference import (FrozenSelectCheckpoint, load_select_checkpoint,
    collect_select_greedy, collect_select_search, seal_and_replay_select, require_select_context)
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
from resectionlab.spatial_policy import parameter_hash
from test_patient_planning_admission import fixture_contract, rebind


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def generated_case(subject, role, protocol, lineage=None):
    _, args = fixture_contract(subject, role)
    case = make_native_opening_task().case
    args['source_binding'].update(source_hash=case.source_hash,
        image_array_hash=array_digest(case.structural_intensity),
        support_array_hash=array_digest(case.observed_support),
        target_array_hash=array_digest(case.nominal_target), affine_array_hash=array_digest(case.affine_ras_mm))
    args['protocol']['learning_protocol_hash'] = semantic_digest(protocol)
    if lineage is not None:
        args['protocol'].update(initialization=SELECT_INITIALIZATION, checkpoint_lineage=lineage)
    rebind(args)
    return case, args


def checkpoint_fixture(tmp_path, *, opening=False, method='IL'):
    protocol = cohort_learning_protocol(1); contexts = []
    for subject in TRAIN:
        case, args = generated_case(subject, 'TRAIN', protocol)
        _, context = make_patient_planning_task(case, **args)
        contexts.append(context)
    il, rl = common_patient_policies(tuple(contexts), protocol)
    policy = il if method == 'IL' else rl; initial = parameter_hash(policy)
    # Deterministic generated weights make STOP or the first legal action greedy.
    # No loss or optimizer is run, and no acquired checkpoint is used.
    with torch.no_grad():
        for parameter in policy.parameters(): parameter.zero_()
        (policy.actor if opening else policy.stop)[-1].bias.fill_(5.)
    sink = OutputBudget(tmp_path, 16*1024**2)
    saved = _save_checkpoint(policy, method=method, contexts=contexts, protocol=protocol,
        updates=1, initial_hash=initial, output=sink, limits={'checkpoint_bytes':8*1024**2})
    pins = {c.patient_group:c.fingerprint for c in contexts}
    loaded = load_select_checkpoint(tmp_path/saved['path'], expected_sha256=saved['sha256'],
        expected_learning_protocol=protocol, expected_context_hashes=pins,
        expected_method=method, training_release_sha256='a'*64)
    return loaded, policy, tuple(contexts), protocol, saved, pins


@pytest.mark.parametrize('method', ['IL', 'RL'])
@pytest.mark.parametrize('opening', [False, True])
def test_generated_reloaded_greedy_complete_native_replay(tmp_path, method, opening):
    checkpoint, original, _, protocol, _, _ = checkpoint_fixture(tmp_path, method=method, opening=opening)
    assert all(torch.equal(a,b) for a,b in zip(original.parameters(), checkpoint.policy.parameters()))
    assert all(not p.requires_grad and p.grad is None for p in checkpoint.policy.parameters())
    case, args = generated_case('ReMIND-013', 'SELECT', protocol, checkpoint.lineage_record())
    task, context = make_patient_planning_task(case, **args)
    require_select_context(context)
    assert context.record()['initialization'] == SELECT_INITIALIZATION
    assert context.record()['checkpoint_lineage']['training_release_sha256'] == 'a'*64
    before = parameter_hash(checkpoint.policy); destination = tmp_path/'inference'; destination.mkdir()
    trace = collect_select_greedy(task, context, checkpoint, output=destination, guard=lambda:None)
    result = seal_and_replay_select(task, context, trace, checkpoint, output=destination, guard=lambda:None)
    assert parameter_hash(checkpoint.policy) == before
    assert result['independent_geometry']['accepted'] is True
    assert result['plan_seal'] == semantic_digest(result['plan'])
    assert result['plan']['actions'] == tuple(t.action_id for t in trace.transitions)
    assert result['select_inference']['optimizer_updates_on_SELECT'] == 0
    assert result['select_inference']['EVAL_opened'] is False
    assert result['select_inference']['checkpoint_lineage'] == freeze_json(checkpoint.lineage_record())
    assert json.loads((destination/'complete-trace.json').read_text())['trace_seal'] == trace.seal_hash
    if opening:
        assert trace.transitions[0].action_id != 'STOP'
        assert len(trace.transitions) == context.max_steps or trace.transitions[-1].action_id == 'STOP'
    else:
        assert [t.action_id for t in trace.transitions] == ['STOP']


def test_inference_never_grants_training_authority(tmp_path, monkeypatch):
    checkpoint, _, train_contexts, protocol, _, _ = checkpoint_fixture(tmp_path)
    case, args = generated_case('ReMIND-037', 'SELECT', protocol, checkpoint.lineage_record())
    task, context = make_patient_planning_task(case, **args)
    destination = tmp_path/'inference'; destination.mkdir()
    trace = collect_select_greedy(task, context, checkpoint, output=destination, guard=lambda:None)
    def no_optimizer(*args, **kwargs): raise AssertionError('An optimizer must never be constructed')
    monkeypatch.setattr(torch.optim, 'Adam', no_optimizer)
    with pytest.raises(ValueError, match='TRAIN_gradient'): context.require_training()
    with pytest.raises(ValueError, match='TRAIN_gradient'):
        common_patient_policies((context,), protocol)
    with pytest.raises(ValueError, match='TRAIN_gradient'):
        admit_collected_trace(context, trace.transitions, task.planning_clone())
    with pytest.raises(ValueError, match='TRAIN_gradient'):
        PatientTrainSession((context,), 'IL', checkpoint.policy,
            initial_parameter_hash=checkpoint._lineage['initial_parameter_hash'], protocol=protocol)
    with pytest.raises(ValueError, match='fresh common initialization'):
        PatientTrainSession(train_contexts, 'IL', checkpoint.policy,
            initial_parameter_hash=checkpoint._lineage['initial_parameter_hash'], protocol=protocol)


@pytest.mark.parametrize('change', ['TRAIN','EVAL','other_SELECT','updates','missing_lineage','wrong_variant','wrong_protocol'])
def test_select_admission_refuses_role_budget_or_lineage_change(tmp_path, change):
    checkpoint, _, _, protocol, _, _ = checkpoint_fixture(tmp_path)
    subject, role = ('ReMIND-008','TRAIN') if change=='TRAIN' else (
        ('ReMIND-067','MEASUREMENT_EVAL') if change=='EVAL' else (
        ('ReMIND-014','SELECT') if change=='other_SELECT' else ('ReMIND-013','SELECT')))
    case, args = generated_case(subject, role, protocol, checkpoint.lineage_record())
    if change=='updates': args['protocol']['max_optimizer_updates']=1
    elif change=='missing_lineage': args['protocol'].pop('checkpoint_lineage')
    elif change=='wrong_variant':
        args['protocol']['checkpoint_lineage']['public_target_context_variant']='full-supplied-public-target-context-v1'
    elif change=='wrong_protocol': args['protocol']['learning_protocol_hash']='sha256:'+'9'*64
    with pytest.raises(ValueError): make_patient_planning_task(case, **args)


def test_context_and_checkpoint_lineage_are_immutable_snapshots(tmp_path):
    checkpoint, _, _, protocol, _, _ = checkpoint_fixture(tmp_path)
    lineage=checkpoint.lineage_record();case,args=generated_case('ReMIND-013','SELECT',protocol,lineage)
    _,context=make_patient_planning_task(case,**args);before=context.fingerprint
    lineage['method']='RL';lineage['training_context_hashes']['ReMIND:008']='sha256:'+'8'*64
    assert context.fingerprint==before
    checkpoint.require(context)
    with pytest.raises(ValueError): validate_select_checkpoint_lineage({**checkpoint.lineage_record(),'private_reference':'forbidden'})


@pytest.mark.parametrize('change', ['weights','grad_enabled','train_mode'])
def test_mutable_weight_state_refused_before_any_forward(tmp_path, monkeypatch, change):
    checkpoint, _, _, protocol, _, _ = checkpoint_fixture(tmp_path)
    case,args=generated_case('ReMIND-013','SELECT',protocol,checkpoint.lineage_record())
    task,context=make_patient_planning_task(case,**args)
    if change=='weights':
        with torch.no_grad():next(checkpoint.policy.parameters()).add_(1.)
    elif change=='grad_enabled':next(checkpoint.policy.parameters()).requires_grad_(True)
    else:checkpoint.policy.train()
    monkeypatch.setattr(checkpoint.policy,'forward',lambda *args:(_ for _ in ()).throw(AssertionError('forward must not run')))
    destination=tmp_path/'inference';destination.mkdir()
    with pytest.raises(ValueError,match='Frozen SELECT'):
        collect_select_greedy(task,context,checkpoint,output=destination,guard=lambda:None)
    assert list(destination.iterdir())==[]


def test_incomplete_or_tampered_trace_and_replay_mutation_refused(tmp_path):
    checkpoint,_,_,protocol,_,_=checkpoint_fixture(tmp_path,opening=True)
    case,args=generated_case('ReMIND-013','SELECT',protocol,checkpoint.lineage_record())
    task,context=make_patient_planning_task(case,**args);destination=tmp_path/'inference';destination.mkdir()
    trace=collect_select_greedy(task,context,checkpoint,output=destination,guard=lambda:None)
    with pytest.raises(ValueError):replace(trace,transitions=trace.transitions[:-1]).require()
    changed=thaw_json(trace.history);changed[0]['reward']+=1.
    with pytest.raises(ValueError):replace(trace,history=freeze_json(changed)).require()
    def mutate():
        with torch.no_grad():next(checkpoint.policy.parameters()).add_(1.)
    with pytest.raises(ValueError,match='Frozen SELECT'):
        seal_and_replay_select(task,context,trace,checkpoint,output=destination,guard=mutate)
    assert not (destination/'select-replay.json').exists()


def test_loader_wrong_pin_refuses_and_default_admission_has_no_new_fields(tmp_path):
    checkpoint,original,_,protocol,saved,pins=checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError,match='bounded checkpoint loader'):
        FrozenSelectCheckpoint(original,checkpoint.lineage_record(),protocol,object())
    with pytest.raises(ValueError,match='bytes/hash'):
        load_select_checkpoint(tmp_path/saved['path'],expected_sha256='f'*64,
            expected_learning_protocol=protocol,expected_context_hashes=pins,
            expected_method='IL',training_release_sha256='a'*64)
    path=tmp_path/saved['path'];path.write_bytes(path.read_bytes()+b'changed-generated-checkpoint')
    with pytest.raises(ValueError,match='bytes/hash'):
        load_select_checkpoint(path,expected_sha256=saved['sha256'],
            expected_learning_protocol=protocol,expected_context_hashes=pins,
            expected_method='IL',training_release_sha256='a'*64)
    case,args=generated_case('ReMIND-008','TRAIN',protocol)
    _,context=make_patient_planning_task(case,**args)
    assert 'checkpoint_lineage' not in context.record() and 'initialization' not in context.record()
    context.require_training()


def test_generated_returned_search_plan_replays_without_checkpoint_authorship(tmp_path, monkeypatch):
    checkpoint, _, _, protocol, _, _ = checkpoint_fixture(tmp_path)
    case, args = generated_case('ReMIND-037', 'SELECT', protocol, checkpoint.lineage_record())
    # This small control runs the unchanged search on fabricated geometry only.
    args['protocol']['search'] = {'max_calls':512, 'beam_width':2, 'seconds':5}
    task, context = make_patient_planning_task(case, **args)
    def no_forward(*args, **kwargs): raise AssertionError('SEARCH cannot call the loaded actor')
    monkeypatch.setattr(checkpoint.policy, 'forward', no_forward)
    actions, accounting = observed_beam_search(task, **context.record()['budgets']['search'],
        objective_source='supplied_public_whole_tumor_and_frozen_geometric_costs',
        transition_mode='lazy_planning')
    assert accounting['call_cap_reached'] is False and accounting['time_cap_reached'] is False
    destination = tmp_path/'SEARCH'; destination.mkdir()
    trace = collect_select_search(task, context, checkpoint, actions=actions, accounting=accounting,
        output=destination, guard=lambda:None)
    result = seal_and_replay_select(task, context, trace, checkpoint, output=destination, guard=lambda:None)
    assert result['method'] == trace.method == 'SEARCH'
    assert trace.behavior_parameter_hash is None
    assert result['plan']['actions'] == tuple(actions)
    assert result['plan']['parameter_hash'] is None and result['plan']['architecture_hash'] is None
    assert result['plan']['learning_updates'] == 0
    assert result['independent_geometry']['accepted'] is True
    assert result['select_inference']['checkpoint_lineage_scope'] == 'comparison_reference_only_not_action_author'
    assert result['select_inference']['optimizer_updates_on_SELECT'] == 0
    assert json.loads((destination/'search-return.json').read_text())['actions'] == list(actions)
    checkpoint.require(context)


@pytest.mark.parametrize('failure', ['capped','incomplete','after_STOP'])
def test_search_rejects_capped_or_incomplete_return(tmp_path, failure):
    checkpoint, _, _, protocol, _, _ = checkpoint_fixture(tmp_path)
    case, args = generated_case('ReMIND-013', 'SELECT', protocol, checkpoint.lineage_record())
    task, context = make_patient_planning_task(case, **args)
    observation = task.observation()
    first = next(a for a, legal in zip(observation.action_ids[1:], observation.action_mask[1:]) if legal)
    actions = (first,) if failure == 'incomplete' else ('STOP','STOP') if failure == 'after_STOP' else ('STOP',)
    destination = tmp_path/'SEARCH'; destination.mkdir()
    with pytest.raises(ValueError, match='uncapped|Incomplete|after termination'):
        collect_select_search(task, context, checkpoint, actions=actions,
            accounting={'call_cap_reached':failure == 'capped','time_cap_reached':False},
            output=destination, guard=lambda:None)
    assert not (destination/'complete-trace.json').exists()


def generated_sequential_context(task, protocol, subject, role, lineage=None):
    """Fabricated DTO binding; not acquired-source QC or factory qualification.

The real native task exercises protocol/task/observation and reload seams. The
separate tests above exercise admission; no synthetic array is labelled acquired.
"""
    record = {'scope':'patient_native_planning_experiment', 'subject':subject,
        'patient_group':'ReMIND:'+subject[-3:], 'role':role, 'cohort_sha256':COHORT_SHA256,
        'evidence_domain':'generated_interface_control', 'real_patient_count':0,
        'source_hash':task.case.source_hash, 'decision_model_hash':task.decision_model_hash,
        'max_steps':task.max_steps, 'observation_track':task.case.track,
        'public_observation_binding':_public_observation_binding(task.observation()),
        'protocol_sha256':semantic_digest({'generated_only':True,'role':role,'subject':subject}),
        'learning_protocol_hash':semantic_digest(protocol),
        'public_target_context_variant':TARGET_CONTEXT,
        'max_optimizer_updates':16 if role == 'TRAIN' else 0,
        'budgets':{'search':thaw_json(protocol['cohort_execution']['search'])},
        'private_reference_in_task':False}
    if lineage is not None:
        record.update(initialization=SELECT_INITIALIZATION, checkpoint_lineage=lineage)
    context = PatientPlanningContext(freeze_json(record), semantic_digest(record))
    context.require_task(task); context.require_observations((task.observation(),))
    return context


def test_actual_sequential_global_context_checkpoint_condition_smoke(tmp_path):
    config = NominalCavityProposalConfig(max_candidates=120, intermediate_opening_mm=1.,
        tool_footprint_opening=True)
    protocol = sequential_learning_protocol(updates=8, max_steps=24,
        search={'max_calls':5640,'beam_width':2,'seconds':300}, proposal_config=config,
        retention_mode='return_plus_opening_depth_volume_v1')
    original_case = make_native_opening_task().case
    case = replace(original_case, track='annotation_assisted', support_source_kind='supplied_annotation',
        target_source_kind='supplied_annotation', proposal_mode='nominal_cavity_v1', proposal_config=config,
        target_semantics=SUPPLIED_GOAL_REGION, public_target_context_variant=TARGET_CONTEXT,
        public_target_domain=np.ones(original_case.observed_support.shape, bool), crop_shape=(9,9,7))
    task = NativeSpatialTask(case, max_steps=24)
    contexts = tuple(generated_sequential_context(task,protocol,s,'TRAIN') for s in TRAIN)
    il, _ = common_patient_policies(contexts, protocol); initial = parameter_hash(il)
    # Generated endpoint-format fixture only; eight metadata updates are not a
    # claim that this test trained a model. One deterministic STOP is replayed.
    with torch.no_grad():
        for parameter in il.parameters(): parameter.zero_()
        il.stop[-1].bias.fill_(5.)
    saved = _save_checkpoint(il,method='IL',contexts=contexts,protocol=protocol,updates=8,
        initial_hash=initial,output=OutputBudget(tmp_path,16*1024**2),limits={'checkpoint_bytes':8*1024**2})
    checkpoint = load_select_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=protocol,expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},
        expected_method='IL',training_release_sha256='a'*64)
    assert checkpoint.policy.public_target_context_variant == TARGET_CONTEXT
    assert parameter_hash(checkpoint.policy) == parameter_hash(il)
    context = generated_sequential_context(task,protocol,'ReMIND-013','SELECT',checkpoint.lineage_record())
    checkpoint.require_task(task,context)
    destination = tmp_path/'sequential-SELECT'; destination.mkdir()
    trace = collect_select_greedy(task,context,checkpoint,output=destination,guard=lambda:None)
    result = seal_and_replay_select(task,context,trace,checkpoint,output=destination,guard=lambda:None)
    assert result['plan']['actions'] == ('STOP',) and result['plan']['learning_updates'] == 8
    assert trace.transitions[0].observation.public_target_context is not None
    assert result['independent_geometry']['accepted'] is True
    assert result['select_inference']['optimizer_updates_on_SELECT'] == 0
    # Rebind each changed generated world so require_task tests the checkpoint
    # condition itself, rather than only detecting a stale context hash.
    changed_horizon = NativeSpatialTask(case,max_steps=23)
    changed_config = replace(config,max_candidates=96)
    changed_proposals = NativeSpatialTask(replace(case,proposal_config=changed_config),max_steps=24)
    for changed in (changed_horizon,changed_proposals):
        changed_context = generated_sequential_context(changed,protocol,'ReMIND-013','SELECT',checkpoint.lineage_record())
        with pytest.raises(ValueError,match='frozen TRAIN condition'):
            checkpoint.require_task(changed,changed_context)
    record = context.record(); record['budgets']['search']['seconds'] = 299
    changed_context = PatientPlanningContext(freeze_json(record),semantic_digest(record))
    with pytest.raises(ValueError,match='frozen TRAIN condition'):
        checkpoint.require_task(task,changed_context)
