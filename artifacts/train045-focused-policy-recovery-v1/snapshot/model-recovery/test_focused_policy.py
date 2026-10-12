"""Generated native packet replay and frozen actor controls; no saved payloads."""
from dataclasses import replace
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(HERE.parent/'model'));sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'tests'))
import numpy as np
import pytest
import torch
from resectionlab.core import freeze_json,thaw_json,semantic_digest
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.patient_planning_admission import PatientPlanningContext,COHORT_SHA256,_public_observation_binding
from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash
from resectionlab.spatial_continuation_value import _FrozenPolicy,_world,TRAIN_SUBJECTS
from resectionlab.spatial_continuation_tail import FittedContinuationTail,VERSION,TARGET,PREDICTION_SEMANTICS,DEPLOYMENT
from test_obstruction_opening_proposals import source
import focused_policy as adapter
assert Path(adapter.__file__).resolve().parent == HERE, 'Recovery controls require recovered helper bytes'
from saved_tail import load_saved_tail


def generated_fixture():
    """One REAL old/base and focused physical transition, then admitted live clones.

    Smaller prefix is explicit generated evidence. Production runner requires12.
    The fixture deliberately has blocked base rays and legal extra-page cuts.
    """
    case=source(enabled=False,cap=120,extra=((2,9,9),))
    old=NativeSpatialTask(case,max_steps=24)
    observation=old.observation()
    record=freeze_json({'scope':'patient_native_planning_experiment','role':'TRAIN',
        'cohort_sha256':COHORT_SHA256,'subject':'ReMIND-045','patient_group':'ReMIND:045',
        'evidence_domain':'generated_interface_control','source_hash':case.source_hash,
        'decision_model_hash':old.decision_model_hash,'max_optimizer_updates':64,'max_steps':24,
        'observation_track':observation.track,'public_observation_binding':_public_observation_binding(observation)})
    context=PatientPlanningContext(record,semantic_digest(record))
    action=next(iter(old._prepare_inventory()))
    expected=old.clone().step(action).info
    template,binding=adapter.protocol.enabled_task(old);focused=template.planning_clone()
    packets,frontier,joins=adapter.protocol.map_prefix(old,focused,[action],adapter.protocol.compact_history([expected]))
    admission=adapter.admit_focused_context(old,context,focused,packets,frontier,expected_prefix_actions=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        policy=SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2,2),hidden_features=4,
            ray_samples=2,critic_candidate_context=True)).eval()
    for p in policy.parameters():p.requires_grad_(False)
    return dict(original=old,context=context,focused=focused,template=template,packets=packets,
        frontier=frontier,admission=admission,binding=binding,joins=joins,policy=policy)


def generated_tail(fixture,constant=0.):
    policy=fixture['policy'];head=copy.deepcopy(policy.actor)
    with torch.no_grad():
        for p in head.parameters():p.zero_()
        head[2].bias.fill_(constant)
    head.eval()
    for p in head.parameters():p.requires_grad_(False);p.grad=None
    world=_world(fixture['context'],training=False)
    worlds=freeze_json({subject:{**thaw_json(world),'subject':subject} for subject in TRAIN_SUBJECTS})
    report=freeze_json({'version':VERSION,'target':TARGET,'prediction_semantics':PREDICTION_SEMANTICS,
        'deployment':DEPLOYMENT,'base_parameter_hash':parameter_hash(policy),'architecture_hash':policy.architecture_hash,
        'fitted_head_hash':parameter_hash(head),'frozen_hidden_hash':parameter_hash(head[0]),
        'corpus_seal':semantic_digest({'generated_corpus_only':True})})
    return FittedContinuationTail(_FrozenPolicy(policy),head,worlds,report,max_scores=24,seconds=120,cancelled=None)


def archive_for(tail):
    entries=[];payloads=[]
    for i,(name,tensor) in enumerate(sorted(tail._head.state_dict().items())):
        stream=io.BytesIO();np.lib.format.write_array(stream,tensor.detach().numpy(),allow_pickle=False)
        raw=stream.getvalue();member='head/%03d.npy'%i
        entries.append({'name':name,'member':member,'shape':list(tensor.shape),'sha256':hashlib.sha256(raw).hexdigest()})
        payloads.append((member,raw))
    expected={'schema':'logged-continuation-head-only-v2','method':'TAIL','objective':TARGET,
        'prediction_semantics':PREDICTION_SEMANTICS,'base':{'generated_checkpoint_pin':'not_acquired'},
        'head_hash':tail._head_hash,'corpus_seal':tail.report['corpus_seal']}
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('manifest.json',json.dumps({**expected,'entries':entries},sort_keys=True))
        for name,raw in payloads:z.writestr(name,raw)
    raw=stream.getvalue()
    options=dict(expected_sha256=hashlib.sha256(raw).hexdigest(),expected_manifest=expected,
        tail_report=tail.report,expected_report_hash=semantic_digest(tail.report),worlds=tail._worlds,
        expected_worlds_hash=semantic_digest(tail._worlds),max_scores=24,seconds=120)
    return raw,options


def test_saved_only_head_reload_exact_old_world_and_external_pin_refusals():
    fixture=generated_fixture();tail=generated_tail(fixture);raw,options=archive_for(tail)
    loaded=load_saved_tail(fixture['policy'],raw,**options)
    assert loaded._head_hash==tail._head_hash and loaded.report==tail.report and loaded._worlds==tail._worlds
    assert loaded._guard.policy is fixture['policy']
    for bad in (raw+b'x',raw[:-1],b'x'*131073):
        with pytest.raises(ValueError):load_saved_tail(fixture['policy'],bad,**options)
    for key,value in [('expected_report_hash','bad'),('expected_worlds_hash','bad'),
                      ('expected_manifest',{**options['expected_manifest'],'method':'Q64'})]:
        with pytest.raises(ValueError):load_saved_tail(fixture['policy'],raw,**{**options,key:value})
    stream=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src,zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():dst.writestr(item.filename,src.read(item.filename))
    compressed=stream.getvalue()
    with pytest.raises(ValueError):load_saved_tail(fixture['policy'],compressed,
        **{**options,'expected_sha256':hashlib.sha256(compressed).hexdigest()})


def test_union_extra_certificate_packet_replay_and_legacy_context_refusal():
    f=generated_fixture();catalog=adapter.collect_catalog(f['focused'].clone(),f['admission'])
    assert len(catalog.pages)>1 and any(owner>0 for owner in catalog.owners)
    assert len(catalog.identifiers)==len(set(catalog.identifiers))
    assert catalog.observation.action_ids==catalog.identifiers and catalog.identifiers[0]=='STOP'
    assert catalog.record['goal_focus'] is None and catalog.record['complete_page_enumeration']
    with pytest.raises(ValueError):f['context'].require_task(f['focused'])
    with pytest.raises(ValueError):f['context'].require_observations((catalog.observation,))
    index=next(i for i,owner in enumerate(catalog.owners) if owner>0)
    page,packet,result=catalog.commit(index)
    assert result.info['outcome_scope']=='permitted_nominal_model'
    with pytest.raises(ValueError):catalog.commit(index)
    replay=f['template'].fresh()
    for p in (*f['packets'],packet):replay.step_record(p)
    assert adapter.protocol.physical(replay.metrics()['history'][-1])==adapter.protocol.physical(result.info)
    with pytest.raises(ValueError):replay.step_record(packet)


def test_zero_tail_matches_greedy_complete_packets_native_history_and_fresh_audit():
    f=generated_fixture();tail=generated_tail(f);before=parameter_hash(f['policy'])
    greedy,a=adapter.rollout(f['focused'],f['admission'],f['packets'],f['frontier'],method='GREEDY',seconds=20)
    scorer=adapter.FrozenFocusedTail(tail,f['admission'],seconds=60)
    learned,b=adapter.rollout(f['focused'],f['admission'],f['packets'],f['frontier'],method='TAIL',scorer=scorer,seconds=20)
    assert semantic_digest(greedy)==semantic_digest(learned)
    assert a['explored_history']==b['explored_history'] and a['complete'] and b['complete']
    assert a['selected_equals_explored'] and b['selected_net_suffix_return']>=0
    assert a['costs']==b['costs']  # Same paid discovery; only TAIL pays policy/head.
    productive=sum(row['selected']['action_id']!='STOP' for row in b['states'])
    assert scorer.costs['policy_forwards_completed']==scorer.costs['head_forwards_completed']==productive
    assert b['model_transition_calls']==len(learned)-1 and len(learned)<=24
    assert parameter_hash(f['policy'])==before and parameter_hash(tail._head)==tail._head_hash
    replay=f['template'].fresh()
    for packet in learned:replay.step_record(packet)
    assert replay.terminated and evaluate_native_spatial_episode(replay)['accepted']
    assert [adapter.protocol.physical(r) for r in replay.metrics()['history']]==[
        adapter.protocol.physical(r) for r in b['explored_history']]
    assert f['focused']._steps==1


def test_stale_foreign_or_forged_catalog_rejected_before_forward_and_native(monkeypatch):
    f=generated_fixture();tail=generated_tail(f);catalog=adapter.collect_catalog(f['focused'].clone(),f['admission'])
    scorer=adapter.FrozenFocusedTail(tail,f['admission'])
    def forbidden(*a,**k):raise AssertionError('Invalid admission must fail before work')
    from resectionlab.native_resection import NativeResectionEngine
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    monkeypatch.setattr(adapter,'_feature',forbidden)
    forged=replace(f['admission'],record=freeze_json({**thaw_json(f['admission'].record),'focused_source_hash':'foreign'}))
    with pytest.raises(ValueError):forged.require_task(f['focused'])
    with pytest.raises(ValueError):f['admission'].require_task(f['template'])  # Authoritative evaluator task.
    catalog.observation=replace(catalog.observation,source_id='foreign')
    with pytest.raises(ValueError):scorer.score_catalog(catalog)
    assert scorer.costs['policy_forwards_started']==0


def test_nominal_reference_boundary_and_changed_prefix_refuse_before_queries(monkeypatch):
    f=generated_fixture()
    from resectionlab.native_resection import NativeResectionEngine
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',lambda *a,**k:(_ for _ in ()).throw(AssertionError('query')))
    bad={**f['frontier'],'total_reward':-1.}
    with pytest.raises(ValueError):adapter.admit_focused_context(f['original'],f['context'],f['focused'],
        f['packets'],bad,expected_prefix_actions=1)
    foreign=f['focused'].clone();foreign.case=replace(foreign.case,reference_target=np.zeros_like(foreign.case.nominal_target))
    with pytest.raises((ValueError,RuntimeError)):f['admission'].require_task(foreign)


def test_score_failure_reports_attempted_forward_and_does_not_change_state(monkeypatch):
    f=generated_fixture();catalog=adapter.collect_catalog(f['focused'].clone(),f['admission'])
    scorer=adapter.FrozenFocusedTail(generated_tail(f),f['admission']);state=catalog.work._engine.state_hash
    def failure(observation):raise RuntimeError('injected genuine forward failure')
    monkeypatch.setattr(scorer.fitted._guard.policy,'forward',failure)
    with pytest.raises(RuntimeError,match='injected') as caught:scorer.score_catalog(catalog)
    assert caught.value.focused_score_accounting['policy_forwards_started']==1
    assert caught.value.focused_score_accounting['policy_forwards_completed']==0
    assert catalog.work._engine.state_hash==state and not scorer.fitted._guard.policy.actor._forward_pre_hooks


def test_guard_stable_ties_clipping_and_exact_immediate_once():
    rows=[{'reward':r} for r in (0.,1.,1.,-1.)]
    assert adapter.choose(rows)[0]==1
    assert adapter.choose(rows,(0.,-.5,0.,100.))[0]==1
    winner,values=adapter.choose(rows,(0.,0.,2.,100.))
    assert winner==2 and values[2]==3. and values[3]==99.
    assert adapter.choose([{'reward':0.},{'reward':-1.}],(0.,100.))[0]==0


def test_cap_preserves_genuine_prefix_and_incomplete_accounting():
    f=generated_fixture()
    with pytest.raises(TimeoutError) as caught:
        adapter.rollout(f['focused'],f['admission'],f['packets'],f['frontier'],method='GREEDY',seconds=1e-12)
    row=caught.value.focused_policy_accounting
    assert not row['complete'] and row['model_transition_calls']==0
    assert row['explored_action_records']==f['packets'] and len(row['explored_history'])==1


def test_guard_uses_sealed_history_without_metrics_once_per_page_and_checks_every_packet(monkeypatch):
    f=generated_fixture();context=f['admission']
    assert semantic_digest(f['focused'].metrics()['history'][:1])==context._prefix_hash
    catalog=adapter.collect_catalog(f['focused'].clone(),context)
    assert len(catalog.pages)>1 and len(catalog.identifiers)>1
    before=semantic_digest(catalog.packets)
    def forbidden(*args,**kwargs):
        raise AssertionError('Guard must not query native geometry or scan metrics volumes')
    from resectionlab.native_resection import NativeResectionEngine
    monkeypatch.setattr(NativeSpatialTask,'metrics',forbidden)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    context.require_task(f['focused'])
    checked=[];packets=[];certificates=[]
    require=adapter.FocusedContext.require_task
    export=adapter.FocusedPagedObstructionTask.export_action
    certificate=adapter._certificate
    def require_counted(self,task):
        checked.append(id(task));return require(self,task)
    def export_counted(self,identifier):
        packets.append((id(self),identifier));return export(self,identifier)
    def certificate_counted(page,identifier):
        certificates.append((id(page),identifier));return certificate(page,identifier)
    monkeypatch.setattr(adapter.FocusedContext,'require_task',require_counted)
    monkeypatch.setattr(adapter.FocusedPagedObstructionTask,'export_action',export_counted)
    monkeypatch.setattr(adapter,'_certificate',certificate_counted)
    assert catalog.require() is catalog
    assert len(checked)==len(set(checked))==len({id(catalog.work),*(id(p) for p in catalog.pages)})
    expected=[(id(catalog.pages[owner]),identifier) for owner,identifier in zip(catalog.owners,catalog.identifiers)]
    assert packets==expected and certificates==expected[1:]
    assert semantic_digest(catalog.packets)==before
    # Genuine sealed state with a different admission prefix must still fail.
    forged_context=replace(context,_prefix_hash=semantic_digest({'wrong_prefix':True}))
    with pytest.raises(ValueError,match='prefix changed'):forged_context.require_task(f['focused'])
    # Mutating committed history must fail the original task seal before use.
    tampered=f['focused'].clone();tampered._history[0]['reward']+=1.
    with pytest.raises(RuntimeError,match='outside a transition'):context.require_task(tampered)
