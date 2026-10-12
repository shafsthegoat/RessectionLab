"""Generated same-object reference equivalence; no acquired payloads or weights."""
from dataclasses import replace
from pathlib import Path
import ast
import hashlib
import sys

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'tests'))
import numpy as np
import pytest
from resectionlab.core import freeze_json,thaw_json,semantic_digest
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.patient_planning_admission import PatientPlanningContext,COHORT_SHA256,_public_observation_binding
from test_obstruction_opening_proposals import source
import focused_policy as adapter
assert Path(adapter.__file__).resolve().parent==HERE,'Batched controls require exact staged helper'


def generated_catalog():
    """Real paired old/focused prefix then one complete current certified catalog."""
    old=NativeSpatialTask(source(enabled=False,cap=120,extra=((2,9,9),)),max_steps=24)
    observation=old.observation()
    record=freeze_json({'scope':'patient_native_planning_experiment','role':'TRAIN',
        'cohort_sha256':COHORT_SHA256,'subject':'ReMIND-045','patient_group':'ReMIND:045',
        'evidence_domain':'generated_interface_control','source_hash':old.case.source_hash,
        'decision_model_hash':old.decision_model_hash,'max_optimizer_updates':64,'max_steps':24,
        'observation_track':observation.track,'public_observation_binding':_public_observation_binding(observation)})
    context=PatientPlanningContext(record,semantic_digest(record))
    action=next(iter(old._prepare_inventory()));expected=old.clone().step(action).info
    template,_=adapter.protocol.enabled_task(old);work=template.planning_clone()
    packets,frontier,_=adapter.protocol.map_prefix(old,work,[action],adapter.protocol.compact_history([expected]))
    admitted=adapter.admit_focused_context(old,context,work,packets,frontier,expected_prefix_actions=1)
    catalog=adapter.collect_catalog(work.clone(),admitted)
    assert len(catalog.pages)>1 and any(owner>0 for owner in catalog.owners)
    return catalog,template,packets


def snapshot(catalog):
    return semantic_digest({'catalog_seal':catalog._seal,'current_digest':catalog._digest(),
        'observation':catalog.observation.fingerprint,'packets':catalog.packets,
        'pages':[{'state':p._engine.state_hash,'masks':p._engine.committed_mask_digests(),
                  'history':p._history,'task_seal':p._state_seal,'page':p._active_page,
                  'registry':sorted((str(k),id(v),p._engine._preview_digests[k])
                                    for k,v in p._engine._preview_records.items())}
                 for p in (catalog.work,*catalog.pages)]})


def refuse_both(catalog):
    for mode in ('reference','batched'):
        with pytest.raises((ValueError,RuntimeError,KeyError,IndexError)):
            adapter.validate_catalog(catalog,mode=mode)


def test_reference_callable_is_exact_frozen_validator_body():
    old=ROOT/'build/train045-focused-policy-v1/model-recovery/focused_policy.py'
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='64da4aa41a5aeccc798005d8432ae4276fcc9b442790e2b9859dbd8d5fd4d50f'
    def method(path,name):
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Catalog')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)
    assert method(old,'require')==method(HERE/'focused_policy.py','require_reference')


def test_same_catalog_alternating_validation_has_no_native_or_observation_work(monkeypatch):
    catalog,_,_=generated_catalog();before=snapshot(catalog)
    def forbidden(*a,**k):raise AssertionError('Validation cannot query/commit/observe')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    monkeypatch.setattr(NativeSpatialTask,'_transition',forbidden)
    monkeypatch.setattr(NativeSpatialTask,'observation',forbidden)
    monkeypatch.setattr(adapter,'_feature',forbidden)
    for pair in range(4):
        order=('reference','batched') if pair%2==0 else ('batched','reference')
        for mode in order:
            assert adapter.validate_catalog(catalog,mode=mode) is catalog
            assert snapshot(catalog)==before


def test_batched_reduces_state_authentication_but_retains_all_certificates_and_scores(monkeypatch):
    catalog,_,_=generated_catalog();counts={'state':0,'certificate':0,'score':0}
    original_assert=FocusedPagedObstructionTask._assert_frozen
    original_digest=NativeResectionEngine._result_digest
    original_score=NativeSpatialTask._score_record
    def counted_assert(self):counts['state']+=1;return original_assert(self)
    def counted_digest(result):counts['certificate']+=1;return original_digest(result)
    def counted_score(self,*a,**k):counts['score']+=1;return original_score(self,*a,**k)
    monkeypatch.setattr(FocusedPagedObstructionTask,'_assert_frozen',counted_assert)
    monkeypatch.setattr(NativeResectionEngine,'_result_digest',staticmethod(counted_digest))
    monkeypatch.setattr(NativeSpatialTask,'_score_record',counted_score)
    adapter.validate_catalog(catalog,mode='reference');reference=dict(counts)
    counts.update(state=0,certificate=0,score=0)
    adapter.validate_catalog(catalog,mode='batched');batched=dict(counts)
    assert reference['state']>batched['state']==len({id(p) for p in (catalog.work,*catalog.pages)})
    assert reference['certificate']==batched['certificate']==len(catalog.identifiers)-1
    assert reference['score']==batched['score']==len(catalog.identifiers)-1


@pytest.mark.parametrize('field',['schema','source_hash','decision_model_hash','source_state_hash',
    'steps_taken','current_tool','focus','query_catalog_digest','action_id','tool','entry','tip','seal','extra'])
def test_every_packet_field_and_seal_refused_even_if_outer_catalog_is_resealed(field):
    catalog,_,_=generated_catalog()
    i=next(i for i,owner in enumerate(catalog.owners) if owner>0)
    rows=[thaw_json(p) for p in catalog.packets];packet=rows[i]
    if field in ('tool','entry','tip'):
        if field=='tool':packet['physical_action']['tool_id']='foreign_tool'
        else:
            key='entry_mm' if field=='entry' else 'tip_mm'
            packet['physical_action'][key][0]+=1.
    elif field=='focus':packet[field]={'unknown_focus':True}
    elif field=='steps_taken':packet[field]+=1
    elif field=='extra':packet['unrecognized_field']=True
    else:packet[field]='foreign'
    # A forged non-authority digest is not enough to authorize changed physics.
    if field!='seal':packet['seal']=semantic_digest({k:v for k,v in packet.items() if k!='seal'})
    catalog.packets=freeze_json(rows);catalog._seal=catalog._digest()
    refuse_both(catalog)


def test_stop_retains_full_page_binding_and_null_physical_geometry():
    catalog,_,_=generated_catalog();rows=[thaw_json(p) for p in catalog.packets]
    rows[0]['physical_action']={'tool_id':'fake','entry_mm':[0,0,0],'tip_mm':[1,0,0]}
    rows[0]['seal']=semantic_digest({k:v for k,v in rows[0].items() if k!='seal'})
    catalog.packets=freeze_json(rows);catalog._seal=catalog._digest();refuse_both(catalog)


@pytest.mark.parametrize('mutation',['page','page_state','inventory','owner','registry','certificate_digest',
    'certificate_geometry','score','history','source_case','focus','terminated'])
def test_page_certificate_score_and_history_mutations_fail_on_the_next_pass(mutation):
    catalog,_,_=generated_catalog()
    adapter.validate_catalog(catalog,mode='batched')  # Prior success grants no later authority.
    i=next(i for i,owner in enumerate(catalog.owners) if owner>0)
    page=catalog.pages[catalog.owners[i]];identifier=catalog.identifiers[i]
    preview=page._inventory[identifier];restore=lambda:None
    if mutation=='page':page._active_page=None
    elif mutation=='page_state':page._page_state_hash='foreign'
    elif mutation=='inventory':page._inventory=None
    elif mutation=='owner':
        owners=list(catalog.owners);owners[i]=0;catalog.owners=tuple(owners);catalog._seal=catalog._digest()
    elif mutation=='registry':page._engine._preview_records.pop(id(preview))
    elif mutation=='certificate_digest':page._engine._preview_digests[id(preview)]='foreign'
    elif mutation=='certificate_geometry':
        prior=preview.tip_mm;object.__setattr__(preview,'tip_mm',(prior[0]+1,*prior[1:]))
        restore=lambda:object.__setattr__(preview,'tip_mm',prior)
    elif mutation=='score':
        rows=[thaw_json(r) for r in catalog.scores];rows[i]['reward']+=1
        catalog.scores=freeze_json(rows);catalog._seal=catalog._digest()
    elif mutation=='history':page._history[0]['reward']+=1.
    elif mutation=='source_case':page.case=replace(page.case,reference_target=np.zeros_like(page.case.nominal_target))
    elif mutation=='focus':page._goal_focus=freeze_json({'wrong_focus':True})
    elif mutation=='terminated':page._terminated=True
    try:refuse_both(catalog)
    finally:restore()


def test_same_catalog_cancellation_refuses_both_validators_without_state_change():
    catalog,_,_=generated_catalog();before=snapshot(catalog)
    page=catalog.pages[0];prior=page._cancelled
    page._cancelled=lambda:True
    try:
        for mode in ('reference','batched'):
            with pytest.raises(InterruptedError):adapter.validate_catalog(catalog,mode=mode)
            assert snapshot(catalog)==before
    finally:page._cancelled=prior
    for mode in ('reference','batched'):
        assert adapter.validate_catalog(catalog,mode=mode) is catalog


def test_commit_still_uses_canonical_packet_verification_and_native_transition(monkeypatch):
    catalog,template,prefix=generated_catalog();i=next(i for i,owner in enumerate(catalog.owners) if owner>0)
    calls=[];original=FocusedPagedObstructionTask._execute_record
    def tracked(self,packet,*,planning):
        calls.append((packet['seal'],planning));return original(self,packet,planning=planning)
    monkeypatch.setattr(FocusedPagedObstructionTask,'_execute_record',tracked)
    page,packet,transition=catalog.commit(i)
    assert calls==[(packet['seal'],True)] and transition.info['outcome_scope']=='permitted_nominal_model'
    replay=template.fresh()
    for p in (*prefix,packet):replay.step_record(p)
    assert adapter.protocol.physical(replay._history[-1])==adapter.protocol.physical(transition.info)
    assert replay._engine.state_hash==page._engine.state_hash
    with pytest.raises(ValueError):adapter.validate_catalog(catalog,mode='batched')
