"""Generated proposal controls; no acquired arrays, trained models or search."""
from dataclasses import asdict, fields, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import array_digest, freeze_json, semantic_digest, thaw_json
from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask
from resectionlab.native_proposals import (NominalCavityProposalConfig, OBSTRUCTION_OPENING_FAMILY,
    OBSTRUCTION_CELLS_PER_PREVIEW, OBSTRUCTION_SELECTED_CELLS)


def source(*, enabled=True, cap=96, sign=1, rotated=False, covered=False, extra=(), domain=False):
    shape=(16,19,19); support=np.zeros(shape,bool)
    blocker=(4 if sign>0 else 11,10,10); goal=(9 if sign>0 else 6,9,9)
    support[blocker]=True; support[goal]=True
    for cell in extra: support[cell]=True
    target=support.astype(np.float32)
    affine=np.diag([1.,.9,1.1,1.])
    if rotated:
        theta=.31; rotation=np.array([[np.cos(theta),-np.sin(theta),0.],
            [np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
        affine[:3,:3]=rotation@affine[:3,:3]
    affine[:3,3]=[12.,-7.,8.]
    center=np.array([1.5 if sign>0 else 13.5,9.,9.])
    normal=sign*affine[:3,0]
    access=AccessWindow(affine[:3,:3]@center+affine[:3,3],normal,6.)
    offsets=((0,0),(1,1)) if covered else ((0,0),)
    config=NominalCavityProposalConfig(offsets_source_voxels=offsets,max_candidates=cap,
        tool_footprint_opening=True,obstruction_opening=enabled)
    return NativeSpatialCase(np.arange(np.prod(shape),dtype=np.float32).reshape(shape),
        support,target,affine,access,GENERIC_TOOLS,track="synthetic_scan",
        support_source_kind="derived_from_scan",support_derivation="generated sparse public occupancy",
        nominal_target=target,target_source_kind="derived_from_scan",target_derivation="generated public goal",
        crop_shape=shape,proposal_mode="nominal_cavity_v1",proposal_config=config,
        support_domain=np.ones(shape,bool) if domain else None)


def physical(rows):
    return [{k:v for k,v in row.items() if k not in ("action_id","proposal_id")}
            for row in rows]


@pytest.mark.parametrize("sign",[1,-1])
@pytest.mark.parametrize("rotated",[False,True])
def test_missing_transverse_axis_repairs_generated_blocker_with_full_replay(sign,rotated,monkeypatch):
    original=NativeResectionEngine.preview_stroke; calls=[]
    def preview(self,*args,**kwargs):
        calls.append(kwargs.get("obstruction_diagnostics",False))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(NativeResectionEngine,"preview_stroke",preview)
    case=source(sign=sign,rotated=rotated); task=NativeSpatialTask(case,max_steps=3)
    inventory=task.candidate_inventory();account=inventory['obstruction_accounting']
    assert len(calls)==inventory['emitted_count']==account['base_preview_calls']+account['added_preview_calls']
    assert calls==[True]*account['base_preview_calls']+[False]*account['added_preview_calls']
    assert account['repeated_preview_calls']==0 and account['recursive_expansion'] is False
    rows=[r for r in inventory['emitted'] if r['family']==OBSTRUCTION_OPENING_FAMILY]
    assert rows and all(r['offset_source_voxels']==[1,1] for r in rows)
    ray=next(r for r in rows if r['tool_id']=='generic_suction' and r['feasible'])
    assert task._engine.history==[]
    action=ray['action_id'];branch=task.clone();branch.step(action);branch.step('STOP')
    assert len(branch.metrics()['history'][0]['removed_indices_native'])==1
    replay=task.fresh();replay.step(action);replay.step('STOP')
    assert replay.metrics()['history']==branch.metrics()['history']
    assert replay.independent_geometry_check().feasible
    assert task.candidate_inventory()==inventory


def test_entire_original_prefix_and_default_off_queries_unchanged(monkeypatch):
    old=NativeSpatialTask(source(enabled=False)); new=NativeSpatialTask(source())
    a,b=old.candidate_inventory(),new.candidate_inventory()
    assert physical(a['ledger'])==physical(b['ledger'][:len(a['ledger'])])
    assert old.case.source_hash!=new.case.source_hash
    calls=[];original=NativeResectionEngine.preview_stroke
    def preview(self,*args,**kwargs):
        calls.append(kwargs);return original(self,*args,**kwargs)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',preview)
    repeat=old.fresh()
    assert repeat.candidate_inventory()==a and repeat.observation().fingerprint==old.observation().fingerprint
    assert len(calls)==a['emitted_count'] and all('obstruction_diagnostics' not in row for row in calls)
    assert 'obstruction_accounting' not in a


def test_no_additions_when_radial_noncontainment_not_established():
    task=NativeSpatialTask(source(covered=True))
    assert not any(r['family']==OBSTRUCTION_OPENING_FAMILY for r in task.candidate_inventory()['emitted'])


def test_private_reference_does_not_change_inventory_or_planning():
    case=source();a=NativeSpatialTask(case)
    b=NativeSpatialTask(replace(case,reference_target=np.ones(case.reference_target.shape,np.float32)))
    assert a.candidate_inventory()==b.candidate_inventory()
    assert a.observation().fingerprint==b.observation().fingerprint
    assert a.planning_clone().candidate_inventory()==b.planning_clone().candidate_inventory()


def test_additions_keep_unknown_domain_native_rejection():
    case=source(domain=True); domain=np.array(case.support_domain)
    domain[2,11,11]=False
    task=NativeSpatialTask(replace(case,support_domain=domain))
    # Unknown is neither an exposed free-space seed nor permission to bypass
    # complete-tool checks; all actually offered actions have native certificates.
    rows=[r for r in task.candidate_inventory()['emitted'] if r['family']==OBSTRUCTION_OPENING_FAMILY]
    assert rows and any(r['tool_id']=='generic_suction' and r['reason'].startswith('UNKNOWN_DOMAIN:') for r in rows)
    assert all(not r['feasible'] and r['reason'].startswith('UNKNOWN_DOMAIN:') for r in rows)


def prepared(case):
    engine=NativeResectionEngine(case._native_config)
    provider=case._nominal_proposer;batch=provider.propose(engine)
    evidence=[engine.preview_stroke(r.tool_id,r.tip_mm,entry_mm=r.entry_mm,
        obstruction_diagnostics=True,obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW).obstruction_diagnostic
        for r in batch.proposals]
    return engine,provider,batch,evidence


def test_extension_reuses_sidecars_without_any_preview_and_refuses_stale_tamper(monkeypatch):
    engine,provider,batch,evidence=prepared(source())
    assert any(e is not None for e in evidence)
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected repeated preview')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    extended=provider.append_obstruction_openings(batch,engine,evidence)
    provider.validate_batch(extended,engine,obstruction_results=evidence)
    with pytest.raises(ValueError,match='sidecars'):
        provider.validate_batch(extended,engine)
    altered=list(evidence);i=next(i for i,e in enumerate(evidence) if e is not None)
    record=thaw_json(altered[i]);record['source_state_hash']='changed';altered[i]=freeze_json(record)
    with pytest.raises(ValueError):provider.append_obstruction_openings(batch,engine,altered)
    with pytest.raises(InterruptedError):provider.append_obstruction_openings(batch,engine,evidence,cancelled=lambda:True)


def synthetic_sidecar(case,ray,cells,*,total=None):
    """Contract-only generated metadata, not a native feasibility witness."""
    total=len(cells) if total is None else total
    record={'version':'native-first-shaft-obstruction-v1','reason':'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE',
        'source_hash':case.source_hash,'source_state_hash':NativeResectionEngine(case._native_config).state_hash,
        'decision_model_hash':case._native_config.fingerprint,'native_affine_hash':array_digest(case._native_affine_ras_mm),
        'source_shape':list(case.observed_support.shape),'tool':asdict(next(t for t in case.tools if t.tool_id==ray.tool_id)),
        'entry_mm':ray.entry_mm,'requested_tip_mm':ray.tip_mm,'interaction_mode':'aspirate',
        'cell_limit':16,'blocked_cell_count':total,'retained_cell_count':len(cells),
        'truncated':total>len(cells),'complete_first_failure_set':total==len(cells),
        'blocked_indices_native':cells,'blocked_indices_hash':array_digest(np.asarray(cells,np.int64))}
    return freeze_json({**record,'fingerprint':semantic_digest(record)})


def test_full_selection_cap_and_public_nominal_priority_with_explicit_omissions(monkeypatch):
    cells=[(4,y,z) for y in range(12,16) for z in range(3,11)]
    case=source(extra=cells); nominal=np.zeros(case.nominal_target.shape,np.float32)
    nominal[9,9,9]=1;nominal[cells[-1]]=1
    case=replace(case,nominal_target=nominal)
    engine=NativeResectionEngine(case._native_config);provider=case._nominal_proposer;batch=provider.propose(engine)
    assert len(batch.proposals)>=2
    evidence=[synthetic_sidecar(case,batch.proposals[0],cells[:16]),
        synthetic_sidecar(case,batch.proposals[1],cells[16:])]+[None]*(len(batch.proposals)-2)
    extended=provider.append_obstruction_openings(batch,engine,evidence)
    rows=extended.ledger[len(batch.ledger):]
    assert rows[0].blocker_voxel==cells[-1] and rows[0].public_nominal_priority
    assert extended.obstruction_accounting['selected_cells']==OBSTRUCTION_SELECTED_CELLS==16
    assert extended.obstruction_accounting['selection_omitted_cells']==16
    assert sum(r.reason=='OBSTRUCTION_SELECTION_CAP' for r in rows)==32
    # Exercise the actual inventory accounting, without presenting fabricated
    # sidecars as native geometry: only the bounded ledger/accounting is injected.
    task=NativeSpatialTask(case);task._proposal_batch=extended
    task._ledger=tuple({**asdict(r),'proposal_reason':r.reason,'feasible':False,
        'endpoint_center_in_actor_crop':False} for r in extended.ledger)
    task._inventory={}
    inv=task.candidate_inventory()
    assert inv['obstruction_selection_omitted_count']==32 and inv['omitted_count']>=32 and not inv['complete']


def test_sidecar_truncation_candidate_cap_and_dedup_are_recorded():
    case=source(cap=1);engine,provider,batch,evidence=prepared(case)
    assert len(batch.proposals)==1 and evidence[0] is not None
    extended=provider.append_obstruction_openings(batch,engine,evidence)
    assert extended.proposals==batch.proposals
    assert any(r.family==OBSTRUCTION_OPENING_FAMILY and r.reason=='CANDIDATE_CAP' for r in extended.ledger)
    cells=[(4,y,z) for y in range(12,16) for z in range(3,7)]
    case=source(extra=cells);engine=NativeResectionEngine(case._native_config)
    provider=case._nominal_proposer;batch=provider.propose(engine)
    evidence=[synthetic_sidecar(case,batch.proposals[0],cells,total=20)]+[None]*(len(batch.proposals)-1)
    extended=provider.append_obstruction_openings(batch,engine,evidence)
    assert extended.obstruction_accounting['evidence_truncated']
    assert extended.obstruction_accounting['unretained_blocker_mentions']==4
    assert extended.obstruction_accounting['discovery_complete'] is False
    keys=[(r.tool_id,r.entry_mm,r.tip_mm) for r in extended.proposals]
    assert len(keys)==len(set(keys))


@pytest.mark.parametrize('value',[None,1,'true'])
def test_opt_in_requires_boolean(value):
    with pytest.raises(ValueError,match='explicit bool'):NominalCavityProposalConfig(obstruction_opening=value)


def test_historical_saved_protocols_and_new_enabled_record_round_trip():
    from resectionlab.patient_planning_cohort_spec import validate_sequential_protocol
    root=Path(__file__).resolve()
    root=next(p for p in root.parents if (p/'src/resectionlab').is_dir())
    paths=[root/'build/balanced-teacher-il64-v1/root-release.json',
        root/'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/root-release.json']
    if not all(p.exists() for p in paths):pytest.skip('Local pinned historical metadata unavailable')
    pins=['5facc562a712262b636f354044befaed4a55e6547016f3181268a1df764e2b22',
          'c1cde8eb92f9df5d85ccb91ed0e591adb1da70e011582f3be116d237fecf72db']
    for path,pin in zip(paths,pins):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
        protocol=json.loads(path.read_text())['learning_protocol']
        old=protocol['cohort_execution']['proposal_config']
        assert 'obstruction_opening' not in old
        config=NominalCavityProposalConfig(**old)
        assert json.loads(json.dumps(config.to_record()))==old
        assert semantic_digest(validate_sequential_protocol(protocol))==semantic_digest(protocol)
        enabled=replace(config,obstruction_opening=True)
        changed=json.loads(json.dumps(protocol));changed['cohort_execution']['proposal_config']=enabled.to_record()
        changed['cohort_execution']['proposal_rule_hash']=enabled.fingerprint
        assert semantic_digest(validate_sequential_protocol(changed))==semantic_digest(changed)
        assert enabled.to_record()['obstruction_opening'] is True


def test_historical_source_default_record_fingerprint_inventory_observation_and_replay():
    root=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
    before=root/'build/obstruction-opening-proposals-v1/before'
    if not before.is_dir():pytest.skip('Optional frozen historical source comparison unavailable')
    def load(name):
        alias='resectionlab._before_obstruction_'+name
        spec=importlib.util.spec_from_file_location(alias,before/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
        return module
    prior=load('native_proposals')
    for kwargs in ({},{'intermediate_opening_mm':1.},
            {'intermediate_opening_mm':1.,'tool_footprint_opening':True,'max_candidates':120}):
        old=prior.NominalCavityProposalConfig(**kwargs);new=NominalCavityProposalConfig(**kwargs)
        assert new.to_record()==asdict(old) and new.fingerprint==old.fingerprint
    previous=load('native_spatial_task');case=source(enabled=False)
    # Compare the old API with the unchanged default, not opt-in fields added
    # after this archived constructor was frozen.
    prior_fields={f.name for f in fields(previous.NativeSpatialCase) if f.init}
    newer_fields={f.name for f in fields(case) if f.init}-prior_fields
    assert newer_fields == {'post_exposure'} and case.post_exposure is None
    old_case=previous.NativeSpatialCase(**{name:getattr(case,name) for name in prior_fields})
    old=previous.NativeSpatialTask(old_case);new=NativeSpatialTask(case)
    assert old_case.source_hash==case.source_hash and old.decision_model_hash==new.decision_model_hash
    assert old.candidate_inventory()==new.candidate_inventory()
    assert old.observation().fingerprint==new.observation().fingerprint
    old.step('STOP');new.step('STOP')
    assert old.metrics()==new.metrics()
