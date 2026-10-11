"""Generated-only page lifecycle and complete beam/native replay controls."""
from dataclasses import fields, replace
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.observed_search import observed_beam_search
from test_obstruction_opening_proposals import source, synthetic_sidecar
from test_observed_search_opening_lane import Tree, opening_tree, clean

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
MODE='return_plus_opening_depth_volume_v1'
BASELINE_DIR=Path(os.environ.get('PAGED_ROUTE_BASELINE_DIR',
    str(ROOT/'build/select013-paged-route-v1/before')))
BASELINE_SHA={
 'native_proposals':'9b7a62526aa6abf4933c253f4bd72ab2a4eae2ea0c9df6846227455cc8e16b59',
 'native_spatial_task':'8237f90b3b512d03452be027e6e7e337f69da3e8127f169fe870a56c47418c32',
 'observed_search':'52ed888c43a5b8405b4dd2407b583dcff59e21cf3c08b68a6c6571107b0268fd'}


def load(name):
    alias='resectionlab._paged_baseline_'+name
    path=BASELINE_DIR/(name+'.py')
    if not path.is_file():pytest.skip('Optional exact pre-paging source archive unavailable; set PAGED_ROUTE_BASELINE_DIR')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==BASELINE_SHA[name]
    spec=importlib.util.spec_from_file_location(alias,path)
    module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
    return module


def paged_case(**kwargs):
    case=source(**kwargs)
    return replace(case,proposal_config=replace(case.proposal_config,obstruction_page_size=8))


def test_default_record_fingerprint_inventory_history_and_search_parity():
    oldp=load('native_proposals')
    for kwargs in ({},{'intermediate_opening_mm':1.},
                   {'tool_footprint_opening':True,'max_candidates':120},
                   {'tool_footprint_opening':True,'obstruction_opening':True,'max_candidates':120}):
        old=oldp.NominalCavityProposalConfig(**kwargs);new=NominalCavityProposalConfig(**kwargs)
        assert old.fingerprint==new.fingerprint and old.to_record()==new.to_record()
    previous=load('native_spatial_task');case=source(enabled=False)
    oldcase=previous.NativeSpatialCase(**{f.name:getattr(case,f.name) for f in fields(case) if f.init})
    old=previous.NativeSpatialTask(oldcase);new=NativeSpatialTask(case)
    assert oldcase.source_hash==case.source_hash and old.decision_model_hash==new.decision_model_hash
    assert old.observation().fingerprint==new.observation().fingerprint
    assert old.candidate_inventory()==new.candidate_inventory()
    old.step('STOP');new.step('STOP');assert old.metrics()==new.metrics()
    oldsearch=load('observed_search')
    for cap in (0,1,3,30):
        a,b=opening_tree(),opening_tree()
        options=dict(max_calls=cap,beam_width=2,seconds=20,transition_mode='lazy_planning',
                     retained_prefix_diagnostics=True,retention_mode=MODE)
        pa,ra=oldsearch.observed_beam_search(a,**options);pb,rb=observed_beam_search(b,**options)
        assert pa==pb and clean(ra)==clean(rb) and a.counters==b.counters


def test_opt_in_identity_refuses_partial_actor_or_ordinary_task():
    case=paged_case(cap=120)
    assert case.proposal_config.to_record()['obstruction_page_size']==8
    assert NominalCavityProposalConfig(**case.proposal_config.to_record()).fingerprint==case.proposal_config.fingerprint
    for bad in (0,True,7,16):
        with pytest.raises(ValueError):replace(case.proposal_config,obstruction_page_size=bad)
    with pytest.raises(ValueError):replace(case.proposal_config,obstruction_opening=False)
    with pytest.raises(ValueError):NativeSpatialTask(case)
    task=PagedObstructionTask(case)
    with pytest.raises(ValueError):task.observation()
    with pytest.raises(ValueError):task.candidate_inventory()
    with pytest.raises(ValueError):task.observed_greedy_search()
    with pytest.raises(ValueError):task.advance_planning('STOP')
    with pytest.raises(ValueError):observed_beam_search(task,max_calls=10,beam_width=2,
        action_expansion='obstruction_pages_v1',transition_mode='eager')


def test_real_native_full_route_two_negative_preparations_then_target_and_replay(monkeypatch):
    # Two separated transverse co-blockers, both outside the original sole axis.
    # Neither is a target; public native reward must pay both preparations.
    case=paged_case(cap=120,extra=((4,10,8),))
    nominal=np.zeros(case.nominal_target.shape,np.float32);nominal[9,9,9]=1
    case=replace(case,nominal_target=nominal,reference_target=nominal)
    baseline=NativeSpatialTask(replace(case,proposal_config=replace(case.proposal_config,
        obstruction_opening=False,obstruction_page_size=None)),max_steps=4)
    bp,ba=observed_beam_search(baseline,max_calls=100,beam_width=2,seconds=20,
        transition_mode='lazy_planning',retention_mode=MODE)
    assert bp==('STOP',)
    calls=[];original=NativeResectionEngine.preview_stroke
    def preview(self,*a,**k):
        calls.append((self.state_hash,k.get('obstruction_diagnostics',False)))
        return original(self,*a,**k)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',preview)
    task=PagedObstructionTask(case,max_steps=4)
    path,report=observed_beam_search(task,max_calls=100,beam_width=2,seconds=20,
        transition_mode='lazy_planning',retention_mode=MODE,retained_prefix_diagnostics=True,
        action_expansion='obstruction_pages_v1')
    assert report['estimated_incremental_return']>0 and path!=('STOP',)
    assert all(p['complete'] for p in report['action_pages'])
    assert not report['call_cap_reached'] and not report['time_cap_reached']
    rollout=task.fresh();replay=task.fresh()
    for action in path:rollout.step(action)
    for action in path:replay.step(action)
    assert rollout.terminated and replay.terminated and rollout.metrics()['history']==replay.metrics()['history']
    history=rollout.metrics()['history']
    assert sum(r['reward']<0 for r in history)>=2
    assert sum(r['target_removed_mm3'] for r in history)==pytest.approx(case._native_config.voxel_volume_mm3)
    assert rollout.independent_geometry_check().feasible
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    assert evaluate_native_spatial_episode(rollout)["accepted"]
    assert report['model_transition_calls']==report['lazy_planning_transition_calls']
    assert report['stop_transition_calls']==0
    assert task._steps==0 and len(calls)>0


def test_all_32_selected_pairs_page_without_base_cap_clipping_or_recursive_preview(monkeypatch):
    # Synthetic sidecars test selection/paging, not native obstruction truth.
    # Actual candidate certification still uses the real native engine.
    cells=[(4,y,z) for y in range(12,16) for z in range(3,7)]
    case=paged_case(cap=1,extra=cells)
    task=PagedObstructionTask(case).planning_clone()
    proposer=case._nominal_proposer;original=proposer.append_obstruction_openings
    def inject(batch,engine,results,**kwargs):
        fabricated=[synthetic_sidecar(case,batch.proposals[0],cells)]+[None]*(len(batch.proposals)-1)
        return original(batch,engine,fabricated,**kwargs)
    monkeypatch.setattr(proposer,'append_obstruction_openings',inject)
    pages=list(task.planning_action_pages())
    assert len(pages)==5
    records=[p.candidate_inventory() for p,ids in pages]
    assert records[0]['base_emitted_count']==1
    assert [r['additional_verified_in_page'] for r in records]==[0,8,8,8,8]
    assert records[-1]['additional_emitted_count']==32
    rows=records[0]['ledger'];base_len=len(task._base_batch.ledger)
    assert not any(r['proposal_reason']=='CANDIDATE_CAP' for r in rows[base_len:])
    assert records[0]['obstruction_accounting']['recursive_expansion'] is False
    for page,ids in pages:
        assert len(page._engine._preview_records)<=128
        for cert in page._inventory.values():
            assert page._engine._preview_records[id(cert)] is cert
        assert page._engine.state_hash==task._engine.state_hash
    assert len({a for p,ids in pages for a in ids})==sum(len(ids) for p,ids in pages)
    # Creating later pages does not revoke an earlier page certificate.
    for page,ids in pages:
        if ids:
            child=page.clone();child.advance_planning(ids[0]);assert child._steps==1
            assert page._steps==task._steps==0


def test_private_reference_swap_and_no_page_mutation_of_parent():
    case=paged_case(cap=120)
    tasks=[PagedObstructionTask(c).planning_clone() for c in
           (case,replace(case,reference_target=np.zeros(case.reference_target.shape,np.float32)))]
    pages=[list(t.planning_action_pages()) for t in tasks]
    assert [p.candidate_inventory() for p,a in pages[0]]==[p.candidate_inventory() for p,a in pages[1]]
    for task,group in zip(tasks,pages):
        before=task._engine.committed_mask_digests()
        p,ids=next((p,ids) for p,ids in group if ids)
        branch=p.clone();branch.advance_planning(ids[0]);assert task._engine.committed_mask_digests()==before
        assert branch._active_page is None and branch._base_inventory is None
    assert tasks[0].metrics()==tasks[1].metrics()


def test_cancellation_and_durable_commit_clear_page_cache(monkeypatch):
    cancelled=[False];task=PagedObstructionTask(paged_case(),cancelled=lambda:cancelled[0]).planning_clone()
    p,ids=next((p,ids) for p,ids in task.planning_action_pages() if ids)
    before=p.metrics()['history'];cancelled[0]=True
    with pytest.raises(InterruptedError):p.advance_planning(ids[0])
    cancelled[0]=False;assert p.metrics()['history']==before
    native=p._engine.commit_preview
    def commit(result):native(result);cancelled[0]=True
    monkeypatch.setattr(p._engine,'commit_preview',commit)
    with pytest.raises(InterruptedError) as error:p.advance_planning(ids[0])
    assert getattr(error.value,'committed',False)
    cancelled[0]=False
    assert p._steps==1 and p._active_page is None and p._base_inventory is None


class PageTree(Tree):
    def planning_action_pages(self):
        ids=tuple(self.tree.get(self.path,{}));groups=[ids[:1],ids[1:9],ids[9:]]
        groups=[g for i,g in enumerate(groups) if i==0 or g]
        for i,group in enumerate(groups):
            page=self.clone();page.page_info={'page_index':i,'additional_page_count':len(groups)-1}
            yield page,group
    def planning_page_record(self):return self.page_info


def test_shared_layer_retention_considers_late_page_and_cap_is_partial():
    from resectionlab.observed_search import ObservedSearchLimit
    tree={():{**{'p%02d'%i:(-.1,((i%6,1,1),),False) for i in range(9)},
                   'last':(-2.,((1,1,3),),False)},('last',):{'reach':(4.,((1,1,4),),True)}}
    task=PageTree(tree,max_steps=2)
    path,report=observed_beam_search(task,max_calls=30,beam_width=2,seconds=20,
        transition_mode='lazy_planning',retention_mode=MODE,action_expansion='obstruction_pages_v1')
    assert path==('last','reach') and report['estimated_incremental_return']==2.
    assert [r['page_index'] for r in report['action_pages'][:3]]==[0,1,2]
    capped=PageTree(tree,max_steps=2)
    path,partial=observed_beam_search(capped,max_calls=9,beam_width=2,seconds=20,
        transition_mode='lazy_planning',retention_mode=MODE,action_expansion='obstruction_pages_v1')
    assert partial['call_cap_reached'] and partial['completed_layers']==0
    assert partial['action_pages'][-1]['complete'] is False
    assert path==('STOP',) # Partial incumbent is NOT accepted completion.
    with pytest.raises(ObservedSearchLimit) as e:
        observed_beam_search(PageTree(tree),max_calls=30,beam_width=2,seconds=0,
            transition_mode='lazy_planning',action_expansion='obstruction_pages_v1')
    assert e.value.accounting['time_cap_reached']


def test_real_native_certificate_lifetime_120_plus_8_pages():
    case=source();task=NativeSpatialTask(case)
    ray=next(r for r in task._proposal_batch.proposals if r.proposal_id in task._inventory)
    engine=NativeResectionEngine(case._native_config,immutable_state=True)
    base=[engine.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm) for _ in range(120)]
    assert all(r.feasible for r in base) and len(engine._preview_records)==120
    pages=[]
    for _ in range(4):
        page=engine.clone()
        extra=[page.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm) for _ in range(8)]
        assert len(page._preview_records)==128
        assert all(page._preview_records[id(r)] is r for r in base+extra)
        pages.append((page,extra))
    for page,extra in pages:
        child=page.clone();child.commit_preview(extra[-1])
        assert child.history and len(engine._preview_records)==120
    assert not engine.history
