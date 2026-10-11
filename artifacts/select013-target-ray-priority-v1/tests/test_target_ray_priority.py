"""Bounded generated group coverage, default parity, and actual native replay."""
from dataclasses import asdict,replace
import hashlib,importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from resectionlab.core import thaw_json
from resectionlab.geometry import AccessWindow
from resectionlab.native_proposals import (NominalCavityProposalConfig,RESIDUAL_OBSTRUCTION_RULE,
    TARGET_RAY_OBSTRUCTION_RULE,_target_ray_group_order)
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.paged_obstruction import PagedObstructionTask
from test_obstruction_opening_proposals import source,prepared

HERE=Path(__file__).resolve().parents[1]
def old_module():
 p=HERE/'before/native_proposals.py'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='8d92d6f4c68f394f17d6c2357d8748e7abab8bfac0b555dbf3bb5d81df325a31'
 spec=importlib.util.spec_from_file_location('resectionlab._before_target_group',p)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def ruled(case,rule):return replace(case,proposal_config=replace(case.proposal_config,obstruction_page_size=8,obstruction_selection_rule=rule))

def test_old_rules_serialization_identity_catalog_and_no_extra_queries_exact(monkeypatch):
 old=old_module()
 for rule in (None,RESIDUAL_OBSTRUCTION_RULE):
  case=ruled(source(cap=120),rule);engine,new,batch,evidence=prepared(case)
  cfg=old.NominalCavityProposalConfig(**case.proposal_config.to_record())
  assert cfg.to_record()==case.proposal_config.to_record() and cfg.fingerprint==case.proposal_config.fingerprint
  prior=old.PreparedNominalCavityProposer(case._native_config,new._nominal,nominal_provenance=new._provenance,
   config=cfg,index_affine=new._index_affine,index_frame_record=new._frame_record)
  a=prior.propose(engine);assert asdict(a)==asdict(batch)
  def forbidden(*a,**k):raise AssertionError('No additional discovery preview')
  with monkeypatch.context() as patch:
   patch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
   before=prior.append_obstruction_openings(a,engine,evidence);after=new.append_obstruction_openings(batch,engine,evidence)
  assert asdict(before)==asdict(after) and before.obstruction_accounting==after.obstruction_accounting

def test_crowded_cap_covers_distinct_goal_group_instead_of_native_index_prefix():
 cells=[(i,0,0) for i in range(24)];groups=[((0,0),cells[:12]),((1,0),cells[16:22])]
 order,rows=_target_ray_group_order(cells,groups,set(cells),set(),16)
 assert order[:4]==[cells[0],cells[16],cells[1],cells[17]]
 assert len(set(order))==len(cells) and len(order[:16])==16
 assert set(cells[16:22])<=set(order[:16]) and not set(cells[16:22])<=set(cells[:16])
 assert rows[0]['unselected_co_blockers'] and rows[1]['all_retained_co_blockers_selected']
 assert all(set(map(tuple,r['selected_co_blockers']))|set(map(tuple,r['unselected_co_blockers']))==set(map(tuple,r['retained_co_blockers'])) for r in rows)

def test_aliases_group_input_order_and_shared_cells_do_not_get_extra_turns():
 cells=[(i,1,1) for i in range(22)]
 groups=[((0,0),cells[:10]),((1,0),cells[10:]),((2,1),cells[5:9])]
 base=_target_ray_group_order(cells,groups,set(cells),{cells[-1]},16)
 aliases=[groups[1],groups[0],groups[0],((0,0),cells[2:6]),groups[2],groups[1]]
 assert _target_ray_group_order(reversed(cells),aliases,set(cells),{cells[-1]},16)==base
 assert base[0][0]==cells[-1] and len(set(base[0][:16]))==16

def test_ineligible_and_unselected_co_blockers_remain_explicit_and_inputs_are_required():
 cells=[(i,0,0) for i in range(20)];eligible=set(cells[2:])
 order,rows=_target_ray_group_order(cells,[((0,0),cells)],eligible,set(),16)
 assert len(order)==20 and order[-2:]==cells[:2]
 assert rows[0]['unselected_co_blockers']==[cells[0],cells[1],cells[18],cells[19]]
 assert not rows[0]['all_retained_co_blockers_selected']
 with pytest.raises(ValueError):_target_ray_group_order(cells,[((0,0),[(999,0,0)])],eligible,set(),16)
 with pytest.raises(ValueError):_target_ray_group_order(cells,[],eligible,set(),False)

@pytest.mark.parametrize('sign,rotated',[(1,False),(-1,True)])
def test_real_alias_aware_negative_preparation_then_target_and_full_native_replay(sign,rotated):
 case=source(cap=120,sign=sign,rotated=rotated)
 blocker=(4 if sign>0 else 11,10,9);goal=(9 if sign>0 else 6,9,9)
 support=np.zeros(case.observed_support.shape,bool);support[blocker]=support[goal]=True
 target=np.zeros(support.shape,np.float32);target[goal]=1
 case=ruled(replace(case,observed_support=support,nominal_target=target,reference_target=target,
  access=AccessWindow(case.access.center_mm,case.access.normal_inward,2.5)),TARGET_RAY_OBSTRUCTION_RULE)
 assert NominalCavityProposalConfig(**case.proposal_config.to_record()).fingerprint==case.proposal_config.fingerprint
 assert replace(case.proposal_config,obstruction_selection_rule=RESIDUAL_OBSTRUCTION_RULE).fingerprint!=case.proposal_config.fingerprint
 task=PagedObstructionTask(case,max_steps=3);planner=task.planning_clone();pages=list(planner.planning_action_pages())
 records=[r for page,_ in pages for r in page.candidate_inventory()['ledger']]
 accounting=pages[0][0].candidate_inventory()['obstruction_accounting']
 assert accounting['selected_cells']<=16 and accounting['added_candidates_to_verify']<=32
 assert accounting['target_ray_groups'] and all(list(blocker) in row['retained_co_blockers'] for row in accounting['target_ray_groups'])
 prep=next(r for r in records if r['verification']=='NATIVE_VERIFIED' and r['feasible'] and tuple(r['voxel'])==blocker)
 first=task.step(prep['action_id']);assert first.reward<0 and first.info['removed_indices_native']==[list(blocker)]
 successors=list(task.planning_clone().planning_action_pages());rows=[r for p,_ in successors for r in p.candidate_inventory()['ledger']]
 ray=next(r for r in rows if r['verification']=='NATIVE_VERIFIED' and r['feasible'] and tuple(r['voxel'])==goal and r['tool_id']=='generic_suction')
 assert any(r['proposal_reason']=='DUPLICATE_GEOMETRY' and r['action_id']==ray['action_id'] for r in rows)
 task.step(ray['action_id']);task.step('STOP');fresh=task.fresh()
 for action in (prep['action_id'],ray['action_id'],'STOP'):fresh.step(action)
 assert fresh.metrics()['history']==task.metrics()['history'] and fresh.metrics()['target_removed_mm3']==pytest.approx(case._native_config.voxel_volume_mm3)
 from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
 assert evaluate_native_spatial_episode(fresh)['accepted']
 assert planner._steps==0
