"""One generated causal control of the actual prospective helper/replay seam."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from resectionlab.evaluation import independent_check_native_history
from resectionlab.geometry import AccessWindow
from resectionlab.paged_obstruction import PagedObstructionTask
from test_obstruction_opening_proposals import source

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prospective_residual_helper',HERE/'residual_frontier.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_prospective_catalog_native_replay_independent_geometry_and_no_authority_mutation():
    case=source(cap=120)
    support=np.zeros(case.observed_support.shape,bool);support[4,10,9]=support[9,9,9]=True
    target=np.zeros(support.shape,np.float32);target[9,9,9]=1
    case=replace(case,observed_support=support,nominal_target=target,reference_target=target,
        access=AccessWindow(case.access.center_mm,case.access.normal_inward,2.5),
        proposal_config=replace(case.proposal_config,obstruction_page_size=8))
    task=PagedObstructionTask(case,max_steps=3).planning_clone()
    before=helper.state(task._engine);records={}
    def write(name,value):
        assert name not in records
        records[name]=value
    result=helper.diagnose(task,write=write)
    histories=result.pop('_audit_histories')
    assert result['branch_count']==1 and result['feasible_extras']==1
    assert result['original_task_state_unchanged'] and helper.state(task._engine)==before
    assert task._steps==0 and task._history==[]
    branch=result['branch_results'][0]
    assert branch['first_blockers_removed']==[(4,10,9)]
    assert branch['prefix_plus_prep_STOP_return']<0
    assert not branch['accepted_plan_claim'] and not branch['complete_new_rule_task_episode']
    assert branch['deterministic_native_commit_replay_exact']
    unlocked=branch['newly_feasible_original_rays']
    assert any(tuple(row['proposal']['voxel'])==(9,9,9) and row['nominal_increment']>0 for row in unlocked)
    assert all(row['executed'] is False for row in unlocked)
    raw=records['branch-00/result.json']
    assert raw['full_native_history']==histories[0]
    bound=SimpleNamespace(mri=case.structural_intensity,affine=case._native_affine_ras_mm,
        frame='RAS+',semantic_hash=task._engine.config.source_hash)
    audit=independent_check_native_history(bound,case.tools,histories[0],
        tissue_mask=case.observed_support,access=case.access,tool_modes=task.tool_modes,
        interaction_domain=case._interaction_domain,post_exposure=case.post_exposure)
    assert audit.feasible and audit.action_count==1 and audit.complete_tool_checked and audit.frontier_checked
    # Refused evidence is from the actual native engine, not a display-point approximation.
    failed=[row for name,row in records.items() if name.startswith('base-ray-') and row['sidecar']]
    assert failed and all(not row['feasible'] and row['sidecar']['complete_first_failure_set'] for row in failed)
    assert records['prospective-catalog.json']['old_proposal_model_hash']!=records['prospective-catalog.json']['prospective_proposal_model_hash']
