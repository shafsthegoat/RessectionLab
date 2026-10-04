"""Analytic proposal contracts only; no synthetic training or patient execution."""
from dataclasses import replace
import warnings

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow
from resectionlab.native_proposals import (
    DEFAULT_COLUMN_OFFSETS, NominalCavityProposalConfig, PreparedNominalCavityProposer,
)
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask, OPENING_TOOLS, make_native_opening_task


UNIT_OFFSETS = tuple((a,b) for a in (-1,0,1) for b in (-1,0,1))


def task_with_nominal_proposals(**changes):
    source = replace(make_native_opening_task().case, proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(UNIT_OFFSETS), **changes)
    return NativeSpatialTask(source, max_steps=2)


def pick(task, tool, voxel):
    return next(row["action_id"] for row in task.candidate_inventory()["emitted"]
        if row["feasible"] and row["tool_id"] == tool and row["voxel"] == list(voxel))


def same_observation(a,b):
    assert a.action_ids == b.action_ids and a.action_tool_ids == b.action_tool_ids
    assert a.fingerprint == b.fingerprint
    for key in ("image_channels", "coverage", "channel_available", "affine_ras_mm", "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(a,key), getattr(b,key))


def test_opt_in_preserves_production_offsets_and_requires_permitted_target():
    assert NominalCavityProposalConfig().offsets_source_voxels == DEFAULT_COLUMN_OFFSETS
    assert len(DEFAULT_COLUMN_OFFSETS) == 13
    legacy = make_native_opening_task()
    assert legacy.case.proposal_mode == "fixed_lattice"
    assert legacy.candidate_inventory()["evaluated_slots"] == 12
    with pytest.raises(ValueError, match="permitted target evidence"):
        replace(legacy.case, nominal_target=None, proposal_mode="nominal_cavity_v1")
    with pytest.raises(ValueError, match="between one and96"):
        NominalCavityProposalConfig(max_candidates=128)


def test_geometry_only_engine_and_every_emitted_choice_gets_a_native_preview(monkeypatch):
    calls=[]
    original = NativeResectionEngine.preview_stroke
    def record(self, tool_id, tip_mm, **kwargs):
        calls.append((tool_id, tuple(tip_mm), tuple(kwargs["entry_mm"])))
        return original(self,tool_id,tip_mm,**kwargs)
    monkeypatch.setattr(NativeResectionEngine,"preview_stroke",record)
    task = task_with_nominal_proposals()
    calls = calls[-task.candidate_inventory()["emitted_count"]:]
    inventory=task.candidate_inventory()
    assert not task._config.target_labels.any()
    assert inventory["declared_slots"] == len(UNIT_OFFSETS)*2*3
    assert len(inventory["ledger"]) == inventory["declared_slots"]
    assert inventory["evaluated_slots"] == inventory["emitted_count"] == len(calls)
    assert inventory["accepted_count"] + inventory["rejected_count"] == len(calls)
    assert inventory["complete"] and inventory["ledger_complete"] and not inventory["crop_clipping"]
    assert set(calls)=={(r["tool_id"],tuple(r["tip_mm"]),tuple(r["entry_mm"])) for r in inventory["emitted"]}
    assert task.observation().action_ids[0]=="STOP"
    np.testing.assert_array_equal(task.case._nominal_proposer._nominal, task.case.nominal_target)


def test_paid_exposed_opening_enables_distal_nominal_cut_with_distinct_tool():
    task=task_with_nominal_proposals()
    initial=task.candidate_inventory()
    distal=[row for row in initial["emitted"] if row["family"]=="distal_nominal" and row["tool_id"]==OPENING_TOOLS[1].tool_id]
    assert distal and all(not row["feasible"] for row in distal)
    opening=pick(task,OPENING_TOOLS[0].tool_id,(4,4,1))
    assert task.step(opening).reward < 0
    next_inventory=task.candidate_inventory()
    assert next_inventory["provider_model_hash"]==initial["provider_model_hash"]
    assert next_inventory["emitted"] != initial["emitted"]
    distal=pick(task,OPENING_TOOLS[1].tool_id,(4,4,5))
    assert task.step(distal).reward > 0
    assert task.metrics()["target_removed_mm3"]==2
    assert task.independent_geometry_check().feasible
    terminal=task.candidate_inventory()
    assert terminal["terminal"] and terminal["terminated_slots"]==54
    assert terminal["emitted"]==[] and terminal["ledger"]==[] and terminal["remaining_steps"]==0


def test_hidden_reference_changes_neither_proposals_masks_nor_teacher_before_or_after_cut():
    first=task_with_nominal_proposals()
    second=NativeSpatialTask(replace(first.case,reference_target=np.ones(first.case.reference_target.shape)),max_steps=2)
    same_observation(first.observation(),second.observation())
    assert first.candidate_inventory()==second.candidate_inventory()
    action=pick(first,OPENING_TOOLS[0].tool_id,(4,4,1))
    assert first.step(action).reward != second.step(action).reward
    same_observation(first.observation(),second.observation())
    assert first.candidate_inventory()==second.candidate_inventory()
    a,b=first.planning_clone(),second.planning_clone()
    assert a.candidate_inventory()==b.candidate_inventory() and a.metrics()==b.metrics()


def test_cap_dispositions_and_geometry_duplicates_are_explicit():
    source=replace(make_native_opening_task().case, proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(UNIT_OFFSETS,max_candidates=2))
    task=NativeSpatialTask(source)
    inventory=task.candidate_inventory()
    assert inventory["emitted_count"]==2 and inventory["omitted_count"]>0
    assert not inventory["complete"] and inventory["ledger_complete"]
    assert sum(inventory["disposition_counts"].values())==inventory["declared_slots"]
    assert task.observation().action_ids[0]=="STOP"
    nominal=np.zeros(source.nominal_target.shape,np.float32);nominal[4,4,5]=1
    duplicate_task=NativeSpatialTask(replace(source,nominal_target=nominal,
        proposal_config=NominalCavityProposalConfig(UNIT_OFFSETS)))
    duplicate_inventory=duplicate_task.candidate_inventory()
    assert duplicate_inventory["duplicate_count"]>=2
    geometry=[(r["tool_id"],tuple(r["entry_mm"]),tuple(r["tip_mm"])) for r in duplicate_inventory["emitted"]]
    assert len(geometry)==len(set(geometry))


def test_nominal_extent_is_not_clipped_to_actor_crop_and_geometry_failure_is_retained():
    shape=(9,9,40)
    support=np.zeros(shape,bool);support[4,4,1:37]=True
    nominal=np.zeros(shape,np.float32);nominal[4,4,35:37]=1
    source=NativeSpatialCase(support.astype(float),support,nominal,np.eye(4),
        AccessWindow((4,4,.5),(0,0,1),6),OPENING_TOOLS,nominal_target=nominal,
        target_derivation="explicit analytic annotation",crop_shape=(9,9,16),
        proposal_mode="nominal_cavity_v1",proposal_config=NominalCavityProposalConfig(((0,0),)))
    task=NativeSpatialTask(source)
    inventory=task.candidate_inventory()
    distal=[row for row in inventory["emitted"] if row["family"]=="distal_nominal"]
    assert len(distal)==2 and all(row["voxel"]==[4,4,36] for row in distal)
    assert all(not row["endpoint_center_in_actor_crop"] for row in distal)
    assert all(not row["feasible"] and row["reason"].startswith("HARD_GEOMETRY:") for row in distal)
    assert inventory["endpoint_centers_in_actor_crop"] < inventory["emitted_count"]


def test_proposer_never_accepts_engine_labels_or_misbound_nominal_evidence():
    task=task_with_nominal_proposals();provider=task.case._nominal_proposer
    labels=np.zeros(task._config.tissue_mask.shape,np.int16);labels[4,4,5]=1
    with pytest.raises(ValueError,match="zero engine target labels"):
        PreparedNominalCavityProposer(replace(task._config,target_labels=labels),task.case.nominal_target,
            nominal_provenance=provider._provenance)
    with pytest.raises(ValueError,match="exact permitted target"):
        PreparedNominalCavityProposer(task._config,task.case.nominal_target,
            nominal_provenance={**provider._provenance,"nominal_target_hash":"sha256:"+"f"*64})
    task._engine.remaining_mask[4,4,1]=False
    with pytest.raises(RuntimeError,match="legitimate committed history"):
        provider.propose(task._engine)


def test_original_index_and_reconciled_physical_frame_keep_actual_entries():
    base=make_native_opening_task().case
    original=np.eye(4);original[0,2]=7e-10
    normal=original[:3,2]/np.linalg.norm(original[:3,2])
    access=AccessWindow(original[:3,:3]@base.access.center_mm,normal,base.access.radius_mm)
    task=task_with_nominal_proposals(affine_ras_mm=original,access=access,
        native_grid_reconciliation="orthogonal_roundoff_1e-6mm")
    for row in task.candidate_inventory()["emitted"]:
        expected=task.case._native_affine_ras_mm[:3,:3]@row["voxel"]+task.case._native_affine_ras_mm[:3,3]
        np.testing.assert_array_equal(row["tip_mm"],expected)
        assert abs((np.array(row["entry_mm"])-access.center_mm)@access.normal_inward)<1e-12
    task.step(pick(task,OPENING_TOOLS[0].tool_id,(4,4,1)))
    task.step(pick(task,OPENING_TOOLS[1].tool_id,(4,4,5)))
    assert task.independent_geometry_check().feasible


def test_cancellation_and_nominal_array_layout_tampering_are_refused():
    task=task_with_nominal_proposals();provider=task.case._nominal_proposer
    with pytest.raises(InterruptedError):
        provider.propose(task._engine,cancelled=lambda:True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Setting the strides on a NumPy array",category=DeprecationWarning)
        provider._nominal.strides=provider._nominal.strides[::-1]
    with pytest.raises((ValueError,RuntimeError)):
        provider.propose(task._engine)


def test_batch_dispositions_are_bound_to_current_cavity_and_cannot_authorize_changed_rays():
    task=task_with_nominal_proposals();provider=task.case._nominal_proposer
    batch=provider.propose(task._engine)
    provider.validate_batch(batch,task._engine)
    changed=replace(batch,ledger=batch.ledger[:-1])
    with pytest.raises(ValueError,match="stale, altered"):
        provider.validate_batch(changed,task._engine)
    task.step(pick(task,OPENING_TOOLS[0].tool_id,(4,4,1)))
    with pytest.raises(ValueError,match="stale, altered"):
        provider.validate_batch(batch,task._engine)


@pytest.mark.parametrize("kwargs,error", [({"config":{}},TypeError), ({"index_frame_record":[]},ValueError)])
def test_provider_rejects_untyped_rule_or_frame_record(kwargs,error):
    task=task_with_nominal_proposals()
    with pytest.raises(error):
        PreparedNominalCavityProposer(task._config,task.case.nominal_target,
            nominal_provenance=task.case._nominal_proposer._provenance,**kwargs)
