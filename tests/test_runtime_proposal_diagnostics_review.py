"""Independent runtime diagnostics binding on analytic fixtures only."""
import copy

import numpy as np
import pytest

from resectionlab.spatial_policy_diagnostics import runtime_proposal_coverage, nominal_depth_coverage
from test_nominal_cavity_proposals_review import tiny_task


def test_runtime_report_separates_rejected_envelope_from_accepted_and_early_stop():
    task = tiny_task()
    inventory, observation = task.candidate_inventory(), task.observation()
    report = runtime_proposal_coverage(task.case, inventory, observation)
    assert report["emitted_envelope"]["proposal_count"] == inventory["emitted_count"]
    assert report["accepted_envelope"]["proposal_count"] == inventory["accepted_count"]
    assert report["emitted_envelope"]["distal_active_tip_axial_upper_bound_mm"] > report["accepted_envelope"]["distal_active_tip_axial_upper_bound_mm"]
    assert report["nominal_target_centers_total"] == np.count_nonzero(task.case.nominal_target)
    assert sum(report["preview_dispositions"].values()) == inventory["emitted_count"]
    assert sum(report["proposal_dispositions"].values()) == inventory["declared_slots"]
    assert "future dynamic catalogs may extend" in report["interpretation"]
    with pytest.raises(ValueError, match="actual emitted inventory"):
        nominal_depth_coverage(task.case)
    task.step("STOP")
    terminal = runtime_proposal_coverage(task.case, task.candidate_inventory(), task.observation())
    assert terminal["episode_terminal"] and terminal["remaining_step_budget"] == 1
    assert terminal["actions"] == [] and terminal["accepted_envelope"]["distal_active_tip_axial_upper_bound_mm"] is None


@pytest.mark.parametrize("mutation", ["tip", "entry", "tool"])
def test_runtime_report_refuses_accepted_geometry_changed_under_same_legal_identifier(mutation):
    task = tiny_task()
    inventory = copy.deepcopy(task.candidate_inventory())
    row = next(r for r in inventory["emitted"] if r["feasible"])
    if mutation == "tool":
        row["tool_id"] = next(tool.tool_id for tool in task.case.tools if tool.tool_id != row["tool_id"])
    else:
        row[mutation + "_mm"][2] += 40.
    with pytest.raises(ValueError):
        runtime_proposal_coverage(task.case, inventory, task.observation())


def test_runtime_report_binds_explicit_original_and_derived_physical_frames():
    affine = np.eye(4)
    affine[0, 2] = 8e-10
    task = tiny_task(affine=affine)
    native = task.case._native_affine_ras_mm
    report = runtime_proposal_coverage(task.case, task.candidate_inventory(), task.observation(), native_affine=native)
    assert report["original_source_affine_ras_mm"] == affine.tolist()
    assert report["native_physical_affine_ras_mm"] == native.tolist()


@pytest.mark.parametrize("mutation", ["rejected_tip", "source_hash", "steps_taken", "accepted_order"])
def test_runtime_report_refuses_inconsistent_catalog_geometry_source_or_observed_state(mutation):
    task = tiny_task()
    inventory = copy.deepcopy(task.candidate_inventory())
    if mutation == "rejected_tip":
        row = next(r for r in inventory["emitted"] if not r["feasible"])
        row["tip_mm"][2] += 40.
    elif mutation == "source_hash":
        inventory["source_hash"] = "sha256:" + "f" * 64
    elif mutation == "steps_taken":
        inventory["steps_taken"] += 1
    else:
        indices = [i for i, row in enumerate(inventory["emitted"]) if row["feasible"]]
        assert len(indices) >= 2
        first, second = indices[:2]
        inventory["emitted"][first], inventory["emitted"][second] = inventory["emitted"][second], inventory["emitted"][first]
    with pytest.raises(ValueError):
        runtime_proposal_coverage(task.case, inventory, task.observation())
