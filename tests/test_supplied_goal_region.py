"""Generated task semantics; no patient arrays, search, models or updates."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.native_proposals import (
    NominalCavityProposalConfig, PreparedNominalCavityProposer,
    SUPPLIED_GOAL_REGION, TARGET_WITHIN_SUPPORT,
)
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task


def public_fixture(*, outside_only=False):
    old = make_native_opening_task().case
    target = np.zeros(old.observed_support.shape, np.float32)
    if not outside_only:
        target[4, 4, 4:6] = 1
    target[4, 4, 6] = 1  # Unoccupied cell remains in the supplied goal.
    return replace(old, track="annotation_assisted", support_source_kind="supplied_annotation",
        target_source_kind="supplied_annotation", support_derivation="generated supplied-support control",
        target_derivation="generated supplied-goal control; outside annotation domain is not normal truth",
        reference_target=target, nominal_target=target, target_semantics=SUPPLIED_GOAL_REGION,
        proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))))


def pick(task, tool, voxel):
    return next(row["action_id"] for row in task.candidate_inventory()["emitted"]
                if row["feasible"] and row["tool_id"] == tool and row["voxel"] == list(voxel))


def test_default_remains_strict_and_explicit_default_preserves_identity():
    old = make_native_opening_task().case
    same = replace(old, target_semantics=TARGET_WITHIN_SUPPORT)
    assert same.source_hash == old.source_hash
    a, b = NativeSpatialTask(old), NativeSpatialTask(same)
    assert a.decision_model_hash == b.decision_model_hash
    assert a.observation().fingerprint == b.observation().fingerprint
    assert "supplied_goal_region" not in a.metrics()
    with pytest.raises(ValueError, match="conflict"):
        replace(public_fixture(), target_semantics=TARGET_WITHIN_SUPPORT)


def test_explicit_region_preserves_arrays_and_full_denominator():
    source = public_fixture()
    old = make_native_opening_task().case
    np.testing.assert_array_equal(source.observed_support, old.observed_support)
    assert source.nominal_target[4, 4, 6] == source.reference_target[4, 4, 6] == 1
    task = NativeSpatialTask(source)
    assert task.observation().image_channels[2, 4, 4, 6] == 1
    extent = task.metrics()["supplied_goal_region"]
    assert extent["condition"] == "PARTIAL_TARGET_PROGRESS"
    assert extent["full_region_positive_voxels"] == 3
    assert extent["unsupported_region_positive_voxels"] == 1
    assert extent["full_region_membership_mm3"] == 3
    assert extent["supported_region_membership_mm3"] == 2
    assert extent["fraction_of_full_region_removed"] == 0
    assert not extent["occupancy_modified"] and not extent["target_modified"]


def test_semantics_are_versioned_even_when_target_fits_support():
    source = public_fixture()
    target = source.nominal_target.copy(); target[4, 4, 6] = 0
    strict = replace(source, nominal_target=target, reference_target=target,
                     target_semantics=TARGET_WITHIN_SUPPORT)
    region = replace(strict, target_semantics=SUPPLIED_GOAL_REGION)
    assert strict.source_hash != region.source_hash
    assert NativeSpatialTask(strict).decision_model_hash != NativeSpatialTask(region).decision_model_hash
    assert strict._nominal_proposer.model_hash != region._nominal_proposer.model_hash


def test_full_native_execution_never_removes_or_rewards_unsupported_goal():
    task = NativeSpatialTask(public_fixture(), max_steps=2)
    assert all(task.case.observed_support[tuple(row["voxel"])]
               for row in task.candidate_inventory()["emitted"])
    task.step(pick(task, "short-wide-opener", (4, 4, 1)))
    task.step(pick(task, "long-narrow-cutter", (4, 4, 5)))
    assert not task._engine.removed_mask[4, 4, 6]
    assert not np.any(task._engine.removed_mask & ~task.case.observed_support)
    metrics = task.metrics()
    assert metrics["target_removed_mm3"] == 2
    assert sum(row["target_removed_mm3"] for row in metrics["history"]) == 2
    assert metrics["supplied_goal_region"]["fraction_of_full_region_removed"] == pytest.approx(2/3)
    assert metrics["supplied_goal_region"]["unsupported_region_membership_mm3"] == 1
    assert task.independent_geometry_check().feasible


def test_only_unsupported_goal_emits_no_nominal_ray_and_cannot_earn_target_reward():
    task = NativeSpatialTask(public_fixture(outside_only=True), max_steps=1)
    assert all(row["family"] == "exposed_opening" for row in task.candidate_inventory()["emitted"])
    result = task.step(pick(task, "short-wide-opener", (4, 4, 1)))
    assert result.reward < 0
    assert task.metrics()["target_removed_mm3"] == 0
    assert task.metrics()["supplied_goal_region"]["fraction_of_full_region_removed"] == 0


def test_proposer_default_refuses_conflict_and_opt_in_requires_supplied_provenance():
    source = public_fixture(); provider = source._nominal_proposer
    with pytest.raises(ValueError, match="inside observed support"):
        PreparedNominalCavityProposer(source._native_config, source.nominal_target,
                                     nominal_provenance=provider._provenance)
    with pytest.raises(ValueError, match="explicit public annotation"):
        PreparedNominalCavityProposer(source._native_config, source.nominal_target,
            nominal_provenance={**provider._provenance, "target_semantics": TARGET_WITHIN_SUPPORT},
            target_semantics=SUPPLIED_GOAL_REGION)
    with pytest.raises(ValueError, match="Unknown explicit"):
        replace(source, target_semantics="infer_or_union_occupancy")
    with pytest.raises(ValueError, match="annotation-assisted"):
        replace(source, track="synthetic_scan")
