"""Independent attacks on native tissue removal and its displayed evidence.

Source-grid geometry, active-tool containment, and accounting must agree. A
previously stored certificate must not excuse changed replay data or source mass.
"""

from copy import deepcopy
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.app.refinement import replay_artifact_hash, validate_replay
from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.native_simulation import NativeSequentialSimulator
from resectionlab.worlds import WorldGeneratorConfig


def _accounting_case_and_replay():
    target = np.zeros((4, 4, 4), bool)
    target[1, 1, 1] = True
    case = CaseData(
        "native-accounting-fixture", np.ones(target.shape), {"enhancing": target},
        np.eye(4), (SourceRef("synthetic", "synthetic://native-accounting", provenance="simulated"),),
        brain_mask=np.ones(target.shape, bool),
    )
    replay = {
        "case_hash": case.semantic_hash, "role": "selection", "final_evaluation": False,
        "independent_certificate": {"feasible": True},
        "native_removal_audit": {"feasible": True}, "source_block_size": 1,
        "shape": list(target.shape), "affine": case.affine.tolist(),
        "metrics": {
            "history": [{"removed_indices": [[1, 1, 1]], "target_removed_mm3": 1.0, "normal_removed_mm3": 0.0}],
            "simulated_removed_target_volume_mm3": 1.0,
            "simulated_removed_normal_volume_mm3": 0.0,
            "modeled_residual_target_volume_mm3": 0.0,
        },
    }
    replay["artifact_hash"] = replay_artifact_hash(replay)
    return case, replay


@pytest.mark.parametrize("tampering", ["history_only", "consistent_but_impossible_totals"])
def test_selection_replay_cannot_display_unsupported_source_volume(tampering):
    case, replay = _accounting_case_and_replay()
    assert validate_replay(case, replay)
    changed = deepcopy(replay)
    changed["metrics"]["history"][0]["target_removed_mm3"] = 1e9
    if tampering == "consistent_but_impossible_totals":
        changed["metrics"]["simulated_removed_target_volume_mm3"] = 1e9
    changed["artifact_hash"] = replay_artifact_hash(changed)
    with pytest.raises(ValueError):
        validate_replay(case, changed)


def test_selection_replay_cannot_relabel_normal_source_tissue_as_target():
    case, replay = _accounting_case_and_replay()
    assert validate_replay(case, replay)
    changed = deepcopy(replay)
    changed["metrics"]["history"][0]["removed_indices"] = [[0, 0, 0]]
    # Totals are numerically consistent, but this cell is ordinary source tissue.
    changed["artifact_hash"] = replay_artifact_hash(changed)
    with pytest.raises(ValueError):
        validate_replay(case, changed)


@pytest.mark.parametrize("microstep_mm", [0.5, 0.25, 0.1])
def test_shaft_cannot_borrow_tissue_removal_from_later_in_same_microstep(microstep_mm):
    tissue = np.zeros((5, 5, 10), bool)
    tissue[2, 2, 5] = True
    tool = ToolGeometry("short-active", 0.9, 0.1, 10.0, tip_length_mm=0.1)
    config = NativeResectionConfig(
        tissue, tissue.astype(np.int16), np.eye(4),
        AccessWindow([2, 2, 4.49], [0, 0, 1], 2.0), (tool,),
        "synthetic-source", "analytic single-cell tissue", max_tip_step_mm=microstep_mm,
    )
    engine = NativeResectionEngine(config)
    initial_state = engine.state_hash
    # The voxel lower face is z=4.5. Shaft contact begins before the active
    # capsule wholly contains that voxel. A coarse step must not clear it early.
    preview = engine.preview_stroke(tool.tool_id, [2, 2, 5.49])
    assert not preview.feasible
    assert "SHAFT_BLOCKED" in preview.reason
    assert engine.state_hash == initial_state
    assert not engine.removed_mask.any()
    np.testing.assert_array_equal(engine.remaining_mask, tissue)


def _native_corridor_engine(affine=None):
    affine = np.eye(4) if affine is None else np.asarray(affine, float)
    tissue = np.zeros((7, 7, 10), bool)
    tissue[1:6, 1:6, 2:8] = True
    target = np.zeros(tissue.shape, np.int16)
    target[3, 3, 4] = 1
    tool = ToolGeometry("analytic-aspiration", 1.25, 0.1, 10.0, tip_length_mm=2.0)
    entry = affine[:3, :3] @ [3, 3, 1.49] + affine[:3, 3]
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    config = NativeResectionConfig(
        tissue, target, affine, AccessWindow(entry, normal, 2.0), (tool,),
        "synthetic-native-source", "analytic native tissue mask",
        max_tip_step_mm=min(0.25, np.linalg.norm(affine[:3, :3], axis=0).min() / 2),
    )
    endpoint = affine[:3, :3] @ [3, 3, 4] + affine[:3, 3]
    return NativeResectionEngine(config), tool.tool_id, endpoint


def test_native_partial_contact_does_not_create_free_corridor_or_removal_credit():
    engine, tool_id, endpoint = _native_corridor_engine()
    before = engine.remaining_mask.copy()
    preview = engine.preview_stroke(tool_id, endpoint)
    assert preview.feasible
    np.testing.assert_array_equal(engine.remaining_mask, before)  # Preview is pure.
    engine.commit_preview(preview)
    assert engine.remaining_mask[4, 3, 4]  # Adjacent cell is touched, not contained.
    assert engine.contact_mask[4, 3, 4]
    assert not engine.removed_mask[4, 3, 4]
    expected_removed = np.zeros(before.shape, bool)
    expected_removed[3, 3, 2:5] = True
    np.testing.assert_array_equal(engine.removed_mask, expected_removed)
    metrics = engine.metrics()
    assert metrics["simulated_removed_target_mm3"] == 1.0
    assert metrics["simulated_removed_normal_mm3"] == 2.0
    assert metrics["contacted_but_not_removed_upper_bound_mm3"] > 0
    prior_state = engine.state_hash
    repeated = engine.execute_stroke(tool_id, endpoint)
    assert not repeated.feasible
    assert engine.state_hash == prior_state
    np.testing.assert_array_equal(engine.removed_mask, expected_removed)


def test_native_stroke_full_cell_removal_and_mass_are_rigid_frame_invariant():
    affine = np.diag([0.8, 1.2, 1.0, 1.0])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [21, -17, 37], degrees=True).as_matrix()
    transform[:3, 3] = [30.25, -12.1, 8.4]
    first, first_tool, first_tip = _native_corridor_engine(affine)
    rotated, rotated_tool, rotated_tip = _native_corridor_engine(transform @ affine)
    first_result = first.execute_stroke(first_tool, first_tip)
    rotated_result = rotated.execute_stroke(rotated_tool, rotated_tip)
    assert first_result.feasible
    assert rotated_result.feasible
    np.testing.assert_array_equal(first_result.removed_indices_native, rotated_result.removed_indices_native)
    for key in ("simulated_removed_target_mm3", "simulated_removed_normal_mm3", "modeled_residual_target_mm3"):
        assert rotated.metrics()[key] == pytest.approx(first.metrics()[key], abs=1e-10)
    assert first.metrics()["simulated_removed_target_mm3"] == pytest.approx(0.96)
    assert first.metrics()["simulated_removed_normal_mm3"] == pytest.approx(1.92)


def test_preview_array_descriptor_changes_cannot_redirect_certified_removal():
    tissue = np.ones((7, 7, 6), bool)
    tool = ToolGeometry("preview-integrity", 1.25, 0.1, 10.0, tip_length_mm=2.0)
    config = NativeResectionConfig(
        tissue, tissue.astype(np.int16), np.eye(4),
        AccessWindow([3, 3, -0.5], [0, 0, 1], 2.0), (tool,),
        "synthetic-source", "analytic dense tissue",
    )
    engine = NativeResectionEngine(config)
    preview = engine.preview_stroke(tool.tool_id, [3, 3, 4])
    assert preview.feasible
    before = engine.remaining_mask.copy()
    state = engine.state_hash
    # NumPy bytes backing prevents writes, but permits reinterpreting ndarray
    # descriptors. An object's identity alone is therefore not a certificate.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            preview.removed_indices_native.dtype = np.int32
            preview.removed_indices_native.shape = (-1, 3)
    except (AttributeError, TypeError, ValueError):
        assert engine.state_hash == state
        np.testing.assert_array_equal(engine.remaining_mask, before)
        return  # Preventing descriptor changes is also an intact certificate.
    with pytest.raises((ValueError, RuntimeError)):
        engine.commit_preview(preview)
    assert engine.state_hash == state
    np.testing.assert_array_equal(engine.remaining_mask, before)


@pytest.mark.parametrize("invalid_label", [0.5, -0.5, np.nan, np.inf])
def test_native_label_validation_precedes_integer_cast(invalid_label):
    engine, _, _ = _native_corridor_engine()
    labels = engine.config.target_labels.astype(float)
    labels[3, 3, 4] = invalid_label
    with pytest.raises((TypeError, ValueError)):
        NativeResectionConfig(
            engine.config.tissue_mask, labels, engine.config.affine,
            engine.config.access, engine.config.tools,
            "synthetic-source", "analytic source masks",
        )


def test_mutable_scalar_tool_dimension_cannot_change_frozen_model():
    radius = np.array(1.25)
    try:
        tool = ToolGeometry("mutable-dimension", radius, 0.1, 10.0, tip_length_mm=2.0)
    except (TypeError, ValueError):
        return  # Rejecting nonscalar-array inputs is also a valid contract.
    radius[...] = 0.01
    assert tool.tip_radius_mm == 1.25


def test_native_world_truth_stays_out_of_actor_after_a_removal_action():
    engine, _, endpoint = _native_corridor_engine()
    motor = np.indices(engine.config.tissue_mask.shape)[0].astype(float) / 6.0
    world = WorldGeneratorConfig(translation_scale_mm=(0.5, 0.0, 0.0))
    simulator = NativeSequentialSimulator(
        engine.config, [endpoint], nominal_motor=motor, world_generator=world,
    )
    first = simulator.clone()
    second = simulator.clone()
    first_observation = first.reset(10)
    second_observation = second.reset(11)
    first_world = first.metrics()["episode_world_hash"]
    second_world = second.metrics()["episode_world_hash"]
    assert first_world != second_world
    assert first_observation.action_ids == second_observation.action_ids
    assert len(first_observation.action_ids) > 1
    action_id = first_observation.action_ids[1]
    first_step = first.step(action_id)
    second_step = second.step(action_id)
    assert first_step.reward != pytest.approx(second_step.reward)
    assert first.metrics()["episode_world_hash"] == first_world
    assert second.metrics()["episode_world_hash"] == second_world
    assert first_step.observation.action_ids == second_step.observation.action_ids
    np.testing.assert_array_equal(first_step.observation.action_features, second_step.observation.action_features)
    np.testing.assert_array_equal(first_step.observation.state_features, second_step.observation.state_features)
    np.testing.assert_array_equal(first_step.observation.action_mask, second_step.observation.action_mask)
    np.testing.assert_array_equal(first.removed_mask, second.removed_mask)
