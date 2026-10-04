from dataclasses import replace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.native_simulation import NativeSequentialSimulator, native_beam_search
from resectionlab.simulation import InvalidActionError
from resectionlab.worlds import WorldGeneratorConfig


def fixture():
    tissue = np.ones((7, 7, 6), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 3:] = 1
    config = NativeResectionConfig(tissue, labels, np.eye(4),
                                   AccessWindow((3, 3, -.5), (0, 0, 1), 3.),
                                   NATIVE_GENERIC_TOOLS, "native-analytic-source", "synthetic solid cube")
    return config


def test_native_adapter_is_paid_and_contained_with_distinct_partial_contacts():
    sim = NativeSequentialSimulator(fixture(), [(3, 3, 4), (2, 3, 4), (4, 3, 4)], max_steps=3)
    assert not sim.removed_mask.any()
    assert sim.observation().action_features.shape[1] == 15
    result = sim.step(1)
    assert result.info["normal_removed_mm3"] > 0
    assert result.info["native_footprint"] == "fully_contained_connected_cells_v1"
    assert sim.metrics()["cumulative_partial_normal_contact_mm3"] > 0
    assert sim.metrics()["motor_surrogate"] is None
    assert sim.metrics()["language_surrogate"] is None
    assert np.count_nonzero(sim.exposed_mask) > np.count_nonzero(sim.removed_mask)


def test_native_search_changes_cavity_and_replays_exactly():
    sim = NativeSequentialSimulator(fixture(), [(3, 3, 4), (2, 3, 4), (4, 3, 4)], max_steps=3)
    result = native_beam_search(sim, max_expansions=24, max_wall_seconds=10.)
    assert result.environment_steps > 1
    assert not sim.removed_mask.any()
    sim.replay(result.actions)
    metrics = sim.metrics()
    assert metrics["total_reward"] == pytest.approx(result.nominal_score)
    removed = sim.removed_mask.copy()
    sim.replay(result.actions)
    np.testing.assert_array_equal(sim.removed_mask, removed)
    assert sim.metrics() == metrics


def test_native_hidden_world_is_not_in_actor_features():
    config = fixture()
    motor = np.zeros(config.tissue_mask.shape)
    motor[2, :, :] = 1
    sim = NativeSequentialSimulator(config, [(3, 3, 4)], nominal_motor=motor,
                                    world_generator=WorldGeneratorConfig(translation_scale_mm=(.3, .3, .3)))
    one = sim.reset(10)
    world = sim.metrics()["episode_world_hash"]
    two = sim.reset(11)
    assert world != sim.metrics()["episode_world_hash"]
    np.testing.assert_array_equal(one.action_features, two.action_features)
    np.testing.assert_array_equal(one.state_features, two.state_features)
    np.testing.assert_array_equal(one.action_mask, two.action_mask)


def test_native_repeated_action_gives_no_duplicate_removal_credit():
    sim = NativeSequentialSimulator(fixture(), [(3, 3, 4)], max_steps=3)
    first = sim.observation().action_ids[1]
    sim.step(first)
    before = sim.metrics()
    with pytest.raises(InvalidActionError):
        sim.step(first)
    assert sim.metrics() == before
    assert sim.step("STOP").reward == 0
    assert sim.step("STOP").reward == 0


def test_native_adapter_detects_objective_or_proposal_mutation():
    sim = NativeSequentialSimulator(fixture(), [(3, 3, 4)])
    sim.partial_contact_weight = .001
    with pytest.raises(RuntimeError, match="partial-contact objective"):
        sim.step("STOP")
    sim = NativeSequentialSimulator(fixture(), [(3, 3, 4)])
    sim.candidate_tips_mm = sim.candidate_tips_mm.view(np.int64)
    with pytest.raises(RuntimeError, match="partial-contact objective"):
        sim.step("STOP")


def test_native_factory_ras_lps_geometry_is_equivalent():
    from resectionlab.imaging import create_synthetic_case
    from resectionlab.native_simulation import make_native_patient_simulator
    ras = create_synthetic_case((24, 24, 24))
    lps = replace(ras, frame="LPS+", affine=np.diag([-1., -1., 1., 1.]) @ ras.affine)
    first = make_native_patient_simulator(ras, candidate_count=1, max_actions=3)
    second = make_native_patient_simulator(lps, candidate_count=1, max_actions=3)
    np.testing.assert_allclose(first.config.affine, second.config.affine)
    np.testing.assert_allclose(first.config.access.center_mm, second.config.access.center_mm)
    np.testing.assert_allclose(first.candidate_tips_mm, second.candidate_tips_mm)
    np.testing.assert_array_equal(first.config.target_labels, second.config.target_labels)
    assert "source_frame=LPS+" in second.native_config.tissue_support_provenance


def test_native_factory_rejects_contradictory_tissue_metadata():
    from resectionlab.imaging import create_synthetic_case
    from resectionlab.native_simulation import make_native_patient_simulator
    case = create_synthetic_case((24, 24, 24))
    full_head = replace(case, brain_mask=None, metadata={"source_collection": {"name": "UCSF-PDGM"}, "skull_stripped": False})
    with pytest.raises(ValueError, match="reviewed brain mask"):
        make_native_patient_simulator(full_head)
    missing_target = case.brain_mask.copy()
    missing_target[next(iter(case.compartments.values()))] = False
    inconsistent = replace(case, brain_mask=missing_target)
    with pytest.raises(ValueError, match="outside the supplied brain mask"):
        make_native_patient_simulator(inconsistent)


def test_native_cancellation_stops_preparation_and_next_scan():
    with pytest.raises(InterruptedError, match="cancelled"):
        NativeSequentialSimulator(fixture(), [(3, 3, 4)], cancelled=lambda: True)
    cancelled = [False]
    sim = NativeSequentialSimulator(fixture(), [(3, 3, 4)], cancelled=lambda: cancelled[0])
    cancelled[0] = True
    with pytest.raises(InterruptedError, match="cancelled"):
        sim.reset()


def test_parallel_native_columns_charge_access_and_preserve_entry_poses():
    sim = NativeSequentialSimulator(fixture(), [(2, 3, 4), (4, 3, 4)],
                                    candidate_entries_mm=[(2, 3, -.5), (4, 3, -.5)], max_steps=3)
    actions = sim.observation().action_ids
    first = next(action for action in actions if action.endswith(":0"))
    second = next(action for action in actions if action.endswith(":1"))
    one = sim.step(first)
    two = sim.step(second)
    assert one.info["normal_removed_mm3"] > 0 and two.info["normal_removed_mm3"] > 0
    assert one.info["entry_mm"] == (2., 3., -.5)
    assert two.info["entry_mm"] == (4., 3., -.5)
    first_cells = {tuple(cell) for cell in one.info["removed_indices_native"]}
    second_cells = {tuple(cell) for cell in two.info["removed_indices_native"]}
    assert first_cells.isdisjoint(second_cells)
    assert sim.metrics()["simulated_removed_target_volume_mm3"] == pytest.approx(one.info["target_removed_mm3"] + two.info["target_removed_mm3"])
