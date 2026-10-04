"""Analytic counterexamples for connected tissue removal and actor observability."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.geometry import ToolGeometry
from resectionlab.simulation import (InvalidActionError, RewardSpec, SequentialSimulator,
                                    beam_search, greedy_search, make_synthetic_simulator)
from resectionlab.worlds import WorldGeneratorConfig


def _first_removal(simulator):
    actions = simulator.observation().action_ids
    assert len(actions) > 1, "Fixture needs at least one executable full-tool frontier action"
    return actions[1]


def test_paid_corridor_and_known_optimum():
    sim = make_synthetic_simulator()
    assert not sim.cavity_mask.any()
    assert len(sim.proposed_actions()) == 2
    assert sim.proposed_actions()[1].target_index == (1, 1, 0)
    first = sim.step(_first_removal(sim))
    assert first.reward == pytest.approx(-.22)
    assert first.info["normal_removed_mm3"] == 1
    assert sim.metrics()["simulated_removed_target_volume_mm3"] == 0
    sim.step(_first_removal(sim))
    sim.step(_first_removal(sim))
    sim.step("STOP")
    assert sim.total_reward == pytest.approx(1.74)
    assert sim.metrics()["modeled_residual_target_volume_mm3"] == 0
    assert sim.metrics()["simulated_removed_normal_volume_mm3"] == 1
    assert sim.metrics()["clinical_deficit_probability"] is None


def test_stop_is_always_available_and_has_no_terminal_bonus():
    sim = make_synthetic_simulator()
    for _ in range(3):
        assert sim.action_mask()[0]
        result = sim.step("STOP")
        assert result.reward == 0 and result.terminated
    assert sim.metrics()["environment_steps"] == 0
    assert not sim.removed_mask.any()


def test_internal_voxel_and_repeated_removal_rejected_without_mutation():
    sim = make_synthetic_simulator()
    with pytest.raises(InvalidActionError):
        sim.step("REMOVE:synthetic-cannula:1,1,2")
    assert not sim.removed_mask.any()
    action = _first_removal(sim)
    sim.step(action)
    before = sim.metrics()
    with pytest.raises(InvalidActionError):
        sim.step(action)
    assert sim.metrics() == before


def test_complete_shaft_larger_than_aperture_is_rejected():
    sim = make_synthetic_simulator()
    wide = ToolGeometry("wide", .2, .7, 10, max_access_angle_deg=5, tip_length_mm=.2)
    sim = SequentialSimulator(replace(sim.config, tools=(wide,)))
    assert sim.observation().action_ids == ("STOP",)


def test_replay_and_cavity_are_deterministic_and_monotonic():
    sim = make_synthetic_simulator()
    actions = []
    previous = sim.cavity_mask
    for _ in range(3):
        action = _first_removal(sim)
        actions.append(action)
        sim.step(action)
        assert np.all(sim.cavity_mask[previous])
        previous = sim.cavity_mask
    actions.append("STOP")
    sim.step("STOP")
    expected = sim.metrics()
    sim.replay(actions)
    assert sim.metrics() == expected


def test_graph_edge_is_disrupted_once():
    original = make_synthetic_simulator()
    edge = np.zeros(original.remaining_mask.shape, bool)
    edge[1, 1, 1:] = True
    sim = SequentialSimulator(replace(original.config, graph_edges={"motor-edge": edge}))
    sim.step(_first_removal(sim))
    first = sim.step(_first_removal(sim))
    second = sim.step(_first_removal(sim))
    assert first.info["newly_severed_edges"] == ["motor-edge"]
    assert second.info["newly_severed_edges"] == []
    assert sim.total_reward == pytest.approx(.74)


def test_hidden_world_never_changes_actor_observation_or_action_mask():
    original = make_synthetic_simulator()
    hazard = np.zeros(original.remaining_mask.shape)
    hazard[1, 1, 1] = 1
    config = replace(original.config, nominal_motor=hazard,
                     world_generator=WorldGeneratorConfig(translation_scale_mm=(.7, .7, .7)))
    sim = SequentialSimulator(config)
    first = sim.reset(11)
    world_a = sim.metrics()["episode_world_hash"]
    second = sim.reset(12)
    assert world_a != sim.metrics()["episode_world_hash"]
    np.testing.assert_array_equal(first.action_features, second.action_features)
    np.testing.assert_array_equal(first.state_features, second.state_features)
    np.testing.assert_array_equal(first.action_mask, second.action_mask)
    before_world = sim.metrics()["episode_world_hash"]
    sim.step(_first_removal(sim))
    assert sim.metrics()["episode_world_hash"] == before_world
    assert sim.world_generator_fingerprint == config.world_generator.fingerprint


def test_model_arrays_and_mappings_cannot_mutate():
    sim = make_synthetic_simulator()
    with pytest.raises(ValueError):
        sim.config.target_labels.setflags(write=True)
    with pytest.raises(TypeError):
        sim.config.compartment_names[1] = "changed"
    object.__setattr__(sim.config, "reward", RewardSpec(target_per_mm3=100))
    with pytest.raises(RuntimeError, match="Decision model changed"):
        sim.step("STOP")


def test_search_can_pay_access_cost_that_immediate_greedy_avoids():
    sim = make_synthetic_simulator()
    greedy = greedy_search(sim)
    searched = beam_search(sim, max_expansions=16)
    assert greedy.actions == ("STOP",)
    assert greedy.nominal_score == 0
    assert searched.nominal_score == pytest.approx(1.74)
    sim.replay(searched.actions)
    assert sim.total_reward == pytest.approx(searched.nominal_score)
    assert not make_synthetic_simulator().removed_mask.any()


def test_partial_target_cell_pays_for_normal_tissue_and_preserves_compartments():
    original = make_synthetic_simulator()
    fractions = np.zeros(original.remaining_mask.shape)
    fractions[1, 1, 1:] = .25
    sim = SequentialSimulator(replace(original.config, target_fractions={1: fractions}))
    sim.step(_first_removal(sim))
    one = sim.step(_first_removal(sim))
    assert one.info["target_removed_mm3"] == pytest.approx(.25)
    assert one.info["normal_removed_mm3"] == pytest.approx(.75)
    assert one.reward == pytest.approx(.25 - .75 * .2 - .02)
    assert sim.metrics()["modeled_residual_target_volume_mm3"] == pytest.approx(.25)
    assert sim.metrics()["removed_by_compartment_mm3"]["enhancing"] == pytest.approx(.25)


def test_missing_functional_evidence_is_null_in_results():
    original = make_synthetic_simulator()
    sim = SequentialSimulator(replace(original.config, evidence_available=(False, False)))
    assert sim.metrics()["motor_surrogate"] is None
    assert sim.metrics()["language_surrogate"] is None
    np.testing.assert_array_equal(sim.observation().state_features[-2:], [0, 0])


def test_patient_pooling_preserves_source_volumes_and_native_affine():
    from resectionlab.imaging import create_synthetic_case
    from resectionlab.simulation import make_patient_simulator
    case = create_synthetic_case()
    sim = make_patient_simulator(case, block_size=4, proposal_scan_limit=12)
    source = sum(np.count_nonzero(v) * case.voxel_volume_mm3 for v in case.compartments.values())
    assert sim.metrics()["modeled_residual_target_volume_mm3"] == pytest.approx(source)
    np.testing.assert_allclose(sim.config.affine[:3, :3], case.affine[:3, :3] * 4)
    np.testing.assert_allclose(sim.config.affine[:3, 3], case.affine[:3, 3] + case.affine[:3, :3] @ [1.5, 1.5, 1.5])
    assert not sim.cavity_mask.any()
    assert sim.case_hash == case.semantic_hash


def test_full_head_or_unknown_source_cannot_use_mri_as_brain_envelope():
    from resectionlab.imaging import create_synthetic_case
    from resectionlab.simulation import make_patient_simulator
    case = replace(create_synthetic_case((24, 24, 24)), brain_mask=None)
    with pytest.raises(ValueError, match="reviewed brain mask"):
        make_patient_simulator(case)


def test_branching_fixture_has_paid_access_and_exhaustive_reference():
    from resectionlab.simulation import make_branching_simulator
    sim = make_branching_simulator()
    assert {a.target_index for a in sim.proposed_actions()[1:]} == {(1, 1, 0)}
    sim.step(1)
    locations = {a.target_index for a in sim.proposed_actions()[1:]}
    assert (0, 1, 0) in locations and (2, 1, 0) in locations
    # The wider shaft cannot take the same immediate oblique branch.
    assert not any(a.target_index == (0, 1, 0) and a.tool_id == "branch-wide" for a in sim.proposed_actions())
    initial = make_branching_simulator()
    reference = beam_search(initial, beam_width=20000, max_expansions=20000)
    assert reference.termination == "search_exhausted"
    assert reference.environment_steps == 814
    assert reference.nominal_score == pytest.approx(1.3)
    initial.replay(reference.actions)
    assert initial.metrics()["simulated_removed_normal_volume_mm3"] == 3
    assert initial.metrics()["simulated_removed_target_volume_mm3"] == 2


def test_deterministic_world_preserves_oblique_grid_fields_exactly():
    from scipy.spatial.transform import Rotation
    original = make_synthetic_simulator()
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler("xyz", [13, 27, 8], degrees=True).as_matrix()
    affine[:3, 3] = [17.3, -81.5, .7]
    evidence = np.arange(27, dtype=float).reshape(3, 3, 3) / 27
    sim = SequentialSimulator(replace(original.config, affine=affine, nominal_motor=evidence))
    np.testing.assert_array_equal(sim._hidden_motor, evidence)
    assert sim._hidden_known_coverage.all()


def test_fresh_coarse_arm_does_not_share_dynamic_caches():
    sim = make_synthetic_simulator()
    sim.step(1)
    fresh = sim.fresh()
    assert type(fresh) is SequentialSimulator
    assert fresh.decision_model_hash == sim.decision_model_hash
    assert not fresh.removed_mask.any()
    assert fresh._geometry_cache is not sim._geometry_cache
    changed = sim.fresh(max_steps=2, world_generator=WorldGeneratorConfig(family="rigid_uniform"))
    assert changed.config.max_steps == 2
    assert changed.decision_model_hash != sim.decision_model_hash
