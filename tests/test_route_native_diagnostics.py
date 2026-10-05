from types import SimpleNamespace

import numpy as np

from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.native_simulation import NativeSequentialSimulator
from resectionlab.route_native_diagnostics import initial_native_action_diagnostic, route_proposal_coverage


def simulator(tool):
    tissue = np.zeros((13, 13, 20), bool)
    tissue[2:11, 2:11, 4:18] = True
    labels = np.zeros_like(tissue, np.int16)
    labels[5:8, 5:8, 10:16] = 1
    config = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow([6, 6, 3.5], [0, 0, 1], 4), (tool,), "synthetic-source", "analytic box")
    return NativeSequentialSimulator(config, [[6, 6, 13]], max_actions=3, max_steps=2)


def test_generic_static_route_tool_does_not_imply_native_cutting_action():
    sim = simulator(GENERIC_TOOLS[1])
    before = sim.engine.state_hash
    report = initial_native_action_diagnostic(sim)
    assert report["status"] == "no_legal_native_actions"
    assert report["non_stop_action_count"] == 0
    assert report["rejection_counts"] == {"SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE": 1}
    assert report["tool_dimensions"][0]["shaft_wider_than_active_tip"]
    assert sim.engine.state_hash == before
    assert sim.metrics()["environment_steps"] == 0


def test_explicit_alternative_tool_can_enable_a_native_action_without_training():
    sim = simulator(NATIVE_GENERIC_TOOLS[0])
    report = initial_native_action_diagnostic(sim)
    assert report["policy_has_nontrivial_action_choice"]
    assert report["non_stop_action_count"] == 1
    assert not sim.engine.removed_mask.any()


def test_missing_selected_endpoint_and_changed_entry_are_reported_separately():
    sim = simulator(NATIVE_GENERIC_TOOLS[0])
    intended = SimpleNamespace(route_id="selected", target_mm=(7, 6, 13), entry_mm=(6, 6, 3.5))
    missing = route_proposal_coverage(intended, sim)
    assert not missing["selected_route_target_proposed"]
    assert "selected_route_target_not_proposed" in missing["proposal_problems"]
    intended.target_mm = (6, 6, 13)
    intended.entry_mm = (5, 6, 3.5)
    changed = route_proposal_coverage(intended, sim)
    assert changed["selected_route_target_proposed"] and not changed["selected_route_exact_ray_proposed"]
    intended.entry_mm = (6, 6, 3.5)
    assert route_proposal_coverage(intended, sim)["selected_route_exact_ray_proposed"]
