"""Selected routes remain exact physical rays during native preparation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS
from resectionlab.native_simulation import make_native_patient_simulator
from resectionlab.route_native_diagnostics import initial_native_action_diagnostic, route_proposal_coverage


def case_and_access():
    tissue = np.zeros((13, 13, 20), bool)
    tissue[2:11, 2:11, 4:18] = True
    target = np.zeros_like(tissue)
    target[5:8, 5:8, 10:16] = True
    case = CaseData("selected-route-analytic", tissue.astype(float), {"target": target}, np.eye(4),
        (SourceRef("analytic", "simulated:box", provenance="simulated"),), brain_mask=tissue)
    return case, AccessWindow((6, 6, 3.5), (0, 0, 1), 4)


def test_selected_route_uses_exact_entry_and_target_without_centroid_alternatives():
    case, access = case_and_access()
    ray = {"selected_entry_mm": (5, 6, 3.5), "selected_target_mm": (5, 6, 13)}
    sim = make_native_patient_simulator(case, access=access, tools=(NATIVE_GENERIC_TOOLS[0],),
                                       candidate_count=99, **ray)
    np.testing.assert_array_equal(sim.candidate_entries_mm, [ray["selected_entry_mm"]])
    np.testing.assert_array_equal(sim.candidate_tips_mm, [ray["selected_target_mm"]])
    assert sim.config.access == access
    assert sim.config.tools == (NATIVE_GENERIC_TOOLS[0],)
    selected = SimpleNamespace(route_id="chosen", entry_mm=ray["selected_entry_mm"], target_mm=ray["selected_target_mm"])
    assert route_proposal_coverage(selected, sim)["selected_route_exact_ray_proposed"]
    before = sim.metrics()
    report = initial_native_action_diagnostic(sim)
    assert report["non_stop_action_count"] == 1
    assert sim.metrics() == before
    result = sim.step(report["non_stop_action_ids"][0])
    assert result.info["entry_mm"] == ray["selected_entry_mm"]
    assert result.info["tip_mm"] == ray["selected_target_mm"]
    assert result.info["normal_removed_mm3"] > 0
    fresh = sim.fresh()
    np.testing.assert_array_equal(fresh.candidate_entries_mm, sim.candidate_entries_mm)
    np.testing.assert_array_equal(fresh.candidate_tips_mm, sim.candidate_tips_mm)
    assert fresh.decision_model_hash == sim.decision_model_hash


def test_blocked_selected_tool_is_reported_without_native_profile_substitution():
    case, access = case_and_access()
    sim = make_native_patient_simulator(case, access=access, tools=(GENERIC_TOOLS[1],),
        selected_entry_mm=(6, 6, 3.5), selected_target_mm=(6, 6, 13))
    report = initial_native_action_diagnostic(sim)
    assert report["status"] == "no_legal_native_actions"
    assert report["rejection_counts"] == {"SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE": 1}
    assert sim.config.tools == (GENERIC_TOOLS[1],)
    assert not sim.removed_mask.any()
    assert sim.engine.revision == 0


def test_selected_ras_ray_stays_ras_with_lps_source_and_lps_access():
    case, access = case_and_access()
    flip = np.diag([-1., -1., 1., 1.])
    lps = replace(case, frame="LPS+", affine=flip @ case.affine)
    lps_access = AccessWindow(flip[:3, :3] @ access.center_mm, flip[:3, :3] @ access.normal_inward,
                              access.radius_mm, access.window_id)
    ray = {"selected_entry_mm": (5, 6, 3.5), "selected_target_mm": (5, 6, 13)}
    sim = make_native_patient_simulator(lps, access=lps_access, access_frame="LPS+",
                                       tools=(NATIVE_GENERIC_TOOLS[0],), **ray)
    np.testing.assert_array_equal(sim.candidate_entries_mm, [ray["selected_entry_mm"]])
    np.testing.assert_array_equal(sim.candidate_tips_mm, [ray["selected_target_mm"]])
    np.testing.assert_array_equal(sim.config.affine, np.eye(4))
    np.testing.assert_array_equal(sim.config.access.center_mm, access.center_mm)


@pytest.mark.parametrize("kwargs,reason", [
    ({"selected_entry_mm": (5, 6, 3.5)}, "supplied together"),
    ({"selected_target_mm": (5, 6, 13)}, "supplied together"),
    ({"selected_entry_mm": (5, 6, 3.5), "selected_target_mm": (5, 6, 13), "tools": None}, "explicit access"),
    ({"selected_entry_mm": (5, 6, 3.5), "selected_target_mm": (5, 6, 13), "access": None}, "explicit access"),
    ({"selected_entry_mm": (5, 6, 3.5), "selected_target_mm": (np.nan, 6, 13)}, "finite RAS"),
])
def test_incomplete_selected_route_fails_closed(kwargs, reason):
    case, access = case_and_access()
    options = {"access": access, "tools": (NATIVE_GENERIC_TOOLS[0],), **kwargs}
    with pytest.raises(ValueError, match=reason):
        make_native_patient_simulator(case, **options)
