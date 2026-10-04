"""Early route-to-native action diagnostics; these functions never train a policy.

A static instrument route may intersect ordinary tissue conditionally on access.
That does not establish that its tip can cut a connected channel for its shaft.
Retain that distinction and identify when a selected route was not actually
included among a refinement factory's endpoint/entry proposals.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def initial_native_action_diagnostic(simulator: Any) -> dict[str, Any]:
    """Inspect already generated, checked proposals without taking transitions."""
    actions = simulator.proposed_actions()
    non_stop = [action for action in actions if action.action_id != "STOP"]
    failures = list(simulator._proposal_failures)
    tools = simulator.native_config.tools
    return {
        "status": "native_actions_available" if non_stop else "no_legal_native_actions",
        "policy_has_nontrivial_action_choice": bool(non_stop),
        "non_stop_action_count": len(non_stop),
        "non_stop_action_ids": [action.action_id for action in non_stop],
        "rejected_proposal_count": len(failures),
        "rejection_counts": dict(Counter(row["reason"] for row in failures)),
        "rejected_proposals": failures,
        "candidate_tips_mm": simulator.candidate_tips_mm.tolist(),
        "candidate_entries_mm": simulator.candidate_entries_mm.tolist(),
        "decision_model_hash": simulator.decision_model_hash,
        "source_hash": simulator.native_config.source_hash,
        "tool_dimensions": [{"tool_id": tool.tool_id, "tip_radius_mm": tool.tip_radius_mm,
                             "shaft_radius_mm": tool.shaft_radius_mm,
                             "tip_length_mm": tool.tip_length_mm,
                             "shaft_wider_than_active_tip": tool.shaft_radius_mm > tool.tip_radius_mm}
                            for tool in tools],
        "interpretation": "Initial native action availability only; no learning, removal, or clinical benefit is asserted.",
        "clinical_deficit_probability": None,
    }


def route_proposal_coverage(route: Any, simulator: Any, *, tolerance_mm: float = 1e-6) -> dict[str, Any]:
    """Check that a route-conditioned factory actually proposes that exact ray."""
    target = np.asarray(route.target_mm, float)
    entry = np.asarray(route.entry_mm, float)
    tips = np.asarray(simulator.candidate_tips_mm)
    entries = np.asarray(simulator.candidate_entries_mm)
    target_matches = np.linalg.norm(tips - target, axis=1) <= tolerance_mm
    entry_matches = np.linalg.norm(entries - entry, axis=1) <= tolerance_mm
    exact = target_matches & entry_matches
    problems = []
    if not target_matches.any():
        problems.append("selected_route_target_not_proposed")
    if not exact.any():
        problems.append("selected_route_entry_target_pair_not_proposed")
    window = simulator.native_config.access
    offsets = entries - np.asarray(window.center_mm)
    radial = offsets - (offsets @ window.normal_inward)[:, None] * window.normal_inward
    return {
        "route_id": route.route_id,
        "selected_route_target_proposed": bool(target_matches.any()),
        "selected_route_exact_ray_proposed": bool(exact.any()),
        "matching_indices": np.flatnonzero(exact).tolist(),
        "proposal_problems": problems,
        "projected_entry_offsets_mm": np.linalg.norm(radial, axis=1).tolist(),
        "access_radius_mm": float(window.radius_mm),
        "interpretation": "Proposal coverage differs from native feasibility; a present route can still be rejected by complete-tool checks.",
    }
