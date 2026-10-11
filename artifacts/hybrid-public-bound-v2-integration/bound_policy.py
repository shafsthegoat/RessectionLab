"""Public target bound over conditional expansion; frozen helper unchanged.

The first eight certificates and their ordering are exactly the frozen policy.
No-positive shortlist still expands the full emitted catalog. A positive winner
that would switch tools additionally compares unqueried current-tool proposals.
This is not a tool-switch ban or a global optimality/reachability claim.
"""
import copy
import math
from dataclasses import asdict
import numpy as np

from hybrid_proposal import GeneratedSparseTask
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.post_exposure import preentry_tip
from resectionlab.public_reward_window import PublicTargetWindow as PublicTargetBoundIndex
from resectionlab.public_reward_bound import RewardUpperBound

VERSION = "nominal-top8-current-tool-public-window-v2"
POLICY = {
    "version": VERSION,
    "trigger": "no_strictly_positive_feasible_nominal_reward_in_initial_top8",
    "expansion": "all_remaining_emitted_raw_ids_once_in_original_order",
    "positive_switch_trigger": "best_positive_top8_action_changes_current_tool_and_current_tool_unqueried",
    "positive_switch_expansion": "remaining_current_tool_emitted_ids_once_in_original_order",
    "tool_switches": "allowed_if_still_best_after_unchanged_reward_comparison",
    "selection": "unchanged_nominal_reward_STOP_zero_original_raw_order_ties",
    "fit_updates": 0,
    "omitted_catalog_slots": "not_a_global_exhaustion_claim",
    "bound": "full_active_sweep_public_target_envelope; no_legality_claim",
    "positive_ties": "exact_preview_retained; original_raw_order",
}


def nominal_scores(task, inventory):
    """Canonical reward arithmetic with explicitly public nominal membership.

    A shallow scoring view replaces only the reference pointer. No engine clone,
    transition, observation, preview, private membership, or learning call occurs.
    """
    if task.case.nominal_target is None:
        raise ValueError("Explicit public nominal target required")
    view = copy.copy(task)
    view.case = copy.copy(task.case)
    object.__setattr__(view.case, 'reference_target', task.case.nominal_target)
    view._planning = True
    scores = []
    for identifier, result in inventory.items():
        task._check_cancelled()
        task.costs.count("fallback_nominal_reward_checks")
        record = NativeSpatialTask._score_record(view,
            {**result.to_history_record(), "action_id": identifier}, task._current_tool)
        value = float(record["reward"])
        if not math.isfinite(value):
            raise ValueError("Finite public nominal reward required")
        scores.append({"action_id": identifier, "nominal_reward": value})
    return scores


class PublicBoundFallbackMixin:
    """Opt-in behavior over the existing generated or admitted sparse task."""

    def _contract_record(self):
        record = super()._contract_record()
        if self._selection != "ridge_top8":
            raise ValueError("Fallback requires the frozen ridge_top8 selector")
        return {**record, "conditional_catalog_expansion": copy.deepcopy(POLICY)}

    def _prepare_inventory(self):
        if getattr(self, "_fallback_failure", None) is not None:
            raise RuntimeError("Failed fallback inventory cannot be resumed")
        if (self._inventory is not None and
                getattr(self, "_fallback_complete_state", None) == self._engine.state_hash):
            return super()._prepare_inventory()  # Retains the normal integrity check.
        callback = self.costs.inventory_callback
        self.costs.inventory_callback = None
        try:
            inventory = super()._prepare_inventory()
            if self._terminated:
                return inventory
            # The inherited timer covers the initial top8 work; this second
            # interval covers only the additional gate, expansion and logging.
            with self.costs.measure("inventory_inclusive"):
                initial = copy.deepcopy(self._shortlist_record)
                initial_ids = initial["selected_ids"]
                raw_ids = initial["raw_ids"]
                if len(set(raw_ids)) != len(raw_ids) or len(set(initial_ids)) != len(initial_ids):
                    raise ValueError("Unique emitted proposal IDs required")
                with self.costs.measure("fallback_nominal_gate"):
                    rewards = nominal_scores(self, inventory)
                best = max(rewards, key=lambda row: row["nominal_reward"], default=None)
                triggered = best is None or best["nominal_reward"] <= 0.
                queried = set(initial_ids)
                remaining_current = [ray.proposal_id for ray in self._proposal_batch.proposals
                    if ray.tool_id == self._current_tool and ray.proposal_id not in queried]
                switch_triggered = bool(not triggered and self._current_tool is not None
                    and inventory[best["action_id"]].tool_id != self._current_tool and remaining_current)
                self.costs.count("fallback_gate_states")
                self.costs.count("fallback_triggered_states", int(triggered))
                self.costs.count("current_tool_comparison_states", int(switch_triggered))
                additional, outcomes, bounds = [], {}, []
                incumbent = max(0., 0. if best is None else best["nominal_reward"])
                if triggered or switch_triggered:
                    with self.costs.measure("public_bound_setup"):
                        self.costs.count("public_bound_states")
                        cavity = (self._engine.removed_mask if self.case.post_exposure is None
                                  else self._engine.connected_free_mask)
                        public_index = PublicTargetBoundIndex.prepare(
                            affine=self.case._native_affine_ras_mm,
                            public_target=self.case.nominal_target,
                            target_domain=self.case.public_target_domain,
                            observed_cavity=cavity, cancelled=self._check_cancelled)
                        self.costs.count("public_bound_snapshot_bytes", public_index.snapshot_bytes)
                        self.costs.count("public_bound_snapshot_target_cells", public_index.target_cells)
                        self.costs.count("public_bound_snapshot_unknown_cells", public_index.unknown_cells)
                    tools = {tool.tool_id: tool for tool in self.case.tools}
                    for ray in self._proposal_batch.proposals:
                        if ray.proposal_id in queried or (not triggered and ray.tool_id != self._current_tool):
                            continue
                        self._check_cancelled()
                        with self.costs.measure("public_bound_queries"):
                            self.costs.count("public_bound_queries")
                            tool = tools[ray.tool_id]
                            delta = np.subtract(ray.tip_mm, ray.entry_mm)
                            length = float(np.linalg.norm(delta))
                            upper = RewardUpperBound(None, None, "unsupported_trajectory")
                            if math.isfinite(length) and length >= 1e-9:
                                axis = delta / length
                                physical_start = np.asarray(ray.entry_mm, float)
                                if (self.case.post_exposure is not None and
                                        float(axis @ self.case.access.normal_inward) > 0):
                                    physical_start = preentry_tip(tool, physical_start, axis,
                                                                  self.case.access.normal_inward)
                                upper = public_index.bound(physical_start_mm=physical_start,
                                    tip_mm=ray.tip_mm, axis_unit=axis,
                                    tip_length_mm=float(tool.tip_length_mm),
                                    tip_radius_mm=float(tool.tip_radius_mm), reward=asdict(self.reward_spec),
                                    reward_voxel_volume_mm3=float(self._config.voxel_volume_mm3),
                                    tool_change=self._current_tool is not None and self._current_tool != ray.tool_id,
                                    cancelled=self._check_cancelled)
                            pruned = upper.cannot_change_winner(incumbent)
                            bounds.append({"proposal_id": ray.proposal_id, **asdict(upper),
                                           "incumbent_reward": incumbent, "pruned": pruned})
                            self.costs.count("public_bound_pruned", int(pruned))
                            self.costs.count("public_bound_unavailable", int(upper.reward_upper is None))
                        if pruned:
                            continue  # Reward domination, NEVER a geometric rejection.
                        with self.costs.measure("exact_verification"):
                            self.costs.count("native_previews")
                            self.costs.count("fallback_additional_previews")
                            if switch_triggered:
                                self.costs.count("current_tool_additional_previews")
                            result = self._engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
                        queried.add(ray.proposal_id)
                        additional.append(ray.proposal_id)
                        outcomes[ray.proposal_id] = result
                        if result.feasible:
                            inventory[ray.proposal_id] = result
                            with self.costs.measure("fallback_nominal_gate"):
                                score = nominal_scores(self, {ray.proposal_id: result})[0]["nominal_reward"]
                            incumbent = max(incumbent, score)
                # Reward tie order is the original raw order, never rank order.
                inventory = {identifier: inventory[identifier] for identifier in raw_ids if identifier in inventory}
                ledger = copy.deepcopy(self._ledger)
                pruned_ids = {row["proposal_id"] for row in bounds if row["pruned"]}
                for row in ledger:
                    if row.get("proposal_id") in pruned_ids:
                        row["query_disposition"] = "BOUND_NOT_COMPETITIVE"
                    result = outcomes.get(row.get("proposal_id"))
                    if result is not None:
                        row.update(query_disposition="CERTIFIED_LEGAL" if result.feasible else "CERTIFIED_REJECTED",
                            feasible=bool(result.feasible), native_reason=result.reason)
                selected = initial_ids + additional
                self._shortlist_record = {**initial, "selected_ids": selected,
                    "queried_count": len(selected), "not_queried_count": len(raw_ids)-len(selected),
                    "legal_count": len(inventory), "exhaustive_over_emitted_catalog": len(selected)==len(raw_ids),
                    "fallback_policy": copy.deepcopy(POLICY), "initial_selected_ids": initial_ids,
                    "initial_nominal_scores": rewards, "fallback_triggered": triggered,
                    "current_tool": self._current_tool,
                    "initial_best_positive_action": None if triggered else best["action_id"],
                    "unqueried_current_tool_count_before_expansion": len(remaining_current),
                    "current_tool_comparison_triggered": switch_triggered,
                    "additional_selected_ids": additional, "fallback_completed": True,
                    "public_reward_bounds": bounds, "bound_pruned_count": len(pruned_ids),
                    "conditional_reward_comparison_complete": True}
                self._check_cancelled()
                self._inventory, self._ledger = inventory, ledger
                self._fallback_complete_state = self._engine.state_hash
                if callback is not None:
                    with self.costs.measure("inventory_logging"):
                        keys = ("action_id", "proposal_id", "tool_id", "column_index", "family", "voxel",
                                "reason", "query_disposition", "feasible", "native_reason")
                        callback({"step": self._steps, "source_state_hash": self._engine.state_hash,
                            "shortlist": copy.deepcopy(self._shortlist_record),
                            "ledger": [{k:r[k] for k in keys if k in r} for r in self._ledger]})
                return inventory
        except BaseException as error:
            # Paid counters survive, but no partial catalog can become a STOP.
            self._fallback_failure = type(error).__name__ + ": " + str(error)
            self._inventory = None
            raise
        finally:
            self.costs.inventory_callback = callback

    def _qualify_accounting(self, accounting):
        value = super()._qualify_accounting(accounting)
        value.update(method=VERSION, verification_scope="top8_then_no_positive_full_or_positive_switch_current_tool_comparison",
            fallback_policy=copy.deepcopy(POLICY),
            STOP_meaning="no_positive_verified_or_potentially_positive_conditionally_requested_action; no_global_reachability_claim")
        return value


class GeneratedBoundTask(PublicBoundFallbackMixin, GeneratedSparseTask):
    """Generated fixture entry; acquired sources still refused by the base."""


# Acquired construction must separately compose this mixin with an already
# admitted source task. This module provides no new acquired-source entry.
