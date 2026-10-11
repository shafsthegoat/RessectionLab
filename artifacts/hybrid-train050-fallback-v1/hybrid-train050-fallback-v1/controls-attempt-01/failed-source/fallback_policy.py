"""Versioned no-positive-shortlist expansion; frozen ranker/helper stay unchanged.

The first eight certificates and their ordering are exactly the frozen policy.
Only when none has strictly positive permitted nominal immediate reward do we
certify the remaining emitted catalog. This is not a global reachability claim,
and does not repair a missed better action when a shortlist action is positive.
"""
import copy
import math

from hybrid_proposal import GeneratedSparseTask
from resectionlab.native_spatial_task import NativeSpatialTask

VERSION = "nominal-no-positive-top8-exhaustive-fallback-v1"
POLICY = {
    "version": VERSION,
    "trigger": "no_strictly_positive_feasible_nominal_reward_in_initial_top8",
    "expansion": "all_remaining_emitted_raw_ids_once_in_original_order",
    "selection": "unchanged_nominal_reward_STOP_zero_original_raw_order_ties",
    "fit_updates": 0,
    "omitted_catalog_slots": "not_a_global_exhaustion_claim",
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
    view.case.reference_target = task.case.nominal_target
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


class NoPositiveFallbackMixin:
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
                triggered = not any(row["nominal_reward"] > 0. for row in rewards)
                self.costs.count("fallback_gate_states")
                self.costs.count("fallback_triggered_states", int(triggered))
                additional, outcomes = [], {}
                if triggered:
                    queried = set(initial_ids)
                    for ray in self._proposal_batch.proposals:
                        if ray.proposal_id in queried:
                            continue
                        self._check_cancelled()
                        with self.costs.measure("exact_verification"):
                            self.costs.count("native_previews")
                            self.costs.count("fallback_additional_previews")
                            result = self._engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
                        queried.add(ray.proposal_id)
                        additional.append(ray.proposal_id)
                        outcomes[ray.proposal_id] = result
                        if result.feasible:
                            inventory[ray.proposal_id] = result
                # Reward tie order is the original raw order, never rank order.
                inventory = {identifier: inventory[identifier] for identifier in raw_ids if identifier in inventory}
                ledger = copy.deepcopy(self._ledger)
                for row in ledger:
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
                    "additional_selected_ids": additional, "fallback_completed": True}
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
        value.update(method=VERSION, verification_scope="top8_then_conditional_full_emitted_catalog",
            fallback_policy=copy.deepcopy(POLICY),
            STOP_meaning="no_positive_verified_action_after_conditional_expansion; no_global_reachability_claim")
        return value


class GeneratedFallbackTask(NoPositiveFallbackMixin, GeneratedSparseTask):
    """Generated fixture entry; acquired sources still refused by the base."""


def admitted_task_class():
    # Keep the generated entry importable without any acquired-source contract.
    from patient_transfer import AdmittedTransferTask

    class AdmittedFallbackTask(NoPositiveFallbackMixin, AdmittedTransferTask):
        pass

    return AdmittedFallbackTask
