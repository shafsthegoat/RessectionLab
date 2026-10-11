"""Explicit bounded page verification over unchanged native transitions.

This task deliberately has no whole-catalog policy observation. Search iterates
pages; authoritative execution locates an action in the same declared catalog
and verifies its page before the ordinary native commit. The base catalog and
its sidecars are prepared once per state. Each page clones that clean base,
retains <=120 base certificates and verifies <=8 extras. No page contributes
its rejected-ray evidence to another page, and no reference target is queried.
"""
from dataclasses import asdict
from types import MappingProxyType
import copy

import numpy as np

from .core import freeze_json, thaw_json
from .native_proposals import OBSTRUCTION_CELLS_PER_PREVIEW
from .native_spatial_task import NativeSpatialTask

PAGED_EXPANSION = "native-obstruction-pages-v1"


class PagedObstructionTask(NativeSpatialTask):
    def __init__(self, case, **kwargs):
        if (case.proposal_config is None or case.proposal_config.obstruction_page_size != 8
                or not 1 <= len(case.tools) <= 2 or case.proposal_mode != "nominal_cavity_v1"):
            raise ValueError("Explicit eight-pair obstruction pages and at most two tools required")
        self._clear_pages()
        self._resetting_pages = False
        super().__init__(case, **kwargs)

    def _contract_record(self):
        return {**super()._contract_record(), "action_verification": PAGED_EXPANSION,
                "whole_catalog_actor_observation": False}

    def _clear_pages(self):
        self._base_inventory = None
        self._base_outcomes = None
        self._page_catalog = None
        self._base_batch = None
        self._page_state_hash = None
        self._active_page = None
        self._page_actions = ()
        self._page_rows = None

    def reset(self, seed=0):
        self._resetting_pages = True
        try:
            super().reset(seed)
        finally:
            self._resetting_pages = False
            self._clear_pages()
        # No partial base observation is passed off as the full action set.
        return None

    def observation(self):
        if self._resetting_pages:
            return None
        if self._active_page is None:
            raise ValueError("A page observation requires explicit page selection")
        return super().observation()

    def _prepare_inventory(self):
        self._assert_frozen()
        self._check_cancelled()
        if self._active_page is None or self._inventory is None:
            raise ValueError("No certified active page; iterate pages or execute an explicit catalog action")
        if self._page_state_hash != self._engine.state_hash:
            raise ValueError("Stale page state")
        return self._inventory

    def _prepare_base(self):
        self._assert_frozen()
        self._check_cancelled()
        if self._base_inventory is not None:
            if self._page_state_hash != self._engine.state_hash:
                raise ValueError("Stale base certificate state")
            return
        proposer = self.case._nominal_proposer
        base = proposer.propose(self._engine, cancelled=self._cancelled)
        if len(base.proposals) > 120:
            raise ValueError("Base proposal ceiling exceeded")
        inventory, outcomes, evidence = {}, {}, []
        origin = np.asarray(self.case._crop_origin)
        end = origin + self.case._crop_shape
        for ray in base.proposals:
            self._check_cancelled()
            result = self._engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm,
                obstruction_diagnostics=True, obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
            if (result.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE") != (result.obstruction_diagnostic is not None):
                raise RuntimeError("Missing or misattributed base first-obstruction sidecar")
            evidence.append(result.obstruction_diagnostic)
            outcomes[ray.proposal_id] = self._outcome(ray, result, origin, end)
            if result.feasible:
                inventory[ray.proposal_id] = result
        full = proposer.append_obstruction_openings(base, self._engine, evidence, cancelled=self._cancelled)
        if len(full.proposals)-len(base.proposals) > 32:
            raise ValueError("Bounded obstruction pair ceiling exceeded")
        if len(self._engine._preview_records) != len(inventory):
            raise ValueError("Base certificates include unaccounted external previews")
        self._base_inventory = MappingProxyType(inventory)
        self._base_outcomes = freeze_json(outcomes)
        self._base_batch, self._page_catalog = base, full
        self._page_state_hash = self._engine.state_hash

    @staticmethod
    def _outcome(ray, result, origin, end):
        return {"entry_mm": list(ray.entry_mm), "tip_mm": list(ray.tip_mm),
            "feasible": bool(result.feasible), "reason": result.reason,
            "endpoint_center_in_actor_crop": bool(np.all(np.asarray(ray.voxel) >= origin)
                                                  and np.all(np.asarray(ray.voxel) < end))}

    def _activate(self, page):
        self._prepare_base()
        if self._active_page is not None:
            raise ValueError("Page clones may be activated exactly once")
        extra = self._page_catalog.proposals[len(self._base_batch.proposals):]
        total = (len(extra)+7)//8
        if type(page) is not int or not 0 <= page <= total:
            raise ValueError("Page outside the declared catalog")
        inventory = dict(self._base_inventory)
        outcomes = thaw_json(self._base_outcomes)
        selected = () if page == 0 else extra[(page-1)*8:page*8]
        origin = np.asarray(self.case._crop_origin)
        end = origin + self.case._crop_shape
        for ray in selected:
            self._check_cancelled()
            result = self._engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
            outcomes[ray.proposal_id] = self._outcome(ray, result, origin, end)
            if result.feasible:
                inventory[ray.proposal_id] = result
        if len(self._engine._preview_records) > 128:
            raise RuntimeError("Native certificate lifetime bound exceeded")
        for preview in inventory.values():
            if self._engine._preview_records.get(id(preview)) is not preview:
                raise RuntimeError("Page evicted a still-live native certificate")
        self._inventory = inventory
        self._proposal_batch = self._page_catalog
        self._active_page = page
        self._page_actions = (tuple(self._base_inventory) if page == 0 else
                              tuple(r.proposal_id for r in selected if r.proposal_id in inventory))
        rows = []
        for slot in self._page_catalog.ledger:
            row = {**asdict(slot), "offset_source_voxels": list(slot.offset_source_voxels),
                "voxel": None if slot.voxel is None else list(slot.voxel),
                "action_id": slot.proposal_id, "proposal_reason": slot.reason,
                "feasible": None, "verification": "NOT_QUERIED_IN_THIS_PAGE"}
            if slot.reason == "PROPOSED_UNCERTIFIED" and slot.proposal_id in outcomes:
                row.update(outcomes[slot.proposal_id], verification="NATIVE_VERIFIED")
            elif slot.reason != "PROPOSED_UNCERTIFIED":
                row.update(feasible=False, verification="PROVIDER_DISPOSITION")
            rows.append(row)
        self._ledger = tuple(rows)
        self._page_rows = freeze_json({"schema": PAGED_EXPANSION, "page_index": page,
            "additional_page_count": total, "base_emitted_count": len(self._base_batch.proposals),
            "additional_emitted_count": len(extra), "additional_verified_in_page": len(selected),
            "source_hash": self._source_hash, "decision_model_hash": self.decision_model_hash,
            "source_state_hash": self._page_state_hash, "steps_taken": self._steps,
            "page_action_ids": self._page_actions, "ledger": rows,
            "obstruction_accounting": thaw_json(self._page_catalog.obstruction_accounting),
            "all_bounded_pairs_considered": page == 0 and not extra,
            "discovery_complete": False, "full_catalog_actor_mask": False})

    def planning_action_pages(self):
        """Yield (certified task page, legal actions to expand) in stable order.

        The consumer must exhaust all pages before marking an expanded node
        complete. Page children are normal native tasks with ordinary history.
        Every page starts from the unmodified base engine, not the prior page.
        """
        if not self._planning or self._active_page is not None:
            raise ValueError("Page enumeration requires an unselected nominal planning state")
        self._prepare_base()
        extra = len(self._page_catalog.proposals)-len(self._base_batch.proposals)
        for index in range(1+(extra+7)//8):
            self._check_cancelled()
            page = self.clone()
            page._activate(index)
            yield page, page._page_actions

    def planning_page_record(self):
        if self._active_page is None:
            raise ValueError("No active verification page")
        row = self._page_rows
        return {key: row[key] for key in ("page_index", "additional_page_count",
            "base_emitted_count", "additional_emitted_count", "additional_verified_in_page",
            "source_hash", "decision_model_hash", "source_state_hash", "steps_taken")}

    def candidate_inventory(self):
        if self._active_page is None:
            raise ValueError("Page-specific inventory requires explicit page selection")
        self._prepare_inventory()
        return thaw_json(self._page_rows)

    def _activate_for_action(self, action):
        if not isinstance(action, str):
            raise ValueError("Paged execution requires a stable catalog action ID")
        if self._active_page is not None:
            if action != "STOP" and action not in self._inventory:
                raise ValueError("Action is not certified in the selected page")
            return
        self._prepare_base()
        if action == "STOP" or action in self._base_inventory:
            self._activate(0)
            return
        extra = self._page_catalog.proposals[len(self._base_batch.proposals):]
        matches = [index for index, ray in enumerate(extra) if ray.proposal_id == action]
        if len(matches) != 1:
            raise ValueError("Action not in the current public obstruction catalog")
        self._activate(1+matches[0]//8)
        if action not in self._inventory:
            raise ValueError("Catalog action failed unchanged native certification")

    def _execute(self, action, *, planning):
        if self.terminated:
            raise ValueError("Paged native episode has terminated")
        if planning and not self._planning:
            raise ValueError("LAZY_PLANNING_ONLY: nominal model required")
        self._activate_for_action(action)
        before = self._steps
        try:
            # Successor observations are deferred here too. This does not
            # relax authoritative scoring or the unchanged native commit.
            return super()._transition(action, observe_successor=False)
        finally:
            if self._steps != before:
                self._clear_pages()

    def step(self, action):
        return self._execute(action, planning=False)

    def advance_planning(self, action):
        return self._execute(action, planning=True)

    def clone(self):
        # Base certificates/outcomes and batch are immutable; super clones
        # only native committed state/history and the active inventory map.
        return super().clone()

    def observed_greedy_search(self, **kwargs):
        raise ValueError("Use explicit paged beam expansion; ordinary greedy assumes one full inventory")

    observed_one_step_search = observed_greedy_search
