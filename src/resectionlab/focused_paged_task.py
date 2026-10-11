"""Explicit state-local goal focus and replayable query/action packets.

Focus affects only bounded public candidate allocation. Native world, proposal
configuration, reward, certificate ownership and history are unchanged. The
query context is sealed separately and may be chosen only before certification.
"""
from dataclasses import asdict
from .core import freeze_json, thaw_json, semantic_digest
from .native_proposals import TARGET_RAY_OBSTRUCTION_RULE
from .paged_obstruction import PagedObstructionTask

QUERY_SCHEMA = "focused-public-goal-catalog-v1"
ACTION_SCHEMA = "focused-public-goal-action-v1"


class FocusedPagedObstructionTask(PagedObstructionTask):
    def __init__(self, case, **kwargs):
        if case.proposal_config is None or case.proposal_config.obstruction_selection_rule != TARGET_RAY_OBSTRUCTION_RULE:
            raise ValueError("Focused queries require unchanged v3 proposal configuration")
        super().__init__(case, **kwargs)

    def _clear_pages(self):
        super()._clear_pages()
        self._goal_focus = None
        self._goal_focus_seal = semantic_digest(None)
        self._query_catalog_digest = None

    def _assert_frozen(self):
        super()._assert_frozen()
        if semantic_digest(self._goal_focus) != self._goal_focus_seal:
            raise RuntimeError("Goal query changed outside explicit focus selection")
        if self._goal_focus is not None and self._goal_focus["source_state_hash"] != self._engine.state_hash:
            raise RuntimeError("Goal query is stale for committed native state")

    def focus_record(self, physical_key):
        self._assert_frozen(); self._check_cancelled()
        proposer = self.case._nominal_proposer
        base = proposer.propose(self._engine, cancelled=self._cancelled)
        return proposer.goal_focus_record(base, self._engine, physical_key)

    def set_focus(self, focus):
        self._assert_frozen(); self._check_cancelled()
        if self.terminated or self._base_inventory is not None or self._active_page is not None:
            raise ValueError("Focus must be selected before any state certification")
        if focus is not None:
            proposer = self.case._nominal_proposer
            base = proposer.propose(self._engine, cancelled=self._cancelled)
            focus = proposer.validate_goal_focus(base, self._engine, focus)
        self._goal_focus = focus
        self._goal_focus_seal = semantic_digest(focus)

    def _extend_base_catalog(self, base, evidence):
        return self.case._nominal_proposer.append_obstruction_openings(
            base, self._engine, evidence, cancelled=self._cancelled, focus=self._goal_focus)

    def _prepare_base(self):
        super()._prepare_base()
        if self._query_catalog_digest is None:
            full = self._page_catalog
            self._query_catalog_digest = semantic_digest({"schema": QUERY_SCHEMA,
                "source_state_hash": self._engine.state_hash, "focus": self._goal_focus,
                "catalog": [asdict(ray) for ray in full.proposals],
                "ledger": [asdict(slot) for slot in full.ledger],
                "accounting": full.obstruction_accounting})

    def export_action(self, action):
        """Seal an action already owned by this certified page, including STOP."""
        self._prepare_inventory()
        if self.terminated or not isinstance(action, str) or (action != "STOP" and action not in self._inventory):
            raise ValueError("Current certified page action required")
        physical = None
        if action != "STOP":
            record = self._inventory[action].to_history_record()
            physical = {key: record[key] for key in ("tool_id", "entry_mm", "tip_mm")}
        record = {"schema": ACTION_SCHEMA, "source_hash": self._source_hash,
            "decision_model_hash": self.decision_model_hash,
            "source_state_hash": self._engine.state_hash, "steps_taken": self._steps,
            "current_tool": self._current_tool, "focus": thaw_json(self._goal_focus),
            "query_catalog_digest": self._query_catalog_digest,
            "action_id": action, "physical_action": physical}
        return freeze_json({**record, "seal": semantic_digest(record)})

    def _execute_record(self, packet, *, planning):
        self._assert_frozen()
        if self.terminated or (planning and not self._planning):
            raise ValueError("Live task and nominal context for planning required")
        try:
            record = thaw_json(packet)
            unsigned = {key: value for key, value in record.items() if key != "seal"}
            if (semantic_digest(unsigned) != record["seal"] or record["schema"] != ACTION_SCHEMA
                    or record["source_hash"] != self._source_hash
                    or record["decision_model_hash"] != self.decision_model_hash
                    or record["source_state_hash"] != self._engine.state_hash
                    or record["steps_taken"] != self._steps or record["current_tool"] != self._current_tool):
                raise ValueError("Stale or altered focused action packet")
            if self._base_inventory is None:
                self.set_focus(record["focus"])
            elif thaw_json(self._goal_focus) != record["focus"]:
                raise ValueError("Packet belongs to a different certified goal query")
            self._activate_for_action(record["action_id"])
            if thaw_json(self.export_action(record["action_id"])) != record:
                raise ValueError("Packet query/catalog or physical action differs")
        except (TypeError, KeyError, IndexError) as error:
            raise ValueError("Malformed focused action packet") from error
        return super()._execute(record["action_id"], planning=planning)

    def step_record(self, packet):
        return self._execute_record(packet, planning=False)

    def advance_record(self, packet):
        return self._execute_record(packet, planning=True)

    def step(self, action):
        raise ValueError("Focused execution requires a sealed query/action packet")

    def advance_planning(self, action):
        raise ValueError("Focused planning requires a sealed query/action packet")
