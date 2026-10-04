"""Experimental RAW-only dynamic inventory over the unchanged native engine.

This module is deliberately outside fixed-route refinement and procedural
transfer. It shares the native feature/reward implementation, not its fixed
candidate constructor or initial-geometry cache. Each instance belongs to one
worker; search branches use clone(), and independent runs use fresh().
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import time
from typing import Any, Callable

import numpy as np

from .native_proposals import AxisColumnProposalConfig, PreparedAxisColumnProposer
from .native_resection import NativeResectionConfig, NativeResectionEngine
from .native_simulation import (NATIVE_ACTION_FEATURE_NAMES, NativeSequentialSimulator,
                                _immutable_model_descriptor)
from .simulation import (InvalidActionError, MacroAction, RewardSpec, SequentialSimulator,
                         SimulationConfig, SimulationObservation, StepResult, _readonly)
from .worlds import WorldGeneratorConfig

AXIS_ADAPTER_VERSION = "experimental-native-axis-column-policy-v1"
NATIVE_CERTIFICATE_CAPACITY = 128


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CommittedTransitionInterrupted(InterruptedError):
    """Cancellation interrupted NEXT inventory preparation after a real commit.

    The caller must preserve the executed transition in its receipt. This is
    never represented as rollback, successful STOP, or a complete observation.
    Generic learners currently fail the run on this exception; resumable
    handling requires a separately reviewed experimental runner.
    """

    def __init__(self, info: dict[str, Any], reward: float):
        super().__init__("Native axis transition committed; next inventory preparation interrupted")
        self.committed = True
        self.info = copy.deepcopy(info)
        self.reward = float(reward)


class AxisColumnNativeSimulator(NativeSequentialSimulator):
    """One ordered, certified dynamic inventory shared by SEARCH and RAW RL."""

    def __init__(self, native_config: NativeResectionConfig, *,
                 proposal_config: AxisColumnProposalConfig | None = None,
                 nominal_motor: np.ndarray | None = None,
                 nominal_language: np.ndarray | None = None,
                 reward: RewardSpec = RewardSpec(),
                 world_generator: WorldGeneratorConfig | None = None,
                 max_steps: int = 8, partial_contact_weight: float = .05,
                 compartment_names: dict[int, str] | None = None,
                 cancelled: Callable[[], bool] | None = None):
        initialization_started = time.perf_counter()
        self._cancelled = cancelled
        self._check_cancelled()
        if not np.isfinite(partial_contact_weight) or partial_contact_weight < 0:
            raise ValueError("Partial-contact surrogate weight must be finite and nonnegative")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("Axis episode horizon must be a positive integer")
        self.proposal_config = proposal_config or AxisColumnProposalConfig()
        if not isinstance(self.proposal_config, AxisColumnProposalConfig):
            raise TypeError("An AxisColumnProposalConfig is required")
        if self.proposal_config.max_primary_rays > NATIVE_CERTIFICATE_CAPACITY:
            raise ValueError("Primary rays exceed the native 128-certificate capacity")
        self.native_config = native_config
        self.partial_contact_weight = float(partial_contact_weight)
        started = time.perf_counter()
        self._proposer = PreparedAxisColumnProposer(native_config, self.proposal_config)
        self._preparer_seconds = time.perf_counter() - started
        self._check_cancelled()
        started = time.perf_counter()
        self.engine = NativeResectionEngine(native_config)
        self._engine_setup_seconds = time.perf_counter() - started
        self._check_cancelled()
        self._native_config_hash = native_config.fingerprint
        self._native_inverse = _readonly(np.linalg.inv(native_config.affine))
        self._episode_seal: str | None = None
        self._episode_ready = False
        self._hidden_descriptor = None
        self._derived_descriptor = None
        self._observation_inventory = None
        self._reading_observation = False
        self._batch = None
        self._previews: dict[str, Any] = {}
        self._proposal_failures: list[dict[str, Any]] = []
        self._inventory_descriptor = None
        self._action_provenance: dict[str, Any] = {}
        self._inventory_receipts: list[dict[str, Any]] = []
        self._integrity_calls = 0
        self._integrity_seconds = 0.
        self._preview_calls = 0
        self._preview_seconds = 0.
        config = SimulationConfig(
            native_config.tissue_mask, native_config.target_labels, native_config.affine,
            native_config.access, native_config.tools, nominal_motor=nominal_motor,
            nominal_language=nominal_language, hard_exclusion=native_config.hard_exclusion,
            reward=reward, world_generator=world_generator, max_steps=max_steps,
            max_actions=self.proposal_config.max_primary_rays + 1,
            case_id=native_config.case_id, source_hash=native_config.source_hash,
            evidence_available=(nominal_motor is not None, nominal_language is not None),
            compartment_names=compartment_names or {1: "radiological_target"},
            derivation={"track": AXIS_ADAPTER_VERSION,
                        "candidate_sampling": "current_residual_axis_columns_shared_by_all_methods",
                        "proposal_model_hash": self._proposer.model_hash,
                        "tissue_support_provenance": native_config.tissue_support_provenance,
                        "removal_primitive": "full_affine_cells_contained_by_continuous_active_brush",
                        "partial_contact_policy": "retained_tissue_exposure_not_removed",
                        "partial_contact_weight": self.partial_contact_weight})
        self.config = config
        self._native_decision_hash = _digest({
            "adapter": AXIS_ADAPTER_VERSION, "native_config": self._native_config_hash,
            "simulation_config": config.decision_model_hash,
            "proposal_model": self._proposer.model_hash,
            "features": NATIVE_ACTION_FEATURE_NAMES, "input_profile": "RAW",
            "fallback": "only_after_primary_preview_rejection",
            "ordering": "provider_column_tool_order; STOP_first",
            "max_preview_attempts_per_inventory": 2 * self.proposal_config.max_primary_rays,
            "certificate_capacity": NATIVE_CERTIFICATE_CAPACITY,
        })
        self._model_descriptor = self._current_model_descriptor()
        # Deliberately bypass the fixed-point constructor and its virtual-reset cache.
        SequentialSimulator.__init__(self, config)
        self._target_fraction = _readonly(self._target_fraction)
        self._normal_fraction = _readonly(self._normal_fraction)
        self._spacing = _readonly(self._spacing)
        self._derived_descriptor = self._current_derived_descriptor()
        self._initialization_seconds = time.perf_counter() - initialization_started

    def _current_model_descriptor(self):
        return _immutable_model_descriptor((self.config, self.native_config,
            self.proposal_config, self.partial_contact_weight, self._native_inverse,
            self._proposer.model_hash, self._proposer.rule_hash,
            self._native_decision_hash, self._native_config_hash))

    def _current_derived_descriptor(self):
        return _immutable_model_descriptor((self._target_fraction, self._normal_fraction,
                                            self._spacing, self._total_target, self.voxel_volume_mm3))

    def assert_model_frozen(self) -> None:
        if self._current_model_descriptor() != self._model_descriptor:
            raise RuntimeError("Axis proposal decision model changed during optimization")
        if self._derived_descriptor is not None and self._current_derived_descriptor() != self._derived_descriptor:
            raise RuntimeError("Frozen derived native volume/reward model changed")

    def _episode_signature(self) -> str:
        record = {"actions": self._actions, "history": self._history,
                  "current_tool": self._current_tool, "reward": self.total_reward,
                  "terminated": self.terminated, "reason": self.termination_reason,
                  "seed": self.seed, "world_hash": self._hidden_world_hash,
                  "severed": sorted(self._severed_edges),
                  "nominal_severed": sorted(self._nominal_severed_edges)}
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode())
        partial = self._partial_contact_mask
        digest.update(repr((partial.shape, partial.dtype.str)).encode())
        digest.update(partial.tobytes())
        return digest.hexdigest()

    def _assert_episode(self) -> None:
        if not self._episode_ready and not getattr(self, "_resetting", False):
            raise RuntimeError("Axis episode reset is incomplete; call reset() successfully before use")
        if self._episode_seal is not None and self._episode_signature() != self._episode_seal:
            raise RuntimeError("Axis adapter history or partial-contact accounting changed")
        if self._hidden_descriptor is not None and self._current_hidden_descriptor() != self._hidden_descriptor:
            raise RuntimeError("Frozen episode world changed")
        if self.remaining_mask is not self.engine.remaining_mask or self.removed_mask is not self.engine.removed_mask or self.exposed_mask is not self.engine.contact_mask:
            raise RuntimeError("Axis adapter masks no longer bind the native engine")

    def _current_hidden_descriptor(self):
        return _immutable_model_descriptor((self._hidden_motor, self._hidden_language,
                                            self._hidden_known_coverage, self._hidden_graph))

    def _verified_batch(self):
        self.assert_model_frozen()
        self._assert_episode()
        started = time.perf_counter()
        try:
            return self._proposer.propose(self.engine)
        finally:
            self._integrity_calls += 1
            self._integrity_seconds += time.perf_counter() - started

    def reset(self, seed: int = 0) -> SimulationObservation:
        self._check_cancelled()
        self.assert_model_frozen()
        if self._episode_ready:
            self._verified_batch()
        self._episode_ready = False
        self.engine.reset()
        self._episode_seal = None
        self._hidden_descriptor = None
        self._partial_contact_mask = np.zeros(self.config.tissue_mask.shape, bool)
        self._batch = None
        self._previews = {}
        self._proposal_failures = []
        self._action_provenance = {}
        self._inventory_descriptor = None
        self._inventory_receipts = []
        self._integrity_calls = self._preview_calls = 0
        self._integrity_seconds = self._preview_seconds = 0.
        # Parent reset prepares fixed-world state; bind its temporary copies before
        # its virtual observation reads the native cavity.
        self._resetting = True
        try:
            SequentialSimulator.reset(self, seed)
            self._bind_engine_masks()
            self._hidden_motor = _readonly(self._hidden_motor)
            self._hidden_language = _readonly(self._hidden_language)
            self._hidden_known_coverage = _readonly(self._hidden_known_coverage)
            self._hidden_graph = {key: _readonly(value) for key, value in self._hidden_graph.items()}
            self._hidden_world_hash = _digest((self._hidden_world_hash, self.decision_model_hash))
            self._hidden_descriptor = self._current_hidden_descriptor()
            self._episode_seal = self._episode_signature()
            observation = self.observation()
            self._episode_ready = True
            return observation
        finally:
            self._resetting = False

    def _cached_inventory_descriptor(self):
        return _immutable_model_descriptor((self._proposals, tuple(self._previews.items()), self._action_provenance))

    def proposed_actions(self) -> tuple[MacroAction, ...]:
        if self._observation_inventory is not None:
            if not self._reading_observation:
                raise RuntimeError("Transient observation inventory used outside a guarded read")
            return self._observation_inventory
        self._check_cancelled()
        if getattr(self, "_resetting", False):
            self._bind_engine_masks()
        batch = self._verified_batch()
        self._check_cancelled()
        if self._proposals is not None and batch == self._batch:
            if self._cached_inventory_descriptor() != self._inventory_descriptor:
                raise RuntimeError("Certified axis action inventory changed")
            return self._proposals
        empty = _readonly(np.empty((0, 3)), np.int64)
        actions = [MacroAction("STOP", None, None, None, None, empty, empty)]
        previews, provenance, failures = {}, {}, []
        receipt = {"batch": batch.to_dict(), "attempts": [], "status": "building",
                   "certified_action_ids": [], "terminated": self.terminated}
        try:
            if not self.terminated:
                for proposal in batch.proposals:
                    endpoints = [("primary", proposal.primary_target_mm)]
                    if proposal.fallback_target_mm is not None:
                        endpoints.append(("fallback", proposal.fallback_target_mm))
                    for phase, tip in endpoints:
                        self._check_cancelled()
                        attempt = {"proposal_id": proposal.proposal_id, "phase": phase,
                                   "tool_id": proposal.tool_id, "tip_mm": tip,
                                   "entry_mm": proposal.entry_mm, "feasible": None,
                                   "reason": None, "status": "started"}
                        receipt["attempts"].append(attempt)
                        started = time.perf_counter()
                        try:
                            preview = self.engine.preview_stroke(proposal.tool_id, tip, entry_mm=proposal.entry_mm)
                        except Exception as error:
                            attempt.update(status="exception", reason=type(error).__name__)
                            raise
                        finally:
                            self._preview_calls += 1
                            attempt["elapsed_seconds"] = time.perf_counter() - started
                            self._preview_seconds += attempt["elapsed_seconds"]
                        attempt.update(status="complete", feasible=preview.feasible, reason=preview.reason)
                        self._check_cancelled()
                        if not preview.feasible:
                            failures.append(attempt)
                            continue
                        if not len(preview.removed_indices_native):
                            raise RuntimeError("Feasible native preview has no contained-cell removal")
                        if preview.source_state_hash != batch.cavity_state_hash or preview.decision_model_hash != self._native_config_hash:
                            raise RuntimeError("Native preview does not bind the current proposal cavity")
                        action_id = "AXISv1:" + _digest((self.decision_model_hash, proposal.proposal_id, phase)).split(":")[1][:24]
                        if action_id in previews:
                            raise RuntimeError("Axis action identifier collision")
                        target = tuple(np.rint(self._native_inverse[:3, :3] @ tip + self._native_inverse[:3, 3]).astype(int))
                        actions.append(MacroAction(action_id, proposal.tool_id, target, tuple(tip),
                            tuple(preview.axis_unit), preview.removed_indices_native,
                            preview.contact_indices_native, float(np.linalg.norm(np.asarray(tip) - proposal.entry_mm))))
                        previews[action_id] = preview
                        provenance[action_id] = {**{key: value for key, value in attempt.items() if key != "elapsed_seconds"}, "cavity_state_hash": batch.cavity_state_hash,
                                                 "proposal_model_hash": batch.proposal_model_hash}
                        break  # A feasible primary never yields a fallback, regardless of value.
            self._check_cancelled()
        except Exception as error:
            receipt["status"] = "cancelled" if isinstance(error, InterruptedError) else "failed"
            receipt["error_type"] = type(error).__name__
            self._inventory_receipts.append(copy.deepcopy(receipt))
            raise
        receipt["status"] = "complete"
        receipt["certified_action_ids"] = [action.action_id for action in actions[1:]]
        self._proposals, self._previews = tuple(actions), previews
        self._batch, self._action_provenance = batch, provenance
        self._proposal_failures = failures
        self._inventory_descriptor = self._cached_inventory_descriptor()
        self._inventory_receipts.append(copy.deepcopy(receipt))
        return self._proposals

    def observation(self) -> SimulationObservation:
        # Native observation internally asks for actions three times. Validate
        # once for this synchronous read-only operation, not once per feature pass.
        if self._reading_observation:
            raise RuntimeError("Axis observations cannot be reentered")
        self._observation_inventory = self.proposed_actions()
        self._reading_observation = True
        try:
            return NativeSequentialSimulator.observation(self)
        finally:
            self._observation_inventory = None
            self._reading_observation = False

    def nominal_action_value(self, action: MacroAction) -> float:
        self.assert_model_frozen()
        self._assert_episode()
        return NativeSequentialSimulator.nominal_action_value(self, action)

    def step(self, action: str | int) -> StepResult:
        if self._reading_observation:
            raise RuntimeError("Axis transitions cannot occur during observation reads")
        options = self.proposed_actions()
        if isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_)):
            selected = options[int(action)] if 0 <= action < len(options) else None
        else:
            selected = next((item for item in options if item.action_id == action), None)
        if selected is None:
            raise InvalidActionError("Unknown or stale axis action")
        self._check_cancelled()
        if selected.action_id == "STOP":
            self.terminated = True
            self.termination_reason = self.termination_reason or "explicit_stop"
            self._proposals = None
            self._episode_seal = self._episode_signature()
            return StepResult(self.observation(), 0., True, {"action_id": "STOP", "termination_reason": self.termination_reason})
        if self.terminated:
            raise InvalidActionError("Axis episode terminated")
        value, costs, partial = self._action_value(selected, hidden=True)
        preview = self._previews[selected.action_id]
        provenance = copy.deepcopy(self._action_provenance[selected.action_id])
        self._check_cancelled()  # Last cooperative boundary before irreversible commit.
        self._assert_episode()
        self.engine.commit_preview(preview)
        self._bind_engine_masks()
        if len(partial):
            self._partial_contact_mask[tuple(partial.T)] = True
        self._current_tool = selected.tool_id
        self._actions.append(selected.action_id)
        self.total_reward += value
        if len(self._actions) >= self.config.max_steps:
            self.terminated, self.termination_reason = True, "step_budget"
        info = {**preview.to_history_record(), **costs, "action_id": selected.action_id,
                "axis_proposal": provenance, "termination_reason": self.termination_reason,
                "clinical_deficit_probability": None}
        self._history.append(copy.deepcopy(info))
        self._proposals, self._previews = None, {}
        self._inventory_descriptor = None
        self._episode_seal = self._episode_signature()
        try:
            observation = self.observation()
        except InterruptedError as error:
            raise CommittedTransitionInterrupted(info, value) from error
        return StepResult(observation, value, self.terminated, copy.deepcopy(info))

    def fresh(self, *, max_steps: int | None = None,
              world_generator: WorldGeneratorConfig | None = None) -> AxisColumnNativeSimulator:
        self.assert_model_frozen()
        self._assert_episode()
        return type(self)(self.native_config, proposal_config=self.proposal_config,
            nominal_motor=self.config.nominal_motor if self.config.evidence_available[0] else None,
            nominal_language=self.config.nominal_language if self.config.evidence_available[1] else None,
            reward=self.config.reward,
            world_generator=self.config.world_generator if world_generator is None else world_generator,
            max_steps=self.config.max_steps if max_steps is None else max_steps,
            partial_contact_weight=self.partial_contact_weight,
            compartment_names=dict(self.config.compartment_names), cancelled=self._cancelled)

    def clone(self) -> AxisColumnNativeSimulator:
        self._verified_batch()
        result = NativeSequentialSimulator.clone(self)
        result._action_provenance = copy.deepcopy(self._action_provenance)
        result._inventory_receipts = copy.deepcopy(self._inventory_receipts)
        result._observation_inventory = None
        result._reading_observation = False
        # Provenance dictionaries were copied; their value descriptor is unchanged.
        return result

    def metrics(self) -> dict[str, Any]:
        self._verified_batch()
        result = NativeSequentialSimulator.metrics(self)
        result.update({"simulation_version": AXIS_ADAPTER_VERSION, "input_profile": "RAW",
            "proposal_model_hash": self._proposer.model_hash,
            "proposal_rule": asdict(self.proposal_config),
            "inventory_receipts": copy.deepcopy(self._inventory_receipts),
            "proposal_accounting": {"integrity_calls": self._integrity_calls,
                "integrity_seconds": self._integrity_seconds, "preview_calls": self._preview_calls,
                "preview_seconds": self._preview_seconds},
            "initialization_timing": {"total_seconds": self._initialization_seconds,
                "proposer_preparation_seconds": self._preparer_seconds,
                "native_engine_setup_seconds": self._engine_setup_seconds,
                "scope": "includes initial complete inventory; local diagnostic only"},
            "scope": "experimental_dynamic_inventory; not_selected_route_refinement"})
        return result
