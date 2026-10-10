"""Small common execution/replay pathway for generated development episodes.

Both search and a real ``nn.Module`` actor select IDs from the same task
observation and commit through ``NativeSpatialTask.step``. This module does not
load checkpoints, fit a policy, or admit patient data. An untrained actor is
reported as a software control, never as an RL result.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.observed_search import observed_beam_search
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation, VERSION
from resectionlab.spatial_observations import SpatialObservation
from resectionlab.spatial_policy import parameter_hash


SCHEMA = "shared-sequential-strategy-v1"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
PHYSICAL_KEYS = frozenset({
    "action_id", "interaction_mode", "tool_id", "entry_mm", "tip_mm",
    "axis_unit", "removed_indices_native", "contact_indices_native",
    "probe_contact_indices_native", "microsteps", "source_hash",
    "source_state_hash", "decision_model_hash", "native_affine", "source_shape",
    "native_footprint", "removed_volume_mm3", "geometry_unknowns",
    "temporal_rule", "retraction", "complete_tool_path_length_mm",
})


@dataclass(frozen=True)
class PolicyIdentity:
    policy_id: str
    architecture_hash: str
    parameter_hash: str
    observation_contract: str
    training_status: str
    checkpoint_sha256: str | None = None
    checkpoint_validation_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_id or len(self.policy_id) > 128:
            raise ValueError("A bounded policy ID is required")
        if self.training_status not in {"untrained_software_control", "trained_checkpoint"}:
            raise ValueError("Policy training status must be explicit")
        if self.observation_contract not in {"permitted-spatial-observation-v1", VERSION}:
            raise ValueError("Policy observation contract is unsupported")
        for value in (self.architecture_hash, self.parameter_hash):
            if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
                raise ValueError("Policy architecture and parameter digests are required")
        if self.training_status == "trained_checkpoint":
            for value in (self.checkpoint_sha256, self.checkpoint_validation_receipt_sha256):
                if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
                    raise ValueError("A trained policy needs checkpoint and validation receipt hashes")
        elif self.checkpoint_sha256 is not None or self.checkpoint_validation_receipt_sha256 is not None:
            raise ValueError("An untrained software control cannot claim a checkpoint")


def identity_for_untrained_spatial_policy(policy, *, policy_id: str) -> PolicyIdentity:
    """Bind a newly constructed, untrained v1 actor without claiming learning."""
    return PolicyIdentity(policy_id, policy.architecture_hash, parameter_hash(policy),
                          "permitted-spatial-observation-v1", "untrained_software_control")


def _validate_actor(policy, identity: PolicyIdentity) -> None:
    if not isinstance(policy, torch.nn.Module) or not callable(getattr(policy, "act", None)):
        raise TypeError("A policy arm needs an actual executable torch policy, not a scripted selector")
    if not isinstance(identity, PolicyIdentity):
        raise TypeError("Policy identity is required")
    if (getattr(policy, "architecture_hash", None) != identity.architecture_hash
            or parameter_hash(policy) != identity.parameter_hash):
        raise ValueError("Executable policy parameters or architecture differ from the identity")
    if (identity.training_status == "trained_checkpoint" and
            getattr(policy, "_integration_verified_checkpoint_sha256", None) !=
                identity.checkpoint_sha256):
        raise ValueError("TRAINED_CHECKPOINT_LOADER_REQUIRED: model lacks a byte-bound safe load")


def _observation_contract(observation) -> str:
    if type(observation) is SpatialObservation:
        observation.assert_intact()
        return "permitted-spatial-observation-v1"
    if type(observation) is SequentialSpatialObservation:
        observation.assert_intact()
        return VERSION
    raise TypeError("Task returned an unsupported observation contract")


def _mode(observation, action_id: str) -> str:
    index = observation.action_ids.index(action_id)
    if type(observation) is SequentialSpatialObservation:
        return observation.action_modes[index]
    return "stop" if index == 0 else "aspirate"


def _commit(task, observation, action_id: str) -> dict[str, Any]:
    if action_id not in observation.action_ids:
        raise ValueError("Selected action is absent from the current observation")
    index = observation.action_ids.index(action_id)
    if not bool(observation.action_mask[index]):
        raise ValueError("Selected action is masked")
    mode = _mode(observation, action_id)
    before = observation.fingerprint
    step = task.step(action_id)
    if step.observation is None and not step.terminated:
        raise RuntimeError("Nonterminal transition did not return its successor observation")
    physical = {key: value for key, value in step.info.items() if key in PHYSICAL_KEYS}
    if physical.get("action_id") != action_id:
        raise RuntimeError("Native transition committed a different action ID")
    native_mode = step.info.get("interaction_mode")
    if native_mode is not None and native_mode != mode:
        raise RuntimeError("Observed action mode differs from the committed native interaction")
    physical["interaction_mode"] = mode
    return thaw_json(freeze_json({"action_id": action_id, "action_mode": mode,
        "tool_id": observation.action_tool_ids[index],
        "observation_before": before,
        "observation_after": None if step.observation is None else step.observation.fingerprint,
        "terminated": bool(step.terminated), "physical_transition": physical,
        "post_model_state_hash": task._engine.state_hash}))


def _record(task, *, method: str, identity: dict, decisions: list[dict],
            search_accounting: dict | None = None) -> dict:
    metrics = task.metrics()
    if not task.terminated or len(decisions) != metrics["steps"]:
        raise RuntimeError("Only a complete authoritative native episode can be exported")
    record = {"version": SCHEMA, "evidence_level": "generated_software_development",
              "clinical_validation": "none", "method": method, "policy_identity": identity,
              "source_hash": metrics["source_hash"],
              "environment_contract_hash": metrics["decision_model_hash"],
              "observation_track": metrics["observation_track"],
              "action_ids": [row["action_id"] for row in decisions],
              "decisions": decisions, "physical_history_hash": semantic_digest(
                  [row["physical_transition"] for row in decisions]),
              "terminal_model_state_hash": task._engine.state_hash,
              "search_accounting": search_accounting}
    return thaw_json(freeze_json(record))


def _require_generated_task(task) -> None:
    # This first adapter has no patient-source admission or withheld evaluation.
    if getattr(getattr(task, "case", None), "track", None) != "synthetic_scan":
        raise ValueError("GENERATED_ONLY: patient tasks require a separate admitted adapter")


def execute_policy_episode(task, policy, identity: PolicyIdentity) -> dict:
    """Run an actual policy forward at each evolving state, then native step."""
    _require_generated_task(task)
    _validate_actor(policy, identity)
    worker = task.fresh()
    decisions: list[dict] = []
    initial = worker.observation()
    if _observation_contract(initial) != identity.observation_contract:
        raise ValueError("Policy was not built for this task observation contract")
    if identity.observation_contract == VERSION and getattr(policy, "observation_contract", None) != VERSION:
        raise ValueError("Legacy spatial policy cannot silently decide mixed-mode actions")
    policy.eval()
    with torch.no_grad():
        while not worker.terminated:
            observation = worker.observation()
            if _observation_contract(observation) != identity.observation_contract:
                raise RuntimeError("Observation schema changed inside an episode")
            chosen = policy.act(observation)
            decisions.append(_commit(worker, observation, chosen))
    if parameter_hash(policy) != identity.parameter_hash:
        raise RuntimeError("Policy changed during inference")
    status = identity.training_status
    method = "POLICY_SOFTWARE_CONTROL" if status == "untrained_software_control" else "LEARNED_POLICY"
    return _record(worker, method=method, identity=identity.__dict__, decisions=decisions)


def execute_search_episode(task, *, max_calls: int, beam_width: int,
                           seconds: float) -> dict:
    """Search on the task's permitted nominal clone; replay on native step."""
    _require_generated_task(task)
    sequence, accounting = observed_beam_search(task, max_calls=max_calls,
        beam_width=beam_width, seconds=seconds, transition_mode="eager")
    worker = task.fresh()
    decisions: list[dict] = []
    for action_id in sequence:
        decisions.append(_commit(worker, worker.observation(), action_id))
    if not worker.terminated:
        raise RuntimeError("Search returned an incomplete episode")
    return _record(worker, method="SEARCH", identity={"policy_id": None,
        "training_status": None, "checkpoint_sha256": None,
        "observation_contract": _observation_contract(task.observation())},
        decisions=decisions, search_accounting=accounting)


def execute_action_sequence(task, action_ids: tuple[str, ...] | list[str]) -> dict:
    """Exercise a declared sequence; clearly label it as a scripted interface test."""
    _require_generated_task(task)
    worker = task.fresh()
    decisions: list[dict] = []
    for action_id in action_ids:
        if type(action_id) is not str:
            raise ValueError("Sequence action IDs must be strings")
        decisions.append(_commit(worker, worker.observation(), action_id))
    return _record(worker, method="SCRIPTED_INTERFACE_TEST", identity={
        "policy_id": None, "training_status": None, "checkpoint_sha256": None,
        "observation_contract": _observation_contract(task.observation())},
        decisions=decisions)


def verify_strategy_replay(task, record: dict) -> bool:
    """Re-execute UI action history through the authoritative task transition.

    This verifies modeled physical replay only. It does not validate clinical
    effects, checkpoint provenance, or private patient outcomes.
    """
    _require_generated_task(task)
    if (type(record) is not dict or record.get("version") != SCHEMA
            or record.get("evidence_level") != "generated_software_development"
            or type(record.get("action_ids")) is not list
            or type(record.get("decisions")) is not list):
        raise ValueError("Unsupported or incomplete strategy record")
    worker = task.fresh()
    decisions: list[dict] = []
    for action_id in record["action_ids"]:
        if type(action_id) is not str:
            raise ValueError("Replay action IDs must be strings")
        decisions.append(_commit(worker, worker.observation(), action_id))
    if (not worker.terminated or decisions != record["decisions"]
            or semantic_digest([row["physical_transition"] for row in decisions]) !=
                record.get("physical_history_hash")
            or worker._engine.state_hash != record.get("terminal_model_state_hash")
            or worker.decision_model_hash != record.get("environment_contract_hash")
            or worker.case.source_hash != record.get("source_hash")):
        raise ValueError("Replayed transitions differ from the displayed strategy")
    return True
