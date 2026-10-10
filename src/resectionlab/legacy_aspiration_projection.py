"""Generated-only, aspiration-restricted view of the shared mixed native task.

This is an integration candidate, not a deployed policy adapter. The native
engine, source, action IDs, rewards and transitions remain authoritative. The
view removes probe candidates from BOTH actor and search; it never presents a
probe as an aspiration action or changes the simulator's tool registry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.native_spatial_task import NativeSpatialStep, NativeSpatialTask
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.spatial_observations import SpatialObservation
from resectionlab.spatial_policy import SpatialPolicy, parameter_hash
from resectionlab.shared_episode import PolicyIdentity
from resectionlab.shared_episode import (execute_policy_episode, execute_search_episode,
                                         verify_strategy_replay)


PROJECTION = "generated-mixed-native-aspiration-view-v1"
PROJECTION_RECORD = {
    "version": PROJECTION,
    "source_observation": "sequential-spatial-observation-v1",
    "actor_observation": "permitted-spatial-observation-v1",
    "row_rule": "retain STOP and current feasible aspirate rows in original order; reject probe rows",
    "transition": "unchanged NativeSpatialTask.step/advance_planning on retained native action IDs",
    "missing_probe_contact": "no probe can be selected by either arm in this constrained episode",
}
PROJECTION_HASH = semantic_digest(PROJECTION_RECORD)
FIXED_SOURCE_HASH = "sha256:271789605cf5b6510f47995b743782d3d298d44b780a39e9929aec72b37b18b4"
FIXED_DECISION_MODEL_HASH = "sha256:f9df2f92c00c26084a28aa49d998db7153915bb559d5c4e10790fa9c662e7bc1"


@dataclass(frozen=True)
class ProjectionEvidence:
    full_observation_hash: str
    projected_observation_hash: str
    retained_action_ids: tuple[str, ...]
    omitted_probe_action_ids: tuple[str, ...]
    projection_hash: str = PROJECTION_HASH


def project_aspiration_observation(observation: SequentialSpatialObservation):
    """Build a new validated v1 DTO; never mutate or unwrap the mixed DTO.

    STOP alone remains legal when no aspiration stroke is currently feasible.
    Each omitted row is explicitly certified as probe by the mixed observation.
    """
    if type(observation) is not SequentialSpatialObservation:
        raise TypeError("Aspiration projection requires the exact mixed observation")
    observation.assert_intact()
    if observation.observed_probe_contact_grid.any():
        raise ValueError("A probe-exposed state cannot be hidden from a legacy aspiration actor")
    modes = observation.action_modes
    if modes[0] != "stop" or any(mode not in ("aspirate", "probe") for mode in modes[1:]):
        raise ValueError("Unknown mixed action mode")
    keep = tuple(index for index, mode in enumerate(modes) if mode != "probe")
    omitted = tuple(observation.action_ids[index] for index, mode in enumerate(modes) if mode == "probe")
    base = observation.base
    projected = replace(base,
        action_ids=tuple(base.action_ids[index] for index in keep),
        action_tool_ids=tuple(base.action_tool_ids[index] for index in keep),
        action_geometry=base.action_geometry[list(keep)].copy(),
        action_mask=base.action_mask[list(keep)].copy())
    if type(projected) is not SpatialObservation:
        raise RuntimeError("Projection did not produce the exact v1 spatial DTO")
    evidence = ProjectionEvidence(observation.fingerprint, projected.fingerprint,
        projected.action_ids, omitted)
    return projected, evidence


class AspirationOnlyNativeTask:
    """Narrow task view shared by policy execution and observed beam search.

    Every legal action is obtained from the projected observation. A direct
    probe ID passed to step/advance_planning is rejected before native commit.
    Clone/fresh/planning_clone carry the same restriction to every search node.
    """

    def __init__(self, native: NativeSpatialTask):
        if type(native) is not NativeSpatialTask or native.case.track != "synthetic_scan":
            raise TypeError("Only an exact generated native task is supported")
        if (native.case.source_hash != FIXED_SOURCE_HASH
                or native.decision_model_hash != FIXED_DECISION_MODEL_HASH
                or native.max_steps != 6):
            raise ValueError("Only the frozen generated desktop source/environment is supported")
        if native.tool_modes is None or set(native.tool_modes.values()) != {"aspirate", "probe"}:
            raise ValueError("A generated mixed aspirate/probe registry is required")
        if (native._engine.probe_contact_mask.any()
                or any(row.get("interaction_mode") == "probe" for row in native.metrics()["history"])):
            raise ValueError("A probe-exposed native history cannot enter the aspiration-only view")
        self._native = native
        self.projection_hash = PROJECTION_HASH

    def _require_clean(self):
        # The native object can be held elsewhere. Refuse any later committed
        # probe, including a zero-contact probe, before exposing another view.
        if (self._native._engine.probe_contact_mask.any()
                or any(row.get("interaction_mode") == "probe"
                       for row in self._native.metrics()["history"])):
            raise ValueError("A probe-exposed native history cannot enter the aspiration-only view")

    @property
    def case(self):
        return self._native.case

    @property
    def decision_model_hash(self):
        return self._native.decision_model_hash

    @property
    def max_steps(self):
        return self._native.max_steps

    @property
    def tool_modes(self):
        return self._native.tool_modes

    @property
    def _engine(self):
        return self._native._engine

    def metrics(self):
        self._require_clean()
        return self._native.metrics()

    @property
    def terminated(self):
        return self._native.terminated

    def observation(self):
        self._require_clean()
        return project_aspiration_observation(self._native.observation())[0]

    def projection_evidence(self):
        self._require_clean()
        return project_aspiration_observation(self._native.observation())[1]

    def _require_projected_action(self, action):
        if type(action) is not str or action not in self.observation().action_ids:
            raise ValueError("Action is absent from the aspiration-only observed inventory")

    def step(self, action):
        self._require_projected_action(action)
        result = self._native.step(action)
        projected = None if result.observation is None else project_aspiration_observation(result.observation)[0]
        return NativeSpatialStep(projected, result.reward, result.terminated, result.info)

    def advance_planning(self, action):
        self._require_projected_action(action)
        result = self._native.advance_planning(action)
        return NativeSpatialStep(None, result.reward, result.terminated, result.info)

    def clone(self):
        self._require_clean()
        return type(self)(self._native.clone())

    def planning_clone(self):
        self._require_clean()
        return type(self)(self._native.planning_clone())

    def fresh(self):
        self._require_clean()
        return type(self)(self._native.fresh())


class BoundAspirationTransferPolicy(torch.nn.Module):
    """Bind an existing v1 actor to this restricted generated task view.

    The wrapper has no new fitted parameters. Its identity records the frozen
    actor identity AND the STOP/aspirate projection. Inference never receives
    native state, reference labels, probe contact or future rewards.
    """

    def __init__(self, base_policy: SpatialPolicy, base_identity: PolicyIdentity,
                 task: AspirationOnlyNativeTask):
        super().__init__()
        if type(base_policy) is not SpatialPolicy or type(base_identity) is not PolicyIdentity:
            raise TypeError("Exact v1 actor and validated identity are required")
        if base_identity.observation_contract != "permitted-spatial-observation-v1":
            raise ValueError("The checkpoint is not a v1 spatial actor")
        if (base_policy.architecture_hash != base_identity.architecture_hash
                or parameter_hash(base_policy) != base_identity.parameter_hash):
            raise ValueError("Base actor no longer matches its frozen identity")
        if (base_identity.training_status == "trained_checkpoint"
                and getattr(base_policy, "_integration_verified_checkpoint_sha256", None)
                    != base_identity.checkpoint_sha256):
            raise ValueError("A trained actor requires the existing safe checkpoint loader")
        if type(task) is not AspirationOnlyNativeTask or task.projection_hash != PROJECTION_HASH:
            raise TypeError("Bound transfer requires the exact aspiration-only task view")
        self.base_policy = base_policy
        self.expected_source_hash = task.case.source_hash
        self.expected_decision_model_hash = task.decision_model_hash
        self.aspirator_tool_ids = frozenset(tool_id for tool_id, mode in task.tool_modes.items()
                                           if mode == "aspirate")
        self.architecture_hash = semantic_digest({
            "base_architecture_hash": base_identity.architecture_hash,
            "projection_hash": PROJECTION_HASH,
            "wrapper": "bound-legacy-aspiration-transfer-policy-v1",
        })
        self.projection_receipt = freeze_json({
            "base_policy_id": base_identity.policy_id,
            "base_architecture_hash": base_identity.architecture_hash,
            "base_parameter_hash": base_identity.parameter_hash,
            "checkpoint_sha256": base_identity.checkpoint_sha256,
            "checkpoint_validation_receipt_sha256": base_identity.checkpoint_validation_receipt_sha256,
            "target_source_hash": self.expected_source_hash,
            "target_decision_model_hash": self.expected_decision_model_hash,
            "projection_hash": PROJECTION_HASH,
            "aspirator_tool_ids": sorted(self.aspirator_tool_ids),
            "scope": "generated_transfer_not_checkpoint_training_domain",
        })
        self._projection_receipt_hash = semantic_digest(self.projection_receipt)
        self._integration_verified_checkpoint_sha256 = base_identity.checkpoint_sha256
        self.identity = PolicyIdentity(
            (base_identity.policy_id + "-aspirate-transfer")[:128], self.architecture_hash,
            parameter_hash(self), "permitted-spatial-observation-v1",
            base_identity.training_status, base_identity.checkpoint_sha256,
            (self._projection_receipt_hash if base_identity.training_status == "trained_checkpoint"
             else None))
        self._identity_hash = semantic_digest(self.identity.__dict__)

    def assert_binding_intact(self):
        if (semantic_digest(self.projection_receipt) != self._projection_receipt_hash
                or semantic_digest(self.identity.__dict__) != self._identity_hash
                or self.expected_source_hash != self.projection_receipt["target_source_hash"]
                or self.expected_decision_model_hash != self.projection_receipt["target_decision_model_hash"]
                or sorted(self.aspirator_tool_ids) != list(self.projection_receipt["aspirator_tool_ids"])
                or self.base_policy.architecture_hash != self.projection_receipt["base_architecture_hash"]
                or parameter_hash(self.base_policy) != self.projection_receipt["base_parameter_hash"]
                or self._integration_verified_checkpoint_sha256 != self.projection_receipt["checkpoint_sha256"]
                or (self.identity.training_status == "trained_checkpoint" and
                    getattr(self.base_policy, "_integration_verified_checkpoint_sha256", None)
                        != self.projection_receipt["checkpoint_sha256"])
                or self.identity.architecture_hash != self.architecture_hash
                or self.identity.parameter_hash != parameter_hash(self)):
            raise RuntimeError("Transfer projection or trained actor changed after binding")

    def act(self, observation, *, stochastic=False, generator=None):
        self.assert_binding_intact()
        if type(observation) is not SpatialObservation:
            raise TypeError("The transfer actor requires the projected exact v1 DTO")
        observation.assert_intact()
        if (observation.source_id != self.expected_source_hash
                or any(tool_id not in self.aspirator_tool_ids for tool_id in observation.action_tool_ids[1:])):
            raise ValueError("Actor observation includes another source or a non-aspiration tool")
        return self.base_policy.act(observation, stochastic=stochastic, generator=generator)


def _projection_trace(task: AspirationOnlyNativeTask, strategy: dict) -> list[dict]:
    """Bind each selected ID to the complete mixed inventory at that state."""
    worker = task.fresh()
    trace = []
    for decision in strategy["decisions"]:
        evidence = worker.projection_evidence()
        if (decision["observation_before"] != evidence.projected_observation_hash
                or decision["action_id"] not in evidence.retained_action_ids):
            raise RuntimeError("Strategy differs from its projected current inventory")
        trace.append({
            "full_observation_hash": evidence.full_observation_hash,
            "projected_observation_hash": evidence.projected_observation_hash,
            "retained_action_ids": list(evidence.retained_action_ids),
            "omitted_probe_action_ids": list(evidence.omitted_probe_action_ids),
            "chosen_action_id": decision["action_id"],
            "projection_hash": PROJECTION_HASH,
        })
        worker.step(decision["action_id"])
    if not worker.terminated:
        raise RuntimeError("Only a complete restricted strategy may be sealed")
    return trace


def execute_matched_aspiration_transfer_pair(task: AspirationOnlyNativeTask,
                                             policy: BoundAspirationTransferPolicy,
                                             *, max_calls: int, beam_width: int,
                                             seconds: float) -> dict:
    """Run actual actor and SEARCH over the same restricted generated inventory.

    Both branches use a nominal planning clone. This is a bounded software
    transfer comparison; it does not score private vascular or patient labels.
    The complete native task may be replayed/scored only after this record is
    sealed and its independent geometry check succeeds.
    """
    if type(task) is not AspirationOnlyNativeTask or type(policy) is not BoundAspirationTransferPolicy:
        raise TypeError("Expected the paired generated projection task and bound actor")
    if (policy.expected_source_hash != task.case.source_hash
            or policy.expected_decision_model_hash != task.decision_model_hash
            or task.projection_hash != PROJECTION_HASH):
        raise ValueError("Actor and restricted search task bindings differ")
    if task.terminated or task.metrics()["steps"] != 0:
        raise ValueError("Matched actor/search requires the same untouched native initial state")
    policy.assert_binding_intact()
    nominal = task.planning_clone()
    actor_record = execute_policy_episode(nominal, policy, policy.identity)
    search_record = execute_search_episode(task, max_calls=max_calls, beam_width=beam_width, seconds=seconds)
    if (not verify_strategy_replay(nominal, actor_record)
            or not verify_strategy_replay(nominal, search_record)
            or actor_record["source_hash"] != search_record["source_hash"]
            or actor_record["environment_contract_hash"] != search_record["environment_contract_hash"]
            or actor_record["decisions"][0]["observation_before"] !=
               search_record["decisions"][0]["observation_before"]):
        raise RuntimeError("Actor/search nominal shared replay or source binding failed")
    actor_trace = _projection_trace(nominal, actor_record)
    search_trace = _projection_trace(nominal, search_record)
    policy.assert_binding_intact()
    record = {
        "schema": "generated-legacy-aspiration-transfer-pair-v1",
        "source_hash": task.case.source_hash,
        "decision_model_hash": task.decision_model_hash,
        "projection_hash": PROJECTION_HASH,
        "projection_receipt": thaw_json(policy.projection_receipt),
        "projection_receipt_hash": policy._projection_receipt_hash,
        "original_checkpoint_validation_receipt_sha256":
            policy.projection_receipt["checkpoint_validation_receipt_sha256"],
        "checkpoint_training_domain_matches_target": False,
        "search_action_availability": "same STOP+aspirate projection as actor at each state",
        "search_reward_source": "permitted_generated_nominal_target_only",
        "private_reference_scored": False,
        "clinical_validation": False,
        "actor": {"strategy": actor_record, "projection_trace": actor_trace},
        "search": {"strategy": search_record, "projection_trace": search_trace},
    }
    record["seal"] = semantic_digest(record)
    return record
