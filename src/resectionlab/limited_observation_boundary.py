"""Generated-fixture limited planner -> sealed sequence -> private target audit.

V1 admits only the exact two-step opening fixture and two private target worlds.
It does not admit patients, hidden support/hazards, or new observation regimes.
This is an accidental-misuse boundary, not Python or operating-system security.
No reference loader, reference identifier, or rich task enters the planner API.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
import re

import numpy as np

from .core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .geometry import AccessWindow, ToolGeometry
from .native_spatial_evaluation import evaluate_native_spatial_episode
from .native_spatial_task import NativeSpatialCase, NativeSpatialTask
from .simulation import RewardSpec

VERSION = "limited-observation-boundary-v1"
SCOPE = "generated-private-target-boundary-v1"
# Fixed permitted fixture lineage, deliberately not a caller's scope assertion.
_FIXTURE_INPUT_HASH = "sha256:fec81aa8923aa3924b98fe5efefb37a2b63f470cc5c93fefbe80c36b2c9ae522"
_REFERENCE_TARGET_HASHES = {
    "original_generated_target": "sha256:686c7325140a03d147ac6bc9c2692306b03eeb0f57ed7048ee8b96856dcb64cb",
    "zero_target_counterfactual": "sha256:f21cf7b01e3781999b28748a803db4f1536a1d65213b6b80b312dff511b027a7",
}
_METHOD_FIELDS = {"method", "parameter_hash", "configuration_hash",
                  "model_transition_calls", "actor_forward_calls", "completed", "time_cap_reached"}
_ACTION_FIELDS = {"action_id", "entry_mm", "tip_mm", "tool_id"}


def _exact(record, names, label):
    if not isinstance(record, Mapping) or set(record) != set(names):
        raise ValueError(label + ": exact fields required")
    return dict(record)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _version(version, scope):
    if version != VERSION or scope != SCOPE:
        raise ValueError("Only the versioned generated private-target fixture is supported")


def _access_record(access):
    return {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
            "radius_mm": access.radius_mm, "window_id": access.window_id}


def _array(value, shape, dtype, *, binary=False):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "biuf" or not np.isfinite(array).all()
            or (binary and not np.isin(array, (0, 1)).all())):
        raise ValueError("Exact finite generated-fixture arrays are required")
    return immutable_array(array, dtype)


@dataclass(frozen=True, slots=True)
class LimitedPlanningSpec:
    version: str
    scope: str
    structural_intensity: np.ndarray
    affine_ras_mm: np.ndarray
    nominal_support: np.ndarray
    nominal_target: np.ndarray
    support_provenance: Mapping
    target_provenance: Mapping
    access: AccessWindow
    tools: tuple[ToolGeometry, ...]
    crop_shape: tuple[int, int, int]
    intensity_normalization: str
    proposal_mode: str
    max_steps: int
    reward: RewardSpec
    fingerprint: str = field(init=False)

    def __post_init__(self):
        _version(self.version, self.scope)
        if (type(self.max_steps) is not int or self.max_steps != 2
                or type(self.access) is not AccessWindow or type(self.reward) is not RewardSpec
                or not isinstance(self.tools, (list, tuple)) or len(self.tools) != 2
                or any(type(tool) is not ToolGeometry for tool in self.tools)):
            raise ValueError("The typed two-step generated fixture is required")
        for name, shape, dtype, binary in (
                ("structural_intensity", (9, 9, 7), np.float32, False),
                ("affine_ras_mm", (4, 4), np.float64, False),
                ("nominal_support", (9, 9, 7), bool, True),
                ("nominal_target", (9, 9, 7), np.float32, True)):
            object.__setattr__(self, name, _array(getattr(self, name), shape, dtype, binary=binary))
        for name in ("support_provenance", "target_provenance"):
            object.__setattr__(self, name, freeze_json(_exact(getattr(self, name),
                {"source_kind", "derivation"}, name)))
        object.__setattr__(self, "access", AccessWindow(**_access_record(self.access)))
        object.__setattr__(self, "tools", tuple(ToolGeometry(**asdict(tool)) for tool in self.tools))
        object.__setattr__(self, "reward", RewardSpec(**asdict(self.reward)))
        object.__setattr__(self, "crop_shape", tuple(self.crop_shape))
        object.__setattr__(self, "fingerprint", semantic_digest(self._payload()))
        self.assert_intact()

    def _payload(self):
        return {"version": self.version, "scope": self.scope,
            **{name: getattr(self, name).tolist() for name in
               ("structural_intensity", "affine_ras_mm", "nominal_support", "nominal_target")},
            "support_provenance": thaw_json(self.support_provenance),
            "target_provenance": thaw_json(self.target_provenance),
            "access": _access_record(self.access), "tools": [asdict(tool) for tool in self.tools],
            "crop_shape": list(self.crop_shape), "intensity_normalization": self.intensity_normalization,
            "proposal_mode": self.proposal_mode, "max_steps": self.max_steps, "reward": asdict(self.reward)}

    def assert_intact(self):
        if semantic_digest(self._payload()) != self.fingerprint or self.fingerprint != _FIXTURE_INPUT_HASH:
            raise ValueError("Limited inputs differ from the frozen generated-fixture lineage")

    def to_record(self):
        self.assert_intact()
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_record(cls, record):
        values = _exact(record, {item.name for item in fields(cls)}, "limited planning spec")
        claimed = values.pop("fingerprint")
        values["access"] = AccessWindow(**_exact(values["access"],
            {item.name for item in fields(AccessWindow)}, "access"))
        values["tools"] = tuple(ToolGeometry(**_exact(tool,
            {item.name for item in fields(ToolGeometry)}, "tool")) for tool in values["tools"])
        values["reward"] = RewardSpec(**_exact(values["reward"],
            {item.name for item in fields(RewardSpec)}, "reward"))
        result = cls(**values)
        if result.fingerprint != claimed:
            raise ValueError("Limited input fingerprint mismatch")
        return result


def _require_spec(spec):
    if type(spec) is not LimitedPlanningSpec:
        raise ValueError("Only an exact LimitedPlanningSpec is accepted; no rich task handoff")
    spec.assert_intact()


def _case(spec, target):
    return NativeSpatialCase(spec.structural_intensity, spec.nominal_support, target,
        spec.affine_ras_mm, spec.access, spec.tools, track="synthetic_scan",
        support_source_kind=spec.support_provenance["source_kind"],
        support_derivation=spec.support_provenance["derivation"],
        nominal_target=spec.nominal_target, target_source_kind=spec.target_provenance["source_kind"],
        target_derivation=spec.target_provenance["derivation"], crop_shape=spec.crop_shape,
        intensity_normalization=spec.intensity_normalization, proposal_mode=spec.proposal_mode)


def build_limited_planning_task(spec: LimitedPlanningSpec) -> NativeSpatialTask:
    """Build solely from permitted inputs; every method gets the same nominal model."""
    _require_spec(spec)
    return NativeSpatialTask(_case(spec, spec.nominal_target), max_steps=spec.max_steps,
                             reward=spec.reward).planning_clone()


def _method_record(record):
    value = _exact(record, _METHOD_FIELDS, "method record")
    if (value["method"] not in {"SEARCH", "IL", "RL", "HYBRID", "MANUAL", "STOP"}
            or not _digest(value["parameter_hash"]) or not _digest(value["configuration_hash"])
            or any(type(value[key]) is not int or value[key] < 0
                   for key in ("model_transition_calls", "actor_forward_calls"))
            or value["completed"] is not True or value["time_cap_reached"] is not False):
        raise ValueError("Only explicitly completed methods with frozen identities and accounting can seal")
    return freeze_json(value)


def _physical_records(history):
    return tuple(freeze_json({key: row.get(key) for key in sorted(_ACTION_FIELDS)}) for row in history)


def _complete(action_ids, horizon, terminal_reason):
    if (type(horizon) is not int or horizon != 2 or not 1 <= len(action_ids) <= horizon
            or any(type(action) is not str or not action for action in action_ids)):
        raise ValueError("A complete STOP-or-horizon action sequence is required")
    stops = [index for index, action in enumerate(action_ids) if action == "STOP"]
    expected = "STOP" if stops else "HORIZON"
    if ((stops and stops != [len(action_ids) - 1]) or (not stops and len(action_ids) != horizon)
            or terminal_reason != expected):
        raise ValueError("Incomplete sequence or invalid STOP termination")


@dataclass(frozen=True, slots=True)
class FrozenLimitedPlan:
    version: str
    scope: str
    limited_input_hash: str
    decision_model_hash: str
    initial_observation_hash: str
    tool_catalog_hash: str
    action_ids: tuple[str, ...]
    physical_actions: tuple[Mapping, ...]
    horizon: int
    terminal_reason: str
    method_record: Mapping
    seal_hash: str = field(init=False)

    def __post_init__(self):
        _version(self.version, self.scope)
        for name in ("limited_input_hash", "decision_model_hash", "initial_observation_hash", "tool_catalog_hash"):
            if not _digest(getattr(self, name)):
                raise ValueError("Complete plan binding hashes are required")
        object.__setattr__(self, "action_ids", tuple(self.action_ids))
        _complete(self.action_ids, self.horizon, self.terminal_reason)
        records = tuple(freeze_json(_exact(row, _ACTION_FIELDS, "physical action"))
                        for row in self.physical_actions)
        if tuple(row["action_id"] for row in records) != self.action_ids:
            raise ValueError("Physical actions must match the ordered action sequence")
        object.__setattr__(self, "physical_actions", records)
        object.__setattr__(self, "method_record", _method_record(self.method_record))
        object.__setattr__(self, "seal_hash", semantic_digest(self._payload()))

    def _payload(self):
        return {item.name: thaw_json(getattr(self, item.name)) for item in fields(self) if item.init}

    def to_record(self):
        self.assert_intact()
        return {**self._payload(), "seal_hash": self.seal_hash}

    def assert_intact(self):
        if semantic_digest(self._payload()) != self.seal_hash:
            raise ValueError("Frozen plan seal mismatch")

    @classmethod
    def from_record(cls, record):
        values = _exact(record, {item.name for item in fields(cls)}, "frozen plan")
        claimed = values.pop("seal_hash")
        result = cls(**values)
        if result.seal_hash != claimed:
            raise ValueError("Frozen plan seal mismatch")
        return result


def _replay_nominal(plan, spec):
    """Validate every sealed action and physical binding before any private read."""
    _require_spec(spec)
    if type(plan) is not FrozenLimitedPlan:
        raise ValueError("An exact FrozenLimitedPlan is required")
    plan.assert_intact()
    _version(plan.version, plan.scope)
    _complete(plan.action_ids, plan.horizon, plan.terminal_reason)
    _method_record(plan.method_record)
    task = build_limited_planning_task(spec)
    if (plan.limited_input_hash != spec.fingerprint or plan.horizon != spec.max_steps
            or plan.decision_model_hash != task.decision_model_hash
            or plan.initial_observation_hash != task.observation().fingerprint
            or plan.tool_catalog_hash != semantic_digest([asdict(tool) for tool in spec.tools])):
        raise ValueError("Plan input, observation, tool or objective binding mismatch")
    for action in plan.action_ids:
        task.advance_planning(action)
    if not task.terminated or _physical_records(task.metrics()["history"]) != plan.physical_actions:
        raise ValueError("Frozen physical sequence differs from nominal execution")
    return task


def freeze_limited_plan(planning_task, *, spec: LimitedPlanningSpec, method_record) -> FrozenLimitedPlan:
    """Seal verified nominal execution and caller-declared method identity/costs.

    Digests detect content changes; they do not authenticate who planned a
    sequence or independently measure the caller's earlier planning work.
    """
    _require_spec(spec)
    if type(planning_task) is not NativeSpatialTask:
        raise ValueError("An exact nominal planning task is required")
    current = planning_task.metrics()
    initial = build_limited_planning_task(spec)
    if (current["planning_estimator_only"] is not True or current["terminated"] is not True
            or current["source_hash"] != initial.case.source_hash
            or current["reference_hash"] != initial.case.reference_hash
            or planning_task.decision_model_hash != initial.decision_model_hash):
        raise ValueError("Only a completed, bound nominal planning task can seal")
    action_ids = tuple(row["action_id"] for row in current["history"])
    plan = FrozenLimitedPlan(VERSION, SCOPE, spec.fingerprint, initial.decision_model_hash,
        initial.observation().fingerprint, semantic_digest([asdict(tool) for tool in spec.tools]),
        action_ids, _physical_records(current["history"]), spec.max_steps,
        "STOP" if action_ids and action_ids[-1] == "STOP" else "HORIZON", method_record)
    replay = _replay_nominal(plan, spec)
    if semantic_digest(replay.metrics()) != semantic_digest(current):
        raise ValueError("Nominal committed history differs from independent nominal replay")
    return plan


@dataclass(frozen=True, slots=True)
class HeldOutTargetBinding:
    """Evaluator-owned metadata checked before calling its zero-argument loader."""
    version: str
    scope: str
    limited_input_hash: str
    reference_kind: str

    def __post_init__(self):
        self.assert_intact()

    def assert_intact(self):
        _version(self.version, self.scope)
        if (self.limited_input_hash != _FIXTURE_INPUT_HASH
                or self.reference_kind not in _REFERENCE_TARGET_HASHES):
            raise ValueError("Reference source binding or generated-target lineage mismatch")

    def to_record(self):
        self.assert_intact()
        return asdict(self)

    @classmethod
    def from_record(cls, record):
        return cls(**_exact(record, {item.name for item in fields(cls)}, "reference binding"))


@dataclass(frozen=True, slots=True)
class HeldOutTargetReference:
    binding: HeldOutTargetBinding
    reference_target: np.ndarray
    fingerprint: str = field(init=False)

    def __post_init__(self):
        if type(self.binding) is not HeldOutTargetBinding:
            raise ValueError("An exact generated target binding is required")
        self.binding.assert_intact()
        object.__setattr__(self, "reference_target", _array(self.reference_target, (9, 9, 7),
                                                           np.float32, binary=True))
        object.__setattr__(self, "fingerprint", semantic_digest(self._payload()))
        self.assert_intact()

    def _payload(self):
        return {"binding": self.binding.to_record(), "target_hash": array_digest(self.reference_target),
                "provenance": "generated opening fixture; no patient anatomy",
                "coverage": "complete generated grid; target only"}

    def assert_intact(self):
        self.binding.assert_intact()
        if (array_digest(self.reference_target) != _REFERENCE_TARGET_HASHES[self.binding.reference_kind]
                or semantic_digest(self._payload()) != self.fingerprint):
            raise ValueError("Private target differs from its declared generated reference")


def _report(plan, binding, *, status, reference_hash=None, episode=None):
    record = {"version": VERSION, "scope": SCOPE, "status": status,
        "plan_seal_hash": plan.seal_hash, "limited_input_hash": plan.limited_input_hash,
        "action_ids": list(plan.action_ids), "method_record": thaw_json(plan.method_record),
        "reference_binding": binding.to_record(), "reference_hash": reference_hash,
        "independent_episode": episode,
        "hidden_support_validated": False, "hidden_hazards_validated": False,
        "patient_generalization": None, "motor_surrogate": None, "language_surrogate": None,
        "clinical_deficit_probability": None, "file_access_isolation": False}
    return freeze_json({**record, "report_hash": semantic_digest(record)})


def evaluate_frozen_limited_plan(plan: FrozenLimitedPlan, *, spec: LimitedPlanningSpec,
        reference_binding: HeldOutTargetBinding,
        load_reference: Callable[[], HeldOutTargetReference]) -> Mapping:
    """Replay a sealed sequence, then independently reconstruct private outcomes.

    All preflight errors raise before the loader. Loader/reference/replay failures
    return explicit sealed failure records with no outcome. These records must
    be retained by the caller; this in-memory API does not persist files.
    No policy/search callback is accepted after reference access.
    """
    nominal = _replay_nominal(plan, spec)
    if type(reference_binding) is not HeldOutTargetBinding:
        raise ValueError("An exact evaluator-owned reference binding is required")
    reference_binding.assert_intact()
    if reference_binding.limited_input_hash != plan.limited_input_hash or not callable(load_reference):
        raise ValueError("Reference binding and zero-argument loader required")
    # Detached validated snapshots cannot be changed by ordinary caller mutation.
    spec = LimitedPlanningSpec.from_record(spec.to_record())
    plan = FrozenLimitedPlan.from_record(plan.to_record())
    reference_binding = HeldOutTargetBinding.from_record(reference_binding.to_record())
    try:
        reference = load_reference()
    except Exception:
        return _report(plan, reference_binding, status="reference_load_failed")
    try:
        if type(reference) is not HeldOutTargetReference:
            raise ValueError("The loader must return an exact HeldOutTargetReference")
        reference.assert_intact()
        if reference.binding != reference_binding:
            raise ValueError("Loaded reference binding mismatch")
    except Exception:
        return _report(plan, reference_binding, status="reference_rejected")
    try:
        evaluator = NativeSpatialTask(_case(spec, reference.reference_target),
                                     max_steps=spec.max_steps, reward=spec.reward)
        if evaluator.case.source_hash != nominal.case.source_hash:
            raise ValueError("Reference replacement changed permitted source")
        for action in plan.action_ids:
            evaluator.step(action)
        if _physical_records(evaluator.metrics()["history"]) != plan.physical_actions:
            raise ValueError("Reference replacement changed the frozen physical sequence")
        episode = evaluate_native_spatial_episode(evaluator)
        status = "evaluated" if episode["accepted"] else "independent_audit_failed"
    except Exception:
        return _report(plan, reference_binding, status="evaluator_replay_failed", reference_hash=reference.fingerprint)
    return _report(plan, reference_binding, status=status, reference_hash=reference.fingerprint, episode=episode)
