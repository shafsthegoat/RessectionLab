"""Bounded, patient-specific masked policy-gradient optimization.

This module has no final-evaluation entry point. Checkpoints are selected only
on explicitly designated selection worlds. The simulator supplies observations
and legal macro-actions; latent anatomy is never an actor input. Rewards are
declared simulator surrogates, and learning need not outperform search.

The compact baseline is REINFORCE with a learned state-value baseline, Adam,
entropy regularization and gradient clipping. CPU execution and a private random
generator make local runs small and resumable without a shared GPU service.
"""

from __future__ import annotations

from .data_policy import historical_only

import copy
import functools
import hashlib
import inspect
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
from torch import nn

from .policy_inputs import policy_input_profile


class Observation(Protocol):
    action_features: np.ndarray
    state_features: np.ndarray
    action_mask: np.ndarray
    action_ids: tuple[str, ...]


class Simulator(Protocol):
    decision_model_hash: str
    case_hash: str
    world_generator_fingerprint: str

    def reset(self, seed: int = 0) -> Observation: ...
    def step(self, action: int) -> Any: ...
    def metrics(self) -> dict[str, Any]: ...


def _decision_array(value: Any) -> np.ndarray:
    """Detached, bytes-backed observation evidence; callbacks cannot edit inputs."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _decision_array_identity(array: np.ndarray | None):
    return None if array is None else (array.shape, array.dtype.str, array.strides,
                                      hashlib.sha256(array.tobytes()).hexdigest())


@dataclass(frozen=True)
class DecisionContext:
    """Caller labels only; recording never grants a world an evaluation role."""

    role: str | None = None
    update: int | None = None
    episode: int | None = None
    panel: int | None = None


@dataclass(frozen=True)
class DecisionInputs:
    source_action_features: np.ndarray
    source_state_features: np.ndarray
    action_mask: np.ndarray
    action_ids: tuple[str, ...]
    action_features: np.ndarray | None
    state_features: np.ndarray | None
    actor_action_features: np.ndarray | None
    _identity: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_identity", self._current_identity())

    def _current_identity(self):
        return tuple(_decision_array_identity(getattr(self, name)) for name in (
            "source_action_features", "source_state_features", "action_mask",
            "action_features", "state_features", "actor_action_features")) + (self.action_ids,)

    def assert_unchanged(self):
        if self._current_identity() != self._identity:
            raise ValueError("Recorded decision input array layout or content changed")


@dataclass(frozen=True)
class DecisionRecord:
    context: DecisionContext
    seed: int
    step: int
    inputs: DecisionInputs
    logits: np.ndarray | None
    value: float | None
    selected_index: int
    selected_action_id: str
    decision_rule: str
    forced_reason: str | None
    forward_evaluated: bool
    _logits_identity: Any = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_logits_identity", _decision_array_identity(self.logits))

    def to_dict(self) -> dict[str, Any]:
        """Strict JSON; masked negative infinity is encoded as the string '-inf'."""
        self.inputs.assert_unchanged()
        if _decision_array_identity(self.logits) != self._logits_identity:
            raise ValueError("Recorded decision logits layout or content changed")
        inputs = {}
        for name in ("source_action_features", "source_state_features", "action_mask",
                     "action_features", "state_features", "actor_action_features"):
            array = getattr(self.inputs, name)
            inputs[name] = None if array is None else {
                "dtype": array.dtype.str, "shape": list(array.shape), "values": array.tolist()}
        inputs["action_ids"] = list(self.inputs.action_ids)
        logits = None if self.logits is None else {
            "dtype": self.logits.dtype.str, "shape": list(self.logits.shape),
            "values": ["-inf" if np.isneginf(value) else float(value) for value in self.logits]}
        return {"version": "learner-decision-observer-v1", **asdict(self.context),
                "seed": self.seed, "step": self.step, "inputs": inputs,
                "logits": logits, "value": self.value, "selected_index": self.selected_index,
                "selected_action_id": self.selected_action_id, "decision_rule": self.decision_rule,
                "forced_reason": self.forced_reason, "forward_evaluated": self.forward_evaluated}


DecisionObserver = Callable[[DecisionRecord], None]


def _emit_decision(observer: DecisionObserver, inputs: DecisionInputs,
                   logits: torch.Tensor | None, value: torch.Tensor | None, *,
                   context: DecisionContext, seed: int, step: int, action: int,
                   rule: str, forced_reason: str | None = None) -> None:
    copied_logits = None if logits is None else _decision_array(logits)
    copied_value = None if value is None else float(value.detach())
    if copied_logits is not None and (
            copied_logits.shape != inputs.action_mask.shape
            or not np.isfinite(copied_logits[inputs.action_mask]).all()
            or not np.all(np.isfinite(copied_logits) | np.isneginf(copied_logits))
            or copied_value is None or not math.isfinite(copied_value)):
        raise ValueError("Decision record contains invalid policy outputs")
    observer(DecisionRecord(copy.deepcopy(context), int(seed), step, inputs,
        copied_logits, copied_value, action, inputs.action_ids[action],
        rule, forced_reason, logits is not None))


def _unevaluated_inputs(observation: Observation) -> DecisionInputs:
    """A forced rollout STOP reads source evidence but performs no policy forward."""
    return DecisionInputs(_decision_array(observation.action_features),
        _decision_array(observation.state_features), _decision_array(observation.action_mask),
        tuple(observation.action_ids), None, None, None)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization-step and learner-wall limits, preserved across resume.

    Selection transitions are counted separately; ``max_environment_steps``
    limits optimization transitions, not a combined search/selection budget.
    """

    seed: int = 0
    max_environment_steps: int = 4096
    max_gradient_steps: int = 128
    max_wall_seconds: float = 60.0
    max_episode_steps: int = 64
    episodes_per_update: int = 4
    checkpoint_interval: int = 8
    hidden_features: int = 32
    learning_rate: float = 0.003
    gamma: float = 1.0
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    max_gradient_norm: float = 5.0

    def __post_init__(self) -> None:
        for name in ("max_environment_steps", "max_gradient_steps", "max_episode_steps",
                     "episodes_per_update", "checkpoint_interval", "hidden_features"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("max_wall_seconds", "learning_rate", "max_gradient_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if any(not math.isfinite(x) or x < 0 for x in (self.entropy_weight, self.value_weight)):
            raise ValueError("loss weights must be finite and nonnegative")


@dataclass(frozen=True)
class PolicyRollout:
    actions: tuple[str, ...]
    total_reward: float
    environment_steps: int
    termination: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TrainingResult:
    status: str
    optimizer_mode: str
    initial_checkpoint_hash: str
    selected_checkpoint_hash: str
    latest_checkpoint_hash: str
    decision_model_hash: str
    gradient_steps: int
    optimization_environment_steps: int
    selection_environment_steps: int
    elapsed_seconds: float
    initial_selection_return: float | None
    selected_selection_return: float | None
    output_dir: str
    shared_checkpoint_hash: str | None = None
    initialization_seconds: float | None = None


class RolloutInterrupted(RuntimeError):
    """Partial rollouts retain their resource count but cannot rank a checkpoint."""

    def __init__(self, environment_steps: int):
        super().__init__("rollout interrupted by cancellation or wall-time limit")
        self.environment_steps = environment_steps


class MaskedPatientPolicy(nn.Module):
    """A shared candidate scorer permits variable action counts across cases."""

    def __init__(self, action_features: int, state_features: int, hidden: int = 32, *,
                 input_profile: str = "RAW"):
        super().__init__()
        self.dimensions = (action_features, state_features, hidden)
        self._input_profile = policy_input_profile(input_profile, action_features=action_features,
                                                  state_features=state_features)
        self.actor = nn.Sequential(nn.Linear(action_features + state_features, hidden),
                                   nn.Tanh(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(state_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        if input_profile == "FEATURE_UNITS":
            self.register_buffer("action_divisors", torch.tensor(self._input_profile.action_divisors,
                                                                 dtype=torch.float32))

    @property
    def input_profile(self):
        return self._input_profile

    def checkpoint_profile(self) -> dict[str, Any]:
        return {"input_profile": self.input_profile.to_dict(),
                "input_profile_hash": self.input_profile.fingerprint}

    def actor_inputs(self, actions: torch.Tensor) -> torch.Tensor:
        """Fixed units only; identity mode performs no arithmetic at all."""
        if self.input_profile.profile_id == "RAW":
            if "action_divisors" in self._buffers:
                raise ValueError("RAW policy unexpectedly contains a feature transform")
            return actions
        expected = torch.tensor(self.input_profile.action_divisors, dtype=torch.float32,
                                device=self.action_divisors.device)
        if self.action_divisors.requires_grad or not torch.equal(self.action_divisors, expected):
            raise ValueError("Policy input divisor buffer changed")
        return actions / self.action_divisors

    def forward(self, observation: Observation, *,
                capture_inputs: Callable[[DecisionInputs], None] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        source_actions = np.array(observation.action_features, copy=True)
        actions = torch.as_tensor(source_actions, dtype=torch.float32)
        source_state = np.array(observation.state_features, copy=True)
        state = torch.as_tensor(source_state, dtype=torch.float32).flatten()
        mask = torch.as_tensor(np.array(observation.action_mask, copy=True), dtype=torch.bool)
        action_ids = observation.action_ids
        if (actions.ndim != 2 or mask.shape != (len(actions),)
                or actions.shape[1] != self.dimensions[0] or len(state) != self.dimensions[1]
                or len(action_ids) != len(actions)):
            raise ValueError("observation feature dimensions changed")
        if not torch.isfinite(actions).all() or not torch.isfinite(state).all():
            raise ValueError("nonfinite policy observation")
        if len(mask) == 0 or not bool(mask[0]) or observation.action_ids[0] != "STOP":
            raise ValueError("STOP must remain action zero and available")
        actor_actions = self.actor_inputs(actions)
        logits = self.actor(torch.cat((actor_actions, state.expand(len(actions), -1)), dim=1)).squeeze(-1)
        logits, value = logits.masked_fill(~mask, -torch.inf), self.value(state).squeeze(-1)
        if capture_inputs is not None:
            capture_inputs(DecisionInputs(_decision_array(source_actions), _decision_array(source_state),
                _decision_array(mask), tuple(action_ids), _decision_array(actions),
                _decision_array(state), _decision_array(actor_actions)))
        return logits, value


def policy_hash(policy_or_state: nn.Module | Mapping[str, torch.Tensor]) -> str:
    state = policy_or_state.state_dict() if isinstance(policy_or_state, nn.Module) else policy_or_state
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def trainable_parameter_hash(policy: nn.Module) -> str:
    """Compare paired initialization tensors, not behavioral policy identity."""
    return policy_hash(dict(policy.named_parameters()))


def checkpoint_input_profile(state: Mapping[str, Any], *, expected_input_profile: str | None = None):
    """Validate registry metadata and every saved policy buffer before loading.

    Legacy checkpoints without profile metadata are identity-only. A coherent
    tensor checksum does not authorize arbitrary divisors or schema changes.
    """
    dimensions = state["dimensions"]
    metadata = state.get("input_profile")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Checkpoint input profile metadata must be a registry record")
    profile_id = "RAW" if metadata is None else metadata.get("profile_id")
    profile = policy_input_profile(profile_id, action_features=dimensions[0], state_features=dimensions[1])
    if expected_input_profile is not None and profile_id != expected_input_profile:
        raise ValueError("Checkpoint policy input profile differs from requested profile")
    if metadata is None:
        if state.get("input_profile_hash") is not None:
            raise ValueError("Checkpoint input profile metadata is incomplete")
    elif metadata != profile.to_dict() or state.get("input_profile_hash") != profile.fingerprint:
        raise ValueError("Checkpoint input profile metadata differs from immutable registry")
    for key in ("policy", "selected_policy"):
        if key not in state:
            continue
        weights = state[key]
        buffer = weights.get("action_divisors")
        if profile_id == "RAW":
            if "action_divisors" in weights:
                raise ValueError("RAW checkpoint contains a divisor buffer")
        elif (not isinstance(buffer, torch.Tensor) or buffer.dtype != torch.float32
              or buffer.requires_grad or not torch.equal(buffer.cpu(), torch.tensor(profile.action_divisors, dtype=torch.float32))):
            raise ValueError("Checkpoint input divisor buffer differs from registered profile")
    return profile


def _validate_simulator_profile(simulator: Simulator, profile_id: str,
                                observation: Observation | None = None) -> dict[str, Any] | None:
    from .native_axis_accounting import _RecordedAxisSimulator
    from .native_axis_simulation import AxisColumnNativeSimulator
    from .native_axis_policy_schema import axis_observation_contract
    raw_axis = simulator
    if type(simulator) is _RecordedAxisSimulator:
        simulator.assert_model_frozen()
        if simulator._owner.input_profile.profile_id != profile_id:
            raise ValueError("Axis accounting policy profile differs from requested RAW/FEATURE_UNITS profile")
        raw_axis = simulator._raw
    if type(raw_axis) is AxisColumnNativeSimulator:
        if observation is None:
            if profile_id == "RAW":
                return None  # Preserve the pre-existing identity-only query.
            raise ValueError("FEATURE_UNITS requires an actual native axis observation supplied explicitly")
        policy_input_profile(profile_id)  # Only immutable registered profiles.
        return axis_observation_contract(raw_axis, observation)
    if profile_id == "RAW":
        return
    from .native_simulation import NATIVE_ACTION_FEATURE_NAMES, NativeSequentialSimulator
    from .procedural_learning import _ProceduralPool, native_observation_schema
    if type(simulator) is _ProceduralPool:
        simulator = simulator.active
    if type(simulator) is not NativeSequentialSimulator:
        raise ValueError("FEATURE_UNITS requires actual native observation semantics")
    schema = native_observation_schema(simulator)
    profile = policy_input_profile(profile_id)
    if (tuple(NATIVE_ACTION_FEATURE_NAMES) != profile.action_feature_names
            or tuple(schema["action_feature_names"]) != profile.action_feature_names
            or tuple(schema["state_feature_names"]) != profile.state_feature_names):
        raise ValueError("Policy input profile native feature order or semantics changed")


def _json_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                                separators=(",", ":")).encode()).hexdigest()


def numerical_source_hashes() -> dict[str, str]:
    """Conservatively bind every headless module, including inherited geometry.

    Hashing only a simulator's class file misses imported transition helpers and
    inherited methods. GUI edits remain outside this numerical resume contract.
    """
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def _partition_record(manifest: Any, expected_role: str) -> dict[str, Any]:
    role = getattr(manifest.role, "value", manifest.role)
    if role != expected_role:
        raise ValueError(f"expected {expected_role} worlds, received {role}")
    seeds = tuple(manifest.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("world seeds must be distinct nonnegative integers")
    generator = manifest.generator
    generator_record = asdict(generator) if hasattr(generator, "__dataclass_fields__") else generator
    return {"role": role, "case_hash": manifest.case_hash, "seeds": list(seeds),
            "generator": generator_record, "partition_hash": manifest.partition_hash,
            "planning_hash": getattr(manifest, "planning_hash", None)}


def _validate_partitions(optimization: Any, selection: Any) -> dict[str, Any]:
    opt = _partition_record(optimization, "optimization")
    sel = _partition_record(selection, "selection")
    if opt["case_hash"] != sel["case_hash"] or opt["generator"] != sel["generator"]:
        raise ValueError("optimization and selection must use the same frozen case/world generator")
    if set(opt["seeds"]) & set(sel["seeds"]):
        raise ValueError("optimization and selection worlds overlap")
    return {"optimization": opt, "selection": sel}


def _assert_model(simulator: Simulator, expected: str) -> None:
    if hasattr(simulator, "assert_model_frozen"):
        simulator.assert_model_frozen()
    if simulator.decision_model_hash != expected:
        raise ValueError("decision model changed during optimization; start a new experiment")


def _assert_partition_binding(simulator: Simulator, manifest: Any) -> None:
    if getattr(simulator, "case_hash", None) != manifest.case_hash:
        raise ValueError("world partition case differs from actual simulator case")
    if getattr(simulator, "world_generator_fingerprint", None) != manifest.generator.fingerprint:
        raise ValueError("world partition generator differs from actual simulator sampler")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@historical_only("WEIGHT_LINEAGE_UNVERIFIED")
def load_policy(checkpoint: str | Path, *, selected: bool = True,
                expected_input_profile: str | None = None) -> MaskedPatientPolicy:
    """Load only tensor/primitive checkpoints, never arbitrary pickled classes."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    profile = checkpoint_input_profile(state, expected_input_profile=expected_input_profile)
    policy = MaskedPatientPolicy(*state["dimensions"], input_profile=profile.profile_id)
    weights = state["selected_policy"] if selected and "selected_policy" in state else state["policy"]
    policy.load_state_dict(weights)
    expected = state.get("selected_hash" if selected and "selected_policy" in state else "policy_hash")
    if expected is not None and policy_hash(policy) != expected:
        raise ValueError("checkpoint weight hash mismatch")
    policy.eval()
    return policy


def clone_checkpoint_policy(checkpoint: str | Path, *, expected_input_profile: str | None = None) -> MaskedPatientPolicy:
    """Independent weights for a future adaptation caller; no evaluation claim."""
    return copy.deepcopy(load_policy(checkpoint, expected_input_profile=expected_input_profile))


def rollout_policy(policy: MaskedPatientPolicy | None, simulator: Simulator, *, seed: int,
                   max_steps: int = 64, expected_model_hash: str | None = None,
                   interrupt: Callable[[], bool] | None = None,
                   decision_observer: DecisionObserver | None = None,
                   decision_context: DecisionContext = DecisionContext()) -> PolicyRollout:
    """Deterministic replay; ``None`` is the explicit immediate-STOP baseline.

    This does not independently certify geometry or grant an evaluation role.
    Callers must retain the role of their world manifest in the saved evaluator.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    expected = expected_model_hash or simulator.decision_model_hash
    observation = simulator.reset(seed)
    _assert_model(simulator, expected)
    actions: list[str] = []
    total = 0.0
    for step in range(max_steps):
        if interrupt is not None and interrupt():
            raise RolloutInterrupted(len(actions))
        with torch.no_grad():
            if decision_observer is None:
                action = 0 if policy is None or step == max_steps - 1 else int(policy(observation)[0].argmax())
            elif policy is None or step == max_steps - 1:
                action = 0
                inputs = _unevaluated_inputs(observation)
                logits, value = None, None
                rule = "stop_baseline" if policy is None else "forced_stop"
                forced_reason = None if policy is None else "episode_step_limit"
            else:
                captured: list[DecisionInputs] = []
                logits, value = policy(observation, capture_inputs=captured.append)
                action = int(logits.argmax())
                inputs = captured[0]
                rule, forced_reason = "deterministic_argmax", None
        actions.append(observation.action_ids[action])
        if decision_observer is not None:
            if actions[-1] != inputs.action_ids[action]:
                raise ValueError("Observation action IDs changed during decision recording")
            _emit_decision(decision_observer, inputs, logits, value, context=decision_context,
                seed=seed, step=step, action=action, rule=rule, forced_reason=forced_reason)
        result = simulator.step(action)
        _assert_model(simulator, expected)
        if not math.isfinite(float(result.reward)):
            raise ValueError("nonfinite simulator reward")
        total += float(result.reward)
        observation = result.observation
        if result.terminated:
            return PolicyRollout(tuple(actions), total, len(actions),
                                 "STOP" if action == 0 else "environment_terminal", simulator.metrics())
    raise ValueError("simulator did not terminate after STOP")


def _record_failures(function: Callable[..., TrainingResult]) -> Callable[..., TrainingResult]:
    @functools.wraps(function)
    def run(*args: Any, **kwargs: Any) -> TrainingResult:
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if "output_dir" in kwargs:
                try:
                    directory = Path(kwargs["output_dir"]).resolve()
                    directory.mkdir(parents=True, exist_ok=True)
                    failure = {"status": "failed", "exception": type(error).__name__,
                               "reason": str(error), "at_unix_seconds": time.time(),
                               "latest_checkpoint_may_precede_failure": True}
                    with (directory / "failures.jsonl").open("a") as log:
                        log.write(json.dumps(failure, allow_nan=False) + "\n")
                except OSError:
                    pass  # Preserve the original error when its destination is unwritable.
            raise
    return run


@historical_only("RECORDED_EXPERIENCE_REQUIRED")
@_record_failures
def train_patient_policy(
    simulator_factory: Callable[[], Simulator],
    optimization_manifest: Any,
    selection_manifest: Any,
    *,
    config: TrainingConfig = TrainingConfig(),
    output_dir: str | Path,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    resume: bool = False,
    shared_checkpoint: str | Path | None = None,
    population_case_group: str | None = None,
    population_case_aliases: tuple[str, ...] = (),
    procedural_checkpoint: str | Path | None = None,
    procedural_target: Any = None,
    input_profile: str = "RAW",
    procedural_study_id: str | None = None,
    decision_observer: DecisionObserver | None = None,
) -> TrainingResult:
    """Train a fresh policy or isolated clone, with selection-world-only ranking.

    Cancellation is checked before every transition. An interrupted unfinished
    batch is discarded, its executed transitions remain counted, and the latest
    optimizer/RNG state is saved. Resume preserves *total* limits; it cannot
    silently extend a run. This function never opens final-evaluation manifests.
    Initialization is measured separately; initial selection and subsequent
    optimization/selection share the cumulative learner wall-time budget.
    """
    initialization_started = time.perf_counter()
    partitions = _validate_partitions(optimization_manifest, selection_manifest)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("run exists; use resume or a new output directory")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError("no checkpoint to resume")
    if resume and (shared_checkpoint is not None or procedural_checkpoint is not None):
        raise ValueError("resume uses the saved initialization, not a new shared checkpoint")
    if shared_checkpoint is not None and (procedural_checkpoint is not None or procedural_target is not None):
        raise ValueError("A run cannot mix population and procedural initializations")
    if procedural_target is not None and procedural_checkpoint is None and not resume:
        raise ValueError("Procedural target requires its validated initialization checkpoint")
    simulator = simulator_factory()
    _assert_partition_binding(simulator, optimization_manifest)
    frozen_hash = simulator.decision_model_hash
    first = simulator.reset(optimization_manifest.seeds[0])
    dimensions = (np.asarray(first.action_features).shape[1], np.asarray(first.state_features).size,
                  config.hidden_features)
    axis_observation = _validate_simulator_profile(simulator, input_profile, first)
    if axis_observation is not None and (resume or shared_checkpoint is not None
            or procedural_checkpoint is not None or procedural_target is not None):
        raise ValueError("Axis policy compatibility permits fresh scratch training only; no transfer or resume")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = MaskedPatientPolicy(*dimensions, input_profile=input_profile)
    shared_hash = None
    population_context = None
    procedural_context = None
    if resume:
        initialization = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_input_profile(initialization, expected_input_profile=input_profile)
        saved_population = initialization.get("population_initialization")
        if saved_population is not None:
            shared_checkpoint = directory / "population-source.pt"
            if population_case_group is None:
                population_case_group = saved_population["target_group"]
                population_case_aliases = tuple(saved_population["target_aliases"])
        saved_procedural = initialization.get("procedural_initialization")
        if saved_procedural is not None:
            from .procedural_learning import TransferTarget
            if saved_population is not None:
                raise ValueError("Saved run mixes incompatible initialization domains")
            procedural_checkpoint = directory / "procedural-source.pt"
            if procedural_target is None:
                procedural_target = TransferTarget(**saved_procedural["target"])
            if procedural_study_id is None:
                procedural_study_id = saved_procedural.get("study_id")
    if shared_checkpoint is not None:
        if input_profile != "RAW":
            raise ValueError("Analytic population initialization requires its original RAW profile")
        from .population_learning import feature_schema, validate_population_checkpoint
        source = torch.load(shared_checkpoint, map_location="cpu", weights_only=True)
        if source.get("kind") != "population_checkpoint":
            raise ValueError("population adaptation requires an actual shared training checkpoint with provenance")
        if population_case_group is None:
            raise ValueError("Population adaptation requires the declared target patient group and aliases")
        validation = validate_population_checkpoint(shared_checkpoint, target_case_hash=optimization_manifest.case_hash,
            target_group=population_case_group, target_aliases=population_case_aliases,
            expected_dimensions=dimensions, expected_feature_schema=feature_schema(simulator))
        copied = directory / "population-source.pt"
        if not resume:
            copied.write_bytes(Path(shared_checkpoint).read_bytes())
        if hashlib.sha256(copied.read_bytes()).hexdigest() != validation["checkpoint_file_sha256"]:
            raise ValueError("Shared population checkpoint changed while copying or resuming")
        population_context = {"target_case_hash": optimization_manifest.case_hash,
            "target_group": population_case_group, "target_aliases": list(population_case_aliases),
            "checkpoint_file_sha256": validation["checkpoint_file_sha256"],
            "provenance_hash": validation["provenance_hash"], "scope": validation["scope"]}
        shared = clone_checkpoint_policy(copied, expected_input_profile=input_profile)
        policy.load_state_dict(copy.deepcopy(shared.state_dict()))
        shared_hash = policy_hash(shared)
    if procedural_checkpoint is not None:
        from .procedural_learning import validate_procedural_adaptation
        validation = validate_procedural_adaptation(procedural_checkpoint, procedural_target,
            simulator, optimization_manifest, selection_manifest, config,
            input_profile=input_profile, study_id=procedural_study_id)
        copied = directory / "procedural-source.pt"
        if not resume:
            copied.write_bytes(Path(procedural_checkpoint).read_bytes())
        if hashlib.sha256(copied.read_bytes()).hexdigest() != validation["checkpoint_file_sha256"]:
            raise ValueError("Procedural checkpoint changed while copying or resuming")
        procedural_context = {"target": asdict(procedural_target),
            "checkpoint_file_sha256": validation["checkpoint_file_sha256"],
            "provenance_hash": validation["provenance_hash"], "scope": validation["scope"]}
        if procedural_study_id is not None:
            procedural_context["study_id"] = procedural_study_id
        procedural_context.update(policy.checkpoint_profile())
        shared = clone_checkpoint_policy(copied, expected_input_profile=input_profile)
        if policy_hash(shared) != validation["policy_hash"]:
            raise ValueError("Procedural actor changed after provenance validation")
        policy.load_state_dict(copy.deepcopy(shared.state_dict()))
        shared_hash = policy_hash(shared)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initial_hash = policy_hash(policy)
    initial_trainable_hash = trainable_parameter_hash(policy)
    state: dict[str, Any] = {
        "gradient_steps": 0, "optimization_environment_steps": 0,
        "selection_environment_steps": 0, "episode_index": 0,
        "completed_episodes": 0, "discarded_partial_batches": 0,
        "elapsed_seconds": 0.0, "selection_seconds": 0.0,
        "initial_selection_return": None, "selected_selection_return": None,
        "initial_actor_hash": policy_hash(policy.actor),
        "initial_trainable_parameter_hash": initial_trainable_hash,
        "selection_history": [], "optimization_history": [],
    }
    selected_weights = copy.deepcopy(policy.state_dict())
    selected_hash = initial_hash
    simulator_source = Path(inspect.getfile(type(simulator)))
    contract = {"schema_version": 1, "algorithm": "masked_reinforce_state_value_v2",
                **policy.checkpoint_profile(),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "simulator_implementation_sha256": hashlib.sha256(simulator_source.read_bytes()).hexdigest(),
                "numerical_source_sha256": numerical_source_hashes(),
                "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                            "python": platform.python_version(), "device": "cpu"},
                "config": asdict(config), "partitions": partitions,
                "population_initialization": population_context,
                "procedural_initialization": procedural_context,
                "timing_contract": "optimization_selection_budget_v2_initialization_separate",
                "elapsed_seconds_scope": "cumulative optimization and selection, including initial selection; initialization excluded",
                "decision_model_hash": frozen_hash, "dimensions": list(dimensions)}
    if axis_observation is not None:
        contract["axis_observation_contract"] = axis_observation
    contract_hash = _json_hash(contract)
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["contract_hash"] != contract_hash:
            raise ValueError("resume contract changed: model, worlds or optimization settings differ")
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    if resume:
        policy.load_state_dict(saved["policy"])
        if policy_hash(policy) != saved["policy_hash"]:
            raise ValueError("latest checkpoint weight hash mismatch")
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["random_state"])
        state = saved["state"]
        initial_hash, shared_hash = saved["initial_hash"], saved["shared_hash"]
        selected_weights, selected_hash = saved["selected_policy"], saved["selected_hash"]
        if policy_hash(selected_weights) != selected_hash:
            raise ValueError("selected checkpoint weight hash mismatch")
    else:
        _atomic_checkpoint(directory / "initial.pt", {"dimensions": list(dimensions),
                           **policy.checkpoint_profile(), "trainable_parameter_hash": initial_trainable_hash,
                           "policy": copy.deepcopy(policy.state_dict()), "policy_hash": initial_hash})
        _atomic_json(directory / "contract.json", {**contract, "contract_hash": contract_hash,
                     "initial_checkpoint_hash": initial_hash, "shared_checkpoint_hash": shared_hash,
                     "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                                  "processor": platform.processor(), "torch": str(torch.__version__),
                                  "device": "cpu", "torch_threads": torch.get_num_threads()},
                     "clinical_deficit_probability": None,
                     "final_evaluation_used_for_optimization": False})
    # Each invocation measures setup independently. Resume retains only the
    # already consumed optimization/selection budget from this same contract.
    # Initial selection below is part of the learner budget, not preparation.
    started = time.perf_counter()
    initialization_seconds = started - initialization_started
    state["initialization_seconds"] = state.get("initialization_seconds", 0.0) + initialization_seconds
    state["initialization_seconds_this_invocation"] = initialization_seconds
    elapsed_before = state["elapsed_seconds"]
    cancelled = cancelled or (lambda: False)

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def save(status: str) -> TrainingResult:
        state["elapsed_seconds"] = elapsed()
        latest_hash = policy_hash(policy)
        checkpoint_export_started = time.perf_counter()
        _atomic_checkpoint(checkpoint_path, {
            "contract_hash": contract_hash, "dimensions": list(dimensions),
            **policy.checkpoint_profile(), "trainable_parameter_hash": trainable_parameter_hash(policy),
            "policy": policy.state_dict(), "policy_hash": latest_hash,
            "selected_policy": selected_weights, "selected_hash": selected_hash,
            "optimizer": optimizer.state_dict(), "random_state": generator.get_state(),
            "initial_hash": initial_hash, "shared_hash": shared_hash, "state": state,
            "population_initialization": population_context,
            "procedural_initialization": procedural_context,
        })
        checkpoint_export_seconds = time.perf_counter() - checkpoint_export_started
        optimizer_mode = ("PROCEDURAL_PRETRAINED_ADAPTED" if procedural_context is not None
                          else "POPULATION_ADAPTED" if shared_hash else "PATIENT_SCRATCH_RL")
        result = TrainingResult(status, optimizer_mode,
                                initial_hash, selected_hash, latest_hash, frozen_hash,
                                state["gradient_steps"], state["optimization_environment_steps"],
                                state["selection_environment_steps"], state["elapsed_seconds"],
                                state["initial_selection_return"], state["selected_selection_return"],
                                str(directory), shared_hash, state["initialization_seconds"])
        _atomic_json(directory / "result.json", {**asdict(result), **state,
                     **policy.checkpoint_profile(),
                     "latest_trainable_parameter_hash": trainable_parameter_hash(policy),
                     "final_checkpoint_export_seconds": checkpoint_export_seconds if status != "running" else None,
                     "final_checkpoint_export_seconds_scope": "last atomic checkpoint write only; result JSON write and return are outside this measurement",
                     "latest_actor_hash": policy_hash(policy.actor),
                     "actor_parameters_changed": policy_hash(policy.actor) != state["initial_actor_hash"],
                     "selection_rule": "maximum mean deterministic selection return; earliest wins ties",
                     "wall_limit_semantics": "cooperative; a simulator/gradient call may overrun the deadline",
                     "elapsed_seconds_scope": contract["elapsed_seconds_scope"],
                     "initialization_seconds_scope": "cumulative model/policy/provenance/optimizer/contract setup across invocations; excluded from learner budget",
                     "clinical_deficit_probability": None, "learning_improvement_guaranteed": False})
        return result

    def select() -> bool:
        nonlocal selected_weights, selected_hash
        rewards: list[float] = []
        selection_started = time.perf_counter()
        for selection_episode, seed in enumerate(selection_manifest.seeds):
            if cancelled() or elapsed() >= config.max_wall_seconds:
                state["selection_seconds"] += time.perf_counter() - selection_started
                return False
            instance = simulator_factory()
            _assert_partition_binding(instance, selection_manifest)
            _assert_model(instance, frozen_hash)
            try:
                replay = rollout_policy(policy, instance, seed=seed, max_steps=config.max_episode_steps,
                                        expected_model_hash=frozen_hash,
                                        interrupt=lambda: cancelled() or elapsed() >= config.max_wall_seconds,
                                        decision_observer=decision_observer,
                                        decision_context=DecisionContext("selection", state["gradient_steps"],
                                            selection_episode, len(state["selection_history"])))
            except RolloutInterrupted as interrupted:
                state["selection_environment_steps"] += interrupted.environment_steps
                state["selection_seconds"] += time.perf_counter() - selection_started
                return False
            rewards.append(replay.total_reward)
            state["selection_environment_steps"] += replay.environment_steps
        panel_elapsed_seconds = time.perf_counter() - selection_started
        state["selection_seconds"] += panel_elapsed_seconds
        mean = float(np.mean(rewards))
        state["selection_history"].append({"gradient_steps": state["gradient_steps"],
            "optimization_environment_steps": state["optimization_environment_steps"],
            "panel_elapsed_seconds": panel_elapsed_seconds,
            "mean_return": mean, "world_count": len(rewards), "checkpoint_hash": policy_hash(policy)})
        if state["initial_selection_return"] is None:
            state["initial_selection_return"] = mean
        if state["selected_selection_return"] is None or mean > state["selected_selection_return"]:
            selected_weights = copy.deepcopy(policy.state_dict())
            selected_hash = policy_hash(selected_weights)
            state["selected_selection_return"] = mean
        return True

    if state["initial_selection_return"] is None:
        if not select():
            return save("cancelled" if cancelled() else "wall_time_budget")
        save("running")
    status = "gradient_budget"
    while state["gradient_steps"] < config.max_gradient_steps:
        if cancelled():
            status = "cancelled"
            break
        if elapsed() >= config.max_wall_seconds:
            status = "wall_time_budget"
            break
        remaining = config.max_environment_steps - state["optimization_environment_steps"]
        if remaining <= 0:
            status = "environment_budget"
            break
        losses: list[torch.Tensor] = []
        episode_returns: list[float] = []
        episode_source_hashes: list[str] = []
        for _ in range(config.episodes_per_update):
            if state["optimization_environment_steps"] >= config.max_environment_steps:
                break
            seed = optimization_manifest.seeds[state["episode_index"] % len(optimization_manifest.seeds)]
            state["episode_index"] += 1
            instance = simulator_factory()
            _assert_partition_binding(instance, optimization_manifest)
            _assert_model(instance, frozen_hash)
            observation = instance.reset(seed)
            log_probs, entropies, values, rewards = [], [], [], []
            complete = False
            for step in range(config.max_episode_steps):
                if cancelled() or elapsed() >= config.max_wall_seconds:
                    break
                remaining = config.max_environment_steps - state["optimization_environment_steps"]
                if remaining <= 0:
                    break
                if decision_observer is None:
                    logits, value = policy(observation)
                else:
                    captured = []
                    logits, value = policy(observation, capture_inputs=captured.append)
                distribution = torch.distributions.Categorical(logits=logits)
                forced_stop = step == config.max_episode_steps - 1 or remaining == 1
                action = 0 if forced_stop else int(torch.multinomial(distribution.probs, 1, generator=generator))
                if decision_observer is not None:
                    reasons = []
                    if step == config.max_episode_steps - 1:
                        reasons.append("episode_step_limit")
                    if remaining == 1:
                        reasons.append("optimization_transition_budget")
                    _emit_decision(decision_observer, captured[0], logits, value,
                        context=DecisionContext("optimization", state["gradient_steps"], state["episode_index"] - 1),
                        seed=seed, step=step, action=action,
                        rule="forced_stop" if forced_stop else "sampled_categorical",
                        forced_reason="+".join(reasons) if reasons else None)
                transition = instance.step(action)
                state["optimization_environment_steps"] += 1
                _assert_model(instance, frozen_hash)
                reward = float(transition.reward)
                if not math.isfinite(reward):
                    raise ValueError("nonfinite simulator reward")
                # A budget-forced STOP is outside the behavior policy and has no actor term.
                log_probs.append(None if forced_stop else distribution.log_prob(torch.tensor(action)))
                entropies.append(distribution.entropy())
                values.append(value)
                rewards.append(reward)
                observation = transition.observation
                if transition.terminated:
                    complete = True
                    break
            if not complete:
                state["discarded_partial_batches"] += 1
                losses.clear()
                break
            state["completed_episodes"] += 1
            episode_returns.append(sum(rewards))
            episode_source_hashes.append(getattr(instance, "optimization_source_hash", instance.case_hash))
            future = 0.0
            returns: list[float] = []
            for reward in reversed(rewards):
                future = reward + config.gamma * future
                returns.append(future)
            returns.reverse()
            terms = []
            for time_index, (log_prob, entropy, value, target) in enumerate(zip(log_probs, entropies, values, returns)):
                advantage = target - value
                term = config.value_weight * advantage.square()
                if log_prob is not None:
                    term = term - (config.gamma ** time_index) * log_prob * advantage.detach() - config.entropy_weight * entropy
                terms.append(term)
            # Sum within each episode, then average episodes. Dividing by the
            # sampled episode length would downweight longer action sequences
            # and bias the episodic-return gradient toward short trajectories.
            losses.append(torch.stack(terms).sum())
        if cancelled() or elapsed() >= config.max_wall_seconds:
            status = "cancelled" if cancelled() else "wall_time_budget"
            break
        if not losses:
            status = "environment_budget"
            break
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite policy loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite gradient")
        actor_norm = math.sqrt(sum(float(parameter.grad.square().sum()) for parameter in policy.actor.parameters()
                                   if parameter.grad is not None))
        optimizer.step()
        state["gradient_steps"] += 1
        state["optimization_history"].append({"gradient_steps": state["gradient_steps"],
            "optimization_environment_steps": state["optimization_environment_steps"],
            "episode_source_case_hashes": episode_source_hashes,
            "mean_return": float(np.mean(episode_returns)), "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm), "actor_gradient_norm_after_clip": actor_norm})
        if state["gradient_steps"] % config.checkpoint_interval == 0:
            if not select():
                status = "cancelled" if cancelled() else "wall_time_budget"
                break
            snapshot = save("running")
            if progress:
                progress(asdict(snapshot))
    # Rank only checkpoints with completed selection panels. A time/cancel stop
    # retains the last valid selection rather than ranking a partial world panel.
    if status not in ("cancelled", "wall_time_budget"):
        last_selected_update = state["selection_history"][-1]["gradient_steps"]
        if last_selected_update != state["gradient_steps"]:
            if not select():
                status = "cancelled" if cancelled() else "wall_time_budget"
    return save(status)
