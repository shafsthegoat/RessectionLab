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

import copy
import functools
import hashlib
import inspect
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
from torch import nn


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


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization-step and total-wall limits, preserved across resume.

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


class RolloutInterrupted(RuntimeError):
    """Partial rollouts retain their resource count but cannot rank a checkpoint."""

    def __init__(self, environment_steps: int):
        super().__init__("rollout interrupted by cancellation or wall-time limit")
        self.environment_steps = environment_steps


class MaskedPatientPolicy(nn.Module):
    """A shared candidate scorer permits variable action counts across cases."""

    def __init__(self, action_features: int, state_features: int, hidden: int = 32):
        super().__init__()
        self.dimensions = (action_features, state_features, hidden)
        self.actor = nn.Sequential(nn.Linear(action_features + state_features, hidden),
                                   nn.Tanh(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(state_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, observation: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.as_tensor(np.array(observation.action_features, copy=True), dtype=torch.float32)
        state = torch.as_tensor(np.array(observation.state_features, copy=True), dtype=torch.float32).flatten()
        mask = torch.as_tensor(np.array(observation.action_mask, copy=True), dtype=torch.bool)
        if (actions.ndim != 2 or mask.shape != (len(actions),)
                or actions.shape[1] != self.dimensions[0] or len(state) != self.dimensions[1]
                or len(observation.action_ids) != len(actions)):
            raise ValueError("observation feature dimensions changed")
        if not torch.isfinite(actions).all() or not torch.isfinite(state).all():
            raise ValueError("nonfinite policy observation")
        if len(mask) == 0 or not bool(mask[0]) or observation.action_ids[0] != "STOP":
            raise ValueError("STOP must remain action zero and available")
        logits = self.actor(torch.cat((actions, state.expand(len(actions), -1)), dim=1)).squeeze(-1)
        return logits.masked_fill(~mask, -torch.inf), self.value(state).squeeze(-1)


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


def _json_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                                separators=(",", ":")).encode()).hexdigest()


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
            "generator": generator_record, "partition_hash": manifest.partition_hash}


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


def load_policy(checkpoint: str | Path, *, selected: bool = True) -> MaskedPatientPolicy:
    """Load only tensor/primitive checkpoints, never arbitrary pickled classes."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = MaskedPatientPolicy(*state["dimensions"])
    weights = state["selected_policy"] if selected and "selected_policy" in state else state["policy"]
    policy.load_state_dict(weights)
    expected = state.get("selected_hash" if selected and "selected_policy" in state else "policy_hash")
    if expected is not None and policy_hash(policy) != expected:
        raise ValueError("checkpoint weight hash mismatch")
    policy.eval()
    return policy


def clone_checkpoint_policy(checkpoint: str | Path) -> MaskedPatientPolicy:
    """Independent weights for a future adaptation caller; no evaluation claim."""
    return copy.deepcopy(load_policy(checkpoint))


def rollout_policy(policy: MaskedPatientPolicy | None, simulator: Simulator, *, seed: int,
                   max_steps: int = 64, expected_model_hash: str | None = None,
                   interrupt: Callable[[], bool] | None = None) -> PolicyRollout:
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
            action = 0 if policy is None or step == max_steps - 1 else int(policy(observation)[0].argmax())
        actions.append(observation.action_ids[action])
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
) -> TrainingResult:
    """Train a fresh policy or isolated clone, with selection-world-only ranking.

    Cancellation is checked before every transition. An interrupted unfinished
    batch is discarded, its executed transitions remain counted, and the latest
    optimizer/RNG state is saved. Resume preserves *total* limits; it cannot
    silently extend a run. This function never opens final-evaluation manifests.
    """
    started = time.perf_counter()
    partitions = _validate_partitions(optimization_manifest, selection_manifest)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("run exists; use resume or a new output directory")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError("no checkpoint to resume")
    if resume and shared_checkpoint is not None:
        raise ValueError("resume uses the saved initialization, not a new shared checkpoint")
    simulator = simulator_factory()
    _assert_partition_binding(simulator, optimization_manifest)
    frozen_hash = simulator.decision_model_hash
    first = simulator.reset(optimization_manifest.seeds[0])
    dimensions = (np.asarray(first.action_features).shape[1], np.asarray(first.state_features).size,
                  config.hidden_features)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = MaskedPatientPolicy(*dimensions)
    shared_hash = None
    if shared_checkpoint is not None:
        source = torch.load(shared_checkpoint, map_location="cpu", weights_only=True)
        provenance = source.get("population_provenance", {})
        if (source.get("kind") != "population_checkpoint"
                or not provenance.get("development_case_hashes")
                or not provenance.get("training_run_hash")):
            raise ValueError("population adaptation requires an actual shared training checkpoint with provenance")
        if optimization_manifest.case_hash in provenance["development_case_hashes"]:
            raise ValueError("patient overlaps shared population development data")
        shared = clone_checkpoint_policy(shared_checkpoint)
        if shared.dimensions != dimensions:
            raise ValueError("shared checkpoint observation schema differs")
        policy.load_state_dict(copy.deepcopy(shared.state_dict()))
        shared_hash = policy_hash(shared)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initial_hash = policy_hash(policy)
    state: dict[str, Any] = {
        "gradient_steps": 0, "optimization_environment_steps": 0,
        "selection_environment_steps": 0, "episode_index": 0,
        "completed_episodes": 0, "discarded_partial_batches": 0,
        "elapsed_seconds": 0.0, "selection_seconds": 0.0,
        "initial_selection_return": None, "selected_selection_return": None,
        "initial_actor_hash": policy_hash(policy.actor),
        "selection_history": [], "optimization_history": [],
    }
    selected_weights = copy.deepcopy(policy.state_dict())
    selected_hash = initial_hash
    simulator_source = Path(inspect.getfile(type(simulator)))
    contract = {"schema_version": 1, "algorithm": "masked_reinforce_state_value_v2",
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "simulator_implementation_sha256": hashlib.sha256(simulator_source.read_bytes()).hexdigest(),
                "config": asdict(config), "partitions": partitions,
                "decision_model_hash": frozen_hash, "dimensions": list(dimensions)}
    contract_hash = _json_hash(contract)
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["contract_hash"] != contract_hash:
            raise ValueError("resume contract changed: model, worlds or optimization settings differ")
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
                           "policy": copy.deepcopy(policy.state_dict()), "policy_hash": initial_hash})
        _atomic_json(directory / "contract.json", {**contract, "contract_hash": contract_hash,
                     "initial_checkpoint_hash": initial_hash, "shared_checkpoint_hash": shared_hash,
                     "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                                  "processor": platform.processor(), "torch": str(torch.__version__),
                                  "device": "cpu", "torch_threads": torch.get_num_threads()},
                     "clinical_deficit_probability": None,
                     "final_evaluation_used_for_optimization": False})
    elapsed_before = state["elapsed_seconds"]
    cancelled = cancelled or (lambda: False)

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def save(status: str) -> TrainingResult:
        state["elapsed_seconds"] = elapsed()
        latest_hash = policy_hash(policy)
        _atomic_checkpoint(checkpoint_path, {
            "contract_hash": contract_hash, "dimensions": list(dimensions),
            "policy": policy.state_dict(), "policy_hash": latest_hash,
            "selected_policy": selected_weights, "selected_hash": selected_hash,
            "optimizer": optimizer.state_dict(), "random_state": generator.get_state(),
            "initial_hash": initial_hash, "shared_hash": shared_hash, "state": state,
        })
        result = TrainingResult(status, "POPULATION_ADAPTED" if shared_hash else "PATIENT_SCRATCH_RL",
                                initial_hash, selected_hash, latest_hash, frozen_hash,
                                state["gradient_steps"], state["optimization_environment_steps"],
                                state["selection_environment_steps"], state["elapsed_seconds"],
                                state["initial_selection_return"], state["selected_selection_return"],
                                str(directory), shared_hash)
        _atomic_json(directory / "result.json", {**asdict(result), **state,
                     "latest_actor_hash": policy_hash(policy.actor),
                     "actor_parameters_changed": policy_hash(policy.actor) != state["initial_actor_hash"],
                     "selection_rule": "maximum mean deterministic selection return; earliest wins ties",
                     "wall_limit_semantics": "cooperative; a simulator/gradient call may overrun the deadline",
                     "clinical_deficit_probability": None, "learning_improvement_guaranteed": False})
        return result

    def select() -> bool:
        nonlocal selected_weights, selected_hash
        rewards: list[float] = []
        selection_started = time.perf_counter()
        for seed in selection_manifest.seeds:
            if cancelled() or elapsed() >= config.max_wall_seconds:
                state["selection_seconds"] += time.perf_counter() - selection_started
                return False
            instance = simulator_factory()
            _assert_partition_binding(instance, selection_manifest)
            _assert_model(instance, frozen_hash)
            try:
                replay = rollout_policy(policy, instance, seed=seed, max_steps=config.max_episode_steps,
                                        expected_model_hash=frozen_hash,
                                        interrupt=lambda: cancelled() or elapsed() >= config.max_wall_seconds)
            except RolloutInterrupted as interrupted:
                state["selection_environment_steps"] += interrupted.environment_steps
                state["selection_seconds"] += time.perf_counter() - selection_started
                return False
            rewards.append(replay.total_reward)
            state["selection_environment_steps"] += replay.environment_steps
        state["selection_seconds"] += time.perf_counter() - selection_started
        mean = float(np.mean(rewards))
        state["selection_history"].append({"gradient_steps": state["gradient_steps"],
            "optimization_environment_steps": state["optimization_environment_steps"],
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
                logits, value = policy(observation)
                distribution = torch.distributions.Categorical(logits=logits)
                forced_stop = step == config.max_episode_steps - 1 or remaining == 1
                action = 0 if forced_stop else int(torch.multinomial(distribution.probs, 1, generator=generator))
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
