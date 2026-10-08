"""Separate masked clipped-PPO baseline for patient simulation experiments.

The clipped objective and GAE follow Schulman et al. (2017), equations 7 and
11–12: https://arxiv.org/pdf/1707.06347 . Action masks are stored with rollout
observations and reused for both sides of the likelihood ratio. The actor and
categorical API are shared with REINFORCE; algorithm and artifacts remain distinct.

Every Adam minibatch step consumes one gradient-step budget unit. Repeated epochs
are not free. No final-evaluation manifest or evaluator enters this module.
"""
from __future__ import annotations

from .data_policy import historical_only

import copy
import hashlib
import inspect
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import learning
from .learning import (MaskedPatientPolicy, Observation, RolloutInterrupted,
                       Simulator, TrainingConfig, TrainingResult, policy_hash,
                       rollout_policy)


@dataclass(frozen=True)
class PPOConfig(TrainingConfig):
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    epochs_per_batch: int = 4
    minibatch_size: int = 64
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.clip_epsilon) or not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if not math.isfinite(self.gae_lambda) or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        for field in ("epochs_per_batch", "minibatch_size"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("normalize_advantages must be boolean")


def clipped_surrogate(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                      advantage: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Per-transition PPO objective; old likelihoods and advantages are constants."""
    if not 0 < epsilon < 1:
        raise ValueError("clipping epsilon must be in (0, 1)")
    if new_log_prob.shape != old_log_prob.shape or new_log_prob.shape != advantage.shape:
        raise ValueError("PPO likelihood/advantage shapes differ")
    if not all(torch.isfinite(tensor).all() for tensor in (new_log_prob, old_log_prob, advantage)):
        raise ValueError("PPO likelihoods and advantages must be finite")
    ratio = torch.exp(new_log_prob - old_log_prob.detach())
    fixed_advantage = advantage.detach()
    return torch.minimum(ratio * fixed_advantage,
                         ratio.clamp(1 - epsilon, 1 + epsilon) * fixed_advantage)


def generalized_advantages(rewards: Sequence[float], values: Sequence[float],
                           terminated: Sequence[bool], *, gamma: float = 1.0,
                           gae_lambda: float = .95,
                           bootstrap_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """GAE with explicit terminal boundaries; STOP never bootstraps a future state."""
    rewards_array, values_array = np.asarray(rewards, float), np.asarray(values, float)
    terminal_array = np.asarray(terminated, bool)
    if (rewards_array.ndim != 1 or not len(rewards_array) or values_array.shape != rewards_array.shape
            or terminal_array.shape != rewards_array.shape):
        raise ValueError("GAE requires equally sized nonempty vectors")
    if (not np.isfinite(rewards_array).all() or not np.isfinite(values_array).all()
            or not math.isfinite(bootstrap_value) or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1):
        raise ValueError("invalid GAE inputs")
    advantage = np.zeros_like(rewards_array)
    future_advantage = 0.0
    for index in range(len(rewards_array) - 1, -1, -1):
        continuation = 0.0 if terminal_array[index] else 1.0
        next_value = values_array[index + 1] if index + 1 < len(values_array) else bootstrap_value
        delta = rewards_array[index] + gamma * continuation * next_value - values_array[index]
        future_advantage = delta + gamma * gae_lambda * continuation * future_advantage
        advantage[index] = future_advantage
    return advantage.astype(np.float32), (advantage + values_array).astype(np.float32)


def _immutable_array(value: Any, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class FrozenObservation:
    action_features: np.ndarray
    state_features: np.ndarray
    action_mask: np.ndarray
    action_ids: tuple[str, ...]

    @classmethod
    def snapshot(cls, observation: Observation) -> FrozenObservation:
        return cls(_immutable_array(observation.action_features, np.float32),
                   _immutable_array(observation.state_features, np.float32),
                   _immutable_array(observation.action_mask, bool), tuple(observation.action_ids))

    def pack(self) -> dict[str, Any]:
        return {"action_features": self.action_features.tolist(), "state_features": self.state_features.tolist(),
                "action_mask": self.action_mask.tolist(), "action_ids": list(self.action_ids)}

    @classmethod
    def unpack(cls, value: dict[str, Any]) -> FrozenObservation:
        return cls(_immutable_array(value["action_features"], np.float32),
                   _immutable_array(value["state_features"], np.float32),
                   _immutable_array(value["action_mask"], bool), tuple(value["action_ids"]))


@dataclass(frozen=True)
class PPOBatch:
    observations: tuple[FrozenObservation, ...]
    actions: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]
    actor_valid: tuple[bool, ...]
    old_policy_hash: str
    episode_returns: tuple[float, ...]

    @property
    def fingerprint(self) -> str:
        return learning._json_hash(self.pack())

    def pack(self) -> dict[str, Any]:
        return {"observations": [observation.pack() for observation in self.observations],
                "actions": list(self.actions), "old_log_probs": list(self.old_log_probs),
                "advantages": list(self.advantages), "returns": list(self.returns),
                "actor_valid": list(self.actor_valid), "old_policy_hash": self.old_policy_hash,
                "episode_returns": list(self.episode_returns)}

    @classmethod
    def unpack(cls, value: dict[str, Any]) -> PPOBatch:
        return cls(tuple(FrozenObservation.unpack(row) for row in value["observations"]),
                   *(tuple(value[key]) for key in ("actions", "old_log_probs", "advantages", "returns", "actor_valid")),
                   value["old_policy_hash"], tuple(value["episode_returns"]))


def ppo_minibatch_loss(policy: MaskedPatientPolicy, batch: PPOBatch,
                       indices: Sequence[int], config: PPOConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate the *stored* proposal sets, never a recomputed or later action mask."""
    if not indices:
        raise ValueError("empty PPO minibatch")
    new_log_probs, values, entropies = [], [], []
    for index in indices:
        observation, action = batch.observations[index], batch.actions[index]
        if not observation.action_mask[action]:
            raise ValueError("rollout contains a masked action")
        logits, value = policy(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probs.append(distribution.log_prob(torch.tensor(action)))
        values.append(value)
        entropies.append(distribution.entropy())
    new = torch.stack(new_log_probs)
    old = torch.tensor([batch.old_log_probs[index] for index in indices])
    advantages = torch.tensor([batch.advantages[index] for index in indices])
    actor_valid = torch.tensor([batch.actor_valid[index] for index in indices], dtype=torch.bool)
    targets = torch.tensor([batch.returns[index] for index in indices])
    value_loss = (torch.stack(values) - targets).square().mean()
    if bool(actor_valid.any()):
        objective = clipped_surrogate(new[actor_valid], old[actor_valid], advantages[actor_valid], config.clip_epsilon).mean()
        entropy = torch.stack(entropies)[actor_valid].mean()
        with torch.no_grad():
            log_ratio = new[actor_valid] - old[actor_valid]
            ratio = log_ratio.exp()
            approximate_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > config.clip_epsilon).float().mean()
    else:
        objective = new.sum() * 0
        entropy = objective
        approximate_kl = torch.tensor(0.0)
        clip_fraction = torch.tensor(0.0)
    loss = -objective + config.value_weight * value_loss - config.entropy_weight * entropy
    return loss, {"policy_objective": float(objective.detach()), "value_loss": float(value_loss.detach()),
                  "entropy": float(entropy.detach()), "approximate_kl": float(approximate_kl),
                  "clip_fraction": float(clip_fraction), "actor_sample_count": int(actor_valid.sum())}


@historical_only("RECORDED_EXPERIENCE_REQUIRED")
@learning._record_failures
def train_patient_ppo(simulator_factory: Callable[[], Simulator], optimization_manifest: Any,
                      selection_manifest: Any, *, config: PPOConfig = PPOConfig(),
                      output_dir: str | Path, cancelled: Callable[[], bool] | None = None,
                      progress: Callable[[dict[str, Any]], None] | None = None,
                      resume: bool = False) -> TrainingResult:
    """Scratch PPO with bounded Adam steps, frozen rollouts and resumable epochs.

    Cancelled partial collection is discarded but counted. Cancellation during
    minibatch reuse saves the exact old rollout, shuffle and epoch cursor. Resume
    continues those epochs; it cannot resample them at no resource cost.
    """
    started = time.perf_counter()
    partitions = learning._validate_partitions(optimization_manifest, selection_manifest)
    if not isinstance(config, PPOConfig):
        raise TypeError("PPOConfig is required to declare clipping and epoch budgets")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() != resume:
        raise ValueError("use a new run directory or resume an existing PPO checkpoint")
    base = simulator_factory()
    learning._assert_partition_binding(base, optimization_manifest)
    frozen_hash = base.decision_model_hash
    first = base.reset(optimization_manifest.seeds[0])
    dimensions = (np.asarray(first.action_features).shape[1], np.asarray(first.state_features).size, config.hidden_features)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = MaskedPatientPolicy(*dimensions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    initial_hash = policy_hash(policy)
    selected_weights = copy.deepcopy(policy.state_dict())
    selected_hash = initial_hash
    state: dict[str, Any] = {"gradient_steps": 0, "optimization_environment_steps": 0,
        "selection_environment_steps": 0, "rollout_batches": 0, "episode_index": 0,
        "completed_episodes": 0, "discarded_partial_batches": 0, "elapsed_seconds": 0.0,
        "selection_seconds": 0.0, "optimization_sample_presentations": 0,
        "initial_selection_return": None, "selected_selection_return": None,
        "initial_actor_hash": policy_hash(policy.actor), "optimization_history": [],
        "selection_history": [], "rollout_history": []}
    pending: dict[str, Any] | None = None
    batch: PPOBatch | None = None
    contract = {"algorithm": "masked_clipped_ppo_gae_v1", "config": asdict(config),
                "decision_model_hash": frozen_hash, "partitions": partitions, "dimensions": list(dimensions),
                "numerical_source_sha256": learning.numerical_source_hashes(),
                "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                            "python": platform.python_version(), "device": "cpu"},
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                    (("ppo", Path(__file__)), ("shared_learning", Path(learning.__file__)),
                     ("simulator", Path(inspect.getfile(type(base)))))},
                "gradient_budget_definition": "each Adam minibatch step including every epoch",
                "clinical_deficit_probability": None, "final_evaluation_used": False}
    contract_hash = learning._json_hash(contract)
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["contract_hash"] != contract_hash:
            raise ValueError("PPO resume contract changed")
        policy.load_state_dict(saved["policy"])
        if policy_hash(policy) != saved["policy_hash"] or policy_hash(saved["selected_policy"]) != saved["selected_hash"]:
            raise ValueError("PPO checkpoint weight hash mismatch")
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["random_state"])
        state, initial_hash = saved["state"], saved["initial_hash"]
        selected_weights, selected_hash, pending = saved["selected_policy"], saved["selected_hash"], saved["pending"]
        if pending is not None:
            batch = PPOBatch.unpack(pending["batch"])
            if batch.fingerprint != pending["batch_hash"]:
                raise ValueError("saved old-policy PPO rollout changed")
    else:
        learning._atomic_checkpoint(directory / "initial.pt", {"dimensions": list(dimensions),
                                     "policy": policy.state_dict(), "policy_hash": initial_hash})
        learning._atomic_json(directory / "contract.json", {**contract, "contract_hash": contract_hash,
            "initial_checkpoint_hash": initial_hash, "hardware": {"platform": platform.platform(),
            "torch": str(torch.__version__), "device": "cpu", "torch_threads": torch.get_num_threads()}})
    elapsed_before = state["elapsed_seconds"]
    cancelled = cancelled or (lambda: False)

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def interrupted() -> bool:
        return cancelled() or elapsed() >= config.max_wall_seconds

    def save(status: str) -> TrainingResult:
        state["elapsed_seconds"] = elapsed()
        latest_hash = policy_hash(policy)
        learning._atomic_checkpoint(checkpoint, {"contract_hash": contract_hash, "dimensions": list(dimensions),
            "policy": policy.state_dict(), "policy_hash": latest_hash, "selected_policy": selected_weights,
            "selected_hash": selected_hash, "optimizer": optimizer.state_dict(), "random_state": generator.get_state(),
            "state": state, "pending": pending, "initial_hash": initial_hash})
        result = TrainingResult(status, "PATIENT_SCRATCH_RL", initial_hash, selected_hash, latest_hash, frozen_hash,
            state["gradient_steps"], state["optimization_environment_steps"], state["selection_environment_steps"],
            state["elapsed_seconds"], state["initial_selection_return"], state["selected_selection_return"], str(directory))
        learning._atomic_json(directory / "result.json", {**asdict(result), **state, "algorithm": contract["algorithm"],
            "actor_parameters_changed": policy_hash(policy.actor) != state["initial_actor_hash"],
            "latest_actor_hash": policy_hash(policy.actor), "clinical_deficit_probability": None,
            "pending_old_policy_batch": pending is not None, "final_evaluation_used": False,
            "selection_rule": "maximum complete-panel mean deterministic selection return; earliest tie",
            "wall_limit_semantics": "cooperative; one simulator or optimizer call may overrun"})
        return result

    def select() -> bool:
        nonlocal selected_weights, selected_hash
        began = time.perf_counter()
        rewards = []
        for seed in selection_manifest.seeds:
            if interrupted():
                state["selection_seconds"] += time.perf_counter() - began
                return False
            simulator = simulator_factory()
            learning._assert_partition_binding(simulator, selection_manifest)
            try:
                trajectory = rollout_policy(policy, simulator, seed=seed, max_steps=config.max_episode_steps,
                                            expected_model_hash=frozen_hash, interrupt=interrupted)
            except RolloutInterrupted as partial:
                state["selection_environment_steps"] += partial.environment_steps
                state["selection_seconds"] += time.perf_counter() - began
                return False
            state["selection_environment_steps"] += trajectory.environment_steps
            rewards.append(trajectory.total_reward)
        state["selection_seconds"] += time.perf_counter() - began
        mean = float(np.mean(rewards))
        if state["initial_selection_return"] is None:
            state["initial_selection_return"] = mean
        if state["selected_selection_return"] is None or mean > state["selected_selection_return"]:
            selected_weights, selected_hash = copy.deepcopy(policy.state_dict()), policy_hash(policy)
            state["selected_selection_return"] = mean
        state["selection_history"].append({"gradient_steps": state["gradient_steps"],
            "optimization_environment_steps": state["optimization_environment_steps"],
            "mean_return": mean, "world_count": len(rewards), "checkpoint_hash": policy_hash(policy)})
        return True

    def collect() -> PPOBatch | None:
        old_hash = policy_hash(policy)
        observations, actions, old_probs, advantages, targets, actor_valid, episode_returns = [], [], [], [], [], [], []
        for _ in range(config.episodes_per_update):
            if state["optimization_environment_steps"] >= config.max_environment_steps:
                break
            seed = optimization_manifest.seeds[state["episode_index"] % len(optimization_manifest.seeds)]
            state["episode_index"] += 1
            simulator = simulator_factory()
            learning._assert_partition_binding(simulator, optimization_manifest)
            learning._assert_model(simulator, frozen_hash)
            observation = simulator.reset(seed)
            episode = []
            complete = False
            for step in range(config.max_episode_steps):
                if interrupted():
                    break
                remaining = config.max_environment_steps - state["optimization_environment_steps"]
                if remaining <= 0:
                    break
                snapshot = FrozenObservation.snapshot(observation)
                with torch.no_grad():
                    logits, value = policy(snapshot)
                    distribution = torch.distributions.Categorical(logits=logits)
                    forced = step == config.max_episode_steps - 1 or remaining == 1
                    action = 0 if forced else int(torch.multinomial(distribution.probs, 1, generator=generator))
                    old_probability = float(distribution.log_prob(torch.tensor(action)))
                result = simulator.step(action)
                state["optimization_environment_steps"] += 1
                learning._assert_model(simulator, frozen_hash)
                if not math.isfinite(float(result.reward)):
                    raise ValueError("nonfinite PPO rollout reward")
                episode.append((snapshot, action, old_probability, float(value), float(result.reward), bool(result.terminated), not forced))
                observation = result.observation
                if result.terminated:
                    complete = True
                    break
            if not complete:
                state["discarded_partial_batches"] += 1
                return None
            advantage, returns = generalized_advantages([row[4] for row in episode], [row[3] for row in episode],
                [row[5] for row in episode], gamma=config.gamma, gae_lambda=config.gae_lambda)
            state["completed_episodes"] += 1
            episode_returns.append(sum(row[4] for row in episode))
            observations.extend(row[0] for row in episode)
            actions.extend(row[1] for row in episode)
            old_probs.extend(row[2] for row in episode)
            advantages.extend(float(value) for value in advantage)
            targets.extend(float(value) for value in returns)
            actor_valid.extend(row[6] for row in episode)
        if not observations:
            return None
        if policy_hash(policy) != old_hash:
            raise ValueError("old policy changed during PPO rollout collection")
        if config.normalize_advantages and sum(actor_valid) > 1:
            values = np.asarray(advantages, float)
            mask = np.asarray(actor_valid, bool)
            standard_deviation = float(values[mask].std())
            values[mask] = (values[mask] - values[mask].mean()) / max(standard_deviation, 1e-8)
            advantages = values.tolist()
        return PPOBatch(tuple(observations), tuple(actions), tuple(old_probs), tuple(advantages),
                        tuple(targets), tuple(actor_valid), old_hash, tuple(episode_returns))

    if state["initial_selection_return"] is None:
        if not select():
            return save("cancelled" if cancelled() else "wall_time_budget")
        save("running")
    status = "gradient_budget"
    while state["gradient_steps"] < config.max_gradient_steps:
        if interrupted():
            status = "cancelled" if cancelled() else "wall_time_budget"
            break
        if pending is None:
            if state["optimization_environment_steps"] >= config.max_environment_steps:
                status = "environment_budget"
                break
            batch = collect()
            if batch is None:
                status = "cancelled" if cancelled() else "wall_time_budget" if interrupted() else "environment_budget"
                break
            state["rollout_batches"] += 1
            pending = {"batch": batch.pack(), "batch_hash": batch.fingerprint, "epoch": 0, "permutation": [], "offset": 0}
            state["rollout_history"].append({"rollout_batch": state["rollout_batches"],
                "old_policy_hash": batch.old_policy_hash, "batch_hash": batch.fingerprint,
                "sample_count": len(batch.actions), "episode_returns": list(batch.episode_returns),
                "gradient_steps_before_reuse": state["gradient_steps"]})
        assert batch is not None
        if not pending["permutation"]:
            pending["permutation"] = torch.randperm(len(batch.actions), generator=generator).tolist()
        indices = pending["permutation"][pending["offset"]:pending["offset"] + config.minibatch_size]
        loss, diagnostics = ppo_minibatch_loss(policy, batch, indices, config)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PPO loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite PPO gradient")
        actor_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in policy.actor.parameters() if p.grad is not None))
        optimizer.step()
        state["gradient_steps"] += 1
        state["optimization_sample_presentations"] += len(indices)
        state["optimization_history"].append({"gradient_steps": state["gradient_steps"],
            "rollout_batch": state["rollout_batches"], "epoch": pending["epoch"],
            "minibatch_size": len(indices), "old_policy_hash": batch.old_policy_hash,
            "frozen_batch_hash": pending["batch_hash"], "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm), "actor_gradient_norm_after_clip": actor_norm, **diagnostics})
        pending["offset"] += len(indices)
        if pending["offset"] == len(batch.actions):
            pending.update(offset=0, permutation=[], epoch=pending["epoch"] + 1)
            if pending["epoch"] == config.epochs_per_batch:
                pending = None
                batch = None
        if state["gradient_steps"] % config.checkpoint_interval == 0:
            if not select():
                status = "cancelled" if cancelled() else "wall_time_budget"
                break
            snapshot = save("running")
            if progress:
                progress(asdict(snapshot))
    if status not in ("cancelled", "wall_time_budget") and state["selection_history"][-1]["gradient_steps"] != state["gradient_steps"]:
        if not select():
            status = "cancelled" if cancelled() else "wall_time_budget"
    return save(status)
