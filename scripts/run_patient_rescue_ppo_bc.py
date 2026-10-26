#!/usr/bin/env python3
"""One capped PPO pilot initialized from a verified frozen rescue BC actor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO

from patient_rescue_env import from_manifest, sha
from resectionlab.patient_rescue_learning import (OPTIONS, FEATURE_NAMES,
    load_bc_checkpoint, publish_observation)

MAX_WALL_SECONDS = 900.
MAX_OPTION_DECISIONS = 110
MAX_RSS_BYTES = 3 * 1024 ** 3
TRAIN_DECISIONS = 64
SCRIPT = ("coagulate_center", "coagulate_lower", "coagulate_forward",
          "aspirate_pool", "finish_verify")


def write(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class ObservedRescueEnv(gym.Wrapper):
    def __init__(self, task, check):
        super().__init__(task)
        self.check = check
        self.observation_space = spaces.Box(-1e6, 1e6, (len(FEATURE_NAMES),), np.float32)
        self.total_decisions = 0
        self.training_episodes = []
        self.mode = "other"
        self._episode = []

    def reset(self, **kwargs):
        if self._episode:
            self.training_episodes.append(dict(mode=self.mode, options=self._episode,
                reason="external_reset", success=False))
        self._episode = []
        _, info = self.env.reset(**kwargs)
        observation, self._published = publish_observation(self.env)
        return observation, info

    def step(self, action):
        self.check()
        before = self._published
        _, reward, terminated, truncated, info = self.env.step(int(action))
        self.total_decisions += 1
        self._episode.append(dict(option=OPTIONS[int(action)], observationSha256=before["featuresSha256"],
                                  reward=reward, stateId=self.env.raw.engine.state_id))
        observation, self._published = publish_observation(self.env)
        if terminated or truncated:
            self.training_episodes.append(dict(mode=self.mode, options=self._episode,
                reason=info["termination_reason"], success=info["success"],
                returnValue=self.env.raw.total_reward))
            self._episode = []
        return observation, reward, terminated, truncated, info


def rollout(env, chooser, *, seed, label):
    observation, _ = env.reset(seed=seed)
    began = time.perf_counter()
    steps = []
    for index in range(env.env.max_decisions):
        env.check()
        action, scores = chooser(observation, index)
        published = copy.deepcopy(env._published)
        observation, reward, terminated, truncated, info = env.step(action)
        steps.append(dict(option=OPTIONS[action], published=published, scores=scores,
                          reward=reward, stateId=env.env.raw.engine.state_id))
        if terminated or truncated:
            break
    return dict(label=label, success=info["success"], reason=info["termination_reason"],
        returnValue=env.env.raw.total_reward, decisions=steps,
        optionDecisions=len(steps), rlTransitionAttempts=len(env.env.raw.attempts),
        native=env.env.native_snapshot(), attempts=copy.deepcopy(env.env.raw.attempts),
        modeledField=env.env.raw.field().tolist(), elapsedSeconds=time.perf_counter() - began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "testcases") or output.exists():
        parser.error("Use a new ignored testcases output directory")
    output.mkdir(parents=True)
    began = time.perf_counter()
    env = None
    model = None
    def check():
        if time.perf_counter() - began > MAX_WALL_SECONDS:
            raise TimeoutError("BC-initialized PPO pilot exceeded its 15-minute wall budget")
        if env is not None and env.total_decisions >= MAX_OPTION_DECISIONS:
            raise RuntimeError("BC-initialized PPO pilot exceeded option transition budget")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > MAX_RSS_BYTES:
            raise MemoryError("BC-initialized PPO pilot exceeded 3 GiB")
    report = dict(schema="resectionlab.patient-rescue-ppo-from-bc.v1", status="running",
        seed=args.seed, manifestSha256=sha(args.manifest), bcCheckpointSha256=sha(args.bc_checkpoint),
        trainingAllowance=TRAIN_DECISIONS, totalDecisionAllowance=MAX_OPTION_DECISIONS,
        limitations=["One authored injury on a development meningioma patient; no glioma resection or transfer.",
                     "Prepared history is privileged task setup, not raw-RL training experience.",
                     "This is one pilot seed and no checkpoint selection sweep."])
    report["codeSha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (
        Path(__file__), ROOT / "scripts/patient_rescue_env.py",
        ROOT / "src/resectionlab/patient_rescue_learning.py")}
    try:
        bc, bc_hash = load_bc_checkpoint(args.bc_checkpoint,
            manifest_sha256=report["manifestSha256"])
        task, manifest = from_manifest(args.manifest, check=check)
        report["preparation"] = task.preparation
        env = ObservedRescueEnv(task, check)
        torch.set_num_threads(1)
        model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=4,
            learning_rate=.001, gamma=.97, ent_coef=.02,
            policy_kwargs=dict(net_arch=dict(pi=[], vf=[32])),
            seed=args.seed, device="cpu", verbose=0)
        initial_hash = parameter_hash(model)
        env.mode = "untrained"
        initial = rollout(env, lambda obs, index: (int(model.predict(obs, deterministic=True)[0]), []),
                          seed=args.seed, label="untrained-v2")
        write(output / "untrained.json", initial)
        report["untrained"] = {key: initial[key] for key in ("success", "reason", "returnValue",
            "optionDecisions", "modeledField", "elapsedSeconds")}
        report["untrained"]["options"] = [r["option"] for r in initial["decisions"]]
        env.mode = "scripted"
        scripted = rollout(env, lambda obs, index: (OPTIONS.index(SCRIPT[index]), []),
                           seed=args.seed, label="scripted-v2")
        write(output / "scripted.json", scripted)
        report["scripted"] = {key: scripted[key] for key in ("success", "reason", "returnValue",
            "optionDecisions", "modeledField", "elapsedSeconds")}
        report["scripted"]["options"] = [r["option"] for r in scripted["decisions"]]
        if not scripted["success"]:
            raise ValueError("Scripted v2 same-task baseline no longer succeeds")
        if tuple(model.policy.action_net.weight.shape) != bc.weight.shape:
            raise ValueError("BC actor does not match PPO action head")
        with torch.no_grad():
            model.policy.action_net.weight.copy_(torch.as_tensor(bc.weight))
            model.policy.action_net.bias.copy_(torch.as_tensor(bc.bias))
        inherited_hash = parameter_hash(model)
        if initial_hash == inherited_hash:
            raise ValueError("BC actor did not change the PPO initialization")
        bc_live_path = args.bc_checkpoint.parent / "bc-live.json"
        bc_live = json.loads(bc_live_path.read_text())
        for row in bc_live["observations"]:
            vector = np.asarray(row["observation"], dtype=np.float32)
            inherited = int(model.predict(vector, deterministic=True)[0])
            if inherited != bc.predict(vector)[0] or OPTIONS[inherited] != row["option"]:
                raise ValueError("PPO actor initialization differs from the verified BC actor")
        report["initialization"] = dict(randomPolicySha256=initial_hash,
            bcInitializedPolicySha256=inherited_hash, bcCheckpointSha256=bc_hash,
            actorWeightsExact=True, verifiedBCDecisions=len(bc_live["observations"]), criticRandom=True)
        env.mode = "ppo-training"
        before = env.total_decisions
        training_start = time.perf_counter()
        optimizer_step = model.policy.optimizer.step
        optimizer_steps = 0
        def counted_step(*args, **kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            return optimizer_step(*args, **kwargs)
        model.policy.optimizer.step = counted_step
        try:
            model.learn(total_timesteps=TRAIN_DECISIONS)
        finally:
            model.policy.optimizer.step = optimizer_step
        report["training"] = dict(optionDecisions=env.total_decisions - before,
            elapsedSeconds=time.perf_counter() - training_start,
            optimizerSteps=optimizer_steps, ppoUpdateEpochs=model._n_updates,
            finalPolicySha256=parameter_hash(model),
            episodes=copy.deepcopy(env.training_episodes),
            openEpisode=copy.deepcopy(env._episode))
        if report["training"]["optionDecisions"] != TRAIN_DECISIONS:
            raise ValueError("PPO optimization interactions differ from declared budget")
        checkpoint = output / f"ppo-from-bc-seed-{args.seed}.zip"
        model.save(checkpoint)
        report["checkpoint"] = dict(path=checkpoint.name, sha256=sha(checkpoint),
            policySha256=parameter_hash(model))
        env.mode = "trained"
        trained = rollout(env, lambda obs, index: (int(model.predict(obs, deterministic=True)[0]), []),
                          seed=args.seed, label="trained-v2")
        write(output / "trained.json", trained)
        report["trained"] = {key: trained[key] for key in ("success", "reason", "returnValue",
            "optionDecisions", "modeledField", "elapsedSeconds")}
        report["trained"]["options"] = [r["option"] for r in trained["decisions"]]
        report["status"] = "completed_trained_success" if trained["success"] else "completed_trained_failed_gate"
    except BaseException as error:
        report.update(status="failed_run", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report["wallSeconds"] = time.perf_counter() - began
        report["expensiveOptionDecisions"] = env.total_decisions if env else 0
        report["peakResidentBytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
        if model is not None and not report.get("checkpoint"):
            partial = output / "ppo-partial.zip"
            model.save(partial)
            report["partialCheckpoint"] = dict(path=partial.name, sha256=sha(partial),
                policySha256=parameter_hash(model))
        write(output / "result.json", report)
        print(json.dumps({key: report[key] for key in ("status", "wallSeconds", "expensiveOptionDecisions", "error")
            if key in report}), flush=True)


if __name__ == "__main__":
    main()
