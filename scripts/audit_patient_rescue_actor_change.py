#!/usr/bin/env python3
"""Audit that PPO changed the inherited BC actor, not just its critic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
from stable_baselines3 import PPO
from patient_rescue_env import sha
from run_patient_rescue_ppo_bc import parameter_hash, write
from resectionlab.patient_rescue_learning import load_bc_checkpoint


def actor_hash(weight, bias):
    h = hashlib.sha256()
    h.update(b"action_net.weight")
    h.update(np.asarray(weight, dtype=np.float32).tobytes())
    h.update(b"action_net.bias")
    h.update(np.asarray(bias, dtype=np.float32).tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "testcases"):
        parser.error("Write a new supplement under ignored testcases")
    result = json.loads((args.run / "result.json").read_text())
    if result["status"] not in ("completed_trained_success", "completed_trained_failed_gate"):
        raise ValueError("Only a completed PPO pilot can receive an actor audit")
    if sha(args.bc_checkpoint) != result["bcCheckpointSha256"]:
        raise ValueError("Inherited BC checkpoint differs")
    bc, bc_hash = load_bc_checkpoint(args.bc_checkpoint,
        manifest_sha256=result["manifestSha256"])
    checkpoint = args.run / result["checkpoint"]["path"]
    if sha(checkpoint) != result["checkpoint"]["sha256"]:
        raise ValueError("PPO checkpoint differs")
    model = PPO.load(checkpoint, device="cpu")
    if parameter_hash(model) != result["checkpoint"]["policySha256"]:
        raise ValueError("PPO tensor hash differs")
    weight = model.policy.action_net.weight.detach().cpu().numpy()
    bias = model.policy.action_net.bias.detach().cpu().numpy()
    if weight.shape != bc.weight.shape or bias.shape != bc.bias.shape:
        raise ValueError("Actor shapes differ")
    delta = np.concatenate(((weight - bc.weight).ravel(), (bias - bc.bias).ravel()))
    actor_changed = bool(np.any(delta != 0.))
    if not actor_changed:
        raise ValueError("PPO actor did not change from the BC initialization")
    training = result["training"]
    train_episodes = [row for row in training["episodes"] if row["mode"] == "ppo-training"]
    other_episodes = [row for row in training["episodes"] if row["mode"] != "ppo-training"]
    accounted = sum(len(row["options"]) for row in train_episodes) + len(training["openEpisode"])
    if accounted != training["optionDecisions"] or accounted != 64:
        raise ValueError("Actual PPO episodes do not account for all 64 interactions")
    if model.num_timesteps != 64:
        raise ValueError("Saved PPO checkpoint timestep counter differs")
    receipt = dict(schema="resectionlab.patient-rescue-actor-change-audit.v1",
        status="passed", seed=result["seed"], bcCheckpointSha256=bc_hash,
        ppoCheckpointSha256=sha(checkpoint), inheritedBCActorSha256=actor_hash(bc.weight, bc.bias),
        finalPpoActorSha256=actor_hash(weight, bias), actorChanged=actor_changed,
        actorDeltaL2=float(np.linalg.norm(delta)), actorDeltaMaxAbs=float(np.max(np.abs(delta))),
        modelNumTimesteps=model.num_timesteps, modelUpdateEpochs=model._n_updates,
        observedOptimizerSteps=training.get("optimizerSteps"),
        trainingOptionDecisions=accounted, completeTrainingEpisodes=len(train_episodes),
        successfulCompleteTrainingEpisodes=sum(bool(row["success"]) for row in train_episodes),
        failedCompleteTrainingEpisodes=sum(not row["success"] for row in train_episodes),
        openTrainingEpisodeDecisions=len(training["openEpisode"]),
        nonTrainingEpisodesExcluded=[dict(mode=row["mode"], options=len(row["options"]))
                                    for row in other_episodes],
        limitation="Actor change and same-injury success do not establish improvement over the already successful BC actor.")
    write(args.output, receipt)
    print(json.dumps({k:receipt[k] for k in ("status", "seed", "actorDeltaL2",
        "actorDeltaMaxAbs", "trainingOptionDecisions")}))


if __name__ == "__main__":
    main()
