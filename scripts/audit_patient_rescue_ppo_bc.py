#!/usr/bin/env python3
"""Cold checkpoint, actor-choice, public-action and native replay audit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from stable_baselines3 import PPO
from patient_rescue_env import from_manifest, sha
from run_patient_rescue_ppo_bc import parameter_hash, write
from resectionlab.patient_rescue_learning import OPTIONS, publish_observation
from resectionlab.workspace_bundle import load_workspace, save_workspace


def export(manifest, recorded, run, output, report):
    source = Path(manifest["nativeWorkspace"])
    if sha(source) != manifest["nativeWorkspaceSha256"]:
        raise ValueError("Original patient workspace changed")
    case, artifacts, workspace = load_workspace(source)
    record = artifacts["surgical_toolbox"]
    original = copy.deepcopy(record["branches"])
    branch = f"rescue-ppo-from-bc-live-seed-{report['seed']}"
    if branch in record["branches"]:
        raise ValueError("Audit branch already exists")
    record["branches"][branch] = recorded["native"]
    record.setdefault("parents", {})[branch] = dict(branchId="catalog-rescue-task-start",
        position=len(record["branches"]["catalog-rescue-task-start"]["actions"]))
    record["branchId"] = branch
    record["position"] = len(recorded["native"]["actions"])
    record.pop("timeline", None)
    artifacts["patient_rescue_ppo_bc"] = dict(schema="resectionlab.patient-rescue-ppo-bc-audit.v1",
        status="checkpoint_verified_success", method="ppo_initialized_from_behavior_cloning",
        branchId=branch, seed=report["seed"], samePreparedInjuryOnly=True,
        noGeneralizationClaim=True, noIncrementalRLBenefitClaim=True, promoted=False,
        sourceRun=str(run.resolve()), checkpointSha256=report["checkpoint"]["sha256"],
        initialBCCheckpointSha256=report["bcCheckpointSha256"],
        observationSchema="resectionlab.prepared-rescue-observation.v2",
        actualPpoOptionTransitions=report["training"]["optionDecisions"],
        actualPpoUpdateEpochs=report["training"].get("ppoUpdateEpochs"),
        actualOptimizerSteps=report["training"].get("optimizerSteps"),
        auditPassed=True,
        actualActorChoices=[row["option"] for row in recorded["decisions"]],
        taskManifestContentHash=manifest["contentHash"],
        limitations=["One fixed prepared injury, not held-out glioma resection or clinical validation.",
                     "BC initialization already succeeded; no incremental RL advantage established."])
    destination = output / "patient-rescue-ppo-live.ressectionlab"
    save_workspace(case, destination, sources=workspace["sources"],
        imaging=workspace["imagingState"], episode=workspace["episode"],
        selection=workspace["selection"], originals=workspace["originals"], artifacts=artifacts)
    reopened, saved, _ = load_workspace(destination)
    if reopened.semantic_hash != case.semantic_hash or saved["surgical_toolbox"]["branches"] != record["branches"]:
        raise ValueError("Learned policy workspace save/reopen differs")
    if any(record["branches"][name] != snapshot for name, snapshot in original.items()):
        raise ValueError("Prior patient branches changed")
    return dict(path=str(destination), sha256=sha(destination), branch=branch,
        sourceAndAllPriorBranchesExact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "testcases") or output.exists():
        parser.error("Use a new ignored testcases output directory")
    output.mkdir(parents=True)
    start = time.perf_counter()
    def check():
        if time.perf_counter() - start > 300.:
            raise TimeoutError("Independent PPO replay exceeded five-minute audit budget")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > 3 * 1024 ** 3:
            raise MemoryError("Independent PPO replay exceeded 3 GiB")
    audit = dict(schema="resectionlab.patient-rescue-ppo-bc-independent-audit.v1",
        status="running", run=str(args.run.resolve()), manifestSha256=sha(args.manifest))
    try:
        report = json.loads((args.run / "result.json").read_text())
        recorded = json.loads((args.run / "trained.json").read_text())
        checkpoint = args.run / report["checkpoint"]["path"]
        if (report["status"] != "completed_trained_success" or not recorded["success"]
                or report["manifestSha256"] != audit["manifestSha256"]
                or sha(checkpoint) != report["checkpoint"]["sha256"]):
            raise ValueError("Successful source-bound checkpoint receipt required")
        model = PPO.load(checkpoint, device="cpu")
        model_hash = parameter_hash(model)
        if model_hash != report["checkpoint"]["policySha256"]:
            raise ValueError("Loaded checkpoint policy tensors differ")
        task, manifest = from_manifest(args.manifest, check=check)
        task.reset(seed=report["seed"])
        proof = []
        for saved in recorded["decisions"]:
            check()
            observation, published = publish_observation(task)
            if published["featuresSha256"] != saved["published"]["featuresSha256"]:
                raise ValueError("Checkpoint policy input differs")
            action = int(model.predict(observation, deterministic=True)[0])
            if OPTIONS[action] != saved["option"]:
                raise ValueError("Loaded actor option differs")
            _, reward, terminated, truncated, info = task.step(action)
            if reward != saved["reward"] or task.raw.engine.state_id != saved["stateId"]:
                raise ValueError("Public action reward/state differs")
            proof.append(dict(option=OPTIONS[action], observationSha256=published["featuresSha256"],
                              reward=reward, stateId=task.raw.engine.state_id))
        if (task.native_snapshot() != recorded["native"] or task.raw.attempts != recorded["attempts"]
                or task.raw.total_reward != recorded["returnValue"]
                or info["termination_reason"] != recorded["reason"]
                or not info["success"] or parameter_hash(model) != model_hash):
            raise ValueError("Final native/reward/terminal/checkpoint replay differs")
        audit.update(checkpointSha256=sha(checkpoint), loadedPolicySha256=model_hash,
            sourceSceneId=manifest["sourceSceneId"], taskStartStateId=manifest["taskStartStateId"],
            actorPredictionsExact=True, observationsExact=True, nativeAndRewardExact=True,
            terminalReasonExact=True, unchangedFrozenPolicy=True,
            optionDecisions=len(proof), rlTransitionAttempts=len(task.raw.attempts),
            publicEngineCalls=task.raw.engine.public_calls-task.raw._prepared.public_calls,
            proof=proof)
        audit["workspace"] = export(manifest, recorded, args.run, output, report)
        audit["status"] = "passed"
    except BaseException as error:
        audit.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        audit["wallSeconds"] = time.perf_counter() - start
        write(output / "audit.json", audit)
        print(json.dumps({k:audit[k] for k in ("status", "wallSeconds", "error") if k in audit}), flush=True)


if __name__ == "__main__":
    main()
