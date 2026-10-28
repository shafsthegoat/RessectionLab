#!/usr/bin/env python3
"""Fit one five-state rescue demonstrator and test its frozen live decisions.

The teacher profile is an actual public-action rollout on the declared patient
task. This diagnostic is same-state imitation, not patient transfer or RL.
"""
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

import numpy as np
import torch
from patient_rescue_env import from_manifest, sha
from resectionlab.patient_rescue_learning import (SCHEMA, OPTIONS, FEATURE_NAMES,
    FrozenRescueBCPolicy, features_from_sensors, publish_observation)
from resectionlab.workspace_bundle import load_workspace, save_workspace

TEACHER = OPTIONS[:4] + ("finish_verify",)
MAX_WALL_SECONDS = 900.
MAX_OPTION_DECISIONS = 128
MAX_RSS_BYTES = 3 * 1024 ** 3


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit(profile, seed):
    if profile.get("schema") != "rescue-bc-profile.v1":
        raise ValueError("Only the measured public teacher profile is supported")
    rows = profile["rows"]
    if len(rows) != 5 or tuple(row["option"] for row in rows) != TEACHER:
        raise ValueError("Full successful five-option teacher sequence required")
    if not rows[-1]["terminated"] or rows[-1]["reason"] != "rescue_cleanup_verified":
        raise ValueError("Teacher did not prove live cleanup")
    data = []
    for row in rows:
        if tuple(s["name"] for s in row["routeLocalBefore"]) != (
                "contact", "coverage_lower", "coverage_forward", "suction_pool"):
            raise ValueError("Teacher route sensor order differs")
        slots = np.asarray(row["featureBefore"][:24], dtype=float).reshape(2, 12)
        data.append(features_from_sensors(row["fieldBefore"],
            [s["fluidSum5"] for s in row["routeLocalBefore"]], slots))
    x = torch.as_tensor(np.stack(data), dtype=torch.float32)
    y = torch.as_tensor([OPTIONS.index(row["option"]) for row in rows], dtype=torch.long)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = torch.nn.Linear(len(FEATURE_NAMES), len(OPTIONS))
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    started = time.perf_counter()
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.cross_entropy(model(x), y)
    predictions = model(x).argmax(dim=1).tolist()
    if predictions != y.tolist() or any(torch.equal(initial[k], v) for k, v in model.state_dict().items()):
        raise ValueError("Supervised teacher reproduction gate failed")
    weight = model.weight.detach().numpy().copy()
    bias = model.bias.detach().numpy().copy()
    return FrozenRescueBCPolicy(weight, bias), FrozenRescueBCPolicy(
        initial["weight"].numpy().copy(), initial["bias"].numpy().copy()), dict(
        optimizerSteps=1000, loss=float(final.detach()), trainPredictions=[OPTIONS[i] for i in predictions],
        teacherLabels=list(TEACHER), trainingSeconds=time.perf_counter() - started,
        inputCount=len(rows), uniqueInputHashes=len({hashlib.sha256(a.tobytes()).hexdigest() for a in data}),
        featureNames=list(FEATURE_NAMES), seed=seed)


def rollout(task, chooser, *, seed, label, check):
    task.reset(seed=seed)
    trail = []
    began = time.perf_counter()
    for _ in range(task.max_decisions):
        check()
        vector, published = publish_observation(task)
        selected, scores = chooser(vector, len(trail))
        option = OPTIONS[selected]
        started = time.perf_counter()
        _, reward, terminated, truncated, info = task.step(selected)
        trail.append(dict(option=option, observation=vector.tolist(),
            published=published, scores=scores, reward=reward,
            stateId=task.raw.engine.state_id, seconds=time.perf_counter() - started,
            terminated=terminated, truncated=truncated, reason=info["termination_reason"]))
        if terminated or truncated:
            break
    return dict(label=label, seed=seed, success=info["success"],
        reason=info["termination_reason"], returnValue=task.raw.total_reward,
        modeledField=task.raw.field().tolist(), decisions=copy.deepcopy(task.decisions),
        observations=trail, attempts=copy.deepcopy(task.raw.attempts),
        native=task.native_snapshot(), elapsedSeconds=time.perf_counter() - began,
        optionDecisions=len(trail), rlTransitionAttempts=len(task.raw.attempts),
        publicEngineCalls=task.raw.engine.public_calls-task.raw._prepared.public_calls)


def export_workspace(manifest, run, output, fit):
    source = Path(manifest["nativeWorkspace"])
    if sha(source) != manifest["nativeWorkspaceSha256"]:
        raise ValueError("Source workspace changed")
    case, artifacts, workspace = load_workspace(source)
    record = artifacts["surgical_toolbox"]
    originals = copy.deepcopy(record["branches"])
    branch = "rescue-bc-live-seed-17"
    if branch in record["branches"]:
        raise ValueError("BC branch already exists")
    record["branches"][branch] = run["native"]
    record.setdefault("parents", {})[branch] = dict(branchId="catalog-rescue-task-start",
        position=len(record["branches"]["catalog-rescue-task-start"]["actions"]))
    record["branchId"] = branch
    record["position"] = len(run["native"]["actions"])
    record.pop("timeline", None)
    artifacts["patient_rescue_bc"] = dict(schema="resectionlab.patient-rescue-bc.v1",
        method="frozen_behavior_cloning", checkpointSha256=run["checkpointSha256"],
        observationSchema=SCHEMA, trajectory=str(output / "bc-live.json"),
        taskManifestContentHash=manifest["contentHash"], success=run["success"],
        branchId=branch, actualSupervisedOptimizerSteps=fit["optimizerSteps"],
        actualRlOptionTransitions=0, auditPassed=True, samePreparedInjuryOnly=True,
        noGeneralizationClaim=True, promoted=False,
        actualActorChoices=[row["option"] for row in run["observations"]])
    destination = output / "patient-rescue-bc-live.ressectionlab"
    save_workspace(case, destination, sources=workspace["sources"],
        imaging=workspace["imagingState"], episode=workspace["episode"],
        selection=workspace["selection"], originals=workspace["originals"], artifacts=artifacts)
    reopened, saved, _ = load_workspace(destination)
    if reopened.semantic_hash != case.semantic_hash or saved["surgical_toolbox"]["branches"] != record["branches"]:
        raise ValueError("BC case save/reopen differs")
    if any(record["branches"][name] != snapshot for name, snapshot in originals.items()):
        raise ValueError("Prior patient branches changed")
    return dict(path=str(destination), sha256=sha(destination), branch=branch,
        sourceAndAllPriorBranchesExact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "testcases") or output.exists():
        parser.error("Use a new ignored testcases output directory")
    output.mkdir(parents=True)
    began = time.perf_counter()
    decisions = 0
    def check():
        if time.perf_counter() - began > MAX_WALL_SECONDS:
            raise TimeoutError("Bounded BC experiment wall budget exhausted")
        if decisions >= MAX_OPTION_DECISIONS:
            raise RuntimeError("Bounded BC option budget exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > MAX_RSS_BYTES:
            raise MemoryError("Bounded BC memory budget exhausted")
    report = dict(schema="resectionlab.patient-rescue-bc-run.v1", status="running",
        manifest=str(args.manifest.resolve()), manifestSha256=sha(args.manifest),
        teacherProfile=str(args.teacher_profile.resolve()), teacherProfileSha256=sha(args.teacher_profile),
        observationSchema=SCHEMA, options=list(OPTIONS), limitations=[
            "One fixed prepared injury on a development meningioma patient; no transfer or tumor resection.",
            "Five teacher states from a public scripted trajectory; simulator experience, not patient observation.",
            "BC is supervised imitation, not RL; its live result cannot qualify an RL checkpoint."])
    task = None
    try:
        profile = json.loads(args.teacher_profile.read_text())
        manifest_data = json.loads(args.manifest.read_text())
        if profile.get("manifestContentHash") != manifest_data["contentHash"]:
            raise ValueError("Teacher profile belongs to another task")
        policy, initial, report["fit"] = fit(profile, args.seed)
        checkpoint = dict(schema="resectionlab.patient-rescue-bc-checkpoint.v1",
            observationSchema=SCHEMA, options=list(OPTIONS), featureNames=list(FEATURE_NAMES),
            taskManifestSha256=report["manifestSha256"], teacherProfileSha256=report["teacherProfileSha256"],
            seed=args.seed, weight=policy.weight.tolist(), bias=policy.bias.tolist())
        write(output / "bc-checkpoint.json", checkpoint)
        report["checkpoint"] = dict(path="bc-checkpoint.json", sha256=sha(output / "bc-checkpoint.json"))
        task, manifest = from_manifest(args.manifest, check=check)
        report["preparation"] = task.preparation
        def choose_bc(vector, _): return policy.predict(vector)
        live = rollout(task, choose_bc, seed=args.seed, label="bc-frozen-live", check=check)
        decisions += live["optionDecisions"]
        live["checkpointSha256"] = report["checkpoint"]["sha256"]
        write(output / "bc-live.json", live)
        report["live"] = {k:live[k] for k in ("success", "reason", "returnValue", "modeledField",
            "elapsedSeconds", "optionDecisions", "rlTransitionAttempts", "publicEngineCalls")}
        report["live"]["options"] = [r["option"] for r in live["observations"]]
        if report["live"]["options"] != list(TEACHER) or not live["success"]:
            raise ValueError("Frozen live BC did not reproduce the full successful teacher sequence")
        replay = rollout(task, lambda vector, index: (OPTIONS.index(live["observations"][index]["option"]), []),
            seed=args.seed, label="bc-exact-replay", check=check)
        decisions += replay["optionDecisions"]
        if (replay["native"] != live["native"] or replay["attempts"] != live["attempts"]
                or replay["returnValue"] != live["returnValue"]
                or [r["published"]["featuresSha256"] for r in replay["observations"]]
                    != [r["published"]["featuresSha256"] for r in live["observations"]]):
            raise ValueError("BC native/observation/reward replay differs")
        report["exactReplay"] = dict(passed=True, optionDecisions=replay["optionDecisions"],
            elapsedSeconds=replay["elapsedSeconds"])
        report["workspace"] = export_workspace(manifest, live, output, report["fit"])
        report["status"] = "completed_bc_only"
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report["wallSeconds"] = time.perf_counter() - began
        report["expensiveOptionDecisions"] = decisions
        report["peakResidentBytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
        write(output / "result.json", report)
        print(json.dumps({k:report[k] for k in ("status", "wallSeconds", "expensiveOptionDecisions", "error")
            if k in report}), flush=True)


if __name__ == "__main__":
    main()
