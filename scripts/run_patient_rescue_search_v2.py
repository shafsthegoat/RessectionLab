#!/usr/bin/env python3
"""Capped permutation-search and scripted baselines on the v2 rescue sensors."""
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

import numpy as np
from patient_rescue_env import from_manifest, sha
from run_patient_rescue_ppo_bc import (ObservedRescueEnv, rollout, SCRIPT, write)
from resectionlab.patient_rescue_learning import OPTIONS

MAX_WALL_SECONDS = 900.
MAX_OPTION_DECISIONS = 110
SEARCH_DECISIONS = 64


def compact(row):
    return {key: row[key] for key in ("success", "reason", "returnValue",
        "optionDecisions", "rlTransitionAttempts", "modeledField", "elapsedSeconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "testcases") or output.exists():
        parser.error("Use a new ignored testcases output directory")
    output.mkdir(parents=True)
    began = time.perf_counter()
    env = None
    def check():
        if time.perf_counter() - began > MAX_WALL_SECONDS:
            raise TimeoutError("Matched v2 search exceeded its 15-minute wall cap")
        if env is not None and env.total_decisions >= MAX_OPTION_DECISIONS:
            raise RuntimeError("Matched v2 search exceeded option transition cap")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > 3 * 1024 ** 3:
            raise MemoryError("Matched v2 search exceeded 3 GiB")
    report = dict(schema="resectionlab.patient-rescue-v2-search.v1", status="running",
        seed=args.seed, manifestSha256=sha(args.manifest), searchDecisionAllowance=SEARCH_DECISIONS,
        maxTotalOptionDecisions=MAX_OPTION_DECISIONS,
        observationSchema="resectionlab.prepared-rescue-observation.v2",
        searchPrior="seeded full six-option permutations; no action mask or preview",
        limitations=["One fixed prepared injury; no patient transfer or clinical claim.",
                     "Search uses a permutation prior unlike unconstrained PPO; matching the interaction allowance does not remove this structural difference."])
    report["codeSha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT / "scripts/patient_rescue_env.py",
        ROOT / "src/resectionlab/patient_rescue_learning.py")}
    try:
        task, manifest = from_manifest(args.manifest, check=check)
        report["preparation"] = task.preparation
        env = ObservedRescueEnv(task, check)
        env.mode = "scripted"
        scripted = rollout(env, lambda obs, i: (OPTIONS.index(SCRIPT[i]), []),
                           seed=args.seed, label="scripted-v2")
        write(output / "scripted.json", scripted)
        report["scripted"] = compact(scripted)
        report["scripted"]["options"] = [r["option"] for r in scripted["decisions"]]
        if not scripted["success"]:
            raise ValueError("Scripted baseline no longer solves the exact task")
        rng = np.random.default_rng(args.seed)
        report["candidates"] = []
        used = 0
        best = None
        while used < SEARCH_DECISIONS:
            check()
            sequence = tuple(str(item) for item in rng.permutation(OPTIONS))
            remaining = SEARCH_DECISIONS - used
            old_limit = task.max_decisions
            task.max_decisions = min(old_limit, remaining)
            try:
                env.mode = "search"
                row = rollout(env, lambda obs, i: (
                    OPTIONS.index(sequence[i] if i < len(sequence) else "finish_verify"), []),
                    seed=args.seed, label="search-candidate")
            finally:
                task.max_decisions = old_limit
            path = f"candidate-{len(report['candidates']):02d}.json"
            write(output / path, row)
            used += row["optionDecisions"]
            summary = dict(path=path, sequence=list(sequence), **compact(row))
            report["candidates"].append(summary)
            if best is None or row["returnValue"] > best["returnValue"]:
                best = summary
        if used != SEARCH_DECISIONS:
            raise ValueError("Search interaction allowance differs")
        report["searchOptimizationOptionDecisions"] = used
        report["selectedCandidate"] = best
        sequence = best["sequence"]
        env.mode = "search-selected"
        selected = rollout(env, lambda obs, i: (
            OPTIONS.index(sequence[i] if i < len(sequence) else "finish_verify"), []),
            seed=args.seed, label="search-selected-v2")
        write(output / "search-selected.json", selected)
        report["searchSelected"] = compact(selected)
        report["searchSelected"]["options"] = [r["option"] for r in selected["decisions"]]
        report["selectedCandidateReproducedExactly"] = (
            selected["returnValue"] == best["returnValue"]
            and selected["native"] == json.loads((output / best["path"]).read_text())["native"])
        env.mode = "search-replay"
        replay_options = report["searchSelected"]["options"]
        replay = rollout(env, lambda obs, i: (OPTIONS.index(replay_options[i]), []),
            seed=args.seed, label="search-replay-v2")
        if (replay["native"] != selected["native"] or replay["attempts"] != selected["attempts"]
                or replay["returnValue"] != selected["returnValue"]
                or [r["published"]["featuresSha256"] for r in replay["decisions"]]
                    != [r["published"]["featuresSha256"] for r in selected["decisions"]]):
            raise ValueError("Selected search observation/native/reward replay differs")
        report["exactReplay"] = dict(passed=True, optionDecisions=replay["optionDecisions"],
            elapsedSeconds=replay["elapsedSeconds"])
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report["wallSeconds"] = time.perf_counter() - began
        report["totalOptionDecisions"] = env.total_decisions if env else 0
        report["peakResidentBytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
        write(output / "result.json", report)
        print(json.dumps({k:report[k] for k in ("status", "wallSeconds", "totalOptionDecisions", "error")
            if k in report}), flush=True)


if __name__ == "__main__":
    main()
