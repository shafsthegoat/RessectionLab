#!/usr/bin/env python3
"""Collect the one successful public-action teacher trace for bounded rescue BC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
from patient_rescue_env import OPTIONS, from_manifest, sha
from resectionlab.patient_rescue_learning import ROUTES

SCRIPT = ("coagulate_center", "coagulate_lower", "coagulate_forward",
          "aspirate_pool", "finish_verify")


def route_samples(task):
    scene = task.raw.scene
    fluid = task.raw.engine.state["fluid"]
    result = []
    for route in ROUTES:
        xyz = np.asarray(task.waypoints[route]["position"], dtype=float)
        ijk = np.rint(scene.inverse[:3, :3] @ xyz + scene.inverse[:3, 3]).astype(int)
        values = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    point = ijk + (i, j, k)
                    if np.all((point >= 0) & (point < scene.shape)):
                        index = int(np.ravel_multi_index(tuple(point), scene.shape))
                        values.append(float(fluid.get(str(index), 0.)))
        result.append(dict(name=route, ijk=ijk.tolist(),
                           fluidSum5=sum(values), fluidMax5=max(values, default=0.)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "testcases") or output.exists():
        parser.error("Use a new ignored testcases output JSON path")
    output.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    def check():
        if time.perf_counter() - began > 360.:
            raise TimeoutError("Teacher profiling wall budget exceeded")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > 3 * 1024 ** 3:
            raise MemoryError("Teacher profiling memory budget exceeded")
    task, manifest = from_manifest(args.manifest, check=check)
    setup = time.perf_counter() - began
    rows = []
    for option in SCRIPT:
        check()
        observed = task._observation()
        field = task.raw.field()
        samples = route_samples(task)
        counts = task.counts.copy()
        action_start = time.perf_counter()
        _, reward, terminated, truncated, info = task.step(OPTIONS.index(option))
        rows.append(dict(option=option, fieldBefore=field.tolist(),
            featureBefore=observed.tolist(), countsBefore=counts.tolist(),
            routeLocalBefore=samples, reward=reward,
            elapsedSeconds=time.perf_counter() - action_start,
            terminated=terminated, truncated=truncated,
            reason=info["termination_reason"], fieldAfter=task.raw.field().tolist()))
        if terminated or truncated:
            break
    if len(rows) != len(SCRIPT) or not rows[-1]["terminated"] or not info["success"]:
        raise ValueError("Public teacher did not complete the five-skill task")
    record = dict(schema="rescue-bc-profile.v1", manifestContentHash=manifest["contentHash"],
        manifestSha256=sha(args.manifest), sourceSceneId=manifest["sourceSceneId"],
        taskStartStateId=manifest["taskStartStateId"], setupSeconds=setup,
        rows=rows, wallSeconds=time.perf_counter() - began,
        stateId=task.raw.engine.state_id,
        provenance="Simulated public-action teacher on acquired development anatomy; not patient observation")
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "success": True, "wallSeconds": record["wallSeconds"]}))


if __name__ == "__main__":
    main()
