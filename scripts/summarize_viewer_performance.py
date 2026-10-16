"""Summarize retained Electron traces without dropping long frames or stalls."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {"count": len(values), "median": median(values),
            "p95": values[max(0, math.ceil(len(values) * .95) - 1)], "max": values[-1]}


def summarize(report):
    groups = {}
    for name in ("rotate", "pan", "zoom", "slices", "sidebar", "resident-toggle", "dense-rotate"):
        runs = [r for r in report["runs"] if r["name"].startswith(name + "-")]
        frames = [f for r in runs for f in r.get("frames", [])]
        intervals = [b["at"] - a["at"] for r in runs for a, b in zip(r["frames"], r["frames"][1:])]
        delivered = [b - a for r in runs for a, b in zip(r["presented"], r["presented"][1:])]
        active_delivery = []
        for run in runs:
            start = next((e["at"] for e in run["inputs"] if e["type"] in ("mouseDown", "mouseWheel")), None)
            active = [at for at in run["presented"] if start is not None and start <= at <= run["inputEnded"]]
            active_delivery.extend(b-a for a,b in zip(active, active[1:]))
        response = []
        for run in runs:
            gpu = {g["serial"]: g for g in run.get("gpu", []) if g["completed"]}
            for event in run.get("domInputs", []):
                if event["type"] not in ("wheel", "change"):
                    continue
                following = next((f for f in run["frames"] if (f.get("inputAt") or -1) >= event["at"]), None)
                completion = gpu.get(following.get("serial")) if following else None
                if completion:
                    response.append(completion["at"] - event.get("timeStamp", event["at"]))
        groups[name] = {
            "repetitions": len(runs), "submission_interval_ms": stats(intervals),
            "capture_delivery_interval_ms": stats(delivered),
            "capture_active_interval_ms": stats(active_delivery),
            "cpu_submission_ms": stats([f["cpuMs"] for f in frames]),
            "input_to_gpu_completion_ms": stats(response),
            "refinement_ms": stats([v["milliseconds"] for r in runs for v in r["refined"] if v["completed"]]),
            "asset_reads": sum(len(r["transfers"]) for r in runs),
            "canvas_replacements": sum(sum(e["canvasChanged"] for e in r["lifecycle"]) for r in runs),
            "slice_redraws": sum(f.get("sliceDraws", 0) for f in frames) if any("sliceDraws" in f for f in frames) else None,
            "long_tasks_ms": stats([e["ms"] for r in runs for e in r.get("longTasks", [])]),
        }
    load = next(r for r in report["runs"] if r["name"] == "first-layer")
    groups["first_layer"] = {"request_to_ready_ms": load["inputEnded"] - load["start"],
                             "asset_reads": len(load["transfers"]),
                             "asset_bytes": sum(t["bytes"] for t in load["transfers"]),
                             "canvas_replacements": sum(e["canvasChanged"] for e in load["lifecycle"]),
                             "navigation_available": report["checks"]["navigationAvailableDuringLoad"]}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before, after = (json.loads(p.read_text()) for p in (args.before, args.after))
    if not before.get("passed") or not after.get("passed"):
        raise ValueError("Both runs must complete before comparison")
    if before["source"] != after["source"] or before["window"] != after["window"]:
        raise ValueError("Benchmark sources/window differ")
    result = {"before_trace": str(args.before), "after_trace": str(args.after), "source": before["source"],
              "window": before["window"], "before": summarize(before), "after": summarize(after),
              "measurement": "Submission, asynchronous GPU completion, and captured frame delivery are separate. Capture has observer overhead; GPU completion is an upper bound and is not presentation. All intra-scenario gaps, including refinement and stalls, are retained. First layer is one separately reported cold-residency load; recurring gestures use five repetitions."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
