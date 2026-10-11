"""Focused v2 state integrity regressions on generated geometry only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
          ROOT / "build/hybrid-observation-decision-v1/stage/resectionlab/persistent_passive.py")
source_before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location("resectionlab.independent_passive_state_review", SOURCE)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
START = time.monotonic()
signal.signal(signal.SIGALRM, lambda *_: sys.exit("28 s generated control cap"))
signal.alarm(28)


def stop_is_history_truthful_or_mutation_is_refused():
    task = p.make_passive_decision_task()
    for action in (p.MOVE, p.INSPECT, p.ADVANCE, p.WITHDRAW, p.WITHDRAW_STATIONARY):
        task.step(action)
    task._advance_pose = None
    try:
        record = task.step(p.STOP).info
    except RuntimeError as error:
        assert "changed outside a transition" in str(error)
        return {"tampered_postwithdraw_advance_cache_refused": True}
    assert record["stop_reason"] == "declared_advance_completed_without_injury_claim", record["stop_reason"]
    return {"stop_reason_derived_from_committed_advance": record["stop_reason"]}


def clone_registration_mutation_does_not_corrupt_parent():
    task = p.make_passive_decision_task()
    for action in (p.MOVE, p.INSPECT):
        task.step(action)
    original_observation = task.observation().fingerprint
    clone = task.clone()
    try:
        clone._visible_registration["translationNavToSourceRasMm"][0] = 1.
    except TypeError:
        pass  # An immutable registration is an equally valid correction.
    task._assert_frozen()
    assert task.observation().fingerprint == original_observation
    assert list(task._visible_registration["translationNavToSourceRasMm"]) == [0., 0., 0.]
    return {"clone_registration_mutation_cannot_corrupt_original": True}


results = []
for test in (stop_is_history_truthful_or_mutation_is_refused,
             clone_registration_mutation_does_not_corrupt_parent):
    start = time.monotonic()
    try:
        results.append({"name": test.__name__, "passed": True,
                        "detail": test(), "seconds": time.monotonic() - start})
    except Exception as error:
        results.append({"name": test.__name__, "passed": False,
                        "error": type(error).__name__ + ": " + str(error),
                        "seconds": time.monotonic() - start})
    print(json.dumps(results[-1]), flush=True)
signal.alarm(0)
source_after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
receipt = {"review": "independent-generated-registration-state-integrity-v1",
           "source_sha256_before": source_before, "source_sha256_after": source_after,
           "source_stable_during_run": source_before == source_after,
           "elapsed_seconds": time.monotonic() - START, "bounded_process_seconds": 28,
           "acquired_data_reads": False, "model_reads_or_fitting": False,
           "results": results, "passed": source_before == source_after and all(r["passed"] for r in results)}
output = Path(__file__).with_name("state-receipt-" + source_before[:12] + ".json")
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": str(output), "passed": receipt["passed"]}), flush=True)
raise SystemExit(0 if receipt["passed"] else 1)
