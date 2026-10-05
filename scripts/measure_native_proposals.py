#!/usr/bin/env python3
"""One bounded actual-case observation; keep integrity and preview costs separate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import resource
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Retain the existing observation and choose a new output")
    from resectionlab.imaging import load_case
    from resectionlab.native_simulation import make_native_patient_simulator
    from resectionlab.native_proposals import PreparedAxisColumnProposer, _verify_cavity
    import resectionlab.native_proposals as source_module
    phases = []
    def peak_rss():
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    def measure(name, function):
        before = peak_rss()
        started = perf_counter()
        result = function()
        phases.append({"phase": name, "seconds": perf_counter() - started,
                       "process_peak_rss_before_bytes": before, "process_peak_rss_after_bytes": peak_rss()})
        return result
    source_path = Path(source_module.__file__)
    source_sha = sha256(source_path.read_bytes()).hexdigest()
    case = measure("load_source_case", lambda: load_case(args.case))
    sim = measure("native_factory_setup", lambda: make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7))
    engine = sim.engine
    provider = measure("provider_source_verification_and_static_preparation", lambda: PreparedAxisColumnProposer(engine.config))
    prototype_path = Path(__file__).resolve().parents[1] / "artifacts/native-frontier-expansion-v1/experiment-2/experiment-script.py"
    spec = importlib.util.spec_from_file_location("native_frontier_frozen_prototype", prototype_path)
    prototype = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prototype)
    def equivalent(batch):
        old = [(ray["tool_id"], tuple(ray["entry_mm"]), tuple(tuple(tip) for tip in ray["endpoints_mm"]))
               for ray in prototype.axis_rays(engine)[0]]
        new = [(ray.tool_id, ray.entry_mm, (ray.primary_target_mm,) + (() if ray.fallback_target_mm is None else (ray.fallback_target_mm,)))
               for ray in batch.proposals]
        return old == new
    measure("initial_history_and_whole_grid_integrity_only", lambda: _verify_cavity(engine))
    initial = measure("initial_propose_including_integrity", lambda: provider.propose(engine))
    measure("initial_validate_batch_including_integrity", lambda: provider.validate_batch(initial, engine))
    assert equivalent(initial)
    ray = next(row for row in initial.proposals if row.column_index == 0 and row.tool_id == "native-wide-aspiration")
    preview = measure("one_complete_native_preview", lambda: engine.preview_stroke(ray.tool_id, ray.primary_target_mm, entry_mm=ray.entry_mm))
    assert preview.feasible
    measure("one_native_certificate_commit", lambda: engine.commit_preview(preview))
    measure("updated_history_and_whole_grid_integrity_only", lambda: _verify_cavity(engine))
    updated = measure("updated_propose_including_integrity", lambda: provider.propose(engine))
    measure("updated_validate_batch_including_integrity", lambda: provider.validate_batch(updated, engine))
    assert equivalent(updated)
    stale_rejected = False
    try:
        provider.validate_batch(initial, engine)
    except ValueError:
        stale_rejected = True
    assert stale_rejected and source_sha == sha256(source_path.read_bytes()).hexdigest()
    report = {"source_hash": case.semantic_hash, "native_engine_hash": engine.config.fingerprint,
        "proposal_model_hash": provider.model_hash, "rule_hash": provider.rule_hash,
        "source_module_sha256": source_sha, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "phases": phases, "measurement_repetitions_per_phase": 1,
        "memory_semantics": "Cumulative process peak resident set; differences are not isolated per-phase allocation costs.",
        "initial_batch": initial.to_dict(), "updated_batch": updated.to_dict(),
        "prototype_equivalence_initial_and_updated": True, "stale_batch_rejected": stale_rejected,
        "preview_feasible": preview.feasible, "completed_native_strokes": engine.revision,
        "gradient_steps": 0, "final_worlds_used": False, "stress_worlds_used": False,
        "clinical_deficit_probability": None,
        "scope": "One local performance/correctness observation. Integrity verification is not independent geometric certification or clinical validation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"phases": phases, "prototype_equivalent": True, "stale_batch_rejected": stale_rejected}))


if __name__ == "__main__":
    main()
