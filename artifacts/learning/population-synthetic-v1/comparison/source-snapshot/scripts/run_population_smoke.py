#!/usr/bin/env python3
"""Freeze source, pretrain on analytic groups, compare four arms on one excluded task.

This is a synthetic development experiment. It never opens final/stress worlds
and makes no clinical population or independent patient performance claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_patient_learning import (preserve_source, run_experiment, source_snapshot, write_json)
from resectionlab.learning import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline-gradient-steps", type=int, default=8)
    parser.add_argument("--online-gradient-steps", type=int, default=8)
    parser.add_argument("--environment-steps", type=int, default=128)
    parser.add_argument("--wall-seconds", type=float, default=20.)
    parser.add_argument("--frozen-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    common = dict(hidden_features=16, max_environment_steps=args.environment_steps,
                  max_wall_seconds=args.wall_seconds, max_episode_steps=8,
                  episodes_per_update=2, checkpoint_interval=2)
    offline_config = TrainingConfig(max_gradient_steps=args.offline_gradient_steps, **common)
    online_config = TrainingConfig(max_gradient_steps=args.online_gradient_steps, **common)
    if not args.frozen_worker:
        launch_started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "experiment-status.json", {"status": "running", "final_worlds_used": False})
        snapshot = source_snapshot()
        write_json(output / "prespecified-design.json", {
            "study_role": "analytic_synthetic_development_four_arm_comparison",
            "independent_patient_count": 0, "offline_config": asdict(offline_config),
            "online_config": asdict(online_config), "online_seeds": [11, 23, 47],
            "online_world_counts": [3, 2, 3, 2], "final_worlds_will_be_used": False,
            "population_checkpoint_rule": "fixed_budget_latest",
            "online_checkpoint_rule": "maximum_selection_mean_earliest_tie",
            "source": snapshot})
        frozen = output / "frozen-source"
        try:
            preserve_source(snapshot, frozen)
            subprocess.run([sys.executable, str(frozen / "scripts/run_population_smoke.py"),
                "--output", str(output), "--offline-gradient-steps", str(args.offline_gradient_steps),
                "--online-gradient-steps", str(args.online_gradient_steps),
                "--environment-steps", str(args.environment_steps),
                "--wall-seconds", str(args.wall_seconds), "--frozen-worker"], cwd=frozen, check=True)
        except Exception as exc:
            write_json(output / "experiment-status.json", {"status": "failed", "final_worlds_used": False,
                "reason": str(exc), "exception": type(exc).__name__,
                "elapsed_seconds_including_source_freeze": time.perf_counter() - launch_started})
            raise
        import json
        summary = json.loads((output / "summary.json").read_text())
        write_json(output / "experiment-status.json", {"status": summary["status"], "final_worlds_used": False,
            "elapsed_seconds_including_source_freeze": time.perf_counter() - launch_started})
        return
    from resectionlab.population_learning import make_analytic_population_fixture, train_population_policy
    started = time.perf_counter()
    members, target_factory, target = make_analytic_population_fixture()
    write_json(output / "cohort.json", {"development": [asdict(member.identity) for member in members],
        "excluded_target": asdict(target), "independent_patient_count": 0,
        "limits": ["Analytic tasks only", "Target excluded from this shared checkpoint",
                   "Target remains an algorithm-development fixture, not a locked test patient"]})
    pretrained = train_population_policy(members, exclusions=(target,), config=offline_config,
                                        output_dir=output / "pretraining")
    if pretrained.get("status") != "completed" or not pretrained.get("checkpoint_path"):
        write_json(output / "summary.json", {"status": "failed_pretraining", "pretraining": pretrained,
                                             "final_worlds_used": False})
        raise RuntimeError("Population pretraining did not export a qualified shared checkpoint")
    checkpoint = Path(pretrained["checkpoint_path"])
    comparison = output / "comparison"
    status = run_experiment(target_factory, comparison, config=online_config,
        seeds=(11, 23, 47), counts=(3, 2, 3, 2), evaluate=False,
        population_checkpoint=checkpoint, population_case_group=target.group_id,
        population_case_aliases=target.aliases)
    import json
    def read(name: str):
        return json.loads((comparison / name).read_text())
    records = read("training.json")
    rows = [{key: record[key] for key in (
        "optimizer_mode", "initial_checkpoint_hash", "selected_checkpoint_hash", "shared_checkpoint_hash",
        "initial_selection_return", "selected_selection_return", "gradient_steps",
        "optimization_environment_steps", "selection_environment_steps", "elapsed_seconds",
        "preparation_seconds", "candidate_extraction_seconds")} for record in records]
    for row, record in zip(rows, records):
        row["seed"] = int(Path(record["output_dir"]).name.rsplit("-", 1)[1])
    audit_path = comparison / status.get("geometry_validation_run_id", "") / "coarse-sequence-audit.json"
    audits = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    summary = {"status": status["status"], "study_role": "analytic_synthetic_development_four_arm_comparison",
        "independent_patient_count": 0, "target": asdict(target), "seed_order": [11, 23, 47],
        "offline_pretraining": pretrained["provenance"]["offline_pretraining"],
        "population_checkpoint_hash": pretrained["policy_hash"],
        "frozen": read("population-frozen.json"), "search": read("search.json"), "learning": rows,
        "independent_geometry": {name: audit["feasible"] for name, audit in audits.items()},
        "full_validation_seconds": status.get("full_validation_seconds"),
        "total_experiment_seconds": time.perf_counter() - started,
        "final_worlds_used": False, "clinical_deficit_probability": None,
        "limitations": ["Two or more procedural development groups are not human patients",
            "One shared initialization does not estimate population-training-seed variation",
            "Only the supplied action/reward/world assumptions were tested",
            "Source snapshot and explicit exclusions support reproducibility, not clinical generalization",
            "Offline pretraining and online inference/adaptation costs are separate"]}
    write_json(output / "summary.json", summary)
    print(f"Synthetic four-arm result: {status['status']}; summary saved to {output / 'summary.json'}")


if __name__ == "__main__":
    main()
