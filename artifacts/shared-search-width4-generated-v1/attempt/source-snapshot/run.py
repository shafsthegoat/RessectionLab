"""One generated-only observed-search width-4 control on the shared native task.

No checkpoint load, learned policy forward, training, patient data, or reference
target is used to select actions. The sole search call is bounded separately by
the reviewed subprocess supervisor. Root authorized exactly one attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3_WRAPPER = ROOT / "build/integration-first-policy-v3/supervise_legacy_rollout.py"
V3_WRAPPER_SHA256 = "02b7b5ca7b0a93fa4bae55694a342d48dfbe8fab024b42cfdbaec3096616bdb8"
V3_COMPARISON = ROOT / "build/legacy-RL256-shared-executed-v3/comparison.json"
V3_COMPARISON_SHA256 = "07d61374b4b97271e1f5c6f5e6eab395725afdcef953acdd2799c5a178e917c8"
SOURCE_HASH = "sha256:9b39c31e1e012b0822d538451a0452fa9c564acc863d22dabb311ec4cf8c5d4e"
MODEL_HASH = "sha256:f607b51b66d00164c33ff9c8fa7725b0e6e15bab1b566d161efbe3fb4f4e730b"
OBS_HASH = "sha256:c9458a99068bfaae90ba69062073a674c55ee31ad66d11a5c86461ef2c834eba"
MAX_WALL_SECONDS = 30
MAX_RSS_BYTES = 1024 ** 3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_support():
    if sha(V3_WRAPPER) != V3_WRAPPER_SHA256:
        raise ValueError("Reviewed generated supervisor wrapper changed")
    spec = importlib.util.spec_from_file_location("bound_v3_generated_supervisor", V3_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Bound V3 supervisor unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_hashes = module.check_sources()
    return module, source_hashes


def write_reserved(stream, payload: dict) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def child(result: Path, expected_runner_sha256: str) -> None:
    with result.open("x") as stream:
        write_reserved(stream, {"status": "started", "scope": "generated_search_only",
            "automatic_retry": False})
        stage = "source_and_task_preflight"
        try:
            if sha(Path(__file__).resolve()) != expected_runner_sha256:
                raise ValueError("Search runner changed after parent source binding")
            validated_support()
            import torch
            torch.set_num_threads(1)
            from resectionlab.native_spatial_task import make_native_opening_task
            from resectionlab.shared_episode import execute_search_episode, verify_strategy_replay
            task = make_native_opening_task()
            observation = task.observation()
            if (task.case.source_hash != SOURCE_HASH or task.decision_model_hash != MODEL_HASH
                    or observation.fingerprint != OBS_HASH or observation.track != "synthetic_scan"):
                raise ValueError("Generated source/model/starting observation changed")
            stage = "one_observed_search"
            record = execute_search_episode(task, max_calls=24, beam_width=4, seconds=10.)
            stage = "replay_and_result"
            if not verify_strategy_replay(task, record):
                raise ValueError("Shared native search history did not replay")
            if (record["method"] != "SEARCH" or record["policy_identity"]["checkpoint_sha256"] is not None
                    or record["search_accounting"]["actor_forward_calls"] != 0
                    or record["source_hash"] != SOURCE_HASH):
                raise ValueError("Search-only identity or accounting changed")
            replay = task.fresh()
            for action in record["action_ids"]:
                replay.step(action)
            if not replay.terminated or not replay.independent_geometry_check().feasible:
                raise ValueError("Generated search route failed native audit")
            metrics = replay.metrics()
            result_record = {"status": "complete", "version": "generated-shared-search-width4-v1",
                "scope": "generated_search_only_no_patient_or_learner", "new_training_updates": 0,
                "policy_forward_calls": 0, "checkpoint_loads": 0,
                "same_v3_source": True, "same_v3_environment": True,
                "same_v3_initial_observation": True, "search": record,
                "generated_modeled_outcome": {key: metrics[key] for key in
                    ("total_reward", "target_removed_mm3", "normal_removed_mm3",
                     "simulated_removed_volume_mm3", "clinical_deficit_probability")},
                "clinical_validation": None}
            write_reserved(stream, result_record)
        except BaseException as error:
            write_reserved(stream, {"status": "failed", "stage": stage,
                "error_type": type(error).__name__, "error": str(error),
                "scope": "generated_search_only", "automatic_retry": False})
            raise


def parent(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    own_sha = sha(Path(__file__).resolve())
    started = {"status": "started", "scope": "generated_search_only",
        "runner_sha256": own_sha, "max_wall_seconds": MAX_WALL_SECONDS,
        "max_sampled_worker_rss_bytes": MAX_RSS_BYTES, "automatic_retry": False}
    (output / "attempt.json").write_text(json.dumps(started, sort_keys=True, indent=2) + "\n")
    controller, sources = validated_support()
    started["source_sha256"] = sources
    (output / "attempt.json").write_text(json.dumps(started, sort_keys=True, indent=2) + "\n")
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    supervised = controller.bound_supervisor()([sys.executable, "-B", str(Path(__file__).resolve()),
        "--child", "--output", str(output / "search.json"), "--expected-runner-sha256", own_sha],
        output, {"max_wall_seconds": MAX_WALL_SECONDS, "max_rss_bytes": MAX_RSS_BYTES}, own_sha)
    if sha(Path(__file__).resolve()) != own_sha or controller.check_sources() != sources:
        raise ValueError("Runner or shared numerical source changed during search attempt")
    if supervised["status"] != "complete":
        return supervised
    search_path = output / "search.json"
    if not search_path.is_file() or search_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("No bounded search-only result")
    result = json.loads(search_path.read_text())
    if result.get("status") != "complete" or result.get("checkpoint_loads") != 0:
        raise ValueError("Supervised child did not complete a search-only result")
    if sha(V3_COMPARISON) != V3_COMPARISON_SHA256:
        raise ValueError("Saved V3 actor/search anchor changed")
    previous = json.loads(V3_COMPARISON.read_text())
    current = result["search"]
    anchor = previous["policy"]
    if (current["source_hash"] != anchor["source_hash"]
            or current["environment_contract_hash"] != anchor["environment_contract_hash"]
            or current["decisions"][0]["observation_before"] != anchor["decisions"][0]["observation_before"]):
        raise ValueError("Search control differs from saved RL256 task/start")
    comparison = {"status": "complete", "scope": "saved_generated_software_comparison_only",
        "new_training_updates": 0, "new_policy_forward_calls": 0, "v3_comparison_sha256": V3_COMPARISON_SHA256,
        "width2": {"beam_width": previous["search"]["search_accounting"]["beam_width"],
            "model_transition_calls": previous["search"]["search_accounting"]["model_transition_calls"],
            "action_ids": previous["search"]["action_ids"],
            "modeled_return": previous["generated_modeled_outcomes"]["search"]["total_reward"]},
        "width4": {"beam_width": current["search_accounting"]["beam_width"],
            "model_transition_calls": current["search_accounting"]["model_transition_calls"],
            "call_cap_reached": current["search_accounting"]["call_cap_reached"],
            "time_cap_reached": current["search_accounting"]["time_cap_reached"],
            "action_ids": current["action_ids"],
            "modeled_return": result["generated_modeled_outcome"]["total_reward"]},
        "saved_rl256_modeled_return": previous["generated_modeled_outcomes"]["policy"]["total_reward"],
        "privileged_complete_tree_reference_return": 1.1,
        "privileged_reference_is_not_matched_limited_input_search": True,
        "clinical_validation": None}
    (output / "saved-comparison.json").write_text(json.dumps(comparison, sort_keys=True, indent=2,
        allow_nan=False) + "\n")
    return supervised


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--expected-runner-sha256")
    args = parser.parse_args(argv)
    if args.child:
        if not args.expected_runner_sha256:
            parser.error("Child requires parent-bound runner SHA-256")
        child(args.output, args.expected_runner_sha256)
        return 0
    if args.expected_runner_sha256:
        parser.error("Runner hash is child-only")
    outcome = parent(args.output)
    print(json.dumps({"status": outcome["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if outcome["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
