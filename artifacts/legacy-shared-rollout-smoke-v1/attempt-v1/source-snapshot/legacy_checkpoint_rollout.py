"""Prospective one-shot generated opening RL256/search integration control.

The default invocation is preflight only and NEVER loads weights. Root may run
``--execute`` after coordinating resources. This does not support mixed-mode
probe observations, patients, training, retries, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import resectionlab

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import torch

from resectionlab.core import semantic_digest
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
from resectionlab.shared_episode import (PolicyIdentity, execute_policy_episode,
                                        execute_search_episode, verify_strategy_replay)


ART = ROOT / "artifacts/native-opening-rl-capacity-v1"
CHECKPOINT = ART / "RL-256.pt"
EXPECTED = {
    "checkpoint_file_sha256": "1d391665f66cdd0c1fa1db150261b853820a9d30d62fc20c99cf021cf0229fbd",
    "checkpoint_bytes": 394009,
    "result_sha256": "301e690fb87adff99b715c1371cb0fc692b0adc16fc3576ff7b0725336c56f68",
    "output_index_sha256": "d378524c205838c28f09b2bc02e2a1e08dadc9a25acd63a0170043afc3aeea3d",
    "independent_verification_sha256": "2489aff1b8a783f1fd6ad97a1085abb4b54aff30b44c337deb42a654d5afc95f",
    "spatial_policy_source_sha256": "5801f1b59b5eed90edeb33e5182d57547de35dd0fd326c76c4e427f89c05b6a4",
    "spatial_observation_source_sha256": "48a2a9d1c7faee791f7f327ac928e133f2d0f80f60a045a868802bb75785d711",
    "architecture_hash": "sha256:a3b6740ab188f01016610c566e2009c57ba422c8f9be24adca014f695ae9c917",
    "parameter_hash": "sha256:f52e14097ea6a35436eaf30f0ece5658e24987e289ca30ff6a529816545b5721",
    "initial_parameter_hash": "sha256:2fa97c0e730db09371786d5ba903ffdf467b3cc7149a111c182d6a7499cdd9b8",
    "decision_model_hash": "sha256:f607b51b66d00164c33ff9c8fa7725b0e6e15bab1b566d161efbe3fb4f4e730b",
    "source_hash": "sha256:9b39c31e1e012b0822d538451a0452fa9c564acc863d22dabb311ec4cf8c5d4e",
    "initial_observation_hash": "sha256:c9458a99068bfaae90ba69062073a674c55ee31ad66d11a5c86461ef2c834eba",
    "updates": 256,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> tuple[object, dict]:
    files = {
        ART / "result.json": "result_sha256",
        ART / "output-sha256.json": "output_index_sha256",
        ART / "independent-verification.json": "independent_verification_sha256",
        ROOT / "src/resectionlab/spatial_policy.py": "spatial_policy_source_sha256",
        ROOT / "src/resectionlab/spatial_observations.py": "spatial_observation_source_sha256",
    }
    for path, key in files.items():
        if sha(path) != EXPECTED[key]:
            raise ValueError(f"Frozen source/evidence bytes changed: {path}")
    if CHECKPOINT.stat().st_size != EXPECTED["checkpoint_bytes"] or sha(CHECKPOINT) != EXPECTED["checkpoint_file_sha256"]:
        raise ValueError("RL256 checkpoint bytes differ from the independently reviewed artifact")
    index = json.loads((ART / "output-sha256.json").read_text())
    if index.get("RL-256.pt") != EXPECTED["checkpoint_file_sha256"]:
        raise ValueError("Checkpoint differs from the released output index")
    release = json.loads((ART / "result.json").read_text())
    method = release["methods"]["RL256"]
    context = release["context"]
    task = make_native_opening_task()
    observation = task.observation()
    if (release["status"] != "complete" or release["updates"] != EXPECTED["updates"]
            or method["status"] != "complete" or method["parameter_hash"] != EXPECTED["parameter_hash"]
            or release["architecture_hash"] != EXPECTED["architecture_hash"]
            or context["decision_model_hash"] != EXPECTED["decision_model_hash"]
            or context["source_ids"] != [EXPECTED["source_hash"]]
            or context["real_patient_count"] != 0 or context["transition_lineage"] != "simulator_generated"
            or context["max_steps"] != 2 or task.decision_model_hash != EXPECTED["decision_model_hash"]
            or task.case.source_hash != EXPECTED["source_hash"]
            or observation.fingerprint != EXPECTED["initial_observation_hash"]
            or observation.track != "synthetic_scan" or task.max_steps != 2):
        raise ValueError("Generated task, source, objective, or release context differs")
    return task, release


def load_bound_policy(task, release) -> tuple[SpatialPolicy, PolicyIdentity, dict]:
    """One CPU weights_only load after complete byte/task preflight; no fallback."""
    payload = CHECKPOINT.read_bytes()
    if len(payload) != EXPECTED["checkpoint_bytes"] or hashlib.sha256(payload).hexdigest() != EXPECTED["checkpoint_file_sha256"]:
        raise ValueError("Checkpoint changed between preflight and safe load")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    required = {"policy", "optimizer", "updates", "parameter_hash", "initial_parameter_hash",
                "architecture", "context", "action_rng"}
    if type(checkpoint) is not dict or set(checkpoint) != required:
        raise ValueError("RL256 checkpoint schema differs")
    if (checkpoint["updates"] != EXPECTED["updates"]
            or checkpoint["parameter_hash"] != EXPECTED["parameter_hash"]
            or checkpoint["initial_parameter_hash"] != EXPECTED["initial_parameter_hash"]):
        raise ValueError("RL256 checkpoint training or architecture binding differs")
    # The release records only the architecture digest. The checkpoint carries
    # the exact typed constructor configuration and full architecture record.
    architecture = checkpoint["architecture"]
    if (not isinstance(architecture, dict) or not isinstance(architecture.get("config"), dict)
            or checkpoint["context"] != release["context"]):
        raise ValueError("Checkpoint context/architecture differs from release")
    policy = SpatialPolicy(SpatialPolicyConfig(**architecture["config"]))
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    if (policy.architecture_record() != architecture or policy.architecture_hash != EXPECTED["architecture_hash"]
            or parameter_hash(policy) != EXPECTED["parameter_hash"]
            or task.decision_model_hash != checkpoint["context"]["decision_model_hash"]):
        raise ValueError("Safely loaded policy differs from the released RL256 identity")
    receipt = {"checkpoint_file_sha256": "sha256:" + EXPECTED["checkpoint_file_sha256"],
               "checkpoint_bytes": len(payload), "safe_loader": "torch.load(weights_only=True,map_location=cpu)",
               "strict_state_dict": True, "updates": EXPECTED["updates"],
               "policy_parameter_hash": parameter_hash(policy),
               "architecture_hash": policy.architecture_hash,
               "decision_model_hash": task.decision_model_hash,
               "source_hash": task.case.source_hash,
               "scope": "generated_legacy_aspirate_only"}
    identity = PolicyIdentity("native-opening-RL256", policy.architecture_hash,
        parameter_hash(policy), "permitted-spatial-observation-v1", "trained_checkpoint",
        "sha256:" + EXPECTED["checkpoint_file_sha256"], semantic_digest(receipt))
    policy._integration_verified_checkpoint_sha256 = identity.checkpoint_sha256
    return policy, identity, receipt


def generated_outcomes(task, actions: list[str]) -> dict:
    worker = task.fresh()
    for action in actions:
        worker.step(action)
    if not worker.terminated:
        raise RuntimeError("Generated outcome requires a complete route")
    if not worker.independent_geometry_check().feasible:
        raise RuntimeError("Generated route failed independent geometry check")
    metrics = worker.metrics()
    return {key: metrics[key] for key in ("total_reward", "target_removed_mm3",
        "normal_removed_mm3", "simulated_removed_volume_mm3", "clinical_deficit_probability")}


def write_reserved(destination, payload: dict) -> None:
    """Replace only this attempt's already-exclusive output; keep killed runs visible."""
    destination.seek(0)
    destination.truncate()
    json.dump(payload, destination, sort_keys=True, indent=2, allow_nan=False)
    destination.write("\n")
    destination.flush()
    os.fsync(destination.fileno())


def execute(output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        write_reserved(destination, {"status": "started", "scope": "generated_legacy_aspirate_only",
            "checkpoint_sha256": EXPECTED["checkpoint_file_sha256"], "automatic_retry": False})
        stage = "preflight"
        try:
            torch.set_num_threads(1)
            task, release = preflight()
            stage = "safe_checkpoint_load"
            load_started = time.perf_counter()
            policy, identity, receipt = load_bound_policy(task, release)
            load_seconds = time.perf_counter() - load_started
            stage = "policy_episode"
            policy_started = time.perf_counter()
            policy_record = execute_policy_episode(task, policy, identity)
            policy_seconds = time.perf_counter() - policy_started
            stage = "search_episode"
            search_started = time.perf_counter()
            search_record = execute_search_episode(task, max_calls=24, beam_width=2, seconds=10.)
            search_seconds = time.perf_counter() - search_started
            stage = "replay_verification"
            for record in (policy_record, search_record):
                if not verify_strategy_replay(task, record):
                    raise RuntimeError("Saved strategy did not replay")
            comparison = {"status": "complete", "version": "legacy-RL256-shared-transition-control-v1",
                "scope": "generated_fixed_task_only_no_patient_generalization",
                "training_updates_preexisting": 256, "new_training_updates": 0,
                "checkpoint_load_seconds": load_seconds,
                "policy_online_seconds": policy_seconds, "policy_forward_calls": len(policy_record["decisions"]),
                "search_online_seconds": search_seconds,
                "same_source": policy_record["source_hash"] == search_record["source_hash"],
                "same_environment": policy_record["environment_contract_hash"] == search_record["environment_contract_hash"],
                "same_initial_observation": policy_record["decisions"][0]["observation_before"] ==
                                            search_record["decisions"][0]["observation_before"],
                "checkpoint_validation": receipt, "policy": policy_record, "search": search_record,
                "generated_modeled_outcomes": {
                    "policy": generated_outcomes(task, policy_record["action_ids"]),
                    "search": generated_outcomes(task, search_record["action_ids"])},
                "clinical_validation": None}
            write_reserved(destination, comparison)
        except BaseException as error:
            write_reserved(destination, {"status": "failed", "stage": stage,
                "error_type": type(error).__name__, "error": str(error),
                "scope": "generated_legacy_aspirate_only", "automatic_retry": False})
            raise
    return comparison


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="One frozen generated checkpoint rollout + search")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.execute:
        if args.output is None:
            parser.error("--execute requires a new --output path")
        result = execute(args.output)
        print(json.dumps({"status": "completed", "output": str(args.output),
            "policy_actions": result["policy"]["action_ids"],
            "search_actions": result["search"]["action_ids"]}, sort_keys=True))
        return 0
    task, _ = preflight()
    print(json.dumps({"status": "preflight_passed_no_checkpoint_load",
        "task_decision_model_hash": task.decision_model_hash,
        "checkpoint_sha256": EXPECTED["checkpoint_file_sha256"],
        "execute_authority": "root_resource_coordination_required"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
