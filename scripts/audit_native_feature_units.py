#!/usr/bin/env python3
"""Read-only, independent receipt and source-cell score audit of the frozen study.

No experiment modules are imported, no policy is trained, and no simulation
world is opened. Refuses a running/incomplete study before inspecting outcomes.
Checkpoint tensors are read with torch's restricted weights-only loader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

DECLARATION_HASH = "sha256:b19aac85f9241e37b98ae53d539ad923f3258bf42a106d67e520f37abcf6a237"
RUNTIME_HASH = "sha256:850dab806bca81be6d18f25cc39643e76d7cf1621d7cbad1f96669228fed65fa"
IMPLEMENTATION_COMMIT = "0bffeaa33dc3120183d0c6fb124aff6e2d327a63"
PROFILES = ("RAW", "FEATURE_UNITS")
SEEDS = (11, 23, 47)


class AuditError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def close(actual: Any, expected: float, name: str, *, tolerance: float = 1e-7) -> None:
    require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance),
            f"{name}: expected {expected!r}, received {actual!r}")


def expected_candidates() -> set[str]:
    names = {"STOP", "GREEDY", "SEARCH"}
    for profile in PROFILES:
        names.add(f"{profile}:PROCEDURAL_PRETRAINED_FROZEN")
        for seed in SEEDS:
            names.update(f"{profile}:{mode}:{seed}" for mode in
                         ("PATIENT_SCRATCH_RL", "PROCEDURAL_PRETRAINED_ADAPTED", "INITIAL"))
    return names


def check_selection(record: dict, world_count: int) -> dict:
    history = record.get("selection_history", [])
    require(history and finite(record.get("selected_selection_return")), "Missing complete selection")
    for panel in history:
        require(type(panel.get("world_count")) is int and panel["world_count"] == world_count,
                "Partial panel entered checkpoint selection")
        require(finite(panel.get("mean_return")), "Nonfinite selection return")
        require(0 <= panel["gradient_steps"] <= record["gradient_steps"], "Future selection update")
        require(0 <= panel["optimization_environment_steps"] <= record["optimization_environment_steps"],
                "Selection resource count exceeds run")
    best = max(history, key=lambda panel: panel["mean_return"])  # first wins ties
    require(best["checkpoint_hash"] == record["selected_checkpoint_hash"], "Wrong selected checkpoint or tie rule")
    close(record["selected_selection_return"], best["mean_return"], "selected return")
    close(record["initial_selection_return"], history[0]["mean_return"], "initial return")
    return best


def check_timing(record: dict, arm: dict, budget: dict) -> dict:
    for key in ("elapsed_seconds", "initialization_seconds", "selection_seconds", "final_checkpoint_export_seconds"):
        require(finite(record.get(key)) and record[key] >= 0, f"Invalid measured phase: {key}")
    panels = [row["panel_elapsed_seconds"] for row in record["selection_history"]]
    require(all(finite(item) and item >= 0 for item in panels), "Invalid selection panel timing")
    require(sum(panels) <= record["selection_seconds"] + 1e-7, "Selection duration omits a completed panel")
    require(record["selection_seconds"] <= record["elapsed_seconds"] + 1e-7,
            "Learner budget omits initial or subsequent selection")
    require(record["gradient_steps"] <= budget["max_gradient_steps"], "Gradient cap exceeded")
    require(record["optimization_environment_steps"] <= budget["max_environment_steps"], "Optimization transition cap exceeded")
    overshoot = max(0., record["elapsed_seconds"] - budget["max_wall_seconds"])
    if arm:
        close(arm["learner_wall_budget_overshoot_seconds"], overshoot, "cooperative cap overshoot")
        close(arm["initial_selection_seconds"], panels[0], "initial selection timing")
        close(arm["later_and_partial_selection_seconds"], record["selection_seconds"] - panels[0], "additional selection timing")
        close(arm["incomplete_selection_seconds"], max(0., record["selection_seconds"] - sum(panels)), "partial selection timing")
        require(arm["trainer_call_seconds"] + .001 >= record["elapsed_seconds"] + record["initialization_seconds"],
                "Full call is shorter than its disjoint learner/setup clocks")
        for key in ("preparation_seconds", "candidate_extraction_seconds", "complete_arm_seconds_before_independent_audit"):
            require(finite(arm.get(key)) and arm[key] >= 0, f"Missing external phase cost: {key}")
    return {"learner_seconds": record["elapsed_seconds"], "initialization_seconds": record["initialization_seconds"],
        "selection_seconds": record["selection_seconds"], "initial_selection_seconds": panels[0],
        "additional_selection_seconds": record["selection_seconds"] - panels[0],
        "cooperative_wall_overshoot_seconds": overshoot, "gradient_steps": record["gradient_steps"],
        "optimization_environment_steps": record["optimization_environment_steps"],
        "selection_environment_steps": record["selection_environment_steps"]}


def tensor_array(tensor: Any) -> np.ndarray:
    return np.asarray(tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy())


def tensor_hash(weights: dict, *, trainable: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        if trainable and name == "action_divisors":
            continue
        array = np.ascontiguousarray(tensor_array(tensor))
        require(np.isfinite(array).all(), "Nonfinite checkpoint tensor")
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def check_profile(checkpoint: dict, declared: dict) -> None:
    metadata = {key: value for key, value in declared.items() if key != "profile_content_hash"}
    require(checkpoint.get("input_profile") == metadata, "Checkpoint profile order/units changed")
    require(checkpoint.get("input_profile_hash") == declared["profile_content_hash"] == canonical_hash(metadata),
            "Checkpoint profile hash changed")
    require(list(checkpoint.get("dimensions", [])) == [15, 6, 16], "Checkpoint architecture differs")
    shapes = {"actor.0.weight": (16, 21), "actor.0.bias": (16,), "actor.2.weight": (1, 16), "actor.2.bias": (1,),
        "value.0.weight": (16, 6), "value.0.bias": (16,), "value.2.weight": (1, 16), "value.2.bias": (1,)}
    for key in ("policy", "selected_policy"):
        if key not in checkpoint:
            continue
        weights = checkpoint[key]
        require(set(weights) == set(shapes) | ({"action_divisors"} if declared["profile_id"] == "FEATURE_UNITS" else set()),
                "Checkpoint parameter/buffer keys changed")
        require(all(tensor_array(weights[name]).shape == shape and tensor_array(weights[name]).dtype == np.float32
                    for name, shape in shapes.items()), "Checkpoint parameter shape/dtype changed")
        if declared["profile_id"] == "RAW":
            require("action_divisors" not in weights, "RAW contains a scaling buffer")
        else:
            require("action_divisors" in weights, "FEATURE_UNITS buffer missing")
            buffer = tensor_array(weights["action_divisors"])
            require(buffer.dtype == np.float32 and np.array_equal(buffer, np.asarray(declared["action_divisors"], np.float32)),
                    "FEATURE_UNITS buffer differs from declaration")
        expected_hash = checkpoint["policy_hash" if key == "policy" else "selected_hash"]
        require(tensor_hash(weights) == expected_hash, "Checkpoint tensor hash differs")


def seeded_trainable_hash(seed: int) -> str:
    """Reconstruct declared fresh tensors without importing the trained policy."""
    import torch
    from torch import nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = nn.Sequential(nn.Linear(21, 16), nn.Tanh(), nn.Linear(16, 1))
        value = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))
    return tensor_hash({**{f"actor.{name}": item for name, item in actor.state_dict().items()},
                        **{f"value.{name}": item for name, item in value.state_dict().items()}})


def check_partitions(panels: dict) -> None:
    seen = set()
    for role in ("optimization", "selection", "final_evaluation", "stress"):
        panel = panels[role]
        require(panel["role"] == role, "World role relabeled")
        seeds = panel["seeds"]
        require(seeds and all(type(seed) is int and seed >= 0 for seed in seeds), "Invalid world seeds")
        require(len(set(seeds)) == len(seeds) and not seen.intersection(seeds), "World partitions overlap")
        seen.update(seeds)


def index_set(points: list, shape: tuple, name: str) -> set[tuple[int, int, int]]:
    require(all(len(point) == 3 and all(type(v) is int and 0 <= v < shape[i] for i, v in enumerate(point))
                for point in points), f"Invalid {name} source indices")
    result = {tuple(point) for point in points}
    require(len(result) == len(points), f"Duplicate {name} source indices")
    return result


def recompute_score(metrics: dict, compartments: dict[str, np.ndarray], volume: float,
                    reward: dict, partial_weight: float) -> dict:
    """Independent set accounting from native source cells, not saved cost sums."""
    target = np.logical_or.reduce(list(compartments.values()))
    removed, exposed_partial = set(), set()
    total, previous_tool = 0., None
    normal_total = target_total = normal_contact = 0
    for step in metrics["history"]:
        cells = index_set(step["removed_indices_native"], target.shape, "removed")
        contacts = index_set(step["contact_indices_native"], target.shape, "contact")
        require(not removed.intersection(cells), "Repeated removal received credit")
        new_partial = contacts - removed - cells - exposed_partial
        target_count = sum(bool(target[cell]) for cell in cells)
        normal_count = len(cells) - target_count
        partial_normal = sum(not target[cell] for cell in new_partial)
        close(step["target_removed_mm3"], target_count * volume, "step target source cells")
        close(step["normal_removed_mm3"], normal_count * volume, "step normal source cells")
        close(step["partial_normal_contact_mm3"], partial_normal * volume, "step distinct partial contacts")
        for key in ("motor_surrogate_delta", "language_surrogate_delta", "partial_motor_contact_surrogate", "partial_language_contact_surrogate"):
            close(step[key], 0., "absent functional evidence contribution")
        distance = float(np.linalg.norm(np.asarray(step["tip_mm"]) - np.asarray(step["entry_mm"])))
        changed = previous_tool is not None and previous_tool != step["tool_id"]
        total += (reward["target_per_mm3"] * target_count * volume
            - reward["normal_per_mm3"] * normal_count * volume
            - partial_weight * reward["normal_per_mm3"] * partial_normal * volume
            - reward["action_cost"] - reward["motion_per_mm"] * 2 * distance - reward["tool_change_cost"] * changed)
        removed.update(cells)
        exposed_partial.update(new_partial)
        target_total += target_count
        normal_total += normal_count
        normal_contact += partial_normal
        previous_tool = step["tool_id"]
    close(metrics["total_reward"], total, "independently recomputed physical score")
    close(metrics["simulated_removed_target_volume_mm3"], target_total * volume, "total target removal")
    close(metrics["simulated_removed_normal_volume_mm3"], normal_total * volume, "total normal removal")
    close(metrics["cumulative_partial_normal_contact_mm3"], normal_contact * volume, "total partial normal contacts")
    close(metrics["modeled_residual_target_volume_mm3"], (np.count_nonzero(target) - target_total) * volume, "residual target")
    require(metrics["clinical_deficit_probability"] is None, "Manufactured clinical probability")
    require(metrics["functional_evidence_available"] == {"motor": False, "language": False}, "Study functional evidence changed")
    require(metrics["motor_surrogate"] is None and metrics["language_surrogate"] is None, "Unknown functional evidence became a measured zero")
    for name, mask in compartments.items():
        count = sum(bool(mask[cell]) for cell in removed)
        close(metrics["removed_by_compartment_mm3"][name], count * volume, f"removed {name}")
        close(metrics["residual_by_compartment_mm3"][name], (np.count_nonzero(mask) - count) * volume, f"residual {name}")
    return {"score": total, "target_removed_mm3": target_total * volume, "normal_removed_mm3": normal_total * volume,
        "cumulative_partial_normal_contact_mm3": normal_contact * volume, "nonstop_actions": len(metrics["history"])}


class Audit:
    def __init__(self, attempt: Path, archive: Path):
        self.attempt, self.archive, self.study = attempt, archive, attempt / "study"
        self.inputs: dict[str, str] = {}

    def read(self, path: Path) -> dict | list:
        payload = path.read_bytes()
        self.inputs[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def checkpoint(self, path: Path) -> dict:
        import torch
        self.inputs[str(path.resolve())] = file_hash(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    def run(self) -> dict:
        # Deliberately open only status until both execution levels finish.
        status = self.read(self.study / "status.json")
        launch_status = self.read(self.attempt / "experiment-status.json")
        require(status["status"] in {"completed", "invalidated"}
                and launch_status["status"] == status["status"], "Study still running or incomplete; outcomes remain unopened")
        declaration_path = Path("manifests/experiments/procedural-native-feature-units-v1.json")
        declaration = self.read(self.archive / declaration_path)
        require(declaration["declaration_content_hash"] == DECLARATION_HASH
                == canonical_hash({k: v for k, v in declaration.items() if k != "declaration_content_hash"}), "Declaration changed")
        reference = self.read(self.archive / declaration["reference_design"]["path"])
        require(reference["declaration_content_hash"] == declaration["reference_design"]["declaration_content_hash"]
                == canonical_hash({k: v for k, v in reference.items() if k != "declaration_content_hash"}), "Physical reference changed")
        snapshot = self.read(self.study / "source.json")
        require(snapshot["numerical_runtime_content_hash"] == RUNTIME_HASH
                == canonical_hash(snapshot["numerical_runtime_sha256"]), "Executed runtime differs from released source")
        for name, digest in snapshot["file_sha256"].items():
            for root in (self.archive, self.attempt / "frozen-source", self.study / "source-snapshot"):
                require(file_hash(root / name) == digest, f"Source snapshot bytes differ: {root / name}")
        for name, digest in reference["source_hashes_at_declaration"].items():
            require(file_hash(self.archive / name) == digest, f"Protected physical source changed: {name}")
        manifest = self.read(self.study / "manifest.json")
        summary = self.read(self.study / "summary.json")
        panels = manifest["world_partitions"]
        check_partitions(panels)
        require(panels == reference["target"]["world_partitions"], "Target world panels changed")
        require(manifest["declaration"] == declaration and manifest["source_hash"] == RUNTIME_HASH, "Run declaration/source binding changed")
        for record in (status, summary, manifest):
            require(record["final_worlds_used"] is False, "Final/stress world execution disclosed")
        for forbidden in ("final_evaluation.json", "stress.json", "ledger.json"):
            require(not any(self.study.rglob(forbidden)), f"Unexpected final/stress artifact: {forbidden}")
        operations = [":".join(str(op[key]) for key in ("phase", "profile", "seed") if key in op)
                      for op in declaration["execution_order"]]
        require(status["attempted_operations"] == status["completed_operations"] == operations, "Execution order/denominator differs")
        planned = [name for name in operations if name.startswith(("scratch:", "adapted:"))]
        require(len(planned) == 12 and status["planned_learning_runs"] == status["completed_learning_runs"] == planned, "Missing learned arm")
        for phase in ("offline", "frozen"):
            require(status[f"planned_{phase}_runs"] == status[f"completed_{phase}_runs"] == list(PROFILES), f"Missing {phase} profile")
        preflight = self.read(self.study / "public-world-preflight.json")
        require(preflight["before_offline_pretraining"] and preflight["gradient_steps"] == 0 and preflight["final_worlds_used"] is False,
                "Invalid pre-execution gate")
        require([(row["input_profile"], row["seed"]) for row in preflight["checks"]] == [(p, s) for p in PROFILES for s in SEEDS], "Incomplete public preflight")
        paired = {row["seed"]: row for row in preflight["paired_initialization"]}
        initial_hashes = {seed: seeded_trainable_hash(seed) for seed in (101, *SEEDS)}
        for seed, expected in initial_hashes.items():
            row = paired[seed]
            require(row["trainable_parameter_hash"] == expected, "Preflight is not fresh seeded initialization")
            require(row["policy_hashes"]["RAW"] != row["policy_hashes"]["FEATURE_UNITS"], "Profiles share an incomplete semantic policy hash")
            require(row["diagnostics"]["RAW"]["critic_physical_value"] == row["diagnostics"]["FEATURE_UNITS"]["critic_physical_value"], "Paired critic initialization differs")
        shared, offline_times = self.audit_offline(declaration, reference, snapshot, initial_hashes)
        arm_times, selected_scores, candidate_hashes = self.audit_online(declaration, reference, panels, initial_hashes, shared)
        frozen = self.read(self.study / "frozen.json")
        require(set(frozen) == set(PROFILES), "Frozen denominator changed")
        for profile, record in frozen.items():
            require(record["gradient_steps"] == record["optimization_environment_steps"] == 0 and record["selection_panel_complete"], "Frozen arm updated or unfinished")
            require(len(record["selection_returns"]) == len(panels["selection"]["seeds"]), "Frozen partial panel")
            close(record["selection_return"], sum(record["selection_returns"]) / len(record["selection_returns"]), "frozen mean")
            require(record["selection_partition_hash"] == canonical_hash(panels["selection"]), "Frozen world split changed")
            label = f"{profile}:PROCEDURAL_PRETRAINED_FROZEN"
            candidate_hashes[label] = shared[profile]["policy_hash"]
            require(record["selected_checkpoint_hash"] == candidate_hashes[label], "Frozen shared identity changed")
            selected_scores[label] = record["selection_return"]
        searches = self.read(self.study / "search.json")
        require([row["method"] for row in searches] == ["GREEDY", "SEARCH"], "Shared baseline count changed")
        for record in searches:
            require(record["reused_by_profiles"] == list(PROFILES) and record["shared_baseline_id"] == record["method"], "Baseline reuse concealed")
            selected_scores[record["method"]] = record["nominal_score"]
        selected_scores["STOP"] = 0.
        scores, rejected = self.audit_candidates(declaration, reference, manifest, status, selected_scores, candidate_hashes)
        return {"status": "passed", "audit_kind": "independent saved-source checkpoint/phase/split/source-cell-score audit",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "implementation_commit": IMPLEMENTATION_COMMIT,
            "runtime_content_hash": RUNTIME_HASH, "declaration_hash": DECLARATION_HASH,
            "denominators": {"offline": 2, "online": 12, "frozen": 2, "shared_baselines": 3, "candidates": 23},
            "online_timing": arm_times, "offline_timing": offline_times, "source_cell_scores": scores,
            "recorded_geometry_rejected_candidates": rejected, "final_worlds_used": False,
            "clinical_deficit_probability": None, "input_file_sha256": self.inputs,
            "limits": ["No policy retraining, policy rollout or final/stress world execution in this audit",
                "Geometry certificates are source-bound and cross-checked, not rerun by this receipt audit",
                "Split closure uses saved artifacts and frozen execution paths; no per-reset execution trace was recorded",
                "Fresh initialization and within-attempt shared copies are checked; tensor equality alone cannot prove process execution",
                "One structural development case, absent functional evidence; no clinical or population inference"]}

    def audit_offline(self, declaration, reference, snapshot, initial_hashes):
        outputs, timings = {}, {}
        for profile in PROFILES:
            folder = self.study / f"pretraining-{profile.lower()}"
            record = self.read(folder / "training/result.json")
            contract = self.read(folder / "training/contract.json")
            initial = self.checkpoint(folder / "training/initial.pt")
            latest = self.checkpoint(folder / "training/checkpoint.pt")
            exported = self.checkpoint(folder / "procedural.pt")
            for checkpoint in (initial, latest, exported):
                check_profile(checkpoint, declaration["profiles"][profile])
            require(tensor_hash(initial["policy"], trainable=True) == initial_hashes[101], "Offline weights not fresh seeded tensors")
            require(contract["config"] == declaration["budgets"]["offline_training"], "Offline budget differs")
            require(exported["policy_hash"] == latest["policy_hash"] == record["latest_checkpoint_hash"], "Fixed-budget latest export changed")
            provenance = exported["procedural_provenance"]
            require(exported["provenance_hash"] == canonical_hash(provenance), "Offline provenance hash changed")
            require(provenance["study_id"] == declaration["study_id"] and provenance["study_declaration_hash"] == DECLARATION_HASH,
                    "Offline checkpoint reused a different study")
            require(provenance["training_run_hash"] == canonical_hash(record), "Offline training receipt changed")
            require(provenance["human_patients_in_pretraining"] == 0 and provenance["final_worlds_used"] is False, "Offline clinical/final-world claim changed")
            sources = {row["source_hash"] for row in reference["procedural_training"]["members"]}
            counts = Counter(source for update in record["optimization_history"] for source in update["episode_source_case_hashes"])
            require(set(counts) == sources and dict(counts) == provenance["gradient_episode_sources"], "Offline source families changed or target leaked")
            for name, source_text in provenance["source_snapshot"].items():
                require(hashlib.sha256(source_text.encode()).hexdigest() == provenance["source_sha256"][name]
                    == snapshot["numerical_runtime_sha256"]["src/resectionlab/" + name], "Offline source snapshot changed")
            check_selection(record, len(contract["partitions"]["selection"]["seeds"]))
            timings[profile] = check_timing(record, {}, declaration["budgets"]["offline_training"])
            timings[profile]["complete_offline_seconds"] = self.read(folder / "procedural-result.json")["total_offline_seconds"]
            copied = self.study / f"{profile.lower()}-procedural-source.pt"
            require(file_hash(copied) == file_hash(folder / "procedural.pt"), "Shared profile did not come from this attempt's offline export")
            outputs[profile] = {"policy_hash": exported["policy_hash"], "file_hash": file_hash(copied)}
        return outputs, timings

    def audit_online(self, declaration, reference, panels, initial_hashes, shared):
        timings, scores, hashes = {}, {}, {}
        rows = self.read(self.study / "training.json")
        require(len(rows) == 12 and len({row["operation_id"] for row in rows}) == 12, "Online denominator/identity changed")
        for row in rows:
            phase, profile, seed = row["phase"], row["input_profile"], row["seed"]
            folder = self.study / f"{profile.lower()}-{phase}-{seed}"
            record = self.read(folder / "result.json")
            require(row == self.read(folder / "arm-record.json"), "Aggregate/per-arm receipt differs")
            contract = self.read(folder / "contract.json")
            require(contract["config"] == {**declaration["budgets"]["online_scratch_and_adapted_each_seed"], "seed": seed}, "Online budget/config differs")
            require(contract["decision_model_hash"] == reference["target"]["decision_model_hash"], "Physical model changed")
            for role in ("optimization", "selection"):
                panel = contract["partitions"][role]
                require(panel["partition_hash"] == canonical_hash(panels[role]), "Online split hash changed")
                for key in ("role", "case_hash", "planning_hash", "generator", "seeds"):
                    require(panel[key] == panels[role][key], f"Online {role} assumptions changed")
            initial = self.checkpoint(folder / "initial.pt")
            checkpoint = self.checkpoint(folder / "checkpoint.pt")
            for state in (initial, checkpoint):
                check_profile(state, declaration["profiles"][profile])
            require(initial["policy_hash"] == record["initial_checkpoint_hash"], "Initial checkpoint identity changed")
            require(checkpoint["selected_hash"] == record["selected_checkpoint_hash"], "Selected policy tensor identity changed")
            require(checkpoint["contract_hash"] == contract["contract_hash"]
                == canonical_hash({k: v for k, v in contract.items() if k not in {
                    "contract_hash", "initial_checkpoint_hash", "shared_checkpoint_hash", "code_sha256", "hardware",
                    "clinical_deficit_probability", "final_evaluation_used_for_optimization"}}), "Frozen learner contract hash changed")
            require(contract["final_evaluation_used_for_optimization"] is False, "Final feedback used")
            check_selection(record, len(panels["selection"]["seeds"]))
            timings[row["operation_id"]] = check_timing(record, row, declaration["budgets"]["online_scratch_and_adapted_each_seed"])
            for key in ("selected_selection_return", "selected_checkpoint_hash", "initial_checkpoint_hash", "gradient_steps", "elapsed_seconds"):
                require(row[key] == record[key], "Aggregated online outcome differs from raw result")
            mode = "PATIENT_SCRATCH_RL" if phase == "scratch" else "PROCEDURAL_PRETRAINED_ADAPTED"
            label = f"{profile}:{mode}:{seed}"
            hashes[label], scores[label] = record["selected_checkpoint_hash"], record["selected_selection_return"]
            if phase == "scratch":
                require(tensor_hash(initial["policy"], trainable=True) == initial_hashes[seed], "Scratch initial reused nonfresh tensors")
                require(contract["procedural_initialization"] is None and record["shared_checkpoint_hash"] is None, "Scratch imported shared policy")
                label = f"{profile}:INITIAL:{seed}"
                hashes[label], scores[label] = initial["policy_hash"], record["initial_selection_return"]
            else:
                require(initial["policy_hash"] == shared[profile]["policy_hash"] == record["shared_checkpoint_hash"], "Adaptation did not clone matching fresh shared profile")
                require(file_hash(folder / "procedural-source.pt") == shared[profile]["file_hash"], "Adaptation reused another attempt's source bytes")
            steps = [float(item["step"]) for item in checkpoint["optimizer"]["state"].values()]
            require(not steps or max(steps) <= record["gradient_steps"], "Optimizer has steps from another run")
        return timings, scores, hashes

    def audit_candidates(self, declaration, reference, manifest, status, expected_scores, checkpoint_hashes):
        freeze = self.read(self.study / "candidate-freeze.json")
        candidates = {row["plan_id"]: row for row in freeze["candidates"]}
        require(len(freeze["candidates"]) == 23 and set(candidates) == expected_candidates(), "Candidate denominator or identities differ")
        require(dict(freeze["candidate_hashes"]) == {name: canonical_hash(value) for name, value in candidates.items()}, "Frozen candidate content changed")
        fields = ("case_hash", "decision_model_fingerprint", "candidate_hashes", "optimization_partition_hash",
            "selection_partition_hash", "development_seeds", "generator_fingerprint", "generator_family", "selection_rule", "frozen_at")
        require(canonical_hash({key: freeze[key] for key in fields}) == freeze["fingerprint"], "Candidate freeze fingerprint changed")
        for role in ("optimization", "selection"):
            require(freeze[role + "_partition_hash"] == canonical_hash(manifest["world_partitions"][role]), "Candidate freeze split differs")
        require(freeze["case_hash"] == reference["target"]["semantic_hash"], "Candidate freeze case differs")
        require(freeze["decision_model_fingerprint"] == manifest["decision_model"]["fingerprint"]
            == canonical_hash({k: v for k, v in manifest["decision_model"].items() if k != "fingerprint"}), "Frozen physical objective/model differs")
        require(freeze["input_profiles"] == declaration["profiles"] and len(freeze["candidate_profiles"]) == 20, "Candidate profile provenance incomplete")
        for name, expected in checkpoint_hashes.items():
            profile = name.split(":")[0]
            require(candidates[name]["selected_checkpoint_hash"] == expected, "Candidate checkpoint substitution")
            require(freeze["candidate_profiles"][name] == {"profile_id": profile,
                "profile_content_hash": declaration["profiles"][profile]["profile_content_hash"], "full_policy_hash": expected}, "Candidate profile identity changed")
        geometry = self.study / status["geometry_validation_run_id"]
        audits, replays = self.read(geometry / "native-history-audit.json"), self.read(geometry / "native-history-replay.json")
        require(set(audits) == set(replays) == set(candidates), "Independent audit omitted a candidate")
        require(self.read(geometry / "manifest.json")["candidate_freeze_hash"] == freeze["fingerprint"], "Geometry inspected an unfrozen candidate set")
        bundle = self.attempt / "source-case.ressectionlab"
        require(file_hash(bundle) == reference["target"]["bundle_sha256"], "Patient source bytes changed")
        with ZipFile(bundle) as source:
            source_manifest = json.loads(source.read("manifest.json"))
            payload = source.read("arrays.npz")
        require(hashlib.sha256(payload).hexdigest() == source_manifest["array_sha256"], "Patient array bytes changed")
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            compartments = {name: arrays[key] for name, key in source_manifest["array_index"]["compartments"].items()}
            volume = abs(float(np.linalg.det(arrays["affine"][:3, :3])))
        require(source_manifest["case_semantic_hash"] == reference["target"]["semantic_hash"], "Source case identity changed")
        seen, scores = set(), {}
        for name, candidate in candidates.items():
            receipt, audit = replays[name], audits[name]
            key = canonical_hash({"model": reference["target"]["decision_model_hash"], "actions": candidate["actions"]})
            require(receipt["audit_key"] == audit["audit_key"] == key, "Independent audit reuse crossed action/model identity")
            require(audit["shared_audit_reused"] == (key in seen), "Independent audit reuse accounting differs")
            seen.add(key)
            metrics = receipt["metrics"]
            require(receipt["checkpoint_hash"] == candidate["selected_checkpoint_hash"]
                and receipt["shared_checkpoint_hash"] == candidate["shared_checkpoint_hash"], "Replayed checkpoint identity differs")
            require(metrics["decision_model_hash"] == reference["target"]["decision_model_hash"], "Replayed physical model differs")
            require(metrics["seed"] == manifest["world_partitions"]["selection"]["seeds"][0], "Native audit opened another world partition")
            require([row["action_id"] for row in metrics["history"]] == [action for action in candidate["actions"] if action != "STOP"], "Replayed action sequence changed")
            if audit["feasible"]:
                require(audit["complete_tool_checked"] and audit["frontier_checked"], "Feasible certificate lacks required geometry gates")
                close(audit["unsupported_source_tissue_volume_mm3"], 0., "Certified unsupported removal")
            require(audit["source_case_hash"] == reference["target"]["semantic_hash"], "Independent certificate uses another patient")
            scores[name] = recompute_score(metrics, compartments, volume, reference["target"]["reward"], reference["target"]["partial_contact_weight"])
            close(scores[name]["score"], expected_scores[name], "Candidate score vs complete deterministic selection")
        rejected = [name for name, record in audits.items() if not record["feasible"]]
        require(set(rejected) == set(status["validation"]["rejected_candidate_ids"]), "Rejected candidate was hidden")
        require(len(seen) == status["validation"]["unique_audits"], "Geometry reuse denominator differs")
        return scores, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output exists; preserve prior receipts")
    audit = Audit(args.attempt.resolve(), args.archive.resolve())
    try:
        result = audit.run()
    except Exception as error:
        result = {"status": "failed", "exception": type(error).__name__, "reason": str(error),
                  "input_file_sha256": audit.inputs, "final_worlds_used": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise
    result["auditor_sha256"] = file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {args.output}")


if __name__ == "__main__":
    main()
