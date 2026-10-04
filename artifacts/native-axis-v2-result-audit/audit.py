#!/usr/bin/env python3
"""Independent read-only audit of completed axis V2 records.

No resectionlab imports, policy inference, simulator, optimizer, or geometry
replay. Source-cell accounting helpers are copied from the independently tested
feature-unit auditor (cc736200...), not from the simulation implementation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import numpy as np

DECLARATION_HASH = "sha256:089dca680b992a5808c104a559ab562a034de9bea91de6df939da40d87650fe3"
RUNTIME_HASH = "sha256:ee931752f4af906c8561a8ea334bf4233fad0dd78b68fdc8f22846c17d4151f3"
COMMIT = "69304172a74c50a8b83025c863edd979ff26e4fd"
INITIAL_TENSOR_HASH = "sha256:e9399769384678821712a540bab184d36f6f38fc0778a30f0d7ef8973022f322"
RAW_PROFILE_HASH = "sha256:b0ae5e34b7fbd206ca7e690fc29c5283ab4be69af25ba45b67f4f5d858b27e12"
EPISODES = ("greedy", "selection-1", "selection-2", "untrained-selection-1", "untrained-selection-2")

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
    step_rewards = []
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
        reward_value = (reward["target_per_mm3"] * target_count * volume
            - reward["normal_per_mm3"] * normal_count * volume
            - partial_weight * reward["normal_per_mm3"] * partial_normal * volume
            - reward["action_cost"] - reward["motion_per_mm"] * 2 * distance - reward["tool_change_cost"] * changed)
        step_rewards.append(reward_value)
        total += reward_value
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
        "cumulative_partial_normal_contact_mm3": normal_contact * volume, "nonstop_actions": len(metrics["history"]),
        "step_rewards": step_rewards}


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": value.dtype.str}, sort_keys=True).encode())
    h.update(value.tobytes())
    return "sha256:" + h.hexdigest()


def action_id(model, proposal, phase):
    payload = json.dumps((model, proposal, phase), sort_keys=True, allow_nan=False).encode()
    return "AXISv1:" + hashlib.sha256(payload).hexdigest()[:24]


def check_publication(launcher, status, payload_bytes):
    require(launcher["status"] == status["status"] == "completed", "Both publication authorities must complete")
    require(launcher["worker_returncode"] == 0 and not launcher["parent_timeout_requested"] and not launcher["hard_killed"],
            "Timeout or failed worker cannot publish")
    require(launcher["eligible_candidate_count"] == status["eligible_candidate_count"] == 2, "Candidate denominator changed")
    require(status["candidate_records_sha256"] == hashlib.sha256(payload_bytes).hexdigest(), "Publication payload hash differs")
    for record in (launcher, status):
        require(record["gradient_steps"] == 0 and record["final_worlds_used"] is False
                and record["stress_worlds_used"] is False, "Training/final/stress scope changed")


def check_inventory(record, model):
    batch = record["batch"]
    require(record["status"] == "complete", "Incomplete inventory")
    offsets, tools = model["proposal_rule"]["offsets_source_voxels"], model["tools"]
    slots = [(i, tuple(offset), tool["tool_id"]) for i, offset in enumerate(offsets) for tool in tools]
    ledger = batch["ledger"]
    require(batch["slot_count"] == len(ledger) == len(slots), "Inventory slot denominator changed")
    require([(row["column_index"], tuple(row["offset_source_voxels"]), row["tool_id"]) for row in ledger] == slots,
            "Incomplete, reordered or duplicated slots")
    require(batch["counts"] == dict(Counter(row["reason"] for row in ledger)), "Provider counts differ")
    require(batch["source_hash"] == model["case_hash"] and batch["engine_model_hash"] == model["native_config_hash"]
        and batch["proposal_model_hash"] == model["proposal_model_hash"], "Inventory source/model changed")
    require(batch["geometry_certified"] is False and batch["removal_authorized"] is False, "Uncertified provider claims clearance")
    require(batch["unsupported_reason"] is None, "Unsupported proposal model")
    proposals = batch["proposals"]
    require([row["proposal_id"] for row in ledger if row["reason"] == "PROPOSED_UNCERTIFIED"]
            == [row["proposal_id"] for row in proposals], "Provider proposal coverage differs")
    require(len({row["proposal_id"] for row in proposals}) == len(proposals), "Duplicate proposal identity")
    attempts = record["attempts"]
    if record["terminated"]:
        require(not attempts and not record["certified_action_ids"], "Terminal inventory must not preview or certify")
        return {"slots": len(slots), "proposals": len(proposals), "previews": 0, "primary": 0,
                "fallback": 0, "certified": 0, "terminal": True, "preview_seconds": 0.}
    cursor, accepted = 0, []
    for proposal in proposals:
        require(proposal["fallback_condition"] == "primary_preview_rejected", "Fallback rule changed")
        for phase in ("primary", "fallback"):
            if phase == "fallback" and proposal["fallback_target_mm"] is None:
                break
            require(cursor < len(attempts), "Uninspected proposed ray")
            attempt = attempts[cursor]; cursor += 1
            require(attempt["status"] == "complete" and type(attempt["feasible"]) is bool, "Incomplete geometry attempt")
            require(attempt["proposal_id"] == proposal["proposal_id"] and attempt["phase"] == phase,
                    "Fallback without primary rejection or reordered preview")
            require(attempt["tool_id"] == proposal["tool_id"] and attempt["entry_mm"] == proposal["entry_mm"]
                    and attempt["tip_mm"] == proposal[phase + "_target_mm"], "Preview geometry differs from proposal")
            require(finite(attempt["elapsed_seconds"]) and attempt["elapsed_seconds"] >= 0, "Invalid preview time")
            if attempt["feasible"]:
                accepted.append(action_id(model["decision_model_hash"], proposal["proposal_id"], phase))
                break
    require(cursor == len(attempts), "Undeclared or forbidden fallback attempt")
    require(accepted == record["certified_action_ids"], "Certified action identities/order differ")
    require(len(accepted) + 1 <= model["max_actions"], "Certified inventory exceeds action cap")
    return {"slots": len(slots), "proposals": len(proposals), "previews": len(attempts),
        "primary": sum(row["phase"] == "primary" for row in attempts),
        "fallback": sum(row["phase"] == "fallback" for row in attempts),
        "certified": len(accepted), "terminal": False,
        "preview_seconds": sum(row["elapsed_seconds"] for row in attempts)}


def check_microstep_accounting(step, tissue):
    shape = tissue.shape
    removed = index_set(step["removed_indices_native"], shape, "action removed")
    contacts = index_set(step["contact_indices_native"], shape, "action contact")
    require(all(tissue[cell] for cell in removed | contacts), "Claimed cells lie outside modeled source tissue")
    micro_removed, micro_contacts = set(), set()
    for micro in step["microsteps"]:
        cells = index_set(micro["removed_indices_native"], shape, "micro removed")
        require(not micro_removed.intersection(cells), "Duplicate microstep removal")
        micro_removed.update(cells)
        micro_contacts.update(index_set(micro["contact_indices_native"], shape, "micro contact"))
    require(micro_removed == removed and micro_contacts == contacts, "Microstep/action source-cell unions differ")
    require(removed <= contacts, "Removed source cell lacks declared active contact")
    return removed, contacts


class Reader:
    def __init__(self):
        self.hashes = {}

    def bytes(self, path):
        path = Path(path)
        packed = Path(str(path) + ".gz")
        data = path.read_bytes() if path.exists() else gzip.decompress(packed.read_bytes())
        if path.exists() and packed.exists():
            require(gzip.decompress(packed.read_bytes()) == data, f"Raw/gzip disagreement: {path}")
        self.hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path):
        return json.loads(self.bytes(path))


def audit(attempt: Path, baseline_dir: Path, bundle: Path, archive: Path):
    read = Reader()
    profile = attempt / "profile"
    launcher, status = read.json(attempt / "launcher-status.json"), read.json(profile / "status.json")
    payload_bytes = read.bytes(profile / "candidate-records.json")
    check_publication(launcher, status, payload_bytes)
    candidates = json.loads(payload_bytes)["candidates"]
    require(set(candidates) == {"frozen_greedy_sequence", "untrained_raw_policy"}, "Candidate identities differ")
    declaration = read.json(attempt / "declaration.json")
    body = dict(declaration); supplied = body.pop("declaration_content_hash")
    require(supplied == canonical_hash(body) == DECLARATION_HASH, "Declaration changed")
    baseline = read.json(baseline_dir / "execution-baseline.json")
    require(baseline["source_commit"] == COMMIT and baseline["declaration_content_hash"] == DECLARATION_HASH,
            "Released baseline identity changed")
    source = read.json(attempt / "launch-source.json")
    worker_source = read.json(profile / "source.json")
    require({k:v for k,v in source.items() if k != "source_root"}
            == {k:v for k,v in worker_source.items() if k != "source_root"}, "Launcher/worker runtime differs")
    require(canonical_hash({"files": source["file_sha256"], "versions": source["runtime_versions"], "python": source["python"]})
            == source["runtime_content_hash"] == RUNTIME_HASH, "Runtime fingerprint differs")
    for name, digest in source["file_sha256"].items():
        require(file_hash(archive / name) == file_hash(attempt / "frozen-source" / name) == digest,
                f"Frozen/archive source changed: {name}")
    require(source["file_sha256"]["scripts/preflight_native_axis.py"] == baseline["runner_sha256"], "Released runner differs")
    require(source["file_sha256"]["manifests/experiments/native-axis-preflight-v2.json"] == baseline["declaration_sha256"],
            "Released declaration bytes differ")
    reference = read.json(archive / declaration["source_reference"]["path"])
    reference_body = dict(reference); reference_hash = reference_body.pop("declaration_content_hash")
    require(canonical_hash(reference_body) == reference_hash == declaration["source_reference"]["declaration_content_hash"],
            "Reference declaration changed")
    target = reference["target"]
    require(file_hash(bundle) == target["bundle_sha256"] == baseline["case_bundle_sha256"], "Source bundle differs")
    with ZipFile(bundle) as z:
        manifest = json.loads(z.read("manifest.json"))
        arrays = np.load(BytesIO(z.read("arrays.npz")), allow_pickle=False)
        compartments = {name: np.array(arrays[key], bool) for name, key in manifest["array_index"]["compartments"].items()}
        affine = np.array(arrays["affine"])
        from scipy.ndimage import binary_fill_holes
        tissue = binary_fill_holes(np.array(arrays["mri"]) != 0) | np.logical_or.reduce(list(compartments.values()))
    require(manifest["case_semantic_hash"] == target["semantic_hash"] and manifest["frame"] == "RAS+", "Case/frame changed")
    for name, mask in compartments.items():
        require(array_hash(mask) == target["compartment_mask_hashes"][name], "Source compartment changed")
    require(array_hash(tissue) == target["tissue_support_hash"], "Modeled source tissue changed")
    volume = abs(float(np.linalg.det(affine[:3,:3])))
    close(volume, target["voxel_volume_mm3"], "source cell volume")
    config = read.json(attempt / "native-configuration-preflight.json")
    require(config["actual"] == declaration["native_configuration"] and config["physical_component_equality"] is True
            and config["different_components"] == ["tissue_support_provenance"], "Native configuration identity differs")
    labels = np.zeros(tissue.shape, np.int16)
    for label, name in enumerate(sorted(compartments), 1):
        require(not np.any(labels[compartments[name]]), "Overlapping source compartments")
        labels[compartments[name]] = label
    physical_arrays = {"tissue_mask": tissue, "target_labels": labels, "affine": affine,
                       "hard_exclusion": np.zeros(tissue.shape, bool)}
    for name, array in physical_arrays.items():
        require(config["actual"]["components"][name] == {"shape":list(array.shape), "dtype":str(array.dtype), "array_digest":array_hash(array)},
                f"Actual native {name} differs from reconstructed source cells")
    model = read.json(profile / "model.json")
    for key, expected in (("native_config_hash", config["actual"]["fingerprint"]), ("case_hash", target["semantic_hash"]),
        ("planning_hash", target["planning_hash"]), ("tools", target["tools"]), ("access", target["access"]),
        ("reward", target["reward"]), ("partial_contact_weight", target["partial_contact_weight"]),
        ("proposal_rule", declaration["action_model"]["proposal_rule"]), ("max_steps", 3), ("max_actions", 27),
        ("input_profile", "RAW"), ("evidence_available", [False, False])):
        require(model[key] == expected, f"Frozen model {key} differs")
    require(model["decision_model_hash"] != target["decision_model_hash"], "Different action model relabeled as prior model")
    preparation = read.json(attempt / "preparation.json")
    seen = set()
    for role, panel in target["world_partitions"].items():
        require(panel["role"] == role and not seen.intersection(panel["seeds"]), "World partitions overlap")
        seen.update(panel["seeds"])
        if role in ("optimization", "selection"):
            require(preparation[role] == panel, "Opened panel differs from declaration")
    generator = preparation["optimization"]["generator"]
    require(generator["translation_scale_mm"] == generator["rotation_scale_deg"] == [0.,0.,0.], "Perturbation model changed")
    seeds = [preparation["optimization"]["seeds"][0], *preparation["selection"]["seeds"], *preparation["selection"]["seeds"]]
    partition_hash = canonical_hash(preparation["selection"])
    freeze = read.json(profile / "sequence-freeze.json")
    freeze_body = dict(freeze); freeze_hash = freeze_body.pop("freeze_hash")
    require(freeze_hash == canonical_hash(freeze_body) and freeze["source_hash"] == RUNTIME_HASH
            and freeze["decision_model_hash"] == model["decision_model_hash"]
            and freeze["selection_partition_hash"] == partition_hash, "Sequence freeze differs")
    policy = read.json(profile / "untrained-policy-freeze.json")
    require(policy["seed"] == 11 and policy["hidden_features"] == 16 and policy["gradient_steps"] == 0
            and policy["optimizer_constructed"] is False and policy["decision_model_hash"] == model["decision_model_hash"]
            and policy["selection_partition_hash"] == partition_hash, "Initial policy contract differs")
    require(policy["input_profile"]["profile_id"] == "RAW" and canonical_hash(policy["input_profile"]) == policy["input_profile_hash"] == RAW_PROFILE_HASH,
            "RAW profile changed")
    import torch
    checkpoint_path = profile / "untrained-raw-initial.pt"
    require(file_hash(checkpoint_path) == policy["checkpoint_sha256"], "Frozen initial policy file changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["dimensions"] == [15,6,16] or checkpoint["dimensions"] == (15,6,16), "Policy dimensions differ")
    require(set(checkpoint["policy"]) == {f"{head}.{layer}.{parameter}" for head in ("actor","value")
            for layer in ("0","2") for parameter in ("weight","bias")}, "Unexpected learned parameters/buffers")
    require(tensor_hash(checkpoint["policy"]) == checkpoint["policy_hash"] == policy["policy_hash"] == INITIAL_TENSOR_HASH,
            "Policy no longer matches independently verified fresh seed11 RAW tensors")
    require(checkpoint["input_profile"] == policy["input_profile"] and checkpoint["input_profile_hash"] == policy["input_profile_hash"],
            "Checkpoint profile differs from frozen profile")
    initial = read.json(profile / "initial/inventory.json")
    inventory_summaries = [check_inventory(initial["complete_inventory"], model)]
    require(initial["action_ids"] == ["STOP", *initial["complete_inventory"]["certified_action_ids"]], "Initial observation inventory differs")
    require(len(initial["action_features"]) == len(initial["action_mask"]) == len(initial["action_ids"])
            and all(initial["action_mask"]), "Initial observation shape/mask differs")
    require(np.asarray(initial["action_features"]).shape == (len(initial["action_ids"]),15)
            and np.asarray(initial["state_features"]).shape == (6,)
            and np.isfinite(initial["action_features"]).all() and np.isfinite(initial["state_features"]).all()
            and len(initial["nominal_action_values"]) == len(initial["action_ids"])
            and np.isfinite(initial["nominal_action_values"]).all(), "Recorded initial observation/values malformed")
    first_greedy_choice = initial["action_ids"][int(np.argmax(initial["nominal_action_values"]))]
    episodes, costs = {}, {}
    for name, expected_seed in zip(EPISODES, seeds):
        episode = read.json(profile / name / "episode.json")
        metrics = episode["metrics"]
        require(episode["status"] == "completed" and metrics["seed"] == expected_seed, "Episode incomplete or world changed")
        require(episode["actions"] == metrics["actions"] == [row["action_id"] for row in metrics["history"]], "Actual action sequence differs")
        require(len(episode["actions"]) == metrics["environment_steps"] == 3 and metrics["termination_reason"] == "step_budget",
                "Actual episode horizon/denominator differs")
        require(metrics["decision_model_hash"] == model["decision_model_hash"] and metrics["native_engine_config_hash"] == model["native_config_hash"]
                and metrics["proposal_model_hash"] == model["proposal_model_hash"] and metrics["world_generator"] == generator,
                "Episode model/world changed")
        inventories = metrics["inventory_receipts"]
        require(len(inventories) == len(metrics["history"]) + 1 and inventories[-1]["terminated"] is True, "State inventory coverage incomplete")
        summaries = [check_inventory(row, model) for row in inventories]
        inventory_summaries.extend(summaries)
        close(metrics["proposal_accounting"]["preview_calls"], sum(row["previews"] for row in summaries), "actual preview denominator")
        close(metrics["proposal_accounting"]["preview_seconds"], sum(row["preview_seconds"] for row in summaries), "summed preview time")
        all_removed, all_contacts = set(), set()
        for index, step in enumerate(metrics["history"]):
            inventory = inventories[index]
            require(step["action_id"] in inventory["certified_action_ids"] and step["source_state_hash"] == inventory["batch"]["cavity_state_hash"],
                    "Executed action lacks current-state certificate")
            require(step["source_hash"] == target["semantic_hash"] and step["decision_model_hash"] == model["native_config_hash"]
                and step["tissue_support_provenance"] == config["actual"]["components"]["tissue_support_provenance"]
                and np.array_equal(step["native_affine"], affine), "Executed native geometry source changed")
            matching = [row for row in inventory["attempts"] if row["proposal_id"] == step["axis_proposal"]["proposal_id"] and row["feasible"]]
            require(len(matching) == 1, "Executed phase lacks unique feasible preview")
            expected = {k:v for k,v in matching[0].items() if k != "elapsed_seconds"}
            expected.update(cavity_state_hash=inventory["batch"]["cavity_state_hash"], proposal_model_hash=model["proposal_model_hash"])
            require(step["axis_proposal"] == expected, "Executed phase/proposal differs")
            removed, contacts = check_microstep_accounting(step, tissue)
            require(not all_removed.intersection(removed), "Source tissue removed twice")
            all_removed.update(removed); all_contacts.update(contacts)
            close(step["removed_volume_mm3"], len(removed)*volume, "native action removal volume")
        score = recompute_score(metrics, compartments, volume, model["reward"], model["partial_contact_weight"])
        close(episode["total_reward"], score["score"], "episode reported reward")
        close(sum(row["reward"] for row in episode["transitions"]), score["score"], "transition reward sum")
        for row, expected_reward in zip(episode["transitions"], score["step_rewards"]):
            close(row["reward"], expected_reward, "per-transition source-cell reward")
        require([row["action_id"] for row in episode["transitions"]] == episode["actions"]
                and [row["terminated"] for row in episode["transitions"]] == [False,False,True]
                and all(row["committed"] for row in episode["transitions"]), "Transition/commit coverage differs")
        timing = read.json(profile / name / "timing.json")
        require(timing["episode_seconds_including_export"] >= episode["episode_seconds_before_export"] >=
            sum(row["decision_seconds"] + row["transition_and_next_inventory_seconds"] for row in episode["transitions"]), "Episode timing scope inconsistent")
        require(read.json(profile / name / "progress.json")["transitions"] == episode["transitions"], "Progress/final transitions differ")
        episodes[name] = {**score, "history_hash": canonical_hash(metrics["history"]), "actions": episode["actions"],
            "native_removed_mm3": len(all_removed)*volume, "retained_contact_mm3": len(all_contacts-all_removed)*volume,
            "episode_world_hash": metrics["episode_world_hash"], "inventory_summaries": summaries}
        costs[name] = {**timing, "decision_seconds": sum(row["decision_seconds"] for row in episode["transitions"]),
            "transition_seconds": sum(row["transition_and_next_inventory_seconds"] for row in episode["transitions"]),
            "reset_local_integrity_seconds": metrics["proposal_accounting"]["integrity_seconds"],
            "preview_seconds": metrics["proposal_accounting"]["preview_seconds"]}
    require(freeze["actions"] == episodes["greedy"]["actions"] and freeze["greedy_history_hash"] == episodes["greedy"]["history_hash"],
            "Greedy freeze differs from executed episode")
    require(episodes["greedy"]["actions"][0] == first_greedy_choice, "Initial greedy choice differs from saved complete nominal values")
    certificates = (read.json(profile / "independent-audit.json"), read.json(profile / "untrained-independent-audit.json"))
    groups = (("greedy", "selection-1", "selection-2"), ("untrained-selection-1", "untrained-selection-2"))
    for cert, group in zip(certificates, groups):
        history = {episodes[name]["history_hash"] for name in group}
        require(len(history) == 1 and cert["shared_exact_history_hash"] in history
                and cert["certified_complete_episodes"] == len(group), "Certificate history/episode coverage differs")
        check = cert["audit"]
        require(check["feasible"] is True and check["complete_tool_checked"] is True and check["frontier_checked"] is True
                and not check["failures"] and check["first_failed_action"] is None, "Independent certificate incomplete or failed")
        require(check["source_case_hash"] == target["semantic_hash"] and check["action_count"] == 3, "Certificate source/action scope differs")
        for key in ("claimed_source_tissue_volume_mm3", "contained_source_tissue_volume_mm3"):
            close(check[key], episodes[group[0]]["native_removed_mm3"], "certificate native cell volume")
        close(check["unsupported_source_tissue_volume_mm3"], 0., "unsupported certified tissue")
        close(check["source_voxel_volume_mm3"], volume, "certificate source voxel volume")
    require(len({episodes[name]["history_hash"] for name in EPISODES}) == status["unique_native_audits"] == 2
            and certificates[1]["reused_greedy_certificate"] is False and status["certified_complete_episodes"] == 5,
            "Distinct geometry audit denominator differs")
    panels = []
    for candidate_name, panel_file, names, mean_key in (
            ("frozen_greedy_sequence", "selection-progress.json", ("selection-1", "selection-2"), "mean_selection_return"),
            ("untrained_raw_policy", "untrained-selection-progress.json", ("untrained-selection-1", "untrained-selection-2"), "untrained_mean_selection_return")):
        panel = read.json(profile / panel_file)
        candidate = candidates[candidate_name]
        require(panel["status"] == "complete" and panel["partition"] == preparation["selection"]
                and len(panel["worlds"]) == 2 and panel["worlds"] == candidate["selection_worlds"], "Incomplete/mismatched candidate panel")
        require(candidate["decision_model_hash"] == model["decision_model_hash"]
                and candidate["selection_partition_hash"] == partition_hash, "Candidate model/panel differs")
        for row, name, seed in zip(panel["worlds"], names, preparation["selection"]["seeds"]):
            require(row["seed"] == seed and row["history_hash"] == episodes[name]["history_hash"]
                    and row["episode_world_hash"] == episodes[name]["episode_world_hash"], "Panel world/history identity differs")
            close(row["return"], episodes[name]["score"], "panel return")
            close(row["episode_seconds"], costs[name]["episode_seconds_including_export"], "panel episode time")
            costs[name]["reset_seconds"] = row["reset_seconds"]
        mean = float(np.mean([episodes[name]["score"] for name in names]))
        close(status[mean_key], mean, "final authority return")
        close(candidate["mean_selection_return"], mean, "candidate mean")
        require(candidate["independent_history_hash"] == episodes[names[0]]["history_hash"], "Published history differs from audit")
        panels.append(sum(costs[name]["reset_seconds"] + costs[name]["episode_seconds_including_export"] for name in names))
    require(candidates["frozen_greedy_sequence"]["freeze_hash"] == freeze_hash
            and candidates["frozen_greedy_sequence"]["actions"] == freeze["actions"], "Published sequence differs from freeze")
    require(all(candidates["untrained_raw_policy"][key] == value for key,value in policy.items()), "Published initial policy differs from freeze")
    cold = read.json(profile / "cold-setup.json")
    clone = read.json(profile / "clone-probe.json")
    initial_inventory_time = initial["inspection_seconds"]
    require(clone["model_hash"] == model["decision_model_hash"]
            and clone["cavity_state_hash"] == initial["complete_inventory"]["batch"]["cavity_state_hash"], "Clone identity differs")
    greedy_reset = read.json(profile / "greedy-reset.json")
    require(greedy_reset["optimization_seed"] == seeds[0]
            and greedy_reset["partition_hash"] == canonical_hash(preparation["optimization"]), "Greedy reset differs")
    costs["greedy"]["reset_seconds"] = greedy_reset["seconds"]
    audit_seconds = sum(cert["seconds"] for cert in certificates)
    close(status["independent_audit_seconds"], audit_seconds, "independent geometry time")
    accounted_profile = (cold["factory_seconds"] + initial_inventory_time + clone["seconds"]
        + policy["initialization_seconds"] + audit_seconds
        + sum(row["reset_seconds"] + row["episode_seconds_including_export"] for row in costs.values()))
    require(status["full_preflight_seconds"] >= accounted_profile, "Phase timings overlap or exceed whole profile")
    resource = read.json(attempt / "worker-resource.json")
    require(resource["budget"] == declaration["resource_budget"] == launcher["resource_budget"], "Resource cap changed")
    require(resource["cancellation_reason"] is None and resource["observed_peak_rss_bytes"] <= resource["budget"]["process_peak_rss_bytes"]
            and resource["elapsed_seconds"] < resource["budget"]["worker_wall_seconds"], "Completed attempt exceeded declared bounds")
    require(launcher["full_launcher_seconds"] >= resource["elapsed_seconds"] >= status["full_preflight_seconds"], "Timing hierarchy differs")
    probe = read.json(profile / "cancellation-probe.json")
    require(probe["committed_state_unchanged"] is True, "Cancellation probe changed state")
    forbidden_files = [p.name for p in profile.iterdir() if any(part in p.name for part in ("final-evaluation", "stress", "optimizer"))]
    require(not forbidden_files, "Unexpected final/stress/optimizer output artifacts")
    totals = {key: sum(row[key] for row in inventory_summaries) for key in ("slots", "proposals", "previews", "primary", "fallback", "certified")}
    return {"status": "passed", "source_commit": COMMIT, "declaration_content_hash": DECLARATION_HASH,
        "runtime_content_hash": RUNTIME_HASH, "runtime_versions": source["runtime_versions"],
        "frozen_source_files_checked": len(source["file_sha256"]), "model": model,
        "verified_episodes": episodes, "inventory_totals": totals,
        "inventory_scope": "constructor plus actual state inventories for five episodes; terminal provider rays remain unpreviewed at horizon",
        "costs": {"episode_phases": costs, "frozen_sequence_panel_seconds": panels[0], "untrained_policy_panel_seconds": panels[1],
            "constructor_seconds": cold["factory_seconds"], "initial_inventory_inspection_seconds": initial_inventory_time,
            "initial_clone_seconds": clone["seconds"], "initial_policy_setup_seconds": policy["initialization_seconds"],
            "independent_geometry_seconds": audit_seconds, "profile_seconds": status["full_preflight_seconds"],
            "profile_unattributed_overhead_seconds": status["full_preflight_seconds"] - accounted_profile,
            "worker_seconds": resource["elapsed_seconds"], "launcher_seconds": launcher["full_launcher_seconds"],
            "process_peak_rss_bytes": resource["observed_peak_rss_bytes"],
            "scope": "preview/integrity subtimers overlap reset/episode intervals; do not add them to phase wall time. Reset-local integrity counters omit some pre-reset checks and are not a global count."},
        "input_sha256": read.hashes, "checkpoint_sha256": file_hash(checkpoint_path),
        "initial_saved_value_argmax_verified": True,
        "completed_episodes": 5, "unique_certified_histories": 2, "eligible_development_records": 2,
        "gradients": 0, "final_worlds": 0, "stress_worlds": 0,
        "limitations": [
            "This auditor verified existing source-bound independent geometry certificates; it did not rerun geometry, policies or episodes.",
            "Later decision feature matrices/logits/nominal values were not persisted. Choice semantics are supported by frozen source and unchanged checkpoint identity, not independently replayed argmax calculations.",
            "Known fresh seed11 tensor identity was independently reconstructed in the earlier feature-unit audit; this run compares checkpoint tensor bytes to that identity without creating or running a model.",
            "File provenance and stored flags cannot independently attest every process operation; observed seeds/episodes, source gates and saved tensors are consistent with zero gradients and closed final/stress worlds.",
            "Deterministic replay is not Monte Carlo uncertainty or independent clinical validation; source support/access remain unreviewed and motor/language evidence absent.",
            "Episode panels reset one simulator and exclude generic learner clone/optimizer/backpropagation costs; no training budget or throughput claim follows."],
        "next_declared_logging_requirement": "Before a future training run, persist at every decision the ordered action IDs, mask, nominal actor observation arrays or their lossless artifact references, policy/profile/checkpoint hash, logits for policy or complete nominal values for greedy, and chosen index/ID. Record their serialization cost explicitly; keep latent world variables out of actor-observation logs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh audit output required")
    try:
        result = audit(args.attempt, args.baseline, args.bundle, args.archive)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    result.update(auditor_sha256=file_hash(Path(__file__)), created_utc=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:result[key] for key in ("status", "error_type", "error", "inventory_totals") if key in result}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
