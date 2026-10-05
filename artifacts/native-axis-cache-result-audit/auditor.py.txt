#!/usr/bin/env python3
"""Read-only independent audit of the four-phase fixed-trace cache experiment.

No production resectionlab import, cache query, geometry check, policy, simulator
or optimizer is executed. Source-cell masks are reconstructed from saved cells;
connected free space uses an independent full six-neighbor exterior flood.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECLARATION_HASH = "sha256:90b245b72f84acac1cbe32cb338d64e47222fd75b2f20673439dbb34e6e4f9d6"
HELPER = ROOT / "artifacts/native-axis-v2-result-audit/audit.py"
HELPER_SHA256 = "8390d427f70635549554c3935fb4b77a3027baa39994cb41af2dd443c303b346"
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA256:
    raise RuntimeError("Independent source-cell audit helper changed")
_spec = importlib.util.spec_from_file_location("independent_cache_audit_helpers", HELPER)
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)
AuditError, Reader = _h.AuditError, _h.Reader
require, close, finite = _h.require, _h.close, _h.finite
canonical_hash, file_hash, array_hash = _h.canonical_hash, _h.file_hash, _h.array_hash
MODES = ("reference_before", "cached_cold", "cached_warm", "reference_after")
NAMESPACE_FIELDS = ("cache_version", "geometry_version", "native_version", "geometry_source_sha256",
    "source_guard_sha256", "mode", "native_config_hash", "source_hash", "geometry_epsilon",
    "guarded_direct_geometry_callables", "max_payload_bytes", "max_entries")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strip_inventory_timing(receipt):
    result = copy.deepcopy(receipt)
    for attempt in result["attempts"]:
        attempt.pop("elapsed_seconds", None)
    return result


def check_authorities(read, output):
    names = ("launcher-status.json", "worker-status.json", "result.json")
    rows = [read.json(output / name) if (output / name).exists() else None for name in names]
    launcher, worker, result = rows
    if not all(row is not None and row.get("status") == "completed" for row in rows):
        require(launcher is None or launcher.get("status") != "completed", "Completed launcher lacks completed worker/result")
        return None, {"audit_status": "incomplete_attempt_retained", "scientific_equivalence_verified": False,
                      "authorities": dict(zip(names, rows)), "interpretation": "Partial phase files are retained diagnostics, not a completed benchmark."}
    require(launcher["worker_returncode"] == 0 and launcher["parent_timeout_requested"] is False
            and launcher["hard_killed"] is False and launcher["receipt_error"] is None, "Worker failed or timed out")
    require(worker["result_hash"] == canonical_hash(result), "Result does not match worker authority")
    require(worker["runtime_content_hash"] == result["runtime_content_hash"]
            and worker["scientific_model_hash"] == result["scientific_model_hash"], "Authority source/model differs")
    require(result["gradient_updates"] == 0 and result["final_worlds_used"] is False
            and result["stress_worlds_used"] is False and result["clinical_deficit_probability"] is None,
            "Gradient/final/stress/clinical scope differs")
    return result, dict(zip(names, rows))


def literal_constant(source: Path, name: str):
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AuditError("Missing literal source constant: " + name)


def check_sources(read, output, baseline_path, archive, declared):
    body = dict(declared); supplied = body.pop("declaration_content_hash")
    require(supplied == canonical_hash(body) == DECLARATION_HASH, "Cache declaration changed")
    baseline = read.json(baseline_path)
    require(baseline["declaration_content_hash"] == DECLARATION_HASH, "Released declaration differs")
    source = read.json(output / "launch-source.json")
    require(canonical_hash({"files": source["file_sha256"], "versions": source["runtime_versions"], "python": source["python"]})
            == source["runtime_content_hash"], "Frozen runtime hash differs")
    for name, digest in source["file_sha256"].items():
        require(file_hash(output / "frozen-source" / name) == file_hash(archive / name) == digest,
                "Frozen execution source differs: " + name)
    for name, digest in declared["bound_input_file_sha256"].items():
        require(source["file_sha256"][name] == digest, "Bound original V2 input differs")
    require(source["file_sha256"]["src/resectionlab/experimental_capsule_cache.py"] == declared["cache"]["tested_helper_sha256"],
            "Tested cache prototype differs")
    require(source["file_sha256"]["scripts/probe_public_native_capsule_cache.py"] == baseline["runner_sha256"], "Released runner differs")
    if "runtime_content_hash" in baseline:
        require(source["runtime_content_hash"] == baseline["runtime_content_hash"], "Released runtime differs")
    return source, baseline


def check_inventory(record, model):
    """Audit all ordered proposal slots, including rejections and conditional fallbacks."""
    batch = record["batch"]
    offsets, tools = model["proposal_rule"]["offsets_source_voxels"], model["tools"]
    slots = [(index, offset, tool["tool_id"]) for index, offset in enumerate(offsets) for tool in tools]
    require(record["status"] == "complete" and batch["slot_count"] == len(slots) == len(batch["ledger"]), "Incomplete inventory slots")
    require([(row["column_index"], row["offset_source_voxels"], row["tool_id"]) for row in batch["ledger"]] == slots,
            "Inventory slots changed or reordered")
    counts = {}
    for row in batch["ledger"]:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    require(batch["counts"] == counts and batch["source_hash"] == model["case_hash"]
            and batch["engine_model_hash"] == model["native_config_hash"]
            and batch["proposal_model_hash"] == model["proposal_model_hash"], "Inventory counts/source/model differs")
    require(batch["geometry_certified"] is False and batch["removal_authorized"] is False
            and batch["unsupported_reason"] is None, "Uncertified proposal batch claims clearance")
    proposed = [row["proposal_id"] for row in batch["ledger"] if row["reason"] == "PROPOSED_UNCERTIFIED"]
    require(proposed == [row["proposal_id"] for row in batch["proposals"]] and len(proposed) == len(set(proposed)), "Proposed ray coverage differs")
    if record["terminated"]:
        require(record["attempts"] == [] and record["certified_action_ids"] == [], "Terminal inventory generated previews")
        return {"slots": len(slots), "previews": 0, "rejected_previews": 0, "fallback_previews": 0, "certified": 0}
    cursor, accepted = 0, []
    for proposal in batch["proposals"]:
        require(proposal["fallback_condition"] == "primary_preview_rejected", "Fallback policy differs")
        for phase in ("primary", "fallback"):
            if phase == "fallback" and proposal["fallback_target_mm"] is None:
                break
            require(cursor < len(record["attempts"]), "Proposed ray was not checked")
            attempt = record["attempts"][cursor]; cursor += 1
            require(attempt["status"] == "complete" and type(attempt["feasible"]) is bool
                    and attempt["proposal_id"] == proposal["proposal_id"] and attempt["phase"] == phase,
                    "Incomplete preview or fallback without primary rejection")
            require(attempt["tool_id"] == proposal["tool_id"] and attempt["entry_mm"] == proposal["entry_mm"]
                    and attempt["tip_mm"] == proposal[phase + "_target_mm"], "Preview geometry differs from proposal")
            if attempt["feasible"]:
                accepted.append(_h.action_id(model["decision_model_hash"], proposal["proposal_id"], phase))
                break
    require(cursor == len(record["attempts"]) and accepted == record["certified_action_ids"], "Accepted actions or preview denominator differs")
    return {"slots": len(slots), "previews": cursor, "certified": len(accepted),
            "rejected_previews": sum(not row["feasible"] for row in record["attempts"]),
            "fallback_previews": sum(row["phase"] == "fallback" for row in record["attempts"])}


def mask_digest(mask):
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def exterior_free_mask(remaining):
    from scipy.ndimage import binary_propagation, generate_binary_structure
    free = ~remaining
    boundary = np.zeros_like(free)
    for axis in range(3):
        for position in (0, -1):
            selection = [slice(None)] * 3
            selection[axis] = position
            boundary[tuple(selection)] = free[tuple(selection)]
    return binary_propagation(boundary, structure=generate_binary_structure(3, 1), mask=free)


def certificate_digest(record):
    """Reconstruct the documented native ancestry checksum, not geometric clearance."""
    value = copy.deepcopy(record)
    groups = [record["removed_indices_native"], record["contact_indices_native"]]
    descriptors = [("int64", [len(points), 3]) for points in groups]
    descriptors.append(("float64", [4, 4]))
    for key in ("removed_indices_native", "contact_indices_native"):
        descriptors.extend(("int64", [len(row[key]), 3]) for row in record["microsteps"])
    value["array_descriptors"] = descriptors
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_trace(trace, original, model, declaration, tissue, affine, compartments):
    metrics, snapshots = trace["metrics"], trace["snapshots"]
    expected = declaration["v2_model"]
    require(metrics["actions"] == expected["fixed_actions"] == original["actions"]
            and canonical_hash(metrics["history"]) == expected["greedy_history_hash"]
            and metrics["seed"] == expected["optimization_seed"], "Fixed action/history/world differs")
    require(metrics["decision_model_hash"] == expected["decision_model_hash"]
            and metrics["native_engine_config_hash"] == expected["native_config_hash"]
            and metrics["proposal_model_hash"] == expected["proposal_model_hash"]
            and metrics["world_generator"] == original["metrics"]["world_generator"]
            and metrics["episode_world_hash"] == original["metrics"]["episode_world_hash"], "Scientific model/world changed")
    require(len(snapshots) == len(metrics["inventory_receipts"]) == 4 and len(trace["transitions"]) == len(metrics["history"]) == 3,
            "Four complete inventories and three cuts required")
    require(metrics["inventory_receipts"] == [strip_inventory_timing(row) for row in original["metrics"]["inventory_receipts"]],
            "Ordered inventory ledgers differ from original V2")
    require(metrics["total_reward"] == original["total_reward"] and metrics["termination_reason"] == "step_budget",
            "Fixed score or episode horizon differs")
    removed, contacts, partial = np.zeros_like(tissue), np.zeros_like(tissue), np.zeros_like(tissue)
    state = "sha256:" + hashlib.sha256((expected["native_config_hash"] + ":initial").encode()).hexdigest()
    inventory_counts = []
    for index, snapshot in enumerate(snapshots):
        inventory = snapshot["inventory"]
        require(inventory == metrics["inventory_receipts"][index], "Snapshot and metrics inventory differ")
        inventory_counts.append(check_inventory(inventory, model))
        require(snapshot["action_ids"] == ["STOP", *inventory["certified_action_ids"]]
                and set(snapshot["certificates"]) == set(inventory["certified_action_ids"])
                and snapshot["action_mask"] == [True] * len(snapshot["action_ids"]), "Observation/certificate inventory differs")
        require(np.asarray(snapshot["action_features"]).shape == (len(snapshot["action_ids"]), 15)
                and np.asarray(snapshot["state_features"]).shape == (6,)
                and np.isfinite(snapshot["action_features"]).all() and np.isfinite(snapshot["state_features"]).all(), "Observation shape/value differs")
        require(snapshot["cavity_state_hash"] == inventory["batch"]["cavity_state_hash"] == state, "Cavity ancestry differs")
        remaining = tissue & ~removed
        expected_masks = {"remaining_mask": remaining, "removed_mask": removed, "contact_mask": contacts,
                          "partial_contact_mask": partial, "connected_free_mask": exterior_free_mask(remaining)}
        require(snapshot["mask_digests"] == {name: mask_digest(mask) for name, mask in expected_masks.items()},
                "Saved mask differs from independently reconstructed source cells/exterior flood")
        for identifier, text in snapshot["certificates"].items():
            cert = json.loads(text)
            require(canonical(cert) == text and cert["source_hash"] == model["case_hash"]
                    and cert["decision_model_hash"] == model["native_config_hash"] and cert["source_state_hash"] == state
                    and np.array_equal(cert["native_affine"], affine) and cert["clinical_deficit_probability"] is None,
                    "Accepted preview certificate provenance differs")
            cells, touched = _h.check_microstep_accounting(cert, tissue)
            require(cells and all(remaining[cell] for cell in cells), "Preview removes missing/already removed source cells")
            close(cert["removed_volume_mm3"], len(cells) * abs(float(np.linalg.det(affine[:3, :3]))), "Preview removed source-cell volume")
        if index == 3:
            continue
        action = expected["fixed_actions"][index]
        require(action in snapshot["certificates"], "Executed action lacks complete current-state certificate")
        cert = json.loads(snapshot["certificates"][action])
        committed = metrics["history"][index]
        require(all(committed.get(key) == value for key, value in cert.items()), "Committed native history differs from accepted preview")
        cells, touched = _h.check_microstep_accounting(committed, tissue)
        new_partial = {cell for cell in touched if remaining[cell] and cell not in cells and not partial[cell]}
        for cell in cells: removed[cell] = True
        for cell in touched: contacts[cell] = True
        for cell in new_partial: partial[cell] = True
        state = "sha256:" + hashlib.sha256((state + ":committed:" + certificate_digest(cert)).encode()).hexdigest()
    require([row["previews"] for row in inventory_counts] == declaration["fixed_work"]["preview_attempts_per_inventory"],
            "Declared fixed inventory work differs")
    score = _h.recompute_score(metrics, compartments, abs(float(np.linalg.det(affine[:3, :3]))),
                               model["reward"], model["partial_contact_weight"])
    require([row["action_id"] for row in trace["transitions"]] == expected["fixed_actions"]
            and [row["terminated"] for row in trace["transitions"]] == [False, False, True], "Fixed transitions differ")
    for actual, reward in zip(trace["transitions"], score["step_rewards"]):
        close(actual["reward"], reward, "Independent transition reward")
    return {"score": score, "inventory_counts": inventory_counts, "source_mask_states_verified": 4,
            "accepted_preview_certificates": sum(row["certified"] for row in inventory_counts),
            "removed_mm3": int(removed.sum()) * abs(float(np.linalg.det(affine[:3, :3])))}


def check_cache_stats(stats, declaration, source, constants):
    settings = declaration["cache"]
    namespace = {key: stats[key] for key in NAMESPACE_FIELDS}
    require(stats["namespace"] == "sha256:" + hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest(), "Cache namespace checksum differs")
    require(stats["cache_version"] == settings["version"] and stats["native_config_hash"] == declaration["v2_model"]["native_config_hash"]
            and stats["source_hash"] == declaration["source_case"]["semantic_hash"] and stats["mode"] == "exact_orthogonal_cells"
            and stats["max_payload_bytes"] == settings["max_payload_bytes"] and stats["max_entries"] == settings["max_entries"], "Cache scientific/source/limit identity differs")
    require(stats["geometry_source_sha256"] == source["file_sha256"]["src/resectionlab/geometry.py"]
            and stats["source_guard_sha256"] == source["file_sha256"]["src/resectionlab/native_proposals.py"], "Cache implementation source differs")
    require(all(stats[key] == value for key, value in constants.items()), "Cache geometry version/tolerance differs")
    require(all(type(stats[key]) is int and stats[key] >= 0 for key in
                ("calls", "hits", "misses", "evictions", "bypasses", "entries", "retained_payload_bytes")), "Invalid cache counters")
    require(stats["calls"] == stats["hits"] + stats["misses"]
            and stats["entries"] == stats["misses"] - stats["bypasses"] - stats["evictions"]
            and stats["entries"] <= settings["max_entries"] and stats["retained_payload_bytes"] <= settings["max_payload_bytes"],
            "Cache hit/miss/eviction accounting or bound differs")
    require(all(stats[key] is False for key in ("feasibility_cached", "cavity_cached", "integrity_cached")), "Cache scope extended to scientific decisions")
    require(finite(stats["query_seconds"]) and finite(stats["compute_seconds"])
            and stats["query_seconds"] >= stats["compute_seconds"] >= 0., "Cache nested timer differs")


def check_cache_sequence(phases, construction, declaration, source, constants):
    require([row["mode"] for row in phases] == list(MODES), "Cache experiment order differs")
    before, cold, warm, after = phases
    require(before["cache_before"] is None and before["cache_after"] is None, "Reference warmed a cache")
    empty = construction["stats"]
    require(empty == cold["cache_before"] and all(empty[key] == 0 for key in
        ("calls", "hits", "misses", "entries", "retained_payload_bytes", "evictions", "bypasses", "query_seconds", "compute_seconds")), "Cold cache not empty")
    require(cold["cache_after"] == warm["cache_before"] and warm["cache_after"] == after["cache_before"] == after["cache_after"],
            "Cache continuity changed or reference-after used cache")
    for stats in (empty, cold["cache_after"], warm["cache_after"]):
        check_cache_stats(stats, declaration, source, constants)
    deltas = {}
    for phase in (cold, warm):
        one, two = phase["cache_before"], phase["cache_after"]
        delta = {key: two[key] - one[key] for key in ("calls", "hits", "misses", "evictions", "bypasses", "query_seconds", "compute_seconds")}
        require(all(value >= 0 for value in delta.values()), "Cumulative cache counters decreased")
        deltas[phase["mode"]] = delta
    return deltas


def read_case(read, bundle, archive, declared, preparation):
    require(file_hash(bundle) == declared["source_case"]["bundle_sha256"], "Case bundle differs")
    target = read.json(archive / "manifests/experiments/procedural-native-to-ucsf-v1.json")["target"]
    v2 = read.json(archive / "manifests/experiments/native-axis-preflight-v2.json")
    with ZipFile(bundle) as zipped:
        manifest = json.loads(zipped.read("manifest.json"))
        with np.load(BytesIO(zipped.read("arrays.npz")), allow_pickle=False) as arrays:
            compartments = {name: np.array(arrays[key], bool) for name, key in manifest["array_index"]["compartments"].items()}
            affine = np.array(arrays["affine"])
            from scipy.ndimage import binary_fill_holes
            tissue = binary_fill_holes(np.asarray(arrays["mri"]) != 0) | np.logical_or.reduce(list(compartments.values()))
    require(manifest["case_semantic_hash"] == preparation["case_hash"] == target["semantic_hash"]
            and preparation["planning_hash"] == target["planning_hash"] and manifest["frame"] == "RAS+", "Case/frame identity differs")
    require(array_hash(tissue) == target["tissue_support_hash"], "Source tissue support differs")
    labels = np.zeros(tissue.shape, np.int16)
    for label, name in enumerate(sorted(compartments), 1):
        require(array_hash(compartments[name]) == target["compartment_mask_hashes"][name]
                and not np.any(labels[compartments[name]]), "Source compartment differs/overlaps")
        labels[compartments[name]] = label
    cfg = preparation["native_configuration"]
    require(cfg["actual"] == v2["native_configuration"] and cfg["physical_component_equality"] is True
            and cfg["different_components"] == ["tissue_support_provenance"], "Native model components differ")
    for name, array in {"tissue_mask": tissue, "target_labels": labels, "affine": affine, "hard_exclusion": np.zeros(tissue.shape, bool)}.items():
        require(cfg["actual"]["components"][name] == {"shape": list(array.shape), "dtype": str(array.dtype), "array_digest": array_hash(array)}, "Native source array differs")
    require(preparation["optimization"] == target["world_partitions"]["optimization"]
            and preparation["optimization"]["seeds"][0] == declared["v2_model"]["optimization_seed"], "Frozen optimization world differs")
    return tissue, affine, compartments


def audit_attempt(output, baseline_path, bundle, archive, reader=None):
    read = reader or Reader()
    result, authorities = check_authorities(read, output)
    if result is None:
        return authorities
    declared = read.json(output / "declaration.json")
    source, baseline = check_sources(read, output, baseline_path, archive, declared)
    require(result["runtime_content_hash"] == source["runtime_content_hash"]
            and result["scientific_model_hash"] == declared["v2_model"]["decision_model_hash"], "Final source/model differs")
    preparation = read.json(output / "preparation.json")
    tissue, affine, compartments = read_case(read, bundle, archive, declared, preparation)
    model = read.json(archive / "artifacts/preflight/native-axis-v2/profile/model.json")
    original = json.loads(gzip.decompress((archive / "artifacts/preflight/native-axis-v2/profile/greedy/episode.json.gz").read_bytes()))
    freeze = read.json(archive / "artifacts/preflight/native-axis-v2/profile/sequence-freeze.json")
    require(result["original_v2_runtime_hash"] == freeze["source_hash"] != source["runtime_content_hash"], "Prototype runtime confused with unchanged scientific model")
    phases = result["phases"]
    constants = {"geometry_version": literal_constant(archive / "src/resectionlab/geometry.py", "GEOMETRY_VERSION"),
                 "native_version": literal_constant(archive / "src/resectionlab/native_resection.py", "NATIVE_RESECTION_VERSION"),
                 "geometry_epsilon": literal_constant(archive / "src/resectionlab/geometry.py", "_EPS")}
    cache_deltas = check_cache_sequence(phases, read.json(output / "cache-construction.json"), declared, source, constants)
    canonical_trace, verified, phase_times = None, None, []
    for phase in phases:
        mode = phase["mode"]
        require(read.json(output / mode / "receipt.json") == phase, "Phase receipt differs from final result")
        packed = output / mode / "scientific-trace.json.gz"
        data = gzip.decompress(packed.read_bytes())
        export = phase["scientific_trace_export"]
        require(len(data) == export["canonical_bytes"] and hashlib.sha256(data).hexdigest() == export["canonical_sha256"]
                and packed.stat().st_size == export["compressed_bytes"] and file_hash(packed) == export["compressed_sha256"], "Trace export bytes changed")
        trace = json.loads(data)
        require(canonical(trace).encode() == data and canonical_hash(trace) == phase["scientific_trace_hash"]
                and phase["exact_scientific_equality"] is True, "Trace canonical/hash identity differs")
        if canonical_trace is None:
            canonical_trace = data
            verified = check_trace(trace, original, model, declared, tissue, affine, compartments)
        else:
            require(data == canonical_trace, "A full scientific trace differs; timing-only filtering cannot hide this")
        require(phase["actions"] == declared["v2_model"]["fixed_actions"]
                and phase["history_hash"] == declared["v2_model"]["greedy_history_hash"], "Phase action/history differs")
        close(phase["total_reward"], verified["score"]["score"], "Phase source-cell score")
        timing = read.json(output / mode / "timing.json")
        require(timing == phase["timing"] and read.json(output / mode / "progress.json")["transitions"] == timing["transitions"], "Timing/progress records differ")
        require([{key: value for key, value in row.items() if key != "transition_and_next_inventory_seconds"} for row in timing["transitions"]]
                == trace["transitions"], "Timed transitions differ from science")
        measured = timing["full_proposal_verification"]
        adapter = timing["adapter_proposal_accounting"]
        require(type(measured["calls"]) is int and measured["calls"] >= adapter["integrity_calls"]
                and measured["seconds"] >= adapter["integrity_seconds"] >= 0.
                and adapter["preview_calls"] == 66 and adapter["preview_seconds"] >= 0., "Proposal verification/count/timing differs")
        transition_seconds = sum(row["transition_and_next_inventory_seconds"] for row in timing["transitions"])
        require(all(finite(row["transition_and_next_inventory_seconds"]) and row["transition_and_next_inventory_seconds"] >= 0 for row in timing["transitions"])
                and timing["seconds_before_export"] >= timing["reset_seconds"] + transition_seconds >= 0.
                and phase["seconds_including_scientific_comparison_and_export"] >= timing["seconds_before_export"] + phase["scientific_trace_export_seconds"], "Phase timing scopes differ")
        phase_times.append({"mode": mode, "reset_seconds": timing["reset_seconds"], "transitions_seconds": transition_seconds,
            "before_export_seconds": timing["seconds_before_export"], "comparison_and_export_inclusive_seconds": phase["seconds_including_scientific_comparison_and_export"],
            "full_proposal_verification": measured, "adapter_proposal_accounting": adapter,
            "cache_delta": cache_deltas.get(mode), "process_cumulative_peak_rss_bytes": phase["process_cumulative_peak_rss_bytes"]})
    require(len({row["full_proposal_verification"]["calls"] for row in phase_times}) == 1, "Measured full verification work differs across phases")
    audits = result["independent_audits"]
    require(set(audits) == set(MODES), "Four independent native audit records required")
    for mode in MODES:
        record = read.json(output / f"{mode}-independent-audit.json")
        require(record == audits[mode] and finite(record["seconds"]) and record["seconds"] >= 0, "Independent audit record differs")
        cert = record["receipt"]
        require(all(cert[key] is True for key in ("feasible", "complete_tool_checked", "frontier_checked"))
                and cert["source_case_hash"] == declared["source_case"]["semantic_hash"] and cert["action_count"] == 3
                and not cert["failures"] and cert["first_failed_action"] is None
                and cert["first_unsupported_source_voxel"] is None and cert["first_unsupported_position_mm"] is None
                and cert["checker_version"] == "independent-native-sequence-v2", "Native certificate incomplete or failed")
        for name in ("claimed_source_tissue_volume_mm3", "contained_source_tissue_volume_mm3"):
            close(cert[name], verified["removed_mm3"], "Certified source-cell volume")
        close(cert["unsupported_source_tissue_volume_mm3"], 0., "Unsupported certified tissue")
        close(cert["source_voxel_volume_mm3"], abs(float(np.linalg.det(affine[:3, :3]))), "Source cell volume")
    template = read.json(output / "template.json")
    require(template["proposal_accounting"]["preview_calls"] == 26, "Common uncached template work changed")
    resource = read.json(output / "worker-resource.json")
    require(resource["budget"] == declared["resource_budget"] and resource["cancellation_reason"] is None
            and resource["observed_peak_rss_bytes"] <= declared["resource_budget"]["process_peak_rss_bytes"], "Worker resource contract failed")
    times = {row["mode"]: row["comparison_and_export_inclusive_seconds"] for row in phase_times}
    require(all(value > 0 for value in times.values()), "Invalid phase duration")
    return {"audit_status": "passed", "scientific_equivalence_verified": True,
        "declaration_hash": DECLARATION_HASH, "runtime_content_hash": source["runtime_content_hash"], "source_commit": baseline["source_commit"],
        "scientific_model_hash": result["scientific_model_hash"], "trace_hash": phases[0]["scientific_trace_hash"],
        "full_scientific_traces_compared": 4, "unique_scientific_trace": verified, "native_certificate_receipts": 4,
        "mask_evidence": "Saved digests were compared with independent source-cell set reconstruction and full exterior flood. Raw engine masks were not persisted; their direct byte equality is a separate frozen-runner observation.",
        "phase_timing": phase_times, "cache_deltas": cache_deltas,
        "inclusive_time_ratios": {mode: {reference: times[reference] / times[mode] for reference in ("reference_before", "reference_after")}
                                  for mode in ("cached_cold", "cached_warm")},
        "restoration_evidence": {"explicit_runtime_restoration_receipt": False,
            "source_bound_completion_inference": True,
            "basis": "Frozen runner checks native callable/proposer restoration after each phase and checks unchanged cache calls around four audits before completing. Reference-after saved cache counters are unchanged. No pointer observation or before/after-audit counter values were persisted."},
        "negative_preview_count_in_this_trace": sum(row["rejected_previews"] for row in verified["inventory_counts"]),
        "scope": "Four paired fixed replays on one previously studied structural case. Time ratios include comparison/export and are descriptive; not a whole-learning or clinical-performance claim.",
        "timers_nested_not_additive": True, "geometry_checks_rerun": 0, "public_episodes_rerun": 0,
        "new_gradients": 0, "clinical_deficit_probability": None, "final_worlds_used": False, "stress_worlds_used": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attempt", "baseline", "case-bundle", "archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.attempt.resolve()), "Preserve original experiment/audit artifacts")
    read = Reader()
    try:
        result = audit_attempt(args.attempt, args.baseline, args.case_bundle, args.archive, read)
    except Exception as error:
        result = {"audit_status": "failed", "scientific_equivalence_verified": False, "error_type": type(error).__name__, "error": str(error)}
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=file_hash(Path(__file__)),
                  independent_helper_sha256=HELPER_SHA256, input_file_sha256=read.hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"audit_status": result["audit_status"], "receipt": str(args.output)}))
    return 0 if result["audit_status"] == "passed" else 2 if result["audit_status"] == "incomplete_attempt_retained" else 1


if __name__ == "__main__":
    raise SystemExit(main())
