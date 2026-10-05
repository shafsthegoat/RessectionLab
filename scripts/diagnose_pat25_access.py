#!/usr/bin/env python3
"""Initial inventories at PAT25's six existing hypothetical axis exits.

No transitions, search, policy, training or replacement-window selection. The
optional interpreter observer copies compact return-time diagnostics from the
unchanged preview implementation. Its cost is included in the worker budget.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_real_training_cases as prep
from preflight_real_spatial_policy import (
    peak_rss_bytes, read_declaration, sha256, supervise_worker, write_json,
)

VERSION = "pat25-six-existing-exits-initial-inventory-v1"
SUBJECT = "sub-PAT25"
SAVED = "artifacts/remaining-training-frozen-spatial-float64-v1"
MANIFEST = "manifests/experiments/pat25-six-exit-access-diagnostic-v1.json"
SETTINGS = {
    "whole_worker_envelope_seconds": 180.,
    "cooperative_seconds": 170.,
    # Existing supervisor allows <=2 s measurement + two <=2 s stop waits.
    "max_wall_seconds": 174.,
    "max_rss_bytes": 6 * 1024**3,
    "cpu_threads": 1,
    "max_exits": 6,
    "max_declared_slots_per_exit": 78,
    "max_total_previews": 468,
}
EXIT_ORDER = [(axis, sign) for axis in range(3) for sign in (-1, 1)]


def source_inventory():
    paths = [Path(__file__), ROOT / "scripts/prepare_real_training_cases.py",
             ROOT / "scripts/preflight_real_spatial_policy.py",
             *(ROOT / "src/resectionlab").rglob("*.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def saved_preparation():
    """Only the existing PAT25 JSON receipt, never another patient's images."""
    name = f"{SUBJECT}/preparation.json"
    path = ROOT / SAVED / name
    if path.exists():
        raw = path.read_bytes()
    else:
        with tarfile.open(ROOT / SAVED / "completed-run.tar.gz") as archive:
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError("Missing original PAT25 preparation receipt")
            raw = stream.read()
    saved = json.loads(raw)
    if (saved.get("subject") != SUBJECT or saved.get("role") != "TRAIN"
            or saved.get("status") != "prepared"
            or prep.binding_hash(saved["binding"]) != saved.get("binding_hash")):
        raise ValueError("Original successful PAT25 TRAIN binding required")
    return saved, hashlib.sha256(raw).hexdigest()


def declaration():
    cohort = prep.read_development_cohort(ROOT / prep.COHORT_PATH)
    prep.require_development_role(cohort, SUBJECT, role="TRAIN")
    saved, digest = saved_preparation()
    return {
        "version": VERSION, "subject": SUBJECT, "role": "TRAIN",
        "declared_at": datetime.now(timezone.utc).isoformat(), "settings": SETTINGS,
        "source_sha256": source_inventory(), "cohort_sha256": prep.COHORT_SHA256,
        "cohort_path": prep.COHORT_PATH,
        "original_preparation": {"path": f"{SAVED}/{SUBJECT}/preparation.json",
            "archive_fallback": f"{SAVED}/completed-run.tar.gz", "sha256": digest,
            "binding_hash": saved["binding_hash"]},
        "member": prep.known_members()[SUBJECT], "common_task": prep.common_task_definition(),
        "access_rule": prep.ACCESS_RULE,
        "six_exits": saved["binding"]["member"]["access_derivation"]["six_axis_exit_distances_mm"],
        "support_acknowledgment": saved["binding"]["member"]["research_support_acknowledgment"],
        "scope": "Six separate hypothetical accesses, initial declared inventories only; no combined action set or replacement access",
        "prohibited": ["step", "commit", "search", "policy", "training", "outcome_selected_access", "support_or_label_edit", "automatic_retry"],
        "trace_scope": "Return-time compact failure pose, native blocked-cell bounds and local exterior-free membership; diagnostic outputs never become actor inputs",
        "failure_policy": "All six exits retained; partial previews are diagnostic only, never a complete inventory or zero outcome",
        "clinical_deficit_probability": None, "clinical_use_permitted": False,
    }


def validate(record):
    # Role gate before any saved receipt, registry or bundle access.
    if (record.get("version") != VERSION or record.get("subject") != SUBJECT
            or record.get("role") != "TRAIN" or record.get("settings") != SETTINGS):
        raise ValueError("Only the fixed PAT25 TRAIN six-exit diagnostic is permitted")
    cohort = prep.read_development_cohort(ROOT / prep.COHORT_PATH)
    prep.require_development_role(cohort, SUBJECT, role="TRAIN")
    expected = declaration()
    expected["declared_at"] = record.get("declared_at")
    if record != expected:
        raise ValueError("Frozen diagnostic source, source binding, exits or scope changed")
    stamp = datetime.fromisoformat(record["declared_at"])
    if stamp.utcoffset() is None:
        raise ValueError("An aware prospective declaration timestamp is required")
    return saved_preparation()[0], cohort


def six_accesses(derivation, affine, original_access):
    """Keep recorded order; original selected geometry/identifier is unchanged."""
    import numpy as np
    from resectionlab.geometry import AccessWindow

    matrix = np.asarray(affine, dtype=float)
    exits = derivation["six_axis_exit_distances_mm"]
    if (derivation.get("rule") != prep.ACCESS_RULE or len(exits) != 6
            or [(row.get("axis"), row.get("outward_sign")) for row in exits] != EXIT_ORDER
            or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12):
        raise ValueError("Six original ordered exits and physical affine required")
    result = []
    for index, row in enumerate(exits):
        boundary = np.asarray(row["boundary_voxel"], dtype=float)
        if boundary.shape != (3,) or not np.isfinite(boundary).all():
            raise ValueError("Malformed existing exit")
        axis, sign = row["axis"], row["outward_sign"]
        representative = np.asarray(derivation["representative_voxel"])
        other = [dim for dim in range(3) if dim != axis]
        spacing = float(np.linalg.norm(matrix[:3, axis]))
        if (not np.array_equal(boundary[other], representative[other])
                or sign * (boundary[axis] - representative[axis]) <= 0
                or abs(abs(boundary[axis] - representative[axis]) * spacing - row["distance_mm"]) > 1e-10
                or abs((boundary[axis] - .5) - round(boundary[axis] - .5)) > 1e-10):
            raise ValueError("Existing exit no longer matches the recorded source-axis walk")
        selected = row["boundary_voxel"] == derivation["selected_boundary_voxel"]
        access = AccessWindow(matrix[:3, :3] @ boundary + matrix[:3, 3],
            -sign * matrix[:3, axis] / spacing, 6.,
            original_access["window_id"] if selected else f"PAT25-existing-exit-axis{axis}-sign{sign:+d}-v1")
        geometry = {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
                    "radius_mm": access.radius_mm, "window_id": access.window_id}
        if selected and geometry != original_access:
            raise ValueError("Original selected exit geometry changed")
        outside = boundary.copy()
        outside[axis] += .5 * sign
        result.append({"exit_index": index, **row, "selected_original": selected,
                       "outside_neighbor_voxel": np.rint(outside).astype(int).tolist(), "access": geometry})
    if sum(row["selected_original"] for row in result) != 1:
        raise ValueError("Exactly one original selected exit is required")
    return result


class PreviewTrace:
    """Observe existing return events without monkeypatches or retained frames.

    No collision routine is re-executed. A completed preview can populate its
    normal certificate cache; the committed procedure state must stay initial.
    """
    def __init__(self, check=lambda: None):
        self.rows = []
        self.preview_calls = 0
        self.check = check

    def __enter__(self):
        from resectionlab.native_resection import NativeResectionEngine
        if sys.getprofile() is not None or sys.gettrace() is not None:
            raise RuntimeError("Diagnostic needs an exclusive interpreter trace")
        self.preview_code = NativeResectionEngine.preview_stroke.__code__
        self.commit_code = NativeResectionEngine.commit_preview.__code__
        sys.setprofile(self._observe)
        return self

    def __exit__(self, *args):
        sys.setprofile(None)

    def _observe(self, frame, event, returned):
        if event == "call" and frame.f_code is self.commit_code:
            raise RuntimeError("Commit is prohibited in the initial-inventory diagnostic")
        if event == "call" and frame.f_code is self.preview_code:
            if self.preview_calls >= SETTINGS["max_declared_slots_per_exit"]:
                raise RuntimeError("Initial preview budget exceeded before execution")
            self.preview_calls += 1
        if event != "return" or frame.f_code is not self.preview_code:
            return
        from resectionlab.native_resection import NativeStrokeResult
        if not isinstance(returned, NativeStrokeResult):
            return  # Raising previews remain failed/incomplete at the worker boundary.
        import numpy as np

        def indices_hash(array):
            # Empty (0,3) arrays need tobytes rather than memoryview.cast.
            header = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode()
            return "sha256:" + hashlib.sha256(header + array.tobytes(order="C")).hexdigest()

        values = frame.f_locals
        engine = values["self"]
        if engine.revision or engine.history:
            raise RuntimeError("Trace observed a noninitial native state")
        row = {"tool_id": returned.tool_id, "entry_mm": list(returned.entry_mm),
            "tip_mm": list(returned.tip_mm), "feasible": returned.feasible, "reason": returned.reason,
            "source_state_hash": returned.source_state_hash,
            "failure_tip_mm": None if returned.failure_tip_mm is None else list(returned.failure_tip_mm),
            "completed_preview_microsteps": len(returned.microsteps),
            "preview_removed_cells": len(returned.removed_indices_native),
            "preview_removed_indices_hash": indices_hash(returned.removed_indices_native),
            "preview_contact_indices_hash": indices_hash(returned.contact_indices_native),
            "failure_step_index": None, "failure_insertion_fraction": None,
            "blocked_cell_count": None, "blocked_cell_index_bounds": None,
            "blocked_cell_ras_aabb_mm": None,
            "scope": "uncommitted preview diagnostics; no removal credited"}
        if returned.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE":
            blocked = values.get("blocked")
            if blocked is None or not len(blocked) or "step" not in values:
                raise RuntimeError("Unchanged shaft rejection lacks expected trace locals")
            low, high = blocked.min(axis=0), blocked.max(axis=0)
            from itertools import product
            corners = np.asarray(list(product(*zip(low - .5, high + .5))))
            physical = corners @ returned.native_affine[:3, :3].T + returned.native_affine[:3, 3]
            row.update(failure_step_index=int(values["step"]),
                failure_insertion_fraction=float(values["step"] / values["number"]),
                blocked_cell_count=len(blocked), blocked_cell_index_bounds=[low.tolist(), high.tolist()],
                blocked_cell_ras_aabb_mm=[physical.min(axis=0).tolist(), physical.max(axis=0).tolist()])
        self.rows.append(row)
        self.check()


def inspect_task(factory, exit_record, check=lambda: None):
    """One unchanged task factory; retain partial diagnostics if it fails."""
    import numpy as np
    from resectionlab.spatial_policy_diagnostics import spatial_coverage, runtime_proposal_coverage

    started = time.perf_counter()
    trace = PreviewTrace(check)
    try:
        check()
        with trace:
            task = factory()
        check()
        task._assert_frozen()
        engine = task._engine
        if (engine.revision or engine.history or task._steps or task._history
                or np.any(engine.removed_mask) or np.any(engine.contact_mask)
                or not np.array_equal(engine.remaining_mask, engine.config.tissue_mask)
                or not np.array_equal(engine.connected_free_mask, engine._initial_connected_free)):
            raise RuntimeError("Initial-inventory diagnostic altered the committed state")
        inventory, observation = task.candidate_inventory(), task.observation()
        if (not inventory["complete"] or not inventory["ledger_complete"]
                or inventory["declared_slots"] != SETTINGS["max_declared_slots_per_exit"]
                or inventory["omitted_count"] != 0
                or trace.preview_calls != len(trace.rows)
                or len(trace.rows) != inventory["emitted_count"]):
            raise RuntimeError("Missing complete declared inventory or preview trace")
        for row, returned in zip(inventory["emitted"], trace.rows, strict=True):
            if any(row[key] != returned[key] for key in ("tool_id", "entry_mm", "tip_mm", "feasible", "reason")):
                raise RuntimeError("Trace does not match the original preview acceptance/order")
            returned["action_id"] = row["action_id"]
        cell = np.asarray(exit_record["outside_neighbor_voxel"], dtype=int)
        inside = bool(np.all((cell >= 0) & (cell < np.asarray(engine.remaining_mask.shape))))
        exterior = {"outside_neighbor_voxel": cell.tolist(), "inside_source_grid": inside,
            "in_estimated_support": bool(engine.config.tissue_mask[tuple(cell)]) if inside else False,
            "connected_to_external_free": bool(engine.connected_free_mask[tuple(cell)]) if inside else True,
            "scope": "one source cell immediately outward from recorded local exit; not whole-shaft or clinical clearance"}
        native = task.case._native_affine_ras_mm
        coverage = spatial_coverage(observation, source_shape=task.case.observed_support.shape,
            source_affine=task.case.affine_ras_mm, nominal_target=task.case.nominal_target, native_affine=native)
        proposal_coverage = runtime_proposal_coverage(task.case, inventory, observation, native_affine=native)
        task._assert_frozen()
        check()
        return {"status": "complete", "exit": exit_record, "initial_inventory": inventory,
            "trace": trace.rows, "exterior_neighbor": exterior, "actor_coverage": coverage,
            "proposal_coverage": proposal_coverage, "decision_model_hash": task.decision_model_hash,
            "source_hash": task.case.source_hash, "native_config_hash": engine.config.fingerprint,
            "initial_state_hash": engine.state_hash, "reward": asdict(task.reward_spec),
            "executed_transitions": 0, "commits": 0, "preview_calls": trace.preview_calls,
            "elapsed_seconds": time.perf_counter() - started}
    except BaseException as error:
        error.diagnostic = {"status": "unassessed", "exit": exit_record,
            "failure": {"type": type(error).__name__, "message": str(error)},
            "partial_preview_trace": trace.rows, "partial_preview_count": len(trace.rows),
            "started_preview_calls": trace.preview_calls,
            "inventory_complete": False, "elapsed_seconds": time.perf_counter() - started}
        raise


def load_bound_case(record, check):
    """The fixed TRAIN and byte identities are checked before image decoding."""
    saved, cohort = validate(record)
    member = record["member"]
    path = ROOT / member["case_bundle"]
    check()
    if sha256(path) != member["case_bundle_sha256"]:
        raise ValueError("Original PAT25 source bundle bytes changed")
    check()
    case = prep._decode_case(path)
    collection = case.metadata.get("source_collection", {})
    if (case.case_id != "BTC-ds001226-sub-PAT25-preop" or case.metadata.get("is_synthetic")
            or case.semantic_hash != member["case_semantic_hash"] or case.planning_hash != member["planning_hash"]
            or collection.get("accession") != "ds001226" or collection.get("release") != "5.0.1"
            or collection.get("git_commit") != cohort["source"]["git_commit"]
            or not any(ref.source_id == "structural" and ref.provenance == "observed" for ref in case.source_refs)):
        raise ValueError("Decoded source differs from original real preoperative PAT25")
    evidence = case.structural_evidence.get(member["evidence_id"])
    if (evidence is None or evidence.evidence_hash != member["evidence_hash"]
            or evidence.review_status != "review_required" or evidence.provenance != "estimated"
            or evidence.review is not None or evidence.model_sha256 is None):
        raise ValueError("Original unreviewed support binding changed")
    evidence.assert_matches(case)
    import numpy as np
    target = np.logical_or.reduce(tuple(case.compartments.values()))
    support = np.asarray(evidence.mask, dtype=bool)
    affine = np.asarray(case.affine)
    if case.frame == "LPS+":
        affine = np.diag([-1., -1., 1., 1.]) @ affine
    elif case.frame != "RAS+":
        raise ValueError("Unsupported source frame")
    access, derivation = prep.derive_access(target, support, affine, SUBJECT)
    original = saved["binding"]["member"]
    if (access != original["access"] or derivation != original["access_derivation"]
            or int(target.sum()) != saved["coverage"]["full_target_source_cells"]):
        raise ValueError("Original annotation/support/axis-walk binding changed")
    check()
    return case, saved, six_accesses(derivation, affine, access)


def worker(record, output):
    started = time.perf_counter()
    def check():
        if time.perf_counter() - started >= SETTINGS["cooperative_seconds"] or peak_rss_bytes() > SETTINGS["max_rss_bytes"]:
            raise TimeoutError("Whole-worker wall or memory envelope exhausted")
    report = {"version": VERSION, "subject": SUBJECT, "status": "preparing", "settings": SETTINGS,
        "executed_transitions": 0, "commits": 0, "optimizer_updates": 0,
        "replacement_access_selected": False, "clinical_deficit_probability": None,
        "exits": [{"exit_index": i, "axis": axis, "outward_sign": sign, "status": "unassessed"}
                  for i, (axis, sign) in enumerate(EXIT_ORDER)]}
    def save():
        report.update(elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes())
        write_json(output / "receipt.json", report)
    save()
    try:
        # Native NumPy/SciPy threads are restricted by the supervising process.
        case, saved, exits = load_bound_case(record, check)
        report["original_preparation_sha256"] = record["original_preparation"]["sha256"]
        report["status"] = "inspecting"
        total_previews = 0
        for exit_record in exits:
            index = exit_record["exit_index"]
            check()
            member = {**saved["binding"]["member"], "access": exit_record["access"]}
            try:
                row = inspect_task(lambda: prep._construct_task(case, member, record["common_task"], lambda: (
                    time.perf_counter() - started >= SETTINGS["cooperative_seconds"] or peak_rss_bytes() > SETTINGS["max_rss_bytes"])),
                    exit_record, check)
                if row["reward"] != record["common_task"]["objective"]:
                    raise ValueError("Factory changed the frozen objective")
                if exit_record["selected_original"]:
                    if (row["initial_inventory"] != saved["initial_inventory"]
                            or row["decision_model_hash"] != saved["binding"]["decision_model_hash"]):
                        raise ValueError("Original selected-access inventory no longer reproduces")
                    row["original_inventory_exactly_reproduced"] = True
                total_previews += len(row["trace"])
                if total_previews > SETTINGS["max_total_previews"]:
                    raise RuntimeError("Whole-worker preview budget exceeded")
                report["exits"][index] = row
                save()
            except BaseException as error:
                report["exits"][index] = getattr(error, "diagnostic", {
                    "status": "unassessed", "exit": exit_record,
                    "failure": {"type": type(error).__name__, "message": str(error)}})
                raise
            gc.collect()
        if (source_inventory() != record["source_sha256"]
                or sha256(ROOT / record["member"]["case_bundle"]) != record["member"]["case_bundle_sha256"]):
            raise ValueError("Source bytes changed during diagnostic")
        check()
        report.update(status="complete", total_previews=total_previews, all_six_initial_inventories_complete=True)
        save()
    except BaseException as error:
        report.update(status="incomplete", failure={"type": type(error).__name__, "message": str(error)},
                      all_six_initial_inventories_complete=False)
        save()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", help="Write metadata-only prospective manifest; no image decoding")
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.declare:
        if args.manifest.exists():
            raise ValueError("Preserve existing declaration; do not silently overwrite")
        write_json(args.manifest, declaration())
        return
    if args.output is None:
        parser.error("--output is required for an explicitly authorized execution")
    record, digest, raw = read_declaration(args.manifest, args.expected_sha256)
    if args.worker:
        if args.expected_sha256 is None:
            raise ValueError("Worker requires the supervisor-bound manifest identity")
        worker(record, args.output)
        return
    validate(record)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "declaration-input.json").write_bytes(raw)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--manifest",
               str(args.output / "declaration-input.json"), "--expected-sha256", digest,
               "--output", str(args.output)]
    result = supervise_worker(command, args.output, SETTINGS, digest)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
