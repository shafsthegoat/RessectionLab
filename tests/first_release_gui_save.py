"""Opt-in independent audit of the actual Electron-created review files.

No GUI interaction, patient/source edits, inference, or model-weight hash sweep.
The output contains portable relative paths, hashes and small scalar records.
Run with this checkout's src on PYTHONPATH; --case can audit one later arrival.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from resectionlab.core import thaw_json
from resectionlab.rehearsal_episode import restore_episode
from resectionlab.workspace_bundle import load_workspace
from first_release_qa import independent_episode_frames

ROOT = Path(__file__).resolve().parents[1]
CASES = ("UCSF-PDGM-0004", "SynthRAD2023-1BA336", "Lausanne-sub-000-ses-20110101")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stamp(path):
    s = path.stat()
    return {"sha256": sha(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def same_case_arrays(source, reviewed):
    assert source.semantic_hash == reviewed.semantic_hash
    assert source.case_id == reviewed.case_id and source.frame == reviewed.frame
    assert np.array_equal(source.affine, reviewed.affine)
    assert np.array_equal(source.mri, reviewed.mri)
    rows = {"mri": {"dtype": reviewed.mri.dtype.str, "shape": list(reviewed.mri.shape), "raw_c_order_sha256": memory_sha(reviewed.mri)}}
    assert (source.brain_mask is None) == (reviewed.brain_mask is None)
    if source.brain_mask is not None:
        assert np.array_equal(source.brain_mask, reviewed.brain_mask)
        rows["brain_mask"] = {"foreground_voxels": int(reviewed.brain_mask.sum()), "raw_c_order_sha256": memory_sha(reviewed.brain_mask)}
    for category in ("compartments", "source_compartments"):
        a, b = getattr(source, category), getattr(reviewed, category)
        assert set(a) == set(b)
        for name in a:
            assert np.array_equal(a[name], b[name]), (category, name)
        rows[category] = {name: {"foreground_voxels": int(array.sum()), "raw_c_order_sha256": memory_sha(array)} for name, array in b.items()}
    assert [r.to_dict() for r in source.source_refs] == [r.to_dict() for r in reviewed.source_refs]
    assert thaw_json(source.metadata) == thaw_json(reviewed.metadata)
    return rows


def archive_originals(source, reviewed):
    with ZipFile(source) as a, ZipFile(reviewed) as b:
        original_doc, reviewed_doc = json.loads(a.read("workspace.json")), json.loads(b.read("workspace.json"))
        assert original_doc["originalFiles"] == reviewed_doc["originalFiles"]
        rows = []
        for item in original_doc["originalFiles"]:
            left, right = a.read(item["member"]), b.read(item["member"])
            assert left == right
            assert hashlib.sha256(right).hexdigest() == item["sha256"]
            assert len(right) == item["bytes"]
            rows.append({key: item[key] for key in ("name", "role", "sha256", "bytes")})
        arrays_a, arrays_b = a.read("arrays.npz"), b.read("arrays.npz")
        # Numbered compartment entries may be reordered when loaded and saved.
        # Named array equality is established separately; require the complete
        # multiset of embedded NPY bytes as a second, serializer-independent check.
        def npy_hashes(payload):
            with ZipFile(BytesIO(payload)) as archive:
                return sorted(hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist())
        assert npy_hashes(arrays_a) == npy_hashes(arrays_b)
        return {"count": len(rows), "bytes": sum(row["bytes"] for row in rows), "files": rows,
                "all_original_bytes_exact": True, "primary_npz_bytes_exact": arrays_a == arrays_b,
                "primary_npy_payload_multiset_exact": True,
                "primary_npz_source_sha256": hashlib.sha256(arrays_a).hexdigest(),
                "primary_npz_sha256": hashlib.sha256(arrays_b).hexdigest(),
                "packaging_note": None if arrays_a == arrays_b else "Numbered compartment entries reordered; all named arrays and embedded NPY payload bytes remain exact."}


def replay_saved(case, artifact):
    record = artifact["patient_rehearsal"]
    assert record["schema"] == "patient-rehearsal-workspace-v1"
    live = restore_episode(case, record["live"])
    assert live.serialize() == record["live"]
    live_export = live.export()
    live_audit = independent_episode_frames(case, live_export)
    comparison = record["comparison"]
    snapshot = restore_episode(case, comparison["snapshot"])
    assert snapshot.serialize() == comparison["snapshot"]
    assert snapshot.state_id == comparison["snapshotId"] == live.state_id
    prefix = snapshot.actions
    branches = []
    branch_exports = {}
    for item in comparison["alternatives"]:
        branch = restore_episode(case, item["record"])
        assert branch.serialize() == item["record"]
        assert branch.actions[:len(prefix)] == prefix
        assert branch.frames[:len(snapshot.frames)] == snapshot.frames
        assert branch.source_binding == live.source_binding
        exported = branch.export()
        audit = independent_episode_frames(case, exported)
        branch_exports[item["id"]] = exported
        branches.append({"id": item["id"], "label": item["label"], "record_hash": item["record"]["recordHash"],
                         "initial_state_id": branch._initial_state_id, "final_state_id": branch.state_id,
                         "action_count": len(branch.actions), "actions": branch.actions,
                         "exact_saved_action_prefix": True, "exact_replayed_frame_prefix": True,
                         "replayed_serialization_exact": True, "independent_frame_audit": audit})
    assert len(branches) == 2 and len(branch_exports) == 2
    view = artifact["workspace"]["rehearsalView"]
    selected = branch_exports[view["branchId"]]
    assert 0 <= view["frameIndex"] < len(selected["replayFrames"])
    assert view["episodeInitialStateId"] == live_export["initialStateId"]
    assert view["selectedTool"] in {t["toolId"] for t in selected["tools"]}
    assert type(view["visible"]) is bool
    selected_frame = selected["replayFrames"][view["frameIndex"]]
    return {"schema": live_export["schema"], "live_record_hash": record["live"]["recordHash"],
            "initial_state_id": live_export["initialStateId"], "live_state_id": live.state_id,
            "comparison_snapshot_id": comparison["snapshotId"], "planning_input_hash": live.source_binding["planningInputHash"],
            "source_frame_hash": live.source_binding["frameHash"], "source_target_hash": live.source_binding["targetHash"],
            "target_layer_names": list(live.source_binding["targetNames"]), "live_actions": live.actions,
            "live_replay": live_audit, "branches": branches, "saved_view": view,
            "selected_frame": {"index": selected_frame["frameIndex"], "last_index": len(selected["replayFrames"])-1,
                               "elapsed_seconds": selected_frame["elapsedSeconds"], "state_id": selected_frame["stateAfter"],
                               "kind": selected_frame["kind"], "metrics": selected_frame["metrics"]},
            "source_arrays_modified": False, "clinical_validation": False,
            "limitations": ["Assumed post-exposure cavity is generated and separately accounted.",
                            "Field management is a dimensionless debris proxy, not coagulation or hemostasis.",
                            "Vessels, motor/language function and vascular outcomes remain unknown on this real MRI case.",
                            "The independent checks establish software/replay consistency, not surgical realism or clinical accuracy."]}


def audit(name, source_dir, reviewed_dir):
    source = source_dir / f"{name}.ressectionlab"
    reviewed = reviewed_dir / f"{name}-reviewed.ressectionlab"
    if not reviewed.exists():
        return {"case": name, "status": "pending_reviewed_file"}
    before_source, before_reviewed = stamp(source), stamp(reviewed)
    case, _, old_saved = load_workspace(source)
    actual, artifacts, saved = load_workspace(reviewed)
    artifact = thaw_json(artifacts)
    arrays = same_case_arrays(case, actual)
    assert len(old_saved["sources"]) == len(saved["sources"])
    auxiliary = []
    for original, reopened in zip(old_saved["sources"], saved["sources"]):
        auxiliary.append({"case": reopened.case.case_id, "case_hash": reopened.case.semantic_hash,
                          "source_arrays_exact": True, "arrays": same_case_arrays(original.case, reopened.case)})
    result = {"case": name, "status": "pass", "audit_utc": datetime.now(timezone.utc).isoformat(),
              "canonical_bundle": str(source.relative_to(ROOT)), "reviewed_bundle": str(reviewed.relative_to(ROOT)),
              "canonical_stamp": before_source, "reviewed_stamp": before_reviewed,
              "case_hash": actual.semantic_hash, "session_hash": saved["sessionHash"],
              "source_geometry": {"shape": list(actual.mri.shape), "affine_ras_mm": actual.affine.tolist(), "frame": actual.frame},
              "source_arrays_exact": True, "source_provenance_and_layer_metadata_exact": True,
              "arrays": arrays, "auxiliary_sources": auxiliary,
              "archived_originals": archive_originals(source, reviewed),
              "imaging_state": saved["imagingState"], "workspace_preferences": artifact.get("workspace", {})}
    if "patient_rehearsal" in artifact:
        assert name == "UCSF-PDGM-0004", "Unexpected rehearsal attached to separate source case"
        result["rehearsal"] = replay_saved(actual, artifact)
    else:
        result["rehearsal"] = None
    assert stamp(source) == before_source and stamp(reviewed) == before_reviewed
    result["source_and_reviewed_files_unchanged_during_audit"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/poc-cases")
    parser.add_argument("--reviewed-dir", type=Path, default=ROOT / "outputs/poc-rehearsal")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/poc-first-rehearsal-v1/gui-save-audit.json")
    args = parser.parse_args()
    if args.case and args.output.exists():
        report = json.loads(args.output.read_text())
    else:
        report = {"schema": "poc-first-electron-save-independent-audit-v1",
                  "scope": "Read-only independent audit of files saved through the actual Electron session; GUI interaction observed by parent, not this auditor",
                  "cases": [], "limitations": ["No new inference or model-weight hash sweep performed.", "No patient arrays or absolute local paths are included in this report."]}
    requested = [args.case] if args.case else CASES
    by_name = {row["case"]: row for row in report["cases"]}
    for name in requested:
        try:
            by_name[name] = audit(name, args.source_dir, args.reviewed_dir)
        except Exception as error:
            # Error type/message only: no tracebacks leaking local absolute paths.
            message = str(error).replace(str(ROOT), "<checkout>")
            by_name[name] = {"case": name, "status": "fail", "error_type": type(error).__name__, "message": message}
        print(name, by_name[name]["status"], flush=True)
    report["cases"] = [by_name[name] for name in CASES if name in by_name]
    report["updated_utc"] = datetime.now(timezone.utc).isoformat()
    source_files = ["tests/first_release_gui_save.py", "tests/first_release_qa.py", "src/resectionlab/rehearsal_episode.py", "src/resectionlab/workspace_bundle.py"]
    report["audit_source_sha256"] = {name: sha(ROOT / name) for name in source_files}
    report["available_copies_passed"] = all(row["status"] != "fail" for row in report["cases"])
    report["all_three_reviewed_copies_passed"] = len(report["cases"]) == 3 and all(row["status"] == "pass" for row in report["cases"])
    encoded = json.dumps(report, indent=2)+"\n"
    assert "/Users/" not in encoded and "file://" not in encoded, "Report unexpectedly contains absolute local paths"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(json.dumps({"available_passed": report["available_copies_passed"], "all_three_passed": report["all_three_reviewed_copies_passed"], "output": str(args.output.relative_to(ROOT))}))
    raise SystemExit(0 if report["available_copies_passed"] else 1)


if __name__ == "__main__":
    main()
