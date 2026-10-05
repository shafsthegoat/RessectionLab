"""Summarize saved JSON inventories only; never load patient images or a simulator.

Run from any directory with Python 3. The completed archive supplies records when
the ignored per-patient preparation JSON files are absent from a fresh checkout.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "artifacts/remaining-training-frozen-spatial-float64-v1"
SUBJECTS = ("sub-PAT22", "sub-PAT25", "sub-PAT28")


def preparation_bytes(subject):
    name = f"{subject}/preparation.json"
    local = SAVED / name
    if local.exists():
        return local.read_bytes()
    with tarfile.open(SAVED / "completed-run.tar.gz") as archive:
        stream = archive.extractfile(name)
        if stream is None:
            raise ValueError(f"Missing saved preparation: {name}")
        return stream.read()


def main():
    cases = {}
    for subject in SUBJECTS:
        raw = preparation_bytes(subject)
        record = json.loads(raw)
        assert record["subject"] == subject and record["role"] == "TRAIN"
        coverage, inventory = record["coverage"], record["initial_inventory"]
        proposals = coverage["proposals"]
        rows = inventory["emitted"]
        common = record["binding"]["common_task"]
        offsets = common["adapter_options"]["proposal_config"]["offsets_source_voxels"]
        families = sorted({row["family"] for row in rows})
        assert len(rows) == inventory["emitted_count"] == 78
        assert inventory["declared_slots"] == len(offsets) * len(common["tools"]) * len(families)
        assert inventory["complete"] and inventory["ledger_complete"]
        assert inventory["omitted_count"] == inventory["duplicate_count"] == inventory["unavailable_count"] == 0
        assert coverage["target_outside_support_source_cells"] == 0
        assert coverage["target_outside_actor_crop_source_cells"] == 0
        assert coverage["actor"]["nominal_target_mass_fraction_visible"] == 1.0
        dispositions = dict(sorted(Counter(row["reason"] for row in rows).items()))
        assert dispositions == proposals["preview_dispositions"]
        total = coverage["full_target_source_cells"]
        outside = proposals["accepted_envelope"]["target_centers_outside_aabb"]
        central_ids = {row["action_id"] for row in rows if row["column_index"] == 0 and row["family"] == "exposed_opening"}
        cases[subject] = {
            "preparation_sha256": sha256(raw).hexdigest(),
            "binding_hash": record["binding_hash"],
            "full_target_source_cells": total,
            "full_target_volume_mm3": coverage["full_target_volume_mm3"],
            "target_outside_support_source_cells": 0,
            "target_outside_actor_crop_source_cells": 0,
            "declared_slots": inventory["declared_slots"],
            "emitted_count": len(rows),
            "accepted_count": inventory["accepted_count"],
            "candidate_cap": inventory["candidate_cap"],
            "omitted_duplicate_unavailable_counts": [0, 0, 0],
            "dispositions": dispositions,
            "family_dispositions": {family: dict(sorted(Counter(row["reason"] for row in rows if row["family"] == family).items())) for family in families},
            "tool_dispositions": {tool["tool_id"]: dict(sorted(Counter(row["reason"] for row in rows if row["tool_id"] == tool["tool_id"]).items())) for tool in common["tools"]},
            "offsets_source_voxels": offsets,
            "tools": common["tools"],
            "access": record["binding"]["member"]["access"],
            "access_derivation": record["binding"]["member"]["access_derivation"],
            "central_exposed_opening": [row for row in proposals["actions"] if row["action_id"] in central_ids],
            "accepted_envelope": proposals["accepted_envelope"],
            "emitted_envelope": proposals["emitted_envelope"],
            "target_center_depth_range_mm": proposals["nominal_target_center_depth_range_mm"],
            "target_centers_outside_accepted_aabb_percent": None if outside is None else 100 * outside / total,
        }
    summary_path = SAVED / "compact-summary.json"
    summary_bytes = summary_path.read_bytes()
    pat25 = next(row for row in json.loads(summary_bytes)["patients"] if row["subject"] == "sub-PAT25")
    for method in pat25["methods"].values():
        assert method["actions"] == ["STOP"]
        assert method["outcomes"]["simulated_removed_volume_mm3"] == 0
    source_paths = (
        "scripts/prepare_real_training_cases.py",
        "src/resectionlab/native_proposals.py",
        "src/resectionlab/native_resection.py",
        "src/resectionlab/native_spatial_task.py",
        "src/resectionlab/spatial_policy_diagnostics.py",
    )
    receipt = {
        "scope": "Saved JSON inventories and source-code inspection only; no image decode, native preview, training or new patient-role access",
        "reproduce": "python artifacts/frozen-transfer-access-diagnosis-v1/diagnose.py",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "saved_run": str(SAVED.relative_to(ROOT)),
        "compact_summary_sha256": sha256(summary_bytes).hexdigest(),
        "read_source_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths},
        "aabb_caution": "Saved initial active-sweep bounding boxes are optimistic envelopes, not union coverage, removed volume, eventual reachability or clinical clearance. Target-center counts are reused from the saved diagnostic, not recomputed from patient arrays.",
        "pat25_all_three_methods_stop_only": True,
        "cases": cases,
    }
    destination = Path(__file__).with_name("receipt.json")
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({subject: {"accepted": item["accepted_count"], "excluded_target_centers_percent": item["target_centers_outside_accepted_aabb_percent"]} for subject, item in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
