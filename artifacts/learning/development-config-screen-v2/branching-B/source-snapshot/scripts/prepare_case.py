#!/usr/bin/env python3
"""Prepare a pinned structural case for the desktop without modifying sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from resectionlab.imaging import file_sha256, load_nifti_case, load_case, save_case


def prepare(manifest_path: Path, data_root: Path, output: Path) -> dict:
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    root = data_root.resolve()
    paths = {}
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest source path escapes data root")
        if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Source checksum/size mismatch: {path.name}")
        paths[item["modality"]] = path
    selected = "T1c" if "T1c" in paths else "T1"
    label_map = {int(label): info["name"] for label, info in manifest["label_dictionary"].items()
                 if int(label) != 0}
    case = load_nifti_case(
        paths[selected], paths["tumor_segmentation"], case_id=manifest["case_id"],
        label_map=label_map, source_url=manifest["source_collection"]["official_url"],
        license=manifest["license"]["identifier"],
        metadata={
            "acquisition_manifest_sha256": file_sha256(manifest_path),
            "source_collection": manifest["source_collection"],
            "source_distribution": manifest["mirror"],
            "source_frame_declaration": manifest["spatial_frame"]["meaning"],
            "source_files": manifest["files"], "split": manifest["split"],
            "selected_modality": selected,
            "primary_source_equivalence": "unverified",
        },
    )
    imported = time.perf_counter()
    save_case(case, output)
    saved = time.perf_counter()
    reopened = load_case(output)
    finished = time.perf_counter()
    if reopened.semantic_hash != case.semantic_hash:
        raise RuntimeError("Case identity changed across save/reopen")
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case.case_id,
        "case_hash": case.semantic_hash,
        "planning_hash": case.planning_hash,
        "acquisition_manifest_sha256": file_sha256(manifest_path),
        "source_hashes_checked": len(paths),
        "shape": list(case.mri.shape), "affine_ras_mm": case.affine.tolist(),
        "compartment_volumes_mm3": {
            name: int(mask.sum()) * case.voxel_volume_mm3
            for name, mask in case.compartments.items()
        },
        "reopened_identical": True,
        "timings_seconds": {"verify_and_import": imported - started,
                            "save": saved - imported, "reopen": finished - saved},
        "visual_alignment_review": "pending",
        "brain_segmentation_review": "unassessed",
        "source_byte_equivalence_to_TCIA": "unverified_public_mirror",
        "benchmark_role": "development_demo",
        "clinical_deficit_probability": None,
        "unknowns": list(case.unknowns),
        "interpretation": "File/header and persistence checks only; no anatomy or clinical validation.",
    }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repository / "manifests/data_acquisition_ucsf.json")
    parser.add_argument("--data-root", type=Path, default=repository / "data")
    parser.add_argument("--output", type=Path, default=repository / "outputs/cases/UCSF-PDGM-0004.ressectionlab")
    parser.add_argument("--report", type=Path, default=repository / "outputs/qc/UCSF-PDGM-0004.json")
    args = parser.parse_args()
    report = prepare(args.manifest, args.data_root, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_id": report["case_id"], "bundle": str(args.output),
                      "report": str(args.report), "reopened_identical": True}))


if __name__ == "__main__":
    main()
