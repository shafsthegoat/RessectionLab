#!/usr/bin/env python3
"""Verify the pinned BTC PAT28 inputs and prepare an inspectable local case.

The full-head T1 never supplies cortical access by nonzero-intensity threshold.
The fractional annotation becomes a separately declared binary threshold
scenario. Source pathology has no established preoperative availability.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import md5
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from resectionlab.core import ContextField, PatientContext, SourceRef, thaw_json
from resectionlab.imaging import (
    audit_diffusion, file_sha256, load_case, load_fractional_annotation_case, save_case,
)


ACCESSION = "ds001226"
RELEASE = "5.0.1"
REVISION = "359d372c5e972a161966312128adb365870df949"
SUBJECT = "sub-PAT28"
T1 = "sub-PAT28/ses-preop/anat/sub-PAT28_ses-preop_T1w.nii.gz"
ANNOTATION = "derivatives/tumor_masks/sub-PAT28/anat/sub-PAT28_space_T1_label-tumor.nii"
AP = "sub-PAT28/ses-preop/dwi/sub-PAT28_ses-preop_acq-AP_dwi"
PA = "sub-PAT28/ses-preop/dwi/sub-PAT28_ses-preop_acq-PA_dwi"
EXPECTED_FILES = frozenset({
    "dataset_description.json", "participants.tsv", "README", "CHANGES", ANNOTATION,
    T1, T1.removesuffix(".nii.gz") + ".json",
    *(base + extension for base in (AP, PA) for extension in (".nii.gz", ".json", ".bval", ".bvec")),
})


def _verify_sources(manifest_path: Path, data_root: Path) -> tuple[dict, dict[str, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"accession": ACCESSION, "release": RELEASE, "git_commit": REVISION, "subject": SUBJECT}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"BTC preparation requires pinned {key}={value!r}")
    if manifest.get("license") not in {"CC0", "CC0-1.0"}:
        raise ValueError("Pinned source manifest must retain the creator release's CC0 license")
    entries = manifest.get("files", [])
    names = [item["path"] for item in entries]
    if len(names) != len(set(names)) or set(names) != EXPECTED_FILES:
        raise ValueError("Manifest must identify exactly the 15 pinned PAT28 source artifacts without duplicates")
    root = data_root.resolve()
    paths: dict[str, Path] = {}
    for item in entries:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest source path escapes data root")
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Source checksum/size mismatch: {item['path']}")
        if item.get("expected_bytes") is not None and path.stat().st_size != item["expected_bytes"]:
            raise ValueError(f"Pinned annex size mismatch: {item['path']}")
        if item.get("expected_md5") is not None:
            checksum = md5(usedforsecurity=False)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != item["expected_md5"]:
                raise ValueError(f"Pinned annex MD5 mismatch: {item['path']}")
        paths[item["path"]] = path
    return manifest, paths


def _context(paths: dict[str, Path], planning_as_of: datetime | None) -> tuple[PatientContext | None, dict]:
    with paths["participants.tsv"].open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]
    matches = [row for row in rows if row.get("participant_id") == SUBJECT]
    if len(matches) != 1:
        raise ValueError("Participant table must contain exactly one PAT28 record")
    row = matches[0]
    # Eligibility is an archival cohort-selection property. It is never an
    # input to the reward, functional protection, or clinical outcome model.
    if "glioma" not in row.get("tumor type & grade", "").lower():
        raise ValueError("Pinned participant record does not establish glioma cohort eligibility")
    source = SourceRef("btc_participant_table", paths["participants.tsv"].as_uri(),
                       sha256=file_sha256(paths["participants.tsv"]), license="CC0-1.0", provenance="observed")
    selected = (
        ("source_diagnosis", row.get("tumor type & grade"), None),
        ("age", row.get("age"), "years"),
        ("source_reported_tumor_volume", row.get("tumor size (cub cm)"), "cm3"),
        ("source_location", row.get("tumor location"), None),
        ("source_handedness", row.get("handedness"), "source_scale"),
        ("IDH1_IDH2_status", None, None), ("1p19q_codeletion", None, None),
        ("MGMT_promoter_methylation", None, None),
        ("baseline_motor_assessment", None, None), ("baseline_language_assessment", None, None),
    )
    unavailable = {
        name: {"value": None, "unit": unit, "available_at": None, "measurement_at": None,
               "reason": "availability_time_unknown" if value else "not_supplied_in_selected_inputs",
               "source_id": source.source_id}
        for name, value, unit in selected
    }
    context = None
    if planning_as_of is not None:
        context = PatientContext(planning_as_of=planning_as_of, fields=tuple(
            ContextField(name, value, source, available_at=None, measurement_at=None, unit=unit,
                         evidence_type="observed" if value else "unknown")
            for name, value, unit in selected
        ))
        if context.planner_values():
            raise AssertionError("Unknown-timed source context must remain unavailable to the planner")
    return context, unavailable


def prepare(
    manifest_path: Path, data_root: Path, output: Path, *, annotation_threshold: float = 0.5,
    planning_as_of: datetime | None = None,
) -> dict:
    """Prepare exactly one pinned development case without rewriting any source."""
    started = time.perf_counter()
    root = data_root.resolve()
    output = output.resolve()
    if output.is_relative_to(root) or output == manifest_path.resolve():
        raise ValueError("Prepared output must be outside the immutable source tree and manifest")
    manifest, paths = _verify_sources(manifest_path, root)
    verified_at = time.perf_counter()
    context, withheld_context = _context(paths, planning_as_of)
    diffusion = audit_diffusion(paths[AP + ".nii.gz"], paths[AP + ".bval"], paths[AP + ".bvec"],
                                gradient_frame="voxel", gradient_convention="BIDS_FSL")
    case = load_fractional_annotation_case(
        paths[T1], paths[ANNOTATION], threshold=annotation_threshold,
        annotation_interpretation="Creator-supplied manual/semiautomated fractional annotation; file-specific smoothing history unverified.",
        compartment_name="fractional_source_target_threshold_scenario",
        case_id="BTC-ds001226-sub-PAT28-preop", license="CC0-1.0",
        source_url="https://openneuro.org/datasets/ds001226/versions/5.0.1", context=context,
        metadata={
            "acquisition_manifest_sha256": file_sha256(manifest_path),
            "source_collection": {"accession": ACCESSION, "release": RELEASE, "git_commit": REVISION,
                                  "dataset_doi": "10.18112/openneuro.ds001226.v5.0.1",
                                  "descriptor_doi": "10.1038/s41597-022-01806-4"},
            "source_distribution": "creator_release_pinned_OpenNeuro_git_and_annex_S3_versions",
            "source_files": manifest["files"], "source_hashes_checked": len(paths),
            "source_frame_declaration": "Creator-supplied native T1 frame; stored voxel-to-world transform retained.",
            "selected_modality": "T1w", "missing_structural_modalities": ["T1ce", "T2", "FLAIR"],
            "structural_coverage": "full_head", "allow_nonzero_mri_access_support": False,
            "automatic_cortical_access_status": "blocked_without_reviewed_cerebral_mask",
            "brain_segmentation_status": "unassessed", "diffusion_input_audit": diffusion,
            "split": {"role": "development_demo", "patient_group": "BTC-sub-PAT28", "visit": "preop",
                      "external_holdout_eligible": False},
            "context_availability": withheld_context,
            "planning_cutoff_status": "historical_preoperative_cutoff_unknown" if planning_as_of is None else "declared_research_replay_cutoff",
            "planning_as_of": None if planning_as_of is None else planning_as_of.isoformat(),
            "cohort_eligibility_basis": "Archival glioma identification only; unknown-timed pathology excluded from planning.",
        },
    )
    case = case.revised(unknowns=case.unknowns + (
        "full_head_MRI_is_not_a_cortical_surface", "automatic_cortical_access_requires_reviewed_cerebral_mask",
        "historical_preoperative_information_cutoff_unavailable", "clinical_context_availability_unknown",
        "missing_T1ce_T2_FLAIR", "DWI_motion_distortion_and_registration_unreviewed",
    ))
    if case.brain_mask is not None:
        raise AssertionError("Preparation must not invent a cerebral segmentation")
    imported_at = time.perf_counter()
    raw_annotation = nib.load(paths[ANNOTATION]).get_fdata(dtype=np.float32)
    thresholds = sorted({0.25, 0.5, 0.75, float(annotation_threshold)})
    sensitivity = [{"threshold": value, "source_voxel_count": int(np.count_nonzero(raw_annotation >= value)),
                    "interpretation": "source-intensity threshold sensitivity; not resection or clinical probability"}
                   for value in thresholds]
    artifacts = {"preparation": {"source_manifest_sha256": file_sha256(manifest_path),
                                 "annotation_threshold": annotation_threshold,
                                 "threshold_sensitivity": sensitivity,
                                 "diffusion_input_audit": diffusion}, "plans": []}
    save_case(case, output, artifacts=artifacts)
    saved_at = time.perf_counter()
    reopened = load_case(output)
    finished = time.perf_counter()
    if reopened.semantic_hash != case.semantic_hash or reopened.planning_hash != case.planning_hash:
        raise RuntimeError("Case identity changed across save/reopen")
    if file_sha256(paths[ANNOTATION]) != next(item["sha256"] for item in manifest["files"] if item["path"] == ANNOTATION):
        raise RuntimeError("Fractional source annotation changed during preparation")
    return {
        "schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case.case_id, "case_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "source_hashes_checked": len(paths), "source_files_unchanged": True,
        "acquisition_manifest_sha256": file_sha256(manifest_path),
        "source_release": {"accession": ACCESSION, "release": RELEASE, "git_commit": REVISION},
        "bundle": str(output), "bundle_bytes": output.stat().st_size, "reopened_identical": True,
        "shape": list(case.mri.shape), "affine_ras_mm": case.affine.tolist(),
        "compartment_volumes_mm3": {name: int(mask.sum()) * case.voxel_volume_mm3 for name, mask in case.compartments.items()},
        "annotation_derivation": thaw_json(case.metadata["fractional_annotation"]),
        "threshold_sensitivity": sensitivity, "diffusion_input_audit": diffusion,
        "structural_coverage": "full_head", "brain_mask_supplied_or_derived": False,
        "automatic_cortical_access_status": "blocked_without_reviewed_cerebral_mask",
        "allow_nonzero_mri_access_support": False, "withheld_context": withheld_context,
        "planning_as_of": None if planning_as_of is None else planning_as_of.isoformat(),
        "planning_cutoff_status": case.metadata["planning_cutoff_status"],
        "visual_alignment_review": "pending", "clinical_deficit_probability": None,
        "benchmark_role": "single_development_case", "unknowns": list(case.unknowns),
        "timings_seconds": {"source_verification": verified_at - started, "import_and_audit": imported_at - verified_at,
                            "sensitivity_and_save": saved_at - imported_at, "reopen": finished - saved_at},
        "interpretation": "Verified creator-source inputs, explicit annotation derivation and persistence checks; no cortical access, functional localization, resection, or clinical validation.",
    }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repository / "manifests/btc_acquisition.json")
    parser.add_argument("--data-root", type=Path, default=repository / "data/diffusion_source/ds001226-v5.0.1")
    parser.add_argument("--output", type=Path, default=repository / "outputs/cases/BTC-sub-PAT28.ressectionlab")
    parser.add_argument("--report", type=Path, default=repository / "outputs/qc/BTC-sub-PAT28.json")
    parser.add_argument("--annotation-threshold", type=float, default=0.5,
                        help="Declared research threshold on fractional source intensity (default: 0.5; not a probability).")
    parser.add_argument("--planning-as-of", help="Optional timezone-aware research replay cutoff. Historical operation timing remains unknown.")
    args = parser.parse_args()
    try:
        cutoff = None if args.planning_as_of is None else datetime.fromisoformat(args.planning_as_of.replace("Z", "+00:00"))
        if cutoff is not None and (cutoff.tzinfo is None or cutoff.utcoffset() is None):
            raise ValueError("--planning-as-of must include a timezone offset")
        if args.report.resolve().is_relative_to(args.data_root.resolve()):
            raise ValueError("Report must be outside the immutable source tree")
        if args.report.resolve() in {args.manifest.resolve(), args.output.resolve()}:
            raise ValueError("Report must not overwrite the acquisition manifest or prepared bundle")
        report = prepare(args.manifest, args.data_root, args.output,
                         annotation_threshold=args.annotation_threshold, planning_as_of=cutoff)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"BTC preparation failed: {exc}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": report["case_id"], "bundle": str(args.output), "report": str(args.report),
                      "reopened_identical": True, "automatic_cortical_access": "blocked_without_reviewed_cerebral_mask"}))


if __name__ == "__main__":
    main()
