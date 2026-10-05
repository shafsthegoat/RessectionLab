#!/usr/bin/env python3
"""Independently audit only the two saved PAT16/PAT20 proposal bundles.

Uses retained files and independent NiBabel reindexing. No model inference,
downloads, source edits, review acceptance, or clinical accuracy measurement.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import resource
import time
from zipfile import ZipFile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import nibabel as nib
import numpy as np

from audit_brain_extraction import digest, read_native, source_annotation, verify_file
from audit_pat16_pat20_extraction import DECLARATION, DECLARATION_SHA256, BATCH, require
from resectionlab.core import array_digest, thaw_json
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.structural_evidence import (
    declared_mri_support_allowed, planning_brain_support, structural_frame_hash,
    validate_explicit_support,
)


INTEGRATION = "docs/brain-extraction-pat16-pat20-bundle-integration.json"
INTEGRATION_SHA256 = "006e223b8b6a2b689173c264584e007fdbce0e96da611b67fb74f4e841d88946"
EXTRACTION_AUDITS = {
    "docs/brain-extraction-pat16-pat20-independent-qc.json": "237e66d6a22905917b52bbe0f970ab709c6e3e1f61cc12fb06a3c3575a3281e8",
    "docs/brain-extraction-pat16-pat20-independent-visual-review.json": "67f001678c947ce15f2e394e1c6d8955a1d3ee3edbc2796e17b9e3f7e0a236ea",
}


def check_portable_artifacts(portable: dict, report_bytes: bytes, run_directory: str) -> dict:
    """Require exact report bytes and a complete, honest mask/distance inventory."""
    require(portable["extraction_report_utf8"].encode("utf-8") == report_bytes,
            "Embedded extraction report bytes differ")
    report_hash = sha256(report_bytes).hexdigest()
    require(portable["extraction_report_sha256"] == report_hash, "Embedded report hash differs")
    require(portable["selected_repetition"] == "r1", "Only the first declared repetition is selected")
    require(portable["mask_arrays_embedded"] is True and portable["predicted_distance_arrays_embedded"] is False,
            "Distance portability claim is inaccurate")
    require(portable["brain_reviewed"] is False and portable["cortical_access_permitted"] is False
            and portable["clinical_deficit_probability"] is None, "Portable metadata cannot approve anatomy")
    historical = portable["historical_binary_attestation"]
    require(all(historical[key] is False for key in (
        "historical_installed_package_binary_hashes_predeclared", "historical_interpreter_binary_hash_predeclared",
        "unrecorded_runs_excluded_by_attestation")), "Historical binary proof must not be invented")
    require({item["path"]: item["sha256"] for item in portable["independent_extraction_audits"]} == EXTRACTION_AUDITS,
            "Independent extraction references changed")
    report = json.loads(report_bytes)
    expected = {(variant, kind) for variant in ("main", "nocsf")
                for kind in ("source_mask", "predicted_signed_distance")}
    entries = portable["external_extraction_artifacts"]
    require(len(entries) == 4 and {(e["variant"], e["artifact_kind"]) for e in entries} == expected,
            "External artifact inventory incomplete or duplicated")
    for entry in entries:
        is_mask = entry["artifact_kind"] == "source_mask"
        filename = f'{entry["variant"]}_{"mask" if is_mask else "distance_mm"}.nii.gz'
        expected_hash = report["variants"][entry["variant"]]["inference"]["artifact_hashes"][filename]
        require(entry["path"] == f"{run_directory}/{filename}" and entry["sha256"] == expected_hash,
                "External artifact path or hash differs from the frozen run")
        require(entry["native_array_embedded_as_proposal"] is is_mask
                and entry["embedded_as_original_file"] is False, "External artifact embedding claim differs")
    return report


def manifest(path: Path) -> dict:
    with ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def audit(repository: Path) -> dict:
    started = time.perf_counter()
    tracked = {}

    def checked(relative: str, expected: str, size: int | None = None) -> Path:
        path = repository / relative
        verify_file(path, expected)
        if size is not None:
            require(path.stat().st_size == size, "Pinned file size differs")
        tracked[relative] = expected
        return path

    declaration = json.loads(checked(DECLARATION, DECLARATION_SHA256).read_text())
    integration = json.loads(checked(INTEGRATION, INTEGRATION_SHA256).read_text())
    audits = {name: json.loads(checked(name, expected).read_text()) for name, expected in EXTRACTION_AUDITS.items()}
    extraction_qc = audits["docs/brain-extraction-pat16-pat20-independent-qc.json"]
    batch = json.loads(checked(f"{BATCH}/batch_record.json", extraction_qc["batch_record_sha256"]).read_text())
    require([c["subject"] for c in integration["cases"]] == ["sub-PAT16", "sub-PAT20"], "Unexpected bundle scope")
    for asset in declaration["model_assets"].values():
        checked(asset["path"], asset["sha256"], asset["bytes"])
    results = []
    for source, integrated in zip(declaration["subjects"], integration["cases"], strict=True):
        subject = source["subject"]
        require(subject == integrated["subject"], "Patient mismatch")
        source_qc = json.loads(checked(source["independent_source_qc"], source["independent_source_qc_sha256"]).read_text())
        acquisition = json.loads(checked(source["source_manifest"], source["source_manifest_sha256"]).read_text())
        for entry in acquisition["files"]:
            file = checked("data/diffusion_source/ds001226-v5.0.1/" + entry["path"], entry["sha256"], entry["bytes"])
            if entry.get("expected_md5"):
                require(md5(file.read_bytes()).hexdigest() == entry["expected_md5"], "Annex MD5 differs")
        original_path = checked(source["prepared_case"], source_qc["bundle_sha256"])
        derived_path = checked(integrated["derived_case"], integrated["derived_case_sha256"], integrated["derived_case_bytes"])
        original, derived = load_case(original_path), load_case(derived_path)
        before, after = manifest(original_path), manifest(derived_path)
        allowed_changes = {"revision", "case_semantic_hash", "array_sha256", "structural_evidence", "artifacts"}
        require({k: v for k, v in before.items() if k not in allowed_changes}
                == {k: v for k, v in after.items() if k not in allowed_changes}, "Source manifest fields changed")
        require(not original.structural_evidence and len(derived.structural_evidence) == 2, "Proposal count differs")
        require(original.semantic_hash != derived.semantic_hash == integrated["derived_semantic_hash"], "Wrong enriched semantic identity")
        require(original.planning_hash == derived.planning_hash == integrated["planning_hash"], "Planning identity changed")
        require(derived.revision == original.revision + 2 == integrated["derived_revision"], "Unexpected revision change")
        require(original.revision == integrated["original_revision"], "Original revision receipt differs")
        require(original.brain_mask is derived.brain_mask is None and original.context is derived.context is None,
                "Working anatomy or unknown-timed context changed")
        require(not declared_mri_support_allowed(derived) and planning_brain_support(derived) == (None, {}),
                "Full-head or proposal data became working support")
        t1 = checked(source["source_t1"], source["source_t1_sha256"])
        annotation = checked(source["source_annotation"], source["source_annotation_sha256"])
        reference = nib.load(t1)
        mri, geometry = read_native(t1, reference, binary=False)
        require(np.array_equal(original.mri, mri) and np.array_equal(derived.mri, mri), "MRI samples changed")
        require(np.array_equal(original.affine, reference.affine) and np.array_equal(derived.affine, reference.affine), "Affine changed")
        threshold = derived.metadata["fractional_annotation"]["threshold"]
        require(threshold == source["annotation_threshold"] == 0.5, "Annotation threshold changed")
        target = source_annotation(annotation, reference, threshold=threshold)
        for group in ("compartments", "source_compartments"):
            left, right = getattr(original, group), getattr(derived, group)
            require(left.keys() == right.keys() and len(right) == 1, "Annotation compartment identity changed")
            for key in left:
                require(np.array_equal(left[key], right[key]) and np.array_equal(right[key], target), "Annotation voxels changed")
        require(int(target.sum()) == source["declared_annotation_voxels"], "Annotation count differs")
        old_artifacts, new_artifacts = read_case_artifacts(original_path), read_case_artifacts(derived_path)
        require({k: v for k, v in new_artifacts.items() if k != "structural_proposal_integration"} == old_artifacts,
                "Original saved artifacts changed")
        run = next(r for r in declaration["runs"] if r["subject"] == subject and r["repetition"] == 1)
        attempt = next(a for a in batch["attempts"] if a["run_id"] == run["run_id"])
        directory = run["output_directory"]
        report_path = checked(f"{directory}/brain_extraction_report.json", attempt["report_sha256"])
        portable = new_artifacts["structural_proposal_integration"]
        report = check_portable_artifacts(portable, report_path.read_bytes(), directory)
        for entry in portable["external_extraction_artifacts"]:
            checked(entry["path"], entry["sha256"], entry["bytes"])
        evidence_results = []
        require({e.metadata["variant"] for e in derived.structural_evidence.values()} == {"main", "nocsf"}, "Variant missing")
        for evidence in derived.structural_evidence.values():
            variant = evidence.metadata["variant"]
            record = report["variants"][variant]
            inference = record["inference"]
            metadata = thaw_json(evidence.metadata)
            frozen_mask = repository / directory / f"{variant}_mask.nii.gz"
            mask, mask_geometry = read_native(frozen_mask, reference, binary=True)
            require(np.array_equal(mask, evidence.mask) and not evidence.mask.flags.writeable, "Proposal mask changed or mutable")
            require(evidence.review is None and evidence.review_status == "review_required"
                    and evidence.provenance == "estimated" and evidence.cortical_access_permitted is False, "Proposal review changed")
            evidence.assert_matches(derived)
            weight = "synthstrip.1.pt" if variant == "main" else "synthstrip.nocsf.1.pt"
            require(evidence.source_file_sha256 == "sha256:" + source["source_t1_sha256"]
                    and evidence.source_image_hash == array_digest(mri)
                    and evidence.source_frame_hash == structural_frame_hash(original)
                    and evidence.run_sha256 == "sha256:" + attempt["report_sha256"]
                    and evidence.model_sha256 == "sha256:" + declaration["model_assets"][weight]["sha256"], "Proposal provenance differs")
            require(metadata["model"] == inference["model"] and metadata["source_qc"] == record["qc"]
                    and metadata["inference_configuration"] == inference["configuration"]
                    and metadata["runtime_versions"] == inference["runtime_versions"]
                    and metadata["executed_runner_sha256"] == inference["executed_runner_sha256"], "Embedded run metadata differs")
            require(metadata["mask_file_sha256"] == digest(frozen_mask)
                    and metadata["mask_source_uri"] == frozen_mask.resolve().as_uri()
                    and metadata["report_source_uri"] == report_path.resolve().as_uri(), "Proposal artifact binding differs")
            outside = int(np.count_nonzero(target & ~mask))
            require(metadata["current_target_annotation_outside_voxels"] == record["qc"]["source_annotation_outside_voxels"] == outside,
                    "Omission count differs")
            require(metadata["current_target_union_hash"] == array_digest(target)
                    and metadata["current_target_voxels"] == int(target.sum()), "Target binding differs")
            require(set(metadata["qc_flags"]) == set(record["qc"]["flags"]) | {
                "CURRENT_TARGET_ANNOTATION_OUTSIDE_ESTIMATED_ENVELOPE"}, "Omission flag lost")
            require(metadata["cortex_localized"] is False and metadata["clinical_deficit_probability"] is None, "Proposal clinical claim changed")
            try:
                validate_explicit_support(derived, evidence.mask, {})
            except ValueError as exc:
                require(str(exc).startswith("BRAIN_MASK_REVIEW_REQUIRED:"), "Unexpected access rejection")
            else:
                raise ValueError("Unreviewed proposal accepted as working support")
            evidence_results.append({"variant": variant, "evidence_hash": evidence.evidence_hash,
                "mask_hash": evidence.mask_hash, "mask_geometry": mask_geometry, "omitted_annotation_voxels": outside,
                "source_model_run_bound": True, "proposal_as_working_support_rejected": True})
        roundtrip = repository / "outputs/portable-proposal-independent" / derived_path.name
        save_case(derived, roundtrip, artifacts=new_artifacts)
        reopened = load_case(roundtrip)
        require(reopened.semantic_hash == derived.semantic_hash and reopened.planning_hash == derived.planning_hash,
                "Independent save/reopen identity differs")
        require(manifest(roundtrip) == after and read_case_artifacts(roundtrip) == new_artifacts,
                "Independent save/reopen manifest differs")
        require(np.array_equal(reopened.mri, derived.mri) and np.array_equal(reopened.affine, derived.affine)
                and reopened.brain_mask is None and reopened.context is None, "Independent save/reopen source changed")
        for key, evidence in reopened.structural_evidence.items():
            require(evidence.to_manifest() == derived.structural_evidence[key].to_manifest()
                    and np.array_equal(evidence.mask, derived.structural_evidence[key].mask), "Independent reopened proposal differs")
        results.append({"subject": subject, "source_bundle_sha256": digest(original_path),
            "derived_bundle": integrated["derived_case"], "derived_bundle_sha256": digest(derived_path),
            "original_revision": original.revision, "derived_revision": derived.revision,
            "original_semantic_hash": original.semantic_hash, "derived_semantic_hash": derived.semantic_hash,
            "planning_hash": derived.planning_hash, "planning_hash_unchanged": True,
            "source_MRI_affine_active_and_original_annotation_exact": True, "source_geometry": geometry,
            "source_files_verified": len(acquisition["files"]), "annotation_threshold": threshold,
            "annotation_voxels": int(target.sum()), "selected_run": run["run_id"],
            "embedded_report_sha256": attempt["report_sha256"], "evidence": evidence_results,
            "independent_save_reopen_exact": True, "roundtrip_sha256": digest(roundtrip),
            "working_brain_mask": None, "unknown_timed_context_excluded": True,
            "mask_arrays_embedded": True, "predicted_distance_arrays_embedded": False})
        del original, derived, reopened, mri, target, mask
    for relative, expected in tracked.items():
        verify_file(repository / relative, expected)
    return {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "audit_kind": "independent_PAT16_PAT20_portable_proposal_engineering_QC", "engineering_checks_passed": True,
        "auditor_sha256": digest(Path(__file__)), "independent_array_helper_sha256": digest(Path(__file__).with_name("audit_brain_extraction.py")),
        "integration_receipt_sha256": INTEGRATION_SHA256, "extraction_declaration_sha256": DECLARATION_SHA256,
        "prior_independent_extraction_audits": EXTRACTION_AUDITS, "cases": results,
        "retained_input_files_verified_unchanged": len(tracked), "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "numerical_threads_requested": 1, "inference_executed": False, "downloads_performed": False,
        "brain_reviewed": False, "cortex_localized": False, "cortical_access_permitted": False,
        "clinical_deficit_probability": None,
        "limits": ["Engineering identity and portability checks do not establish anatomical accuracy.",
            "Masks and exact extraction-report text travel with the bundle; SDT arrays remain externally hashed artifacts.",
            "Raw fractional annotations remain externally hashed source files; embedded source compartments are threshold-derived binary masks.",
            "Historical runtime binary hashes were not predeclared; version records do not provide binary attestation."]}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    destination = root / "docs/brain-extraction-pat16-pat20-portable-independent-qc.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(destination), "sha256": digest(destination),
                      "elapsed_seconds": result["elapsed_seconds"], "cases_passed": len(result["cases"])}))
