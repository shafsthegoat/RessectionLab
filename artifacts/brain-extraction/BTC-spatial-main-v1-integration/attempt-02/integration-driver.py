"""Post-inference QC and portable unreviewed proposals; no inference or training."""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


ROOT = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
INITIAL = ROOT / "artifacts/brain-extraction/BTC-spatial-main-v1-batch"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


release = json.loads((INITIAL / "root-release.json").read_text())
assert digest(INITIAL / "inference-batch.json") == "57b41c12d937358165ca4aa8e46d0085b9577ee1619034a375e2c90d84bbc975"
batch = json.loads((INITIAL / "inference-batch.json").read_text())
assert batch["status"] == "all_four_inferences_completed_before_overlap_QC" and len(batch["attempts"]) == 4
snapshot = Path(release["snapshot"])
declaration_path = snapshot / "manifests/experiments/brain-extraction-btc-spatial-main-v1.json"
assert digest(declaration_path) == release["declaration_sha256"]
declaration = json.loads(declaration_path.read_text())
for name, checksum in release["snapshot_files_sha256"].items():
    assert digest(snapshot / name) == checksum
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(snapshot / "src")]
sys.path.append(release["main_site_packages"])
import numpy as np
from resectionlab.brain_extraction import extraction_qc, validate_mask, validate_distance_map
from resectionlab.core import thaw_json
from resectionlab.imaging import import_brain_extraction_evidence, load_case, read_case_artifacts, save_case


receipt_path = HERE / "integration-record.json"
assert not receipt_path.exists(), "Existing integration must be inspected, not repeated"
state = {"schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
         "driver_sha256": digest(__file__), "root_release_sha256": digest(HERE / "root-release.json"),
         "inference_batch_sha256": digest(INITIAL / "inference-batch.json"), "cases": [], "failures": [],
         "inference_or_training_performed": False, "independent_review_status": "pending_separate_receipt",
         "brain_reviewed": False, "working_brain_mask_attached": False, "cortical_access_permitted": False}
tracked = dict(release["source_files_sha256_before"])
for attempt in batch["attempts"]:
    directory = Path(attempt["output_directory"])
    for filename, item in attempt["artifact_files"].items():
        path = directory / filename
        assert digest(path) == item["sha256"]
        tracked[str(path.relative_to(ROOT))] = item["sha256"]
save(receipt_path, state)
started = time.perf_counter()
try:
    for source in declaration["subjects"]:
        subject = source["subject"]
        original_path = ROOT / source["source_case"]
        output_path = ROOT / source["proposed_evidence_bundle"]
        qc_directory = ROOT / "outputs/brain-extraction/BTC-spatial-main-v1-qc" / subject
        assert not output_path.exists() and not qc_directory.exists(), "Never overwrite original or prior derived artifacts"
        manifest = json.loads((ROOT / source["source_manifest"]).read_text())
        for entry in manifest["files"]:
            path = ROOT / "data/diffusion_source/ds001226-v5.0.1" / entry["path"]
            assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]
            tracked[str(path.relative_to(ROOT))] = entry["sha256"]
        original = load_case(original_path)
        assert original.semantic_hash == source["source_case_semantic_hash"]
        assert original.planning_hash == source["source_case_planning_hash"]
        assert original.brain_mask is None and not original.structural_evidence
        assert original.metadata["split"]["development_role"] == source["development_role"]
        assert original.metadata["split"]["policy_training_allowed"] is source["policy_training_allowed"]
        assert original.metadata["target_policy_input_allowed"] is False
        assert original.metadata["fractional_annotation"]["threshold"] == 0.5
        run_directory = ROOT / source["output_directory"]
        mask_path, distance_path = run_directory / "main_mask.nii.gz", run_directory / "main_distance_mm.nii.gz"
        mask, native_qc = validate_mask(mask_path, ROOT / source["source_t1"])
        distance_qc = validate_distance_map(distance_path, ROOT / source["source_t1"])
        target = np.logical_or.reduce(list(original.compartments.values()))
        qc = extraction_qc(mask, original.affine, tumor=target)
        report_path = qc_directory / "brain_extraction_report.json"
        original_report_path = run_directory / "brain_extraction_report.json"
        report = json.loads(original_report_path.read_text())
        report.update(created_at=datetime.now(timezone.utc).isoformat(),
                      initial_inference_report_sha256=digest(original_report_path),
                      fixed_inference_batch_sha256=digest(INITIAL / "inference-batch.json"),
                      post_inference_QC_only=True, source_annotation={
                          "derivation": thaw_json(original.metadata["fractional_annotation"]),
                          "case_source_annotation_meaning": "unchanged threshold-derived source annotation; QC only, not independent truth"})
        report["variants"]["main"]["qc"] = qc
        qc_directory.mkdir(parents=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        updated = import_brain_extraction_evidence(original, source_image_path=ROOT / source["source_t1"],
                    mask_path=mask_path, report_path=report_path, variant="main")
        evidence = next(iter(updated.structural_evidence.values()))
        assert updated.brain_mask is None and evidence.review is None and evidence.review_status == "review_required"
        assert evidence.provenance == "estimated" and evidence.cortical_access_permitted is False
        assert updated.metadata == original.metadata and updated.planning_hash == original.planning_hash
        assert updated.semantic_hash != original.semantic_hash
        assert evidence.metadata["current_target_annotation_outside_voxels"] == qc["source_annotation_outside_voxels"]
        for name in ("mri", "affine"):
            assert np.array_equal(getattr(updated, name), getattr(original, name))
        for name in ("compartments", "source_compartments"):
            assert getattr(updated, name).keys() == getattr(original, name).keys()
            assert all(np.array_equal(value, getattr(original, name)[key]) for key, value in getattr(updated, name).items())
        assert updated.source_refs == original.source_refs and updated.context == original.context
        artifacts = read_case_artifacts(original_path)
        assert "structural_proposal_integration" not in artifacts
        portable = {"schema_version": 1, "variant": "main", "selected_repetition": "only_declared_run",
                    "declaration_sha256": digest(declaration_path), "inference_batch_sha256": digest(INITIAL / "inference-batch.json"),
                    "initial_inference_report_sha256": digest(original_report_path),
                    "extraction_report_utf8": report_path.read_text(), "extraction_report_sha256": digest(report_path),
                    "mask_arrays_embedded": True, "predicted_distance_arrays_embedded": False,
                    "brain_reviewed": False, "cortical_access_permitted": False, "clinical_deficit_probability": None,
                    "independent_extraction_audits": [], "independent_review_status": "pending_separate_receipt",
                    "historical_PAT05_runtime_binary_equivalence_attested": False,
                    "external_extraction_artifacts": [
                        {"variant": "main", "artifact_kind": kind, "path": str(path.relative_to(ROOT)),
                         "sha256": digest(path), "bytes": path.stat().st_size,
                         "native_array_embedded_as_proposal": kind == "source_mask", "embedded_as_original_file": False}
                        for kind, path in (("source_mask", mask_path), ("predicted_signed_distance", distance_path))]}
        save_case(updated, output_path, artifacts={**artifacts, "structural_proposal_integration": portable})
        reopened = load_case(output_path)
        assert reopened.semantic_hash == updated.semantic_hash and reopened.planning_hash == original.planning_hash
        assert reopened.brain_mask is None and reopened.metadata == original.metadata
        assert read_case_artifacts(output_path) == {**artifacts, "structural_proposal_integration": portable}
        re_evidence = next(iter(reopened.structural_evidence.values()))
        assert np.array_equal(re_evidence.mask, mask) and re_evidence.review_status == "review_required"
        row = {"subject": subject, "development_role": source["development_role"], "patient_group": source["patient_group"],
               "original_case": source["source_case"], "original_case_sha256": digest(original_path),
               "derived_case": str(output_path.relative_to(ROOT)), "derived_case_sha256": digest(output_path),
               "derived_case_bytes": output_path.stat().st_size, "original_semantic_hash": original.semantic_hash,
               "derived_semantic_hash": reopened.semantic_hash, "original_revision": original.revision,
               "derived_revision": reopened.revision, "planning_hash": reopened.planning_hash,
               "planning_hash_unchanged": True, "source_MRI_affine_annotations_metadata_unchanged": True,
               "reopened_identical": True, "working_brain_mask": None, "review_status": re_evidence.review_status,
               "proposal_ready_status": "saved_reopened_unreviewed_proposal; independent_QC_pending",
               "evidence_id": re_evidence.evidence_id, "evidence_hash": re_evidence.evidence_hash,
               "source_file_sha256": re_evidence.source_file_sha256, "model_sha256": re_evidence.model_sha256,
               "run_sha256": re_evidence.run_sha256, "qc_flags": list(re_evidence.metadata["qc_flags"]),
               "native_mask_QC": native_qc, "distance_QC": distance_qc, "annotation_overlap_and_geometry_QC": qc,
               "qc_report": str(report_path.relative_to(ROOT)), "qc_report_sha256": digest(report_path),
               "mask_arrays_embedded": True, "predicted_distance_arrays_embedded": False}
        state["cases"].append(row)
        save(receipt_path, state)
        print(json.dumps({"subject": subject, "saved": str(output_path), "omitted_annotation_voxels": qc["source_annotation_outside_voxels"]}), flush=True)
        del original, updated, reopened, evidence, re_evidence, mask, target
        gc.collect()
    state["status"] = "completed"
except BaseException as error:
    state["status"] = "failed"
    state["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
finally:
    state["original_source_and_inference_files_sha256"] = tracked
    state["original_source_and_inference_files_unchanged"] = all(digest(ROOT / p) == h for p, h in tracked.items())
    state["frozen_source_unchanged"] = all(digest(snapshot / p) == h for p, h in release["snapshot_files_sha256"].items())
    state["project_module_origins"] = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                                      if name.startswith("resectionlab") and getattr(module, "__file__", None)}
    assert all(Path(p).is_relative_to(snapshot / "src") for p in state["project_module_origins"].values())
    if not state["original_source_and_inference_files_unchanged"] or not state["frozen_source_unchanged"]:
        state["status"] = "failed"
    state.update(elapsed_seconds=time.perf_counter() - started, finished_at=datetime.now(timezone.utc).isoformat(),
                 peak_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(receipt_path, state)
print(json.dumps({"status": state["status"], "cases": len(state["cases"]), "failures": state["failures"],
                  "elapsed_seconds": state["elapsed_seconds"], "receipt_sha256": digest(receipt_path)}), flush=True)
raise SystemExit(0 if state["status"] == "completed" else 1)
