#!/usr/bin/env python3
"""Independent saved-artifact audit of the frozen PAT16/PAT20 extraction batch.

No model import, inference, training, source mutation or anatomy approval.
The established independent audit helpers supply native-grid and mask checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

# Keep this audit separate from competing patient inference/timing workloads.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import nibabel as nib
import numpy as np
from scipy import ndimage

from audit_brain_extraction import digest, measure, read_native, source_annotation, verify_file


DECLARATION = "manifests/experiments/brain-extraction-pat16-pat20-repeatability-v1.json"
DECLARATION_SHA256 = "5be3ade2eb6b383e796cdc34def48dae66a4fb68649b282723daa83e771477db"
BATCH = "artifacts/brain-extraction/PAT16-PAT20-repeatability-v1-batch"
SUBJECTS = ("sub-PAT16", "sub-PAT20")
VARIANTS = ("nocsf", "main")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_run_contract(declaration: dict, run: dict, attempt: dict, report: dict, subject: dict) -> None:
    """Compare saved settings before allowing any array-based acceptance."""
    require(attempt["run_id"] == run["run_id"] and attempt["subject"] == subject["subject"], "Run identity mismatch")
    require(attempt["argv"] == run["argv"] and attempt["output_directory"] == run["output_directory"], "Run command/scope mismatch")
    require(attempt["status"] == "completed" and attempt["exit_code"] == 0, "Declared run did not complete")
    start, end, created = (datetime.fromisoformat(value) for value in
                           (attempt["started_at"], attempt["finished_at"], report["created_at"]))
    require(all(value.tzinfo is not None for value in (start, end, created)) and start <= created <= end,
            "Saved report time is outside the recorded attempt")
    require(list(report["variants"]) == run["variant_order"] == list(VARIANTS), "Variant order changed")
    require(not report["brain_reviewed"] and not report["cortical_access_permitted"], "Extraction cannot approve anatomy")
    require(report["implementation_sha256"] == declaration["wrapper"]["sha256"], "Wrapper changed")
    require(report["source_t1_sha256"] == subject["source_t1_sha256"], "Wrong source T1")
    require(report["source_annotation"]["sha256"] == subject["source_annotation_sha256"], "Wrong annotation")
    config = declaration["frozen_configuration"]
    require(report["source_annotation"]["geometry_and_threshold"]["threshold"] == config["annotation_threshold"], "Changed annotation threshold")
    for key in ("python", "numpy", "torch", "surfa", "scipy", "nibabel"):
        require(attempt["runtime_metadata_preflight"][key] == declaration["inference_runtime"][key], "Runtime preflight version mismatch")
    for name in VARIANTS:
        record = report["variants"][name]["inference"]
        qc = report["variants"][name]["qc"]
        require(record["exit_code"] == 0 and record["failure"] is None, "Child inference failed")
        require(record["input_sha256"] == subject["source_t1_sha256"], "Child input mismatch")
        require(record["executed_runner_sha256"] == declaration["expected_executed_MPS_runner_sha256"], "Runner changed")
        require(record["model"]["variant"] == name and record["model"]["upstream_code_commit"] == declaration["model_code_commit"], "Model identity mismatch")
        require(record["model"]["weights_version"] == declaration["weights_version"], "Model weight version mismatch")
        weight = "synthstrip.nocsf.1.pt" if name == "nocsf" else "synthstrip.1.pt"
        require(set(record["model"]["files"]) == {"mri_synthstrip", "LICENSE_FreeSurfer.txt", weight}, "Incomplete model asset identity")
        for key in ("device", "cpu_threads", "border_mm", "timeout_seconds", "maximum_rss_bytes"):
            require(record["configuration"][key] == config[key], f"Changed child configuration: {key}")
        require(record["configuration"]["weights_only_load"] is True, "Weights-only load not recorded")
        for key, value in record["runtime_versions"].items():
            require(value == declaration["inference_runtime"][key], "Child runtime version mismatch")
        require(not record["brain_reviewed"] and not record["cortical_access_permitted"], "Child anatomical approval forbidden")
        require(not qc["brain_reviewed"] and not qc["cortex_localized"] and not qc["cortical_access_permitted"], "QC anatomical approval forbidden")
        require(qc["clinical_deficit_probability"] is None, "Clinical probability must remain absent")
        require(attempt["children"][name]["artifact_hashes"] == record["artifact_hashes"], "Batch child hashes disagree")
        require(attempt["children"][name]["source_annotation_outside_voxels"] == qc["source_annotation_outside_voxels"], "Batch omission count disagrees")
        for filename, asset in record["model"]["files"].items():
            pinned = declaration["model_assets"][filename]
            require(asset["sha256"] == pinned["sha256"] and asset["bytes"] == pinned["bytes"], "Model asset pin mismatch")


def runtime_metadata(repository: Path, declaration: dict) -> dict:
    """Read distribution metadata without importing Torch or executing a model."""
    program = (
        "import importlib.metadata,json,sys; "
        "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),"
        "'executable':sys.executable,'packages':{n:importlib.metadata.version(n) "
        "for n in ('numpy','torch','surfa','scipy','nibabel')}}))"
    )
    result = subprocess.run([str(repository / declaration["inference_runtime"]["interpreter"]), "-c", program],
                            check=True, capture_output=True, text=True, timeout=30)
    actual = json.loads(result.stdout)
    require(actual["python"] == declaration["inference_runtime"]["python"], "Current runtime Python version changed")
    for name, version in actual["packages"].items():
        require(version == declaration["inference_runtime"][name], "Current runtime package version changed")
    return {**actual, "current_interpreter_sha256": digest(Path(actual["executable"]).resolve()),
            "historical_runtime_binary_hash_available": False,
            "scope": "Frozen and child package-version records agree. Interpreter hash is measured now; the declaration did not pin historical interpreter or installed-package binary hashes."}


def render_overlay(path: Path, subject: str, mri: np.ndarray, tumor: np.ndarray, masks: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    centers = [np.rint(ndimage.center_of_mass(masks["main"])).astype(int),
               np.rint(ndimage.center_of_mass(tumor)).astype(int)]
    sample = mri.ravel()[::64]
    low, high = np.percentile(sample[sample > 0], [1, 99])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), facecolor="#111923")
    for row, center in enumerate(centers):
        for axis in range(3):
            ax = axes[row, axis]
            index = int(center[axis])
            plane = np.take(mri, index, axis=axis).T
            ax.imshow(plane, cmap="gray", origin="lower", vmin=low, vmax=high)
            for label, array, color in (("main", masks["main"], "#8df5a7"),
                                        ("nocsf", masks["nocsf"], "#35d6ff"),
                                        ("annotation", tumor, "#fa80db")):
                contour = np.take(array, index, axis=axis).T
                if contour.any() and not contour.all():
                    ax.contour(contour, levels=[0.5], colors=[color], linewidths=0.8)
            ax.set_title(f'{"Envelope" if row == 0 else "Annotation"} centre · native axis {axis}, index {index}', color="white", fontsize=10)
            ax.axis("off")
    legend = [Line2D([0], [0], color=c, label=n) for n,c in (("Main envelope", "#8df5a7"), ("No-CSF envelope", "#35d6ff"), ("Source annotation ≥0.5", "#fa80db"))]
    fig.legend(handles=legend, loc="lower center", ncol=3, facecolor="#111923", labelcolor="white")
    fig.suptitle(f"{subject} · independent engineering overlays\nNative oblique planes; estimated envelopes remain unreviewed", color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"path": str(path), "sha256": digest(path), "native_voxel_centres": [c.tolist() for c in centers],
            "inspection_status": "rendered_pending_separate_visual_review"}


def audit(repository: Path, output_directory: Path) -> dict:
    started = time.perf_counter()
    declaration_path = repository / DECLARATION
    verify_file(declaration_path, DECLARATION_SHA256)
    declaration = json.loads(declaration_path.read_text())
    batch_directory = repository / BATCH
    batch_path = batch_directory / "batch_record.json"
    batch = json.loads(batch_path.read_text())
    verify_file(batch_directory / "frozen_declaration.json", DECLARATION_SHA256)
    verify_file(batch_directory / "execute_batch.py", batch["executor_sha256"])
    verify_file(batch_directory / "brain_extraction_snapshot.py", declaration["wrapper"]["sha256"])
    require(batch["declaration_sha256"] == DECLARATION_SHA256 and batch["status"] == "completed", "Batch declaration/status mismatch")
    require(batch["wrapper_sha256"] == declaration["wrapper"]["sha256"], "Batch wrapper mismatch")
    require(not batch["working_anatomy_changed"] and not batch["brain_reviewed"] and not batch["cortical_access_permitted"], "Batch must preserve unreviewed anatomy")
    require(len(batch["attempts"]) == len(declaration["runs"]) == 4, "Unexpected attempt count")
    require([a["run_id"] for a in batch["attempts"]] == [r["run_id"] for r in declaration["runs"]], "Attempt identity/order mismatch")
    times = [datetime.fromisoformat(declaration["declared_at"]), datetime.fromisoformat(batch["started_at"])]
    for attempt in batch["attempts"]:
        times.extend(datetime.fromisoformat(attempt[k]) for k in ("started_at", "finished_at"))
    times.append(datetime.fromisoformat(batch["finished_at"]))
    require(all(t.tzinfo is not None for t in times) and times == sorted(times), "Recorded times violate declared sequential order")
    for asset in declaration["model_assets"].values():
        path = repository / asset["path"]
        verify_file(path, asset["sha256"])
        require(path.stat().st_size == asset["bytes"], "Model asset size mismatch")
    runtime = runtime_metadata(repository, declaration)
    results = {}
    attempts = {a["run_id"]: a for a in batch["attempts"]}
    subjects = {s["subject"]: s for s in declaration["subjects"]}
    require(tuple(subjects) == SUBJECTS, "This audit is limited to the two declared subjects")
    for subject, source in subjects.items():
        for path_key, hash_key in (("source_manifest", "source_manifest_sha256"),
                                   ("independent_source_qc", "independent_source_qc_sha256"),
                                   ("source_t1", "source_t1_sha256"),
                                   ("source_annotation", "source_annotation_sha256")):
            verify_file(repository / source[path_key], source[hash_key])
        source_qc = json.loads((repository / source["independent_source_qc"]).read_text())
        verify_file(repository / source["prepared_case"], source_qc["bundle_sha256"])
        manifest = json.loads((repository / source["source_manifest"]).read_text())
        for entry in manifest["files"]:
            file = repository / "data/diffusion_source/ds001226-v5.0.1" / entry["path"]
            verify_file(file, entry["sha256"])
            require(file.stat().st_size == entry["bytes"], "Source size mismatch")
        reference = nib.load(repository / source["source_t1"])
        mri, source_geometry = read_native(repository / source["source_t1"], reference, binary=False)
        tumor = source_annotation(repository / source["source_annotation"], reference, threshold=source["annotation_threshold"])
        require(int(tumor.sum()) == source["declared_annotation_voxels"], "Independent target count disagrees")
        runs = [r for r in declaration["runs"] if r["subject"] == subject]
        per_run, repeat_arrays, overlay_masks = {}, {}, {}
        repeat_checks = {}
        for run in runs:
            directory = repository / run["output_directory"]
            attempt = attempts[run["run_id"]]
            report_path = directory / "brain_extraction_report.json"
            verify_file(report_path, attempt["report_sha256"])
            verify_file(batch_directory / f'{run["run_id"]}.log', attempt["log_sha256"])
            report = json.loads(report_path.read_text())
            check_run_contract(declaration, run, attempt, report, source)
            verify_file(directory / "implementation_snapshot.py", declaration["wrapper"]["sha256"])
            verify_file(directory / "LICENSE_FreeSurfer.txt", declaration["model_assets"]["LICENSE_FreeSurfer.txt"]["sha256"])
            variants = {}
            for name in VARIANTS:
                record = report["variants"][name]["inference"]
                require(json.loads((directory / f"{name}_inference_record.json").read_text()) == record, "Separate child record mismatch")
                verify_file(directory / f"{name}_synthstrip_mps.py", declaration["expected_executed_MPS_runner_sha256"])
                for filename, expected in record["artifact_hashes"].items():
                    verify_file(directory / filename, expected)
                mask, mask_geometry = read_native(directory / f"{name}_mask.nii.gz", reference, binary=True)
                distance, distance_geometry = read_native(directory / f"{name}_distance_mm.nii.gz", reference, binary=False)
                distance_dtype = nib.load(directory / f"{name}_distance_mm.nii.gz").get_data_dtype()
                require(distance_dtype == np.dtype(np.float32), "Unexpected stored SDT dtype; do not round before repeat comparison")
                distance_geometry["stored_dtype"] = str(distance_dtype)
                measured = measure(mask, distance, tumor, reference.affine, border_mm=1)
                stated = report["variants"][name]["qc"]
                for key in ("mask_voxels", "connected_components", "source_annotation_voxels", "source_annotation_outside_voxels"):
                    require(stated[key] == measured[key], f"Reported QC disagrees: {subject}/{name}/{key}")
                require(stated["boundary_face_voxels"] == measured["input_face_contact_voxels"], "Boundary count mismatch")
                require(np.isclose(stated["mask_volume_ml"], measured["volume_ml"], rtol=0, atol=1e-6), "Volume mismatch")
                require(np.isclose(stated["source_annotation_inclusion_fraction"], measured["source_annotation_inclusion_fraction"], rtol=0, atol=1e-12), "Inclusion mismatch")
                require((measured["source_annotation_outside_voxels"] > 0) == ("SOURCE_ANNOTATION_EXTENDS_OUTSIDE_EXTRACTION" in stated["flags"]), "Omission flag mismatch")
                variants[name] = {"mask_geometry": mask_geometry, "distance_geometry": distance_geometry,
                                  "independent_measurements": measured, "artifact_hashes": record["artifact_hashes"],
                                  "inference_record_sha256": digest(directory / f"{name}_inference_record.json"),
                                  "inference_log_sha256": digest(directory / f"{name}_inference.log")}
                if run["repetition"] == 1:
                    repeat_arrays[name] = (mask, distance)
                    overlay_masks[name] = mask
                else:
                    old_mask, old_distance = repeat_arrays.pop(name)
                    repeat = {"mask_differing_voxels": int(np.count_nonzero(old_mask != mask)),
                              "maximum_absolute_predicted_distance_difference_mm": float(np.max(np.abs(old_distance - distance))),
                              "predicted_distance_arrays_identical": bool(np.array_equal(old_distance, distance))}
                    for key, value in repeat.items():
                        require(batch["repeat_comparisons"][subject][name][key] == value, "Repeatability report mismatch")
                    repeat["compressed_artifact_hashes_identical"] = (
                        per_run[runs[0]["run_id"]]["variants"][name]["artifact_hashes"] == record["artifact_hashes"])
                    repeat_checks[name] = repeat
                    del old_mask, old_distance
                del distance, mask
            per_run[run["run_id"]] = {"source_report_sha256": digest(report_path), "variants": variants}
        overlay = render_overlay(output_directory / f'{subject}-independent-extraction.png', subject, mri, tumor, overlay_masks)
        results[subject] = {"source_geometry": source_geometry, "source_t1_sha256": source["source_t1_sha256"],
                            "source_annotation_sha256": source["source_annotation_sha256"],
                            "source_manifest_sha256": source["source_manifest_sha256"],
                            "source_annotation_voxels": int(tumor.sum()), "runs": per_run,
                            "repeats": repeat_checks, "visual_overlay": overlay, "source_case_bytes_unchanged": True}
        del reference, mri, tumor, overlay_masks
    return {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
            "audit_kind": "independent_frozen_PAT16_PAT20_saved_extraction_engineering_QC",
            "declaration_sha256": DECLARATION_SHA256, "declaration_commit": "2e54b98",
            "batch_record_sha256": digest(batch_path), "wrapper_sha256": declaration["wrapper"]["sha256"],
            "auditor_sha256": digest(Path(__file__)),
            "independent_helper_sha256": digest(Path(__file__).with_name("audit_brain_extraction.py")),
            "source_model_runner_artifact_pins_verified": True, "model_assets": declaration["model_assets"],
            "engineering_checks_passed": True, "visual_review_completed": False,
            "runtime": runtime, "recorded_execution_order_validated": True,
            "subjects": results, "elapsed_seconds": time.perf_counter() - started,
            "audit_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "audit_peak_rss_platform": sys.platform, "numerical_threads_requested": 1,
            "brain_reviewed": False, "cortex_localized": False, "cortical_access_permitted": False,
            "working_brain_masks_changed": False, "clinical_deficit_probability": None,
            "network_inference_executed": False, "training_executed": False,
            "limitations": ["Engineering consistency and repeatability only; no independent anatomy labels or clinical accuracy.",
                            "Source-annotation omission is preserved, not repaired or relabeled as safe.",
                            "Learned SDT, including 100-mm exterior fill, is not measured surgical clearance.",
                            "Runtime versions are recorded; historical runtime binary hashes were not predeclared.",
                            "Retained timestamps and logs cannot prove that no unrecorded runs occurred."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/brain-extraction-pat16-pat20-independent-qc.json"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    report = args.report.resolve()
    require(report.is_relative_to(repository / "docs") and report.name.startswith("brain-extraction-pat16-pat20-independent"), "Use a separately named independent docs report")
    result = audit(repository, repository / "outputs/brain-extraction-pat16-pat20-independent")
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(report), "elapsed_seconds": result["elapsed_seconds"],
                      "audit_peak_rss_bytes": result["audit_peak_rss_bytes"], "brain_reviewed": False}))


if __name__ == "__main__":
    main()
