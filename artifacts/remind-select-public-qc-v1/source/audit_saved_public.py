"""Saved public arrays only; plane-stream checks and exact four-input manifests."""
import hashlib
import itertools
import json
from pathlib import Path
import signal
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = HERE / "actual-public-qc-v1"
COHORT = "326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05"
RESULTS = {"ReMIND-013": "9d126eaa926d7e6c4e11903d119a9446f1e38ea98993a652735cabd72ecd0fe9",
           "ReMIND-037": "e3671d0089279f786225f63180159ae656f37d086aa159c5f622808a3198d41f"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""): h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream: json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    return sha(path)


def fit_saved_geometry(g):
    orientation = np.asarray(g["raw_orientation"], float); spacing = np.asarray(g["raw_pixel_spacing"], float)
    positions = np.asarray(g["raw_positions"], float)
    order = np.argsort(positions @ np.cross(orientation[:3], orientation[3:]), kind="stable")
    assert order.tolist() == g["sorted_source_indices"]
    positions = positions[order]; k = np.arange(len(positions), dtype=float); centered = k-k.mean()
    slope = (centered[:, None]*(positions-positions.mean(0))).sum(0)/(centered@centered)
    origin = positions.mean(0)-k.mean()*slope
    affine = np.eye(4); affine[:3, :3] = np.column_stack((orientation[:3]*spacing[1], orientation[3:]*spacing[0], slope)); affine[:3, 3] = origin
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    difference = float(np.max(np.abs(affine-np.asarray(g["affine_xyz_to_ras_mm"]))))
    residual = float(np.linalg.norm(positions-origin-k[:, None]*slope, axis=1).max())
    assert difference < 1e-9 and residual <= .001
    return {"centred_OLS_affine_max_difference_mm": difference, "plane_residual_mm": residual}


def main():
    signal.alarm(60); started = time.monotonic()
    output = HERE / "public-select-inputs-v1"; output.mkdir(exist_ok=False)
    index = {"schema": "remind-public-select-manifest-index-v1", "role": "SELECT", "max_optimizer_updates": 0,
             "cases": [], "failed_cases_replaced": False, "anatomical_accuracy_reviewed": False}
    pilot = ROOT / "manifests/experiments/remind-planning-pilot-v1.json"
    for subject, result_sha in RESULTS.items():
        folder = BASE / (subject + "-crop-mr"); result_path = folder / "conversion-result.json"
        assert sha(result_path) == result_sha
        report = json.loads(result_path.read_bytes())
        header_path = BASE / (subject + "-headers/result.json")
        assert sha(header_path) == report["header_snapshot_sha256"]
        headers = json.loads(header_path.read_bytes())
        case_path = ROOT / "build/remind-select-public-bindings-v1" / (subject + "-case.json")
        assert sha(case_path) == report["case_sha256"]
        case = json.loads(case_path.read_bytes())
        assert report["role"] == case["role"] == headers["role"] == "SELECT"
        assert report["public_only"] and headers["public_only"] and not report["private_reference_loaded"]
        assert report["status"] == "public_crop_source_samples_and_label_grids_converted_anatomy_unreviewed"
        assert set(report["annotations"]) == {"whole_tumor", "cerebrum"}
        assert report["immutable_role_binding"]["cohort_sha256"] == COHORT
        for name, binding in report["artifacts"].items():
            assert (folder/name).stat().st_size == binding["bytes"] and sha(folder/name) == binding["sha256"]
        names = {"image": "MR_native_crop.npy", "supplied_support": "cerebrum_source_label.npy",
                 "supplied_whole_tumor": "whole_tumor_source_label.npy", "whole_tumor_domain": "whole_tumor_source_grid_domain.npy"}
        arrays = {key: np.load(folder/name, mmap_mode="r", allow_pickle=False) for key, name in names.items()}
        support_domain = np.load(folder / "cerebrum_source_grid_domain.npy", mmap_mode="r", allow_pickle=False)
        shape = tuple(report["shape_xyz"]); nvox = int(np.prod(shape)); affine = np.asarray(report["planning_derived_affine_ras_mm"])
        assert all(a.shape == shape for a in (*arrays.values(), support_domain))
        assert arrays["image"].dtype == np.float32 and all(a.dtype == np.uint8 for k, a in arrays.items() if k != "image")
        count = {"support": 0, "target": 0, "target_domain": 0, "target_outside_support": 0}
        image_min, image_max = float("inf"), float("-inf")
        x, y = np.indices(shape[:2], dtype=float)
        transforms = {kind: np.linalg.solve(np.asarray(report["annotations"][kind]["source_native_affine_ras_mm"]), affine)
                      for kind in ("cerebrum", "whole_tumor")}
        for z in range(shape[2]):
            image = arrays["image"][:, :, z]; support = arrays["supplied_support"][:, :, z]
            target = arrays["supplied_whole_tumor"][:, :, z]; domain = arrays["whole_tumor_domain"][:, :, z]
            assert np.isfinite(image).all()
            image_min = min(image_min, float(image.min())); image_max = max(image_max, float(image.max()))
            assert all(np.isin(a, (0, 1)).all() for a in (support, target, domain, support_domain[:, :, z]))
            assert np.all((target == 0) | (domain == 1))
            for kind, saved_domain in (("cerebrum", support_domain[:, :, z]), ("whole_tumor", domain)):
                t = transforms[kind]
                coordinates = t[:3, 0, None, None]*x+t[:3, 1, None, None]*y+(t[:3, 2]*z+t[:3, 3])[:, None, None]
                expected = np.all((coordinates >= -.5) & (coordinates < np.asarray(report["annotations"][kind]["source_native_shape"])[:, None, None]-.5), axis=0)
                assert np.array_equal(saved_domain, expected)
            assert support_domain[:, :, z].all()
            count["support"] += int(support.sum()); count["target"] += int(target.sum()); count["target_domain"] += int(domain.sum())
            count["target_outside_support"] += int(np.count_nonzero((target != 0) & (support == 0)))
        relation = report["public_target_support_relation"]
        assert count["support"] > 0 and count["target"] > 0
        assert count["target"] == relation["whole_tumor_positive_voxels"]
        assert count["target_outside_support"] == relation["whole_tumor_positive_outside_public_support"]
        summaries = {}
        volume = abs(float(np.linalg.det(affine[:3, :3])))
        for kind, total, domains in (("cerebrum", count["support"], nvox), ("whole_tumor", count["target"], count["target_domain"])):
            annotation = report["annotations"][kind]; placement = annotation["placement"]
            assert total == annotation["placed_positive_voxels"] == placement["resampled_positive_voxels"]
            assert domains == annotation["placed_source_domain_voxels"]
            summaries[kind] = {"saved_positive_voxels": total, "saved_domain_voxels": domains,
                "saved_positive_volume_mm3": total*volume,
                "source_positive_centres_cropped_from_bound_execution": placement["source_positive_centres_outside_target_grid"],
                "source_positive_voxels_from_bound_execution": placement["source_positive_voxels"],
                "volume_change_percent": 100*(total*volume/placement["source_positive_volume_mm3"]-1)}
        geometry = {row["kind"]: fit_saved_geometry(row["geometry"]) for row in headers["series"]}
        limits = ["Only an annotation-assisted geometric research condition; no anatomical or clinical accuracy pass.",
            "Full supplied target remains the denominator; unchanged support excludes some target positives.",
            "Automatic support zeros are uncertain, informative public estimates, not confirmed empty anatomy; no filling.",
            "Source whole tumor is not a prescribed resection target. Missing vessels and function remain unknown.",
            "Exact acquired/annotation availability timestamps are unverified. SELECT permits zero optimizer updates.",
            "NN sampling may omit thin labels; source positive crop losses and volume changes remain explicit."]
        visual = ("Target visibly lies predominantly in the automatic support exclusion. Shared-frame geometry is consistent; clinical cause versus source annotation error remains uncertain."
                  if subject == "ReMIND-013" else
                  "Automatic support boundary appears striped on two orthogonal views. Header/grid and saved-sampling checks pass; annotation quality and the cause of this pattern remain uncertain.")
        audit = {"status": "saved_public_geometry_domains_samples_checked_anatomy_unreviewed", "patient_id": subject,
            "role": "SELECT", "original_source_reopened": False, "private_reference_loaded": False,
            "conversion_receipt_sha256": result_sha, "header_snapshot_sha256": sha(header_path),
            "shape_xyz": shape, "source_geometry_refit_from_saved_headers": geometry,
            "saved_counts": count, "image_range": [image_min, image_max], "label_summaries": summaries,
            "public_support_domain_fully_covered": True,
            "source_MR_sample_equality": "bound conversion worker independently compared decoded samples to original bytes; this saved audit checks finite saved samples only",
            "all_saved_artifact_hashes_verified": True, "all_source_domain_voxels_world_checked": True,
            "visual_review": {"views": report["public_overlays"]["images"], "observation": visual, "expert_anatomy_review": False},
            "limits": limits, "audit_script_sha256": sha(__file__)}
        audit_path = output / (subject + "-saved-public-review.json"); audit_sha = save(audit_path, audit)
        manifest = {"schema": "remind-supplied-annotation-public-inputs-v2", "patient_id": subject,
            "patient_group": case["patient_group"], "role": "SELECT", "public_only": True,
            "input_files": {key: {**report["artifacts"][name], "path": str(folder/name), "dtype": str(arrays[key].dtype)} for key, name in names.items()},
            "shape_xyz": shape, "affine_ras_mm": report["planning_derived_affine_ras_mm"],
            "source_MR_crop_affine_ras_mm": report["MR_native_crop_affine_ras_mm"], "reindex_policy": report["reindex_policy"],
            "public_grid_selection": report["source_cropping_map"], "public_support_domain_fully_covered": True,
            "public_label_resampling": {kind: {"placement": report["annotations"][kind]["placement"],
                "source_native_affine_ras_mm": report["annotations"][kind]["source_native_affine_ras_mm"],
                "source_native_shape": report["annotations"][kind]["source_native_shape"], "saved_volume_summary": summaries[kind]} for kind in summaries},
            "public_target_support_consistency": {"whole_tumor_positive_voxels": count["target"],
                "whole_tumor_positive_outside_supplied_support": count["target_outside_support"],
                "outside_fraction": count["target_outside_support"]/count["target"], "target_or_support_modified_for_task": False},
            "source_bindings": {"cohort_sha256": COHORT, "pilot_sha256": sha(pilot), "case_manifest_sha256": sha(case_path),
                "acquisition_source_binding_sha256": report["immutable_role_binding"]["source_binding_sha256"],
                "conversion_receipt_sha256": result_sha, "header_snapshot_sha256": sha(header_path),
                "converter_source_sha256": report["executing_script_sha256"], "saved_array_review_sha256": audit_sha},
            "public_source_series": [{k: s[k] for k in ("kind", "series_uuid", "SeriesInstanceUID", "StudyInstanceUID", "source_description", "annotation_provenance")} for s in case["series"]],
            "condition": "supplied preoperative source-described MRI, automatic cerebrum support and manual whole-tumor annotation; partial supported progress only",
            "task_condition": "PARTIAL_TARGET_PROGRESS", "private_evaluation_files_included": False,
            "anatomical_accuracy_reviewed": False, "clinical_suitability": False, "blanket_training_or_actor_admission": False,
            "intended_use_status": "saved geometry/domain/sampling qualification only; source annotation conflicts retained; separately bounded zero-update SELECT inference",
            "visual_quality_uncertainty": visual, "task_limits": limits}
        manifest_path = output / (subject + "-public-inputs-v1.json"); manifest_sha = save(manifest_path, manifest)
        index["cases"].append({"patient_id": subject, "patient_group": case["patient_group"], "role": "SELECT",
            "path": str(manifest_path.relative_to(ROOT)), "sha256": manifest_sha, "saved_review_sha256": audit_sha})
        del arrays, support_domain
    index["elapsed_seconds"] = time.monotonic()-started
    save(output / "public-manifest-index.json", index)
    print(json.dumps(index))


if __name__ == "__main__": main()
