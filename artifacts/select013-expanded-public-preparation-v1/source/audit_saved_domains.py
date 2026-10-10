"""Saved expanded SELECT013 arrays only; no runnable manifest or admission."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import signal
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = HERE / "actual-domain-qc-v1"
ORIGINAL = ROOT / "build/remind-select-public-preparation-v1"
COHORT = "326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05"
SUBJECTS = ("ReMIND-013",)
CONVERTER_SHA = "565843dac7efeef29640bb1690a7f1f167d38310d8b7e771babad5302571d796"


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


def domain_plane_counts(support, target, support_domain, target_domain):
    assert support.shape == target.shape == support_domain.shape == target_domain.shape
    assert all(a.dtype == np.uint8 and np.isin(a, (0, 1)).all() for a in (support, target, support_domain, target_domain))
    assert np.all((support == 0) | (support_domain == 1))
    assert np.all((target == 0) | (target_domain == 1))
    return {"support": int(support.sum()), "support_domain": int(support_domain.sum()),
            "target": int(target.sum()), "target_domain": int(target_domain.sum()),
            "target_outside_support": int(np.count_nonzero((target != 0) & (support == 0))),
            "target_in_known_support_positive": int(np.count_nonzero((target != 0) & (support != 0) & (support_domain != 0))),
            "target_in_known_support_zero": int(np.count_nonzero((target != 0) & (support == 0) & (support_domain != 0))),
            "target_in_unknown_support_domain": int(np.count_nonzero((target != 0) & (support_domain == 0)))}


def source_centres_inside_crop(geometry, crop_affine, crop_shape):
    points = np.asarray(list(itertools.product(*[(0, n-1) for n in geometry["shape_xyz"]])), dtype=float)
    mapped = (np.c_[points, np.ones(8)] @ (np.linalg.inv(crop_affine) @ np.asarray(geometry["affine_xyz_to_ras_mm"])).T)[:, :3]
    assert np.all(mapped >= -.5) and np.all(mapped < np.asarray(crop_shape)-.5)
    return {"all_source_grid_centres_inside_crop": True, "mapped_centre_minimum": mapped.min(0).tolist(), "mapped_centre_maximum": mapped.max(0).tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=SUBJECTS, required=True)
    parser.add_argument("--conversion-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(290); started = time.monotonic()
    output = args.output.resolve(); assert output == (BASE / (args.subject + "-saved-domain-review")).resolve(); output.mkdir(exist_ok=False)
    index = {"schema": "remind-public-select-expanded-saved-review-index-v1", "role": "SELECT", "max_optimizer_updates": 0,
             "cases": [], "failed_cases_replaced": False, "anatomical_accuracy_reviewed": False}
    pilot = ROOT / "manifests/experiments/remind-planning-pilot-v1.json"
    index_metadata = json.loads((HERE / "public-select-index.json").read_bytes())
    row = next(v for v in index_metadata["cases"] if v["patient_id"] == args.subject)
    for subject, result_sha in [(args.subject, args.conversion_sha256)]:
        folder = BASE / (subject + "-crop-mr-domains"); result_path = folder / "conversion-result.json"
        assert sha(result_path) == result_sha
        report = json.loads(result_path.read_bytes())
        header_path = ROOT / row["headers"]["path"]
        assert sha(header_path) == row["headers"]["sha256"]
        assert sha(header_path) == report["header_snapshot_sha256"]
        headers = json.loads(header_path.read_bytes())
        case_path = ROOT / row["case"]["path"]
        assert sha(case_path) == row["case"]["sha256"]
        assert sha(case_path) == report["case_sha256"]
        case = json.loads(case_path.read_bytes())
        assert report["role"] == case["role"] == headers["role"] == "SELECT"
        assert report["public_only"] and headers["public_only"] and not report["private_reference_loaded"]
        assert report["status"] == "public_crop_source_samples_and_label_grids_converted_anatomy_unreviewed"
        assert report["executing_script_sha256"] == CONVERTER_SHA
        assert report["crop_policy"] == "public_source_domain_union_no_extrapolation"
        assert report["full_coverage_factory_compatible"] is False and report["support_source_domain_required"] is True
        assert set(report["annotations"]) == {"whole_tumor", "cerebrum"}
        assert report["immutable_role_binding"]["cohort_sha256"] == COHORT
        for name, binding in report["artifacts"].items():
            assert Path(name).name == name and not (folder/name).is_symlink()
            assert (folder/name).stat().st_size == binding["bytes"] and sha(folder/name) == binding["sha256"]
        names = {"image": "MR_native_crop.npy", "supplied_support": "cerebrum_source_label.npy",
                 "supplied_whole_tumor": "whole_tumor_source_label.npy", "whole_tumor_domain": "whole_tumor_source_grid_domain.npy"}
        arrays = {key: np.load(folder/name, mmap_mode="r", allow_pickle=False) for key, name in names.items()}
        support_domain = np.load(folder / "cerebrum_source_grid_domain.npy", mmap_mode="r", allow_pickle=False)
        shape = tuple(report["shape_xyz"]); nvox = int(np.prod(shape)); affine = np.asarray(report["planning_derived_affine_ras_mm"])
        assert all(a.shape == shape for a in (*arrays.values(), support_domain))
        assert arrays["image"].dtype == np.float32 and all(a.dtype == np.uint8 for k, a in arrays.items() if k != "image")
        old_manifest_path = ORIGINAL / "public-select-inputs-v1/ReMIND-013-public-inputs-v1.json"
        assert sha(old_manifest_path) == "563963e7cf6ae7c1dff9f163db3208bf1af89b5611bc20e76a84c442a2b407a8"
        old_manifest = json.loads(old_manifest_path.read_bytes())
        diagnostic_path = ROOT / "build/select013-source-domain-diagnostic-v1/attempt-03/result.json"
        assert sha(diagnostic_path) == "531e60692158457077d1a11b951bb5b217b8333abd269968a651eca474828445"
        diagnostic = json.loads(diagnostic_path.read_bytes())
        assert report["public_crop_start_MR"] == [61,70,23] and list(shape) == [131,154,117]
        assert np.allclose(affine, diagnostic["derived_affine"], rtol=0, atol=1e-12)
        assert np.allclose(report["MR_native_crop_affine_ras_mm"], diagnostic["native_crop_affine"], rtol=0, atol=1e-12)
        overlap = tuple(slice(2,2+n) for n in old_manifest["shape_xyz"])
        overlap_hashes = {}
        for key, array in arrays.items():
            buf = io.BytesIO(); np.save(buf,np.ascontiguousarray(array[overlap]),allow_pickle=False)
            overlap_hashes[key] = hashlib.sha256(buf.getvalue()).hexdigest()
            assert overlap_hashes[key] == old_manifest["input_files"][key]["sha256"]
        assert np.all(support_domain[overlap] == 1)
        expanded_provenance = {"old_manifest_sha256":sha(old_manifest_path),
            "old_conversion_sha256":"9d126eaa926d7e6c4e11903d119a9446f1e38ea98993a652735cabd72ecd0fe9",
            "diagnostic_result_sha256":sha(diagnostic_path),"old_overlap_file_sha256":overlap_hashes,
            "old_support_placed_voxels":745523,"restored_support_placed_voxels":69,
            "original_full_target_voxels":35260,"old_crop_start_MR":[63,72,25],
            "expanded_crop_start_MR":[61,70,23],"native_MR_samples_retained":True,
            "fixed_opening_from_bound_diagnostic":diagnostic["unchanged_rule"],
            "preentry_point_bounds_from_bound_diagnostic":diagnostic["preentry_distal_point_bounds"],
            "old_sources_or_failed_attempt_modified":False,"new_access_or_route_admitted":False}
        count = {k: 0 for k in ("support", "support_domain", "target", "target_domain", "target_outside_support", "target_in_known_support_positive", "target_in_known_support_zero", "target_in_unknown_support_domain")}
        image_min, image_max = float("inf"), float("-inf")
        x, y = np.indices(shape[:2], dtype=float)
        transforms = {kind: np.linalg.solve(np.asarray(report["annotations"][kind]["source_native_affine_ras_mm"]), affine)
                      for kind in ("cerebrum", "whole_tumor")}
        for z in range(shape[2]):
            image = arrays["image"][:, :, z]; support = arrays["supplied_support"][:, :, z]
            target = arrays["supplied_whole_tumor"][:, :, z]; domain = arrays["whole_tumor_domain"][:, :, z]
            assert np.isfinite(image).all()
            image_min = min(image_min, float(image.min())); image_max = max(image_max, float(image.max()))
            plane_counts = domain_plane_counts(support, target, support_domain[:, :, z], domain)
            for kind, saved_domain in (("cerebrum", support_domain[:, :, z]), ("whole_tumor", domain)):
                t = transforms[kind]
                coordinates = t[:3, 0, None, None]*x+t[:3, 1, None, None]*y+(t[:3, 2]*z+t[:3, 3])[:, None, None]
                expected = np.all((coordinates >= -.5) & (coordinates < np.asarray(report["annotations"][kind]["source_native_shape"])[:, None, None]-.5), axis=0)
                assert np.array_equal(saved_domain, expected)
            for key, value in plane_counts.items(): count[key] += value
        relation = report["public_target_support_relation"]
        assert count["support"] > 0 and count["target"] > 0
        assert count["target"] == relation["whole_tumor_positive_voxels"]
        assert count["target_outside_support"] == relation["whole_tumor_positive_outside_public_support"]
        partition = report["public_target_support_domain_relation"]
        assert count["target"] == partition["full_placed_target_voxels"]
        for key in ("target_in_known_support_positive", "target_in_known_support_zero", "target_in_unknown_support_domain"):
            assert count[key] == partition[key]
        assert count["target"] == sum(count[k] for k in ("target_in_known_support_positive", "target_in_known_support_zero", "target_in_unknown_support_domain"))
        assert count["target_outside_support"] == count["target_in_known_support_zero"] + count["target_in_unknown_support_domain"]
        assert count["support"] == 745592 and count["target"] == 35260 and count["support_domain"] == 2184064
        assert partition["known_label_zero_is_physical_empty"] is False and partition["unknown_label_zero_is_observed_empty"] is False
        assert report["public_support_source_domain_unknown_voxels"] == nvox-count["support_domain"]
        assert report["public_support_source_domain_complete"] == (count["support_domain"] == nvox)
        summaries = {}
        volume = abs(float(np.linalg.det(affine[:3, :3])))
        for kind, total, domains in (("cerebrum", count["support"], count["support_domain"]), ("whole_tumor", count["target"], count["target_domain"])):
            annotation = report["annotations"][kind]; placement = annotation["placement"]
            assert placement["source_positive_centres_outside_target_grid"] == 0
            assert annotation["source_samples_equal"] is True
            assert total == annotation["placed_positive_voxels"] == placement["resampled_positive_voxels"]
            assert domains == annotation["placed_source_domain_voxels"]
            summaries[kind] = {"saved_positive_voxels": total, "saved_domain_voxels": domains,
                "saved_positive_volume_mm3": total*volume,
                "source_positive_centres_cropped_from_bound_execution": placement["source_positive_centres_outside_target_grid"],
                "source_positive_voxels_from_bound_execution": placement["source_positive_voxels"],
                "source_positive_volume_mm3_from_bound_execution": placement["source_positive_volume_mm3"],
                "resampling_can_omit_subvoxel_labels": placement["resampling_can_omit_subvoxel_labels"],
                "volume_change_percent": 100*(total*volume/placement["source_positive_volume_mm3"]-1)}
        geometry = {row["kind"]: fit_saved_geometry(row["geometry"]) for row in headers["series"]}
        centre_coverage = {row["kind"]: source_centres_inside_crop(row["geometry"], affine, shape) for row in headers["series"] if row["kind"] in ("cerebrum", "whole_tumor")}
        limits = ["Only an annotation-assisted geometric research condition; no anatomical or clinical accuracy pass.",
            "Full native and placed target denominators remain; label-positive centre crop loss is zero, NN sampling differences stay explicit.",
            "Known automatic label zero is not physical empty. Outside its saved source domain is explicitly unknown; no filling or full-coverage factory admission.",
            "Source whole tumor is not a prescribed resection target. Missing vessels and function remain unknown.",
            "Exact acquired/annotation availability timestamps are unverified. This audit permits zero optimizer updates.",
            "NN sampling may omit thin labels; source positive crop losses and volume changes remain explicit."]
        visual = "PENDING_ROOT_VISUAL_AND_SOURCE_ARTIFACT_REVIEW; no visual claim generated by this reader."
        audit = {"status": "saved_public_domain_geometry_checked_anatomy_unreviewed", "patient_id": subject,
            "role": "SELECT", "original_source_reopened": False, "private_reference_loaded": False, "public_only": True, "training_admitted": False, "actor_inputs_admitted": False, "optimizer_updates_performed": 0, "executing_script_sha256": sha(__file__), "case_sha256": report["case_sha256"], "full_coverage_factory_compatible": False,
            "conversion_receipt_sha256": result_sha, "header_snapshot_sha256": sha(header_path),
            "shape_xyz": shape, "source_geometry_refit_from_saved_headers": geometry,
            "saved_counts": count, "image_range": [image_min, image_max], "label_summaries": summaries,
            "public_support_domain_fully_covered": count["support_domain"] == nvox, "support_unknown_domain_voxels": nvox-count["support_domain"], "source_centre_coverage": centre_coverage, "public_input_manifest": None,
            "source_MR_sample_equality": "bound conversion worker independently compared decoded samples to original bytes; this saved audit checks finite saved samples only",
            "all_saved_artifact_hashes_verified": True, "all_source_domain_voxels_world_checked": True,
            "visual_review": {"views": report["public_overlays"]["images"], "observation": visual, "expert_anatomy_review": False},
            "expanded_preparation_provenance":expanded_provenance,
            "limits": limits, "audit_script_sha256": sha(__file__)}
        audit_path = output / "saved-domain-review.json"; audit_sha = save(audit_path, audit)
        index["cases"].append({"patient_id": subject, "patient_group": case["patient_group"], "role": "SELECT",
            "public_input_manifest": None, "training_admitted": False, "source_artifact_clearance": False,
            "saved_review_path": str(audit_path.relative_to(ROOT)), "saved_review_sha256": audit_sha})
        del arrays, support_domain
    index["elapsed_seconds"] = time.monotonic()-started
    save(output / "saved-review-index.json", index)
    print(json.dumps(index))


if __name__ == "__main__": main()
