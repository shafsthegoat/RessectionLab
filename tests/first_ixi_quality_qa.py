"""Opt-in independent fixed IXI265 failed-fit and vessel-set audit.

--released-payload-audit is required for real, already-local source decoding.
No ANTs, model, fitter, production diagnostic, GUI or bundle decoder is used.
Results are new records; existing receipts and prior audits are never rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/ixi265-quality-independent-v1"
METADATA = {
    "scripts/poc_ixi265_atropos.py": "4a5c5b4b2f85153212ac27e6356d5f1dca5c8b42d77257fb55a2fee53253624b",
    "artifacts/poc-ixi265-atropos-run-v1/declaration.json": "0a7358f2a6623953af832946af2c2b6dd94c948b32bf531cd2b688c3d2f83788",
    "artifacts/poc-ixi265-atropos-run-v1/outcome.json": "55f4dd46f7cc692d4a66799cefca21bfa37430d3f5cf2ebe3eda675b6e4c989e",
    "artifacts/poc-ixi265-atropos-run-v1/attempt.json": "d94e31d123ecced47179b025c778ebcb7c3f59f00f68b3ca2ed73ed487344ffc",
    "artifacts/poc-ixi265-atropos-preparation-v1/prefit-guard-followup.json": "9d9f5c8cb1493aacdb21b4d2e0a6aa559c409ff6dedcfe747d8e58d3e0e865f0",
    "scripts/poc_ixi265_vessel_agreement.py": "6065b78d86e1bfc8293ae6c32a27a0963e8d8ec2b5d939294579efc37b6dc668",
    "artifacts/poc-ixi265-vessel-agreement-v1/protocol.json": "027e97310ba84177b69d8f874a0a0d6b8d2f360d5a118987842baf422e631e66",
    "artifacts/poc-ixi265-vessel-agreement-v1/receipt.json": "b2efe42553817dba6900aedc2f6d4b7bd47cfcde002a4938a2fb2d248d2ea5c6",
    "outputs/poc-ixi265-vessel-agreement-v1/agreement.json": "bebf7ae7354fe198fda4f2ee24e422a385b84697f7096cfa8a9c8b5172b3e85e",
    "outputs/poc-ixi265-vessel-agreement-v1/runtime.json": "871e15aeb2039dd55ddacf8a96ef1e05af203df4a54a0030d1675fe84d1387a6",
}
FIT_INPUTS = {
    "T1": ("outputs/poc-ixi265/originals/IXI265-T1.nii.gz", "317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950"),
    "brain_mask": ("outputs/poc-ixi265/synthstrip/brain_mask.nii.gz", "98710d7d1fa578dd510bf40b953e6d34926d8ff0c766cba16c005ffb6dbff192"),
    "host_bundle": ("outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab", "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"),
}
VESSEL_INPUTS = {
    "MRA": ("outputs/poc-ixi265/originals/IXI265-MRA.nii.gz", "a87b381b882242b7d61a1930a8332d44200fdb3c56c68fc09c24a057eea86dc4"),
    "reference": ("outputs/poc-ixi265/originals/IXI265-derived_vessel_label.nii.gz", "9da13c332156c71a5200dbf88368bb87478914fe73034a42ca4eabf1dd5cdbfe"),
    "prediction": ("outputs/poc-ixi265/vesselfm-base-v1/vessels_mask.nii.gz", "d3eedb5a89e598cddd5eb017054f7fb0ce91c678278411c285cc00c2b8d277bc"),
    "score": ("outputs/poc-ixi265/vesselfm-base-v1/vessels_probability.nii.gz", "1745f34949bec22c4c2418e0f7e115c8e3acf0cd974e6ba4b18f7e81e80ecc75"),
    "coverage": ("outputs/poc-ixi265/vesselfm-base-v1/native_sampling_coverage.nii.gz", "a92bf82175e1208ebef037bf3117fc40f779c4cf5669f70422be7b0015bc863b"),
}
NEIGHBORS26 = [p for p in itertools.product((-1, 0, 1), repeat=3) if p != (0, 0, 0)]
NEIGHBORS6 = [(s if a == 0 else 0, s if a == 1 else 0, s if a == 2 else 0) for a in range(3) for s in (-1, 1)]
SAMPLES = []


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    p = Path(path)
    return {"sha256": sha(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def check_memory(phase):
    current = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    SAMPLES.append({"phase": phase, "own_rss_bytes": current, "available_bytes": available})
    assert current < 1024**3 and available >= 1024**3, "Independent audit memory guard"


def check_pins(mapping):
    result = {}
    for name, (path, pin) in mapping.items():
        row = stamp(ROOT / path)
        assert row["sha256"] == pin, "Changed pinned input: " + name
        result[name] = {"path": path, **row}
    return result


def bounds(points):
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()] if len(points) else None


def corners(shape, affine, *, cell):
    pairs = [(-0.5, n - 0.5) if cell else (0, n - 1) for n in shape]
    idx = np.asarray(list(itertools.product(*pairs)), dtype=np.float64)
    return np.column_stack((idx, np.ones(8))) @ np.asarray(affine).T


def frame(image, reference, *, require_header_consistency=False):
    assert image.shape == reference.shape
    result = {"shape": list(image.shape), "storage_dtype": str(image.get_data_dtype()),
              "affine_ras_mm": image.affine.tolist()}
    for mode in (False, True):
        err = np.linalg.norm((corners(image.shape, image.affine, cell=mode) -
                              corners(reference.shape, reference.affine, cell=mode))[:, :3], axis=1)
        assert err.max() < 1e-4
        result["eight_" + ("cell" if mode else "center") + "_corner_errors_mm"] = err.tolist()
    result["qform_sform"] = {}
    for kind in ("qform", "sform"):
        matrix, code = getattr(image, "get_" + kind)(coded=True)
        result["qform_sform"][kind + "_code"] = int(code)
        if code:
            err = np.linalg.norm((corners(image.shape, matrix, cell=True) -
                                  corners(image.shape, image.affine, cell=True))[:, :3], axis=1)
            if require_header_consistency:
                assert err.max() < 1e-4
            result["qform_sform"][kind + "_max_cell_corner_error_mm"] = float(err.max())
            result["qform_sform"][kind + "_within_0_0001_mm_of_selected_affine"] = bool(err.max() < 1e-4)
    return result


def sparse_components(points):
    """Sparse DFS, independently implemented; no scipy volume labelling."""
    remaining = {tuple(map(int, point)) for point in points}
    sizes = []
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            x, y, z = pending.pop()
            size += 1
            for dx, dy, dz in NEIGHBORS26:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        sizes.append(size)
    return sizes


def sparse_summary(mask, coverage):
    points = np.argwhere(mask & coverage)
    members = {tuple(map(int, p)) for p in points}
    sizes = sparse_components(points)
    boundary, coverage_edge = 0, 0
    shape = coverage.shape
    for x, y, z in members:
        neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in NEIGHBORS6]
        boundary += any(p not in members for p in neighbors)
        coverage_edge += any(not all(0 <= p[a] < shape[a] for a in range(3)) or not coverage[p] for p in neighbors)
    return {"connectivity": 26, "connected_components": len(sizes),
            "largest_component_voxels": max(sizes, default=0),
            "component_count_by_size": {"1": sizes.count(1), "2_to_9": sum(2 <= n <= 9 for n in sizes),
                "10_to_99": sum(10 <= n <= 99 for n in sizes), "100_or_more": sum(n >= 100 for n in sizes)},
            "boundary_connectivity": 6, "boundary_voxels": int(boundary),
            "foreground_on_sampling_domain_boundary_voxels": int(coverage_edge)}


def contingency(prediction, reference, coverage):
    """Eight joint-state bins; slab accumulation avoids full integer volumes."""
    table = np.zeros(8, np.int64)
    for i in range(prediction.shape[0]):
        code = (prediction[i].astype(np.uint8) + 2 * reference[i].astype(np.uint8)
                + 4 * coverage[i].astype(np.uint8))
        table += np.bincount(code.ravel(), minlength=8)
    return table


def counts_from_table(table, voxel_volume):
    a, b, c, d, e, f, g, h = map(int, table)
    pred, ref, union = f + h, g + h, f + g + h
    ratio = lambda x, y: x / y if y else None
    return {"sampling_domain_voxels": e + f + g + h, "native_grid_voxels": int(sum(table)),
        "native_prediction_voxels": b + d + f + h, "native_reference_positive_voxels": c + d + g + h,
        "prediction_in_domain_voxels": pred, "reference_in_domain_voxels": ref,
        "intersection_voxels": h, "union_voxels": union,
        "positive_mask_dice": ratio(2*h, pred + ref), "positive_mask_jaccard": ratio(h, union),
        "provided_positive_reference_coverage_fraction": ratio(h, ref),
        "reference_positive_voxels_not_covered_by_prediction": g,
        "prediction_in_UNLABELLED_reference_voxels": f,
        "reference_positive_fraction_retained_by_sampling_domain": ratio(ref, c + d + g + h),
        "prediction_voxels_excluded_by_sampling_domain": b + d,
        "reference_voxels_excluded_by_sampling_domain": c + d,
        "physical_volumes_mm3": {"prediction_in_domain": pred*voxel_volume,
            "reference_in_domain": ref*voxel_volume, "intersection": h*voxel_volume}}


def generated_controls():
    cube = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    assert contingency((cube & 1) != 0, (cube & 2) != 0, (cube & 4) != 0).tolist() == [1]*8
    zero = counts_from_table(np.zeros(8, np.int64), 1)
    assert all(zero[k] is None for k in ("positive_mask_dice", "positive_mask_jaccard", "provided_positive_reference_coverage_fraction"))
    coverage = np.zeros((6, 6, 6), bool); coverage[1:-1, 1:-1, 1:-1] = True
    mask = np.zeros_like(coverage); mask[1, 1, 1] = True; mask[2, 2, 2] = True; mask[4, 4, 4] = True
    result = sparse_summary(mask, coverage)
    assert result["connected_components"] == 2 and result["largest_component_voxels"] == 2
    assert result["boundary_voxels"] == 3 and result["foreground_on_sampling_domain_boundary_voxels"] == 2
    angle = 0.2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4); affine[:3, :3] = rotation @ np.diag([-0.7, 1.3, 2.0]); affine[:3, 3] = [4, -8, 2]
    shape = (3, 4, 5)
    ras = corners(shape, affine, cell=True)
    lps = np.diag([-1, -1, 1, 1]) @ affine
    assert np.array_equal(corners(shape, lps, cell=True)[:, :3], ras[:, :3] * [-1, -1, 1])
    assert np.isclose(abs(np.linalg.det(affine[:3, :3])), 0.7*1.3*2.0)
    return {"truth_table_all_eight_states": True, "empty_denominators_null": True,
            "diagonal_26_connectivity_and_6_boundary": True, "reflected_oblique_anisotropic_cell_corner_transport": True}


def audit_failed_fit():
    import nibabel as nib
    before = check_pins(FIT_INPUTS)
    outcome = json.loads((ROOT / "artifacts/poc-ixi265-atropos-run-v1/outcome.json").read_text())
    attempt = json.loads((ROOT / "artifacts/poc-ixi265-atropos-run-v1/attempt.json").read_text())
    assert outcome["status"] == "failed_before_N4_or_Atropos" and attempt["status"] == "failed"
    assert outcome["attempts_started"] == 1 and outcome["successful_fits"] == outcome["accepted_tissue_results"] == 0
    assert outcome["automatic_retries"] == 0 and outcome["worker_exit_code"] == attempt["exit_code"] == 1
    assert attempt["patient_group"] == "IXI:265" and attempt["role"] == "TRAIN" and attempt["training_admitted"] is False
    assert not any(attempt[k] for k in ("anatomy_accepted", "host_accepted", "growth_use_accepted"))
    assert attempt["parameters"]["priors"] is None and attempt["parameters"]["intensity_clipping_or_normalization"] is None
    images = {k: nib.load(ROOT / FIT_INPUTS[k][0]) for k in ("T1", "brain_mask")}
    source, mask_image = images["T1"], images["brain_mask"]
    frames = {k: frame(im, source, require_header_consistency=True) for k, im in images.items()}
    values = np.asarray(source.dataobj, dtype=np.float32)
    raw_mask = np.asarray(mask_image.dataobj)
    assert set(np.unique(raw_mask)) == {0, 1}
    mask = raw_mask > 0
    predicates = {"three_dimensions": values.ndim == 3, "within_12000000_voxels": values.size <= 12_000_000,
        "mask_shape_matches": mask.shape == values.shape, "mask_dtype_bool": mask.dtype == np.bool_,
        "mask_nonempty": bool(mask.any()), "all_source_values_finite": bool(np.isfinite(values).all()),
        "all_masked_source_values_strictly_positive": bool(np.all(values[mask] > 0))}
    assert [name for name, passed in predicates.items() if not passed] == ["all_masked_source_values_strictly_positive"]
    rejected = np.argwhere(mask & (values <= 0))
    assert len(rejected) == 8 and np.count_nonzero(mask & (values < 0)) == 0
    world = rejected @ source.affine[:3, :3].T + source.affine[:3, 3]
    voxel_volume = abs(float(np.linalg.det(source.affine[:3, :3])))
    details = []
    for idx, ras in zip(rejected, world):
        coords = [tuple(idx + off) for off in NEIGHBORS26]
        neighboring = np.array([values[p] for p in coords])
        details.append({"index_ijk": idx.tolist(), "world_center_ras_mm": ras.tolist(),
            "value": float(values[tuple(idx)]), "brain_neighbors_of_26": sum(bool(mask[p]) for p in coords),
            "positive_intensity_neighbors_of_26": int(np.count_nonzero(neighboring > 0)),
            "neighbor_intensity_min_median_max": [float(neighboring.min()), float(np.median(neighboring)), float(neighboring.max())]})
    separations = np.linalg.norm(world[:, None] - world[None, :], axis=2)
    separations[np.diag_indices_from(separations)] = np.inf
    sizes = sparse_components(rejected)
    assert sizes == [1]*8
    check_memory("fit_input_diagnosis")
    # Only inspect names/metadata; no candidate or other person's output is read.
    output = ROOT / "outputs/poc-ixi265-atropos-v1"
    assert not (output / "candidate.json").exists()
    assert not any(output.rglob("*.nii*"))
    after = check_pins(FIT_INPUTS)
    assert before == after
    return {"status": "failed_attempt_independently_explained", "inputs": before,
        "guard_predicates": predicates, "exact_failing_clause": "np.any(values[mask] <= 0)",
        "source_native_shape": list(values.shape), "source_voxels": values.size,
        "source_storage_dtype": str(source.get_data_dtype()), "guard_decoded_dtype": str(values.dtype),
        "mask_storage_dtype": str(mask_image.get_data_dtype()), "mask_unique_values": [0, 1],
        "estimated_brain_voxels": int(mask.sum()), "source_intensity_min_max": [float(values.min()), float(values.max())],
        "masked_intensity_min_max": [float(values[mask].min()), float(values[mask].max())],
        "masked_zero_voxels": len(rejected), "masked_negative_voxels": 0,
        "masked_zero_fraction": len(rejected)/int(mask.sum()), "masked_zero_voxel_volume_mm3": len(rejected)*voxel_volume,
        "rejected_index_bbox": bounds(rejected), "rejected_world_center_bbox_ras_mm": bounds(world),
        "rejected_26_components": len(sizes), "rejected_component_sizes": sizes,
        "minimum_pairwise_rejected_center_distance_mm": float(separations.min()), "rejected_cells": details,
        "distribution_interpretation": "Eight isolated zero-valued samples spread across the estimated mask; no negative or nonfinite source values. Their imaging/anatomical cause is undetermined. This does not establish the mask's anatomical validity.",
        "native_frame_checks": frames, "source_bytes_unchanged": True,
        "host_bundle_hashed_only_not_decoded": True,
        "denominator": {"started_attempts": 1, "N4_completions": 0, "Atropos_completions": 0,
            "accepted_tissue_results": 0, "retries": 0},
        "retained_attempt": {key: outcome[key] for key in ("worker_exit_code", "elapsed_seconds", "peak_supervisor_plus_worker_family_rss_bytes_sampled", "minimum_available_memory_bytes_sampled", "failure_was_resource_limit")},
        "candidate_or_posteriors_exist": False, "source_fix_clipping_or_normalization": False,
        "anatomical_acceptance": False, "host_or_catalog_promotion": False, "growth_use_accepted": False,
        "limitations": ["No real N4 or Atropos result exists; solver resource feasibility and tissue accuracy remain untested.",
            "No retry, new mask, intensity repair, or model import was performed by this audit.",
            "Any future intensity-class posterior would still not establish physical partial-volume fractions or ventricles."]}


def audit_vessels():
    import nibabel as nib
    before = check_pins(VESSEL_INPUTS)
    result = json.loads((ROOT / "outputs/poc-ixi265-vessel-agreement-v1/agreement.json").read_text())
    protocol = json.loads((ROOT / "artifacts/poc-ixi265-vessel-agreement-v1/protocol.json").read_text())
    assert result["protocol"] == protocol
    assert protocol["patient_group"] == "IXI:265" and protocol["role"] == "TRAIN" and protocol["training_admitted"] is False
    assert protocol["fixed_parameters"]["threshold"] == 0.5 and protocol["fixed_parameters"]["postprocessing"] is False
    assert protocol["model_runs"] == 0 and protocol["threshold_candidates"] == 1
    assert all(protocol[k] is False for k in ("independent_accuracy_claim", "clinical_accuracy_claim", "held_out_claim", "anatomical_registration_used", "source_reference_used_in_inference"))
    for name, pin in protocol["metadata_pins"].items():
        assert sha(ROOT / name) == pin
    images = {k: nib.load(ROOT / p) for k, (p, _) in VESSEL_INPUTS.items()}
    native = images["MRA"]
    assert native.shape == (512, 512, 100)
    frames = {key: frame(image, native) for key, image in images.items()}
    masks = {}
    for key in ("prediction", "reference", "coverage"):
        raw = np.asarray(images[key].dataobj)
        assert set(np.unique(raw)) == {0, 1}
        masks[key] = raw.astype(bool)
        del raw
    p, r, c = (masks[k] for k in ("prediction", "reference", "coverage"))
    assert c[1:-1, 1:-1, 1:-1].all() and c.sum() == np.prod(np.array(c.shape) - 2)
    score = np.asarray(images["score"].dataobj, dtype=np.float32)
    assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1
    assert np.array_equal(p, score > 0.5)
    score_range = [float(score.min()), float(score.max())]
    check_memory("vessel_threshold")
    del score
    gc.collect()
    table = contingency(p, r, c)
    voxel_volume = abs(float(np.linalg.det(native.affine[:3, :3])))
    counts = counts_from_table(table, voxel_volume)
    assert counts == {k: v for k, v in result["counts_and_agreement"].items() if k != "interpretation"}
    summaries = {}
    for name, mask in (("prediction", p), ("reference", r)):
        check_memory("vessel_" + name + "_before_sparse_graph")
        summaries[name] = sparse_summary(mask, c)
        assert summaries[name] == {k: v for k, v in result[name + "_components"].items() if k != "interpretation"}
        check_memory("vessel_" + name + "_after_sparse_graph")
        print(name + " independent sparse connectivity and boundary recount passed", flush=True)
    assert before == check_pins(VESSEL_INPUTS)
    return {"status": "pass_descriptive_set_agreement", "inputs": before,
        "source_MRA_hashed_and_header_checked_only": True, "native_frame_checks": frames,
        "binary_sets_exact": True, "score_min_max": score_range, "threshold": 0.5,
        "saved_prediction_equals_score_strictly_above_threshold": True,
        "coverage_exactly_one_native_border_cell_excluded": True,
        "joint_state_bit_order": "prediction + 2*reference + 4*coverage", "eight_state_counts": table.tolist(),
        "counts_and_agreement": counts, "sparse_graph_components_and_boundaries": summaries,
        "oracle": "Independent eight-bin slab contingency; Python sparse 26-neighbor DFS and six-neighbor membership boundary checks; no production diagnostic or scipy labeling/erosion invoked.",
        "all_reported_counts_reproduced": True, "source_bytes_unchanged": True,
        "patient_group": "IXI:265", "role": "TRAIN", "training_admitted": False,
        "model_weight_sha256_retained_from_frozen_inference": protocol["model"]["weights_sha256"],
        "new_weight_hash_sweep": False, "reference_used_only_for_this_diagnostic": True,
        "new_model_runs": 0, "new_thresholds_or_postprocessing": False,
        "limitations": ["Positive-set agreement is descriptive; no independent, held-out or clinical accuracy claim.",
            "110729 prediction voxels in UNLABELLED reference remain unknown, not false positives.",
            "Reference is model-assisted and manually refined; completeness and negative-safe background are unestablished.",
            "Provided-positive-reference coverage is not sensitivity to all vessels.",
            "IXI occurs in checkpoint training collection; exact IXI265 subject overlap is unresolved.",
            "No T1 registration, brain clipping, anatomical topology or surface-distance validation is implied."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-payload-audit", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    controls = generated_controls()
    if not args.released_payload_audit:
        print(json.dumps({"generated_controls": controls, "real_payloads_read": False}))
        return
    target = DEST / "independent-qa.json"
    assert not target.exists(), "Frozen audit already exists; choose a new audit version instead"
    metadata_before = {}
    for name, pin in METADATA.items():
        metadata_before[name] = stamp(ROOT / name)
        assert metadata_before[name]["sha256"] == pin, "Changed metadata: " + name
    check_memory("start")
    fit = audit_failed_fit()
    print("Failed fit explained independently: eight isolated masked zero samples; no real fitting completed", flush=True)
    gc.collect()
    vessel = audit_vessels()
    assert metadata_before == {name: stamp(ROOT / name) for name in METADATA}
    assert not any(name in sys.modules for name in ("ants", "torch", "totalsegmentator", "resectionlab.poc_models"))
    check_memory("finished")
    report = {"schema": "ixi265-quality-independent-audit-v1", "status": "pass_audit_with_failed_fit_retained",
        "created_utc": datetime.now(timezone.utc).isoformat(), "generated_controls": controls,
        "pinned_metadata_unchanged": metadata_before, "failed_tissue_fit": fit, "vessel_agreement": vessel,
        "auditor_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - started,
        "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_samples": SAMPLES, "heavy_model_or_solver_execution": False,
        "real_payload_scope": "Only fixed IXI265 T1/mask and native vessel set/score/coverage sources; host bundle byte hash only. No protected people or source edits.",
        "prior_audit_rewrites": False}
    report["audit_harness_corrections"] = [{
        "initial_check": "Applied the tissue runner's strict qform-versus-sform cell-corner tolerance to the separate vessel diagnostic.",
        "observed": "Native MRA qform differs from its selected sform by 0.000228071386490438 mm at the furthest cell corner; derived reference differs by 0.00022984447755395885 mm. These existing header values are retained.",
        "correction": "Vessel comparison uses the fixed selected native affines, whose corresponding corners agree within 0.0001 mm. Both header forms and their small discrepancy are reported; no header or source was rewritten. The stricter tissue source-frame guard remains checked."}]
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    assert "/Users/" not in content and "file://" not in content
    DEST.mkdir(parents=True, exist_ok=True)
    with target.open("x") as out:
        out.write(content)
    print(json.dumps({"status": report["status"], "artifact": str(target.relative_to(ROOT)), "sha256": sha(target),
        "elapsed_seconds": report["elapsed_seconds"], "peak_rss_bytes": report["own_process_peak_rss_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
