"""Opt-in independent SynthRAD B/C audit; never executes a model or solver.

Default/--prepare reads only small metadata and runs generated controls. Actual
source/bundle inspection additionally requires --released-pins after the parent
has explicitly released model-worker ownership. Existing audits are immutable.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/poc-synthrad-bc-source-v1"
ADMISSION = ROOT/"manifests/poc-synthrad-bc-source-admission-v1.json"
SOURCE = BASE/"source-result.json"
DECLARATION = BASE/"model-execution-declaration.json"
RUNNER = ROOT/"scripts/poc_synthrad_extension.py"
SUBJECTS = ("1BB033", "1BC077")
PINS = {str(ADMISSION.relative_to(ROOT)): "75dc377966ab15da03ebb454a79059414876676cf39cd790019798924f828dd8",
    str(SOURCE.relative_to(ROOT)): "4d2e85f0e2ba808f922f4d1f6f364e52e5f51d589e83746eb81793106b19e2ff",
    str(DECLARATION.relative_to(ROOT)): "7a541611e509fd7368abac9b153eaec477011ea2717a984129656d6fb0c7c252"}
FROZEN = {
    "artifacts/patient-corpus-first-tranche-v1/independent-qa.json": "5f112d987639408911b379bd72792cab2504f68599981ef833445e31bcf7e0bd",
    "artifacts/ixi265-virtual-pilot-v1/independent-qa.json": "61e449b05aaf1396e7383a23bb903238ab3e7e223899c23a0ed64e40c814eef0",
    "artifacts/ixi265-virtual-pilot-v1/gui-and-vessel-independent-qa.json": "0a1700dd15a784405d47b1d66aa7713eb6dab3cc92faf3179bc608a6da36d31b",
    "artifacts/patient-corpus-ten-person-v1/independent-qa.json": "2d28dcde7d6e7ff3a432ae6fe81e3cab7fa082e6ab5f05b5ecfbcd90c685951d",
}
WATCHED = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def watch(path, expected=None):
    path = Path(path)
    value = {"sha256": sha(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    assert expected is None or value["sha256"] == expected, "Pinned input differs: "+path.name
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "external-model/"+path.name
    WATCHED[key] = (path, value)
    return value


def small_json(path, expected=None):
    path = Path(path)
    assert path.suffix == ".json" and path.stat().st_size < 16*1024**2
    watch(path, expected)
    return json.loads(path.read_text())


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def metadata_and_controls():
    admission = small_json(ADMISSION, PINS[str(ADMISSION.relative_to(ROOT))])
    source = small_json(SOURCE, PINS[str(SOURCE.relative_to(ROOT))])
    declaration = small_json(DECLARATION, PINS[str(DECLARATION.relative_to(ROOT))])
    watch(RUNNER, declaration["script_sha256"])
    worker = ROOT/"src/resectionlab/poc_models.py"
    watch(worker, declaration["worker_sha256"])
    assert admission["subjects"] == list(SUBJECTS)
    assert admission["model_execution_authorized"] is admission["registration_execution_authorized"] is admission["training_admitted"] is False
    assert declaration["execution_authorized"] is True and declaration["training_admitted"] is False
    assert declaration["source_result_sha256"] == PINS[str(SOURCE.relative_to(ROOT))]
    assert declaration["source_admission_sha256"] == source["admission_sha256"] == PINS[str(ADMISSION.relative_to(ROOT))]
    assert [row["subject_id"] for row in source["denominator"]] == [row["subject_id"] for row in source["pairs"]] == list(SUBJECTS)
    assert all(row["role"] == "TRAIN" and row["training_admitted"] is False for row in source["denominator"])
    assert source["model_runs"] == source["registration_runs"] == 0 and not source["training_admitted"]
    cohort = small_json(ROOT/"manifests/synthrad-component-cohort-v1.json", admission["cohort_sha256"])
    ordered = []
    for center in "ABC":
        members = [row for row in cohort["members"] if row["anonymous_center"] == center]
        assert len(members) == 60
        members.sort(key=lambda row: (hashlib.sha256((cohort["split"]["namespace"]+row["subject_id"]).encode()).hexdigest(), row["subject_id"]))
        for rank, row in enumerate(members, 1):
            assert row["rank_within_center"] == rank
            assert row["role"] == ("TRAIN" if rank <= 42 else "SELECT" if rank <= 51 else "MEASUREMENT_EVAL")
        ordered.append(members[0]["subject_id"])
    assert ordered == ["1BA336", *SUBJECTS]
    requests = {r["case"].removeprefix("SynthRAD2023-"): r for r in declaration["requests"]}
    assert set(requests) == set(SUBJECTS)
    params = {"task": "total", "fast": True, "tta": False, "skull_label": 91, "brain_label": 90}
    model = literal_assignment(worker, "MODEL_CONFIG")["totalsegmentator-skull"]
    assert model["parameters"] == params
    for pair in source["pairs"]:
        sid = pair["subject_id"]
        assert pair["patient_group"] == "SynthRAD2023:"+sid and pair["role"] == "TRAIN" and not pair["training_admitted"]
        assert set(pair["images"]) == {"ct", "mr"}
        for modality, image in pair["images"].items():
            assert Path(image["path"]) == BASE/sid/(modality+".nii.gz")
            assert image["sha256"] == declaration["source_hashes"][sid][modality]
            assert image["gzip_crc_and_exact_payload_length"] == "passed" and not image["clinical_or_training_admission"]
        request = requests[sid]
        assert request["image"] == pair["images"]["ct"]["path"] and request["modality"] == "CT"
        assert request["model_id"] == "totalsegmentator-skull" and request["device"] == "mps" and request["parameters"] == params
        assert len(request["weights"]) == 1 and request["weights"][0]["sha256"] == admission["model_assets"]["weights_sha256"]
    weight = Path(declaration["requests"][0]["weights"][0]["path"])
    for name, expected in declaration["model_support"].items():
        small_json(weight.parent.parent/name, expected)
    budget = declaration["budget"]
    assert budget["workers"] == 1 and budget["subjects_sequential"] == list(SUBJECTS)
    assert budget["automatic_retries"] == 0 and budget["threads"] == 2 and budget["model_parameters_changed"] is False
    assert budget["wall_seconds_per_inference"] == 180 and budget["campaign_wall_seconds"] == 600
    assert budget["process_family_rss_bytes"] == 5*1024**3 and budget["minimum_system_available_bytes"] == 1024**3
    assert budget["new_outputs_bytes"] == 512*1024**2 and budget["free_disk_reserve_bytes"] == 50*1024**3

    # Compile only three pure/metadata guard functions; never import the runner,
    # canonical model module, TotalSegmentator, torch or any inference framework.
    names = {"require", "bound_source", "model_resource_failure"}
    tree = ast.parse(RUNNER.read_text())
    pure = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    def metadata_digest(path):
        assert Path(path) == SOURCE, "Guard attempted a real payload hash before rejection"
        return PINS[str(SOURCE.relative_to(ROOT))]
    env = {"Path": Path, "json": json, "SUBJECTS": SUBJECTS, "OUTPUT": BASE,
        "SOURCE_RESULT_SHA": PINS[str(SOURCE.relative_to(ROOT))], "SOURCE_HASHES": declaration["source_hashes"],
        "MODEL_BUDGET": budget, "digest": metadata_digest}
    exec(compile(pure, str(RUNNER), "exec"), env)
    controls = []
    for change in ("role", "subject", "training", "ct_hash", "ct_path", "nested_role", "mr_hash", "extra_generated"):
        changed = deepcopy(source["pairs"][0])
        if change == "role": changed["role"] = "SELECT"
        elif change == "subject": changed["subject_id"] = "1BA336"
        elif change == "training": changed["training_admitted"] = True
        elif change == "ct_hash": changed["images"]["ct"]["sha256"] = "0"*64
        elif change == "ct_path": changed["images"]["ct"]["path"] = "/protected/ct.nii.gz"
        elif change == "nested_role": changed["images"]["ct"]["role"] = "SELECT"
        elif change == "mr_hash": changed["images"]["mr"]["sha256"] = "0"*64
        else: changed["virtual_geometry"] = {"provenance": "generated"}
        try: env["bound_source"](changed)
        except ValueError: controls.append(change)
        else: raise AssertionError("Changed source was admitted")
    good = {"rss": budget["process_family_rss_bytes"], "available": budget["minimum_system_available_bytes"],
        "elapsed": budget["campaign_wall_seconds"], "free_disk": budget["free_disk_reserve_bytes"], "output_bytes": budget["new_outputs_bytes"]}
    assert env["model_resource_failure"](**good) is None
    for field, step in (("rss", 1), ("available", -1), ("elapsed", 1), ("free_disk", -1), ("output_bytes", 1)):
        assert env["model_resource_failure"](**{**good, field: good[field]+step})
    assert not any(name in sys.modules for name in ("torch", "totalsegmentator", "resectionlab.poc_models"))
    report = {"status": "pass", "exact_subjects": list(SUBJECTS), "rank_and_roles_independently_recomputed": True,
        "source_only_admission_unchanged": True, "separate_model_execution_declaration_bound": True,
        "source_rebinding_controls_rejected_before_payload": controls, "five_resource_limit_edges_pass": True,
        "budgets": budget, "saved_model_config_matches_worker_literal": True, "small_model_support_files_rehashed": declaration["model_support"],
        "training_admitted": False, "registration_runs": 0, "preparation_real_payload_reads": 0,
        "limitations": ["Clinical HU calibration and source anatomical coverage remain unreviewed.",
            "Publisher-preprocessed acquired CT/MR are preserved; this does not establish raw scanner-native fidelity."]}
    return admission, source, declaration, model, report


def assert_native_transfer(values, affine, saved_values, saved_affine):
    """RAS world coordinates do not imply RAS-ordered array axes.

The workbench preserves acquired voxel ordering and the full source affine.
Any implicit orientation flip, even with a compensated affine, fails this check.
"""
    import numpy as np
    assert np.array_equal(values, saved_values)
    assert np.array_equal(affine, saved_affine)


def generated_transfer_control():
    import numpy as np
    shape = (3, 4, 5)
    values = np.arange(np.prod(shape)).reshape(shape)
    affine = np.diag([-2., -3., 4., 1.]);affine[:3, 3] = [10., 20., -5.]
    assert_native_transfer(values, affine, values.copy(), affine.copy())
    for index in np.ndindex(shape):
        world = affine@np.r_[index, 1]
        assert np.array_equal(world, [10-2*index[0], 20-3*index[1], -5+4*index[2], 1])
    altered_affine = affine.copy();altered_affine[0, 3] += 1
    for wrong_values, wrong_affine in ((values[::-1], affine), (values, altered_affine), (values+1, affine)):
        try: assert_native_transfer(values, affine, wrong_values, wrong_affine)
        except AssertionError: pass
        else: raise AssertionError("Generated transfer mutation was not detected")
    return {"status": "pass", "generated_voxels": 60, "every_value_and_world_center_preserved": True,
        "array_flip_affine_shift_and_value_change_detected": True}


def actual_pair(sid, source, declaration, model, pin):
    import nibabel as nib
    import numpy as np
    from resectionlab.core import array_digest
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.workspace_imaging import source_registration
    pair = next(p for p in source["pairs"] if p["subject_id"] == sid)
    bundle = BASE/sid/("SynthRAD2023-"+sid+".ressectionlab")
    watch(bundle, pin["bundle_sha256"])
    inference_path = ROOT/pin["inference_path"]
    assert inference_path.is_relative_to(ROOT/"outputs/poc-models"/("SynthRAD2023-"+sid))
    inference = small_json(inference_path, pin["inference_sha256"])
    assert inference["status"] == "succeeded" and inference["failure"] is None and inference["exit_code"] == 0
    request = next(r for r in declaration["requests"] if r["case"] == "SynthRAD2023-"+sid)
    assert inference["weights"] == request["weights"] and inference["model"] == model
    identity = {"image_sha256": pair["images"]["ct"]["sha256"], "weights": [request["weights"][0]["sha256"]],
        "model_id": request["model_id"], "parameters": request["parameters"], "device": request["device"], "runner_sha256": declaration["worker_sha256"]}
    assert inference["identity"] == identity
    assert inference["cache_key"] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    assert inference["input"]["modality"] == "CT" and inference["input"]["sha256"] == pair["images"]["ct"]["sha256"]
    assert inference["probability"] is None and inference["clinical_validation"] == "none"
    assert inference["elapsed_seconds"] <= declaration["budget"]["wall_seconds_per_inference"]
    assert inference["peak_process_family_rss_bytes_sampled"] <= declaration["budget"]["process_family_rss_bytes"]
    case, artifacts, workspace = load_workspace(bundle)
    assert case.case_id == "SynthRAD2023-"+sid and case.frame == "RAS+"
    assert set(case.compartments) == set(case.source_compartments) == {"brain", "skull"} and case.brain_mask is None
    assert not case.critical_evidence and not case.prior_proposals and "patient_rehearsal" not in artifacts
    assert len(workspace["sources"]) == 1
    display = workspace["sources"][0]
    assert display.modality == "T1" and display.annotation_kind == "none"
    auxiliary = display.case
    assert not auxiliary.compartments and not auxiliary.source_compartments and auxiliary.brain_mask is None
    images = {}
    originals_by_hash = {item.sha256: item for item in workspace["originals"]}
    assert len(workspace["originals"]) == len(originals_by_hash) == 5
    for modality, actual in (("ct", case), ("mr", auxiliary)):
        record = pair["images"][modality]; path = BASE/sid/(modality+".nii.gz")
        watch(path, record["sha256"])
        original = nib.load(path); native = original.get_fdata(dtype=np.float32)
        assert np.array_equal(original.affine, record["grid"]["affine_ras_mm"])
        assert list(native.shape) == record["grid"]["shape"] and np.isfinite(native).all()
        assert_native_transfer(native, original.affine, actual.mri, actual.affine)
        assert actual.metadata["split"]["role"] == "TRAIN" and actual.metadata["split"]["training_admitted"] is False
        assert actual.metadata["training_admitted"] is False and actual.metadata["is_synthetic"] is False
        assert set(actual.metadata["missing_anatomy"]) == {"tumor", "vessels", "ventricles"}
        ref = next(r for r in actual.source_refs if r.source_id == "structural")
        assert ref.sha256 == record["sha256"] and ref.provenance == "observed" and ref.license == "CC-BY-NC-4.0"
        assert originals_by_hash[record["sha256"]].data == path.read_bytes()
        images[modality] = {"source_sha256": record["sha256"], "native_shape": list(native.shape), "native_affine": original.affine.tolist(),
            "saved_affine": actual.affine.tolist(), "saved_array_hash": array_digest(actual.mri), "native_order_and_affine_exact_no_reorientation": True,
            "original_bytes_exact": True, "scalar_minimum": float(native.min()), "scalar_maximum": float(native.max())}
    assert {row["name"] for row in inference["outputs"]} == {"brain_mask.nii.gz", "skull_mask.nii.gz"}
    masks = {}
    for row in inference["outputs"]:
        path = inference_path.parent/row["name"]; assert Path(row["path"]) == path
        watch(path, row["sha256"])
        image = nib.load(path); mask = np.asarray(image.dataobj)
        assert set(np.unique(mask)) <= {0, 1} and mask.any()
        assert np.array_equal(image.affine, inference["input"]["affine_ras_mm"])
        name = row["name"].removesuffix("_mask.nii.gz")
        assert_native_transfer(mask, image.affine, case.compartments[name], case.affine)
        assert np.array_equal(mask, case.source_compartments[name])
        assert int(mask.sum()) == row["foreground_voxels"]
        layer = case.metadata["anatomy_layers"][name]
        assert layer["provenance"] == "estimated" and layer["kind"] == name and layer["source_role"] == "cached_model_inference"
        assert layer["source_sha256"] == row["sha256"] and layer["input_sha256"] == pair["images"]["ct"]["sha256"]
        assert layer["inference_manifest_sha256"] == pin["inference_sha256"] and layer["cache_key"] == inference["cache_key"]
        assert layer["model"]["weights_sha256"] == request["weights"][0]["sha256"]
        assert "pending" in layer["review_status"] and "unknown" in layer["coverage"]
        ref = next(r for r in case.source_refs if r.source_id == "model_"+name)
        assert ref.provenance == "estimated" and ref.sha256 == row["sha256"]
        assert originals_by_hash[row["sha256"]].data == path.read_bytes()
        masks[name] = {"native_saved_mask_sha256": row["sha256"], "voxels": int(mask.sum()), "volume_mm3": float(mask.sum()*abs(np.linalg.det(image.affine[:3, :3]))),
            "saved_array_hash": array_digest(case.compartments[name]), "native_to_workspace_arrays_and_affines_exact": True,
            "positive_boundary_voxels_face_sum": sum(int(np.count_nonzero(np.take(mask, index, axis=axis))) for axis in range(3) for index in (0, -1)),
            "provenance": "estimated", "anatomical_accuracy_validated": False}
    assert originals_by_hash[pin["inference_sha256"]].data == inference_path.read_bytes()
    registration = source_registration(case, auxiliary)[1]
    assert registration["overlayPermitted"] is False
    return {"status": "pass", "subject_id": sid, "role": "TRAIN", "training_admitted": False, "bundle_sha256": pin["bundle_sha256"],
        "case_hash": case.semantic_hash, "source_arrays": images, "mask_transfer": masks, "all_five_embedded_originals_exact": True,
        "MR_native_source_separate": True, "fusion_permitted": False, "registration_status": registration["status"],
        "model_cache_key": inference["cache_key"], "model_seconds": inference["elapsed_seconds"],
        "sampled_process_family_rss_bytes": inference["peak_process_family_rss_bytes_sampled"],
        "missing_anatomy": ["tumor", "vessels", "ventricles"], "clinical_validation": False,
        "limitation": "Saved native binary model masks match the workspace exactly; the original full multiclass prediction was not retained for a separate label-extraction replay. No inference or accuracy evaluation was performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--released-pins", type=Path)
    args = parser.parse_args()
    for name, expected in FROZEN.items(): small_json(ROOT/name, expected)
    admission, source, declaration, model, preparation = metadata_and_controls()
    transfer = generated_transfer_control()
    if args.prepare or args.released_pins is None:
        destination = ROOT/"outputs/poc-first-qa/synthrad-bc-preparation-controls.json"
        destination.write_text(json.dumps({"status": "preparation_pass_payload_release_pending", "metadata": preparation, "generated_transfer": transfer}, indent=2)+"\n")
        print(json.dumps({"status": "preparation_pass_payload_release_pending", "artifact": str(destination.relative_to(ROOT))}))
        return
    pins = small_json(args.released_pins.resolve())
    assert pins["model_worker_released_by_parent"] is True and set(pins["subjects"]) == set(SUBJECTS)
    receipt = small_json(ROOT/"artifacts/poc-synthrad-bc-preparation-v1/receipt.json", "50ba6725e4c118f8f13f7acfcc27359be55028f13e948cccb5bb30b9333ebc35")
    small_json(ROOT/"outputs/patient-corpus-ixi265-v1/catalog.json", "2a7ba1c26f101261e3a5fecf84ae998b637a70c5a7f66d43d224153c738ee673")
    campaign = small_json(BASE/"model-campaign.json", pins["campaign_sha256"])
    assert campaign["release_sha256"] == PINS[str(DECLARATION.relative_to(ROOT))] and not campaign["training_admitted"] and campaign["registration_runs"] == 0
    assert [row["subject_id"] for row in campaign["subjects"]] == list(SUBJECTS)
    assert campaign["elapsed_seconds"] <= declaration["budget"]["campaign_wall_seconds"]
    assert receipt["model_runtime"] == campaign and receipt["source_runtime"] == source["runtime"]
    assert receipt["model_runs"] == 2 and receipt["registration_runs"] == receipt["data_or_model_downloads"] == 0
    assert receipt["training_admitted"] is receipt["ten_person_catalog_modified"] is False
    measured_bytes = sum(p.stat().st_size for root in [BASE, *[ROOT/"outputs/poc-models"/("SynthRAD2023-"+sid) for sid in SUBJECTS]] for p in root.rglob("*") if p.is_file())
    assert measured_bytes == receipt["measured_output_bytes_before_final_receipt"] < declaration["budget"]["new_outputs_bytes"]
    initial = small_json(BASE/"model-denominator-initial.json")
    assert [row["subject_id"] for row in initial["subjects"]] == list(SUBJECTS)
    assert all(row["status"] == "not_attempted" and row["role"] == "TRAIN" and not row["training_admitted"] for row in initial["subjects"])
    weight = declaration["requests"][0]["weights"][0]
    watch(Path(weight["path"]), weight["sha256"])
    assert Path(weight["path"]).stat().st_size == weight["bytes"]
    cases = []
    for sid in SUBJECTS:
        row = next(r for r in campaign["subjects"] if r["subject_id"] == sid)
        assert row["status"] == "model_saved_bundle_unreviewed" and row["guard_failure"] is None
        assert row["bundle"]["sha256"] == pins["subjects"][sid]["bundle_sha256"]
        assert row["inference_sha256"] == pins["subjects"][sid]["inference_sha256"]
        assert row["peak_process_family_rss_bytes_sampled"] <= declaration["budget"]["process_family_rss_bytes"]
        assert row["minimum_system_available_bytes_sampled"] >= declaration["budget"]["minimum_system_available_bytes"]
        cases.append(actual_pair(sid, source, declaration, model, pins["subjects"][sid]))
        print(sid+" independent source/mask/bundle pass", flush=True)
    for path, original in list(WATCHED.values()):
        assert sha(path) == original["sha256"] and path.stat().st_mtime_ns == original["mtime_ns"]
    assert not any(name in sys.modules for name in ("torch", "totalsegmentator", "resectionlab.poc_models"))
    report = {"schema": "independent-synthrad-bc-integration-audit-v1", "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_and_controls": preparation, "generated_transfer_control": transfer, "cases": cases,
        "campaign_sha256": pins["campaign_sha256"], "campaign_seconds": campaign["elapsed_seconds"],
        "measured_source_and_model_output_bytes": measured_bytes, "fixed_source_people_materialized": 2,
        "ten_person_catalog_unchanged": True,
        "model_weight_rehashed_once": True, "inputs_unchanged": True, "prior_audits_unchanged": True,
        "model_or_solver_executed": False, "model_frameworks_imported": False,
        "auditor_adjustments": ["Initial oracle incorrectly assumed RAS world coordinates meant canonical RAS array-axis ordering. Corrected to require exact native arrays and affines, matching the loader's no-reorientation contract. No product change was needed."],
        "input_stamps": {key: value for key, (_, value) in WATCHED.items()}, "auditor_sha256": sha(Path(__file__)),
        "limitations": ["Sampled RSS is not an exact peak and must not be added to shared Metal allocations.",
            "Dataset overlap, HU calibration, anatomical completeness and model accuracy are unvalidated.",
            "Model estimates are not source annotations. Matching CT/MR coded grids do not grant fusion."]}
    content = json.dumps(report, indent=2)+"\n";assert "/Users/" not in content
    destination = ROOT/"artifacts/poc-synthrad-bc-integration-v1/independent-qa.json"
    destination.parent.mkdir(parents=True, exist_ok=True);destination.write_text(content)
    print(json.dumps({"status": "pass", "artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
