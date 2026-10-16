"""Prepare-only independent tissue display bundle/save audit helpers.

Default runs tiny generated fixtures only. Actual package and GUI-save decoding
must be added behind the parent's separately sealed path/hash release. No ANTs,
models, imaging payloads, or existing bundles are opened by default.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import resource
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab"
HOST_SHA = "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"
RESULT_QA = ROOT/"artifacts/ixi265-typed-mask-independent-v3/result-qa.json"
RESULT_QA_SHA = "0b2918d3d6d1c17e772cad4649689f67ff9546083eced8bfe7d5796db0ed6509"
AUDIT_RSS_CAP = 1024**3
AVAILABLE_MEMORY_FLOOR = 1024**3
PREP = ROOT/"artifacts/ixi265-tissue-display-preparation-v1"
PREP_SHA = "820d552f82e07fe2ab2d0dcac9df0c4c464c7075b3b4b756f795a171a51a0ead"
PACKAGE = ROOT/"outputs/poc-ixi265-tissue-display-v1/IXI265-Guys-0845-tissue-estimates.ressectionlab"
FIT = ROOT/"outputs/poc-ixi265-atropos-typed-mask-v3"
LAYERS = ("atropos_csf_like_estimate", "atropos_gray_matter_like_estimate",
          "atropos_white_matter_like_estimate", "unknown_missing_intensity")
TISSUE_FILES = ("csf_like_intensity_estimated", "gray_matter_like_intensity_estimated", "white_matter_like_intensity_estimated")
MEMORY = []
UNKNOWN = ((60,125,75), (67,135,49), (69,131,40), (89,120,83),
           (107,123,65), (127,219,68), (128,115,125), (185,127,119))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory_guard(phase):
    import psutil
    own = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    assert own < AUDIT_RSS_CAP and available >= AVAILABLE_MEMORY_FLOOR
    row = {"phase": phase, "rss_bytes": own, "available_bytes": available}
    MEMORY.append(row)
    return row


def stamp(path):
    stat = Path(path).stat()
    return {"sha256": sha(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def exact_members(left, right, left_name, right_name=None):
    """Bounded streaming comparison; no full original-file copy is allocated."""
    hashed = hashlib.sha256(); size = 0
    with left.open(left_name) as a, right.open(right_name or left_name) as b:
        while True:
            first, second = a.read(1024**2), b.read(1024**2)
            assert first == second, "Archived source bytes changed"
            if not first: break
            hashed.update(first); size += len(first)
    return {"sha256": hashed.hexdigest(), "bytes": size}


def nested_members(left, right, left_name, right_name=None):
    """Compare inner member bytes, independent of ZIP timestamps/compression."""
    with ZipFile(BytesIO(left.read(left_name))) as a, ZipFile(BytesIO(right.read(right_name or left_name))) as b:
        assert len(a.namelist()) == len(set(a.namelist()))
        assert set(a.namelist()) == set(b.namelist()) and len(b.namelist()) == len(set(b.namelist()))
        return {name: exact_members(a, b, name) for name in a.namelist()}


def preservation(reference, candidate, *, original_count=16, display_count=2, added_originals=0, added_displays=1):
    with ZipFile(reference) as a, ZipFile(candidate) as b:
        am, bm = (json.loads(z.read("manifest.json")) for z in (a, b))
        aw, bw = (json.loads(z.read("workspace.json")) for z in (a, b))
        assert {k: v for k, v in am.items() if k != "workspace_sha256"} == {
            k: v for k, v in bm.items() if k != "workspace_sha256"}
        assert aw["referenceCaseHash"] == bw["referenceCaseHash"]
        assert aw["primaryPlanningHash"] == bw["primaryPlanningHash"]
        primary_arrays = nested_members(a, b, "arrays.npz")
        assert len(aw["originalFiles"]) == original_count
        assert len(bw["originalFiles"]) == original_count+added_originals
        assert aw["originalFiles"] == bw["originalFiles"][:original_count]
        originals = []
        for row in aw["originalFiles"]:
            checked = exact_members(a, b, row["member"])
            assert checked == {"sha256": row["sha256"], "bytes": row["bytes"]}
            originals.append({"name": row["name"], "role": row["role"], **checked})
        assert len(aw["images"]) == display_count and len(bw["images"]) == display_count+added_displays
        sources = []
        for old, new in zip(aw["images"], bw["images"][:display_count], strict=True):
            assert old["source"] == new["source"]
            members = nested_members(a, b, old["member"], new["member"])
            assert digest(a.read(old["member"])) == old["sha256"]
            assert digest(b.read(new["member"])) == new["sha256"]
            sources.append({"source_identity": new["source"], "all_member_bytes_equal": True,
                            "member_sha256": {k: v["sha256"] for k, v in members.items()}})
        return {"primary_manifest_except_workspace_binding_exact": True,
                "primary_arrays": primary_arrays, "originals": originals,
                "preserved_reference_sources": sources,
                "source_and_planning_identity_unchanged": True}


def check_tissue_masks(labels, domain, unknown, brain, masks, unknown_mask):
    assert labels.shape == domain.shape == unknown.shape == brain.shape == unknown_mask.shape
    assert domain.dtype == unknown.dtype == brain.dtype == unknown_mask.dtype == np.bool_
    assert np.array_equal(domain | unknown, brain) and not np.any(domain & unknown)
    assert np.array_equal(labels > 0, domain) and np.isin(labels, [0, 1, 2, 3]).all()
    assert np.array_equal(unknown_mask, unknown)
    assert set(masks) == {1, 2, 3}
    for code, mask in masks.items():
        assert mask.dtype == np.bool_ and np.array_equal(mask, labels == code)
    assert np.array_equal(sum(mask.astype(np.uint8) for mask in masks.values()), domain.astype(np.uint8))


def zip_bytes(entries):
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, value in entries.items(): archive.writestr(name, value)
    return buffer.getvalue()


def generated_controls():
    labels = np.arange(24).reshape(3, 4, 2) % 4
    domain = labels > 0; unknown = np.zeros_like(domain); unknown[0, 0, 0] = True
    brain = domain | unknown; masks = {code: labels == code for code in (1, 2, 3)}
    check_tissue_masks(labels, domain, unknown, brain, masks, unknown)
    failures = 0
    for kind in ("unknown_lost", "wrong_class", "domain_overlap", "background_as_tissue"):
        m = {k: v.copy() for k, v in masks.items()}; u = unknown.copy(); d = domain.copy()
        if kind == "unknown_lost": u[:] = False
        elif kind == "wrong_class": m[1], m[2] = m[2], m[1]
        elif kind == "domain_overlap": d |= unknown
        else: m[1][0, 0, 0] = True
        try: check_tissue_masks(labels, d, unknown, brain, m, u)
        except AssertionError: failures += 1
    assert failures == 4
    inner = zip_bytes({"mri.npy": b"tiny generated image bytes", "affine.npy": b"tiny generated frame"})
    display = zip_bytes({"manifest.json": b'{"generated":true}', "arrays.npz": inner})
    entries = {"arrays.npz": inner, "workspace/series-0.ressectionlab": display,
               "workspace/original-0.bin": b"source A", "workspace/original-1.bin": b"source B"}
    workspace = {"referenceCaseHash": "generated-case", "primaryPlanningHash": "generated-plan",
        "originalFiles": [{"member": f"workspace/original-{i}.bin", "name": f"source-{i}",
            "role": "imaging", "sha256": digest(entries[f"workspace/original-{i}.bin"]),
            "bytes": len(entries[f"workspace/original-{i}.bin"])} for i in range(2)],
        "images": [{"member": "workspace/series-0.ressectionlab", "source": {"annotationKind": "reference"},
                    "sha256": digest(display)}]}
    entries["workspace.json"] = json.dumps(workspace).encode()
    entries["manifest.json"] = json.dumps({"case_semantic_hash": "generated-case", "workspace_sha256": "one"}).encode()
    reference = zip_bytes(entries)
    candidate = dict(entries)
    candidate["manifest.json"] = json.dumps({"case_semantic_hash": "generated-case", "workspace_sha256": "two"}).encode()
    preservation(BytesIO(reference), BytesIO(zip_bytes(candidate)), original_count=2, display_count=1, added_displays=0)
    corrupted = 0
    for member in ("workspace/original-0.bin", "arrays.npz", "workspace/series-0.ressectionlab"):
        bad = dict(candidate)
        bad[member] = b"changed" if member.endswith(".bin") else zip_bytes({"wrong": b"changed"})
        try: preservation(BytesIO(reference), BytesIO(zip_bytes(bad)), original_count=2, display_count=1, added_displays=0)
        except AssertionError: corrupted += 1
    assert corrupted == 3
    return {"tiny_generated_positive_cases_passed": 2, "mask_partition_corruptions_refused": failures,
        "original_primary_or_reference_byte_corruptions_refused": corrupted,
        "patient_bundle_reads": False, "native_or_model_imports": False,
        "actual_package_or_GUI_save_audit_released": False}


def admitted_records(pins):
    assert pins["packager_exited"] is True and pins["payload_audit_released"] is True
    assert pins["auditor_sha256"] == sha(__file__)
    assert sha(PREP/"preparation.json") == PREP_SHA and sha(RESULT_QA) == RESULT_QA_SHA
    prep = json.loads((PREP/"preparation.json").read_text())
    inputs = json.loads((PREP/"expected-inputs.json").read_text())
    frozen = {row["path"]: row["sha256"] for row in prep["frozen_files"]}
    frozen.update({row["path"]: row["sha256"] for row in inputs["files"]})
    frozen[str((PREP/"preparation.json").relative_to(ROOT))] = PREP_SHA
    for relative in frozen:
        path = ROOT/relative
        assert path.is_relative_to(ROOT) and path.resolve() == path
    before = {relative: stamp(ROOT/relative) for relative in frozen}
    assert {relative: row["sha256"] for relative, row in before.items()} == frozen
    assert sha(HOST) == HOST_SHA
    assert PACKAGE.resolve() == PACKAGE and sha(PACKAGE) == pins["package_sha256"]
    sealed = {}
    for name in ("package.json", "attempt.json", "declaration.json", "worker.log"):
        path = PACKAGE.parent/name
        assert sha(path) == pins["metadata_files"][name]
        sealed[name] = stamp(path)
    receipt = json.loads((PACKAGE.parent/"package.json").read_text())
    attempt = json.loads((PACKAGE.parent/"attempt.json").read_text())
    declaration = json.loads((PACKAGE.parent/"declaration.json").read_text())
    assert receipt["status"] == attempt["status"] == "display_only_candidate_packaged_unaccepted"
    assert receipt["bundle_sha256"] == pins["package_sha256"] and receipt["bundle_bytes"] == PACKAGE.stat().st_size
    assert receipt["preparation_sha256"] == declaration["preparation_sha256"] == PREP_SHA
    assert declaration["released_single_package"] is True and attempt["exit_code"] == 0
    assert declaration["expected_inputs"] == inputs and attempt["source_and_result_inputs_unchanged"] is True
    marker = ROOT/"outputs/poc-ixi265-tissue-display-v1-attempt.json"
    assert json.loads(marker.read_text()) == declaration
    before[str(marker.relative_to(ROOT))] = stamp(marker)
    limits = prep["resources"]
    assert attempt["resources"] == limits
    assert attempt["peak_supervisor_and_worker_rss_bytes_sampled"] <= limits["rss_limit_bytes"]
    assert attempt["minimum_available_memory_bytes_sampled"] >= limits["available_memory_floor_bytes"]
    assert attempt["minimum_free_disk_bytes_sampled"] >= limits["disk_floor_bytes"]
    assert attempt["elapsed_seconds"] <= limits["seconds"]
    assert attempt["output_bytes"] <= limits["output_temporary_limit_bytes"]
    assert receipt["source_host_bytes_unchanged"] is True and receipt["fusion_permitted"] is False
    for key in ("anatomy_accepted", "host_accepted", "growth_use_accepted"):
        assert receipt[key] is attempt[key] is False
    assert receipt["training_admitted"] is False and receipt["catalog_modified"] is False
    assert receipt["fitting_or_registration_run"] is False
    return inputs, before, sealed, receipt, attempt


def compare_original_to_file(archive, member, path):
    checked = hashlib.sha256(); size = 0
    with archive.open(member) as embedded, Path(path).open("rb") as original:
        while True:
            x, y = embedded.read(1024**2), original.read(1024**2)
            assert x == y
            if not x: break
            checked.update(x); size += len(x)
    return {"sha256": checked.hexdigest(), "bytes": size}


def audit_package(candidate, inputs, receipt, pins):
    # Existing reference payloads are compared as streamed bytes, never loaded
    # into an image/model API. Only primary and the new tissue display decode.
    preserved = preservation(HOST, candidate, added_originals=21, added_displays=1)
    expected_host = json.loads((PREP/"expected-host.json").read_text())
    expected = json.loads((PREP/"expected-outputs.json").read_text())
    result_qa = json.loads(RESULT_QA.read_text())
    appended = []
    memory_guard("before_bundle_arrays")
    with ZipFile(candidate) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        workspace = json.loads(archive.read("workspace.json"))
        assert manifest["case_semantic_hash"] == expected_host["primary_case_hash"] == receipt["primary_case_hash"]
        assert workspace["primaryPlanningHash"] == expected_host["primary_planning_hash"] == receipt["primary_planning_hash"]
        assert workspace["originalFiles"][:16] == expected_host["originals"]
        assert [row["source"] for row in workspace["images"][:2]] == expected_host["reference_sources"]
        assert len(workspace["originalFiles"]) == 37 and len(workspace["images"]) == 3
        for row, original in zip(workspace["originalFiles"][16:], inputs["append_originals"], strict=True):
            path = ROOT/original["path"]
            assert row["name"] == path.name and row["sourceUri"] == path.as_uri() and row["role"] == original["role"]
            checked = compare_original_to_file(archive, row["member"], path)
            assert checked == {"sha256": row["sha256"], "bytes": row["bytes"]} == {
                "sha256": original["sha256"], "bytes": original["bytes"]}
            appended.append({"name": row["name"], "role": row["role"], **checked, "semantics": original["semantics"]})
        assert sum(row["bytes"] for row in workspace["originalFiles"]) == 69176286
        added = workspace["images"][2]
        assert added["source"] == receipt["source_manifest"]
        assert added["source"]["annotationKind"] == "estimated" and added["source"]["modality"] == "T1"
        assert added["source"]["scope"] == "display-only-native-grid"
        assert digest(archive.read(added["member"])) == added["sha256"]
        with np.load(BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            primary_mri = arrays["mri"]; affine = arrays["affine"]; brain = arrays["brain_mask"]
        assert primary_mri.shape == brain.shape == (256, 256, 150)
        assert brain.dtype == np.bool_ and int(brain.sum()) == 1451509
        assert manifest["array_index"]["compartments"] == manifest["array_index"]["source_compartments"] == {}
        with ZipFile(BytesIO(archive.read(added["member"]))) as nested:
            display = json.loads(nested.read("manifest.json"))
            assert display["case_semantic_hash"] == added["source"]["sourceCaseHash"]
            assert display["frame"] == manifest["frame"] == "RAS+"
            assert display["context"] is None
            for key in ("structural_evidence", "critical_evidence", "prior_proposals"): assert not display.get(key)
            assert display.get("functional_evidence") is None
            index = display["array_index"]
            assert set(index["compartments"]) == set(index["source_compartments"]) == set(LAYERS)
            with np.load(BytesIO(nested.read("arrays.npz")), allow_pickle=False) as arrays:
                assert set(arrays.files) == {"mri", "affine", *index["compartments"].values(), *index["source_compartments"].values()}
                assert np.array_equal(arrays["mri"], primary_mri) and np.array_equal(arrays["affine"], affine)
                masks = {name: arrays[key] for name, key in index["compartments"].items()}
                for name, key in index["source_compartments"].items(): assert np.array_equal(masks[name], arrays[key])
        memory_guard("primary_and_added_display_loaded")
    metadata = display["metadata"]
    assert metadata["patient_group"] == "IXI:265" and metadata["split"]["role"] == "TRAIN"
    assert metadata["split"]["training_admitted"] is False
    assert all(metadata[key] is False for key in ("training_admitted", "anatomy_accepted", "host_accepted", "growth_use_accepted"))
    assert metadata["annotation_use"] == "display_only" and metadata["annotation_policy_use"] == "excluded"
    assert metadata["workspace_registration"] == {"kind": "estimated-unreviewed", "overlayPermitted": False}
    assert metadata["display_label"] == expected["new_display_source_label"]
    assert metadata["anatomical_review"] == "pending"
    refs = display["source_refs"]
    assert len(refs) == 2
    acquired = next(row for row in refs if row["source_id"] == "structural")
    assert acquired == next(row for row in manifest["source_refs"] if row["source_id"] == "structural")
    annotation = next(row for row in refs if row["source_id"] == "supplied_target_annotation")
    assert annotation["provenance"] == "estimated" and annotation["sha256"] == sha(FIT/"class_labels_raw.nii.gz")
    provenance = metadata["tissue_estimate_provenance"]
    for key, value in receipt["provenance"].items(): assert provenance[key] == value
    assert provenance["posterior_semantics"] == "membership_probabilities_not_PVE"
    assert provenance["native_indices_unchanged"] is True and provenance["resampling_or_thresholding"] is False
    roundoff = provenance["raw_native_posterior_header_roundoff_record"]
    assert roundoff["independent_result_QA_sha256"] == RESULT_QA_SHA
    assert roundoff["per_class_native_checks"] == result_qa["result"]["raw_native_temporary_posterior_comparisons"]
    assert roundoff["raw_probability_values_unchanged"] is True and roundoff["bitwise_affine_equality_claimed"] is False
    for i, name in enumerate(LAYERS):
        layer = metadata["anatomy_layers"][name]
        assert layer["name"] == name and layer["label"] == expected["new_display_mask_labels"][i]
        assert layer["policy_use"] == "excluded" and layer["independent_numerical_QA_sha256"] == RESULT_QA_SHA
        assert layer["provenance"] == ("observed" if i == 3 else "estimated")
        assert layer["source_class_id"] == (None if i == 3 else i+1)
        assert layer["kind"] == ("unknown" if i == 3 else "tissue_estimate")
        expected_hash = provenance["unknown_sha256"] if i == 3 else provenance["mask_sha256"][i]
        assert layer["source_sha256"] == expected_hash
    # Reuse only our independent saved-result frame checker, never the packager's validator.
    import first_ixi_typed_result_qa as result_auditor
    import nibabel as nib
    source = nib.load(ROOT/"outputs/poc-ixi265/originals/IXI265-T1.nii.gz")
    assert np.array_equal(np.asarray(source.dataobj, dtype=np.float32), primary_mri)
    assert np.max(abs(affine-source.affine)) <= 1e-4
    frames = {}
    def native(name):
        image = nib.load(FIT/(name+".nii.gz"))
        frames[name] = result_auditor.check_frame(image, source)
        return np.asarray(image.dataobj)
    labels = native("class_labels_raw"); domain = native("fit_domain_native").astype(bool)
    unknown = native("excluded_unknown_native").astype(bool)
    check_tissue_masks(labels, domain, unknown, brain, {i+1: masks[name] for i, name in enumerate(LAYERS[:3])}, masks[LAYERS[3]])
    assert np.array_equal(domain, brain & (primary_mri > 0))
    assert set(map(tuple, np.argwhere(unknown))) == set(UNKNOWN)
    assert int(domain.sum()) == 1451501 and int(unknown.sum()) == 8
    for name, file_name in zip(LAYERS[:3], TISSUE_FILES, strict=True):
        raw = native(file_name); assert np.isin(raw, [0, 1]).all() and np.array_equal(masks[name], raw)
        del raw
    counts = [int(masks[name].sum()) for name in LAYERS]
    assert counts == [263320, 607861, 580320, 8]
    series_id = added["source"]["seriesId"]
    state = workspace["imagingState"]
    expected_series = pins.get("expected_selected_series_id", series_id)
    assert state["selectedSeriesId"] == expected_series
    if "expected_viewer" in pins:
        actual_viewer = state["states"][expected_series]["viewer"]
        for key, value in pins["expected_viewer"].items(): assert actual_viewer[key] == value
    for series_state in state["states"].values():
        fusion = series_state.get("viewer", {}).get("fusion")
        assert fusion is None or fusion["seriesId"] is None
    memory_guard("bundle_check_complete")
    return {"preserved": preserved, "new_originals": appended, "all_37_original_bytes_unchanged": True,
        "total_original_bytes": 69176286, "added_source": added["source"], "display_label": metadata["display_label"],
        "mri_exact_acquired_T1": True, "stored_display_affine_exact_primary_array": True,
        "native_output_frames": frames, "all_working_and_source_masks_exact": True,
        "layer_voxel_counts": dict(zip(LAYERS, counts)), "unknown_indices_native": [list(p) for p in UNKNOWN],
        "known_and_unknown_exact_original_brain_partition": True,
        "raw_posteriors_preserved_as_archived_membership_probabilities": True,
        "fitting_result_QA_sha256": RESULT_QA_SHA, "no_fusion_planning_host_anatomy_growth_training_acceptance": True,
        "selected_series_id": state["selectedSeriesId"], "selected_viewer": state["states"][expected_series]["viewer"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-pins", type=Path)
    args = parser.parse_args()
    controls = generated_controls()
    if args.released_pins is None:
        print(json.dumps({"status": "prepared_generated_controls_pass", "controls": controls})); return
    started = time.monotonic(); pins = json.loads(args.released_pins.read_text())
    mode = pins.get("mode", "package"); assert mode in {"package", "gui_save"}
    destination = ROOT/"artifacts/ixi265-tissue-display-independent-v1"/(
        "independent-qa.json" if mode == "package" else "gui-save-independent-qa.json")
    assert not destination.exists(), "Frozen audit report must not be replaced"
    inputs, before, sealed, receipt, attempt = admitted_records(pins)
    candidate = PACKAGE
    saved_preservation = None
    if mode == "gui_save":
        candidate = ROOT/pins["saved_bundle_path"]
        assert candidate.is_relative_to(ROOT/"outputs") and candidate.resolve() == candidate
        assert candidate != PACKAGE and sha(candidate) == pins["saved_bundle_sha256"]
        saved_preservation = preservation(PACKAGE, candidate, original_count=37, display_count=3, added_displays=0)
    bundle_before = stamp(candidate)
    memory_guard("admitted_inputs_hashed_before_decode")
    result = audit_package(candidate, inputs, receipt, pins)
    gc.collect()
    assert before == {relative: stamp(ROOT/relative) for relative in before}
    assert sealed == {name: stamp(PACKAGE.parent/name) for name in sealed}
    assert bundle_before == stamp(candidate) and sha(PACKAGE) == pins["package_sha256"]
    assert not any(name.split(".")[0] in {"ants", "torch", "monai", "totalsegmentator"} for name in sys.modules)
    memory_guard("finished")
    report = {"schema": "ixi265-tissue-display-independent-bundle-audit-v1", "mode": mode,
        "status": "pass_display_only_bundle_preserved_unaccepted", "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": {"path": str(candidate.relative_to(ROOT)), **bundle_before}, "result": result,
        "gui_save_preservation": saved_preservation, "frozen_inputs_unchanged": before,
        "package_metadata_unchanged": sealed, "preparation_sha256": PREP_SHA,
        "packaging_resources": {k: attempt[k] for k in ("elapsed_seconds", "exit_code", "peak_supervisor_and_worker_rss_bytes_sampled",
            "minimum_available_memory_bytes_sampled", "minimum_free_disk_bytes_sampled", "output_bytes")},
        "generated_controls": controls, "memory_samples": MEMORY,
        "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": time.monotonic()-started, "auditor_sha256": sha(__file__), "release_pins_sha256": sha(args.released_pins),
        "no_native_model_solver_or_GUI_execution": True,
        "limitations": ["This verifies saved display data and provenance, not anatomical validity.",
            "Three tissue names remain intensity-order assumptions; eight missing-intensity cells remain UNKNOWN.",
            "Raw posteriors remain membership probabilities, not physical partial-volume fractions.",
            "No planning, training, host or growth promotion is established.",
            "Actual native GUI observations belong to the parent; this audit independently verifies file persistence.",
            "The prior three real fitting attempts, two failures and one unaccepted numerical candidate remain unchanged."]}
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"; assert "/Users/" not in content
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream: stream.write(content)
    print(json.dumps({"artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination),
        "status": report["status"], "peak_rss_bytes": report["own_process_peak_rss_bytes"]}))


if __name__ == "__main__": main()
