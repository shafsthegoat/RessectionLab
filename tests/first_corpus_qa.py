"""Opt-in independent small-array numerical and metadata-only corpus audit.

No patient image/bundle/weight is opened; no inference, growth pilot or downloads.
Execute with this checkout's src on PYTHONPATH. This is not a pytest dependency.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from resectionlab import patient_corpus as pc, virtual_glioma as vg
from resectionlab.core import CaseData, SourceRef, array_digest, thaw_json
from resectionlab.structural_evidence import structural_frame_hash

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PEOPLE = {**{f"BTC:sub-PAT{x}": "TRAIN" for x in ("05", "16", "20", "22", "25", "28")},
                   "UCSF-PDGM:004": "DEVELOPMENT", "SynthRAD2023:1BA336": "TRAIN", "Lausanne:sub-000": "TRAIN"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sealed(document, key):
    assert document[key] == json_digest({k: v for k, v in document.items() if k != key}), key


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden image/array/file access in metadata-only control")


def must_reject(call):
    try:
        call()
    except ValueError as error:
        return type(error).__name__
    raise AssertionError("Protected or invalid metadata was admitted")


def generated_host():
    shape = (11, 10, 9)
    brain = np.ones(shape, bool)
    brain[0, :, :] = False
    wm = np.indices(shape)[0] < 5
    gm = ~wm
    wm[5:7, 4:6, 3:5] = gm[5:7, 4:6, 3:5] = False
    angle = .61
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1.]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-2., 1.5, 2.5])
    affine[:3, 3] = [19., -7., 5.]
    case = CaseData("generated-independent-fk", np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        {"wm": wm, "gm": gm}, affine,
        (SourceRef("generated", "generated://independent-fk", provenance="simulated", native_frame="RAS+"),),
        brain_mask=brain, metadata={"is_synthetic": True, "role": "TRAIN", "patient_group": "generated:independent-fk"})
    layers = {name: {"kind": "brain_tissue", "tissue_class": kind, "source_role": "generated_numerical_control",
        "source_sha256": "1" * 64, "source_image_hash": array_digest(case.mri), "source_frame_hash": structural_frame_hash(case)}
        for name, kind in (("wm", "white_matter"), ("gm", "gray_matter"))}
    return replace(case, metadata={**case.metadata, "anatomy_layers": layers})


def qualification(case, shape, affine):
    return vg.qualify_host(case, patient_group="generated:independent-fk", role="TRAIN",
        healthy_source_evidence={"status": "generated_numerical_control", "patient_group": "generated:independent-fk",
            "source_url": "generated://independent-control", "source_sha256": "2" * 64,
            "statement": "Generated numerical control, no person or clinical evidence"},
        qualified_by="independent-stage3-qa", grid_shape=list(shape), grid_affine=affine,
        source_wm_layer="wm", source_gm_layer="gm")


def manual_linear(array, xyz):
    # Explicit voxel-center interpolation; no scipy resampling or wrapper helpers.
    if np.any(xyz < 0) or np.any(xyz > np.array(array.shape) - 1):
        return 0.
    low = np.floor(xyz).astype(int)
    fraction = xyz - low
    value = 0.
    for corner in itertools.product((0, 1), repeat=3):
        weight = np.prod([fraction[i] if corner[i] else 1 - fraction[i] for i in range(3)])
        index = tuple(np.minimum(low + corner, np.array(array.shape)-1))
        value += weight * float(array[index])
    return value


def independent_generator(domain, wm, gm, spacing, diffusion, ratio):
    # Dense conservative face operator for a tiny control, not production code.
    size = domain.size
    matrix = np.zeros((size, size))
    conductivity = diffusion * (wm + gm / ratio)
    for index in np.ndindex(domain.shape):
        if not domain[index]:
            continue
        i = np.ravel_multi_index(index, domain.shape)
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] >= domain.shape[axis] or not domain[tuple(neighbor)]:
                continue
            neighbor = tuple(neighbor)
            j = np.ravel_multi_index(neighbor, domain.shape)
            rate = (conductivity[index] + conductivity[neighbor]) / (2 * spacing[axis]**2)
            matrix[i, j] += rate
            matrix[j, i] += rate
            matrix[i, i] -= rate
            matrix[j, j] -= rate
    assert np.allclose(matrix, matrix.T, rtol=0, atol=0)
    assert np.max(np.abs(matrix.sum(axis=0))) < 1e-15
    return matrix


def numerical_audit():
    case = generated_host()
    for name, values, order in (("brain", case.brain_mask, 0), ("wm", case.compartments["wm"], 1), ("gm", case.compartments["gm"], 1)):
        copied = vg._resample(values, case.affine, case.affine, values.shape, order)
        assert np.array_equal(copied, values), "Identical rotated grid loses or changes source " + name + " voxels"
    shape = (6, 5, 4)
    mapping = np.diag([1.5, 1.25, 1.125, 1.])
    mapping[:3, 3] = [.2, .3, .4]
    affine = case.affine @ mapping
    receipt = qualification(case, shape, affine)
    _, _, spacing, domain, wm, gm, _ = vg._validate_qualification(case, receipt)
    expected_wm = np.zeros(shape)
    expected_gm = np.zeros(shape)
    expected_brain = np.zeros(shape, bool)
    world = np.zeros((*shape, 3))
    for index in np.ndindex(shape):
        xyz = (mapping @ np.r_[index, 1.])[:3]
        expected_wm[index] = manual_linear(case.compartments["wm"], xyz)
        expected_gm[index] = manual_linear(case.compartments["gm"], xyz)
        expected_brain[index] = case.brain_mask[tuple(np.floor(xyz+.5).astype(int))]
        world[index] = (affine @ np.r_[index, 1.])[:3]
    assert np.allclose(wm, expected_wm, rtol=0, atol=5e-15)
    assert np.allclose(gm, expected_gm, rtol=0, atol=5e-15)
    expected_domain = expected_brain & (expected_wm + expected_gm >= .1)
    assert np.array_equal(domain, expected_domain)
    center = world[2, 2, 1]
    params = vg.FKParameters(seed_center_ras_mm=tuple(center), seed_sigma_mm=3.1, seed_peak=.27,
        diffusion_mm2_per_day=.65, growth_per_day=.14, wm_to_gm_diffusion_ratio=6., duration_days=1.7, max_dt_days=.13)
    radius_squared = np.sum((world-center)**2, axis=-1)
    expected_initial = .27 * np.exp(-.5 * radius_squared/3.1**2)
    expected_initial[(~domain) | (radius_squared > (3.1*3)**2)] = 0.
    matrix = independent_generator(domain, expected_wm, expected_gm, spacing, .65, 6.)
    step_limit = 1 / (float((-np.diag(matrix)).max())+.14)
    steps = math.ceil(1.7/min(.13, .9*step_limit))
    dt = 1.7/steps
    expected = expected_initial.flatten()
    for _ in range(steps):
        expected = expected + dt*(matrix @ expected + .14*expected*(1-expected))
    result = vg.simulate_virtual_glioma(case, receipt, params)
    initial_error = float(np.max(np.abs(result.initial_concentration-expected_initial)))
    final_error = float(np.max(np.abs(result.concentration-expected.reshape(shape))))
    assert initial_error < 2e-15 and final_error < 2e-15
    assert result.manifest["finalTimeDays"] == 1.7
    assert result.manifest["numericalContract"]["executedSteps"] == steps
    assert not np.any(result.concentration[~domain])
    assert abs(result.manifest["integralConcentrationMm3"]-float(result.concentration.sum())*abs(np.linalg.det(affine[:3,:3]))) < 1e-11
    private = replace(case, source_compartments={"hidden_annotation": np.ones(case.mri.shape, bool)},
        metadata={**case.metadata, "hidden_hazards": {"future_motor_outcome": "counterfactual"}})
    alternate = vg.simulate_virtual_glioma(private, qualification(private, shape, affine), params)
    assert np.array_equal(result.concentration, alternate.concentration)
    must_reject(lambda: vg.simulate_virtual_glioma(case, receipt, params, wm_fraction=gm))
    before = case.semantic_hash
    assert before == receipt["sourceBinding"]["caseHash"]
    assert result.manifest["newPatientCount"] == 0 and not result.manifest["clinicalValidation"]
    return {"status": "pass", "generated_control_only": True, "source_shape": list(case.mri.shape), "simulation_shape": list(shape),
        "oblique_reflected_anisotropic_coarse_mapping_exact": True, "manual_trilinear_fractions_and_nearest_brain": True,
        "identical_oblique_source_grid_preserves_every_boundary_voxel": True,
        "physical_gaussian_max_error": initial_error, "dense_independent_face_operator_max_error": final_error,
        "steps": steps, "physical_time_days": 1.7, "source_preserved": True,
        "foreign_fractions_rejected": True, "hidden_annotation_and_hazard_metadata_invariance": True,
        "source_wm_gm_are_generated_controls": True, "simulation_id": result.manifest["simulationId"]}


def analytic_diffusion_audit():
    kernel, upstream = vg.load_pinned_kernel()
    errors = []
    for n in (8, 16, 32):
        spacing = (16/n, 2., 3.)
        domain = np.ones((n, 2, 2), bool)
        phase = np.cos(np.pi*(np.arange(n)+.5)/n)[:, None, None]
        initial = np.broadcast_to(.3+.2*phase, domain.shape).copy()
        coefficients = vg.prepare_fk_coefficients(kernel, domain, None, None,
            tissue_mode=vg.UNIFORM_DIFFUSION_DOMAIN, threshold=.1, diffusion=.8, ratio=1.)
        dt = .05*spacing[0]**2/.8
        steps = math.ceil(1/dt)
        dt = 1/steps
        actual = np.pad(initial, 1)
        for _ in range(steps):
            actual = kernel.FK_update(actual, coefficients, 0., dt, *spacing)
        final = actual[1:-1, 1:-1, 1:-1]
        exact = .3+.2*phase*np.exp(-.8*(np.pi/16)**2)
        error = float(np.max(np.abs(final-exact)))
        assert abs(float(final.sum())-float(initial.sum())) < 5e-13
        assert np.min(final) >= initial.min() and np.max(final) <= initial.max()
        errors.append(error)
    ratios = [errors[i]/errors[i+1] for i in range(2)]
    assert all(3.7 < ratio < 4.3 for ratio in ratios)
    files = {name: {"sha256": sha(vg.DEFAULT_CACHE/name), "bytes": (vg.DEFAULT_CACHE/name).stat().st_size}
             for name in vg.SOURCE_FILES}
    assert all(files[name]["sha256"] == expected for name, expected in vg.SOURCE_FILES.items())
    license_text = (vg.DEFAULT_CACHE/"LICENSE").read_text()
    assert license_text.startswith("MIT License") and "Copyright (c) 2023 Michal Balcerak" in license_text
    return {"status": "pass", "analytic_problem": "Neumann no-flux cosine diffusion on a 16mm domain",
        "cells": [8, 16, 32], "max_errors": errors, "refinement_ratios": ratios,
        "conserved_integral_and_discrete_maximum_principle": True, "upstream_commit": upstream["commit"],
        "upstream_source_urls": [row["url"] for row in upstream["files"].values()],
        "upstream_files": files, "license": "MIT", "upstream_solve_not_executed": True}


def metadata_guards():
    candidate = {"patient_group": "BTC:sub-PAT05", "role": "TRAIN", "existing_materialized": True,
        "rights": {"license": "CC0", "source": "https://example.org/generated-control"}}
    declarations = [{"roles": {"SELECT": True}}, {"split": {"SELECT": True}}, {"split": {"name": "SELECT"}},
        {"roles": {"SELECT": "true"}}, {"roles": {"SELECT": 1}},
        {"source_collection": {"name": "UPENN-GBM"}}, {"parent": {"source_collection": {"name": "UPENN-GBM"}}},
        {"source_collection": {"name": "UTSW"}}, {"parent": {"usageRole": "MEASUREMENT_EVAL"}}]
    failures, rows = [], []
    host = generated_host()
    for declaration in declarations:
        manifest = {"case_id": "BTC-ds001226-sub-PAT05-preop", "metadata": {"is_synthetic": False, **declaration}}
        for component in ("corpus", "virtual_glioma"):
            try:
                protected = replace(host, metadata={**host.metadata, **declaration})
                with ExitStack() as stack:
                    stack.enter_context(patch("builtins.open", forbidden))
                    stack.enter_context(patch.object(vg, "array_digest", forbidden))
                    stack.enter_context(patch.object(vg, "_resample", forbidden))
                    stack.enter_context(patch.object(vg, "load_pinned_kernel", forbidden))
                    if component == "corpus":
                        must_reject(lambda: pc.require_manifest(manifest, candidate))
                    else:
                        must_reject(lambda: qualification(protected, [6, 5, 4], host.affine))
                rows.append({"component": component, "declaration": declaration, "rejected_before_payload": True})
            except Exception as error:
                failures.append({"component": component, "declaration": declaration, "failure": str(error)})
    pc.require_manifest({"case_id": "BTC-ds001226-sub-PAT05-preop",
        "metadata": {"is_synthetic": False, "roles": {"SELECT": False, "TRAIN": True}}}, candidate)
    return {"status": "pass" if not failures else "fail", "cases": rows, "failures": failures,
        "false_protected_flag_true_train_control_accepted": True}


def real_catalog_metadata(output):
    plan = json.loads((output/"plan.json").read_text())
    catalog = json.loads((output/"catalog.json").read_text())
    sealed(plan, "plan_hash")
    sealed(catalog, "catalog_hash")
    assert {row["patient_group"]: row["role"] for row in plan["candidates"]} == EXPECTED_PEOPLE
    assert {row["patient_group"]: row["role"] for row in catalog["entries"]} == EXPECTED_PEOPLE
    assert catalog["counts"]["qualified_existing_people"] == 9
    assert catalog["counts"]["newly_acquired_people"] == catalog["counts"]["anatomical_truth_accepted_people"] == 0
    assert not catalog["protected_imaging_opened"] and not plan["new_patient_acquisition_allowed"]
    candidates = {row["patient_group"]: row for row in plan["candidates"]}
    summaries = []
    for row in catalog["entries"]:
        receipt = json.loads((output/row["receipt_path"]).read_text())
        sealed(receipt, "receipt_hash")
        assert {key: value for key, value in row.items() if key != "receipt_path"} == receipt
        candidate = candidates[row["patient_group"]]
        source_hashes = {source["sha256"] for source in candidate["source_files"]}
        assert all(image["source_sha256"] in source_hashes and image["source_voxels_equal"] for image in row["imaging"])
        assert row["rights"] == candidate["rights"]
        assert not row["new_patient"] and not row["training_admission_established"] and not row["clinical_use_permitted"]
        assert row["tumor_location"]["lobe"] == row["tumor_location"]["hemisphere"] == "UNKNOWN"
        assert all(not layer["accepted_anatomical_truth"] for layer in row["layers"])
        assert all(not item["independently_validated_registration"] for item in row["registrations"])
        summaries.append({"patient_group": row["patient_group"], "role": row["role"], "receipt_hash": row["receipt_hash"],
            "bundle_sha256": row["bundle_sha256"], "anatomy_present": row["anatomy_present"], "missing_anatomy": row["missing_anatomy"],
            "lobe": "UNKNOWN", "hemisphere": "UNKNOWN", "registration_statuses": [item["status"] for item in row["registrations"]]})
    assert len(pc.query_catalog(output)) == 9
    assert len(pc.query_catalog(output, anatomy="tumor")) == 7
    assert len(pc.query_catalog(output, anatomy="skull")) == 1
    assert len(pc.query_catalog(output, anatomy="vessels")) == 1
    runs = [json.loads(path.read_text()) for path in (output/"runs").glob("*.json")]
    reused = [run for run in runs if run["catalog_hash"] == catalog["catalog_hash"] and run["new_qualification_receipts"] == 0 and run["reused_qualification_receipts"] == 9]
    return {"status": "pass", "catalog_hash": catalog["catalog_hash"], "plan_hash": plan["plan_hash"],
        "counts": catalog["counts"], "receipt_seals_and_source_hash_lineage_verified": True,
        "latest_catalog_has_nine_receipt_resume_record": bool(reused), "entries": summaries,
        "scope": "Metadata-only receipt consistency; previously recorded imaging equality not independently decoded here",
        "real_patient_arrays_opened_by_this_audit": 0, "protected_payloads_opened_by_this_audit": 0}


def generated_resume_audit():
    import nibabel as nib
    from resectionlab.imaging import load_nifti_case, save_case
    with tempfile.TemporaryDirectory(prefix="corpus-independent-generated-") as directory:
        root = Path(directory)
        # This is an explicitly generated software fixture, not actual BTC data.
        image_path, mask_path = root/"generated.nii.gz", root/"generated-label.nii.gz"
        shape = (5, 6, 7)
        affine = np.diag([1.5, 2., 3., 1.])
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        mask = np.zeros(shape, np.uint8)
        mask[1:3, 2:4, 3] = 1
        for path, values in ((image_path, image), (mask_path, mask)):
            nii = nib.Nifti1Image(values, affine)
            nii.header.set_xyzt_units("mm")
            nii.set_qform(affine, 1)
            nii.set_sform(affine, 1)
            nib.save(nii, path)
        case = load_nifti_case(image_path, mask_path, case_id="BTC-ds001226-sub-PAT05-preop", label_map={1: "target"},
            license="CC0-1.0", metadata={"generated_software_fixture": True, "selected_modality": "T1",
                "split": {"role": "TRAIN", "patient_group": "BTC:sub-PAT05"},
                "anatomy_layers": {"target": {"kind": "tumor", "source_role": "imported_source_annotation", "source_sha256": sha(mask_path)}}})
        bundle = root/"BTC-sub-PAT05.ressectionlab"
        save_case(case, bundle)
        participant = root/"participants.tsv"
        participant.write_text("participant_id\nsub-PAT05\n")
        candidate = {"patient_group": "BTC:sub-PAT05", "case_id": case.case_id, "role": "TRAIN", "existing_materialized": True,
            "bundle": bundle.name, "rights": {"license": "CC0-1.0", "source": "https://example.org/generated-software-control"},
            "participant": {"path": participant.name, "sha256": sha(participant), "delimiter": "\t", "key": "participant_id", "value": "sub-PAT05"},
            "source_files": [{"path": path.name, "sha256": sha(path), "role": role} for path, role in ((image_path, "imaging"), (mask_path, "annotation"))]}
        plan = {"schema": pc.VERSION+".plan", "candidates": [candidate], "metadata_evidence": [],
                "existing_materialized_people": 1, "new_patient_acquisition_allowed": False}
        plan["plan_hash"] = json_digest(plan)
        output = root/"audit"
        first = pc.run_qualification(root, output, plan)
        assert first["new_qualification_receipts"] == 1 and not first["rejections"]
        with patch.object(pc, "load_workspace", forbidden):
            second = pc.run_qualification(root, output, plan)
            assert second["new_qualification_receipts"] == 0 and second["reused_qualification_receipts"] == 1
            assert first["catalog_hash"] == second["catalog_hash"]
            original = image_path.read_bytes()
            image_path.write_bytes(original+b"changed-source-hash")
            changed = pc.run_qualification(root, output, plan)
            assert changed["counts"]["existing_materialized_people"] == changed["counts"]["rejected"] == 1
            assert changed["rejections"][0]["failure_code"] == "SOURCE_HASH"
            assert changed["reused_qualification_receipts"] == 0
            image_path.write_bytes(original)
            altered_plan = deepcopy(plan)
            altered_plan["candidates"][0]["role"] = "SELECT"
            altered_plan["plan_hash"] = json_digest({k: v for k, v in altered_plan.items() if k != "plan_hash"})
            must_reject(lambda: pc.run_qualification(root, output, altered_plan))
    return {"status": "pass", "generated_software_fixture_only": True,
        "resume_same_catalog_hash_without_numeric_decode": True, "source_byte_change_refuses_cached_receipt_before_decode": True,
        "rejection_stays_in_denominator": True, "protected_role_change_rejected": True,
        "temporary_generated_files_removed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT/"outputs/patient-corpus-v1")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/patient-corpus-first-tranche-v1/independent-qa.json")
    args = parser.parse_args()
    files = ["src/resectionlab/virtual_glioma.py", "src/resectionlab/patient_corpus.py", "src/resectionlab/rehearsal_episode.py",
             "scripts/qualify_patient_corpus.py", "tests/first_corpus_qa.py"]
    source_hashes = {name: sha(ROOT/name) for name in files}
    checks = {}
    for name, fn in (("numerical", numerical_audit), ("analytic_diffusion", analytic_diffusion_audit),
                     ("metadata_guards", metadata_guards), ("generated_resume", generated_resume_audit),
                     ("real_catalog", lambda: real_catalog_metadata(args.catalog))):
        try:
            checks[name] = fn()
        except Exception as error:
            checks[name] = {"status": "fail", "error_type": type(error).__name__, "message": str(error)}
        print(name, checks[name]["status"], flush=True)
    assert source_hashes == {name: sha(ROOT/name) for name in files}, "Implementation changed during independent audit; rerun once frozen"
    report = {"schema": "independent-stage3-software-qa-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(check["status"] == "pass" for check in checks.values()) else "fail",
        "source_sha256": source_hashes, "checks": checks,
        "retained_findings": [{"id": "CORPUS_STRUCTURED_PROTECTED_ROLE_BYPASS", "initial_status": "reproduced_metadata_only",
            "examples": ["roles={SELECT:true}", "split={SELECT:true}", "source_collection.name=UPENN-GBM"],
            "initial_effect": "Accepted under an allowlisted TRAIN identity; no protected payload was opened by audit"},
            {"id": "VALID_OBLIQUE_TISSUE_MAPPING_REJECTED", "initial_status": "reproduced_generated_control",
             "initial_effect": "Linear interpolation of valid binary tissue masks produced1.0000000000000002; exact per-layer bound rejected the host",
             "required_fix": "Separate documented floating-point interpolation tolerance from material invalid fractions and solver result clipping"},
            {"id": "VIRTUAL_HOST_STRUCTURED_ROLE_AND_NESTED_COLLECTION_BYPASS", "initial_status": "reproduced_metadata_only",
             "examples": ["roles={SELECT:'true'}", "roles={SELECT:1}", "parent.source_collection.name=UPENN-GBM", "source_collection.name=UTSW"],
             "initial_effect": "Shared development admission helper accepted these declarations; only generated numerical controls were used"},
            {"id": "IDENTICAL_OBLIQUE_GRID_BOUNDARY_LOSS", "initial_status": "reproduced_generated_control",
             "initial_effect": "Floating roundoff in inv(A)@A made constant-zero interpolation drop63 brain,54 WM and27 GM control voxels on an identical source/target grid",
             "required_fix": "Preserve exact same-grid identity and document numerical handling of physical mapping"}],
        "limitations": ["Generated numerical controls do not establish biological validity or patient-specific growth accuracy.",
            "No real-host growth pilot, training, inference or new patient acquisition performed by this auditor.",
            "Catalog source-image equality is a retained engineering receipt claim; this review opens only metadata."]}
    encoded = json.dumps(report, indent=2)+"\n"
    assert "/Users/" not in encoded and "file://" not in encoded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
