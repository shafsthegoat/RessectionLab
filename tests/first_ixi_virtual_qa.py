"""Opt-in audit of only the two exact admitted IXI265 pilot bundles.

No model execution, protected cohort, GUI or implementation changes. The scalar
FK result is checked using an independent pairwise conservative replay, without
calling the production simulator or upstream solver. Existing files stay intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import resource
import sys
from zipfile import ZipFile

import numpy as np

from resectionlab.core import array_digest
from resectionlab.workspace_bundle import load_workspace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/poc-ixi265"
PILOT = BASE/"virtual-FK-pilot-001"
HOST = BASE/"IXI265-Guys-0845.ressectionlab"
DESCENDANT = PILOT/"IXI265-Guys-0845-virtual-FK-pilot-001.ressectionlab"
HOST_SHA = "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"
DESCENDANT_SHA = "aeb2d7574bcd091009da60252763ac401a11478b66d8c3a7e71d35ef56ba2b51"
TARGET = "virtual_glioma_threshold_0p1"
FROZEN_QA = ROOT/"artifacts/patient-corpus-first-tranche-v1/independent-qa.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    return "sha256:"+hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stamp(path):
    s = path.stat()
    return {"sha256": sha(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def exact_streams(a, b, name):
    digest, size = hashlib.sha256(), 0
    with a.open(name) as left, b.open(name) as right:
        while True:
            x, y = left.read(1024*1024), right.read(1024*1024)
            assert x == y, "Embedded original bytes changed"
            if not x:
                break
            digest.update(x)
            size += len(x)
    return digest.hexdigest(), size


def source_preservation():
    with ZipFile(HOST) as a, ZipFile(DESCENDANT) as b:
        hm, dm = (json.loads(z.read("manifest.json")) for z in (a, b))
        hw, dw = (json.loads(z.read("workspace.json")) for z in (a, b))
        assert hm["frame"] == dm["frame"] == "RAS+"
        assert hw["originalFiles"] == dw["originalFiles"] and len(hw["originalFiles"]) == 16
        originals = []
        for item in hw["originalFiles"]:
            digest, size = exact_streams(a, b, item["member"])
            assert digest == item["sha256"] and size == item["bytes"]
            originals.append({key: item[key] for key in ("name", "role", "sha256", "bytes")})
        displays = []
        assert len(hw["images"]) == len(dw["images"]) == 2
        for old, new in zip(hw["images"], dw["images"]):
            for key in ("sourceCaseHash", "sourceFrameHash", "scope", "modality", "annotationKind"):
                assert old["source"][key] == new["source"][key]
            xb, yb = a.read(old["member"]), b.read(new["member"])
            assert hashlib.sha256(xb).hexdigest() == old["sha256"]
            assert hashlib.sha256(yb).hexdigest() == new["sha256"]
            with ZipFile(BytesIO(xb)) as x, ZipFile(BytesIO(yb)) as y:
                assert set(x.namelist()) == set(y.namelist())
                for member in x.namelist():
                    assert x.read(member) == y.read(member), "Display reference contents changed"
                nested = json.loads(x.read("manifest.json"))
                assert nested["case_semantic_hash"] == old["source"]["sourceCaseHash"]
            displays.append({"modality": old["source"]["modality"], "scope": old["source"]["scope"],
                "case_hash": old["source"]["sourceCaseHash"], "frame_hash": old["source"]["sourceFrameHash"],
                "all_nested_member_bytes_exact": True, "container_bytes_exact": xb == yb,
                "container_note": "ZIP container bytes and series IDs may change on save; every nested member, array and metadata byte remains exact."})
        assert hm["array_index"]["source_compartments"] == dm["array_index"]["source_compartments"] == {}
        assert hm["array_index"]["compartments"] == {}
        assert set(dm["array_index"]["compartments"]) == {TARGET}
        with np.load(BytesIO(a.read("arrays.npz")), allow_pickle=False) as x, np.load(BytesIO(b.read("arrays.npz")), allow_pickle=False) as y:
            rows = {}
            for name in x.files:
                assert np.array_equal(x[name], y[name]), "Primary source array changed"
                rows[name] = {"shape": list(x[name].shape), "dtype": x[name].dtype.str, "array_hash": array_digest(x[name])}
            affine = y["affine"].copy()
            brain = y["brain_mask"].astype(bool)
            target = y[dm["array_index"]["compartments"][TARGET]].astype(bool)
        assert dm["source_refs"][:-1] == hm["source_refs"]
        generated = dm["source_refs"][-1]
        assert generated["provenance"] == "simulated" and generated["source_id"] == "virtual_glioma_fk"
        assert dm["unknowns"][:-1] == hm["unknowns"]
        for key, value in hm["metadata"].items():
            if key == "anatomy_layers":
                for name, layer in value.items():
                    assert dm["metadata"][key][name] == layer
            else:
                assert dm["metadata"][key] == value
        assert set(dm["metadata"])-set(hm["metadata"]) == {"virtual_geometry"}
        assert dm["metadata"]["split"]["role"] == "TRAIN" and dm["metadata"]["training_admitted"] is False
        assert dm["metadata"]["workspace_registration"]["overlayPermitted"] is False
        layer = dm["metadata"]["anatomy_layers"][TARGET]
        assert layer["provenance"] == "simulated" and not layer["observed_patient_tumor"] and not layer["clinical_validation"]
        virtual = dm["metadata"]["virtual_geometry"]
        assert virtual["new_patient_count"] == 0 and virtual["virtual_lesion_count"] == 1
        assert virtual["placement"]["hemisphere"] == virtual["placement"]["lobe"] == "UNKNOWN_UNREVIEWED"
        return hm, dm, dw, affine, brain, target, {"primary_arrays": rows, "source_compartments_exact_and_empty": True,
            "originals": originals, "all_16_original_bytes_exact": True, "display_references": displays,
            "source_metadata_and_references_preserved": True, "inherited_role": "TRAIN", "training_admitted": False,
            "generated_layer_is_not_observed_anatomy": True, "registration_remains_unreviewed_and_overlay_not_permitted": True}


def independent_nearest(source, source_affine, target_shape, target_affine):
    # Explicit world->source coordinates, sliced to avoid a full-volume point cloud.
    inverse = np.linalg.inv(source_affine)
    output = np.zeros(target_shape, bool)
    j, k = np.indices(target_shape[1:], dtype=float)
    for i in range(target_shape[0]):
        world = [target_affine[axis, 0]*i + target_affine[axis, 1]*j + target_affine[axis, 2]*k + target_affine[axis, 3] for axis in range(3)]
        coordinates = [sum(inverse[axis, n]*world[n] for n in range(3))+inverse[axis, 3] for axis in range(3)]
        valid = np.ones(j.shape, bool)
        for axis in range(3):
            valid &= (coordinates[axis] >= 0) & (coordinates[axis] <= source.shape[axis]-1)
        index = tuple(np.floor(coordinates[axis][valid]+.5).astype(int) for axis in range(3))
        output[i][valid] = source[index]
    return output


def numerical_and_transfer(affine, brain, target, manifest, receipt):
    shape = tuple(manifest["gridShape"])
    grid = np.asarray(manifest["gridAffineRasMm"], float)
    domain = independent_nearest(brain, affine, shape, grid)
    assert array_digest(domain) == manifest["mapping"]["admittedDomainHash"]
    assert manifest["tissueMode"] == "UNIFORM_DIFFUSION_DOMAIN" and manifest["mapping"]["sourceTissues"] == {}
    assert manifest["mapping"]["wmFractionHash"] is None and manifest["mapping"]["gmFractionHash"] is None
    p, contract = manifest["parameters"], manifest["numericalContract"]
    spacing = np.linalg.norm(grid[:3, :3], axis=0)
    voxel_volume = abs(np.linalg.det(grid[:3, :3]))
    indices = np.indices(shape, dtype=float)
    distance_squared = np.zeros(shape)
    for axis in range(3):
        coordinate = sum(grid[axis, n]*indices[n] for n in range(3))+grid[axis, 3]
        distance_squared += (coordinate-p["seed_center_ras_mm"][axis])**2
    expected = p["seed_peak"]*np.exp(-distance_squared/(2*p["seed_sigma_mm"]**2))
    expected[(~domain) | (distance_squared > (p["seed_support_sigma"]*p["seed_sigma_mm"])**2)] = 0.
    with np.load(PILOT/"virtual-glioma.npz", allow_pickle=False) as data:
        initial, final = data["initial_concentration"], data["concentration"]
        assert np.array_equal(initial, expected)
        assert array_digest(initial) == manifest["initialConcentrationHash"]
        assert array_digest(final) == manifest["concentrationHash"]
        masks = {row["name"]: data[row["name"]].copy() for row in manifest["thresholdMasks"]}
    edges = []
    outflow = np.zeros(shape)
    for axis, h in enumerate(spacing):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        rate = (domain[left] & domain[right])*p["diffusion_mm2_per_day"]/h**2
        outflow[left] += rate
        outflow[right] += rate
        edges.append((left, right, rate))
    stable = 1/(float(outflow.max())+p["growth_per_day"])
    planned = math.ceil(p["duration_days"]/min(p["max_dt_days"], p["stability_safety"]*stable))
    dt = p["duration_days"]/planned
    assert planned == contract["plannedSteps"] == contract["executedSteps"] == 60
    assert dt == contract["dtDays"] and manifest["stopReason"] == "time"
    assert abs(stable-contract["stabilityBoundDays"]) < 1e-12
    history = {row["step"]: row for row in manifest["timeSeriesMetrics"]}
    history_error = 0.
    balance_residual = 0.
    for step in range(planned+1):
        if step in history:
            row = history[step]
            assert row["timeDays"] == step*dt
            history_error = max(history_error, abs(float(expected.sum())*voxel_volume-row["integralConcentrationMm3"]))
            assert abs(expected.max()-row["maximum"]) < 1e-13 and expected.min() == row["minimum"]
        if step == planned:
            break
        reaction = p["growth_per_day"]*expected*(1-expected)
        change = reaction.copy()
        for left, right, rate in edges:
            flux = rate*(expected[right]-expected[left])
            change[left] += flux
            change[right] -= flux
        predicted_sum = float(expected.sum())+dt*float(reaction.sum())
        expected += dt*change
        balance_residual = max(balance_residual, abs(float(expected.sum())-predicted_sum)*voxel_volume)
        assert expected.min() >= 0 and expected.max() <= 1 and not expected[~domain].any()
    error = float(np.max(np.abs(expected-final)))
    assert error < 1e-13 and history_error < 1e-9
    assert balance_residual < contract["integralBalanceToleranceMm3"]
    assert manifest["finalTimeDays"] == planned*dt
    assert manifest["resources"]["cellUpdates"] == planned*int(np.prod(np.array(shape)+2))
    assert abs(float(final.sum())*voxel_volume-manifest["integralConcentrationMm3"]) < 1e-10
    threshold_summary = []
    for row in manifest["thresholdMasks"]:
        mask = masks[row["name"]]
        assert np.array_equal(mask, (final >= row["threshold"]) & domain)
        assert array_digest(mask) == row["maskHash"]
        assert int(mask.sum())*voxel_volume == row["volumeMm3"]
        threshold_summary.append({"threshold": row["threshold"], "coarse_cells": int(mask.sum()), "coarse_volume_mm3": row["volumeMm3"]})
    transferred = independent_nearest(masks["virtual_concentration_ge_0p1"], grid, brain.shape, affine)
    native = transferred & brain
    assert np.array_equal(native, target)
    display = receipt["display_transfer"]
    assert int(transferred.sum()) == display["transferred_cells_before_intersection"]
    assert int((transferred & ~brain).sum()) == display["cells_outside_source_brain_excluded_by_display_definition"]
    assert int(native.sum()) == display["native_cells"] == 841
    native_volume = int(native.sum())*abs(np.linalg.det(affine[:3, :3]))
    assert native_volume == display["native_volume_mm3"]
    assert not display["pde_arrays_modified"]
    return {"grid_shape": list(shape), "grid_affine_ras_mm": grid.tolist(), "independent_domain_mapping_exact": True,
        "independent_physical_seed_exact": True, "independent_uniform_pairwise_flux_replay": True,
        "steps": planned, "dt_days": dt, "final_time_days": manifest["finalTimeDays"],
        "maximum_final_concentration_error": error, "maximum_history_integral_error_mm3": history_error,
        "independent_maximum_reaction_balance_residual_mm3": balance_residual,
        "thresholds": threshold_summary, "independent_native_transfer_exact": True,
        "native_target_cells": int(native.sum()), "native_display_volume_mm3": native_volume,
        "outside_brain_display_cells_excluded": int((transferred & ~brain).sum()),
        "coarse_and_native_volumes_are_distinct_declared_grids": True,
        "uniform_mode_has_no_GM_WM_or_CSF_barrier_claim": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/ixi265-virtual-pilot-v1/independent-qa.json")
    args = parser.parse_args()
    watched = [HOST, DESCENDANT, PILOT/"manifest.json", PILOT/"bundle-receipt.json", PILOT/"virtual-glioma.npz", FROZEN_QA]
    before = {str(path.relative_to(ROOT)): stamp(path) for path in watched}
    assert before[str(HOST.relative_to(ROOT))]["sha256"] == HOST_SHA
    assert before[str(DESCENDANT.relative_to(ROOT))]["sha256"] == DESCENDANT_SHA
    manifest = json.loads((PILOT/"manifest.json").read_text())
    receipt = json.loads((PILOT/"bundle-receipt.json").read_text())
    hm, dm, workspace, affine, brain, target, preservation = source_preservation()
    assert manifest["sourceBinding"]["caseHash"] == hm["case_semantic_hash"] == receipt["host"]["case_hash"]
    assert receipt["case_hash"] == dm["case_semantic_hash"]
    assert manifest["sourceBinding"]["imageHash"] == preservation["primary_arrays"]["mri"]["array_hash"]
    assert manifest["sourceBinding"]["brainEnvelopeHash"] == array_digest(brain)
    assert manifest["sourceBinding"]["affineHash"] == array_digest(affine)
    assert manifest["sourceBinding"]["sourceRefsHash"] == json_digest(hm["source_refs"])
    frame_hash = json_digest({"shape": list(brain.shape), "affine": affine.tolist(), "frame": "RAS+", "physical_units": "mm"})
    assert frame_hash == manifest["sourceBinding"]["frameHash"]
    assert manifest["simulationId"] == json_digest({key: value for key, value in manifest.items() if key not in {"simulationId", "resources", "outputFiles"}})
    q = manifest["qualification"]
    assert q["qualificationHash"] == json_digest({key: value for key, value in q.items() if key != "qualificationHash"})
    assert q["patientGroup"] == "IXI:265" and q["role"] == manifest["inheritedHostRole"] == receipt["role"] == "TRAIN"
    assert q["sourceBinding"] == manifest["sourceBinding"]
    health = q["healthySourceEvidence"]
    assert health["patient_group"] == "IXI:265" and health["status"] == "source_reported_healthy"
    assert len([item for item in preservation["originals"] if item["sha256"] == health["source_sha256"]]) == 1
    assert manifest["newPatientCount"] == receipt["new_patient_count"] == 0
    assert manifest["virtualLesionCount"] == receipt["virtual_lesion_count"] == 1
    assert receipt["training_admitted"] is False and manifest["originalAcquiredStructuresChanged"] is False
    output_record = manifest["outputFiles"]["virtual-glioma.npz"]
    assert sha(PILOT/"virtual-glioma.npz") == output_record["sha256"] and (PILOT/"virtual-glioma.npz").stat().st_size == output_record["bytes"]
    assert dm["source_refs"][-1]["sha256"] == output_record["sha256"]
    assert receipt["bundle_sha256"] == DESCENDANT_SHA and receipt["host_bundle_sha256"] == HOST_SHA
    assert manifest["hostPatientGroup"] == receipt["patient_group"] == "IXI:265"
    assert not manifest["clinicalValidation"] and not receipt["anatomical_acceptance"]
    numeric = numerical_and_transfer(affine, brain, target, manifest, receipt)
    # Standard fresh loader independently validates all saved hashes and semantics.
    case, _, restored = load_workspace(DESCENDANT)
    assert case.semantic_hash == receipt["case_hash"] and restored["sessionHash"] == receipt["session_hash"]
    assert len(restored["originals"]) == 16 and len(restored["sources"]) == 2
    assert array_digest(case.mri) == receipt["host"]["image_hash"]
    assert np.array_equal(case.brain_mask, brain) and np.array_equal(case.compartments[TARGET], target)
    assert before == {str(path.relative_to(ROOT)): stamp(path) for path in watched}
    report = {"schema": "independent-ixi265-virtual-pilot-audit-v1", "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Only exact admitted host and generated descendant; independent artifact/accounting verification, not clinical acceptance",
        "input_stamps": before, "source_preservation": preservation, "numerical_and_physical_transfer": numeric,
        "source_frame_hash": frame_hash, "simulation_id": manifest["simulationId"], "fresh_workspace_load_hashes_valid": True,
        "source_health_evidence_bound_to_exact_archived_page": True,
        "source_and_output_files_unchanged": True, "frozen_stage3_audit_unchanged": True,
        "new_model_inference_or_training": False, "production_solver_executed_by_audit": False,
        "auditor_sha256": sha(Path(__file__)), "auditor_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": ["Uniform parameter fiction: no classified WM/GM or CSF barriers.",
            "No vascular avoidance, mass effect, functional risk, edema/enhancement or synthetic MRI appearance.",
            "Native threshold overlay volume differs from coarse-grid scalar-threshold volume by declared sampling.",
            "Software and source preservation verified; no anatomical or biological validation implied.",
            "Actual GUI inspection and a later GUI-reviewed save are separate evidence."]}
    encoded = json.dumps(report, indent=2)+"\n"
    assert "/Users/" not in encoded and "file://" not in encoded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(json.dumps({"status": "pass", "artifact": str(args.output.relative_to(ROOT)), "sha256": sha(args.output), "numeric": numeric}))


if __name__ == "__main__":
    main()
