"""Opt-in final IXI GUI-rehearsal and vessel-artifact audit; no inference or FK solve.

Reads only the named admitted IXI bundles and local saved model outputs. Frozen
earlier audit artifacts are watched but never rewritten. Geometry is explicitly
audited as the conservative bounding-sphere cell cover used for this NIfTI affine.
"""
from datetime import datetime, timezone
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import nibabel as nib
import numpy as np

from resectionlab.core import array_digest, thaw_json
from resectionlab.workspace_bundle import load_workspace
from resectionlab.workspace_imaging import source_registration
import first_release_qa as episode_qa
import first_release_gui_save as gui_qa
from first_ixi_virtual_qa import ROOT, BASE, HOST, HOST_SHA, DESCENDANT, DESCENDANT_SHA, TARGET, sha, stamp, exact_streams

REVIEWED = BASE/"virtual-FK-pilot-001/IXI265-virtual-FK-pilot-001-reviewed.ressectionlab"
VESSEL_DIR = BASE/"vesselfm-base-v1"
VESSEL = VESSEL_DIR/"IXI265-vesselfm-estimates.ressectionlab"
VESSEL_REVIEWED = VESSEL_DIR/"IXI265-vesselfm-reviewed.ressectionlab"
VESSEL_SHA = "55a38ae5792c889d3f8a8e28c635c92f78ca80d42690028b7eb028d638980e0e"
FROZEN = [ROOT/"artifacts/patient-corpus-first-tranche-v1/independent-qa.json", ROOT/"artifacts/ixi265-virtual-pilot-v1/independent-qa.json"]


def conservative_capsule_cells(affine, shape, start, end, radius):
    """Independent center-to-segment test with enclosing voxel-sphere radius.

This intentionally reproduces conservative cell intersection, not exact box
intersection. It calls no runtime geometry helpers and supports any orientation.
"""
    start, end = np.asarray(start), np.asarray(end)
    inverse = np.linalg.inv(affine)
    corners = np.array(list(product((-.5, .5), repeat=3))) @ affine[:3, :3].T
    reach = radius + float(np.linalg.norm(corners, axis=1).max())
    points = np.array([start, end]) @ inverse[:3, :3].T + inverse[:3, 3]
    extent = np.linalg.norm(inverse[:3, :3], axis=1)*reach
    lo = np.maximum(0, np.floor(points.min(0)-extent).astype(int))
    hi = np.minimum(np.array(shape)-1, np.ceil(points.max(0)+extent).astype(int))
    if np.any(lo > hi):
        return set()
    indices = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]+1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    centers = indices @ affine[:3, :3].T + affine[:3, 3]
    direction = end-start
    fraction = np.clip((centers-start) @ direction/(direction@direction), 0, 1) if direction@direction > 1e-18 else np.zeros(len(indices))
    distances = np.linalg.norm(centers-(start+fraction[:, None]*direction), axis=1)
    hit = indices[distances <= reach+1e-9]
    return set(map(int, np.ravel_multi_index(hit.T, shape))) if len(hit) else set()


def source_compare(reference, actual, *, additional_originals=0, additional_displays=0, reference_originals=16, reference_displays=2):
    with ZipFile(reference) as a, ZipFile(actual) as b:
        am, bm = (json.loads(z.read("manifest.json")) for z in (a, b))
        aw, bw = (json.loads(z.read("workspace.json")) for z in (a, b))
        assert {k: v for k, v in am.items() if k not in {"artifacts", "workspace_sha256"}} == {k: v for k, v in bm.items() if k not in {"artifacts", "workspace_sha256"}}
        with ZipFile(BytesIO(a.read("arrays.npz"))) as x, ZipFile(BytesIO(b.read("arrays.npz"))) as y:
            assert set(x.namelist()) == set(y.namelist())
            for member in x.namelist():
                assert x.read(member) == y.read(member)
        assert len(aw["originalFiles"]) == reference_originals and len(bw["originalFiles"]) == reference_originals+additional_originals
        assert aw["originalFiles"] == bw["originalFiles"][:reference_originals]
        for original in aw["originalFiles"]:
            digest, size = exact_streams(a, b, original["member"])
            assert digest == original["sha256"] and size == original["bytes"]
        import hashlib
        for original in bw["originalFiles"][reference_originals:]:
            raw = b.read(original["member"])
            assert hashlib.sha256(raw).hexdigest() == original["sha256"] and len(raw) == original["bytes"]
        assert len(aw["images"]) == reference_displays and len(bw["images"]) == reference_displays+additional_displays
        for old, new in zip(aw["images"], bw["images"][:reference_displays]):
            assert old["source"] == new["source"]
            with ZipFile(BytesIO(a.read(old["member"]))) as x, ZipFile(BytesIO(b.read(new["member"]))) as y:
                assert set(x.namelist()) == set(y.namelist())
                for member in x.namelist():
                    assert x.read(member) == y.read(member)
        return {"primary_arrays_and_semantic_metadata_exact": True, "case_hash": bm["case_semantic_hash"],
            "parent_original_count": reference_originals, "all_parent_original_bytes_exact": True,
            "parent_display_count": reference_displays, "all_parent_display_contents_exact": True,
            "additional_originals_hash_verified": additional_originals, "additional_display_series": additional_displays}


def rehearsal_audit():
    preservation = source_compare(DESCENDANT, REVIEWED)
    case, artifacts, _ = load_workspace(REVIEWED)
    artifacts = thaw_json(artifacts)
    axes = case.affine[:3, :3]/np.linalg.norm(case.affine[:3, :3], axis=0)
    defect = float(np.max(np.abs(axes.T@axes-np.eye(3))))
    assert 1e-10 < defect < 1e-8

    def frames(actual_case, exported):
        # Adapt the older oracle's fixed target key to this exact virtual mask;
        # no case, source binding, target values or saved state is changed.
        adapter = SimpleNamespace(mri=actual_case.mri, affine=actual_case.affine, brain_mask=actual_case.brain_mask,
            compartments={"monai_tumor_core": actual_case.compartments[TARGET]})
        row = episode_qa.independent_episode_frames(adapter, exported)
        row.pop("independent_capsule_cell_removal_exact")
        row["independent_conservative_sphere_cell_cover_exact"] = True
        return row

    with patch.object(episode_qa, "box_capsule_cells", conservative_capsule_cells), patch.object(gui_qa, "independent_episode_frames", frames):
        replay = gui_qa.replay_saved(case, artifacts)
    assert replay["saved_view"]["branchId"] == "deeper_pass_reassess"
    assert replay["selected_frame"]["index"] == replay["selected_frame"]["last_index"] == 21
    assert replay["selected_frame"]["kind"] == "STOP" and replay["selected_frame"]["metrics"]["terminated"]
    assert replay["live_replay"]["metrics"]["elapsedSeconds"] == 1 and replay["live_replay"]["metrics"]["removedMm3"] > 0
    assert all(row["independent_frame_audit"]["metrics"]["elapsedSeconds"] == 7 for row in replay["branches"])
    return {"status": "pass", "source_preservation": preservation, "rehearsal": replay,
        "cell_geometry": {"source_affine_orthogonality_defect": defect, "runtime_exact_box_cutoff": 1e-10,
            "selected_geometry": "conservative voxel-bounding-sphere cell cover",
            "meaning": "Independent match to declared conservative removal/clearance coverage; not exact physical tissue intersection"},
        "paused_reopen": {"saved_playing_flag_exists": False, "saved_terminal_frame_verified": True,
            "frontend_control_flow_reviewed": "useRehearsal initializes and resets playing=false, including inspect; branch/frame/tool preferences are restored",
            "frontend_source_sha256": sha(ROOT/"desktop/src/use-rehearsal.ts"),
            "actual_GUI_pause_observation": "Owned by parent; this file audit does not claim an independent GUI observation"}}


def independent_itk_nearest(source, source_affine, target_shape, target_affine):
    """Manual nearest-cell mapping using ITK's half-voxel image extent.

The virtual pilot uses scipy's narrower continuous-center extent. That prior
oracle cannot check pre-coverage ITK border counts without this explicit change.
No production resampling or ITK operation is called here.
"""
    inverse = np.linalg.inv(source_affine)
    output = np.zeros(target_shape, bool)
    j, k = np.indices(target_shape[1:], dtype=float)
    for i in range(target_shape[0]):
        world = [target_affine[axis, 0]*i+target_affine[axis, 1]*j+target_affine[axis, 2]*k+target_affine[axis, 3] for axis in range(3)]
        coordinates = [sum(inverse[axis, n]*world[n] for n in range(3))+inverse[axis, 3] for axis in range(3)]
        indices = [np.floor(coordinate+.5).astype(int) for coordinate in coordinates]
        valid = np.ones(j.shape, bool)
        for axis in range(3):
            valid &= (indices[axis] >= 0) & (indices[axis] < source.shape[axis])
        output[i][valid] = source[tuple(index[valid] for index in indices)]
    return output


def vessel_audit():
    assert sha(VESSEL) == VESSEL_SHA
    preservation = source_compare(HOST, VESSEL, additional_originals=8, additional_displays=1)
    inference = json.loads((VESSEL_DIR/"inference.json").read_text())
    registered = json.loads((VESSEL_DIR/"registered.json").read_text())
    prepared = json.loads((VESSEL_DIR/"prepared.json").read_text())
    assert inference["status"] == "succeeded" and inference["split"]["role"] == "TRAIN"
    assert inference["split"]["training_admitted"] is False and not inference["training_lineage"]["held_out_accuracy_claim"]
    hashes = {}
    for row in inference["outputs"]+registered["outputs"]:
        path = VESSEL_DIR/Path(row["path"]).name
        assert sha(path) == row["sha256"]
        hashes[path.name] = row["sha256"]
    native_mask_image = nib.load(VESSEL_DIR/"vessels_mask.nii.gz")
    native_mask = np.asarray(native_mask_image.dataobj) > 0
    score_image = nib.load(VESSEL_DIR/"vessels_probability.nii.gz")
    score = score_image.get_fdata(dtype=np.float32)
    assert np.array_equal(native_mask_image.affine, score_image.affine)
    assert np.array_equal(native_mask_image.affine, inference["input"]["affine_ras_mm"])
    assert list(score.shape) == inference["input"]["shape"] == [512, 512, 100]
    assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1
    assert np.array_equal(native_mask, score > inference["parameters"]["threshold"])
    assert int(native_mask.sum()) == inference["native_validation"]["foreground_voxels"] == 185231
    score_range = [float(score.min()), float(score.max())]
    del score
    case, _, workspace = load_workspace(VESSEL)
    assert len(workspace["sources"]) == 3 and not case.compartments
    estimates = [item for item in workspace["sources"] if item.annotation_kind == "estimated"]
    assert len(estimates) == 1
    native = estimates[0].case
    assert set(native.compartments) == {"estimated_vessels"}
    assert np.array_equal(native.compartments["estimated_vessels"], native_mask)
    assert np.array_equal(native.affine, native_mask_image.affine)
    source_reference = workspace["sources"][0].case
    assert np.array_equal(native.mri, source_reference.mri) and np.array_equal(native.affine, source_reference.affine)
    assert set(native.compartments) != set(source_reference.compartments)
    layer = native.metadata["anatomy_layers"]["estimated_vessels"]
    assert layer["provenance"] == "estimated" and layer["policy_use"] == "excluded" and layer["anatomical_review"] == "pending"
    assert layer["source_role"] == "display_only_model_estimate"
    assert layer["inference_sha256"] == sha(VESSEL_DIR/"inference.json") == registered["inference_sha256"]
    expected_weights = "6da4fbd06dd9c3b597e5d50cc628deb3026919ff126d01afbd13ac8fe22ee10c"
    assert layer["model_weights_sha256"] == inference["model"]["weights_sha256"] == inference["identity"]["weights_sha256"] == prepared["pinned"]["weights"]["sha256"] == expected_weights
    assert inference["identity"]["worker_sha256"] == sha(ROOT/"src/resectionlab/poc_models.py")
    assert inference["model"]["revision"] == "dd70bbdd2b9de06b1316b4d1105bda9766c24fd7" and inference["model"]["variant"] == "base"
    assert inference["identity"]["input_sha256"] == inference["input"]["sha256"] == native.source_refs[0].sha256
    registrations = []
    for display in workspace["sources"]:
        _, mapping = source_registration(case, display.case)
        assert mapping["overlayPermitted"] is False
        registrations.append({"modality": display.modality, "annotation_kind": display.annotation_kind,
            "status": mapping["status"], "overlay_permitted": False})
    assert not registered["anatomical_registration_accepted"] and registered["registration_status"] == "estimated_unreviewed"
    native_to_primary = np.asarray(registered["MRA_RAS_mm_to_T1_RAS_mm"])
    primary_to_native = np.asarray(registered["T1_RAS_mm_to_MRA_RAS_mm"])
    assert np.max(abs(native_to_primary@primary_to_native-np.eye(4))) < 1e-12
    coverage_image = nib.load(VESSEL_DIR/"T1_sampling_coverage.nii.gz")
    mapped_image = nib.load(VESSEL_DIR/"vessels_mask_on_T1_estimated.nii.gz")
    coverage, mapped = np.asarray(coverage_image.dataobj) > 0, np.asarray(mapped_image.dataobj) > 0
    assert np.array_equal(mapped_image.affine, case.affine) and np.array_equal(coverage_image.affine, case.affine)
    expected_mapped = independent_itk_nearest(native_mask, native_to_primary@native.affine, case.mri.shape, case.affine)
    excluded = int((expected_mapped & ~coverage).sum())
    expected_mapped &= coverage
    assert np.array_equal(expected_mapped, mapped)
    assert excluded == registered["mapped_prediction_voxels_excluded_by_coverage"] == 375
    assert int(mapped.sum()) == registered["registered_foreground_voxels"] == 24219
    return {"status": "pass", "source_preservation": preservation, "native_geometry": {"shape": list(native_mask.shape),
        "affine_ras_mm": native.affine.tolist(), "mask_voxels": int(native_mask.sum()), "mask_array_hash": array_digest(native_mask)},
        "score_threshold_rule": "saved uncalibrated score >0.5", "threshold_exact": True, "score_range": score_range,
        "native_estimate_separate_from_source_annotations": True, "source_MRA_arrays_exact": True,
        "saved_output_hashes": hashes, "saved_weight_sha256": expected_weights, "weight_file_rehashed_this_audit": False,
        "worker_source_sha256_verified": inference["identity"]["worker_sha256"], "model_revision": inference["model"]["revision"],
        "registrations": registrations, "independent_registered_mask_transfer_exact": True,
        "registered_mask_voxels": int(mapped.sum()), "mapped_cells_excluded_by_recorded_coverage": excluded,
        "anatomical_registration_accepted": False, "training_admitted": False, "held_out_accuracy_claim": False,
        "limitations": ["Native prediction is limited to acquired partial MRA coverage; completeness unknown.",
            "IXI collection occurs in model training lineage; individual subject overlap unknown.",
            "Score is uncalibrated. Research/noncommercial model restrictions remain attached.",
            "Weights hash matched across saved records and earlier audit; no repeated weight-file read here.",
            "Registered T1 mask remains an unreviewed archived candidate, excluded from fusion and primary planning."]}


def vessel_reviewed_audit():
    preservation = source_compare(VESSEL, VESSEL_REVIEWED, reference_originals=24, reference_displays=3)
    with ZipFile(VESSEL_REVIEWED) as archive:
        saved = json.loads(archive.read("workspace.json"))
        manifest = json.loads(archive.read("manifest.json"))
    imaging = saved["imagingState"]
    selected = imaging["selectedSeriesId"]
    source = next(item["source"] for item in saved["images"] if item["source"]["seriesId"] == selected)
    assert source["annotationKind"] == "estimated" and source["modality"] == "MRA"
    assert source["scope"] == "display-only-native-grid"
    state = imaging["states"][selected]
    assert state["visibleLayers"]["estimated_vessels"] is True
    assert state["viewer"]["layerOpacity"]["estimated_vessels"] == .65
    assert "patient_rehearsal" not in manifest["artifacts"]
    return {"status": "pass", "preservation": preservation, "selected_source": source,
        "selected_estimate_visible": True, "saved_estimate_opacity": .65,
        "rehearsal_attached": False, "anatomical_review_status_changed": False,
        "actual_GUI_observation": "Owned by parent; independent file inspection confirms persisted settings only"}


def main():
    inputs = [HOST, DESCENDANT, REVIEWED, VESSEL, VESSEL_REVIEWED, *FROZEN]
    inputs += [VESSEL_DIR/name for name in ("inference.json", "registered.json", "prepared.json", "vessels_mask.nii.gz", "vessels_probability.nii.gz", "native_sampling_coverage.nii.gz", "T1_sampling_coverage.nii.gz", "vessels_mask_on_T1_estimated.nii.gz", "vessels_score_on_T1_estimated.nii.gz")]
    before = {str(path.relative_to(ROOT)): stamp(path) for path in inputs}
    assert sha(HOST) == HOST_SHA and sha(DESCENDANT) == DESCENDANT_SHA
    rehearsal = rehearsal_audit()
    print("GUI rehearsal pass", flush=True)
    vessel = vessel_audit()
    print("Vessel artifact pass", flush=True)
    vessel_reviewed = vessel_reviewed_audit()
    print("Vessel GUI save pass", flush=True)
    assert before == {str(path.relative_to(ROOT)): stamp(path) for path in inputs}
    report = {"schema": "independent-final-ixi-gui-vessel-audit-v1", "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_stamps": before, "gui_rehearsal": rehearsal, "vascular_candidate": vessel, "vascular_GUI_save": vessel_reviewed,
        "auditor_adjustments": ["Initial vessel pre-coverage count oracle used scipy continuous-center boundaries; corrected to the actual declared ITK half-voxel extent. Covered output mask was exact in both checks; no product change was needed."],
        "all_inputs_and_frozen_audits_unchanged": True, "no_model_or_FK_solver_executed": True,
        "scope": "Fresh saved-artifact inspection; GUI actions observed separately by parent; no clinical or anatomical validation",
        "auditor_sha256": sha(Path(__file__)), "rehearsal_runtime_sha256": sha(ROOT/"src/resectionlab/rehearsal_episode.py")}
    text = json.dumps(report, indent=2)+"\n"
    assert "/Users/" not in text and "file://" not in text
    output = ROOT/"artifacts/ixi265-virtual-pilot-v1/gui-and-vessel-independent-qa.json"
    output.write_text(text)
    print(json.dumps({"status": "pass", "artifact": str(output.relative_to(ROOT)), "sha256": sha(output)}))


if __name__ == "__main__":
    main()
