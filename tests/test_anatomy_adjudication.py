import numpy as np
import pytest

from resectionlab.anatomy_adjudication import BoneEvidence, acquisition_contributions
from resectionlab.anatomy_import import review_registration
from resectionlab.anatomy_sections import section
from resectionlab.patient_anatomy import PatientAnatomySnapshot, world
from test_patient_anatomy import patient
from test_anatomy_siam_display import study


def test_ct_precedence_is_local_source_bound_and_reversible(patient, tmp_path):
    c = patient
    ct = c.add_series(tmp_path / "mri.nii.gz", series_id="ct", modality="CT")
    ct["intensity_units"] = "HU"
    a = np.asarray(ct["affine"])
    shape = ct["data"]["shape"]
    mask = np.zeros(shape, np.uint8)
    mask[4, 4, 4] = 1
    ct_row = c.add_layer(mask, a, layer_id="ct-bone", source_series="ct", kind="bone",
                         role="context", color="#eeeedd", provenance={"method": "native CT 300 HU boundary proposal"})
    ct_row["coverage_fields"] = {"inference_bounds_voxel_inclusive": [[0, 0, 0], [8, 8, 8]]}
    mr_row = c.add_layer(np.ones(shape, np.uint8), a, layer_id="mr-bone", source_series="mri",
                         kind="bone", role="context", color="#eeeeee", provenance={"model": "supersynth"})
    point = world(a, [6, 6, 6])
    c.set_registration("ct", np.eye(4), status="unreviewed", method="test")
    c.save()
    unreviewed = PatientAnatomySnapshot.capture(c).sample(point)
    assert unreviewed["boneAdjudication"] is None
    assert not unreviewed["layers"]["ct-bone"]["known"]
    c.set_registration("ct", np.eye(4), status="source-provided", method="fixture transform")
    c.save()
    snap = PatientAnatomySnapshot.capture(c)
    sample = snap.sample(point)
    decision = sample["boneAdjudication"]
    assert decision["disagreement"]
    assert decision["preferred"] == [{"seriesId": "ct", "layerId": "ct-bone", "model": "native CT 300 HU boundary proposal", "positiveLabel": False}]
    assert decision["alternatives"][0]["positiveLabel"] and sample["layers"]["mr-bone"]["occupied"]
    assert snap.sample(world(a, [10, 10, 10]))["boneAdjudication"] is None
    bools = lambda values: np.array(values, bool)
    field = BoneEvidence(c, [mr_row, ct_row], (3,))
    assert field.order(ct_row) < field.order(mr_row)
    field.apply(ct_row, bools([1, 0, 0]), bools([1, 1, 0]))
    original = bools([0, 1, 1])
    occupied, coverage, audit = field.apply(mr_row, original, bools([1, 1, 1]))
    assert occupied.tolist() == coverage.tolist() == [False, False, True]
    assert audit["disagreementVoxels"] == 2
    assert original.tolist() == [False, True, True]


def test_landmark_review_uses_inverse_source_mapping_and_rejects_outside_support(patient, tmp_path):
    c = patient
    ct = c.add_series(tmp_path / "mri.nii.gz", series_id="ct", modality="CT")
    estimate = np.eye(4)
    estimate[:3, 3] = [1.25, -.5, .75]
    c.set_registration("ct", estimate, status="unreviewed", method="fixture estimate")
    a = np.asarray(ct["affine"])
    m = world(a, [[4, 4, 4], [8, 4, 4], [4, 8, 4]])
    r = world(estimate, m)
    result = section(c, series_id="ct", center=r[0], normal=[0, 0, 1], size=19, registration_review=True)
    assert result["frame"] == "registration preview"
    np.testing.assert_allclose(world(np.asarray(result["sourceFromSection"]), result["center"]), m[0])
    np.testing.assert_allclose(world(np.asarray(result["voxelFromSection"]), result["center"]), [4, 4, 4])
    with pytest.raises(ValueError, match="outside acquired source coverage"):
        review_registration(c, "ct", m + 100, r + 100, "fixture reviewer")
    assert c.document["transforms"]["ct"]["status"] == "unreviewed"
    assert review_registration(c, "ct", m, r, "fixture reviewer") == pytest.approx([0, 0, 0])


def test_contributions_include_unregistered_and_archived_acquisitions(patient, tmp_path):
    c = patient
    mri = c.series("mri")
    ct = c.add_series(tmp_path / "mri.nii.gz", series_id="ct", modality="CT")
    c.set_registration("ct", np.eye(4), status="unreviewed", method="fixture estimate")
    ct_bone = c.add_layer(np.ones(ct["data"]["shape"], np.uint8), ct["affine"], layer_id="ct-bone", kind="bone",
                          source_series="ct", color="#eeddee", role="context")
    ct_bone["provenance"]["inputs"] = {"channels": [{"series_id": "mri", "original_sha256": mri["original"]["sha256"]}]}
    c.document["source_assets"] = [{"id": "raw", "modality": "DWI", "storage": "source-asset",
                                   "original": {"sha256": "0"*64}},
                                  {"name": "old-diffusion", "kind": "diffusion_4d", "storage": "external"}]
    rows = acquisition_contributions(c, [{"layerId": "tumor", "target": [1, 1, 1]}],
                                     {"layers": [{"layerId": "ct-bone", "status": "estimated_alignment"}]})
    scans = {r["seriesId"]: r for r in rows}
    assert scans["ct"]["disposition"] == "inspection context"
    assert scans["mri"]["disposition"] == "quantitative geometry" and scans["mri"]["modelInputs"] == ["ct-bone"]
    assert scans["raw"]["disposition"] == scans["old-diffusion"]["disposition"] == "unsupported"


def test_siam_bone_is_retained_for_raw_qc_but_yields_to_ct_where_covered(study, tmp_path):
    c = study
    scan = c.series("scan")
    ct = c.add_series(tmp_path / "scan.nii", series_id="ct", modality="CT")
    ct["intensity_units"] = "HU"
    c.set_registration("ct", np.eye(4), status="source-provided", method="fixture source frame")
    a = np.asarray(scan["affine"])
    ct_mask = np.zeros(scan["data"]["shape"], np.uint8)
    ct_mask[32, 31, 10] = 1
    c.add_layer(ct_mask, a, layer_id="dense-bone", source_series="ct", kind="skull", role="context",
                color="#aaaaaa", provenance={"model": "totalsegmentator-skull",
                                             "parameters": {"positive_ct_bone_gate_hu": 250}})
    c.save()
    snap = PatientAnatomySnapshot.capture(c)
    inside = snap.sample(world(a, [32, 31, 37]))
    assert inside["boneAdjudication"]["disagreement"]
    assert inside["boneAdjudication"]["alternatives"][0]["layerId"] == "siam-15"
    outside_head = snap.sample(world(a, [4, 4, 15]))
    assert outside_head["layers"]["siam-15"]["occupied"]
    assert not outside_head["boneAdjudication"]["alternatives"][0]["positiveLabel"]
    assert not outside_head["boneAdjudication"]["disagreement"]


def test_disagreeing_vessel_labels_retain_both_readings_without_negative_clearance(patient):
    c = patient
    mask = np.zeros(c.layer("vessels")["data"]["shape"], np.uint8)
    mask[2, 2, 2] = 1
    a = c.patient_affine(c.series("mri"))
    c.add_layer(mask, a, layer_id="mr-vessel-estimate", source_series="mri", kind="vessel", role="context",
                color="#aaaaaa", provenance={"model": "experimental", "evidence_use": "context_only"})
    c.save()
    snap = PatientAnatomySnapshot.capture(c)
    point = world(a, [2, 2, 2])
    assert not snap.distance("mr-vessel-estimate", point)["available"]
    result = snap.sample(point)
    diff, = result["structureDisagreements"]
    assert diff["structure"] == "vessels"
    assert {r["layerId"]: r["positiveLabel"] for r in diff["sources"]} == {"vessels": False, "mr-vessel-estimate": True}
    assert "absence" in diff["meaning"]


def test_recalculating_one_alignment_preserves_unrelated_review(patient, tmp_path, monkeypatch):
    from copy import deepcopy
    from resectionlab.anatomy_alignment import ensure_alignment

    c = patient
    for sid, modality in [("flair", "FLAIR"), ("ct", "CT")]:
        c.add_series(tmp_path / "mri.nii.gz", series_id=sid, modality=modality)
        c.set_registration(sid, np.eye(4), status="reviewed", method="saved landmark review")
    other = deepcopy(c.document["transforms"]["flair"])
    calls = []
    def estimate(*args):
        calls.append(True)
        return np.eye(4), {"status": "consistent", "warnings": [], "metrics": {}}
    monkeypatch.setattr("resectionlab.anatomy_alignment.estimate_pair", estimate)
    assert ensure_alignment(c, force=True, series_id="ct")
    assert calls == [True]
    assert c.document["transforms"]["ct"]["status"] == "unreviewed"
    assert c.document["transforms"]["flair"] == other


def test_composite_display_uses_hidden_ct_and_raw_inspection_preserves_mri(study, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request

    c = study
    source = c.series("scan")
    ct = c.add_series(tmp_path / "scan.nii", series_id="ct", modality="CT")
    ct["intensity_units"] = "HU"
    c.set_registration("ct", np.eye(4), status="source-provided", method="fixture")
    mask = np.zeros(source["data"]["shape"], np.uint8)
    mask[32, 31, 12] = 1
    c.add_layer(mask, source["affine"], source_series="ct", kind="bone", role="context", layer_id="density",
                color="#eeeeee", provenance={"method": "native CT 300 HU boundary proposal"})
    c.save()
    b = BridgeSession(tmp_path / "transfers")
    def call(op, args):
        return b.execute(op, args, _Request(op), lambda *_: None)
    try:
        bound = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
        args = {"collectionId": bound["collectionId"], "snapshotId": c.identity(), "layers": ["siam-15"],
                "seriesId": "scan", "patientModel": True}
        fused, = call("anatomyView", args)["compartments"]
        assert fused["boneAdjudication"]["preferredLayerIds"] == ["density"]
        assert fused["boneAdjudication"]["alternativeOnlyVoxels"] == 1
        assert fused["surface"]["empty"]
        raw, = call("anatomyView", {**args, "original": True})["compartments"]
        assert "boneAdjudication" not in raw and "empty" not in raw["surface"]
        assert raw["volumeMm3"] == fused["volumeMm3"]
    finally:
        b.close()


def test_new_acquisition_retires_previous_snapshot_removal_preview(patient, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request

    c = patient
    b = BridgeSession(tmp_path / "transfers")
    def call(op, args):
        return b.execute(op, args, _Request(op), lambda *_: None)
    try:
        bound = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
        args = {"collectionId": bound["collectionId"], "snapshotId": c.identity()}
        target = {"seriesId": "mri", "layers": ["tumor"]}
        original, = call("anatomyView", {**args, **target})["compartments"]
        centre = world(c.patient_affine(c.series("mri")), [12, 12, 12]).tolist()
        assert call("anatomyRemoval", {**args, "point": centre, "radiusMm": 2})["preview"]
        preview, = call("anatomyView", {**args, **target})["compartments"]
        assert preview["volumeMm3"] < original["volumeMm3"]
        updated = call("anatomyImport", {**args, "kind": "nifti", "path": str(tmp_path / "mri.nii.gz"),
                                         "modality": "FLAIR", "patientId": c.document["patient_id"]})
        revised = updated["document"]["snapshot_id"]
        assert revised != args["snapshotId"]
        restored, = call("anatomyView", {**args, **target, "snapshotId": revised})["compartments"]
        assert restored["volumeMm3"] == original["volumeMm3"]
    finally:
        b.close()


def test_acquisition_baseline_is_portable_and_rotates_only_on_new_acquisitions(patient, tmp_path):
    from resectionlab.anatomy_package import open_package, save_package
    from resectionlab.patient_anatomy import PatientImagingCollection

    c = patient
    old = c.identity()
    c.add_series(tmp_path / "mri.nii.gz", series_id="second", modality="T2")
    c.save("Add a second scan")
    before = c.document["before_acquisition"]
    assert before["snapshotId"] == old
    assert [r["id"] for r in before["sources"]] == ["mri"]
    previous = next(r for r in before["layers"] if r["id"] == "tumor")
    assert previous["arraySha256"] == c.layer("tumor")["data"]["sha256"]
    c.set_registration("second", np.eye(4), status="reviewed", method="fixture")
    c.save()
    archive = tmp_path / "two.rsworkspace"
    save_package(c, archive)
    reopened = PatientImagingCollection(open_package(archive, tmp_path / "reopen").parent)
    assert reopened.document["before_acquisition"] == before


def test_fresh_and_stale_multimodal_dependencies_are_separate_from_source_labels(patient):
    c = patient
    source = c.series("mri")
    tumor = c.layer("tumor")
    channel = {"series_id": "mri", "original_sha256": source["original"]["sha256"], "array_sha256": source["data"]["sha256"]}
    tumor["provenance"]["inputs"] = {"channels": [channel]}
    targets = [{"layerId": "tumor", "target": [1, 1, 1]}]
    row, = acquisition_contributions(c, targets, {"layers": []})
    assert row["modelInputs"] == [tumor["label"]]
    channel["array_sha256"] = "a"*64
    row, = acquisition_contributions(c, targets, {"layers": []})
    link = next(r for r in row["structures"] if r["layerId"] == "tumor")
    assert link == {"layerId": "tumor", "anchor": True, "channel": True, "staleInputs": ["mri: array_sha256 changed"]}
    assert not row["modelInputs"] and not row["uses"]


def test_single_layer_pick_reports_exact_ijk_and_independent_frame_restriction(patient, tmp_path):
    c = patient
    r = c.add_series(tmp_path / "mri.nii.gz", series_id="independent", modality="CT")
    c.add_layer(np.ones(r["data"]["shape"], np.uint8), r["affine"], layer_id="independent-skull",
                source_series=r["id"], kind="skull", color="#ffffff", role="context")
    c.save()
    snap = PatientAnatomySnapshot.capture(c)
    p = world(np.array(r["affine"]), [3, 5, 6])
    def reading(frame):
        return snap.sample(p, layer_id="independent-skull", independent_series=frame)["layers"]["independent-skull"]
    assert reading(None)["reason"] == "registration_unreviewed"
    assert "another source" in reading("mri")["detail"]
    assert reading("independent")["sourceVoxel"] == [3, 5, 6] and reading("independent")["occupied"]


def test_distinct_named_arteries_do_not_disagree_on_complementary_voxels(patient):
    from resectionlab.anatomy_adjudication import structure_disagreements
    c = patient
    r = c.series("mri")
    data = np.ones(r["data"]["shape"], np.uint8)
    for name in ["BA", "R-Pcom"]:
        c.add_layer(data, r["affine"], layer_id=name, source_series="mri", kind="artery", color="#ffffff",
                    role="context", provenance={"ontology_class": name})
    readings = {row["id"]: {"known": True, "occupied": row["id"] == "BA"} for row in c.document["layers"]}
    assert not structure_disagreements(c, readings)


def test_section_contours_and_cursor_respect_the_same_declared_slices(patient):
    c = patient
    grid = c.patient_affine(c.series("mri"))
    shape = c.series("mri")["data"]["shape"]
    mask = np.zeros(shape, np.uint8)
    mask[4:8, 4:8, 5:7] = 1
    row = c.add_layer(mask, grid, layer_id="limited", kind="artery", source_series="mri",
                      color="#ff0000", role="context")
    row["coverage_fields"] = {"supported_axial_indices": [5]}
    c.save()
    for z, covered in [(5, True), (6, False)]:
        point = world(grid, [5, 5, z])
        args = {"center": point, "normal": grid[:3, 2], "size": 32, "spacing_mm": .5, "series_id": "mri"}
        plain = section(c, **args)["pixels"]
        contour = section(c, **args, layers=["limited"])["pixels"]
        assert (not np.array_equal(plain, contour)) == covered
        reading = PatientAnatomySnapshot.capture(c).sample(point, layer_id="limited")["layers"]["limited"]
        assert reading["known"] == covered and reading["sourceVoxel"] == [5, 5, z]
