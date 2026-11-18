"""Native route-local display geometry, binding, source coverage and transfer."""

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_approaches import dispatch_plan
from resectionlab.anatomy_operative_field import operative_field
from resectionlab.anatomy_planning import native_target_binding
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientImagingCollection, world


@pytest.fixture
def study(tmp_path):
    shape = (30, 32, 38)
    affine = np.array([[0, -1, 0, 12], [1, 0, 0, -9], [0, 0, 1, 7], [0, 0, 0, 1]], float)
    image = nib.Nifti1Image(np.ones(shape, np.float32), affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "native.nii.gz"
    nib.save(image, path)
    c = PatientImagingCollection.create(tmp_path / "patient", "fixture")
    c.add_series(path, series_id="reference", modality="T1")
    c.add_series(path, series_id="moving", modality="CT")
    c.set_registration("moving", np.eye(4), status="unreviewed", method="fixture")

    def add(name, kind, data, **kwargs):
        return c.add_layer(data.astype(np.uint8), affine, layer_id=name, source_series=kwargs.pop("source", "reference"),
                           role=kwargs.pop("role", "context"), kind=kind, color="#ffccff", **kwargs)

    x, y, z = np.indices(shape)
    target_label = ((x-14)**2+(y-16)**2+(z-10)**2) <= 9
    target_label |= ((x-14)**2+(y-16)**2+(z-19)**2) <= 4
    target_label |= ((x-14)**2+(y-16)**2+(z-34)**2) <= 4
    add("lesion", "tumor", target_label, role="target")
    add("other-lesion", "tumor", z > 26, role="target")
    add("brain", "brain", (z >= 4) & (z <= 24))
    add("bone-outer", "skull", z == 29)
    add("bone-inner", "bone", z == 27)
    add("direct", "dura", z == 25, provenance={"kind": "supplied_annotation"})
    add("proxy", "dura", z == 31, provenance={"model": "dural-envelope"})
    add("outside-field", "bone", z == 34, source="moving")
    add("artery", "artery", (z == 17) & (y == 16), role="critical")
    add("siam-vascular", "vessels", (z == 23) & (y == 16), provenance={"model": "siam-v2"})
    add("mri-vein", "vein", (z == 30) & (y == 16), provenance={"model": "SuperSynth"})
    vein = add("vein", "vein", (z == 19) & (y == 15), role="critical")
    vein["coverage_fields"] = {"inference_bounds_voxel_inclusive": [[0, 0, 0], [29, 31, 18]]}
    slices = add("dura-slices", "dura", (z == 35) | (z == 22))
    slices["coverage_fields"] = {"supported_axial_indices": [22]}
    c.save()
    target = "lesion"
    binding = native_target_binding(c, c.layer(target))
    dispatch_plan(c, {"action": "selectTarget", "targetLayerId": target, "candidateBinding": binding})
    route = dispatch_plan(c, {"action": "upsert", "approach": {
        "entry": world(affine, [14, 16, 23]).tolist(), "target": world(affine, [14, 16, 10]).tolist(),
        "radiusMm": 1, "targetLayerId": target}})["approaches"][0]
    return c, dict(action="operativeField", snapshotId=c.identity(), approachId=route["id"],
                   parameterHash=route["parameterHash"], targetLayerId=target)


def test_native_bits_stations_order_and_transfer(study, tmp_path):
    c, args = study
    b = BridgeSession(tmp_path / "transfer")

    def call(op, args):
        return b.execute(op, args, _Request("test"), lambda *unused: None)

    try:
        opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
        field = call("anatomyPlan", {**args, "collectionId": opened["collectionId"]})
        labels = field["labels"]
        raw = Path(labels["path"]).read_bytes()
        assert labels["path"] in b.anatomy_transfer_paths
        assert hashlib.sha256(raw).hexdigest() == labels["sha256"]
        assert labels["dtype"] == "uint8" and labels["order"] == "C" and labels["shape"] == field["shape"]
        bits = np.frombuffer(raw, dtype=np.uint8).reshape(field["shape"])
        spacing, origin = field["spacingMm"], np.array(field["origin"])
        center = np.rint(-origin / spacing).astype(int)
        assert origin + spacing * center == pytest.approx([0, 0, 0])
        assert bits[tuple(center)] == 4 | 8
        assert field["basisRas"] == np.eye(3).tolist()
        assert field["targetRasMm"] == world(c.patient_affine(c.series("reference")), [14, 16, 10]).tolist()
        assert field["stations"] == pytest.approx({"bone": 19, "dura": 15, "brain": 14, "tumor": 3}, abs=.5)
        assert field["entryDepthMm"] == 13 and len(field["stationRuns"]["bone"]) == 2
        assert any("Multiple bone" in issue for issue in field["stationIssues"])
        assert any(row["status"] == "contact" and row["id"] == "artery" for row in field["encounters"])
        assert any(row["status"] == "unknown" and row["id"] == "siam-vascular" for row in field["encounters"])
        assert field["blockers"] and field["unknowns"] and field["simulationExecutionAuthorized"] is False
        assert {row["id"] for row in field["excluded"]} == {"outside-field"}
        assert field["stationEvidence"]["dura"] == "source" and field["stationLayerIds"]["dura"] == ["direct"]
        layers = {r["id"]: r for r in field["layers"]}
        assert "other-lesion" not in layers and layers["vein"]["sampledVoxels"] == 0
        assert layers["siam-vascular"]["vascularContextOnly"] and layers["siam-vascular"]["sampledVoxels"] > 0
        assert layers["mri-vein"]["vascularContextOnly"] and any("mri-vein" in issue for issue in field["stationIssues"])
        assert layers["dura-slices"]["centerlineRunsMm"][0][1] < 13
        assert layers["direct"]["evidence"] == "Supplied annotation"
        assert layers["bone-inner"]["dataSha256"] == c.layer("bone-inner")["data"]["sha256"]
    finally:
        b.close()


def test_selection_staleness_and_cancellation(study):
    c, args = study
    for key in ("snapshotId", "approachId", "parameterHash", "targetLayerId", "candidateBinding"):
        with pytest.raises(ValueError):
            operative_field(c, {**args, key: "invalid"})

    def cancel():
        raise InterruptedError("cancel")

    with pytest.raises(InterruptedError, match="cancel"):
        operative_field(c, args, check=cancel)
    c.layer("brain")["revision"] += 1
    c.save()
    with pytest.raises(ValueError, match="currently assessed"):
        operative_field(c, {**args, "snapshotId": c.identity()})


@pytest.mark.parametrize(("endpoint_z", "entry_z", "expected_depth"), [
    (10, 23, 3.125),    # Other foci lie along the route and beyond the entry.
    (10, 11.1, .625),   # Entry is inside the selected focus, between samples.
    (5, 23, None),      # Several intervals lie on the route, none at its endpoint.
])
def test_aspiration_binds_endpoint_interval(study, endpoint_z, entry_z, expected_depth):
    c, args = study
    a = c.patient_affine(c.series("reference"))
    route = dispatch_plan(c, {"action": "upsert", "approach": {
        "id": args["approachId"], "entry": world(a, [14, 16, entry_z]).tolist(),
        "target": world(a, [14, 16, endpoint_z]).tolist(), "radiusMm": 1, "targetLayerId": "lesion"}})["approaches"][0]
    field = operative_field(c, {**args, "parameterHash": route["parameterHash"]})
    assert len(field["stationRuns"]["tumor"]) == 3
    if expected_depth is None:
        assert field["stations"]["tumor"] is None and field["stationLayerIds"]["tumor"] == []
        assert any("endpoint lacks supported selected-target" in issue for issue in field["stationIssues"])
    else:
        assert field["stations"]["tumor"] == pytest.approx(expected_depth)
        assert field["stationLayerIds"]["tumor"] == ["lesion"] and expected_depth <= field["entryDepthMm"]


def test_target_without_slice_support_at_endpoint_has_no_aspiration(study):
    c, args = study
    c.layer("lesion")["coverage_fields"] = {"supported_axial_indices": [11, 12, 13, 18, 19]}
    c.save()
    dispatch_plan(c, {"action": "assess", "approachId": args["approachId"]})
    field = operative_field(c, {**args, "snapshotId": c.identity()})
    assert field["stationRuns"]["tumor"]
    assert field["stations"]["tumor"] is None and field["stationEvidence"]["tumor"] is None
    assert any("endpoint lacks supported selected-target" in issue for issue in field["stationIssues"])


def test_empty_and_off_axis_stations_never_generate_geometry(study):
    c, args = study
    direct = c.layer("direct")
    mask = np.zeros(direct["data"]["shape"], np.uint8)
    mask[3, 3, 25] = 1
    direct["data"] = c.put_array(mask)
    c.layer("proxy")["availability"] = "empty_output"
    c.layer("dura-slices")["coverage_fields"] = {"supported_axial_indices": []}
    c.save()
    dispatch_plan(c, {"action": "assess", "approachId": args["approachId"]})
    result = operative_field(c, {**args, "snapshotId": c.identity()})
    assert result["stations"]["dura"] is None
    row = next(r for r in result["layers"] if r["id"] == "direct")
    assert row["sampledVoxels"] > 0 and not row["centerlineRunsMm"]
    assert any(r["id"] == "proxy" and "empty" in r["reason"] for r in result["excluded"])
    c.layer("proxy")["availability"] = "available"
    c.save()
    dispatch_plan(c, {"action": "assess", "approachId": args["approachId"]})
    result = operative_field(c, {**args, "snapshotId": c.identity()})
    assert result["stationEvidence"]["dura"] == "proxy" and result["stationLayerIds"]["dura"] == ["proxy"]
    assert result["stations"]["dura"] == pytest.approx(21., abs=.5)


def test_registered_ct_skull_beyond_reference_and_proxy_coverage(study, tmp_path):
    c, args = study
    a = c.patient_affine(c.series("reference")).copy()
    a[2, 3] -= 5.
    shape = (30, 32, 90)
    image = nib.Nifti1Image(np.ones(shape, np.float32), a)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "whole-head.nii.gz"
    nib.save(image, path)
    c.add_series(path, series_id="ct", modality="CT")
    transform = np.eye(4)
    transform[2, 3] = 8
    c.set_registration("ct", transform, status="reviewed", method="fixture known translation")
    # CT voxel 77 maps to reference z=80, outside both the cropped MRI and
    # the minimum 32 mm display extension beyond the saved entry (z=23).
    mask = np.zeros(shape, np.uint8)
    mask[:, :, 77] = 1
    c.add_layer(mask, a, source_series="ct", layer_id="ct-bone", kind="skull", role="context", color="#abcdef")
    direct = c.layer("direct")
    direct["coverage_fields"] = {"supported_axial_indices": [25]}
    # The proxy's supported z=31 lies outside direct coverage, so it is
    # retained separately. Its outer station lies above the cropped MRI bone.
    c.save()
    dispatch_plan(c, {"action": "assess", "approachId": args["approachId"]})
    field = operative_field(c, {**args, "snapshotId": c.identity()})
    layers = {r["id"]: r for r in field["layers"]}
    assert layers["ct-bone"]["sourceSeries"] == "ct"
    assert field["stations"]["bone"] == pytest.approx(70., abs=.5)
    assert layers["proxy"]["proxy"] and layers["proxy"]["centerlineRunsMm"]
    assert field["stations"]["dura"] == pytest.approx(15., abs=.5)
    assert {r["id"] for r in field["excluded"]} == {"outside-field"}


def test_reviewed_projection_cannot_extend_field_or_supply_bone(study, tmp_path):
    c, args = study
    before = operative_field(c, args)
    a = c.patient_affine(c.series("reference"))
    shape = (30, 32, 180)
    image = nib.Nifti1Image(np.ones(shape, np.float32), a)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "angiogram.nii.gz"
    nib.save(image, path)
    scan = c.add_series(path, series_id="reformatted", modality="MRA")
    scan["metadata"] = {"ImageType": ["DERIVED", "SECONDARY", "MIP"]}
    c.set_registration("reformatted", np.eye(4), status="reviewed", method="fixture")
    mask = np.zeros(shape, np.uint8)
    mask[:, :, 120] = 1
    c.add_layer(mask, a, source_series="reformatted", layer_id="projected-bone", kind="skull",
                role="context", color="#abcdef")
    c.save()
    dispatch_plan(c, {"action": "assess", "approachId": args["approachId"]})
    field = operative_field(c, {**args, "snapshotId": c.identity()})
    assert field["stations"] == before["stations"] and field["shape"] == before["shape"]
    assert field["spacingMm"] == before["spacingMm"]
    assert any(row["id"] == "projected-bone" for row in field["excluded"])
