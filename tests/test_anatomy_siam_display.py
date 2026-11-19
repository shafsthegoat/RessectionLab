"""Native and restored SIAM display supports retain source labels and dark interior tissue."""

from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_siam_display as display
from resectionlab.anatomy_package import open_package, save_package
from resectionlab.anatomy_sections import section
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientImagingCollection, world


@pytest.fixture
def study(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "native", "display-test")
    coords = np.indices((64, 64, 64))
    distance2 = np.sum((coords - 32) ** 2, axis=0)
    # A hollow bright head with a separate signal island from the scanner.
    intensity = ((distance2 < 18**2) & (distance2 > 10**2)).astype(np.float32) * 120
    intensity[3:7, 3:7, 3:7] = 160
    matrix = np.array([[0, -1., 0, 80], [0, 0, 1., 18], [1., 0, 0, -30], [0, 0, 0, 1.]])
    image = nib.Nifti1Image(intensity, matrix)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "scan.nii"
    nib.save(image, path)
    source = c.add_series(path, series_id="scan", modality="T1")
    labels = np.zeros(intensity.shape, np.uint8)
    for i in range(1, 17):
        labels[32, 31, 22+i] = i
        labels[4, 4, i] = i
    for i, kind in ((1, "brain-region"), (13, "dura"), (14, "vessels"), (15, "skull"), (16, "review-region")):
        c.add_layer(labels, matrix, layer_id="siam-"+str(i), kind=kind, source_series="scan",
                    role="context", color="#ff00ff", label_value=i, provenance={
                        "model": "siam-v2", "source": source["original"]["sha256"],
                        "source_array_sha256": source["data"]["sha256"], "label_ontology": {"value": i},
                    })
    marker = np.zeros_like(labels)
    marker[4, 4, 15] = 1
    c.add_layer(marker, matrix, layer_id="supplied", source_series="scan", kind="tumor",
                role="target", color="#ffffff", provenance={"kind": "supplied_annotation"})
    c.save()
    return c


def test_shared_support_all_codes_dark_interior_saved_package_and_qc(study, tmp_path, monkeypatch):
    c = study
    original = c.layer("siam-15")
    source = c.series("scan")
    layer, report = display.supported_tissue_layer(c, original)
    assert (report["status"], report["sourceVoxels"], report["retainedVoxels"], report["hiddenVoxels"]) == (
        "supported", 2, 1, 1)
    mask = display.source_head_support(c, source)
    assert mask[32, 31, 36] and not mask[4, 4, 4]
    assert c.array(source["data"])[32, 31, 36] == 0
    labels = c.array(layer["data"])
    np.testing.assert_array_equal(np.unique(labels, return_counts=True)[1][1:], np.ones(16))
    assert np.array_equal(labels, np.where(mask, c.array(original["data"]), 0))
    assert c.layer("siam-16")["data"] == original["data"] != layer["data"]
    marker = c.layer("supplied")
    assert display.supported_tissue_layer(c, marker) == (marker, None) and c.mask(marker)[4, 4, 15]

    package = tmp_path / "study.rsworkspace"
    save_package(c, package)
    with ZipFile(package) as archive:
        assert original["data"]["path"] in archive.namelist() and layer["data"]["path"] not in archive.namelist()
        assert not any(name.startswith("display-cache") for name in archive.namelist())
    cold = PatientImagingCollection(open_package(package, tmp_path / "restore").parent)
    assert display.supported_tissue_layer(cold, cold.layer("siam-15"))[1] == report

    original_build = display._build
    def unexpected(*args):
        pytest.fail("cache hit recomputed source support")
    monkeypatch.setattr(display, "_build", unexpected)
    reopened = PatientImagingCollection(cold.root)
    assert display.supported_tissue_layer(reopened, reopened.layer("siam-16"))[0]["data"] == layer["data"]
    raw, raw_report = display.supported_tissue_layer(reopened, reopened.layer("siam-15"), raw_qc=True)
    assert raw["data"] == original["data"] and raw_report["hiddenVoxels"] == 0
    assert raw_report["sampledDataSha256"] == report["rawDataSha256"]
    edited = {**raw, "edits": [{"author": "fixture"}], "data": marker["data"], "label_value": 1,
              "original_data": raw["data"], "original_label_value": 15}
    assert display.supported_tissue_layer(reopened, edited) == (edited, None)
    assert display.supported_tissue_layer(reopened, edited, raw_qc=True)[0]["data"] == original["data"]
    bad = {**original, "provenance": {**original["provenance"], "source_array_sha256": "0"*64}}
    with pytest.raises(ValueError, match="original MRI"):
        display.supported_tissue_layer(reopened, bad)
    monkeypatch.setattr(display, "_build", original_build)
    # A missing derived asset can be rebuilt. A changed acquired MRI cannot reuse the receipt.
    reopened.path(layer["data"]["path"]).unlink()
    assert display.supported_tissue_layer(reopened, reopened.layer("siam-15"))[1] == report
    with reopened.path(source["data"]["path"]).open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        display.supported_tissue_layer(reopened, reopened.layer("siam-15"))


def test_native_view_mesh_registered_slices_and_oblique_use_supported_labels(study, tmp_path):
    c = study
    bridge = BridgeSession(tmp_path / "transfers")
    def call(op, args):
        return bridge.execute(op, args, _Request("siam-display"), lambda *_: None)
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    args = {"collectionId": opened["collectionId"], "snapshotId": c.identity(), "seriesId": "scan",
            "layers": ["siam-1", "siam-15"]}
    def get_array(d):
        return np.fromfile(d["path"], dtype=d["dtype"]).reshape(d["shape"])
    for raw in (False, True, False):
        view = call("anatomyView", {**args, "original": raw})
        cortical, skull = view["compartments"]
        categorical = get_array(view["labelSets"][0]["data"])
        assert categorical[32, 31, 23] == 1 and bool(categorical[4, 4, 1]) == raw
        mask = get_array(skull["mask"])
        assert mask[32, 31, 37] == 1 and bool(mask[4, 4, 15]) == raw
        assert float(get_array(skull["surface"]["source"])[:, 0].min()) == pytest.approx(3.5 if raw else 31.5)
        assert cortical["sourceSupport"]["status"] == ("unfiltered" if raw else "supported")
        assert get_array(view["mri"])[4, 4, 4] == 160
    centre = world(np.array(c.series("scan")["affine"]), [4, 4, 15])
    args = {"series_id": "scan", "center": centre, "normal": [0, 0, 1], "spacing_mm": 1, "size": 17}
    plain = section(c, **args)["pixels"]
    assert np.array_equal(section(c, layers=["siam-15"], **args)["pixels"], plain)
    assert not np.array_equal(section(c, layers=["siam-15"], original=True, **args)["pixels"], plain)


def test_insufficient_source_support_retains_raw_and_explains_unavailability(study):
    c = study
    s = c.series("scan")
    s["data"] = c.put_array(np.zeros(s["data"]["shape"], np.float32))
    row = c.layer("siam-1")
    row["provenance"]["source_array_sha256"] = s["data"]["sha256"]
    displayed, report = display.supported_tissue_layer(c, row)
    assert displayed["data"] == row["data"] and report["status"] == "unavailable"
    assert "insufficient intensity" in report["reason"]
    assert display.source_head_support(c, s) is None


def test_reviewed_component_removal_keeps_source_support_after_reopen(study, tmp_path):
    c = study
    row = c.layer("siam-15")
    original = c.mask(row)
    corrected = original.copy()
    corrected[32, 31, 37] = False
    operations = [{"operation": "remove-components", "componentIds": [2]}, {"operation": "undo"}, {"operation": "redo"}]
    c.commit_edit(row["id"], corrected, author="fixture", expected=c.identity(), operations=operations)
    package = tmp_path / "reviewed.rsworkspace"
    save_package(c, package)
    reopened = PatientImagingCollection(open_package(package, tmp_path / "copies").parent)
    row = reopened.layer("siam-15")
    layer, report = display.supported_tissue_layer(reopened, row)
    assert (report["sourceVoxels"], report["retainedVoxels"], report["hiddenVoxels"]) == (1, 0, 1)
    assert np.array_equal(reopened.mask(row), corrected) and not reopened.mask(layer).any()
    raw, _ = display.supported_tissue_layer(reopened, row, raw_qc=True)
    assert np.array_equal(reopened.mask(raw), original)
    # Reject an inconsistent remove-only history without discarding the extra voxel.
    added = corrected.copy()
    added[4, 4, 20] = True
    malformed = {**row, "data": reopened.put_array(added.astype(np.uint8))}
    assert display.supported_tissue_layer(reopened, malformed) == (malformed, None)
    reopened.commit_edit(row["id"], added, author="fixture", expected=reopened.identity(),
                         operations=[{"operation": "add"}])
    mixed = reopened.layer("siam-15")
    assert display.supported_tissue_layer(reopened, mixed) == (mixed, None)


def test_outer_margin_excludes_air_chip_preserving_dark_surface_and_bright_island(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "native", "margin-test")
    mri = np.zeros((72, 72, 72), np.float32)
    mri[17:54, 17:54, 17:54] = 150
    mri[56, 44, 32] = 95
    image = nib.Nifti1Image(mri, np.eye(4))
    image.header.set_xyzt_units("mm")
    path = tmp_path / "scan.nii"
    nib.save(image, path)
    s = c.add_series(path, series_id="scan", modality="T1")
    points = [(55, 30, 32), (56, 30, 32), (56, 44, 32)]
    labels = np.zeros_like(mri, np.uint8)
    for p in points:
        labels[p] = 15
    row = c.add_layer(labels, s["affine"], layer_id="skull", kind="skull", role="context",
                      source_series="scan", color="#ffffff", label_value=15, provenance={
                          "model": "siam-v2", "source": s["original"]["sha256"],
                          "source_array_sha256": s["data"]["sha256"], "label_ontology": {"value": 15}})
    # The coarse margin encompasses all three. Their size/class cannot
    # distinguish the air prediction from the separate MRI-supported island.
    (outer, _, stride), _ = display._head_support(mri, np.eye(4), lambda: None)
    assert all(outer[tuple((p + stride // 2) // stride)] for p in points)
    layer, report = display.supported_tissue_layer(c, row)
    assert [bool(c.mask(layer)[p]) for p in points] == [True, False, True]
    assert report["nativeAirClearanceMm"] == 2 and report["hiddenVoxels"] == 1
    assert c.mask(row).sum() == 3
