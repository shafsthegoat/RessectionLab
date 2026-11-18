"""Independent native viewing when global alignment or geometry is unresolved."""

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_import import import_anatomy_derivative, import_nifti_series
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientImagingCollection


def _image(path, values, *, units="mm", translation=0., conflicting=False):
    matrix = np.eye(4)
    matrix[0, 3] = translation
    image = nib.Nifti1Image(values, matrix)
    image.header.set_xyzt_units(units)
    if conflicting:
        alternative = matrix.copy()
        alternative[0, 3] += 4.
        image.set_qform(alternative, code=1)
    nib.save(image, path)
    return path


def _view(collection, tmp_path, series, layers):
    bridge = BridgeSession(tmp_path / "transfers")
    opened = bridge.execute("anatomyOpen", {"path": str(collection.root / "patient-anatomy.json")},
                            _Request("open"), lambda *_: None)
    return bridge.execute("anatomyView", {"collectionId": opened["collectionId"],
                          "snapshotId": collection.identity(), "seriesId": series,
                          "layers": layers, "previewAlignment": True},
                          _Request("view"), lambda *_: None)


@pytest.mark.parametrize(("units", "conflicting"), [("unknown", False), ("mm", True)])
def test_an_unresolved_scan_stays_viewable_without_fused_layers(tmp_path, units, conflicting):
    c = PatientImagingCollection.create(tmp_path / "study", "generated-patient")
    image = _image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32), units=units, conflicting=conflicting)
    import_nifti_series(c, image, series_id="t1")
    label = _image(tmp_path / "mask.nii.gz", np.ones((8, 8, 8), np.uint8), units="unknown")
    [layer] = import_anatomy_derivative(c, label, source_series="t1", kind="labels", label="Supplied",
                                       role_map={"1": "tumor"})
    c.save()
    payload = _view(c, tmp_path, "t1", [layer["id"]])
    assert payload["registrationStatus"] == "independent source frame"
    assert payload["shape"] == [8, 8, 8] and payload["mri"]
    assert payload["compartments"] == []
    assert payload["unavailableLayers"] == [layer["id"]]


def test_unregistered_valid_scan_preserves_local_geometry_and_exact_lattice_preview(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "study", "generated-patient")
    image = _image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    local = _image(tmp_path / "t2.nii.gz", np.full((8, 8, 8), 4., np.float32))
    import_nifti_series(c, image, series_id="t1")
    import_nifti_series(c, local, series_id="t2")
    data = np.zeros((8, 8, 8), np.uint8)
    data[2:4, 2:4, 2:4] = 1
    ids = {}
    for label, units, offset in (("metric", "mm", 0.), ("preview", "unknown", 0.), ("misaligned", "unknown", 3.)):
        path = _image(tmp_path / (label + ".nii.gz"), data, units=units, translation=offset)
        [layer] = import_anatomy_derivative(c, path, source_series="t2", kind="labels", label=label,
                                           role_map={"1": "tumor"})
        ids[label] = layer["id"]
    c.save()
    payload = _view(c, tmp_path, "t2", list(ids.values()))
    assert payload["registrationStatus"] == "independent source frame"
    visible = {item["name"]: item for item in payload["compartments"]}
    assert visible[ids["metric"]]["volumeMm3"] == 8.
    assert "volumeMm3" not in visible[ids["preview"]]
    assert "Source-grid preview" in visible[ids["preview"]]["label"]
    assert payload["unavailableLayers"] == [ids["misaligned"]]
