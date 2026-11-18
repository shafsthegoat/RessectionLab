"""Grouped native intake checks with small synthetic images."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_intake import inspect_intake, reviewed_intake
from resectionlab.anatomy_planning import native_target_options, route_clearance
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientAnatomySnapshot, PatientImagingCollection


def _image(path, data):
    im = nib.Nifti1Image(np.asarray(data), np.eye(4))
    im.header.set_xyzt_units("mm")
    nib.save(im, path)
    return path


def _study(tmp_path, patient="patient-a"):
    c = PatientImagingCollection.create(tmp_path / "study", patient, timepoint="preoperative")
    c.save()
    return c


def _item(path, kind="nifti", **selected):
    [discovered] = inspect_intake(kind, [str(path)])["items"]
    assert discovered["supported"]
    return {**{key: discovered[key] for key in ("path", "kind", "sourceReceipt")}, **selected}


def _request(c, items, **fields):
    return dict(snapshotId=c.identity(), patientId=c.document["patient_id"],
                timepoint="preoperative", confirmedAssociation=True, items=items, **fields)


def test_grouped_intake_attaches_annotation_to_selected_scan_and_retries_once(tmp_path):
    c = _study(tmp_path)
    scan = _image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[2:4, 2:4, 2:4] = 3
    segmentation = _image(tmp_path / "seg.nii.gz", labels)
    items = [_item(segmentation, "labels", sourceInputIndex=1,
                   roleMap={"3": {"kind": "tumor", "label": "Enhancing lesion"}}),
             _item(scan, modality="T1")]
    commits = []
    revised = reviewed_intake(c, _request(c, items), commit=lambda: commits.append(True))
    assert revised.document["revision"] == c.document["revision"] + 1
    [layer] = revised.document["layers"]
    assert layer["source_series"] == revised.document["reference_series"]
    assert layer["role"] == "context" and layer["provenance"]["planning_target_candidate"]
    assert layer["provenance"]["label_ontology"]["value"] == 3
    assert layer["review_status"] == "unreviewed imported annotation"
    assert revised.mask(layer).sum() == 8
    assert reviewed_intake(revised, _request(revised, items), commit=lambda: commits.append(True)).identity() == revised.identity()
    assert commits == [True]


def test_changed_input_or_mismatched_annotation_cannot_publish_partial_group(tmp_path):
    c = _study(tmp_path)
    scan = _image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    scan_item = _item(scan, modality="T1")
    _image(scan, np.full((8, 8, 8), 2, np.float32))
    with pytest.raises(ValueError, match="changed"):
        reviewed_intake(c, _request(c, [scan_item]))
    assert PatientImagingCollection(c.root).identity() == c.identity()

    labels = _image(tmp_path / "seg.nii.gz", np.zeros((8, 8, 8), np.uint8))
    sidecar = labels.with_name("seg.json")
    sidecar.write_text('{"SourcePatientID":"other-patient"}')
    label_item = _item(labels, "labels", sourceInputIndex=0, roleMap={"1": "tumor"})
    with pytest.raises(ValueError, match="Source patient ID differs"):
        reviewed_intake(c, _request(c, [_item(scan, modality="T1"), label_item]))
    assert PatientImagingCollection(c.root).identity() == c.identity()

    sidecar.write_text('{"SourceSeriesInstanceUID":"1.2.3"}')
    label_item = _item(labels, "labels", sourceInputIndex=0, roleMap={"1": "tumor"})
    with pytest.raises(ValueError, match="different source series"):
        reviewed_intake(c, _request(c, [_item(scan, modality="T1"), label_item]))
    assert PatientImagingCollection(c.root).identity() == c.identity()


def test_attestation_alias_and_visit_checks_are_independent_of_namespaces(tmp_path):
    c = _study(tmp_path, "local-alias")
    one = _image(tmp_path / "sub-patientA_ses-v1_T1w.nii.gz", np.ones((8, 8, 8), np.float32))
    other = _image(tmp_path / "sub-patientA_ses-v2_T2w.nii.gz", np.ones((8, 8, 8), np.float32))
    scan = _item(one, modality="T1")
    with pytest.raises(ValueError, match="Confirm the selected patient"):
        reviewed_intake(c, {**_request(c, [scan]), "confirmedAssociation": False})
    with pytest.raises(ValueError, match="exact source patient ID"):
        reviewed_intake(c, _request(c, [scan], sourcePatientId="wrong"))
    with pytest.raises(ValueError, match="study visit"):
        reviewed_intake(c, _request(c, [scan, _item(other, modality="T2")], sourcePatientId="patientA"))
    c = reviewed_intake(c, _request(c, [scan], sourcePatientId="patientA"))
    with pytest.raises(ValueError, match="study visit"):
        reviewed_intake(c, _request(c, [_item(other, modality="T2")], sourcePatientId="patientA"))


def test_empty_supplied_label_is_retained_with_source_receipt(tmp_path):
    c = _study(tmp_path)
    scan = _image(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    labels = _image(tmp_path / "empty.nii.gz", np.zeros((8, 8, 8), np.uint8))
    c = reviewed_intake(c, _request(c, [
        _item(scan, modality="T1"),
        _item(labels, "labels", sourceInputIndex=0, roleMap={"1": "tumor"}),
    ]))
    row, = c.document["layers"]
    assert row["evidence_status"] == "empty-output" and not c.mask(row).any()
    assert Path(c.path(row["provenance"]["source"]["path"])).is_file()


def _view(c, tmp_path, layers=()):
    bridge = BridgeSession(tmp_path / "transfers")
    opened = bridge.execute("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")},
                            _Request("open"), lambda *_: None)
    return bridge.execute("anatomyView", {"collectionId": opened["collectionId"],
                          "snapshotId": c.identity(), "layers": list(layers)},
                          _Request("view"), lambda *_: None)


@pytest.mark.parametrize(("problem", "reason"), [
    ("units", "declared millimeter units"),
    ("conflict", "qform/sform disagreement"),
    ("uncoded", "no coded qform or sform"),
])
def test_unresolved_single_series_is_viewable_without_a_metric_reference(tmp_path, problem, reason):
    c = _study(tmp_path)
    path = tmp_path / "t1.nii.gz"
    image = nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4))
    image.header.set_xyzt_units("unknown" if problem == "units" else "mm")
    if problem == "conflict":
        shifted = np.eye(4)
        shifted[0, 3] = 6
        image.set_qform(shifted, code=1)
    if problem == "uncoded":
        image.set_sform(None, code=0)
    nib.save(image, path)
    [discovered] = inspect_intake("nifti", [str(path)])["items"]
    assert discovered["supported"] and reason in discovered["warning"]
    c = reviewed_intake(c, _request(c, [_item(path, modality="T1")]))
    row = c.series(c.document["reference_series"])
    assert row["native_geometry"]["spatial_units"] == row["units_status"]
    assert _view(c, tmp_path)["registrationStatus"] == "independent source frame"
    with pytest.raises(ValueError, match=reason):
        route_clearance(PatientAnatomySnapshot.capture(c), [1, 1, 1], [2, 2, 2])
    # Older manifests lack the new header descriptor; the preserved original
    # still determines whether their geometry can support a measurement.
    row.pop("native_geometry")
    c.save("Fixture with legacy source geometry fields")
    reopened = PatientImagingCollection(c.root)
    with pytest.raises(ValueError, match=reason):
        reopened.patient_affine(reopened.series(row["id"]))


def test_unknown_annotation_geometry_leaves_its_source_viewable(tmp_path):
    c = _study(tmp_path)
    path = tmp_path / "t1.nii.gz"
    scan = nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4))
    scan.header.set_xyzt_units("mm")
    scan.set_qform(np.eye(4), code=1)
    scan.set_sform(None, code=0)
    nib.save(scan, path)
    annotation = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), np.eye(4)), annotation)
    c = reviewed_intake(c, _request(c, [
        _item(path, modality="T1"),
        _item(annotation, "labels", sourceInputIndex=0, roleMap={"1": "tumor"}),
    ]))
    [layer] = c.document["layers"]
    [target] = native_target_options(c)
    assert layer["native_geometry"]["spatial_units"] == "unknown"
    assert target["target"] is None and "declared millimeter units" in target["reason"]
    np.testing.assert_array_equal(c.patient_affine(c.series(c.document["reference_series"])), np.eye(4))
    displayed = _view(c, tmp_path, [layer["id"]])
    assert displayed["registrationStatus"] == "patient frame"
    assert displayed["unavailableLayers"] == [layer["id"]]
    layer.pop("native_geometry")
    layer.pop("units_status")
    layer["provenance"]["kind"] = "supplied_annotation"
    with pytest.raises(ValueError, match="declared millimeter units"):
        c.patient_affine(layer)
    layer.update(kind="functional", score=layer["data"])
    layer["provenance"].pop("kind")
    with pytest.raises(ValueError, match="declared millimeter units"):
        c.patient_affine(layer)


def test_two_dimensional_projection_keeps_its_native_header_as_raw_evidence(tmp_path):
    c = _study(tmp_path)
    path = tmp_path / "TOF-MIP.nii.gz"
    image = nib.Nifti1Image(np.ones((8, 8), np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    image.set_qform(np.eye(4), code=1)
    nib.save(image, path)
    c = reviewed_intake(c, _request(c, [_item(path)]))
    [raw] = c.document["source_assets"]
    assert raw["shape"] == [8, 8] and raw["native_geometry"]["frame_status"] == "coded"
    assert not c.document["series"]
