"""Grouped native intake checks with small synthetic images."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_intake import inspect_intake, reviewed_intake
from resectionlab.patient_anatomy import PatientImagingCollection


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
