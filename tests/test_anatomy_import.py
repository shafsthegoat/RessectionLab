"""Fresh-source intake contracts, using only small generated geometry fixtures."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from resectionlab.anatomy_import import (
    imaging_modality, import_anatomy_derivative, import_dicom, import_nifti_series,
)
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientAnatomySnapshot, PatientImagingCollection
from resectionlab.surgical_patient import prepare_surgical_environment


def volume(path, data, affine=None):
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def collection(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "study", "patient-a")
    source = volume(tmp_path / "source.nii.gz", np.ones((8, 8, 8), np.float32))
    import_nifti_series(c, source, series_id="t1", modality="T1")
    c.save("fresh source")
    return c


def test_semantic_sequence_required_for_nifti_and_dicom_family():
    for raw in (None, "MRI", "MR", "PET", ""):
        with pytest.raises(ValueError, match="Unsupported imaging sequence"):
            imaging_modality(raw)
    for sequence in ("T1", "T1CE", "T2", "FLAIR", "MRA", "TOF"):
        assert imaging_modality(sequence, dicom_modality="MR") == sequence
    assert imaging_modality(None, dicom_modality="CT") == "CT"
    assert imaging_modality("CTA", dicom_modality="CT") == "CTA"
    with pytest.raises(ValueError, match="DICOM MR cannot"):
        imaging_modality("CT", dicom_modality="MR")
    with pytest.raises(ValueError, match="Unsupported DICOM Modality"):
        imaging_modality("T1", dicom_modality="SEG")


def test_dicom_requires_explicit_mr_series_identity_and_retains_originals(tmp_path, monkeypatch):
    c = PatientImagingCollection.create(tmp_path / "study", "patient-a")
    uids = ("1.2.3", "1.2.4")
    paths = {}
    for uid in uids:
        p = tmp_path / f"{uid}.dcm"
        p.write_bytes(uid.encode())
        paths[uid] = str(p)

    class Reader:
        @staticmethod
        def GetGDCMSeriesIDs(_directory):
            return uids

        @staticmethod
        def GetGDCMSeriesFileNames(_directory, uid):
            return [paths[uid]]

        def SetFileNames(self, files):
            self.files = files

        def Execute(self):
            return sitk.GetImageFromArray(np.ones((3, 4, 5), np.int16))

    monkeypatch.setattr(sitk, "ImageSeriesReader", Reader)
    monkeypatch.setattr(
        pydicom, "dcmread", lambda path, **_kw: SimpleNamespace(
            PatientID="patient-a", Modality="MR", SeriesInstanceUID=Path(path).stem,
            SeriesDescription="acquired MR", FrameOfReferenceUID="9.8.7",
        ),
    )
    with pytest.raises(ValueError, match="Unsupported imaging sequence"):
        import_dicom(c, tmp_path)
    with pytest.raises(ValueError, match="Multiple DICOM series"):
        import_dicom(c, tmp_path, modality="T1")
    with pytest.raises(ValueError, match="every DICOM SeriesInstanceUID"):
        import_dicom(c, tmp_path, sequence_by_series={uids[0]: "T1"})
    assert not c.document["series"]
    rows = import_dicom(c, tmp_path, sequence_by_series={uids[0]: "T1", uids[1]: "FLAIR"})
    assert [row["modality"] for row in rows] == ["T1", "FLAIR"]
    assert all(row["dicom_modality"] == "MR" and row["dicom_frame_uid"] == "9.8.7" for row in rows)
    assert all(c.path(row["dicom_originals"][0]["path"]).read_bytes() == row["id"].encode() for row in rows)
    c.save("typed DICOM")
    assert PatientImagingCollection(c.root).document["series"] == rows


def test_label_role_map_reaches_simulation_grid_and_reopens(tmp_path):
    c = collection(tmp_path)
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    labels[3:5, 3:5, 3:5] = 2
    labels[5:7, 5:7, 5:7] = 3
    labels[1:3, 5:7, 1:3] = 4
    path = volume(tmp_path / "seg.nii.gz", labels)
    with pytest.raises(ValueError, match="must cover every nonzero label"):
        import_anatomy_derivative(c, path, source_series="t1", kind="labels", label="source", role_map={1: "tumor"})
    assert c.document["layers"] == []
    rows = import_anatomy_derivative(c, path, source_series="t1", kind="labels", label="source", role_map={
        "1": {"kind": "tumor", "label": "enhancing target"}, "2": "brain", "3": "bone", "4": "vessels",
    })
    assert [(row["kind"], row["role"]) for row in rows] == [
        ("tumor", "target"), ("brain", "context"), ("skull", "context"), ("vessels", "critical")
    ]
    assert all(row["provenance"]["source"]["sha256"] == rows[0]["provenance"]["source"]["sha256"] for row in rows)
    c.save("mapped source labels")
    reopened = PatientImagingCollection(c.root)
    grid = PatientAnatomySnapshot.capture(reopened).simulation_grid()
    assert set(grid["targets"]) == {rows[0]["id"]}
    assert set(grid["critical"]) == {rows[3]["id"]}
    assert int(grid["targets"][rows[0]["id"]].sum()) == 8
    assert int(grid["critical"][rows[3]["id"]].sum()) == 8
    report = prepare_surgical_environment(c.root, tmp_path / "scene", layer_ids={
        "tumor": rows[0]["id"], "brain": rows[1]["id"], "bone": rows[2]["id"], "vessels": rows[3]["id"],
    })
    assert set(report["sourceLayers"]) == {"tumor", "brain", "bone", "vessels"}
    assert Path(report["casePath"]).exists()


def test_imported_layer_units_must_be_mm_for_scene(tmp_path):
    c = collection(tmp_path)
    segmentation = volume(tmp_path / "unknown-units.nii.gz", np.ones((8, 8, 8), np.uint8))
    image = nib.load(segmentation)
    image.header.set_xyzt_units("unknown")
    nib.save(image, segmentation)
    row = import_anatomy_derivative(c, segmentation, source_series="t1", kind="labels",
        label="unknown", role_map={1: "tumor"})[0]
    assert row["units_status"] == "unknown"
    c.save("unknown annotation units")
    with pytest.raises(ValueError, match="declared millimeter units"):
        prepare_surgical_environment(c.root, tmp_path / "scene", layer_ids={"tumor": row["id"]})


def test_bridge_import_validates_sequence_and_passes_label_map(tmp_path):
    c = collection(tmp_path)
    bridge = BridgeSession(tmp_path / "transfers")
    call = lambda op, args: bridge.execute(op, args, _Request("import-test"), lambda *_: None)
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    base = {"collectionId": opened["collectionId"], "snapshotId": opened["document"]["snapshot_id"]}
    second = volume(tmp_path / "second.nii.gz", np.ones((8, 8, 8), np.float32))
    with pytest.raises(ValueError, match="Unsupported imaging sequence"):
        call("anatomyImport", {**base, "kind": "nifti", "path": str(second), "modality": "MRI", "patientId": "patient-a"})
    assert len(PatientImagingCollection(c.root).document["series"]) == 1
    imported = call("anatomyImport", {**base, "kind": "nifti", "path": str(second), "modality": "T2", "patientId": "patient-a"})
    assert imported["document"]["series"][-1]["modality"] == "T2"
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[2:4, 2:4, 2:4] = 1
    segmentation = volume(tmp_path / "labels.nii.gz", labels)
    updated = call("anatomyImport", {"collectionId": base["collectionId"], "snapshotId": imported["document"]["snapshot_id"],
        "kind": "labels", "path": str(segmentation), "seriesId": "t1", "label": "supplied", "roleMap": {"1": "tumor"}})
    assert updated["document"]["layers"][0]["role"] == "target"
    assert updated["document"]["layers"][0]["kind"] == "tumor"


def test_ct_skull_adapter_selects_skull_output_without_running_model(tmp_path, monkeypatch):
    from resectionlab.anatomy_models import import_proposal, reconstruct
    from resectionlab.patient_anatomy import file_hash
    from resectionlab import poc_models

    c = PatientImagingCollection.create(tmp_path / "study", "patient-a")
    t1 = volume(tmp_path / "t1.nii.gz", np.ones((8, 8, 8), np.float32))
    import_nifti_series(c, t1, series_id="t1", modality="T1")
    with pytest.raises(ValueError, match="requires an acquired CT"):
        reconstruct(c, "t1", "totalsegmentator-skull")
    ct = volume(tmp_path / "ct.nii.gz", np.full((8, 8, 8), 100.0, np.float32))
    import_nifti_series(c, ct, series_id="ct", modality="CT")
    mask = np.zeros((8, 8, 8), np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    skull = volume(tmp_path / "skull_mask.nii.gz", mask)
    brain = volume(tmp_path / "brain_mask.nii.gz", mask)
    calls = []

    def fake_run(source, case, model, modality, **kwargs):
        calls.append((Path(source), case, model, modality))
        return {"outputs": [
            {"name": "brain_mask.nii.gz", "path": str(brain), "sha256": file_hash(brain)},
            {"name": "skull_mask.nii.gz", "path": str(skull), "sha256": file_hash(skull)},
        ]}

    monkeypatch.setattr(poc_models, "run", fake_run)
    proposal = reconstruct(c, "ct", "totalsegmentator-skull")
    assert calls[0][2:] == ("totalsegmentator-skull", "CT")
    assert proposal["kind"] == "skull" and proposal["path"] == str(skull)
    key = import_proposal(c, "ct", proposal)
    row = c.layer(key)
    assert row["kind"] == "skull" and row["role"] == "context"
    assert row["source_series"] == "ct"
    assert np.array_equal(c.mask(row), mask.astype(bool))
