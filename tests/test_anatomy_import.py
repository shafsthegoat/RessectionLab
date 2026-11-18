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
    inspect_nifti_series, select_intake_reference,
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
    for sequence in ("T1", "T1CE", "T2", "FLAIR", "MRA", "TOF", "SWI", "SWI_MAGNITUDE", "SWI_PHASE"):
        assert imaging_modality(sequence, dicom_modality="MR") == sequence
    assert imaging_modality(None, dicom_modality="CT") == "CT"
    assert imaging_modality("CTA", dicom_modality="CT") == "CTA"
    with pytest.raises(ValueError, match="DICOM MR cannot"):
        imaging_modality("CT", dicom_modality="MR")
    with pytest.raises(ValueError, match="Unsupported DICOM Modality"):
        imaging_modality("T1", dicom_modality="SEG")


def test_swi_components_remain_distinct_and_diffusion_is_not_flattened(tmp_path):
    c = collection(tmp_path)
    source = volume(tmp_path / "swi.nii.gz", np.ones((8, 8, 8), np.float32))
    for sequence, component in (("SWI", "processed susceptibility-weighted image"),
                                ("SWI_MAGNITUDE", "magnitude"), ("SWI_PHASE", "phase")):
        row = import_nifti_series(c, source, series_id=sequence.lower(), modality=sequence)
        assert row["modality"] == sequence
        assert row["susceptibility_component"] == component
    diffusion = volume(tmp_path / "raw_dti.nii.gz", np.ones((8, 8, 8, 4), np.float32))
    with pytest.raises(ValueError, match="bounded 3D anatomical volume"):
        import_nifti_series(c, diffusion, series_id="dti", modality="T1")
    assert len(c.document["series"]) == 4
    assert not c.document["layers"]
    c.save("distinct susceptibility contrasts")
    assert PatientImagingCollection(c.root).series("swi_phase")["susceptibility_component"] == "phase"


@pytest.mark.parametrize("extension", ["trk", "tck"])
def test_supplied_tract_file_coordinates_roundtrip_in_source_ras_mm(tmp_path, extension):
    c = collection(tmp_path)
    source_affine = np.array([[0, -2, 0, 31], [1.5, 0, 0, -17], [0, 0, 3, 9], [0, 0, 0, 1]], float)
    source = volume(tmp_path / "flair.nii.gz", np.ones((8, 8, 8), np.float32), source_affine)
    import_nifti_series(c, source, series_id="flair", modality="FLAIR")
    voxels = [np.array([[1, 2, 3], [2, 2, 3], [3, 4, 4]], np.float32),
              np.array([[1, 1, 1], [2, 1, 1]], np.float32)]
    physical = [nib.affines.apply_affine(source_affine, v).astype(np.float32) for v in voxels]
    tractogram = nib.streamlines.Tractogram(physical, affine_to_rasmm=np.eye(4))
    path = tmp_path / ("supplied_bundle." + extension)
    if extension == "trk":
        header = nib.streamlines.TrkFile.create_empty_header()
        header["voxel_to_rasmm"] = source_affine
        header["voxel_sizes"] = nib.affines.voxel_sizes(source_affine)
        header["dimensions"] = [8, 8, 8]
        nib.streamlines.TrkFile(tractogram, header=header).save(path)
    else:
        nib.streamlines.save(tractogram, path)
    row = import_anatomy_derivative(c, path, source_series="flair", kind="tractography", label="Supplied bundle")
    np.testing.assert_allclose(c.array(row["points_ras_mm"]), np.concatenate(physical), atol=1e-5)
    np.testing.assert_array_equal(c.array(row["offsets"]), [0, 3, 5])
    assert row["source_series"] == "flair" and row["label"] == "Supplied bundle"
    assert "unreviewed" in row["review_status"]
    c.save("source tract geometry")
    reopened = PatientImagingCollection(c.root)
    np.testing.assert_allclose(reopened.array(row["points_ras_mm"]), np.concatenate(physical), atol=1e-5)


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
            SeriesDescription="acquired MR", FrameOfReferenceUID="9.8.7", StudyInstanceUID="2.8.1",
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
    repeated = import_dicom(c, tmp_path, modality="FLAIR", selected_series=[uids[1]])
    assert [row["id"] for row in repeated] == [uids[1]]
    assert c.identity() == c.document["snapshot_id"]


def test_nifti_session_association_and_stable_reference(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "study", "deidentified", timepoint="preoperative")
    c.save()
    angiography = volume(tmp_path / "sub-a_ses-visit1_TOF.nii.gz", np.ones((8, 8, 8), np.float32))
    structural = volume(tmp_path / "sub-a_ses-visit1_T1w.nii.gz", np.zeros((8, 8, 8), np.float32))
    assert inspect_nifti_series(structural)["visit_id"] == "visit1"
    with pytest.raises(ValueError, match="Source patient ID differs"):
        import_nifti_series(c, structural, series_id="t1")
    first = import_nifti_series(c, angiography, series_id="tof", source_patient_id="a")
    import_nifti_series(c, structural, series_id="t1", source_patient_id="a")
    assert select_intake_reference(c) == "t1"
    c.save()
    reopened = PatientImagingCollection(c.root)
    repeated = import_nifti_series(reopened, angiography, series_id="copy", source_patient_id="a")
    assert repeated == first
    assert reopened.identity() == reopened.document["snapshot_id"]
    other_visit = volume(tmp_path / "sub-a_ses-visit2_T2w.nii.gz", np.ones((8, 8, 8), np.float32))
    with pytest.raises(ValueError, match="different visit"):
        import_nifti_series(reopened, other_visit, series_id="t2", source_patient_id="a")
    assert select_intake_reference(reopened) == "t1"


def test_raw_diffusion_companions_and_projection_stay_out_of_scalar_series(tmp_path):
    c = collection(tmp_path)
    c.document["source_assets"] = [{"name": "legacy-dwi", "storage": "unchanged external source"}]
    dwi = volume(tmp_path / "scan.nii.gz", np.ones((8, 8, 8, 2), np.float32))
    (tmp_path / "scan.bval").write_text("0 1000\n")
    (tmp_path / "scan.bvec").write_text("0 1\n0 0\n0 0\n")
    details = inspect_nifti_series(dwi)
    assert details["modality"] == "DWI" and details["storage"] == "source-asset"
    row = import_nifti_series(c, dwi, series_id="dwi")
    assert row["shape"] == [8, 8, 8, 2] and "data" not in row
    assert set(row["companions"]) == {"bval", "bvec"}
    mip = volume(tmp_path / "TOF-MIP.nii.gz", np.ones((8, 8, 8), np.float32))
    with pytest.raises(ValueError, match="projection cannot supply"):
        import_nifti_series(c, mip, series_id="mip", modality="TOF")
    assert import_nifti_series(c, mip, series_id="mip")["storage"] == "source-asset"
    c.save()
    reopened = PatientImagingCollection(c.root)
    assert [row["id"] for row in reopened.source_records()] == ["t1", "dwi", "mip"]
    assert reopened.document["reference_series"] == "t1"


def test_sidecar_preserves_processed_swi_and_reports_declared_coverage(tmp_path):
    c = collection(tmp_path)
    path = volume(tmp_path / "scan.nii.gz", np.zeros((8, 8, 8), np.float32))
    (tmp_path / "scan.json").write_text('{"SeriesDescription":"SWIp","ImageType":["DERIVED","M"],"Defaced":true}')
    row = import_nifti_series(c, path, series_id="swi")
    assert row["modality"] == "SWI"
    assert row["coverage"]["limitations"] == [{"kind": "privacy-redacted", "basis": "source metadata"}]
    assert c.path(row["companions"]["metadata"]["path"]).exists()


def test_sidecar_declares_hu_and_complete_head_in_an_unspecified_study(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "study", "patient-a", timepoint="source-session-unknown")
    timepoint = c.document["study_timepoint"]
    path = volume(tmp_path / "head_CT.nii.gz", np.full((8, 8, 8), 650, np.float32))
    sidecar = tmp_path / "head_CT.json"
    sidecar.write_text('{"IntensityUnits":"HU","FullHeadAcquired":true}')
    row = import_nifti_series(c, path, series_id="ct", timepoint=timepoint)
    assert row["intensity_units"] == "HU"
    assert row["coverage"]["full_head_acquired"] is True
    sidecar.write_text('{"FullHeadAcquired":true,"Cropped":true}')
    details = inspect_nifti_series(path)
    assert details["coverage"]["status"] == "limited"
    assert "full_head_acquired" not in details["coverage"]


def test_dicom_decoded_ct_retains_declared_hu(tmp_path, monkeypatch):
    c = PatientImagingCollection.create(tmp_path / "study", "patient-a")
    source = tmp_path / "1.2.3.dcm"
    source.write_bytes(b"CT fixture")

    class Reader:
        @staticmethod
        def GetGDCMSeriesIDs(_directory):
            return ["1.2.3"]

        @staticmethod
        def GetGDCMSeriesFileNames(_directory, _uid):
            return [str(source)]

        def SetFileNames(self, files):
            assert len(files) == 1

        def Execute(self):
            return sitk.GetImageFromArray(np.full((3, 4, 5), 650, np.int16))

    monkeypatch.setattr(sitk, "ImageSeriesReader", Reader)
    monkeypatch.setattr(pydicom, "dcmread", lambda *_args, **_kwargs: SimpleNamespace(
        PatientID="patient-a", Modality="CT", SeriesInstanceUID="1.2.3",
        FrameOfReferenceUID="9.8.7", StudyInstanceUID="2.8.1", RescaleType="HU",
    ))
    [row] = import_dicom(c, tmp_path)
    assert row["modality"] == "CT" and row["intensity_units"] == "HU"
    assert c.array(row["data"]).min() == 650


def test_patient_affine_requires_review_of_the_entire_registration_chain(tmp_path):
    c = collection(tmp_path)
    path = volume(tmp_path / "head_CT.nii.gz", np.full((8, 8, 8), 650, np.float32))
    imported = import_nifti_series(c, path, series_id="bone-ct")
    other = volume(tmp_path / "soft_CT.nii.gz", np.full((8, 8, 8), 54, np.float32))
    import_nifti_series(c, other, series_id="soft-ct")
    c.set_registration("soft-ct", np.eye(4), status="unreviewed", method="estimated")
    c.set_registration("bone-ct", np.eye(4), status="reviewed", method="estimated")
    c.document["transforms"]["bone-ct"].update(
        parent_series="soft-ct", reference_series="t1", source_to_parent=np.eye(4).tolist()
    )
    np.testing.assert_array_equal(c.registration_matrix("bone-ct"), np.eye(4))
    with pytest.raises(ValueError, match="soft-ct needs review"):
        c.patient_affine(imported)
    c.set_registration("soft-ct", np.eye(4), status="reviewed", method="estimated")
    np.testing.assert_array_equal(c.patient_affine(imported), np.eye(4))


def test_stale_collection_cannot_save_implicitly_and_new_file_create_cannot_replace(tmp_path):
    c = collection(tmp_path)
    other = PatientImagingCollection(c.root)
    c.document["missing"] = ["vessels unprocessed"]
    saved = c.save()
    other.document["missing"] = ["brain unprocessed"]
    with pytest.raises(ValueError, match="Patient changed"):
        other.save()
    assert PatientImagingCollection(c.root).identity() == saved
    new = tmp_path / "blank"
    first = PatientImagingCollection.create(new, "patient-a")
    second = PatientImagingCollection.create(new, "patient-b")
    first.save()
    with pytest.raises(ValueError, match="Patient changed"):
        second.save()
    assert PatientImagingCollection(new).document["patient_id"] == "patient-a"


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


def test_unknown_units_annotation_preview_stays_on_its_own_source_grid(tmp_path):
    c = collection(tmp_path)
    c.add_series(volume(tmp_path / "second-source.nii.gz", np.ones((8, 8, 8), np.float32)),
                 series_id="tof", modality="TOF")
    c.set_registration("tof", np.eye(4), status="source-provided", method="fixture")
    pixels = np.zeros((8, 8, 8), np.uint8)
    pixels[2:4, 2:5, 2:6] = 1
    layers = []
    for label, variant in (("matched", np.eye(4)), ("conflicted", np.diag([2, 1, 1, 1]))):
        filepath = tmp_path / (label + ".nii.gz")
        img = nib.Nifti1Image(pixels, np.eye(4))
        img.header.set_xyzt_units("unknown")
        img.set_qform(variant, code=1)
        nib.save(img, filepath)
        row = import_anatomy_derivative(c, filepath, source_series="t1", kind="labels",
            label=label, role_map={1: "vessels"})[0]
        assert row["units_status"] == "unknown"
        layers.append(row["id"])
    c.save("supplied exact lattice and conflicting control")
    with pytest.raises(ValueError, match="declared millimeter"):
        c.patient_affine(c.layer(layers[0]))

    bridge = BridgeSession(tmp_path / "transfers")
    call = lambda op, args: bridge.execute(op, args, _Request("source-lattice-preview"), lambda *_: None)
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    args = {"collectionId": opened["collectionId"], "snapshotId": opened["document"]["snapshot_id"],
            "layers": layers, "previewAlignment": True}
    prepared = call("anatomyView", {**args, "seriesId": "t1"})
    assert prepared["unavailableLayers"] == layers[1:]
    (shown,) = prepared["compartments"]
    assert shown["name"] == layers[0] and shown["label"].endswith("Source-grid preview")
    assert shown["qa"]["voxelCount"] == 24 and "volumeMm3" not in shown
    assert any("physical measurements unavailable" in issue for issue in shown["qa"]["issues"])
    assert call("anatomyView", {**args, "seriesId": "tof"})["unavailableLayers"] == layers
    assert call("anatomyView", {**args, "seriesId": "t1", "previewAlignment": False})["unavailableLayers"] == layers


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
    hu = np.full((8, 8, 8), -1000.0, np.float32)
    hu[2:6, 2:6, 2:6] = 600.0
    ct = volume(tmp_path / "ct.nii.gz", hu)
    row = import_nifti_series(c, ct, series_id="ct", modality="CT")
    row["intensity_units"] = "HU"
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
