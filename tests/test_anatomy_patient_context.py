"""Disjoint native fields in a single, source-bound patient scene."""
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientImagingCollection, world


@pytest.mark.parametrize("ct_alignment", ["source-provided", "unreviewed"])
def test_shared_cutaway_saves_and_reopens_independently_of_selected_source(tmp_path, ct_alignment, monkeypatch):
    c = PatientImagingCollection.create(tmp_path / "study", "patient")
    volume = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    for sid, modality, offset in (("mri", "T1", 0), ("ct", "CT", 0), ("scout", "scout", 80)):
        path = tmp_path / f"{sid}.nii"
        frame = np.eye(4)
        frame[0, 3] = offset
        image = nib.Nifti1Image(volume, frame)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        c.add_series(path, series_id=sid, modality=modality)
    matrix = np.eye(4)
    matrix[0, 3] = 20
    c.set_registration("ct", matrix, status=ct_alignment, method="fixture")
    c.save()

    def client(directory):
        session = BridgeSession(tmp_path / directory)
        return lambda op, args: session.execute(op, args, _Request("cutaway"), lambda *_: None)

    call = client("transfers")
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    args = {"collectionId": opened["collectionId"], "snapshotId": c.identity()}
    view_args = {"patientModel": True, "previewAlignment": True}
    mri = call("anatomyView", {**args, **view_args, "seriesId": "mri"})
    assert mri["shape"] == [10, 10, 10]
    assert mri["patientContext"]["shape"] == [30, 10, 10]
    assert mri["patientContext"]["affine"] == np.eye(4).tolist()

    def scan_view(mode, x):
        return {"cursor": None, "visible": {}, "settings": {"presentation": mode,
                "clipping": {"axis": "x", "value": x, "enabled": True}}}

    saved = {"schema": "patient-anatomy-view-v1", "patientId": "patient",
             "snapshotId": c.identity(), "seriesId": "mri",
             "states": {"mri": scan_view("segmentation", 3), "ct": scan_view("scan", 29.5),
                        "scout": scan_view("scan", 89.5)},
             "patientCutaway": {"clipping": {"axis": "x", "value": 20, "enabled": True, "flipped": True}}}
    # Putting the shared plane in the per-scan preference reproduces the old failure.
    invalid = deepcopy(saved)
    invalid["states"]["mri"]["settings"]["clipping"] = saved["patientCutaway"]["clipping"]
    with pytest.raises(ValueError, match="outside the source image"):
        call("anatomySaveView", {**args, "viewState": invalid})
    for selected in ("mri", "ct", "scout"):
        for enabled in (True, False):
            state = deepcopy(saved)
            state["seriesId"] = selected
            state["patientCutaway"]["clipping"]["enabled"] = enabled
            assert call("anatomySaveView", {**args, "viewState": state, "saveArchive": False})["saved"]
            archive = tmp_path / f"{selected}-{enabled}.rsworkspace"
            call("anatomySavePackage", {**args, "path": str(archive)})
            reopen = client(f"cold-{selected}-{enabled}")
            result = reopen("anatomyOpen", {"path": str(archive), "workingRoot": str(tmp_path / "reopen")})
            assert result["viewState"] == state
            restored = reopen("anatomyView", {**args, **view_args, "collectionId": result["collectionId"],
                                             "seriesId": selected})
            if selected == "scout":
                assert restored["registrationStatus"] == "independent source frame"
                assert "patientContext" not in restored
                assert restored["affine"][0][3] == 80
            else:
                assert restored["patientContext"]["shape"] == [30, 10, 10]
    for enabled in (True, False):
        invalid = deepcopy(saved)
        invalid["patientCutaway"]["clipping"].update(value=29.6, enabled=enabled)
        with pytest.raises(ValueError, match="outside the shared patient context"):
            call("anatomySaveView", {**args, "viewState": invalid})
        invalid = deepcopy(saved)
        invalid["states"]["scout"]["settings"]["clipping"].update(value=20, enabled=enabled)
        with pytest.raises(ValueError, match="outside the source image"):
            call("anatomySaveView", {**args, "viewState": invalid})

    # A newly estimated transform can shrink the patient grid. Retire only the
    # old preferred planes that would no longer fit and keep the live study open.
    def realign(collection, **_):
        collection.set_registration("ct", np.eye(4), status="unreviewed", method="fixture recalculation")
        return True

    monkeypatch.setattr("resectionlab.anatomy_alignment.ensure_alignment", realign)
    call("anatomySaveView", {**args, "viewState": saved, "saveArchive": False})
    registered = call("anatomyRegister", {**args, "action": "estimate", "seriesId": "ct"})
    assert registered["retiredClipping"] == [
        {"seriesId": None, "cutaway": saved["patientCutaway"]},
        {"seriesId": "ct", "cutaway": {"clipping": saved["states"]["ct"]["settings"]["clipping"]}},
    ]
    updated = call("anatomyView", {**args, **view_args, "seriesId": "mri",
                                  "snapshotId": registered["document"]["snapshot_id"]})
    assert updated["patientContext"]["shape"] == [10, 10, 10]
    restored = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})["viewState"]
    assert restored["patientCutaway"] == {}
    assert restored["states"]["mri"] == saved["states"]["mri"]
    assert restored["states"]["scout"] == saved["states"]["scout"]
    assert restored["states"]["ct"]["settings"] == {"presentation": "scan"}


def test_unified_patient_context_and_surfaces_survive_linked_scan_switch(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "study", "patient")
    shape = (12, 10, 10)
    mask = np.zeros(shape, np.uint8)
    mask[2:8, 2:8, 2:8] = 1
    for name, modality, brightness, offset in (
        ("mr", "T1", 80, 0), ("ct", "CT", 800, 35), ("other", "T2", 42, 70)
    ):
        path = tmp_path / f"{name}.nii"
        image = nib.Nifti1Image(mask.astype(np.float32) * brightness, np.eye(4))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        scan = c.add_series(path, series_id=name, modality=modality)
        if modality == "CT":
            scan["intensity_units"] = "HU"
        c.add_layer(mask, np.eye(4), layer_id=name+"-structure", source_series=name,
                    kind="bone" if name == "ct" else "brain", role="context", color="#aaaaaa")
        if name != "other":
            c.add_layer(mask, np.eye(4), layer_id=name+"-region", source_series=name,
                        kind="brain-region", role="context", color="#ccbb99")
        if offset:
            transform = np.eye(4)
            transform[0, 3] = offset
            c.set_registration(name, transform, status="unreviewed", method="fixture")
    c.document["transforms"]["other"]["source_sha256"] = "0"*64
    c.save()
    bridge = BridgeSession(tmp_path / "transfers")
    call = lambda op, args: bridge.execute(op, args, _Request("patient-context"), lambda *_: None)
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    args = {"collectionId": opened["collectionId"], "snapshotId": c.identity(),
            "layers": ["mr-structure", "ct-structure", "other-structure"], "patientModel": True}
    def as_array(d):
        return np.fromfile(Path(d["path"]), dtype=d["dtype"]).reshape(d["shape"])
    def view(series, preview=True):
        return call("anatomyView", {**args, "seriesId": series, "previewAlignment": preview})

    mr, ct = view("mr"), view("ct")
    context = mr["patientContext"]
    assert context == ct["patientContext"]
    assert mr["registrationStatus"] == ct["registrationStatus"] == "alignment preview"
    assert [s["alignment"] for s in context["sources"]] == ["reference", "estimated", "unavailable"]
    assert context["sources"][1]["displayWindow"] == [0., 1000.]
    signal = as_array(context["signal"])
    assert signal[5, 5, 5] == pytest.approx(1.)
    assert signal[40, 5, 5] == pytest.approx(.8)
    assert signal[20, 5, 5] == 0 and signal.shape[0] == 47
    for selected, original_max in ((mr, 80), (ct, 800)):
        assert selected["unavailableLayers"] == ["other-structure"]
        assert as_array(selected["mri"]).max() == original_max
        assert selected["shape"] == list(shape)
        first, second = selected["compartments"]
        assert first["sourceSeries"] == "mr" and first["alignment"] == "reference"
        assert second["sourceSeries"] == "ct" and second["alignment"] == "estimated"
        assert as_array((second if selected is mr else first)["mask"]).sum() == 0
        for layer, expected_x in ((first, 4.5), (second, 39.5)):
            physical = world(np.asarray(selected["affine"]), as_array(layer["surface"]["source"]))
            assert physical[:, 0].mean() == pytest.approx(expected_x)
    # Equal categorical bytes on disjoint native fields need separate slice overlays.
    regions = call("anatomyView", {**args, "seriesId": "mr", "previewAlignment": True,
                                   "layers": ["mr-region", "ct-region"]})
    sets = {row["id"]: as_array(row["data"]) for row in regions["labelSets"]}
    first, second = regions["compartments"]
    assert len(sets) == 2 and first["labelSet"] != second["labelSet"]
    assert np.count_nonzero(sets[first["labelSet"]]) == np.count_nonzero(mask)
    assert not np.any(sets[second["labelSet"]])
    reviewed = view("mr", False)
    assert reviewed["unavailableLayers"] == ["ct-structure", "other-structure"]
    assert reviewed["patientContext"]["shape"] == list(shape)
    independent = view("other")
    assert independent["registrationStatus"] == "independent source frame"
    assert "patientContext" not in independent
    assert independent["unavailableLayers"] == ["mr-structure", "ct-structure"]


@pytest.mark.parametrize(("filename", "metadata"), [
    ("source.nii", {"SeriesDescription": "TOF_MIP"}),
    ("source.nii", {"ProtocolName": "sagittal scout localizer"}),
    ("coronal_MIP.nii", {}),
])
def test_projection_metadata_preserves_source_preview_and_excludes_patient_fusion(tmp_path, filename, metadata):
    from resectionlab.anatomy_alignment import alignment_pairs
    from resectionlab.anatomy_model_registry import source_volume
    from resectionlab.anatomy_sections import section
    from resectionlab.anatomy_workspace import checked_view

    c = PatientImagingCollection.create(tmp_path / "study", "patient")
    volume = np.zeros((10, 10, 10), np.float32)
    volume[2:7, 3:7, 4:8] = 1
    for sid, name in (("mr", "mr.nii"), ("tof", filename)):
        path = tmp_path / name
        image = nib.Nifti1Image(volume * (10 if sid == "mr" else 40), np.eye(4))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        row = c.add_series(path, series_id=sid, modality="T1" if sid == "mr" else "TOF")
    row["metadata"] = metadata
    c.set_registration("tof", np.eye(4), status="source-provided", method="fixture")
    c.add_layer(volume.astype(np.uint8), np.eye(4), layer_id="projection-label", source_series="tof",
                kind="vessels", role="context", color="#aa3322")
    c.save()
    assert not source_volume(row) and not alignment_pairs(c)
    with pytest.raises(ValueError, match="Projection or localizer"):
        c.patient_affine(row)
    with pytest.raises(ValueError, match="Projection or localizer"):
        section(c, series_id="mr", center=[5,5,5], normal=[0,0,1], size=16, moving_id="tof")
    bridge = BridgeSession(tmp_path / "transfers")
    call = lambda op, args: bridge.execute(op, args, _Request("projection"), lambda *_: None)
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    args = {"collectionId": opened["collectionId"], "snapshotId": c.identity(), "layers": ["projection-label"],
            "previewAlignment": True, "patientModel": True}
    primary = call("anatomyView", {**args, "seriesId":"mr"})
    assert primary["unavailableLayers"] == ["projection-label"]
    excluded = primary["patientContext"]["sources"][1]
    assert "displayWindow" not in excluded and "Projection or localizer" in excluded["reason"]
    preview = call("anatomyView", {**args, "seriesId":"tof"})
    assert preview["registrationStatus"] == "independent source frame" and "patientContext" not in preview
    assert np.fromfile(preview["mri"]["path"], np.float32).max() == 40
    assert "Projection or localizer" in preview["sourceFrameIssue"]
    assert preview["compartments"][0]["alignment"] == "independent"
    checked_view(c, {"schema":"patient-anatomy-view-v1","snapshotId":c.identity(),"patientId":"patient",
                     "seriesId":"tof","states":{"tof":{"cursor":None,"visible":{},"settings":{}}}})

def test_sidecar_only_scout_stays_independent_after_import_and_reopen(tmp_path):
    from resectionlab.anatomy_import import import_nifti_series, select_intake_reference
    from resectionlab.anatomy_model_registry import source_volume

    c = PatientImagingCollection.create(tmp_path / "study", "patient")
    for filename, sid in (("a", "series-a"), ("b", "series-b")):
        path = tmp_path / (filename + ".nii")
        image = nib.Nifti1Image(np.ones((12, 12, 10), np.float32), np.eye(4))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        if sid == "series-a":
            path.with_suffix(".json").write_text('{"ProtocolName":"Scout", "SeriesDescription":"T1"}')
        import_nifti_series(c, path, series_id=sid, modality="T1")
    assert select_intake_reference(c) == "series-b"
    c.save()
    reopened = PatientImagingCollection(c.root)
    assert not source_volume(reopened.series("series-a"))
    assert source_volume(reopened.series("series-b"))


def test_head_support_restricts_only_the_3d_native_and_fused_scalar(tmp_path, monkeypatch):
    from resectionlab import anatomy_siam_display
    c = PatientImagingCollection.create(tmp_path / "study", "patient")
    shape = (10, 10, 10)
    raw = np.zeros(shape, dtype=np.float32)
    raw[2:5, 2:6, 2:6] = 100
    raw[8, 8, 8] = 1500
    support = (raw == 100).astype(np.uint8)
    image = nib.Nifti1Image(raw, np.eye(4))
    image.header.set_xyzt_units("mm")
    path = tmp_path / "signal.nii"
    nib.save(image, path)
    c.add_series(path, series_id="t1", modality="T1")
    c.save()
    monkeypatch.setattr(anatomy_siam_display, 'source_head_support', lambda c, row, check: support)
    bridge = BridgeSession(tmp_path / "transfers")
    call = lambda op, args: bridge.execute(op, args, _Request("head-support"), lambda *_: None)
    opened = call("anatomyOpen", {"path": str(c.root / "patient-anatomy.json")})
    args = {"collectionId": opened["collectionId"], "snapshotId": c.identity(), "patientModel": True, "seriesId":"t1"}
    for original in [False,True,False]:
        view = call("anatomyView", {**args, "original":original})
        assert view['patientContext']['sources'][0]['headSupport'] is (not original)
        assert np.array_equal(np.fromfile(view['mri']['path'],np.float32).reshape(shape), raw)
        field = np.fromfile(view['patientContext']['signal']['path'],np.float32).reshape(shape)
        assert field[3, 3, 3] > 0
        assert (field[8, 8, 8] > 0) == original
        if original:
            assert 'headSupport' not in view
        else:
            assert np.array_equal(np.fromfile(view['headSupport']['path'],np.uint8).reshape(shape), support)
