"""SIAM preparation preserves publisher classes, native geometry and source scope."""
import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_models, anatomy_model_registry as registry, anatomy_siam, siam_worker
from resectionlab.anatomy_evidence import is_dural_proxy, vascular_evidence_supported
from resectionlab.anatomy_preparation import preparation_plan
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock


def scan(c, sid, original):
    image = nib.Nifti1Image(np.ones((8, 9, 10), np.float32), [
        [0, -.9, 0, 11], [0, 0, 1.1, -21], [.8, 0, 0, 16], [0, 0, 0, 1],
    ])
    image.header.set_xyzt_units("mm")
    nib.save(image, original)
    return c.add_series(original, series_id=sid, modality="T1")


def test_siam_admission_native_grid_exact_ontology_and_cached_output(tmp_path, monkeypatch):
    c = PatientImagingCollection.create(tmp_path / "patient", "fixture")
    row = scan(c, "t1", tmp_path / "t1.nii.gz")
    monkeypatch.setattr(registry, "head_coverage", lambda *a, **k: {"status": "eligible"})
    monkeypatch.setattr(registry, "model_environment", lambda *a, **k: {"version": "test"})
    monkeypatch.setattr(anatomy_siam, "reconstruction_lock", lambda: reconstruction_lock(tmp_path / "locks"))
    calls = []

    def worker(command, directory, **limits):
        calls.append(limits)
        src = nib.load(command[-3])
        labels = (np.arange(np.prod(src.shape)).reshape(src.shape) % 17).astype(np.uint8)
        image = nib.Nifti1Image(labels, src.affine, src.header.copy())
        image.set_data_dtype(np.uint8)
        nib.save(image, command[-2])
        return {"peak_sampled_rss_bytes": 1}

    monkeypatch.setattr(anatomy_models, "run_bounded", worker)
    record = anatomy_models.reconstruct(c, "t1", "siam-v2", full_head_confirmed=True)
    assert len(calls) == 1 and 0 < calls[0]["minimum_available_bytes"] < calls[0]["memory_bytes"]
    assert record["availability"] == "available" and record["model_source"] == siam_worker.RECORD
    assert record["source_array_sha256"] == row["data"]["sha256"]
    anatomy_models.import_proposal(c, "t1", record)
    layers = c.document["layers"]
    assert [r["label_value"] for r in layers] == list(range(1, 17))
    assert len({r["data"]["sha256"] for r in layers}) == 1
    assert all(r["role"] == "context" and r["source_series"] == "t1" for r in layers)
    assert all(r["affine"] == row["affine"] and r["data"]["shape"] == row["data"]["shape"] for r in layers)
    dura, vessel, skull = layers[12:15]
    assert [r["kind"] for r in (dura, vessel, skull)] == ["dura", "vessels", "skull"]
    assert not is_dural_proxy(dura) and not vascular_evidence_supported(c, vessel)
    assert dura["provenance"]["kind"] == "image_derived_estimate"
    assert skull["provenance"]["label_ontology"]["name"] == "Skull"
    assert all(r["provenance"]["training_admission"] is False for r in layers)
    assert anatomy_models.reconstruct(c, "t1", "siam-v2", full_head_confirmed=True) == record
    anatomy_models.import_proposal(c, "t1", record)
    assert len(calls) == 1 and len(layers) == 16
    with pytest.raises(ValueError, match="source data differs"):
        anatomy_models.import_proposal(c, "t1", dict(record, source_array_sha256="0" * 64))
    # A changed on-disk output fails independently of any selection metadata.
    with open(record["path"], "ab") as out:
        out.write(b"modified")
    with pytest.raises(ValueError, match="output differs"):
        anatomy_models.reconstruct(c, "t1", "siam-v2", full_head_confirmed=True)


def test_siam_rejects_unknown_codes_non_native_geometry_and_projection(tmp_path):
    c = PatientImagingCollection.create(tmp_path / "patient", "fixture")
    row = scan(c, "t1", tmp_path / "t1.nii.gz")
    shape = row["data"]["shape"]

    def image(value, affine):
        im = nib.Nifti1Image(np.full(shape, value, np.uint8), affine)
        im.header.set_xyzt_units("mm")
        return im

    with pytest.raises(ValueError, match="ontology"):
        anatomy_siam.categorical(image(17, row["affine"]), row)
    with pytest.raises(ValueError, match="native source grid"):
        anatomy_siam.categorical(image(13, np.eye(4)), row)
    row["series_description"] = "Maximum intensity projection"
    stage = next(s for s in preparation_plan(c, full_head_confirmed=True)["stages"] if s["id"] == "siam-tissues")
    assert stage["status"] == "unsupported_input"
    with pytest.raises(ValueError, match="full-head structural source"):
        anatomy_models.reconstruct(c, "t1", "siam-v2", full_head_confirmed=True)


def test_siam_restores_permutation_flips_and_oblique_native_geometry():
    oblique = nib.eulerangles.euler2mat(z=.32)
    a = np.eye(4)
    a[:3, :3] = oblique @ np.array([[0, -.8, 0], [0, 0, 1.4], [-1.1, 0, 0]])
    a[:3, 3] = [19, -16, 44]
    data = np.zeros((7, 8, 9), np.uint8)
    data[2, 4, 7] = 13
    data[5, 1, 2] = 14
    src = nib.Nifti1Image(data, a)
    native = siam_worker.restore_orientation(np.asarray(nib.as_closest_canonical(src).dataobj).T, src)
    assert np.array_equal(native, data)


def test_siam_preflight_refuses_changed_local_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(registry.sys, "platform", "darwin")
    monkeypatch.setenv("RESECTIONLAB_SIAM_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Local SIAM"):
        registry.model_environment("siam-v2")
    for name in siam_worker.ASSETS:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="pinned publisher"):
        registry.model_environment("siam-v2")


def test_each_eligible_mri_retains_its_own_stage_when_reference_is_cropped(tmp_path, monkeypatch):
    from resectionlab import anatomy_preparation as preparation

    c = PatientImagingCollection.create(tmp_path / "patient", "fixture")
    cropped = scan(c, "reference", tmp_path / "reference.nii.gz")
    t1 = scan(c, "intact-t1", tmp_path / "raw1.nii.gz")
    t2 = scan(c, "intact-t2", tmp_path / "raw2.nii.gz")
    t2["modality"] = "T2"
    cropped["coverage"] = {"cropped": True}

    def coverage(collection, row, *, confirmed=False, **options):
        if row.get("coverage", {}).get("cropped"):
            return {"status": "unsupported_input", "reason": "Source is cropped"}
        return {"status": "eligible" if confirmed else "unprocessed", "reason": "Confirm coverage"}

    monkeypatch.setattr(registry, "head_coverage", coverage)
    monkeypatch.setattr(registry, "model_environment", lambda *a, **k: {})
    jobs = [j for j in preparation_plan(c, full_head_confirmed=True)["stages"] if j["model"] == "siam-v2"]
    primary, extra, partial = jobs
    assert primary["id"] == "siam-tissues" and primary["seriesId"] == t1["id"]
    assert extra["id"] == "siam-tissues:intact-t2" and extra["seriesId"] == t2["id"]
    assert primary["status"] == extra["status"] == "ready" and primary["inputKey"] != extra["inputKey"]
    assert partial["seriesId"] == "reference" and partial["status"] == "unsupported_input"
    assert partial["reason"] == "Source is cropped"
    # If a preferred previously-cropped source becomes eligible, a completed
    # primary job remains attached to the source that actually produced it.
    receipt_layer = c.add_layer(np.ones((8, 9, 10), np.uint8), t1["affine"], source_series=t1["id"],
                                layer_id="first-output", kind="dura", color="#bb8844", role="context")
    receipts = {"siam-tissues": {"status": "cached", "inputKey": primary["inputKey"],
                                 "outputs": [{"id": receipt_layer["id"], "sha256": receipt_layer["data"]["sha256"]}]}}
    del cropped["coverage"]["cropped"]
    stages = preparation._plan(c, {"full_head_confirmed": True}, receipts)["stages"]
    bound = next(j for j in stages if j["id"] == "siam-tissues")
    assert bound["status"] == "cached" and bound["seriesId"] == t1["id"]
    assert next(j for j in stages if j["id"] == "siam-tissues:reference")["seriesId"] == "reference"
