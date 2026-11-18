"""Source capability, model availability and geometric context regressions."""

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_model_registry as registry
from resectionlab import anatomy_models, head_segmentation
from resectionlab.anatomy_evidence import vascular_evidence_supported
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash


def source(c, directory, series_id, values, *, modality="T1", spacing=1):
    image = nib.Nifti1Image(values, np.diag([spacing, spacing, spacing, 1.0]))
    image.header.set_xyzt_units("mm")
    path = directory / (series_id + ".nii.gz")
    nib.save(image, path)
    return c.add_series(path, series_id=series_id, modality=modality)


def make_study(tmp_path):
    return PatientImagingCollection.create(tmp_path / "patient", "fixture")


def ellipsoid(radii):
    grid = np.ogrid[:80, :80, :80]
    return sum(((axis - 40) / radius) ** 2 for axis, radius in zip(grid, radii)) <= 1


def test_outer_head_requires_coverage_and_stays_unavailable_when_redacted(tmp_path):
    c = make_study(tmp_path)
    full = source(c, tmp_path, "full", ellipsoid([28, 32, 33]).astype(np.float32), spacing=3)
    assert registry.head_coverage(c, full)["status"] == "unprocessed"
    assert registry.head_coverage(c, full, confirmed=True)["status"] == "eligible"
    full["coverage"] = {"status": "unassessed", "full_head_acquired": True}
    assert registry.head_coverage(c, full)["status"] == "eligible"
    full["coverage"] = {"status": "limited", "limitations": [{"kind": "privacy-redacted"}]}
    assert registry.head_coverage(c, full, confirmed=True)["status"] == "unsupported_input"
    stripped = source(c, tmp_path, "structural", ellipsoid([17, 20, 20]).astype(np.float32), spacing=3)
    assert registry.head_coverage(c, stripped, confirmed=True)["status"] == "unsupported_input"
    clipped = source(c, tmp_path, "limited", ellipsoid([28, 32, 33]).astype(np.float32)[:45], spacing=3)
    assert registry.head_coverage(c, clipped, confirmed=True)["status"] == "unsupported_input"


def test_plan_keeps_ct_angiography_and_missing_tumor_distinct(tmp_path, monkeypatch):
    c = make_study(tmp_path)
    hu = np.full((16, 16, 16), -1000, np.float32)
    hu[3:12, 3:12, 3:12] = 500
    source(c, tmp_path, "angiography", hu, modality="CTA")
    source(c, tmp_path, "swi", hu, modality="SWI_MAGNITUDE")

    def installed(model, **options):
        assert model in {"vesselfm", "totalsegmentator-skull", "dural-envelope"}
        return {"fixture": model, "options": options}

    monkeypatch.setattr(registry, "model_environment", installed)
    stages = {row["id"]: row for row in registry.preparation_model_jobs(c, indication="meningioma", include_dural_proxy=True)}
    assert stages["ct-skull"]["status"] == "ready"
    assert stages["ct-skull"]["series_id"] == stages["vascular"]["series_id"] == "angiography"
    assert stages["tumor"]["status"] == "missing_input"
    assert stages["venous"]["status"] == "unsupported_input"
    assert stages["dural-envelope"]["depends_on"] == ["ct-skull"]
    assert stages["functional"]["status"] == "missing_input"


def test_projection_and_empty_vessel_output_cannot_supply_constraints(tmp_path, monkeypatch):
    c = make_study(tmp_path)
    source(c, tmp_path, "TOF_MIP", np.ones((8, 8, 8), np.float32), modality="TOF")
    with pytest.raises(ValueError, match="volumetric"):
        anatomy_models.reconstruct(c, "TOF_MIP", "vesselfm")
    source(c, tmp_path, "TOF", np.ones((8, 8, 8), np.float32), modality="TOF")
    path = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.uint8), np.eye(4)), path)
    from resectionlab import poc_models
    monkeypatch.setattr(poc_models, "run", lambda *args, **kwargs: {"outputs": [
        {"name": "vessels_mask.nii.gz", "path": str(path), "sha256": file_hash(path), "foreground_voxels": 0}
    ]})
    record = anatomy_models.reconstruct(c, "TOF", "vesselfm")
    layer_id = anatomy_models.import_proposal(c, "TOF", record)
    layer = c.layer(layer_id)
    assert record["availability"] == layer["availability"] == "empty_output"
    assert layer["role"] == "context"
    assert not vascular_evidence_supported(c, layer)


def test_pinned_head_ontology_keeps_mri_vessels_context_only(tmp_path, monkeypatch):
    c = make_study(tmp_path)
    row = source(c, tmp_path, "head", ellipsoid([28, 32, 33]).astype(np.float32), spacing=3)
    c.add_layer(ellipsoid([17, 20, 20]).astype(np.uint8), row["affine"], layer_id="brain",
                source_series="head", kind="brain", role="context", color="#334456")
    lut = tmp_path / "FreeSurferColorLUT.txt"
    lut.write_text("16 Brain-Stem 10 10 10 0\n902 Artery 200 0 0 0\n911 Skin 80 80 80 0\n915 Bone 40 50 60 0\n")
    monkeypatch.setattr(head_segmentation, "ASSETS", {lut.name: file_hash(lut)})
    monkeypatch.setattr(anatomy_models, "supersynth_home", lambda: tmp_path)
    labels = np.zeros((80, 80, 80), np.uint16)
    labels[33, 32, 33] = 16
    labels[36, 32, 36] = 902
    labels[44, 58, 40] = 911
    labels[49, 56, 42] = 915
    path = tmp_path / "labels_1mm.nii.gz"
    nib.save(nib.Nifti1Image(labels, row["affine"]), path)
    fake = {"output": str(tmp_path), "output_sha256": file_hash(path), "labels_present": [0, 16, 902, 911, 915]}
    monkeypatch.setattr(head_segmentation, "segment_head", lambda *args, **kwargs: fake)

    record = anatomy_models.reconstruct(c, "head", "supersynth", full_head_confirmed=True, include_mri_skull=False)
    anatomy_models.import_proposal(c, "head", record)
    added = [item for item in c.document["layers"] if item["id"] != "brain"]
    assert {item["kind"] for item in added} == {"brainstem", "artery", "scalp"}
    artery = next(item for item in added if item["kind"] == "artery")
    assert artery["role"] == "context" and not vascular_evidence_supported(c, artery)
    assert artery["provenance"]["label_ontology"]["value"] == 902
    assert len({item["data"]["sha256"] for item in added}) == 1
    anatomy_models.import_proposal(c, "head", record)
    assert len(c.document["layers"]) == 4


def test_dural_proxy_stops_at_open_axial_slices(tmp_path, monkeypatch):
    from resectionlab import anatomy_dural
    from resectionlab.patient_anatomy import reconstruction_lock

    monkeypatch.setattr(anatomy_dural, "reconstruction_lock", lambda: reconstruction_lock(tmp_path / "fixture-lock"))
    c = make_study(tmp_path)
    x, y, z = np.ogrid[:32, :32, :24]
    radius = np.sqrt((x - 16) ** 2 + (y - 16) ** 2)
    shell = (radius >= 10) & (radius <= 12) & (z >= 4) & (z < 16)
    hu = np.where(shell, 700, 0).astype(np.float32)
    row = source(c, tmp_path, "ct", hu, modality="CT", spacing=1.5)
    row["intensity_units"] = "HU"
    brain = ((radius < 9) & (z >= 4) & (z < 20)).astype(np.uint8)
    c.add_layer(brain, row["affine"], layer_id="brain", source_series="ct", kind="brain",
                role="context", color="#334456")
    record = anatomy_models.reconstruct(c, "ct", "dural-envelope")
    assert record["availability"] == "available"
    assert set(record["supported_axial_indices"]) == set(range(4, 16))
    key = anatomy_models.import_proposal(c, "ct", record)
    proxy = c.layer(key)
    assert proxy["label"] == "Estimated dural envelope — geometric context"
    assert proxy["role"] == "context" and proxy["provenance"]["kind"] == "geometric_estimate"
    assert not c.mask(proxy)[:, :, 16:].any()


def test_prepared_regions_keep_established_structural_reference(tmp_path, monkeypatch):
    c = make_study(tmp_path)
    source(c, tmp_path, "first", np.ones((8, 8, 8), np.float32), modality="T1CE")
    source(c, tmp_path, "additional", np.ones((8, 8, 8), np.float32), modality="T1")
    monkeypatch.setattr(registry, "model_environment", lambda *args, **kw: {})
    stages = {item["id"]: item for item in registry.preparation_model_jobs(c)}
    assert stages["brain:first"]["series_id"] == stages["brain-regions"]["series_id"] == "first"
    assert "brain:additional" not in stages


def test_bounded_region_field_contains_complete_estimated_brain(tmp_path):
    c = make_study(tmp_path)
    row = source(c, tmp_path, "mri", np.ones((80, 80, 80), np.float32), spacing=3)
    brain = c.add_layer(ellipsoid([19, 23, 19]).astype(np.uint8), row["affine"], layer_id="brain",
                        source_series=row["id"], kind="brain", role="context", color="#334455")
    assert registry.synthseg_requires_roi(row)
    roi = anatomy_models._synthseg_brain_bounds(c, row, brain)
    bounds = roi["bounds_native_exclusive"]
    slices = tuple(slice(lo, hi) for lo, hi in bounds)
    assert c.mask(brain)[slices].sum() == c.mask(brain).sum()
    assert np.prod(roi["padded_working_shape"]) <= registry.SYNTHSEG_MAX_PADDED_CELLS
    cropped = nib.load(anatomy_models._write_synthseg_roi(c, row, c.path(row["original"]["path"]), tmp_path, roi))
    assert np.array_equal(cropped.affine[:3, 3], np.array(bounds)[:, 0] * 3)
    assert cropped.shape == tuple(hi - lo for lo, hi in bounds)


def test_migrated_cache_is_bound_to_native_source_and_ontology(tmp_path):
    import json
    from resectionlab.anatomy_model_cache import reusable_model_layer

    c = make_study(tmp_path)
    source_row = source(c, tmp_path, "mri", np.ones((8, 8, 8), np.float32))
    labels = np.zeros((8, 8, 8), np.uint16)
    labels[2:4, 2:4, 2:4] = 16
    path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), path)
    output = c.original(path)
    layer = c.add_layer(labels, np.eye(4), layer_id="region", source_series="mri",
                        kind="brainstem", role="context", color="#334455", label_value=16)
    identity = {"source_sha256": source_row["original"]["sha256"], "model_sha256": head_segmentation.MODEL_SHA,
                "parameters": head_segmentation.PARAMETERS}
    receipt_path = tmp_path / "head-receipt.json"
    receipt_path.write_text(json.dumps({"status": "succeeded", "identity": identity, "output_sha256": output["sha256"]}))
    receipt = c.original(receipt_path)
    p = {
        "model": "SuperSynth", "preparation_cache_verified": True, "source": output,
        "model_identity": identity, "inference_receipt_original": receipt,
        "inference_receipt_sha256": receipt["sha256"], "output_sha256": output["sha256"],
        "source_sha256": source_row["original"]["sha256"], "source_array_sha256": source_row["data"]["sha256"],
        "prepared_array_sha256": layer["data"]["sha256"], "evidence_use": "context_only", "clinical_validation": False,
        "label_ontology": {"name": "Brainstem", "value": 16, "lut_sha256": head_segmentation.ASSETS["FreeSurferColorLUT.txt"]},
    }
    layer["provenance"] = p
    assert reusable_model_layer(c, source_row, {"brainstem"}, model="supersynth") == layer
    p["label_ontology"]["value"] = 17
    assert reusable_model_layer(c, source_row, {"brainstem"}, model="supersynth") is None
    p["label_ontology"]["value"] = 16
    p["source_array_sha256"] = "0" * 64
    assert reusable_model_layer(c, source_row, {"brainstem"}, model="supersynth") is None


def test_poc_model_key_survives_model_cache_and_python_location_changes(tmp_path, monkeypatch):
    from shutil import copy2

    from resectionlab import anatomy_preparation as preparation, data_root, poc_models
    from resectionlab.anatomy_model_cache import LEGACY_REGISTRY_SHA
    from resectionlab.anatomy_model_identity import LEGACY_ADAPTER_SHA, LEGACY_POC_LAUNCHER_SHA

    repository = tmp_path / "checkout"
    external = tmp_path / "private-drive"
    relative = "poc-totalsegmentator/weights/class/fold_0/checkpoint_final.pth"
    old_weight = external / "models" / relative
    weight = repository / "data/models" / relative
    old_runtime = external / "runtime/python"
    runtime = tmp_path / "new-runtime/interpreter"
    for target, contents in ((old_weight, b"stable weights"), (old_runtime, b"stable executable")):
        target.parent.mkdir(parents=True)
        target.write_bytes(contents)
    for original, relocated in ((old_weight, weight), (old_runtime, runtime)):
        relocated.parent.mkdir(parents=True)
        copy2(original, relocated)

    monkeypatch.setattr(registry, "ROOT", repository)
    monkeypatch.setattr(registry.sys, "platform", "darwin")
    monkeypatch.setattr(data_root, "checked_root", lambda: external)
    monkeypatch.setenv("RESSECTIONLAB_MODEL_ROOT", str(external / "models"))
    monkeypatch.setattr(poc_models, "RUNTIME", old_runtime)
    model = "totalsegmentator-skull"
    old_portable = registry.model_environment(model)
    assert "file_paths" not in old_portable
    assert str(old_runtime) not in str(old_portable)

    c = make_study(tmp_path)
    scan = source(c, tmp_path, "cta", np.ones((8, 8, 8), dtype=np.float32), modality="CTA")
    job = dict(id="ct-skull", series_id="cta", model=model, status="ready", options={})
    old_environment = {"files": {
        str(old_runtime): file_hash(old_runtime), str(old_weight): file_hash(old_weight),
        poc_models.__file__: LEGACY_POC_LAUNCHER_SHA, anatomy_models.__file__: LEGACY_ADAPTER_SHA,
        registry.__file__: LEGACY_REGISTRY_SHA,
    }}
    old_key = preparation._model_input(c, {**job, "environment": old_environment})
    layer = c.add_layer(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4), layer_id="prepared-skull",
                        source_series=scan["id"], kind="skull", role="context", color="#aaaaaa",
                        provenance={"inference": {"model_id": model, "command": [str(old_runtime)],
                                                  "weights": [{"path": str(old_weight), "sha256": file_hash(old_weight)}]}})
    layer["preparation_input"] = old_key
    receipt = {"ct-skull": {"inputKey": old_key, "status": "cached",
                            "outputs": [{"id": layer["id"], "sha256": layer["data"]["sha256"]}]}}
    old_prediction = preparation._stage(c, {**job, "environment": old_portable}, {})
    old_runtime.unlink()
    old_weight.unlink()
    monkeypatch.delenv("RESSECTIONLAB_MODEL_ROOT")
    monkeypatch.setattr(poc_models, "RUNTIME", runtime)

    assert registry.model_environment(model) == old_portable
    relocated = {**job, "environment": registry.model_environment(model, include_paths=True)}
    cached = preparation._stage(c, relocated, receipt)
    assert cached["status"] == "cached"
    assert cached["inputKey"] == old_prediction["inputKey"] != old_key
    assert "file_paths" not in cached["environment"]
    assert preparation._stage(c, {**relocated, "options": {"changed": True}}, receipt)["status"] == "ready"
    modified_environment = {**relocated["environment"],
                            "files": {**relocated["environment"]["files"], "registry": "a" * 64}}
    assert preparation._stage(c, {**relocated, "environment": modified_environment}, receipt)["status"] == "ready"

    weight.write_bytes(b"different weight bytes")
    changed = {**job, "environment": registry.model_environment(model, include_paths=True)}
    assert preparation._stage(c, changed, receipt)["status"] == "ready"
    assert preparation._stage(c, changed, {})["inputKey"] != cached["inputKey"]


def test_synthseg_identity_preserves_distinct_bundle_paths_and_class_config(tmp_path, monkeypatch):
    from shutil import copytree

    home = tmp_path / "installed"
    relocated = tmp_path / "restaged"
    runtime = tmp_path / "python"
    runtime.write_bytes(b"test executable")
    paths = {
        "scripts/commands/SynthSeg_predict.py": b"prediction code",
        "models/synthseg_1.0.h5": b"model v1",
        "models/synthseg_robust_2.0.h5": b"model v2",
        "models/synthseg_parc_2.0.h5": b"parcellation model",
        "models/synthseg_qc_2.0.h5": b"qc model",
        "data/labels_classes_priors/synthseg_segmentation_labels.npy": b"label ontology",
        "SynthSeg/a/utils.py": b"first utils",
        "ext/b/utils.py": b"other utils",
    }
    for name, contents in paths.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    copytree(home, relocated)
    monkeypatch.setenv("RESECTIONLAB_SYNTHSEG_HOME", str(home))
    monkeypatch.setenv("RESECTIONLAB_SYNTHSEG_PYTHON", str(runtime))
    original = registry.model_environment("synthseg", synthseg_version=1)
    assert registry.model_environment("synthseg", synthseg_version=2) != original
    monkeypatch.setenv("RESECTIONLAB_SYNTHSEG_HOME", str(relocated))
    assert registry.model_environment("synthseg", synthseg_version=1) == original

    (relocated / "SynthSeg/a/utils.py").write_bytes(paths["ext/b/utils.py"])
    (relocated / "ext/b/utils.py").write_bytes(paths["SynthSeg/a/utils.py"])
    swapped = registry.model_environment("synthseg", synthseg_version=1)
    assert sorted(swapped["files"].values()) == sorted(original["files"].values())
    assert swapped != original
    copytree(home, relocated, dirs_exist_ok=True)
    (relocated / "data/labels_classes_priors/synthseg_segmentation_labels.npy").write_bytes(b"changed ontology")
    assert registry.model_environment("synthseg", synthseg_version=1) != original
