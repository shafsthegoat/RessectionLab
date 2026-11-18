"""Publisher sequence, brain mask, and per-run ONNX staging contracts."""

import configparser
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_models
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock


def image(path, data):
    volume = nib.Nifti1Image(np.asarray(data), np.eye(4))
    volume.header.set_xyzt_units("mm")
    nib.save(volume, path)
    return path


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(anatomy_models, "reconstruction_lock", lambda: reconstruction_lock(tmp_path / "lock"))
    c = PatientImagingCollection.create(tmp_path / "native", "test-glioma")
    source = image(tmp_path / "t1ce.nii.gz", np.ones((8, 8, 8), np.float32))
    c.add_series(source, series_id="T1CE", modality="T1CE")
    other = image(tmp_path / "t2.nii.gz", np.ones((8, 8, 8), np.float32))
    c.add_series(other, series_id="T2", modality="T2")
    mask = np.ones((8, 8, 8), np.uint8)
    c.add_layer(mask, np.eye(4), layer_id="brain-t1ce", source_series="T1CE",
                kind="brain", role="context", color="#aaaaaa")
    c.add_layer(mask, np.eye(4), layer_id="brain-t2", source_series="T2",
                kind="brain", role="context", color="#aaaaaa")
    home = tmp_path / "models"
    folder = home / "MRI_GBM"
    folder.mkdir(parents=True)
    (folder / "pipeline.json").write_text(json.dumps({
        "1": {"model": "MRI_Brain", "target": ["Brain"], "inputs": {"0": {"sequence": "T1-CE"}}},
        "2": {"model": "MRI_GBM", "target": ["Tumor"], "inputs": {"0": {"sequence": "T1-CE"}}},
    }))
    (folder / "pre_processing.ini").write_text("[PreProcessing]\nbackground_cropping=brain_mask\n")
    (folder / "model_config.txt").write_text("model test")
    (folder / "model.onnx").write_bytes(b"model fixture")
    packages = tmp_path / "packages/raidionicsseg"
    packages.mkdir(parents=True)
    (packages / "fit.py").write_text("# runtime fixture\n")
    monkeypatch.setenv("RESECTIONLAB_RAIDIONICS_MODELS", str(home))
    monkeypatch.setenv("RESECTIONLAB_RAIDIONICS_PACKAGES", str(packages.parent))
    return c, folder


def test_gbm_requires_publisher_sequence_and_matching_brain(tmp_path, monkeypatch):
    c, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="publisher-declared input sequence"):
        anatomy_models.reconstruct(c, "T2", "glioma-gbm", brain_layer_id="brain-t2")
    with pytest.raises(ValueError, match="requires a brain mask"):
        anatomy_models.reconstruct(c, "T1CE", "glioma-gbm")
    with pytest.raises(ValueError, match="same source scan"):
        anatomy_models.reconstruct(c, "T1CE", "glioma-gbm", brain_layer_id="brain-t2")
    assert not list((c.root / "reconstructions").glob("*/receipt.json"))


def test_gbm_stages_exact_fold_and_imports_separate_context_estimate(tmp_path, monkeypatch):
    c, folder = fixture(tmp_path, monkeypatch)
    before = c.identity()

    def bounded(_command, directory, **_kwargs):
        ini = configparser.ConfigParser()
        ini.read(Path(directory) / "raidionics.ini")
        staged = Path(ini["System"]["model_folder"])
        assert (staged / "fold-0/model.onnx").resolve() == (folder / "model.onnx").resolve()
        assert (staged / "pre_processing.ini").read_bytes() == (folder / "pre_processing.ini").read_bytes()
        brain = nib.load(ini["Neuro"]["brain_segmentation_filename"])
        assert brain.shape == (8, 8, 8) and np.asanyarray(brain.dataobj).all()
        probability = np.zeros((8, 8, 8), np.float32)
        probability[2:4, 2:4, 2:4] = .8
        output = Path(ini["System"]["output_folder"])
        image(output / "pred_Tumor.nii.gz", probability)
        return {"elapsed_seconds": .1, "peak_sampled_rss_bytes": 1000}

    monkeypatch.setattr(anatomy_models, "run_bounded", bounded)
    record = anatomy_models.reconstruct(c, "T1CE", "glioma-gbm", brain_layer_id="brain-t1ce")
    assert record["foreground_voxels"] == 8 and record["status"] == "proposal"
    assert record["brain_mask"]["layer_id"] == "brain-t1ce"
    assert c.identity() == before  # reconstruction alone does not admit a layer
    key = anatomy_models.import_proposal(c, "T1CE", record)
    assert c.layer(key)["kind"] == "tumor" and c.layer(key)["role"] == "context"


def test_brain_crop_preserves_identical_old_scores_and_corrected_positive_labels(tmp_path, monkeypatch):
    from resectionlab.anatomy_evidence import inference_bounds

    c, _ = fixture(tmp_path, monkeypatch)
    brain_row = c.layer("brain-t1ce")
    brain = np.zeros((8, 8, 8), np.uint8)
    brain[2:6, 1:7, 2:7] = 1
    brain[0, 0, 0] = 1
    brain_row["data"] = c.put_array(brain)

    def bounded(_command, directory, **_kwargs):
        probability = np.zeros((8, 8, 8), np.float32)
        probability[2:4, 2:4, 2:4] = .8
        image(directory / "output" / "pred_Tumor.nii.gz", probability)
        return {"elapsed_seconds": .1, "peak_sampled_rss_bytes": 1000}

    monkeypatch.setattr(anatomy_models, "run_bounded", bounded)
    record = anatomy_models.reconstruct(c, "T1CE", "glioma-gbm", brain_layer_id="brain-t1ce")
    key = anatomy_models.import_proposal(c, "T1CE", record)
    row = c.layer(key)
    assert inference_bounds(row) == ((2, 1, 2), (5, 6, 6))

    # Represent an earlier descriptor and an edit beyond its model field.
    row["coverage_fields"].pop("inference_bounds_voxel_inclusive")
    row["coverage_fields"].pop("model_inference_bounds_voxel_inclusive")
    row["id"] = "previous-proposal"
    c.save("preserve earlier proposal")
    corrected = c.mask(row)
    corrected[7, 7, 7] = True
    c.commit_edit(row["id"], corrected, author="test reviewer", expected=c.identity(), operations=[])
    before = row["data"].copy()
    old_record = {k: v for k, v in record.items() if k != "brain_input_label_value"}
    assert anatomy_models.import_proposal(c, "T1CE", old_record) == "previous-proposal"
    assert row["data"] == before and row["revision"] == 1 and row["edits"][0]["author"] == "test reviewer"
    assert row["coverage_fields"]["model_inference_bounds_voxel_inclusive"] == [[2, 1, 2], [5, 6, 6]]
    assert inference_bounds(row) is None and c.mask(row)[7, 7, 7]

    # A genuinely different retained score map can support a different edit,
    # even if the current binary threshold yields the same contour.
    new_probability = nib.load(record["probability"]["path"]).get_fdata(dtype=np.float32)
    new_probability[new_probability > 0] = .9
    path = image(tmp_path / "new-scores.nii.gz", new_probability)
    alternative = {**record, "probability": {"path": str(path), "sha256": anatomy_models.file_hash(path)}}
    alternate_id = anatomy_models.import_proposal(c, "T1CE", alternative)
    assert alternate_id != row["id"]
    assert len([item for item in c.document["layers"] if item["kind"] == "tumor"]) == 2
    assert anatomy_models.import_proposal(c, "T1CE", alternative) == alternate_id


def test_gbm_declines_brain_output_with_cropped_geometry(tmp_path, monkeypatch):
    c, _ = fixture(tmp_path, monkeypatch)
    c.add_layer(np.ones((4, 4, 4), np.uint8), np.eye(4), layer_id="cropped",
                source_series="T1CE", kind="brain", role="context", color="#aaaaaa")
    with pytest.raises(ValueError, match="match the selected native scan geometry"):
        anatomy_models.reconstruct(c, "T1CE", "glioma-gbm", brain_layer_id="cropped")


def test_warm_preparation_backfills_crop_without_running_inference(tmp_path, monkeypatch):
    from resectionlab import anatomy_model_registry, anatomy_preparation
    from resectionlab.anatomy_evidence import inference_bounds

    c, _ = fixture(tmp_path, monkeypatch)
    brain = c.layer("brain-t1ce")
    brain["data"] = c.put_array(np.pad(np.ones((4, 4, 4), np.uint8), 2))

    def bounded(_command, directory, **_kwargs):
        probability = np.zeros((8, 8, 8), np.float32)
        probability[2:4, 2:4, 2:4] = .8
        image(directory / "output" / "pred_Tumor.nii.gz", probability)
        return {"elapsed_seconds": .1, "peak_sampled_rss_bytes": 1000}

    monkeypatch.setattr(anatomy_models, "run_bounded", bounded)
    record = anatomy_models.reconstruct(c, "T1CE", "glioma-gbm", brain_layer_id="brain-t1ce")
    key = anatomy_models.import_proposal(c, "T1CE", record)
    row = c.layer(key)
    for field in ("model_inference_bounds_voxel_inclusive", "inference_bounds_voxel_inclusive"):
        row["coverage_fields"].pop(field)
    original_mask, retained_score = row["data"].copy(), row["probability"].copy()
    c.save("earlier native study with retained model scores")
    Path(record["path"]).unlink()
    Path(record["probability"]["path"]).unlink()

    def no_inference(*args, **kwargs):
        raise AssertionError("Warm coverage upgrade must not invoke a model")

    monkeypatch.setattr(anatomy_models, "reconstruct", no_inference)
    monkeypatch.setattr(anatomy_model_registry, "preparation_model_jobs", lambda *_a, **_kw: [])
    monkeypatch.setattr(anatomy_preparation, "alignment_pairs", lambda *_a, **_kw: [])
    plan = anatomy_preparation.preparation_plan(c)
    assert plan["stages"][0]["status"] == "ready" and plan["stages"][0]["model"] == "restore-tumor-crop-coverage"
    result = anatomy_preparation.prepare_anatomy(c, plan_id=plan["id"])
    assert result["outcomes"][0]["status"] == "complete"
    reopened = PatientImagingCollection(c.root)
    assert not anatomy_preparation.preparation_plan(reopened)["stages"]
    row = reopened.layer(key)
    assert row["data"] == original_mask and row["probability"] == retained_score
    assert inference_bounds(row) == ((2, 2, 2), (5, 5, 5))
