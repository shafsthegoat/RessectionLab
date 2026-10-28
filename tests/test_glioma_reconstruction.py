"""Publisher sequence, brain mask, and per-run ONNX staging contracts."""

import configparser
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_models
from resectionlab.patient_anatomy import PatientImagingCollection


def image(path, data):
    volume = nib.Nifti1Image(np.asarray(data), np.eye(4))
    volume.header.set_xyzt_units("mm")
    nib.save(volume, path)
    return path


def fixture(tmp_path, monkeypatch):
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
