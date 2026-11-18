"""One bounded, source-bound CPU retry for a proven MPS allocator failure."""

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_models, anatomy_preparation, anatomy_synthstrip_fallback, poc_models
from resectionlab.patient_anatomy import PatientImagingCollection, digest, file_hash


def setup(tmp_path, monkeypatch, *, oom=True):
    c = PatientImagingCollection.create(tmp_path / "study", "fixture")
    path = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4)), path)
    source = c.add_series(path, series_id="t1", modality="T1")
    c.save("source fixture")
    weights = ["1" * 64]
    env = {"files": {"weights/model.pt": weights[0]}}
    monkeypatch.setattr(anatomy_synthstrip_fallback, "model_environment", lambda *_, **__: env)
    monkeypatch.setattr(anatomy_preparation, "_jobs", lambda *_: [{
        "id": "brain:t1", "label": "Brain", "capability": "brain", "series_id": "t1",
        "model": "synthstrip", "options": {}, "environment": env, "status": "ready", "depends_on": [],
    }])
    gpu_input = anatomy_preparation.preparation_plan(c)["stages"][0]["inputKey"]
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(anatomy_synthstrip_fallback, "_cache_roots", lambda: [cache_root])
    case = "anatomy-" + digest(c.document["patient_id"])[:16]
    config = poc_models.MODEL_CONFIG["synthstrip-main"]
    gpu = {"model_id": "synthstrip-main", "device": "mps", "image_sha256": source["original"]["sha256"],
           "weights": weights, "parameters": config["parameters"]}
    key = hashlib.sha256(json.dumps(gpu, sort_keys=True).encode()).hexdigest()[:20]
    attempt = cache_root / case / "synthstrip-main" / key
    attempt.mkdir(parents=True)
    log = attempt / "inference.log"
    log.write_text("RuntimeError: MPS backend out of memory (MPS allocated: 5.17 GiB, max allowed: 6.00 GiB).\n"
                   if oom else "RuntimeError: unrelated failure.\n")
    (attempt / "inference.json").write_text(json.dumps({
        "identity": gpu, "status": "failed", "failure": "worker_exit_1", "device": "mps", "case": case,
        "model_id": "synthstrip-main", "model": config,
        "input": {"sha256": gpu["image_sha256"], "modality": "T1",
                  "shape": source["data"]["shape"], "affine_ras_mm": source["affine"]},
        "cache_key": key,
    }))
    prior = {"status": "failed", "reason": f"RuntimeError: worker_exit_1; inspect {log}",
             "inputKey": gpu_input, "outputs": [], "elapsedSeconds": 5.1}
    anatomy_preparation._checkpoint(c, {"brain:t1": prior})
    return c, path, prior, log


@pytest.mark.parametrize("cpu_fails", [False, True])
def test_retry_uses_cpu_once_and_preserves_mps_checkpoint(tmp_path, monkeypatch, cpu_fails):
    c, path, prior, log = setup(tmp_path, monkeypatch)
    initial_log_hash = file_hash(log)
    attempts = []
    monkeypatch.setattr(anatomy_models, "reconstruct", lambda *_a, **_kw: pytest.fail("MPS must not repeat"))

    def worker(*args, **kw):
        assert kw["device"] == "cpu" and kw["timeout"] == 600
        assert anatomy_preparation._receipts(c)["brain-cpu:t1"]["reason"] == "The one CPU retry was started"
        attempts.append(kw["device"])
        if cpu_fails:
            raise RuntimeError("the worker exceeded its RSS guard")
        foreground = np.zeros((8, 8, 8), np.uint8)
        foreground[2:6, 2:6, 2:6] = 1
        output = tmp_path / "brain_mask.nii.gz"
        nib.save(nib.Nifti1Image(foreground, np.eye(4)), output)
        return {
            "device": "cpu", "input": {"sha256": file_hash(path)},
            "model": {"version": "fixture", "parameters": {"threads": 2}},
            "outputs": [{"name": output.name, "path": str(output), "sha256": file_hash(output),
                         "foreground_voxels": int(foreground.sum())}],
        }

    monkeypatch.setattr(poc_models, "run", worker)
    stages = anatomy_preparation.preparation_plan(c)["stages"]
    assert [(s["id"], s["status"]) for s in stages] == [("brain:t1", "failed"), ("brain-cpu:t1", "retry_available")]
    assert stages[0]["inputKey"] != stages[1]["inputKey"] and stages[1]["options"]["device"] == "cpu"
    anatomy_preparation.prepare_anatomy(c)
    assert not attempts
    anatomy_preparation.prepare_anatomy(c, retry_failed=True)
    for _ in range(2):
        anatomy_preparation.prepare_anatomy(c, retry_failed=True)
    assert attempts == ["cpu"]
    receipts = anatomy_preparation._receipts(c)
    assert receipts["brain:t1"] == prior and file_hash(log) == initial_log_hash
    assert receipts["brain-cpu:t1"]["status"] == ("failed" if cpu_fails else "cached")
    if not cpu_fails:
        row = c.document["layers"][0]
        assert row["label"] == "Brain envelope · SynthStrip CPU" and row["provenance"]["device"] == "cpu"
        assert row["provenance"]["fallback_from"]["log_sha256"] == initial_log_hash


def test_unrelated_mps_failure_has_no_cpu_retry(tmp_path, monkeypatch):
    c, _, _, _ = setup(tmp_path, monkeypatch, oom=False)
    stages = anatomy_preparation.preparation_plan(c)["stages"]
    assert len(stages) == 1 and stages[0]["status"] == "failed" and "retryable" not in stages[0]
