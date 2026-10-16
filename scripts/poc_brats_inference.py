#!/usr/bin/env python3
"""Pinned MONAI BraTS inference for an explicitly registered four-MRI case.

Run --download once, then run with the source acquisition manifest. No supplied
annotation is read by inference. Outputs stay on the imported preprocessed grid.
This is a research demonstration, not a clinical or held-out accuracy result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import urllib.request

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "370f7f9d062745fbac445e7fe6d6616d35df04ec"
MODEL_ID = "monai-brats-0.5.4"
WEIGHT_SHA256 = "860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae"
MODEL_DIR = ROOT / "data/models/poc-monai-brats-0.5.4"
CHANNELS = ("T1c", "T1", "T2", "FLAIR")
OUTPUTS = ("monai_tumor_labels.nii.gz", "monai_whole_tumor_mask.nii.gz", "monai_tumor_core_mask.nii.gz", "monai_enhancing_tumor_mask.nii.gz")
PARAMETERS = {
    "channel_order": list(CHANNELS), "spacing_mm": [1, 1, 1],
    "normalization": "MONAI NormalizeIntensity(nonzero=True, channel_wise=True)",
    "roi_size": [128, 128, 128], "sw_batch_size": 1, "overlap": 0.5,
    "blend_mode": "constant", "precision": "float32", "threshold": 0.5,
    "postprocessing": "upstream priority: ET=4, else TC=1, else WT=2, else background=0",
    "tta": False, "registration_performed": False, "resampling_performed": False,
    "adaptation": "128-cubed tiles instead of upstream 240x240x160; CPU accumulation; no CUDA AMP",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def acquire() -> dict:
    files = []
    for name, maximum, expected in [
        ("models/model.pt", 18_840_620, WEIGHT_SHA256),
        ("LICENSE", 20_000, None), ("docs/README.md", 20_000, None),
        ("docs/data_license.txt", 10_000, None), ("configs/inference.json", 20_000, None),
        ("configs/metadata.json", 20_000, None), ("scripts/prepare_datalist.py", 10_000, None),
    ]:
        path = MODEL_DIR / name
        url = f"https://huggingface.co/MONAI/brats_mri_segmentation/resolve/{REVISION}/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            count = 0
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > maximum:
                        raise ValueError(f"Download budget exceeded: {name}")
                    stream.write(block)
            if expected and sha256(temporary) != expected:
                raise ValueError(f"Publisher SHA256 mismatch: {name}")
            temporary.replace(path)
        if path.stat().st_size > maximum or (expected and sha256(path) != expected):
            raise ValueError(f"Invalid cached model asset: {name}")
        files.append({"path": str(path.resolve()), "url": url, "bytes": path.stat().st_size,
                      "sha256": sha256(path), "publisher_sha256": expected})
    record = {"model_id": MODEL_ID, "version": "0.5.4", "revision": REVISION,
              "source": "https://huggingface.co/MONAI/brats_mri_segmentation", "license": "Apache-2.0",
              "intended_use": "Example; not for diagnostic purposes", "files": files,
              "training_data_license_notice": "docs/data_license.txt retained alongside model",
              "retrieved_at": datetime.now(timezone.utc).isoformat()}
    write_json(MODEL_DIR / "acquisition.json", record)
    return record


def verify_geometry(images) -> None:
    reference = images[0]
    for image in images:
        if len(image.shape) != 3 or image.shape != reference.shape:
            raise ValueError("All four MRI volumes must have one identical 3D shape")
        if not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError("MRI affine mismatch: register the images explicitly before inference")
        if not np.allclose(nib.affines.voxel_sizes(image.affine), [1, 1, 1], atol=1e-4, rtol=0):
            raise ValueError("This pinned bundle requires 1 mm isotropic input")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
            raise ValueError("Invalid NIfTI affine")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if qcode and scode and not np.allclose(qform, sform, atol=1e-5, rtol=0):
            raise ValueError("Conflicting qform and sform: resolve the physical frame before inference")


def prepare_inputs(manifest_path: Path, data_root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("spatial_frame", {}).get("cross_modality_geometry_equal"):
        raise ValueError("Source manifest must explicitly declare its registered common grid")
    case = manifest["case_id"]
    if not case or Path(case).name != case or case in {".", ".."}:
        raise ValueError("Case ID must be a safe directory name")
    inputs, images = [], []
    for channel in CHANNELS:
        matches = [row for row in manifest["files"] if row.get("modality") == channel
                   and row.get("role") == "structural_mri"]
        if len(matches) != 1:
            raise ValueError(f"Exactly one acquired {channel} volume is required")
        row = matches[0]
        path = (data_root / row["path"]).resolve()
        if not path.is_relative_to(data_root.resolve()):
            raise ValueError("Source path escapes the declared data root")
        digest = sha256(path)
        if digest != row["sha256"]:
            raise ValueError(f"Source checksum mismatch: {channel}")
        image = nib.load(path)
        images.append(image)
        inputs.append({"modality": channel, "path": str(path), "sha256": digest,
                       "source_url": row.get("source_url"), "source_remote_path": row.get("source_remote_path"),
                       "provenance": row.get("provenance"), "shape": list(image.shape),
                       "affine_ras_mm": image.affine.tolist()})
    verify_geometry(images)
    return manifest, inputs


def labels_from_channels(channels: np.ndarray) -> np.ndarray:
    """Use the exact class precedence in the upstream inference config."""
    tc, wt, et = channels
    return np.where(et, 4, np.where(tc, 1, np.where(wt, 2, 0))).astype(np.uint8)


def save_mask(data, reference, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, path)


def worker(request: dict) -> None:
    import torch
    from monai.inferers import SlidingWindowInferer
    from monai.networks.nets import SegResNet
    from monai.transforms import NormalizeIntensity

    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = request["device"]
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; rerun with --device cpu for explicit fallback")
        torch.mps.set_per_process_memory_fraction(0.50)
    started = time.monotonic()
    images = [nib.load(item["path"]) for item in request["inputs"]]
    verify_geometry(images)
    values = np.stack([image.get_fdata(dtype=np.float32) for image in images])
    if not np.isfinite(values).all() or any(not np.any(channel) for channel in values):
        raise ValueError("MRI inputs must be finite and nonempty")
    tensor = NormalizeIntensity(nonzero=True, channel_wise=True)(values)[None]
    del values
    config = json.loads((MODEL_DIR / "configs/inference.json").read_text())["network_def"]
    if config.pop("_target_") != "SegResNet":
        raise ValueError("Unexpected network architecture")
    model = SegResNet(**config)
    state = torch.load(request["weights"]["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    samples = {"tensor_bytes_max_sampled": 0, "driver_bytes_max_sampled": 0, "tiles_completed": 0}

    def sample(*_):
        samples["tiles_completed"] += 1
        if device == "mps":
            torch.mps.synchronize()
            samples["tensor_bytes_max_sampled"] = max(samples["tensor_bytes_max_sampled"], torch.mps.current_allocated_memory())
            samples["driver_bytes_max_sampled"] = max(samples["driver_bytes_max_sampled"], torch.mps.driver_allocated_memory())
        print(json.dumps({"tile": samples["tiles_completed"], "elapsed_seconds": time.monotonic()-started}), flush=True)

    model.register_forward_hook(sample)
    inferer = SlidingWindowInferer(roi_size=PARAMETERS["roi_size"], sw_batch_size=1,
                                  overlap=0.5, mode="constant", sw_device=device, device="cpu")
    with torch.inference_mode():
        logits = inferer(tensor, model)
    # Sigmoid(logit) >= 0.5 is equivalent to logit >= 0, avoiding another volume.
    channels = logits[0].numpy() >= 0
    labels = labels_from_channels(channels)
    output = Path(request["output"])
    for name, data in zip(OUTPUTS, (labels, labels > 0, np.isin(labels, [1, 4]), labels == 4)):
        save_mask(data, images[0], output / name)
    write_json(output / "worker.json", {
        "compute_seconds": time.monotonic()-started, "mps_allocator_samples": samples,
        "raw_channel_foreground_voxels": {name: int(np.count_nonzero(data)) for name, data in zip(("TC", "WT", "ET"), channels)},
        "nested_channel_violations_before_upstream_label_priority": {
            "TC_outside_WT": int(np.count_nonzero(channels[0] & ~channels[1])),
            "ET_outside_TC": int(np.count_nonzero(channels[2] & ~channels[0])),
        },
    })


def validate_outputs(output: Path, reference) -> list[dict]:
    records = []
    for name in OUTPUTS:
        path = output / name
        mask = nib.load(path)
        verify_geometry([reference, mask])
        values = np.asanyarray(mask.dataobj)
        allowed = [0, 1, 2, 4] if name == OUTPUTS[0] else [0, 1]
        if not np.isin(values, allowed).all():
            raise ValueError(f"Invalid output labels: {name}")
        records.append({"path": str(path), "name": name, "sha256": sha256(path),
                        "foreground_voxels": int(np.count_nonzero(values)), "shape": list(mask.shape),
                        "affine_ras_mm": mask.affine.tolist(), "provenance": "model_inference_unreviewed"})
    if not records[1]["foreground_voxels"]:
        raise ValueError("Empty whole-tumor output; manual review required")
    return records


def run(args) -> dict:
    manifest, inputs = prepare_inputs(args.source_manifest, args.data_root)
    weights = MODEL_DIR / "models/model.pt"
    if not weights.exists() or sha256(weights) != WEIGHT_SHA256:
        raise ValueError("Pinned weights missing or changed; run --download")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "monai", "numpy", "nibabel")}
    model_config_sha = sha256(MODEL_DIR / "configs/inference.json")
    identity = {"model_id": MODEL_ID, "model_revision": REVISION, "weight_sha256": WEIGHT_SHA256,
                "inputs": [{"modality": row["modality"], "sha256": row["sha256"]} for row in inputs],
                "source_manifest_sha256": sha256(args.source_manifest), "parameters": PARAMETERS,
                "device": args.device, "package_versions": versions, "runner_sha256": sha256(Path(__file__)),
                "model_config_sha256": model_config_sha}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output = args.output_root.resolve() / manifest["case_id"] / MODEL_ID / key
    record_path = output / "inference.json"
    reference = nib.load(inputs[0]["path"])
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("status") == "succeeded" and record.get("identity") == identity:
            current = validate_outputs(output, reference)
            if current == record["outputs"]:
                print(json.dumps({"cache_hit": True, "record": str(record_path)}))
                return record
        raise ValueError(f"Existing incomplete or changed cache; preserved for inspection: {output}")
    if args.check_only:
        print(json.dumps({"check_only": True, "input_channels": list(CHANNELS), "cache_key": key,
                          "expected_output": str(output), "package_versions": versions}, indent=2))
        return {"status": "inputs_verified"}
    output.mkdir(parents=True, exist_ok=False)
    request = {"inputs": inputs, "device": args.device, "output": str(output),
               "weights": {"path": str(weights), "sha256": WEIGHT_SHA256, "bytes": weights.stat().st_size}}
    write_json(output / "request.json", request)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(output / "request.json")]
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       PYTORCH_ENABLE_MPS_FALLBACK="1", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    started, peak_rss, failure = time.monotonic(), 0, None
    with (output / "inference.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            status = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
            if status.stdout.strip():
                peak_rss = max(peak_rss, int(status.stdout.strip()) * 1024)
            if time.monotonic()-started > args.timeout or peak_rss > 7 * 1024**3:
                failure = "time_budget_exceeded" if time.monotonic()-started > args.timeout else "memory_budget_exceeded"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        process.wait()
    outputs = []
    if process.returncode:
        failure = failure or "inference_process_failed"
    else:
        try:
            outputs = validate_outputs(output, reference)
        except (ValueError, OSError) as error:
            failure = str(error)
    record = {
        "schema": "resectionlab-poc-inference-v1", "status": "failed" if failure else "succeeded",
        "failure": failure, "exit_code": process.returncode, "model_id": MODEL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(), "case": manifest["case_id"],
        "model": {"name": "MONAI BraTS MRI segmentation", "version": "0.5.4", "revision": REVISION,
                  "source": "https://huggingface.co/MONAI/brats_mri_segmentation", "license": "Apache-2.0",
                  "license_url": f"https://huggingface.co/MONAI/brats_mri_segmentation/blob/{REVISION}/LICENSE",
                  "parameters": PARAMETERS},
        "weights": [request["weights"]], "inputs": inputs, "outputs": outputs,
        "identity": identity, "cache_key": key, "device": args.device, "package_versions": versions,
        "elapsed_seconds": time.monotonic()-started, "peak_process_rss_bytes_sampled": peak_rss,
        "memory_caveat": "RSS sampled every 0.5 s; MPS allocator sampled after each tile; neither is an exact peak",
        "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": sha256(args.source_manifest)},
        "source_provenance": {key: manifest.get(key) for key in ("source_collection", "license", "mirror", "spatial_frame", "split")},
        "registration": "Source preprocessed co-registered grid; identical shapes/affines checked; no registration estimated here",
        "clinical_validation": "none", "supplied_annotation_used_for_inference": False,
        "host": platform.platform(), "command": command,
    }
    if (output / "worker.json").exists():
        record["worker"] = json.loads((output / "worker.json").read_text())
    write_json(record_path, record)
    print(json.dumps({"status": record["status"], "record": str(record_path), "elapsed_seconds": record["elapsed_seconds"]}))
    if failure:
        raise RuntimeError(f"{failure}; inspect {output / 'inference.log'}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "manifests/data_acquisition_ucsf.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/poc-models")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.download:
        print(json.dumps(acquire(), indent=2))
    elif args.worker:
        worker(json.loads(args.worker.read_text()))
    else:
        run(args)


if __name__ == "__main__":
    main()
