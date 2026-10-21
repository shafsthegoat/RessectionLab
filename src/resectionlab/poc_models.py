"""Explicit, cached local inference for the anatomical-workbench POC.

Source masks never enter this runner. Each completed record describes actual
model execution and native-grid geometry. Clinical accuracy is not established.
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
import subprocess
import sys
import time
import copy

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / ".tools/poc-model-runtime/venv/bin/python"
MODEL_CONFIG = {
    "totalsegmentator-skull": {
        "name": "TotalSegmentator", "version": "2.12.0 / Dataset297 v2.0.0-weights",
        "license": "Apache-2.0", "license_url": "https://github.com/wasserth/TotalSegmentator#subtasks",
        "modality": ["CT"], "requirements": "Acquired CT in Hounsfield units; fast model at 3 mm",
        "weights_glob": "data/models/poc-totalsegmentator/weights/**/checkpoint_final.pth",
        "parameters": {"task": "total", "fast": True, "tta": False, "skull_label": 91, "brain_label": 90},
    },
    "vesselfm-base": {
        "name": "vesselFM", "version": "base / dd70bbdd2b9de06b1316b4d1105bda9766c24fd7",
        "license": "vesselFM Open RAIL++-M; Attachment A: research purposes and non-commercial use only",
        "license_url": "https://huggingface.co/bwittmann/vesselFM/blob/dd70bbdd2b9de06b1316b4d1105bda9766c24fd7/LICENSE",
        "modality": ["MRA", "CTA"], "requirements": "Angiographic acquired image; isotropic resampling; 1st–99th percentile normalization",
        "weights_glob": "data/models/poc-vesselfm-base/vesselFM_base.pt",
        "parameters": {"patch_size": 128, "batch_size": 1, "overlap": 0.5, "threshold": 0.5, "tta": False, "postprocessing": False},
    },
    "synthstrip-main": {
        "name": "SynthStrip", "version": "1 / cf4bccf24875a47245b4df9fc9372d0f1c3d784f",
        "license": "Weights MIT (official offered choice); runner FreeSurfer Software License 1.0",
        "license_url": "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/",
        "modality": ["T1", "T1c", "T2", "FLAIR", "MRI"],
        "requirements": "One structural image; brain extraction, not skull bone",
        "weights_glob": "data/models/synthstrip-v1/synthstrip.1.pt",
        "parameters": {"border_mm": 1, "threads": 2},
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_mask(data, reference, destination):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), reference.affine, header)
    output.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output, destination)


def worker(request):
    import torch
    torch.set_num_threads(2)
    device = request["device"]
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; choose explicit CPU fallback")
        torch.mps.set_per_process_memory_fraction(0.55)
    source = Path(request["image"])
    destination = Path(request["output"])
    reference = nib.load(source)
    model_id = request["model_id"]
    started = time.monotonic()
    memory = {"tensor_bytes_max_sampled": 0, "driver_bytes_max_sampled": 0}

    def sample(*_):
        if device == "mps":
            memory["tensor_bytes_max_sampled"] = max(memory["tensor_bytes_max_sampled"], torch.mps.current_allocated_memory())
            memory["driver_bytes_max_sampled"] = max(memory["driver_bytes_max_sampled"], torch.mps.driver_allocated_memory())

    if model_id == "totalsegmentator-skull":
        os.environ["TOTALSEG_HOME_DIR"] = str(ROOT / "data/models/poc-totalsegmentator/config")
        os.environ["TOTALSEG_WEIGHTS_PATH"] = str(ROOT / "data/models/poc-totalsegmentator/weights")
        from totalsegmentator.config import setup_totalseg, set_config_key
        setup_totalseg()
        set_config_key("send_usage_stats", False)
        from totalsegmentator.python_api import totalsegmentator
        prediction = totalsegmentator(source, output=None, task="total", fast=True, device=device,
                                     nr_thr_resamp=1, nr_thr_saving=1, quiet=False)
        labels = np.asanyarray(prediction.dataobj)
        if prediction.shape != reference.shape or not np.allclose(prediction.affine, reference.affine, atol=1e-5):
            raise ValueError("TotalSegmentator did not restore native geometry")
        for name, label in [("skull", 91), ("brain", 90)]:
            write_mask(labels == label, reference, destination / f"{name}_mask.nii.gz")
        sample()
    elif model_id == "vesselfm-base":
        import SimpleITK as sitk
        from monai.networks.nets import DynUNet
        from monai.inferers import SlidingWindowInferer
        from monai.transforms import ScaleIntensityRangePercentiles
        image = sitk.ReadImage(str(source))
        spacing = np.array(image.GetSpacing())
        target_spacing = float(np.median(spacing)) if len(set(spacing)) == 2 else float(min(spacing))
        size = np.round(np.array(image.GetSize()) * spacing / target_spacing).astype(int)
        isotropic = sitk.Resample(image, list(map(int, size)), sitk.Transform(), sitk.sitkLinear,
                                 image.GetOrigin(), [target_spacing]*3, image.GetDirection(), 0., sitk.sitkFloat32)
        values = sitk.GetArrayFromImage(isotropic)
        values = ScaleIntensityRangePercentiles(1, 99, 0, 1, clip=True)(values)
        tensor = torch.as_tensor(values, dtype=torch.float32)[None, None]
        model = DynUNet(spatial_dims=3, in_channels=1, out_channels=1,
                       strides=[[1,1,1]]+[[2,2,2]]*5, kernel_size=[[3,3,3]]*6,
                       upsample_kernel_size=[[2,2,2]]*5,
                       filters=[32,64,128,256,320,320], res_block=True)
        model.load_state_dict(torch.load(request["weights"][0]["path"], map_location="cpu", weights_only=True))
        model.to(device).eval()
        model.register_forward_hook(sample)
        parameters = request.get("parameters", {})
        inferer = SlidingWindowInferer(roi_size=(128,)*3, sw_batch_size=1, overlap=0.5,
                                      mode=parameters.get("blend", "constant"), sw_device=device, device="cpu", progress=True)
        with torch.inference_mode():
            logits = inferer(tensor, model)
        probability = sitk.GetImageFromArray(logits[0,0].sigmoid().numpy())
        probability.CopyInformation(isotropic)
        native = sitk.Resample(probability, image, sitk.Transform(), sitk.sitkLinear, 0, sitk.sitkFloat32)
        values = sitk.GetArrayFromImage(native).transpose(2,1,0)
        output = nib.Nifti1Image(values, reference.affine)
        output.header.set_xyzt_units('mm')
        nib.save(output, destination / "vessels_probability.nii.gz")
        write_mask(values > parameters.get("threshold", .5), reference, destination / "vessels_mask.nii.gz")
    elif model_id == "synthstrip-main":
        from .brain_extraction import mps_runner_source
        original = ROOT / "data/models/synthstrip-v1/mri_synthstrip"
        runner = original
        if device == "mps":
            runner = destination / "synthstrip_mps.py"
            runner.write_text(mps_runner_source(original.read_text(), 6*1024**3))
        command = [str(RUNTIME), str(runner), "-i", str(source), "-m",
                   str(destination / "brain_mask.nii.gz"), "--model", request["weights"][0]["path"], "--threads", "2"]
        if device == "mps":
            command.append("--gpu")
        subprocess.run(command, check=True)
    else:
        raise ValueError("Unknown model")
    versions = {}
    for package in ("torch", "numpy", "scipy", "nibabel", "TotalSegmentator", "nnunetv2", "monai", "surfa", "SimpleITK"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    (destination / "worker.json").write_text(json.dumps({"compute_seconds": time.monotonic()-started,
                                                           "runtime_versions": versions,
                                                           "mps_allocator_samples": memory}, indent=2)+"\n")


def _run(image: Path, case: str, model_id: str, modality: str, device: str="mps", timeout: int=1800,
        *, experimental_t1ce=False, blend="constant", threshold=.5, cancelled=lambda: False,
        output_root: Path | None = None):
    image = image.resolve()
    model = copy.deepcopy(MODEL_CONFIG[model_id])
    if model_id == "vesselfm-base":
        if blend not in {"constant", "gaussian"} or not 0 < threshold < 1:
            raise ValueError("Invalid vessel inference parameters")
        model["parameters"].update(blend=blend, threshold=threshold, retain_probabilities=True)
        if modality == "T1CE" and experimental_t1ce:
            model["modality"] = [*model["modality"], "T1CE"]
            model["requirements"] += "; T1CE is experimental, vascular completeness unvalidated"
            model["parameters"]["input_domain"] = "T1CE-experimental"
    if modality not in model["modality"]:
        raise ValueError(f"{model_id} requires {model['modality']}, got {modality}")
    weights = [{"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size}
               for p in sorted(ROOT.glob(model["weights_glob"]))]
    if not weights:
        raise FileNotFoundError("Weights missing; run scripts/poc_model_assets.py")
    source_hash = sha256(image)
    runner_hash = sha256(Path(__file__))
    identity = {"image_sha256": source_hash, "weights": [w["sha256"] for w in weights],
                "model_id": model_id, "parameters": model["parameters"], "device": device, "runner_sha256": runner_hash}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    if not case or Path(case).name != case or case in {".", ".."}:
        raise ValueError("Case must be a single safe directory name")
    cache_root = ROOT / "outputs/poc-models" if output_root is None else Path(output_root).resolve()
    if output_root is not None and not cache_root.is_relative_to((ROOT / 'testcases').resolve()):
        raise ValueError('Explicit app-testing inference output must remain under testcases/')
    output = cache_root / case / model_id / key
    # Cancelled/failed stages are retained. Resume starts a fresh attempt and
    # searches successful attempts first; it never overwrites historical logs.
    for attempt in [output,*sorted(output.parent.glob(key+"-retry-*"))]:
        cached=attempt/"inference.json"
        if not cached.exists():
            continue
        record=json.loads(cached.read_text())
        if record.get("status") != "succeeded":
            continue
        members=[*record["outputs"],*([record["probability"]] if record.get("probability") else [])]
        if not all(Path(x["path"]).exists() and sha256(Path(x["path"]))==x["sha256"] for x in members):
            raise ValueError(f"Successful inference cache changed: {attempt}")
        print(json.dumps({"cache_hit":True,"record":str(cached)}))
        return record
    if output.exists():
        output=output.with_name(key+"-retry-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    record_path = output / "inference.json"
    output.mkdir(parents=True, exist_ok=False)
    request = {"image": str(image), "case": case, "model_id": model_id, "modality": modality,
               "device": device, "output": str(output), "weights": weights, "parameters": model["parameters"]}
    (output / "request.json").write_text(json.dumps(request, indent=2)+"\n")
    command = [str(RUNTIME), "-m", "resectionlab.poc_models", "--worker", str(output / "request.json")]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       PYTORCH_ENABLE_MPS_FALLBACK="1", MPLBACKEND="Agg", nnUNet_compile="false")
    start = time.monotonic()
    maximum_rss = 0
    failure = None
    with (output / "inference.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            # Include child SynthStrip process; excludes shared Metal allocations.
            listing = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, check=True)
            rows = [list(map(int, row.split())) for row in listing.stdout.splitlines() if row.strip()]
            family = {process.pid}
            for _ in range(5):
                family |= {pid for pid, ppid, rss in rows if ppid in family}
            maximum_rss = max(maximum_rss, sum(rss*1024 for pid, _, rss in rows if pid in family))
            if cancelled() or time.monotonic()-start > timeout or maximum_rss > 8*1024**3:
                import signal
                failure = "cancelled" if cancelled() else "time_budget_exceeded" if time.monotonic()-start > timeout else "memory_budget_exceeded"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        process.wait()
    reference = nib.load(image)
    outputs = []
    if process.returncode == 0:
        for mask_path in sorted(output.glob("*_mask.nii.gz")):
            mask = nib.load(mask_path)
            data = np.asanyarray(mask.dataobj)
            if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine, atol=1e-5):
                failure = "native_geometry_mismatch"
            if not np.isin(data, [0,1]).all() or not data.any():
                failure = "invalid_or_empty_mask"
            outputs.append({"path": str(mask_path), "sha256": sha256(mask_path), "name": mask_path.name,
                            "foreground_voxels": int(np.count_nonzero(data)), "shape": list(mask.shape),
                            "affine_ras_mm": mask.affine.tolist(), "provenance": "model_inference_unreviewed"})
    if not outputs:
        failure = failure or "no_saved_masks"
    probability = output / 'vessels_probability.nii.gz'
    probability_record = None if not probability.exists() else {"path": str(probability), "sha256": sha256(probability), "meaning": "uncalibrated_model_score"}
    record = {"schema": "resectionlab-poc-inference-v1", "status": "failed" if failure else "succeeded",
              "failure": failure, "exit_code": process.returncode, "created_at": datetime.now(timezone.utc).isoformat(),
              "case": case, "model_id": model_id, "model": model, "weights": weights,
              "input": {"path": str(image), "sha256": source_hash, "modality": modality,
                        "shape": list(reference.shape), "affine_ras_mm": reference.affine.tolist()},
              "cache_key": key, "identity": identity, "device": device, "elapsed_seconds": time.monotonic()-start,
              "peak_process_family_rss_bytes_sampled": maximum_rss,
              "memory_caveat": "RSS sampled every 0.5 s; MPS allocated memory separately sampled where available; neither is exact peak",
              "host": platform.platform(), "outputs": outputs, "probability": probability_record, "clinical_validation": "none", "command": command}
    worker_path = output / "worker.json"
    if worker_path.exists():
        record["worker"] = json.loads(worker_path.read_text())
    for line in (output / "inference.log").read_text().splitlines():
        if line.startswith("RESECTIONLAB_MPS_MEMORY "):
            record["synthstrip_mps_allocator_samples"] = json.loads(line.split(" ", 1)[1])
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"status": record["status"], "record": str(record_path), "elapsed_seconds": record["elapsed_seconds"]}))
    if failure:
        raise RuntimeError(f"{failure}; inspect {output / 'inference.log'}")
    return record


def run(*args, **kwargs):
    from .patient_anatomy import reconstruction_lock
    with reconstruction_lock():
        return _run(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--model", choices=MODEL_CONFIG)
    parser.add_argument("--modality")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--experimental-t1ce", action="store_true")
    parser.add_argument("--blend", choices=["constant","gaussian"], default="gaussian")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--output-root", type=Path, help="Explicit app-testing cache directory under testcases/")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
    else:
        if not all([args.image, args.case, args.model, args.modality]):
            parser.error("--image, --case, --model, --modality are required")
        run(args.image, args.case, args.model, args.modality, args.device, args.timeout, experimental_t1ce=args.experimental_t1ce, blend=args.blend, threshold=args.threshold, output_root=args.output_root)


if __name__ == "__main__":
    main()
