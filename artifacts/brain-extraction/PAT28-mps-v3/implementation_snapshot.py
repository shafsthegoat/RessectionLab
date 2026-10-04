"""Pinned local brain extraction with unreviewed-mask provenance and anomaly QC.

Estimated extraction masks are not reviewed brain/cortex labels, clinical
probabilities, or permission to enter tissue. They never grant cortical access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable
import urllib.request

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from .imaging import ImagingError, file_sha256, inspect_nifti, load_fractional_annotation_case


CODE_COMMIT = "cf4bccf24875a47245b4df9fc9372d0f1c3d784f"
UPSTREAM_BASE = f"https://raw.githubusercontent.com/freesurfer/freesurfer/{CODE_COMMIT}"
MODEL_BASE = "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements"
ASSETS = {
    "mri_synthstrip": (f"{UPSTREAM_BASE}/mri_synthstrip/mri_synthstrip", "291c253ab2f7c0cbcb84afa769ae10909549537e727db5b92d268754b7cc7871"),
    "LICENSE_FreeSurfer.txt": (f"{UPSTREAM_BASE}/LICENSE.txt", "632d404ea17b9101d9ac87cf8f09e651d18eb20d2cc0ba8dbee77e85b8516307"),
    "synthstrip.nocsf.1.pt": (f"{MODEL_BASE}/synthstrip.nocsf.1.pt", "62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28"),
    "synthstrip.1.pt": (f"{MODEL_BASE}/synthstrip.1.pt", "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"),
}


class BrainExtractionError(ImagingError):
    """Named inference/provenance failure without silently accepting a mask."""


def mps_runner_source(original: str, maximum_memory_bytes: int) -> str:
    """Mark a minimal device adapter; leave architecture, weights and resampling intact."""
    old = """        if not torch.cuda.is_available():
            sf.system.fatal('-g flag provided but CUDA is not available')
        device = torch.device('cuda')
        device_name = 'GPU'"""
    new = f"""        if not torch.backends.mps.is_available():
            sf.system.fatal('RessectionLab MPS adapter requires available Apple Metal')
        device = torch.device('mps')
        torch.mps.set_per_process_memory_fraction(min(0.7, {maximum_memory_bytes} / torch.mps.recommended_max_memory()))
        device_name = 'Apple Metal (RessectionLab modified device adapter)'"""
    if original.count(old) != 1:
        raise BrainExtractionError("UPSTREAM_ADAPTER_MISMATCH", "Pinned device-selection block was not found exactly once.")
    notice = ("# MODIFIED by RessectionLab: bounded Apple Metal device and allocator instrumentation.\n"
              "# Original architecture, weights, preprocessing and native-grid resampling retained.\n"
              "# Original FreeSurfer license applies; see LICENSE_FreeSurfer.txt.\n")
    adapted = original.replace(old, new)
    hook_anchor = "    # load input volume"
    if adapted.count(hook_anchor) == 1:
        hook = """    # RessectionLab: observational allocator samples; no tensor values are changed.
    mps_memory = {'tensor_bytes_max_sampled': 0, 'driver_bytes_max_sampled': 0}
    def sample_mps_memory(module, inputs, output):
        mps_memory['tensor_bytes_max_sampled'] = max(mps_memory['tensor_bytes_max_sampled'], torch.mps.current_allocated_memory())
        mps_memory['driver_bytes_max_sampled'] = max(mps_memory['driver_bytes_max_sampled'], torch.mps.driver_allocated_memory())
    if device.type == 'mps':
        for layer in model.modules():
            layer.register_forward_hook(sample_mps_memory)

"""
        adapted = adapted.replace(hook_anchor, hook + hook_anchor)
        anchor = "    print(ref)\n\n\n# execute script"
        if adapted.count(anchor) != 1:
            raise BrainExtractionError("UPSTREAM_ADAPTER_MISMATCH", "Pinned end-of-run block changed.")
        adapted = adapted.replace(anchor, "    import json\n    print('RESECTIONLAB_MPS_MEMORY ' + json.dumps(mps_memory))\n\n" + anchor)
    return notice + adapted


def ensure_model_assets(cache: Path, *, model: str, allow_download: bool = False) -> dict:
    if model not in {"nocsf", "main"}:
        raise BrainExtractionError("UNSUPPORTED_EXTRACTION_MODEL", "Choose the pinned main or no-CSF adult model.")
    weights = "synthstrip.nocsf.1.pt" if model == "nocsf" else "synthstrip.1.pt"
    records = {}
    for name in ("mri_synthstrip", "LICENSE_FreeSurfer.txt", weights):
        url, digest = ASSETS[name]
        destination = cache / name
        if not destination.exists():
            if not allow_download:
                raise BrainExtractionError("MODEL_ASSET_MISSING", f"Missing {name}; explicitly enable the public-model downloader.")
            cache.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = response.read(40 * 1024 * 1024)
                if response.read(1):
                    raise BrainExtractionError("MODEL_DOWNLOAD_BUDGET_EXCEEDED", "Single model asset exceeded 40 MiB.")
            from hashlib import sha256
            if sha256(payload).hexdigest() != digest:
                raise BrainExtractionError("MODEL_ASSET_HASH_MISMATCH", f"Downloaded {name} does not match the pinned artifact.")
            destination.write_bytes(payload)
        if file_sha256(destination) != digest:
            raise BrainExtractionError("MODEL_ASSET_HASH_MISMATCH", f"Cached {name} does not match the pinned artifact.")
        records[name] = {"url": url, "sha256": digest, "bytes": destination.stat().st_size}
    return {"name": "SynthStrip", "variant": model, "weights_version": 1,
            "upstream_code_commit": CODE_COMMIT, "files": records,
            "code_license": "FreeSurfer Software License Agreement 1.0, February 2011",
            "weights_license_choice": "MIT, explicitly offered by official SynthStrip page",
            "weights_license_source": "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/"}


def validate_mask(mask_path: Path, image_path: Path) -> tuple[np.ndarray, dict]:
    mask_qc, image_qc = inspect_nifti(mask_path), inspect_nifti(image_path)
    if (mask_qc["shape"] != image_qc["shape"]
            or not np.allclose(mask_qc["affine_ras_mm"], image_qc["affine_ras_mm"], atol=0.01, rtol=1e-5)):
        raise BrainExtractionError("EXTRACTION_FRAME_MISMATCH", "Estimated mask must retain the input native grid.")
    mask = nib.load(mask_path).get_fdata(dtype=np.float32)
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise BrainExtractionError("EXTRACTION_MASK_NONBINARY", "Extraction output must contain only finite zero/one values.")
    if not mask.any():
        raise BrainExtractionError("EMPTY_EXTRACTION_MASK", "Model produced an empty brain mask.")
    return mask.astype(bool), mask_qc


def run_synthstrip(image_path: Path, cache: Path, output: Path, *, model: str = "nocsf",
                   allow_download: bool = False, timeout_seconds: float = 300,
                   maximum_rss_bytes: int = 6 * 1024**3,
                   inference_python: Path | None = None,
                   device: str = "cpu",
                   cancelled: Callable[[], bool] | None = None) -> dict:
    """Execute pinned inference in a bounded child; MPS uses a marked device adapter."""
    if not 1 <= timeout_seconds <= 900 or not 512 * 1024**2 <= maximum_rss_bytes <= 8 * 1024**3:
        raise BrainExtractionError("INVALID_EXTRACTION_BUDGET", "Use 1–900 s and 0.5–8 GiB process memory budget.")
    if device not in {"cpu", "mps"}:
        raise BrainExtractionError("UNSUPPORTED_EXTRACTION_DEVICE", "Choose CPU or the marked MPS device adapter.")
    source_qc = inspect_nifti(image_path)
    if len(source_qc["shape"]) != 3:
        raise BrainExtractionError("EXTRACTION_REQUIRES_3D", "Extract one structural volume at a time.")
    assets = ensure_model_assets(cache, model=model, allow_download=allow_download)
    output.mkdir(parents=True, exist_ok=True)
    mask_path, distance_path = output / f"{model}_mask.nii.gz", output / f"{model}_distance_mm.nii.gz"
    # A failed rerun must not appear to succeed because older mask files exist.
    if mask_path.exists() or distance_path.exists():
        raise BrainExtractionError("EXTRACTION_OUTPUT_EXISTS", "Choose a new output directory to preserve earlier inference artifacts.")
    weights = "synthstrip.nocsf.1.pt" if model == "nocsf" else "synthstrip.1.pt"
    executable = str(inference_python) if inference_python is not None else sys.executable
    runtime_probe = subprocess.run([executable, "-c", "import json,sys,numpy,torch,surfa; print(json.dumps({'python':sys.version.split()[0], 'numpy':numpy.__version__, 'torch':torch.__version__, 'surfa':surfa.__version__}))"],
                                   capture_output=True, text=True, check=True, timeout=30)
    runtime_versions = json.loads(runtime_probe.stdout)
    runner = cache / "mri_synthstrip"
    if device == "mps":
        runner = output / f"{model}_synthstrip_mps.py"
        runner.write_text(mps_runner_source((cache / "mri_synthstrip").read_text(), maximum_rss_bytes))
        (output / "LICENSE_FreeSurfer.txt").write_bytes((cache / "LICENSE_FreeSurfer.txt").read_bytes())
    command = [executable, str(runner), "--image", str(image_path),
               "--mask", str(mask_path), "--sdt", str(distance_path), "--model", str(cache / weights),
               "--threads", "2", "--border", "1"]
    if device == "mps":
        command.append("--gpu")
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"})
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    start = time.monotonic()
    peak_rss = 0
    failure = None
    with (output / f"{model}_inference.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    failure = "EXTRACTION_CANCELLED"
                elif time.monotonic() - start > timeout_seconds:
                    failure = "EXTRACTION_TIME_BUDGET_EXCEEDED"
                memory = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                        capture_output=True, text=True, check=False)
                if memory.stdout.strip():
                    peak_rss = max(peak_rss, int(memory.stdout.strip()) * 1024)
                if peak_rss > maximum_rss_bytes:
                    failure = "EXTRACTION_MEMORY_BUDGET_EXCEEDED"
                if failure:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    record = {"model": assets, "input_sha256": file_sha256(image_path), "input_qc": source_qc,
              "elapsed_seconds": time.monotonic() - start, "sampled_process_peak_rss_bytes": peak_rss,
              "memory_measurement": "child-process RSS sampled with ps approximately every 0.25 s; may miss brief peaks",
              "configuration": {"cpu_threads": 2, "border_mm": 1, "device": device,
                                "timeout_seconds": timeout_seconds, "maximum_rss_bytes": maximum_rss_bytes,
                                "weights_only_load": True}, "runtime_versions": runtime_versions,
              "executed_runner_sha256": file_sha256(runner),
              "upstream_modified": device == "mps",
              "exit_code": process.returncode,
              "failure": failure or ("EXTRACTION_PROCESS_FAILED" if process.returncode else None),
              "provenance": "estimated", "brain_reviewed": False, "cortical_access_permitted": False}
    if device == "mps":
        for line in (output / f"{model}_inference.log").read_text().splitlines():
            if line.startswith("RESECTIONLAB_MPS_MEMORY "):
                record["mps_memory_samples"] = json.loads(line.split(" ", 1)[1])
        record["mps_memory_measurement"] = "allocator samples after module forwards, not exact peak; driver includes cached allocations; process RSS alone excludes GPU accounting"
    if not record["failure"]:
        try:
            validate_mask(mask_path, image_path)
            record["artifact_hashes"] = {p.name: file_sha256(p) for p in (mask_path, distance_path)}
        except (ImagingError, OSError) as error:
            record["failure"] = getattr(error, "code", "EXTRACTION_OUTPUT_INVALID")
            record["validation_error"] = str(error)
    (output / f"{model}_inference_record.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["failure"]:
        raise BrainExtractionError(record["failure"], f"Inference stopped; inspect {model}_inference_record.json and log.")
    return record


def intensity_core_baseline(image: np.ndarray, spacing_mm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Erosive T1 intensity-support comparator; explicitly not a whole-brain mask."""
    if image.ndim != 3 or not np.isfinite(image).all() or np.ptp(image) <= 0:
        raise BrainExtractionError("INVALID_BASELINE_IMAGE", "T1 baseline needs finite, varying 3D intensities.")
    spacing_mm = np.asarray(spacing_mm, dtype=float)
    if spacing_mm.shape != (3,) or not np.isfinite(spacing_mm).all() or (spacing_mm <= 0).any():
        raise BrainExtractionError("INVALID_BASELINE_SPACING", "Baseline requires three positive finite spacings in mm.")
    positive = image[image > 0]
    if len(positive) < 100:
        raise BrainExtractionError("BASELINE_SUPPORT_INSUFFICIENT", "Insufficient positive T1 intensity support.")
    threshold = float(threshold_otsu(positive))
    radii = np.ceil(2.5 / spacing_mm).astype(int)
    grid = np.meshgrid(*[np.arange(-r, r + 1) * s for r, s in zip(radii, spacing_mm)], indexing="ij")
    structure = sum(axis**2 for axis in grid) <= 2.5**2
    opened = ndimage.binary_opening(image > threshold, structure=structure)
    labels, count = ndimage.label(opened)
    if not count:
        raise BrainExtractionError("BASELINE_SUPPORT_EMPTY", "No connected intensity component survived physical opening.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == sizes.argmax()
    filled = ndimage.binary_fill_holes(largest)
    core = ndimage.distance_transform_edt(filled, sampling=spacing_mm) > 1
    return core, {"method": "positive-intensity Otsu, 2.5 mm opening, largest component, hole fill, 1 mm erosion",
                  "otsu_threshold": threshold, "meaning": "erosive T1 intensity core; not anatomical truth or complete brain",
                  "whole_brain_coverage": "unknown", "reviewed": False}


def extraction_qc(mask: np.ndarray, affine: np.ndarray, *, tumor: np.ndarray | None = None,
                   baseline: np.ndarray | None = None) -> dict:
    if mask.ndim != 3 or mask.dtype != bool or not mask.any():
        raise BrainExtractionError("INVALID_QC_MASK", "QC needs a nonempty boolean 3D mask.")
    affine = np.asarray(affine, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-9):
        raise BrainExtractionError("INVALID_QC_AFFINE", "QC requires a finite invertible physical affine.")
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    volume_ml = float(mask.sum() * voxel_mm3 / 1000)
    _, component_count = ndimage.label(mask)
    boundary = np.zeros(mask.shape, dtype=bool)
    for axis in range(3):
        selector = [slice(None)] * 3
        for edge in (0, -1):
            selector[axis] = edge
            boundary[tuple(selector)] = True
    face_voxels = int(np.count_nonzero(mask & boundary))
    flags = []
    if not 400 <= volume_ml <= 2200:
        flags.append("EXTRACTION_VOLUME_OUTSIDE_BROAD_ADULT_SANITY_RANGE")
    if component_count != 1:
        flags.append("MULTIPLE_EXTRACTION_COMPONENTS")
    if face_voxels:
        flags.append("EXTRACTION_TOUCHES_INPUT_FOV_BOUNDARY")
    result = {"mask_voxels": int(mask.sum()), "mask_volume_ml": volume_ml,
              "connected_components": component_count, "boundary_face_voxels": face_voxels,
              "sanity_volume_range_ml": [400, 2200], "sanity_range_meaning": "broad engineering alert, not clinical acceptance",
              "flags": flags, "brain_reviewed": False, "cortex_localized": False,
              "cortical_access_permitted": False, "clinical_deficit_probability": None}
    if tumor is not None:
        if tumor.shape != mask.shape or tumor.dtype != bool or not tumor.any():
            raise BrainExtractionError("TUMOR_QC_GRID_INVALID", "QC annotation must be a nonempty boolean mask on the same grid.")
        fraction = float(np.count_nonzero(tumor & mask) / tumor.sum())
        result["source_annotation_inclusion_fraction"] = fraction
        result["source_annotation_voxels"] = int(tumor.sum())
        result["annotation_fraction_meaning"] = "overlap of threshold-derived source annotation; not extraction accuracy or removal"
        if fraction < 0.98:
            flags.append("SOURCE_ANNOTATION_EXTENDS_OUTSIDE_EXTRACTION")
    if baseline is not None:
        if baseline.shape != mask.shape or baseline.dtype != bool:
            raise BrainExtractionError("BASELINE_QC_GRID_INVALID", "Baseline must be boolean on the native image grid.")
        denominator = int(mask.sum() + baseline.sum())
        result["baseline_dice_agreement"] = float(2 * np.count_nonzero(mask & baseline) / denominator)
        result["baseline_agreement_meaning"] = "agreement with an erosive intensity core, not measured segmentation accuracy"
        result["baseline_volume_ml"] = float(baseline.sum() * voxel_mm3 / 1000)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--tumor-fractional", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--compare-main-model", action="store_true")
    parser.add_argument("--inference-python", type=Path, help="Optional isolated compatible inference interpreter.")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise BrainExtractionError("EXTRACTION_OUTPUT_EXISTS", "Use a new experiment directory; earlier artifacts are preserved.")
    qc = inspect_nifti(args.t1)
    affine = np.asarray(qc["affine_ras_mm"])
    image = nib.load(args.t1).get_fdata(dtype=np.float32)
    tumor, annotation = None, None
    if args.tumor_fractional:
        case = load_fractional_annotation_case(args.t1, args.tumor_fractional, threshold=0.5,
                    annotation_interpretation="source fractional annotation; threshold research scenario, not probability")
        tumor = next(iter(case.compartments.values()))
        annotation = {"sha256": file_sha256(args.tumor_fractional),
                      "geometry_and_threshold": dict(case.metadata["fractional_annotation"])}
    baseline, baseline_method = intensity_core_baseline(image, np.linalg.norm(affine[:3, :3], axis=0))
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_image = nib.Nifti1Image(baseline.astype(np.uint8), affine)
    baseline_image.header.set_xyzt_units("mm")
    nib.save(baseline_image, args.output / "intensity_core_baseline.nii.gz")
    variants = {}
    for variant in (["nocsf", "main"] if args.compare_main_model else ["nocsf"]):
        record = run_synthstrip(args.t1, args.cache, args.output, model=variant,
                                allow_download=args.allow_model_download, inference_python=args.inference_python,
                                device=args.device)
        mask, _ = validate_mask(args.output / f"{variant}_mask.nii.gz", args.t1)
        variants[variant] = {"inference": record, "qc": extraction_qc(mask, affine, tumor=tumor, baseline=baseline)}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "implementation_sha256": file_sha256(__file__), "source_t1_sha256": file_sha256(args.t1),
              "source_annotation": annotation, "baseline": baseline_method, "variants": variants,
              "clinical_use_status": "research_only", "brain_reviewed": False,
              "cortical_access_permitted": False, "training_overlap_audit": "see docs/brain_extraction.md; patient-level disjointness not certified"}
    (args.output / "brain_extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
