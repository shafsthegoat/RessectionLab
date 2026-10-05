#!/usr/bin/env python3
"""Prepare, then explicitly run a bounded real-scan CPU spatial-policy profile."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
from itertools import product
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PLAN = ((32, 16), (32, 128), (64, 16), (64, 128))
TOTAL_SECONDS = 55.0
RSS_LIMIT_BYTES = 2 * 1024 ** 3
RSS_STOP_BYTES = 1792 * 1024 ** 2  # Monitoring headroom; not an OS allocation cap.
SOURCE_ROOT = ROOT / "data/diffusion_source/ds001226-v5.0.1"
SUBJECT = "sub-PAT22"
COHORT_PATH = ROOT / "manifests/experiments/btc-spatial-development-cohort-v1.json"
MANIFEST_PATH = SOURCE_ROOT / "btc_sub-PAT22_verified_manifest.json"
IMAGE_RELATIVE = "sub-PAT22/ses-preop/anat/sub-PAT22_ses-preop_T1w.nii.gz"
def source_paths():
    return ("scripts/profile_spatial_policy_scale.py", *sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "src/resectionlab").rglob("*.py")))


def enforce_import_root():
    """An archived script must never import the editable working package."""
    expected = (ROOT / "src").resolve()
    if not (expected / "resectionlab/__init__.py").is_file():
        raise ValueError("Profile execution requires a complete local source archive")
    sys.path[:] = [str(expected), *[entry for entry in sys.path if entry != str(expected)]]
    return module_provenance()


def module_provenance():
    expected = (ROOT / "src/resectionlab").resolve()
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == "resectionlab" or name.startswith("resectionlab."):
            filename = getattr(module, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(expected):
                raise ValueError(f"Imported package escaped the profile source archive: {name}")
            path = Path(filename).resolve()
            result[name] = {"path": str(path), "sha256": digest(path)}
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes():
    return {name: digest(ROOT / name) for name in source_paths()}


def peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def verified_source(source_root=SOURCE_ROOT, manifest_path=MANIFEST_PATH, cohort_path=COHORT_PATH):
    """Read only the assigned TRAIN T1; annotation files are never opened."""
    import nibabel as nib
    import numpy as np
    manifest_path, cohort_path, source_root = map(Path, (manifest_path, cohort_path, source_root))
    manifest, cohort = json.loads(manifest_path.read_text()), json.loads(cohort_path.read_text())
    if (manifest.get("subject") != SUBJECT or manifest.get("development_role") != "population_training"
            or manifest.get("selection_cohort_sha256") != digest(cohort_path)):
        raise ValueError("Only the source-bound PAT22 training record is permitted")
    candidates = [r for r in cohort["candidates"] if r["subject"] == SUBJECT]
    if (len(candidates) != 1 or candidates[0]["development_role"] != "population_training"
            or candidates[0].get("role_locked_before_image_access") is not True):
        raise ValueError("PAT22 must have a prospective locked training role")
    records = [r for r in manifest["files"] if r["path"] == IMAGE_RELATIVE]
    if len(records) != 1:
        raise ValueError("Exactly one assigned structural T1 source is required")
    path, record = source_root / IMAGE_RELATIVE, records[0]
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("Structural source checksum or byte count changed")
    image = nib.load(path)
    if len(image.shape) != 3 or any(n < 64 for n in image.shape) or np.prod(image.shape) > 16_000_000:
        raise ValueError("Source header is outside the bounded native structural grid")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Source spatial units must explicitly be millimetres")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not (qcode or scode):
        raise ValueError("Source has no declared physical transform")
    corners = np.asarray(list(product(*[(0, n - 1) for n in image.shape])))
    if qcode and scode:
        difference = corners @ (qform[:3, :3] - sform[:3, :3]).T + qform[:3, 3] - sform[:3, 3]
        if np.linalg.norm(difference, axis=1).max() > .01:
            raise ValueError("Source qform and sform disagree physically")
    affine = image.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (not np.isfinite(affine).all() or not np.array_equal(affine[3], (0, 0, 0, 1))
            or np.any(spacing <= 0) or not np.allclose((affine[:3, :3] / spacing).T @
            (affine[:3, :3] / spacing), np.eye(3), atol=1e-8, rtol=0)):
        raise ValueError("Source physical grid must be finite and orthogonal")
    return {"subject": SUBJECT, "development_role": "population_training", "image_path": str(path.resolve()),
        "image_sha256": record["sha256"], "image_bytes": record["bytes"], "shape": list(image.shape),
        "affine_ras_mm": affine.tolist(), "spacing_mm": spacing.tolist(),
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": digest(manifest_path),
        "cohort_path": str(cohort_path.resolve()), "cohort_sha256": digest(cohort_path),
        "source_root": str(source_root.resolve()), "qform_code": int(qcode), "sform_code": int(scode),
        "annotation_opened": False}


def recheck_source(record):
    current = verified_source(record["source_root"], record["manifest_path"], record["cohort_path"])
    if current != record:
        raise ValueError("Declared image, frame, or source/role manifest changed")


def prepare(output, source_declaration=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if source_declaration is None:
        source = verified_source()
    else:
        source = json.loads(Path(source_declaration).read_text())["scan_source"]
        recheck_source(source)
    output.mkdir(parents=True, exist_ok=False)
    declaration = {"schema_version": 2, "role": "real_training_scan_untrained_policy_engineering_profile",
        "plan": [{"side": side, "non_stop_rays": rays} for side, rays in PLAN],
        "cpu_threads": 1, "seed": 139, "inference_calls_per_cell": 1,
        "training_forward_calls_per_cell": 0, "optimizer_steps_per_cell": 0,
        "total_deadline_seconds": TOTAL_SECONDS, "rss_limit_bytes": RSS_LIMIT_BYTES,
        "sampled_rss_stop_bytes": RSS_STOP_BYTES, "sample_period_seconds": .025,
        "source_sha256": source_hashes(), "source_root": str(ROOT.resolve()),
        "import_root": str((ROOT / "src").resolve()), "python_executable": sys.executable,
        "python_executable_sha256": digest(Path(sys.executable).resolve()), "scan_source": source,
        "crop_rule": "Native centered index crop: start=floor((shape-side)/2); no resampling, labels or brain mask.",
        "normalization": "All finite voxels of this one source image, including zero background: float64 population mean/std; cropped z-score float32; no clipping or cohort statistics.",
        "action_rule": "Fixed geometry-only rays through the crop along its third voxel axis; arbitrary engineering queries, no anatomical access/route claim.",
        "public_data": True, "synthetic_scan": False, "clinical_or_native_geometry_validation": False,
        "limits": ["One un-warmed forward per cell, not a latency distribution; no training or saved checkpoint.",
                   "Sampled RSS termination is not an OS allocation guarantee; an over-cap high-water mark fails the cell.",
                   "Centered native crops omit most MRI anatomy and may omit the lesion; they do not establish whole-volume fit.",
                   "Later resampling can erase vessels and narrow structures; no clinical safety or model accuracy claim."]}
    write_json(output / "declaration.json", declaration)
    snapshot = output / "source_snapshot"
    for relative in source_paths():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    for name in ("manifest", "cohort"):
        (snapshot / f"{name}.json").write_bytes(Path(source[f"{name}_path"]).read_bytes())
    return declaration


class PhaseTimers:
    def __init__(self):
        self.seconds, self.calls = {}, {}

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - started
            self.calls[name] = self.calls.get(name, 0) + 1

    def wrap(self, name, function):
        def call(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return call


def centered_case_crop(scan, affine, side):
    """Pure array/frame operation, independently testable without a model."""
    import numpy as np
    scan, affine = np.asarray(scan), np.asarray(affine, dtype=np.float64)
    if (scan.ndim != 3 or not np.isfinite(scan).all() or type(side) is not int
            or not 4 <= side <= 64 or any(n < side for n in scan.shape)):
        raise ValueError("Finite native image and a bounded fitting crop are required")
    mean, std = float(np.mean(scan, dtype=np.float64)), float(np.std(scan, dtype=np.float64))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Constant/nonfinite scan cannot be normalized")
    start = (np.asarray(scan.shape) - side) // 2
    selection = tuple(slice(int(a), int(a + side)) for a in start)
    crop = ((scan[selection].astype(np.float64) - mean) / std).astype(np.float32)
    cropped_affine = affine.copy()
    cropped_affine[:3, 3] = affine[:3, :3] @ start + affine[:3, 3]
    return crop, cropped_affine, {"start_voxel": start.tolist(), "side": side,
        "case_intensity_mean": mean, "case_intensity_std": std, "normalization_voxels": int(scan.size),
        "crop_affine_ras_mm": cropped_affine.tolist(), "resampled": False, "annotation_used": False}


def real_observation(side, rays, source):
    enforce_import_root()
    import nibabel as nib
    import numpy as np
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_resection import NATIVE_GENERIC_TOOLS
    from resectionlab.spatial_observations import (ObservedChannel, ObservedProcedureState,
        SpatialAction, SpatialInputs, build_spatial_observation)
    if type(rays) is not int or not 1 <= rays <= 128:
        raise ValueError("Only bounded geometric ray inventories are supported")
    recheck_source(source)
    image = nib.load(source["image_path"])
    scan, affine, receipt = centered_case_crop(image.get_fdata(dtype=np.float32), image.affine, side)
    inputs = SpatialInputs({
        "structural_intensity": ObservedChannel(scan, source_kind="observed_scan",
            derivation="Centered native T1 crop; entire same-case intensity mean/std; no labels"),
        "observed_cavity": ObservedChannel(np.zeros(scan.shape, bool), source_kind="observed_procedure_state",
            derivation="Initial preoperative state: no observed removal")},
        affine, "inference_only", f"{SUBJECT}:{source['image_sha256']}:center-{side}:rays-{rays}")
    world = lambda point: tuple(affine[:3, :3] @ point + affine[:3, 3])
    actions = [SpatialAction("STOP")]
    for index in range(rays):
        x, y = 1 + index % (side - 2), 1 + (index // (side - 2)) % (side - 2)
        actions.append(SpatialAction(f"ray-{index}", world((x, y, .5)), world((x, y, side - 1.5)),
                                     NATIVE_GENERIC_TOOLS[0]))
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    access = AccessWindow(world(((side - 1) / 2, (side - 1) / 2, .5)), tuple(normal),
                          float(side * np.linalg.norm(affine[:3, :3], axis=0).max()))
    return build_spatial_observation(inputs, actions, ObservedProcedureState(access, 0, 5)), receipt


def profile_cell(side, rays, source):
    """One untrained forward on an observed TRAIN image; no losses or updates."""
    started = time.perf_counter()
    import numpy as np
    import torch
    enforce_import_root()
    from resectionlab import spatial_policy as module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(139)
    timings = PhaseTimers()
    with timings.measure("source_checks_load_normalization_crop_observation_hash"):
        observation, crop = real_observation(side, rays, source)
    model = module.SpatialPolicy().cpu()
    before = module.parameter_hash(model)
    original_integrity = type(observation).assert_intact
    with patch.object(model, "_inputs", timings.wrap("input_preparation_inclusive", model._inputs)), \
         patch.object(module, "sample_ray_features", timings.wrap("ray_sampling", module.sample_ray_features)), \
         patch.object(type(observation), "assert_intact", timings.wrap("immutable_layout_validation", original_integrity)), \
         patch.object(model.encoder, "forward", timings.wrap("encoder", model.encoder.forward)), \
         patch.object(model.actor, "forward", timings.wrap("actor_head", model.actor.forward)), \
         patch.object(model.stop, "forward", timings.wrap("stop_head", model.stop.forward)), \
         patch.object(model.critic, "forward", timings.wrap("value_head", model.critic.forward)):
        model.eval()
        with torch.no_grad(), timings.measure("inference_full_policy"):
            logits, value = model(observation)
    if logits.shape != (rays + 1,) or not torch.isfinite(logits).all() or not torch.isfinite(value):
        raise RuntimeError("Invalid full-policy inference output")
    after = module.parameter_hash(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Inference unexpectedly changed parameters or accumulated gradients")
    return {"status": "complete", "side": side, "non_stop_rays": rays, "total_actions": rays + 1,
        "inference": {"seconds": timings.seconds, "calls": timings.calls},
        "measurement_scope": "Nested phase timers wrap the actual forward; inclusive/child durations are not additive.",
        "wall_seconds_including_import_and_setup": time.perf_counter() - started,
        "process_high_water_rss_bytes": peak_rss_bytes(), "crop": crop,
        "parameters": sum(p.numel() for p in model.parameters()), "architecture": model.architecture_record(),
        "architecture_hash": model.architecture_hash, "observation_hash": observation.fingerprint,
        "source_id": observation.source_id, "source_image_sha256": source["image_sha256"],
        "channel_available": observation.channel_available.tolist(), "input_value_bytes": observation.image_channels.nbytes,
        "convolution_input_bytes": 18 * side ** 3 * 4, "optimizer_steps": 0, "gradients_accumulated": False,
        "initial_parameter_hash": before, "final_parameter_hash": after,
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "platform": platform.platform(), "device": "cpu",
        "import_root": str((ROOT / "src").resolve()), "imported_module_provenance": module_provenance()}


def current_rss(pid):
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=.5)
    return int(result.stdout.strip()) * 1024 if result.returncode == 0 and result.stdout.strip() else 0


def supervise(command, log_path, deadline, *, rss_reader=current_rss, rss_stop_bytes=RSS_STOP_BYTES):
    started, maximum, refusal = time.monotonic(), 0, None
    environment = {**os.environ, **{key: "1" for key in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    with Path(log_path).open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    refusal = "total_deadline_reached"
                    break
                maximum = max(maximum, rss_reader(child.pid))
                if maximum >= rss_stop_bytes:
                    refusal = "rss_headroom_stop_reached"
                    break
                time.sleep(.025)
        except Exception as error:
            refusal = f"resource_monitor_failed:{type(error).__name__}:{error}"
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
    return {"exit_code": child.returncode, "refusal": refusal, "sampled_peak_rss_bytes": maximum,
            "supervised_wall_seconds": time.monotonic() - started}


def run_prepared(output):
    output = Path(output)
    declaration_path = output / "declaration.json"
    declaration = json.loads(declaration_path.read_text())
    if declaration.get("source_root") != str(ROOT.resolve()) or declaration.get("import_root") != str((ROOT / "src").resolve()):
        raise ValueError("Profile must execute in its declared complete source archive")
    enforce_import_root()
    if declaration["source_sha256"] != source_hashes():
        raise ValueError("Source changed after declaration; retain it and prepare a new profile")
    if declaration["python_executable_sha256"] != digest(Path(sys.executable).resolve()):
        raise ValueError("Declared Python executable changed")
    expected = {"plan": [{"side": s, "non_stop_rays": n} for s, n in PLAN],
        "total_deadline_seconds": TOTAL_SECONDS, "rss_limit_bytes": RSS_LIMIT_BYTES,
        "sampled_rss_stop_bytes": RSS_STOP_BYTES, "cpu_threads": 1, "seed": 139,
        "inference_calls_per_cell": 1, "training_forward_calls_per_cell": 0,
        "optimizer_steps_per_cell": 0, "sample_period_seconds": .025}
    if any(declaration.get(key) != value for key, value in expected.items()):
        raise ValueError("Profile declaration does not match the bounded plan")
    recheck_source(declaration["scan_source"])
    # Exclusive attempt marker prevents retries from concealing an earlier run.
    write_json(output / "attempt.json", {"declaration_sha256": digest(declaration_path),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    started = time.monotonic()
    rows, status = [], "complete"
    for side, rays in PLAN:
        path = output / f"{side}-cubed-{rays}-rays.json"
        child = supervise([sys.executable, str(Path(__file__).resolve()), "--child", str(side), str(rays),
                           "--declaration", str(declaration_path), "--output", str(path)], output / f"{side}-cubed-{rays}-rays.log", started + TOTAL_SECONDS)
        row = {"side": side, "non_stop_rays": rays, "supervision": child}
        if child["exit_code"] == 0 and child["refusal"] is None and path.is_file():
            try:
                result = json.loads(path.read_text())
                row.update(result=result, result_sha256=digest(path))
                if (result["status"] != "complete" or result["side"] != side
                        or result["non_stop_rays"] != rays or result["optimizer_steps"] != 0
                        or result["source_image_sha256"] != declaration["scan_source"]["image_sha256"]
                        or result["initial_parameter_hash"] != result["final_parameter_hash"]):
                    row["failure"] = "child_report_contract_mismatch"
                if result["process_high_water_rss_bytes"] > RSS_LIMIT_BYTES:
                    row["failure"] = "high_water_rss_exceeded_limit"
            except (KeyError, TypeError, ValueError):
                row["failure"] = "invalid_child_report"
        else:
            row["failure"] = child["refusal"] or "child_failed"
        rows.append(row)
        if row.get("failure"):
            status = "failed_stopped_without_retry"
            break
    if source_hashes() != declaration["source_sha256"]:
        status = "source_changed_during_profile"
    source_failure = None
    try:
        recheck_source(declaration["scan_source"])
    except (ValueError, OSError) as error:
        source_failure = str(error)
        status = "source_changed_during_profile"
    report = {"schema_version": 2, "status": status, "declaration_sha256": digest(declaration_path),
        "total_wall_seconds": time.monotonic() - started, "cells": rows, "source_failure": source_failure,
        "unexecuted_cells": len(PLAN) - len(rows), "retries": 0, "public_data": True, "synthetic_scan": False, "optimizer_steps": 0}
    write_json(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--child", nargs=2, type=int, metavar=("SIDE", "RAYS"), help=argparse.SUPPRESS)
    parser.add_argument("--source-declaration", type=Path, help="Reuse verified absolute scan references when preparing an isolated source archive")
    parser.add_argument("--declaration", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.source_declaration)
    elif args.run:
        result = run_prepared(args.output)
        print(json.dumps({"status": result["status"], "total_wall_seconds": result["total_wall_seconds"]}))
        return 0 if result["status"] == "complete" else 1
    else:
        side, rays = args.child
        if (side, rays) not in PLAN:
            raise ValueError("Child cell is outside the declared scale plan")
        if args.output.exists():
            raise FileExistsError("Profile outputs cannot overwrite previous observations")
        if args.declaration is None:
            raise ValueError("Child requires its exact prepared declaration")
        declaration = json.loads(args.declaration.read_text())
        if declaration["source_sha256"] != source_hashes():
            raise ValueError("Declared source changed before child execution")
        write_json(args.output, profile_cell(side, rays, declaration["scan_source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
