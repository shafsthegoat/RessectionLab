#!/usr/bin/env python3
"""Audit an optional local FSL CPU backend without silently accepting its license.

The default command downloads only public package metadata. Installation needs
an existing micromamba executable and a separate documented use-scope record.
This script never runs patient correction or changes source images/headers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

FSL_CHANNEL = "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public"
LICENSE_URL = "https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html"
BIDS_URL = "https://bids-specification.readthedocs.io/en/v1.11.0/modality-specific-files/magnetic-resonance-imaging-data.html"
PINNED_PACKAGES = {
    "fsl-eddy": "fsl-eddy-2602.0-hed4507f_0.conda",
    "fsl-topup": "fsl-topup-2203.6-ha7be7b2_0.conda",
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/dwi"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquisition_row(metadata: dict) -> list[float]:
    import math

    direction = metadata.get("PhaseEncodingDirection")
    if direction not in ("i", "i-", "j", "j-", "k", "k-"):
        raise ValueError("missing or invalid BIDS PhaseEncodingDirection")
    readout = metadata.get("TotalReadoutTime")
    if isinstance(readout, bool) or not isinstance(readout, (int, float)) or not math.isfinite(readout) or readout <= 0:
        raise ValueError("TotalReadoutTime must be a finite positive number in seconds")
    vector = [0.0, 0.0, 0.0]
    vector["ijk".index(direction[0])] = -1.0 if direction.endswith("-") else 1.0
    return [*vector, float(readout)]


def inspect_patient(data: Path) -> dict:
    import nibabel as nib
    import numpy as np

    series = {}
    images = {}
    failures = []
    for label in ("AP", "PA"):
        prefix = data / f"sub-PAT28_ses-preop_acq-{label}_dwi"
        paths = {kind: Path(str(prefix) + suffix) for kind, suffix in
                 (("image", ".nii.gz"), ("metadata", ".json"), ("bvals", ".bval"), ("bvecs", ".bvec"))}
        metadata = json.loads(paths["metadata"].read_text())
        image = nib.load(paths["image"])
        images[label] = image
        bvals = np.loadtxt(paths["bvals"]).reshape(-1)
        bvecs = np.loadtxt(paths["bvecs"])
        if image.ndim != 4 or bvals.size != image.shape[3] or bvecs.shape != (3, image.shape[3]):
            raise ValueError(f"{label}: image/gradient counts do not agree")
        if not np.all(np.isfinite(bvals)) or not np.all(np.isfinite(bvecs)) or np.any(bvals < 0):
            raise ValueError(f"{label}: gradients contain invalid numbers")
        b0 = bvals <= 50
        norms = np.linalg.norm(bvecs, axis=0)
        if not b0.any() or np.any(norms[b0] > 0.01) or np.any(np.abs(norms[~b0] - 1) > 0.01):
            raise ValueError(f"{label}: missing b0 or invalid gradient norms")
        row = acquisition_row(metadata)
        axis = "ijk".index(metadata["PhaseEncodingDirection"][0])
        echo_spacing = metadata.get("EffectiveEchoSpacing")
        implied_readout = None if echo_spacing is None else float(echo_spacing) * (image.shape[axis] - 1)
        readout_matches = implied_readout is None or bool(np.isclose(row[3], implied_readout, rtol=0.01, atol=1e-7))
        if not readout_matches:
            failures.append(f"{label}_READOUT_METADATA_CONFLICT")
        if image.header.get_xyzt_units()[0] != "mm":
            failures.append(f"{label}_SPATIAL_UNITS_NOT_MM")
        series[label] = {
            "shape": list(image.shape), "affine": image.affine.tolist(),
            "orientation": list(nib.aff2axcodes(image.affine)),
            "b0_indices": np.flatnonzero(b0).tolist(), "bvec_shape": list(bvecs.shape),
            "source_acquisition_row": row, "effective_echo_spacing_seconds": echo_spacing,
            "readout_from_echo_spacing_and_image_matrix_seconds": implied_readout,
            "readout_metadata_consistent": readout_matches,
            "converter": metadata.get("ConversionSoftware"),
            "converter_version": metadata.get("ConversionSoftwareVersion"),
            "files": {kind: {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                              "bytes": path.stat().st_size, "sha256": sha256(path)} for kind, path in paths.items()},
        }
    ap, pa = images["AP"], images["PA"]
    if ap.shape[:3] != pa.shape[:3]:
        failures.append("AP_PA_SPATIAL_SHAPE_MISMATCH")
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in ap.shape[:3]], indexing="ij")).reshape(3, -1).T
    corner_distance = float(np.max(np.linalg.norm(nib.affines.apply_affine(ap.affine, corners) - nib.affines.apply_affine(pa.affine, corners), axis=1)))
    if corner_distance > 0.01:
        failures.append("AP_PA_GRID_RECONCILIATION_REQUIRED")
    if not np.allclose(series["AP"]["source_acquisition_row"][:3], -np.array(series["PA"]["source_acquisition_row"][:3])):
        failures.append("PHASE_ENCODINGS_NOT_OPPOSED")
    return {
        "case": "BTC_preop_sub-PAT28", "status": "blocked" if failures else "metadata_checks_passed",
        "series": series, "ap_pa_max_corner_distance_mm": corner_distance,
        "pa_voxel_to_ap_voxel": (np.linalg.inv(ap.affine) @ pa.affine).tolist(),
        "failure_reasons": failures,
        "correction_executed": False, "raw_files_modified": False,
        "gradient_status": "source_dcm2niix_gradients_unchanged_not_motion_rotated",
        "note": "Source readout values are retained. No header replacement, resampling, readout substitution, or phase-direction override was performed.",
    }


def package_audit() -> dict:
    url = FSL_CHANNEL + "/osx-arm64/repodata.json"
    request = urllib.request.Request(url, headers={"User-Agent": "RessectionLab-diffusion-preflight/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read(4 * 1024 * 1024 + 1)
    if len(content) > 4 * 1024 * 1024:
        raise ValueError("package metadata exceeds bounded 4 MiB limit")
    metadata = json.loads(content)
    available = {**metadata.get("packages", {}), **metadata.get("packages.conda", {})}
    selected = {}
    for name, filename in PINNED_PACKAGES.items():
        record = available[filename]
        selected[name] = {key: record.get(key) for key in ("version", "build", "size", "sha256", "md5", "depends", "license")}
        selected[name]["url"] = FSL_CHANNEL + "/osx-arm64/" + filename
        selected[name]["effective_license_source"] = LICENSE_URL
    return {
        "metadata_url": url, "metadata_sha256": hashlib.sha256(content).hexdigest(),
        "metadata_download_bytes": len(content), "target_platform": "osx-arm64",
        "packages": selected, "direct_package_bytes_excluding_dependencies": sum(item["size"] for item in selected.values()),
        "dependency_download_bytes": None, "installed": False,
        "license_status": "use_scope_clarification_required",
        "license_url": LICENSE_URL,
        "note": "Null conda license metadata is not a permissive license. Full FSL terms still apply.",
    }


def validate_use_scope(record: dict) -> None:
    if record.get("permission_status") != "confirmed":
        raise ValueError("FSL use scope is unconfirmed; installation remains disabled")
    if record.get("scope") not in ("noncommercial_research", "commercial_permission"):
        raise ValueError("a noncommercial research scope or commercial permission must be documented")
    for key in ("confirmed_by", "basis", "reference"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"use-scope record needs {key}")


def optional_install(args: argparse.Namespace, report: dict) -> None:
    """Install only a resolved, size-bounded transaction into an ignored folder."""
    if args.license_record is None:
        raise ValueError("--install requires a documented --license-record; no license acceptance is inferred")
    use_scope = json.loads(args.license_record.read_text())
    validate_use_scope(use_scope)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("the pinned backend is for macOS arm64 only")
    if args.micromamba is None or not args.micromamba.is_file():
        raise ValueError("supply an existing --micromamba; no global package manager is installed")
    target = ROOT / ".tools/fsl"
    ignored = subprocess.run(["git", "check-ignore", "--quiet", ".tools/fsl/backend"], cwd=ROOT)
    if ignored.returncode != 0:
        raise ValueError(".tools/ must be ignored before installing the optional backend")
    if target.exists():
        raise ValueError("existing FSL prefix preserved; use --probe or explicitly manage it outside this script")
    base = [str(args.micromamba.resolve()), "--no-rc", "--root-prefix", str(ROOT / ".tools/mamba")]
    specs = [f"{name}={item['version']}={item['build']}" for name, item in report["packages"]["packages"].items()]
    command = [*base, "create", "--prefix", str(target), "--override-channels", "--strict-channel-priority",
               "-c", FSL_CHANNEL, "-c", "conda-forge", *specs, "--dry-run", "--json"]
    transaction = subprocess.run(command, text=True, capture_output=True, timeout=180)
    (args.output / "solver.log").write_text(transaction.stdout + "\n" + transaction.stderr)
    if transaction.returncode:
        raise RuntimeError("FSL dependency resolution failed; inspect solver.log")
    resolved = json.loads(transaction.stdout)
    write_json(args.output / "transaction.json", resolved)
    if not resolved.get("success"):
        raise RuntimeError("FSL solver did not report a successful transaction")
    packages = resolved.get("actions", {}).get("LINK", [])
    if not packages:
        raise ValueError("solver returned no installable packages")
    total = sum(item.get("size", 0) for item in packages)
    if any(not item.get("size") for item in packages) or total > args.max_download_mb * 1024 * 1024:
        raise ValueError("resolved package size is unknown or exceeds --max-download-mb")
    explicit = args.output / "fsl-osx-arm64-explicit.txt"
    lines = []
    for item in packages:
        if not item.get("url") or not item.get("md5"):
            raise ValueError("resolved package lacks URL/checksum for a reproducible explicit lock")
        lines.append(item["url"] + "#" + item["md5"])
    explicit.write_text("@EXPLICIT\n" + "\n".join(lines) + "\n")
    with (args.output / "install.log").open("w") as log:
        result = subprocess.run([*base, "create", "--prefix", str(target), "--file", str(explicit.resolve()), "--yes"], stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise RuntimeError("FSL installation failed; partial local prefix retained for inspection")
    report["installation"] = {"prefix": str(target.relative_to(ROOT)), "resolved_package_bytes": total,
                               "license_record_sha256": sha256(args.license_record), "installed": True}


def probe(output: Path) -> dict:
    prefix = ROOT / ".tools/fsl"
    result = {}
    for label, choices in {"topup": ("topup",), "eddy_cpu": ("eddy_cpu", "eddy_openmp", "eddy")}.items():
        binary = next((prefix / "bin" / choice for choice in choices if (prefix / "bin" / choice).is_file()), None)
        if binary is None:
            result[label] = {"status": "not_installed"}
            continue
        import os
        environment = dict(os.environ, FSLDIR=str(prefix), FSLOUTPUTTYPE="NIFTI_GZ", OMP_NUM_THREADS="2")
        completed = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=30, env=environment)
        log_path = output / f"{label}-help.txt"
        log_path.write_text(completed.stdout + completed.stderr)
        text = completed.stdout + completed.stderr
        result[label] = {"status": "help_verified" if "--imain" in text else "failed",
                         "returncode": completed.returncode, "binary_sha256": sha256(binary), "log": log_path.name}
    return result


def self_test() -> dict:
    checks = 0
    for direction, vector in (("i", [1, 0, 0]), ("j-", [0, -1, 0]), ("k", [0, 0, 1])):
        assert acquisition_row({"PhaseEncodingDirection": direction, "TotalReadoutTime": 0.03}) == [*vector, 0.03]
        checks += 1
    for bad in ({}, {"PhaseEncodingDirection": "AP", "TotalReadoutTime": 0.03},
                {"PhaseEncodingDirection": "j", "TotalReadoutTime": float("nan")},
                {"PhaseEncodingDirection": "j", "TotalReadoutTime": -1},
                {"PhaseEncodingDirection": "j", "TotalReadoutTime": True}):
        try:
            acquisition_row(bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("invalid acquisition metadata was accepted")
    for record in ({}, {"permission_status": "confirmed", "scope": "commercial"}):
        try:
            validate_use_scope(record)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("unconfirmed license scope was accepted")
    return {"passed": checks, "status": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preprocessing/PAT28-fsl-preflight-v1")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--offline", action="store_true", help="skip official package metadata retrieval")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--license-record", type=Path)
    parser.add_argument("--micromamba", type=Path)
    parser.add_argument("--max-download-mb", type=int, default=400)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "host": {"system": platform.system(), "machine": platform.machine()},
              "script_sha256": sha256(Path(__file__)), "clinical_use_status": "research_only",
              "sources": {"fsl_license": LICENSE_URL, "bids_metadata": BIDS_URL}}
    try:
        report["patient"] = inspect_patient(args.data)
        if not args.offline:
            report["packages"] = package_audit()
        if args.install:
            if args.offline:
                raise ValueError("install needs official package metadata")
            optional_install(args, report)
        if args.probe:
            report["probe"] = probe(args.output)
        report["status"] = "preflight_recorded_correction_not_executed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.output / "report.json", report)
        print(json.dumps({"status": "failed", "reason": str(exc), "report": str(args.output / "report.json")}))
        return 1
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "patient_gates": report["patient"]["failure_reasons"], "report": str(args.output / "report.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
