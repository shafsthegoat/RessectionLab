"""Bounded header-only audit of the three root-authorized RESECT baselines."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ACQUISITION = ROOT / "artifacts/mechanics/resect-case4-acquisition-v1/acquisition-receipt.json"
ACQUISITION_SHA = "54c39954cefd2200935d2225bad457ae4aee191690a65d27f8c92cfdd1cbc38b"
ALLOWED = (
    "RESECT/NIFTI/Case4/MRI/Case4-T1.nii.gz",
    "RESECT/NIFTI/Case4/MRI/Case4-FLAIR.nii.gz",
    "RESECT/NIFTI/Case4/US/Case4-US-before.nii.gz",
)
WALL_SECONDS = 55.0
RSS_BYTES = 1024**3


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit():
    started = time.monotonic()
    import nibabel as nib
    import numpy as np

    def scalar(value):
        value = float(value)
        return value if np.isfinite(value) else str(value)

    def forbidden_array(*args, **kwargs):
        raise RuntimeError("IMAGE_ARRAY_ACCESS_PROHIBITED")

    nib.load = forbidden_array
    nib.arrayproxy.ArrayProxy.__array__ = forbidden_array
    if sha(ACQUISITION) != ACQUISITION_SHA:
        raise RuntimeError("ACQUISITION_RECEIPT_CHANGED")
    acquired = json.loads(ACQUISITION.read_text())
    lookup = {item["source_path"]: item for item in acquired["files"]}
    records = []
    # Verify all authorized original compressed bytes before any header decoding.
    for relative in ALLOWED:
        expected = lookup[relative]
        path = ROOT / "data/mechanics/resect-case4-v1" / relative
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError("ORIGINAL_PATH_ALIAS")
        digest = hashlib.sha256()
        md5 = hashlib.md5()
        total = 0
        with path.open("rb") as stream:
            while block := stream.read(1024**2):
                total += len(block)
                digest.update(block)
                md5.update(block)
        if (total, digest.hexdigest(), md5.hexdigest()) != (
            expected["bytes"], expected["sha256"], expected["md5"]
        ):
            raise RuntimeError("ORIGINAL_HASH_MISMATCH")
        records.append({"source_path": relative, "bytes": total,
                        "sha256": digest.hexdigest(), "md5": md5.hexdigest(),
                        "source_verified": True})

    selected = {}
    for record in records:
        path = ROOT / "data/mechanics/resect-case4-v1" / record["source_path"]
        with gzip.open(path, "rb") as stream:
            raw = stream.read(348)
        if len(raw) != 348:
            raise RuntimeError("INCOMPLETE_NIFTI1_HEADER")
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw), check=False)
        if int(header["sizeof_hdr"]) != 348 or bytes(header["magic"]) != b"n+1\x00":
            raise RuntimeError("EXPECTED_ORIGINAL_NIFTI1_SINGLE_FILE")
        shape = tuple(int(item) for item in header.get_data_shape())
        if len(shape) != 3 or min(shape) < 1:
            raise RuntimeError("EXPECTED_THREE_DIMENSIONAL_HEADER")
        corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in shape])))
        qform, qcode = header.get_qform(coded=True)
        sform, scode = header.get_sform(coded=True)
        affine = header.get_best_affine()
        if not np.isfinite(affine).all():
            raise RuntimeError("NONFINITE_HEADER_AFFINE")
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        gram = (affine[:3, :3] / spacing).T @ (affine[:3, :3] / spacing)
        physical = nib.affines.apply_affine(affine, corners)
        qvs = None
        if qcode and scode:
            qvs = float(np.max(np.linalg.norm(
                nib.affines.apply_affine(qform, corners) -
                nib.affines.apply_affine(sform, corners), axis=1)))
        record["header"] = {
            "raw_header_sha256": hashlib.sha256(raw).hexdigest(),
            "decompressed_bytes_returned": 348,
            "nifti_header_check_fix_applied": False,
            "shape": list(shape), "dtype": str(header.get_data_dtype()),
            "endianness": header.endianness,
            "spatial_units": header.get_xyzt_units()[0],
            "time_units": header.get_xyzt_units()[1],
            "zooms": list(map(float, header.get_zooms())),
            "pixdim": header["pixdim"].astype(float).tolist(),
            "vox_offset": float(header["vox_offset"]),
            "scl_slope": scalar(header["scl_slope"]),
            "scl_inter": scalar(header["scl_inter"]),
            "qform_code": int(qcode), "sform_code": int(scode),
            "qform": None if qform is None else qform.tolist(),
            "sform": None if sform is None else sform.tolist(),
            "quaternion_bcd": [float(header[name]) for name in
                               ("quatern_b", "quatern_c", "quatern_d")],
            "qoffset_xyz": [float(header[name]) for name in
                            ("qoffset_x", "qoffset_y", "qoffset_z")],
            "selected_by_nifti_header": "sform" if scode else "qform" if qcode else "fallback_UNVERIFIED",
            "selected_affine": affine.tolist(),
            "native_axis_codes": list(nib.aff2axcodes(affine)),
            "determinant": float(np.linalg.det(affine[:3, :3])),
            "column_lengths_native_units": spacing.tolist(),
            "normalized_gram": gram.tolist(),
            "maximum_absolute_gram_off_identity": float(np.max(np.abs(gram-np.eye(3)))),
            "full_cell_bounds_world_native_units": {
                "minimum": physical.min(axis=0).tolist(),
                "maximum": physical.max(axis=0).tolist(),
            },
            "qform_sform_full_cell_corner_max_difference_native_units": qvs,
        }
        selected[path.name] = (shape, affine)
    t1shape, t1affine = selected["Case4-T1.nii.gz"]
    flairshape, flairaffine = selected["Case4-FLAIR.nii.gz"]
    result = {
        "status": "completed_header_audit_not_anatomical_or_landmark_frame_acceptance",
        "finished_at_utc": now(), "source_receipt_sha256": ACQUISITION_SHA,
        "probe_sha256": sha(Path(__file__)),
        "root_release_sha256": sha(HERE / "root-release.json"),
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "nibabel": nib.__version__, "numpy": np.__version__,
                    "nibabel_path": str(Path(nib.__file__).resolve()),
                    "numpy_path": str(Path(np.__file__).resolve())},
        "files": records,
        "T1_FLAIR_header_grid": {
            "same_shape": t1shape == flairshape,
            "exact_same_affine": bool(np.array_equal(t1affine, flairaffine)),
            "maximum_absolute_affine_entry_difference": float(np.max(np.abs(t1affine-flairaffine))),
            "T1_voxel_to_FLAIR_voxel_from_headers_only": (np.linalg.inv(flairaffine) @ t1affine).tolist(),
            "registration_verified": False,
        },
        "patient_image_arrays_accessed": False,
        "during_US_file_opened": False, "landmark_files_opened": False,
        "source_files_modified": False,
        "segmentation_or_registration_performed": False,
        "landmark_frame_qc_accepted": False,
        "elapsed_seconds": time.monotonic()-started,
        "child_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "Header affine axes are declared NIfTI coordinates, not independently reviewed anatomical alignment.",
            "No image intensities, brain envelope, landmark positions, or MRI-US transform assessed.",
            "gzip may buffer compressed data; exactly 348 decompressed header bytes per image are requested and interpreted.",
        ],
    }
    save("header-receipt.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        audit()
        return
    if (HERE / "header-receipt.json").exists() or (HERE / "resource-receipt.json").exists():
        raise SystemExit("Immutable receipt already exists; do not overwrite or retry here")
    save("root-release.json", {
        "recorded_at_utc": now(), "authorizer": "/root",
        "access_contract_commit": "cb9f71b", "acquisition_commit": "65bf883",
        "allowed_originals": list(ALLOWED), "allowed": "verify compressed hashes then native NIfTI headers only",
        "closed": ["all image arrays", "during-US file", "all landmark files", "inference", "registration", "model solve"],
        "wall_limit_seconds": WALL_SECONDS, "combined_sampled_rss_limit_bytes": RSS_BYTES,
        "dependency_probe": "psutil absent; no installation. Existing Python/nibabel/numpy only; system ps monitors RSS.",
        "probe_sha256": sha(Path(__file__)),
    })
    start = time.monotonic()
    max_rss = 0
    failure = None
    with (HERE / "probe.log").open("x") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child"], stdout=log, stderr=log)
        while child.poll() is None:
            rss_result = subprocess.run(
                ["ps", "-o", "rss=", "-p", f"{os.getpid()},{child.pid}"],
                capture_output=True, text=True, timeout=2, check=False)
            rss = sum(int(line) * 1024 for line in rss_result.stdout.splitlines() if line.strip())
            max_rss = max(max_rss, rss)
            if time.monotonic()-start > WALL_SECONDS or rss > RSS_BYTES:
                failure = "WALL_OR_RSS_LIMIT_EXCEEDED"
                child.kill()
                break
            time.sleep(.025)
        code = child.wait(timeout=2)
    save("resource-receipt.json", {
        "status": "passed" if code == 0 and failure is None else "failed",
        "exit_code": code, "failure": failure,
        "elapsed_seconds": time.monotonic()-start,
        "wall_limit_seconds": WALL_SECONDS,
        "maximum_sampled_parent_plus_child_rss_bytes": max_rss,
        "rss_limit_bytes": RSS_BYTES,
        "sampling_caveat": "Sampled ps RSS is not continuous enforcement; child ru_maxrss independently records its lifetime peak.",
        "probe_sha256": sha(Path(__file__)),
        "header_receipt_sha256": sha(HERE / "header-receipt.json") if (HERE / "header-receipt.json").exists() else None,
        "log_sha256": sha(HERE / "probe.log"),
    })
    print(json.dumps({"exit_code": code, "failure": failure, "elapsed_seconds": time.monotonic()-start}))
    raise SystemExit(code if code else int(failure is not None))


if __name__ == "__main__":
    main()
