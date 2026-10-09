#!/usr/bin/env python3
"""One-shot, restricted local QC of the frozen SynthRAD 1BA336 TRAIN CT/MR.

`preflight` reads small metadata only. `execute` requires a separately frozen
root declaration. This module never releases views or grants scientific use.
No cohort, subject, archive, bounds, or output-directory CLI overrides exist.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "artifacts/synthrad-first-pair-qc-preparation-v1"
OUTPUT = ROOT / "data/qc/synthrad-task1-first-pair-v1"
DECLARATION = ROOT / "build/synthrad-qc-first-pair-v1/execution-declaration.json"
SCHEMA = "synthrad-first-train-pair-qc/1"
SUBJECT = "1BA336"
NAMES = tuple(f"Task1/brain/{SUBJECT}/{m}.nii.gz" for m in ("ct", "mr"))
PINS = {
    "artifacts/synthrad-first-pair-qc-preparation-v1/proposed-qc-contract.json": "1a073d036a6e2c2328bd97720a5c27625fd9f1457f9394ba7070ce0dc0607332",
    "artifacts/synthrad-first-pair-qc-preparation-v1/central-directory.json": "d22c9e6a7dd28c2425b8e7c76b3bd323d6033d57a128e23ca6d3dfe7da4af3a7",
    "artifacts/synthrad-first-pair-qc-preparation-v1/directory-review.json": "96d29da46f2ba4ee856fe04f08cf751671f1b9cf80f4274c0aa7af101081680e",
    "artifacts/synthrad-first-pair-qc-preparation-v1/brain-pair-map.json": "2d3a10f7bdbeec1c6d6499d487e42d789eb8d46f92185389ef6ea76647feee20",
    "artifacts/synthrad-first-pair-qc-preparation-v1/source-evidence.json": "f4f360579c5da1a91059358cd3898a752d304c9b82021da424d15296b182286b",
    "artifacts/synthrad-first-pair-qc-preparation-v1/preparation-independent-review.md": "ae7d2f970c8b07f8a44e26e255c12923b88951ba6365e988be56fc7fc26336c2",
    "manifests/synthrad-component-cohort-v1.json": "3bb4dd48e2e674bd921653d938ec7c7695c77afb8776c517800bbefb6a32ff30",
}
POLICY = {
    "schema": "synthrad-first-pair-display-statistics/1",
    "percentiles": [1, 5, 50, 95, 99],
    "quantiles": "two-pass equal-width 16384-bin histogram of all effective float64 voxels, including zeros; nearest-rank bin midpoint; absolute error <= bin width, no distribution-free rank-error claim",
    "bins": 16384,
    "background": "effective value exactly zero; descriptive fraction only, no anatomical interpretation",
    "numeric_precision": "float64 effective arithmetic; 64-bit integer magnitudes >=2^53 are unsupported and fail closed; previews alone quantize to uint8",
    "clipping": "fractions strictly below/above each display window; extrema fractions are descriptive, not saturation diagnoses",
    "ct_windows": [[-200.0, 200.0], [-1000.0, 2000.0]],
    "ct_window_units": "source effective values; HU and clinical bone interpretation remain unverified",
    "mr_window": "histogram p1 to p99; if equal, use observed min/max; source arbitrary units",
    "fractional_levels": [0.1, 0.5, 0.9],
    "slice_index": "floor(fraction * (axis_size - 1) + 0.5)",
    "planes": "native voxel-axis planes; affine orientation labels and centre RAS mm, no world-axis reformat; oblique remains oblique",
    "pixels": "uint8 display mapping round-to-nearest via floor(255*clip((value-low)/(high-low),0,1)+0.5); nearest-neighbour thumbnail only",
    "panel": "256x256 letterbox with physical in-plane aspect ratio; transpose source plane, increasing second in-plane axis points up",
    "extent": "every native axis-2 slice in source order, 64x64 nearest-neighbour thumbnail; 64 slices per page; no omitted slice; tiny lesions cannot be assessed from thumbnails",
    "max_axis_size": 4096,
    "privacy": "all previews and raw metadata restricted locally (directories 0700, files 0600); portable image release unsupported by this runner",
    "rubrics": {
        "gross_correspondence": "review_pending until human reviews paired common-grid source views for intracranial/skull landmarks and left/right consistency; fail for localized gross mismatch, pending for uncertainty; no quantitative accuracy claim",
        "intracranial_coverage": "review_pending until human reviews every extent thumbnail plus vertex/skull-base source panels; fail for visible missing required intracranial extent, pending if insufficient resolution or uncertain",
        "calvarium_usability": "review_pending until human reviews CT source-scale contrast, skull coverage and defacing/crop/metal/motion defects; source scaling evidence also required before HU or bone claims",
        "visual_privacy": "review_pending until human checks identifying text and residual face features in restricted views; unresolved finding prevents release and admission; no face reconstruction",
    },
}
CLAIMS = {"training_admitted": False, "spatial_planning_admitted": False,
          "scanner_frame_admitted": False, "anatomical_admission": False,
          "source_mask_is_human_anatomical_truth": False, "optimizer_updates": 0,
          "recorded_rl_transitions": 0, "network_requests": 0, "portable_views_released": False}
STAT_KEYS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
TERMINAL_RESERVE = 2 * 1024**2
WORKER_RECEIPT_RESERVE = 1024**2
PARENT_RECEIPT_RESERVE = 65536


class Refusal(Exception):
    """Only controlled codes, never strings from source files, cross receipt boundary."""


class BoundExceeded(Refusal):
    pass


def require(condition, code):
    if not condition:
        raise Refusal(code)


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "unsafe_local_path")
    for item in (path, *path.parents):
        require(not item.is_symlink(), "symlink_local_path")
    return path


def metadata(path, expected=None, limit=2 * 1024**2):
    path = safe_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as f:
        require(stat.S_ISREG(os.fstat(f.fileno()).st_mode), "metadata_not_regular")
        raw = f.read(limit + 1)
    require(len(raw) <= limit, "metadata_size")
    require(expected is None or sha(raw) == expected, "metadata_hash_changed")
    return raw


def stat_record(s):
    return {k: int(getattr(s, k)) for k in STAT_KEYS}


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def private_dir(path):
    safe_path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(stat.S_IMODE(path.stat().st_mode) == 0o700, "output_directory_permissions")


def durable_new(path, raw):
    """Exclusive durable publication; an interrupted partial remains evidence."""
    safe_path(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def output_bytes(path):
    total = 0
    for p in path.rglob("*"):
        require(not p.is_symlink(), "output_symlink")
        if p.is_file():
            total += p.stat().st_size
    return total


def receipt_bytes(output):
    """Shared aggregate for all portable initial, stage and terminal receipts."""
    total = 0
    for path in output.glob("*.json"):
        require(not path.is_symlink(), "receipt_symlink")
        total += path.stat().st_size
    return total


def durable_receipt(path, value, bounds, reserve=0):
    raw = encode(value)
    if receipt_bytes(path.parent) + len(raw) + reserve > bounds["max_receipt_bytes_per_pair"]:
        raise BoundExceeded("aggregate_receipt_cap")
    if output_bytes(path.parent) + len(raw) > bounds["first_pair_new_disk_cap_bytes"]:
        raise BoundExceeded("output_cap")
    durable_new(path, raw)
    return raw


class Budget:
    def __init__(self, output, bounds, deadline):
        self.output, self.bounds, self.deadline = output, bounds, deadline
        self.view_bytes = 0
        self.receipt_bytes = 0
        self.written = output_bytes(output)
        self.peak_rss = rss_bytes()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError()
        self.peak_rss = max(self.peak_rss, rss_bytes())
        if self.peak_rss > self.bounds["worker_rss_stop_bytes"]:
            raise BoundExceeded("rss_cap")
        if shutil.disk_usage(self.output).free < self.bounds["free_disk_reserve_bytes"]:
            raise BoundExceeded("free_disk_reserve")

    def reserve(self, count, kind="data"):
        self.check()
        if count < 0 or self.written + count > self.bounds["first_pair_new_disk_cap_bytes"] - TERMINAL_RESERVE:
            raise BoundExceeded("output_cap")
        if kind == "view":
            if self.view_bytes + count > self.bounds["max_qc_view_bytes_per_pair"]:
                raise BoundExceeded("view_cap")
            self.view_bytes += count
        if kind == "receipt":
            if receipt_bytes(self.output) + count > self.bounds["max_receipt_bytes_per_pair"] - WORKER_RECEIPT_RESERVE - PARENT_RECEIPT_RESERVE:
                raise BoundExceeded("receipt_cap")
            self.receipt_bytes += count
        self.written += count

    def save(self, path, raw, kind="receipt"):
        self.reserve(len(raw), kind)
        durable_new(path, raw)
        return {"path": str(path.relative_to(self.output)), "bytes": len(raw), "sha256": sha(raw)}


def zip_row(info):
    return {"name": info.filename, "directory": info.is_dir(), "zip_entry_bytes": info.file_size,
            "compressed_bytes": info.compress_size, "compression_method": info.compress_type,
            "crc32": f"{info.CRC:08x}", "header_offset": info.header_offset,
            "flag_bits": info.flag_bits, "create_system": info.create_system,
            "external_attr": info.external_attr, "extra_bytes": len(info.extra),
            "extra_sha256": sha(info.extra), "comment_bytes": len(info.comment),
            "comment_sha256": sha(info.comment)}


def validate_paths(rows):
    seen = set()
    for r in rows:
        name = r["name"]
        require(name and "\\" not in name and "\0" not in name and not name.startswith("/")
                and ":" not in name and all(ord(c) >= 32 for c in name), "unsafe_zip_path")
        require(all(p not in ("", ".", "..") for p in name.rstrip("/").split("/")), "unsafe_zip_path")
        require(str(PurePosixPath(name)) == name.rstrip("/"), "noncanonical_zip_path")
        require(name.casefold() not in seen, "duplicate_zip_path")
        seen.add(name.casefold())
        require(not r["flag_bits"] & 1, "encrypted_zip_member")
        mode = r["external_attr"] >> 16
        require(stat.S_IFMT(mode) in (0, stat.S_IFDIR if r["directory"] else stat.S_IFREG), "nonregular_zip_member")


class ArchiveReader:
    """One descriptor, stat continuity before/after every bounded archive read.

    Initially the central directory/end records are the only readable range.
    Member ranges are added only by exact TRAIN allowlist checks, separately.
    """
    def __init__(self, path, expected_stat, central_start, check):
        self.path = safe_path(path)
        self.expected = expected_stat
        self.handle = os.fdopen(os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
        self.ranges = [(central_start, expected_stat["st_size"])]
        self.read_bytes = 0
        self.check = check
        try:
            self.continuity()
        except BaseException:
            self.close()
            raise

    def continuity(self):
        require(stat_record(os.fstat(self.handle.fileno())) == self.expected
                and stat_record(self.path.lstat()) == self.expected, "archive_continuity_changed")
        require(stat.S_ISREG(self.expected["st_mode"]), "archive_not_regular")
        safe_path(self.path)

    def read(self, count=-1):
        self.check()
        self.continuity()
        start = self.tell()
        if count < 0:
            count = self.expected["st_size"] - start
        require(0 <= count <= 2 * 1024**2, "archive_read_size")
        require(any(lo <= start and start + count <= hi for lo, hi in self.ranges), "archive_read_outside_allowlist")
        raw = self.handle.read(count)
        self.read_bytes += len(raw)
        self.continuity()
        return raw

    def seek(self, *args):
        return self.handle.seek(*args)

    def tell(self):
        return self.handle.tell()

    def seekable(self):
        return True

    def close(self):
        self.handle.close()


def local_header(reader, row, end):
    """No archive member stream is opened until all local/central checks pass."""
    start = row["header_offset"]
    require(start + 30 <= end, "local_header_bounds")
    reader.ranges.append((start, start + 30))
    reader.seek(start)
    raw = reader.read(30)
    require(len(raw) == 30, "local_header_truncated")
    magic, version, flags, method, mtime, mdate, crc, compressed, size, nlen, xlen = struct.unpack("<4s5H3I2H", raw)
    require(magic == b"PK\x03\x04" and flags == row["flag_bits"] == 0
            and method == row["compression_method"] == 8
            and crc == int(row["crc32"], 16) and compressed == row["compressed_bytes"]
            and size == row["zip_entry_bytes"], "local_central_mismatch")
    name = row["name"].encode("ascii")
    require(nlen == len(name) and xlen <= 65535 and start + 30 + nlen + xlen + compressed <= end,
            "local_header_bounds")
    reader.ranges.append((start, start + 30 + nlen + xlen))
    raw = reader.read(nlen + xlen)
    require(raw[:nlen] == name, "local_name_mismatch")
    # ZIP local and central extra fields legitimately differ. Parse framing and
    # preserve only their hash. Unknown field semantics do not clear privacy.
    extra, cursor = raw[nlen:], 0
    while cursor < len(extra):
        require(cursor + 4 <= len(extra), "local_extra_framing")
        kind, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4 + size
        require(cursor <= len(extra) and kind != 1, "local_extra_framing")
    reader.ranges.append((start, start + 30 + nlen + xlen + compressed))
    return {"data_offset": start + 30 + nlen + xlen, "local_extra_sha256": sha(extra), "local_extra_bytes": xlen,
            "local_header_sha256": sha(struct.pack("<4s5H3I2H", magic, version, flags, method, mtime, mdate,
                                                  crc, compressed, row["zip_entry_bytes"], nlen, xlen) + raw)}


def strict_zip_member(reader, row, data_offset, budget):
    """Validate raw deflate itself, not ZipExtFile's advertised-size clipping.

    A ZIP stream with understated size/CRC, a second deflate stream, trailing
    compressed bytes, a truncated end marker or expansion beyond the row fails.
    Only the frozen selected compressed range can be read.
    """
    require(row["name"] in NAMES, "zip_payload_scope")
    reader.seek(data_offset)
    decoder = zlib.decompressobj(-15)
    pending, remaining, count, crc = b"", row["compressed_bytes"], 0, 0
    while not decoder.eof:
        budget.check()
        if not pending:
            require(remaining > 0, "zip_deflate_truncated")
            expected = min(65536, remaining)
            pending = reader.read(expected)
            require(len(pending) == expected, "zip_compressed_truncated")
            remaining -= len(pending)
        block = decoder.decompress(pending, min(1024**2, row["zip_entry_bytes"] - count + 1))
        pending = decoder.unconsumed_tail
        count += len(block)
        require(count <= row["zip_entry_bytes"], "zip_expansion")
        crc = zlib.crc32(block, crc)
        if block:
            yield block
    require(not pending and not decoder.unused_data and remaining == 0, "zip_compressed_trailing_bytes")
    require(count == row["zip_entry_bytes"] and crc == int(row["crc32"], 16), "zip_size_or_crc")


def extract_pair(reader, frozen_rows, pair, budget, restricted, progress=None):
    require(pair["subject_id"] == SUBJECT and pair["role"] == "TRAIN"
            and pair["patient_group"] == "SynthRAD2023:" + SUBJECT
            and tuple(r["name"] for r in pair["members"]) == NAMES, "first_pair_scope")
    validate_paths(frozen_rows)
    with zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
        require([zip_row(i) for i in infos] == frozen_rows and not archive.comment, "directory_changed")
        by_name = {i.filename: i for i in infos}
        offsets = sorted([i.header_offset for i in infos] + [archive.start_dir])
        results = []
        for row in pair["members"]:
            modality = Path(row["name"]).name.split(".")[0]
            if progress is not None:
                progress(modality, "started", None)
            budget.check()
            require(zip_row(by_name[row["name"]]) == row, "member_identity_changed")
            require(not row["directory"] and 0 < row["zip_entry_bytes"] <= budget.bounds["max_extracted_gzip_member_bytes"], "member_size_cap")
            end = next(x for x in offsets if x > row["header_offset"])
            local = local_header(reader, row, end)
            path = restricted / Path(row["name"]).name
            count, digest = 0, hashlib.sha256()
            # ZipFile is directory-only. No ZipExtFile, extract or testzip use.
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as out:
                for block in strict_zip_member(reader, row, local["data_offset"], budget):
                    count += len(block)
                    budget.reserve(len(block))
                    digest.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            require(count == row["zip_entry_bytes"], "zip_truncated")
            reader.continuity()
            proof = {"member": row["name"], "bytes": count, "sha256": digest.hexdigest(),
                     "staged_stat": stat_record(path.stat()), "zip_crc32_verified": row["crc32"], **local}
            receipt = budget.save(budget.output / (modality + "-extraction.json"), encode(proof))
            if progress is not None:
                progress(modality, "completed", receipt)
            results.append(proof)
        reader.continuity()
        return results


def gzip_metadata(path):
    """Bound optional fields before gzip decoding; disclose only lengths/hashes."""
    with os.fdopen(os.open(safe_path(path), os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        prefix = source.read(65536)
    require(len(prefix) >= 10 and prefix[:3] == b"\x1f\x8b\x08" and not prefix[3] & 0xe0, "gzip_header")
    flags, cursor, fields = prefix[3], 10, {}
    if flags & 4:
        require(cursor + 2 <= len(prefix), "gzip_extra_truncated")
        size = struct.unpack_from("<H", prefix, cursor)[0]
        cursor += 2
        require(cursor + size <= len(prefix), "gzip_header_cap")
        fields["extra"] = prefix[cursor:cursor + size]
        cursor += size
    for name, flag in (("filename", 8), ("comment", 16)):
        if flags & flag:
            end = prefix.find(b"\0", cursor)
            require(end >= 0, "gzip_header_cap")
            fields[name] = prefix[cursor:end]
            cursor = end + 1
    if flags & 2:
        require(cursor + 2 <= len(prefix), "gzip_header_crc_truncated")
        require(struct.unpack_from("<H", prefix, cursor)[0] == zlib.crc32(prefix[:cursor]) & 0xffff, "gzip_header_crc")
        cursor += 2
    result = {name: {"bytes": len(value), "sha256": sha(value), "review_pending": bool(value)} for name, value in fields.items()}
    return {"fields": result, "metadata_review_pending": any(bool(v) for v in fields.values()),
            "header_sha256": sha(prefix[:cursor])}, prefix[:cursor]


class StrictGzip:
    """Bounded single-member gzip with compressed fixity and CRC verification.

    gzip.GzipFile accepts concatenated members, whose optional metadata would
    escape the privacy-prefix review. This reader refuses all trailing bytes.
    The staged file identity is checked on the same descriptor at every read.
    """
    def __init__(self, path, proof):
        self.path, self.proof = safe_path(path), proof
        self.source = os.fdopen(os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
        self.decoder = zlib.decompressobj(31)
        self.pending = b""
        self.digest = hashlib.sha256()
        self.finished = False
        try:
            self.continuity()
        except BaseException:
            self.source.close()
            raise

    def continuity(self):
        require(stat_record(os.fstat(self.source.fileno())) == self.proof["staged_stat"]
                and stat_record(self.path.lstat()) == self.proof["staged_stat"], "staged_continuity_changed")

    def read(self, count):
        require(0 <= count <= 1024**2, "gzip_read_cap")
        self.continuity()
        result = bytearray()
        while len(result) < count and not self.finished:
            if not self.pending:
                self.pending = self.source.read(65536)
                self.digest.update(self.pending)
                require(bool(self.pending), "gzip_truncated")
            part = self.decoder.decompress(self.pending, count - len(result))
            result.extend(part)
            self.pending = self.decoder.unconsumed_tail
            if self.decoder.eof:
                require(not self.decoder.unused_data and not self.source.read(1), "gzip_trailing_compressed_bytes")
                require(self.digest.hexdigest() == self.proof["sha256"], "staged_gzip_sha256_changed")
                self.finished = True
        self.continuity()
        return bytes(result)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.source.close()


def float_encoding(value):
    value = float(value)
    return value if math.isfinite(value) else "nan" if math.isnan(value) else "positive_inf" if value > 0 else "negative_inf"


def corners(shape):
    import numpy as np
    return np.array([[i, j, k, 1] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])


def geometry(header, shape):
    import nibabel as nib
    import numpy as np
    units = int(header["xyzt_units"]) & 7
    require(units in (1, 2, 3), "unknown_spatial_units")
    factor = {1: 1000.0, 2: 1.0, 3: .001}[units]
    qcode, scode = int(header["qform_code"]), int(header["sform_code"])
    require(qcode in range(6) and scode in range(6) and (qcode or scode), "active_transform_code")
    forms = []
    for code, getter in ((qcode, header.get_qform), (scode, header.get_sform)):
        if not code:
            continue  # Never decode or validate inactive quaternion encoding.
        if getter == header.get_qform:
            require(float(header["pixdim"][0]) in (-1.0, 1.0), "active_qfac")
        form = getter().copy()
        form[:3] *= factor
        require(np.isfinite(form).all() and np.array_equal(form[3], [0, 0, 0, 1]), "invalid_affine")
        det = float(np.linalg.det(form[:3, :3]))
        require(math.isfinite(det) and abs(det) >= 1e-12, "singular_affine")
        spacing = np.linalg.norm(form[:3, :3], axis=0)
        dirs = form[:3, :3] / spacing
        require(np.allclose(dirs.T @ dirs, np.eye(3), atol=1e-4, rtol=0), "unsupported_shear")
        require(np.allclose(spacing, np.array(header["pixdim"][1:4]) * factor, atol=.01, rtol=1e-4), "spacing_disagreement")
        forms.append(form)
    residual = 0.0 if len(forms) == 1 else float(np.max(np.linalg.norm((corners(shape) @ (forms[0] - forms[1]).T)[:, :3], axis=1)))
    require(residual <= .01, "qform_sform_disagreement")
    affine = forms[-1]
    world = (corners(shape) @ affine.T)[:, :3]
    orientation = nib.aff2axcodes(affine)
    require(all(c is not None for c in orientation), "orientation_unresolved")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return {"affine_ras_mm": affine.tolist(), "spacing_mm": spacing.tolist(), "shape": shape,
            "orientation": list(orientation), "qform_code": qcode, "sform_code": scode,
            "qform_status": "active" if qcode else "inactive_not_decoded",
            "qform_sform_corner_residual_mm": residual,
            "source_unit_code": units, "unit_scale_to_mm": factor,
            "world_corner_bounds_mm": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
            "field_of_view_mm": (spacing * shape).tolist(), "voxel_volume_mm3": float(abs(np.linalg.det(affine[:3, :3]))),
            "handedness": "right" if np.linalg.det(affine[:3, :3]) > 0 else "left",
            "obliquity_degrees": np.rad2deg(nib.affines.obliquity(affine)).tolist(),
            "claim": "source coded geometry only; no scanner or anatomical registration admission"}


def parse_header(stream, bounds):
    import nibabel as nib
    import numpy as np
    raw = stream.read(348)
    require(len(raw) == 348, "nifti_header_truncated")
    header = nib.Nifti1Header(binaryblock=raw, check=False)
    require(int(header["sizeof_hdr"]) == 348 and bytes(header["magic"]) == b"n+1\0", "nifti1_single_file_required")
    require(int(header["dim"][0]) == 3, "scalar_3d_required")
    shape = [int(v) for v in header["dim"][1:4]]
    require(all(0 < n <= POLICY["max_axis_size"] for n in shape), "axis_size_cap")
    dtype = header.get_data_dtype()
    require(dtype.kind in "iuf" and dtype.itemsize in (1, 2, 4, 8)
            and int(header["bitpix"]) == dtype.itemsize * 8, "unsupported_scalar_dtype")
    voxels = math.prod(shape)
    if voxels > bounds["max_voxels_per_image"] or voxels * dtype.itemsize > bounds["max_native_payload_bytes_per_image"]:
        raise BoundExceeded("decoded_payload_cap")
    offset = float(header["vox_offset"])
    require(math.isfinite(offset) and offset.is_integer() and 352 <= offset <= bounds["max_nifti_data_offset"]
            and int(offset) % 16 == 0, "nifti_offset")
    tail = stream.read(int(offset) - 348)
    require(len(tail) == int(offset) - 348 and tail[0] in (0, 1) and tail[1:4] == b"\0\0\0", "extension_flag")
    extensions = []
    if tail[0]:
        cursor = 4
        while cursor < len(tail):
            require(cursor + 8 <= len(tail), "extension_framing")
            size, code = struct.unpack_from(header.endianness + "ii", tail, cursor)
            require(size >= 16 and size % 16 == 0 and cursor + size <= len(tail) and code >= 0, "extension_framing")
            value = tail[cursor + 8:cursor + size]
            extensions.append({"code": code, "bytes": len(value), "sha256": sha(value), "privacy": "opaque_review_pending"})
            cursor += size
    else:
        require(not any(tail[4:]), "unexpected_extension_padding")
    raw_slope, raw_inter = float(header["scl_slope"]), float(header["scl_inter"])
    # NIfTI inactive slope (zero or NaN) explicitly ignores stored intercept.
    if raw_slope == 0 or math.isnan(raw_slope):
        slope, inter, active = 1.0, 0.0, False
    else:
        require(math.isfinite(raw_slope) and math.isfinite(raw_inter), "nonfinite_active_scaling")
        slope, inter, active = raw_slope, raw_inter, True
    spacing = np.array(header["pixdim"][1:4], dtype=float)
    require(np.isfinite(spacing).all() and (spacing > 0).all(), "header_spacing")
    fields = {name: raw[start:end] for name, start, end in (("descrip", 148, 228), ("aux_file", 228, 252), ("intent_name", 328, 344))}
    privacy = {k: {"sha256": sha(v), "nonempty": bool(v.strip(b"\0 "))} for k, v in fields.items()}
    # Fixed upper bound covers float64 values, native bytes, histogram indices,
    # masks and display-map temporaries; no full-array percentile operation.
    chunk = min(65536, bounds["scalar_chunk_voxels"], (bounds["max_scalar_workspace_bytes"] - 1024**2) // 96)
    require(chunk > 0, "scalar_workspace_cap")
    return {"header": header, "raw": raw, "tail": tail, "shape": shape, "dtype": dtype,
            "voxels": voxels, "offset": int(offset), "slope": slope, "inter": inter, "chunk": chunk,
            "summary": {"header_sha256": sha(raw), "extensions": extensions, "text_fields": privacy,
                        "metadata_privacy": "review_pending" if extensions or any(x["nonempty"] for x in privacy.values()) else "no_nonempty_text_or_extension",
                        "dtype_native": dtype.str, "dtype_effective": "float64", "shape": shape,
                        "native_payload_bytes": voxels * dtype.itemsize, "data_offset": int(offset),
                        "raw_scaling": [float_encoding(raw_slope), float_encoding(raw_inter)],
                        "effective_scaling": [slope, inter], "scaling_active": active,
                        "scalar_workspace_bytes_bound": chunk * 96 + 1024**2}}


def scalar_chunks(path, info, budget):
    import numpy as np
    with StrictGzip(path, info["proof"]) as stream:
        prefix = stream.read(info["offset"])
        require(prefix == info["raw"] + info["tail"], "staged_header_changed")
        remaining = info["voxels"]
        while remaining:
            budget.check()
            count = min(remaining, info["chunk"])
            raw = stream.read(count * info["dtype"].itemsize)
            require(len(raw) == count * info["dtype"].itemsize, "scalar_truncated")
            values = np.frombuffer(raw, dtype=info["dtype"]).astype(np.float64)
            if info["dtype"].kind in "iu" and info["dtype"].itemsize == 8:
                require((np.abs(values) < 2**53).all(), "integer_precision_unsupported")
            with np.errstate(over="ignore", invalid="ignore"):
                values *= info["slope"]
                values += info["inter"]
            require(np.isfinite(values).all(), "nonfinite_effective_values")
            yield values
            remaining -= count
        require(not stream.read(1), "unexpected_decompressed_trailing_bytes")


def statistics(path, info, budget):
    import numpy as np
    minimum, maximum, zeros = math.inf, -math.inf, 0
    for values in scalar_chunks(path, info, budget):
        minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
        zeros += int(np.count_nonzero(values == 0))
    require(minimum < maximum, "constant_image")
    width = (maximum - minimum) / POLICY["bins"]
    require(math.isfinite(width) and width > 0, "histogram_numeric_range_unsupported")
    histogram = np.zeros(POLICY["bins"], dtype=np.int64)
    extrema = [0, 0]
    for values in scalar_chunks(path, info, budget):
        extrema[0] += int(np.count_nonzero(values == minimum))
        extrema[1] += int(np.count_nonzero(values == maximum))
        indices = np.minimum(((values - minimum) / width).astype(np.int64), POLICY["bins"] - 1)
        histogram += np.bincount(indices, minlength=POLICY["bins"])
    require(int(histogram.sum()) == info["voxels"], "histogram_count")
    cumulative = np.cumsum(histogram)
    quantiles = {}
    for percentile in POLICY["percentiles"]:
        index = int(np.searchsorted(cumulative, max(1, math.ceil(percentile * info["voxels"] / 100))))
        quantiles[str(percentile)] = min(maximum, minimum + (index + .5) * width)
    return {"status": "finite_nonconstant", "voxels": info["voxels"], "minimum": minimum, "maximum": maximum,
            "effective_zero_fraction": zeros / info["voxels"], "extrema_fractions": [n / info["voxels"] for n in extrema],
            "percentiles": quantiles, "quantile_absolute_error_bound": width,
            "histogram_counts_sha256": sha(histogram.astype("<i8").tobytes()),
            "range_interpretation": "source-scale descriptive values; clinical CT scaling pending; MRI arbitrary",
            "flags": ["clinical_scale_review_pending"]}


def thumbnail(plane, spacings, side):
    """PIL only sees bounded uint8 display pixels, never raw metadata strings."""
    import numpy as np
    from PIL import Image
    physical = np.array(plane.shape) * spacings
    size = np.maximum(1, np.floor(physical / physical.max() * side + .5).astype(int))
    sample0 = np.floor(np.linspace(0, plane.shape[0] - 1, size[0]) + .5).astype(int)
    sample1 = np.floor(np.linspace(0, plane.shape[1] - 1, size[1]) + .5).astype(int)
    image = Image.fromarray(plane[np.ix_(sample0, sample1)].T[::-1])
    canvas = Image.new("L", (side, side))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def render_views(path, info, scalar, grid, modality, restricted, budget):
    import numpy as np
    from PIL import Image, ImageDraw
    windows = POLICY["ct_windows"] if modality == "ct" else [[scalar["percentiles"]["1"], scalar["percentiles"]["99"]]]
    if windows[0][0] == windows[0][1]:
        windows = [[scalar["minimum"], scalar["maximum"]]]
    require(info["voxels"] <= budget.bounds["max_pair_float32_arrays_bytes"], "preview_allocation_cap")
    result = []
    affine = np.array(grid["affine_ras_mm"])
    for window_id, (low, high) in enumerate(windows):
        budget.check()
        display = np.empty(info["voxels"], dtype=np.uint8)
        cursor, clipped = 0, [0, 0]
        for values in scalar_chunks(path, info, budget):
            clipped[0] += int(np.count_nonzero(values < low))
            clipped[1] += int(np.count_nonzero(values > high))
            values -= low
            values /= high - low
            np.clip(values, 0, 1, out=values)
            values *= 255
            values += .5
            display[cursor:cursor + len(values)] = values.astype(np.uint8)
            cursor += len(values)
        display = display.reshape(info["shape"], order="F")
        panels = Image.new("L", (3 * 256, 3 * 300))
        draw = ImageDraw.Draw(panels)
        coordinates = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for column, fraction in enumerate(POLICY["fractional_levels"]):
                index = math.floor(fraction * (info["shape"][axis] - 1) + .5)
                selection = [slice(None)] * 3
                selection[axis] = index
                panels.paste(thumbnail(display[tuple(selection)], np.array(grid["spacing_mm"])[others], 256), (column * 256, axis * 300))
                point = [(size - 1) / 2 for size in info["shape"]] + [1]
                point[axis] = index
                world = (affine @ point)[:3].tolist()
                labels = [grid["orientation"][i] for i in others]
                draw.text((column * 256 + 2, axis * 300 + 257), f"axis {axis} index {index}; right {labels[0]}, up {labels[1]}\nRAS mm: " + ",".join(f"{v:.1f}" for v in world), fill=255)
                coordinates.append({"axis": axis, "fraction": fraction, "index": index, "centre_ras_mm": world})
        def save_image(image, name):
            budget.check()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            return budget.save(restricted / name, buffer.getvalue(), "view")
        files = [save_image(panels, f"{modality}-window{window_id}-planes.png")]
        for first in range(0, info["shape"][2], 64):
            contact = Image.new("L", (8 * 64, 8 * 80))
            draw = ImageDraw.Draw(contact)
            for index in range(first, min(first + 64, info["shape"][2])):
                cell = index - first
                x, y = (cell % 8) * 64, (cell // 8) * 80
                contact.paste(thumbnail(display[:, :, index], np.array(grid["spacing_mm"][:2]), 64), (x, y))
                draw.text((x + 1, y + 64), f"k={index}", fill=255)
            files.append(save_image(contact, f"{modality}-window{window_id}-extent-{first:04d}.png"))
        del display
        result.append({"window": [low, high], "source_units_not_verified_HU": modality == "ct",
                       "clipping_fractions": [n / info["voxels"] for n in clipped],
                       "coordinates": coordinates, "files": files, "extent_axis": 2,
                       "extent_slice_count": info["shape"][2], "privacy_release": "restricted_only"})
    return result


def inspect_image(path, extracted, modality, restricted, budget, progress=None):
    from resectionlab.nifti_header_records import nifti1_header_record_coded_v2
    stage_evidence = {}
    def mark(stage, status, proof=None):
        if progress is not None:
            progress(stage, status, proof)
        record = {"schema": SCHEMA + "-stage", "modality": modality, "stage": stage,
                  "status": status, "source_sha256": extracted["sha256"], "evidence": proof}
        stage_evidence[stage] = budget.save(budget.output / f"{modality}-stage-{stage}-{status}.json", encode(record))

    mark("gzip_metadata", "started")
    gmeta, graw = gzip_metadata(path)
    proof = budget.save(restricted / f"{modality}-gzip-header.bin", graw, "data")
    mark("gzip_metadata", "completed", proof)
    mark("header", "started")
    with StrictGzip(path, extracted) as stream:
        info = parse_header(stream, budget.bounds)
    info["proof"] = extracted
    # Original bytes are restricted even when geometry/scalar checks later fail.
    budget.save(restricted / f"{modality}-nifti-prefix.bin", info["raw"] + info["tail"], "data")
    raw_record = nifti1_header_record_coded_v2(info["raw"], extracted["sha256"])
    budget.save(restricted / f"{modality}-header-record-v2.json", encode(raw_record), "data")
    evidence = {"header": budget.save(budget.output / f"{modality}-header.json", encode({"source_sha256": extracted["sha256"], "header": info["summary"], "gzip_metadata": gmeta}))}
    mark("header", "completed", evidence["header"])
    mark("geometry", "started")
    grid = geometry(info["header"], info["shape"])
    evidence["grid"] = budget.save(budget.output / f"{modality}-grid.json", encode(grid))
    mark("geometry", "completed", evidence["grid"])
    mark("scalar", "started")
    scalar = statistics(path, info, budget)
    evidence["scalar"] = budget.save(budget.output / f"{modality}-scalar.json", encode(scalar))
    mark("scalar", "completed", evidence["scalar"])
    mark("privacy_preview", "started")
    views = render_views(path, info, scalar, grid, modality, restricted, budget)
    view_proof = budget.save(budget.output / f"{modality}-views.json", encode(views))
    mark("privacy_preview", "completed", view_proof)
    # Explicit construction: no raw header, base64, text, extension bytes,
    # uncontrolled errors or patient-identifying strings enter portable JSON.
    return {"source": extracted, "gzip_metadata": gmeta, "header": info["summary"],
            "grid": grid, "scalar": scalar, "restricted_views": views, "stage_evidence": evidence, "stage_journals": stage_evidence,
            "deidentification": "review_pending", "anatomy": "review_pending"}


def compare_grid(images):
    import numpy as np
    first, second = [images[m]["grid"] for m in ("ct", "mr")]
    first_world = corners(first["shape"]) @ np.array(first["affine_ras_mm"]).T
    second_world = corners(second["shape"]) @ np.array(second["affine_ras_mm"]).T
    residual = float(np.max(np.linalg.norm((first_world - second_world)[:, :3], axis=1)))
    same = first["shape"] == second["shape"]
    return {"equal_shapes": same, "ct_grid_corner_displacement_mm": residual,
            "status": "source_coded_grid_correspondence" if same and residual <= .01 else "grid_review_pending",
            "anatomical_registration": "review_pending"}


def execution_identity():
    names = ("scripts/synthrad_first_pair_qc_v1.py", "tests/test_synthrad_first_pair_qc_v1.py",
             "src/resectionlab/nifti_header_records.py", "src/resectionlab/structural_evidence.py",
             "src/resectionlab/core.py", "src/resectionlab/data_policy.py",
             "src/resectionlab/__init__.py")
    return {"files": {n: sha(metadata(ROOT / n)) for n in names}, "python": sys.version,
            "executable": str(Path(sys.executable).resolve()),
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "nibabel", "pillow")},
            "policy_sha256": sha(encode(POLICY)), "pins": PINS}


def preflight():
    values = {p: metadata(ROOT / p, digest) for p, digest in PINS.items()}
    proposal = json.loads(values["artifacts/synthrad-first-pair-qc-preparation-v1/proposed-qc-contract.json"])
    directory = json.loads(values["artifacts/synthrad-first-pair-qc-preparation-v1/directory-review.json"])
    rows = json.loads(values["artifacts/synthrad-first-pair-qc-preparation-v1/central-directory.json"])["entries"]
    cohort = json.loads(values["manifests/synthrad-component-cohort-v1.json"])
    pairs = json.loads(values["artifacts/synthrad-first-pair-qc-preparation-v1/brain-pair-map.json"])["pairs"]
    source = json.loads(values["artifacts/synthrad-first-pair-qc-preparation-v1/source-evidence.json"])
    for proof in source["source_proofs"]:
        raw = metadata(ROOT / proof["path"], proof["sha256"])
        require(len(raw) == proof["bytes"], "source_proof_size")
    validate_paths(rows)
    train = sorted((m for m in cohort["members"] if m["role"] == "TRAIN"), key=lambda m: (m["rank_within_center"], m["anonymous_center"], m["subject_id"]))
    require([m["subject_id"] for m in train] == proposal["identities"]["full_train_order"], "train_order_changed")
    member = train[0]
    require(member["subject_id"] == SUBJECT and member["rank_within_center"] == 1 and member["anonymous_center"] == "A"
            and member["source_family"] == "SynthRAD2023" and tuple(member["candidate_acquired_members"]) == NAMES, "first_pair_identity")
    pair = next(p for p in pairs if p["subject_id"] == SUBJECT)
    require(all(pair[k] == member[k] for k in ("subject_id", "patient_group", "role", "source_family", "anonymous_center", "rank_within_center", "identity_sha256"))
            and pair["candidate_acquired_members"] == proposal["first_pair"]["members"], "pair_join_changed")
    require(directory["archive_stat_before"] == directory["archive_stat_after"], "archive_prior_continuity")
    archive = safe_path(ROOT / proposal["source"]["archive"])
    require(stat_record(archive.lstat()) == directory["archive_stat_after"], "archive_path_continuity")
    return proposal, directory, rows


def declaration_template():
    proposal, _, _ = preflight()
    return {"schema": SCHEMA + "-declaration", "execution_authorized": False,
            "independent_controls_passed": False, "implementation": execution_identity(),
            "subject_id": SUBJECT, "role": "TRAIN", "members": list(NAMES),
            "bounds": proposal["bounds"], "policy": POLICY,
            "output": str(OUTPUT.relative_to(ROOT)), "offline_evidence": [], **CLAIMS}


def validate_declaration():
    raw = metadata(DECLARATION)
    declaration = json.loads(raw)
    expected = declaration_template()
    require(declaration.get("execution_authorized") is True and declaration.get("independent_controls_passed") is True,
            "root_execution_freeze_missing")
    expected.update(execution_authorized=True, independent_controls_passed=True, offline_evidence=declaration.get("offline_evidence"))
    require(declaration == expected, "execution_declaration_changed")
    proofs = declaration["offline_evidence"]
    require(isinstance(proofs, list) and 1 <= len(proofs) <= 16, "control_evidence_missing")
    for proof in proofs:
        require(set(proof) == {"path", "sha256"} and not Path(proof["path"]).is_absolute()
                and proof["path"].startswith("build/synthrad-qc-"), "control_evidence_path")
        metadata(ROOT / proof["path"], proof["sha256"])
    return declaration, sha(raw)


def error_record(exc):
    if isinstance(exc, BoundExceeded):
        return "bound_exceeded", str(exc)
    if isinstance(exc, TimeoutError):
        return "timeout", "deadline"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "interrupted", "signal_or_interrupt"
    # Refusal codes originate in this file, never interpolate external bytes.
    if isinstance(exc, Refusal):
        return "qc_failed", str(exc)
    return "qc_failed", "untrusted_exception_redacted"


@contextmanager
def termination_guard():
    old = signal.getsignal(signal.SIGTERM)
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def worker(intent_sha):
    intent_raw = metadata(OUTPUT / "intent.json", intent_sha)
    intent = json.loads(intent_raw)
    require(os.getppid() == intent["parent_pid"], "worker_parent_changed")
    declaration, declaration_sha = validate_declaration()
    require(declaration_sha == intent["declaration_sha256"], "worker_declaration_changed")
    durable_receipt(OUTPUT / "worker-started.json", {"intent_sha256": intent_sha, "pid": os.getpid(), **CLAIMS}, declaration["bounds"], WORKER_RECEIPT_RESERVE + PARENT_RECEIPT_RESERVE)
    budget = Budget(OUTPUT, declaration["bounds"], intent["deadline_monotonic"])
    restricted = OUTPUT / "restricted"
    private_dir(restricted)
    result = {"schema": SCHEMA + "-worker", "subject_id": SUBJECT, "patient_group": "SynthRAD2023:" + SUBJECT,
              "role": "TRAIN", "intent_sha256": intent_sha, "declaration_sha256": declaration_sha,
              "stage": "source_binding", "images": {}, "image_stage_evidence": {}, "status": "incomplete", **CLAIMS}
    reader = None
    started = time.monotonic()
    try:
        with termination_guard():
            proposal, directory, rows = preflight()
            result["source_basis"] = {"archive_sha256_from_prior_verification": proposal["source"]["measured_archive_sha256_from_prior_completion"],
                                      "full_archive_rehashed": False, "archive_stat_continuity_required": True,
                                      "doi": proposal["source"]["doi"], "version": proposal["source"]["version"],
                                      "license": "CC-BY-NC-4.0", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                                      "scope": "noncommercial research component QC only",
                                      "attribution": "creators retained in frozen zenodo-record-7260705.json source proof",
                                      "changes": "source gzip unchanged; numeric QC and restricted uint8 display derivatives only",
                                      "source_cohort_directory_proof_hashes": PINS,
                                      "implementation": declaration["implementation"]}
            reader = ArchiveReader(ROOT / proposal["source"]["archive"], directory["archive_stat_after"], directory["central_directory_start"], budget.check)
            result["archive_stat_open"] = stat_record(os.fstat(reader.handle.fileno()))
            result["stage"] = "extraction"
            def extraction_progress(modality, status, proof):
                result["stage"] = modality + "_extraction"
                result["image_stage_evidence"].setdefault(modality, {})["extraction"] = {"status": status, "evidence": proof}
            extracted = extract_pair(reader, rows, proposal["first_pair"], budget, restricted, extraction_progress)
            result["extraction"] = extracted
            for modality, proof in zip(("ct", "mr"), extracted, strict=True):
                def progress(stage, status, evidence):
                    result["stage"] = modality + "_" + stage
                    result["image_stage_evidence"].setdefault(modality, {})[stage] = {"status": status, "evidence": evidence}
                result["images"][modality] = inspect_image(restricted / (modality + ".nii.gz"), proof, modality, restricted, budget, progress)
            result["grid"] = compare_grid(result["images"])
            reader.continuity()
            result["archive_stat_close"] = stat_record(os.fstat(reader.handle.fileno()))
            result["archive_bytes_read"] = reader.read_bytes
            result["status"] = "review_pending"
            result["stage"] = "human_privacy_anatomy_review_pending"
            result["stages"] = {"source_binding": "passed", "extraction": "passed", "header": "passed",
                                "scalar": "passed", "grid": result["grid"]["status"], "deidentification": "review_pending",
                                "anatomy": "review_pending", "overlap": "unresolved", "admission": "not_admitted"}
    except BaseException as exc:
        result["status"], result["error_code"] = error_record(exc)
        result["failure_stage"] = result["stage"]
    finally:
        if reader is not None:
            reader.close()
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_rss_bytes"] = max(budget.peak_rss, rss_bytes())
        result["output_bytes_before_receipt"] = output_bytes(OUTPUT)
        result["dispositions"] = {name: "review_pending" for name in POLICY["rubrics"]}
        result["overlap"] = "unresolved_admission_blocker"
        result["clinical_ct_scaling"] = "review_pending"
        result["extraction_receipts"] = [{"path": p.name, "sha256": sha(metadata(p))} for p in sorted(OUTPUT.glob("*-extraction.json"))]
        # Keep failure accounting possible after a budget refusal.
        if len(encode(result)) > WORKER_RECEIPT_RESERVE:
            result.update(status="bound_exceeded", error_code="worker_terminal_receipt_cap", failure_stage="terminal_receipt")
            result["images"] = {m: {"stage_evidence": value["stage_evidence"], "stage_journals": value["stage_journals"]}
                                for m, value in result["images"].items()}
        if result["peak_rss_bytes"] > declaration["bounds"]["worker_rss_stop_bytes"]:
            result.update(status="bound_exceeded", error_code="rss_cap", failure_stage="terminal_resource_check")
        if time.monotonic() >= budget.deadline and result["status"] == "review_pending":
            result.update(status="timeout", error_code="deadline", failure_stage="terminal_resource_check")
        durable_receipt(OUTPUT / "worker-result.json", result, declaration["bounds"], PARENT_RECEIPT_RESERVE)
    return 0 if result["status"] == "review_pending" else 1


def supervise(command, output, bounds, deadline):
    """No raw worker stdout/stderr survives. Parent owns terminal accounting."""
    environment = os.environ.copy()
    environment.update({n: "1" for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True, env=environment)
    reason, peak = "completed", 0
    try:
        with termination_guard():
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    reason = "timeout"
                    break
                if output_bytes(output) > bounds["first_pair_new_disk_cap_bytes"] - TERMINAL_RESERVE:
                    reason = "bound_exceeded"
                    break
                if shutil.disk_usage(output).free < bounds["free_disk_reserve_bytes"]:
                    reason = "bound_exceeded"
                    break
                # Platform process RSS, in KiB on supported macOS/Linux hosts.
                observed = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, timeout=1, check=False)
                value = observed.stdout.strip()
                if value:
                    require(value.isdigit(), "rss_monitor_unavailable")
                    peak = max(peak, int(value) * 1024)
                    if peak > bounds["worker_rss_stop_bytes"]:
                        reason = "bound_exceeded"
                        break
                elif process.poll() is None:
                    raise Refusal("rss_monitor_unavailable")
                time.sleep(min(.25, max(0, deadline - time.monotonic())))
    except BaseException as exc:
        reason, _ = error_record(exc)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=min(5, max(.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        if reason == "completed" and time.monotonic() >= deadline:
            reason = "timeout"
    return {"supervision": reason, "worker_exit_status": process.returncode,
            "parent_observed_peak_rss_bytes": peak, "parent_reaped_worker": True,
            "worker_log_bytes": 0, "log_policy": "stdout_stderr_discarded_for_privacy"}


def execute():
    started = time.monotonic()
    declaration, declaration_sha = validate_declaration()
    proposal, _, _ = preflight()
    deadline = started + 600
    private_dir(OUTPUT.parent)
    # Fixed directory and exclusive intent prevent retries after any attempt.
    safe_path(OUTPUT)
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    intent = {"schema": SCHEMA + "-intent", "subject_id": SUBJECT, "patient_group": "SynthRAD2023:" + SUBJECT,
              "role": "TRAIN", "attempt_id": "first-pair-attempt-01", "members": list(NAMES),
              "declaration_sha256": declaration_sha, "parent_pid": os.getpid(),
              "deadline_monotonic": deadline, "implementation": declaration["implementation"], **CLAIMS}
    reserve = WORKER_RECEIPT_RESERVE + PARENT_RECEIPT_RESERVE
    raw = durable_receipt(OUTPUT / "intent.json", intent, declaration["bounds"], reserve)
    durable_receipt(OUTPUT / "denominator-initial.json", {"rows": [{"subject_id": s, "role": "TRAIN", "status": "not_attempted"} for s in proposal["identities"]["full_train_order"]]}, declaration["bounds"], reserve)
    outcome = {"schema": SCHEMA + "-parent", "intent_sha256": sha(raw), "subject_id": SUBJECT,
               "role": "TRAIN", "declaration_sha256": declaration_sha, "status": "incomplete", **CLAIMS}
    try:
        outcome.update(supervise([sys.executable, str(Path(__file__).resolve()), "_worker", sha(raw)], OUTPUT, declaration["bounds"], deadline))
        path = OUTPUT / "worker-result.json"
        if path.exists():
            child_raw = metadata(path, limit=declaration["bounds"]["max_receipt_bytes_per_pair"] // 2)
            child = json.loads(child_raw)
            require(child["intent_sha256"] == sha(raw) and child["declaration_sha256"] == declaration_sha
                    and all(child.get(k) == v for k, v in CLAIMS.items()), "worker_receipt_binding")
            outcome["worker_receipt_sha256"] = sha(child_raw)
            outcome["status"] = child["status"] if outcome["supervision"] == "completed" and outcome["worker_exit_status"] == 0 else child["status"] if child["status"] != "review_pending" else "interrupted"
        else:
            outcome["status"] = "incomplete"
        if outcome["supervision"] != "completed":
            outcome["status"] = outcome["supervision"]
    except BaseException as exc:
        outcome["status"], outcome["error_code"] = error_record(exc)
    finally:
        outcome["elapsed_seconds"] = time.monotonic() - started
        outcome["output_bytes_before_parent_receipt"] = output_bytes(OUTPUT)
        outcome["denominator"] = [{"subject_id": s, "role": "TRAIN", "status": outcome["status"] if s == SUBJECT else "not_attempted"} for s in proposal["identities"]["full_train_order"]]
        outcome["parent_completed"] = True
        outcome["stage_journals"] = [{"path": p.name, "sha256": sha(metadata(p))} for p in sorted(OUTPUT.glob("*-stage-*.json"))]
        outcome["extraction_receipts"] = [{"path": p.name, "sha256": sha(metadata(p))} for p in sorted(OUTPUT.glob("*-extraction.json"))]
        require(len(encode(outcome)) <= PARENT_RECEIPT_RESERVE, "parent_receipt_cap")
        durable_receipt(OUTPUT / "parent-result.json", outcome, declaration["bounds"])
    return 0 if outcome["status"] == "review_pending" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "execute", "_worker"))
    parser.add_argument("intent_sha", nargs="?")
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            print(encode(declaration_template()).decode(), end="")
            return 0
        if args.mode == "execute":
            return execute()
        require(args.intent_sha is not None and len(args.intent_sha) == 64, "worker_intent_missing")
        return worker(args.intent_sha)
    except BaseException as exc:
        status, code = error_record(exc)
        print(json.dumps({"status": status, "error_code": code, **CLAIMS}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
