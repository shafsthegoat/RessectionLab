#!/usr/bin/env python3
"""Offline, separately released inspection of one reconciled RHUH image.

Default preflight opens bound JSON/index metadata only, never the image. The
pure stream inspector is for analytical controls; it provides no acquisition or
scientific-use authority. Patient execution additionally requires all provenance
joins and a separate source-bound release. No network functions are provided.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import zlib

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
PROPOSAL_SHA256 = "d1d8fea9771865bf33f282dfa5edfbd217021e01c7cd3212a513e5db9e8da9b2"
INDEX_SHA256 = "7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130"
RESOLVED_SOURCE_SHA256 = "d88bfbe72422b010c503aeafe777daf5d74847d46f98e6c86662ce201888686d"
PUBLISHED_TOKEN = "a5c579950d0431d04928e5b59e91ce9d"
MAX_COMPRESSED = 64 * 1024**2
MAX_OFFSET = 1024**2
MAX_UNCOMPRESSED = 256 * 1024**2
MAX_VOXEL_BYTES = 255 * 1024**2
MAX_AXIS = 512
MAX_VOXELS = 64_000_000
FRAME_COMPARISON_MM = 0.001  # Reporting tolerance, not anatomical acceptance.
SUCCESS = "compressed_digest_compatible_quarantined"
HEX = re.compile(r"[0-9a-f]{64}\Z")


class Rejected(ValueError):
    """Fixed, safe reason; never include arbitrary header/receipt strings."""


def need(condition, code):
    if not condition:
        raise Rejected(code)


class _GzipReader:
    """Single-member gzip with bounded decompressor output, CRC and EOF checks."""

    def __init__(self, stream):
        self.stream = stream
        self.decoder = zlib.decompressobj(31)
        self.pending = b""
        self.compressed = 0
        self.uncompressed = 0
        self.sha = hashlib.sha256()
        self.md5 = hashlib.md5(usedforsecurity=False)
        self.finished = False

    def read(self, count):
        need(0 <= count <= MAX_UNCOMPRESSED, "uncompressed_request_cap")
        out = bytearray()
        while len(out) < count and not self.finished:
            if self.decoder.eof:
                need(not self.decoder.unused_data and not self.pending, "gzip_trailing_or_multimember")
                need(not self.stream.read(1), "gzip_trailing_or_multimember")
                self.finished = True
                break
            if not self.pending:
                chunk = self.stream.read(min(65536, MAX_COMPRESSED - self.compressed + 1))
                need(bool(chunk), "gzip_truncated")
                self.compressed += len(chunk)
                need(self.compressed <= MAX_COMPRESSED, "compressed_cap")
                self.sha.update(chunk)
                self.md5.update(chunk)
                self.pending = chunk
            try:
                part = self.decoder.decompress(self.pending, count - len(out))
            except zlib.error:
                raise Rejected("gzip_invalid_header_crc_or_stream") from None
            self.pending = self.decoder.unconsumed_tail
            self.uncompressed += len(part)
            need(self.uncompressed <= MAX_UNCOMPRESSED, "uncompressed_cap")
            out.extend(part)
        return bytes(out)

    def exact(self, count):
        result = self.read(count)
        need(len(result) == count, "nifti_truncated")
        return result

    def finish(self):
        need(self.read(1) == b"", "nifti_extra_uncompressed_bytes")
        need(self.finished, "gzip_footer_not_verified")


def _parse_header(reader):
    initial = reader.exact(4)
    candidates = [int.from_bytes(initial, order, signed=True) for order in ("little", "big")]
    size = next((v for v in candidates if v in (348, 540)), None)
    need(size is not None, "unsupported_nifti_header")
    raw = initial + reader.exact(size - 4)
    marker = reader.exact(4)  # Initial decompressed inspection totals 352 or 544.
    cls = nib.Nifti1Header if size == 348 else nib.Nifti2Header
    try:
        header = cls(raw, check=False)
        magic = bytes(header["magic"])
        need(magic == (b"n+1\x00" if size == 348 else b"n+2\x00"), "unsupported_nifti_pair_or_magic")
        if size == 540:
            need(bytes(header["eol_check"]) == b"\r\n\x1a\n", "invalid_nifti2_eol_check")
        dims = [int(v) for v in header["dim"]]
        need(dims[0] == 3 and all(1 <= v <= MAX_AXIS for v in dims[1:4])
             and all(v in (0, 1) for v in dims[4:]), "unsupported_shape")
        shape = tuple(dims[1:4])
        need(tuple(header.get_data_shape()) == shape, "unsupported_shape_hack")
        count = math.prod(shape)
        need(count <= MAX_VOXELS, "voxel_count_cap")
        code = int(header["datatype"])
        need(code in (2, 4, 8, 16, 64, 256, 512, 768, 1024, 1280), "unsupported_scalar_datatype")
        dtype = header.get_data_dtype()
        need(dtype.kind in "iuf" and int(header["bitpix"]) == dtype.itemsize * 8, "datatype_bitpix_mismatch")
        voxel_bytes = count * dtype.itemsize
        need(voxel_bytes <= MAX_VOXEL_BYTES, "voxel_storage_cap")
        offset_value = float(header["vox_offset"])
        need(math.isfinite(offset_value) and offset_value.is_integer(), "invalid_voxel_offset")
        offset = int(offset_value)
        need(size + 4 <= offset <= MAX_OFFSET, "voxel_offset_cap")
        need(offset + voxel_bytes <= MAX_UNCOMPRESSED, "total_uncompressed_cap")
        need(marker[1:] == b"\x00\x00\x00", "invalid_extension_marker")
        slope, intercept = header.get_slope_inter()
        need(not math.isinf(float(header["scl_slope"])), "invalid_scaling")
    except Rejected:
        raise
    except Exception:
        raise Rejected("invalid_nifti_header_semantics") from None
    return header, shape, dtype, offset, voxel_bytes, marker, slope, intercept


def _extensions(raw, marker, endianness):
    if not marker[0]:
        need(not any(raw), "nonzero_unmarked_header_padding")
        return 0
    count = cursor = 0
    while cursor < len(raw):
        need(len(raw) - cursor >= 16, "invalid_extension_length")
        size = int.from_bytes(raw[cursor:cursor + 4], "little" if endianness == "<" else "big", signed=True)
        need(size >= 16 and size % 16 == 0 and size <= len(raw) - cursor, "invalid_extension_length")
        cursor += size
        count += 1
    need(count > 0, "missing_declared_extension")
    return count  # Bodies are intentionally opaque; no text or DICOM parsing.


def _geometry(header, shape):
    issues = []
    try:
        spatial, temporal = header.get_xyzt_units()
    except Exception:
        spatial = temporal = "unsupported_code"
        issues.append("unsupported_units_code")
    scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(spatial)
    if scale is None:
        issues.append("physical_units_unresolved")
    forms = {}
    for name in ("qform", "sform"):
        code = int(header[name + "_code"])
        item = {"code": code, "present": code > 0, "finite": None, "invertible": None,
                "affine_in_source_units": None, "axis_codes": None}
        if code not in (0, 1, 2, 3, 4, 5):
            issues.append(name + "_code_unsupported")
        elif code > 0:
            try:
                affine = getattr(header, "get_" + name)()
                item["finite"] = bool(np.isfinite(affine).all())
                if item["finite"]:
                    item["affine_in_source_units"] = affine.tolist()
                    sign, logdet = np.linalg.slogdet(affine[:3, :3])
                    item["invertible"] = bool(sign != 0 and np.isfinite(logdet))
                    if item["invertible"]:
                        item["axis_codes"] = list(nib.aff2axcodes(affine))
                if item["finite"] is not True or item["invertible"] is not True:
                    issues.append(name + "_invalid_geometry")
            except Exception:
                issues.append(name + "_invalid_geometry")
        forms[name] = item
    mismatch = None
    q, s = forms["qform"], forms["sform"]
    if not q["present"] and not s["present"]:
        issues.append("world_transform_absent_no_fallback_assumed")
    if q["present"] and s["present"]:
        if q["code"] != s["code"]:
            issues.append("qform_sform_frame_codes_differ")
        if q["invertible"] and s["invertible"] and scale is not None:
            corners = np.array(list(itertools.product(*[(-0.5, n - 0.5) for n in shape])))
            delta = np.array(q["affine_in_source_units"]) - np.array(s["affine_in_source_units"])
            mismatch = float(np.linalg.norm(corners @ delta[:3, :3].T + delta[:3, 3], axis=1).max() * scale)
            if not math.isfinite(mismatch):
                mismatch = None
                issues.append("qform_sform_comparison_nonfinite")
            elif mismatch > FRAME_COMPARISON_MM:
                issues.append("qform_sform_numeric_disagreement")
    zooms = [float(v) for v in header.get_zooms()]
    if not all(math.isfinite(v) and v > 0 for v in zooms):
        issues.append("invalid_spatial_zooms")
    return {"spatial_units": spatial, "temporal_units": temporal,
            "zooms_in_source_units": [v if math.isfinite(v) else None for v in zooms],
            **forms, "max_full_cell_corner_difference_mm": mismatch,
            "numeric_comparison_tolerance_mm": FRAME_COMPARISON_MM,
            "issues": sorted(set(issues)), "anatomical_registration_accepted": False,
            "scanner_or_atlas_provenance_verified": False}


def _intensity(data, dtype, count, slope, intercept):
    # Called only after exact declared length and gzip footer/EOF validation.
    values = np.frombuffer(data, dtype=dtype, count=count)
    finite = nonfinite = 0
    low, high, total = math.inf, -math.inf, 0.0
    a, b = (1.0, 0.0) if slope is None else (float(slope), float(intercept))
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, count, 131072):
            chunk = values[start:start + 131072].astype(np.float64) * a + b
            valid = np.isfinite(chunk)
            finite += int(valid.sum())
            nonfinite += int((~valid).sum())
            good = chunk[valid]
            if good.size:
                low = min(low, float(good.min()))
                high = max(high, float(good.max()))
                total += float(good.sum())
    return {"scaled_finite_voxels": finite, "scaled_nonfinite_voxels": nonfinite,
            "scaled_finite_min": low if finite else None, "scaled_finite_max": high if finite else None,
            "scaled_finite_mean": total / finite if finite and math.isfinite(total) else None,
            "effective_slope": a, "effective_intercept": b, "scaling_enabled": slope is not None}


def inspect_stream(stream):
    """Inspect one analytical/bound stream; caller owns provenance authorization."""
    reader = _GzipReader(stream)
    header, shape, dtype, offset, storage, marker, slope, intercept = _parse_header(reader)
    ext_count = _extensions(reader.exact(offset - reader.uncompressed), marker, header.endianness)
    data = reader.exact(storage)
    reader.finish()
    geometry = _geometry(header, shape)
    summary = _intensity(data, dtype, math.prod(shape), slope, intercept)
    return {"schema": "resectionlab.rhuh-single-image-inspection.v1",
            "status": "format_inspected_scientific_use_unreleased", "binary_structure_valid": True,
            "shape": list(shape), "dtype": dtype.str, "nifti_version": 1 if header.sizeof_hdr == 348 else 2,
            "voxel_count": math.prod(shape), "declared_voxel_bytes": storage, "voxel_offset": offset,
            "extension_count": ext_count, "uncompressed_bytes": reader.uncompressed,
            "compressed_bytes": reader.compressed, "compressed_sha256": reader.sha.hexdigest(),
            "candidate_compressed_md5": reader.md5.hexdigest(), "gzip_footer_and_single_member_verified": True,
            "geometry": geometry, "intensity": summary, "clinical_validation": False,
            "scientific_use_released": False}


def _path(root, text):
    need(isinstance(text, str), "invalid_bound_path")
    rel = Path(text)
    need(not rel.is_absolute() and ".." not in rel.parts and str(rel) == text, "invalid_bound_path")
    path = root / rel
    need(not any(p.is_symlink() for p in [path, *path.parents]), "bound_path_symlink")
    return path


def _bytes(root, binding, cap=262144):
    need(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
         and isinstance(binding["sha256"], str) and HEX.fullmatch(binding["sha256"]), "invalid_file_binding")
    path = _path(root, binding["path"])
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        st = os.fstat(stream.fileno())
        need(stat.S_ISREG(st.st_mode) and st.st_size <= cap, "metadata_file_cap_or_type")
        data = stream.read(cap + 1)
        need(len(data) == st.st_size and len(data) <= cap, "metadata_changed_or_over_cap")
    need(hashlib.sha256(data).hexdigest() == binding["sha256"], "metadata_hash_mismatch")
    return data


def _json(root, binding):
    need(isinstance(binding, dict) and isinstance(binding.get("path"), str)
         and Path(binding["path"]).suffix == ".json", "metadata_json_path_required")
    try:
        value = json.loads(_bytes(root, binding))
        need(isinstance(value, dict), "metadata_object_required")
        return value
    except Rejected:
        raise
    except Exception:
        raise Rejected("invalid_metadata_json") from None


def preflight(root, request):
    """Authenticate receipts/path metadata only; no payload open/hash/decode."""
    fields = {"schema", "source", "proposal", "checksum_index", "parent_summary", "parent_execution",
              "worker_receipt", "reconciliation", "payload", "acquisition_sources", "transfer_release"}
    need(set(request) == fields and request["schema"] == "resectionlab.rhuh-image-inspection-request.v1"
         and request["source"] == SOURCE, "inspection_request_schema_or_source")
    need(request["proposal"]["sha256"] == PROPOSAL_SHA256
         and request["checksum_index"]["sha256"] == INDEX_SHA256, "fixed_provenance_binding")
    proposal = _json(root, request["proposal"])
    selection = proposal.get("selection", {})
    need(selection.get("public_source_path") == SOURCE and selection.get("patient") == "RHUH-0001"
         and selection.get("visit") == 0 and selection.get("modality_source_label") == "T1"
         and selection.get("published_checksum_token") == PUBLISHED_TOKEN
         and selection.get("checksum_index_sha256") == INDEX_SHA256, "proposal_selection")
    need(Path(request["checksum_index"]["path"]).suffix == ".sums", "checksum_index_path_required")
    index = _bytes(root, request["checksum_index"], 65536)
    rows = [line.split() for line in index.decode("ascii").splitlines()]
    need([row for row in rows if len(row) == 2 and row[1] == SOURCE.lstrip("/")] ==
         [[PUBLISHED_TOKEN, SOURCE.lstrip("/")]], "index_source_join")
    parent = _json(root, request["parent_summary"])
    execution = _json(root, request["parent_execution"])
    worker = _json(root, request["worker_receipt"])
    review = _json(root, request["reconciliation"])
    _json(root, request["transfer_release"])
    sources = request["acquisition_sources"]
    need(isinstance(sources, dict) and set(sources) ==
         {"image_helper", "public_transfer", "checksum_helper"}, "acquisition_source_closure")
    for binding in sources.values():
        need(Path(binding["path"]).suffix == ".py", "source_code_path_required")
        _bytes(root, binding)
    payload = request["payload"]
    need(isinstance(payload, dict) and payload == parent.get("final_payload") == worker.get("payload")
         == review.get("payload"), "payload_receipt_join")
    need(parent.get("accepted") is True and parent.get("status") == SUCCESS, "parent_transfer_not_accepted")
    need(type(execution.get("exit_code")) is int and execution["exit_code"] == 0
         and execution.get("parent_summary_sha256") == request["parent_summary"]["sha256"]
         and execution.get("parent_summary") == parent
         and execution.get("release_sha256") == request["transfer_release"]["sha256"]
         and execution.get("source_closure_unchanged") is True, "launcher_not_completed")
    need(worker.get("status") == SUCCESS and worker.get("source") == SOURCE
         and worker.get("proposal_sha256") == PROPOSAL_SHA256
         and worker.get("checksum_index_sha256") == INDEX_SHA256
         and worker.get("resolved_source_sha256") == RESOLVED_SOURCE_SHA256
         and type(worker.get("client_exit_code")) is int and worker["client_exit_code"] == 0
         and worker.get("transfer_started") is True
         and worker.get("decoded") is False and worker.get("scientific_use_released") is False,
         "worker_provenance_join")
    review_fields = {"schema", "accepted", "source", "proposal_sha256", "checksum_index_sha256",
                     "resolved_source_sha256", "parent_summary", "parent_execution", "worker_receipt",
                     "payload", "immutable_original", "publisher_algorithm_confirmed", "prior_length_match_claim",
                     "reviewer_source", "acquisition_sources", "transfer_release", "decoded", "scientific_use_released"}
    documentary_fields = {"acquisition_source", "checker_development_note", "consulted_inputs",
                          "history_checks", "immutable_original_meaning", "independent_computation",
                          "limits", "original_mode", "source_archive"}
    need(review_fields <= set(review) <= review_fields | documentary_fields and review["schema"] == "resectionlab.rhuh-compressed-byte-reconciliation.v1"
         and review["accepted"] is True and review["source"] == SOURCE
         and review["proposal_sha256"] == PROPOSAL_SHA256 and review["checksum_index_sha256"] == INDEX_SHA256
         and review["resolved_source_sha256"] == RESOLVED_SOURCE_SHA256,
         "independent_reconciliation_not_accepted")
    for key in ("parent_summary", "parent_execution", "worker_receipt", "acquisition_sources", "transfer_release"):
        need(review[key] == request[key], "review_source_receipt_join")
    need(review["immutable_original"] is True and review["publisher_algorithm_confirmed"] is False
         and review["prior_length_match_claim"] is False and review["decoded"] is False
         and review["scientific_use_released"] is False, "review_scope")
    need(Path(review["reviewer_source"]["path"]).suffix == ".py", "source_code_path_required")
    _bytes(root, review["reviewer_source"])
    need(type(payload.get("measured_compressed_bytes")) is int
         and 0 < payload["measured_compressed_bytes"] <= MAX_COMPRESSED
         and isinstance(payload.get("sha256"), str) and HEX.fullmatch(payload["sha256"])
         and payload.get("candidate_md5_of_original_compressed_bytes") == PUBLISHED_TOKEN
         and payload.get("published_checksum_token") == PUBLISHED_TOKEN
         and payload.get("candidate_matches_published_token") is True
         and payload.get("publisher_algorithm_confirmed") is False
         and payload.get("expected_compressed_bytes") is None and payload.get("prior_length_match_claim") is False
         and payload.get("decoded") is False and payload.get("scientific_use_released") is False
         and payload.get("acceptance_scope") == "format_byte_domain_compatibility_only", "payload_identity_or_scope")
    path = Path(payload.get("path", ""))
    need(path.parts[:3] == ("outputs", "rhuh-single-image-v2", "quarantine")
         and len(path.parts) == 5 and path.parts[3].startswith("run-")
         and path.name == Path(SOURCE).name, "exact_quarantine_payload_path")
    _path(root, str(path))  # Path/symlink metadata only, no patient bytes.
    original_worker = str(path.parent.parent / (path.parent.name + "-receipt.json"))
    need(parent.get("receipt") == original_worker, "worker_quarantine_path_join")
    # A preserved receipt copy may live elsewhere. Its exact bytes must also
    # match the original receipt named by the accepted parent action.
    _bytes(root, {"path": original_worker, "sha256": request["worker_receipt"]["sha256"]})
    return {"status": "preflight_metadata_only_passed", "patient_bytes_opened": False,
            "source": SOURCE, "payload_sha256": payload["sha256"], "scientific_use_released": False}


def inspect_authorized(root, request_binding, release_binding):
    request = _json(root, request_binding)
    preflight(root, request)
    release = _json(root, release_binding)
    need(set(release) == {"schema", "released", "action", "request", "inspector_source_sha256", "source"}
         and release["schema"] == "resectionlab.rhuh-image-inspection-release.v1"
         and release["released"] is True and release["action"] == "bounded_header_and_voxel_inspection_once"
         and release["request"] == request_binding and release["source"] == SOURCE
         and release["inspector_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "inspection_release_mismatch")
    expected = request["payload"]
    path = _path(root, expected["path"])
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size == expected["measured_compressed_bytes"], "original_size_or_type")
        compressed = stream.read(MAX_COMPRESSED + 1)
        after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(identity(before) == identity(after) and len(compressed) == before.st_size, "original_changed_during_read")
    need(hashlib.sha256(compressed).hexdigest() == expected["sha256"]
         and hashlib.md5(compressed, usedforsecurity=False).hexdigest() == PUBLISHED_TOKEN, "original_compressed_identity_changed")
    result = inspect_stream(io.BytesIO(compressed))
    need(result["compressed_sha256"] == expected["sha256"] and result["compressed_bytes"] == before.st_size,
         "decoded_stream_identity_join")
    result["bindings"] = {"request": request_binding, "release": release_binding,
                          "reconciliation": request["reconciliation"], "source": SOURCE}
    # Source strings are fixed provenance only, never inferred modality/anatomy.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Repository-relative metadata request")
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--release-sha256")
    args = parser.parse_args(argv)
    binding = {"path": args.request, "sha256": args.request_sha256}
    try:
        if args.execute:
            need(args.release is not None and args.release_sha256 is not None, "separate_release_required")
            result = inspect_authorized(ROOT, binding, {"path": args.release, "sha256": args.release_sha256})
        else:
            need(args.release is None and args.release_sha256 is None, "release_only_for_execution")
            result = preflight(ROOT, _json(ROOT, binding))
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, Rejected) else "inspection_failed_" + type(exc).__name__
        print(json.dumps({"status": "rejected_no_inspection_result", "reason": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
