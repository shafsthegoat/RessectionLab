"""Independent constructed-file controls; no real patient headers, pixels or network."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from scripts import inspect_rhuh_single_image as inspection


def nifti1(*, shape=(2, 3, 4), endian="<", datatype=16, bitpix=32,
           offset=352., units=2, qform=0, sform=1, pixels=None, slope=1., intercept=0.):
    """Manual NIfTI-1 single-file fixture, independent of imaging libraries."""
    header = bytearray(348)
    struct.pack_into(endian + "i", header, 0, 348)
    dimensions = [len(shape), *shape, *([1] * (7 - len(shape)))]
    struct.pack_into(endian + "8h", header, 40, *dimensions)
    struct.pack_into(endian + "h", header, 70, datatype)
    struct.pack_into(endian + "h", header, 72, bitpix)
    struct.pack_into(endian + "8f", header, 76, 1., 1.5, 2., 2.5, 1., 1., 1., 1.)
    struct.pack_into(endian + "f", header, 108, offset)
    struct.pack_into(endian + "2f", header, 112, slope, intercept)
    header[123] = units
    header[148:176] = b"FORBIDDEN_HEADER_TEXT_OUTCOME"[:28]
    struct.pack_into(endian + "2h", header, 252, qform, sform)
    struct.pack_into(endian + "4f", header, 280, -1.5, 0, 0, 10.)
    struct.pack_into(endian + "4f", header, 296, 0, 2., 0, -20.)
    struct.pack_into(endian + "4f", header, 312, 0, 0, 2.5, 30.)
    header[344:348] = b"n+1\0"
    if pixels is None:
        pixels = np.arange(24, dtype=endian + "f4").tobytes()
    # Do not allocate attacker-declared dimensions/offsets in negative fixtures.
    return bytes(header) + b"\0" * 4 + pixels


def inspect(raw):
    return inspection.inspect_stream(io.BytesIO(gzip.compress(raw, mtime=0)))


def test_valid_manual_little_and_big_endian_streams_do_not_emit_free_text():
    for endian in ("<", ">"):
        report = inspect(nifti1(endian=endian))
        assert isinstance(report, dict)
        assert "FORBIDDEN_HEADER_TEXT" not in json.dumps(report)


@pytest.mark.parametrize("change", ["truncated_footer", "corrupt_crc", "second_member", "trailing_zero", "trailing_text"])
def test_complete_single_gzip_member_and_footer_are_required(change):
    packed = gzip.compress(nifti1(), mtime=0)
    if change == "truncated_footer":
        packed = packed[:-4]
    elif change == "corrupt_crc":
        data = bytearray(packed); data[-8] ^= 0x20; packed = bytes(data)
    elif change == "second_member":
        packed += gzip.compress(b"", mtime=0)
    elif change == "trailing_zero":
        packed += b"\0"
    else:
        packed += b"unaccepted tail"
    with pytest.raises(inspection.Rejected):
        inspection.inspect_stream(io.BytesIO(packed))


@pytest.mark.parametrize("kwargs", [
    {"shape": (0, 3, 4)}, {"shape": (-1, 3, 4)}, {"shape": (513, 2, 2)},
    {"shape": (511, 511, 511)}, {"shape": (2, 3, 4, 1)},
    {"datatype": 32, "bitpix": 64}, {"datatype": 16, "bitpix": 8},
    {"offset": 351.}, {"offset": 352.5}, {"offset": float("nan")},
    {"offset": 2_097_152.},
])
def test_header_bounds_reject_without_voxel_array_materialization(monkeypatch, kwargs):
    raw = nifti1(**kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid header reached voxel array interpretation")
    monkeypatch.setattr(inspection, "_intensity", forbidden)
    with pytest.raises(inspection.Rejected):
        inspect(raw)


@pytest.mark.parametrize("pixels", [np.full(24, np.nan, dtype="<f4").tobytes(),
    np.full(24, np.inf, dtype="<f4").tobytes()])
def test_nonfinite_intensities_are_counted_without_scientific_acceptance(pixels):
    result = inspect(nifti1(pixels=pixels))
    assert result["binary_structure_valid"] is True
    assert result["scientific_use_released"] is False
    assert result["clinical_validation"] is False
    assert result["intensity"]["scaled_nonfinite_voxels"] == 24
    assert result["intensity"]["scaled_finite_voxels"] == 0
    assert result["intensity"]["scaled_finite_min"] is None
    assert result["intensity"]["scaled_finite_max"] is None
    assert result["intensity"]["scaled_finite_mean"] is None
    json.dumps(result, allow_nan=False)


def test_declared_voxel_storage_truncation_and_extra_decompressed_bytes_are_rejected():
    for raw in (nifti1(pixels=b"\0" * 92), nifti1() + b"\0"):
        with pytest.raises(inspection.Rejected):
            inspect(raw)


def nifti2(endian):
    header = bytearray(540)
    struct.pack_into(endian + "i", header, 0, 540)
    header[4:12] = b"n+2\0\r\n\x1a\n"
    struct.pack_into(endian + "2h", header, 12, 16, 32)
    struct.pack_into(endian + "8q", header, 16, 3, 2, 3, 4, 1, 1, 1, 1)
    struct.pack_into(endian + "8d", header, 104, 1., 1.5, 2., 2.5, 1., 1., 1., 1.)
    struct.pack_into(endian + "q", header, 168, 544)
    struct.pack_into(endian + "2d", header, 176, 2., -5.)
    struct.pack_into(endian + "2i", header, 344, 0, 1)
    struct.pack_into(endian + "4d", header, 400, -1.5, 0, 0, 10.)
    struct.pack_into(endian + "4d", header, 432, 0, 2., 0, -20.)
    struct.pack_into(endian + "4d", header, 464, 0, 0, 2.5, 30.)
    struct.pack_into(endian + "i", header, 500, 2)
    return bytes(header) + b"\0" * 4 + np.arange(24, dtype=endian + "f4").tobytes()


@pytest.mark.parametrize("endian", ["<", ">"])
def test_manual_nifti2_has_exact_scaled_values_and_reflected_source_frame(endian):
    result = inspect(nifti2(endian))
    assert result["nifti_version"] == 2
    assert result["shape"] == [2, 3, 4]
    assert result["uncompressed_bytes"] == 640
    assert result["intensity"]["scaled_finite_min"] == -5.
    assert result["intensity"]["scaled_finite_max"] == 41.
    assert result["intensity"]["scaled_finite_mean"] == 18.
    assert result["geometry"]["sform"]["axis_codes"] == ["L", "A", "S"]


def test_report_whitelist_and_unknown_frame_are_distinct_from_invalid_binary():
    report = inspect(nifti1(units=0, qform=0, sform=0))
    assert set(report) == {
        "schema", "status", "binary_structure_valid", "shape", "dtype", "nifti_version",
        "voxel_count", "declared_voxel_bytes", "voxel_offset", "extension_count",
        "uncompressed_bytes", "compressed_bytes", "compressed_sha256", "candidate_compressed_md5",
        "gzip_footer_and_single_member_verified", "geometry", "intensity", "clinical_validation",
        "scientific_use_released",
    }
    assert report["binary_structure_valid"] is True
    assert report["geometry"]["issues"] == [
        "physical_units_unresolved", "world_transform_absent_no_fallback_assumed"]
    for name in ("qform", "sform"):
        assert report["geometry"][name]["affine_in_source_units"] is None
        assert report["geometry"][name]["axis_codes"] is None
    assert report["geometry"]["scanner_or_atlas_provenance_verified"] is False


def test_nonfinite_header_affine_is_reported_without_nan_or_fallback():
    raw = bytearray(nifti1())
    struct.pack_into("<f", raw, 280, float("nan"))
    report = inspect(bytes(raw))
    assert report["binary_structure_valid"] is True
    assert "sform_invalid_geometry" in report["geometry"]["issues"]
    assert report["geometry"]["sform"]["affine_in_source_units"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad_length", [0, 8, 17, 48, -16])
def test_extension_lengths_fail_before_intensity(bad_length, monkeypatch):
    raw = nifti1(offset=384.)
    extension = struct.pack("<ii", bad_length, 6) + b"PRIVATE OUTCOME".ljust(24, b"\0")
    raw = raw[:348] + b"\1\0\0\0" + extension + raw[352:]
    monkeypatch.setattr(inspection, "_intensity", lambda *args: pytest.fail("invalid extension interpreted"))
    with pytest.raises(inspection.Rejected, match="extension"):
        inspect(raw)


def test_unmarked_nonzero_padding_and_pair_magic_are_not_fallback_formats():
    raw = nifti1(offset=368.)
    bad_padding = raw[:352] + b"x" * 16 + raw[352:]
    pair = bytearray(nifti1()); pair[344:348] = b"ni1\0"
    for invalid in (bad_padding, bytes(pair)):
        with pytest.raises(inspection.Rejected):
            inspect(invalid)


def test_compressed_and_declared_storage_caps_precede_intensity(monkeypatch):
    packed = gzip.compress(nifti1(), mtime=0)
    monkeypatch.setattr(inspection, "_intensity", lambda *a: pytest.fail("unbounded intensity interpretation"))
    monkeypatch.setattr(inspection, "MAX_COMPRESSED", len(packed) - 1)
    with pytest.raises(inspection.Rejected, match="compressed_cap"):
        inspection.inspect_stream(io.BytesIO(packed))
    monkeypatch.setattr(inspection, "MAX_COMPRESSED", len(packed))
    monkeypatch.setattr(inspection, "MAX_VOXEL_BYTES", 95)
    with pytest.raises(inspection.Rejected, match="voxel_storage_cap"):
        inspection.inspect_stream(io.BytesIO(packed))


def binding(root, path, value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False).encode()
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}


class ReceiptGraph:
    """Independent inert receipt graph, never any real acquisition metadata."""

    def __init__(self, root, monkeypatch):
        self.root = root
        self.packed = gzip.compress(nifti1(), mtime=0)
        token = hashlib.md5(self.packed).hexdigest()
        monkeypatch.setattr(inspection, "PUBLISHED_TOKEN", token)
        index = binding(root, "a/index.sums", f"{token} {inspection.SOURCE[1:]}\n".encode())
        monkeypatch.setattr(inspection, "INDEX_SHA256", index["sha256"])
        proposal = binding(root, "a/proposal.json", {"selection": {
            "public_source_path": inspection.SOURCE, "patient": "RHUH-0001", "visit": 0,
            "modality_source_label": "T1", "published_checksum_token": token,
            "checksum_index_sha256": index["sha256"],
        }})
        monkeypatch.setattr(inspection, "PROPOSAL_SHA256", proposal["sha256"])
        self.payload = {
            "path": "outputs/rhuh-single-image-v2/quarantine/run-analytic/RHUH-0001_0_t1.nii.gz",
            "measured_compressed_bytes": len(self.packed), "expected_compressed_bytes": None,
            "prior_length_match_claim": False, "sha256": hashlib.sha256(self.packed).hexdigest(),
            "candidate_md5_of_original_compressed_bytes": token, "published_checksum_token": token,
            "candidate_matches_published_token": True, "publisher_algorithm_confirmed": False,
            "acceptance_scope": "format_byte_domain_compatibility_only", "decoded": False,
            "scientific_use_released": False,
        }
        self.worker_path = "outputs/rhuh-single-image-v2/quarantine/run-analytic-receipt.json"
        self.worker = {
            "status": inspection.SUCCESS, "source": inspection.SOURCE,
            "proposal_sha256": proposal["sha256"], "checksum_index_sha256": index["sha256"],
            "resolved_source_sha256": inspection.RESOLVED_SOURCE_SHA256,
            "client_exit_code": 0, "transfer_started": True, "decoded": False,
            "scientific_use_released": False, "payload": self.payload,
        }
        self.parent = {"accepted": True, "status": inspection.SUCCESS,
                       "receipt": self.worker_path, "final_payload": self.payload}
        sources = {name: binding(root, f"a/{name}.py", b"# inert fixture source\n")
                   for name in ("image_helper", "public_transfer", "checksum_helper")}
        release = binding(root, "a/transfer-release.json", {"analytical_fixture": True})
        self.request = {
            "schema": "resectionlab.rhuh-image-inspection-request.v1", "source": inspection.SOURCE,
            "proposal": proposal, "checksum_index": index, "acquisition_sources": sources,
            "transfer_release": release, "payload": self.payload,
        }
        self.execution_edits = {}
        self.review_edits = {}
        self.reviewer = binding(root, "a/reviewer.py", b"# inert independent fixture source\n")

    def seal(self):
        request = self.request
        request["worker_receipt"] = binding(self.root, self.worker_path, self.worker)
        request["parent_summary"] = binding(self.root, "a/parent.json", self.parent)
        execution = {
            "exit_code": 0, "parent_summary_sha256": request["parent_summary"]["sha256"],
            "parent_summary": self.parent, "release_sha256": request["transfer_release"]["sha256"],
            "source_closure_unchanged": True, **self.execution_edits,
        }
        request["parent_execution"] = binding(self.root, "a/execution.json", execution)
        review = {
            "schema": "resectionlab.rhuh-compressed-byte-reconciliation.v1", "accepted": True,
            "source": inspection.SOURCE, "proposal_sha256": inspection.PROPOSAL_SHA256,
            "checksum_index_sha256": inspection.INDEX_SHA256,
            "resolved_source_sha256": inspection.RESOLVED_SOURCE_SHA256,
            "payload": self.payload, "immutable_original": True, "publisher_algorithm_confirmed": False,
            "prior_length_match_claim": False, "decoded": False, "scientific_use_released": False,
            "reviewer_source": self.reviewer,
            **{key: request[key] for key in ("parent_summary", "parent_execution", "worker_receipt",
                                            "acquisition_sources", "transfer_release")},
            **self.review_edits,
        }
        request["reconciliation"] = binding(self.root, "a/review.json", review)
        return request

    def execution_bindings(self):
        request = binding(self.root, "a/request.json", self.seal())
        release = {
            "schema": "resectionlab.rhuh-image-inspection-release.v1", "released": True,
            "action": "bounded_header_and_voxel_inspection_once", "request": request,
            "inspector_source_sha256": hashlib.sha256(Path(inspection.__file__).read_bytes()).hexdigest(),
            "source": inspection.SOURCE,
        }
        return request, release


def test_metadata_preflight_has_no_payload_file_and_no_decoder(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    monkeypatch.setattr(inspection, "inspect_stream", lambda *a: pytest.fail("preflight attempted decoding"))
    assert not (tmp_path / graph.payload["path"]).exists()
    result = inspection.preflight(tmp_path, graph.seal())
    assert result["patient_bytes_opened"] is False
    assert result["scientific_use_released"] is False


@pytest.mark.parametrize("section,key,value", [
    ("parent", "accepted", False), ("parent", "status", "worker_only_pass"),
    ("parent", "receipt", "elsewhere.json"),
    ("worker", "resolved_source_sha256", "0" * 64), ("worker", "source", "wrong_patient_t1.nii.gz"),
    ("worker", "proposal_sha256", "0" * 64), ("worker", "checksum_index_sha256", "0" * 64),
    ("worker", "client_exit_code", False), ("worker", "decoded", True),
    ("worker", "transfer_started", False),
    ("execution_edits", "exit_code", False), ("execution_edits", "exit_code", 1),
    ("execution_edits", "parent_summary_sha256", "0" * 64),
    ("execution_edits", "parent_summary", {}),
    ("execution_edits", "release_sha256", "0" * 64),
    ("execution_edits", "source_closure_unchanged", False),
    ("review_edits", "accepted", False), ("review_edits", "immutable_original", False),
    ("review_edits", "publisher_algorithm_confirmed", True),
    ("review_edits", "decoded", True), ("review_edits", "parent_execution", {}),
    ("review_edits", "acquisition_sources", {}), ("review_edits", "transfer_release", {}),
    ("payload", "expected_compressed_bytes", 123),
    ("payload", "prior_length_match_claim", True), ("payload", "scientific_use_released", True),
    ("payload", "path", "outputs/rhuh-single-image-v2/quarantine/run-analytic/RHUH-0001_0_flair.nii.gz"),
    ("payload", "path", "outputs/rhuh-single-image-v2/quarantine/run-analytic/../RHUH-0001_0_t1.nii.gz"),
    ("payload", "measured_compressed_bytes", True),
    ("payload", "measured_compressed_bytes", 64 * 1024**2 + 1),
    ("payload", "sha256", "a" * 63),
])
def test_resealed_receipts_cannot_substitute_acceptance_source_or_scope(tmp_path, monkeypatch, section, key, value):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    getattr(graph, section)[key] = value
    monkeypatch.setattr(inspection, "inspect_stream", lambda *a: pytest.fail("rejected input reached decoder"))
    with pytest.raises(inspection.Rejected):
        inspection.preflight(tmp_path, graph.seal())


@pytest.mark.parametrize("field", ["proposal", "checksum_index", "parent_summary", "parent_execution",
                                    "worker_receipt", "reconciliation"])
def test_bound_metadata_byte_drift_is_rejected(tmp_path, monkeypatch, field):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request = graph.seal()
    with (tmp_path / request[field]["path"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(inspection.Rejected, match="hash"):
        inspection.preflight(tmp_path, request)


@pytest.mark.parametrize("mutation", ["unreleased", "source_drift", "request_drift", "extra_path"])
def test_separate_release_and_current_source_are_required_before_payload_open(tmp_path, monkeypatch, mutation):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request, release = graph.execution_bindings()
    if mutation == "unreleased":
        release["released"] = False
    elif mutation == "source_drift":
        release["inspector_source_sha256"] = "0" * 64
    elif mutation == "request_drift":
        release["request"] = {**request, "sha256": "0" * 64}
    else:
        release["postoperative_image"] = "forbidden"
    monkeypatch.setattr(inspection, "inspect_stream", lambda *a: pytest.fail("release gate attempted decoding"))
    assert not (tmp_path / graph.payload["path"]).exists()
    with pytest.raises(inspection.Rejected, match="release"):
        inspection.inspect_authorized(tmp_path, request, binding(tmp_path, "a/release.json", release))


def test_mocked_authorized_wrapper_authenticates_original_bytes_without_actual_decode(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request, release = graph.execution_bindings()
    binding(tmp_path, graph.payload["path"], graph.packed)
    calls = []
    def fake_decoder(stream):
        calls.append(stream.read())
        return {"compressed_sha256": graph.payload["sha256"], "compressed_bytes": len(graph.packed)}
    monkeypatch.setattr(inspection, "inspect_stream", fake_decoder)
    rel = binding(tmp_path, "a/release.json", release)
    result = inspection.inspect_authorized(tmp_path, request, rel)
    assert calls == [graph.packed]
    assert result["bindings"] == {"request": request, "release": rel,
        "reconciliation": graph.request["reconciliation"], "source": inspection.SOURCE}
    (tmp_path / graph.payload["path"]).write_bytes(b"x" * len(graph.packed))
    with pytest.raises(inspection.Rejected, match="identity"):
        inspection.inspect_authorized(tmp_path, request, rel)
    assert len(calls) == 1


def test_symlinked_payload_and_unbound_request_fields_do_not_get_interpreted(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request = graph.seal()
    request["postoperative_outcome"] = "forbidden"
    with pytest.raises(inspection.Rejected, match="schema"):
        inspection.preflight(tmp_path, request)
    del request["postoperative_outcome"]
    original = tmp_path / graph.payload["path"]
    original.parent.mkdir(parents=True)
    outside = tmp_path / "outside.gz"
    outside.write_bytes(graph.packed)
    original.symlink_to(outside)
    with pytest.raises(inspection.Rejected, match="symlink"):
        inspection.preflight(tmp_path, request)


def copied_receipt_request(graph):
    request = graph.seal()
    original = graph.root / request["worker_receipt"]["path"]
    request["worker_receipt"] = binding(graph.root, "a/preserved-worker.json", original.read_bytes())
    review = json.loads((graph.root / request["reconciliation"]["path"]).read_bytes())
    review["worker_receipt"] = request["worker_receipt"]
    request["reconciliation"] = binding(graph.root, "a/review.json", review)
    return request


def test_preserved_worker_copy_must_match_original_bytes_named_by_parent(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request = copied_receipt_request(graph)
    assert request["worker_receipt"]["path"] != graph.parent["receipt"]
    assert inspection.preflight(tmp_path, request)["patient_bytes_opened"] is False
    original = tmp_path / graph.worker_path
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(inspection.Rejected, match="hash"):
        inspection.preflight(tmp_path, request)


def test_different_preserved_worker_copy_cannot_be_resealed_as_original(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request = copied_receipt_request(graph)
    # Same parsed JSON, different bytes: all copy bindings refreshed, original unchanged.
    copied = tmp_path / request["worker_receipt"]["path"]
    request["worker_receipt"] = binding(tmp_path, request["worker_receipt"]["path"], copied.read_bytes() + b" ")
    review = json.loads((tmp_path / request["reconciliation"]["path"]).read_bytes())
    review["worker_receipt"] = request["worker_receipt"]
    request["reconciliation"] = binding(tmp_path, "a/review.json", review)
    with pytest.raises(inspection.Rejected, match="hash"):
        inspection.preflight(tmp_path, request)


def test_documentary_allowlist_is_opaque_and_unknown_extras_fail(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    allowed = {"acquisition_source", "checker_development_note", "consulted_inputs",
               "history_checks", "immutable_original_meaning", "independent_computation",
               "limits", "original_mode", "source_archive"}
    # An inert path-looking documentary value must never be followed or reflected.
    graph.review_edits.update({key: {"path": "NO_SUCH_FILE", "note": "DOCUMENTARY_SENTINEL"}
                              for key in allowed})
    report = inspection.preflight(tmp_path, graph.seal())
    assert "DOCUMENTARY_SENTINEL" not in json.dumps(report)
    graph.review_edits["postoperative_outcome"] = "forbidden"
    with pytest.raises(inspection.Rejected, match="reconciliation"):
        inspection.preflight(tmp_path, graph.seal())


def test_original_worker_receipt_symlink_is_rejected_even_with_valid_copy(tmp_path, monkeypatch):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request = copied_receipt_request(graph)
    original = tmp_path / graph.worker_path
    original.unlink()
    original.symlink_to(tmp_path / request["worker_receipt"]["path"])
    with pytest.raises(inspection.Rejected, match="symlink"):
        inspection.preflight(tmp_path, request)


@pytest.mark.parametrize("key", ["image_helper", "public_transfer", "checksum_helper"])
def test_source_closure_actual_byte_drift_prevents_preflight(tmp_path, monkeypatch, key):
    graph = ReceiptGraph(tmp_path, monkeypatch)
    request = graph.seal()
    source = tmp_path / request["acquisition_sources"][key]["path"]
    source.write_bytes(source.read_bytes() + b"# changed\n")
    with pytest.raises(inspection.Rejected, match="hash"):
        inspection.preflight(tmp_path, request)
