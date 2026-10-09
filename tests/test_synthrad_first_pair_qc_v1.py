"""Offline security/parser controls only; no scientific data or model activity."""
from __future__ import annotations

from copy import deepcopy
import gzip
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
import zipfile
import zlib

import nibabel as nib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synthrad_first_pair_qc_v1.py"
spec = importlib.util.spec_from_file_location("synthrad_first_pair_qc_v1", SCRIPT)
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Offline controls must never use network")
    monkeypatch.setattr(socket, "socket", deny)


@pytest.fixture
def bounds():
    # Frozen metadata, not an image or archive read.
    return json.loads((qc.PREP / "proposed-qc-contract.json").read_text())["bounds"].copy()


@pytest.fixture
def budget(tmp_path, bounds):
    qc.private_dir(tmp_path / "output")
    return qc.Budget(tmp_path / "output", bounds, time.monotonic() + 30)


def nifti_bytes(values=None, modify=None, extension=b"", text=b""):
    values = np.arange(60, dtype=np.int16).reshape((3, 4, 5), order="F") if values is None else np.asarray(values)
    header = nib.Nifti1Header()
    header.set_data_shape(values.shape)
    header.set_data_dtype(values.dtype)
    header.set_xyzt_units("mm")
    header.set_sform(np.eye(4), code=1)
    header["qform_code"] = 0
    header["vox_offset"] = 352 + len(extension)
    header["scl_slope"] = 1
    header["scl_inter"] = 0
    header["descrip"] = text
    if modify:
        modify(header)
    return header.binaryblock + (b"\x01\0\0\0" if extension else bytes(4)) + extension + values.astype(header.get_data_dtype()).tobytes(order="F")


def staged(tmp_path, raw, name="ct.nii.gz", compressed=None):
    payload = gzip.compress(raw, mtime=0) if compressed is None else compressed
    path = tmp_path / name
    qc.durable_new(path, payload)
    return path, {"member": qc.NAMES[0 if name.startswith("ct") else 1], "sha256": qc.sha(payload),
                  "bytes": len(payload), "staged_stat": qc.stat_record(path.stat())}


def parsed(path, proof, bounds):
    with qc.StrictGzip(path, proof) as stream:
        info = qc.parse_header(stream, bounds)
    info["proof"] = proof
    return info


def archive_fixture(tmp_path, payload=None, names=None, method=zipfile.ZIP_DEFLATED):
    path = tmp_path / "synthetic.zip"
    names = list(qc.NAMES) + ["Task1/brain/1BA014/ct.nii.gz", "Task1/brain/1BA336/mask.nii.gz"] if names is None else names
    with zipfile.ZipFile(path, "w", compression=method) as archive:
        for name in names:
            info = zipfile.ZipInfo(name)
            info.compress_type = method
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, gzip.compress(nifti_bytes(), mtime=0) if payload is None else payload)
    with zipfile.ZipFile(path) as archive:
        rows = [qc.zip_row(i) for i in archive.infolist()]
        central = archive.start_dir
    pair = {"subject_id": qc.SUBJECT, "patient_group": "SynthRAD2023:" + qc.SUBJECT,
            "role": "TRAIN", "members": rows[:2]}
    return path, rows, central, pair


def extract_fixture(tmp_path, budget, monkeypatch=None, mutate=None):
    path, rows, central, pair = archive_fixture(tmp_path)
    if mutate:
        mutate(path, rows, central, pair)
    restricted = budget.output / "restricted"
    qc.private_dir(restricted)
    reader = qc.ArchiveReader(path, qc.stat_record(path.stat()), central, budget.check)
    try:
        result = qc.extract_pair(reader, rows, pair, budget, restricted)
    finally:
        reader.close()
    return result, restricted


def deny_member_reads(monkeypatch):
    original = zipfile.ZipFile.open
    def guarded(self, member, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        if mode == "r":
            pytest.fail("member opened")
        return original(self, member, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)


def test_preflight_is_metadata_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Preflight attempted an archive/image operation")
    monkeypatch.setattr(qc, "ArchiveReader", denied)
    monkeypatch.setattr(qc, "StrictGzip", denied)
    proposal, directory, rows = qc.preflight()
    assert proposal["first_pair"]["subject_id"] == "1BA336"
    assert directory["entries"] == len(rows) == 1807
    declaration = qc.declaration_template()
    assert declaration["execution_authorized"] is False
    assert declaration["independent_controls_passed"] is False
    assert declaration["members"] == list(qc.NAMES)


def test_exact_train_pair_only_opens_allowed_members(tmp_path, budget, monkeypatch):
    original = qc.strict_zip_member
    opened = []
    def guarded(reader, row, *args, **kwargs):
        assert row["name"] in qc.NAMES
        opened.append(row["name"])
        return original(reader, row, *args, **kwargs)
    monkeypatch.setattr(qc, "strict_zip_member", guarded)
    deny_member_reads(monkeypatch)
    results, restricted = extract_fixture(tmp_path, budget)
    assert opened == list(qc.NAMES)
    assert [r["member"] for r in results] == list(qc.NAMES)
    assert all((restricted / Path(r["member"]).name).stat().st_mode & 0o777 == 0o600 for r in results)
    assert not (restricted / "mask.nii.gz").exists()


@pytest.mark.parametrize("role", ["SELECT", "MEASUREMENT_EVAL", "train", "TRAIN "])
def test_wrong_role_refused_before_member_open(tmp_path, budget, role, monkeypatch):
    deny_member_reads(monkeypatch)
    with pytest.raises(qc.Refusal, match="first_pair_scope"):
        extract_fixture(tmp_path, budget, mutate=lambda p, r, c, pair: pair.update(role=role))


@pytest.mark.parametrize("change", ["extra", "other", "reordered", "group"])
def test_changed_member_scope_refused_before_open(tmp_path, budget, change, monkeypatch):
    deny_member_reads(monkeypatch)
    def mutate(path, rows, central, pair):
        if change == "extra":
            pair["members"].append(rows[3])
        elif change == "other":
            pair["members"][0] = rows[2]
        elif change == "reordered":
            pair["members"].reverse()
        else:
            pair["patient_group"] = "wrong"
    with pytest.raises(qc.Refusal, match="first_pair_scope"):
        extract_fixture(tmp_path, budget, mutate=mutate)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", "C:bad", "a\x00b", "a\nb"])
def test_unsafe_zip_paths(name):
    row = {"name": name, "flag_bits": 0, "external_attr": (stat.S_IFREG | 0o600) << 16, "directory": False}
    with pytest.raises(qc.Refusal, match="zip_path"):
        qc.validate_paths([row])


def test_duplicate_casefold_and_symlink_members(tmp_path):
    _, rows, _, _ = archive_fixture(tmp_path)
    with pytest.raises(qc.Refusal, match="duplicate"):
        qc.validate_paths([rows[0], {**rows[0], "name": rows[0]["name"].upper()}])
    with pytest.raises(qc.Refusal, match="nonregular"):
        qc.validate_paths([{**rows[0], "external_attr": (stat.S_IFLNK | 0o777) << 16}])
    with pytest.raises(qc.Refusal, match="encrypted"):
        qc.validate_paths([{**rows[0], "flag_bits": 1}])


def test_directory_change_and_member_size_before_open(tmp_path, budget, monkeypatch):
    def mutate(path, rows, central, pair):
        rows[0]["crc32"] = "12345678"
    deny_member_reads(monkeypatch)
    with pytest.raises(qc.Refusal, match="directory_changed"):
        extract_fixture(tmp_path, budget, mutate=mutate)


def test_zip_advertised_size_cap(tmp_path, budget):
    budget.bounds["max_extracted_gzip_member_bytes"] = 1
    with pytest.raises(qc.Refusal, match="member_size_cap"):
        extract_fixture(tmp_path, budget)


def test_local_central_header_mismatch(tmp_path, budget):
    def mutate(path, rows, central, pair):
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, rows[0]["header_offset"] + 14, 0)
        path.write_bytes(raw)
    with pytest.raises(qc.Refusal, match="local_central_mismatch"):
        extract_fixture(tmp_path, budget, mutate=mutate)


def test_local_name_mismatch(tmp_path, budget):
    def mutate(path, rows, central, pair):
        raw = bytearray(path.read_bytes())
        raw[rows[0]["header_offset"] + 30] = ord("X")
        path.write_bytes(raw)
    with pytest.raises(qc.Refusal, match="local_name_mismatch"):
        extract_fixture(tmp_path, budget, mutate=mutate)


def test_bad_outer_zip_crc_preserves_partial(tmp_path, budget):
    def mutate(path, rows, central, pair):
        raw = bytearray(path.read_bytes())
        row = rows[0]
        local = row["header_offset"]
        nlen, xlen = struct.unpack_from("<HH", raw, local + 26)
        raw[local + 30 + nlen + xlen + row["compressed_bytes"] // 2] ^= 1
        path.write_bytes(raw)
    with pytest.raises((qc.Refusal, zlib.error)):
        extract_fixture(tmp_path, budget, mutate=mutate)
    assert (budget.output / "restricted/ct.nii.gz").exists()


def test_archive_replacement_inplace_change_and_read_boundary(tmp_path):
    path, _, central, _ = archive_fixture(tmp_path)
    reader = qc.ArchiveReader(path, qc.stat_record(path.stat()), central, lambda: None)
    try:
        with pytest.raises(qc.Refusal, match="outside_allowlist"):
            reader.read(1)
        reader.seek(central)
        assert reader.read(4) == b"PK\x01\x02"
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with pytest.raises(qc.Refusal, match="continuity"):
            reader.read(1)
    finally:
        reader.close()
    reader = qc.ArchiveReader(path, qc.stat_record(path.stat()), central, lambda: None)
    try:
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        with pytest.raises(qc.Refusal, match="continuity"):
            reader.continuity()
    finally:
        reader.close()


def test_symlink_archive_refused(tmp_path):
    path, _, central, _ = archive_fixture(tmp_path)
    link = tmp_path / "link.zip"
    link.symlink_to(path)
    with pytest.raises(qc.Refusal, match="symlink"):
        qc.ArchiveReader(link, qc.stat_record(path.stat()), central, lambda: None)


@pytest.mark.parametrize("mutation", ["crc", "truncate", "concatenate", "trailing", "expansion"])
def test_inner_gzip_integrity_and_length_fail_closed(tmp_path, budget, mutation):
    raw = nifti_bytes()
    payload = bytearray(gzip.compress(raw, mtime=0))
    if mutation == "crc":
        payload[-8] ^= 1
    elif mutation == "truncate":
        payload = payload[:-3]
    elif mutation == "concatenate":
        payload.extend(gzip.compress(b"", mtime=0))
    elif mutation == "trailing":
        payload.extend(b"x")
    else:
        payload = gzip.compress(raw + bytes(1024**2), mtime=0)
    path, proof = staged(tmp_path, raw, compressed=bytes(payload))
    with pytest.raises(Exception):
        info = parsed(path, proof, budget.bounds)
        qc.statistics(path, info, budget)


def test_staged_changed_descriptor_and_sha(tmp_path, budget):
    path, proof = staged(tmp_path, nifti_bytes())
    bad = {**proof, "sha256": "0" * 64}
    with pytest.raises(qc.Refusal, match="sha256"):
        with qc.StrictGzip(path, bad) as stream:
            stream.read(1024)
    with qc.StrictGzip(path, proof) as stream:
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        with pytest.raises(qc.Refusal, match="continuity"):
            stream.read(1)


@pytest.mark.parametrize("field,value,code", [
    ("sizeof_hdr", 349, "nifti1"), ("magic", b"ni1\0", "nifti1"),
    ("dim", [4, 3, 4, 5, 1, 1, 1, 1], "3d"),
    ("dim", [3, 0, 4, 5, 1, 1, 1, 1], "axis"),
    ("dim", [3, 4097, 4, 5, 1, 1, 1, 1], "axis"),
    ("bitpix", 8, "dtype"), ("vox_offset", 352.5, "offset"),
    ("vox_offset", 2**24, "offset"), ("vox_offset", 351, "offset"),
    ("scl_slope", float("inf"), "scaling"),
    ("scl_inter", float("nan"), "scaling"),
    ("pixdim", [1, -1, 1, 1, 1, 1, 1, 1], "spacing"),
])
def test_nifti_header_refusals(tmp_path, budget, field, value, code):
    raw = nifti_bytes(modify=lambda h: h.__setitem__(field, value))
    path, proof = staged(tmp_path, raw)
    with pytest.raises((qc.Refusal, nib.spatialimages.HeaderDataError), match=code):
        parsed(path, proof, budget.bounds)


def test_native_allocation_cap_before_scalar_access(tmp_path, budget):
    path, proof = staged(tmp_path, nifti_bytes())
    budget.bounds["max_native_payload_bytes_per_image"] = 1
    with pytest.raises(qc.BoundExceeded, match="decoded_payload"):
        parsed(path, proof, budget.bounds)


@pytest.mark.parametrize("slope,inter", [(0, float("inf")), (float("nan"), float("nan"))])
def test_inactive_scaling_is_recorded_and_ignored(tmp_path, budget, slope, inter):
    def modify(h):
        h["scl_slope"], h["scl_inter"] = slope, inter
    path, proof = staged(tmp_path, nifti_bytes(modify=modify))
    info = parsed(path, proof, budget.bounds)
    assert info["summary"]["scaling_active"] is False
    assert info["summary"]["effective_scaling"] == [1, 0]
    assert qc.statistics(path, info, budget)["maximum"] == 59


def test_active_negative_scaling_and_histogram_error(tmp_path, budget):
    def modify(h):
        h["scl_slope"], h["scl_inter"] = -2, 4
    path, proof = staged(tmp_path, nifti_bytes(modify=modify))
    info = parsed(path, proof, budget.bounds)
    result = qc.statistics(path, info, budget)
    values = np.sort(np.arange(60) * -2 + 4)
    assert result["minimum"] == -114
    assert result["maximum"] == 4
    for p in qc.POLICY["percentiles"]:
        expected = values[max(1, math.ceil(p * len(values) / 100)) - 1]
        assert abs(result["percentiles"][str(p)] - expected) <= result["quantile_absolute_error_bound"]
    assert result["effective_zero_fraction"] == 1 / 60
    assert info["summary"]["scalar_workspace_bytes_bound"] <= budget.bounds["max_scalar_workspace_bytes"]


@pytest.mark.parametrize("values", [np.zeros((3, 4, 5), np.int16), np.full((3, 4, 5), np.nan), np.full((3, 4, 5), np.inf)])
def test_constant_and_nonfinite_scalar_failures(tmp_path, budget, values):
    path, proof = staged(tmp_path, nifti_bytes(values))
    info = parsed(path, proof, budget.bounds)
    with pytest.raises(qc.Refusal, match="constant|nonfinite"):
        qc.statistics(path, info, budget)


def test_inactive_invalid_quaternion_not_decoded(tmp_path, budget):
    def modify(h):
        h["pixdim"][0] = 7
        h["quatern_b"] = np.nan
        h["quatern_c"] = 5
    path, proof = staged(tmp_path, nifti_bytes(modify=modify))
    info = parsed(path, proof, budget.bounds)
    result = qc.geometry(info["header"], info["shape"])
    assert result["qform_status"] == "inactive_not_decoded"


@pytest.mark.parametrize("kind", ["unknown_units", "no_transform", "unknown_code", "qfac", "bad_quaternion", "singular", "shear", "spacing", "forms"])
def test_affine_failures(tmp_path, budget, kind):
    def modify(h):
        if kind == "unknown_units":
            h["xyzt_units"] = 0
        elif kind == "no_transform":
            h["sform_code"] = 0
        elif kind == "unknown_code":
            h["sform_code"] = 99
        elif kind in ("qfac", "bad_quaternion", "forms"):
            h.set_qform(np.eye(4), code=1)
            if kind == "qfac":
                h["pixdim"][0] = 0
            elif kind == "bad_quaternion":
                h["quatern_b"] = 2
            else:
                h["qoffset_x"] = .1
        elif kind == "singular":
            h["srow_x"] = 0
        elif kind == "shear":
            h["srow_x"][1] = .1
        else:
            h["pixdim"][1] = 2
    path, proof = staged(tmp_path, nifti_bytes(modify=modify))
    info = parsed(path, proof, budget.bounds)
    with pytest.raises(Exception):
        qc.geometry(info["header"], info["shape"])


@pytest.mark.parametrize("unit,scale", [("meter", 1000), ("micron", .001), ("mm", 1)])
def test_explicit_units_convert_to_mm(tmp_path, budget, unit, scale):
    path, proof = staged(tmp_path, nifti_bytes(modify=lambda h: h.set_xyzt_units(unit)))
    info = parsed(path, proof, budget.bounds)
    grid = qc.geometry(info["header"], info["shape"])
    assert grid["spacing_mm"] == [scale] * 3


def test_grid_comparison_keeps_anatomy_pending():
    grid = {"shape": [3, 4, 5], "affine_ras_mm": np.eye(4).tolist()}
    result = qc.compare_grid({"ct": {"grid": grid}, "mr": {"grid": grid}})
    assert result["status"] == "source_coded_grid_correspondence"
    assert result["anatomical_registration"] == "review_pending"
    other = {**grid, "shape": [3, 4, 6]}
    result = qc.compare_grid({"ct": {"grid": grid}, "mr": {"grid": other}})
    assert result["status"] == "grid_review_pending"
    assert result["ct_grid_corner_displacement_mm"] == 1


def test_extensions_and_text_restricted_not_portable(tmp_path, budget):
    private = budget.output / "restricted"
    qc.private_dir(private)
    extension = struct.pack("<ii", 32, 6) + b"PRIVATE_IDENTIFIER".ljust(24, b"\0")
    path, proof = staged(private, nifti_bytes(extension=extension, text=b"PRIVATE_PATIENT_NAME"))
    result = qc.inspect_image(path, proof, "ct", private, budget)
    portable = qc.encode(result)
    assert b"PRIVATE_PATIENT_NAME" not in portable
    assert b"PRIVATE_IDENTIFIER" not in portable
    assert b"header_base64" not in portable
    assert b"opaque_review_pending" in portable
    assert result["deidentification"] == "review_pending"
    assert b"PRIVATE_PATIENT_NAME" in (private / "ct-nifti-prefix.bin").read_bytes()
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in private.iterdir())
    assert private.stat().st_mode & 0o777 == 0o700
    assert result["restricted_views"][0]["extent_slice_count"] == 5


@pytest.mark.parametrize("extension", [struct.pack("<ii", 8, 6) + bytes(8), struct.pack("<ii", 48, 6) + bytes(8)])
def test_extension_framing_rejected(tmp_path, budget, extension):
    path, proof = staged(tmp_path, nifti_bytes(extension=extension))
    with pytest.raises(qc.Refusal, match="extension_framing"):
        parsed(path, proof, budget.bounds)


def test_gzip_optional_privacy_fields_and_header_crc(tmp_path):
    raw = gzip.compress(nifti_bytes(), mtime=0)
    filename = b"PRIVATE_NAME\0"
    custom = raw[:3] + bytes([8]) + raw[4:10] + filename + raw[10:]
    path, _ = staged(tmp_path, b"", compressed=custom)
    result, header = qc.gzip_metadata(path)
    assert result["metadata_review_pending"] is True
    assert b"PRIVATE_NAME" not in qc.encode(result)
    assert b"PRIVATE_NAME" in header
    bad = tmp_path / "bad.gz"
    bad.write_bytes(raw[:3] + bytes([8]) + raw[4:10] + b"x" * 65536)
    with pytest.raises(qc.Refusal, match="gzip_header_cap"):
        qc.gzip_metadata(bad)


def test_frozen_preview_reproducibility_and_window_clipping(tmp_path, bounds):
    outputs = []
    for attempt in range(2):
        out = tmp_path / str(attempt)
        qc.private_dir(out)
        private = out / "restricted"
        qc.private_dir(private)
        budget = qc.Budget(out, bounds, time.monotonic() + 30)
        path, proof = staged(private, nifti_bytes())
        result = qc.inspect_image(path, proof, "ct", private, budget)
        outputs.append(result)
    for first, second in zip(outputs[0]["restricted_views"], outputs[1]["restricted_views"], strict=True):
        assert first == second
        assert first["privacy_release"] == "restricted_only"
        assert first["clipping_fractions"] == [0, 0]
    assert [x["window"] for x in outputs[0]["restricted_views"]] == qc.POLICY["ct_windows"]


@pytest.mark.parametrize("kind", ["deadline", "rss", "disk", "output", "view", "receipt"])
def test_resource_caps(tmp_path, budget, monkeypatch, kind):
    if kind == "deadline":
        budget.deadline = time.monotonic() - 1
    elif kind == "rss":
        monkeypatch.setattr(qc, "rss_bytes", lambda: 2**40)
    elif kind == "disk":
        monkeypatch.setattr(qc.shutil, "disk_usage", lambda p: type("Disk", (), {"free": 0})())
    elif kind == "output":
        budget.bounds["first_pair_new_disk_cap_bytes"] = qc.TERMINAL_RESERVE
    elif kind == "view":
        budget.bounds["max_qc_view_bytes_per_pair"] = 0
    else:
        budget.bounds["max_receipt_bytes_per_pair"] = 65536
    with pytest.raises((TimeoutError, qc.BoundExceeded)):
        budget.reserve(1, kind if kind in ("view", "receipt") else "data")


def test_durable_no_overwrite_and_private_files(tmp_path):
    path = tmp_path / "intent.json"
    qc.durable_new(path, b"one")
    with pytest.raises(FileExistsError):
        qc.durable_new(path, b"two")
    assert path.read_bytes() == b"one"
    assert path.stat().st_mode & 0o777 == 0o600


def test_supervisor_timeout_and_no_raw_logs(tmp_path, bounds):
    command = [sys.executable, "-c", "import time; print('PRIVATE_SECRET', flush=True); time.sleep(20)"]
    started = time.monotonic()
    result = qc.supervise(command, tmp_path, bounds, started + .3)
    assert result["supervision"] == "timeout"
    assert result["parent_reaped_worker"] is True
    assert result["worker_log_bytes"] == 0
    assert time.monotonic() - started < 3
    assert not list(tmp_path.iterdir())


def test_supervisor_rss_cap_and_worker_failure(tmp_path, bounds):
    limits = {**bounds, "worker_rss_stop_bytes": 1}
    result = qc.supervise([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, limits, time.monotonic() + 5)
    assert result["supervision"] == "bound_exceeded"
    assert result["parent_reaped_worker"] is True
    result = qc.supervise([sys.executable, "-c", "raise RuntimeError('PRIVATE_SECRET')"], tmp_path, bounds, time.monotonic() + 5)
    assert result["worker_exit_status"] != 0
    assert b"PRIVATE_SECRET" not in qc.encode(result)


def test_supervisor_interruption_reaps_worker(tmp_path, bounds, monkeypatch):
    real_sleep = time.sleep
    interrupted = False
    def interrupt(seconds):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("PRIVATE_SECRET")
        real_sleep(seconds)
    monkeypatch.setattr(qc.time, "sleep", interrupt)
    result = qc.supervise([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, bounds, time.monotonic() + 5)
    monkeypatch.setattr(qc.time, "sleep", real_sleep)
    assert result["supervision"] == "interrupted"
    assert result["parent_reaped_worker"] is True


def test_execute_one_shot_failure_keeps_denominator_and_no_admission(tmp_path, bounds, monkeypatch):
    output = tmp_path / "qc" / "attempt"
    monkeypatch.setattr(qc, "OUTPUT", output)
    monkeypatch.setattr(qc, "validate_declaration", lambda: ({"bounds": bounds, "implementation": {}}, "f" * 64))
    order = [qc.SUBJECT] + [f"offline-{i}" for i in range(125)]
    monkeypatch.setattr(qc, "preflight", lambda: ({"identities": {"full_train_order": order}}, {}, []))
    def fail(*args, **kwargs):
        assert (output / "intent.json").exists()
        qc.durable_receipt(output / "ct-extraction.json", {"completed": True, "source_sha256": "a" * 64}, bounds)
        raise RuntimeError("PRIVATE_SECRET")
    monkeypatch.setattr(qc, "supervise", fail)
    assert qc.execute() == 1
    result = json.loads((output / "parent-result.json").read_text())
    assert result["status"] == "qc_failed"
    assert result["parent_completed"] is True
    assert len(result["denominator"]) == 126
    assert sum(r["status"] == "not_attempted" for r in result["denominator"]) == 125
    assert all(result[k] == v for k, v in qc.CLAIMS.items())
    assert "PRIVATE_SECRET" not in json.dumps(result)
    assert result["extraction_receipts"] == [{"path": "ct-extraction.json", "sha256": qc.sha((output / "ct-extraction.json").read_bytes())}]
    with pytest.raises(FileExistsError):
        qc.execute()


def test_missing_child_receipt_never_passes(tmp_path, bounds, monkeypatch):
    monkeypatch.setattr(qc, "OUTPUT", tmp_path / "qc" / "attempt")
    monkeypatch.setattr(qc, "validate_declaration", lambda: ({"bounds": bounds, "implementation": {}}, "f" * 64))
    monkeypatch.setattr(qc, "preflight", lambda: ({"identities": {"full_train_order": [qc.SUBJECT]}}, {}, []))
    monkeypatch.setattr(qc, "supervise", lambda *a: {"supervision": "completed", "worker_exit_status": 0})
    assert qc.execute() == 1
    result = json.loads((qc.OUTPUT / "parent-result.json").read_text())
    assert result["status"] == "incomplete"


def test_root_declaration_required_before_any_archive_access(tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "DECLARATION", tmp_path / "missing-declaration")
    monkeypatch.setattr(qc, "ArchiveReader", lambda *a: pytest.fail("archive opened"))
    with pytest.raises(FileNotFoundError):
        qc.execute()


def test_exception_sanitization_and_no_admission():
    for exc in (ValueError("PRIVATE_NAME"), OSError("PRIVATE_NAME"), KeyboardInterrupt("PRIVATE_NAME")):
        assert "PRIVATE_NAME" not in str(qc.error_record(exc))
    assert not any(qc.CLAIMS[name] for name in ("training_admitted", "spatial_planning_admitted", "scanner_frame_admitted", "portable_views_released"))


@pytest.mark.parametrize("failure", [None, "scalar", "timeout", "interrupted", "mr_extraction"])
def test_worker_synthetic_end_to_end_preserves_receipts(tmp_path, bounds, monkeypatch, failure):
    path, rows, central, pair = archive_fixture(tmp_path)
    if failure == "mr_extraction":
        raw = bytearray(path.read_bytes())
        row = rows[1]
        nlen, xlen = struct.unpack_from("<HH", raw, row["header_offset"] + 26)
        raw[row["header_offset"] + 30 + nlen + xlen + row["compressed_bytes"] // 2] ^= 1
        path.write_bytes(raw)
    output = tmp_path / "output"
    qc.private_dir(output)
    monkeypatch.setattr(qc, "OUTPUT", output)
    monkeypatch.setattr(qc, "ROOT", tmp_path)
    declaration = {"bounds": bounds, "implementation": {"synthetic_control": True}}
    monkeypatch.setattr(qc, "validate_declaration", lambda: (declaration, "f" * 64))
    proposal = {"source": {"archive": "synthetic.zip", "measured_archive_sha256_from_prior_completion": "0" * 64,
                           "doi": "synthetic-control", "version": "offline"}, "first_pair": pair}
    directory = {"archive_stat_after": qc.stat_record(path.stat()), "central_directory_start": central}
    monkeypatch.setattr(qc, "preflight", lambda: (proposal, directory, rows))
    intent = {"parent_pid": os.getppid(), "declaration_sha256": "f" * 64, "deadline_monotonic": time.monotonic() + 30}
    raw = qc.encode(intent)
    qc.durable_new(output / "intent.json", raw)
    if failure and failure != "mr_extraction":
        def fail(*args):
            if failure == "scalar":
                raise qc.Refusal("nonfinite_effective_values")
            if failure == "timeout":
                raise TimeoutError("PRIVATE_SECRET")
            raise KeyboardInterrupt("PRIVATE_SECRET")
        monkeypatch.setattr(qc, "statistics", fail)
    status = qc.worker(qc.sha(raw))
    result = json.loads((output / "worker-result.json").read_text())
    expected = {None: "review_pending", "scalar": "qc_failed", "timeout": "timeout", "interrupted": "interrupted", "mr_extraction": "qc_failed"}[failure]
    assert result["status"] == expected
    assert status == (0 if failure is None else 1)
    if failure == "mr_extraction":
        assert result["failure_stage"] == "mr_extraction"
        assert result["image_stage_evidence"]["ct"]["extraction"]["status"] == "completed"
        assert result["image_stage_evidence"]["mr"]["extraction"]["status"] == "started"
        assert result["extraction_receipts"] == [{"path": "ct-extraction.json", "sha256": qc.sha((output / "ct-extraction.json").read_bytes())}]
        assert not (output / "mr-extraction.json").exists()
        return
    assert len(result["extraction"]) == 2
    assert result["intent_sha256"] == qc.sha(raw)
    assert all(result[k] == v for k, v in qc.CLAIMS.items())
    assert all(value == "review_pending" for value in result["dispositions"].values())
    assert (output / "ct-header.json").exists()
    assert (output / "ct-grid.json").exists()
    if failure:
        assert result["failure_stage"] == "ct_scalar"
        assert result["image_stage_evidence"]["ct"]["header"]["status"] == "completed"
        assert result["image_stage_evidence"]["ct"]["geometry"]["status"] == "completed"
        assert result["image_stage_evidence"]["ct"]["scalar"]["status"] == "started"
        assert (output / "ct-stage-scalar-started.json").exists()
        assert not (output / "ct-stage-scalar-completed.json").exists()
        for stage in ("header", "geometry"):
            evidence = result["image_stage_evidence"]["ct"][stage]["evidence"]
            assert qc.sha((output / evidence["path"]).read_bytes()) == evidence["sha256"]
    assert "PRIVATE_SECRET" not in json.dumps(result)
    with pytest.raises(FileExistsError):
        qc.worker(qc.sha(raw))
    if failure is None:
        assert result["grid"]["status"] == "source_coded_grid_correspondence"
        assert result["stages"]["anatomy"] == "review_pending"
        assert result["images"]["mr"]["restricted_views"][0]["window"][0] < result["images"]["mr"]["restricted_views"][0]["window"][1]


def test_integer_precision_fail_closed(tmp_path, budget):
    values = np.arange(60, dtype=np.uint64).reshape(3, 4, 5) + 2**53
    path, proof = staged(tmp_path, nifti_bytes(values))
    info = parsed(path, proof, budget.bounds)
    with pytest.raises(qc.Refusal, match="integer_precision"):
        qc.statistics(path, info, budget)


def test_malformed_extension_flag_and_truncated_scalar(tmp_path, budget):
    raw = bytearray(nifti_bytes())
    raw[349] = 1
    path, proof = staged(tmp_path, bytes(raw))
    with pytest.raises(qc.Refusal, match="extension_flag"):
        parsed(path, proof, budget.bounds)
    path, proof = staged(tmp_path, nifti_bytes()[:-2], name="mr.nii.gz")
    info = parsed(path, proof, budget.bounds)
    with pytest.raises(qc.Refusal, match="scalar_truncated"):
        qc.statistics(path, info, budget)


def test_gzip_optional_comment_extra_and_crc(tmp_path):
    raw = gzip.compress(nifti_bytes(), mtime=0)
    prefix = raw[:3] + bytes([4 | 16 | 2]) + raw[4:10] + struct.pack("<H", 4) + b"TEXT" + b"COMMENT\0"
    good = prefix + struct.pack("<H", zlib.crc32(prefix) & 0xffff) + raw[10:]
    path, _ = staged(tmp_path, b"", compressed=good)
    result, _ = qc.gzip_metadata(path)
    assert result["fields"]["comment"]["review_pending"]
    assert result["fields"]["extra"]["review_pending"]
    bad = tmp_path / "bad.gz"
    bad.write_bytes(prefix + b"\0\0" + raw[10:])
    with pytest.raises(qc.Refusal, match="gzip_header_crc"):
        qc.gzip_metadata(bad)


def test_active_qform_only_and_big_endian(tmp_path, budget):
    def modify(header):
        header.set_qform(np.diag([1, 2, 3, 1]), code=1)
        header["sform_code"] = 0
    path, proof = staged(tmp_path, nifti_bytes(modify=modify))
    info = parsed(path, proof, budget.bounds)
    grid = qc.geometry(info["header"], info["shape"])
    assert grid["qform_status"] == "active"
    assert grid["spacing_mm"] == [1, 2, 3]
    values = np.arange(60, dtype=">i2").reshape(3, 4, 5)
    header = nib.Nifti1Header(binaryblock=nifti_bytes()[:348], check=False).as_byteswapped(">")
    path, proof = staged(tmp_path, header.binaryblock + bytes(4) + values.tobytes(order="F"), name="mr.nii.gz")
    info = parsed(path, proof, budget.bounds)
    assert qc.statistics(path, info, budget)["maximum"] == 59


@pytest.mark.parametrize("mutation", ["understated", "compressed_tail", "second_stream", "no_end_marker", "bad_crc"])
def test_strict_outer_deflate_accounting(tmp_path, budget, mutation):
    raw = b"source-gzip-test-control" * 8
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    claimed_size, claimed_crc = len(raw), zlib.crc32(raw)
    if mutation == "understated":
        claimed_size = 4
        claimed_crc = zlib.crc32(raw[:4])
    elif mutation == "compressed_tail":
        compressed += b"hidden"
    elif mutation == "second_stream":
        compressed += compressed
    elif mutation == "no_end_marker":
        compressed = compressed[:-1]
    else:
        claimed_crc ^= 1
    path = tmp_path / "raw.deflate"
    path.write_bytes(compressed)
    row = {"name": qc.NAMES[0], "compressed_bytes": len(compressed), "zip_entry_bytes": claimed_size,
           "crc32": f"{claimed_crc:08x}"}
    reader = qc.ArchiveReader(path, qc.stat_record(path.stat()), 0, budget.check)
    try:
        with pytest.raises(qc.Refusal):
            b"".join(qc.strict_zip_member(reader, row, 0, budget))
    finally:
        reader.close()


def test_understated_zip_consistent_local_and_central(tmp_path, budget):
    def mutate(path, rows, central, pair):
        raw = bytearray(path.read_bytes())
        first = rows[0]
        claimed_size = 4
        original = gzip.compress(nifti_bytes(), mtime=0)
        crc = zlib.crc32(original[:claimed_size])
        struct.pack_into("<I", raw, first["header_offset"] + 14, crc)
        struct.pack_into("<I", raw, first["header_offset"] + 22, claimed_size)
        struct.pack_into("<I", raw, central + 16, crc)
        struct.pack_into("<I", raw, central + 24, claimed_size)
        first.update(zip_entry_bytes=claimed_size, crc32=f"{crc:08x}")
        path.write_bytes(raw)
    with pytest.raises(qc.Refusal, match="zip_expansion"):
        extract_fixture(tmp_path, budget, mutate=mutate)


def test_shared_receipt_cap_counts_initial_stages_and_terminals(tmp_path, bounds):
    limits = {**bounds, "max_receipt_bytes_per_pair": 300}
    first = qc.durable_receipt(tmp_path / "intent.json", {"initial": "x" * 50}, limits)
    second = qc.durable_receipt(tmp_path / "denominator-initial.json", {"rows": "y" * 50}, limits)
    third = qc.durable_receipt(tmp_path / "ct-header.json", {"header": "z" * 50}, limits)
    assert qc.receipt_bytes(tmp_path) == len(first) + len(second) + len(third)
    with pytest.raises(qc.BoundExceeded, match="aggregate_receipt_cap"):
        qc.durable_receipt(tmp_path / "worker-result.json", {"result": "w" * 100}, limits)
    fourth = qc.durable_receipt(tmp_path / "worker-result.json", {"failed": True}, limits, reserve=24)
    with pytest.raises(qc.BoundExceeded, match="aggregate_receipt_cap"):
        qc.durable_receipt(tmp_path / "parent-result.json", {"parent": "p" * 100}, limits)
    assert qc.receipt_bytes(tmp_path) == sum(map(len, (first, second, third, fourth))) <= 300


def test_stage_budget_reserves_worker_parent_and_counts_initial(budget):
    available = budget.bounds["max_receipt_bytes_per_pair"] - qc.WORKER_RECEIPT_RESERVE - qc.PARENT_RECEIPT_RESERVE
    initial = b" " * available
    qc.durable_new(budget.output / "intent.json", initial)
    with pytest.raises(qc.BoundExceeded, match="receipt_cap"):
        budget.save(budget.output / "ct-header.json", b"{}")
    qc.durable_receipt(budget.output / "worker-result.json", {"status": "bound_exceeded"}, budget.bounds, qc.PARENT_RECEIPT_RESERVE)
    qc.durable_receipt(budget.output / "parent-result.json", {"status": "bound_exceeded"}, budget.bounds)
    assert qc.receipt_bytes(budget.output) <= budget.bounds["max_receipt_bytes_per_pair"]
