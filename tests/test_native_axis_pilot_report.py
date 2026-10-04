"""Reporting guards only: no model, environment, optimizer or actual-run mutation."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_native_axis_pilot import Inputs, extract
from pack_native_axis_pilot import pack, TARGETS


def test_growing_attempt_is_rejected_before_any_other_file_read(tmp_path):
    (tmp_path / "launcher-status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="growing evidence"):
        extract(tmp_path, tmp_path / "baseline-must-not-be-read")


def test_reader_accepts_gzip_only_and_refuses_conflicting_raw(tmp_path):
    raw = b'{"exact": [1, 2, 3]}\n'
    (tmp_path / "evidence.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    reader = Inputs(tmp_path)
    assert reader.read("evidence.json") == {"exact": [1, 2, 3]}
    assert reader.hashes["evidence.json"]["uncompressed_bytes"] == len(raw)
    (tmp_path / "evidence.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="disagreement"):
        reader.read("evidence.json")


def test_pack_preserves_exact_raw_and_refuses_existing_different_gzip(tmp_path):
    (tmp_path / "launcher-status.json").write_text('{"status":"completed"}')
    for name in TARGETS:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'{"retained": true}\n')
    result = pack(tmp_path)
    assert len(result["files"]) == 3
    for row in result["files"]:
        assert row["byte_roundtrip_equal"] and row["json_roundtrip_equal"]
        assert (tmp_path / row["raw_path"]).read_bytes() == b'{"retained": true}\n'
    (tmp_path / (TARGETS[0] + ".gz")).write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(FileExistsError):
        pack(tmp_path)
