"""Notice inventory integrity tests; no runtime numerical imports or app builds."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "python_notices", Path(__file__).resolve().parents[1] / "packaging/python_notices.py"
)
notices = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notices)


def test_license_tree_retains_vendor_text_but_excludes_executable_packages():
    assert notices.notice_path(Path("h5py.dist-info/licenses/licenses/hdf5.txt"))
    assert notices.notice_path(Path("scipy/spatial/qhull_src/COPYING_QHULL.txt"))
    assert not notices.notice_path(Path("packaging/licenses/__init__.py"))
    assert not notices.notice_path(Path("packaging/licenses/__pycache__/__init__.pyc"))


def test_frozen_code_comparison_rejects_changed_nested_semantics():
    first = compile("def calculate():\n    return 2\n", "installed.py", "exec", dont_inherit=True)
    relocated = compile("def calculate():\n    return 2\n", "frozen.py", "exec", dont_inherit=True)
    changed = compile("def calculate():\n    return 3\n", "installed.py", "exec", dont_inherit=True)
    assert notices.code_matches(first, relocated)
    assert not notices.code_matches(first, changed)


def test_collected_native_file_bytes_are_bound_and_missing_payload_rejected(tmp_path):
    engine = tmp_path / "engine"
    engine.write_bytes(b"executable")
    library = tmp_path / "_internal/lib.dylib"
    library.parent.mkdir()
    library.write_bytes(b"actual collected library")
    row = [("lib.dylib", "/old/source/lib.dylib", "BINARY")]
    inventory = notices.collected_payload(engine, row)
    assert inventory["_internal/lib.dylib"]["sha256"] == notices.sha256(library)
    library.unlink()
    with pytest.raises(ValueError, match="missing"):
        notices.collected_payload(engine, row)


def test_collected_payload_refuses_escape_symlink(tmp_path):
    root = tmp_path / "bundle"
    (root / "_internal").mkdir(parents=True)
    engine = root / "engine"
    engine.write_bytes(b"executable")
    outside = tmp_path / "outside"
    outside.write_bytes(b"library")
    (root / "_internal/lib.dylib").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        notices.collected_payload(engine, [("lib.dylib", "elsewhere", "BINARY")])


def test_copy_preserves_upstream_bytes_and_rejects_traversal(tmp_path):
    source = tmp_path / "LICENSE"
    source.write_bytes(b"Upstream\r\nCopyright\n")
    target = tmp_path / "captured/LICENSE"
    metadata = notices.copy_notice(source, target, origin="fixture/LICENSE")
    assert source.read_bytes() == target.read_bytes()
    assert metadata["sha256"] == notices.sha256(source)
    with pytest.raises(ValueError, match="Unsafe"):
        notices.safe_relative(Path("../LICENSE"))
