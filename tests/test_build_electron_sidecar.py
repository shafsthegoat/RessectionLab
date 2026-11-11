"""Snapshot and final-artifact inventory checks without compiling or freezing."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_electron_sidecar.py"
    spec = importlib.util.spec_from_file_location("sidecar_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_includes_native_sources_headers_configuration_and_surface_policy(builder, tmp_path, monkeypatch):
    names = ["src/resectionlab/geometry.py", "src/resectionlab/geometry_native.cpp",
             "src/resectionlab/native/kernel.hpp", "src/resectionlab/surface_policy.json",
             "src/resectionlab/native-notices/pybind11-LICENSE", "setup.py", "setup.cfg",
             "MANIFEST.in", "pyproject.toml", "requirements-lock.txt", "packaging/sidecar_entry.py",
             "packaging/python_notices.py", "scripts/build_electron_sidecar.py"]
    stale = ["src/resectionlab/_geometry_cpp.cpython-312-darwin.so",
             "src/resectionlab/_geometry_cpp.build.json", "src/resectionlab/__pycache__/cached.py"]
    for name in names + stale:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("captured")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "__file__", str(tmp_path / "scripts/build_electron_sidecar.py"))
    assert {str(path.relative_to(tmp_path)) for path in builder.inputs()} == set(names)


def test_frozen_manifest_binds_relocated_binary_without_changing_build_identity(builder, tmp_path):
    package = tmp_path / "_internal/resectionlab"
    package.mkdir(parents=True)
    binary = package / "_geometry_cpp.test.so"
    binary.write_bytes(b"final relocated signed payload")
    identity = {"sourceSha256": "source", "compiler": "compiler"}
    manifest = package / "_geometry_cpp.build.json"
    manifest.write_text(json.dumps({"buildIdentity": identity, "binaryName": binary.name,
                                    "binarySha256": "wheel digest"}))
    result = builder.bind_frozen_binary(tmp_path / "ressectionlab-engine")
    assert result["wheelBinarySha256"] == "wheel digest"
    assert result["binarySha256"] == builder.digest(binary)
    assert json.loads(manifest.read_text())["buildIdentity"] == identity


def test_snapshot_captures_raw_replay_dependency_source_not_extension_binary(builder, tmp_path, monkeypatch):
    policy = tmp_path / "snapshot/src/resectionlab/surgical_proof_cache.py"
    policy.parent.mkdir(parents=True)
    policy.write_text("_BINARIES = ('dependency.wrapper', 'dependency.compiled')\n")
    wrapper = tmp_path / "installed/dependency/wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# actual dependency source\n")
    compiled = wrapper.with_name("compiled.so")
    compiled.write_bytes(b"extension")
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name:
                        SimpleNamespace(origin=str(wrapper if name.endswith("wrapper") else compiled)))
    records = builder.capture_dependency_sources(tmp_path / "snapshot")
    assert len(records) == 1
    item = records[0]
    assert item["module"] == "dependency.wrapper"
    assert item["destination"] == "dependency/wrapper.py"
    assert (tmp_path / "snapshot" / item["captured"]).read_bytes() == wrapper.read_bytes()
    assert item["sha256"] == builder.digest(wrapper)
