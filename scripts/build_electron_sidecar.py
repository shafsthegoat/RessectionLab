#!/usr/bin/env python3
"""Freeze the supported Electron numerical API from an immutable source copy.

The engine supports imaging, search, workspace persistence and native patient
refinement. Runtime integration checks must validate the frozen training path.
"""
from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "electron-sidecar"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs() -> list[Path]:
    # Capture source/data, never a stale in-place extension from a developer build.
    package = [path for path in (ROOT / "src" / "resectionlab").rglob("*")
               if path.is_file() and "__pycache__" not in path.parts
               and (path.suffix in {".py", ".json", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"}
                    or "native-notices" in path.parts)
               and path.name != "_geometry_cpp.build.json"]
    configuration = [ROOT / name for name in ("pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in")
                     if (ROOT / name).is_file()]
    return sorted(set(package + configuration + [ROOT / "packaging" / "sidecar_entry.py",
        ROOT / "packaging" / "python_notices.py", Path(__file__).resolve()]
        + list(ROOT.glob("*lock*"))))


def build_package(snapshot: Path, output: Path, log) -> tuple[Path, Path]:
    """Compile captured inputs once; freeze only the resulting wheel installation."""
    wheels, installed = output / "wheels", output / "installed"
    wheels.mkdir(parents=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--wheel-dir", str(wheels), str(snapshot)], cwd=output, env=environment,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
    wheel, = wheels.glob("resectionlab-*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                    "--target", str(installed), str(wheel)], cwd=output, env=environment,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
    return wheel, installed


def capture_dependency_sources(snapshot: Path) -> list[dict]:
    """Keep raw dependency images read by the existing replay identity checks.

    Most entries are collected extension binaries. Some SciPy versions expose
    a Python wrapper instead; PYZ bytecode alone cannot satisfy a source hash.
    Read the captured policy rather than maintaining another module allowlist.
    """
    policy = snapshot / "src/resectionlab/surgical_proof_cache.py"
    tree = ast.parse(policy.read_text())
    assignment, = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(name, ast.Name) and name.id == "_BINARIES"
                           for name in node.targets)]
    records = []
    for name in ast.literal_eval(assignment.value):
        source = Path(importlib.util.find_spec(name).origin).resolve(strict=True)
        if source.suffix != ".py":
            continue
        relative = Path(*name.split(".")).with_suffix(".py")
        target = snapshot / "dependency-images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o444)
        if digest(target) != digest(source):
            raise RuntimeError(f"Dependency source changed during capture: {name}")
        records.append({"module": name, "origin": str(source),
                        "captured": str(target.relative_to(snapshot)),
                        "destination": relative.as_posix(), "sha256": digest(target)})
    return records


def bind_frozen_binary(engine: Path) -> dict:
    """Bind the relocated/signed payload, retaining its original build identity."""
    package = engine.parent / "_internal" / "resectionlab"
    path = package / "_geometry_cpp.build.json"
    record = json.loads(path.read_text())
    binary = package / record["binaryName"]
    if not binary.is_file() or binary.parent != package:
        raise RuntimeError("Frozen native manifest does not identify its packaged binary")
    wheel_digest = record["binarySha256"]
    record["binarySha256"] = digest(binary)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return {"binaryName": binary.name, "binarySha256": record["binarySha256"],
            "wheelBinarySha256": wheel_digest, "buildIdentity": record["buildIdentity"]}


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    report = BUILD / "build-report.json"
    report.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        for _ in range(3):
            original = {str(p.relative_to(ROOT)): digest(p) for p in inputs()}
            snapshot = BUILD / "inputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            for relative in original:
                source, destination = ROOT / relative, snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                destination.chmod(0o444)
            captured = {relative: digest(snapshot / relative) for relative in original}
            if original == captured == {str(p.relative_to(ROOT)): digest(p) for p in inputs()}:
                break
            shutil.rmtree(snapshot)
        else:
            raise RuntimeError("Source changed during capture; retry this short snapshot")
        dependency_sources = capture_dependency_sources(snapshot)
        manifest = {"source_files": captured, "source_digest": hashlib.sha256(json.dumps(captured, sort_keys=True).encode()).hexdigest(),
                    "dependency_sources": dependency_sources,
                    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "imaging_search_native_refinement", "clinical_use": "research_only"}
        destination = ROOT / "desktop" / "sidecar"
        with (BUILD / "build.log").open("w") as output:
            wheel, installed = build_package(snapshot, snapshot / "package-build", output)
            package = installed / "resectionlab"
            binaries = {path.name: digest(path) for path in package.glob("_geometry_cpp*")
                        if path.suffix in {".so", ".pyd", ".dylib"}}
            if not binaries:
                raise RuntimeError("Captured wheel did not contain the C++ extension")
            manifest.update(wheel={"path": str(wheel), "sha256": digest(wheel)},
                            native_binaries=binaries, architecture=platform.machine())
            manifest_path = snapshot / "BUILD_INPUT_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            manifest_path.chmod(0o444)
            data = installed.parent / "frozen-data" / "resectionlab"
            shutil.copytree(package, data,
                ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd", "*.dylib"))
            env = dict(os.environ, PYTHONPATH=str(installed), PYTHONNOUSERSITE="1")
            env.pop("PYTHONHOME", None)
            command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console", "--name", "ressectionlab-engine", "--distpath", str(destination), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD), "--paths", str(installed), "--target-architecture", platform.machine(), "--add-data", f"{manifest_path}:build_info", "--add-data", f"{data}:resectionlab", "--collect-binaries", "resectionlab", "--hidden-import", "resectionlab._geometry_cpp"]
            for item in dependency_sources:
                command += ["--add-data", f"{snapshot / item['captured']}:{Path(item['destination']).parent}"]
            for module in ["PySide6", "PyQt6", "vtk", "vtkmodules", "dipy", "fury", "matplotlib", "pandas", "IPython", "pytest", "tkinter"]:
                command += ["--exclude-module", module]
            command += [str(snapshot / "packaging" / "sidecar_entry.py")]
            subprocess.run(command, cwd=installed, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        executable = destination / "ressectionlab-engine" / "ressectionlab-engine"
        manifest["frozen_native"] = bind_frozen_binary(executable)
        if captured != {relative: digest(snapshot / relative) for relative in captured}:
            raise RuntimeError("Captured sources changed during build")
        for item in dependency_sources:
            if digest(snapshot / item["captured"]) != item["sha256"] or digest(Path(item["origin"])) != item["sha256"]:
                raise RuntimeError(f"Dependency source changed during build: {item['module']}")
        # Capture notices beside the engine, never by editing its executable.
        # The application builder binds this inventory to exact payload hashes.
        notice_capture = BUILD / "notice-inputs" / snapshot.name
        subprocess.run([sys.executable, str(snapshot / "packaging/python_notices.py"),
                        "--engine", str(destination / "ressectionlab-engine/ressectionlab-engine"),
                        "--work", str(BUILD / "work/ressectionlab-engine"),
                        "--snapshot", str(snapshot), "--output", str(notice_capture)], check=True)
        notice_target = destination / "notices"
        if notice_target.exists():
            shutil.move(str(notice_target), str(BUILD / f"previous-notices-{snapshot.name}"))
        shutil.copytree(notice_capture, notice_target)
        manifest.update(status="built_unverified", elapsed_seconds=time.perf_counter()-started,
                        executable=str(executable), snapshot=str(snapshot), notice_inventory=str(notice_target / "inventory.json"))
        report.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"status": manifest["status"], "source_digest": manifest["source_digest"], "elapsed_seconds": manifest["elapsed_seconds"]}))
        return 0
    except Exception as error:
        report.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
