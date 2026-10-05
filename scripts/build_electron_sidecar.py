#!/usr/bin/env python3
"""Freeze the supported Electron numerical API from an immutable source copy.

The engine supports imaging, search, workspace persistence and native patient
refinement. Runtime integration checks must validate the frozen training path.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "electron-sidecar"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs() -> list[Path]:
    return sorted([*list((ROOT / "src" / "resectionlab").rglob("*.py")), ROOT / "packaging" / "sidecar_entry.py", ROOT / "packaging" / "python_notices.py", Path(__file__).resolve(), ROOT / "pyproject.toml"] + list(ROOT.glob("*lock*")))


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
        manifest = {"source_files": captured, "source_digest": hashlib.sha256(json.dumps(captured, sort_keys=True).encode()).hexdigest(),
                    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "imaging_search_native_refinement", "clinical_use": "research_only"}
        manifest_path = snapshot / "BUILD_INPUT_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_path.chmod(0o444)
        destination = ROOT / "desktop" / "sidecar"
        env = dict(os.environ, PYTHONPATH=str(snapshot / "src"))
        command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console", "--name", "ressectionlab-engine", "--distpath", str(destination), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD), "--paths", str(snapshot / "src"), "--target-architecture", "arm64", "--add-data", f"{manifest_path}:build_info", "--add-data", f"{snapshot / 'src' / 'resectionlab'}:resectionlab"]
        for module in ["PySide6", "PyQt6", "vtk", "vtkmodules", "dipy", "fury", "matplotlib", "pandas", "IPython", "pytest", "tkinter"]:
            command += ["--exclude-module", module]
        command += [str(snapshot / "packaging" / "sidecar_entry.py")]
        with (BUILD / "build.log").open("w") as output:
            subprocess.run(command, cwd=snapshot, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        if captured != {relative: digest(snapshot / relative) for relative in captured}:
            raise RuntimeError("Captured sources changed during build")
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
                        executable=str(destination / "ressectionlab-engine" / "ressectionlab-engine"), snapshot=str(snapshot), notice_inventory=str(notice_target / "inventory.json"))
        report.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"status": manifest["status"], "source_digest": manifest["source_digest"], "elapsed_seconds": manifest["elapsed_seconds"]}))
        return 0
    except Exception as error:
        report.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
