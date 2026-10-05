#!/usr/bin/env python3
"""Record local compute and runtime capability without collecting device identifiers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


DEPENDENCIES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "torch": "torch",
    "PySide6": "PySide6",
    "vtk": "vtk",
    "pyvista": "pyvista",
    "nibabel": "nibabel",
    "SimpleITK": "SimpleITK",
    "dipy": "dipy",
    "pytest": "pytest",
    "pip": "pip",
}


def command(*args: str) -> str | None:
    """Return a small command result, treating absent optional tools as unknown."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=15
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def probe(root: Path) -> dict:
    packages = {}
    for module, distribution in DEPENDENCIES.items():
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[module] = {"available": available, "version": version}

    memory_bytes = command("sysctl", "-n", "hw.memsize")
    disk = shutil.disk_usage(root)
    swift = command("swift", "--version")
    developer_directory = command("xcode-select", "-p")
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "local development machine; no serial numbers or hardware UUIDs",
        "os": {
            "system": platform.system(),
            "macos_version": platform.mac_ver()[0] or None,
            "architecture": platform.machine(),
        },
        "compute": {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": int(memory_bytes) if memory_bytes else None,
            "cuda_assumed": False,
            "mps_execution_tested": False,
        },
        "storage": {
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "scope": "filesystem containing this repository",
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "isolated_environment": sys.prefix != sys.base_prefix,
            "packages": packages,
        },
        "build_tools": {
            "swift": swift,
            "developer_directory": developer_directory,
            "full_xcode_selected": bool(
                developer_directory and ".app/Contents/Developer" in developer_directory
            ),
            "uv_on_path": shutil.which("uv") is not None,
            "node_on_path": shutil.which("node") is not None,
            "gh_on_path": shutil.which("gh") is not None,
            "slicer_in_applications": Path("/Applications/Slicer.app").is_dir(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    report = json.dumps(probe(Path(__file__).resolve().parents[1]), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
