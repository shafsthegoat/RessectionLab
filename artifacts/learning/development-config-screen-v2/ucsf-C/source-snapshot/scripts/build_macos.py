#!/usr/bin/env python3
"""Build and exercise the native Apple Silicon application, entirely locally."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "macos"
APP = ROOT / "dist" / "RessectionLab.app"


def command(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, **kwargs)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    """Hash build inputs, never the downloaded imaging or a user's case exports."""
    files = list((ROOT / "src").rglob("*")) + list((ROOT / "packaging").rglob("*"))
    files += [ROOT / "pyproject.toml", Path(__file__).resolve()]
    files += list(ROOT.glob("*lock*"))
    files += [path for name in ("LICENSE", "THIRD_PARTY_NOTICES.md") if (path := ROOT / name).is_file()]
    digest = hashlib.sha256()
    for path in sorted(set(path for path in files if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def make_icon() -> Path:
    """Render the original vector identity into macOS's standard icon sizes."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    iconset = BUILD / "RessectionLab.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    renderer = QSvgRenderer(str(ROOT / "packaging" / "icon.svg"))
    if not renderer.isValid():
        raise RuntimeError("The application SVG icon is invalid")
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = size * scale
            image = QImage(pixels, pixels, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            suffix = "@2x" if scale == 2 else ""
            if not image.save(str(iconset / f"icon_{size}x{size}{suffix}.png")):
                raise RuntimeError("Could not render application icon")
    icon_path = BUILD / "RessectionLab.icns"
    command(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(icon_path)])
    return icon_path


def clean_launch_environment() -> dict[str, str]:
    env = os.environ.copy()
    # A package check must not accidentally borrow libraries from the development shell.
    for name in tuple(env):
        if name.startswith(("PYTHON", "DYLD_", "QT_")) or name in {"VIRTUAL_ENV", "QML2_IMPORT_PATH"}:
            env.pop(name, None)
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    return env


def verify_local_bundle_links(bundle: Path) -> int:
    """Reject a package that silently depends on links back into the checkout."""
    count = 0
    for path in bundle.rglob("*"):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_relative_to(bundle.resolve()):
                raise RuntimeError(f"Bundle symlink escapes the application: {path} -> {target}")
            count += 1
    return count


def verify_native_dependencies(bundle: Path) -> int:
    """Allow only bundled-relative or macOS system native-library references."""
    binaries = []
    macho_headers = {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"}
    for path in bundle.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                if stream.read(4) in macho_headers:
                    binaries.append(path)
    for offset in range(0, len(binaries), 100):
        output = command(["/usr/bin/otool", "-L", *map(str, binaries[offset:offset + 100])], capture_output=True).stdout
        for line in output.splitlines():
            if not line.startswith("\t"):
                continue
            dependency = line.strip().partition(" (compatibility version")[0]
            if not dependency.startswith(("@rpath/", "@loader_path/", "@executable_path/", "/System/Library/", "/usr/lib/")):
                raise RuntimeError(f"Unbundled native-library dependency: {dependency}")
    return len(binaries)


def verify_bundle(gui_smoke: bool = True) -> dict:
    executable = APP / "Contents" / "MacOS" / "RessectionLab"
    if not executable.is_file():
        raise RuntimeError(f"No application executable found at {executable}")
    with (APP / "Contents" / "Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    architecture = command(["/usr/bin/lipo", "-archs", str(executable)], capture_output=True).stdout.strip()
    if architecture != "arm64":
        raise RuntimeError(f"Expected arm64 executable, found {architecture}")
    signature = command(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(APP)], capture_output=True
    )
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(APP),
        "bundle_version": info["CFBundleShortVersionString"],
        "architecture": architecture,
        "signature_verified": signature.returncode == 0,
        "internal_symlinks_verified": verify_local_bundle_links(APP),
        "native_dependency_files_verified": verify_native_dependencies(APP),
        "signature_type": "ad_hoc_local",
        "notarized": False,
        "executable_sha256": file_hash(executable),
        "application_bytes": sum(path.stat().st_size for path in APP.rglob("*") if path.is_file() and not path.is_symlink()),
        "clean_environment": True,
        "gui_smoke_requested": gui_smoke,
    }
    environment = clean_launch_environment()
    with tempfile.TemporaryDirectory(prefix="ressectionlab-package-check-") as temporary:
        runtime_report = BUILD / "runtime-check.json"
        runtime_report.unlink(missing_ok=True)
        started = time.monotonic()
        command(
            [str(executable), "--package-check", str(runtime_report)],
            cwd=temporary, env=environment, timeout=180, capture_output=True,
        )
        report["runtime_check_seconds"] = round(time.monotonic() - started, 3)
        runtime = json.loads(runtime_report.read_text())
        if not runtime.get("passed") or not runtime.get("frozen"):
            raise RuntimeError(f"Frozen runtime verification failed: {runtime}")
        report["runtime_check"] = runtime
        if gui_smoke:
            smoke_report = BUILD / "gui-smoke.json"
            smoke_report.unlink(missing_ok=True)
            screenshot = BUILD / "gui-smoke.png"
            screenshot.unlink(missing_ok=True)
            started = time.monotonic()
            command(
                [str(executable), "--smoke-test", "--smoke-report", str(smoke_report), "--screenshot", str(screenshot)],
                cwd=temporary, env=environment, timeout=120, capture_output=True,
            )
            report["gui_smoke_seconds"] = round(time.monotonic() - started, 3)
            smoke = json.loads(smoke_report.read_text())
            if not smoke.get("passed"):
                raise RuntimeError(f"Desktop smoke verification failed: {smoke}")
            if not screenshot.is_file() or screenshot.stat().st_size == 0:
                raise RuntimeError("Desktop smoke did not produce the requested screenshot")
            report["gui_smoke"] = smoke
            report["screenshot"] = str(screenshot)
    report["passed"] = True
    (BUILD / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Clear PyInstaller's generated analysis cache")
    parser.add_argument("--verify-only", action="store_true", help="Check an existing app without rebuilding")
    parser.add_argument("--no-gui-smoke", action="store_true", help="Skip the window/render check when no GUI session is available")
    arguments = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("This release target requires native Apple Silicon macOS and an arm64 Python interpreter")
    BUILD.mkdir(parents=True, exist_ok=True)
    previous = [BUILD / name for name in (
        "build.log", "build-manifest.json", "verification.json", "runtime-check.json", "gui-smoke.json", "gui-smoke.png",
    ) if (BUILD / name).is_file()]
    if previous:
        archive = BUILD / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive.mkdir(parents=True)
        for path in previous:
            shutil.copy2(path, archive / path.name)
    # A failed new attempt must never leave a previous green report looking current.
    (BUILD / "verification.json").write_text(json.dumps({
        "schema_version": 1, "passed": False, "status": "in_progress",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    for name in ("runtime-check.json", "gui-smoke.json", "gui-smoke.png"):
        (BUILD / name).unlink(missing_ok=True)
    if not arguments.verify_only:
        import importlib.metadata

        initial_hash = source_hash()
        make_icon()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        build_command = [
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(ROOT / "dist"),
            "--workpath", str(BUILD / "pyinstaller"), str(ROOT / "packaging" / "RessectionLab.spec"),
        ]
        if arguments.clean:
            build_command.insert(3, "--clean")
        started = time.monotonic()
        with (BUILD / "build.log").open("w") as log:
            command(build_command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        final_hash = source_hash()
        manifest = {
            "schema_version": 1,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "build_seconds": round(time.monotonic() - started, 3),
            "source_revision": command(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True).stdout.strip(),
            "source_tree_sha256_before": initial_hash,
            "source_tree_sha256_after": final_hash,
            "source_changed_during_build": initial_hash != final_hash,
            "python": sys.version,
            "platform": platform.platform(),
            "pyinstaller": importlib.metadata.version("pyinstaller"),
            "dependencies": {
                dist.metadata["Name"]: dist.version
                for dist in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower())
            },
            "lock_files": {path.name: file_hash(path) for path in ROOT.glob("*lock*") if path.is_file()},
            "build_command": build_command,
            "distribution_status": "local_ad_hoc_build_not_notarized",
        }
        (BUILD / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = verify_bundle(gui_smoke=not arguments.no_gui_smoke)
    print(json.dumps({"app": str(APP), "passed": report["passed"], "report": str(BUILD / "verification.json")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        BUILD.mkdir(parents=True, exist_ok=True)
        (BUILD / "verification.json").write_text(json.dumps({
            "schema_version": 1, "passed": False, "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": traceback.format_exc(),
        }, indent=2) + "\n")
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        print(f"Packaging failed: {error}. See {BUILD / 'build.log'} and verification.json", file=sys.stderr)
        raise SystemExit(1) from error
