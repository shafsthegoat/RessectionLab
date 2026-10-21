"""One separately released, bounded asset-only ANTsPy setup; no image access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/"artifacts/poc-ixi265-atropos-preparation-v1"
ASSETS = ROOT/".tools/poc-antspyx-0.6.3-py312-v1"
EXPECTED_BYTES = 95_770_415
DOWNLOAD_CEILING = 192*1024**2
RUNTIME_CEILING = 1536*1024**2
ASSET_CEILING = 2*1024**3
RSS_CEILING = 768*1024**2
AVAILABLE_FLOOR = 1024**3
DISK_FLOOR = 50*1024**3
WALL_SECONDS = 300.


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def download_one(row, directory):
    if (not row["url"].startswith("https://files.pythonhosted.org/packages/")
            or Path(row["filename"]).name != row["filename"] or not row["filename"].endswith(".whl")):
        raise ValueError("Only frozen official PyPI wheel URLs are admitted")
    target = Path(directory)/row["filename"]
    temporary = target.with_suffix(".whl.partial")
    if target.exists() or temporary.exists(): raise ValueError("No automatic asset reuse or retry")
    count = 0; digest = hashlib.sha256()
    with urllib.request.urlopen(row["url"], timeout=30) as response, temporary.open("xb") as stream:
        while chunk := response.read(min(1024**2, row["bytes"]-count+1)):
            count += len(chunk)
            if count > row["bytes"]: raise ValueError("Wheel exceeds frozen expected size")
            stream.write(chunk); digest.update(chunk)
    if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
        raise ValueError("Wheel byte size or SHA256 mismatch")
    temporary.rename(target)
    return count


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if spec["asset_setup_only_released"] is not True: raise ValueError("Asset-only release required")
    assets = Path(spec["assets"]); runtime = assets/"runtime"; wheels = assets/"wheels"
    for row in spec["frozen_files"]:
        if sha(ROOT/row["path"]) != row["sha256"]: raise ValueError("Frozen preparation changed")
    plan = json.loads((PREPARATION/"runtime-plan.json").read_text())
    rows = plan["packages"]
    if len(rows) != 31 or sum(row["bytes"] for row in rows) != EXPECTED_BYTES:
        raise ValueError("Only the frozen 31-wheel inventory is released")
    transferred = 0
    for row in rows:
        transferred += download_one(row, wheels)
        if transferred > DOWNLOAD_CEILING: raise ValueError("Total wheel download ceiling")
        write(assets/"download-progress.json", {"bytes": transferred, "last_completed": row["filename"]})
    environment = dict(os.environ)
    subprocess.run([spec["base_python"], "-I", "-m", "venv", str(runtime)], check=True, env=environment)
    python = str(runtime/"bin/python")
    subprocess.run([python, "-I", "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
        "--only-binary=:all:", "--require-hashes", "--no-deps", "--no-cache-dir", "-r", str(PREPARATION/"requirements.lock")],
        check=True, env=environment)
    subprocess.run([python, "-I", "-m", "pip", "check"], check=True, env=environment)
    for row in rows:
        if sha(wheels/row["filename"]) != row["sha256"]: raise ValueError("Retained wheel changed after install")
    for row in spec["frozen_files"]:
        if sha(ROOT/row["path"]) != row["sha256"]: raise ValueError("Preparation changed during setup")
    # Inspect installed metadata/API/binary hashes; no ANTs package import or fitting.
    check_code = "import importlib.util,json; from pathlib import Path; p=Path("+repr(str(ROOT/"scripts/poc_ixi265_atropos.py"))+ "); s=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.runtime_audit("+repr(str(runtime))+")))"
    checked = subprocess.run([python, "-I", "-c", check_code], check=True, capture_output=True, text=True, env=environment)
    write(assets/"installed-runtime-audit.json", json.loads(checked.stdout))
    write(assets/"worker-success.json", {"downloaded_bytes": transferred, "runtime_bytes": size(runtime),
        "retained_wheel_bytes": size(wheels), "package_dependency_check_passed": True,
        "retained_wheel_hashes_verified_after_install": True, "ANTs_imported": False,
        "source_image_or_mask_decoded": False, "fitting_run": False})


def setup(*, released=False):
    if not released: raise ValueError("Asset setup requires a separate explicit release")
    import psutil
    if sys.version_info[:2] != (3, 12): raise ValueError("Run setup supervisor with the existing Python3.12 runtime")
    if ASSETS.exists(): raise ValueError("One setup attempt only; preserve previous success/failure")
    if psutil.virtual_memory().available < AVAILABLE_FLOOR or psutil.disk_usage(ASSETS.parent).free < DISK_FLOOR:
        raise ValueError("Setup preflight available-memory/free-disk floor")
    frozen = json.loads((PREPARATION/"preparation.json").read_text())["files"]
    frozen += [{"path": str((PREPARATION/"preparation.json").relative_to(ROOT)), "sha256": sha(PREPARATION/"preparation.json")},
               {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(__file__)}]
    for row in frozen:
        if sha(ROOT/row["path"]) != row["sha256"]: raise ValueError("Frozen file changed before setup")
    ASSETS.mkdir(); (ASSETS/"wheels").mkdir(); (ASSETS/"temporary").mkdir()
    spec = {"asset_setup_only_released": True, "assets": str(ASSETS.resolve()),
            "base_python": sys._base_executable, "frozen_files": frozen}
    write(ASSETS/"setup-input.json", spec)
    record = {"schema": "resectionlab-ixi265-atropos-asset-setup-v1", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "frozen_files": frozen,
        "limits": {"download_bytes": DOWNLOAD_CEILING, "runtime_bytes": RUNTIME_CEILING,
            "all_asset_bytes": ASSET_CEILING, "rss_bytes": RSS_CEILING, "available_memory_bytes": AVAILABLE_FLOOR,
            "seconds": WALL_SECONDS, "free_disk_bytes": DISK_FLOOR, "sampling_seconds": .2},
        "source_image_or_mask_decoded": False, "ANTs_fitting_run": False, "fit_attempt_marker_created": False}
    write(ASSETS/"setup.json", record)
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_CACHE_DIR": "1", "TMPDIR": str((ASSETS/"temporary").resolve()),
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
    environment.pop("PYTHONPATH", None)
    process = None; started = time.monotonic(); peak = 0
    try:
        with (ASSETS/"setup.log").open("w") as log:
            process = subprocess.Popen([sys._base_executable, "-I", str(Path(__file__).resolve()),
                "--worker", str((ASSETS/"setup-input.json").resolve())], stdout=log, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True)
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    peak = max(peak, psutil.Process(os.getpid()).memory_info().rss
                        + sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running()))
                except psutil.NoSuchProcess: pass
                if peak > RSS_CEILING: raise RuntimeError("setup_RSS_ceiling")
                if time.monotonic()-started > WALL_SECONDS: raise RuntimeError("setup_wall_clock_limit")
                if psutil.virtual_memory().available < AVAILABLE_FLOOR: raise RuntimeError("setup_available_memory_floor")
                if psutil.disk_usage(ASSETS).free < DISK_FLOOR: raise RuntimeError("setup_free_disk_floor")
                if size(ASSETS) > ASSET_CEILING or size(ASSETS/"runtime") > RUNTIME_CEILING:
                    raise RuntimeError("setup_asset_disk_ceiling")
                time.sleep(.2)
        if process.returncode or not (ASSETS/"worker-success.json").is_file(): raise RuntimeError("setup_worker_failed_see_retained_log")
        record.update(json.loads((ASSETS/"worker-success.json").read_text()), status="succeeded_assets_only")
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
        record.update(status="failed", failure=str(error))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, peak_process_family_rss_bytes_sampled=peak,
            total_asset_bytes=size(ASSETS), exit_code=None if process is None else process.returncode,
            memory_caveat="Supervisor plus worker-process-family RSS sampled at 0.2 seconds; allocator/between-sample peaks may differ")
        write(ASSETS/"setup.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--released-asset-setup", action="store_true")
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    else: print(json.dumps(setup(released=args.released_asset_setup), indent=2))
