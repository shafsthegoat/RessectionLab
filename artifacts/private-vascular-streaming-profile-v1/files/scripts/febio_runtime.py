#!/usr/bin/env python3
"""Acquire or build one pinned private FEBio runtime; never run a mechanics model."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "artifacts/febio-runtime-investigation-v1/prospective-runtime.json"
DECLARATION_SHA = "1fae86f48714f83f1be4fb94be173bc434721ee78043732c695b547529425c48"
PREFIX = ROOT / "data/optional-runtimes/febio-4.13"
ACQUISITION_SECONDS = 300
POLL_SECONDS = .1
MAX_ENTRIES = 30000


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def declaration():
    if sha(DECLARATION) != DECLARATION_SHA:
        raise ValueError("Committed runtime proposal changed")
    return json.loads(DECLARATION.read_text())


def private_environment(spec):
    # Only the child environment changes. HOME and CODEX_HOME retain their values.
    result = dict(os.environ)
    for key in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "DYLD_LIBRARY_PATH",
                "DYLD_FALLBACK_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "MKLROOT",
                "PYTHONPATH", "PYTHONHOME", "MAKEFLAGS", "MFLAGS"):
        result.pop(key, None)
    for key in list(result):
        if key.startswith("DYLD_"):
            result.pop(key)
    result.update(spec["caps"]["thread_environment"])
    return result


def process_group_rss(pgid, *, timeout_seconds):
    """Numeric process metadata only; sum the entire inherited process group."""
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss="],
                            capture_output=True, text=True, timeout=timeout_seconds,
                            check=True)
    members = []
    for line in result.stdout.splitlines():
        cells = line.split()
        if len(cells) == 3 and all(cell.isdigit() for cell in cells):
            pid, group, rss = map(int, cells)
            if group == pgid:
                members.append({"pid": pid, "rss_bytes": rss * 1024})
    return sum(item["rss_bytes"] for item in members), members


def supervise(command, output, *, cwd, environment, seconds, rss_bytes):
    """Kill/reap on caps or observer failure; always preserve a failure receipt."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "running", "command": command, "cwd": str(cwd),
               "wall_cap_seconds": seconds, "rss_cap_bytes": rss_bytes,
               "sampled_peak_process_group_rss_bytes": 0, "exit_code": None,
               "kill_reason": None, "error": None, "cleanup_error": None,
               "no_retry": True, "sampling_note": "Group RSS is sampled; brief between-sample peaks may be missed."}
    write_json(output / "supervision.json", receipt)
    process = None
    try:
        with (output / "combined.log").open("x") as log:
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            receipt["pid"] = process.pid
            write_json(output / "supervision.json", receipt)
            last_progress = started
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= seconds:
                    receipt["kill_reason"] = "wall_cap"
                    break
                code = process.poll()
                rss, members = process_group_rss(process.pid, timeout_seconds=min(1., seconds-elapsed))
                receipt["sampled_peak_process_group_rss_bytes"] = max(
                    receipt["sampled_peak_process_group_rss_bytes"], rss)
                if rss > rss_bytes:
                    receipt["kill_reason"] = "process_group_rss_cap"
                    break
                if code is not None:
                    receipt["exit_code"] = code
                    if members:
                        receipt["kill_reason"] = "descendants_outlived_parent"
                    break
                if not members:
                    # A child can exit between poll and ps; resolve that race once.
                    if process.poll() is None:
                        raise RuntimeError("Running process group RSS unavailable")
                    continue
                if time.monotonic() - last_progress >= 2:
                    receipt["elapsed_seconds"] = time.monotonic() - started
                    write_json(output / "supervision.json", receipt)
                    last_progress = time.monotonic()
                time.sleep(min(POLL_SECONDS, max(0., seconds-(time.monotonic()-started))))
    except BaseException as error:
        receipt["kill_reason"] = receipt["kill_reason"] or "supervision_exception"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None:
            try:
                # Kill the group even if its leader has exited, to clean descendants.
                if receipt["kill_reason"] or process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                receipt["exit_code"] = process.wait(timeout=2)
            except BaseException as error:
                receipt["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["status"] = ("completed" if receipt["exit_code"] == 0
                             and receipt["kill_reason"] is None
                             and receipt["cleanup_error"] is None
                             and receipt["elapsed_seconds"] < seconds else "failed_or_incomplete")
        write_json(output / "supervision.json", receipt)
    return receipt


def safe_path(destination, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    target = destination.joinpath(*relative.parts)
    if target != destination and destination not in target.resolve().parents:
        raise ValueError(f"Archive member escapes destination: {name}")
    return target


def extract_zip(path, destination, *, expanded_limit):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ENTRIES or sum(item.file_size for item in entries) > expanded_limit:
            raise ValueError("Zip expansion cap")
        seen = set()
        for item in entries:
            target = safe_path(destination, item.filename)
            if target in seen:
                raise ValueError("Duplicate zip destination")
            seen.add(target)
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Unexpected wheel symlink")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as stream:
                    copied = 0
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        copied += len(block)
                        if copied > item.file_size:
                            raise ValueError("Zip member exceeded declared size")
                        stream.write(block)
                target.chmod(0o755 if mode & 0o111 else 0o644)


def extract_tar(stream, destination, *, expanded_limit, strip_root=None):
    destination.mkdir(parents=True, exist_ok=True)
    total, count, links, seen = 0, 0, [], set()
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for item in archive:
            count += 1
            total += item.size
            if count > MAX_ENTRIES or total > expanded_limit:
                raise ValueError("Tar expansion cap")
            name = item.name
            if strip_root:
                parts = PurePosixPath(name).parts
                if not parts or parts[0] != strip_root:
                    raise ValueError("Source archive root does not match exact commit")
                name = str(PurePosixPath(*parts[1:]))
            target = safe_path(destination, name)
            if target in seen:
                raise ValueError("Duplicate tar destination")
            seen.add(target)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(item)
                if source is None:
                    raise ValueError("Missing tar member contents")
                with source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                target.chmod(0o755 if item.mode & 0o111 else 0o644)
            elif item.issym():
                if PurePosixPath(item.linkname).is_absolute() or "\\" in item.linkname:
                    raise ValueError("Unsafe archive link")
                resolved = (target.parent / item.linkname).resolve()
                if resolved != destination and destination not in resolved.parents:
                    raise ValueError("Archive link escapes destination")
                links.append((target, item.linkname))
            else:
                raise ValueError("Unsupported archive member type")
    # Materialize links last; no file can be written through an archive-created link.
    for target, linkname in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(linkname)
    for target, _ in links:
        resolved = target.resolve()
        if resolved != destination and destination not in resolved.parents:
            raise ValueError("Resolved archive link chain escapes destination")
    return {"entries": count, "expanded_bytes": total}


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = str(path.relative_to(directory))
        if path.is_symlink():
            result[relative] = {"symlink": os.readlink(path)}
        elif path.is_file():
            result[relative] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def download(url, path, *, limit, expected_sha=None, expected_bytes=None):
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=15) as response, path.open("xb") as output:
        final_url = response.url
        for block in iter(lambda: response.read(1024 * 1024), b""):
            size += len(block)
            if size > limit:
                raise ValueError("Acquisition byte cap")
            digest.update(block)
            output.write(block)
    value = digest.hexdigest()
    if expected_sha is not None and value != expected_sha:
        raise ValueError("Acquisition SHA256 mismatch")
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError("Acquisition size mismatch")
    return {"url": url, "final_url": final_url, "bytes": size, "sha256": value}


def acquire(spec, output):
    PREFIX.mkdir(parents=True, exist_ok=False)
    downloads = PREFIX / "downloads"
    downloads.mkdir()
    record = {"status": "acquiring", "downloads": {}, "declaration_sha256": DECLARATION_SHA}
    consumed = 0
    for item in spec["dependencies"]:
        record["current"] = item["name"]
        write_json(output / "worker-progress.json", record)
        entry = download(item["url"], downloads / item["filename"],
                         limit=min(item["bytes"], spec["caps"]["aggregate_acquisition_bytes"]-consumed),
                         expected_sha=item["sha256"], expected_bytes=item["bytes"])
        consumed += entry["bytes"]
        record["downloads"][item["name"]] = entry
    source_archive = downloads / (spec["source"]["commit"] + ".tar.gz")
    record["current"] = "FEBio source"
    write_json(output / "worker-progress.json", record)
    entry = download(spec["source"]["archive_url"], source_archive,
                     limit=min(spec["caps"]["source_archive_compressed_bytes"],
                               spec["caps"]["aggregate_acquisition_bytes"]-consumed))
    consumed += entry["bytes"]
    record["downloads"]["FEBio source"] = entry
    write_json(output / "worker-progress.json", record)
    by_name = {item["name"]: item for item in spec["dependencies"]}
    extract_zip(downloads / by_name["cmake"]["filename"], PREFIX / "tools",
                expanded_limit=512*1024**2)
    extract_zip(downloads / by_name["zstandard"]["filename"], PREFIX / "zstandard-helper",
                expanded_limit=32*1024**2)
    sys.path.insert(0, str(PREFIX / "zstandard-helper"))
    zstandard = importlib.import_module("zstandard")
    if not Path(zstandard.__file__).resolve().is_relative_to(PREFIX / "zstandard-helper"):
        raise ValueError("Archive helper imported from wrong origin")
    omp_path = downloads / by_name["llvm-openmp"]["filename"]
    with zipfile.ZipFile(omp_path) as archive:
        names = archive.namelist()
        expected_names = {"metadata.json", "pkg-"+omp_path.stem+".tar.zst", "info-"+omp_path.stem+".tar.zst"}
        if set(names) != expected_names or len(names) != 3:
            raise ValueError("Unexpected conda package layout")
        for name in sorted(names):
            if name.endswith(".tar.zst"):
                with archive.open(name) as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                    extract_tar(stream, PREFIX / "openmp", expanded_limit=16*1024**2)
    with source_archive.open("rb") as stream:
        source_expansion = extract_tar(stream, PREFIX / "source",
                                      expanded_limit=spec["caps"]["source_archive_expanded_bytes"],
                                      strip_root="FEBio-"+spec["source"]["commit"])
    inspection_path = DECLARATION.parent / spec["source"]["source_inspection_receipt"]
    if sha(inspection_path) != spec["source"]["source_inspection_receipt_sha256"]:
        raise ValueError("Source inspection receipt changed")
    for name, expected in json.loads(inspection_path.read_text())["files"].items():
        if "sha256" in expected and sha(PREFIX / "source" / name) != expected["sha256"]:
            raise ValueError("Extracted source differs from inspected commit: "+name)
    # Payload license discovery is recorded for explicit review, not treated as a legal conclusion.
    notices = [str(path.relative_to(PREFIX)) for path in PREFIX.rglob("*")
               if path.is_file() and any(word in path.name.lower() for word in ("license", "copyright", "notice"))
               and "source/Documentation/" not in str(path.relative_to(PREFIX))]
    record.update(status="acquired_pending_payload_review", aggregate_download_bytes=consumed,
                  source_expansion=source_expansion, notice_paths=sorted(notices))
    record.pop("current", None)
    write_json(output / "acquisition.json", record)
    write_json(output / "input-inventory.json", inventory(PREFIX))


def verify_stage_receipt(receipt_directory, stage):
    acceptance = json.loads((receipt_directory / "acceptance.json").read_text())
    supervision = json.loads((receipt_directory / "supervision.json").read_text())
    result = json.loads((receipt_directory / "result.json").read_text())
    if any(item["status"] != "completed" for item in (acceptance, supervision, result)):
        raise ValueError("Stage did not receive parent acceptance: "+stage)
    for item in (acceptance, result):
        if (item["stage"] != stage or item["driver_sha256"] != sha(Path(__file__))
                or item["declaration_sha256"] != DECLARATION_SHA):
            raise ValueError("Accepted stage source binding mismatch")
    required = {"supervision.json", "result.json"} | {
        "acquire": {"acquisition.json", "input-inventory.json"},
        "configure": {"verified-cache.json"},
        "build": {"installed-inventory.json"}}[stage]
    if set(acceptance["artifact_sha256"]) != required:
        raise ValueError("Accepted stage artifact inventory incomplete")
    for name, expected in acceptance["artifact_sha256"].items():
        if sha(receipt_directory / name) != expected:
            raise ValueError("Accepted stage artifact changed: "+name)
    return result


def verify_acquisition(receipt_directory):
    verify_stage_receipt(receipt_directory, "acquire")
    expected = json.loads((receipt_directory / "input-inventory.json").read_text())
    # Outputs from later stages live only in build/install; acquired inputs remain pinned.
    actual = {name: value for name, value in inventory(PREFIX).items()
              if name.split("/", 1)[0] not in {"build", "install", "state"}}
    if actual != expected:
        raise ValueError("Acquired runtime inputs changed")


def run_command(command, output, name):
    write_json(output / (name+"-command.json"), {"argv": command, "cwd": str(ROOT)})
    subprocess.run(command, cwd=ROOT, check=True)


def configured_options(spec):
    return {key: value.replace("{repository}", str(ROOT))
            for key, value in spec["configure"]["cache_options"].items()}


def verify_cache(spec):
    cache = {}
    for line in (PREFIX / "build/CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for key, value in configured_options(spec).items():
        if cache.get(key) != value:
            raise ValueError("Configured cache mismatch: "+key)
    for language in ("C", "CXX"):
        if cache.get("OpenMP_"+language+"_LIB_NAMES") != "omp":
            raise ValueError("Private OpenMP not selected")
    return {key: cache[key] for key in configured_options(spec)}


def build_stage(stage, spec, output, acquisition, configuration):
    verify_acquisition(acquisition)
    if stage == "build":
        if configuration is None:
            raise ValueError("Accepted configure receipt required before build")
        configured = verify_stage_receipt(configuration, "configure")
        if (configured["configured_cache_sha256"] != sha(PREFIX / "build/CMakeCache.txt")
                or configured["acquisition_inventory_sha256"] != sha(acquisition / "input-inventory.json")):
            raise ValueError("Configured cache or acquisition binding changed")
    state = PREFIX / "state"
    state.mkdir(exist_ok=True)
    with (state / (stage+"-attempt.json")).open("x") as marker:
        json.dump({"receipt_directory": str(output), "driver_sha256": sha(Path(__file__)),
                   "declaration_sha256": DECLARATION_SHA}, marker)
    cmake = PREFIX / "tools/cmake/data/bin/cmake"
    if stage == "configure":
        if (PREFIX / "build").exists():
            raise FileExistsError("Fresh build directory required; no automatic retry")
        command = [str(cmake), "-S", str(PREFIX / "source"), "-B", str(PREFIX / "build"),
                   "-G", spec["configure"]["generator"]]
        command += ["-D"+key+"="+value for key, value in configured_options(spec).items()]
        run_command(command, output, "configure")
        write_json(output / "verified-cache.json", verify_cache(spec))
    else:
        verify_cache(spec)
        if (PREFIX / "install").exists():
            raise FileExistsError("Fresh install directory required")
        run_command([str(cmake), "--build", str(PREFIX / "build"), "--target", "febio4",
                     "--parallel", str(spec["build"]["parallel_jobs"])], output, "build")
        run_command([str(cmake), "--install", str(PREFIX / "build"), "--prefix", str(PREFIX / "install")],
                    output, "install")
        write_json(output / "installed-inventory.json", inventory(PREFIX / "install"))


def worker(stage, output, acquisition, configuration=None):
    result = {"status": "running", "stage": stage, "declaration_sha256": DECLARATION_SHA,
              "driver_sha256": sha(Path(__file__)), "solver_executed": False}
    write_json(output / "result.json", result)
    try:
        spec = declaration()
        if stage == "acquire":
            acquire(spec, output)
        else:
            if acquisition is None:
                raise ValueError("Acquisition receipt required")
            build_stage(stage, spec, output, acquisition, configuration)
            result["acquisition_inventory_sha256"] = sha(acquisition / "input-inventory.json")
            if stage == "configure":
                result["configured_cache_sha256"] = sha(PREFIX / "build/CMakeCache.txt")
        if sha(Path(__file__)) != result["driver_sha256"] or sha(DECLARATION) != DECLARATION_SHA:
            raise ValueError("Driver or declaration changed during stage")
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed_or_incomplete", error={"type": type(error).__name__, "message": str(error)})
    write_json(output / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "configure", "build"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    acquisition = args.acquisition.resolve() if args.acquisition else None
    configuration = args.configuration.resolve() if args.configuration else None
    if args.worker:
        return worker(args.stage, output, acquisition, configuration)
    spec = declaration()
    driver_sha = sha(Path(__file__))
    seconds = {"acquire": ACQUISITION_SECONDS, "configure": spec["caps"]["configure_seconds"],
               "build": spec["caps"]["build_and_install_seconds"]}[args.stage]
    command = [sys.executable, str(Path(__file__).resolve()), args.stage, "--worker", "--output", str(output)]
    if acquisition:
        command += ["--acquisition", str(acquisition)]
    if configuration:
        command += ["--configuration", str(configuration)]
    receipt = supervise(command, output, cwd=ROOT, environment=private_environment(spec),
                        seconds=seconds, rss_bytes=spec["caps"]["process_family_sampled_rss_bytes"])
    try:
        result = json.loads((output / "result.json").read_text())
        accepted = (receipt["status"] == "completed" and result["status"] == "completed"
                    and result["driver_sha256"] == driver_sha == sha(Path(__file__))
                    and result["declaration_sha256"] == DECLARATION_SHA == sha(DECLARATION)
                    and result["stage"] == args.stage and result["solver_executed"] is False)
        error = None
    except (OSError, ValueError, KeyError) as exception:
        accepted = False
        error = {"type": type(exception).__name__, "message": str(exception)}
    artifacts = {name: sha(output / name) for name in
                 ("supervision.json", "result.json", "acquisition.json", "input-inventory.json",
                  "verified-cache.json", "installed-inventory.json") if (output / name).exists()}
    write_json(output / "acceptance.json", {"status": "completed" if accepted else "failed_or_incomplete",
               "driver_sha256": driver_sha, "declaration_sha256": DECLARATION_SHA,
               "stage": args.stage, "artifact_sha256": artifacts,
               "error": error, "solver_executed": False})
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
