"""One owned generated-transfer child with a sampled RSS and wall watchdog.

Adapted from the previously reviewed frozen_preflight_supervisor single-process
pattern. The child uses native threads and may not create subordinate workers.
Cancellation and every stop retain an attempt receipt; no automatic retry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MAX_WALL_SECONDS = 20.0
MAX_RSS_BYTES = 1024 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_OPERATION_SECONDS = 25.0
SAMPLE_INTERVAL_SECONDS = .2
SOURCE_SHA256 = {
    "local:legacy_transfer_worker.py": "88c4646eaf51b8c96f6962e166f1d2594271807e27a517fcdcd576dd436b2d01",
    "local:fixed_rl256_loader.py": "3907d44928512b897f4a960b5e56f1fa7c00c17e756110b4070b8fb24c553638",
    "local:legacy_transfer_episode.py": "a14e04ab2d948f283a4585b8834693a953cea71eefbf6525b673d7e1993f5bcd",
    "local:development_episode.py": "8c7510cea44b43907c04bb739651edab1fc73d01720490305dfcde0b0f95002e",
    "local:legacy_aspiration_projection.py": "69d0003350f30fa03996a272137521ae19a0f511667872ffb45cc386d7b1132c",
    "local:shared_vascular_evaluation.py": "300f42604a69b207185417045bb88e392cdeea3e2c00713b7dcee9071c1861c0",
    "local:legacy_checkpoint_rollout_v3.py": "b48831841b27742d123d1338ef2ff05005f39f1d6b755617c2c7bc959ea749e4",
    "src/resectionlab/core.py": "094bb902552117ee0be1ee4ed79ae0b09a7005e48d24ca4a9a4674ff54af077b",
    "src/resectionlab/native_resection.py": "0cbbe2904f6b87cfb97d555121cc70690989d70a971361ec417735a7061e3a3c",
    "src/resectionlab/native_spatial_task.py": "1307b444ba3d9457928545834790be182b95891da1a7bc585bff36776d235cea",
    "src/resectionlab/spatial_policy.py": "5801f1b59b5eed90edeb33e5182d57547de35dd0fd326c76c4e427f89c05b6a4",
    "src/resectionlab/spatial_observations.py": "48a2a9d1c7faee791f7f327ac928e133f2d0f80f60a045a868802bb75785d711",
    "src/resectionlab/observed_search.py": "cdbacc079fa3e4f8d49a356e76dfdc77e4484ef4bb1a3383970a0e937c72febd",
    "src/resectionlab/shared_episode.py": "81d6b1faac3433f1a2eb09ff4ef70e773b69922c056495caf1a879fbbc381a4d",
    "src/resectionlab/sequential_spatial_observation.py": "c8791eccbc45c96449ba7dd9578881a486654d2cd65e092157a79cde3b8673a3",
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path, record):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("The fixed local generated release is unavailable")


def _check_sources():
    root = _root()
    if not SOURCE_SHA256:
        raise RuntimeError("Transfer source closure has not been reviewed and frozen")
    for relative, expected in SOURCE_SHA256.items():
        path = (Path(__file__).with_name(relative.removeprefix("local:"))
                if relative.startswith("local:") else root / relative)
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise ValueError("Bound transfer source changed: " + relative)
    return root


def _cleanup(process, errors):
    # This workflow is one owned process with native threads. Do not signal a
    # historical process group or an unrelated acquisition process.
    for label, method in (("TERM", process.terminate), ("KILL", process.kill)):
        if process.poll() is not None:
            return
        try:
            method()
        except ProcessLookupError:
            return
        except OSError as error:
            errors.append(label + ":" + type(error).__name__ + ":" + str(error))
        try:
            process.wait(timeout=2)
            return
        except (subprocess.TimeoutExpired, OSError) as error:
            errors.append("wait_" + label + ":" + type(error).__name__ + ":" + str(error))


def run_attempt(output_directory: Path, *, cancelled=None):
    """Run exactly one attempt; the caller chooses a fresh backend-owned path."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    start_receipt = {"status": "started", "scope": "generated_RL256_aspiration_transfer",
        "maxWallSeconds": MAX_WALL_SECONDS, "maxSampledWorkerRssBytes": MAX_RSS_BYTES,
        "automaticRetry": False, "sampleIntervalSeconds": SAMPLE_INTERVAL_SECONDS}
    _receipt(output / "attempt.json", start_receipt)
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError("Transfer cancelled before checkpoint preflight")
        root = _check_sources()
        worker = Path(__file__).with_name("legacy_transfer_worker.py")
        lease_read, lease_write = os.pipe()
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        env["RESECTIONLAB_PARENT_LEASE_FD"] = str(lease_read)
        reason, errors, samples, peak, process = None, [], 0, 0, None
        with (output / "worker.log").open("w") as log:
            try:
                # A fresh pycache location cannot read an earlier valid .pyc;
                # isolated mode also ignores inherited Python import settings.
                process = subprocess.Popen([sys.executable, "-I", "-B", "-X",
                    "pycache_prefix=" + str(output / "fresh-pycache"), str(worker), "--output",
                    str(output / "result.json")], stdout=log, stderr=subprocess.STDOUT,
                    cwd=root, env=env, start_new_session=True, pass_fds=(lease_read,))
            except BaseException:
                os.close(lease_write)
                raise
            finally:
                os.close(lease_read)
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if cancelled is not None and cancelled():
                        reason = "cancelled"
                        break
                    if elapsed >= MAX_WALL_SECONDS:
                        reason = "wall_budget_exceeded"
                        break
                    if (os.fstat(log.fileno()).st_size > MAX_LOG_BYTES or
                            ((output / "result.json").exists() and
                             (output / "result.json").stat().st_size > MAX_RESULT_BYTES)):
                        reason = "runtime_output_budget_exceeded"
                        break
                    try:
                        measurement = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                            capture_output=True, text=True, timeout=.5, check=False)
                    except subprocess.TimeoutExpired:
                        reason = "rss_measurement_timeout"
                        break
                    try:
                        current = int(measurement.stdout.strip()) * 1024
                    except ValueError:
                        if process.poll() is None:
                            reason = "rss_unavailable_for_live_worker"
                            break
                        current = 0
                    samples += 1
                    peak = max(peak, current)
                    if current > MAX_RSS_BYTES:
                        reason = "sampled_rss_budget_exceeded"
                        break
                    _receipt(output / "progress.json", {"workerPid": process.pid,
                        "elapsedSeconds": elapsed, "sampledPeakRssBytes": peak,
                        "samples": samples, "stopReason": reason})
                    time.sleep(SAMPLE_INTERVAL_SECONDS)
            except BaseException as error:
                reason = "supervisor_error:" + type(error).__name__ + ":" + str(error)
            finally:
                try:
                    _cleanup(process, errors)
                finally:
                    os.close(lease_write)
            code = process.poll()
        elapsed = time.monotonic() - started
        if code is None:
            reason = reason or "worker_cleanup_unconfirmed"
        if elapsed >= MAX_OPERATION_SECONDS:
            reason = reason or "wall_budget_exceeded"
        result_path = output / "result.json"
        complete = (code == 0 and reason is None and not errors and samples > 0 and peak > 0 and
            (output / "worker.log").stat().st_size <= MAX_LOG_BYTES and
            result_path.is_file() and result_path.stat().st_size <= MAX_RESULT_BYTES)
        record = {"status": "complete" if complete else "failed", "workerReturnCode": code,
            "workerTerminationConfirmed": code is not None,
            "unresolvedWorkerPid": process.pid if code is None else None,
            "cleanupErrors": errors, "stopReason": reason, "elapsedSeconds": elapsed,
            "sampledPeakRssBytes": peak, "samples": samples,
            "samplingLimit": "transient peaks between samples may be missed",
            "automaticRetry": False}
        _receipt(output / "supervision.json", record)
        if not complete:
            raise RuntimeError("Owned transfer child did not finish: " + str(reason or code))
        result = json.loads(result_path.read_text())
        if (result.get("status") != "complete" or result.get("clinicalValidation") is not False
                or result.get("newOptimizerUpdates") != 0):
            raise RuntimeError("Completed transfer child output has wrong scope")
        _check_sources()
        _receipt(output / "completion.json", {"status": "complete", "resultSha256": _sha(result_path),
            "supervisionSha256": _sha(output / "supervision.json")})
        return result
    except BaseException as error:
        _receipt(output / "failure.json", {"status": "failed", "errorType": type(error).__name__,
            "error": str(error), "automaticRetry": False})
        raise
