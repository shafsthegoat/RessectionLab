"""One owned generated contact-family child with RSS and wall watchdog.

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
# Root fills this exact-source closure after promoting the frozen learning
# modules, new family candidate, and this worker. An empty or partial closure
# refuses release, including STOP.
SOURCE_FILES = frozenset({
    "src/resectionlab/goal_relation_spatial_policy.py",
    "local:contact_family_worker.py",
    "local:contact_family_desktop_release.py",
    "src/resectionlab/contact_learning_contract.py",
    "src/resectionlab/contact_learning.py",
    "src/resectionlab/contact_checkpoint.py",
    "src/resectionlab/contact_family_episode.py",
    "src/resectionlab/goal_mode_episode_adapter.py",
    "src/resectionlab/goal_mode_spatial_policy.py",
    "src/resectionlab/public_contact_family.py",
    "src/resectionlab/public_surface_contact.py",
    "src/resectionlab/surface_contact_episode.py",
    "src/resectionlab/development_episode.py",
    "src/resectionlab/core.py",
    "src/resectionlab/native_resection.py",
    "src/resectionlab/native_spatial_task.py",
    "src/resectionlab/native_proposals.py",
    "src/resectionlab/data_policy.py",
    "src/resectionlab/geometry.py",
    "src/resectionlab/observed_search.py",
    "src/resectionlab/simulation.py",
    "src/resectionlab/worlds.py",
    "src/resectionlab/structural_evidence.py",
    "src/resectionlab/shared_episode.py",
    "src/resectionlab/spatial_policy.py",
    "src/resectionlab/spatial_observations.py",
    "src/resectionlab/sequential_spatial_observation.py",
    "artifacts/public-contact-train-refit-v1/desktop-release.json",
    "local:contact_train_refit_release.py",
})
SOURCE_SHA256 = {
    "artifacts/public-contact-train-refit-v1/desktop-release.json": "68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887",
    "local:contact_family_desktop_release.py": "df739d8a0c855e37aa7d7bce036af402f5dfc054a1fa2d7fcd3e24852a1cd08b",
    "local:contact_family_worker.py": "2b42a3509bdf9a612a3d3a00e3d554b5355bf575df4073a6019c81f8c4f38ee4",
    "local:contact_train_refit_release.py": "005961a81b96e22d78882038e20b792209025d4df69fbc175a751e04985413ee",
    "src/resectionlab/contact_checkpoint.py": "5cb06e8d03d88abca1f0fbc4892a3f09866cb696d4cc20daf860afd9fe136f40",
    "src/resectionlab/contact_family_episode.py": "b8f5a0cecf622cd6c341d91ad3009fff9ed455d9960550b026bc82a8639960a6",
    "src/resectionlab/contact_learning.py": "b4812a076b588594e95979ab8d7c078f2d7e317293438e73a96e86c5bd24a08b",
    "src/resectionlab/contact_learning_contract.py": "8234a4dc0740469a18cb472ce8a5873c7ee12fe9dfa89c362b7ce7263afdeb19",
    "src/resectionlab/core.py": "094bb902552117ee0be1ee4ed79ae0b09a7005e48d24ca4a9a4674ff54af077b",
    "src/resectionlab/data_policy.py": "a06a3ca0712bec146990808157e2214a28185dbf4677d2aa34b0a09ec8255208",
    "src/resectionlab/development_episode.py": "8c7510cea44b43907c04bb739651edab1fc73d01720490305dfcde0b0f95002e",
    "src/resectionlab/geometry.py": "426dedbd11f5f88ab7c85dd3fbb188915e82db7a7e08c2ff9669d77168eb3184",
    "src/resectionlab/goal_mode_episode_adapter.py": "e101e2a8929687816b738aee0411a0d3cdaeb55c77ce9747aed28c0495c9b865",
    "src/resectionlab/goal_mode_spatial_policy.py": "cc1b1384fd719d50e33d1c00c40e54a1bcce9a73069ef56f5a197cde1a85d01a",
    "src/resectionlab/goal_relation_spatial_policy.py": "8cf23e155fd1fd4d1994b4a032ded7c67a23cb8cec50496cd21f3ce2029c02dd",
    "src/resectionlab/native_proposals.py": "f0d37d3b427579073a3d9052c5ee769711149a9b4c2856a4852de743245b12ee",
    "src/resectionlab/native_resection.py": "0cbbe2904f6b87cfb97d555121cc70690989d70a971361ec417735a7061e3a3c",
    "src/resectionlab/native_spatial_task.py": "1307b444ba3d9457928545834790be182b95891da1a7bc585bff36776d235cea",
    "src/resectionlab/observed_search.py": "cdbacc079fa3e4f8d49a356e76dfdc77e4484ef4bb1a3383970a0e937c72febd",
    "src/resectionlab/public_contact_family.py": "f39039f2e52b2fb4d4eaa7781b9a1a8d8e01bf806293f9872241f16b203d7800",
    "src/resectionlab/public_surface_contact.py": "def4c741df95d6c6b4e208189a0ac46db4ac90d9d1d911b5dc266901f09c23a0",
    "src/resectionlab/sequential_spatial_observation.py": "c8791eccbc45c96449ba7dd9578881a486654d2cd65e092157a79cde3b8673a3",
    "src/resectionlab/shared_episode.py": "81d6b1faac3433f1a2eb09ff4ef70e773b69922c056495caf1a879fbbc381a4d",
    "src/resectionlab/simulation.py": "b8e26d956c3d5637fbcd3f00962bc912264cdb1d5544356a919cb84b98991253",
    "src/resectionlab/spatial_observations.py": "48a2a9d1c7faee791f7f327ac928e133f2d0f80f60a045a868802bb75785d711",
    "src/resectionlab/spatial_policy.py": "5801f1b59b5eed90edeb33e5182d57547de35dd0fd326c76c4e427f89c05b6a4",
    "src/resectionlab/structural_evidence.py": "37b7a71c51855750d15ee2570197a75c2484e6e653550f547f9a696ce06c9019",
    "src/resectionlab/surface_contact_episode.py": "718321ef213ecdf7d7c1eeeba06bb607e7357c791dd58c0b645cefb0053f672e",
    "src/resectionlab/worlds.py": "f8bb81ecc0f5fad822878cb4ea8af6b2a2451e6c5c42f76c163b5261d955fa91"
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
    if set(SOURCE_SHA256) != SOURCE_FILES:
        raise RuntimeError("Contact-family source closure has not been reviewed and frozen")
    for relative, expected in SOURCE_SHA256.items():
        path = (Path(__file__).with_name(relative.removeprefix("local:"))
                if relative.startswith("local:") else root / relative)
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise ValueError("Bound contact-family source changed: " + relative)
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


def run_attempt(output_directory: Path, *, layout_id: str, goal_id: str,
                selector: str, cancelled=None):
    """Run exactly one attempt; the caller chooses a fresh backend-owned path."""
    if (type(layout_id) is not str or type(goal_id) is not str or
            type(selector) is not str or selector not in ("STOP", "SEARCH", "IL", "RL", "IL_TRAIN_REFIT")):
        raise ValueError("Choose one exact contact-family layout, goal and selector")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    start_receipt = {"status": "started", "scope": "generated_public_contact_family_v3",
        "layoutId": layout_id, "goalId": goal_id, "selector": selector,
        "maxWallSeconds": MAX_WALL_SECONDS, "maxSampledWorkerRssBytes": MAX_RSS_BYTES,
        "automaticRetry": False, "sampleIntervalSeconds": SAMPLE_INTERVAL_SECONDS}
    _receipt(output / "attempt.json", start_receipt)
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError("Contact-family episode cancelled before source preflight")
        root = _check_sources()
        worker = Path(__file__).with_name("contact_family_worker.py")
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
                    str(output / "result.json"), "--layout-id", layout_id,
                    "--goal-id", goal_id, "--selector", selector], stdout=log, stderr=subprocess.STDOUT,
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
            raise RuntimeError("Owned contact-family child did not finish: " + str(reason or code))
        result = json.loads(result_path.read_text())
        if (result.get("status") != "complete" or result.get("clinicalValidation") is not False
                or type(result.get("newOptimizerUpdates")) is not int or
                result["newOptimizerUpdates"] != 0 or result.get("layoutId") != layout_id
                or result.get("goalId") != goal_id or result.get("selector") != selector):
            raise RuntimeError("Completed contact-family child output has wrong scope")
        _check_sources()
        _receipt(output / "completion.json", {"status": "complete", "resultSha256": _sha(result_path),
            "supervisionSha256": _sha(output / "supervision.json")})
        return {**result, "ownedResultSha256": _sha(result_path),
                "ownedSupervisionSha256": _sha(output / "supervision.json")}
    except BaseException as error:
        _receipt(output / "failure.json", {"status": "failed", "errorType": type(error).__name__,
            "error": str(error), "automaticRetry": False})
        raise
