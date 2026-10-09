"""One bounded generated-only child with identity-checked RSS/time stop."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "build/limited-input-guard-design/gliomoda-full128-lazy-feasibility-v2"))
from darwin_fast_sampler import FastDarwinSampler


CAP_BYTES = 512 * 1024 * 1024
WALL_SECONDS = 20.0
OUTPUT = HERE / "six-stage-generated"
CHILD = HERE / "generated_six_stage_control.py"
RUNTIME = ROOT / ".tools/scan-target-runtime/venv/bin/python"
SAMPLER = ROOT / "build/limited-input-guard-design/gliomoda-full128-lazy-feasibility-v2/darwin_fast_sampler.py"
SAMPLER_SHA = "906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists() or digest(SAMPLER) != SAMPLER_SHA:
        raise RuntimeError("output exists or pinned sampler changed")
    sampler = FastDarwinSampler()
    host = sampler.host()
    if host["kernel_pressure_mask"] != 1 or host["available_percent"] < 45:
        raise RuntimeError("host is not quiet enough for tiny generated control")
    OUTPUT.mkdir(mode=0o700)
    log_path = OUTPUT / "worker.log"
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    start = time.monotonic()
    samples = []
    stop_reason = None
    signal_result = None
    with log_path.open("wb") as log:
        child = subprocess.Popen([str(RUNTIME), "-B", str(CHILD)], cwd=ROOT,
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            identity = sampler.process_start_identity(child.pid)
            bound_pgid = os.getpgid(child.pid)
        except (OSError, ProcessLookupError):
            identity = None
            bound_pgid = None
        if identity is None or bound_pgid != child.pid:
            stop_reason = "launch_identity_unverified"
        while child.poll() is None and stop_reason is None:
            elapsed = time.monotonic() - start
            try:
                group = sampler.group(child.pid, child.pid)
            except (OSError, RuntimeError, ValueError) as error:
                stop_reason = "group_sample_error:" + type(error).__name__
                break
            samples.append({"elapsed_seconds": elapsed,
                            "resident_bytes": group["process_group_resident_bytes"],
                            "pids": group["pids"]})
            if group["process_group_resident_bytes"] > CAP_BYTES:
                stop_reason = "sampled_rss_cap_exceeded"
            elif elapsed > WALL_SECONDS:
                stop_reason = "wall_cap_exceeded"
            elif group["pids"] != [child.pid]:
                stop_reason = "unexpected_child_process"
            else:
                time.sleep(0.02)
        if stop_reason is not None and child.poll() is None and identity is not None:
            signal_result = sampler.signal_if_same_process(child.pid, identity, signal.SIGKILL)
        try:
            exit_code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            exit_code = None
            stop_reason = "child_wait_timeout"
    report = {"scope": "one_generated_untrained_tiny_six_stage_control",
              "child_source_sha256": digest(CHILD),
              "sampler_sha256": digest(SAMPLER),
              "cap_bytes": CAP_BYTES, "wall_seconds": WALL_SECONDS,
              "preflight_host": host, "child_pid": child.pid,
              "child_start_identity": identity,
              "elapsed_seconds": time.monotonic() - start,
              "sampled_peak_resident_bytes": max((s["resident_bytes"] for s in samples), default=None),
              "samples": samples, "stop_reason": stop_reason,
              "signal_result": signal_result, "exit_code": exit_code,
              "worker_log_bytes": log_path.stat().st_size}
    (OUTPUT / "supervision.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("exit_code", "stop_reason", "sampled_peak_resident_bytes", "elapsed_seconds")},
                     sort_keys=True), flush=True)
    if exit_code != 0 or stop_reason is not None or report["worker_log_bytes"] > 65_536:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
