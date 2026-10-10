"""Owned60s/2GiB/4MiB supervisor for one authorized public SEG diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "build/goal-conditioned-policy-v1"))
from run_contact_owned import FastDarwinSampler, cleanup_owned, _receipt

def main():
    target = HERE / "exact037-seg-diagnostic-v2"
    if target.exists(): raise FileExistsError("One attempt only")
    receipt = HERE / "exact037-seg-supervision-v2"; receipt.mkdir(exist_ok=False)
    worker = HERE / "diagnose_exact037_seg_v2.py"
    _receipt(receipt / "declaration.json", {"worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "caps": {"seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 4*1024**2, "threads": 1},
        "scope": "one exact public ReMIND037 cerebrum SEG and saved public support only"})
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): env[key] = "1"
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "LD_PRELOAD", "DYLD_LIBRARY_PATH"): env.pop(key, None)
    sampler = FastDarwinSampler(); process = None; started = time.monotonic(); peak = 0; reason = None
    actions = []; errors = []; detached = {}; remaining = []
    def stop(*args): raise InterruptedError("public SEG parent interrupted")
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        with (receipt / "worker.log").open("x") as log:
            process = subprocess.Popen([str(ROOT / "build/idc-acquisition-venv/bin/python"), "-I", "-B", str(worker)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.monotonic()-started >= 60: reason = "wall_cap"; break
                if target.exists() and sum(p.stat().st_size for p in target.iterdir() if p.is_file()) > 4*1024**2: reason = "output_cap"; break
                measured = sampler.group(process.pid, process.pid); detached.update(measured["detached_descendant_start_identities"])
                peak = max(peak, measured["process_group_resident_bytes"])
                if peak > 2*1024**3: reason = "memory_cap"; break
                time.sleep(.1)
    except BaseException as error: reason = type(error).__name__+":"+str(error)
    finally:
        if process is not None: remaining = cleanup_owned(process, sampler, detached, actions, errors)
        size = sum(p.stat().st_size for p in target.iterdir() if p.is_file()) if target.exists() else 0
        if size > 4*1024**2: reason = reason or "output_cap"
        record = {"elapsed_seconds": time.monotonic()-started, "sampled_peak_rss_bytes": peak,
            "exit_code": None if process is None else process.poll(), "stop_reason": reason, "output_bytes": size,
            "cleanup_actions": actions, "cleanup_errors": errors, "remaining_owned_pids": remaining,
            "reaped": process is not None and process.poll() is not None, "automatic_retry": False}
        _receipt(receipt / "receipt.json", record); signal.signal(signal.SIGTERM, previous)
    return 0 if record["exit_code"] == 0 and not reason and not errors and not remaining else 1

if __name__ == "__main__": raise SystemExit(main())
