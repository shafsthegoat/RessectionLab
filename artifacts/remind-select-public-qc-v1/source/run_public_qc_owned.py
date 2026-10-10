"""One acquired-public SELECT batch; original frozen roles and canonical reader."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "build/goal-conditioned-policy-v1"))
from run_contact_owned import FastDarwinSampler, cleanup_owned, _receipt

CASES = (("ReMIND-013", "9d4448328e979c856e3c8e08599b9cc2b66fc7d324f1652d7970468099267c65"),
         ("ReMIND-037", "89ebb69cdd8d104eb726e402b2c09da9a4c3799a6bef97f7a646903562120fd9"))
PINS = {"src/resectionlab/remind_planning_qc.py": "db146b0610b0b2aaf8c9d3cc08478920b2746f4b5065937a43a320054700f9a8",
        "scripts/prepare_remind_planning.py": "155e32f3c4605407fe4c3e65de5c0445bae7918002daf803d2dde470fe350b35",
        "scripts/convert_remind_development.py": "34e6ceb9560344a88226489b2dd8ca6bfb90cb87c52d2ff3184d393293df4ef6"}
CAPS = {"seconds_per_case": 300, "sampled_rss_bytes": 3*1024**3, "output_bytes_per_case": 256*1024**2,
        "threads": 1, "automatic_retry": False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for path, expected in PINS.items():
        if sha(ROOT / path) != expected: raise ValueError("canonical source changed: " + path)
    bound = []
    for subject, expected in CASES:
        path = ROOT / "build/remind-select-public-bindings-v1" / (subject + "-case.json")
        if sha(path) != expected: raise ValueError("case changed")
        case = json.loads(path.read_bytes())
        if (case["patient_id"] != subject or case["role"] != "SELECT" or len(case["series"]) != 3
                or {s["kind"] for s in case["series"]} != {"structural_t1ce", "cerebrum", "whole_tumor"}):
            raise ValueError("exact public SELECT scope required")
        bound.append((subject, path, expected))
    output = HERE / "actual-public-qc-v1"; output.mkdir(exist_ok=False)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "LD_PRELOAD", "DYLD_LIBRARY_PATH"):
        env.pop(key, None)
    runtime = ROOT / "build/idc-acquisition-venv/bin/python"
    _receipt(output / "declaration.json", {"source_sha256": PINS, "parent_sha256": sha(__file__), "cases": CASES,
        "caps": CAPS, "runtime": str(runtime), "public_only": True, "policy_or_training": False,
        "internal_crop_cap": "existing110s/2GiB checkpoints retained", "scope": "public source qualification, anatomy unreviewed"})
    sampler = FastDarwinSampler(); started = time.monotonic(); outcomes = []; fatal = None
    def stop(*args): raise InterruptedError("public QC parent interrupted")
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for subject, case, expected in bound:
            case_started = time.monotonic(); case_paths = []
            for phase in ("headers", "crop-mr"):
                target = output / (subject + "-" + phase)
                args = [str(runtime), "-I", "-B", "-X", "pycache_prefix=" + str(output / "fresh-pycache"),
                    str(ROOT / "scripts/prepare_remind_planning.py"), phase, "--public-only", "--case", str(case),
                    "--case-sha256", expected, "--repository-root", str(ROOT), "--output", str(target)]
                if phase == "crop-mr":
                    header = output / (subject + "-headers/result.json")
                    args += ["--headers", str(header), "--headers-sha256", sha(header)]
                process = None; peak = 0; reason = None; actions = []; errors = []; detached = {}; remaining = []
                phase_started = time.monotonic(); case_paths.append(target)
                try:
                    with (output / (subject + "-" + phase + ".log")).open("x") as log:
                        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        while process.poll() is None:
                            if time.monotonic()-case_started >= CAPS["seconds_per_case"]: reason = "case_wall_cap"; break
                            total = sum(p.stat().st_size for folder in case_paths if folder.exists() for p in folder.rglob("*") if p.is_file())
                            if total > CAPS["output_bytes_per_case"] or os.fstat(log.fileno()).st_size > 4*1024**2:
                                reason = "output_cap"; break
                            measured = sampler.group(process.pid, process.pid)
                            detached.update(measured["detached_descendant_start_identities"])
                            peak = max(peak, measured["process_group_resident_bytes"])
                            if peak > CAPS["sampled_rss_bytes"]: reason = "sampled_memory_cap"; break
                            time.sleep(.1)
                finally:
                    if process is not None: remaining = cleanup_owned(process, sampler, detached, actions, errors)
                    total = sum(p.stat().st_size for folder in case_paths if folder.exists() for p in folder.rglob("*") if p.is_file())
                    if total > CAPS["output_bytes_per_case"]: reason = reason or "output_cap"
                    if time.monotonic()-case_started >= CAPS["seconds_per_case"]: reason = reason or "case_wall_cap"
                    record = {"subject": subject, "phase": phase, "argv": args, "elapsed_seconds": time.monotonic()-phase_started,
                        "case_elapsed_seconds": time.monotonic()-case_started, "sampled_peak_rss_bytes": peak,
                        "case_output_bytes": total,
                        "exit_code": None if process is None else process.poll(), "stop_reason": reason,
                        "cleanup_actions": actions, "cleanup_errors": errors, "remaining_owned_pids": remaining,
                        "reaped": process is not None and process.poll() is not None}
                    outcomes.append(record)
                    _receipt(output / (subject + "-" + phase + "-supervisor.json"), record)
                if reason or errors or remaining:
                    fatal = reason or "cleanup_failure"; break
                result_path = target / ("result.json" if phase == "headers" else "conversion-result.json")
                result = json.loads(result_path.read_bytes()) if result_path.exists() else {}
                record["result_sha256"] = sha(result_path) if result_path.exists() else None
                record["result_status"] = result.get("status")
                if record["exit_code"] != 0 or result.get("status") == "failed": break
            if fatal: break
    except BaseException as error:
        fatal = type(error).__name__ + ":" + str(error)
    finally:
        _receipt(output / "receipt.json", {"outcomes": outcomes, "elapsed_seconds": time.monotonic()-started,
            "fatal": fatal, "roles_changed": False, "public_only": True, "caps": CAPS,
            "memory_limit": "sampled, not kernel hard", "failed_cases_replaced": False})
        signal.signal(signal.SIGTERM, previous)
    return 0 if fatal is None and len(outcomes) == 4 and all(r["exit_code"] == 0 for r in outcomes) else 1


if __name__ == "__main__": raise SystemExit(main())
