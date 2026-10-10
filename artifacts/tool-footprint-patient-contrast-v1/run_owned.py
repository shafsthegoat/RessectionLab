"""One fixed footprint-opening proposal contrast, reusing the existing owned process monitor/cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build/goal-conditioned-policy-v1"))
from run_contact_owned import sha, cleanup_owned, FastDarwinSampler, _receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    if sha(args.release) != args.release_sha256: raise ValueError("Root release changed")
    release = json.loads(args.release.read_text())
    if release["status"] != "released_one_attempt": raise ValueError("Root execution release required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    if head != release["expected_head"]: raise ValueError("Released canonical HEAD changed")
    index_path = ROOT / release["source_index"]["path"]
    if sha(index_path) != release["source_index"]["sha256"]: raise ValueError("Source index changed")
    index = json.loads(index_path.read_text())
    for relative, expected in index["source_files"].items():
        if sha(ROOT / relative) != expected: raise ValueError("Bound source changed: "+relative)
    public = release["public_manifest"]
    if sha(ROOT/public["path"]) != public["sha256"]: raise ValueError("Public manifest changed")
    if (release["output"] != "build/cross-patient-planning-v1/ReMIND-008-tool-footprint-opening-search-v1"
            or release["profile"] is not False):
        raise ValueError("One fixed output and this unprofiled contrast required")
    expected_limits = {"memory_bytes": 3221225472, "native_previews": 2000,
        "optimizer_updates": 0, "parent_seconds": 120, "policy_forwards": 0,
        "search_transition_calls": 858, "search_seconds": 60, "threads": 1,
        "worker_seconds": 110, "beam_width": 2, "horizon": 6}
    if release["limits"] != expected_limits: raise ValueError("Fixed diagnostic limits changed")
    if release["retention_policy"] != "return_plus_opening_depth_v1":
        raise ValueError("One declared two-lane retention policy required")
    if release["tool_footprint_opening"] is not True or release["requested_opening_lookahead_mm"] != 1.:
        raise ValueError("Fixed original intermediate rule plus footprint option required")
    output = ROOT/release["output"]
    if output.exists(): raise FileExistsError("Attempt already exists; no retry or overwrite")
    supervision = output.with_name(output.name+".supervision"); supervision.mkdir(exist_ok=False)
    caps = {"hard_total_wall_seconds": 120., "worker_deadline_seconds": 110.,
        "sampled_owned_tree_rss_bytes": 3221225472, "threads": 1,
        "sample_interval_seconds": .2, "attempts": 1, "automatic_retry": False}
    _receipt(supervision/"declaration.json", {"version": "owned-ReMIND-008-tool-footprint-opening-search-v1",
        "head": head, "release_path": str(args.release.resolve()), "release_sha256": args.release_sha256,
        "source_index": release["source_index"], "source_files": index["source_files"],
        "caps": caps, "scope": "tool_footprint_opening_search_zero_learning_or_inference",
        "profiled_timing": release["profile"], "output": str(output)})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    lease_read, lease_write = os.pipe(); env["RESECTIONLAB_PARENT_LEASE_FD"] = str(lease_read)
    command = [sys.executable, "-I", "-B", "-X", "pycache_prefix="+str(supervision/"fresh-pycache"),
        str(Path(__file__).with_name("search_worker.py")), "--release", str(args.release.resolve()),
        "--release-sha256", args.release_sha256]
    sampler = FastDarwinSampler(); started = time.monotonic(); peak = samples = 0
    reason = None; errors = []; process = None; actions = []; detached = {}; remaining = []
    try:
        with (supervision/"worker.log").open("x") as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(lease_read,), start_new_session=True)
            finally: os.close(lease_read)
            try:
                while process.poll() is None:
                    elapsed = time.monotonic()-started
                    if elapsed >= caps["hard_total_wall_seconds"]: reason = "hard_total_wall_cap"; break
                    if os.fstat(log.fileno()).st_size > 8*1024*1024: reason = "worker_log_cap"; break
                    measured = sampler.group(process.pid, process.pid)
                    detached.update(measured["detached_descendant_start_identities"])
                    samples += 1; peak = max(peak, measured["process_group_resident_bytes"])
                    if measured["process_group_resident_bytes"] > caps["sampled_owned_tree_rss_bytes"]:
                        reason = "sampled_owned_tree_rss_cap"; break
                    _receipt(supervision/"progress.json", {"elapsed_seconds": elapsed, "worker_pid": process.pid,
                        "sampled_peak_rss_bytes": peak, "samples": samples, "owned_process_measurement": measured})
                    time.sleep(.2)
            finally: remaining = cleanup_owned(process, sampler, detached, actions, errors)
    except BaseException as error:
        reason = reason or type(error).__name__+":"+str(error); raise
    finally:
        os.close(lease_write)
        elapsed = time.monotonic()-started; code = None if process is None else process.poll()
        if elapsed >= caps["hard_total_wall_seconds"]: reason = reason or "hard_total_wall_cap"
        result_path = output/"result.json"
        result = {}; result_read_error = None
        try:
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if not isinstance(result, dict): raise ValueError("Worker result must be a JSON object")
        except (OSError, ValueError) as error:
            result = {}; result_read_error = type(error).__name__+":"+str(error)
            reason = reason or "worker_result_unreadable"
        complete = (reason is None and code == 0 and not errors and not remaining
                    and result.get("status") == "complete_tool_footprint_opening_search_contrast")
        _receipt(supervision/"receipt.json", {"status": "complete" if complete else "failed",
            "stop_reason": reason, "exit_code": code, "elapsed_seconds": elapsed,
            "sampled_peak_rss_bytes": peak, "samples": samples, "cleanup_actions": actions,
            "cleanup_errors": errors, "final_owned_pids": remaining, "worker_termination_confirmed": code is not None,
            "result_read_error": result_read_error,
            "result_sha256": sha(result_path) if result_path.is_file() and result_read_error is None else None,
            "profile_sha256": sha(output/"profile.pstats") if (output/"profile.pstats").is_file() else None,
            "caps": caps, "automatic_retry": False,
            "sampling_limit": "transient peaks or detached descendants between samples may be missed",
            "worker_process_scope": "owned process group plus discovered descendants via FastDarwinSampler"})
    if not complete: raise RuntimeError("Owned public preflight failed or capped: "+str(reason or code))


if __name__ == "__main__": main()
