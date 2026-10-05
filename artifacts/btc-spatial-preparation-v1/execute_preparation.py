"""One-shot execution of the released four-case frozen preparation snapshot."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


BOOTSTRAP = r'''
import json,pathlib,resource,runpy,sys,time
snapshot=pathlib.Path(sys.argv[1])
dependencies=sys.argv[2]
runtime=pathlib.Path(sys.argv[3])
sys.path[:0]=[str(snapshot/'src'),str(snapshot/'scripts')]
sys.path.append(dependencies)
args=sys.argv[4:]
sys.argv=args
started=time.perf_counter()
runpy.run_path(args[0],run_name='__main__')
import nibabel,numpy,resectionlab.core,resectionlab.imaging
origins={name:str(pathlib.Path(module.__file__).resolve()) for name,module in sys.modules.items()
         if name.startswith('resectionlab') and getattr(module,'__file__',None)}
assert origins and all(pathlib.Path(path).is_relative_to(snapshot/'src') for path in origins.values()), 'Live source imported'
record={'elapsed_seconds':time.perf_counter()-started,
        'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'python':sys.version,'numpy':numpy.__version__,'nibabel':nibabel.__version__,
        'numpy_module':numpy.__file__,'nibabel_module':nibabel.__file__,
        'resectionlab_module_origins':origins,'sys_path':sys.path,
        'live_repository_modules_imported':False}
with runtime.open('x') as handle:json.dump(record,handle,indent=2);handle.write('\n')
'''


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    release_path = directory / "root-release.json"
    expected_release = "62fcd60d8dde099ee73f10ab1ede01e0a12b6a07e1c442aa08ded75b373942de"
    if digest(release_path) != expected_release:
        raise ValueError("Root release changed")
    release = json.loads(release_path.read_text())
    snapshot = root / release["source_snapshot"]
    data = Path(release["data_root"])
    batch_path = directory / "batch-record.json"
    if batch_path.exists():
        raise ValueError("Existing execution record: inspect, never duplicate or overwrite")
    for relative, checksum in release["snapshot_files_sha256"].items():
        if digest(snapshot / relative) != checksum:
            raise ValueError("Frozen implementation changed: " + relative)
    for path in release["output_cases"].values():
        if Path(path).exists():
            raise ValueError("Output already exists; do not overwrite")
    started = time.perf_counter()
    batch = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
             "root_release_sha256": expected_release, "git_commit": release["git_commit"],
             "executor_sha256": digest(Path(__file__)), "status": "running", "attempts": [],
             "failures": [], "training_inference_or_registration_executed": False}
    write(batch_path, batch)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for subject in release["subjects"]:
        log = directory / f"{subject}.log"
        report = directory / f"{subject}-preparation-report.json"
        runtime = directory / f"{subject}-runtime.json"
        source = snapshot / f"artifacts/btc-spatial-acquisition-v1/{subject}-source-manifest.json"
        output = Path(release["output_cases"][subject])
        command = [release["python_executable"], "-I", "-S", "-c", BOOTSTRAP,
                   str(snapshot), release["dependency_site_packages"], str(runtime),
                   str(snapshot / "scripts/prepare_btc_case.py"), "--manifest", str(source),
                   "--data-root", str(data), "--output", str(output), "--report", str(report),
                   "--annotation-threshold", "0.5"]
        attempt = {"subject": subject, "development_role": release["roles"][subject],
                   "source_manifest_sha256": digest(source), "argv": command,
                   "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
        batch["attempts"].append(attempt)
        write(batch_path, batch)
        try:
            with log.open("x") as handle:
                result = subprocess.run(command, cwd=snapshot, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False)
                handle.flush()
                os.fsync(handle.fileno())
            attempt.update(exit_code=result.returncode, log_sha256=digest(log))
            if result.returncode:
                raise RuntimeError(f"Preparation process exited {result.returncode}")
            prepared = json.loads(report.read_text())
            if (not prepared["reopened_identical"] or not prepared["source_files_unchanged"]
                    or prepared["benchmark_role"] != release["roles"][subject]
                    or prepared["brain_mask_supplied_or_derived"] is not False
                    or prepared["target_policy_input_allowed"] is not False
                    or prepared["automatic_cortical_access_status"] != "blocked_without_reviewed_cerebral_mask"):
                raise ValueError("Prepared report weakened identity, role, or review gates")
            attempt.update(status="completed", report_sha256=digest(report),
                           runtime_sha256=digest(runtime), bundle_sha256=digest(output),
                           bundle_bytes=output.stat().st_size, case_hash=prepared["case_hash"],
                           planning_hash=prepared["planning_hash"],
                           peak_rss_bytes_macos=json.loads(runtime.read_text())["peak_rss_bytes_macos"])
        except Exception as error:
            attempt["status"] = "failed"
            batch["failures"].append({"subject": subject, "type": type(error).__name__, "message": str(error)})
        attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(batch_path, batch)
        if batch["failures"]:
            break
    batch["source_file_sha256_after"] = {p: digest(data / p) for p in release["source_file_sha256_before"]}
    batch["source_files_unchanged"] = batch["source_file_sha256_after"] == release["source_file_sha256_before"]
    batch["frozen_source_unchanged"] = all(digest(snapshot / p) == h for p, h in release["snapshot_files_sha256"].items())
    batch["unopened_PAT29_PAT31_absent"] = all(
        not (data / subject).exists() and not (data / f"derivatives/tumor_masks/{subject}").exists()
        for subject in release["forbidden_subjects"])
    complete = (len(batch["attempts"]) == 4 and not batch["failures"] and batch["source_files_unchanged"]
                and batch["frozen_source_unchanged"] and batch["unopened_PAT29_PAT31_absent"])
    batch.update(status="completed" if complete else "failed", elapsed_seconds=time.perf_counter() - started,
                 finished_at=datetime.now(timezone.utc).isoformat())
    write(batch_path, batch)
    print(json.dumps({"status": batch["status"], "attempts": len(batch["attempts"]),
                      "elapsed_seconds": batch["elapsed_seconds"], "failures": batch["failures"],
                      "batch_sha256": digest(batch_path)}))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
