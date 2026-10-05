"""One released, bounded acquisition batch using an immutable Git snapshot."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    release_path = directory / "root-release.json"
    release = json.loads(release_path.read_text())
    snapshot = root / release["source_snapshot"]
    data = Path(release["output_root"])
    record_path = directory / "batch-record.json"
    if record_path.exists():
        raise RuntimeError("Batch history exists: do not retry without a new explicit release")
    expected_subjects = ["sub-PAT22", "sub-PAT25", "sub-PAT26", "sub-PAT27"]
    if release["allowed_subjects"] != expected_subjects:
        raise RuntimeError("Released subject scope differs")
    for relative, expected in release["frozen_source_files"].items():
        if digest(snapshot / relative) != expected:
            raise RuntimeError("Frozen source differs before execution")
    bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
    started = time.perf_counter()
    record = {"schema_version": 1, "status": "running", "started_at": now(),
              "root_release_sha256": digest(release_path), "executor_sha256": digest(Path(__file__)),
              "git_commit": release["git_commit"], "attempts": [], "failures": [],
              "expected_new_bytes": release["new_subject_bytes_maximum"],
              "new_downloaded_bytes": 0, "preparation_or_training_executed": False,
              "brain_reviewed": False, "cortical_access_permitted": False}

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        for subject in expected_subjects:
            command = [release["interpreter"], "-I", "-S", "-c", bootstrap, str(snapshot / "scripts"),
                       str(snapshot / "scripts/acquire_btc_case.py"), "--spatial-subject", subject,
                       "--output-root", str(data)]
            attempt = {"subject": subject, "started_at": now(), "argv": command, "status": "running"}
            record["attempts"].append(attempt)
            save()
            print(f"Starting released structural acquisition: {subject}", flush=True)
            log_path = directory / f"{subject}.log"
            with log_path.open("x") as log:
                outcome = subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, check=False)
            attempt.update(finished_at=now(), exit_code=outcome.returncode, log_sha256=digest(log_path))
            if outcome.returncode:
                attempt["status"] = "failed"
                raise RuntimeError(f"{subject} acquisition failed; original log and partial files retained")
            manifest_path = data / f"btc_{subject}_verified_manifest.json"
            report_path = data / f"btc_{subject}_acquisition_report.json"
            manifest = json.loads(manifest_path.read_text())
            report = json.loads(report_path.read_text())
            if manifest["selection_cohort_sha256"] != release["frozen_source_files"]["manifests/experiments/btc-spatial-development-cohort-v1.json"]:
                raise RuntimeError("Completed source manifest lost frozen cohort identity")
            statuses = {item["path"]: item["status"] for item in report["files"]}
            verified = []
            for entry in manifest["files"]:
                path = data / entry["path"]
                if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                    raise RuntimeError("Independent post-download SHA256/length check failed")
                md5 = None
                if entry.get("expected_md5"):
                    with path.open("rb") as handle:
                        md5 = hashlib.file_digest(handle, "md5").hexdigest()
                    if md5 != entry["expected_md5"]:
                        raise RuntimeError("Independent source-annex MD5 check failed")
                verified.append({"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"],
                                 "annex_md5": md5, "status": statuses[entry["path"]]})
                if statuses[entry["path"]] == "downloaded_and_verified":
                    record["new_downloaded_bytes"] += entry["bytes"]
            for source, suffix in ((manifest_path, "source-manifest.json"), (report_path, "acquisition-report.json")):
                (directory / f"{subject}-{suffix}").write_bytes(source.read_bytes())
            attempt.update(status="completed", source_manifest_sha256=digest(manifest_path),
                           acquisition_report_sha256=digest(report_path), files=verified,
                           development_role=manifest["development_role"])
            save()
            print(f"Completed and independently rehashed: {subject}", flush=True)
        if record["new_downloaded_bytes"] != release["new_subject_bytes_maximum"]:
            raise RuntimeError("Actual new bytes differ from the released clean acquisition forecast")
        for entry in release["existing_shared_metadata"]:
            path = data / entry["path"]
            if digest(path) != entry["sha256"] or path.stat().st_mtime_ns != entry["mtime_ns"]:
                raise RuntimeError("Previously verified shared metadata changed")
        declaration = json.loads((snapshot / "manifests/experiments/btc-spatial-development-cohort-v1.json").read_text())
        for candidate in declaration["candidates"]:
            if candidate["subject"] in release["forbidden_subjects"]:
                if any((data / item["path"]).exists() for item in candidate["files"]):
                    raise RuntimeError("An unopened transfer candidate now has local source files")
        record["unopened_transfer_files_absent_after_batch"] = True
        record["existing_shared_metadata_bytes_and_mtimes_unchanged"] = True
        record["status"] = "completed"
    except Exception as error:
        record["status"] = "failed"
        record["failures"].append({"type": type(error).__name__, "message": str(error), "recorded_at": now()})
    finally:
        record["frozen_source_after_sha256"] = {p: digest(snapshot / p) for p in release["frozen_source_files"]}
        record["frozen_source_unchanged"] = record["frozen_source_after_sha256"] == release["frozen_source_files"]
        record["finished_at"] = now()
        record["elapsed_seconds"] = time.perf_counter() - started
        save()
    print(json.dumps({"status": record["status"], "new_downloaded_bytes": record["new_downloaded_bytes"],
                      "elapsed_seconds": record["elapsed_seconds"], "failures": record["failures"]}), flush=True)
    return 0 if record["status"] == "completed" and record["frozen_source_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
