"""Run released frozen-source regression; retain all results without repair."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def process_family_rss(root_pid):
    """Sample descendants plus the dedicated process group; RSS is summed."""
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,rss="], text=True)
    rows = [tuple(map(int, row.split())) for row in output.splitlines() if row.strip()]
    selected = {root_pid} | {pid for pid, _, group, _ in rows if group == root_pid}
    while True:
        expanded = selected | {pid for pid, parent, _, _ in rows if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    return sum(rss * 1024 for pid, _, _, rss in rows if pid in selected)


def capped_run(command, *, cwd, env, handle, wall_seconds, memory_bytes, interval=0.5):
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    peak = 0
    samples = 0
    reason = None
    try:
        while process.poll() is None:
            rss = process_family_rss(process.pid)
            samples += 1
            peak = max(peak, rss)
            if rss > memory_bytes:
                reason = "sampled_process_family_memory_cap"
            elif time.monotonic() - started > wall_seconds:
                reason = "wall_time_cap"
            if reason is not None:
                break
            time.sleep(interval)
    finally:
        if reason is not None or process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
    return {"exit_code": process.returncode, "cap_failure": reason,
            "sampled_peak_process_family_rss_bytes": peak, "resource_samples": samples,
            "sample_interval_seconds": interval, "wall_cap_seconds": wall_seconds,
            "memory_cap_bytes": memory_bytes}


BOOTSTRAP = r'''
import hashlib,importlib.metadata,json,os,pathlib,resource,sys,time
source=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2]);mode=sys.argv[3]
app_dependencies=sys.argv[4];idc_dependencies=sys.argv[5]
sys.path[:0]=[str(source/'src'),str(source)]
sys.path.extend(([idc_dependencies] if mode=='dicom' else [])+[app_dependencies])
import pytest
started=time.perf_counter()
progress={'collected':None,'finished':0,'failed':[],'skipped':[]}
def write(name,value):
    path=output/(mode+'-'+name+'.json');temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
class Receipt:
    def pytest_collection_finish(self,session):
        progress['collected']=len(session.items);write('progress',progress)
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed or report.skipped:
            if report.when=='call':progress['finished']+=1
            if report.failed:progress['failed'].append({'nodeid':report.nodeid,'when':report.when,'failure':str(report.longrepr)})
            if report.skipped:progress['skipped'].append({'nodeid':report.nodeid,'when':report.when,'reason':str(report.longrepr)})
            if report.failed or report.skipped or progress['finished']%20==0:write('progress',progress)
paths=['tests'] if mode=='main' else ['tests/test_remind_development_acquisition.py']
args=paths+['-q','-ra','--durations=0','--durations-min=0','--junitxml='+str(output/(mode+'.xml')),'-p','no:cacheprovider']
exit_code=pytest.main(args,plugins=[Receipt()])
write('progress',progress)
origins={name:str(pathlib.Path(module.__file__).resolve()) for name,module in sys.modules.items()
         if getattr(module,'__file__',None) and (name.startswith('resectionlab') or pathlib.Path(module.__file__).resolve().is_relative_to(source))}
bad={name:path for name,path in origins.items() if name.startswith('resectionlab') and not pathlib.Path(path).is_relative_to(source/'src')}
versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=([idc_dependencies] if mode=='dicom' else [])+[app_dependencies])}
write('runtime',{'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'python':sys.version,'sys_path':sys.path,'pytest_arguments':args,'package_versions':versions,
    'project_module_origins':origins,'unexpected_resectionlab_origins':bad,'pytest_exit_code':int(exit_code)})
if bad:raise RuntimeError('Mutable checkout imported: '+repr(bad))
raise SystemExit(int(exit_code))
'''


def main():
    directory = Path(__file__).resolve().parent
    release_path = directory / "root-release.json"
    if digest(release_path) != "c421face3319b214d7ab582a0aa0852eeda218a99c4a600055104e1a40d00dd4":
        raise ValueError("Released baseline changed")
    release = json.loads(release_path.read_text())
    source = Path(release["snapshot"])
    record_path = directory / "execution-record.json"
    if record_path.exists():
        raise ValueError("Existing attempt must be inspected, never repeated implicitly")
    for relative, expected in release["baseline_files"].items():
        if digest(source / relative) != expected["sha256"]:
            raise ValueError("Baseline changed: " + relative)
    record = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "git_commit": release["git_commit"],
              "root_release_sha256": digest(release_path), "executor_sha256": digest(Path(__file__)),
              "runs": [], "no_installs": True, "numerical_threads_requested": 1}
    save(record_path, record)
    env = dict(os.environ)
    env.update(QT_QPA_PLATFORM="offscreen", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=os.pathsep.join([str(source / "src"), str(source)]),
               GIT_CEILING_DIRECTORIES=str(source.parent),
               RESECTIONLAB_TEST_CASE_BUNDLE=release["public_bundle"])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        env[key] = "1"
    for mode in ("main", "dicom"):
        interpreter = release["main_interpreter"] if mode == "main" else release["idc_interpreter"]
        command = [interpreter, "-I", "-S", "-B", "-c", BOOTSTRAP, str(source), str(directory), mode,
                   release["main_site_packages"], release["idc_site_packages"]]
        row = {"mode": mode, "argv": command, "started_at": datetime.now(timezone.utc).isoformat()}
        record["runs"].append(row)
        save(record_path, record)
        started = time.perf_counter()
        log = directory / f"{mode}.log"
        with log.open("x") as handle:
            outcome = capped_run(command, cwd=source, env=env, handle=handle,
                                 wall_seconds=release["caps"][mode + "_wall_seconds"],
                                 memory_bytes=release["caps"]["process_family_rss_bytes"])
            handle.flush()
            os.fsync(handle.fileno())
        row.update(**outcome, elapsed_seconds=time.perf_counter() - started,
                   finished_at=datetime.now(timezone.utc).isoformat(), log_sha256=digest(log))
        for suffix in (".xml", "-runtime.json", "-progress.json"):
            path = directory / (mode + suffix)
            row[path.name + "_sha256"] = digest(path) if path.exists() else None
        save(record_path, record)
        print(json.dumps({"mode": mode, "exit_code": row["exit_code"], "elapsed_seconds": row["elapsed_seconds"],
                          "cap_failure": row["cap_failure"]}), flush=True)
    current = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    before = {p: value["sha256"] for p, value in release["baseline_files"].items()}
    record["changed_baseline_files"] = [p for p, value in before.items() if current.get(p) != value]
    record["added_files"] = sorted(set(current) - set(before))
    record["public_bundle_unchanged"] = digest(Path(release["public_bundle"])) == release["public_bundle_sha256"]
    record["status"] = "passed" if all(x["exit_code"] == 0 and x["cap_failure"] is None for x in record["runs"]) and not record["changed_baseline_files"] and not record["added_files"] and record["public_bundle_unchanged"] else "failed"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(record_path, record)
    print(json.dumps({"status": record["status"], "receipt_sha256": digest(record_path),
                      "changed_baseline_files": record["changed_baseline_files"], "added_files": len(record["added_files"])}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
