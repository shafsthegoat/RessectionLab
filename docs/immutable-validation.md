# Full-suite validation of committed source

The previous full-suite receipt is
`artifacts/validation/integrated-axis-v1/attempt-02/receipt.json`: commit
`558b2e32693a002acc1afbb29a40649bc5296423`, 959 tests passed, 2,669 baseline files
unchanged. `artifacts/validation/native-axis-public-v2/receipt.json` is a later
77-test focused check at `69304172a74c50a8b83025c863edd979ff26e4fd`.
Neither receipt certifies later code. Their archive launchers were ad hoc;
the procedure below makes the required steps explicit.

Commit all intended source, tests and declarations first. Pause other timed
training, imaging and test workloads before execution. This uses the existing
`.venv` and installed optional research dependencies; it is not a clean install
test. The optional legacy Qt worker tests skip if Qt is absent, and diffusion
tests skip if their optional dependencies are absent. Report the actual skip
count instead of assuming the previous test total.

The only currently required ignored input found in the full test tree is
`outputs/cases/UCSF-PDGM-0004.ressectionlab`, SHA-256
`11cca797ad0216e5c8d940e7354f2dd23fb6954dc2aed7e4b0423f6adc6cefcf`.
The procedural configuration test reads the declaration's relative path;
the native-axis public configuration test also accepts
`RESECTIONLAB_TEST_CASE_BUNDLE`. Supply both with the copied, verified archive
fixture. Do not symlink the live `outputs/`, `data/`, source tree or `.venv` into
the archive. Raw BTC imaging, SynthStrip weights and historical checkpoints are
not required by the currently inspected test fixtures. Recheck this requirement
if new tests add inputs.

From the repository root, choose a fresh attempt name and run the following
only when full-suite execution is intended. It captures the resolved commit,
archive SHA-256, actual dependency versions, environment overrides, case bytes,
before/after source manifests, pytest output and JUnit results. Existing attempt
directories are refused. An interrupted or failed suite retains its evidence;
never reuse its output directory to hide a failure.

```sh
validation_revision="$(git rev-parse HEAD)"
validation_attempt=full-suite-final-01
.venv/bin/python - "$validation_revision" "$validation_attempt" <<'PY'
import hashlib, importlib.metadata, json, os, platform, re, shutil
import subprocess, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd().resolve()
revision, attempt = sys.argv[1:]
assert re.fullmatch(r"[0-9a-f]{40}", revision), "Use a resolved commit SHA"
assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", attempt), "Use a plain attempt name"
assert subprocess.check_output(["git", "rev-parse", revision + "^{commit}"], cwd=root, text=True).strip() == revision
run = root / "build/validation" / attempt / revision
run.mkdir(parents=True, exist_ok=False)
source = run / "source"
source.mkdir()

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write(name, value):
    (run / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

def manifest():
    result = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            result[str(path.relative_to(source))] = {"symlink": os.readlink(path)}
        elif path.is_file():
            result[str(path.relative_to(source))] = {
                "sha256": sha(path), "mode": path.stat().st_mode & 0o777}
    return result

archive = run / "source.tar"
subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), revision], cwd=root, check=True)
with tarfile.open(archive) as stream:
    assert all(item.isfile() or item.isdir() for item in stream.getmembers()), "Unexpected archive link or special file"
    stream.extractall(source, filter="data")
declaration = json.loads((source / "manifests/experiments/procedural-native-to-ucsf-v1.json").read_text())
relative_case = "outputs/cases/UCSF-PDGM-0004.ressectionlab"
expected = "11cca797ad0216e5c8d940e7354f2dd23fb6954dc2aed7e4b0423f6adc6cefcf"
assert declaration["target"]["bundle_path"] == relative_case
assert declaration["target"]["bundle_sha256"] == expected
case = source / relative_case
case.parent.mkdir(parents=True)
assert sha(root / relative_case) == expected, "Local case differs from declared bytes"
shutil.copyfile(root / relative_case, case)
assert sha(case) == expected, "Case changed during copy"
before = manifest()
write("before.json", before)

overrides = {"PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_ADDOPTS": "", "PYTEST_PLUGINS": "", "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "QT_QPA_PLATFORM": "offscreen",
    "RESECTIONLAB_TEST_CASE_BUNDLE": str(case)}
env = dict(os.environ)
env.pop("PYTHONHOME", None)
env.update(overrides)
argv = [sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider",
    "--basetemp", str(run / "pytest-temp"), "--junitxml", str(run / "pytest-junit.xml")]
receipt = {"schema_version": 1, "status": "prepared", "revision": revision,
    "archive_sha256": sha(archive), "source_path": str(source),
    "source_file_count": len(before), "source_manifest_sha256": sha(run / "before.json"),
    "case_sha256": sha(case), "argv": argv, "environment_overrides": overrides,
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "load_average_at_start": os.getloadavg(),
    "dependencies": sorted((item.metadata["Name"], item.version) for item in importlib.metadata.distributions()),
    "started_utc": datetime.now(timezone.utc).isoformat()}
write("receipt.json", receipt)
print(f"Running {revision}; durable output: {run}", flush=True)
started = time.monotonic()
try:
    with (run / "pytest.log").open("w") as log:
        receipt["exit_code"] = subprocess.run(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    receipt["status"] = "completed" if receipt["exit_code"] == 0 else "failed"
except BaseException as error:
    receipt.update(status="interrupted_or_launcher_failed", error=repr(error))
    raise
finally:
    after = manifest()
    write("after.json", after)
    receipt.update(elapsed_seconds=time.monotonic() - started,
        source_changed=[name for name, value in before.items() if after.get(name) != value],
        source_added=sorted(set(after) - set(before)),
        after_manifest_sha256=sha(run / "after.json"),
        log_sha256=sha(run / "pytest.log"), finished_utc=datetime.now(timezone.utc).isoformat())
    if (run / "pytest-junit.xml").is_file():
        receipt["junit_sha256"] = sha(run / "pytest-junit.xml")
    if receipt["source_changed"]:
        receipt["status"] = "source_changed"
    write("receipt.json", receipt)
print(json.dumps({key: receipt.get(key) for key in ("status", "exit_code", "source_changed", "source_added")}, indent=2))
raise SystemExit(0 if receipt["status"] == "completed" else 1)
PY
```

Read the actual test summary and JUnit skip/failure details. A successful process
exit with nonempty `source_changed` is not an immutable pass. Review any
`source_added` files separately; they are recorded even when all baseline bytes
are unchanged. The suite can exercise real updates on its small synthetic
fixtures, but this procedure does not run the public-patient pilot command.

Keep `build/validation/<attempt>/<commit>/` locally, including `source.tar`, the
copied case and `pytest-temp/`. Version only the new compact receipt, before/after
manifests and complete output under a fresh `artifacts/validation/` namespace;
do not publish ignored imaging or checkpoints. Historical failed receipts and
manifests remain unchanged. This procedure was prepared by static inspection;
no suite run is claimed by this document.
