"""Offline launcher controls: inert metadata/bytes and simulated processes only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

FILE = Path(__file__).resolve().parents[1] / "artifacts/rhuh-single-image-inspection-v1/run_once.py"
SPEC = importlib.util.spec_from_file_location("rhuh_inspection_launch_owner", FILE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def bind(root, path, content):
    raw = content if isinstance(content, bytes) else json.dumps(content).encode()
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def fixture(tmp_path):
    sources = {key: bind(tmp_path, path, b"# inert software source") for key, path in mod.SOURCE_PATHS.items()}
    original = bind(tmp_path, "outputs/rhuh-single-image-v2/quarantine/run-fixture/RHUH-0001_0_t1.nii.gz", b"inert non-image")
    request = {"schema": "resectionlab.rhuh-image-inspection-request.v1", "source": mod.SOURCE,
               "payload": {**original, "measured_compressed_bytes": len(b"inert non-image")},
               "reconciliation": {"path": "artifacts/inert-reconciliation.json", "sha256": "a" * 64}}
    request_binding = bind(tmp_path, "artifacts/inert-request.json", request)
    release = {"schema": "resectionlab.rhuh-image-inspection-release.v1", "released": True,
               "action": "bounded_header_and_voxel_inspection_once", "request": request_binding,
               "inspector_source_sha256": sources["inspector"]["sha256"], "source": mod.SOURCE}
    release_binding = bind(tmp_path, "artifacts/inert-release.json", release)
    plan = {"schema": "resectionlab.rhuh-image-inspection-launch.v1", "released": True,
            "source": mod.SOURCE, "request": request_binding, "release": release_binding,
            **sources, "output_directory": "outputs/rhuh-single-image-inspection-v1/run-fixture"}
    manifest = bind(tmp_path, "artifacts/inert-launch.json", plan)
    result = {"schema": "resectionlab.rhuh-single-image-inspection.v1", "status": mod.SUCCESS,
              "binary_structure_valid": True, "scientific_use_released": False, "clinical_validation": False,
              "bindings": {"request": request_binding, "release": release_binding,
                           "reconciliation": request["reconciliation"], "source": mod.SOURCE},
              "compressed_sha256": original["sha256"], "compressed_bytes": len(b"inert non-image"),
              "candidate_compressed_md5": mod.TOKEN}
    return tmp_path, plan, request, manifest, result


def read(fixture):
    root, plan, _, manifest, _ = fixture
    return mod.read_plan(root, manifest, plan["request"]["sha256"], plan["release"]["sha256"])


@pytest.fixture
def process_control(fixture, monkeypatch):
    state = {"now": 10.0, "launches": 0, "kills": [], "rss": 1000, "exit": 0, "alive": False}
    root, plan, request, manifest, result = fixture

    class Child:
        pid = 456789
        returncode = None

        def poll(self):
            if not state["alive"]:
                self.returncode = state["exit"]
            return self.returncode

        def wait(self, timeout):
            self.returncode = state["exit"]
            return self.returncode

    def popen(argv, **kwargs):
        state["launches"] += 1
        state["argv"], state["kwargs"] = argv, kwargs
        raw = state.get("raw", json.dumps(result).encode())
        kwargs["stdout"].write(raw)
        return Child()

    def kill(pid, sig):
        state["kills"].append((pid, sig))
        if not state.get("survivor"):
            raise ProcessLookupError()

    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(mod.os, "killpg", kill)
    monkeypatch.setattr(mod.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: state.update(now=state["now"] + seconds))
    monkeypatch.setattr(mod, "sample_rss", lambda pid: state["rss"])
    return state


def run(fixture):
    root, plan, _, manifest, _ = fixture
    return mod.run(root, manifest, plan["request"]["sha256"], plan["release"]["sha256"], 10.0, [])


def test_metadata_read_does_not_open_payload(fixture):
    root, plan, request, _, _ = fixture
    (root / request["payload"]["path"]).unlink()
    assert read(fixture) == (plan, request)


@pytest.mark.parametrize("key", ["request", "release", "inspector", "launcher", "public_transfer"])
def test_changed_bound_inputs_rejected(fixture, key):
    root, plan, _, _, _ = fixture
    (root / plan[key]["path"]).write_bytes(b"changed")
    with pytest.raises(mod.Rejected, match="hash_changed"):
        read(fixture)


def test_explicit_hash_mismatch_before_spawn(fixture, monkeypatch):
    root, plan, _, manifest, _ = fixture
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: pytest.fail("no child"))
    with pytest.raises(mod.Rejected, match="explicit"):
        mod.read_plan(root, manifest, "0" * 64, plan["release"]["sha256"])


def test_success_checks_original_bytes_and_process_exit(fixture, process_control, capsys):
    assert run(fixture) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] and result["original"]["parent_decoded_image"] is False
    assert result["launcher_exit_zero_required"] is True
    assert process_control["launches"] == 1 and len(process_control["kills"]) == 1
    kwargs = process_control["kwargs"]
    assert kwargs["start_new_session"] is True and kwargs["preexec_fn"] is mod.child_limits
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert "--execute" in process_control["argv"]


def test_attempt_directory_prevents_repeat(fixture, process_control):
    assert run(fixture) == 0
    assert run(fixture) == 1
    assert process_control["launches"] == 1


@pytest.mark.parametrize("change,status", [
    ({"exit": 2}, "inspector_exit_nonzero"),
    ({"rss": mod.RSS_CAP + 1}, "sampled_rss_cap"),
    ({"survivor": True}, "surviving_group_unaccepted"),
    ({"raw": b"{"}, "reconciliation_failed_JSONDecodeError"),
    ({"raw": b"x" * (mod.LOG_CAP + 1)}, "not_bounded_regular_file"),
    ({"alive": True}, "inspection_timeout"),
])
def test_failed_run_consumes_attempt_and_cleans_group(fixture, process_control, capsys, change, status):
    process_control.update(change)
    assert run(fixture) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] is False and result["status"] == status
    assert len(process_control["kills"]) == 1
    assert (fixture[0] / fixture[1]["output_directory"]).exists()


def test_original_changed_after_inspector_not_promoted(fixture, process_control, capsys):
    root, _, request, _, _ = fixture
    (root / request["payload"]["path"]).write_bytes(b"other nonimage")
    assert run(fixture) == 1
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_precheck_time_counts_against_launch_deadline(fixture, process_control, monkeypatch):
    original = mod.read_plan
    def delayed(*args):
        result = original(*args)
        process_control["now"] = 68.0
        return result
    monkeypatch.setattr(mod, "read_plan", delayed)
    assert run(fixture) == 1
    assert process_control["launches"] == 0


def test_final_read_time_counts_against_deadline(fixture, process_control, monkeypatch, capsys):
    original = mod.reconcile
    def delayed(*args):
        result = original(*args)
        process_control["now"] = 69.0
        return result
    monkeypatch.setattr(mod, "reconcile", delayed)
    assert run(fixture) == 1
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_shared_hard_watchdog_owns_full_action(fixture, monkeypatch):
    calls = []
    def watchdog(started, action, **kwargs):
        calls.append((started, kwargs))
        return 73
    monkeypatch.setattr(mod.shared, "hard_watchdog", watchdog)
    root, plan, _, manifest, _ = fixture
    assert mod.launch(root, manifest, plan["request"]["sha256"], plan["release"]["sha256"], 10) == 73
    assert calls == [(10, {"wall_seconds": 60})]


def test_final_publication_time_cannot_return_success(fixture, process_control, monkeypatch, capsys):
    original = open
    class DelayedWrite:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def write(self, raw):
            self.stream.write(raw)
            process_control["now"] = 69.0
        def __exit__(self, *args):
            self.stream.close()
    def slow_open(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return DelayedWrite(stream) if Path(path).name == "launch.json" else stream
    monkeypatch.setattr(mod, "open", slow_open, raising=False)
    assert run(fixture) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] is False and result["status"] == "deadline_during_publication"
    # The earlier provisional file requires exit0, which this run did not provide.
    receipt = json.loads((fixture[0] / fixture[1]["output_directory"] / "launch.json").read_bytes())
    assert receipt["launcher_exit_zero_required"] is True


def test_os_limits_and_rss_sum_are_explicit(monkeypatch):
    limits = []
    monkeypatch.setattr(mod.resource, "setrlimit", lambda key, value: limits.append((key, value)))
    mod.child_limits()
    assert (mod.resource.RLIMIT_FSIZE, (65536, 65536)) in limits
    class Result:
        stdout = f"{mod.os.getpid()} 123 10\n200 777 20\n201 777 30\n999 888 5000\n".encode()
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Result())
    assert mod.sample_rss(777) == 60 * 1024
