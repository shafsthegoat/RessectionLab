"""Independent control-launcher checks; mocks only, no solver or patient data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "accelerate_controls_independent", ROOT / "scripts/mechanics_accelerate_controls.py"
)
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def test_external_archive_refused_before_runtime_work(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    archive = tmp_path / "outside-archive"
    archive.mkdir()
    release = tmp_path / "release.json"
    release.write_text(json.dumps({
        "schema": v.VERSION + "-release", "authorized": True,
        "root_release": "Constructed review fixture; no execution authorization",
        "attempt_directory": str(repository / "new-attempt"),
        "source_commit": "a" * 40, "source_directory": str(archive),
        "repository_directory": str(repository),
        "runtime_identity": {"path": "identity.json", "sha256": "b" * 64},
    }))
    calls = []
    monkeypatch.setattr(v, "ROOT", archive)
    monkeypatch.setattr(v, "CLOSURE", ())
    monkeypatch.setattr(v, "declaration", lambda: {})
    def runtime(*args):
        calls.append(args)
        return {"inputs": {}, "runtime": {"executable": "never-executed"}}
    monkeypatch.setattr(v, "runtime_binding", runtime)
    with pytest.raises(ValueError, match="[Aa]rchive|[Ss]ource|repository"):
        v.baseline(release)
    assert not calls


@pytest.mark.parametrize("changed_group", ["hex8", "tet10_mpc"])
def test_failed_summary_reports_changed_checker_truthfully(tmp_path, monkeypatch, changed_group):
    archive = tmp_path / "archive"
    checks = {}
    for relative in v.CHECKER_PINS:
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
        checks[str(path)] = v.sha(path)
    changed = "mechanics_febio_verification.py" if changed_group == "hex8" else "mechanics_patient_constraints.py"
    with (archive / "scripts" / changed).open("ab") as stream:
        stream.write(b"\n# deliberately changed temporary checker\n")
    output = tmp_path / "attempt"
    output.mkdir()
    base = {"source_commit": "a" * 40, "runtime_identity_sha256": "b" * 64,
            "release": {"repository_directory": str(tmp_path)}, "input_hashes": checks}
    result = v.initial_result()
    result.update(status="failed_or_incomplete", inputs_after=v.unchanged(checks))
    for name, value in (("execution.json", {"status": "failed_or_incomplete"}),
                        ("results.json", result), ("execution-baseline.json", base)):
        (output / name).write_text(json.dumps(value))
    monkeypatch.setattr(v, "ROOT", archive)
    v.summaries(output, base, result, False)
    summary = json.loads((output / (changed_group + "-summary.json")).read_text())
    assert summary["status"] == "failed_or_incomplete"
    assert summary["original_numerical_checker_unchanged"] is False


def test_fixed_decks_preserve_complete_xml_except_explicit_solver():
    """Independent XML comparison complements the runner's literal reversal."""
    expected = {"iterative": "0", "factorization": "4", "order_method": "0",
                "print_condition_number": "0"}
    v.original_bindings()
    for name in v.CASES:
        original = ET.fromstring((ROOT / v.ORIGINAL[name]).read_bytes())
        adapted = ET.fromstring((ROOT / v.BUNDLE / "decks" / (name + ".feb")).read_bytes())
        old = original.findall(".//linear_solver")
        new = adapted.findall(".//linear_solver")
        assert len(old) == len(new) == 1
        assert old[0].attrib == {"type": "skyline"}
        assert new[0].attrib == {"type": "accelerate"}
        assert {child.tag: child.text for child in new[0]} == expected
        assert len(new[0]) == len(expected)
        new[0].attrib = old[0].attrib.copy()
        new[0].text = old[0].text
        for child in list(new[0]):
            new[0].remove(child)
        assert ET.tostring(original) == ET.tostring(adapted)
        assert adapted.findtext("Control/time_steps") == "4"
        assert float(adapted.findtext("Control/step_size")) == .25


def prepared_worker(tmp_path, monkeypatch, *, behavior="success"):
    """Only real analytical deck bytes; solver/checker contents are fabricated."""
    output = tmp_path / "attempt"
    output.mkdir()
    fixed = tmp_path / "fixed-input"
    fixed.write_text("constructed input")
    base = {"release_path": "constructed-release", "attempt_directory": str(output),
            "input_hashes": {str(fixed): v.sha(fixed)},
            "runtime_identity_sha256": "a" * 64, "executable": "NEVER_EXECUTE_SOLVER"}
    v.write(output / "results.json", v.initial_result())
    v.write(output / "execution-baseline.json", base)
    monkeypatch.setattr(v, "baseline", lambda _: base)
    state = {"clock": 0., "solver": [], "checker": [], "scaling": 0}
    monkeypatch.setattr(v.time, "monotonic", lambda: state["clock"])
    def solver(command, **kwargs):
        name = Path(command[-1]).stem
        state["solver"].append((name, kwargs["timeout"]))
        assert command[0] == "NEVER_EXECUTE_SOLVER"
        assert kwargs["stderr"] == v.subprocess.STDOUT
        assert "shell" not in kwargs
        if behavior == "timeout":
            state["clock"] = 60.
            raise v.subprocess.TimeoutExpired(command, kwargs["timeout"])
        kwargs["stdout"].write("Selecting linear solver accelerate\n")
        for suffix in ("nodes.log", "elements.log", "log"):
            (kwargs["cwd"] / (name + "." + suffix)).write_text("MOCK PRIMITIVE ONLY")
        state["clock"] += 1.
        if behavior == "exhaust_after_first":
            state["clock"] = 60.
        if behavior == "input_changes":
            fixed.write_text("changed constructed input")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(v.subprocess, "run", solver)
    def checker_module(path, _):
        kind = "hex" if Path(path).name == "mechanics_febio_verification.py" else "tet"
        def check(name, *raw):
            state["checker"].append((kind, name))
            assert raw == ("MOCK PRIMITIVE ONLY",) * 3
            return {"case": name, "passed": behavior != "primitive_reject"}
        def scaling(*_):
            state["scaling"] += 1
            assert len(state["solver"]) == 5
            return {"passed": True}
        return SimpleNamespace(__file__=str(path), np=SimpleNamespace(__version__="mock"),
                               read_bounded_text=lambda p: Path(p).read_text(),
                               check_outputs=check, check_stiffness_scaling=scaling)
    monkeypatch.setattr(v, "load", checker_module)
    return output, base, state


def test_eight_calls_use_remaining_aggregate_budget_and_exact_checker_groups(tmp_path, monkeypatch):
    output, base, state = prepared_worker(tmp_path, monkeypatch)
    assert v.worker(output) == 0
    assert state["solver"] == list(zip(v.CASES, range(60, 52, -1)))
    assert state["checker"] == [("hex", n) for n in v.HEX_CASES] + [("tet", n) for n in v.TET_CASES]
    assert state["scaling"] == 1
    result = json.loads((output / "results.json").read_text())
    assert result["solver_invocations"] == 8 and result["worker_seconds"] == 8.
    assert v.accept({"status": "completed", "exit_code": 0}, result, v.unchanged(base["input_hashes"]))


@pytest.mark.parametrize("behavior,expected_checker_count", [
    ("timeout", 0), ("exhaust_after_first", 1), ("primitive_reject", 1)
])
def test_worker_stops_without_next_call_or_retry(tmp_path, monkeypatch, behavior, expected_checker_count):
    output, base, state = prepared_worker(tmp_path, monkeypatch, behavior=behavior)
    assert v.worker(output) == 1
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "failed_or_incomplete"
    assert result["solver_invocations"] == len(state["solver"]) == 1
    assert len(state["checker"]) == expected_checker_count
    assert state["scaling"] == 0
    if behavior == "exhaust_after_first":
        assert result["cases"][0]["status"] == "passed"
        assert result["first_failure"] == "translation"
        assert not (output / "translation").exists()
        assert all(row["status"] == "not_executed" for row in result["cases"][2:])
    else:
        assert all(row["status"] == "not_executed" for row in result["cases"][1:])
    assert not v.accept({"status": "completed", "exit_code": 0}, result, v.unchanged(base["input_hashes"]))


def test_end_integrity_change_cannot_publish_accepted_result(tmp_path, monkeypatch):
    output, base, _ = prepared_worker(tmp_path, monkeypatch, behavior="input_changes")
    assert v.worker(output) == 1
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "failed_or_incomplete"
    assert not v.accept({"status": "completed", "exit_code": 0}, result, v.unchanged(base["input_hashes"]))


def test_existing_attempt_refuses_before_any_setup_and_preserves_bytes(tmp_path, monkeypatch):
    output = tmp_path / "attempt"
    output.mkdir()
    prior = output / "results.json"
    prior.write_bytes(b"prior failure must remain untouched")
    def forbidden(*_):
        pytest.fail("existing attempt must fail before setup/runtime")
    monkeypatch.setattr(v, "baseline", forbidden)
    with pytest.raises(FileExistsError):
        v.launch("unused", output)
    assert prior.read_bytes() == b"prior failure must remain untouched"


def test_publication_failure_invalidates_preceding_summary_parent_binding(tmp_path, monkeypatch):
    output = tmp_path / "new-attempt"
    fixed = tmp_path / "fixed-input"
    fixed.write_text("constructed")
    base = {"attempt_directory": str(output), "input_hashes": {str(fixed): v.sha(fixed)},
            "runtime_identity_sha256": "a" * 64,
            "release": {"repository_directory": str(tmp_path)}}
    monkeypatch.setattr(v, "baseline", lambda _: base)
    def supervise(*_, **kwargs):
        assert kwargs["seconds"] == 60 and kwargs["rss_bytes"] == 3 * 1024**3
        assert kwargs["environment"] == {"OMP_NUM_THREADS": "1"}
        v.write(output / "results.json", {"runtime_identity_sha256": "a" * 64})
        return {"status": "completed", "exit_code": 0}
    runtime = SimpleNamespace(
        declaration=lambda: {"caps": {"thread_environment": {"OMP_NUM_THREADS": "1"}}},
        private_environment=lambda _: {"OMP_NUM_THREADS": "1"}, supervise=supervise)
    monkeypatch.setattr(v, "load", lambda *_: runtime)
    monkeypatch.setattr(v, "accept", lambda *_: True)  # Isolates publication, never numerical success.
    def summaries(*args):
        assert args[3] is True
        v.write(output / "hex8-summary.json", {"execution_sha256": v.sha(output / "execution.json")})
        raise OSError("constructed second-summary publication failure")
    monkeypatch.setattr(v, "summaries", summaries)
    assert v.launch("unused", output) == 1
    receipt = json.loads((output / "execution.json").read_text())
    partial = json.loads((output / "hex8-summary.json").read_text())
    assert receipt["status"] == "failed_or_incomplete"
    assert "publication failure" in receipt["summary_error"]
    assert partial["execution_sha256"] != v.sha(output / "execution.json")
