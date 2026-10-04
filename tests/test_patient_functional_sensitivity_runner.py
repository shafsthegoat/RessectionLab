"""Small analytical workflow checks; no training or public-case evaluation."""
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.functional_evidence import FunctionalEvidence, anatomy_identity
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import save_case
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionEngine, native_config_from_case
from resectionlab.worlds import content_hash


PATH = Path(__file__).parents[1] / "scripts/evaluate_patient_functional_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("patient_functional_sensitivity_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_static_selection_ignores_outcome_values_and_row_order():
    rows = [{"window_id": "window1", "route_id": "z", "return": 999},
            {"window_id": "window2", "route_id": "b", "return": 0},
            {"window_id": "window1", "route_id": "a", "return": -999}]
    declaration = {"selected_static_route_ids": ["a", "b"]}
    assert [r["route_id"] for r in RUNNER.select_static_routes(rows[::-1], declaration)] == ["a", "b"]
    with pytest.raises(ValueError, match="prospective"):
        RUNNER.select_static_routes(rows, {"selected_static_route_ids": ["z", "b"]})


def test_declared_input_byte_change_and_escape_are_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_bytes(b"source")
    declaration = {"inputs": {"case": {"path": "source.json", "sha256": sha256(b"source").hexdigest()}}}
    assert RUNNER.verify_inputs(tmp_path, declaration)["case"]["bytes"] == 6
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source bytes changed"):
        RUNNER.verify_inputs(tmp_path, declaration)
    declaration["inputs"]["case"]["path"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        RUNNER.verify_inputs(tmp_path, declaration)


@pytest.fixture
def workflow_fixture(tmp_path, monkeypatch):
    tissue = np.ones((5, 5, 4), bool)
    target = np.zeros(tissue.shape, bool)
    target[1:4, 1:4, 2:] = True
    case = CaseData("analytical-workflow-control", tissue.astype(np.float32), {"target": target}, np.eye(4),
        (SourceRef("analytical-control", "test:analytical-control", provenance="simulated"),), brain_mask=tissue)
    case_path = tmp_path / "original.ressectionlab"
    save_case(case, case_path)
    source = SourceRef("analytical-prior", "test:prior", "a" * 64, provenance="prior")

    def prior_evidence(actual_case, *, uncertainty):
        values = np.zeros(tissue.shape, np.float32)
        values[2, 2, 1] = .75
        return FunctionalEvidence("analytical-prior-control", values, values,
            np.ones(tissue.shape, bool), np.ones(tissue.shape, bool), np.eye(4),
            array_digest(actual_case.mri), anatomy_identity(actual_case),
            {"analytical": {"source": source.to_dict(), "mni_ras_to_patient_ras_mm": np.eye(4).tolist()}}, uncertainty)

    monkeypatch.setattr("resectionlab.functional_evidence.population_prior_sensitivity", prior_evidence)
    window = AccessWindow((2, 2, -.5), (0, 0, 1), 4, "analytical-window")
    native_paths = []
    for tool in NATIVE_GENERIC_TOOLS:
        config = native_config_from_case(case, access=window, tools=(tool,))
        engine = NativeResectionEngine(config)
        result = engine.execute_stroke(tool.tool_id, (2, 2, 2))
        assert result.feasible
        record = {"source_hash": case.semantic_hash, "native_model_hash": config.fingerprint,
            "route": {"tool": asdict(tool), "window": {"center_mm": window.center_mm.tolist(),
                      "normal_inward": window.normal_inward.tolist(), "radius_mm": window.radius_mm,
                      "window_id": window.window_id},
                      "entry_mm": [2, 2, -.5], "target_mm": [2, 2, 2]}, "history": engine.history}
        record["candidate_hash"] = content_hash(record)
        path = tmp_path / (tool.tool_id + ".json")
        RUNNER.write_json(path, record)
        native_paths.append(path)
    rows = [{"route_id": "analytical-" + str(i), "window_id": "window" + str(i),
        "entry_mm": [2, 2, -.5], "target_mm": [2, 2, 2], "tool": asdict(NATIVE_GENERIC_TOOLS[0]),
        "exact_native_stroke_feasible": False, "exact_native_stroke_reason": "retained-negative-control"}
        for i in range(3)]
    route_path = tmp_path / "routes.json"
    RUNNER.write_json(route_path, {"routes": rows})
    declaration = json.loads((PATH.parents[1] / "manifests/experiments/ucsf-functional-sensitivity-v1.json").read_text())
    declaration.update(original_case_hash=case.semantic_hash, prior_case_hash=case.semantic_hash,
        selected_static_route_ids=[r["route_id"] for r in rows],
        static_window_normals_ras={r["window_id"]: [0, 0, 1] for r in rows})
    for panel in declaration["uncertainty_panels"]:
        panel["worlds"] = 2
    declaration["inputs"] = {name: {"path": str(path.relative_to(tmp_path)), "sha256": RUNNER.file_sha(path)}
        for name, path in zip(("original_case", "prior_case", "fine_history", "wide_history", "original_routes"),
                              (case_path, case_path, *native_paths, route_path))}
    declaration_path = tmp_path / "declaration.json"
    RUNNER.write_json(declaration_path, declaration)
    return tmp_path, declaration_path


def test_end_to_end_analytical_workflow_freezes_all_panels_and_preserves_unknowns(workflow_fixture, monkeypatch):
    root, declaration = workflow_fixture
    from resectionlab import functional_events
    original_evaluate = functional_events.evaluate_functional_candidates
    observed = []

    def checked_evaluate(*args, **kwargs):
        frozen = json.loads((root / "run/all-panels-frozen.json").read_text())
        assert len(frozen) == 8
        observed.append((kwargs["evidence"].motor is None, kwargs["evidence"].language is None))
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(functional_events, "evaluate_functional_candidates", checked_evaluate)
    before = RUNNER.file_sha(root / "original.ressectionlab")
    report = RUNNER.run(root, root / "run", root / "new-evidence.ressectionlab", declaration)
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert len(report["panels"]) == 8 and len(observed) == 8
    assert observed[-2:] == [(False, True), (True, False)]
    assert RUNNER.file_sha(root / "original.ressectionlab") == before
    assert report["source_and_inputs_unchanged"]
    for panel in report["panels"]:
        assert len(panel["candidates"]) == 6
        if "deterministic" in panel["panel"]:
            assert panel["distinct_transforms"] == 1
    geometry = json.loads((root / "run/geometry.json").read_text())
    assert len(geometry["retained_historical_rejections"]) == 3
    assert all(row["independent_native_audit"]["feasible"] for row in geometry["alternatives"][:2])
    assert (root / "new-evidence.ressectionlab").is_file()


def test_changed_source_failure_is_retained_and_does_not_create_evidence(workflow_fixture):
    root, declaration = workflow_fixture
    (root / "routes.json").write_text("changed")
    with pytest.raises(ValueError, match="Source bytes changed"):
        RUNNER.run(root, root / "failed", root / "should-not-exist.ressectionlab", declaration)
    failure = json.loads((root / "failed/failure.json").read_text())
    assert failure["automatic_retry"] is False and failure["completed_panels"] == []
    assert not (root / "should-not-exist.ressectionlab").exists()


def test_parent_watchdog_terminates_stalled_worker_and_keeps_phase_log(tmp_path, monkeypatch):
    child = tmp_path / "stall_control.py"
    child.write_text("import time\nprint('phase-before-stall', flush=True)\ntime.sleep(30)\n")
    monkeypatch.setattr(RUNNER, "__file__", str(child))
    declaration = tmp_path / "declaration.json"
    RUNNER.write_json(declaration, {"budgets": {"wall_seconds": .3, "peak_rss_gib": 3}})
    report = RUNNER.supervise(tmp_path, tmp_path / "supervised", tmp_path / "unused.ressectionlab", declaration)
    assert report["status"] == "failed" and report["termination_reason"] == "parent_wall_budget_exceeded"
    assert report["worker_exit_code"] != 0 and report["parent_wall_seconds"] < 4
    assert "phase-before-stall" in (tmp_path / "supervised/worker.log").read_text()
    assert (tmp_path / "supervised/failure.json").is_file()
    assert (tmp_path / "supervised/frozen-declaration.json").read_bytes() == declaration.read_bytes()


def test_parent_does_not_accept_zero_exit_without_worker_completion_receipt(tmp_path, monkeypatch):
    child = tmp_path / "empty_control.py"
    child.write_text("print('exited without result', flush=True)\n")
    monkeypatch.setattr(RUNNER, "__file__", str(child))
    declaration = tmp_path / "declaration.json"
    RUNNER.write_json(declaration, {"budgets": {"wall_seconds": 5, "peak_rss_gib": 3}})
    report = RUNNER.supervise(tmp_path, tmp_path / "supervised", tmp_path / "unused.ressectionlab", declaration)
    assert report["worker_exit_code"] == 0
    assert report["status"] == "failed" and report["termination_reason"] == "worker_exit_without_valid_completion_receipt"
