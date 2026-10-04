"""Independent analytical release controls; no real-patient event execution."""
import json

import pytest

from test_patient_functional_sensitivity_runner import RUNNER, workflow_fixture


def test_historical_native_rejections_cannot_reenter_evaluated_frontier(workflow_fixture):
    root, declaration = workflow_fixture
    report = RUNNER.run(root, root / "review-run", root / "review-case.ressectionlab", declaration)
    for panel in report["panels"]:
        rejected = [item for item in panel["candidates"] if item["plan_id"].startswith("static_")]
        assert len(rejected) == 3
        assert all(item["status"] == "rejected_native_execution" for item in rejected)
        assert all(item["events"] == [] for item in rejected)
        assert all(all(cost["mean"] is None and cost["assessed_worlds"] == 0
                       for cost in item["costs"].values()) for item in rejected)


def test_declaration_change_after_freeze_cannot_be_reported_as_original_settings(workflow_fixture, monkeypatch):
    root, declaration = workflow_fixture
    from resectionlab import functional_events
    original = functional_events.evaluate_functional_candidates
    changed = False

    def replace_declaration(*args, **kwargs):
        nonlocal changed
        if not changed:
            assert (root / "mutated-run/all-panels-frozen.json").is_file()
            data = json.loads(declaration.read_text())
            data["thresholds"] = [.9]
            declaration.write_text(json.dumps(data))
            changed = True
        return original(*args, **kwargs)

    monkeypatch.setattr(functional_events, "evaluate_functional_candidates", replace_declaration)
    with pytest.raises(ValueError, match="[Dd]eclaration|[Ss]ource|settings"):
        RUNNER.run(root, root / "mutated-run", root / "mutated-case.ressectionlab", declaration)
    failure = json.loads((root / "mutated-run/failure.json").read_text())
    assert failure["automatic_retry"] is False
    assert not (root / "mutated-run/report.json").exists()


def test_memory_watchdog_stops_actual_bounded_allocation_child(tmp_path, monkeypatch):
    child = tmp_path / "memory_control.py"
    child.write_text("import time\npayload=bytearray(64*1024*1024)\nprint('allocated', flush=True)\ntime.sleep(30)\n")
    monkeypatch.setattr(RUNNER, "__file__", str(child))
    declaration = tmp_path / "declaration.json"
    RUNNER.write_json(declaration, {"budgets": {"wall_seconds": 5, "peak_rss_gib": .02}})
    report = RUNNER.supervise(tmp_path, tmp_path / "supervised", tmp_path / "unused.ressectionlab", declaration)
    assert report["status"] == "failed"
    assert report["termination_reason"] == "parent_sampled_rss_budget_exceeded"
    assert report["worker_exit_code"] != 0
    assert report["parent_sampled_peak_rss_gib"] > .02
    assert report["parent_wall_seconds"] < 4
    assert (tmp_path / "supervised/failure.json").is_file()
