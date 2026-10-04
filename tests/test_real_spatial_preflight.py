"""Patient-role and audit-retention checks, without opening patient images."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("real_spatial_preflight", ROOT / "scripts/preflight_real_spatial_policy.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def cohort():
    return json.loads((ROOT / "manifests/experiments/btc-spatial-development-cohort-v1.json").read_text())


@pytest.mark.parametrize("subject", ["sub-PAT05", "sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28"])
def test_existing_and_added_training_groups_keep_one_patient_identity(cohort, subject):
    assert runner.require_train_role(cohort, subject) == "BTC:" + subject


@pytest.mark.parametrize("subject", ["sub-PAT26", "sub-PAT27", "sub-PAT29", "sub-PAT31", "sub-PAT07", "unknown"])
def test_selection_and_unopened_cases_cannot_enter_training_preflight(cohort, subject):
    with pytest.raises(ValueError, match="only.*TRAIN"):
        runner.require_train_role(cohort, subject)


def test_duplicate_patient_identity_is_not_two_training_cases(cohort):
    cohort["candidates"].append(cohort["existing_development_records"][0])
    with pytest.raises(ValueError, match="Duplicate patient"):
        runner.require_train_role(cohort, "sub-PAT05")


def test_changed_source_stops_before_any_bundle_read(monkeypatch):
    reads = []
    monkeypatch.setattr(runner, "numerical_source_inventory", lambda: {"src/resectionlab/spatial_policy.py": "actual"})
    monkeypatch.setattr(runner, "sha256", lambda path: reads.append(str(path)) or "actual")
    declaration = {"version": runner.VERSION, "mode": "profile", "track": "annotation_assisted",
                   "source_sha256": {"src/resectionlab/spatial_policy.py": "different"}}
    with pytest.raises(ValueError, match="source hashes"):
        runner.load_inputs(declaration)
    assert reads == []


def test_sealed_role_is_rejected_before_case_checksum_or_image_load(cohort, monkeypatch, tmp_path):
    reads = []
    monkeypatch.setattr(runner, "numerical_source_inventory", lambda: {"src/resectionlab/spatial_policy.py": "known"})
    monkeypatch.setattr(runner, "sha256", lambda path: reads.append(str(path)) or "known")
    declaration = {"version": runner.VERSION, "mode": "profile", "track": "annotation_assisted",
        "source_sha256": {"src/resectionlab/spatial_policy.py": "known"},
        "cohort_path": runner.COHORT_PATH, "cohort_sha256": runner.COHORT_SHA256, "subject": "sub-PAT29",
        "case_bundle": "THIS_MUST_NOT_BE_OPENED"}
    with pytest.raises(ValueError, match="only.*TRAIN"):
        runner.load_inputs(declaration)
    assert all("THIS_MUST_NOT_BE_OPENED" not in path for path in reads)


def test_committed_history_is_persisted_before_independent_rejection():
    from types import SimpleNamespace
    import torch
    from resectionlab.evaluation import IndependentGeometryResult

    class Policy(torch.nn.Linear):
        def act(self, observation, **kwargs):
            return "STOP"

    class Task:
        terminated = False
        def fresh(self):
            return Task()
        def observation(self):
            return SimpleNamespace(action_ids=("STOP", "CUT"), fingerprint="observed")
        def step(self, action):
            assert action == "CUT"  # Profile rule must exercise the available native path.
            self.terminated = True
            return SimpleNamespace(reward=1., terminated=True, info={"committed": True})
        def metrics(self):
            return {"history": [{"action_id": "CUT", "removed_indices_native": [[1, 2, 3]]}]}
        def independent_geometry_check(self):
            return IndependentGeometryResult(False, ("deliberate_checker_rejection",))

    records = []
    with pytest.raises(RuntimeError, match="Independent native checker"):
        runner.episode(Task(), Policy(1, 1), None, stochastic=False, profile_actions=True,
                       checkpoint=lambda row: records.append(row.copy()))
    assert any(row.get("latest_transition_info", {}).get("committed") for row in records)
    assert any(row.get("metrics", {}).get("history") for row in records)
    assert records[-1]["independent_geometry_check"]["failures"] == ("deliberate_checker_rejection",)


def test_live_memory_watchdog_stops_child_and_preserves_partial_output(tmp_path):
    child = (
        "import pathlib,time; "
        "pathlib.Path('partial.json').write_text('{\"phase\":\"before_allocation\"}'); "
        "payload=bytearray(96*1024*1024); print('allocated',flush=True); time.sleep(20)"
    )
    # Only the child changes directory, leaving concurrently running tests alone.
    command = [sys.executable, "-c", "import os; os.chdir(" + repr(str(tmp_path)) + "); " + child]
    result = runner.supervise_worker(command, tmp_path,
        {"max_wall_seconds": 5, "max_rss_bytes": 48 * 1024 * 1024}, "frozen-declaration")
    assert result["termination_reason"] == "parent_sampled_rss_budget_exceeded"
    assert result["sampled_peak_rss_bytes"] > 48 * 1024 * 1024
    assert result["status"] == "failed" and result["returncode"] != 0
    assert result["seconds"] < 5
    assert json.loads((tmp_path / "partial.json").read_text()) == {"phase": "before_allocation"}
    assert json.loads((tmp_path / "supervisor-failure.json").read_text()) == result
    assert result["declaration_sha256"] == "frozen-declaration"
    assert "allocated" in (tmp_path / "worker.log").read_text()


def test_live_deadline_watchdog_stops_child_and_retains_logs(tmp_path):
    result = runner.supervise_worker([sys.executable, "-c",
        "import time; print('child-started',flush=True); time.sleep(20)"], tmp_path,
        {"max_wall_seconds": .6, "max_rss_bytes": 512 * 1024 * 1024}, "frozen-declaration")
    assert result["termination_reason"] == "parent_wall_budget_exceeded"
    assert result["timed_out"] and result["returncode"] != 0
    assert .6 <= result["seconds"] < 5
    assert result["rss_samples"] > 0
    assert json.loads((tmp_path / "supervisor-failure.json").read_text()) == result
    assert "child-started" in (tmp_path / "worker.log").read_text()
