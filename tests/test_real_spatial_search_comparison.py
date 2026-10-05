"""Metadata, flow and failure controls; no patient loading or optimizer work."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("real_spatial_comparison", ROOT / "scripts/compare_real_spatial_search.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
HEX = "a" * 64


@pytest.fixture
def declared(monkeypatch):
    source = {"unit_source.py": HEX}
    monkeypatch.setattr(runner, "numerical_source_inventory", lambda: source)
    monkeypatch.setattr(runner.preflight, "sha256", lambda path: HEX)
    return {"version": runner.VERSION, "status": "prospective_frozen", "track": "annotation_assisted",
        "methods": ["SEARCH", "untrained_policy"], "source_sha256": source,
        "cohort_path": runner.COHORT_PATH, "cohort_sha256": runner.COHORT_SHA256,
        "member": {"subject": runner.SUBJECT, "role": "TRAIN", "case_bundle": runner.BUNDLE_PATH,
            "case_bundle_sha256": HEX, "case_semantic_hash": "sha256:" + HEX, "access": {},
            "research_support_acknowledgment": {}, "expected_native_grid_binding": {"version": "unit"}},
        "tools": [{}], "objective": {}, "policy_config": {}, "expected_policy_architecture_hash": "sha256:" + HEX,
        "adapter_options": {}, "coverage_evidence": {"path": "artifacts/unit/receipt.json", "sha256": HEX},
        "source_profile_declaration": {"path": "manifests/experiments/unit.json", "sha256": HEX},
        "settings": {"optimizer_updates": 0, "seed": 11, "max_steps": 3, "max_rss_bytes": 2**40,
                     "max_wall_seconds": 300, "online_episode_seconds": 300,
                     "device": "cpu", "torch_threads": 1, "blas_thread_caps": 1},
        "search": {"max_calls": 512, "beam_width": 2, "seconds": 180., "transition_mode": "lazy_planning"}}


def test_valid_declaration_requires_only_metadata_and_never_case_bytes(declared, monkeypatch):
    reads = []
    monkeypatch.setattr(runner.preflight, "sha256", lambda path: reads.append(str(path)) or HEX)
    runner.validate_declaration(declared)
    assert reads == [str(ROOT / "artifacts/unit/receipt.json"), str(ROOT / "manifests/experiments/unit.json")]


@pytest.mark.parametrize("subject", ["sub-PAT16", "sub-PAT26", "sub-PAT29", "sub-PAT31", "UPENN-GBM-00001"])
def test_other_patients_are_rejected_before_any_bound_file_read(declared, monkeypatch, subject):
    declared["member"]["subject"] = subject
    monkeypatch.setattr(runner.preflight, "sha256", lambda path: pytest.fail("Unexpected file read"))
    with pytest.raises(ValueError, match="only TRAIN"):
        runner.validate_declaration(declared)
    with pytest.raises(ValueError, match="Only the bound"):
        runner.load_member(declared, {})


def test_bundle_path_cannot_alias_an_unopened_patient(declared, monkeypatch):
    declared["member"]["case_bundle"] = "outputs/cases/BTC-sub-PAT29.ressectionlab"
    monkeypatch.setattr(runner.preflight, "sha256", lambda path: pytest.fail("Unexpected case checksum"))
    with pytest.raises(ValueError, match="Explicit patient"):
        runner.validate_declaration(declared)
    with pytest.raises(ValueError, match="Only the bound"):
        runner.load_member(declared, {})


@pytest.mark.parametrize("updates", [1, -1, True, None])
def test_optimizer_execution_is_not_a_declaration_option(declared, updates):
    declared["settings"]["optimizer_updates"] = updates
    with pytest.raises(ValueError, match="Optimizer execution is disabled"):
        runner.validate_declaration(declared)


def test_no_implicit_search_or_wall_budget_and_no_image_as_profile_evidence(declared):
    for key in ("max_calls", "beam_width", "seconds", "transition_mode"):
        altered = deepcopy(declared)
        altered["search"].pop(key)
        with pytest.raises(ValueError, match="Search limits"):
            runner.validate_declaration(altered)
    declared["coverage_evidence"]["path"] = runner.BUNDLE_PATH
    with pytest.raises(ValueError, match="JSON artifact"):
        runner.validate_declaration(declared)


def test_modified_source_stops_before_coverage_or_case_reads(declared, monkeypatch):
    declared["source_sha256"] = {"changed": HEX}
    monkeypatch.setattr(runner.preflight, "sha256", lambda path: pytest.fail("Unexpected file read"))
    with pytest.raises(ValueError, match="source hashes"):
        runner.validate_declaration(declared)


def test_validation_is_default_and_execution_requires_a_bound_hash(declared, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(runner.preflight, "read_declaration", lambda *args: (declared, HEX, b"{}"))
    monkeypatch.setattr(runner, "worker", lambda *args: pytest.fail("Worker must remain unreleased"))
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--declaration", "unit.json"])
    runner.main()
    assert "no patient opened" in capsys.readouterr().out
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--declaration", "unit.json", "--execute",
                                           "--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="frozen declaration hash"):
        runner.main()
    assert not (tmp_path / "out").exists()


class Profiler:
    def __init__(self, *args):
        self.phases = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    @contextmanager
    def phase(self, name):
        self.phases.append(name)
        yield
    def snapshot(self):
        return {"phases": list(self.phases)}


@dataclass
class Audit:
    feasible: bool


class ReplayTask:
    def __init__(self, *, audit=True):
        self.terminated = False
        self.history = []
        self.audit = audit
    def fresh(self):
        return ReplayTask(audit=self.audit)
    def step(self, action):
        self.history.append(action)
        self.terminated = action == "STOP"
        return SimpleNamespace(info={"action_id": action})
    def metrics(self):
        return {"history": list(self.history), "total_reward": 1.}
    def independent_geometry_check(self):
        return Audit(self.audit)


def test_search_replay_records_commits_and_independent_audit(monkeypatch):
    import resectionlab.observed_search as search
    monkeypatch.setattr(search, "observed_beam_search", lambda *args, **kwargs: (("CUT", "STOP"), {"calls": 1}))
    base, rows, profiler = ReplayTask(), [], Profiler()
    result = runner.replay_search(base, {}, preserve=rows.append, profiler=profiler)
    assert result["status"] == "complete" and result["independent_geometry_check"]["feasible"]
    assert result["metrics"]["history"] == ["CUT", "STOP"] and base.history == []
    assert any(row.get("latest_committed_transition", {}).get("action_id") == "CUT" for row in rows)
    committed = [row for row in rows if "committed_transition_records" in row]
    assert [row["committed_transition_count"] for row in committed] == [1, 2]
    assert committed[0]["committed_transition_records"] == [{"action_id": "CUT"}]
    assert committed[1]["committed_transition_records"] == [{"action_id": "CUT"}, {"action_id": "STOP"}]
    assert profiler.phases == ["SEARCH:planning", "SEARCH:execution", "SEARCH:independent_audit"]


def test_search_timeout_cannot_become_a_fabricated_stop_teacher(monkeypatch):
    import resectionlab.observed_search as search
    failure = search.ObservedSearchLimit("time exhausted", accounting={"model_transition_calls": 4},
                                        best_sequence=("STOP",))
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(search, "observed_beam_search", fail)
    rows = []
    with pytest.raises(search.ObservedSearchLimit) as caught:
        runner.replay_search(ReplayTask(), {}, preserve=rows.append, profiler=Profiler())
    assert caught.value is failure and rows == []


def test_failed_search_audit_preserves_actual_history(monkeypatch):
    import resectionlab.observed_search as search
    monkeypatch.setattr(search, "observed_beam_search", lambda *args, **kwargs: (("STOP",), {}))
    rows = []
    with pytest.raises(RuntimeError, match="Independent native checker"):
        runner.replay_search(ReplayTask(audit=False), {}, preserve=rows.append, profiler=Profiler())
    assert any(row.get("metrics", {}).get("history") == ["STOP"] for row in rows)
    assert rows[-1] == {"independent_geometry_check": {"feasible": False}}


@pytest.mark.parametrize("search_fails", [False, True])
def test_worker_flow_has_no_optimizer_and_preserves_failed_search_cost(declared, monkeypatch, tmp_path, search_fails):
    import torch
    import resectionlab.spatial_policy as policies
    import resectionlab.spatial_policy_diagnostics as diagnostics
    from resectionlab.observed_search import ObservedSearchLimit
    class Policy:
        def __init__(self, config):
            self.config = config
            self.architecture_hash = "sha256:" + HEX
        def architecture_record(self):
            return {"test_control": True}
    monkeypatch.setattr(policies, "SpatialPolicy", Policy)
    monkeypatch.setattr(policies, "parameter_hash", lambda policy: "sha256:" + HEX)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: pytest.fail("No optimizer may be constructed"))
    monkeypatch.setattr(diagnostics, "NativePreviewProfiler", Profiler)
    monkeypatch.setattr(diagnostics, "spatial_coverage", lambda *args, **kwargs: {"coverage": "unit"})
    monkeypatch.setattr(diagnostics, "runtime_proposal_coverage", lambda *args, **kwargs: {"coverage": "unit"})
    base = SimpleNamespace(metrics=lambda: {"native_grid_reconciliation": {}},
        observation=lambda: object(), candidate_inventory=lambda: {},
        case=SimpleNamespace(structural_intensity=SimpleNamespace(shape=(1, 1, 1)), affine_ras_mm=None, nominal_target=None))
    monkeypatch.setattr(runner, "load_member", lambda *args: base)
    failure = ObservedSearchLimit("declared timeout", accounting={"model_transition_calls": 7}, best_sequence=("STOP",))
    def search(*args, **kwargs):
        import json
        durable = json.loads((tmp_path / "receipt.json").read_text())
        assert durable["methods"]["SEARCH"]["status"] == "starting"
        assert durable["methods"]["untrained_policy"]["status"] == "not_executed"
        with kwargs["profiler"].phase("SEARCH:planning"):
            if search_fails:
                raise failure
            return {"status": "complete", "online_seconds": .01, "metrics": {"total_reward": 1.}}
    monkeypatch.setattr(runner, "replay_search", search)
    def episode(*args, **kwargs):
        assert not search_fails
        assert kwargs["stochastic"] is False and kwargs["diagnostics"] is False
        return (), {"online_seconds": .01, "metrics": {"total_reward": 0.}}
    monkeypatch.setattr(runner.preflight, "episode", episode)
    if search_fails:
        with pytest.raises(ObservedSearchLimit):
            runner.worker(declared, tmp_path)
    else:
        runner.worker(declared, tmp_path)
    import json
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["optimizer_updates"] == 0
    if search_fails:
        assert receipt["status"] == "failed"
        assert receipt["methods"]["SEARCH"]["partial_search_accounting"]["model_transition_calls"] == 7
        assert receipt["methods"]["untrained_policy"]["status"] == "not_executed"
    else:
        assert receipt["status"] == "complete"
        assert all(row["status"] == "complete" for row in receipt["methods"].values())
        assert receipt["initial_parameter_hash"] == receipt["final_parameter_hash"]
    assert "SEARCH:planning" in receipt["native_preview_cost"]["phases"]


def test_committed_interruption_preserves_ordered_replay_history(monkeypatch):
    import resectionlab.observed_search as search
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    monkeypatch.setattr(search, "observed_beam_search", lambda *args, **kwargs: (("FIRST", "SECOND"), {}))
    class InterruptedTask(ReplayTask):
        def fresh(self):
            return InterruptedTask()
        def step(self, action):
            if action == "SECOND":
                raise CommittedTransitionInterrupted({"action_id": action}, 1.)
            return super().step(action)
    rows = []
    with pytest.raises(CommittedTransitionInterrupted):
        runner.replay_search(InterruptedTask(), {}, preserve=rows.append, profiler=Profiler())
    assert rows[-1]["committed_transition_count"] == 2
    assert rows[-1]["committed_transition_records"] == [{"action_id": "FIRST"}, {"action_id": "SECOND"}]


def test_source_profile_metadata_hash_is_enforced(declared, monkeypatch):
    monkeypatch.setattr(runner.preflight, "sha256", lambda path: "b" * 64 if "manifests" in str(path) else HEX)
    with pytest.raises(ValueError, match="source_profile_declaration"):
        runner.validate_declaration(declared)


def test_prospective_case_geometry_and_objective_match_measured_nominal_profile():
    import json
    prior = json.loads((ROOT / "manifests/experiments/pat05-real-spatial-profile-v3-nominal64.json").read_text())
    current = json.loads((ROOT / "manifests/experiments/pat05-real-spatial-search-comparison-v1.json").read_text())
    for field in ("case_bundle", "case_bundle_sha256", "case_semantic_hash", "access", "expected_native_grid_binding"):
        assert current["member"][field] == prior[field]
    for field in ("tools", "policy_config", "expected_policy_architecture_hash"):
        assert current[field] == prior[field]
    expected_options = deepcopy(prior["adapter_options"])
    acknowledgment = expected_options.pop("research_support_acknowledgment")
    assert current["member"]["research_support_acknowledgment"] == acknowledgment
    assert current["adapter_options"] == expected_options
    assert current["objective"] == {key: prior["objective"][key] for key in current["objective"]}
    assert current["settings"]["max_steps"] == prior["settings"]["max_steps"] == 3
    assert current["settings"]["optimizer_updates"] == 0
    assert current["search"] == {"max_calls": 512, "beam_width": 2, "seconds": 180., "transition_mode": "lazy_planning"}
