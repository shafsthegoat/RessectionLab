"""Independent synthetic orchestration review; never opens public patient data."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_public_native_capsule_cache as probe
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NativeResectionConfig, NATIVE_GENERIC_TOOLS
from resectionlab.simulation import RewardSpec


def tiny_simulator():
    tissue = np.zeros((7, 7, 7), bool)
    tissue[1:6, 1:6, 1:6] = True
    labels = np.zeros_like(tissue, dtype=np.int16)
    labels[3, 3, 3:5] = 1
    cfg = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow([3, 3, .5], [0, 0, 1], 4), (NATIVE_GENERIC_TOOLS[0],),
        "synthetic-independent-runner-review", "synthetic explicit cube")
    return AxisColumnNativeSimulator(cfg,
        proposal_config=AxisColumnProposalConfig(offsets_source_voxels=((0, 0),), max_primary_rays=1),
        max_steps=1)


@pytest.fixture
def synthetic_worker(monkeypatch, tmp_path):
    """Replace patient binding with an explicitly separate one-cut phantom contract."""
    sim = tiny_simulator()
    cfg = sim.native_config
    actions = (sim.proposed_actions()[1].action_id,)
    guard = probe.pre.ResourceGuard(probe.pre.PreflightBudget(worker_wall_seconds=60.))
    trace, _, _ = probe.run_fixed_trace(sim, actions, 13, guard, tmp_path / "synthetic-reference")
    root, output = tmp_path / "synthetic-source", tmp_path / "output"
    output.mkdir()
    root.mkdir()
    bundle = root / "synthetic.bundle"
    bundle.write_bytes(b"explicit synthetic fixture; not a patient")
    runtime = "sha256:synthetic-runtime"
    probe.pre.write_json(output / "launch-source.json", {"runtime_content_hash": runtime})
    profile = root / probe.V2_PROFILE
    probe.pre.write_json(profile / "model.json", {"decision_model_hash": sim.decision_model_hash})
    probe.pre.write_json(profile / "initial/inventory.json", {"complete_inventory": trace["snapshots"][0]["inventory"]})
    probe.pre.write_json(profile / "sequence-freeze.json", {"source_hash": "synthetic-historical-runtime"})
    (profile / "greedy").mkdir()
    probe._write_gzip(profile / "greedy/episode.json.gz", {"actions": actions, "metrics": trace["metrics"],
        "total_reward": trace["metrics"]["total_reward"]})
    target = {"bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "semantic_hash": "synthetic-case", "planning_hash": "synthetic-planning", "access": asdict(cfg.access),
        "reward": asdict(RewardSpec()), "partial_contact_weight": .05,
        "world_partitions": {"optimization": {"synthetic": True}}}
    reference = {"target": target}
    value = {"v2_model": {"native_config_hash": cfg.fingerprint, "decision_model_hash": sim.decision_model_hash,
        "proposal_model_hash": sim._proposer.model_hash, "fixed_actions": actions, "optimization_seed": 13,
        "greedy_history_hash": probe.content_hash(trace["metrics"]["history"])},
        "cache": {"max_payload_bytes": 1024**2, "max_entries": 256},
        "fixed_work": {"preview_attempts_per_inventory": [1, 0], "preview_attempts_per_episode": 1,
                       "common_template_preview_attempts": 1},
        "scope": "synthetic orchestration fixture"}
    case = SimpleNamespace(semantic_hash="synthetic-case", planning_hash="synthetic-planning", frame="RAS+",
                           compartments={"target": cfg.target_labels > 0})
    partition = SimpleNamespace(seeds=(13,), role=SimpleNamespace(value="optimization"), generator=None,
                                to_dict=lambda: {"role": "optimization", "seeds": [13]})
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "source_snapshot", lambda: {"runtime_content_hash": runtime})
    monkeypatch.setattr(probe, "_check_bound_inputs", lambda value: None)
    monkeypatch.setattr(probe.pre, "_checked_manifest", lambda path, digest: reference if path.name == probe.pre.REFERENCE_PATH.name else {})
    monkeypatch.setattr(probe.pre, "_partition", lambda value: partition)
    monkeypatch.setattr(probe.pre, "assert_declared_native_configuration", lambda *args: {"synthetic": True})
    monkeypatch.setattr(probe.pre, "assert_public_action_model", lambda *args: None)
    import resectionlab.imaging
    monkeypatch.setattr(resectionlab.imaging, "load_case", lambda path: case)
    monkeypatch.setattr(probe.native, "native_config_from_case", lambda *args, **kwargs: cfg)
    created, audits = [], []
    def factory(*args, **kwargs):
        current = tiny_simulator()
        created.append(current)
        return current
    monkeypatch.setattr(probe, "AxisColumnNativeSimulator", factory)
    def audit(*args, **kwargs):
        assert probe.native.capsule_voxel_indices is probe.geometry.capsule_voxel_indices
        assert all("propose" not in current._proposer.__dict__ for current in created)
        audits.append(copy.deepcopy(args[2]))
        return SimpleNamespace(feasible=True, complete_tool_checked=True, frontier_checked=True,
            to_dict=lambda: {"feasible": True, "synthetic_audit_stub": True})
    monkeypatch.setattr(probe.pre, "independent_check_native_history", audit)
    return SimpleNamespace(output=output, bundle=bundle, value=value, guard=guard,
                           audits=audits, created=created, runtime=runtime)


def test_worker_restores_before_all_audits_and_cold_cache_starts_empty(synthetic_worker):
    env = synthetic_worker
    probe._public_worker(env.output, env.bundle, env.value, env.guard)
    result = json.loads((env.output / "result.json").read_text())
    assert result["status"] == "completed" and len(env.audits) == 4
    phases = result["phases"]
    assert [row["mode"] for row in phases] == list(probe.MODES)
    assert len({row["scientific_trace_hash"] for row in phases}) == 1
    assert phases[0]["cache_before"] is None
    assert phases[1]["cache_before"]["calls"] == phases[1]["cache_before"]["entries"] == 0
    assert phases[2]["cache_before"] == phases[1]["cache_after"]
    assert phases[3]["cache_after"] == phases[2]["cache_after"]
    assert result["gradient_updates"] == 0 and not result["final_worlds_used"] and not result["stress_worlds_used"]


def test_common_uncached_setup_preview_count_is_bound(synthetic_worker):
    env = synthetic_worker
    env.value["fixed_work"]["common_template_preview_attempts"] = 2
    with pytest.raises(RuntimeError, match="template|denominator"):
        probe._public_worker(env.output, env.bundle, env.value, env.guard)
    assert not (env.output / "result.json").exists() and not env.audits


@pytest.mark.parametrize("changed", ["mask", "certificate", "action_feature", "inventory", "work_count"])
def test_worker_rejects_changed_scientific_or_work_outputs(monkeypatch, synthetic_worker, changed):
    env = synthetic_worker
    original = probe.run_fixed_trace
    def altered(*args):
        trace, masks, timing = original(*args)
        if args[-1].name == "cached_cold":
            if changed == "mask":
                masks[0]["removed_mask"] += b"changed"
            elif changed == "certificate":
                trace["snapshots"][0]["certificates"]["invented"] = "different certificate"
            elif changed == "action_feature":
                trace["snapshots"][0]["action_features"][0][0] += .125
            elif changed == "inventory":
                trace["snapshots"][0]["inventory"]["attempts"][0]["reason"] = "different reason"
            else:
                timing["adapter_proposal_accounting"]["preview_calls"] += 1
        return trace, masks, timing
    monkeypatch.setattr(probe, "run_fixed_trace", altered)
    with pytest.raises(RuntimeError, match="differs|denominator"):
        probe._public_worker(env.output, env.bundle, env.value, env.guard)
    assert not (env.output / "result.json").exists() and not env.audits
    assert probe.native.capsule_voxel_indices is probe.geometry.capsule_voxel_indices
    assert "propose" not in env.created[0]._proposer.__dict__


def test_worker_rejects_independent_audit_failure_after_restoration(monkeypatch, synthetic_worker):
    env = synthetic_worker
    def rejected_audit(*args, **kwargs):
        assert probe.native.capsule_voxel_indices is probe.geometry.capsule_voxel_indices
        return SimpleNamespace(feasible=False, complete_tool_checked=True, frontier_checked=True,
                               to_dict=lambda: {"feasible": False, "reason": "synthetic deliberate rejection"})
    monkeypatch.setattr(probe.pre, "independent_check_native_history", rejected_audit)
    with pytest.raises(RuntimeError, match="Independent native replay failed"):
        probe._public_worker(env.output, env.bundle, env.value, env.guard)
    assert not (env.output / "result.json").exists()
    saved = json.loads((env.output / "reference_before-independent-audit.json").read_text())
    assert saved["receipt"]["feasible"] is False
    assert all((env.output / mode / "receipt.json").exists() for mode in probe.MODES)


def test_trace_hook_restoration_failure_keeps_incomplete_diagnostic(monkeypatch, tmp_path):
    sim = tiny_simulator()
    actions = (sim.proposed_actions()[1].action_id,)
    original = probe.pre.write_json
    def changed_hook(path, value):
        original(path, value)
        if path.name == "timing.json":
            sim._proposer.propose = lambda engine: None
    monkeypatch.setattr(probe.pre, "write_json", changed_hook)
    guard = probe.pre.ResourceGuard(probe.pre.PreflightBudget(worker_wall_seconds=60.))
    folder = tmp_path / "restoration-failure"
    with pytest.raises(RuntimeError):
        probe.run_fixed_trace(sim, actions, 13, guard, folder)
    assert "propose" not in sim._proposer.__dict__
    assert (folder / "failure.json").exists()


def test_committed_interruption_records_paid_history_and_restores(monkeypatch, tmp_path):
    sim = tiny_simulator()
    actions = (sim.proposed_actions()[1].action_id,)
    original = sim.step
    class InterruptedAfterCommit(RuntimeError):
        committed = True
    def interrupted(action):
        original(action)
        raise InterruptedAfterCommit("synthetic failure after a real native commit")
    monkeypatch.setattr(sim, "step", interrupted)
    guard = probe.pre.ResourceGuard(probe.pre.PreflightBudget(worker_wall_seconds=60.))
    folder = tmp_path / "committed-failure"
    with pytest.raises(InterruptedAfterCommit):
        probe.run_fixed_trace(sim, actions, 13, guard, folder)
    record = json.loads((folder / "failure.json").read_text())
    assert record["status"] == "incomplete" and record["committed_transition_after_interruption"]
    assert record["raw_state"]["committed_cut_count"] == 1
    assert probe._canonical(record["raw_state"]["history"]) == probe._canonical(sim._history)
    assert sim.engine.removed_mask.any()
    assert "propose" not in sim._proposer.__dict__


def test_stop_cannot_substitute_for_the_declared_cut(tmp_path):
    sim = tiny_simulator()
    guard = probe.pre.ResourceGuard(probe.pre.PreflightBudget(worker_wall_seconds=60.))
    folder = tmp_path / "stop-instead-of-cut"
    with pytest.raises(RuntimeError, match="complete declared episode"):
        probe.run_fixed_trace(sim, ("STOP",), 13, guard, folder)
    record = json.loads((folder / "failure.json").read_text())
    assert record["raw_state"]["committed_cut_count"] == 0


@pytest.mark.parametrize("mutation", ["source", "bundle"])
def test_worker_rechecks_source_and_bundle_before_publication(monkeypatch, synthetic_worker, mutation):
    env = synthetic_worker
    if mutation == "source":
        calls = 0
        def snapshot():
            nonlocal calls
            calls += 1
            return {"runtime_content_hash": env.runtime if calls == 1 else "sha256:changed-source"}
        monkeypatch.setattr(probe, "source_snapshot", snapshot)
    else:
        original = probe.pre.independent_check_native_history
        def audit(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(env.audits) == 4:
                env.bundle.write_bytes(b"changed synthetic bundle")
            return result
        monkeypatch.setattr(probe.pre, "independent_check_native_history", audit)
    with pytest.raises(RuntimeError, match="changed during public probe"):
        probe._public_worker(env.output, env.bundle, env.value, env.guard)
    assert len(env.audits) == 4 and not (env.output / "result.json").exists()


@pytest.fixture
def mocked_launcher(monkeypatch, tmp_path):
    """Exercise launcher publication/timeout without starting any child process."""
    output = tmp_path / "launch"
    value = {"resource_budget": {"worker_wall_seconds": 590., "process_peak_rss_bytes": 6 * 1024**3,
                                  "hard_termination_grace_seconds": 10.}}
    snapshot = {"file_sha256": {}, "runtime_content_hash": "sha256:synthetic-runtime"}
    monkeypatch.setattr(probe, "declaration", lambda: value)
    monkeypatch.setattr(probe, "_check_bound_inputs", lambda value: None)
    monkeypatch.setattr(probe, "source_snapshot", lambda: snapshot)
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(output), "--execute", "--case-bundle", str(tmp_path / "fake.bundle")])
    state = SimpleNamespace(waits=[], terminated=False, killed=False, returncode=0, timeout=False,
        result={"status": "completed", "runtime_content_hash": snapshot["runtime_content_hash"]},
        worker_status="completed", result_hash_override=None, raw_result_text=None,
        worker_runtime=snapshot["runtime_content_hash"], worker_not_object=False, raw_worker_text=None)
    class Process:
        @property
        def returncode(self):
            return state.returncode
        def wait(self, timeout=None):
            state.waits.append(timeout)
            if state.timeout and len(state.waits) <= 2:
                raise subprocess.TimeoutExpired("synthetic mocked child", timeout)
            return state.returncode
        def terminate(self):
            state.terminated = True
        def kill(self):
            state.killed = True
    def launch(*args, **kwargs):
        probe.pre.write_json(output / "result.json", state.result)
        status = {"status": state.worker_status,
            "result_hash": state.result_hash_override or probe.content_hash(state.result),
            "runtime_content_hash": state.worker_runtime}
        probe.pre.write_json(output / "worker-status.json", [] if state.worker_not_object else status)
        if state.raw_result_text is not None:
            (output / "result.json").write_text(state.raw_result_text)
        if state.raw_worker_text is not None:
            (output / "worker-status.json").write_text(state.raw_worker_text)
        return Process()
    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    return output, state


@pytest.mark.parametrize("failure", ["timeout", "worker_failed", "exit_nonzero", "result_hash", "runtime_hash", "result_failed",
                                     "worker_runtime_hash", "result_not_object", "worker_not_object", "malformed_result", "malformed_worker"])
def test_launcher_never_publishes_failed_or_timed_out_payload(mocked_launcher, failure):
    output, state = mocked_launcher
    if failure == "timeout":
        state.timeout = True
    elif failure == "worker_failed":
        state.worker_status = "failed"
    elif failure == "exit_nonzero":
        state.returncode = 1
    elif failure == "result_hash":
        state.result_hash_override = "sha256:changed"
    elif failure == "runtime_hash":
        state.result["runtime_content_hash"] = "sha256:other-runtime"
    elif failure == "worker_runtime_hash":
        state.worker_runtime = "sha256:other-runtime"
    elif failure == "result_not_object":
        state.result = []
    elif failure == "worker_not_object":
        state.worker_not_object = True
    elif failure == "malformed_result":
        state.raw_result_text = "{"
    elif failure == "malformed_worker":
        state.raw_worker_text = "{"
    else:
        state.result["status"] = "failed"
    with pytest.raises(SystemExit):
        probe.main()
    status = json.loads((output / "launcher-status.json").read_text())
    assert status["status"] == "failed"
    if failure == "timeout":
        assert state.waits == [590., 10., None]
        assert state.terminated and state.killed and status["parent_timeout_requested"]


def test_launcher_success_requires_matching_complete_records(mocked_launcher):
    output, state = mocked_launcher
    probe.main()
    status = json.loads((output / "launcher-status.json").read_text())
    assert status["status"] == "completed" and state.waits == [590.]
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        probe.main()
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == before
