"""One-update orchestration on a constructed cube; never a patient experiment."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_native_axis_pilot as runner
from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.worlds import WorldGeneratorConfig, generate_partitions


def fixture(tmp_path):
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros_like(tissue)
    target[1:6, 1:6, 2:7] = True
    case = CaseData("axis-pilot-test", tissue.astype(float), {"target": target}, np.eye(4),
        (SourceRef("analytic", "simulated:pilot", provenance="simulated"),), brain_mask=tissue)
    cfg = NativeResectionConfig(tissue, target.astype(np.int16), np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        case.semantic_hash, "explicit synthetic tissue", case_id=case.case_id)
    base = AxisColumnNativeSimulator(cfg, proposal_config=AxisColumnProposalConfig(((0, 0),), 2), max_steps=3)
    panels = generate_partitions(case.semantic_hash, WorldGeneratorConfig(), 71,
        optimization=3, selection=2, final_evaluation=1, stress=1, planning_hash=case.planning_hash)
    declared = runner.load_declaration()
    # Explicit private test contract: never passed to the pinned public entry.
    declared["pilot_id"] = "private_synthetic_orchestration_test"
    declared["declaration_content_hash"] = "test-only"
    declared["frozen_model"].update(case_hash=case.semantic_hash, planning_hash=case.planning_hash,
        decision_model_hash=base.decision_model_hash, native_config_hash=cfg.fingerprint)
    declared["world_partitions"] = {p.role.value: p.to_dict() for p in (panels.optimization, panels.selection)}
    declared["optimization_episode_seeds"] = list(panels.optimization.seeds[:2])
    return case, base, panels, declared


def guard():
    return runner.physical.ResourceGuard(runner.physical.PreflightBudget(120, 6 * 1024**3, 1))


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path = tmp_path_factory.mktemp("actual-axis-pilot")
    case, base, panels, declared = fixture(path)
    output = path / "pilot"
    status = runner.run_pilot(case, base, panels.optimization, panels.selection, declared,
        output, guard(), source_check=lambda: None)
    return {"output": output, "declaration": declared, "status": status,
        "result": json.loads((output / "learner/result.json").read_text()),
        "accounting": json.loads((output / "accounting.json").read_text()),
        "checkpoints": runner.checkpoint_receipt(output / "learner", declared)}


def test_actual_one_update_six_histories_and_shape_probe(completed):
    status = completed["status"]
    assert status["status"] == "completed" and status["eligible_candidate_count"] == 1
    assert status["completed_episode_audit_receipts"] == 6
    result = completed["result"]
    assert result["gradient_steps"] == 1 and result["completed_episodes"] == 2
    assert result["actor_parameters_changed"]
    frozen = runner.validate_completion(completed["declaration"], result,
        completed["accounting"], completed["checkpoints"])
    assert frozen["shape_probe"]["episode"] == 0
    assert len(frozen["episodes"]) == 6
    assert {e["role"] for e in frozen["episodes"]} == {"optimization", "selection"}
    assert not frozen["candidate_eligible"] and not frozen["final_worlds_used"]
    for episode in frozen["episodes"]:
        expected = result["initial_checkpoint_hash" if episode["update"] == 0 else "latest_checkpoint_hash"]
        assert all(d["policy_hash"] == expected for d in episode["decisions"])
    record = json.loads((completed["output"] / "candidate-record.json").read_text())
    assert len(record["timing"]["factory_clone_seconds"]) == 7
    runner.validate_audits(frozen, record["audits"])


@pytest.mark.parametrize("failure", ["incomplete_panel", "critic_only", "zero_actor_gradient", "counter", "initial_hash"])
def test_incomplete_or_non_actor_update_is_not_a_candidate(completed, failure):
    result = copy.deepcopy(completed["result"])
    journal = copy.deepcopy(completed["accounting"])
    if failure == "incomplete_panel":
        result["selection_history"].pop()
    elif failure == "critic_only":
        result["actor_parameters_changed"] = False
    elif failure == "zero_actor_gradient":
        result["optimization_history"][0]["actor_gradient_norm_after_clip"] = 0.
    elif failure == "counter":
        result["optimization_environment_steps"] += 1
    else:
        result["initial_checkpoint_hash"] = "changed"
    with pytest.raises(ValueError):
        runner.validate_completion(completed["declaration"], result, journal, completed["checkpoints"])


def test_each_history_requires_its_own_receipt_even_if_certificate_reused(completed):
    frozen = json.loads((completed["output"] / "history-freeze.json").read_text())
    audits = json.loads((completed["output"] / "native-audits.json").read_text())
    with pytest.raises(ValueError, match="Six"):
        runner.validate_audits(frozen, audits[:-1])
    audits[-1]["history_hash"] = "wrong"
    with pytest.raises(ValueError, match="audit failed"):
        runner.validate_audits(frozen, audits)


def test_default_cli_does_not_construct_simulator_or_train(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "public_worker", lambda *a: pytest.fail("default must not execute"))
    output = tmp_path / "declaration"
    runner.main(["--output", str(output)])
    assert json.loads((output / "launcher-status.json").read_text())["status"] == "declared_not_executed"
    assert not (output / "frozen-source").exists()
    with pytest.raises(FileExistsError):
        runner.main(["--output", str(output)])


def test_source_capture_includes_actual_pilot_and_helper_but_not_mutable_results():
    snapshot = runner.source_snapshot()
    paths = snapshot["file_sha256"]
    assert "scripts/run_native_axis_pilot.py" in paths
    assert "scripts/preflight_native_axis.py" in paths
    assert str(runner.DECLARATION_PATH) in paths
    assert "src/resectionlab/native_axis_accounting.py" in paths
    assert not any(path.startswith("artifacts/") for path in paths)


def test_declared_reference_change_rejected(tmp_path):
    import shutil
    manifest = runner.load_declaration()
    names = {str(runner.DECLARATION_PATH)} | {v["path"] for v in manifest["references"].values()
        if isinstance(v, dict) and "path" in v}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    runner.load_declaration(tmp_path)
    (tmp_path / manifest["references"]["preflight_cost_report"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="reference changed"):
        runner.load_declaration(tmp_path)


def test_parent_timeout_or_payload_drift_invalidates_otherwise_complete(completed, tmp_path):
    import shutil
    shutil.copytree(completed["output"], tmp_path / "pilot")
    status = json.loads((tmp_path / "pilot/status.json").read_text())
    runner.write_json(tmp_path / "worker-status.json", {"status": "completed", "eligible_candidate_count": 1,
        "pilot_status_sha256": runner.file_hash(tmp_path / "pilot/status.json"),
        "candidate_record_sha256": status["candidate_record_sha256"]})
    assert runner.launch_complete(tmp_path, 0, False)
    assert not runner.launch_complete(tmp_path, 0, True)
    assert not runner.launch_complete(tmp_path, 1, False)
    with (tmp_path / "pilot/candidate-record.json").open("a") as stream:
        stream.write(" ")
    assert not runner.launch_complete(tmp_path, 0, False)


def _retained_tiny_trainer(completed):
    """Reuse authenticated tiny outputs only for publication side-effect attacks."""
    import shutil
    def trainer(accounting, *, output_dir, **kwargs):
        shutil.copytree(completed["output"] / "learner", output_dir)
        runner.write_json(accounting.receipt_path, completed["accounting"])
    return trainer


@pytest.mark.parametrize("failure", ["candidate_export", "source_after_export", "resource_after_export"])
def test_publication_failure_never_leaves_completed_authority(completed, tmp_path, monkeypatch, failure):
    case, base, panels, declared = fixture(tmp_path)
    output = tmp_path / "attempt"
    resource = guard()
    original = runner.write_json
    def write(path, data):
        if failure == "candidate_export" and path.name == "candidate-record.json":
            raise OSError("controlled candidate export failure")
        original(path, data)
        if failure == "resource_after_export" and path.name == "candidate-record.json":
            resource.cancel("controlled post-export cancellation")
    monkeypatch.setattr(runner, "write_json", write)
    def source_check():
        if failure == "source_after_export" and (output / "candidate-record.json").exists():
            raise RuntimeError("controlled source change after export")
    with pytest.raises((OSError, RuntimeError, InterruptedError), match="controlled|cancel"):
        runner.run_pilot(case, base, panels.optimization, panels.selection, declared, output,
            resource, source_check=source_check, trainer=_retained_tiny_trainer(completed))
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["eligible_candidate_count"] == 0
    assert status["expected_complete_episodes"] == 6
    assert (output / "accounting.json").exists()
    assert not runner.launch_complete(tmp_path, 0, False)


def test_preparation_failure_has_worker_resource_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_declaration", lambda: (_ for _ in ()).throw(ValueError("controlled declaration failure")))
    with pytest.raises(ValueError, match="controlled declaration"):
        runner.public_worker(tmp_path, tmp_path / "not-loaded")
    status = json.loads((tmp_path / "worker-status.json").read_text())
    assert status["stage"] == "preparation" and status["eligible_candidate_count"] == 0
    assert status["resource"]["observed_peak_rss_bytes"] > 0
    original = (tmp_path / "worker-status.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.public_worker(tmp_path, tmp_path / "not-loaded")
    assert (tmp_path / "worker-status.json").read_bytes() == original
