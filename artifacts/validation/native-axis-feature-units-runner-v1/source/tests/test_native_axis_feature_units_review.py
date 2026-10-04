"""Independent copied-record attacks; actual updates use tiny owner fixtures only."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from test_native_axis_feature_units import completed, fixture, guard, make_authorities, runner
from resectionlab.worlds import content_hash


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _seal_contract(contract):
    publication = {"contract_hash", "initial_checkpoint_hash", "shared_checkpoint_hash", "code_sha256",
                   "hardware", "clinical_deficit_probability", "final_evaluation_used_for_optimization"}
    contract["contract_hash"] = content_hash({k:v for k,v in contract.items() if k not in publication})


@pytest.mark.parametrize("attack", ["stale_algorithm_hash", "coherent_units", "coherent_partial_semantics"])
def test_initial_contract_rejects_stale_or_coherently_resealed_semantics(completed, tmp_path, attack):
    item = completed["FEATURE_UNITS"]
    folder = tmp_path / "learner"
    shutil.copytree(item["output"] / "learner", folder)
    path = folder / "contract.json"
    contract = json.loads(path.read_text())
    if attack == "stale_algorithm_hash":
        contract["algorithm"] = "different_unexecuted_algorithm"
    else:
        schema = contract["axis_observation_contract"]
        if attack == "coherent_units":
            schema["action_feature_units"][5] = "cm"
        else:
            schema["partial_contact_semantics"] = "all contact including already charged removed tissue"
        schema["contract_hash"] = content_hash({k:v for k,v in schema.items() if k != "contract_hash"})
        _seal_contract(contract)
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        checkpoint["contract_hash"] = contract["contract_hash"]
        torch.save(checkpoint, folder / "checkpoint.pt")
    _write(path, contract)
    with pytest.raises(ValueError):
        runner.initial_receipt(folder, item["declaration"])


def test_one_saved_adam_state_cannot_stand_for_complete_optimizer(completed, tmp_path):
    item = completed["FEATURE_UNITS"]
    folder = tmp_path / "learner"
    shutil.copytree(item["output"] / "learner", folder)
    path = folder / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    states = checkpoint["optimizer"]["state"]
    assert len(states) == 8  # Actual actor and critic parameters all participated.
    checkpoint["optimizer"]["state"] = {next(iter(states)): next(iter(states.values()))}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        runner.checkpoint_receipt(folder, item["declaration"])


@pytest.mark.parametrize("attack", ["nonfinite", "shape", "negative_squared_moment", "second_step"])
def test_optimizer_evidence_is_complete_finite_and_exactly_one_update(completed, tmp_path, attack):
    item = completed["RAW"]
    folder = tmp_path / "learner"
    shutil.copytree(item["output"] / "learner", folder)
    path = folder / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = next(iter(checkpoint["optimizer"]["state"].values()))
    if attack == "nonfinite":
        state["exp_avg"].flatten()[0] = float("nan")
    elif attack == "shape":
        state["exp_avg"] = state["exp_avg"].flatten()[:1]
    elif attack == "negative_squared_moment":
        state["exp_avg_sq"].flatten()[0] = -1.
    else:
        state["step"] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        runner.checkpoint_receipt(folder, item["declaration"])


@pytest.mark.parametrize("attack", ["world_order", "final_role", "world_generator", "model", "profile"])
def test_declared_mismatch_precedes_any_optimizer_or_scored_episode(tmp_path, monkeypatch, attack):
    case, base, panels, arm = fixture(tmp_path, "FEATURE_UNITS")
    if attack == "world_order":
        arm["world_partitions"]["optimization"]["seeds"].reverse()
    elif attack == "final_role":
        arm["world_partitions"]["selection"]["role"] = "final_evaluation"
    elif attack == "world_generator":
        arm["world_partitions"]["selection"]["generator"]["translation_scale_mm"] = (1., 0., 0.)
    elif attack == "model":
        arm["frozen_model"]["proposal_model_hash"] = "undeclared-proposals"
    else:
        arm["frozen_model"]["input_profile_manifest"]["action_divisors"][1] = 1.
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("Mismatch created Adam"))
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="before optimizer"):
        runner.run_pilot(case, base, panels.optimization, panels.selection, arm, out, guard(),
            source_check=lambda: None, trainer=lambda *a, **k: pytest.fail("Mismatch invoked trainer"))
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed" and status["eligible_candidate_count"] == 0
    assert not (out / "learner").exists()


def _rebind_authorities(folder):
    """Constructed checksum repair, never a claim of authenticated execution."""
    path = folder / "pilot/status.json"
    status = json.loads(path.read_text())
    status["candidate_record_sha256"] = runner.file_hash(folder / "pilot/candidate-record.json")
    _write(path, status)
    path = folder / "worker-status.json"
    worker = json.loads(path.read_text())
    worker["pilot_status_sha256"] = runner.file_hash(folder / "pilot/status.json")
    worker["candidate_record_sha256"] = status["candidate_record_sha256"]
    _write(path, worker)
    _rebind_launcher(folder)


def _rebind_launcher(folder):
    path = folder / "launcher-status.json"
    launcher = json.loads(path.read_text())
    launcher["worker_status_sha256"] = runner.file_hash(folder / "worker-status.json")
    _write(path, launcher)


@pytest.mark.parametrize("attack", ["missing_receipt", "false_geometry", "wrong_source"])
def test_pair_rechecks_all_twelve_audits_after_coherent_authority_reseal(completed, tmp_path, monkeypatch, attack):
    make_authorities(completed, tmp_path, monkeypatch)
    folder = tmp_path / "FEATURE_UNITS"
    path = folder / "pilot/native-audits.json"
    audits = json.loads(path.read_text())
    if attack == "missing_receipt":
        audits.pop()
    elif attack == "false_geometry":
        audits[-1]["audit"]["feasible"] = False
    else:
        audits[-1]["audit"]["source_case_hash"] = "another-synthetic-source"
    _write(path, audits)
    path = folder / "pilot/candidate-record.json"
    candidate = json.loads(path.read_text())
    candidate["audits"] = audits
    _write(path, candidate)
    _rebind_authorities(folder)
    assert runner.launch_complete(folder, 0, False)
    with pytest.raises(ValueError, match="audit"):
        runner.pair_receipt(tmp_path, runner.load_declaration())


@pytest.mark.parametrize("attack", ["cancelled", "rss", "elapsed", "budget", "missing_resource",
    "nonfinite_elapsed", "time_requested", "hard_killed"])
def test_completed_authority_cannot_contradict_worker_resource_bounds(completed, tmp_path, monkeypatch, attack):
    make_authorities(completed, tmp_path, monkeypatch)
    declaration = runner.load_declaration()
    folder = tmp_path / "FEATURE_UNITS"
    path = folder / "worker-status.json"
    worker = json.loads(path.read_text())
    resource = {"budget": copy.deepcopy(declaration["fixed_protocol"]["resource_budget_per_arm"]),
        "elapsed_seconds": 1., "observed_peak_rss_bytes": 1024, "cancellation_reason": None,
        "cancellation_callback_calls": 1, "cancellation_callback_seconds": 0.,
        "budget_request_to_receipt_seconds": None}
    if attack == "cancelled":
        resource["cancellation_reason"] = "parent_cancellation"
    elif attack == "rss":
        resource["observed_peak_rss_bytes"] = resource["budget"]["process_peak_rss_bytes"] + 1
    elif attack == "elapsed":
        resource["elapsed_seconds"] = resource["budget"]["worker_wall_seconds"] + 1
    elif attack == "budget":
        resource["budget"]["worker_wall_seconds"] *= 10
    elif attack == "nonfinite_elapsed":
        resource["elapsed_seconds"] = "NaN"
    elif attack == "time_requested":
        resource["budget_request_to_receipt_seconds"] = .1
    worker["resource"] = resource
    if attack == "missing_resource":
        del worker["resource"]
    _write(path, worker)
    _rebind_launcher(folder)
    if attack == "hard_killed":
        path = folder / "launcher-status.json"
        launcher = json.loads(path.read_text())
        launcher["hard_killed"] = True
        _write(path, launcher)
    with pytest.raises(ValueError):
        runner.pair_receipt(tmp_path, declaration)


def test_pair_measures_initial_and_latest_separately_and_ties_keep_initial(completed, tmp_path, monkeypatch):
    make_authorities(completed, tmp_path, monkeypatch)
    result = runner.pair_receipt(tmp_path, runner.load_declaration())
    assert result["completed_episode_audit_receipts"] == 12
    hashes = []
    for profile in ("RAW", "FEATURE_UNITS"):
        arm = result["arms"][profile]
        assert arm["gradient_steps"] == 1 and arm["actor_parameters_changed"]
        assert arm["latest_minus_initial"] == arm["latest_return"] - arm["initial_return"]
        assert arm["selected_minus_initial"] == arm["selected_return"] - arm["initial_return"]
        if arm["latest_return"] == arm["initial_return"]:
            assert arm["selected_phase"] == "initial"
            assert arm["checkpoint_identities"]["latest_policy_hash"] != arm["checkpoint_identities"]["selected_policy_hash"]
        hashes.append(arm["initialization"]["initial_trainable_parameter_hash"])
        assert arm["online_seconds"] >= sum(p["panel_elapsed_seconds"] for p in completed[profile]["result"]["selection_history"])
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("attack", ["source_after_export", "cancel_after_export"])
def test_final_export_failure_preserves_payload_without_publication_authority(completed, tmp_path, monkeypatch, attack):
    item = completed["RAW"]
    case, base, panels, arm = fixture(tmp_path, "RAW")
    out, resource = tmp_path / "attempt", guard()
    original_write = runner.write_json
    def write(path, value):
        original_write(path, value)
        if path.name == "candidate-record.json" and attack == "cancel_after_export":
            resource.cancel("controlled post-export cancellation")
    monkeypatch.setattr(runner, "write_json", write)
    def trainer(accounting, *, output_dir, **kwargs):
        # Reuse actual tiny evidence solely to test publication. These two
        # factories perform no policy/gradient/transition or world reset.
        first = accounting.recorded_factory()
        shutil.copytree(item["output"] / "learner", output_dir)
        second = accounting.recorded_factory()  # Actual initializer gate.
        original_write(accounting.receipt_path, item["journal"])
        assert first is not second
    original_audits = json.loads((item["output"] / "native-audits.json").read_text())
    by_history = {row["history_hash"]: row["audit"] for row in original_audits}
    def auditor(_case, _tools, history, **kwargs):
        value = copy.deepcopy(by_history[content_hash(history)])
        return SimpleNamespace(to_dict=lambda: value)
    def source_check():
        if attack == "source_after_export" and (out / "candidate-record.json").exists():
            raise RuntimeError("controlled post-export source change")
    with pytest.raises((RuntimeError, InterruptedError), match="controlled"):
        runner.run_pilot(case, base, panels.optimization, panels.selection, arm, out, resource,
            source_check=source_check, trainer=trainer, auditor=auditor)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed" and status["eligible_candidate_count"] == 0
    assert (out / "candidate-record.json").exists() and (out / "failure-accounting.json").exists()
    assert not runner.launch_complete(tmp_path, 0, False)
