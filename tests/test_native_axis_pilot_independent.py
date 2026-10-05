"""Independent pilot declaration and orchestration gates; no public execution."""
import hashlib
import json
from pathlib import Path
import copy
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "manifests/experiments/native-axis-raw-update-pilot-v1.json"
sys.path.insert(0, str(ROOT / "scripts"))
import run_native_axis_pilot as runner
from test_native_axis_pilot import completed, fixture, guard


def declaration():
    return json.loads(DECLARATION.read_text())


def test_declared_pilot_references_match_actual_reviewed_bytes():
    declared = declaration()
    body = {key:value for key,value in declared.items() if key != "declaration_content_hash"}
    digest = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert declared["declaration_content_hash"] == digest
    for value in declared["references"].values():
        if isinstance(value, dict) and "file_sha256" in value:
            assert hashlib.sha256((ROOT / value["path"]).read_bytes()).hexdigest() == value["file_sha256"]


def test_pilot_retains_exact_v2_physics_worlds_and_honest_scope():
    declared = declaration()
    v2 = json.loads((ROOT / declared["references"]["preflight_declaration"]["path"]).read_text())
    reference = json.loads((ROOT / declared["references"]["physical_and_world_declaration"]["path"]).read_text())
    model = declared["frozen_model"]
    completed = json.loads((ROOT / "artifacts/preflight/native-axis-v2/profile/model.json").read_text())
    initial = json.loads((ROOT / "artifacts/preflight/native-axis-v2/profile/untrained-policy-freeze.json").read_text())
    assert model["decision_model_hash"] == completed["decision_model_hash"]
    assert model["proposal_model_hash"] == completed["proposal_model_hash"]
    assert model["initial_policy_hash"] == initial["policy_hash"]
    assert model["native_config_hash"] == v2["native_configuration"]["fingerprint"]
    assert model["action_model"] == v2["action_model"]
    assert model["case_hash"] == reference["target"]["semantic_hash"]
    assert model["planning_hash"] == reference["target"]["planning_hash"]
    assert model["physical_reward"] == reference["target"]["reward"]
    assert model["partial_contact_weight"] == reference["target"]["partial_contact_weight"]
    assert set(declared["world_partitions"]) == {"optimization", "selection"}
    assert all(panel == reference["target"]["world_partitions"][role]
               for role, panel in declared["world_partitions"].items())
    assert declared["optimization_episode_seeds"] == declared["world_partitions"]["optimization"]["seeds"][:2]
    assert declared["interpretation"]["clinical_deficit_probability"] is None
    assert declared["interpretation"]["efficacy_comparison"] is False
    assert declared["interpretation"]["motor_language_assessed"] is False
    assert declared["interpretation"]["clinical_use_validated"] is False


def test_one_update_pilot_has_exact_counts_and_separate_time_budgets():
    declared = declaration()
    config, counts = declared["training_config"], declared["expected_denominators"]
    assert config["seed"] == 11 and config["hidden_features"] == 16
    assert config["max_gradient_steps"] == counts["adam_steps"] == 1
    assert config["episodes_per_update"] == counts["complete_optimization_episodes"] == 2
    assert counts["initial_selection_episodes"] == counts["updated_selection_episodes"] == 2
    assert counts["completed_episode_audit_receipts"] == 6
    assert counts["shape_probe_resets"] == 1 and counts["shape_probe_transitions"] == 0
    assert counts["final_evaluation_worlds"] == counts["stress_worlds"] == 0
    assert config["max_wall_seconds"] == 300.
    assert declared["resource_budget"]["worker_wall_seconds"] == 600.
    assert declared["resource_budget"]["process_peak_rss_bytes"] == 6 * 1024**3
    assert declared["budget_rationale"]["is_measured_generic_learner_time"] is False
    assert declared["budget_rationale"]["is_guaranteed_bound"] is False


def test_forward_capture_is_distinct_from_checkpoint_joins_and_derived_probability():
    logging = declaration()["logging_implementation_gate"]
    actual = set(logging["actual_record_fields"])
    assert {"inputs.action_ids", "inputs.action_features", "inputs.state_features",
            "inputs.action_mask", "logits", "value", "selected_index", "selected_action_id",
            "role", "update", "episode", "panel", "forward_evaluated"} <= actual
    assert "policy_hash" not in actual and "profile_hash" not in actual and "probabilities" not in actual
    assert "latest checkpoint at update1" in logging["joined_provenance_fields"]["policy_hash"]
    assert "label derived" in logging["optional_derived_fields"]["probabilities"]
    assert logging["no_extra_policy_forwards_or_random_draws"]
    assert logging["no_extra_simulator_reads_or_transitions"]


def reseal(journal):
    from resectionlab.native_axis_accounting import _hash
    journal["receipt_hash"] = _hash({key: value for key, value in journal.items() if key != "receipt_hash"})


@pytest.mark.parametrize("attack", ["wrong_world", "final_role", "shape_probe_transition", "extra_episode",
                                    "missing_decision", "repeated_decision", "transition_before_decision",
                                    "unreturned", "native_counter", "unrecorded_forward"])
def test_resealed_incomplete_or_misjoined_execution_cannot_be_selected(completed, attack):
    journal = copy.deepcopy(completed["accounting"])
    episode = journal["episodes"][1]
    decision = next(e for e in journal["events"] if e["kind"] == "decision")
    transition = next(e for e in journal["events"] if e["kind"] == "transition")
    if attack == "wrong_world":
        episode["seed"] += 1
    elif attack == "final_role":
        episode["role"] = decision["role"] = transition["role"] = "final_evaluation"
        decision["payload"]["role"] = "final_evaluation"
    elif attack == "shape_probe_transition":
        journal["events"].append({**transition, "episode": 0})
    elif attack == "extra_episode":
        journal["episodes"].append({**episode, "episode": 7})
    elif attack == "missing_decision":
        journal["events"].remove(decision)
    elif attack == "repeated_decision":
        journal["events"].append(copy.deepcopy(decision))
    elif attack == "transition_before_decision":
        first, second = journal["events"].index(decision), journal["events"].index(transition)
        journal["events"][first], journal["events"][second] = transition, decision
    elif attack == "unreturned":
        transition["returned_to_learner"] = False
    elif attack == "native_counter":
        journal["totals"]["selection"]["native_commits"] += 1
    else:
        decision["payload"]["inputs"]["actor_action_features"]["values"][0][0] += 1
    reseal(journal)
    with pytest.raises(ValueError):
        runner.validate_completion(completed["declaration"], completed["result"], journal, completed["checkpoints"])


@pytest.mark.parametrize("field,value", [
    ("source_case_hash", "another-source"), ("action_count", 999),
    ("unsupported_source_tissue_volume_mm3", 1.), ("failures", ["shaft_collision"]),
    ("first_failed_action", 0), ("feasible", "true"),
    ("claimed_source_tissue_volume_mm3", 99999.), ("contained_source_tissue_volume_mm3", 99999.),
])
def test_geometry_certificate_cannot_contradict_saved_native_history(completed, field, value):
    frozen = json.loads((completed["output"] / "history-freeze.json").read_text())
    audits = json.loads((completed["output"] / "native-audits.json").read_text())
    assert field in audits[0]["audit"], "attack must target an actual receipt field"
    audits[0]["audit"][field] = value
    with pytest.raises(ValueError):
        runner.validate_audits(frozen, audits)


def test_updated_decisions_join_latest_weights_even_if_initial_is_selected(completed):
    frozen = runner.validate_completion(completed["declaration"], completed["result"],
                                        completed["accounting"], completed["checkpoints"])
    # This fixture's unchanged deterministic return selects the earliest panel.
    assert frozen["selected_panel"] == 0
    assert completed["checkpoints"]["initial_policy_hash"] != completed["checkpoints"]["latest_policy_hash"]
    assert all(d["policy_hash"] == completed["checkpoints"]["latest_policy_hash"]
               for episode in frozen["episodes"] if episode["update"] == 1 for d in episode["decisions"])


@pytest.mark.parametrize("attack", ["nonmonotonic_ids", "sampling_as_argmax", "argmax_as_sampling",
                                    "string_forward_flag", "wrong_observer_version", "false_source_dtype",
                                    "false_logits_dtype"])
def test_trace_metadata_must_preserve_actual_decision_semantics(completed, attack):
    journal = copy.deepcopy(completed["accounting"])
    decisions = [row for row in journal["events"] if row["kind"] == "decision"]
    if attack == "nonmonotonic_ids":
        mapping = {row["decision_id"]: len(decisions) - index for index, row in enumerate(decisions)}
        for event in journal["events"]:
            if event["kind"] in {"decision", "transition"}:
                event["decision_id"] = mapping[event["decision_id"]]
    elif attack == "sampling_as_argmax":
        row = next(row for row in decisions if row["role"] == "optimization")
        row["payload"]["decision_rule"] = "deterministic_argmax"
    elif attack == "argmax_as_sampling":
        decisions[0]["payload"]["decision_rule"] = "sampled_categorical"
    elif attack == "string_forward_flag":
        decisions[0]["payload"]["forward_evaluated"] = "true"
    elif attack == "wrong_observer_version":
        decisions[0]["payload"]["version"] = "reconstructed_after_run"
    elif attack == "false_source_dtype":
        decisions[0]["payload"]["inputs"]["source_action_features"]["dtype"] = "int8"
    else:
        decisions[0]["payload"]["logits"]["dtype"] = "int8"
    reseal(journal)
    with pytest.raises(ValueError):
        runner.validate_completion(completed["declaration"], completed["result"], journal, completed["checkpoints"])


def setup_replay_of_saved_fixture(completed, tmp_path):
    """Reuse completed synthetic evidence; injected trainer performs no updates."""
    case, base, panels, declared = fixture(tmp_path)
    output = tmp_path / "publication"

    def saved_trainer(accounting, *, output_dir, **kwargs):
        shutil.copytree(completed["output"] / "learner", output_dir)
        shutil.copy2(completed["output"] / "accounting.json", output / "accounting.json")

    resource = guard()
    return (lambda: runner.run_pilot(case, base, panels.optimization, panels.selection, declared,
        output, resource, source_check=lambda: None, trainer=saved_trainer)), output, resource


@pytest.mark.parametrize("artifact", ["learner/result.json", "accounting.json", "history-freeze.json"])
def test_late_candidate_export_cannot_change_authenticated_inputs(completed, tmp_path, monkeypatch, artifact):
    execute, output, _ = setup_replay_of_saved_fixture(completed, tmp_path)
    original = runner.write_json

    def mutate_after_export(path, value):
        result = original(path, value)
        if path.name == "candidate-record.json":
            target = output / artifact
            payload = json.loads(target.read_text())
            payload["late_mutation"] = True
            target.write_text(json.dumps(payload))
        return result

    monkeypatch.setattr(runner, "write_json", mutate_after_export)
    with pytest.raises(ValueError):
        execute()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["eligible_candidate_count"] == 0
    assert (output / "candidate-record.json").exists()  # retained diagnostic only
    assert (output / "failure-accounting.json").exists()


@pytest.mark.parametrize("reason", ["worker_wall_budget", "process_peak_rss_budget", "parent_cancellation"])
def test_late_resource_limit_cannot_publish(completed, tmp_path, monkeypatch, reason):
    execute, output, resource = setup_replay_of_saved_fixture(completed, tmp_path)
    original = runner.write_json

    def cancel_after_export(path, value):
        result = original(path, value)
        if path.name == "candidate-record.json":
            resource.cancel(reason)
        return result

    monkeypatch.setattr(runner, "write_json", cancel_after_export)
    with pytest.raises(InterruptedError):
        execute()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["eligible_candidate_count"] == 0
    assert status["resource"]["cancellation_reason"] == reason


def test_runtime_mismatch_fails_before_loading_any_patient(tmp_path, monkeypatch):
    import resectionlab.imaging as imaging
    runner.write_json(tmp_path / "launch-source.json", {"runtime_content_hash": "declared-runtime"})
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"runtime_content_hash": "changed-runtime"})
    monkeypatch.setattr(imaging, "load_case", lambda *a, **kw: pytest.fail("runtime gate must precede MRI load"))
    with pytest.raises(ValueError, match="differs from launch source"):
        runner.public_worker(tmp_path, tmp_path / "not-loaded")
    status = json.loads((tmp_path / "worker-status.json").read_text())
    assert status["status"] == "failed" and status["stage"] == "preparation"
    assert status["eligible_candidate_count"] == 0
    assert status["resource"]["observed_peak_rss_bytes"] > 0


def test_parent_timeout_remains_failure_when_worker_finishes_during_grace(tmp_path, monkeypatch):
    output = tmp_path / "launcher"

    class CompletedDuringGrace:
        returncode = 0

        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.terminated = False

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                # No executable is started; construct only minimal completed authorities.
                runner.write_json(output / "pilot/candidate-record.json", {"private_test": True})
                digest = runner.file_hash(output / "pilot/candidate-record.json")
                runner.write_json(output / "pilot/status.json", {"status": "completed",
                    "eligible_candidate_count": 1, "candidate_record_sha256": digest})
                runner.write_json(output / "worker-status.json", {"status": "completed",
                    "eligible_candidate_count": 1, "candidate_record_sha256": digest,
                    "pilot_status_sha256": runner.file_hash(output / "pilot/status.json")})
                raise runner.subprocess.TimeoutExpired("synthetic-worker", timeout)
            assert self.terminated
            return 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            pytest.fail("grace-completed worker must not be hard-killed")

    original_popen = runner.subprocess.Popen

    def controlled_popen(command, *args, **kwargs):
        if isinstance(command, list) and len(command) > 1 and command[1].endswith("run_native_axis_pilot.py"):
            return CompletedDuringGrace(command, *args, **kwargs)
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", controlled_popen)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output", str(output), "--execute", "--case-bundle", str(tmp_path / "not-read")])
    assert error.value.code == 1
    status = json.loads((output / "launcher-status.json").read_text())
    assert status["status"] == "failed" and status["eligible_candidate_count"] == 0
    assert status["parent_timeout_requested"] and status["worker_returncode"] == 0
    assert status["hard_killed"] is False
    assert (output / "pilot/candidate-record.json").exists()  # retained, no launcher authority
