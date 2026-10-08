"""Policy control checks: no patient fixture, model, rollout or optimizer run.

Missing arguments are intentional sentinels: refusal must precede all use of
caller inputs. Saved actual experiment metadata remains readable unchanged.
"""
import importlib
import json
from pathlib import Path
import sys

import pytest

from resectionlab.data_policy import DataPolicyError, POLICY_VERSION

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module,name,reason", [
    ("learning", "train_patient_policy", "RECORDED_EXPERIENCE_REQUIRED"),
    ("learning_ppo", "train_patient_ppo", "RECORDED_EXPERIENCE_REQUIRED"),
    ("learning", "load_policy", "WEIGHT_LINEAGE_UNVERIFIED"),
    ("spatial_policy", "imitation_loss", "RECORDED_EXPERIENCE_REQUIRED"),
    ("spatial_policy", "reinforce_loss", "RECORDED_EXPERIENCE_REQUIRED"),
    ("spatial_policy", "gradient_step", "RECORDED_EXPERIENCE_REQUIRED"),
    ("brain_extraction", "ensure_model_assets", "SYNTHETIC_MODEL_INELIGIBLE"),
    ("brain_extraction", "run_synthstrip", "SYNTHETIC_MODEL_INELIGIBLE"),
    ("imaging", "import_brain_extraction_evidence", "SYNTHETIC_MODEL_INELIGIBLE"),
    ("imaging", "create_synthetic_case", "SYNTHETIC_CASE_DISABLED"),
    ("brain_extraction", "main", "SYNTHETIC_MODEL_INELIGIBLE"),
    ("population_learning", "validate_population_checkpoint", "GENERATED_POLICY_INELIGIBLE"),
    ("population_learning", "load_frozen_population_policy", "GENERATED_POLICY_INELIGIBLE"),
    ("population_learning", "train_population_policy", "RECORDED_EXPERIENCE_REQUIRED"),
    ("population_learning", "make_analytic_population_fixture", "SYNTHETIC_CASE_DISABLED"),
    ("procedural_learning", "validate_procedural_checkpoint", "GENERATED_POLICY_INELIGIBLE"),
    ("procedural_learning", "load_frozen_procedural_policy", "GENERATED_POLICY_INELIGIBLE"),
    ("procedural_learning", "train_procedural_native_policy", "RECORDED_EXPERIENCE_REQUIRED"),
    ("procedural_learning", "train_procedural_adapted_policy", "RECORDED_EXPERIENCE_REQUIRED"),
    ("procedural_learning", "make_procedural_test_target", "SYNTHETIC_CASE_DISABLED"),
    ("procedural_learning", "make_native_procedural_fixture", "SYNTHETIC_CASE_DISABLED"),
])
def test_incompatible_public_api_refuses_before_inputs(module, name, reason):
    function = getattr(importlib.import_module("resectionlab." + module), name)
    with pytest.raises(DataPolicyError) as caught:
        function()
    assert caught.value.code == reason
    assert caught.value.policy_version == POLICY_VERSION


@pytest.mark.parametrize("module,function", [
    ("learning", "train_patient_policy"),
    ("learning_ppo", "train_patient_ppo"),
    ("population_learning", "train_population_policy"),
    ("procedural_learning", "train_procedural_native_policy"),
])
def test_training_refusal_precedes_factory_and_failure_directory(tmp_path, module, function):
    def forbidden_factory():
        pytest.fail("Simulator factory must not run")

    destination = tmp_path / "must-not-exist"
    method = getattr(importlib.import_module("resectionlab." + module), function)
    with pytest.raises(DataPolicyError):
        method(forbidden_factory, None, None, output_dir=destination, resume=True)
    assert not destination.exists()


def test_checkpoint_refusal_precedes_deserialization(monkeypatch):
    from resectionlab import learning

    def forbidden_load(*args, **kwargs):
        pytest.fail("Checkpoint deserialization must not run")

    monkeypatch.setattr(learning.torch, "load", forbidden_load)
    with pytest.raises(DataPolicyError, match="WEIGHT_LINEAGE_UNVERIFIED"):
        learning.load_policy(ROOT / "artifacts/pat05-real-geometric-learning-v1/initial.pt")


@pytest.mark.parametrize("name", [
    "run_real_patient_learning", "run_real_patient_imitation",
    "run_real_patient_visited_imitation",
])
def test_historical_workers_refuse_before_patient_or_checkpoint_reads(monkeypatch, name, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module(name)
    with pytest.raises(DataPolicyError, match="RECORDED_EXPERIENCE_REQUIRED"):
        module.worker(None, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("name", [
    "run_real_patient_learning", "run_real_patient_imitation", "run_real_patient_visited_imitation",
])
@pytest.mark.parametrize("mode", ["--execute", "--worker"])
def test_learning_cli_refuses_before_declaration_read_or_output(monkeypatch, name, mode, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module(name)
    monkeypatch.setattr(sys, "argv", [name, "--declaration", str(tmp_path / "absent.json"),
                                     "--output", str(tmp_path / "output"), mode])
    with pytest.raises(DataPolicyError, match="RECORDED_EXPERIENCE_REQUIRED"):
        module.main()
    assert not (tmp_path / "output").exists()


def test_generated_frozen_policy_is_refused(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("run_real_training_transfer")
    with pytest.raises(DataPolicyError, match="GENERATED_POLICY_INELIGIBLE"):
        module.load_frozen_policy(None, None)


@pytest.mark.parametrize("name", ["diagnose_procedural_transfer", "diagnose_feature_unit_mechanisms"])
def test_frozen_runtime_diagnostic_refuses_before_import_switch(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module(name)
    with pytest.raises(DataPolicyError, match="GENERATED_POLICY_INELIGIBLE"):
        module.diagnose(None, None)


def test_direct_desktop_training_refuses_without_session_or_patient():
    from resectionlab.desktop_bridge import BridgeSession
    with pytest.raises(DataPolicyError, match="RECORDED_EXPERIENCE_REQUIRED"):
        BridgeSession._train_patient(None, None, None, None)


def test_desktop_protocol_reports_policy_codes_and_remains_responsive(tmp_path):
    from resectionlab.data_policy import LEGACY_OPERATION_EXCLUSIONS
    from resectionlab.desktop_bridge import BridgeRuntime
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    try:
        for operation, reason in LEGACY_OPERATION_EXCLUSIONS.items():
            runtime.submit({"id": operation, "op": operation, "args": {}})
            assert runtime.wait_idle(5)
            response = [event for event in events if event["id"] == operation][-1]
            assert response["event"] == "error"
            assert response["error"]["code"] == reason
        assert not runtime.session.cases
        assert not list((tmp_path / "runs").glob("*/"))
        runtime.submit({"id": "ping", "op": "ping", "args": {}})
        assert events[-1]["event"] == "result"
    finally:
        runtime.close()


def test_cached_model_dependency_requires_ancestry_even_with_known_bytes():
    from resectionlab.data_policy import require_admitted_model, SYNTHETIC_MODEL_SHA256
    for digest in SYNTHETIC_MODEL_SHA256:
        for value in (digest, "sha256:" + digest):
            with pytest.raises(DataPolicyError, match="SYNTHETIC_MODEL_INELIGIBLE"):
                require_admitted_model(value, "cached_support")
    with pytest.raises(DataPolicyError, match="WEIGHT_LINEAGE_UNVERIFIED"):
        require_admitted_model("unknown-lineage", "cached_support")


def test_historical_result_metadata_stays_inspectable():
    result = json.loads((ROOT / "artifacts/prepared-training-planner-comparison-v2/summary.json").read_text())
    assert result["optimizer_updates"] == 0
    assert result["patients_prescribed"] == 6
    assert result["complete_comparisons"] == 4
    assert [p["subject"] for p in result["patients"] if p["status"] == "historical_support_block"] == [
        "sub-PAT16", "sub-PAT20"]
