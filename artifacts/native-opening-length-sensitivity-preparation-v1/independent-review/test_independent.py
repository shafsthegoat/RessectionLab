"""Pure DTO and refusal controls; no experiment checkpoint/forward/engine."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_native_opening_length_sensitivity as runner


def forbidden(*args, **kwargs):
    raise AssertionError("Independent review forbids patient, native, model or gradient execution")


@pytest.fixture(autouse=True)
def guard(monkeypatch):
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_proposals import PreparedNominalCavityProposer
    from resectionlab.spatial_policy import SpatialPolicy
    import resectionlab.imaging as imaging
    monkeypatch.setattr(NativeSpatialTask, "__init__", forbidden)
    monkeypatch.setattr(NativeResectionEngine, "__init__", forbidden)
    monkeypatch.setattr(PreparedNominalCavityProposer, "propose", forbidden)
    monkeypatch.setattr(imaging, "load_case", forbidden)
    monkeypatch.setattr(SpatialPolicy, "forward", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(runner.shared, "read_authorities", forbidden)
    monkeypatch.setattr(runner, "load_frozen_checkpoint", forbidden)


@pytest.fixture
def prepared():
    records, baselines = runner.read_inputs()
    original = runner.reconstruct_root(records, baselines)
    return original, runner.intervene(original), baselines


@pytest.mark.parametrize("failed_name", ["initial", "RL256"])
def test_terminal_failure_closes_active_slot_preserves_prior(monkeypatch, tmp_path, prepared, failed_name):
    original, changed, baselines = prepared
    monkeypatch.setattr(runner, "validate", lambda record: ({}, baselines))
    monkeypatch.setattr(runner, "reconstruct_root", lambda *args: original)
    def fake_forward(name, original, changed, baseline, ledger):
        ledger["checkpoint_attempts"].append(name)
        if name == failed_name:
            raise ValueError("injected checkpoint load refusal")
        # Pure controller completion, no checkpoint/model was loaded or called.
        ledger["forward_attempts"].append(name)
        ledger["completed_forwards"].append(name)
        return {"margin_change": 0., "stop_probability": .2, "stop_logit_change": 0.,
                "highest_ranked_id_not_executed": "STOP", "scope": "test-controller-only"}
    monkeypatch.setattr(runner, "forward_once", fake_forward)
    with pytest.raises(ValueError, match="injected checkpoint"):
        runner.worker({}, tmp_path, "0" * 64)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    expected_attempts = ["initial"] if failed_name == "initial" else ["initial", "RL256"]
    expected_completed = [] if failed_name == "initial" else ["initial"]
    assert result["ledger"]["checkpoint_attempts"] == expected_attempts
    assert result["ledger"]["completed_forwards"] == expected_completed
    assert result["methods"][failed_name] == {"status": "failed", "outputs": None}
    if failed_name == "initial":
        assert result["methods"]["RL256"] == {"status": "not_started", "outputs": None}
        assert not (tmp_path / "initial.json").exists()
    else:
        assert result["methods"]["initial"]["status"] == "complete"
        assert result["methods"]["initial"]["outputs"] == "initial.json"
        assert json.loads((tmp_path / "initial.json").read_text())["scope"] == "test-controller-only"
    assert not (tmp_path / "RL256.json").exists()


def test_only_four_descriptors_changed_all_other_bytes_and_identities_equal(prepared):
    original, changed, _ = prepared
    assert original.fingerprint == runner.ROOT_OBSERVATION_HASH
    assert original.source_id == changed.source_id == runner.SOURCE_HASH
    assert np.argwhere(original.action_geometry != changed.action_geometry).tolist() == [[1, 12], [2, 12], [3, 12], [4, 12]]
    assert original.action_geometry[1:, 12].tolist() == [2.2, 12., 2.2, 12.]
    assert changed.action_geometry[1:, 12].tolist() == [120.] * 4
    assert changed.action_geometry[0].tobytes() == original.action_geometry[0].tobytes()
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm", "state_features", "action_mask"):
        a, b = getattr(original, name), getattr(changed, name)
        assert (a.shape, a.dtype.str, a.tobytes()) == (b.shape, b.dtype.str, b.tobytes()), name
        assert not a.flags.writeable and not b.flags.writeable
    for name in ("source_id", "track", "action_ids", "action_tool_ids", "channel_provenance"):
        assert getattr(original, name) == getattr(changed, name), name
    assert changed.fingerprint != original.fingerprint
    assert runner.intervene(original).fingerprint == changed.fingerprint
    with pytest.raises(ValueError):
        changed.action_geometry.setflags(write=True)
    with pytest.raises(ValueError):
        runner.intervene(changed)


def test_extra_state_change_refused_before_attempt_or_loader(prepared):
    original, changed, baselines = prepared
    state = np.array(changed.state_features)
    state[1] = 3.
    invalid = replace(changed, state_features=state)
    ledger = {"checkpoint_attempts": [], "forward_attempts": [], "completed_forwards": []}
    with pytest.raises(ValueError, match="undeclared observation field"):
        runner.forward_once("initial", original, invalid, baselines["initial"], ledger)
    assert not any(ledger.values())


def test_unmodified_original_refused_before_attempt_or_loader(prepared):
    original, changed, baselines = prepared
    ledger = {"checkpoint_attempts": [], "forward_attempts": [], "completed_forwards": []}
    with pytest.raises(ValueError, match="Only nonSTOP"):
        runner.forward_once("initial", original, original, baselines["initial"], ledger)
    assert not any(ledger.values())


def test_frozen_manifest_validates_without_patient_checkpoint_or_forward_reads():
    manifest = ROOT / "manifests/experiments/native-opening-length-sensitivity-v1.json"
    declaration = json.loads(manifest.read_text())
    records, baselines = runner.validate(declaration)
    assert len(declaration["source_sha256"]) == 29
    assert len(records) == 7
    assert all("pat05" not in path.lower() for path in records)
    assert declaration["settings"]["new_baseline_forwards"] == 0
    assert declaration["settings"]["forward_order"] == ["initial", "RL256"]
    assert declaration["settings"]["max_wall_seconds"] == 20.
    assert declaration["settings"]["max_rss_bytes"] == 1024 ** 3
    assert declaration["settings"]["cpu_threads"] == 1
    assert "uncertified descriptor" in declaration["scope"]
    assert "ancestry only" in declaration["input_identity"]
    for item in ("clinical_correctness_measured", "planning_efficacy_measured"):
        assert declaration[item] is False
    wrong = copy.deepcopy(declaration)
    wrong["settings"]["new_baseline_forwards"] = 1
    with pytest.raises(ValueError):
        runner.validate(wrong)


def test_saved_baseline_arithmetic_requires_no_model(prepared):
    original, _, baselines = prepared
    for name, baseline in baselines.items():
        values = baseline["logits"]
        row = runner.contrast(values, baseline["probabilities"], baseline["critic_value"], baseline, original.action_ids)
        assert row["stop_minus_best_nonstop_logit"] == values[0] - max(values[1:])
        assert row["margin_change"] == row["stop_logit_change"] == row["stop_probability_change"] == 0.
        assert row["per_action_logit_change"] == [0.] * 5
        assert row["per_action_probability_change"] == [0.] * 5
        assert "no recertified physical actions" in row["comparison_scope"]


def test_output_index_includes_inherited_index_but_not_itself(tmp_path):
    nested = tmp_path / "input-metadata" / "output-sha256.json"
    nested.parent.mkdir()
    nested.write_text('{"old":"fixed"}\n')
    (tmp_path / "result.json").write_text('{}\n')
    runner.infrastructure.write_output_index(tmp_path)
    index = json.loads((tmp_path / "output-sha256.json").read_text())
    assert index == {"input-metadata/output-sha256.json": hashlib.sha256(nested.read_bytes()).hexdigest(),
                     "result.json": hashlib.sha256((tmp_path / "result.json").read_bytes()).hexdigest()}
