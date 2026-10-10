"""No-weight controls for one-shot output reservation and resource launch."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_existing_result_refuses_before_preflight_or_weights(tmp_path, monkeypatch):
    runner = load("legacy_checkpoint_rollout_guard_existing", "legacy_checkpoint_rollout.py")
    result = tmp_path / "result.json"
    result.write_text("original\n")
    called = []
    monkeypatch.setattr(runner, "preflight", lambda: called.append("preflight"))
    monkeypatch.setattr(runner, "load_bound_policy", lambda *_: called.append("weights"))
    try:
        runner.execute(result)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was not refused")
    assert result.read_text() == "original\n"
    assert called == []


def test_failed_preflight_keeps_reserved_receipt_without_weight_load(tmp_path, monkeypatch):
    runner = load("legacy_checkpoint_rollout_guard_failure", "legacy_checkpoint_rollout.py")
    result = tmp_path / "result.json"
    called = []

    def fail_preflight():
        called.append("preflight")
        raise ValueError("generated-control-refusal")

    monkeypatch.setattr(runner, "preflight", fail_preflight)
    monkeypatch.setattr(runner, "load_bound_policy", lambda *_: called.append("weights"))
    try:
        runner.execute(result)
    except ValueError as error:
        assert str(error) == "generated-control-refusal"
    else:
        raise AssertionError("Expected preflight refusal")
    assert called == ["preflight"]
    receipt = json.loads(result.read_text())
    assert receipt["status"] == "failed" and receipt["stage"] == "preflight"
    assert receipt["automatic_retry"] is False


def test_existing_attempt_refuses_before_source_check_or_spawn(tmp_path, monkeypatch):
    wrapper = load("supervise_legacy_rollout_guard_existing", "supervise_legacy_rollout.py")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    called = []
    monkeypatch.setattr(wrapper, "check_sources", lambda: called.append("source"))
    monkeypatch.setattr(wrapper, "bound_supervisor", lambda: called.append("load_supervisor"))
    try:
        wrapper.run_attempt(attempt)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing attempt was not refused")
    assert called == []


def test_source_mismatch_creates_failed_attempt_without_spawn(tmp_path, monkeypatch):
    wrapper = load("supervise_legacy_rollout_guard_source", "supervise_legacy_rollout.py")
    attempt = tmp_path / "attempt"
    called = []
    monkeypatch.setattr(wrapper, "SOURCE_SHA256", {"src/resectionlab/core.py": "0" * 64})
    monkeypatch.setattr(wrapper, "bound_supervisor", lambda: called.append("load_supervisor"))
    try:
        wrapper.run_attempt(attempt)
    except ValueError:
        pass
    else:
        raise AssertionError("Source drift was not refused")
    assert called == []
    failure = json.loads((attempt / "preflight-or-completion-failure.json").read_text())
    assert failure["status"] == "failed" and failure["automatic_retry"] is False


def test_frozen_limits_and_bound_sources():
    wrapper = load("supervise_legacy_rollout_guard_pins", "supervise_legacy_rollout.py")
    assert wrapper.MAX_WALL_SECONDS == 90
    assert wrapper.MAX_RSS_BYTES == 1024 ** 3
    assert wrapper.check_sources() == wrapper.SOURCE_SHA256
    assert callable(wrapper.bound_supervisor())


def test_supervised_command_is_single_bounded_generated_attempt(tmp_path, monkeypatch):
    wrapper = load("supervise_legacy_rollout_guard_command", "supervise_legacy_rollout.py")
    attempt = tmp_path / "attempt"
    captured = {}

    def fake_supervise(command, directory, settings, declaration_sha256):
        captured.update(command=command, directory=directory, settings=settings,
            declaration_sha256=declaration_sha256)
        (directory / "comparison.json").write_text(json.dumps({"status": "complete",
            "clinical_validation": None, "new_training_updates": 0, "same_source": True,
            "same_environment": True, "same_initial_observation": True}) + "\n")
        return {"status": "complete", "returncode": 0}

    monkeypatch.setattr(wrapper, "bound_supervisor", lambda: fake_supervise)
    assert wrapper.run_attempt(attempt)["status"] == "complete"
    assert captured["command"][1:4] == ["-B", str(HERE / "legacy_checkpoint_rollout.py"), "--execute"]
    assert captured["command"][-2:] == ["--output", str(attempt / "comparison.json")]
    assert captured["settings"] == {"max_wall_seconds": 90, "max_rss_bytes": 1024 ** 3}
    assert len(captured["declaration_sha256"]) == 64
    assert json.loads((attempt / "completion.json").read_text())["comparison_sha256"]


def test_completed_child_with_mismatched_environment_is_not_accepted(tmp_path, monkeypatch):
    wrapper = load("supervise_legacy_rollout_guard_mismatch", "supervise_legacy_rollout.py")
    attempt = tmp_path / "attempt"

    def fake_supervise(_command, directory, _settings, _identity):
        (directory / "comparison.json").write_text(json.dumps({"status": "complete",
            "clinical_validation": None, "new_training_updates": 0, "same_source": True,
            "same_environment": False, "same_initial_observation": True}) + "\n")
        return {"status": "complete", "returncode": 0}

    monkeypatch.setattr(wrapper, "bound_supervisor", lambda: fake_supervise)
    try:
        wrapper.run_attempt(attempt)
    except RuntimeError as error:
        assert "not a generated-only comparison" in str(error)
    else:
        raise AssertionError("Mismatched environments were accepted")
    assert not (attempt / "completion.json").exists()
    assert json.loads((attempt / "preflight-or-completion-failure.json").read_text())["status"] == "failed"


def test_exact_generated_numpy_float64_scalar_loads_scoped_weights_only(monkeypatch):
    runner = load("legacy_checkpoint_rollout_guard_numpy_scalar", "legacy_checkpoint_rollout.py")
    generated = io.BytesIO()
    torch.save({"grid_roundoff_band": np.float64(2.0 ** -52)}, generated)
    baseline = frozenset(torch.serialization.get_safe_globals())
    real_load = torch.load
    calls = []

    def guarded_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    loaded = runner.safe_load_checkpoint(generated.getvalue())
    assert type(loaded["grid_roundoff_band"]) is np.float64
    assert loaded["grid_roundoff_band"] == np.float64(2.0 ** -52)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert frozenset(torch.serialization.get_safe_globals()) == baseline


def test_unreviewed_pickle_global_refuses_before_torch_load(monkeypatch):
    runner = load("legacy_checkpoint_rollout_guard_extra_global", "legacy_checkpoint_rollout.py")
    generated = io.BytesIO()
    torch.save({"unexpected": Path("generated-only")}, generated)
    called = []
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: called.append("unsafe"))
    try:
        runner.safe_load_checkpoint(generated.getvalue())
    except ValueError as error:
        assert "pickle globals differ" in str(error)
    else:
        raise AssertionError("Unreviewed pickle type was accepted")
    assert called == []


def test_loader_exception_preserves_original_error_and_global_set(monkeypatch):
    runner = load("legacy_checkpoint_rollout_guard_restore_error", "legacy_checkpoint_rollout.py")
    generated = io.BytesIO()
    torch.save({"grid_roundoff_band": np.float64(1.0)}, generated)
    baseline = frozenset(torch.serialization.get_safe_globals())

    def fail_load(*_args, **_kwargs):
        raise ValueError("generated-loader-error")

    monkeypatch.setattr(torch, "load", fail_load)
    try:
        runner.safe_load_checkpoint(generated.getvalue())
    except ValueError as error:
        assert str(error) == "generated-loader-error"
    else:
        raise AssertionError("Expected generated load error")
    assert frozenset(torch.serialization.get_safe_globals()) == baseline


def test_unreviewed_numpy_float32_dtype_not_allowlisted():
    runner = load("legacy_checkpoint_rollout_guard_dtype", "legacy_checkpoint_rollout.py")
    generated = io.BytesIO()
    torch.save({"wrong_dtype": np.float32(1.0)}, generated)
    baseline = frozenset(torch.serialization.get_safe_globals())
    try:
        runner.safe_load_checkpoint(generated.getvalue())
    except Exception as error:
        assert "Float32DType" in str(error)
    else:
        raise AssertionError("Unreviewed NumPy float32 dtype was accepted")
    assert frozenset(torch.serialization.get_safe_globals()) == baseline
