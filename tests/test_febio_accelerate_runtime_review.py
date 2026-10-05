"""Independent orchestration controls; no compiler or solver execution."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import febio_accelerate_runtime as driver


def declaration():
    return json.loads((driver.ROOT / "manifests/experiments/febio-accelerate-csc-runtime-v1.json").read_bytes())


@pytest.mark.parametrize("preserved", [True, False])
def test_parent_publishes_only_after_matching_preservation(tmp_path, monkeypatch, preserved):
    """Successful worker output alone cannot overrule the final integrity check."""
    monkeypatch.setattr(driver, "OUTPUT", tmp_path / "receipts")
    monkeypatch.setattr(driver, "load_declaration", lambda *_: ({}, declaration()))
    monkeypatch.setattr(driver, "verify_original", lambda *_: {"inventory": "original" if preserved else "changed"})
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: pytest.fail("No real command permitted"))
    calls = []

    def completed_worker(command, output, **limits):
        calls.append((command, limits))
        output.mkdir()
        result = {
            "status": "completed", "stage": "build", "solver_executed": False,
            "declaration_sha256": "a" * 64,
            "driver_sha256": driver.rt.sha(Path(driver.__file__)),
            "original_before": {"inventory": "original"},
            "original_after": {"inventory": "original"},
        }
        for name in driver.STAGE_FILES["build"]:
            value = result if name == "result.json" else {"status": "completed"}
            if name == "runtime-identity-candidate.json":
                value = {"status": "candidate_pending_parent_build_acceptance", "models_executed": 0,
                         "numerically_validated": False, "executable_sha256": "b" * 64}
            driver.rt.write_json(output / name, value)
        return {"status": "completed"}

    monkeypatch.setattr(driver.rt, "supervise", completed_worker)
    monkeypatch.setattr(sys, "argv", ["driver", "build", "--declaration-sha256", "a" * 64])
    assert driver.main() == (0 if preserved else 1)
    assert len(calls) == 1
    assert calls[0][1]["seconds"] == 900
    assert calls[0][1]["rss_bytes"] == 3 * 1024 ** 3
    assert calls[0][0][1:4] == ["-I", "-S", "-B"]
    acceptance_path = driver.OUTPUT / "build-01/acceptance.json"
    acceptance = json.loads(acceptance_path.read_bytes())
    identity_path = driver.OUTPUT / "runtime-identity.json"
    assert identity_path.exists() is preserved
    if preserved:
        identity = json.loads(identity_path.read_bytes())
        assert identity["build_acceptance_sha256"] == driver.rt.sha(acceptance_path)
        assert identity["status"] == "isolated_patched_runtime_built_pending_numerical_controls"
        assert identity["models_executed"] == 0 and identity["numerically_validated"] is False
        for name, digest in identity["receipts_sha256"].items():
            assert driver.rt.sha(driver.OUTPUT / name) == digest
    else:
        assert acceptance["status"] == "failed_or_incomplete"
        assert acceptance["parent_original_after"] == {"inventory": "changed"}
    with pytest.raises(FileExistsError):
        driver.main()
    assert len(calls) == 1


def test_worker_command_failure_keeps_original_recheck_and_actual_error(tmp_path, monkeypatch):
    output = tmp_path / "receipts"
    (output / "configure-01").mkdir(parents=True)
    monkeypatch.setattr(driver, "OUTPUT", output)
    monkeypatch.setattr(driver, "load_declaration", lambda *_: ({}, declaration()))
    driver.rt.write_json(output / "configure-attempt.json", {
        "stage": "configure", "driver_sha256": driver.rt.sha(Path(driver.__file__)),
        "declaration_sha256": "a" * 64,
    })
    checks = []
    monkeypatch.setattr(driver, "verify_original", lambda *_: checks.append("check") or {"unchanged": True})
    monkeypatch.setattr(driver, "prepare_source", lambda *_: None)
    monkeypatch.setattr(driver, "commands", lambda *_: [["mock-compiler"]])
    monkeypatch.setattr(driver, "verify_cache", lambda *_: pytest.fail("Failed configure has no accepted cache"))
    commands = []

    def fail_command(argv, **kwargs):
        commands.append(argv)
        raise subprocess.CalledProcessError(7, argv)

    monkeypatch.setattr(driver.subprocess, "run", fail_command)
    assert driver.worker("configure", "a" * 64) == 1
    result = json.loads((output / "configure-01/result.json").read_bytes())
    assert result["status"] == "failed_or_incomplete"
    assert result["error"]["type"] == "CalledProcessError"
    assert "7" in result["error"]["message"]
    assert result["original_before"] == result["original_after"] == {"unchanged": True}
    assert checks == ["check", "check"] and commands == [["mock-compiler"]]
    assert not (output / "runtime-identity.json").exists()
