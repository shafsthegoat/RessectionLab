"""Native sidecar contracts without a GUI toolkit or network listener."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from resectionlab.desktop_bridge import BridgeRuntime, BinaryTransfers
from resectionlab.imaging import create_synthetic_case, save_case


@pytest.fixture
def bridge(tmp_path):
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append)
    yield runtime, events
    runtime.close()


def request(bridge, identity, operation, args=None, **options):
    runtime, events = bridge
    runtime.submit({"id": identity, "op": operation, "args": args or {}, **options})
    assert runtime.wait_idle(30)
    selected = [event for event in events if event["id"] == identity]
    return selected[-1]


def synthetic(bridge, identity="load"):
    result = request(bridge, identity, "createSyntheticCase", {"shape": [24, 24, 24]})
    assert result["event"] == "result", result
    return result["result"]


def test_case_binary_roundtrip_versions_and_null_clinical_fields(bridge):
    value = synthetic(bridge)
    descriptor = value["mri"]
    contents = Path(descriptor["path"]).read_bytes()
    assert len(contents) == descriptor["byteLength"] == 24 ** 3 * 4
    assert hashlib.sha256(contents).hexdigest() == descriptor["sha256"]
    assert descriptor["byteOrder"] == "little" and descriptor["order"] == "C"
    image = np.frombuffer(contents, dtype="<f4").reshape(descriptor["shape"])
    assert image.max() == value["intensityRange"][1]
    assert value["frame"] == "RAS+"
    assert value["clinicalDeficitProbability"] is None
    assert "not_a_patient" in value["unknowns"]
    for compartment in value["compartments"]:
        assert compartment["array"]["dtype"] == "uint8"


def test_generate_save_reopen_retains_real_evaluator_records(bridge, tmp_path):
    value = synthetic(bridge)
    generated = request(bridge, "routes", "generateRoutes", {"caseHash": value["caseHash"]})
    assert generated["event"] == "result", generated
    routes = generated["result"]
    assert len(routes["candidates"]) == 54
    assert all(route["simulated_removed_target_volume_mm3"] is None for route in routes["candidates"])
    destination = tmp_path / "saved.ressectionlab"
    saved = request(bridge, "save", "saveCase", {"caseHash": value["caseHash"], "path": str(destination),
                                                "workspace": {"compare_a": 2, "compare_b": 4}})
    assert saved["event"] == "result", saved
    reopened = request(bridge, "reopen", "loadCase", {"path": str(destination)})
    assert reopened["result"]["caseHash"] == value["caseHash"]
    workspace = reopened["result"]["artifacts"]["workspace"]
    assert workspace["compare_a"] == 2
    assert workspace["routes"] == routes["candidates"]
    refused = request(bridge, "overwrite-refusal", "saveCase", {"caseHash": value["caseHash"], "path": str(destination)})
    assert refused["error"]["code"] == "DESTINATION_EXISTS"


def test_unknown_operation_stale_hash_and_injected_evaluator_rejected(bridge, tmp_path):
    value = synthetic(bridge)
    assert request(bridge, "exec", "exec", {"command": "rm"})["error"]["code"] == "UNSUPPORTED_OPERATION"
    assert request(bridge, "stale", "inspectEvidence", {"caseHash": "sha256:stale"})["error"]["code"] == "CASE_VERSION_UNAVAILABLE"
    assert request(bridge, "injected", "generateRoutes", {"caseHash": value["caseHash"], "evaluationWorlds": [1]})["error"]["code"] == "INVALID_ARGUMENT"
    injected = request(bridge, "fake-results", "saveCase", {"caseHash": value["caseHash"], "path": str(tmp_path / "false.ressectionlab"), "workspace": {"routes": []}})
    assert injected["error"]["code"] == "INVALID_ARGUMENT"


def test_stale_saved_workspace_is_withheld(bridge, tmp_path):
    destination = tmp_path / "stale.ressectionlab"
    save_case(create_synthetic_case((24, 24, 24)), destination, artifacts={"workspace": {"case_hash": "wrong", "routes": [{"clinical_deficit_probability": 0.1}]}})
    value = request(bridge, "stale-bundle", "loadCase", {"path": str(destination)})["result"]
    assert "workspace" not in value["artifacts"]
    assert value["artifacts"]["withheldWorkspaceReason"] == "saved_workspace_case_version_mismatch"


def test_cancel_and_ping_are_immediate_while_worker_busy(bridge, monkeypatch):
    runtime, events = bridge
    entered = threading.Event()
    def slow(operation, args, request_state, progress):
        entered.set()
        while not request_state.cancelled.wait(0.005):
            pass
        request_state.check()
    monkeypatch.setattr(runtime.session, "execute", slow)
    runtime.submit({"id": "slow", "op": "createSyntheticCase", "args": {}})
    assert entered.wait(1)
    runtime.submit({"id": "ping", "op": "ping", "args": {}})
    assert events[-1]["result"]["protocolVersion"] == 1
    runtime.submit({"id": "cancel", "op": "cancel", "args": {"requestId": "slow"}})
    assert runtime.wait_idle(1)
    assert [event["event"] for event in events if event["id"] == "slow"] == ["started", "cancelled"]


def test_timeout_emits_one_terminal_error(bridge, monkeypatch):
    runtime, events = bridge
    def slow(operation, args, request_state, progress):
        request_state.cancelled.wait(2)
        request_state.check()
    monkeypatch.setattr(runtime.session, "execute", slow)
    runtime.submit({"id": "timeout", "op": "createSyntheticCase", "args": {}, "timeoutMs": 100})
    assert runtime.wait_idle(2)
    selected = [event for event in events if event["id"] == "timeout"]
    assert [event["event"] for event in selected] == ["started", "error"]
    assert selected[-1]["error"]["code"] == "TIMEOUT"


def test_cached_cases_bounded_and_evicted_hash_fails(bridge):
    first = synthetic(bridge)
    request(bridge, "second", "createSyntheticCase", {"shape": [25, 24, 24]})
    request(bridge, "third", "createSyntheticCase", {"shape": [26, 24, 24]})
    assert len(bridge[0].session.cases) == 2
    assert not Path(first["mri"]["path"]).exists()
    assert request(bridge, "evicted", "inspectEvidence", {"caseHash": first["caseHash"]})["error"]["code"] == "CASE_VERSION_UNAVAILABLE"


def test_transfer_cleanup_preserves_parent_files(tmp_path):
    parent = tmp_path / "transfers"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("user file")
    transfers = BinaryTransfers(parent)
    descriptor = transfers.array(np.arange(8).reshape(2, 2, 2), "float32")
    assert Path(descriptor["path"]).is_relative_to(transfers.root)
    transfers.close()
    assert sentinel.read_text() == "user file"


def test_jsonl_process_stdout_is_protocol_only_and_imports_no_gui(tmp_path):
    source = "\n".join(["not json", json.dumps({"id": "ping", "op": "ping", "args": {}}), json.dumps({"id": "stop", "op": "shutdown", "args": {}})]) + "\n"
    probe = ("import sys; from resectionlab.desktop_bridge import main; code = main(); "
             "assert not any(name.startswith(('PySide6', 'vtk')) for name in sys.modules), 'Sidecar imported a GUI toolkit'; "
             "raise SystemExit(code)")
    process = subprocess.run([sys.executable, "-c", probe, "--transfer-dir", str(tmp_path / "wire")], input=source, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert events[0]["error"]["code"] == "INVALID_JSON"
    assert events[1]["result"]["protocolVersion"] == 1
    assert events[2]["result"]["shutdown"] is True


def test_actual_native_training_replay_export_and_restart_recheck(tmp_path):
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    try:
        case = synthetic(bridge)
        trained = request(bridge, "train", "trainPatient", {"caseHash": case["caseHash"], "budgetSeconds": 5, "seed": 11})
        assert trained["event"] == "result", trained
        result = trained["result"]
        training = result["training"]
        assert training["gradient_steps"] > 0
        assert training["actor_parameters_changed"] is True
        assert training["replay_status"] == "accepted_independent_geometry"
        assert training["replay"]["final_evaluation"] is False
        assert training["replay"]["native_certificate"]["complete_tool_checked"] is True
        assert "history" not in training["replay"]["metrics"]
        assert any(event.get("progress", {}).get("runId") == result["runId"] for event in events)
        selected = {"caseHash": case["caseHash"], "runId": result["runId"]}
        first = request(bridge, "initial-mask", "replayTraining", {**selected, "step": 0})["result"]
        assert not np.fromfile(first["removedMask"]["path"], dtype="uint8").any()
        final = request(bridge, "final-mask", "replayTraining", selected)["result"]
        removed = np.fromfile(final["removedMask"]["path"], dtype="uint8").reshape(case["shape"])
        voxel_volume = abs(np.linalg.det(np.asarray(case["affine"])[:3, :3]))
        assert removed.sum() * voxel_volume == pytest.approx(final["simulatedRemovedTargetVolumeMm3"] + final["simulatedRemovedNormalVolumeMm3"])
        assert final["clinicalDeficitProbability"] is None
        destination = tmp_path / "candidate.json"
        exported = request(bridge, "export", "exportCandidate", {**selected, "path": str(destination)})
        assert exported["event"] == "result", exported
        artifact = json.loads(destination.read_text())
        assert artifact["candidate"]["artifact_hash"] == training["replay"]["artifact_hash"]
        assert artifact["clinical_deficit_probability"] is None and artifact["final_evaluation"] is False
        runs = request(bridge, "runs", "listRuns", {"caseHash": case["caseHash"]})["result"]["runs"]
        assert runs[0]["hasCheckpoint"] and runs[0]["hasAcceptedReplay"]
        refused = request(bridge, "changed-budget", "trainPatient", {"caseHash": case["caseHash"], "resumeRunId": result["runId"], "budgetSeconds": 6})
        assert refused["error"]["code"] == "RESUME_CONTRACT_CHANGED"
    finally:
        runtime.close()
    # A new process/session must rebuild the model and recheck the exact native
    # actions instead of trusting a supplied saved certificate.
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    try:
        synthetic(bridge)
        reopened = request(bridge, "restored-replay", "replayTraining", selected)
        assert reopened["event"] == "result", reopened
        assert reopened["result"]["simulatedRemovedTargetVolumeMm3"] == final["simulatedRemovedTargetVolumeMm3"]
        path = tmp_path / "runs" / result["runId"] / "native-refinement.json"
        saved = json.loads(path.read_text())
        saved["replay"]["metrics"]["clinical_deficit_probability"] = .9
        path.write_text(json.dumps(saved))
        tampered = request(bridge, "tampered", "replayTraining", selected)
        assert tampered["error"]["code"] == "RUN_INTEGRITY_FAILED"
    finally:
        runtime.close()


def test_training_controls_reject_final_worlds_stale_case_and_raw_paths(tmp_path):
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    try:
        case = synthetic(bridge)
        for index, extra in enumerate(({"finalEvaluationSeeds": [1]}, {"checkpointPath": "/tmp/weights.pt"},
                                       {"worlds": [1]}, {"rewardWeights": {"target": 100}})):
            refused = request(bridge, f"bad-{index}", "trainPatient", {"caseHash": case["caseHash"], **extra})
            assert refused["error"]["code"] == "INVALID_ARGUMENT"
        refused = request(bridge, "path", "replayTraining", {"caseHash": case["caseHash"], "runId": "../../checkpoint"})
        assert refused["error"]["code"] == "INVALID_RUN_ID"
        refused = request(bridge, "unknown-route", "trainPatient", {"caseHash": case["caseHash"], "routeId": "forged"})
        assert refused["error"]["code"] == "ROUTE_UNAVAILABLE"
        assert not list((tmp_path / "runs").glob("*/bridge-run.json"))
    finally:
        runtime.close()


def test_training_cancellation_keeps_checkpoint_and_original_resume_budget(tmp_path):
    events = []
    cancelled = False
    def emit(event):
        nonlocal cancelled
        events.append(event)
        if event.get("progress", {}).get("phase") == "training" and not cancelled:
            cancelled = True
            runtime.submit({"id": "cancel-training", "op": "cancel", "args": {"requestId": "train"}})
    runtime = BridgeRuntime(tmp_path / "transfers", emit, run_dir=tmp_path / "runs")
    bridge = runtime, events
    try:
        case = synthetic(bridge)
        request(bridge, "train", "trainPatient", {"caseHash": case["caseHash"], "budgetSeconds": 5, "seed": 11})
        assert cancelled
        terminals = [event["event"] for event in events if event["id"] == "train" and event["event"] in {"result", "error", "cancelled"}]
        assert terminals == ["cancelled"]
        runs = request(bridge, "runs", "listRuns", {"caseHash": case["caseHash"]})["result"]["runs"]
        assert runs[0]["status"] == "cancelled" and runs[0]["hasCheckpoint"]
        assert runs[0]["hasAcceptedReplay"] is False
        resumed = request(bridge, "resume", "trainPatient", {"caseHash": case["caseHash"], "resumeRunId": runs[0]["runId"]})
        assert resumed["event"] == "result", resumed
        assert resumed["result"]["config"]["budgetSeconds"] == 5
        assert resumed["result"]["training"]["gradient_steps"] >= 2
        assert resumed["result"]["training"]["replay_status"] == "accepted_independent_geometry"
    finally:
        runtime.close()


def test_selected_route_access_is_converted_from_lps_before_native_training(tmp_path, monkeypatch):
    from dataclasses import replace
    case = replace(create_synthetic_case((24, 24, 24)), frame="LPS+")
    path = save_case(case, tmp_path / "lps.ressectionlab")
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    captured = {}
    def capture_native_options(case, directory, **options):
        captured.update(options)
        raise RuntimeError("stop after observing native coordinate contract")
    monkeypatch.setattr("resectionlab.native_refinement.run_native_refinement", capture_native_options)
    try:
        loaded = request(bridge, "load", "loadCase", {"path": str(path)})["result"]
        routes = request(bridge, "routes", "generateRoutes", {"caseHash": loaded["caseHash"]})["result"]["candidates"]
        selected = next(route for route in routes if route["geometry"]["feasible"])
        trained = request(bridge, "train", "trainPatient", {"caseHash": loaded["caseHash"], "routeId": selected["route_id"]})
        assert trained["event"] == "error"
        np.testing.assert_allclose(captured["access"].center_mm, np.asarray(selected["window"]["center_mm"]) * [-1, -1, 1])
        np.testing.assert_allclose(captured["access"].normal_inward, np.asarray(selected["window"]["normal_inward"]) * [-1, -1, 1])
        assert captured["tools"][0].tool_id == selected["tool"]["tool_id"]
    finally:
        runtime.close()
