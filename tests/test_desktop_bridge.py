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


def test_full_head_layer_inventory_keeps_separate_count_and_byte_bounds(bridge,tmp_path,monkeypatch):
    from dataclasses import replace
    from resectionlab.desktop_bridge import MAX_COMPARTMENTS,MAX_CASE_BYTES
    base=create_synthetic_case((24,24,24));masks={}
    for index in range(MAX_COMPARTMENTS+1):
        mask=np.zeros(base.mri.shape,bool);mask.flat[index]=True;masks[f'structure_{index}']=mask
    path=tmp_path/'full-head.ressectionlab'
    case=replace(base,compartments=dict(list(masks.items())[:MAX_COMPARTMENTS]),source_compartments=None)
    save_case(case,path)
    result=request(bridge,'head','loadCase',{'path':str(path)})
    assert result['event']=='result' and len(result['result']['compartments'])==MAX_COMPARTMENTS
    excess=tmp_path/'too-many.ressectionlab';save_case(replace(case,compartments=masks,source_compartments=None),excess)
    assert request(bridge,'count-bound','loadCase',{'path':str(excess)})['error']['code']=='LAYER_COUNT_LIMIT'
    monkeypatch.setattr(bridge[0].session,'_case_array_bytes',lambda value:MAX_CASE_BYTES+1)
    assert request(bridge,'byte-bound','loadCase',{'path':str(path)})['error']['code']=='CASE_SIZE_LIMIT'


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


def test_native_vascular_scores_round_trip_and_bind_mesh_to_source_cells(bridge, tmp_path):
    pytest.importorskip('vtkmodules')
    from dataclasses import replace
    from resectionlab.anatomy_geometry import vascular_probability_mesh
    from resectionlab.imaging import load_case
    from resectionlab.workspace_bundle import save_workspace

    base = create_synthetic_case((24, 24, 24))
    scores = np.zeros(base.mri.shape, dtype=np.float32)
    scores[5:16, 6:10, 8:13] = .9
    scores[5:16, 10, 8:13] = .35
    scores[18, 18, 18] = .97
    mask = scores >= .5
    mask[18, 18, 18] = False
    name = 'vessel'
    affine = np.diag([.30, .30, .42, 1.])
    case = replace(base, affine=affine, compartments={name: mask}, source_compartments=None,
                   metadata={'anatomy_layers': {name: {'kind': 'vessel'}}})
    native = replace(case, vascular_probabilities={name: scores})
    assert native.semantic_hash != case.semantic_hash
    assert native.planning_hash == case.planning_hash
    path = save_case(native, tmp_path / 'score-bound.ressectionlab')
    cold = load_case(path)
    np.testing.assert_array_equal(cold.vascular_probabilities[name], scores)
    path = tmp_path / 'score-bound-workspace.ressectionlab'
    save_workspace(native, path, sources=[], imaging=None, episode=None, selection=None, artifacts={})
    result = request(bridge, 'score-case', 'loadCase', {'path': str(path)})
    assert result['event'] == 'result', result
    layer = result['result']['compartments'][0]
    surface = layer['vesselSurface']

    def read(field, dtype):
        descriptor = surface[field]
        return np.fromfile(descriptor['path'], dtype=dtype).reshape(descriptor['shape'])

    positions, owners = read('positions', '<f4'), read('sourceCells', '<u4')
    assert len(owners) == len(positions)
    assert mask.ravel()[owners].all()
    assert positions[:, 0].max() < 16
    raw = vascular_probability_mesh(scores, mask)
    for provided, physical in zip((positions, read('indices', '<u4'), read('normals', '<f4')), raw):
        np.testing.assert_array_equal(provided, physical)
    lighting = read('scoreLighting', '<f4')
    assert lighting.shape == (len(positions), 4) and np.isfinite(lighting).all()
    assert np.max(lighting[:, 3]) > 0 and np.all((lighting[:, 3] >= 0) & (lighting[:, 3] <= 1))
    # This wall crosses the retained .9 -> .35 score edge away from its midpoint.
    assert positions[:, 1].max() == pytest.approx(9 + (.9 - .5) / (.9 - .35), abs=1e-5)
    correction = mask.copy()
    correction[5, 5, 10] = True
    edited = save_case(native.revised(compartments={name: correction}), tmp_path / 'edited.ressectionlab')
    result = request(bridge, 'edited-score-case', 'loadCase', {'path': str(edited)})
    assert result['event'] == 'result', result
    assert 'vesselSurface' not in result['result']['compartments'][0]


def test_vessel_score_lighting_respects_source_components_budget_and_cancellation(monkeypatch):
    from resectionlab import vascular_surface_lighting as lighting

    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[3:9, 3:9, 3:9] = True
    scores = mask.astype(np.float32)
    positions = np.array([[5., 5., 8.5]], dtype=np.float32)
    normals = np.array([[0., 0., 1.]], dtype=np.float32)
    cells = np.array([np.ravel_multi_index((5, 5, 8), mask.shape)], dtype=np.uint32)
    affine = np.diag([.30, .30, .42, 1.])
    args = (scores, mask, positions, normals, cells, affine)
    assert lighting.score_lighting_normals(*args)[0, 3] > 0
    beside = mask.copy()
    beside[5, 5, 10] = True
    assert lighting.score_lighting_normals(beside.astype(np.float32), beside, *args[2:]) is None
    with monkeypatch.context() as budget:
        budget.setattr(lighting, '_WORK_BYTES', 0)
        assert lighting.score_lighting_normals(*args) is None

    def cancelled():
        raise InterruptedError('lighting preparation cancelled')

    with pytest.raises(InterruptedError, match='lighting preparation cancelled'):
        lighting.score_lighting_normals(*args, check=cancelled)
    np.testing.assert_array_equal(scores, mask)
    np.testing.assert_array_equal(positions, [[5., 5., 8.5]])


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


def test_new_practice_save_preserves_a_file_created_while_staging(bridge, tmp_path, monkeypatch):
    from resectionlab import workspace_bundle

    episode_id = "practice-24087e0d-f6db-4a1a-89c6-e573edf345cf"
    source = tmp_path / "practice.ressectionlab"
    case = create_synthetic_case((24, 24, 24))
    save_case(case, source, artifacts={"surgical_toolbox": {
        "branchId": episode_id, "practiceEpisodes": {episode_id: {"episodeId": episode_id}},
    }})
    opened = request(bridge, "open", "loadCase", {"path": str(source)})["result"]
    save_workspace = workspace_bundle.save_workspace
    destination = tmp_path / "shared.ressectionlab"

    def save_during_competing_write(*args, **kwargs):
        result = save_workspace(*args, **kwargs)
        destination.write_bytes(b"concurrent workspace")
        return result

    monkeypatch.setattr(workspace_bundle, "save_workspace", save_during_competing_write)
    result = request(bridge, "save", "saveCase", {
        "caseHash": opened["caseHash"], "path": str(destination), "overwrite": True,
    })
    assert result["error"]["code"] == "PRACTICE_SAVE_AS"
    assert destination.read_bytes() == b"concurrent workspace"


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
    from resectionlab.native_backend import backend_identity
    assert events[-1]["result"]["numericalBackend"] == backend_identity()
    assert events[-1]["result"]["runtime"]["executable"] == sys.executable
    assert Path(events[-1]["result"]["runtime"]["modulePath"]).name == "desktop_bridge.py"
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


@pytest.mark.parametrize('command,timeout',[('inspect',3600000),('generatePreparations',600000)])
def test_long_replay_admission_is_cancellable_without_extending_action_budgets(bridge, monkeypatch, command, timeout):
    runtime, events = bridge
    entered = threading.Event()
    def slow(operation, args, request_state, progress):
        entered.set()
        request_state.cancelled.wait(2)
        request_state.check()
    monkeypatch.setattr(runtime.session, "execute", slow)
    runtime.submit({"id": "cold-replay", "op": "surgicalRehearsal", "args": {"command": command}, "timeoutMs": timeout})
    assert entered.wait(1)
    runtime.submit({"id": "cancel-replay", "op": "cancel", "args": {"requestId": "cold-replay"}})
    assert runtime.wait_idle(1)
    assert [event['event'] for event in events if event['id'] == 'cold-replay'] == ['started', 'cancelled']
    for identity, operation, args, timeout in [
        ('long-step', 'surgicalRehearsal', {'command': 'step'}, 300001),
        ('long-slice', 'surgicalRehearsal', {'command': 'fullCase'}, 300001),
        ('long-load', 'loadCase', {}, 300001),
        ('unbounded-replay', 'surgicalRehearsal', {'command': 'seekFrame'}, 3600001),
        ('unbounded-preparation', 'surgicalRehearsal', {'command': 'generatePreparations'}, 600001),
    ]:
        result = request(bridge, identity, operation, args, timeoutMs=timeout)
        assert result['event'] == 'error' and result['error']['code'] == 'INVALID_ARGUMENT'


def test_replay_admission_budget_is_a_stall_window_renewed_by_progress(bridge, monkeypatch):
    runtime, events = bridge
    def advancing(operation, args, request_state, progress):
        # Three times the stall window in total, but never silent for long.
        for index in range(12):
            time.sleep(.05)
            progress(index / 12, f'Verifying saved actions by replay: {index} of 12')
            request_state.check()
        return {'verified': True}
    def silent(operation, args, request_state, progress):
        progress(0., 'Verifying saved actions by replay: 0 of 12')
        request_state.cancelled.wait(2)
        request_state.check()
    monkeypatch.setattr(runtime.session, "execute", advancing)
    result = request(bridge, 'advancing', 'surgicalRehearsal', {'command': 'inspect'}, timeoutMs=200)
    assert result['event'] == 'result' and result['result'] == {'verified': True}
    assert sum(event['event'] == 'progress' for event in events if event['id'] == 'advancing') == 12
    monkeypatch.setattr(runtime.session, "execute", silent)
    result = request(bridge, 'stalled', 'surgicalRehearsal', {'command': 'inspect'}, timeoutMs=200)
    assert result['event'] == 'error' and result['error']['code'] == 'TIMEOUT'
    # Ordinary actions keep a fixed budget: progress does not extend it.
    monkeypatch.setattr(runtime.session, "execute", advancing)
    result = request(bridge, 'fixed', 'surgicalRehearsal', {'command': 'step'}, timeoutMs=200)
    assert result['event'] == 'error' and result['error']['code'] == 'TIMEOUT'


def test_reconstruction_transport_accepts_bounded_model_job_and_cancellation(bridge, monkeypatch):
    runtime, events = bridge
    entered = threading.Event()
    def slow(operation, args, request_state, progress):
        assert operation == 'anatomyReconstruct'
        entered.set()
        request_state.cancelled.wait(2)
        request_state.check()
    monkeypatch.setattr(runtime.session, 'execute', slow)
    runtime.submit({'id': 'model-job', 'op': 'anatomyReconstruct', 'args': {}, 'timeoutMs': 1800000})
    assert entered.wait(1)
    runtime.submit({'id': 'cancel-model', 'op': 'cancel', 'args': {'requestId': 'model-job'}})
    assert runtime.wait_idle(1)
    assert [e['event'] for e in events if e['id'] == 'model-job'] == ['started', 'cancelled']
    for operation, timeout in [('anatomyReconstruct', 1800001), ('anatomyImport', 1800000)]:
        result = request(bridge, operation, operation, {}, timeoutMs=timeout)
        assert result['event'] == 'error' and result['error']['code'] == 'INVALID_ARGUMENT'


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
        np.testing.assert_allclose(captured["selected_entry_mm"], np.asarray(selected["entry_mm"]) * [-1, -1, 1])
        np.testing.assert_allclose(captured["selected_target_mm"], np.asarray(selected["target_mm"]) * [-1, -1, 1])
        assert captured["tools"][0].tool_id == selected["tool"]["tool_id"]
    finally:
        runtime.close()


def test_native_alternatives_preserve_original_routes_and_separate_model_receipts(bridge, tmp_path):
    case = synthetic(bridge)
    source = {"caseHash": case["caseHash"]}
    original = request(bridge, "original", "generateRoutes", source)["result"]
    combined = request(bridge, "alternatives", "generateNativeRoutes", source)["result"]
    assert combined["candidates"][:54] == original["candidates"]
    assert len(combined["candidates"]) == 56
    assert combined["combined_models"] is True and combined["planning_model_hash"] is None
    assert len(combined["search_models"]) == 2 and len(set(combined["planning_model_hashes"])) == 2
    assert combined["access_support"] == {}
    by_model = {receipt["planning_model_hash"]: set(receipt["candidate_ids"]) for receipt in combined["search_models"]}
    for candidate in combined["candidates"]:
        assert set(candidate["dominated_by"]) <= by_model[candidate["planning_model_hash"]]
        assert candidate["clinical_deficit_probability"] is None
        assert candidate["simulated_removed_target_volume_mm3"] is None
    assert {item["tool_id"] for item in combined["candidates"][54:]} == {"native-fine-aspiration", "native-wide-aspiration"}
    repeated = request(bridge, "repeat", "generateNativeRoutes", source)["result"]
    assert len(repeated["candidates"]) == 56 and len(repeated["search_models"]) == 2
    path = tmp_path / "alternatives.ressectionlab"
    request(bridge, "save", "saveCase", {**source, "path": str(path)})
    restored = request(bridge, "reopen", "loadCase", {"path": str(path)})["result"]
    assert restored["artifacts"]["workspace"]["routes"] == repeated["candidates"]
    assert restored["artifacts"]["searchModels"] == repeated["search_models"]


def test_blocked_selected_route_reports_exact_geometry_without_starting_optimizer(tmp_path, monkeypatch):
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("STOP-only geometry must not start a learner")
    monkeypatch.setattr("resectionlab.learning.train_patient_policy", forbidden_optimizer)
    try:
        case = synthetic(bridge)
        routes = request(bridge, "routes", "generateRoutes", {"caseHash": case["caseHash"]})["result"]["candidates"]
        selected = next(item for item in routes if item["geometry"]["feasible"])
        args = {"caseHash": case["caseHash"], "routeId": selected["route_id"]}
        readiness = request(bridge, "inspect", "inspectRefinement", args)["result"]
        assert readiness["status"] == "no_actionable_moves" and readiness["legalNonStopActions"] == 0
        assert "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE" in readiness["reasons"]
        assert readiness["optimizationChoiceScope"] == "STOP_or_declared_native_stroke"
        assert readiness["route_binding"]["candidate_entries_mm"] == [selected["entry_mm"]]
        assert readiness["route_binding"]["candidate_targets_mm"] == [selected["target_mm"]]
        assert readiness["route_binding"]["tools"] == [selected["tool"]]
        assert not list((tmp_path / "runs").glob("*/bridge-run.json"))
        blocked = request(bridge, "train", "trainPatient", {**args, "budgetSeconds": 5})
        assert blocked["event"] == "result", blocked
        result = blocked["result"]
        assert result["status"] == "no_actionable_moves"
        assert result["training"]["role"] == "preflight"
        assert result["training"]["optimizer_mode"] is None
        assert result["training"]["gradient_steps"] == 0
        assert result["training"]["optimization_environment_steps"] == 0
        assert result["training"]["replay"] is None
        directory = tmp_path / "runs" / result["runId"]
        assert not any((directory / name).exists() for name in ("checkpoint.pt", "initial.pt", "contract.json"))
        listed = request(bridge, "list", "listRuns", {"caseHash": case["caseHash"]})["result"]["runs"][0]
        assert listed["hasCheckpoint"] is False and listed["hasAcceptedReplay"] is False
    finally:
        runtime.close()


def test_selected_native_route_updates_and_restored_replay_keep_exact_binding(tmp_path):
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    try:
        case = synthetic(bridge)
        generated = request(bridge, "native", "generateNativeRoutes", {"caseHash": case["caseHash"]})["result"]
        selected = next(item for item in generated["candidates"] if item["tool_id"] == "native-wide-aspiration")
        args = {"caseHash": case["caseHash"], "routeId": selected["route_id"]}
        readiness = request(bridge, "inspect", "inspectRefinement", args)["result"]
        assert readiness["status"] == "ready" and readiness["legalNonStopActions"] == 1
        trained = request(bridge, "train", "trainPatient", {**args, "budgetSeconds": 5, "seed": 11})
        assert trained["event"] == "result", trained
        result = trained["result"]
        training = result["training"]
        assert training["gradient_steps"] > 0 and training["actor_parameters_changed"] is True
        assert training["route_binding"] == readiness["route_binding"]
        assert training["replay_status"] == "accepted_independent_geometry"
        assert result["config"]["selectedEntryMm"] == selected["entry_mm"]
        assert result["config"]["selectedTargetMm"] == selected["target_mm"]
        assert result["config"]["optimizationChoiceScope"] == "STOP_or_declared_native_stroke"
        replay_args = {"caseHash": case["caseHash"], "runId": result["runId"]}
        replay = request(bridge, "replay", "replayTraining", replay_args)
        assert replay["event"] == "result", replay
        assert replay["result"]["training"]["route_binding"] == readiness["route_binding"]
    finally:
        runtime.close()
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    bridge = runtime, events
    try:
        synthetic(bridge)
        reopened = request(bridge, "reopened-replay", "replayTraining", replay_args)
        assert reopened["event"] == "result", reopened
        assert reopened["result"]["training"]["route_binding"] == readiness["route_binding"]
        assert reopened["result"]["training"]["replay"]["native_certificate"]["feasible"] is True
    finally:
        runtime.close()
