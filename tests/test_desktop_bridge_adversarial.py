"""Independent failure-oriented checks for the local desktop trust boundary."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import nibabel as nib
import numpy as np
import pytest

import resectionlab.desktop_bridge as bridge
from resectionlab.imaging import create_synthetic_case, save_case


@pytest.fixture
def runtime(tmp_path):
    events = []
    runtime = bridge.BridgeRuntime(tmp_path / "transfer", events.append)
    yield runtime, events
    runtime.close()


def terminal(events, request_id):
    return [event for event in events if event.get("id") == request_id
            and event["event"] in {"result", "error", "cancelled"}]


def install(runtime):
    runtime.submit({"id": "case", "op": "createSyntheticCase", "args": {"shape": [24, 24, 24]}})
    assert runtime.wait_idle(10)
    return next(iter(runtime.session.cases))


def test_nested_json_parse_error_does_not_kill_sidecar(tmp_path):
    # Much smaller than the line byte limit, but beyond Python's recursion limit.
    deep = '[' * 1500 + '0' + ']' * 1500
    ping = json.dumps({"id": "still-alive", "op": "ping", "args": {}})
    shutdown = json.dumps({"id": "close", "op": "shutdown", "args": {}})
    completed = subprocess.run(
        [sys.executable, "-m", "resectionlab.desktop_bridge", "--transfer-dir", str(tmp_path / "wire")],
        input=deep + '\n' + ping + '\n' + shutdown + '\n', text=True,
        capture_output=True, timeout=15,
    )
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert any(event.get("id") == "still-alive" and event["event"] == "result" for event in events)
    assert any(event["event"] == "error" for event in events)


def test_duplicate_active_id_cannot_terminate_the_original_request(runtime, monkeypatch):
    instance, events = runtime
    entered, release = threading.Event(), threading.Event()

    def slow(*_args):
        entered.set()
        assert release.wait(5)
        return {"realResult": True}

    monkeypatch.setattr(instance.session, "execute", slow)
    instance.submit({"id": "original", "op": "createSyntheticCase", "args": {}})
    assert entered.wait(3)
    try:
        instance.submit({"id": "original", "op": "createSyntheticCase", "args": {}})
    finally:
        release.set()
    assert instance.wait_idle(5)
    results = terminal(events, "original")
    assert len(results) == 1, results
    assert results[0]["event"] == "result"
    assert results[0]["result"] == {"realResult": True}


def test_cancelled_queued_work_still_counts_against_memory_bound(runtime, monkeypatch):
    instance, events = runtime
    entered, release = threading.Event(), threading.Event()

    def blocked(_op, _args, request, _progress):
        entered.set()
        assert release.wait(5)
        request.check()
        return {}

    monkeypatch.setattr(instance.session, "execute", blocked)
    instance.submit({"id": "head", "op": "createSyntheticCase", "args": {}})
    assert entered.wait(3)
    try:
        for number in range(bridge.MAX_PENDING + 3):
            name = f"queued-{number}"
            instance.submit({"id": name, "op": "createSyntheticCase", "args": {}})
            instance.submit({"id": f"cancel-{number}", "op": "cancel", "args": {"requestId": name}})
        busy = [event for event in events if event.get("error", {}).get("code") == "BUSY"]
        assert busy, "Cancelled queued futures bypassed the numerical worker capacity limit"
        assert len(instance._requests) <= bridge.MAX_PENDING
    finally:
        release.set()
    assert instance.wait_idle(5)


def test_timeout_has_exactly_one_terminal_even_when_worker_finishes_late(runtime, monkeypatch):
    instance, events = runtime
    entered, release = threading.Event(), threading.Event()

    def slow(*_args):
        entered.set()
        assert release.wait(5)
        return {"late": True}

    monkeypatch.setattr(instance.session, "execute", slow)
    instance.submit({"id": "timed", "op": "createSyntheticCase", "args": {}, "timeoutMs": 100})
    assert entered.wait(3)
    deadline = time.monotonic() + 3
    while not terminal(events, "timed") and time.monotonic() < deadline:
        time.sleep(.005)
    release.set()
    assert instance.wait_idle(5)
    results = terminal(events, "timed")
    assert len(results) == 1
    assert results[0]["event"] == "error"
    assert results[0]["error"]["code"] == "TIMEOUT"


def test_oversized_nifti_header_rejected_before_any_voxel_read(runtime, monkeypatch, tmp_path):
    import resectionlab.imaging as imaging
    instance, events = runtime
    path = tmp_path / "oversized.nii"
    header = nib.Nifti1Header()
    header.set_data_shape((2048, 2048, 512))
    header.set_data_dtype(np.float32)
    header.set_xyzt_units("mm")
    header.set_sform(np.eye(4), code=1)
    header.set_qform(np.eye(4), code=1)
    header["vox_offset"] = 352
    with path.open("wb") as output:
        header.write_to(output)
        output.write(b'\x00' * 4)
    attempted = []

    def reject_read(source):
        attempted.append(source)
        raise AssertionError("Voxel allocation attempted before header size validation")

    monkeypatch.setattr(imaging, "_read_finite", reject_read)
    instance.submit({"id": "huge", "op": "importNifti", "args": {"structuralPath": str(path)}})
    assert instance.wait_idle(5)
    assert not attempted, "Untrusted header reached the voxel materializer"
    result = terminal(events, "huge")
    assert len(result) == 1 and result[0]["event"] == "error"
    assert not instance.session.cases


def test_imported_case_cannot_supply_clinical_or_removal_claims(runtime, tmp_path):
    instance, events = runtime
    case = create_synthetic_case((24, 24, 24))
    forged = {"case_hash": case.semantic_hash, "route_id": "fabricated",
              "clinical_deficit_probability": .91,
              "simulated_removed_target_volume_mm3": 1e9,
              "geometry": {"feasible": True}}
    source = save_case(case, tmp_path / "edited.ressectionlab", artifacts={
        "workspace": {"case_hash": case.semantic_hash, "routes": [forged]},
    })
    instance.submit({"id": "load", "op": "loadCase", "args": {"path": str(source)}})
    assert instance.wait_idle(5)
    result = terminal(events, "load")[0]
    if result["event"] == "result":
        routes = result["result"].get("artifacts", {}).get("workspace", {}).get("routes", [])
        assert not routes, "Same-case hash alone admitted forged evaluator claims"
    else:
        assert result["event"] == "error"


def test_cancelled_save_never_reports_cancelled_after_committing_destination(runtime, monkeypatch, tmp_path):
    instance, events = runtime
    case_hash = install(instance)
    destination = tmp_path / "cancelled.ressectionlab"
    entered, release = threading.Event(), threading.Event()
    real_replace = bridge.os.replace

    def paused_replace(source, target):
        if Path(target) == destination:
            entered.set()
            assert release.wait(5)
        return real_replace(source, target)

    monkeypatch.setattr(bridge.os, "replace", paused_replace)
    instance.submit({"id": "save", "op": "saveCase", "args": {"caseHash": case_hash, "path": str(destination)}})
    assert entered.wait(5)
    cancel = threading.Thread(target=lambda: instance.submit({
        "id": "cancel-save", "op": "cancel", "args": {"requestId": "save"}}))
    cancel.start()
    # Give cancellation a chance to win; a commit lock may deliberately delay it.
    cancel.join(.05)
    release.set()
    cancel.join(5)
    assert not cancel.is_alive()
    assert instance.wait_idle(5)
    result = terminal(events, "save")
    assert len(result) == 1
    assert not (destination.exists() and result[0]["event"] == "cancelled"), (
        "Save reported cancellation while publishing the destination file")


@pytest.mark.parametrize("label_count", [33, 65])
def test_unmapped_annotation_label_fanout_is_bounded_before_case_masks(runtime, monkeypatch, tmp_path, label_count):
    import resectionlab.imaging as imaging
    instance, events = runtime
    structural = tmp_path / "structural.nii"
    annotation = tmp_path / "labels.nii"
    mri = np.arange(24 ** 3, dtype=np.float32).reshape(24, 24, 24)
    labels = ((np.arange(24 ** 3) % label_count) + 1).astype(np.uint16).reshape(24, 24, 24)
    for path, values in ((structural, mri), (annotation, labels)):
        image = nib.Nifti1Image(values, np.eye(4))
        image.header.set_xyzt_units("mm")
        image.set_qform(np.eye(4), code=1)
        image.set_sform(np.eye(4), code=1)
        nib.save(image, path)
    constructed = []
    original = imaging.CaseData

    def count_masks(*args, **kwargs):
        constructed.append(len(kwargs.get("compartments", {})))
        return original(*args, **kwargs)

    monkeypatch.setattr(imaging, "CaseData", count_masks)
    instance.submit({"id": "labels", "op": "importNifti", "args": {
        "structuralPath": str(structural), "tumorMaskPath": str(annotation)}})
    assert instance.wait_idle(5)
    result = terminal(events, "labels")
    assert len(result) == 1 and result[0]["event"] == "error"
    assert not constructed, "Unbounded label fanout reached full case mask construction"


def test_cancelled_case_install_does_not_leave_unreferenced_binary_assets(runtime, monkeypatch):
    instance, events = runtime
    entered, release = threading.Event(), threading.Event()
    original = instance.session.transfers.array

    def first_binary(*args, **kwargs):
        result = original(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(instance.session.transfers, "array", first_binary)
    instance.submit({"id": "cancelled-case", "op": "createSyntheticCase", "args": {"shape": [24, 24, 24]}})
    assert entered.wait(5)
    instance.submit({"id": "cancel-install", "op": "cancel", "args": {"requestId": "cancelled-case"}})
    release.set()
    assert instance.wait_idle(5)
    assert terminal(events, "cancelled-case")[0]["event"] == "cancelled"
    assert not instance.session.cases
    assert not list(instance.session.transfers.root.glob("*.bin")), (
        "Repeated cancelled imports can accumulate binary files outside the case cache bound")


@pytest.fixture
def native_runtime(tmp_path, monkeypatch):
    """Small real native cutting model; only the patient factory is substituted."""
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
    from resectionlab.native_simulation import NativeSequentialSimulator
    tissue = np.ones((5, 5, 4), bool)
    target = np.zeros(tissue.shape, bool)
    target[2, 2, 2:] = True
    case = CaseData(case_id="bridge-native-audit", mri=np.arange(tissue.size, dtype=np.float32).reshape(tissue.shape),
        compartments={"target": target}, affine=np.eye(4), brain_mask=tissue,
        source_refs=(SourceRef("fixture", "synthetic://bridge-native-audit", provenance="simulated"),))

    def factory(source_case, **options):
        native = NativeResectionConfig(tissue, target.astype(np.int16), np.eye(4),
            AccessWindow((2, 2, -.5), (0, 0, 1), 3.), NATIVE_GENERIC_TOOLS,
            source_case.semantic_hash, "synthetic solid tissue fixture")
        return NativeSequentialSimulator(native, [(2, 2, 3)], max_steps=options["max_steps"],
            max_actions=options["max_actions"], cancelled=options["cancelled"])

    monkeypatch.setattr("resectionlab.native_simulation.make_native_patient_simulator", factory)
    events = []
    instance = bridge.BridgeRuntime(tmp_path / "native-transfer", events.append, run_dir=tmp_path / "runs")
    source = save_case(case, tmp_path / "native.ressectionlab")
    instance.submit({"id": "load-native", "op": "loadCase", "args": {"path": str(source)}})
    assert instance.wait_idle(5)
    assert terminal(events, "load-native")[0]["event"] == "result"
    yield instance, events, case
    instance.close()


def train_native(native_runtime, identity="train"):
    instance, events, case = native_runtime
    instance.submit({"id": identity, "op": "trainPatient", "args": {
        "caseHash": case.semantic_hash, "budgetSeconds": 5, "seed": 11}})
    assert instance.wait_idle(20)
    result = terminal(events, identity)
    assert len(result) == 1 and result[0]["event"] == "result", result
    training = result[0]["result"]
    assert training["training"]["gradient_steps"] > 0
    assert training["training"]["actor_parameters_changed"] is True
    return training


@pytest.mark.parametrize("changed_file", ["checkpoint.pt", "contract.json", "native-refinement.json"])
def test_saved_training_integrity_includes_budget_counters_and_report(native_runtime, changed_file):
    instance, events, case = native_runtime
    run = train_native(native_runtime)
    run_id = run["runId"]
    directory = instance.session.run_dir / run_id
    path = directory / changed_file
    if changed_file == "checkpoint.pt":
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # Preserve policy weights and their hashes; attack only resumable budget state.
        checkpoint["state"]["gradient_steps"] = 0
        checkpoint["state"]["elapsed_seconds"] = 0.
        torch.save(checkpoint, path)
    else:
        data = json.loads(path.read_text())
        data["external_audit_mutation"] = True
        path.write_text(json.dumps(data))
    before = path.read_bytes()
    operation = "replayTraining" if changed_file == "native-refinement.json" else "trainPatient"
    args = {"caseHash": case.semantic_hash,
            "runId" if operation == "replayTraining" else "resumeRunId": run_id}
    instance.submit({"id": "tampered", "op": operation, "args": args})
    assert instance.wait_idle(10)
    outcome = terminal(events, "tampered")
    assert len(outcome) == 1 and outcome[0]["event"] == "error"
    assert outcome[0]["error"]["code"] == "RUN_INTEGRITY_FAILED"
    assert path.read_bytes() == before, "An invalid saved artifact was overwritten instead of refused"


def test_final_worlds_cannot_enter_desktop_training_configuration(native_runtime):
    instance, events, case = native_runtime
    for number, injected in enumerate(({"finalEvaluationSeeds": [123]}, {"optimization": {"role": "final_evaluation"}},
                                        {"reward": {"target_per_mm3": 1000}}, {"outputDir": "/tmp/other-run"})):
        instance.submit({"id": f"injection-{number}", "op": "trainPatient", "args": {
            "caseHash": case.semantic_hash, **injected}})
    assert instance.wait_idle(5)
    for number in range(4):
        outcome = terminal(events, f"injection-{number}")
        assert len(outcome) == 1 and outcome[0]["event"] == "error"
        assert outcome[0]["error"]["code"] == "INVALID_ARGUMENT"
    assert not [path for path in instance.session.run_dir.iterdir() if path.is_dir()]


def test_native_resume_rejects_budget_change_before_training_or_rewriting(native_runtime):
    instance, events, case = native_runtime
    run = train_native(native_runtime)
    directory = instance.session.run_dir / run["runId"]
    manifest_before = (directory / "bridge-run.json").read_bytes()
    checkpoint_before = (directory / "checkpoint.pt").read_bytes()
    instance.submit({"id": "changed-budget", "op": "trainPatient", "args": {
        "caseHash": case.semantic_hash, "resumeRunId": run["runId"], "budgetSeconds": 120}})
    assert instance.wait_idle(5)
    outcome = terminal(events, "changed-budget")[0]
    assert outcome["event"] == "error" and outcome["error"]["code"] == "RESUME_CONTRACT_CHANGED"
    assert (directory / "bridge-run.json").read_bytes() == manifest_before
    assert (directory / "checkpoint.pt").read_bytes() == checkpoint_before


def test_cancellation_keeps_real_checkpoint_but_exposes_no_replay(native_runtime):
    instance, events, case = native_runtime
    original_emit = instance.emit
    requested = threading.Event()

    def emit(event):
        original_emit(event)
        progress = event.get("progress", {})
        if (event.get("id") == "cancelled-training" and event["event"] == "progress"
                and progress.get("phase") == "training" and not requested.is_set()):
            requested.set()
            instance.submit({"id": "cancel-training", "op": "cancel", "args": {"requestId": "cancelled-training"}})

    instance.emit = emit
    instance.submit({"id": "cancelled-training", "op": "trainPatient", "args": {
        "caseHash": case.semantic_hash, "budgetSeconds": 5, "seed": 11}})
    assert instance.wait_idle(20)
    assert requested.is_set(), events
    assert terminal(events, "cancelled-training")[0]["event"] == "cancelled"
    instance.submit({"id": "runs", "op": "listRuns", "args": {"caseHash": case.semantic_hash}})
    assert instance.wait_idle(5)
    runs = terminal(events, "runs")[0]["result"]["runs"]
    assert len(runs) == 1
    assert runs[0]["hasCheckpoint"] is True
    assert runs[0]["hasAcceptedReplay"] is False
    directory = instance.session.run_dir / runs[0]["runId"]
    assert (directory / "checkpoint.pt").is_file()
    instance.submit({"id": "cancelled-replay", "op": "replayTraining", "args": {
        "caseHash": case.semantic_hash, "runId": runs[0]["runId"]}})
    assert instance.wait_idle(5)
    assert terminal(events, "cancelled-replay")[0]["error"]["code"] == "REPLAY_UNAVAILABLE"


def test_incomplete_initial_selection_is_retained_but_never_installed(native_runtime, monkeypatch):
    from resectionlab import learning
    instance, events, case = native_runtime

    def interrupted(*args, **kwargs):
        raise learning.RolloutInterrupted(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete selection must not load a replay checkpoint")

    monkeypatch.setattr(learning, "rollout_policy", interrupted)
    monkeypatch.setattr(learning, "load_policy", forbidden)
    outcome = bridge_request(instance, events, "incomplete-native", "trainPatient",
        {"caseHash": case.semantic_hash, "budgetSeconds": 5, "seed": 11})
    assert outcome["event"] == "result", outcome
    run = outcome["result"]
    assert run["training"]["status"] == "incomplete_selection"
    assert run["training"]["selection_panel_complete"] is False
    assert run["training"]["replay"] is None
    directory = instance.session.run_dir / run["runId"]
    assert (directory / "checkpoint.pt").is_file()
    assert json.loads((directory / "result.json").read_text())["status"] == "wall_time_budget"
    replay = bridge_request(instance, events, "incomplete-replay", "replayTraining",
        {"caseHash": case.semantic_hash, "runId": run["runId"]})
    assert replay["error"]["code"] == "REPLAY_UNAVAILABLE"
    assert not (directory / "native-candidate-freeze.json").exists()


def test_historical_unselected_fallback_replay_is_rejected_before_checkpoint_load(native_runtime, monkeypatch):
    import hashlib
    from resectionlab import learning
    instance, events, case = native_runtime
    run = train_native(native_runtime)
    directory = instance.session.run_dir / run["runId"]
    report_path = directory / "native-refinement.json"
    historical = json.loads(report_path.read_text())
    historical.update(selected_selection_return=None, selection_history=[], status="wall_time_budget")
    # Reproduce a receipt created by the former consumer bug in this private
    # fixture: its bytes are intact and signed, but it never completed selection.
    contents = json.dumps(historical).encode()
    report_path.write_bytes(contents)
    manifest = json.loads((directory / "bridge-run.json").read_text())
    manifest["reportSha256"] = hashlib.sha256(contents).hexdigest()
    instance.session._write_run(directory, manifest)
    instance.session.run_reports.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical incomplete selection must be rejected before checkpoint load")

    monkeypatch.setattr(learning, "load_policy", forbidden)
    replay = bridge_request(instance, events, "historical-incomplete", "replayTraining",
        {"caseHash": case.semantic_hash, "runId": run["runId"]})
    assert replay["error"]["code"] == "REPLAY_UNAVAILABLE", replay
    assert report_path.read_bytes() == contents
    assert run["runId"] not in instance.session.run_reports
    listing = bridge_request(instance, events, "historical-listing", "listRuns", {"caseHash": case.semantic_hash})
    assert listing["result"]["runs"][0]["hasAcceptedReplay"] is False
    assert listing["result"]["runs"][0]["hasCheckpoint"] is True


def test_native_export_and_timeline_share_certified_source_cell_accounting(native_runtime, tmp_path):
    instance, events, case = native_runtime
    run = train_native(native_runtime)
    assert run["training"]["replay_status"] == "accepted_independent_geometry"
    run_id = run["runId"]
    for identity, step in (("start", 0), ("end", None)):
        args = {"caseHash": case.semantic_hash, "runId": run_id}
        if step is not None:
            args["step"] = step
        instance.submit({"id": identity, "op": "replayTraining", "args": args})
        assert instance.wait_idle(5)
        outcome = terminal(events, identity)[0]
        assert outcome["event"] == "result", outcome
        replay = outcome["result"]
        descriptor = replay["removedMask"]
        removed = np.frombuffer(Path(descriptor["path"]).read_bytes(), dtype=np.uint8).reshape(case.mri.shape).astype(bool)
        target = case.compartments["target"]
        assert replay["role"] == "selection" and replay["finalEvaluation"] is False
        assert replay["clinicalDeficitProbability"] is None
        assert replay["simulatedRemovedTargetVolumeMm3"] == (removed & target).sum() * case.voxel_volume_mm3
        assert replay["simulatedRemovedNormalVolumeMm3"] == (removed & ~target).sum() * case.voxel_volume_mm3
        if step == 0:
            assert not removed.any()
        # STOP is a valid selected policy; native geometry tests separately
        # establish positive cutting fixtures without assuming RL beats search.
    destination = tmp_path / "candidate.json"
    instance.submit({"id": "export", "op": "exportCandidate", "args": {
        "caseHash": case.semantic_hash, "runId": run_id, "path": str(destination)}})
    assert instance.wait_idle(5)
    assert terminal(events, "export")[0]["event"] == "result"
    exported = json.loads(destination.read_text())
    candidate = exported["candidate"]
    assert exported["final_evaluation"] is False and exported["clinical_deficit_probability"] is None
    assert candidate["role"] == "selection" and candidate["case_hash"] == case.semantic_hash
    assert candidate["metrics"]["simulated_removed_target_volume_mm3"] == replay["simulatedRemovedTargetVolumeMm3"]
    assert candidate["metrics"]["simulated_removed_normal_volume_mm3"] == replay["simulatedRemovedNormalVolumeMm3"]


def test_native_replay_resealed_wrong_shape_affine_case_or_probability_is_rejected(native_runtime):
    import copy
    from resectionlab.native_refinement import replay_artifact_hash, validate_native_replay
    instance, _events, case = native_runtime
    run = train_native(native_runtime)
    path = instance.session.run_dir / run["runId"] / "native-refinement.json"
    original = json.loads(path.read_text())["replay"]
    assert original is not None
    changes = [
        lambda replay: replay.update(shape=[case.mri.shape[0] + 1, *case.mri.shape[1:]]),
        lambda replay: replay["affine"][0].__setitem__(3, 10.),
        lambda replay: replay["native_certificate"].update(source_case_hash="sha256:another-case"),
        lambda replay: replay["metrics"].update(clinical_deficit_probability=.2),
        lambda replay: replay.update(role="final_evaluation"),
    ]
    for change in changes:
        altered = copy.deepcopy(original)
        change(altered)
        altered["artifact_hash"] = replay_artifact_hash(altered)
        with pytest.raises(ValueError):
            validate_native_replay(case, altered)


def test_independent_rejection_never_becomes_an_accepted_desktop_run(native_runtime, monkeypatch, tmp_path):
    from resectionlab.evaluation import NativeRemovalAudit
    instance, events, case = native_runtime

    def reject(*_args, **_kwargs):
        return NativeRemovalAudit(False, ("independent_audit_failure_fixture",), None, None, None,
                                  0., 0., 0., case.semantic_hash, case.voxel_volume_mm3, 0)

    monkeypatch.setattr("resectionlab.evaluation.independent_check_native_history", reject)
    run = train_native(native_runtime)
    assert run["training"]["replay"] is None
    assert run["training"]["replay_status"] == "rejected_independent_geometry"
    instance.submit({"id": "rejected-list", "op": "listRuns", "args": {"caseHash": case.semantic_hash}})
    assert instance.wait_idle(5)
    listed = terminal(events, "rejected-list")[0]["result"]["runs"]
    assert len(listed) == 1 and listed[0]["hasAcceptedReplay"] is False
    assert listed[0]["hasCheckpoint"] is True
    destination = tmp_path / "must-not-export.json"
    instance.submit({"id": "rejected-export", "op": "exportCandidate", "args": {
        "caseHash": case.semantic_hash, "runId": run["runId"], "path": str(destination)}})
    assert instance.wait_idle(5)
    outcome = terminal(events, "rejected-export")[0]
    assert outcome["event"] == "error" and outcome["error"]["code"] == "REPLAY_UNAVAILABLE"
    assert not destination.exists()


def test_crash_stage_manifest_cannot_admit_an_unsigned_resumable_checkpoint(native_runtime):
    import torch
    instance, events, case = native_runtime
    original_emit = instance.emit
    snapshots = {}

    def emit(event):
        original_emit(event)
        progress = event.get("progress", {})
        if event.get("id") == "crash-stage" and progress.get("phase") == "preparing":
            run_id = progress["runId"]
            path = instance.session.run_dir / run_id / "bridge-run.json"
            snapshots["manifest"] = path.read_bytes()

    instance.emit = emit
    run = train_native(native_runtime, identity="crash-stage")
    directory = instance.session.run_dir / run["runId"]
    assert snapshots, "Expected a signed preparing-stage run manifest"
    # Reconstruct the artifact state possible after a process exit between a
    # checkpoint write and the next signed metadata write; its hash is absent.
    (directory / "bridge-run.json").write_bytes(snapshots["manifest"])
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    checkpoint["state"]["gradient_steps"] = 0
    checkpoint["state"]["elapsed_seconds"] = 0.
    torch.save(checkpoint, directory / "checkpoint.pt")
    instance.submit({"id": "unsigned-resume", "op": "trainPatient", "args": {
        "caseHash": case.semantic_hash, "resumeRunId": run["runId"]}})
    assert instance.wait_idle(10)
    outcome = terminal(events, "unsigned-resume")[0]
    assert outcome["event"] == "error", "An unsigned checkpoint restored mutable training budget state"
    assert outcome["error"]["code"] in {"RUN_INTEGRITY_FAILED", "CHECKPOINT_UNVERIFIED", "RUN_NOT_RESUMABLE"}


def bridge_request(instance, events, identity, operation, args, *, timeout=20):
    instance.submit({"id": identity, "op": operation, "args": args})
    assert instance.wait_idle(timeout)
    outcomes = terminal(events, identity)
    assert len(outcomes) == 1, outcomes
    return outcomes[0]


@pytest.fixture
def route_runtime(tmp_path):
    events = []
    instance = bridge.BridgeRuntime(tmp_path / "route-transfer", events.append, run_dir=tmp_path / "route-runs")
    case_hash = install(instance)
    outcome = bridge_request(instance, events, "base-search", "generateRoutes", {"caseHash": case_hash})
    assert outcome["event"] == "result", outcome
    yield instance, events, case_hash, outcome["result"]
    instance.close()


def test_native_route_append_preserves_original_models_and_pareto_cohorts(route_runtime):
    instance, events, case_hash, original = route_runtime
    assert len(original["candidates"]) == 54
    initial = {item["route_id"]: item for item in original["candidates"]}
    for number in range(2):
        outcome = bridge_request(instance, events, f"append-{number}", "generateNativeRoutes", {"caseHash": case_hash})
        assert outcome["event"] == "result", outcome
        result = outcome["result"]
        candidates = {item["route_id"]: item for item in result["candidates"]}
        assert len(candidates) == 56 and len(result["search_models"]) == 2
        assert result["combined_models"] is True and result["planning_model_hash"] is None
        assert {identity: candidates[identity] for identity in initial} == initial
        memberships = {}
        for model in result["search_models"]:
            for identity in model["candidate_ids"]:
                assert identity not in memberships
                memberships[identity] = model["planning_model_hash"]
        assert memberships.keys() == candidates.keys()
        for identity, candidate in candidates.items():
            assert candidate["planning_model_hash"] == memberships[identity]
            assert all(memberships[other] == memberships[identity] for other in candidate["dominated_by"])
        assert result["requested_candidates"] == sum(model["requested_candidates"] for model in result["search_models"])


def test_selected_lps_route_preserves_exact_canonical_geometry_and_tool(route_runtime, tmp_path):
    from dataclasses import replace
    instance, events, case_hash, _original = route_runtime
    original_case = instance.session.cases[case_hash].case
    conversion = np.diag([-1., -1., 1., 1.])
    case = replace(original_case, frame="LPS+", affine=conversion @ original_case.affine)
    source = save_case(case, tmp_path / "lps-selection.ressectionlab")
    loaded = bridge_request(instance, events, "load-lps-selected", "loadCase", {"path": str(source)})
    assert loaded["event"] == "result", loaded
    generated = bridge_request(instance, events, "lps-native-search", "generateNativeRoutes", {"caseHash": case.semantic_hash})
    assert generated["event"] == "result", generated
    route = next(item for item in generated["result"]["candidates"] if item["tool"]["tool_id"] == "native-wide-aspiration")
    inspected = bridge_request(instance, events, "inspect-lps-selected", "inspectRefinement",
                               {"caseHash": case.semantic_hash, "routeId": route["route_id"]})
    assert inspected["event"] == "result", inspected
    readiness = inspected["result"]
    assert readiness["status"] == "ready" and readiness["legalNonStopActions"] > 0
    binding = readiness["route_binding"]
    assert binding["geometry_frame"] == "RAS+" and binding["mode"] == "exact_selected_route"
    np.testing.assert_array_equal(binding["candidate_entries_mm"], [np.asarray(route["entry_mm"]) * (-1, -1, 1)])
    np.testing.assert_array_equal(binding["candidate_targets_mm"], [np.asarray(route["target_mm"]) * (-1, -1, 1)])
    for key in ("center_mm", "normal_inward"):
        np.testing.assert_array_equal(binding["access"][key], np.asarray(route["window"][key]) * (-1, -1, 1))
    assert binding["access"]["radius_mm"] == route["window"]["radius_mm"]
    assert binding["tools"] == [route["tool"]]
    assert readiness["clinicalDeficitProbability"] is None
    assert not [path for path in instance.session.run_dir.iterdir() if path.is_dir()]


def test_stop_only_selected_route_never_initializes_optimizer(route_runtime, monkeypatch):
    instance, events, case_hash, original = route_runtime
    route = next(item for item in original["candidates"] if item["geometry"]["feasible"])
    attempts = []

    def forbidden_optimizer(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError("A STOP-only model attempted to initialize the learner")

    monkeypatch.setattr("resectionlab.learning.train_patient_policy", forbidden_optimizer)
    outcome = bridge_request(instance, events, "no-action-training", "trainPatient",
                               {"caseHash": case_hash, "routeId": route["route_id"], "budgetSeconds": 5})
    assert outcome["event"] == "result", outcome
    run = outcome["result"]
    report = run["training"]
    assert not attempts
    assert report["status"] == "no_actionable_moves" and report["role"] == "preflight"
    assert report["gradient_steps"] == report["optimization_environment_steps"] == 0
    assert report["actor_parameters_changed"] is False and report["replay"] is None
    assert report["route_binding"]["tools"] == [route["tool"]]
    directory = instance.session.run_dir / run["runId"]
    assert (directory / "native-refinement.json").is_file()
    assert not (directory / "checkpoint.pt").exists() and not (directory / "contract.json").exists()
    listed = bridge_request(instance, events, "no-action-list", "listRuns", {"caseHash": case_hash})
    assert listed["event"] == "result", listed
    summary = next(item for item in listed["result"]["runs"] if item["runId"] == run["runId"])
    assert summary["hasCheckpoint"] is False and summary["hasAcceptedReplay"] is False


def test_changed_cached_tool_is_rejected_before_geometry_or_training(route_runtime, monkeypatch):
    from resectionlab.core import freeze_json
    import copy
    instance, events, case_hash, original = route_runtime
    record = copy.deepcopy(original)
    route = next(item for item in record["candidates"] if item["geometry"]["feasible"])
    route["tool"]["shaft_radius_mm"] *= .5
    instance.session.cases[case_hash].routes = freeze_json(record)
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError("Changed cached tool escaped route binding validation")

    monkeypatch.setattr("resectionlab.native_refinement.inspect_native_refinement", forbidden)
    monkeypatch.setattr("resectionlab.native_refinement.run_native_refinement", forbidden)
    for op in ("inspectRefinement", "trainPatient"):
        result = bridge_request(instance, events, "changed-tool-" + op, op,
                                {"caseHash": case_hash, "routeId": route["route_id"]})
        assert result["event"] == "error" and result["error"]["code"] == "TOOL_UNAVAILABLE"
    assert not attempts
    assert not [path for path in instance.session.run_dir.iterdir() if path.is_dir()]


def test_cancelled_native_append_preserves_original_cache(route_runtime, monkeypatch):
    from resectionlab.native_routes import generate_native_axis_routes
    from resectionlab.core import thaw_json
    instance, events, case_hash, original = route_runtime
    result = generate_native_axis_routes(instance.session.cases[case_hash].case)
    entered, release = threading.Event(), threading.Event()

    def delayed(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr("resectionlab.native_routes.generate_native_axis_routes", delayed)
    instance.submit({"id": "cancelled-append", "op": "generateNativeRoutes", "args": {"caseHash": case_hash}})
    assert entered.wait(5)
    try:
        instance.submit({"id": "cancel-append", "op": "cancel", "args": {"requestId": "cancelled-append"}})
    finally:
        release.set()
    assert instance.wait_idle(5)
    assert terminal(events, "cancelled-append")[0]["event"] == "cancelled"
    assert thaw_json(instance.session.cases[case_hash].routes) == original


def test_stale_case_cannot_fall_back_to_current_route_or_factory(route_runtime, monkeypatch):
    from resectionlab.core import thaw_json
    instance, events, case_hash, original = route_runtime
    route = original["candidates"][0]
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError("A stale case request reached current anatomy")

    monkeypatch.setattr("resectionlab.native_routes.generate_native_axis_routes", forbidden)
    monkeypatch.setattr("resectionlab.native_refinement.inspect_native_refinement", forbidden)
    monkeypatch.setattr("resectionlab.native_refinement.run_native_refinement", forbidden)
    for operation in ("generateNativeRoutes", "inspectRefinement", "trainPatient"):
        args = {"caseHash": "sha256:" + "0" * 64}
        if operation != "generateNativeRoutes":
            args["routeId"] = route["route_id"]
        outcome = bridge_request(instance, events, "stale-" + operation, operation, args)
        assert outcome["event"] == "error", outcome
    assert not attempts
    assert thaw_json(instance.session.cases[case_hash].routes) == original
    assert not [path for path in instance.session.run_dir.iterdir() if path.is_dir()]


@pytest.mark.parametrize("changed_component", ["tool", "window"])
def test_factory_cannot_substitute_selected_tool_or_window(route_runtime, monkeypatch, changed_component):
    from dataclasses import replace
    from resectionlab.native_simulation import NativeSequentialSimulator, make_native_patient_simulator
    instance, events, case_hash, _original = route_runtime
    generated = bridge_request(instance, events, "factory-native-search", "generateNativeRoutes", {"caseHash": case_hash})
    assert generated["event"] == "result", generated
    route = generated["result"]["candidates"][-1]

    def substituted(case, **options):
        original = make_native_patient_simulator(case, **options)
        config = original.native_config
        if changed_component == "tool":
            config = replace(config, tools=(replace(config.tools[0], shaft_radius_mm=.2),))
        else:
            config = replace(config, access=replace(config.access, radius_mm=config.access.radius_mm + 1.))
        return NativeSequentialSimulator(config, original.candidate_tips_mm,
            candidate_entries_mm=original.candidate_entries_mm, max_steps=options["max_steps"],
            max_actions=options["max_actions"], cancelled=options["cancelled"])

    monkeypatch.setattr("resectionlab.native_simulation.make_native_patient_simulator", substituted)
    outcome = bridge_request(instance, events, "substituted-selected-geometry", "inspectRefinement",
                              {"caseHash": case_hash, "routeId": route["route_id"]})
    assert outcome["event"] == "error", "Exact points must not conceal a substituted instrument or access window"
    assert any(word in outcome["error"]["message"].lower() for word in ("tool", "window", "geometry", "access"))
    assert not [path for path in instance.session.run_dir.iterdir() if path.is_dir()]


def test_refused_resume_preserves_existing_accepted_report(native_runtime, monkeypatch):
    instance, events, case = native_runtime
    run = train_native(native_runtime)
    assert run["training"]["replay_status"] == "accepted_independent_geometry"
    directory = instance.session.run_dir / run["runId"]
    names = ("bridge-run.json", "checkpoint.pt", "contract.json", "native-refinement.json", "native-request.json")
    before = {name: (directory / name).read_bytes() for name in names}

    def refuse_resume(*_args, **_kwargs):
        raise ValueError("Resume numerical source changed; historical native results remain preserved")

    monkeypatch.setattr("resectionlab.native_refinement.run_native_refinement", refuse_resume)
    outcome = bridge_request(instance, events, "source-refused-resume", "trainPatient",
                             {"caseHash": case.semantic_hash, "resumeRunId": run["runId"]})
    assert outcome["event"] == "error", outcome
    assert {name: (directory / name).read_bytes() for name in names if name != "bridge-run.json"} == {
        name: contents for name, contents in before.items() if name != "bridge-run.json"}
    previous_manifest = json.loads(before["bridge-run.json"])
    current_manifest = json.loads((directory / "bridge-run.json").read_bytes())
    previous_manifest.pop("integrity")
    current_manifest.pop("integrity")
    attempt = current_manifest.pop("lastResumeAttempt", None)
    assert attempt is None or attempt["status"] == "failed"
    assert current_manifest == previous_manifest
    instance.session.run_reports.clear()
    reopened = bridge_request(instance, events, "prior-replay", "replayTraining",
                              {"caseHash": case.semantic_hash, "runId": run["runId"]})
    assert reopened["event"] == "result", reopened


def test_native_request_geometry_integrity_is_checked_before_resume(native_runtime, monkeypatch):
    instance, events, case = native_runtime
    run = train_native(native_runtime)
    directory = instance.session.run_dir / run["runId"]
    path = directory / "native-request.json"
    data = json.loads(path.read_text())
    data["request"]["geometry"]["max_actions"] += 1
    path.write_text(json.dumps(data))
    before = (directory / "bridge-run.json").read_bytes()
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError("Changed native route request reached the facade")

    monkeypatch.setattr("resectionlab.native_refinement.run_native_refinement", forbidden)
    outcome = bridge_request(instance, events, "altered-native-request", "trainPatient",
                             {"caseHash": case.semantic_hash, "resumeRunId": run["runId"]})
    assert outcome["event"] == "error" and outcome["error"]["code"] == "RUN_INTEGRITY_FAILED", outcome
    assert not attempts
    assert (directory / "bridge-run.json").read_bytes() == before
