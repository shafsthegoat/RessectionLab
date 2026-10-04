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
