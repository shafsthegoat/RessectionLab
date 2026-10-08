"""Software controls using the exact released six-B record; no raw/V access."""

import json
from pathlib import Path
import threading

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.observed_landmark_replay import (
    B_IDS, FIELD_PATH, FIELD_SHA256, ObservedLandmarkError, ObservedLandmarkReplay, STUDY_ID,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source():
    # Never replace unavailable real evidence with a constructed patient.
    return json.loads((ROOT / FIELD_PATH).read_bytes())["baseline"]


def test_actual_six_observations_two_phases_and_frozen_prediction(source):
    replay = ObservedLandmarkReplay.load(ROOT)
    before = replay.frame()
    assert before["prediction"] is None
    assert before["stateHash"] == semantic_digest(before["state"])
    assert before["resultHash"] == semantic_digest({key: value for key, value in before.items() if key not in {"resultHash", "snapshot"}})
    assert before["binding"]["fieldSha256"] == FIELD_SHA256
    assert [point["landmarkId"] for point in before["state"]["observations"]] == list(B_IDS)
    assert [point["sourceRasMm"] for point in before["state"]["observations"]] == source["observations"]["source_ras_mm"]
    assert "destinationWorldToRasMm" not in before["state"]
    assert "destinationImageSha256" not in before["state"]
    assert all(set(point) == {"landmarkId", "sourceRasMm"} for point in before["state"]["observations"])
    state_hashes = {before["stateHash"]}
    for index, row_id in enumerate(B_IDS):
        during_replay = replay.advance("during", row_id)
        during = during_replay.frame()
        point = during["state"]["observations"][index]
        assert point["observedRasMm"] == source["observations"]["observed_ras_mm"][index]
        assert during["state"]["selectedInspectionPoint"]["positionRasMm"] == point["observedRasMm"]
        np.testing.assert_array_equal(point["measuredDisplacementMm"], np.subtract(point["observedRasMm"], point["sourceRasMm"]))
        expected = np.asarray(source["rigid"]["rotation"]) @ point["sourceRasMm"] + source["rigid"]["translation_mm"]
        np.testing.assert_allclose(during["prediction"]["positionRasMm"], expected, rtol=0, atol=2e-14)
        assert during["stateHash"] not in state_hashes
        state_hashes.add(during["stateHash"])
        for field in ("acquisitionTimestamp", "elapsedSeconds", "measurementAvailabilityTimestamp",
                      "anatomicalTarget", "toolPose", "brainSupport", "cavitySupport", "physicalClearanceMm", "totalRegistrationUncertaintyMm"):
            assert during["state"][field] is None
        assert during["state"]["selectedAction"] == "ABSTAIN_UNSUPPORTED_CLEARANCE"
        assert during_replay.reopen(before["snapshot"]).frame() == before
        assert ObservedLandmarkReplay.load(ROOT).reopen(during["snapshot"]).frame() == during
        with pytest.raises(ObservedLandmarkError, match="current source, phase, point and frame"):
            during_replay.assert_result_current(before)
        with pytest.raises(ObservedLandmarkError, match="Reopen"):
            during_replay.advance("before", row_id)
    # Output mutation cannot modify the immutable source or another frame.
    before["state"]["observations"][0]["sourceRasMm"][0] = None
    assert replay.frame()["state"]["observations"][0]["sourceRasMm"] == source["observations"]["source_ras_mm"][0]


@pytest.mark.parametrize("field,value", [
    ("phase", "during"), ("landmarkId", 14), ("landmarkId", True), ("schemaVersion", True),
    ("stateHash", "sha256:" + "0" * 64), ("resultHash", "sha256:" + "0" * 64),
    ("binding", {}), ("kind", "other"), ("extra", "injected"),
])
def test_snapshot_tampering_refuses(field, value):
    replay = ObservedLandmarkReplay.load(ROOT)
    snapshot = replay.snapshot()
    snapshot[field] = value
    with pytest.raises(ObservedLandmarkError):
        replay.reopen(snapshot)


def test_changed_frame_result_refuses_even_with_recomputed_hash():
    replay = ObservedLandmarkReplay.load(ROOT)
    result = replay.frame()
    result["state"]["frame"] = "LPS+"
    result["stateHash"] = semantic_digest(result["state"])
    with pytest.raises(ObservedLandmarkError, match="current source, phase, point and frame"):
        replay.assert_result_current(result)


def test_loader_reads_only_fixed_field_and_source_refusals(tmp_path, monkeypatch):
    original = Path.open
    opened = []

    def restricted(path, *args, **kwargs):
        opened.append(path)
        assert path == ROOT / FIELD_PATH
        return original(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", restricted)
        ObservedLandmarkReplay.load(ROOT).advance("during", 1).frame()
    assert opened == [ROOT / FIELD_PATH]
    for root in (None, tmp_path, Path("relative")):
        with pytest.raises(ObservedLandmarkError) as error:
            ObservedLandmarkReplay.load(root)
        assert error.value.code == "OBSERVED_SOURCE_UNAVAILABLE"
    destination = tmp_path / FIELD_PATH
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / FIELD_PATH).read_bytes() + b" ")
    with pytest.raises(ObservedLandmarkError) as error:
        ObservedLandmarkReplay.load(tmp_path)
    assert error.value.code == "OBSERVED_SOURCE_CHANGED"
    destination.unlink()
    destination.symlink_to(ROOT / FIELD_PATH)
    with pytest.raises(ObservedLandmarkError) as error:
        ObservedLandmarkReplay.load(tmp_path)
    assert error.value.code == "OBSERVED_SOURCE_INVALID"


@pytest.fixture
def bridge(tmp_path):
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append, observed_source_root=ROOT)
    yield runtime, events
    runtime.close()


def request(bridge, identity, **args):
    runtime, events = bridge
    runtime.submit({"id": identity, "op": "inspectObservedLandmarkUpdate", "args": {"studyId": STUDY_ID, **args}})
    assert runtime.wait_idle(10)
    return [event for event in events if event["id"] == identity][-1]


def test_actual_bridge_route_stale_reopen_and_no_anatomical_case(bridge):
    before = request(bridge, "open", action="open")["result"]
    during = request(bridge, "during", action="advance", expectedStateHash=before["stateHash"], phase="during", landmarkId=1)["result"]
    stale = request(bridge, "stale", action="advance", expectedStateHash=before["stateHash"], phase="during", landmarkId=14)
    assert stale["error"]["code"] == "OBSERVED_STATE_STALE"
    assert request(bridge, "current", action="open")["result"] == during
    changed_row = request(bridge, "row", action="advance", expectedStateHash=during["stateHash"], phase="during", landmarkId=14)["result"]
    assert changed_row["stateHash"] != during["stateHash"]
    backwards = request(bridge, "back", action="advance", expectedStateHash=changed_row["stateHash"], phase="before", landmarkId=1)
    assert backwards["error"]["code"] == "OBSERVED_REPLAY_BACKWARDS"
    reopened = request(bridge, "reopen", action="reopen", expectedStateHash=changed_row["stateHash"], snapshot=before["snapshot"])
    assert reopened["result"] == before
    assert not bridge[0].session.cases and not bridge[0].session.replay_transfers
    bridge[0].session.observed_replay = None
    assert request(bridge, "fresh-reopen", action="reopen", expectedStateHash=None, snapshot=during["snapshot"])["result"] == during


@pytest.mark.parametrize("injected", ["path", "observedSourceRoot", "source_ras_mm", "observed_ras_mm", "frame", "method", "config", "toolPose"])
def test_bridge_refuses_renderer_inputs(bridge, injected):
    refused = request(bridge, "injected", action="open", **{injected: "injected"})
    assert refused["error"]["code"] == "INVALID_ARGUMENT"
    assert bridge[0].session.observed_replay is None


def test_bridge_cancel_before_commit_leaves_actual_state_unchanged(bridge, monkeypatch):
    runtime, events = bridge
    before = request(bridge, "open", action="open")["result"]
    entered, release = threading.Event(), threading.Event()
    original = ObservedLandmarkReplay.advance

    def held(*args):
        candidate = original(*args)
        entered.set()
        assert release.wait(5)
        return candidate

    monkeypatch.setattr(ObservedLandmarkReplay, "advance", held)
    runtime.submit({"id": "advance", "op": "inspectObservedLandmarkUpdate", "args": {
        "studyId": STUDY_ID, "action": "advance", "expectedStateHash": before["stateHash"], "phase": "during", "landmarkId": 1}})
    assert entered.wait(5)
    runtime.submit({"id": "cancel", "op": "cancel", "args": {"requestId": "advance"}})
    release.set()
    assert runtime.wait_idle(5)
    assert [event["event"] for event in events if event["id"] == "advance"] == ["started", "cancelled"]
    assert runtime.session.observed_replay.frame() == before


def test_bridge_rechecks_saved_artifact_and_missing_source(bridge, tmp_path):
    runtime, _ = bridge
    before = request(bridge, "open", action="open")["result"]
    runtime.session.observed_source_root = tmp_path
    refused = request(bridge, "missing", action="advance", expectedStateHash=before["stateHash"], phase="during", landmarkId=1)
    assert refused["error"]["code"] == "OBSERVED_SOURCE_UNAVAILABLE"
    assert runtime.session.observed_replay.frame() == before
