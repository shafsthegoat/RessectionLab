"""Generated parity controls for immutable preview copy-on-first-cut.

Use an isolated package overlay for the candidate. Historical task parity also
uses NATIVE_PREVIEW_BASELINE_SOURCE; no patient/model/checkpoint inputs exist.
"""
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab import native_resection as native
from resectionlab.core import array_digest, thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry, _ImmutableArray


BASELINE_SHA = "2aa191370f2dab441089dcd3d6b925c80d0da450d72fdc464d42fb93d13c8e1f"
MASKS = native._STATE_MASKS


def config(kind="line"):
    tissue = np.zeros((9, 9, 10), bool)
    tissue[4, 4, 1:6] = True
    tools = (ToolGeometry("fine", 1.25, .45, 12., 35., 2.),)
    access = AccessWindow([4, 4, .5], [0, 0, 1], 2.4)
    if kind in {"early", "hard"}:
        tissue[:] = False; tissue[4, 4, 5] = True
        tools = (ToolGeometry("short-active", .9, .1, 10., tip_length_mm=.1),)
        access = AccessWindow([4, 4, 4.49], [0, 0, 1], 2.)
    elif kind == "partial":
        tissue[5, 5, 1] = True
        tools = (ToolGeometry("opener", 2.25, .45, 2.2, 35., .75),
                 ToolGeometry("cutter", .9, 1.1, 12., 35., 3.))
    elif kind == "empty":
        tissue[:] = False
    hard = np.zeros(tissue.shape, bool)
    if kind == "hard": hard[4, 4, 5] = True
    return native.NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), np.eye(4),
        access, tools, "generated-lazy-preview-source", "generated parity only",
        hard_exclusion=hard)


def record(result):
    return {
        "dataclass": json.loads(json.dumps(asdict(result), default=lambda x: x.tolist())),
        "history": result.to_history_record() if result.feasible else None,
        "diagnostic": thaw_json(result.obstruction_diagnostic),
    }


def state(engine):
    return (engine.state_hash, engine.decision_model_hash, engine.revision,
            copy.deepcopy(engine.history), tuple(array_digest(getattr(engine, n)) for n in MASKS))


def track_mask_copies(monkeypatch, engine, *, fail_at=None):
    watched = {id(engine.remaining_mask): "remaining", id(engine.connected_free_mask): "connected_free"}
    # The canonical bytes-backed constructor returns this Python subclass.
    # Never attempt to monkeypatch the built-in ndarray or replace snapshots.
    assert type(engine.remaining_mask) is _ImmutableArray
    assert type(engine.connected_free_mask) is _ImmutableArray
    original = np.ndarray.copy
    calls = []
    def counted(value, *args, **kwargs):
        if id(value) in watched:
            calls.append(watched[id(value)])
            if len(calls) == fail_at:
                raise MemoryError("generated temporary-mask allocation failure")
        return original(value, *args, **kwargs)
    monkeypatch.setattr(_ImmutableArray, "copy", counted)
    return calls


@pytest.mark.parametrize("kind,tool,tip,reason,copies", [
    ("early", "short-active", [4, 4, 5.49], "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE", 0),
    ("hard", "short-active", [4, 4, 5.49], "HARD_GEOMETRY:", 0),
    ("empty", "fine", [4, 4, 5], "NO_NEW_FULLY_CONTAINED_SURFACE_CELLS", 0),
    ("line", "fine", [4, 4, 5], "NATIVE_CONNECTED_STROKE", 2),
    ("partial", "cutter", [4, 4, 5], "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE", 2),
])
def test_exact_mutable_reference_parity_and_actual_copy_counts(monkeypatch, kind, tool, tip, reason, copies):
    cfg = config(kind)
    eager = native.NativeResectionEngine(cfg)  # Standalone eager-copy path stays unchanged.
    lazy = native.NativeResectionEngine(cfg, immutable_state=True)
    before = state(lazy)
    expected = eager.preview_stroke(tool, tip, obstruction_diagnostics=True)
    calls = track_mask_copies(monkeypatch, lazy)
    actual = lazy.preview_stroke(tool, tip, obstruction_diagnostics=True)
    assert actual.reason.startswith(reason)
    assert len(calls) == copies
    assert record(actual) == record(expected)
    assert state(lazy) == before == state(eager)
    assert all(not getattr(lazy, name).flags.writeable for name in MASKS)
    if kind == "line":
        assert len(actual.removed_indices_native) == 5
        assert sum(bool(len(m.removed_indices_native)) for m in actual.microsteps) > 1
        assert native.NativeResectionEngine._result_digest(actual) == eager._result_digest(expected)
        eager.commit_preview(expected); lazy.commit_preview(actual)
        assert state(lazy) == state(eager)
        assert lazy.history == eager.history
    if kind == "partial":
        diagnostic = thaw_json(actual.obstruction_diagnostic)
        assert diagnostic["prior_temporary_removed_count"] > 0
        assert [5, 5, 1] in diagnostic["blocked_indices_native"]
        assert not actual.removed_indices_native.size and not lazy.removed_mask.any()


@pytest.mark.parametrize("failure_copy", [1, 2])
def test_failed_private_allocation_preserves_committed_state_and_certificates(monkeypatch, failure_copy):
    engine = native.NativeResectionEngine(config(), immutable_state=True)
    old = engine.preview_stroke("fine", [4, 4, 1])
    assert old.feasible
    before, snapshots = state(engine), engine._committed_snapshots
    records, digests = dict(engine._preview_records), dict(engine._preview_digests)
    with monkeypatch.context() as patch:
        calls = track_mask_copies(patch, engine, fail_at=failure_copy)
        with pytest.raises(MemoryError, match="temporary-mask"):
            engine.preview_stroke("fine", [4, 4, 5], obstruction_diagnostics=True)
        assert len(calls) == failure_copy
    assert state(engine) == before and engine._committed_snapshots is snapshots
    assert engine._preview_records == records and engine._preview_digests == digests
    engine.committed_mask_digests()
    engine.commit_preview(old)  # Existing valid certificates survive allocation failure.


def test_failed_geometry_after_temporary_cut_preserves_original_masks(monkeypatch):
    engine = native.NativeResectionEngine(config(), immutable_state=True)
    before, snapshots = state(engine), engine._committed_snapshots
    original = native._extend_connected_free
    def fail_after_mutation(cells, remaining, connected_free, domain):
        original(cells, remaining, connected_free, domain)
        assert remaining is not engine.remaining_mask
        assert connected_free is not engine.connected_free_mask
        raise RuntimeError("generated post-cut geometry failure")
    monkeypatch.setattr(native, "_extend_connected_free", fail_after_mutation)
    with pytest.raises(RuntimeError, match="post-cut"):
        engine.preview_stroke("fine", [4, 4, 5])
    assert state(engine) == before and engine._committed_snapshots is snapshots
    engine.committed_mask_digests()


def test_probe_and_repeat_without_new_removal_never_copy_committed_masks(monkeypatch):
    from resectionlab.development_episode import make_development_task, TOOLS, _select
    task = make_development_task()
    task.step(_select(task, TOOLS[0].tool_id, (6, 6, 2)))
    engine = task._engine
    before = state(engine)
    calls = track_mask_copies(monkeypatch, engine)
    probe = engine.preview_stroke(TOOLS[1].tool_id, [6, 6, 2], interaction_mode="probe")
    assert probe.feasible and not probe.removed_indices_native.size
    repeat = engine.preview_stroke(TOOLS[0].tool_id, [6, 6, 2])
    assert not repeat.feasible and repeat.reason == "NO_NEW_FULLY_CONTAINED_SURFACE_CELLS"
    assert calls == [] and state(engine) == before


def test_borrowed_snapshot_replacement_still_refused_before_geometry(monkeypatch):
    engine = native.NativeResectionEngine(config(), immutable_state=True)
    engine.remaining_mask = engine.remaining_mask.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("geometry must not run")
    monkeypatch.setattr(native, "check_motion", forbidden)
    with pytest.raises(RuntimeError, match="replaced"):
        engine.preview_stroke("fine", [4, 4, 5])


def test_standalone_mutable_engine_retains_eager_copy_behavior(monkeypatch):
    engine = native.NativeResectionEngine(config("early"))
    calls = track_mask_copies(monkeypatch, engine)
    before = state(engine)
    result = engine.preview_stroke("short-active", [4, 4, 5.49])
    assert not result.feasible and calls == ["remaining", "connected_free"]
    assert engine.remaining_mask.flags.writeable and state(engine) == before


def test_historical_mixed_task_full_identity_history_clone_stop_and_reset_parity(monkeypatch):
    path = os.environ.get("NATIVE_PREVIEW_BASELINE_SOURCE")
    if not path:
        pytest.skip("historical exact-file parity requires its pinned baseline source")
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASELINE_SHA
    name = "resectionlab._lazy_preview_historical_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    baseline = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, baseline)
    spec.loader.exec_module(baseline)
    from resectionlab import native_spatial_task as spatial
    from resectionlab.development_episode import make_development_task, TOOLS, _select
    def run(engine_class):
        with monkeypatch.context() as patch:
            patch.setattr(spatial, "NativeResectionEngine", engine_class)
            task = make_development_task()
            records = [(task.case.source_hash, task.decision_model_hash, task.observation().fingerprint,
                        task.candidate_inventory())]
            for tool, depth in ((TOOLS[0], 2), (TOOLS[1], 2), (TOOLS[0], 4), (TOOLS[1], 4)):
                action = _select(task, tool.tool_id, (6, 6, depth))
                untouched = state(task._engine)
                clone = task.clone(); clone.step("STOP")
                assert state(task._engine) == untouched
                task.step(action)
                records.append((action, state(task._engine), task.observation().fingerprint))
            task.step("STOP")
            records.append(task.metrics())
            task._engine.reset()
            records.append(state(task._engine))
            return records
    assert run(native.NativeResectionEngine) == run(baseline.NativeResectionEngine)
