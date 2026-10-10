"""Committed mask ownership and digest contracts on generated native fixtures."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab import native_resection as native
from resectionlab.development_episode import make_development_task, TOOLS, _select
from resectionlab.native_spatial_task import make_native_opening_task


MASKS = ("remaining_mask", "removed_mask", "contact_mask", "probe_contact_mask", "connected_free_mask")


def opening(task):
    return next(row["action_id"] for row in task.candidate_inventory()["ledger"]
                if row.get("voxel") == [4, 4, 1] and row["tool_id"] == task.case.tools[0].tool_id
                and row["feasible"])


@pytest.mark.parametrize("name", MASKS)
def test_masks_reject_alias_writes_and_equal_byte_mutable_replacement(name):
    task = make_native_opening_task().planning_clone()
    engine = task._engine
    original = getattr(engine, name)
    with pytest.raises(ValueError):
        original.flat[0] = not original.flat[0]
    with pytest.raises(ValueError):
        original.setflags(write=True)
    alias = original.view(np.ndarray)
    with pytest.raises(ValueError):
        alias.setflags(write=True)
    for attribute, value in (("shape", (original.size,)), ("dtype", np.uint8), ("strides", (1, 1, 1))):
        with pytest.raises(ValueError):
            setattr(original, attribute, value)
    setattr(engine, name, original.copy())
    with pytest.raises(RuntimeError):
        task.advance_planning("STOP")
    with pytest.raises(RuntimeError):
        engine.clone()
    assert task._steps == engine.revision == 0


@pytest.mark.parametrize("change", ("mode", "cache", "digest", "history", "scalar"))
def test_storage_identity_preserves_task_tamper_guards(change):
    task = make_native_opening_task().planning_clone()
    engine = task._engine
    if change == "mode":
        with pytest.raises(AttributeError):
            engine._immutable_state = False
        object.__setattr__(engine, "_immutable_state", False)
    elif change == "cache":
        with pytest.raises(AttributeError):
            engine._committed_snapshots = ()
        object.__setattr__(engine, "_committed_snapshots", tuple(replace(s) for s in engine._committed_snapshots))
    elif change == "digest":
        object.__setattr__(engine._committed_snapshots[0], "digest", "sha256:" + "0" * 64)
    elif change == "history":
        task._history.append({"action_id": "STOP", "reward": 123})
    else:
        task._total_reward = 123
    with pytest.raises(RuntimeError):
        task.advance_planning("STOP")
    assert task._steps == engine.revision == 0


def test_shared_snapshots_preserve_clone_isolation_and_standalone_default():
    task = make_native_opening_task().planning_clone()
    sibling = task.clone()
    assert task._engine._committed_snapshots is sibling._engine._committed_snapshots
    for name in MASKS:
        assert getattr(task._engine, name) is getattr(sibling._engine, name)
    original = copy.deepcopy(task._state_record())
    sibling.advance_planning(opening(sibling))
    assert task._state_record() == original
    assert sibling._state_record() != original
    mutable = native.NativeResectionEngine(task._config)
    clone = mutable.clone()
    assert mutable.remaining_mask.flags.writeable
    assert clone.remaining_mask is not mutable.remaining_mask
    clone.removed_mask[0, 0, 0] = True
    assert not mutable.removed_mask[0, 0, 0]


@pytest.mark.parametrize("failure", ("foreign", "stale", "altered", "allocation"))
def test_failed_certificate_or_allocation_preserves_committed_state(monkeypatch, failure):
    task = make_native_opening_task().planning_clone()
    engine = task._engine
    certificate = task._inventory[opening(task)]
    if failure == "foreign":
        certificate = replace(certificate)
    elif failure == "altered":
        object.__setattr__(certificate, "tip_mm", (certificate.tip_mm[0] + 1., *certificate.tip_mm[1:]))
    elif failure == "stale":
        engine.commit_preview(certificate)
    else:
        def fail(value):
            raise MemoryError("generated capture allocation fault")
        monkeypatch.setattr(native._CommittedMaskSnapshot, "capture", staticmethod(fail))
    masks = [getattr(engine, name) for name in MASKS]
    history, state, revision = copy.deepcopy(engine.history), engine.state_hash, engine.revision
    with pytest.raises((ValueError, MemoryError)):
        engine.commit_preview(certificate)
    assert all(getattr(engine, name) is old for name, old in zip(MASKS, masks))
    assert engine.history == history and engine.state_hash == state and engine.revision == revision


def test_failed_reset_keeps_prior_snapshot_and_task_seal(monkeypatch):
    task = make_native_opening_task().planning_clone()
    task.advance_planning(opening(task))
    engine = task._engine
    original = copy.deepcopy(task._state_record())
    snapshots, seal = engine._committed_snapshots, task._state_seal
    capture = native._CommittedMaskSnapshot.capture
    calls = []
    def fail_second(value):
        calls.append(None)
        if len(calls) == 2:
            raise MemoryError("generated reset allocation fault")
        return capture(value)
    monkeypatch.setattr(native._CommittedMaskSnapshot, "capture", staticmethod(fail_second))
    with pytest.raises(MemoryError):
        engine.reset()
    assert engine._committed_snapshots is snapshots
    assert task._state_record() == original and task._state_seal == seal
    task._assert_frozen()


def test_unchanged_guards_and_clone_reuse_digests(monkeypatch):
    task = make_native_opening_task().planning_clone()
    calls, digest = [], native.array_digest
    def counted(array):
        calls.append(array.shape)
        return digest(array)
    monkeypatch.setattr(native, "array_digest", counted)
    task._assert_frozen()
    task.clone()
    task.planning_clone()
    assert calls == []
    task.advance_planning(opening(task))
    assert calls == [task.case.observed_support.shape] * 4
    task._assert_frozen()
    assert len(calls) == 4


def test_probe_reuses_unchanged_cavity_snapshots():
    task = make_development_task()
    for mode, depth in (("aspirate", 2), ("probe", 2), ("aspirate", 4), ("probe", 4)):
        old = task._engine._committed_snapshots
        tool = TOOLS[0 if mode == "aspirate" else 1]
        task.step(_select(task, tool.tool_id, (6, 6, depth)))
        if mode == "probe":
            for name in ("remaining_mask", "removed_mask", "connected_free_mask"):
                assert task._engine._committed_snapshots[MASKS.index(name)] is old[MASKS.index(name)]
