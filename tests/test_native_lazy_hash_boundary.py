"""Generated lazy-boundary integrity and exact search parity; no patient data."""
import numpy as np
import pytest

from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.observed_search import observed_beam_search


@pytest.mark.parametrize("name", ["remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"])
def test_lazy_entry_refuses_every_aspiration_mask_corruption_before_commit(name):
    task = make_native_opening_task().planning_clone()
    before_revision = task._engine.revision
    mask = getattr(task._engine, name)
    mask[0, 0, 0] = not mask[0, 0, 0]
    with pytest.raises(RuntimeError, match="outside a transition"):
        task.advance_planning("STOP")
    assert task._engine.revision == before_revision
    assert task._steps == 0 and task._history == []


def test_lazy_entry_refuses_history_corruption_and_nonplanning_use():
    task = make_native_opening_task().planning_clone()
    task._history.append({"action_id": "STOP", "reward": 999})
    with pytest.raises(RuntimeError, match="outside a transition"):
        task.advance_planning("STOP")
    with pytest.raises(ValueError, match="LAZY_PLANNING_ONLY"):
        make_native_opening_task().advance_planning("STOP")


def test_lazy_transition_keeps_full_entry_check_and_commit_seal_without_duplicate(monkeypatch):
    task = make_native_opening_task().planning_clone()
    action = task.observation().action_ids[1]
    calls = []
    original = NativeSpatialTask._state_record
    def counted(self):
        calls.append((self._steps, self._engine.revision))
        return original(self)
    monkeypatch.setattr(NativeSpatialTask, "_state_record", counted)
    task.advance_planning(action)
    assert calls == [(0, 0), (1, 1)]


def test_prior_redundant_check_and_optimized_search_have_identical_behavior(monkeypatch):
    task = make_native_opening_task()
    original = NativeSpatialTask.advance_planning
    def prior(self, action):
        self._assert_frozen()
        return original(self, action)
    monkeypatch.setattr(NativeSpatialTask, "advance_planning", prior)
    old_actions, old = observed_beam_search(task, max_calls=64, beam_width=2, seconds=20,
                                           transition_mode="lazy_planning")
    monkeypatch.setattr(NativeSpatialTask, "advance_planning", original)
    new_actions, new = observed_beam_search(task, max_calls=64, beam_width=2, seconds=20,
                                           transition_mode="lazy_planning")
    assert old_actions == new_actions
    assert {k:v for k,v in old.items() if k != "planning_seconds"} == {
        k:v for k,v in new.items() if k != "planning_seconds"}
    eager, lazy = task.planning_clone(), task.planning_clone()
    for action in new_actions:
        eager.step(action); lazy.advance_planning(action)
    assert eager.metrics() == lazy.metrics()
    for name in ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"):
        np.testing.assert_array_equal(getattr(eager._engine, name), getattr(lazy._engine, name))
