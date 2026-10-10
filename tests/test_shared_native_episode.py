"""Generated controls for one native state/action/search/display pathway."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.development_episode import (make_development_task, public_display_case,
    execute_development_episode, plan_development_episode, _select, TOOLS)
from resectionlab.evaluation import independent_check_native_history
from resectionlab.simulation import InvalidActionError


@pytest.fixture(scope="module")
def scripted():
    return execute_development_episode()


def test_executed_sequence_has_two_mechanisms_and_real_persistent_effect(scripted):
    case, episode = scripted
    assert case.metadata["is_synthetic"] is True
    assert case.mri.shape == (13, 13, 12) and case.brain_mask.sum() == 175
    assert [r["interaction_mode"] for r in episode["history"]] == ["aspirate", "probe", "aspirate", "probe", "aspirate", "stop"]
    assert [len(r["removed_indices_native"]) for r in episode["history"]] == [1, 0, 2, 0, 2, 0]
    assert episode["geometryAudit"]["feasible"]
    assert episode["metrics"]["target_removed_mm3"] == 2.
    assert episode["metrics"]["normal_removed_mm3"] == 3.
    assert episode["sequentialEffect"]["preOpeningProbeFeasible"] is False
    refusal = episode["attemptDiagnostics"][0]
    assert refusal["status"] == "rejected" and refusal["stateBefore"] == refusal["stateAfter"]
    assert not refusal["removedIndicesNative"]
    assert all(r["source_state_hash"] != r["result_state_hash"] for r in episode["history"][:-1])


def test_replay_cell_deltas_and_recorded_reverse_poses_exactly_reproduce_native_masks(scripted):
    case, episode = scripted
    removed, contact, probe = (np.zeros(case.mri.shape, bool) for _ in range(3))
    prior = None
    for frame in episode["replayFrames"]:
        if prior is not None:
            assert frame["stateBefore"] == prior
        prior = frame["stateAfter"]
        for mask, key in ((removed, "removedIndicesNative"), (contact, "contactIndicesNative"), (probe, "probeContactIndicesNative")):
            cells = np.asarray(frame[key], dtype=int).reshape(-1, 3)
            if len(cells):
                mask[tuple(cells.T)] = True
        assert array_digest(removed) == frame["cavityHash"]
        assert array_digest(case.brain_mask & ~removed) == frame["remainingHash"]
        assert array_digest(contact) == frame["contactHash"]
        assert array_digest(probe) == frame["probeContactHash"]
        if frame["phase"] in {"withdrawal", "stop"}:
            assert not frame["removedIndicesNative"] and not frame["contactIndicesNative"]
    assert np.argwhere(removed).tolist() == episode["finalRemovedIndicesNative"]
    for index, action in enumerate(episode["history"][:-1]):
        frames = [f for f in episode["replayFrames"] if f["actionIndex"] == index]
        assert [f["tipRasMm"] for f in frames if f["phase"] == "insertion"] == [m["tip_end_mm"] for m in action["microsteps"]]
        assert [f["tipRasMm"] for f in frames if f["phase"] == "withdrawal"] == [m["tip_start_mm"] for m in reversed(action["microsteps"])]
    assert episode["initialStateId"] == episode["replayFrames"][0]["stateAfter"]
    assert episode["finalStateId"] == prior


def test_public_display_and_nominal_plan_ignore_private_reference():
    original = make_development_task()
    changed = make_development_task(reference_target=np.zeros(original.case.observed_support.shape, np.float32))
    assert public_display_case(original).semantic_hash == public_display_case(changed).semantic_hash
    assert original.observation().fingerprint == changed.observation().fingerprint
    first, seal, _ = plan_development_episode(original, "scripted")
    second, other_seal, _ = plan_development_episode(changed, "scripted")
    assert seal == other_seal and first == second
    assert semantic_digest(first) == seal
    for action in first["actions"]:
        original.step(action)
        changed.step(action)
    np.testing.assert_array_equal(original._engine.removed_mask, changed._engine.removed_mask)
    assert original.metrics()["target_removed_mm3"] == 2.
    assert changed.metrics()["target_removed_mm3"] == 0.


def test_search_uses_same_native_task_without_artificial_probe_utility():
    _, episode = execute_development_episode(selector="SEARCH")
    assert episode["planning"]["selector"] == "existing_observed_beam_search"
    assert episode["planning"]["model_transition_calls"] <= 24
    assert episode["planning"]["actor_forward_calls"] == 0
    assert episode["planning"]["learnedPolicyExecuted"] is False
    assert all(row["interaction_mode"] != "probe" for row in episode["history"])
    assert episode["metrics"]["target_removed_mm3"] >= 2.
    assert episode["geometryAudit"]["feasible"]


def test_probe_changes_observed_contact_and_state_but_not_cavity_and_stale_ids_fail():
    task = make_development_task()
    task.step(_select(task, TOOLS[0].tool_id, (6, 6, 2)))
    before = task.observation()
    cavity = task._engine.removed_mask.copy()
    probe = _select(task, TOOLS[1].tool_id, (6, 6, 2))
    result = task.step(probe)
    assert result.info["removed_indices_native"] == []
    np.testing.assert_array_equal(cavity, task._engine.removed_mask)
    assert not before.observed_probe_contact_grid.any()
    assert task.observation().observed_probe_contact_grid.sum() == 5
    with pytest.raises(InvalidActionError):
        task.step(probe)
    clone = task.clone()
    with pytest.raises(ValueError):
        clone._engine.probe_contact_mask[:] = False
    clone._engine.probe_contact_mask = clone._engine.probe_contact_mask.copy()
    clone._engine.probe_contact_mask[:] = False
    with pytest.raises(RuntimeError):
        clone.observation()
    assert task.observation().observed_probe_contact_grid.sum() == 5


def test_frozen_probe_registry_prevents_relabelling_away_contact_constraints():
    task = make_development_task()
    task.step(_select(task, TOOLS[0].tool_id, (6, 6, 2)))
    task.step(_select(task, TOOLS[1].tool_id, (6, 6, 2)))
    task._history[1]["interaction_mode"] = "aspirate"
    # Test oracle directly: the task itself already rejects this state tamper.
    from types import SimpleNamespace
    source = SimpleNamespace(mri=task.case.structural_intensity, affine=task.case.affine_ras_mm,
                             frame="RAS+", semantic_hash=task.case.source_hash)
    result = independent_check_native_history(source, task.case.tools, task._history,
        tissue_mask=task.case.observed_support, access=task.case.access, tool_modes=task.tool_modes)
    assert not result.feasible and "native_interaction_differs_from_frozen_tool_registry" in result.failures


def test_cancel_and_bad_selector_do_not_masquerade_as_stop():
    with pytest.raises(InterruptedError):
        execute_development_episode(cancelled=lambda: True)
    with pytest.raises(ValueError):
        execute_development_episode(selector="RL")
