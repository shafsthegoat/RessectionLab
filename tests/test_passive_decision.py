"""Generated bounded-information controls; no patient or learned model."""
from __future__ import annotations

import json

import numpy as np
import pytest

from resectionlab.core import semantic_digest


from resectionlab import persistent_passive as module


def _task(translation=(0., 0., 0.), bound=.1, *, paired=True):
    return module.make_passive_decision_task(translation_ras_mm=translation,
        worst_case_error_bound_mm=bound, paired=paired)


def _masks(task):
    return task._engine.committed_mask_digests()


def test_hidden_measurements_have_identical_public_prefix(monkeypatch):
    alternatives = ((0., 0., 0.), (2., 0., 0.), None)
    tasks = [_task(value, .1 if value is not None else None) for value in alternatives]
    calls = []
    native = module.check_motion
    def counted(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)
    monkeypatch.setattr(module, "check_motion", counted)
    initial = [t.observation() for t in tasks]
    assert len(calls) == 0
    assert len({o.fingerprint for o in initial}) == 1
    assert all(o.registration_observation is None and o.candidate_decision is None for o in initial)
    first_records = [t.step(module.MOVE).info for t in tasks]
    held = [t.observation() for t in tasks]
    assert first_records[0] == first_records[1] == first_records[2]
    assert len({o.fingerprint for o in held}) == 1
    assert all(o.registration_observation is None and o.candidate_decision is None for o in held)
    assert all(o.action_ids == (module.STOP, module.INSPECT) for o in held)
    assert len(calls) == len(tasks)  # Physical MOVE only; no pre-reveal candidate preview.


def test_revealed_measurement_changes_action_and_preserves_physical_masks():
    task = _task()
    before = _masks(task)
    task.step(module.MOVE)
    task.step(module.INSPECT)
    observed = task.observation()
    assert list(observed.registration_observation["translationNavToSourceRasMm"]) == [0., 0., 0.]
    assert observed.candidate_decision["eligible"] is True
    assert observed.candidate_decision["inflatedToolRadiiMm"] == {"tip": .25, "shaft": .2}
    assert module.ADVANCE in observed.action_ids
    step = task.step(module.ADVANCE)
    assert step.info["interaction_mode"] == "passive_motion"
    assert task._operative_record()["moving_tool"]["pose"]["tip_mm"] == [6.5, 6., 6.2]
    task.step(module.WITHDRAW)
    task.step(module.WITHDRAW_STATIONARY)
    task.step(module.STOP)
    episode = module.export_passive_episode(task)
    assert episode["schema"] == "resectionlab.generated-persistent-passive-decision-episode.v2"
    assert len(episode["actions"]) == 6 and len(episode["frames"]) == 7
    assert episode["informationEventIndex"] == 1
    assert episode["physicalHistoryHash"] == semantic_digest(episode["history"])
    assert json.loads(episode["physicalHistoryCanonicalJson"]) == episode["history"]
    assert _masks(task) == before


@pytest.mark.parametrize("translation,bound,expected", [
    ((2., 0., 0.), .1, "mapped_waypoint_not_forward_on_fixed_axis"),
    ((0., 0., 0.), .8, "declared_error_bound_exceeds_public_limit"),
    (None, None, "registration_measurement_unavailable"),
    ((0., 0., .5), .1, "bounded_translation_violates_access_direction_or_reach"),
])
def test_rejected_measurement_retains_withdrawal_and_unknowns(translation, bound, expected):
    task = _task(translation, bound)
    initial = _masks(task)
    task.step(module.MOVE)
    task.step(module.INSPECT)
    observed = task.observation()
    assert observed.candidate_decision["reason"] == expected
    assert module.ADVANCE not in observed.action_ids
    if translation is None:
        assert observed.registration_observation["translationNavToSourceRasMm"] is None
        assert observed.registration_observation["worstCaseErrorBoundMm"] is None
    with pytest.raises(ValueError, match="Unknown, stale"):
        task.step(module.ADVANCE)
    task.step(module.WITHDRAW)
    task.step(module.WITHDRAW_STATIONARY)
    task.step(module.STOP)
    assert task._history[-1]["stop_reason"] == "registration_measurement_unavailable_or_unqualified"
    assert _masks(task) == initial


def test_eligible_advance_may_be_declined_without_false_unavailability():
    episode = module.run_passive_decision_episode(choose_advance=False)
    assert module.ADVANCE not in episode["actions"]
    assert episode["history"][1]["information_event"]["candidate_decision"]["eligible"] is True
    assert episode["history"][-1]["stop_reason"] == "declined_eligible_generated_advance"


def test_committed_advance_cannot_be_erased_after_withdrawal():
    task = _task()
    for action in (module.MOVE, module.INSPECT, module.ADVANCE, module.WITHDRAW):
        task.step(action)
    assert task._history[2]["action_id"] == module.ADVANCE
    task._advance_pose = None
    with pytest.raises(RuntimeError, match="committed procedure state changed"):
        task.observation()


def test_revealed_measurement_clone_is_branch_isolated():
    task = _task()
    task.step(module.MOVE)
    task.step(module.INSPECT)
    original = task.observation().fingerprint
    branch = task.clone()
    assert branch.observation().fingerprint == original
    assert branch._visible_registration is not task._visible_registration
    branch._visible_registration["translationNavToSourceRasMm"][0] = 2.
    with pytest.raises(RuntimeError, match="committed procedure state changed"):
        branch.observation()
    assert task.observation().fingerprint == original


def test_v1_generated_episode_id_and_default_record_unchanged():
    episode = module.run_passive_development_episode(paired=True)
    assert episode["episodeId"] == "sha256:33cbb8bdea5873acece2461fbf1c992791ff7cae18ba2099c4bc1724cd76717a"
    assert episode["schema"] == "resectionlab.generated-persistent-passive-episode.v1"
    assert "decisionProtocol" not in episode and "revealedRegistration" not in episode
