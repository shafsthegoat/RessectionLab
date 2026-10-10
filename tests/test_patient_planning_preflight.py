"""Generated public native seal/export seam; no patient or model execution."""
import json

import pytest

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.patient_planning_learning import common_patient_policies
from resectionlab.patient_planning_preflight import _seal_and_replay, _history_identity
from test_patient_planning_learning import bound_fixture, trace


@pytest.mark.parametrize("opening", [False, True])
def test_exact_native_replay_and_stop_frame_export(tmp_path, opening):
    task, context, _ = bound_fixture()
    teacher = trace(task, context, opening=opening)
    before = semantic_digest(teacher.history)
    result = _seal_and_replay(task, context, teacher, method="SEARCH", policy=None,
        updates=0, output=tmp_path, guard=lambda: None)
    frames = json.loads((tmp_path/"frames.json").read_text())["frames"]
    assert result["plan_seal"] == semantic_digest(result["plan"])
    assert semantic_digest(teacher.history) == before
    assert "interaction_mode" not in result["plan"]["history"][-1]
    assert frames[-1]["mode"] == "stop"
    assert result["independent_geometry"]["accepted"] is True
    assert result["independent_geometry"]["committed_history_hash"] == semantic_digest(result["replayed_history"])
    assert _history_identity(result["replayed_history"]) == _history_identity(result["plan"]["history"])
    assert all(frame["mode"] in (None, "stop", "aspirate") for frame in frames)
    if opening: assert any(frame["phase"] == "insertion" for frame in frames)


def test_scope_normalization_does_not_hide_physical_or_reward_change():
    planning = [{"action_id": "STOP", "reward": 0., "outcome_scope": "permitted_nominal_model"}]
    replay = [{"action_id": "STOP", "reward": 0., "outcome_scope": "separate_evaluator_reference"}]
    assert _history_identity(planning) == _history_identity(replay)
    replay[0]["reward"] = 1.
    assert _history_identity(planning) != _history_identity(replay)


def test_aggregate_update_declaration_not_weakened_by_reusable_session():
    _, context, protocol = bound_fixture()
    changed = context.record(); changed["max_optimizer_updates"] = 1
    from resectionlab.patient_planning_admission import PatientPlanningContext
    from resectionlab.core import freeze_json
    # A smaller bound remains otherwise intact; factory provenance is not being
    # fabricated as acquired evidence by this generated negative control.
    smaller = PatientPlanningContext(freeze_json(changed), semantic_digest(changed))
    with pytest.raises(ValueError, match="aggregate"):
        common_patient_policies((smaller,), protocol)
