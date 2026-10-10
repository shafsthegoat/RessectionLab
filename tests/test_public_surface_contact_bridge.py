"""Small generated-only bridge controls for the distinct public goal."""
from __future__ import annotations

import json

import pytest

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
from resectionlab.surface_contact_episode import execute_surface_contact_episode


def _run(bridge, *, goal="near", selector="scripted"):
    return bridge.execute("executePublicSurfaceContactEpisode",
        {"fixture": "generated-public-surface-contact-v1", "goalId": goal,
         "selector": selector}, _Request("contact-" + goal + "-" + selector),
        lambda _value, _message: None)


def test_contact_response_is_new_v2_transient_objective_and_clears_old_actor(tmp_path):
    bridge = BridgeSession(tmp_path / "transfer")
    try:
        previous_case, _previous_episode = execute_surface_contact_episode()
        bridge._install_case(previous_case, {}, _Request("prior-case"))
        previous_entry = bridge.cases[previous_case.semantic_hash]
        previous_entry.episode = {"episode": {"episodeId": "old-actor"},
                                  "episodeCanonicalJson": "old-actor"}
        previous_entry.episode_selection = {"episodeId": "old-actor", "frameIndex": 0,
                                            "visible": True}
        previous_entry.comparison_pair = freeze_json({"seal": "old-pair"})
        previous_entry.comparison_actor_episode_id = "old-actor"
        response = _run(bridge)
        episode = response["episode"]
        assert set(response) == {"case", "episode", "episodeCanonicalJson"}
        assert "workspaceSession" not in response["case"]
        assert response["case"]["caseHash"] == episode["caseHash"]
        assert episode["schema"] == "resectionlab.shared-native-development-episode.v2"
        assert episode["taskKind"] == "public_retained_surface_contact"
        assert episode["publicGoal"]["goalId"] == "near"
        assert episode["planning"]["sealedBeforeExecution"] is True
        assert episode["planning"]["referenceScoringPerformed"] is False
        assert episode["planning"]["learnedPolicyExecuted"] is False
        assert episode["patientAdmission"] is False and episode["clinicalValidation"] is False
        assert "tumor" not in json.dumps(episode["metrics"]).lower()
        assert semantic_digest({key: value for key, value in episode.items()
                                if key != "episodeId"}) == episode["episodeId"]
        assert bridge.cases[episode["caseHash"]].episode is None
        assert bridge.cases[episode["caseHash"]].comparison_pair is None
        assert bridge.cases[episode["caseHash"]].public_surface_contact_active is True

        path = tmp_path / "forbidden.ressectionlab"
        with pytest.raises(BridgeError) as denied:
            bridge.execute("saveCase", {"caseHash": episode["caseHash"], "path": str(path),
                "workspace": {}, "overwrite": False}, _Request("save"), lambda *_: None)
        assert denied.value.code == "CONTACT_EPISODE_TRANSIENT"
        assert not path.exists()

        # Installing a case or workspace later clears the transient mode.
        bridge._install_case(previous_case, {}, _Request("replace"))
        assert bridge.cases[previous_case.semantic_hash].public_surface_contact_active is False
    finally:
        bridge.close()


def test_search_and_costly_goal_keep_public_objective_separate(tmp_path):
    bridge = BridgeSession(tmp_path / "transfer")
    try:
        response = _run(bridge, goal="costly", selector="SEARCH")
        episode = response["episode"]
        assert episode["publicGoal"]["goalId"] == "costly"
        assert episode["selector"] == "SEARCH"
        assert episode["taskContract"]["learnedPolicySupported"] is False
        assert episode["planning"]["objectiveSource"] == "public_goal_and_shared_observed_native_state_only"
        assert episode["metrics"]["goal_contacted_and_retained"] is False
        assert episode["metrics"]["total_reward"] == 0.0
        assert "workspaceSession" not in response["case"]
    finally:
        bridge.close()


@pytest.mark.parametrize("payload", [
    {"fixture": "generated-public-surface-contact-v1", "goalId": "near", "selector": "RL256_ASPIRATION_TRANSFER"},
    {"fixture": "generated-public-surface-contact-v1", "goalId": "unknown", "selector": "SEARCH"},
    {"fixture": "generated-sequential-v1", "goalId": "near", "selector": "scripted"},
    {"fixture": "generated-public-surface-contact-v1", "goalId": "near", "selector": "scripted", "path": "/tmp/untrusted"},
])
def test_bad_request_refuses_before_evaluator(tmp_path, monkeypatch, payload):
    import resectionlab.surface_contact_episode as contact
    monkeypatch.setattr(contact, "execute_surface_contact_episode",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("No generated execution")))
    bridge = BridgeSession(tmp_path / "transfer")
    try:
        with pytest.raises(BridgeError) as denied:
            bridge.execute("executePublicSurfaceContactEpisode", payload,
                _Request("invalid"), lambda *_: None)
        assert denied.value.code == "INVALID_ARGUMENT"
    finally:
        bridge.close()
