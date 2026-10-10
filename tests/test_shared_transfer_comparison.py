"""Generated-only companion controls; fabricated identity is not RL evidence."""
from __future__ import annotations

import copy
import json
import pytest

import torch
from resectionlab.core import freeze_json, semantic_digest
from resectionlab.development_episode import make_development_task
from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
from resectionlab.legacy_aspiration_projection import (
    AspirationOnlyNativeTask, BoundAspirationTransferPolicy,
    execute_matched_aspiration_transfer_pair)
from resectionlab.legacy_transfer_episode import (
    CHECKPOINT_SHA, ORIGINAL_ARCHITECTURE_SHA, ORIGINAL_PARAMETER_SHA,
    ORIGINAL_VALIDATION_SHA, execute_transfer_episode_from_pair)
from resectionlab.shared_episode import identity_for_untrained_spatial_policy
from resectionlab.shared_transfer_comparison import (
    COMPARATOR_SELECTOR, execute_transfer_search_companion_from_pair)
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
import resectionlab.shared_vascular_evaluation as vascular


def _fake_pair():
    torch.set_num_threads(1)
    task = AspirationOnlyNativeTask(make_development_task())
    actor = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2),
        hidden_features=4, ray_samples=2)).eval()
    identity = identity_for_untrained_spatial_policy(actor, policy_id="test-untrained")
    pair = execute_matched_aspiration_transfer_pair(task,
        BoundAspirationTransferPolicy(actor, identity, task),
        max_calls=24, beam_width=4, seconds=10)
    # Fabricate metadata only to exercise the already reviewed geometry/DTO
    # boundary. This does not represent a trained actor or authentic authorship.
    receipt = pair["projection_receipt"]
    receipt.update(base_policy_id="native-opening-RL256",
        base_architecture_hash=ORIGINAL_ARCHITECTURE_SHA,
        base_parameter_hash=ORIGINAL_PARAMETER_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_validation_receipt_sha256=ORIGINAL_VALIDATION_SHA)
    pair["projection_receipt_hash"] = semantic_digest(receipt)
    pair["original_checkpoint_validation_receipt_sha256"] = ORIGINAL_VALIDATION_SHA
    identity = pair["actor"]["strategy"]["policy_identity"]
    identity.update(policy_id="native-opening-RL256-aspirate-transfer",
        training_status="trained_checkpoint", checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_validation_receipt_sha256=pair["projection_receipt_hash"])
    pair["actor"]["strategy"]["method"] = "LEARNED_POLICY"
    pair["seal"] = semantic_digest({key: value for key, value in pair.items() if key != "seal"})
    return pair


def _fixture(monkeypatch):
    pair = _fake_pair()
    case, actor_episode = execute_transfer_episode_from_pair(pair)
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("No checkpoint load in companion inspection")))
    monkeypatch.setattr(vascular, "_load_generated_reference", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("No private reference read in companion inspection")))
    return pair, case, actor_episode


def test_companion_replays_existing_search_without_actor_or_search(monkeypatch):
    pair, case, actor_episode = _fixture(monkeypatch)
    import resectionlab.shared_episode as shared_episode
    monkeypatch.setattr(shared_episode, "observed_beam_search", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("No new SEARCH expansion in companion inspection")))
    monkeypatch.setattr(SpatialPolicy, "forward", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("No actor forward in companion inspection")))
    replay_case, companion = execute_transfer_search_companion_from_pair(pair, actor_episode)
    assert replay_case.semantic_hash == case.semantic_hash
    assert companion["selector"] == COMPARATOR_SELECTOR
    assert companion["planning"]["learnedPolicyExecuted"] is False
    assert companion["planning"]["actorForwardCalls"] == 0
    assert companion["planning"]["optimizerUpdates"] == 0
    assert companion["planning"]["companionExecutionTransitions"] == len(companion["history"])
    assert "nativeReplayTransitionCalls" not in companion["planning"]
    assert companion["planning"]["originalSearchModelTransitionCalls"] == pair["search"]["strategy"]["search_accounting"]["model_transition_calls"]
    assert companion["planning"]["strategy"]["actions"] == pair["search"]["strategy"]["action_ids"]
    assert companion["planning"]["matchedActorEpisodeId"] == actor_episode["episodeId"]
    assert companion["planning"]["pairSeal"] == pair["seal"]
    assert companion["clinicalValidation"] is False and companion["patientAdmission"] is False
    assert companion["initialStateId"] == actor_episode["initialStateId"]


def test_companion_refuses_changed_actor_or_pair(monkeypatch):
    pair, _case, actor_episode = _fixture(monkeypatch)
    changed_actor = copy.deepcopy(actor_episode)
    changed_actor["history"][0]["action_id"] = "tampered-action"
    with pytest.raises(ValueError, match="differs"):
        execute_transfer_search_companion_from_pair(pair, changed_actor)
    changed_pair = copy.deepcopy(pair)
    changed_pair["search"]["strategy"]["action_ids"][0] = "STOP"
    with pytest.raises(ValueError, match="seal"):
        execute_transfer_search_companion_from_pair(changed_pair, actor_episode)


def test_bridge_live_only_and_identity_bound(tmp_path, monkeypatch):
    pair, case, actor_episode = _fixture(monkeypatch)
    bridge = BridgeSession(tmp_path / "transfers")
    try:
        bridge._install_case(case, {}, _Request("install"))
        entry = bridge.cases[case.semantic_hash]
        canonical = json.dumps({key: value for key, value in actor_episode.items()
            if key != "episodeId"}, sort_keys=True, separators=(",", ":"), allow_nan=False)
        entry.episode = {"episode": actor_episode, "episodeCanonicalJson": canonical}
        entry.episode_selection = {"episodeId": actor_episode["episodeId"],
            "frameIndex": 0, "visible": True}
        entry.comparison_pair = freeze_json(pair)
        entry.comparison_actor_episode_id = actor_episode["episodeId"]
        args = {"caseHash": case.semantic_hash, "episodeId": actor_episode["episodeId"]}
        response = bridge.execute("inspectDevelopmentEpisodeComparison", args,
            _Request("inspect"), lambda _v, _m: None)
        assert set(response) == {"caseHash", "actorEpisodeId", "actorStrategySeal",
            "pairSeal", "projectionHash", "initialProjectedObservationHash", "companion"}
        assert response["pairSeal"] == pair["seal"]
        assert response["companion"]["episode"]["selector"] == COMPARATOR_SELECTOR
        assert response["companion"]["episode"]["episodeId"] == semantic_digest(
            {key: value for key, value in response["companion"]["episode"].items()
             if key != "episodeId"})
        with pytest.raises(BridgeError):
            bridge.execute("inspectDevelopmentEpisodeComparison",
                {**args, "episodeId": "sha256:" + "0" * 64},
                _Request("wrong"), lambda _v, _m: None)
        entry.comparison_pair = None  # Reopened workspace has no authenticated pair.
        with pytest.raises(BridgeError) as unavailable:
            bridge.execute("inspectDevelopmentEpisodeComparison", args,
                _Request("reopened"), lambda _v, _m: None)
        assert unavailable.value.code == "COMPARISON_UNAVAILABLE"
    finally:
        bridge.close()
