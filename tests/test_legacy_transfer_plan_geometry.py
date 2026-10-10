"""Generated parser controls. Fabricated identity is NEVER an authorship proof."""
import copy
import tempfile
from pathlib import Path

import torch
from resectionlab.core import semantic_digest
from resectionlab.development_episode import make_development_task
from resectionlab.legacy_aspiration_projection import (
    AspirationOnlyNativeTask, BoundAspirationTransferPolicy,
    execute_matched_aspiration_transfer_pair)
from resectionlab.legacy_transfer_episode import (
    CHECKPOINT_SHA, ORIGINAL_ARCHITECTURE_SHA, ORIGINAL_PARAMETER_SHA,
    ORIGINAL_VALIDATION_SHA, execute_transfer_episode_from_pair,
    validate_transfer_planning, transfer_authorship)
from resectionlab.shared_episode import identity_for_untrained_spatial_policy
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
from resectionlab.shared_vascular_evaluation import _preflight
import resectionlab.shared_vascular_evaluation as vascular
from resectionlab.workspace_bundle import canonical_episode, validate_episode


def fake_sealed_pair():
    torch.set_num_threads(1)
    task = AspirationOnlyNativeTask(make_development_task())
    actor = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2), hidden_features=4,
                                             ray_samples=2)).eval()
    identity = identity_for_untrained_spatial_policy(actor, policy_id="test-untrained")
    pair = execute_matched_aspiration_transfer_pair(task,
        BoundAspirationTransferPolicy(actor, identity, task),
        max_calls=24, beam_width=4, seconds=10)
    # Test only the geometry/JSON admission path with a fabricated metadata
    # record. The real live-authorship boundary is the separate owned worker.
    receipt = pair["projection_receipt"]
    receipt.update(base_policy_id="native-opening-RL256",
        base_architecture_hash=ORIGINAL_ARCHITECTURE_SHA,
        base_parameter_hash=ORIGINAL_PARAMETER_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_validation_receipt_sha256=ORIGINAL_VALIDATION_SHA)
    pair["projection_receipt_hash"] = semantic_digest(receipt)
    pair["original_checkpoint_validation_receipt_sha256"] = ORIGINAL_VALIDATION_SHA
    policy_identity = pair["actor"]["strategy"]["policy_identity"]
    policy_identity.update(policy_id="native-opening-RL256-aspirate-transfer",
        training_status="trained_checkpoint", checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_validation_receipt_sha256=pair["projection_receipt_hash"])
    pair["actor"]["strategy"]["method"] = "LEARNED_POLICY"
    pair["seal"] = semantic_digest({key: value for key, value in pair.items() if key != "seal"})
    return pair


def test_pair_replay_geometry_without_authorship_claim():
    pair = fake_sealed_pair()
    case, episode = execute_transfer_episode_from_pair(pair)
    assert episode["caseHash"] == case.semantic_hash
    assert validate_transfer_planning(make_development_task(), episode)
    wrong_count = copy.deepcopy(episode)
    wrong_count["planning"]["model_transition_calls"] = -17
    try:
        validate_transfer_planning(make_development_task(), wrong_count)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative imported actor transition count was accepted")
    fake_search = copy.deepcopy(episode)
    fake_search["planning"]["matchedSearch"]["searchAccounting"] = {
        "actor_forward_calls": 123456, "made_up": "claims"}
    try:
        validate_transfer_planning(make_development_task(), fake_search)
    except ValueError:
        pass
    else:
        raise AssertionError("Arbitrary imported search accounting was accepted")
    assert transfer_authorship(episode, live_backend_run=False)["status"] == "unverified_imported"
    assert episode["planning"]["actor_forward_calls"] == len(episode["history"])
    assert all(row["interaction_mode"] in ("aspirate", "stop") for row in episode["history"])
    assert _preflight(episode, None)[0].terminated
    envelope = {"episode": episode, "episodeCanonicalJson": canonical_episode(episode)}
    assert validate_episode(case, envelope) == envelope
    tampered = copy.deepcopy(episode)
    tampered["planning"]["projectionTrace"][0]["full_observation_hash"] = "sha256:" + "0" * 64
    try:
        validate_transfer_planning(make_development_task(), tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("Tampered full-inventory trace was accepted")
    tampered["episodeId"] = semantic_digest({key: value for key, value in tampered.items()
                                              if key != "episodeId"})
    calls = []
    old_loader = vascular._load_generated_reference
    try:
        vascular._load_generated_reference = lambda binding: calls.append(binding)
        with tempfile.TemporaryDirectory() as directory:
            try:
                vascular.evaluate_development_episode_vascular(
                    episode=tampered, output_directory=Path(directory) / "unopened")
            except ValueError:
                pass
            else:
                raise AssertionError("Tampered actor observation reached private evaluation")
            assert not (Path(directory) / "unopened").exists()
    finally:
        vascular._load_generated_reference = old_loader
    assert calls == []


if __name__ == "__main__":
    test_pair_replay_geometry_without_authorship_claim()
