"""Transient matched SEARCH view of an already executed RL256 transfer pair.

The original paired record is held by the backend after its bounded live run.
This module performs native replay only: no actor forward, search expansion,
checkpoint load, patient read or private-reference scoring.
"""
from __future__ import annotations

from .core import freeze_json, semantic_digest, thaw_json
from .development_episode import (TOOLS, _execute_sealed_development_plan,
                                  make_development_task)
from .legacy_aspiration_projection import PROJECTION_HASH
from .legacy_transfer_episode import (GEOMETRY_KEYS, SELECTOR, _check_pair,
    execute_transfer_episode_from_pair)

COMPARATOR_SELECTOR = "RL256_ASPIRATION_MATCHED_SEARCH"


def execute_transfer_search_companion_from_pair(pair, actor_episode, *, cancelled=None):
    """Recheck a full sealed pair, then replay its SEARCH IDs on fresh native state.

    The caller owns the pair. An imported actor episode alone is insufficient.
    Actor authorship is verified only by the original live backend operation.
    """
    if (type(actor_episode) is not dict or actor_episode.get("selector") != SELECTOR or
            actor_episode.get("patientAdmission") is not False or
            actor_episode.get("clinicalValidation") is not False):
        raise ValueError("Only the live generated actor episode has a companion")
    task = make_development_task(cancelled=cancelled)
    _actor, search, _identity = _check_pair(task, pair)
    actor_case, actor_replay = execute_transfer_episode_from_pair(pair, cancelled=cancelled)
    if actor_replay != actor_episode or actor_case.semantic_hash != actor_episode.get("caseHash"):
        raise ValueError("Stored actor episode differs from the full paired record")

    initial = task._engine.state_hash
    rejected = task._engine.preview_stroke(TOOLS[1].tool_id, (6., 6., 2.),
        entry_mm=(6., 6., 1.5), interaction_mode="probe")
    if rejected.feasible or task._engine.state_hash != initial:
        raise RuntimeError("Preopening probe refusal changed before matched replay")

    nominal = task.planning_clone()
    for action in search["action_ids"]:
        nominal.step(action)
    history = thaw_json(freeze_json(nominal.metrics()["history"]))
    if not nominal.terminated or len(history) != len(search["decisions"]):
        raise ValueError("Matched SEARCH did not terminate in the same native task")
    for row, decision in zip(history, search["decisions"]):
        physical = decision["physical_transition"]
        if (any(thaw_json(freeze_json(row.get(key))) != physical.get(key)
                for key in GEOMETRY_KEYS) or
                row["result_state_hash"] != decision["post_model_state_hash"] or
                row["interaction_mode"] not in ("aspirate", "stop")):
            raise ValueError("Matched SEARCH physical history differs from native replay")
    plan = thaw_json(freeze_json({"decision_model_hash": task.decision_model_hash,
        "source_hash": task.case.source_hash, "actions": search["action_ids"],
        "history": history, "observation_contract": "sequential-spatial-observation-v1",
        "max_steps": task.max_steps}))
    seal = semantic_digest(plan)
    accounting = {"selector": "matched_projected_SEARCH_from_existing_pair",
        # Only the final companion execution is counted here. The pair and
        # actor-identity checks above perform additional validation replays.
        "companionExecutionTransitions": len(search["action_ids"]),
        "originalSearchModelTransitionCalls": search["search_accounting"]["model_transition_calls"],
        "optimizer_updates": 0, "actor_forward_calls": 0,
        "optimizerUpdates": 0, "actorForwardCalls": 0,
        "searchAccounting": search["search_accounting"],
        "projectionTrace": pair["search"]["projection_trace"],
        "pairSeal": pair["seal"], "projectionHash": PROJECTION_HASH,
        "matchedActorEpisodeId": actor_episode["episodeId"],
        "matchedActorStrategySeal": actor_episode["planning"]["strategySeal"],
        "nativeSearchStrategySeal": seal,
        "attributionScope": "matched_observed_search_record_from_live_pair_no_new_search",
        "trainingDomainMatchesTarget": False,
        "privateReferenceScored": False}
    case, episode = _execute_sealed_development_plan(task, COMPARATOR_SELECTOR,
        freeze_json(plan), seal, freeze_json(accounting), rejected)
    if (case.semantic_hash != actor_case.semantic_hash or
            episode["sourceHash"] != actor_episode["sourceHash"] or
            episode["decisionModelHash"] != actor_episode["decisionModelHash"] or
            episode["planning"]["learnedPolicyExecuted"] is not False):
        raise RuntimeError("Matched SEARCH companion changed source or policy attribution")
    return case, episode
