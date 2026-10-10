"""Fixed generated transfer plan admission; no checkpoint or private-reference loading.

The paired record must originate from the bounded backend worker. Replaying a
saved record proves modeled geometry, not that a trained actor authored it.
"""
from __future__ import annotations

from dataclasses import asdict
import math

from .core import freeze_json, semantic_digest, thaw_json
from .development_episode import (TOOLS, _execute_sealed_development_plan,
                                  make_development_task)
from .legacy_aspiration_projection import (AspirationOnlyNativeTask,
    FIXED_DECISION_MODEL_HASH, FIXED_SOURCE_HASH, PROJECTION_HASH,
    _projection_trace)
from .shared_episode import PolicyIdentity, verify_strategy_replay

SELECTOR = "RL256_ASPIRATION_TRANSFER"
CHECKPOINT_SHA = "sha256:1d391665f66cdd0c1fa1db150261b853820a9d30d62fc20c99cf021cf0229fbd"
ORIGINAL_ARCHITECTURE_SHA = "sha256:a3b6740ab188f01016610c566e2009c57ba422c8f9be24adca014f695ae9c917"
ORIGINAL_PARAMETER_SHA = "sha256:f52e14097ea6a35436eaf30f0ece5658e24987e289ca30ff6a529816545b5721"
ORIGINAL_VALIDATION_SHA = "sha256:4e6723d76f61612b7fdf4761e72b15ad7b90b6bd77e77fe7486dab1d73de4247"
GEOMETRY_KEYS = ("action_id", "interaction_mode", "source_state_hash",
                 "removed_indices_native", "contact_indices_native", "microsteps")
SEARCH_ACCOUNTING_KEYS = frozenset({
    "model_transition_calls", "planning_seconds", "completed_layers", "call_cap_reached",
    "time_cap_reached", "beam_width", "max_calls", "estimated_incremental_return",
    "stop_baseline_return", "implicit_stop_evaluations", "implicit_stop_prefix_count",
    "evaluated_transition_prefixes", "objective_source", "root_legal_nonstop_actions",
    "expanded_nodes", "transition_mode", "eager_transition_calls",
    "lazy_planning_transition_calls", "observation_requests", "actor_forward_calls",
    "guidance", "negative_prefixes_evaluated", "negative_prefixes_retained",
    "negative_prefixes_pruned_by_beam", "beam_pruned_prefixes", "peak_next_layer_states",
    "layers", "policy_state_checks", "policy_state_check_seconds",
    "initial_policy_state_hash", "final_policy_state_hash"})
SEARCH_LAYER_KEYS = frozenset({"depth", "completed", "expanded_nodes",
    "model_transition_calls", "negative_prefixes_evaluated",
    "negative_prefixes_retained", "negative_prefixes_pruned_by_beam",
    "negative_prefixes_pending_at_cap"})


def _check_search_accounting(value):
    """Bound saved counters; imported compute authorship remains unverified."""
    if type(value) is not dict or set(value) != SEARCH_ACCOUNTING_KEYS:
        raise ValueError("Saved search accounting schema differs")
    calls = value["model_transition_calls"]
    layers = value["layers"]
    if (type(calls) is not int or not 0 <= calls <= 24 or
            type(layers) is not list or not 1 <= len(layers) <= 6 or
            value["beam_width"] != 4 or type(value["beam_width"]) is not int or
            value["max_calls"] != 24 or type(value["max_calls"]) is not int or
            value["transition_mode"] != "eager" or value["guidance"] != "none" or
            value["objective_source"] != "observed_scan_estimator_only" or
            value["actor_forward_calls"] != 0 or type(value["actor_forward_calls"]) is not int or
            value["lazy_planning_transition_calls"] != 0 or
            value["eager_transition_calls"] != calls or
            value["evaluated_transition_prefixes"] != calls or
            value["implicit_stop_prefix_count"] != calls + 1 or
            value["policy_state_checks"] != 0 or
            value["policy_state_check_seconds"] != 0. or
            value["initial_policy_state_hash"] is not None or
            value["final_policy_state_hash"] is not None or
            value["stop_baseline_return"] != 0. or
            value["implicit_stop_evaluations"] != "all evaluated prefixes; known zero incremental reward" or
            type(value["planning_seconds"]) not in (int, float) or
            not math.isfinite(value["planning_seconds"]) or
            not 0 <= value["planning_seconds"] <= 10. or
            type(value["estimated_incremental_return"]) not in (int, float) or
            not math.isfinite(value["estimated_incremental_return"]) or
            type(value["call_cap_reached"]) is not bool or
            type(value["time_cap_reached"]) is not bool or
            value["time_cap_reached"] is not False):
        raise ValueError("Saved search counters exceed the fixed method")
    for key in ("completed_layers", "root_legal_nonstop_actions", "expanded_nodes",
                "observation_requests", "negative_prefixes_evaluated", "negative_prefixes_retained",
                "negative_prefixes_pruned_by_beam", "beam_pruned_prefixes", "peak_next_layer_states"):
        if type(value[key]) is not int or not 0 <= value[key] <= 24:
            raise ValueError("Saved search counter is malformed: " + key)
    if value["completed_layers"] > len(layers):
        raise ValueError("Saved search layer count differs")
    for depth, layer in enumerate(layers, 1):
        if (type(layer) is not dict or set(layer) != SEARCH_LAYER_KEYS or
                type(layer["depth"]) is not int or layer["depth"] != depth or
                type(layer["completed"]) is not bool):
            raise ValueError("Saved search layer schema differs")
        for key in SEARCH_LAYER_KEYS - {"depth", "completed"}:
            if type(layer[key]) is not int or not 0 <= layer[key] <= 24:
                raise ValueError("Saved search layer count is malformed")
    if (sum(layer["model_transition_calls"] for layer in layers) != calls or
            value["expanded_nodes"] != sum(layer["expanded_nodes"] for layer in layers) or
            value["observation_requests"] != value["expanded_nodes"] or
            value["negative_prefixes_evaluated"] != sum(
                layer["negative_prefixes_evaluated"] for layer in layers) or
            sum(layer["completed"] for layer in layers) != value["completed_layers"] or
            sum(layer["negative_prefixes_pruned_by_beam"] for layer in layers)
                != value["negative_prefixes_pruned_by_beam"]):
        raise ValueError("Saved search layer accounting differs")


def _check_pair(task, pair):
    if type(pair) is not dict or pair.get("schema") != "generated-legacy-aspiration-transfer-pair-v1":
        raise ValueError("Unsupported generated transfer pair")
    body = {key: value for key, value in pair.items() if key != "seal"}
    if semantic_digest(body) != pair.get("seal"):
        raise ValueError("Transfer pair seal differs from its body")
    if (pair.get("source_hash") != FIXED_SOURCE_HASH or
            pair.get("decision_model_hash") != FIXED_DECISION_MODEL_HASH or
            pair.get("projection_hash") != PROJECTION_HASH or
            pair.get("private_reference_scored") is not False or
            pair.get("clinical_validation") is not False or
            pair.get("checkpoint_training_domain_matches_target") is not False or
            pair.get("search_reward_source") != "permitted_generated_nominal_target_only"):
        raise ValueError("Transfer pair changed source, projection or evaluation boundary")
    receipt = pair.get("projection_receipt")
    if type(receipt) is not dict or semantic_digest(receipt) != pair.get("projection_receipt_hash"):
        raise ValueError("Projection receipt is missing or changed")
    if (receipt.get("base_policy_id") != "native-opening-RL256" or
            receipt.get("base_architecture_hash") != ORIGINAL_ARCHITECTURE_SHA or
            receipt.get("base_parameter_hash") != ORIGINAL_PARAMETER_SHA or
            receipt.get("checkpoint_sha256") != CHECKPOINT_SHA or
            receipt.get("checkpoint_validation_receipt_sha256") != ORIGINAL_VALIDATION_SHA or
            receipt.get("target_source_hash") != FIXED_SOURCE_HASH or
            receipt.get("target_decision_model_hash") != FIXED_DECISION_MODEL_HASH or
            receipt.get("projection_hash") != PROJECTION_HASH or
            receipt.get("aspirator_tool_ids") != [TOOLS[0].tool_id] or
            pair.get("original_checkpoint_validation_receipt_sha256") != ORIGINAL_VALIDATION_SHA or
            receipt.get("scope") != "generated_transfer_not_checkpoint_training_domain"):
        raise ValueError("Transfer has different checkpoint ancestry or action scope")
    view = AspirationOnlyNativeTask(task.planning_clone())
    arms = {}
    for name in ("actor", "search"):
        arm = pair.get(name)
        if type(arm) is not dict or set(arm) != {"strategy", "projection_trace"}:
            raise ValueError("Transfer pair arm is incomplete")
        strategy = arm["strategy"]
        if not verify_strategy_replay(view, strategy):
            raise ValueError("Paired arm failed native replay")
        if _projection_trace(view, strategy) != arm["projection_trace"]:
            raise ValueError("Paired arm differs from the full mixed action inventory")
        arms[name] = strategy
    actor = arms["actor"]
    search = arms["search"]
    identity = PolicyIdentity(**actor["policy_identity"])
    if (actor["method"] != "LEARNED_POLICY" or
            identity.training_status != "trained_checkpoint" or
            identity.policy_id != "native-opening-RL256-aspirate-transfer" or
            identity.observation_contract != "permitted-spatial-observation-v1" or
            identity.checkpoint_sha256 != CHECKPOINT_SHA or
            identity.checkpoint_validation_receipt_sha256 != pair["projection_receipt_hash"] or
            search["method"] != "SEARCH" or
            search.get("search_accounting") is None or
            actor["source_hash"] != search["source_hash"] or
            actor["environment_contract_hash"] != search["environment_contract_hash"] or
            actor["decisions"][0]["observation_before"] != search["decisions"][0]["observation_before"]):
        raise ValueError("Actor/search attribution or matched start differs")
    _check_search_accounting(search["search_accounting"])
    return actor, search, identity


def plan_from_transfer_pair(task, pair):
    """Translate a sealed paired actor strategy to the existing native plan."""
    actor, search, identity = _check_pair(task, pair)
    nominal = task.planning_clone()
    for action in actor["action_ids"]:
        nominal.step(action)
    history = thaw_json(freeze_json(nominal.metrics()["history"]))
    if not nominal.terminated or len(history) != len(actor["decisions"]):
        raise ValueError("Transfer actor did not complete within the fixed horizon")
    for row, decision in zip(history, actor["decisions"]):
        physical = decision["physical_transition"]
        if (any(thaw_json(freeze_json(row.get(key))) != physical.get(key) for key in GEOMETRY_KEYS)
                or row["result_state_hash"] != decision["post_model_state_hash"]):
            raise ValueError("Transfer actor physical history differs from native replay")
        if row["interaction_mode"] not in ("aspirate", "stop"):
            raise ValueError("Legacy actor cannot execute a probe")
    plan = thaw_json(freeze_json({"decision_model_hash": task.decision_model_hash,
        "source_hash": task.case.source_hash, "actions": actor["action_ids"], "history": history,
        "observation_contract": "sequential-spatial-observation-v1", "max_steps": task.max_steps}))
    accounting = {"selector": "fixed_trained_RL256_aspiration_transfer",
        "model_transition_calls": len(actor["decisions"]),
        "actor_forward_calls": len(actor["decisions"]), "optimizer_updates": 0,
        "actorForwardCalls": len(actor["decisions"]), "optimizerUpdates": 0,
        "policyIdentity": asdict(identity), "projectionHash": PROJECTION_HASH,
        "projectionReceipt": pair["projection_receipt"],
        "projectionReceiptHash": pair["projection_receipt_hash"],
        "originalCheckpointValidationReceiptSha256": ORIGINAL_VALIDATION_SHA,
        "projectionTrace": pair["actor"]["projection_trace"],
        "attributionScope": "live_backend_checkpoint_run_only",
        "actorObservationContract": "permitted-spatial-observation-v1",
        "matchedSearch": {"method": "SEARCH", "actionIds": search["action_ids"],
            "searchAccounting": search["search_accounting"],
            "sourceHash": search["source_hash"],
            "environmentContractHash": search["environment_contract_hash"]},
        "trainingDomainMatchesTarget": False,
        "pairSeal": pair["seal"]}
    return freeze_json(plan), semantic_digest(plan), freeze_json(accounting)


def execute_transfer_episode_from_pair(pair, *, cancelled=None):
    """Execute a backend-owned completed pair; this function does not load weights."""
    task = make_development_task(cancelled=cancelled)
    initial = task._engine.state_hash
    rejected = task._engine.preview_stroke(TOOLS[1].tool_id, (6., 6., 2.),
        entry_mm=(6., 6., 1.5), interaction_mode="probe")
    if rejected.feasible or task._engine.state_hash != initial:
        raise RuntimeError("Preopening probe refusal changed")
    plan, seal, accounting = plan_from_transfer_pair(task, pair)
    return _execute_sealed_development_plan(task, SELECTOR, plan, seal, accounting, rejected)


def transfer_authorship(episode, *, live_backend_run: bool):
    """Authorship is a process trust claim, never inferred from imported JSON."""
    if episode.get("selector") != SELECTOR:
        raise ValueError("Not a transfer episode")
    planning = episode["planning"]
    return {"status": "verified_live_backend_run" if live_backend_run else "unverified_imported",
            "checkpointSha256": planning["policyIdentity"]["checkpoint_sha256"],
            "projectionHash": planning["projectionHash"]}


def validate_transfer_planning(task, episode):
    """Independently check saved action availability, not imported actor authorship.

    The caller must also replay native geometry, frames and source binding.
    This check never loads a checkpoint and cannot certify who selected IDs.
    """
    if type(episode) is not dict or episode.get("selector") != SELECTOR:
        raise ValueError("Expected a generated legacy-transfer episode")
    planning = episode.get("planning")
    if type(planning) is not dict or planning.get("learnedPolicyExecuted") is not True:
        raise ValueError("Transfer planning metadata is incomplete")
    identity = PolicyIdentity(**planning["policyIdentity"])
    receipt = planning.get("projectionReceipt")
    if (identity.policy_id != "native-opening-RL256-aspirate-transfer" or
            identity.training_status != "trained_checkpoint" or
            identity.observation_contract != "permitted-spatial-observation-v1" or
            identity.checkpoint_sha256 != CHECKPOINT_SHA or
            type(receipt) is not dict or
            receipt.get("base_policy_id") != "native-opening-RL256" or
            receipt.get("base_architecture_hash") != ORIGINAL_ARCHITECTURE_SHA or
            receipt.get("base_parameter_hash") != ORIGINAL_PARAMETER_SHA or
            receipt.get("checkpoint_sha256") != CHECKPOINT_SHA or
            receipt.get("checkpoint_validation_receipt_sha256") != ORIGINAL_VALIDATION_SHA or
            receipt.get("target_source_hash") != FIXED_SOURCE_HASH or
            receipt.get("target_decision_model_hash") != FIXED_DECISION_MODEL_HASH or
            receipt.get("projection_hash") != PROJECTION_HASH or
            receipt.get("aspirator_tool_ids") != [TOOLS[0].tool_id] or
            receipt.get("scope") != "generated_transfer_not_checkpoint_training_domain" or
            semantic_digest(receipt) != planning.get("projectionReceiptHash") or
            identity.checkpoint_validation_receipt_sha256 != planning.get("projectionReceiptHash") or
            planning.get("originalCheckpointValidationReceiptSha256") != ORIGINAL_VALIDATION_SHA or
            planning.get("projectionHash") != PROJECTION_HASH or
            planning.get("trainingDomainMatchesTarget") is not False or
            planning.get("attributionScope") != "live_backend_checkpoint_run_only" or
            planning.get("actorObservationContract") != "permitted-spatial-observation-v1"):
        raise ValueError("Saved transfer identity/projection ancestry changed")
    history = episode.get("history")
    strategy = planning.get("strategy")
    trace = planning.get("projectionTrace")
    if (type(history) is not list or type(strategy) is not dict or type(trace) is not list or
            len(history) < 1 or len(history) > 6 or len(trace) != len(history) or
            strategy.get("actions") != [row.get("action_id") for row in history] or
            type(planning.get("actorForwardCalls")) is not int or
            planning["actorForwardCalls"] != len(history) or
            type(planning.get("optimizerUpdates")) is not int or
            planning["optimizerUpdates"] != 0 or
            type(planning.get("actor_forward_calls")) is not int or
            planning["actor_forward_calls"] != planning["actorForwardCalls"] or
            type(planning.get("optimizer_updates")) is not int or
            planning["optimizer_updates"] != planning["optimizerUpdates"] or
            task.case.source_hash != FIXED_SOURCE_HASH or
            task.decision_model_hash != FIXED_DECISION_MODEL_HASH):
        raise ValueError("Saved transfer actions or actor accounting changed")
    view = AspirationOnlyNativeTask(task.planning_clone())
    for row, evidence in zip(history, trace):
        projected = view.projection_evidence()
        if (type(evidence) is not dict or evidence != {
                "full_observation_hash": projected.full_observation_hash,
                "projected_observation_hash": projected.projected_observation_hash,
                "retained_action_ids": list(projected.retained_action_ids),
                "omitted_probe_action_ids": list(projected.omitted_probe_action_ids),
                "chosen_action_id": row["action_id"],
                "projection_hash": PROJECTION_HASH} or
                row.get("interaction_mode") not in ("aspirate", "stop")):
            raise ValueError("Saved transfer trace hides probe actions or changed observations")
        view.step(row["action_id"])
    if not view.terminated:
        raise ValueError("Saved transfer does not terminate at the declared horizon")
    search = planning.get("matchedSearch")
    if (type(search) is not dict or search.get("method") != "SEARCH" or
            search.get("sourceHash") != FIXED_SOURCE_HASH or
            search.get("environmentContractHash") != FIXED_DECISION_MODEL_HASH or
            type(search.get("actionIds")) is not list or not search["actionIds"] or
            type(search.get("searchAccounting")) is not dict):
        raise ValueError("Saved matched-search provenance is incomplete")
    if (type(planning.get("model_transition_calls")) is not int or
            planning["model_transition_calls"] != planning["actorForwardCalls"]):
        raise ValueError("Saved actor transition accounting differs")
    _check_search_accounting(search["searchAccounting"])
    # Search is independently executable without loading the actor or private
    # labels; the stored budget/result is not an authorship proof.
    search_view = AspirationOnlyNativeTask(task.planning_clone())
    for action in search["actionIds"]:
        search_view.step(action)
    if not search_view.terminated:
        raise ValueError("Saved matched search actions do not terminate")
    return True
