# Derived from canonical planner SHA256 fe43285ae70d3ff9944e2f4b2e532f7205d0148e5fd30aae99216a8d1243ff37; only fixed index8 treatment/prefix joins are added.
"""Bounded actual-positive nominal rollout with one direct-Q batch per state.

The guard is the existing full-inventory immediate nominal objective. Learned Q
only ranks already productive actions; it does not demonstrate learned costly
preparation, optimal values or physical safety. Independent native replay remains
the caller's acceptance gate. No fitting, source loading or reference labels live
in this module.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from resectionlab.core import semantic_digest
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.observed_search import ObservedSearchLimit
from resectionlab.patient_planning_admission import PatientPlanningContext
from resectionlab.spatial_continuation_value import FittedContinuationValue
from resectionlab.spatial_continuation_tail import FittedContinuationTail


VERSION = "actual-positive-guarded-direct-q-nominal-rollout-v1"
TAIL_VERSION = "actual-positive-exact-reward-plus-recorded-tail-rollout-v1"
_SCORE_FIELDS = ("action_id", "reward", "target_removed_mm3", "normal_removed_mm3",
                 "insertion_distance_mm", "complete_tool_path_length_mm")


def _choose_positive(scores, predicted_q=None):
    """STOP0; stable original-inventory ties; Q already includes action reward."""
    positive = [i for i, row in enumerate(scores) if i and row["reward"] > 0.]
    if not positive:
        return 0, "no_actual_positive_action"
    if predicted_q is None or len(predicted_q) != len(scores):
        raise ValueError("One aligned Q batch is required for productive candidates")
    ranked = [i for i in positive if predicted_q[i] > 0.]
    if ranked:
        return max(ranked, key=lambda i: predicted_q[i]), "positive_q_among_actual_positive"
    return max(positive, key=lambda i: scores[i]["reward"]), "actual_greedy_fallback_no_positive_q"


def _combine_tail(scores, predicted_tail):
    """Current certified nominal reward exactly once, nonnegative tail, STOP0."""
    if len(predicted_tail) != len(scores) or predicted_tail[0] != 0.:
        raise ValueError('One aligned raw-tail batch with fixed STOP0 required')
    combined = tuple(0. if i == 0 else row['reward'] + max(0., predicted_tail[i])
                     for i, row in enumerate(scores))
    if not all(math.isfinite(value) for value in combined):
        raise FloatingPointError('Nonfinite exact reward plus continuation tail')
    return combined


def endgame_tail_search(task,context,scorer,*,forced_action,expected_prefix,expected_common,
        seconds=300.,cancelled=None,max_calls=24):
    """One predeclared index8 intervention; no suffix actions/outcomes admitted."""
    validate_treatment(forced_action,expected_prefix,expected_common)
    return _guarded_nominal_search(task,context,scorer,seconds=seconds,cancelled=cancelled,
        max_calls=max_calls,tail_mode=True,forced_action=forced_action,
        expected_prefix=expected_prefix,expected_common=expected_common)


def validate_treatment(forced_action,prefix,common):
    if (not isinstance(forced_action,str) or len(prefix)!=8
            or set(common)!={'step','source_state_hash','observation_hash','scores','current_tool'}
            or common['step']!=8 or forced_action not in [r['action_id'] for r in common['scores']]
            or forced_action=='STOP'):
        raise ValueError('Exact8-prefix/index8 nonSTOP common-state treatment required')
    fields={'selected_action_id','observation_hash','source_state_hash','result_state_hash',
            'actual_reward','terminated','current_tool_before'}
    if any(set(row)!=fields or row['terminated'] is not False for row in prefix):
        raise ValueError('Only exact completed prefix fields accepted; no future suffix')


def aggregate_catalog(ledger,certified_count,*,state_hash,step,selected_action):
    """Output-only scalar snapshot; never issues an inventory/native query."""
    groups={};proposed=0;certified=0
    for row in ledger:
        emitted=row.get('proposal_reason')=='PROPOSED_UNCERTIFIED'
        feasible=row.get('feasible') is True
        proposed+=int(emitted);certified+=int(feasible)
        key=tuple(str(row.get(k,'UNKNOWN')) for k in ('tool_id','family','proposal_reason','reason'))
        group=groups.setdefault(key,{'slots':0,'proposed':0,'certified':0})
        group['slots']+=1;group['proposed']+=int(emitted);group['certified']+=int(feasible)
    if certified!=certified_count:raise ValueError('Catalog certified count differs from existing inventory')
    return {'captured':'after_final_choice_before_terminal_transition','state_hash':state_hash,
        'step':step,'selected_action':selected_action,'ledger_slots':len(ledger),
        'proposed_count':proposed,'certified_count':certified,'additional_queries':0,
        'groups':[dict(zip(('tool_id','proposal_kind','proposal_disposition','native_reason'),key),**value)
                  for key,value in sorted(groups.items())]}


def _guarded_nominal_search(task, context, scorer, *, seconds, cancelled, max_calls, tail_mode, forced_action, expected_prefix, expected_common):
    expected_type = FittedContinuationTail if tail_mode else FittedContinuationValue
    if (type(task) is not NativeSpatialTask or type(context) is not PatientPlanningContext
            or type(scorer) is not expected_type):
        raise TypeError("Exact admitted native task, context and matching fitted scorer type required")
    if (type(max_calls) is not int or not 1 <= max_calls <= 24
            or isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= 3600
            or (cancelled is not None and not callable(cancelled))):
        raise ValueError("Explicit bounded rollout budgets are required")
    if task.max_steps != 24 or task._steps or task._history or task.terminated:
        raise ValueError("Guarded Q rollout requires a fresh task with horizon at most 24")

    started = time.perf_counter()
    counts = {"inventory_requests": 0, "inventory_builds_started": 0,
              "inventory_builds_completed": 0, "complete_inventory_rows": 0,
              "immediate_scores_started": 0, "immediate_scores_completed": 0,
              "observation_requests": 0, "observations_completed": 0,
              "q_batches_started": 0, "q_batches_completed": 0,
              "model_transition_calls": 0, "committed_transitions": 0,
              "native_previews_started": 0, "native_previews_completed": 0,
              "native_previews_feasible": 0, "native_preview_seconds": 0.}
    flags = {"time_cap_reached": False, "call_cap_reached": False, "cancelled": False}
    sequence, decisions = [], []
    terminal_catalog=None
    model = None
    error = None
    complete = False
    original_preview = None
    prior_instance_preview = None
    had_instance_preview = False
    before_costs = dict(scorer.costs)
    fit_seal = semantic_digest(scorer.report)
    context_seal = context.fingerprint

    def check():
        if cancelled is not None and cancelled():
            flags["cancelled"] = True
            raise InterruptedError("Guarded Q rollout cancelled")
        try:
            task._check_cancelled()
        except InterruptedError:
            flags["cancelled"] = True
            raise
        if time.perf_counter()-started > seconds:
            flags["time_cap_reached"] = True
            raise ObservedSearchLimit("Guarded Q rollout time cap", accounting={}, best_sequence=tuple(sequence))

    def counted_preview(*args, **kwargs):
        stamp = time.perf_counter()
        counts["native_previews_started"] += 1
        try:
            result = original_preview(*args, **kwargs)
            counts["native_previews_completed"] += 1
            counts["native_previews_feasible"] += int(result.feasible)
            return result
        finally:
            counts["native_preview_seconds"] += time.perf_counter()-stamp

    try:
        check()
        context.require_task(task)
        scorer.require(); scorer.bind_context(context)
        model = task.planning_clone()
        context.require_task(model)
        if not model._planning:
            raise ValueError("Q rollout requires permitted-nominal planning state")
        original_preview = model._engine.preview_stroke
        had_instance_preview = "preview_stroke" in vars(model._engine)
        prior_instance_preview = vars(model._engine).get("preview_stroke")
        model._engine.preview_stroke = counted_preview
        while not model.terminated:
            check()
            if counts["model_transition_calls"] >= max_calls:
                flags["call_cap_reached"] = True
                raise ObservedSearchLimit("Guarded Q rollout transition cap", accounting={}, best_sequence=tuple(sequence))
            counts["inventory_requests"] += 1
            building = model._inventory is None
            counts["inventory_builds_started"] += int(building)
            inventory = model._prepare_inventory()
            counts["inventory_builds_completed"] += int(building)
            counts["complete_inventory_rows"] += len(model._ledger)
            scores = [dict(action_id="STOP", reward=0., target_removed_mm3=0., normal_removed_mm3=0.,
                           insertion_distance_mm=0., complete_tool_path_length_mm=0.)]
            decision = {"step": model._steps, "source_state_hash": model._engine.state_hash,
                        "legal_nonstop_actions": len(inventory), "scores": scores,
                        "all_current_legal_actions_scored": False, "transition_status": "not_started"}
            decisions.append(decision)
            for identifier, preview in inventory.items():
                check()
                counts["immediate_scores_started"] += 1
                engine = model._engine
                if (not preview.feasible or engine._preview_records.get(id(preview)) is not preview
                        or preview.source_state_hash != engine.state_hash
                        or preview.decision_model_hash != engine.config.fingerprint
                        or engine._preview_digests.get(id(preview)) != engine._result_digest(preview)):
                    raise ValueError("Q rollout encountered a stale, foreign or changed native certificate")
                scored = model._score_record({**preview.to_history_record(), "action_id": identifier}, model._current_tool)
                if scored.get("outcome_scope") != "permitted_nominal_model":
                    raise ValueError("Only actual permitted-nominal immediate rewards may guard Q")
                row = {key: scored[key] for key in _SCORE_FIELDS}
                if any(isinstance(row[key], bool) or not math.isfinite(row[key]) for key in _SCORE_FIELDS[1:]):
                    raise FloatingPointError("Nonfinite nominal candidate score")
                scores.append(row)
                counts["immediate_scores_completed"] += 1
            decision["all_current_legal_actions_scored"] = True
            check()
            counts["observation_requests"] += 1
            observation = model.observation()  # Reuses the already complete inventory.
            counts["observations_completed"] += 1
            context.require_observations((observation,))
            ids = tuple(row["action_id"] for row in scores)
            if (observation.action_ids != ids or len(set(ids)) != len(ids)
                    or observation.action_mask.shape != (len(ids),) or not observation.action_mask.all()):
                raise ValueError("Public observation differs from current certified action inventory/budget")
            observation_hash = observation.fingerprint
            decision["observation_hash"] = observation_hash
            positive = sum(row["reward"] > 0. for row in scores[1:])
            decision["actual_positive_candidates"] = positive
            predicted = None
            if positive:
                check()
                counts["q_batches_started"] += 1
                predicted = (scorer.score_tails(observation, context) if tail_mode
                             else scorer.score_actions(observation, context))
                counts["q_batches_completed"] += 1
                if (not isinstance(predicted, tuple) or len(predicted) != len(ids)
                        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in predicted) or predicted[0] != 0.):
                    raise ValueError("Scorer must align to all current legal actions with exact STOP0")
                observation.assert_intact()
                if observation.fingerprint != observation_hash:
                    raise ValueError("Q inference changed its public observation")
            combined = (_combine_tail(scores, predicted)
                        if tail_mode and predicted is not None else predicted)
            chosen, reason = _choose_positive(scores, combined)
            if tail_mode and chosen:
                reason = 'exact_immediate_reward_plus_nonnegative_recorded_tail'
            for i, row in enumerate(scores):
                row["predicted_q"] = None if combined is None else combined[i]
                row["actual_positive_guard"] = bool(i and row["reward"] > 0.)
                if tail_mode:
                    row.update(raw_predicted_tail=0. if i == 0 else None if predicted is None else predicted[i],
                        clipped_predicted_tail=0. if i == 0 else None if predicted is None else max(0., predicted[i]),
                        combined_nominal_q=0. if i == 0 else None if combined is None else combined[i])
                    if i == 0:
                        row['predicted_q'] = 0.
            if model._steps < 8:
                expected=expected_prefix[model._steps]
                if (ids[chosen]!=expected['selected_action_id']
                        or observation_hash!=expected['observation_hash']
                        or model._engine.state_hash!=expected['source_state_hash']
                        or model._current_tool!=expected['current_tool_before']):
                    raise ValueError('Frozen prefix action/observation/state/current tool changed')
            elif model._steps == 8:
                if ({'step':model._steps,'source_state_hash':model._engine.state_hash,
                        'observation_hash':observation_hash,'scores':scores,
                        'current_tool':model._current_tool} != expected_common):
                    raise ValueError('Common-state complete ordered inventory or score changed')
                decision['unforced_selected_action_id']=ids[chosen]
                chosen=ids.index(forced_action)
                if scores[chosen]['reward']<=0:raise ValueError('Forced action violates unchanged productive guard')
                reason='predeclared_single_index8_intervention'
                decision['intervention']=True
            decision.update(selected_action_id=ids[chosen], selection_reason=reason,
                            selected_q=0. if tail_mode and chosen == 0 else None if combined is None else combined[chosen])
            if ids[chosen]=='STOP' or model._steps+1==model.max_steps:
                terminal_catalog=aggregate_catalog(model._ledger,len(inventory),state_hash=model._engine.state_hash,
                    step=model._steps,selected_action=ids[chosen])
            check()
            counts["model_transition_calls"] += 1
            decision["transition_status"] = "attempted"
            prior_steps = model._steps
            try:
                outcome = model.advance_planning(ids[chosen])
            finally:
                # Cancellation after durable commit still belongs in the cost
                # denominator and partial trace; never erase that transition.
                if model._steps == prior_steps+1:
                    counts["committed_transitions"] += 1
                    sequence.append(ids[chosen])
                    committed = model._history[-1]
                    decision.update(transition_status="committed", actual_reward=committed["reward"],
                        actual_total_nominal_return=model._total_reward, terminated=model.terminated,
                        result_state_hash=model._engine.state_hash)
            if (outcome.info.get("action_id") != ids[chosen] or outcome.reward != scores[chosen]["reward"]
                    or (chosen and outcome.info.get("outcome_scope") != "permitted_nominal_model")):
                raise ValueError("Committed nominal transition differs from its authenticated immediate score")
            if prior_steps < 8:
                expected=expected_prefix[prior_steps]
                if (outcome.reward!=expected['actual_reward'] or model.terminated!=expected['terminated']
                        or model._engine.state_hash!=expected['result_state_hash']):
                    raise ValueError('Frozen prefix realized reward/state changed')
            check()
        complete = True
    except BaseException as exc:
        if isinstance(exc, InterruptedError):
            flags["cancelled"] = True
        error = exc
    finally:
        if original_preview is not None:
            if had_instance_preview:
                model._engine.preview_stroke = prior_instance_preview
            else:
                del model._engine.preview_stroke
        try:
            scorer.require(); scorer.bind_context(context)
            if context.fingerprint != context_seal or semantic_digest(scorer.report) != fit_seal:
                raise ValueError("Frozen continuation fit or deployment context changed")
            context.require_task(task)
            if model is not None:
                context.require_task(model)
            check()
        except BaseException as final_error:
            if error is not None:
                final_error.__cause__ = error
            error = final_error

    score_costs = {key: value-before_costs.get(key, 0) for key, value in scorer.costs.items()}
    accounting = {"method": TAIL_VERSION if tail_mode else VERSION,
        "complete": complete and error is None, **counts, **flags,
        "planning_seconds": time.perf_counter()-started, "time_budget_seconds": seconds,
        "max_calls": max_calls, "max_steps": task.max_steps, "initial_steps": 0,
        "explored_actions": list(sequence), "selected_actions": list(sequence) if error is None else None,
        "decisions": copy.deepcopy(decisions), "scorer_costs": score_costs,
        "estimated_incremental_return": 0. if model is None else model._total_reward,
        "nominal_history_hash": None if model is None else semantic_digest(model._history),
        "context_hash": context_seal, "fitted_report_hash": fit_seal,
        "source_hash": task._source_hash, "decision_model_hash": task.decision_model_hash,
        "base_parameter_hash": scorer.report["base_parameter_hash"],
        "fitted_head_hash": scorer.report["fitted_head_hash"],
        "STOP_prediction": "fixed_zero; chosen_only_if_no_actual_positive_action",
        "ranking": "positive_direct_Q_within_actual_positive_candidates; original_inventory_ties; actual_greedy_fallback",
        "immediate_reward_added_to_Q": False, "negative_preparation_allowed": False,
        "learned_negative_avoidance_claim": False, "successor_CNN_calls": 0,
        "initial_source_preparation": "outside_this_call; caller_must_charge",
        "intervention_index":8,"forced_action_id":forced_action,"diagnostic_only":True,
        "within_call_costs": "clone, full native inventories, immediate scoring, public observations, one Q batch per productive state, commits, integrity checks",
        "native_replay_required": True, "global_optimality_proven": False,
        "failure": None if error is None else {"type": type(error).__name__, "message": str(error)}}
    accounting['terminal_decision_catalog']=terminal_catalog if complete and error is None else None
    if tail_mode:
        accounting.update(
            prediction_semantics='raw_tail_clipped_at_zero_then_exact_current_nominal_reward_added_once',
            ranking='exact_immediate_reward_plus_nonnegative_tail_within_actual_positive_candidates; original_inventory_ties',
            immediate_reward_added_to_Q=False, immediate_reward_added_to_tail=True,
            additional_native_previews_for_tail=0, additional_observations_for_tail=0)
    if error is not None:
        error.accounting = accounting
        raise error
    return tuple(sequence), accounting
