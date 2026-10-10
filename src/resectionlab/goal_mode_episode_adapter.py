"""Policy proposal -> existing shared sealed-plan contract; no alternate world.

This callable is staged but has not been run against a native task. The caller
must install the plan through surface_contact_episode's shared exporter and
retain its full-history equality check, public goal decoration and envelope.
"""
from __future__ import annotations

from .core import freeze_json, semantic_digest
from .goal_mode_spatial_policy import GoalModeSpatialPolicy
from .public_surface_contact import OBSERVATION_VERSION, SurfaceContactDevelopmentContext
from .spatial_policy import parameter_hash


def plan_goal_mode_strategy(policy, task, *, context):
    """Return the same (plan, seal, accounting) shape as the existing planner.

    Planning uses SurfaceContactTask.planning_clone, observations, step, metrics
    and termination. Geometry, candidate certification, modes, reward and state
    evolution therefore remain owned by the existing NativeSpatialTask path.
    """
    if type(policy) is not GoalModeSpatialPolicy:
        raise TypeError("Shared contact planner requires the versioned goal/mode policy")
    if type(context) is not SurfaceContactDevelopmentContext:
        raise TypeError("Shared contact planner requires its captured public task context")
    context.require_task(task)
    initial_observation = task.observation()
    if task.terminated or initial_observation.base.base.state_features[0] != 0:
        raise ValueError("A sealed whole-episode plan requires a fresh shared task")
    nominal = task.planning_clone()
    context.require_task(nominal)
    initial_weights = parameter_hash(policy)
    architecture = policy.architecture_hash
    actions, observation_ids = [], []
    was_training = policy.training
    try:
        policy.eval()
        while not nominal.terminated:
            if len(actions) >= nominal.max_steps:
                raise RuntimeError("Shared episode exceeded its declared action horizon")
            context.require_task(nominal)
            if parameter_hash(policy) != initial_weights or policy.architecture_hash != architecture:
                raise RuntimeError("Policy changed during inference")
            observation = nominal.observation()
            action = policy.act(observation, context=context)
            if (action not in observation.action_ids
                    or not observation.action_mask[observation.action_ids.index(action)]):
                raise RuntimeError("Policy selected an action outside the observed inventory")
            observation_ids.append(observation.fingerprint)
            actions.append(action)
            nominal.step(action)
        if parameter_hash(policy) != initial_weights or policy.architecture_hash != architecture:
            raise RuntimeError("Policy changed during inference")
        context.require_task(nominal)
    finally:
        policy.train(was_training)
    plan = freeze_json({"decision_model_hash": task.decision_model_hash,
        "source_hash": task.case.source_hash, "actions": actions,
        "history": nominal.metrics()["history"], "observation_contract": OBSERVATION_VERSION,
        "max_steps": task.max_steps})
    accounting = {"selector": "GOAL_MODE_POLICY_V1", "actor_forward_calls": len(actions),
        "model_transition_calls": len(actions), "optimizer_updates": 0,
        "observation_ids": observation_ids, "architecture_hash": architecture,
        "parameter_hash": initial_weights, "context_declaration_hash": context.declaration_hash,
        "checkpoint_identity": policy.checkpoint_identity(),
        "cost_scope": "planning_calls_only; caller_accounts_inventory_execution_replay_and_wall_time",
        "training_or_checkpoint_lineage_verified": False}
    return plan, semantic_digest(plan), accounting
