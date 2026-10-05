"""Closed raw-observation contract for experimental axis scratch policies.

This does not authorize procedural transfer, population policies, a different
proposal model, or a new simulator encoding. Policy unit transforms are bound
separately. No extra observation, proposal, transition or policy call is made.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import numpy as np

from .native_axis_simulation import AXIS_ADAPTER_VERSION, AxisColumnNativeSimulator
from .native_resection import NATIVE_RESECTION_VERSION
from .native_simulation import NATIVE_ACTION_FEATURE_NAMES
from .policy_inputs import NATIVE_ACTION_NAMES, NATIVE_STATE_NAMES

AXIS_OBSERVATION_SCHEMA_VERSION = "native-axis-raw-15x6-observation-v1"


def axis_observation_contract(simulator: Any, observation: Any) -> dict[str, Any]:
    """Bind an already served observation from the exact supported axis class.

    Dimensions alone do not authorize a simulator or wrapper. The raw schema
    retains the inherited physical feature definitions and source-bound model;
    the learner's independent profile contract controls the actor transform.
    """
    if type(simulator) is not AxisColumnNativeSimulator:
        raise ValueError("Axis input profiles require the exact axis observation backend")
    simulator.assert_model_frozen()
    if tuple(NATIVE_ACTION_FEATURE_NAMES) != NATIVE_ACTION_NAMES:
        raise ValueError("Axis native action feature order changed")
    actions, state = np.asarray(observation.action_features), np.asarray(observation.state_features)
    mask, ids = np.asarray(observation.action_mask), tuple(observation.action_ids)
    if (actions.ndim != 2 or actions.shape[1] != 15 or state.shape != (6,)
            or actions.dtype != np.float32 or state.dtype != np.float32
            or mask.dtype != np.bool_ or mask.shape != (len(actions),)
            or len(ids) != len(actions) or not 1 <= len(ids) <= simulator.config.max_actions
            or ids[0] != "STOP" or len(set(ids)) != len(ids)
            or not mask.all() or not np.isfinite(actions).all() or not np.isfinite(state).all()):
        raise ValueError("Axis raw observation differs from the certified native 15+6 schema")
    # Axis emits actual certified rows without padding; generic policies still
    # support masked padding in other environments and algebra-only tests.
    if actions[0, 0] != 1 or np.any(actions[0, 1:] != 0) or np.any(actions[1:, 0] != 0):
        raise ValueError("Axis STOP row or action indicator changed")
    record = {
        "version": AXIS_OBSERVATION_SCHEMA_VERSION,
        "backend": AXIS_ADAPTER_VERSION,
        "observation_encoding": "RAW",
        "legacy_model_input_profile_scope": "RAW simulator feature encoding; policy transform bound separately",
        "action_feature_names": list(NATIVE_ACTION_NAMES),
        "action_feature_units": ["binary", "mm3", "mm3", "spatial_surrogate_mm3", "spatial_surrogate_mm3",
            "mm", "mm", "mm", "binary", "fraction_of_action_budget", "fraction", "fraction", "mm3",
            "spatial_surrogate_mm3", "spatial_surrogate_mm3"],
        "state_feature_names": list(NATIVE_STATE_NAMES),
        "state_feature_units": ["fraction", "fraction", "fraction", "fraction", "binary", "binary"],
        "depth_semantics": "completed nonSTOP actions / max_steps; not physical depth",
        "adjacent_target_semantics": "supplied target-label occupancy in clipped 3x3x3 endpoint neighborhood; not residual-only",
        "partial_contact_semantics": "new retained contact excluding prior charged contact and cells removed now",
        "actor_evidence": "nominal fields and actual cavity only; hidden world excluded",
        "removal_version": NATIVE_RESECTION_VERSION,
        "decision_model_hash": simulator.decision_model_hash,
        "proposal_model_hash": simulator._proposer.model_hash,
        "reward": asdict(simulator.config.reward),
        "partial_contact_weight": simulator.partial_contact_weight,
        "max_steps": simulator.config.max_steps,
        "max_actions_including_stop": simulator.config.max_actions,
        "scope": "scratch REINFORCE compatibility only; no transfer or resume authorization",
    }
    record["contract_hash"] = "sha256:" + hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return record
