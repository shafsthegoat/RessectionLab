"""Focused contract checks for the observed prepared-rescue policy adapter."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.patient_rescue_learning import (FEATURE_NAMES, OPTIONS, SCHEMA,
    FrozenRescueBCPolicy, features_from_sensors, load_bc_checkpoint, publish_observation)


def _task():
    shape = (20, 20, 20)
    centers = ((4, 4, 4), (4, 11, 4), (11, 4, 4), (11, 11, 4))
    fluid = {str(int(np.ravel_multi_index(center, shape))): float(i + 1)
             for i, center in enumerate(centers)}
    slots = np.zeros((2, 12), dtype=float)
    raw = SimpleNamespace(scene=SimpleNamespace(inverse=np.eye(4), shape=shape),
        engine=SimpleNamespace(state={"fluid": fluid}),
        field=lambda: np.array([10., 1., .2, 0.]),
        _observe=lambda: {"slots": slots})
    waypoints = {name: {"position": list(center)} for name, center in zip(
        ("contact", "coverage_lower", "coverage_forward", "suction_pool"), centers)}
    return SimpleNamespace(raw=raw, waypoints=waypoints, counts=np.zeros(6), decisions=[])


def test_current_route_observation_keeps_direction_and_ignores_history():
    task = _task()
    first, receipt = publish_observation(task)
    assert receipt["schema"] == SCHEMA
    assert receipt["routeFluidMm3"] == [1., 2., 3., 4.]
    assert first[3:7].tolist() == pytest.approx([1 / 3, 2 / 3, 1, 4 / 3])
    task.counts[:] = 12
    task.decisions.append({"option": "coagulate_forward"})
    again, repeated = publish_observation(task)
    np.testing.assert_array_equal(first, again)
    assert receipt["featuresSha256"] == repeated["featuresSha256"]
    task.raw.engine.state["fluid"][str(np.ravel_multi_index((4, 11, 4), task.raw.scene.shape))] = 5.
    changed, _ = publish_observation(task)
    assert changed[4] > first[4]
    assert changed[3] == first[3]


def test_checkpoint_binds_task_options_and_current_sensor_actor(tmp_path):
    weight = np.zeros((len(OPTIONS), len(FEATURE_NAMES)), dtype=np.float32)
    bias = np.full(len(OPTIONS), -10., dtype=np.float32)
    bias[OPTIONS.index("finish_verify")] = 1.
    weight[OPTIONS.index("coagulate_center"), 2] = 12.
    weight[OPTIONS.index("aspirate_pool"), 0] = 12.
    record = dict(schema="resectionlab.patient-rescue-bc-checkpoint.v1",
        observationSchema=SCHEMA, taskManifestSha256="source-bound-task",
        options=list(OPTIONS), featureNames=list(FEATURE_NAMES),
        weight=weight.tolist(), bias=bias.tolist())
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(record))
    policy, digest = load_bc_checkpoint(path, manifest_sha256="source-bound-task")
    assert len(digest) == 64
    slots = np.zeros((2, 12))
    bleeding = features_from_sensors([.3, 1., .5, 0.], [0., 0., 0., 0.], slots)
    pooled = features_from_sensors([3., 0., 0., 0.], [0., 0., 0., 3.], slots)
    clean = features_from_sensors([0., 0., 0., 0.], [0., 0., 0., 0.], slots)
    assert OPTIONS[policy.predict(bleeding)[0]] == "coagulate_center"
    assert OPTIONS[policy.predict(pooled)[0]] == "aspirate_pool"
    assert OPTIONS[policy.predict(clean)[0]] == "finish_verify"
    with pytest.raises(ValueError, match="identity differs"):
        load_bc_checkpoint(path, manifest_sha256="different-task")
    record["options"][0], record["options"][1] = record["options"][1], record["options"][0]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity differs"):
        load_bc_checkpoint(path, manifest_sha256="source-bound-task")


def test_policy_prediction_keeps_parameters_unchanged():
    weight = np.zeros((len(OPTIONS), len(FEATURE_NAMES)), dtype=np.float32)
    bias = np.arange(len(OPTIONS), dtype=np.float32)
    policy = FrozenRescueBCPolicy(weight, bias)
    before = (policy.weight.copy(), policy.bias.copy())
    for _ in range(3):
        assert policy.predict(np.zeros(len(FEATURE_NAMES), dtype=np.float32))[0] == len(OPTIONS) - 1
    np.testing.assert_array_equal(policy.weight, before[0])
    np.testing.assert_array_equal(policy.bias, before[1])
