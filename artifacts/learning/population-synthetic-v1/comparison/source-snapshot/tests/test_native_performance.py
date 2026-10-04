"""Semantic regression checks for bounded native preparation reuse.

Call-count checks verify the bottleneck is removed without brittle wall-clock
thresholds. Source-case timings are retained separately in artifacts/performance.
"""
from __future__ import annotations

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine
from resectionlab.native_simulation import NativeSequentialSimulator
from resectionlab.worlds import WorldGeneratorConfig


def make_simulator():
    tissue = np.ones((7, 7, 6), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 3:] = 1
    config = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 3.), NATIVE_GENERIC_TOOLS,
        "native-cache-analytic-source", "synthetic solid cube")
    motor = np.zeros(tissue.shape)
    motor[2, :, :] = 1
    return NativeSequentialSimulator(config, [(2, 3, 4), (4, 3, 4)],
        candidate_entries_mm=[(2, 3, -.5), (4, 3, -.5)], nominal_motor=motor,
        world_generator=WorldGeneratorConfig(translation_scale_mm=(.3, .3, .3)), max_steps=3)


def test_initial_geometry_reuse_eliminates_repeated_previews_but_fresh_arm_is_cold(monkeypatch):
    original = NativeResectionEngine.preview_stroke
    calls = []
    def counted(self, *args, **kwargs):
        calls.append(self.state_hash)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", counted)
    sim = make_simulator()
    initial_calls = len(calls)
    assert initial_calls == 4
    original_observation = sim.observation()
    hashes = []
    for seed in (10, 11, 12):
        observation = sim.reset(seed)
        np.testing.assert_array_equal(observation.action_features, original_observation.action_features)
        np.testing.assert_array_equal(observation.state_features, original_observation.state_features)
        hashes.append(sim.metrics()["episode_world_hash"])
    assert len(calls) == initial_calls
    assert len(set(hashes)) == 3
    fresh = sim.fresh()
    assert len(calls) == 2 * initial_calls
    assert fresh._initial_geometry_snapshot is not sim._initial_geometry_snapshot
    assert fresh.decision_model_hash == sim.decision_model_hash


def test_cached_reset_keeps_native_certificates_cavity_and_history_identical():
    sim = make_simulator()
    action = sim.observation().action_ids[1]
    sim.reset(10)
    cold = sim.fresh()
    cold.reset(10)
    first = sim.step(action)
    expected = cold.step(action)
    assert first.reward == expected.reward
    assert first.info == expected.info
    assert sim.metrics() == cold.metrics()
    sim.reset(10)
    assert not sim.removed_mask.any()
    assert not sim.engine.history
    assert sim.engine.revision == 0
    repeated = sim.step(action)
    assert repeated.info == first.info
    assert repeated.reward == first.reward
    branch = sim.clone()
    branch.reset(10)
    assert branch.step(action).info == first.info
    assert sim.metrics() == cold.metrics()


def test_cached_reset_still_rejects_model_mutation_before_reusing_certificates():
    sim = make_simulator()
    object.__setattr__(sim.native_config, "max_tip_step_mm", .1)
    with pytest.raises(RuntimeError, match="Decision model changed"):
        sim.reset(12)
