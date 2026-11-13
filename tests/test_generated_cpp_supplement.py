"""Supplementary fixture qualification; these are never patient outcomes."""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from resectionlab.generated_cpp_supplement import SIMULATION_FAMILIES, simulation_fixture


ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(str(ROOT/'scripts/benchmark_cpp_acceleration.py'))


def test_literal_simulation_traces_match_the_declared_proposal():
    proposal = json.loads((ROOT/'manifests/experiments/cpp-acceleration-simulation-v1.proposed.json').read_text())
    for family in SIMULATION_FAMILIES:
        _, actions = simulation_fixture(family)
        assert actions == proposal['families'][family]['actions']


def test_bare_invalid_pose_is_an_atomic_refusal_in_the_fixed_trace():
    engine, actions = simulation_fixture('simulation-target-episode')
    before = engine.snapshot()
    assert engine.step(actions[0])['status'] == 'rejected'
    assert engine.snapshot() == before


def test_exact_projection_keeps_array_types_bytes_and_negative_zero():
    bits = RUNNER['exact_bits']
    assert bits(-0.) != bits(0.)
    assert bits(np.array([1.], dtype='<f8')) != bits(np.array([1.], dtype='>f8'))
    assert bits(np.array([1.], dtype=np.float32)) != bits(np.array([1.], dtype=np.float64))
    assert bits(np.float32(1.)) != bits(1.)


@pytest.mark.parametrize('workload,episodes', [('simulation-steps', 0), ('rl-episodes', 6)])
def test_complete_suites_exercise_their_physics_and_rl_endpoints(workload, episodes):
    if episodes:
        pytest.importorskip('gymnasium')
    result, timings, diagnostics = RUNNER['perform_supplement'](workload)
    assert result['work']['episodes'] == episodes and result['work']['requestedAttempts'] > 0
    assert len(result['families']) == 6
    assert timings['completeSeconds'] >= timings['fixtureSeconds'] >= 0
    assert diagnostics['noPolicyTraining'] and not diagnostics['pytorchExecuted']
    assert diagnostics['auditProjectionAndRestoreSeconds'] >= 0
