"""Opt-in V2 lattice bond scans must return the reference loop's exact floats."""
import numpy as np
import pytest

from resectionlab import accel_mechanics


def reference_scale(u, edges, strain):
    return min((strain * length / max(np.linalg.norm(u[j] - u[i]), 1e-15) for i, j, length in edges), default=1.)


def reference_ratio(u, edges):
    return max((float(np.linalg.norm(u[j] - u[i]) / length) for i, j, length in edges), default=0.)


@pytest.mark.parametrize('seed', range(40))
def test_bond_scans_are_bit_identical(seed):
    rng = np.random.default_rng(seed)
    nodes = int(rng.integers(2, 400))
    u = rng.normal(size=(nodes, 3)) * 10.0 ** rng.integers(-9, 1)
    if seed % 5 == 0:
        u[rng.integers(0, nodes)] = 0.  # zero stretches hit the 1e-15 floor
    count = int(rng.integers(1, 1200))
    lengths = rng.choice([1., 1.25, .7, np.sqrt(2.)], count)
    edges = [(int(i), int(j), float(length)) for i, j, length in
             zip(rng.integers(0, nodes, count), rng.integers(0, nodes, count), lengths)]
    strain = float(rng.choice([.05, .2, .35]))
    assert accel_mechanics.bond_scale(u, edges, strain) == reference_scale(u, edges, strain)
    assert accel_mechanics.bond_ratio(u, edges) == reference_ratio(u, edges)


def test_ties_and_empty_bonds():
    u = np.zeros((3, 3)); u[1] = [1., 0., 0.]; u[2] = [0., 1., 0.]
    edges = [(0, 1, 1.), (0, 2, 1.), (1, 2, 2.)]
    assert accel_mechanics.bond_scale(u, edges, .2) == reference_scale(u, edges, .2)
    assert accel_mechanics.bond_ratio(u, edges) == reference_ratio(u, edges)
    assert accel_mechanics.bond_scale(u, [], .2) == 1. and accel_mechanics.bond_ratio(u, []) == 0.
