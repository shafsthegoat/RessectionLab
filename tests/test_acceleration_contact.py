"""Opt-in contact QP solver: certified KKT, agreement with the reference optimizer."""
import numpy as np
import pytest
from scipy.optimize import minimize

from resectionlab import accel_contact

GAP, ENERGY = 1e-9, 1e-10


def reference(c, d):
    """The contact modules' L-BFGS-B solve plus their active-set polish."""
    m = len(d)
    def objective(x):
        return float(.5 * x @ (c @ x) - x @ d), c @ x - d
    x = minimize(objective, np.zeros(m), method='L-BFGS-B', jac=True, bounds=[(0., None)] * m,
                 options={'gtol': GAP / 10, 'ftol': 0., 'maxiter': 1000, 'maxfun': 5000, 'maxls': 50}).x
    active = np.flatnonzero(x > 0.)
    if len(active):
        values = np.linalg.lstsq(c[np.ix_(active, active)], d[active], rcond=64 * np.finfo(float).eps * len(active))[0]
        candidate = np.zeros(m); candidate[active] = values; gap = c @ candidate - d
        if candidate.min(initial=0.) >= 0 and gap.min(initial=0.) >= -GAP and np.abs(candidate * gap).max(initial=0.) <= ENERGY:
            x = candidate
    return x


def problem(seed, m, rank=None):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(m, rank or m)) * .05
    c = basis @ basis.T + (0. if rank else 1e-3) * np.eye(m)
    d = rng.normal(size=m) * 1e-3
    if rank:  # keep the problem bounded: the deficit lies in the range of C
        d = c @ np.abs(rng.normal(size=m)) * .5 - np.abs(d)
    return c, d


@pytest.mark.parametrize('seed', range(30))
@pytest.mark.parametrize('m', [1, 4, 32, 120])
def test_certified_and_matches_reference(seed, m):
    c, d = problem(seed, m)
    found = accel_contact.polished(c, d, gap_tolerance=GAP, energy_tolerance=ENERGY)
    assert found is not None
    x, pivots, _ = found
    gap = c @ x - d
    assert x.min() >= 0 and gap.min() >= -GAP and np.abs(x * gap).max() <= ENERGY
    expected = reference(c, d)
    assert np.abs(x - expected).max() <= 1e-9 * max(np.abs(expected).max(), 1e-12)
    assert pivots <= 10 * m + 20


@pytest.mark.parametrize('seed', range(10))
def test_rank_deficient_compliance_is_solved_or_deferred(seed):
    c, d = problem(seed, 24, rank=6)
    found = accel_contact.polished(c, d, gap_tolerance=GAP, energy_tolerance=ENERGY)
    if found is not None:  # None means the caller runs its reference optimizer
        x, _, _ = found
        gap = c @ x - d
        assert x.min() >= 0 and gap.min() >= -GAP and np.abs(x * gap).max() <= ENERGY


def test_nonfinite_or_misshaped_input_defers_to_reference():
    assert accel_contact.solve(np.eye(2), np.array([np.nan, 1.]), gap_tolerance=GAP, energy_tolerance=ENERGY) is None
    assert accel_contact.solve(np.eye(3), np.ones(2), gap_tolerance=GAP, energy_tolerance=ENERGY) is None
