"""Opt-in exact contact QP solver (``contact``).

Solves the frictionless unilateral problem both contact modules pose:

    minimize 0.5 x'Cx - d'x  subject to x >= 0,

with C the symmetric positive-semidefinite contact compliance (mm/N) and d the
gap deficit (mm). Equivalently the LCP  g = Cx - d >= 0, x >= 0, x'g = 0.

Block principal pivoting (Judice and Pires 1994), with Murty's single-index
rule as the anti-cycling safeguard, reaches the active set in a few linear
solves instead of the many L-BFGS-B iterations a tight gradient tolerance
needs. The result is certified here against the caller's own gap and
complementarity tolerances; anything uncertified returns ``None`` so the
caller runs its reference optimizer unchanged. Callers then apply their usual
equilibrium/KKT refusal checks.
"""
from __future__ import annotations

import numpy as np

_BACKOFF = 3


def _subsolve(compliance, deficit, passive):
    x = np.zeros(len(deficit))
    index = np.flatnonzero(passive)
    if len(index):
        block = compliance[np.ix_(index, index)]
        rhs = deficit[index]
        try:
            factor = np.linalg.cholesky(block)
            x[index] = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
        except np.linalg.LinAlgError:
            rcond = 64 * np.finfo(float).eps * len(index)
            x[index] = np.linalg.lstsq(block, rhs, rcond=rcond)[0]
    return x


def solve(compliance, deficit, *, gap_tolerance, energy_tolerance, check=lambda: None, max_pivots=None):
    """Return ``(forces, pivots)`` certified to the tolerances, or ``None``."""
    compliance = np.asarray(compliance, float)
    deficit = np.asarray(deficit, float)
    m = len(deficit)
    if compliance.shape != (m, m) or not np.isfinite(compliance).all() or not np.isfinite(deficit).all():
        return None
    if max_pivots is None:
        max_pivots = 10 * m + 20
    # Warm start: constraints the unloaded body already violates.
    passive = deficit > gap_tolerance / 10
    best = m + 1
    backoff = _BACKOFF
    x = np.zeros(m)
    for pivot in range(max_pivots):
        check()
        x = _subsolve(compliance, deficit, passive)
        gap = compliance @ x - deficit
        bad = (passive & (x < 0.)) | (~passive & (gap < -gap_tolerance / 10))
        count = int(np.count_nonzero(bad))
        if not count:
            break
        if count < best:
            best = count; backoff = _BACKOFF
            passive ^= bad
        elif backoff > 0:
            backoff -= 1
            passive ^= bad
        else:
            # Murty's rule: flip only the largest infeasible index.
            passive[np.flatnonzero(bad)[-1]] ^= True
    else:
        return None
    x = np.maximum(x, 0.)
    gap = compliance @ x - deficit
    if (not np.isfinite(x).all() or np.min(gap, initial=0.) < -gap_tolerance
            or np.max(np.abs(x * gap), initial=0.) > energy_tolerance):
        return None
    return x, pivot + 1


def applies(minimize):
    """Only the stock SciPy optimizer is replaced; an instrumented one is kept."""
    from scipy import optimize
    return minimize is optimize.minimize


def polished(compliance, deficit, *, gap_tolerance, energy_tolerance, check=lambda: None):
    """``solve`` followed by the callers' own active-set ``lstsq`` polish.

    Both contact modules finish L-BFGS-B with this exact polish, so whenever
    the active sets agree the returned forces are the reference's floats.
    Returns ``(forces, pivots, polished)`` or ``None``.
    """
    found = solve(compliance, deficit, gap_tolerance=gap_tolerance,
                  energy_tolerance=energy_tolerance, check=check)
    if found is None:
        return None
    forces, pivots = found
    active = np.flatnonzero(forces > 0.)
    if len(active):
        values = np.linalg.lstsq(compliance[np.ix_(active, active)], deficit[active],
                                 rcond=64 * np.finfo(float).eps * len(active))[0]
        candidate = np.zeros(len(deficit)); candidate[active] = values
        gap = compliance @ candidate - deficit
        if (np.min(candidate, initial=0.) >= 0 and np.min(gap, initial=0.) >= -gap_tolerance
                and np.max(np.abs(candidate * gap), initial=0.) <= energy_tolerance):
            return candidate, pivots, True
    return forces, pivots, False
