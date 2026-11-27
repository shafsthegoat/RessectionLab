"""Opt-in ``rl-ledger`` (exact): one vectorized pass for the RL near-margin ledger.

``surgical_rl_env._ProtectedEngine._substep`` counts protected source cells
within ``near_margin_mm`` of every swept instrument capsule, calling
``ProtectionSet.capsule_hits`` once per capsule over all current cell centres.
Here all capsules of a substep are screened together. A cell is decided by the
screen only when its distance is farther than ``_BAND`` from the capsule's
threshold; any capsule with a cell inside that band is re-evaluated with
``capsule_hits`` itself, so the returned ID set equals the reference union.

Anything outside the stock path (a subclassed protection set, positions that
the reference would refuse, non-finite capsules) returns ``None`` and the caller
runs its reference loop, which raises the reference errors.
"""
from __future__ import annotations

import numpy as np

from . import surgical_rl_protection as _protection

# Absolute + relative band (mm) around each threshold. Different summation
# orders change distances by ~1e-13 mm at scene scale; this is ~1e4 wider.
_BAND = 1e-9


def _stock(protection):
    return type(protection).capsule_hits is _protection.ProtectionSet.capsule_hits


def capsule_union(protection, capsules, positions):
    """Set of ``int`` IDs equal to the union of ``capsule_hits`` calls, or None."""
    if not capsules:
        return set()
    if positions is None or not _stock(protection):
        return None
    _protection.ProtectionSet._require_geometry(protection)
    _, _, source_radius, index, _, _, _, source_cells = _protection._GEOMETRY[protection]
    clearance = protection.clearance_mm
    if not np.isfinite(clearance) or clearance < 0:
        return None
    clearance = float(clearance)
    _protection._ProtectionIndex._validate_metadata(index)
    points = np.asarray(positions, float)
    if points.shape != _protection._INDEXES[index][0].data.shape or not np.isfinite(points).all():
        return None
    starts, deltas, lengths, limits = [], [], [], []
    for start, end, radius in capsules:
        a, b = np.asarray(start, float), np.asarray(end, float)
        if (a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all()
                or not np.isfinite(radius) or radius < 0):
            return None
        delta = b - a
        starts.append(a); deltas.append(delta)
        lengths.append(float(delta @ delta))
        limits.append(float(radius) + clearance + source_radius + 1e-9)
    starts, deltas = np.array(starts), np.array(deltas)
    lengths, limits = np.array(lengths), np.array(limits)
    scale = float(np.abs(points).max(initial=0.) + np.abs(starts).max() + np.abs(deltas).max())
    bands = _BAND * (1. + limits + scale)
    # Cells outside every capsule's padded bounding box cannot be hits.
    reach = (limits + 2 * bands)[:, None]
    low = (np.minimum(starts, starts + deltas) - reach).min(0)
    high = (np.maximum(starts, starts + deltas) + reach).max(0)
    near = np.flatnonzero(((points >= low) & (points <= high)).all(1))
    offsets = points[near][None, :, :] - starts[:, None, :]
    along = np.einsum('kni,ki->kn', offsets, deltas)
    safe = np.where(lengths > 0, lengths, 1.)
    t = np.where(lengths[:, None] > 0, np.clip(along / safe[:, None], 0, 1), 0.)
    distance = np.linalg.norm(offsets - t[:, :, None] * deltas[:, None, :], axis=2)
    inside = distance <= (limits - bands)[:, None]
    uncertain = ~inside & (distance <= (limits + bands)[:, None])
    cells = source_cells[near]
    found = set()
    for row, (start, end, radius) in enumerate(capsules):
        if uncertain[row].any():
            found.update(map(int, protection.capsule_hits(start, end, radius, positions=positions)))
        else:
            found.update(map(int, cells[inside[row]]))
    return found


def near_margin_ids(engine, key, position, quaternion, positions, tools):
    """The ``_ProtectedEngine._substep`` near-margin set, or None for the reference."""
    from .surgical_motion import swept_envelopes
    if positions is None:
        return None  # Undeformed tissue: the reference already uses its k-d tree.
    margin = engine.rl_config.near_margin_mm
    capsules = []
    for other_key, other in engine.state['slots'].items():
        end = {'position': position, 'quaternion': quaternion} if other_key == key else other
        for a, b, radius, _ in swept_envelopes(tools[other['instrument']], engine.parameters(other), other, end):
            capsules.append((a, b, radius + margin))
    return capsule_union(engine.protection, capsules, positions)
