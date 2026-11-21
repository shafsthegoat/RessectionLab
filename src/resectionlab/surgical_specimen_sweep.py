"""Whole-cell clearance for a tumor fragment held by operative forceps."""

from itertools import product

import numpy as np

from .surgical_motion import envelopes, pair_conflict
from .surgical_operative_exposure import object_envelopes
from .surgical_rl_protection import cell_radius


def _swept_material(engine, hold, offset):
    # Endpoint occupancy handles a single face step exactly. On a diagonal,
    # two translated voxel interiors can intersect between the reported raster
    # poses. In voxel coordinates, two unit cells overlap if all center
    # separations are strictly below one at the same fractional time.
    start = np.asarray(hold['offset'])
    delta = np.asarray(offset) - start
    if np.abs(delta).sum() <= 1:
        return
    cells = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape)) + start
    owner = engine.occupancy()
    own = set(hold['cells'])
    for shift in product(*(range(min(0, n), max(0, n) + 1) for n in delta)):
        if not any(shift) or np.array_equal(shift, delta):
            continue
        bounds = [sorted(((j - 1) / n, (j + 1) / n)) for j, n in zip(shift, delta) if n]
        if max(0.0, *(a for a, _ in bounds)) >= min(1.0, *(b for _, b in bounds)):
            continue
        indices = cells + shift
        if np.any(indices < 0) or np.any(indices >= engine.scene.shape):
            raise ValueError('DISPLACEMENT_OUTSIDE_GRID')
        if set(owner.material_ids(np.ravel_multi_index(indices.T, engine.scene.shape))) - own:
            raise ValueError('OPERATIVE_SPECIMEN_SWEPT_ANATOMY_COLLISION')


def check(engine, holder, actor, swept, offset):
    """Check the accepted-to-proposed cargo translation against current equipment."""
    hold = engine.state['holds'][holder]
    basis = engine.scene.affine[:3, :3]
    centers = engine.scene.points(hold['cells'])
    starts = centers + basis @ np.asarray(hold['offset'])
    ends = centers + basis @ np.asarray(offset)
    radius = cell_radius(engine.scene.affine)
    cargo = [(a, b, radius, 'specimen') for a, b in zip(starts, ends)]
    if engine._protected_ids:
        protected = engine._positions()
        for a, b, r, _ in cargo:
            if len(engine.protection.capsule_hits(a, b, r, positions=protected)):
                raise ValueError('PROTECTED_STRUCTURE: held specimen enters declared clearance')
    _swept_material(engine, hold, offset)

    for hand, slot in engine.state['slots'].items():
        parts = (
            swept
            if hand == actor
            else envelopes(
                engine.tools[slot['instrument']],
                engine.parameters(slot),
                slot['position'],
                slot['quaternion'],
            )
        )
        if hand == holder:
            parts = [part for part in parts if part[3] == 'shaft']
        if pair_conflict(cargo, parts):
            raise ValueError('OPERATIVE_SPECIMEN_TOOL_COLLISION:' + hand)

    rest = getattr(engine, '_bone_rest', ())
    if rest and pair_conflict(cargo, rest):
        raise ValueError('OPERATIVE_SPECIMEN_BONE_REST_COLLISION')
    for obj in engine.state['operativeExposureV1']['objects']:
        if obj['status'] != 'retrieved' and pair_conflict(cargo, object_envelopes(engine, obj)):
            raise ValueError('OPERATIVE_SPECIMEN_OBJECT_COLLISION:' + obj['kind'])
