"""Opt-in contact/time-limited ultrasonic aspiration research surrogate.

The rate is an explicit uncalibrated engineering input, not a device/patient
fit. Capacity is allocated across actual current-tip contacts, never a whole
requested stroke. Partial capacity does not remove tissue. The parent engine
must retain its ordinary shaft/material/protection checks and limit advance
against retained contacts before calling apply. No learning is performed here.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import math

import numpy as np

SCHEMA = 'resectionlab.cusa-contact.v3'
TIP_RADIUS_MM = 1.5
SHAFT_RADIUS_MM = .5
DEFAULT_PROFILE = {
    'schema': SCHEMA,
    'provenance': {'kind': 'assumed-research', 'description':
        'Uncalibrated contact-capacity surrogate; rate is an engineering assumption, not device or patient calibration.'},
    'tipRadiusMm': TIP_RADIUS_MM, 'shaftRadiusMm': SHAFT_RADIUS_MM,
    'capacityMm3PerSecond': 2., 'maxActivation': 1.,
}


def default_profile():
    return copy.deepcopy(DEFAULT_PROFILE)


def validate(value):
    """Validate a complete profile, or an optional profile on an exact scene."""
    if hasattr(value, 'properties'):
        value = value.properties.get('cusaV3')
        if value is None:
            return None
    if not isinstance(value, dict) or set(value) != set(DEFAULT_PROFILE) or value.get('schema') != SCHEMA:
        raise ValueError('COMPLETE_EXPLICIT_CUSA_V3_PROFILE_REQUIRED')
    provenance = value['provenance']
    if not isinstance(provenance, dict) or set(provenance) != {'kind', 'description'} or provenance.get('kind') != 'assumed-research' or not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
        raise ValueError('EXPLICIT_ASSUMED_RESEARCH_CUSA_PROVENANCE_REQUIRED')
    for key in ('tipRadiusMm', 'shaftRadiusMm', 'capacityMm3PerSecond', 'maxActivation'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('INVALID_CUSA_V3_PROFILE:' + key)
    if value['tipRadiusMm'] != TIP_RADIUS_MM or value['shaftRadiusMm'] != SHAFT_RADIUS_MM:
        raise ValueError('CUSA_V3_FIXED_CATALOG_GEOMETRY_REQUIRED')
    if value['maxActivation'] != 1.:
        raise ValueError('CUSA_V3_NORMALIZED_ACTIVATION_REQUIRED')
    return value


def profile(engine):
    return engine.scene.properties.get('cusaV3') or (DEFAULT_PROFILE if engine.state.get('toolMechanicsRevision') == 1 else None)


def initialize(engine):
    p = profile(engine)
    if p is None:
        return
    validate(p)
    engine.state.setdefault('cusaV3', {'schema': SCHEMA,
        'progressMm3': {}, 'progressExact': {}, 'contactSeconds': {},
        'inputCapacityMm3': 0., 'aspiratedSolidMm3': 0.,
        'unusedCapacityMm3': 0., 'discardedCapacityMm3': 0.,
        'effectiveActivationSeconds': 0.})


def _nonnegative(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('INVALID_CUSA_V3_INPUT:' + label)
    return float(value)


def validate_parameters(params):
    """Power is a normalized CUSA-only activation proxy, not watts."""
    if not isinstance(params, dict):
        raise ValueError('INVALID_CUSA_V3_PARAMETERS')
    radius = _nonnegative(params.get('radiusMm', TIP_RADIUS_MM), 'radiusMm')
    shaft = _nonnegative(params.get('shaftRadiusMm', SHAFT_RADIUS_MM), 'shaftRadiusMm')
    if radius != TIP_RADIUS_MM or shaft != SHAFT_RADIUS_MM:
        raise ValueError('CUSA_V3_FIXED_CATALOG_GEOMETRY_REQUIRED')
    activation = _nonnegative(params.get('power', 1.), 'power')
    duty = _nonnegative(params.get('duty', 1.), 'duty')
    if activation > 1. or duty > 1.:
        raise ValueError('CUSA_V3_ACTIVATION_AND_DUTY_MUST_BE_IN_0_1')
    return activation, duty


def reconcile_removed(engine):
    """Discard work proxies for source cells removed by another mechanism.

    This is discarded *capacity*, not aspirated mass. Integrators should call
    it at action finalization as well as apply, so mixed-tool ledgers stay fresh.
    """
    if profile(engine) is None:
        return 0.
    initialize(engine)
    state = engine.state['cusaV3']
    removed = set(engine.state['removed'])
    stale = [key for key in state['progressMm3'] if int(key) in removed]
    discarded = math.fsum(state['progressMm3'].pop(key) for key in stale)
    for key in stale:
        state['contactSeconds'].pop(key, None)
        state['progressExact'].pop(key, None)
    state['discardedCapacityMm3'] += discarded
    return discarded


def _position(value):
    if value is None:
        raise ValueError('CUSA_V3_EXPLICIT_TIP_POSITION_REQUIRED')
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('CUSA_V3_FINITE_RAS_MM_POSITION_REQUIRED')
    return point


def _ordered_contacts(engine, cells, position):
    labels = engine.scene.labels.ravel()
    ids = set()
    for cell in cells:
        if not isinstance(cell, (int, np.integer)) or isinstance(cell, (bool, np.bool_)) or not 0 <= cell < labels.size:
            raise ValueError('INVALID_CUSA_V3_SOURCE_CELL')
        ids.add(int(cell))
    # A caller cannot turn a swept/remote candidate set into a removal eraser.
    ids.intersection_update(engine.touched(position, position, TIP_RADIUS_MM))
    ids.difference_update(engine.state['removed'])
    ids = [cell for cell in ids if labels[cell] and engine.by_label[int(labels[cell])]['material'] in ('soft_tissue', 'tumor')]
    positions = {cell: center for cell, center in zip(ids, engine.scene.points(ids))}
    for hold in engine.state['holds'].values():
        if 'target' not in hold:
            displacement = engine.scene.affine[:3, :3] @ np.asarray(hold['offset'])
            for cell in set(hold['cells']).intersection(positions):
                positions[cell] = positions[cell] + displacement
    for cell in positions:
        displacement = engine.state.get('deformation', {}).get(str(cell))
        if displacement is not None:
            positions[cell] = engine.scene.points([cell])[0] + np.asarray(displacement)
    return sorted(ids, key=lambda cell: (float(np.sum((positions[cell] - position) ** 2)), cell))


def apply(engine, cells, dt, params, *, position=None, active=True):
    """Apply one total mm³ capacity budget to instantaneous reachable contacts.

    Explicit position is required even when no cells are supplied. Effective
    activation belongs to the engine operation (apply can be active without a
    latched switch). Inactive calls and zero activation cannot remove tissue.
    Returns events only; engine.state.cusaV3 contains the cumulative ledger.
    """
    p = validate(profile(engine))
    dt = _nonnegative(dt, 'dt')
    activation, duty = validate_parameters(params)
    position = _position(position)
    if type(active) is not bool:
        raise ValueError('CUSA_V3_EFFECTIVE_ACTIVATION_MUST_BE_BOOLEAN')
    ids = _ordered_contacts(engine, cells, position)
    if set(ids).intersection(getattr(engine, '_protected_ids', ())):
        raise ValueError('PROTECTED_STRUCTURE: CUSA contact includes protected source membership')
    budget = p['capacityMm3PerSecond'] * activation * duty * dt if active else 0.
    if not math.isfinite(budget):
        raise ValueError('CUSA_V3_CAPACITY_OVERFLOW')
    initialize(engine)
    discarded = reconcile_removed(engine)
    state = engine.state['cusaV3']
    # Preserve the exact binary capacity supplied by this interval. Rational
    # pending work prevents a rounded display value from authorizing clearance;
    # JSON integer pairs keep this state replayable without Python objects.
    remaining = Fraction.from_float(budget)
    volume = Fraction.from_float(float(engine.scene.voxel_mm3))
    new = []
    spent = []
    for cell in ids if budget > 0. else ():
        key = str(cell)
        old = Fraction(*state['progressExact'][key]) if key in state['progressExact'] else Fraction(0)
        required = volume - old
        allocated = min(required, remaining)
        if allocated <= 0.:
            continue
        state['contactSeconds'][key] = state['contactSeconds'].get(key, 0.) + dt
        remaining -= allocated
        spent.append(float(allocated))
        # No epsilon/percentage rounding can authorize whole-cell removal.
        if allocated >= required:
            new.append(cell)
            state['progressMm3'].pop(key, None)
            state['progressExact'].pop(key, None)
            state['contactSeconds'].pop(key, None)
        else:
            progress = old + allocated
            state['progressMm3'][key] = float(progress)
            state['progressExact'][key] = [progress.numerator, progress.denominator]
    aspirated = len(new) * engine.scene.voxel_mm3
    state['inputCapacityMm3'] += budget
    state['aspiratedSolidMm3'] += aspirated
    state['unusedCapacityMm3'] += float(remaining)
    state['effectiveActivationSeconds'] += activation * duty * dt if active else 0.
    if new:
        engine.state['removed'] = sorted(set(engine.state['removed']).union(new))
        engine._occupancy_cache = None
    return [{'kind': 'cusa_contact', 'inputCapacityMm3': budget,
        'allocatedCapacityMm3': math.fsum(spent), 'unusedCapacityMm3': float(remaining),
        'aspiratedSolidMm3': aspirated, 'discardedCapacityMm3': discarded,
        'removedCells': new, 'pendingCells': len(state['progressMm3'])}] if budget or discarded else []


def contact_limited_pose(engine, start, requested, radius=TIP_RADIUS_MM):
    """Stop at first retained compatible tissue; no state/progress is changed."""
    validate(profile(engine))
    validate_parameters({'radiusMm': radius})
    start, requested = _position(start), _position(requested)
    def contacts(point):
        return _ordered_contacts(engine, engine.touched(point, point, TIP_RADIUS_MM), point)
    initial = contacts(start)
    if initial:
        return start.copy(), initial
    distance = float(np.linalg.norm(requested - start))
    steps = max(1, math.ceil(distance / (float(engine.scene.spacing.min()) / 4)))
    low = start.copy()
    for index in range(1, steps + 1):
        high = start + (requested - start) * (index / steps)
        found = contacts(high)
        if found:
            for _ in range(32):
                mid = (low + high) / 2
                if contacts(mid):
                    high = mid
                else:
                    low = mid
            return high, contacts(high)
        low = high
    return requested.copy(), []


def metrics(engine):
    state = engine.state.get('cusaV3', {})
    pending = math.fsum(state.get('progressMm3', {}).values())
    return {'cusaInputCapacityMm3': state.get('inputCapacityMm3', 0.),
        'cusaAspiratedSolidMm3': state.get('aspiratedSolidMm3', 0.),
        'cusaPendingCapacityMm3': pending,
        'cusaUnusedCapacityMm3': state.get('unusedCapacityMm3', 0.),
        'cusaDiscardedCapacityMm3': state.get('discardedCapacityMm3', 0.),
        'cusaPendingCells': len(state.get('progressMm3', {}))}


def descriptor(engine):
    return {'enabled': profile(engine) is not None, 'profile': copy.deepcopy(profile(engine)),
        'calibration': 'uncalibrated assumed research surrogate',
        'activation': 'power and duty are dimensionless fractions in [0,1]',
        'limits': ['Fixed generic catalog geometry is not a manufacturer tip specification.',
            'Rate, constant capture, and material response are uncalibrated engineering assumptions.',
            'Whole source voxels remain until their complete capacity threshold is reached.',
            'No acoustic, suction-pressure, irrigation, cavitation, heating, or tissue selectivity physics.']}
