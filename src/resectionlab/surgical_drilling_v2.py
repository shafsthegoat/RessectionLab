"""Opt-in uncalibrated work-limited bone drilling; SI energy, mm geometry.

The force-to-torque coefficient is an explicit engineering assumption, not a
patient fit. No density or CT calibration is inferred. Integrators must use
contact_limited_pose before apply, commit the limited pose and elapsed time even
when advanceAllowed is false, and retry requested travel after clearance. Never
roll back pending work as a collision. One thermal tick per simulated interval
is required (apply cools by default; shared steps use begin_step, apply with
cool=False for each drill, and end_step).
"""
from __future__ import annotations
import copy
import math
import numpy as np

MODEL = 'work-limited-drill-v2'
DEFAULT_PROFILE = {
    'model': MODEL, 'specificEnergyJPerMm3': 0.5,
    'tangentialForceFraction': 0.1, 'maxRpm': 80000., 'maxForceN': 20.,
    'maxPowerW': 30., 'maxRadiusMm': 10.,
    'heatFraction': 0.35, 'heatCapacityJPerMm3K': 0.002,
    'coolingTimeSeconds': 8., 'ambientTemperatureC': 37.,
    'doseThresholdC': 43.,
}


def validate(profile):
    if not isinstance(profile, dict) or profile.get('model') != MODEL:
        raise ValueError('DRILL_V2_PROFILE_REQUIRED')
    if set(profile) != set(DEFAULT_PROFILE):
        raise ValueError('COMPLETE_EXPLICIT_DRILL_V2_PROFILE_REQUIRED')
    for key, value in profile.items():
        if key == 'model': continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('INVALID_DRILL_PARAMETER:' + key)
        if key not in ('ambientTemperatureC', 'doseThresholdC') and value <= 0:
            raise ValueError('INVALID_DRILL_PARAMETER:' + key)
    if profile['heatFraction'] > 1 or profile['tangentialForceFraction'] > 1:
        raise ValueError('INVALID_DRILL_FRACTION')
    if profile['doseThresholdC'] < profile['ambientTemperatureC']:
        raise ValueError('DOSE_THRESHOLD_BELOW_AMBIENT')


def profile(engine):
    value = engine.scene.properties.get('tissuePhysicsV2', {}).get('bone')
    return value if isinstance(value, dict) and value.get('model') == MODEL else None


def initialize(engine):
    p = profile(engine)
    if p is None: return
    validate(p)
    engine.state.setdefault('drillingV2', {'workJ': {}, 'temperatureC': {},
        'thermalDoseCSeconds': {}, 'inputEnergyJ': 0., 'cuttingWorkJ': 0.,
        'heatGeneratedJ': 0., 'coolingLossJ': 0., 'unusedEnergyJ': 0.,
        'removedVolumeMm3': 0.})
    engine.state.setdefault('debris', {})


def _nonnegative(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError('INVALID_DRILL_INPUT:' + name)
    return float(value)


def _dose_integral(excess, equilibrium, threshold, tau, dt):
    """Exact integral of max(T-threshold,0) for constant power cooling."""
    def primitive(t):
        return (equilibrium - threshold) * t + (excess - equilibrium) * tau * (-math.expm1(-t / tau))
    end = equilibrium + (excess - equilibrium) * math.exp(-dt / tau)
    if min(excess, end) >= threshold: return max(0., primitive(dt))
    if max(excess, end) <= threshold: return 0.
    crossing = -tau * math.log((threshold - equilibrium) / (excess - equilibrium))
    return max(0., primitive(crossing) if excess > threshold else primitive(dt) - primitive(crossing))


def _thermal(engine, dt, powers):
    p = profile(engine); state = engine.state['drillingV2']
    capacity = p['heatCapacityJPerMm3K'] * engine.scene.voxel_mm3
    tau = p['coolingTimeSeconds']; ambient = p['ambientTemperatureC']
    decay = math.exp(-dt / tau)
    for key in sorted(set(state['temperatureC']) | set(powers), key=int):
        old = state['temperatureC'].get(key, ambient) - ambient
        power = powers.get(key, 0.)
        equilibrium = power * tau / capacity
        new = old * decay + equilibrium * (-math.expm1(-dt / tau))
        state['temperatureC'][key] = ambient + new
        state['thermalDoseCSeconds'][key] = state['thermalDoseCSeconds'].get(key, 0.) + _dose_integral(old, equilibrium, p['doseThresholdC'] - ambient, tau, dt)
        state['coolingLossJ'] += max(0., power * dt - capacity * (new - old))


def cool(engine, dt):
    """Advance all stored temperatures during idle time; never removes tissue."""
    dt = _nonnegative(dt, 'dt')
    if profile(engine) is None: return
    initialize(engine); _thermal(engine, dt, {})


def begin_step(engine):
    """Start one shared thermal interval for all simultaneous drills."""
    engine._drill_step_powers = {}


def end_step(engine, dt):
    """Integrate all queued powers and idle cooling exactly once for dt.

    Every apply(cool=False) in this interval must use this same dt. Queued
    powers are transient engine data, not serialized patient state.
    """
    dt = _nonnegative(dt, 'dt')
    if profile(engine) is not None:
        initialize(engine)
        _thermal(engine, dt, getattr(engine, '_drill_step_powers', {}))
    engine._drill_step_powers = {}


def apply(engine, cells, dt, params, *, cool=True):
    """Spend one total power budget across actual frontier contact cells.

    cells must describe only reachable instantaneous tip contact, never the
    complete requested swept path. Equal physical cell volumes give equal work
    shares; adding contacts cannot multiply power. Work persists until a whole
    voxel has been eroded. That conservative quantization releases exactly one
    voxel of debris on clearance. Excess budget is unused, not spent at depth.
    """
    p = profile(engine)
    if p is None: raise ValueError('DRILL_V2_PROFILE_REQUIRED')
    validate(p)
    dt = _nonnegative(dt, 'dt')
    rpm = min(_nonnegative(params.get('rpm', 0.), 'rpm'), p['maxRpm'])
    force = min(_nonnegative(params.get('forceN', 0.), 'forceN'), p['maxForceN'])
    radius = min(_nonnegative(params.get('radiusMm', 0.), 'radiusMm'), p['maxRadiusMm'])
    duty = min(_nonnegative(params.get('duty', 1.), 'duty'), 1.)
    ids = sorted(set(int(i) for i in cells))
    labels = engine.scene.labels.ravel()
    if any(i < 0 or i >= len(labels) for i in ids): raise ValueError('INVALID_DRILL_CELL')
    removed = set(engine.state['removed'])
    ids = [i for i in ids if i not in removed and labels[i] and engine.by_label[int(labels[i])]['material'] == 'bone']
    initialize(engine); state = engine.state['drillingV2']
    power = min(p['maxPowerW'], force * p['tangentialForceFraction'] * radius * .001 * rpm * (2 * math.pi / 60)) * duty
    budget = power * dt if ids else 0.
    required = p['specificEnergyJPerMm3'] * engine.scene.voxel_mm3
    new = []; spent = 0.; powers = {}
    for cell in ids:
        key = str(cell); old = state['workJ'].get(key, 0.)
        work = min(max(0., required - old), budget / len(ids))
        state['workJ'][key] = old + work; spent += work
        powers[key] = work * p['heatFraction'] / dt if dt else 0.
        if state['workJ'][key] >= required:
            new.append(cell); del state['workJ'][key]
            engine.state['debris'][key] = engine.state['debris'].get(key, 0.) + engine.scene.voxel_mm3
    # Heat is a partition of cutting work, not additional input energy.
    state['inputEnergyJ'] += budget; state['cuttingWorkJ'] += spent
    state['heatGeneratedJ'] += spent * p['heatFraction']
    state['unusedEnergyJ'] += budget - spent
    state['removedVolumeMm3'] += len(new) * engine.scene.voxel_mm3
    if cool: _thermal(engine, dt, powers)
    else:
        # Assign rather than mutate: checkpoint clones cannot share an ephemeral
        # accumulator by accident. Sum simultaneous sources before integrating.
        accumulated = dict(getattr(engine, '_drill_step_powers', {}))
        for key, power_i in powers.items():
            accumulated[key] = accumulated.get(key, 0.) + power_i
        engine._drill_step_powers = accumulated
    engine.state['removed'] = sorted(removed | set(new))
    engine.state['debrisMm3'] = sum(engine.state['debris'].values())
    if new: engine.state['boneTopologyDirty'] = True
    blocked = sorted(set(ids) - set(new))
    contact = {'blockedCells': blocked, 'advanceAllowed': not blocked, 'consumedSeconds': dt,
               'contactVolumeMm3': len(ids) * engine.scene.voxel_mm3, 'powerW': power if ids else 0.,
               'rpm': rpm, 'forceN': force, 'duty': duty}
    events = [{'kind': 'drill_work', 'inputEnergyJ': budget, 'cuttingWorkJ': spent,
               'removedCells': new, 'removedVolumeMm3': len(new) * engine.scene.voxel_mm3,
               'pendingCells': len(blocked)}] if ids else []
    return events, new, contact


def contact_limited_pose(engine, previous, requested, radius):
    """Return reachable position and first-contact cells; do not mutate state.

    A center-based voxel contact approximation using engine.touched. Existing
    contact blocks forward travel until cleared; withdrawal is left to the
    integrator's withdrawal validation. Bisection localizes first contact, and
    stepping at <= min spacing/4 prevents skipping a voxel on a long path.
    """
    previous = np.asarray(previous, dtype=float); requested = np.asarray(requested, dtype=float)
    def bone_at(point):
        labels = engine.scene.labels.ravel()
        return sorted(i for i in engine.touched(point, point, radius)
                      if labels[i] and engine.by_label[int(labels[i])]['material'] == 'bone')
    initial = bone_at(previous)
    if initial: return previous.copy(), initial
    distance = float(np.linalg.norm(requested - previous))
    steps = max(1, math.ceil(distance / (float(engine.scene.spacing.min()) / 4)))
    low = previous.copy()
    for index in range(1, steps + 1):
        high = previous + (requested - previous) * (index / steps)
        contacts = bone_at(high)
        if contacts:
            for _ in range(32):
                mid = (low + high) / 2
                if bone_at(mid): high = mid
                else: low = mid
            return high, bone_at(high)
        low = high
    return requested.copy(), []


def descriptor(engine):
    return {'enabled': profile(engine) is not None, 'profile': copy.deepcopy(profile(engine)),
            'calibration': 'assumed-research; baseline uncalibrated; no CT density inference',
            'model': MODEL, 'thermalDoseUnits': 'degree-C seconds above declared threshold',
            'limits': ['voxel-quantized removal', 'assumed force-to-torque relation',
                       'lumped per-cell cooling; no conduction or irrigation coupling',
                       'thermal dose is an exposure proxy, not validated necrosis']}


def metrics(engine):
    """Read-only snapshot, including a checkable energy partition."""
    state = engine.state.get('drillingV2', {})
    return {'drillInputEnergyJ': state.get('inputEnergyJ', 0.),
            'drillCuttingWorkJ': state.get('cuttingWorkJ', 0.),
            'drillUnusedEnergyJ': state.get('unusedEnergyJ', 0.),
            'drillHeatGeneratedJ': state.get('heatGeneratedJ', 0.),
            'drillCoolingLossJ': state.get('coolingLossJ', 0.),
            'drillPendingWorkJ': sum(state.get('workJ', {}).values()),
            'drillRemovedVolumeMm3': state.get('removedVolumeMm3', 0.),
            'drillMaxTemperatureC': max(state.get('temperatureC', {}).values(), default=(profile(engine) or DEFAULT_PROFILE)['ambientTemperatureC'])}
