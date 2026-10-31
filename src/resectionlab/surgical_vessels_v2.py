"""Opt-in local compliant wound model, with explicitly assumed boundary conditions.

This is an engineering surrogate, not inferred patient hemodynamics. A single
finite compliant vascular compartment feeds clustered grid-face wounds. Leak
conductance is proportional to physical face area, squared injury severity and
remaining unclotted fraction. This avoids a fixed outlet flow per voxel edge.
Pressure is declared pressurePa * remaining/capacity. Refill comes from a declared
constant-pressure external supply and is counted separately. No anatomical
territory, vessel diameter, pulsatility or collateral perfusion is inferred.

Call tick exactly once immediately before surgical_effects.tick_fields with the
same dt. tick reserves vascular volume and sets interval-average source rates;
tick_fields alone emits it and updates emittedMm3/fluidAddedMm3. A subsequent
tick checks this handoff, detecting skipped or duplicated field emission.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from scipy.integrate import solve_ivp

MODEL = 'compliant-wounds-v2'
PARAMETERS = ('pressurePa', 'reservoirMm3', 'refillConductanceMm3PerPaS',
              'leakConductanceMmPerPaS', 'mechanicalDoseNs', 'thermalDoseCs',
              'coagulationDose', 'clotRatePerSecond')


def validate(profile):
    if not isinstance(profile, dict) or set(profile) != {'model', *PARAMETERS} or profile['model'] != MODEL:
        raise ValueError('COMPLETE_EXPLICIT_V2_VESSEL_PROFILE_REQUIRED')
    for name in PARAMETERS:
        value = profile[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('INVALID_V2_VESSEL_PARAMETER:' + name)
        if name in ('reservoirMm3', 'mechanicalDoseNs', 'thermalDoseCs', 'coagulationDose') and value == 0:
            raise ValueError('INVALID_V2_VESSEL_PARAMETER:' + name)
    # Reject declarations whose combined rates cannot be represented safely.
    rate = profile['pressurePa'] * (profile['refillConductanceMm3PerPaS'] + profile['leakConductanceMmPerPaS']) / profile['reservoirMm3']
    if not math.isfinite(rate) or rate > 1e6:
        raise ValueError('V2_VESSEL_RATE_OUTSIDE_NUMERICAL_DOMAIN')


def _profile(engine):
    return engine.scene.properties.get('tissuePhysicsV2', {}).get('vessel')


def initialize(engine):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine): return graph.initialize(engine)
    p = _profile(engine)
    if p is None: return None
    validate(p)
    return engine.state.setdefault('vesselsV2', {
        'faces': {}, 'wounds': {}, 'contactDose': {}, 'nextWound': 0,
        'remainingMm3': float(p['reservoirMm3']), 'refilledMm3': 0.,
        'generatedMm3': 0., 'initialMm3': float(p['reservoirMm3']),
        'elapsedSeconds': 0., 'pressurePa': float(p['pressurePa'])})


def _positive(value, name, zero=True):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or not zero and value == 0:
        raise ValueError('INVALID_V2_VESSEL_' + name)
    return float(value)


def _vessel(engine, cell):
    if type(cell) not in (int, np.int64, np.int32) or not 0 <= cell < engine.scene.labels.size:
        raise ValueError('INVALID_V2_VESSEL_CELL')
    label = int(engine.scene.labels.ravel()[cell])
    return bool(label and engine.by_label[label]['material'] == 'vessel')


def _face(engine, a, b):
    a, b = sorted((int(a), int(b)))
    if not _vessel(engine, a) or not _vessel(engine, b): return None
    ia, ib = (np.array(np.unravel_index(i, engine.scene.shape)) for i in (a, b))
    delta = np.abs(ia - ib)
    if delta.sum() != 1: raise ValueError('V2_WOUND_REQUIRES_NEIGHBOR_FACE')
    axis = int(np.argmax(delta)); other = [i for i in range(3) if i != axis]
    basis = engine.scene.affine[:3, :3]
    area = float(np.linalg.norm(np.cross(basis[:, other[0]], basis[:, other[1]])))
    if not math.isfinite(area) or area <= 0: raise ValueError('INVALID_V2_WOUND_FACE_AREA')
    return f'{a}:{b}', {'cells': [a, b], 'areaMm2': area, 'axis': axis,
                         'damage': 0., 'patent': True, 'clot': 0., 'coagulationDose': 0.}


def _neighbors(engine, cell):
    ijk = np.array(np.unravel_index(cell, engine.scene.shape))
    for axis in range(3):
        for step in (-1, 1):
            j = ijk.copy(); j[axis] += step
            if np.all(j >= 0) and np.all(j < engine.scene.shape):
                yield int(np.ravel_multi_index(j, engine.scene.shape))


def _recluster(engine):
    """Only revisit damaged faces; connected local faces share one fluid outlet.

    Faces sharing endpoints or adjacent endpoints are clustered. This deliberately
    treats unresolved adjacent injured vessels as one local wound, not evidence
    that they share anatomical perfusion. Face area is still counted exactly once.
    Existing emitted blood/reservoirs are transferred, never discarded on merge.
    """
    state = engine.state['vesselsV2']; faces = state['faces']
    keys = sorted(k for k, f in faces.items() if f['damage'] > 0)
    if not keys: return
    parents = {key: key for key in keys}; cells = {}
    def root(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]; k = parents[k]
        return k
    for key in keys:
        for cell in faces[key]['cells']:
            for neighbor in [cell, *_neighbors(engine, cell)]:
                for other in cells.get(neighbor, []):
                    a, b = sorted((root(key), root(other))); parents[b] = a
            cells.setdefault(cell, []).append(key)
    groups = {}
    for key in keys: groups.setdefault(root(key), []).append(key)
    old = state['wounds']; sources = {s['id']: s for s in engine.state['sources'] if s.get('modelVersion') == 2}
    wounds = {}; new_sources = []
    for members in groups.values():
        prior = sorted({faces[k]['woundId'] for k in members if faces[k].get('woundId') in old})
        if prior:
            identifier = prior[0]; source = sources[identifier]
            for other in prior[1:]:
                for field in ('emittedMm3', 'reservoirMm3', 'expectedEmittedMm3'):
                    source[field] += sources[other].get(field, 0.)
        else:
            identifier = 'vessel-v2:' + str(state['nextWound']); state['nextWound'] += 1
            source = {'id': identifier, 'kind': 'modeled-vessel-wound', 'modelVersion': 2,
                      'rateMm3PerSecond': 0., 'emittedMm3': 0., 'expectedEmittedMm3': 0.,
                      'reservoirMm3': 0., 'coagulationExposure': _profile(engine)['coagulationDose'],
                      'exposure': 0., 'bleedingActive': True, 'patent': False}
        source['cell'] = min(c for k in members for c in faces[k]['cells'])
        source['patent'] = all(faces[k]['patent'] for k in members)
        for key in members: faces[key]['woundId'] = identifier
        wounds[identifier] = {'faces': members, 'sourceId': identifier}
        new_sources.append(source)
    engine.state['sources'] = [s for s in engine.state['sources'] if s.get('modelVersion') != 2] + new_sources
    state['wounds'] = wounds


def damage(engine, edges, dt=0., kind='transection'):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine): return graph.damage(engine, edges, dt, kind)
    _positive(dt, 'DT')
    if kind not in ('transection', 'incision', 'ablation', 'drill'): raise ValueError('UNKNOWN_V2_VESSEL_DAMAGE_KIND')
    state = initialize(engine)
    if state is None: return []
    changed = []
    # Validate all supplied geometry before mutation.
    supplied = [_face(engine, *edge) for edge in sorted(set(tuple(sorted(map(int, e))) for e in edges))]
    for item in supplied:
        if item is None: continue
        key, new = item; face = state['faces'].setdefault(key, new)
        if face['damage'] < 1.: changed.append(key)
        face['damage'] = 1.; face['patent'] = False
        # Repeated exposure to the same cut never restores patency or clears clot.
    if changed: _recluster(engine)
    return changed


def contact(engine, cells, dt, force_n, temperature=None):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine): return graph.contact(engine, cells, dt, force_n, temperature)
    dt = _positive(dt, 'DT'); force_n = _positive(force_n, 'FORCE')
    if temperature is not None: _positive(temperature, 'TEMPERATURE')
    state = initialize(engine)
    if state is None or dt == 0: return []
    p = _profile(engine); touched = sorted(set(map(int, cells)))
    touched = [i for i in touched if _vessel(engine, i)]
    if not touched: return []
    # Force is total tool force, shared over the supplied contact footprint.
    # Dose density uses physical face area, keeping planar refinement comparable.
    basis = engine.scene.affine[:3, :3]
    areas = [float(np.linalg.norm(np.cross(basis[:, i], basis[:, j]))) for i, j in ((0,1),(0,2),(1,2))]
    footprint = len(touched) * min(areas)
    mechanical = force_n * dt / footprint / p['mechanicalDoseNs']
    thermal = max(0., (temperature or 0.) - 43.) * dt / p['thermalDoseCs']
    changed = []
    for cell in touched:
        key = str(cell); dose = state['contactDose'].setdefault(key, {'mechanical': 0., 'thermal': 0.})
        dose['mechanical'] = min(2., dose['mechanical'] + mechanical)
        dose['thermal'] = min(2., dose['thermal'] + thermal)
        severity = min(1., max(0., dose['mechanical'] + dose['thermal'] - 1.))
        if severity == 0: continue
        for other in _neighbors(engine, cell):
            item = _face(engine, cell, other)
            if item is None: continue
            edge, new = item; face = state['faces'].setdefault(edge, new)
            if severity > face['damage']:
                face['damage'] = severity; face['patent'] = severity < 1.; changed.append(edge)
    if changed: _recluster(engine)
    return sorted(set(changed))


def coagulate(engine, cells, power, dt):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine): return graph.coagulate(engine, cells, power, dt)
    power = _positive(power, 'POWER'); dt = _positive(dt, 'DT')
    state = initialize(engine)
    if state is None: return []
    touched = set(map(int, cells)); p = _profile(engine); changed = []
    for key, face in state['faces'].items():
        if touched.intersection(face['cells']):
            face['coagulationDose'] = min(p['coagulationDose'], face['coagulationDose'] + power * dt)
            # Persistent sealing progresses with integrated energy exposure.
            face['clot'] = max(face['clot'], min(1., face['coagulationDose'] / p['coagulationDose']))
            changed.append(key)
    return changed


def tick(engine, dt):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine):
        graph.refresh_compression(engine)
        return graph.tick(engine, dt)
    dt = _positive(dt, 'DT')
    state = initialize(engine)
    if state is None: return
    p = _profile(engine)
    sources = [s for s in engine.state['sources'] if s.get('modelVersion') == 2]
    for source in sources:
        if not math.isclose(source['emittedMm3'], source['expectedEmittedMm3'], rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError('V2_VESSEL_TICK_REQUIRES_MATCHING_FIELD_EMISSION')
        source['rateMm3PerSecond'] = 0.
    if dt == 0: return
    conductances = {}
    for identifier, wound in state['wounds'].items():
        conductances[identifier] = p['leakConductanceMmPerPaS'] * sum(
            f['areaMm2'] * f['damage'] ** 2 * (1. - f['clot'])
            for f in (state['faces'][k] for k in wound['faces']))
    total = sum(conductances.values())
    if not math.isfinite(total): raise ValueError('V2_WOUND_CONDUCTANCE_OVERFLOW')
    capacity = p['reservoirMm3']; pressure = p['pressurePa']
    refill_rate = p['refillConductanceMm3PerPaS'] * pressure
    clot_rate = p['clotRatePerSecond']
    initial = state['remainingMm3']
    if total == 0:
        remaining = capacity - (capacity - initial) * math.exp(-refill_rate * dt / capacity)
        refill = remaining - initial; generated = 0.
    else:
        # Integrate the compartment and two volume ledgers together. The scalar
        # linear pressure law and exponentially closing conductance stay smooth
        # across action partitions; tight tolerances make timing independent of
        # microstep count to floating-point integration tolerance.
        def rhs(t, y):
            fraction = min(1., max(0., y[0] / capacity))
            inflow = refill_rate * (1. - fraction)
            leak = total * math.exp(-clot_rate * t) * pressure * fraction
            return [inflow - leak, inflow, leak]
        result = solve_ivp(rhs, (0., dt), [initial, 0., 0.], rtol=1e-10, atol=1e-11, method='LSODA')
        if not result.success or not np.isfinite(result.y[:, -1]).all():
            raise ValueError('V2_VESSEL_INTEGRATION_FAILED')
        remaining, refill, generated = map(float, result.y[:, -1])
        remaining = min(capacity, max(0., remaining)); refill = max(0., refill)
        # Enforce the conservative ledger after roundoff projection.
        generated = max(0., initial + refill - remaining)
    state['remainingMm3'] = remaining; state['refilledMm3'] += refill
    state['generatedMm3'] += generated; state['elapsedSeconds'] += dt
    state['pressurePa'] = pressure * remaining / capacity
    for face in state['faces'].values():
        face['clot'] = 1. - (1. - face['clot']) * math.exp(-clot_rate * dt)
    for source in sources:
        amount = generated * conductances[source['id']] / total if total else 0.
        source['rateMm3PerSecond'] = amount / dt
        source['expectedEmittedMm3'] += amount
        source['bleedingActive'] = amount > 0


def descriptor(engine):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine): return graph.descriptor(engine)
    p = _profile(engine)
    return {'enabled': p is not None, 'model': MODEL, 'profile': copy.deepcopy(p),
            'patientCalibratedPhysiology': False, 'globalPerfusion': 'unknown',
            'assumptions': ['one finite compliant compartment with declared external refill',
                'unique affine voxel-face area; adjacent wounds share an outlet',
                'damage squared scales conductance; first-order persistent clot',
                'contact mechanical dose threshold is N s per square mm; thermal threshold is degree C s above 43 C'],
            'limits': ['not an anatomical vessel network', 'grid-face staircase surface approximation',
                       'no patient pressure, coagulation, vessel caliber or blood loss calibration']}


def metrics(engine):
    from . import surgical_vascular_graph as graph
    if graph.enabled(engine): return graph.metrics(engine)
    state = engine.state.get('vesselsV2')
    if state is None: return {'enabled': False}
    sources = [s for s in engine.state['sources'] if s.get('modelVersion') == 2]
    return {'enabled': True, 'woundCount': len(state['wounds']), 'damagedFaceCount': len(state['faces']),
            'woundAreaMm2': sum(f['areaMm2'] for f in state['faces'].values()),
            'activeWounds': sum(s['bleedingActive'] for s in sources),
            'pressurePa': state['pressurePa'], 'vascularRemainingMm3': state['remainingMm3'],
            'refilledMm3': state['refilledMm3'], 'generatedMm3': state['generatedMm3'],
            'emittedMm3': sum(s['emittedMm3'] for s in sources),
            'blockedOutletReservoirMm3': sum(s['reservoirMm3'] for s in sources),
            'conservationResidualMm3': state['initialMm3'] + state['refilledMm3'] - state['remainingMm3'] - state['generatedMm3'],
            'globalPerfusion': 'unknown'}
