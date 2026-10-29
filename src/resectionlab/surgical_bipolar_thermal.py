"""Opt-in source-owned passive heat conduction, not coagulation or injury.

The electrical problem is not solved. An explicit assumed watts-per-activation
coefficient supplies total deposited heat at the existing electrode contacts.
Reference orthogonal voxel faces conduct heat; cut/removed faces cannot bridge
a cavity. Backward Euler accounts for storage, bath exchange and removed heat.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

PROPERTY = 'bipolarThermalV1'
SCHEMA = 'resectionlab.bipolar-thermal.v1'
SOFT = {'soft_tissue', 'tumor', 'vessel', 'membrane'}


def research_profile(cells, materials, *, ambient_c, watts_per_activation,
                     maximum_c, retained_boundary='insulated', exposed_transfer=0.):
    return {'schema': SCHEMA, 'sourceCells': list(cells), 'materials': materials,
            'ambientTemperatureC': ambient_c, 'heatWattsPerActivation': watts_per_activation,
            'maximumTemperatureC': maximum_c, 'retainedPatchBoundary': retained_boundary,
            'exposedHeatTransferWPerMm2K': exposed_transfer,
            'provenance': {'kind': 'assumed-research', 'description':
                'Explicit delivered heat, reference-grid conduction and bath assumptions; '
                'not calibrated device, patient, coagulation or injury response.'}}


def _number(value, name, low=0., high=1e8):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('INVALID_BIPOLAR_THERMAL_' + name)
    return float(value)


def validate(scene):
    if PROPERTY not in scene.properties: return None
    p = scene.properties.get(PROPERTY)
    if 'tissuePhysicsV2' not in scene.properties:
        raise ValueError('BIPOLAR_THERMAL_REQUIRES_EXPLICIT_V2')
    expected = research_profile([], {}, ambient_c=0., watts_per_activation=0., maximum_c=1.)
    if not isinstance(p, dict) or set(p) != set(expected) or p['schema'] != SCHEMA:
        raise ValueError('COMPLETE_EXPLICIT_BIPOLAR_THERMAL_PROFILE_REQUIRED')
    provenance = p['provenance']
    if (not isinstance(provenance, dict) or provenance.get('kind') != 'assumed-research'
            or not isinstance(provenance.get('description'), str) or not provenance['description'].strip()):
        raise ValueError('EXPLICIT_BIPOLAR_THERMAL_PROVENANCE_REQUIRED')
    cells = p['sourceCells']
    if (not isinstance(cells, list) or not 1 <= len(cells) <= 4096
            or any(type(c) is not int or not 0 <= c < scene.labels.size for c in cells)
            or cells != sorted(set(cells))):
        raise ValueError('BIPOLAR_THERMAL_SORTED_SOURCE_PATCH_REQUIRED')
    regions = {r['id']: r for r in scene.regions}
    labels = {int(scene.labels.flat[c]) for c in cells}
    if any(label not in regions or regions[label]['material'] not in SOFT for label in labels):
        raise ValueError('BIPOLAR_THERMAL_PATCH_REQUIRES_DECLARED_TISSUE')
    if not isinstance(p['materials'], dict) or set(p['materials']) != {str(v) for v in labels}:
        raise ValueError('BIPOLAR_THERMAL_EVERY_PATCH_MATERIAL_REQUIRED')
    for m in p['materials'].values():
        if not isinstance(m, dict) or set(m) != {'volumetricHeatCapacityJPerMm3K', 'conductivityWPerMmK', 'source'}:
            raise ValueError('COMPLETE_BIPOLAR_THERMAL_MATERIAL_REQUIRED')
        _number(m['volumetricHeatCapacityJPerMm3K'], 'CAPACITY', 1e-9, 1.)
        _number(m['conductivityWPerMmK'], 'CONDUCTIVITY', 1e-12, 1.)
        if not isinstance(m['source'], str) or not m['source'].strip():
            raise ValueError('BIPOLAR_THERMAL_MATERIAL_SOURCE_REQUIRED')
    ambient = _number(p['ambientTemperatureC'], 'AMBIENT', 0., 99.)
    _number(p['maximumTemperatureC'], 'MAXIMUM', ambient + .001, 99.)
    _number(p['heatWattsPerActivation'], 'HEAT_POWER', 0., 100.)
    _number(p['exposedHeatTransferWPerMm2K'], 'SURFACE_TRANSFER', 0., 1.)
    if p['retainedPatchBoundary'] not in ('insulated', 'ambient-at-patch-face'):
        raise ValueError('BIPOLAR_THERMAL_BOUNDARY_REQUIRED')
    a = scene.affine[:3, :3]
    gram = a.T @ a
    if not np.allclose(gram, np.diag(np.diag(gram)), rtol=0, atol=1e-10 * np.diag(gram).max()):
        raise ValueError('BIPOLAR_THERMAL_ORTHOGONAL_GRID_REQUIRED')
    return p


def initialize(engine):
    if PROPERTY not in engine.scene.properties: return None
    p = validate(engine.scene)
    return engine.state.setdefault(PROPERTY, {'schema': SCHEMA, 'temperatureRiseK': {},
        'excessTemperatureIntegralKSeconds': {}, 'inputEnergyJ': 0., 'storedHeatJ': 0.,
        'bathExportJ': 0., 'removedHeatExportJ': 0., 'peakTemperatureC': p['ambientTemperatureC'],
        'energyResidualJ': 0., 'elapsedSeconds': 0.})


def begin_step(engine):
    if PROPERTY in engine.scene.properties:
        engine._bipolar_step_powers = {}


def deposit(engine, cells, activation):
    if PROPERTY not in engine.scene.properties: return []
    activation = _number(activation, 'ACTIVATION', 0., 10.)
    powers = getattr(engine, '_bipolar_step_powers', None)
    if not isinstance(powers, dict): raise ValueError('BIPOLAR_THERMAL_INTERVAL_REQUIRED')
    cells = sorted(set(map(int, cells)))
    if not cells or activation == 0: return []
    p = engine.scene.properties[PROPERTY]
    if not set(cells) <= set(p['sourceCells']):
        raise ValueError('BIPOLAR_THERMAL_COVERAGE_UNKNOWN')
    if set(cells).intersection(engine.state['removed']):
        raise ValueError('BIPOLAR_THERMAL_CONTACT_REMOVED_WITHIN_INTERVAL')
    power = p['heatWattsPerActivation'] * activation
    for cell in cells: powers[cell] = powers.get(cell, 0.) + power / len(cells)
    return [{'kind': 'bipolar_heat_input', 'deliveredHeatWatts': power,
             'sourceContactCells': cells, 'calibration': 'assumed watts per activation; not generator power'}]


def _operator(engine, active, blocked, dt):
    scene = engine.scene; p = scene.properties[PROPERTY]
    scene.require_cache_source()
    # Cache only immutable geometry/linear factors. Clones share no mutable fields.
    signature = (scene.identity, tuple(active), tuple(engine.state['removed']), tuple(sorted(blocked)), dt)
    cached = getattr(engine, '_bipolar_thermal_operator', None)
    if cached is not None and cached[0] == signature: return cached[1]
    index = {c: i for i, c in enumerate(active)}
    removed = set(engine.state['removed'])
    spacing = scene.spacing
    areas = scene.voxel_mm3 / spacing
    conductivity = {}; capacity = np.empty(len(active))
    for c, i in index.items():
        m = p['materials'][str(int(scene.labels.flat[c]))]
        conductivity[c] = m['conductivityWPerMmK']
        capacity[i] = m['volumetricHeatCapacityJPerMm3K'] * scene.voxel_mm3
    bath = np.zeros(len(active)); diagonal = np.zeros(len(active)); rows = []; cols = []; values = []
    for c, i in index.items():
        ijk = np.array(np.unravel_index(c, scene.shape))
        for axis in range(3):
            for sign in (-1, 1):
                j = ijk.copy(); j[axis] += sign
                in_grid = np.all(j >= 0) and np.all(j < scene.shape)
                other = int(np.ravel_multi_index(j, scene.shape)) if in_grid else None
                retained = other is not None and other not in removed and bool(scene.labels.flat[other])
                cut = retained and tuple(sorted((c, other))) in blocked
                if other in index and not cut:
                    if c < other:
                        k1, k2 = conductivity[c], conductivity[other]
                        g = 2 * k1 * k2 / (k1 + k2) * areas[axis] / spacing[axis]
                        n = index[other]; diagonal[i] += g; diagonal[n] += g
                        rows.extend((i, n)); cols.extend((n, i)); values.extend((-g, -g))
                elif retained and not cut and p['retainedPatchBoundary'] == 'ambient-at-patch-face':
                    bath[i] += 2 * conductivity[c] * areas[axis] / spacing[axis]
                elif not retained or cut:
                    bath[i] += p['exposedHeatTransferWPerMm2K'] * areas[axis]
    diagonal += bath
    rows.extend(range(len(active))); cols.extend(range(len(active))); values.extend(diagonal)
    laplacian = sparse.coo_matrix((values, (rows, cols)), shape=(len(active), len(active))).tocsc()
    solve = factorized(laplacian + sparse.diags(capacity / dt, format='csc'))
    result = (capacity, bath, solve)
    engine._bipolar_thermal_operator = (signature, result)
    return result


def end_step(engine, dt):
    if PROPERTY not in engine.scene.properties: return
    dt = _number(dt, 'DURATION', 1e-9, 60.)
    powers = getattr(engine, '_bipolar_step_powers', None)
    if not isinstance(powers, dict): raise ValueError('BIPOLAR_THERMAL_INTERVAL_REQUIRED')
    p = engine.scene.properties[PROPERTY]; state = engine.state[PROPERTY]
    removed = set(engine.state['removed'])
    if set(powers).intersection(removed):
        raise ValueError('BIPOLAR_THERMAL_CONTACT_REMOVED_WITHIN_INTERVAL')
    rise = state['temperatureRiseK']
    for key in list(rise):
        if int(key) in removed:
            m = p['materials'][str(int(engine.scene.labels.flat[int(key)]))]
            state['removedHeatExportJ'] += m['volumetricHeatCapacityJPerMm3K'] * engine.scene.voxel_mm3 * rise.pop(key)
    active = [c for c in p['sourceCells'] if c not in removed]
    if active:
        blocked = {tuple(sorted(e)) for field in ('cutEdges', 'transectedEdges') for e in engine.state.get(field, [])}
        capacity, bath, solve = _operator(engine, active, blocked, dt)
        old = np.array([rise.get(str(c), 0.) for c in active])
        if not np.isfinite(old).all() or np.any(old < -1e-10):
            raise ValueError('INVALID_BIPOLAR_THERMAL_STATE')
        power = np.array([powers.get(c, 0.) for c in active])
        new = solve(capacity / dt * old + power)
        if not np.isfinite(new).all() or np.any(new < -1e-10):
            raise ValueError('BIPOLAR_THERMAL_SOLVE_FAILED')
        maximum = p['ambientTemperatureC'] + float(np.max(new))
        if maximum > p['maximumTemperatureC'] + 1e-10:
            raise ValueError('BIPOLAR_THERMAL_TEMPERATURE_DOMAIN: no clipping or phase-change model')
        state['temperatureRiseK'] = {str(c): float(v) for c, v in zip(active, new) if v != 0.}
        for c, value in zip(active, new):
            key = str(c)
            state['excessTemperatureIntegralKSeconds'][key] = state['excessTemperatureIntegralKSeconds'].get(key, 0.) + max(0., float(value)) * dt
        state['inputEnergyJ'] += float(power.sum()) * dt
        state['bathExportJ'] += float(bath @ new) * dt
        state['storedHeatJ'] = float(capacity @ new)
        state['peakTemperatureC'] = max(state['peakTemperatureC'], maximum)
    else:
        state['storedHeatJ'] = 0.
    state['energyResidualJ'] = state['inputEnergyJ'] - state['storedHeatJ'] - state['bathExportJ'] - state['removedHeatExportJ']
    if abs(state['energyResidualJ']) > 1e-10 * max(1., state['inputEnergyJ']):
        raise ValueError('BIPOLAR_THERMAL_ENERGY_RESIDUAL')
    state['elapsedSeconds'] += dt
    engine._bipolar_step_powers = None


def metrics(engine):
    if PROPERTY not in engine.scene.properties: return {}
    s = engine.state[PROPERTY]; p = engine.scene.properties[PROPERTY]
    return {'bipolarHeatInputJ': s['inputEnergyJ'], 'bipolarHeatStoredJ': s['storedHeatJ'],
            'bipolarHeatBathExportJ': s['bathExportJ'], 'bipolarHeatRemovedJ': s['removedHeatExportJ'],
            'bipolarHeatResidualJ': s['energyResidualJ'], 'bipolarPeakTemperatureC': s['peakTemperatureC'],
            'bipolarMaxTemperatureC': p['ambientTemperatureC'] + max(s['temperatureRiseK'].values(), default=0.),
            'bipolarExcessTemperatureVolumeTimeMm3KSeconds': sum(s['excessTemperatureIntegralKSeconds'].values()) * engine.scene.voxel_mm3}


def descriptor(engine):
    if PROPERTY not in engine.scene.properties: return None
    return {'model': SCHEMA, 'profile': copy.deepcopy(engine.scene.properties[PROPERTY]),
            'summary': metrics(engine), 'limits': [
                'Delivered watts and deposition are assumptions, not an electrical current or generator calibration.',
                'Constant coefficients on reference orthogonal cells; no deformed geometry, perfusion, evaporation, irrigation or electrode cooling.',
                'Partial explicit patch; missing thermal coverage refuses heating, not a safe empty region.',
                'Cuts and cavity faces do not conduct; optional declared bath exchange is separately budgeted.',
                'Backward-Euler temperature exposure is numerical, not tissue injury or a vessel sealing/patency law.']}
