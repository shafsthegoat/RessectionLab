"""Opt-in patient posture and bounded gravity surrogates in physical RAS.

Version 1 is an uncalibrated engineering model, never a camera transform or a
prediction of ICP, perfusion, venous patency, or clinical positioning safety.
An absent ``patientPosition`` key leaves earlier episode mechanics unchanged.
"""
from __future__ import annotations

from collections import deque
import copy
from itertools import islice
import math
import numpy as np


GRAVITY_M_PER_S2 = 9.80665
PA_PER_MMHG = 133.3224
MODEL_VERSION = 1
# Roll about superior in the neutral supine room frame; positive left side down.
PRESETS = {
    'supine': (0., 0.), 'prone': (180., 0.),
    'left_lateral': (90., 0.), 'right_lateral': (-90., 0.),
    'park_bench': (110., 0.), 'semisitting': (0., 60.),
}
SOURCES = [
    {'id': 'position-physiology-1987', 'url': 'https://pubmed.ncbi.nlm.nih.gov/3591477/',
     'scope': 'Physiological parameters varied with posture in a posterior-fossa surgical cohort; no coefficients imported.'},
    {'id': 'neck-position-2000', 'url': 'https://pubmed.ncbi.nlm.nih.gov/10636614/',
     'scope': 'Small anesthetized surgical cohort observed ICP changes with head/neck and table position; qualitative only.'},
    {'id': 'gravity-icp-2017', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5350445/',
     'scope': 'Measured posture-related ICP and cardiovascular changes; hydrostatic reference height matters; no patient inference.'},
    {'id': 'venous-pressure-2020', 'url': 'https://pubmed.ncbi.nlm.nih.gov/33222920/',
     'scope': 'Semi-sitting surgical cohort associated subatmospheric jugular bulb pressure with air embolism; qualitative context only.'},
    {'id': 'nist-pressure-units', 'url': 'https://www.nist.gov/document/00-20-h133-apdxefinal-17pdf',
     'scope': 'Pressure conversion: 1 mmHg = 133.3224 Pa.'},
]
_SOFT = ('soft_tissue', 'brain', 'tumor')


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('INVALID_PATIENT_POSITION:' + name)
    return float(value)


def _vector(value, name, limit):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('INVALID_PATIENT_POSITION:' + name)
    return [_number(v, name, -limit, limit) for v in value]


def validate_position(payload):
    """Normalize an entire position action; omitted settings use preset defaults.

    ``heartToSiteRasMm`` is an estimated heart-to-operative-site vector expressed
    in *body* RAS. A supplied signed vertical height overrides that vector only
    for the hydrostatic descriptor. Neither represents an image measurement.
    """
    allowed = {'modelVersion', 'preset', 'bodyTiltDeg', 'headRotationRasDeg',
               'heartToSiteRasMm', 'siteHeightAboveHeartMm', 'heightEstimateSource',
               'tissueDensityKgPerM3', 'fluidDensityKgPerM3'}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError('INVALID_PATIENT_POSITION:fields')
    if type(payload.get('modelVersion', 1)) is not int or payload.get('modelVersion', 1) != MODEL_VERSION:
        raise ValueError('INVALID_PATIENT_POSITION:modelVersion')
    preset = payload.get('preset')
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError('INVALID_PATIENT_POSITION:preset')
    result = {'modelVersion': MODEL_VERSION, 'preset': preset,
              'bodyTiltDeg': _number(payload.get('bodyTiltDeg', PRESETS[preset][1]), 'bodyTiltDeg', -45, 85),
              'headRotationRasDeg': _vector(payload.get('headRotationRasDeg', [0., 0., 0.]), 'headRotationRasDeg', 90),
              'heartToSiteRasMm': _vector(payload.get('heartToSiteRasMm', [0., 0., 300.]), 'heartToSiteRasMm', 2000),
              'tissueDensityKgPerM3': _number(payload.get('tissueDensityKgPerM3', 1040.), 'tissueDensityKgPerM3', 500, 2000),
              'fluidDensityKgPerM3': _number(payload.get('fluidDensityKgPerM3', 1060.), 'fluidDensityKgPerM3', 500, 2000),
              'heightEstimateSource': payload.get('heightEstimateSource', 'unmeasured research assumption')}
    source = result['heightEstimateSource']
    if not isinstance(source, str) or not source.strip() or len(source) > 500:
        raise ValueError('INVALID_PATIENT_POSITION:heightEstimateSource')
    if 'siteHeightAboveHeartMm' in payload:
        result['siteHeightAboveHeartMm'] = _number(payload['siteHeightAboveHeartMm'], 'siteHeightAboveHeartMm', -2000, 2000)
    return result


def apply_position(state, payload):
    """Atomically install validated, JSON-serializable opt-in state."""
    position = validate_position(payload)
    state['patientPosition'] = position
    return describe_position(position)


def _rotation(axis, degrees):
    angle = math.radians(degrees); c = math.cos(angle); s = math.sin(angle)
    if axis == 0: return np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1: return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _frames(position):
    # Room Y points upward when supine; room Z is the horizontal head direction.
    body = _rotation(0, -position['bodyTiltDeg']) @ _rotation(2, PRESETS[position['preset']][0])
    x, y, z = position['headRotationRasDeg']
    head = body @ _rotation(2, z) @ _rotation(1, y) @ _rotation(0, x)
    return body, head


def describe_position(position):
    """Return replayable geometric/qualitative context, without clinical scores."""
    position = validate_position(position)
    body, head = _frames(position); down = np.array([0., -GRAVITY_M_PER_S2, 0.])
    g = head.T @ down; body_g = body.T @ down
    height = position.get('siteHeightAboveHeartMm', float((body @ position['heartToSiteRasMm'])[1]))
    # p_site - p_heart = -rho*g*height; height mm -> m, pressure Pa.
    pressure = -position['fluidDensityKgPerM3'] * GRAVITY_M_PER_S2 * height * .001
    return {**copy.deepcopy(position), 'coordinateFrame': 'patient physical RAS: +R right, +A anterior, +S superior',
            'bodyRasToRoom': body.tolist(), 'headRasToRoom': head.tolist(),
            'gravityRasMPerS2': g.tolist(), 'bodyGravityRasMPerS2': body_g.tolist(),
            'siteHeightAboveHeartEstimateMm': height,
            'hydrostaticSiteMinusHeartPa': pressure,
            'hydrostaticSiteMinusHeartMmHg': pressure / PA_PER_MMHG,
            'heightIsMeasured': False, 'hydrostaticModel': 'static open fluid column only; not venous pressure, ICP or perfusion',
            'qualitativeContext': ['Posture can change venous drainage and intracranial pressure; response is patient dependent.',
                'Head and neck rotation or flexion can impede venous drainage; no jugular geometry or patency is modeled.',
                'An elevated open venous site can permit air entrainment; no embolism or clinical risk prediction is modeled.'],
            'mechanicsAssumptions': 'full tissue weight; no CSF buoyancy, whole-brain FEM or calibrated constitutive response',
            'sources': copy.deepcopy(SOURCES)}


def gravity_forces_n(engine, ids):
    """Nx3 force in head/image physical RAS, N, for existing lattice node IDs."""
    position = engine.state.get('patientPosition')
    if position is None: return np.zeros((len(ids), 3))
    position = validate_position(position); _, head = _frames(position)
    gravity = head.T @ np.array([0., -GRAVITY_M_PER_S2, 0.])
    mass = position['tissueDensityKgPerM3'] * abs(float(np.linalg.det(engine.scene.affine[:3, :3]))) * 1e-9
    return np.tile(mass * gravity, (len(ids), 1))


def _neighbors(engine, cell):
    ijk = list(np.unravel_index(cell, engine.scene.shape))
    for axis in range(3):
        for sign in (-1, 1):
            nxt = ijk.copy(); nxt[axis] += sign
            if 0 <= nxt[axis] < engine.scene.shape[axis]:
                yield int(np.ravel_multi_index(nxt, engine.scene.shape)), float(np.linalg.norm(engine.scene.affine[:3, axis]))


def _material(engine, cell):
    return engine.by_label.get(int(engine.scene.labels.flat[cell]), {}).get('material')


def activate_exposed_patch(engine, *, radius_mm=4., max_nodes=2000, max_seeds=128, exposed_cells=None):
    """Bounded local patch adjoining resection, with cut bonds and outer anchors.

    Only the last 512 entries of the canonical removed list are tested.
    Optional already identified exposure candidates avoid any whole-grid scan.
    Neither image borders nor arbitrary zero-label background create exposure.
    """
    if 'patientPosition' not in engine.state: return []
    _number(radius_mm, 'patchRadiusMm', .1, 12)
    if type(max_nodes) is not int or not 1 <= max_nodes <= 2000 or type(max_seeds) is not int or not 1 <= max_seeds <= 128:
        raise ValueError('INVALID_PATIENT_POSITION:patchLimits')
    removed = set(engine.state['removed'])
    from .surgical_deformation_v2 import rigid_cells
    rigid = rigid_cells(engine)
    blocked = {tuple(sorted(map(int, e))) for field in ('cutEdges', 'transectedEdges') for e in engine.state.get(field, [])}
    held = {c for h in engine.state.get('holds', {}).values() for c in h['cells']}
    def signature(active):
        return (len(removed), tuple(engine.state['removed'][-512:]), tuple(sorted(blocked)),
                radius_mm, max_nodes, max_seeds, tuple(active), tuple(sorted(held)), tuple(sorted(rigid)))
    current_active = engine.state.get('deformationModel', {}).get('active', [])
    if exposed_cells is None and getattr(engine, '_patient_position_patch_signature', None) == signature(current_active):
        return list(current_active)
    candidates = exposed_cells
    if candidates is None:
        candidates = (n for c in engine.state['removed'][-512:] for n, _ in _neighbors(engine, c))
    seeds = []
    for c in islice(candidates, 3072):
        c = int(c)
        if len(seeds) >= max_seeds: break
        if c not in removed and c not in rigid and c not in seeds and _material(engine, c) in _SOFT and any(n in removed for n, _ in _neighbors(engine, c)):
            seeds.append(c)
    if not seeds:
        if exposed_cells is None: engine._patient_position_patch_signature = signature(current_active)
        return []
    engine.state.setdefault('deformation', {})
    model = engine.state.setdefault('deformationModel', {'active': [], 'anchors': [], 'remainderSeconds': 0., 'velocities': {}})
    active = set(model['active']) - removed - rigid
    distance = {c: 0. for c in seeds}; queue = deque(seeds)
    while queue:
        c = queue.popleft()
        if c not in active and len(active) >= max_nodes: continue
        active.add(c)
        for n, length in _neighbors(engine, c):
            d = distance[c] + length
            if d > radius_mm or n in removed or n in rigid or (min(c, n), max(c, n)) in blocked or _material(engine, n) not in _SOFT: continue
            if d < distance.get(n, float('inf')): distance[n] = d; queue.append(n)
    anchors = {c for c in active if any(n not in active and n not in removed and n not in rigid and _material(engine, n) in _SOFT and
               (min(c, n), max(c, n)) not in blocked for n, _ in _neighbors(engine, c))}
    model['active'] = sorted(active); model['anchors'] = sorted(anchors - held)
    if exposed_cells is None: engine._patient_position_patch_signature = signature(model['active'])
    return sorted(active)


def relax_fluid(engine, dt, *, mobility_per_second=4., max_cells=2000):
    """Conservative downhill free-space relaxation, one edge per call.

    Caller invokes once per fixed physics microstep after source/overflow work.
    The dimensionless rate and capacity are disclosed lattice assumptions, not
    viscosity or a Navier-Stokes solver. No escape reservoir is touched here.
    Mixed debris moves with its local carrier concentration. Returns transfers.
    """
    if 'patientPosition' not in engine.state: return []
    _number(dt, 'fluidTimeSeconds', 0, 1)
    _number(mobility_per_second, 'fluidMobilityPerSecond', 0, 100)
    if type(max_cells) is not int or not 1 <= max_cells <= 2000:
        raise ValueError('INVALID_PATIENT_POSITION:fluidCellLimit')
    if not dt: return []
    position = validate_position(engine.state['patientPosition']); _, head = _frames(position)
    down = head.T @ np.array([0., -1., 0.])
    occupancy = engine.occupancy(); fluid = engine.state['fluid']; debris = engine.state.setdefault('debris', {})
    capacity = abs(float(np.linalg.det(engine.scene.affine[:3, :3])))
    initial = {int(k): float(v) for k, v in fluid.items() if v > 0}; outgoing = {}; incoming = {}; sediment_out = {}; sediment_in = {}; transfers = []
    fraction = min(1., mobility_per_second * dt)
    for c in sorted(initial)[:max_cells]:
        if initial[c] <= 0 or occupancy[c]: continue
        ijk = np.array(np.unravel_index(c, engine.scene.shape)); choices = []
        for n, length in _neighbors(engine, c):
            if occupancy[n]: continue
            delta = (np.array(np.unravel_index(n, engine.scene.shape)) - ijk) @ engine.scene.affine[:3, :3].T
            descent = float(delta @ down)
            if descent > 1e-10: choices.append((-descent / length, n))
        for direction, n in sorted(choices):
            available = initial[c] * fraction - outgoing.get(c, 0.)
            room = capacity - initial.get(n, 0.) - incoming.get(n, 0.)
            amount = min(available, max(0., room))
            if amount <= 0: continue
            outgoing[c] = outgoing.get(c, 0.) + amount; incoming[n] = incoming.get(n, 0.) + amount
            sediment = debris.get(str(c), 0.) * amount / initial[c]
            sediment_out[c] = sediment_out.get(c, 0.) + sediment; sediment_in[n] = sediment_in.get(n, 0.) + sediment
            transfers.append({'fromCell': c, 'toCell': n, 'fluidMm3': amount, 'debrisMm3': sediment})
    for field, out, ins in ((fluid, outgoing, incoming), (debris, sediment_out, sediment_in)):
        for c in sorted(set(out) | set(ins)):
            amount = max(0., field.get(str(c), 0.) - out.get(c, 0.) + ins.get(c, 0.))
            if amount: field[str(c)] = amount
            else: field.pop(str(c), None)
    engine.state['debrisMm3'] = sum(debris.values())
    trace = getattr(engine, '_flow_display_trace', None)
    if trace is not None and trace.active:
        for transfer in transfers:
            trace.record('transfer', 'fluid', 'mixed', transfer['fromCell'], transfer['toCell'], transfer['fluidMm3'])
            if transfer['debrisMm3']: trace.record('transfer', 'debris', 'debris', transfer['fromCell'], transfer['toCell'], transfer['debrisMm3'])
    return transfers
