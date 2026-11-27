"""Opt-in tissue-selective ultrasonic aspirator (``cusaV4``), a research surrogate.

What it answers for a surgeon: how tumour and brain behave under the CUSA
tip. Ultrasonic aspirators fragment high-water-content parenchyma and
relatively spare collagen- and elastin-rich vessel walls and membranes; that
well-known rationale is the only qualitative rule taken from practice. Every
number below is an assumed research value, not a device or patient fit.

* Settings per application: ``amplitude`` (0-1 tip stroke), ``aspiration``
  (0-1 vacuum), ``irrigation`` (mL/min) and ``duty`` (0-1).
* Fragmentation capacity = base rate x amplitude x duty x tip efficiency,
  shared across the current tip contacts nearest first, exactly as V3: whole
  source cells with exact rational pending progress per cell under
  ``state['cusaV4']`` (same ``progressMm3``/``progressExact`` layout).
* Selectivity: a cell needs voxel volume / relative rate of capacity, the
  rate set by its tissue kind (tumour core fastest, rim and grey matter
  slower, white matter and tracts slower still).
* Vessel and membrane cells in contact take no capacity. They accumulate a
  dwell dose only above a sparing amplitude and fail (are removed, so the
  bleeding and injury models see them) when the dose passes a threshold that
  rises with vessel calibre.
* Fragmented tissue leaves through the tip up to an aspiration capacity; the
  excess stays as debris in the cavity. Remaining capacity takes fluid then
  debris in reach, with the suction accounting fields.
* Irrigation adds fluid at the tip. Running dry with the tip vibrating builds
  a ``tipHeat`` proxy that lowers efficiency. No thermal injury is claimed.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import math

import numpy as np

PROPERTY = 'cusaV4'
SCHEMA = 'resectionlab.cusa-selective.v4'
STATE_SCHEMA = 'resectionlab.cusa-selective-state.v4'
TIP_RADIUS_MM = 1.5
SHAFT_RADIUS_MM = .5
IRRIGATION_LIMIT_ML_PER_MINUTE = 30.
# 1 mL = 1000 mm3 and 1 min = 60 s.
ML_PER_MIN_TO_MM3_PER_S = 1000. / 60.
ASSUMED = 'assumed-research'

FRAGMENTABLE = ('tumour_core', 'tumour', 'tumour_rim', 'grey_matter', 'white_matter', 'tract', 'soft_tissue')
SPARED = ('vessel', 'dura', 'membrane')
_MATERIAL_KINDS = {'tumor': ('tumour_core', 'tumour', 'tumour_rim'),
                   'soft_tissue': ('grey_matter', 'white_matter', 'tract', 'soft_tissue'),
                   'vessel': ('vessel',), 'membrane': ('dura', 'membrane')}

# (value, provenance) for every number; the profile may override values only.
PARAMETERS = {
    # Clinical ultrasonic aspirators debulk soft glioma on the order of cm3 per
    # minute; 40 mm3/s is 2.4 cm3/min at full amplitude on tumour core.
    'baseRateMm3PerSecond': (40., ASSUMED),
    # Relative fragmentation rate by tissue kind (1 = tumour core).
    'relativeRate': ({'tumour_core': 1., 'tumour': 1., 'tumour_rim': .6, 'grey_matter': .5,
                      'white_matter': .3, 'tract': .3, 'soft_tissue': .4}, ASSUMED),
    # Amplitude at or below which vessel walls and membranes take no dose.
    'sparingAmplitude': (.3, ASSUMED),
    # Full-amplitude-equivalent seconds to breach a vessel of the reference calibre.
    'vesselReferenceSeconds': (1., ASSUMED),
    'vesselReferenceCalibreMm': (1., ASSUMED),
    # Wall strength ~ diameter x wall thickness ~ d^2 (thickness scales with d).
    'vesselCalibreExponent': (2., ASSUMED),
    'duraFailureSeconds': (3., ASSUMED),
    'membraneFailureSeconds': (1., ASSUMED),
    # Solid-or-fluid volume the tip can carry away per second at full vacuum.
    'aspirationCapacityMm3PerSecond': (150., ASSUMED),
    # Irrigation needed at full amplitude to keep the tip cool.
    'irrigationRequiredMlPerMinute': (2., ASSUMED),
    'tipHeatingPerSecond': (1., ASSUMED),
    'tipCoolingPerSecond': (.5, ASSUMED),
    'defaultAmplitude': (.5, ASSUMED),
    'defaultAspiration': (.5, ASSUMED),
    'defaultIrrigationMlPerMinute': (4., ASSUMED),
}
LIMITS = (
    'Fixed generic 1.5 mm tip; not a manufacturer tip specification.',
    'Rates, selectivity, sparing amplitude, failure doses, aspiration capacity and tip heat are uncalibrated.',
    'Whole source voxels remain until their complete capacity threshold is reached.',
    'No acoustics, cavitation, tissue temperature or thermal injury; tipHeat is an efficiency proxy only.',
    'Vessel calibre is a grid local-thickness estimate; vessels thinner than one voxel read as one voxel.',
    'Irrigation that finds no free cell at the tip runs off and is counted as escaped fluid.',
)
_calibre_cache = {}


def default_profile():
    return {'schema': SCHEMA, **{name: copy.deepcopy(value) for name, (value, _) in PARAMETERS.items()},
            'regionKinds': {},
            'provenance': {'kind': ASSUMED, 'description':
                'Uncalibrated tissue-selective ultrasonic aspirator surrogate; values are engineering assumptions.'}}


def parameter_sources():
    return {name: source for name, (_, source) in PARAMETERS.items()}


def _positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError('INVALID_CUSA_V4_PROFILE:' + name)
    return float(value)


def validate(value):
    """Validate a declared profile, or a scene's optional one; returns the merged profile."""
    if hasattr(value, 'properties'):
        if PROPERTY not in value.properties:
            return None
        scene, value = value, value.properties[PROPERTY]
    else:
        scene = None
    defaults = default_profile()
    if not isinstance(value, dict) or set(value) - set(defaults) or value.get('schema') != SCHEMA:
        raise ValueError('CUSA_V4_PROFILE_INVALID: schema and known fields required')
    result = {**defaults, **copy.deepcopy(value)}
    provenance = result['provenance']
    if not isinstance(provenance, dict) or provenance.get('kind') != ASSUMED or not str(provenance.get('description', '')).strip():
        raise ValueError('EXPLICIT_ASSUMED_RESEARCH_CUSA_PROVENANCE_REQUIRED')
    for name in PARAMETERS:
        if name == 'relativeRate':
            continue
        _positive(result[name], name, zero=name in ('sparingAmplitude',))
    if not result['sparingAmplitude'] < 1.:
        raise ValueError('INVALID_CUSA_V4_PROFILE:sparingAmplitude')
    for name in ('defaultAmplitude', 'defaultAspiration'):
        if result[name] > 1.:
            raise ValueError('INVALID_CUSA_V4_PROFILE:' + name)
    if result['defaultIrrigationMlPerMinute'] > IRRIGATION_LIMIT_ML_PER_MINUTE:
        raise ValueError('INVALID_CUSA_V4_PROFILE:defaultIrrigationMlPerMinute')
    rates = result['relativeRate']
    if not isinstance(rates, dict) or set(rates) != set(FRAGMENTABLE):
        raise ValueError('INVALID_CUSA_V4_PROFILE:relativeRate')
    for kind, rate in rates.items():
        _positive(rate, 'relativeRate.' + kind)
    kinds = result['regionKinds']
    if not isinstance(kinds, dict) or any(k not in FRAGMENTABLE + SPARED for k in kinds.values()):
        raise ValueError('INVALID_CUSA_V4_PROFILE:regionKinds')
    if scene is not None:
        for region in scene.regions:
            kind = _region_kind(result, region)
            if kind is not None and kind not in _MATERIAL_KINDS.get(region['material'], ()):
                raise ValueError('CUSA_V4_REGION_KIND_MATERIAL_MISMATCH:' + str(region['name']))
    return result


def profile(engine):
    cached = engine.__dict__.get('_cusa_v4_profile')
    if cached is not None and cached[0] is engine.scene.properties.get(PROPERTY):
        return cached[1]
    result = validate(engine.scene)
    engine.__dict__['_cusa_v4_profile'] = (engine.scene.properties.get(PROPERTY), result)
    return result


def enabled(engine):
    return PROPERTY in engine.scene.properties


def _region_kind(p, region):
    """Declared kind, else inferred from the region name and material."""
    declared = p['regionKinds'].get(region['name'], p['regionKinds'].get(str(region['id'])))
    if declared is None:
        declared = region.get('cusaTissueKind')
    if declared is not None:
        return declared
    name = str(region['name']).lower()
    material = region['material']
    if material == 'tumor':
        if 'core' in name or 'necro' in name:
            return 'tumour_core'
        if any(word in name for word in ('rim', 'margin', 'infiltrat', 'edge')):
            return 'tumour_rim'
        return 'tumour'
    if material == 'soft_tissue':
        if 'tract' in name:
            return 'tract'
        if 'white' in name or 'brainstem' in name:
            return 'white_matter'
        if any(word in name for word in ('cortex', 'grey', 'gray', 'cerebellum', 'deep')):
            return 'grey_matter'
        return 'soft_tissue'
    if material == 'vessel':
        return 'vessel'
    if material == 'membrane':
        return 'dura' if 'dura' in name else 'membrane'
    return None


def _kind(engine, p, cell):
    label = int(engine.scene.labels.flat[cell])
    return _region_kind(p, engine.by_label[label]) if label else None


def initialize(engine):
    if not enabled(engine):
        return
    profile(engine)
    engine.state.setdefault(PROPERTY, {'schema': STATE_SCHEMA,
        'progressMm3': {}, 'progressExact': {}, 'contactSeconds': {},
        'sparedDoseSeconds': {}, 'sparedDoseExact': {}, 'sparedFailures': [],
        'inputCapacityMm3': 0., 'unusedCapacityMm3': 0., 'discardedCapacityMm3': 0.,
        'selectivityCapacityMm3': 0., 'fragmentedSolidMm3': 0., 'sparedFailedMm3': 0.,
        'aspiratedSolidMm3': 0., 'debrisLeftMm3': 0., 'fluidAspiratedMm3': 0., 'debrisAspiratedMm3': 0.,
        'irrigationAddedMm3': 0., 'irrigationEscapedMm3': 0., 'effectiveActivationSeconds': 0.,
        'tipHeat': 0., 'lastActiveSeconds': None})


def _bounded(value, label, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError('INVALID_CUSA_V4_INPUT:' + label)
    return float(value)


def validate_parameters(params, p=None):
    """Return amplitude, aspiration, irrigation (mL/min) and duty for one application."""
    if not isinstance(params, dict):
        raise ValueError('INVALID_CUSA_V4_PARAMETERS')
    p = p or default_profile()
    if params.get('radiusMm', TIP_RADIUS_MM) != TIP_RADIUS_MM or params.get('shaftRadiusMm', SHAFT_RADIUS_MM) != SHAFT_RADIUS_MM:
        raise ValueError('CUSA_V4_FIXED_CATALOG_GEOMETRY_REQUIRED')
    return (_bounded(params.get('amplitude', p['defaultAmplitude']), 'amplitude', 1.),
            _bounded(params.get('aspiration', p['defaultAspiration']), 'aspiration', 1.),
            _bounded(params.get('irrigation', p['defaultIrrigationMlPerMinute']), 'irrigation', IRRIGATION_LIMIT_ML_PER_MINUTE),
            _bounded(params.get('duty', 1.), 'duty', 1.))


def reconcile_removed(engine):
    """Discard pending work and dwell for cells another mechanism removed."""
    if PROPERTY not in engine.state:
        return 0.
    state = engine.state[PROPERTY]
    removed = set(engine.state['removed'])
    stale = [key for key in state['progressMm3'] if int(key) in removed]
    discarded = math.fsum(state['progressMm3'].pop(key) for key in stale)
    for key in stale:
        state['contactSeconds'].pop(key, None)
        state['progressExact'].pop(key, None)
    for key in [key for key in state['sparedDoseSeconds'] if int(key) in removed]:
        state['sparedDoseSeconds'].pop(key)
        state['sparedDoseExact'].pop(key, None)
    state['discardedCapacityMm3'] += discarded
    return discarded


def _position(value):
    if value is None:
        raise ValueError('CUSA_V4_EXPLICIT_TIP_POSITION_REQUIRED')
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('CUSA_V4_FINITE_RAS_MM_POSITION_REQUIRED')
    return point


def _ordered_contacts(engine, cells, position):
    """Reachable fragmentable and spared tip contacts, nearest first (V3 order)."""
    p = profile(engine)
    labels = engine.scene.labels.ravel()
    ids = set()
    for cell in cells:
        if not isinstance(cell, (int, np.integer)) or isinstance(cell, (bool, np.bool_)) or not 0 <= cell < labels.size:
            raise ValueError('INVALID_CUSA_V4_SOURCE_CELL')
        ids.add(int(cell))
    ids.intersection_update(engine.touched(position, position, TIP_RADIUS_MM))
    ids.difference_update(engine.state['removed'])
    ids = [cell for cell in ids if labels[cell] and _kind(engine, p, cell) in FRAGMENTABLE + SPARED]
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


def spared_contacts(engine, cells):
    """Vessel and membrane cells the active V4 tip may touch without refusal."""
    p = profile(engine)
    return {int(cell) for cell in cells if engine.scene.labels.flat[cell] and _kind(engine, p, cell) in SPARED}


def calibre_mm(engine, cell):
    """Vessel diameter (mm) at a cell and its basis.

    Uses the vesselBleedingV1 segment estimate when declared, else a declared
    region ``calibreMm``, else a local-thickness estimate on the region mask
    (2R - s with s the smallest spacing, floored at s; same convention).
    """
    cell = int(cell)
    if 'vesselBleedingV1' in engine.scene.properties:
        from .surgical_vessel_bleeding import local_calibre, segment_of
        segment = int(segment_of(engine.scene, [cell])[0])
        if segment >= 0:
            return float(local_calibre(engine, cell, segment)[0]), 'vesselBleedingV1'
    label = int(engine.scene.labels.flat[cell])
    declared = engine.by_label[label].get('calibreMm')
    if type(declared) in (int, float) and math.isfinite(declared) and declared > 0:
        return float(declared), 'declared-region'
    key = (engine.scene.identity, cell)
    if key not in _calibre_cache:
        from scipy import ndimage
        scene = engine.scene
        shape = np.array(scene.shape); spacing = np.asarray(scene.spacing, float); floor = float(spacing.min())
        centre = np.array(np.unravel_index(cell, scene.shape)); half = 6
        lo = np.maximum(centre - half, 0); hi = np.minimum(centre + half + 1, shape)
        window = scene.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == label
        distance = ndimage.distance_transform_edt(np.pad(window, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        local = centre - lo
        near = tuple(slice(max(0, c - 1), c + 2) for c in local)
        radius = float(np.max(np.where(window[near], distance[near], 0.)))
        _calibre_cache[key] = max(floor, 2 * radius - floor)
        if len(_calibre_cache) > 4096:
            _calibre_cache.pop(next(iter(_calibre_cache)))
    return _calibre_cache[key], 'local-thickness'


def failure_seconds(engine, cell, kind=None):
    """Full-amplitude-equivalent dwell that breaches a spared cell, and its calibre."""
    p = profile(engine)
    kind = kind or _kind(engine, p, cell)
    if kind == 'dura':
        return p['duraFailureSeconds'], None
    if kind == 'membrane':
        return p['membraneFailureSeconds'], None
    calibre, _ = calibre_mm(engine, cell)
    return p['vesselReferenceSeconds'] * (calibre / p['vesselReferenceCalibreMm']) ** p['vesselCalibreExponent'], calibre


def _free_cell(engine, position):
    """Nearest free spatial cell in the tip sphere, or None."""
    occupancy = engine.occupancy()
    cells = sorted(map(int, engine.cells(position, position, TIP_RADIUS_MM)))
    free = [cell for cell in cells if occupancy[cell] == 0]
    if not free:
        return None
    points = engine.scene.points(free)
    return min(zip(free, points), key=lambda item: (float(np.sum((item[1] - position) ** 2)), item[0]))[0]


def _record_injury(engine, key, cells):
    """Injury mode records a breached vessel or membrane as an ultrasonic laceration."""
    from . import surgical_injury as injury
    if key is None or not cells or not injury.converting(engine):
        return
    record = engine.state['injuryV1']
    vessels = injury._remove(engine, record, cells)
    injury._event(engine, record, 'laceration', 'CUSA_SELECTIVITY_EXCEEDED', cells, key, mechanism='ultrasonic')
    if vessels:
        injury._event(engine, record, 'vessel_injury', 'CUSA_SELECTIVITY_EXCEEDED', vessels, key,
                      cell_ids=True, mechanism='ultrasonic')
    injury._refresh(engine, record)


def apply(engine, cells, dt, params, *, position=None, active=True, key=None):
    """One application interval at the current tip; returns events, ledgers in state['cusaV4']."""
    p = profile(engine)
    if p is None:
        raise ValueError('CUSA_V4_NOT_DECLARED')
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt < 0:
        raise ValueError('INVALID_CUSA_V4_INPUT:dt')
    dt = float(dt)
    amplitude, aspiration, irrigation, duty = validate_parameters(params, p)
    position = _position(position)
    if type(active) is not bool:
        raise ValueError('CUSA_V4_EFFECTIVE_ACTIVATION_MUST_BE_BOOLEAN')
    ids = _ordered_contacts(engine, cells, position)
    protected = set(ids).intersection(getattr(engine, '_protected_ids', ()))
    if protected:
        from . import surgical_injury as injury
        if key is None:
            raise ValueError('PROTECTED_STRUCTURE: CUSA contact includes protected source membership')
        injury.protected(engine, 'PROTECTED_STRUCTURE: CUSA contact includes protected source membership', protected, key)
    initialize(engine)
    discarded = reconcile_removed(engine)
    state = engine.state[PROPERTY]
    now = float(engine.state['elapsedSeconds'])
    if state['lastActiveSeconds'] is not None and now > state['lastActiveSeconds']:
        # Idle time since the last application cools the tip (implicit decay).
        state['tipHeat'] = state['tipHeat'] / (1. + p['tipCoolingPerSecond'] * (now - state['lastActiveSeconds']))
    drive = amplitude * duty if active else 0.
    efficiency = 1. / (1. + state['tipHeat'])
    budget = p['baseRateMm3PerSecond'] * drive * efficiency * dt
    if not math.isfinite(budget):
        raise ValueError('CUSA_V4_CAPACITY_OVERFLOW')
    remaining = Fraction.from_float(budget)
    volume = Fraction.from_float(float(engine.scene.voxel_mm3))
    new, spent, failed = [], [], []
    kinds = {cell: _kind(engine, p, cell) for cell in ids}
    for cell in (ids if budget > 0. else ()):
        if kinds[cell] in SPARED:
            continue
        name = str(cell)
        old = Fraction(*state['progressExact'][name]) if name in state['progressExact'] else Fraction(0)
        threshold = volume / Fraction.from_float(float(p['relativeRate'][kinds[cell]]))
        required = threshold - old
        allocated = min(required, remaining)
        if allocated <= 0:
            continue
        state['contactSeconds'][name] = state['contactSeconds'].get(name, 0.) + dt
        remaining -= allocated
        spent.append(float(allocated))
        # No epsilon or rounded display value can authorize whole-cell removal.
        if allocated >= required:
            new.append(cell)
            state['selectivityCapacityMm3'] += float(threshold - volume)
            state['progressMm3'].pop(name, None)
            state['progressExact'].pop(name, None)
            state['contactSeconds'].pop(name, None)
        else:
            progress = old + allocated
            state['progressMm3'][name] = float(progress)
            state['progressExact'][name] = [progress.numerator, progress.denominator]
    # Vessel walls and membranes take a dwell dose only above the sparing amplitude.
    floor = p['sparingAmplitude']
    weight = (amplitude - floor) / (1. - floor) * duty if active and amplitude > floor else 0.
    dose = Fraction.from_float(weight) * Fraction.from_float(dt)
    for cell in ids:
        if kinds[cell] not in SPARED or not dose:
            continue
        name = str(cell)
        old = Fraction(*state['sparedDoseExact'][name]) if name in state['sparedDoseExact'] else Fraction(0)
        total = old + dose
        threshold, calibre = failure_seconds(engine, cell, kinds[cell])
        if total >= Fraction.from_float(float(threshold)):
            failed.append(cell)
            state['sparedDoseSeconds'].pop(name, None)
            state['sparedDoseExact'].pop(name, None)
            state['sparedFailures'].append({'cell': cell, 'kind': kinds[cell], 'calibreMm': calibre,
                                            'failureSeconds': float(threshold), 'timeSeconds': now})
        else:
            state['sparedDoseSeconds'][name] = float(total)
            state['sparedDoseExact'][name] = [total.numerator, total.denominator]
    voxel = float(engine.scene.voxel_mm3)
    removed_now = new + failed
    if removed_now:
        engine.state['removed'] = sorted(set(engine.state['removed']).union(removed_now))
        engine._occupancy_cache = None
        _record_injury(engine, key, failed)
    fragmented = len(new) * voxel
    solid = len(removed_now) * voxel
    # Irrigation leaves the sleeve at the tip; a blocked tip runs off the shaft.
    flow = getattr(engine, '_flow_display_trace', None)
    if flow is not None and not flow.active:
        flow = None
    irrigated = irrigation * ML_PER_MIN_TO_MM3_PER_S * dt if active else 0.
    escaped = 0.
    if irrigated:
        outlet = _free_cell(engine, position)
        if outlet is None:
            escaped = irrigated
            engine.state['fluidEscapedMm3'] += irrigated
            if flow is not None:
                flow.record('escape', 'fluid', 'irrigation', None, None, irrigated)
        else:
            name = str(outlet)
            engine.state['fluid'][name] = engine.state['fluid'].get(name, 0.) + irrigated
            if flow is not None:
                flow.record('deposit', 'fluid', 'irrigation', None, outlet, irrigated)
        engine.state['fluidAddedMm3'] += irrigated
    # Fragments leave first; the excess stays as debris in the cells it came from.
    capacity = p['aspirationCapacityMm3PerSecond'] * aspiration * dt if active else 0.
    carried = min(capacity, solid)
    left = solid - carried
    if left > 0.:
        share = left / len(removed_now)
        for cell in removed_now:
            name = str(cell)
            engine.state['debris'][name] = engine.state['debris'].get(name, 0.) + share
            if flow is not None:
                flow.record('deposit', 'debris', 'debris', None, cell, share)
    capacity -= carried
    taken = {'fluid': 0., 'debris': 0.}
    if capacity > 0.:
        spatial = sorted(map(int, engine.cells(position, position, TIP_RADIUS_MM)))
        for field in ('fluid', 'debris'):
            for cell in spatial:
                name = str(cell)
                take = min(capacity, engine.state[field].get(name, 0.))
                if take:
                    engine.state[field][name] -= take
                    taken[field] += take
                    capacity -= take
                    if flow is not None:
                        flow.record('uptake', field, 'mixed' if field == 'fluid' else 'debris', cell, None, take)
                    if engine.state[field][name] <= 1e-12:
                        del engine.state[field][name]
    engine.state['fluidRemovedMm3'] += taken['fluid']
    if taken['debris']:
        engine.state['debrisRemovedMm3'] = engine.state.get('debrisRemovedMm3', 0.) + taken['debris']
    engine.state['debrisMm3'] = sum(engine.state['debris'].values())
    # Dry vibration heats the tip; irrigation and idle time cool it.
    if active:
        wet = min(1., irrigation / (p['irrigationRequiredMlPerMinute'] * amplitude)) if amplitude > 0 else 1.
        heating = p['tipHeatingPerSecond'] * drive * (1. - wet)
        state['tipHeat'] = (state['tipHeat'] + heating * dt) / (1. + p['tipCoolingPerSecond'] * dt)
        state['lastActiveSeconds'] = now + dt
    state['inputCapacityMm3'] += budget
    state['unusedCapacityMm3'] += float(remaining)
    state['fragmentedSolidMm3'] += fragmented
    state['sparedFailedMm3'] += len(failed) * voxel
    state['aspiratedSolidMm3'] += carried
    state['debrisLeftMm3'] += left
    state['fluidAspiratedMm3'] += taken['fluid']
    state['debrisAspiratedMm3'] += taken['debris']
    state['irrigationAddedMm3'] += irrigated
    state['irrigationEscapedMm3'] += escaped
    state['effectiveActivationSeconds'] += drive * dt
    events = []
    if budget or discarded or irrigated or carried or any(taken.values()) or failed:
        events.append({'kind': 'cusa_contact', 'model': SCHEMA, 'amplitude': amplitude, 'aspiration': aspiration,
            'irrigationMlPerMinute': irrigation, 'inputCapacityMm3': budget,
            'allocatedCapacityMm3': math.fsum(spent), 'unusedCapacityMm3': float(remaining),
            'fragmentedSolidMm3': fragmented, 'aspiratedSolidMm3': carried, 'debrisLeftMm3': left,
            'fluidAspiratedMm3': taken['fluid'], 'debrisAspiratedMm3': taken['debris'],
            'irrigationMm3': irrigated, 'tipHeat': state['tipHeat'], 'efficiency': efficiency,
            'discardedCapacityMm3': discarded, 'removedCells': new,
            'sparedContacts': sum(kinds[c] in SPARED for c in ids), 'pendingCells': len(state['progressMm3'])})
    if failed:
        events.append({'kind': 'cusa_spared_failure', 'cells': failed,
                       'kinds': [kinds[c] for c in failed], 'amplitude': amplitude,
                       'regions': sorted({engine.by_label[int(engine.scene.labels.flat[c])]['name'] for c in failed})})
    return events


def contact_limited_pose(engine, start, requested, radius=TIP_RADIUS_MM):
    """Stop at the first retained fragmentable or spared tissue; changes no state."""
    profile(engine)
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
        if contacts(high):
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
    state = engine.state.get(PROPERTY, {})
    return {'cusaV4InputCapacityMm3': state.get('inputCapacityMm3', 0.),
            'cusaV4FragmentedSolidMm3': state.get('fragmentedSolidMm3', 0.),
            'cusaV4AspiratedSolidMm3': state.get('aspiratedSolidMm3', 0.),
            'cusaV4DebrisLeftMm3': state.get('debrisLeftMm3', 0.),
            'cusaV4PendingCapacityMm3': math.fsum(state.get('progressMm3', {}).values()),
            'cusaV4PendingCells': len(state.get('progressMm3', {})),
            'cusaV4SelectivityCapacityMm3': state.get('selectivityCapacityMm3', 0.),
            'cusaV4UnusedCapacityMm3': state.get('unusedCapacityMm3', 0.),
            'cusaV4DiscardedCapacityMm3': state.get('discardedCapacityMm3', 0.),
            'cusaV4SparedFailures': len(state.get('sparedFailures', [])),
            'cusaV4SparedFailedMm3': state.get('sparedFailedMm3', 0.),
            'cusaV4IrrigationAddedMm3': state.get('irrigationAddedMm3', 0.),
            'cusaV4TipHeat': state.get('tipHeat', 0.)}


def descriptor(engine):
    return {'enabled': enabled(engine), 'profile': copy.deepcopy(profile(engine)),
            'sources': parameter_sources(), 'calibration': 'uncalibrated assumed research surrogate',
            'rationale': 'Ultrasonic aspirators fragment high-water-content parenchyma and relatively spare '
                         'collagen- and elastin-rich vessel walls and membranes.',
            'settings': 'amplitude, aspiration and duty are fractions in [0,1]; irrigation is mL/min',
            'limits': list(LIMITS)}
