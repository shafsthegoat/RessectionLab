"""Opt-in suction as the second hand (``suctionTractionV1``), a research surrogate.

A surgeon's suction does more than clear the field: its tip, resting on brain
or tumour with the vacuum on, holds a small patch so the other hand can work
against gentle traction or counter-traction. Through a cottonoid the same tip
only takes fluid and holds nothing.

* An active Frazier suction whose tip touches soft tissue or tumour, and is
  not over a placed cottonoid, grips the touched cells through the existing
  coupled hold of ``surgical_deformation_v2`` (the brain retractor's
  mechanism). Moving the active suction moves the hold target.
* Hold force = aspiration x maximum vacuum x tip lumen area. The vacuum
  coupling is the hold spring of the lattice: its load is the hold stiffness
  times the gap between where the tip asks the held cells to be and where
  they are, summed over the held cells. When that load exceeds the hold force,
  or the vacuum is switched off, the patch is released. In injury mode a pull
  that tears the patch off records a traction injury of the gripped cells.
* Fluid and debris uptake stay the ordinary suction law.
* With this property declared, every coupled hold (retractor or suction)
  also reports a read-only tissue reaction estimate at the end of an action.
  Dynamics are unchanged by the estimate.

Every number is an assumed research value. No tissue aspiration by the
suction itself, no pial tearing threshold and no vacuum dynamics are modelled.
"""
from __future__ import annotations

import copy
import math

import numpy as np

PROPERTY = 'suctionTractionV1'
SCHEMA = 'resectionlab.suction-traction.v1'
STATE_SCHEMA = 'resectionlab.suction-traction-state.v1'
INSTRUMENT = 'frazier_suction'
ASSUMED = 'assumed-research'
# 1 mmHg = 133.322 Pa.
PA_PER_MMHG = 133.322
PARAMETERS = {
    # Neurosurgical wall suction is commonly run at a fraction of the wall
    # maximum; 200 mmHg at full aspiration is an assumption, not a device spec.
    'maxVacuumMmHg': (200., ASSUMED),
    # Lumen of a fine (about 10 French) Frazier tip; assumed.
    'tipLumenDiameterMm': (2., ASSUMED),
    'defaultAspiration': (.5, ASSUMED),
}
LIMITS = (
    'Hold force is vacuum pressure times lumen area; no seal quality, tissue porosity or tip angle.',
    'The suction never aspirates brain or tumour itself; tissue is only held.',
    'Release is a force threshold on the lattice hold load; no pial tearing mechanics.',
    'The reaction estimate is the lattice hold-spring load, an uncalibrated proxy, not a measured force.',
    'A cottonoid is detected by its recorded centre and radius; its absorbency is not modelled.',
)


def default_profile():
    return {'schema': SCHEMA, **{name: value for name, (value, _) in PARAMETERS.items()},
            'provenance': {'kind': ASSUMED, 'description':
                'Uncalibrated suction hold surrogate; vacuum and lumen are engineering assumptions.'}}


def parameter_sources():
    return {name: source for name, (_, source) in PARAMETERS.items()}


def validate(scene):
    if PROPERTY not in scene.properties:
        return None
    value = scene.properties[PROPERTY]
    defaults = default_profile()
    if not isinstance(value, dict) or set(value) - set(defaults) or value.get('schema') != SCHEMA:
        raise ValueError('SUCTION_TRACTION_PROFILE_INVALID: schema and known fields required')
    result = {**defaults, **copy.deepcopy(value)}
    for name in PARAMETERS:
        number = result[name]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError('SUCTION_TRACTION_PROFILE_INVALID:' + name)
    if result['defaultAspiration'] > 1:
        raise ValueError('SUCTION_TRACTION_PROFILE_INVALID:defaultAspiration')
    provenance = result['provenance']
    if not isinstance(provenance, dict) or provenance.get('kind') != ASSUMED or not str(provenance.get('description', '')).strip():
        raise ValueError('SUCTION_TRACTION_PROFILE_INVALID:provenance')
    if not scene.properties.get('tissuePhysicsV2'):
        raise ValueError('SUCTION_TRACTION_REQUIRES_TISSUE_PHYSICS_V2')
    return result


def profile(engine):
    cached = engine.__dict__.get('_suction_traction_profile')
    if cached is not None and cached[0] is engine.scene.properties.get(PROPERTY):
        return cached[1]
    result = validate(engine.scene)
    engine.__dict__['_suction_traction_profile'] = (engine.scene.properties.get(PROPERTY), result)
    return result


def enabled(engine):
    return PROPERTY in engine.scene.properties


def initialize(engine):
    if not enabled(engine):
        return
    profile(engine)
    engine.state.setdefault(PROPERTY, {'schema': STATE_SCHEMA, 'holds': {}, 'releases': [], 'regripBlocked': {},
                                       'throughCottonoidSeconds': 0., 'peakLoadN': 0.})


def hold_force_n(engine, aspiration):
    """Vacuum x lumen area: Pa x mm2 x 1e-6 = N."""
    p = profile(engine)
    area = math.pi * (p['tipLumenDiameterMm'] / 2.) ** 2
    return float(aspiration) * p['maxVacuumMmHg'] * PA_PER_MMHG * area * 1e-6


def aspiration(engine, params):
    value = params.get('aspiration', profile(engine)['defaultAspiration'])
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('INVALID_SUCTION_ASPIRATION')
    return float(value)


def cottonoid_over(engine, position):
    """The placed cottonoid whose recorded patch covers the tip, or None (read-only)."""
    objects = engine.state.get('operativeExposureV1', {}).get('objects', [])
    margin = .5 * float(np.linalg.norm(engine.scene.spacing))
    for obj in objects:
        if obj.get('kind') != 'cottonoid' or obj.get('heldBy') is not None:
            continue
        if obj.get('status') not in ('placed', 'intended-retained'):
            continue
        centre = np.asarray(obj['positionMm'], float)
        if np.linalg.norm(np.asarray(position, float) - centre) <= float(obj['radiusMm']) + margin:
            return obj
    return None


def _soft_contacts(engine, position, radius):
    removed = set(engine.state['removed'])
    held = {c for hold in engine.state['holds'].values() for c in hold['cells']}
    protected = getattr(engine, '_protected_ids', frozenset())
    cells = []
    for cell in sorted(map(int, engine.touched(position, position, radius))):
        label = int(engine.scene.labels.flat[cell])
        if (label and cell not in removed and cell not in held and cell not in protected
                and engine.by_label[label]['material'] in ('soft_tissue', 'tumor')):
            cells.append(cell)
    return cells


def apply(engine, key, op, cells, p, q, dt):
    """Active Frazier suction: grip tissue unless over a cottonoid, then the ordinary fluid uptake."""
    from .surgical_effects import apply_effect
    initialize(engine)
    state = engine.state[PROPERTY]
    slot = engine.state['slots'][key]
    params = engine.parameters(slot)
    level = aspiration(engine, params)
    events = []
    cover = cottonoid_over(engine, p)
    if cover is not None:
        state['throughCottonoidSeconds'] += float(dt)
        events.append({'kind': 'suction_through_cottonoid', 'objectId': cover['id']})
    elif key not in engine.state['holds'] and level > 0:
        contacts = _soft_contacts(engine, p, params['radiusMm'])
        blocked = state['regripBlocked'].get(key)
        if blocked is not None and (not contacts or blocked != len(engine.actions)):
            # A torn-off patch is not re-gripped in the same action until the tip has lost contact.
            del state['regripBlocked'][key]
            blocked = None
        if contacts and blocked is None:
            from .surgical_deformation_v2 import begin_hold
            begin_hold(engine, key, contacts, p)
            hold = engine.state['holds'].get(key)
            if hold is not None:
                force = hold_force_n(engine, level)
                state['holds'][key] = {'cells': hold['cells'][:], 'holdForceN': force,
                                       'aspiration': level, 'peakLoadN': 0.}
                events.append({'kind': 'suction_hold_created', 'cells': len(hold['cells']), 'holdForceN': force})
    events.extend(apply_effect(engine, key, op, cells, p, q, dt))
    return events


def hold_load(engine, key):
    """Load (N, vector) of one coupled hold spring: stiffness x (asked - current displacement)."""
    from .surgical_deformation_v2 import _settings
    hold = engine.state['holds'][key]
    settings = _settings(engine)
    delta = np.subtract(hold.get('target', hold['origin']), hold['origin'])
    total = np.zeros(3)
    deformation = engine.state.get('deformation', {})
    removed = set(engine.state['removed'])
    for cell in hold['cells']:
        if cell in removed:
            continue
        asked = delta + np.asarray(hold.get('baseDisplacements', {}).get(str(cell), [0., 0., 0.]))
        total += settings['holdStiffnessNPerMm'] * (asked - np.asarray(deformation.get(str(cell), [0., 0., 0.])))
    return total


def after_tick(engine, key):
    """Check suction holds after the lattice step; release on vacuum off or overload."""
    state = engine.state.get(PROPERTY)
    if not state or not state['holds']:
        return []
    events = []
    for hand in sorted(state['holds']):
        row = state['holds'][hand]
        hold = engine.state['holds'].get(hand)
        slot = engine.state['slots'][hand]
        if hold is None or 'target' not in hold or slot['instrument'] != INSTRUMENT:
            del state['holds'][hand]
            continue
        load = float(np.linalg.norm(hold_load(engine, hand)))
        row['peakLoadN'] = max(row['peakLoadN'], load)
        state['peakLoadN'] = max(state['peakLoadN'], load)
        reason = 'vacuum-off' if not slot['active'] else 'hold-force-exceeded' if load > row['holdForceN'] else None
        if reason is None:
            continue
        cells = [c for c in hold['cells'] if c not in set(engine.state['removed'])]
        del engine.state['holds'][hand]
        slot['grip'] = []
        del state['holds'][hand]
        state['releases'].append({'slot': hand, 'reason': reason, 'loadN': load, 'holdForceN': row['holdForceN'],
                                  'cells': len(cells), 'timeSeconds': float(engine.state['elapsedSeconds'])})
        events.append({'kind': 'suction_hold_released', 'slot': hand, 'reason': reason,
                       'loadN': load, 'holdForceN': row['holdForceN']})
        if reason == 'hold-force-exceeded':
            state['regripBlocked'][hand] = len(engine.actions)
            from . import surgical_injury as injury
            if injury.converting(engine):
                injury.tear(engine, 'SUCTION_HOLD_EXCEEDED: traction tore the suction patch free', hand,
                            cells, 'suction-patch:' + hand)
    return events


def reaction_estimates(engine):
    """Read-only tissue reaction on each coupled hold (retractor or suction), in newtons."""
    rows = []
    for hand in sorted(engine.state['holds']):
        hold = engine.state['holds'][hand]
        if 'target' not in hold:
            continue
        load = hold_load(engine, hand)
        slot = engine.state['slots'][hand]
        rows.append({'kind': 'tissue_reaction_estimate', 'slot': hand, 'instrument': slot['instrument'],
                     'reactionN': [float(v) for v in -load], 'magnitudeN': float(np.linalg.norm(load)),
                     **({'inputForceN': float(engine.parameters(slot)['forceN'])}
                        if 'forceN' in engine.parameters(slot) else {}),
                     'basis': 'lattice hold-spring load; uncalibrated proxy'})
    return rows


def metrics(engine):
    state = engine.state.get(PROPERTY, {})
    return {'suctionHolds': len(state.get('holds', {})), 'suctionReleases': len(state.get('releases', [])),
            'suctionThroughCottonoidSeconds': state.get('throughCottonoidSeconds', 0.),
            'suctionPeakLoadN': state.get('peakLoadN', 0.)}


def descriptor(engine):
    return {'enabled': enabled(engine), 'profile': copy.deepcopy(profile(engine)),
            'sources': parameter_sources(), 'holdForceAtFullAspirationN': hold_force_n(engine, 1.),
            'limits': list(LIMITS)}
