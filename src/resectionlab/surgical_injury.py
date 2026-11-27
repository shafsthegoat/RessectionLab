"""Opt-in injury mode: show what an unsafe instrument move does to tissue.

Guarded mode refuses a physically unsafe microstep and leaves tissue
untouched. A scene that declares ``surgicalInjuryV1`` with ``mode: 'injury'``
instead records the consequence and lets the motion continue where that is
physically meaningful:

* a shaft or other non-cutting part forced into soft tissue, tumour, membrane
  or vessel accumulates a contusion dose (contact seconds x force proxy); a
  cell whose dose passes the threshold is ``contused`` and stays in place; a
  cell the rigid part travels through beyond the laceration distance is
  ``lacerated`` and removed, the removal attributed to injury;
* a cutting or aspirating working part on material it is not meant to treat
  (CUSA or scalpel on a vessel, drill or craniotome on dura or brain) lacerates
  the contacted cells at once; vessel cells are recorded by id as vessel
  injury so a bleeding model reading ``state['removed']`` can open a wound;
* a forceps pull against an intact tumour attachment avulses it;
* contact with a declared protected structure is recorded against that named
  structure instead of refusing the move.

Bone, unknown material, fluid labels, tool-to-tool and fixture collisions stay
refusals: the model has nothing meaningful to injure there. Sequence and
input-validity refusals are guards and are never converted.

Every threshold is an assumed research value, not a calibrated tissue
tolerance. There is no oedema, delayed deficit or functional outcome here.
"""
from __future__ import annotations

import math

PROPERTY = 'surgicalInjuryV1'
SCHEMA = 'resectionlab.surgical-injury.v1'
STATE_SCHEMA = 'resectionlab.surgical-injury-state.v1'
MODES = ('injury', 'guarded')
INJURABLE = frozenset(('soft_tissue', 'tumor', 'membrane', 'vessel', 'functional_region'))
SHARP_EFFECTS = frozenset(('remove', 'incise', 'transect', 'kerf', 'sheet_cut'))
BONE_TOOLS = frozenset(('cranial_drill', 'craniotome'))
BLUNT_CODES = frozenset(('SHAFT_COLLISION', 'RETAINED_TISSUE_COLLISION', 'WITHDRAWAL_PATH_BLOCKED',
                         'POST_DEFORMATION_SHAFT_COLLISION'))
MATERIAL_CODES = frozenset(('UNSUPPORTED_MATERIAL', 'OTHER_ACTIVE_TOOL_UNSUPPORTED_MATERIAL'))
AVULSION_CODES = frozenset(('ATTACHMENT_PREVENTS_DISPLACEMENT', 'SEPARATE_ATTACHMENTS_BEFORE_DISPLACEMENT'))
_NUMBERS = ('contusionDoseThresholdNs', 'lacerationTravelMm', 'defaultForceN')


def default_profile():
    """The one source of injury-mode defaults; all values are assumed research."""
    return {'schema': SCHEMA, 'mode': 'injury',
            # Contact seconds x force proxy (N*s) a cell must take before it is
            # recorded as contused. Assumed; no cited tolerance.
            'contusionDoseThresholdNs': .05,
            # Travel (mm) of a rigid part through one cell before the cell is
            # torn through and removed. Assumed: about one to two voxels.
            'lacerationTravelMm': 1.5,
            # Force proxy when the scene has no per-instrument forceN parameter.
            'defaultForceN': 1.,
            'provenance': {'evidenceClass': 'assumed-research', 'calibration': 'none',
                           'statement': 'Engineering thresholds chosen to make unsafe motion visible; '
                                        'not measured tissue tolerances and not a clinical injury model.'}}


def profile(scene):
    """Validated injury profile, or None when the scene does not declare one."""
    declared = scene.properties.get(PROPERTY)
    if PROPERTY not in scene.properties:
        return None
    defaults = default_profile()
    if (not isinstance(declared, dict) or set(declared) - set(defaults)
            or declared.get('schema') != SCHEMA or declared.get('mode') not in MODES):
        raise ValueError('SURGICAL_INJURY_PROFILE_INVALID: schema, mode and known thresholds required')
    result = {**defaults, **declared}
    for name in _NUMBERS:
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('SURGICAL_INJURY_PROFILE_INVALID: ' + name + ' must be a positive number')
    if not isinstance(result['provenance'], (dict, str)) or not result['provenance']:
        raise ValueError('SURGICAL_INJURY_PROFILE_INVALID: provenance required')
    return result


def initialize(engine):
    engine.state['injuryV1'] = {
        'schema': STATE_SCHEMA, 'contusionDoseNs': {}, 'contactTravelMm': {},
        'contused': [], 'lacerated': [], 'vesselInjured': [], 'avulsions': [],
        'protectedCells': {}, 'events': [],
        'totalsMm3': {'contused': 0., 'lacerated': 0., 'vessel': 0.}, 'byRegionMm3': {}}


def converting(engine):
    """True when unsafe physical motion is recorded instead of refused."""
    settings = engine.__dict__.get('_injury')
    return settings is not None and settings['mode'] == 'injury'


def _material(engine, cell):
    return engine.by_label[int(engine.scene.labels.flat[cell])]['material'] if engine.scene.labels.flat[cell] else None


def _structures(engine, cells):
    if not engine._protected_ids or not cells:
        return []
    cells = set(cells)
    return sorted({a.structure_id for a in engine.protection.annotations if cells.intersection(a.cells)})


def _regions(engine, cells):
    return sorted({engine.by_label[int(engine.scene.labels.flat[c])]['name'] for c in cells
                   if engine.scene.labels.flat[c]})


def _refresh(engine, record):
    volume = float(engine.scene.voxel_mm3)
    by_region = {}
    for kind, name in (('contused', 'contused'), ('lacerated', 'lacerated'), ('vessel', 'vesselInjured')):
        for cell in record[name]:
            region = engine.by_label[int(engine.scene.labels.flat[cell])]['name']
            row = by_region.setdefault(region, {'contused': 0., 'lacerated': 0., 'vessel': 0.})
            row[kind] += volume
        record['totalsMm3'][kind] = len(record[name]) * volume
    record['byRegionMm3'] = {name: by_region[name] for name in sorted(by_region)}


def _event(engine, record, kind, code, cells, key, *, cell_ids=False, **extra):
    slot = engine.state['slots'][key]
    event = {'kind': kind, 'code': code, 'cells': len(cells), 'structures': _structures(engine, cells),
             'regions': _regions(engine, cells), 'timeSeconds': float(engine.state['elapsedSeconds']),
             'slot': key, 'instrument': slot['instrument'], **extra}
    if cell_ids:
        event['cellIds'] = sorted(cells)
    record['events'].append(event)


def _remove(engine, record, cells):
    cells = set(cells)
    engine.state['removed'] = sorted(set(engine.state['removed']).union(cells))
    record['lacerated'] = sorted(set(record['lacerated']).union(cells))
    record['contused'] = [c for c in record['contused'] if c not in cells]
    vessels = [c for c in cells if _material(engine, c) == 'vessel']
    if vessels:
        record['vesselInjured'] = sorted(set(record['vesselInjured']).union(vessels))
    return vessels


def admit(engine, code, cells, key, op, dt, params, *, active=False, travel_mm=0.):
    """Record an unsafe physical contact, or raise exactly as guarded mode does.

    ``code`` is the refusal message the engine would raise. ``cells`` are the
    material cells that caused it and ``key`` the hand whose instrument made the
    contact. Without injury mode, or for rigid contact, this raises ``code``.
    """
    reason = code.split(':', 1)[0]
    if not converting(engine) or reason not in BLUNT_CODES | MATERIAL_CODES:
        raise ValueError(code)
    record = engine.state['injuryV1']
    settings = engine._injury
    seen = engine.__dict__.setdefault('_injury_seen', set())
    removed = set(engine.state['removed'])
    cells = sorted({int(c) for c in cells} - removed - seen)
    if not cells:
        return
    if any(_material(engine, c) not in INJURABLE for c in cells):
        raise ValueError(code)
    seen.update(cells)
    slot = engine.state['slots'][key]
    tool = engine.tools[slot['instrument']]
    sharp = reason in MATERIAL_CODES and tool['effect'] in SHARP_EFFECTS and (active or reason != 'UNSUPPORTED_MATERIAL' or tool['geometry'] == 'blade')
    if sharp:
        vessels = _remove(engine, record, cells)
        kind = 'plunge' if slot['instrument'] in BONE_TOOLS else 'laceration'
        _event(engine, record, kind, reason, cells, key, mechanism='sharp')
        if vessels:
            _event(engine, record, 'vessel_injury', reason, vessels, key, cell_ids=True, mechanism='sharp')
        _refresh(engine, record)
        return
    force = float(params.get('forceN', settings['defaultForceN']))
    dose, travel = record['contusionDoseNs'], record['contactTravelMm']
    newly_contused, torn = [], []
    contused = set(record['contused'])
    for cell in cells:
        name = str(cell)
        dose[name] = dose.get(name, 0.) + max(0., float(dt)) * force
        travel[name] = travel.get(name, 0.) + max(0., float(travel_mm))
        if travel[name] >= settings['lacerationTravelMm'] - 1e-12:
            torn.append(cell)
        elif dose[name] >= settings['contusionDoseThresholdNs'] - 1e-12 and cell not in contused:
            newly_contused.append(cell)
    if newly_contused:
        record['contused'] = sorted(contused.union(newly_contused))
        _event(engine, record, 'contusion', reason, newly_contused, key, mechanism='blunt')
    if torn:
        vessels = _remove(engine, record, torn)
        _event(engine, record, 'laceration', reason, torn, key, mechanism='blunt')
        if vessels:
            _event(engine, record, 'vessel_injury', reason, vessels, key, cell_ids=True, mechanism='blunt')
    if newly_contused or torn:
        _refresh(engine, record)


def avulse(engine, code, key):
    """Break the intact attachments a forceps pull is tearing, or raise ``code``."""
    reason = code.split(':', 1)[0]
    if not converting(engine) or reason not in AVULSION_CODES:
        raise ValueError(code)
    record = engine.state['injuryV1']
    hold = engine.state['holds'][key]
    attached = [a for a in engine.scene.properties.get('attachments', [])
                if set(a['cells']) & set(hold['cells']) and a['id'] not in engine.state['separatedAttachments']]
    if not attached:
        raise ValueError(code)
    torn = sorted({int(c) for a in attached for c in a['cells']} - set(engine.state['removed']))
    engine.state['separatedAttachments'] = sorted(set(engine.state['separatedAttachments']) | {a['id'] for a in attached})
    record['avulsions'] = sorted(set(record['avulsions']) | {a['id'] for a in attached})
    record['contused'] = sorted((set(record['contused']) | set(torn)) - set(record['lacerated']))
    _event(engine, record, 'avulsion', reason, torn, key, attachments=sorted(a['id'] for a in attached),
           mechanism='traction')
    _refresh(engine, record)


def tear(engine, code, key, cells, structure):
    """Record a traction tear of a named hinge or tether, or raise ``code``.

    For traction limits owned by other modules (flap hinge, vessel tether):
    the caller breaks its own coupling after this returns.
    """
    if not converting(engine):
        raise ValueError(code)
    record = engine.state['injuryV1']
    cells = sorted({int(c) for c in cells} - set(engine.state['removed']))
    record['avulsions'] = sorted(set(record['avulsions']) | {str(structure)})
    record['contused'] = sorted((set(record['contused']) | set(cells)) - set(record['lacerated']))
    _event(engine, record, 'avulsion', code.split(':', 1)[0], cells, key, attachments=[str(structure)],
           mechanism='traction')
    _refresh(engine, record)


def protected(engine, message, cells, key):
    """Record injury to a named protected structure, or raise the guard refusal."""
    if not converting(engine):
        raise ValueError(message)
    record = engine.state['injuryV1']
    mechanism = message.split(':', 1)[1].strip() if ':' in message else message
    known = record['protectedCells'].setdefault(mechanism, [])
    new = sorted({int(c) for c in cells} & engine._protected_ids - set(known))
    if not new:
        return
    record['protectedCells'][mechanism] = sorted(set(known) | set(new))
    kind = 'protected_clearance' if 'clearance' in mechanism else 'protected_structure_injury'
    _event(engine, record, kind, 'PROTECTED_STRUCTURE', new, key, mechanism=mechanism)


def metrics(engine):
    record = engine.state['injuryV1']
    return {'injuryContusedMm3': record['totalsMm3']['contused'],
            'injuryLaceratedMm3': record['totalsMm3']['lacerated'],
            'injuryVesselCells': len(record['vesselInjured']),
            'injuryAvulsions': len(record['avulsions']),
            'injuryEvents': len(record['events']),
            'injuryByRegionMm3': {name: dict(row) for name, row in record['byRegionMm3'].items()},
            'injuryProtectedStructures': sorted({s for e in record['events'] for s in e['structures']})}
