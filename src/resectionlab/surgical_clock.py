"""Versioned scheduling of persistent instruments on the engine's physical time.

Revision 1 is explicitly enabled by a replayed configure action. The absent
revision retains historical transitions. This schedules existing research laws;
it does not calibrate them or claim continuous simultaneous tool trajectories.
"""
from __future__ import annotations

import numpy as np

REVISION = 1
MAX_STEP_SECONDS = .1


def enabled(engine):
    return engine.state.get('physicalClockRevision') == REVISION


def configure(engine):
    # The finite-blade experiment has its own prescribed integration and must
    # not advertise a clock it does not execute.
    if 'planarBladeManeuverV1' in engine.scene.properties:
        raise ValueError('SHARED_CLOCK_UNSUPPORTED_PLANAR_MANEUVER')
    if 'retractorSurfaceContactV1' in engine.scene.properties:
        raise ValueError('SHARED_CLOCK_UNQUALIFIED_SURFACE_CONTACT')
    engine.state['physicalClockRevision'] = REVISION
    return {'kind': 'physical_clock_enabled', 'revision': REVISION}


def descriptor():
    return {'revision': REVISION, 'maxStepSeconds': MAX_STEP_SECONDS,
            'effectOrder': ['left', 'right'], 'coupling': 'instrument-effects-then-fields',
            'activation': 'persistent-switch-or-current-action',
            'scope': 'bounded sequential motion with persistent stationary other hand'}


def advance_instruments(engine, key, op, p, q, active, dt, working_cells, eligible):
    """Validate contacts, apply each active tool once, and account contact time.

    Contacts are collected before either effect. Effects execute in stable slot
    order, independent of which hand owns an inspection. Current-tip removal
    queries still see preceding effects in that declared order. Global solvers,
    cooling, fluids, vascular evolution and elapsed time run once in the caller.
    """
    from .surgical_motion import envelopes
    from .surgical_tools import TOOLS

    records = []
    for hand in sorted(engine.state['slots']):
        slot = engine.state['slots'][hand]
        acting = hand == key
        effective = bool(active if acting else slot['active'])
        if not (effective or slot['inserted'] or acting and op not in ('inspect', 'place', 'exchange')):
            continue
        tool = TOOLS[slot['instrument']]
        params = engine.parameters(slot)
        position = p if acting else np.asarray(slot['position'])
        quaternion = q if acting else np.asarray(slot['quaternion'])
        if acting:
            contacts = set(working_cells)
            effect_cells = eligible
        else:
            tip = set(engine.touched(position, position, params['radiusMm']))
            revised = engine.state.get('toolMechanicsRevision') in (1, 2) and tool['effect'] not in ('remove', 'suction', 'irrigate', 'probe')
            contacts = set() if revised else set(tip)
            for a, b, radius, part in envelopes(tool, params, position, quaternion):
                cells = set(engine.touched(a, b, radius))
                if part == 'shaft':
                    if cells - set(engine.state['holds'].get(hand, {}).get('cells', ())):
                        raise ValueError('SHAFT_COLLISION:' + hand)
                else:
                    contacts.update(cells)
            effect_cells = sorted(contacts if revised else tip)
        if effective and tool['materials'] and any(
                engine.by_label[int(engine.scene.labels.flat[cell])]['material'] not in tool['materials']
                for cell in contacts):
            raise ValueError('ACTIVE_TOOL_UNSUPPORTED_MATERIAL:' + hand)
        effect_cells = [cell for cell in effect_cells
                        if engine.by_label[int(engine.scene.labels.flat[cell])]['material'] in tool['materials']]
        records.append((hand, tool, params, position, quaternion, effective, contacts, effect_cells))

    events = []
    contact_cells = set()
    vessels = set()
    v2 = bool(engine.scene.properties.get('tissuePhysicsV2'))
    for hand, tool, params, position, quaternion, effective, contacts, effect_cells in records:
        if effective:
            events.extend(engine._effect(hand, op if hand == key else 'apply', effect_cells, position, quaternion, dt))
        vessel_cells = []
        for cell in sorted(contacts):
            material = engine.by_label[int(engine.scene.labels.flat[cell])]['material']
            engine.state['materialExposure'][material] = engine.state['materialExposure'].get(material, 0.) + dt
            if material == 'vessel': vessel_cells.append(cell)
        vessels.update(vessel_cells)
        if v2:
            from .surgical_vessels_v2 import contact
            contact(engine, vessel_cells, dt, params['forceN'] if effective else 0.)
        if tool['effect'] in ('elevate', 'dissect'):
            contact_cells.update(contacts)
    engine.state['vesselEncounters'] = sorted(set(engine.state['vesselEncounters']) | vessels)
    for hold in engine.state['holds'].values(): contact_cells.update(hold['cells'])
    contact_cells.difference_update(engine.state['removed'])
    for cell in sorted(contact_cells):
        name = str(cell)
        engine.state['contact'][name] = engine.state['contact'].get(name, 0.) + dt
    return events
