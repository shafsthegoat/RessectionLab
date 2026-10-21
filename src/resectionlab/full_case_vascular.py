"""Public-action peripheral vessel complication for the supplied sub-06 scene.

These are software test access poses, not a tumor vascular dissection plan or
clinical guidance. The source segmentation and declared physics remain intact.
Every action must still pass the engine's current geometry and guard checks.
"""
from __future__ import annotations

import copy


SCENE_ID = 'sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856'
START = [114.53500366210938, -118.558837890625, -533.25]
CONTACT = [33.535003662109375, -118.558837890625, -533.25]
FIELD = [35.535003662109375, -118.558837890625, -533.25]
QUATERNION = [.5, .5, -.5, -.5]


def _action(operation, duration=.2, **values):
    return {'slot': 'left', 'operation': operation, 'duration': duration, **values}


def _pose(position):
    return {'position': list(position), 'quaternion': list(QUATERNION)}


def _cleanup(engine):
    # Stop both instruments before release/withdrawal advances the shared clock.
    actions = [{'slot': slot, 'operation': 'deactivate', 'duration': .2}
               for slot in ('left', 'right') if engine.state['slots'][slot]['active']]
    for slot in ('left', 'right'):
        state = engine.state['slots'][slot]
        if state['grip'] or slot in engine.state['holds']:
            actions.append({'slot': slot, 'operation': 'release', 'duration': .2})
        if state['inserted']:
            actions.append({'slot': slot, 'operation': 'withdraw', 'duration': .2})
    return actions


def rescue_plan(engine):
    """Return phase metadata and actions without changing the supplied engine.

    Accepts the exact saved standardized environment, including after a common
    skull/access/removal prefix. Current hand cleanup is generated from observed
    slot state. A changed route can still reject at execution; no collision or
    protection bypass is supplied by this plan.
    """
    if engine.scene.identity != SCENE_ID:
        raise ValueError('FULL_CASE_VASCULAR_REQUIRES_EXACT_STANDARD_SUB06_SCENE')
    if engine.state.get('vesselsV2', {}).get('faces'):
        raise ValueError('FULL_CASE_VASCULAR_REQUIRES_INITIALLY_UNINJURED_VESSELS')
    phases = [
        {'id': 'injury', 'label': 'Peripheral supplied vessel transection',
         'actions': _cleanup(engine) + [
             _action('exchange', instrument='micro_scissors'),
             _action('place', pose=_pose(START), parameters={
                 'radiusMm': 1.5, 'depthMm': 2., 'jawOpeningMm': 0., 'forceN': 0.}),
             _action('apply', 1., pose=_pose(CONTACT)),
             _action('deactivate')]},
        {'id': 'untreated_bleeding', 'label': 'Untreated wound persists during idle time',
         'actions': [_action('inspect', 2.), _action('inspect', 5.)]},
        {'id': 'source_control', 'label': 'Bipolar seal preserves severed nonpatency',
         'actions': [
             _action('withdraw'), _action('exchange', instrument='bipolar_forceps'),
             _action('place', pose=_pose(START), parameters={
                 'radiusMm': 2., 'jawOpeningMm': 0., 'power': 1., 'forceN': 0.}),
             _action('apply', 1., pose=_pose(CONTACT)), _action('apply', 3.),
             _action('deactivate'), _action('inspect', 5.), _action('withdraw')]},
        {'id': 'field_clearance', 'label': 'Irrigation and local aspiration with a conserved fluid budget',
         'actions': []},
        {'id': 'reevaluation', 'label': 'Idle reassessment of persistent source control',
         'actions': [_action('inspect', 5.)]},
    ]
    for tool in ('irrigation', 'frazier_suction'):
        phases[3]['actions'].extend([
            _action('exchange', instrument=tool),
            _action('place', pose=_pose(START), parameters={
                'radiusMm': 3., 'depthMm': .1, 'flowMm3PerSecond': 2., 'forceN': 0.}),
            _action('apply', 1., pose=_pose(FIELD)), _action('apply', 2.),
            _action('deactivate'), _action('withdraw'),
        ])
    return {'schema': 'resectionlab.full-case-vascular-plan.v1',
            'scenarioId': 'sub06-peripheral-supplied-vessel-complication',
            'label': 'Peripheral supplied vessel complication and rescue',
            'sceneId': SCENE_ID, 'phases': phases,
            'actions': [copy.deepcopy(action) for phase in phases for action in phase['actions']],
            'limits': ['Peripheral supplied vessel access, separate from the tumor approach.',
                       'Declared uncalibrated wound and fluid model; no patient blood-loss prediction.',
                       'Local aspiration clears reachable fluid and does not repair the severed vessel.']}


def rescue_actions(engine):
    """Return the concrete public actions for :func:`rescue_plan`."""
    return rescue_plan(engine)['actions']
