"""Bounded declared options from one published observation, without execution.

Ordering and option identities belong to the caller. Nominal proposals neither
rank alternatives nor assert feasibility, cleanup, success, or task completion.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .core import array_digest, semantic_digest
from .surgical_rl_env import (BUDGET_METRICS, CLEANUP_METRICS, INSTRUMENTS,
    OBS_METRICS, PARAMETERS, PARAMETER_HIGH, PARAMETER_LOW)
from .surgical_skills import SKILLS, SkillTask, propose_skill

SCHEMA = 'resectionlab.surgical-skill-options.v1'
MAX_OPTIONS = 32
MAX_REQUEST_JSON_BYTES = 64 * 1024
MAX_PRIMITIVES_PER_OPTION = 6
MAX_TOTAL_PRIMITIVES = 192
MAX_RESPONSE_JSON_BYTES = 256 * 1024
_FEASIBILITY = 'unknown_until_public_execution'
_FIELDS = frozenset(('id', 'skill', 'slot', 'pose', 'parameters', 'duration', 'instrument'))
# Shapes/dtypes/Box bounds match SurgicalRLEnv's published v2 observation space.
_OBSERVATION = {
    'local': ((2, 8, 9, 9, 9), np.dtype('float32'), 0., 1.),
    'slots': ((2, 12), np.dtype('float32'), -1e6, 1e6),
    'tool_parameters': ((2, len(PARAMETERS)), np.dtype('float64'), PARAMETER_LOW, PARAMETER_HIGH),
    'metrics': ((len(OBS_METRICS),), np.dtype('float64'), 0., math.inf),
    'goal': ((4,), np.dtype('float64'), -1e6, 1e6),
    'coverage': ((3,), np.dtype('int8'), 0, 1),
    'local_spacing_mm': ((1,), np.dtype('float64'), .1, 10.),
    'cleanup': ((len(CLEANUP_METRICS),), np.dtype('float64'), 0., math.inf),
    'budgets': ((len(BUDGET_METRICS),), np.dtype('float64'), 0., 1.),
}


def _number(value, label):
    finite = False
    if type(value) in (int, float):
        try:finite = math.isfinite(value)
        except OverflowError:pass
    if not finite:
        raise ValueError(label + ' must be a finite JSON number')


def _string(value, label):
    if type(value) is not str or not value.strip() or len(value) > MAX_REQUEST_JSON_BYTES:
        raise ValueError(label + ' must be a nonempty bounded JSON string')


def _object(value, fields, label):
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) - fields:
        raise ValueError(label + ' has unknown fields or is not a JSON object')


def _serialized(value, limit, label):
    try:
        text = json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + ' must be finite JSON') from error
    if len(text.encode('utf-8')) > limit:
        raise ValueError(label + ' exceeds its serialized UTF-8 byte limit')
    return text


def _requests(requests, max_action_seconds):
    if type(requests) is not list or not 1 <= len(requests) <= MAX_OPTIONS:
        raise ValueError('Declare an ordered JSON list of at most 32 options')
    ids, controls = set(), set()
    stop = False
    for request in requests:
        _object(request, _FIELDS, 'Skill request')
        if not {'id', 'skill'} <= set(request):
            raise ValueError('Each skill request needs id and skill')
        _string(request['id'], 'Option id')
        if request['id'] in ids:
            raise ValueError('Option ids must be unique')
        ids.add(request['id'])
        if type(request['skill']) is not str or request['skill'] not in SKILLS:
            raise ValueError('Unknown declared skill')
        if 'slot' in request and (type(request['slot']) is not str or request['slot'] not in ('left', 'right')):
            raise ValueError('Unknown declared slot')
        duration = request.get('duration', .1)
        _number(duration, 'Skill duration')
        if not .01 <= duration <= min(30., max_action_seconds):
            raise ValueError('Skill duration exceeds the explicit action duration limit')
        if 'instrument' in request and (type(request['instrument']) is not str or request['instrument'] not in INSTRUMENTS):
            raise ValueError('Unknown declared instrument')
        if 'parameters' in request:
            _object(request['parameters'], frozenset(PARAMETERS), 'Skill parameters')
            for value in request['parameters'].values():
                _number(value, 'Skill parameter')
        if 'pose' in request:
            pose = request['pose']
            _object(pose, frozenset(('position', 'quaternion')), 'Skill pose')
            if set(pose) != {'position', 'quaternion'}:
                raise ValueError('Skill pose needs position and quaternion')
            for name, size in (('position', 3), ('quaternion', 4)):
                if type(pose[name]) is not list or len(pose[name]) != size:
                    raise ValueError('Skill pose arrays have invalid shape')
                for value in pose[name]:
                    _number(value, 'Skill pose coordinate')
            if math.hypot(*pose['quaternion']) < 1e-12:
                raise ValueError('Skill pose quaternion must be nonzero')
        plain_control = not {'pose', 'parameters', 'instrument'}.intersection(request)
        if request['skill'] in ('release', 'withdraw') and 'slot' in request and plain_control:
            controls.add((request['skill'], request['slot']))
        if request['skill'] == 'stop' and plain_control:
            stop = True
    required = {(skill, slot) for skill in ('release', 'withdraw') for slot in ('left', 'right')}
    if controls != required or not stop:
        raise ValueError('Declare route/parameter/instrument-free release and withdraw for both explicit slots and stop')
    # The schema is now flat bounded JSON: no arbitrary recursive containers,
    # ndarray coercions, custom objects, or non-string keys can reach encoding.
    return json.loads(_serialized(requests, MAX_REQUEST_JSON_BYTES, 'Skill requests'))


def _observation(observation):
    if (type(observation) is not dict or any(type(key) is not str for key in observation)
            or set(observation) != set(_OBSERVATION)):
        raise ValueError('Expected only the published surgical RL v2 observation fields')
    copied = {}
    for key, (shape, dtype, low, high) in _OBSERVATION.items():
        value = observation[key]
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != dtype:
            raise ValueError('Published observation shape or dtype differs: ' + key)
        value = value.copy(order='C')
        if not np.isfinite(value).all() or np.any(value < low) or np.any(value > high):
            raise ValueError('Published observation bounds differ: ' + key)
        value.setflags(write=False)
        copied[key] = value
    # These are categorical/count columns emitted by _observe, not a mask
    # predicting whether any requested transition will succeed.
    slots = copied['slots']
    if (np.any(slots[:, 7] != np.floor(slots[:, 7])) or np.any(slots[:, 7] < 0)
            or np.any(slots[:, 7] >= len(INSTRUMENTS))
            or np.any((slots[:, 8:11] != 0) & (slots[:, 8:11] != 1))
            or np.any(slots[:, 11] < 0) or np.any(slots[:, 11] != np.floor(slots[:, 11]))):
        raise ValueError('Published slot categories or counts differ')
    return copied


def build_skill_options(observation, requests, *, task: SkillTask, max_action_seconds: float):
    """Return every declared option in order; never choose or execute one.

    Malformed declarations fail as a set before proposing anything. A valid
    declaration can still yield a nominal refusal, retained as a row rather
    than silently removed. Explicit action seconds are independent of the
    observation's normalized episode budgets. Serialization caps use compact,
    ensure_ascii=True JSON encoded as UTF-8.
    """
    if type(task) is not SkillTask:
        raise TypeError('Explicit SkillTask required')
    # Revalidate the declaration without deriving or constructing an env/config.
    declared = SkillTask(task.name, task.mode)
    _number(max_action_seconds, 'max_action_seconds')
    if not .01 <= max_action_seconds <= 30.:
        raise ValueError('max_action_seconds must be in [0.01, 30]')
    requests = _requests(requests, max_action_seconds)
    observed = _observation(observation)
    observation_id = semantic_digest({key: array_digest(value) for key, value in observed.items()})
    declaration = {'name': declared.name, 'mode': declared.mode}
    limits = {'max_options': MAX_OPTIONS, 'max_request_json_bytes': MAX_REQUEST_JSON_BYTES,
        'max_primitives_per_option': MAX_PRIMITIVES_PER_OPTION, 'max_total_primitives': MAX_TOTAL_PRIMITIVES,
        'max_response_json_bytes': MAX_RESPONSE_JSON_BYTES, 'max_action_seconds': float(max_action_seconds)}
    request_set_id = semantic_digest({'schema': SCHEMA, 'task': declaration, 'limits': limits, 'requests': requests})
    rows, total = [], 0
    for request in requests:
        try:
            plan = propose_skill(observed, request['skill'], **{key: value for key, value in request.items() if key not in ('id', 'skill')})
        except ValueError as error:
            rows.append({'id': request['id'], 'request': request, 'status': 'rejected_nominal_proposal',
                'reason': str(error), 'plan': None, 'feasibility': _FEASIBILITY})
            continue
        if plan.get('observationId') != observation_id or type(plan.get('actions')) is not list:
            raise ValueError('Nominal proposal changed its observation or action contract')
        count = len(plan['actions'])
        if count > MAX_PRIMITIVES_PER_OPTION or total + count > MAX_TOTAL_PRIMITIVES:
            raise ValueError('Nominal primitives exceed option or total cap')
        if any(type(action) is not dict or type(action.get('duration')) not in (int, float)
               or not math.isfinite(action['duration']) or not .01 <= action['duration'] <= max_action_seconds
               for action in plan['actions']):
            raise ValueError('Nominal primitive exceeds explicit action duration limit')
        total += count
        rows.append({'id': request['id'], 'request': request, 'status': 'nominal_proposal',
            'reason': None, 'plan': plan, 'feasibility': _FEASIBILITY})
    result = {'schema': SCHEMA, 'observationId': observation_id, 'requestSetId': request_set_id,
        'task': declaration, 'limits': limits, 'primitiveCount': total, 'options': rows}
    return json.loads(_serialized(result, MAX_RESPONSE_JSON_BYTES, 'Skill options response'))
