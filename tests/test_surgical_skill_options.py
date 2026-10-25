"""Declared-option adapter tests using arrays only; no engine or learning runs."""
import copy
import json
import math

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from resectionlab.core import array_digest
from resectionlab.surgical_rl_env import (BUDGET_METRICS, CLEANUP_METRICS,
    OBS_METRICS, PARAMETERS, PARAMETER_LOW)
from resectionlab.surgical_skills import SkillTask, propose_skill
from resectionlab import surgical_skill_options as options


def observation():
    slots = np.zeros((2, 12), np.float32)
    slots[:, 6] = 1.
    parameters = np.tile(PARAMETER_LOW, (2, 1))
    parameters[:, 0] = 2.
    return {'local': np.zeros((2, 8, 9, 9, 9), np.float32), 'slots': slots,
        'tool_parameters': parameters, 'metrics': np.zeros(len(OBS_METRICS), np.float64),
        'goal': np.array([0., 0., 0., 1.], np.float64), 'coverage': np.array([1, 0, 0], np.int8),
        'local_spacing_mm': np.array([2.], np.float64),
        'cleanup': np.zeros(len(CLEANUP_METRICS), np.float64),
        'budgets': np.ones(len(BUDGET_METRICS), np.float64)}


def controls():
    return [{'id': skill + ':' + slot, 'skill': skill, 'slot': slot}
        for skill in ('release', 'withdraw') for slot in ('left', 'right')] + [{'id': 'stop', 'skill': 'stop'}]


def pose():
    return {'position': [-0., 2., 3.], 'quaternion': [0., 0., 0., 1.]}


def build(obs=None, requests=None, **values):
    return options.build_skill_options(observation() if obs is None else obs,
        controls() if requests is None else requests, task=values.get('task', SkillTask('access')),
        max_action_seconds=values.get('max_action_seconds', 5.))


def test_ordered_options_are_deterministic_exact_helper_plans_without_selection_or_success():
    obs = observation()
    requests = controls() + [{'id': 'declared-route', 'skill': 'approach', 'pose': pose()}]
    result = build(obs, requests)
    assert result == build(obs, requests)
    assert result['schema'] == options.SCHEMA
    assert [row['id'] for row in result['options']] == [row['id'] for row in requests]
    for request, row in zip(requests, result['options']):
        assert row['request'] == request and row['status'] == 'nominal_proposal'
        expected = propose_skill(obs, request['skill'], **{key: value for key, value in request.items() if key not in ('id', 'skill')})
        assert row['plan'] == expected and row['plan']['observationId'] == result['observationId']
        assert row['feasibility'] == 'unknown_until_public_execution'
        assert not {'safe', 'success', 'selected', 'score', 'rank'}.intersection(row)
    assert result['primitiveCount'] == sum(len(row['plan']['actions']) for row in result['options'])
    assert len(json.dumps(result, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()) <= 256 * 1024


def test_ids_bind_full_requests_order_task_and_explicit_limits_independently_of_observation():
    obs = observation(); requests = controls(); initial = build(obs, requests)
    for changed in (requests[::-1], [{**requests[0], 'duration': .2}, *requests[1:]],
                    [{**requests[0], 'id': 'different-id'}, *requests[1:]]):
        assert build(obs, changed)['requestSetId'] != initial['requestSetId']
    assert build(obs, requests, task=SkillTask('access', 'preinjured_rescue'))['requestSetId'] != initial['requestSetId']
    assert build(obs, requests, max_action_seconds=1.)['requestSetId'] != initial['requestSetId']
    obs['local'][0, 7, 0, 0, 0] = 1
    changed = build(obs, requests)
    assert changed['observationId'] != initial['observationId']
    assert changed['requestSetId'] == initial['requestSetId']


def test_results_and_observation_snapshot_are_detached_without_mutating_callers():
    obs = observation(); before = {key: array_digest(value) for key, value in obs.items()}
    requests = controls() + [{'id': 'route', 'skill': 'approach', 'pose': pose(), 'parameters': {'forceN': 1.}}]
    original = copy.deepcopy(requests); result = build(obs, requests)
    result['options'][-1]['request']['pose']['position'][0] = 100
    result['options'][-1]['plan']['actions'][-1]['pose']['position'][1] = 200
    assert requests == original
    assert {key: array_digest(value) for key, value in obs.items()} == before
    obs['slots'][0, 0] = 100
    assert result['options'][-1]['plan']['actions'][-1]['pose']['position'][0] == 0
    assert math.copysign(1., build(requests=original)['options'][-1]['request']['pose']['position'][0]) == -1.


def test_stop_keeps_observed_left_right_control_order_and_empty_stop_is_not_success():
    obs = observation(); obs['slots'][:, 8:11] = 1; obs['slots'][:, 11] = 2
    requests = [controls()[-1], *controls()[:-1]]
    result = build(obs, requests); stop = result['options'][0]
    assert [(a['slot'], a['operation']) for a in stop['plan']['actions']] == [
        (hand, operation) for hand in ('left', 'right') for operation in ('deactivate', 'release', 'withdraw')]
    assert stop['status'] == 'nominal_proposal' and len(stop['plan']['actions']) == 6
    assert 'success' not in result and 'success' not in stop
    empty = build()['options'][-1]
    assert empty['plan']['actions'] == [] and empty['status'] == 'nominal_proposal'
    assert empty['feasibility'] == 'unknown_until_public_execution' and 'success' not in empty['plan']


def test_held_exchange_radius_and_withdraw_refusals_are_preserved_as_nominal_rows():
    obs = observation(); obs['tool_parameters'][0, 0] = 1.5
    obs['slots'][0, 9:11] = 1; obs['slots'][0, 11] = 1
    requests = controls() + [
        {'id': 'exchange', 'skill': 'approach', 'instrument': 'cusa', 'pose': pose()},
        {'id': 'resize', 'skill': 'inspect', 'parameters': {'radiusMm': 1.}}]
    result = build(obs, requests); rows = {row['id']: row for row in result['options']}
    for identity in ('exchange', 'resize', 'withdraw:left'):
        assert rows[identity]['status'] == 'rejected_nominal_proposal'
        assert rows[identity]['plan'] is None and rows[identity]['reason']
        assert rows[identity]['feasibility'] == 'unknown_until_public_execution'
    assert 'instrument' in rows['exchange']['reason'] and 'radius' in rows['resize']['reason']
    assert rows['release:left']['status'] == 'nominal_proposal'
    assert [row['id'] for row in result['options']] == [row['id'] for row in requests]


def test_normalized_budgets_do_not_infer_action_seconds_or_filter_declared_options():
    obs = observation(); obs['budgets'][:] = 0; obs['metrics'][OBS_METRICS.index('attempts')] = 100000
    requests = controls() + [{'id': 'declared-longer', 'skill': 'inspect', 'duration': 2.}]
    result = build(obs, requests, max_action_seconds=3.)
    assert result['options'][-1]['plan']['actions'][0]['duration'] == 2.
    assert result['options'][-1]['status'] == 'nominal_proposal'


def test_decorated_controls_cannot_replace_plain_coverage_but_remain_as_extra_rows(monkeypatch):
    requests = [{**row, 'pose': pose()} for row in controls()[:-1]] + [controls()[-1]]
    original = options.propose_skill
    def forbidden(*args, **kwargs):raise AssertionError('Decorated controls reached proposals without plain coverage')
    monkeypatch.setattr(options, 'propose_skill', forbidden)
    with pytest.raises(ValueError, match='route/parameter/instrument-free release'):
        build(requests=requests)
    monkeypatch.setattr(options, 'propose_skill', original)
    extra = {'id': 'decorated-release', 'skill': 'release', 'slot': 'left', 'pose': pose()}
    result = build(requests=controls() + [extra])
    assert [row['id'] for row in result['options']] == [row['id'] for row in controls()] + [extra['id']]
    assert result['options'][0]['status'] == 'nominal_proposal'
    assert result['options'][-1]['request'] == extra
    assert result['options'][-1]['status'] == 'rejected_nominal_proposal'
    assert result['options'][-1]['plan'] is None and 'route' in result['options'][-1]['reason']


@pytest.mark.parametrize('change', [
    lambda rows: rows + [{'id': 'unknown', 'skill': 'inspect', 'hidden_info': {}}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'not-a-skill'}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'inspect', 'parameters': {'hidden': 1}}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'inspect', 'parameters': {'power': np.float64(1)}}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'inspect', 'parameters': {'power': True}}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'approach', 'pose': {'position': (0, 0, 0), 'quaternion': [0, 0, 0, 1]}}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'inspect', 'duration': float('nan')}],
    lambda rows: rows + [{'id': 'unknown', 'skill': 'inspect', 'duration': 6.}],
    lambda rows: rows + [copy.deepcopy(rows[0])],
    lambda rows: [{**row, 'id': ('x' * 14000) + str(i)} for i, row in enumerate(rows)],
    lambda rows: rows + [{'id': str(i), 'skill': 'inspect'} for i in range(28)],
    lambda rows: rows[:-1],
    lambda rows: [{key: value for key, value in row.items() if key != 'slot'} for row in rows],
    lambda rows: [*rows[:-1], {**rows[-1], 'parameters': {}}],
])
def test_malformed_unknown_caps_and_missing_controls_fail_globally_before_any_proposal(monkeypatch, change):
    def forbidden(*args, **kwargs):raise AssertionError('proposal called before declaration validation')
    monkeypatch.setattr(options, 'propose_skill', forbidden)
    with pytest.raises((TypeError, ValueError)):
        build(requests=change(controls()))


@pytest.mark.parametrize('change', [
    lambda obs: obs.update(hidden_world={}),
    lambda obs: obs.update(local=obs['local'].astype(np.float64)),
    lambda obs: obs.update(local=obs['local'][:, :, :8]),
    lambda obs: obs['budgets'].__setitem__(0, 1.1),
    lambda obs: obs['coverage'].__setitem__(0, 2),
    lambda obs: obs['slots'].__setitem__((0, 7), .5),
    lambda obs: obs['metrics'].__setitem__(0, float('inf')),
])
def test_only_exact_finite_published_observation_contract_is_accepted(monkeypatch, change):
    obs = observation(); change(obs)
    monkeypatch.setattr(options, 'propose_skill', lambda *_a, **_k: pytest.fail('invalid observation reached proposal'))
    with pytest.raises(ValueError):build(obs)


def test_output_primitive_and_response_caps_remain_fail_closed(monkeypatch):
    original = options.propose_skill
    def too_many(*args, **kwargs):
        plan = original(*args, **kwargs); plan['actions'] = [{'slot': 'left', 'operation': 'inspect', 'duration': .1}] * 7
        return plan
    monkeypatch.setattr(options, 'propose_skill', too_many)
    with pytest.raises(ValueError, match='primitives'):build()
    monkeypatch.setattr(options, 'propose_skill', original)
    monkeypatch.setattr(options, 'MAX_TOTAL_PRIMITIVES', 4)
    with pytest.raises(ValueError, match='primitives'):build(requests=controls() + [{'id': 'extra', 'skill': 'inspect'}])
    monkeypatch.setattr(options, 'MAX_TOTAL_PRIMITIVES', 192)
    def oversized(*args, **kwargs):
        plan = original(*args, **kwargs); plan['extra'] = 'x' * (256 * 1024)
        return plan
    monkeypatch.setattr(options, 'propose_skill', oversized)
    with pytest.raises(ValueError, match='response.*byte limit'):build()


@pytest.mark.parametrize('seconds', [True, None, '5', 0, .001, 31, float('inf')])
def test_action_duration_configuration_is_explicit_and_bounded(seconds):
    with pytest.raises(ValueError):build(max_action_seconds=seconds)
