"""Search dispatch controls; native full milling is verified in the skill suite."""
import copy

import pytest

from resectionlab.surgical_joint_cranial_search import (
    _proposal, checked_joint_sequence_search, freeze_joint_resection_goal)
from resectionlab.surgical_joint_cranial_skills import checked_joint_actions
from test_surgical_joint_milling_opening import fixture


def declaration(approach):
    return {'approach': approach, 'drill_seconds': 30., 'suction_seconds': 2.,
        'staging_seconds': .1, 'slot': 'left', 'max_batches': 8,
        'max_contacts': 32, 'max_examined_cells': 81,
        'max_contact_probes': 729, 'point_scheme': 'center_and_quarter_cell_corners'}


def proposal(source, approach):
    return {'id': 'adaptive-opening', 'sourceStateId': source.state_id,
        'kind': 'milling_opening', 'milling': declaration(approach)}


def configured_source():
    source, _, _, approach, meter, limits = fixture(
        point_scheme='center_and_quarter_cell_corners', park_other=False)
    goal = freeze_joint_resection_goal(source, limits=limits)
    current, receipt = checked_joint_actions(source, [
        {'slot': 'left', 'operation': 'configure', 'duration': .1,
         'mechanicsRevision': 2, 'clockRevision': 1}], limits=limits, meter=meter)
    assert receipt['status'] == 'accepted'
    return source, current, approach, meter, limits, goal


def test_structured_milling_spec_is_copied_and_policy_cannot_be_injected():
    source, _, _, approach, meter, _ = fixture(
        point_scheme='center_and_quarter_cell_corners', park_other=False)
    raw = proposal(source, approach)
    actual = _proposal(raw, source.state_id)
    assert actual == raw and actual is not raw and actual['milling'] is not raw['milling']
    for key, value in [('meter', 'replacement'), ('limits', {}), ('check', None), ('phase', 'hidden')]:
        bad = copy.deepcopy(raw)
        bad['milling'][key] = value
        with pytest.raises(ValueError, match='search-owned'):
            _proposal(bad, source.state_id)
    for key in ('actions', 'motion'):
        bad = copy.deepcopy(raw)
        bad[key] = []
        with pytest.raises(ValueError, match='explicit bounded milling spec'):
            _proposal(bad, source.state_id)
    missing = copy.deepcopy(raw)
    missing['milling'].pop('max_contact_probes')
    with pytest.raises(ValueError, match='explicit bounded milling spec'):
        _proposal(missing, source.state_id)
    assert meter.calls == 0 and not source.actions


def test_search_dispatches_one_macro_expansion_on_original_policy_and_meter(monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, source, approach, meter, limits, goal = configured_source()
    before, before_calls, calls = source.snapshot(), meter.calls, []
    check = lambda: None

    def inspect_only_dispatch(engine, **kwargs):
        # Exercise real checked native accounting at the dispatch boundary.
        # This stub does not claim to perform or complete a milling opening.
        calls.append((engine, kwargs))
        return checked_joint_actions(engine,
            [{'slot': 'left', 'operation': 'inspect', 'duration': .1}],
            limits=kwargs['limits'], meter=kwargs['meter'], check=kwargs['check'], phase=kwargs['phase'])

    monkeypatch.setattr(module, 'checked_joint_milling_opening', inspect_only_dispatch)
    result = checked_joint_sequence_search(source, propose=lambda e: [proposal(e, approach)],
        limits=limits, goal=goal, meter=meter, check=check, horizon=1)
    assert len(calls) == 1 and calls[0][0] is source
    kwargs = calls[0][1]
    assert kwargs['limits'] is limits and kwargs['meter'] is meter and kwargs['check'] is check
    assert kwargs['max_batches'] == 8 and kwargs['max_contact_probes'] == 729
    assert kwargs['phase'].endswith('.depth1.proposal1.work')
    assert result.evidence['expansions'] == 1 and meter.calls == before_calls+2
    assert result.attempts[0]['status'] == 'no_immediate_target'
    assert result.attempts[0]['work']['ledger']['targetRemovedMm3'] == 0.
    assert not result.selected and not result.alternatives and result.evidence['decision'] == 'STOP'
    assert source.snapshot() == before


def test_local_milling_controller_limit_retains_cost_and_allows_next_candidate(monkeypatch):
    import resectionlab.surgical_joint_milling_opening as module
    _, source, approach, meter, limits, goal = configured_source()
    before, before_calls = source.snapshot(), meter.calls

    def local_stop(engine, **kwargs):
        current, receipt = checked_joint_actions(engine,
            [{'slot': 'left', 'operation': 'inspect', 'duration': .1}],
            limits=kwargs['limits'], meter=kwargs['meter'], check=kwargs['check'], phase=kwargs['phase'])
        receipt.update(status='refused', reason='MILLING_BATCH_BUDGET')
        return current, receipt

    monkeypatch.setattr(module, 'checked_joint_milling_opening', local_stop)
    def inventory(engine):
        return [proposal(engine, approach), {'id': 'next-native-program', 'kind': 'configuration',
            'sourceStateId': engine.state_id,
            'actions': [{'slot': 'left', 'operation': 'inspect', 'duration': .1}]}]
    result = checked_joint_sequence_search(source, propose=inventory, limits=limits,
        goal=goal, meter=meter, horizon=1)
    assert len(result.attempts) == 2 and result.evidence['expansions'] == 2
    assert result.attempts[0]['failureCategory'] == 'CANDIDATE_CONTROLLER_LIMIT'
    assert result.attempts[0]['work']['publicCalls'] == 2
    assert result.attempts[1]['status'] == 'no_immediate_target'
    assert meter.calls == before_calls+4 and not result.evidence['budgetExhausted']
    assert result.stopped_prefixes and not result.selected and source.snapshot() == before


def test_invalid_macro_bound_is_refused_before_any_native_search_work():
    source, _, _, approach, meter, limits = fixture(
        point_scheme='center_and_quarter_cell_corners', park_other=False)
    goal = freeze_joint_resection_goal(source, limits=limits)
    bad = proposal(source, approach)
    bad['milling']['max_contacts'] = 33
    before = source.snapshot()
    with pytest.raises(ValueError, match='EXPLICIT_BOUNDED_SPEC'):
        checked_joint_sequence_search(source, propose=lambda e: [bad],
            limits=limits, goal=goal, meter=meter)
    assert meter.calls == 0 and source.snapshot() == before


def test_unsupported_initial_attitude_is_a_candidate_limit_without_suppressing_next():
    from scipy.spatial.transform import Rotation
    from resectionlab.surgical_milling_contact_proposals import bind_milling_approach
    source, _, _, _, meter, limits = fixture(
        point_scheme='center_and_quarter_cell_corners', park_other=False)
    goal = freeze_joint_resection_goal(source, limits=limits)
    current, setup = checked_joint_actions(source, [
        {'slot': 'left', 'operation': 'configure', 'duration': .1,
         'mechanicsRevision': 2, 'clockRevision': 1},
        {'slot': 'left', 'operation': 'place', 'duration': .1,
         'pose': {'position': [32., 32., 55.], 'quaternion': [0., 0., 0., 1.]}}],
        limits=limits, meter=meter)
    assert setup['status'] == 'accepted'
    approach = bind_milling_approach(current,
        quaternion=Rotation.from_euler('x', 10., degrees=True).as_quat().tolist(),
        outside_plane_point_ras_mm=[32., 32., 55.], maximum_approach_mm=10.,
        point_scheme='center_and_quarter_cell_corners')
    before, calls = current.snapshot(), meter.calls
    def inventory(engine):
        return [proposal(engine, approach), {'id': 'next-native-program', 'kind': 'configuration',
            'sourceStateId': engine.state_id,
            'actions': [{'slot': 'left', 'operation': 'inspect', 'duration': .1}]}]
    result = checked_joint_sequence_search(current, propose=inventory, limits=limits,
        goal=goal, meter=meter, horizon=1)
    first = result.attempts[0]
    assert first['failureCategory'] == 'CANDIDATE_CONTROLLER_LIMIT'
    assert first['work']['reason'] == 'JOINT_MILLING_POSED_ATTITUDE_STAGING_UNQUALIFIED'
    assert first['work']['publicCalls'] == 0 and first['work']['nativeActionsAppended'] == 0
    assert result.attempts[1]['status'] == 'no_immediate_target'
    assert meter.calls == calls+2 and not result.evidence['budgetExhausted']
    assert current.snapshot() == before and not result.selected
