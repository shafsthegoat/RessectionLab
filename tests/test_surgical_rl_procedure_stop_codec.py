"""The procedure learner can express checked STOP through its fixed action space."""
import copy
import json

import pytest

pytest.importorskip('gymnasium')

from resectionlab.surgical_rl_env import (
    FIXED_SPACE_STOP_REASON, OPERATIONS, PREVIOUS_PROCEDURE_SCHEMA,
    PROCEDURE_OPERATIONS, PROCEDURE_SCHEMA, RLConfig, SurgicalRLEnv,
)
from test_surgical_rl_procedure import command, env, prepare


def encoded_stop(e):
    value = e.encode_action({'slot': 'left', 'operation': 'stop'})
    assert e.action_space.contains(value)
    assert e.decode_action(value) == {'slot': 'left', 'operation': 'stop', 'reason': FIXED_SPACE_STOP_REASON}
    return value


def test_procedure_codec_exposes_stop_without_changing_legacy_operation_ids():
    e = env(); prepare(e)
    assert e.operations == PROCEDURE_OPERATIONS == OPERATIONS + ('stop',)
    assert e.action_space['operation'].n == len(OPERATIONS) + 1
    for index, operation in enumerate(OPERATIONS):
        assert e.operations[index] == operation
    assert e.schema == PROCEDURE_SCHEMA == 'resectionlab.surgical-rl.procedure-evaluation.v4'
    assert e.action_contract['fixedSpaceCodec']['operations'] == list(e.operations)
    encoded_stop(e)


def test_encoded_stop_refuses_inserted_tool_then_preserves_residual_after_real_exit():
    e = env(); prepare(e, x=55.)
    assert e.step_action(command('apply', [55., 40., 100.], duration=1.))[4]['outcome']['status'] == 'accepted'
    before = copy.deepcopy(e.engine.snapshot()); measures = e.measurements.copy()
    result = e.step(encoded_stop(e))
    assert result[4]['outcome']['reasonCode'] == 'PROCEDURE_STOP_REQUIRES_CHECKED_EXIT'
    assert result[2:4] == (False, False)
    assert e.engine.snapshot() == before and e.measurements == measures
    assert e.step_action(command('withdraw', [55., 40., 65.], duration=1.))[4]['outcome']['status'] == 'accepted'
    before = copy.deepcopy(e.engine.snapshot()); measures = e.measurements.copy()
    result = e.step(encoded_stop(e))
    assert result[2:4] == (True, False) and result[4]['termination_reason'] == 'stopped_with_residual'
    assert not result[4]['success'] and result[4]['procedure']['outcome']['target']['residualMm3'] > 0
    assert result[4]['outcome']['elapsedSeconds'] == 0.
    assert e.engine.snapshot() == before and e.measurements == measures
    assert result[4]['reward_terms']['tumor_removal'] == 0
    restored = env(); restored.restore(json.loads(json.dumps(e.snapshot(), allow_nan=False)))
    assert restored.engine.snapshot() == e.engine.snapshot()
    assert restored.attempts == e.attempts and restored.stop_reason == FIXED_SPACE_STOP_REASON


def test_encoded_stop_cannot_use_pose_duration_or_parameters_to_supply_free_exit():
    e = env(); prepare(e, x=55.)
    assert e.step_action(command('apply', [55., 40., 100.], duration=1.))[4]['outcome']['status'] == 'accepted'
    action = encoded_stop(e)
    action['use_pose'] = 1
    action['pose'][:] = 0.  # Irrelevant quaternion is never used as a motion.
    action['parameters'][:] = e.action_space['parameters'].high
    action['parameter_mask'][:] = 1
    action['duration'][:] = e.config.max_action_seconds
    before = copy.deepcopy(e.engine.snapshot())
    assert e.step(action)[4]['outcome']['reasonCode'] == 'PROCEDURE_STOP_REQUIRES_CHECKED_EXIT'
    assert e.engine.snapshot() == before


def test_encoded_stop_requires_whole_instrument_exterior_after_shallow_withdraw():
    e = env(); prepare(e, x=55.)
    assert e.step_action(command('apply', [55., 40., 100.], duration=1.))[4]['outcome']['status'] == 'accepted'
    before = copy.deepcopy(e.engine.snapshot())
    assert e.step_action(command('withdraw', [55., 40., 99.]))[4]['outcome']['reasonCode'] == 'PROCEDURE_EXTERIOR_WITHDRAWAL_REQUIRED'
    assert e.step(encoded_stop(e))[4]['outcome']['reasonCode'] == 'PROCEDURE_STOP_REQUIRES_CHECKED_EXIT'
    assert e.engine.snapshot() == before


def test_v3_procedure_snapshot_is_refused_explicitly_and_atomically():
    e = env(); prepare(e)
    snapshot = e.snapshot(); snapshot['schema'] = PREVIOUS_PROCEDURE_SCHEMA
    fresh = env(); fresh.reset(seed=9); before = fresh.engine.snapshot()
    with pytest.raises(ValueError, match='v3 lacks the fixed-space STOP codec'):
        fresh.restore(snapshot)
    assert fresh.engine.snapshot() == before


@pytest.mark.parametrize('maximum', [.01, .1, .5])
def test_stop_encoder_respects_subsecond_action_spaces_without_advancing_time(maximum):
    e = env(max_action_seconds=maximum); e.reset(seed=2)
    before = copy.deepcopy(e.engine.snapshot())
    result = e.step(encoded_stop(e))
    assert result[2:4] == (True, False) and result[4]['outcome']['reasonCode'] == 'PROCEDURE_STOP'
    assert e.engine.snapshot() == before


def test_legacy_environment_keeps_twelve_operations_and_codec():
    from test_surgical_planning_boundary import source
    e = SurgicalRLEnv(source('direct-generated'), config=RLConfig(require_critical_coverage=False))
    e.reset(seed=4)
    assert e.operations == OPERATIONS and len(e.operations) == 12
    assert e.action_space['operation'].n == 12 and 'fixedSpaceCodec' not in e.action_contract
    with pytest.raises(ValueError):
        e.encode_action({'slot': 'left', 'operation': 'stop'})


@pytest.mark.parametrize('extra', [{'duration': .1}, {'pose': {'position': [0., 0., 0.]}}, {'parameters': {}}])
def test_stop_encoder_refuses_physical_action_fields(extra):
    e = env(); e.reset(seed=1)
    with pytest.raises(ValueError, match='PROCEDURE_STOP_CANNOT_MOVE_OR_CHANGE_TOOLS'):
        e.encode_action({'slot': 'left', 'operation': 'stop', **extra})
