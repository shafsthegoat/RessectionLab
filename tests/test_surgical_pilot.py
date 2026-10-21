"""Tiny generated contract fixtures; never fit or open a patient in tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('gymnasium')
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import train_surgical_pilot
    return train_surgical_pilot


@pytest.fixture
def exposed():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:17, 8:17, 8:17] = 1
    scene = SurgicalScene(labels, np.eye(4), [dict(id=1, name='Generated test target',
        material='tumor', target=True)], {'provenance': 'generated software test geometry'}, {})
    env = SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False, local_spacing_mm=1.))
    env.reset(seed=17)
    for action in [dict(operation='exchange', instrument='cusa'),
                   dict(operation='place', pose=dict(position=[12., 12., -5.], quaternion=[0., 0., 0., 1.]),
                        parameters={'radiusMm': 1.}),
                   dict(operation='apply', pose=dict(position=[12., 12., 10.], quaternion=[0., 0., 0., 1.]))]:
        obs, _, _, _, info = env.step_action(dict(slot='left', duration=.1, **action))
        assert info['outcome']['status'] == 'accepted'
    return env, obs


def test_candidates_are_observation_only_and_do_not_mutate_inputs(runner, exposed):
    _, obs = exposed
    before = runner.observation_hash(obs)
    features, actions = runner.candidates(obs)
    assert features.shape == (9, 20)
    assert len(actions) == len(runner.ACTION_IDS) == 9
    assert np.isfinite(features).all()
    assert runner.observation_hash(obs) == before
    for action in actions[:7]:
        assert np.linalg.norm(np.asarray(action['pose']['position']) - obs['slots'][0, :3]) <= 2.
    # Global goal, aggregate reward metrics and privileged-looking extra fields
    # cannot influence the candidate provider or its nominal greedy baseline.
    changed = copy.deepcopy(obs)
    changed['goal'] *= 100
    changed['metrics'] += 987
    changed['hidden_truth'] = object()
    second, same_actions = runner.candidates(changed)
    np.testing.assert_array_equal(features, second)
    assert actions == same_actions


def test_cached_prefix_fork_matches_public_replay_and_is_independent(runner, exposed):
    env, obs = exposed
    before = env.engine.state_id
    clone = runner.fork_prepared(env)
    assert clone.scene is env.scene and clone.engine is not env.engine
    action = runner.candidates(obs)[1][5]
    result = clone.step_action(action)
    assert env.engine.state_id == before and env.attempts != clone.attempts
    fresh = SurgicalRLEnv(env.scene, protection=env.protection, config=env.config)
    verified, replay_obs = runner.replay(fresh, clone.snapshot(), runner.Budget())
    assert verified['reward_ledger_equal']
    for key in replay_obs: np.testing.assert_array_equal(replay_obs[key], result[0][key])


def test_actual_transition_batch_changes_torch_parameters(runner, exposed):
    env, obs = exposed
    torch.manual_seed(17); model = runner.Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=runner.SETTINGS['learning_rate'])
    budget = runner.Budget(); generator = torch.Generator().manual_seed(17)
    samples, report, snapshot = runner.rollout(env, obs, 'training', 17, budget, model, generator)
    assert budget.calls == len(samples) == len(report['decisions'])
    assert all(sample[2] == row['reward'] for sample, row in zip(samples, report['decisions']))
    stats = runner.optimizer_update(model, optimizer, [samples])
    assert stats['parameters_changed'] and stats['before_sha256'] != stats['after_sha256']
    assert np.isfinite(stats['loss']) and np.isfinite(stats['gradient_norm'])
    assert optimizer.state_dict()['state']
    assert snapshot['attempts'][:len(env.attempts)] == env.attempts


def test_baselines_share_options_and_prefix_with_zero_fitting(runner, exposed):
    env, obs = exposed
    model = runner.Policy(); before = runner.parameter_hash(model)
    for method in ('random', 'scripted', 'observed_greedy_local', 'untrained'):
        _, report, snapshot = runner.rollout(env, obs, method, 17, runner.Budget(), model)
        assert report['prefix_metrics'] == env.measurements
        assert snapshot['attempts'][:len(env.attempts)] == env.attempts
        assert set(row['action_id'] for row in report['decisions']) <= set(runner.ACTION_IDS)
    assert runner.parameter_hash(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_stop_is_a_real_deactivation_and_distinct_from_horizon_truncation(runner, exposed):
    env, obs = exposed
    class StopPolicy(torch.nn.Module):
        def forward(self, features):
            scores = torch.zeros(len(features)); scores[-1] = 10.
            return scores
    budget = runner.Budget()
    samples, report, snapshot = runner.rollout(env, obs, 'trained', 17, budget, StopPolicy())
    assert len(samples) == budget.calls == 1
    assert report['stop_reason'] == 'policy_stop'
    assert report['local_task_terminated'] and not report['local_task_truncated']
    assert report['decisions'][0]['action']['operation'] == 'deactivate'
    assert snapshot['attempts'][-1]['action']['operation'] == 'deactivate'
    assert report['local_return'] < 0
    empty = np.zeros((9, 20), np.float32)
    assert runner.observed_greedy(empty)[0] == 8


def test_preparation_alone_never_authorizes_launch(runner, monkeypatch):
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'fixture': 'hash'})
    preparation = dict(status='prepared_awaiting_desktop_and_guard_gate',
                       settings=runner.SETTINGS, source_hashes=runner.source_hashes())
    preparation['preparation_sha256'] = runner.digest(preparation)
    with pytest.raises(ValueError, match='launch receipt'):
        runner.validate_launch(preparation, {})
    gate = dict(preparation_sha256=preparation['preparation_sha256'], authorized=True,
                desktop_sub06_demonstrated=True, desktop_ucsf0004_demonstrated=True,
                shared_guards_verified=True, performance_and_telemetry_verified=True)
    runner.validate_launch(preparation, gate)
    for key in list(gate)[2:]:
        bad = {**gate, key: False}
        with pytest.raises(ValueError, match='prerequisite'):
            runner.validate_launch(preparation, bad)


def test_bad_replay_and_changed_declarations_fail_closed(runner, exposed, monkeypatch):
    env, _ = exposed
    snapshot = json.loads(json.dumps(env.snapshot()))
    snapshot['attempts'][-1]['reward'] += 1
    with pytest.raises(RuntimeError, match='REPLAY_ATTEMPT'):
        runner.replay(SurgicalRLEnv(env.scene, protection=env.protection, config=env.config), snapshot, runner.Budget())
    monkeypatch.setattr(runner, 'source_hashes', lambda: {})
    preparation = dict(status='prepared_awaiting_desktop_and_guard_gate', settings=runner.SETTINGS, source_hashes={})
    preparation['preparation_sha256'] = runner.digest(preparation)
    preparation['settings'] = {**runner.SETTINGS, 'updates': 999}
    with pytest.raises(ValueError, match='integrity'):
        runner.validate_launch(preparation, {})
