"""Metadata-only access deferral over actual small public engine transitions."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import full_case_rehearsal as director
from resectionlab import full_case_debulking
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def commit(engine, action):
    preview = engine.preview(action, include_view=False)
    result = engine.step(action)
    assert result['status'] == 'accepted', result
    assert result['stateId'] == preview['stateId']
    return result


def fixture(monkeypatch, *, legacy=False):
    labels = np.zeros((48, 48, 48), np.uint16)
    for x in (10, 24, 38): labels[x - 1:x + 2, 23:26, 16:19] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'explicit-access-deferral-engineering-fixture'}, {})
    engine = SurgicalEngine(scene)
    q = [1., 0., 0., 0.]
    accesses = [{'id': 'fixture-' + str(x), 'outsidePose': {'position': [x, 24., 55.], 'quaternion': q[:]},
        'entryPose': {'position': [x, 24., 18.], 'quaternion': q[:]},
        'targetEntryIndex': [x, 24, 18], 'drillRadiusMm': 3.} for x in (10, 38, 24)]
    monkeypatch.setattr(director, 'access_candidates', lambda *_args, **_kwargs: copy.deepcopy(accesses))
    commit(engine, {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2})
    commit(engine, {'slot': 'left', 'operation': 'place', 'pose': accesses[0]['outsidePose'],
        'parameters': {'radiusMm': 1.5}, 'duration': .2})
    meta = director.new_director(engine)
    meta.update(stage='core', coreAttempts=0, corePositioning=0, cycleStartTargetMm3=0.,
        cycleBoneRemovedCells=0, coreSearchExhausted=False, drillAttempts=7, passIndex=1)
    if legacy:
        meta.pop('passIndex'); meta.pop('passStartTargetMm3')
    stroke = {'slot': 'left', 'operation': 'apply', 'pose': accesses[0]['entryPose'], 'duration': 1.}
    # Only scheduling is bounded here. Geometry, target removal, previews,
    # clock, cleanup and withdrawals all use the actual ordinary engine.
    monkeypatch.setattr(full_case_debulking, 'local_debulking_candidates', lambda current, **_kwargs:
        [{'kind': 'fixture-core', 'action': copy.deepcopy(stroke)}] if not current.state['removed'] else [])
    monkeypatch.setattr(director, '_frontier_candidates', lambda *_args: [])
    return engine, meta


def defer_at_natural_finish(monkeypatch, *, legacy=False):
    engine, meta = fixture(monkeypatch, legacy=legacy)
    before, before_meta = engine.snapshot(), copy.deepcopy(meta)
    requested = director.request_access_pass_deferral(engine, meta)
    assert {k: v for k, v in requested.items() if k != 'accessPassDeferral'} == meta
    assert requested['accessPassDeferral']['status'] == 'requested'
    assert engine.snapshot() == before and meta == before_meta
    productive, running = director.run_slice(engine, requested, max_actions=1)
    assert productive.metrics()['targetRemovedMm3'] > 0.
    assert running['stage'] == 'core' and running['accessPassDeferral']['status'] == 'requested'
    assert 'deferredAccessPass' not in running
    stopped, report = director.run_slice(productive, running, max_actions=8)
    assert report['status'] == 'stopped' and report['stage'] == 'endpoint'
    assert 'deferred' in report['stopReason'] and 'exhausted' not in report['stopReason']
    assert all(not slot['inserted'] and not slot['active'] and not slot['grip'] for slot in stopped.state['slots'].values())
    assert not stopped.state['holds']
    pending = report['deferredAccessPass']
    assert pending['status'] == 'pending' and pending['pass']['accesses'] == meta['accesses']
    assert pending['pass']['accessIndex'] == 1
    assert pending['pass']['passIndex'] == (0 if legacy else 1)
    assert pending['targetAtDeferMm3'] == stopped.metrics()['targetRemovedMm3']
    assert pending['deferBoundary']['position'] < pending['stoppedBoundary']['position'] == len(stopped.actions)
    tail = stopped.actions[len(productive.actions):]
    assert any(action['operation'] == 'withdraw' for action in tail)
    assert not any(action['operation'] == 'place' for action in tail)
    assert engine.snapshot() == before and meta == before_meta
    assert array_digest(stopped.scene.labels) == array_digest(engine.scene.labels)
    return stopped, report


@pytest.mark.parametrize('legacy', [False, True])
def test_request_preserves_metadata_until_natural_finish_and_public_cleanup(monkeypatch, legacy):
    defer_at_natural_finish(monkeypatch, legacy=legacy)


def real_bounded_trial(monkeypatch):
    engine, meta = defer_at_natural_finish(monkeypatch)
    before, before_meta = engine.snapshot(), copy.deepcopy(meta)
    prepared = director.configure_directional_fallback(engine, meta, axis=0, limit=1)
    assert engine.snapshot() == before and meta == before_meta
    assert prepared.get('completedAccessPasses', []) == []
    assert prepared['deferredAccessPass']['pass'] == meta['deferredAccessPass']['pass']
    assert prepared['deferredAccessPass']['trials'][0]['startBoundary']['stateId'] == engine.state_id
    trial = engine.clone()
    # Execute the actual prepared exterior path, then a deliberately bounded
    # CUSA trial on the rightmost target cluster. No source/boundary is faked.
    for pose in prepared['directionalTransition']['poses']:
        slot = trial.state['slots']['left']
        distance = np.linalg.norm(np.subtract(pose['position'], slot['position']))
        angle = (Rotation.from_quat(pose['quaternion']) * Rotation.from_quat(slot['quaternion']).inv()).magnitude()
        commit(trial, {'slot': 'left', 'operation': 'withdraw', 'pose': pose,
            'duration': max(.2, (distance + 40. * angle) / 20.)})
    q = prepared['accesses'][0]['outsidePose']['quaternion']
    commit(trial, {'slot': 'left', 'operation': 'inspect', 'duration': .2, 'parameters': {'radiusMm': 1.5}})
    prior_target = trial.metrics()['targetRemovedMm3']
    commit(trial, {'slot': 'left', 'operation': 'apply', 'duration': 2.,
        'pose': {'position': [37., 24., 17.], 'quaternion': q}})
    assert trial.metrics()['targetRemovedMm3'] > prior_target
    commit(trial, {'slot': 'left', 'operation': 'inspect', 'duration': .2, 'parameters': {'radiusMm': .5}})
    commit(trial, {'slot': 'left', 'operation': 'withdraw', 'duration': 2.,
        'pose': prepared['accesses'][0]['outsidePose']})
    # An honest stop may retain its real stage label after actual public
    # cleanup. Resume checks physical flags/history rather than inventing an
    # exhausted endpoint to satisfy a stage-name requirement.
    prepared.update(status='stopped', stage='core', lastStateId=trial.state_id,
        lastPosition=len(trial.actions), stopReason='Bounded engineering trial stopped after public withdrawal.',
        coreAttempts=31, passIndex=0)
    return engine, meta, trial, prepared


def test_resume_retains_intermediate_trial_actions_reranks_tail_and_offsets_baseline(monkeypatch):
    engine, deferred, trial, meta = real_bounded_trial(monkeypatch)
    before, before_meta = trial.snapshot(), copy.deepcopy(meta)
    resumed = director.resume_deferred_access_pass(trial, meta)
    assert trial.snapshot() == before and meta == before_meta
    assert trial.actions[:len(engine.actions)] == engine.actions
    frozen = deferred['deferredAccessPass']['pass']
    assert resumed['accessIndex'] == frozen['accessIndex'] == 1
    assert resumed['passIndex'] == frozen['passIndex'] == 1
    for key in ('drillAttempts', 'cycleStartTargetMm3', 'cycleBoneRemovedCells', 'coreAttempts', 'corePositioning', 'coreSearchExhausted'):
        assert resumed.get(key) == frozen.get(key)
    assert resumed['accesses'][0] == frozen['accesses'][0]
    assert [r['id'] for r in frozen['accesses'][1:]] == ['fixture-38', 'fixture-24']
    assert [r['id'] for r in resumed['accesses'][1:]] == ['fixture-24', 'fixture-38']
    assert all(row['capacityStateId'] == trial.state_id for row in resumed['accesses'][1:])
    target_now = trial.metrics()['targetRemovedMm3']; target_at_defer = deferred['deferredAccessPass']['targetAtDeferMm3']
    assert resumed['passStartTargetMm3'] == frozen['passStartTargetMm3'] + target_now - target_at_defer
    assert target_now - resumed['passStartTargetMm3'] == target_at_defer - frozen['passStartTargetMm3']
    assert resumed['deferredAccessPass']['pass'] == frozen and resumed['deferredAccessPass']['status'] == 'resumed'
    assert resumed['accessPassDeferral']['status'] == 'resumed'
    assert resumed['interveningAccessTrials'][-1]['stoppedStage'] == 'core'
    assert resumed['status'] == 'ready' and resumed['stage'] == 'directional_transition'
    assert resumed['directionalTransition']['poses']
    # The resumed exterior transit itself is still an ordinary public action.
    candidate, running = director.run_slice(trial, resumed, max_actions=1)
    assert candidate.actions[-1]['operation'] == 'withdraw'
    assert running['directionalTransition']['results'][0]['arrived']
    assert candidate.actions[:len(trial.actions)] == trial.actions
    with pytest.raises(ValueError): director.resume_deferred_access_pass(trial, resumed)


def test_resume_refuses_inserted_and_accepts_real_outside_blocked_boundary(monkeypatch):
    _, _, trial, meta = real_bounded_trial(monkeypatch)
    point = {'position': trial.state['slots']['left']['position'][:], 'quaternion': trial.state['slots']['left']['quaternion'][:]}
    commit(trial, {'slot': 'left', 'operation': 'move', 'duration': .2, 'pose': point})
    assert trial.state['slots']['left']['inserted']
    meta.update(stage='directional_transition', lastStateId=trial.state_id,
        directionalTransition={'status': 'blocked'}, stopReason='Recorded outside trial transition blocked.')
    with pytest.raises(ValueError, match='Withdraw'):
        director.resume_deferred_access_pass(trial, meta)
    commit(trial, {'slot': 'left', 'operation': 'withdraw', 'duration': .2, 'pose': point})
    meta['lastStateId'] = trial.state_id
    before = trial.snapshot()
    resumed = director.resume_deferred_access_pass(trial, meta)
    assert resumed['status'] == 'ready' and trial.snapshot() == before


def test_source_state_archive_and_real_history_boundaries_are_required(monkeypatch):
    _, _, trial, meta = real_bounded_trial(monkeypatch)
    baseline = trial.snapshot()
    mutations = [
        lambda m: m.update(sourceSceneId='wrong'),
        lambda m: m.update(lastStateId='wrong'),
        lambda m: m['deferredAccessPass'].update(sourceSceneId='wrong'),
        lambda m: m['deferredAccessPass']['stoppedBoundary'].update(stateId='wrong'),
        lambda m: m['deferredAccessPass']['pass'].update(accessIndex=0),
        lambda m: m['deferredAccessPass'].update(targetAtDeferMm3=-1.),
        lambda m: m['deferredAccessPass']['trials'][0]['startBoundary'].update(position=0),
        lambda m: m['deferredAccessPass'].update(status='resumed'),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(meta); mutate(invalid)
        with pytest.raises(ValueError): director.resume_deferred_access_pass(trial, invalid)
        assert trial.snapshot() == baseline
    trial.history[1]['stateBefore'] = 'broken ancestry'
    with pytest.raises(ValueError, match='matching public action history'):
        director.resume_deferred_access_pass(trial, meta)


def test_request_rejects_duplicate_stale_or_non_superior_boundaries(monkeypatch):
    engine, meta = fixture(monkeypatch)
    requested = director.request_access_pass_deferral(engine, meta)
    before = engine.snapshot()
    with pytest.raises(ValueError, match='already requested'):
        director.request_access_pass_deferral(engine, requested)
    for changes in ({'sourceSceneId': 'wrong'}, {'startStateId': 'wrong'},
                    {'accessAxis': 0}, {'stage': 'cleanup'}, {'status': 'stopped'}):
        invalid = copy.deepcopy(meta); invalid.update(changes)
        with pytest.raises(ValueError): director.request_access_pass_deferral(engine, invalid)
    assert engine.snapshot() == before


def test_second_directional_trial_validates_pending_archive_before_candidate_generation(monkeypatch):
    from resectionlab import full_case_directional
    _, _, trial, meta = real_bounded_trial(monkeypatch)
    # An actual outside inspection binds the stopped endpoint; it does not
    # label the retained superior pass or the bounded trial as exhausted.
    commit(trial, {'slot': 'right', 'operation': 'inspect', 'duration': .2})
    meta.update(stage='endpoint', lastStateId=trial.state_id, lastPosition=len(trial.actions),
                stopReason='Bounded trial ended with both tools withdrawn and an outside inspection.')
    assert meta['accessAxis'] == 0 and meta['deferredAccessPass']['status'] == 'pending'
    before, before_meta = trial.snapshot(), copy.deepcopy(meta)
    calls = 0
    original = full_case_directional.directional_access_candidates
    def candidates(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(full_case_directional, 'directional_access_candidates', candidates)
    for mutate in (
        lambda m: m['deferredAccessPass'].update(sourceSceneId='foreign'),
        lambda m: m['deferredAccessPass']['stoppedBoundary'].update(stateId='stale'),
    ):
        invalid = copy.deepcopy(meta); mutate(invalid)
        with pytest.raises(ValueError):
            director.configure_directional_fallback(trial, invalid, axis=1, limit=1)
        assert calls == 0 and trial.snapshot() == before
    valid = director.configure_directional_fallback(trial, meta, axis=1, limit=1)
    assert calls == 1 and valid['accessAxis'] == 1
    assert valid['deferredAccessPass']['status'] == 'pending'
    assert valid['deferredAccessPass']['pass'] == meta['deferredAccessPass']['pass']
    assert [row['axis'] for row in valid['deferredAccessPass']['trials']] == [0, 1]
    assert valid['completedAccessPasses'][-1]['axis'] == 0
    assert not any(row['axis'] == 2 for row in valid['completedAccessPasses'])
    assert trial.snapshot() == before and meta == before_meta
