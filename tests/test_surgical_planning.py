"""Generated contract tests; no patient training or checkpoint fitting."""
import copy
import numpy as np
import pytest

pytest.importorskip('gymnasium')
from resectionlab.core import semantic_digest
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionSet, ProtectionAnnotation
from resectionlab.surgical_planning import (availability, compare_continuations,
    select_observed, validate_saved_comparison, _observer)
from resectionlab.surgical_rl_rewards import RewardTracker


@pytest.fixture
def exposed(monkeypatch):
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:17, 8:17, 8:17] = 1
    scene = SurgicalScene(labels, np.eye(4), [dict(id=1, name='Generated target', material='tumor', target=True)],
                          {'provenance': 'generated contract test'}, {})
    mask = np.zeros(scene.shape, bool); mask[0,0,0] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask, provenance='generated exclusion test')])
    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig())
    from resectionlab.surgical_planning import SOURCE_SCENES, SOURCE_CONTRACTS
    monkeypatch.setitem(SOURCE_SCENES, scene.identity, 'generated test only')
    monkeypatch.setitem(SOURCE_CONTRACTS, scene.identity, engine.guard_report()['contractId'])
    for command in [dict(operation='exchange', instrument='cusa'),
                    dict(operation='place', pose=dict(position=[12., 12., -5.], quaternion=[0.,0.,0.,1.]), parameters={'radiusMm':1.}),
                    dict(operation='apply', pose=dict(position=[12.,12.,10.], quaternion=[0.,0.,0.,1.]))]:
        assert engine.step(dict(slot='left', duration=.1, **command))['status'] == 'accepted'
    return engine


def test_explicit_opt_in_and_source_and_tool_gates(exposed):
    assert availability(exposed)['available']
    assert not availability(exposed)['checkpointAvailable']
    with pytest.raises(ValueError, match='acknowledge'):
        compare_continuations(exposed)
    with pytest.raises(ValueError, match='exact development source'):
        compare_continuations(exposed, acknowledge_unknowns=True, opponent='checkpoint')
    engine = exposed.fresh()
    assert not availability(engine)['available']
    naked = SurgicalEngine(engine.scene)
    assert any('exclusion' in reason for reason in availability(naked)['reasons'])


def test_comparison_keeps_live_state_and_exact_source_and_records_actual_stop(exposed):
    before = exposed.snapshot(); labels = exposed.scene.labels.copy()
    branches, report = compare_continuations(exposed, acknowledge_unknowns=True)
    assert exposed.snapshot() == before
    np.testing.assert_array_equal(exposed.scene.labels, labels)
    assert len(branches) == 2
    for engine, plan in branches:
        assert engine.actions[:len(exposed.actions)] == exposed.actions
        assert plan['startStateId'] == exposed.state_id
        assert plan['verification']['exactStateMatch'] and plan['verification']['endpointVectorMatch']
        assert plan['cost']['previewCalls'] == plan['cost']['optimizerUpdates'] == 0
        assert plan['reachableTargetFractionRemoved'] is None
        assert plan['invalidActions'] == sum(r['status'] == 'rejected' for r in plan['decisions'])
    _, stop = branches[1]
    assert stop['earlyStop'] and stop['stopReason'] == 'policy_stop'
    assert stop['incrementalEndpoints']['tumor_removed_mm3'] == 0
    assert stop['decisions'][0]['action']['operation'] == 'deactivate'
    assert stop['localReward'] < 0
    assert stop['cumulativeEndpoints']['near_contact_exposure_mm3_seconds'] is None
    assert 'deformation_injury_mm3' in stop['unassessedEndpoints']


def test_selector_ignores_global_truth_and_receives_only_local_observation(exposed):
    obs, _ = _observer(exposed, RewardTracker(exposed, exposed._protected_ids))
    first = select_observed(obs, 'greedy')
    other = copy.deepcopy(obs); other['goal'] *= 1000; other['metrics'] += 99
    other['private_geometry'] = object()
    assert select_observed(other, 'greedy') == first
    # Only explicit supplied observations can change logits, not preview info.
    chosen, action, _ = select_observed(obs, 'checkpoint', lambda f: np.arange(len(f)))
    assert chosen == 8 and action['operation'] == 'deactivate'
    with pytest.raises(ValueError, match='logits'):
        select_observed(obs, 'checkpoint', lambda f: np.full(len(f), np.nan))


def test_cancellation_cannot_mutate_live_state(exposed):
    before = exposed.snapshot()
    def cancel(): raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        compare_continuations(exposed, acknowledge_unknowns=True, check=cancel)
    assert exposed.snapshot() == before


def test_saved_receipts_bind_source_branches_decisions_and_endpoint(exposed):
    branches, report = compare_continuations(exposed, acknowledge_unknowns=True)
    record = {'branches': {}}
    for index, (engine, plan) in enumerate(branches):
        plan['branchId'] = str(index); record['branches'][str(index)] = engine.snapshot()
    report['comparisonSeal'] = semantic_digest(report)
    validate_saved_comparison(exposed.scene, record, report)
    bad = copy.deepcopy(report); bad['alternatives'][0]['cumulativeEndpoints']['tumor_removed_mm3'] += 100
    with pytest.raises(ValueError, match='integrity'):
        validate_saved_comparison(exposed.scene, record, bad)
    bad = copy.deepcopy(record); bad['branches']['0']['stateId'] = 'changed'
    with pytest.raises(ValueError, match='branch identity'):
        validate_saved_comparison(exposed.scene, bad, report)


def test_plan_branch_edits_fork_and_saved_report_survives_native_reopen(tmp_path, monkeypatch):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.core import freeze_json, thaw_json
    from resectionlab.surgical_planning import SOURCE_SCENES, SOURCE_CONTRACTS
    session = BridgeSession(tmp_path / 'transfers')
    def call(operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)
    try:
        case = call('createSurgicalFixture', {})
        binding = {'caseHash': case['caseHash']}
        call('surgicalRehearsal', {**binding, 'command':'create'})
        entry = session._get_case(case['caseHash'])
        scene = session.surgical_sessions[case['caseHash']][1].scene
        mask = np.zeros(scene.shape, bool); mask[0,0,0] = True
        protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask, provenance='generated bridge test')])
        engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig())
        for command in [dict(operation='exchange',instrument='cusa'),
                        dict(operation='place',pose=dict(position=[43.,20.,10.],quaternion=[0.,0.,0.,1.]),parameters={'radiusMm':.6}),
                        dict(operation='apply',pose=dict(position=[43.,20.,25.],quaternion=[0.,0.,0.,1.]))]:
            assert engine.step(dict(slot='left',duration=.1,**command))['status']=='accepted'
        artifacts = thaw_json(entry.artifacts); record = artifacts['surgical_toolbox']
        record.update(branches={'main':engine.snapshot()},position=len(engine.actions),timeline={'main':engine.history})
        entry.artifacts = freeze_json(artifacts)
        monkeypatch.setitem(SOURCE_SCENES, scene.identity, 'generated bridge test only')
        monkeypatch.setitem(SOURCE_CONTRACTS, scene.identity, engine.guard_report()['contractId'])
        state = call('surgicalRehearsal',{**binding,'command':'plan','expectedStateId':engine.state_id,'acknowledgeResearchUnknowns':True})
        assert state['view']['stateId']==engine.state_id and state['branchId']=='main'
        stopped = state['planning']['alternatives'][1]
        state = call('surgicalRehearsal',{**binding,'command':'selectBranch','expectedStateId':state['view']['stateId'],'branchId':stopped['branchId']})
        assert state['view']['stateId']==stopped['finalStateId']
        state = call('surgicalRehearsal',{**binding,'command':'step','expectedStateId':state['view']['stateId'],'action':dict(slot='left',operation='inspect',duration=.1)})
        assert state['branchId']!=stopped['branchId']
        assert state['planning']['alternatives'][1]==stopped
        path = tmp_path / 'compared.ressectionlab'
        call('saveCase',{**binding,'path':str(path),'overwrite':False})
        session.close()
        session = BridgeSession(tmp_path / 'fresh')
        opened = call('loadCase',{'path':str(path)})
        restored = call('surgicalRehearsal',{'caseHash':opened['caseHash'],'command':'inspect'})
        assert restored['planning']['alternatives'][1]==stopped
        assert restored['view']['stateId']==state['view']['stateId']
    finally:
        session.close()
