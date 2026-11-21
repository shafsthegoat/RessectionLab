"""Manual child identity, checkpoint admission, partial contact and durable replay."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_environment import SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_practice import THERMAL, verify_episode

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def request(session, operation, **args):
    return session.execute(operation, args, _Request('practice-test'), lambda *unused: None)


@pytest.fixture
def practice_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from qualify_bipolar_thermal import make_case
    case, scene = make_case()
    assert case.semantic_hash == THERMAL
    engine = SurgicalEngine(scene)
    act = {'slot': 'left', 'operation': 'configure', 'duration': .1, 'mechanicsRevision': 2}
    assert engine.step(act)['status'] == 'accepted'
    record = {'environmentSchema': SCHEMA, 'mapping': scene.regions,
              'branchId': 'configured-start', 'position': 1, 'parents': {'configured-start': None},
              'branches': {'configured-start': engine.snapshot()}}
    session = BridgeSession(tmp_path / 'cache')
    try:
        from resectionlab.imaging import save_case
        parent = tmp_path / 'parent.ressectionlab'
        save_case(case, parent, artifacts={'surgical_toolbox': record})
        request(session, 'loadCase', path=str(parent))
        yield session, scene, record
    finally:
        session.close()


def surgical(session, command, **args):
    return request(session, 'surgicalRehearsal', caseHash=THERMAL, command=command, **args)


def selection(view):
    return {'expectedStateId': view['view']['stateId'], 'position': view['position'], 'branchId': view['branchId'],
            'checkpointSeal': view['practiceAdmission']['checkpointSeal']}


def test_fork_partial_rejection_save_and_fresh_process_reopen(practice_source, tmp_path):
    session, scene, record = practice_source
    source = surgical(session, 'inspect')
    args = selection(source)
    before = semantic_digest(thaw_json(session._get_case(THERMAL).artifacts))
    for changed in ({'branchId': 'wrong'}, {'position': True}, {'position': 0}, {'checkpointSeal': 'other'},
                    {'expectedStateId': 'stale'}, {'frameIndex': 1}):
        with pytest.raises((ValueError, RuntimeError)):
            surgical(session, 'practice', **{**args, **changed})
        assert semantic_digest(thaw_json(session._get_case(THERMAL).artifacts)) == before
    child = surgical(session, 'practice', **args)
    episode = child['practiceEpisode']
    assert episode['parent']['stateId'] == source['view']['stateId'] == child['view']['stateId']
    assert episode['parent']['position'] == 1 and episode['parent']['branchId'] == source['branchId']
    assert child['branchId'] == episode['episodeId'] != source['branchId']
    assert child['practiceMetrics']['originalTargetVolumeMm3'] > 0
    assert child['practiceMetrics']['nonTargetBrainRemovedMm3'] == 0
    assert child['view']['metrics'] == source['view']['metrics']
    assert len(child['branches']) == 1
    state = child['view']['stateId']
    def step(state_id, operation, **extra):
        action = {'slot': 'right', 'operation': operation, 'duration': .1, **extra}
        return surgical(session, 'step', expectedStateId=state_id, expectedEpisodeId=episode['episodeId'], action=action)
    with pytest.raises(ValueError, match='displayed practice episode'):
        surgical(session, 'step', expectedStateId=state, expectedEpisodeId='other', action={})
    placed = step(state, 'place', pose={'position': [11., 10., -5.], 'quaternion': [0., 0., 0., 1.]},
                  parameters={'radiusMm': 1.5, 'power': 1., 'duty': 1.})
    requested = [11., 10., 21.2]
    partial = step(placed['view']['stateId'], 'apply', pose={'position': requested, 'quaternion': [0., 0., 0., 1.]},
                   duration=3.)
    assert partial['outcome']['status'] == 'partial' and 0 < partial['outcome']['completedFraction'] < 1
    assert partial['position'] == placed['position'] + 1 and partial['view']['slots']['right']['position'] != requested
    before = semantic_digest(thaw_json(session._get_case(THERMAL).artifacts))
    rejected = step(partial['view']['stateId'], 'exchange', instrument='bipolar_forceps')
    assert rejected['outcome']['status'] == 'rejected' and rejected['position'] == partial['position']
    assert rejected['view'] == partial['view'] and rejected['practiceEpisode'] == episode
    assert semantic_digest(thaw_json(session._get_case(THERMAL).artifacts)) == before
    frame = surgical(session, 'seekFrame', expectedStateId=partial['view']['stateId'], position=2, frameIndex=0)
    assert frame['readOnly'] and frame['practiceEpisode'] == episode
    assert frame['practiceMetrics'] == placed['practiceMetrics'] and frame['view']['stateId'] == placed['view']['stateId']
    earlier = surgical(session, 'seek', expectedStateId=partial['view']['stateId'], position=1)
    assert earlier['readOnly'] and earlier['view']['stateId'] == source['view']['stateId']
    with pytest.raises(ValueError, match='endpoint'):
        step(earlier['view']['stateId'], 'withdraw')
    tip = surgical(session, 'seek', expectedStateId=earlier['view']['stateId'], position=partial['position'])
    released = step(tip['view']['stateId'], 'deactivate')
    final = step(released['view']['stateId'], 'withdraw',
                 pose={'position': [11., 10., -5.], 'quaternion': [0., 0., 0., 1.]}, duration=1.)
    assert final['outcome']['status'] == 'accepted' and not final['view']['slots']['right']['inserted']
    parent_file = tmp_path / 'parent.ressectionlab'
    original_sha = hashlib.sha256(parent_file.read_bytes()).hexdigest()
    from resectionlab.desktop_bridge import BridgeError
    with pytest.raises(BridgeError) as blocked:
        request(session, 'saveCase', caseHash=THERMAL, path=str(parent_file), overwrite=True)
    assert blocked.value.code == 'PRACTICE_SAVE_AS' and hashlib.sha256(parent_file.read_bytes()).hexdigest() == original_sha
    archive = tmp_path / 'child.ressectionlab'
    request(session, 'saveCase', caseHash=THERMAL, path=str(archive))
    request(session, 'saveCase', caseHash=THERMAL, path=str(archive), overwrite=True)
    from resectionlab.surgical_rl_env import SurgicalRLEnv
    with pytest.raises(ValueError, match='STUDY_REPLAY_ONLY'):
        SurgicalRLEnv.from_patient(archive)
    for op in ('generateRoutes', 'trainPatient', 'createRehearsal'):
        with pytest.raises(BridgeError) as restricted:
            request(session, op, caseHash=THERMAL)
        assert restricted.value.code == 'STUDY_REPLAY_ONLY'
    cold = subprocess.check_output([sys.executable, '-c', """
import json,sys
from pathlib import Path
from resectionlab.desktop_bridge import BridgeSession,_Request
s=BridgeSession(Path(sys.argv[1]))
try:
    def call(op,**args):return s.execute(op,args,_Request('cold'),lambda *unused:None)
    c=call('loadCase',path=sys.argv[2])
    nav=c['artifacts']['surgical_toolbox']
    assert nav['requiresReplayVerification'] and 'archivePreviewAvailable' not in nav and 'branches' not in nav
    result=call('surgicalRehearsal',caseHash=c['caseHash'],command='inspect')
    assert nav['practiceEpisodeId']==result['branchId']
    print(json.dumps({k:result[k] for k in ('practiceEpisode','practiceMetrics','view','position','branchId','timeline')}))
finally:s.close()
""", str(tmp_path / 'cold'), str(archive)])
    assert json.loads(cold) == {k: final[k] for k in json.loads(cold)}
    saved = thaw_json(session._get_case(THERMAL).artifacts)['surgical_toolbox']
    engine = SurgicalEngine.restore(scene, saved['branches'][episode['episodeId']])
    for field in ('stateId', 'targetBinding', 'actionsHash', 'modelHash', 'checkpointSeal'):
        bad = copy.deepcopy(saved)
        bad['practiceEpisodes'][episode['episodeId']]['parent'][field] = 'wrong'
        with pytest.raises(ValueError, match='parent checkpoint'):
            verify_episode(engine, bad, THERMAL)


def test_read_only_recording_and_unqualified_case_stay_closed(practice_source):
    session, _, record = practice_source
    entry = session._get_case(THERMAL)
    from resectionlab.core import freeze_json
    receipt = {'branchId': record['branchId']}
    entry.artifacts = freeze_json({'surgical_toolbox': record, 'nativeStudyReplay': receipt})
    observed = surgical(session, 'inspect')
    assert observed['readOnly'] and not observed['practiceAdmission']['available']
    with pytest.raises(ValueError, match='STUDY_REPLAY_ONLY'):
        surgical(session, 'practice', **selection(observed))
    generated = request(session, 'createSurgicalFixture')['caseHash']
    other = request(session, 'surgicalRehearsal', command='create', caseHash=generated)
    assert not other['practiceAdmission']['available']
    with pytest.raises(ValueError, match='qualified manual practice domain'):
        request(session, 'surgicalRehearsal', caseHash=generated, command='practice', **selection(other))


def test_explicit_generated_procedure_policy_admits_existing_guarded_case(tmp_path):
    from test_structure_aware_cranial_planning import procedure_case
    case, base, record = procedure_case(cleanup=False)
    session = BridgeSession(tmp_path / 'procedure')
    try:
        session._install_case(case, {'surgical_toolbox': record}, _Request('install'))
        def run(command, **kwargs):
            return request(session, 'surgicalRehearsal', caseHash=case.semantic_hash, command=command, **kwargs)
        shown = run('inspect')
        assert shown['practiceAdmission']['available']
        child = run('practice', **selection(shown))
        assert child['view']['stateId'] == shown['view']['stateId'] and child['practiceMetrics']['nonTargetBrainRemovedMm3'] is None
        saved = thaw_json(session._get_case(case.semantic_hash).artifacts)['surgical_toolbox']
        restored = SurgicalEngine.restore(base, saved['branches'][child['branchId']])
        verify_episode(restored, saved, case.semantic_hash)
        assert restored.protection is not None and restored._procedure_policy['policyHash'] == shown['planningAdmission']['policyHash']
    finally:
        session.close()


def test_frozen_cranial_opening_keeps_stage_and_child_identity_at_boundary_zero(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from demonstrate_generated_kerf_cleanup import square_source
    from resectionlab.surgical_cranial_planning_roots import bind_branch_root
    from resectionlab.surgical_environment import resolve_branch_scene
    from resectionlab.surgical_scene import from_case

    case, record = square_source()
    base = from_case(case, record['mapping'])
    root = record['cranialPlanningRoots']['roots'][0]
    parent_id = 'declared-opening'
    record['branches'][parent_id] = root['sourceSnapshot']
    record['parents'][parent_id] = {'branchId': 'main', 'position': 0}
    record['timeline'][parent_id] = []
    bind_branch_root(base, record, parent_id, root['id'])
    record['branchId'] = parent_id
    session = BridgeSession(tmp_path / 'cranial')
    try:
        session._install_case(case, {'surgical_toolbox': record}, _Request('shell'))
        def run(command, **kwargs):
            return request(session, 'surgicalRehearsal', caseHash=case.semantic_hash, command=command, **kwargs)
        shown = run('inspect')
        child = run('practice', **selection(shown))
        assert child['practiceMetrics'] == {'originalTargetVolumeMm3': 27., 'nonTargetBrainRemovedMm3': None}
        response = run('step', expectedEpisodeId=child['branchId'], expectedStateId=child['view']['stateId'],
                       action=root['planning']['openingSkill']['actions'][0])
        assert response['outcome']['status'] == 'accepted' and response['branchId'] == child['branchId']
        saved = thaw_json(session._get_case(case.semantic_hash).artifacts)['surgical_toolbox']
        snapshot = saved['branches'][response['branchId']]
        native = SurgicalEngine.restore(resolve_branch_scene(base, saved), snapshot)
        verify_episode(native, saved, case.semantic_hash)
        assert snapshot['cranialStagesContract'] == root['sourceSnapshot']['cranialStagesContract']
    finally:
        session.close()


def test_unsaved_parent_siblings_survive_starting_practice(practice_source):
    session, _, _ = practice_source
    source = surgical(session, 'inspect')
    sibling = surgical(session, 'branch', expectedStateId=source['view']['stateId'])
    entry = session._get_case(THERMAL)
    original = thaw_json(entry.artifacts)['surgical_toolbox']
    assert len(original['branches']) == 2
    child = surgical(session, 'practice', **selection(sibling))
    assert len(child['branches']) == 1 and thaw_json(entry.artifacts['surgical_prior_scenes'])[-1] == original


def test_exact_enclosed_outlet_control_preserves_native_blockage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from prepare_blocked_outlet_practice import control
    from resectionlab.imaging import save_case
    from resectionlab.surgical_practice import BLOCKED_OUTLET, CONTROLS, model_hash
    case, record = control()
    assert case.semantic_hash == BLOCKED_OUTLET
    native = record['branches']['main']
    assert (native['sceneId'], model_hash(native)) == CONTROLS[BLOCKED_OUTLET]
    session = BridgeSession(tmp_path / 'cache')
    parent = tmp_path / 'enclosed.ressectionlab'
    save_case(case, parent, artifacts={'surgical_toolbox': record})
    try:
        request(session, 'loadCase', path=str(parent))
        def run(command, **args):
            return request(session, 'surgicalRehearsal', caseHash=BLOCKED_OUTLET, command=command, **args)
        source = run('inspect')
        assert source['practiceAdmission']['available']
        child = run('practice', **selection(source))
        episode = child['practiceEpisode']['episodeId']
        def step(result, operation, duration=.1, **args):
            return run('step', expectedStateId=result['view']['stateId'], expectedEpisodeId=episode,
                       action={'slot': 'left', 'operation': operation, 'duration': duration, **args})
        waiting = step(child, 'inspect', 15.)
        assert waiting['view']['fluid'] == {}
        assert waiting['view']['sources'][0]['reservoirMm3'] == pytest.approx(1.5)
        assert waiting['view']['metrics']['fluidPresentMm3'] == pytest.approx(1.5)
        exchanged = step(waiting, 'exchange', instrument='cusa')
        def pose(z):
            return {'position': [9., 9., z], 'quaternion': [0., 0., 0., 1.]}
        placed = step(exchanged, 'place', pose=pose(-5.), parameters={'radiusMm': 1.5, 'power': 1., 'duty': 1.})
        opened = step(placed, 'apply', 3., pose=pose(8.8))
        assert opened['outcome']['status'] == 'partial'
        view = opened['view']
        assert view['sources'][0]['reservoirMm3'] == 0
        assert sum(view['fluid'].values()) == pytest.approx(view['metrics']['fluidPresentMm3'])
        assert view['metrics']['fluidPresentMm3'] > 1.5
    finally:
        session.close()
