import copy
import hashlib
import json

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest
from resectionlab.development_episode import execute_development_episode, make_development_task, replay_frames
from resectionlab.public_surface_contact import make_contact_task, OBSERVATION_VERSION
from resectionlab.surface_contact_episode import execute_surface_contact_episode, episode_envelope, SCHEMA


def test_application_export_uses_shared_case_and_exact_authoritative_replay():
    display, episode = execute_surface_contact_episode()
    task = make_contact_task()
    for action in episode['planning']['strategy']['actions']:
        task.step(action)
    assert episode['schema'] == SCHEMA and episode['taskKind'] == 'public_retained_surface_contact'
    assert episode['caseHash'] == display.semantic_hash
    assert display.metadata['is_synthetic'] is True
    assert episode['sourceBinding']['structural_intensity_hash'] == array_digest(display.mri)
    assert episode['sourceBinding']['private_reference_published'] is False
    assert semantic_digest(episode['replayFrames']) == semantic_digest(replay_frames(task, task.metrics()['history']))
    assert episode['nativeEngineFinalStateId'] == task._engine.state_hash
    assert episode['geometryAudit']['feasible'] and episode['geometryAudit']['complete_tool_checked']
    assert episode['metrics']['total_reward'] == pytest.approx(.708)
    assert episode['publicGoal']['nativeIndex'] == [6, 6, 3] == episode['publicGoal']['rasMm']
    assert episode['planning']['referenceScoringPerformed'] is False
    assert episode['planning']['sealedBeforeExecution'] is True
    assert 'sealedBeforeReferenceScoring' not in episode['planning']
    assert episode['planning']['learnedPolicyExecuted'] is False
    assert episode['planning']['strategy']['observation_contract'] == OBSERVATION_VERSION
    assert all('target_removed_mm3' not in row and 'normal_removed_mm3' not in row for row in episode['history'])
    assert episode['taskContract']['trainingAdmission'] is False
    envelope = episode_envelope(episode)
    assert 'sha256:' + hashlib.sha256(envelope['episodeCanonicalJson'].encode()).hexdigest() == episode['episodeId']
    assert json.loads(envelope['episodeCanonicalJson']) == {k:v for k,v in episode.items() if k!='episodeId'}


def test_both_public_goals_share_physical_source_and_search_selects_meaningfully():
    near_case, near = execute_surface_contact_episode(selector='SEARCH', goal_id='near')
    costly_case, costly = execute_surface_contact_episode(selector='SEARCH', goal_id='costly')
    assert near_case.semantic_hash == costly_case.semantic_hash
    assert near['sourceHash'] == costly['sourceHash']
    assert near['decisionModelHash'] != costly['decisionModelHash']
    assert [row['interaction_mode'] for row in near['history']] == ['aspirate','probe']
    assert [row['interaction_mode'] for row in costly['history']] == ['stop']
    assert near['metrics']['total_reward'] == pytest.approx(.708)
    assert costly['metrics']['total_reward'] == 0
    for episode in (near,costly):
        assert episode['planning']['completed_layers']==2
        assert episode['planning']['beam_pruned_prefixes']==0
        assert not episode['planning']['call_cap_reached'] and not episode['planning']['time_cap_reached']


def test_costly_script_is_labeled_demonstration_not_optimal_policy():
    _, episode = execute_surface_contact_episode(goal_id='costly')
    assert episode['metrics']['goal_contacted_and_retained']
    assert episode['metrics']['total_reward'] < 0
    assert episode['planning']['selector']=='scripted_public_contact_demonstration'
    assert episode['planning']['actor_forward_calls']==0


def test_export_has_no_private_target_dependency(monkeypatch):
    import resectionlab.surface_contact_episode as module
    original = module.make_contact_task
    outputs = []
    for value in (0.,1.):
        monkeypatch.setattr(module,'make_contact_task',lambda **kw: original(
            **kw,reference_target=np.full((13,13,12),value,np.float32)))
        display,episode=module.execute_surface_contact_episode()
        outputs.append((display.semantic_hash,episode))
    assert outputs[0] == outputs[1]


def test_legacy_aspiration_episode_retains_v1_contract():
    legacy=make_development_task();old_hash=legacy.decision_model_hash
    _,episode=execute_development_episode(selector='SEARCH')
    assert episode['schema']=='resectionlab.shared-native-development-episode.v1'
    assert episode['decisionModelHash']==old_hash
    assert 'publicGoal' not in episode and 'taskContract' not in episode
    assert episode['planning']['strategy']['observation_contract']=='sequential-spatial-observation-v1'
    assert all('target_removed_mm3' in row for row in episode['history'])


def test_unsupported_selection_cancel_and_changed_envelope_refused():
    with pytest.raises(ValueError):execute_surface_contact_episode(selector='RL256_ASPIRATION_TRANSFER')
    with pytest.raises(ValueError):execute_surface_contact_episode(goal_id='arbitrary')
    with pytest.raises(InterruptedError):execute_surface_contact_episode(cancelled=lambda:True)
    _,episode=execute_surface_contact_episode()
    changed=copy.deepcopy(episode);changed['publicGoal']['nativeIndex'][2]+=1
    with pytest.raises(ValueError):episode_envelope(changed)
