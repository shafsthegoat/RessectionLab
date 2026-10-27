"""Focused actor contract checks without generated patient data in the repository."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from resectionlab.patient_sector_learning import (ACTION_IDS, OptionActor,
    OBSERVATION_SCHEMA, PolicyInput, actor_choice, frozen_actor, input_digest,
    fixed_choice, model_binding, parameter_hash, policy_input, prepare_case,
    rollout, seal_actor)
from resectionlab.surgical_engine import SurgicalEngine


def _observation():
    return {'local': np.zeros((2, 8, 9, 9, 9)),
            'slots': np.zeros((2, 12)),
            'tool_parameters': np.zeros((2, 11)),
            'metrics': np.zeros(14), 'goal': np.zeros(4),
            'coverage': np.zeros(3), 'local_spacing_mm': np.ones(1),
            'cleanup': np.zeros(3), 'budgets': np.zeros(3)}


def _intact():
    return {'boundary': 'intact_before_access',
            'observation': _observation(), 'remainingNativeActions': 48,
            'options': [
                {'id': 'corridor:left', 'hand': 'left',
                 'entryRelativeToToolMm': [2., 1., 3.],
                 'structurallyAvailable': True},
                {'id': 'corridor:right', 'hand': 'right',
                 'entryRelativeToToolMm': [-2., 1., 3.],
                 'structurallyAvailable': True},
                {'id': 'STOP', 'hand': None,
                 'entryRelativeToToolMm': None,
                 'structurallyAvailable': True}]}


def _post_corridor():
    dto = _intact()
    dto['boundary'] = 'post_corridor'
    del dto['remainingNativeActions']
    dto['ledger'] = {'additionalTargetMm3': 12.,
        'additionalNonTargetMm3': 2.,
        'additionalPerSectorRemovedMm3': {'near': 5., 'far': 0.},
        'remainingNativeActions': 29, 'activeHand': 'left',
        'thresholds': {'additionalTargetMm3': 30., 'eachSectorMm3': 10.,
                       'additionalNonTargetMm3': 20.,
                       'totalNonTargetMm3': 200.}}
    dto['options'] = [
        {'id': sector+':'+hand, 'sector': sector, 'hand': hand,
         'centerRelativeToToolMm': [x, y, 2.],
         'structurallyAvailable': hand == 'left'}
        for sector, x in (('near', 1.), ('far', 5.))
        for hand, y in (('left', -1.), ('right', 1.))]
    dto['options'].append({'id': 'STOP', 'sector': None, 'hand': None,
                           'centerRelativeToToolMm': None,
                           'structurallyAvailable': True})
    return dto


def test_exact_observed_state_contract_rejects_privileged_or_history_fields():
    dto = _intact()
    original = policy_input(dto)
    assert original.option_ids == ('corridor:left', 'corridor:right', 'STOP')
    assert original.local.shape == (16, 9, 9, 9)
    assert original.global_features.shape == (79,)
    assert original.options.shape == (3, 11)
    dto['observation']['local'][0, 0, 0, 0, 0] = 1.
    assert original.local[0, 0, 0, 0] == 0.  # No live engine array alias.
    for forbidden in ('stateId', 'history', 'sourceCellIds', 'futurePreview'):
        altered = deepcopy(_intact())
        altered[forbidden] = 'hidden'
        with pytest.raises(ValueError, match='unsupported actor fields'):
            policy_input(altered)
    altered = _intact()
    altered['observation']['history'] = np.zeros(1)
    with pytest.raises(ValueError, match='published surgical observation'):
        policy_input(altered)


def test_actor_scores_current_sensors_and_directional_option_identity():
    torch.manual_seed(17)
    actor = OptionActor().eval()
    first = policy_input(_post_corridor())
    logits = actor(first).detach().numpy()
    assert first.option_ids == ACTION_IDS[2:]
    assert first.available.tolist() == [True, False, True, False, True]
    assert logits[1] < -1.e8 and logits[3] < -1.e8
    assert logits[0] != logits[2]  # Near/far tool-frame geometry matters.
    altered = _post_corridor()
    altered['observation']['local'][0, 0, 4, 4, 4] = 100.
    altered['ledger']['additionalTargetMm3'] = 24.
    second = policy_input(altered)
    assert input_digest(first) != input_digest(second)
    assert not np.allclose(actor(first).detach().numpy()[[0, 2, 4]],
                           actor(second).detach().numpy()[[0, 2, 4]])
    selected, scores = actor_choice(actor)(second)
    assert second.available[selected] and len(scores) == 5


def test_checkpoint_binds_task_action_schema_and_actor_parameters(tmp_path):
    class Scene:
        identity = 'test-scene'
    class Prepared:
        source_sha256 = 'source-sha'
        scene = Scene()
        access_comparison_seal = 'comparison-seal'
        access = {'planSeal': 'access-seal'}
        access_method = 'superior-access'
        sector_evidence = {'near': {'sourceCellSha256': 'near-sha'},
                           'far': {'sourceCellSha256': 'far-sha'}}
    prepared = Prepared()
    actor = OptionActor().eval()
    producer = {'runnerSourceSha256': 'runner-sha',
                'skillSourceSha256': 'skill-sha'}
    actor_sha = parameter_hash(actor)
    receipt = {'sourceBinding': model_binding(prepared),
               'producerIdentity': producer,
               'bcFit': {'fittedExact': True,
                         'finalActorSha256': actor_sha}}
    checkpoint = tmp_path/'actor.pt'
    sealed = seal_actor(actor, checkpoint, stage='bc', prepared=prepared,
                        producer=producer, training_receipt=receipt)
    assert sealed['actorSha256'] == actor_sha
    assert parameter_hash(frozen_actor(checkpoint,
                        expected_producer=producer)) == actor_sha
    with pytest.raises(ValueError, match='metadata differs'):
        frozen_actor(checkpoint, expected_producer={'skillSourceSha256': 'changed'})
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    payload['actionIds'] = ('near:left', 'STOP')
    torch.save(payload, tmp_path/'wrong-actions.pt')
    with pytest.raises(ValueError, match='metadata differs'):
        frozen_actor(tmp_path/'wrong-actions.pt', expected_producer=producer)
    payload['actionIds'] = ACTION_IDS
    payload['stateDict']['actor.2.bias'] = payload['stateDict']['actor.2.bias'] + 1.
    torch.save(payload, tmp_path/'wrong-weights.pt')
    with pytest.raises(ValueError, match='parameters differ'):
        frozen_actor(tmp_path/'wrong-weights.pt', expected_producer=producer)


def test_declared_development_native_options_append_and_replay():
    source = (Path(__file__).resolve().parents[1]/'testcases/outputs/'
        'integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/'
        'sector-comparison-v3/sector-comparison.ressectionlab')
    if not source.exists():
        pytest.skip('Ignored DEVELOPMENT source archive is unavailable')
    prepared = prepare_case(source)
    result = rollout(prepared, fixed_choice(('corridor:left', 'far:left',
        'near:left', 'STOP')), method='fixed-native-regression')
    report = result['report']
    assert report['taskGateMet'] and report['exactNativeReplay']
    assert report['deltaTargetMm3'] == 84.
    assert report['deltaNonTargetMm3'] == 3.
    assert report['perSectorRemovedMm3'] == {'near': 12., 'far': 19.}
    assert report['nativeActions'] == 14
    assert [item['optionId'] for item in report['decisions']] == [
        'corridor:left', 'far:left', 'near:left', 'STOP']
    assert result['engine'].actions[:len(prepared.intact.actions)] == prepared.intact.actions
    assert result['training'][0][2] > -1.  # Access uses the 200 mm3 exposure cap.
    assert sum(item[2] for item in result['training']) > -2.  # Intact STOP return.
    assert report['terminalReward'] == 4.
    assert report['terminalAppliedDecisionIndex'] == 3
    assert [row['trainingReward'] for row in report['decisions']] == [
        item[2] for item in result['training']]
    assert report['cost']['discardedProbePreviewCalls'] > 0
    assert report['cost']['discardedProbeCommittedCalls'] > 0
    assert report['cost']['retainedBranchCommittedCalls'] == 14
    other = prepare_case(source, access_method='axis0-access')
    assert other.access_method == 'axis0-access'
    assert other.access['planSeal'] != prepared.access['planSeal']


def test_failed_escape_probe_keeps_prefix_and_counts_actual_physics(monkeypatch):
    source = (Path(__file__).resolve().parents[1]/'testcases/outputs/'
        'integrated-goal-20261026/fresh-patients/UCSF-PDGM-0005/'
        'sector-comparison-v3/sector-comparison.ressectionlab')
    if not source.exists():
        pytest.skip('Ignored DEVELOPMENT source archive is unavailable')
    prepared = prepare_case(source)
    real_preview, real_step = SurgicalEngine.preview, SurgicalEngine.step
    observed = {'previewCalls': 0, 'externalStepCalls': 0, 'insidePreview': 0}

    def counted_preview(self, *args, **kwargs):
        observed['previewCalls'] += 1
        observed['insidePreview'] += 1
        try:
            return real_preview(self, *args, **kwargs)
        finally:
            observed['insidePreview'] -= 1

    def counted_step(self, *args, **kwargs):
        if not observed['insidePreview']:
            observed['externalStepCalls'] += 1
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(SurgicalEngine, 'preview', counted_preview)
    monkeypatch.setattr(SurgicalEngine, 'step', counted_step)
    result = rollout(prepared, fixed_choice(('corridor:left', 'far:left', 'STOP')),
                     method='instrumented-refused-escape')
    report = result['report']
    decision = report['decisions'][1]
    option = decision['nativeOption']
    assert not report['taskGateMet']
    assert report['status'] == 'incomplete' and not report['stopSelected']
    assert report['cleanupComplete'] is True
    assert report['deltaTargetMm3'] == report['deltaNonTargetMm3'] == 0.
    assert report['perSectorRemovedMm3'] == {'near': 0., 'far': 0.}
    assert decision['optionId'] == 'far:left'
    assert option['status'] == 'refused'
    assert option['attempts'] == [] and option['nativeActionsAppended'] == 0
    assert option['startStateId'] == option['finalStateId']
    assert decision['stateId'] == report['decisions'][0]['stateId']
    assert report['cost']['discardedProbePreviewCalls'] > 0
    assert report['cost']['discardedProbeCommittedCalls'] > 0
    assert report['cost']['previewCalls'] == observed['previewCalls']
    assert report['cost']['committedCalls']+report['cost']['independentReplayCalls'] == \
        observed['externalStepCalls']
    assert report['forcedCleanupResourcePenalty'] >= 0.
    assert report['decisions'][-1]['boundary'] == 'forced_cleanup'
    assert abs(report['decisions'][1]['trainingReward']-(
        report['decisions'][1]['reward']+report['terminalReward']-
        report['forcedCleanupResourcePenalty'])) < 1.e-9
