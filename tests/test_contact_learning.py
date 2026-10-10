"""Generated tensor/control checks only: zero native transitions and updates.

A canonical public source is placed in a handcrafted DTO with deliberately
software-only action/context identities. This checks admission/math plumbing,
not native trajectory authenticity; the runner must separately replay collectors.
"""
from dataclasses import replace
import hashlib
import io
import json
import os
import zipfile
import numpy as np
import pytest
import torch

from resectionlab.contact_checkpoint import (MAX_BYTES, VerifiedContactCheckpoint, decode_contact_checkpoint,
    encode_contact_checkpoint, save_contact_checkpoint, load_contact_checkpoint, initial_lineage,
    final_lineage, partial_lineage)
from resectionlab.contact_costs import ContactCostMeter
from resectionlab.contact_family_episode import (ContactEvaluationFreeze, make_frozen_evaluation_task,
    freeze_final_checkpoints)
from resectionlab.contact_learning_contract import (ContactExperiment, ContactTrainBinding, ContactSample,
    ContactTransition, PROTOCOL, freeze_contact_experiment, common_initial_policies)
from resectionlab.contact_learning import (ContactLearningSession, contact_imitation_loss,
    contact_reinforce_loss, contact_gradient_step)
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.public_contact_family import FAMILY_VERSION, SOURCE_CANDIDATE_VERSION, build_family_source
from resectionlab.public_surface_contact import PublicSurfaceGoal, SurfaceContactDevelopmentContext, SurfaceContactObservation
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.spatial_observations import SpatialAction, ObservedProcedureState, build_spatial_observation
from resectionlab.spatial_policy import parameter_hash


@pytest.fixture(scope='module')
def experiment():
    return freeze_contact_experiment()


def tensor_sample(experiment, *, role='TRAIN', goal_id='surface', steps=0, action='STOP'):
    layout, _ = experiment.keys(role)[0]; row = experiment.row(layout)
    case = build_family_source(layout)
    base = build_spatial_observation(case.spatial_inputs(np.zeros(case.observed_support.shape, bool)),
        (SpatialAction('STOP'), SpatialAction('tensor-control-only', tuple(case.access.center_mm),
          tuple(case.access.center_mm + np.array([0., 0., 1.])), case.tools[0])),
        ObservedProcedureState(case.access, steps, 2))
    contact = np.zeros(base.image_channels.shape[1:], bool)
    grid = contact.copy(); point = tuple(np.subtract(row['goals'][goal_id]['native_index'], row['crop_origin_native']))
    grid[point] = True
    model_hash = semantic_digest({'unit_context_only_not_native_model': True, 'source': case.source_hash, 'goal': goal_id})
    obs = SurfaceContactObservation(SequentialSpatialObservation(base, ('stop', 'aspirate'), contact), grid,
        PublicSurfaceGoal(case.source_hash, tuple(row['goals'][goal_id]['native_index'])), tuple(row['crop_origin_native']), model_hash)
    declaration = {'version': FAMILY_VERSION, 'experiment_hash': experiment.fingerprint,
        'proposal_mode': SOURCE_CANDIDATE_VERSION, 'source_candidate_version': SOURCE_CANDIDATE_VERSION,
        'layout_id': layout, 'goal_id': goal_id, 'role': role, 'family_hash': experiment.manifest['family_hash'],
        'recipe_hash': row['recipe_hash'], 'source_hash': row['source_hash'], 'objective_hash': obs.objective_hash,
        'decision_model_hash': model_hash, 'training_admission': False, 'scope': 'generated_forward_context_only'}
    context = SurfaceContactDevelopmentContext(semantic_digest(declaration), row['source_hash'], model_hash,
        obs.objective_hash, array_digest(grid), tuple(row['crop_origin_native']), row['crop_affine_hash'])
    return obs, context, layout, goal_id


def sample(experiment, *, steps=0, action='STOP'):
    obs, context, layout, goal = tensor_sample(experiment, steps=steps)
    return ContactSample(obs, action, ContactTrainBinding(experiment, layout, goal, context))


def session(experiment, method):
    model = common_initial_policies(experiment)[method]
    return ContactLearningSession(experiment, method, model, parameter_hash(model))


def optimizer(model, *, wrong=False):
    return torch.optim.Adam(list(model.parameters())[:1] if wrong else model.parameters(),
        lr=PROTOCOL['learning_rate'], betas=tuple(PROTOCOL['betas']),
        **{key: PROTOCOL[key] for key in ('eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'fused')})


def test_canonical_family_roles_and_common_initialization(experiment):
    assert tuple(len(experiment.keys(r)) for r in ('TRAIN', 'SELECT', 'MEASUREMENT_EVAL')) == (24, 8, 16)
    models = common_initial_policies(experiment)
    assert parameter_hash(models['IL']) == parameter_hash(models['RL'])
    assert models['IL'] is not models['RL']
    altered = thaw_json(experiment.manifest); altered['source_bindings'][0]['role'] = 'invented_TRAIN'
    with pytest.raises(ValueError): ContactExperiment(altered)


@pytest.mark.parametrize('role', ['SELECT', 'MEASUREMENT_EVAL'])
def test_nontrain_binding_refused_before_loss(experiment, role):
    obs, context, layout, goal = tensor_sample(experiment, role=role)
    with pytest.raises(ValueError, match='TRAIN_ONLY'): ContactTrainBinding(experiment, layout, goal, context)


@pytest.mark.parametrize('field', ['source_hash', 'goal_grid_hash', 'objective_hash', 'crop_affine_hash', 'declaration_hash'])
def test_binding_substitution_refused(experiment, field):
    _, context, layout, goal = tensor_sample(experiment)
    with pytest.raises(ValueError): ContactTrainBinding(experiment, layout, goal,
        replace(context, **{field: 'sha256:'+'f'*64}))


def test_binding_mutation_and_cross_goal_observation_refused(experiment):
    chosen = sample(experiment)
    other, _, _, _ = tensor_sample(experiment, goal_id='deep')
    with pytest.raises(ValueError): chosen.binding.require_observation(other)
    object.__setattr__(chosen.binding, 'goal_id', 'deep')
    with pytest.raises(ValueError): chosen.validate()


def test_fixed_il_loss_has_encoder_gradients_without_update(experiment):
    current = session(experiment, 'IL'); chosen = sample(experiment, action='tensor-control-only')
    before = parameter_hash(current.policy)
    loss, receipt = contact_imitation_loss(current, [chosen]*4)
    loss.backward()
    assert receipt['loss_forward_calls'] == 4
    assert current.policy.encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in current.policy.critic.parameters())
    assert parameter_hash(current.policy) == before and current.updates == 0


def test_fixed_rl_complete_episode_loss_has_finite_gradients_without_update(experiment):
    current = session(experiment, 'RL'); chosen = sample(experiment)
    before = parameter_hash(current.policy)
    loss, stats = contact_reinforce_loss(current, [[ContactTransition(chosen, 0., True)]]*4,
                                        behavior_parameter_hash=before)
    loss.backward()
    assert stats['actor_reduction'] == 'mean_episodes_sum_discounted_score_terms'
    assert current.policy.encoder[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in current.policy.parameters() if p.grad is not None)
    assert current.updates == 0 and parameter_hash(current.policy) == before


def test_loss_refuses_incomplete_spliced_and_off_policy_episodes(experiment):
    current = session(experiment, 'RL'); chosen = sample(experiment)
    incomplete = [[ContactTransition(chosen, 0., False)]]*4
    with pytest.raises(ValueError): contact_reinforce_loss(current, incomplete, behavior_parameter_hash=parameter_hash(current.policy))
    with pytest.raises(ValueError): contact_reinforce_loss(current, [[ContactTransition(chosen, 0., True)]]*4,
        behavior_parameter_hash='sha256:'+'f'*64)
    with pytest.raises(ValueError): contact_imitation_loss(current, [chosen]*4)


def test_il_requires_fixed_batch_and_unchanged_common_policy(experiment):
    current = session(experiment, 'IL'); chosen = sample(experiment)
    with pytest.raises(ValueError): contact_imitation_loss(current, [chosen])
    with torch.no_grad(): next(current.policy.parameters()).add_(1.)
    with pytest.raises(ValueError): contact_imitation_loss(current, [chosen]*4)


def test_foreign_and_wrong_optimizer_losses_refuse_without_step(experiment, monkeypatch):
    current = session(experiment, 'IL'); chosen = sample(experiment)
    calls = []
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *a, **k: calls.append(True))
    with pytest.raises(ValueError): contact_gradient_step(current, optimizer(current.policy), torch.tensor(0., requires_grad=True))
    loss, _ = contact_imitation_loss(current, [chosen]*4)
    with pytest.raises(ValueError): contact_gradient_step(current, optimizer(current.policy, wrong=True), loss)
    with pytest.raises(ValueError): contact_gradient_step(current, optimizer(current.policy), loss)
    assert not calls and current.updates == 0


def test_initial_checkpoint_bounded_file_roundtrip_and_no_overwrite(experiment, tmp_path):
    model = common_initial_policies(experiment)['IL']; path = tmp_path/'scratch-only.gmckpt'
    receipt = save_contact_checkpoint(path, model, experiment, initial_lineage(experiment))
    restored, metadata = load_contact_checkpoint(path, expected_sha256=receipt['sha256'], experiment=experiment, kind='initial')
    assert type(metadata) is VerifiedContactCheckpoint
    assert parameter_hash(restored) == parameter_hash(model)
    assert metadata['lineage']['optimizer_updates'] == 0
    with pytest.raises(FileExistsError): save_contact_checkpoint(path, model, experiment, initial_lineage(experiment))
    with pytest.raises(ValueError): load_contact_checkpoint(path, expected_sha256='f'*64, experiment=experiment, kind='initial')
    with pytest.raises(ValueError): load_contact_checkpoint(path, expected_sha256=receipt['sha256'], experiment=experiment, kind='final')
    link = tmp_path/'symlink.gmckpt'; link.symlink_to(path)
    with pytest.raises(OSError): load_contact_checkpoint(link, expected_sha256=receipt['sha256'], experiment=experiment, kind='initial')


def test_checkpoint_fifo_refuses_without_writer(experiment, tmp_path):
    path = tmp_path/'fifo.gmckpt'; os.mkfifo(path)
    with pytest.raises(ValueError, match='bounded regular file'):
        load_contact_checkpoint(path, expected_sha256='0'*64, experiment=experiment, kind='initial')


def alter_checkpoint(payload, mutator):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive: entries = {n: archive.read(n) for n in archive.namelist()}
    metadata = json.loads(entries['manifest.json']); mutator(metadata, entries)
    entries['manifest.json'] = json.dumps(metadata).encode(); out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items(): archive.writestr(name, data)
    return out.getvalue()


@pytest.mark.parametrize('change', ['version', 'shape', 'parameter', 'extra_entry', 'lineage'])
def test_resealed_checkpoint_substitutions_refused(experiment, change):
    model = common_initial_policies(experiment)['IL']
    payload = encode_contact_checkpoint(model, experiment, initial_lineage(experiment))
    def mutate(metadata, entries):
        if change == 'version': metadata['version'] = 'legacy-aspiration'
        elif change == 'shape': metadata['tensors'][0]['shape'] = [10**12]
        elif change == 'parameter': metadata['parameter_hash'] = 'sha256:'+'f'*64
        elif change == 'extra_entry': entries['extra.bin'] = b'unexpected'
        elif change == 'lineage': metadata['lineage']['kind'] = 'final'
    changed = alter_checkpoint(payload, mutate)
    with pytest.raises(ValueError): decode_contact_checkpoint(changed, expected_sha256=hashlib.sha256(changed).hexdigest(),
        experiment=experiment, kind='initial')


def test_partial_zero_update_state_is_preserved_but_not_final(experiment):
    current = session(experiment, 'RL')
    payload = encode_contact_checkpoint(current.policy, experiment, partial_lineage(current))
    restored, metadata = decode_contact_checkpoint(payload, expected_sha256=hashlib.sha256(payload).hexdigest(),
        experiment=experiment, kind='partial')
    assert metadata['lineage']['optimizer_updates'] == 0
    with pytest.raises(ValueError): final_lineage(current)
    current.updates = 32
    with pytest.raises(ValueError): final_lineage(current)
    with pytest.raises(ValueError): VerifiedContactCheckpoint({}, 'f'*64, object())


def test_checkpoint_lineage_refuses_relabelled_session(experiment):
    current = session(experiment, 'IL')
    current.method = 'RL'
    with pytest.raises(ValueError, match='Session/architecture/method changed'):
        partial_lineage(current)
    with pytest.raises(ValueError, match='Session/architecture/method changed'):
        final_lineage(current)


def test_measurement_cannot_construct_task_without_verified_final_freeze(experiment, monkeypatch):
    import resectionlab.contact_family_episode as module
    calls = []
    monkeypatch.setattr(module, 'build_family_source', lambda *a: calls.append(True))
    layout, goal = experiment.keys('MEASUREMENT_EVAL')[0]
    with pytest.raises(ValueError, match='HELD_OUT_EXECUTION_CLOSED'):
        make_frozen_evaluation_task(experiment, layout, goal)
    with pytest.raises(ValueError): ContactEvaluationFreeze(experiment.fingerprint, {'IL': {}, 'RL': {}}, object())
    with pytest.raises(ValueError): freeze_final_checkpoints(experiment, {})
    assert not calls


def test_cost_meter_restores_shared_functions_without_executing_native():
    from resectionlab.native_spatial_task import NativeSpatialTask
    before = NativeSpatialTask._transition
    with ContactCostMeter() as meter:
        assert NativeSpatialTask._transition is not before
        with meter.scope('empty_control'): pass
    assert NativeSpatialTask._transition is before
    assert 'empty_control' in meter.rows
    assert not any(k.endswith('_calls') for k in meter.rows['empty_control'])


@pytest.mark.parametrize('change', ['label', 'loss_tensor', 'adam_state'])
def test_label_loss_or_optimizer_state_substitution_refuses_before_update(experiment, change):
    current = session(experiment, 'IL'); chosen = sample(experiment)
    loss, _ = contact_imitation_loss(current, [chosen]*4)
    opt = optimizer(current.policy)
    if change == 'label': object.__setattr__(chosen, 'action_id', 'tensor-control-only')
    elif change == 'loss_tensor':
        with torch.no_grad(): loss.add_(1.)
    else:
        opt.state[next(current.policy.parameters())]['step'] = torch.tensor(3.)
    with pytest.raises(ValueError): contact_gradient_step(current, opt, loss)
    assert current.updates == 0
