"""Separate, generated TRAIN-only admission for the frozen public contact family.

A declaration is a consistency contract, not source authenticity or permission to
run. The owned runner must construct canonical tasks and retain native replays.
Legacy opening/real-record admission is neither modified nor reused here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np
import torch

from .core import freeze_json, semantic_digest, thaw_json
from .goal_mode_spatial_policy import GoalModeSpatialPolicy
from .public_contact_family import (FAMILY_VERSION, GOAL_IDS, family_digest, family_manifest,
    layout_metadata, bind_family_context)
from .public_surface_contact import HASH, SurfaceContactDevelopmentContext, SurfaceContactObservation
from .spatial_policy import SpatialPolicyConfig, parameter_hash

VERSION = 'generated-public-contact-learning-v1'
PROTOCOL = freeze_json({'version': VERSION, 'seed': 20261009,
    'methods': ['IL', 'RL'], 'updates': 32, 'batch_size': 4,
    'teacher_tasks': 24, 'rl_rollout_transition_cap': 256,
    'optimizer': 'Adam', 'learning_rate': .001, 'betas': [.9, .999], 'eps': 1e-8,
    'weight_decay': 0., 'amsgrad': False, 'maximize': False, 'foreach': False, 'fused': False, 'gamma': 1., 'entropy_weight': .01,
    'value_weight': .5, 'max_gradient_norm': 5.,
    'architecture_config': {'encoder_channels': [8, 16], 'hidden_features': 32,
        'ray_samples': 5, 'physical_reference_mm': 10., 'critic_candidate_context': True},
    'search': {'max_calls': 256, 'beam_width': 96, 'seconds': 4.},
    'teacher_seconds': 120., 'method_seconds': 60., 'online_seconds': 12., 'memory_bytes': 1073741824,
    'online_method_order': 'fixed_cyclic_rotation_by_goal_index',
    'threads': 1, 'interop_threads': 1, 'device': 'cpu', 'dtype': 'float32', 'attempts': 1, 'select_passes': 1, 'measurement_passes': 1,
    'checkpoint_selection': 'fixed_final_update_no_selection', 'real_patient_count': 0,
    'transition_lineage': 'same_native_generated_public_contact',
    'unresolved_teachers': 'preserve_slot_without_label_no_replacement',
    'hybrid_arm': False})


def _hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def policy_config():
    return SpatialPolicyConfig(**thaw_json(PROTOCOL['architecture_config']))


def policy_architecture_hash():
    with torch.random.fork_rng(devices=[]):
        return GoalModeSpatialPolicy(policy_config()).architecture_hash


@dataclass(frozen=True)
class ContactExperiment:
    """Full source-only manifest frozen before the first teacher or rollout."""
    manifest: Mapping
    _identity: str = field(init=False, repr=False)
    _architecture_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        candidate = freeze_json(self.manifest)
        # Validate canonical sources/roles, not a caller-written TRAIN label.
        if semantic_digest(candidate) != semantic_digest(family_manifest()):
            raise ValueError('Experiment must bind the exact canonical generated family manifest')
        object.__setattr__(self, 'manifest', candidate)
        object.__setattr__(self, '_architecture_hash', policy_architecture_hash())
        object.__setattr__(self, '_identity', semantic_digest(self.record()))

    def record(self):
        return {'version': VERSION, 'protocol': PROTOCOL, 'family_manifest': self.manifest,
                'architecture_hash': self._architecture_hash}

    def assert_intact(self):
        if (self.manifest['family_hash'] != family_digest()
                or semantic_digest(self.record()) != self._identity):
            raise ValueError('Frozen experiment, family or protocol changed')

    @property
    def fingerprint(self):
        self.assert_intact()
        return self._identity

    def row(self, layout_id):
        self.assert_intact()
        canonical = layout_metadata(layout_id)
        rows = [row for row in self.manifest['source_bindings'] if row['layout_id'] == layout_id]
        if len(rows) != 1 or rows[0]['role'] != canonical['role']:
            raise ValueError('Layout or role differs from the frozen family')
        return rows[0]

    def keys(self, role):
        if role not in ('TRAIN', 'SELECT', 'MEASUREMENT_EVAL'):
            raise ValueError('Unknown experiment role')
        self.assert_intact()
        return tuple((row['layout_id'], goal) for row in self.manifest['source_bindings']
                     if row['role'] == role for goal in GOAL_IDS)


def freeze_contact_experiment():
    return ContactExperiment(family_manifest())


@dataclass(frozen=True)
class ContactTrainBinding:
    experiment: ContactExperiment
    layout_id: str
    goal_id: str
    context: SurfaceContactDevelopmentContext
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_identity', semantic_digest(self.record()))

    def _validate(self):
        if type(self.experiment) is not ContactExperiment or type(self.context) is not SurfaceContactDevelopmentContext:
            raise TypeError('Exact frozen family experiment and public observation context required')
        row = self.experiment.row(self.layout_id)
        # Refuse non-TRAIN before context/observation/actor consumption.
        if row['role'] != 'TRAIN':
            raise ValueError('TRAIN_ONLY: SELECT and MEASUREMENT_EVAL cannot enter loss or update')
        if self.goal_id not in GOAL_IDS:
            raise ValueError('Unknown prospectively fixed goal')
        self.context.assert_intact()
        goal = row['goals'][self.goal_id]
        declaration = {'version': FAMILY_VERSION, 'experiment_hash': self.experiment.fingerprint,
            'layout_id': self.layout_id, 'goal_id': self.goal_id, 'role': 'TRAIN',
            'family_hash': family_digest(), 'recipe_hash': row['recipe_hash'],
            'source_hash': row['source_hash'], 'objective_hash': goal['objective_hash'],
            'decision_model_hash': self.context.decision_model_hash,
            'training_admission': False, 'scope': 'generated_forward_context_only'}
        expected = (row['source_hash'], goal['objective_hash'], goal['goal_grid_hash'],
                    tuple(row['crop_origin_native']), row['crop_affine_hash'], semantic_digest(declaration))
        actual = (self.context.source_hash, self.context.objective_hash, self.context.goal_grid_hash,
                  self.context.crop_origin_native, self.context.crop_affine_hash, self.context.declaration_hash)
        if actual != expected:
            raise ValueError('Training context differs from frozen role/source/goal/crop/frame declaration')

    def record(self):
        return {'version': VERSION, 'experiment_hash': self.experiment.fingerprint,
            'layout_id': self.layout_id, 'goal_id': self.goal_id, 'role': 'TRAIN',
            'context': self.context._record()}

    def assert_intact(self):
        self._validate()
        if semantic_digest(self.record()) != self._identity:
            raise ValueError('Training binding changed')

    @property
    def fingerprint(self):
        self.assert_intact()
        return self._identity

    def require_observation(self, observation):
        self.assert_intact()
        self.context.require_observation(observation)


def bind_training_task(experiment, task, *, layout_id, goal_id):
    if type(experiment) is not ContactExperiment or experiment.row(layout_id)['role'] != 'TRAIN':
        raise ValueError('TRAIN_ONLY: no task or observation may be bound for held-out learning')
    context, _ = bind_family_context(task, layout_id=layout_id, goal_id=goal_id,
                                    experiment_hash=experiment.fingerprint)
    return ContactTrainBinding(experiment, layout_id, goal_id, context)


@dataclass(frozen=True)
class ContactSample:
    observation: SurfaceContactObservation
    action_id: str
    binding: ContactTrainBinding

    def validate(self):
        if type(self.binding) is not ContactTrainBinding:
            raise TypeError('TRAIN-bound sample required')
        self.binding.require_observation(self.observation)
        if self.action_id not in self.observation.action_ids:
            raise ValueError('Chosen action is absent from the bound observation')
        index = self.observation.action_ids.index(self.action_id)
        if not self.observation.action_mask[index]:
            raise ValueError('Chosen action is masked')
        return index


@dataclass(frozen=True)
class ContactTransition:
    sample: ContactSample
    reward: float
    terminated: bool

    def validate(self):
        if (type(self.sample) is not ContactSample or type(self.terminated) is not bool
                or isinstance(self.reward, bool) or not np.isfinite(self.reward)):
            raise ValueError('Finite native reward and explicit termination required outside actor input')
        self.sample.validate()


def common_initial_policies(experiment):
    if type(experiment) is not ContactExperiment:
        raise TypeError('Frozen experiment required for common initialization')
    experiment.assert_intact()
    # Preserve caller RNG; both methods start from the same new architecture.
    policies = {}
    for method in ('IL', 'RL'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(PROTOCOL['seed'])
            policies[method] = GoalModeSpatialPolicy(policy_config())
    if parameter_hash(policies['IL']) != parameter_hash(policies['RL']):
        raise RuntimeError('Common scratch initialization differed')
    return policies


def expected_initial_parameter_hash():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROTOCOL['seed'])
        return parameter_hash(GoalModeSpatialPolicy(policy_config()))
