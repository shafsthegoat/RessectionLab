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
from .public_contact_family import (FAMILY_VERSION, SOURCE_CANDIDATE_VERSION, GOAL_IDS, family_digest, family_manifest,
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

RELATION_FIT_VERSION = 'generated-public-contact-goal-relation-fit-v1'
RELATION_VARIANT = 'public_goal_tip_relation_v1'

FULL_TEACHER_VERSION = 'generated-public-contact-full-teacher-refit-v1'
FULL_TEACHER_PROTOCOL = freeze_json({**thaw_json(PROTOCOL), 'version': FULL_TEACHER_VERSION,
    'methods': ['IL'], 'batch_size': 40, 'select_passes': 0, 'measurement_passes': 0,
    'new_teacher_searches': 0, 'sample_schedule': 'same_complete_ordered_40_teacher_states_each_update',
    'imitation_objective': 'unchanged_mean_action_categorical_cross_entropy',
    'initialization': 'same_fresh_scratch_seed_not_a_pilot_checkpoint',
    'loss_forward_calls': 1280, 'fixed_before_after_TRAIN_forwards': 80,
    'method_seconds': 180., 'scope': 'extra_compute_TRAIN_optimization_diagnosis_no_comparison_claim'})


RELATION_FIT_PROTOCOL = freeze_json({**thaw_json(FULL_TEACHER_PROTOCOL), 'version': RELATION_FIT_VERSION,
    'policy_variant': RELATION_VARIANT, 'shared_initialization': 'exact_original_shared_tensors;added_columns_zero',
    'baseline_shared_initial_parameter_hash': 'sha256:e7215950e221045b8e9612e4482837f9b1ff2c52278454b26efc746598cf1487',
    'comparison': 'fixed_full40_baseline_same_updates_samples_forwards;report_extra_parameters_and_actual_cost'})


def _validated_teacher_states(manifest, states):
    """Bind the complete previously sealed corpus, not arbitrary repeated40 samples.

    Original teacher seals/bindings identify provenance. The owned collector must
    still reconstruct and verify them; this metadata is not a native certificate.
    """
    rows = freeze_json(states)
    fields = {'layout_id', 'goal_id', 'step', 'observation_hash', 'action_id',
              'original_binding_hash', 'strategy_seal'}
    train = {(row['layout_id'], goal) for row in manifest['source_bindings']
             if row['role'] == 'TRAIN' for goal in GOAL_IDS}
    if (len(rows) != 40 or any(set(row) != fields for row in rows)
            or any((row['layout_id'], row['goal_id']) not in train for row in rows)
            or any(type(row['step']) is not int or row['step'] not in (0, 1) for row in rows)
            or any(not all(_hash(row[key]) for key in ('observation_hash', 'original_binding_hash', 'strategy_seal'))
                   or not isinstance(row['action_id'], str) or not row['action_id'] for row in rows)
            or len({(row['layout_id'], row['goal_id'], row['step']) for row in rows}) != 40
            or {(row['layout_id'], row['goal_id']) for row in rows if row['step'] == 0} != train
            or sum(row['step'] == 0 for row in rows) != 24
            or sum(row['action_id'] == 'STOP' for row in rows) != 8
            or any(row['action_id'] == 'STOP' and row['step'] != 0 for row in rows)):
        raise ValueError('Full-teacher refit requires the exact40-state TRAIN corpus, including eight STOP labels')
    for key in train:
        group = [row for row in rows if (row['layout_id'], row['goal_id']) == key]
        if (len({row['original_binding_hash'] for row in group}) != 1
                or len({row['strategy_seal'] for row in group}) != 1
                or (len(group) == 1) != (group[0]['action_id'] == 'STOP')):
            raise ValueError('Teacher corpus splices or truncates a complete strategy')
    ordered_tasks = [(row['layout_id'], goal) for row in manifest['source_bindings']
                     if row['role'] == 'TRAIN' for goal in GOAL_IDS]
    positions = [(ordered_tasks.index((row['layout_id'], row['goal_id'])), row['step']) for row in rows]
    if positions != sorted(positions):
        raise ValueError('Teacher states must retain canonical teacher/action order')
    return rows


def _hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def policy_config():
    return SpatialPolicyConfig(**thaw_json(PROTOCOL['architecture_config']))


def policy_class(variant=None):
    if variant is None: return GoalModeSpatialPolicy
    if variant == RELATION_VARIANT:
        from .goal_relation_spatial_policy import GoalRelationSpatialPolicy
        return GoalRelationSpatialPolicy
    raise ValueError('Unknown explicit policy variant')


def policy_architecture_hash(variant=None):
    with torch.random.fork_rng(devices=[]):
        return policy_class(variant)(policy_config()).architecture_hash


@dataclass(frozen=True)
class ContactExperiment:
    """Full source-only manifest frozen before the first teacher or rollout."""
    manifest: Mapping
    teacher_states: tuple[Mapping, ...] | None = None
    policy_variant: str | None = None
    _identity: str = field(init=False, repr=False)
    _architecture_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        if self.policy_variant not in (None, RELATION_VARIANT) or (self.policy_variant is not None and self.teacher_states is None):
            raise ValueError('Goal-relation fit requires the exact full40 TRAIN corpus')
        candidate = freeze_json(self.manifest)
        # Validate canonical sources/roles, not a caller-written TRAIN label.
        if semantic_digest(candidate) != semantic_digest(family_manifest()):
            raise ValueError('Experiment must bind the exact canonical generated family manifest')
        object.__setattr__(self, 'manifest', candidate)
        if self.teacher_states is not None:
            object.__setattr__(self, 'teacher_states', _validated_teacher_states(candidate, self.teacher_states))
        object.__setattr__(self, '_architecture_hash', policy_architecture_hash(self.policy_variant))
        object.__setattr__(self, '_identity', semantic_digest(self.record()))

    def record(self):
        record = {'version': VERSION, 'protocol': PROTOCOL, 'family_manifest': self.manifest,
                  'architecture_hash': self._architecture_hash}
        if self.teacher_states is not None:
            record.update(version=FULL_TEACHER_VERSION, protocol=FULL_TEACHER_PROTOCOL,
                          teacher_states=self.teacher_states)
        if self.policy_variant is not None:
            record.update(version=RELATION_FIT_VERSION, protocol=RELATION_FIT_PROTOCOL, policy_variant=self.policy_variant)
        return record

    @property
    def policy_type(self):
        return policy_class(self.policy_variant)

    @property
    def protocol(self):
        if self.policy_variant is not None: return RELATION_FIT_PROTOCOL
        return PROTOCOL if self.teacher_states is None else FULL_TEACHER_PROTOCOL

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


def freeze_full_teacher_refit(teacher_states):
    """Explicit opt-in; default pilot records and checkpoint identities stay exact."""
    return ContactExperiment(family_manifest(), teacher_states=teacher_states)


def freeze_goal_relation_fit(teacher_states):
    return ContactExperiment(family_manifest(), teacher_states=teacher_states, policy_variant=RELATION_VARIANT)


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
            'proposal_mode': SOURCE_CANDIDATE_VERSION, 'source_candidate_version': SOURCE_CANDIDATE_VERSION,
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
    for method in experiment.protocol['methods']:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(PROTOCOL['seed'])
            policies[method] = experiment.policy_type(policy_config())
    if len({parameter_hash(policy) for policy in policies.values()}) != 1:
        raise RuntimeError('Common scratch initialization differed')
    return policies


def expected_initial_parameter_hash(variant=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROTOCOL['seed'])
        return parameter_hash(policy_class(variant)(policy_config()))
