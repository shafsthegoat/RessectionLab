"""Frozen-checkpoint evaluation release and opt-in v3 family native replay.

The current desktop v2 fixed fixture is NOT an admission path for this schema.
One authoritative exporter executes the same sealed native strategy; only public
family provenance and versioned metadata differ. No private reference is scored.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

from .contact_checkpoint import load_contact_checkpoint, VerifiedContactCheckpoint
from .contact_learning_contract import ContactExperiment, PROTOCOL
from .core import CaseData, SourceRef, array_digest, freeze_json, semantic_digest, thaw_json
from .development_episode import _execute_sealed_development_plan
from .public_contact_family import (FAMILY_VERSION, SOURCE_CANDIDATE_VERSION, build_family_source,
    bind_family_context, make_family_task)
from .public_surface_contact import (PublicSurfaceGoal, SurfaceContactTask, SurfaceContactDevelopmentContext,
    OBJECTIVE_VERSION, OBSERVATION_VERSION, CONTEXT_VERSION)
from .spatial_observations import CHANNEL_NAMES

SCHEMA = 'resectionlab.shared-native-contact-learning-episode.v3'
FREEZE_VERSION = 'public-contact-final-checkpoint-freeze-v1'
_CAPABILITY = object()


@dataclass(frozen=True)
class ContactEvaluationFreeze:
    experiment_hash: str
    checkpoints: Mapping
    _capability: object = field(repr=False)
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        if self._capability is not _CAPABILITY or set(self.checkpoints) != {'IL', 'RL'}:
            raise ValueError('Held-out release requires both verified final checkpoint files')
        object.__setattr__(self, 'checkpoints', freeze_json(self.checkpoints))
        object.__setattr__(self, '_identity', semantic_digest(self.record()))

    def record(self):
        return {'version': FREEZE_VERSION, 'experiment_hash': self.experiment_hash,
                'checkpoints': self.checkpoints, 'checkpoint_selection': 'none_fixed_32_update_endpoint'}

    def require(self, experiment):
        if (type(experiment) is not ContactExperiment or self._capability is not _CAPABILITY
                or experiment.fingerprint != self.experiment_hash or semantic_digest(self.record()) != self._identity):
            raise ValueError('Frozen evaluation release changed or belongs to another experiment')


def freeze_final_checkpoints(experiment, receipts):
    if type(experiment) is not ContactExperiment or set(receipts) != {'IL', 'RL'}:
        raise ValueError('Exactly IL and RL final artifacts required before held-out release')
    records = {}
    for method in ('IL', 'RL'):
        receipt = receipts[method]
        model, metadata = load_contact_checkpoint(receipt['path'], expected_sha256=receipt['sha256'],
                                                   experiment=experiment, kind='final')
        if metadata['lineage']['method'] != method or metadata['parameter_hash'] != receipt['parameter_hash']:
            raise ValueError('Final checkpoint method/parameter record substituted')
        records[method] = {'file_sha256': receipt['sha256'], 'parameter_hash': metadata['parameter_hash'],
                          'lineage_hash': semantic_digest(metadata['lineage'])}
        del model
    return ContactEvaluationFreeze(experiment.fingerprint, records, _CAPABILITY)


def make_frozen_evaluation_task(experiment, layout_id, goal_id, *, release=None, cancelled=None):
    row = experiment.row(layout_id)
    if goal_id not in row['goals']: raise ValueError('Unknown fixed public goal')
    if row['role'] != 'MEASUREMENT_EVAL':
        return make_family_task(layout_id, goal_id, cancelled=cancelled)
    # Must fail before held-out source/task construction when no release exists.
    if type(release) is not ContactEvaluationFreeze:
        raise ValueError('HELD_OUT_EXECUTION_CLOSED: both completed checkpoints must be verified and frozen')
    release.require(experiment)
    case = build_family_source(layout_id)
    if case.source_hash != row['source_hash']: raise ValueError('Frozen held-out source changed')
    return SurfaceContactTask(case, objective=PublicSurfaceGoal(case.source_hash,
        tuple(row['goals'][goal_id]['native_index'])), cancelled=cancelled)


def bind_family_episode(experiment, task, *, layout_id, goal_id, release=None):
    row = experiment.row(layout_id)
    if type(task) is not SurfaceContactTask or goal_id not in row['goals']:
        raise TypeError('Exact shared public task and fixed goal required')
    if row['role'] != 'MEASUREMENT_EVAL':
        return bind_family_context(task, layout_id=layout_id, goal_id=goal_id, experiment_hash=experiment.fingerprint)
    if type(release) is not ContactEvaluationFreeze: raise ValueError('HELD_OUT_EXECUTION_CLOSED')
    release.require(experiment)
    if task.case.source_hash != row['source_hash'] or task.objective.fingerprint != row['goals'][goal_id]['objective_hash']:
        raise ValueError('Held-out task differs from the frozen source/objective')
    declaration = freeze_json({'version': FAMILY_VERSION, 'experiment_hash': experiment.fingerprint,
        'proposal_mode': SOURCE_CANDIDATE_VERSION, 'source_candidate_version': SOURCE_CANDIDATE_VERSION,
        'layout_id': layout_id, 'goal_id': goal_id, 'role': row['role'], 'family_hash': experiment.manifest['family_hash'],
        'recipe_hash': row['recipe_hash'], 'source_hash': row['source_hash'],
        'objective_hash': task.objective.fingerprint, 'decision_model_hash': task.decision_model_hash,
        'training_admission': False, 'scope': 'frozen_generated_measurement_only',
        'checkpoint_freeze_hash': semantic_digest(release.record())})
    context = task.development_context(semantic_digest(declaration)); context.require_task(task)
    return context, declaration


def export_family_strategy(experiment, task, *, layout_id, goal_id, selector, plan, seal,
                           accounting, context, release=None, checkpoint_metadata=None):
    """Execute/replay exactly the existing full native exporter, then label v3."""
    bound, declaration = bind_family_episode(experiment, task, layout_id=layout_id, goal_id=goal_id, release=release)
    if type(context) is not SurfaceContactDevelopmentContext or semantic_digest(context._record()) != semantic_digest(bound._record()):
        raise ValueError('Exporter received a foreign public episode context')
    context.require_task(task)
    initial_observation = task.observation()
    context.require_observation(initial_observation)
    if task.terminated or initial_observation.base.base.state_features[0] != 0:
        raise ValueError('Family exporter requires an unchanged fresh task')
    if selector not in ('IL', 'RL', 'SEARCH', 'STOP'): raise ValueError('Unknown fixed protocol method')
    if accounting.get('optimizer_updates', 0) != 0:
        raise ValueError('Online family export cannot include optimizer updates')
    accounting = {**accounting, 'optimizer_updates': 0}
    authorship = None
    if selector in ('IL', 'RL'):
        if type(checkpoint_metadata) is not VerifiedContactCheckpoint:
            raise TypeError('Learned export requires bounded-decoder verified metadata, not caller JSON')
        checkpoint_metadata.require(experiment, kind='final')
        if (checkpoint_metadata is None or checkpoint_metadata['lineage']['method'] != selector
                or checkpoint_metadata['lineage']['kind'] != 'final'
                or checkpoint_metadata['lineage']['experiment_hash'] != experiment.fingerprint
                or checkpoint_metadata['parameter_hash'] != accounting.get('parameter_hash')
                or checkpoint_metadata['architecture_hash'] != accounting.get('architecture_hash')
                or accounting.get('context_declaration_hash') != context.declaration_hash
                or accounting.get('optimizer_updates') != 0):
            raise ValueError('Learned export requires matching verified final checkpoint lineage')
        if release is not None:
            if type(release) is not ContactEvaluationFreeze: raise TypeError('Exact checkpoint freeze required')
            release.require(experiment)
            if (checkpoint_metadata['parameter_hash'] != release.checkpoints[selector]['parameter_hash']
                    or checkpoint_metadata.file_sha256 != release.checkpoints[selector]['file_sha256']):
                raise ValueError('Published learned weights differ from checkpoint freeze')
        authorship = {'version': 'public-contact-learned-authorship-v1', 'method': selector,
            'checkpointFileSha256': checkpoint_metadata.file_sha256,
            'checkpointVersion': checkpoint_metadata['version'],
            'architectureHash': checkpoint_metadata['architecture_hash'],
            'parameterHash': checkpoint_metadata['parameter_hash'],
            'experimentHash': experiment.fingerprint, 'familyHash': experiment.manifest['family_hash'],
            'trainingLineageHash': semantic_digest(checkpoint_metadata['lineage']),
            'completedUpdates': checkpoint_metadata['lineage']['optimizer_updates'],
            'checkpointKind': 'final', 'inferenceOptimizerUpdates': 0,
            'verificationScope': 'bounded_checkpoint_bytes_and_declared_lineage_owned_native_replay_not_signed_training_proof'}
    elif checkpoint_metadata is not None or accounting.get('parameter_hash') is not None:
        raise ValueError('SEARCH and STOP cannot publish learned checkpoint authorship')
    base = initial_observation.base.base
    public_values = [*base.image_channels, initial_observation.public_goal_grid,
                     initial_observation.base.observed_probe_contact_grid]
    public_coverage = [*base.coverage, base.coverage[0], base.coverage[0]]
    observation_binding = {'version': 'public-contact-initial-observation-binding-v1',
        'observationHash': initial_observation.fingerprint, 'sourceHash': context.source_hash,
        'decisionModelHash': context.decision_model_hash, 'declarationHash': context.declaration_hash,
        'objectiveHash': context.objective_hash, 'goalGridHash': context.goal_grid_hash,
        'cropOriginNative': list(context.crop_origin_native), 'cropShape': list(base.image_channels.shape[1:]),
        'cropAffine': base.affine_ras_mm.tolist(), 'cropAffineHash': context.crop_affine_hash,
        'frame': 'RAS+', 'physicalUnits': 'mm',
        'channelNames': [*CHANNEL_NAMES, 'public_goal', 'committed_probe_contact'],
        'channelAvailable': [*base.channel_available.tolist(), True, True],
        'channelValueHashes': [array_digest(values) for values in public_values],
        'channelCoverageHashes': [array_digest(coverage) for coverage in public_coverage],
        'scope': 'raw_public_crop_values_before_policy_normalization_not_full_native_coverage',
        'extraChannelCoverage': 'inherits_required_structural_intensity_coverage'}
    source = task.case
    display = CaseData(f'{FAMILY_VERSION}:{layout_id}', source.structural_intensity,
        {'generated_nominal_target': source.nominal_target > 0}, source.affine_ras_mm,
        (SourceRef(layout_id, 'generated://' + FAMILY_VERSION + '/' + layout_id,
                   native_frame='RAS+', provenance='simulated'),), frame='RAS+', brain_mask=source.observed_support,
        metadata={'is_synthetic': True, 'evidence_kind': 'generated_software_fixture', 'patient_admission': False,
            'source_task_hash': source.source_hash, 'family_hash': experiment.manifest['family_hash'],
            'layout_id': layout_id, 'scope': 'public_generated_contact_learning_no_patient_transfer'})
    display, episode = _execute_sealed_development_plan(task, selector, plan, seal, accounting,
                                                       rejected=None, display_case=display)
    if semantic_digest(plan['history']) != semantic_digest(episode['history']):
        raise RuntimeError('Executed public contact rewards/history differ from the complete seal')
    context.require_observation(task.observation())
    goal_ras = (task.case.affine_ras_mm @ np.r_[task.objective.native_index, 1.])[:3]
    episode.pop('episodeId')
    episode.update({'schema': SCHEMA, 'taskKind': 'generated_family_public_retained_surface_contact',
        'fixture': FAMILY_VERSION, 'familyHash': experiment.manifest['family_hash'],
        'layoutId': layout_id, 'splitRole': declaration['role'],
        'learnedAuthorship': authorship, 'initialObservationBinding': observation_binding,
        'publicGoal': {'goalId': goal_id, 'nativeIndex': list(task.objective.native_index),
            'rasMm': goal_ras.tolist(), 'frame': 'RAS+', 'physicalUnits': 'mm',
            'goalGridHash': context.goal_grid_hash, 'objectiveHash': context.objective_hash,
            'meaning': 'committed geometric probe contact with a retained source cell'},
        'taskContract': {'objectiveVersion': OBJECTIVE_VERSION, 'observationVersion': OBSERVATION_VERSION,
            'proposalMode': SOURCE_CANDIDATE_VERSION, 'sourceCandidateVersion': SOURCE_CANDIDATE_VERSION,
            'contextVersion': CONTEXT_VERSION, 'maxSteps': 2, 'objective': task.objective.record(),
            'declaration': thaw_json(declaration), 'detachedObservationBinding': context._record(),
            'nominalTargetRole': 'zero compatibility field unused by public contact objective',
            'learnedPolicySupported': True, 'trainingAdmission': False,
            'checkpoint': None if checkpoint_metadata is None else thaw_json(checkpoint_metadata['lineage'])},
        'interpretation': 'Generated public retained-contact task. Geometric probe contact is not a sensor '
            'measurement or clinical outcome; the analytic scan exposes generated support. All removal is charged.'})
    episode['planning'].pop('sealedBeforeReferenceScoring')
    episode['planning'].update({'sealedBeforeExecution': True, 'referenceScoringPerformed': False,
        'objectiveSource': 'same_public_goal_observed_state_and_geometry',
        'learnedPolicyExecuted': selector in ('IL', 'RL'),
        'training_or_checkpoint_lineage_verified': selector in ('IL', 'RL'),
        'experimentHash': experiment.fingerprint})
    episode['sequentialEffect']['publicGoalContactedAndRetained'] = task.metrics()['goal_contacted_and_retained']
    episode = thaw_json(freeze_json(episode)); episode['episodeId'] = semantic_digest(episode)
    return display, episode
