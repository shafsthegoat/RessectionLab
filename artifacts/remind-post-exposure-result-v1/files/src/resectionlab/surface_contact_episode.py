"""Application-facing generated contact task using the existing native exporter.

This v2 task is deliberately separate from the unchanged aspiration episode.
No learner, patient loader or additional mechanics are introduced.
"""
from __future__ import annotations

import json
from types import MappingProxyType

import numpy as np

from .core import freeze_json, semantic_digest, thaw_json
from .development_episode import TOOLS, _select, _execute_sealed_development_plan
from .observed_search import observed_beam_search
from .public_surface_contact import (CONTEXT_VERSION, OBJECTIVE_VERSION,
    OBSERVATION_VERSION, make_contact_task)

FIXTURE = 'generated-public-surface-contact-v1'
SCHEMA = 'resectionlab.shared-native-development-episode.v2'
GOALS = MappingProxyType({'near': (6, 6, 3), 'costly': (6, 6, 7)})
SEARCH_BUDGET = MappingProxyType({'max_calls': 256, 'beam_width': 96, 'seconds': 4.})


def execute_surface_contact_episode(*, selector='scripted', goal_id='near', cancelled=None):
    """Return (existing CaseData, v2 episode) for one fixed public goal.

The sidecar can use its existing case installation, canonical JSON digest,
workspace persistence and asset-transfer path. Current v1 UI admission must
explicitly opt into this task before displaying it; no silent schema coercion.
"""
    if selector not in ('scripted', 'SEARCH') or goal_id not in GOALS:
        raise ValueError('Choose scripted/SEARCH and the near/costly public goal')
    task = make_contact_task(goal=GOALS[goal_id], cancelled=cancelled)
    declaration = {'fixture': FIXTURE, 'selector': selector, 'goal_id': goal_id,
        'objective': task.objective.record(), 'observation_version': OBSERVATION_VERSION,
        'max_steps': task.max_steps, 'search_budget': dict(SEARCH_BUDGET)}
    context = task.development_context(semantic_digest(declaration))
    context.require_task(task)
    initial = task._engine.state_hash
    rejected = task._engine.preview_stroke(TOOLS[1].tool_id, (6., 6., 2.),
        entry_mm=(6., 6., 1.5), interaction_mode='probe')
    if rejected.feasible or task._engine.state_hash != initial:
        raise RuntimeError('Pre-opening probe refusal failed to preserve native state')

    nominal = task.planning_clone()
    if selector == 'scripted':
        actions = []
        point = (*GOALS[goal_id][:2], GOALS[goal_id][2] - 1)
        for tool in TOOLS:
            context.require_observation(nominal.observation())
            action = _select(nominal, tool.tool_id, point)
            actions.append(action)
            nominal.step(action)
        accounting = {'selector': 'scripted_public_contact_demonstration',
            'model_transition_calls': 2, 'actor_forward_calls': 0, 'optimizer_updates': 0}
    else:
        actions, accounting = observed_beam_search(nominal, **SEARCH_BUDGET,
            objective_source='explicit_public_retained_surface_goal', transition_mode='lazy_planning')
        for action in actions:
            context.require_observation(nominal.observation())
            nominal.step(action)
        accounting = {**accounting, 'selector': 'existing_observed_beam_search', 'optimizer_updates': 0}
    if not nominal.terminated:
        raise RuntimeError('Contact episode must stop explicitly or exhaust its two-action horizon')
    plan = freeze_json({'decision_model_hash': task.decision_model_hash,
        'source_hash': task.case.source_hash, 'actions': list(actions),
        'history': nominal.metrics()['history'], 'observation_contract': OBSERVATION_VERSION,
        'max_steps': task.max_steps})
    seal = semantic_digest(plan)
    display, episode = _execute_sealed_development_plan(task, selector, plan, seal, accounting, rejected)
    # This public objective has no separate reference score. Bind full rewards
    # as well as the physical fields checked by the existing shared exporter.
    if semantic_digest(plan['history']) != semantic_digest(episode['history']):
        raise RuntimeError('Executed contact history differs from the complete public strategy')
    context.require_observation(task.observation())
    goal_ras = (task.case.affine_ras_mm @ np.array([*task.objective.native_index, 1.]))[:3]
    episode.pop('episodeId')
    episode.update({
        'schema': SCHEMA, 'taskKind': 'public_retained_surface_contact', 'fixture': FIXTURE,
        'publicGoal': {'goalId': goal_id, 'nativeIndex': list(task.objective.native_index),
            'rasMm': goal_ras.tolist(), 'frame': 'RAS+', 'physicalUnits': 'mm',
            'goalGridHash': context.goal_grid_hash, 'objectiveHash': context.objective_hash,
            'meaning': 'committed geometric probe contact with a retained source cell'},
        'taskContract': {'objectiveVersion': OBJECTIVE_VERSION,
            'observationVersion': OBSERVATION_VERSION, 'contextVersion': CONTEXT_VERSION,
            'maxSteps': 2, 'objective': task.objective.record(), 'declaration': declaration,
            'detachedObservationBinding': context._record(),
            'nominalTargetRole': 'permitted generated fixture background; unused by contact reward',
            'learnedPolicySupported': False, 'trainingAdmission': False},
        'interpretation': 'Generated public geometric task: retain and touch the declared cell. '
            'Probe does not measure force or reveal anatomy. All removal is charged; '
            'STOP may be optimal when contact costs exceed its declared value. '
            'The displayed nominal target is fixture background, not this task objective.',
    })
    episode['planning'].pop('sealedBeforeReferenceScoring')
    episode['planning'].update({'sealedBeforeExecution': True, 'referenceScoringPerformed': False,
        'objectiveSource': 'public_goal_and_shared_observed_native_state_only',
        'searchBudget': dict(SEARCH_BUDGET), 'learnedPolicyExecuted': False})
    episode['sequentialEffect']['publicGoalContactedAndRetained'] = task.metrics()['goal_contacted_and_retained']
    episode = thaw_json(freeze_json(episode))
    episode['episodeId'] = semantic_digest(episode)
    return display, episode


def episode_envelope(episode):
    """Use the existing bridge's exact canonical-JSON publication convention."""
    body = {key: value for key, value in episode.items() if key != 'episodeId'}
    if semantic_digest(body) != episode.get('episodeId'):
        raise ValueError('Contact episode identity changed before publication')
    return {'episode': thaw_json(freeze_json(episode)), 'episodeCanonicalJson': json.dumps(
        body, sort_keys=True, separators=(',', ':'), allow_nan=False)}
