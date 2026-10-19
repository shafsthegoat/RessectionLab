"""Bounded anatomy A/B verification over the desktop's authoritative engine.

Geometric previews are evaluator evidence only. The frozen pilot greedy policy
receives only SurgicalRLEnv's declared local observation, never these previews.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .core import array_digest, semantic_digest, thaw_json
from .surgical_environment import SCHEMA as ENVIRONMENT_SCHEMA, standardize
from .surgical_scene import from_case
from .surgical_observed_policy import ACTION_IDS, candidates, observed_greedy

SCHEMA = 'resectionlab.surgical-anatomy-comparison.v1'


def _bound_inputs(before, after, changed_layers):
    """Require acquired-data, registration, task and undeclared-mask equality."""
    cases = (before.case, after.case)
    adapters = [thaw_json(c.metadata).get('surgical_adapter') or {} for c in cases]
    if any(a.get('type') != 'native-patient-research-derivative' for a in adapters):
        raise ValueError('Comparison requires two source-bound prepared patient environments')
    if not adapters[0].get('sourcePatient') or adapters[0]['sourcePatient'] != adapters[1].get('sourcePatient'):
        raise ValueError('Comparison requires the same patient')
    manifests = [a['sourceManifest'] for a in adapters]
    def acquisition(manifest):
        return {'patient': manifest['patient_id'], 'reference': manifest['reference_series'],
                'series': manifest['series'], 'transforms': manifest['transforms']}
    if acquisition(manifests[0]) != acquisition(manifests[1]):
        raise ValueError('Acquired source hashes, series or registrations differ')
    if any(not row.get('original', {}).get('sha256') for m in manifests for row in m['series']):
        raise ValueError('Acquired source hashes must be explicit')
    if cases[0].frame != cases[1].frame or not np.array_equal(cases[0].affine, cases[1].affine) or array_digest(cases[0].mri) != array_digest(cases[1].mri):
        raise ValueError('Acquired reference image or simulation grid differs')
    if set(cases[0].compartments) != set(cases[1].compartments) or not set(changed_layers) <= set(cases[0].compartments):
        raise ValueError('Declared anatomy roles differ')
    changed = {}
    for name in cases[0].compartments:
        a, b = (c.compartments[name] for c in cases)
        delta = int(np.count_nonzero(a != b))
        changed[name] = {'changedCells': delta, 'addedCells': int(np.count_nonzero(b & ~a)),
                         'removedCells': int(np.count_nonzero(a & ~b))}
        reports = [adapter['sourceLayers'][name] for adapter in adapters]
        if reports[0]['layerId'] != reports[1]['layerId']:
            raise ValueError('Selected source layer identity differs')
        if name not in changed_layers and (delta or reports[0] != reports[1]):
            raise ValueError('Undeclared anatomy changed: ' + name)
    # Preserve task mapping/profile exactly. Display smoothing never enters here.
    records = [thaw_json(e.artifacts).get('surgical_toolbox') or {} for e in (before, after)]
    if any(r.get('environmentSchema') != ENVIRONMENT_SCHEMA for r in records):
        raise ValueError('Prepare both cases through the standard surgical environment interface')
    if records[0]['mapping'] != records[1]['mapping'] or records[0]['physicsProfile'] != records[1]['physicsProfile']:
        raise ValueError('Surgical task mapping or physics differs')
    scenes = [standardize(from_case(c, r['mapping']), physics_profile=r['physicsProfile']) for c, r in zip(cases, records)]
    if scenes[0].properties != scenes[1].properties:
        raise ValueError('Surgical scene assumptions differ')
    from .surgical_rl_protection import restore_protection_contract
    for scene, record in zip(scenes, records):
        for snapshot in record['branches'].values():
            if snapshot.get('sceneId') != scene.identity:
                raise ValueError('Saved environment differs from its source scene')
            embedded = snapshot.get('protectionContract')
            if embedded is not None:
                protection, guard = restore_protection_contract(scene, embedded)
                # Additional supplied masks need independently source-rebound
                # annotations. Never silently omit an existing exclusion mask.
                if protection.annotations or guard.require_critical_coverage or guard.critical_max_displacement_mm != 0 or protection.clearance_mm != 2 or protection.required != ('brainstem',):
                    raise ValueError('Explicit saved protection requires a separately rebound comparison contract')
    return adapters, scenes, changed, acquisition(manifests[0])


def _observation_hashes(observation):
    return {key: array_digest(value) for key, value in observation.items()}


def _compact_outcome(outcome):
    return {k: outcome.get(k) for k in ('status', 'reasonCode', 'reason', 'completedFraction', 'stateId')}


def compare_anatomy(before, after, *, changed_layers=('vessels',), prefix=(), decisions=1, check=lambda: None, progress=lambda *_: None):
    """Run one fixed public prefix and a frozen existing policy on both estimates.

    Research coverage is explicitly relaxed because available patients do not
    supply complete functional critical anatomy. No evidence is marked reviewed.
    Prefixes are evaluator-defined task setup, disclosed and fixed before runs.
    """
    from .surgical_rl_env import SurgicalRLEnv, RLConfig, LOCAL_CHANNELS
    from .surgical_rl_rewards import RewardConfig
    from .surgical_rl_protection import ProtectionSet, cell_radius

    if type(decisions) is not int or not 1 <= decisions <= 4:
        raise ValueError('Use one to four frozen-policy decisions')
    if not isinstance(changed_layers, (tuple, list)) or not changed_layers or len(set(changed_layers)) != len(changed_layers):
        raise ValueError('Declare distinct changed anatomy roles')
    if not isinstance(prefix, (tuple, list)) or len(prefix) > 64:
        raise ValueError('Fixed setup prefix exceeds 64 public actions')
    adapters, scenes, changed, acquired = _bound_inputs(before, after, changed_layers)
    config = RLConfig(max_steps=80, max_seconds=120., max_healthy_removed_mm3=100.,
                      require_critical_coverage=False, local_spacing_mm=1.)
    reward_config = RewardConfig()
    variants, observations = [], []
    for variant, scene in enumerate(scenes):
        check(); progress(.1 + .4 * variant, 'Replaying fixed task on ' + ('baseline' if variant == 0 else 'refined') + ' source anatomy')
        env = SurgicalRLEnv(scene, protection=ProtectionSet(scene), config=config, reward_config=reward_config)
        observation, _ = env.reset(seed=17)
        vessel_ids = np.flatnonzero(np.isin(scene.labels, [r['id'] for r in scene.regions if r['material'] == 'vessel']))
        vessel_tree = cKDTree(scene.points(vessel_ids)) if len(vessel_ids) else None
        def clearance():
            slot = env.engine.state['slots']['left']
            return None if vessel_tree is None else float(vessel_tree.query(slot['position'])[0] - cell_radius(scene.affine) - env.engine.parameters(slot)['radiusMm'])
        prefix_outcomes = []
        for action in prefix:
            check()
            observation, _, terminated, truncated, info = env.step_action(action)
            prefix_outcomes.append(_compact_outcome(info['outcome']))
            if terminated or truncated:
                raise ValueError('Fixed setup prefix terminates the episode; select a bounded task')
        initial_observation = {k: v.copy() for k, v in observation.items()}
        observations.append(initial_observation)
        probe = {'observationHashes': _observation_hashes(observation), 'metrics': env.measurements.copy(),
                 'tipVesselClearanceLowerBoundMm': clearance(), 'vesselEncounters': len(env.engine.state['vesselEncounters']),
                 'contactCells': len(env.engine.state['contact']), 'stateId': env.engine.state_id}
        policy_rows = []
        for _ in range(decisions):
            check()
            # This is the complete policy boundary. Evaluator geometry below is
            # never supplied to candidates() or observed_greedy().
            features, actions = candidates(observation)
            selected, scores = observed_greedy(features)
            feasibility = []
            for action_id, action in zip(ACTION_IDS, actions):
                check()
                candidate_engine = env.engine.clone()
                outcome = candidate_engine.step(action)
                feasibility.append({'id': action_id, **_compact_outcome(outcome),
                    'vesselEncounters': len(candidate_engine.state['vesselEncounters']),
                    'targetRemovedMm3': candidate_engine.metrics()['targetRemovedMm3']})
            before_hashes = _observation_hashes(observation)
            observation, reward, terminated, truncated, info = env.step_action(actions[selected])
            policy_rows.append({'selected': ACTION_IDS[selected], 'action': actions[selected], 'scores': scores,
                'observationHashes': before_hashes, 'candidateOutcomes': feasibility,
                'outcome': _compact_outcome(info['outcome']), 'reward': reward, 'stateId': env.engine.state_id})
            if terminated or truncated or selected == 8:
                break
        target_ids = [r['id'] for r in scene.regions if r.get('target')]
        variants.append({'caseHash': (before, after)[variant].case.semantic_hash, 'sceneId': scene.identity,
            'sourceSnapshot': adapters[variant]['sourceCollectionHash'],
            'anatomy': {name: {'occupiedCells': int(np.count_nonzero(scene.labels == r['id'])), 'material': r['material'], 'target': bool(r.get('target'))}
                        for r in scene.regions for name in [r['name']]},
            'sourceLayers': {name: {k: row.get(k) for k in ('layerId', 'sampledMaskHash', 'provenance', 'reviewStatus', 'revision', 'originalData', 'edits', 'uncertainty', 'scoreMeaning', 'probability', 'sampling')}
                             for name, row in adapters[variant]['sourceLayers'].items()},
            'targetVolumeMm3': float(np.count_nonzero(np.isin(scene.labels, target_ids)) * scene.voxel_mm3),
            'prefixOutcomes': prefix_outcomes, 'probe': probe, 'policy': policy_rows,
            'final': {'stateId': env.engine.state_id, 'metrics': env.measurements.copy(), 'reward': env.total_reward,
                      'tipVesselClearanceLowerBoundMm': clearance(),
                      'vesselEncounters': len(env.engine.state['vesselEncounters']), 'contactCells': len(env.engine.state['contact']),
                      'guard': env.engine.guard_report()}})
        env.close()
    local_delta = observations[0]['local'] != observations[1]['local']
    return {'schema': SCHEMA, 'patient': adapters[0]['sourcePatient'],
        'controls': {'acquiredSourceHash': semantic_digest(acquired), 'taskMappingHash': semantic_digest(scenes[0].regions),
            'physicsHash': semantic_digest(scenes[0].properties), 'rlConfig': asdict(config), 'rewardConfig': asdict(reward_config),
            'policy': 'existing observed_greedy_local', 'policySourceSha256': hashlib.sha256(Path(__file__).with_name('surgical_observed_policy.py').read_bytes()).hexdigest(),
            'policyInformation': 'published partial local observation only; previews and source arrays evaluator-only',
            'training': False, 'seed': 17, 'declaredChangedLayers': list(changed_layers), 'prefix': copy.deepcopy(prefix),
            'prefixHash': semantic_digest(prefix), 'decisions': decisions},
        'maskChanges': changed, 'before': variants[0], 'after': variants[1],
        'delta': {'matchedPrefixObservationChangedValues': {key: int(np.count_nonzero(observations[0][key] != observations[1][key])) for key in observations[0]},
                  'matchedPrefixLocalChannels': {name: int(np.count_nonzero(local_delta[:, i])) for i, name in enumerate(LOCAL_CHANNELS)},
                  'policyActionsChanged': [r['selected'] for r in variants[0]['policy']] != [r['selected'] for r in variants[1]['policy']],
                  'targetVolumeMm3': variants[1]['targetVolumeMm3'] - variants[0]['targetVolumeMm3']},
        'limitations': ['Software integration verification; no evidence of surgical safety.',
            'Source estimates and registrations retain their original review and uncertainty status.',
            'Missing functional anatomy remains unknown; explicit relaxed research coverage is identical in both runs.',
            'Tip clearance uses static source-estimate vessel cells and their circumspheres; it ignores simulated deformation and is neither shaft clearance nor measured vessel wall distance.',
            'Candidate outcomes use the actual shared engine; evaluator previews do not enter policy input.',
            'Display-only smoothing is excluded from simulation masks, physics, observations and rewards.']}
