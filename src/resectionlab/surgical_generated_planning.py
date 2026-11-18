"""Source-bound contact projection for the generated post-access controls."""
from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import math

import numpy as np

from .core import semantic_digest
from .geometry import capsule_voxel_indices_batch
from .surgical_procedure_policy import _validated
from .surgical_sequential_task import training_admission
from .surgical_tools import TOOLS
from .surgical_treatment_intent import decide_treatment_intent

SCHEMA = 'resectionlab.generated-source-contact-projection.v1'
NEURAL = 'neuralRemovedMm3'
EPSILON = 1e-9


class SourceContactProjection:
    """Compare finite single-radius programs using nominal source contact only."""

    def __init__(self, task):
        training_admission(task)
        self.task = task
        self.engine = engine = task.initial_engine()
        self.policy = policy = _validated(engine.scene)
        self.intent = policy['treatmentIntent']
        raw = policy['record']
        self.binding = dict(schema=SCHEMA, sourceSceneId=engine.scene.identity,
            policyHash=raw['policyHash'], sourceBinding=raw['sourceBinding'],
            inventoryHash=semantic_digest([asdict(skill) for skill in task.skills]),
            requiredHarmMetrics=[NEURAL, *(f'structure:{name}:removedMm3'
                                           for name in sorted(policy['structureCells']))],
            preservationPreference=raw['preservationPreference'],
            enumeration='Available work subsets at the current tip radius or after one declared exchange; reserve checked STOP.',
            evidence='Nominal active-tip capsule and source removal membership. Full-tool motion, physical effects and STOP require checked execution.')
        self.members = {NEURAL: policy['neural'], **{
            f'structure:{name}:removedMm3': cells for name, cells in policy['structureCells'].items()}}
        self.limits = {NEURAL: raw['maximumNeuralRemovalMm3'], **{
            f'structure:{name}:removedMm3': cap for name, cap in raw['maximumStructureRemovalMm3'].items()}}
        self.removable_labels = {row['id'] for row in engine.scene.regions
            if row['material'] in TOOLS['cusa']['materials']
            and raw['tissuePermissions'][row['material']]['remove']}
        self.slots = {action['slot'] for skill in task.skills for action in skill.actions}
        if len(self.slots) != 1:
            raise ValueError('Generated contact projection requires a single declared work slot')
        self.slot = next(iter(self.slots))
        self.stop = next(index for index, skill in enumerate(task.skills) if skill.kind == 'stop')

    def project(self, observed, mask):
        skills = self.task.skills
        scene = self.engine.scene
        state = observed['native_state']
        slot = state['slots'][self.slot]
        params = slot['parameters']
        protected = self.engine._protected_ids
        labels = scene.labels
        removed = set(state['removed'])
        remaining = self.intent['original'] - removed
        eligible = self.intent['eligible'] - protected - removed
        voxel = scene.voxel_mm3
        limits = observed['remaining']
        current = self.engine.parameters(slot)['radiusMm']
        preparation = [(s.radius, i) for i, s in enumerate(skills) if mask[i] and s.kind == 'prepare']
        work = [i for i, s in enumerate(skills) if mask[i] and s.kind == 'work']
        coverage = self.policy['intracranial'] <= self.policy['observed'] and not self.engine.protection.missing
        posed = any(a['slot'] == self.slot and 'pose' in a for a in observed['accepted_history'])
        parked = (not (slot['inserted'] or slot['active'] or slot['grip'] or self.slot in state['holds'])
            and (not posed or slot['position'][2] == observed['exterior_withdrawal_plane_ras_mm']))
        projection_complete = bool(coverage and parked and slot['instrument'] == 'cusa'
            and set(params) <= {'radiusMm'} and math.isfinite(current) and current > 0.)
        stop_fits = bool(mask[self.stop] and limits['skills'] >= 1 and limits['attempts'] > 1
                         and limits['invalid'] > 0 and limits['seconds'] > EPSILON)
        setups = [(current, None), *sorted(preparation)] if projection_complete else []
        queries = [(radius, index) for radius, _ in setups for index in work]
        contacts = capsule_voxel_indices_batch(self.engine.grid,
            [(*skills[i].route, radius) for radius, i in queries])

        def harms(cells):
            return {metric: len(cells & self.members[metric]) * voxel
                    for metric in self.binding['requiredHarmMetrics']}

        def harm_blockers(values):
            return {metric for metric, cap in self.limits.items() if values[metric] > cap + EPSILON}

        # Capsule contact describes the modeled source tip stroke. Empty source
        # labels can still carry protection, so check the whole contact set.
        estimates = {}
        for (radius, index), ijk in zip(queries, contacts, strict=True):
            cells = set(map(int, np.ravel_multi_index(ijk.T, labels.shape)))
            occupied = {cell for cell in cells - removed if labels.flat[cell] != 0}
            effect = {cell for cell in occupied if int(labels.flat[cell]) in self.removable_labels}
            blockers = set()
            if cells & protected:
                blockers.add('protected_source_contact')
            if occupied - effect:
                blockers.add('source_material_incompatible_or_forbidden')
            unknown = not cells <= self.policy['observed']
            estimates[radius, index] = (effect, blockers, unknown)
            projection_complete = projection_complete and not unknown

        alternatives = []
        programs = {}
        for radius, preparation_index in setups:
            prefix = () if preparation_index is None else (preparation_index,)
            candidates = [index for index in work if estimates[radius, index][0] & eligible]
            for length in range(1, len(candidates) + 1):
                for group in combinations(candidates, length):
                    program = (*prefix, *group)
                    cells = set().union(*(estimates[radius, i][0] for i in group))
                    blockers = set().union(*(estimates[radius, i][1] for i in group))
                    values = harms(removed | cells)
                    blockers.update(harm_blockers(values))
                    steps = [action for index in program for action in skills[index].actions]
                    seconds = sum(action.get('duration', 1.) for action in steps)
                    if len(program) + 1 > limits['skills']:
                        blockers.add('remaining_skills_with_stop')
                    if len(steps) + 1 >= limits['attempts']:
                        blockers.add('remaining_attempts_with_stop')
                    if seconds >= limits['seconds'] - EPSILON:
                        blockers.add('remaining_modeled_time')
                    if state['elapsedSeconds'] + seconds > self.policy['record']['maximumModeledElapsedSeconds'] + EPSILON:
                        blockers.add('procedure_modeled_time')
                    unknown = any(estimates[radius, i][2] for i in group)
                    key = '+'.join(skills[i].id for i in program)
                    alternatives.append(dict(id=key, treatmentMm3=len(eligible & cells) * voxel,
                        harms=values, admissible=None if unknown else stop_fits and not blockers,
                        blockingBoundaries=[], projectedExclusions=sorted(blockers),
                        projectedPhysicalRequests=len(steps), projectedModeledSeconds=seconds))
                    programs[key] = program

        boundaries = []
        explained = set()
        for name, (source, cells) in self.intent['boundaries'].items():
            overlap = cells & remaining
            if overlap:
                explained.update(overlap)
                boundaries.append(dict(id=name, label=source['label'], structure=source['structure'],
                    provenance=source['provenance'], evidenceSha256=source['evidenceSha256'],
                    residualMm3=len(overlap) * voxel, eligibleResidualMm3=len(overlap & eligible) * voxel))
        residual = dict(residualMm3=len(remaining) * voxel, eligibleResidualMm3=len(eligible) * voxel,
            unexplainedResidualMm3=len(remaining - explained) * voxel, boundaries=boundaries)
        prefix_harms = harms(removed)
        incumbent = dict(id='checked-stop', treatmentMm3=0., harms=prefix_harms,
                         admissible=stop_fits and parked and not harm_blockers(prefix_harms)
                         and not (removed & protected), blockingBoundaries=[])
        decision = decide_treatment_intent(alternatives, incumbent=incumbent,
            required_harms=self.binding['requiredHarmMetrics'],
            preference_harms=self.binding['preservationPreference']['metrics'],
            residual=residual, coverage_complete=coverage, projection_complete=projection_complete)
        index = programs[decision['selectedId']][0] if decision['status'] == 'continue' else self.stop
        return index, dict(decision=decision, selectedSkillId=skills[index].id,
            availableSkillIds=[s.id for i, s in enumerate(skills) if mask[i]],
            alternatives=alternatives, incumbent=incumbent, residual=residual, coverageComplete=coverage,
            projectionComplete=projection_complete, capsuleQueries=len(queries), projectionBinding=self.binding)
