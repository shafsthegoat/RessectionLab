"""Bounded native search across opening, preparation, work and checked exit.

Proposal labels confer no permission. All actions use the same source-frozen
joint policy, meter and native stage guards. This module supplies search, not
patient qualification or a clinical model of surgical access.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import time
import weakref

import numpy as np

from .core import semantic_digest
from .surgical_joint_cranial_skills import (
    _readiness, _require, checked_joint_actions, checked_joint_cusa_motion,
    joint_cranial_ledger)
from .surgical_transition_meter import TransitionBudgetExceeded


SCHEMA = 'resectionlab.joint-cranial-search.v2'
_GOALS = weakref.WeakValueDictionary()
_GOAL_FACTS = {}
_INTERRUPTIONS = (TransitionBudgetExceeded, TimeoutError, InterruptedError,
                  KeyboardInterrupt, asyncio.CancelledError)
_MILLING_STAGING_LIMITS = frozenset(('JOINT_MILLING_INITIAL_STAGING_UNQUALIFIED',
    'JOINT_MILLING_INITIAL_STAGING_NOT_EXTERIOR',
    'JOINT_MILLING_POSED_ATTITUDE_STAGING_UNQUALIFIED',
    'JOINT_MILLING_INITIAL_STOCK_STAGING_UNQUALIFIED',
    'JOINT_MILLING_DECLARED_GLOBAL_START_POSE_DIFFERS'))


def assess_original_target_progress(engine, *, original_target_cells=None, credit_labels=None,
        minimum_target_fraction=.5, minimum_zones=3, minimum_zone_fraction_of_original=.05):
    """Pure shared progress assessment over the full immutable source target.

    A protected or later-owned overlap stays in the denominator. Treatment
    credit requires original membership and source-owned target. This assesses
    progress only; native stage, checked exit and harm constraints remain
    separate requirements for task completion.
    """
    from .surgical_scene import original_target_memberships
    scene = engine.scene
    scene.require_cache_source()
    if (type(minimum_target_fraction) not in (int, float) or not .5 <= minimum_target_fraction <= 1.
            or type(minimum_zones) is not int or not 3 <= minimum_zones <= 8
            or type(minimum_zone_fraction_of_original) not in (int, float)
            or not .05 <= minimum_zone_fraction_of_original <= 1.):
        raise ValueError('Original target progress requires fraction .5..1, zones 3..8 and zone fraction .05..1')
    memberships = original_target_memberships(scene)
    basis = 'original_annotation'
    if original_target_cells is None:
        if memberships is not None:
            target = frozenset(int(c) for cells in memberships.values() for c in cells)
        elif (scene.source.get('evidenceClass') == 'generated-engineering' or
                scene.source.get('kind') in ('generated-control', 'engineering-control')):
            labels = [r['id'] for r in scene.regions if r.get('target')]
            target = frozenset(map(int, np.flatnonzero(np.isin(scene.labels, labels))))
            basis = 'generated_owned_target_labels_fallback'
        else:
            return {'targetBasis': 'unavailable', 'originalTargetCells': None,
                'residualOriginalTargetCells': None, 'originalTargetTreatmentFraction': None,
                'completedProgressRequirement': False, 'reasons': ['ORIGINAL_TARGET_MEMBERSHIPS_UNAVAILABLE']}
    else:
        target = frozenset(map(int, original_target_cells))
        if memberships is None:
            basis = 'explicit_frozen_source_target'
    if not target or any(c < 0 or c >= scene.labels.size for c in target):
        raise ValueError('ORIGINAL_TARGET_SOURCE_MEMBERSHIPS_REQUIRED')
    if credit_labels is None:
        credit_labels = {r['id'] for r in scene.regions if r.get('target') and
            r['material']=='tumor' and not r.get('protected') and not r.get('critical')}
    credit = {c for c in target if int(scene.labels.flat[c]) in credit_labels}
    removed = set(engine.state['removed']) & target
    credited = removed & credit
    cells = sorted(target); points = scene.points(cells)
    codes = ((points >= points.mean(axis=0)).astype(np.int8) @ np.asarray([1, 2, 4])).tolist()
    zones = tuple(frozenset(c for c, code in zip(cells, codes) if code==i) for i in range(8))
    zone_removed = [len(removed & zone) for zone in zones]
    zone_credit = [len(credited & zone) for zone in zones]
    total = len(target)
    reached = sum(n/total+1e-12 >= minimum_zone_fraction_of_original for n in zone_credit)
    reasons = []
    if len(credited)/total+1e-12 < minimum_target_fraction:
        reasons.append('ORIGINAL_TARGET_FRACTION_BELOW_GOAL')
    if reached < minimum_zones:
        reasons.append('ORIGINAL_SPATIAL_PROGRESS_BELOW_GOAL')
    return {'targetBasis': basis, 'originalTargetCells': total,
        'removedOriginalTargetCells': len(removed), 'ownedOriginalTargetCells': len(credit),
        'creditedOriginalTargetCells': len(credited), 'residualOriginalTargetCells': total-len(removed),
        'originalTargetFractionRemoved': len(removed)/total,
        'originalTargetRemovedMm3': len(removed)*scene.voxel_mm3,
        'originalTargetTreatmentFraction': len(credited)/total,
        'originalTargetTreatmentMm3': len(credited)*scene.voxel_mm3,
        'zoneRemovedCells': zone_removed, 'zoneResidualCells': [len(z)-n for z,n in zip(zones,zone_removed)],
        'zoneTreatmentCreditCells': zone_credit, 'zonesReachingRequiredOriginalFraction': reached,
        'completedProgressRequirement': not reasons, 'reasons': reasons}


@dataclass(frozen=True)
class JointResectionGoal:
    source_state_id: str
    limits_hash: str
    minimum_target_fraction: float
    minimum_zones: int
    minimum_zone_fraction_of_original: float
    target_basis: str
    zone_counts: tuple
    content_hash: str
    _limits: object = field(repr=False, compare=False)
    _target: frozenset = field(repr=False, compare=False)
    _zones: tuple = field(repr=False, compare=False)
    _credit: frozenset = field(repr=False, compare=False)

    def declaration(self):
        return {'sourceStateId': self.source_state_id, 'limitsHash': self.limits_hash,
            'minimumTargetFraction': self.minimum_target_fraction,
            'minimumZones': self.minimum_zones,
            'minimumZoneFractionOfOriginal': self.minimum_zone_fraction_of_original,
            'targetBasis': self.target_basis, 'originalTargetCells': len(self._target),
            'ownedOriginalTargetCells': len(self._credit),
            'treatmentCreditRule': 'Removed source cells must belong to both the original annotation and a source-owned target label. The denominator remains the full original annotation.',
            'zoneDefinition': 'Eight RAS octants about the original target centroid; equality belongs to the positive half.',
            'originalZoneCells': list(self.zone_counts),
            'spatialPartitionHash': semantic_digest([sorted(z) for z in self._zones])}


def freeze_joint_resection_goal(engine, *, limits, minimum_target_fraction=.5,
        minimum_zones=3, minimum_zone_fraction_of_original=.05):
    """Freeze the full original target and spatial goal before native actions.

    The minimum meaningful task is half the original target, with at least
    three frozen spatial zones each contributing 5% of the original target.
    A small or narrow target can make this spatial task unachievable; it does
    not license choosing new zones after seeing which cells were removed.
    """
    _require(engine, limits, limits._meter)
    if engine.actions or engine.history or engine.state_id != limits.source_state_id or limits._meter.calls:
        raise ValueError('JOINT_GOAL_FREEZE_BEFORE_ACTIONS_REQUIRED')
    if (type(minimum_target_fraction) not in (int, float) or
            not .5 <= minimum_target_fraction <= 1. or
            type(minimum_zones) is not int or not 3 <= minimum_zones <= 8 or
            type(minimum_zone_fraction_of_original) not in (int, float) or
            not .05 <= minimum_zone_fraction_of_original <= 1.):
        raise ValueError('Joint task needs target fraction .5..1, zones 3..8 and zone fraction .05..1')
    target = frozenset(limits._contract.original_target_cells)
    if not target:
        raise ValueError('JOINT_GOAL_ORIGINAL_TARGET_REQUIRED')
    cells = sorted(target)
    points = engine.scene.points(cells)
    centroid = points.mean(axis=0)
    codes = ((points >= centroid).astype(np.int8) @ np.asarray([1, 2, 4])).tolist()
    zones = tuple(frozenset(c for c, code in zip(cells, codes) if code == i) for i in range(8))
    credit = frozenset(c for c in cells if int(engine.scene.labels.flat[c]) in limits._target_labels)
    values = dict(source_state_id=engine.state_id, limits_hash=limits.content_hash,
        minimum_target_fraction=float(minimum_target_fraction), minimum_zones=minimum_zones,
        minimum_zone_fraction_of_original=float(minimum_zone_fraction_of_original),
        target_basis='original_annotation' if engine.scene._original_target_proof is not None
            else 'owned_target_labels_fallback',
        zone_counts=tuple(map(len, zones)), _limits=limits, _target=target, _zones=zones,
        _credit=credit)
    provisional = JointResectionGoal(**values, content_hash='')
    goal = JointResectionGoal(**values, content_hash=semantic_digest(provisional.declaration()))
    _GOALS[id(goal)] = goal
    # Retain exact immutable set identities as well as scalar values. Replacing
    # a field and recomputing a public digest cannot change a registered goal.
    _GOAL_FACTS[id(goal)] = tuple((name, getattr(goal, name)) for name in goal.__dataclass_fields__)
    weakref.finalize(goal, _GOAL_FACTS.pop, id(goal), None)
    return goal


def _require_goal(goal, limits):
    if not isinstance(goal, JointResectionGoal) or _GOALS.get(id(goal)) is not goal:
        raise ValueError('JOINT_REGISTERED_GOAL_REQUIRED')
    if goal._limits is not limits or goal.limits_hash != limits.content_hash:
        raise ValueError('JOINT_GOAL_SOURCE_POLICY_CHANGED')
    if any((getattr(goal, k) is not v if k.startswith('_') or isinstance(v, tuple)
            else getattr(goal, k) != v) for k, v in _GOAL_FACTS[id(goal)]):
        raise ValueError('JOINT_FROZEN_GOAL_CHANGED')


def joint_goal_assessment(engine, *, goal, limits, exit_receipt):
    """Assess actual unique removals against the prospectively frozen target."""
    _require(engine, limits, limits._meter)
    _require_goal(goal, limits)
    progress = assess_original_target_progress(engine, original_target_cells=goal._target,
        credit_labels=limits._target_labels, minimum_target_fraction=goal.minimum_target_fraction,
        minimum_zones=goal.minimum_zones,
        minimum_zone_fraction_of_original=goal.minimum_zone_fraction_of_original)
    readiness = _readiness(engine) or {}
    ledger = joint_cranial_ledger(engine, limits)
    reasons = list(progress['reasons'])
    if (exit_receipt.get('status') != 'accepted' or
            exit_receipt.get('finalStateId') != engine.state_id or
            not readiness.get('wholeExitComplete') or not readiness.get('workEntrySupported')):
        reasons.append('NATIVE_WORK_AND_WHOLE_EXIT_REQUIRED')
    if not ledger['withinLimits']:
        reasons.extend(ledger['refusalReasons'])
    return {**progress, 'goalHash': goal.content_hash, 'targetBasis': goal.target_basis,
        'completedFrozenTask': not reasons, 'reasons': reasons,
        'admittedPatientTask': False, 'caseQualificationRequired': True}


@dataclass
class JointSequenceNode:
    work: object
    exited: object
    path: tuple
    ledger: dict
    goal_assessment: dict
    sequence_outcome: dict


@dataclass
class JointSearchResult:
    selected: JointSequenceNode | None
    alternatives: tuple
    progress: tuple
    stopped_prefixes: tuple
    attempts: tuple
    evidence: dict


def _key(node):
    return (-node.goal_assessment['originalTargetTreatmentMm3'],
        *_harm_key(node),
        tuple(row['proposalId'] for row in node.path))


def _harm_key(node):
    """Neural source loss and bone burden cannot substitute for one another."""
    dimensions = node.sequence_outcome['dimensions']
    def value(name, fallback=float('inf')):
        v = dimensions.get(name, {}).get('cumulative')
        return fallback if v is None else v
    structures = {f'structure:{name}:{effect}': q.get('cumulative')
        for field,effect in (('structureRemovedMm3','removedMm3'),('structureCutSourceMm3','cutSourceMm3'))
        for name,q in dimensions.get(field, {}).items()}
    from .surgical_consequences import EXPOSURE_METRICS
    return (value('neuralRemovedMm3'), value('neuralCutSourceMm3'),
        *[float('inf') if v is None else v for _,v in sorted(structures.items())],
        value('protectedRemovedMm3'),
        *[value(name) for name in EXPOSURE_METRICS],
        value('nonbonePreparationRemovedMm3',node.ledger['preparationNonboneRemovedMm3']),
        node.ledger['boneRemovedMm3'], node.ledger['modeledElapsedSeconds'])


def _retain_beam(children, beam_width):
    """Reserve treatment, lower-harm and delayed-preparation continuations."""
    frontier = [n for n in children if not any(_dominates(o, n) for o in children)]
    ordered = sorted(frontier, key=_key)
    # Reserve different neural/exposure vectors before a second lower-burden
    # representative of the same vector. Otherwise bone/time can erase an
    # unresolved tradeoff before the canonical final comparison sees it.
    representatives, duplicates, seen = [], [], set()
    for node in ordered:
        harm = _harm_key(node)
        signature = (node.goal_assessment['originalTargetTreatmentMm3'], harm[:-3])
        if signature in seen:
            duplicates.append(node)
        else:
            seen.add(signature); representatives.append(node)
    ordered = representatives + duplicates
    retained = ordered[:1]
    lower_harm = min(frontier, key=lambda n: (*_harm_key(n), _key(n)))
    if not any(n is lower_harm for n in retained):
        retained.append(lower_harm)
    preparation = next((n for n in sorted(children, key=_key)
                        if n.path[-1]['preparationOnly']), None)
    if preparation is not None and len(retained) < beam_width and not any(
            n is preparation for n in retained):
        retained.append(preparation)
    for node in ordered:
        if len(retained) >= beam_width:
            break
        if not any(node is n for n in retained):
            retained.append(node)
    return retained


def _dominates(a, b):
    am, av = _comparison_vector(a)
    bm, bv = _comparison_vector(b)
    # Missing evidence never becomes zero. Different availability is explicitly
    # incomparable rather than an apparent reduction in modeled consequences.
    if am != bm:
        return False
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def _comparison_vector(node):
    names = ['negativeOriginalTargetMm3', 'boneRemovedMm3',
        'preparationNonboneRemovedMm3', 'workNonTargetRemovedMm3', 'modeledElapsedSeconds']
    values = [-node.goal_assessment['originalTargetTreatmentMm3'],
        *[node.ledger[k] for k in names[1:]]]
    dimensions = node.sequence_outcome['dimensions']
    anatomical = {'neuralRemovedMm3': dimensions.get('neuralRemovedMm3',
        {'status': 'unavailable', 'cumulative': None}),
        'neuralCutSourceMm3': dimensions.get('neuralCutSourceMm3',
        {'status': 'unavailable', 'cumulative': None})}
    anatomical.update({'structure:' + name + ':removedMm3': d for name, d in
        dimensions.get('structureRemovedMm3', {}).items()})
    anatomical.update({'structure:' + name + ':cutSourceMm3': d for name, d in
        dimensions.get('structureCutSourceMm3', {}).items()})
    for name, d in sorted(anatomical.items()):
        value = d.get('cumulative')
        names.append((name, d['status'], d.get('aggregation', 'cumulative')))
        if value is not None:
            if type(value) not in (float, int) or not np.isfinite(value):
                raise ValueError('Joint comparison requires finite source consequences')
            values.append(value)
    for name in ('toolExchanges', 'brainStrainSquaredMm3S', 'brainStressSquaredPa2Mm3S',
            'thermalPeakC', 'thermalExposureMm3KS', 'bloodGeneratedMm3',
            'motorEncounterMm3', 'languageEncounterMm3',
            'supplyDeficitSeconds', 'drainageDeficitSeconds'):
        d = dimensions.get(name, {'status': 'unavailable', 'cumulative': None})
        value = d.get('cumulative')
        names.append((name, d['status'], d.get('aggregation', 'cumulative')))
        if value is not None:
            if type(value) not in (float, int) or not np.isfinite(value):
                raise ValueError('Joint comparison requires finite supported outcome dimensions')
            values.append(value)
    return tuple(names), tuple(values)


def _compare_checked_nodes(nodes, stop, limits):
    """Drive the existing consequence comparator with actual exited engines."""
    from .surgical_consequences import compare_samples, consequence_vector
    from .surgical_procedure_policy import procedure_comparison_policy
    samples, by_id = [], {}
    for index, node in enumerate([stop, *nodes]):
        branch = 'STOP' if index == 0 else 'candidate-' + str(index)
        by_id[branch] = node
        vector = consequence_vector(node.exited, limits._source_engine)
        # Treatment credit stays within the full original annotation.
        d = vector['dimensions']['targetRemovedMm3']
        d['cumulative'] = d['incremental'] = node.goal_assessment['originalTargetTreatmentMm3']
        samples.append({'branchId': branch, 'scenarioId': 'nominal', 'vector': vector})
    if not nodes:
        return stop, (), {'selectedBranchId': 'STOP', 'pairedScenarioParetoFront': [],
            'reason': 'no_qualified_continuation_stop_with_residual', 'stopExecutable': True,
            'scenarios': ['nominal']}
    policy = procedure_comparison_policy(stop.exited.scene)
    # The source's declared evaluators determine required supported exposures.
    # If a continuation loses one, comparison refuses it instead of assigning
    # zero or discarding that dimension from the entire frontier.
    source_dimensions = samples[0]['vector']['dimensions']
    for metric in ('brainStrainSquaredMm3S', 'brainStressSquaredPa2Mm3S',
            'thermalPeakC', 'thermalExposureMm3KS', 'bloodGeneratedMm3',
            'supplyDeficitSeconds', 'drainageDeficitSeconds'):
        if source_dimensions[metric]['status'] != 'unavailable':
            policy['objectives'].append(metric)
            policy['requiredMetrics'].append(metric)
    decision = compare_samples(samples, policy,
        stop_branch='STOP', scenario_ids=['nominal'], branch_ids=list(by_id),
        selection_rule='max_treatment_within_harm_bounds')
    selected = by_id.get(decision['selectedBranchId'])
    return selected, tuple(by_id[b] for b in decision['pairedScenarioParetoFront']), decision


def _physical_key(engine):
    # Time/travel and passive observations alone do not create useful preparation.
    return semantic_digest({k: v for k, v in engine.state.items()
        if k not in ('elapsedSeconds', 'travelMm', 'observations')})


def _failure_category(reason):
    text = str(reason)
    if text.startswith(('PROCEDURE_HARM_', 'PROCEDURE_TISSUE_', 'PROCEDURE_CRITICAL_')):
        return 'DECLARED_ANATOMICAL_HARM_CONSTRAINT'
    if text in _MILLING_STAGING_LIMITS or any(s in text for s in ('CONTACT_CONTINUATION', 'ENDPOINT', 'NO_PHYSICAL_PROGRESS',
            'MILLING_CONTACT_CONTROLLER_DEFICIT', 'MILLING_NO_ACTUAL_BONE_OR_DEBRIS_PROGRESS',
            'MILLING_BATCH_BUDGET', 'POSED_ATTITUDE_STAGING_UNQUALIFIED')):
        return 'CANDIDATE_CONTROLLER_LIMIT'
    if any(s in text for s in ('Budget', 'BUDGET', 'Timeout', 'Interrupted', 'Cancelled')):
        return 'COMPUTATION_BUDGET_OR_CANCELLATION'
    if any(s in text.upper() for s in ('UNSUPPORTED', 'EVIDENCE', 'COVERAGE_REQUIRED', 'UNKNOWN')):
        return 'MISSING_ANATOMY_OR_UNSUPPORTED_MODEL'
    if 'CONTINUATION' in text:
        return 'CANDIDATE_CONTROLLER_LIMIT'
    return 'CHECKED_NATIVE_CONSTRAINT'


def _global_stop(reason):
    # A finite contact controller can stop one branch without exhausting the
    # shared computation. Other candidates may still be executable.
    return any(s in str(reason) for s in ('TransitionBudget', 'TransitionReserve',
        'PUBLIC_CALL_BUDGET', 'WALL_BUDGET',
        'TimeoutError', 'InterruptedError', 'KeyboardInterrupt', 'CancelledError'))


def _cancelled(reason):
    return any(s in str(reason) for s in ('TimeoutError', 'InterruptedError',
        'KeyboardInterrupt', 'CancelledError'))


def _proposal(value, state_id):
    if type(value) is not dict or not {'id', 'sourceStateId', 'kind'} <= set(value):
        raise ValueError('Joint proposal requires id, sourceStateId and kind')
    p = copy.deepcopy(value)
    if (type(p['id']) is not str or not p['id'] or p['sourceStateId'] != state_id or
            p['kind'] not in ('opening', 'milling_opening', 'preparation', 'configuration', 'work', 'cusa')):
        raise ValueError('Joint proposal must name a supported kind and the current native source state')
    if p['kind'] == 'milling_opening':
        required = {'approach', 'drill_seconds', 'suction_seconds', 'staging_seconds',
            'slot', 'max_batches', 'max_contacts', 'max_examined_cells',
            'max_contact_probes', 'point_scheme'}
        spec = p.get('milling')
        if ('actions' in p or 'motion' in p or type(spec) is not dict or
                not required <= set(spec) <= required | {'staging_plan', 'cell_schedule', 'contact_pose_family'} or
                type(spec['approach']) is not dict or
                ('staging_plan' in spec and type(spec['staging_plan']) is not dict) or
                ('cell_schedule' in spec and (type(spec['cell_schedule']) is not str or
                    spec['cell_schedule'] not in ('surface_first', 'axial_diverse_current_a'))) or
                ('contact_pose_family' in spec and (type(spec['contact_pose_family']) is not str or
                    spec['contact_pose_family'] not in ('leading_touch', 'cavity_carried_rays')))):
            raise ValueError('Joint milling proposal needs one explicit bounded milling spec; policy, meter and callbacks remain search-owned')
    elif p['kind'] == 'cusa':
        if 'actions' in p or type(p.get('motion')) is not dict:
            raise ValueError('Joint CUSA proposal needs one native motion')
    elif ('motion' in p or not isinstance(p.get('actions'), (list, tuple)) or
            not p['actions'] or any(type(a) is not dict for a in p['actions'])):
        raise ValueError('Joint proposal needs a nonempty finite native action list')
    return p


def checked_joint_sequence_search(engine, route=None, *, propose, limits, goal,
        meter=None, hand='left', outside_poses=None, horizon=3, beam_width=3,
        max_expansions=24, max_exit_candidates=6, check=lambda: None,
        phase='cranial.joint.search'):
    """Search up to five skills, retaining zero-target preparation continuations.

    ``propose(current_native_state)`` returns a finite list of source-bound
    opening/preparation/configuration/work programs or CUSA motions. Geometry
    generation is the caller's responsibility and confers no executable proof.
    Horizon one is an immediate-target ablation using the same inventory and
    joint limits, not a strongest-controller patient benchmark. Exit probes
    branch from work; they are not inserted between
    consecutive skills in the selected program.
    """
    from .surgical_planning_admission import require_planning_admission
    require_planning_admission(engine)
    from .surgical_joint_cranial_exit import find_checked_joint_exit
    if (not callable(propose) or not callable(check) or
            type(horizon) is not int or not 1 <= horizon <= 5 or
            type(beam_width) is not int or not 2 <= beam_width <= 4 or
            type(max_expansions) is not int or not 1 <= max_expansions <= 24 or
            type(max_exit_candidates) is not int or not 1 <= max_exit_candidates <= 6 or
            hand not in ('left', 'right') or type(phase) is not str or not phase):
        raise ValueError('Keep joint horizon 1..5, beam 2..4, expansions 1..24 and exit candidates 1..6')
    meter = limits._meter if meter is None else meter
    _require(engine, limits, meter)
    _require_goal(goal, limits)
    from .surgical_procedure_policy import PROPERTY as PROCEDURE_PROPERTY
    supported_procedure = PROCEDURE_PROPERTY in engine.scene.properties
    first_calls = meter.calls
    attempts, stopped, nodes = [], [], []
    frontier = [(engine, ())]
    expansions, deepest, exhausted, interrupted = 0, 0, False, False
    stop_reason = None
    initial_stop = None
    if supported_procedure:
        exited, receipt = find_checked_joint_exit(engine, route, limits=limits,
            meter=meter, hand=hand, outside_poses=outside_poses, check=check,
            max_candidates=max_exit_candidates, phase=phase+'.initial-stop')
        if receipt['status'] != 'accepted':
            raise ValueError('PROCEDURE_EXECUTABLE_INITIAL_STOP_REQUIRED: ' + str(receipt.get('reason')))
        initial_stop = JointSequenceNode(engine, exited, (), joint_cranial_ledger(exited, limits),
            joint_goal_assessment(exited, goal=goal, limits=limits, exit_receipt=receipt),
            receipt['sequenceOutcome'])

    def guard():
        check()
        if time.monotonic()-limits.frozen_at >= limits.max_wall_seconds:
            raise TimeoutError('Joint search wall bound since shared meter creation')
        if meter.calls >= min(meter.max_calls, limits.max_public_calls):
            raise TransitionBudgetExceeded('Joint search shared public-call bound')

    try:
        for depth in range(horizon):
            children = []
            for work_parent, path in frontier:
                guard()
                proposals = propose(work_parent)
                if not isinstance(proposals, (list, tuple)):
                    raise ValueError('Joint proposal generator must return a finite list or tuple')
                guard()
                local_ids = set()
                for value in proposals[:6]:
                    guard()
                    if expansions >= max_expansions:
                        exhausted, stop_reason = True, 'EXPANSION_BUDGET'
                        break
                    p = _proposal(value, work_parent.state_id)
                    if p['id'] in local_ids:
                        raise ValueError('Joint proposal identifiers must be distinct at each native state')
                    local_ids.add(p['id'])
                    expansions += 1
                    row = {'proposalId': p['id'], 'kind': p['kind'], 'depth': depth+1,
                        'sourceStateId': work_parent.state_id, 'status': 'work_refused'}
                    attempts.append(row)
                    kwargs = dict(limits=limits, meter=meter, check=check,
                        phase=f'{phase}.depth{depth+1}.proposal{expansions}.work')
                    if p['kind'] == 'milling_opening':
                        from .surgical_joint_milling_opening import checked_joint_milling_opening
                        before_calls, before_state = meter.calls, work_parent.state_id
                        try:
                            work, receipt = checked_joint_milling_opening(work_parent, **p['milling'], **kwargs)
                        except ValueError as error:
                            # These named adapter checks occur before native work.
                            # An unsupported staging candidate must not suppress
                            # another declared alternative or invent a free exit.
                            if str(error) not in _MILLING_STAGING_LIMITS:
                                raise
                            if meter.calls != before_calls or work_parent.state_id != before_state:
                                raise ValueError('JOINT_MILLING_STAGING_BOUNDARY_CHANGED_DURING_REFUSAL') from error
                            work = work_parent
                            receipt = {'status': 'refused', 'reason': str(error),
                                'sourceStateId': before_state, 'finalStateId': before_state,
                                'nativeActionsAppended': 0, 'publicCalls': 0,
                                'retentionRole': 'unsupported_initial_staging_candidate',
                                'admittedBoneOpening': False, 'admittedPatientTask': False,
                                'ledger': joint_cranial_ledger(work_parent, limits)}
                    elif p['kind'] == 'cusa':
                        work, receipt = checked_joint_cusa_motion(work_parent, p['motion'], **kwargs)
                    else:
                        work, receipt = checked_joint_actions(work_parent, p['actions'], **kwargs)
                    row['work'] = receipt
                    if receipt['status'] != 'accepted':
                        row['failureCategory'] = _failure_category(receipt.get('reason'))
                        # A committed prefix remains available for review even
                        # when the proposed program cannot be admitted.
                        if work.state_id != work_parent.state_id:
                            stopped.append(work)
                            row['stoppedPrefixStateId'] = work.state_id
                        if receipt['status'] == 'interrupted' and _global_stop(receipt.get('reason')):
                            exhausted, stop_reason = True, receipt.get('reason')
                            interrupted = _cancelled(stop_reason)
                            break
                        continue
                    physical = _physical_key(work)
                    if physical == _physical_key(work_parent):
                        row.update(status='no_new_physical_state', failureCategory='CANDIDATE_CONTROLLER_LIMIT')
                        continue
                    before_target = len(set(work_parent.state['removed']) & goal._credit)
                    target_delta = len(set(work.state['removed']) & goal._credit)-before_target
                    if horizon == 1 and target_delta <= 0:
                        row.update(status='no_immediate_target', preparationOnly=True)
                        continue
                    exited, exit_receipt = find_checked_joint_exit(work, route,
                        limits=limits, meter=meter, hand=hand, outside_poses=outside_poses,
                        check=check, max_candidates=max_exit_candidates,
                        phase=f'{phase}.depth{depth+1}.proposal{expansions}.exit')
                    row['exit'] = exit_receipt
                    if exit_receipt['status'] != 'accepted':
                        row.update(status='checked_exit_refused',
                            failureCategory=_failure_category(exit_receipt.get('reason')))
                        stopped.append(exited)
                        if exit_receipt['status'] == 'interrupted' and _global_stop(exit_receipt.get('reason')):
                            exhausted, stop_reason = True, exit_receipt.get('reason')
                            interrupted = _cancelled(stop_reason)
                            break
                        continue
                    assessment = joint_goal_assessment(exited, goal=goal, limits=limits,
                        exit_receipt=exit_receipt)
                    ledger = joint_cranial_ledger(exited, limits)
                    row.update(status='work_and_exit_checked', preparationOnly=target_delta <= 0,
                        workStateId=work.state_id, checkedExitStateId=exited.state_id,
                        frozenTaskCheckedAtNode=assessment['completedFrozenTask'])
                    node = JointSequenceNode(work, exited, (*path, row), ledger, assessment,
                        exit_receipt['sequenceOutcome'])
                    children.append(node)
                    nodes.append(node)
                if exhausted:
                    break
            if children:
                deepest = depth+1
            if not children or exhausted:
                break
            retained = _retain_beam(children, beam_width)
            frontier = [(n.work, n.path) for n in retained]
    except _INTERRUPTIONS as error:
        exhausted, stop_reason = True, type(error).__name__+': '+str(error)
        interrupted = not isinstance(error, TransitionBudgetExceeded)

    # The entire computation counts, including discarded probes and readout.
    completed = [n for n in nodes if n.goal_assessment['completedFrozenTask']]
    pareto = sorted([n for n in completed if not any(_dominates(o, n) for o in completed)], key=_key)
    distinct, signatures = [], set()
    for node in pareto:
        signature = _physical_key(node.work)
        if signature not in signatures:
            signatures.add(signature)
            distinct.append(node)
    pareto = distinct
    progress = sorted([n for n in nodes if n.goal_assessment['creditedOriginalTargetCells'] > 0], key=_key)
    selected = pareto[0] if pareto else None
    consequence_decision = None
    if supported_procedure:
        # Useful work plus withdrawal may be selected with an honest residual.
        selected, frontier_nodes, consequence_decision = _compare_checked_nodes(progress,
            initial_stop, limits)
        pareto = sorted(frontier_nodes, key=_key)
        if selected is initial_stop:
            pareto = [initial_stop]
        elif selected is not None:
            pareto = [selected, *[n for n in pareto if n is not selected]]
    evidence = {'schema': SCHEMA, 'horizon': horizon, 'beamWidth': beam_width,
        'maxExpansions': max_expansions, 'expansions': expansions, 'maxProposalsPerNode': 6,
        'maxExitCandidatesPerNode': max_exit_candidates, 'deepestCheckedDepth': deepest,
        'budgetExhausted': exhausted, 'interrupted': interrupted, 'stopReason': stop_reason,
        'physicalCallsDuringSearch': meter.calls-first_calls, 'spentPublicCalls': meter.calls,
        'sourceStateId': limits.source_state_id, 'limitsHash': limits.content_hash,
        'goal': {**goal.declaration(), 'contentHash': goal.content_hash},
        'decision': ('COMPLETED_FROZEN_TASK' if selected.goal_assessment['completedFrozenTask'] else
            'STOP_WITH_RESIDUAL' if selected.goal_assessment['creditedOriginalTargetCells'] > 0 else 'STOP')
            if selected is not None else 'CHECKED_PROGRESS_ONLY' if progress else 'STOP',
        'selectedProposalIds': [r['proposalId'] for r in selected.path] if selected else [],
        'preference': 'Exclude native or source-policy violations first. Retain owned treatment credit against the full original target, separate neural, bone, B/P/W, time and available exposure Pareto outcomes. Reserve a beam slot for a lower-neural-loss continuation. Different evidence availability is incomparable; unavailable values never become zero. Among admissible frontier members prefer greater original-target treatment, then componentwise neural/structure effects and supported exposures before preparation, bone and time; incomparable neural/exposure effects require explicit unresolved STOP. Distinct harm vectors receive beam slots before duplicate burden variants. Finite beam truncation is a computation limit, never proof of clinical optimality.',
        'openingConfigurationScope': 'The source-bound opening and anatomy contract are fixed. Proposal programs may choose supported physical opening, preparation, tool configuration and work order; this search does not invent another source window.',
        'proposalScope': 'Caller-supplied source geometry/native skill inventory; not automatic patient access generation. A declarative milling_opening spec invokes the bounded adaptive checked skill as one search expansion, including all internal contact/debris batches on this same policy and meter.',
        'costScope': 'All search previews, commits, discarded candidates and exit probes share the original meter. Later export/verification replay must use that same meter and be reported separately; the caller must reserve it prospectively.',
        'admittedPatientTask': False, 'caseQualificationRequired': True,
        'scope': 'Finite proposal inventory and controller; search failure does not prove anatomical impossibility.',
        'clinicalValidation': 'none'}
    if consequence_decision is not None:
        evidence.update(consequenceDecision=consequence_decision,
            procedurePolicyHash=engine.scene.properties[PROCEDURE_PROPERTY]['policyHash'],
            scenarios=['nominal'], uncertaintyLimitation='Nominal deterministic source only; existing stress evaluators do not yet qualify this procedure contract.',
            residualStopReason=('BOUNDED_INVENTORY_OR_HORIZON_STOP_WITH_RESIDUAL'
                if selected and selected.goal_assessment['residualOriginalTargetCells'] else None))
    final_wall = time.monotonic()-limits.frozen_at
    over_wall = final_wall >= limits.max_wall_seconds
    over_calls = meter.calls > min(limits.max_public_calls, meter.max_calls)
    for node in nodes:
        node.ledger.update(spentPublicCalls=meter.calls, wallSecondsSinceMeterCreation=final_wall)
        for failed, reason in ((over_wall, 'WALL_BUDGET'), (over_calls, 'PUBLIC_CALL_BUDGET'),
                (interrupted, 'SEARCH_CANCELLED_OR_INTERRUPTED_NOT_ADMITTED')):
            if failed:
                if reason != 'SEARCH_CANCELLED_OR_INTERRUPTED_NOT_ADMITTED':
                    node.ledger['withinLimits'] = False
                    if reason not in node.ledger['refusalReasons']:
                        node.ledger['refusalReasons'].append(reason)
                node.goal_assessment['completedFrozenTask'] = False
                if reason not in node.goal_assessment['reasons']:
                    node.goal_assessment['reasons'].append(reason)
    if over_wall or over_calls or interrupted:
        selected, pareto = None, []
        evidence.update(decision='STOP', selectedProposalIds=[], budgetExhausted=True,
            stopReason=stop_reason or ('WALL_BUDGET_AT_FINAL_READOUT' if over_wall else 'PUBLIC_CALL_BUDGET'))
    evidence['wallSecondsSinceMeterCreation'] = final_wall
    evidence['returnedFrozenTask'] = bool(selected and selected.goal_assessment['completedFrozenTask'])
    return JointSearchResult(selected, tuple(pareto[:3]), tuple(progress),
        tuple(stopped), tuple(attempts), evidence)
