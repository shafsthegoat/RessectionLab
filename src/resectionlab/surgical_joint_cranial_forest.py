"""Checked comparisons across immutable opening roots for one source patient.

Each opening keeps its own registered native policy. One outer budget charges
all roots, discarded probes and mandatory export replay. No opening contract is
substituted into an acted branch, and skull progress is never target treatment.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import time

from .core import semantic_digest
from .surgical_transition_meter import TransitionMeter, TransitionBudgetExceeded, TransitionReserveReached


class OpeningComparisonBudget:
    def __init__(self, max_calls, max_wall_seconds, *, export_reserve, check=lambda: None,
                 started_at=None, clock=time.monotonic):
        if (type(max_calls) is not int or max_calls < 1 or
                type(export_reserve) is not int or not 0 <= export_reserve < max_calls or
                type(max_wall_seconds) not in (float, int) or
                not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0 or not callable(check)):
            raise ValueError('CRANIAL_COMPARISON_EXPLICIT_BUDGET_REQUIRED')
        self.max_calls = max_calls
        self.max_wall_seconds = float(max_wall_seconds)
        self.export_reserve = export_reserve
        self._clock, self._check = clock, check
        self.started_at = clock() if started_at is None else started_at
        if (type(self.started_at) not in (float, int) or not math.isfinite(self.started_at)
                or not 0 <= self.started_at <= clock()):
            raise ValueError('CRANIAL_COMPARISON_CLOCK_ORIGIN_REQUIRED')
        self._meters = {}
        self._exporting = False
        if clock is time.monotonic:
            from .surgical_native_control import derive_native_check
            self.check = derive_native_check(check, self.check,
                deadline=self.started_at + self.max_wall_seconds)

    @property
    def calls(self):
        return sum(m.calls for m in self._meters.values())

    def check(self):
        self._check()
        if self._clock() - self.started_at >= self.max_wall_seconds:
            raise TimeoutError('CRANIAL_COMPARISON_WHOLE_WALL_BUDGET')
        if self.calls > self.max_calls:
            raise TransitionBudgetExceeded('CRANIAL_COMPARISON_WHOLE_CALL_BUDGET')

    def before_call(self):
        self.check()
        floor = 0 if self._exporting else self.export_reserve
        if self.calls >= self.max_calls - floor:
            error = TransitionReserveReached if floor else TransitionBudgetExceeded
            raise error('CRANIAL_COMPARISON_EXPORT_RESERVE' if floor else
                        'CRANIAL_COMPARISON_WHOLE_CALL_BUDGET')

    def meter(self, root_id):
        self.check()
        if type(root_id) is not str or not root_id or root_id in self._meters or len(self._meters) >= 3:
            raise ValueError('CRANIAL_COMPARISON_DISTINCT_ROOTS_MAX3')
        meter = _OpeningMeter(self, root_id)
        self._meters[root_id] = meter
        return meter

    @contextmanager
    def export(self):
        if self._exporting:
            raise ValueError('CRANIAL_COMPARISON_EXPORT_ALREADY_ACTIVE')
        self.check()
        self._exporting = True
        try:
            yield
            self.check()
        finally:
            self._exporting = False

    def report(self):
        return {'physicalCalls': self.calls,
                'previewCalls': sum(m.preview_calls for m in self._meters.values()),
                'directStepCalls': sum(m.direct_step_calls for m in self._meters.values()),
                'wallSeconds': self._clock() - self.started_at,
                'maxPhysicalCalls': self.max_calls, 'maxWallSeconds': self.max_wall_seconds,
                'exportReplayReserveCalls': self.export_reserve,
                'roots': {key: meter.report() for key, meter in self._meters.items()},
                'scope': 'All opening roots, rejected/discarded calls and export replay; no reset between roots.'}


class _OpeningMeter(TransitionMeter):
    def __init__(self, comparison, root_id):
        super().__init__(comparison.max_calls, check=comparison.check)
        self.comparison, self.root_id = comparison, root_id

    def _charge(self, kind, branch_id, before_state_id, before_actions):
        self.comparison.before_call()
        row = super()._charge(kind, branch_id, before_state_id, before_actions)
        row['openingRootId'] = self.root_id
        row['comparisonCallIndex'] = self.comparison.calls
        return row


def replay_export(source, candidate, meter, *, check):
    """Reproduce the exact expanded native history once, charging every step."""
    if (source.actions or source.history or source.scene.identity != candidate.scene.identity or
            len(candidate.actions) != len(candidate.history)):
        raise ValueError('CRANIAL_EXPORT_CANONICAL_MATCHING_ROOT_REQUIRED')
    fresh = meter.wrap(source.fresh())
    with meter.phase('cranial.desktop.export-literal'):
        for index, (action, expected) in enumerate(zip(candidate.actions, candidate.history)):
            check()
            fresh.step(copy.deepcopy(action), check=check)
            if (len(fresh.actions) != index + 1 or
                    semantic_digest(fresh.history[-1]) != semantic_digest(expected)):
                raise ValueError('CRANIAL_EXPORT_NATIVE_HISTORY_MISMATCH')
    if semantic_digest(fresh.snapshot()) != semantic_digest(candidate.snapshot()):
        raise ValueError('CRANIAL_EXPORT_COMPLETE_SNAPSHOT_MISMATCH')
    check()
    return fresh.unwrap()


def candidate_key(root_id, node):
    return semantic_digest({'rootId': root_id, 'sceneId': node.exited.scene.identity,
        'stateId': node.exited.state_id, 'actionHash': semantic_digest(node.exited.actions)})


def candidate_branch_id(root_id, node):
    return 'cranial-'+root_id+'-'+candidate_key(root_id,node).split(':')[-1][:16]


def forest_priority(candidates, *, procedure):
    """Compare the complete endpoint-eligible forest before display limits.

    Display ordering and the number of exported branches never participate in
    preference. Unknown required dimensions reject a candidate, not become zero.
    The returned certificate is recomputed from source-bound states on reopen.
    """
    from .surgical_consequences import procedure_priority_choice, PROCEDURE_PRIORITY, EXPOSURE_METRICS
    from .surgical_joint_cranial_search import endpoint_eligible
    entries = {candidate_key(root, node): (root, node) for root, node in candidates}
    required = {'targetRemovedMm3', 'nonbonePreparationRemovedMm3', 'boneRemovedMm3', 'elapsedSeconds'}
    anatomical = {'neuralRemovedMm3', 'neuralCutSourceMm3', 'protectedRemovedMm3'}
    for _, node in candidates:
        dims = node.sequence_outcome['dimensions']
        anatomical.update('structure:'+name+':'+effect
            for field, effect in (('structureRemovedMm3','removedMm3'), ('structureCutSourceMm3','cutSourceMm3'))
            for name in dims.get(field, {}))
        required.update(name for name in EXPOSURE_METRICS if dims.get(name, {}).get('cumulative') is not None)
    def value(node, name):
        if name == 'targetRemovedMm3':return node.goal_assessment['originalTargetTreatmentMm3']
        if name == 'nonbonePreparationRemovedMm3':return node.ledger['preparationNonboneRemovedMm3']
        if name == 'boneRemovedMm3':return node.ledger['boneRemovedMm3']
        if name == 'elapsedSeconds':return node.ledger['modeledElapsedSeconds']
        dims = node.sequence_outcome['dimensions']
        if name.startswith('structure:'):
            _, structure, effect = name.split(':')
            quantity = dims.get('structureRemovedMm3' if effect == 'removedMm3' else
                'structureCutSourceMm3', {}).get(structure, {})
        else:quantity = dims.get(name, {})
        return None if quantity.get('status') == 'unavailable' else quantity.get('cumulative')
    required.update(name for name in anatomical if procedure or any(value(n, name) is not None for _, n in candidates))
    rows, eligible = [], []
    for key, (root, node) in sorted(entries.items()):
        missing = sorted(name for name in required if value(node, name) is None)
        if any(type(value(node, name)) not in (float, int) or not math.isfinite(value(node, name)) or
                value(node, name) < 0 for name in required if name not in missing):
            raise ValueError('CRANIAL_FOREST_FINITE_TYPED_CONSEQUENCES_REQUIRED')
        endpoint = node.goal_assessment.get('endpointAssessment')
        valid = node.ledger['withinLimits'] and endpoint_eligible(endpoint) and not missing
        rows.append({'candidateKey': key, 'rootId': root, 'sceneId': node.exited.scene.identity,
            'finalStateId': node.exited.state_id, 'eligible': bool(valid),
            'endpointAssessment': copy.deepcopy(endpoint), 'missingRequiredMetrics': missing,
            'values': {name: value(node, name) for name in sorted(required)}})
        if valid:eligible.append(key)
    selected, reason = (procedure_priority_choice(eligible,
        lambda key, name: value(entries[key][1], name), required) if eligible else
        (None, 'no_verified_endpoint_stop_unavailable'))
    return entries.get(selected), {'schema': 'resectionlab.opening-forest-priority.v1',
        'priority': copy.deepcopy(PROCEDURE_PRIORITY), 'requiredMetrics': sorted(required),
        'candidates': rows, 'selectedCandidateKey': selected, 'reason': reason,
        'candidateCount': len(rows), 'eligibleCount': len(eligible),
        'checkedStopCandidates': [key for key in eligible if
            entries[key][1].goal_assessment['creditedOriginalTargetCells'] == 0],
        'scope': 'Complete checked forest before display/export truncation; generated engineering consequences only.'}


def search_opening_forest(roots, *, budget, horizon=5, beam_width=3,
                          max_expansions=24, progress=lambda *args: None):
    """Search qualified caller-owned roots; return only exactly replayed plans.

    A root supplies its unacted native engine, explicit limits, source-only
    proposer and exit poses. Anatomical qualification belongs to the source
    adapter; this function cannot confer it or construct a missing opening.
    """
    from .surgical_joint_cranial_search import (
        checked_joint_sequence_search, freeze_joint_resection_goal, _dominates, _key, _physical_key)
    from .surgical_joint_cranial_skills import freeze_joint_cranial_limits
    from .surgical_cranial_planning_roots import source_binding
    from .surgical_procedure_policy import PROPERTY as PROCEDURE_PROPERTY
    if not isinstance(budget, OpeningComparisonBudget) or not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= 3:
        raise ValueError('CRANIAL_FOREST_ONE_TO_THREE_ROOTS_REQUIRED')
    from .surgical_planning_admission import require_planning_admission
    for root in roots:
        require_planning_admission(root['engine'])
    candidates, failures, searches, sources, meters = [], [], [], {}, {}
    policy_hash, target_partition, source_hash, guard_hash = None, None, None, None
    prepared = []
    procedure_mode = PROCEDURE_PROPERTY in roots[0]['engine'].scene.properties
    # Validate the entire forest before its first native call. A later root
    # cannot replace the patient, target, protection or comparison policy.
    for root in roots:
        budget.check()
        root_id, source = root['id'], root['engine']
        if source.actions or source.history:
            raise ValueError('CRANIAL_FOREST_UNACTED_CANONICAL_ROOT_REQUIRED')
        current_policy = semantic_digest(root['limits'])
        if policy_hash is not None and policy_hash != current_policy:
            raise ValueError('CRANIAL_FOREST_COMMON_COMPARISON_LIMITS_REQUIRED')
        policy_hash = current_policy
        binding = source_binding(source.scene)
        binding.pop('baseSceneId')
        current_source = semantic_digest(binding)
        guard = source.snapshot().get('protectionContract')
        current_guard = semantic_digest(None if guard is None else {
            k: v for k, v in guard.items() if k not in ('sceneId', 'contentHash')})
        if (source_hash is not None and source_hash != current_source or
                guard_hash is not None and guard_hash != current_guard):
            raise ValueError('CRANIAL_FOREST_SAME_ANATOMY_AND_PROTECTION_REQUIRED')
        source_hash, guard_hash = current_source, current_guard
        meter = budget.meter(root_id)
        sources[root_id], meters[root_id] = source, meter
        limits = freeze_joint_cranial_limits(source, meter=meter, **root['limits'])
        goal = freeze_joint_resection_goal(source, limits=limits)
        partition = semantic_digest([sorted(zone) for zone in goal._zones])
        if target_partition is not None and target_partition != partition:
            raise ValueError('CRANIAL_FOREST_SAME_FULL_TARGET_PARTITION_REQUIRED')
        target_partition = partition
        prepared.append((root, source, meter, limits, goal))
    for index, (root, source, meter, limits, goal) in enumerate(prepared):
        root_id = root['id']
        budget.check()
        progress(.1 + .55 * index / len(roots), 'Checking opening ' + root.get('label', root_id))
        result = checked_joint_sequence_search(source, root.get('route'), propose=root['propose'],
            limits=limits, goal=goal, meter=meter, outside_poses=root.get('outside_poses'),
            horizon=horizon, beam_width=beam_width, max_expansions=max_expansions,
            check=budget.check, phase='cranial.desktop.' + root_id)
        searches.append({'rootId': root_id, 'evidence': result.evidence,
                         'attempts': result.attempts})
        retained_nodes = getattr(result, 'checked_nodes', ()) or (*result.alternatives, *result.progress)
        for node in retained_nodes:
            if node.ledger['withinLimits']:
                candidates.append((root_id, node))
        for attempt in result.attempts:
            if attempt.get('failureCategory'):
                failures.append({'rootId': root_id, 'code': attempt['failureCategory'],
                                 'message': str(attempt.get('work', {}).get('reason') or attempt.get('reason')),
                                 'constraintViolations': copy.deepcopy(attempt.get('work', {}).get('constraintViolations', []))})
        if result.evidence.get('interrupted'):
            raise InterruptedError('CRANIAL_FOREST_SEARCH_INTERRUPTED')
    # Only equivalent evidence availability is Pareto-comparable. Preserve
    # physically distinct root choices; a different label alone is insufficient.
    unique = {}
    for root_id, node in candidates:
        signature = candidate_key(root_id, node)
        unique.setdefault(signature, (root_id, node))
    ordered = sorted(unique.values(), key=lambda item: (*_key(item[1]), candidate_key(*item)))
    selected_candidate, selection = forest_priority(ordered, procedure=procedure_mode)
    eligible_keys = {r['candidateKey'] for r in selection['candidates'] if r['eligible']}
    eligible = [item for item in ordered if candidate_key(*item) in eligible_keys]
    frontier = [item for item in eligible if not any(
        _dominates(other[1], item[1]) for other in eligible if other is not item)]
    # Recommendation was settled over the full set above. These lists affect
    # inspection only, including honest dirty/truncated prefixes when useful.
    pool = [*frontier, *[item for item in ordered if not any(item is f for f in frontier)]]
    if selected_candidate is not None:
        pool = [selected_candidate, *[item for item in pool if candidate_key(*item) != candidate_key(*selected_candidate)]]
    elif selection['checkedStopCandidates']:
        stop_key = min(selection['checkedStopCandidates'])
        stop = next(item for item in eligible if candidate_key(*item) == stop_key)
        pool = [stop, *[item for item in pool if candidate_key(*item) != stop_key]]
    if pool:
        # Once preference is fixed, reserve inspection space for another
        # physical opening rather than three variants from the same root.
        leading=pool[0];represented={leading[0]};representatives=[]
        for item in ordered:
            if item[0] not in represented and item[1].goal_assessment['creditedOriginalTargetCells']>0:
                represented.add(item[0]);representatives.append(item)
        front=[leading,*representatives]
        front_keys={candidate_key(*item) for item in front}
        pool=[*front,*[item for item in pool if candidate_key(*item) not in front_keys]]
    exported = []
    with budget.export():
        for root_id, node in pool[:3]:
            budget.check()
            if len(node.exited.actions) > budget.max_calls - budget.calls:
                failures.append({'rootId': root_id, 'code': 'EXPORT_REPLAY_CALL_BUDGET',
                                 'message': 'Exact full-sequence replay does not fit the remaining shared budget.'})
                continue
            progress(.7, 'Verifying complete sequence for ' + root_id)
            replayed = replay_export(sources[root_id], node.exited, meters[root_id], check=budget.check)
            exported.append({'rootId': root_id, 'engine': replayed, 'node': node})
    budget.check()
    selected_key = selection['selectedCandidateKey']
    selected = next((item for item in exported if candidate_key(item['rootId'], item['node']) == selected_key), None)
    if selected_key is not None and selected is None:
        selection['exportFailure'] = 'PREFERRED_ENDPOINT_EXPORT_UNAVAILABLE'
    decision = ('COMPLETED_FROZEN_TASK' if selected['node'].goal_assessment['completedFrozenTask'] else
        'STOP_WITH_RESIDUAL' if selected['node'].goal_assessment['creditedOriginalTargetCells'] > 0 else 'STOP') if selected else 'STOP'
    return {'selected': selected, 'branches': exported, 'failures': failures,
            'searches': searches, 'cost': budget.report(), 'selectionEvidence': selection,
            'candidates': ordered,
            'decision': decision,
            'admittedPatientTask': False, 'clinicalValidation': 'none'}
