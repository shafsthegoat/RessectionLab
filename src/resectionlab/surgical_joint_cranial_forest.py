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
    if not isinstance(budget, OpeningComparisonBudget) or not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= 3:
        raise ValueError('CRANIAL_FOREST_ONE_TO_THREE_ROOTS_REQUIRED')
    candidates, failures, searches, sources, meters = [], [], [], {}, {}
    policy_hash, target_partition, source_hash, guard_hash = None, None, None, None
    prepared = []
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
        for node in (*result.alternatives, *result.progress):
            if node.ledger['withinLimits']:
                candidates.append((root_id, node))
        for attempt in result.attempts:
            if attempt.get('failureCategory'):
                failures.append({'rootId': root_id, 'code': attempt['failureCategory'],
                                 'message': str(attempt.get('work', {}).get('reason') or attempt.get('reason'))})
        if result.evidence.get('interrupted'):
            raise InterruptedError('CRANIAL_FOREST_SEARCH_INTERRUPTED')
    # Only equivalent evidence availability is Pareto-comparable. Preserve
    # physically distinct root choices; a different label alone is insufficient.
    unique = {}
    for root_id, node in candidates:
        signature = (root_id, _physical_key(node.exited))
        unique.setdefault(signature, (root_id, node))
    ordered = sorted(unique.values(), key=lambda item: (
        not item[1].goal_assessment['completedFrozenTask'], *_key(item[1]), item[0]))
    complete = [item for item in ordered if item[1].goal_assessment['completedFrozenTask']]
    frontier = [item for item in complete if not any(
        _dominates(other[1], item[1]) for other in complete if other is not item)]
    pool = frontier if frontier else ordered
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
    selected = next((item for item in exported if item['node'].goal_assessment['completedFrozenTask']), None)
    return {'selected': selected, 'branches': exported, 'failures': failures,
            'searches': searches, 'cost': budget.report(),
            'decision': 'COMPLETED_FROZEN_TASK' if selected else 'CHECKED_PROGRESS_ONLY' if exported else 'STOP',
            'admittedPatientTask': False, 'clinicalValidation': 'none'}
