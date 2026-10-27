"""Bounded public native transition accounting across speculative engine clones.

The wrapper delegates state reads to an unmodified SurgicalEngine. Each public
preview costs one call, including its internal native step; each direct public
step costs one more. Clones and fresh engines share one meter. A failed probe
therefore consumes budget even when its branch is discarded.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

from .surgical_engine import SurgicalEngine


class TransitionBudgetExceeded(RuntimeError):
    """A new public native call would cross the suite's hard call limit."""


class TransitionReserveReached(TransitionBudgetExceeded):
    """Optional work reached calls reserved for later verification or cleanup."""


class TransitionMeter:
    """One sequential ledger for every native branch of a bounded comparison."""

    def __init__(self, max_calls: int, *, check=None):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('Native transition limit must be a positive integer')
        if check is not None and not callable(check):
            raise TypeError('Native transition cancellation check must be callable')
        self.max_calls = max_calls
        self.check = check
        self.preview_calls = 0
        self.direct_step_calls = 0
        self.trace = []
        self._phase = 'unlabeled'
        self._reserve_floor = 0
        self._next_branch_id = 0
        self._branches = {}
        self._last_committed_by_phase = {}
        self._last_committed_by_branch = {}

    @property
    def calls(self):
        return self.preview_calls + self.direct_step_calls

    @property
    def remaining(self):
        return self.max_calls - self.calls

    @property
    def remaining_calls(self):
        return self.remaining

    @property
    def current_phase(self):
        return self._phase

    @property
    def reserved_calls(self):
        return self._reserve_floor

    def require_headroom(self, calls: int):
        """Check optional work before starting it, without consuming a call."""
        if type(calls) is not int or calls < 0:
            raise ValueError('Required native-call headroom must be nonnegative')
        if self.remaining - self._reserve_floor < calls:
            raise TransitionReserveReached(
                f'Native transition headroom below {calls} calls in {self._phase}')

    @contextmanager
    def reserve(self, calls: int):
        """Hold calls for a later mandatory phase while optional work runs."""
        with self.phase(self._phase, reserve_calls=calls):
            yield self

    @contextmanager
    def phase(self, name: str, *, reserve_calls: int = 0):
        """Label calls and leave a floor for subsequent mandatory work."""
        if not isinstance(name, str) or not name:
            raise ValueError('Native transition phase must be named')
        if type(reserve_calls) is not int or reserve_calls < 0:
            raise ValueError('Reserved native calls must be nonnegative')
        old_phase, old_floor = self._phase, self._reserve_floor
        floor = max(old_floor, reserve_calls)
        if self.remaining < floor:
            raise TransitionReserveReached(
                f'Native transition reserve unavailable for {name}')
        self._phase, self._reserve_floor = name, floor
        try:
            yield self
        finally:
            self._phase, self._reserve_floor = old_phase, old_floor

    def wrap(self, engine: SurgicalEngine):
        """Wrap a concrete engine; callers must perform transitions on the wrapper."""
        if isinstance(engine, MeteredEngine):
            if engine.meter is not self:
                raise ValueError('Cannot mix native transition meters')
            return engine
        if not isinstance(engine, SurgicalEngine):
            raise TypeError('Only an authoritative SurgicalEngine can be metered')
        return self._wrap(engine, parent_branch_id=None, origin='source')

    def _wrap(self, engine, *, parent_branch_id, origin):
        branch_id = self._next_branch_id
        self._next_branch_id += 1
        self._branches[branch_id] = {'parentBranchId': parent_branch_id,
                                     'origin': origin,
                                     'initialStateId': engine.state_id,
                                     'initialActions': len(engine.actions)}
        return MeteredEngine(engine, self, branch_id)

    def _combined_check(self, caller_check):
        if caller_check is not None and not callable(caller_check):
            raise TypeError('Native transition check must be callable')
        if self.check is None:
            return caller_check
        if caller_check is None or caller_check is self.check:
            return self.check

        def combined():
            self.check()
            caller_check()
        return combined

    def _charge(self, kind, branch_id, before_state_id, before_actions):
        if self.calls >= self.max_calls:
            raise TransitionBudgetExceeded(
                f'Native transition hard limit {self.max_calls} reached')
        if self.remaining <= self._reserve_floor:
            raise TransitionReserveReached(
                f'Native transition reserve {self._reserve_floor} reached in {self._phase}')
        if kind == 'preview':
            self.preview_calls += 1
        else:
            self.direct_step_calls += 1
        row = {'callIndex': self.calls, 'phase': self._phase, 'branchId': branch_id,
               'kind': kind, 'stateBefore': before_state_id,
               'actionsBefore': before_actions}
        self.trace.append(row)
        return row

    def _committed(self, phase, branch_id, engine):
        # Retain one detached latest committed state per phase and branch.
        # Speculative branches remain identified separately and are never
        # automatically promoted to a returned surgical plan.
        snapshot = engine.clone()
        self._last_committed_by_phase[phase] = (branch_id, snapshot)
        self._last_committed_by_branch[branch_id] = snapshot

    def last_committed_engine(self, *, phase=None, branch_id=None):
        """Return a detached last commit; its branch may be speculative."""
        if phase is not None and branch_id is not None:
            raise ValueError('Select a phase or a branch, not both')
        if branch_id is not None:
            saved = self._last_committed_by_branch.get(branch_id)
        else:
            saved = self._last_committed_by_phase.get(
                self._phase if phase is None else phase)
            saved = None if saved is None else saved[1]
        return None if saved is None else saved.clone()

    def last_committed_engine_for(self, phase):
        return self.last_committed_engine(phase=phase)

    def report(self):
        """Return detached counters and every public call, including failures."""
        by_phase = {}
        for row in self.trace:
            counts = by_phase.setdefault(row['phase'],
                {'previewCalls': 0, 'directStepCalls': 0, 'physicalCalls': 0})
            counts['previewCalls' if row['kind'] == 'preview' else 'directStepCalls'] += 1
            counts['physicalCalls'] += 1
        latest = {phase: {'branchId': branch_id, 'stateId': engine.state_id,
                          'nativeActions': len(engine.actions)}
                  for phase, (branch_id, engine) in self._last_committed_by_phase.items()}
        return {'maxPhysicalCalls': self.max_calls, 'physicalCalls': self.calls,
                'remainingPhysicalCalls': self.remaining,
                'previewCalls': self.preview_calls,
                'directStepCalls': self.direct_step_calls,
                'byPhase': by_phase,
                'branches': {str(key): copy.deepcopy(value)
                             for key, value in self._branches.items()},
                'lastCommittedByPhase': latest, 'trace': copy.deepcopy(self.trace)}


class MeteredEngine:
    """Plain delegated wrapper; it never changes SurgicalEngine class methods."""

    __slots__ = ('_engine', 'meter', 'branch_id')

    def __init__(self, engine: SurgicalEngine, meter: TransitionMeter, branch_id: int):
        self._engine = engine
        self.meter = meter
        self.branch_id = branch_id

    def unwrap(self):
        """Return the concrete engine for strictly read-only native consumers."""
        return self._engine

    def __getattr__(self, name):
        if name in {'reset', 'replay', 'branch', 'restore', '_transition_clone', '_execute'}:
            raise AttributeError('Use metered public preview/step transitions')
        return getattr(self._engine, name)

    def clone(self):
        return self.meter._wrap(self._engine.clone(),
            parent_branch_id=self.branch_id, origin='clone')

    def fresh(self):
        return self.meter._wrap(self._engine.fresh(),
            parent_branch_id=self.branch_id, origin='fresh')

    def preview(self, action, *, check=None, include_view=True, include_history=True):
        guard = self.meter._combined_check(check)
        if guard is not None:
            guard()
        before_state_id, before_actions = self._engine.state_id, len(self._engine.actions)
        row = self.meter._charge('preview', self.branch_id, before_state_id, before_actions)
        try:
            result = self._engine.preview(action, check=guard,
                include_view=include_view, include_history=include_history)
        except BaseException as error:
            row.update(exceptionType=type(error).__name__, exceptionMessage=str(error),
                       stateAfter=self._engine.state_id,
                       actionsAfter=len(self._engine.actions), action=copy.deepcopy(action))
            raise
        row.update(status=result.get('status'), reason=result.get('reason'),
                   stateAfter=self._engine.state_id,
                   actionsAfter=len(self._engine.actions), action=copy.deepcopy(action))
        return result

    def step(self, action, *, check=None):
        guard = self.meter._combined_check(check)
        if guard is not None:
            guard()
        before_state_id, before_actions = self._engine.state_id, len(self._engine.actions)
        row = self.meter._charge('step', self.branch_id, before_state_id, before_actions)
        try:
            result = self._engine.step(action, check=guard)
        except BaseException as error:
            row.update(exceptionType=type(error).__name__, exceptionMessage=str(error),
                       stateAfter=self._engine.state_id,
                       actionsAfter=len(self._engine.actions), action=copy.deepcopy(action))
            raise
        row.update(status=result.get('status'), reason=result.get('reason'),
                   stateAfter=self._engine.state_id,
                   actionsAfter=len(self._engine.actions), action=copy.deepcopy(action))
        if len(self._engine.actions) > before_actions:
            self.meter._committed(self.meter.current_phase, self.branch_id, self._engine)
        return result
