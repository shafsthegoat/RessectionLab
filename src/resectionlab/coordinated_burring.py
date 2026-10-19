"""Bounded drill/aspirate scheduling over unchanged SurgicalEngine transitions.

This task prepares a shallow bone pocket, NOT a completed craniotomy. Public
simulator counters are observations; there is no claim they are sensed in vivo.
The fixed motion macros are scripts. Only their scheduling is learned.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import numpy as np

from .surgical_rl_rewards import RewardTracker

ACTIONS = ('STOP', 'BURR', 'CLEAR')


@dataclass(frozen=True)
class TaskConfig:
    horizon: int = 4
    minimum_bone_mm3: float = 30.
    maximum_bone_mm3: float = 40.
    maximum_debris_mm3: float = 1.
    burr_seconds: float = 5.
    clear_seconds: float = 2.

    def __post_init__(self):
        if (type(self.horizon) is not int or not 1 <= self.horizon <= 8 or
            not all(np.isfinite(v) and v > 0 for k, v in asdict(self).items()) or
            self.minimum_bone_mm3 > self.maximum_bone_mm3 or
            max(self.burr_seconds, self.clear_seconds) > 5):
            raise ValueError('Invalid bounded task configuration')


class BurringTask:
    """Every pose, removal and aspiration goes through the authoritative engine.

    No direct checkpoint/state edits, action masking, preview oracle, changed
    anatomy, new physics or relaxed protection. Clones only amortize exact
    deterministic prefixes; fresh public-action replay is checked separately.
    """
    def __init__(self, engine, entry_pose, pocket_pose, config=None, *, burr_pose=None):
        self.engine = engine
        self.entry = copy.deepcopy(entry_pose)
        self.pocket = copy.deepcopy(pocket_pose)
        self.burr_pose = copy.deepcopy(burr_pose or pocket_pose)
        self.config = config or TaskConfig()
        self.decisions = []
        self.attempts = []
        self.done = False
        self.reason = None
        self.tracker = RewardTracker(engine, engine._protected_ids)
        self.measurements = self.tracker.measure(engine)
        self.total_reward = 0.

    def clone(self):
        other = copy.copy(self)
        other.engine = self.engine.clone()
        other.tracker = copy.copy(self.tracker)
        for key in ('_strain', '_thermal', '_ledger'):
            setattr(other.tracker, key, getattr(self.tracker, key).copy())
        other.decisions = copy.deepcopy(self.decisions)
        other.attempts = copy.deepcopy(self.attempts)
        other.measurements = self.measurements.copy()
        return other

    def observe(self):
        # Published task counters only: no labels, private occupancy, prospective
        # effects or source maps are accepted by the policy.
        return dict(bone_mm3=self.measurements['bone_removed_mm3'],
                    debris_mm3=float(self.engine.state['debrisMm3']),
                    remaining=self.config.horizon - len(self.decisions))

    def success(self):
        m, c = self.measurements, self.config
        return (c.minimum_bone_mm3 <= m['bone_removed_mm3'] <= c.maximum_bone_mm3 and
                self.engine.state['debrisMm3'] <= c.maximum_debris_mm3 and
                not self.failure())

    def failure(self):
        m = self.measurements
        return (m['bone_removed_mm3'] > self.config.maximum_bone_mm3 or
                any(m[k] > 0 for k in ('healthy_removed_mm3', 'tumor_removed_mm3',
                    'vessel_damage_mm2', 'bleeding_generated_mm3')) or
                self.engine.protection.damage_mm3(self.engine.state['removed']) > 0)

    def _command(self, hand, operation, **kwargs):
        action = dict(slot=hand, operation=operation, duration=.2)
        action.update(kwargs)
        outcome = self.engine.step(action)
        self.attempts.append(dict(action=action, status=outcome['status'],
                                 reason=outcome.get('reason'), stateId=self.engine.state_id))
        self.measurements = self.tracker.measure(self.engine)
        # Guard rejection and unexpected contact stops abort this macro. A
        # work-limited drill stop and a suction contact stop are visible outcomes.
        if outcome['status'] == 'rejected' or self.failure():
            self.done = True
            self.reason = 'guard_or_budget_failure'
        return not self.done

    def _park(self, hand):
        slot = self.engine.state['slots'][hand]
        if (slot['inserted'] or slot['active']) and not self._command(hand, 'withdraw'):
            return False
        pose = copy.deepcopy(self.entry)
        pose['position'][0] += 20 if hand == 'left' else -20
        return self._command(hand, 'place', pose=pose)

    def step(self, action_id):
        if self.done:
            raise RuntimeError('Task ended')
        if type(action_id) is not int or not 0 <= action_id < len(ACTIONS):
            raise ValueError('Unknown macro action')
        before = self.measurements.copy()
        action = ACTIONS[action_id]
        if action == 'STOP':
            for hand in ('left', 'right'):
                if not self._command(hand, 'deactivate'):
                    break
            self.done = True
            self.reason = self.reason or 'policy_stop'
        else:
            hand, other = ('left', 'right') if action == 'BURR' else ('right', 'left')
            if self._park(other):
                slot = self.engine.state['slots'][hand]
                ready = not (slot['inserted'] or slot['active']) or self._command(hand, 'withdraw')
                params = (dict(radiusMm=3., forceN=5., rpm=60000., duty=1.) if action == 'BURR'
                          else dict(radiusMm=3., flowMm3PerSecond=1000.))
                if ready and self._command(hand, 'place', pose=self.entry, parameters=params):
                    if self._command(hand, 'apply', pose=self.burr_pose if action == 'BURR' else self.pocket,
                                     duration=self.config.burr_seconds if action == 'BURR' else self.config.clear_seconds):
                        self._command(hand, 'deactivate')
        m, c = self.measurements, self.config
        # Monotone source removal and aspirated-debris ledgers prevent cycling
        # refunds. Reward coefficients are frozen before collecting experience.
        gain = (min(c.minimum_bone_mm3, m['bone_removed_mm3']) -
                min(c.minimum_bone_mm3, before['bone_removed_mm3'])) / c.minimum_bone_mm3
        aspirated = float(self.engine.state.get('debrisRemovedMm3', 0.))
        previous_aspirated = getattr(self, '_credited_aspiration', 0.)
        cleared = (min(c.minimum_bone_mm3, aspirated) - min(c.minimum_bone_mm3, previous_aspirated)) / c.minimum_bone_mm3
        self._credited_aspiration = aspirated
        terms = dict(bone_progress=.2 * gain, clearance=.2 * cleared,
                     decision=-.05, time=-.001 * (m['elapsed_seconds'] - before['elapsed_seconds']),
                     failure=-1. if self.reason == 'guard_or_budget_failure' else 0., success=0.)
        if self.success():
            self.done = True
            self.reason = 'success'
            terms['success'] = 1.
        elif len(self.decisions) + 1 >= c.horizon and not self.done:
            self.done = True
            self.reason = 'horizon'
        reward = float(sum(terms.values()))
        self.total_reward += reward
        self.decisions.append(dict(action=action, reward=reward, terms=terms, reason=self.reason))
        self.decisions[-1]['observation'] = self.observe()
        return self.observe(), reward, self.done


def state_key(observation):
    return (round(observation['bone_mm3'], 6), round(observation['debris_mm3'], 6),
            observation['remaining'])


class QPolicy:
    """Tabular, off-policy Q learning. Ties choose STOP, including initialization."""
    def __init__(self):
        self.table = {}
        self.updates = 0

    def values(self, observation):
        return self.table.get(state_key(observation), np.zeros(len(ACTIONS)))

    def choose(self, observation):
        return int(np.argmax(self.values(observation)))

    def update(self, observation, action, reward, following, done, alpha=.25, gamma=.95):
        key = state_key(observation)
        values = self.table.setdefault(key, np.zeros(len(ACTIONS)))
        target = reward + (0. if done else gamma * float(self.values(following).max()))
        error = float(target - values[action])
        values[action] += alpha * error
        self.updates += 1
        return error

    def record(self):
        return dict(updates=self.updates, table=[dict(state=list(k), values=v.tolist())
                                               for k, v in sorted(self.table.items())])
