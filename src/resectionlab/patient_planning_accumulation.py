"""One admitted shared update, backward one complete-trace action at a time.

Same IL action mean / RL episode mean, one global clip and one Adam step. Only
one action graph lives at once. A complete patient trace is still required for
admission and RL returns; this is not truncated BPTT or online patient adaptation.
Failed partial accumulation poisons the session and cannot be retried.
"""
import math

import torch
from torch import nn

from .patient_planning_learning import (PatientTrainSession, PatientTrainingTrace,
    _index, patient_imitation_group_weights)
from .spatial_policy import parameter_hash


class PatientGradientAccumulator:
    def __init__(self, session, *, teacher_pins=None, rl_diagnostics=False, motion_ranking=None):
        if type(session) is not PatientTrainSession:
            raise TypeError('Exact admitted patient learning session required')
        session.require(session.method)
        if type(rl_diagnostics) is not bool or (rl_diagnostics and session.method != 'RL'):
            raise ValueError('RL decision diagnostics require an explicit boolean and an RL session')
        self.motion_ranking = motion_ranking
        ranking_spec = session.protocol.get('cohort_execution', {}).get('il_motion_supervision')
        if ranking_spec is not None:
            from .public_motion_ranking import PublicMotionRankingCorpus
            if (session.method != 'IL' or type(motion_ranking) is not PublicMotionRankingCorpus
                    or motion_ranking.require().fingerprint != ranking_spec['corpus_hash']):
                raise ValueError('Exact protocol-bound public motion score corpus required')
        elif motion_ranking is not None:
            raise ValueError('Ranking labels need explicit prospective protocol opt-in')
        self.rl_diagnostics = rl_diagnostics
        if session._permit is not None:
            raise ValueError('An admitted loss or accumulation already owns this update')
        self.session = session
        self.order = tuple(c.fingerprint for c in session.contexts)
        self.before = parameter_hash(session.policy)
        self.teacher_pins = None
        self.balanced = (session.method == 'IL' and session.protocol.get('cohort_execution', {}).get('il_teacher_weighting') is not None)
        if session.method == 'IL':
            fields = {'steps', 'trace_seal'} | ({'stop_steps'} if self.balanced else set())
            if (type(teacher_pins) is not dict or set(teacher_pins) != set(self.order)
                    or any(type(v) is not dict or set(v) != fields
                           or type(v['steps']) is not int or not 1 <= v['steps'] <= c.max_steps
                           or not isinstance(v['trace_seal'], str)
                           for c in session.contexts for k,v in [(c.fingerprint, teacher_pins[c.fingerprint])])):
                raise ValueError('Pinned complete teacher count and seal for every TRAIN context required')
            self.teacher_pins = {k: dict(v) for k,v in teacher_pins.items()}
            self.action_count = sum(v['steps'] for v in self.teacher_pins.values())
            if self.balanced:
                if any(type(v['stop_steps']) is not int or not 0 <= v['stop_steps'] <= v['steps']
                       for v in self.teacher_pins.values()):
                    raise ValueError('Exact pinned STOP count required for each complete teacher')
                stops = sum(v['stop_steps'] for v in self.teacher_pins.values())
                self.group_counts = {'STOP': stops, 'motion': self.action_count-stops}
                self.group_weights = patient_imitation_group_weights(session.protocol,
                    stop_count=stops, motion_count=self.action_count-stops)
        elif teacher_pins is not None:
            raise ValueError('RL must collect fresh on-policy episodes, not teacher pins')
        self.rows = []; self.finished = False; self.failed = False
        self.loss = self.actor = self.value = self.entropy = 0.
        session._permit = self
        session._optimizer.zero_grad(set_to_none=True)

    def __enter__(self):
        self._require()
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is not None or not self.finished:
            self.abort()
        if error_type is None and not self.finished:
            raise ValueError('Incomplete cohort accumulation cannot commit an update')

    def _require(self):
        self.session.require(self.session.method)
        if (self.finished or self.failed or self.session._permit is not self
                or parameter_hash(self.session.policy) != self.before):
            raise ValueError('Changed, stale, consumed or failed accumulation')

    def abort(self):
        if not self.finished:
            self.failed = True
            self.session._permit = None
            self.session._attempted = True
            self.session._optimizer.zero_grad(set_to_none=True)

    def add_trace(self, trace, *, guard=lambda: None):
        """Consume one complete trace in declared context order; retain no arrays.

The caller may release the trace and its source after return. Autograd frees
each action's activations immediately after backward; the shared leaf gradients
remain until finish. Reconstructed teachers must match the initial trace seal.
"""
        try:
            self._require(); guard()
            if type(trace) is not PatientTrainingTrace:
                raise TypeError('Exact admitted complete native trace required')
            self.session.require_trace(trace)
            if (len(self.rows) >= len(self.order)
                    or trace.context.fingerprint != self.order[len(self.rows)]):
                raise ValueError('Duplicate, foreign or out-of-order TRAIN contribution')
            count = len(trace.transitions); seal = trace.seal_hash
            if self.session.method == 'IL':
                pin = self.teacher_pins[trace.context.fingerprint]
                if count != pin['steps'] or seal != pin['trace_seal']:
                    raise ValueError('Reconstructed complete teacher differs from its frozen plan')
                if self.balanced and sum(row.action_id == 'STOP' for row in trace.transitions) != pin['stop_steps']:
                    raise ValueError('Reconstructed complete teacher STOP count differs from its pin')
                ranking_rows = (None if self.motion_ranking is None
                                else self.motion_ranking.require_trace(trace))
                targets = [None]*count
            else:
                if trace.behavior_parameter_hash != self.before:
                    raise ValueError('Stale on-policy episode')
                targets = []; total = 0.
                for row in reversed(trace.transitions):
                    total = row.reward+self.session.protocol['gamma']*total
                    targets.append(total)
                targets.reverse()
            local = {'loss': 0., 'actor_loss': 0., 'value_loss': 0., 'entropy': 0.}
            decisions = [] if self.rl_diagnostics else None
            for step, (row, target) in enumerate(zip(trace.transitions, targets)):
                guard()
                logits, value = self.session.policy(row.observation)
                index = _index(row.observation, row.action_id)
                if self.session.method == 'IL':
                    if self.motion_ranking is not None:
                        from .public_motion_ranking import motion_ranking_loss
                        labels = ranking_rows[step]
                        term = motion_ranking_loss(logits, action_ids=labels['action_ids'],
                            action_mask=labels['action_mask'], rewards=labels['rewards'],
                            teacher_action=row.action_id)*self.group_weights['STOP' if row.action_id=='STOP' else 'motion']
                    else:
                        term = (-logits.log_softmax(-1)[index]*self.group_weights['STOP' if row.action_id=='STOP' else 'motion']
                                if self.balanced else -logits.log_softmax(-1)[index]/self.action_count)
                else:
                    distribution = torch.distributions.Categorical(logits=logits)
                    actor = -(self.session.protocol['gamma']**step)*distribution.log_prob(
                        logits.new_tensor(index, dtype=torch.long))*(value.new_tensor(target)-value.detach())
                    value_term = (value-target).square()
                    entropy = distribution.entropy()
                    episodes = len(self.order)
                    term = (actor/episodes
                        +self.session.protocol['value_weight']*value_term/(episodes*count)
                        -self.session.protocol['entropy_weight']*entropy/(episodes*count))
                    local['actor_loss'] += float(actor.detach())/episodes
                    local['value_loss'] += float(value_term.detach())/(episodes*count)
                    local['entropy'] += float(entropy.detach())/(episodes*count)
                    if self.rl_diagnostics:
                        # Read the existing forward only. Diagnostic scalar math
                        # stays outside autograd and never changes the loss term.
                        with torch.no_grad():
                            decisions.append({
                                'step': step, 'observation_hash': row.observation.fingerprint,
                                'action_id': row.action_id, 'action_index': index,
                                'legal_action_count': int(row.observation.action_mask.sum()),
                                'reward': float(row.reward), 'terminated': bool(row.terminated),
                                'return_to_go': float(target),
                                'return_to_go_model_dtype': float(value.new_tensor(target)),
                                'value': float(value.detach()),
                                'detached_advantage': float(value.new_tensor(target)-value.detach()),
                                'chosen_log_probability': float(distribution.log_prob(
                                    logits.new_tensor(index, dtype=torch.long))),
                                'entropy': float(entropy.detach()),
                                'actor_discount': float(self.session.protocol['gamma']**step),
                                'actor_score_term': float(actor.detach()),
                                'value_squared_error': float(value_term.detach())})
                if not torch.isfinite(term):
                    raise FloatingPointError('Nonfinite accumulated patient loss')
                local['loss'] += float(term.detach())
                term.backward()
                # Do not keep a tensor, distribution or graph in the receipt.
                del term, logits, value
                if self.session.method == 'RL': del actor, value_term, entropy, distribution
            trace.require(); self._require(); guard()
            if not all(math.isfinite(v) for v in local.values()):
                raise FloatingPointError('Nonfinite accumulated scalar accounting')
            row = {'context_hash': trace.context.fingerprint,
                'patient_group': trace.context.patient_group, 'trace_seal': seal,
                'steps': count, 'loss_forward_calls': count, **local}
            if self.session.method == 'RL': row['return'] = targets[0]
            if self.rl_diagnostics:
                row['rl_decision_diagnostics'] = {
                    'version': 'patient-RL-decision-scalars-v1',
                    'scope': 'same_on_policy_loss_forwards_before_shared_update',
                    'behavior_parameter_hash': self.before,
                    'episodes_in_update': len(self.order),
                    'value_weight': self.session.protocol['value_weight'],
                    'entropy_weight': self.session.protocol['entropy_weight'],
                    'decisions': decisions}
            self.rows.append(row)
            self.loss += local['loss']; self.actor += local['actor_loss']
            self.value += local['value_loss']; self.entropy += local['entropy']
            return dict(row)
        except BaseException:
            self.abort()
            raise

    def finish(self, *, guard=lambda: None):
        try:
            self._require(); guard()
            if tuple(row['context_hash'] for row in self.rows) != self.order:
                raise ValueError('Every admitted TRAIN context must contribute before the shared update')
            session = self.session; policy = session.policy
            # Mirrors patient_gradient_step: only optimizer failure can leave a
            # partial parameter mutation, and that session remains poisoned.
            session._permit = None; session._attempted = True
            norms = {}
            for name in ('encoder', 'actor', 'stop', 'critic'):
                squares = [p.grad.detach().square().sum() for p in getattr(policy, name).parameters() if p.grad is not None]
                norms[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.
            total = float(nn.utils.clip_grad_norm_(policy.parameters(),
                session.protocol['max_gradient_norm'], error_if_nonfinite=True))
            session._optimizer.step(); after = parameter_hash(policy)
            session.updates += 1; session._completed_updates += 1
            session._expected_parameter_hash = after; session._attempted = False
            self.finished = True
            return {'version': session.protocol['version'], 'method': session.method,
                'optimizer_updates': 1, 'before_parameter_hash': self.before,
                'after_parameter_hash': after, 'parameters_changed': after != self.before,
                'gradient_norm_before_clip': total, 'module_gradient_norms_before_clip': norms,
                'context_hashes': list(self.order), 'trace_seals': [r['trace_seal'] for r in self.rows],
                'completed_updates': session.updates, 'population_training': session.protocol['population_training'],
                'patient_adaptation': False, 'loss': self.loss,
                'loss_forward_calls': sum(r['steps'] for r in self.rows),
                'actor_loss': self.actor, 'value_loss': self.value, 'entropy': self.entropy,
                'IL_reduction': ('balanced_STOP_gate_public_nominal_motion_gap_ranking_v1' if self.motion_ranking is not None
                    else 'balanced_STOP_motion_CE_v1' if self.balanced else 'mean_all_teacher_actions'),
                **({'public_motion_ranking_corpus_hash': self.motion_ranking.fingerprint,
                    'supervision': 'saved_public_nominal_scores_training_only_no_added_forwards_or_previews'}
                   if self.motion_ranking is not None else {}),
                **({'teacher_group_counts': self.group_counts,
                    'teacher_group_weights': self.group_weights} if self.balanced else {}),
                'RL_reduction': 'mean_episodes_of_actor_sum_and_value_entropy_step_means',
                'accumulation': 'one_action_graph_at_a_time_one_clip_one_Adam_step',
                'numerical_equivalence': 'same_objective_floating_point_order_may_differ'}
        except BaseException:
            self.abort()
            raise
