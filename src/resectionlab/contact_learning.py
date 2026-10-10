"""Minimal IL/episodic policy-gradient math behind separate TRAIN admission."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .contact_learning_contract import (ContactExperiment, ContactSample, ContactTransition,
    PROTOCOL, VERSION, expected_initial_parameter_hash)
from .core import semantic_digest, array_digest
from .goal_mode_spatial_policy import GoalModeSpatialPolicy
from .spatial_policy import parameter_hash


@dataclass
class ContactLearningSession:
    experiment: ContactExperiment
    method: str
    policy: GoalModeSpatialPolicy
    initial_parameter_hash: str
    updates: int = 0

    def __post_init__(self):
        if (type(self.experiment) is not ContactExperiment or self.method not in ('IL', 'RL')
                or type(self.policy) is not GoalModeSpatialPolicy or self.updates != 0
                or self.policy.architecture_hash != self.experiment.record()['architecture_hash']
                or parameter_hash(self.policy) != self.initial_parameter_hash
                or self.initial_parameter_hash != expected_initial_parameter_hash()):
            raise ValueError('New common-initialized policy and exact fixed learning contract required')
        self._session_identity = (id(self.experiment), self.experiment.fingerprint, self.method,
            id(self.policy), self.initial_parameter_hash, self.policy.architecture_hash)
        self._expected_parameter_hash = self.initial_parameter_hash
        self._completed_updates = 0
        self._permits = {}
        self._used_bindings = {}
        self._optimizer_identity = None

    def assert_intact(self):
        """Check the frozen session through its final checkpoint boundary."""
        self.experiment.assert_intact()
        current = (id(self.experiment), self.experiment.fingerprint, self.method,
            id(self.policy), self.initial_parameter_hash, self.policy.architecture_hash)
        if (current != self._session_identity
                or self.updates != self._completed_updates
                or parameter_hash(self.policy) != self._expected_parameter_hash
                or not 0 <= self.updates <= PROTOCOL['updates']):
            raise ValueError('Session/architecture/method changed or invalid update count')

    def require(self, method):
        self.assert_intact()
        if method != self.method or self.updates >= PROTOCOL['updates']:
            raise ValueError('Session method differs or fixed update cap exhausted')

    def require_samples(self, samples):
        for sample in samples:
            if type(sample) is not ContactSample:
                raise TypeError('Exact TRAIN-bound contact samples required')
            sample.validate()
            if sample.binding.experiment.fingerprint != self.experiment.fingerprint:
                raise ValueError('Loss sample belongs to another frozen experiment')

    def register_loss(self, loss, samples):
        samples = tuple(samples)
        self.require_samples(samples)
        binding = (self.experiment.fingerprint, self.method, id(self.policy),
            self.policy.architecture_hash, parameter_hash(self.policy), self.updates,
            tuple((name, id(p)) for name, p in self.policy.named_parameters()),
            tuple((s.binding.fingerprint, s.observation.fingerprint, s.action_id) for s in samples),
            loss._version, float(loss.detach()))
        self._permits[id(loss)] = (loss, binding, samples)
        return loss


def _session(session, method):
    if type(session) is not ContactLearningSession:
        raise TypeError('Separate explicit public-contact TRAIN learning session required')
    session.require(method)
    return session.policy


def contact_imitation_loss(session, samples):
    policy = _session(session, 'IL'); samples = tuple(samples)
    if len(samples) != PROTOCOL['batch_size']:
        raise ValueError('The fixed IL pilot requires four TRAIN state samples per update')
    session.require_samples(samples)
    terms = []
    for sample in samples:
        logits, _ = policy(sample.observation, context=sample.binding.context)
        terms.append(-logits.log_softmax(-1)[sample.validate()])
    loss = torch.stack(terms).mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite admitted IL loss')
    session.register_loss(loss, samples)
    return loss, {'kind': 'public_contact_search_action_imitation_v1', 'loss': float(loss.detach()),
                  'loss_forward_calls': len(samples), 'supervised_actions': len(samples)}


def contact_reinforce_loss(session, episodes, *, behavior_parameter_hash):
    policy = _session(session, 'RL'); episodes = tuple(tuple(ep) for ep in episodes)
    if len(episodes) != PROTOCOL['batch_size'] or behavior_parameter_hash != parameter_hash(policy):
        raise ValueError('Four complete on-policy TRAIN episodes under unchanged weights required')
    actor_terms, values, entropies, returns, samples = [], [], [], [], []
    for episode in episodes:
        if (not 1 <= len(episode) <= 2 or not episode[-1].terminated
                or any(t.terminated for t in episode[:-1])):
            raise ValueError('Only complete one/two-step native contact episodes enter Monte Carlo loss')
        for transition in episode:
            if type(transition) is not ContactTransition: raise TypeError('Exact contact transitions required')
            transition.validate()
        binding = episode[0].sample.binding.fingerprint
        if any(t.sample.binding.fingerprint != binding for t in episode):
            raise ValueError('A trajectory cannot splice different family goals or contexts')
        if any(t.sample.observation.base.base.state_features[0] != i for i, t in enumerate(episode)):
            raise ValueError('Trajectory observations must begin at zero and advance one step')
        if any(t.sample.action_id == 'STOP' for t in episode[:-1]):
            raise ValueError('STOP cannot precede another transition')
        if any(t.sample.action_id == 'STOP' and t.reward != 0. for t in episode):
            raise ValueError('Public STOP has exactly zero incremental reward')
        if len(episode) == 1 and episode[0].sample.action_id != 'STOP':
            raise ValueError('One-step complete trajectory must explicitly STOP')
        session.require_samples(t.sample for t in episode)
        targets = []; total = 0.
        for t in reversed(episode):
            total = float(t.reward) + PROTOCOL['gamma'] * total; targets.append(total)
        targets.reverse(); returns.append(targets[0])
        actor, value_terms, entropy = [], [], []
        for step, (transition, target) in enumerate(zip(episode, targets)):
            sample = transition.sample; samples.append(sample)
            logits, value = policy(sample.observation, context=sample.binding.context)
            distribution = torch.distributions.Categorical(logits=logits)
            index = sample.validate()
            actor.append(-(PROTOCOL['gamma'] ** step) * distribution.log_prob(logits.new_tensor(index, dtype=torch.long))
                         * (value.new_tensor(target)-value.detach()))
            value_terms.append((value-target).square()); entropy.append(distribution.entropy())
        actor_terms.append(torch.stack(actor).sum())
        values.append(torch.stack(value_terms).mean()); entropies.append(torch.stack(entropy).mean())
    actor = torch.stack(actor_terms).mean(); value = torch.stack(values).mean(); entropy = torch.stack(entropies).mean()
    loss = actor + PROTOCOL['value_weight']*value - PROTOCOL['entropy_weight']*entropy
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite admitted policy-gradient loss')
    session.register_loss(loss, samples)
    return loss, {'kind': 'public_contact_on_policy_reinforce_v1', 'loss': float(loss.detach()),
        'policy_loss': float(actor.detach()), 'value_loss': float(value.detach()), 'entropy': float(entropy.detach()),
        'mean_return': float(np.mean(returns)), 'loss_forward_calls': len(samples), 'completed_episodes': len(episodes),
        'actor_reduction': 'mean_episodes_sum_discounted_score_terms',
        'regularizer_reduction': 'mean_episodes_mean_actions'}


def _optimizer_state_identity(optimizer, policy):
    names = {id(p): name for name, p in policy.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        state[names[id(parameter)]] = {key: array_digest(value.detach().cpu().numpy())
            if isinstance(value, torch.Tensor) else value for key, value in values.items()}
    return id(optimizer), semantic_digest(state)


def contact_gradient_step(session, optimizer, loss):
    policy = _session(session, session.method if type(session) is ContactLearningSession else '')
    permit = session._permits.pop(id(loss), None)
    if permit is None or permit[0] is not loss: raise ValueError('Loss is foreign, stale or already consumed')
    _, binding, samples = permit
    session.require_samples(samples)
    expected = (session.experiment.fingerprint, session.method, id(policy), policy.architecture_hash,
        parameter_hash(policy), session.updates, tuple((n, id(p)) for n, p in policy.named_parameters()),
        tuple((s.binding.fingerprint, s.observation.fingerprint, s.action_id) for s in samples),
        loss._version, float(loss.detach()))
    if binding != expected or not torch.isfinite(loss): raise ValueError('Loss/policy/context changed before update')
    members = [id(p) for group in optimizer.param_groups for p in group['params']]
    if (type(optimizer) is not torch.optim.Adam or len(members) != len(set(members))
            or set(members) != {id(p) for p in policy.parameters()}
            or any(group['lr'] != PROTOCOL['learning_rate'] or tuple(group['betas']) != tuple(PROTOCOL['betas'])
                or any(group[key] != PROTOCOL[key] for key in ('eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'fused'))
                for group in optimizer.param_groups)):
        raise ValueError('Fixed Adam optimizer must own exactly the admitted policy')
    if (session._optimizer_identity is None and optimizer.state) or (session._optimizer_identity is not None
            and _optimizer_state_identity(optimizer, policy) != session._optimizer_identity):
        raise ValueError('Adam state was not fresh initially or changed/replaced between admitted steps')
    before = parameter_hash(policy)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norms = {name: float(torch.stack([p.grad.square().sum() for p in getattr(policy, name).parameters()
             if p.grad is not None]).sum().sqrt()) if any(p.grad is not None for p in getattr(policy, name).parameters()) else 0.
             for name in ('encoder', 'actor', 'stop', 'critic')}
    total = float(nn.utils.clip_grad_norm_(policy.parameters(), PROTOCOL['max_gradient_norm'], error_if_nonfinite=True))
    optimizer.step(); session.updates += 1
    session._completed_updates = session.updates
    session._expected_parameter_hash = parameter_hash(policy)
    session._optimizer_identity = _optimizer_state_identity(optimizer, policy)
    session._permits.clear()  # Every loss made under the previous weights is stale.
    for sample in samples: session._used_bindings[sample.binding.fingerprint] = sample.binding.record()
    return {'version': VERSION, 'optimizer_steps': 1, 'cumulative_updates': session.updates,
        'gradient_norm_before_clip': total, 'module_gradient_norms_before_clip': norms,
        'initial_parameter_hash': before, 'updated_parameter_hash': parameter_hash(policy),
        'parameters_changed': before != parameter_hash(policy)}
