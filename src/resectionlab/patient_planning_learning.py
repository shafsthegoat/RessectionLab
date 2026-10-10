"""Context-bound patient IL/RL, reused by a one-update integration preflight.

The network and loss equations match the existing spatial learner. Admission is
separate from its generated-only API. A complete native trace supplies labels and
public nominal rewards; no private evaluation payload is accepted here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .core import freeze_json, semantic_digest, thaw_json
from .patient_planning_admission import PatientPlanningContext
from .spatial_policy import SpatialPolicy, SpatialPolicyConfig, SpatialTransition, parameter_hash


PREFLIGHT_PROTOCOL = freeze_json({
    "version": "qualified-patient-learning-preflight-v1", "seed": 20261010,
    "methods": ["IL", "RL"], "updates_per_method": 1, "rl_episodes": 1,
    "learning_rate": .001, "betas": [.9, .999], "eps": 1e-8,
    "gamma": 1., "entropy_weight": .01, "value_weight": .5,
    "max_gradient_norm": 5., "architecture": {
        "encoder_channels": [8, 16], "hidden_features": 32,
        "ray_samples": 5, "physical_reference_mm": 10., "critic_candidate_context": True},
    "initialization": "same_fresh_spatial_parameters_no_checkpoint_load",
    "scope": "one_qualified_TRAIN_patient_software_and_scale_preflight",
    "population_training": False, "patient_adaptation": False,
    "private_training_reward": False, "clinical_claim": False,
})
_TRACE = object()
_INITIALIZATIONS = WeakKeyDictionary()


def _context(context):
    if type(context) is not PatientPlanningContext:
        raise TypeError("Exact admitted patient planning context required")
    context.require_training()
    return context.fingerprint


def common_patient_policies(contexts, protocol):
    """Construct once and copy exact tensors; do not reinterpret old weights."""
    contexts = tuple(contexts); protocol = freeze_json(protocol)
    if not contexts: raise ValueError("At least one admitted TRAIN context required")
    sequential = 'cohort_execution' in protocol
    if sequential:
        from .patient_planning_cohort_spec import validate_sequential_protocol
        validate_sequential_protocol(protocol)
    variant=protocol.get('public_target_context_variant')
    if variant is not None:
        from .public_target_context import VERSION
        if variant!=VERSION:raise ValueError('Unknown opt-in patient public target variant')
    if (set(protocol) != set(PREFLIGHT_PROTOCOL)|({'public_target_context_variant'} if variant is not None else set())|({'cohort_execution'} if sequential else set()) or protocol["methods"] != PREFLIGHT_PROTOCOL["methods"]
            or type(protocol["seed"]) is not int or protocol["seed"] < 0
            or type(protocol["updates_per_method"]) is not int or protocol["updates_per_method"] < 1
            or protocol["architecture"] != PREFLIGHT_PROTOCOL["architecture"]
            or any(protocol[key] != PREFLIGHT_PROTOCOL[key] for key in
                   ("learning_rate", "betas", "eps", "gamma", "entropy_weight", "value_weight",
                    "max_gradient_norm", "initialization", "private_training_reward", "clinical_claim"))
            or protocol["patient_adaptation"] is not False):
        raise ValueError("Frozen public-reward patient learning protocol required")
    for context in contexts:
        _context(context)
        if context.record()["learning_protocol_hash"] != semantic_digest(protocol):
            raise ValueError("Learning protocol differs from admitted patient context")
        if sequential and context.record().get('occupancy_condition') != protocol['cohort_execution'].get('occupancy_condition'):
            raise ValueError('Learning occupancy condition differs from the admitted source')
        if context.record().get('public_target_context_variant')!=variant:
            raise ValueError('Patient observation and policy context variants differ')
        if len(protocol["methods"])*protocol["updates_per_method"] > context.record()["max_optimizer_updates"]:
            raise ValueError("Learning updates exceed aggregate admitted context budget")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol["seed"])
        il = SpatialPolicy(SpatialPolicyConfig(**thaw_json(protocol["architecture"])),
            **({} if variant is None else {'public_target_context_variant':variant}))
        rl = copy.deepcopy(il)
    if parameter_hash(il) != parameter_hash(rl):
        raise RuntimeError("Common fresh initialization differs")
    for policy in (il, rl):
        _INITIALIZATIONS[policy] = (parameter_hash(policy), policy.architecture_hash,
            semantic_digest(protocol), tuple(c.fingerprint for c in contexts))
    return il, rl


def _index(observation, action_id):
    observation.assert_intact()
    if action_id not in observation.action_ids:
        raise ValueError("Action absent from its actual observation")
    index = observation.action_ids.index(action_id)
    if not observation.action_mask[index]:
        raise ValueError("Chosen action is masked")
    return index


@dataclass(frozen=True)
class PatientTrainingTrace:
    context: PatientPlanningContext
    transitions: tuple
    history: tuple
    behavior_parameter_hash: str | None
    seal_hash: str
    _capability: object

    def require(self):
        _context(self.context)
        if self._capability is not _TRACE:
            raise ValueError("Only the owned native collector may admit a trace")
        rows = self.transitions
        if (not rows or len(rows) > self.context.max_steps
                or len(rows) != len(self.history) or not rows[-1].terminated
                or any(row.terminated for row in rows[:-1])):
            raise ValueError("A complete native STOP-or-horizon trajectory is required")
        self.context.require_observations(row.observation for row in rows)
        for step, (row, native) in enumerate(zip(rows, self.history)):
            if (type(row) is not SpatialTransition or row.observation.state_features[0] != step
                    or not math.isfinite(row.reward) or row.action_id != native["action_id"]
                    or row.reward != native["reward"]):
                raise ValueError("Observation/action/reward differs from its complete native history")
            _index(row.observation, row.action_id)
        stops = [i for i, row in enumerate(rows) if row.action_id == "STOP"]
        if ((stops and stops != [len(rows)-1])
                or (not stops and len(rows) != self.context.max_steps)):
            raise ValueError("Incomplete or improperly terminated trajectory")
        if any(row.action_id == "STOP" and row.reward != 0. for row in rows):
            raise ValueError("STOP must retain the unchanged zero nominal increment")
        expected = semantic_digest({"context": self.context.fingerprint,
            "observations": [row.observation.fingerprint for row in rows],
            "history": self.history, "behavior_parameter_hash": self.behavior_parameter_hash})
        if expected != self.seal_hash:
            raise ValueError("Trace identity changed")
        return self


def admit_collected_trace(context, transitions, task, *, behavior_parameter_hash=None):
    """Owned-run provenance, not a security boundary against hostile Python."""
    _context(context); context.require_task(task)
    metrics = task.metrics()
    if not task.terminated or metrics.get("planning_estimator_only") is not True:
        raise ValueError("TRAIN trace must terminate on the admitted public nominal world")
    rows = tuple(transitions); history = freeze_json(metrics["history"])
    seal = semantic_digest({"context": context.fingerprint,
        "observations": [row.observation.fingerprint for row in rows],
        "history": history, "behavior_parameter_hash": behavior_parameter_hash})
    return PatientTrainingTrace(context, rows, history, behavior_parameter_hash, seal, _TRACE).require()


class PatientTrainSession:
    """Frozen TRAIN contexts and update budget; preflight imposes one in its runner."""
    def __init__(self, contexts, method, policy, *, initial_parameter_hash, protocol):
        self.contexts = tuple(contexts); self.protocol = freeze_json(protocol)
        identities = tuple((id(c), _context(c)) for c in self.contexts)
        if len({c.patient_group for c in self.contexts}) != len(self.contexts):
            raise ValueError("One context per canonical patient in a learning session")
        if type(policy) is not SpatialPolicy or method not in ("IL", "RL"):
            raise ValueError("Existing spatial architecture and IL/RL method required")
        expected = _INITIALIZATIONS.pop(policy, None)
        if expected != (initial_parameter_hash, policy.architecture_hash,
                semantic_digest(self.protocol), tuple(c.fingerprint for c in self.contexts)) or parameter_hash(policy) != initial_parameter_hash:
            raise ValueError("Exact fresh common initialization required")
        self.method, self.policy = method, policy
        self.initial_parameter_hash = initial_parameter_hash
        self.updates = 0; self._completed_updates = 0; self._attempted = False; self._permit = None
        self._expected_parameter_hash = initial_parameter_hash
        self._identity = (identities, semantic_digest(self.protocol), method, id(policy), policy.architecture_hash,
            tuple((name, id(p)) for name, p in policy.named_parameters()))
        self._optimizer = torch.optim.Adam(policy.parameters(), lr=.001, betas=(.9, .999),
            eps=1e-8, weight_decay=0., amsgrad=False, maximize=False, foreach=False, fused=False)

    def require(self, method):
        identities = tuple((id(c), _context(c)) for c in self.contexts)
        current = (identities, semantic_digest(self.protocol), self.method, id(self.policy), self.policy.architecture_hash,
            tuple((name, id(p)) for name, p in self.policy.named_parameters()))
        if (current != self._identity or method != self.method or self._attempted
                or self.updates != self._completed_updates
                or not 0 <= self.updates < self.protocol["updates_per_method"]
                or parameter_hash(self.policy) != self._expected_parameter_hash):
            raise ValueError("Changed session/weights or frozen update budget exhausted")

    def require_trace(self, trace):
        trace.require()
        if trace.context.fingerprint not in {c.fingerprint for c in self.contexts}:
            raise ValueError("Foreign TRAIN patient/world")

    def register(self, loss, traces):
        self.require(self.method)
        for trace in traces: self.require_trace(trace)
        if self._permit is not None:
            raise ValueError("Foreign trace or unconsumed admitted loss")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite patient preflight loss")
        self._permit = (loss, loss._version, tuple(traces), tuple(t.seal_hash for t in traces), parameter_hash(self.policy))


@dataclass(frozen=True)
class PatientImitationSample:
    trace: PatientTrainingTrace
    step: int

    def transition(self):
        self.trace.require()
        if type(self.step) is not int or not 0 <= self.step < len(self.trace.transitions):
            raise ValueError("Teacher sample index outside its complete trace")
        return self.trace.transitions[self.step]


def patient_imitation_group_weights(protocol, *, stop_count, motion_count):
    """Declared label groups only; no action IDs beyond reserved STOP or rewards.

The opt-in objective is .5*mean(STOP CE)+.5*mean(motion CE). Both
groups must exist before loss/gradient work. No logits or inference masks change.
"""
    from .patient_planning_cohort_spec import BALANCED_TEACHER_CE
    if (type(stop_count) is not int or type(motion_count) is not int
            or stop_count < 0 or motion_count < 0 or stop_count+motion_count == 0):
        raise ValueError('Exact nonnegative teacher group counts required')
    variant = protocol.get('cohort_execution', {}).get('il_teacher_weighting')
    if variant is None:
        return {'STOP': 1./(stop_count+motion_count), 'motion': 1./(stop_count+motion_count)}
    if variant != BALANCED_TEACHER_CE or stop_count == 0 or motion_count == 0:
        raise ValueError('Balanced teacher CE requires both STOP and motion groups')
    return {'STOP': .5/stop_count, 'motion': .5/motion_count}


def patient_imitation_loss(session, samples):
    if type(session) is not PatientTrainSession:
        raise TypeError("Exact patient learning session required")
    session.require("IL"); samples = tuple(samples)
    if not samples: raise ValueError("Empty patient imitation batch")
    balanced = session.protocol.get('cohort_execution', {}).get('il_teacher_weighting') is not None
    if balanced:
        labels = []
        for sample in samples:
            if type(sample) is not PatientImitationSample: raise TypeError("Trace-bound teacher sample required")
            session.require_trace(sample.trace)
            labels.append(sample.transition().action_id == 'STOP')
        stop_count = sum(labels); motion_count = len(labels)-stop_count
        weights = patient_imitation_group_weights(session.protocol, stop_count=stop_count, motion_count=motion_count)
    terms = []; traces = []
    for sample in samples:
        if type(sample) is not PatientImitationSample: raise TypeError("Trace-bound teacher sample required")
        session.require_trace(sample.trace); row = sample.transition(); traces.append(sample.trace)
        logits, _ = session.policy(row.observation)
        terms.append(-logits.log_softmax(-1)[_index(row.observation, row.action_id)])
    loss = (torch.stack([term*weights['STOP' if stop else 'motion'] for term,stop in zip(terms,labels)]).sum()
            if balanced else torch.stack(terms).mean())
    session.register(loss, traces)
    return loss, {"objective": "balanced_STOP_motion_CE_v1" if balanced else "unchanged_mean_action_categorical_CE", "loss": float(loss.detach()),
        "loss_forward_calls": len(terms), "teacher_actions": len(terms),
        **({'teacher_group_counts': {'STOP': stop_count, 'motion': motion_count},
            'teacher_group_weights': weights} if balanced else {}),
        "sample_bindings": [{"patient_group": s.trace.context.patient_group,
                             "trace_seal": s.trace.seal_hash, "step": s.step} for s in samples]}


def patient_reinforce_loss(session, traces):
    if type(session) is not PatientTrainSession:
        raise TypeError("Exact patient learning session required")
    session.require("RL"); traces = tuple(traces)
    if not traces: raise ValueError("Empty patient on-policy batch")
    actors = []; values = []; entropies = []; returns = []
    for trace in traces:
        if type(trace) is not PatientTrainingTrace: raise TypeError("Collected complete native trace required")
        session.require_trace(trace)
        if trace.behavior_parameter_hash != parameter_hash(session.policy):
            raise ValueError("Stale on-policy trajectory")
        targets = []; total = 0.
        for row in reversed(trace.transitions):
            total = row.reward + session.protocol["gamma"] * total; targets.append(total)
        targets.reverse(); returns.append(targets[0]); actor_terms = []; ep_values = []; ep_entropies = []
        for step, (row, target) in enumerate(zip(trace.transitions, targets)):
            logits, value = session.policy(row.observation)
            distribution = torch.distributions.Categorical(logits=logits)
            index = _index(row.observation, row.action_id)
            actor_terms.append(-(session.protocol["gamma"] ** step) * distribution.log_prob(
                logits.new_tensor(index, dtype=torch.long)) * (value.new_tensor(target)-value.detach()))
            ep_values.append((value-target).square()); ep_entropies.append(distribution.entropy())
        actors.append(torch.stack(actor_terms).sum()); values.append(torch.stack(ep_values).mean())
        entropies.append(torch.stack(ep_entropies).mean())
    actor = torch.stack(actors).mean(); value = torch.stack(values).mean(); entropy = torch.stack(entropies).mean()
    loss = actor + session.protocol["value_weight"]*value - session.protocol["entropy_weight"]*entropy
    session.register(loss, traces)
    return loss, {"objective": "unchanged_complete_episode_REINFORCE", "loss": float(loss.detach()),
        "actor_loss": float(actor.detach()), "value_loss": float(value.detach()),
        "entropy": float(entropy.detach()), "complete_episodes": len(traces), "returns": returns,
        "actor_reduction": "mean_episodes_sum_score_terms",
        "loss_forward_calls": sum(len(t.transitions) for t in traces),
        "trace_seals": [t.seal_hash for t in traces]}


def patient_gradient_step(session, loss):
    if type(session) is not PatientTrainSession:
        raise TypeError("Exact patient preflight session required")
    session.require(session.method)
    permit = session._permit
    if (permit is None or permit[0] is not loss or loss._version != permit[1]
            or permit[3] != tuple(t.require().seal_hash for t in permit[2])
            or permit[4] != parameter_hash(session.policy) or not torch.isfinite(loss)):
        raise ValueError("Foreign, stale, changed or consumed loss")
    session._permit = None; session._attempted = True
    optimizer = session._optimizer; policy = session.policy
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norms = {}
    for name in ("encoder", "actor", "stop", "critic"):
        squares = [p.grad.detach().square().sum() for p in getattr(policy, name).parameters() if p.grad is not None]
        norms[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.
    total = float(nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True))
    optimizer.step(); after = parameter_hash(policy)
    session.updates += 1; session._completed_updates += 1
    session._expected_parameter_hash = after; session._attempted = False
    return {"version": session.protocol["version"], "method": session.method,
        "optimizer_updates": 1, "before_parameter_hash": permit[4], "after_parameter_hash": after,
        "parameters_changed": after != permit[4], "gradient_norm_before_clip": total,
        "module_gradient_norms_before_clip": norms, "context_hashes": [c.fingerprint for c in session.contexts],
        "trace_seals": permit[3], "completed_updates": session.updates,
        "population_training": session.protocol["population_training"], "patient_adaptation": False}
