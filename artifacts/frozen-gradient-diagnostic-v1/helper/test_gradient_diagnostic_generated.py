"""Unrun generated helper controls; no acquired files, native calls or optimizer.

The exact frozen helper is exercised with a four-parameter deterministic policy
and explicit cache/trace boundary doubles. The production corpus schema/join,
motion loss, parameter hash and torch autograd remain real. These controls test
helper arithmetic/lifetime, not the owned runner's acquired-data admission.
Only this helper's source is read; no saved study files or checkpoints are used.
"""
from dataclasses import dataclass
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.public_motion_ranking import (
    OBJECTIVE_SOURCE, SCOPE, STEPS, SUBJECTS, VERSION,
    PublicMotionRankingCorpus, motion_ranking_loss,
)
from resectionlab.spatial_policy import parameter_hash
import resectionlab.spatial_policy as policy_module
import resectionlab.patient_teacher_trace_cache as cache_module


HELPER_SHA = 'cc183fcd3e76667ab81a2b117ee3d058269a6831f0fd33318504c81e4d7140dd'
H = 'sha256:' + 'a' * 64
IDS = ('STOP', 'good', 'bad', 'masked')
MASK = (True, True, True, False)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@dataclass(frozen=True)
class GeneratedObservation:
    fingerprint: str
    action_ids: tuple = IDS
    action_mask: tuple = MASK

    def assert_intact(self):
        assert self.action_ids == IDS and self.action_mask == MASK


class GeneratedPolicy(torch.nn.Module):
    """Wrong motion winner; the gate and ranking push its margin oppositely."""
    architecture_hash = semantic_digest('four-parameter-generated-policy')

    def __init__(self):
        super().__init__()
        for name, value in (('encoder', 1.), ('actor', .4), ('stop', -.2), ('critic', .1)):
            layer = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                layer.weight.fill_(value)
            setattr(self, name, layer)
        self.forward_calls = 0

    def forward(self, observation):
        observation.assert_intact()
        self.forward_calls += 1
        h = self.encoder(self.encoder.weight.new_tensor([1.]))
        a = self.actor(h).reshape(())
        stop = self.stop(h).reshape(())
        # One masked action must never contribute to loss or margin selection.
        return torch.stack((stop, -a, a, a.new_tensor(-float('inf')))), self.critic(h).reshape(())


class GeneratedTrace:
    def __init__(self, context, transitions):
        self.context = context
        self.transitions = tuple(transitions)
        self.behavior_parameter_hash = None
        self.history = tuple({'source_state_hash': H} for _ in transitions)
        self.seal_hash = semantic_digest([context.fingerprint,
            [row.observation.fingerprint for row in transitions]])

    def require(self):
        return self


class GeneratedCache:
    """Declared boundary double; native replay qualification is not simulated."""
    def __init__(self, protocol, traces):
        self.protocol = protocol
        self.traces = traces

    def record(self):
        pins = [{'subject': s, 'trace_seal': t.seal_hash,
                 'context_hash': t.context.fingerprint, 'steps': len(t.transitions),
                 'stop_steps': sum(row.action_id == 'STOP' for row in t.transitions)}
                for s, t in self.traces.items()]
        return {'complete': True, 'learning_protocol_hash': semantic_digest(self.protocol),
                'cache_seal': semantic_digest(pins), 'traces': pins}


def build_fixture(monkeypatch):
    path = Path(__file__).with_name('gradient_diagnostic.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('_frozen_gradient_generated_control', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Scope these replacements to the helper's exact-type boundary. No policy
    # math/loss/autograd or corpus validation method is patched.
    monkeypatch.setattr(policy_module, 'SpatialPolicy', GeneratedPolicy)
    monkeypatch.setattr(cache_module, 'PatientTeacherTraceCache', GeneratedCache)
    policy = GeneratedPolicy().eval()
    traces, cases = {}, []
    for subject, count in zip(SUBJECTS, STEPS):
        record = dict(subject=subject, role='TRAIN', source_hash=H, decision_model_hash=H)
        context = SimpleNamespace(patient_group=subject, fingerprint=semantic_digest(record),
                                  record=lambda r=record: copy.deepcopy(r))
        transitions, decisions = [], []
        for step in range(count):
            stop = step == count - 1
            observation = GeneratedObservation(semantic_digest(['generated', subject, step]))
            action = 'STOP' if stop else 'good'
            rewards = (0., -1., -2., None) if stop else (0., 3., 1., None)
            transitions.append(SimpleNamespace(observation=observation, action_id=action,
                                                reward=rewards[IDS.index(action)]))
            decisions.append(dict(step=step, observation_hash=observation.fingerprint,
                source_state_hash=H, action_ids=list(IDS), action_mask=list(MASK),
                teacher_action=action, rewards=list(rewards)))
        traces[subject] = GeneratedTrace(context, transitions)
        cases.append({**record, 'input_sha256': {key: '1' * 64 for key in
            ('scores', 'trace', 'plan', 'replay')}, 'decisions': decisions})
    record = dict(version=VERSION, scope=SCOPE, objective_source=OBJECTIVE_SOURCE,
                  private_reference_used=False, subjects=cases)
    corpus = PublicMotionRankingCorpus.admit(record, expected_hash=semantic_digest(record))
    protocol = freeze_json({'cohort_execution': {'il_motion_supervision':
                                                {'corpus_hash': corpus.fingerprint}}})
    cache = GeneratedCache(protocol, traces)
    metadata = dict(method='IL', completed_updates=64, parameter_hash=parameter_hash(policy),
        architecture_hash=policy.architecture_hash, learning_protocol_hash=semantic_digest(protocol),
        contexts=freeze_json([{'patient_group': t.context.patient_group,
                              'context_hash': t.context.fingerprint} for t in traces.values()]))
    readouts = {}
    with torch.no_grad():
        for subject, trace in traces.items():
            for step, transition in enumerate(trace.transitions):
                logits, _ = policy(transition.observation)
                readouts[f'{subject}:{step}'] = dict(subject=subject, step=step,
                    observation_hash=transition.observation.fingerprint,
                    action_ids=list(IDS), action_mask=list(MASK), teacher_action=transition.action_id,
                    scores={'logits': [float(logits[i]) if m else None for i, m in enumerate(MASK)]})
    policy.forward_calls = 0
    # Success and failure must restore a genuinely mixed original flag pattern.
    policy.actor.weight.requires_grad_(False)
    policy.stop.weight.requires_grad_(False)
    fixture = SimpleNamespace(helper=helper, policy=policy, traces=traces, corpus=corpus,
        cache=cache, metadata=metadata, readouts=readouts, emitted=[])
    fixture.flags = tuple(p.requires_grad for p in policy.parameters())
    fixture.values = {n: p.detach().clone() for n, p in policy.named_parameters()}
    return fixture


def call_helper(f, *, guard=lambda: None, emit=None):
    return f.helper.inspect_frozen_gradients(f.policy, checkpoint_metadata=f.metadata,
        expected_parameter_hash=f.metadata['parameter_hash'], cache=f.cache, traces=f.traces,
        ranking=f.corpus, saved_readouts=f.readouts, guard=guard,
        emit=emit or (lambda name, record: f.emitted.append((name, record))))


def unchanged(f):
    assert tuple(p.requires_grad for p in f.policy.parameters()) == f.flags
    assert parameter_hash(f.policy) == f.metadata['parameter_hash']
    assert all(torch.equal(p.detach(), f.values[n]) for n, p in f.policy.named_parameters())
    assert all(p.grad is None for p in f.policy.parameters())
    assert not f.policy.training


def independent_production_vectors(f):
    """One combined production objective plus analytic two-motion components."""
    named = [(n, p) for n, p in f.policy.named_parameters() if not n.startswith('critic.')]
    parameters = tuple(p for _, p in named)
    flags = tuple(p.requires_grad for p in f.policy.parameters())
    try:
        for _, p in named:
            p.requires_grad_(True)
        objectives = {key: [] for key in ('full', 'gate', 'ranking', 'STOP')}
        for subject, trace in f.traces.items():
            for transition, label in zip(trace.transitions, f.corpus.require_trace(trace)):
                z, _ = f.policy(transition.observation)
                weight = .5 / (4 if transition.action_id == 'STOP' else 25)
                objectives['full'].append(weight * motion_ranking_loss(z,
                    action_ids=IDS, action_mask=MASK, rewards=label['rewards'],
                    teacher_action=transition.action_id))
                if transition.action_id == 'STOP':
                    objectives['STOP'].append(weight * (-z.log_softmax(0)[0]))
                else:
                    objectives['gate'].append(weight * (torch.logsumexp(z[:3], 0) - torch.logsumexp(z[1:3], 0)))
                    # Exactly one strict pair: normalization gives unit weight.
                    objectives['ranking'].append(weight * torch.nn.functional.softplus(z[2] - z[1]))
        vectors = {}
        for key, parts in objectives.items():
            grads = torch.autograd.grad(torch.stack(parts).sum(), parameters,
                                        allow_unused=True, retain_graph=True)
            vectors[key] = torch.cat([torch.zeros(p.numel(), dtype=torch.float64) if g is None
                else g.detach().reshape(-1).double() for p, g in zip(parameters, grads)])
        return vectors
    finally:
        for p, flag in zip(f.policy.parameters(), flags):
            p.requires_grad_(flag)


def test_exact_helper_counts_production_decomposition_and_margin_signs(monkeypatch):
    f = build_fixture(monkeypatch)
    expected = independent_production_vectors(f)
    f.policy.forward_calls = 0
    actual_calls = []
    original = torch.autograd.grad
    def counted(*args, **kwargs):
        actual_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, 'grad', counted)
    report = call_helper(f)
    assert f.policy.forward_calls == 29 and len(actual_calls) == 79
    assert report['counts'] == dict(policy_forwards=29, autograd_grad_calls=79, optimizer_updates=0)
    assert len(report['rows']) == 29 and len(f.emitted) == 1
    assert f.emitted[0] == ('gradient-alignment.json', report)
    close = dict(rel=2e-6, abs=2e-7)  # float32 aggregate order differs by design
    assert report['aggregate_total']['norm'] == pytest.approx(float(expected['full'].norm()), **close)
    combined = expected['gate'] + expected['ranking'] + expected['STOP']
    assert torch.allclose(combined, expected['full'], rtol=2e-6, atol=2e-7)
    for key in ('ranking', 'gate', 'STOP'):
        assert report['aggregate_components'][key]['norm'] == pytest.approx(float(expected[key].norm()), **close)
        for other in ('ranking', 'gate', 'STOP'):
            assert report['component_cross_dots'][key][other] == pytest.approx(
                float(torch.dot(expected[key], expected[other])), **close)
    motion_rows = [r for r in report['rows'] if r['teacher_action'] != 'STOP']
    assert len(motion_rows) == 25
    for row in motion_rows:
        assert row['chosen_motion'] == 'bad' and row['public_nominal_regret'] == 2.
        assert row['teacher_minus_chosen_logit'] < 0.
        derivatives = row['margin_directional_derivative_under_negative_raw_gradient']
        assert derivatives['same_state_ranking'] > 0.
        assert derivatives['ranking'] > 0. and derivatives['gate'] < 0.
        assert derivatives['full_corpus'] == pytest.approx(
            sum(derivatives[k] for k in ('ranking', 'gate', 'STOP')), **close)
    assert len([r for r in report['rows'] if r['teacher_action'] == 'STOP']) == 4
    unchanged(f)


@pytest.mark.parametrize('failure', ['readout', 'guard', 'emit'])
def test_failure_restores_flags_without_clearing_or_changing_state(monkeypatch, failure):
    f = build_fixture(monkeypatch)
    if failure == 'readout':
        # First motion follows an already evaluated STOP, so restoration is
        # tested after at least one completed VJP, not only at the entry guard.
        f.readouts[f'{SUBJECTS[1]}:0']['scores']['logits'][1] += .125
    def guard():
        if failure == 'guard' and f.policy.forward_calls >= 2:
            raise RuntimeError('generated guard failure')
    def emit(name, record):
        if failure == 'emit':
            raise RuntimeError('generated output failure')
        f.emitted.append((name, record))
    expected = ValueError if failure == 'readout' else RuntimeError
    with pytest.raises(expected, match='readout|generated'):
        call_helper(f, guard=guard, emit=emit)
    assert f.policy.forward_calls >= 2
    assert f.emitted == []
    unchanged(f)


def test_dirty_gradient_buffer_is_refused_and_never_silently_cleared(monkeypatch):
    f = build_fixture(monkeypatch)
    dirty = torch.full_like(f.policy.critic.weight, .125)
    f.policy.critic.weight.grad = dirty
    with pytest.raises(ValueError, match='gradient buffers'):
        call_helper(f)
    assert f.policy.forward_calls == 0 and f.policy.critic.weight.grad is dirty
    assert torch.equal(dirty, torch.full_like(dirty, .125))
    assert tuple(p.requires_grad for p in f.policy.parameters()) == f.flags
    assert parameter_hash(f.policy) == f.metadata['parameter_hash']


def test_no_grad_caller_refuses_and_restores_original_flags(monkeypatch):
    f = build_fixture(monkeypatch)
    with torch.no_grad(), pytest.raises(RuntimeError, match='does not require grad|grad_fn'):
        call_helper(f)
    assert f.policy.forward_calls == 1 and f.emitted == []
    unchanged(f)
