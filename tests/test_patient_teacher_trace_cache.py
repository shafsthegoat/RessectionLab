"""Generated controls: detached-cache ownership and unchanged IL arithmetic.

Synthetic DTO controls below do not certify native provenance. One separate
small native fixture exercises the real collector and independent replay.
No patient payload, checkpoint, teacher search or RL update is used.
"""
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.patient_planning_admission import PatientPlanningContext
from resectionlab.patient_planning_accumulation import PatientGradientAccumulator
from resectionlab.patient_planning_cohort_spec import (TRAIN, CACHED_TEACHERS, RECOLLECT_TEACHERS,
    BALANCED_TEACHER_CE, sequential_learning_protocol, validate_sequential_protocol)
from resectionlab.patient_teacher_trace_cache import PatientTeacherTraceCache
from resectionlab.spatial_policy import parameter_hash
from test_patient_cohort_sequential import config, generated_contexts, synthetic_trace, session_for, fake_runtime


def protocol_pair(*, balanced=True, updates=2, payload=64*1024**2):
    base, _ = config(updates=1, horizon=2)
    old = thaw_json(base)
    old['updates_per_method'] = updates
    if balanced:
        old['cohort_execution']['il_teacher_weighting'] = BALANCED_TEACHER_CE
    old = validate_sequential_protocol(freeze_json(old))
    cached = thaw_json(old)
    cached['cohort_execution'].update(teacher_observations=CACHED_TEACHERS,
        teacher_cache_payload_bytes=payload)
    return old, validate_sequential_protocol(freeze_json(cached))


def contexts_for(protocol):
    contexts, observations = generated_contexts(protocol)
    rebound = []
    for context in contexts:
        record = context.record()
        record['decision_model_hash'] = semantic_digest({'generated_dto_model': context.patient_group})
        rebound.append(PatientPlanningContext(freeze_json(record), semantic_digest(record)))
    return tuple(rebound), observations


def simulated_receipt(trace):
    """DTO-only replay stand-in; the native lifetime control uses actual replay."""
    context = trace.context.record()
    plan = freeze_json({'context_hash': trace.context.fingerprint, 'source_hash': context['source_hash'],
        'decision_model_hash': context['decision_model_hash'],
        'initial_observation_hash': trace.transitions[0].observation.fingerprint,
        'max_steps': trace.context.max_steps, 'actions': [r.action_id for r in trace.transitions],
        'history': trace.history, 'terminal_reason': 'STOP', 'parameter_hash': None,
        'architecture_hash': None, 'learning_updates': 0})
    return {'method': 'SEARCH', 'plan': plan, 'plan_seal': semantic_digest(plan),
        'replayed_history': trace.history,
        'independent_geometry': {'schema': 'native-spatial-independent-episode-v1',
            'accepted': True, 'complete_episode': True, 'source_hash': context['source_hash'],
            'decision_model_hash': context['decision_model_hash'],
            'committed_history_hash': semantic_digest(trace.history), 'geometry': {'feasible': True},
            'generated_DTO_control_only': True}}


def fill(protocol):
    contexts, observations = contexts_for(protocol)
    cache = PatientTeacherTraceCache(protocol)
    traces = tuple(synthetic_trace(c, o, n) for c, o, n in zip(contexts, observations, (1, 1, 1, 2)))
    for trace in traces:
        cache.add_replayed(trace, simulated_receipt(trace))
    return cache, contexts, observations, traces


@pytest.mark.parametrize('updates', [1, 8, 64])
def test_default_and_fixed64_records_remain_exact(updates):
    old, cached = protocol_pair(updates=updates)
    execution = old['cohort_execution']
    from resectionlab.native_proposals import NominalCavityProposalConfig
    explicit = sequential_learning_protocol(updates=updates, max_steps=2,
        search=execution['search'], proposal_config=NominalCavityProposalConfig(**thaw_json(execution['proposal_config'])),
        il_teacher_weighting=BALANCED_TEACHER_CE, teacher_observations=RECOLLECT_TEACHERS)
    assert semantic_digest(old) == semantic_digest(explicit)
    assert 'teacher_cache_payload_bytes' not in execution
    assert semantic_digest(old) != semantic_digest(cached)
    restored = thaw_json(cached)
    restored['cohort_execution']['teacher_observations'] = RECOLLECT_TEACHERS
    restored['cohort_execution'].pop('teacher_cache_payload_bytes')
    assert semantic_digest(restored) == semantic_digest(old)
    with pytest.raises(ValueError): PatientTeacherTraceCache(old)


@pytest.mark.parametrize('balanced', [False, True])
def test_two_updates_exact_gradients_parameters_and_Adam_state(balanced):
    torch.set_num_threads(1)
    old, cached_protocol = protocol_pair(balanced=balanced)
    old_contexts, old_observations = contexts_for(old)
    cache, contexts, observations, traces = fill(cached_protocol)
    baseline = session_for(old_contexts, old, 'IL')
    cached = session_for(contexts, cached_protocol, 'IL')
    assert parameter_hash(baseline.policy) == parameter_hash(cached.policy)
    fingerprints = tuple(t.transitions[0].observation.fingerprint for t in traces)
    for update in range(2):
        repeated = tuple(synthetic_trace(c, o, n) for c, o, n in zip(old_contexts, old_observations, (1, 1, 1, 2)))
        receipts = []
        for session, rows in ((baseline, repeated), (cached, traces)):
            pins = {t.context.fingerprint: {'steps': len(t.transitions), 'trace_seal': t.seal_hash,
                **({'stop_steps': sum(r.action_id == 'STOP' for r in t.transitions)} if balanced else {})} for t in rows}
            with PatientGradientAccumulator(session, teacher_pins=pins) as accumulation:
                for subject, trace in zip(TRAIN, rows):
                    selected = cache.trace_for_il(session, subject) if session is cached else trace
                    accumulation.add_trace(selected)
                receipts.append(accumulation.finish())
        a, b = receipts
        for key in ('loss', 'gradient_norm_before_clip', 'module_gradient_norms_before_clip',
                    'before_parameter_hash', 'after_parameter_hash', 'loss_forward_calls'):
            assert a[key] == b[key]
        assert a['loss_forward_calls'] == 5
        for pa, pb in zip(baseline.policy.parameters(), cached.policy.parameters()):
            assert torch.equal(pa, pb)
            assert (pa.grad is None and pb.grad is None) or torch.equal(pa.grad, pb.grad)
            sa, sb = baseline._optimizer.state.get(pa, {}), cached._optimizer.state.get(pb, {})
            assert sa.keys() == sb.keys()
            for key in sa: assert torch.equal(sa[key], sb[key])
        assert tuple(t.transitions[0].observation.fingerprint for t in traces) == fingerprints


def test_real_replayed_trace_releases_source_and_source_arrays(tmp_path):
    from test_patient_planning_learning import bound_fixture, trace as collect_fixture
    from resectionlab.patient_planning_preflight import _seal_and_replay
    task, context, _ = bound_fixture()
    _, protocol = protocol_pair(updates=1)
    record = context.record(); record['learning_protocol_hash'] = semantic_digest(protocol)
    context = PatientPlanningContext(freeze_json(record), semantic_digest(record))
    complete = collect_fixture(task, context, opening=True)
    sealed = _seal_and_replay(task, context, complete, method='SEARCH', policy=None,
        updates=0, output=tmp_path, guard=lambda: None)
    source = weakref.ref(task.case)
    arrays = [weakref.ref(a) for a in (task.case.structural_intensity,
        task.case.observed_support, task.case.nominal_target, task.case.reference_target)]
    cache = PatientTeacherTraceCache(protocol)
    cache.add_replayed(complete, sealed)
    observed = weakref.ref(complete.transitions[0].observation.image_channels)
    del task, complete, sealed
    gc.collect()
    assert source() is None and all(ref() is None for ref in arrays)
    assert observed() is not None
    assert cache.record()['traces'][0]['steps'] == 2
    del cache
    gc.collect()
    assert observed() is None


@pytest.mark.parametrize('field', ['history', 'observation', 'context', 'seal'])
def test_post_admission_tampering_is_refused(field):
    _, protocol = protocol_pair()
    cache, contexts, _, traces = fill(protocol)
    if field == 'history': object.__setattr__(traces[0], 'history', ())
    elif field == 'observation':
        obs = traces[0].transitions[0].observation
        object.__setattr__(obs, 'image_channels', obs.image_channels.copy())
    elif field == 'context': object.__setattr__(contexts[0], '_seal', 'sha256:'+'0'*64)
    else: object.__setattr__(traces[0], 'seal_hash', 'sha256:'+'0'*64)
    with pytest.raises(ValueError): cache.record()


@pytest.mark.parametrize('change', ['geometry', 'history', 'plan', 'behavior', 'order'])
def test_no_unverified_or_on_policy_teacher_can_enter(change):
    _, protocol = protocol_pair()
    contexts, observations = contexts_for(protocol)
    cache = PatientTeacherTraceCache(protocol)
    idx = 1 if change == 'order' else 0
    trace = synthetic_trace(contexts[idx], observations[idx], 1,
        'sha256:'+'0'*64 if change == 'behavior' else None)
    receipt = thaw_json(freeze_json(simulated_receipt(trace)))
    if change == 'geometry': receipt['independent_geometry']['geometry']['feasible'] = False
    elif change == 'history': receipt['replayed_history'][0]['reward'] = 1.
    elif change == 'plan': receipt['plan']['actions'] = ['other']
    with pytest.raises(ValueError): cache.add_replayed(trace, receipt)
    assert cache.record()['traces'] == []


def test_no_RL_reuse_and_atomic_payload_budget_refusal():
    _, protocol = protocol_pair()
    cache, contexts, _, _ = fill(protocol)
    rl = session_for(contexts, protocol, 'RL')
    with pytest.raises(ValueError): cache.trace_for_il(rl, TRAIN[0])
    _, tiny = protocol_pair(payload=1)
    contexts, observations = contexts_for(tiny)
    trace = synthetic_trace(contexts[0], observations[0], 1)
    limited = PatientTeacherTraceCache(tiny)
    with pytest.raises(MemoryError): limited.add_replayed(trace, simulated_receipt(trace))
    assert limited.record()['traces'] == []


def test_cached_rows_are_reused_without_copy_or_native_callback(monkeypatch):
    from resectionlab.native_resection import NativeResectionEngine
    _, protocol = protocol_pair()
    cache, contexts, _, traces = fill(protocol)
    session = session_for(contexts, protocol, 'IL')
    def refuse(*a, **k): raise AssertionError('Cache reuse called native geometry')
    monkeypatch.setattr(NativeResectionEngine, 'preview_stroke', refuse)
    for subject, original in zip(TRAIN, traces):
        actual = cache.trace_for_il(session, subject)
        assert actual is original
        assert actual.transitions[0].observation.image_channels is original.transitions[0].observation.image_channels


def test_runner_replays_once_for_IL_but_collects_fresh_RL(tmp_path, fake_runtime, monkeypatch):
    """Fake runtime isolates scheduling; other controls exercise real DTO/native paths."""
    import resectionlab.patient_teacher_trace_cache as module
    from resectionlab import patient_planning_cohort_sequential as runner
    protocol, limits, factories, events, live, _ = fake_runtime
    changed = thaw_json(protocol)
    changed['cohort_execution'].update(teacher_observations=CACHED_TEACHERS,
        teacher_cache_payload_bytes=64*1024**2)
    cache_lifetimes = []
    class Cache:
        def __init__(self, protocol):
            self.traces = {}; cache_lifetimes.append(weakref.ref(self))
        def add_replayed(self, trace, sealed):
            self.traces[trace.context.subject] = trace
            events.append(('cache_fill', trace.context.subject))
        def trace_for_il(self, session, subject):
            assert session.method == 'IL' and len(live) == 0
            events.append(('cache_use', subject))
            return self.traces[subject]
        def record(self): return {'mode': CACHED_TEACHERS, 'complete': len(self.traces) == 4}
    monkeypatch.setattr(module, 'PatientTeacherTraceCache', Cache)
    collect = runner.preflight._collect
    def fresh_rl(base, context, **kwargs):
        if kwargs.get('policy') is not None:
            assert all(ref() is None for ref in cache_lifetimes)
        return collect(base, context, **kwargs)
    monkeypatch.setattr(runner.preflight, '_collect', fresh_rl)
    result = runner.run_train_cohort_sequential(factories, learning_protocol=changed,
        limits=limits, output=tmp_path/'run')
    assert [e[1] for e in events if e[0] == 'cache_fill'] == list(TRAIN)
    assert [e[1] for e in events if e[0] == 'cache_use'] == list(TRAIN)
    assert [e[1] for e in events if e[0] == 'factory'] == list(TRAIN)*4
    assert [e for e in events if e[0] == 'step'] == [('step', 'IL'), ('step', 'RL')]
    assert result['selection_readiness']['ready'] is True and len(live) == 0
