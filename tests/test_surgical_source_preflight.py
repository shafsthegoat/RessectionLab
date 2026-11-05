"""Source inspection is diagnostic, detached and cancellation preserving."""
import copy

import pytest

from resectionlab import surgical_source_preflight as preflight
from resectionlab import surgical_task_admission as admission
from resectionlab.desktop_bridge import _Request, BridgeError
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_interactive_planning import interactive_planning_availability
from test_surgical_task_admission import prepared


def statuses(value): return {r['id']: r['status'] for r in value['checks']}


def test_intact_source_can_pass_bytes_and_replay_without_becoming_a_planning_task(prepared):
    _, original, _, collection = prepared[0]
    engine = SurgicalEngine(original.scene); before = engine.snapshot()
    value = preflight.source_preflight(engine, native_collection=collection.root, verify_replay=True)
    state = statuses(value)
    assert all(state[k] == 'passed' for k in ('source-role', 'source-bytes', 'native-replay', 'withdrawn-baseline', 'static-baseline'))
    assert all(state[k] == 'required' for k in ('model-versions', 'executed-exposure', 'declared-task'))
    assert state['evidence-review'] == 'requires-research-opt-in'
    assert not value['planningAdmitted'] and admission.admission(engine) is None
    assert not interactive_planning_availability(engine)['available']
    assert engine.snapshot() == before and 'functional' in value['sourceEvidence']['unknownEvidence']
    value['sourceEvidence']['anatomy']['brain']['reviewStatus'] = 'reviewed'
    assert admission.source_evidence(engine.scene)['anatomy']['brain']['reviewStatus'] == 'unreviewed'


def test_metadata_inspection_never_reads_source_arrays_or_grants_execution(prepared, monkeypatch):
    engine = SurgicalEngine(prepared[0][1].scene); before = engine.snapshot()
    def forbidden(*a, **k): pytest.fail('Implicit source-byte or native replay work')
    monkeypatch.setattr(preflight, 'verify_source', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'restore', forbidden)
    value = interactive_planning_availability(engine)['sourcePreflight']
    assert statuses(value)['source-bytes'] == statuses(value)['native-replay'] == 'not-checked'
    assert not value['planningAdmitted'] and engine.snapshot() == before


def test_existing_admission_is_reported_only_after_current_validation(prepared):
    _, source, _, collection = prepared[0]; engine = source.clone(); before = engine.snapshot()
    value = preflight.source_preflight(engine, native_collection=collection.root, verify_replay=True)
    assert value['planningAdmitted'] and statuses(value)['declared-task'] == 'passed'
    assert statuses(value)['evidence-review'] == 'accepted-for-research'
    assert engine.snapshot() == before
    engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    other = preflight.source_preflight(engine)
    assert not other['planningAdmitted'] and statuses(other)['declared-task'] == 'refused'


@pytest.mark.parametrize('stage', ['source-bytes', 'native-replay'])
def test_bridge_cancellation_is_not_swallowed_as_an_incomplete_gate(prepared, monkeypatch, stage):
    engine = SurgicalEngine(prepared[0][1].scene); before = engine.snapshot(); request = _Request('preflight-cancel')
    def cancel(*a, **k): request.cancel(); request.check()
    if stage == 'source-bytes': monkeypatch.setattr(preflight, 'verify_source', cancel)
    else: monkeypatch.setattr(SurgicalEngine, 'restore', cancel)
    with pytest.raises(BridgeError, match='cancelled'):
        preflight.source_preflight(engine, native_collection=prepared[0][3].root, verify_replay=True, check=request.check)
    assert engine.snapshot() == before and admission.admission(engine) is None


def test_byte_refusal_remains_an_explicit_gate_and_does_not_admit(prepared, monkeypatch):
    engine = SurgicalEngine(prepared[0][1].scene); before = engine.snapshot()
    def changed(*a, **k): raise ValueError('Original bytes changed')
    monkeypatch.setattr(preflight, 'verify_source', changed)
    value = preflight.source_preflight(engine, native_collection=prepared[0][3].root, verify_replay=True)
    assert statuses(value)['source-bytes'] == 'refused' and statuses(value)['native-replay'] == 'not-checked'
    assert not value['planningAdmitted'] and engine.snapshot() == before


@pytest.mark.parametrize('change', ['native', 'declaration'])
def test_inspection_cannot_publish_a_mixture_of_changed_inputs(prepared, monkeypatch, change):
    engine = SurgicalEngine(prepared[0][1].scene); checks = 0
    def check():
        nonlocal checks
        checks += 1
        if change == 'native' and checks == 2:
            engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    evidence = preflight.source_evidence; calls = 0
    def changing(scene):
        nonlocal calls
        value = copy.deepcopy(evidence(scene)); calls += 1
        if calls == 2: value['declarationSha256'] = 'changed-authority'
        return value
    if change == 'declaration': monkeypatch.setattr(preflight, 'source_evidence', changing)
    with pytest.raises(ValueError, match='changed during inspection'):
        preflight.source_preflight(engine, check=check)
    assert admission.admission(engine) is None


def test_refused_source_never_opens_native_collection_or_replays(prepared, monkeypatch):
    engine = SurgicalEngine(prepared[0][1].scene)
    def refused(*a, **k): raise ValueError('Source role is reserved')
    def forbidden(*a, **k): pytest.fail('Refused source reached native payloads')
    monkeypatch.setattr(preflight, 'source_evidence', refused)
    monkeypatch.setattr(preflight, 'verify_source', forbidden); monkeypatch.setattr(SurgicalEngine, 'restore', forbidden)
    value = preflight.source_preflight(engine, native_collection=prepared[0][3].root, verify_replay=True)
    assert statuses(value)['source-role'] == 'refused' and not value['planningAdmitted']
