"""Cancellation during canonical proof must not publish a native study."""
import copy

import pytest

from resectionlab import surgical_strategy_workspace as workspace
from resectionlab import surgical_two_tool as planning
from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
from resectionlab.imaging import save_case, read_case_artifacts
from test_surgical_two_tool import prepared, qualified, comparison


def cancel_on_prefix_hit(monkeypatch, request):
    hits = []
    original = planning.ProgramPrefixCache.get
    def get(cache, actions, *, check):
        value = original(cache, actions, check=check)
        if value is not None:
            hits.append(1)
            assert request.cancel()
        return value
    monkeypatch.setattr(planning.ProgramPrefixCache, 'get', get)
    return hits


def test_cold_validation_cancel_releases_prefixes_without_authority(qualified, comparison, monkeypatch):
    engine = qualified[1]; before = engine.snapshot()
    planning._VALIDATED_REPORTS.clear()
    request = _Request('cold-proof')
    hits = cancel_on_prefix_hit(monkeypatch, request)
    with pytest.raises(BridgeError, match='cancelled'):
        planning.validate_report(engine, comparison, check=request.check)
    assert hits and not planning._VALIDATED_REPORTS
    stats = planning.prefix_cache_stats()['validation']
    assert not stats['completed'] and stats['closed'] and stats['retainedAfterClose'] == 0
    assert engine.snapshot() == before


def test_warm_validation_cannot_hide_cancellation_after_live_binding(qualified, comparison, monkeypatch):
    engine = qualified[1]; planning._VALIDATED_REPORTS.clear()
    assert planning.validate_report(engine, comparison)
    request = _Request('warm-proof'); original = planning._binding
    checked = []
    def binding(engine):
        value = original(engine)
        checked.append(1); assert request.cancel()
        return value
    monkeypatch.setattr(planning, '_binding', binding)
    with pytest.raises(BridgeError, match='cancelled'):
        planning.validate_report(engine, comparison, check=request.check)
    assert checked


def test_workspace_install_cancel_is_atomic(qualified, comparison, monkeypatch):
    case, engine, original, _ = qualified
    record = copy.deepcopy(original); before = copy.deepcopy(record)
    planning._VALIDATED_REPORTS.clear(); workspace._VALIDATED.clear()
    request = _Request('workspace-proof'); hits = cancel_on_prefix_hit(monkeypatch, request)
    with pytest.raises(BridgeError, match='cancelled'):
        workspace.install_report(engine, record, case.semantic_hash, comparison, check=request.check)
    assert hits and record == before
    assert not workspace._VALIDATED and not planning._VALIDATED_REPORTS


def test_bridge_cancelling_proof_leaves_case_and_live_record_unchanged(qualified, comparison, monkeypatch, tmp_path):
    case, engine, record, controls = qualified
    path = tmp_path/'cancel-proof.ressectionlab'
    save_case(case, path, artifacts={'surgical_toolbox': record})
    original_bytes = path.read_bytes()
    session = BridgeSession(tmp_path/'transfers')
    try:
        session.execute('loadCase', {'path': str(path)}, _Request('load'), lambda *_: None)
        args = {'caseHash': case.semantic_hash, 'command': 'inspect'}
        before = session.execute('surgicalRehearsal', args, _Request('inspect'), lambda *_: None)
        cached = copy.deepcopy(session.surgical_sessions[case.semantic_hash][2])
        planning._VALIDATED_REPORTS.clear(); workspace._VALIDATED.clear()
        # Generation is already covered independently. Cancel the subsequent
        # real cold canonical proof of this exact generated report.
        monkeypatch.setattr(planning, 'generate_comparison', lambda *a, **k: copy.deepcopy(comparison))
        request = _Request('bridge-proof'); hits = cancel_on_prefix_hit(monkeypatch, request)
        with pytest.raises(BridgeError, match='cancelled'):
            session.execute('surgicalRehearsal', {**args, 'command': 'generateStrategies',
                'expectedStateId': before['view']['stateId'], 'constraints': controls,
                'acknowledgeResearchUnknowns': True}, request, lambda *_: None)
        assert hits and not request.committing
        assert session.surgical_sessions[case.semantic_hash][2] == cached
        assert path.read_bytes() == original_bytes
        assert read_case_artifacts(path)['surgical_toolbox'] == record
        after = session.execute('surgicalRehearsal', args, _Request('after'), lambda *_: None)
        assert after['view']['stateId'] == before['view']['stateId'] == engine.state_id
        assert after['strategyPlanning'] is None
    finally:
        session.close()
