"""Joint orchestration failure retention and fixed native resource contracts."""
import copy
import json

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab import surgical_joint_planning as joint


def _engine():
    labels = np.zeros((9, 9, 9), np.uint16)
    labels[2:7, 2:7, 2:7] = 1
    labels[4, 4, 4] = 2
    return SurgicalEngine(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'caseHash': 'generated joint orchestration test'}, {}))


def test_insufficient_source_routes_preserve_intact_state_and_json_seals():
    engine = _engine()
    before = engine.snapshot()
    result = joint.compare_joint_strategies(engine, acknowledge_unknowns=True)
    report = result['report']
    assert report['threeApproachTaskGateMet'] is False
    assert report['accessComparison']['status'] == 'insufficient_distinct_routes'
    assert len(report['accessComparison']['directionFailures']) == 3
    assert result['accessBranches'] == result['sectorBranches'] == result['failedBranches'] == []
    assert report['cost']['physicalCalls'] == 0
    assert engine.snapshot() == before
    record = {'branches': {'main': before}}
    assert joint.validate_joint_report(engine.scene, record, json.loads(json.dumps(report)))[
        'failedBranches'] == 0
    assert semantic_digest(report) == semantic_digest(json.loads(json.dumps(report)))


def test_failed_native_branch_receipt_survives_and_never_becomes_a_strategy(monkeypatch):
    engine = _engine()
    before = engine.snapshot()
    route = {'id': 'generated-source-route', 'nativeAxis': 2,
             'outsidePose': {'position': [4., 4., 12.], 'quaternion': [1., 0., 0., 0.]}}
    monkeypatch.setattr(joint, 'select_joint_corridors', lambda *_args, **_kw:
                        ([route], {'scope': 'generated source-only unit shortlist'}))

    def stopped(base, *_args, **_kwargs):
        candidate = base.clone()
        candidate.step({'slot': 'left', 'operation': 'inspect', 'duration': .2})
        raise joint.AccessNotExecutable('GENERATED_ACCESS_REFUSAL')

    monkeypatch.setattr(joint, '_execute', stopped)
    result = joint.compare_joint_strategies(engine, acknowledge_unknowns=True)
    report = result['report']
    assert not result['accessBranches'] and not result['sectorBranches']
    assert len(result['failedBranches']) == len(report['failedExecutions']) == 1
    candidate, failure = result['failedBranches'][0]
    assert failure['retentionRole'] == 'stopped_native_branch_not_admitted'
    assert failure['verification']['exactNativeReplay']
    assert report['cost']['physicalCalls'] == 2  # Commit and separate fresh replay.
    assert engine.snapshot() == before
    record = {'branches': {'main': before, failure['branchId']: candidate.snapshot()}}
    joint.validate_joint_report(engine.scene, record, json.loads(json.dumps(report)))
    forged = copy.deepcopy(report)
    forged['failedExecutions'][0]['finalStateId'] = before['stateId']
    forged['jointSeal'] = semantic_digest({k: v for k, v in forged.items() if k != 'jointSeal'})
    with pytest.raises(ValueError, match='failed branch receipt'):
        joint.validate_joint_report(engine.scene, record, forged)


def test_joint_planning_requires_opt_in_and_fixed_action_ceiling():
    engine = _engine()
    with pytest.raises(ValueError, match='acknowledgment'):
        joint.compare_joint_strategies(engine)
    for kwargs in ({'max_native_actions': 49}, {'max_physical_calls': 1501},
                   {'max_wall_seconds': 121.}, {'max_physical_calls': 352}):
        with pytest.raises(ValueError, match='ceilings'):
            joint.compare_joint_strategies(engine, acknowledge_unknowns=True, **kwargs)
