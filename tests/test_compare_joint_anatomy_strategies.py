"""Joint native export retains history and accounts for reopened replay."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import compare_joint_anatomy_strategies as exporter

from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_transition_meter import TransitionMeter


INSPECT = {'slot': 'left', 'operation': 'inspect', 'duration': .2}


def _base():
    labels = np.zeros((3, 3, 3), np.uint16)
    labels[1, 1, 1] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'caseHash': 'generated joint export contract test'}, {})
    return SurgicalEngine(scene)


def _executed(base, count):
    candidate = base.clone()
    for _ in range(count):
        assert candidate.step(INSPECT)['status'] == 'accepted'
    return candidate


def test_joint_install_retains_classic_pair_and_every_original_branch():
    base = _base()
    originals = {'main': base.snapshot()}
    originals.update({f'classic-{index}': base.snapshot() for index in range(15)})
    record = {'branches': copy.deepcopy(originals), 'branchId': 'main', 'position': 0,
              'parents': {'classic-0': {'branchId': 'main', 'position': 0}},
              'planning': {'comparisonSeal': 'old-access'},
              'sectorPlanning': {'comparisonSeal': 'old-sector'}}
    parents = copy.deepcopy(record['parents'])
    access_engine, sector_engine, failed_engine = (
        _executed(base, 1), _executed(base, 2), _executed(base, 1))
    access_plan = {'branchId': 'joint-access-superior-access',
                   'method': 'superior-access',
                   'finalStateId': access_engine.state_id}
    sector_plan = {'branchId': 'joint-sector-superior-access-left-far-first',
                   'strategy': {'sourceAccessMethod': 'superior-access'},
                   'finalStateId': sector_engine.state_id}
    failure = {'schema': 'resectionlab.joint-anatomy-failure.v1',
               'branchId': 'joint-failed-sector-superior-access',
               'retentionRole': 'stopped_native_branch_not_admitted',
               'startStateId': base.state_id, 'finalStateId': failed_engine.state_id,
               'nativeActions': 1}
    result = {'accessBranches': [(access_engine, access_plan)],
              'sectorBranches': [(sector_engine, sector_plan)],
              'failedBranches': [(failed_engine, failure)]}
    access = {'comparisonSeal': 'new-access', 'alternatives': [access_plan]}
    sectors = {'sourceAccessComparisonSeal': 'new-access',
               'comparisonSeal': 'new-sector', 'alternatives': [sector_plan]}
    created = exporter._install_joint(record, result, access, sectors)
    assert len(created) == 3 and len(record['branches']) == 19
    assert {key: record['branches'][key] for key in originals} == originals
    assert {key: record['parents'][key] for key in parents} == parents
    assert all(record['parents'][key] == {'branchId': 'main', 'position': 0}
               for key in created)
    assert record['sectorPlanningHistory'][-1] == {
        'sourceAccessComparison': {'comparisonSeal': 'old-access'},
        'comparison': {'comparisonSeal': 'old-sector'}}
    assert record['planning'] == record['sectorSourceAccessComparison'] == access
    assert record['sectorPlanning'] == sectors

    meter = TransitionMeter(10)
    replayed = exporter._replay_exported(meter, base, record, created)
    assert len(replayed) == 3
    assert meter.report()['byPhase']['export-reopen-replay']['physicalCalls'] == 4
    assert meter.calls == sum(row['nativeActions'] for row in replayed) == 4


def test_reopened_joint_replay_rejects_spliced_actions_before_any_call():
    base = _base()
    candidate = _executed(base, 1)
    branch_id = 'joint-access-superior-access'
    saved = {'branches': {branch_id: candidate.snapshot()}}
    created = {branch_id: {'candidate': candidate, 'kind': 'access',
                           'stateId': candidate.state_id,
                           'actionsHash': semantic_digest(candidate.actions)}}
    saved['branches'][branch_id]['actions'][0]['duration'] = .4
    meter = TransitionMeter(2)
    with pytest.raises(ValueError, match='changed before native replay'):
        exporter._replay_exported(meter, base, saved, created)
    assert meter.calls == 0


def test_reopened_joint_replay_honors_global_deadline_before_native_call():
    base = _base()
    candidate = _executed(base, 1)
    branch_id = 'joint-access-superior-access'
    saved = {'branches': {branch_id: candidate.snapshot()}}
    created = {branch_id: {'candidate': candidate, 'kind': 'access',
                           'stateId': candidate.state_id,
                           'actionsHash': semantic_digest(candidate.actions)}}
    meter = TransitionMeter(2)
    with pytest.raises(TimeoutError, match='end-to-end deadline'):
        exporter._replay_exported(meter, base, saved, created,
            check=lambda: (_ for _ in ()).throw(TimeoutError('end-to-end deadline')))
    assert meter.calls == 0


def test_install_refuses_branch_beyond_export_replay_reserve():
    base = _base()
    candidate = _executed(base, 49)
    record = {'branchId': 'main', 'position': 0,
              'branches': {'main': base.snapshot()},
              'planning': {'comparisonSeal': 'old-access'},
              'sectorPlanning': {'comparisonSeal': 'old-sector'}}
    result = {'accessBranches': [(candidate, {
        'branchId': 'joint-access-superior-access',
        'method': 'superior-access'})],
        'sectorBranches': [], 'failedBranches': []}
    with pytest.raises(ValueError, match='Joint native branch or receipt differs'):
        exporter._install_joint(record, result,
            {'alternatives': []}, {'alternatives': []})
    assert record['branches'] == {'main': base.snapshot()}


def test_generic_classic_source_and_full_failed_receipt_binding(monkeypatch):
    base = _base()
    candidate = _executed(base, 1)
    record = {'branchId': 'main', 'position': 0,
              'branches': {'main': base.snapshot(), 'existing-learned': candidate.snapshot()},
              'planning': {'comparisonSeal': 'access'},
              'sectorPlanning': {'sourceAccessComparisonSeal': 'access',
                                 'comparisonSeal': 'sector'},
              'learnedSectorExecutions': [{'branchId': 'existing-learned'}]}
    monkeypatch.setattr(exporter, 'validate_saved_comparison', lambda *_: None)
    monkeypatch.setattr(exporter, 'validate_saved_sector_comparison', lambda *_: None)
    source = exporter._source_record(record, base.scene)
    assert source['branchCount'] == 2
    assert source['branchMapHash'] == semantic_digest(record['branches'])
    record['jointPlanning'] = {'jointSeal': 'existing'}
    with pytest.raises(ValueError, match='unmodified classic source'):
        exporter._source_record(record, base.scene)
    record['jointPlanning'] = None
    with pytest.raises(ValueError, match='unmodified classic source'):
        exporter._source_record(record, base.scene)
    record.pop('jointPlanning')

    failure = {'schema': 'resectionlab.joint-anatomy-failure.v1',
               'branchId': 'joint-failed-access-superior-access',
               'retentionRole': 'stopped_native_branch_not_admitted',
               'startStateId': base.state_id, 'finalStateId': candidate.state_id,
               'nativeActions': 1, 'cleanupComplete': True,
               'cleanupPending': False,
               'verification': {'exactNativeReplay': True},
               'modeledOutcomes': {
                   'schema': 'resectionlab.surgical-modeled-outcomes.v1',
                   'sceneId': base.scene.identity,
                   'sourceLabelsHash': array_digest(base.scene.labels),
                   'stateId': candidate.state_id,
                   'baselineStateId': base.state_id,
                   'actionCount': 1,
                   'actionHistoryHash': semantic_digest(candidate.actions)}}
    record['branches'][failure['branchId']] = candidate.snapshot()
    assert exporter._validate_failed_receipts(base.scene, record,
        {'failedExecutions': [failure]}, base.state_id) == [failure['branchId']]
    forged = copy.deepcopy(failure)
    forged['modeledOutcomes']['actionHistoryHash'] = 'sha256:forged'
    with pytest.raises(ValueError, match='modeled outcomes differ'):
        exporter._validate_failed_receipts(base.scene, record,
            {'failedExecutions': [forged]}, base.state_id)


def test_joint_report_seal_rejects_changed_cost_and_execution_rule():
    class Identity:
        identity = 'sha256:scene'
        state_id = 'sha256:start'
    access = {'comparisonSeal': 'sha256:access'}
    sectors = {'comparisonSeal': 'sha256:sectors',
               'sourceAccessComparisonSeal': 'sha256:access'}
    report = {'schema': 'resectionlab.joint-anatomy-strategy.v1',
              'sourceSceneId': Identity.identity, 'startStateId': Identity.state_id,
              'geometry': {}, 'trials': [], 'failedExecutions': [],
              'cost': {'physicalCalls': 1},
              'status': 'joint_strategies_compared',
              'executionRule': 'one far-first left-CUSA sector strategy per source-selected route',
              'optimizerUpdates': 0, 'accessComparison': access,
              'sectorComparison': sectors}
    report['jointSeal'] = semantic_digest(report)
    result = {'accessBranches': [], 'sectorBranches': [], 'failedBranches': [],
              'report': report, 'meter': TransitionMeter(3)}
    assert exporter._joint_documents(result, Identity, Identity) == (access, sectors, report)
    altered = copy.deepcopy(result)
    altered['report']['cost']['physicalCalls'] = 2
    with pytest.raises(ValueError, match='Joint report seal differs'):
        exporter._joint_documents(altered, Identity, Identity)
    altered = copy.deepcopy(result)
    altered['report']['executionRule'] = 'another strategy'
    with pytest.raises(ValueError, match='source or evidence fields differ'):
        exporter._joint_documents(altered, Identity, Identity)
    excessive = copy.deepcopy(result)
    excessive['accessBranches'] = [(None, {})]*7
    with pytest.raises(ValueError, match='six retained native branches'):
        exporter._joint_documents(excessive, Identity, Identity)


def test_joint_run_refuses_existing_or_nonignored_output(tmp_path):
    source = tmp_path/'source.ressectionlab'
    source.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='new ignored testcases'):
        exporter.run(source, tmp_path/'outside')
    ignored = tmp_path/'testcases'
    ignored.mkdir()
    exists = ignored/'used'
    exists.mkdir()
    with pytest.raises(ValueError, match='new ignored testcases'):
        exporter.run(source, exists)
