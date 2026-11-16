"""Source sector geometry and physical order/hand option contracts."""
import copy
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_environment import load_environment, standardize
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_sector_planning import (
    _sector_cells, compare_sector_strategies, execute_strategy_option,
    intact_strategy_observation,
    validate_saved_sector_comparison)
from resectionlab.workspace_bundle import load_workspace


FRESH_0007 = Path(__file__).resolve().parents[1]/(
    'testcases/outputs/integrated-goal-20261026/fresh-patients/'
    'UCSF-PDGM-0007/access-comparison-v4/access-comparison.ressectionlab')
FRESH_0005 = Path(__file__).resolve().parents[1]/(
    'testcases/outputs/integrated-goal-20261026/fresh-patients/'
    'UCSF-PDGM-0005/access-comparison-v6/access-comparison.ressectionlab')
JOINT_0005 = Path(__file__).resolve().parents[1]/(
    'testcases/outputs/integrated-goal-20261026/'
    'joint-native-profile-0005-v2/joint-comparison.ressectionlab')


def test_generated_source_target_sectors_are_disjoint_and_affine_bound():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[5:15, 7:14, 7:14] = 2
    affine = np.eye(4)
    affine[:3, 3] = [10., -20., 30.]
    scene = standardize(SurgicalScene(labels, affine,
        [{'id': 2, 'name': 'supplied target', 'material': 'tumor', 'target': True}],
        {'evidenceClass': 'generated-engineering', 'provenance': 'generated sector geometry test'}, {}))
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
             'accessCompleteness': 'unsupported_skull_access_brain_corridor',
             'surfacePose': {'position': [10., -10., 40.]},
             'inwardRasUnit': [1., 0., 0.], 'pathLengthMm': 5.}
    cells, evidence = _sector_cells(scene, route)
    assert cells['near'] and cells['far'] and not cells['near'] & cells['far']
    assert evidence['near']['sourceCellCount'] == len(cells['near'])
    assert evidence['near']['sourceCellSha256'] != evidence['far']['sourceCellSha256']
    assert evidence['near']['centerRasMm'][0] < evidence['far']['centerRasMm'][0]
    changed = copy.deepcopy(route)
    changed['sourceSceneId'] = 'sha256:different'
    with pytest.raises(ValueError, match='source-bound'):
        _sector_cells(scene, changed)


def test_source_sector_capacity_refuses_before_any_native_traversal():
    labels = np.zeros((12, 12, 12), np.uint16)
    labels[5, 5, 5] = 2
    scene = standardize(SurgicalScene(labels, np.eye(4),
        [{'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'evidenceClass': 'generated-engineering', 'provenance': 'generated impossible-sector test'}, {}))
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
             'accessCompleteness': 'unsupported_skull_access_brain_corridor',
             'surfacePose': {'position': [0., 5., 5.]},
             'inwardRasUnit': [1., 0., 0.], 'pathLengthMm': 5.}
    source = {'task': 'anatomy-access', 'sourceSceneId': scene.identity,
              'comparisonSeal': 'generated-source-seal', 'status': 'insufficient_distinct_routes',
              'alternatives': [{'method': 'generated', 'branchId': 'generated',
                                'finalStateId': engine.state_id, 'decisions': [],
                                'route': route}]}
    branches, report = compare_sector_strategies(engine, source,
        {'generated': before}, acknowledge_unknowns=True)
    assert branches == [] and engine.snapshot() == before
    assert report['passedCount'] == 0 and report['optionCount'] == 4
    assert all(row['reason'] == 'PREDECLARED_SOURCE_SECTOR_CAPACITY_BELOW_10_MM3'
               for row in report['optionFailures'])


@pytest.mark.skipif(not FRESH_0007.is_file(), reason='Ignored fresh-development case is unavailable')
def test_historical_patient_sector_comparison_requires_procedure_policy():
    _, artifacts, _ = load_workspace(FRESH_0007)
    record = artifacts['surgical_toolbox']; scene = load_environment(FRESH_0007)
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    before = intact.snapshot()
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'):
        compare_sector_strategies(intact, record['planning'], record['branches'], acknowledge_unknowns=True)
    assert intact.snapshot() == before


@pytest.mark.skipif(not FRESH_0007.is_file(), reason='Ignored fresh-development case is unavailable')
def test_historical_patient_sector_observation_requires_procedure_policy():
    _, artifacts, _ = load_workspace(FRESH_0007)
    record = artifacts['surgical_toolbox']; scene = load_environment(FRESH_0007)
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    before = intact.snapshot()
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'):
        intact_strategy_observation(intact, record['planning']['alternatives'][0])
    assert intact.snapshot() == before


@pytest.mark.skipif(not FRESH_0005.is_file(), reason='Ignored fresh-development case is unavailable')
def test_negative_case_retries_same_endpoint_then_checked_reverse_cleanup():
    _, artifacts, _ = load_workspace(FRESH_0005)
    record = artifacts['surgical_toolbox']
    scene = load_environment(FRESH_0005)
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    original = intact.snapshot()
    superior = record['planning']['alternatives'][0]
    corridor, first = execute_strategy_option(intact, superior, 'corridor:left')
    near, second = execute_strategy_option(corridor, superior, 'near:left',
        corridor_binding=first['corridorBinding'])
    assert second['status'] == 'accepted' and second['sectorDeltaMm3']['near'] == 5.
    finished, stop = execute_strategy_option(near, superior, 'STOP',
        corridor_binding=first['corridorBinding'])
    assert stop['status'] == 'accepted' and stop['cleanupComplete']
    assert stop['cleanupMode'] == 'generic' and not stop['sourceCleanupAttempted']
    assert any(row['action']['operation'] == 'move' and
               row['action']['parameters']['radiusMm'] == .6 and
               row['status'] == 'accepted' for row in stop['attempts'])
    assert len(finished.actions) <= 48 and intact.snapshot() == original

    axis0 = record['planning']['alternatives'][1]
    corridor, first = execute_strategy_option(intact, axis0, 'corridor:left')
    near, second = execute_strategy_option(corridor, axis0, 'near:left',
        corridor_binding=first['corridorBinding'])
    trials = second['attempts']
    assert trials[0]['reason'] == 'SHAFT_COLLISION'
    assert trials[1]['status'] == 'accepted'
    assert trials[0]['action']['pose'] == trials[1]['action']['pose']
    assert trials[0]['action']['duration'] == .2 and trials[1]['action']['duration'] == .6
    assert (near.metrics()['nonTargetRemovedMm3']-
            first['corridorBinding']['prefixMetrics']['nonTargetRemovedMm3']) <= 20.


@pytest.mark.skipif(not JOINT_0005.is_file(), reason='Ignored joint 0005 native case is unavailable')
def test_exact_joint_0005_source_boundary_uses_recorded_checked_stop_for_both_hands():
    _, artifacts, _ = load_workspace(JOINT_0005)
    record = artifacts['surgical_toolbox']
    scene = load_environment(JOINT_0005)
    access = next(row for row in record['planning']['alternatives']
                  if row['method'] == 'axis0-access')
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    source = intact.snapshot()
    prefix_length = len(access['decisions'])-2
    for hand in ('left', 'right'):
        corridor, entry = execute_strategy_option(intact, access, 'corridor:'+hand)
        assert entry['status'] == 'accepted'
        assert len(corridor.actions) == prefix_length == 38
        original_corridor = corridor.snapshot()
        if hand == 'left':
            refused, far = execute_strategy_option(corridor, access, 'far:left',
                corridor_binding=entry['corridorBinding'])
            assert far['status'] == 'refused'
            assert far['reason'].startswith('ESCAPE_PREFLIGHT_MACRO_')
            assert refused.snapshot() == original_corridor
        else:
            refused = corridor
        stopped, stop = execute_strategy_option(refused, access, 'STOP',
            corridor_binding=entry['corridorBinding'])
        expected = [dict(copy.deepcopy(row['action']), slot=hand)
                    for row in access['decisions'][prefix_length:]]
        assert stop['status'] == 'accepted' and stop['cleanupComplete']
        assert stop['cleanupMode'] == 'saved_source_access'
        assert stop['sourceCleanupAttempted'] and stop['sourceCleanupReused']
        assert stop['sourceCleanupRefusal'] is None
        assert stop['nativeActionsAppended'] == 2 and len(stopped.actions) == 40
        assert stopped.actions[prefix_length:] == expected
        assert [row['action'] for row in stop['attempts']] == expected
        assert all(row['cleanupPath'] == 'saved_source_access' and
                   row['status'] == 'accepted' and
                   row['previewStateId'] == row['committedStateId']
                   for row in stop['attempts'])
        assert stopped.metrics()['targetRemovedMm3'] == corridor.metrics()['targetRemovedMm3']
        assert stopped.metrics()['nonTargetRemovedMm3'] == corridor.metrics()['nonTargetRemovedMm3']
        replay = SurgicalEngine.restore(scene, stopped.snapshot(), use_cache=False)
        assert replay.state_id == stopped.state_id and replay.actions == stopped.actions
        if hand == 'left':
            assert stopped.state_id == access['finalStateId']
    assert intact.snapshot() == source


@pytest.mark.skipif(not JOINT_0005.is_file(), reason='Ignored joint 0005 native case is unavailable')
def test_saved_source_stop_budget_and_changed_binding_do_not_splice_history():
    _, artifacts, _ = load_workspace(JOINT_0005)
    record = artifacts['surgical_toolbox']
    scene = load_environment(JOINT_0005)
    access = next(row for row in record['planning']['alternatives']
                  if row['method'] == 'axis0-access')
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    corridor, entry = execute_strategy_option(intact, access, 'corridor:left')
    original = corridor.snapshot()
    budgeted, stop = execute_strategy_option(corridor, access, 'STOP',
        corridor_binding=entry['corridorBinding'], max_actions=len(corridor.actions)+1)
    assert stop['status'] == 'refused' and stop['reason'] == 'ACTION_BUDGET'
    assert stop['sourceCleanupAttempted'] and not stop['sourceCleanupReused']
    assert stop['sourceCleanupRefusal'] == 'ACTION_BUDGET'
    assert stop['cleanupMode'] == 'generic_after_source_refusal'
    assert stop['nativeActionsAppended'] == 1 and len(budgeted.actions) == 39
    assert any(row.get('cleanupPath') == 'saved_source_access' and
               row['action']['operation'] == 'withdraw' and
               row['status'] == 'not_attempted' for row in stop['attempts'])
    assert any(row.get('cleanupPath') == 'generic_after_source_refusal'
               for row in stop['attempts'])
    assert corridor.snapshot() == original
    changed = copy.deepcopy(entry['corridorBinding'])
    changed['prefixStateId'] = 'sha256:changed-boundary'
    with pytest.raises(ValueError, match='corridor ancestry'):
        execute_strategy_option(corridor, access, 'STOP', corridor_binding=changed)
    tampered = copy.deepcopy(access)
    tampered['decisions'][-1]['action']['duration'] = 30.
    _, stop = execute_strategy_option(corridor, tampered, 'STOP',
        corridor_binding=entry['corridorBinding'])
    assert not stop['sourceCleanupAttempted'] and not stop['sourceCleanupReused']
    assert stop['cleanupMode'] == 'generic'
    assert all(row.get('cleanupPath') != 'saved_source_access'
               for row in stop['attempts'])
    assert corridor.snapshot() == original


@pytest.mark.skipif(not FRESH_0005.is_file(), reason='Ignored fresh-development case is unavailable')
def test_failed_far_macro_preflight_leaves_0005_corridor_unchanged():
    _, artifacts, _ = load_workspace(FRESH_0005)
    record = artifacts['surgical_toolbox']
    scene = load_environment(FRESH_0005)
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    access = record['planning']['alternatives'][0]
    corridor, entry = execute_strategy_option(intact, access, 'corridor:left')
    before = corridor.snapshot()
    refused, option = execute_strategy_option(corridor, access, 'far:left',
        corridor_binding=entry['corridorBinding'])
    assert refused.snapshot() == before == corridor.snapshot()
    assert option['status'] == 'refused'
    assert option['reason'].startswith('ESCAPE_PREFLIGHT_MACRO_')
    assert option['nativeActionsAppended'] == 0 and option['attempts'] == []
    probe = option['escapePreflight']
    assert probe['status'] == 'refused' and not probe['macroReusedAsReturnedBranch']
    assert probe['macroAttempts'] and probe['cleanupAttempts']
    assert probe['committedCalls'] == probe['proposedNativeActions'] + \
        probe['checkedCleanupNativeActions']
    assert option['physicalCalls'] == option['totalPreviewCalls'] + \
        option['totalCommittedCalls'] == probe['previewCalls'] + probe['committedCalls']


@pytest.mark.skipif(not FRESH_0007.is_file(), reason='Ignored fresh-development case is unavailable')
def test_cleanup_budget_is_probed_before_admitting_sector_macro():
    _, artifacts, _ = load_workspace(FRESH_0007)
    record = artifacts['surgical_toolbox']
    scene = load_environment(FRESH_0007)
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    access = record['planning']['alternatives'][0]
    corridor, entry = execute_strategy_option(intact, access, 'corridor:left')
    original = corridor.snapshot()
    # One remaining slot admits the near stroke but cannot admit checked STOP.
    refused, option = execute_strategy_option(corridor, access, 'near:left',
        corridor_binding=entry['corridorBinding'],
        max_actions=len(corridor.actions)+1)
    assert refused.snapshot() == original == corridor.snapshot()
    assert option['reason'] == 'ESCAPE_PREFLIGHT_CLEANUP_ACTION_BUDGET'
    probe = option['escapePreflight']
    assert probe['proposedNativeActions'] == 1 and not probe['cleanupComplete']
    assert probe['macroFinalStateId'] != corridor.state_id
    assert option['nativeActionsAppended'] == 0
    # With the ordinary ceiling the exact macro clone is retained; only STOP
    # transitions are discarded and charged to the preflight counters.
    admitted, option = execute_strategy_option(corridor, access, 'near:left',
        corridor_binding=entry['corridorBinding'])
    assert option['status'] == 'accepted' and option['escapePreflight']['status'] == 'passed'
    assert option['escapePreflight']['macroReusedAsReturnedBranch']
    assert admitted.actions[:len(corridor.actions)] == corridor.actions
    assert option['nativeActionsAppended'] == len(admitted.actions)-len(corridor.actions)
    assert option['totalCommittedCalls'] == option['nativeActionsAppended'] + \
        option['escapePreflight']['committedCalls']


@pytest.mark.skipif(not FRESH_0005.is_file(), reason='Ignored fresh-development case is unavailable')
def test_patient_comparison_refuses_before_any_preview_or_native_transition(monkeypatch):
    _, artifacts, _ = load_workspace(FRESH_0005)
    record = artifacts['surgical_toolbox']; scene = load_environment(FRESH_0005)
    intact = SurgicalEngine.restore(scene, record['branches']['main'])
    before = intact.snapshot()
    def forbidden(*_args, **_kwargs):
        pytest.fail('Patient admission refusal executed native or preview work')
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'):
        compare_sector_strategies(intact, record['planning'], record['branches'], acknowledge_unknowns=True)
    assert intact.snapshot() == before


@pytest.mark.skipif(not JOINT_0005.is_file(), reason='Ignored historical joint case is unavailable')
def test_saved_patient_sector_comparison_remains_readable_without_new_planning():
    _, artifacts, _ = load_workspace(JOINT_0005)
    record = artifacts['surgical_toolbox']; scene = load_environment(JOINT_0005)
    comparison = record['sectorPlanning']
    before = copy.deepcopy(record)
    assert validate_saved_sector_comparison(scene, record, comparison, replay=True)['verified']
    assert comparison['passedCount'] == 0
    assert all(not plan['continuationOutcome']['taskGateMet'] for plan in comparison['alternatives'])
    assert record == before


@pytest.mark.skipif(not JOINT_0005.is_file(), reason='Ignored historical joint case is unavailable')
def test_saved_patient_sector_seal_and_task_gate_cannot_be_promoted():
    _, artifacts, _ = load_workspace(JOINT_0005)
    record = artifacts['surgical_toolbox']; scene = load_environment(JOINT_0005)
    original = record['sectorPlanning']
    changed = copy.deepcopy(original)
    changed['alternatives'][0]['decisions'][-1]['sectorDeltaMm3']['near'] += 1.
    changed['comparisonSeal'] = semantic_digest({k: v for k, v in changed.items() if k != 'comparisonSeal'})
    with pytest.raises(ValueError, match='strategy seal'):
        validate_saved_sector_comparison(scene, record, changed)
    changed = copy.deepcopy(original)
    changed['alternatives'][0]['continuationOutcome']['taskGateMet'] = True
    changed['alternatives'][0]['status'] = 'passed'
    changed['comparisonSeal'] = semantic_digest({k: v for k, v in changed.items() if k != 'comparisonSeal'})
    with pytest.raises(ValueError, match='task gate'):
        validate_saved_sector_comparison(scene, record, changed)
