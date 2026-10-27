"""Source inventory invariants and separate full-task acceptance gates."""
import copy

import numpy as np
import pytest
import json
from types import SimpleNamespace

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_staged_planning import (
    SCHEMA, assess_task, build_inventory, larger_task, source_zones,
    require_staged_development, validate_staged_report)


def source():
    labels = np.zeros((31, 31, 31), np.uint16)
    labels[2:29, 2:29, 2:29] = 1
    labels[10:21, 10:21, 10:21] = 2
    return SurgicalScene(labels, np.diag([-1., 1., 1., 1.]), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'enhancing target', 'material': 'tumor', 'target': True}],
        {'caseHash': 'generated native inventory fixture'}, {})


def record(scene):
    engine = SurgicalEngine(scene)
    pose = {'position': [-15., 15., 20.], 'quaternion': [1., 0., 0., 0.]}
    route = {'id': 'source-ray', 'sourceSceneId': scene.identity,
        'pathType': 'oblique_mapped_brain', 'nativeAxis': 2,
        'entryPose': pose, 'surfacePose': {**pose, 'position': [-15., 15., 28.]},
        'outsidePose': {**pose, 'position': [-15., 15., 36.]},
        'inwardRasUnit': [0., 0., -1.], 'pathLengthMm': 8.,
        'accessCompleteness': 'unsupported_skull_access_brain_corridor'}
    plan = {'method': 'superior-access', 'route': route,
        'startStateId': engine.state_id, 'decisions': [
            {'action': {'slot': 'left', 'operation': 'apply', 'pose': pose, 'duration': 1.}},
            {'action': {'slot': 'left', 'operation': 'deactivate', 'duration': .2}}]}
    plan['planSeal'] = semantic_digest({'sourceSceneId': scene.identity,
        'startStateId': engine.state_id, 'decisions': plan['decisions']})
    comparison = {'task': 'anatomy-access', 'alternatives': [plan]}
    comparison['comparisonSeal'] = semantic_digest(comparison)
    return {'planning': comparison}


def test_physical_ras_zones_partition_unique_source_target():
    scene = source()
    zones, evidence = source_zones(scene)
    target = set(map(int, np.flatnonzero(scene.labels.ravel() == 2)))
    assert set.union(*(set(row) for row in zones.values())) == target
    assert sum(len(row) for row in zones.values()) == len(target)
    assert len([row for row in zones.values() if row]) == 8
    assert evidence['centerRasMm'] == [-15., 15., 15.]
    assert evidence['sourceTargetMm3'] == len(target)


def test_inventory_is_source_only_bounded_and_does_not_mutate_baselines(monkeypatch):
    scene = source()
    saved = record(scene)
    before, before_labels = copy.deepcopy(saved), scene.labels.copy()
    monkeypatch.setattr(SurgicalEngine, 'preview', lambda *_a, **_kw:
                        pytest.fail('Source inventory must never preview native outcomes'))
    monkeypatch.setattr(SurgicalEngine, 'step', lambda *_a, **_kw:
                        pytest.fail('Source inventory must never commit native actions'))
    inventory = build_inventory(scene, saved, max_tiles=12)
    assert inventory['nativeSelectionCalls'] == 0
    assert len(inventory['routes']) == 1
    assert len(inventory['routes'][0]['tiles']) <= 12
    assert inventory['routes'][0]['tiles']
    axial = [row for row in inventory['routes'][0]['tiles']
             if row['orientationMode'] == 'source-ray']
    assert axial and inventory['routes'][0]['tiles'][0] in axial
    for row in axial:
        assert row['pose']['position'][:2] == [-15., 15.]
        assert row['pose']['quaternion'] == [1., 0., 0., 0.]
    assert inventory['shaftRadiusMm'] == .5
    assert saved == before
    assert np.array_equal(scene.labels, before_labels)
    assert inventory['inventorySeal'] == semantic_digest(
        {k: v for k, v in inventory.items() if k != 'inventorySeal'})


def test_historical_joint_archive_retains_classic_routes_without_new_pin_field():
    scene = source()
    saved = record(scene)
    saved['sectorPlanningHistory'] = [{'sourceAccessComparison': saved.pop('planning')}]
    inventory = build_inventory(scene, saved)
    assert inventory['candidateRoutesExamined'] == 1
    assert len(inventory['routes']) == 1


def test_success_does_not_follow_from_cleanup_or_tiny_target_progress():
    scene = source()
    engine = SurgicalEngine(scene)
    task, (zones, _) = larger_task(scene), source_zones(scene)
    outcome = assess_task(engine, task, zones, {}, exact_replay=True,
        baseline={**engine.metrics(), 'removedCells': []},
        legacy_sectors={'near': frozenset(), 'far': frozenset()})
    assert outcome['cleanupComplete']
    assert not outcome['taskGateMet']
    assert not outcome['legacyTask']['taskGateMet']
    assert set(outcome['failedGates']) == {'wholeTarget', 'zones', 'configurations', 'ratio'}
    assert task['maxNativeActions'] == 240
    assert task['maximumContinuationNonTargetMm3'] == 20
    assert task['maximumWholeNonTargetMm3'] == 200


def test_frozen_patient_cannot_enter_staged_development_work(tmp_path):
    metadata = {'surgical_adapter': {'sourcePatient': 'fixture-assessment',
        'sourceManifest': {'patient_id': 'fixture-assessment', 'series': [
            {'patient_id': 'fixture-assessment', 'source_mirror_revision': 'revision'}]}}}
    case = SimpleNamespace(metadata=metadata)
    with pytest.raises(ValueError, match='excludes frozen'):
        require_staged_development(case, tmp_path)
    folder = tmp_path/'declaration'
    folder.mkdir()
    (folder/'development-declaration.json').write_text(json.dumps({
        'role': 'FROZEN_ASSESSMENT', 'cases': ['fixture-assessment'],
        'mirror_revision': 'revision'}))
    with pytest.raises(ValueError, match='conflicts'):
        require_staged_development(case, tmp_path)


def _native_report():
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[6:9, 6:9, 6:9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated native report validation'}, {})
    engine = SurgicalEngine(scene)
    pose = lambda z: {'position': [6., 6., z], 'quaternion': [0., 0., 0., 1.]}
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
             'entryPose': pose(6.), 'surfacePose': pose(2.), 'pathLengthMm': 4.,
             'accessCompleteness': 'unsupported_skull_access_brain_corridor',
             'inwardRasUnit': [0., 0., 1.]}
    for action in (
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose(2.),
         'parameters': {'radiusMm': 1.5}, 'duration': .1}):
        assert engine.step(action)['status'] == 'accepted'
    baseline = {**engine.metrics(), 'removedCells': [], 'nativeActions': 2}
    assert engine.step({'slot': 'left', 'operation': 'apply', 'pose': pose(8.),
                        'parameters': {'radiusMm': 1.5}, 'duration': .2})['status'] == 'accepted'
    gain = engine.metrics()['targetRemovedMm3']
    stage = {'stage': 0, 'tileId': 'generated', 'workStartPosition': 2,
        'configuration': 'radius-1.5', 'stateId': engine.state_id,
        'nativeActions': 3, 'targetIncrementMm3': gain}
    for action in (
        {'slot': 'left', 'operation': 'deactivate', 'duration': .2},
        {'slot': 'left', 'operation': 'withdraw', 'pose': pose(-5.), 'duration': 10.}):
        assert engine.step(action)['status'] == 'accepted'
    task = larger_task(scene)
    from resectionlab.surgical_sector_planning import _sector_cells
    sectors, _ = _sector_cells(scene, route)
    zones, _ = source_zones(scene)
    outcome = assess_task(engine, task, zones, {'radius-1.5': gain},
        exact_replay=True, baseline=baseline, admitted_strokes=1, legacy_sectors=sectors)
    row = {'branchId': 'staged-generated', 'sourceMethod': 'generated',
        'sourceAccessPlanSeal': 'sha256:generated', 'route': route,
        'finalStateId': engine.state_id, 'status': 'unpassed',
        'nativeActions': len(engine.actions), 'exactNativeReplay': True,
        'outcome': outcome, 'stages': [stage], 'approachBaseline': baseline}
    record = {'branches': {row['branchId']: engine.snapshot()},
              'timeline': {row['branchId']: copy.deepcopy(engine.history)}}
    report = {'schema': SCHEMA, 'sourceSceneId': scene.identity,
        'sourceStateId': engine.fresh().state_id, 'sourceRole': 'DEVELOPMENT',
        'optimizerUpdates': 0, 'task': task, 'wallSeconds': 1.,
        'cost': {'physicalCalls': 5, 'previewCalls': 0, 'directStepCalls': 5,
                 'maxPhysicalCalls': 100}, 'alternatives': [row],
        'sourceAccessPlans': [{'method': row['sourceMethod'], 'planSeal': row['sourceAccessPlanSeal'],
            'sourceSceneId': scene.identity, 'startStateId': engine.fresh().state_id,
            'routeSeal': semantic_digest(route)}]}
    report['comparisonSeal'] = semantic_digest(report)
    return scene, record, report


def _seal_report(report):
    report['comparisonSeal'] = semantic_digest({
        key: value for key, value in report.items() if key != 'comparisonSeal'})


def test_report_rechecks_native_gates_and_rejects_overlapping_configuration_work():
    scene, record, report = _native_report()
    assert validate_staged_report(scene, record, report)['verified']
    duplicate = copy.deepcopy(report)
    row = duplicate['alternatives'][0]
    row['stages'].append(copy.deepcopy(row['stages'][0]))
    row['outcome']['configurationTargetRemovedMm3']['radius-1.5'] *= 2
    _seal_report(duplicate)
    with pytest.raises(ValueError, match='work interval'):
        validate_staged_report(scene, record, duplicate)
    manufactured = copy.deepcopy(report)
    row = manufactured['alternatives'][0]
    row['status'] = 'passed'
    row['outcome']['taskGateMet'] = True
    row['outcome']['checks'] = {key: True for key in row['outcome']['checks']}
    row['outcome']['failedGates'] = []
    _seal_report(manufactured)
    with pytest.raises(ValueError, match='task gates'):
        validate_staged_report(scene, record, manufactured)
