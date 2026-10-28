"""Native re-entry, unchanged original baselines, and persisted plan binding."""
import copy

import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_staged_planning import (
    compare_staged_strategies, install_staged_comparison, validate_staged_report)
from test_surgical_fan_inventory import coupon


def test_complete_native_fan_keeps_source_and_original_tissue_ledger(monkeypatch):
    import resectionlab.surgical_staged_fan as fan
    scene, routes = coupon()
    scene = standardize(scene)
    for route in routes:
        route['sourceSceneId'] = scene.identity
        route['targetEntryIndex'] = [round(value) for value in route['entryPose']['position']]
    monkeypatch.setattr(fan, 'source_routes', lambda *_a: routes[:1])
    engine = SurgicalEngine(scene)
    original = engine.snapshot()
    result = compare_staged_strategies(engine, {}, cumulative_reentry=True,
        acknowledge_unknowns=True, max_physical_calls=1200, max_wall_seconds=60.)
    assert engine.snapshot() == original
    assert not result['report']['interruption']
    assert result['meter'].remaining >= 720
    assert len(result['branches']) == 1
    native, row = result['branches'][0]
    assert row['stages'] and row['outcome']['cleanupComplete']
    assert row['interruptionRecovery']['exactResume']
    assert row['outcome']['legacyTask']['additionalNonTargetMm3'] <= 20.
    assert row['outcome']['wholeNonTargetRemovedMm3'] <= 200.
    assert row['outcome']['wholeTargetRemovedMm3'] > row['approachBaseline']['targetRemovedMm3']
    inserted = False
    geometry = {}
    for event in native.history:
        action = event['action']
        if inserted:
            assert all(value == geometry[key] for key, value in action.get('parameters', {}).items()
                       if key in geometry)
        if action['operation'] == 'place':
            geometry = {key: action['parameters'].get(key, default)
                        for key, default in (('radiusMm', 1.5), ('depthMm', 2.))}
        if action['operation'] in ('apply', 'move'):
            inserted = True
        if action['operation'] == 'withdraw' and event['status'] == 'accepted':
            inserted = False
    record = {'branchId': 'main', 'position': 0, 'branches': {'main': original}, 'timeline': {'main': []}}
    install_staged_comparison(record, result, scene)
    assert validate_staged_report(scene, record, record['stagedPlanning'])['verified']
    for field in ('checkedExitStateId', 'checkedExitActions', 'baseline'):
        forged = copy.deepcopy(record['stagedPlanning'])
        changed = forged['alternatives'][0]
        if field == 'baseline':
            changed['approachBaseline']['nativeActions'] += 1
        elif field == 'checkedExitActions':
            changed['stages'][0][field] += 1
        else:
            changed['stages'][0][field] = 'sha256:'+'0'*64
        forged['comparisonSeal'] = semantic_digest({key: value for key, value in forged.items() if key != 'comparisonSeal'})
        with pytest.raises(ValueError, match='Cumulative'):
            validate_staged_report(scene, record, forged)
    altered = copy.deepcopy(record['stagedPlanning'])
    altered['inventory']['fans'][0]['orderedRays'][0]['raySpec']['entryDurationSeconds'] += .1
    altered['inventory']['inventorySeal'] = semantic_digest({
        key: value for key, value in altered['inventory'].items() if key != 'inventorySeal'})
    altered['comparisonSeal'] = semantic_digest({key: value for key, value in altered.items() if key != 'comparisonSeal'})
    with pytest.raises(ValueError, match='selected source fan'):
        validate_staged_report(scene, record, altered)


@pytest.mark.parametrize('options', [
    {'include_opposite_faces': True}, {'initial_access_radius_mm': .6},
    {'explore_fresh_corridors': True}])
def test_cumulative_mode_does_not_silently_replace_its_frozen_configuration_rule(options):
    with pytest.raises(ValueError, match='selects its initial configuration'):
        compare_staged_strategies(None, {}, cumulative_reentry=True,
            acknowledge_unknowns=True, **options)
