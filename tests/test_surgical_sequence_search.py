"""Generated engineering controls over real native work, protection and replay."""
from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_sequence_search import checked_sequence_search
from resectionlab.surgical_staged_planning import _choose_work, _work_actions
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter
from resectionlab.core import semantic_digest


def delayed_cavity():
    """A lateral target moves the shaft through a proximal brain obstruction.

    Retreating the tip then clearing that cell removes no target. A subsequent
    target skill becomes executable. Labels are generated, not clinical data.
    """
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[13, 10, 5] = 1
    labels[15, 10, 10] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated delayed cavity engineering control'},
        {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    pose = lambda p: {'position': list(p), 'quaternion': [0., 0., 0., 1.]}
    outside, entry = pose([10., 10., -4.]), pose([10., 10., 8.])
    for action in (
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': outside,
         'parameters': {'radiusMm': 1.5}, 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'pose': entry, 'duration': 3.}):
        assert engine.step(action)['status'] == 'accepted'
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
        'surfacePose': pose([10., 10., 0.]), 'entryPose': entry,
        'outsidePose': outside, 'inwardRasUnit': [0., 0., 1.]}
    inventory = {'route': route, 'tiles': [{'id': 'target',
        'pose': pose([15., 10., 10.]), 'orientationMode': 'source-ray'}]}
    assert engine.metrics()['targetRemovedMm3'] == engine.metrics()['nonTargetRemovedMm3'] == 0
    return engine, route, inventory


def run(engine, route, inventory, *, horizon=2, cap=20., expansions=24,
        calls=500, check=lambda: None):
    meter = TransitionMeter(calls)
    result = checked_sequence_search(meter.wrap(engine), route,
        propose=lambda state: _choose_work(state, inventory, set(), radius=1.5,
            include_preparation=True),
        motion=lambda state, tile: _work_actions(state, tile, 1.5, 'left')[0],
        hand='left', radius=1.5,
        limits=SkillLimits(80, calls, max_additional_non_target_mm3=cap,
            baseline_non_target_mm3=0.), meter=meter, horizon=horizon,
        max_expansions=expansions, check=check)
    return (*result, meter)


def test_native_delayed_cavity_preparation_beats_immediate_removal_and_replays():
    engine, route, inventory = delayed_cavity()
    before, labels = engine.snapshot(), engine.scene.labels.copy()
    greedy, greedy_attempts, greedy_search, _ = run(engine, route, inventory, horizon=1)
    assert greedy is None and greedy_search['decision'] == 'STOP'
    assert any(a['tileId'].startswith('prepare-') and a['status'] == 'no_new_target'
               for a in greedy_attempts)
    selected, attempts, search, meter = run(engine, route, inventory)
    assert selected is not None and search['decision'] == 'CONTINUE'
    assert [r['preparationOnly'] for r in selected.path] == [True, False]
    assert [r['targetIncrementMm3'] for r in selected.path] == [0., 1.]
    assert selected.target_mm3 == selected.non_target_mm3 == 1.
    assert all(r['exit']['status'] == 'accepted' for r in selected.path)
    assert selected.exited.actions[:len(engine.actions)] == engine.actions
    assert all(not s['inserted'] and not s['active'] and not s['grip']
               for s in selected.exited.state['slots'].values())
    assert selected.exited.state['holds'] == {}
    assert engine.snapshot() == before and np.array_equal(engine.scene.labels, labels)
    assert search['physicalCalls'] == meter.calls == 23
    # Failed/clipped probes consume computation but contribute no tissue credit.
    assert any(a['tileId'] == 'target' and a['status'] == 'no_cavity_or_target_progress'
               for a in attempts)
    replay = meter.wrap(engine.fresh())
    for action in selected.exited.actions:
        assert replay.step(copy.deepcopy(action))['status'] != 'rejected'
    assert replay.snapshot() == selected.exited.snapshot()
    assert len(replay.state['removed']) == len(set(replay.state['removed'])) == 2
    assert meter.calls == search['physicalCalls']+len(selected.exited.actions)


def test_native_lookahead_cannot_purchase_preparation_beyond_hard_tissue_cap():
    engine, route, inventory = delayed_cavity()
    before = engine.snapshot()
    selected, attempts, search, _ = run(engine, route, inventory, cap=0.)
    assert selected is None and search['decision'] == 'STOP'
    assert any('ADDITIONAL_NON_TARGET_CAP' in str(a['work']['reason']) for a in attempts)
    assert engine.snapshot() == before


def test_preparation_without_positive_continuation_stops_at_original_state():
    engine, route, inventory = delayed_cavity()
    before = engine.snapshot()
    selected, attempts, search, meter = run(engine, route, inventory, expansions=2)
    assert selected is None and search['decision'] == 'STOP'
    assert search['budgetExhausted'] and search['reason'] == 'COMPUTATION_BUDGET'
    assert any(a.get('preparationOnly') is True for a in attempts)
    assert engine.snapshot() == before and meter.calls > 0


def test_native_call_exhaustion_is_reported_as_budget_not_physical_impossibility():
    engine, route, inventory = delayed_cavity()
    before = engine.snapshot()
    selected, _, search, meter = run(engine, route, inventory, calls=12)
    assert selected is None and meter.calls == 12
    assert search['budgetExhausted'] and search['reason'] == 'COMPUTATION_BUDGET'
    assert engine.snapshot() == before


def test_search_meter_counts_every_native_preview_and_direct_step():
    engine, route, inventory = delayed_cavity()
    observed = {'preview': 0, 'step': 0}
    def profile(frame, event, arg):
        if event == 'call':
            if frame.f_code is SurgicalEngine.preview.__code__:
                observed['preview'] += 1
            if frame.f_code is SurgicalEngine.step.__code__:
                observed['step'] += 1
    sys.setprofile(profile)
    try:
        selected, _, _, meter = run(engine, route, inventory)
    finally:
        sys.setprofile(None)
    assert selected is not None
    assert observed == {'preview': meter.preview_calls, 'step': meter.direct_step_calls}


def test_explicit_cancellation_preserves_source_and_never_publishes_tentative_work():
    engine, route, inventory = delayed_cavity()
    before = engine.snapshot()
    def cancel():
        raise InterruptedError('cancelled by caller')
    with pytest.raises(InterruptedError, match='cancelled by caller'):
        run(engine, route, inventory, check=cancel)
    assert engine.snapshot() == before


def test_actual_staged_entry_preserves_baseline_native_branch_and_validates_resume():
    from resectionlab.surgical_staged_planning import (
        compare_staged_strategies, install_staged_comparison, validate_staged_report)
    engine, route, _ = delayed_cavity()
    # An axial target is already reachable; looking ahead must retain this
    # baseline option rather than replace it with unnecessary preparation.
    labels = engine.scene.labels.copy()
    labels[15, 10, 10] = 0
    labels[10, 10, 12] = 2
    source = SurgicalEngine(SurgicalScene(labels, engine.scene.affine,
        engine.scene.regions, {'kind': 'generated baseline preservation control'},
        engine.scene.properties))
    for action in engine.actions:
        assert source.step(copy.deepcopy(action))['status'] == 'accepted'
    route = {**route, 'sourceSceneId': source.scene.identity,
        'nativeAxis': 2, 'pathLengthMm': 8., 'corridorRadiusMm': 1.5,
        'accessCompleteness': 'unsupported_skull_access_brain_corridor'}
    plan = {'method': 'superior-access', 'route': route,
        'startStateId': source.fresh().state_id,
        'decisions': [{'action': copy.deepcopy(a), 'status': 'accepted'} for a in source.actions]}
    plan['planSeal'] = semantic_digest({key: value for key, value in (
        ('sourceSceneId', source.scene.identity), ('startStateId', plan['startStateId']),
        ('decisions', plan['decisions']))})
    comparison = {'task': 'anatomy-access', 'alternatives': [plan]}
    comparison['comparisonSeal'] = semantic_digest(comparison)
    record = {'planning': comparison, 'branchId': 'main', 'position': 0,
        'branches': {'main': source.fresh().snapshot()}}
    baseline = compare_staged_strategies(source.fresh(), record,
        acknowledge_unknowns=True, exercise_resume=False,
        max_physical_calls=2000, max_wall_seconds=30.)
    result = compare_staged_strategies(source.fresh(), record,
        acknowledge_unknowns=True, planning_horizon=2,
        max_physical_calls=2000, max_wall_seconds=30.)
    assert result['report']['interruption'] is None
    final, reference = result['branches'][0][0], baseline['branches'][0][0]
    assert final.state['removed'] == reference.state['removed']
    assert final.metrics()['elapsedSeconds'] <= reference.metrics()['elapsedSeconds']
    row = result['report']['alternatives'][0]
    assert row['interruptionRecovery']['exactResume']
    assert row['outcome']['wholeTargetRemovedMm3'] == 1.
    assert row['outcome']['wholeNonTargetRemovedMm3'] == 0.
    assert not row['outcome']['taskGateMet']  # Tiny software coupon is not a larger-task pass.
    assert row['exactNativeReplay'] and row['sequenceSearches']
    assert row['modeledOutcomes']['target']['removedMm3'] == 1.
    saved = copy.deepcopy(record)
    install_staged_comparison(saved, result, source.scene)
    assert validate_staged_report(source.scene, saved, result['report'])['verified']


@pytest.mark.parametrize('kwargs', [{'horizon': True}, {'horizon': 4}, {'expansions': 25}])
def test_search_declared_bounds(kwargs):
    engine, route, inventory = delayed_cavity()
    with pytest.raises(ValueError, match='Keep sequence'):
        run(engine, route, inventory, **kwargs)
