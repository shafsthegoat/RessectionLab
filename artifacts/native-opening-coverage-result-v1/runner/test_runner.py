"""Tiny scalar/fake accounting controls; no task or model construction."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('opening_common', HERE/'common.py')
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)


def arm():
    return {'status': 'completed', 'terminated': True, 'audit': {'feasible': True},
        'search': {'layers': [{'completed': True}], 'completed_layers': 1, 'call_cap_reached': False,
                   'time_cap_reached': False, 'actor_forward_calls': 0, 'model_transition_calls': 20},
        'identity': dict(source_hash='source', decision_model_hash='model', initial_inventory_sha256='candidates', reward={}, model={}, volume={}),
        'metrics': {'target_removed_mm3': 2., 'normal_removed_mm3': 4., 'total_reward': 1.1},
        'physical': {'target_mm3': 2., 'outside_supplied_target_mm3': 4.}, 'preparation_outside_mm3': 3.}


def test_tie_is_negative_but_complete():
    result = common.useful_gate([arm(), arm()], {'passed': True})
    assert result['comparison_admissible'] and not result['useful_gain']


def test_one_strict_gain_and_all_outcomes_required():
    before, after = arm(), arm(); after['search']['model_transition_calls'] -= 1
    assert common.useful_gate([before, after], {'passed': True})['useful_gain']
    for group, key, value in [('metrics', 'target_removed_mm3', 1.), ('metrics', 'normal_removed_mm3', 5.),
                             ('metrics', 'total_reward', 1.), ('physical', 'target_mm3', 1.),
                             ('physical', 'outside_supplied_target_mm3', 5.)]:
        changed = deepcopy(after); changed[group][key] = value
        assert not common.useful_gate([before, changed], {'passed': True})['useful_gain']


def test_preparation_shift_does_not_hide_final_outside_cost():
    before, after = arm(), arm(); after['preparation_outside_mm3'] -= .5
    assert common.useful_gate([before, after], {'passed': True})['useful_gain']
    after['physical']['outside_supplied_target_mm3'] += .5
    assert not common.useful_gate([before, after], {'passed': True})['useful_gain']


def test_truncation_and_failed_positive_refuse_comparison():
    for key in ('call_cap_reached', 'time_cap_reached'):
        after = arm(); after['search'][key] = True
        assert not common.useful_gate([arm(), after], {'passed': True})['comparison_admissible']
    after = arm(); after['search']['layers'][0]['completed'] = False
    assert not common.useful_gate([arm(), after], {'passed': True})['comparison_admissible']
    assert not common.useful_gate([arm()], {})['comparison_admissible']


def test_wrong_source_cannot_gain():
    after = arm(); after['identity']['source_hash'] = 'other'; after['preparation_outside_mm3'] = 0.
    assert not common.useful_gate([arm(), after], {'passed': True})['useful_gain']


def test_fake_preview_includes_refused_and_exception_costs(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(HERE))
    spec = importlib.util.spec_from_file_location('opening_worker', HERE/'worker.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    class Engine:
        def preview_stroke(self, status):
            if status == 'error': raise RuntimeError('generated refusal')
            return SimpleNamespace(feasible=status == 'ok', microsteps=[None], cut_union_attempt={
                'leaf_tests': 3, 'new_leaf_count': 1, 'witness_rows': 2, 'peak_tracked_cells': 1, 'budget_refused': status == 'refused'})
        def committed_snapshot_identity(self): return 'identity'
    class Task:
        _inventory = None
        def _transition(self): return None
        def _prepare_inventory(self): return {}
        def _assert_frozen(self): return None
    work = worker.Work(tmp_path, lambda: None); work.install(Engine, Task)
    try:
        for status in ('ok', 'refused', 'error'):
            try: Engine().preview_stroke(status)
            except RuntimeError: pass
        row = work.row()
        assert row['previews_started'] == 3 and row['previews_returned'] == 2 and row['preview_exceptions'] == 1
        assert row['previews_refused'] == row['budget_refused'] == 1 and row['leaf_tests'] == 6
        assert row['preview_seconds'] >= 0. and work.active is None
    finally: work.restore()


def test_fewer_calls_for_stop_only_is_still_negative():
    before, after = arm(), arm()
    for row in (before, after):
        row['metrics']['target_removed_mm3'] = 0.
        row['metrics']['total_reward'] = 0.
    after['search']['model_transition_calls'] -= 1
    result = common.useful_gate([before, after], {'passed': True})
    assert result['comparison_admissible'] and not result['useful_gain']
    assert not result['both_selected_routes_positive']


def test_actual_instrumented_inventory_stop_and_volume_keys(tmp_path, monkeypatch):
    # Root-dispatched single tiny interface control; no search or trained actor.
    import sys
    monkeypatch.syspath_prepend(str(HERE))
    root = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
    monkeypatch.syspath_prepend(str(root/'src'))
    spec = importlib.util.spec_from_file_location('opening_worker_actual', HERE/'worker.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
    from resectionlab.native_cut_union import CutUnionSpec
    work = worker.Work(tmp_path, lambda: None); work.install(NativeResectionEngine, NativeSpatialTask)
    try:
        task = make_native_opening_task(max_steps=8, generated_cut_union=CutUnionSpec(
            exposure_mode='same_step_parent_closure', closure_schedule='terminal_microstep_only'))
        inventory = task.candidate_inventory()
        assert inventory['complete'] and inventory['declared_slots'] == 12
        assert work.row()['previews_started'] == work.row()['previews_returned'] == 12
        assert work.row()['pre_union_refusals'] > 0
        result = worker.route_readout(task, ['STOP'], work, 'control', lambda: None)
        assert result['terminated'] and result['audit']['feasible']
        assert result['physical']['source_mm3'] == result['physical']['target_mm3'] == 0.
        assert result['metrics']['normal_removed_mm3'] == result['metrics']['total_reward'] == 0.
        common.write_json(tmp_path/'report.json', result)
        assert sum(row['previews_started'] for row in work.phases.values()) == 12
    finally: work.restore()
