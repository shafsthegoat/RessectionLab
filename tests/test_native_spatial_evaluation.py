"""Analytical transition/accounting checks; no patient or training run."""
import pytest

from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.native_spatial_task import make_native_opening_task, OPENING_TOOLS


def action(task, tool, voxel):
    return next(row['action_id'] for row in task.candidate_inventory()['ledger']
                if row['feasible'] and row['tool_id'] == tool and row['voxel'] == list(voxel))


def completed():
    task = make_native_opening_task()
    task.step(action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1)))
    task.step(action(task, OPENING_TOOLS[1].tool_id, (4, 4, 5)))
    return task


def test_independent_complete_tool_reward_union_contacts_and_fraction():
    task = completed()
    record = evaluate_native_spatial_episode(task)
    assert record['accepted'] and record['target_access_success']
    assert record['geometry']['complete_tool_checked'] and record['geometry']['frontier_checked']
    metrics = record['outcomes']
    assert metrics['target_removed_mm3'] == 2.
    assert metrics['normal_removed_mm3'] == 4.
    assert metrics['simulated_removed_volume_mm3'] == 6.
    assert metrics['total_reward'] == pytest.approx(1.1)
    assert metrics['reference_target_fraction_removed'] == 1.
    assert metrics['tool_changes'] == 1
    assert metrics['currently_retained_contacted_tissue_upper_bound_mm3'] >= 0
    assert metrics['motor_surrogate'] is metrics['clinical_deficit_probability'] is None
    assert record['invalid_action_attempts'] is None


def test_stop_is_feasible_but_not_target_access_and_prefix_cannot_pass():
    task = make_native_opening_task()
    with pytest.raises(ValueError, match='completed'):
        evaluate_native_spatial_episode(task)
    task.step('STOP')
    record = evaluate_native_spatial_episode(task)
    assert record['accepted'] and not record['target_access_success']
    assert record['outcomes']['total_reward'] == 0
    assert record['outcomes']['simulated_removed_volume_mm3'] == 0


@pytest.mark.parametrize('name', ['total_reward', 'target_removed_mm3', 'normal_removed_mm3',
                                 'simulated_removed_volume_mm3', 'currently_retained_contacted_tissue_upper_bound_mm3'])
def test_inflated_captured_metrics_cannot_replace_counted_outcomes(name):
    task = completed()
    metrics = task.metrics()
    metrics[name] += 20
    with pytest.raises(ValueError, match='accounting mismatch'):
        evaluate_native_spatial_episode(task, metrics=metrics)


def test_tampered_history_and_reference_binding_rejected():
    task = completed()
    metrics = task.metrics()
    metrics['history'][0]['reward'] += 1
    with pytest.raises(ValueError, match='committed episode'):
        evaluate_native_spatial_episode(task, metrics=metrics)
    metrics = task.metrics()
    metrics['reference_hash'] = 'foreign'
    with pytest.raises(ValueError, match='binding changed'):
        evaluate_native_spatial_episode(task, metrics=metrics)


def test_environment_reward_bug_is_caught_independently(monkeypatch):
    task = make_native_opening_task()
    original = task._score_record

    def biased(*args):
        row = original(*args)
        row['reward'] += 1  # Deliberately simulate a coherent but wrong environment score.
        return row

    monkeypatch.setattr(task, '_score_record', biased)
    task.step(action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1)))
    task.step(action(task, OPENING_TOOLS[1].tool_id, (4, 4, 5)))
    with pytest.raises(ValueError, match='accounting mismatch: reward'):
        evaluate_native_spatial_episode(task)


def test_nominal_planning_clone_is_not_final_episode_evaluation():
    task = make_native_opening_task().planning_clone()
    task.step('STOP')
    with pytest.raises(ValueError, match='planning clone'):
        evaluate_native_spatial_episode(task)


def test_scalar_and_batch_evaluation_agree_without_promoting_attempt_counts():
    task = completed()
    one = evaluate_native_spatial_episode(task, distance_backend='scalar')
    two = evaluate_native_spatial_episode(task, distance_backend='batch', distance_batch_size=7)
    assert one['geometry'] == two['geometry']
    assert one['outcomes'] == two['outcomes']
    assert one['invalid_action_attempts'] is two['invalid_action_attempts'] is None


def test_mid_audit_cancellation_cannot_report_success():
    record = evaluate_native_spatial_episode(completed(), cancelled=lambda: True)
    assert not record['accepted']
    assert record['target_access_success'] is None
    assert record['outcomes'] is None
    assert record['geometry']['failures'] == ('independent_validation_cancelled',)
