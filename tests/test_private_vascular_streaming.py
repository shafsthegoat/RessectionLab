"""Generated integration boundaries for the private scorer's tiled kernel."""
from dataclasses import replace
import pytest
import test_private_vascular_evaluation as generated
from resectionlab import private_vascular_evaluation as v

tmp_path = generated.tmp_path
setup = generated.setup


def test_full_checked_history_uses_tiled_work_with_ordered_sweep_rows(setup,tmp_path):
    reference = generated.reference(setup,partial=True)
    result = generated.evaluate(setup,reference,tmp_path/'tiled')
    assert result['status'] == 'evaluated_generated_vascular_reference'
    assert result['nonstop_sweeps'] == len(result['per_nonstop_action']) == 2
    assert result['whole_tool']['touched_reference_cells'] == result['contact_work']['reference_sampled_cells']
    assert result['contact_work']['maximum_tile_cells'] <= 4096
    assert result['shaft']['positive_reference_cells'] == 1
    assert result['tip']['positive_reference_cells'] == 1
    assert result['removed_overlap']['positive_overlap_cells'] == 1


def test_contact_budget_failure_is_saved_without_partial_success_or_replanning(setup,tmp_path,monkeypatch):
    reference = generated.reference(setup)
    budget = v.ContactBudget
    monkeypatch.setattr(v,'ContactBudget',lambda **kwargs:replace(budget(**kwargs),max_sampled_cells=0))
    calls=[]
    def loader():
        calls.append(1)
        return reference
    result = generated.evaluate(setup,reference,tmp_path/'budget-refusal',loader)
    assert result['status'] == 'evaluation_failed' and result['outcomes'] is None
    assert result['error_type'] == 'BudgetExceeded'
    assert calls == [1] and result['planning_after_private_load'] is False


def test_admission_caps_and_reference_body_remain_closed():
    assert v.MAX_REFERENCE_VOXELS == 32**3
    assert v.MAX_PLANNING_VOXELS == 16**3
    assert v.MAX_MICROSTEPS == 256
    assert not hasattr(v,'_capsule_cells')
