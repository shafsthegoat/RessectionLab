"""A useful but failed prefix cannot become a successful quality advantage."""
import importlib.util
from pathlib import Path


def test_paired_summary_keeps_raw_gain_and_penalizes_failed_endpoint_separately():
    path=Path(__file__).resolve().parents[1]/'scripts/summarize_sequential_milestone.py'
    spec=importlib.util.spec_from_file_location('summary_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=dict(tier='high',parent=200,pattern=0,seed=0,treated_mm3=27.,treatment_fraction=.25,
        qualified_treatment_fraction=.25,valid_stop_withdrawal=True,neural_removed_mm3=0.,
        physical_requests=10,online_wall_seconds=4.)
    greedy=dict(base,method='greedy');search=dict(base,method='search')
    learned=dict(base,method='learned',seed=17,treated_mm3=54.,treatment_fraction=.5,
        qualified_treatment_fraction=0.,valid_stop_withdrawal=False)
    paired=module.paired([greedy,search,learned])
    assert len(paired)==2
    assert all(r['treatment_fraction_difference']==.25 for r in paired)
    assert all(r['qualified_fraction_difference']==-.25 for r in paired)
    assert all(r['valid_stop_difference']==-1 for r in paired)
