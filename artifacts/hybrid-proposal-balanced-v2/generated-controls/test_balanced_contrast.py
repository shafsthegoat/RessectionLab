"""Generated controls for one loss change and lean exhaustive parity."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_proposal as new
from test_hybrid_proposal import fixture


def old_module():
    path=HERE.parent/'hybrid-proposal-pilot-v1/hybrid_proposal.py'
    spec=importlib.util.spec_from_file_location('_frozen_v1_ridge_controls',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def data():
    rng=np.random.default_rng(27)
    x=rng.normal(size=(12,new.FEATURES))
    y=np.array([0.,0.,0.,2., 0.,0.,0.,0., 1.,2.,3.,4.])
    states=np.repeat([0,1,2],4)
    return x,y,states


def test_original_parameters_exact_and_balanced_normalization_unchanged():
    x,y,states=data();old=old_module().RidgePriority.fit(x,y,states)
    control=new.RidgePriority.fit(x,y,states)
    balanced=new.RidgePriority.fit(x,y,states,positive_zero_balanced=True)
    assert old.fingerprint==control.fingerprint
    for name in ('mean','scale','weights'):
        assert np.array_equal(getattr(old,name),getattr(control,name))
    assert old.intercept==control.intercept
    for name in ('mean','scale'):
        assert np.array_equal(getattr(control,name),getattr(balanced,name))
    assert not np.array_equal(control.weights,balanced.weights)


def test_equal_state_mass_and_single_class_states_preserved():
    _,y,states=data()
    old=new.regression_weights(y,states)
    new_weights=new.regression_weights(y,states,positive_zero_balanced=True)
    for state in range(3):
        assert new_weights[states==state].sum()==pytest.approx(1/3)
    assert new_weights.sum()==pytest.approx(1.)
    assert new_weights[(states==0)&(y>0)].sum()==pytest.approx(1/6)
    assert new_weights[(states==0)&(y==0)].sum()==pytest.approx(1/6)
    assert np.array_equal(new_weights[states!=0],old[states!=0])
    # All-zero state remains present and supplies its full mass.
    assert new_weights[states==1].sum()==pytest.approx(1/3)


@pytest.mark.parametrize('primal',[False,True])
def test_balanced_fit_matches_independent_unpenalized_intercept_normal_equations(primal):
    x,y,states=data()
    if primal:
        x=np.random.default_rng(31).normal(size=(684,new.FEATURES))
        y=np.tile(np.array([0.,0.,1.,2.]),171);states=np.repeat(np.arange(19),36)
    fit=new.RidgePriority.fit(x,y,states,positive_zero_balanced=True)
    weights=new.regression_weights(y,states,positive_zero_balanced=True)
    z=(x-fit.mean)/fit.scale
    # First-order conditions independently check both coefficient and free
    # intercept optima without reproducing the fit's centering/dual solve.
    residual=z@fit.weights+fit.intercept-y
    assert abs(weights@residual)<1e-10
    np.testing.assert_allclose(z.T@(weights*residual)+new.RIDGE*fit.weights,0.,atol=1e-10)


def test_all_zero_groups_have_no_fabricated_positive_mass():
    x,y,states=data();y[:]=0
    fit=new.RidgePriority.fit(x,y,states,positive_zero_balanced=True)
    assert np.count_nonzero(fit.weights)==0 and fit.intercept==0
    np.testing.assert_array_equal(new.regression_weights(y,states,positive_zero_balanced=True),new.regression_weights(y,states))


def test_lean_exhaustive_has_no_feature_proxy_work_and_preserves_full_route(monkeypatch):
    case=fixture();old=old_module().GeneratedSparseTask(case,mode='exhaustive',max_steps=6)
    old_actions,old_account=old.observed_greedy_search(seconds=10)
    def forbidden(*a,**k):raise AssertionError('unused exhaustive feature work')
    monkeypatch.setattr(new,'public_rows',forbidden)
    events=[];costs=new.Costs(inventory_callback=events.append)
    lean=new.GeneratedSparseTask(case,mode='exhaustive',max_steps=6,costs=costs)
    actions,account=lean.observed_greedy_search(seconds=10)
    assert actions==old_actions
    assert account['decisions']==old_account['decisions']
    assert account['estimated_incremental_return']==old_account['estimated_incremental_return']
    assert costs.counts.get('feature_rows',0)==0 and costs.counts.get('ridge_calls',0)==0
    assert 'public_features' not in costs.seconds and 'ranking' not in costs.seconds
    assert 'inventory_logging' in costs.seconds and events
    for action in actions:
        assert lean.step(action).info==old.step(action).info
    assert lean.metrics()==old.metrics()
    assert all(event['shortlist']['priority_scores'] is None for event in events)


def test_complete_old_ridge_route_and_full_scores_are_preserved():
    case=fixture();old=old_module()
    x,y,states=data();old_fit=old.RidgePriority.fit(x,y,states);fit=new.RidgePriority.fit(x,y,states)
    prior=old.GeneratedSparseTask(case,mode='ridge_top8',ranker=old_fit,max_steps=6)
    task=new.GeneratedSparseTask(case,mode='ridge_top8',ranker=fit,max_steps=6)
    old_actions,old_account=prior.observed_greedy_search(seconds=10)
    actions,account=task.observed_greedy_search(seconds=10)
    assert actions==old_actions and account['decisions']==old_account['decisions']
    record=task.candidate_inventory()['shortlist']
    assert len(record['raw_ids'])==len(record['priority_scores'])==record['raw_count']
    for action in actions:assert task.step(action).info==prior.step(action).info
    assert task.metrics()==prior.metrics()
