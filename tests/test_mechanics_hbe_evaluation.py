"""Analytical curve controls only: no released HBE response is accessed."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('hbe_evaluation',Path(__file__).resolve().parents[1]/'scripts/mechanics_hbe_evaluation.py')
e=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(e)


def curves(scale=1):
    return {'compression':e.curve('compression',[0,-.0002,-.001],[0,-.002*scale,-.01*scale]),
            'tension':e.curve('tension',[0,.0005,.001],[0,.005*scale,.01*scale])}


def test_closed_form_scale_and_equal_mode_weight_with_unequal_rows():
    ref=curves();observed=curves(2.5)
    observed['tension']=e.curve('tension',np.linspace(0,.001,51),np.linspace(0,.025,51))
    fit=e.fit_scale(observed,ref)
    assert fit['scale']==pytest.approx(2.5)
    assert fit['mu_Pa']==pytest.approx(2500)
    for details in fit['modes'].values():assert sum(details['weights'])==pytest.approx(.5)
    predictions={k:e.scale_prediction(v,fit['scale']) for k,v in ref.items()}
    assert e.paired_metrics(observed,predictions,characteristic_response_scale=.04)['equal_branch_RMSE']<1e-17


def test_nonpositive_fit_and_zero_identifiability_rejected_not_clipped():
    with pytest.raises(ValueError):e.fit_scale(curves(-1),curves())
    with pytest.raises(ValueError,match='denominator'):e.fit_scale(curves(),curves(0))
    with pytest.raises(ValueError):e.fit_scale(curves(),curves(),reference_mu_Pa=float('inf'))


def test_trapezoidal_weights_and_closed_form_metrics():
    np.testing.assert_allclose(e.quadrature_weights([0,1,3]),[1/6,1/2,1/3])
    obs=e.curve('torsion_pos',[0,1,3],[0,2,6]);pred=e.curve('torsion_pos',[0,1,3],[1,3,7])
    result=e.branch_metrics(obs,pred,characteristic_response_scale=1)
    assert result['RMSE']==pytest.approx(1)
    assert result['MAE']==pytest.approx(1)
    assert result['observed_RMS_denominator']==pytest.approx(np.sqrt(14))
    assert result['signed_endpoint_bias']==1
    assert result['physical_validation_pass'] is None


def test_no_offset_sign_or_endpoint_hiding_and_no_extrapolation():
    with pytest.raises(ValueError,match='outside'):e.interpolate(curves()['tension'],[.00101])
    with pytest.raises(ValueError,match='sign'):e.curve('compression',[0,1],[0,1])
    with pytest.raises(ValueError):e.curve('tension',[0,1,1],[0,1,1])
    with pytest.raises(ValueError):e.curve('tension',[0,1,2],[0,np.nan,2])
    observed=curves();observed['tension']=e.curve('tension',[0,.0005,.001],[.01,.015,.02])
    fit=e.fit_scale(observed,curves())
    assert fit['offset_or_sign_fitted'] is False
    pred=e.scale_prediction(curves()['tension'],fit['scale'])
    assert e.branch_metrics(observed['tension'],pred,characteristic_response_scale=.02)['RMSE']>0


def test_zero_denominator_metrics_null_and_signed_torsion_asymmetry_preserved():
    z=e.curve('torsion_pos',[0,.1,.2],[0,0,0]);pred=e.curve('torsion_pos',[0,.1,.2],[0,1,2])
    assert e.branch_metrics(z,pred,characteristic_response_scale=1)['normalized_RMSE'] is None
    observed={'torsion_pos':pred,'torsion_neg':e.curve('torsion_neg',[0,-.1,-.2],[0,-2,-4])}
    predictions={'torsion_pos':pred,'torsion_neg':e.curve('torsion_neg',[0,-.1,-.2],[0,-1,-2])}
    result=e.paired_metrics(observed,predictions,characteristic_response_scale=1)
    assert result['odd_symmetry']['observed_RMSE_Nm']>0
    assert result['odd_symmetry']['predicted_RMSE_Nm']==pytest.approx(0,abs=1e-14)
    assert result['branches']['torsion_neg']['signed_endpoint_bias']==2


def test_mutating_source_lists_does_not_change_curve():
    x=[0,1];y=[0,2];c=e.curve('tension',x,y)
    x[1]=9;y[1]=99
    assert c.coordinate==(0,1) and c.response==(0,2)
