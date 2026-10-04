"""Frozen one-scale calibration and held-out metrics; no solver or implicit data IO.

The actual archive reader is role gated in mechanics_hbe_access. This module's
small pure functions also support analytical unit controls without patient data.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class Curve(NamedTuple):
    branch: str
    coordinate: tuple[float, ...]
    response: tuple[float, ...]
    response_unit: str
    source_sha256: str | None = None


def curve(branch, coordinate, response, *, source_sha256=None):
    if branch not in ('compression','tension','torsion_neg','torsion_pos'):
        raise ValueError('Unknown declared branch')
    x,y = np.array(coordinate,dtype=float,copy=True),np.array(response,dtype=float,copy=True)
    if x.ndim != 1 or len(x)<2 or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Complete finite paired curve with at least two coordinates required')
    dx=np.diff(x)
    if not (np.all(dx>0) or np.all(dx<0)):
        raise ValueError('Repeated or nonmonotone input coordinates; no row removal')
    sign=-1 if branch in ('compression','torsion_neg') else 1
    if np.any(sign*x<0):
        raise ValueError('Input-coordinate sign disagrees with frozen branch; no sign fitting')
    return Curve(branch,tuple(map(float,x)),tuple(map(float,y)), 'Nm' if branch.startswith('torsion') else 'N',source_sha256)


def checked_curve(value):
    if not isinstance(value,Curve):
        raise ValueError('Expected explicit Curve contract')
    checked=curve(value.branch,value.coordinate,value.response,source_sha256=value.source_sha256)
    if value.response_unit!=checked.response_unit:
        raise ValueError('Response units disagree with branch')
    return checked


def quadrature_weights(coordinate):
    x=np.asarray(coordinate,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional coordinate required')
    dx=np.diff(x)
    if not (np.all(dx>0) or np.all(dx<0)):
        raise ValueError('Coordinate must be strictly monotone')
    lengths=np.abs(dx)
    w=np.r_[lengths[0]/2,(lengths[:-1]+lengths[1:])/2,lengths[-1]/2]
    if not np.isfinite(w).all() or w.sum()<=0 or not math.isfinite(float(w.sum())):
        raise ValueError('Unusable quadrature scale')
    return w/w.sum()


def interpolate(reference, coordinates):
    reference=checked_curve(reference)
    x,y=np.asarray(reference.coordinate),np.asarray(reference.response)
    query=np.asarray(coordinates,dtype=float)
    if query.ndim!=1 or not np.isfinite(query).all():
        raise ValueError('Finite coordinate vector required')
    if x[0]>x[-1]: x,y=x[::-1],y[::-1]
    if np.any(query<x[0]) or np.any(query>x[-1]):
        raise ValueError('Measured coordinate outside solved range; no extrapolation')
    return np.interp(query,x,y)


def _positive(value):
    value=float(value)
    if not math.isfinite(value) or value<=0:
        raise ValueError('Positive finite physical scale required')
    return value


def branch_metrics(observed, prediction, *, characteristic_response_scale):
    observed,prediction=checked_curve(observed),checked_curve(prediction)
    if observed.branch!=prediction.branch:
        raise ValueError('Branch mismatch')
    scale=_positive(characteristic_response_scale)
    y=np.asarray(observed.response); predicted=interpolate(prediction,observed.coordinate)
    residual=predicted-y; weights=quadrature_weights(observed.coordinate)
    with np.errstate(over='raise',invalid='raise'):
        mse=float(weights@(residual*residual)); denominator=float(np.sqrt(weights@(y*y)))
        mae=float(weights@np.abs(residual)); rms=math.sqrt(mse)
    minimum=64*np.finfo(np.float64).eps*max(scale,float(np.max(np.abs(y))))
    normalized=rms/denominator if denominator>minimum else None
    endpoint=int(np.argmax(np.abs(observed.coordinate)))
    return {'branch':observed.branch,'role':'held_out_torque' if observed.branch.startswith('torsion') else 'calibration_fit',
            'response_unit':observed.response_unit,'row_count':len(y),'RMSE':rms,'MAE':mae,
            'observed_RMS_denominator':denominator,'normalized_RMSE':normalized,
            'normalized_error_reason':None if normalized is not None else 'observed_RMS_below_declared_numerical_floor',
            'signed_endpoint_bias':float(residual[endpoint]),'weights':weights.tolist(),
            'predicted_at_observed_coordinates':predicted.tolist(),'residuals':residual.tolist(),
            'row_IID_confidence_interval':None,'physical_validation_pass':None}


def fit_scale(calibration, reference, *, reference_mu_Pa=1000., radius_m=.004):
    """Analytical positive scale, equal total mode weights, no intercept or clipping."""
    if set(calibration)!= {'compression','tension'} or set(reference)!=set(calibration):
        raise ValueError('Exactly the two declared axial calibration modes required')
    mu,R=_positive(reference_mu_Pa),_positive(radius_m)
    F0=_positive(mu*R*R); numerator=0.;denominator=0.;details={}
    for mode in ('compression','tension'):
        obs,ref=checked_curve(calibration[mode]),checked_curve(reference[mode])
        if obs.branch!=mode or ref.branch!=mode:
            raise ValueError('Wrong curve supplied under branch label')
        y=np.asarray(obs.response);g=interpolate(ref,obs.coordinate);w=.5*quadrature_weights(obs.coordinate)
        with np.errstate(over='raise',invalid='raise'):
            numerator+=float(w@(g*y));denominator+=float(w@(g*g))
        details[mode]={'source_sha256':obs.source_sha256,'coordinate':list(obs.coordinate),
                       'observed':y.tolist(),'reference_prediction':g.tolist(),'weights':w.tolist()}
    threshold=_positive(1e-12*F0*F0)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator<=threshold:
        raise ValueError('Unidentified/nonfinite calibration denominator')
    scale=_positive(numerator/denominator); fitted_mu=_positive(scale*mu)
    return {'schema':'hbe-positive-scale-fit-v1','scale':scale,'mu_Pa':fitted_mu,'reference_mu_Pa':mu,
            'numerator':numerator,'denominator':denominator,'denominator_minimum':threshold,
            'fitted_parameters':['positive_mu_scale_only'],'modes':details,
            'held_out_values_used':False,'offset_or_sign_fitted':False}


def scale_prediction(reference, scale):
    reference=checked_curve(reference);scale=_positive(scale)
    return curve(reference.branch,reference.coordinate,np.asarray(reference.response)*scale,
                 source_sha256=reference.source_sha256)


def paired_metrics(observed, predictions, *, characteristic_response_scale):
    if set(observed)!=set(predictions) or set(observed) not in ({'compression','tension'},{'torsion_neg','torsion_pos'}):
        raise ValueError('Exactly one declared two-branch mode pair required')
    if any(observed[mode].branch!=mode or predictions[mode].branch!=mode for mode in observed):
        raise ValueError('Curve branch disagrees with pair dictionary key')
    records={mode:branch_metrics(observed[mode],predictions[mode],characteristic_response_scale=characteristic_response_scale) for mode in sorted(observed)}
    normalized=[r['normalized_RMSE'] for r in records.values()]
    result={'branches':records,'equal_branch_RMSE':math.sqrt(sum(r['RMSE']**2 for r in records.values())/2),
            'equal_branch_normalized_RMSE':None if any(x is None for x in normalized) else math.sqrt(sum(x*x for x in normalized)/2),
            'physical_validation_pass':None,'empirical_tolerance':None,'independent_donor_validation':False}
    if set(observed)=={'torsion_neg','torsion_pos'}:
        neg,pos=observed['torsion_neg'],observed['torsion_pos']
        lower=max(min(map(abs,neg.coordinate)),min(pos.coordinate)); upper=min(max(map(abs,neg.coordinate)),max(pos.coordinate))
        if upper<=lower:
            result['odd_symmetry']={'reason':'no_nonzero_shared_angle_interval','observed_RMSE_Nm':None}
        else:
            angles=np.linspace(lower,upper,21)
            obs_sum=interpolate(pos,angles)+interpolate(neg,-angles)
            pred_sum=interpolate(predictions['torsion_pos'],angles)+interpolate(predictions['torsion_neg'],-angles)
            weights=quadrature_weights(angles)
            result['odd_symmetry']={'angles_rad':angles.tolist(),'observed_sum_Nm':obs_sum.tolist(),
                                    'predicted_sum_Nm':pred_sum.tolist(),
                                    'observed_RMSE_Nm':float(np.sqrt(weights@(obs_sum**2))),
                                    'predicted_RMSE_Nm':float(np.sqrt(weights@(pred_sum**2)))}
    return result
