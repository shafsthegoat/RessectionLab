"""Generated-only independent arithmetic check; no acquired source/weights.

Normal equations below are solved by scalar Gaussian elimination, not the
production SVD. Existing public synthetic DTO fixtures are reused solely to
exercise the wrapper's actual frozen lifecycle; their values are not evidence
about the acquired corpus. Run only in the root-coordinated numerical slot.
"""
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
SOURCE_SHA = '22853df2126d2eb6efda545abb7d8108eb9f501f493fdf1e3d95a5948d5c0b11'
CHECKS = 0

def need(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)

def near(left, right, message, atol=3e-11):
    need(math.isfinite(float(left)) and math.isfinite(float(right))
         and abs(float(left)-float(right)) <= atol, message)

def normal_equations(rows, targets, weights):
    """Small independent scalar normal-equation oracle with pivoting."""
    n = len(rows[0])
    augmented = [[math.fsum(w*x[i]*x[j] for x,w in zip(rows,weights))
                  for j in range(n)] +
                 [math.fsum(w*x[i]*y for x,y,w in zip(rows,targets,weights))]
                 for i in range(n)]
    for column in range(n):
        pivot = max(range(column,n), key=lambda i: abs(augmented[i][column]))
        augmented[column],augmented[pivot] = augmented[pivot],augmented[column]
        need(abs(augmented[column][column]) > 1e-9, 'Well-conditioned oracle pivot')
        scale = augmented[column][column]
        augmented[column] = [v/scale for v in augmented[column]]
        for i in range(n):
            if i != column:
                factor = augmented[i][column]
                augmented[i] = [v-factor*q for v,q in zip(augmented[i],augmented[column])]
    return [row[-1] for row in augmented]

def residual(rows, y, weights, theta):
    return math.fsum(w*(math.fsum(a*b for a,b in zip(row,theta))-target)**2
                     for row,target,w in zip(rows,y,weights))

def work():
    return {'svd_calls_attempted':0, 'svd_calls_completed':0}

def run():
    import numpy as np
    import torch
    from resectionlab.spatial_continuation_calibration import (
        _minimum_change, calibrate_continuation_readout)
    from resectionlab.spatial_continuation_value import fit_continuation_value
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.core import semantic_digest
    from test_spatial_continuation_value import generated_fixture, corpus_for
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Nontrivial overdetermined, unequal weights, signed targets and STOP mass.
    rows = [[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.],
            [2.,1.,1.],[1.,2.,1.],[2.,2.,1.],[3.,0.,1.]]
    y = [-1.,.8,2.1,1.4,3.8,4.1,4.4,2.3]
    weights = [.03,.07,.11,.09,.13,.08,.14,.10]
    initial = [2.5,-1.25,3.0]
    oracle = normal_equations(rows,y,weights)
    design = np.array(rows+[[1e9,-1e9,1.]])
    targets = np.array(y+[0.]); mass = np.array(weights+[.25])
    stops = np.array([False]*8+[True]); counts=work()
    solution, report = _minimum_change(design,targets,mass,stops,np.array(initial),counts)
    for actual,expected in zip(solution,oracle):
        near(actual,expected,'SVD agrees with independent scalar normal equations')
    expected_residual = residual(rows,y,weights,oracle)
    near(report['weighted_residual_l2_float64']**2,expected_residual,
         'Unnormalized active residual; STOP mass is retained')
    need(report['rank']==3 and report['STOP_rows_excluded']==1,'Full rank/STOP denominator')
    need(expected_residual > .01,'Fixture has irreducible residual')
    need(counts=={'svd_calls_attempted':1,'svd_calls_completed':1},'Single solve')
    # STOP has no effect on coefficients even when its unused design changes.
    design[-1] = [-1e12,2e12,1.]
    changed,_ = _minimum_change(design,targets,mass,stops,np.array(initial),work())
    need(np.array_equal(changed,solution),'Excluded STOP has no solve influence')

    # Exact structural nullspace: duplicate slope columns and dead coordinate.
    z=[0.,.5,1.,2.,3.,4.]
    targets2=[-.5,1.2,2.8,5.3,8.7,10.8]
    weights2=[.05,.15,.10,.20,.20,.30]
    identifiable=normal_equations([[v,1.] for v in z],targets2,weights2)
    old=np.array([4.,-2.,6.25,1.5]); shift=(identifiable[0]-sum(old[:2]))/2
    expected=[old[0]+shift,old[1]+shift,old[2],identifiable[1]]
    solved2,r2=_minimum_change(np.array([[v,v,0.,1.] for v in z]),
        np.array(targets2),np.array(weights2),np.zeros(6,dtype=bool),old,work())
    for a,b in zip(solved2,expected): near(a,b,'Minimum-change identifiable solution')
    near(solved2[0]-solved2[1],old[0]-old[1],'Duplicate-column nullspace retained')
    need(solved2[2]==old[2],'Dead-feature coefficient retained exactly')
    need(r2['rank']==2 and r2['discarded_or_null_directions']==2,'Rank deficiency disclosed')
    # Fixed cutoff is an estimator definition, not an adaptive performance sweep.
    small,r3=_minimum_change(np.diag([1.,1e-10]),np.array([-.5,4.]),
        np.array([.5,.5]),np.zeros(2,dtype=bool),np.array([2.,7.]),work())
    near(small[0],-.5,'Retained direction solved'); near(small[1],7.,'Tiny direction unchanged')
    need(r3['relative_cutoff']==1e-8 and r3['rank']==1,'Declared fixed finite-precision rule')

    # Real generated corpus/head fit exercises authenticated wrapper lifecycle.
    policy,traces,manifest=generated_fixture()
    corpus=corpus_for(policy,traces,manifest)
    oldfit=fit_continuation_value(policy,corpus)
    old_snapshot=(parameter_hash(policy),parameter_hash(oldfit._head),
                  semantic_digest(oldfit.report),corpus.seal)
    parameters=[(id(p),p._version,p.requires_grad) for p in oldfit._head.parameters()]
    hidden_hash=parameter_hash(oldfit._head[0])
    calibrated=calibrate_continuation_readout(policy,corpus,oldfit,max_scores=1)
    need(calibrated is not oldfit and calibrated._head is not oldfit._head,'Separate fitted object')
    need(old_snapshot==(parameter_hash(policy),parameter_hash(oldfit._head),
                       semantic_digest(oldfit.report),corpus.seal),'Base, old64 and corpus unchanged')
    need(parameters==[(id(p),p._version,p.requires_grad) for p in oldfit._head.parameters()],
         'Original parameter identities/versions/flags unchanged')
    need(parameter_hash(calibrated._head[0])==hidden_hash,'Hidden parameters unchanged')
    need(all(p.grad is None and not p.requires_grad for p in calibrated._head.parameters()),
         'Returned head frozen and gradient free')
    need(all(p.grad is None for p in policy.parameters()),'No base gradients')
    need(all(not m.training for m in calibrated._head.modules()),'Returned head eval mode')
    # Independently evaluate the deployed readout using actual Torch float32 ops.
    with torch.no_grad():
        predicted=calibrated._head(torch.tensor(np.array(corpus.features,copy=True))).ravel().tolist()
    deployed=math.fsum(float(w)*((0. if bool(stop) else p)-float(y))**2
        for p,y,w,stop in zip(predicted,corpus.targets,corpus.weights,corpus.stop_rows))
    near(calibrated.report['final_weighted_mse'],deployed,'Actual deployed weighted objective',1e-13)
    near(sum(v['weighted_SSE_after_deployed_float32'] for v in calibrated.report['per_case'].values()),
         deployed,'Per-case denominator decomposition',1e-13)
    need(deployed <= calibrated.report['initial_weighted_mse']+1e-6,
         'Generated deployed objective nonregression')
    costs=calibrated.report['costs']
    need(costs['hidden_forwards_completed']==1 and costs['head_readout_forwards_completed']==2
         and costs['svd_calls_completed']==1,'Actual wrapper work declaration')
    need(all(costs[k]==0 for k in ('base_policy_forwards','base_optimizer_steps',
         'head_optimizer_steps','head_backward_calls')),'Calibration uses no policy/optimizer/gradient work')
    trace=traces[0][2]
    scores=calibrated.score_actions(trace.transitions[0].observation,trace.context)
    need(scores[0]==0. and all(math.isfinite(s) for s in scores),'Public scorer STOP0/finite')
    return {'full_rank':{'solution':solution.tolist(),'normal_equation_oracle':oracle,
                       'weighted_residual':expected_residual},
            'rank_deficient':{'solution':solved2.tolist(),'oracle':expected,
                              'rank':r2['rank'],'nullspace_dimensions':2},
            'wrapper':{'generated_states':len(corpus.targets),'generated_initial_fit_updates':64,
                       'deployed_mse':deployed,'before_mse':calibrated.report['initial_weighted_mse'],
                       'hidden_unchanged':True,'base_unchanged':True,'old64_unchanged':True,
                       'calibration_costs':dict(costs)},
            'limits':'Generated arithmetic/lifecycle only; no acquired fit, route, native geometry or clinical evidence.'}

if __name__=='__main__':
    started=time.perf_counter()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60 s audit cap')))
    signal.alarm(60)
    source=ROOT/'src/resectionlab/spatial_continuation_calibration.py'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    result={'schema':'independent-continuation-calibration-generated-check-v1',
            'source_sha256':before,'scope':'generated_only_no_acquired_inputs_or_weights'}
    try:
        need(before==SOURCE_SHA,'Exact reviewed production source')
        result['checks']=run()
        need(hashlib.sha256(source.read_bytes()).hexdigest()==before,'Source unchanged')
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error=type(error).__name__+': '+str(error))
    result.update(check_count=CHECKS,wall_seconds=time.perf_counter()-started)
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    need(len(raw.encode()) < 32*1024,'Compact independent result')
    out=HERE/'check-01.json'
    with out.open('x') as stream: stream.write(raw)
    print(json.dumps({'status':result['status'],'checks':CHECKS,'output':str(out)}))
    raise SystemExit(result['status']!='PASS')
