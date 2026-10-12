"""Root-coordinated generated oracle. Never reads acquired inputs or weights."""
import hashlib,itertools,json,math,pathlib,signal,sys,time
HERE=pathlib.Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'tests')]
import tail_oracle as oracle
COUNT=0

def need(value,message):
    global COUNT
    COUNT+=1
    if not value:raise AssertionError(message)

def close(a,b,message,tol=2e-10):
    need(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tol,message)

def run():
    import numpy as np
    import torch
    from resectionlab.spatial_continuation_tail import recorded_tail_targets,fit_continuation_tail,_bound_targets
    from resectionlab.spatial_continuation_calibration import calibrate_continuation_readout,_minimum_change
    from resectionlab.spatial_continuation_value import fit_continuation_value
    from resectionlab.spatial_continuation_planner import _combine_tail,_choose_positive
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.core import semantic_digest
    from test_spatial_continuation_value import generated_fixture,corpus_for
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    count=0
    for length in range(1,5):
        for rewards in itertools.product((-2.,0.,3.),repeat=length):
            expected=oracle.recorded_tail_targets(rewards);actual=recorded_tail_targets(rewards)
            need(actual==expected,'Shifted recurrence equals independent future-prefix enumeration')
            mandatory=oracle.mandatory_action_targets(rewards)
            need(tuple(r+t for r,t in zip(rewards,actual))==mandatory,'Current reward counted exactly once')
            need(actual[-1]==0. and all(t>=0 for t in actual),'No future invented beyond last recorded action')
            count+=1
    need(recorded_tail_targets((-5.,2.,0.))==(2.,0.,0.),'Negative mandatory action is not erased')
    # The policy gets only current rewards and predictions, never future labels.
    choices=[
        ([0.,1.,2.],[0.,3.,0.],'a'),
        ([0.,1.,2.],[0.,-10.,-20.],'b'),
        ([0.,-1.,2.],[0.,1e6,0.],'b'),
        ([0.,1.,2.],[0.,1.,0.],'a'),
        ([0.,0.,-1.],None,'STOP'),
    ]
    for rewards,predicted,expected in choices:
        rows=[{'action_id':a,'reward':r} for a,r in zip(('STOP','a','b'),rewards)]
        combined=None if predicted is None else _combine_tail(rows,predicted)
        chosen,reason=_choose_positive(rows,combined)
        audited=oracle.audit_choice(['STOP','a','b'],[True]*3,rewards,predicted,expected)
        need(rows[chosen]['action_id']==expected,'Exact reward plus clipped-tail guard and stable tie')
        need(combined is None if audited['combined_scores'] is None else list(combined)==audited['combined_scores'],
             'Production combined scores agree with independent scalar algebra')

    # Tail-specific targets, different numerical solution path from SVD.
    rows=[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.],
          [2.,1.,1.],[1.,2.,1.],[2.,2.,1.]]
    y=list(oracle.recorded_tail_targets((-3.,2.,-4.,1.,3.,-1.,0.,0.)))
    weights=[.07,.08,.11,.09,.10,.12,.13,.30]
    independent=oracle.weighted_normal_equation(rows,y[:-1],weights[:-1])
    design=np.array(rows+[[1e9,-1e9,1.]])
    solution,solver=_minimum_change(design,np.array(y),np.array(weights),
        np.array([False]*7+[True]),np.array([2.,-1.,3.]),
        {'svd_calls_attempted':0,'svd_calls_completed':0})
    for a,b in zip(solution,independent):close(a,b,'Tail weighted solution agrees with scalar normal equations')
    residual=math.fsum(w*(math.fsum(a*b for a,b in zip(row,independent))-target)**2
                      for w,row,target in zip(weights,rows,y))
    close(solver['weighted_residual_l2_float64']**2,residual,'Tail solve keeps original STOP denominator mass')

    # Reuse generated DTO construction, not any actual/cache-derived feature.
    policy,traces,manifest=generated_fixture();corpus=corpus_for(policy,traces,manifest)
    fitted=fit_continuation_value(policy,corpus)
    direct=calibrate_continuation_readout(policy,corpus,fitted)
    before=(parameter_hash(policy),parameter_hash(fitted._head),parameter_hash(direct._head),
            semantic_digest(direct.report),corpus.seal,parameter_hash(direct._head[0]))
    tail=fit_continuation_tail(policy,corpus,direct,max_scores=1)
    need(before==(parameter_hash(policy),parameter_hash(fitted._head),parameter_hash(direct._head),
                  semantic_digest(direct.report),corpus.seal,parameter_hash(tail._head[0])),
         'Base, original64, direct-calibrated head, hidden layer and corpus unchanged')
    rewards,targets,subjects,declaration=_bound_targets(corpus)
    independent_targets=[]
    for record in corpus.metadata['manifest']:
        independent_targets.extend(oracle.recorded_tail_targets([float(d['reward']) for d in record['decisions']]))
    for a,b in zip(targets,independent_targets):close(a,b,'Actual wrapper uses independently derived tail labels')
    need(np.array_equal(np.asarray(rewards+targets,dtype=np.float32),corpus.targets),'Exact original one-rounding Q join')
    need(declaration['tail_precision']=='float64_manifest_recurrence','Manifest precision declared')
    with torch.no_grad():prediction=tail._head(torch.tensor(np.array(corpus.features,copy=True))).ravel().tolist()
    prediction=[0. if s else p for p,s in zip(prediction,corpus.stop_rows)]
    loss=oracle.weighted_residual(prediction,targets.tolist(),corpus.weights.tolist(),corpus.stop_rows.tolist())
    close(loss,tail.report['raw_tail_weighted_mse'],'Independent deployed raw-tail weighted residual')
    reconstructed=math.fsum(float(w)*(float(r)+max(0.,p)-float(q))**2
        for r,p,q,w,s in zip(rewards,prediction,corpus.targets,corpus.weights,corpus.stop_rows) if not s)
    close(reconstructed,tail.report['reconstructed_Q_weighted_mse'],'Reward plus clipped tail evaluated once')
    need(tail.report['costs']['svd_calls_completed']==1 and tail.report['costs']['head_optimizer_steps']==0
         and tail.report['costs']['head_backward_calls']==0 and tail.report['costs']['base_policy_forwards']==0,
         'One solve, zero tail optimizer/base work')
    trace=traces[0][2]
    try:tail.score_actions(trace.transitions[0].observation,trace.context)
    except TypeError:pass
    else:raise AssertionError('Raw tail must refuse the direct-Q scoring API')
    need(tail.costs['policy_forwards_started']==0 and tail.costs['head_forwards']==0
         and tail.costs['score_calls']==0,'Refused direct-Q API incurs no policy forward')
    scores=tail.score_tails(trace.transitions[0].observation,trace.context)
    need(scores[0]==0. and all(math.isfinite(v) for v in scores),'Tail inference preserves STOP0')
    return {'reward_sequences':count,'normal_equation_solution':independent,'SVD_solution':solution.tolist(),
        'tail_weighted_residual':loss,'reconstructed_Q_weighted_residual':reconstructed,
        'generated_rows':len(targets),'calibration_and_tail_solves':2,'guarded_choice_cases':len(choices),
        'scope':'Generated arithmetic/wrapper only, no patient/cache/weights/native access'}

if __name__=='__main__':
    started=time.monotonic();pins=json.loads((HERE/'reviewed-pins.json').read_text())
    if not pins:raise ValueError('Final source pins required before execution')
    for name,digest in pins.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source pin differs: '+name)
    def timeout(*_):raise TimeoutError('60s generated check cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
    try:r=run();r['status']='PASS';code=0
    except BaseException as e:r={'status':'FAIL','error':type(e).__name__+': '+str(e)};code=1
    signal.alarm(0)
    for name,digest in pins.items():
        need(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Final source unchanged')
    r.update(checks=COUNT,seconds=time.monotonic()-started,pins=pins,
             source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(r,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n';assert len(raw)<65536
    with (HERE/'controls-01.json').open('xb') as f:f.write(raw)
    print(json.dumps({k:r[k] for k in ('status','checks','seconds')}));sys.exit(code)
