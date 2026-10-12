"""Independent stdlib arithmetic for the fixed recorded-tail contrast.

This is an evaluator, never a deployed selector or training capability. Inputs
must be authenticated by the caller. Future rewards are offline labels only.
No source arrays, policy weights, cached features, or native tools are imported.
"""
import math


def _finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Finite real scalar required')
    return float(value)


def recorded_tail_targets(rewards):
    """Direct enumeration, excluding the current action; no recurrence reuse.

No suffix beyond the recorded STOP/horizon is invented. Unlike subtraction
from an already float32-rounded Q label, this uses the original reward values.
"""
    values=tuple(_finite(v) for v in rewards)
    if not values:raise ValueError('Nonempty complete recorded episode required')
    return tuple(max([0.]+[math.fsum(values[i+1:j+1])
                           for j in range(i+1,len(values))]) for i in range(len(values)))


def mandatory_action_targets(rewards):
    values=tuple(_finite(v) for v in rewards)
    if not values:raise ValueError('Nonempty complete recorded episode required')
    return tuple(max(math.fsum(values[i:j+1]) for j in range(i,len(values)))
                 for i in range(len(values)))


def audit_choice(ids,legal,rewards,predicted_tail,chosen):
    """Actual-positive guard; add current actual reward exactly once."""
    if not ids or ids[0]!='STOP' or len(set(ids))!=len(ids):raise ValueError('Ordered unique STOP-first catalog')
    if len(ids)!=len(legal) or len(ids)!=len(rewards) or not legal[0] or rewards[0]!=0:
        raise ValueError('Aligned legal/immediate fields with STOP0 required')
    positive=[]
    for i,(enabled,reward) in enumerate(zip(legal,rewards)):
        if type(enabled) is not bool:raise ValueError('Boolean legal mask required')
        if enabled:
            _finite(reward)
            if i and reward>0:positive.append(i)
        elif reward is not None:raise ValueError('Unqueried/illegal reward must be absent')
    if not positive:
        if predicted_tail is not None:raise ValueError('Declared STOP-only inference must be omitted')
        expected=0;combined=None
    else:
        if predicted_tail is None or len(predicted_tail)!=len(ids) or predicted_tail[0]!=0:
            raise ValueError('Aligned finite tail scores and fixed STOP0 required')
        combined=[]
        for i,(enabled,actual,prediction) in enumerate(zip(legal,rewards,predicted_tail)):
            if enabled:
                value=0. if i==0 else actual+max(0.,_finite(prediction))
                combined.append(_finite(value))
            else:
                if prediction is not None:raise ValueError('No illegal predicted action value')
                combined.append(None)
        # max's first occurrence is the original physical inventory order.
        expected=max(positive,key=lambda i:combined[i])
    if chosen!=ids[expected]:raise ValueError('Saved choice differs from exact r+positive-tail stable order')
    return {'chosen':chosen,'actual_positive_count':len(positive),
            'actual_reward':rewards[expected],'combined_scores':combined,
            'reason':'no_actual_positive_STOP' if expected==0 else 'actual_reward_plus_nonnegative_predicted_tail'}


def weighted_residual(predictions,targets,weights,stops):
    if not len(predictions)==len(targets)==len(weights)==len(stops):raise ValueError('Aligned scalar rows')
    result=[]
    for p,y,w,stop in zip(predictions,targets,weights,stops):
        p,y,w=_finite(p),_finite(y),_finite(w)
        if type(stop) is not bool or w<=0:raise ValueError('Positive weight/fixed STOP mask required')
        if stop and (p!=0 or y!=0):raise ValueError('STOP target and deployed prediction fixed0')
        result.append(w*(p-y)**2)
    return math.fsum(result)


def weighted_normal_equation(rows,targets,weights):
    """Independent small full-rank solver: scalar pivoted Gaussian elimination.

Only generated well-conditioned fixtures use this solver; never real cached
features or a replacement for the declared production SVD algorithm.
"""
    n=len(rows[0])
    aug=[[math.fsum(w*x[i]*x[j] for x,w in zip(rows,weights)) for j in range(n)]
         +[math.fsum(w*x[i]*y for x,y,w in zip(rows,targets,weights))] for i in range(n)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda i:abs(aug[i][col]));aug[col],aug[pivot]=aug[pivot],aug[col]
        if abs(aug[col][col])<1e-9:raise ValueError('Oracle fixture not well-conditioned/full rank')
        divisor=aug[col][col];aug[col]=[v/divisor for v in aug[col]]
        for i in range(n):
            if i!=col:
                factor=aug[i][col];aug[i]=[v-factor*q for v,q in zip(aug[i],aug[col])]
    return [row[-1] for row in aug]
