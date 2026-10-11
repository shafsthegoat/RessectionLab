"""Unexecuted owned-run helper: frozen29 TRAIN gradients, no optimizer/IO/native.

The existing owner reconstructs/replays the pinned teachers, fills its existing
cache and loads one pinned ranking checkpoint. This helper accepts that boundary;
it neither opens sources/checkpoints nor supplies execution authority.
"""
import math

VERSION='frozen_public_motion_gradient_alignment_v1'
COUNTS={'teacher_states':29,'motion_states':25,'STOP_states':4,
        'policy_forwards':29,'autograd_grad_calls':79,'optimizer_updates':0,
        'checkpoint_loads_in_owner':1,'source_visits_in_owner':4}
VECTOR_PAYLOAD_CAP=64*1024**2


def inspect_frozen_gradients(policy, *, checkpoint_metadata, expected_parameter_hash,
        cache, traces, ranking, saved_readouts, guard, emit):
    """Use only replay-admitted cache traces and the exact saved endpoint.

29 forwards; each motion uses two loss VJPs and one margin VJP, each STOP one
loss VJP:79 autograd.grad calls. Never calls backward, writes .grad, constructs an
optimizer, changes weights, evaluates new actions or emits tensors. Parameter
requires_grad flags are temporarily enabled then restored. Derived float64
gradient-vector sums are diagnostic raw-gradient directions, NOT Adam steps.
"""
    import torch
    from resectionlab.core import semantic_digest
    from resectionlab.spatial_policy import SpatialPolicy,parameter_hash
    from resectionlab.patient_teacher_trace_cache import PatientTeacherTraceCache
    from resectionlab.public_motion_ranking import PublicMotionRankingCorpus,motion_ranking_loss,SUBJECTS,STEPS
    if (type(policy) is not SpatialPolicy or policy.training
            or type(cache) is not PatientTeacherTraceCache
            or type(ranking) is not PublicMotionRankingCorpus
            or tuple(traces) != SUBJECTS or set(saved_readouts) != {f'{s}:{k}' for s,n in zip(SUBJECTS,STEPS) for k in range(n)}):
        raise ValueError('Exact frozen eval policy and ordered four TRAIN teacher inputs required')
    guard();cache_record=cache.record();ranking.require()
    if (not cache_record['complete'] or cache_record['learning_protocol_hash']!=semantic_digest(cache.protocol)
            or cache.protocol['cohort_execution']['il_motion_supervision']['corpus_hash']!=ranking.fingerprint
            or checkpoint_metadata['learning_protocol_hash']!=cache_record['learning_protocol_hash']
            or checkpoint_metadata['method']!='IL' or checkpoint_metadata['completed_updates']!=64
            or checkpoint_metadata['parameter_hash']!=expected_parameter_hash
            or checkpoint_metadata['architecture_hash']!=policy.architecture_hash
            or parameter_hash(policy)!=expected_parameter_hash
            or semantic_digest(checkpoint_metadata['contexts'])!=semantic_digest([
                {'patient_group':traces[s].context.patient_group,'context_hash':traces[s].context.fingerprint} for s in SUBJECTS])):
        raise ValueError('Frozen endpoint, protocol or complete teacher contexts differ')
    for pin,s,n in zip(cache_record['traces'],SUBJECTS,STEPS):
        t=traces[s].require()
        if (pin['subject']!=s or pin['trace_seal']!=t.seal_hash or pin['context_hash']!=t.context.fingerprint
                or pin['steps']!=n or pin['stop_steps']!=1 or len(t.transitions)!=n):
            raise ValueError('Detached diagnostic trace differs from admitted replay cache')
        ranking.require_trace(t)
    all_parameters=tuple(policy.named_parameters())
    if any(p.grad is not None for _,p in all_parameters):
        raise ValueError('Diagnostic needs a fresh loaded checkpoint without gradient buffers')
    named=tuple((n,p) for n,p in all_parameters if n.split('.')[0] in ('encoder','actor','stop'))
    if {n.split('.')[0] for n,p in named}!={'encoder','actor','stop'}:
        raise ValueError('Existing actor/encoder/STOP architecture required')
    dimension=sum(p.numel() for _,p in named)
    # At most54 local component vectors+25 margin vectors+3 corpus components;
    # extra working vectors bounded separately inside the same64MiB allowance.
    if 90*dimension*8>VECTOR_PAYLOAD_CAP:
        raise MemoryError('Fixed diagnostic gradient-vector allowance exceeded')
    parameters=tuple(p for _,p in named);old_flags=tuple(p.requires_grad for _,p in all_parameters)
    partitions={};offset=0
    for name,p in named:
        partitions.setdefault(name.split('.')[0],[]).append((offset,offset+p.numel()));offset+=p.numel()
    zeros=lambda:torch.zeros(dimension,dtype=torch.float64)
    totals={key:zeros() for key in ('ranking','gate','STOP')};internal=[]
    counters={'policy_forwards':0,'autograd_grad_calls':0,'optimizer_updates':0}
    def norm(v):
        value=float(torch.linalg.vector_norm(v))
        if not math.isfinite(value):raise FloatingPointError('Nonfinite diagnostic gradient')
        return value
    def dot(a,b):
        value=float(torch.dot(a,b))
        if not math.isfinite(value):raise FloatingPointError('Nonfinite diagnostic derivative')
        return value
    def summary(v):
        return {'norm':norm(v),'module_norms':{k:math.sqrt(sum(float((v[a:b]*v[a:b]).sum()) for a,b in ranges)) for k,ranges in partitions.items()}}
    def alignment(v,g):
        a,b=norm(v),norm(g);d=dot(v,g)
        return {'dot':d,'cosine':None if not a or not b else d/(a*b),**summary(v)}
    def gradient(loss,retain):
        guard()
        result=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=retain,create_graph=False)
        counters['autograd_grad_calls']+=1
        v=torch.cat([torch.zeros(p.numel(),dtype=torch.float64) if g is None else g.detach().reshape(-1).to(dtype=torch.float64,device='cpu') for p,g in zip(parameters,result)])
        norm(v);guard();return v
    try:
        for name,p in all_parameters:p.requires_grad_(name.split('.')[0] in ('encoder','actor','stop'))
        for subject in SUBJECTS:
            trace=traces[subject];labels=ranking.require_trace(trace)
            for step,(transition,label) in enumerate(zip(trace.transitions,labels)):
                guard();observation=transition.observation;observation.assert_intact()
                logits,value=policy(observation);counters['policy_forwards']+=1
                prior=saved_readouts[f'{subject}:{step}']
                if (prior['subject']!=subject or prior['step']!=step
                        or prior['observation_hash']!=observation.fingerprint
                        or tuple(prior['action_ids'])!=tuple(observation.action_ids)
                        or tuple(prior['action_mask'])!=tuple(bool(v) for v in observation.action_mask)
                        or prior['teacher_action']!=transition.action_id
                        or prior['scores']['logits']!=[float(logits[i].detach()) if m else None for i,m in enumerate(observation.action_mask)]):
                    raise ValueError('Frozen diagnostic forward differs from saved endpoint readout')
                teacher=observation.action_ids.index(transition.action_id)
                moves=[i for i,m in enumerate(observation.action_mask) if m and i]
                row={'subject':subject,'step':step,'observation_hash':observation.fingerprint,
                     'teacher_action':transition.action_id,'legal_motion_count':len(moves)}
                if teacher==0:
                    loss=-logits.log_softmax(-1)[0]*(.5/4)
                    vector=gradient(loss,False);totals['STOP']+=vector
                    row['weighted_loss']={'STOP':float(loss.detach())}
                    internal.append((row,{'STOP':vector},None))
                    del loss,vector
                else:
                    legal=[0,*moves]
                    full=motion_ranking_loss(logits,action_ids=label['action_ids'],action_mask=label['action_mask'],
                        rewards=label['rewards'],teacher_action=transition.action_id)*(.5/25)
                    gate=(torch.logsumexp(logits[legal],0)-torch.logsumexp(logits[moves],0))*(.5/25)
                    chosen=max(moves,key=lambda i:float(logits[i].detach()))
                    margin=logits[teacher]-logits[chosen]
                    whole=gradient(full,True);g_gate=gradient(gate,True);g_margin=gradient(margin,False)
                    # Decompose the actual production total gradient; float64
                    # subtraction avoids a second independently rewritten loss.
                    g_rank=whole-g_gate;totals['ranking']+=g_rank;totals['gate']+=g_gate
                    row.update(chosen_motion=observation.action_ids[chosen],
                        chosen_overall=observation.action_ids[int(logits.detach().argmax())],
                        public_nominal_regret=label['rewards'][teacher]-label['rewards'][chosen],
                        teacher_minus_chosen_logit=float(margin.detach()),
                        weighted_loss={'ranking':float((full-gate).detach()),'gate':float(gate.detach())})
                    internal.append((row,{'ranking':g_rank,'gate':g_gate},g_margin))
                    del full,gate,margin,whole,g_gate,g_rank,g_margin
                del logits,value
                guard()
        if counters!={'policy_forwards':29,'autograd_grad_calls':79,'optimizer_updates':0}:
            raise ValueError('Diagnostic counts differ from the fixed29 states')
        total=totals['ranking']+totals['gate']+totals['STOP'];rows=[]
        for row,components,jacobian in internal:
            row['component_alignment_with_corpus_gradient']={k:alignment(v,total) for k,v in components.items()}
            local=sum(components.values(),zeros());row['whole_state_alignment_with_corpus_gradient']=alignment(local,total)
            if jacobian is not None:
                row['margin_parameter_gradient']=summary(jacobian)
                row['margin_directional_derivative_under_negative_raw_gradient']={
                    **{k:-dot(jacobian,g) for k,g in totals.items()},'full_corpus':-dot(jacobian,total),
                    'same_state_ranking':-dot(jacobian,components['ranking']),
                    'same_state_total':-dot(jacobian,local)}
            rows.append(row)
        cache.record();ranking.require();guard()
        if parameter_hash(policy)!=expected_parameter_hash or any(p.grad is not None for _,p in all_parameters):
            raise ValueError('Frozen checkpoint or gradient buffers changed')
        report={'version':VERSION,'status':'complete_frozen_gradient_readout',
            'parameter_hash':expected_parameter_hash,'architecture_hash':policy.architecture_hash,
            'learning_protocol_hash':cache_record['learning_protocol_hash'],'cache_seal':cache_record['cache_seal'],
            'corpus_hash':ranking.fingerprint,'counts':counters,'rows':rows,
            'aggregate_components':{k:summary(v) for k,v in totals.items()},'aggregate_total':summary(total),
            'component_corpus_alignment':{k:alignment(v,total) for k,v in totals.items()},
            'component_cross_dots':{a:{b:dot(x,y) for b,y in totals.items()} for a,x in totals.items()},
            'gradient_parameter_count':dimension,'estimated_vector_payload_bound_bytes':90*dimension*8,
            'interpretation':'instantaneous raw-gradient directional derivatives at one fixed endpoint; not Adam steps, optimizer dynamics, causal proof or route performance',
            'numerics':'autograd gradients float32 then detached float64 vector sums; ranking component equals production total minus gate',
            'private_or_SELECT_access':False,'source_arrays_written':False,'checkpoint_writes':0}
        emit('gradient-alignment.json',report)
        return report
    finally:
        for (_,p),flag in zip(all_parameters,old_flags):p.requires_grad_(flag)
        # autograd.grad never creates leaf .grad buffers. Do not silently clear
        # unexpected mutations or alter the checkpoint to conceal a failure.
