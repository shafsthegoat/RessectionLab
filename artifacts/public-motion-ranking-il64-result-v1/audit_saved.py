"""Prepared terminal-only saved IL audit, adapted from prior IL64/RL8 audits.

Stdlib JSON/hash/scalar work only. Root must supply terminal receipt/result pins.
No tensors, acquired arrays, imports of project code, model or native replay.
"""
import argparse, hashlib, json, math, os, resource, signal, stat, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PREP=ROOT/'build/public-motion-ranking-il64-v1/IL64'
RUN=PREP/'attempt-01'; SUP=PREP/'attempt-01.supervision'
TRAIN=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
STEPS=(1,14,1,13); PAIRS=(0,2516,0,2454)
RELEASE='37365da5ec1421a6cd9665f1c10011497ee0c79b46bf0f76f766f1039ce7f33d'
INDEX='efe3917d049e3278f2f68c7229f811692050efbc15e6591ac69c4f9eb92028aa'
HEAD='3d6d997e79b7752518984ad4aeb5a0ef22bc2dc3'
CORPUS='sha256:5147c784a1e4aee38ffd2cf95881c40914f9807727d54eeaa4dce618100da0be'
seen={}; checks=0

def need(ok,msg):
    global checks
    checks+=1
    if not ok: raise AssertionError(msg)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024**2:
        raise MemoryError('512 MiB saved audit cap')

def raw(path,pin=None):
    path=Path(path)
    need(path.is_relative_to(ROOT) and '..' not in path.parts and path.suffix in ('.json','.py'),'contained JSON/source only')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        st=os.fstat(fd); need(stat.S_ISREG(st.st_mode) and 0<st.st_size<=8*1024**2,'bounded regular evidence')
        with os.fdopen(fd,'rb',closefd=False) as stream: data=stream.read(8*1024**2+1)
    finally: os.close(fd)
    need(len(data)==st.st_size and len(data)<=8*1024**2,'stable bounded read')
    sha=hashlib.sha256(data).hexdigest(); key=str(path.relative_to(ROOT))
    need(pin is None or sha==pin,'exact input pin '+key)
    need(key not in seen or seen[key]['sha256']==sha,'unchanged repeated evidence '+key)
    seen[key]={'sha256':sha,'bytes':len(data)}
    return data

def read(path,pin=None):
    return json.loads(raw(path,pin),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def ref(row): return read(ROOT/row['path'],row['sha256'])
def digest(x): return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b,msg,tol=1e-7): need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=tol,abs_tol=tol),msg)
def identity(h): return [{k:v for k,v in row.items() if k!='outcome_scope'} for row in h]
def save(name,x):
    with (OUT/name).open('x') as stream: json.dump(x,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')

def visit(folder,context,parameter=None):
    sealed=read(folder/'plan.json'); plan=sealed['plan']; trace=read(folder/'complete-trace.json')
    replay=read(folder/'native-replay.json'); m=replay['metrics']; audit=replay['independent_geometry']
    h=m['history']; d=trace['decisions']; actions=plan['actions']
    need(digest(plan)==sealed['plan_seal'],'complete plan seal')
    need(plan['context_hash']==trace['context_hash']==digest(context),'context join')
    need(plan['history']==trace['metrics']['history'] and identity(plan['history'])==identity(h),'nominal/replay exact apart from scope')
    need(audit['accepted'] is True and audit['complete_episode'] is True and audit['geometry']['complete_tool_checked'] is True
         and audit['geometry']['failures']==[] and audit['committed_history_hash']==digest(h),'saved full geometry certificate')
    need(len(actions)==len(d)==len(h)==m['steps'] and m['terminated'],'complete route length')
    need(actions==[x['action_id'] for x in h]==[x['action_id'] for x in d],'all saved actions')
    need(actions[-1]=='STOP' or len(actions)==24,'STOP or full HORIZON')
    need(plan['terminal_reason']==('STOP' if actions[-1]=='STOP' else 'HORIZON'),'terminal reason')
    need(all(not x['terminated'] for x in d[:-1]) and d[-1]['terminated'],'only last decision terminal')
    behavior=d[0]['behavior_parameter_hash']; need(all(x['behavior_parameter_hash']==behavior for x in d),'fixed behavior')
    need(digest({'context':trace['context_hash'],'observations':[x['observation_hash'] for x in d],
                 'history':trace['metrics']['history'],'behavior_parameter_hash':behavior})==trace['trace_seal'],'trace seal')
    need(all(x['action_mask'][x['action_ids'].index(x['action_id'])] for x in d),'legal recorded choices')
    for x,y in zip(d,h): near(x['reward'],y['reward'],'decision reward')
    for field in ('reward','target_removed_mm3','normal_removed_mm3'):
        key='total_reward' if field=='reward' else field
        near(math.fsum(x.get(field,0.) for x in h),m[key],'history sum '+field)
        near(m[key],audit['outcomes'][key],'certificate sum '+field)
    need(plan['source_hash']==m['source_hash'] and plan['decision_model_hash']==m['decision_model_hash'],'source/world join')
    region=m['supplied_goal_region']
    need(region['fraction_denominator']=='entire_unchanged_supplied_region' and region['target_modified'] is False,'full target denominator')
    if parameter is not None: need(plan['parameter_hash']==behavior==parameter and plan['learning_updates']==64,'reloaded fixed64 policy')
    moves=[r for r in h if r['action_id']!='STOP']; tools=[r['tool_id'] for r in moves]
    row={'subject':context['subject'],'steps':len(actions),'actions':actions,'terminal_reason':plan['terminal_reason'],
         'plan_seal':sealed['plan_seal'],'trace_seal':trace['trace_seal'],'physical_history_hash':digest(identity(h)),
         'source_hash':plan['source_hash'],'decision_model_hash':plan['decision_model_hash'],
         'initial_observation_hash':plan['initial_observation_hash'],'public_return':m['total_reward'],
         'target_removed_mm3':m['target_removed_mm3'],'outside_supplied_target_removed_mm3':m['normal_removed_mm3'],
         'motion_count':len(moves),'complete_tool_path_length_mm':math.fsum(x['complete_tool_path_length_mm'] for x in moves),
         'tool_changes':sum(a!=b for a,b in zip(tools,tools[1:])), 'outcomes':audit['outcomes'],
         'target_denominator':region,'unknowns':m.get('unknowns',[])}
    return row,d,plan

def score(scores,ids,mask,label):
    z=scores['logits']; legal=[i for i,v in enumerate(mask) if v]; moves=[i for i in legal if i]
    need(len(z)==len(ids)==len(mask)==len(scores['probabilities']) and ids[0]=='STOP' and mask[0],'score alignment')
    need(all(z[i] is None and scores['probabilities'][i]==0 for i,v in enumerate(mask) if not v),'masked convention')
    top=max(z[i] for i in legal); denominator=math.fsum(math.exp(z[i]-top) for i in legal)
    p=[math.exp(z[i]-top)/denominator if mask[i] else 0. for i in range(len(ids))]
    for a,b in zip(p,scores['probabilities']): near(a,b,'softmax',2**-22)
    near(math.log(denominator)+top-z[label],scores['teacher_CE'],'CE',2e-6)
    order=sorted(legal,key=lambda i:(-z[i],i)); movement=[i for i in order if i]
    need(scores['teacher_rank']==order.index(label)+1 and scores['greedy_action']==ids[order[0]],'exact rank and greedy')
    need(scores['teacher_movement_rank']==(None if label==0 else movement.index(label)+1),'motion rank')
    near(scores['teacher_probability'],p[label],'teacher probability',2**-22)
    near(scores['stop_probability'],p[0],'STOP probability',2**-22)
    near(scores['teacher_minus_STOP'],z[label]-z[0],'teacher margin',2e-6)
    if moves: near(scores['best_movement_minus_STOP'],max(z[i] for i in moves)-z[0],'best movement margin',2e-6)
    if label:
        peak=max(z[i] for i in moves)
        conditional=math.exp(z[label]-peak)/math.fsum(math.exp(z[i]-peak) for i in moves)
        near(scores['conditional_teacher_movement_probability'],conditional,'conditional motion probability',2**-22)
    else: need(scores['conditional_teacher_movement_probability'] is None,'STOP conditional probability absent')

def rank_metrics(scores,labels):
    z=scores['logits']; r=labels['rewards']; legal=[i for i,v in enumerate(labels['action_mask']) if v]
    motion=[i for i in legal if i]; best=max(legal,key=lambda i:z[i]); bm=max(motion,key=lambda i:z[i]) if motion else None
    # Independently enumerate unordered pairs, then orient each by public reward.
    pairs=[]
    if labels['teacher_action']!='STOP':
        for n,i in enumerate(motion):
            for j in motion[n+1:]:
                if r[i]!=r[j]: pairs.append((i,j) if r[i]>r[j] else (j,i))
    def lse(indices):
        peak=max(z[i] for i in indices); return peak+math.log(math.fsum(math.exp(z[i]-peak) for i in indices))
    gate=lse(legal)-lse(motion) if motion else 0.
    scale=max([1.]+[abs(r[i]) for i in motion]); gaps=[r[i]/scale-r[j]/scale for i,j in pairs]
    total=math.fsum(gaps)
    soft=lambda x:max(x,0.)+math.log1p(math.exp(-abs(x)))
    pair_loss=math.fsum(w*soft(z[j]-z[i]) for w,(i,j) in zip(gaps,pairs))/total if total else 0.
    correct=sum(z[i]>z[j] for i,j in pairs); reverse=sum(z[i]<z[j] for i,j in pairs)
    return {'public_nominal_regret':max(r[i] for i in legal)-r[best],
            'conditional_motion_regret':None if bm is None else max(r[i] for i in motion)-r[bm],
            'strict_motion_pairs':len(pairs),'correct_pairs':correct,'reversed_pairs':reverse,'logit_tied_pairs':len(pairs)-correct-reverse,
            'motion_gate':gate,'normalized_pair_loss':pair_loss,
            'ranking_objective':scores['teacher_CE'] if labels['teacher_action']=='STOP' else gate+pair_loss}

def compare_scalars(actual,expected,label):
    need(set(actual)==set(expected),label+' keys')
    for key,value in expected.items():
        if value is None: need(actual[key] is None,label+' '+key)
        elif type(value) is int: need(actual[key]==value,label+' '+key)
        else: near(actual[key],value,label+' '+key)

def main(receipt_sha,result_sha):
    start=time.monotonic(); signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s saved audit cap'))); signal.alarm(60)
    for value in (receipt_sha,result_sha): need(len(value)==64 and all(x in '0123456789abcdef' for x in value),'root terminal pin required')
    receipt=read(SUP/'receipt.json',receipt_sha)
    need(receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'],'owned child terminal clean')
    need(receipt['release_sha256']==RELEASE and receipt['result_sha256']==result_sha,'root terminal joins')
    release=read(PREP/'root-release.json',RELEASE); index=read(PREP/'source-index.json',INDEX)
    need(index['head']==release['expected_head']==HEAD and release['method']=='IL' and release['TRAIN']==list(TRAIN),'fixed source and TRAIN endpoint')
    need(receipt['source_index']==release['source_index'] and release['source_index']['sha256']==INDEX,'source inventory joins')
    for group in ('source_files','metadata_files'):
        for path,pin in index[group].items(): raw(ROOT/path,pin)
    result=read(RUN/'result.json',result_sha); worker=read(SUP/'worker-final.json')
    need(worker['canonical_result_sha256']==result_sha,'worker result join')
    need(receipt['elapsed_seconds']<960 and receipt['sampled_peak_rss_bytes']<3*1024**3 and receipt['output_bytes']<128*1024**2,'owned caps')
    if receipt['status']!='complete':
        need(receipt['status']==result['status']==worker['status']=='failed_or_unresolved','failure retained')
        save('audit-result.json',{'status':'PASS_saved_failure_no_endpoint_acceptance','checks':checks,'result':result,'receipt':receipt,
            'scope':'No failed/partial endpoint is admitted; scientific comparisons require terminal complete evidence.'})
        save('input-index.json',seen); return
    need(receipt['exit_code']==0 and receipt['stop_reason'] is None and worker['wall_seconds']<900,'completed resource boundary')
    control=read(SUP/'endpoint-control.json',receipt['endpoint_control_sha256'])
    need(receipt['worker_final_sha256']==seen[str((SUP/'worker-final.json').relative_to(ROOT))]['sha256']
         and worker['endpoint_control_sha256']==receipt['endpoint_control_sha256'],'final receipt control')
    old={k:ref(v) for k,v in release['baseline_inputs'].items()}; teachers_in={k:ref(v) for k,v in release['inputs'].items()}
    corpus=ref(release['ranking_inputs']['public-score-corpus.json']); need(digest(corpus)==CORPUS,'exact public corpus')
    labels={c['subject']:c['decisions'] for c in corpus['subjects']}
    protocol=release['learning_protocol']; projected=json.loads(json.dumps(protocol)); projected['cohort_execution'].pop('il_motion_supervision')
    need(projected==old['release']['learning_protocol'],'only ranking supervision protocol change')
    need(old['result']['status']=='failed_or_unresolved' and old['recovery_result']['independent_accepted'] is True
         and old['recovery_result']['full_history_equal'] is True,'failed original plus separate accepted recovery')
    need(result['status']=='complete_matched_TRAIN_endpoint' and result['TRAIN']==list(TRAIN),'all four complete')
    need(result['optimizer_updates']=={'IL':64,'RL':0} and result['initial_parameter_hash']==old['result']['initial_parameter_hash'],'64 updates, original init')
    need(result['same_initial_parameters_as_original_IL'] and result['fresh_common_initialization_verified'],'saved common initialization')
    need(result['loss_forward_calls']==1856 and result['teacher_logit_forwards']==result['teacher_decisions']==29
         and result['teacher_trace_reuses']==256 and result['checkpoint_loads']==1 and result['completed_source_visits']==8,'fixed work counts')
    need(result['confirmed_ranking_pair_terms']==318080 and result['endpoint_pair_evaluations']==9940
         and result['public_motion_ranking_corpus_hash']==CORPUS,'ranking count/digest')
    need(result['search_calls']==result['private_reference_reads']==0 and result['SELECT_EVAL_opened'] is False,'no search/private/heldout')
    need(result['selection_readiness']['ready'] is True and result['selection_readiness']['execution_admitted'] is False,'completion without future execution grant')
    config=read(RUN/'configuration.json'); need(config['learning_protocol']==protocol and digest(protocol)==release['learning_protocol_hash'],'configuration/protocol')
    reload=read(RUN/'checkpoint-reload.json'); final=result['checkpoints']['IL']['parameter_hash']
    need(reload['parameter_hash']==final and reload['exact_parameter_match'] and reload['completed_updates']==64
         and reload['sha256']==result['checkpoints']['IL']['sha256'],'checkpoint metadata, no tensor opens')
    contexts={s:read(RUN/(s+'-context.json')) for s in TRAIN}; pins=read(RUN/'teacher-pins.json'); teacher_rows={}; decisions={}
    for s,n in zip(TRAIN,STEPS):
        row,d,plan=visit(RUN/'teachers'/s,contexts[s]); teacher_rows[s]=row; decisions[s]=d
        original=teachers_in['metrics_'+s]; need(row['steps']==n and identity(plan['history'])==identity(original['history']),'same full teacher history')
        need(row['source_hash']==original['source_hash'] and row['decision_model_hash']==original['decision_model_hash'],'same public teacher world')
        need(pins[s]['trace_seal']==row['trace_seal'] and pins[s]['plan_seal']==row['plan_seal'] and pins[s]['steps']==n and pins[s]['stop_steps']==1,'teacher pin')
    dynamics=read(RUN/'training-dynamics.json',result['training_dynamics_sha256']); need(len(dynamics['updates'])==64,'64 dynamics')
    previous=result['initial_parameter_hash']; updates=[]
    for u in range(1,65):
        folder=RUN/'IL'/f'update-{u:02d}'; d=dynamics['updates'][u-1]; row=read(folder/'update.json',d['update_record_sha256'])
        need(row['completed_updates']==d['completed_updates']==u and row['optimizer_updates']==1 and row['loss_forward_calls']==29,'one full shared update')
        need(row['before_parameter_hash']==d['before_parameter_hash']==previous and row['after_parameter_hash']==d['after_parameter_hash'],'parameter chain')
        need(row['teacher_group_counts']=={'STOP':4,'motion':25} and row['teacher_group_weights']=={'STOP':.125,'motion':.02},'balanced reduction')
        need(row['IL_reduction']=='balanced_STOP_gate_public_nominal_motion_gap_ranking_v1' and row['public_motion_ranking_corpus_hash']==CORPUS,'declared ranking objective')
        contributions=[]
        for s,n,p in zip(TRAIN,STEPS,PAIRS):
            c=read(folder/s/'gradient-contribution.json'); contributions.append(c)
            need(c['context_hash']==digest(contexts[s]) and c['trace_seal']==pins[s]['trace_seal'] and c['plan_seal']==pins[s]['plan_seal'],'same trace contribution')
            need(c['steps']==c['loss_forward_calls']==n and c['confirmed_ranking_pair_terms']==p,'contribution work')
        need(row['context_hashes']==[c['context_hash'] for c in contributions] and row['trace_seals']==[c['trace_seal'] for c in contributions],'fixed shared cohort order')
        near(math.fsum(c['loss'] for c in contributions),row['loss'],'contribution loss sum')
        near(row['loss'],d['pre_update_loss'],'dynamics loss'); near(row['gradient_norm_before_clip'],d['gradient_norm_before_clip'],'gradient norm')
        need(d['module_gradient_norms_before_clip']==row['module_gradient_norms_before_clip'] and d['confirmed_ranking_pair_terms']==4970,'module gradients/pairs')
        need(d['clip_threshold']==protocol['max_gradient_norm'] and d['gradient_exceeds_clip_threshold']==(row['gradient_norm_before_clip']>protocol['max_gradient_norm']),'clip accounting')
        if u==1: need(d['loss_change_from_previous'] is None,'initial loss delta absent')
        else: near(d['loss_change_from_previous'],d['pre_update_loss']-updates[-1]['pre_update_loss'],'loss delta')
        previous=row['after_parameter_hash']; updates.append(d)
    need(previous==final,'final64 parameter identity')
    cache=read(RUN/'teacher-cache.json'); need(cache['complete'] and cache['RL_reuse'] is False and cache['array_bytes']+cache['metadata_json_bytes']<=256*1024**2,'bounded complete IL cache')
    need([t['subject'] for t in cache['traces']]==list(TRAIN) and sum(t['steps'] for t in cache['traces'])==29,'cache denominator')
    readouts=[]; pair_metrics={'ranking':[],'original':[]}
    for s,n in zip(TRAIN,STEPS):
        for i in range(n):
            row=read(RUN/'teacher-readout'/s/f'state-{i:02d}.json'); oldrow=old[f'{s}:{i}']; label=labels[s][i]; d=decisions[s][i]
            need(row['teacher_action']==oldrow['teacher_action']==d['action_id']==label['teacher_action'],'same teacher action')
            need(row['observation_hash']==oldrow['observation_hash']==d['observation_hash']==label['observation_hash'],'same state')
            need(row['action_ids']==oldrow['action_ids']==d['action_ids']==label['action_ids'] and row['action_mask']==oldrow['action_mask']==d['action_mask']==label['action_mask'],'same legal inventory')
            need(row['original_IL_scores']==oldrow['scores'],'unchanged original saved logits')
            for endpoint,key,mkey in [('ranking','scores','public_nominal_metrics'),('original','original_IL_scores','original_IL_public_nominal_metrics')]:
                score(row[key],row['action_ids'],row['action_mask'],row['action_ids'].index(row['teacher_action']))
                m=rank_metrics(row[key],label); compare_scalars(row[mkey],m,'independent ranking scalar'); pair_metrics[endpoint].append(m)
            readouts.append({'subject':s,'step':i,'teacher_action':row['teacher_action'],
                **{k:{**{n:v for n,v in row[key].items() if n not in ('logits','probabilities')},**row[mkey]}
                   for k,key,mkey in [('ranking','scores','public_nominal_metrics'),('original','original_IL_scores','original_IL_public_nominal_metrics')]}})
    endpoint=read(RUN/'endpoint-teacher-metrics.json'); need(endpoint==result['endpoint_teacher_metrics'],'endpoint join')
    stops=[r['ranking']['teacher_CE'] for r in readouts if r['teacher_action']=='STOP']; moves=[r['ranking']['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
    need(len(stops)==4 and len(moves)==25,'29-state groups')
    for key,value in {'STOP_mean_NLL':math.fsum(stops)/4,'motion_mean_NLL':math.fsum(moves)/25,
        'balanced_mean_CE':.5*math.fsum(stops)/4+.5*math.fsum(moves)/25,'unweighted_mean_CE':math.fsum(stops+moves)/29,
        'teacher_action_accuracy':sum(r['ranking']['greedy_action']==r['teacher_action'] for r in readouts)/29}.items(): near(endpoint[key],value,'endpoint '+key)
    for name,field in [('ranking','ranking'),('original','original_IL_ranking')]:
        rows=pair_metrics[name]; stopped=[r for r,x in zip(rows,readouts) if x['teacher_action']=='STOP']; moved=[r for r,x in zip(rows,readouts) if x['teacher_action']!='STOP']
        expected={'state_count':29,'STOP_count':4,'motion_count':25,'mean_public_nominal_regret':math.fsum(r['public_nominal_regret'] for r in rows)/29,
            'balanced_ranking_objective':.5*math.fsum(r['ranking_objective'] for r in stopped)/4+.5*math.fsum(r['ranking_objective'] for r in moved)/25,
            **{k:sum(r[k] for r in rows) for k in ('strict_motion_pairs','correct_pairs','reversed_pairs','logit_tied_pairs')}}
        compare_scalars(endpoint[field],expected,'aggregate '+name); need(expected['strict_motion_pairs']==4970,'pairs per endpoint')
    outcomes=[]
    for s in TRAIN:
        row,_,plan=visit(RUN/'TRAIN-greedy'/'IL'/s,contexts[s],final); saved=result['TRAIN_greedy'][s]
        need(saved['complete'] and saved['checkpoint_reloaded'] and saved['parameter_hash']==final,'complete reloaded route')
        for key in ('actions','steps','plan_seal','motion_count','tool_changes','terminal_reason'): need(saved[key]==row[key],'endpoint '+key)
        for key in ('public_return','target_removed_mm3','outside_supplied_target_removed_mm3','complete_tool_path_length_mm'): near(saved[key],row[key],'endpoint '+key)
        original=old['plan_'+s]['plan']; search=teachers_in['metrics_'+s]
        need(digest(original)==old['plan_'+s]['plan_seal'],'original plan seal')
        refs={}
        for name,history in [('original_IL',original['history']),('SEARCH',search['history'])]:
            refs[name]={'steps':len(history),'same_actions':row['actions']==[h['action_id'] for h in history],
                **{k:math.fsum(h.get(f,0.) for h in history) for k,f in [('public_return','reward'),('target_removed_mm3','target_removed_mm3'),('outside_supplied_target_removed_mm3','normal_removed_mm3')]}}
            refs[name]['return_difference']=row['public_return']-refs[name]['public_return']
        row['comparisons']=refs; outcomes.append(row)
    comparison=read(RUN/'original-IL-comparison.json',result['original_IL_comparison_sha256'])
    need(comparison['original_attempt_status']=='failed_or_unresolved' and comparison['ranking_IL_outcomes']==result['TRAIN_greedy'] and comparison['fixed_teacher_metrics']==endpoint,'full original/new comparison')
    near(comparison['original_parent_seconds'],old['parent']['elapsed_seconds'],'original failed cost'); near(comparison['separate_recovery_parent_seconds'],old['recovery_parent']['elapsed_seconds'],'separate recovery cost')
    costs=read(RUN/'costs.json'); forwards=1856+29+sum(r['steps'] for r in outcomes)
    need(forwards==result['total_policy_forward_calls']==costs['total_policy_forward_calls'] and forwards<=1981,'all actual forwards')
    need(len(costs['completed_patient_visits'])==8 and all(v['source_released'] for v in costs['completed_patient_visits']),'eight released visits')
    need(costs['native_budget']['native_preview_entries']==result['native_preview_entries']<=70080 and costs['native_budget']['failure'] is None,'actual native budget')
    need(control['TRAIN_greedy']==result['TRAIN_greedy'] and control['teacher_metrics']==endpoint and control['confirmed_ranking_pair_terms']==318080,'parent endpoint control')
    for path,meta in list(seen.items()): raw(ROOT/path,meta['sha256'])
    summary={'status':'PASS_saved_ranking_IL64_audit','checks':checks,'inputs':len(seen),'result_sha256':result_sha,
        'receipt_sha256':receipt_sha,'release_sha256':RELEASE,'source_index_sha256':INDEX,'initial_parameter_hash':result['initial_parameter_hash'],
        'checkpoint_metadata':result['checkpoints']['IL'],'counts':{'shared_IL_updates':64,'loss_forwards':1856,'teacher_states':29,
            'confirmed_training_pair_terms':318080,'endpoint_pair_evaluations':9940,'source_visits':8,'cache_uses':256,'checkpoint_reloads':1,
            'policy_forwards':forwards,'native_previews':result['native_preview_entries'],'search_calls':0,'RL_updates':0},
        'dynamics':{'first_pre_update_loss':updates[0]['pre_update_loss'],'final_pre_update_loss':updates[-1]['pre_update_loss'],
            'gradient_min':min(d['gradient_norm_before_clip'] for d in updates),'gradient_max':max(d['gradient_norm_before_clip'] for d in updates),
            'clipped_updates':sum(d['gradient_exceeds_clip_threshold'] for d in updates),
            'loss_increases':sum(b['pre_update_loss']>a['pre_update_loss'] for a,b in zip(updates,updates[1:]))},
        'endpoint_metrics':endpoint,'routes':outcomes,'parent':{k:receipt[k] for k in ('elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids')},
        'prior_costs':{'original_failed_IL_seconds':comparison['original_parent_seconds'],'separate_045_recovery_seconds':comparison['separate_recovery_parent_seconds']},
        'cost_scope':'Inclusive nested timings must not be added. Ranking adds pair work; no equal-compute or repeated-run speed claim.',
        'scope':'Saved scalar/hash/complete-history certificate consistency only; no weights, arrays, geometry replay or model execution. Full tensor identity is authenticated recorded evidence, not independently reloaded. Original IL failed status plus separate 045 acceptance retained. TRAIN-only; no transfer or physical validation claim.',
        'audit_seconds':time.monotonic()-start,'audit_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save('update-summary.json',updates); save('teacher-readout-summary.json',readouts); save('audit-result.json',summary); save('input-index.json',seen)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('routes','endpoint_metrics')},sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-receipt-sha256',required=True); parser.add_argument('--terminal-result-sha256',required=True)
    args=parser.parse_args(); main(args.terminal_receipt_sha256,args.terminal_result_sha256)
