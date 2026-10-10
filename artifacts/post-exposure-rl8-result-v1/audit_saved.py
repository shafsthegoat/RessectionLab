"""Prepared terminal-only stdlib audit; no model, weights, arrays or new replay.

Requires root's exact terminal receipt/result pins before opening attempt files.
Reuses prior saved-record seal and arithmetic checks, retaining every TRAIN slot.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREP = ROOT/'build/post-exposure-learning-rl8-recovery-v1/RL8'
RUN = PREP/'attempt-01'
SUP = PREP/'attempt-01.supervision'
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
STEPS = (1, 14, 1, 13)
RELEASE = '4da18b8bd21e6a54308d0b58b3d234c49c7351b56040e448ac7107a28a773e3e'
INDEX = '636f4b75ae95afa2a54c024882ca3b808246a6938db1fa1c5dd2a33443405589'
HEAD = 'e9a6ce8045a7c819c409d894573fdc5cef1980cd'
IL = ROOT/'build/post-exposure-learning-v1/IL64/attempt-01'
REC = ROOT/'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01'
SEARCH = ROOT/'build/remind-post-exposure-feasibility-v1/attempt-01'
CAP = 8*1024**2
seen = {}
checks = 0

def need(ok, message):
    global checks
    checks += 1
    if not ok:
        raise AssertionError(message)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 512*1024**2:
        raise MemoryError('512 MiB saved-data audit cap')

def raw(path, expected=None):
    path = Path(path)
    need(path.is_relative_to(ROOT) and '..' not in path.parts, 'contained metadata path')
    need(path.suffix in ('.json', '.py'), 'JSON/source only; checkpoint and patient arrays refused')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= CAP, 'bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(CAP+1)
        need(len(data) == info.st_size and len(data) <= CAP, 'bounded stable read')
    finally:
        os.close(fd)
    digest = hashlib.sha256(data).hexdigest()
    key = str(path.relative_to(ROOT))
    need(key not in seen or seen[key]['sha256'] == digest, 'repeat evidence unchanged: '+key)
    need(expected is None or digest == expected, 'exact root/source pin: '+key)
    seen[key] = {'sha256':digest, 'bytes':len(data)}
    return data

def read(path, expected=None):
    return json.loads(raw(path, expected), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def digest(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def close(a, b, label, rel=1e-7, absolute=1e-7):
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=rel,abs_tol=absolute),label)

def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]

def identity(history):
    return [{k:v for k,v in row.items() if k != 'outcome_scope'} for row in history]

def save(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')

def visit(folder, context, parameter=None, updates=None):
    sealed=read(folder/'plan.json'); plan=sealed['plan']
    trace=read(folder/'complete-trace.json'); replay=read(folder/'native-replay.json')
    metrics=replay['metrics']; audit=replay['independent_geometry']; history=metrics['history']
    decisions=trace['decisions']; actions=plan['actions']
    need(digest(plan)==sealed['plan_seal'], 'plan seal')
    need(plan['context_hash']==trace['context_hash']==digest(context),'context join')
    need(plan['history']==trace['metrics']['history'] and identity(plan['history'])==identity(history),'exact nominal/replayed history except scope')
    need(audit['committed_history_hash']==digest(history) and audit['accepted'] is True
         and audit['complete_episode'] is True and audit['geometry']['complete_tool_checked'] is True
         and audit['geometry']['failures']==[], 'complete saved independent certificate')
    need(plan['source_hash']==metrics['source_hash'] and plan['decision_model_hash']==metrics['decision_model_hash'], 'source/world binding')
    need(len(actions)==len(decisions)==len(history)==metrics['steps'] and metrics['terminated'], 'complete saved route')
    need(actions==[d['action_id'] for d in decisions]==[h['action_id'] for h in history], 'complete action identity')
    need(actions[-1]=='STOP' or len(actions)==24, 'STOP or complete horizon')
    need(plan['terminal_reason']==('STOP' if actions[-1]=='STOP' else 'HORIZON'), 'terminal reason')
    need(all(not d['terminated'] for d in decisions[:-1]) and decisions[-1]['terminated'], 'termination only on final decision')
    behavior=decisions[0]['behavior_parameter_hash']
    need(all(d['behavior_parameter_hash']==behavior for d in decisions), 'fixed collection behavior')
    need(digest({'context':trace['context_hash'],'observations':[d['observation_hash'] for d in decisions],
                 'history':trace['metrics']['history'],'behavior_parameter_hash':behavior})==trace['trace_seal'], 'trace seal')
    need(all(d['action_mask'][d['action_ids'].index(d['action_id'])] for d in decisions),'legal choices')
    for d,h in zip(decisions,history):close(d['reward'],h['reward'],'decision reward')
    for field in ('reward','target_removed_mm3','normal_removed_mm3'):
        key='total_reward' if field=='reward' else field
        close(math.fsum(h.get(field,0.) for h in history),metrics[key],'history sum '+field)
        close(metrics[key],audit['outcomes'][key],'certificate outcome '+field)
    if parameter is not None:
        need(plan['parameter_hash']==behavior==parameter and plan['learning_updates']==updates, 'frozen policy/collection lineage')
    target=metrics['supplied_goal_region']
    need(target['fraction_denominator']=='entire_unchanged_supplied_region' and target['target_modified'] is False,
         'full immutable supplied target denominator')
    return {'actions':actions,'steps':len(actions),'terminal':plan['terminal_reason'],
        'public_return':metrics['total_reward'],'target_mm3':metrics['target_removed_mm3'],
        'outside_target_mm3':metrics['normal_removed_mm3'],'outcomes':audit['outcomes'],
        'plan_seal':sealed['plan_seal'],'trace_seal':trace['trace_seal'],
        'source_hash':plan['source_hash'],'decision_model_hash':plan['decision_model_hash'],
        'initial_observation_hash':plan['initial_observation_hash'],'unknowns':metrics.get('unknowns',[])}, decisions

def main(receipt_sha, result_sha):
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60 second saved audit cap')))
    signal.alarm(60)
    for pin in (receipt_sha,result_sha):need(len(pin)==64 and all(c in '0123456789abcdef' for c in pin),'root terminal SHA required')
    receipt=read(SUP/'receipt.json',receipt_sha)
    need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] is True
         and receipt['cleanup_errors']==[] and receipt['final_owned_pids']==[] and receipt['stop_reason'] is None,'terminal complete and clean before attempt reads')
    need(receipt['release_sha256']==RELEASE and receipt['result_sha256']==result_sha,'exact terminal joins')
    release=read(PREP/'root-release.json',RELEASE); index=read(PREP/'source-index.json',INDEX)
    need(release['expected_head']==index['head']==HEAD and release['method']=='RL' and release['TRAIN']==list(TRAIN),'fixed experiment')
    need(receipt['source_index']==release['source_index'] and release['source_index']['sha256']==INDEX,'source inventory joins')
    excluded_payload_pins={}
    for group in ('source_files','metadata_files'):
        for path,pin in index[group].items():
            if Path(path).suffix=='.psckpt':
                # Historical index calls this metadata; it is a weight payload.
                # Retain its declared pin but never open or rehash its bytes.
                excluded_payload_pins[path]=pin
            else:
                raw(ROOT/path,pin)
    result=read(RUN/'result.json',result_sha); worker=read(SUP/'worker-final.json')
    need(worker['canonical_result_sha256']==worker['result_sha256']==result_sha,'worker canonical result')
    need(receipt['elapsed_seconds']<3600 and receipt['sampled_peak_rss_bytes']<3*1024**3
         and receipt['output_bytes']<128*1024**2 and worker['wall_seconds']<3540,'actual resource caps')
    need(result['status']=='complete_matched_TRAIN_endpoint' and result['TRAIN']==list(TRAIN)
         and result['optimizer_updates']=={'IL':0,'RL':8},'four subjects/eight shared scratch RL updates')
    need(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==result['search_calls']==0,'closed roles/private/search')
    protocol=release['learning_protocol']; config=read(RUN/'configuration.json')
    need(config['learning_protocol']==protocol and config['execution_limits']==release['execution_limits'],'exact protocol/limits')
    contexts={s:read(RUN/(s+'-context.json')) for s in TRAIN}
    need(all(contexts[s]['role']=='TRAIN' and contexts[s]['subject']==s for s in TRAIN),'ordered TRAIN contexts')
    il=read(IL/'result.json','eedc4857c184d94913976cb5b59e01341e1b3a8bb0f001af43f8d079e3d87975')
    recovered=read(REC/'result.json','432535613f699604a4b0f3b0aaa0fedebbe932e5ca6b7006beedd66ff823a7ce')
    need(il['status']=='failed_or_unresolved' and recovered['independent_accepted'] and recovered['full_history_equal'],'original IL failure retained with separate accepted recovery')
    need(result['initial_parameter_hash']==il['initial_parameter_hash'],'matched fresh initial tensors by saved hash')
    teachers=[]
    for subject,count in zip(TRAIN,STEPS):
        row,_=visit(RUN/'teachers'/subject,contexts[subject]); old=read(SEARCH/subject/'episode-metrics.json')
        need(row['steps']==count and row['actions']==[h['action_id'] for h in old['history']]
             and row['source_hash']==old['source_hash'] and row['decision_model_hash']==old['decision_model_hash'],'same complete public teacher/world')
        for key,field in [('public_return','total_reward'),('target_mm3','target_removed_mm3'),('outside_target_mm3','normal_removed_mm3')]:close(row[key],old[field],'teacher physical outcomes')
        teachers.append({'subject':subject,**row})
    dynamics=read(RUN/'training-dynamics.json',result['training_dynamics_sha256'])
    need(len(dynamics['updates'])==8,'eight dynamics records')
    previous=result['initial_parameter_hash']; updates=[]; episodes=[]; all_decisions=[]
    for update in range(1,9):
        folder=RUN/'RL'/f'update-{update:02d}'
        saved=read(folder/'update.json'); dynamics_row=dynamics['updates'][update-1]
        need(seen[str((folder/'update.json').relative_to(ROOT))]['sha256']==dynamics_row['update_record_sha256'],'dynamics exact update')
        need(saved['completed_updates']==update and saved['optimizer_updates']==1 and saved['before_parameter_hash']==previous
             and saved['patient_adaptation'] is False,'shared parameter chain')
        contributions=[]
        for subject in TRAIN:
            c=read(folder/subject/'gradient-contribution.json'); row,trace=visit(folder/subject,contexts[subject],previous,update-1)
            need(c['context_hash']==digest(contexts[subject]) and c['trace_seal']==row['trace_seal']
                 and c['plan_seal']==row['plan_seal'] and c['steps']==c['loss_forward_calls']==row['steps'],'contribution joins')
            ds=c['rl_decision_diagnostics']['decisions']; diag=c['rl_decision_diagnostics']
            need(len(ds)==len(trace) and diag['episodes_in_update']==4 and diag['behavior_parameter_hash']==previous,'same-forward diagnostics')
            need(diag['value_weight']==protocol['value_weight'] and diag['entropy_weight']==protocol['entropy_weight'],'objective weights')
            rtg=0.; targets=[]
            for d in reversed(ds):rtg=d['reward']+protocol['gamma']*rtg;targets.append(rtg)
            targets.reverse()
            for i,(d,t,target) in enumerate(zip(ds,trace,targets)):
                need(d['step']==i and d['observation_hash']==t['observation_hash'] and d['action_id']==t['action_id']
                     and d['terminated']==t['terminated'] and d['legal_action_count']==sum(t['action_mask']),'diagnostic trace identity')
                close(d['reward'],t['reward'],'diagnostic reward');close(d['return_to_go'],target,'discounted RTG',1e-12,1e-9)
                need(d['return_to_go_model_dtype']==f32(target),'float32 RTG conversion')
                close(d['detached_advantage'],f32(f32(target)-d['value']),'detached advantage',1e-6,1e-5)
                close(d['actor_discount'],protocol['gamma']**i,'actor discount')
                close(d['actor_score_term'],-d['actor_discount']*d['chosen_log_probability']*d['detached_advantage'],'actor term',2e-6,1e-5)
                close(d['value_squared_error'],d['detached_advantage']**2,'value square',2e-6,1e-5)
                need(d['chosen_log_probability']<=1e-6 and -1e-5<=d['entropy']<=math.log(d['legal_action_count'])+1e-5,'probability/entropy bounds')
                all_decisions.append({'update':update,'subject':subject,**d})
            close(c['return'],targets[0],'episode discounted return')
            actor=math.fsum(d['actor_score_term'] for d in ds)/4
            value=math.fsum(d['value_squared_error'] for d in ds)/(4*len(ds))
            entropy=math.fsum(d['entropy'] for d in ds)/(4*len(ds))
            for field,total in [('actor_loss',actor),('value_loss',value),('entropy',entropy)]:close(c[field],total,'episode reduction '+field,2e-6,1e-5)
            close(c['loss'],actor+protocol['value_weight']*value-protocol['entropy_weight']*entropy,'combined episode objective',3e-6,2e-5)
            contributions.append(c)
            episodes.append({'update':update,'subject':subject,**row,
                'positive_immediate_cuts':sum(d['action_id']!='STOP' and d['reward']>0 for d in ds),
                'positive_cuts_with_negative_RTG':sum(d['action_id']!='STOP' and d['reward']>0 and d['return_to_go']<0 for d in ds),
                'positive_cuts_with_negative_advantage':sum(d['action_id']!='STOP' and d['reward']>0 and d['detached_advantage']<0 for d in ds)})
        need(saved['context_hashes']==[c['context_hash'] for c in contributions] and saved['trace_seals']==[c['trace_seal'] for c in contributions],'all four ordered contributions')
        for field in ('loss','actor_loss','value_loss','entropy','loss_forward_calls'):close(saved[field],math.fsum(c[field] for c in contributions),'shared sum '+field)
        for field in ('before_parameter_hash','after_parameter_hash','gradient_norm_before_clip','module_gradient_norms_before_clip','loss_forward_calls'):
            need(saved[field]==dynamics_row[field],'dynamics join '+field)
        close(saved['loss'],dynamics_row['pre_update_loss'],'loss dynamics')
        previous=saved['after_parameter_hash'];updates.append(saved)
    reload=read(RUN/'checkpoint-reload.json'); checkpoint=result['checkpoints']['RL']
    need(previous==reload['parameter_hash']==checkpoint['parameter_hash'] and reload['sha256']==checkpoint['sha256']
         and reload['exact_parameter_match'] is True and reload['completed_updates']==8,'saved reload/endpoint hash chain; no weights opened')
    endpoints=[]
    for subject,teacher in zip(TRAIN,teachers):
        row,_=visit(RUN/'TRAIN-greedy/RL'/subject,contexts[subject],previous,8); saved=result['TRAIN_greedy'][subject]
        need(saved['complete'] and saved['checkpoint_reloaded'] and saved['actions']==row['actions']
             and saved['plan_seal']==row['plan_seal'],'endpoint result joins')
        for key,field in [('public_return','public_return'),('target_mm3','target_removed_mm3'),('outside_target_mm3','outside_supplied_target_removed_mm3')]:close(row[key],saved[field],'endpoint scalar joins')
        need(row['source_hash']==teacher['source_hash'] and row['decision_model_hash']==teacher['decision_model_hash']
             and row['initial_observation_hash']==teacher['initial_observation_hash'],'same public world and initial actor observation')
        old=read(REC/'native-replay.json' if subject=='ReMIND-045' else IL/'TRAIN-greedy/IL'/subject/'native-replay.json')
        need(old['independent_geometry']['accepted'] is True,'accepted IL route or separate recovery')
        close(row['outcomes']['total_reference_target_mm3'],old['independent_geometry']['outcomes']['total_reference_target_mm3'],'full target denominator match')
        row['SEARCH']={k:teacher[k] for k in ('steps','actions','public_return','target_mm3','outside_target_mm3')}
        row['IL64']={k:old['metrics'][v] for k,v in [('steps','steps'),('public_return','total_reward'),('target_mm3','target_removed_mm3'),('outside_target_mm3','normal_removed_mm3')]}
        endpoints.append({'subject':subject,**row})
    costs=read(RUN/'costs.json')
    loss=sum(u['loss_forward_calls'] for u in updates); forwards=2*loss+29+sum(e['steps'] for e in endpoints)
    need(len(episodes)==32 and result['loss_forward_calls']==loss and result['total_policy_forward_calls']==costs['total_policy_forward_calls']==forwards<=1661,'32 episodes and exact forwards')
    need(result['teacher_logit_forwards']==29 and result['teacher_trace_reuses']==0 and result['checkpoint_loads']==1,'RL readout/cache/reload counts')
    need(result['completed_source_visits']==len(costs['completed_patient_visits'])==44
         and all(v['source_released'] for v in costs['completed_patient_visits']),'44 released source visits')
    need(result['native_preview_entries']==costs['native_budget']['native_preview_entries']<=385440,'preview counts/cap')
    need(costs['native_budget']['blocked_preview_attempts']==0 and costs['native_budget']['failure'] is None,'no native-budget refusal')
    # Scalar logs are checked independently; no new forward or original observation is needed.
    readouts=[]
    for subject,count in zip(TRAIN,STEPS):
        for step in range(count):
            r=read(RUN/'teacher-readout'/subject/f'state-{step:02d}.json');s=r['scores'];mask=r['action_mask'];logits=s['logits']
            legal=[i for i,v in enumerate(mask) if v];label=r['action_ids'].index(r['teacher_action'])
            ranking=sorted(legal,key=lambda i:(-logits[i],i));peak=max(logits[i] for i in legal)
            denominator=math.fsum(math.exp(logits[i]-peak) for i in legal)
            need(r['subject']==subject and r['step']==step and r['endpoint']=='RL8' and mask[label],'readout identity/legal teacher')
            for i,p in enumerate(s['probabilities']):close(p,math.exp(logits[i]-peak)/denominator if mask[i] else 0.,'softmax',2e-6,2e-6)
            close(s['teacher_CE'],math.log(denominator)+peak-logits[label],'teacher CE',2e-6,2e-6)
            need(s['teacher_rank']==ranking.index(label)+1 and s['greedy_action']==r['action_ids'][ranking[0]],'teacher rank/greedy')
            readouts.append({k:v for k,v in r.items() if k not in ('action_ids','action_mask')} | {'scores':{k:v for k,v in s.items() if k not in ('logits','probabilities')}})
    metrics=read(RUN/'endpoint-teacher-metrics.json')
    need(metrics==result['endpoint_teacher_metrics'],'endpoint teacher metric join')
    stops=[r['scores']['teacher_CE'] for r in readouts if r['teacher_action']=='STOP']
    motions=[r['scores']['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
    need(len(stops)==metrics['STOP_count']==4 and len(motions)==metrics['motion_count']==25,'teacher group denominators')
    close(metrics['STOP_mean_NLL'],math.fsum(stops)/4,'STOP mean NLL')
    close(metrics['motion_mean_NLL'],math.fsum(motions)/25,'motion mean NLL')
    close(metrics['balanced_mean_CE'],.5*math.fsum(stops)/4+.5*math.fsum(motions)/25,'balanced teacher CE')
    close(metrics['teacher_action_accuracy'],sum(r['scores']['greedy_action']==r['teacher_action'] for r in readouts)/29,'teacher action accuracy')
    phases=costs['costs']; nested=[k for k in phases if '.action_backward.' in k or k.startswith('offline.endpoint_teacher_logits.')]
    for key,phase in phases.items():
        if 'complete_wall_seconds' in phase:
            need(math.isfinite(phase['complete_wall_seconds']) and phase['complete_wall_seconds']>=0,'finite phase time')
    # Input identities are rechecked; outputs below are compact scalars/metadata only.
    for path,pin in list(seen.items()):raw(ROOT/path,pin['sha256'])
    summary={'status':'PASS_saved_metadata_consistency','checks':checks,'all_four_denominator':list(TRAIN),
        'episodes':32,'updates':8,'teacher_readouts':29,'original_IL_status':il['status'],
        'endpoints':endpoints,'training_episode_returns':{'positive':sum(e['public_return']>0 for e in episodes),
            'zero':sum(e['public_return']==0 for e in episodes),'negative':sum(e['public_return']<0 for e in episodes)},
        'positive_immediate_cuts':sum(e['positive_immediate_cuts'] for e in episodes),
        'positive_cuts_with_negative_RTG':sum(e['positive_cuts_with_negative_RTG'] for e in episodes),
        'positive_cuts_with_negative_advantage':sum(e['positive_cuts_with_negative_advantage'] for e in episodes),
        'gradient_norms':[u['gradient_norm_before_clip'] for u in updates],
        'checkpoint_metadata':checkpoint,'teacher_metrics':result['endpoint_teacher_metrics'],
        'excluded_weight_payload_pins_not_opened':excluded_payload_pins,
        'parent':{k:receipt[k] for k in ('elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids')},
        'actual_counts':{'forwards':forwards,'loss_forwards':loss,'source_visits':44,'native_previews':result['native_preview_entries']},
        'costs':{'worker_seconds':costs['complete_wall_seconds'],'recorded_phases':phases,
            'nested_not_additive':nested,'scope':'Reconstruction is separate; collection/replay includes nested backward; readout includes nested logits. Do not sum nested timers or infer selection-only runtime.'},
        'limits':['TRAIN fitted outcomes only; no SELECT/EVAL or clinical result.',
            'Saved full-tool certificates checked, not recomputed geometry; no arrays or checkpoint bytes opened.',
            'Matched public world with unequal search/actor observation bandwidth and unequal compute.',
            'Post-exposure assumptions, partial support domain and external workspace unknowns remain.',
            'Outside-target removal/contact upper bounds are not injury or surgical safety.',
            'Negative RTG on a positive cut is full sampled continuation credit, not automatically a gradient bug.'],
        'audit_seconds':time.monotonic()-started,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save('audit-result.json',summary);save('episodes.json',episodes);save('update-summaries.json',updates)
    save('teacher-readouts.json',readouts);save('input-index.json',seen)
    signal.alarm(0)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('costs','endpoints')},sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--result-sha256',required=True)
    args=parser.parse_args();main(args.receipt_sha256,args.result_sha256)
