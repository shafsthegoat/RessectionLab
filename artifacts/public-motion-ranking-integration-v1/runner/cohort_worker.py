"""One fixed64 public-motion-ranking IL endpoint on unchanged TRAIN teachers."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from pilot_contract import (METHODS,TRAIN,TEACHER_STEPS,TEACHER_STATES,EXPOSURE,PUBLIC_INDEX,PUBLIC_SHA,COHORT,WORKER_SECONDS,
    canonical_configuration,canonical,inputs,sha,source_guard,complete_result,endpoint_control,
    ranking_inputs,RANKING_HASH,RANKING_SPEC,PAIR_COUNTS,PAIR_TERMS_TOTAL,BASELINE_REFS)


def require_teacher_identity(actual,prior,history_identity):
    """Admitted reconstruction may change only context and history scope labels."""
    if (canonical({k:v for k,v in actual.items() if k not in ('context_hash','history')})
            !=canonical({k:v for k,v in prior.items() if k not in ('context_hash','history')})
            or history_identity(actual['history'])!=history_identity(prior['history'])):
        raise ValueError('Saved post-exposure greedy teacher world/actions/history changed')


def route_costs(history):
    moves=[r for r in history if r['action_id']!='STOP']
    tools=[r['tool_id'] for r in moves]
    return {'motion_count':len(moves),'complete_tool_path_length_mm':math.fsum(r['complete_tool_path_length_mm'] for r in moves),
        'tool_changes':sum(a!=b for a,b in zip(tools,tools[1:])),
        'terminal_reason':'STOP' if history[-1]['action_id']=='STOP' else 'HORIZON'}


def state_metrics(scores,labels):
    """Scalar endpoint diagnostics on already-produced logits; no policy call."""
    ids=labels['action_ids'];mask=labels['action_mask'];rewards=labels['rewards'];z=scores['logits']
    if len(z)!=len(ids):raise ValueError('Readout logit/action count changed')
    legal=[i for i,m in enumerate(mask) if m];moves=[i for i in legal if i]
    if any(z[i] is None or not math.isfinite(z[i]) for i in legal):raise ValueError('Finite legal logits required')
    best=max(legal,key=lambda i:z[i]);best_motion=max(moves,key=lambda i:z[i]) if moves else None
    if ids[best]!=scores['greedy_action']:raise ValueError('Readout argmax differs')
    pairs=[] if labels['teacher_action']=='STOP' else [(i,j) for i in moves for j in moves if rewards[i]>rewards[j]]
    correct=sum(z[i]>z[j] for i,j in pairs);reverse=sum(z[i]<z[j] for i,j in pairs)
    def lse(values):
        maximum=max(values);return maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))
    gate=0. if not moves else lse([z[i] for i in legal])-lse([z[i] for i in moves])
    scale=max([1.]+[abs(rewards[i]) for i in moves])
    gaps=[rewards[i]/scale-rewards[j]/scale for i,j in pairs];denom=math.fsum(gaps)
    soft=lambda v:max(v,0.)+math.log1p(math.exp(-abs(v)))
    pair_loss=0. if not pairs else math.fsum(g/denom*soft(z[j]-z[i]) for g,(i,j) in zip(gaps,pairs))
    return {'public_nominal_regret':max(rewards[i] for i in legal)-rewards[best],
        'conditional_motion_regret':None if best_motion is None else max(rewards[i] for i in moves)-rewards[best_motion],
        'strict_motion_pairs':len(pairs),'correct_pairs':correct,'reversed_pairs':reverse,
        'logit_tied_pairs':len(pairs)-correct-reverse,'motion_gate':gate,
        'normalized_pair_loss':pair_loss,'ranking_objective':scores['teacher_CE'] if labels['teacher_action']=='STOP' else gate+pair_loss}


def aggregate_state_metrics(rows,key):
    stop=[r[key]['ranking_objective'] for r in rows if r['teacher_action']=='STOP']
    motion=[r[key]['ranking_objective'] for r in rows if r['teacher_action']!='STOP']
    return {'state_count':len(rows),'STOP_count':len(stop),'motion_count':len(motion),
        'mean_public_nominal_regret':math.fsum(r[key]['public_nominal_regret'] for r in rows)/len(rows),
        'balanced_ranking_objective':.5*math.fsum(stop)/len(stop)+.5*math.fsum(motion)/len(motion),
        **{k:sum(r[key][k] for r in rows) for k in ('strict_motion_pairs','correct_pairs','reversed_pairs','logit_tied_pairs')}}


def execute(release,release_sha,output,progress):
    import numpy as np
    import torch
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget,_save_checkpoint,load_cohort_checkpoint
    from resectionlab.patient_planning_accumulation import PatientGradientAccumulator
    from resectionlab.patient_teacher_trace_cache import PatientTeacherTraceCache
    from resectionlab.patient_planning_learning import PatientTrainSession,common_patient_policies
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.public_motion_ranking import PublicMotionRankingCorpus

    method=release['method'];cfg=METHODS[method]
    protocol,limits=canonical_configuration(method)
    if canonical(protocol)!=canonical(release['learning_protocol']) or canonical(limits)!=canonical(release['cohort_limits']):
        raise ValueError('Canonical endpoint objective/admission differs')
    baseline=inputs()
    factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC_INDEX,
        manifest_index_sha256=PUBLIC_SHA,cohort_bytes=(ROOT/COHORT).read_bytes(),
        learning_protocol=protocol,limits=limits,released_record=release,
        released_sha256=release_sha,progress=progress)
    output.mkdir(exist_ok=False);sink=OutputBudget(output,limits['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=cfg['native_preview_cap'],seconds=cfg['worker_seconds'])
    costs=preflight._CallCosts(budget,cfg['policy_forward_cap']);started=time.perf_counter()
    teachers={};readouts=[];dynamics=[];detached={};cache_reuses=0
    cache=PatientTeacherTraceCache(protocol) if method=='IL' else None
    result={'status':'started','method':method,'TRAIN':list(TRAIN),'SELECT_EVAL_opened':False,
        'post_exposure_condition':EXPOSURE,'fresh_common_initialization_verified':False,'private_reference_reads':0,'optimizer_updates':{'IL':0,'RL':0},'search_calls':0,
        'loss_forward_calls':0,'teacher_logit_forwards':0,'checkpoint_loads':0,
        'teacher_trace_reuses':0,'checkpoints':{},'TRAIN_greedy':{},
        'confirmed_ranking_pair_terms':0,'public_motion_ranking_corpus_hash':RANKING_HASH,
        'teacher_statuses':{s:'not_started' for s in TRAIN},
        'selection_readiness':{'ready':False,'execution_admitted':False}}
    def guard():budget.check();sink.check()
    visits=_SequentialVisits(factories,protocol,limits,sink,costs,guard)
    def directory(*parts):
        path=output.joinpath(*parts);path.mkdir(parents=True,exist_ok=False);return path
    def teacher_replay(base,context,dest,subject):
        prior=baseline['plan_'+subject]['plan']
        trace=preflight._collect(base,context,actions=prior['actions'],output=dest,guard=guard)
        sealed=preflight._seal_and_replay(base,context,trace,method='SEARCH',policy=None,updates=0,output=dest,guard=guard)
        actual=thaw_json(sealed['plan'])
        require_teacher_identity(actual,prior,preflight._history_identity)
        return trace,sealed
    def score(policy,observation,action):
        with torch.no_grad():logits,value=policy(observation)
        raw=logits.detach().cpu().numpy();legal=np.flatnonzero(observation.action_mask)
        probabilities=logits.softmax(-1);logp=logits.log_softmax(-1)
        ranking=sorted(legal,key=lambda i:(-float(raw[i]),int(i)))
        movement=[int(i) for i in ranking if i];label=observation.action_ids.index(action)
        best=movement[0] if movement else None
        return {'logits':[float(raw[i]) if observation.action_mask[i] else None for i in range(len(raw))],
            'probabilities':probabilities.tolist(),'teacher_CE':float(-logp[label]),
            'teacher_probability':float(probabilities[label]),'stop_probability':float(probabilities[0]),
            'teacher_rank':ranking.index(label)+1,'teacher_movement_rank':None if label==0 else movement.index(label)+1,
            'conditional_teacher_movement_probability':None if label==0 else float(logits[1:].softmax(-1)[label-1]),
            'greedy_action':observation.action_ids[ranking[0]],'teacher_minus_STOP':float(raw[label]-raw[0]),
            'best_movement_minus_STOP':None if best is None else float(raw[best]-raw[0]),'value':float(value)}
    def read_teacher_rows(trace_rows,context,subject,dest,reloaded,frozen):
        context.require_observations(row.observation for row in trace_rows)
        binding=semantic_digest([[r.observation.fingerprint,r.action_id,r.reward,r.terminated] for r in trace_rows])
        if binding!=teachers[subject]['readout_binding_hash']:raise ValueError('Teacher readout binding changed')
        rows=[]
        for step,row in enumerate(trace_rows):
            guard()
            with costs.scope('offline.endpoint_teacher_logits.'+subject):new=score(reloaded,row.observation,row.action_id)
            old=original_IL[f'{subject}:{step}']
            label=labels[subject][step]
            if (old['observation_hash']!=row.observation.fingerprint or old['action_ids']!=list(row.observation.action_ids)
                    or old['action_mask']!=row.observation.action_mask.tolist() or old['teacher_action']!=row.action_id):
                raise ValueError('Original IL endpoint readout state changed')
            item={'public_nominal_metrics':state_metrics(new,label),
                'original_IL_scores':old['scores'],'original_IL_public_nominal_metrics':state_metrics(old['scores'],label),
                'subject':subject,'step':step,'teacher_action':row.action_id,
                'observation_hash':row.observation.fingerprint,'action_ids':list(row.observation.action_ids),
                'action_mask':row.observation.action_mask.tolist(),'endpoint':method+str(cfg['updates']),'scores':new}
            sink.write(dest/f'state-{step:02d}.json',item);rows.append(item)
        if parameter_hash(reloaded)!=frozen:raise ValueError('Readout changed endpoint weights')
        return rows
    sink.write(output/'configuration.json',{'learning_protocol':protocol,'admission_limits':limits,'execution_limits':cfg,
        'initialization':'same canonical fresh initializer; unused identical clone discarded',
        'preview_sizing':'source visits * 120 candidates * (1 + 3 * horizon24); conservative, not measured',
        'fixed_teachers':'29 complete replayed saved public greedy states:4 STOP+25 motion',
        'objective':'0.5 mean(motion gate + normalized public reward-gap ranking) + 0.5 mean STOP NLL',
        'ranking_supervision':RANKING_SPEC,'pair_terms_per_update':4970,'pair_terms_fixed64':PAIR_TERMS_TOTAL,
        'comparison':'separate fixed endpoints; actual method costs differ; no intermediate selection or held-out execution'})
    try:
        with budget,costs,_bounded_writes(sink):
            with costs.scope('offline.public_ranking_label_validation'):
                corpus_record,original_IL=ranking_inputs()
                ranking=PublicMotionRankingCorpus.admit(corpus_record,expected_hash=RANKING_HASH)
                labels={case['subject']:case['decisions'] for case in corpus_record['subjects']}
            for subject in TRAIN:
                dest=directory('teachers',subject)
                def admit(base,context):
                    trace,sealed=teacher_replay(base,context,dest,subject)
                    if cache is not None:
                        cache.add_replayed(trace,sealed);detached[subject]=trace.transitions
                    return {'readout_binding_hash':semantic_digest([[r.observation.fingerprint,r.action_id,r.reward,r.terminated] for r in trace.transitions]),
                        'context_hash':context.fingerprint,'steps':len(trace.transitions),
                        'stop_steps':sum(r.action_id=='STOP' for r in trace.transitions),'trace_seal':trace.seal_hash,
                        'plan_seal':sealed['plan_seal'],'actions':[r.action_id for r in trace.transitions]}
                teachers[subject]=visits.run(subject,dest,'offline.fixed_teacher_replay.'+subject,admit)
                result['teacher_statuses'][subject]='complete_replayed'
                if teachers[subject]['steps']!=TEACHER_STEPS[subject]:raise ValueError('Exact fixed teacher length changed')
            if sum(t['steps'] for t in teachers.values())!=TEACHER_STATES or sum(t['stop_steps'] for t in teachers.values())!=4:
                raise ValueError('29 fixed labels/four STOP required')
            result['teacher_decisions']=sum(t['steps'] for t in teachers.values())
            result['teacher_steps']={s:teachers[s]['steps'] for s in TRAIN}
            if cache is not None:sink.write(output/'teacher-cache.json',cache.record())
            contexts=tuple(visits.contexts[s] for s in TRAIN)
            with costs.scope('offline.common_initialization'):
                il,rl=common_patient_policies(contexts,protocol);policy=il if method=='IL' else rl;del il,rl
                initial=parameter_hash(policy);result['initial_parameter_hash']=initial
                if initial!=original_IL['result']['initial_parameter_hash']:raise ValueError('Original IL initialization changed')
                result['same_initial_parameters_as_original_IL']=True
                result['fresh_common_initialization_verified']=True
                session=PatientTrainSession(contexts,method,policy,initial_parameter_hash=initial,protocol=protocol)
            pins={t['context_hash']:{k:t[k] for k in ('steps','stop_steps','trace_seal')} for t in teachers.values()}
            sink.write(output/'teacher-pins.json',teachers)
            generator=torch.Generator(device='cpu').manual_seed(protocol['seed']+1)
            for update in range(cfg['updates']):
                if session.updates!=update:raise ValueError('Unexpected shared update count')
                dest=directory(method,f'update-{update+1:02d}')
                with PatientGradientAccumulator(session,teacher_pins=pins if method=='IL' else None,rl_diagnostics=False,motion_ranking=ranking) as accumulation:
                    for subject in TRAIN:
                        visit=directory(method,f'update-{update+1:02d}',subject)
                        if method=='IL':
                            with costs.scope(f'offline.IL.update-{update+1:02d}.cached_loss_backward.'+subject):
                                trace=cache.trace_for_il(session,subject);details=accumulation.add_trace(trace,guard=guard);del trace
                            cache_reuses+=1
                            result['confirmed_ranking_pair_terms']+=PAIR_COUNTS[subject]
                            details={**details,'confirmed_ranking_pair_terms':PAIR_COUNTS[subject]}
                            details={**details,'plan_seal':teachers[subject]['plan_seal'],'observation_storage':protocol['cohort_execution']['teacher_observations']}
                        else:
                            def contribute(base,context):
                                trace=preflight._collect(base,context,policy=policy,generator=generator,output=visit,guard=guard)
                                sealed=preflight._seal_and_replay(base,context,trace,method='RL_COLLECTION',policy=policy,
                                    updates=session.updates,output=visit,guard=guard)
                                with costs.scope(f'offline.RL.update-{update+1:02d}.action_backward.'+subject):
                                    row=accumulation.add_trace(trace,guard=guard)
                                return {**row,'plan_seal':sealed['plan_seal'],'actions':[r.action_id for r in trace.transitions]}
                            details=visits.run(subject,visit,f'offline.RL.update-{update+1:02d}.collection_replay.'+subject,contribute)
                        sink.write(visit/'gradient-contribution.json',details)
                    with costs.scope(f'offline.{method}.update-{update+1:02d}.shared_optimizer_step'):
                        row=accumulation.finish(guard=guard)
                    if (row['loss_forward_calls']!=TEACHER_STATES or row.get('public_motion_ranking_corpus_hash')!=RANKING_HASH
                            or row['IL_reduction']!='balanced_STOP_gate_public_nominal_motion_gap_ranking_v1'):
                        raise ValueError('IL update must use all29 exact ranking labels')
                    result['loss_forward_calls']+=row['loss_forward_calls'];result['optimizer_updates'][method]=session.updates
                    sink.write(dest/'update.json',row)
                    dynamics.append({'confirmed_ranking_pair_terms':sum(PAIR_COUNTS.values()),'completed_updates':row['completed_updates'],'pre_update_loss':row['loss'],
                        'loss_change_from_previous':None if not dynamics else row['loss']-dynamics[-1]['pre_update_loss'],
                        'gradient_norm_before_clip':row['gradient_norm_before_clip'],'module_gradient_norms_before_clip':row['module_gradient_norms_before_clip'],
                        'clip_threshold':protocol['max_gradient_norm'],'gradient_exceeds_clip_threshold':row['gradient_norm_before_clip']>protocol['max_gradient_norm'],
                        'before_parameter_hash':row['before_parameter_hash'],'after_parameter_hash':row['after_parameter_hash'],
                        'loss_forward_calls':row['loss_forward_calls'],'update_record_sha256':sha(dest/'update.json')})
                    guard()
            if result['loss_forward_calls']>cfg['loss_forward_cap']:raise ValueError('Actual loss-forward envelope exceeded')
            sink.write(output/'training-dynamics.json',{'updates':dynamics,
                'scope':'pre-update diagnostics only; every fixed update runs, no intermediate endpoint or loss stopping rule'})
            result['training_dynamics_sha256']=sha(output/'training-dynamics.json')
            with costs.scope('offline.'+method+'.checkpoint_save_reload'):
                pin=_save_checkpoint(policy,method=method,contexts=contexts,protocol=protocol,updates=session.updates,initial_hash=initial,output=sink,limits=limits)
                result['checkpoints'][method]={k:pin[k] for k in ('path','sha256','bytes','parameter_hash')}
                reloaded,metadata=load_cohort_checkpoint(output/pin['path'],expected_sha256=pin['sha256'],expected_learning_protocol=protocol,
                    expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},expected_method=method)
                result['checkpoint_loads']=1
                if parameter_hash(reloaded)!=parameter_hash(policy) or metadata['completed_updates']!=cfg['updates']:
                    raise ValueError('Reloaded fixed endpoint tensors differ')
                reloaded.eval();reloaded.requires_grad_(False);frozen=parameter_hash(reloaded)
                sink.write(output/'checkpoint-reload.json',{'sha256':pin['sha256'],'parameter_hash':frozen,
                    'exact_parameter_match':True,'completed_updates':cfg['updates']})
            # No optimizer/session is carried into endpoint deployment.
            del accumulation,session,policy
            for subject in TRAIN:
                dest=directory('teacher-readout',subject)
                if cache is not None:
                    cache.record()
                    with costs.scope('offline.endpoint_cached_teacher_readout.'+subject):
                        rows=read_teacher_rows(detached[subject],visits.contexts[subject],subject,dest,reloaded,frozen)
                else:
                    def readout(base,context):
                        trace,sealed=teacher_replay(base,context,dest,subject)
                        return read_teacher_rows(trace.transitions,context,subject,dest,reloaded,frozen)
                    rows=visits.run(subject,dest,'offline.endpoint_teacher_recollection.'+subject,readout)
                readouts.extend(rows);result['teacher_logit_forwards']+=len(rows)
            if cache is not None:result['teacher_cache']={**cache.record(),'trace_reuses':cache_reuses,'readout_rows':len(readouts)}
            result['teacher_trace_reuses']=cache_reuses
            del cache;detached.clear()
            stop=[r['scores']['teacher_CE'] for r in readouts if r['teacher_action']=='STOP']
            motion=[r['scores']['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
            endpoint={'STOP_count':len(stop),'motion_count':len(motion),'STOP_mean_NLL':sum(stop)/len(stop),
                'motion_mean_NLL':sum(motion)/len(motion),'unweighted_mean_CE':(sum(stop)+sum(motion))/TEACHER_STATES,
                'balanced_mean_CE':.5*sum(stop)/len(stop)+.5*sum(motion)/len(motion),
                'teacher_action_accuracy':sum(r['scores']['greedy_action']==r['teacher_action'] for r in readouts)/TEACHER_STATES}
            endpoint.update(ranking=aggregate_state_metrics(readouts,'public_nominal_metrics'),
                original_IL_ranking=aggregate_state_metrics(readouts,'original_IL_public_nominal_metrics'))
            result['endpoint_pair_evaluations']=endpoint['ranking']['strict_motion_pairs']+endpoint['original_IL_ranking']['strict_motion_pairs']
            if result['endpoint_pair_evaluations']!=cfg['endpoint_pair_evaluations']:raise ValueError('Endpoint pair accounting differs')
            sink.write(output/'endpoint-teacher-metrics.json',endpoint);result['endpoint_teacher_metrics']=endpoint
            for subject in TRAIN:
                dest=directory('TRAIN-greedy',method,subject)
                def greedy(base,context):
                    trace=preflight._collect(base,context,policy=reloaded,output=dest,guard=guard)
                    sealed=preflight._seal_and_replay(base,context,trace,method=method,policy=reloaded,updates=cfg['updates'],output=dest,guard=guard)
                    if parameter_hash(reloaded)!=frozen:raise ValueError('Greedy replay changed reloaded endpoint')
                    # Completion is geometry/history validity, never teacher-route or reward agreement.
                    route=route_costs(sealed['replayed_history'])
                    return {**route,'complete':True,'checkpoint_reloaded':True,'parameter_hash':frozen,'plan_seal':sealed['plan_seal'],
                        'steps':len(trace.transitions),'actions':[t.action_id for t in trace.transitions],
                        'public_return':sum(t.reward for t in trace.transitions),
                        'target_removed_mm3':sum(r.get('target_removed_mm3',0.) for r in sealed['replayed_history']),
                        'outside_supplied_target_removed_mm3':sum(r.get('normal_removed_mm3',0.) for r in sealed['replayed_history']),
                        'outside_interpretation':'outside supplied target, not verified normal tissue'}
                result['TRAIN_greedy'][subject]=visits.run(subject,dest,'deployment.reloaded_TRAIN_greedy_replay.'+subject,greedy)
            expected_forwards=result['loss_forward_calls']*(1 if method=='IL' else 2)+TEACHER_STATES+sum(r['steps'] for r in result['TRAIN_greedy'].values())
            if (len(visits.completed)!=cfg['source_visits'] or cache_reuses!=cfg['teacher_trace_reuses']
                    or result['teacher_logit_forwards']!=TEACHER_STATES or costs.forwards!=expected_forwards):
                raise ValueError('Actual visit/forward accounting differs from declared flow')
            if result['confirmed_ranking_pair_terms']!=PAIR_TERMS_TOTAL:raise ValueError('Fixed pair work differs')
            comparison={'original_attempt_status':original_IL['result']['status'],
                'original_and_recovery_pins':{k:BASELINE_REFS[k] for k in ('result','parent','recovery_result','recovery_parent','outcomes')},
                'original_parent_seconds':original_IL['parent']['elapsed_seconds'],
                'separate_recovery_parent_seconds':original_IL['recovery_parent']['elapsed_seconds'],
                'original_IL_outcomes':[{**r,**route_costs(original_IL['plan_'+r['subject']]['plan']['history'])} for r in original_IL['outcomes']],'ranking_IL_outcomes':result['TRAIN_greedy'],
                'fixed_teacher_metrics':endpoint,'same_initial_parameters':True,
                'SELECT_baseline':'separate run; no SELECT data used in this loss or endpoint selection',
                'scope':'retain every route, including negative or different valid actions; no performance gate'}
            sink.write(output/'original-IL-comparison.json',comparison)
            result['original_IL_comparison_sha256']=sha(output/'original-IL-comparison.json')
            guard();budget.complete(history_complete=True)
        result['status']='complete_matched_TRAIN_endpoint'
        result['selection_readiness']={'ready':True,'execution_admitted':False,
            'meaning':'fixed updates, authenticated reload, 29 teacher readouts and four complete TRAIN replays; performance may be negative'}
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error),
            'committed':bool(getattr(error,'committed',False))},selection_readiness={'ready':False,'execution_admitted':False})
        raise
    finally:
        result['completed_source_visits']=len(visits.completed);result['complete_wall_seconds']=time.perf_counter()-started
        result['total_policy_forward_calls']=costs.forwards
        result['native_preview_entries']=budget.snapshot()['native_preview_entries']
        details={'costs':costs.rows,'native_budget':budget.snapshot(),'total_policy_forward_calls':costs.forwards,
            'completed_patient_visits':visits.completed,'complete_wall_seconds':result['complete_wall_seconds'],
            'scope':'actual offline/deployment costs; nested inclusive scopes must not be summed; methods do not have equal compute',
            'resource_authority':'owned parent hard wall, sampled process-tree RSS and final output caps'}
        try:sink.write(output/'costs.json',details)
        except BaseException:
            result['status']='failed_or_unresolved';result['selection_readiness']={'ready':False,'execution_admitted':False}
            sink.write(output/'result.json',result,terminal=True);raise
        sink.write(output/'result.json',result,terminal=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    release=json.loads(args.release.read_text());source_guard(release,args.release,args.release_sha256)
    output=ROOT/release['output']
    if output.exists() or output.is_symlink():raise FileExistsError('One attempt only')
    supervision=output.with_name(output.name+'.supervision')
    if not supervision.is_dir():raise ValueError('Owned parent reservation required')
    started=time.perf_counter();summary={'status':'started','method':release['method'],'SELECT_EVAL_opened':False}
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    def progress(phase,**details):
        path=supervision/'worker-progress.tmp';path.write_text(json.dumps({'phase':phase,'seconds':time.perf_counter()-started,**details})+'\n');path.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Matched TRAIN worker deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,release['cohort_limits']['worker_seconds'])
    try:
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        summary['runtime']={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
            'numpy':np.__version__,'threads':1,'interop_threads':1,'device':'cpu','dtype':'float32'}
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Incomplete matched endpoint')
        control=endpoint_control(output,release);write(supervision/'endpoint-control.json',control)
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_matched_TRAIN_endpoint',result_sha256=sha(output/'result.json'),
            endpoint_control_sha256=sha(supervision/'endpoint-control.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        write(supervision/'worker-final.json',summary)

if __name__=='__main__':main()
