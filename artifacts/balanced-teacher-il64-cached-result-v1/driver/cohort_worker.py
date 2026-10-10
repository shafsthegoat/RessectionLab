"""Owned cached64 IL runtime-parity comparison; exact prior64 math required."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(HERE))
from pilot_contract import (OUTPUT, source_guard, complete_result, endpoint_control, WORKER_SECONDS,TRAIN,PUBLIC_INDEX,PUBLIC_SHA,COHORT,
    EXECUTION,canonical_configuration,canonical,sha,require_baseline,
    first_eight_update_control)


def execute(release,release_sha,output,progress):
    import numpy as np
    import torch
    from resectionlab.core import freeze_json,semantic_digest,thaw_json
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

    protocol,limits=canonical_configuration()
    if canonical(protocol)!=canonical(release['learning_protocol']) or canonical(limits)!=canonical(release['cohort_limits']):
        raise ValueError('Canonical objective or admission limits changed')
    baseline=require_baseline(protocol)
    factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC_INDEX,
        manifest_index_sha256=PUBLIC_SHA,cohort_bytes=(ROOT/COHORT).read_bytes(),
        learning_protocol=protocol,limits=limits,released_record=release,
        released_sha256=release_sha,progress=progress)
    output.mkdir(exist_ok=False)
    sink=OutputBudget(output,limits['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=EXECUTION['max_native_previews'],seconds=WORKER_SECONDS)
    costs=preflight._CallCosts(budget,EXECUTION['max_policy_forwards'])
    started=time.perf_counter(); teachers={}; readouts=[]; dynamics=[]
    cache=PatientTeacherTraceCache(protocol); detached_readouts={}; cache_reuses=0
    result={'status':'started','TRAIN':list(TRAIN),'SELECT_EVAL_opened':False,'private_reference_reads':0,
        'optimizer_updates':{'IL':0,'RL':0},'search_calls':0,'loss_forward_calls':0,
        'teacher_logit_forwards':0,'checkpoint_loads':0,'checkpoints':{},'TRAIN_greedy':{},
        'teacher_statuses':{s:'not_started' for s in TRAIN},
        'selection_readiness':{'ready':False,'execution_admitted':False}}
    def guard():budget.check();sink.check()
    visits=_SequentialVisits(factories,protocol,limits,sink,costs,guard)
    def directory(*parts):
        dest=output.joinpath(*parts);dest.mkdir(parents=True,exist_ok=False);return dest
    def teacher_replay(base,context,dest,subject):
        prior=baseline['fixed64']['teacher_'+subject]['plan']
        trace=preflight._collect(base,context,actions=prior['actions'],output=dest,guard=guard)
        sealed=preflight._seal_and_replay(base,context,trace,method='SEARCH',policy=None,updates=0,output=dest,guard=guard)
        actual=thaw_json(sealed['plan'])
        # The new objective/limits context changes its seal; the physical task,
        # complete committed history, observations and exact selected actions do not.
        if canonical({k:v for k,v in actual.items() if k!='context_hash'})!=canonical({k:v for k,v in prior.items() if k!='context_hash'}):
            raise ValueError('Frozen complete teacher world/plan/history changed: '+subject)
        for step,row in enumerate(trace.transitions):
            saved=baseline[f'state_{subject}_{step}']
            if (row.observation.fingerprint!=saved['inputs']['observation_hash']
                    or list(row.observation.action_ids)!=saved['action_ids']
                    or row.observation.action_mask.tolist()!=saved['action_mask']
                    or row.action_id!=saved['teacher_action']):
                raise ValueError('Exact saved teacher observation/inventory changed')
        return trace,sealed
    def score(policy,observation,action):
        with torch.no_grad():logits,value=policy(observation)
        legal=np.flatnonzero(observation.action_mask);raw=logits.detach().cpu().numpy()
        probabilities=logits.softmax(-1);logp=logits.log_softmax(-1)
        ranking=sorted(legal,key=lambda i:(-float(raw[i]),int(i)))
        movement=[int(i) for i in ranking if i];label=observation.action_ids.index(action)
        best=movement[0] if movement else None
        conditional=None if label==0 else float(logits[1:].softmax(-1)[label-1])
        return {'logits':[float(raw[i]) if observation.action_mask[i] else None for i in range(len(raw))],
            'probabilities':probabilities.tolist(),'teacher_CE':float(-logp[label]),
            'teacher_probability':float(probabilities[label]),'stop_probability':float(probabilities[0]),
            'teacher_rank':ranking.index(label)+1,'teacher_movement_rank':None if label==0 else movement.index(label)+1,
            'conditional_teacher_movement_probability':conditional,
            'greedy_action':observation.action_ids[ranking[0]],'teacher_minus_STOP':float(raw[label]-raw[0]),
            'best_movement_minus_STOP':None if best is None else float(raw[best]-raw[0]),'value':float(value)}
    sink.write(output/'configuration.json',{'learning_protocol':protocol,'admission_limits':limits,
        'execution_limits':EXECUTION,'initialization':'canonical common initializer; unused identical clone discarded',
        'runtime_preview_sizing':'8 visits * 120 candidates * (1 construction + 3 * horizon24); conservative, no search',
        'teacher_observations':protocol['cohort_execution']['teacher_observations'],
        'teacher_cache_payload_bytes':protocol['cohort_execution']['teacher_cache_payload_bytes'],
        'objective':'0.5 mean motion NLL + 0.5 mean STOP NLL',
        'comparison':'fixed64 endpoint, no intermediate checkpoint selection or stopping rule on identical five labels; no held-out selection'})
    try:
        with budget,costs,_bounded_writes(sink):
            for subject in TRAIN:
                dest=directory('teachers',subject)
                def admit(base,context):
                    trace,sealed=teacher_replay(base,context,dest,subject)
                    cache.add_replayed(trace,sealed)
                    detached_readouts[subject]=trace.transitions
                    return {'readout_binding_hash':semantic_digest([
                        [r.observation.fingerprint,r.action_id,r.reward,r.terminated] for r in trace.transitions]),
                        'context_hash':context.fingerprint,'steps':len(trace.transitions),
                        'stop_steps':sum(t.action_id=='STOP' for t in trace.transitions),
                        'trace_seal':trace.seal_hash,'plan_seal':sealed['plan_seal'],
                        'actions':[t.action_id for t in trace.transitions]}
                teachers[subject]=visits.run(subject,dest,'offline.fixed_teacher_replay.'+subject,admit)
                result['teacher_statuses'][subject]='complete_replayed'
            if sum(t['steps'] for t in teachers.values())!=5 or sum(t['stop_steps'] for t in teachers.values())!=4:
                raise ValueError('Exact fixed label inventory changed')
            sink.write(output/'teacher-cache.json',cache.record())
            contexts=tuple(visits.contexts[s] for s in TRAIN)
            with costs.scope('offline.common_initialization'):
                policy,unused_clone=common_patient_policies(contexts,protocol);del unused_clone
                initial=parameter_hash(policy);result['initial_parameter_hash']=initial
                if initial!=baseline['result']['initial_parameter_hash']:
                    raise ValueError('Initial tensor hash differs from unweighted eight-update run')
                session=PatientTrainSession(contexts,'IL',policy,initial_parameter_hash=initial,protocol=protocol)
            pins={t['context_hash']:{k:t[k] for k in ('steps','stop_steps','trace_seal')} for t in teachers.values()}
            sink.write(output/'teacher-pins.json',teachers)
            for update in range(64):
                if session.updates!=update:raise ValueError('Unexpected optimizer update count')
                dest=directory('IL',f'update-{update+1:02d}')
                with PatientGradientAccumulator(session,teacher_pins=pins) as accumulation:
                    for subject in TRAIN:
                        visit=directory('IL',f'update-{update+1:02d}',subject)
                        with costs.scope(f'offline.IL.update-{update+1:02d}.cached_loss_backward.'+subject):
                            trace=cache.trace_for_il(session,subject)
                            details=accumulation.add_trace(trace,guard=guard)
                            del trace
                        cache_reuses+=1
                        row={**details,'plan_seal':teachers[subject]['plan_seal'],
                            'observation_storage':protocol['cohort_execution']['teacher_observations']}
                        sink.write(visit/'gradient-contribution.json',row)
                    with costs.scope(f'offline.IL.update-{update+1:02d}.shared_optimizer_step'):
                        row=accumulation.finish(guard=guard)
                    if row['loss_forward_calls']!=5:raise ValueError('Each update must use all five fixed labels')
                    result['loss_forward_calls']+=row['loss_forward_calls'];result['optimizer_updates']['IL']=session.updates
                    sink.write(dest/'update.json',row)
                    dynamics.append({'completed_updates':row['completed_updates'],
                        'pre_update_loss':row['loss'],'loss_change_from_previous':None if not dynamics else row['loss']-dynamics[-1]['pre_update_loss'],
                        'gradient_norm_before_clip':row['gradient_norm_before_clip'],
                        'module_gradient_norms_before_clip':row['module_gradient_norms_before_clip'],
                        'clip_threshold':protocol['max_gradient_norm'],
                        'gradient_exceeds_clip_threshold':row['gradient_norm_before_clip']>protocol['max_gradient_norm'],
                        'before_parameter_hash':row['before_parameter_hash'],'after_parameter_hash':row['after_parameter_hash'],
                        'loss_forward_calls':row['loss_forward_calls'],'update_record_sha256':sha(dest/'update.json')})
                    if session.updates==8:
                        control=first_eight_update_control(output,release)
                        result['first_eight_update_control']=control
                        sink.write(output/'first-eight-update-control.json',control)
                    guard()
            sink.write(output/'training-dynamics.json',{'updates':dynamics,
                'loss_scope':'each receipt is before its update; final teacher metrics are after update64',
                'schedule':'all64 updates mandatory; diagnostics never select an endpoint or stop training'})
            result['training_dynamics_updates']=len(dynamics)
            result['training_dynamics_sha256']=sha(output/'training-dynamics.json')
            with costs.scope('offline.IL.checkpoint_save_reload'):
                pin=_save_checkpoint(policy,method='IL',contexts=contexts,protocol=protocol,updates=session.updates,
                    initial_hash=initial,output=sink,limits=limits)
                result['checkpoints']['IL']={k:pin[k] for k in ('path','sha256','bytes','parameter_hash')}
                reloaded,metadata=load_cohort_checkpoint(output/pin['path'],expected_sha256=pin['sha256'],
                    expected_learning_protocol=protocol,expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},expected_method='IL')
                result['checkpoint_loads']=1
                if parameter_hash(reloaded)!=parameter_hash(policy) or metadata['completed_updates']!=64:
                    raise ValueError('Reloaded endpoint differs from fixed64 tensors')
                reloaded.eval();reloaded.requires_grad_(False)
                frozen=parameter_hash(reloaded)
                sink.write(output/'checkpoint-reload.json',{'sha256':pin['sha256'],'parameter_hash':frozen,
                    'trained_parameter_hash':parameter_hash(policy),'exact_parameter_match':True,'completed_updates':64})
            del session,policy
            for subject in TRAIN:
                dest=directory('teacher-readout',subject)
                def readout():
                    # No optimizer admission at an exhausted session: these are
                    # immutable rows captured at the initial verified replay.
                    cache.record()
                    detached=detached_readouts[subject]
                    visits.contexts[subject].require_observations(r.observation for r in detached)
                    if semantic_digest([[r.observation.fingerprint,r.action_id,r.reward,r.terminated]
                            for r in detached])!=teachers[subject]['readout_binding_hash']:
                        raise ValueError('Detached teacher readout binding changed')
                    rows=[]
                    for step,row in enumerate(detached):
                        guard();saved=baseline[f'state_{subject}_{step}'];old=saved['scores']['IL_8']
                        previous=baseline['balanced'][f'state_{subject}_{step}']
                        if (previous['observation_hash']!=row.observation.fingerprint
                                or previous['action_ids']!=list(row.observation.action_ids)
                                or previous['action_mask']!=row.observation.action_mask.tolist()
                                or previous['teacher_action']!=row.action_id):
                            raise ValueError('Balanced8 endpoint observation differs from fixed teacher state')
                        with costs.scope('offline.endpoint_teacher_logits.'+subject):
                            new=score(reloaded,row.observation,row.action_id)
                        old=dict(old)
                        label=saved['action_ids'].index(row.action_id)
                        legal_motion=[i for i in range(1,len(saved['action_ids'])) if saved['action_mask'][i]]
                        if label:
                            top=max(old['logits'][i] for i in legal_motion)
                            denominator=sum(np.exp(old['logits'][i]-top) for i in legal_motion)
                            old['conditional_teacher_movement_probability']=float(np.exp(old['logits'][label]-top)/denominator)
                        else:old['conditional_teacher_movement_probability']=None
                        item={'subject':subject,'step':step,'teacher_action':row.action_id,
                            'observation_hash':row.observation.fingerprint,'action_ids':list(row.observation.action_ids),
                            'action_mask':row.observation.action_mask.tolist(),'balanced_IL_64':new,'balanced_IL_8':previous['balanced_IL_8'],'unweighted_IL_8':old}
                        sink.write(dest/f'state-{step:02d}.json',item);rows.append(item)
                    if parameter_hash(reloaded)!=frozen:raise ValueError('Teacher readout changed weights')
                    return rows
                with costs.scope('offline.endpoint_cached_teacher_readout.'+subject):
                    rows=readout()
                readouts.extend(rows);result['teacher_logit_forwards']+=len(rows)
            cache_record=cache.record()
            result['teacher_cache']={**cache_record,'trace_reuses':cache_reuses,'readout_rows':len(readouts)}
            result['teacher_trace_reuses']=cache_reuses
            del cache
            detached_readouts.clear()
            endpoint={}
            for method in ('balanced_IL_64','balanced_IL_8','unweighted_IL_8'):
                stop=[r[method]['teacher_CE'] for r in readouts if r['teacher_action']=='STOP']
                motion=[r[method]['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
                endpoint[method]={'STOP_count':len(stop),'motion_count':len(motion),
                    'STOP_mean_NLL':sum(stop)/len(stop),'motion_mean_NLL':sum(motion)/len(motion),
                    'unweighted_mean_CE':(sum(stop)+sum(motion))/5,
                    'balanced_mean_CE':.5*sum(stop)/len(stop)+.5*sum(motion)/len(motion)}
            sink.write(output/'endpoint-teacher-metrics.json',endpoint);result['endpoint_teacher_metrics']=endpoint
            for subject in TRAIN:
                dest=directory('TRAIN-greedy','IL',subject)
                def greedy(base,context):
                    trace=preflight._collect(base,context,policy=reloaded,output=dest,guard=guard)
                    sealed=preflight._seal_and_replay(base,context,trace,method='IL',policy=reloaded,updates=64,output=dest,guard=guard)
                    if parameter_hash(reloaded)!=frozen:raise ValueError('Greedy replay changed reloaded endpoint')
                    return {'complete':True,'checkpoint_reloaded':True,'parameter_hash':frozen,
                        'plan_seal':sealed['plan_seal'],'steps':len(trace.transitions),
                        'actions':[t.action_id for t in trace.transitions],'public_return':sum(t.reward for t in trace.transitions),
                        'target_removed_mm3':sum(r.get('target_removed_mm3',0.) for r in sealed['replayed_history']),
                        'outside_supplied_target_removed_mm3':sum(r.get('normal_removed_mm3',0.) for r in sealed['replayed_history']),
                        'outside_interpretation':'outside supplied target, not verified normal tissue'}
                result['TRAIN_greedy'][subject]=visits.run(subject,dest,'deployment.reloaded_TRAIN_greedy_replay.'+subject,greedy)
            if (len(visits.completed)!=8 or cache_reuses!=256 or result['loss_forward_calls']!=320 or result['teacher_logit_forwards']!=5
                    or costs.forwards!=325+sum(r['steps'] for r in result['TRAIN_greedy'].values())):
                raise ValueError('Executed visit/forward accounting differs from declared endpoint')
            guard();budget.complete(history_complete=True)
        result['status']='complete_cached_balanced_IL64_TRAIN_only'
        result['selection_readiness']={'ready':True,'execution_admitted':False,
            'meaning':'64 IL updates, exact first-eight control, authenticated weight reload, five teacher readouts and four complete TRAIN greedy replays'}
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error),
            'committed':bool(getattr(error,'committed',False))},selection_readiness={'ready':False,'execution_admitted':False})
        raise
    finally:
        result['completed_source_visits']=len(visits.completed)
        result['complete_wall_seconds']=time.perf_counter()-started
        details={'costs':costs.rows,'native_budget':budget.snapshot(),'total_policy_forward_calls':costs.forwards,
            'completed_patient_visits':visits.completed,'complete_wall_seconds':result['complete_wall_seconds'],
            'scope':'offline training/readout and deployment separate; nested inclusive scopes must not be summed',
            'resource_authority':'owned parent hard wall, sampled tree RSS and final output receipt'}
        try:sink.write(output/'costs.json',details)
        except BaseException:
            result['status']='failed_or_unresolved';result['selection_readiness']={'ready':False,'execution_admitted':False}
            sink.write(output/'result.json',result,terminal=True)
            raise
        sink.write(output/'result.json',result,terminal=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    release=json.loads(args.release.read_text());source_guard(release,args.release,args.release_sha256)
    output=ROOT/OUTPUT
    if output.exists() or output.is_symlink():raise FileExistsError('One attempt only')
    supervision=output.with_name(output.name+'.supervision')
    if not supervision.is_dir():raise ValueError('Owned parent reservation required')
    started=time.perf_counter();summary={'status':'started','SELECT_EVAL_opened':False}
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    def progress(phase,**details):
        path=supervision/'worker-progress.tmp';path.write_text(json.dumps({'phase':phase,'seconds':time.perf_counter()-started,**details})+'\n');path.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Cached64 IL worker deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        summary['runtime']={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
            'numpy':np.__version__,'threads':1,'interop_threads':1,'device':'cpu','dtype':'float32'}
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Incomplete cached64 endpoint')
        control=endpoint_control(output,release);write(supervision/'endpoint-control.json',control)
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_cached_balanced_IL64',result_sha256=sha(output/'result.json'),
            endpoint_control_sha256=sha(supervision/'endpoint-control.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        write(supervision/'worker-final.json',summary)

if __name__=='__main__':main()
