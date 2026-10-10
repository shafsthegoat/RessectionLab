"""Two fresh TRAIN045 tasks; existing exact cover cache only during second greedy.

No CPython profiler, model, optimizer, checkpoint, source overlay or cache math.
Fixed baseline-first N=1 pair is preliminary and subject to order/allocator effects.
"""
import argparse,gc,json,os,resource,signal,sys,time,weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import ARMS,CACHE_LIMITS,CAPS,INPUTS,OUTPUT,guard,need,sha,small

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--declaration-sha256',required=True);args=parser.parse_args()
    release,_,_=guard(args.declaration,args.declaration_sha256,execute=True)
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from resectionlab import geometry,native_resection
    from resectionlab.core import freeze_json,semantic_digest,thaw_json
    from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache,inject_native_capsule_cache
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    plain=lambda value:thaw_json(freeze_json(value))
    load=lambda key:small(INPUTS[key]['path'],INPUTS[key]['sha256'])
    original=load('original_release');prior=load('case');plan=load('plan');prior_metrics=load('metrics');prior_audit=load('audit')
    need(prior_audit['accepted'] and prior['status']=='completed_fixed_post_exposure_route','accepted_teacher_required')
    index=small(original['public_index']['path'],original['public_index']['sha256'])
    entry=next(row for row in index['cases'] if row['patient_id']=='ReMIND-045')
    manifest=small(entry['path'],entry['sha256'])
    need(manifest['role']=='TRAIN' and not manifest['private_evaluation_files_included'],'public_TRAIN')
    active={str(Path(value['path']).resolve()) for value in manifest['input_files'].values()}
    need(set(manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'five_public_inputs')
    cohort=ROOT/original['cohort']['path'];cohort_bytes=cohort.read_bytes();need(sha(cohort)==original['cohort']['sha256'],'frozen_cohort')
    qualification={'scope':'post_exposure_initial_and_nominal_greedy','subjects':tuple(original['subjects']),
        'occupancy_condition':original['occupancy_condition'],'proposal_config':original['proposal_config'],
        'max_steps':24,'methods_executed':['observed_greedy'],'post_exposure_condition':original['post_exposure_condition'],
        'greedy_seconds_per_case':60,'retention_reference':original['retention_reference'],'retention_applied':False}
    # Historical context identity stays exact; the active owned pair has tighter caps.
    limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':40000,'max_policy_forwards':0,
        'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    need(sys.getprofile() is None,'unprofiled_pair_required')
    original_cover=native_resection.capsule_voxel_indices
    need(original_cover is geometry.capsule_voxel_indices,'unmodified_native_cover')
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    started=time.perf_counter();deadline=started+CAPS['worker_seconds']
    result={'status':'started','subject':'ReMIND-045','release_sha256':args.declaration_sha256,
        'source_visits':0,'greedy_calls':0,'committed_actions':0,'model_calls':0,'optimizer_calls':0,
        'checkpoint_loads':0,'blocked_external_calls':0,'arms':[],'live_sources_after_cleanup':None,
        'interpretation':'one baseline-first paired observation; same process may retain allocator and filesystem caches; no sustained speed or RL claim'}
    originals=[]
    def forbid(owner,name,counter):
        originals.append((owner,name,getattr(owner,name)))
        def refused(*unused,**kwargs):result[counter]+=1;raise RuntimeError('Forbidden '+counter)
        setattr(owner,name,refused)
    forbid(torch.nn.Module,'__init__','model_calls');forbid(torch.nn.Module,'_call_impl','model_calls')
    forbid(torch.optim.Optimizer,'__init__','optimizer_calls');forbid(torch,'load','checkpoint_loads')
    def access(event,arguments):
        if event=='open' and arguments and isinstance(arguments[0],(str,bytes)):
            path=Path(os.fsdecode(arguments[0])).resolve()
            denied=(path.suffix.lower()=='.npy' and str(path) not in active) or path.suffix.lower() in ('.dcm','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data')
            if path.suffix.lower()=='.npy':denied=denied or (isinstance(arguments[1],str) and any(c in arguments[1] for c in 'wax+')) or (isinstance(arguments[2],int) and arguments[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if denied:result['blocked_external_calls']+=1;raise PermissionError('Only five public arrays permitted')
        if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
            result['blocked_external_calls']+=1;raise PermissionError('No external work')
    def check():
        if time.perf_counter()>=deadline:raise TimeoutError('Paired worker deadline')
        sink.check()
    def deadline_signal(*unused):raise TimeoutError('Paired worker deadline or termination')
    handlers={s:signal.signal(s,deadline_signal) for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL,CAPS['worker_seconds'])
    def run_arm(name):
        check();folder=OUTPUT/name;folder.mkdir(exist_ok=False)
        row={'arm':name,'status':'started','committed_actions':0,'cache_stats':None,'files':{},
            'rss_peak_scope':'ru_maxrss is cumulative process peak, not an independent arm peak; parent also samples arm intervals',
            'process_cumulative_peak_rss_before_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        result['arms'].append(row);arm_started=time.perf_counter()
        consumed=sum(item.get('native_previews',0) for item in result['arms'])
        budget=PlanningBudget(NativeResectionEngine,max_native_previews=CAPS['native_previews']-consumed,seconds=max(.001,deadline-time.perf_counter()))
        cache=None;source_ref=None
        def progress(*unused,**kwargs):check();budget.check()
        try:
            with budget:
                source_dir=folder/'source';source_dir.mkdir()
                source,binding,qc,protocol=prepare_public_source(source_dir,{'limits':limits},INPUTS['original_release']['sha256'],None,progress,
                    public_manifest_path=entry['path'],public_manifest_sha256=entry['sha256'],cohort_bytes=cohort_bytes,
                    learning_protocol_hash=semantic_digest(qualification),proposal_config=NominalCavityProposalConfig(**original['proposal_config']),
                    public_target_context_variant=TARGET_CONTEXT,occupancy_condition=original['occupancy_condition'],post_exposure_condition=original['post_exposure_condition'])
                source_ref=weakref.ref(source);result['source_visits']+=1
                task,context=make_patient_planning_task(source,cohort_bytes=cohort_bytes,source_binding=binding,qc_receipt=qc,protocol=protocol)
                observation=task.observation()
                need(source.source_hash==prior['source_hash'] and context.fingerprint==prior['context_hash']
                     and observation.fingerprint==prior['initial_observation_hash'],'exact_original_world')
                row['world']={'source_hash':source.source_hash,'context_hash':context.fingerprint,
                    'decision_model_hash':task.decision_model_hash,'initial_observation_hash':observation.fingerprint,
                    'action_ids':list(observation.action_ids),'action_mask':observation.action_mask.tolist()}
                row['construction_seconds']=time.perf_counter()-arm_started
                row['cache_construction_seconds']=0.
                if name=='cached':
                    stamp=time.perf_counter();cache=ExactCapsuleCoverCache(task._config,**CACHE_LIMITS)
                    row['cache_construction_seconds']=time.perf_counter()-stamp
                result['greedy_calls']+=1
                with budget.phase('planning'):
                    if cache is None:
                        stamp=time.perf_counter();actions,accounting=task.observed_greedy_search(seconds=60)
                        row['greedy_wall_seconds']=time.perf_counter()-stamp
                    else:
                        try:
                            with inject_native_capsule_cache(cache):
                                stamp=time.perf_counter();actions,accounting=task.observed_greedy_search(seconds=60)
                                row['greedy_wall_seconds']=time.perf_counter()-stamp
                        finally:
                            row['cache_stats']=cache.stats()
                            row['cache_key_bookkeeping_bytes']=None
                            row['cache_key_bookkeeping_note']='Existing stats bound entries and array payload only; Python key/bookkeeping bytes uninstrumented. Parent RSS includes them.'
                need(native_resection.capsule_voxel_indices is original_cover,'cache_restored_before_replay')
                row['cache_restored_before_replay']=True
                cache=None;gc.collect()
                saved_plan={'actions':list(actions),'accounting':plain(accounting)}
                sink.write(folder/'greedy-plan.json',saved_plan);row['files']['greedy-plan']=sha(folder/'greedy-plan.json')
                need(accounting['complete'] and list(actions)==plan['actions'],'unchanged_complete_actions')
                before=dict(plan['accounting']);after=plain(accounting)
                before.pop('planning_seconds');after.pop('planning_seconds')
                need(before==after,'unchanged_scores_and_dispositions')
                stamp=time.perf_counter()
                with budget.phase('execution'):
                    for action in actions:
                        check();task.step(action);row['committed_actions']+=1;result['committed_actions']+=1
                row['authoritative_replay_seconds']=time.perf_counter()-stamp
                # Explicit complete serialization normalization fixes the preserved profile guard.
                metrics=plain(task.metrics());sink.write(folder/'episode-metrics.json',metrics)
                row['files']['episode-metrics']=sha(folder/'episode-metrics.json')
                need(task.terminated and metrics==prior_metrics,'exact_complete_serialized_metrics')
                row['exact_original_metrics']=True
                budget.complete(history_complete=True)
            # Current independent evaluator runs outside both cache and planning wrappers.
            check();need(native_resection.capsule_voxel_indices is original_cover,'uncached_independent_audit')
            stamp=time.perf_counter();evaluated=evaluate_native_spatial_episode(task,cancelled=lambda:(check() or False))
            row['independent_audit_seconds']=time.perf_counter()-stamp
            sink.write(folder/'independent-episode.json',evaluated);row['files']['independent-episode']=sha(folder/'independent-episode.json')
            need(evaluated['accepted'] and evaluated['outcomes']==prior_audit['outcomes'],'current_evaluator_exact_outcome')
            row.update(independent_accepted=True,status='complete')
            return source_ref
        except BaseException as error:
            row.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
            raise
        finally:
            cache=None
            row.update(native_previews=budget.snapshot()['native_preview_entries'],native_budget=budget.snapshot(),
                elapsed_seconds=time.perf_counter()-arm_started,
                process_cumulative_peak_rss_after_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            sink.write(folder/'arm-result.json',row,terminal=True)
    try:
        sys.addaudithook(access)
        for name in ARMS:
            source_ref=run_arm(name);gc.collect()
            need(source_ref() is None,'prior_arm_source_not_released')
            result['live_sources_after_cleanup']=0
        one,two=result['arms'];need(one['world']==two['world'],'exact_initial_world_parity')
        # Both plans, all non-time accounting, full metrics and independent outcomes
        # were compared to the same exact saved original inside each arm.
        result.update(exact_pair_parity=True,status='complete_paired_native_routes',
            greedy_wall_ratio_cached_over_baseline=two['greedy_wall_seconds']/one['greedy_wall_seconds'],
            greedy_seconds_difference_cached_minus_baseline=two['greedy_wall_seconds']-one['greedy_wall_seconds'])
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        result.update(native_previews=sum(row.get('native_previews',0) for row in result['arms']),elapsed_seconds=time.perf_counter()-started)
        sink.write(OUTPUT/'result.json',result,terminal=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
