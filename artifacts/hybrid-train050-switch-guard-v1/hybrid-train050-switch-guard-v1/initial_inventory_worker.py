"""One frozen ridge vs lean exhaustive TRAIN050 comparison; no fit/optimizer."""
import argparse,gc,json,os,signal,sys,time,weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
if __name__=='__main__' and (not sys.pycache_prefix or Path(sys.pycache_prefix).exists()):
    raise ValueError('Fresh nonexistent -X pycache_prefix required before local imports')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import (ARMS,CAPS,CACHE_LIMITS,OUTPUT,RIDGE,COHORT,COHORT_SHA,PROPOSALS,
    OCCUPANCY,EXPOSURE,SUBJECT,POLICY_VERSION,BASELINE,PREVIOUS_FALLBACK,guard,transfer_record,need,sha,small,read_bytes)


def execute(release,release_sha,external_io):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from resectionlab.core import freeze_json,thaw_json,semantic_digest
    from resectionlab import geometry,native_resection
    from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache,inject_native_capsule_cache
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from hybrid_proposal import Costs,RidgePriority
    from patient_transfer import admit_source,AdmittedTransferTask,load_frozen_ridge
    from fallback_policy import admitted_task_class,VERSION as FALLBACK_VERSION
    need(FALLBACK_VERSION==POLICY_VERSION,'exact_fallback_policy')
    FallbackTask=admitted_task_class()
    plain=lambda x:thaw_json(freeze_json(x))
    cohort=read_bytes(COHORT,COHORT_SHA);record=transfer_record(release['inputs'])
    mr=release['inputs']['public_manifest'];manifest=small(mr['path'],mr['sha256'])
    allowed={str(Path(ref['path']).resolve()) for ref in manifest['input_files'].values()}
    source_limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':CAPS['native_previews'],'max_policy_forwards':0,
        'worker_seconds':CAPS['worker_seconds'],'memory_bytes':CAPS['sampled_rss_bytes'],'threads':1,
        'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    result={'status':'started','policy_version':POLICY_VERSION,'original_negative_result':BASELINE,'previous_fallback_result':PREVIOUS_FALLBACK,'release_sha256':release_sha,'subject':SUBJECT,'role':'TRAIN',
        'scope':'development_transfer_not_heldout_SELECT','source_visits':0,'ridge_loads':0,'fits':0,
        'native_transitions':0,'model_calls':0,'optimizer_calls':0,'checkpoint_loads':0,'blocked_external_calls':0,
        'arms':[{'subject':SUBJECT,'method':m,'complete':False,'independent_accepted':False,'status':'not_started','files':{}} for m in ARMS],
        'source_visits_detail':[],'all_plans_sealed_before_evaluation':False,'ridge_unchanged':False,
        'cache_restored':False,'SELECT_EVAL_opened':False,
        'order_caveat':'one fixed switch-guard-first pair; no matched historical learning latency or causal speed generalization',
        'clinical_validation':None,'private_reference_loaded':False}
    started=time.perf_counter();budget=PlanningBudget(NativeResectionEngine,max_native_previews=CAPS['native_previews'],seconds=CAPS['worker_seconds'])
    original_transition=NativeSpatialTask._transition;original_cover=native_resection.capsule_voxel_indices
    need(original_cover is geometry.capsule_voxel_indices,'no_preexisting_cache_overlay')
    originals=[];original_fit=RidgePriority.__dict__['fit'];ranker=None;operation=None;world=None;initial=None;plans={}
    def check():
        budget.check();sink.check()
        need(not external_io['blocked'],'prior_forbidden_access')
        if time.perf_counter()-started>=CAPS['worker_seconds']:raise TimeoutError('Transfer worker cap')
        if ranker is not None:need(ranker.fingerprint==RIDGE['fingerprint'],'frozen_parameters_changed')
    def transition(self,*a,**k):
        check();need(result['native_transitions']<CAPS['native_transitions'],'transition_cap')
        result['native_transitions']+=1;return original_transition(self,*a,**k)
    def source(dest,phase):
        nonlocal world
        check();need(result['source_visits']<4,'source_visit_cap');result['source_visits']+=1
        dest.mkdir(parents=True,exist_ok=False);stamp=time.perf_counter();visit={'phase':phase,'status':'started'}
        result['source_visits_detail'].append(visit)
        try:
            case,binding,qc,protocol=prepare_public_source(dest,{'limits':source_limits},release_sha,None,
                lambda *a,**k:check(),public_manifest_path=mr['path'],public_manifest_sha256=mr['sha256'],cohort_bytes=cohort,
                learning_protocol_hash=semantic_digest(record),proposal_config=NominalCavityProposalConfig(**PROPOSALS),
                public_target_context_variant=TARGET_CONTEXT,occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE,
                development_transfer=record)
            admission=admit_source(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol,release_sha256=release_sha)
            actual={'source':case.source_hash,'reference':case.reference_hash,'native':case._native_config.fingerprint,
                'binding':semantic_digest(binding),'qc':semantic_digest(qc),'protocol':semantic_digest(protocol)}
            if world is None:world=actual;result['public_world']=world
            else:need(actual==world,'exact_world_every_arm_and_replay')
            visit.update(status='complete',**actual)
            return case,admission,time.perf_counter()-stamp
        finally:visit['source_read_preparation_validation_seconds']=time.perf_counter()-stamp
    def record_task(case,admission,mode,folder,costs):
        events=[];directory=folder/'inventories';directory.mkdir(parents=True,exist_ok=False)
        def emit(event):
            nonlocal initial
            check();need(len(events)<24,'inventory_count_cap');sink.write(directory/f'state-{len(events):02d}.json',event)
            if not events:
                current={'state':event['source_state_hash'],'public_input':event['shortlist']['input_binding'],
                    'raw_ids':event['shortlist']['raw_ids']}
                if initial is None:initial=current;result['initial_public_proposals']=current
                else:need(current==initial,'same_initial_public_world_and_raw_catalog')
            events.append(event['source_state_hash'])
        costs.inventory_callback=emit
        cls=FallbackTask if mode=='balanced_ridge_switch_guard' else AdmittedTransferTask
        return cls(case,admission=admission,mode='ridge_top8' if mode=='balanced_ridge_switch_guard' else 'exhaustive',
            ranker=ranker if mode=='balanced_ridge_switch_guard' else None,max_steps=24,cancelled=lambda:(check() or False),costs=costs)
    def access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve();suffix=path.suffix.lower()
            denied=(suffix=='.npy' and str(path) not in allowed) or suffix in ('.dcm','.psckpt','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data')
            if suffix=='.npy':denied|=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            if denied:external_io['blocked']+=1;raise PermissionError('Exact public050 arrays only')
        if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
            external_io['blocked']+=1;raise PermissionError('No external work')
    def forbid(owner,name,key):
        originals.append((owner,name,getattr(owner,name)))
        def refused(*a,**k):result[key]+=1;raise RuntimeError('Forbidden '+key)
        setattr(owner,name,refused)
    def no_fit(*a,**k):result['fits']+=1;raise RuntimeError('Frozen inference cannot fit')
    forbid(torch.nn.Module,'__init__','model_calls');forbid(torch.nn.Module,'_call_impl','model_calls')
    forbid(torch.optim.Optimizer,'__init__','optimizer_calls');forbid(torch,'load','checkpoint_loads');forbid(torch,'save','checkpoint_loads')
    RidgePriority.fit=classmethod(no_fit);sys.addaudithook(access);NativeSpatialTask._transition=transition
    try:
        with budget:
            stamp=time.perf_counter();ranker=load_frozen_ridge();result['ridge_loads']=1
            result['ridge_load_seconds']=time.perf_counter()-stamp;result['ridge_fingerprint']=ranker.fingerprint
            for row in result['arms']:
                mode=row['method'];folder=OUTPUT/mode;before=budget.snapshot()['native_preview_entries'];stamp=time.perf_counter();costs=Costs()
                operation={'method':mode,'phase':'planning','started':stamp,'preview_before':before,'costs':costs};row['status']='source_preparation'
                case,admission,prep=source(folder/'source','planning.'+mode);ref=weakref.ref(case);row['source_preparation_seconds']=prep
                with budget.phase('planning'),budget.request('planning.'+mode):
                    cs=time.perf_counter();cache=ExactCapsuleCoverCache(case._native_config,**CACHE_LIMITS);row['cache_construction_seconds']=time.perf_counter()-cs
                    with inject_native_capsule_cache(cache):
                        task=record_task(case,admission,mode,folder,costs)
                        actions,accounting=task.observed_greedy_search(seconds=60)
                        row['source_through_plan_seconds']=time.perf_counter()-stamp
                        need(accounting['complete'],'complete_plan_not_partial_STOP')
                    row['cache_stats']=cache.stats();cache=None
                row.update(online_previews=budget.snapshot()['native_preview_entries']-before,online_costs=costs.record())
                plan={'subject':SUBJECT,'method':mode,'source_hash':case.source_hash,'decision_model_hash':task.decision_model_hash,
                    'admission_hash':admission.fingerprint,'ridge_hash':None if mode=='exhaustive' else RIDGE['fingerprint'],
                    'actions':list(actions),'accounting':plain(accounting),'max_steps':24}
                plan['seal']=semantic_digest(plan);plans[mode]=plan;sink.write(folder/'plan.json',plan)
                row['files'][str((folder/'plan.json').relative_to(OUTPUT))]=sha(folder/'plan.json');row['status']='plan_sealed'
                del case,admission,task;gc.collect();need(ref() is None,'planning_case_retained');operation=None
            need(len(plans)==2,'both_plans_required_before_any_evaluation');result['all_plans_sealed_before_evaluation']=True
            sink.write(OUTPUT/'sealed-plans.json',{'plans':plans,'private_reference_loaded':False})
            for row in result['arms']:
                mode=row['method'];folder=OUTPUT/mode/'replay';stamp=time.perf_counter();before=budget.snapshot()['native_preview_entries'];costs=Costs()
                operation={'method':mode,'phase':'replay','started':stamp,'preview_before':before,'costs':costs};row['status']='source_preparation_for_replay'
                plan=small(OUTPUT/mode/'plan.json',row['files'][mode+'/plan.json']);unsigned={k:v for k,v in plan.items() if k!='seal'}
                need(semantic_digest(unsigned)==plan['seal'] and plan==plans[mode],'sealed_complete_plan_unchanged')
                case,admission,prep=source(folder/'source','replay.'+mode);ref=weakref.ref(case)
                need(native_resection.capsule_voxel_indices is original_cover,'uncached_authoritative_replay')
                with budget.phase('execution'),budget.request('replay.'+mode):
                    task=record_task(case,admission,mode,folder,costs)
                    need(task.decision_model_hash==plan['decision_model_hash'] and case.source_hash==plan['source_hash'],'exact_replay_identity')
                    for action in plan['actions']:check();task.step(action)
                    need(task.terminated,'complete_replay')
                    metrics=plain(task.metrics());sink.write(folder/'episode-metrics.json',metrics)
                    row['files'][str((folder/'episode-metrics.json').relative_to(OUTPUT))]=sha(folder/'episode-metrics.json')
                    audit=evaluate_native_spatial_episode(task,cancelled=lambda:(check() or False))
                    sink.write(folder/'independent-episode.json',audit)
                    row['files'][str((folder/'independent-episode.json').relative_to(OUTPUT))]=sha(folder/'independent-episode.json')
                    need(audit['accepted'],'full_tool_independent_geometry')
                row.update(complete=True,status='complete',independent_accepted=True,outcomes=audit['outcomes'],
                    replay_source_preparation_seconds=prep,replay_and_audit_seconds=time.perf_counter()-stamp,
                    replay_previews=budget.snapshot()['native_preview_entries']-before,replay_costs=costs.record(),
                    endpoint='STOP' if plan['actions'][-1]=='STOP' else 'HORIZON')
                del case,admission,task;gc.collect();need(ref() is None,'replay_case_retained');operation=None
            a,b=(r['outcomes'] for r in result['arms']);tol=1e-7
            result['comparison']={'numeric_tolerance':tol,'no_quality_loss':a['target_removed_mm3']>=b['target_removed_mm3']-tol
                and a['total_reward']>=b['total_reward']-tol and a['normal_removed_mm3']<=b['normal_removed_mm3']+tol
                and a['cumulative_contacted_tissue_upper_bound_mm3']<=b['cumulative_contacted_tissue_upper_bound_mm3']+tol,
                'target_difference_mm3':a['target_removed_mm3']-b['target_removed_mm3'],'reward_difference':a['total_reward']-b['total_reward'],
                'preview_ratio':result['arms'][0]['online_previews']/max(1,result['arms'][1]['online_previews']),
                'source_through_plan_wall_ratio':result['arms'][0]['source_through_plan_seconds']/result['arms'][1]['source_through_plan_seconds'],
                'favorable_outcome_required_for_completion':False,'TRAIN_only':True,'clinical_validation':None}
            prior=small(BASELINE['path'],BASELINE['sha256'])
            prior_exhaustive=next(r for r in prior['arms'] if r['method']=='exhaustive')
            prior_plan_path=Path(BASELINE['path']).parent/'exhaustive/plan.json'
            prior_plan=small(prior_plan_path,prior_exhaustive['files']['exhaustive/plan.json'])
            result['historical_exhaustive_control']={
                'exact_actions':plans['exhaustive']['actions']==prior_plan['actions'],
                'exact_outcomes':result['arms'][1]['outcomes']==prior_exhaustive['outcomes'],
                'exact_source':plans['exhaustive']['source_hash']==prior_plan['source_hash'],
                'scope':'historical scalar/action corroboration; timing comparison uses this fresh pair only'}
            result['ridge_calls']=sum(r['online_costs']['counts'].get('ridge_calls',0)+r['replay_costs']['counts'].get('ridge_calls',0) for r in result['arms'])
            need(result['ridge_calls']<=48,'ridge_call_cap');result['ridge_unchanged']=ranker.fingerprint==RIDGE['fingerprint']
            result['cache_restored']=native_resection.capsule_voxel_indices is original_cover
            budget.complete(history_complete=True);check();result['status']='complete_TRAIN050_switch_guard_comparison'
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        if hasattr(error,'accounting') and hasattr(error,'best_sequence'):
            sink.write(OUTPUT/'partial-search.json',{'status':'failed_or_unresolved','accounting':plain(error.accounting),
                'best_sequence':list(error.best_sequence),'accepted_plan':False})
            result['partial_search_sha256']=sha(OUTPUT/'partial-search.json')
        if operation is not None:
            row=next(r for r in result['arms'] if r['method']==operation['method']);row['status']='failed_or_unresolved'
            result['failed_operation_costs']={k:v for k,v in operation.items() if k not in ('started','costs')}
            result['failed_operation_costs'].update(seconds=time.perf_counter()-operation['started'],
                native_previews_paid=budget.snapshot()['native_preview_entries']-operation['preview_before'],local_costs=operation['costs'].record())
        raise
    finally:
        NativeSpatialTask._transition=original_transition;RidgePriority.fit=original_fit
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        result.update(native_previews=budget.snapshot()['native_preview_entries'],native_budget=budget.snapshot(),
            elapsed_seconds=time.perf_counter()-started,blocked_external_calls=external_io['blocked'])
        sink.write(OUTPUT/'result.json',result,terminal=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--declaration',type=Path,required=True);p.add_argument('--declaration-sha256',required=True);a=p.parse_args()
    release,_,_=guard(a.declaration,a.declaration_sha256,execute=True)
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    def timeout(*a):raise TimeoutError('Owned transfer deadline or parent termination')
    old={s:signal.signal(s,timeout) for s in (signal.SIGALRM,signal.SIGTERM)};signal.setitimer(signal.ITIMER_REAL,CAPS['worker_seconds'])
    try:execute(release,a.declaration_sha256,{'blocked':0})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in old.items():signal.signal(s,h)
    return 0
if __name__=='__main__':raise SystemExit(main())
