"""Four frozen switch-guard plans, then uncached replay; no fitting or new search control."""
import argparse,gc,json,os,signal,sys,time,weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
if __name__=='__main__' and (not sys.pycache_prefix or Path(sys.pycache_prefix).exists()):
    raise ValueError('Fresh nonexistent -X pycache_prefix required before local imports')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import (ARMS,CAPS,CACHE_LIMITS,OUTPUT,RIDGE,COHORT,COHORT_SHA,PROPOSALS,
    OCCUPANCY,EXPOSURE,SUBJECTS,VERSION,POLICY_VERSION,TRAIN_RESULT,TRAIN_OUTPUT,SOURCE_RELEASE,
    guard,metadata,historical_rows,need,sha,small,read_bytes)


def compare_reference(plan,metrics,outcomes,reference,old_plan,old_metrics):
    """No favorable completion gate; preserve every changed scalar and history."""
    tol=1e-7;full=reference['exhaustive']['outcomes'];old=reference['balanced_ridge_top8']['outcomes']
    signs={'target_removed_mm3':1,'total_reward':1,'normal_removed_mm3':-1,
        'cumulative_contacted_tissue_upper_bound_mm3':-1,'currently_retained_contacted_tissue_upper_bound_mm3':-1,
        'complete_tool_path_length_mm':-1,'nonstop_actions':-1,'tool_changes':-1}
    gates={key:sign*(outcomes[key]-full[key])>=-tol for key,sign in signs.items()}
    gates['same_full_target_denominator']=outcomes['total_reference_target_mm3']==full['total_reference_target_mm3']
    gates['physical_probability_remains_null']=outcomes['clinical_deficit_probability'] is full['clinical_deficit_probability'] is None
    gates['unchanged_partial_contact_reward']=outcomes['partial_contact_reward_weight']==full['partial_contact_reward_weight']==0.
    normalized=dict(metrics);normalized['decision_model_hash']=old_metrics['decision_model_hash']
    return {'numeric_tolerance':tol,'exact_actions_vs_balanced':plan['actions']==old_plan['actions'],
        'exact_source_vs_balanced':plan['source_hash']==old_plan['source_hash'],
        'exact_outcomes_vs_balanced':outcomes==old,'exact_history_vs_balanced':metrics['history']==old_metrics['history'],
        'exact_metrics_except_policy_identity':normalized==old_metrics,
        'sole_normalized_field':'episode-metrics.decision_model_hash (explicit new policy/admission contract)',
        'quality_gates_vs_saved_exhaustive':gates,'no_quality_loss_vs_saved_exhaustive':all(gates.values()),
        'outcome_differences_vs_saved_exhaustive':{k:outcomes[k]-full[k] for k in signs},
        'timing_comparison':'none; retained exhaustive is a historical quality reference'}



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
    from patient_regression import admit_source,AdmittedRegressionTask,load_frozen_ridge
    from fallback_policy import VERSION as FALLBACK_VERSION
    need(FALLBACK_VERSION==POLICY_VERSION,'exact_fallback_policy')
    plain=lambda x:thaw_json(freeze_json(x))
    prior,training_release,source_release=metadata();references=historical_rows(prior)
    cohort=read_bytes(COHORT,COHORT_SHA)
    index=small(release['public_index']['path'],release['public_index']['sha256'])
    entries={row['patient_id']:{'path':row['path'],'sha256':row['sha256']} for row in index['cases']}
    need(list(entries)==SUBJECTS,'exact_public_four_order')
    manifests={subject:small(ref['path'],ref['sha256']) for subject,ref in entries.items()}
    need(all(m['patient_id']==subject and m['role']=='TRAIN' and m['public_only'] is True
        and m['private_evaluation_files_included'] is False
        and set(m['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'}
        for subject,m in manifests.items()),'exact_public_fixed_four_manifests')
    allowed={str(Path(ref['path']).resolve()) for m in manifests.values() for ref in m['input_files'].values()}
    qualification={'scope':'post_exposure_initial_and_nominal_greedy','subjects':tuple(source_release['subjects']),
        'occupancy_condition':OCCUPANCY,'proposal_config':PROPOSALS,'max_steps':24,'methods_executed':['observed_greedy'],
        'post_exposure_condition':EXPOSURE,'greedy_seconds_per_case':60,
        'retention_reference':source_release['retention_reference'],'retention_applied':False}
    declaration={'version':VERSION,'released_sha256':release_sha,
        'source_world_hashes':{s:release['source_world_hashes'][s] for s in SUBJECTS},
        'proposal_config':PROPOSALS,'post_exposure_condition':EXPOSURE,'fits':0,'optimizer_updates':0,'neural_forwards':0,
        'frozen_ridge':RIDGE,'training_result':TRAIN_RESULT}
    source_limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':CAPS['native_previews'],'max_policy_forwards':0,
        'worker_seconds':CAPS['worker_seconds'],'memory_bytes':CAPS['sampled_rss_bytes'],'threads':1,
        'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    result={'status':'started','policy_version':POLICY_VERSION,'training_result':TRAIN_RESULT,'release_sha256':release_sha,'subjects':SUBJECTS,'role':'TRAIN',
        'scope':'fixed_fitted_TRAIN_inference_regression','source_visits':0,'ridge_loads':0,'fits':0,
        'native_transitions':0,'model_calls':0,'optimizer_calls':0,'checkpoint_loads':0,'blocked_external_calls':0,
        'arms':[{'subject':subject,'method':ARMS[0],'complete':False,'independent_accepted':False,'status':'not_started','files':{}} for subject in SUBJECTS],
        'source_visits_detail':[],'all_plans_sealed_before_evaluation':False,'ridge_unchanged':False,
        'cache_restored':False,'SELECT_EVAL_opened':False,
        'order_caveat':'one fixed subject order; historical references are quality controls, not matched latency controls',
        'clinical_validation':None,'private_reference_loaded':False}
    started=time.perf_counter();budget=PlanningBudget(NativeResectionEngine,max_native_previews=CAPS['native_previews'],seconds=CAPS['worker_seconds'])
    original_transition=NativeSpatialTask._transition;original_cover=native_resection.capsule_voxel_indices
    need(original_cover is geometry.capsule_voxel_indices,'no_preexisting_cache_overlay')
    originals=[];original_fit=RidgePriority.__dict__['fit'];ranker=None;operation=None;world={};initial={};plans={}
    def check():
        budget.check();sink.check()
        need(not external_io['blocked'],'prior_forbidden_access')
        if time.perf_counter()-started>=CAPS['worker_seconds']:raise TimeoutError('Transfer worker cap')
        if ranker is not None:need(ranker.fingerprint==RIDGE['fingerprint'],'frozen_parameters_changed')
    def transition(self,*a,**k):
        check();need(result['native_transitions']<CAPS['native_transitions'],'transition_cap')
        result['native_transitions']+=1;return original_transition(self,*a,**k)
    def source(subject,dest,phase):
        check();need(result['source_visits']<CAPS['source_visits'],'source_visit_cap');result['source_visits']+=1
        mr=entries[subject];dest.mkdir(parents=True,exist_ok=False);stamp=time.perf_counter()
        visit={'subject':subject,'phase':phase,'status':'started'};result['source_visits_detail'].append(visit)
        try:
            # Exact original search-only qualification is preserved. It binds
            # source provenance; the separate declaration grants only this new
            # inference regression, never fitting or outside-fit transfer.
            case,binding,qc,protocol=prepare_public_source(dest,{'limits':source_limits},SOURCE_RELEASE['sha256'],None,
                lambda *a,**k:check(),public_manifest_path=mr['path'],public_manifest_sha256=mr['sha256'],cohort_bytes=cohort,
                learning_protocol_hash=semantic_digest(qualification),proposal_config=NominalCavityProposalConfig(**PROPOSALS),
                public_target_context_variant=TARGET_CONTEXT,occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE)
            admission=admit_source(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol,pilot_declaration=declaration)
            actual={'source':case.source_hash,'reference':case.reference_hash,'native':case._native_config.fingerprint,
                'binding':semantic_digest(binding),'qc':semantic_digest(qc),'protocol':semantic_digest(protocol)}
            if subject not in world:world[subject]=actual;result['public_worlds']=world
            else:need(actual==world[subject],'exact_world_planning_and_replay')
            visit.update(status='complete',**actual)
            return case,admission,time.perf_counter()-stamp
        finally:visit['source_read_preparation_validation_seconds']=time.perf_counter()-stamp
    def record_task(subject,case,admission,folder,costs):
        events=[];directory=folder/'inventories';directory.mkdir(parents=True,exist_ok=False)
        def emit(event):
            check();need(len(events)<24,'inventory_count_cap');sink.write(directory/f'state-{len(events):02d}.json',event)
            if not events:
                current={'state':event['source_state_hash'],'public_input':event['shortlist']['input_binding'],'raw_ids':event['shortlist']['raw_ids']}
                if subject not in initial:initial[subject]=current;result['initial_public_proposals']=initial
                else:need(current==initial[subject],'same_initial_public_world_and_raw_catalog')
            events.append(event['source_state_hash'])
        costs.inventory_callback=emit
        return AdmittedRegressionTask(case,admission=admission,ranker=ranker,max_steps=24,
            cancelled=lambda:(check() or False),costs=costs)
    def access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve();suffix=path.suffix.lower()
            denied=(suffix=='.npy' and str(path) not in allowed) or suffix in ('.dcm','.psckpt','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data')
            if suffix=='.npy':denied|=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            if denied:external_io['blocked']+=1;raise PermissionError('Exact public fitted-four arrays only')
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
                mode=row['method'];subject=row['subject'];folder=OUTPUT/subject;before=budget.snapshot()['native_preview_entries'];stamp=time.perf_counter();costs=Costs()
                operation={'method':mode,'subject':subject,'phase':'planning','started':stamp,'preview_before':before,'costs':costs};row['status']='source_preparation'
                case,admission,prep=source(subject,folder/'source','planning.'+subject);ref=weakref.ref(case);row['source_preparation_seconds']=prep
                with budget.phase('planning'),budget.request('planning.'+subject):
                    cs=time.perf_counter();cache=ExactCapsuleCoverCache(case._native_config,**CACHE_LIMITS);row['cache_construction_seconds']=time.perf_counter()-cs
                    with inject_native_capsule_cache(cache):
                        task=record_task(subject,case,admission,folder,costs)
                        actions,accounting=task.observed_greedy_search(seconds=60)
                        row['source_through_plan_seconds']=time.perf_counter()-stamp
                        need(accounting['complete'],'complete_plan_not_partial_STOP')
                    row['cache_stats']=cache.stats();cache=None
                row.update(online_previews=budget.snapshot()['native_preview_entries']-before,online_costs=costs.record())
                plan={'subject':subject,'method':mode,'source_hash':case.source_hash,'decision_model_hash':task.decision_model_hash,
                    'admission_hash':admission.fingerprint,'ridge_hash':RIDGE['fingerprint'],
                    'actions':list(actions),'accounting':plain(accounting),'max_steps':24}
                plan['seal']=semantic_digest(plan);plans[subject]=plan;sink.write(folder/'plan.json',plan)
                row['files'][str((folder/'plan.json').relative_to(OUTPUT))]=sha(folder/'plan.json');row['status']='plan_sealed';row['action_count']=len(actions)
                del case,admission,task;gc.collect();need(ref() is None,'planning_case_retained');operation=None
            need(list(plans)==SUBJECTS,'all_four_plans_required_before_any_evaluation');result['all_plans_sealed_before_evaluation']=True
            sink.write(OUTPUT/'sealed-plans.json',{'plans':plans,'private_reference_loaded':False})
            for row in result['arms']:
                mode=row['method'];subject=row['subject'];folder=OUTPUT/subject/'replay';stamp=time.perf_counter();before=budget.snapshot()['native_preview_entries'];costs=Costs()
                operation={'method':mode,'subject':subject,'phase':'replay','started':stamp,'preview_before':before,'costs':costs};row['status']='source_preparation_for_replay'
                plan=small(OUTPUT/subject/'plan.json',row['files'][subject+'/plan.json']);unsigned={k:v for k,v in plan.items() if k!='seal'}
                need(semantic_digest(unsigned)==plan['seal'] and plan==plans[subject],'sealed_complete_plan_unchanged')
                case,admission,prep=source(subject,folder/'source','replay.'+subject);ref=weakref.ref(case)
                need(native_resection.capsule_voxel_indices is original_cover,'uncached_authoritative_replay')
                with budget.phase('execution'),budget.request('replay.'+subject):
                    task=record_task(subject,case,admission,folder,costs)
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
                balanced=references['balanced_ridge_top8',subject]
                old_plan_key='balanced_ridge_top8/'+subject+'/plan.json'
                old_metrics_key='balanced_ridge_top8/'+subject+'/replay/episode-metrics.json'
                old_plan=small(TRAIN_OUTPUT/old_plan_key,balanced['files'][old_plan_key])
                old_metrics=small(TRAIN_OUTPUT/old_metrics_key,balanced['files'][old_metrics_key])
                row['regression']=compare_reference(plan,metrics,row['outcomes'],
                    {m:references[m,subject] for m in ('balanced_ridge_top8','exhaustive')},old_plan,old_metrics)
            result['comparison']={'regression_passed':all(r['regression']['exact_actions_vs_balanced']
                and r['regression']['exact_source_vs_balanced'] and r['regression']['exact_outcomes_vs_balanced']
                and r['regression']['exact_metrics_except_policy_identity']
                and r['regression']['no_quality_loss_vs_saved_exhaustive'] for r in result['arms']),
                'favorable_outcome_required_for_completion':False,'fitted_TRAIN_only':True,
                'timing_comparison':'none; all measured source/setup/planning/replay costs retained without historical wall ratio',
                'reference':TRAIN_RESULT,'clinical_validation':None}
            result['ridge_calls']=sum(r['online_costs']['counts'].get('ridge_calls',0)+r['replay_costs']['counts'].get('ridge_calls',0) for r in result['arms'])
            need(result['ridge_calls']<=CAPS['ridge_calls'],'ridge_call_cap');result['ridge_unchanged']=ranker.fingerprint==RIDGE['fingerprint']
            result['cache_restored']=native_resection.capsule_voxel_indices is original_cover
            budget.complete(history_complete=True);check();result['status']='complete_fixed_four_switch_regression'
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        if hasattr(error,'accounting') and hasattr(error,'best_sequence'):
            sink.write(OUTPUT/'partial-search.json',{'status':'failed_or_unresolved','accounting':plain(error.accounting),
                'best_sequence':list(error.best_sequence),'accepted_plan':False})
            result['partial_search_sha256']=sha(OUTPUT/'partial-search.json')
        if operation is not None:
            row=next(r for r in result['arms'] if r['method']==operation['method'] and r['subject']==operation['subject']);row['status']='failed_or_unresolved'
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
