"""Five fixed TRAIN baseline/bound pairs; all ten plans before uncached replay."""
import argparse,gc,json,os,signal,sys,time,weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
if __name__=='__main__' and (not sys.pycache_prefix or Path(sys.pycache_prefix).exists()):
    raise ValueError('Fresh nonexistent -X pycache_prefix required before local imports')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import (ARMS,CAPS,CACHE_LIMITS,OUTPUT,RIDGE,COHORT,COHORT_SHA,PROPOSALS,
    OCCUPANCY,EXPOSURE,SUBJECTS,VERSION,POLICY_VERSION,TRAIN_RESULT,TRAIN_OUTPUT,SOURCE_RELEASE,
    guard,metadata,historical_rows,need,sha,small,read_bytes)
from batch_contract import FIT_SUBJECTS,SUBJECT,transfer_record,transfer_metadata,FIFTY_OUTPUT


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
    import resectionlab
    resectionlab.__path__.insert(0,str(HERE/'stage/resectionlab'))
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
    from patient_regression import admit_source,load_frozen_ridge
    from patient_transfer import admit_source as admit_transfer
    from policy_tasks import BaselineTask,BoundTask
    from bound_policy import VERSION as FALLBACK_VERSION
    need(FALLBACK_VERSION==POLICY_VERSION,'exact_fallback_policy')
    plain=lambda x:thaw_json(freeze_json(x))
    prior,training_release,source_release=metadata();references=historical_rows(prior)
    prior_fifty,_=transfer_metadata();transfer=transfer_record(release['transfer_inputs'])
    cohort=read_bytes(COHORT,COHORT_SHA)
    index=small(release['public_index']['path'],release['public_index']['sha256'])
    entries={row['patient_id']:{'path':row['path'],'sha256':row['sha256']} for row in index['cases']}
    need(list(entries)==FIT_SUBJECTS,'exact_public_four_order')
    entries[SUBJECT]=release['transfer_inputs']['public_manifest']
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
        'source_world_hashes':{s:release['source_world_hashes'][s] for s in FIT_SUBJECTS},
        'proposal_config':PROPOSALS,'post_exposure_condition':EXPOSURE,'fits':0,'optimizer_updates':0,'neural_forwards':0,
        'frozen_ridge':RIDGE,'training_result':TRAIN_RESULT}
    source_limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':CAPS['native_previews'],'max_policy_forwards':0,
        'worker_seconds':CAPS['worker_seconds'],'memory_bytes':CAPS['sampled_rss_bytes'],'threads':1,
        'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    result={'status':'started','policy_version':POLICY_VERSION,'training_result':TRAIN_RESULT,'release_sha256':release_sha,'subjects':SUBJECTS,'role':'TRAIN',
        'scope':'four_fitted_TRAIN_plus_exposed_TRAIN050_no_refit','source_visits':0,'ridge_loads':0,'fits':0,
        'native_transitions':0,'model_calls':0,'optimizer_calls':0,'checkpoint_loads':0,'blocked_external_calls':0,
        'arms':[{'subject':subject,'method':mode,'complete':False,'independent_accepted':False,'status':'not_started','files':{}} for subject in SUBJECTS for mode in ARMS],
        'source_visits_detail':[],'all_plans_sealed_before_evaluation':False,'ridge_unchanged':False,
        'cache_restored':False,'SELECT_EVAL_opened':False,
        'order_caveat':'one baseline-first pair per subject; same-run descriptive timing only; no causal speed generalization',
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
            extra={'development_transfer':transfer} if subject==SUBJECT else {}
            case,binding,qc,protocol=prepare_public_source(dest,{'limits':source_limits},
                release_sha if subject==SUBJECT else SOURCE_RELEASE['sha256'],None,
                lambda *a,**k:check(),public_manifest_path=mr['path'],public_manifest_sha256=mr['sha256'],cohort_bytes=cohort,
                learning_protocol_hash=semantic_digest(transfer if subject==SUBJECT else qualification),
                proposal_config=NominalCavityProposalConfig(**PROPOSALS),public_target_context_variant=TARGET_CONTEXT,
                occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE,**extra)
            if subject==SUBJECT:
                admission=admit_transfer(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,
                                         protocol=protocol,release_sha256=release_sha)
            else:
                admission=admit_source(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol,pilot_declaration=declaration)
            actual={'source':case.source_hash,'reference':case.reference_hash,'native':case._native_config.fingerprint,
                'binding':semantic_digest(binding),'qc':semantic_digest(qc),'protocol':semantic_digest(protocol)}
            if subject not in world:world[subject]=actual;result['public_worlds']=world
            else:need(actual==world[subject],'exact_world_planning_and_replay')
            visit.update(status='complete',**actual)
            return case,admission,time.perf_counter()-stamp
        finally:visit['source_read_preparation_validation_seconds']=time.perf_counter()-stamp
    def record_task(subject,mode,case,admission,folder,costs):
        events=[];directory=folder/'inventories';directory.mkdir(parents=True,exist_ok=False)
        def emit(event):
            check();need(len(events)<24,'inventory_count_cap');sink.write(directory/f'state-{len(events):02d}.json',event)
            if not events:
                current={'state':event['source_state_hash'],'public_input':event['shortlist']['input_binding'],'raw_ids':event['shortlist']['raw_ids']}
                if subject not in initial:initial[subject]=current;result['initial_public_proposals']=initial
                else:need(current==initial[subject],'same_initial_public_world_and_raw_catalog')
            events.append(event['source_state_hash'])
        costs.inventory_callback=emit
        cls={'switch_guard':BaselineTask,'public_bound':BoundTask}[mode]
        return cls(case,admission=admission,ranker=ranker,max_steps=24,
            cancelled=lambda:(check() or False),costs=costs)
    def access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve();suffix=path.suffix.lower()
            denied=(suffix=='.npy' and str(path) not in allowed) or suffix in ('.dcm','.psckpt','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data')
            if suffix=='.npy':denied|=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            if denied:external_io['blocked']+=1;raise PermissionError('Exact public fixed-five arrays only')
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
                mode=row['method'];subject=row['subject'];key=mode+'/'+subject;folder=OUTPUT/key;before=budget.snapshot()['native_preview_entries'];stamp=time.perf_counter();costs=Costs()
                operation={'method':mode,'subject':subject,'phase':'planning','started':stamp,'preview_before':before,'costs':costs};row['status']='source_preparation'
                case,admission,prep=source(subject,folder/'source','planning.'+subject);ref=weakref.ref(case);row['source_preparation_seconds']=prep
                with budget.phase('planning'),budget.request('planning.'+subject):
                    cs=time.perf_counter();cache=ExactCapsuleCoverCache(case._native_config,**CACHE_LIMITS);row['cache_construction_seconds']=time.perf_counter()-cs
                    with inject_native_capsule_cache(cache):
                        task=record_task(subject,mode,case,admission,folder,costs)
                        actions,accounting=task.observed_greedy_search(seconds=60)
                        row['source_through_plan_seconds']=time.perf_counter()-stamp
                        need(accounting['complete'],'complete_plan_not_partial_STOP')
                    row['cache_stats']=cache.stats();cache=None
                row.update(online_previews=budget.snapshot()['native_preview_entries']-before,online_costs=costs.record())
                plan={'subject':subject,'method':mode,'source_hash':case.source_hash,'decision_model_hash':task.decision_model_hash,
                    'admission_hash':admission.fingerprint,'ridge_hash':RIDGE['fingerprint'],
                    'actions':list(actions),'accounting':plain(accounting),'max_steps':24}
                plan['seal']=semantic_digest(plan);plans[key]=plan;sink.write(folder/'plan.json',plan)
                row['files'][str((folder/'plan.json').relative_to(OUTPUT))]=sha(folder/'plan.json');row['status']='plan_sealed';row['action_count']=len(actions)
                del case,admission,task;gc.collect();need(ref() is None,'planning_case_retained');operation=None
            need(list(plans)==[m+'/'+s for s in SUBJECTS for m in ARMS],'all_ten_plans_required_before_any_evaluation');result['all_plans_sealed_before_evaluation']=True
            sink.write(OUTPUT/'sealed-plans.json',{'plans':plans,'private_reference_loaded':False})
            for row in result['arms']:
                mode=row['method'];subject=row['subject'];key=mode+'/'+subject;folder=OUTPUT/key/'replay';stamp=time.perf_counter();before=budget.snapshot()['native_preview_entries'];costs=Costs()
                operation={'method':mode,'subject':subject,'phase':'replay','started':stamp,'preview_before':before,'costs':costs};row['status']='source_preparation_for_replay'
                plan=small(OUTPUT/key/'plan.json',row['files'][key+'/plan.json']);unsigned={k:v for k,v in plan.items() if k!='seal'}
                need(semantic_digest(unsigned)==plan['seal'] and plan==plans[key],'sealed_complete_plan_unchanged')
                case,admission,prep=source(subject,folder/'source','replay.'+subject);ref=weakref.ref(case)
                need(native_resection.capsule_voxel_indices is original_cover,'uncached_authoritative_replay')
                with budget.phase('execution'),budget.request('replay.'+subject):
                    task=record_task(subject,mode,case,admission,folder,costs)
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
                if subject in FIT_SUBJECTS:
                    balanced=references['balanced_ridge_top8',subject];old_output=TRAIN_OUTPUT
                    reference={m:references[m,subject] for m in ('balanced_ridge_top8','exhaustive')}
                else:
                    old_output=FIFTY_OUTPUT
                    balanced=next(r for r in prior_fifty['arms'] if r['method']=='balanced_ridge_switch_guard')
                    reference={'balanced_ridge_top8':balanced,'exhaustive':next(r for r in prior_fifty['arms'] if r['method']=='exhaustive')}
                old_plan_key=next(k for k in balanced['files'] if k.endswith('/plan.json'))
                old_metrics_key=next(k for k in balanced['files'] if k.endswith('/episode-metrics.json'))
                old_plan=small(old_output/old_plan_key,balanced['files'][old_plan_key])
                old_metrics=small(old_output/old_metrics_key,balanced['files'][old_metrics_key])
                row['historical_quality']=compare_reference(plan,metrics,row['outcomes'],reference,old_plan,old_metrics)
            pairs=[]
            for subject in SUBJECTS:
                base,bound=[next(r for r in result['arms'] if r['subject']==subject and r['method']==m) for m in ARMS]
                def saved_metrics(row):
                    p=next(k for k in row['files'] if k.endswith('/episode-metrics.json'))
                    return small(OUTPUT/p,row['files'][p])
                bm,um=saved_metrics(base),saved_metrics(bound)
                normalized=dict(um);normalized['decision_model_hash']=bm['decision_model_hash']
                pairs.append({'subject':subject,
                    'exact_actions':plans['switch_guard/'+subject]['actions']==plans['public_bound/'+subject]['actions'],
                    'exact_metrics_except_outer_policy_identity':normalized==bm,
                    'exact_history':bm['history']==um['history'],'exact_outcomes':base['outcomes']==bound['outcomes'],
                    'baseline_online_previews':base['online_previews'],'bound_online_previews':bound['online_previews'],
                    'baseline_source_through_plan_seconds':base['source_through_plan_seconds'],
                    'bound_source_through_plan_seconds':bound['source_through_plan_seconds'],
                    'bound_costs':bound['online_costs'],
                    'baseline_retains_historical_quality':base['historical_quality']['no_quality_loss_vs_saved_exhaustive']})
            result['comparison']={'pairs':pairs,'exact_policy_equivalence':all(p['exact_actions'] and p['exact_metrics_except_outer_policy_identity']
                and p['exact_history'] and p['exact_outcomes'] and p['baseline_retains_historical_quality'] for p in pairs),
                'sole_normalized_field':'episode-metrics.decision_model_hash; distinct policy identities required',
                'favorable_outcome_required_for_completion':False,'heldout_claim':False,
                'timing_comparison':'fresh fixed-order pairs, all source/setup/bound costs included; no historical timing ratios',
                'clinical_validation':None}
            result['ridge_calls']=sum(r['online_costs']['counts'].get('ridge_calls',0)+r['replay_costs']['counts'].get('ridge_calls',0) for r in result['arms'])
            need(result['ridge_calls']<=CAPS['ridge_calls'],'ridge_call_cap');result['ridge_unchanged']=ranker.fingerprint==RIDGE['fingerprint']
            result['cache_restored']=native_resection.capsule_voxel_indices is original_cover
            budget.complete(history_complete=True);check();result['status']='complete_five_TRAIN_public_bound_comparison'
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
