"""Two fixed ridge fits and three verified TRAIN proposal strategies; no neural model."""
import argparse,gc,json,os,signal,sys,time,weakref
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import ARMS,CAPS,CACHE_LIMITS,OUTPUT,SUBJECTS,STEPS,guard,metadata,need,sha,small


def execute(release,release_sha,external_io):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from resectionlab.core import freeze_json,thaw_json,semantic_digest,array_digest
    from resectionlab import geometry,native_resection
    from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache,inject_native_capsule_cache
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from hybrid_proposal import Costs,RidgePriority,PublicRows,public_rows,regression_weights
    from patient_sparse import admit_source,AdmittedSparseTask,PILOT_VERSION
    plain=lambda x:thaw_json(freeze_json(x))
    owner,prior,baseline=metadata();old=prior['release']
    cohort=(ROOT/owner.COHORT).read_bytes();need(sha(ROOT/owner.COHORT)==owner.COHORT_SHA,'cohort')
    index=small(old['public_index']['path'],old['public_index']['sha256'])
    entries={s:next(row for row in index['cases'] if row['patient_id']==s) for s in SUBJECTS}
    manifests={s:small(entries[s]['path'],entries[s]['sha256']) for s in SUBJECTS}
    allowed={str(Path(ref['path']).resolve()) for m in manifests.values() for ref in m['input_files'].values()}
    need(all(m['role']=='TRAIN' and m['public_only'] and not m['private_evaluation_files_included']
        and set(m['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'}
        for m in manifests.values()),'only_exact_public_five_array_TRAIN')
    qualification={'scope':'post_exposure_initial_and_nominal_greedy','subjects':tuple(old['subjects']),
        'occupancy_condition':old['occupancy_condition'],'proposal_config':old['proposal_config'],
        'max_steps':24,'methods_executed':['observed_greedy'],'post_exposure_condition':old['post_exposure_condition'],
        'greedy_seconds_per_case':60,'retention_reference':old['retention_reference'],'retention_applied':False}
    source_limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':40000,'max_policy_forwards':0,
        'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    declaration={'version':PILOT_VERSION,'released_sha256':release_sha,
        'source_world_hashes':{s:release['source_world_hashes'][s] for s in SUBJECTS},
        'proposal_config':release['proposal_config'],'post_exposure_condition':release['post_exposure_condition']}
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    result={'status':'started','release_sha256':release_sha,'source_visits':0,'teacher_decisions':0,'fits':0,
        'native_transitions':0,'teacher_score_calls':0,'fit_rows':0,'model_calls':0,'optimizer_calls':0,'checkpoint_loads':0,
        'blocked_external_calls':0,'arms':[{'subject':subject,'method':mode,'complete':False,'independent_accepted':False,'status':'not_started','files':{}} for mode in ARMS for subject in SUBJECTS],'source_visits_detail':[],'all_plans_sealed_before_evaluation':False,
        'ridge_unchanged':False,'cache_restored':False,'SELECT_EVAL_opened':False,
        'interpretation':'IL proposal-priority pilot on fixed TRAIN states; no RL or transfer improvement claim',
        'order_caveat':'single fixed order; filesystem and allocator carryover can affect timings'}
    started=time.perf_counter();budget=PlanningBudget(NativeResectionEngine,max_native_previews=40000,seconds=600)
    current_operation=None
    rankers={};frozen={};teacher_score_calls=0;original_transition=NativeSpatialTask._transition
    original_cover=native_resection.capsule_voxel_indices;need(original_cover is geometry.capsule_voxel_indices,'no_cover_overlay')
    def check():
        budget.check();sink.check()
        if external_io['blocked']:raise PermissionError('Prior forbidden access')
        if time.perf_counter()-started>=600:raise TimeoutError('Pilot wall cap')
        need(all(r.fingerprint==frozen[m] for m,r in rankers.items()),'ridge_mutated')
    def transition(self,*args,**kwargs):
        check();need(result['native_transitions']<634,'native_transition_cap')
        result['native_transitions']+=1;return original_transition(self,*args,**kwargs)
    def source(subject,dest,phase):
        check();need(result['source_visits']<28,'source_visit_cap');result['source_visits']+=1
        before=time.perf_counter();dest.mkdir(parents=True,exist_ok=False)
        visit={'subject':subject,'phase':phase,'status':'started'}
        result['source_visits_detail'].append(visit)
        try:
            case,binding,qc,protocol=prepare_public_source(dest,{'limits':source_limits},owner.INPUTS['release']['sha256'],None,
                lambda *a,**k:check(),public_manifest_path=entries[subject]['path'],public_manifest_sha256=entries[subject]['sha256'],
                cohort_bytes=cohort,learning_protocol_hash=semantic_digest(qualification),
                proposal_config=NominalCavityProposalConfig(**old['proposal_config']),public_target_context_variant=TARGET_CONTEXT,
                occupancy_condition=old['occupancy_condition'],post_exposure_condition=old['post_exposure_condition'])
            admission=admit_source(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol,pilot_declaration=declaration)
        finally:
            visit['source_read_preparation_validation_seconds']=time.perf_counter()-before
        cost=time.perf_counter()-before
        visit.update(status='complete',source_hash=case.source_hash,admission_hash=admission.fingerprint)
        return case,admission,dict(cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol),cost
    def record_task(case,admission,mode,folder,costs):
        event_dir=folder/'inventories';event_dir.mkdir(parents=True,exist_ok=False);events=[]
        def emit_inventory(event):
            check();need(len(events)<24,'inventory_event_cap')
            sink.write(event_dir/f'state-{len(events):02d}.json',event);events.append(event['source_state_hash'])
        costs.inventory_callback=emit_inventory
        task=AdmittedSparseTask(case,admission=admission,mode='exhaustive' if mode=='exhaustive' else 'ridge_top8',ranker=rankers.get(mode),
            max_steps=24,cancelled=lambda:(check() or False),costs=costs)
        return task
    # Filesystem audit is installed only after exact public descriptors are parsed.
    def access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve();suffix=path.suffix.lower()
            denied=(suffix=='.npy' and str(path) not in allowed) or suffix in ('.dcm','.psckpt','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data')
            if suffix=='.npy':denied|=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            if denied:external_io['blocked']+=1;raise PermissionError('Only declared public TRAIN inputs')
        if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
            external_io['blocked']+=1;raise PermissionError('No external work')
    originals=[]
    def forbid(owner,name,key):
        originals.append((owner,name,getattr(owner,name)))
        def refused(*a,**k):result[key]+=1;raise RuntimeError('Forbidden '+key)
        setattr(owner,name,refused)
    forbid(torch.nn.Module,'__init__','model_calls');forbid(torch.nn.Module,'_call_impl','model_calls')
    forbid(torch.optim.Optimizer,'__init__','optimizer_calls')
    forbid(torch,'load','checkpoint_loads');forbid(torch,'save','checkpoint_loads')
    sys.addaudithook(access);NativeSpatialTask._transition=transition
    xs=[];ys=[];state_ids=[];train_rows=[];plans={};fit_data_hash=None
    try:
        with budget:
            for subject,n in zip(SUBJECTS,STEPS):
                folder=OUTPUT/'training'/subject;case,admission,args,prep=source(subject,folder/'source','training')
                ref=weakref.ref(case);row={'subject':subject,'source_preparation_seconds':prep};stamp=time.perf_counter()
                with budget.phase('planning'), budget.request('training_labels.'+subject):
                    cache=ExactCapsuleCoverCache(case._native_config,**CACHE_LIMITS)
                    with inject_native_capsule_cache(cache):
                        base,ctx=make_patient_planning_task(case,**args)
                        original_case=next(r for r in prior['result']['cases'] if r['patient_id']==subject)
                        need(ctx.fingerprint==original_case['context_hash'] and base.decision_model_hash==prior['metrics_'+subject]['decision_model_hash'], 'exact_original_teacher_context')
                        model=base.planning_clone();actions=prior['plan_'+subject]['actions']
                        need(len(actions)==n,'teacher_count')
                        for step,action in enumerate(actions):
                            check();inventory=model._prepare_inventory();batch=model._proposal_batch
                            cavity=model._engine.connected_free_mask
                            rows=public_rows(model.case,cavity,batch.proposals,steps=step,horizon=24,current_tool=model._current_tool)
                            labels=[];cert=None
                            for identifier in rows.ids:
                                cert=inventory.get(identifier)
                                reward=0. if cert is None else model._score_record({**cert.to_history_record(),'action_id':identifier},model._current_tool)['reward']
                                labels.append(max(0.,reward))
                            if len(labels):xs.append(rows.features);ys.append(np.asarray(labels));state_ids.extend([result['teacher_decisions']]*len(labels))
                            train_rows.append({'subject':subject,'step':step,'teacher_action':action,'public_input_hash':rows.public_binding,
                                'feature_hash':array_digest(rows.features),'raw_ids':list(rows.ids),'legal': [x in inventory for x in rows.ids],
                                'priority_labels':labels,'feature_rows':len(labels)})
                            result['teacher_decisions']+=1;model.advance_planning(action)
                        need(model.terminated,'teacher_complete')
                    row['cache_stats']=cache.stats();cache=None
                    row['label_collection_seconds']=time.perf_counter()-stamp
                need(native_resection.capsule_voxel_indices is original_cover,'restore_before_teacher_replay')
                with budget.phase('execution'), budget.request('training_replay.'+subject):
                    stamp=time.perf_counter();replay=NativeSpatialTask(case,max_steps=24)
                    for action in actions:replay.step(action)
                    metrics=plain(replay.metrics());sink.write(folder/'teacher-metrics.json',metrics)
                    audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(check() or False));sink.write(folder/'teacher-audit.json',audit)
                    row['replay_audit_seconds']=time.perf_counter()-stamp
                    need(metrics==prior['metrics_'+subject],'exact_full_teacher_history')
                    need(audit['accepted'],'teacher_independent_geometry')
                sink.write(folder/'costs.json',row)
                del case,admission,args,base,ctx,model,replay,batch,inventory,cert,rows,cavity
                gc.collect();need(ref() is None,'training_source_retained')
            need(result['teacher_decisions']==sum(STEPS),'all_teacher_states')
            X=np.concatenate(xs);Y=np.concatenate(ys);result['fit_rows']=len(Y);need(len(Y)<=3480,'fit_row_cap')
            fit_data_hash=semantic_digest({'X':array_digest(X),'y':array_digest(Y),'state_ids':state_ids})
            old_control=baseline['proposal_v1']
            need(fit_data_hash==old_control['labels']['data_hash'] and train_rows==old_control['labels']['states'],'exact_v1_teacher_inputs_labels')
            result['teacher_labels_identity']=True
            result['fit_seconds']={};result['ridge_file_sha256']={}
            for mode,balanced in (('old_ridge_top8',False),('balanced_ridge_top8',True)):
                stamp=time.perf_counter();ranker=RidgePriority.fit(X,Y,state_ids,positive_zero_balanced=balanced)
                rankers[mode]=ranker;frozen[mode]=ranker.fingerprint
                result['fits']+=1;result['fit_seconds'][mode]=time.perf_counter()-stamp
                sink.write(OUTPUT/(mode+'.json'),{'feature_version':ranker.fingerprint,'data_hash':fit_data_hash,
                    'mean':ranker.mean.tolist(),'scale':ranker.scale.tolist(),'weights':ranker.weights.tolist(),'intercept':ranker.intercept})
                result['ridge_file_sha256'][mode]=sha(OUTPUT/(mode+'.json'))
            need(all(p['ridge_hash']==frozen['old_ridge_top8'] for p in old_control['plans'].values()),'old_ridge_exact_v1_parameters')
            need(np.array_equal(rankers['old_ridge_top8'].mean,rankers['balanced_ridge_top8'].mean)
                and np.array_equal(rankers['old_ridge_top8'].scale,rankers['balanced_ridge_top8'].scale),'identical_normalization')
            result['old_ridge_parameter_identity']=True
            sink.write(OUTPUT/'training-labels.json',{'states':train_rows,'data_hash':fit_data_hash,
                'weighting':release['fit']['state_weights']})
            readouts=[];offset=0
            for row in train_rows:
                n=row['feature_rows'];features=X[offset:offset+n]
                raw=PublicRows(tuple(row['raw_ids']),features,np.zeros(n),row['public_input_hash'])
                readout={k:row[k] for k in ('subject','step','teacher_action','raw_ids','public_input_hash','priority_labels')}
                readout['models']={}
                for mode,model in rankers.items():
                    teacher_score_calls+=1;result['teacher_score_calls']=teacher_score_calls
                    scores=model.score(raw)
                    order=sorted(range(n),key=lambda i:(-float(scores[i]),i))
                    readout['models'][mode]={'scores':scores.tolist(),'ranks':[order.index(i)+1 for i in range(n)],
                        'teacher_rank':order.index(row['raw_ids'].index(row['teacher_action']))+1 if row['teacher_action']!='STOP' else None,
                        'top8_ids':[row['raw_ids'][i] for i in order[:8]]}
                readouts.append(readout);offset+=n
            sink.write(OUTPUT/'teacher-readouts.json',{'states':readouts,'calls':teacher_score_calls,'no_native_queries':True})
            weights_old=regression_weights(Y,state_ids);weights_balanced=regression_weights(Y,state_ids,positive_zero_balanced=True)
            groups=[]
            for index,row in enumerate(train_rows):
                mask=np.asarray(state_ids)==index;pos=mask & (Y>0);zero=mask & (Y==0)
                groups.append({'subject':row['subject'],'step':row['step'],'rows':int(mask.sum()),'positive_rows':int(pos.sum()),
                    'old_mass':float(weights_old[mask].sum()),'balanced_mass':float(weights_balanced[mask].sum()),
                    'old_positive_mass':float(weights_old[pos].sum()),'balanced_positive_mass':float(weights_balanced[pos].sum()),
                    'balanced_zero_mass':float(weights_balanced[zero].sum())})
            sink.write(OUTPUT/'regression-weights.json',{'normalization':'original_equal_state_mean_scale','groups':groups})
            xs.clear();ys.clear();del X,Y,features,raw,ranker,model,weights_old,weights_balanced;gc.collect()
            for mode in ARMS:
                for subject in SUBJECTS:
                    folder=OUTPUT/mode/subject;before=budget.snapshot()['native_preview_entries'];stamp=time.perf_counter()
                    costs=Costs();row=next(r for r in result['arms'] if r['method']==mode and r['subject']==subject)
                    row['status']='source_preparation'
                    current_operation={'phase':'planning','method':mode,'subject':subject,'started':stamp,'preview_before':before,'costs':costs}
                    case,admission,args,prep=source(subject,folder/'source','planning.'+mode);ref=weakref.ref(case)
                    row.update(status='planning',source_preparation_seconds=prep)
                    with budget.phase('planning'), budget.request('planning.'+mode+'.'+subject):
                        cache_stamp=time.perf_counter();cache=ExactCapsuleCoverCache(case._native_config,**CACHE_LIMITS)
                        row['cache_construction_seconds']=time.perf_counter()-cache_stamp
                        with inject_native_capsule_cache(cache):
                            task=record_task(case,admission,mode,folder,costs)
                            actions,accounting=task.observed_greedy_search(seconds=60)
                            row['source_through_plan_seconds']=time.perf_counter()-stamp
                            need(accounting['complete'],'complete_method_plan')
                        row['cache_stats']=cache.stats();cache=None
                    row['online_previews']=budget.snapshot()['native_preview_entries']-before;row['online_costs']=costs.record()
                    plan={'subject':subject,'method':mode,'source_hash':case.source_hash,'decision_model_hash':task.decision_model_hash,
                        'admission_hash':admission.fingerprint,'ridge_hash':frozen.get(mode),
                        'actions':list(actions),'accounting':plain(accounting),'max_steps':24}
                    plan['seal']=semantic_digest(plan);plans[(mode,subject)]=plan
                    sink.write(folder/'plan.json',plan);row['files'][str((folder/'plan.json').relative_to(OUTPUT))]=sha(folder/'plan.json')
                    if mode=='old_ridge_top8':
                        need(list(actions)==old_control['plans'][subject]['actions'],'old_ridge_exact_v1_actions')
                    if mode=='exhaustive':
                        need(list(actions)==prior['plan_'+subject]['actions'],'exhaustive_original_actions')
                        old_account=dict(prior['plan_'+subject]['accounting']);new_account=plain(accounting)
                        for item in (old_account,new_account):
                            for k in ('planning_seconds','method','verification_scope','STOP_meaning'):item.pop(k,None)
                            for decision in item['decisions']:
                                decision['all_queried_legal_actions_scored']=decision.pop('all_current_legal_actions_scored',True)
                        need(old_account==new_account,'exhaustive_exact_original_scores')
                    del case,admission,args,task;gc.collect();need(ref() is None,'planning_source_retained')
                    current_operation=None
            result['all_plans_sealed_before_evaluation']=len(plans)==12
            need(result['all_plans_sealed_before_evaluation'],'all_methods_sealed')
            for row in result['arms']:
                mode,subject=row['method'],row['subject'];folder=OUTPUT/mode/subject/'replay';before=budget.snapshot()['native_preview_entries'];stamp=time.perf_counter()
                costs=Costs();row['status']='source_preparation_for_replay'
                current_operation={'phase':'replay','method':mode,'subject':subject,'started':stamp,'preview_before':before,'costs':costs}
                case,admission,args,prep=source(subject,folder/'source','replay.'+mode);ref=weakref.ref(case)
                row['status']='replay'
                need(native_resection.capsule_voxel_indices is original_cover,'uncached_replay')
                with budget.phase('execution'), budget.request('replay.'+mode+'.'+subject):
                    task=record_task(case,admission,mode,folder,costs);plan=plans[(mode,subject)]
                    need(task.decision_model_hash==plan['decision_model_hash'] and task.case.source_hash==plan['source_hash'],'exact_plan_world')
                    for action in plan['actions']:check();task.step(action)
                    need(task.terminated,'full_replay')
                    metrics=plain(task.metrics());sink.write(folder/'episode-metrics.json',metrics)
                    row['files'][str((folder/'episode-metrics.json').relative_to(OUTPUT))]=sha(folder/'episode-metrics.json')
                    audit=evaluate_native_spatial_episode(task,cancelled=lambda:(check() or False))
                    sink.write(folder/'independent-episode.json',audit)
                    row['files'][str((folder/'independent-episode.json').relative_to(OUTPUT))]=sha(folder/'independent-episode.json')
                    need(audit['accepted'],'independent_complete_tool_geometry')
                row.update(complete=True,status='complete',independent_accepted=True,outcomes=audit['outcomes'],
                    replay_source_preparation_seconds=prep,replay_and_audit_seconds=time.perf_counter()-stamp,
                    replay_previews=budget.snapshot()['native_preview_entries']-before,replay_costs=costs.record(),
                    endpoint='STOP' if plan['actions'][-1]=='STOP' else 'HORIZON')
                del case,admission,args,task;gc.collect();need(ref() is None,'replay_source_retained')
                current_operation=None
            # Predeclared scalar gate; negative results remain complete experiments.
            grouped={mode:[r for r in result['arms'] if r['method']==mode] for mode in ARMS}
            def no_worse(a,b):
                x,y=a['outcomes'],b['outcomes'];tol=release['gates']['numeric_tolerance']
                return (x['target_removed_mm3']>=y['target_removed_mm3']-tol and x['total_reward']>=y['total_reward']-tol
                    and x['normal_removed_mm3']<=y['normal_removed_mm3']+tol
                    and x['cumulative_contacted_tissue_upper_bound_mm3']<=y['cumulative_contacted_tissue_upper_bound_mm3']+tol)
            totals={m:{'online_previews':sum(r['online_previews'] for r in rows),
                'source_through_plan_seconds':sum(r['source_through_plan_seconds'] for r in rows),
                'replay_previews':sum(r['replay_previews'] for r in rows),
                'replay_and_audit_seconds':sum(r['replay_and_audit_seconds'] for r in rows)} for m,rows in grouped.items()}
            quality=all(no_worse(a,b) for a,b in zip(grouped['balanced_ridge_top8'],grouped['exhaustive']))
            old_quality=all(no_worse(a,b) for a,b in zip(grouped['balanced_ridge_top8'],grouped['old_ridge_top8']))
            improved=any(a['outcomes']['total_reward']>b['outcomes']['total_reward']+release['gates']['numeric_tolerance']
                for a,b in zip(grouped['balanced_ridge_top8'],grouped['old_ridge_top8']))
            preview_ratio=totals['balanced_ridge_top8']['online_previews']/max(1,totals['exhaustive']['online_previews'])
            wall_ratio=totals['balanced_ridge_top8']['source_through_plan_seconds']/totals['exhaustive']['source_through_plan_seconds']
            old_outcomes={a['subject']:a['outcomes'] for a in old_control['result']['arms'] if a['method']=='ridge_top8'}
            need(all(row['outcomes']==old_outcomes[row['subject']] for row in grouped['old_ridge_top8']),'old_ridge_exact_v1_outcomes')
            result['comparison']={'totals':totals,'balanced_no_quality_loss_vs_exhaustive':quality,
                'balanced_no_quality_loss_vs_old_ridge':old_quality,'balanced_strict_return_improvement_vs_old_ridge':improved,
                'preview_ratio':preview_ratio,'online_wall_ratio':wall_ratio,
                'loss_contrast_go':quality and old_quality and improved and preview_ratio<=.5 and wall_ratio<=.8,
                'historical_v1_gates':release['historical_v1']['gates'],
                'historical_v1_comparison':old_control['result']['comparison'],
                'v1_retroactively_promoted':False,'historical_IL64_outcomes':baseline['outcomes'],
                'historical_latency_matched':False,'all_costs_include_rejected_queries':True}
            ridge_calls=sum(r['online_costs']['counts'].get('ridge_calls',0)+r['replay_costs']['counts'].get('ridge_calls',0) for r in result['arms'])
            need(ridge_calls+teacher_score_calls<=442,'bounded_linear_inference_calls')
            result.update(ridge_calls=ridge_calls+teacher_score_calls,online_replay_ridge_calls=ridge_calls,teacher_score_calls=teacher_score_calls)
            result['ridge_unchanged']=all(r.fingerprint==frozen[m] for m,r in rankers.items())
            result['cache_restored']=native_resection.capsule_voxel_indices is original_cover
            budget.complete(history_complete=True);check();result['status']='complete_fixed_TRAIN_balanced_ridge_contrast'
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        if hasattr(error,'accounting') and hasattr(error,'best_sequence'):
            sink.write(OUTPUT/'partial-search.json',{'status':'failed_or_unresolved','accounting':plain(error.accounting),
                'best_sequence':list(error.best_sequence),'accepted_plan':False})
            result['partial_search_sha256']=sha(OUTPUT/'partial-search.json')
        if current_operation is not None:
            op=current_operation
            result['failed_operation_costs']={**{k:v for k,v in op.items() if k not in ('started','costs')},
                'seconds':time.perf_counter()-op['started'],'native_previews_paid':budget.snapshot()['native_preview_entries']-op['preview_before'],
                'local_costs':op['costs'].record()}
            next(r for r in result['arms'] if r['method']==op['method'] and r['subject']==op['subject'])['status']='failed_or_unresolved'
        raise
    finally:
        NativeSpatialTask._transition=original_transition
        for owner,name,old_value in reversed(originals):setattr(owner,name,old_value)
        result.update(native_previews=budget.snapshot()['native_preview_entries'],native_budget=budget.snapshot(),
            elapsed_seconds=time.perf_counter()-started,blocked_external_calls=external_io['blocked'])
        sink.write(OUTPUT/'result.json',result,terminal=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--declaration-sha256',required=True);args=parser.parse_args()
    release,_,_=guard(args.declaration,args.declaration_sha256,execute=True)
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    def timeout(*unused):raise TimeoutError('Owned pilot deadline or parent termination')
    old={s:signal.signal(s,timeout) for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL,600)
    try:execute(release,args.declaration_sha256,{'blocked':0})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in old.items():signal.signal(s,h)
    return 0
if __name__=='__main__':raise SystemExit(main())
