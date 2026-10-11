"""Persistent public corridor selection against fresh STOP and retained historical h48 search."""
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,ARMS,OUTPUT_BYTES,WORKER_SECONDS,LIMITS,PINS,PRIOR,
    historical,prior_contract,complete_result,source_guard,sha)


def execute(release,release_sha,output,progress):
    from dataclasses import replace
    import torch
    sys.path.append(str(PRIOR))
    from frontier_compare import source_physics,post_exposure_authority
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig,OBSTRUCTION_OPENING_FAMILY
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.paged_obstruction import PagedObstructionTask
    from resectionlab.observed_search import ObservedSearchLimit
    from corridor_selector import select_corridor,check_exploration_geometry
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_cohort_sequential import _bounded_writes
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.planning_budget import PlanningBudget
    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=LIMITS['native_previews_total'],seconds=WORKER_SECONDS)
    costs=_CallCosts(budget,0);started=time.perf_counter();originals=[]
    result={'status':'started','subject':'ReMIND-013','role':'SELECT','planned_SELECT_denominator':2,
        'held_cases':['ReMIND-037'],'held_inputs_opened':False,'EVAL_opened':False,
        'checkpoint_loads':0,'model_forwards':0,'updates':0,'optimizer_attempts':0,'gradient_attempts':0,
        'private_reference_reads':0,'all_action_plans_sealed_before_evaluation':False,
        'arms':{arm:{'status':'not_started','native_transitions':{'search':0,'rollout':0,'replay':0}} for arm in ARMS},
        'lineage_scope':'saved lineage authenticates original source admission only; paged rule is an explicitly changed planning model',
        'cache_condition':'canonical native engine only; no external geometry cache',
        'tie_caveat':'changed source-bound action IDs can change exact-return/path ties; no physical path identity claim across arms',
        'success_scope':'complete geometric return and target progress; no clinical safety or transfer-learning claim'}
    phase={'arm':None,'name':'setup'};previews={arm:0 for arm in ARMS};tasks={};plans={};exploration_history=None
    def guard():budget.check();sink.check()
    def write(name,value):
        path=output/name;path.parent.mkdir(exist_ok=True,parents=True);sink.write(path,value)
        return sha(path)
    def forbid(owner,name,key):
        old=getattr(owner,name);originals.append((owner,name,old))
        def refused(*a,**k):result[key]+=1;raise RuntimeError('No learned operation: '+name)
        setattr(owner,name,refused)
    def install(owner,name,replacement):
        old=getattr(owner,name);originals.append((owner,name,old));setattr(owner,name,replacement);return old
    def phase_cost(arm,name):
        phase.update(arm=arm,name=name);return costs.scope(arm+'.'+name)
    def preview(self,*a,**k):
        arm=phase['arm']
        if arm is None:raise ValueError('Unattributed native preview')
        previews[arm]+=1
        if previews[arm]>LIMITS['native_previews_per_arm']:raise ValueError('Per-arm full preview cap')
        return native_preview(self,*a,**k)
    def transition(self,action,*,observe_successor):
        arm,name=phase['arm'],phase['name']
        if name not in ('search','rollout','replay'):raise ValueError('Unattributed native transition')
        counts=result['arms'][arm]['native_transitions'];counts[name]+=1
        ceiling=LIMITS['selector_transition_ceiling_per_arm'] if name=='search' else LIMITS[name+'_transitions_per_arm']
        if counts[name]>ceiling:raise ValueError('Declared per-phase transition cap')
        if sum(sum(row['native_transitions'].values()) for row in result['arms'].values())>LIMITS['native_transitions_total']:
            raise ValueError('Total native transition cap')
        return native_transition(self,action,observe_successor=observe_successor)
    try:
        forbid(torch,'load','checkpoint_loads');forbid(torch.optim.Optimizer,'__init__','optimizer_attempts')
        forbid(torch.autograd,'backward','gradient_attempts');forbid(torch.autograd,'grad','gradient_attempts')
        saved=historical();training=saved['training'];cohort=(ROOT/prior_contract.COHORT).read_bytes()
        active=prior_contract.select_public(prior_contract.read_pinned(saved['release']['public_index']))
        # Budget wraps this counted preview entry and restores it before our
        # final cleanup restores the original canonical method.
        native_preview=install(NativeResectionEngine,'preview_stroke',preview)
        native_transition=install(NativeSpatialTask,'_transition',transition)
        with budget,costs,_bounded_writes(sink):
            original_physics=None
            for arm in ARMS:
                row=result['arms'][arm];row['status']='public_source_preparation';progress('arm',arm=arm,operation='source')
                with phase_cost(arm,'source_and_construction'):
                    folder=output/arm/'source';folder.mkdir(parents=True)
                    case,binding,qc,protocol=prepare_public_source(folder,saved['release'],PINS['release']['sha256'],None,progress,
                        public_manifest_path=ROOT/active['path'],public_manifest_sha256=active['sha256'],cohort_bytes=cohort,
                        learning_protocol_hash=semantic_digest(training),proposal_config=NominalCavityProposalConfig(**training['cohort_execution']['proposal_config']),
                        public_target_context_variant=training['public_target_context_variant'],expected_role='SELECT',
                        checkpoint_lineage=saved['lineage'],occupancy_condition=prior_contract.OCCUPANCY,
                        occupancy_inference_protocol=training,post_exposure_condition=prior_contract.EXPOSURE,partial_domain_inputs=True)
                    base,context=make_patient_planning_task(case,cohort_bytes=cohort,source_binding=binding,qc_receipt=qc,protocol=protocol)
                    obs=base.observation();world={'source_hash':case.source_hash,'decision_model_hash':base.decision_model_hash,
                        'initial_observation_hash':obs.fingerprint,'max_steps':base.max_steps,'proposal_hash':case.proposal_config.fingerprint,
                        'occupancy_condition':context.record()['occupancy_condition'],'post_exposure':None if case.post_exposure is None else thaw_json(case.post_exposure.record),
                        'source_domains':thaw_json(case._domain_record),'action_ids':list(obs.action_ids),'action_mask':obs.action_mask.tolist()}
                    exact=world==saved['world'] and context.record()==saved['context']
                    write(arm+'/original-world-control.json',{'exact':exact,'world':world,'context':context.record()})
                    if not exact:raise ValueError('Original admitted comparison world differs')
                    result['world_equal']=True
                    physical=source_physics(case)
                    if original_physics is None:original_physics=physical
                    if physical!=original_physics:raise ValueError('Fresh arm source differs')
                    if arm==ARMS[1]:
                        changed=replace(case,proposal_config=replace(case.proposal_config,
                            obstruction_opening=True,tool_footprint_opening=True,max_candidates=120,obstruction_page_size=LIMITS['extra_pairs_per_page'],
                            obstruction_selection_rule='target_ray_coverage_v3'))
                        authority=post_exposure_authority(case.post_exposure,changed.post_exposure)
                        if source_physics(changed)!=physical:raise ValueError('Paged rule changed world geometry/inputs')
                        task=PagedObstructionTask(changed,max_steps=LIMITS['max_steps'],reward=base.reward_spec,cancelled=base._cancelled)
                        write(arm+'/declared-rule-change.json',{'source_physics_equal':True,'post_exposure_authority':authority,
                            'original_source_hash':case.source_hash,'source_hash':changed.source_hash,
                            'original_decision_model_hash':base.decision_model_hash,'decision_model_hash':task.decision_model_hash,
                            'proposal_config':changed.proposal_config.to_record(),'proposal_rule_hash':changed.proposal_config.fingerprint})
                        result['source_physics_equal']=True
                    else:
                        task=NativeSpatialTask(case,max_steps=LIMITS['max_steps'],reward=base.reward_spec,cancelled=base._cancelled)
                    write(arm+'/horizon-change.json',{'historical_source_admission_horizon':base.max_steps,
                        'declared_horizon':task.max_steps,'source_hash':task.case.source_hash,
                        'decision_model_hash':task.decision_model_hash,
                        'scope':'unchanged annotation-assisted public S union T world; separate zero-update h48 diagnostic'})
                    tasks[arm]=task;del case,binding,qc,protocol,obs,base,context
                row['status']='search';progress('arm',arm=arm,operation='search')
                with phase_cost(arm,'search'):
                    before=previews[arm];stamp=time.perf_counter()
                    try:
                        if arm==ARMS[0]:
                            actions=('STOP',);accounting={'complete':True,'model_transition_calls':0,'scope':'fresh explicit STOP0 baseline'}
                        else:
                            actions,accounting=select_corridor(task,seconds=LIMITS['search_seconds_per_arm'],
                                cancelled=lambda:(guard() or False),progress=progress)
                    except ObservedSearchLimit as error:
                        row.update(status='search_unresolved',selector_accounting=error.accounting,partial_actions=list(error.best_sequence))
                        write(arm+'/search-unresolved.json',row);continue
                    if arm==ARMS[1]:exploration_history=accounting['explored_history']
                    row['selector_accounting']={k:v for k,v in accounting.items() if k not in ('states','explored_history')}
                    if accounting['model_transition_calls']!=row['native_transitions']['search']:raise ValueError('Selector/native transition counters differ')
                    write(arm+'/selection.json',{'actions':actions,'accounting':accounting,
                        'native_previews':previews[arm]-before,'wall_seconds':time.perf_counter()-stamp})
                    plan={'actions':list(actions),'source_hash':task.case.source_hash,'decision_model_hash':task.decision_model_hash,
                        'proposal_rule_hash':task.case.proposal_config.fingerprint,'max_steps':LIMITS['max_steps'],'objective':'unchanged_public_nominal_reward',
                        'parameter_hash':None,'updates':0,'search_complete':True}
                    plan['seal']=semantic_digest(plan);plans[arm]=plan;write(arm+'/selected-action-plan.json',plan)
                    row.update(status='selected_complete',plan_seal=plan['seal'])
            phase.update(arm=None,name='plans_sealed')
            result['all_action_plans_sealed_before_evaluation']=len(plans)==len(ARMS)
            if not result['all_action_plans_sealed_before_evaluation']:raise ValueError('Unresolved search; no partial sequence evaluated as an accepted plan')
            with phase_cost(ARMS[1],'independent_exploration_geometry'):
                audit=check_exploration_geometry(tasks[ARMS[1]],exploration_history,cancelled=lambda:(guard() or False))
                result['exploration_geometry_sha256']=write(ARMS[1]+'/exploration-geometry.json',audit)
                result['exploration_geometry_feasible']=audit['feasible']
                if not audit['feasible']:raise ValueError('Independent geometry refused nominal exploration')
            del exploration_history
            for arm in ARMS:
                task=tasks[arm];row=result['arms'][arm];plan=plans[arm]
                with phase_cost(arm,'rollout'):
                    row['status']='rollout';progress('arm',arm=arm,operation='rollout')
                    for i,action in enumerate(plan['actions']):
                        guard();write(arm+'/rollout/attempt%02d.json'%i,{'action_id':action})
                        task.step(action)
                    if not task.terminated:raise ValueError('Selected rollout is not complete')
                    metrics=task.metrics();write(arm+'/rollout-metrics.json',metrics)
                    row['complete_history_seal']=semantic_digest({'action_plan_seal':plan['seal'],'history':metrics['history'],'total_reward':metrics['total_reward']})
                with phase_cost(arm,'replay'):
                    row['status']='replay';progress('arm',arm=arm,operation='replay')
                    replay=task.fresh()
                    for i,action in enumerate(plan['actions']):
                        guard();write(arm+'/replay/attempt%02d.json'%i,{'action_id':action});replay.step(action)
                    replayed=replay.metrics();write(arm+'/replay-binding.json',{'metrics_sha256':semantic_digest(replayed),'history_sha256':semantic_digest(replayed['history']),'terminated':replay.terminated})
                    equal=replay.terminated and semantic_digest(replayed['history'])==semantic_digest(metrics['history'])
                    row['full_history_equal']=equal
                    if not equal:raise ValueError('Fresh authoritative history differs from sealed rollout')
                with phase_cost(arm,'independent_audit'):
                    audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(guard() or False))
                    row['independent_certificate_sha256']=write(arm+'/independent-episode.json',audit)
                    row['independent_accepted']=audit['accepted']
                    if not audit['accepted']:raise ValueError('Independent complete-tool geometry refused replay')
                    row.update(status='complete_replayed',outcomes=audit['outcomes'],actions=plan['actions'],
                        endpoint='STOP' if plan['actions'][-1]=='STOP' else 'HORIZON')
                del replay,metrics,replayed,task;tasks.pop(arm);gc.collect()
            phase.update(arm=None,name='finalize')
            baseline,recovery=(result['arms'][a]['outcomes'] for a in ARMS)
            result['comparison']={'nominal_return_improvement':recovery['total_reward']-baseline['total_reward'],
                'positive_complete_return':recovery['total_reward']>0,'target_progress':recovery['target_removed_mm3']>0,
                'target_improvement':recovery['target_removed_mm3']-baseline['target_removed_mm3'],
                'outside_target_delta':recovery['normal_removed_mm3']-baseline['normal_removed_mm3'],
                'contact_delta':recovery['cumulative_contacted_tissue_upper_bound_mm3']-baseline['cumulative_contacted_tissue_upper_bound_mm3'],
                'criterion':'complete accepted route with return above both STOP0 and original baseline; no clinical safety claim'}
            if costs.forwards or result['checkpoint_loads'] or result['optimizer_attempts'] or result['gradient_attempts']:raise ValueError('Forbidden learned operation')
            guard();budget.complete(history_complete=True);result['historical_h48_comparison']={'result':PINS['h48_result'],'outcomes':{a:r['outcomes'] for a,r in saved['h48_result']['arms'].items()},'scope':'retained baseline quality only; no matched runtime claim'};result['status']='complete_SELECT013_corridor_comparison'
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error),
            'phase':dict(phase),'committed':bool(getattr(error,'committed',False))});raise
    finally:
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        result.update(wall_seconds=time.perf_counter()-started,native_budget=budget.snapshot(),model_forwards=costs.forwards,
            native_previews_by_arm=previews)
        try:write('costs.json',{'phases':costs.rows,'native_budget':budget.snapshot(),'native_previews_by_arm':previews,
            'timing_scope':'inclusive phase timers; construction/search/rollout/replay/audit are separate; do not sum nested native timers'})
        finally:sink.write(output/'result.json',result,terminal=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',required=True,type=Path);parser.add_argument('--release-sha256',required=True);args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease();release=json.loads(args.release.read_text());source_guard(release,args.release,args.release_sha256)
    output=ROOT/OUTPUT;supervision=output.with_name(output.name+'.supervision')
    if output.exists() or not supervision.is_dir():raise ValueError('Fresh owned output required')
    started=time.perf_counter();summary={'status':'started'}
    def progress(phase,**details):
        temp=supervision/'worker-progress.tmp';temp.write_text(json.dumps({'phase':phase,'seconds':time.perf_counter()-started,**details})+'\n');temp.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Paged route worker deadline')
    old=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Incomplete paged-route comparison')
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_SELECT013_corridor',result_sha256=sha(output/'result.json'))
    except BaseException as error:summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        summary.update(wall_seconds=time.perf_counter()-started,canonical_result_sha256=sha(output/'result.json') if (output/'result.json').is_file() else None)
        with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
