"""One public prospective catalog after an exact old-rule 47-motion prefix."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,OUTPUT_BYTES,WORKER_SECONDS,LIMITS,PINS,PRIOR,
    historical,prior_contract,complete_result,source_guard,sha)


def execute(release,release_sha,output,progress):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    sys.path.append(str(PRIOR))
    from frontier_compare import source_physics,post_exposure_authority
    from residual_frontier import diagnose,state
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.paged_obstruction import PagedObstructionTask
    from resectionlab.evaluation import independent_check_native_history
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_cohort_sequential import _bounded_writes
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.planning_budget import PlanningBudget
    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=LIMITS['native_previews'],seconds=WORKER_SECONDS)
    costs=_CallCosts(budget,0);started=time.perf_counter();originals=[];phase='initialization'
    result={'status':'started','subject':'ReMIND-013','role':'SELECT','planned_SELECT_denominator':2,
        'held_cases':['ReMIND-037'],'held_inputs_opened':False,'EVAL_opened':False,
        'checkpoint_loads':0,'model_forwards':0,'updates':0,'optimizer_attempts':0,'gradient_attempts':0,
        'private_reference_reads':0,'native_commit_attempts':0,'native_commits':0,'STOP_calls':0,
        'independent_geometry_checks':0,'independent_geometry_all_feasible':False,
        'accepted_plan_claim':False,'new_rule_generated_prefix':False,
        'scope':'unselected negative old prefix and <=3 prospective one-prep diagnostic branches; no accepted complete route',
        'limits':LIMITS,'release_sha256':release_sha}
    def write(name,value):
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        sink.write(path,value);return sha(path)
    def guard():
        budget.check()
        if time.perf_counter()-started>WORKER_SECONDS:raise TimeoutError('Residual diagnostic wall ceiling')
    def forbid(owner,name,key):
        old=getattr(owner,name);originals.append((owner,name,old))
        def refused(*a,**k):result[key]+=1;raise RuntimeError('No learned operation: '+name)
        setattr(owner,name,refused)
    def counted_commit(self,certificate):
        result['native_commit_attempts']+=1
        if result['native_commit_attempts']+result['STOP_calls']>LIMITS['native_commits_and_STOP']:
            raise ValueError('Diagnostic commit/STOP ceiling')
        value=original_commit(self,certificate);result['native_commits']+=1;return value
    try:
        forbid(torch,'load','checkpoint_loads');forbid(torch.optim.Optimizer,'__init__','optimizer_attempts')
        forbid(torch.autograd,'backward','gradient_attempts');forbid(torch.autograd,'grad','gradient_attempts')
        original_commit=NativeResectionEngine.commit_preview
        originals.append((NativeResectionEngine,'commit_preview',original_commit))
        NativeResectionEngine.commit_preview=counted_commit
        saved=historical();training=saved['training'];cohort=(ROOT/prior_contract.COHORT).read_bytes()
        active=prior_contract.select_public(prior_contract.read_pinned(saved['release']['public_index']))
        with costs,_bounded_writes(sink):
            with budget:
                phase='source_and_construction';progress(phase)
                with costs.scope(phase):
                    folder=output/'source';folder.mkdir()
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
                    result.update(world_equal=world==saved['world'],context_equal=context.record()==saved['context'])
                    write('original-world-control.json',{'world':world,'context':context.record(),
                        'world_equal':result['world_equal'],'context_equal':result['context_equal']})
                    if not result['world_equal'] or not result['context_equal']:raise ValueError('Original admitted world/context differs')
                    changed=replace(case,proposal_config=replace(case.proposal_config,obstruction_opening=True,obstruction_page_size=8))
                    authority=post_exposure_authority(case.post_exposure,changed.post_exposure)
                    if source_physics(changed)!=source_physics(case):raise ValueError('Old paged authority changed physical source')
                    task=PagedObstructionTask(changed,max_steps=LIMITS['max_steps'],reward=base.reward_spec,cancelled=base._cancelled).planning_clone()
                    expected=saved['frontier_page']
                    if (task.case.source_hash!=expected['source_hash'] or task.decision_model_hash!=expected['decision_model_hash']
                            or task.case.proposal_config.obstruction_selection_rule is not None):
                        raise ValueError('Old47 source/proposal/task authority differs')
                    write('old-authority.json',{'source_hash':task.case.source_hash,'decision_model_hash':task.decision_model_hash,
                        'proposal_config':task.case.proposal_config.to_record(),'post_exposure_authority':authority,
                        'prefix':list(saved['prefix']),'new_rule_generated_prefix':False})
                    del base,obs,binding,qc,protocol,context
                phase='fixed47_reconstruction';progress(phase)
                with costs.scope(phase):
                    total=target=outside=distance=0.;checks=[]
                    for i,action in enumerate(saved['prefix']):
                        guard();write('prefix/attempt%02d.json'%i,{'action_id':action})
                        outcome=task.advance_planning(action);write('prefix/step%02d.json'%i,outcome.info)
                        total+=outcome.reward;target+=outcome.info['target_removed_mm3'];outside+=outcome.info['normal_removed_mm3']
                        distance=max(distance,outcome.info['insertion_distance_mm'])
                        observed={'actions':list(saved['prefix'][:i+1]),'estimated_incremental_return':total,
                            'target_removed_mm3':target,'outside_supplied_target_removed_mm3':outside,'max_insertion_distance_mm':distance}
                        want=saved['prefix_checkpoints'][i]
                        exact=all(observed[k]==want[k] for k in observed);checks.append({'step':i+1,'exact':exact,'progress':observed})
                        if not exact:raise ValueError('Fixed prefix physical reward/progress differs at step '+str(i+1))
                    if task.terminated or task._steps!=47:raise ValueError('Original47 frontier must remain nonterminal')
                    write('prefix-controls.json',checks);write('original-native-history.json',task._engine.history)
                    result['prefix_exact']=True;result['prefix_return']=total
                phase='old_frontier_inventory';progress(phase)
                with costs.scope(phase):
                    pages=[];legal=set()
                    for page,ids in task.planning_action_pages():
                        guard();record=page.candidate_inventory();pages.append(record);legal.update(ids)
                        write('old-frontier/page%02d.json'%record['page_index'],record)
                    exact=semantic_digest(pages[0])==semantic_digest(saved['frontier_page'])
                    result['frontier_page_exact']=exact
                    if not exact or len(legal)!=7:raise ValueError('Authenticated old frontier inventory differs')
                phase='prospective_catalog_and_branches';progress(phase)
                with costs.scope(phase):
                    before=state(task._engine)
                    diagnosed=diagnose(task,write=lambda name,value:write('prospective/'+name,value),guard=guard,max_branches=LIMITS['branch_cap'])
                    branch_histories=diagnosed.pop('_audit_histories');result.update(diagnosed)
                    if state(task._engine)!=before:raise ValueError('Old authority mutated by prospective catalog')
                    # This source namespace is exactly the independent native checker input,
                    # without forging or mutating any task/checkpoint authority.
                    source=SimpleNamespace(mri=task.case.structural_intensity,affine=task.case._native_affine_ras_mm,
                        frame='RAS+',semantic_hash=task._engine.config.source_hash)
                    histories=[('original-prefix',task._engine.history)]+[('branch-%02d'%i,h) for i,h in enumerate(branch_histories)]
                phase='original_negative_prefix_STOP';progress(phase)
                with costs.scope(phase):
                    guard();result['STOP_calls']+=1
                    if result['native_commit_attempts']+result['STOP_calls']>LIMITS['native_commits_and_STOP']:raise ValueError('STOP budget')
                    outcome=task.advance_planning('STOP');write('original-prefix-STOP.json',outcome.info)
                    result['original_negative_prefix_closed']=task.terminated and task._steps==48 and outcome.reward==0 and task._total_reward==total
                    if not result['original_negative_prefix_closed']:raise ValueError('Negative diagnostic prefix closure differs')
                    write('original-negative-prefix-metrics.json',task.metrics())
                    guard();budget.complete(history_complete=True)
            # Geometry audit uses the separate implementation only after leaving
            # the planning profiler. All prior native costs remain charged.
            phase='independent_full_history_geometry';progress(phase)
            certificates=[]
            with costs.scope(phase):
                for name,history in histories:
                    def cancelled():
                        if time.perf_counter()-started>WORKER_SECONDS:raise TimeoutError('Independent audit wall ceiling')
                        return False
                    stamp=time.perf_counter()
                    audit=independent_check_native_history(source,task.case.tools,history,
                        tissue_mask=task.case.observed_support,access=task.case.access,geometry_frame='RAS+',
                        tool_modes=task.tool_modes,interaction_domain=task.case._interaction_domain,
                        post_exposure=task.case.post_exposure,cancelled=cancelled)
                    certificate={'scope':'independent full native geometry; not a complete task episode acceptance',
                        'name':name,'history_hash':semantic_digest(history),'wall_seconds':time.perf_counter()-stamp,
                        'audit':audit.to_dict()}
                    pin=write('independent/'+name+'.json',certificate)
                    result['independent_geometry_checks']+=1
                    certificates.append({'path':'independent/'+name+'.json','sha256':pin,'feasible':audit.feasible})
                    # Always save refusal evidence before raising.
                    if not audit.feasible:raise ValueError('Independent full-history geometry refused '+name)
                result['independent_geometry_all_feasible']=True;result['independent_certificates']=certificates
            if costs.forwards or any(result[k] for k in ('checkpoint_loads','optimizer_attempts','gradient_attempts')):
                raise ValueError('Forbidden learned operation')
            result['status']='complete_prospective_residual_frontier_diagnostic'
            if not complete_result(result):raise ValueError('Diagnostic counts/completion mismatch')
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error),
            'phase':phase,'committed':bool(getattr(error,'committed',False))});raise
    finally:
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        result.update(wall_seconds=time.perf_counter()-started,native_budget=budget.snapshot(),model_forwards=costs.forwards)
        try:write('costs.json',{'phases':costs.rows,'native_budget':budget.snapshot(),
            'native_commit_attempts':result['native_commit_attempts'],'native_commits':result['native_commits'],'STOP_calls':result['STOP_calls'],
            'timing_scope':'separate inclusive source/reconstruction/catalog/native-branch/STOP/independent-audit phases; native sub-timers nest'})
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
    def deadline(*unused):raise TimeoutError('Residual diagnostic worker deadline')
    old=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Incomplete residual diagnostic')
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_SELECT013_residual_frontier',result_sha256=sha(output/'result.json'))
    except BaseException as error:summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        summary.update(wall_seconds=time.perf_counter()-started,canonical_result_sha256=sha(output/'result.json') if (output/'result.json').is_file() else None)
        with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
