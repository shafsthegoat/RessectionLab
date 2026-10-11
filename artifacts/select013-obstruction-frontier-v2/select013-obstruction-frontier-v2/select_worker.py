"""Same-cap translated obstruction openings at one frozen negative frontier."""
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from comparison_contract import (OUTPUT,OUTPUT_BYTES,WORKER_SECONDS,LIMITS,PINS,
    historical,prior_contract,complete_result,source_guard,sha)


def execute(release,release_sha,output,progress):
    import torch
    sys.path.append(str(ROOT/'build/select013-dead-frontier-witness-v1'))
    from blocker_coverage import inspect_rejected
    from frontier_compare import enabled_fresh,compare_prefix,frontier_branches,mask_record
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_cohort_sequential import _bounded_writes
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.planning_budget import PlanningBudget
    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=LIMITS['native_previews'],seconds=WORKER_SECONDS)
    costs=_CallCosts(budget,0);started=time.perf_counter();originals=[]
    result={'status':'started','subject':'ReMIND-013','role':'SELECT','planned_SELECT_denominator':2,
        'held_cases':['ReMIND-037'],'held_inputs_opened':False,'EVAL_opened':False,'private_reference_reads':0,
        'checkpoint_loads':0,'model_forwards':0,'updates':0,'optimizer_attempts':0,'gradient_attempts':0,
        'accepted_plan_claim':False,'complete_episode':False,'unselected_negative_prefix':True,
        'independent_replay_certificate':False,'committed_actions':0,'committed_prefix_actions':0,
        'diagnostic_previews':0,'diagnostic_STOP_calls':0,
        'lineage_scope':'historical metadata authenticates baseline source only; changed proposal model has no frozen-policy capability',
        'failure_semantics':'first rejected shaft interval only; newly feasible original rays are previewed but never injected or executed'}
    def forbid(owner,name,key):
        old=getattr(owner,name);originals.append((owner,name,old))
        def refused(*a,**k):result[key]+=1;raise RuntimeError('Fixed diagnostic forbids '+name)
        setattr(owner,name,refused)
    def guard():budget.check();sink.check()
    def write(name,value):
        path=output/name;path.parent.mkdir(exist_ok=True,parents=True);sink.write(path,value)
    def transition():
        result['committed_actions']+=1
        if result['committed_actions']>LIMITS['total_commits']:raise ValueError('Declared transition cap')
    def preview():
        result['diagnostic_previews']+=1
        if result['diagnostic_previews']>LIMITS['diagnostic_previews']:raise ValueError('Declared direct preview cap')
    try:
        forbid(torch,'load','checkpoint_loads');forbid(torch.optim.Optimizer,'__init__','optimizer_attempts')
        forbid(torch.autograd,'backward','gradient_attempts');forbid(torch.autograd,'grad','gradient_attempts')
        saved=historical();training=saved['training'];source_dir=output/'source';source_dir.mkdir()
        cohort_bytes=(ROOT/prior_contract.COHORT).read_bytes()
        active=prior_contract.select_public(prior_contract.read_pinned(saved['release']['public_index']))
        with budget,costs,_bounded_writes(sink):
            with costs.scope('original_public_construction'):
                case,binding,qc,protocol=prepare_public_source(source_dir,saved['release'],PINS['release']['sha256'],None,progress,
                    public_manifest_path=ROOT/active['path'],public_manifest_sha256=active['sha256'],cohort_bytes=cohort_bytes,
                    learning_protocol_hash=semantic_digest(training),proposal_config=NominalCavityProposalConfig(**training['cohort_execution']['proposal_config']),
                    public_target_context_variant=training['public_target_context_variant'],expected_role='SELECT',
                    checkpoint_lineage=saved['lineage'],occupancy_condition=prior_contract.OCCUPANCY,
                    occupancy_inference_protocol=training,post_exposure_condition=prior_contract.EXPOSURE,partial_domain_inputs=True)
                base,context=make_patient_planning_task(case,cohort_bytes=cohort_bytes,source_binding=binding,qc_receipt=qc,protocol=protocol)
                obs=base.observation();world={'source_hash':case.source_hash,'decision_model_hash':base.decision_model_hash,
                    'initial_observation_hash':obs.fingerprint,'max_steps':base.max_steps,'proposal_hash':case.proposal_config.fingerprint,
                    'occupancy_condition':context.record()['occupancy_condition'],'post_exposure':thaw_json(case.post_exposure.record),
                    'source_domains':thaw_json(case._domain_record),'action_ids':list(obs.action_ids),'action_mask':obs.action_mask.tolist()}
                result['world_equal']=world==saved['world'];result['context_equal']=context.record()==saved['context']
                write('world-control.json',{'exact':result['world_equal'],'context_exact':result['context_equal'],'actual':world,'expected':saved['world']})
                if not result['world_equal'] or not result['context_equal']:raise ValueError('Exact baseline world/context differs')
                baseline=base.planning_clone();context.require_task(baseline)
                del obs,base,binding,qc,protocol
            with costs.scope('enabled_fresh_source_and_construction'):
                enabled,change=enabled_fresh(baseline);write('declared-proposal-change.json',change)
                result['configuration_change_declared']=True
            with costs.scope('paired_fixed_prefix_replay'):
                mapped,old_history,new_history=compare_prefix(baseline,enabled,saved['prefix'],saved['witness_history'],
                    write=write,guard=guard,on_transition=transition)
                result['committed_prefix_actions']=22;result['prefix_exact']=True
                inventory=baseline.candidate_inventory()
                result['baseline_frontier_exact']=semantic_digest(inventory)==semantic_digest(saved['witness_inventory'])
                if not result['baseline_frontier_exact'] or baseline.terminated or baseline._steps!=11 or baseline._inventory:
                    raise ValueError('Saved original dead frontier differs')
                write('baseline-frontier-inventory.json',inventory)
                write('paired-frontier-physical-state.json',{'equal':mask_record(baseline)==mask_record(enabled),
                    'baseline':mask_record(baseline),'enabled':mask_record(enabled),
                    'baseline_state_hash':baseline._engine.state_hash,'enabled_state_hash':enabled._engine.state_hash,
                    'baseline_total_return':baseline._total_reward,'enabled_total_return':enabled._total_reward,
                    'baseline_actions':list(saved['prefix']),'enabled_actions':mapped})
            with costs.scope('bounded_added_preparations_and_original_ray_repreviews'):
                branches=frontier_branches(enabled,inventory,saved['coverage']['outside_every_fixed_axis_tip_envelope'],
                    write=write,inspect=lambda task,ray:inspect_rejected(task,ray,cell_limit=LIMITS['cells_per_sidecar']),
                    guard=guard,on_transition=transition,on_preview=preview,max_branches=LIMITS['max_added_branches'])
                result.update(branches);result['branches_complete']=True
                result['diagnostic_STOP_calls']=branches['branch_count']
            with costs.scope('baseline_and_enabled_frontier_STOP_controls'):
                for name,task in [('original',baseline),('enabled',enabled)]:
                    stop=task.step('STOP');transition();result['diagnostic_STOP_calls']+=1
                    if not task.terminated or stop.reward!=0:raise ValueError('STOP control failed')
                    write(name+'-closed-negative-history.json',{'history':task.metrics()['history'],'return':task._total_reward,
                        'root_STOP_return':0.,'accepted_plan_claim':False,'complete_episode':True})
            if costs.forwards or result['checkpoint_loads'] or result['optimizer_attempts'] or result['gradient_attempts']:raise ValueError('Forbidden learned operation')
            guard();result['complete_episode']=True;budget.complete(history_complete=True)
            result['status']='complete_fixed_obstruction_frontier_diagnostic'
            del baseline,enabled,case,context;gc.collect()
    except BaseException as e:
        result.update(status='failed_or_unresolved',failure={'type':type(e).__name__,'message':str(e),'committed':bool(getattr(e,'committed',False))});raise
    finally:
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        result.update(wall_seconds=time.perf_counter()-started,native_budget=budget.snapshot(),model_forwards=costs.forwards)
        try:write('costs.json',{'phases':costs.rows,'native_budget':budget.snapshot(),'timing_scope':'inclusive; do not sum nested scopes'})
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
    def deadline(*unused):raise TimeoutError('Fixed witness deadline')
    old=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Incomplete diagnostic')
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_SELECT013_obstruction_frontier',result_sha256=sha(output/'result.json'))
    except BaseException as e:summary.update(status='failed_or_unresolved',failure={'type':type(e).__name__,'message':str(e)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        summary.update(wall_seconds=time.perf_counter()-started,canonical_result_sha256=sha(output/'result.json') if (output/'result.json').is_file() else None)
        with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
