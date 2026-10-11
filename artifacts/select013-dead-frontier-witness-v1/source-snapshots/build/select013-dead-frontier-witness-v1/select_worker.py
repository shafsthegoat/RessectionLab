"""Fixed unselected negative prefix and first-hit witness; no policy or accepted plan."""
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
    from blocker_coverage import inspect_rejected,classify_cells
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
        'independent_replay_certificate':False,'committed_actions':0,'diagnostic_previews':0,
        'lineage_scope':'authenticated historical metadata for source reconstruction only; no frozen-policy capability constructed',
        'source_protocol_limits_scope':'historical context envelope preserved; smaller current owned limits enforced separately',
        'failure_semantics':'first rejected shaft interval, not exact first contact or complete corridor'}
    def forbid(owner,name,key):
        old=getattr(owner,name);originals.append((owner,name,old))
        def refused(*a,**k):result[key]+=1;raise RuntimeError('Fixed witness forbids '+name)
        setattr(owner,name,refused)
    def guard():budget.check();sink.check()
    try:
        forbid(torch,'load','checkpoint_loads');forbid(torch.optim.Optimizer,'__init__','optimizer_attempts')
        forbid(torch.autograd,'backward','gradient_attempts');forbid(torch.autograd,'grad','gradient_attempts')
        saved=historical();training=saved['training'];source_dir=output/'source';source_dir.mkdir()
        cohort_bytes=(ROOT/prior_contract.COHORT).read_bytes()
        active=prior_contract.select_public(prior_contract.read_pinned(saved['release']['public_index']))
        with budget,costs,_bounded_writes(sink):
            with costs.scope('public_construction'):
                # Reconstruct the exact historical context; current execution is
                # separately capped and cannot exercise a model capability.
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
                sink.write(output/'world-control.json',{'exact':result['world_equal'],'context_exact':result['context_equal'],'actual':world,'expected':saved['world']})
                if not result['world_equal'] or not result['context_equal']:raise ValueError('Exact historical world/context differs')
                task=base.planning_clone();context.require_task(task)
                del obs,base,binding,qc,protocol
            history=[];total=target=outside=distance=0.
            with costs.scope('fixed_prefix_replay'):
                for i,action in enumerate(saved['prefix']):
                    guard();progress('fixed_prefix',step=i+1)
                    if result['committed_actions']>=LIMITS['fixed_prefix_commits']:raise ValueError('Fixed commit cap')
                    before=task._engine.state_hash;outcome=task.step(action);result['committed_actions']+=1
                    row=outcome.info
                    if row['outcome_scope']!='permitted_nominal_model':raise ValueError('Private/reference feedback refused')
                    history.append(row);total+=outcome.reward;target+=row['target_removed_mm3'];outside+=row['normal_removed_mm3'];distance=max(distance,row['insertion_distance_mm'])
                    actual={'actions':list(saved['prefix'][:i+1]),'estimated_incremental_return':total,
                        'target_removed_mm3':target,'outside_supplied_target_removed_mm3':outside,
                        'max_insertion_distance_mm':distance,'legal_nonstop_actions':len(task._inventory)}
                    expected=saved['prefix_checkpoints'][i]
                    record={'source_state_before':before,'source_state_after':task._engine.state_hash,
                        'actual':actual,'expected':{k:expected[k] for k in actual},'history':row}
                    sink.write(output/('prefix-%02d.json'%(i+1)),record)
                    if actual!=record['expected']:raise ValueError('Saved fixed prefix changed at '+str(i+1))
                result.update(prefix_exact=True,final_legal_motions=len(task._inventory),public_prefix_return=total,
                    target_removed_mm3=target,outside_supplied_target_removed_mm3=outside)
                if task.terminated or task._steps!=11 or task._inventory:raise ValueError('Exact nonterminal zero-motion frontier required')
                start=budget.snapshot()['native_preview_entries'];inventory=task.candidate_inventory()
                if budget.snapshot()['native_preview_entries']!=start:raise ValueError('Saved inventory unexpectedly queried native geometry')
                sink.write(output/'dead-frontier-inventory.json',inventory)
                sink.write(output/'diagnostic-prefix-history.json',history)
            state=task._engine.state_hash;rows=[];cells=set()
            with costs.scope('first_rejection_sidecars'):
                proposals=task._proposal_batch.proposals
                if len(proposals)>120:raise ValueError('Fixed original candidate cap exceeded')
                for i,ray in enumerate(proposals):
                    guard();progress('first_rejection',proposal=i+1)
                    if result['diagnostic_previews']>=LIMITS['diagnostic_previews']:raise ValueError('Diagnostic query cap')
                    record=inspect_rejected(task,ray,cell_limit=LIMITS['cells_per_sidecar']);result['diagnostic_previews']+=1
                    if record['feasible']:raise ValueError('Dead frontier contains feasible emitted ray')
                    d=record['diagnostic']
                    if d:cells.update(tuple(c) for c in d['blocked_indices_native'])
                    sink.write(output/('rejection-%03d.json'%i),record);rows.append(record)
            selected=sorted(cells)[:LIMITS['classified_unique_cells']]
            with costs.scope('actual_cell_capacity'):
                coverage=classify_cells(task,selected,guard=guard)
                coverage.update(retained_unique_first_blocker_cells=len(cells),classified_unique_cells=len(selected),
                    classification_truncated=len(cells)>len(selected),unclassified_retained_cells=[list(c) for c in sorted(cells)[len(selected):]],
                    sidecar_evidence_truncated=any(r['diagnostic'] and r['diagnostic']['truncated'] for r in rows),
                    complete_corridor_claim=False,global_unreachability_claim=False)
                sink.write(output/'actual-blocker-capacity.json',coverage)
            result.update(state_unchanged_by_diagnostics=task._engine.state_hash==state,
                first_shaft_rejections=sum(r['diagnostic'] is not None for r in rows),
                final_source_state_hash=state,classified_cells=len(selected),
                cells_outside_all_fixed_axis_envelopes=coverage['outside_every_fixed_axis_tip_envelope'],
                additional_containment_kernel_calls=coverage['native_containment_kernel_calls'],
                scope='one selected saved negative prefix; actual current blockers and frozen-axis capacity, not a new accepted plan')
            if costs.forwards or result['checkpoint_loads'] or result['optimizer_attempts'] or result['gradient_attempts']:raise ValueError('Forbidden learned operation')
            guard()
            # Close the fixed negative diagnostic history only after preserving
            # its exact nonterminal dead frontier and all first-hit evidence.
            with costs.scope('diagnostic_STOP_closure'):
                stopped=task.step('STOP');history.append(stopped.info)
                result['diagnostic_STOP_calls']=1
                if not task.terminated or stopped.reward!=0:raise ValueError('Diagnostic STOP closure failed')
                sink.write(output/'closed-negative-diagnostic-history.json',history)
            result['complete_episode']=True
            budget.complete(history_complete=True)
            result['status']='complete_fixed_dead_frontier_diagnostic'
            del task,case,context;gc.collect()
    except BaseException as e:
        result.update(status='failed_or_unresolved',failure={'type':type(e).__name__,'message':str(e),'committed':bool(getattr(e,'committed',False))});raise
    finally:
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        result.update(wall_seconds=time.perf_counter()-started,native_budget=budget.snapshot(),model_forwards=costs.forwards)
        try:sink.write(output/'costs.json',{'phases':costs.rows,'native_budget':budget.snapshot(),'timing_scope':'inclusive; do not sum nested scopes'})
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
        summary.update(status='complete_owned_SELECT013_dead_frontier',result_sha256=sha(output/'result.json'))
    except BaseException as e:summary.update(status='failed_or_unresolved',failure={'type':type(e).__name__,'message':str(e)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        summary.update(wall_seconds=time.perf_counter()-started,canonical_result_sha256=sha(output/'result.json') if (output/'result.json').is_file() else None)
        with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
