"""Four fixed zero-shot/public-planner arms; no patient/model load on import."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import weakref

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from comparison_contract import (ARMS,ENDPOINTS,COHORT,OUTPUT,LIMITS,MODE,WORKER_SECONDS,
    OUTPUT_BYTES,OCCUPANCY,source_guard,sha,read_pinned,select_public,authenticate_comparison,complete_result)

def returned_search_is_capped(accounting):
    """Beam call exhaustion returns normally; time exhaustion usually raises."""
    return accounting.get('call_cap_reached') is True or accounting.get('time_cap_reached') is True

def execute(release,release_sha,output,progress):
    import torch
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_cohort_sequential import _bounded_writes
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab.patient_select_inference import (load_select_checkpoint,collect_select_greedy,
        collect_select_search,seal_and_replay_select)
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.observed_search import observed_beam_search,ObservedSearchLimit
    from resectionlab.planning_budget import PlanningBudget

    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=LIMITS['max_native_previews'],seconds=WORKER_SECONDS)
    costs=preflight._CallCosts(budget,LIMITS['max_policy_forwards'])
    started=time.perf_counter();checkpoints={};originals=[]
    result={'status':'started','subject':'ReMIND-013','role':'SELECT','planned_SELECT_denominator':2,
        'held_cases':['ReMIND-037'],'held_inputs_opened':False,'EVAL_opened':False,
        'private_reference_reads':0,'optimizer_updates_on_SELECT':0,'optimizer_attempts':0,
        'gradient_attempts':0,'arm_order':list(ARMS),'arms':{a:{'status':'not_started'} for a in ARMS},
        'checkpoint_loads':0,'task_condition':'PARTIAL_TARGET_PROGRESS',
        'occupancy_condition':OCCUPANCY,'derived_material_assumption':True,
        'full_supplied_target_voxels':35260,'raw_support_unsupported_target_voxels':33681,
        'comparison_condition':'same explicit S-union-T world and public inputs across all four arms',
        'equal_training_or_planning_budget_claim':False,
        'anatomical_accuracy_claim':False,'physical_surgery_claim':False}
    def forbid(owner,name,count):
        original=getattr(owner,name);originals.append((owner,name,original))
        def refused(*args,**kwargs):
            result[count]+=1;raise RuntimeError('Zero-update inference forbids '+name)
        setattr(owner,name,refused)
    def guard():budget.check();sink.check()
    try:
        forbid(torch.optim.Optimizer,'__init__','optimizer_attempts')
        forbid(torch.autograd,'backward','gradient_attempts')
        forbid(torch.autograd,'grad','gradient_attempts')
        endpoint_data=authenticate_comparison(release)
        active=select_public(read_pinned(release['public_index']))
        cohort_bytes=(ROOT/COHORT).read_bytes()
        with budget,costs,_bounded_writes(sink):
            # All predetermined endpoints must load/authenticate before the first
            # public array is opened, including any canonical refusal of IL64.
            for name in ENDPOINTS:
                pin=endpoint_data[name]
                with costs.scope('checkpoint_reload.'+name):
                    checkpoint=load_select_checkpoint(ROOT/pin['checkpoint']['path'],
                        expected_sha256=pin['checkpoint']['sha256'],expected_learning_protocol=pin['protocol'],
                        expected_context_hashes=pin['context_hashes'],expected_method=pin['method'],
                        training_release_sha256=pin['training_release_sha256'])
                    lineage=checkpoint.lineage_record()
                    if lineage['completed_updates']!=pin['updates'] or lineage['parameter_hash']!=pin['parameter_hash']:
                        raise ValueError('Reloaded terminal endpoint differs')
                    checkpoints[name]=checkpoint;result['checkpoint_loads']+=1
                    sink.write(output/(name+'-lineage.json'),lineage)
            expected_world=None
            for arm in ARMS:
                progress('arm_start',arm=arm)
                dest=output/arm;dest.mkdir(exist_ok=False)
                source_dir=dest/'source';source_dir.mkdir(exist_ok=False)
                name=arm if arm in ENDPOINTS else ENDPOINTS[0]
                checkpoint=checkpoints[name];training=endpoint_data[name]['protocol']
                with costs.scope(arm+'.public_construction'):
                    source,binding,qc,protocol=prepare_public_source(source_dir,release,release_sha,None,progress,
                        public_manifest_path=ROOT/active['path'],public_manifest_sha256=active['sha256'],
                        cohort_bytes=cohort_bytes,learning_protocol_hash=semantic_digest(training),
                        proposal_config=NominalCavityProposalConfig(**training['cohort_execution']['proposal_config']),
                        public_target_context_variant=training['public_target_context_variant'],
                        expected_role='SELECT',checkpoint_lineage=checkpoint.lineage_record(),
                        occupancy_condition=OCCUPANCY,occupancy_inference_protocol=training)
                    base,context=make_patient_planning_task(source,cohort_bytes=cohort_bytes,
                        source_binding=binding,qc_receipt=qc,protocol=protocol)
                    checkpoint.require_task(base,context)
                    observation=base.observation()
                    world={'source_hash':source.source_hash,'decision_model_hash':base.decision_model_hash,
                        'initial_observation_hash':observation.fingerprint,'max_steps':base.max_steps,
                        'proposal_hash':source.proposal_config.fingerprint,'occupancy_condition':context.record()['occupancy_condition'],
                        'action_ids':list(observation.action_ids),'action_mask':observation.action_mask.tolist()}
                    if expected_world is not None and world!=expected_world:
                        raise ValueError('Exact public world or initial action inventory changed across arms')
                    expected_world=world;sink.write(dest/'comparison-world.json',world)
                    sink.write(dest/'context.json',context.record())
                    source_ref=weakref.ref(source)
                del source,observation,binding,qc,protocol
                try:
                    with costs.scope(arm+'.planning_and_replay'):
                        selection_started=time.perf_counter()
                        selection_previews=budget.snapshot()['native_preview_entries']
                        accounting=None
                        if arm=='observed_beam':
                            actions,accounting=observed_beam_search(base,**LIMITS['search'],policy=None,
                                objective_source='permitted_nominal_target_and_frozen_geometric_costs',
                                retained_prefix_diagnostics=True,retention_mode=MODE)
                            if returned_search_is_capped(accounting):
                                raise ObservedSearchLimit('Beam returned capped, unresolved search',
                                    accounting=accounting,best_sequence=actions)
                            sink.write(dest/'selection.json',{'actions':actions,'accounting':accounting,
                                'wall_seconds':time.perf_counter()-selection_started,
                                'native_previews':budget.snapshot()['native_preview_entries']-selection_previews,
                                'scope':'selection only; construction, trace collection and independent replay charged separately'})
                            trace=collect_select_search(base,context,checkpoint,actions=actions,
                                accounting=accounting,output=dest,guard=guard)
                        elif arm=='observed_greedy':
                            actions,accounting=base.observed_greedy_search(seconds=release['greedy_seconds'])
                            sink.write(dest/'selection.json',{'actions':actions,'accounting':accounting,
                                'wall_seconds':time.perf_counter()-selection_started,
                                'native_previews':budget.snapshot()['native_preview_entries']-selection_previews,
                                'scope':'selection only; construction, trace collection and independent replay charged separately'})
                            trace=collect_select_search(base,context,checkpoint,actions=actions,
                                accounting=accounting,output=dest,guard=guard)
                        else:
                            trace=collect_select_greedy(base,context,checkpoint,output=dest,guard=guard)
                        sealed=seal_and_replay_select(base,context,trace,checkpoint,output=dest,guard=guard)
                        history=sealed['replayed_history']
                        result['arms'][arm]={'status':'complete_replayed','steps':len(trace.transitions),
                            'actions':[r.action_id for r in trace.transitions],
                            'public_return':sum(r.reward for r in trace.transitions),
                            'target_removed_mm3':sum(r.get('target_removed_mm3',0.) for r in history),
                            'outside_supplied_target_removed_mm3':sum(r.get('normal_removed_mm3',0.) for r in history),
                            'outside_meaning':'outside supplied target, not verified normal tissue',
                            'plan_seal':sealed['plan_seal'],'context_hash':context.fingerprint,
                            'checkpoint_author':name if arm in ENDPOINTS else None,
                            'comparison_reference_checkpoint':name,'source_hash':world['source_hash'],
                            'decision_model_hash':world['decision_model_hash'],
                            'selector_accounting':accounting}
                        del trace,sealed,history
                except ObservedSearchLimit as error:
                    # A local search timeout is retained as unresolved. No best
                    # partial prefix becomes an accepted STOP or learned label.
                    result['arms'][arm]={'status':'search_unresolved','reason':str(error),
                        'accounting':error.accounting,'partial_actions':list(error.best_sequence)}
                    sink.write(dest/'search-unresolved.json',result['arms'][arm]);guard()
                finally:
                    del base,context;gc.collect()
                if source_ref() is not None:raise RuntimeError('Public source retained between arms')
                sink.write(dest/'summary.json',result['arms'][arm]);guard()
            for checkpoint in checkpoints.values():checkpoint.require()
            if costs.forwards!=sum(result['arms'][name]['steps'] for name in ENDPOINTS):
                raise ValueError('Learned rollout steps and measured policy forwards differ')
            result['status']=('complete_union_obstruction_SELECT013_comparison' if all(
                row['status']=='complete_replayed' for row in result['arms'].values()) else 'comparison_retains_unresolved_search')
            guard();budget.complete(history_complete=True)
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error),
            'committed':bool(getattr(error,'committed',False))})
        raise
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)
        result['wall_seconds']=time.perf_counter()-started
        result['native_budget']=budget.snapshot();result['total_policy_forward_calls']=costs.forwards
        try:sink.write(output/'costs.json',{'phases':costs.rows,'native_budget':budget.snapshot(),
            'policy_forwards':costs.forwards,'timing_scope':'inclusive phases; do not sum nested work'})
        finally:sink.write(output/'result.json',result,terminal=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    release=json.loads(args.release.read_text());source_guard(release,args.release,args.release_sha256)
    output=ROOT/OUTPUT;supervision=output.with_name(output.name+'.supervision')
    if output.exists() or not supervision.is_dir():raise ValueError('Fresh output and owned parent required')
    started=time.perf_counter();summary={'status':'started'}
    def progress(phase,**details):
        temp=supervision/'worker-progress.tmp';temp.write_text(json.dumps({'phase':phase,
            'seconds':time.perf_counter()-started,**details})+'\n');temp.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Fixed SELECT worker deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        summary['runtime']={'python':sys.version,'executable':sys.executable,
            'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','dtype':'float32',
            'torch_default_dtype':str(torch.get_default_dtype()),'cpu_architecture':platform.machine(),
            'logical_cpu_count':os.cpu_count(),'threads':torch.get_num_threads(),
            'interop_threads':torch.get_num_interop_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Comparison has unresolved arms; no complete comparison claim')
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_union_obstruction_SELECT013',result_sha256=sha(output/'result.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
