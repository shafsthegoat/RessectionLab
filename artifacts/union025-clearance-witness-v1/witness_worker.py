"""Owned fixed union025 depth-two diagnostic; no search policy or learned model."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import signal
import sys
import time
import weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from witness_contract import (OUTPUT,SUBJECT,CONDITION,EXECUTION,WORKER_SECONDS,OUTPUT_BYTES,
    inputs,canonical,semantic,sha,small,physical,same_ray,source_guard,complete_result,endpoint_control)

@contextmanager
def execution_counters(guard):
    """Observe existing transitions and prohibit model/search calls; no dynamics edit."""
    import torch
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.spatial_policy import SpatialPolicy
    counters={'phase':'setup','native_counts':{'rollout':0,'replay':0},'diagnostic_previews':0,
        'prohibited_calls':{'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'search_calls':0}}
    original=NativeSpatialTask._transition;saved=[]
    def counted(task,*args,**kwargs):
        guard();phase=counters['phase'];counts=counters['native_counts']
        if phase not in counts:raise ValueError('Native transition outside declared phase')
        if sum(counts.values())>=92 or counts[phase]>=46:raise InterruptedError('Native transition cap')
        counts[phase]+=1
        return original(task,*args,**kwargs)
    def prohibit(owner,name,key):
        old=getattr(owner,name);saved.append((owner,name,old))
        def refused(*args,**kwargs):
            counters['prohibited_calls'][key]+=1
            raise RuntimeError('Prohibited scientific work: '+key)
        setattr(owner,name,refused)
    NativeSpatialTask._transition=counted
    prohibit(SpatialPolicy,'forward','policy_forwards');prohibit(torch.optim.Adam,'step','optimizer_updates')
    prohibit(torch,'load','checkpoint_loads');prohibit(NativeSpatialTask,'observed_greedy_search','search_calls')
    try:yield counters
    finally:
        NativeSpatialTask._transition=original
        for owner,name,old in reversed(saved):setattr(owner,name,old)

def execute(release,release_sha,output,progress):
    import numpy as np
    from resectionlab.core import thaw_json,freeze_json,array_digest
    from resectionlab.native_resection import NativeResectionEngine,_hash_array
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab import public_patient_factory as factory
    records,refs,ray,preps=inputs();baseline=records['learning_release']
    cohort_bytes=(ROOT/refs['cohort']['path']).read_bytes()
    config=NominalCavityProposalConfig(**baseline['learning_protocol']['cohort_execution']['proposal_config'])
    target_variant=baseline['learning_protocol']['public_target_context_variant']
    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=10000,seconds=WORKER_SECONDS)
    started=time.perf_counter();counters=None;life=[None]
    result={'status':'started','subject':SUBJECT,'condition':CONDITION,'source_released':False,
        'branches':[{'ordinal':i,'status':'not_run','preparation_action':None if i==0 else preps[i-1]['action_id']} for i in range(16)],
        'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'private_reference_reads':0,'search_calls':0,
        'SELECT_EVAL_opened':False,'010_omitted_rays':'not accessed; deferred',
        'interpretation':'Supplied S union T is an unvalidated material assumption. First-failure blockers are neither a full corridor nor a minimum clearing set.',
        'negative_scope':'This tests all15 existing legal root moves followed by one fixed ray only, not longer preparation or global STOP optimality.'}
    def guard():budget.check();sink.check()
    def write(path,value):sink.write(path,thaw_json(freeze_json(value)))

    def run_world():
        admission=release['condition_admission'];srcdir=output/'source';srcdir.mkdir()
        progress('public_union025_source')
        source,binding,qc,protocol=factory.prepare_public_source(srcdir,{**release,'limits':admission['limits']},release_sha,None,progress,
            public_manifest_path=ROOT/refs['public_manifest']['path'],public_manifest_sha256=refs['public_manifest']['sha256'],
            cohort_bytes=cohort_bytes,learning_protocol_hash=admission['learning_protocol_hash'],proposal_config=config,
            public_target_context_variant=target_variant,occupancy_condition=CONDITION)
        life[0]=weakref.ref(source)
        base,context=make_patient_planning_task(source,cohort_bytes=cohort_bytes,source_binding=binding,qc_receipt=qc,protocol=protocol)
        context.require_task(base);obs=base.observation();inventory=base.candidate_inventory()
        old=records['union_world'];initial_hash=obs.fingerprint;initial_state=base._engine.state_hash
        current={'source_hash':source.source_hash,'decision_model_hash':base.decision_model_hash,
            'initial_observation_hash':initial_hash,'context_hash':context.fingerprint}
        for key in ('source_hash','decision_model_hash','initial_observation_hash'):
            if current[key]!=old[key]:raise ValueError('Union025 source/model/observation changed')
        if canonical(inventory)!=canonical(records['union_inventory']):raise ValueError('Original root inventory changed')
        if [a for a in obs.action_ids if a!='STOP']!=[r['action_id'] for r in preps]:
            raise ValueError('Exact15 legal root preparation order changed')
        if not np.array_equal(source.reference_target,source.nominal_target):raise ValueError('Only public target scoring permitted')
        if base.max_steps!=24 or source.proposal_config.max_candidates!=120:raise ValueError('Task horizon/candidate cap changed')
        world={**current,'initial_state_hash':initial_state,'condition':CONDITION,
            'derived_occupancy':thaw_json(source._occupancy_derivation),
            'normalization':thaw_json(source._normalization_record),'supplied_goal_extent':thaw_json(source._supplied_goal_extent),
            'common_public_world':old['common_public_world'],'common_public_world_sha256':semantic(old['common_public_world']),
            'previous_world_sha256':refs['union_world']['sha256'],'independent_reference':'public supplied T only, no private annotations'}
        if (canonical(world['derived_occupancy'])!=canonical(old['derived_occupancy'])
                or canonical(world['normalization'])!=canonical(old['normalization'])
                or canonical(world['supplied_goal_extent'])!=canonical(old['supplied_goal_extent'])):
            raise ValueError('Union provenance/normalization/target denominator changed')
        write(output/'world.json',world);write(output/'context.json',context.record())
        write(output/'initial-inventory.json',inventory)

        def diagnostic(task,dest):
            guard();context.require_task(task)
            if counters['diagnostic_previews']>=16:raise InterruptedError('Fixed-ray diagnostic cap')
            counters['diagnostic_previews']+=1
            before=task._engine.state_hash;history_before=semantic(task.metrics()['history'])
            certificate=task._engine.preview_stroke(ray['tool_id'],ray['tip_mm'],entry_mm=ray['entry_mm'],
                obstruction_diagnostics=True,obstruction_cell_limit=4096)
            if (task._engine.state_hash!=before or semantic(task.metrics()['history'])!=history_before
                    or certificate.source_state_hash!=before or certificate.source_hash!=source.source_hash
                    or certificate.decision_model_hash!=task._engine.config.fingerprint
                    or not same_ray({'tool_id':certificate.tool_id,'entry_mm':certificate.entry_mm,'tip_mm':certificate.tip_mm},ray)):
                raise ValueError('Preview changed state or physical source binding')
            evidence=thaw_json(certificate.obstruction_diagnostic)
            microsteps=[s.to_dict() for s in certificate.microsteps]
            temporary_count=sum(len(s.removed_indices_native) for s in certificate.microsteps)
            temporary_hash=semantic([_hash_array(s.removed_indices_native) for s in certificate.microsteps])
            classification=None
            if certificate.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':
                if evidence is None:raise ValueError('Missing first-failure diagnostic')
                payload={k:v for k,v in evidence.items() if k!='fingerprint'}
                cells=np.asarray(evidence['blocked_indices_native'],dtype=np.int64).reshape(-1,3)
                if (semantic(payload)!=evidence['fingerprint'] or evidence['source_state_hash']!=before
                        or evidence['source_hash']!=source.source_hash
                        or evidence['decision_model_hash']!=task._engine.config.fingerprint
                        or evidence['native_affine_hash']!=array_digest(source._native_affine_ras_mm)
                        or canonical(evidence['requested_tip_mm'])!=canonical(ray['tip_mm'])
                        or canonical(evidence['entry_mm'])!=canonical(ray['entry_mm'])
                        or evidence['retained_cell_count']!=len(cells) or len(cells)>4096
                        or evidence['blocked_cell_count']<len(cells) or evidence['blocked_cell_count']<=0
                        or evidence['prior_temporary_removed_count']!=temporary_count
                        or evidence['prior_temporary_removals_hash']!=temporary_hash
                        or evidence['prior_temporary_removals_committed'] is not False
                        or canonical(evidence['failure_tip_mm'])!=canonical(certificate.failure_tip_mm)
                        or evidence['complete_first_failure_set']!=(evidence['blocked_cell_count']==len(cells))):
                    raise ValueError('Incomplete or inconsistent first-failure accounting')
                if evidence['complete_first_failure_set'] and array_digest(cells)!=evidence['blocked_indices_hash']:
                    raise ValueError('Complete blocker coordinates differ from hash')
                if len(cells):
                    if not np.all(task._engine.remaining_mask[tuple(cells.T)]):raise ValueError('Blocker is not committed remaining material')
                    raw=source.occupancy_source_support[tuple(cells.T)]
                    target=source.nominal_target[tuple(cells.T)]>0
                    classification={'scope':'retained coordinate prefix only','retained_count':len(cells),
                        'in_raw_S':int(raw.sum()),'in_added_T_minus_S':int((target & ~raw).sum()),
                        'in_both_S_and_T':int((raw & target).sum()),'full_first_failure_set':evidence['complete_first_failure_set']}
                    if classification['in_raw_S']+classification['in_added_T_minus_S']!=len(cells):
                        raise ValueError('Blocker outside declared union occupancy')
            elif evidence is not None:raise ValueError('Unexpected diagnostic for non-shaft outcome')
            record={**ray,'feasible':bool(certificate.feasible),'reason':certificate.reason,'committed':False,
                'source_hash':certificate.source_hash,'source_state_hash':before,
                'native_model_hash':certificate.decision_model_hash,'task_model_hash':task.decision_model_hash,
                'axis_unit':certificate.axis_unit,'failure_tip_mm':certificate.failure_tip_mm,
                'prior_completed_microsteps':microsteps,'temporary_removed_count':temporary_count,
                'temporary_removed_hash':temporary_hash,'temporary_removals_committed':False,
                'obstruction_diagnostic':evidence,'blocker_occupancy':classification,
                'feasible_preview_removed_indices_native':certificate.removed_indices_native.tolist() if certificate.feasible else None,
                'scope':'read-only certificate and first rejected interval; not an executed action'}
            write(dest/'fixed-ray-preview.json',record)
            return record

        def run_branch(i):
            row=result['branches'][i];dest=output/f'branch-{i:02d}';dest.mkdir()
            row.update(status='running');progress('branch',ordinal=i,preparation_action=row['preparation_action'])
            before_previews=budget.snapshot()['native_preview_entries'];before_counts=dict(counters['native_counts'])
            branch=base.clone();context.require_task(branch);actions=[]
            def commit(action):
                counters['phase']='rollout';guard();context.require_task(branch)
                write(dest/f'attempt-{len(actions):02d}.json',{'action_id':action,'source_state_hash':branch._engine.state_hash})
                outcome=branch.step(action);actions.append(action)
                write(dest/f'returned-{len(actions)-1:02d}.json',outcome.info)
            if i>0:commit(preps[i-1]['action_id'])
            preview=diagnostic(branch,dest)
            if i==0:
                if preview['feasible'] or preview['reason']!='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':
                    raise ValueError('Baseline fixed ray differs from saved union refusal')
                row['ray_status']='baseline_blocked'
            else:
                successor=branch.candidate_inventory();write(dest/'successor-inventory.json',successor)
                matches=[r for r in successor['emitted'] if same_ray(r,ray)]
                legal_ids=set(branch.observation().action_ids)
                legal=[r for r in matches if r['feasible'] and r['action_id'] in legal_ids]
                row['successor_exact_ray_matches']=matches
                if preview['feasible'] and legal:
                    # Existing provider dedup must expose one authoritative action.
                    if len(legal)!=1:raise ValueError('Ambiguous physical successor action')
                    commit(legal[0]['action_id']);row['ray_status']='executed'
                elif preview['feasible']:row['ray_status']='proposal_unavailable'
                else:
                    if legal:raise ValueError('Task and fresh native preview disagree on feasibility')
                    row['ray_status']='blocked'
            commit('STOP')
            metrics=branch.metrics();history=metrics['history']
            if not branch.terminated:raise ValueError('Branch did not complete')
            plan={'context_hash':context.fingerprint,'source_hash':source.source_hash,'decision_model_hash':branch.decision_model_hash,
                'initial_observation_hash':initial_hash,'max_steps':24,'actions':actions,'history':history,
                'terminal_reason':'STOP','parameter_hash':None,'architecture_hash':None,'learning_updates':0}
            seal=semantic(plan);write(dest/'plan.json',{'plan':plan,'plan_seal':seal})
            counters['phase']='replay';replay=base.clone();context.require_task(replay)
            for step,action in enumerate(actions):
                guard();write(dest/f'replay-attempt-{step:02d}.json',{'action_id':action});replay.step(action)
            replayed=replay.metrics()
            if not replay.terminated or canonical(replayed['history'])!=canonical(history):
                raise ValueError('Sealed native history differs from complete replay')
            counters['phase']='audit'
            audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(guard() or False))
            write(dest/'replay.json',{'metrics':replayed,'independent_geometry':audit})
            if audit['accepted'] is not True:raise ValueError('Independent geometry refused replay')
            row.update(status='complete',actions=actions,plan_seal=seal,outcomes=audit['outcomes'],
                target_access_success=audit['target_access_success'],
                fixed_ray_feasible=preview['feasible'],fixed_ray_reason=preview['reason'],
                native_previews=budget.snapshot()['native_preview_entries']-before_previews,
                native_counts={k:counters['native_counts'][k]-before_counts[k] for k in before_counts})
            write(dest/'branch-result.json',row)
            if base._engine.state_hash!=initial_state or base.metrics()['history']:
                raise ValueError('Branch or replay changed shared baseline state')
        for i in range(16):
            try:run_branch(i)
            except BaseException as error:
                result['branches'][i].update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
                write(output/f'branch-{i:02d}'/'terminal.json',result['branches'][i]);raise
        rewards=[row['outcomes']['total_reward'] for row in result['branches']]
        selected=max(range(16),key=lambda i:rewards[i])
        result.update(selected_ordinal=selected,selected_plan_seal=result['branches'][selected]['plan_seal'],
            selected_return=rewards[selected],fixed_ray=ray,full_target_mm3=old['common_public_world']['full_target_mm3'])

    original_write=factory.write;factory.write=sink.write
    try:
        with budget,execution_counters(guard) as counters:
            run_world();gc.collect()
            result['source_released']=life[0] is not None and life[0]() is None
            if not result['source_released']:raise RuntimeError('Source retained after completed witness')
            guard();budget.complete(history_complete=True)
        result['status']='complete_fixed_ray_witness'
    except BaseException as error:
        result.update(status='failed_or_unresolved',attempt_failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        factory.write=original_write
        for row in result['branches']:
            if row['status']=='not_run':row['not_run_reason']='prior failure; no retry or substituted branch'
        counts={'rollout':0,'replay':0} if counters is None else counters['native_counts']
        prohibited={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'search_calls':0} if counters is None else counters['prohibited_calls']
        result.update(native_counts=counts,**prohibited,complete_wall_seconds=time.perf_counter()-started)
        try:write(output/'costs.json',{'native_counts':counts,'prohibited_calls':prohibited,
            'diagnostic_previews':0 if counters is None else counters['diagnostic_previews'],
            'native_budget':budget.snapshot(),'complete_wall_seconds':result['complete_wall_seconds'],
            'source_visits_attempted':1,'count_scope':'every native preview and attempted committed/replay transition; diagnostic rejection is not committed'})
        except BaseException as error:result.update(status='failed_or_unresolved',cost_failure=str(error))
        sink.write(output/'result.json',result,terminal=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    release=small(args.release);source_guard(release,args.release,args.release_sha256)
    output=ROOT/OUTPUT;supervision=output.with_name(output.name+'.supervision')
    if output.exists() or output.is_symlink() or not supervision.is_dir():raise ValueError('Fresh owned reservation required')
    started=time.perf_counter();summary={'status':'started','SELECT_EVAL_opened':False}
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    def progress(phase,**details):
        p=supervision/'worker-progress.tmp';p.write_text(json.dumps({'phase':phase,'seconds':time.perf_counter()-started,**details})+'\n');p.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Fixed-ray witness worker deadline')
    prior=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        summary['runtime']={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'torch':torch.__version__,'threads':1}
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Fixed-ray witness incomplete')
        control=endpoint_control(output,release);write(supervision/'endpoint-control.json',control)
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_union025_fixed_ray',result_sha256=sha(output/'result.json'),
            endpoint_control_sha256=sha(supervision/'endpoint-control.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        write(supervision/'worker-final.json',summary)

if __name__=='__main__':main()
