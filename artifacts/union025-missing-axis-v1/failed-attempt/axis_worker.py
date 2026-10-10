"""One missing public axis diagnostic; existing config/task/geometry, no learning."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
from pathlib import Path
import signal
import sys
import time
import weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from axis_contract import (OUTPUT,SUBJECT,CONDITION,EXTRA_AXIS,CLEARANCE_VOXEL,CLEARANCE_TOOL,
    WORKER_SECONDS,OUTPUT_BYTES,inputs,canonical,semantic,sha,small,physical,same_ray,without_ids,
    source_guard,complete_result,endpoint_control)

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
        if sum(counts.values())>=12 or counts[phase]>=6:raise InterruptedError('Native transition cap')
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
    from resectionlab.native_proposals import NominalCavityProposalConfig,DEFAULT_COLUMN_OFFSETS
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab import public_patient_factory as factory
    records,refs,target_ray=inputs();baseline=records['learning_release'];old=records['union_world']
    kwargs=dict(baseline['learning_protocol']['cohort_execution']['proposal_config'])
    offsets=tuple(tuple(r) for r in kwargs['offsets_source_voxels'])
    if offsets!=DEFAULT_COLUMN_OFFSETS or len(offsets)!=13 or kwargs['max_candidates']!=120:
        raise ValueError('Exact original13-axis config required')
    kwargs['offsets_source_voxels']=offsets+(EXTRA_AXIS,)
    config=NominalCavityProposalConfig(**kwargs)
    target_variant=baseline['learning_protocol']['public_target_context_variant']
    cohort_bytes=(ROOT/refs['cohort']['path']).read_bytes()
    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=1000,seconds=WORKER_SECONDS)
    started=time.perf_counter();counters=None;life=[None]
    result={'status':'started','subject':SUBJECT,'condition':CONDITION,'source_released':False,'histories':[],
        'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'private_reference_reads':0,'search_calls':0,
        'SELECT_EVAL_opened':False,'post_hoc_TRAIN_diagnosis':True,
        'interpretation':'One appended source-normal axis. This is a changed proposal model over the same unvalidated public S union T occupancy, not a transfer result.'}
    def guard():budget.check();sink.check()
    def write(path,value):sink.write(path,thaw_json(freeze_json(value)))

    def run_world():
        admission=release['condition_admission'];srcdir=output/'source';srcdir.mkdir();progress('one_public_source')
        source,binding,qc,protocol=factory.prepare_public_source(srcdir,{**release,'limits':admission['limits']},release_sha,None,progress,
            public_manifest_path=ROOT/refs['public_manifest']['path'],public_manifest_sha256=refs['public_manifest']['sha256'],
            cohort_bytes=cohort_bytes,learning_protocol_hash=admission['learning_protocol_hash'],proposal_config=config,
            public_target_context_variant=target_variant,occupancy_condition=CONDITION)
        life[0]=weakref.ref(source)
        base,context=make_patient_planning_task(source,cohort_bytes=cohort_bytes,source_binding=binding,qc_receipt=qc,protocol=protocol)
        context.require_task(base);obs=base.observation();inv=base.candidate_inventory();initial_state=base._engine.state_hash
        public=old['common_public_world']
        access={'center_mm':source.access.center_mm.tolist(),'normal_inward':source.access.normal_inward.tolist(),
            'radius_mm':source.access.radius_mm,'window_id':source.access.window_id}
        checks={'raw_image_hash':source._normalization_record['source_image_hash'],
            'original_frame_hash':array_digest(source.affine_ras_mm),'native_frame_hash':array_digest(source._native_affine_ras_mm),
            'target_hash':array_digest(source.nominal_target),'domain_hash':source.support_provenance['target_domain_hash'],
            'access':access,'tools':[asdict(t) for t in source.tools],'reward':asdict(base.reward_spec),
            'full_target_mm3':source._supplied_goal_extent['full_region_membership_mm3']}
        if any(canonical(v)!=canonical(public[k]) for k,v in checks.items()):raise ValueError('Underlying public world changed')
        if (canonical(thaw_json(source._occupancy_derivation))!=canonical(old['derived_occupancy'])
                or canonical(thaw_json(source._normalization_record))!=canonical(old['normalization'])
                or canonical(thaw_json(source._supplied_goal_extent))!=canonical(old['supplied_goal_extent'])
                or not np.array_equal(source.reference_target,source.nominal_target)):
            raise ValueError('Material/normalization/full target boundary changed')
        world={'source_hash':source.source_hash,'decision_model_hash':base.decision_model_hash,
            'initial_observation_hash':obs.fingerprint,'context_hash':context.fingerprint,'initial_state_hash':initial_state,
            'unchanged_public_checks':checks,'baseline_world_sha256':refs['union_world']['sha256'],
            'proposal_config':asdict(config),'proposal_rule_hash':config.fingerprint,'baseline_rule_hash':public['proposal_rule_hash'],
            'explicit_new_decision_model':True,'derived_occupancy':thaw_json(source._occupancy_derivation),
            'supplied_goal_extent':thaw_json(source._supplied_goal_extent),'normalization':thaw_json(source._normalization_record)}
        write(output/'world.json',world);write(output/'context.json',context.record());write(output/'initial-inventory.json',inv)
        if any(world[k]==old[k] for k in ('source_hash','decision_model_hash','initial_observation_hash')):
            raise ValueError('Changed config did not change source/model/observation identity')
        old_rows=[r for r in inv['ledger'] if r['column_index']<13]
        if canonical(without_ids(old_rows))!=canonical(without_ids(records['union_inventory']['ledger'])):
            raise ValueError('Axis addition changed original physical rows/dispositions; cap confound is unresolved')
        if base.max_steps!=24 or len(config.offsets_source_voxels)!=14:raise ValueError('Unexpected task change')
        # Exact same physical codec as the existing proposal provider; no rounded RAS literals.
        tip=source._native_affine_ras_mm[:3,:3]@CLEARANCE_VOXEL+source._native_affine_ras_mm[:3,3]
        depth=float((tip-source.access.center_mm)@source.access.normal_inward)
        entry=tip-depth*source.access.normal_inward
        clearance_ray={'tool_id':CLEARANCE_TOOL,'entry_mm':entry.tolist(),'tip_mm':tip.tolist()}
        write(output/'prospective-rays.json',{'clearance':clearance_ray,'target':target_ray,
            'clearance_voxel':CLEARANCE_VOXEL,'appended_axis':EXTRA_AXIS,'formula':release['clearance_ray']['formula']})
        diagnostics={}

        def diagnostic(task,dest,ray):
            guard();context.require_task(task)
            if counters['diagnostic_previews']>=2:raise InterruptedError('Fixed-ray diagnostic cap')
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

        def examine(task,name,ray):
            dest=output/name;dest.mkdir();record=diagnostic(task,dest,ray)
            inventory=task.candidate_inventory();write(dest/'inventory.json',inventory)
            matches=[r for r in inventory['emitted'] if same_ray(r,ray)]
            ids=set(task.observation().action_ids);legal=[r for r in matches if r['feasible'] and r['action_id'] in ids]
            if not record['feasible'] and legal:raise ValueError('Task and fresh native preview disagree')
            if len(legal)>1:raise ValueError('Ambiguous physical action')
            record.update(action_status='executable' if record['feasible'] and legal else
                ('proposal_unavailable' if record['feasible'] else 'blocked'),
                selected_action_id=legal[0]['action_id'] if record['feasible'] and legal else None,
                exact_inventory_matches=matches,inventory_omitted_count=inventory['omitted_count'])
            diagnostics[name]=record;write(output/'diagnostics.json',diagnostics)
            return record

        def history(actions,*,before_stop=None):
            ordinal=len(result['histories']);dest=output/f'history-{ordinal:02d}';dest.mkdir()
            task=base.clone();context.require_task(task);counters['phase']='rollout'
            for i,action in enumerate(actions):
                if action=='STOP' and before_stop is not None:before_stop(task)
                counters['phase']='rollout';guard();write(dest/f'attempt-{i:02d}.json',{'action_id':action})
                out=task.step(action);write(dest/f'returned-{i:02d}.json',out.info)
            metrics=task.metrics()
            if not task.terminated:raise ValueError('Incomplete committed history')
            plan={'source_hash':source.source_hash,'decision_model_hash':task.decision_model_hash,
                'context_hash':context.fingerprint,'initial_observation_hash':obs.fingerprint,'max_steps':24,
                'actions':actions,'history':metrics['history'],'terminal_reason':'STOP',
                'parameter_hash':None,'architecture_hash':None,'learning_updates':0}
            seal=semantic(plan);write(dest/'plan.json',{'plan':plan,'plan_seal':seal})
            replay=base.clone();context.require_task(replay);counters['phase']='replay'
            for i,action in enumerate(actions):
                guard();write(dest/f'replay-attempt-{i:02d}.json',{'action_id':action});replay.step(action)
            measured=replay.metrics()
            if not replay.terminated or canonical(measured['history'])!=canonical(metrics['history']):raise ValueError('Replay differs')
            counters['phase']='audit';audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(guard() or False))
            write(dest/'replay.json',{'metrics':measured,'independent_geometry':audit})
            if not audit['accepted']:raise ValueError('Independent geometry refused history')
            result['histories'].append({'ordinal':ordinal,'status':'complete','actions':actions,
                'plan_seal':seal,'outcomes':audit['outcomes'],'target_access_success':audit['target_access_success']})
            if base._engine.state_hash!=initial_state or base.metrics()['history']:raise ValueError('Baseline clone isolation failed')
        first=examine(base,'clearance',clearance_ray)
        history(['STOP'])
        if first['action_status']=='executable':
            prep=first['selected_action_id']
            history([prep,'STOP'],before_stop=lambda task:examine(task,'target',target_ray))
            if diagnostics['target']['action_status']=='executable':
                history([prep,diagnostics['target']['selected_action_id'],'STOP'])
        result['clearance_status']=first['action_status']
        result['target_status']=diagnostics.get('target',{}).get('action_status','not_attempted_clearance_unavailable')
        selected=max(range(len(result['histories'])),key=lambda i:result['histories'][i]['outcomes']['total_reward'])
        result.update(selected_ordinal=selected,selected_plan_seal=result['histories'][selected]['plan_seal'],
            selected_return=result['histories'][selected]['outcomes']['total_reward'],
            full_target_mm3=public['full_target_mm3'],initial_omissions=inv['omitted_count'])

    original_write=factory.write;factory.write=sink.write
    try:
        with budget,execution_counters(guard) as counters:
            run_world();gc.collect();result['source_released']=life[0] is not None and life[0]() is None
            if not result['source_released']:raise RuntimeError('Source retained')
            guard();budget.complete(history_complete=True)
        result['status']='complete_missing_axis_diagnostic'
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        factory.write=original_write
        counts={'rollout':0,'replay':0} if counters is None else counters['native_counts']
        forbidden={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'search_calls':0} if counters is None else counters['prohibited_calls']
        result.update(native_counts=counts,**forbidden,complete_wall_seconds=time.perf_counter()-started)
        try:write(output/'costs.json',{'native_counts':counts,'prohibited_calls':forbidden,
            'diagnostic_previews':0 if counters is None else counters['diagnostic_previews'],
            'native_budget':budget.snapshot(),'complete_wall_seconds':result['complete_wall_seconds'],'source_visits_attempted':1})
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
        summary.update(status='complete_owned_union025_missing_axis',result_sha256=sha(output/'result.json'),
            endpoint_control_sha256=sha(supervision/'endpoint-control.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        write(supervision/'worker-final.json',summary)

if __name__=='__main__':main()

