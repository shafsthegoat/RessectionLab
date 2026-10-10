"""Owned single-arm TRAIN opening-proposal search; no learning or checkpoint path."""
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
from pilot_contract import (OUTPUT,TRAIN,CONDITIONS,OCCUPANCY,ARMS,EXECUTION,WORKER_SECONDS,OUTPUT_BYTES,IMPLEMENTATION,
    inputs,canonical,semantic,sha,source_guard,complete_result,endpoint_control,without_ids,inventory_summary)


@contextmanager
def execution_counters(guard):
    """Count existing transitions and prohibit model work; no transition changes."""
    import torch
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.spatial_policy import SpatialPolicy
    counters={'phase':'setup','native_counts':{'search':0,'rollout':0,'replay':0},
        'prohibited_calls':{'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0}}
    original=NativeSpatialTask._transition
    def counted(task,*args,**kwargs):
        guard();phase=counters['phase'];counts=counters['native_counts']
        if phase not in counts:raise ValueError('Native transition outside declared phase')
        if (sum(counts.values())>=72 or (phase=='search' and counts['search']>=24)
                or (phase!='search' and counts['rollout']+counts['replay']>=48)):
            raise InterruptedError('Aggregate native transition cap')
        counts[phase]+=1
        return original(task,*args,**kwargs)
    saved=[]
    def prohibit(owner,name,key):
        old=getattr(owner,name);saved.append((owner,name,old))
        def refused(*args,**kwargs):
            counters['prohibited_calls'][key]+=1
            raise RuntimeError('Prohibited model work: '+key)
        setattr(owner,name,refused)
    NativeSpatialTask._transition=counted
    prohibit(SpatialPolicy,'forward','policy_forwards')
    prohibit(torch.optim.Adam,'step','optimizer_updates')
    prohibit(torch,'load','checkpoint_loads')
    try:yield counters
    finally:
        NativeSpatialTask._transition=original
        for owner,name,old in reversed(saved):setattr(owner,name,old)


def execute(release,release_sha,output,progress):
    import numpy as np
    from resectionlab.core import thaw_json,freeze_json,array_digest
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.observed_search import ObservedSearchLimit
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.planning_budget import PlanningBudget,PlanningBudgetViolation
    from resectionlab import public_patient_factory as factory
    records,refs=inputs();baseline=records['release'];cohort_bytes=(ROOT/refs['cohort']['path']).read_bytes()
    selected={r['patient_id']:r for r in records['public_index']['cases']}
    target_variant=baseline['learning_protocol']['public_target_context_variant']
    output.mkdir(exist_ok=False);sink=OutputBudget(output,OUTPUT_BYTES)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=10000,seconds=WORKER_SECONDS)
    started=time.perf_counter();pairs={};counters=None;copy_counts={'status':'not_started'}
    result={'status':'started','implementation':IMPLEMENTATION,'arms':[{**r,'status':'not_run','source_released':True} for r in ARMS],
        'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'private_reference_reads':0,
        'SELECT_EVAL_opened':False,'global_resource_failure':None,
        'interpretation':'Same frozen TRAIN025 union/obstruction world and selector; only engine lazy-copy implementation differs',
        'negative_preparatory_moves':'not explored by existing greedy selector; STOP is not an impossibility proof'}
    def guard():budget.check();sink.check()
    def run_arm(ordinal,life):
        row=result['arms'][ordinal];subject=row['subject'];condition=row['condition']
        dest=output/f'arm-{ordinal:02d}';dest.mkdir(exist_ok=False)
        def write(name,value):sink.write(dest/name,thaw_json(freeze_json(value)))
        arm_started=time.perf_counter();before=budget.snapshot()['native_preview_entries']
        start_counts=dict(counters['native_counts']);row.update(status='running',phase='source')
        config=NominalCavityProposalConfig(**baseline['learning_protocol']['cohort_execution']['proposal_config'],
            obstruction_opening=condition==CONDITIONS[1])
        admission=release['condition_admission'][condition]
        # Both condition declarations are explicitly nested in this root release.
        arm_release={**release,'limits':admission['limits']}
        srcdir=dest/'source';srcdir.mkdir()
        progress('arm_source',ordinal=ordinal,subject=subject,condition=condition)
        item=selected[subject]
        source,binding,qc,protocol=factory.prepare_public_source(srcdir,arm_release,release_sha,None,progress,
            public_manifest_path=Path(item['path']),public_manifest_sha256=item['sha256'],cohort_bytes=cohort_bytes,
            learning_protocol_hash=admission['learning_protocol_hash'],proposal_config=config,
            public_target_context_variant=target_variant,occupancy_condition=OCCUPANCY)
        life[0]=weakref.ref(source)
        task,context=make_patient_planning_task(source,cohort_bytes=cohort_bytes,source_binding=binding,qc_receipt=qc,protocol=protocol)
        context.require_task(task)
        observation=task.observation();initial_hash=observation.fingerprint
        old=records['expected_world']
        if any(value!=old[key] for key,value in (('source_hash',source.source_hash),('decision_model_hash',task.decision_model_hash),('initial_observation_hash',initial_hash))):
            raise ValueError('Historical enabled TRAIN025 world differs')
        derivation=thaw_json(source._occupancy_derivation);extent=thaw_json(source._supplied_goal_extent)
        raw_count=derivation['source_support_positive_voxels'];outside=derivation['added_region_positive_voxels']
        full=extent['full_region_positive_voxels']
        common={'source_files':{k:v['sha256'] for k,v in selected_manifest(item)['input_files'].items()},
            'raw_image_hash':source._normalization_record['source_image_hash'],
            'original_frame_hash':array_digest(source.affine_ras_mm),'native_frame_hash':array_digest(source._native_affine_ras_mm),
            'target_hash':array_digest(source.nominal_target),'domain_hash':source.support_provenance['target_domain_hash'],
            'access':{'center_mm':source.access.center_mm.tolist(),'normal_inward':source.access.normal_inward.tolist(),
                'radius_mm':source.access.radius_mm,'window_id':source.access.window_id},
            'tools':[asdict(t) for t in source.tools],'reward':asdict(task.reward_spec),
            'full_target_voxels':full,'full_target_mm3':extent['full_region_membership_mm3'],
            'S_voxels':raw_count,'T_minus_S_voxels':outside,'S_intersect_T_voxels':full-outside,
            'raw_occupancy_target_fraction_upper_bound':(full-outside)/full,
            'occupancy_hash':array_digest(source.observed_support),'max_steps':24}
        common=thaw_json(freeze_json(common))
        norm=thaw_json(source._normalization_record)
        row.update(source_hash=source.source_hash,decision_model_hash=task.decision_model_hash,
            initial_observation_hash=initial_hash,context_hash=context.fingerprint,
            historical_enabled_union_identity_verified=True,common_public_world=common,
            normalization=norm,derived_occupancy=derivation,supplied_goal_extent=extent,
            proposal_config=thaw_json(freeze_json(config.to_record())),proposal_rule_hash=config.fingerprint,
            actual_model_work_permitted=False)
        initial_inventory=task.candidate_inventory()
        if canonical(initial_inventory)!=canonical(records['expected_initial-inventory']):
            raise ValueError('Historical exact enabled inventory changed')
        row['pairing']={'historical_world_and_initial_inventory_equal':True}
        write('world.json',row);write('context.json',context.record());write('initial-inventory.json',initial_inventory)
        row['initial_inventory_summary']=inventory_summary(initial_inventory)
        # This source is public supplied annotation T in both evaluator slots;
        # no richer/private ventricular or protected data is loaded or consulted.
        if not np.array_equal(source.reference_target,source.nominal_target):raise ValueError('Only public target may score this diagnostic')
        row['phase']='search';counters['phase']='search';progress('arm_greedy_search',ordinal=ordinal)
        inventory_records=[];seen_states=set();capturing=[False]
        from resectionlab.native_spatial_task import NativeSpatialTask
        original_prepare=NativeSpatialTask._prepare_inventory
        def capture_prepare(active):
            prepared=original_prepare(active)
            state=active._engine.state_hash
            if (not capturing[0] and not active.terminated and state not in seen_states):
                if (active._source_hash!=source.source_hash or active.decision_model_hash!=task.decision_model_hash
                        or active._config.fingerprint!=source._native_config.fingerprint
                        or any(getattr(active.case,k) is not getattr(source,k) for k in
                            ('structural_intensity','observed_support','nominal_target','affine_ras_mm','public_target_domain'))
                        or not active._planning or active.case.reference_target is not active.case.nominal_target):
                    raise ValueError('Foreign or nonpublic search model')
                capturing[0]=True
                try:
                    preview_before=budget.snapshot()['native_preview_entries']
                    current=active.candidate_inventory()
                    if budget.snapshot()['native_preview_entries']!=preview_before:
                        raise ValueError('Inventory capture performed an additional native preview')
                    name=f'search-inventory-{len(inventory_records):02d}.json'
                    write(name,current);seen_states.add(state)
                    inventory_records.append({'file':name,'sha256':sha(dest/name),'summary':inventory_summary(current)})
                finally:capturing[0]=False
            return prepared
        NativeSpatialTask._prepare_inventory=capture_prepare
        try:actions,accounting=task.observed_greedy_search(seconds=EXECUTION['search_seconds_per_arm'])
        except ObservedSearchLimit as error:
            write('search.json',{'actions':list(error.best_sequence),'accounting':error.accounting,'status':'capped_unresolved'})
            raise
        finally:
            NativeSpatialTask._prepare_inventory=original_prepare
            write('search-inventories.json',inventory_records)
        write('search.json',{'actions':list(actions),'accounting':accounting,'status':'complete'})
        if not accounting['complete']:raise ValueError('Incomplete greedy result cannot become a STOP plan')
        decision_summaries=[]
        for decision in accounting['decisions']:
            saved=next(r for r in inventory_records if r['summary']['cavity_state_hash']==decision['source_state_hash'])
            inv=json.loads((dest/saved['file']).read_text());by_action={r['action_id']:r for r in inv['ledger'] if r['feasible']}
            added=[r for r in decision['scores'] if r['action_id']!='STOP' and by_action[r['action_id']]['family']=='obstruction_opening']
            selected_action=decision['selected_action_id']
            decision_summaries.append({'step':decision['step'],'state_hash':decision['source_state_hash'],
                'selected_action_id':selected_action,'selected_family':'STOP' if selected_action=='STOP' else by_action[selected_action]['family'],
                'selected_score':next(r for r in decision['scores'] if r['action_id']==selected_action),
                'added_legal_count':len(added),'added_target_removing_count':sum(r['target_removed_mm3']>0 for r in added),
                'added_nonT_removing_count':sum(r['normal_removed_mm3']>0 for r in added),
                'added_positive_reward_count':sum(r['reward']>0 for r in added),'added_scores':added,
                'inventory_file':saved['file'],'inventory_sha256':saved['sha256'],
                'omitted_count':inv['omitted_count'],'candidate_cap_omitted_count':inv.get('candidate_cap_omitted_count',inv['omitted_count']),
                'obstruction_selection_omitted_count':inv.get('obstruction_selection_omitted_count',0),
                'obstruction_accounting':inv.get('obstruction_accounting')})
        write('decision-summary.json',decision_summaries)
        row['decision_summary']=[{k:v for k,v in d.items() if k not in ('added_scores','obstruction_accounting')} for d in decision_summaries]
        if task.observation().fingerprint!=initial_hash:raise ValueError('Search changed authoritative initial task')
        row['phase']='rollout';counters['phase']='rollout';context.require_task(task)
        for step,action in enumerate(actions):
            guard();write(f'rollout-attempt-{step:02d}.json',{'action_id':action})
            outcome=task.step(action);write(f'rollout-returned-{step:02d}.json',outcome.info)
        if not task.terminated:raise ValueError('Returned sequence is not complete')
        metrics=task.metrics();history=metrics['history']
        plan={'context_hash':context.fingerprint,'source_hash':source.source_hash,'decision_model_hash':task.decision_model_hash,
            'initial_observation_hash':initial_hash,'max_steps':24,'actions':list(actions),'history':history,
            'terminal_reason':'STOP' if actions[-1]=='STOP' else 'HORIZON','parameter_hash':None,'architecture_hash':None,'learning_updates':0}
        plan_seal=semantic(plan);write('plan.json',{'plan':plan,'plan_seal':plan_seal})
        row['phase']='replay';counters['phase']='replay';replay=task.fresh();context.require_task(replay)
        for step,action in enumerate(actions):
            guard();write(f'replay-attempt-{step:02d}.json',{'action_id':action})
            replay.step(action)
        replayed=replay.metrics()
        if not replay.terminated or canonical(replayed['history'])!=canonical(history):raise ValueError('Complete native replay differs from sealed history')
        row['phase']='independent_geometry';counters['phase']='audit'
        audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(guard() or False))
        write('replay.json',{'metrics':replayed,'independent_geometry':audit})
        if audit['accepted'] is not True:raise ValueError('Independent geometry refused completed replay')
        removed=[cell for record in history for cell in record.get('removed_indices_native',[])]
        depth=None
        if removed:
            pts=np.asarray(removed)@source._native_affine_ras_mm[:3,:3].T+source._native_affine_ras_mm[:3,3]
            depth=float(np.max((pts-source.access.center_mm)@source.access.normal_inward))
        row.update(status='complete',phase='complete',actions=list(actions),plan_seal=plan_seal,
            final_native_state={'state_hash':replay._engine.state_hash,'mask_digests':replay._engine.committed_mask_digests()},
            outcomes=audit['outcomes'],target_access_success=audit['target_access_success'],
            max_removed_cell_depth_mm=depth,search_accounting={k:v for k,v in accounting.items() if k!='decisions'},
            stop_only=list(actions)==['STOP'],independent_geometry_accepted=True,
            arm_wall_seconds=time.perf_counter()-arm_started,
            native_previews=budget.snapshot()['native_preview_entries']-before,
            native_counts={k:counters['native_counts'][k]-start_counts[k] for k in start_counts})
        write('arm-result.json',row)
    def selected_manifest(item):
        # The source loader already verifies these same bounded public metadata.
        p=Path(item['path'])
        if sha(p)!=item['sha256']:raise ValueError('Public manifest changed after load')
        return json.loads(p.read_text())
    original_write=factory.write;factory.write=sink.write
    try:
        with copy_measurement() as copy_counts, budget,execution_counters(guard) as counters:
            for ordinal,row in enumerate(result['arms']):
                life=[None];phase_started=time.perf_counter();before=budget.snapshot()['native_preview_entries']
                before_counts=dict(counters['native_counts'])
                try:run_arm(ordinal,life)
                except BaseException as error:
                    row.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)},
                        arm_wall_seconds=time.perf_counter()-phase_started,
                        native_previews=budget.snapshot()['native_preview_entries']-before,
                        native_counts={k:counters['native_counts'][k]-before_counts[k] for k in before_counts})
                    if isinstance(error,(PlanningBudgetViolation,InterruptedError,TimeoutError,MemoryError,KeyboardInterrupt,SystemExit)):
                        result['global_resource_failure']=dict(row['failure'])
                gc.collect();row['source_released']=life[0] is None or life[0]() is None
                if not row['source_released']:
                    result['global_resource_failure']={'type':'SourceLifetimeFailure','message':'Source retained after arm'}
                sink.write(output/f'arm-{ordinal:02d}'/'terminal.json',row)
                if result['global_resource_failure'] is not None:break
            if result['global_resource_failure'] is None:
                if any(r['status']!='complete' for r in result['arms']):
                    raise RuntimeError('Unresolved arm: the fixed planned rows retained; paired attempt is not complete')
                guard();budget.complete(history_complete=True)
        result['status']='complete_paired_TRAIN_search'
        if result['global_resource_failure'] is not None:result['status']='failed_or_unresolved'
    except BaseException as error:
        result.update(status='failed_or_unresolved',attempt_failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        factory.write=original_write
        for row in result['arms']:
            if row['status']=='not_run':row['not_run_reason']='earlier global refusal; no replacement or retry'
        counts={'search':0,'rollout':0,'replay':0} if counters is None else counters['native_counts']
        prohibited={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0} if counters is None else counters['prohibited_calls']
        result.update(native_counts=counts,**prohibited,complete_wall_seconds=time.perf_counter()-started)
        try:sink.write(output/'costs.json',{'native_counts':counts,'prohibited_calls':prohibited,
            'native_budget':budget.snapshot(),'complete_wall_seconds':result['complete_wall_seconds'],
            'count_scope':'attempted existing native transitions including failed calls; search,rollout,replay separate',
            'temporary_mask_copies':copy_counts,'source_visits_attempted':sum(r['status']!='not_run' for r in result['arms'])})
        except BaseException as error:result.update(status='failed_or_unresolved',cost_failure=str(error))
        sink.write(output/'result.json',result,terminal=True)
    return result


def install_engine(release):
    """Fresh-process whole-module overlay before any native/scientific task import."""
    import importlib.util
    import resectionlab
    name='resectionlab.native_resection'
    if name in sys.modules:raise RuntimeError('Engine imported before declared implementation')
    entry=release['engine_sources'][IMPLEMENTATION];path=ROOT/entry['path']
    if sha(path)!=entry['sha256']:raise ValueError('Engine implementation changed')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);setattr(resectionlab,'native_resection',module)

@contextmanager
def copy_measurement():
    """Count only copies of the two committed masks while an existing preview runs."""
    import numpy as np
    from resectionlab.geometry import _ImmutableArray
    from resectionlab.native_resection import NativeResectionEngine
    preview=NativeResectionEngine.preview_stroke;copy=_ImmutableArray.copy
    counts={'preview_calls':0,'copy_calls':0,'copied_bytes':0,'by_mask':{},
        'scope':'committed remaining/connected-free copy calls inside preview only; identical wrapper both arms; not total allocator traffic'}
    watched=[]
    def measured_copy(value,*args,**kwargs):
        kind=next((ids[id(value)] for ids in reversed(watched) if id(value) in ids),None)
        if kind is not None:
            counts['copy_calls']+=1;counts['copied_bytes']+=value.nbytes
            row=counts['by_mask'].setdefault(kind,{'calls':0,'bytes':0});row['calls']+=1;row['bytes']+=value.nbytes
        return copy(value,*args,**kwargs)
    def measured_preview(engine,*args,**kwargs):
        counts['preview_calls']+=1
        watched.append({id(engine.remaining_mask):'remaining',id(engine.connected_free_mask):'connected_free'})
        try:return preview(engine,*args,**kwargs)
        finally:watched.pop()
    _ImmutableArray.copy=measured_copy;NativeResectionEngine.preview_stroke=measured_preview
    try:yield counts
    finally:_ImmutableArray.copy=copy;NativeResectionEngine.preview_stroke=preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    release=json.loads(args.release.read_text());source_guard(release,args.release,args.release_sha256)
    output=ROOT/OUTPUT;supervision=output.with_name(output.name+'.supervision')
    if output.exists() or output.is_symlink() or not supervision.is_dir():raise ValueError('Fresh owned reservation required')
    started=time.perf_counter();summary={'status':'started','SELECT_EVAL_opened':False}
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    def progress(phase,**details):
        p=supervision/'worker-progress.tmp';p.write_text(json.dumps({'phase':phase,'seconds':time.perf_counter()-started,**details})+'\n');p.replace(supervision/'worker-progress.json')
    def deadline(*unused):raise TimeoutError('Paired obstruction worker deadline')
    prior=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    try:
        install_engine(release)
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        summary['runtime']={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'torch':torch.__version__,'threads':1}
        result=execute(release,args.release_sha256,output,progress)
        if not complete_result(result):raise ValueError('Single-arm diagnostic incomplete')
        control=endpoint_control(output,release);write(supervision/'endpoint-control.json',control)
        source_guard(release,args.release,args.release_sha256)
        summary.update(status='complete_owned_paired_TRAIN_obstruction',result_sha256=sha(output/'result.json'),
            endpoint_control_sha256=sha(supervision/'endpoint-control.json'))
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
        summary['wall_seconds']=time.perf_counter()-started
        summary['canonical_result_sha256']=sha(output/'result.json') if (output/'result.json').is_file() else None
        write(supervision/'worker-final.json',summary)

if __name__=='__main__':main()
