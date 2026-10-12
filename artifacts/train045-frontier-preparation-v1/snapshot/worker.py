"""One fixed TRAIN045 frontier action-family contrast; root-owned execution only."""
import argparse
from contextlib import contextmanager
import json,os,resource,signal,sys,threading,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT,IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,canonical,semantic,read_bytes,read_json,sha,guard,tree_bytes,write_json,runtime_roots,PAIR_SUBJECT
from inputs import REPORT,REPORT_SHA,PLANS,PLANS_SHA,WORLD_KEYS,retained_prefix

def install_boundary(index, output):
    allowed={str((ROOT/name).absolute()) for name in
             (*index['files'],*index['input_metadata'],*index['public_arrays'])}
    allowed.update(str((ROOT/r['path']).absolute()) for r in index.get('checkpoints',{}).values())
    if index['runtime_roots']!=runtime_roots():raise ValueError('Frozen dependency roots differ from executing interpreter')
    dependencies=tuple(Path(p) for p in index['runtime_roots'])
    def boundary(event, values):
        if event.startswith(('socket.','subprocess.')) or event in ('os.fork','os.forkpty','os.posix_spawn','os.system'):
            raise PermissionError('No network/process or acquisition in scientific worker')
        if event=='open' and isinstance(values[0],(str,bytes)):
            path=Path(os.path.abspath(os.fsdecode(values[0])))
            mode,flags=values[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if any(path.is_relative_to(p) for p in dependencies):
                if writing:raise PermissionError('Runtime dependencies are read-only')
                resolved=path.resolve()
                if not any(resolved.is_relative_to(p) for p in dependencies):raise PermissionError('Runtime symlink escapes dependency root')
                return
            if path.is_relative_to(ROOT) and not path.is_relative_to(output) and writing:
                raise PermissionError('Experiment inputs/source are read-only')
            if path.is_relative_to(ROOT) and not path.is_relative_to(output) and str(path) not in allowed:
                # Absent, isolated bytecode probes are not source admission.
                if path.is_relative_to(output/'absent-worker-cache'): return
                raise PermissionError('Undeclared repository input: '+str(path))
    sys.addaudithook(boundary)


def import_execution_modules():
    import numpy
    import torch
    from resectionlab import (native_resection,patient_planning_cohort_visits,
        patient_planning_cohort_sequential,patient_planning_cohort_io,
        patient_planning_preflight,native_spatial_evaluation,planning_budget)
    return {'numpy':numpy.__version__,'torch':torch.__version__}




def execute(index,index_sha,config,output,started):
    from resectionlab.core import thaw_json,semantic_digest
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_proposals import OBSTRUCTION_CELLS_PER_PREVIEW
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.observed_search import ObservedSearchLimit
    from frontier_selector import select_suffix
    from frontier_protocol import (enabled_task,map_prefix,replay_focused_tree,compact_history,physical,exact,useful_gate)
    caps=config['caps'];end=started+caps['work_seconds'];sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,0);phase=['metadata'];sidecars={};sidecar_calls={};transitions={'attempted':0,'completed':0,'raised':0}
    report={'schema':config['schema'],'status':'running','config':config,'source_index_sha256':index_sha,
        'source_unchanged':False,'branches':{},'checkpoint_loads':0,'cache_loads':0,'head_optimizer_updates':0,
        'closed_roles_opened':False,'plans_sealed_before_replay':False,'baseline_reproduced':False,
        'scope':'fixed original TRAIN04512-motion prefix; newly declared preparation catalog; no learned inference'}
    def checkpoint():
        if time.perf_counter()>=end:raise TimeoutError('End-to-end fixed frontier limit')
        budget.check();sink.check()
        if tree_bytes(HERE)>caps['whole_build_bytes']-65536:raise InterruptedError('Whole stage cap')
    def cancel():checkpoint();return False
    def save_report():write_json(output/'report.json',thaw_json(report))
    def progress(label,**details):
        checkpoint();write_json(output/'progress.json',{'phase':label,'elapsed_seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'transition_counts':transitions,'unique_sidecars':len(sidecars),**details},65536)
    @contextmanager
    def at(label):
        phase[0]=label;progress(label)
        with costs.scope(label):yield
        progress(label+'.complete')
    @contextmanager
    def counters():
        preview=NativeResectionEngine.preview_stroke;transition=NativeSpatialTask._transition
        def measured(engine,*args,**kwargs):
            value=preview(engine,*args,**kwargs)
            if value.obstruction_diagnostic is not None and phase[0] in ('original_frontier_witness','focused_suffix_planning'):
                row=thaw_json(value.obstruction_diagnostic);key=row['fingerprint']
                if semantic({k:v for k,v in row.items() if k!='fingerprint'})!=key:
                    raise ValueError('Native first-failure sidecar digest differs')
                if key not in sidecars and len(sidecars)>=caps['unique_sidecars']:
                    raise InterruptedError('Declared unique first-failure sidecar cap')
                if key in sidecars:exact(sidecars[key],row,'repeated first-failure sidecar')
                sidecars[key]=row
                item=sidecar_calls.setdefault(phase[0],{});item[key]=item.get(key,0)+1
            return value
        def committed(task,*args,**kwargs):
            if transitions['attempted']>=caps['native_transitions']:raise InterruptedError('Declared actual transition cap')
            transitions['attempted']+=1
            try:
                value=transition(task,*args,**kwargs);transitions['completed']+=1;return value
            except BaseException:
                transitions['raised']+=1;raise
        NativeResectionEngine.preview_stroke=measured;NativeSpatialTask._transition=committed
        try:yield
        finally:NativeResectionEngine.preview_stroke=preview;NativeSpatialTask._transition=transition
    def pinned(path):return read_json(ROOT/path,index['input_metadata'][path])
    def endpoint(task,name,world):
        history=task.metrics()['history'];audit=evaluate_native_spatial_episode(task,cancelled=cancel)
        return {'method':name,'complete':task.terminated,'actions':[r['action_id'] for r in history],
            'original_admitted_world':world,'actual_catalog_binding':{
                'source_hash':task.case.source_hash,'decision_model_hash':task.decision_model_hash,
                'native_model_hash':task._config.fingerprint,'observation_catalog_scope':'original' if name=='ORIGINAL_STOP' else 'focused_v3_changed_catalog'},
            'source_hash':task.case.source_hash,'decision_model_hash':task.decision_model_hash,
            'native_model_hash':task._config.fingerprint,'native_state_hash':task._engine.state_hash,
            'committed_masks':dict(task._engine.committed_mask_digests()),'history_hash':semantic(history),
            'history':compact_history(history),'audit':audit}
    try:
        with budget,costs,counters(),_bounded_writes(sink):
            with at('saved_metadata_admission'):
                for name,digest in index['input_metadata'].items():read_bytes(ROOT/name,digest)
                prior=pinned(REPORT);plans=pinned(PLANS);old,oldplan=retained_prefix(prior,plans)
                report['retained_result']={'path':REPORT,'sha256':REPORT_SHA,'plans_sha256':PLANS_SHA,
                    'plan_seal':oldplan['plan_seal'],'prior_costs_not_erased':prior['pair_plus_recovery_costs']}
                original=pinned(IL+'/configuration.json');protocol=original['learning_protocol'];limits=original['admission_limits']
                world_release={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'actual_caps':caps,'root_source_index_sha256':index_sha,
                    'execution_released':True,'scope':'same_original_world_source_admission_only_no_learning'}
                sink.write(output/'world-admission.json',world_release)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=world_release,released_sha256=sha(output/'world-admission.json'),progress=progress)
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,checkpoint)
            dest=output/'evaluation';dest.mkdir()
            def compare(base,context):
                world={k:context.record()[k] for k in WORLD_KEYS};exact(world,old['world'],'original retained world')
                with at('paired_fixed_prefix'):
                    auth=base.fresh();context.require_task(auth)
                    focused,model=enabled_task(base);nominal=focused.planning_clone()
                    records,frontier,joins=map_prefix(auth,nominal,old['actions'][:-1],old['history'][:-1],checkpoint=checkpoint)
                    if len(records)!=12 or auth._steps!=12:raise ValueError('Fixed12 prefix denominator changed')
                    report['physical_model_join']=model;report['fixed_frontier']=frontier
                    sink.write(output/'fixed-prefix.json',{'joins':joins,'original_frontier_state':auth._engine.state_hash,
                        'focused_frontier':frontier,'focused_packets':records,'history_binding_scope':
                        'All physical fields and rewards exact; explicit source/model/action IDs and outcome scope differ.'})
                with at('original_frontier_witness'):
                    inventory=auth.candidate_inventory();raw=auth._proposal_batch.proposals
                    if len(raw)!=91 or len({(r.tool_id,r.entry_mm,r.tip_mm) for r in raw})!=91:
                        raise ValueError('Retained terminal91 distinct original proposals changed')
                    probe=auth._engine.clone();outcomes=[]
                    for ray in raw:
                        checkpoint();value=probe.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm,
                            obstruction_diagnostics=True,obstruction_cell_limit=OBSTRUCTION_CELLS_PER_PREVIEW)
                        outcomes.append({'action_id':ray.proposal_id,'tool_id':ray.tool_id,'entry_mm':ray.entry_mm,
                            'tip_mm':ray.tip_mm,'feasible':value.feasible,'reason':value.reason,
                            'sidecar_fingerprint':None if value.obstruction_diagnostic is None else value.obstruction_diagnostic['fingerprint']})
                    if any(v['feasible'] or v['reason']!='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' for v in outcomes):
                        raise ValueError('Original terminal frontier changed')
                    sink.write(output/'original-frontier.json',{'inventory':inventory,'outcomes':outcomes,
                        'original_native_state_hash':auth._engine.state_hash})
                    del probe
                with at('focused_suffix_planning'):
                    try:chosen,account=select_suffix(nominal,records,frontier,seconds=config['selector_seconds'],cancelled=cancel,progress=progress)
                    except ObservedSearchLimit as error:
                        partial=thaw_json(error.accounting)
                        sink.write(output/'partial-selection.json',partial)
                        report['incomplete_selection']={'path':'partial-selection.json','sha256':sha(output/'partial-selection.json')}
                        raise
                    if (not account['complete'] or account['call_cap_reached'] or account['time_cap_reached']
                        or account['model_transition_calls']>12 or len(account['explored_action_records'])>24
                        or account['selected_prefix_length']<12):raise ValueError('Complete bounded suffix required')
                    account=thaw_json(account);chosen=thaw_json(chosen)
                    sink.write(output/'selection.json',account)
                    report['selection']={'path':'selection.json','sha256':sha(output/'selection.json'),
                        **{k:account[k] for k in ('model_transition_calls','goal_diagnostic_preview_calls','dependency_preview_calls',
                        'selected_prefix_length','selected_total_return','selected_net_suffix_return','explored_total_return','explored_net_suffix_return')}}
                sealed={'ORIGINAL_STOP':{'actions':old['actions'],'original_plan_seal':oldplan['plan_seal']},
                    'FOCUSED_PREPARATION':{'selected_action_records':chosen,'explored_action_records':account['explored_action_records'],
                        'selected_total_return':account['selected_total_return'],'frontier_seal':frontier['seal']}}
                sealed={k:{**v,'seal':semantic(v)} for k,v in sealed.items()}
                sink.write(output/'plans.json',sealed);report['plans_sealed_before_replay']={'sha256':sha(output/'plans.json'),
                    'seals':{k:v['seal'] for k,v in sealed.items()}}
                save_report()
                with at('original_fresh_full_replay_and_audit'):
                    baseline=base.fresh();context.require_task(baseline)
                    for action in sealed['ORIGINAL_STOP']['actions']:checkpoint();baseline.step(action)
                    result=endpoint(baseline,'ORIGINAL_STOP',world)
                    sink.write(dest/'original.json',result);report['branches']['ORIGINAL_STOP']=result;save_report()
                    exact(result['history'],old['history'],'original complete retained history projection')
                    exact(result['audit']['outcomes'],old['audit']['outcomes'],'original retained complete outcomes')
                    if result['audit']['accepted'] is not True:raise ValueError('Original independent replay rejected')
                    report['baseline_reproduced']=True
                with at('focused_shared_fresh_full_replay'):
                    selected,explored,shared=replay_focused_tree(focused.fresh(),chosen,
                        account['explored_action_records'],account['explored_history'],checkpoint=checkpoint)
                    report['shared_replay_cost_scope']=shared
                with at('focused_exploration_independent_audit'):
                    exploration=endpoint(explored,'FOCUSED_EXPLORATION',world)
                    sink.write(dest/'exploration.json',exploration);report['exploration']=exploration;save_report()
                    if exploration['audit']['accepted'] is not True:raise ValueError('Full explored preparation geometry rejected')
                    if exploration['audit']['outcomes']['total_reward']!=account['explored_total_return']:
                        raise ValueError('Explored nominal and full authoritative return differ')
                with at('focused_selected_independent_audit'):
                    result=exploration if selected is explored else endpoint(selected,'FOCUSED_PREPARATION',world)
                    result={**result,'method':'FOCUSED_PREPARATION'}
                    sink.write(dest/'selected.json',result);report['branches']['FOCUSED_PREPARATION']=result;save_report()
                    if result['audit']['accepted'] is not True:raise ValueError('Selected complete endpoint rejected')
                    if result['audit']['outcomes']['total_reward']!=account['selected_total_return']:
                        raise ValueError('Selected nominal and full authoritative return differ')
                return {'subject':PAIR_SUBJECT,'prefix_moves':12,'complete_branches':2,
                    'complete_exploration_audited':True,'new_planning_calls':1}
            visits.run(PAIR_SUBJECT,dest,'same045_source',compare);report['source_visits']=visits.completed
            report['diagnosis']=useful_gate(report['branches'])
            if not report['diagnosis']['comparison_admissible']:raise ValueError('Both complete outcomes required')
            report['status']='completed';budget.complete(history_complete=True)
    except BaseException as error:
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,
            'message':str(error),'committed':bool(getattr(error,'committed',False))})
        traceback.print_exc()
    finally:
        report['native_budget']=budget.snapshot();report['costs']=costs.rows
        report['native_transition_attempts']=transitions['attempted'];report['native_transition_counts']=transitions
        report['total_frozen_policy_forwards']=costs.forwards;report['worker_seconds']=time.perf_counter()-started
        report['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost_scope']='All source construction, both fixed prefix executions, original witness, full suffix exploration, shared fresh replay and independent checks paid. No models/cache/weights. No omitted failed/capped arm.'
        try:
            # Persist each exact first-failure set once; pages/goals link its fingerprint.
            sink.write(output/'first-failure-sidecars.json',{'records':sidecars,'phase_counts':sidecar_calls,
                'scope':'original frontier and every explored suffix query only; immutable fixed-prefix/replay rediscovery excluded from sidecar storage but fully charged. First rejected shaft intervals with explicit truncation, not full corridors'})
            report['sidecars']={'path':'first-failure-sidecars.json','sha256':sha(output/'first-failure-sidecars.json'),'unique_count':len(sidecars)}
        except BaseException as error:
            report['status']='failed_or_incomplete';report['sidecar_output_failure']=str(error)
        try:
            guard(ROOT,index)
            for name,digest in index['input_metadata'].items():read_bytes(ROOT/name,digest)
            report['source_unchanged']=True
        except BaseException as error:
            report['status']='failed_or_incomplete';report['finalization_failure']=str(error)
        save_report()
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-index-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.perf_counter()
    index=read_json(HERE/'source-index.json',args.source_index_sha256);guard(ROOT,index)
    config=read_json(HERE/'config.json');output=Path(args.output).resolve()
    if not index['execution_released'] or output!=ROOT/config['output_directory'] or not output.is_dir():
        raise ValueError('Exact root-released output required')
    if not sys.dont_write_bytecode or not sys.pycache_prefix or Path(sys.pycache_prefix).exists():raise ValueError('Fresh absent source cache required')
    lease_fd=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    def lease():os.read(lease_fd,1);os._exit(72)
    threading.Thread(target=lease,daemon=True).start()
    def deadline(*unused):raise TimeoutError('End-to-end fixed frontier cap')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,config['caps']['work_seconds'])
    install_boundary(index,output);sys.path.insert(0,str(ROOT/'src'))
    import_execution_modules()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=execute(index,args.source_index_sha256,config,output,started)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'status':result['status'],'worker_seconds':result['worker_seconds'],'diagnosis':result.get('diagnosis')}))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
