"""One fixed TRAIN045 matched focused-catalog frozen-tail contrast; root execution only."""
import argparse
from contextlib import contextmanager
import json,os,resource,signal,sys,threading,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'model')]
from common import ROOT,IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,canonical,semantic,read_bytes,read_json,sha,guard,tree_bytes,write_json,runtime_roots,PAIR_SUBJECT
from inputs import (REPORT,REPORT_SHA,PLANS,PLANS_SHA,WORLD_KEYS,retained_prefix,
    PRIOR_REPORT,PRIOR_PARENT,require_reference,TAIL_REPORT,require_tail_metadata,CHECKPOINTS)
from common import TRAIN
from comparison import matched_gate,validate_plan
from compact_output import encode_gzip

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
    from resectionlab.core import freeze_json,thaw_json
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget,load_cohort_checkpoint
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.observed_search import ObservedSearchLimit
    from resectionlab.spatial_policy import parameter_hash
    from frontier_protocol import enabled_task,map_prefix,compact_history,physical,exact,frontier_record
    from focused_policy import admit_focused_context,FrozenFocusedTail,rollout
    from saved_tail import load_saved_tail
    caps=config['caps'];end=started+caps['work_seconds'];sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,caps['frozen_policy_forwards']);phase=['metadata'];sidecars={};sidecar_calls={}
    transitions={'attempted':0,'completed':0,'raised':0};tail=None
    report={'schema':config['schema'],'status':'running','config':config,'source_index_sha256':index_sha,
        'source_unchanged':False,'branches':{},'checkpoint_loads':0,'checkpoint_loads_attempted':0,
        'cache_loads':0,'head_optimizer_updates':0,'closed_roles_opened':False,
        'plans_sealed_before_replay':False,'baseline_prefix_reproduced':False,'frozen_models_unchanged':False,
        'scope':config['scope'],'selections':{}}
    def checkpoint():
        if time.perf_counter()>=end:raise TimeoutError('End-to-end matched focused-policy limit')
        budget.check();sink.check()
        if tree_bytes(HERE.parent)>caps['whole_build_bytes']-65536:raise InterruptedError('Whole stage cap')
    def cancel():checkpoint();return False
    def save_report():write_json(output/'report.json',thaw_json(report))
    def save_gzip(path,value):
        raw,binding=encode_gzip(thaw_json(freeze_json(value)))
        sink.write_bytes(output/path,raw);return {'path':path,**binding}
    def progress(label,**details):
        checkpoint();write_json(output/'progress.json',{'phase':label,'elapsed_seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'transition_counts':transitions,'policy_forwards':costs.forwards,
            'unique_sidecars':len(sidecars),**details},65536)
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
            if value.obstruction_diagnostic is not None and phase[0] in ('GREEDY_suffix_planning','TAIL_suffix_planning'):
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
            'original_admitted_world':world,'source_hash':task.case.source_hash,
            'decision_model_hash':task.decision_model_hash,'native_model_hash':task._config.fingerprint,
            'native_state_hash':task._engine.state_hash,'committed_masks':dict(task._engine.committed_mask_digests()),
            'history_hash':semantic(history),'history':compact_history(history),'audit':audit}
    try:
        # Keep this order: budget must wrap counters, never be replaced by them.
        with counters(),budget,costs,_bounded_writes(sink):
            with at('saved_metadata_admission'):
                for name,digest in index['input_metadata'].items():read_bytes(ROOT/name,digest)
                prior=pinned(REPORT);plans=pinned(PLANS);old,oldplan=retained_prefix(prior,plans)
                report['retained_capability_reference']=require_reference(pinned(PRIOR_REPORT),pinned(PRIOR_PARENT))
                tail_fit,worlds=require_tail_metadata(pinned(TAIL_REPORT))
                report['retained_prefix']={'path':REPORT,'sha256':REPORT_SHA,'plans_sha256':PLANS_SHA,
                    'plan_seal':oldplan['plan_seal'],'prior_costs_not_erased':prior['pair_plus_recovery_costs']}
                original=pinned(IL+'/configuration.json');protocol=original['learning_protocol'];limits=original['admission_limits']
                release={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'actual_caps':caps,'root_source_index_sha256':index_sha,
                    'execution_released':True,'scope':'original045 source admission; separate explicit focused deployment'}
                sink.write(output/'world-admission.json',release)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=release,released_sha256=sha(output/'world-admission.json'),progress=progress)
            with at('load_saved_frozen_IL64_and_TAIL'):
                contexts={pinned(IL+'/'+s+'-context.json')['patient_group']:semantic(pinned(IL+'/'+s+'-context.json')) for s in TRAIN}
                pin=CHECKPOINTS['IL64'];report['checkpoint_loads_attempted']+=1
                policy,meta=load_cohort_checkpoint(ROOT/pin['path'],expected_sha256=pin['sha256'],
                    expected_learning_protocol=protocol,expected_context_hashes=contexts,expected_method='IL')
                if parameter_hash(policy)!=pin['parameters']:raise ValueError('Original base parameter identity differs')
                policy.eval();policy.requires_grad_(False);report['checkpoint_loads']+=1
                pin=CHECKPOINTS['TAIL'];report['checkpoint_loads_attempted']+=1
                raw=read_bytes(ROOT/pin['path'],pin['sha256'],128*1024)
                if len(raw)!=pin['bytes']:raise ValueError('Exact saved tail byte count required')
                manifest={'schema':'logged-continuation-head-only-v2','method':'TAIL','objective':tail_fit['target'],
                    'prediction_semantics':tail_fit['prediction_semantics'],'base':CHECKPOINTS['IL64'],
                    'head_hash':tail_fit['fitted_head_hash'],'corpus_seal':tail_fit['corpus_seal']}
                tail=load_saved_tail(policy,raw,expected_sha256=pin['sha256'],expected_manifest=manifest,
                    tail_report=tail_fit,expected_report_hash=semantic(tail_fit),worlds=worlds,
                    expected_worlds_hash=semantic(worlds),max_scores=24,seconds=caps['work_seconds'],cancelled=cancel)
                del raw;report['checkpoint_loads']+=1
                report['model_provenance']={'checkpoints':CHECKPOINTS,'original_tail_fit':tail_fit,
                    'original_training_worlds_hash':semantic(worlds),'original_training_worlds':worlds,
                    'load_scope':'two saved frozen artifacts; no feature cache, reconstruction, fitting or model copy'}
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,checkpoint)
            dest=output/'evaluation';dest.mkdir()
            def compare(base,context):
                world={k:context.record()[k] for k in WORLD_KEYS};exact(world,old['world'],'original retained world')
                with at('genuinely_execute_original_and_focused_prefix'):
                    auth=base.fresh();context.require_task(auth)
                    focused,model=enabled_task(base);nominal=focused.planning_clone()
                    records,frontier,joins=map_prefix(auth,nominal,old['actions'][:-1],old['history'][:-1],checkpoint=checkpoint)
                    if len(records)!=12 or auth._steps!=12:raise ValueError('Fixed12 prefix denominator changed')
                    deployment=admit_focused_context(auth,context,nominal,records,frontier,expected_prefix_actions=12)
                    report['physical_model_join']=model;report['fixed_frontier']=frontier
                    report['baseline_prefix_reproduced']=True
                    sink.write(output/'focused-deployment.json',{'record':deployment.record,'seal':deployment.seal})
                    sink.write(output/'fixed-prefix.json',{'joins':joins,'original_frontier_state':auth._engine.state_hash,
                        'focused_frontier':frontier,'focused_packets':records,
                        'scope':'Genuine original and focused executions; physical history and rewards exact.'})
                    # Existing terminal ledger only: no diagnostic preview re-query.
                    sink.write(output/'original-frontier-reference.json',{'inventory':auth.candidate_inventory(),
                        'scope':'Already prepared original terminal ledger; reference only, no separate arm or extra preview.'})
                selected={};accounts={}
                scorer=FrozenFocusedTail(tail,deployment,max_scores=12,seconds=caps['work_seconds'],cancelled=cancel)
                for method in config['methods']:
                    with at(method+'_suffix_planning'):
                        try:
                            chosen,account=rollout(nominal,deployment,records,frontier,method=method,
                                scorer=scorer if method=='TAIL' else None,seconds=config['planner_seconds'],
                                cancelled=cancel,progress=progress)
                        except BaseException as error:
                            partial=getattr(error,'focused_policy_accounting',getattr(error,'accounting',None))
                            score_partial=getattr(error,'focused_score_accounting',None)
                            if partial is not None or score_partial is not None:
                                report['incomplete_selection']={method:save_gzip(method+'-partial-selection.json.gz',
                                    {'policy':partial,'scoring':score_partial})}
                            raise
                        account=thaw_json(account);chosen=thaw_json(chosen)
                        validate_plan(chosen,account,records,frontier,config['max_steps'])
                        exact(frontier_record(nominal,records),frontier,'original common frontier unchanged after '+method)
                        selected[method]=chosen;accounts[method]=account
                        if method=='TAIL':
                            scorer.require();report['focused_tail_scoring_costs']=dict(scorer.costs)
                            report['focused_scoring_declaration']={'record':scorer.record,'seal':scorer._seal}
                        report['selections'][method]={**save_gzip(method+'-selection.json.gz',account),
                            'complete':True,'packet_count':len(chosen),'model_transition_calls':account['model_transition_calls'],
                            'planned_total_return':sum(r['reward'] for r in account['explored_history']),
                            'planned_suffix_return':sum(r['reward'] for r in account['explored_history'][12:])}
                sealed={method:{'selected_action_records':selected[method],
                    'planned_history_hash':semantic(accounts[method]['explored_history']),
                    'planned_total_return':sum(r['reward'] for r in accounts[method]['explored_history']),
                    'frontier_seal':frontier['seal']} for method in config['methods']}
                sealed={k:{**v,'seal':semantic(v)} for k,v in sealed.items()}
                sink.write(output/'plans.json',sealed)
                report['plans_sealed_before_replay']={'sha256':sha(output/'plans.json'),
                    'seals':{k:v['seal'] for k,v in sealed.items()},'native_transition_attempts_at_seal':transitions['attempted']}
                save_report()
                with at('shared_fresh_authoritative_prefix_replay'):
                    fresh=focused.fresh()
                    for i,packet in enumerate(records):
                        checkpoint();transition=fresh.step_record(packet)
                        exact(physical(transition.info),physical(accounts['GREEDY']['explored_history'][i]),'fresh physical prefix')
                        if transition.terminated:raise ValueError('Fixed prefix terminated early')
                    if fresh._planning or fresh._steps!=12:raise ValueError('Fresh authoritative prefix required')
                    report['shared_replay']={'prefix_transitions':12,'suffix_transitions':{},
                        'scope':'One genuinely replayed fresh12 prefix, independent authoritative suffix clones; each full history audited.'}
                for method in config['methods']:
                    with at(method+'_authoritative_suffix_replay_and_full_independent_audit'):
                        replay=fresh.clone();chosen=selected[method];history=accounts[method]['explored_history']
                        exact(chosen[:12],records,'sealed shared packet prefix')
                        for i,packet in enumerate(chosen[12:],12):
                            checkpoint();transition=replay.step_record(packet)
                            exact(physical(transition.info),physical(history[i]),method+' full executed native physical history')
                            if transition.terminated and i!=len(chosen)-1:raise ValueError('Packet follows termination')
                        if not replay.terminated:raise ValueError('Partial route cannot be accepted')
                        result=endpoint(replay,method,world)
                        exact(result['actions'],[p['action_id'] for p in chosen],'selected packet and replay action IDs')
                        if not result['audit']['accepted']:raise ValueError('Independent full-tool endpoint rejected')
                        if abs(result['audit']['outcomes']['total_reward']-sealed[method]['planned_total_return'])>1e-10:
                            raise ValueError('Nominal and authoritative cumulative reward differ')
                        result['plan_seal']=sealed[method]['seal'];result['planned_physical_history_matches']=True
                        sink.write(dest/(method+'.json'),result);report['branches'][method]=result
                        report['shared_replay']['suffix_transitions'][method]=len(chosen)-12;save_report()
                # Neither scorer nor these callback-local tasks escape source-release guard.
                tail.require()
                report['frozen_models_unchanged']=True
                return {'subject':PAIR_SUBJECT,'prefix_moves':12,'complete_branches':2,'new_planning_calls':2,
                    'full_history_independent_checks':2,'same_catalog_and_controller':True}
            visits.run(PAIR_SUBJECT,dest,'same045_source',compare);report['source_visits']=visits.completed
            report['diagnosis']=matched_gate(report['branches'])
            if not report['diagnosis']['comparison_admissible']:raise ValueError('Both complete outcomes required')
            tail.require();report['frozen_models_unchanged']=True
            report['status']='completed';budget.complete(history_complete=True)
    except BaseException as error:
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,
            'message':str(error),'committed':bool(getattr(error,'committed',False))})
        traceback.print_exc()
    finally:
        report['native_budget']=budget.snapshot();report['costs']=costs.rows
        report['native_transition_attempts']=transitions['attempted'];report['native_transition_counts']=transitions
        report['total_frozen_policy_forwards']=costs.forwards
        if tail is not None:
            try:report['original_trained_world_scorer_costs']=dict(tail.costs);tail.require()
            except BaseException as error:report['status']='failed_or_incomplete';report['model_finalization_failure']=str(error)
        report['worker_seconds']=time.perf_counter()-started
        report['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost_scope']='All source construction, metadata/model loads, both fixed prefix executions, both full page discoveries and selections, model scores, shared fresh replay and independent checks paid. No failed/capped arm omitted; retained historical costs separate.'
        try:
            report['sidecars']={**save_gzip('first-failure-sidecars.json.gz',{'records':sidecars,'phase_counts':sidecar_calls,
                'scope':'Complete native first-failure records for both suffix planners; fixed-prefix/replay rediscovery is fully charged but not duplicated here.'}),
                'unique_count':len(sidecars)}
        except BaseException as error:report['status']='failed_or_incomplete';report['sidecar_output_failure']=str(error)
        try:
            guard(ROOT,index)
            for name,digest in index['input_metadata'].items():read_bytes(ROOT/name,digest)
            report['source_unchanged']=True
        except BaseException as error:report['status']='failed_or_incomplete';report['finalization_failure']=str(error)
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
    def deadline(*unused):raise TimeoutError('End-to-end focused policy cap')
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
