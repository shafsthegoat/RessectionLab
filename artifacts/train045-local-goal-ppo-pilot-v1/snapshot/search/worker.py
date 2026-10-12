"""One matched local-goal informed search and fresh full native endpoint audit."""
import argparse, json, os, resource, signal, sys, threading, time, traceback
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(HERE))
import paths
from paths import ROOT
from common import IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,read_json,read_bytes,sha,guard,write_json,semantic
from source_inputs import PREFIX_REPORT,PREFIX_PLANS


def reward_audit(goal_cell, history, *, prefix=12):
    """Independent scalar expression; does not call selector/task reward code."""
    if len(history)<prefix+1 or history[prefix-1]['action_id']=='STOP':
        raise ValueError('Executed fixed prefix and completed suffix required')
    if any(tuple(goal_cell) in {tuple(c) for c in r.get('removed_indices_native',())} for r in history[:prefix]):
        raise ValueError('Goal removed before the declared local start')
    prior=history[prefix-1]['tool_id'];succeeded=False;total=0.;values=[];bonuses=0
    for row in history[prefix:]:
        hit=tuple(goal_cell) in {tuple(c) for c in row.get('removed_indices_native',())}
        new=hit and not succeeded
        value=0. if row['action_id']=='STOP' else ((10. if new else 0.)-.03
            -.001*row['complete_tool_path_length_mm']-.2*row['normal_removed_mm3']
            -.03*int(prior is not None and row['tool_id']!=prior))
        if hit and row['action_id']=='STOP':raise ValueError('STOP cannot remove goal')
        if succeeded and row['action_id']!='STOP':raise ValueError('Local task continued after goal termination')
        total+=value;values.append(value);bonuses+=int(new);succeeded=succeeded or new
        if row['action_id']!='STOP':prior=row['tool_id']
    return {'accepted':True,'total':total,'success':succeeded,'rows':values,'bonus_count':bonuses}


def execute(index,index_sha,config,output,started):
    from resectionlab.core import freeze_json,thaw_json,semantic_digest as semantic
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from frontier_protocol import enabled_task,map_prefix,physical,exact
    from inputs import retained_prefix
    from compact_output import encode_gzip
    from local_goal import GoalSpec,admit_local_context
    from beam import search
    caps=config['caps'];end=started+caps['work_seconds'];sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,0);transitions={'attempted':0,'completed':0,'raised':0};phase=['metadata']
    report={'schema':config['schema'],'status':'running','config':config,'source_index_sha256':index_sha,
        'source_unchanged':False,'qualified':False,'checkpoint_loads':0,'policy_forwards':0,'optimizer_updates':0,
        'closed_roles_opened':False,'plans_sealed_before_replay':False,'scope':config['scope']}
    def clean(value):return thaw_json(freeze_json(value))
    def check():
        if time.perf_counter()>=end:raise TimeoutError('Matched search global worker wall cap')
        budget.check();sink.check()
    def usage():return {'previews':budget.snapshot()['native_preview_entries'],'transitions':transitions['attempted']}
    def progress(label,**details):
        check();write_json(output/'progress.json',clean({'phase':label,'seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'transitions':transitions,**details}),65536)
    def save():write_json(output/'report.json',clean(report),4*1024**2)
    def save_gzip(name,value):
        raw,binding=encode_gzip(clean(value));sink.write_bytes(output/name,raw);return {'path':name,**binding}
    @contextmanager
    def at(label):
        phase[0]=label;progress(label)
        with costs.scope(label):yield
        progress(label+'.complete')
    @contextmanager
    def count_transitions():
        old=NativeSpatialTask._transition
        def counted(task,*args,**kwargs):
            if transitions['attempted']>=caps['native_transitions']:raise InterruptedError('Matched search total transition cap')
            transitions['attempted']+=1
            try:value=old(task,*args,**kwargs);transitions['completed']+=1;return value
            except BaseException:transitions['raised']+=1;raise
        NativeSpatialTask._transition=counted
        try:yield
        finally:NativeSpatialTask._transition=old
    def pinned(name):return read_json(ROOT/name,index['input_metadata'][name])
    try:
        with count_transitions(),budget,costs,_bounded_writes(sink):
            with at('qualified_task_source_admission'):
                q=pinned(config['qualification_report']);parent=pinned(config['qualification_launcher'])
                if (q['status']!='completed' or not q['qualified'] or not q['source_unchanged']
                    or not q['independent_audit']['accepted'] or not q['local_reward_audit']['accepted']
                    or parent['status']!='completed' or parent['report_sha256']!=index['input_metadata'][config['qualification_report']]
                    or not parent['owned_lifecycle']['verified'] or not parent['owned_lifecycle']['child_reaped']
                    or parent['owned_lifecycle']['errors'] or parent['owned_lifecycle']['remaining_owned_pids']
                    or q['config']['goal']!=config['goal']):raise ValueError('Exact independently qualified task required')
                report['qualified']=True
                report['qualification']={'report_sha256':index['input_metadata'][config['qualification_report']],
                    'launcher_sha256':index['input_metadata'][config['qualification_launcher']],
                    'role':'feasibility only; reference trajectory never passed to search'}
                retained,_=retained_prefix(pinned(PREFIX_REPORT),pinned(PREFIX_PLANS))
                cfg=pinned(IL+'/configuration.json');protocol=cfg['learning_protocol'];limits=cfg['admission_limits']
                release={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'root_source_index_sha256':index_sha,
                    'actual_caps':caps,'scope':config['scope'],'execution_released':True}
                sink.write(output/'world-admission.json',release)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=release,released_sha256=sha(output/'world-admission.json'),progress=progress)
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,check);dest=output/'source';dest.mkdir()
            def compare(base,original_context):
                with at('genuine_fixed12_prefix'):
                    auth=base.fresh();focused,join=enabled_task(base);nominal=focused.planning_clone()
                    prefix,frontier,joins=map_prefix(auth,nominal,retained['actions'][:-1],retained['history'][:-1],checkpoint=check)
                    declaration=dict(config['goal'])
                    for key in ('cell','entry_mm','tip_mm'):declaration[key]=tuple(declaration[key])
                    context=admit_local_context(auth,original_context,nominal,prefix,frontier,GoalSpec(**declaration))
                    report['local_context']={'record':context.record,'seal':context.seal}
                    report['physical_model_join']=join;report['prefix_frontier']=frontier;report['prefix_joins']=joins
                with at('bounded_informed_beam'):
                    selected,account=search(context,nominal,limits=config['search_limits'],usage=usage,guard=check,
                        progress=lambda **kw:progress('bounded_informed_beam',**kw))
                    # Save all ranked-row and proposed-child evidence losslessly
                    # before replay, without duplicating full branch histories.
                    report['search_trace']=save_gzip('search-trace.json.gz',account)
                    report['search']={k:v for k,v in account.items() if k!='layers'}
                    packets=[*prefix,*selected.packets];history=list(selected.work._history)
                    if not selected.done or not selected.work.terminated or len(packets)>24:
                        raise ValueError('Selected native route must actually be complete within absolute24')
                    plan={'complete':True,'context_seal':context.seal,'packets':packets,
                        'packets_hash':semantic(packets),'history_hash':semantic(history),
                        'local_rows':selected.rows,'return':selected.total_reward,'success':selected.success,
                        'search_complete':account['complete'],'search_cap_reason':account['cap_reason']}
                    plan['seal']=semantic(plan)
                    report['sealed_plan']=save_gzip('sealed-plan.json.gz',plan)
                    report['nominal_history']=save_gzip('nominal-history.json.gz',history)
                    report['plans_sealed_before_replay']=True;save()
                with at('fresh_complete_replay_and_independent_geometry'):
                    replay=focused.fresh()
                    for i,packet in enumerate(packets):
                        check();transition=replay.step_record(packet)
                        exact(physical(transition.info),physical(history[i]),'fresh full matched-search physical history')
                    audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(check() or False))
                    if not audit['accepted']:raise ValueError('Full independent native geometry rejected')
                    rewards=reward_audit(context.goal.cell,replay._history)
                    if rewards['success']!=plan['success'] or abs(rewards['total']-plan['return'])>1e-9:
                        raise ValueError('Independent local success/reward reconstruction differs')
                    report['independent_audit']=audit;report['local_reward_audit']=rewards
                    report['authoritative_history']=save_gzip('authoritative-history.json.gz',replay._history)
                    report['evaluation']={'complete':True,'success':rewards['success'],'return':rewards['total'],
                        'suffix_decisions':len(selected.rows),'suffix_packet_rows':len(selected.packets),
                        'full_history_rows':len(history),'plan_seal':plan['seal'],
                        'history_hash':semantic(replay._history),'search_complete':account['complete'],
                        'bounded_search_only':True,'globally_optimal':False}
                    save()
                return {'subject':'ReMIND-045','patients':1,'complete_endpoint':True,
                    'search_complete':account['complete'],'reference_not_search_input':True}
            visits.run('ReMIND-045',dest,'TRAIN045_matched_local_goal_search',compare)
            report['source_visits']=visits.completed
            budget.complete(history_complete=True);report['status']='completed'
    except BaseException as error:
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,'message':str(error)})
        if hasattr(error,'local_search_accounting'):
            report['failed_search_accounting']=save_gzip('failed-search-trace.json.gz',error.local_search_accounting)
        traceback.print_exc()
    finally:
        report['native_budget']=budget.snapshot();report['costs']=costs.rows;report['native_transition_counts']=transitions
        report['native_transition_attempts']=transitions['attempted'];report['worker_seconds']=time.perf_counter()-started
        report['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:guard(ROOT,index);report['source_unchanged']=True
        except BaseException as error:report['status']='failed_or_incomplete';report['finalization_error']=str(error)
        save()
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-index-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.perf_counter();index=read_json(HERE/'source-index.json',args.source_index_sha256)
    guard(ROOT,index);config=read_json(HERE/'config.json',index['files'][str((HERE/'config.json').relative_to(ROOT))])
    output=Path(args.output).resolve()
    if not index['execution_released'] or output!=ROOT/config['output_directory'] or not output.is_dir():
        raise ValueError('Exact root-released output required')
    if not sys.dont_write_bytecode or not sys.pycache_prefix or Path(sys.pycache_prefix).exists():
        raise ValueError('Isolated no-bytecode bootstrap required')
    lease=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    threading.Thread(target=lambda:(os.read(lease,1),os._exit(72)),daemon=True).start()
    def deadline(*args):raise TimeoutError('Matched search worker deadline')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,config['caps']['work_seconds'])
    import importlib.util
    spec=importlib.util.spec_from_file_location('pinned_local_qualification_worker',STAGE/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.install_boundary(index,output)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=execute(index,args.source_index_sha256,config,output,started)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'status':result['status'],'worker_seconds':result['worker_seconds']}))
    return 0 if result['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
