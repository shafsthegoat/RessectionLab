"""Single root-owned fresh local-task qualification. No fitting or model loads."""
import argparse,json,os,resource,signal,sys,threading,time,traceback
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paths
from paths import HERE,ROOT,DEPS
from common import (IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,read_json,read_bytes,sha,guard,
    runtime_roots,write_json,tree_bytes,semantic)
from source_inputs import *


def install_boundary(index,output):
    expected=runtime_roots()+[str(DEPS.resolve())]
    if index['runtime_roots']!=expected:raise ValueError('Declared existing dependency roots differ')
    dependencies=tuple(Path(p) for p in expected)
    allowed={str((ROOT/name).absolute()) for name in (*index['files'],*index['input_metadata'],*index['public_arrays'])}
    def boundary(event,values):
        if event.startswith(('socket.','subprocess.')) or event in ('os.fork','os.forkpty','os.posix_spawn','os.system'):
            raise PermissionError('No acquisition/network/child process in qualification worker')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.path.abspath(os.fsdecode(values[0])));mode,flags=values[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if any(p.is_relative_to(d) for d in dependencies):
                if writing or not any(p.resolve().is_relative_to(d) for d in dependencies):raise PermissionError('Read-only dependencies')
                return
            if p.is_relative_to(ROOT) and not p.is_relative_to(output):
                if writing or str(p) not in allowed:raise PermissionError('Undeclared scientific input: '+str(p))
    sys.addaudithook(boundary)


def coverage_record(catalog,index):
    import numpy as np
    o=catalog.observation;g=o.action_geometry[index];a=o.affine_ras_mm;shape=np.array(o.image_channels.shape[1:])
    points=g[1:4]+np.linspace(0,1,5)[:,None]*(g[4:7]-g[1:4])
    xyz=(points-a[:3,3])@np.linalg.inv(a[:3,:3]).T
    from resectionlab.spatial_policy import world_to_sample_grid
    grid=world_to_sample_grid(points,a,shape)
    cells=[]
    for p in xyz:
        corners=[]
        for x in (int(np.floor(p[0])),int(np.ceil(p[0]))):
            for y in (int(np.floor(p[1])),int(np.ceil(p[1]))):
                for z in (int(np.floor(p[2])),int(np.ceil(p[2]))):corners.append((x,y,z))
        inside=all(all(0<=c[j]<shape[j] for j in range(3)) for c in corners)
        cells.append([bool(inside and o.channel_available[k] and all(o.coverage[(k,*c)] for c in corners)) for k in range(6)])
    return {'action_id':catalog.identifiers[index],'ray_sample_source_coordinates':xyz.tolist(),
        'five_sample_inbounds':(np.abs(grid)<=1).all(axis=1).tolist(),
        'six_channel_all_interpolation_corners_covered':cells,
        'scope':'Unrecentered public actor crop; source channel coverage, not physical safety'}


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
    from frontier_protocol import enabled_task,map_prefix,physical,exact,compact_history
    from inputs import retained_prefix
    from compact_output import encode_gzip
    from local_goal import GoalSpec,admit_local_context,local_reward
    from gym_adapter import LocalGoalEnv
    import numpy as np
    caps=config['caps'];end=started+caps['work_seconds'];sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,0);transitions={'attempted':0,'completed':0,'raised':0};phase=['metadata']
    report={'schema':config['schema'],'status':'running','config':config,'source_index_sha256':index_sha,
        'source_unchanged':False,'qualified':False,'checkpoint_loads':0,'policy_forwards':0,'optimizer_updates':0,
        'closed_roles_opened':False,'plans_sealed_before_replay':False,'scope':config['scope']}
    def check():
        if time.perf_counter()>=end:raise TimeoutError('Local task qualification deadline')
        budget.check();sink.check()
    def progress(label,**details):
        check();write_json(output/'progress.json',{'phase':label,'seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'transitions':transitions,**details},65536)
    def save():write_json(output/'report.json',thaw_json(report))
    def save_gzip(name,value):
        raw,binding=encode_gzip(thaw_json(freeze_json(value)));sink.write_bytes(output/name,raw)
        return {'path':name,**binding}
    @contextmanager
    def at(label):
        phase[0]=label;progress(label)
        with costs.scope(label):yield
        progress(label+'.complete')
    @contextmanager
    def count_transitions():
        old=NativeSpatialTask._transition
        def counted(task,*a,**k):
            if transitions['attempted']>=caps['native_transitions']:raise InterruptedError('Qualification transition cap')
            transitions['attempted']+=1
            try:r=old(task,*a,**k);transitions['completed']+=1;return r
            except BaseException:transitions['raised']+=1;raise
        NativeSpatialTask._transition=counted
        try:yield
        finally:NativeSpatialTask._transition=old
    def pinned(name):return read_json(ROOT/name,index['input_metadata'][name])
    try:
        with count_transitions(),budget,costs,_bounded_writes(sink):
            with at('metadata_admission'):
                for name,digest in index['input_metadata'].items():read_bytes(ROOT/name,digest)
                retained,_=retained_prefix(pinned(PREFIX_REPORT),pinned(PREFIX_PLANS))
                historical=pinned(REFERENCE_REPORT);reference=pinned(REFERENCE_PLANS)['FOCUSED_PREPARATION']
                if historical['status']!='completed' or not historical['exploration']['audit']['accepted']:
                    raise ValueError('Accepted reference required')
                cfg=pinned(IL+'/configuration.json');protocol=cfg['learning_protocol'];limits=cfg['admission_limits']
                release={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'root_source_index_sha256':index_sha,'actual_caps':caps,
                    'scope':config['scope'],'execution_released':True}
                sink.write(output/'world-admission.json',release)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=release,released_sha256=sha(output/'world-admission.json'),progress=progress)
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,check);dest=output/'evaluation';dest.mkdir()
            def qualify(base,original_context):
                declaration=dict(config['goal'])
                for key in ('cell','entry_mm','tip_mm'):declaration[key]=tuple(declaration[key])
                goal=GoalSpec(**declaration)
                with at('genuine_fixed12_prefix'):
                    auth=base.fresh();focused,join=enabled_task(base);nominal=focused.planning_clone()
                    packets,frontier,joins=map_prefix(auth,nominal,retained['actions'][:-1],retained['history'][:-1],checkpoint=check)
                    context=admit_local_context(auth,original_context,nominal,packets,frontier,goal)
                    report['task_declaration']={'record':context.record,'seal':context.seal};report['physical_model_join']=join
                    report['prefix_frontier']=frontier;report['prefix_joins']=joins
                    # The fixed physical goal must initially be blocked, from the
                    # already prepared authoritative base catalog, no extra query.
                    initial=[r for r in auth._ledger if r.get('tool_id')==goal.tool_id and
                        tuple(r.get('entry_mm',()))==goal.entry_mm and tuple(r.get('tip_mm',()))==goal.tip_mm]
                    if not initial or any(r.get('feasible') or r.get('reason')!='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' for r in initial):
                        raise ValueError('Reference local goal is not initially shaft blocked')
                    report['initial_goal_dispositions']=initial
                snapshots={12:nominal.clone()};env=LocalGoalEnv(context,snapshots,guard=check)
                try:
                    with at('five_action_reference_and_STOP'):
                        observation,info=env.reset();report['observation_shapes']={k:list(v.shape) for k,v in observation.items()}
                        if not env.observation_space.contains(observation):raise ValueError('Gym reset observation differs from declared spaces')
                        report['goal_crop_coverage']={'cell':list(goal.cell),'crop_origin':list(nominal.case._crop_origin),
                            'crop_shape':list(nominal.case._crop_shape),'goal_mask_sum':float(observation['goal_mask'].sum()),
                            'per_channel_covered':observation['coverage'][(slice(None),*(i-a for i,a in zip(goal.cell,nominal.case._crop_origin)))].tolist(),
                            'no_recenter':True}
                        rays=reference['selected_action_records'][12:17];coverage=[]
                        for number,old_packet in enumerate(rays,13):
                            check();catalog=env.episode.observe();physical_ray=old_packet['physical_action']
                            candidates=[i for i,p in enumerate(catalog.packets) if p['physical_action'] is not None and
                                semantic(p['physical_action'])==semantic(physical_ray)]
                            if len(candidates)!=1:raise ValueError('Fixed-focus catalog omits or aliases reference physical action '+str(number))
                            choice=candidates[0];coverage.append(coverage_record(catalog,choice))
                            observation,reward,done,truncated,step_info=env.step(choice)
                            if truncated or (done!=(number==17)):raise ValueError('Goal cell removed at unexpected reference step')
                            # Comparison is physical/outcome based, never old query authority.
                            new=env.episode.work._history[number-1];expected=historical['exploration']['history'][number-1]
                            actual=compact_history([new])[0]
                            for key in ('tool_id','entry_mm','tip_mm','target_removed_mm3','normal_removed_mm3','reward','removed_indices_hash','contact_indices_hash','microsteps'):
                                exact(actual[key],expected[key],'reference physical '+key)
                            if not done:
                                # Captured immediately after canonical commit,
                                # before Gym constructs a successor catalog.
                                snapshots[number]=env.episode.last_clean_snapshot.clone()
                            report['reference_progress']={'absolute_step':number,'task_return':env.episode.total_reward,
                                'costs':dict(env.episode.costs),'coverage':coverage};save()
                        if not env.episode.success or not env.episode.work.terminated:raise ValueError('Reference failed to complete goal and STOP')
                        full_packets=[*packets,*env.episode.packets];nominal_history=list(env.episode.work._history)
                        task_rows=list(env.episode.rows);nominal_return=env.episode.total_reward
                        if len(full_packets)!=18 or full_packets[-1]['action_id']!='STOP' or set(snapshots)!=set(config['curriculum_starts']):
                            raise ValueError('Reference or genuine curriculum denominator differs')
                        report['reference_packets']=save_gzip('reference-packets.json.gz',full_packets)
                        report['nominal_history']=save_gzip('nominal-history.json.gz',nominal_history)
                        plan={'task_context_seal':context.seal,'packets_hash':semantic(full_packets),'history_hash':semantic(nominal_history),
                            'local_rows':task_rows,'local_return':nominal_return,'actions':18,'success':True}
                        sink.write(output/'sealed-reference-plan.json',{**plan,'seal':semantic(plan)})
                        report['plans_sealed_before_replay']=True;save()
                    with at('fresh_authoritative_full_replay_and_independent_geometry'):
                        replay=focused.fresh()
                        for i,packet in enumerate(full_packets):
                            check();transition=replay.step_record(packet)
                            exact(physical(transition.info),physical(nominal_history[i]),'fresh complete physical history')
                        audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(check() or False))
                        if not audit['accepted']:raise ValueError('Full independent native geometry rejected')
                        if goal.cell not in {tuple(c) for c in replay._history[16]['removed_indices_native']}:
                            raise ValueError('Independent replay lacks actual declared goal-cell removal')
                        before={tuple(c) for r in replay._history[:12] for c in r.get('removed_indices_native',())}
                        if goal.cell in before:raise ValueError('Goal already removed in fixed prefix')
                        succeeded=False;total=0.;prior=replay._history[11]['tool_id'];independent_rows=[]
                        for row in replay._history[12:]:
                            # Separately expressed scalar audit, not local_reward.
                            cells={tuple(c) for c in row.get('removed_indices_native',())};new=(goal.cell in cells and not succeeded)
                            value=0. if row['action_id']=='STOP' else ((10. if new else 0.)-.03
                                -.001*row['complete_tool_path_length_mm']-.2*row['normal_removed_mm3']
                                -.03*int(prior is not None and row['tool_id']!=prior))
                            total+=value;succeeded=succeeded or new;independent_rows.append(value)
                            if row['action_id']!='STOP':prior=row['tool_id']
                        if not succeeded or abs(total-nominal_return)>1e-9:raise ValueError('Independent local reward differs')
                        report['independent_audit']=audit;report['local_reward_audit']={'accepted':True,'total':total,'rows':independent_rows,'bonus_count':1}
                        report['authoritative_history']=save_gzip('authoritative-history.json.gz',replay._history)
                    with at('genuine_snapshot_reset_and_observation_qualification'):
                        env.close();env=LocalGoalEnv(context,snapshots,guard=check);reset_rows=[]
                        for step in config['curriculum_starts']:
                            check();before=env._stamp(snapshots[step]);obs,info=env.reset(options={'start_step':step})
                            if not env.observation_space.contains(obs) or env._stamp(snapshots[step])!=before:
                                raise ValueError('Reset changed genuine source snapshot or public observation')
                            reset_rows.append({**info,'legal_rows':int(obs['action_mask'].sum()),'catalog_hash':env.episode.catalog._seal,
                                'native_state_hash':env.episode.work._engine.state_hash,'history_hash':semantic(env.episode.work._history)})
                        report['qualified_resets']=reset_rows
                        report['qualified']=True;report['success_scope']='One fixed simulated local corridor, not whole tumor resection, policy gain or clinical safety'
                finally:env.close()
                return {'subject':'ReMIND-045','qualified':True,'reference_actions':18,'independent_audits':1,'patient_conditions':1}
            visits.run('ReMIND-045',dest,'TRAIN045_local_qualification',qualify);report['source_visits']=visits.completed
            budget.complete(history_complete=True);report['status']='completed'
    except BaseException as error:
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,'message':str(error),
            'committed':bool(getattr(error,'committed',False))});traceback.print_exc()
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
    guard(ROOT,index);config=read_json(HERE/'config.json');output=Path(args.output).resolve()
    if not index['execution_released'] or output!=ROOT/config['output_directory'] or not output.is_dir():raise ValueError('Exact released single output required')
    if not sys.dont_write_bytecode or not sys.pycache_prefix or Path(sys.pycache_prefix).exists():raise ValueError('Fresh absent runtime cache required')
    lease=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    threading.Thread(target=lambda:(os.read(lease,1),os._exit(72)),daemon=True).start()
    def deadline(*a):raise TimeoutError('Qualification worker deadline')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,config['caps']['work_seconds'])
    install_boundary(index,output)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=execute(index,args.source_index_sha256,config,output,started);signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'status':result['status'],'qualified':result['qualified'],'worker_seconds':result['worker_seconds']}))
    return 0 if result['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
