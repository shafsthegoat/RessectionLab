"""Root-released reverse512 checkpoint recovery; one source visit, immutable qualification.

This entry point does not freeze or launch itself. Every checkpoint precedes its
full-start evaluation seal, fresh native replay and independent geometry audit.
"""
import argparse, gc, hashlib, json, os, resource, signal, sys, threading, time, traceback
from pathlib import Path
from contextlib import contextmanager
HERE=Path(__file__).resolve().parent; STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(STAGE/'training'));sys.path.insert(0,str(STAGE/'campaign'));sys.path.insert(0,str(STAGE/'model'));sys.path.insert(0,str(HERE))
import paths
from paths import ROOT
from common import (IL,TRAIN,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,read_bytes,read_json,
                    sha,write_json,guard)
from source_inputs import PREFIX_REPORT,PREFIX_PLANS,REFERENCE_REPORT,REFERENCE_PLANS


def execute(index,index_sha,config,output,started):
    import numpy as np
    import torch
    from resectionlab.core import freeze_json,thaw_json,semantic_digest as semantic
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget,load_cohort_checkpoint
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.spatial_policy import parameter_hash
    from frontier_protocol import enabled_task,map_prefix,physical,exact,compact_history
    from inputs import retained_prefix,CHECKPOINTS
    from compact_output import encode_gzip
    from local_goal import GoalSpec,LocalEpisode,admit_local_context
    from gym_adapter import LocalGoalEnv
    from recovery_learning import run_recovery_arm
    from recovery_contract import require_failure
    from checkpoints import checkpoint_bytes
    from policy import VERSION
    caps=config['caps'];end=started+caps['work_seconds'];sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,0);transitions={'attempted':0,'completed':0,'raised':0};phase=['metadata']
    report={'schema':config['schema'],'status':'running','config':config,'source_index_sha256':index_sha,
        'source_unchanged':False,'qualified':False,'patients':1,'subject':'ReMIND-045','role':'TRAIN',
        'arms':{},'evaluations':{},'checkpoint_loads':0,'closed_roles_opened':False,
        'scope':'Simulated one-patient local task; no held-out/generalization/clinical claim',
        'retained_baselines':'Completed uniform1024 and reverse512 retained; reverse-only restarted trajectory gets one1024 endpoint'}
    def clean(value):return thaw_json(freeze_json(value))
    def check():
        if time.perf_counter()>=end:raise TimeoutError('PPO worker wall cap')
        budget.check();sink.check()
    def save():write_json(output/'report.json',clean(report),4*1024**2)
    def progress(label,**details):
        check();write_json(output/'progress.json',clean({'phase':label,'seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'transitions':transitions,**details}),65536)
    def save_gzip(name,value):
        raw,binding=encode_gzip(clean(value));sink.write_bytes(output/name,raw);return {'path':name,**binding}
    @contextmanager
    def at(label):
        phase[0]=label;progress(label)
        with costs.scope(label):yield
        progress(label+'.complete')
    @contextmanager
    def count_transitions():
        original=NativeSpatialTask._transition
        def counted(task,*args,**kwargs):
            if transitions['attempted']>=caps['native_transitions']:raise InterruptedError('PPO transition cap')
            transitions['attempted']+=1
            try:
                value=original(task,*args,**kwargs);transitions['completed']+=1;return value
            except BaseException:transitions['raised']+=1;raise
        NativeSpatialTask._transition=counted
        try:yield
        finally:NativeSpatialTask._transition=original
    def pinned(name):return read_json(ROOT/name,index['input_metadata'][name])
    try:
        with count_transitions(),budget,costs,_bounded_writes(sink):
            with at('qualification_and_source_admission'):
                pilot=pinned(config['campaign_report']);pilot_parent=pinned(config['campaign_launcher'])
                archive=config['failed_archive']
                failed=pinned(archive+'/stopped-report.json');failed_parent=pinned(archive+'/launcher.json')
                curve=pinned(archive+'/reverse512-curve.json');postmortem=pinned(archive+'/source-postmortem.json')
                old_index=pinned(config['failed_source_index'])
                report['recovery_admission']=require_failure(failed,failed_parent,curve,postmortem,old_index,
                    pins={'report':index['input_metadata'][archive+'/stopped-report.json'],
                        'launcher':index['input_metadata'][archive+'/launcher.json'],
                        'index':index['input_metadata'][config['failed_source_index']],
                        'postmortem':index['input_metadata'][archive+'/source-postmortem.json']},configuration=config)
                report['retained_uniform1024']=failed['arms']['uniform-seed0']['checkpoints'][-1]
                report['retained_reverse512']=curve['checkpoints'][0]
                report['memory_change']=index['memory_change']
                report['inherited_campaign']={'report':archive+'/stopped-report.json',
                    'report_sha256':index['input_metadata'][archive+'/stopped-report.json'],
                    'launcher_sha256':index['input_metadata'][archive+'/launcher.json'],
                    'source_index_sha256':index['input_metadata'][config['failed_source_index']],
                    'original_campaign_completed':False,'old_endpoint_evaluations_reexecuted':False}
                qualification=pinned(config['qualification_report']);parent=pinned(config['qualification_launcher'])
                if (qualification['status']!='completed' or not qualification['qualified']
                    or not qualification['source_unchanged'] or not qualification['independent_audit']['accepted']
                    or not qualification['local_reward_audit']['accepted'] or parent['status']!='completed'
                    or parent['report_sha256']!=index['input_metadata'][config['qualification_report']]
                    or not parent['owned_lifecycle']['verified'] or not parent['owned_lifecycle']['child_reaped']
                    or parent['owned_lifecycle']['errors'] or parent['owned_lifecycle']['remaining_owned_pids']
                    or {r['start_step'] for r in qualification['qualified_resets']}!={12,13,14,15,16}):
                    raise ValueError('Clean independently qualified reference and five genuine resets required')
                if config['goal']!=qualification['config']['goal']:raise ValueError('Qualified goal/reward changed')
                report['qualification']={'report':config['qualification_report'],
                    'report_sha256':index['input_metadata'][config['qualification_report']],
                    'launcher_sha256':index['input_metadata'][config['qualification_launcher']],
                    'historical_costs_separate':qualification['native_budget']}
                report['qualified']=True
                retained,_=retained_prefix(pinned(PREFIX_REPORT),pinned(PREFIX_PLANS))
                historical=pinned(REFERENCE_REPORT);reference=pinned(REFERENCE_PLANS)['FOCUSED_PREPARATION']
                cfg=pinned(IL+'/configuration.json');protocol=cfg['learning_protocol'];limits=cfg['admission_limits']
                release={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'root_source_index_sha256':index_sha,
                    'actual_caps':caps,'execution_released':True,
                    'scope':'Original source loading only; explicit qualified local MaskablePPO training declaration'}
                sink.write(output/'world-admission.json',release)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=release,released_sha256=sha(output/'world-admission.json'),progress=progress)
                contexts={pinned(IL+'/'+s+'-context.json')['patient_group']:semantic(pinned(IL+'/'+s+'-context.json')) for s in TRAIN}
                pin=CHECKPOINTS['IL64'];report['checkpoint_loads']+=1
                original,metadata=load_cohort_checkpoint(ROOT/pin['path'],expected_sha256=pin['sha256'],
                    expected_learning_protocol=protocol,expected_context_hashes=contexts,expected_method='IL')
                if parameter_hash(original)!=pin['parameters']:raise ValueError('Original IL64 parameter binding changed')
                original.eval();original.requires_grad_(False)
                report['initial_model']={'checkpoint':pin,'architecture':original.architecture_record(),'ppo_policy_version':VERSION}
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,check);dest=output/'source';dest.mkdir()
            def campaign(base,original_context):
                with at('fresh_prefix_and_genuine_curriculum_reconstruction'):
                    auth=base.fresh();focused,join=enabled_task(base);nominal=focused.planning_clone()
                    prefix,frontier,joins=map_prefix(auth,nominal,retained['actions'][:-1],retained['history'][:-1],checkpoint=check)
                    declaration=dict(config['goal'])
                    for key in ('cell','entry_mm','tip_mm'):declaration[key]=tuple(declaration[key])
                    context=admit_local_context(auth,original_context,nominal,prefix,frontier,GoalSpec(**declaration))
                    report['local_context']={'record':context.record,'seal':context.seal}
                    old_context=pilot['local_context'];current_context=clean(context.record)
                    if semantic(old_context['record'])!=old_context['seal']:
                        raise ValueError('Pilot context seal differs')
                    provenance=('original_context_hash','source_prefix_validation')
                    exact({k:v for k,v in old_context['record'].items() if k not in provenance},
                          {k:v for k,v in current_context.items() if k not in provenance},
                          'same physical local task across independently admitted runtime releases')
                    report['context_continuity']={'old_seal':old_context['seal'],'new_seal':context.seal,
                        'all_other_fields_exact':True,'independently_reauthenticated':True,
                        'changed_runtime_provenance':{k:{'pilot':old_context['record'][k],
                            'continuation':current_context[k]} for k in provenance}}
                    report['physical_model_join']=join;report['prefix_frontier']=frontier
                    report['prefix_joins']=joins
                    snapshots={12:nominal.clone()};episode=LocalEpisode(context,nominal,guard=check)
                    for absolute,old in enumerate(reference['selected_action_records'][12:16],13):
                        cat=episode.observe();matches=[i for i,p in enumerate(cat.packets)
                            if semantic(p['physical_action'])==semantic(old['physical_action'])]
                        if len(matches)!=1:raise ValueError('Qualified physical curriculum action missing or ambiguous')
                        _,done,_=episode.step(matches[0])
                        if done:raise ValueError('Curriculum snapshot reached goal early')
                        row=compact_history([episode.work._history[-1]])[0]
                        expected=historical['exploration']['history'][absolute-1]
                        for key in ('tool_id','entry_mm','tip_mm','target_removed_mm3','normal_removed_mm3','reward',
                                    'removed_indices_hash','contact_indices_hash','microsteps'):
                            exact(row[key],expected[key],'genuine curriculum '+key)
                        snapshots[absolute]=episode.last_clean_snapshot.clone()
                    if set(snapshots)!={12,13,14,15,16}:raise ValueError('Snapshot denominator changed')
                    report['snapshots']={str(k):{'native_state_hash':v._engine.state_hash,'history_hash':semantic(v._history)}
                                         for k,v in snapshots.items()}
                    exact(report['snapshots'],pilot['snapshots'],'same five qualified physical snapshot states')
                    exact(frontier,pilot['prefix_frontier'],'same authenticated12-step frontier')
                    del episode,cat
                for seed in config['seeds']:
                    for arm in config['arms']:
                        name=arm+'-seed'+str(seed);arm_dir=output/name;arm_dir.mkdir()
                        episode_number=[0]
                        def episodes(value):
                            episode_number[0]+=1
                            save_gzip(name+'/episode-%04d.json.gz'%episode_number[0],value)
                        def curves(value):
                            write_json(arm_dir/'curve.json',clean(value),4*1024**2)
                        def checkpoint(model,step):
                            check();file=arm_dir/('ppo-%04d.zip'%step)
                            if file.exists():raise FileExistsError('Immutable checkpoint endpoint already exists')
                            raw,binding=checkpoint_bytes(model,compressed_cap=caps['checkpoint_bytes'])
                            sink.write_bytes(file,raw);sink.check()
                            return {'path':str(file.relative_to(output)),**binding,'decisions':model.num_timesteps}
                        def evaluate(model,label):
                            tag=name+'-eval-'+label;env=LocalGoalEnv(context,{12:snapshots[12]},eval_mode=True,guard=check)
                            try:
                                with at(tag+'.nominal'):
                                    obs,reset=env.reset();done=False;choices=[]
                                    while not done:
                                        check();cat=env.episode.observe()
                                        action,_=model.predict(obs,deterministic=True,action_masks=env.action_masks())
                                        index_action=int(action)
                                        choices.append({'absolute_step':env.episode.work._steps,
                                            'catalog_digest':cat._seal,'observation_hash':cat.observation.fingerprint,
                                            'action_index':index_action,'action_id':cat.identifiers[index_action]})
                                        obs,reward,done,truncated,info=env.step(index_action)
                                        if truncated:raise ValueError('Resource truncation cannot be a completed evaluation')
                                    packets=[*prefix,*env.episode.packets];history=list(env.episode.work._history)
                                    plan={'complete':True,'context_seal':context.seal,'packets':packets,
                                        'packets_hash':semantic(packets),'history_hash':semantic(history),
                                        'local_rows':env.episode.rows,'return':env.episode.total_reward,
                                        'success':env.episode.success,'choices':choices}
                                    plan['seal']=semantic(plan);sealed=save_gzip(tag+'-sealed-plan.json.gz',plan)
                                with at(tag+'.fresh_replay_and_independent_geometry'):
                                    replay=focused.fresh()
                                    for i,packet in enumerate(packets):
                                        check();transition=replay.step_record(packet)
                                        exact(physical(transition.info),physical(history[i]),'full PPO replay')
                                    audit=evaluate_native_spatial_episode(replay,cancelled=lambda:(check() or False))
                                    if not audit['accepted']:raise ValueError('Independent evaluation geometry rejected')
                                    succeeded=False;total=0.;prior=replay._history[11]['tool_id'];rewards=[]
                                    for row in replay._history[12:]:
                                        new=(context.goal.cell in {tuple(c) for c in row.get('removed_indices_native',())} and not succeeded)
                                        value=0. if row['action_id']=='STOP' else ((10. if new else 0.)-.03
                                            -.001*row['complete_tool_path_length_mm']-.2*row['normal_removed_mm3']
                                            -.03*int(prior is not None and row['tool_id']!=prior))
                                        total+=value;rewards.append(value);succeeded=succeeded or new
                                        if row['action_id']!='STOP':prior=row['tool_id']
                                    if succeeded!=plan['success'] or abs(total-plan['return'])>1e-9:
                                        raise ValueError('Independent local goal/reward replay differs')
                                    result={'complete':True,'success':succeeded,'return':total,'rewards':rewards,
                                        'decision_count':len(choices),'full_history_rows':len(history),'sealed_plan':sealed,
                                        'plan_seal':plan['seal'],'audit':audit,'local_reward_audit_accepted':True,
                                        'history_hash':semantic(replay._history),
                                        'history':save_gzip(tag+'-history.json.gz',replay._history),
                                        'catalog_costs':env.episode.costs}
                                    report['evaluations'][tag]=result;save();return result
                            finally:env.close()
                        with at(name+'.PPO_campaign'):
                            try:
                                pin=index['checkpoints']['REVERSE512'];report['checkpoint_loads']+=1
                                raw=read_bytes(ROOT/pin['path'],pin['sha256'])
                                if len(raw)!=pin['bytes'] or len(raw)>caps['checkpoint_bytes']:
                                    raise ValueError('Exact bounded reverse512 archive required')
                                result=run_recovery_arm(original,context,snapshots,method=arm,seed=seed,
                                    milestones=config['milestones'],guard=check,progress=progress,evaluate=evaluate,
                                    save_checkpoint=checkpoint,episode_sink=episodes,curve_sink=curves,
                                    resume={'arm':curve,'raw':raw,'checkpoint':pin,
                                        'report_binding':report['inherited_campaign'],
                                        'prior_output_directory':config['failed_output_directory']})
                                del raw
                                gc.collect()  # Release old rollout/policy hook cycles before the next arm.
                            except BaseException as error:
                                report['arms'][name]=getattr(error,'ppo_partial_result',{'status':'failed_or_incomplete'})
                                raise
                            report['arms'][name]=result;curves(result);save()
                recovered=report['arms']['reverse-seed0']
                result_endpoint=recovered['checkpoints'][0]['evaluation']
                retained_endpoint=report['retained_uniform1024']['evaluation']
                report['comparison']={'uniform1024_retained':True,'reverse1024_restarted':True,
                    'uniform_success':retained_endpoint['success'],'reverse_success':result_endpoint['success'],
                    'uniform_return':retained_endpoint['return'],'reverse_return':result_endpoint['return'],
                    'caveat':report['recovery_admission']['comparison_caveat']}
                lineage=recovered['experience'];new=recovered['segment_experience']
                uniform=failed['arms']['uniform-seed0']['experience']
                historical=pilot['arms']['trainable_encoder-seed0']['experience']
                report['comparison_experience']={'patients':1,'recovered_reverse_lineage_decisions':1024,
                    'new_recovery_decisions':512,'retained_uniform_lineage_decisions':1024,
                    'shared_frozen256_counted_once':True,
                    'paid_PPO_decisions_lower_bound':256+1024+1024-256+320,
                    'discarded_after_checkpoint_decisions_lower_bound':320,
                    'discarded_postcheckpoint_decisions_not_in_lineage':True,
                    'source_state_hashes_known_union':sorted(set(lineage['source_state_hashes'])|
                        set(uniform['source_state_hashes'])|set(historical['source_state_hashes'])),
                    'state_physical_action_hashes_known_union':sorted(set(lineage['state_physical_action_hashes'])|
                        set(uniform['state_physical_action_hashes'])|set(historical['state_physical_action_hashes'])),
                    'discarded_tail_unique_union_not_reconstructed':True,
                    'new_segment_distinct_states':new['distinct_source_states'],
                    'new_segment_distinct_state_actions':new['distinct_state_physical_actions']}
                snapshots.clear()
                return {'subject':'ReMIND-045','patients':1,'new_datasets':0,'source_visits':1,
                    'arms':list(report['arms']),'reference_not_actor_input':True}
            visits.run('ReMIND-045',dest,'local_goal_MaskablePPO',campaign)
            report['source_visits']=visits.completed
            budget.complete(history_complete=True);report['status']='completed'
    except BaseException as error:
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,'message':str(error)})
        traceback.print_exc()
    finally:
        report['native_budget']=budget.snapshot();report['costs']=costs.rows;report['native_transition_counts']=transitions
        report['worker_seconds']=time.perf_counter()-started;report['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if 'recovery_admission' in report:
            old=pilot['cost_segments'];last=failed_parent['last_saved_costs']
            report['cost_segments']={'scope':'Exact new worker plus paid failed parent; failed terminal native totals are lower bounds, not invented final counts',
                'retained_pilot_and256_segments':old,'failed_parent_seconds':failed_parent['elapsed_seconds'],
                'failed_saved_worker_seconds_lower_bound':last['seconds'],
                'failed_previews_lower_bound':last['native_budget']['native_preview_entries'],
                'failed_transitions_lower_bound':last['transitions']['attempted'],
                'new_recovery_worker_seconds':report['worker_seconds'],
                'new_recovery_previews':report['native_budget']['native_preview_entries'],
                'new_recovery_transitions':dict(transitions),
                'cumulative_worker_seconds_lower_bound':old['cumulative_worker_seconds']+last['seconds']+report['worker_seconds'],
                'cumulative_previews_lower_bound':old['cumulative_native_preview_entries']+last['native_budget']['native_preview_entries']+
                    report['native_budget']['native_preview_entries'] if report['native_budget']['counting_reliable'] else None,
                'new_counting_reliable':report['native_budget']['counting_reliable'],
                'all_failed_work_charged_separately_from_restored_checkpoint_lineage':True}
        try:guard(ROOT,index);report['source_unchanged']=True
        except BaseException as error:report['status']='failed_or_incomplete';report['source_error']=str(error)
        save()
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-index-sha256',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();started=time.perf_counter()
    index=read_json(HERE/'source-index.json',args.source_index_sha256);guard(ROOT,index)
    config=read_json(HERE/'config.json',index['files'][str((HERE/'config.json').relative_to(ROOT))])
    output=Path(args.output).resolve()
    if not index['execution_released'] or output!=ROOT/config['output_directory'] or not output.is_dir():
        raise ValueError('Root-released fixed training output required')
    if not sys.dont_write_bytecode or not sys.pycache_prefix or Path(sys.pycache_prefix).exists():
        raise ValueError('Isolated no-bytecode bootstrap required')
    lease=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    threading.Thread(target=lambda:(os.read(lease,1),os._exit(72)),daemon=True).start()
    def deadline(*a):raise TimeoutError('PPO worker deadline')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,config['caps']['work_seconds'])
    import importlib.util
    spec=importlib.util.spec_from_file_location('pinned_local_qualification_worker',STAGE/'worker.py')
    qualification_worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualification_worker)
    # The reused guard additionally needs the explicitly pinned saved IL weights.
    boundary_index={**index,'input_metadata':{**index['input_metadata'],
        **{p['path']:p['sha256'] for p in index['checkpoints'].values()}}}
    qualification_worker.install_boundary(boundary_index,output)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=execute(index,args.source_index_sha256,config,output,started)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'status':result['status'],'worker_seconds':result['worker_seconds']}))
    return 0 if result['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
