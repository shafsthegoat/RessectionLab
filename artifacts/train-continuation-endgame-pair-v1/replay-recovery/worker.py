"""Replay-only recovery of two exact sealed endgame plans; no fitted model load."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import (ROOT,IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,canonical,semantic,read_bytes,read_json,
    sha,guard,tree_bytes,write_json,runtime_roots,require_baseline_route,join_planning_replay,
    PAIR_SUBJECT,TAIL_RUN,TAIL_PINS,diagnose_pair)
from recovery_inputs import FAILED,OLD_INDEX,OLD_INDEX_SHA,PINS,WORLD_KEYS,FAILURE_AUDIT,FAILURE_AUDIT_SHA,require_saved_plans,exact_json_equal

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


def compact_history(history):
    """Exact action list permits fresh replay; original full histories stay in memory."""
    fields=('action_id','reward','target_removed_mm3','normal_removed_mm3','tool_id','entry_mm','tip_mm',
            'physical_start_mm','complete_tool_path_length_mm','interaction_mode','outcome_scope')
    result=[]
    for row in history:
        value={k:row[k] for k in fields if k in row}
        value.update(history_row_hash=semantic(row), microsteps=len(row.get('microsteps',())),
            removed_indices_hash=semantic(row.get('removed_indices_native',[])),
            contact_indices_hash=semantic(row.get('contact_indices_native',[])))
        result.append(value)
    return result



def execute(index,index_sha,config,output,started):
    from resectionlab.core import thaw_json,semantic_digest
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_preflight import _CallCosts
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.planning_budget import PlanningBudget
    caps=config['caps'];end=started+caps['work_seconds'];sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,0);phase=['metadata']
    report={'schema':config['schema'],'status':'running','config':config,'source_index_sha256':index_sha,
        'source_unchanged':False,'branches':{},'checkpoint_loads':0,'cache_loads':0,'head_optimizer_updates':0,
        'new_planning_calls':0,'new_training_rollouts':0,'closed_roles_opened':False,
        'sealed_plans_admitted':False,'baseline_reproduced':False}
    def checkpoint():
        if time.perf_counter()>=end:raise TimeoutError('End-to-end replay recovery limit')
        budget.check();sink.check()
        if tree_bytes(HERE.parent)>caps['whole_build_bytes']-65536:raise InterruptedError('Whole stage cap')
    def cancel():checkpoint();return False
    def save_report():write_json(output/'report.json',thaw_json(report))
    def progress(label,**details):
        checkpoint();write_json(output/'progress.json',{'phase':label,'elapsed_seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'frozen_forwards':costs.forwards,**details},65536)
    @contextmanager
    def at(label):
        phase[0]=label;progress(label)
        with costs.scope(label):yield
        progress(label+'.complete')
    def pinned(path):return read_json(ROOT/path,index['input_metadata'][path])
    def readout(base,context,actions,name):
        replay=base.fresh();context.require_task(replay);rows=[]
        for ordinal,action in enumerate(actions):
            checkpoint();obs=replay.observation();context.require_observations((obs,))
            if action not in obs.action_ids or not obs.action_mask[obs.action_ids.index(action)]:
                raise ValueError('Sealed action absent/masked on fresh replay')
            source=replay._engine.state_hash;step=replay.step(action)
            rows.append({'observation_hash':obs.fingerprint,'action_id':action,'reward':step.reward,
                'terminated':step.terminated,'source_state_hash':source,'result_state_hash':replay._engine.state_hash})
            if step.terminated and ordinal!=len(actions)-1:raise ValueError('Actions after termination')
        metrics=replay.metrics();history=metrics['history']
        audit=evaluate_native_spatial_episode(replay,cancelled=cancel)
        nominal=replay.planning_clone();nominal_hash=semantic_digest(nominal.metrics()['history']);del nominal
        return {'complete':replay.terminated,'method':name,'actions':list(actions),'decisions':rows,
            'nominal_history_hash':nominal_hash,'world':{k:context.record()[k] for k in WORLD_KEYS},
            'context_hash':context.fingerprint,'history_hash':semantic_digest(history),
            'history':compact_history(history),'audit':audit,'terminal_reason':'STOP' if actions[-1]=='STOP' else 'HORIZON'}
    try:
        with budget,costs,_bounded_writes(sink):
            with at('saved_plan_metadata_admission'):
                for name,digest in index['input_metadata'].items():read_bytes(ROOT/name,digest)
                failed=pinned(FAILED+'/report.json');parent=pinned(FAILED+'/launcher.json')
                plans=pinned(FAILED+'/plans.json');retained=pinned(TAIL_RUN+'/report.json');old=pinned(OLD_INDEX)
                admission=require_saved_plans(failed,parent,plans,retained,old)
                report['sealed_plans_admitted']=True;report['plan_admission']=admission
                report['independent_recovery_point']={'path':FAILURE_AUDIT,'sha256':FAILURE_AUDIT_SHA}
                report['failed_attempt']={'directory':FAILED,'pins':PINS,'source_index_sha256':OLD_INDEX_SHA,
                    'status':failed['status'],'failure':failed['failure'],'worker_seconds':failed['worker_seconds'],
                    'parent_seconds':parent['elapsed_seconds'],'native_previews':failed['native_budget']['native_preview_entries'],
                    'frozen_policy_forwards':failed['total_frozen_policy_forwards'],'head_optimizer_updates':failed['head_optimizer_updates'],
                    'calibration_calls':failed['calibration_calls_completed'],'tail_calls':failed['tail_calls_completed'],
                    'original_failure_preserved':True,'prior_cumulative_costs':failed['cumulative_iteration_costs']}
                sink.write(output/'plan-admission.json',admission)
                retained_arm=retained['cases'][PAIR_SUBJECT]['TAIL']
                report['treatment']=failed['treatment']
                original=pinned(IL+'/configuration.json');protocol=original['learning_protocol'];limits=original['admission_limits']
                world_release={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'actual_caps':caps,'root_source_index_sha256':index_sha,
                    'execution_released':True,'scope':'same_world_replay_only_no_learning'}
                sink.write(output/'world-admission.json',world_release)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=world_release,released_sha256=sha(output/'world-admission.json'),progress=progress)
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,checkpoint)
            dest=output/'evaluation';dest.mkdir()
            def evaluate(base,context):
                exact_json_equal({k:context.record()[k] for k in WORLD_KEYS},retained_arm['world'],'replay world')
                for method in config['methods']:
                    plan=plans[method]
                    if semantic({k:v for k,v in plan.items() if k!='plan_seal'})!=plan['plan_seal']:
                        raise ValueError('Previously admitted sealed plan changed')
                    with at('replay.'+method):
                        result=readout(base,context,plan['actions'],method)
                        result['plan_seal']=plan['plan_seal']
                        # Preserve concrete certificate/projection BEFORE parity assertions.
                        sink.write(dest/(method+'-evidence.json'),thaw_json(result))
                        report['branches'][method]=thaw_json(result);save_report()
                        if not result['complete'] or result['audit']['accepted'] is not True:
                            raise ValueError('Independent complete replay rejected')
                        result['planning_replay_join']=join_planning_replay(plan['planning'],result,method='TAIL')
                        if method=='RETAINED_TAIL':
                            result['original_route_reproduced']=require_baseline_route(result,retained_arm)
                            exact_json_equal(result['decisions'],retained_arm['decisions'],'retained all decisions')
                            exact_json_equal(result['history'],retained_arm['history'],'retained all compact history fields')
                    result['replay_costs']=dict(costs.rows['replay.'+method])
                    report['branches'][method]=thaw_json(result);save_report()
                return {'subject':PAIR_SUBJECT,'complete_branches':2,'new_planning_calls':0}
            visits.run(PAIR_SUBJECT,dest,'same045_source',evaluate)
            report['source_visits']=visits.completed
            report['baseline_reproduced']=report['branches']['RETAINED_TAIL']['original_route_reproduced']
            report['diagnosis']=diagnose_pair(report['branches'],failed['treatment']['common_state'])
            if not report['diagnosis']['comparison_admissible']:raise ValueError('Both complete branches required')
            report['status']='completed';save_report();budget.complete(history_complete=True)
    except BaseException as error:
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,
            'message':str(error),'committed':bool(getattr(error,'committed',False))})
        traceback.print_exc()
    finally:
        report['native_budget']=budget.snapshot();report['costs']=costs.rows
        report['total_frozen_policy_forwards']=costs.forwards
        report['worker_seconds']=time.perf_counter()-started
        report['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost_scope']='New045 source admission and two fresh independent full replays only. Original full fitting/planning plus first replay failure remain paid separately; no source/checkpoint/cache/model copies or decode.'
        if 'failed_attempt' in report:
            oldcost=report['failed_attempt']
            report['pair_plus_recovery_costs']={'native_previews':oldcost['native_previews']+report['native_budget']['native_preview_entries'],
                'frozen_policy_forwards':oldcost['frozen_policy_forwards']+costs.forwards,
                'worker_seconds':oldcost['worker_seconds']+report['worker_seconds'],
                'original_fitting_and_replay_failure_cost_not_erased':True,'earlier_iterations':oldcost['prior_cumulative_costs']}
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
    def deadline(*unused):raise TimeoutError('End-to-end replay-only recovery cap')
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
