"""Cache-only final-readout recovery; prior failure and data stay immutable."""
import argparse
from contextlib import contextmanager
import gc
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT, TRAIN, IL, RL, PUBLIC, PUBLIC_SHA, COHORT, COHORT_SHA, CHECKPOINTS, canonical, semantic, read_bytes, read_json, sha, guard, tree_bytes, write_json, useful_gate, join_planning_replay, runtime_roots, ORIGINAL_RUN,ORIGINAL_REPORT_SHA,ORIGINAL_AUDIT,require_baseline_fit,require_baseline_route,CACHE_ATTEMPT,CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA,CACHE_PINS,require_cache_origin
from inputs import WORLD_KEYS, pinned, manifest


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
    """Shared by actual entrypoint and isolated no-payload bootstrap control."""
    import numpy
    import torch
    from resectionlab import (native_resection,patient_planning_cohort_visits,
        patient_planning_cohort_sequential,patient_planning_cohort_io,patient_planning_learning,
        patient_planning_preflight,native_spatial_evaluation,planning_budget,spatial_policy,
        spatial_continuation_value,spatial_continuation_planner,spatial_continuation_cache,spatial_continuation_calibration)
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


def write_logged_targets(sink,output,target,manifest_hash,targets,weights,stops,cache_file,cache_sha):
    """Publish once after authenticated cache load; OutputBudget is exclusive."""
    sink.write(output/'logged-targets.json',{'target_formula':target,'targets':targets,'weights':weights,
        'STOP_rows':stops,'manifest_hash':manifest_hash,'feature_array_not_saved':False,
        'feature_cache':cache_file,'feature_cache_sha256':cache_sha})


def execute(index, index_sha, config, output, started):
    import numpy as np
    import torch
    from resectionlab.core import thaw_json, semantic_digest, array_digest
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget,load_cohort_checkpoint
    from resectionlab.patient_planning_preflight import _CallCosts,_history_identity
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.spatial_continuation_value import ContinuationFitConfig,fit_continuation_value
    from resectionlab.spatial_continuation_planner import q_guided_nominal_search
    from resectionlab.spatial_continuation_cache import load_continuation_cache,current_feature_source_hashes
    from resectionlab.spatial_continuation_calibration import calibrate_continuation_readout

    caps=config['caps']; end=started+caps['work_seconds']
    sink=OutputBudget(output,caps['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=caps['native_previews'],seconds=max(.01,end-time.perf_counter()))
    costs=_CallCosts(budget,caps['frozen_policy_forwards'])
    report={'schema':config['schema'],'status':'running','source_index_sha256':index_sha,'config':config,
        'source_unchanged':False,'closed_roles_opened':False,'new_training_rollouts':0,
        'original_collection_actor_reexecuted':False,'base_optimizer_updates':0,
        'head_optimizer_updates':0,'reconstructed_traces':[],'cases':{},'checkpoint_loads':0,
        'calibration_calls_attempted':0,'calibration_calls_completed':0,'baseline_reproduced':False,
        'historical_IL_failure_preserved':True,'upfront_costs_excluded_from_gain':False}
    models={};heads={};fitted=calibrated=corpus=None;phase=['metadata']
    def checkpoint():
        if time.perf_counter()>=end: raise TimeoutError('End-to-end worker limit')
        budget.check(); sink.check()
        if tree_bytes(HERE.parent)>caps['whole_build_bytes']-65536: raise InterruptedError('Whole stage cap')
    def cancel(): checkpoint(); return False
    def progress(label,**details):
        checkpoint()
        write_json(output/'progress.json',{'phase':label,'elapsed_seconds':time.perf_counter()-started,
            'native_budget':budget.snapshot(),'frozen_forwards':costs.forwards,
            'head_optimizer_updates':report['head_optimizer_updates'],'reconstructed_traces':len(report['reconstructed_traces']),**details},65536)
    def save_report(): write_json(output/'report.json',thaw_json(report))
    def directory(*parts):
        path=output.joinpath(*parts);path.mkdir(parents=True,exist_ok=False);return path
    @contextmanager
    def at(label):
        phase[0]=label;progress(label)
        with costs.scope(label): yield
        progress(label+'.complete')
    def audit_actual(task):
        audit=evaluate_native_spatial_episode(task,cancelled=cancel)
        if audit['accepted'] is not True: raise ValueError('Independent committed geometry/outcomes rejected')
        return audit
    def readout(task,context,actions,label,expected_history=None):
        replay=task.fresh();context.require_task(replay); rows=[]
        for ordinal,action in enumerate(actions):
            checkpoint();obs=replay.observation();context.require_observations((obs,))
            if action not in obs.action_ids or not obs.action_mask[obs.action_ids.index(action)]:
                raise ValueError('Selected action absent/masked on fresh replay')
            source_state=replay._engine.state_hash
            step=replay.step(action)
            rows.append({'observation_hash':obs.fingerprint,'action_id':action,'reward':step.reward,'terminated':step.terminated,
                         'source_state_hash':source_state,'result_state_hash':replay._engine.state_hash})
            if step.terminated and ordinal!=len(actions)-1: raise ValueError('Actions after termination')
        if not replay.terminated: raise ValueError('Incomplete selected route')
        metrics=replay.metrics();history=metrics['history']
        if expected_history is not None and _history_identity(history)!=_history_identity(expected_history):
            raise ValueError('Fresh route history differs from sealed planning route')
        audit=audit_actual(replay)
        nominal=replay.planning_clone()
        nominal_hash=semantic_digest(nominal.metrics()['history']);del nominal
        value={'complete':True,'method':label,'actions':list(actions),'decisions':rows,'nominal_history_hash':nominal_hash,
            'world':{k:context.record()[k] for k in WORLD_KEYS},'context_hash':context.fingerprint,
            'history_hash':semantic_digest(history),'history':compact_history(history),'audit':audit,
            'terminal_reason':'STOP' if actions[-1]=='STOP' else 'HORIZON'}
        return value
    try:
        with budget,costs,_bounded_writes(sink):
            with at('metadata_authentication'):
                for path,digest in index['input_metadata'].items():read_bytes(ROOT/path,digest)
                rl_result=pinned(index,RL+'/result.json');il_result=pinned(index,IL+'/result.json')
                receipt=pinned(index,RL+'.supervision/receipt.json')
                recovery=pinned(index,'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01/result.json')
                if (rl_result['method']!='RL' or rl_result['optimizer_updates']['RL']!=8
                        or rl_result['private_reference_reads']!=0 or rl_result['SELECT_EVAL_opened'] is not False
                        or il_result['optimizer_updates']['IL']!=64): raise ValueError('Historical fixed endpoints differ')
                original_run=pinned(index,ORIGINAL_RUN+'/report.json')
                original_parent=pinned(index,ORIGINAL_RUN+'/launcher.json')
                original_audit=pinned(index,ORIGINAL_AUDIT)
                if (original_run['status']!='completed' or original_parent['status']!='completed'
                        or original_parent['report_sha256']!=ORIGINAL_REPORT_SHA
                        or original_run['useful_criterion']['useful_gain'] is not False
                        or original_run['head_optimizer_updates']!=64):raise ValueError('Accepted original negative run required')
                report['original_negative']={'report_sha256':ORIGINAL_REPORT_SHA,'audit_sha256':index['input_metadata'][ORIGINAL_AUDIT],
                    'original_upfront_costs_not_erased':True,'worker_seconds':original_run['worker_seconds'],
                    'native_previews':original_run['native_budget']['native_preview_entries'],
                    'retained_IL64_RL8_comparators':{s:{m:original_run['cases'][s][m]['audit']['outcomes'] for m in ('IL64','RL8')} for s in TRAIN}}
                prior=pinned(index,CACHE_ATTEMPT+'/report.json');prior_parent=pinned(index,CACHE_ATTEMPT+'/launcher.json')
                binding=pinned(index,CACHE_ATTEMPT+'/cache-binding.json');cache_receipt=pinned(index,CACHE_ATTEMPT+'/cache-receipt.json')
                evidence=pinned(index,CACHE_ATTEMPT+'/reconstruction-evidence.json');cache_index=pinned(index,CACHE_SOURCE_INDEX)
                require_cache_origin(prior,prior_parent,binding,cache_receipt,evidence,cache_index)
                guard(ROOT,cache_index)
                report['prior_cache_attempt']={'directory':CACHE_ATTEMPT,'pins':CACHE_PINS,'source_index_sha256':CACHE_SOURCE_SHA,
                    'status':prior['status'],'failure':prior['failure'],'worker_seconds':prior['worker_seconds'],
                    'parent_seconds':prior_parent['elapsed_seconds'],'sampled_peak_group_rss_bytes':prior_parent['sampled_peak_group_rss_bytes'],
                    'native_previews':prior['native_budget']['native_preview_entries'],'frozen_policy_forwards':prior['total_frozen_policy_forwards'],
                    'reconstruction_costs':binding['origin']['original_reconstruction_costs'],
                    'head_optimizer_updates':0,'calibration_calls':0,'routes':0,'failure_preserved':True}
                report.update(reconstruction_executed_this_attempt=False,cache_copied=False,
                    reconstructed_traces=evidence['reconstructed_traces'])
                declared=manifest(index);manifest_hash=semantic(declared)
                if manifest_hash!=original_run['trace_manifest_hash']:raise ValueError('Fixed original trace manifest changed')
                sink.write(output/'trace-manifest.json',declared)
                report.update(trace_manifest_hash=manifest_hash,historical_RL_status=rl_result['status'],
                    historical_RL_supervision_status=receipt['status'],historical_IL_status=il_result['status'],
                    historical_IL045_recovery_status=recovery['status'])
                original=pinned(index,IL+'/configuration.json');protocol=original['learning_protocol'];limits=original['admission_limits']
                admission={'schema':config['schema'],'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
                    'learning_protocol_hash':semantic(protocol),'new_objective':config['target'],
                    'actual_caps':caps,'root_source_index_sha256':index_sha,'execution_released':True}
                sink.write(output/'world-admission.json',admission)
                factories=make_train_visit_factories(manifest_index_path=ROOT/PUBLIC,manifest_index_sha256=PUBLIC_SHA,
                    cohort_bytes=read_bytes(ROOT/COHORT,COHORT_SHA),learning_protocol=protocol,limits=limits,
                    released_record=admission,released_sha256=sha(output/'world-admission.json'),progress=progress)
            with at('load_fixed_IL64_parameter_only_checkpoint'):
                for method,base in (('IL64',IL),):
                    original_cfg=pinned(index,base+'/configuration.json')
                    contexts={pinned(index,base+'/'+s+'-context.json')['patient_group']:semantic(pinned(index,base+'/'+s+'-context.json')) for s in TRAIN}
                    pin=CHECKPOINTS[method]
                    policy,meta=load_cohort_checkpoint(ROOT/pin['path'],expected_sha256=pin['sha256'],
                        expected_learning_protocol=original_cfg['learning_protocol'],expected_context_hashes=contexts,
                        expected_method='IL' if method=='IL64' else 'RL')
                    if parameter_hash(policy)!=pin['parameters']:raise ValueError('Fixed parameter identity differs')
                    policy.eval();policy.requires_grad_(False);models[method]=policy;report['checkpoint_loads']+=1
            visits=_SequentialVisits(factories,protocol,limits,sink,costs,checkpoint)
            with at('authenticated_existing_cache_load'):
                feature_sources=current_feature_source_hashes()
                if feature_sources!=binding['origin']['feature_source_sha256']:raise ValueError('Original feature producers changed')
                corpus,loaded=load_continuation_cache(ROOT/CACHE_ATTEMPT/'authenticated-features.zip',models['IL64'],
                    expected_cache_sha256=CACHE_PINS['authenticated-features.zip'],expected_binding=binding)
                if (corpus.seal!=evidence['corpus_seal'] or corpus.metadata['manifest_hash']!=manifest_hash
                        or thaw_json(corpus.metadata['manifest'])!=declared or corpus.metadata['states']!=497
                        or len(corpus.metadata['trace_bindings'])!=36):raise ValueError('Exact original36/497 corpus changed')
                report['corpus']={k:thaw_json(v) for k,v in corpus.metadata.items() if k!='manifest'}
                report['corpus_seal']=corpus.seal
                old_targets=pinned(index,ORIGINAL_RUN+'/logged-targets.json')
                if (corpus.targets.tolist()!=old_targets['targets'] or corpus.weights.tolist()!=old_targets['weights']
                        or corpus.stop_rows.tolist()!=old_targets['STOP_rows']):raise ValueError('Original labels/weights/STOP rows changed')
                report['cache']={'file':CACHE_ATTEMPT+'/authenticated-features.zip',
                    'sha256':CACHE_PINS['authenticated-features.zip'],'bytes':cache_receipt['bytes'],
                    'binding_file':CACHE_ATTEMPT+'/cache-binding.json','binding_sha256':CACHE_PINS['cache-binding.json'],
                    'evidence_file':CACHE_ATTEMPT+'/reconstruction-evidence.json',
                    'evidence_sha256':CACHE_PINS['reconstruction-evidence.json'],'load_costs':thaw_json(loaded),
                    'reconstruction_paid_once':True,'pixels_or_base_weights_copied':False,'copied_this_attempt':False}
                write_logged_targets(sink,output,config['target'],manifest_hash,corpus.targets.tolist(),
                    corpus.weights.tolist(),corpus.stop_rows.tolist(),report['cache']['file'],report['cache']['sha256'])
                del evidence,binding,cache_receipt,loaded,cache_index,prior
            def head_case_mse(head):
                with torch.no_grad():
                    x=torch.tensor(np.array(corpus.features,copy=True)); p=head(x).squeeze(-1).numpy()
                p=np.where(corpus.stop_rows,0.,p);residual=(p.astype(np.float64)-corpus.targets.astype(np.float64))**2
                rows={};offset=0
                for subject in TRAIN:
                    count=sum(len(r['decisions']) for r in declared if r['world']['subject']==subject)
                    weights=corpus.weights[offset:offset+count].astype(np.float64)
                    rows[subject]=float((residual[offset:offset+count]*weights).sum()/weights.sum());offset+=count
                if offset!=497:raise ValueError('Case readout partition')
                return rows
            def save_head(label,scorer):
                entries=[];payloads=[]
                for i,(name,tensor) in enumerate(sorted(scorer._head.state_dict().items())):
                    stream=io.BytesIO();np.lib.format.write_array(stream,tensor.detach().numpy(),allow_pickle=False)
                    raw=stream.getvalue();member='head/%03d.npy'%i
                    import hashlib
                    entries.append({'name':name,'member':member,'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(tensor.shape)})
                    payloads.append((member,raw))
                stream=io.BytesIO()
                with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr('manifest.json',canonical({'schema':'logged-Q-head-only-v1','base':CHECKPOINTS['IL64'],
                        'head_hash':scorer.report['fitted_head_hash'],'corpus_seal':corpus.seal,'entries':entries}))
                    for name,raw in payloads:archive.writestr(name,raw)
                raw=stream.getvalue()
                if len(raw)>128*1024:raise ValueError('Head-only checkpoint cap')
                filename=label+'-head.qckpt';sink.write_bytes(output/filename,raw)
                return {'file':filename,'sha256':sha(output/filename),'bytes':len(raw),'base_weights_copied':False}
            with at('fixed_64_head_updates'):
                report['initial_case_weighted_MSE']=head_case_mse(models['IL64'].actor)
                fitted=fit_continuation_value(models['IL64'],corpus,config=ContinuationFitConfig(updates=64),
                    cancelled=cancel,max_scores=96,score_seconds=3600)
                report['head_optimizer_updates']=len(fitted.report['updates'])
                if report['head_optimizer_updates']!=64:raise ValueError('Fixed64 endpoint incomplete')
                report['head_fit']=thaw_json(fitted.report)
                report['final_case_weighted_MSE']=head_case_mse(fitted._head)
                report['extra_head_case_readout_forwards']=2
                report['baseline_head_reproduced']=require_baseline_fit(report['head_fit'],original_run['head_fit'])
                for key in ('initial_case_weighted_MSE','final_case_weighted_MSE'):
                    if report[key]!=original_run[key]:raise ValueError('Original casewise64 MSE did not reproduce')
                heads['Q64']=fitted
                report['head_checkpoints']={'Q64':save_head('Q64',fitted)}
                save_report()
            with at('single_fixed_final_readout_calibration'):
                report['calibration_calls_attempted']+=1
                calibrated=calibrate_continuation_readout(models['IL64'],corpus,fitted,
                    cancelled=cancel,max_scores=96,score_seconds=3600)
                report['calibration_calls_completed']+=1
                report['calibration']=thaw_json(calibrated.report)
                fitted.require()
                if thaw_json(fitted.report)!=report['head_fit']:raise ValueError('Original64 scorer mutated during calibration')
                heads['CALIBRATED']=calibrated
                report['head_checkpoints']['CALIBRATED']=save_head('CALIBRATED',calibrated)
                save_report()
            with at('two_logged_prediction_readouts'):
                report['logged_prediction_readouts_attempted']=0;report['logged_prediction_readouts_completed']=0
                predictions={}
                with torch.no_grad():
                    x=torch.tensor(np.array(corpus.features,copy=True))
                    for name,scorer in heads.items():
                        report['logged_prediction_readouts_attempted']+=1
                        values=scorer._head(x).squeeze(-1).numpy()
                        report['logged_prediction_readouts_completed']+=1
                        values=np.where(corpus.stop_rows,0.,values)
                        if values.shape!=(497,) or not np.isfinite(values).all():raise ValueError('Finite497 scalar head readout required')
                        predictions[name]=values.tolist()
                    del x,values
                diagnostic={'schema':'same497-logged-action-predictions-v1','corpus_seal':corpus.seal,
                    'manifest_hash':manifest_hash,'labels_file':'logged-targets.json','labels_sha256':sha(output/'logged-targets.json'),
                    'head_hashes':{name:scorer.report['fitted_head_hash'] for name,scorer in heads.items()},
                    'predictions':predictions,'STOP_prediction':0,'head_forwards':2,'encoder_forwards':0,'native_calls':0}
                raw=canonical(diagnostic)+b'\n'
                if len(raw)>65536:raise ValueError('Scalar readout64KiB cap')
                sink.write_bytes(output/'logged-predictions.json',raw)
                report['logged_predictions']={'file':'logged-predictions.json','sha256':sha(output/'logged-predictions.json'),
                    'bytes':len(raw),'head_forwards':2,'encoder_forwards':0,'native_calls':0}
                del predictions,diagnostic,raw,corpus;corpus=None;save_report()
            for subject in TRAIN:
                dest=directory('evaluation',subject)
                def evaluate(base,context):
                    report['cases'][subject]={}
                    for method in config['methods']:
                        label='evaluation.'+subject+'.'+method;expected=None
                        report['cases'][subject][method]={'complete':False,'status':'started'};save_report()
                        with at(label+'.planning'):
                            if method=='GREEDY':actions,accounting=base.observed_greedy_search(seconds=config['search_seconds'])
                            else:
                                actions,accounting=q_guided_nominal_search(base,context,heads[method],
                                    seconds=config['search_seconds'],cancelled=cancel,max_calls=24)
                            if accounting.get('complete') is not True:raise ValueError('Incomplete planner not an endpoint')
                        with at(label+'.fresh_replay_and_independent_certificate'):
                            result=readout(base,context,actions,method,expected)
                            result['planning_replay_join']=join_planning_replay(thaw_json(accounting),result,method=method)
                        result.update(planning=thaw_json(accounting),planning_costs=dict(costs.rows[label+'.planning']),
                            replay_costs=dict(costs.rows[label+'.fresh_replay_and_independent_certificate']))
                        if method in ('GREEDY','Q64'):
                            old_method='Q_GUIDED' if method=='Q64' else method
                            result['original_route_reproduced']=require_baseline_route(result,original_run['cases'][subject][old_method])
                        report['cases'][subject][method]=result
                        sink.write(dest/(method+'.json'),result);save_report()
                        del expected,result,accounting,actions
                    return {'subject':subject,'methods_completed':3}
                visits.run(subject,dest,'evaluation.'+subject,evaluate)
            for method,policy in models.items():
                if parameter_hash(policy)!=CHECKPOINTS[method]['parameters'] or any(p.grad is not None or p.requires_grad for p in policy.parameters()):
                    raise ValueError('Frozen endpoint changed')
            for scorer in heads.values():scorer.require()
            report['head_inference_costs']={key:dict(scorer.costs) for key,scorer in heads.items()}
            report['baseline_reproduced']=report['baseline_head_reproduced'] and all(
                report['cases'][s][m]['original_route_reproduced'] for s in TRAIN for m in ('GREEDY','Q64'))
            report['source_visits']=visits.completed
            report['useful_criterion']=useful_gate(report['cases'])
            if not report['useful_criterion']['comparison_admissible']:raise ValueError('Four-case comparison incomplete')
            report['status']='completed';save_report();budget.complete(history_complete=True)
    except BaseException as error:
        partial_calibration=getattr(error,'continuation_calibration_accounting',None)
        if partial_calibration is not None:report['failed_calibration_accounting']=thaw_json(partial_calibration)
        elif phase[0]=='single_fixed_final_readout_calibration' and not report['calibration_calls_completed']:
            report['failed_calibration_accounting']={'status':'unknown_no_completed_calibration_report'}
        partial_fit=getattr(error,'continuation_fit_accounting',None)
        if partial_fit is not None:
            report['failed_head_fit_accounting']=thaw_json(partial_fit)
            report['head_optimizer_updates']=partial_fit['completed_updates']
            report['head_optimizer_step_completion_unknown']=partial_fit['optimizer_step_completion_unknown']
        elif phase[0]=='fixed_64_head_updates' and report['head_optimizer_updates']!=64:
            report['head_optimizer_updates']=None
            report['failed_head_fit_accounting']={'status':'unknown_no_completed_fit_report'}
        report.update(status='failed_or_incomplete',failure={'phase':phase[0],'type':type(error).__name__,'message':str(error),
            'committed':bool(getattr(error,'committed',False)),'accounting':getattr(error,'accounting',None)})
        traceback.print_exc()
    finally:
        report['native_budget']=budget.snapshot();report['costs']=costs.rows
        report['total_frozen_policy_forwards']=costs.forwards
        report['worker_seconds']=time.perf_counter()-started
        report['self_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost_scope']='This recovery: exact prior cache load,64 head updates, one readout calibration, two logged prediction head forwards, all12 fresh GREEDY/Q64/CALIBRATED routes and independent replay. The prior cache-fill failure and all497 reconstruction/feature forwards stay paid separately. Inclusive phases overlap; do not sum them. Historical IL64/RL8 are referenced without rerun.'
        if 'prior_cache_attempt' in report:
            paid=report['prior_cache_attempt']
            report['cumulative_iteration_costs']={'failed_cache_attempt_worker_seconds':paid['worker_seconds'],
                'recovery_worker_seconds':report['worker_seconds'],
                'native_preview_entries':paid['native_previews']+report['native_budget']['native_preview_entries'],
                'frozen_policy_forwards':paid['frozen_policy_forwards']+report['total_frozen_policy_forwards'],
                'head_optimizer_updates':report['head_optimizer_updates'],
                'prior_failed_attempt_not_amortized':True}
        report['head_inference_costs']={key:dict(scorer.costs) for key,scorer in heads.items()}
        try:
            guard(ROOT,index)
            for path,digest in index['input_metadata'].items():read_bytes(ROOT/path,digest)
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
    def deadline(*unused):raise TimeoutError('End-to-end fixed contrast cap')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,config['caps']['work_seconds'])
    install_boundary(index,output);sys.path.insert(0,str(ROOT/'src'))
    import_execution_modules()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=execute(index,args.source_index_sha256,config,output,started)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'status':result['status'],'worker_seconds':result['worker_seconds'],'useful_criterion':result.get('useful_criterion')}))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
