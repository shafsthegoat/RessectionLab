"""Existing four-source replay/cache plus one frozen endpoint gradient readout."""
import argparse,json,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import (CAPS,CHECKPOINT,HELPER,OUTPUT,SUBJECTS,STEPS,canonical,
    guard as source_guard,metadata,need,sha,source_module)

def execute(release,release_sha,progress,external_io=None):
    import torch
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_cohort_sequential import _SequentialVisits,_bounded_writes
    from resectionlab.patient_planning_cohort_io import OutputBudget,load_cohort_checkpoint
    from resectionlab.patient_teacher_trace_cache import PatientTeacherTraceCache
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.public_motion_ranking import PublicMotionRankingCorpus
    owner,saved,pins=metadata()
    protocol,limits=owner.canonical_configuration('IL')
    need(canonical(protocol)==canonical(release['learning_protocol']) and
         canonical(limits)==canonical(release['cohort_limits']),'historical_checkpoint_context')
    prior=owner.inputs();corpus=owner.small(owner.LABEL_REFS['public-score-corpus.json'])
    ranking=PublicMotionRankingCorpus.admit(corpus,expected_hash=owner.RANKING_HASH)
    helper=source_module(HELPER,'bound_frozen_gradient_diagnostic')
    factories=make_train_visit_factories(manifest_index_path=ROOT/owner.PUBLIC_INDEX,
        manifest_index_sha256=owner.PUBLIC_SHA,cohort_bytes=(ROOT/owner.COHORT).read_bytes(),
        learning_protocol=protocol,limits=limits,released_record=saved['release'],
        released_sha256=pins['release']['sha256'],progress=progress)
    # The context hash includes the historical runtime-release hash. Reconstruct
    # that immutable training identity; the current owner separately enforces
    # zero updates and its tighter actual execution limits below.
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=CAPS['native_previews'],seconds=CAPS['worker_seconds'])
    costs=preflight._CallCosts(budget,CAPS['policy_forwards']);started=time.perf_counter()
    result={'status':'started','release_sha256':release_sha,'TRAIN':SUBJECTS,'SELECT_EVAL_opened':False,
        'checkpoint':CHECKPOINT,'learning_protocol_hash':semantic_digest(protocol),
        'optimizer_calls':0,'backward_calls':0,'forbidden_io_calls':0,'search_calls':0,
        'collector_replay_steps':0,'autograd_grad_calls':0,'checkpoint_loads':0,
        'parameters_unchanged':None,'gradient_buffers_unchanged':None,'requires_grad_flags_restored':None}
    cache=PatientTeacherTraceCache(protocol);traces={};policy=None;frozen_versions=None;frozen_flags=None
    def frozen_check():
        if policy is not None and frozen_versions is not None:
            need(tuple(p._version for p in policy.parameters())==frozen_versions,'parameter_write_detected')
            need(all(p.grad is None for p in policy.parameters()),'leaf_grad_write_detected')
    def check():
        budget.check();sink.check();frozen_check()
        if external_io is not None and external_io['blocked']:raise PermissionError('Prior blocked payload access')
    visits=_SequentialVisits(factories,protocol,limits,sink,costs,check)
    originals=[]
    def replace(owner,name,replacement):
        originals.append((owner,name,getattr(owner,name)));setattr(owner,name,replacement)
    def deny(counter):
        def refused(*unused,**kwargs):result[counter]+=1;raise RuntimeError('Forbidden '+counter)
        return refused
    replace(torch.optim.Optimizer,'__init__',deny('optimizer_calls'))
    replace(torch.Tensor,'backward',deny('backward_calls'))
    replace(torch.autograd,'backward',deny('backward_calls'))
    replace(torch,'save',deny('forbidden_io_calls'));replace(torch,'load',deny('forbidden_io_calls'))
    replace(NativeSpatialTask,'observed_greedy_search',deny('search_calls'))
    replace(preflight,'observed_beam_search',deny('search_calls'))
    original_grad=torch.autograd.grad;original_transition=NativeSpatialTask._transition
    def counted_grad(*args,**kwargs):
        check();need(result['autograd_grad_calls']<79,'gradient_call_cap')
        result['autograd_grad_calls']+=1
        answer=original_grad(*args,**kwargs);frozen_check();return answer
    def counted_transition(*args,**kwargs):
        check();need(result['collector_replay_steps']<58,'collector_replay_step_cap')
        result['collector_replay_steps']+=1;return original_transition(*args,**kwargs)
    replace(torch.autograd,'grad',counted_grad);replace(NativeSpatialTask,'_transition',counted_transition)
    def teacher(base,context,subject,dest):
        expected=prior['plan_'+subject]['plan']
        trace=preflight._collect(base,context,actions=expected['actions'],output=dest,guard=check)
        sealed=preflight._seal_and_replay(base,context,trace,method='SEARCH',policy=None,updates=0,output=dest,guard=check)
        actual=thaw_json(sealed['plan'])
        need(canonical({k:v for k,v in actual.items() if k not in ('context_hash','history')})==
             canonical({k:v for k,v in expected.items() if k not in ('context_hash','history')}) and
             preflight._history_identity(actual['history'])==preflight._history_identity(expected['history']),
             'exact_saved_teacher_world_and_history')
        old=saved['teacher_pins'][subject]
        need(trace.seal_hash==old['trace_seal'] and context.fingerprint==old['context_hash']
             and sealed['plan_seal']==old['plan_seal'],'exact_ranking_teacher_seals')
        cache.add_replayed(trace,sealed);traces[subject]=trace
        return {'trace_seal':trace.seal_hash,'plan_seal':sealed['plan_seal'],
            'context_hash':context.fingerprint,'steps':len(trace.transitions)}
    try:
        with budget,costs,_bounded_writes(sink):
            for subject,n in zip(SUBJECTS,STEPS):
                dest=OUTPUT/'teachers'/subject;dest.mkdir(parents=True,exist_ok=False)
                row=visits.run(subject,dest,'teacher_replay.'+subject,lambda base,ctx:teacher(base,ctx,subject,dest))
                need(row['steps']==n,'fixed_teacher_count')
            need(len(visits.completed)==4 and result['collector_replay_steps']==58,'four_replayed_sources')
            sink.write(OUTPUT/'teacher-cache.json',cache.record())
            contexts=tuple(visits.contexts[s] for s in SUBJECTS)
            with costs.scope('checkpoint_reload'):
                result['checkpoint_loads']+=1
                policy,checkpoint_metadata=load_cohort_checkpoint(ROOT/CHECKPOINT['path'],
                    expected_sha256=CHECKPOINT['sha256'],expected_learning_protocol=protocol,
                    expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},expected_method='IL')
                policy.eval();policy.requires_grad_(False)
                need(parameter_hash(policy)==CHECKPOINT['parameter_hash'],'exact_frozen_parameter_hash')
                frozen_versions=tuple(p._version for p in policy.parameters())
                frozen_flags=tuple(p.requires_grad for p in policy.parameters());frozen_check()
                sink.write(OUTPUT/'checkpoint-metadata.json',checkpoint_metadata)
            need(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),'gradient_enabled_diagnostic_scope')
            with costs.scope('frozen_gradient_readout'):
                report=helper.inspect_frozen_gradients(policy,checkpoint_metadata=checkpoint_metadata,
                    expected_parameter_hash=CHECKPOINT['parameter_hash'],cache=cache,traces=traces,ranking=ranking,
                    saved_readouts={f'{s}:{k}':saved[f'{s}:{k}'] for s,n in zip(SUBJECTS,STEPS) for k in range(n)},
                    guard=check,emit=lambda name,value:sink.write(OUTPUT/name,value))
            need(costs.forwards==29 and result['autograd_grad_calls']==79,'exact_forward_and_gradient_counts')
            frozen_check();need(parameter_hash(policy)==CHECKPOINT['parameter_hash'],'unchanged_full_checkpoint')
            need(tuple(p.requires_grad for p in policy.parameters())==frozen_flags,'gradient_flags_restored')
            result.update(parameters_unchanged=True,gradient_buffers_unchanged=True,requires_grad_flags_restored=True,
                gradient_report_sha256=sha(OUTPUT/'gradient-alignment.json'),cache_seal=cache.record()['cache_seal'])
            budget.complete(history_complete=True)
        result['status']='complete_frozen_gradient_diagnostic'
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        try:
            if policy is not None and frozen_flags is not None:
                # Observe failures; never erase .grad or overwrite values to conceal them.
                result['requires_grad_flags_restored']=tuple(p.requires_grad for p in policy.parameters())==frozen_flags
                result['gradient_buffers_unchanged']=all(p.grad is None for p in policy.parameters())
                result['parameters_unchanged']=(tuple(p._version for p in policy.parameters())==frozen_versions
                    and parameter_hash(policy)==CHECKPOINT['parameter_hash'])
        except BaseException as error:
            result.update(status='failed_or_unresolved',frozen_check_failure={'type':type(error).__name__,'message':str(error)})
        finally:
            for owner,name,old in reversed(originals):setattr(owner,name,old)
        result.update(completed_source_visits=len(visits.completed),policy_forwards=costs.forwards,
            native_previews=budget.snapshot()['native_preview_entries'],elapsed_seconds=time.perf_counter()-started)
        if external_io is not None:result['forbidden_io_calls']+=external_io['blocked']
        try:
            sink.write(OUTPUT/'costs.json',{'costs':costs.rows,'native_budget':budget.snapshot(),
                'completed_patient_visits':visits.completed,'counters':{k:result[k] for k in
                 ('policy_forwards','autograd_grad_calls','collector_replay_steps','checkpoint_loads')},
                'scope':'frozen endpoint diagnostic; inclusive nested timings overlap; no optimizer or deployment rollout'})
        except BaseException as error:
            result.update(status='failed_or_unresolved',cost_write_failure={'type':type(error).__name__,'message':str(error)})
            raise
        finally:sink.write(OUTPUT/'result.json',result,terminal=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--declaration-sha256',required=True);args=parser.parse_args()
    release,_,_=source_guard(args.declaration,args.declaration_sha256,execute=True)
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(torch.get_default_dtype()==torch.float32 and str(torch.get_default_device())=='cpu','CPU_float32')
    public=json.loads((ROOT/release['public_index']['path']).read_text());allowed=set()
    for row in public['cases']:
        manifest=json.loads(Path(row['path']).read_text())
        need(manifest['role']=='TRAIN' and manifest['public_only'] and not manifest['private_evaluation_files_included'],'public_TRAIN_only')
        allowed.update(str(Path(v['path']).resolve()) for v in manifest['input_files'].values())
    payload=ROOT/CHECKPOINT['path'];io={'blocked':0}
    def access(event,arguments):
        if event=='open' and arguments and isinstance(arguments[0],(str,bytes)):
            path=Path(os.fsdecode(arguments[0])).resolve();mode=arguments[1];flags=arguments[2]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            denied=(path.suffix.lower()=='.npy' and (str(path) not in allowed or writing)) or path.is_relative_to(ROOT/'data') or path.suffix.lower() in ('.dcm','.pt','.pth','.ckpt')
            if path.suffix.lower()=='.psckpt':denied=path!=payload or writing
            if denied:io['blocked']+=1;raise PermissionError('Only exact TRAIN inputs and one read-only checkpoint')
        if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
            io['blocked']+=1;raise PermissionError('No external work')
    # Execute imports/metadata are read-only; the hook closes acquired payloads
    # before any factory is called, including after a caught forbidden open.
    sys.addaudithook(access)
    supervision=OUTPUT.with_name(OUTPUT.name+'.supervision')
    need(supervision.is_dir(),'owned_supervision_required')
    def progress(*unused,**details):
        if io['blocked']:raise PermissionError('Prior blocked payload access')
    def deadline(*unused):raise TimeoutError('Frozen gradient worker deadline or termination')
    prior={s:signal.signal(s,deadline) for s in (signal.SIGALRM,signal.SIGTERM)}
    started=time.perf_counter();summary={'status':'started','runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'threads':1,'interop_threads':1,'device':'cpu','dtype':'float32'}}
    signal.setitimer(signal.ITIMER_REAL,CAPS['worker_seconds'])
    try:
        result=execute(release,args.declaration_sha256,progress,external_io=io)
        need(io['blocked']==0,'blocked_IO')
        summary['status']='complete'
    except BaseException as error:
        summary.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in prior.items():signal.signal(sig,handler)
        summary.update(elapsed_seconds=time.perf_counter()-started,blocked_payload_opens=io['blocked'],
            result_sha256=sha(OUTPUT/'result.json') if (OUTPUT/'result.json').is_file() else None)
        with (supervision/'worker-final.json').open('x') as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return 0
if __name__=='__main__':raise SystemExit(main())
