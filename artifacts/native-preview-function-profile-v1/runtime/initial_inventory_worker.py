"""One accepted TRAIN045 native route, profiled greedy only; no model work."""
import argparse,json,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import CAPS,INPUTS,OUTPUT,guard,need,sha,small
from profile_window import profiled_call

def main():
    p=argparse.ArgumentParser();p.add_argument('--declaration',type=Path,required=True);p.add_argument('--declaration-sha256',required=True);a=p.parse_args()
    release,_,_=guard(a.declaration,a.declaration_sha256,execute=True)
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from resectionlab.core import semantic_digest
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_admission import make_patient_planning_task
    from resectionlab.public_patient_factory import prepare_public_source
    from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    load=lambda key:small(INPUTS[key]['path'],INPUTS[key]['sha256'])
    original=load('original_release');prior=load('case');plan=load('plan');prior_metrics=load('metrics');prior_audit=load('audit')
    need(prior_audit['accepted'] and prior['status']=='completed_fixed_post_exposure_route','accepted_teacher_required')
    index=small(original['public_index']['path'],original['public_index']['sha256'])
    entry=next(r for r in index['cases'] if r['patient_id']=='ReMIND-045')
    manifest=small(entry['path'],entry['sha256'])
    need(manifest['role']=='TRAIN' and not manifest['private_evaluation_files_included'],'public_TRAIN')
    active={str(Path(v['path']).resolve()) for v in manifest['input_files'].values()}
    need(set(manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'five_public_inputs')
    cohort=ROOT/original['cohort']['path'];need(sha(cohort)==original['cohort']['sha256'],'frozen_cohort')
    qualification={'scope':'post_exposure_initial_and_nominal_greedy','subjects':tuple(original['subjects']),
        'occupancy_condition':original['occupancy_condition'],'proposal_config':original['proposal_config'],
        'max_steps':24,'methods_executed':['observed_greedy'],'post_exposure_condition':original['post_exposure_condition'],
        'greedy_seconds_per_case':60,'retention_reference':original['retention_reference'],'retention_applied':False}
    limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':40000,'max_policy_forwards':0,
        'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=CAPS['native_previews'],seconds=CAPS['worker_seconds'])
    started=time.perf_counter();result={'status':'started','subject':'ReMIND-045','release_sha256':a.declaration_sha256,
        'source_visits':0,'greedy_calls':0,'committed_actions':0,'model_calls':0,'optimizer_calls':0,'checkpoint_loads':0,'blocked_external_calls':0}
    originals=[]
    def forbid(owner,name,counter):
        originals.append((owner,name,getattr(owner,name)))
        def refused(*args,**kwargs):result[counter]+=1;raise RuntimeError('Forbidden '+counter)
        setattr(owner,name,refused)
    forbid(torch.nn.Module,'__init__','model_calls');forbid(torch.nn.Module,'_call_impl','model_calls')
    forbid(torch.optim.Optimizer,'__init__','optimizer_calls');forbid(torch,'load','checkpoint_loads')
    def access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve()
            denied=(path.suffix.lower()=='.npy' and str(path) not in active) or path.suffix.lower() in ('.dcm','.pt','.pth','.ckpt') or path.is_relative_to(ROOT/'data')
            if path.suffix.lower()=='.npy':denied=denied or (isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if denied:result['blocked_external_calls']+=1;raise PermissionError('Only five public arrays permitted')
        if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
            result['blocked_external_calls']+=1;raise PermissionError('No external work')
    def check():budget.check();sink.check()
    def progress(*args,**kwargs):check()
    def deadline(*unused):raise TimeoutError('Profile worker deadline')
    handlers={s:signal.signal(s,deadline) for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL,CAPS['worker_seconds'])
    try:
        with budget:
            # Audit hook is intentionally installed after imports and metadata preflight.
            sys.addaudithook(access)
            source_dir=OUTPUT/'source';source_dir.mkdir()
            source,binding,qc,protocol=prepare_public_source(source_dir,{'limits':limits},INPUTS['original_release']['sha256'],None,progress,
                public_manifest_path=entry['path'],public_manifest_sha256=entry['sha256'],cohort_bytes=cohort.read_bytes(),
                learning_protocol_hash=semantic_digest(qualification),proposal_config=NominalCavityProposalConfig(**original['proposal_config']),
                public_target_context_variant=TARGET_CONTEXT,occupancy_condition=original['occupancy_condition'],post_exposure_condition=original['post_exposure_condition'])
            result['source_visits']=1
            task,context=make_patient_planning_task(source,cohort_bytes=cohort.read_bytes(),source_binding=binding,qc_receipt=qc,protocol=protocol)
            need(source.source_hash==prior['source_hash'] and context.fingerprint==prior['context_hash']
                 and task.observation().fingerprint==prior['initial_observation_hash'],'exact_original_world')
            result['construction_seconds']=time.perf_counter()-started
            result['greedy_calls']=1
            with budget.phase('planning'):
                actions,accounting=profiled_call(task.observed_greedy_search,OUTPUT/'native-functions.json',seconds=60)
            sink.write(OUTPUT/'greedy-plan.json',{'actions':actions,'accounting':accounting})
            need(accounting['complete'] and list(actions)==plan['actions'],'unchanged_complete_actions')
            before=dict(plan['accounting']);after=dict(accounting)
            before.pop('planning_seconds');after.pop('planning_seconds')
            need(before==after,'unchanged_scores_and_dispositions')
            replay_started=time.perf_counter()
            with budget.phase('execution'):
                for action in actions:check();task.step(action);result['committed_actions']+=1
            metrics=task.metrics();sink.write(OUTPUT/'episode-metrics.json',metrics)
            need(task.terminated and metrics['history']==prior_metrics['history'],'exact_full_history')
            result['authoritative_replay_seconds']=time.perf_counter()-replay_started
            audit_started=time.perf_counter()
            evaluated=evaluate_native_spatial_episode(task,cancelled=lambda:(check() or False))
            sink.write(OUTPUT/'independent-episode.json',evaluated)
            need(evaluated['accepted'] and evaluated['outcomes']==prior_audit['outcomes'],'current_evaluator_exact_outcome')
            result.update(independent_audit_seconds=time.perf_counter()-audit_started,independent_accepted=True,
                exact_history_and_actions=True,profile_sha256=sha(OUTPUT/'native-functions.json'),status='complete_profiled_native_route')
            budget.complete(history_complete=True)
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for owner,name,old in reversed(originals):setattr(owner,name,old)
        for s,handler in handlers.items():signal.signal(s,handler)
        result.update(native_previews=budget.snapshot()['native_preview_entries'],native_budget=budget.snapshot(),elapsed_seconds=time.perf_counter()-started)
        sink.write(OUTPUT/'result.json',result,terminal=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
