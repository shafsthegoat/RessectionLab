"""Replay saved IL045 actions once; no policy construction, forwards or updates."""
import argparse,json,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from batch_contract import CAPS,OUTPUT,guard,need,sha,small

def main():
    p=argparse.ArgumentParser();p.add_argument('--declaration',type=Path,required=True);p.add_argument('--declaration-sha256',required=True);a=p.parse_args()
    release,_,_=guard(a.declaration,a.declaration_sha256,execute=True)
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    from resectionlab.core import semantic_digest,thaw_json
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.patient_planning_cohort_io import OutputBudget
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    from resectionlab.patient_planning_preflight import _history_identity
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    inputs=release['inputs'];load=lambda key:small(inputs[key]['path'],inputs[key]['sha256'])
    original=load('original_release');config=load('configuration');saved=load('plan')
    plan=saved['plan'];need(semantic_digest(plan)==saved['plan_seal'],'plan_seal')
    need(plan['terminal_reason']=='HORIZON' and plan['max_steps']==24 and len(plan['actions'])==24 and 'STOP' not in plan['actions'],'exact_saved_horizon')
    need(plan['parameter_hash']==release['parameter_hash'] and plan['learning_updates']==64,'checkpoint_identity')
    original_result=load('failed_result')
    need(original_result['status']=='failed_or_unresolved' and original_result['optimizer_updates']=={'IL':64,'RL':0}
        and original_result['checkpoints']['IL']['parameter_hash']==plan['parameter_hash']
        and original_result['checkpoints']['IL']['sha256']==inputs['checkpoint']['sha256'],'original_completed_updates_checkpoint')
    # Only hash the already published small checkpoint; do not deserialize it.
    need(sha(ROOT/inputs['checkpoint']['path'])==inputs['checkpoint']['sha256'],'saved_checkpoint_bytes')
    protocol=config['learning_protocol'];limits=config['admission_limits']
    OUTPUT.mkdir(exist_ok=False);sink=OutputBudget(OUTPUT,CAPS['output_bytes'])
    started=time.perf_counter();budget=PlanningBudget(NativeResectionEngine,max_native_previews=CAPS['native_previews'],seconds=CAPS['worker_seconds'])
    def check():budget.check();sink.check()
    progress_count=0
    def progress(phase,**details):
        nonlocal progress_count
        check();sink.write(OUTPUT/f'source-progress-{progress_count:03d}.json',{'phase':phase,**details});progress_count+=1
    result={'status':'started','subject':'ReMIND-045','release_sha256':a.declaration_sha256,
        'saved_plan_sha256':inputs['plan']['sha256'],'checkpoint_sha256':inputs['checkpoint']['sha256'],
        'source_visits':0,'committed_actions':0,'policy_forwards':0,'optimizer_updates':0,
        'checkpoint_loads':0,'teacher_search_calls':0,'training_admitted':False}
    result.update(original_IL_result_sha256=inputs['failed_result']['sha256'],
        original_IL_parent_receipt_sha256=inputs['failed_receipt']['sha256'],
        saved_plan_seal=saved['plan_seal'],parameter_hash=plan['parameter_hash'],
        prior_learning_updates=64,source_hash=plan['source_hash'],context_hash=plan['context_hash'],
        decision_model_hash=plan['decision_model_hash'],saved_history_hash=semantic_digest(plan['history']))
    def deadline(*unused):raise TimeoutError('Saved-plan diagnostic deadline')
    previous=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,CAPS['worker_seconds'])
    try:
        with budget:
            factories=make_train_visit_factories(manifest_index_path=ROOT/original['public_manifest_index']['path'],
                manifest_index_sha256=original['public_manifest_index']['sha256'],
                cohort_bytes=(ROOT/'manifests/experiments/remind-component-cohort-v1.json').read_bytes(),
                learning_protocol=protocol,limits=limits,released_record=original,
                released_sha256=inputs['original_release']['sha256'],progress=progress)
            source=OUTPUT/'source';source.mkdir()
            base,context=factories['ReMIND-045'](output=source);result['source_visits']=1
            need(context.fingerprint==plan['context_hash'] and base.case.source_hash==plan['source_hash']
                and base.decision_model_hash==plan['decision_model_hash']
                and base.observation().fingerprint==plan['initial_observation_hash'],'exact_source_context_world')
            for index,action in enumerate(plan['actions']):
                check();sink.write(OUTPUT/f'replay-attempt-{index:02d}.json',{'phase':'saved_action_replay','index':index,'action_id':action})
                base.step(action);result['committed_actions']+=1
            metrics=base.metrics()
            result['full_history_equal']=_history_identity(metrics['history'])==_history_identity(plan['history'])
            need(base.terminated and result['full_history_equal'],'exact_committed_history')
            # Preserve the complete physical history before the potentially rejecting audit.
            sink.write(OUTPUT/'committed-replay-metrics.json',metrics)
            check();audit=evaluate_native_spatial_episode(base,cancelled=lambda:(check() or False))
            sink.write(OUTPUT/'native-replay.json',{'metrics':metrics,'independent_geometry':audit})
            result.update(independent_accepted=audit['accepted'],independent_geometry=audit['geometry'],
                native_replay_sha256=sha(OUTPUT/'native-replay.json'),status='complete_saved_plan_diagnostic',
                first_failing_action_indices_zero_based=[i for i,x in enumerate(plan['actions']) if x==audit['geometry'].get('first_failed_action')],
                interpretation='Completed fixed-action diagnostic only; accepted=false remains an invalid/unaccepted geometric route. No training or route change.')
            # Completion certifies diagnostic execution, never a failed geometry history.
            budget.complete(history_complete=True)
    except BaseException as error:
        result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result.update(native_previews=budget.snapshot()['native_preview_entries'],elapsed_seconds=time.perf_counter()-started)
        sink.write(OUTPUT/'result.json',result,terminal=True)
    guard(a.declaration,a.declaration_sha256,execute=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
