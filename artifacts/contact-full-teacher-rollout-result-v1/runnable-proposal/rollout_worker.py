"""One fixed 24-TRAIN-goal greedy pass; existing native planner/exporter only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from rollout_summary import summarize


def execute(output,index):
    output.mkdir(exist_ok=False);start=time.perf_counter();meter=None;rows=[]
    result={'status':'started','scope':'fixed_TRAIN_native_usefulness_check_not_generalization',
        'rows':rows,'checkpoint_loads':0,'actor_forward_calls':0,'geometry_previews':0,
        'optimizer_updates':0,'new_search_calls':0,'new_teacher_calls':0,'patient_reads':0,
        'SELECT_or_MEASUREMENT_reads':0,'partial_plans':[]}
    def write(name,value):
        with (output/name).open('x') as file:json.dump(value,file,indent=2,allow_nan=False);file.write('\n');file.flush()
    def read(name):
        record=index['data'][name];data=(ROOT/record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Released saved input changed: '+name)
        return json.loads(data)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!='cpu':
            raise ValueError('Owned rollout requires CPU float32 defaults')
        from resectionlab.contact_learning_contract import freeze_full_teacher_refit,freeze_contact_experiment,bind_training_task
        from resectionlab.contact_checkpoint import load_contact_checkpoint
        from resectionlab.contact_family_episode import export_family_strategy
        from resectionlab.goal_mode_episode_adapter import plan_goal_mode_strategy
        from resectionlab.goal_mode_spatial_policy import GoalModeSpatialPolicy
        from resectionlab.surface_contact_episode import episode_envelope
        from resectionlab.public_contact_family import make_family_task
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.contact_costs import ContactCostMeter,peak_rss_bytes
        from resectionlab.core import semantic_digest,thaw_json
        from resectionlab.spatial_policy import parameter_hash
        def cancelled():return time.perf_counter()-start>=45 or peak_rss_bytes()>1073741824
        def check():
            if cancelled():raise RuntimeError('Fixed45s/1GiB rollout cap; preserve partial, no retry')
        original_preview=NativeResectionEngine.preview_stroke;original_forward=GoalModeSpatialPolicy.forward
        def bounded_preview(instance,*args,**kwargs):
            check()
            if result['geometry_previews']>=2048:raise RuntimeError('Fixed2048 preview cap')
            result['geometry_previews']+=1;return original_preview(instance,*args,**kwargs)
        def bounded_forward(instance,*args,**kwargs):
            check()
            if result['actor_forward_calls']>=48:raise RuntimeError('Fixed48 actor-forward cap')
            result['actor_forward_calls']+=1;return original_forward(instance,*args,**kwargs)
        NativeResectionEngine.preview_stroke=bounded_preview;GoalModeSpatialPolicy.forward=bounded_forward
        try:
            with ContactCostMeter() as meter:
                with meter.scope('fixed_inputs_and_checkpoint'):
                    frozen=read('refit-experiment.json');experiment=freeze_full_teacher_refit(frozen['teacher_states'])
                    if semantic_digest(frozen)!=experiment.fingerprint or experiment.fingerprint!=index['experiment_hash']:
                        raise ValueError('Fixed refit experiment changed')
                    baseline=freeze_contact_experiment()
                    teachers=[read(f'teacher-{i:02d}.json') for i in range(24)]
                    keys=experiment.keys('TRAIN')
                    if len(keys)!=24 or [(t['layout_id'],t['goal_id']) for t in teachers]!=list(keys):
                        raise ValueError('Original ordered24 TRAIN tasks required; no replacement')
                    for i,teacher in enumerate(teachers):
                        if (teacher['teacher_index']!=i or teacher['role']!='TRAIN'
                                or teacher['status']!='complete_bounded_teacher'
                                or semantic_digest(teacher['strategy']['strategy'])!=teacher['strategy']['strategySeal']):
                            raise ValueError('Original saved SEARCH outcome/role/seal changed')
                    checkpoint=index['checkpoint'];fit=read('refit-result.json')
                    if (fit['status']!='complete_fixed_full40_TRAIN_refit' or fit['experiment_hash']!=experiment.fingerprint
                            or fit['optimizer_updates']!=32 or fit['final_checkpoint']['sha256']!=checkpoint['sha256']
                            or fit['final_checkpoint']['parameter_hash']!=checkpoint['parameter_hash']):
                        raise ValueError('Saved fixed-fit result differs from released final artifact')
                    result['prior_fit_costs']={key:fit[key] for key in ('wall_seconds','loss_forward_calls','readout_forward_calls','optimizer_updates')}
                    result['checkpoint_loads']+=1
                    policy,metadata=load_contact_checkpoint(ROOT/checkpoint['path'],expected_sha256=checkpoint['sha256'],
                                                            experiment=experiment,kind='final')
                    if (metadata['parameter_hash']!=checkpoint['parameter_hash'] or metadata['lineage']['method']!='IL'
                            or metadata['lineage']['optimizer_updates']!=32):raise ValueError('Released final32 IL identity changed')
                    result['checkpoint']={**checkpoint,'lineage_hash':semantic_digest(metadata['lineage'])}
                    result['experiment_hash']=experiment.fingerprint;initial_weights=parameter_hash(policy)
                    rows.extend({'index':i,'layout_id':layout,'goal_id':goal,'role':'TRAIN','status':'not_executed'}
                                for i,(layout,goal) in enumerate(keys))
                    write('declaration.json',{'experiment_hash':experiment.fingerprint,'checkpoint':result['checkpoint'],
                        'keys':keys,'horizon':2,'search_comparator':'original_saved_bounded_TRAIN_teacher_native_outcomes',
                        'comparison_scope':'same_tasks_goals_masks_reward_horizon_not_matched_training_or_total_compute',
                        'caps':index['caps'],'optimizer_updates':0,'new_search_calls':0})
                for row,teacher in zip(rows,teachers):
                    task=None;row_start=time.perf_counter();i=row['index'];layout,goal=row['layout_id'],row['goal_id']
                    try:
                        check()
                        with meter.scope('source_and_inventory'):
                            task=make_family_task(layout,goal,cancelled=cancelled)
                            if task.max_steps!=2:raise ValueError('Fixed two-decision horizon changed')
                            original_binding=bind_training_task(baseline,task,layout_id=layout,goal_id=goal)
                            binding=bind_training_task(experiment,task,layout_id=layout,goal_id=goal)
                            if semantic_digest(original_binding.record())!=semantic_digest(teacher['binding']):
                                raise ValueError('Saved SEARCH and policy task/source/goal/context differ')
                            saved=teacher['strategy']['strategy']
                            if saved['source_hash']!=task.case.source_hash or saved['decision_model_hash']!=task.decision_model_hash:
                                raise ValueError('Saved SEARCH decision/source model changed')
                            root_state=next(s for s in experiment.teacher_states if s['layout_id']==layout and s['goal_id']==goal and s['step']==0)
                            if task.observation().fingerprint!=root_state['observation_hash']:
                                raise ValueError('Exact initial public observation/mask changed')
                        with meter.scope('greedy_planning'):
                            plan,seal,accounting=plan_goal_mode_strategy(policy,task,context=binding.context)
                            write(f'TRAIN-{i:02d}-plan.json',thaw_json({'plan':plan,'seal':seal,'accounting':accounting}))
                            result['partial_plans'].append(i);check()
                        with meter.scope('authoritative_native_execution_and_v3_replay'):
                            display,episode=export_family_strategy(experiment,task,layout_id=layout,goal_id=goal,selector='IL',
                                plan=plan,seal=seal,accounting=accounting,context=binding.context,checkpoint_metadata=metadata)
                            write(f'TRAIN-{i:02d}-episode.json',episode_envelope(episode));check()
                        metrics=episode['metrics'];reference=teacher['strategy']['metrics']
                        keep=('total_reward','goal_contacted_and_retained','goal_retained','removed_volume_mm3','steps','terminated')
                        row.update(status='complete',metrics={k:metrics[k] for k in keep},
                            saved_search_metrics={k:reference[k] for k in keep},saved_search_accounting=teacher['search'],
                            saved_search_strategy_seal=teacher['strategy']['strategySeal'],strategy_seal=seal,
                            actions=list(plan['actions']),saved_search_actions=list(saved['actions']),
                            history_hash=semantic_digest(episode['history']),episode_id=episode['episodeId'],
                            case_hash=display.semantic_hash,source_hash=task.case.source_hash,decision_model_hash=task.decision_model_hash,
                            reward_difference_from_saved_search=metrics['total_reward']-reference['total_reward'],
                            planning=accounting,geometry_audit=episode['geometryAudit'])
                        if parameter_hash(policy)!=initial_weights:raise RuntimeError('Frozen model changed during inference')
                    except BaseException as error:
                        row.update(status='failed_or_capped',error_type=type(error).__name__,message=str(error),
                            durable_transition_committed=bool(getattr(error,'committed',False)),
                            committed_transition=getattr(error,'info',None),committed_reward=getattr(error,'reward',None),
                            partial_native_metrics=None if task is None else thaw_json(task.metrics()))
                        raise
                    finally:
                        row['wall_seconds']=time.perf_counter()-row_start;write(f'TRAIN-{i:02d}-result.json',thaw_json(row))
                if len(rows)!=24 or any(row['status']!='complete' for row in rows):raise RuntimeError('Fixed24 pass incomplete')
                result['planning_transition_calls']=sum(len(row['actions']) for row in rows)
                result['authoritative_execution_transition_calls']=sum(len(row['actions']) for row in rows)
                result['observed_native_transition_calls']=sum(cost.get('native_transition_calls',0) for cost in meter.rows.values())
                if (result['actor_forward_calls']!=result['planning_transition_calls']
                        or result['observed_native_transition_calls']!=2*result['planning_transition_calls']
                        or result['observed_native_transition_calls']>96):
                    raise RuntimeError('Native/forward inventory differs from fixed planning plus authoritative execution')
                result['summary']=summarize(rows);result['parameter_hash_after']=parameter_hash(policy)
                check();result['status']='complete_fixed24_TRAIN_greedy_pass'
        finally:NativeResectionEngine.preview_stroke=original_preview;GoalModeSpatialPolicy.forward=original_forward
    except BaseException as error:
        result.update(status='failed_or_capped',error_type=type(error).__name__,message=str(error));raise
    finally:
        result['wall_seconds']=time.perf_counter()-start;result['costs']={} if meter is None else meter.rows
        result['clinical_validation']=False;result['checkpoint_selection']=False;result['desktop_publication']=False
        write('result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-index',type=Path,required=True);parser.add_argument('--input-index-sha256',required=True)
    args=parser.parse_args();from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease();data=args.input_index.read_bytes()
    if hashlib.sha256(data).hexdigest()!=args.input_index_sha256:raise ValueError('Released rollout input index changed')
    execute(args.output,json.loads(data))


if __name__=='__main__':main()
