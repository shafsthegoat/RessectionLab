"""Owned, explicitly released full40 TRAIN refit. Canonical promoted imports only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent.parent))
from train_diagnostic_readout import index_teacher_states, summarize_logits


def execute(output,input_index):
    output.mkdir(exist_ok=False);started=time.perf_counter();session=None;experiment=None;meter=None
    result={'status':'started','scope':'additional_compute_TRAIN_IL_fit_not_a_new_benchmark',
        'optimizer_updates':0,'native_teacher_steps':0,'geometry_previews':0,'loss_forward_calls':0,
        'readout_forward_calls':0,'new_checkpoint_loads':0,'prior_checkpoint_loads':0,'search_calls':0,
        'patient_reads':0,'SELECT_or_MEASUREMENT_reads':0,'completed_teacher_reconstructions':[]}
    def write(name,value):
        with (output/name).open('x') as file:
            json.dump(value,file,indent=2,allow_nan=False);file.write('\n');file.flush()
    def read(name):
        row=input_index['data'][name];data=(ROOT/row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Fixed TRAIN evidence changed: '+name)
        return json.loads(data)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from resectionlab.contact_learning_contract import (freeze_contact_experiment,freeze_full_teacher_refit,
            bind_training_task,ContactSample,common_initial_policies)
        from resectionlab.contact_learning import ContactLearningSession,contact_imitation_loss,contact_gradient_step
        from resectionlab.contact_checkpoint import (save_contact_checkpoint,load_contact_checkpoint,
            initial_lineage,final_lineage,partial_lineage)
        from resectionlab.contact_costs import ContactCostMeter,peak_rss_bytes
        from resectionlab.public_contact_family import make_family_task
        from resectionlab.public_surface_contact import OBSERVATION_VERSION
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.core import semantic_digest,thaw_json
        from resectionlab.spatial_policy import parameter_hash
        def check():
            if time.perf_counter()-started>=180 or peak_rss_bytes()>1073741824:
                raise RuntimeError('Fixed180s/1GiB refit cap; preserve partial, no retry')
        original_preview=NativeResectionEngine.preview_stroke
        def bounded_preview(instance,*args,**kwargs):
            check()
            if result['geometry_previews']>=2048:raise RuntimeError('Fixed2048 preview cap')
            result['geometry_previews']+=1;return original_preview(instance,*args,**kwargs)
        NativeResectionEngine.preview_stroke=bounded_preview
        try:
            with ContactCostMeter() as meter:
                with meter.scope('refit.fixed_corpus_identity'):
                    teachers=[read(f'teacher-{i:02d}.json') for i in range(24)]
                    updates=[read(f'IL-update-{i:02d}.json') for i in range(1,33)]
                    states=index_teacher_states(teachers,updates,semantic_digest)
                    corpus=[{'layout_id':s['layout_id'],'goal_id':s['goal_id'],'step':s['step'],
                        'observation_hash':s['observation_hash'],'action_id':s['teacher_action'],
                        'original_binding_hash':s['binding_hash'],
                        'strategy_seal':teachers[s['teacher_index']]['strategy']['strategySeal']} for s in states]
                    baseline=freeze_contact_experiment();experiment=freeze_full_teacher_refit(corpus)
                    if (baseline.fingerprint!=input_index['baseline_experiment_hash']
                            or semantic_digest(read('experiment-freeze.json'))!=baseline.fingerprint
                            or experiment.fingerprint!=input_index['refit_experiment_hash']):
                        raise ValueError('Explicit baseline/refit experiment identities differ from release')
                    if [(t['layout_id'],t['goal_id']) for t in teachers]!=list(baseline.keys('TRAIN')):
                        raise ValueError('Original ordered24 TRAIN tasks changed')
                    result['baseline_experiment_hash']=baseline.fingerprint
                    result['experiment_hash']=experiment.fingerprint
                    write('experiment-freeze.json',thaw_json(experiment.record()))
                samples=[]
                with meter.scope('refit.reconstruct40_native_teacher_states'):
                    for i,teacher in enumerate(teachers):
                        check();layout,goal=teacher['layout_id'],teacher['goal_id']
                        task=make_family_task(layout,goal,cancelled=lambda:time.perf_counter()-started>=180)
                        old_binding=bind_training_task(baseline,task,layout_id=layout,goal_id=goal)
                        new_binding=bind_training_task(experiment,task,layout_id=layout,goal_id=goal)
                        if semantic_digest(old_binding.record())!=semantic_digest(teacher['binding']):
                            raise ValueError('Original teacher binding changed before new admission')
                        worker=task.planning_clone()
                        for state in [s for s in states if s['teacher_index']==i]:
                            check();obs=worker.observation();old_binding.require_observation(obs);new_binding.require_observation(obs)
                            if obs.fingerprint!=state['observation_hash']:
                                raise ValueError('Fixed teacher observation changed')
                            sample=ContactSample(obs,state['teacher_action'],new_binding);sample.validate();samples.append(sample)
                            worker.step(state['teacher_action']);result['native_teacher_steps']+=1
                        history=worker.metrics()['history'];saved=teacher['strategy']
                        strategy={'decision_model_hash':task.decision_model_hash,'source_hash':task.case.source_hash,
                            'actions':saved['strategy']['actions'],'history':history,
                            'observation_contract':OBSERVATION_VERSION,'max_steps':task.max_steps}
                        if (not worker.terminated or semantic_digest(history)!=semantic_digest(saved['strategy']['history'])
                                or semantic_digest(strategy)!=saved['strategySeal']
                                or semantic_digest(worker.independent_geometry_check().to_dict())!=semantic_digest(saved['geometryAudit'])):
                            raise ValueError('Original native history/seal/geometry changed')
                        record={'teacher_index':i,'original_binding_hash':old_binding.fingerprint,
                            'refit_binding_hash':new_binding.fingerprint,'strategy_seal':semantic_digest(strategy)}
                        result['completed_teacher_reconstructions'].append(record);write(f'teacher-reconstruction-{i:02d}.json',record)
                if len(samples)!=40 or result['native_teacher_steps']!=40:raise ValueError('Incomplete fixed40 corpus')
                samples=tuple(samples);protocol=experiment.protocol
                with meter.scope('refit.fresh_initialization'):
                    policy=common_initial_policies(experiment)['IL'];initial_hash=parameter_hash(policy)
                    session=ContactLearningSession(experiment,'IL',policy,initial_hash)
                    result['initial_checkpoint']=save_contact_checkpoint(output/'refit-initial.gmckpt',policy,experiment,initial_lineage(experiment))
                def readout(label,model):
                    model.eval();before=parameter_hash(model);rows=[]
                    with meter.scope('refit.'+label+'.readout40'),torch.no_grad():
                        for i,(state,sample) in enumerate(zip(states,samples)):
                            check();result['readout_forward_calls']+=1
                            logits,value=model(sample.observation,context=sample.binding.context)
                            obs=sample.observation
                            row={**state,**summarize_logits(logits.tolist(),list(map(bool,obs.action_mask)),
                                obs.action_ids,obs.action_modes,sample.action_id),'parameter_hash':before,'value':float(value)}
                            movement=sorted((j for j,ok in enumerate(obs.action_mask) if ok and obs.action_modes[j]!='stop'),
                                            key=lambda j:(-float(logits[j]),j))
                            target=sample.validate()
                            row['teacher_rank_among_legal_movements']=None if sample.action_id=='STOP' else movement.index(target)+1
                            row['movement_inventory_counts']={mode:sum(bool(ok) and m==mode for ok,m in zip(obs.action_mask,obs.action_modes))
                                for mode in ('aspirate','probe')}
                            rows.append(row);write(f'{label}-state-{i:02d}.json',row)
                    if parameter_hash(model)!=before:raise ValueError('Readout changed parameters')
                    groups={'all40':rows,'root24':[r for r in rows if r['step']==0],
                        'second16':[r for r in rows if r['step']==1],
                        'STOP8':[r for r in rows if r['teacher_action']=='STOP'],
                        'movement32':[r for r in rows if r['teacher_action']!='STOP'],
                        'aspiration_root16':[r for r in rows if r['step']==0 and r['teacher_action']!='STOP']}
                    return {name:{'n':len(group),'mean_cross_entropy':sum(r['cross_entropy'] for r in group)/len(group),
                        'accuracy':sum(r['correct'] for r in group)/len(group),
                        'greedy_STOP':sum(r['greedy_mode']=='stop' for r in group),
                        'teacher_movement_rank1':sum(r['teacher_rank_among_legal_movements']==1 for r in group),
                        'mean_STOP_probability':sum(r['STOP_probability'] for r in group)/len(group)} for name,group in groups.items()}
                result['before']=readout('before',policy)
                optimizer=torch.optim.Adam(policy.parameters(),lr=protocol['learning_rate'],betas=tuple(protocol['betas']),
                    **{key:protocol[key] for key in ('eps','weight_decay','amsgrad','maximize','foreach','fused')})
                policy.train()
                for update in range(32):
                    check()
                    with meter.scope('refit.loss40'):
                        loss,stats=contact_imitation_loss(session,samples);result['loss_forward_calls']+=stats['loss_forward_calls']
                    check()
                    with meter.scope('refit.optimizer'):
                        receipt=contact_gradient_step(session,optimizer,loss)
                    result['optimizer_updates']=session.updates
                    write(f'update-{update+1:02d}.json',{'update':update+1,'loss':stats,'update_receipt':receipt,
                        'experiment_hash':experiment.fingerprint,'all40_ordered_state_manifest_hash':semantic_digest(experiment.teacher_states),
                        'elapsed_seconds':time.perf_counter()-started})
                check()
                with meter.scope('refit.final_save_reload'):
                    checkpoint=save_contact_checkpoint(output/'refit-IL-final.gmckpt',policy,experiment,final_lineage(session))
                    result['final_checkpoint']=checkpoint;result['new_checkpoint_loads']+=1
                    restored,metadata=load_contact_checkpoint(checkpoint['path'],expected_sha256=checkpoint['sha256'],experiment=experiment,kind='final')
                    if metadata['lineage']['optimizer_updates']!=32 or parameter_hash(restored)!=parameter_hash(policy):
                        raise ValueError('Fixed final checkpoint changed on reload')
                result['after']=readout('after',restored)
                if (result['optimizer_updates'],result['loss_forward_calls'],result['readout_forward_calls'])!=(32,1280,80):
                    raise ValueError('Fixed refit work inventory differs')
                check();result['status']='complete_fixed_full40_TRAIN_refit'
        finally:NativeResectionEngine.preview_stroke=original_preview
    except BaseException as error:
        result.update(status='failed_or_capped',error_type=type(error).__name__,message=str(error))
        if session is not None and 'final_checkpoint' not in result:
            try:
                result['partial_checkpoint']=save_contact_checkpoint(output/'refit-IL-partial.gmckpt',session.policy,experiment,partial_lineage(session))
            except BaseException as save_error:result['partial_checkpoint_error']=str(save_error)
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-started;result['costs']={} if meter is None else meter.rows
        result['clinical_validation']=False;result['greedy_native_rollout_executed']=False
        write('result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--input-index',type=Path,required=True)
    parser.add_argument('--input-index-sha256',required=True);args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease();data=args.input_index.read_bytes()
    if hashlib.sha256(data).hexdigest()!=args.input_index_sha256:raise ValueError('Released refit input index changed')
    execute(args.output,json.loads(data))


if __name__=='__main__':main()
