"""One released TRAIN reconstruction plus fixed initial/IL/RL forward readout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from train_diagnostic_readout import index_teacher_states, summarize_logits, aggregate_readouts


def execute(output, input_index):
    output.mkdir(exist_ok=False)
    started=time.perf_counter()
    result={'status':'started','scope':'fixed_TRAIN_forward_diagnosis_not_a_new_benchmark',
        'checkpoint_loads':0,'forward_calls':0,'native_steps':0,'geometry_previews':0,
        'optimizer_updates':0,'search_calls':0,'sampled_actions':0,'new_teacher_calls':0,
        'SELECT_or_MEASUREMENT_reads':0,'patient_reads':0,'readouts':{},'completed_reconstructions':[]}
    def write(name,value):
        with (output/name).open('x') as file:
            json.dump(value,file,indent=2,allow_nan=False);file.write('\n');file.flush()
    def read(name):
        row=input_index['data'][name]; path=ROOT/row['path']; data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Pinned TRAIN diagnostic input changed: '+name)
        return json.loads(data)
    meter=None
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from resectionlab.legacy_transfer_worker import _require_parent_lease
        from resectionlab.contact_costs import ContactCostMeter, peak_rss_bytes
        from resectionlab.contact_learning_contract import freeze_contact_experiment, bind_training_task
        from resectionlab.contact_checkpoint import load_contact_checkpoint
        from resectionlab.public_contact_family import make_family_task
        from resectionlab.public_surface_contact import OBSERVATION_VERSION
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.core import semantic_digest
        from resectionlab.spatial_policy import parameter_hash
        # main establishes the lease before imports; direct execute is not a run entry point.
        def check():
            if time.perf_counter()-started>30 or peak_rss_bytes()>1073741824:
                raise RuntimeError('Fixed diagnostic wall/RSS cap; preserve partial, no retry')
        original_preview=NativeResectionEngine.preview_stroke
        def bounded_preview(instance,*args,**kwargs):
            check()
            if result['geometry_previews']>=2048:
                raise RuntimeError('Fixed2048 geometry-preview cap; preserve partial, no retry')
            result['geometry_previews']+=1
            return original_preview(instance,*args,**kwargs)
        NativeResectionEngine.preview_stroke=bounded_preview
        try:
            with ContactCostMeter() as meter:
                with meter.scope('diagnostic.input_validation'):
                    teachers=[read(f'teacher-{i:02d}.json') for i in range(24)]
                    updates=[read(f'IL-update-{i:02d}.json') for i in range(1,33)]
                    states=index_teacher_states(teachers,updates,semantic_digest)
                    prospective=read('experiment-freeze.json')
                    final_freeze=read('final-checkpoint-freeze.json')
                    experiment=freeze_contact_experiment()
                    if experiment.fingerprint!=input_index['experiment_hash'] or semantic_digest(prospective)!=experiment.fingerprint:
                        raise ValueError('Prospective experiment identity changed')
                    if final_freeze['experiment_hash']!=experiment.fingerprint:
                        raise ValueError('Final checkpoint freeze belongs to another experiment')
                    if [(t['layout_id'],t['goal_id']) for t in teachers] != list(experiment.keys('TRAIN')):
                        raise ValueError('Frozen ordered TRAIN task inventory changed')
                detached=[]
                with meter.scope('diagnostic.reconstruct40'):
                    for i,teacher in enumerate(teachers):
                        check()
                        task=make_family_task(teacher['layout_id'],teacher['goal_id'],cancelled=lambda:time.perf_counter()-started>30)
                        binding=bind_training_task(experiment,task,layout_id=teacher['layout_id'],goal_id=teacher['goal_id'])
                        if semantic_digest(binding.record())!=semantic_digest(teacher['binding']):
                            raise ValueError('Canonical TRAIN binding changed')
                        worker=task.planning_clone()
                        for state in [s for s in states if s['teacher_index']==i]:
                            check(); obs=worker.observation();binding.require_observation(obs)
                            if obs.fingerprint!=state['observation_hash'] or binding.fingerprint!=state['binding_hash']:
                                raise ValueError('Reconstructed observation differs from exact saved IL input')
                            action=state['teacher_action']; action_index=obs.action_ids.index(action)
                            if not obs.action_mask[action_index]:raise ValueError('Saved teacher action became illegal')
                            detached.append((state,obs,binding.context))
                            worker.step(action);result['native_steps']+=1
                        history=worker.metrics()['history']
                        strategy={'decision_model_hash':task.decision_model_hash,'source_hash':task.case.source_hash,
                            'actions':teacher['strategy']['strategy']['actions'],'history':history,
                            'observation_contract':OBSERVATION_VERSION,'max_steps':task.max_steps}
                        if not worker.terminated or semantic_digest(history)!=semantic_digest(teacher['strategy']['strategy']['history']):
                            raise ValueError('Authoritative native teacher history changed')
                        if semantic_digest(strategy)!=teacher['strategy']['strategySeal']:
                            raise ValueError('Reconstructed complete strategy seal changed')
                        audit=worker.independent_geometry_check().to_dict()
                        if semantic_digest(audit)!=semantic_digest(teacher['strategy']['geometryAudit']):
                            raise ValueError('Saved full-tool geometry audit changed')
                        record={'teacher_index':i,'binding_hash':binding.fingerprint,
                            'strategy_seal':semantic_digest(strategy),'history_hash':semantic_digest(history),
                            'geometry_audit_hash':semantic_digest(audit),'states':sum(s['teacher_index']==i for s in states)}
                        result['completed_reconstructions'].append(record)
                        write(f'reconstruction-{i:02d}.json',record)
                if len(detached)!=40 or result['native_steps']!=40:
                    raise ValueError('Reconstruction must complete all40 original teacher states')
                for method in ('INITIAL','IL','RL'):
                    check(); ck=input_index['checkpoints'][method]
                    with meter.scope('diagnostic.'+method+'.load'):
                        result['checkpoint_loads']+=1
                        policy,metadata=load_contact_checkpoint(ROOT/ck['path'],expected_sha256=ck['sha256'],
                            experiment=experiment,kind=ck['kind'])
                        if parameter_hash(policy)!=ck['parameter_hash']:
                            raise ValueError('Decoded diagnostic checkpoint identity changed')
                        if method!='INITIAL' and (final_freeze['checkpoints'][method]['file_sha256']!=ck['sha256']
                            or metadata['lineage']['optimizer_updates']!=32):
                            raise ValueError('Diagnostic requires the original fixed32 endpoint')
                        policy.eval()
                    rows=[]
                    with meter.scope('diagnostic.'+method+'.forward40'), torch.no_grad():
                        for index,(state,obs,context) in enumerate(detached):
                            check();result['forward_calls']+=1
                            logits,value=policy(obs,context=context)
                            readout=summarize_logits(logits.tolist(),list(map(bool,obs.action_mask)),
                                obs.action_ids,obs.action_modes,state['teacher_action'])
                            row={**state,**readout,'value':float(value),'checkpoint':method,
                                'checkpoint_file_sha256':ck['sha256'],'parameter_hash':ck['parameter_hash']}
                            rows.append(row);write(f'{method}-state-{index:02d}.json',row)
                    if parameter_hash(policy)!=ck['parameter_hash']:
                        raise ValueError('Forward-only diagnostic mutated checkpoint parameters')
                    result['readouts'][method]=aggregate_readouts(rows)
                    del policy
                if (result['checkpoint_loads'],result['forward_calls'])!=(3,120):
                    raise ValueError('Fixed load/forward inventory did not complete')
                check();result['status']='complete_fixed_TRAIN_readout'
        finally:
            NativeResectionEngine.preview_stroke=original_preview
    except BaseException as error:
        result.update(status='failed_or_capped',error_type=type(error).__name__,message=str(error));raise
    finally:
        result['wall_seconds']=time.perf_counter()-started
        result['costs']={} if meter is None else meter.rows
        result['scope_limit']='Prospective all-role manifest identity only; no SELECT/MEASUREMENT task, observation or outcome file is read. No checkpoint selection or training follows from this diagnostic.'
        write('result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-index',type=Path,required=True)
    parser.add_argument('--input-index-sha256',required=True)
    args=parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    data=args.input_index.read_bytes()
    if hashlib.sha256(data).hexdigest()!=args.input_index_sha256:raise ValueError('Diagnostic input index changed')
    execute(args.output,json.loads(data))


if __name__=='__main__':main()
