#!/usr/bin/env python3
"""Controlled visited-state imitation diagnostic on the existing real PAT05 task."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import run_real_patient_imitation as bc
import run_real_patient_learning as rl
from preflight_real_spatial_policy import write_json,sha256,read_declaration,supervise_worker,peak_rss_bytes

VERSION='pat05-real-visited-imitation-v1'
ANCHOR=ROOT/'artifacts/pat05-real-geometric-imitation-v1'
ANCHOR_FILES=('declaration-input.json','receipt.json','latest_policy.json','latest.pt','output-sha256.json')
SETTINGS={'seed':11,'updates_per_branch':8,'examples_per_update':6,'learning_rate':.001,
          'gradient_clip':5.,'label_seconds':30.,'max_wall_seconds':360.,
          'max_rss_bytes':6*1024**3,'torch_threads':1,'device':'cpu'}


def source_inventory():
    return {**bc.source_inventory(),str(Path(__file__).relative_to(ROOT)):sha256(__file__)}


def declaration():
    return {'version':VERSION,'settings':SETTINGS,'source_sha256':source_inventory(),
        'anchor_sha256':{name:sha256(ANCHOR/name)for name in ANCHOR_FILES},
        'branch_order':['control','augmented'],
        'initialization':'separate reloads of identical BC8 policy AND Adam checkpoint',
        'control':'original three teacher examples repeated twice',
        'augmented':'original three teacher examples plus three frozen-BC8 visited states labeled by permitted immediate greedy search',
        'checkpoint_rule':'fixed latest after eight additional updates, no selection',
        'scope':'one TRAIN anatomy; imitation only, no RL, clinical or transfer claim'}


def validate(record):
    if record.get('version')!=VERSION or record.get('settings')!=SETTINGS or record.get('branch_order')!=['control','augmented']:
        raise ValueError('Fixed two-branch/eight-update/six-example comparison required')
    if record.get('source_sha256')!=source_inventory():raise ValueError('Source changed')
    if record.get('anchor_sha256')!={name:sha256(ANCHOR/name)for name in ANCHOR_FILES}:
        raise ValueError('BC8 anchor changed')
    definition,cohort,original,teacher=bc.validate(json.loads((ANCHOR/'declaration-input.json').read_text()))
    index=json.loads((ANCHOR/'output-sha256.json').read_text())
    if any(index.get(name)!=sha256(ANCHOR/name)for name in ANCHOR_FILES if name!='output-sha256.json'):
        raise ValueError('BC8 anchor differs from original execution bytes')
    previous=json.loads((ANCHOR/'receipt.json').read_text())
    visited=json.loads((ANCHOR/'latest_policy.json').read_text())
    if (previous.get('status')!='complete' or previous.get('optimizer_updates')!=8 or visited.get('status')!='complete'
            or visited['independent_evaluation'].get('accepted') is not True or len(visited['decisions'])!=3):
        raise ValueError('A completed independently checked BC8 source is required')
    return definition,cohort,original,teacher,previous,visited


def optimizer_digest(state):
    """Hash actual Adam tensor values and parameter-group metadata without pickle."""
    import torch
    h=hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            a=value.detach().cpu().contiguous().numpy()
            h.update(b'tensor');h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(value,dict):
            h.update(b'dict')
            for key in sorted(value,key=lambda x:(type(x).__name__,repr(x))):add(key);add(value[key])
        elif isinstance(value,(tuple,list)):
            h.update(type(value).__name__.encode())
            for item in value:add(item)
        else:h.update((type(value).__name__+':'+repr(value)+';').encode())
    add(state)
    return 'sha256:'+h.hexdigest()


def restore_branch(definition, expected_hash):
    import torch
    from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash
    # Reload separately on every invocation: optimizer.load_state_dict may keep
    # tensor storage supplied in its input. No branch can mutate the next seed.
    checkpoint=torch.load(ANCHOR/'latest.pt',map_location='cpu',weights_only=False)
    policy=SpatialPolicy(SpatialPolicyConfig(**definition['policy_config']))
    policy.load_state_dict(checkpoint['policy'])
    if checkpoint['update']!=8 or parameter_hash(policy)!=expected_hash or policy.architecture_hash!=definition['expected_policy_architecture_hash']:
        raise ValueError('Changed BC8 model initialization')
    optimizer=torch.optim.Adam(policy.parameters(),lr=SETTINGS['learning_rate'])
    expected_optimizer=optimizer_digest(checkpoint['optimizer'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if optimizer_digest(optimizer.state_dict())!=expected_optimizer:
        raise ValueError('Adam checkpoint changed during independent restoration')
    return policy,optimizer,expected_optimizer


def validate_label(observation,metrics,inventory,sequence,accounting):
    """Accept a label only from a complete current-state immediate comparison."""
    import math
    rows=accounting.get('decisions',[])
    if not accounting.get('complete') or not sequence or len(rows)!=1:
        raise ValueError('One complete live-state teacher decision required')
    row=rows[0];scores=row.get('scores',[])
    legal=[action for action,mask in zip(observation.action_ids,observation.action_mask)if mask]
    if ([x.get('action_id')for x in scores]!=legal or row.get('step')!=metrics['steps']
            or row.get('source_state_hash')!=inventory['cavity_state_hash']
            or row.get('all_current_legal_actions_scored') is not True
            or row.get('legal_nonstop_actions')!=len(legal)-1
            or row.get('scored_nonstop_actions')!=len(legal)-1
            or any(not math.isfinite(x['reward'])for x in scores)):
        raise ValueError('Teacher scores differ from exact current observed inventory/state')
    chosen=max(scores,key=lambda x:x['reward'])['action_id']
    if chosen!=row.get('selected_action_id') or chosen!=sequence[0]:
        raise ValueError('Teacher label differs from stable maximum permitted immediate score')
    return chosen


def collect_visited(base,policy,expected,output,guard,audit):
    import numpy as np
    import torch
    from resectionlab.core import semantic_digest
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    started=time.perf_counter();before=parameter_hash(policy);task=base.clone();samples=[]
    report={'status':'collecting','policy_hash_before':before,'decisions':[],
            'attempted_actions':0,'committed_transitions':0,'invalid_actions':0}
    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(output/'visited_collection.json',report);guard()
    try:
        while not task.terminated:
            guard();observation=task.observation();metrics=task.metrics();inventory=task.candidate_inventory()
            state_hash=semantic_digest(metrics)
            sequence,accounting=task.observed_one_step_search(seconds=SETTINGS['label_seconds'])
            label=validate_label(observation,metrics,inventory,sequence,accounting)
            if semantic_digest(task.metrics())!=state_hash or task.observation().fingerprint!=observation.fingerprint:
                raise RuntimeError('Teacher query changed the actual frozen-policy state')
            with torch.no_grad():
                logits,value=policy(observation);probabilities=logits.softmax(-1);index=int(logits.argmax())
            action=observation.action_ids[index]
            row={'step':metrics['steps'],'observation_hash':observation.fingerprint,
                'action_ids':list(observation.action_ids),'action_mask':observation.action_mask.tolist(),
                'label_action_id':label,'label_scope':'current decision only; appended teacher STOP is not an example',
                'teacher_accounting':accounting,'actual_action_id':action,
                'logits':[float(v)if np.isfinite(v)else None for v in logits.detach().numpy()],
                'probabilities':probabilities.tolist(),'value':float(value.detach()),
                'state_hash_before_teacher':state_hash,'state_hash_after_teacher':semantic_digest(task.metrics()),
                'status':'attempted'}
            report['decisions'].append(row);report['attempted_actions']+=1;save()
            try:result=task.step(action)
            except CommittedTransitionInterrupted as error:
                row.update(status='committed_unreturned',info=error.info);report['committed_transitions']+=1;raise
            except (ValueError,IndexError,KeyError):
                report['invalid_actions']+=1;raise
            report['committed_transitions']+=1
            row.update(status='returned',info=result.info,reward=result.reward,terminated=result.terminated)
            samples.append((observation,label));save()
        report['metrics']=task.metrics()
        report['independent_evaluation']=audit(task,metrics=report['metrics'])
        if report['independent_evaluation'].get('accepted') is not True:
            raise RuntimeError('Collected frozen-policy episode failed independent check')
        if report['independent_evaluation']['committed_history_hash']!=expected['independent_evaluation']['committed_history_hash']:
            raise RuntimeError('Frozen BC8 collection differs from its original completed rollout')
        report['policy_hash_after']=parameter_hash(policy)
        if report['policy_hash_after']!=before or len(samples)!=3:
            raise RuntimeError('Collection changed behavior weights or lost a prescribed live state')
        report['status']='complete';save()
        return tuple(samples),report
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error)})
        report['metrics_at_failure']=task.metrics();write_json(output/'visited_collection.json',report);raise


def make_batches(original_samples,visited_samples):
    if len(original_samples)!=3 or len(visited_samples)!=3:
        raise ValueError('Exactly three original and three live visited examples required')
    original_samples,visited_samples=tuple(original_samples),tuple(visited_samples)
    return {'control':original_samples*2,'augmented':original_samples+visited_samples}


def worker(record,output):
    definition,cohort,original,teacher,previous,visited=validate(record)
    import torch
    from resectionlab.spatial_policy import parameter_hash,imitation_loss,gradient_step
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    started=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(SETTINGS['seed'])
    receipt={'version':VERSION,'status':'running','algorithm':'controlled_visited_state_behavior_cloning',
             'patient':'sub-PAT05','role':'TRAIN','other_patients_opened':0,'settings':SETTINGS,'branches':{},
             'initial_parameter_hash':previous['latest_parameter_hash'],'generalization_measured':False}
    def guard():
        if time.perf_counter()-started>SETTINGS['max_wall_seconds'] or peak_rss_bytes()>SETTINGS['max_rss_bytes']:
            raise RuntimeError('Visited-state comparison resource cap exceeded')
    def preserve():
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=peak_rss_bytes())
        write_json(output/'receipt.json',receipt);guard()
    def audit(task,*,metrics):
        return evaluate_native_spatial_episode(task,metrics=metrics,minimum_target_cells=1,
            distance_backend='batch',distance_batch_size=256)
    try:
        policy,unused_optimizer,initial_adam=restore_branch(definition,previous['latest_parameter_hash'])
        del unused_optimizer
        receipt['initial_optimizer_hash']=initial_adam;preserve()
        prep=time.perf_counter();base=rl.inputs.load_member(definition,cohort)
        receipt['shared_preparation_seconds']=time.perf_counter()-prep
        if base.decision_model_hash!=original['decision_model_hash']:raise ValueError('Changed physical task')
        sequence=tuple(x['action_id']for x in teacher['decisions'])
        transitions,old=rl.run_episode(base,policy,None,mode='sequence',name='original_teacher',
            output=output,guard=guard,audit=audit,sequence=sequence)
        if old['independent_evaluation']['committed_history_hash']!=teacher['independent_evaluation']['committed_history_hash']:
            raise ValueError('Original teacher replay differs from complete audited source')
        original_samples=tuple((t.observation,t.action_id)for t in transitions)
        visited_samples,collected=collect_visited(base,policy,visited,output,guard,audit)
        receipt['original_teacher_replay_seconds']=old['elapsed_seconds']
        receipt['visited_collection_seconds']=collected['elapsed_seconds']
        receipt['visited_label_search_seconds']=sum(x['teacher_accounting']['planning_seconds']for x in collected['decisions'])
        batches=make_batches(original_samples,visited_samples)
        for name in ('control','augmented'):
            policy,optimizer,adam=restore_branch(definition,previous['latest_parameter_hash'])
            if adam!=initial_adam:raise ValueError('Branch starts from a changed Adam state')
            samples=batches[name]
            branch={'initial_parameter_hash':parameter_hash(policy),'initial_optimizer_hash':adam,'optimizer_updates':0,
                'samples':[{'observation_hash':o.fingerprint,'label_action_id':a}for o,a in samples],
                'example_count':len(samples),'unique_observation_count':len({o.fingerprint for o,a in samples}),'curve':[],
                'initial_diagnostics':bc.teacher_diagnostics(policy,samples)}
            receipt['branches'][name]=branch;preserve()
            for update in range(8):
                guard();step_started=time.perf_counter();loss,loss_record=imitation_loss(policy,samples)
                gradients=gradient_step(policy,optimizer,loss,max_norm=SETTINGS['gradient_clip'])
                branch['optimizer_updates']+=gradients['optimizer_steps']
                branch['curve'].append({'update':update+1,**loss_record,**gradients,
                    'loss_backward_optimizer_seconds':time.perf_counter()-step_started})
                torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),
                    'parameter_hash':parameter_hash(policy),'additional_updates':update+1,
                    'architecture':policy.architecture_record()},output/(name+'_latest.pt'))
                del loss;preserve()
            branch['latest_diagnostics']=bc.teacher_diagnostics(policy,samples)
            with torch.no_grad():_,branch['latest_training_loss']=imitation_loss(policy,samples)
            _,latest=rl.run_episode(base,policy,None,mode='argmax',name=name+'_latest',output=output,guard=guard,audit=audit)
            branch.update(status='complete',latest_parameter_hash=parameter_hash(policy),
                latest_optimizer_hash=optimizer_digest(optimizer.state_dict()),
                training_loss_forward_calls=sum(x['loss_forward_calls']for x in branch['curve']),
                diagnostic_forward_calls=18,rollout_forward_calls=latest['policy_forward_calls'],
                latest={'return':latest['simulated_return'],'actions':[x['action_id']for x in latest['decisions']],
                    'seconds':latest['elapsed_seconds'],'independent_evaluation':latest['independent_evaluation']},
                gain_over_BC8=latest['simulated_return']-previous['latest_policy']['return'])
            preserve()
        receipt.update(status='complete',augmented_minus_control_return=receipt['branches']['augmented']['latest']['return']-receipt['branches']['control']['latest']['return'])
        if source_inventory()!=record['source_sha256']:raise RuntimeError('Source changed during comparison')
        preserve()
    except BaseException as error:
        receipt.update(status='failed',failure={'type':type(error).__name__,'message':str(error)},elapsed_seconds=time.perf_counter()-started)
        write_json(output/'receipt.json',receipt);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--write-declaration',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--expected-declaration-sha256',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.write_declaration:
        if args.declaration.exists():raise SystemExit('Preserve existing declaration')
        write_json(args.declaration,declaration());return
    record,digest,payload=read_declaration(args.declaration,args.expected_declaration_sha256);validate(record)
    if args.worker:
        if not args.expected_declaration_sha256:raise ValueError('Frozen declaration identity required')
        try:worker(record,args.output)
        except BaseException as error:
            write_json(args.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
        return
    if not args.execute:print(json.dumps({'status':'validated_not_executed','declaration_sha256':digest}));return
    if args.output is None or args.output.exists():raise SystemExit('Choose new output directory')
    args.output.mkdir(parents=True);(args.output/'declaration-input.json').write_bytes(payload)
    for relative in record['source_sha256']:
        path=args.output/'source-snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,path)
    result=supervise_worker([sys.executable,str(Path(__file__).resolve()),'--worker','--declaration',str((args.output/'declaration-input.json').resolve()),
        '--expected-declaration-sha256',digest,'--output',str(args.output.resolve())],args.output,SETTINGS,digest)
    write_json(args.output/'output-sha256.json',{p.name:sha256(p)for p in sorted(args.output.iterdir())if p.is_file()and p.name!='output-sha256.json'})
    if result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
