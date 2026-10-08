#!/usr/bin/env python3
"""Eight fixed BC updates from the verified PAT05 greedy demonstration.

This is an imitation capacity diagnostic, not RL or a transfer experiment.
The initial weights, task and three-action teacher come from the completed
negative two-update RL experiment; the teacher is replayed and checked again.
"""
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_real_patient_learning as rl
from resectionlab.data_policy import DataPolicyError, historical_only
from preflight_real_spatial_policy import sha256, write_json, read_declaration, supervise_worker, peak_rss_bytes

VERSION = 'pat05-real-geometric-imitation-v1'
ANCHOR = ROOT / 'artifacts/pat05-real-geometric-learning-v1'
ANCHOR_FILES = ('declaration-input.json', 'receipt.json', 'greedy_search.json', 'initial.pt', 'output-sha256.json')
SETTINGS = {'seed':11, 'optimizer_updates':8, 'learning_rate':.001, 'gradient_clip':5.,
            'max_wall_seconds':240., 'max_rss_bytes':6*1024**3,
            'torch_threads':1, 'device':'cpu'}


def source_inventory():
    return {**rl.source_inventory(), str(Path(__file__).relative_to(ROOT)):sha256(__file__)}


def declaration():
    return {'version':VERSION, 'settings':SETTINGS, 'source_sha256':source_inventory(),
        'anchor_sha256':{name:sha256(ANCHOR/name)for name in ANCHOR_FILES},
        'initialization':'exact saved pre-RL initial checkpoint, fresh Adam',
        'teacher':'three complete independently checked greedy decisions, replayed once before BC',
        'checkpoint_rule':'fixed latest after eight updates, no best-loss or best-return selection',
        'same_patient_development_only':True, 'other_patients_opened':0}


def validate(record):
    if record.get('version')!=VERSION or record.get('settings')!=SETTINGS:
        raise ValueError('Expected fixed eight-update imitation diagnostic')
    if record.get('source_sha256')!=source_inventory():
        raise ValueError('Current source bytes changed')
    if record.get('anchor_sha256')!={name:sha256(ANCHOR/name)for name in ANCHOR_FILES}:
        raise ValueError('Completed real experiment/checkpoint/teacher changed')
    definition=json.loads((ANCHOR/'declaration-input.json').read_text())
    cohort=rl.validate_declaration(definition)
    previous=json.loads((ANCHOR/'receipt.json').read_text())
    teacher=json.loads((ANCHOR/'greedy_search.json').read_text())
    index=json.loads((ANCHOR/'output-sha256.json').read_text())
    if any(index.get(name)!=sha256(ANCHOR/name)for name in ANCHOR_FILES if name!='output-sha256.json'):
        raise ValueError('Anchor no longer matches its execution-time byte inventory')
    if (previous.get('status')!='complete' or previous.get('optimizer_updates')!=2
            or teacher.get('status')!='complete' or teacher['independent_evaluation'].get('accepted') is not True
            or len(teacher['decisions'])!=3 or any(x.get('status')!='returned'for x in teacher['decisions'])
            or not teacher['decisions'][-1]['terminated']):
        raise ValueError('Only the completed independently checked three-action teacher is eligible')
    return definition,cohort,previous,teacher


def teacher_diagnostics(policy, samples):
    import torch
    rows=[]
    with torch.no_grad():
        for observation,action in samples:
            logits,_=policy(observation)
            index=observation.action_ids.index(action)
            probability=logits.softmax(-1)
            rows.append({'observation_hash':observation.fingerprint,'teacher_action_id':action,
                'teacher_probability':float(probability[index]),
                'teacher_rank':1+int((logits>logits[index]).sum()),
                'argmax_action_id':observation.action_ids[int(logits.argmax())],
                'action_count':len(observation.action_ids)})
    return rows


@historical_only("RECORDED_EXPERIENCE_REQUIRED")
def worker(record, output):
    definition,cohort,previous,teacher=validate(record)
    import torch
    from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash,imitation_loss,gradient_step
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SETTINGS['seed'])
    policy=SpatialPolicy(SpatialPolicyConfig(**definition['policy_config']))
    checkpoint=torch.load(ANCHOR/'initial.pt',map_location='cpu',weights_only=False)
    policy.load_state_dict(checkpoint['policy'])
    if (policy.architecture_hash!=definition['expected_policy_architecture_hash']
            or parameter_hash(policy)!=previous['initial_parameter_hash']
            or checkpoint['update']!=0 or checkpoint['optimizer'] is not None):
        raise ValueError('BC must start from exact original untrained tensors without optimizer state')
    generator=torch.Generator().manual_seed(SETTINGS['seed']+100000)
    receipt={'version':VERSION,'status':'running','algorithm':'behavior_cloning_cross_entropy_only',
        'role':'TRAIN','patient':'sub-PAT05','track':'annotation_assisted','settings':SETTINGS,
        'initial_parameter_hash':parameter_hash(policy),'optimizer_updates':0,'curve':[],
        'other_patients_opened':0,'generalization_measured':False,
        'previous_greedy_planning_seconds':previous['search']['planning_seconds'],
        'previous_shared_source_preparation_seconds':previous['preparation_seconds'],
        'baseline_scope':'reuse completed prior random/initial/greedy outcomes on identical fixed task; no new independent samples'}

    def guard():
        if time.perf_counter()-started>SETTINGS['max_wall_seconds'] or peak_rss_bytes()>SETTINGS['max_rss_bytes']:
            raise RuntimeError('Imitation diagnostic resource limit exceeded')

    def preserve():
        receipt.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=peak_rss_bytes())
        write_json(output/'receipt.json',receipt)
        guard()

    def audit(task,*,metrics):
        return evaluate_native_spatial_episode(task,metrics=metrics,minimum_target_cells=1,
            distance_backend='batch',distance_batch_size=256)

    preserve()
    try:
        prep=time.perf_counter()
        base=rl.inputs.load_member(definition,cohort)
        receipt['shared_preparation_seconds']=time.perf_counter()-prep
        if base.decision_model_hash!=previous['decision_model_hash']:
            raise ValueError('Task decision model changed since the baseline')
        sequence=tuple(row['action_id']for row in teacher['decisions'])
        transitions,replay=rl.run_episode(base,policy,generator,mode='sequence',name='teacher_replay',
            output=output,guard=guard,audit=audit,sequence=sequence)
        if replay['independent_evaluation']['committed_history_hash']!=teacher['independent_evaluation']['committed_history_hash']:
            raise ValueError('Reconstructed demonstration differs from the audited baseline history')
        samples=tuple((t.observation,t.action_id)for t in transitions)
        receipt['teacher_replay']={'seconds':replay['elapsed_seconds'],'return':replay['simulated_return'],
            'independent_evaluation':replay['independent_evaluation']}
        receipt['initial_teacher_diagnostics']=teacher_diagnostics(policy,samples)
        with torch.no_grad():
            _,initial_loss=imitation_loss(policy,samples)
        receipt['initial_teacher_loss']=initial_loss
        optimizer=torch.optim.Adam(policy.parameters(),lr=SETTINGS['learning_rate'])
        for update in range(SETTINGS['optimizer_updates']):
            guard()
            update_started=time.perf_counter()
            loss,loss_record=imitation_loss(policy,samples)
            step=gradient_step(policy,optimizer,loss,max_norm=SETTINGS['gradient_clip'])
            receipt['optimizer_updates']+=step['optimizer_steps']
            receipt['curve'].append({'update':update+1,**loss_record,**step,
                                    'loss_backward_optimizer_seconds':time.perf_counter()-update_started})
            torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),
                'parameter_hash':parameter_hash(policy),'update':update+1,
                'architecture':policy.architecture_record()},output/'latest.pt')
            del loss
            preserve()
        with torch.no_grad():
            _,final_loss=imitation_loss(policy,samples)
        receipt['latest_teacher_loss']=final_loss
        receipt['latest_teacher_diagnostics']=teacher_diagnostics(policy,samples)
        _,latest=rl.run_episode(base,policy,generator,mode='argmax',name='latest_policy',
            output=output,guard=guard,audit=audit)
        receipt['latest_policy']={'return':latest['simulated_return'],
            'actions':[x['action_id']for x in latest['decisions']],
            'seconds':latest['elapsed_seconds'],'independent_evaluation':latest['independent_evaluation']}
        initial=previous['episodes']['initial_policy']
        greedy=previous['episodes']['greedy_search']
        receipt.update(status='complete',latest_parameter_hash=parameter_hash(policy),
            return_change_from_initial=latest['simulated_return']-initial['return'],
            latest_minus_greedy_return=latest['simulated_return']-greedy['return'],
            behavior_changed=receipt['latest_policy']['actions']!=initial['actions'],
            loss_forward_calls=3*(SETTINGS['optimizer_updates']+2),
            teacher_diagnostic_forward_calls=6,rollout_policy_forward_calls=latest['policy_forward_calls'])
        if source_inventory()!=record['source_sha256']:
            raise RuntimeError('Source changed during imitation diagnostic')
        preserve()
    except BaseException as error:
        receipt.update(status='failed',failure={'type':type(error).__name__,'message':str(error)},
            elapsed_seconds=time.perf_counter()-started)
        write_json(output/'receipt.json',receipt)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--write-declaration',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--expected-declaration-sha256',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.execute or args.worker:
        raise DataPolicyError("RECORDED_EXPERIENCE_REQUIRED", "run_real_patient_imitation.main")
    if args.write_declaration:
        if args.declaration.exists():raise SystemExit('Preserve existing declaration')
        write_json(args.declaration,declaration());return
    record,digest,payload=read_declaration(args.declaration,args.expected_declaration_sha256)
    validate(record)
    if args.worker:
        if not args.expected_declaration_sha256:raise ValueError('Frozen declaration identity required')
        try:worker(record,args.output)
        except BaseException as error:
            write_json(args.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
            raise
        return
    if not args.execute:
        print(json.dumps({'status':'validated_not_executed','declaration_sha256':digest}));return
    if args.output is None or args.output.exists():raise SystemExit('Choose a new output directory')
    args.output.mkdir(parents=True)
    (args.output/'declaration-input.json').write_bytes(payload)
    for relative in record['source_sha256']:
        path=args.output/'source-snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,path)
    result=supervise_worker([sys.executable,str(Path(__file__).resolve()),'--worker',
        '--declaration',str((args.output/'declaration-input.json').resolve()),'--expected-declaration-sha256',digest,
        '--output',str(args.output.resolve())],args.output,SETTINGS,digest)
    write_json(args.output/'output-sha256.json',{p.name:sha256(p)for p in sorted(args.output.iterdir())if p.is_file() and p.name!='output-sha256.json'})
    if result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
