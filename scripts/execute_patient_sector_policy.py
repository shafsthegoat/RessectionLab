#!/usr/bin/env python3
"""Cold execute one sealed hierarchical checkpoint; never fit or update it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from resectionlab.core import semantic_digest
from resectionlab.patient_sector_learning import (
    actor_choice, frozen_actor, parameter_hash, prepare_case, rollout, sha256)
from run_patient_sector_learning import (
    development_declaration, export_native, producer_identity)


def execution_export_context(payload, scene_id):
    """Preserve fitting exposure when a cold episode uses a saved DEV source.

    A desktop save changes archive bytes. Patient exposure follows its sealed
    scene identity; the execution receipt separately binds the actual archive.
    """
    proof = payload['trainingReceipt']
    schedule = proof.get('ppoSourceSchedule', 'train-only')
    binding = payload['sourceBinding']
    bindings = proof.get('optimizationSourceBindings', {'train': binding})
    expected = ({'train', 'transfer'} if schedule == 'paired-development' and
                payload['stage'] == 'bc-initialized-ppo' else {'train'})
    if (schedule not in ('train-only', 'paired-development') or
            set(bindings) != expected or bindings['train'] != binding):
        raise ValueError('Frozen checkpoint optimization source bindings differ')
    role = 'train' if binding['sceneId'] == scene_id else 'transfer'
    if schedule == 'paired-development' and role == 'transfer' and \
            bindings['transfer']['sceneId'] != scene_id:
        raise ValueError('Source is outside this paired DEVELOPMENT checkpoint exposure')
    return role, schedule, bindings


def run(source, checkpoint, output, *, access_method='superior-access'):
    started = time.perf_counter()
    source, checkpoint, output = Path(source).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    receipt = {'schema':'resectionlab.frozen-sector-execution.v1','status':'running',
        'source':{'path':str(source),'sha256':sha256(source)},
        'task':{'role':'DEVELOPMENT','accessMethod':access_method},
        'optimizerUpdates':0,'executorSourceSha256':sha256(__file__),
        'declaredBudget':{'maxWallSeconds':600.,'maxPhysicalCalls':512}}
    try:
        declaration = development_declaration(source)
        prepared = prepare_case(source, access_method=access_method)
        producer = producer_identity()
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        actor = frozen_actor(checkpoint,expected_producer=producer)
        actor_hash = parameter_hash(actor)
        proof = payload['trainingReceipt']
        updates = len(proof.get('ppoUpdates',[]))
        model = {'path':str(checkpoint),'fileSha256':sha256(checkpoint),
            'actorSha256':actor_hash,'stage':payload['stage'],
            'sourceBinding':payload['sourceBinding'],
            'trainingReceiptSha256':semantic_digest(proof)}
        receipt.update(model=model,producerIdentity=producer,sourceRoleDeclaration=declaration)
        source_key, schedule, bindings = execution_export_context(payload, prepared.scene.identity)
        receipt['executionMode'] = 'new-frozen-inference'
        receipt['optimizationSourceKey'] = source_key
        receipt['ppoSourceSchedule'] = schedule
        trial = rollout(prepared,actor_choice(actor),method='desktop-frozen-hierarchical-policy')
        # This token identifies an inference export; it is not a new fitting seed.
        derived = export_native(prepared,trial,model,source_key,output,
            seed='runtime-'+uuid.uuid4().hex[:8],declaration=declaration,
            optimizer_updates=updates, ppo_source_schedule=schedule,
            optimization_source_bindings=bindings)
        execution = trial['report']
        calls = (execution['cost']['previewCalls']+execution['cost']['committedCalls']+
            execution['cost']['independentReplayCalls']+derived['nativeReplayCalls'])
        wall = time.perf_counter()-started
        if calls>512 or wall>600.:
            raise TimeoutError('Frozen hierarchy exceeded its declared execution/export budget')
        if parameter_hash(actor)!=actor_hash:
            raise ValueError('Frozen actor changed during inference')
        receipt.update(status='evaluated',execution={**execution,
            'admittedNonStopActions':execution['admittedSectorOptions']},
            derived={**derived,'sha256':derived['fileSha256']},
            exactReplay={'exactNativeReplay':execution['exactNativeReplay'],
                'stateId':execution['finalStateId']},
            budget={'wallSeconds':wall,'physicalCalls':calls,
                'rollout':execution['cost'],'exportReplayCalls':derived['nativeReplayCalls'],
                'sourceValidationPhysicsCalls':0})
    except Exception:
        receipt.update(status='failed',failure=traceback.format_exc())
    finally:
        receipt['source']['unchanged']=sha256(source)==receipt['source']['sha256']
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--access-method',default='superior-access')
    parser.add_argument('--source-role',choices=['DEVELOPMENT'],required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if 'testcases' not in output.parts or output.exists():
        parser.error('Choose a new output directory under ignored testcases/')
    output.mkdir(parents=True)
    receipt=run(args.source,args.checkpoint,output,access_method=args.access_method)
    print(json.dumps({'status':receipt['status'],'execution':receipt.get('execution',{}).get('status'),
        'receipt':str(output/'receipt.json')}))
    if receipt['status']!='evaluated':
        raise SystemExit(1)


if __name__=='__main__':
    main()
