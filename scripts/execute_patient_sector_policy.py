#!/usr/bin/env python3
"""Cold execute one sealed hierarchical checkpoint; never fit or update it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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

MATCHING_RUNTIME_COMMIT = '01e5a365abb7cf9b158a27d49f1e1bfa83e593a2'
MATCHING_RUNTIME_ARCHIVE_SHA256 = 'bbaeaa5098c918b566523b4c098db96c742454f593a772fa84e597de1d51b3e2'
MATCHING_RUNTIME_MANIFEST_SHA256 = 'ee0dfb0e4cff704d5f42b2669efb3a24db0da615f01cc4ee631b525cb6d12ccb'
MATCHING_RUNTIME = ROOT / 'testcases/outputs/integrated-goal-20261026/joint-frozen-runtime-v1/runtime'
FROZEN_V6 = ROOT / 'testcases/downloaded-datasets/UCSF-frozen-assessment-20261027'


def _matching_runtime_files(checkpoint_producer):
    """Admit only the code files from the pinned plain source export."""
    runtime = MATCHING_RUNTIME / 'source'
    manifest_path = MATCHING_RUNTIME / 'files-manifest.json'
    if not manifest_path.is_file() or sha256(manifest_path) != MATCHING_RUNTIME_MANIFEST_SHA256:
        raise ValueError('Pinned matching-runtime file manifest missing or changed')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'resectionlab.frozen-matching-runtime-files.v1' or
            manifest.get('sourceArchiveSha256') != MATCHING_RUNTIME_ARCHIVE_SHA256 or
            manifest.get('gitCommit') != MATCHING_RUNTIME_COMMIT or
            manifest.get('producerIdentity') != checkpoint_producer):
        raise ValueError('Matching-runtime export, commit or producer differs')
    files = manifest.get('files')
    if not isinstance(files, dict) or len(files) != 236:
        raise ValueError('Matching-runtime source inventory differs')
    allowed_roots = (runtime / 'src/resectionlab', runtime / 'scripts')
    actual = {str(path.relative_to(runtime)) for root in allowed_roots
              for path in root.rglob('*') if path.is_file() and
              '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if actual != set(files):
        raise ValueError('Matching-runtime source file inventory changed')
    for relative, expected in files.items():
        file = runtime / relative
        if (not relative.startswith(('src/resectionlab/', 'scripts/')) or
                file.is_symlink() or not file.is_file() or
                not file.resolve().is_relative_to(runtime.resolve()) or
                sha256(file) != expected):
            raise ValueError('Matching-runtime source file changed: ' + relative)
    linked_testcases = runtime / 'testcases'
    if not linked_testcases.is_symlink() or linked_testcases.resolve() != (ROOT / 'testcases').resolve():
        raise ValueError('Matching runtime must link the canonical local testcases directory')
    return runtime, manifest


def _matching_v6_checkpoint(checkpoint, payload, checkpoint_producer):
    """Require one of the six pre-pixel frozen v6 weight files."""
    freeze_path = FROZEN_V6 / 'checkpoint-freeze.json'
    receipt_path = FROZEN_V6 / 'UCSF-PDGM-0008/source-receipt.json'
    protocol_path = FROZEN_V6 / 'protocol.json'
    freeze = json.loads(freeze_path.read_text())
    source_receipt = json.loads(receipt_path.read_text())
    if (source_receipt.get('role') != 'FROZEN_ASSESSMENT' or
            source_receipt.get('checkpoint_freeze_sha256') != sha256(freeze_path) or
            source_receipt.get('role_declaration_sha256') != sha256(protocol_path) or
            freeze.get('protocolSha256') != sha256(protocol_path) or
            freeze.get('producerIdentity') != checkpoint_producer or
            freeze.get('modelSelectionUsedAssessmentPixels') is not False or
            freeze.get('optimizerUpdatesAfterFreeze') != 0):
        raise ValueError('Pre-pixel six-checkpoint freeze differs')
    roster = freeze.get('checkpoints', {})
    expected_names = {f'{stage}-seed{seed}.pt' for seed in (17, 29, 43)
                      for stage in ('bc', 'ppo')}
    if set(roster) != expected_names or checkpoint.name not in roster:
        raise ValueError('Checkpoint is outside the frozen v6 roster')
    row = roster[checkpoint.name]
    if (Path(row.get('path', '')).resolve() != checkpoint or
            sha256(checkpoint) != row.get('fileSha256') or
            payload.get('actorSha256') != row.get('actorSha256') or
            payload.get('stage') != row.get('stage') or
            semantic_digest(payload.get('trainingReceipt', {})) != row.get('trainingReceiptSha256')):
        raise ValueError('Checkpoint bytes or actor identity differ from the freeze')
    return row


def _archived_producer_and_actor(runtime, checkpoint, expected_producer):
    """Fresh old-source interpreter checks loaded code and actor, without a scene."""
    code = ("import json,sys; from pathlib import Path; "
            "root=Path(sys.argv[1]); sys.path[:0]=[str(root/'scripts'),str(root/'src')]; "
            "from run_patient_sector_learning import producer_identity; "
            "from resectionlab.patient_sector_learning import frozen_actor,parameter_hash; "
            "producer=producer_identity(); actor=frozen_actor(sys.argv[2],expected_producer=producer); "
            "print(json.dumps({'producerIdentity':producer,'actorSha256':parameter_hash(actor)},"
            "sort_keys=True,allow_nan=False))")
    completed = subprocess.run([sys.executable, '-c', code, str(runtime), str(checkpoint)],
        cwd=runtime, capture_output=True, text=True, timeout=60, check=True)
    result = json.loads(completed.stdout)
    if result.get('producerIdentity') != expected_producer:
        raise ValueError('Loaded archived producer differs from frozen checkpoint')
    return result


def _execute_matching_runtime(source, checkpoint, output, access_method, payload, current_producer):
    checkpoint_producer = payload.get('producerIdentity')
    runtime, manifest = _matching_runtime_files(checkpoint_producer)
    frozen = _matching_v6_checkpoint(checkpoint, payload, checkpoint_producer)
    observed = _archived_producer_and_actor(runtime, checkpoint, checkpoint_producer)
    if observed['actorSha256'] != frozen['actorSha256']:
        raise ValueError('Loaded archived actor differs from frozen weight identity')
    admission = {'schema': 'resectionlab.frozen-matching-runtime-admission.v1',
        'reason': 'current_producer_differs_from_frozen_checkpoint',
        'sourceArchiveSha256': MATCHING_RUNTIME_ARCHIVE_SHA256,
        'gitCommit': MATCHING_RUNTIME_COMMIT,
        'manifestSha256': MATCHING_RUNTIME_MANIFEST_SHA256,
        'runtimeSource': str(runtime.resolve()),
        'executorSourceSha256': manifest['files']['scripts/execute_patient_sector_policy.py'],
        'checkpointPath': str(checkpoint), 'checkpointFileSha256': frozen['fileSha256'],
        'checkpointActorSha256': frozen['actorSha256'],
        'currentProducerIdentity': current_producer,
        'matchingProducerIdentity': checkpoint_producer,
        'canonicalTestcases': str((ROOT / 'testcases').resolve())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / 'matching-runtime-admission.json').open('x') as stream:
        json.dump(admission, stream, indent=2, allow_nan=False)
        stream.write('\n')
    archived = runtime / 'scripts/execute_patient_sector_policy.py'
    # The archived executor predates the opt-in flag; pass its original CLI only.
    os.execv(sys.executable, [sys.executable, str(archived), '--source', str(source),
        '--checkpoint', str(checkpoint), '--output', str(output),
        '--access-method', access_method, '--source-role', 'DEVELOPMENT'])


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
        producer = producer_identity()
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload.get('producerIdentity') != producer:
            raise ValueError('Frozen checkpoint producer differs from the current execution source')
        actor = frozen_actor(checkpoint,expected_producer=producer)
        actor_hash = parameter_hash(actor)
        declaration = development_declaration(source)
        prepared = prepare_case(source, access_method=access_method)
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
    parser.add_argument('--allow-matching-runtime',action='store_true',
                        help='Use only the pinned 01e5 frozen v6 runtime on a producer mismatch')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/'testcases').resolve()) or output.exists():
        parser.error('Choose a new output directory under ignored testcases/')
    checkpoint=args.checkpoint.resolve()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    current_producer=producer_identity()
    if payload.get('producerIdentity') != current_producer:
        if not args.allow_matching_runtime:
            parser.error('Frozen checkpoint producer differs; explicit matching-runtime opt-in required')
        _execute_matching_runtime(args.source.resolve(),checkpoint,output,
                                  args.access_method,payload,current_producer)
    output.mkdir(parents=True)
    receipt=run(args.source,args.checkpoint,output,access_method=args.access_method)
    print(json.dumps({'status':receipt['status'],'execution':receipt.get('execution',{}).get('status'),
        'receipt':str(output/'receipt.json')}))
    if receipt['status']!='evaluated':
        raise SystemExit(1)


if __name__=='__main__':
    main()
