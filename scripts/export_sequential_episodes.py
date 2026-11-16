#!/usr/bin/env python3
"""Export real immutable finite-skill episodes into the existing native replay UI.

No fitting, new controller rollout, search trajectory substitution or source
migration. Every recorded physical request (including refusals) is re-executed;
frozen learned choices are independently checked against the original weights.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_sequential_task import (generated_task, observation,
    available_mask, endpoint_assessment, COMMON_CONFIG)
from resectionlab.surgical_sequential_benchmark import feature_matrix, verify_replay
from resectionlab.surgical_recorded_sequential import display_case, source_geometry, validate_episode_record
from resectionlab.surgical_procedure_policy import procedure_source_memberships, evaluate_procedure_sequence

SCHEMA = 'resectionlab.recorded-sequential-episodes.v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def bound_file(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def original_inputs(source_dir):
    """Anchor original files to the previously committed delivery manifest."""
    manifest_path=ROOT/'docs/sequential-milestone-manifest.json'
    manifest=read(manifest_path)
    for name,key in [('protocol.json','protocolSha256'),('freeze.json','freezeSha256'),('evaluation.json','evaluationSha256')]:
        if sha(source_dir/name)!=manifest[key]:
            raise ValueError('ORIGINAL_COMMITTED_EXPERIMENT_BYTES_CHANGED: '+name)
    return read(source_dir/'protocol.json'),read(source_dir/'freeze.json'),bound_file(manifest_path)


def _frozen_model(freeze, row):
    if row['method'] != 'learned': return None, None
    import torch
    from run_sequential_milestone import Selector
    candidates = [r for r in freeze['checkpoints'] if r['seed'] == row['seed']]
    if len(candidates) != 1: raise ValueError('RECORDED_CHECKPOINT_SEED_UNAVAILABLE')
    spec = candidates[0]
    if sha(spec['path']) != spec['sha256'] or not spec['rewardDrivenUpdates'] or spec['selectedUpdates'] <= 0:
        raise ValueError('RECORDED_REWARD_TRAINED_CHECKPOINT_CHANGED')
    model = Selector(); model.load_state_dict(torch.load(spec['path'], map_location='cpu', weights_only=True)); model.eval()
    return model, {**bound_file(spec['path']), 'seed':row['seed'], 'selectedUpdates':spec['selectedUpdates']}


def verify_recorded_episode(task, row, *, model=None):
    """Prove physical receipts and optional checkpoint choice before exporting."""
    if (row['sourceSceneId'] != task.scene.identity or row['configHash'] != task.config.fingerprint or
            row['parent'] != task.audit['parent'] or row['snapshot']['sceneId'] != task.scene.identity):
        raise ValueError('RECORDED_EPISODE_SOURCE_CHANGED')
    engine = task.initial_engine(); used = set(); attempts = invalid = native_position = 0
    requests = []; choices = []; accepted_states = [engine.state_id]
    for selection, skill_row in enumerate(row['skillHistory']):
        if model is not None:
            import torch
            with torch.no_grad():
                index, _ = model.select(feature_matrix(observation(task, engine, attempts=attempts,
                    invalid=invalid, selections=selection), task.skills), available_mask(task, engine, used), False)
            if task.skills[index].id != skill_row['skillId']:
                raise ValueError('RECORDED_LEARNED_CHOICE_DIFFERS_FROM_CHECKPOINT')
            choices.append(dict(selection=selection, skillId=skill_row['skillId'], matched=True))
        for receipt in skill_row['receipts']:
            before = engine.state_id; action = copy.deepcopy(receipt['action'])
            tool_before = copy.deepcopy(engine.state['slots'][action['slot']])
            settings = copy.deepcopy(engine.parameters(engine.state['slots'][action['slot']]))
            actual = engine.step(action)
            if semantic_digest(actual) != semantic_digest(receipt['outcome']):
                raise ValueError('RECORDED_PHYSICAL_OUTCOME_CHANGED')
            accepted = actual['status'] in ('accepted', 'partial')
            requests.append(dict(attemptIndex=len(requests), nativePositionBefore=native_position,
                nativePositionAfter=native_position+int(accepted), skillId=skill_row['skillId'], kind=skill_row['kind'],
                action=action, toolPositionBefore=tool_before['position'], toolPositionAfter=copy.deepcopy(engine.state['slots'][action['slot']]['position']), instrumentBefore=tool_before['instrument'], instrumentAfter=engine.state['slots'][action['slot']]['instrument'], effectiveParametersBefore=settings,
                effectiveParametersAfter=copy.deepcopy(engine.parameters(engine.state['slots'][action['slot']])),
                status=actual['status'], reason=actual.get('reason'), reasonCode=actual.get('reasonCode'),
                stateBefore=before, stateAfter=engine.state_id, elapsedSeconds=actual['elapsedSeconds'],
                completedFraction=actual['completedFraction']))
            if accepted: native_position += 1; accepted_states.append(engine.state_id)
        attempts += skill_row['attemptsConsumed']; invalid += skill_row['invalidAttemptsConsumed']
        used.add(skill_row['skillId'])
        if skill_row['finalStateId'] != engine.state_id: raise ValueError('RECORDED_SKILL_ENDPOINT_CHANGED')
    stopped = bool(row['skillHistory'] and row['skillHistory'][-1]['checkedStopAttempted'] and row['skillHistory'][-1]['status'] == 'accepted')
    endpoint = endpoint_assessment(engine, task.config, attempts=attempts, invalid_actions=invalid,
        stopped=stopped, externally_truncated=not stopped)
    if (engine.snapshot() != row['snapshot'] or engine.state_id != row['finalStateId'] or
            endpoint != row['endpoint'] or attempts != row['attempts'] or invalid != row['rejections'] or
            sum(r['status']=='rejected' for r in requests) != row['rejections']):
        raise ValueError('RECORDED_ACCEPTED_HISTORY_OR_ENDPOINT_CHANGED')
    # Restore independently from the original accepted snapshot too. Refusals
    # remain in requests, never inserted as invented accepted replay frames.
    cold = SurgicalEngine.restore(task.scene, row['snapshot'], use_cache=False)
    if cold.state != engine.state or cold.history != engine.history:
        raise ValueError('RECORDED_COLD_NATIVE_REPLAY_CHANGED')
    adapter = verify_replay(task, row)
    return engine, requests, accepted_states, dict(physicalRequests=len(requests),
        acceptedActions=len(engine.actions), actualRefusals=invalid, checkedStop=stopped,
        checkpointSelections=choices, originalSnapshotIdentical=True,
        coldNativeIdentical=True, adapterParity=adapter)


def export(source_dir, control_path, output, *, parent=200, tier='low'):
    import torch
    from resectionlab.imaging import save_case, load_case, read_case_artifacts
    from resectionlab.surgical_scene import from_case
    from resectionlab.native_backend import backend_identity
    torch.set_num_threads(1)
    source_dir = Path(source_dir).resolve(); output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'testcases'): raise ValueError('Keep testing artifacts under testcases/')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=SCHEMA, passed=False, startedAt=time.time(), originalSourcesImmutable=True,
        sourceCommit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), episodes=[],
        sourceHashes={name:sha(ROOT/name) for name in ['scripts/export_sequential_episodes.py','src/resectionlab/surgical_recorded_sequential.py','src/resectionlab/surgical_scene.py','src/resectionlab/surgical_sequential_task.py','src/resectionlab/surgical_sequential_benchmark.py','src/resectionlab/surgical_consequences.py']})
    source_bindings=[]
    try:
        protocol,freeze,original_manifest=original_inputs(source_dir)
        receipt['originalCommittedManifest']=original_manifest
        if freeze['protocolSha256'] != sha(source_dir/'protocol.json') or freeze['configHash'] != COMMON_CONFIG.fingerprint:
            raise ValueError('FROZEN_PROTOCOL_OR_CONFIG_CHANGED')
        task=generated_task(parent); case=display_case(task)
        admitted=[r for r in protocol['cohort'] if r['parent']==parent and r['split']=='heldout']
        if len(admitted)!=1 or admitted[0]['sourceSceneId']!=task.scene.identity: raise ValueError('ORIGINAL_LAYOUT_CHANGED')
        aggregate=read(source_dir/'evaluation.json')['rows']
        paths=[source_dir/'results'/f'{tier}-{parent}-{method}-{seed}.json' for method,seed in [('learned',17),('learned',43),('greedy',0),('search',0)]]+[Path(control_path)]
        record=dict(mapping=task.scene.regions, branches={'main':task.initial_engine().snapshot()},
            branchId='main', position=0, parents={'main':None}, timeline={'main':[]})
        packet=dict(schema=SCHEMA, caseHash=case.semantic_hash, sourceSceneId=task.scene.identity,
            sourceStateId=record['branches']['main']['stateId'], policyHash=record['branches']['main']['procedurePolicy']['policyHash'],
            configHash=task.config.fingerprint, parent=parent, tier=tier,
            scope='Generated supplied post-access continuation. Earlier opening/access costs unknown; no acquired operation or functional outcome.',
            originalProtocol=bound_file(source_dir/'protocol.json'), originalFreeze=bound_file(source_dir/'freeze.json'),
            geometry=source_geometry(task.scene), episodes=[])
        for index,path in enumerate(paths):
            row=read(path); binding=bound_file(path); source_bindings.append(binding)
            if row['parent']!=parent or row['tier']!=tier: raise ValueError('REPLAY_LAYOUT_AND_TIER_MUST_MATCH')
            if index<4 and not any(r==row for r in aggregate): raise ValueError('ORIGINAL_RESULT_DOES_NOT_MATCH_FROZEN_AGGREGATE')
            if index==4 and row['method']!='fixed-narrow-greedy': raise ValueError('FIXED_NARROW_CONTROL_REQUIRED')
            model,checkpoint=_frozen_model(freeze,row)
            engine,requests,states,proof=verify_recorded_episode(task,row,model=model)
            branch=f"recorded-{row['method']}-{row['seed']}"
            record['branches'][branch]=copy.deepcopy(row['snapshot']);record['parents'][branch]={'branchId':'main','position':0}
            record['timeline'][branch]=copy.deepcopy(engine.history)
            outcome=evaluate_procedure_sequence(engine)['outcome']; dims=outcome['dimensions']
            entry=dict(branchId=branch,label=f"{row['method']} · seed {row['seed']}" if row['method']=='learned' else row['method'],
                method=row['method'],seed=row['seed'],checkpoint=checkpoint,originalEpisode=binding,
                nativeActions=len(engine.actions), acceptedStateIds=states, finalStateId=engine.state_id,
                snapshotHash=semantic_digest(row['snapshot']), actionHistoryHash=semantic_digest(engine.history),
                attempts=requests, checkedStop=proof['checkedStop'], endpoint=row['endpoint'], progress=row['progress'],
                neuralRemovedMm3=row['neuralRemovedMm3'],structureRemovedMm3=row['structureRemovedMm3'],
                structureCutSourceMm3={k:v['cumulative'] for k,v in dims['structureCutSourceMm3'].items()},
                modeledSeconds=row['modeledSeconds'], computation={**{k:row['computation'][k] for k in ['physicalCalls','planningPhysicalCalls','executionPhysicalCalls','wallSeconds','tier','canceledRequests']},
                    'previewSkillAttempts':len(row['computation']['previews']),
                    'previewRefusals':sum(r['outcome']['status']=='rejected' for p in row['computation']['previews'] for r in p['receipt']['receipts'])},
                cleanup=dict(debrisMm3=row['endpoint']['debrisMm3'],fluidMm3=row['endpoint']['fluidMm3'],
                    debrisCollectedMm3=engine.metrics()['debrisRemovedMm3']),
                originalSkillIds=row['selectedSkillIds'], proof=proof)
            packet['episodes'].append(entry);receipt['episodes'].append(dict(branchId=branch,originalEpisode=binding,checkpoint=checkpoint,proof=proof))
        packet['contentHash']=semantic_digest(packet);record['sequentialEpisodes']=packet
        selected=packet['episodes'][0];record.update(branchId=selected['branchId'],position=selected['nativeActions'])
        destination=output/'recorded-episodes.ressectionlab';save_case(case,destination,artifacts={'surgical_toolbox':record})
        reopened=load_case(destination);saved=read_case_artifacts(destination)['surgical_toolbox'];scene=from_case(reopened,saved['mapping'])
        if scene.identity!=task.scene.identity or saved['sequentialEpisodes']!=packet: raise ValueError('COLD_PACKAGE_BINDING_CHANGED')
        validate_episode_record(scene,saved,reopened.semantic_hash)
        for entry in packet['episodes']:
            cold=SurgicalEngine.restore(scene,saved['branches'][entry['branchId']],use_cache=False)
            if cold.state_id!=entry['finalStateId'] or semantic_digest(cold.history)!=entry['actionHistoryHash']: raise ValueError('COLD_EXPORTED_HISTORY_CHANGED')
        backend=backend_identity()
        if backend['selected']!='cpp' or backend['binarySha256']!='304dea4eae193865a3fcb7e87a288ba58f4812174bfa70dcd7c6e9367e719e78': raise ValueError('ORIGINAL_COMPILED_KERNEL_REQUIRED')
        receipt.update(passed=True,package=bound_file(destination),caseHash=case.semantic_hash,sourceSceneId=scene.identity,
            packetHash=packet['contentHash'],backend=backend,originalProtocol=packet['originalProtocol'],originalFreeze=packet['originalFreeze'])
        write(output/'episode-index.json',packet)
    except BaseException as error:
        receipt.update(error=str(error),traceback=traceback.format_exc())
    finally:
        for item in source_bindings:
            if sha(item['path'])!=item['sha256']: receipt.update(passed=False,originalSourcesImmutable=False)
        receipt['codeStable']=all(sha(ROOT/name)==value for name,value in receipt['sourceHashes'].items())
        if not receipt['codeStable']: receipt.update(passed=False,codeChangedDuringExport=True)
        receipt['finishedAt']=time.time();write(output/'export-report.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'testcases/overnight-sequential-v1/benchmark-task-v2');p.add_argument('--control-row',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=int,default=200);p.add_argument('--tier',choices=['low','high'],default='low');a=p.parse_args()
    r=export(a.source,a.control_row,a.output,parent=a.parent,tier=a.tier);print(json.dumps({k:r.get(k) for k in ['passed','error','package']}));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
