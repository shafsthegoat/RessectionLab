#!/usr/bin/env python3
"""Replay an immutable full physical history through a metadata-only source change.

No checkpoint transplantation, new search, positive action, material edit, or
thermal/clock reset is allowed. Source-dependent container IDs may differ;
all checkpoint state fields and physical event/cell IDs remain in comparison.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src')]
from resectionlab.core import array_digest,semantic_digest,thaw_json
from resectionlab.data_root import checked_root,sha256_file,atomic_json
from resectionlab.desktop_bridge import BridgeSession,_Request
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_task_domain import reviewed_domain


def physical_snapshot(snapshot):
    """Ignore only container IDs, never any field inside physical state."""
    return {'actions':copy.deepcopy(snapshot['actions']),
            'physical_state':copy.deepcopy(snapshot['checkpoint']['state'])}


def physical_outcome(outcome):
    result=copy.deepcopy(outcome)
    for key in ('stateBefore','stateAfter','stateId'):
        result.pop(key,None)
    if isinstance(result.get('deltas'),dict):result['deltas'].pop('geometryRevision',None)
    return result


def assert_numeric_identity(old,new):
    """Every source array/grid and material property must remain unchanged."""
    if old.mri.shape!=new.mri.shape or not np.array_equal(old.mri,new.mri) or not np.array_equal(old.affine,new.affine):
        raise ValueError('Primary image or physical grid changed')
    bindings={'mri':array_digest(old.mri),'affine':array_digest(old.affine),'compartments':{},'source_compartments':{}}
    for collection in ('compartments','source_compartments'):
        first,second=getattr(old,collection),getattr(new,collection)
        if set(first)!=set(second):raise ValueError('Source annotation names changed')
        for name,array in first.items():
            if not np.array_equal(array,second[name]):raise ValueError('Source annotation changed: '+name)
            bindings[collection][name]=array_digest(array)
    if (old.brain_mask is None)!=(new.brain_mask is None) or (old.brain_mask is not None and not np.array_equal(old.brain_mask,new.brain_mask)):
        raise ValueError('Source brain array changed')
    if thaw_json(old.metadata.get('surgical_properties',{}))!=thaw_json(new.metadata.get('surgical_properties',{})):
        raise ValueError('Material/physics properties changed')
    return bindings


def checked_input(root,path):
    path=Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Existing input must remain under guarded data root')
    return path


def rebind(history,source,*,root=None,name,evidence=()):
    root=checked_root(root,required_bytes=2*1024**3)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Use a plain immutable output name')
    history,source=checked_input(root,history),checked_input(root,source)
    old_sha,new_sha=sha256_file(history),sha256_file(source)
    old,artifacts,old_workspace=load_workspace(history);new,_,new_workspace=load_workspace(source)
    old_contract=thaw_json(old.metadata.get('data_case',{}));new_contract=thaw_json(new.metadata.get('data_case',{}))
    if new_contract.get('role') not in ('TRAIN','DEVELOPMENT') or old_contract.get('family_id')!=new_contract.get('family_id'):
        raise ValueError('Same declared development source family required')
    bindings=assert_numeric_identity(old,new)
    if reviewed_domain(from_case(new)) is None:raise ValueError('New source requires an explicit reviewed task domain')
    if old_workspace is None or new_workspace is None:raise ValueError('Original native workspace bindings required')
    originals=lambda workspace:[(x.name,x.sha256,x.role) for x in workspace['originals']]
    if originals(old_workspace)!=originals(new_workspace):raise ValueError('Original source-file bytes changed')
    record=artifacts['surgical_toolbox'];branch=record['branchId'];snapshot=record['branches'][branch]
    if record['position']!=len(snapshot['actions']) or snapshot['checkpoint']['position']!=len(snapshot['actions']):
        raise ValueError('Choose an immutable final full-prefix history, not a scrubbed cursor')
    timeline=record['timeline'][branch]
    if len(timeline)!=len(snapshot['actions']):raise ValueError('Complete canonical physical outcome history required')
    if [row.get('action') for row in timeline]!=snapshot['actions']:raise ValueError('Canonical timeline requests differ from full original action prefix')
    references=[]
    for entry in evidence:
        path=checked_input(root,entry);references.append({'path':str(path.relative_to(root)),'sha256':sha256_file(path)})
    directory=root/'integrated-v1/rebinds'/name
    if directory.exists():raise FileExistsError('Preserve prior rebind attempt')
    checked_root(root,required_bytes=2*1024**3);directory.mkdir(parents=True)
    started=time.perf_counter();actions=[];session=BridgeSession(directory/'transfers',run_dir=directory/'runs')
    def call(operation,args):return session.execute(operation,args,_Request(operation),lambda *_:None)
    try:
        identity=call('loadCase',{'path':str(source)})['caseHash']
        current=call('surgicalRehearsal',{'caseHash':identity,'command':'create','mapping':record['mapping']})
        if current['view']['scene']['environment']['physicsHash']!=semantic_digest(record['physicsProfile']):
            raise ValueError('Original physics profile changed before replay')
        for index,(action,prior) in enumerate(zip(snapshot['actions'],timeline)):
            # Only the exact committed original sequence is executed. Previews
            # and prior refusals remain hash-bound evidence, never new actions.
            result=call('surgicalRehearsal',{'caseHash':identity,'command':'step','expectedStateId':current['view']['stateId'],'action':copy.deepcopy(action)})
            outcome=result['outcome']
            canonical=(result['timeline'][index] if outcome['status']!='rejected' and result['position']==index+1 else outcome)
            equal=physical_outcome(canonical)==physical_outcome(prior)
            actions.append({'index':index,'action':action,'original_status':prior['status'],'status':outcome['status'],
                            'reason':outcome.get('reason'),'physical_outcome_equal':equal,
                            'state_id':result['view']['stateId'],'metrics':result['view']['metrics']})
            atomic_json(directory/'actions.json',actions)
            if not equal or outcome['status']=='rejected':
                atomic_json(directory/'failure.json',{'index':index,'status':'stopped_without_search_or_repair',
                    'original_outcome':physical_outcome(prior),'new_outcome':physical_outcome(canonical),'history_sha256':old_sha,'source_sha256':new_sha})
                raise ValueError(f'Original action {index} did not reproduce under persisted guard')
            current=result
        saved=directory/'replay.ressectionlab';call('saveCase',{'caseHash':identity,'path':str(saved)})
        reopened,updated,reopened_workspace=load_workspace(saved)
        saved_record=updated['surgical_toolbox'];saved_snapshot=saved_record['branches'][saved_record['branchId']]
        if physical_snapshot(snapshot)!=physical_snapshot(saved_snapshot):raise ValueError('Physical state differs: clock/thermal/removal/holds/cell IDs cannot be ignored')
        if originals(reopened_workspace)!=originals(new_workspace):raise ValueError('Rebound source originals changed')
        assert_numeric_identity(new,reopened)
        expected=current['view']['stateId'];session.close()
        # Cold public inspect explicitly asks for branch verification, forcing a
        # fresh source-bound replay rather than trusting/transplanting checkpoint.
        cold=BridgeSession(directory/'cold-transfers',run_dir=directory/'cold-runs')
        try:
            cold_id=cold.execute('loadCase',{'path':str(saved)},_Request('cold-load'),lambda *_:None)['caseHash']
            cold_result=cold.execute('surgicalRehearsal',{'caseHash':cold_id,'command':'inspect','branchId':saved_record['branchId']},_Request('cold-inspect'),lambda *_:None)
            if cold_result['view']['stateId']!=expected or cold_result['position']!=len(snapshot['actions']):raise ValueError('Cold public replay changed final state')
        finally:cold.close()
        if sha256_file(history)!=old_sha or sha256_file(source)!=new_sha:raise ValueError('Immutable input changed')
        report={'schema':'integrated-metadata-rebind-v1','history':str(history.relative_to(root)),'history_sha256':old_sha,
            'guarded_source':str(source.relative_to(root)),'source_sha256':new_sha,'saved':str(saved.relative_to(root)),'saved_sha256':sha256_file(saved),
            'saved_case_hash':reopened.semantic_hash,'source_numeric_bindings':bindings,'original_files_preserved':len(reopened_workspace['originals']),
            'full_prefix_action_count':len(snapshot['actions']),'new_actions_added':0,'all_physical_outcomes_exact':True,
            'all_physical_checkpoint_fields_exact':True,'physics_profile_exact':True,'physics_profile_digest':semantic_digest(record['physicsProfile']),'physical_state_digest':semantic_digest(saved_snapshot['checkpoint']['state']),
            'ignored_source_dependent_container_ids':['sceneId','stateId','stateBefore','stateAfter','deltas.geometryRevision'],
            'ignored_physical_state_fields':[],'cold_public_replay_exact':True,'final_state_id':expected,
            'metrics':cold_result['view']['metrics'],'elapsed_physical_seconds':saved_snapshot['checkpoint']['state']['elapsedSeconds'],
            'trajectory_max_per_action_recorded_temperature_C':max(row['metrics'].get('drillMaxTemperatureC',37.) for row in actions),
            'original_preview_refusal_temporal_burden_evidence':references,'scope':'Exact old full physical history under persisted reviewed task box; no positive search, temporal reset, anatomy or clinical promotion',
            'independent_review_required':True,'seconds':time.perf_counter()-started,'generator_sha256':sha256_file(__file__)}
        (directory/'data_integrated_rebind.py.snapshot').write_bytes(Path(__file__).read_bytes())
        atomic_json(directory/'report.json',report);return report
    finally:session.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history',required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--name',required=True);parser.add_argument('--root');parser.add_argument('--evidence',action='append',default=[])
    args=parser.parse_args();print(json.dumps(rebind(args.history,args.source,root=args.root,name=args.name,evidence=args.evidence),indent=2))
