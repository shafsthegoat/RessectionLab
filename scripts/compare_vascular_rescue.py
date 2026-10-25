#!/usr/bin/env python3
"""Execute three source-preserving rescue strategies and save a native workspace."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace, save_workspace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path,help='Complete canonical action receipts for the retained vascular branch')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if 'testcases' not in args.output.resolve().parts or args.output.exists():
        parser.error('Use a new output directory under testcases/')
    args.output.mkdir(parents=True)
    from resectionlab.surgical_strategy_planning import compare_rescue
    before=hashlib.sha256(args.case.read_bytes()).hexdigest()
    case,artifacts,workspace=load_workspace(args.case)
    scene=load_environment(args.case)
    record=artifacts['surgical_toolbox']
    snapshot=record['branches']['full-vascular']
    reference=[json.loads(line) for line in args.reference.read_text().splitlines()]
    if snapshot['actions'] != [row['action'] for row in reference]:
        raise ValueError('Canonical reference differs from retained action log')
    injury=next(i for i,row in enumerate(reference) if any(e.get('kind')=='strand_transection' for e in row.get('events',[])))+1
    prefix={k:copy.deepcopy(v) for k,v in snapshot.items() if k!='checkpoint'}
    prefix.update(actions=snapshot['actions'][:injury],stateId=reference[injury-1]['stateAfter'])
    started=time.perf_counter()
    def check():
        if time.perf_counter()-started>600:raise TimeoutError('Whole strategy preparation exceeded 600 seconds')
    print(json.dumps(dict(stage='verify_common_prefix',actions=injury)),flush=True)
    engine=SurgicalEngine.restore(scene,prefix,use_cache=True,check=check)
    branches,comparison=compare_rescue(engine,record,acknowledge_unknowns=True,check=check,
        progress=lambda fraction,message:print(json.dumps(dict(progress=fraction,message=message)),flush=True))
    record['branches']['rescue-decision']=engine.snapshot()
    record.setdefault('parents',{})['rescue-decision']={'branchId':'full-vascular','position':injury}
    for candidate,report in branches:
        branch_id='strategy-'+report['method']
        report['branchId']=branch_id
        record['branches'][branch_id]=candidate.snapshot()
        record['parents'][branch_id]={'branchId':'rescue-decision','position':injury}
    comparison['comparisonSeal']=semantic_digest(comparison)
    record['planning']=comparison
    record.update(branchId='rescue-decision',position=injury)
    record.pop('timeline',None)
    destination=args.output/'rescue-strategies.ressectionlab'
    save_workspace(case,destination,sources=workspace['sources'],imaging=workspace['imagingState'],
        episode=workspace['episode'],selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts)
    # Reopen the serialized workspace and recheck every retained decision/action
    # binding. This is separate from the executed suffix replay checks above.
    from resectionlab.surgical_planning import validate_saved_comparison
    reopened_case,reopened_artifacts,_=load_workspace(destination)
    validate_saved_comparison(scene,reopened_artifacts['surgical_toolbox'],reopened_artifacts['surgical_toolbox']['planning'])
    if reopened_case.semantic_hash!=case.semantic_hash or hashlib.sha256(args.case.read_bytes()).hexdigest()!=before:
        raise ValueError('Source anatomy or original workspace changed')
    report=dict(schema='resectionlab.operative-strategy-run.v1',case=str(args.case.resolve()),
        sourceSha256=before,sourcePreserved=True,savedWorkspace=str(destination.resolve()),
        savedSha256=hashlib.sha256(destination.read_bytes()).hexdigest(),commonPrefixActions=injury,
        commonStateId=engine.state_id,wallSeconds=time.perf_counter()-started,comparison=comparison)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='saved',workspace=str(destination),wallSeconds=report['wallSeconds'],
        results=[dict(label=p['label'],bleeding=p['incrementalEndpoints']['bleeding_generated_mm3'],
            seconds=p['incrementalEndpoints']['elapsed_seconds'],invalid=p['invalidActions']) for p in comparison['alternatives']])),flush=True)


if __name__=='__main__':main()
