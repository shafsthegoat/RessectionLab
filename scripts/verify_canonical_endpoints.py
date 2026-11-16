#!/usr/bin/env python3
"""Generate and cold-replay bounded clean/dirty/two-root engineering controls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
FILES=['scripts/generated_cranial_demo_case.py','scripts/verify_canonical_endpoints.py',
    *['src/resectionlab/'+name+'.py' for name in ('surgical_cranial_planning','surgical_joint_cranial_search',
       'surgical_joint_cranial_forest','surgical_consequences','surgical_sequential_task')]]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashes():return {p:sha(ROOT/p) for p in FILES}
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify_variant(output,variant):
    from generated_cranial_demo_case import build_generated_cranial_case,add_lateral_entry_root
    from resectionlab.core import semantic_digest
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies,install_cranial_comparison,validate_cranial_report
    from resectionlab.surgical_cranial_planning_roots import FIELD,validate_registry
    from resectionlab.surgical_sequential_task import SequentialPrimitiveAdapter
    output.mkdir()
    case,base,record=build_generated_cranial_case(1.,include_cleanup=variant!='debris',endpoint_fluid_source=variant=='fluid')
    if variant=='two-roots':add_lateral_entry_root(base,record)
    source_path=output/'source.ressectionlab';save_case(case,source_path,artifacts={'surgical_toolbox':record})
    comparison=compare_cranial_strategies(base,record,planning_horizon=3)
    save(output/'attempted-report.json',comparison['report'])
    install_cranial_comparison(record,comparison,base)
    path=output/'comparison.ressectionlab';save_case(case,path,artifacts={'surgical_toolbox':record})
    loaded=load_case(path);cold=read_case_artifacts(path)['surgical_toolbox'];scene=from_case(loaded,cold['mapping'])
    cold_proof=validate_cranial_report(scene,cold,cold['cranialPlanning'])
    scenes=validate_registry(scene,cold[FIELD]);roots={r['id']:r for r in cold[FIELD]['roots']}
    rows=[]
    for row in cold['cranialPlanning']['alternatives']:
        replay_started=time.monotonic()
        branch=row['branchId'];snapshot=cold['branches'][branch]
        native=SurgicalEngine.restore(scenes[row['rootId']],snapshot,use_cache=False,force_verify=True)
        assert native.snapshot()==snapshot
        adapter=SequentialPrimitiveAdapter(native.scene,cranial_stages=roots[row['rootId']]['sourceSnapshot']['cranialStagesContract'])
        adapter.reset(seed=17);states=[]
        for event in native.history:
            _,_,terminated,truncated,info=adapter.step_action(event['action'])
            assert info['outcome']['status']==event['status'] and adapter.engine.state_id==event['stateAfter']
            assert not terminated and not truncated
            states.append(adapter.engine.state_id)
        before=adapter.engine.state_id
        _,_,terminated,truncated,info=adapter.stop('Bounded source inventory endpoint verification')
        assert adapter.engine.state_id==before==native.state_id
        assert adapter.completion_assessment==row['endpointAssessment']
        assert (info['outcome']['status']=='accepted')==row['validStop']
        assert not truncated
        rows.append(dict(branchId=branch,rootId=row['rootId'],nativeActions=len(native.actions),
            targetMm3=row['goalAssessment']['originalTargetTreatmentMm3'],residualMm3=row['goalAssessment']['residualOriginalTargetMm3'],
            neuralMm3=row['ledger']['neuralRemovedMm3'],endpoint=row['endpointAssessment'],
            recommendationEligible=row['recommendationEligible'],stopOutcome=info['outcome'],
            nativeReplay=True,adapterReplay=True,acceptedStateHash=semantic_digest(states),
            verificationPhysicalCalls=2*len(native.actions),checkedAdapterStopAttempts=1,
            verificationWallSeconds=time.monotonic()-replay_started))
    useful=[r for r in rows if r['targetMm3']>0];assert useful
    if variant in ('clean','two-roots'):
        assert any(r['endpoint']['completedTask'] for r in useful)
        assert all(r['endpoint']['cleanupComplete'] for r in rows)
    else:
        assert all(not r['endpoint']['completedTask'] and not r['recommendationEligible'] for r in useful)
        assert all(variant in r['endpoint']['pending'] for r in useful)
    if variant=='two-roots':assert len({r['rootId'] for r in useful})==2
    save(output/'adapter-native-proof.json',rows)
    return dict(variant=variant,passed=True,sourceCase=str(source_path),case=str(path),caseSha256=sha(path),
        originalCaseHash=case.semantic_hash,sourceSceneId=base.identity,coldValidation=cold_proof,
        initialFluidMeaning='Supplied 1 mm3 nonbleeding source reservoir; native deposition, not an irrigation operation.' if variant=='fluid' else 'No supplied fluid reservoir.',
        decision=comparison['report']['decision'],selectedBranchId=comparison['report']['selectedBranchId'],
        selectionReason=comparison['report']['selectionEvidence']['reason'],
        checkedCandidates=comparison['report']['selectionEvidence']['candidateCount'],
        cost=comparison['report']['cost'],alternatives=rows,adapterProofSha256=sha(output/'adapter-native-proof.json'),
        verificationPhysicalCalls=sum(r['verificationPhysicalCalls'] for r in rows),
        checkedAdapterStopAttempts=len(rows),verificationWallSeconds=sum(r['verificationWallSeconds'] for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=['clean','debris','fluid','two-roots'],default=['clean','debris','fluid','two-roots'])
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases') or output.exists():parser.error('Use a new ignored testcases output directory')
    output.mkdir(parents=True);started=time.monotonic();initial=hashes()
    summary=dict(schema='resectionlab.canonical-endpoint-verification.v1',sourceCommit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sourceHashes=initial,generatedOnly=True,patientOperations=0,variants=[],attempts=[],passed=False)
    try:
        for variant in args.variants:
            try:summary['variants'].append(verify_variant(output/variant,variant))
            except Exception as error:
                summary['attempts'].append(dict(variant=variant,error=str(error),traceback=traceback.format_exc(),
                    incompleteVerificationCallCount=None))
        summary['codeStable']=hashes()==initial
        summary['passed']=not summary['attempts'] and len(summary['variants'])==len(args.variants) and summary['codeStable']
    finally:
        summary['wallSeconds']=time.monotonic()-started;save(output/'summary.json',summary)
    print(json.dumps(dict(passed=summary['passed'],output=str(output),attempts=summary['attempts'])))
    return 0 if summary['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
