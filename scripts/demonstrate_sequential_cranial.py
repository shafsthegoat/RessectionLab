"""Self-contained generated opening/preparation/work/exit comparison.

No test modules, acquired images, or acquisition. Outputs are append-only under
ignored testcases. A summary is written even when a stage fails.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_source_case(case, record, destination):
    # Bundle transport timestamps can differ while scientific/native content
    # is identical. Compare both complete case and artifact content before save.
    import os
    from resectionlab.imaging import load_case, read_case_artifacts, save_case
    from resectionlab.core import semantic_digest
    expected = semantic_digest({'surgical_toolbox':record})
    for candidate in sorted(destination.parents[2].glob('*/harm-*/source.ressectionlab')):
        if candidate == destination or not candidate.exists(): continue
        try:
            if load_case(candidate).semantic_hash == case.semantic_hash and semantic_digest(read_case_artifacts(candidate)) == expected:
                canonical = candidate.resolve()
                destination.symlink_to(os.path.relpath(canonical,destination.parent))
                return {'reusedIdenticalContent':True,'canonicalPath':str(canonical),
                    'caseHash':case.semantic_hash,'artifactHash':expected}
        except (ValueError,KeyError,OSError): continue
    save_case(case,destination,artifacts={'surgical_toolbox':record})
    return {'reusedIdenticalContent':False,'canonicalPath':str(destination),
        'caseHash':case.semantic_hash,'artifactHash':expected}


def replay_adapter(scene, root, engine):
    from resectionlab.surgical_sequential_task import SequentialPrimitiveAdapter,COMMON_CONFIG,endpoint_assessment
    from resectionlab.core import semantic_digest
    env=SequentialPrimitiveAdapter(scene,task_config=COMMON_CONFIG,
        cranial_stages=root['sourceSnapshot']['cranialStagesContract'])
    env.reset(seed=17)
    matched=[]
    for index,event in enumerate(engine.history):
        _,reward,terminated,truncated,info=env.step_action(event['action'])
        outcome=info['outcome']
        assert outcome['status']==event['status'], (index,outcome,event['status'])
        assert env.engine.state_id==event['stateAfter'], (index,env.engine.state_id,event['stateAfter'])
        assert not truncated, (index,info)
        assert not terminated or index==len(engine.history)-1, (index,info)
        matched.append(dict(actionIndex=index,status=outcome['status'],stateId=env.engine.state_id,reward=reward))
    stop_outcome=None
    if not env.terminated:
        before_stop=env.engine.state_id
        _,_,terminated,truncated,info=env.stop('Bounded inventory or horizon leaves original target residual')
        assert info['outcome']['status']=='accepted' and terminated and not truncated
        stop_outcome=info['outcome']
        assert env.engine.state_id==before_stop
    assert stop_outcome is not None, 'The shared task requires explicit checked STOP'
    native=endpoint_assessment(engine,COMMON_CONFIG,stopped=True)
    adapter=endpoint_assessment(env.engine,COMMON_CONFIG,stopped=True)
    assert native==adapter==env.completion_assessment
    assert env.engine.state==engine.state
    return dict(configHash=COMMON_CONFIG.fingerprint,primitiveTransitions=len(matched),
        matchedStatesHash=semantic_digest(matched),allAcceptedOutcomesAndStatesIdentical=True,
        terminalClassification=native['classification'],finalStateId=engine.state_id,
        adapterTerminationReason=env.termination_reason,checkedStopOutcome=stop_outcome,attempts=len(env.attempts),transitions=matched)


def build_variant(output,neural_limit):
    from generated_cranial_demo_case import build_generated_cranial_case
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies,install_cranial_comparison,validate_cranial_report
    from resectionlab.surgical_cranial_planning_roots import FIELD,validate_registry
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_sequential_task import COMMON_CONFIG,endpoint_assessment
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    from resectionlab.core import semantic_digest
    output.mkdir()
    case,scene,record=build_generated_cranial_case(neural_limit)
    source=output/'source.ressectionlab';source_reuse=save_source_case(case,record,source)
    write(output/'source-provenance.json',source_reuse)
    result=compare_cranial_strategies(scene,record,planning_horizon=3)
    install_cranial_comparison(record,result,scene);report=result['report']
    selected=next(r for r in report['alternatives'] if r['branchId']==report['selectedBranchId'])
    record.update(branchId=selected['branchId'],position=selected['nativeActions'])
    package=output/'comparison.ressectionlab';save_case(case,package,artifacts={'surgical_toolbox':record})
    write(output/'comparison-report.json',report)
    endpoints=[]
    for engine,row in result['branches']:
        endpoint=endpoint_assessment(engine,COMMON_CONFIG,stopped=True)
        assert endpoint['validWithdrawal'] and endpoint['cleanupComplete'], endpoint
        assert endpoint['debrisMm3']==0., endpoint
        assert abs(row['recordedGeometry']['cleanup']['debrisRemovedMm3']-236.)<1e-8
        assert endpoint['classification'] in ('successful-treatment','valid-stop-with-residual'), endpoint
        assert row['goalAssessment']['originalTargetTreatmentMm3']>0.
        assert row['ledger']['structureRemovedMm3']['motor-cortex']==row['ledger']['structureRemovedMm3']['corticospinal-tract']
        root=next(r for r in record[FIELD]['roots'] if r['id']==row['rootId'])
        parity=replay_adapter(engine.scene,root,engine)
        write(output/(row['branchId']+'-adapter-parity.json'),parity)
        endpoints.append(dict(branchId=row['branchId'],endpoint=endpoint,snapshotHash=semantic_digest(engine.snapshot()),historyHash=semantic_digest(engine.history),adapterParity=True))
    loaded=load_case(package);saved=read_case_artifacts(package)['surgical_toolbox'];restored=from_case(loaded,saved['mapping'])
    validate_cranial_report(restored,saved,saved['cranialPlanning']);scenes=validate_registry(restored,saved[FIELD])
    for proof in endpoints:
        row=next(r for r in report['alternatives'] if r['branchId']==proof['branchId'])
        cold=SurgicalEngine.restore(scenes[row['rootId']],saved['branches'][row['branchId']],use_cache=False)
        assert semantic_digest(cold.snapshot())==proof['snapshotHash']
        assert semantic_digest(cold.history)==proof['historyHash']
        assert endpoint_assessment(cold,COMMON_CONFIG,stopped=True)==proof['endpoint']
    write(output/'endpoints.json',dict(config=asdict(COMMON_CONFIG),configHash=COMMON_CONFIG.fingerprint,alternatives=endpoints))
    return dict(neuralLimitMm3=neural_limit,package=str(package),source=str(source),sourceReuse=source_reuse,caseHash=case.semantic_hash,
        policyHash=report['procedurePolicy']['policyHash'],selectedBranchId=report['selectedBranchId'],comparisonSeal=report['comparisonSeal'],
        decision=report['decision'],stopReason=report['stopReason'],
        alternatives=[dict(branchId=r['branchId'],nativeActions=r['nativeActions'],targetMm3=r['goalAssessment']['originalTargetTreatmentMm3'],
            residualMm3=r['goalAssessment']['residualOriginalTargetMm3'],neuralMm3=r['ledger']['neuralRemovedMm3'],boneMm3=r['ledger']['boneRemovedMm3'],
            preparationMm3=r['ledger']['preparationNonboneRemovedMm3'],modeledSeconds=r['ledger']['modeledElapsedSeconds'],
            structureRemovedMm3=r['ledger']['structureRemovedMm3'],cleanup=r['recordedGeometry']['cleanup'],
            endpoint=next(e['endpoint'] for e in endpoints if e['branchId']==r['branchId'])) for r in report['alternatives']],
        searchCost=report['cost'],coldNativeReplayIdentical=True,adapterParity=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-only',action='store_true',help='Explicitly select the supported generated demo (default).')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):parser.error('Keep generated outputs under ignored testcases/')
    output.mkdir(parents=True,exist_ok=False)
    summary=dict(schema='resectionlab.sequential-cranial-demonstration.v1',sourceCommit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        generatedOnly=True,clinicalValidation=False,startedAt=time.time(),variants=[],passed=False,
        scope='Generated opening-to-withdrawal from declared post-scalp start; earlier scalp access costs unknown. Native constrained search, no learned-policy claim.')
    try:
        for limit in (0.,1.):summary['variants'].append(build_variant(output/('harm-'+str(int(limit))),limit))
        low,high=summary['variants'];selected=lambda v:next(r for r in v['alternatives'] if r['branchId']==v['selectedBranchId'])
        assert selected(low)['neuralMm3']==0. and selected(high)['neuralMm3']==1.
        assert selected(low)['targetMm3']<selected(high)['targetMm3']
        assert len(high['alternatives'])>=2
        summary.update(harmChangedSelection=True,passed=True)
    except BaseException as error:
        summary.update(error=str(error),traceback=traceback.format_exc())
    finally:
        summary['finishedAt']=time.time()
        summary['sourceHashes']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/generated_cranial_demo_case.py','scripts/demonstrate_sequential_cranial.py','src/resectionlab/surgical_cranial_geometry.py','src/resectionlab/surgical_cranial_planning.py','src/resectionlab/surgical_sequential_task.py']}
        write(output/'summary.json',summary)
    print(json.dumps(dict(passed=summary['passed'],output=str(output),error=summary.get('error'))))
    return 0 if summary['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
