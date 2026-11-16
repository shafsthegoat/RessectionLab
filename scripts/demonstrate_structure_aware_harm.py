"""Reproduce three bounded scenarios; all generated data belongs in testcases."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def shortcut(out):
    import numpy as np
    from test_surgical_procedure_policy import source,action,prepared
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.surgical_rl_protection import ProtectionSet,ProtectionAnnotation,GuardConfig
    from resectionlab.surgical_procedure_policy import PROPERTY,procedure_admission
    from resectionlab.core import semantic_digest
    scene=source(critical_shortcut=True)
    legacy=SurgicalScene(scene.labels,scene.affine,scene.regions,scene.source,{})
    mask=np.zeros(scene.shape,bool);mask[2,2,20]=True
    protection=ProtectionSet(legacy,[ProtectionAnnotation.from_mask(legacy,'brainstem',mask,
        provenance='Explicit generated witness brainstem membership')],clearance_mm=0.,required=('brainstem',))
    old=SurgicalEngine(legacy,protection=protection,guard_config=GuardConfig(require_critical_coverage=True))
    for command in [action('exchange',instrument='cusa'),action('place',[40.,40.,65.])]:
        assert old.step(command)['status']=='accepted'
    motion=action('apply',[40.,40.,100.],duration=1.)
    old_result=old.step(motion);assert old_result['status']=='accepted'
    new=prepared(scene,x=40.);before=new.snapshot();refused=new.step(motion)
    assert refused['status']=='rejected' and new.snapshot()==before
    raw=copy.deepcopy(scene.properties[PROPERTY]);del raw['structures']['cerebellum']
    raw['policyHash']=semantic_digest({k:v for k,v in raw.items() if k!='policyHash'})
    missing=SurgicalScene(scene.labels,scene.affine,scene.regions,scene.source,{PROPERTY:raw})
    unavailable=procedure_admission(missing);assert not unavailable['available']
    safe=prepared(scene,x=55.)
    assert safe.step(action('apply',[55.,40.,100.],duration=1.))['status']=='accepted'
    assert safe.step(action('withdraw',[55.,40.,65.],duration=1.))['status']=='accepted'
    assert SurgicalEngine.restore(scene,safe.snapshot(),use_cache=False).snapshot()==safe.snapshot()
    evidence={'historicalResult':old_result,'historicalMetrics':old.metrics(),'procedureRefusal':refused,
        'unchangedAcceptedSnapshot':new.snapshot()==before,'missingMaskAdmission':unavailable,
        'legitimateWorkMetrics':safe.metrics(),'scope':'Generated replacement witness, not original patient replay.'}
    evidence['planning']=shortcut_planning(out)
    write(out/'shortcut.json',evidence);write(out/'shortcut-safe-replay.json',safe.snapshot())
    return {'historicalNeuralRemovalMm3':old.metrics()['nonTargetRemovedMm3'],
        'procedureNeuralRemovalMm3':new.metrics()['nonTargetRemovedMm3'],
        'refusal':refused['reason'],'missingMask':unavailable['blockers']}


def shortcut_planning(out):
    from test_structure_aware_cranial_planning import procedure_case
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies,install_cranial_comparison
    from resectionlab.imaging import save_case
    case,scene,record=procedure_case(1000.,cerebellar_shortcut=True)
    save_case(case,out/'shortcut-planning-source.ressectionlab',artifacts={'surgical_toolbox':record})
    result=compare_cranial_strategies(scene,record,planning_horizon=3)
    install_cranial_comparison(record,result,scene)
    report=result['report'];row=next(r for r in report['alternatives'] if r['branchId']==report['selectedBranchId'])
    assert row['goalAssessment']['originalTargetTreatmentMm3']>0
    assert row['ledger']['structureRemovedMm3']['cerebellum']==0
    assert any('PROTECTED_STRUCTURE' in f['message'] for f in report['failures'])
    record.update(branchId=row['branchId'],position=row['nativeActions'])
    path=out/'shortcut-planning-replay.ressectionlab';save_case(case,path,artifacts={'surgical_toolbox':record})
    write(out/'shortcut-planning-report.json',report)
    return {'policyHash':report['procedurePolicy']['policyHash'],'targetMm3':row['goalAssessment']['originalTargetTreatmentMm3'],
        'residualMm3':row['goalAssessment']['residualOriginalTargetMm3'],'neuralRemovedMm3':row['ledger']['neuralRemovedMm3'],
        'cerebellumRemovedMm3':row['ledger']['structureRemovedMm3']['cerebellum'],'replay':str(path),
        'neuralBudgetMm3':1000.,'failures':report['failures']}


def approaches(out):
    from test_structure_aware_cranial_planning import procedure_case
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies,install_cranial_comparison,validate_cranial_report
    from resectionlab.surgical_cranial_planning_roots import FIELD,validate_registry
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_sequence_outcomes import sequence_outcome
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    rows=[]
    for limit in (0.,1.):
        directory=out/('neural-limit-'+str(int(limit)));directory.mkdir()
        case,base,record=procedure_case(limit);source_path=directory/'source.ressectionlab'
        save_case(case,source_path,artifacts={'surgical_toolbox':record})
        start=time.monotonic();result=compare_cranial_strategies(base,record,planning_horizon=3)
        install_cranial_comparison(record,result,base);report=result['report']
        selected=next(r for r in report['alternatives'] if r['branchId']==report['selectedBranchId'])
        assert selected['wholeExitComplete'] and selected['sequenceOutcome']['withdrawalComplete']
        record.update(branchId=selected['branchId'],position=selected['nativeActions'])
        path=directory/'selected-replay.ressectionlab';save_case(case,path,artifacts={'surgical_toolbox':record})
        loaded=load_case(path);saved=read_case_artifacts(path)['surgical_toolbox'];fresh_base=from_case(loaded,saved['mapping'])
        validate_cranial_report(fresh_base,saved,saved['cranialPlanning'])
        scenes=validate_registry(fresh_base,saved[FIELD]);root=next(r for r in saved[FIELD]['roots'] if r['id']==selected['rootId'])
        engine=SurgicalEngine.restore(scenes[root['id']],saved['branches'][selected['branchId']],use_cache=False)
        baseline=SurgicalEngine.restore(scenes[root['id']],root['sourceSnapshot'],use_cache=False)
        assert sequence_outcome(engine,baseline=baseline)==selected['sequenceOutcome']
        write(directory/'report.json',report)
        rows.append({'neuralLimitMm3':limit,'decision':report['decision'],
            'selectedTargetMm3':selected['goalAssessment']['originalTargetTreatmentMm3'],
            'residualMm3':selected['goalAssessment']['residualOriginalTargetMm3'],
            'neuralRemovalMm3':selected['ledger']['neuralRemovedMm3'],'boneRemovalMm3':selected['ledger']['boneRemovedMm3'],
            'preparationMm3':selected['ledger']['preparationNonboneRemovedMm3'],'alternatives':len(report['alternatives']),
            'replay':str(path),'source':str(source_path),'policyHash':report['procedurePolicy']['policyHash'],
            'stopReason':report['stopReason'],'coldOutcomeIdentical':True,'wholeWallSeconds':time.monotonic()-start,
            'searchWallSeconds':report['cost']['wallSeconds'],'physicalCalls':report['cost']['physicalCalls'],'failures':report['failures']})
    assert rows[0]['selectedTargetMm3']<rows[1]['selectedTargetMm3']
    assert rows[0]['neuralRemovalMm3']==0 and rows[1]['neuralRemovalMm3']==1
    return rows


def acquired(out):
    from resectionlab.imaging import load_case
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_procedure_policy import procedure_admission
    path=ROOT/'testcases/cases/UCSF-PDGM-0004-standard-v2.ressectionlab'
    case=load_case(path);scene=from_case(case);admission=procedure_admission(scene)
    assert not admission['available']
    result={'casePath':str(path),'caseHash':case.semantic_hash,'admission':admission,'nativeActions':0,
        'acquiredExecutableValidation':False,'blockers':['Source-matched brainstem/cerebellar estimates are unreviewed.',
            'Inference is cropped; complete intracranial/unknown coverage and independent procedure domain are absent.',
            'No trusted review receipt for this procedure; existing development role retained.']}
    write(out/'acquired-refusal.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'testcases'):parser.error('Store testing datasets under testcases/')
    out.mkdir(parents=True,exist_ok=False)
    summary={'scope':'Three bounded scenarios: generated software evidence, acquired admission refusal; no clinical validation.',
        'shortcut':shortcut(out),'approaches':approaches(out),'acquired':acquired(out)}
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
