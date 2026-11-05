"""Independent source-grid label-location audit of held complete programs.

No planner or sensitivity predictor is imported. Destination masks and coverage
are rebuilt by source-index lookup; literal source-box geometry, native prefixes
and exact CUSA work ledgers independently reconstruct the complete assessment.
"""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_anatomical_encounter import (ROOT, IndependentFootprint, CaptureEngine, certify_trace,
    load_source_context, physical_hash, file_hash, semantic_digest, SurgicalEngine, load_environment, load_workspace)
from verify_strategy_sensitivity import summary, burden_reason, clearance_reason
from resectionlab.core import array_digest
from resectionlab.surgical_planning_observation import project_observation
from resectionlab.surgical_task_admission import admission

SCHEMA='resectionlab.two-tool-context-label-sensitivity.v1'
PUBLIC_SCHEMA='resectionlab.public-two-tool-context-label-sensitivity.v1'
SCOPE=('Declared integer source-grid context-label translations on fixed tissue and target; '
    'held complete nominal tool programs, including withdrawal. No interpolation, '
    'patient error calibration, registration correction, brain shift, clinical risk, '
    'guarantee between tested shifts or new route search. Outside the original source '
    'grid remains unassessed; newly uncovered in-grid cells are unknown, never safe empty space.')


def require(ok,message):
    if not ok:raise ValueError('INDEPENDENT_ANATOMICAL_SENSITIVITY: '+message)


def same(actual,expected,message):
    require(semantic_digest(actual)==semantic_digest(expected),message)


def seal(value,key):
    require(type(value) is dict and value.get(key)==semantic_digest({k:v for k,v in value.items() if k!=key}),key+' differs')


def scenarios(mask,affine,step):
    require(type(step) is int and 0<=step<=2,'integer source shift outside 0–2')
    offsets=[('nominal',[0,0,0])]
    if step:
        for dimension in range(3):
            for sign in (-1,1):
                vector=[0,0,0];vector[dimension]=sign*step
                offsets.append(('grid-'+str(dimension)+('-' if sign<0 else '+'),vector))
    destinations=np.indices(mask.shape).reshape(3,-1);values=[]
    for name,vector in offsets:
        indices=destinations-np.asarray(vector)[:,None]
        valid=np.all((indices>=0)&(indices<np.asarray(mask.shape)[:,None]),axis=0)
        translated=np.zeros(mask.size,bool)
        translated[valid]=mask.ravel()[np.ravel_multi_index(indices[:,valid],mask.shape)]
        moved=destinations+np.asarray(vector)[:,None]
        retained=np.all((moved>=0)&(moved<np.asarray(mask.shape)[:,None]),axis=0)
        values.append({'id':name,'voxelShift':vector,'translationRasMm':(affine[:3,:3]@vector).tolist(),
            'mask':translated.reshape(mask.shape),'known':valid.reshape(mask.shape),
            'outOfFieldContextCells':int(np.count_nonzero(mask.ravel()&~retained))})
    return values


def endpoint(engine,before,tracker,scenario,worked,limit):
    mask=scenario['mask'].ravel();known=scenario['known'].ravel();volume=engine.scene.voxel_mm3
    parts={p:np.asarray(sorted(cells),dtype=np.intp) for p,cells in tracker.seen.items()}
    near={p:indices[mask[indices]] for p,indices in parts.items()}
    union=np.union1d(near['shaft'],near['working']);all_seen=np.union1d(parts['shaft'],parts['working'])
    encountered=np.union1d(all_seen,np.asarray(sorted(worked),dtype=np.intp));unknown=encountered[~known[encountered]]
    work=np.asarray(sorted(worked),dtype=np.intp);work=work[mask[work]]
    removed=np.asarray(sorted(set(engine.state['removed'])-set(before['removed'])),dtype=np.intp)
    historical=int(np.count_nonzero(mask[before['removed']]))
    return {**limit,'sourceContextMm3':int(mask.sum())*volume,'baselineRemovedContextMm3':historical*volume,
        'baselineRetainedContextMm3':(int(mask.sum())-historical)*volume,'newRemovedContextMm3':int(mask[removed].sum())*volume,
        'newContextWorkCells':work.tolist(),'newContextWorkMm3':len(work)*volume,
        'nearSourceCells':union.tolist(),'nearPositionsMm':engine.scene.points(union[:12]).tolist(),'nearMm3':len(union)*volume,
        'shaftNearMm3':len(near['shaft'])*volume,'workingNearMm3':len(near['working'])*volume,
        'unknownEncounteredCells':unknown.tolist(),'unknownEncounteredMm3':len(unknown)*volume,
        'withinLimit':len(union)*volume<=limit['maxNearMm3']}


def exact_progress(engine):
    ledger=engine.state['cusaV3'];require(set(ledger['progressExact'])==set(ledger['progressMm3']),'exact work ledger membership differs')
    return {int(c):Fraction(*pair) for c,pair in ledger['progressExact'].items()}


def replay(baseline,row,controls,worlds,extra,*,check=lambda:None):
    before=copy.deepcopy(baseline.state);engine=baseline.clone();engine.__class__=CaptureEngine;engine._audit_trace=[]
    grid=np.ones(engine.scene.shape,bool);limit=controls['encounter']
    footprint=IndependentFootprint(grid,engine.scene.affine,before['removed'],limit)
    expanded_limit={**limit,'marginMm':limit['marginMm']+extra}
    expanded=IndependentFootprint(grid,engine.scene.affine,before['removed'],expanded_limit)
    initial=previous_progress=exact_progress(engine);worked=set();receipts=[];hashes=[];prefixes=[[] for _ in worlds]
    program=row['actionProgram']
    require(type(program) is list and len(program)<=24 and all(type(a.get('duration')) in (int,float) and
        np.isfinite(a['duration']) and a['duration']>0 for a in program) and sum(a['duration'] for a in program)<=10.+1e-10,'unbounded held program')
    for action in program:
        check();require(action.get('slot') in ('left','right') and not action.get('path') and action.get('operation') in
            ('exchange','inspect','place','move','apply','activate','deactivate','withdraw'),'unsupported held action')
        previous=copy.deepcopy(engine.state['slots']);engine._audit_trace=[]
        result=engine.step(copy.deepcopy(action),check=check)
        finite=action['slot']=='left' and action['operation']=='apply' and result['status']=='partial' and result.get('reasonCode')=='TISSUE_ASPIRATION_IN_PROGRESS' and abs(result['elapsedSeconds']-action['duration'])<=1e-8
        require(result['status']=='accepted' or finite,'held nominal program refused');certify_trace(action,previous,engine.state['slots'],engine._audit_trace)
        require(burden_reason(engine,before,controls) is None,'native task burden violated')
        require(clearance_reason(engine,action,previous,controls,0.,fixed=False) is None,'native task clearance violated')
        current=exact_progress(engine);removed=set(engine.state['removed'])
        # CUSA/Frazier retained progress is monotone; disappearance is allowed
        # only on cell removal. This scoped assertion justifies prefix-ledger work.
        require(all(c in removed or current.get(c,Fraction(0))>=v for c,v in previous_progress.items()),'retained CUSA work decreased or disappeared')
        worked.update(removed-set(before['removed']));worked.update(c for c,v in current.items() if v>initial.get(c,Fraction(0)))
        previous_progress=current
        footprint.observe(action,previous,engine.state['slots'],engine.state['removed'])
        expanded.observe(action,previous,engine.state['slots'],engine.state['removed'])
        hashes.append(physical_hash(engine));receipts.append({k:result.get(k) for k in ('status','reason','completedFraction','elapsedSeconds')})
        for scenario,prefix in zip(worlds,prefixes):prefix.append(semantic_digest(endpoint(engine,before,footprint,scenario,worked,limit)))
    require(not engine.state['holds'] and all(not s['active'] and not s['inserted'] and not s['grip'] for s in engine.state['slots'].values()),'held program did not withdraw')
    measured=summary(engine,before);same(measured,row['summary'],'native summary differs')
    same(hashes,row.get('physicalStateHashes',[]),'native physical prefixes differ');same(receipts,row.get('actionReceipts',[]),'native action receipts differ')
    outcomes=[]
    for scenario,prefix in zip(worlds,prefixes):
        measured_context=endpoint(engine,before,footprint,scenario,worked,limit);violations=[]
        if not measured_context['withinLimit']:violations.append('ANATOMICAL_CONTEXT_LIMIT')
        if measured_context['newContextWorkCells']:violations.append('NEW_CONTEXT_MANIPULATION')
        if measured_context['unknownEncounteredCells']:violations.append('UNKNOWN_CONTEXT_COVERAGE')
        if measured['additionalTargetMm3']<controls['minimumTargetMm3']:violations.append('PROGRESS_THRESHOLD')
        outcomes.append({'scenarioId':scenario['id'],'encounter':measured_context,'contextPrefixHashes':prefix,'violations':violations,'meetsTask':not violations})
    same({k:outcomes[0]['encounter'][k] for k in row['anatomicalEncounter'] if k in outcomes[0]['encounter']},
        {k:v for k,v in row['anatomicalEncounter'].items() if k in outcomes[0]['encounter']},'nominal source context differs')
    same(SurgicalEngine.restore(engine.scene,engine.snapshot(),use_cache=False).snapshot(),engine.snapshot(),'cold native replay differs')
    return {'id':row['id'],'actionReceipts':receipts,'physicalStateHashes':hashes,'completeExecution':True,'withdrawalComplete':True,
        'summary':measured,'outcomes':outcomes,'fixedMarginEncounter':endpoint(engine,before,expanded,worlds[0],worked,expanded_limit)}


def choose(rows,index=0,*,every=False,fixed=False):
    eligible=[]
    for row in rows:
        allowed=all(o['meetsTask'] for o in row['outcomes']) if every else row['outcomes'][index]['meetsTask']
        if fixed:allowed &= row['fixedMarginEncounter']['withinLimit']
        if allowed:eligible.append((row['summary']['elapsedSeconds'],-row['summary']['additionalTargetMm3'],row['id']))
    return min(eligible)[2] if eligible else 'stop'


def verify(study,report,baseline,*,check=lambda:None):
    require(type(baseline) is SurgicalEngine,'authoritative native baseline required');original=baseline.snapshot()
    seal(report,'reportSeal');plan=report['publicPlan'];seal(plan,'publicPlanSeal');seal(study,'studySeal')
    require(report['schema']=='resectionlab.two-tool-comparison.v2','source-context comparison required')
    task={'id':'post-exposure-two-tool-source-context-v2','inputContract':'annotation-assisted-two-tool-source-context-v2',
        'baselineQualification':'existing source-byte-qualified inactive-access admission',
        'allowedOperations':['exchange','inspect','place','move','apply','activate','deactivate','withdraw'],'allowedTools':['cusa','frazier_suction'],
        'maxActions':24,'maxDurationSeconds':10.,'maxCusaActivationSeconds':3.,'maxAdditionalTargetMm3':8.,'maxAdditionalNonTargetMm3':0.,
        'maxPairCandidatesPerRoute':12,'cusaRadiusMm':1.5,'suctionRadiusMm':1.5,'clinicalOutcomes':'unsupported',
        'anatomicalEndpoint':'unique retained supplied context cells in complete nominal axial-tool clearance envelope','maxContextMarginMm':5.,'maxContextNearMm3':1000.}
    same(plan['task'],task,'finite public task contract differs')
    same([report[k] for k in ('sourceSceneId','startStateId','baselineSnapshotHash','baselineActionHash','baselineHistoryHash','prefixActions')],
        [baseline.scene.identity,baseline.state_id,semantic_digest(original),semantic_digest(baseline.actions),semantic_digest(baseline.history),len(baseline.actions)],'source baseline differs')
    mask,affine=load_source_context(baseline,report)
    require(set(baseline.scene.properties)<={'tissuePhysicsV2','cusaV3'} and baseline.state['toolMechanicsRevision']==2 and baseline.state['physicalClockRevision']==1,'unsupported native model')
    public=project_observation(baseline,admission(baseline)['task'],report['sourceEvidence']['availableEvidence']).manifest()
    body={'schema':'resectionlab.two-tool-observation.v2','baseline':public,'taskHash':semantic_digest(plan['task']),
        'anatomicalContext':report['contextBinding']['publicObservation']}
    manifest={**body,'inputHash':semantic_digest(body),'evaluationReferencesExposed':False};same(plan['publicObservation'],manifest,'public source input differs')
    require(set(study)=={'schema','comparisonReportSeal','contextBindingSeal','assessment','nativeReplayVerified','studySeal'} and study['schema']==SCHEMA
        and study['comparisonReportSeal']==report['reportSeal'] and study['contextBindingSeal']==report['contextBinding']['bindingSeal'] and study['nativeReplayVerified'] is True,'study source linkage differs')
    assessment=study['assessment'];seal(assessment,'assessmentSeal');raw=assessment['controls']
    require(type(raw) is dict and set(raw)=={'contextVoxelShift'},'exact label-shift controls required')
    worlds=scenarios(mask,affine,raw['contextVoxelShift']);extra=raw['contextVoxelShift']*float(np.max(np.linalg.norm(affine[:3,:3],axis=0)))
    controls=plan['constraints'];require(extra<=2.+1e-10 and controls['encounter']['marginMm']+extra<=5.+1e-10,'physical shift or margin exceeds software bound')
    configuration={'family':'integer-source-grid-context-label-translations-v1','appliedTo':'context membership on fixed destination tissue cells only',
        'parameterBasis':'declared-uncalibrated-label-location-stress','distribution':None,'patientCalibration':None,'maximumTranslationMm':extra,
        'scenarios':[{k:copy.deepcopy(s[k]) for k in ('id','voxelShift','translationRasMm','outOfFieldContextCells')}|{'maskHash':array_digest(s['mask']),'knownCoverageHash':array_digest(s['known'])} for s in worlds]}
    same(assessment.get('scenarioConfiguration'),configuration,'derived destination masks/coverage or source shifts differ')
    require(type(assessment.get('scenarioCount')) is int and assessment['scenarioCount']==len(worlds),'scenario count differs')
    require(type(plan['alternatives']) is list and 1<=len(plan['alternatives'])<=3 and len({r['id'] for r in plan['alternatives']})==len(plan['alternatives']),'invalid held family')
    require(type(assessment.get('candidates')) is list and [c['id'] for c in assessment['candidates']]==[r['id'] for r in plan['alternatives']],'held candidate inventory differs')
    expected=[]
    for row,observed in zip(plan['alternatives'],assessment['candidates']):
        check();value=replay(baseline,row,controls,worlds,extra,check=check)
        same(observed,value,'independent native/context outcome differs: '+row['id']);expected.append(value)
    nominal=choose(expected);require(nominal==plan['decision']['selectedId'],'source nominal decision differs')
    preferences=[{'scenarioId':s['id'],'selectedId':choose(expected,i)} for i,s in enumerate(worlds)]
    value={'schema':PUBLIC_SCHEMA,'publicObservation':manifest,'comparisonPublicPlanSeal':plan['publicPlanSeal'],'controls':{'contextVoxelShift':raw['contextVoxelShift']},
        'scenarioConfiguration':configuration,'scenarioCount':len(worlds),'candidates':expected,'policies':{'nominal':nominal,'fixedContextMargin':choose(expected,fixed=True),'allDeclaredScenarios':choose(expected,every=True)},
        'fixedMarginDefinition':{'contextMarginMm':controls['encounter']['marginMm']+extra,'isGuarantee':False,
            'scope':'Nominal retained context with an expanded tool envelope; does not bound changed label membership in removed or manipulated cells.'},
        'perScenarioPreference':preferences,'preferenceChanges':[p for p in preferences if p['selectedId']!=nominal],
        'calibratedUncertainty':{'supported':False,'reason':'No patient-specific label-location error calibration is supplied'},'scope':SCOPE}
    value['assessmentSeal']=semantic_digest(value);same(assessment,value,'independent policies, metadata or assessment seal differ')
    require(baseline.snapshot()==original,'authoritative baseline changed');load_source_context(baseline,report)
    return {'schema':'resectionlab.independent-anatomical-sensitivity-audit.v1','passed':True,'studySeal':study['studySeal'],
        'comparisonReportSeal':report['reportSeal'],'sourceContextBindingVerified':True,'baselineUnchanged':True,
        'scenarioCount':len(worlds),'policies':value['policies'],'candidates':expected,
        'scope':'Held complete programs and finite integer destination-label scenarios only; independent source boxes and exact partial-work ledgers. Fixed tissue mechanics; no clinical, patient-calibration, continuous-shift or search-completeness claim.'}


def verify_workspace(workspace,*,source=None,revision=1,check=lambda:None):
    path=Path(workspace).resolve();source=Path(source).resolve() if source else path
    require(path.is_relative_to(ROOT/'testcases') and source.is_relative_to(ROOT/'testcases'),'keep inputs in ignored testcases')
    hashes={str(p):file_hash(p) for p in (path,source)};case,artifacts,_=load_workspace(path);record=artifacts['surgical_toolbox']
    studies=record.get('strategySensitivityStudies',[]);require(type(revision) is int and 1<=revision<=len(studies),'choose a saved sensitivity revision')
    item=studies[revision-1];seal(item,'workspaceSeal')
    require(set(item)=={'schema','revision','comparisonRevision','study','workspaceSeal'} and item['schema']=='resectionlab.strategy-sensitivity-workspace.v1'
        and type(item['revision']) is int and item['revision']==revision and type(item['comparisonRevision']) is int,'workspace study linkage differs')
    comparisons=[r for r in record.get('strategyComparisons',[]) if r['revision']==item['comparisonRevision']]
    require(len(comparisons)==1,'linked comparison missing or duplicated');comparison=comparisons[0];seal(comparison,'workspaceSeal')
    require(comparison['schema']=='resectionlab.two-tool-workspace.v1' and comparison['caseHash']==case.semantic_hash,'workspace comparison identity differs')
    snapshot=record['branches'][comparison['sourceBranchId']];same(semantic_digest(snapshot),comparison['sourceSnapshotHash'],'saved source snapshot differs')
    baseline=SurgicalEngine.restore(load_environment(source),snapshot,use_cache=False,check=check)
    result=verify(item['study'],comparison['report'],baseline,check=check)
    require(all(file_hash(p)==h for p,h in hashes.items()),'native input bytes changed')
    return {**result,'workspace':str(path),'sensitivityRevision':revision,'comparisonRevision':item['comparisonRevision'],'inputSha256':hashes,'inputsUnchanged':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('workspace',type=Path);parser.add_argument('--source',type=Path)
    parser.add_argument('--revision',type=int,default=1);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(args.output.resolve().is_relative_to(ROOT/'testcases') and not args.output.exists(),'preserve evidence under testcases')
    result=verify_workspace(args.workspace,source=args.source,revision=args.revision)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'scenarioCount':result['scenarioCount'],'policies':result['policies']}))
