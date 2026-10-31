"""Source-bound preparation alternatives, with explicit public choice and STOP.

Every candidate starts from the same physical source checkpoint. Preparation
cost is separate from a downstream finite two-tool suffix. Missing anatomy and
unknown exterior access remain unknown; no empirical or clinical claim follows.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
import time
from urllib.parse import urlparse, unquote

import numpy as np

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_exposure import DEFAULTS, normalize, normalize_source_ray, project_exposure, prepare, _shortlist
from .surgical_task_admission import source_evidence, verify_source, task_spec, prepare_admission, bind_admission
from .surgical_anatomical_encounter import normalize_limit
from . import surgical_two_tool as two_tool

SCHEMA = 'resectionlab.source-preparation-comparison.v1'
SCOPE = ('Annotation-assisted intracerebral preparation and finite two-tool comparison. '
    'Executed mapped non-target removal is an explicit preparation cost; target progress is counted only in the suffix. '
    'Exterior/skull/dura, complete vessels, optical visibility and functional/clinical outcomes remain unassessed. '
    'STOP and feasibility are conditional on this finite generator, tool pair and assumed mechanics.')
MAX_WALL_SECONDS = 180.
_VALIDATED = OrderedDict()
DEFAULT_CONTROLS = {'sourceRay':None,'target':None,'widthsMm':[0.,.3,1.],
    'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.,'entryRadiusMm':.5,
    'shortSeconds':.25,'longSeconds':1.,'minimumTargetMm3':1.,'requiredEvidence':['brain','target'],'encounter':None}


def require(condition, reason):
    if not condition:raise ValueError('PREPARATION_COMPARISON: '+reason)


def native_collection(case, scene):
    """Resolve only the source-bound native manifest URI, never renderer paths."""
    refs=[r for r in case.source_refs if r.source_id=='source-manifest']
    require(len(refs)==1,'one native source-manifest reference is required')
    ref=refs[0];uri=urlparse(ref.uri)
    require(uri.scheme=='file' and uri.netloc in ('','localhost') and not uri.query and not uri.fragment,
            'a local native source-manifest reference is required')
    path=Path(unquote(uri.path)).resolve()
    require(path.is_absolute() and path.name=='patient-anatomy.json' and
        ref.sha256==(scene.source.get('adapter') or {}).get('sourceManifestSha256'),'native manifest reference differs from source binding')
    return path.parent


def availability(engine, record, collection_path):
    value={'available':False,'reason':None,'scope':SCOPE,'defaults':None,'contextOptions':[]}
    if (engine.scene.source.get('adapter') or {}).get('type')!='native-patient-research-derivative':
        return {**value,'reason':'An acquired native source derivative is required.'}
    if record.get('taskAdmission') is not None:
        return {**value,'reason':'Use the original unadmitted source to compare preparation alternatives; the existing task remains preserved.'}
    if collection_path is None:return {**value,'reason':'The matching native source-manifest reference is unavailable.'}
    if engine.state['holds'] or any(s['active'] or s['inserted'] or s['grip'] for s in engine.state['slots'].values()):
        return {**value,'reason':'Withdraw and release both instruments before comparing preparations.'}
    try:
        source_evidence(engine.scene)
    except ValueError as error:return {**value,'reason':str(error)}
    rows=engine.scene.source['adapter']['sourceManifest']['layers']
    options=[{'layerId':r['id'],'labelValue':r['label_value'],'description':r.get('label') or r['id'],
        'reviewStatus':r['review_status'],'coverage':r.get('coverage','unknown'),'scope':SCOPE}
        for r in rows if r.get('kind') in ('context','lesion-context') and r.get('role')=='context'
        and type(r.get('label_value')) is int and r['label_value']>0
        and r.get('provenance',{}).get('kind')=='observed_source_annotation']
    return {**value,'available':True,'defaults':copy.deepcopy(DEFAULT_CONTROLS),'contextOptions':options,
        'reason':'Original bytes and the complete source checkpoint are verified when a suggestion or comparison is requested.'}


def normalize_controls(raw):
    require(type(raw) is dict and set(raw)==set(DEFAULT_CONTROLS),'exact preparation controls required')
    ray=normalize_source_ray(raw['sourceRay']);widths=raw['widthsMm']
    require(type(widths) is list and 1<=len(widths)<=3 and all(type(w) in (int,float) and np.isfinite(w) and 0<=w<=1 for w in widths)
        and len(set(widths))==len(widths),'one to three distinct displacements from zero to one mm required')
    limits=normalize({**DEFAULTS,'maxNonTargetMm3':raw['maxNonTargetMm3'],'maxActions':raw['maxActions'],
        'maxSeconds':raw['maxSeconds'],'maxCandidates':1,'maxWallSeconds':90.})
    task=task_spec(entry={'centerMm':ray['surfaceMm'],'radiusMm':raw['entryRadiusMm']},target=raw['target'],
        axis=np.asarray(ray['sourceTargetMm'])-ray['surfaceMm'],required_evidence=raw['requiredEvidence'],allow_unreviewed=True)
    require({'brain','target'}<=set(task['requiredEvidence']),'brain and target evidence must remain required')
    for key,low,high in (('shortSeconds',.1,2.),('longSeconds',.1,2.),('minimumTargetMm3',0.,8.)):
        v=raw[key];require(type(v) in (int,float) and np.isfinite(v) and low<=v<=high,'invalid operating control: '+key)
    require(raw['shortSeconds']<raw['longSeconds'],'contact durations must increase')
    encounter=raw['encounter']
    if encounter is not None:
        require(type(encounter) is dict and set(encounter)=={'layerId','marginMm','maxNearMm3'}
            and type(encounter['layerId']) is str and bool(encounter['layerId']),'explicit supplied context selection required')
        encounter={'layerId':encounter['layerId'],**normalize_limit({k:encounter[k] for k in ('marginMm','maxNearMm3')})}
    return {'sourceRay':ray,'target':task['target'],'widthsMm':list(map(float,widths)),
        **{k:limits[k] for k in ('maxNonTargetMm3','maxActions','maxSeconds')},'entryRadiusMm':task['entry']['radiusMm'],
        **{k:float(raw[k]) for k in ('shortSeconds','longSeconds','minimumTargetMm3')},
        'requiredEvidence':task['requiredEvidence'],'encounter':encounter}


def configured_source(engine, *, check):
    candidate=SurgicalEngine.restore(engine.scene,engine.snapshot(),check=check,use_cache=False)
    if candidate.state.get('toolMechanicsRevision')!=2 or candidate.state.get('physicalClockRevision')!=1:
        outcome=candidate.step({'slot':'right','operation':'configure','mechanicsRevision':2,'clockRevision':1,'duration':.1},check=check)
        require(outcome['status']=='accepted','revised model setup refused')
    return candidate


def suggest(engine, collection_path, *, check=lambda:None):
    before=engine.snapshot();verify_source(engine.scene,native_collection=collection_path,check=check)
    candidate=configured_source(engine,check=check)
    controls={**DEFAULTS,'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.,'maxCandidates':1}
    started=time.monotonic()
    def guard():
        check();require(time.monotonic()-started<=30.,'suggestion computation budget exhausted')
    rows=_shortlist(project_exposure(candidate),controls,guard)
    require(bool(rows),'no source ray in this finite proposal family; choose source cells explicitly')
    row=rows[0];guard();require(semantic_digest(engine.snapshot())==semantic_digest(before),'source changed during suggestion')
    verify_source(engine.scene,native_collection=collection_path,check=guard)
    return {'sourceSceneId':engine.scene.identity,'startStateId':before['stateId'],
        'sourceRay':{k:row[k] for k in ('surfaceMm','sourceTargetMm')},
        'target':{'centerMm':row['sourceTargetMm'],'radiusMm':6.},'entryRadiusMm':.5}


def choose_policies(public):
    def cost(row):
        item=row['preparation']['attempts'][row['preparation']['decision']['selectedIndex']]['summary']
        return item['additionalNonTargetMm3'],item['elapsedSeconds'],public.index(row)
    access=[r for r in public if r['strategy'] is not None and r['strategy']['approaches']['routes']]
    corridor=min(access,key=cost) if access else None
    complete=[]
    for row in public:
        plan=row['strategy']
        if plan is None or plan['decision']['selectedId']=='stop':continue
        selected=next(a for a in plan['alternatives'] if a['id']==plan['decision']['selectedId'])
        if selected['feasible'] is True and selected['taskProgressMet'] is True:complete.append(row)
    def total_cost(row):
        choice=next(a for a in row['strategy']['alternatives'] if a['id']==row['strategy']['decision']['selectedId'])
        a,b,c=cost(row)
        return a+choice['summary']['additionalNonTargetMm3'],b+choice['summary']['elapsedSeconds'],c
    return corridor,min(complete,key=total_cost) if complete else None


def incomplete_search_rows(public):
    result=[]
    for row in public:
        plan=row['strategy']
        if plan is None:continue
        reasons=[plan['decision'].get('reason'),*[r.get('reason') for r in plan['rejections']],
            *[r.get('reason') for r in plan['approaches']['rejections']]]
        if any(r in ('COMPUTATION_BUDGET','GEOMETRY_WALL_BUDGET') for r in reasons):result.append(row['id'])
    return result


def generate(engine, raw, collection_path, *, check=lambda:None, progress=lambda *_:None):
    controls=normalize_controls(copy.deepcopy(raw));before=engine.snapshot()
    verification=verify_source(engine.scene,native_collection=collection_path,check=check)
    source=configured_source(engine,check=check);configured=source.snapshot();started=time.monotonic()
    def guard():
        check();require(time.monotonic()-started<=MAX_WALL_SECONDS,'comparison computation budget exhausted; no partial report may publish')
    missing=sorted(set(controls['requiredEvidence'])-set(source_evidence(engine.scene)['availableEvidence']))
    rows=[]
    for index,width in enumerate(controls['widthsMm']):
        guard();progress(index/len(controls['widthsMm']),'Comparing preparation '+str(index+1)+' / '+str(len(controls['widthsMm'])))
        row={'id':'width-'+str(index),'widthMm':width,'preparation':None,'preparedSnapshot':None,
            'preparedAdmission':None,'comparison':None,'reason':None,'missingEvidence':missing}
        if missing:row['reason']='REQUIRED_EVIDENCE_UNAVAILABLE'
        else:
            limits={**DEFAULTS,**{k:controls[k] for k in ('maxNonTargetMm3','maxActions','maxSeconds')},
                'maxCandidates':1,'maxWallSeconds':90.,'lateralSweepMm':width}
            prepared,proof=prepare(source,limits,native_collection=collection_path,source_ray=controls['sourceRay'],check=guard)
            row['preparation']=proof
            if prepared is None:row['reason']='PREPARATION_GENERATOR_REFUSAL' if not proof['publicPlan']['attempts'] else 'PREPARATION_EXECUTION_REFUSED'
            else:
                task=task_spec(entry={'centerMm':controls['sourceRay']['surfaceMm'],'radiusMm':controls['entryRadiusMm']},
                    target=controls['target'],axis=proof['publicPlan']['attempts'][0]['route']['axisRasUnit'],
                    required_evidence=controls['requiredEvidence'],allow_unreviewed=True)
                contract=prepare_admission(prepared,native_collection=collection_path,task=task);bind_admission(prepared,contract)
                downstream=two_tool.defaults(prepared)['defaults']
                for key in ('shortSeconds','longSeconds','minimumTargetMm3'):downstream[key]=controls[key]
                if controls['encounter'] is not None:downstream['encounter']=copy.deepcopy(controls['encounter'])
                comparison=two_tool.generate_comparison(prepared,downstream,check=guard,progress=progress)
                row.update(preparedSnapshot=prepared.snapshot(),preparedAdmission=contract,comparison=comparison,
                    reason=comparison['publicPlan']['decision'].get('reason'))
        rows.append(row)
    public=[{'id':r['id'],'preparation':r['preparation']['publicPlan'] if r['preparation'] else None,
        'strategy':r['comparison']['publicPlan'] if r['comparison'] else None} for r in rows]
    corridor,joint=choose_policies(public)
    def decision(row):return {'selectedPreparationId':row['id'] if row else None,'suffixId':row['strategy']['decision']['selectedId'] if row else 'stop'}
    result={'schema':SCHEMA,'sourceSceneId':engine.scene.identity,'startStateId':before['stateId'],
        'prefixActions':len(engine.actions),'sourceSnapshotHash':semantic_digest(before),'configuredSnapshot':configured,
        'sourceVerification':verification,'controls':controls,'rows':rows,'publicPolicyInputHash':semantic_digest(public),
        'policies':{'corridorFirst':decision(corridor),'completePlanAware':decision(joint)},
        'incompleteRows':incomplete_search_rows(public),'scope':SCOPE}
    targets={r['id'] for r in engine.scene.regions if r.get('target') and r['material']=='tumor'}
    target_removed=sum(int(engine.scene.labels.flat[c]) in targets for c in engine.state['removed'])*engine.scene.voxel_mm3
    result['baselineBurden']={'targetRemovedMm3':target_removed,
        'nonTargetRemovedMm3':len(engine.state['removed'])*engine.scene.voxel_mm3-target_removed,'elapsedSeconds':engine.state['elapsedSeconds']}
    result['reportSeal']=semantic_digest(result)
    # The public policy choices are fixed before downstream native validation.
    for row in rows:
        if row['comparison'] is not None:
            prepared=SurgicalEngine.restore(engine.scene,row['preparedSnapshot'],check=guard)
            bind_admission(prepared,row['preparedAdmission']);two_tool.validate_report(prepared,row['comparison'],check=guard)
    guard();require(semantic_digest(engine.snapshot())==semantic_digest(before),'source changed during comparison')
    verify_source(engine.scene,native_collection=collection_path,check=guard)
    return result


def validate_report(engine, report, *, check=lambda:None):
    check();engine.scene.require_cache_source()
    require(type(report) is dict and report.get('schema')==SCHEMA and report.get('reportSeal')==semantic_digest({k:v for k,v in report.items() if k!='reportSeal'}),
        'comparison content or schema differs')
    require(report['sourceSceneId']==engine.scene.identity and report['startStateId']==engine.state_id
        and report['sourceSnapshotHash']==semantic_digest(engine.snapshot()),'comparison source checkpoint differs')
    verification=verify_source(engine.scene,native_collection=report['sourceVerification']['nativeCollectionPath'],check=check)
    require(semantic_digest(verification)==semantic_digest(report['sourceVerification']),'source verification differs')
    key=(engine.scene.identity,semantic_digest(engine.snapshot()),semantic_digest(report))
    if key in _VALIDATED:
        check();_VALIDATED.move_to_end(key);return True
    expected=generate(engine,report['controls'],report['sourceVerification']['nativeCollectionPath'],check=check)
    require(semantic_digest(expected)==semantic_digest(report),'canonical public preparation comparison differs')
    check();_VALIDATED[key]=True
    while len(_VALIDATED)>16:_VALIDATED.popitem(last=False)
    return True


def materialize(engine, report, candidate_id, phase, *, check=lambda:None):
    validate_report(engine,report,check=check)
    require(phase in ('prepared','withdraw'),'unknown preparation checkpoint')
    row=next((r for r in report['rows'] if r['id']==candidate_id),None)
    require(row is not None and row['preparedSnapshot'] is not None,'selected preparation is not complete')
    prepared=SurgicalEngine.restore(engine.scene,row['preparedSnapshot'],check=check,use_cache=False)
    bind_admission(prepared,row['preparedAdmission']);candidate=prepared
    if phase=='withdraw':
        comparison=row['comparison'];selected=comparison['publicPlan']['decision']['selectedId']
        require(selected!='stop','no complete progress-qualified two-tool sequence is available')
        chosen=next(a for a in comparison['publicPlan']['alternatives'] if a['id']==selected)
        require(chosen['feasible'] is True and chosen['taskProgressMet'] is True,'complete suffix does not meet the declared task')
        candidate,_,_=two_tool.materialize(prepared,comparison,selected,check=check)
    require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in candidate.state['slots'].values()),'withdrawal incomplete')
    check()
    return candidate,{'candidateId':candidate_id,'phaseId':phase,'reportSeal':report['reportSeal'],
        'startStateId':engine.state_id,'preparationStateId':prepared.state_id,'resultStateId':candidate.state_id,
        'nativeMatchesPublicPrediction':True,'withdrawalComplete':True,
        'preparationActions':len(prepared.actions)-len(engine.actions),'suffixActions':len(candidate.actions)-len(prepared.actions)}
