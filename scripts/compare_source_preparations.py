"""Compare a declared source-ray/width matrix at one acquired native checkpoint.

Both policies use the same complete held public table. This is a bounded offline
engineering experiment, not an algorithm-cost advantage or clinical assessment.
Independent geometry audits remain separate from this native validation.
"""
from pathlib import Path
import argparse,copy,json,sys,time,hashlib,re
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from resectionlab.core import semantic_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_task_admission import verify_source,prepare_admission,bind_admission,task_spec,source_evidence
from resectionlab.surgical_exposure import DEFAULTS,prepare,normalize,normalize_source_ray
from resectionlab.surgical_two_tool import defaults,generate_comparison,validate_report
from prepare_source_exposure import save_native
from resectionlab.surgical_anatomical_encounter import normalize_limit
import numpy as np


def validate_operating_design(design):
    task=design['task'];downstream=design['downstream'];ray=design['rays'][0]['sourceRay']
    if task['allowUnreviewedEvidence'] is not True:raise ValueError('Explicit research acceptance required')
    task_spec(entry={'centerMm':ray['surfaceMm'],'radiusMm':task['entryRadiusMm']},
        target={'centerMm':task['targetCenterMm'],'radiusMm':task['targetRadiusMm']},
        axis=np.asarray(ray['sourceTargetMm'])-ray['surfaceMm'],required_evidence=task['requiredEvidence'],allow_unreviewed=True)
    for key,low,high in (('shortSeconds',.1,2.),('longSeconds',.1,2.),('minimumTargetMm3',0.,8.)):
        value=downstream[key]
        if type(value) not in (int,float) or not np.isfinite(value) or not low<=value<=high:
            raise ValueError('Operating design exceeds finite task bounds: '+key)
    if downstream['shortSeconds']>=downstream['longSeconds']:raise ValueError('Short contact must precede longer contact')
    if type(downstream['contextLabelValue']) is not int or downstream['contextLabelValue']<=0:raise ValueError('Explicit categorical context required')
    normalize_limit({'marginMm':downstream['contextMarginMm'],'maxNearMm3':downstream['contextMaxNearMm3']})

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
    joint=min(complete,key=total_cost) if complete else None
    return corridor,joint


def incomplete_search_rows(public):
    result=[]
    for row in public:
        plan=row["strategy"]
        if plan is None:continue
        reasons=[plan["decision"].get("reason"),*[r.get("reason") for r in plan["rejections"]],
                 *[r.get("reason") for r in plan["approaches"]["rejections"]]]
        if any(reason in ("COMPUTATION_BUDGET","GEOMETRY_WALL_BUDGET") for reason in reasons):result.append(row["id"])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-collection',type=Path,required=True)
    parser.add_argument('--reuse-case',type=Path,action='append',default=[])
    parser.add_argument('--acknowledge-research',action='store_true',required=True)
    args=parser.parse_args();design_path=args.design.resolve();output=args.output.resolve();native=args.native_collection.resolve()
    if not all(p.is_relative_to(ROOT/'testcases') for p in [design_path,output,native,*[p.resolve() for p in args.reuse_case]]) or output.exists():
        parser.error('Reuse ignored testcases inputs and choose a fresh testcases output')
    design_bytes=design_path.read_bytes();design_sha256=hashlib.sha256(design_bytes).hexdigest()
    design=json.loads(design_bytes)
    assert design['schema']=='resectionlab.matched-exposure-development-design.v1'
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=(ROOT/design['source']).resolve();assert source.is_relative_to(ROOT/'testcases') and digest(source)==design['sourceSha256']
    assert type(design['rays']) is list and 1<=len(design['rays'])<=4
    assert len({r['id'] for r in design['rays']})==len(design['rays'])
    assert all(type(r['id']) is str and re.fullmatch('[a-z][a-z0-9_-]{0,31}',r['id']) for r in design['rays'])
    for seed in design['rays']:normalize_source_ray(seed['sourceRay'])
    assert type(design['widthsMm']) is list and 1<=len(design['widthsMm'])<=3 and len(set(design['widthsMm']))==len(design['widthsMm'])
    assert all(type(w) in (int,float) and 0<=w<=1. for w in design['widthsMm'])
    assert len(design['rays'])*len(design['widthsMm'])<=12
    assert set(design['controls'])==set(DEFAULTS)-{'lateralSweepMm'} and design['controls']['maxCandidates']==1
    normalize({**design['controls'],'lateralSweepMm':0.})
    validate_operating_design(design)
    implementation_paths=[Path(__file__),*[ROOT/'src/resectionlab'/name for name in ('surgical_exposure.py','surgical_engine.py',
        'surgical_motion.py','surgical_planning_observation.py','surgical_two_tool.py','surgical_task_admission.py','surgical_anatomical_encounter.py')]]
    implementation={str(p.relative_to(ROOT)):digest(p) for p in implementation_paths}
    output.mkdir(parents=True)
    case,artifacts,workspace=load_workspace(source);scene=load_environment(source)
    original=copy.deepcopy(artifacts['surgical_toolbox'])
    if any(original.get(key) for key in ('taskAdmission','strategyComparisons','interactiveApproaches')):
        raise ValueError('Preserve existing task/studies; use their original unadmitted source')
    baseline=SurgicalEngine.restore(scene,original['branches'][original['branchId']],use_cache=False)
    verification=verify_source(scene,native_collection=native)
    missing=set(design['task']['requiredEvidence'])-set(source_evidence(scene)['availableEvidence'])
    if missing:raise ValueError('Required source evidence unavailable: '+', '.join(sorted(missing)))
    if baseline.state.get('toolMechanicsRevision')!=2 or baseline.state.get('physicalClockRevision')!=1:
        assert baseline.step(dict(slot='right',operation='configure',mechanicsRevision=2,clockRevision=1,duration=.1))['status']=='accepted'
    before=baseline.snapshot();public=[];private=[];reports=[]
    reuse=[]
    for candidate in args.reuse_case:
        p=candidate.resolve();c,a,_=load_workspace(p);r=a['surgical_toolbox']
        if c.semantic_hash!=case.semantic_hash:continue
        reuse.append((p,r,digest(p)))

    def save_checkpoint(engine,contract,path):
        a=copy.deepcopy(artifacts);record=a['surgical_toolbox']
        def unused(prefix):
            number=len(original['branches'])
            while prefix+str(number) in record['branches']:number+=1
            return prefix+str(number)
        source_branch=unused('preparation-source-')
        record['branches'][source_branch]=copy.deepcopy(before)
        record.setdefault('parents',{})[source_branch]={'branchId':original['branchId'],'position':len(original['branches'][original['branchId']]['actions'])}
        record.setdefault('timeline',{})[source_branch]=copy.deepcopy(baseline.history)
        branch=source_branch
        if contract is not None:
            branch=unused('prepared-task-')
            record['branches'][branch]=engine.snapshot();record['parents'][branch]={'branchId':source_branch,'position':len(baseline.actions)}
            record['timeline'][branch]=copy.deepcopy(engine.history);record['taskAdmission']=contract
        record.update(branchId=branch,position=len(engine.actions))
        if any(semantic_digest(record['branches'][k])!=semantic_digest(v) for k,v in original['branches'].items()):
            raise ValueError('Historical source branch changed')
        save_native(case,a,workspace,path)
        loaded,saved,_=load_workspace(path);assert loaded.semantic_hash==case.semantic_hash and saved['surgical_toolbox']==record
        cold=SurgicalEngine.restore(load_environment(path),record['branches'][branch],use_cache=False)
        assert cold.snapshot()==engine.snapshot()
        if contract is not None:bind_admission(cold,contract)
        return path

    configured=next((p for p,r,_ in reuse if not r.get('taskAdmission')
        and semantic_digest(r['branches'][r['branchId']])==semantic_digest(before)),None)
    if configured is None:configured=save_checkpoint(baseline,None,output/'configured-source.ressectionlab')
    for seed in design['rays']:
        if 'seedProof' in seed:
            path=(ROOT/seed['seedProof']).resolve();assert path.is_relative_to(ROOT/'testcases') and digest(path)==seed['seedSha256']
        for width in design['widthsMm']:
            identity=seed['id']+'-w'+str(width).replace('.','p');folder=output/identity;folder.mkdir()
            started=time.perf_counter();controls={**DEFAULTS,**design['controls'],'lateralSweepMm':width}
            selected,proof=prepare(baseline,controls,native_collection=native,source_ray=seed['sourceRay'])
            (folder/'preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
            row={'id':identity,'sourceRay':seed['sourceRay'],'lateralSweepMm':width,'preparation':proof['publicPlan'],'strategy':None}
            detail={'id':identity,'preparationSeconds':time.perf_counter()-started,'case':str(configured),'caseReused':True}
            if selected is not None:
                assert selected.actions[:len(baseline.actions)]==baseline.actions
                attempt=proof['publicPlan']['attempts'][0]
                task=task_spec(entry={'centerMm':seed['sourceRay']['surfaceMm'],'radiusMm':design['task']['entryRadiusMm']},
                    target={'centerMm':design['task']['targetCenterMm'],'radiusMm':design['task']['targetRadiusMm']},axis=attempt['route']['axisRasUnit'],
                    required_evidence=design['task']['requiredEvidence'],allow_unreviewed=True)
                contract=prepare_admission(selected,native_collection=native,task=task);bind_admission(selected,contract)
                found=next((p for p,r,_ in reuse if semantic_digest(r['branches'][r['branchId']])==semantic_digest(selected.snapshot()) and semantic_digest(r.get('taskAdmission'))==semantic_digest(contract)),None)
                native_case=found or save_checkpoint(selected,contract,folder/'admitted-preparation.ressectionlab')
                detail.update(case=str(native_case),caseReused=found is not None)
                settings=defaults(selected);raw=settings['defaults']
                for key in ('shortSeconds','longSeconds','minimumTargetMm3'):raw[key]=design['downstream'][key]
                contexts=[r for r in settings['contextOptions'] if r['labelValue']==design['downstream']['contextLabelValue']]
                if len(contexts)!=1:raise ValueError('Exactly one eligible supplied context must match the declared source label')
                context=contexts[0]
                raw['encounter']={'layerId':context['layerId'],'marginMm':design['downstream']['contextMarginMm'],'maxNearMm3':design['downstream']['contextMaxNearMm3']}
                phase=time.perf_counter();comparison=generate_comparison(selected,raw);detail['strategySeconds']=time.perf_counter()-phase
                (folder/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
                row['strategy']=comparison['publicPlan'];reports.append((selected,comparison,folder))
            public.append(row);private.append(detail)
            (output/'progress.json').write_text(json.dumps(private,indent=2)+'\n')
            assert baseline.snapshot()==before and digest(source)==design['sourceSha256']
            print(json.dumps({'id':identity,'prepared':selected is not None,'decision':row['strategy']['decision'] if row['strategy'] else 'PREPARATION_STOP','seconds':detail}),flush=True)

    corridor,joint=choose_policies(public)
    decisions={'schema':'resectionlab.matched-exposure-development-policies.v1','publicRows':public,
        'corridorFirst':{'selectedPreparationId':corridor['id'] if corridor else None,'suffixId':corridor['strategy']['decision']['selectedId'] if corridor else 'stop'},
        'completePlanAware':{'selectedPreparationId':joint['id'] if joint else None,'suffixId':joint['strategy']['decision']['selectedId'] if joint else 'stop'},
        'scope':design['scope'],'budget':design['budget']}
    decisions['policySeal']=semantic_digest(decisions)
    (output/'sealed-policies.json').write_text(json.dumps(decisions,indent=2)+'\n')
    # Independent/native downstream validation follows both sealed public choices.
    for engine,comparison,folder in reports:
        start=time.perf_counter();validate_report(engine,comparison)
        (folder/'native-validation.json').write_text(json.dumps({'passed':True,'reportSeal':comparison['reportSeal'],'elapsedSeconds':time.perf_counter()-start},indent=2)+'\n')
    assert digest(source)==design['sourceSha256'] and baseline.snapshot()==before
    if digest(design_path)!=design_sha256:raise ValueError('Design bytes changed during the comparison')
    if any(digest(ROOT/path)!=value for path,value in implementation.items()):raise ValueError('Implementation changed during the comparison')
    assert all(digest(p)==h for p,_,h in reuse)
    incomplete=incomplete_search_rows(public)
    result={'passed':True,'designSha256':design_sha256,'policySeal':decisions['policySeal'],'sourceUnchanged':True,'baselineUnchanged':True,'designUnchanged':True,
            'implementationSha256':implementation,'implementationUnchanged':True,
            'rows':private,'completePlanFeasibilityChanged':bool(joint and (not corridor or corridor['strategy']['decision']['selectedId']=='stop')) if not incomplete else None,
            'comparisonConclusive':not incomplete,'incompleteRows':incomplete,'scope':design['scope'],
            'allDeclaredCellsRetained':len(public)==len(design['rays'])*len(design['widthsMm']),'independentAuditsPending':True}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__ == "__main__":main()
