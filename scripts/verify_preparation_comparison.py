"""Independent audit of held preparation programs and installed native branches.

No preparation proposer, workspace validator or two-tool compiler is imported.
Empty generator refusals are not proofs of infeasibility or search completeness.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_source_exposure as exposure
import verify_two_tool_geometry as geometry
import verify_anatomical_encounter as encounter
from verify_anatomical_encounter import ROOT, SurgicalEngine, load_environment, load_workspace, file_hash, semantic_digest
from resectionlab.surgical_task_admission import verify_source, source_evidence, validate_admission

SCOPE = ('Held native programs, source bindings, literal costs and public policy choices only. '
    'Empty preparation-generator refusals and unexported alternatives are not independently qualified; '
    'no exhaustive-generator STOP, clinical, complete-anatomy or empirical-mechanics claim.')


def require(ok, reason):
    if not ok: raise ValueError('INDEPENDENT_PREPARATION_COMPARISON: '+reason)


def seal(value, field):
    require(type(value) is dict and value.get(field) == semantic_digest({k:v for k,v in value.items() if k != field}), field+' differs')


def equal(a, b, reason):
    require(semantic_digest(a) == semantic_digest(b), reason)


def replay(engine, program):
    require(type(program) is list and len(program)<=24 and all(a.get('slot') in ('left','right') and a.get('operation') in
        ('exchange','inspect','place','move','apply','activate','deactivate','withdraw') and type(a.get('duration')) in (int,float)
        and np.isfinite(a['duration']) and a['duration']>0 for a in program) and sum(a['duration'] for a in program)<=10.+1e-10,'unsupported suffix program')
    result = engine.clone()
    for action in program:
        receipt = result.step(copy.deepcopy(action))
        require(receipt['status'] != 'rejected', 'exported native action refused')
    return result


def cost(engine, baseline):
    cells = set(engine.state['removed'])-set(baseline.state['removed'])
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material']=='tumor'}
    target = sum(int(engine.scene.labels.flat[c]) in targets for c in cells)*engine.scene.voxel_mm3
    return {'additionalTargetMm3':target, 'additionalNonTargetMm3':len(cells)*engine.scene.voxel_mm3-target,
        'elapsedSeconds':engine.state['elapsedSeconds']-baseline.state['elapsedSeconds']}


def policies(rows):
    def prep(row):
        public=row['preparation']['publicPlan'];return public['attempts'][public['decision']['selectedIndex']]['summary']
    access=[r for r in rows if r['comparison'] is not None and r['comparison']['publicPlan']['approaches']['routes']]
    corridor=min(access,key=lambda r:(prep(r)['additionalNonTargetMm3'],prep(r)['elapsedSeconds'],rows.index(r))) if access else None
    full=[]
    for row in rows:
        if row['comparison'] is None:continue
        public=row['comparison']['publicPlan'];choice=next(a for a in public['alternatives'] if a['id']==public['decision']['selectedId'])
        if choice['id']!='stop' and choice['feasible'] is True and choice['taskProgressMet'] is True:
            full.append((prep(row)['additionalNonTargetMm3']+choice['summary']['additionalNonTargetMm3'],
                prep(row)['elapsedSeconds']+choice['summary']['elapsedSeconds'],rows.index(row),row))
    joint=min(full,key=lambda x:x[:3])[3] if full else None
    def selected(row):return {'selectedPreparationId':row['id'] if row else None,
        'suffixId':row['comparison']['publicPlan']['decision']['selectedId'] if row else 'stop'}
    return {'corridorFirst':selected(corridor),'completePlanAware':selected(joint)}


def verify(case_path, output, *, revision=None):
    case_path, output = Path(case_path).resolve(), Path(output).resolve()
    require(case_path.is_relative_to(ROOT/'testcases') and output.is_relative_to(ROOT/'testcases'), 'keep audit inputs and outputs in testcases')
    require(not output.exists(), 'preserve earlier evidence; output directory must be fresh')
    original_hash=file_hash(case_path);case, artifacts, _=load_workspace(case_path)
    record=artifacts['surgical_toolbox'];history=record.get('preparationComparisons',[])
    require(type(history) is list and 1<=len(history)<=4, 'bounded saved preparation study required')
    require(revision is None or type(revision) is int and 1<=revision<=len(history), 'choose a saved revision')
    require(type(record.get('preparationApplications',[])) is list and len(record.get('preparationApplications',[]))<=12,'bounded applications required')
    for app in record.get('preparationApplications',[]):
        seal(app,'applicationSeal')
        require(set(app)=={'revision','reportSeal','candidateId','phase','branchId','preparationBranchId','snapshotHash','preparedSnapshotHash','receipt','applicationSeal'}
            and type(app['revision']) is int and 1<=app['revision']<=len(history),'application schema/revision differs')
    for start in record['branches']:
        branch=start;seen=set()
        while branch is not None:
            require(branch in record['branches'] and branch not in seen,'unknown or cyclic branch ancestry');seen.add(branch)
            parent=record.get('parents',{}).get(branch)
            if parent is None:break
            require(type(parent) is dict and parent.get('branchId') in record['branches'] and type(parent.get('position')) is int
                and 0<=parent['position']<=len(record['branches'][parent['branchId']]['actions']),'branch parent differs')
            branch=parent['branchId']
    scene=load_environment(case_path);output.mkdir(parents=True);results=[];owned=set()
    refs=[r for r in case.source_refs if r.source_id=='source-manifest'];require(len(refs)==1,'one source-manifest reference required')
    uri=urlparse(refs[0].uri);manifest=Path(unquote(uri.path)).resolve()
    require(uri.scheme=='file' and uri.netloc in ('','localhost') and not uri.query and not uri.fragment
        and manifest.name=='patient-anatomy.json' and file_hash(manifest)==refs[0].sha256==scene.source['adapter']['sourceManifestSha256'], 'native source URI/checksum differs')
    for number,item in enumerate(history,1):
        if revision is not None and number!=revision:continue
        seal(item,'workspaceSeal');report=item['report'];seal(report,'reportSeal')
        require(set(item)=={'schema','revision','caseHash','sourceBranchId','sourceSnapshotHash','configuredBranchId','report','workspaceSeal'},'workspace schema differs')
        require(item['schema']=='resectionlab.source-preparation-workspace.v1' and type(item['revision']) is int
            and item['revision']==number and item['caseHash']==case.semantic_hash, 'workspace source/revision differs')
        require(report['schema']=='resectionlab.source-preparation-comparison.v1', 'report schema differs')
        snap=record['branches'][item['sourceBranchId']];equal(semantic_digest(snap),item['sourceSnapshotHash'],'workspace source hash differs')
        equal(item['sourceSnapshotHash'],report['sourceSnapshotHash'],'report source hash differs')
        root=SurgicalEngine.restore(scene,snap,use_cache=False)
        require(root.scene.identity==report['sourceSceneId'] and root.state_id==report['startStateId']
            and type(report['prefixActions']) is int and report['prefixActions']==len(root.actions),'source checkpoint differs')
        targets={r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor'}
        target_removed=sum(int(scene.labels.flat[c]) in targets for c in root.state['removed'])*scene.voxel_mm3
        equal(report['baselineBurden'],{'targetRemovedMm3':target_removed,'nonTargetRemovedMm3':len(root.state['removed'])*scene.voxel_mm3-target_removed,
            'elapsedSeconds':root.state['elapsedSeconds']},'source historical burden differs')
        equal(verify_source(scene,native_collection=manifest.parent),report['sourceVerification'],'source verification differs')
        configured=root.clone()
        if root.state.get('toolMechanicsRevision')!=2 or root.state.get('physicalClockRevision')!=1:
            require(configured.step({'slot':'right','operation':'configure','mechanicsRevision':2,'clockRevision':1,'duration':.1})['status']=='accepted','literal setup refused')
        equal(configured.snapshot(),report['configuredSnapshot'],'configured snapshot differs')
        equal(record['branches'][item['configuredBranchId']],configured.snapshot(),'configured branch differs')
        require(item['sourceBranchId']!=item['configuredBranchId'],'study branch ownership repeats')
        controls=report['controls'];rows=report['rows'];widths=controls['widthsMm']
        require({'brain','target'}<=set(controls['requiredEvidence']),'brain and target evidence must remain required')
        require(type(rows) is list and type(widths) is list and 1<=len(rows)==len(widths)<=3 and len(set(widths))==len(widths), 'row/width inventory differs')
        missing=sorted(set(controls['requiredEvidence'])-set(source_evidence(scene)['availableEvidence']));audits=[];native_rows={}
        for index,(row,width) in enumerate(zip(rows,widths,strict=True)):
            require(set(row)=={'id','widthMm','preparation','preparedSnapshot','preparedAdmission','comparison','reason','missingEvidence'},'row schema differs')
            require(row['id']=='width-'+str(index) and row['widthMm']==width and type(width) in (int,float) and np.isfinite(width) and 0<=width<=1,'row identity/width differs')
            equal(row['missingEvidence'],missing,'missing evidence differs')
            if row['preparation'] is None:
                require(bool(missing) and row['reason']=='REQUIRED_EVIDENCE_UNAVAILABLE' and all(row[k] is None for k in ('preparedSnapshot','preparedAdmission','comparison')), 'empty row has effects or false missing evidence')
                audits.append({'id':row['id'],'missingEvidenceVerified':True});continue
            require(not missing,'preparation executed despite missing required evidence')
            proof=row['preparation'];public=proof['publicPlan']
            require(proof['schema']=='resectionlab.source-exposure-preparation.v2','fixed source-ray proof required')
            equal(proof['sourceSnapshotHash'],semantic_digest(configured.snapshot()),'preparation baseline differs')
            equal(public['sourceRay'],controls['sourceRay'],'row source ray differs')
            equal(public['controls'],{**{k:controls[k] for k in ('maxNonTargetMm3','maxActions','maxSeconds')},
                'maxCandidates':1,'maxProposals':256,'maxPathMm':25.,'maxWallSeconds':90.,'lateralSweepMm':width},'row preparation controls differ')
            proof_path=output/('revision-'+str(number)+'-'+row['id']+'-preparation.json');proof_path.write_text(json.dumps(proof,indent=2)+'\n')
            audit=exposure.verify(case_path,proof_path,expected_snapshot=row['preparedSnapshot'] or configured.snapshot())
            audits.append({'id':row['id'],'preparation':audit})
            if row['preparedSnapshot'] is None:
                require(row['preparedAdmission'] is None and row['comparison'] is None and public['decision']['selectedIndex'] is None,'unselected preparation has effects');continue
            prepared=SurgicalEngine.restore(scene,row['preparedSnapshot'],use_cache=False);contract=row['preparedAdmission'];validate_admission(prepared,contract)
            equal(contract['task']['entry'],{'centerMm':controls['sourceRay']['surfaceMm'],'radiusMm':controls['entryRadiusMm']},'entry differs')
            equal(contract['task']['target'],controls['target'],'target differs');equal(contract['task']['requiredEvidence'],controls['requiredEvidence'],'required evidence differs')
            axis=np.asarray(controls['sourceRay']['sourceTargetMm'])-controls['sourceRay']['surfaceMm'];axis=axis/np.linalg.norm(axis)
            require(np.allclose(contract['task']['axisRasUnit'],axis,atol=1e-9,rtol=0.),'task axis differs')
            comparison=row['comparison'];seal(comparison,'reportSeal');plan=comparison['publicPlan'];seal(plan,'publicPlanSeal')
            equal(comparison['taskAdmission'],contract,'suffix admission differs')
            require(comparison['startStateId']==prepared.state_id and comparison['prefixActions']==len(prepared.actions),'suffix baseline differs')
            equal([comparison[k] for k in ('baselineSnapshotHash','baselineActionHash','baselineHistoryHash')],[semantic_digest(v) for v in (prepared.snapshot(),prepared.actions,prepared.history)],'suffix native baseline hashes differ')
            for key in ('shortSeconds','longSeconds','minimumTargetMm3'):equal(plan['constraints'][key],controls[key],'suffix control differs: '+key)
            require({'brain','target'}<=set(controls['requiredEvidence']) and comparison['schema'] in ('resectionlab.two-tool-comparison.v1','resectionlab.two-tool-comparison.v2'),'suffix schema/evidence differs')
            equal(plan['constraints'].get('encounter'),None if controls['encounter'] is None else {k:controls['encounter'][k] for k in ('marginMm','maxNearMm3')},'context controls differ')
            audits[-1]['geometry']=geometry.verify(comparison,case_path,source_snapshot=row['preparedSnapshot']);require(audits[-1]['geometry']['passed'],'independent suffix geometry failed')
            if controls['encounter'] is not None:
                equal(comparison['contextBinding']['layerId'],controls['encounter']['layerId'],'context layer differs')
                audits[-1]['context']=encounter.verify(comparison,case_path,source_snapshot=row['preparedSnapshot'])
            alternatives={}
            require(1<=len(plan['alternatives'])<=3 and len({a['id'] for a in plan['alternatives']})==len(plan['alternatives'])
                and all(a['id'] in ('short','long','stop') for a in plan['alternatives']),'suffix inventory differs')
            for alt in plan['alternatives']:
                if alt['id']=='stop':require(not alt['actionProgram'],'STOP has native effects')
                else:
                    seconds=controls['shortSeconds' if alt['id']=='short' else 'longSeconds'];equal(alt['contactSeconds'],seconds,'declared contact differs')
                    equal([a['duration'] for a in alt['actionProgram'] if a['slot']=='left' and a['operation']=='apply'],[seconds],'native contact duration differs')
                native=replay(prepared,alt['actionProgram']);actual=cost(native,prepared)
                for key,value in actual.items():equal(alt['summary'][key],value,'suffix native cost differs: '+key)
                require(alt['taskProgressMet'] is (actual['additionalTargetMm3']>=controls['minimumTargetMm3']), 'suffix task progress differs')
                require(type(alt['feasible']) is bool and (controls['encounter'] is not None or alt['feasible'] is True),'suffix feasibility differs')
                alternatives[alt['id']]=native
            feasible=[a for a in plan['alternatives'] if a['feasible'] is True and a['taskProgressMet'] is True]
            chosen=min(feasible,key=lambda a:(a['summary']['elapsedSeconds'],-a['summary']['additionalTargetMm3'],a['id']))['id'] if feasible else 'stop'
            equal(plan['decision']['selectedId'],chosen,'suffix public selection differs');native_rows[row['id']]=(prepared,alternatives)
        public=[{'id':r['id'],'preparation':r['preparation']['publicPlan'] if r['preparation'] else None,'strategy':r['comparison']['publicPlan'] if r['comparison'] else None} for r in rows]
        equal(report['publicPolicyInputHash'],semantic_digest(public),'public policy input differs');equal(report['policies'],policies(rows),'public policy choices differ')
        incomplete=[]
        for row in rows:
            if row['comparison'] is None:continue
            plan=row['comparison']['publicPlan'];reasons=[plan['decision'].get('reason'),*[r.get('reason') for r in plan['rejections']],*[r.get('reason') for r in plan['approaches']['rejections']]]
            if any(r in ('COMPUTATION_BUDGET','GEOMETRY_WALL_BUDGET') for r in reasons):incomplete.append(row['id'])
        equal(report['incompleteRows'],incomplete,'incomplete search inventory differs')
        applications=[]
        for app in record.get('preparationApplications',[]):
            if app['revision']!=number:continue
            seal(app,'applicationSeal');require(app['reportSeal']==report['reportSeal'] and app['candidateId'] in native_rows,'application identity differs')
            prepared,alternatives=native_rows[app['candidateId']];row=rows[int(app['candidateId'].split('-')[1])]
            require(app['phase'] in ('prepared','withdraw'),'application phase differs');native=prepared
            branches={app['preparationBranchId'],app['branchId']}
            require(not branches & (owned|{item['sourceBranchId'],item['configuredBranchId']}),'application branch ownership repeats');owned.update(branches)
            parent=record.get('parents',{}).get(app['preparationBranchId']);require(type(parent) is dict and parent['position']==len(root.actions),'preparation parent differs')
            equal(record['branches'][parent['branchId']]['actions'][:parent['position']],root.actions,'preparation parent native prefix differs')
            equal([record['branches'][parent['branchId']].get(k) for k in ('sceneId','protectionContract')],[snap.get(k) for k in ('sceneId','protectionContract')],'preparation parent source/protection differs')
            if app['phase']=='prepared':require(app['branchId']==app['preparationBranchId'],'prepared application branch differs')
            if app['phase']=='withdraw':
                selected=row['comparison']['publicPlan']['decision']['selectedId'];require(selected!='stop','STOP cannot install full suffix');native=alternatives[selected]
                equal(record['parents'][app['branchId']],{'branchId':app['preparationBranchId'],'position':len(prepared.actions)},'full suffix parent differs')
            equal(record['branches'][app['preparationBranchId']],prepared.snapshot(),'installed preparation differs')
            equal(record['branches'][app['branchId']],native.snapshot(),'installed native result differs')
            equal(app['snapshotHash'],semantic_digest(native.snapshot()),'application snapshot hash differs');equal(app['preparedSnapshotHash'],semantic_digest(prepared.snapshot()),'prepared hash differs')
            expected={'candidateId':row['id'],'phaseId':app['phase'],'reportSeal':report['reportSeal'],'startStateId':root.state_id,
                'preparationStateId':prepared.state_id,'resultStateId':native.state_id,'nativeMatchesPublicPrediction':True,'withdrawalComplete':True,
                'preparationActions':len(prepared.actions)-len(root.actions),'suffixActions':len(native.actions)-len(prepared.actions)}
            equal(app['receipt'],expected,'application receipt differs');applications.append(app['branchId'])
        results.append({'revision':number,'reportSeal':report['reportSeal'],'rows':audits,'applications':applications})
    require(file_hash(case_path)==original_hash,'source case bytes changed')
    result={'schema':'resectionlab.independent-preparation-comparison-audit.v1','passed':True,'source':str(case_path),
        'sourceSha256':original_hash,'sourceUnchanged':True,'revisions':results,'scope':SCOPE}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--revision',type=int,help='omit to audit all bounded saved revisions');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.case,args.output,revision=args.revision),indent=2))
