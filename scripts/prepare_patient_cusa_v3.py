#!/usr/bin/env python3
"""Inspect, then explicitly evaluate a new patient-derived CUSA-v3 scene.

Inspect never constructs/steps an engine. Evaluate executes a fresh new scene,
not an old checkpoint: five explicitly privileged archival bone actions, one
fixed contact line and three small duration/partition arms. No fitting/sweep.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import SCHEMA as ENV_SCHEMA, load_environment, standardize
from resectionlab.surgical_rl_protection import (ProtectionAnnotation, ProtectionSet,
    grid_binding, protection_contract, restore_protection_contract)
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace, save_workspace

DEFAULT_SOURCE=ROOT/'testcases/outputs/product-first-20261023/rescue-v2/rescue-strategies.ressectionlab'
DEFAULT_REFERENCE=ROOT/'testcases/outputs/full-case-rehearsal-20261020/final/vascular-canonical.actions.jsonl'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def require(value,message):
    if not value:raise ValueError(message)


def validate_outcome(action,result,*,historical_setup=None):
    """Bind privileged setup to exact evidence; keep new CUSA stops strict."""
    if historical_setup is not None:
        require(action==historical_setup['action'],'Historical setup action differs from the recorded recipe')
        require(historical_setup['status']!='rejected','Historical setup cannot authorize rejected actions')
        for key in ('status','reason','reasonCode','completedFraction','elapsedSeconds','events'):
            require(result.get(key)==historical_setup.get(key),'Historical setup receipt differs: '+key)
        require(result.get('deltas',{}).get('removedCells')==historical_setup['removedCells'],
            'Historical setup removed-cell evidence differs')
        return
    require(result['status']!='rejected','Public CUSA experiment action refused: '+str(result.get('reason')))
    require(result['status']!='partial' or result.get('reasonCode')=='TISSUE_ASPIRATION_IN_PROGRESS',
        'Public CUSA primitive stopped at a physical guard: '+str(result.get('reason')))


def numeric_hashes(case):
    values=dict(mri=array_digest(case.mri),affine=array_digest(case.affine),
        compartments={k:array_digest(v) for k,v in case.compartments.items()},
        sourceCompartments={k:array_digest(v) for k,v in case.source_compartments.items()})
    if case.brain_mask is not None:values['brainMask']=array_digest(case.brain_mask)
    return values


def derive(case,old,source,source_hash):
    """Change only explicit simulation metadata; source numeric arrays persist."""
    require(old.get('environmentSchema')==ENV_SCHEMA,'Source is not the explicit V2 environment')
    old_scene=standardize(from_case(case,old['mapping']),physics_profile=old['physicsProfile'])
    require(all(b['sceneId']==old_scene.identity for b in old['branches'].values()),'Historical scene binding differs')
    require('cusaV3' not in old_scene.properties,'Source already opts into CUSA-v3')
    metadata=thaw_json(case.metadata)
    metadata['surgical_properties']=copy.deepcopy(old_scene.properties)
    metadata['surgical_properties']['cusaV3']=default_profile()
    metadata['cusaV3ResearchDerivative']=dict(source=str(source),sourceSha256=source_hash,
        sourceCaseHash=case.semantic_hash,sourceSceneId=old_scene.identity,
        kind='explicit uncalibrated patient-derived physics experiment; not a new patient')
    derived=case.revised(metadata=metadata)
    scene=standardize(from_case(derived,old['mapping']),physics_profile=old['physicsProfile'])
    require(scene.identity!=old_scene.identity and derived.semantic_hash!=case.semantic_hash,'Derived identities must be distinct')
    require(numeric_hashes(derived)==numeric_hashes(case) and derived.source_refs==case.source_refs,'Numeric/source-reference preservation failed')
    require(np.array_equal(scene.labels,old_scene.labels) and np.array_equal(scene.affine,old_scene.affine),'Material ownership/grid changed')
    require(scene.properties=={**old_scene.properties,'cusaV3':default_profile()},'Existing physics assumptions changed')
    original_contract=old['branches']['rescue-decision']['protectionContract']
    original_protection,config=restore_protection_contract(old_scene,original_contract)
    binding=grid_binding(scene)
    annotations=[ProtectionAnnotation(a.structure_id,a.cells,binding,a.provenance,a.review_status,a.display_name)
                 for a in original_protection.annotations]
    protection=ProtectionSet(scene,annotations,clearance_mm=original_protection.clearance_mm,required=original_protection.required)
    rebound=protection_contract(protection,config)
    require(rebound['annotations']==original_contract['annotations'] and rebound['config']==original_contract['config']
        and rebound['clearanceMm']==original_contract['clearanceMm'] and rebound['required']==original_contract['required'],
        'Protection masks or guard settings changed during source rebind')
    return derived,scene,old_scene,protection,config,rebound


def recipe_and_screen(scene,old,reference,protection,check):
    """Static screen using recorded bone removals, never checkpoint admission."""
    source_actions=old['branches']['full-vascular']['actions']
    require([r['action'] for r in reference[:5]]==source_actions[:5],'Reference five-action access differs')
    require([a['operation'] for a in source_actions[:5]]==['place','apply','apply','apply','withdraw'], 'Unexpected access recipe')
    removed=set(c for row in reference[:5] for c in row['removedCells'])
    material_by_id={r['id']:r['material'] for r in scene.regions}
    require(all(material_by_id[int(scene.labels.flat[c])]=='bone' for c in removed),'Access removed non-bone material')
    outside=copy.deepcopy(source_actions[7]['path'][1]);requested=copy.deepcopy(source_actions[7]['path'][2])
    require(outside['quaternion']==requested['quaternion']==[1.,0.,0.,0.],'Expected explicit downward contact line')
    grid=GeometryScene(np.zeros(scene.shape,bool),scene.affine);labels=scene.labels.ravel()
    materials={r['id']:r['material'] for r in scene.regions}
    def contacts(a,b,radius):
        indices=capsule_voxel_indices(grid,a,b,radius)
        ids=np.ravel_multi_index(indices.T,scene.shape) if len(indices) else []
        return {int(c) for c in ids if labels[c] and int(c) not in removed}
    start=np.asarray(outside['position']);end=np.asarray(requested['position']);low=start.copy();first=None
    for fraction in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(end-start)/.25)+1)):
        check();point=start+(end-start)*fraction
        compatible={c for c in contacts(point,point,1.5) if materials[int(labels[c])] in ('soft_tissue','tumor')}
        if compatible:
            high=point.copy()
            for _ in range(32):
                mid=(low+high)/2
                if any(materials[int(labels[c])] in ('soft_tissue','tumor') for c in contacts(mid,mid,1.5)):high=mid
                else:low=mid
            first=high;break
        low=point
    require(first is not None,'Fixed source contact line reaches no compatible tissue after recorded access')
    touched=contacts(first,first,1.5);unsupported=contacts(start,first,1.5)-{c for c in contacts(start,first,1.5) if materials[int(labels[c])] in ('soft_tissue','tumor')}
    # Axis is -Z; a fixed40mm shaft lies above the actual contact tip.
    axis=np.array([0.,0.,-1.]);shaft=contacts(start-axis*40,first-axis*2,.7)
    excluded=protection.capsule_hits(start,first,1.7).tolist()
    excluded+=protection.capsule_hits(start-axis*40,first-axis*2,.7).tolist()
    result=dict(outside=outside,requestedEndpoint=requested,staticFirstContactMm=first.tolist(),
        compatibleContactSourceCells=sorted(c for c in touched if materials[int(labels[c])] in ('soft_tissue','tumor')),
        unsupportedSourceCells=sorted(unsupported),conservativeShaftSourceCells=sorted(shaft),
        excludedSourceCells=sorted(set(excluded)),historicalBoneRemovalCells=len(removed),
        interpretation='Read-only source/material screen using documented first-five bone removals; actual new-physics execution remains required.')
    result['screenPassed']=not unsupported and not shaft and not excluded
    return result


def ledger(engine):
    state=engine.state['cusaV3'];m=engine.metrics()
    capacity=state['inputCapacityMm3'];accounted=math.fsum([state['aspiratedSolidMm3'],sum(state['progressMm3'].values()),
        state['unusedCapacityMm3'],state['discardedCapacityMm3']])
    require(abs(capacity-accounted)<1e-9,'CUSA capacity conservation differs')
    return dict(stateId=engine.state_id,simulatedSeconds=engine.state['elapsedSeconds'],removedCells=len(engine.state['removed']),
        **{k:m[k] for k in ('cusaInputCapacityMm3','cusaAspiratedSolidMm3','cusaPendingCapacityMm3','cusaUnusedCapacityMm3',
            'cusaDiscardedCapacityMm3','cusaPendingCells','targetRemovedMm3','nonTargetRemovedMm3')},
        actualTip=copy.deepcopy(engine.state['slots']['left']['position']),pendingExact=copy.deepcopy(state['progressExact']),
        inputEqualsAspiratedPendingUnusedDiscarded=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['inspect','evaluate']);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();source=args.source.resolve()
    require(out.is_relative_to(ROOT/'testcases') and not out.exists(),'Use a new output directory under repository testcases/')
    out.mkdir(parents=True);started=time.perf_counter()
    report=dict(schema='resectionlab.patient-cusa-v3-experiment.v1',passed=False,mode=args.mode,source=str(source),sourceSha256=sha(source),
        scriptSha256=sha(__file__),maximumWallSeconds=240,maximumRssBytes=3*1024**3,trainingStarted=False,optimizerUpdates=0,
        physicalSteps=0,limitations=['Explicit uncalibrated CUSA rate2mm3/s, catalog tip1.5mm; no acoustic/device/patient calibration.',
            'Historical five-action bone access uses oversized archival burr8mm and30s actions; privileged native setup, not raw-RL admission.',
            'New scene starts intact; no old checkpoint or CUSA trajectory supplies physical authority.',
            'Fixed source-author line may remove non-target brain tissue; this is a contact-law experiment, not a tumor-access plan.',
            'Existing registration/protection/functional-anatomy unknowns remain; no clinical or policy-generalization claim.'])
    def check():
        if time.perf_counter()-started>240:raise TimeoutError('CUSA patient experiment wall cap exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)>3*1024**3:raise MemoryError('CUSA patient experiment RSS cap exceeded')
    def persist():
        # Keep completed evidence even if a later cold replay is interrupted.
        report['wallSeconds']=time.perf_counter()-started
        pending=out/'report.pending.json';write(pending,report);pending.replace(out/'report.json')
    try:
        case,artifacts,workspace=load_workspace(source);old=artifacts['surgical_toolbox'];original_arrays=numeric_hashes(case)
        derived,scene,old_scene,protection,config,contract=derive(case,old,source,report['sourceSha256']);check()
        reference=[json.loads(line) for line in args.reference.read_text().splitlines()]
        screen=recipe_and_screen(scene,old,reference,protection,check)
        report.update(sourceCaseHash=case.semantic_hash,derivedCaseHash=derived.semantic_hash,sourceSceneId=old_scene.identity,
            derivedSceneId=scene.identity,arrayHashes=original_arrays,sourceRefsPreserved=True,
            originalV2Profile=copy.deepcopy(old['physicsProfile']),cusaV3=default_profile(),reboundProtectionContract=contract,
            originalProtectionContractHash=old['branches']['rescue-decision']['protectionContract']['contentHash'],
            historicalBranchesHash=semantic_digest(old['branches']),historicalBranchIds=sorted(old['branches']),recipeScreen=screen,
            referenceSha256=sha(args.reference),derivationVerified=True)
        write(out/'recipe-screen.json',screen)
        if args.mode=='inspect':
            report.update(passed=True,readyForBoundedEvaluation=screen['screenPassed']);return
        require(screen['screenPassed'],'Fixed contact recipe blocked by retained material/shaft/protection; no physics executed')
        initial=SurgicalEngine(scene,protection=protection,guard_config=config)
        record=dict(environmentSchema=ENV_SCHEMA,physicsProfile=copy.deepcopy(old['physicsProfile']),mapping=copy.deepcopy(scene.regions),
            branches={},parents={},sourceHistoryReference=dict(path=str(source),sha256=report['sourceSha256'],sceneId=old_scene.identity,
                branchesHash=report['historicalBranchesHash'],meaning='Historical branches remain unchanged in their original source case; none migrated into this scene.'))
        def snapshot(engine):
            row=engine.snapshot();row.pop('checkpoint',None);return row
        def execute(engine,action,rows,*,historical_setup=None):
            report['pendingAction']=copy.deepcopy(action);persist()
            check();step_started=time.perf_counter();result=engine.step(action,check=check);report['physicalSteps']+=1
            rows.append(dict(action=copy.deepcopy(action),outcome=result,stateId=engine.state_id,ledger=ledger(engine),
                wallSeconds=time.perf_counter()-step_started))
            report.pop('pendingAction',None);persist()
            validate_outcome(action,result,historical_setup=historical_setup)
            if historical_setup is not None:
                rows[-1]['historicalSetupReceiptExact']=True;persist()
            return result
        common=[];report['nativeSetupAndContact']=common;report['stage']='fresh bone access'
        for action,expected in zip(old['branches']['full-vascular']['actions'][:5],reference[:5]):
            execute(initial,action,common,historical_setup=expected)
        require(not initial.state['slots']['left']['active'] and not initial.state['slots']['left']['inserted'],
            'Historical setup did not end at its original inactive withdrawal')
        report['historicalSetupReceiptsExact']=True
        record['branches']['v3-bone-access']=snapshot(initial)
        report['boneAccessWallSeconds']=sum(row['wallSeconds'] for row in common)
        report['stage']='fresh common contact'
        outside=screen['outside'];target=screen['requestedEndpoint']
        for action in [dict(slot='left',operation='exchange',instrument='cusa',duration=.1),
            dict(slot='left',operation='place',pose=outside,parameters=dict(radiusMm=1.5,power=1.,duty=1.,forceN=0.),duration=.1),
            dict(slot='left',operation='apply',pose=target,duration=.05)]:execute(initial,action,common)
        require(initial.state['cusaV3']['progressExact'],'First actual contact has no pending exact work; fixed recipe needs review')
        require(initial.state['cusaV3']['aspiratedSolidMm3']==0,'Short first contact unexpectedly clears a whole voxel')
        report['commonContactLedger']=ledger(initial)
        record['branches']['v3-common-contact']=snapshot(initial)
        record['parents']['v3-common-contact']=dict(branchId='v3-bone-access',position=5)
        arms=[];report['arms']=arms
        for name,durations in [('short',[.125]),('long',[1.125]),('partitioned',[.375,.75])]:
            candidate=initial.clone();rows=[]
            branch='cusa-v3-'+name
            arm=dict(name=name,branchId=branch,dwellDurations=durations,transitions=rows,completed=False)
            arms.append(arm);report['stage']='duration arm '+name;persist()
            for duration in durations:execute(candidate,dict(slot='left',operation='apply',duration=duration),rows)
            arm['afterDwell']=ledger(candidate);persist()
            if name!='short':execute(candidate,dict(slot='left',operation='apply',pose=target,duration=.5),rows)
            execute(candidate,dict(slot='left',operation='withdraw',pose=outside,duration=.2),rows)
            require(not candidate.state['slots']['left']['active'] and not candidate.state['slots']['left']['inserted'],'CUSA controls not clear after guarded withdrawal')
            record['branches'][branch]=snapshot(candidate);record['parents'][branch]=dict(branchId='v3-common-contact',position=len(initial.actions))
            arm.update(endpoint=ledger(candidate),expectedRemovedCells=copy.deepcopy(candidate.state['removed']),completed=True)
            persist()
        require(arms[0]['afterDwell']['cusaAspiratedSolidMm3']==0 and arms[0]['afterDwell']['cusaPendingCapacityMm3']>0,'Short dwell did not preserve pending partial capacity')
        require(arms[1]['afterDwell']['cusaAspiratedSolidMm3']>arms[0]['afterDwell']['cusaAspiratedSolidMm3'],'Long dwell did not cause progressive whole-cell removal')
        require(arms[1]['expectedRemovedCells']==arms[2]['expectedRemovedCells'],'Matched duration partitions cleared different source cells')
        for key in ('cusaInputCapacityMm3','cusaAspiratedSolidMm3','cusaPendingCapacityMm3','cusaUnusedCapacityMm3'):
            require(abs(arms[1]['endpoint'][key]-arms[2]['endpoint'][key])<1e-9,'Matched dwell partition capacity differs: '+key)
        record.update(branchId='cusa-v3-long',position=len(record['branches']['cusa-v3-long']['actions']))
        preserved_artifacts={k:v for k,v in artifacts.items() if k not in ('surgical_toolbox','full_case_branch_provenance','workspace')}
        preserved_artifacts['surgical_toolbox']=record
        preserved_artifacts['cusa_v3_derivation']=dict(sourceCase=str(source),sourceSha256=report['sourceSha256'],
            historicalBranchesHash=report['historicalBranchesHash'],scope='Distinct case metadata/scene; unchanged numeric anatomy and no migrated historical checkpoints.')
        destination=out/'patient-cusa-v3-research.ressectionlab'
        report['stage']='save and reopen';persist()
        save_workspace(derived,destination,sources=workspace['sources'],imaging=workspace['imagingState'],episode=None,selection=None,
            originals=workspace['originals'],artifacts=preserved_artifacts);check()
        reopened,saved,_=load_workspace(destination);new_scene=load_environment(destination)
        require(numeric_hashes(reopened)==original_arrays and reopened.source_refs==case.source_refs,'Reopened numeric data/source refs changed')
        require(reopened.semantic_hash==derived.semantic_hash and new_scene.identity==scene.identity,'New scene/case save binding differs')
        require(saved['surgical_toolbox']==record,'New native action/source record changed during serialization')
        report.update(savedCase=str(destination),savedSha256=sha(destination),numericArraysAndSourceRefsPreserved=True,
            saveReopenVerified=True);persist()
        for arm in arms:
            # Whole native branch starts fresh under the new physics. Imported
            # old checkpoints never enter this replay; saved records omit them.
            report['stage']='cold full replay '+arm['branchId'];arm['coldFullNativeReplayStatus']='running';persist()
            replay_started=time.perf_counter()
            checked=SurgicalEngine.restore(new_scene,saved['surgical_toolbox']['branches'][arm['branchId']],use_cache=False,check=check)
            report['physicalSteps']+=len(checked.actions)
            require(ledger(checked)==arm['endpoint'],'Cold full native replay changed the exact CUSA endpoint ledger')
            expected=common+arm['transitions']
            require(len(checked.history)==len(expected),'Reopened complete action inventory differs')
            # History owns complete public receipts including original before/
            # after identities. Compare its projection to recorded full results.
            for history,item in zip(checked.history,expected):
                require(history['action']==item['action'] and history['stateAfter']==item['stateId'],'Reopened action/state prefix differs')
                for key in ('status','reason','reasonCode','completedFraction','elapsedSeconds','events','deltas'):
                    require(semantic_digest(history.get(key))==semantic_digest(item['outcome'].get(key)),'Reopened complete receipt differs: '+key)
            arm.update(coldFullNativeReplayExact=True,coldFullNativeReplayStatus='passed',
                coldFullNativeReplayWallSeconds=time.perf_counter()-replay_started);persist()
        _,original_after,_=load_workspace(source)
        require(semantic_digest(original_after['surgical_toolbox']['branches'])==report['historicalBranchesHash'],'Historical source branches changed')
        report.update(passed=True,stage='complete',
            historicalBranchesUnchanged=True,newActiveBranchesOnly=True,coldFullNativeReplaysExact=True)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        report.update(sourcePreserved=sha(source)==report['sourceSha256'],wallSeconds=time.perf_counter()-started,
            peakResidentBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        persist()
        print(json.dumps({k:report[k] for k in ('mode','passed','physicalSteps','sourcePreserved','readyForBoundedEvaluation','wallSeconds','error') if k in report}),flush=True)

if __name__=='__main__':main()
