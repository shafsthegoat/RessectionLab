#!/usr/bin/env python3
"""Extend the existing exposed UCSF episode with checked, time-limited work.

Keeps the source and recorded exposure. No new anatomy, rate change or policy
inference. The fixed overlapping shallow raster excavates a visible cup.
"""
from pathlib import Path
import argparse, copy, hashlib, json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import load_workspace, save_workspace
from resectionlab.imaging import save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize,SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.core import array_digest


def prepare(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'testcases') or output.exists():raise ValueError('Choose a NEW output under testcases/')
    output.mkdir(parents=True);started=time.perf_counter();case,artifacts,workspace=load_workspace(source)
    previous=artifacts['surgical_toolbox'];scene=standardize(from_case(case,previous['mapping']),physics_profile=previous['physicsProfile'])
    if scene.identity!='sha256:b7c1e36161051be29b5e694aab35c9b11828f63b0c3444016824f6d506c0db12':
        raise ValueError('This fixed authoring path is bound to the existing exposed UCSF-PDGM-0007 source scene')
    source_hash=array_digest(scene.labels);mri_hash=array_digest(case.mri)
    # Exact original four-action exposure prefix, never newly carved anatomy.
    e=SurgicalEngine.restore(scene,previous['branches']['surgeon-exposure']).branch(4)
    inherited=e.snapshot();rows=[];q=[0.,0.,0.,1.];outside=np.array([-84.,118.,26.])
    def act(op,position=None,slot='left',duration=.2,**kw):
        if op=='apply' and slot=='left' and position is not None:
            # Choose a bounded authoring duration from retained source contact
            # work at the unchanged2mm3/s rate; avoid long free-air pedal dwell.
            contacts=e.touched(e.state['slots'][slot]['position'],position,1.5)
            removed=set(e.state['removed']);pending=e.state.get('cusaV3',{}).get('progressMm3',{})
            required=sum(max(0.,scene.voxel_mm3-pending.get(str(int(c)),0.)) for c in contacts if int(c) not in removed)
            duration=min(30.,max(.4,required/2.+.1))
        a={'slot':slot,'operation':op,'duration':duration,**kw}
        if position is not None:a['pose']={'position':list(map(float,position)),'quaternion':q}
        before=e.metrics();sid=e.state_id;t=time.perf_counter();r=e.step(a);after=e.metrics()
        rows.append({'index':len(e.actions),'action':a,'status':r['status'],'reason':r.get('reason'),'elapsedSeconds':e.state['elapsedSeconds'],
                     'beforeStateId':sid,'stateId':e.state_id,'targetDeltaMm3':after['targetRemovedMm3']-before['targetRemovedMm3'],
                     'nonTargetDeltaMm3':after['nonTargetRemovedMm3']-before['nonTargetRemovedMm3'],
                     'fluidRemovedDeltaMm3':after.get('fluidRemovedMm3',0)-before.get('fluidRemovedMm3',0),'wallSeconds':time.perf_counter()-t})
        (output/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n')
        (output/'actions.json').write_text(json.dumps(rows,indent=2)+'\n')
        if r['status']=='rejected':
            (output/'failure-snapshot.json').write_text(json.dumps(e.snapshot())+'\n')
            raise ValueError(str(rows[-1]))
        return r
    act('configure',mechanicsRevision=2,duration=.1)
    # Reuse the original field and tip axis. The right hand is introduced by a
    # checked path; it is not teleported into tissue.
    act('place',outside,parameters={'radiusMm':1.5,'power':1.,'duty':1.})
    act('apply',[-84,118,62],duration=2.)
    # Declared irrigation, not manufactured bleeding. Its outlet and the
    # Frazier pickup share the same cleared physical cell neighbourhood.
    act('exchange',slot='right',instrument='irrigation')
    act('place',outside+[-5,0,0],slot='right')
    act('move',[-89,118,59],slot='right',duration=1.)
    act('apply',slot='right',duration=.6,parameters={'flowMm3PerSecond':20.})
    act('withdraw',outside+[-5,0,0],slot='right',duration=1.)
    act('exchange',slot='right',instrument='frazier_suction')
    act('place',outside+[-5,0,0],slot='right')
    act('move',[-90,118,61],slot='right',duration=1.)
    act('activate',slot='right',duration=.2,parameters={'flowMm3PerSecond':.4})
    start=len(e.actions);baseline=e.snapshot();before=e.metrics()
    # Fixed 5x5-mm overlapping raster, shallow first. Each stage has a small
    # requested advance; work accumulates at native2mm3/s until clearance.
    for y in (116.,118.,120.):
        for x in ((-82.,-84.,-86.) if y!=118. else (-86.,-84.,-82.)):
            act('withdraw',[*e.state['slots']['left']['position'][:2],58.],duration=.5)
            act('move',[x,y,58.],duration=.5)
            for z in (62.,64.,66.,68.):
                for attempt in range(3):
                    r=act('apply',[x,y,z],duration=8.)
                    if np.linalg.norm(np.array(e.state['slots']['left']['position'])-[x,y,z])<1e-6:break
                    if r.get('reason') not in ('TISSUE_ASPIRATION_IN_PROGRESS',):break
            if y==116. and x==-82.:
                act('activate',duration=.1)
                act('move',[-89.5,118,61],slot='right',duration=2.)
                act('move',[-90,118,61],slot='right',duration=2.)
                act('deactivate',duration=.1)
            if e.metrics()['targetRemovedMm3']-before['targetRemovedMm3']>=180:break
        if e.metrics()['targetRemovedMm3']-before['targetRemovedMm3']>=180:break
    # Deepen the existing cup through small checked tip-led requests. This is
    # real owned-cell removal, rather than a darkened fake crater.
    act('withdraw',[*e.state['slots']['left']['position'][:2],58.],duration=.5)
    act('move',[-84,118,58],duration=.5)
    for z in (64.,66.,68.,70.,72.,74.,76.):
        for attempt in range(3):
            r=act('apply',[-84,118,z],duration=6.)
            if np.linalg.norm(np.array(e.state['slots']['left']['position'])-[-84,118,z])<1e-6:break
            if r.get('reason')!='TISSUE_ASPIRATION_IN_PROGRESS':break
    end=len(e.actions);after=e.metrics();print('work',start,end,'delta',after['targetRemovedMm3']-before['targetRemovedMm3'],after['nonTargetRemovedMm3']-before['nonTargetRemovedMm3'],flush=True)
    act('deactivate');act('withdraw',outside,duration=1.);act('withdraw',outside+[-5,0,0],slot='right',duration=1.)
    act('exchange',slot='right',instrument='irrigation')
    act('place',outside+[-5,0,0],slot='right')
    act('move',[-89,118,59],slot='right',duration=1.)
    act('apply',slot='right',duration=.3,parameters={'flowMm3PerSecond':20.})
    act('withdraw',outside+[-5,0,0],slot='right',duration=1.)
    final=e.snapshot();restored=SurgicalEngine.restore(scene,json.loads(json.dumps(final)));assert restored.state_id==e.state_id
    setup={**previous['researchSetup'],'kind':'extended-source-bound-operative-demo','description':previous['researchSetup']['description']+' Illustrative operative surface colours are display only.',
           'finiteCusaStart':start,'finiteCusaEnd':end,'mechanicsRevision':2,'originalSceneId':scene.identity,'extendedFromSha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    record={'environmentSchema':SCHEMA,'physicsProfile':copy.deepcopy(scene.properties['tissuePhysicsV2']),'mapping':scene.regions,
            'branches':{'main':SurgicalEngine(scene).snapshot(),'original-exposure':inherited,'operative-start':baseline,'operative-cusa':final},
            'branchId':'operative-cusa','position':start,'parents':{'main':None,'original-exposure':{'branchId':'main','position':0},'operative-start':{'branchId':'original-exposure','position':4},'operative-cusa':{'branchId':'operative-start','position':start}},'researchSetup':setup}
    posture_rows=[]
    for preset in ('supine','semisitting','left_lateral'):
        branch=e.clone();r=branch.step({'slot':'left','operation':'position','position':{'preset':preset},'duration':1.})
        if r['status']!='accepted':raise ValueError('Posture action blocked: '+str(r))
        branch_id='position-'+preset;record['branches'][branch_id]=branch.snapshot();record['parents'][branch_id]={'branchId':'operative-cusa','position':len(e.actions)}
        posture_rows.append({'preset':preset,'beforeStateId':e.state_id,'stateId':branch.state_id,'effects':branch.observe()['positionEffects'],'metrics':branch.metrics(),'fluid':copy.deepcopy(branch.state['fluid'])})
    path=output/'operative-rehearsal.ressectionlab';out_artifacts={k:v for k,v in artifacts.items() if k not in ('surgical_toolbox','workspace','surgical_prior_scenes')};out_artifacts['surgical_toolbox']=record
    if workspace:save_workspace(case,path,sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],selection=workspace['selection'],artifacts=out_artifacts,originals=workspace['originals'])
    else:save_case(case,path,artifacts=out_artifacts)
    assert source_hash==array_digest(scene.labels) and mri_hash==array_digest(case.mri)
    report={'input':str(source),'case':str(path),'sceneId':scene.identity,'sourceLabelsHash':source_hash,'mriHash':mri_hash,'sourceUnchanged':True,
            'postures':posture_rows,'exactReplay':True,'workStart':start,'workEnd':end,'before':before,'after':after,'rows':rows,'wallSeconds':time.perf_counter()-started,
            'deltaTargetMm3':after['targetRemovedMm3']-before['targetRemovedMm3'],'deltaNonTargetMm3':after['nonTargetRemovedMm3']-before['nonTargetRemovedMm3']}
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('case','exactReplay','workStart','workEnd','deltaTargetMm3','deltaNonTargetMm3','wallSeconds')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');a=p.parse_args();prepare(a.source,a.output)
