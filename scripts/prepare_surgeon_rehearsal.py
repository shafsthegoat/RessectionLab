#!/usr/bin/env python3
"""Prepare a source-bound, explicitly exposed microscope demonstration.

No source masks/images are edited. A recorded historical CUSA exposure prefix
creates the opening; revised finite-capacity CUSA performs the demonstrated
resection. Exposure is an authored research setup, never validated access.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import load_workspace,save_workspace
from resectionlab.imaging import save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize,SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.core import array_digest


def prepare(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if not destination.is_relative_to(ROOT/'testcases'):raise ValueError('Store testing data under testcases/')
    if destination.exists():raise ValueError('Preserve previous example; select a new destination')
    destination.mkdir(parents=True)
    case,artifacts,workspace=load_workspace(source)
    old=artifacts['surgical_toolbox'];scene=standardize(from_case(case,old['mapping']),physics_profile=old['physicsProfile'])
    source_labels=array_digest(scene.labels);mri=array_digest(case.mri)
    target_ids=[r['id'] for r in scene.regions if r.get('target')]
    target=np.argwhere(np.isin(scene.labels,target_ids));center=np.median(target,axis=0)
    candidates=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        columns=np.unique(target[:,other],axis=0)
        columns=columns[np.argsort(np.linalg.norm((columns-center[other])*scene.spacing[other],axis=1))][:20]
        for col in columns:
            index=[slice(None)]*3
            for i,n in zip(other,col):index[i]=int(n)
            ray=scene.labels[tuple(index)];tumor=np.flatnonzero(np.isin(ray,target_ids));solid=np.flatnonzero(ray)
            for sign in (-1,1):
                t=int(tumor[0] if sign==1 else tumor[-1]);surface=int(solid[0] if sign==1 else solid[-1])
                distance=(t-surface)*sign*scene.spacing[axis]
                if distance<0 or distance>50:continue
                candidates.append((distance,axis,sign,col,t,surface))
    for _,axis,sign,col,t,surface in sorted(candidates,key=lambda x:x[0]):
        e=SurgicalEngine(scene);actions=[]
        def act(op,p=None,q=None,duration=.2,**kwargs):
            a={'slot':kwargs.pop('slot','left'),'operation':op,'duration':duration,**kwargs}
            if p is not None:a['pose']={'position':np.asarray(p).tolist(),'quaternion':q}
            result=e.step(a)
            if result['status']=='rejected':raise ValueError(result['reason'])
            actions.append({'action':a,'status':result['status'],'reason':result.get('reason'),'removed':len(e.state['removed'])})
            return result
        inward=scene.affine[:3,axis]*sign/scene.spacing[axis]
        q=Rotation.align_vectors([inward],[[0,0,1]])[0].as_quat().tolist()
        other=[i for i in range(3) if i!=axis]
        def point(coord):
            ijk=np.zeros(3);ijk[other]=col;ijk[axis]=coord
            return scene.affine[:3,:3]@ijk+scene.affine[:3,3]
        outside=point(surface-sign*18/scene.spacing[axis]);end=point(t-sign*3/scene.spacing[axis])
        try:
            act('exchange',instrument='cusa')
            act('place',outside,q,parameters={'radiusMm':9})
            act('apply',end,q,duration=1)
            act('withdraw',outside,q)
            act('configure',duration=.1)
            act('place',outside,q,parameters={'radiusMm':1.5,'power':1.,'duty':1.})
            # This enters the already checked/removed opening. Contact at its
            # floor is rate-limited by V3; the source target stays authoritative.
            contact=end+inward*8
            act('apply',contact,q,duration=1)
            lateral=scene.affine[:3,other[0]]/scene.spacing[other[0]]*5
            act('exchange',slot='right',instrument='frazier_suction')
            act('place',outside+lateral,q,slot='right')
            act('move',contact-inward*4+lateral,q,slot='right',duration=1.)
            start=len(e.actions);prefix=e.snapshot();before=e.metrics()['targetRemovedMm3']
            for _ in range(6):act('apply',contact+inward*1.5,q,duration=2)
            if e.metrics()['targetRemovedMm3']<=before:raise ValueError('No gradual target progress')
            work_end=len(e.actions)
            act('deactivate');act('withdraw',outside,q)
            # Small declared irrigation deposit supplies a real field ledger
            # for the same-prefix posture comparison; no invented bleeding.
            act('withdraw',outside+lateral,q,slot='right')
            act('exchange',slot='right',instrument='irrigation')
            act('place',contact-inward*4+lateral,q,slot='right')
            act('apply',slot='right',duration=.1,parameters={'flowMm3PerSecond':10.})
            act('withdraw',outside+lateral,q,slot='right')
            final=e.snapshot();restored=SurgicalEngine.restore(scene,json.loads(json.dumps(final)))
            assert restored.state_id==e.state_id
            setup={'kind':'authored-research-exposure','sourceArchive':str(source),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                   'description':'An authored 18 mm opening was created by the recorded historical CUSA exposure prefix. This is not a clinically planned or validated access route. Source images and labels are unchanged. Skull, complete vessels and functional anatomy are unavailable in this case.',
                   'exposureActions':4,'finiteCusaStart':start,'finiteCusaEnd':work_end,'exposureRemovedCells':len(prefix['checkpoint']['state']['removed'])}
            record={'environmentSchema':SCHEMA,'physicsProfile':copy.deepcopy(scene.properties['tissuePhysicsV2']),'mapping':scene.regions,
                    'branches':{'main':SurgicalEngine(scene).snapshot(),'surgeon-exposure':prefix,'surgeon-cusa':final},
                    'branchId':'surgeon-cusa','position':start,'parents':{'main':None,'surgeon-exposure':{'branchId':'main','position':0},'surgeon-cusa':{'branchId':'surgeon-exposure','position':start}},'researchSetup':setup}
            for preset in ('supine','semisitting'):
                posture=restored.clone();result=posture.step({'slot':'left','operation':'position','duration':1,'position':{'preset':preset}})
                if result['status']!='accepted':raise ValueError('Posture failed: '+str(result))
                record['branches']['position-'+preset]=posture.snapshot();record['parents']['position-'+preset]={'branchId':'surgeon-cusa','position':len(e.actions)}
            out=destination/'surgeon-rehearsal.ressectionlab';new_artifacts={k:v for k,v in artifacts.items() if k not in ('surgical_toolbox','surgical_prior_scenes','workspace')};new_artifacts['surgical_toolbox']=record
            if workspace:save_workspace(case,out,sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],selection=workspace['selection'],artifacts=new_artifacts,originals=workspace['originals'])
            else:save_case(case,out,artifacts=new_artifacts)
            assert array_digest(scene.labels)==source_labels and array_digest(case.mri)==mri
            report={'source':str(source),'case':str(out),'sceneId':scene.identity,'sourceLabelsUnchanged':True,'mriUnchanged':True,'canonicalReplay':True,'setup':setup,'actions':actions,'targetBefore':before,'targetAfter':e.metrics()['targetRemovedMm3'],'fixedViewTargetRasMm':contact.tolist(),'outwardRasUnit':(-inward).tolist(),'metrics':e.metrics()}
            (destination/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('case','targetBefore','targetAfter','setup')},indent=2));return
        except ValueError as error:
            print('Rejected source exposure candidate:',axis,sign,col,str(error),file=sys.stderr,flush=True)
    raise RuntimeError('No source-bound exposure candidate admitted; no anatomy was edited')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination');a=p.parse_args();prepare(a.source,a.destination)
