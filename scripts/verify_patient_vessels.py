#!/usr/bin/env python3
"""Public-action smoke test of experimental vessel estimates, not anatomy QC.

No cells are pre-removed and no instrument or physics state is injected. A
bounded six-axis search chooses an already exposed estimated vessel surface.
"""
from pathlib import Path
import argparse
import json
import time
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.core import array_digest


def candidates(scene, vessel):
    labels=scene.labels; coords=np.argwhere(labels==vessel); found=[]
    for axis in range(3):
        others=[k for k in range(3) if k!=axis]
        columns=np.unique(coords[:,others],axis=0)
        for column in columns[::max(1,len(columns)//250)]:
            index=[slice(None)]*3
            for k,value in zip(others,column):index[k]=int(value)
            ray=labels[tuple(index)]; occupied=np.flatnonzero(ray)
            if not len(occupied):continue
            for sign,tip_index in ((1,int(occupied[0])),(-1,int(occupied[-1]))):
                if ray[tip_index]!=vessel:continue
                tip=np.zeros(3,int);tip[others]=column;tip[axis]=tip_index
                lo=np.maximum(tip-2,0);hi=np.minimum(tip+3,labels.shape)
                local=labels[tuple(slice(a,b) for a,b in zip(lo,hi))]
                if np.any((local!=0)&(local!=vessel)):continue
                if np.count_nonzero(local==vessel)<2:continue
                inward=scene.affine[:3,axis]*sign/scene.spacing[axis]
                x=scene.affine[:3,others[0]]/scene.spacing[others[0]]
                y=np.cross(inward,x);y/=np.linalg.norm(y);x=np.cross(y,inward)
                q=Rotation.from_matrix(np.column_stack((x,y,inward))).as_quat().tolist()
                start=tip.astype(float);start[axis]=-5 if sign==1 else labels.shape[axis]+4
                world=lambda i:scene.affine[:3,:3]@i+scene.affine[:3,3]
                found.append((axis,sign,tip,world(start),world(tip),inward,q))
    return found


def verify(case_path,output):
    started=time.perf_counter();case=load_case(case_path);scene=from_case(case)
    vessel=next(r for r in scene.regions if r['material']=='vessel')
    before_labels=array_digest(scene.labels);before_mri=array_digest(case.mri)
    attempts=[]
    for axis,sign,index,start,tip,inward,q in candidates(scene,vessel['id'])[:80]:
        engine=SurgicalEngine(scene);records=[]
        def perform(operation,position=None,**kwargs):
            command={'slot':'left','operation':operation,'duration':.2,**kwargs}
            if position is not None:command['pose']={'position':position.tolist(),'quaternion':q}
            result=engine.step(command)
            records.append({'action':command,'status':result['status'],'reason':result.get('reason'),
                            'stateId':result['stateId'],'events':sorted({event['kind'] for event in result['events']})})
            return result['status']=='accepted'
        print(f'Trying exposed vessel {index.tolist()} axis={axis} sign={sign}',file=sys.stderr,flush=True)
        ok=perform('exchange',instrument='micro_scissors') and perform('place',start,parameters={'radiusMm':1.5,'jawOpeningMm':0.,'depthMm':2.}) and perform('apply',tip,duration=1.)
        if not ok or not engine.state.get('vesselConnections'):
            attempts.append({'index':index.tolist(),'axis':axis,'sign':sign,'lastStatus':records[-1]['status'],'reason':records[-1]['reason'],'wounds':len(engine.state.get('vesselWounds',{}))});continue
        wounds_after_cut=len(engine.state['vesselWounds']);cut_metrics=engine.metrics()
        ok=perform('withdraw') and perform('exchange',instrument='bipolar_forceps') and perform('place',start,parameters={'radiusMm':2.,'jawOpeningMm':0.,'power':1.}) and perform('apply',tip,duration=1.) and perform('apply',duration=3.)
        if not ok:
            attempts.append({'index':index.tolist(),'stage':'coagulation','reason':records[-1]['reason']});continue
        sources=[s for s in engine.state['sources'] if s.get('kind')=='modeled-vessel-wound']
        if any(s['bleedingActive'] for s in sources):
            attempts.append({'index':index.tolist(),'stage':'coagulation','reason':'some local wounds outside bipolar exposure'});continue
        coag_metrics=engine.metrics()
        # Remain outside tissue with the physical catheter while sampling fluid.
        suction_tip=tip-inward*1.3
        suction_ok=perform('withdraw') and perform('exchange',instrument='frazier_suction') and perform('place',start,parameters={'radiusMm':3.,'flowMm3PerSecond':50.}) and perform('apply',suction_tip,duration=1.)
        metrics=engine.metrics();balance=metrics['fluidAddedMm3']-metrics['fluidPresentMm3']-metrics['fluidRemovedMm3']-metrics['fluidEscapedMm3']
        assert abs(balance)<1e-7 and not engine.state['removed']
        assert all(connection['patent'] is False and connection['perfusion']=='unknown' for connection in engine.state['vesselConnections'].values())
        assert array_digest(scene.labels)==before_labels and array_digest(case.mri)==before_mri
        restored=SurgicalEngine.restore(scene,json.loads(json.dumps(engine.snapshot())))
        assert restored.state_id==engine.state_id
        report={'schema':'resectionlab.experimental-patient-vessel-verification.v1','status':'passed',
                'sourceCase':str(Path(case_path).resolve()),'sourceIdentity':scene.identity,'labelArrayHash':before_labels,
                'intensityArrayHash':before_mri,'vesselRegion':vessel,'tissuePhysics':scene.properties['tissuePhysics'],
                'sourceUnchanged':True,'preRemovedCells':0,'estimatedVesselIndexIjk':index.tolist(),
                'suggestedGuiPose':{'start':start.tolist(),'contact':tip.tolist(),'quaternion':q},
                'actions':records,'failedAttempts':attempts,'woundsAfterCut':wounds_after_cut,
                'cutMetrics':cut_metrics,'coagulationMetrics':coag_metrics,'metrics':metrics,
                'suctionActionAccepted':suction_ok,'fluidBalanceErrorMm3':balance,'replayVerified':True,
                'connections':engine.state['vesselConnections'],'sources':sources,
                'stateId':engine.state_id,'elapsedSeconds':time.perf_counter()-started,
                'limitations':['Experimental image-derived vessel estimate; unvalidated anatomical vessel identity',
                               'Straight exposed-surface software test, not a clinical access route',
                               'Local voxel adjacency wounds and explicitly assumed outlet law, not patient physiology',
                               'No centerline reconstruction, vascular network, or tissue perfusion inference']}
        Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2)+'\n')
        return report
    report={'status':'not_demonstrated','attempts':attempts,'reason':'No bounded exposed estimate candidate completed the checked sequence','elapsedSeconds':time.perf_counter()-started}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case');parser.add_argument('output');args=parser.parse_args()
    result=verify(args.case,args.output);print(json.dumps({k:result[k] for k in ('status','elapsedSeconds')}))
    sys.exit(0 if result['status']=='passed' else 1)
