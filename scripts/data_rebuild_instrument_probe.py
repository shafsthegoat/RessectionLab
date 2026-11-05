#!/usr/bin/env python3
"""Bounded production instrument probe on an admitted engineering scenario.

This is an instrument/data-path test on estimated geometry, not a surgical plan,
clinical access claim or a new tumor variant. The target is left untouched.
"""
import json
from pathlib import Path
import numpy as np

from resectionlab.data_root import checked_root,atomic_json,sha256_file
from resectionlab.desktop_bridge import BridgeSession,_Request
from resectionlab.workspace_bundle import load_workspace


def run(root=None):
    root=checked_root(root,required_bytes=1024**3)
    source=root/'staging/candidates/synthetic/IXI-016-SYNTHETIC-appearance-01-v5.ressectionlab'
    case,_,_=load_workspace(source)
    center=np.asarray(case.metadata['data_case']['recipe']['center_ras_mm'])
    occupied=np.argwhere(case.brain_mask)
    world=case.voxel_to_world(occupied)
    # Locate first estimated brain support along a world-superior vertical ray,
    # using allowed observed/estimated geometry only, never private evaluation.
    z=np.arange(float(world[:,2].max()+10),float(world[:,2].min()),-.1)
    points=np.column_stack([np.full_like(z,center[0]),np.full_like(z,center[1]),z])
    voxels=np.rint(case.world_to_voxel(points)).astype(int)
    valid=np.all((voxels>=0)&(voxels<np.asarray(case.mri.shape)),axis=1)
    inside=np.zeros(len(z),bool);indices=voxels[valid];inside[valid]=case.brain_mask[tuple(indices.T)]
    first=points[np.flatnonzero(inside)[0]]
    start=first+np.array([0.,0.,8.]);end=first-np.array([0.,0.,.5])
    directory=root/'qc/curated-instrument-probe-v1';directory.mkdir(exist_ok=True)
    output=directory/'bounded-source-replay.ressectionlab'
    if output.exists():
        prior=json.loads((directory/'report.json').read_text())
        if sha256_file(output)!=prior['saved_sha256']:raise ValueError('Prior native replay changed')
        load_workspace(output);print(json.dumps({'cache_hit':True,'saved':str(output)}));return
    session=BridgeSession(directory/'transfers',run_dir=directory/'runs')
    records=[]
    def call(operation,args):
        return session.execute(operation,args,_Request(operation),lambda *_:None)
    try:
        installed=call('loadCase',{'path':str(source)});identity=installed['caseHash']
        current=call('surgicalRehearsal',{'caseHash':identity,'command':'create'})
        actions=[{'slot':'left','operation':'exchange','instrument':'cusa','duration':.1},
            {'slot':'left','operation':'place','duration':.1,'parameters':{'radiusMm':2.},
             'pose':{'position':start.tolist(),'quaternion':[1.,0.,0.,0.]}},
            {'slot':'left','operation':'apply','duration':.2,'parameters':{'flowMm3PerSecond':5.,'power':1.},
             'path':[{'position':end.tolist(),'quaternion':[1.,0.,0.,0.]}]}]
        for action in actions:
            result=call('surgicalRehearsal',{'caseHash':identity,'command':'step','expectedStateId':current['view']['stateId'],'action':action})
            records.append({'action':action,'result':result});current=result
            if result.get('outcome',result).get('status')=='rejected':break
        call('saveCase',{'caseHash':identity,'path':str(output)})
        reopened,_,_=load_workspace(output)
        if reopened.semantic_hash!=case.semantic_hash:raise ValueError('Instrument probe changed source anatomy')
        report={'schema':'ressectionlab-curated-bounded-instrument-probe-v1','source':str(source),'source_sha256':sha256_file(source),
            'saved':str(output),'saved_sha256':sha256_file(output),'source_case_hash_preserved':True,'new_people':0,'new_synthetic_variants':0,
            'role':'TRAIN','scope':'annotation-assisted engineering check on estimated brain; no actual skull exposure, clinical corridor or safe access claim',
            'first_estimated_brain_ras_mm':first.tolist(),'bounded_endpoint_ras_mm':end.tolist(),
            'flow_parameter_integral_mm3':1.,'flow_is_not_a_tissue_resection_budget':True,
            'assessment':'Negative engineering action: non-target tissue removal, zero target removal; not a proposed surgical plan',
            'actions':records,'training_performed':False}
        atomic_json(directory/'report.json',report)
        print(json.dumps({'saved':str(output),'actions':[(r['result'].get('status'),r['result'].get('reason')) for r in records],
            'metrics':current.get('view',{}).get('metrics')}))
    finally:session.close()


if __name__=='__main__':run()
