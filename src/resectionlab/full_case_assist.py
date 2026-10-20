"""Read-only stationary right-hand suction proposals in an actual opened cavity.

The field is a generic research radius, not a physical tip enlargement. Every
proposal still requires ordinary public tool setup, preview/step and guards.
This module neither mutates an engine nor manufactures field material.
"""
from __future__ import annotations

import numpy as np


def assistant_field_candidate(engine, current_access):
    """Return a separated free superior suction ray with real nearby field.

    Selects source-bound removed target columns, then checks current occupancy
    over the whole 0.5 mm physical tip/shaft insertion capsule, with an additional
    0.17 mm margin for the ordinary engine's <=1/3 mm translation microsteps.
    The stationary aspiration field is 1.5--3 mm, flow 20 mm^3/s and force zero.
    Field contents are measured from actual current debris/fluid dictionaries;
    a completely empty candidate is not meaningful assistance and is declined.
    """
    scene = engine.scene
    if not np.allclose(scene.affine[:3,:3], np.diag(np.diag(scene.affine[:3,:3]))) or scene.affine[2,2] <= 0:
        return None
    target_ids = {r['id'] for r in scene.regions if r.get('target')}
    removed = np.asarray(engine.state['removed'], dtype=np.int64)
    if not len(removed): return None
    labels = scene.labels.ravel()
    target = removed[np.isin(labels[removed], list(target_ids))]
    if not len(target): return None
    indices = np.column_stack(np.unravel_index(target, scene.shape))
    top_by_column = {}
    for i,j,k in indices:
        key=(int(i),int(j)); top_by_column[key]=max(top_by_column.get(key,-1),int(k))
    left = np.asarray(engine.state['slots']['left']['position'],float)
    access = np.asarray(current_access['entryPose']['position'],float)
    outside_z = float(scene.affine[2,3]+scene.affine[2,2]*(scene.shape[2]-1)+8.)
    owner = engine.occupancy()
    records=[]
    for (i,j),k in sorted(top_by_column.items()):
        point = scene.affine[:3,:3] @ np.asarray([i,j,k],float)+scene.affine[:3,3]
        left_distance=float(np.linalg.norm(point[:2]-left[:2]))
        access_distance=float(np.linalg.norm(point[:2]-access[:2]))
        if min(left_distance,access_distance) < 15.: continue
        point[2] += 2.
        outside=point.copy();outside[2]=outside_z
        # Downward physical tip plus the entire proximal shaft at every point.
        proximal=outside.copy();proximal[2]+=40.
        if owner.material_ids(engine.cells(point,proximal,.67)): continue
        # A physical tip can be free while its stationary field reaches debris.
        for field_radius in (3.,2.,1.5):
            cells = engine.cells(point,point,field_radius)
            debris=sum(float(engine.state.get('debris',{}).get(str(int(c)),0.)) for c in cells)
            fluid=sum(float(engine.state.get('fluid',{}).get(str(int(c)),0.)) for c in cells)
            if debris+fluid <= 1e-9: continue
            q=[1.,0.,0.,0.]
            records.append((debris+fluid,min(left_distance,access_distance),{
                'outsidePose':{'position':outside.tolist(),'quaternion':q},
                'fieldPose':{'position':point.tolist(),'quaternion':q},
                'parameters':{'radiusMm':field_radius,'depthMm':2.,'forceN':0.,'flowMm3PerSecond':20.},
                'instrument':'frazier_suction','slot':'right',
                'physicalTipRadiusMm':.5,'shaftRadiusMm':.5,
                'checkedInsertionCapsuleRadiusMm':.67,
                'distanceFromCurrentLeftXYMm':left_distance,
                'distanceFromCurrentAccessXYMm':access_distance,
                'minimumAccessSeparationMm':10.,
                'nearbyDebrisMm3':debris,'nearbyFluidMm3':fluid,
                'removedTargetColumn':[i,j], 'highestRemovedTargetIndexZ':k,
                'sourceSceneId':scene.identity,'requiresPublicPreview':True,
                'provenance':'actual removed source target column and current shared occupancy/fields',
                'assumptions':'generic stationary research suction field; unchanged physical shaft; no calibrated aspiration model',
                'coordination':'deactivate and withdraw before left access approaches within 10 mm in XY',
            }))
    if not records:return None
    records.sort(key=lambda row:(-row[0],-row[1]))
    return records[0][2]
