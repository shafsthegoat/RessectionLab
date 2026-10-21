"""One read-only source guard/vessel/entry-envelope diagnosis; no engine."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from resectionlab.geometry import GeometryScene,capsule_voxel_indices
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_rl_protection import ProtectionAnnotation,ProtectionSet,cell_radius
from resectionlab.surgical_tools import TOOLS
base=ROOT/'testcases/outputs/full-case-rehearsal-20261020'
spec=importlib.util.spec_from_file_location('residual_record_reader',base/'geometry/check_current_residual.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
scene,snapshot,meta,identity=reader.load_stable_records(base/'primary-v2/primary.ressectionlab')
state=snapshot['checkpoint']['state'];contract=snapshot['protectionContract']
assert not state['holds'] and not state.get('deformation',{}).get('nodes')
annotations=[ProtectionAnnotation(r['structureId'],tuple(r['cells']),contract['sourceBinding'],r['provenance'],r['reviewStatus'],r['displayName']) for r in contract['annotations']]
policy=ProtectionSet(scene,annotations,clearance_mm=contract['clearanceMm'],required=contract['required'])
removed=np.zeros(scene.shape,dtype=bool);removed.ravel()[np.asarray(state['removed'],dtype=np.int64)]=True
regions={r['id']:r for r in scene.regions};target_ids=[r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor'];vessel_ids=[r['id'] for r in scene.regions if r['material']=='vessel']
remaining=np.isin(scene.labels,target_ids)&~removed
residual_ids=np.flatnonzero(remaining);points=scene.points(residual_ids)
protected=policy.cells;guard_points=scene.points(protected)
dist,index=cKDTree(guard_points).query(points) if len(points) else (np.array([]),np.array([],int))
source_radius=cell_radius(scene.affine);base_guard=contract['clearanceMm']+source_radius
structures=[]
for row in contract['annotations']:
 ids=np.asarray(row['cells'],dtype=np.int64)
 d=cKDTree(scene.points(ids)).query(points)[0] if len(ids) and len(points) else np.array([])
 structures.append({'id':row['structureId'],'name':row['displayName'],'sourceCells':len(ids),'reviewStatus':row['reviewStatus'],'provenance':row['provenance'],
                    'minResidualCenterDistanceMm':float(d.min()) if len(d) else None,
                    'residualOwnedTargetIdentityOverlapCells':int(np.count_nonzero(remaining.ravel()[ids])),
                    'residualCentersWithinBaseGuardPlusPoint3Mm':int(np.count_nonzero(d<=base_guard+.3+1e-9))})
vessel=np.isin(scene.labels,vessel_ids)&~removed
vessel_points=scene.points(np.flatnonzero(vessel))
vessel_dist=cKDTree(vessel_points).query(points)[0] if len(points) and len(vessel_points) else np.array([])
# Exact source-cell closed-box distance for the small supplied owned-vessel set.
vessel_box_dist=np.full(len(points),np.inf)
for point in vessel_points:
 delta=np.maximum(np.abs(points-point)-scene.spacing/2.,0.)
 vessel_box_dist=np.minimum(vessel_box_dist,np.linalg.norm(delta,axis=1))
# Pure source-grid query object, NOT a SurgicalEngine or a mutable occupancy.
grid=GeometryScene(np.zeros(scene.shape,dtype=bool),scene.affine)

def labels_in_capsule(a,b,radius):
 indices=capsule_voxel_indices(grid,a,b,radius)
 if not len(indices):return []
 ids=np.ravel_multi_index(indices.T,scene.shape)
 return [int(c) for c in ids if scene.labels.ravel()[c] and not removed.ravel()[c]]

def describe(ids):
 result={}
 for c in ids:
  label=int(scene.labels.ravel()[c]);name=regions[label]['material']+(' target' if regions[label].get('target') else '')
  result[name]=result.get(name,0)+1
 return result

entries=[]
for access in meta['accesses']:
 assert access['nativeAxis']==0
 original_cell=int(np.ravel_multi_index(tuple(access['targetEntryIndex']),scene.shape))
 point=np.asarray(access['entryPose']['position'],float);q=access['entryPose']['quaternion'];outward=-Rotation.from_quat(q).apply([0,0,1])
 d,at=cKDTree(guard_points).query(point)
 row={'id':access['id'],'targetEntryIndex':access['targetEntryIndex'],'entryRAS':point.tolist(),'entryOwnedTargetStillRetained':bool(remaining.ravel()[original_cell]),
      'minEntryCenterToGuardCellCenterMm':float(d),'nearestGuardCellSourceId':int(protected[at]),'nearestGuardNames':[a.structure_id for a in annotations if int(protected[at]) in a.cells],
      'bounds':[]}
 for radius in (.3,1.3):
  for name,center in [('at-entry-center',point),('just-contact-outer-cell-face',point+outward*(scene.spacing[0]/2.+radius-1e-4))]:
   work=set();shaft=set();guard=set()
   for a,b,width,part in envelopes(TOOLS['cusa'],{'radiusMm':radius,'depthMm':2.},center,q):
    ids=labels_in_capsule(a,b,width)
    (shaft if part=='shaft' else work).update(ids)
    guard.update(map(int,policy.capsule_hits(a,b,width)))
   tip=set(labels_in_capsule(center,center,radius))
   row['bounds'].append({'radiusMm':radius,'pose':name,'guardHitCells':len(guard),
                         'shaftRetainedLabels':describe(shaft),'workingRetainedLabels':describe(work|tip),
                         'shaftBlocksThisStaticPose':bool(shaft),
                         'workingNontargetBlocksTargetOnlyAdmission':any(int(scene.labels.ravel()[c]) not in target_ids for c in work|tip),
                         'tipWouldIntersectOriginalEntryCell':original_cell in tip,
                         'representativeShaftSourceIds':sorted(shaft)[:4],
                         'representativeNontargetWorkingSourceIds':[c for c in sorted(work|tip) if int(scene.labels.ravel()[c]) not in target_ids][:4]})
 entries.append(row)
report={'readOnly':True,'noEngineConstructionPreviewStepReplay':True,'position':len(snapshot['actions']),'stateId':snapshot['stateId'],'sceneId':scene.identity,
        'identityVerification':identity,'remainingOwnedTargetCells':int(remaining.sum()),'guardContractId':contract['contentHash'],
        'guardConfig':contract['config'],'clearanceMm':contract['clearanceMm'],'sourceCellCircumsphereRadiusMm':source_radius,
        'guardSemantics':'Each working and physical-shaft swept capsule is excluded when distance from its centerline to any protected source-cell center <= tool capsule radius + swept inflation + clearance + source-cell circumsphere radius. Swept inflation is translation/2 +84*sin(rotationAngle/4). Protected memberships persist across allowed movements; displacement/thermal damage is also checked. This receipt has no holds/deformation, so original protected positions are current.',
        'guardStructures':structures,
        'residualAdjacency':{'minGuardCenterDistanceMm':float(dist.min()) if len(dist) else None,
            'centerDistanceQuantilesMm':dict(zip(['min','p10','median'],map(float,np.quantile(dist,[0,.1,.5])))) if len(dist) else {},
            'centersWithinBaseGuardMm':int(np.count_nonzero(dist<=base_guard+1e-9)),
            'centersWithinBaseGuardPlusPoint3Mm':int(np.count_nonzero(dist<=base_guard+.3+1e-9)),
            'centersWithinBaseGuardPlusShaftPoint5Mm':int(np.count_nonzero(dist<=base_guard+.5+1e-9)),
            'centersWithinBaseGuardPlusWorking1Point3Mm':int(np.count_nonzero(dist<=base_guard+1.3+1e-9)),
            'ownedVesselCells':int(vessel.sum()),'minVesselCenterDistanceMm':float(vessel_dist.min()) if len(vessel_dist) else None,
            'residualFaceAdjacentToOwnedVesselCells':int(np.count_nonzero(remaining&ndi.binary_dilation(vessel,structure=ndi.generate_binary_structure(3,1)))),
            'residualCentersWithin1MmOfVesselClosedBoxes':int(np.count_nonzero(vessel_box_dist<=1.+1e-9)),
            'residualCentersWithin2MmOfVesselClosedBoxes':int(np.count_nonzero(vessel_box_dist<=2.+1e-9))},
        'leftEntryStaticBounds':entries,
        'limits':['Exact source masks/ownership and static full-tool geometry only; no public action or physical trajectory was admitted or tested.',
                  'Center distances are not proof that every voxel-contact pose or an alternative tilted/subvoxel route is infeasible.',
                  'A blocked exact center/contact pose does not prove the source cell cannot be removed by contact at another face or through another opening.',
                  'Guard estimates are unreviewed and do not supply motor/language cortex, tracts, optic pathways or cranial nerves. Owned vessel mask is experimental and incomplete; no completeness or clinical safety claim.']}
out=base/f'review/residual-guard-adjacency-{len(snapshot["actions"])}.json'
with out.open('x') as f:json.dump(report,f,indent=2)
print('position',report['position'],'remaining',report['remainingOwnedTargetCells'])
print('structures',json.dumps(structures,indent=2))
print('adjacency',json.dumps(report['residualAdjacency'],indent=2))
for row in entries: print(json.dumps({'id':row['id'],'retained':row['entryOwnedTargetStillRetained'],'guardDistance':row['minEntryCenterToGuardCellCenterMm'],'bounds':row['bounds']},indent=2))
print(out)
