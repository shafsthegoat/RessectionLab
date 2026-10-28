"""Bounded source-only shortlist of mapped-brain corridors with sector capacity.

This is privileged planning geometry. It does not execute a tool, estimate
clinical safety, or expose source tumor membership to an actor observation.
The classic access selector remains available for its existing callers.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .core import array_digest, semantic_digest
from .geometry import GeometryScene, capsule_voxel_indices, point_segment_distances
from .surgical_access_planning import (
    _brain_regions, _brain_route_novelty, _layer_evidence,
)
from .surgical_sector_planning import (
    MIN_SECTOR_MM3, SECTOR_RADIAL_MM, SECTOR_STROKE,
    SECTOR_WINDOWS_MM, _sector_cells,
)


_AXES = (2, 0, 1)
_MAX_PATH_MM = 45.
_SHAFT_LENGTH_MM = 40.
_SHAFT_RADIUS_MM = .5
_MIN_WORKING_TARGET_MM3 = 4.  # Source-only shortlist cutoff, not a task gate.
_SOURCE_CONTACT_POLICY_MM3 = 200.  # Contact, not calibrated removal.
_FINALISTS_PER_AXIS = 64
_STAGED_CONTINUATION_BRAIN_REFERENCE_MM3 = 20.
_STAGED_WHOLE_TARGET_FLOOR_MM3 = 500.
_STAGED_WHOLE_TARGET_FRACTION = .1


def _points(scene, indices):
    return indices @ scene.affine[:3, :3].T + scene.affine[:3, 3]


def _source_labels_at(scene, points):
    indices = np.rint((points-scene.affine[:3, 3]) @ scene.inverse[:3, :3].T).astype(int)
    inside = np.all((indices >= 0) & (indices < np.asarray(scene.shape)), axis=1)
    values = np.full(len(indices), -1, dtype=np.int32)
    if np.any(inside):
        values[inside] = scene.labels[tuple(indices[inside].T)]
    return values, indices, inside


def _working_sphere(scene, center, radius, brain_ids, target_ids):
    """Count source cell centers in one fixed working sphere, in RAS mm."""
    index = scene.inverse[:3, :3] @ (center-scene.affine[:3, 3])
    extent = np.linalg.norm(scene.inverse[:3, :3], axis=1)*radius + 1.
    lo = np.maximum(0, np.ceil(index-extent).astype(int))
    hi = np.minimum(np.asarray(scene.shape)-1, np.floor(index+extent).astype(int))
    if np.any(lo > hi):
        return 0, 0
    grid = np.stack(np.meshgrid(*(np.arange(lo[i], hi[i]+1) for i in range(3)),
                                indexing='ij'), axis=-1).reshape(-1, 3)
    near = np.linalg.norm(_points(scene, grid)-center, axis=1) <= radius+1e-9
    values = scene.labels[tuple(grid[near].T)]
    return int(np.count_nonzero(np.isin(values, target_ids))), int(np.count_nonzero(
        np.isin(values, brain_ids)))


def _route(scene, item, brain_regions, target_regions):
    direction = np.asarray(item['inwardRasUnit'])
    surface = np.asarray(item['surfaceRasMm'])
    goal = np.asarray(item['targetRasMm'])
    quat = Rotation.align_vectors([direction], [[0., 0., 1.]])[0].as_quat().tolist()
    outside = surface-direction*8.
    radius = 1. if item['pathLengthMm'] <= 20. else .6
    return {
        'id': 'mapped-brain-joint-'+str(item['axis'])+
              ('-negative' if item.get('sourceFaceSign', 1) == -1 else '')+
              '-'+str(item['surfaceFlatId'])+
              '-'+str(item['targetFlatId']),
        'nativeAxis': item['axis'], 'pathType': 'oblique_mapped_brain',
        **({'sourceFaceSign': item['sourceFaceSign']} if 'sourceFaceSign' in item else {}),
        'entryPose': {'position': goal.tolist(), 'quaternion': quat},
        'outsidePose': {'position': outside.tolist(), 'quaternion': quat},
        'surfacePose': {'position': surface.tolist(), 'quaternion': quat},
        'inwardRasUnit': direction.tolist(), 'pathLengthMm': item['pathLengthMm'],
        'noveltyComparisons': [], 'targetEntryIndex': item['targetBoundaryIndex'],
        'sourceBrainSurfaceIndex': item['surfaceIndex'][item['axis']],
        'sourceBrainCellsOnRay': item['sourceBrainCellsOnRay'],
        'sourceBrainThicknessMm': item['pathLengthMm'],
        'unmappedExterior': True, 'corridorRadiusMm': radius,
        'drillRadiusMm': None, 'routePreflightCandidates': 0,
        'geometricTargetCapacityCells': item['sourceTargetCellsOnRay'],
        'sourceSceneId': scene.identity,
        'sourceBrainLayers': [_layer_evidence(scene, r['name']) for r in brain_regions],
        'sourceTargetLayers': [_layer_evidence(scene, r['name']) for r in target_regions],
        'accessCompleteness': 'unsupported_skull_access_brain_corridor',
        'evidence': ('source-only mapped brain and tumor ray, exterior source void, '
                     'exterior-anchored fixed CUSA shaft; native swept clearance and '
                     'skull/vascular/functional completeness unverified'),
    }


def _geometry_burden(scene, grid, item, brain_ids, target_ids):
    """Whole source corridor/shaft occupancy; no native transitions."""
    surface = np.asarray(item['surfaceRasMm'])
    inward = np.asarray(item['inwardRasUnit'])
    length = item['pathLengthMm']
    radius = 1. if length <= 20. else .6
    start = surface-inward*1.5
    end = surface+inward*(length+.5)
    corridor = capsule_voxel_indices(grid, start, end, radius)
    labels = scene.labels[tuple(corridor.T)]
    brain = np.isin(labels, brain_ids)
    distances = point_segment_distances(grid.voxel_to_world(corridor), start, end)
    center_contact = int(np.count_nonzero(brain & (distances <= radius+1e-9)))
    brain_supercover = int(np.count_nonzero(brain))
    far_tip = surface+inward*(length+SECTOR_STROKE['far'][0])
    shaft = capsule_voxel_indices(grid, far_tip-inward*_SHAFT_LENGTH_MM,
                                  far_tip-inward*2., _SHAFT_RADIUS_MM)
    shaft_labels = scene.labels[tuple(shaft.T)]
    allowed = np.isin(shaft_labels, [0, *brain_ids, *target_ids]).all()
    shaft_brain = int(np.count_nonzero(np.isin(shaft_labels, brain_ids)))
    # Count each source brain cell once across the access corridor and shaft.
    corridor_flat = np.ravel_multi_index(corridor[brain].T, scene.shape)
    shaft_flat = np.ravel_multi_index(shaft[np.isin(shaft_labels, brain_ids)].T,
                                     scene.shape)
    union_brain = int(len(np.union1d(corridor_flat, shaft_flat)))
    measurements = {'corridorCenterContactBrainMm3': center_contact*scene.voxel_mm3,
            'corridorBrainSupercoverMm3': brain_supercover*scene.voxel_mm3,
            'farShaftBrainSupercoverMm3': shaft_brain*scene.voxel_mm3,
            'corridorAndShaftBrainUnionMm3': union_brain*scene.voxel_mm3,
            'farShaftSourceLabelsAllowed': bool(allowed),
            'farShaftTailDepthMm': length+SECTOR_STROKE['far'][0]-_SHAFT_LENGTH_MM}
    corridor_all = set(map(int,np.ravel_multi_index(corridor.T,scene.shape)))
    corridor_centers = set(map(int,np.ravel_multi_index(
        corridor[distances<=radius+1e-9].T,scene.shape)))
    return measurements,corridor_all,corridor_centers


def _fixed_far_first_contact(scene, grid, item, corridor, corridor_centers,
                             brain_flat, target_flat):
    """Undeformed source contact union for the fixed far-first-left tip macro.

    Individual capsules are source voxel supercovers and cell-center bounds.
    Their set difference against the corridor is a ranking proxy; it is not
    itself an upper bound on incremental native contact or removal.
    """
    surface=np.asarray(item['surfaceRasMm'])
    inward=np.asarray(item['inwardRasUnit'])
    length=item['pathLengthMm']
    pose=lambda offset:surface+inward*(length+offset)
    phases=((pose(0.),pose(4.),1.),
            (pose(4.),pose(4.),1.8),
            (pose(4.),pose(1.),1.6))
    supercovers=[]
    centers=[]
    for start,end,radius in phases:
        ijk=capsule_voxel_indices(grid,start,end,radius)
        flat=np.ravel_multi_index(ijk.T,scene.shape)
        supercovers.append(set(map(int,flat)))
        distance=point_segment_distances(grid.voxel_to_world(ijk),start,end)
        centers.append(set(map(int,flat[distance<=radius+1e-9])))
    macro=set.union(*supercovers)
    macro_centers=set.union(*centers)
    additional=macro-corridor
    additional_centers=macro_centers-corridor_centers
    def source_volume(ids,mask):
        if not ids:return 0.
        ordered=np.fromiter(ids,dtype=np.int64)
        return float(np.count_nonzero(mask[ordered])*scene.voxel_mm3)
    near=set(item['_nearSectorFlatIds'])
    far=set(item['_farSectorFlatIds'])
    near_coverage=len(additional&near)*scene.voxel_mm3
    far_coverage=len(additional&far)*scene.voxel_mm3
    brain_super=source_volume(additional,brain_flat)
    brain_center=source_volume(additional_centers,brain_flat)
    near_deficit=max(0.,MIN_SECTOR_MM3-near_coverage)
    far_deficit=max(0.,MIN_SECTOR_MM3-far_coverage)
    brain_excess=max(0.,brain_super-20.)
    return {
        'farFirstAdditionalBrainSupercoverMm3':brain_super,
        'farFirstAdditionalBrainCenterMm3':brain_center,
        'farFirstAdditionalTargetSupercoverMm3':source_volume(additional,target_flat),
        'farFirstSectorSourceContactMm3':{'near':near_coverage,'far':far_coverage},
        'farFirstNearSourceContactDeficitMm3':near_deficit,
        'farFirstFarSourceContactDeficitMm3':far_deficit,
        'farFirstBrainSourceContactExcessMm3':brain_excess,
        'farFirstJointSourceDeficitMm3':near_deficit+far_deficit+brain_excess,
        'farFirstMaxSourceDeficitMm3':max(near_deficit,far_deficit,brain_excess),
        'farFirstPhaseBrainBeyondCorridorMm3':{
            name:source_volume(part-corridor,brain_flat)
            for name,part in zip(('farTransit','farWorking','nearReturn'),supercovers)}}


def _staged_work_tube_contact(scene, grid, item, corridor,
                              brain_flat, target_flat, target_reference_mm3):
    """Undeformed extended ray contact beyond the source access capsule.

    This fixed-radius supercover is a bounded shortlist proxy. A dynamically
    cleared or deformed native scene can touch a different set of cells.
    """
    surface = np.asarray(item['surfaceRasMm'], float)
    inward = np.asarray(item['inwardRasUnit'], float)
    length = float(item['pathLengthMm'])
    far_offset = min(24., 38.-length)
    if far_offset < 1.:
        raise ValueError('Staged source tube has no exterior-anchored working depth')
    radius = 1. if length <= 20. else .6
    start = surface+inward*(length+1.)
    end = surface+inward*(length+far_offset)
    ijk = capsule_voxel_indices(grid, start, end, radius)
    tube = set(map(int, np.ravel_multi_index(ijk.T, scene.shape)))
    shaft = capsule_voxel_indices(grid, end-inward*_SHAFT_LENGTH_MM,
                                  end-inward*2., _SHAFT_RADIUS_MM)
    shaft_labels = scene.labels[tuple(shaft.T)]
    allowed_ids = {0, *[r['id'] for r in _brain_regions(scene)],
                   *[r['id'] for r in scene.regions if r.get('target') and
                      r['material'] == 'tumor']}
    deep_shaft_allowed = bool(np.isin(shaft_labels, list(allowed_ids)).all())
    additional = tube-corridor
    ordered = np.fromiter(additional, dtype=np.int64)
    brain = float(np.count_nonzero(brain_flat[ordered])*scene.voxel_mm3)
    target = float(np.count_nonzero(target_flat[ordered])*scene.voxel_mm3)
    brain_excess = max(0., brain-_STAGED_CONTINUATION_BRAIN_REFERENCE_MM3)
    # The rank uses missing target contact, not target contact above the
    # objective. Keep the explicit positive deficit in the sealed report.
    target_deficit = max(0., target_reference_mm3-target)
    return {
        'stagedWorkTubeOffsetFromTargetEntryMm': [1., far_offset],
        'stagedWorkTubeRadiusMm': radius,
        'stagedDeepShaftSourceLabelsAllowed': deep_shaft_allowed,
        'stagedDeepShaftTailDepthMm': length+far_offset-_SHAFT_LENGTH_MM,
        'stagedWorkTubeSourceCellCount': len(tube),
        'stagedWorkTubeBeyondCorridorCellCount': len(additional),
        'stagedWorkTubeAdditionalTargetSupercoverMm3': target,
        'stagedWorkTubeAdditionalBrainSupercoverMm3': brain,
        'stagedContinuationBrainExcessMm3': brain_excess,
        'stagedTargetContactDeficitMm3': target_deficit,
        'stagedJointSourceDeficitMm3': brain_excess+target_deficit}


def select_joint_corridors(scene, *, max_routes=3, max_surface_samples=4096,
                           nearest_targets=8, max_geometry_pairs=32768,
                           goal='two-sector', include_opposite_faces=False,
                           check=lambda: None):
    """Return at most one source-only route per native axis and a sealed report.

    The bounded scan samples mapped-brain exterior faces, pairing each with
    nearby source tumor-boundary cells. Both goals require an exterior-anchored
    fixed shaft and the same mapped source labels. ``two-sector`` retains the
    original fixed near/far capacity cuts; ``staged`` records those capacities
    without requiring them and ranks extended source work-tube contact first.
    Source contact cutoffs are shortlist heuristics, never modeled removal or
    surgical safety claims. With ``include_opposite_faces=True``, split the SAME
    surface/pair budgets across both exterior signs of each native axis. Face
    sign is in source-index coordinates, independent of the RAS affine.
    """
    if type(include_opposite_faces) is not bool:
        raise ValueError('include_opposite_faces must be boolean')
    bounds = (max_routes, max_surface_samples, nearest_targets, max_geometry_pairs)
    if any(type(v) is not int or v < 1 for v in bounds):
        raise ValueError('Joint corridor bounds must be positive integers')
    if max_routes > 3:
        raise ValueError('At most one route per three native axes is supported')
    if goal not in ('two-sector', 'staged'):
        raise ValueError('Joint corridor goal must be two-sector or staged')
    if not callable(check):
        raise ValueError('check must be callable')
    check()
    regions = _brain_regions(scene)
    targets = [r for r in scene.regions if r.get('target') and r['material']=='tumor']
    brain_ids = [r['id'] for r in regions]
    target_ids = [r['id'] for r in targets]
    staged_target_reference = (max(_STAGED_WHOLE_TARGET_FLOOR_MM3,
        _STAGED_WHOLE_TARGET_FRACTION*float(np.count_nonzero(
            np.isin(scene.labels, target_ids))*scene.voxel_mm3))
        if goal == 'staged' else None)
    basis = {'sceneId': scene.identity, 'sourceCaseHash': scene.source.get('caseHash'),
             'sourceLabelsHash': array_digest(scene.labels),
             'affineRasMm': scene.affine.tolist(),
             'brainLayers': [_layer_evidence(scene, r['name']) for r in regions],
             'targetLayers': [_layer_evidence(scene, r['name']) for r in targets],
             'skullSourceAvailable': any(r['material']=='bone' and
                np.any(scene.labels==r['id']) for r in scene.regions),
             'shaftLengthMm': _SHAFT_LENGTH_MM,
             'shaftRadiusMm': _SHAFT_RADIUS_MM}
    histogram = {}
    def reject(name):
        histogram[name] = histogram.get(name, 0)+1
    rule = {'nativeAxes': list(_AXES),'maxRoutes': max_routes,
            'maxSurfaceSamples': max_surface_samples,
            'nearestTargets': nearest_targets,
            'maxGeometryPairs': max_geometry_pairs,
            'maxPathMm': _MAX_PATH_MM,
            'minSourceSectorVolumeMm3': MIN_SECTOR_MM3,
            'sectorWindowsMm': {k:list(v) for k,v in SECTOR_WINDOWS_MM.items()},
            'sectorRadialMm': list(SECTOR_RADIAL_MM),
            'minWorkingTargetMm3': _MIN_WORKING_TARGET_MM3,
            'workingFootprintMeaning': ('source cell centers inside fixed near/far '
                                        'stroke spheres; not swept native removal'),
            'fixedFarFirstMacro':{
                'hand': 'left CUSA; source tip geometry is hand independent',
                'farTransit': 'target entry to depth +4 mm, radius 1 mm',
                'farWorking': 'depth +4 mm sphere, radius 1.8 mm',
                'nearReturn': 'depth +4 to +1 mm, radius 1.6 mm',
                'incremental': ('macro source-contact union minus corridor tip supercover; '
                                'set difference is a ranking proxy, not an incremental contact bound'),
                'contactMeaning': 'undeformed source capsule supercover and cell-center overlap; not native removal'},
            'workingContactReferenceMm3':{'near':10.,'far':10.,'additionalBrain':20.},
            'workingContactReferenceUse':'ranking deficits only; authoritative task gates unchanged',
            'jointRankFormula':('near sector contact deficit to 10 + far sector contact deficit '
                                'to 10 + additional brain supercover excess over 20; then '
                                'largest individual deficit, corridor/shaft brain union, '
                                'corridor burden, path length, target coverage and action headroom'),
            'sourceContactPolicyCutoffMm3': _SOURCE_CONTACT_POLICY_MM3,
            'sourceContactPolicyMeaning': 'geometric shortlist cutoff; not modeled removal proof',
            'farTipExteriorAnchoring': True,
            'heuristicActionMeaning': 'ceil(path/3) assumes travel to boundary; aspiration may contact sooner',
            'finalistsPerAxis': _FINALISTS_PER_AXIS,
            'noveltyMinEntryMm': 12., 'noveltyMinAngleDeg': 30.}
    if include_opposite_faces:
        rule.update(includeOppositeFaces=True, sourceExteriorFaceSigns=[1, -1],
                    sourceFaceSignMeaning='outward neighbor sign in native index axis',
                    surfaceBudgetMeaning='shared across all six axis/sign face groups')
    if goal == 'staged':
        rule.update(selectionGoal='staged',
            minSourceSectorVolumeMm3=None,
            minWorkingTargetMm3=None,
            sourceCuts={'required': ['mapped brain/tumor ray', 'source-void exterior',
                'exterior-anchored fixed 40 mm/0.5 mm shaft',
                'full-shaft allowed source labels', 'corridor center brain contact at most 200 mm3',
                '12 mm entry and 30 degree direction novelty'],
                'diagnosticOnly': ['near/far 10 mm3 sector capacity',
                    'two fixed 4 mm3 working spheres',
                    '48-action two-sector heuristic']},
            sourceSectorCapacityUse='recorded only; no staged eligibility cutoff',
            workingFootprintUse='recorded only; no staged eligibility cutoff',
            workingContactReferenceUse='recorded diagnostic only; no staged rank or task gate',
            stagedMaxNativeActions=240,
            stagedSourceTargetContactReferenceMm3=staged_target_reference,
            stagedSourceTargetContactReferenceDefinition=(
                'max(500 mm3, 10% of source target volume), matching the larger '
                'research task whole-target threshold; tube contact is not removal'),
            stagedContinuationBrainContactReferenceMm3=
                _STAGED_CONTINUATION_BRAIN_REFERENCE_MM3,
            stagedHeuristicHeadroomMeaning=('240 minus the old fixed-sequence action '
                'estimate, retained only as a source planning diagnostic; extended '
                'work and checked exit consume additional actions'),
            stagedLocalTargetContactMeaning=('source target centers within 7 mm of the '
                'target boundary, at nonnegative axial depth and at most 6 mm from the '
                'access ray; shortlist proxy, not native reachability'),
            stagedTubeContactMeaning=('undeformed source capsule supercover from '
                'target entry +1 mm through +min(24 mm, 38 mm minus corridor length) '
                'at unchanged .6/1 mm corridor radius, minus source access capsule; '
                'neither predicted native removal nor a reachability proof'),
            stagedDeepShaftMeaning=('unchanged 40 mm/0.5 mm shaft checked at the '
                'deepest modeled tube tip, with its proximal tail at least 2 mm '
                'outside the source brain surface'),
            jointRankFormula=('among at most 64 exact finalists per axis: extended '
                'tube brain contact excess over 20 mm3 plus target contact deficit '
                'relative to the larger task whole-target threshold; then whole '
                'corridor/shaft brain supercover union, path length, broader tube '
                'target contact and deterministic source indices'))
    report = {'schema': 'resectionlab.joint-source-corridor-selection.v1',
              'task': ('source-only-joint-access-sector-shortlist' if goal == 'two-sector'
                       else 'source-only-joint-access-staged-shortlist'),
              'sourceBasis': basis,'selectionRule': rule,
              'counts': {'surfacePointsAvailable': {}, 'surfacePointsSampled': {},
                         'pairsExamined':0,'cheapEligible':0,'exactFinalists':0,
                         'rankedFinalists':0,'selectedRoutes':0},
              'rejectionHistogram': histogram, 'selected':[],
              'status': 'insufficient_routes'}
    if not brain_ids or not target_ids:
        reject('MISSING_MAPPED_BRAIN_OR_TARGET')
        report['selectionSeal'] = semantic_digest(report)
        return [], report
    labels = scene.labels
    brain = np.isin(labels, brain_ids)
    tumor = np.isin(labels, target_ids)
    boundary = np.argwhere(tumor & ~binary_erosion(tumor))
    if not len(boundary):
        reject('EMPTY_TARGET_BOUNDARY')
        report['selectionSeal'] = semantic_digest(report)
        return [], report
    k = min(nearest_targets, len(boundary), max_geometry_pairs)
    sample_budget = min(max_surface_samples, max_geometry_pairs//k)
    if not sample_budget:
        reject('GEOMETRY_PAIR_BUDGET_TOO_SMALL')
        report['selectionSeal'] = semantic_digest(report)
        return [], report
    face_groups = ([(axis, sign) for axis in _AXES for sign in (1, -1)]
                   if include_opposite_faces else [(axis, 1) for axis in _AXES])
    per_face = {group: sample_budget//len(face_groups) +
                (i < sample_budget%len(face_groups))
                for i, group in enumerate(face_groups)}
    boundary_world = _points(scene, boundary)
    target_tree = cKDTree(boundary_world)
    tumor_ijk=np.argwhere(tumor)
    tumor_world = _points(scene, tumor_ijk)
    tumor_flat=np.ravel_multi_index(tumor_ijk.T,scene.shape)
    tumor_tree = cKDTree(tumor_world)
    brain_flat=brain.ravel()
    target_flat=tumor.ravel()
    preliminary = {axis: [] for axis in _AXES}
    for axis, face_sign in face_groups:
        check()
        face = np.zeros_like(brain)
        inner = [slice(None)]*3; outer = [slice(None)]*3
        if face_sign == 1:
            inner[axis] = slice(0,-1); outer[axis] = slice(1,None)
        else:
            inner[axis] = slice(1,None); outer[axis] = slice(0,-1)
        face[tuple(inner)] = brain[tuple(inner)] & (labels[tuple(outer)]==0)
        surface = np.argwhere(face)
        face_key = f'{axis}:{face_sign:+d}' if include_opposite_faces else str(axis)
        report['counts']['surfacePointsAvailable'][face_key] = int(len(surface))
        n = min(len(surface), per_face[(axis, face_sign)])
        sampled = (surface[np.linspace(0,len(surface)-1,n,dtype=int)] if n else
                   np.empty((0,3),dtype=int))
        report['counts']['surfacePointsSampled'][face_key] = int(len(sampled))
        if not n:
            reject('NO_SOURCE_EXTERIOR_FACE_AXIS_'+face_key)
            continue
        surface_world = _points(scene, sampled)
        distances, neighbors = target_tree.query(surface_world,k=k)
        if k==1:
            distances=distances[:,None]; neighbors=neighbors[:,None]
        for row, entry_index in enumerate(sampled):
            check()
            surface_point = surface_world[row]
            for col in range(k):
                report['counts']['pairsExamined'] += 1
                length = float(distances[row,col])
                if not 1. <= length <= _MAX_PATH_MM:
                    reject('PATH_LENGTH'); continue
                goal_index = boundary[int(neighbors[row,col])]
                if face_sign*(goal_index[axis]-entry_index[axis])>=-1:
                    reject('WRONG_INWARD_AXIS'); continue
                # The 40 mm shaft must still span the source surface at the
                # predeclared far working tip. This is a bounded access mode,
                # not a proof that deeper routes are physically impossible.
                if length+SECTOR_STROKE['far'][0] > _SHAFT_LENGTH_MM+1e-9:
                    reject('FAR_TIP_SHAFT_NOT_EXTERIOR_ANCHORED'); continue
                target_point = boundary_world[int(neighbors[row,col])]
                inward = (target_point-surface_point)/length
                nearby = tumor_tree.query_ball_point(target_point,7.)
                if not nearby:
                    reject('NO_LOCAL_TARGET'); continue
                points = tumor_world[nearby]
                relative = points-surface_point
                depth = relative@inward
                radial = np.linalg.norm(relative-depth[:,None]*inward,axis=1)
                sector_members={name:(depth>=length+lo)&(depth<length+hi)&
                    (radial>=SECTOR_RADIAL_MM[0])&(radial<SECTOR_RADIAL_MM[1])
                    for name,(lo,hi) in SECTOR_WINDOWS_MM.items()}
                capacity={name:int(np.count_nonzero(mask))*scene.voxel_mm3
                          for name,mask in sector_members.items()}
                if goal == 'two-sector' and any(value < MIN_SECTOR_MM3
                                                for value in capacity.values()):
                    reject('SECTOR_CAPACITY'); continue
                local_reachable = ((depth >= length) & (depth <= length+7.) &
                                   (radial <= 6.))
                local_target_contact = float(np.count_nonzero(local_reachable)*
                                             scene.voxel_mm3)
                near_center = surface_point+inward*(length+SECTOR_STROKE['near'][0])
                far_center = surface_point+inward*(length+SECTOR_STROKE['far'][0])
                near_t,near_b=_working_sphere(scene,near_center,SECTOR_STROKE['near'][1],
                                              brain_ids,target_ids)
                far_t,far_b=_working_sphere(scene,far_center,SECTOR_STROKE['far'][1],
                                            brain_ids,target_ids)
                if goal == 'two-sector' and min(near_t,far_t)*scene.voxel_mm3 < \
                        _MIN_WORKING_TARGET_MM3:
                    reject('WORKING_TARGET_FOOTPRINT'); continue
                count = max(2,int(math.ceil(length/.4)))
                ray = surface_point[None,:]+np.linspace(0.,length,count)[:,None]*inward
                ray_values,ray_indices,inside=_source_labels_at(scene,ray)
                if not inside.all() or not np.isin(ray_values[:-1],brain_ids+target_ids).all() or \
                        not np.isin(ray_values,brain_ids).any() or \
                        not np.isin(ray_values,target_ids).any():
                    reject('UNMAPPED_SOURCE_RAY'); continue
                first_tumor=np.flatnonzero(np.isin(ray_values,target_ids))[0]
                if length*(1.-first_tumor/(count-1))>2.:
                    reject('TARGET_BOUNDARY_NOT_FIRST_CONTACT'); continue
                exterior = np.linspace(surface_point-inward*.8,
                                       surface_point-inward*8.,20)
                exterior_values,_,exterior_inside=_source_labels_at(scene,exterior)
                if not exterior_inside.all() or np.any(exterior_values!=0):
                    reject('EXTERIOR_NOT_SOURCE_VOID'); continue
                far_tip=far_center
                shaft_line=np.linspace(far_tip-inward*_SHAFT_LENGTH_MM,
                                       far_tip-inward*2.,96)
                shaft_values,_,shaft_inside=_source_labels_at(scene,shaft_line)
                if not shaft_inside.all() or not np.isin(
                        shaft_values,[0,*brain_ids,*target_ids]).all():
                    reject('UNMAPPED_SHAFT_CENTERLINE'); continue
                brain_ijk=ray_indices[np.isin(ray_values,brain_ids)]
                brain_count=len(np.unique(np.ravel_multi_index(brain_ijk.T,scene.shape)))
                target_ijk=ray_indices[np.isin(ray_values,target_ids)]
                target_count=len(np.unique(np.ravel_multi_index(target_ijk.T,scene.shape)))
                item={'axis':axis,'surfaceIndex':entry_index.astype(int).tolist(),
                      'targetBoundaryIndex':goal_index.astype(int).tolist(),
                      'surfaceFlatId':int(np.ravel_multi_index(entry_index,scene.shape)),
                      'targetFlatId':int(np.ravel_multi_index(goal_index,scene.shape)),
                      'surfaceRasMm':surface_point.tolist(),
                      'targetRasMm':target_point.tolist(),
                      'inwardRasUnit':inward.tolist(),'pathLengthMm':length,
                      'sourceBrainCellsOnRay':brain_count,
                      'sourceTargetCellsOnRay':target_count,
                      'sourceSectorCapacityMm3':capacity,
                      '_nearSectorFlatIds':tumor_flat[np.asarray(nearby,dtype=int)][
                          sector_members['near']].astype(int).tolist(),
                      '_farSectorFlatIds':tumor_flat[np.asarray(nearby,dtype=int)][
                          sector_members['far']].astype(int).tolist(),
                      'stagedLocalTargetContactMm3':local_target_contact,
                      'nearWorkingTargetMm3':near_t*scene.voxel_mm3,
                      'farWorkingTargetMm3':far_t*scene.voxel_mm3,
                      'nearWorkingBrainMm3':near_b*scene.voxel_mm3,
                      'farWorkingBrainMm3':far_b*scene.voxel_mm3}
                if include_opposite_faces:
                    item['sourceFaceSign'] = face_sign
                preliminary[axis].append(item)
                report['counts']['cheapEligible'] += 1
    grid = GeometryScene(np.zeros(scene.shape,bool),scene.affine)
    finalists=[]
    for axis in _AXES:
        # Bounded exact capsule work, chosen without any native outcomes.
        if goal == 'two-sector':
            preliminary[axis].sort(key=lambda r:(r['sourceBrainCellsOnRay'],
                    r['pathLengthMm'],-min(r['nearWorkingTargetMm3'],r['farWorkingTargetMm3']),
                    -min(r['sourceSectorCapacityMm3'].values()),
                    r['surfaceFlatId'],r['targetFlatId']))
        else:
            preliminary[axis].sort(key=lambda r:(r['sourceBrainCellsOnRay'],
                    r['pathLengthMm'],-r['stagedLocalTargetContactMm3'],
                    -r['sourceTargetCellsOnRay'],r['surfaceFlatId'],r['targetFlatId']))
        for item in preliminary[axis][:_FINALISTS_PER_AXIS]:
            check()
            burden,corridor,corridor_centers = _geometry_burden(
                scene,grid,item,brain_ids,target_ids)
            report['counts']['exactFinalists'] += 1
            if not burden['farShaftSourceLabelsAllowed']:
                reject('UNMAPPED_FULL_SHAFT_CAPSULE'); continue
            if burden['corridorCenterContactBrainMm3'] > _SOURCE_CONTACT_POLICY_MM3:
                reject('SOURCE_CONTACT_POLICY_CUTOFF'); continue
            length=item['pathLengthMm']
            # This is a ranking heuristic, not a lower bound on the native
            # action count: aspiration can reach source tumor earlier.
            heuristic = 3+math.ceil((length+1.)/3.)+3+4
            if goal == 'two-sector' and heuristic>48:
                reject('HEURISTIC_ACTION_POLICY_CUTOFF'); continue
            item.update(burden)
            item.update(_fixed_far_first_contact(scene,grid,item,corridor,
                corridor_centers,brain_flat,target_flat))
            if goal == 'staged':
                item.update(_staged_work_tube_contact(scene,grid,item,corridor,
                    brain_flat,target_flat,staged_target_reference))
                if not item['stagedDeepShaftSourceLabelsAllowed']:
                    reject('UNMAPPED_STAGED_DEEP_SHAFT_CAPSULE'); continue
            item['heuristicActionCost']=heuristic
            item['heuristicActionHeadroom']=(48 if goal == 'two-sector' else 240)-heuristic
            # Three setup, three sector, and two cleanup actions are the
            # non-corridor minimum. The rank separately reserves four cleanup
            # actions for checked recovery; neither predicts native success.
            item['mathematicalMinimumNonCorridorActions']=8
            item['reservedNonCorridorActions']=10
            item['rankKey']=([item['farFirstJointSourceDeficitMm3'],
                              item['farFirstMaxSourceDeficitMm3'],
                              burden['corridorAndShaftBrainUnionMm3'],
                              burden['corridorBrainSupercoverMm3'],
                              length,-min(item['nearWorkingTargetMm3'],
                                          item['farWorkingTargetMm3']),
                              -min(item['farFirstSectorSourceContactMm3'].values()),
                              -item['heuristicActionHeadroom'],
                              axis,item['surfaceFlatId'],item['targetFlatId']]
                             if goal == 'two-sector' else
                             [item['stagedJointSourceDeficitMm3'],
                              burden['corridorAndShaftBrainUnionMm3'],length,
                              -item['stagedWorkTubeAdditionalTargetSupercoverMm3'],
                              -item['stagedLocalTargetContactMm3'],
                              -item['sourceTargetCellsOnRay'],
                              burden['corridorBrainSupercoverMm3'],
                              axis,item['surfaceFlatId'],item['targetFlatId']])
            finalists.append(item)
    finalists.sort(key=lambda r:tuple(r['rankKey']))
    routes=[]
    used_axes=set()
    for item in finalists:
        check()
        if item['axis'] in used_axes:
            reject('ONE_PER_NATIVE_AXIS'); continue
        route=_route(scene,item,regions,targets)
        novelty=_brain_route_novelty(np.asarray(item['surfaceRasMm']),
                                     np.asarray(item['inwardRasUnit']),routes)
        if novelty is None:
            reject('NOVELTY_12MM_AND_30DEG'); continue
        route['noveltyComparisons']=novelty
        _, evidence=_sector_cells(scene,route)
        exact_capacity={name:evidence[name]['sourceVolumeMm3']
                        for name in ('near','far')}
        if goal == 'two-sector' and any(value<MIN_SECTOR_MM3
                                        for value in exact_capacity.values()):
            reject('EXACT_SECTOR_CAPACITY_MISMATCH'); continue
        if any(abs(exact_capacity[name]-item['sourceSectorCapacityMm3'][name])>1e-9
               for name in exact_capacity):
            raise ValueError('Cheap source sector capacity disagrees with exact source sectors')
        routes.append(route)
        used_axes.add(item['axis'])
        report['selected'].append({'rank':len(routes),'routeId':route['id'],
            'nativeAxis':item['axis'],'sourceSurfaceIndex':item['surfaceIndex'],
            **({'sourceFaceSign': item['sourceFaceSign']} if 'sourceFaceSign' in item else {}),
            'sourceTargetBoundaryIndex':item['targetBoundaryIndex'],
            'sourceSectorCapacityMm3':exact_capacity,
            'sourceSectorCellSha256':{name:evidence[name]['sourceCellSha256']
                                     for name in ('near','far')},
            'nearWorkingTargetMm3':item['nearWorkingTargetMm3'],
            'farWorkingTargetMm3':item['farWorkingTargetMm3'],
            'nearWorkingBrainMm3':item['nearWorkingBrainMm3'],
            'farWorkingBrainMm3':item['farWorkingBrainMm3'],
            **({'stagedLocalTargetContactMm3':item['stagedLocalTargetContactMm3']}
               if goal == 'staged' else {}),
            **({name:item[name] for name in (
                'stagedWorkTubeOffsetFromTargetEntryMm',
                'stagedWorkTubeRadiusMm',
                'stagedDeepShaftSourceLabelsAllowed',
                'stagedDeepShaftTailDepthMm',
                'stagedWorkTubeSourceCellCount',
                'stagedWorkTubeBeyondCorridorCellCount',
                'stagedWorkTubeAdditionalTargetSupercoverMm3',
                'stagedWorkTubeAdditionalBrainSupercoverMm3',
                'stagedContinuationBrainExcessMm3',
                'stagedTargetContactDeficitMm3',
                'stagedJointSourceDeficitMm3')}
               if goal == 'staged' else {}),
            'corridorCenterContactBrainMm3':item['corridorCenterContactBrainMm3'],
            'corridorBrainSupercoverMm3':item['corridorBrainSupercoverMm3'],
            'farShaftBrainSupercoverMm3':item['farShaftBrainSupercoverMm3'],
            'corridorAndShaftBrainUnionMm3':item['corridorAndShaftBrainUnionMm3'],
            'farFirstAdditionalBrainSupercoverMm3':item['farFirstAdditionalBrainSupercoverMm3'],
            'farFirstAdditionalBrainCenterMm3':item['farFirstAdditionalBrainCenterMm3'],
            'farFirstAdditionalTargetSupercoverMm3':item['farFirstAdditionalTargetSupercoverMm3'],
            'farFirstSectorSourceContactMm3':item['farFirstSectorSourceContactMm3'],
            'farFirstNearSourceContactDeficitMm3':item['farFirstNearSourceContactDeficitMm3'],
            'farFirstFarSourceContactDeficitMm3':item['farFirstFarSourceContactDeficitMm3'],
            'farFirstBrainSourceContactExcessMm3':item['farFirstBrainSourceContactExcessMm3'],
            'farFirstJointSourceDeficitMm3':item['farFirstJointSourceDeficitMm3'],
            'farFirstMaxSourceDeficitMm3':item['farFirstMaxSourceDeficitMm3'],
            'farFirstPhaseBrainBeyondCorridorMm3':item['farFirstPhaseBrainBeyondCorridorMm3'],
            'farShaftTailDepthMm':item['farShaftTailDepthMm'],
            'heuristicActionCost':item['heuristicActionCost'],
            'heuristicActionHeadroom':item['heuristicActionHeadroom'],
            'mathematicalMinimumNonCorridorActions':item['mathematicalMinimumNonCorridorActions'],
            'reservedNonCorridorActions':item['reservedNonCorridorActions'],
            'pathLengthMm':item['pathLengthMm'],
            'noveltyComparisons':novelty,
            'rankKey':item['rankKey']})
        if len(routes)>=max_routes:
            break
    report['counts']['rankedFinalists']=len(finalists)
    report['counts']['selectedRoutes']=len(routes)
    report['status']='selected' if len(routes)==max_routes else 'insufficient_routes'
    report['limitations']=[
        'Source geometry only; no native place, corridor, sector, or cleanup action was attempted.',
        'Source brain center contact and capsule supercover are geometric proxies, not modeled removal or calibrated risk.',
        'Subtracting corridor source supercover can undercount or overcount actual additional native contact.',
        'Skull, vessel, functional and unannotated anatomy remain unknown unless supplied.',
        'The sample and exact-finalist caps can miss other feasible rays.',
        'A selected route is a candidate, not an executable access plan.']
    if goal == 'staged':
        report['limitations'].append(
            'Small fixed near/far sectors and their working spheres are recorded but do not gate staged selection.')
    report['selectionSeal']=semantic_digest(report)
    return routes,report
