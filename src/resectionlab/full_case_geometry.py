"""Read-only geometric candidates for the acquired full-case research rehearsal.

These proposals are not clinical routes. The public engine preview/step remains
responsible for every transition, guard, collision and resulting tissue effect.
Source labels and memberships are never edited here.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_SUPERIOR_Q = [1., 0., 0., 0.]


def _pose(position):
    return {'position': list(map(float, position)), 'quaternion': list(_SUPERIOR_Q)}


def _world(scene, ijk):
    return scene.affine[:3, :3] @ np.asarray(ijk, float) + scene.affine[:3, 3]


def _superior_grid(engine):
    scene = engine.scene
    # This acquired case is axis aligned with increasing index Z superior.
    if not np.allclose(scene.affine[:3, :3], np.diag(np.diag(scene.affine[:3, :3]))) or scene.affine[2, 2] <= 0:
        raise ValueError('Superior candidates require an axis-aligned superior-increasing grid')
    labels = scene.labels
    target_ids = [r['id'] for r in scene.regions if r.get('target')]
    bone_ids = [r['id'] for r in scene.regions if r['material'] == 'bone']
    target = np.isin(labels, target_ids)
    allowed = target | np.isin(labels, bone_ids) | (labels == 0)
    z = np.arange(scene.shape[2])
    top = np.max(np.where(target, z, -1), axis=2)
    bad_top = np.max(np.where(~allowed, z, -1), axis=2)
    window = (top > bad_top) & (top >= 0)
    clearance = ndi.distance_transform_edt(window, sampling=scene.spacing[:2])
    return target, top, bad_top, window, clearance


def access_candidates(engine, *, limit=160, max_radius_mm=None, separation_mm=3.5):
    """Return superior bone-only window centers, largest clearance first.

    ``entryPose`` is the first owned target center. ``outsidePose`` is above all
    source material. Drill access must stop before the target and be previewed.
    The radius is only a geometric proposal and respects the declared profile.
    """
    scene = engine.scene
    target, top, bad_top, window, clearance = _superior_grid(engine)
    ranking = clearance.copy()
    source_top = _world(scene, [0, 0, scene.shape[2] - 1])[2]
    outside_z = source_top + 8.
    max_radius = min(10. if max_radius_mm is None else float(max_radius_mm), float(scene.properties.get('tissuePhysicsV2', {}).get('bone', {}).get('maxRadiusMm', 10.)))
    if not .1 <= max_radius <= 10. or separation_mm <= 0:
        raise ValueError('Bounded drill radius and positive candidate separation required')
    dense = max_radius_mm is not None
    potential_counts = np.sum(target & (np.arange(scene.shape[2])[None,None,:] > bad_top[:,:,None]), axis=2)
    score = ndi.gaussian_filter(potential_counts.astype(float), 1.)
    if dense:
        # Broad exposure first uses the same maximum-clearance center. Following
        # narrow accesses greedily cover the largest retained target columns.
        score *= (clearance >= max_radius + 1.)
    records = []
    for number in range(limit):
        ij = np.unravel_index(np.argmax(ranking), ranking.shape)
        distance = float(clearance[ij])
        if float(ranking[ij]) <= 0. or distance < 2.: break
        i, j = map(int, ij)
        entry = _world(scene, [i, j, int(top[ij])])
        radius = max(.5, min(max_radius, float(clearance[ij]) - 1.5))
        outside = entry.copy(); outside[2] = outside_z
        labels = scene.labels[i, j]
        bones = np.flatnonzero(np.isin(labels, [r['id'] for r in scene.regions if r['material'] == 'bone']) & (np.arange(scene.shape[2]) > top[ij]))
        disk = ((np.arange(scene.shape[0])[:, None] - i) * scene.spacing[0]) ** 2 + ((np.arange(scene.shape[1])[None, :] - j) * scene.spacing[1]) ** 2 <= max(1., radius - 1.) ** 2
        potential = target & (np.arange(scene.shape[2])[None, None, :] > bad_top[:, :, None]) & disk[:, :, None]
        records.append({'id': f'superior-{i}-{j}', 'entryPose': _pose(entry), 'outsidePose': _pose(outside),
                        'drillRadiusMm': radius, 'strategy': 'dense-tapered-superior' if dense else 'broad-superior', 'windowClearanceMm': float(clearance[ij]),
                        'targetEntryIndex': [i, j, int(top[ij])], 'geometricTargetCapacityCells': int(potential.sum()),
                        'boneBoundsMm': None if not len(bones) else [_world(scene, [i,j,int(bones.min())])[2], _world(scene, [i,j,int(bones.max())])[2]],
                        'evidence': 'supplied labels only; attachments and complete functional anatomy UNKNOWN'})
        # Keep nearby but distinct openings without returning the same maximum.
        xx, yy = np.ogrid[:scene.shape[0], :scene.shape[1]]
        suppress = ((xx-i)*scene.spacing[0])**2 + ((yy-j)*scene.spacing[1])**2 < (float(separation_mm) if dense else max(5., radius))**2
        if dense:
            score[suppress] = 0.
            # Selection score is capacity, while the loop's stop and radius use
            # true window clearance. Keep zero scores out of the candidate set.
            ranking = np.where(score > 0., score, 0.)
        else:
            ranking[suppress] = 0
    return records


def next_removal_candidates(engine, access, *, slot='left', limit=8, radii=None):
    """Rank continuous superior CUSA strokes with target/free working cover.

    Proposed paths first retreat vertically into free space, translate above the
    source grid, then descend. Radius applies to the entire proposed action and
    is checked against its full working cover. Shaft cover may contain target
    only when that target is in the preceding aspirator sweep. Exact engine
    preview remains mandatory, including intermediate collision inflation.

    Acquired sub-06 passage checked through the unchanged public engine: a
    0.6 mm aperture on a quarter-voxel shifted ray with fine entry/end waypoints
    added 206 owned target cells (310 -> 516), with matching preview/commit and
    zero brain/vessel removal. This is a generic tool aperture assumption,
    not a calibrated energy model or target-mask exemption. Local evidence:
    testcases/outputs/full-case-rehearsal-20261020/geometry/
    first-passage-preview-commit-proof.json.
    """
    state = engine.state['slots'][slot]
    if state['instrument'] != 'cusa': return []
    if not np.allclose(np.abs(state['quaternion']), np.abs(_SUPERIOR_Q), atol=1e-6): return []
    scene = engine.scene
    target, top, bad_top, window, clearance = _superior_grid(engine)
    owner = engine.occupancy()
    target_labels = {r['id'] for r in scene.regions if r.get('target')}
    source_labels = scene.labels.ravel()
    current = np.asarray(state['position'], float)
    params = engine.parameters(state)
    if radii is None:
        radii = sorted(set([float(params['radiusMm']), .3, .45, .6, 1., 1.3, 1.5, 2.5, 4., 6.]), reverse=True)
    outside_z = float(access['outsidePose']['position'][2])
    center = np.asarray(access['entryPose']['position'], float)
    max_offset = max(2., float(access.get('searchRadiusMm', float(access['drillRadiusMm']) + 2.)))
    fine_step = min(.05, max(.02, float(access.get('frontierStepMm', .05))))
    frontier_inflation = fine_step / 2. + 1e-4
    spacing = float(min(scene.spacing[:2]))
    # Reuse a source lattice, sorting all remaining target columns by proximity
    # to the selected opening. Existing cavities can connect adjacent columns.
    columns = np.argwhere(window)
    records = []
    removed = set(engine.state['removed'])
    column_specs = []
    deformation = engine.state.get('deformation', {})
    for i, j in columns:
        xy = _world(scene, [i, j, 0])[:2]
        distance = float(np.linalg.norm(xy-center[:2]))
        if distance > max_offset + .5: continue
        remaining = [int(k) for k in np.flatnonzero(target[i,j]) if k > bad_top[i,j]
                     and np.ravel_multi_index((i,j,k), scene.shape) not in removed]
        if not remaining: continue
        if not deformation and not engine.state.get('holds'):
            roof = [int(k) for k in np.flatnonzero(scene.labels[i,j]) if k > max(remaining)
                    and np.ravel_multi_index((i,j,k), scene.shape) not in removed]
            if roof: continue
        column_specs.append((int(i), int(j), min(remaining), len(remaining), distance))
    # Integer rays are cheap baseline proposals. Spend subvoxel fallback work
    # only on six promising exposed columns, with narrow working apertures.
    proposals = [(i,j,bottom,0.,0.) for i,j,bottom,_,_ in column_specs]
    offsets = [(-.25,0.),(.25,0.),(0.,-.25),(0.,.25),
               (-.25,-.25),(-.25,.25),(.25,-.25),(.25,.25),
               (-.4,0.),(.4,0.),(0.,-.4),(0.,.4)]
    for i,j,bottom,_,_ in sorted(column_specs, key=lambda r:(-r[3],r[4]))[:6]:
        proposals.extend((i,j,bottom,di,dj) for di,dj in offsets)
    for i, j, bottom, offset_i, offset_j in proposals:
        xy = _world(scene, [i+offset_i, j+offset_j, 0])[:2]
        if np.linalg.norm(xy-center[:2]) > max_offset: continue
        for radius in radii:
            if not .1 <= radius <= 12.: continue
            if (offset_i or offset_j) and radius > 1.3: continue
            end = _world(scene, [i+offset_i,j+offset_j,bottom]); end[2] += radius - .3
            if end[2] >= outside_z: continue
            def covers(endpoint):
                lifted = current.copy(); lifted[2] = outside_z
                above = endpoint.copy(); above[2] = outside_z
                poses = [lifted, above, endpoint]
                working = set(); shafts = set(); effect = set(); previous = current
                depth = min(float(params.get('depthMm', 2.)), 3.)
                for point in poses:
                    if np.linalg.norm(point-previous) < 1e-8:
                        previous=point; continue
                    effect.update(owner.material_ids(engine.cells(previous,point,radius)))
                    # Vertical capsules are exact unions at fixed orientation;
                    # the lateral segment is above the complete source grid.
                    if np.linalg.norm(point[:2]-previous[:2]) < 1e-8:
                        lo = min(point[2], previous[2]); hi = max(point[2], previous[2])
                        a=point.copy(); b=point.copy(); a[2]=lo; b[2]=hi+depth
                        working.update(owner.material_ids(engine.cells(a,b,radius+frontier_inflation)))
                        a[2]=lo+max(1.,min(float(params.get('depthMm',2.)),5.)); b[2]=hi+40.
                        shafts.update(owner.material_ids(engine.cells(a,b,.5+frontier_inflation)))
                    else:
                        working.update(owner.material_ids(engine.cells(previous,point,radius+.17)))
                    previous=point
                return poses, working, shafts, effect
            poses, working, shafts, effect = covers(end)
            blocked = [c for c in working if int(source_labels[c]) not in target_labels]
            if blocked:
                # A neighboring brain/bone/vessel boundary can be higher than
                # this column's bottom. Stop above that boundary and retain a
                # shallower legal frontier stroke rather than dropping it.
                boundary = scene.points(blocked)
                # Exact vertical capsule-to-axis-aligned-cell cap, rather than
                # adding a full radius vertically to lateral rim obstacles.
                # The old bound could stop 0.2--1 mm above an exposed target
                # and discard every initial frontier nibble in a narrow bore.
                xy_gap = np.maximum(np.abs(boundary[:, :2] - end[:2]) - scene.spacing[:2] / 2., 0.)
                horizontal2 = np.sum(xy_gap ** 2, axis=1)
                cap_height = np.sqrt(np.maximum((radius + frontier_inflation) ** 2 - horizontal2, 0.))
                clipped_z = float(np.max(boundary[:, 2] + scene.spacing[2] / 2. + cap_height)) + 1e-4
                if clipped_z <= end[2] or clipped_z >= outside_z: continue
                end[2] = clipped_z
                poses, working, shafts, effect = covers(end)
            if not working or any(int(source_labels[c]) not in target_labels for c in working): continue
            if shafts-effect: continue
            # Removal cells are source identities, never assigned by this helper.
            predicted = len(effect-removed)
            if not predicted: continue
            # Preserve the existing engine collision law. Explicit close-contact
            # waypoints reduce actual per-segment sweep inflation at the final
            # frontier; public preview still admits every physical microstep.
            source_top = int(np.flatnonzero(target[i,j]).max())
            entry_z = float(_world(scene, [i+offset_i,j+offset_j,source_top])[2])
            entry_approach = end.copy()
            entry_approach[2] = min(outside_z, max(end[2]+3., entry_z+radius+3.))
            fine_poses = [poses[0], poses[1], entry_approach]
            entry_exit_z = max(end[2], entry_z-4.)
            def append_fine(end_z):
                start_point = fine_poses[-1]
                finish = end.copy(); finish[2] = end_z
                count = max(1, int(np.ceil(abs(start_point[2]-end_z)/fine_step)))
                for fraction in np.linspace(0.,1.,count+1)[1:]:
                    fine_poses.append(start_point*(1.-fraction)+finish*fraction)
            # The narrow bony aperture is near the first target surface even
            # when this ray continues deep into a compatible target core.
            append_fine(entry_exit_z)
            if entry_exit_z > end[2]:
                approach = end.copy(); approach[2] = min(entry_exit_z,end[2]+3.)
                if entry_exit_z > approach[2]: fine_poses.append(approach)
                append_fine(end[2])
            if len(fine_poses) > 512: continue
            action={'slot':slot,'operation':'apply','path':[_pose(p) for p in fine_poses],
                    'parameters':{'radiusMm':float(radius)},'duration':max(.1,min(30.,float(sum(np.linalg.norm(b-a) for a,b in zip([current,*poses[:-1]],poses)))/20.))}
            records.append((predicted, -float(np.linalg.norm(xy-center[:2])), action))
    records.sort(key=lambda r:(-r[0],-r[1]))
    return [r[2] for r in records[:limit]]

