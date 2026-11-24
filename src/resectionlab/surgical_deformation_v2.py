"""Local, deterministic Kelvin–Voigt lattice surrogate (mm, N, seconds).

These explicit engineering assumptions are NOT patient measurements or clinical
validation. The small-strain vector Laplacian couples neighboring source cells;
a squared discrete divergence penalizes volume change approximately. This is
not incompressible FEM. Opt-in patient posture supplies disclosed body forces;
fluid pressure, buoyancy and fracture remain unsupported.
Cuts delete lattice bonds, while source removal permanently deletes nodes.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from . import surgical_mechanical_exposure as mechanical


_diagnostics = ContextVar('surgical_deformation_diagnostics', default=None)
CONTACT_ACTIVATION_FORCE_FLOOR_N = 1e-10
CONTACT_OMITTED_FORCE_LIMIT_N = 1e-8
CONTACT_RECEPTOR_RESERVE_NODES = 512


@contextmanager
def diagnostic_scope():
    """Observe attempted solves without changing canonical state or recordings.

    Includes preview/rolled-back work if the caller executes it in this scope.
    Aggregate storage is constant size. No diagnostic is a tissue injury limit.
    """
    report = dict(scope='attempted numerical solves, including rolled-back trials',
        ticks=0, substeps=0, targetCapNodeTicks=0, forceCapNodeTicks=0,
        displacementCapNodeSteps=0, strainProjectionSteps=0, collisionRejectedSteps=0,
        maxPreStrainProjectionNeighborRatio=0., maxFreeResidualN=0.,
        maxAbsDiscreteEnergyBalanceNmm=0., maxOmittedWallReactionN=0., omittedWallReceptorTicks=0)
    token = _diagnostics.set(report)
    try:
        yield report
    finally:
        _diagnostics.reset(token)


def profile():
    return dict(radiusMm=4., maxNodes=2000, fixedStepSeconds=1 / 120,
                brainStiffnessNPerMm=.0006, tumorStiffnessNPerMm=.0012,
                dragNsPerMm=.0003, holdStiffnessNPerMm=.02,
                bulkPenaltyRatio=2., maxDisplacementMm=3., maxEdgeStrain=.65,
                maxForceN=.08,
                model='local small-strain Kelvin-Voigt lattice; uncalibrated engineering assumptions',
                units='displacement mm; stiffness N/mm; drag N s/mm; time s')


def initialize(engine):
    engine.state.setdefault('deformation', {})
    engine.state.setdefault('deformationModel', {'active': [], 'anchors': [], 'remainderSeconds': 0., 'velocities': {}})


def _settings(engine):
    p = profile()
    declared = engine.scene.properties.get('tissuePhysicsV2', {}).get('brain')
    if declared is not None:
        if not isinstance(declared, dict) or set(declared) != set(p):
            raise ValueError('INVALID_DEFORMATION_PROFILE: expected complete brain profile')
        p.update(declared)
    for name in ('radiusMm', 'fixedStepSeconds', 'brainStiffnessNPerMm', 'tumorStiffnessNPerMm',
                 'dragNsPerMm', 'holdStiffnessNPerMm', 'maxDisplacementMm', 'maxEdgeStrain', 'maxForceN'):
        if type(p[name]) not in (int, float) or not np.isfinite(p[name]) or p[name] <= 0: raise ValueError('INVALID_DEFORMATION_PROFILE:' + name)
    if type(p['maxNodes']) is not int or not 1 <= p['maxNodes'] <= 2000 or type(p['bulkPenaltyRatio']) not in (int, float) or not 0 <= p['bulkPenaltyRatio'] <= 100:
        raise ValueError('INVALID_DEFORMATION_PROFILE')
    if not .001 <= p['fixedStepSeconds'] <= .1 or not .1 <= p['radiusMm'] <= 12:
        raise ValueError('DEFORMATION_PROFILE_OUTSIDE_NUMERICAL_DOMAIN')
    return p


def _holds(engine):
    return [h for h in engine.state['holds'].values() if 'target' in h]


def rigid_cells(engine):
    if engine.state.get('toolMechanicsRevision') not in (1, 2): return set()
    return {c for h in [*engine.state['holds'].values(), *engine.state.get('detachedFragments', [])]
            if 'target' not in h for c in h['cells']}


def _material(engine, cell):
    return engine.by_label.get(int(engine.scene.labels.flat[cell]), {}).get('material')


def _blocked(engine):
    return {tuple(sorted(map(int, e))) for name in ('cutEdges', 'transectedEdges')
            for e in engine.state.get(name, []) if isinstance(e, (list, tuple)) and len(e) == 2}


def _neighbor_ids(shape, cell):
    cell = int(cell)
    wide = shape[1] * shape[2]
    i, tail = divmod(cell, wide)
    j, k = divmod(tail, shape[2])
    for axis, (coordinate, extent, stride) in enumerate(zip((i, j, k), shape, (wide, shape[2], 1))):
        if coordinate > 0:
            yield cell - stride, axis
        if coordinate + 1 < extent:
            yield cell + stride, axis


def _neighbors(engine, cell):
    for neighbor, axis in _neighbor_ids(engine.scene.shape, cell):
        yield neighbor, float(np.linalg.norm(engine.scene.affine[:3, axis]))


def begin_hold(engine, key, cells, position):
    initialize(engine)
    p = _settings(engine); removed = set(engine.state['removed'])
    cells = sorted(set(map(int, cells)) - removed)
    rigid = rigid_cells(engine)
    if set(cells).intersection(rigid): raise ValueError('MATERIAL_ALREADY_HELD')
    removed |= rigid
    cells = [c for c in cells if _material(engine, c) in ('soft_tissue', 'brain', 'tumor')]
    if not cells: return
    other = {c for k, hold in engine.state['holds'].items() if k != key for c in hold['cells']}
    if other.intersection(cells): raise ValueError('MATERIAL_ALREADY_HELD')
    position = _position(position)
    model = engine.state['deformationModel']; active = set(model['active']) - removed
    blocked = _blocked(engine)
    distance = {c: 0. for c in cells}; queue = deque(cells)
    while queue:
        c = queue.popleft()
        if c not in active and len(active) >= p['maxNodes']:
            if 'brainContinuumV1' in engine.scene.properties:
                raise ValueError('CONTINUUM_PATCH_TRUNCATED: source-cell budget cannot cover the declared physical radius')
            continue
        active.add(c)
        for n, length in _neighbors(engine, c):
            d = distance[c] + length
            if d > p['radiusMm'] or n in removed or tuple(sorted((c, n))) in blocked: continue
            if _material(engine, n) not in ('soft_tissue', 'brain', 'tumor'): continue
            if d < distance.get(n, float('inf')):
                distance[n] = d; queue.append(n)
    if not set(cells) <= active: raise ValueError('DEFORMATION_NODE_LIMIT')
    # Only the outer patch interface anchors: missing anatomy/cut neighbors do not.
    anchors = {c for c in active if any(n not in active and n not in removed and
               _material(engine, n) in ('soft_tissue', 'brain', 'tumor') and
               tuple(sorted((c, n))) not in blocked for n, _ in _neighbors(engine, c))}
    held_all = other | set(cells)
    model['active'] = sorted(active); model['anchors'] = sorted(anchors - held_all)
    engine.state['holds'][key] = dict(cells=cells, origin=position.tolist(), target=position.tolist(),
        offset=[0, 0, 0], baseDisplacements={str(c): engine.state['deformation'].get(str(c), [0., 0., 0.])[:] for c in cells})


def _position(position):
    p = np.asarray(position, float)
    if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('INVALID_DEFORMATION_TARGET')
    return p


def set_target(engine, key, position):
    engine.state['holds'][key]['target'] = _position(position).tolist()


def release(engine, key):
    engine.state['holds'].pop(key, None)


def _system(engine, p, ids, anchors, retained, blocked):
    held = tuple(sorted({c for h in _holds(engine) for c in h['cells']} & set(ids)))
    shape = engine.scene.shape
    strides = {1, shape[2], shape[1] * shape[2]}
    # Only cuts between potential axial neighbors can alter this system. At
    # most three forward pairs per active node enter the cache, even with a
    # large cut ledger elsewhere. _neighbors handles the actual grid bounds.
    local_cuts = frozenset((a, b) for a, b in blocked
                           if a in retained and b in retained and b - a in strides)
    grid = (tuple(shape), engine.scene.affine[:3, :3].tobytes())
    material_profile = engine.scene.properties.get('gliomaMaterialsV1')
    signature = (grid, engine.scene.identity if material_profile else None,
                 tuple(ids), tuple(_material(engine, c) for c in ids), tuple(anchors),
                 local_cuts, held, tuple((k, p[k]) for k in sorted(profile())))
    cached = getattr(engine, '_deformation_system_v2', None)
    if cached and cached[0] == signature: return cached[1:]
    bonds = sorted((c, n) for c in ids for n, _ in _neighbor_ids(shape, c)
                   if c < n and n in retained and (c, n) not in local_cuts)
    n = len(ids); index = {c: i for i, c in enumerate(ids)}
    stiffness = np.array([p['tumorStiffnessNPerMm'] if _material(engine, c) == 'tumor' else p['brainStiffnessNPerMm'] for c in ids])
    if material_profile:
        from .surgical_glioma_materials import material_values
        # Pa -> N/mm² times representative face area / bond length (mm).
        stiffness = 1e-6 * engine.scene.voxel_mm3 ** (1/3) * material_values(engine, ids, 'equilibriumShearPa')
    rows = []; cols = []; vals = []; dr = []; dc = []; dv = []
    rest = np.column_stack(np.unravel_index(ids, engine.scene.shape)) @ engine.scene.affine[:3, :3].T
    edges = []
    for a, b in bonds:
        i, j = index[a], index[b]; vector = rest[j] - rest[i]; length = np.linalg.norm(vector); direction = vector / length
        k = 2 * stiffness[i] * stiffness[j] / (stiffness[i] + stiffness[j])
        edges.append((i, j, length))
        for axis in range(3):
            ai, bj = 3 * i + axis, 3 * j + axis
            rows.extend((ai, ai, bj, bj)); cols.extend((ai, bj, ai, bj)); vals.extend((k, -k, -k, k))
            # Each endpoint estimates divergence from incident oriented differences.
            for node in (i, j):
                dr.extend((node, node)); dc.extend((ai, bj)); dv.extend((-direction[axis] / (2 * length), direction[axis] / (2 * length)))
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(3*n, 3*n)).tocsr()
    D = sparse.coo_matrix((dv, (dr, dc)), shape=(n, 3*n)).tocsr()
    cell_length_squared = abs(np.linalg.det(engine.scene.affine[:3, :3])) ** (2 / 3)
    K += D.T @ sparse.diags(stiffness * p['bulkPenaltyRatio'] * cell_length_squared) @ D
    # Weak local foundation represents unresolved surrounding tissue and removes
    # rigid modes in detached fragments; it is a disclosed surrogate assumption.
    K += sparse.diags(np.repeat(stiffness * .05, 3))
    hold_diag = np.repeat([p['holdStiffnessNPerMm'] if c in held else 0 for c in ids], 3)
    free = np.repeat([c not in anchors for c in ids], 3)
    effective_K = K
    if material_profile:
        tau = material_profile['relaxationSeconds']
        effective_K = K * (1 + material_profile['maxwellShearRatio'] * tau / (tau + p['fixedStepSeconds']))
    A = effective_K + sparse.diags(hold_diag + p['dragNsPerMm'] / p['fixedStepSeconds'])
    solve = factorized(A[free][:, free].tocsc()) if free.any() else lambda rhs: np.empty(0)
    cached = (signature, solve, free, index, edges, K)
    engine._deformation_system_v2 = cached
    return cached[1:]


def _obstacle_collision(engine, vox, inside, removed, blocking_labels):
    """Same whole-increment obstacle decision, using only sampled grid cells."""
    if not np.all(inside):
        return True
    cells = np.ravel_multi_index(vox.T, engine.scene.shape)
    blocked = np.isin(engine.scene.labels.ravel()[cells], blocking_labels)
    return bool(np.any(blocked & ~np.isin(cells, removed)))


def _bounded_contact_reactions(reactions, active):
    """Keep existing receivers and bound the sum of omitted new loads."""
    new_cells = np.array([cell for cell in reactions if cell not in active], dtype=np.intp)
    if not len(new_cells):
        return reactions, 0.
    magnitudes = np.linalg.norm([reactions[cell] for cell in new_cells], axis=1)
    weak = magnitudes < CONTACT_ACTIVATION_FORCE_FLOOR_N
    cells, loads = new_cells[weak], magnitudes[weak]
    total = float(np.sum(loads))
    if total > CONTACT_OMITTED_FORCE_LIMIT_N:
        # Keep larger loads when the aggregate omission limit is reached.
        # Source-cell order breaks ties independently of reaction assembly.
        order = np.lexsort((cells, loads))
        running = np.cumsum(loads[order])
        count = np.searchsorted(running, CONTACT_OMITTED_FORCE_LIMIT_N, side='right')
        cells = cells[order[:count]]
        total = float(running[count - 1]) if count else 0.
    if not len(cells):
        return reactions, 0.
    omitted = set(cells)
    return {cell: force for cell, force in reactions.items() if cell not in omitted}, total


def _activate_contact_patch(engine, model, cells, excluded, max_nodes, *, reserve_nodes=0):
    active = set(model['active']) - excluded
    anchors = set(model['anchors'])
    free = active - anchors
    seeds = set(cells) - excluded
    if seeds <= free:
        return
    active.update(seeds)
    if len(active) > max_nodes:
        raise ValueError('VESSEL_TISSUE_NODE_LIMIT: insufficient local lattice nodes for contact')
    blocked = _blocked(engine)
    cache = {}

    def neighbors(cell):
        if cell not in cache:
            cache[cell] = {n for n, _ in _neighbor_ids(engine.scene.shape, cell) if n not in excluded
                           and tuple(sorted((cell, n))) not in blocked
                           and _material(engine, n) in ('soft_tissue', 'brain', 'tumor')}
        return cache[cell]

    # Bind the loaded cells to at most two local tissue shells. Existing holds
    # keep their active cells and the patch boundary anchors remain clamped.
    # Optional neighbors only use the remaining expansion budget. Receptors
    # already added above can use the full hard node limit.
    neighborhood_limit = max(len(active), max_nodes - reserve_nodes)
    frontier = seeds
    for _ in range(2):
        new = set().union(*(neighbors(cell) for cell in frontier)) - active
        frontier = set(sorted(new)[:neighborhood_limit - len(active)])
        active.update(frontier)
    held = {c for hold in _holds(engine) for c in hold['cells']}
    model['active'] = sorted(active)
    model['anchors'] = sorted(cell for cell in active - held - seeds - free if neighbors(cell) - active)


def tick(engine, dt, *, vessel_tool_motion=None):
    initialize(engine)
    if not np.isfinite(dt) or dt < 0: raise ValueError('INVALID_DEFORMATION_TIME')
    p = _settings(engine); model = engine.state['deformationModel']; removed = set(engine.state['removed'])
    diagnostic = _diagnostics.get()
    if diagnostic is not None: diagnostic['ticks'] += 1
    if 'brainContinuumV1' in engine.scene.properties:
        from .surgical_continuum import tick as continuum_tick
        return continuum_tick(engine, dt, diagnostic)
    if 'patientPosition' in engine.state:
        from .surgical_position import activate_exposed_patch
        activate_exposed_patch(engine, radius_mm=p['radiusMm'], max_nodes=p['maxNodes'])
    rigid = rigid_cells(engine)
    for h in engine.state['holds'].values(): h['cells'] = [c for c in h['cells'] if c not in removed]
    material_profile = engine.scene.properties.get('gliomaMaterialsV1')
    vessel_displacements = {}
    distributed = False
    coupled_vessels = False
    wall_reactions = {}
    if material_profile:
        from .surgical_glioma_materials import move_vessels, retained_vessel_segments, vessel_displacements as vessel_motion
        from . import surgical_vessel_rods as rods
        distributed = rods.enabled(engine)
        if distributed:
            coupled_vessels = rods.soft_contact_enabled(engine)
            if coupled_vessels:
                # Advance rods against the current lattice and hold their mean
                # accepted contact reaction over this lattice increment.
                wall_reactions = rods.tick(engine, dt, tool_motion=vessel_tool_motion, excluded_tissue=rigid) or {}
                if wall_reactions:
                    excluded = removed | rigid
                    reserve = 0
                    if rods.bounded_tissue_capacity_enabled(engine):
                        active = set(model['active']) - excluded
                        supplied = len(wall_reactions)
                        wall_reactions, omitted = _bounded_contact_reactions(wall_reactions, active)
                        reserve = min(CONTACT_RECEPTOR_RESERVE_NODES, p['maxNodes'] // 3)
                        if diagnostic is not None:
                            diagnostic['maxOmittedWallReactionN'] = max(diagnostic['maxOmittedWallReactionN'], omitted)
                            diagnostic['omittedWallReceptorTicks'] += supplied - len(wall_reactions)
                    if wall_reactions:
                        _activate_contact_patch(engine, model, wall_reactions, excluded, p['maxNodes'], reserve_nodes=reserve)
            vessel_displacements = {str(cell): value for cell, value in vessel_motion(engine).items()}
        elif retained_vessel_segments(engine):
            deformation = engine.state['deformation']
            vessel_displacements = {str(vessel): deformation[str(vessel)]
                                    for vessel, _ in material_profile['vesselAttachments']
                                    if vessel not in removed and str(vessel) in deformation}
    ids = [c for c in model['active'] if c not in removed and c not in rigid]
    model['active'] = ids
    retained = set(ids)
    for c in list(engine.state['deformation']):
        if int(c) not in retained and c not in vessel_displacements:
            engine.state['deformation'].pop(c)
    for c in list(model['velocities']):
        if int(c) not in retained:
            model['velocities'].pop(c)
    if material_profile:
        memory_field = model.setdefault('maxwellMemoryMm', {})
        for c in list(memory_field):
            if int(c) not in retained:
                memory_field.pop(c)
        if not ids and 'activeReferenceVolumeMm3' in model:
            model.update(activeNodes=0, elasticEnergyNmm=0., summedHandleLoadN=0.,
                         maxNeighborDisplacementRatio=0., maxGradientNorm=0.,
                         activeReferenceVolumeMm3=0., activeDeformedVolumeMm3=0.,
                         minVolumeRatio=1., maxVolumeRatio=1.)
    if not ids:
        if distributed and not coupled_vessels:
            rods.tick(engine, dt, tool_motion=vessel_tool_motion)
        if coupled_vessels:
            model['remainderSeconds'] = engine.state[rods.FIELD]['remainderSeconds']
        return
    blocked = _blocked(engine)
    anchors = sorted(set(model['anchors']) & retained)
    solve, free, index, edges, K = _system(engine, p, ids, anchors, retained, blocked)
    h = p['fixedStepSeconds']; total = model['remainderSeconds'] + dt
    steps = int(np.floor((total + 1e-12) / h))
    model['remainderSeconds'] = max(0., round(total - steps*h, 14))
    u = np.array([engine.state['deformation'].get(str(c), [0., 0., 0.]) for c in ids])
    if material_profile:
        from .surgical_glioma_materials import gradient_stencil
        stencil = gradient_stencil(engine, ids, edges)
        memory = np.array([memory_field.get(str(c), [0., 0., 0.]) for c in ids])
        tau = material_profile['relaxationSeconds']
        memory_weight = tau / (tau + h)
        branch_ratio = material_profile['maxwellShearRatio']
    targets = np.zeros_like(u)
    for hold in _holds(engine):
        delta = np.subtract(hold.get('target', hold['origin']), hold['origin'])
        for c in hold['cells']:
            if c in index: targets[index[c]] = delta + hold.get('baseDisplacements', {}).get(str(c), [0., 0., 0.])
    lengths = np.linalg.norm(targets, axis=1)
    if diagnostic is not None:
        diagnostic['targetCapNodeTicks'] += int(np.count_nonzero(lengths > p['maxDisplacementMm']))
    targets *= np.minimum(1., p['maxDisplacementMm'] / np.maximum(lengths, 1e-15))[:, None]
    held = {c for hold in _holds(engine) for c in hold['cells']}
    forcing = targets * np.array([p['holdStiffnessNPerMm'] if c in held else 0 for c in ids])[:, None]
    if diagnostic is not None:
        diagnostic_hold_diagonal = np.repeat([p['holdStiffnessNPerMm'] if c in held else 0 for c in ids], 3)
    if 'patientPosition' in engine.state:
        from .surgical_position import gravity_forces_n
        forcing += gravity_forces_n(engine, ids)
    for cell, force in wall_reactions.items():
        forcing[index[cell]] += force
    if diagnostic is not None:
        diagnostic['forceCapNodeTicks'] += int(np.count_nonzero(np.linalg.norm(forcing, axis=1) > p['maxForceN']))
    force_scale = np.minimum(1., p['maxForceN'] / np.maximum(np.linalg.norm(forcing, axis=1), 1e-15))
    forcing *= force_scale[:, None]
    track_mechanics = steps and mechanical.enabled(engine)
    if track_mechanics or coupled_vessels:
        edge_nodes = np.asarray([(i, j) for i, j, _ in edges], dtype=int).reshape(-1, 2)
        edge_lengths = np.asarray([length for _, _, length in edges])
    if track_mechanics:
        labels = engine.scene.labels.ravel()[ids]
        retractor_cells = sorted({c for key, hold in engine.state['holds'].items()
                                  if 'target' in hold and engine.state['slots'][key]['instrument'] == 'brain_retractor'
                                  for c in hold['cells'] if c in index})
        retractor_nodes = np.asarray([index[cell] for cell in retractor_cells], dtype=int)
        effective_targets = targets[retractor_nodes] * force_scale[retractor_nodes, None]
    rest = np.column_stack(np.unravel_index(ids, engine.scene.shape)) @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
    removed_array = np.fromiter(removed, dtype=np.int64)
    blocking_labels = np.array([label for label, region in engine.by_label.items()
        if region.get('material') in ('bone', 'vessel', 'unknown')], dtype=engine.scene.labels.dtype)
    # For every bond the displacement difference is at most 2*sqrt(3)
    # times the largest absolute node component. Staying below one eighth
    # of the shortest bond's allowed extension leaves the projection idle,
    # including ample margin for roundoff. Keep the exact scalar test near
    # the limit, outside ordinary magnitudes, or when collecting diagnostics.
    strain_limit = p['maxEdgeStrain'] * min((length for _, _, length in edges), default=0.)
    safe_component = strain_limit / 8 if 2e-15 < strain_limit < 1e150 and diagnostic is None else 0.
    inverse = engine.scene.inverse[:3, :3]
    # Each index-space gradient entry is bounded by twice the maximum
    # displacement component. The smaller determinant bound is sufficient
    # both for the deformation gradient cap and for its volume bounds.
    gradient_safe = (1.18 ** (1 / 3) - 1) / (7. * np.linalg.norm(inverse)) if coupled_vessels else 0.
    same_voxel_safe = .49 / np.abs(inverse).sum(axis=1).max() if coupled_vessels else 0.
    velocity = np.zeros_like(u)
    for _ in range(steps):
        old = u.copy(); rhs = (p['dragNsPerMm'] / h * u + forcing).ravel()
        if material_profile:
            rhs += (branch_ratio * memory_weight) * (K @ memory.ravel())
        result = np.zeros(u.size); result[free] = solve(rhs[free]); u = result.reshape((-1, 3))
        if diagnostic is not None:
            diagnostic['substeps'] += 1
            diagnostic['displacementCapNodeSteps'] += int(np.count_nonzero(np.linalg.norm(u, axis=1) > p['maxDisplacementMm']))
        u *= np.minimum(1., p['maxDisplacementMm'] / np.maximum(np.linalg.norm(u, axis=1), 1e-15))[:, None]
        # Global rescaling preserves continuity while bounding every bond strain.
        scale = 1.
        if not (safe_component and np.max(np.abs(u)) < safe_component):
            if coupled_vessels:
                stretch = np.linalg.norm(u[edge_nodes[:, 1]] - u[edge_nodes[:, 0]], axis=1)
                scale = np.min(p['maxEdgeStrain'] * edge_lengths / np.maximum(stretch, 1e-15), initial=1.)
            else:
                scale = min((p['maxEdgeStrain'] * length / max(np.linalg.norm(u[j] - u[i]), 1e-15) for i, j, length in edges), default=1.)
        if diagnostic is not None:
            diagnostic['strainProjectionSteps'] += int(scale < 1)
            ratio = max((float(np.linalg.norm(u[j] - u[i]) / length) for i, j, length in edges), default=0.)
            diagnostic['maxPreStrainProjectionNeighborRatio'] = max(diagnostic['maxPreStrainProjectionNeighborRatio'], ratio)
        if scale < 1: u *= scale
        if material_profile and not (gradient_safe and np.max(np.abs(u)) < gradient_safe):
            from .surgical_glioma_materials import limit_gradient
            u = limit_gradient(stencil, u)
        # Conservative swept point-in-source-cell obstacle constraint; collision
        # integration additionally accounts for the displaced voxel's extent.
        collision = False
        if not (same_voxel_safe and max(np.max(np.abs(old)), np.max(np.abs(u))) < same_voxel_safe):
            for alpha in (.25, .5, .75, 1.):
                world = rest + old + alpha * (u - old)
                vox = np.rint(world @ inverse.T + engine.scene.inverse[:3, 3]).astype(int)
                inside = np.all((vox >= 0) & (vox < np.array(engine.scene.shape)), axis=1)
                collision |= _obstacle_collision(engine, vox, inside, removed_array, blocking_labels)
        # Reject the coupled increment as a whole: clamping single nodes would
        # introduce fresh strain discontinuities after the strain projection.
        if collision: u = old
        if material_profile:
            memory = memory_weight * memory + (1 - memory_weight) * u
        if track_mechanics:
            mechanical.record_lattice_step(engine, labels, edge_nodes, edge_lengths, u, h)
            if retractor_cells:
                forces = p['holdStiffnessNPerMm'] * (effective_targets - u[retractor_nodes])
                mechanical.record_bonded_step(engine, retractor_cells, forces, h)
        if diagnostic is not None:
            diagnostic['collisionRejectedSteps'] += int(collision)
            delta = (u - old).ravel()
            ku = K @ u.ravel() + diagnostic_hold_diagonal * u.ravel()
            kold = K @ old.ravel() + diagnostic_hold_diagonal * old.ravel()
            kdelta = K @ delta + diagnostic_hold_diagonal * delta
            residual = ku + p['dragNsPerMm'] / h * delta - forcing.ravel()
            if material_profile:
                residual += branch_ratio * (K @ (u - memory).ravel())
            residual = residual[free]
            diagnostic['maxFreeResidualN'] = max(diagnostic['maxFreeResidualN'], float(np.max(np.abs(residual), initial=0.)))
            # Constant-target implicit Euler balance includes the hold-spring
            # quadratic potential, viscous and algorithmic dissipation. This
            # is solver verification, not measured retractor work.
            change = .5 * (u.ravel() @ ku - old.ravel() @ kold)
            balance = forcing.ravel() @ delta - change - p['dragNsPerMm'] / h * (delta @ delta) - .5 * delta @ kdelta
            if material_profile:
                balance -= branch_ratio * ((u - memory).ravel() @ (K @ delta))
            diagnostic['maxAbsDiscreteEnergyBalanceNmm'] = max(diagnostic['maxAbsDiscreteEnergyBalanceNmm'], abs(float(balance)))
        velocity = (u - old) / h
    if steps:
        engine.state['deformation'] = {**vessel_displacements, **{str(c): u[i].tolist() for i, c in enumerate(ids)}}
        model['velocities'] = {str(c): velocity[i].tolist() for i, c in enumerate(ids)}
        model['elasticEnergyNmm'] = float(.5 * u.ravel() @ (K @ u.ravel()))
        if coupled_vessels:
            stretch = np.linalg.norm(u[edge_nodes[:, 1]] - u[edge_nodes[:, 0]], axis=1)
            model['maxNeighborDisplacementRatio'] = float(np.max(stretch / edge_lengths, initial=0.))
        else:
            model['maxNeighborDisplacementRatio'] = max((float(np.linalg.norm(u[j] - u[i]) / length) for i, j, length in edges), default=0.)
        model['activeNodes'] = len(ids)
        model['model'] = p['model']
        if material_profile:
            from .surgical_glioma_materials import record_lattice
            record_lattice(engine, ids, stencil, u, memory, K, targets * force_scale[:, None], held, p)
    if distributed:
        if not coupled_vessels:
            rods.tick(engine, dt, tool_motion=vessel_tool_motion)
    elif material_profile:
        move_vessels(engine)
