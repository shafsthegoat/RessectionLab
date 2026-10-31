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


_diagnostics = ContextVar('surgical_deformation_diagnostics', default=None)


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
        maxAbsDiscreteEnergyBalanceNmm=0.)
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


def _neighbors(engine, cell):
    ijk = list(np.unravel_index(cell, engine.scene.shape))
    for axis in range(3):
        for sign in (-1, 1):
            other = ijk.copy(); other[axis] += sign
            if 0 <= other[axis] < engine.scene.shape[axis]:
                yield int(np.ravel_multi_index(tuple(other), engine.scene.shape)), float(np.linalg.norm(engine.scene.affine[:3, axis]))


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


def _system(engine, p, ids, anchors, bonds):
    held = tuple(sorted({c for h in _holds(engine) for c in h['cells']} & set(ids)))
    signature = (tuple(ids), tuple(_material(engine, c) for c in ids), tuple(anchors), tuple(bonds), held, tuple((k, p[k]) for k in sorted(profile())))
    cached = getattr(engine, '_deformation_system_v2', None)
    if cached and cached[0] == signature: return cached[1:]
    n = len(ids); index = {c: i for i, c in enumerate(ids)}
    stiffness = np.array([p['tumorStiffnessNPerMm'] if _material(engine, c) == 'tumor' else p['brainStiffnessNPerMm'] for c in ids])
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
    A = K + sparse.diags(hold_diag + p['dragNsPerMm'] / p['fixedStepSeconds'])
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


def tick(engine, dt):
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
    ids = [c for c in model['active'] if c not in removed and c not in rigid]
    model['active'] = ids
    for h in engine.state['holds'].values(): h['cells'] = [c for c in h['cells'] if c not in removed]
    for field in (engine.state['deformation'], model['velocities']):
        for c in list(field):
            if int(c) not in ids: field.pop(c)
    if not ids: return
    blocked = _blocked(engine); retained = set(ids)
    bonds = sorted((c, n) for c in ids for n, _ in _neighbors(engine, c)
                   if c < n and n in retained and (c, n) not in blocked)
    anchors = sorted(set(model['anchors']) & retained)
    solve, free, index, edges, K = _system(engine, p, ids, anchors, bonds)
    h = p['fixedStepSeconds']; total = model['remainderSeconds'] + dt
    steps = int(np.floor((total + 1e-12) / h))
    model['remainderSeconds'] = max(0., round(total - steps*h, 14))
    u = np.array([engine.state['deformation'].get(str(c), [0., 0., 0.]) for c in ids])
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
    if diagnostic is not None:
        diagnostic['forceCapNodeTicks'] += int(np.count_nonzero(np.linalg.norm(forcing, axis=1) > p['maxForceN']))
    forcing *= np.minimum(1., p['maxForceN'] / np.maximum(np.linalg.norm(forcing, axis=1), 1e-15))[:, None]
    rest = np.column_stack(np.unravel_index(ids, engine.scene.shape)) @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
    removed_array = np.fromiter(removed, dtype=np.int64)
    blocking_labels = np.array([label for label, region in engine.by_label.items()
        if region.get('material') in ('bone', 'vessel', 'unknown')], dtype=engine.scene.labels.dtype)
    velocity = np.zeros_like(u)
    for _ in range(steps):
        old = u.copy(); rhs = (p['dragNsPerMm'] / h * u + forcing).ravel()
        result = np.zeros(u.size); result[free] = solve(rhs[free]); u = result.reshape((-1, 3))
        if diagnostic is not None:
            diagnostic['substeps'] += 1
            diagnostic['displacementCapNodeSteps'] += int(np.count_nonzero(np.linalg.norm(u, axis=1) > p['maxDisplacementMm']))
        u *= np.minimum(1., p['maxDisplacementMm'] / np.maximum(np.linalg.norm(u, axis=1), 1e-15))[:, None]
        # Global rescaling preserves continuity while bounding every bond strain.
        scale = min((p['maxEdgeStrain'] * length / max(np.linalg.norm(u[j] - u[i]), 1e-15) for i, j, length in edges), default=1.)
        if diagnostic is not None:
            diagnostic['strainProjectionSteps'] += int(scale < 1)
            ratio = max((float(np.linalg.norm(u[j] - u[i]) / length) for i, j, length in edges), default=0.)
            diagnostic['maxPreStrainProjectionNeighborRatio'] = max(diagnostic['maxPreStrainProjectionNeighborRatio'], ratio)
        if scale < 1: u *= scale
        # Conservative swept point-in-source-cell obstacle constraint; collision
        # integration additionally accounts for the displaced voxel's extent.
        collision = False
        for alpha in (.25, .5, .75, 1.):
            world = rest + old + alpha * (u - old)
            vox = np.rint(world @ engine.scene.inverse[:3, :3].T + engine.scene.inverse[:3, 3]).astype(int)
            inside = np.all((vox >= 0) & (vox < np.array(engine.scene.shape)), axis=1)
            collision |= _obstacle_collision(engine, vox, inside, removed_array, blocking_labels)
        # Reject the coupled increment as a whole: clamping single nodes would
        # introduce fresh strain discontinuities after the strain projection.
        if collision: u = old
        if diagnostic is not None:
            diagnostic['collisionRejectedSteps'] += int(collision)
            delta = (u - old).ravel()
            ku = K @ u.ravel() + diagnostic_hold_diagonal * u.ravel()
            kold = K @ old.ravel() + diagnostic_hold_diagonal * old.ravel()
            kdelta = K @ delta + diagnostic_hold_diagonal * delta
            residual = (ku + p['dragNsPerMm'] / h * delta - forcing.ravel())[free]
            diagnostic['maxFreeResidualN'] = max(diagnostic['maxFreeResidualN'], float(np.max(np.abs(residual), initial=0.)))
            # Constant-target implicit Euler balance includes the hold-spring
            # quadratic potential, viscous and algorithmic dissipation. This
            # is solver verification, not measured retractor work.
            change = .5 * (u.ravel() @ ku - old.ravel() @ kold)
            balance = forcing.ravel() @ delta - change - p['dragNsPerMm'] / h * (delta @ delta) - .5 * delta @ kdelta
            diagnostic['maxAbsDiscreteEnergyBalanceNmm'] = max(diagnostic['maxAbsDiscreteEnergyBalanceNmm'], abs(float(balance)))
        velocity = (u - old) / h
    if steps:
        engine.state['deformation'] = {str(c): u[i].tolist() for i, c in enumerate(ids)}
        model['velocities'] = {str(c): velocity[i].tolist() for i, c in enumerate(ids)}
        model['elasticEnergyNmm'] = float(.5 * u.ravel() @ (K @ u.ravel()))
        model['maxNeighborDisplacementRatio'] = max((float(np.linalg.norm(u[j] - u[i]) / length) for i, j, length in edges), default=0.)
        model['activeNodes'] = len(ids)
        model['model'] = p['model']
