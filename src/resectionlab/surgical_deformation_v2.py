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

from . import native_backend, surgical_mechanical_exposure as mechanical


_diagnostics = ContextVar('surgical_deformation_diagnostics', default=None)
_kernel_execution = ContextVar('surgical_deformation_kernel_execution', default=None)
_strain_margin = 1. + 64. * np.finfo(np.float64).eps
_sweep_samples = (.25, .5, .75, 1.)
_numpy_error_defaults = {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
_integer_coordinate_limit = float(1 << (np.dtype(int).itemsize * 8 - 1))


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


@contextmanager
def kernel_execution_scope():
    """Count actual lattice kernel work, including uncommitted attempts."""
    report = dict(latticeTicks=0, continuumTicks=0, latticeSubsteps=0,
        sweepNativeCalls=0, sweepNativeSamplePoints=0, sweepVectorCalls=0, sweepScalarCalls=0, sweepFallbackReasons={},
        strainNativeCalls=0, strainNativeEdges=0, strainVectorCalls=0, strainRefinedEdges=0,
        strainIdleSubsteps=0, strainFallbackReasons={})
    token = _kernel_execution.set(report)
    try:
        yield report
    finally:
        _kernel_execution.reset(token)


def _record_fallback(report, kernel, reason):
    reasons = report[kernel]
    reasons[reason] = reasons.get(reason, 0) + 1


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
    signature = (grid, tuple(ids), tuple(_material(engine, c) for c in ids), tuple(anchors),
                 local_cuts, held, tuple((k, p[k]) for k in sorted(profile())))
    cached = getattr(engine, '_deformation_system_v2', None)
    if cached and cached[0] == signature: return cached[1:]
    bonds = sorted((c, n) for c in ids for n, _ in _neighbors(engine, c)
                   if c < n and n in retained and (c, n) not in local_cuts)
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
    edge_nodes = np.array([(i, j) for i, j, _ in edges], dtype=np.int64).reshape(-1, 2)
    edge_lengths = np.array([length for _, _, length in edges], dtype=float)
    shortest = min((length for _, _, length in edges), default=0.)
    cached = (signature, solve, free, index, edge_nodes, edge_lengths, shortest, K)
    engine._deformation_system_v2 = cached
    return cached[1:]


def _obstacle_collision(engine, vox, inside, removed, blocking_labels):
    """Same whole-increment obstacle decision, using only sampled grid cells."""
    if not np.all(inside):
        return True
    cells = np.ravel_multi_index(vox.T, engine.scene.shape)
    blocked = np.isin(engine.scene.labels.ravel()[cells], blocking_labels)
    return bool(np.any(blocked & ~np.isin(cells, removed)))


def _sweep_scalar(engine, rest, old, proposed, removed, blocking_labels, execution=None):
    if execution is not None: execution['sweepScalarCalls'] += 1
    inverse = engine.scene.inverse
    collision = False
    for alpha in _sweep_samples:
        world = rest + old + alpha * (proposed - old)
        vox = np.rint(world @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
        inside = np.all((vox >= 0) & (vox < engine.scene.shape), axis=1)
        collision |= _obstacle_collision(engine, vox, inside, removed, blocking_labels)
    return collision


def _sweep_numpy(engine, rest, old, proposed, removed, blocking_labels, execution=None):
    inverse = engine.scene.inverse
    # Keep the original affine multiplication shape at each alpha so BLAS
    # evaluates it in the same order. Classify the rounded samples together.
    with np.errstate(all='ignore'):
        start = rest + old
        travel = proposed - old
        worlds = [(start + alpha * travel) @ inverse[:3, :3].T for alpha in _sweep_samples]
        rounded = np.rint(np.concatenate(worlds) + inverse[:3, 3])
    if not np.all((rounded >= -_integer_coordinate_limit) & (rounded < _integer_coordinate_limit)):
        # Keep the original per-sample validation and warning/exception order.
        if execution is not None: _record_fallback(execution, 'sweepFallbackReasons', 'vector-numeric-domain')
        return _sweep_scalar(engine, rest, old, proposed, removed, blocking_labels, execution)
    if execution is not None: execution['sweepVectorCalls'] += 1
    vox = rounded.astype(int)
    inside = np.all((vox >= 0) & (vox < engine.scene.shape), axis=1)
    return _obstacle_collision(engine, vox, inside, removed, blocking_labels)


def _strain_candidates_numpy(u, edges, lengths, max_strain):
    if u.dtype != np.float64:
        return None
    if max_strain and not 1e-120 <= max_strain <= 1e120:
        return None
    if not np.all((lengths >= 1e-120) & (lengths <= 1e120)):
        return None
    # Speculative scores only decide which edges to refine. Let the original
    # scalar arithmetic handle extreme values (and any configured FP errors).
    with np.errstate(all='ignore'):
        delta = u[edges[:, 1]] - u[edges[:, 0]]
        norms = np.linalg.norm(delta, axis=1)
        zero = norms == 0.
        if not np.all(zero | ((norms >= 1e-120) & (norms <= 1e120))) or np.any(delta[zero]):
            return None
        if max_strain:
            scores = max_strain * lengths / np.maximum(norms, 1e-15)
            return np.flatnonzero(scores <= scores.min() * _strain_margin)
        scores = norms / lengths
        best = scores.max()
        return np.flatnonzero(scores >= best / _strain_margin) if best else (0,)


def _strain_extreme(u, edges, lengths, max_strain, native, execution, fallback=None):
    if not len(lengths):
        return 1. if max_strain else 0.
    candidates = None
    if native is not None and fallback is None:
        selected = native(u, edges, lengths, max_strain)
        if type(selected) is str:
            fallback = selected
        else:
            candidates = selected
            if execution is not None:
                execution['strainNativeCalls'] += 1
                execution['strainNativeEdges'] += len(lengths)
    scalar_reference = fallback in ('diagnostic-reference', 'numpy-error-policy', 'rounding-mode', 'floating-point-format')
    if candidates is None and not scalar_reference:
        if execution is not None:
            _record_fallback(execution, 'strainFallbackReasons', fallback or 'native-numeric-domain')
        candidates = _strain_candidates_numpy(u, edges, lengths, max_strain)
        if execution is not None and candidates is not None:
            execution['strainVectorCalls'] += 1
    if candidates is None:
        candidates = range(len(lengths))
        if execution is not None:
            reason = fallback if scalar_reference else 'scalar-numeric-domain'
            _record_fallback(execution, 'strainFallbackReasons', reason)
    if execution is not None:
        execution['strainRefinedEdges'] += len(candidates)
    # Evaluate contenders in their original edge order with the stock scalar
    # norm. A BLAS reduction may differ by ulps from the batched estimate.
    if max_strain:
        return min(max_strain * lengths[k] / max(np.linalg.norm(u[edges[k, 1]] - u[edges[k, 0]]), 1e-15)
                   for k in candidates)
    return max(float(np.linalg.norm(u[edges[k, 1]] - u[edges[k, 0]]) / lengths[k]) for k in candidates)


def tick(engine, dt):
    initialize(engine)
    if not np.isfinite(dt) or dt < 0: raise ValueError('INVALID_DEFORMATION_TIME')
    p = _settings(engine); model = engine.state['deformationModel']; removed = set(engine.state['removed'])
    diagnostic = _diagnostics.get()
    execution = _kernel_execution.get()
    if diagnostic is not None: diagnostic['ticks'] += 1
    if 'brainContinuumV1' in engine.scene.properties:
        from .surgical_continuum import tick as continuum_tick
        if execution is not None: execution['continuumTicks'] += 1
        return continuum_tick(engine, dt, diagnostic)
    if execution is not None: execution['latticeTicks'] += 1
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
    anchors = sorted(set(model['anchors']) & retained)
    solve, free, index, edge_nodes, edge_lengths, shortest, K = _system(engine, p, ids, anchors, retained, blocked)
    h = p['fixedStepSeconds']; total = model['remainderSeconds'] + dt
    steps = int(np.floor((total + 1e-12) / h))
    model['remainderSeconds'] = max(0., round(total - steps*h, 14))
    sweep, strain_candidates, no_native = None, None, None
    if steps:
        # A batched estimate may skip an edge where a caller configured
        # per-operation floating-point errors. Honor these policies verbatim.
        if np.geterr() == _numpy_error_defaults:
            sweep, strain_candidates, no_native = native_backend.deformation_kernels()
        else:
            no_native = 'numpy-error-policy'
    strain_fallback = 'diagnostic-reference' if diagnostic is not None else no_native
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
    force_scale = np.minimum(1., p['maxForceN'] / np.maximum(np.linalg.norm(forcing, axis=1), 1e-15))
    forcing *= force_scale[:, None]
    track_mechanics = steps and mechanical.enabled(engine)
    if track_mechanics:
        labels = engine.scene.labels.ravel()[ids]
        retractor_cells = sorted({c for key, hold in engine.state['holds'].items()
                                  if 'target' in hold and engine.state['slots'][key]['instrument'] == 'brain_retractor'
                                  for c in hold['cells'] if c in index})
        retractor_nodes = np.asarray([index[cell] for cell in retractor_cells], dtype=int)
        effective_targets = targets[retractor_nodes] * force_scale[retractor_nodes, None]
    rest = np.column_stack(np.unravel_index(ids, engine.scene.shape)) @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
    removed_array = np.fromiter(removed, dtype=np.int64)
    if sweep is not None: removed_array.sort()
    blocking_labels = np.array([label for label, region in engine.by_label.items()
        if region.get('material') in ('bone', 'vessel', 'unknown')], dtype=engine.scene.labels.dtype)
    # For every bond the displacement difference is at most 2*sqrt(3)
    # times the largest absolute node component. Staying below one eighth
    # of the shortest bond's allowed extension leaves the projection idle,
    # including ample margin for roundoff. Resolve the edge limit near the
    # threshold, outside ordinary magnitudes, or when collecting diagnostics.
    strain_limit = p['maxEdgeStrain'] * shortest
    safe_component = strain_limit / 8 if 2e-15 < strain_limit < 1e150 and diagnostic is None else 0.
    velocity = np.zeros_like(u)
    for _ in range(steps):
        old = u.copy(); rhs = (p['dragNsPerMm'] / h * u + forcing).ravel()
        result = np.zeros(u.size); result[free] = solve(rhs[free]); u = result.reshape((-1, 3))
        if execution is not None: execution['latticeSubsteps'] += 1
        if diagnostic is not None:
            diagnostic['substeps'] += 1
            diagnostic['displacementCapNodeSteps'] += int(np.count_nonzero(np.linalg.norm(u, axis=1) > p['maxDisplacementMm']))
        u *= np.minimum(1., p['maxDisplacementMm'] / np.maximum(np.linalg.norm(u, axis=1), 1e-15))[:, None]
        # Global rescaling preserves continuity while bounding every bond strain.
        scale = 1.
        if safe_component and np.max(np.abs(u)) < safe_component:
            if execution is not None: execution['strainIdleSubsteps'] += 1
        else:
            scale = _strain_extreme(u, edge_nodes, edge_lengths, p['maxEdgeStrain'],
                strain_candidates, execution, strain_fallback)
        if diagnostic is not None:
            diagnostic['strainProjectionSteps'] += int(scale < 1)
            ratio = _strain_extreme(u, edge_nodes, edge_lengths, 0., None, None, 'diagnostic-reference')
            diagnostic['maxPreStrainProjectionNeighborRatio'] = max(diagnostic['maxPreStrainProjectionNeighborRatio'], ratio)
        if scale < 1: u *= scale
        # Conservative swept point-in-source-cell obstacle constraint; collision
        # integration additionally accounts for the displaced voxel's extent.
        collision = (no_native if sweep is None else
            sweep(rest, old, u, engine.scene.labels, engine.scene.inverse, removed_array, blocking_labels))
        if type(collision) is str:
            if execution is not None:
                _record_fallback(execution, 'sweepFallbackReasons', collision)
            sweep_fallback = _sweep_scalar if collision in (
                'numpy-error-policy', 'numeric-domain', 'rounding-mode', 'floating-point-format') else _sweep_numpy
            collision = sweep_fallback(engine, rest, old, u, removed_array, blocking_labels, execution)
        elif execution is not None:
            execution['sweepNativeCalls'] += 1
            execution['sweepNativeSamplePoints'] += len(_sweep_samples) * len(rest)
        # Reject the coupled increment as a whole: clamping single nodes would
        # introduce fresh strain discontinuities after the strain projection.
        if collision: u = old
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
        model['maxNeighborDisplacementRatio'] = _strain_extreme(u, edge_nodes, edge_lengths, 0.,
            strain_candidates, execution, strain_fallback)
        model['activeNodes'] = len(ids)
        model['model'] = p['model']
