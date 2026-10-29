"""Opt-in voxel-owned, small-gradient linear viscoelastic continuum.

Each retained source voxel owns a fully integrated hex8 element. Corner nodes
are shared only through uncut faces, including connectivity around each corner.
This is a bounded local research model, not patient-calibrated physiology.
No foundation, clipping, hourglass stabilization or invented CSF load is added.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import copy
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .core import semantic_digest

PROPERTY = 'brainContinuumV1'
SCHEMA = 'resectionlab.voxel-continuum.v1'
CORNERS = np.array(list(product((0, 1), repeat=3)), dtype=np.int64)
SOFT = {'soft_tissue', 'brain', 'tumor'}


def research_profile(materials, *, relaxation_seconds=(), hold_stiffness=.02):
    """Require explicit material assignments; supply no inferred tumor property."""
    return {'schema': SCHEMA, 'materials': materials,
        'relaxationSeconds': list(relaxation_seconds),
        'bulkRelaxation': 'same-spectrum-as-shear',
        'handleStiffnessNPerMm': hold_stiffness,
        'maxElementGradientNorm': .2, 'maxDofs': 24000,
        'provenance': {'kind': 'assumed-research', 'description':
            'Explicit reduced small-gradient research model. Material transfer, relaxation, '
            'contact and open/CSF boundaries are not patient measurements.'}}


def _number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('INVALID_CONTINUUM_PROFILE:' + label)


def validate(scene):
    p = scene.properties.get(PROPERTY)
    if p is None: return None
    if 'tissuePhysicsV2' not in scene.properties:
        raise ValueError('CONTINUUM_REQUIRES_EXPLICIT_V2_ENVIRONMENT')
    if not isinstance(p, dict) or set(p) != set(research_profile({})) or p.get('schema') != SCHEMA:
        raise ValueError('COMPLETE_CONTINUUM_PROFILE_REQUIRED')
    provenance = p['provenance']
    if not isinstance(provenance, dict) or provenance.get('kind') != 'assumed-research' or not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
        raise ValueError('EXPLICIT_CONTINUUM_RESEARCH_PROVENANCE_REQUIRED')
    times = p['relaxationSeconds']
    if not isinstance(times, list) or len(times) > 2:
        raise ValueError('AT_MOST_TWO_EXPLICIT_RELAXATION_BRANCHES')
    for t in times: _number(t, .01, 1000., 'relaxationSeconds')
    if p['bulkRelaxation'] != 'same-spectrum-as-shear':
        raise ValueError('CONTINUUM_REQUIRES_EXPLICIT_BULK_RELAXATION_ASSUMPTION')
    _number(p['handleStiffnessNPerMm'], 1e-6, 10., 'handleStiffnessNPerMm')
    _number(p['maxElementGradientNorm'], .001, .2, 'maxElementGradientNorm')
    if type(p['maxDofs']) is not int or not 24 <= p['maxDofs'] <= 24000:
        raise ValueError('CONTINUUM_DOF_LIMIT')
    expected = {str(r['id']) for r in scene.regions if r['material'] in SOFT}
    if not isinstance(p['materials'], dict) or set(p['materials']) != expected:
        raise ValueError('CONTINUUM_REQUIRES_EVERY_SOFT_REGION_MATERIAL')
    for m in p['materials'].values():
        if not isinstance(m, dict) or set(m) != {'shearPa', 'poissonRatio', 'relaxationShearPa', 'source'}:
            raise ValueError('COMPLETE_CONTINUUM_MATERIAL_REQUIRED')
        _number(m['shearPa'], .01, 1e6, 'shearPa')
        # This displacement-only first release is not qualified near nu=.5.
        _number(m['poissonRatio'], 0., .45, 'poissonRatio')
        if not isinstance(m['relaxationShearPa'], list) or len(m['relaxationShearPa']) != len(times):
            raise ValueError('CONTINUUM_BRANCH_MATERIAL_MISMATCH')
        for mu in m['relaxationShearPa']: _number(mu, 0., 1e6, 'relaxationShearPa')
        if not isinstance(m['source'], str) or not m['source'].strip():
            raise ValueError('CONTINUUM_MATERIAL_SOURCE_REQUIRED')
    return p


def _shape_gradients(inverse, point):
    signs = 2 * CORNERS - 1
    derivatives = np.empty((8, 3))
    for axis in range(3):
        others = [j for j in range(3) if j != axis]
        derivatives[:, axis] = signs[:, axis] * np.prod(1 + signs[:, others] * np.asarray(point)[others], axis=1) / 8.
    return derivatives @ inverse


def element_operators(affine):
    """Geometric shear/Lame matrices; full Gauss integration, mm units."""
    a = np.asarray(affine, float)[:3, :3]
    inverse = np.linalg.inv(a / 2.)
    volume_weight = abs(float(np.linalg.det(a / 2.)))
    shear_d = np.diag([2., 2., 2., 1., 1., 1.])
    lame_d = np.zeros((6, 6)); lame_d[:3, :3] = 1.
    shear = np.zeros((24, 24)); lame = np.zeros_like(shear)
    gradients = []
    for point in product((-1 / np.sqrt(3.), 1 / np.sqrt(3.)), repeat=3):
        gradient = _shape_gradients(inverse, point)
        gradients.append(gradient)
        b = np.zeros((6, 24))
        for node, (x, y, z) in enumerate(gradient):
            col = 3 * node
            b[0, col] = x; b[1, col+1] = y; b[2, col+2] = z
            b[3, col:col+2] = [y, x]
            b[4, col+1:col+3] = [z, y]
            b[5, [col, col+2]] = [z, x]
        shear += b.T @ shear_d @ b * volume_weight
        lame += b.T @ lame_d @ b * volume_weight
    # Q1 gradient norms can exceed the Gauss values near element corners.
    # Each gradient is affine in each reference coordinate separately, so a
    # corner bound covers the entire affine voxel (including oblique grids).
    gradients.extend(_shape_gradients(inverse, point) for point in product((-1., 1.), repeat=3))
    return shear, lame, np.asarray(gradients)


@dataclass
class Mesh:
    cells: tuple
    nodes: np.ndarray
    element_nodes: np.ndarray
    element_dofs: np.ndarray
    node_owners: tuple
    components: tuple
    volume_mm3: float


def voxel_mesh(scene, cells, blocked=(), *, max_dofs=24000):
    """Split a geometric corner by incident uncut-face connectivity."""
    cells = tuple(cells)
    if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) or not 0 <= c < scene.labels.size for c in cells):
        raise ValueError('INVALID_CONTINUUM_SOURCE_CELL')
    cells = tuple(sorted(set(map(int, cells))))
    soft_ids = {r['id'] for r in scene.regions if r['material'] in SOFT}
    if any(int(scene.labels.flat[c]) not in soft_ids for c in cells):
        raise ValueError('CONTINUUM_REQUIRES_SOFT_SOURCE_OWNERSHIP')
    blocked = {tuple(sorted(map(int, edge))) for edge in blocked}
    indices = {c: np.array(np.unravel_index(c, scene.shape)) for c in cells}
    lookup = {tuple(i): c for c, i in indices.items()}
    neighbors = {c: [] for c in cells}
    for c, ijk in indices.items():
        for axis in range(3):
            j = ijk.copy(); j[axis] += 1
            other = lookup.get(tuple(j))
            if other is not None and (min(c, other), max(c, other)) not in blocked:
                neighbors[c].append(other); neighbors[other].append(c)
    incident = {}
    for c, ijk in indices.items():
        for local, offset in enumerate(CORNERS):
            incident.setdefault(tuple(ijk + offset), []).append((c, local))
    nodes = []; owners = []; cell_nodes = {c: [None] * 8 for c in cells}
    for corner, entries in sorted(incident.items()):
        remaining = {c for c, _ in entries}
        local_ids = dict(entries)
        while remaining:
            first = min(remaining); remaining.remove(first); group = {first}; queue = [first]
            while queue:
                for other in neighbors[queue.pop()]:
                    if other in remaining:
                        remaining.remove(other); group.add(other); queue.append(other)
            n = len(nodes)
            nodes.append(scene.affine[:3, :3] @ (np.asarray(corner) - .5) + scene.affine[:3, 3])
            owner = tuple((c, local_ids[c]) for c in sorted(group)); owners.append(owner)
            for c, local in owner: cell_nodes[c][local] = n
            if len(nodes) * 3 > max_dofs: raise ValueError('CONTINUUM_DOF_LIMIT')
    remaining = set(cells); components = []
    while remaining:
        first = min(remaining); remaining.remove(first); group = {first}; queue = [first]
        while queue:
            for other in neighbors[queue.pop()]:
                if other in remaining:
                    remaining.remove(other); group.add(other); queue.append(other)
        components.append(tuple(sorted(group)))
    element_nodes = np.array([cell_nodes[c] for c in cells], dtype=np.int64).reshape((-1, 8))
    dofs = (3 * element_nodes[:, :, None] + np.arange(3)).reshape((-1, 24))
    return Mesh(cells, np.asarray(nodes).reshape((-1, 3)), element_nodes, dofs,
        tuple(owners), tuple(components), abs(float(np.linalg.det(scene.affine[:3, :3]))))


def assemble(scene, mesh, materials):
    """Assemble equilibrium and Maxwell-branch stiffness, N/mm."""
    shear, lame, gradients = element_operators(scene.affine)
    branch_count = len(next(iter(materials.values()))['relaxationShearPa'])
    values = [[] for _ in range(1 + branch_count)]; rows = []; cols = []
    for c, dofs in zip(mesh.cells, mesh.element_dofs):
        m = materials[str(int(scene.labels.flat[c]))]
        rows.extend(np.repeat(dofs, 24)); cols.extend(np.tile(dofs, 24))
        for branch, pa in enumerate([m['shearPa'], *m['relaxationShearPa']]):
            mu = pa * 1e-6  # Pa -> N/mm².
            lam = 2 * mu * m['poissonRatio'] / (1 - 2 * m['poissonRatio'])
            values[branch].extend((mu * shear + lam * lame).ravel())
    shape = (len(mesh.nodes) * 3,) * 2
    matrices = tuple(sparse.coo_matrix((v, (rows, cols)), shape=shape).tocsr() for v in values)
    return matrices, gradients


def _face_neighbors(scene, c):
    ijk = np.array(np.unravel_index(c, scene.shape))
    for axis in range(3):
        for sign in (-1, 1):
            j = ijk.copy(); j[axis] += sign
            if np.all((j >= 0) & (j < scene.shape)):
                yield int(np.ravel_multi_index(j, scene.shape)), axis, sign


def _fixed_nodes(engine, mesh, anchors, blocked):
    active = set(mesh.cells); removed = set(engine.state['removed']); fixed = set()
    index = {c: i for i, c in enumerate(mesh.cells)}
    for c in sorted(set(anchors) & active):
        for other, axis, sign in _face_neighbors(engine.scene, c):
            if other in active or other in removed or tuple(sorted((c, other))) in blocked: continue
            material = engine.by_label.get(int(engine.scene.labels.flat[other]), {}).get('material')
            if material not in SOFT: continue
            local = np.flatnonzero(CORNERS[:, axis] == (sign + 1) // 2)
            fixed.update(map(int, mesh.element_nodes[index[c], local]))
    return fixed


def _rigid_rows(point):
    x, y, z = point
    return np.array([[1., 0., 0., 0., z, -y], [0., 1., 0., -z, 0., x], [0., 0., 1., y, -x, 0.]])


def check_support(mesh, fixed_nodes, held_cells):
    index = {c: i for i, c in enumerate(mesh.cells)}
    for component in mesh.components:
        nodes = sorted(set(mesh.element_nodes[[index[c] for c in component]].ravel()))
        center = mesh.nodes[nodes].mean(axis=0)
        scale = max(float(np.linalg.norm(mesh.nodes[nodes] - center, axis=1).max()), 1e-12)
        rows = [_rigid_rows((mesh.nodes[n] - center) / scale) for n in nodes if n in fixed_nodes]
        for c in component:
            if c in held_cells:
                point = mesh.nodes[mesh.element_nodes[index[c]]].mean(axis=0)
                rows.append(_rigid_rows((point - center) / scale))
        if not rows or np.linalg.matrix_rank(np.vstack(rows), tol=1e-9) < 6:
            raise ValueError('CONTINUUM_UNSUPPORTED_RIGID_COMPONENT: explicit support is required')


def _system(engine, ids, anchors, blocked, p, h):
    from .surgical_deformation_v2 import _neighbors as old_neighbors
    # Source arrays/material declarations are shared, never mutable episode
    # state. Refuse an exchanged source instead of reusing a stale factorization.
    engine.scene.require_cache_source()
    holds = {key: tuple(sorted(set(hold['cells']) & set(ids))) for key, hold in engine.state['holds'].items() if 'target' in hold}
    signature = (tuple(ids), tuple(anchors), tuple(sorted(blocked)), tuple(sorted(holds.items())),
        semantic_digest(p), tuple(engine.scene.affine.ravel()), h)
    cached = getattr(engine, '_continuum_system_v1', None)
    if cached is not None and cached[0] == signature: return cached[1]
    mesh = voxel_mesh(engine.scene, ids, blocked, max_dofs=p['maxDofs'])
    matrices, gradients = assemble(engine.scene, mesh, p['materials'])
    fixed_nodes = _fixed_nodes(engine, mesh, anchors, blocked)
    held_cells = {c for cells in holds.values() for c in cells}
    check_support(mesh, fixed_nodes, held_cells)
    index = {c: i for i, c in enumerate(mesh.cells)}
    rows = []; cols = []; values = []; contacts = []
    for key, cells in sorted(holds.items()):
        if not cells: continue
        k = p['handleStiffnessNPerMm'] / len(cells)
        for c in cells:
            nodes = mesh.element_nodes[index[c]]
            contacts.append((key, c, nodes, k))
            for axis in range(3):
                dofs = nodes * 3 + axis
                rows.extend(np.repeat(dofs, 8)); cols.extend(np.tile(dofs, 8)); values.extend([k / 64.] * 64)
    handle = sparse.coo_matrix((values, (rows, cols)), shape=matrices[0].shape).tocsr()
    ratios = [t / (t + h) for t in p['relaxationSeconds']]
    effective = matrices[0] + handle
    for ratio, matrix in zip(ratios, matrices[1:]): effective = effective + ratio * matrix
    free = np.ones(len(mesh.nodes) * 3, dtype=bool)
    for n in fixed_nodes: free[3*n:3*n+3] = False
    try:
        solve = factorized(effective[free][:, free].tocsc()) if free.any() else lambda rhs: np.empty(0)
    except RuntimeError as error:
        raise ValueError('CONTINUUM_FACTORIZATION_FAILED: unsupported or ill-conditioned local system') from error
    result = dict(mesh=mesh, matrices=matrices, gradients=gradients, handle=handle,
        contacts=contacts, effective=effective, free=free, solve=solve, ratios=ratios)
    # The diagnostic uses unchanged reference adjacency/lengths. Cache its
    # original iteration order with this exact source/topology signature;
    # preserve scalar norm arithmetic so canonical recorded bits do not change.
    retained = set(ids)
    result['neighbor_pairs'] = tuple((i, index[other], length)
        for i, c in enumerate(mesh.cells) for other, length in old_neighbors(engine, c)
        if other in retained and c < other and (c, other) not in blocked)
    engine._continuum_system_v1 = (signature, result)
    return result


def _unpack(mesh, field):
    vector = np.zeros((len(mesh.nodes), 3))
    for n, owners in enumerate(mesh.node_owners):
        existing = [field[str(c)][local] for c, local in owners if str(c) in field]
        if existing:
            value = np.asarray(existing, float)
            if value.shape != (len(existing), 3) or not np.isfinite(value).all() or np.max(np.abs(value - value[0])) > 1e-10:
                raise ValueError('CONTINUUM_INCONSISTENT_SHARED_CORNER_HISTORY')
            vector[n] = value[0]
    return vector.ravel()


def _pack(mesh, vector):
    vector = vector.reshape((-1, 3))
    return {str(c): vector[nodes].tolist() for c, nodes in zip(mesh.cells, mesh.element_nodes)}


def update_memory(memories, displacement, relaxation_seconds, h):
    """Backward Euler of z'=(u-z)/tau; also used by imposed-strain controls."""
    return [(t * z + h * displacement) / (t + h) for z, t in zip(memories, relaxation_seconds)]


def exposure_increment(scene, mesh, gradients, displacement, memories, materials, h):
    """Integrate modeled strain/stress squared over volume/time; NOT injury.

    Full Gauss integration matches the element's constitutive quadrature. The
    invariants provide an inspectable cumulative exposure, with no biological
    threshold, neurological endpoint or calibrated risk conversion.
    """
    gradients = gradients[:8]
    local = displacement.reshape((-1, 3))[mesh.element_nodes]
    gradient = np.einsum('eai,gaj->egij', local, gradients)
    strain = (gradient + gradient.transpose(0, 1, 3, 2)) / 2.
    labels = np.asarray([int(scene.labels.flat[c]) for c in mesh.cells])
    stress = np.zeros_like(strain)
    for branch, field in enumerate([displacement, *[displacement-z for z in memories]]):
        g = np.einsum('eai,gaj->egij', field.reshape((-1, 3))[mesh.element_nodes], gradients)
        eps = (g + g.transpose(0, 1, 3, 2)) / 2.
        mu = np.asarray([([materials[str(label)]['shearPa'], *materials[str(label)]['relaxationShearPa']])[branch] for label in labels])
        nu = np.asarray([materials[str(label)]['poissonRatio'] for label in labels])
        lam = 2 * mu * nu / (1 - 2 * nu)
        stress += 2 * mu[:, None, None, None] * eps + lam[:, None, None, None] * np.trace(eps, axis1=2, axis2=3)[:, :, None, None] * np.eye(3)
    weight = mesh.volume_mm3 * h / 8.
    return {str(label): {'strainSquaredVolumeTimeMm3Seconds': float(np.sum(strain[labels == label]**2) * weight),
        'stressSquaredVolumeTimePa2Mm3Seconds': float(np.sum(stress[labels == label]**2) * weight)} for label in sorted(set(labels))}


def _cell_energy(scene, p, cell, displacement, memories):
    shear, lame, _ = element_operators(scene.affine)
    m = p['materials'][str(int(scene.labels.flat[cell]))]
    energy = 0.
    for mu_pa, field in zip([m['shearPa'], *m['relaxationShearPa']], [displacement, *memories]):
        mu = mu_pa * 1e-6; lam = 2 * mu * m['poissonRatio'] / (1 - 2 * m['poissonRatio'])
        field = np.asarray(field, float).ravel()
        energy += .5 * field @ ((mu * shear + lam * lame) @ field)
    return float(energy)


def tick(engine, dt, diagnostic=None):
    """Same shared clock/ownership/holds; separate opt-in constitutive semantics."""
    from . import surgical_deformation_v2 as old
    p = engine.scene.properties[PROPERTY]
    settings = old._settings(engine); model = engine.state['deformationModel']
    removed = set(engine.state['removed']); rigid = old.rigid_cells(engine)
    if 'patientPosition' in engine.state:
        from .surgical_position import activate_exposed_patch
        activate_exposed_patch(engine, radius_mm=settings['radiusMm'], max_nodes=settings['maxNodes'])
    ids = sorted(c for c in model['active'] if c not in removed and c not in rigid)
    model['active'] = ids
    for hold in engine.state['holds'].values(): hold['cells'] = [c for c in hold['cells'] if c not in removed]
    for name in ('deformation',):
        for key in list(engine.state[name]):
            if int(key) not in ids: engine.state[name].pop(key)
    if PROPERTY not in engine.state:
        if engine.state['deformation']:
            raise ValueError('CONTINUUM_REQUIRES_FRESH_CORNER_HISTORY')
        engine.state[PROPERTY] = {'schema': SCHEMA, 'cornerDisplacementMm': {},
            'memoryDisplacementMm': [{} for _ in p['relaxationSeconds']], 'dissipatedEnergyNmm': 0.,
            'removedStoredEnergyNmm': 0., 'numericalDissipationNmm': 0., 'removedElementCount': 0,
            'exposureBySourceLabel': {}}
    state = engine.state[PROPERTY]
    gone = set(state['cornerDisplacementMm']) - {str(c) for c in ids}
    for key in sorted(gone, key=int):
        field = np.asarray(state['cornerDisplacementMm'][key])
        branch_fields = [field - np.asarray(memory[key]) for memory in state['memoryDisplacementMm']]
        exported = _cell_energy(engine.scene, p, int(key), field, branch_fields)
        if int(key) in removed:
            state['removedStoredEnergyNmm'] += max(0., exported)
            state['removedElementCount'] += 1
        elif exported > 1e-10 or np.max(np.abs(field), initial=0.) > 1e-10 or any(
                np.max(np.abs(memory[key]), initial=0.) > 1e-10 for memory in state['memoryDisplacementMm']):
            raise ValueError('CONTINUUM_UNSUPPORTED_HISTORY_TRANSFER: displaced tissue cannot silently become rigid/inactive')
        state['cornerDisplacementMm'].pop(key)
        for memory in state['memoryDisplacementMm']: memory.pop(key)
    if not ids:
        state['cornerDisplacementMm'] = {}; state['memoryDisplacementMm'] = [{} for _ in p['relaxationSeconds']]
        if 'activeElements' in state:
            state.update(activeElements=0, activeDofs=0, handleReactionsN={})
        model['velocities'] = {}
        model.update(elasticEnergyNmm=0., activeNodes=0, maxNeighborDisplacementRatio=0., model=SCHEMA)
        return
    blocked = old._blocked(engine); anchors = sorted(set(model['anchors']) & set(ids))
    h = settings['fixedStepSeconds']; total = model['remainderSeconds'] + dt
    steps = int(np.floor((total + 1e-12) / h))
    model['remainderSeconds'] = max(0., round(total - steps*h, 14))
    if not steps: return
    system = _system(engine, ids, anchors, blocked, p, h)
    mesh = system['mesh']; matrices = system['matrices']; free = system['free']
    u = _unpack(mesh, state['cornerDisplacementMm'])
    memories = [_unpack(mesh, field) for field in state['memoryDisplacementMm']]
    if np.max(np.abs(u[~free]), initial=0.) > 1e-10:
        raise ValueError('CONTINUUM_SUPPORT_CHANGED_WITH_NONZERO_DISPLACEMENT')
    for i, c in enumerate(mesh.cells):
        if str(c) in state['cornerDisplacementMm']: continue
        dofs = mesh.element_dofs[i]
        energy = _cell_energy(engine.scene, p, c, u[dofs], [u[dofs] - z[dofs] for z in memories])
        if energy > 1e-10 or np.max(np.abs(u[dofs]), initial=0.) > 1e-10 or any(
                np.max(np.abs(z[dofs]), initial=0.) > 1e-10 for z in memories):
            raise ValueError('CONTINUUM_PATCH_ACTIVATION_CHANGES_HISTORY: a validated history transfer is required')
    gravity = np.zeros_like(u)
    if 'patientPosition' in engine.state:
        from .surgical_position import gravity_forces_n
        forces = gravity_forces_n(engine, mesh.cells)
        for nodes, force in zip(mesh.element_nodes, forces):
            for n in nodes: gravity[3*n:3*n+3] += force / 8.
    handle_rhs = np.zeros_like(u); targets = []
    for key, c, nodes, k in system['contacts']:
        hold = engine.state['holds'][key]
        target = np.subtract(hold['target'], hold['origin']) + np.asarray(hold.get('baseDisplacements', {}).get(str(c), [0., 0., 0.]))
        if np.linalg.norm(target) > settings['maxDisplacementMm']:
            raise ValueError('CONTINUUM_TARGET_DOMAIN: no target clipping is allowed')
        for n in nodes: handle_rhs[3*n:3*n+3] += k * target / 8.
        targets.append((key, nodes, k, target))
    external = gravity + handle_rhs
    rest = engine.scene.points(ids)
    max_residual = state.get('maxFreeResidualN', 0.)
    max_balance = state.get('maxEnergyBalanceResidualNmm', 0.)
    max_gradient = state.get('maxGradientNorm', 0.)
    dissipation = state['dissipatedEnergyNmm']
    numerical_total = state['numericalDissipationNmm']
    for _ in range(steps):
        before = u.copy(); old_memories = memories
        rhs = external.copy()
        for ratio, matrix, z in zip(system['ratios'], matrices[1:], memories): rhs += ratio * (matrix @ z)
        u = np.zeros_like(before); u[free] = system['solve'](rhs[free])
        if not np.isfinite(u).all(): raise ValueError('CONTINUUM_NONFINITE_RESPONSE')
        memories = update_memory(old_memories, u, p['relaxationSeconds'], h)
        nodal = u.reshape((-1, 3))
        local_u = nodal[mesh.element_nodes]
        gradient = np.einsum('eai,gaj->egij', local_u, system['gradients'])
        max_gradient = max(max_gradient, float(np.linalg.norm(gradient, axis=(2, 3)).max()))
        if max_gradient > p['maxElementGradientNorm'] + 1e-12:
            raise ValueError(f'CONTINUUM_SMALL_GRADIENT_DOMAIN: gradient norm {max_gradient:.6g} exceeds {p["maxElementGradientNorm"]:.6g}; no strain projection is allowed')
        centers = local_u.mean(axis=1)
        if np.max(np.linalg.norm(centers, axis=1)) > settings['maxDisplacementMm']:
            raise ValueError('CONTINUUM_DISPLACEMENT_DOMAIN')
        old_centers = before.reshape((-1, 3))[mesh.element_nodes].mean(axis=1)
        # Retain the existing conservative source-center sweep, then the parent
        # engine rechecks occupancy and both complete instrument envelopes.
        labels = np.array([label for label, region in engine.by_label.items() if region['material'] in ('bone', 'vessel', 'unknown')], dtype=engine.scene.labels.dtype)
        removed_array = np.asarray(sorted(removed), dtype=np.int64)
        for alpha in (.25, .5, .75, 1.):
            world = rest + old_centers + alpha * (centers - old_centers)
            vox = np.rint(world @ engine.scene.inverse[:3, :3].T + engine.scene.inverse[:3, 3]).astype(int)
            inside = np.all((vox >= 0) & (vox < np.asarray(engine.scene.shape)), axis=1)
            if old._obstacle_collision(engine, vox, inside, removed_array, labels):
                raise ValueError('CONTINUUM_OBSTACLE_COLLISION: coupled increment rejected')
        reactions = {}
        for key, nodes, k, target in targets:
            reactions[key] = reactions.get(key, np.zeros(3)) + k * (target - nodal[nodes].mean(axis=0))
        if any(np.linalg.norm(force) > settings['maxForceN'] for force in reactions.values()):
            raise ValueError('CONTINUUM_FORCE_DOMAIN: no force clipping is allowed')
        delta = u - before
        residual = matrices[0] @ u + system['handle'] @ u - external
        old_energy = .5 * before @ (matrices[0] @ before) + .5 * before @ (system['handle'] @ before)
        energy = .5 * u @ (matrices[0] @ u) + .5 * u @ (system['handle'] @ u)
        numerical_dissipation = .5 * delta @ ((matrices[0] + system['handle']) @ delta)
        viscous = 0.
        for t, matrix, z, old_z in zip(p['relaxationSeconds'], matrices[1:], memories, old_memories):
            stretch = u - z; old_stretch = before - old_z; dz = z - old_z
            residual += matrix @ stretch
            energy += .5 * stretch @ (matrix @ stretch)
            old_energy += .5 * old_stretch @ (matrix @ old_stretch)
            viscous += t / h * dz @ (matrix @ dz)
            increment = stretch - old_stretch
            numerical_dissipation += .5 * increment @ (matrix @ increment)
        # Use f=gravity+H*target and the quadratic H term; the omitted constant
        # target potential cancels on each fixed-target step.
        balance = external @ delta - (energy - old_energy) - viscous - numerical_dissipation
        max_residual = max(max_residual, float(np.max(np.abs(residual[free]), initial=0.)))
        max_balance = max(max_balance, abs(float(balance)))
        if max_residual > 1e-8 or max_balance > 1e-8:
            raise ValueError('CONTINUUM_NUMERICAL_RESIDUAL')
        dissipation += max(0., float(viscous))
        numerical_total += max(0., float(numerical_dissipation))
        for label, increments in exposure_increment(engine.scene, mesh, system['gradients'], u, memories, p['materials'], h).items():
            ledger = state['exposureBySourceLabel'].setdefault(label, {key: 0. for key in increments})
            for key, value in increments.items(): ledger[key] += value
    state['cornerDisplacementMm'] = _pack(mesh, u)
    state['memoryDisplacementMm'] = [_pack(mesh, z) for z in memories]
    state['dissipatedEnergyNmm'] = dissipation
    state['numericalDissipationNmm'] = numerical_total
    state.update(activeElements=len(ids), activeDofs=len(u), maxGradientNorm=max_gradient,
        maxFreeResidualN=max_residual, maxEnergyBalanceResidualNmm=max_balance,
        handleReactionsN={key: value.tolist() for key, value in reactions.items()},
        energyBalanceScope='Fixed-topology, fixed-target substeps; removed tissue energy exported separately; target/handle changes are not measured actuator work',
        boundary='Clamped artificial patch interface with retained inactive soft tissue; other faces traction-free; no CSF/ICP support')
    centers = u.reshape((-1, 3))[mesh.element_nodes].mean(axis=1)
    engine.state['deformation'] = {str(c): centers[i].tolist() for i, c in enumerate(mesh.cells)}
    # Velocity is the last fixed-step increment, matching the shared clock.
    last_centers = before.reshape((-1, 3))[mesh.element_nodes].mean(axis=1)
    model['velocities'] = {str(c): ((centers[i] - last_centers[i]) / h).tolist() for i, c in enumerate(mesh.cells)}
    tissue_energy = .5 * u @ (matrices[0] @ u)
    for matrix, z in zip(matrices[1:], memories): tissue_energy += .5 * (u-z) @ (matrix @ (u-z))
    model.update(elasticEnergyNmm=float(tissue_energy), activeNodes=len(ids), model=SCHEMA,
        maxNeighborDisplacementRatio=max((float(np.linalg.norm(centers[i] - centers[j]) / length)
            for i, j, length in system['neighbor_pairs']), default=0.))
    if diagnostic is not None:
        diagnostic['substeps'] += steps
        diagnostic['maxFreeResidualN'] = max(diagnostic['maxFreeResidualN'], max_residual)
        diagnostic['maxAbsDiscreteEnergyBalanceNmm'] = max(diagnostic['maxAbsDiscreteEnergyBalanceNmm'], max_balance)


def descriptor(engine):
    p = engine.scene.properties.get(PROPERTY)
    if p is None: return None
    return {'model': SCHEMA, 'element': 'source-voxel-owned full-integration hex8',
        'materialParameterUnits': 'Pa; N/mm and N mm internally; displacement mm',
        'profile': copy.deepcopy(p), 'summary': {key: copy.deepcopy(value) for key, value in engine.state.get(PROPERTY, {}).items()
            if key not in ('cornerDisplacementMm', 'memoryDisplacementMm')},
        'limits': ['Small gradients only; no finite-strain or clinical injury calibration.',
            'Moderate compressibility only; near-incompressible locking is not qualified.',
            'Bulk and shear use the same assumed relaxation spectrum; no independent volumetric relaxation measurement.',
            'Removed stored energy is exported, not converted to blood, heat or injury; handle/target changes have separate unmeasured work.',
            'Strain/stress volume-time integrals describe modeled exposure only; no injury threshold or deficit probability.',
            'No measured patient loads, CSF support, ICP or cavity-collapse validation.',
            'Source-center raster and existing collision guards remain approximations.']}
