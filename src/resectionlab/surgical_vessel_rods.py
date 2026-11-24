"""Source-bound vascular rods with compliant instrument contact.

An implicit damped position solve uses axial length and rotation-invariant
bending constraints. Geometry is a union of tapered capsules through shared
source sites. Parameters are engineering assignments in mm, N and seconds.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .surgical_glioma_materials import PROPERTY, oriented_source_cells
from . import surgical_vessel_surface as surfaces
from . import surgical_vessel_vertex_motion as vertex_motion
from . import surgical_wall_sweep as wall_sweep


MODEL = 'distributed-source-rods-v1'
FIELD = 'vesselMechanics'
GEOMETRY = 'vesselRodGeometry'
SECTION_MECHANICS = 'vesselSectionMechanics'
SECTION_MODEL = 'linear-annular-research-v1'
STEP_SECONDS = 1 / 120
SUPPORT_N_PER_MM = .035
CONTACT_N_PER_MM = .18
RADIAL_N_PER_MM = .25
DRAG_NS_PER_MM = .0025
MASS_N_S2_PER_MM = .000035
BENDING_RATIO = .24
MAX_COMPRESSION = .25
MIN_CENTERLINE_STRETCH = .7
SOLVER_ITERATIONS = 4
CONTACT_MARGIN_MM = 1e-7


def section_profile():
    """Assumed local tangent response and radial compliance for sensitivity studies."""
    return {
        'model': SECTION_MODEL,
        'provenance': 'assumed-research',
        'axialModulusMPa': {'artery': 1.8, 'vein': .7, 'uncertain': 1.},
        'wallThicknessToLumenRadius': .15,
        'radialRelaxationSeconds': .08,
        'frictionCoefficient': .15,
    }


def mechanics(engine):
    return engine.scene.properties.get(PROPERTY, {}).get(SECTION_MECHANICS)


def validate_section_profile(profile, bounded):
    if SECTION_MECHANICS not in profile:
        return
    section = profile[SECTION_MECHANICS]
    limits = {'wallThicknessToLumenRadius': (.05, .5), 'radialRelaxationSeconds': (.02, .5),
              'frictionCoefficient': (0., .5)}
    if (profile.get('vesselMotionModel') != MODEL or GEOMETRY not in profile
            or profile.get(wall_sweep.PROPERTY) != wall_sweep.POLICY
            or profile.get(vertex_motion.PROPERTY) != vertex_motion.POLICY
            or not isinstance(section, dict) or set(section) != set(section_profile())
            or section['model'] != SECTION_MODEL
            or section['provenance'] != 'assumed-research'
            or not isinstance(section['axialModulusMPa'], dict)
            or set(section['axialModulusMPa']) != set(section_profile()['axialModulusMPa'])
            or any(not bounded(value, .2, 6.) for value in section['axialModulusMPa'].values())
            or any(not bounded(section[key], *bounds) for key, bounds in limits.items())):
        raise ValueError('VESSEL_SECTION_MECHANICS_REQUIRES_BOUNDED_SWEPT_SOURCE_GEOMETRY')


def enabled(engine):
    return engine.scene.properties.get(PROPERTY, {}).get('vesselMotionModel') == MODEL


def wall_sweep_enabled(engine):
    return engine.scene.properties.get(PROPERTY, {}).get(wall_sweep.PROPERTY) == wall_sweep.POLICY


def wall_topology(engine):
    return wall_sweep.topology(_accepted_frames(engine, definitions(engine)))


def check_wall_topology(engine, previous=None):
    frames = _accepted_frames(engine, definitions(engine))
    current = wall_sweep.topology(frames)
    if current != previous:
        wall_sweep.require_clear(engine, frames, frames, static=True)
    return current


def validate_geometry(scene, profile, cells):
    if GEOMETRY not in profile:
        return
    geometry = profile[GEOMETRY]
    if (not isinstance(geometry, dict) or set(geometry) != {'source', 'sites'}
            or not isinstance(geometry['source'], str) or not geometry['source'].strip()
            or not isinstance(geometry['sites'], dict) or set(geometry['sites']) != {str(c) for c in cells}):
        raise ValueError('GLIOMA_ROD_GEOMETRY_REQUIRES_ALL_MAPPED_SITES')
    for cell in cells:
        point = geometry['sites'][str(cell)]
        if (not isinstance(point, list) or len(point) != 4
                or any(type(x) not in (float, int) or not np.isfinite(x) for x in point)
                or not .03 <= point[3] <= 8):
            raise ValueError('GLIOMA_ROD_GEOMETRY_REQUIRES_FINITE_POSITIVE_CALIBER')
        ijk = scene.inverse[:3, :3] @ point[:3] + scene.inverse[:3, 3]
        if np.max(np.abs(ijk - np.unravel_index(cell, scene.shape))) > .5000001:
            raise ValueError('GLIOMA_ROD_GEOMETRY_SITE_OUTSIDE_SOURCE_CELL')


@dataclass(frozen=True)
class Rods:
    cells: tuple
    index: dict
    rest: np.ndarray
    radius: np.ndarray
    bonds: tuple
    supports: tuple
    bending_rigidity: tuple | None


def _section_rigidity(radius, thickness_ratio, modulus):
    outer = radius * (1. + thickness_ratio)
    return (modulus * np.pi * (outer ** 2 - radius ** 2),
            modulus * np.pi / 4. * (outer ** 4 - radius ** 4))


def definitions(engine):
    cached = getattr(engine, '_vessel_rod_definitions', None)
    if cached is not None:
        return cached
    scene = engine.scene
    profile = scene.properties[PROPERTY]
    stiffness, area = profile['axialStiffnessN'], profile['effectiveLumenAreaMm2']
    bounds = {}
    bonds = []
    for edge in scene.properties['vascularGraphV1']['edges']:
        kind = edge['kind']
        if kind not in stiffness:
            continue
        chain = oriented_source_cells(edge)
        for cell in chain:
            bounds[cell] = max(bounds.get(cell, 0.), area[kind])
        bonds.extend((a, b, edge['id'], stiffness[kind]) for a, b in zip(chain, chain[1:]))
    cells = tuple(sorted(bounds))
    index = {cell: i for i, cell in enumerate(cells)}
    geometry = profile.get(GEOMETRY)
    if geometry is None:
        rest = scene.points(cells)
        radius = np.sqrt(np.array([bounds[cell] for cell in cells]) / np.pi)
    else:
        sites = np.array([geometry['sites'][str(cell)] for cell in cells])
        rest, radius = sites[:, :3], sites[:, 3]
    indexed = tuple((index[a], index[b], edge, k) for a, b, edge, k in bonds)
    bending_rigidity = None
    section = mechanics(engine)
    if section is not None:
        kinds = {edge['id']: edge['kind'] for edge in scene.properties['vascularGraphV1']['edges']}
        responses = [_section_rigidity((radius[a] + radius[b]) / 2., section['wallThicknessToLumenRadius'],
                                      section['axialModulusMPa'][kinds[edge]]) for a, b, edge, _ in indexed]
        bending_rigidity = tuple(ei for _, ei in responses)
        indexed = tuple((a, b, edge, ea) for (a, b, edge, _), (ea, _) in zip(indexed, responses))
    cached = Rods(cells, index, rest, radius, indexed,
                  tuple((index[a], b) for a, b in profile['vesselAttachments']), bending_rigidity)
    engine._vessel_rod_definitions = cached
    return cached


def _topology(engine, rods):
    removed = set(engine.state['removed'])
    blocked = {tuple(sorted(pair)) for name in ('cutEdges', 'transectedEdges')
               for pair in engine.state.get(name, ())}
    graph = engine.state.get('vascularGraphV1', {})
    blocked.update(tuple(sorted(row['cutFace'])) for row in graph.get('edges', {}).values() if row['cutFace'])
    retired = tuple(i for i, cell in enumerate(rods.cells) if cell in removed)
    retained = tuple(i for i, cell in enumerate(rods.cells) if cell not in removed)
    bonds = tuple(k for k, (a, b, _, _) in enumerate(rods.bonds)
                  if rods.cells[a] not in removed and rods.cells[b] not in removed
                  and tuple(sorted((rods.cells[a], rods.cells[b]))) not in blocked)
    supports = tuple((i, cell) for i, cell in rods.supports if rods.cells[i] not in removed and cell not in removed
                     and tuple(sorted((rods.cells[i], cell))) not in blocked)
    return retained, retired, bonds, supports


def _current(engine, rods):
    deformation = engine.state.get('deformation', {})
    compression = engine.state.get(FIELD, {}).get('compressionMm', {})
    position = rods.rest + np.array([deformation.get(str(c), (0., 0., 0.)) for c in rods.cells])
    radius = rods.radius - np.array([compression.get(str(c), 0.) for c in rods.cells])
    return position, radius


def source_points(engine, cells, *, rest=False):
    cells = tuple(cells)
    rods = definitions(engine)
    deformation = engine.state.get('deformation', {})
    points = {cell: rods.rest[rods.index[cell]] + (0. if rest else np.asarray(deformation.get(str(cell), (0., 0., 0.))))
              for cell in cells if cell in rods.index}
    if not rest and getattr(engine.scene, '_vessel_source_walls', ()):
        frames = _accepted_frames(engine, rods)
        for wall, offsets, _, _, _, edited in frames:
            vertex_owners = edited.owners if edited is not None else wall.source.owners
            for cell in cells:
                i = np.searchsorted(wall.source.cells, cell)
                if cell not in points and i < len(wall.source.cells) and wall.source.cells[i] == cell:
                    motion = offsets[i]
                    if wall.vertex_motion is not None:
                        samples = wall.vertex_motion.offsets[vertex_owners == i]
                        if len(samples):
                            motion = samples.mean(axis=0)
                    points[cell] = wall.positions[i] + motion
    return points


def _capsules(rods, retained, bonds):
    pairs = [(rods.bonds[k][0], rods.bonds[k][1]) for k in bonds]
    connected = {i for pair in pairs for i in pair}
    return np.array([*pairs, *((i, i) for i in retained if i not in connected)], dtype=int).reshape(-1, 2)


def _closest(a, b, start, end, r0, r1):
    """Minimize distance to the tool segment minus tapered radius.

    Tool projection has three linear pieces (line interior and two endpoint
    regions). Evaluate their stationary points and the interval boundaries.
    """
    direction = end - start
    tool_axis = b - a
    tool_len2 = float(tool_axis @ tool_axis)
    slope = r1 - r0
    candidates = [np.zeros(len(start)), np.ones(len(start))]

    def stationary(offset, delta):
        length2 = np.einsum('ij,ij->i', delta, delta)
        center = -np.einsum('ij,ij->i', offset, delta) / np.maximum(length2, 1e-30)
        orthogonal = np.linalg.norm(offset + center[:, None] * delta, axis=1)
        denominator = np.sqrt(np.maximum(length2 * (length2 - slope ** 2), 1e-30))
        return np.where(length2 > slope ** 2, center + slope * orthogonal / denominator, 0.)

    candidates.append(stationary(start - a, direction))
    if tool_len2 > 1e-20:
        along = (start - a) @ tool_axis / tool_len2
        rate = direction @ tool_axis / tool_len2
        safe_rate = np.where(np.abs(rate) > 1e-20, rate, 1.)
        candidates.extend((-along / safe_rate, (1 - along) / safe_rate,
                           stationary(start - b, direction),
                           stationary(start - a - along[:, None] * tool_axis,
                                      direction - rate[:, None] * tool_axis)))
    fractions = np.clip(candidates, 0., 1.)
    samples = start + fractions[:, :, None] * direction
    separation = samples - a
    if tool_len2 > 1e-20:
        tool_fraction = np.clip(separation @ tool_axis / tool_len2, 0., 1.)
        separation = separation - tool_fraction[:, :, None] * tool_axis
    distance = np.linalg.norm(separation, axis=-1) - (r0 + fractions * slope)
    best = distance.argmin(axis=0)
    rows = np.arange(len(start))
    return distance[best, rows], fractions[best, rows], separation[best, rows]


def _query(a, b, radius, pairs, position, calibre):
    a, b = np.asarray(a), np.asarray(b)
    first, second = pairs.T
    start, end = position[first], position[second]
    extent = (radius + np.maximum(calibre[first], calibre[second]) + CONTACT_MARGIN_MM)[:, None]
    plausible = np.flatnonzero(np.all(
        (np.minimum(start, end) - extent <= np.maximum(a, b))
        & (np.maximum(start, end) + extent >= np.minimum(a, b)), axis=1))
    if not len(plausible):
        return pairs[:0], np.empty(0), np.empty(0), np.empty((0, 3))
    i, j = first[plausible], second[plausible]
    clearance, fraction, delta = _closest(a, b, position[i], position[j], calibre[i], calibre[j])
    hit = clearance <= radius + CONTACT_MARGIN_MM
    return pairs[plausible[hit]], fraction[hit], radius - clearance[hit], delta[hit]


def contacts(engine, a, b, radius, raster):
    rods = definitions(engine)
    retained, _, bonds, _ = _topology(engine, rods)
    position, calibre = _current(engine, rods)
    frames = _accepted_frames(engine, rods, position, calibre, bonds)
    if not frames:
        hit, _, _, _ = _query(a, b, radius, _capsules(rods, retained, bonds), position, calibre)
        return sorted((set(raster) - rods.index.keys()) | {rods.cells[i] for i in hit.flat})
    labels = engine.scene.labels.ravel()
    wall_labels = {frame[0].source.label for frame in frames}
    result = {cell for cell in raster if cell not in rods.index and labels[cell] not in wall_labels}
    for frame in frames:
        wall = frame[0]
        owners, _, _ = surfaces.query(engine, frame, np.asarray(a), np.asarray(b), radius)
        # Keep source attribution distinct from the wider force stencil.
        linked, weights = surfaces.sites_for_owners(frame, owners)
        result.update(map(int, wall.source.cells[owners.ravel()]))
        result.update(rods.cells[i] for i in linked[weights > 1e-12] if i in retained)
    # A generated branch in another layer can still coexist with score-bound
    # source walls. Its capsule geometry remains its declared contact contract.
    pairs = _capsules(rods, retained, bonds)
    synthetic = np.array([labels[rods.cells[i]] not in wall_labels for i in pairs[:, 0]], dtype=bool)
    if synthetic.any():
        hit, _, _, _ = _query(a, b, radius, pairs[synthetic], position, calibre)
        result.update(rods.cells[i] for i in hit.flat)
    return sorted(result)


def _accepted_frames(engine, rods, position=None, radius=None, bonds=None):
    if not getattr(engine.scene, '_vessel_source_walls', ()):
        return ()
    if position is None:
        position, radius = _current(engine, rods)
    if bonds is None:
        _, _, bonds, _ = _topology(engine, rods)
    key = (position.tobytes(), radius.tobytes(), bonds, tuple(engine.state['removed']),
           tuple(tuple(pair) for name in ('cutEdges', 'transectedEdges') for pair in engine.state.get(name, ())),
           tuple(tuple(row['cutFace'] or ()) for row in engine.state.get('vascularGraphV1', {}).get('edges', {}).values()))
    cached = getattr(engine, '_vessel_accepted_walls', None)
    if cached is None or cached[0] != key:
        cached = (key, tuple(surfaces.frames(engine, rods, position, radius, bonds)))
        engine._vessel_accepted_walls = cached
    return cached[1]


def allows_tip_contact(engine, key, operation):
    """Allow external working-tip load in two explicitly non-cutting modes."""
    if not enabled(engine) or operation not in ('move', 'apply'):
        return False
    slot = engine.state['slots'][key]
    return slot['instrument'] == 'dissector' or (slot['instrument'] == 'bipolar_forceps'
            and not engine.parameters(slot)['power'])


def compliant_cells(engine, contacts):
    rods = definitions(engine)
    contact = set(contacts)
    accepted = contact.intersection(rods.index)
    if getattr(engine.scene, '_vessel_source_walls', ()):
        for wall in surfaces.definitions(engine, rods):
            accepted.update(contact.intersection(wall.source.cells[wall.influenced]))
    return accepted


def check_working_axis(engine, key, operation):
    """A compliant shell may overlap locally while its axis remains outside."""
    if not allows_tip_contact(engine, key, operation):
        return
    for hand, a, b, _, _ in _work_capsules(engine):
        if hand == key and compliant_cells(engine, engine.touched(a, b, 0.)):
            raise ValueError('VESSEL_COMPLIANT_CONTACT_LIMIT: working axis entered the retained vessel')


def cut_candidates(engine, cells, position, normal, depth):
    rods = definitions(engine)
    _, _, bonds, _ = _topology(engine, rods)
    points, _ = _current(engine, rods)
    closest = {}
    for k in bonds:
        i, j, branch, _ = rods.bonds[k]
        if engine.state.get('vascularGraphV1', {}).get('edges', {}).get(branch, {}).get('cutFace'):
            continue
        a, b = rods.cells[i], rods.cells[j]
        if a not in cells and b not in cells:
            continue
        ua, ub = points[[i, j]] - position
        side_a, side_b = ua @ normal, ub @ normal
        if min(side_a, side_b) <= 0 < max(side_a, side_b):
            crossing = ua - (ub - ua) * side_a / (side_b - side_a)
            distance = float(np.linalg.norm(crossing))
            if distance <= depth and (branch not in closest or distance < closest[branch][0]):
                closest[branch] = (distance, tuple(sorted((a, b))))
    return sorted(face for _, face in closest.values())


def _springs(rods, bonds, support_diagonal):
    # A row maps the global positions onto an edge vector or a difference
    # of normalized adjacent edge vectors. Its projection preserves rest length
    # or rest curvature magnitude under rigid translation and rotation.
    graph = [[] for _ in rods.cells]
    rows, weights = [], []
    for k in bonds:
        i, j, _, ea = rods.bonds[k]
        length = np.linalg.norm(rods.rest[j] - rods.rest[i])
        rows.append(((i, -1.), (j, 1.)))
        weights.append(ea / length)
        rigidity = None if rods.bending_rigidity is None else rods.bending_rigidity[k]
        graph[i].append((j, length, ea, rigidity))
        graph[j].append((i, length, ea, rigidity))
    for i, neighbors in enumerate(graph):
        for (j, a, ea, ei), (k, b, eb, ej) in combinations(neighbors, 2):
            rows.append(((j, 1 / a), (i, -1 / a - 1 / b), (k, 1 / b)))
            # Each half segment contributes compliance in series at the bend.
            weight = BENDING_RATIO * min(ea, eb) * min(a, b) if ei is None else 2. / (a / ei + b / ej)
            weights.append(weight / (len(neighbors) - 1))
    indices = [(r, i, coefficient) for r, row in enumerate(rows) for i, coefficient in row]
    row, column, data = zip(*indices) if indices else ((), (), ())
    operators = sparse.csr_matrix((data, (row, column)), shape=(len(rows), len(rods.cells)))
    transpose = operators.T @ sparse.diags(weights, shape=(len(rows), len(rows)))
    stiffness = transpose @ operators
    inertial = MASS_N_S2_PER_MM / STEP_SECONDS ** 2
    damping = DRAG_NS_PER_MM / STEP_SECONDS
    solve = factorized(sparse.diags(inertial + damping + support_diagonal).tocsc() + stiffness)
    return operators, transpose, operators @ rods.rest, solve


def _work_capsules(engine):
    from .surgical_motion import envelopes

    for hand, slot in engine.state['slots'].items():
        if not slot.get('inserted') or not allows_tip_contact(engine, hand, 'move'):
            continue
        params = engine.parameters(slot)
        for a, b, radius, part in envelopes(engine.tools[slot['instrument']], params, slot['position'], slot['quaternion']):
            if part != 'shaft':
                yield hand, a, b, radius, params['forceN']


def _native_tool_contact(engine, frames, rods, position, tools):
    force = np.zeros_like(position)
    compression = np.zeros(len(position))
    touched = {}
    for hand, a, b, calibre, max_force in tools:
        best = None
        for frame in frames:
            owners, distance, difference, vertices = surfaces.contact_features(engine, frame, a, b, calibre)
            if not len(owners):
                continue
            links, weights = surfaces.contact_sites(frame, owners, vertices)
            usable = np.flatnonzero(np.any(weights > 1e-12, axis=(1, 2)))
            if not len(usable):
                continue
            index = usable[np.argmin(distance[usable])]
            if best is None or distance[index] < best[0]:
                best = distance[index], difference[index], links[index], weights[index]
        if best is None:
            continue
        distance, direction, links, weights = best
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            # At an axis/wall intersection the nearest-point vector is zero.
            # Use the capsule-to-rod direction for the continuation.
            support = links[weights > 1e-12]
            center = position[support].mean(axis=0)
            direction = center - surfaces._projection(center, a, b)[0]
            norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        # The deepest accepted triangle is sufficient to define one net force
        # per instrument capsule regardless of surface tessellation density.
        active = weights > 1e-12
        indices = links[active]
        loads = min(max_force, max(0., calibre - distance) * CONTACT_N_PER_MM) * weights[active] / np.sum(active.any(axis=1))
        np.add.at(force, indices, loads[:, None] * (direction / norm))
        np.maximum.at(compression, indices, np.minimum(rods.radius[indices] * MAX_COMPRESSION,
                                                      loads / RADIAL_N_PER_MM))
        touched.setdefault(hand, set()).update(indices.tolist())
    return force, compression, touched


def _patch_quadrature(triangles, points):
    """Area and barycentric coordinates at the closest wall features."""
    x, y, z = triangles.transpose(1, 0, 2)
    u, v = y - x, z - x
    normal = np.cross(u, v)
    norm_squared = np.einsum('ij,ij->i', normal, normal)
    denominator = np.maximum(norm_squared, 1e-30)
    relative = points - x
    s = np.einsum('ij,ij->i', np.cross(relative, v), normal) / denominator
    t = np.einsum('ij,ij->i', np.cross(u, relative), normal) / denominator
    weights = np.clip(np.column_stack((1. - s - t, s, t)), 0., 1.)
    return np.sqrt(norm_squared) / 2., weights / weights.sum(axis=1)[:, None]


def _native_patch_contact(engine, frames, rods, position, velocity, tools, tool_motion, friction, measure_compression):
    """Resolve a finite wall patch using its accepted vertex displacement map."""
    force = np.zeros_like(position)
    radial_force = np.zeros(len(position))
    touched = {}
    for hand, a, b, calibre, limit in tools:
        patches = []
        for frame in frames:
            _, distance, difference, vertices, points, triangles = surfaces.contact_features(
                engine, frame, a, b, calibre, with_points=True)
            if not len(vertices):
                continue
            field = frame[0].vertex_motion.binding
            area, corners = _patch_quadrature(triangles, points)
            rows = np.arange(len(vertices)).repeat(3)
            quadrature = sparse.csr_matrix((corners.ravel(), (rows, vertices.ravel())),
                                           shape=(len(vertices), field.translation.shape[0]))
            weights = quadrature @ field.translation
            area = np.where(np.asarray(weights.sum(axis=1)).ravel() > 1e-12, area, 0.)
            if not area.any():
                continue
            center = (weights @ position) / np.maximum(np.asarray(weights.sum(axis=1)), 1e-30)
            # A centerline-to-tool direction is defined even at an axis/wall
            # crossing. The engine separately refuses final axis penetration.
            fallback = center - surfaces._projection(center, a, b)[0]
            direction = np.where((distance > 1e-10)[:, None], difference, fallback)
            norm = np.linalg.norm(direction, axis=1)
            area = np.where(norm > 1e-10, area, 0.)
            overlap = np.maximum(0., calibre - distance)
            patches.append((area, overlap, weights, quadrature, field, points,
                            direction / np.maximum(norm[:, None], 1e-30)))
        patch_load = sum(float(area @ overlap) for area, overlap, *_ in patches)
        if patch_load <= 1e-30:
            continue
        peak_overlap = max(float(overlap[area > 0.].max(initial=0.)) for area, overlap, *_ in patches)
        scale = min(limit, CONTACT_N_PER_MM * peak_overlap) / patch_load
        selected = touched.setdefault(hand, set())
        for area, overlap, weights, quadrature, field, points, normal in patches:
            normal_load = area * overlap * scale
            loads = normal_load[:, None] * normal
            if friction:
                relative = weights @ velocity
                if hand in tool_motion:
                    origin, linear, angular = tool_motion[hand]
                    relative -= linear + np.cross(angular, points - origin)
                slip = relative - np.einsum('ij,ij->i', relative, normal)[:, None] * normal
                # Smooth Coulomb bound at zero slip, in mm/s.
                loads -= (friction * normal_load / np.sqrt(1. + np.sum(slip ** 2, axis=1)))[:, None] * slip
            force += weights.T @ loads
            if measure_compression:
                selected.update(weights[area > 0.].indices.tolist())
                for k, radial in enumerate(field.radial):
                    radial_force -= (quadrature @ radial).T @ loads[:, k] / rods.radius
    compression = np.clip(radial_force / RADIAL_N_PER_MM, 0., rods.radius * MAX_COMPRESSION)
    return force, compression, touched


def _tool_contact(pairs, position, radius, rest_radius, tools):
    force = np.zeros_like(position)
    compression = np.zeros(len(position))
    touched = {}
    for hand, a, b, calibre, max_force in tools:
        hit, fractions, overlap, difference = _query(a, b, calibre, pairs, position, radius)
        if not len(hit):
            continue
        is_degenerate = np.linalg.norm(difference, axis=1) < 1e-10
        if np.any(is_degenerate):
            # Exact centerline coincidence has no unique normal. Choose a
            # deterministic perpendicular to the tool axis.
            axis = b - a
            unit = np.eye(3)[np.argmin(np.abs(axis))]
            lateral = np.cross(axis, unit)
            difference[is_degenerate] = lateral if np.any(lateral) else unit
        normals = difference / np.linalg.norm(difference, axis=1)[:, None]
        loads = np.maximum(overlap, 0.) * CONTACT_N_PER_MM
        loads *= min(1., max_force / max(loads.sum(), 1e-30))
        for index, weight in zip(hit.T, (1. - fractions, fractions)):
            weighted = weight * loads
            np.add.at(force, index, weighted[:, None] * normals)
            np.maximum.at(compression, index, np.minimum(rest_radius[index] * MAX_COMPRESSION,
                                                        weighted / RADIAL_N_PER_MM))
        touched.setdefault(hand, set()).update(hit.flat.tolist())
    return force, compression, touched


def _valid_motion(engine, rods, old, new, retained):
    if not retained:
        return new
    # Sample the increment so a free site cannot tunnel across a hard source
    # cell or leave the declared patch. The score-wall sweep is a separate check.
    points = np.asarray(retained)
    step = new[points] - old[points]
    if np.max(np.abs(step), initial=0.) < 1e-10:
        return new
    shape = np.array(engine.scene.shape)
    inv = engine.scene.inverse
    removed = set(engine.state['removed'])
    center = engine.scene.points([rods.cells[i] for i in retained])
    walls = surfaces.definitions(engine, rods)
    allowed = getattr(engine, '_vessel_movable_wall_cells', None)
    if allowed is None:
        allowed = frozenset(int(cell) for wall in walls for cell in wall.source.cells[wall.influenced])
        engine._vessel_movable_wall_cells = allowed
    for alpha in (.25, .5, .75, 1.):
        position = old[points] + alpha * step
        voxels = np.rint(position @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        display_voxels = np.rint((position - rods.rest[points] + center) @ inv[:3, :3].T + inv[:3, 3])
        if np.any((voxels < 0) | (voxels >= shape) | (display_voxels < 0) | (display_voxels >= shape)):
            return old
        sampled = np.ravel_multi_index(voxels.T, engine.scene.shape)
        labels = engine.scene.labels.flat
        if any(cell not in removed and cell not in rods.index and cell not in allowed and labels[cell]
               and engine.by_label[int(labels[cell])]['material'] in ('bone', 'unknown', 'vessel') for cell in sampled):
            return old
    return new


def tick(engine, dt, *, tool_motion=None):
    rods = definitions(engine)
    retained, retired, bonds, supports = _topology(engine, rods)
    state = engine.state.setdefault(FIELD, {'remainderSeconds': 0., 'velocitiesMmPerSecond': {}, 'compressionMm': {},
                                           'contactedCells': {}})
    total = state['remainderSeconds'] + dt
    steps = int(np.floor((total + 1e-12) / STEP_SECONDS))
    state['remainderSeconds'] = max(0., round(total - steps * STEP_SECONDS, 14))
    deformation = engine.state['deformation']
    velocities, compressed = state['velocitiesMmPerSecond'], state['compressionMm']
    for i in retired:
        cell = str(rods.cells[i])
        for field in (deformation, velocities, compressed):
            field.pop(cell, None)
    if not steps or not retained:
        return
    state['contactedCells'] = {}
    position, radius = _current(engine, rods)
    old_rate = np.array([velocities.get(str(c), (0., 0., 0.)) for c in rods.cells])
    extent = (np.max(rods.radius) + 2 * np.linalg.norm(engine.scene.spacing))
    lo, hi = np.min(position, axis=0) - extent, np.max(position, axis=0) + extent
    tools = tuple(tool for tool in _work_capsules(engine)
                  if np.all(np.minimum(tool[1], tool[2]) - tool[3] <= hi)
                  and np.all(np.maximum(tool[1], tool[2]) + tool[3] >= lo))
    if not tools and not velocities and not compressed and not any(str(c) in deformation for c in rods.cells) and not any(
            np.any(deformation.get(str(cell), (0., 0., 0.))) for _, cell in supports):
        return
    support_force = np.zeros_like(position)
    support_count = np.zeros(len(position))
    for i, cell in supports:
        support_force[i] += rods.rest[i] + np.asarray(deformation.get(str(cell), (0., 0., 0.)))
        support_count[i] += 1
    support_force *= SUPPORT_N_PER_MM
    spring_key = (bonds, tuple(support_count))
    cached = getattr(engine, '_vessel_rod_springs', None)
    if cached is None or cached[0] != spring_key:
        cached = (spring_key, *_springs(rods, bonds, support_count * SUPPORT_N_PER_MM))
        engine._vessel_rod_springs = cached
    _, operators, transpose, rest_projection, solve = cached
    norms = np.linalg.norm(rest_projection, axis=1)[:, None]
    pairs = _capsules(rods, retained, bonds)
    native = bool(getattr(engine.scene, '_vessel_source_walls', ()))
    section = mechanics(engine)
    radial_response = .35 if section is None else -np.expm1(-STEP_SECONDS / section['radialRelaxationSeconds'])
    if native:
        labels = engine.scene.labels.ravel()
        registered = {wall.label for wall in engine.scene._vessel_source_walls}
        capsule_pairs = np.array([labels[rods.cells[i]] not in registered for i in pairs[:, 0]], dtype=bool)
        pairs = pairs[capsule_pairs]
    if section is not None:
        first, second = np.asarray([rods.bonds[k][:2] for k in bonds], dtype=int).reshape(-1, 2).T
        reference_length = np.linalg.norm(rods.rest[first] - rods.rest[second], axis=1)
        site_length = (np.bincount(first, reference_length, minlength=len(position))
                       + np.bincount(second, reference_length, minlength=len(position)))

    def contact(positions, radii, rate, *, measure_compression=False):
        force, compression, touched = _tool_contact(pairs, positions, radii, rods.radius, tools)
        if native and tools:
            frames = tuple(surfaces.frames(engine, rods, positions, radii, bonds))
            if section is None:
                load, collapse, contacts = _native_tool_contact(engine, frames, rods, positions, tools)
            else:
                load, collapse, contacts = _native_patch_contact(engine, frames, rods, positions, rate, tools,
                                                                tool_motion or {}, section['frictionCoefficient'],
                                                                measure_compression)
            force += load
            compression = np.maximum(compression, collapse)
            for hand, indices in contacts.items():
                touched.setdefault(hand, set()).update(indices)
        return force, compression, touched

    inertial = MASS_N_S2_PER_MM / STEP_SECONDS ** 2
    damping = DRAG_NS_PER_MM / STEP_SECONDS
    swept = wall_sweep_enabled(engine)
    if swept:
        prior_wall = tuple(surfaces.frames(engine, rods, position, radius, bonds))
    for _ in range(steps):
        previous = position.copy()
        base = inertial * (previous + STEP_SECONDS * old_rate) + damping * previous + support_force
        for _ in range(SOLVER_ITERATIONS):
            project = operators @ position
            scale = np.linalg.norm(project, axis=1)[:, None]
            target = np.where(scale > 1e-15, project * norms / np.maximum(scale, 1e-15), rest_projection)
            force, _, _ = contact(position, radius, old_rate)
            position = solve(base + transpose @ target + force)
        if section is None:
            position = _valid_motion(engine, rods, previous, position, retained)
        rate = (position - previous) / STEP_SECONDS
        _, compression_target, touched = contact(position, radius, rate, measure_compression=True)
        if section is not None:
            lengths = np.linalg.norm(position[first] - position[second], axis=1)
            # A positive axial strain draws down the effective lumen area in
            # this local near-volume-preserving approximation.
            extension = np.maximum(0., lengths - reference_length)
            axial_extension = (np.bincount(first, extension, minlength=len(position))
                               + np.bincount(second, extension, minlength=len(position)))
            stretch = 1. + axial_extension / np.maximum(site_length, 1e-30)
            compression_target += rods.radius * (1. - 1. / np.sqrt(stretch))
            compression_target = np.minimum(compression_target, rods.radius * MAX_COMPRESSION)
        compression = (1. - radial_response) * (rods.radius - radius) + radial_response * compression_target
        proposed_radius = rods.radius - compression
        if section is not None:
            # Preserve the first obstacle- and sweep-certified fraction of
            # the increment. An uncertifiable motion remains transactional.
            delta, radial_delta = position - previous, proposed_radius - radius
            for fraction in (1., .5, .25, .125, .0625, .03125, .015625):
                trial = previous + fraction * delta
                trial_radius = radius + fraction * radial_delta
                if (np.any(np.linalg.norm(trial[first] - trial[second], axis=1) < MIN_CENTERLINE_STRETCH * reference_length)
                        or _valid_motion(engine, rods, previous, trial, retained) is previous):
                    continue
                next_wall = tuple(surfaces.frames(engine, rods, trial, trial_radius, bonds))
                if wall_sweep.is_clear(engine, prior_wall, next_wall):
                    position, proposed_radius, prior_wall = trial, trial_radius, next_wall
                    if fraction < 1.:
                        _, _, touched = contact(position, proposed_radius, (position - previous) / STEP_SECONDS,
                                                measure_compression=True)
                    break
            else:
                raise ValueError('VESSEL_WALL_SWEEP_UNSAFE: no limited section response retained clearance')
        elif swept:
            next_wall = tuple(surfaces.frames(engine, rods, position, proposed_radius, bonds))
            wall_sweep.require_clear(engine, prior_wall, next_wall)
            prior_wall = next_wall
        radius = proposed_radius
        old_rate = (position - previous) / STEP_SECONDS
        for hand, indices in touched.items():
            state['contactedCells'][hand] = sorted(set(state['contactedCells'].get(hand, ())) |
                                                   {rods.cells[i] for i in indices})
    delta = position - rods.rest
    for i in retained:
        key = str(rods.cells[i])
        for field, value in ((deformation, delta[i]), (velocities, old_rate[i]), (compressed, rods.radius[i] - radius[i])):
            if np.max(np.abs(value)) > 1e-10:
                field[key] = np.asarray(value).tolist()
            else:
                field.pop(key, None)


def view(engine):
    rods = definitions(engine)
    retained, _, bonds, supports = _topology(engine, rods)
    position, radius = _current(engine, rods)
    support_by_site = {}
    for i, cell in supports:
        support_by_site.setdefault(i, []).append(cell)
    state = engine.state.get(FIELD, {})
    frames = _accepted_frames(engine, rods, position, radius, bonds)
    return {
        'model': MODEL,
        **({'sectionModel': SECTION_MODEL} if mechanics(engine) is not None else {}),
        'sites': [{'sourceCell': rods.cells[i], 'restPositionMm': rods.rest[i].tolist(),
                   'positionMm': position[i].tolist(), 'restRadiusMm': float(rods.radius[i]), 'radiusMm': float(radius[i]),
                   'activeSupportCells': support_by_site.get(i, [])} for i in retained],
        'segments': [{'sourceCells': [rods.cells[i], rods.cells[j]], 'edgeId': edge}
                     for i, j, edge, _ in (rods.bonds[k] for k in bonds)],
        'contactedCells': {hand: cells[:] for hand, cells in state.get('contactedCells', {}).items()},
        **({'surfaceDisplacements': surfaces.displacements(frames), **surfaces.geometry_metadata(frames)} if frames else {}),
    }
