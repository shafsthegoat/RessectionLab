"""Native TOF wall contact and transport for the distributed rod model."""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .core import immutable_array, semantic_digest
from .geometry import _segment_cell_distances
from . import surgical_vessel_vertex_motion as vertex_kinematics


GEOMETRY_PROPERTY = 'operative_vascular_geometry'
CONDITIONED_GEOMETRY = 'source-conditioned-v1'
ISOLATED_SOURCE_GEOMETRY = 'source-conditioned-v2'
CONDITIONED_GEOMETRIES = (CONDITIONED_GEOMETRY, ISOLATED_SOURCE_GEOMETRY)


@dataclass(frozen=True)
class SourceWall:
    label: int
    cells: np.ndarray
    components: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    owners: np.ndarray
    scores: np.ndarray
    geometry_binding: str | None = None
    positions_ijk: np.ndarray | None = None
    geometry_policy: str | None = None


def source_cell_radius(scene):
    corners = np.array(list(product((-.5, .5), repeat=3))) @ scene.affine[:3, :3].T
    return float(np.linalg.norm(corners, axis=1).max())


def _wall_geometry(scores, mask, affine, policy, *, native_mask=None, cancelled=None):
    from .anatomy_geometry import vascular_mesh_source_cells, vascular_probability_mesh

    if policy is None:
        raw = vascular_probability_mesh(scores, mask)
        return None if raw is None else (*raw, vascular_mesh_source_cells(raw[0], mask))
    if policy not in CONDITIONED_GEOMETRIES:
        raise ValueError('VASCULAR_WALL_REQUIRES_SUPPORTED_GEOMETRY_POLICY')
    from .anatomy_geometry import conditioned_vascular_probability_mesh

    mesh = conditioned_vascular_probability_mesh(scores, mask, affine, native_mask=native_mask,
                                                 retain_isolated_components=policy == ISOLATED_SOURCE_GEOMETRY,
                                                 **({'cancelled': cancelled} if cancelled is not None else {}))
    return None if mesh is None else (mesh.positions, mesh.indices, mesh.normals, mesh.source_cells)


def case_wall_geometry(case, name, *, cancelled=None):
    """Canonical contour and source owner indices for contact and rendering."""
    return _wall_geometry(case.vascular_probabilities[name], case.compartments[name], case.affine,
                          case.metadata.get(GEOMETRY_PROPERTY), cancelled=cancelled)


def bind_case(scene, case):
    """Use the same extraction and source-owner map carried by the desktop."""
    walls = []
    for region in scene.regions:
        name = region['name']
        if region['material'] != 'vessel' or name not in case.vascular_probabilities:
            continue
        mask = case.compartments[name]
        if np.any(scene.labels[mask] != region['id']):
            raise ValueError('VASCULAR_WALL_REQUIRES_EXACT_NATIVE_MATERIAL_OWNERSHIP')
        result = case_wall_geometry(case, name)
        if result is None:
            raise ValueError('VASCULAR_WALL_REQUIRES_VALID_SOURCE_PROBABILITIES')
        ijk, triangles, _, owners = result
        cells = np.flatnonzero(mask)
        components = ndimage.label(mask)[0].flat[cells]
        walls.append(SourceWall(region['id'], immutable_array(cells), immutable_array(components),
                                immutable_array(ijk @ scene.affine[:3, :3].T + scene.affine[:3, 3]),
                                immutable_array(triangles), immutable_array(np.searchsorted(cells, owners)),
                                case.vascular_probabilities[name],
                                vertex_kinematics.geometry_binding(ijk, triangles, owners), immutable_array(ijk),
                                case.metadata.get(GEOMETRY_PROPERTY)))
    scene._vessel_source_walls = tuple(walls)
    scene._vessel_source_wall_proof = (scene.identity, scene._vessel_source_walls)


def require_binding(scene):
    proof = getattr(scene, '_vessel_source_wall_proof', None)
    walls = getattr(scene, '_vessel_source_walls', None)
    if proof is None and walls is None:
        return
    if proof is None or proof[0] != scene.identity or proof[1] is not walls:
        raise ValueError('Native vessel contact source binding changed')


def propagate_binding(source, target):
    require_binding(source)
    if (source.source.get('caseHash') != target.source.get('caseHash') or source.shape != target.shape
            or not np.array_equal(source.affine, target.affine) or not np.array_equal(source.labels, target.labels)):
        raise ValueError('Native vessel contact source requires an unchanged affine and source owners')
    target._vessel_source_walls = source._vessel_source_walls
    target._vessel_source_wall_proof = target.identity, target._vessel_source_walls


def _projection(points, start, end):
    vector = end - start
    length2 = np.sum(vector * vector, axis=-1)
    fraction = np.clip(np.sum((points - start) * vector, axis=-1) / np.maximum(length2, 1e-30), 0, 1)
    return start + fraction[..., None] * vector, fraction


@dataclass(frozen=True)
class Wall:
    source: SourceWall
    shape: tuple
    positions: np.ndarray
    triangles: np.ndarray
    tree: cKDTree
    triangle_extent: float
    cell_tree: cKDTree
    cell_extent: float
    influenced: np.ndarray
    links: np.ndarray
    fractions: np.ndarray
    weights: np.ndarray
    radial_offsets: np.ndarray
    vertex_motion: vertex_kinematics.VertexMotion | None = None


@dataclass(frozen=True)
class EditedWall:
    identity: str
    positions_ijk: np.ndarray
    triangles: np.ndarray
    owners: np.ndarray
    world_triangles: np.ndarray
    tree: cKDTree
    extent: float
    cuboids: np.ndarray


def definitions(engine, rods):
    cached = getattr(engine, '_vessel_wall_definitions', None)
    if cached is not None:
        return cached
    sources = getattr(engine.scene, '_vessel_source_walls', ())
    walls = []
    for source in sources:
        points = engine.scene.points(source.cells)
        triangles = source.vertices[source.triangles]
        centers = triangles.mean(axis=1)
        # Fit the motion once in rest geometry. Membership is limited to the
        # same face-connected foreground as the declaring centerline.
        bonds = np.array([(a, b) for a, b, _, _ in rods.bonds], dtype=int)
        midpoints = rods.rest[bonds].mean(axis=1)
        near = cKDTree(midpoints).query(points, k=list(range(1, min(12, len(bonds)) + 1)))[1]
        segments = bonds[near]
        foot, fraction = _projection(points[:, None, :], rods.rest[segments[:, :, 0]], rods.rest[segments[:, :, 1]])
        radii = (1 - fraction) * rods.radius[segments[:, :, 0]] + fraction * rods.radius[segments[:, :, 1]]
        d = np.linalg.norm(points[:, None] - foot, axis=2)
        components = np.zeros(len(rods.cells), dtype=np.int64)
        owns = engine.scene.labels.ravel()[list(rods.cells)] == source.label
        components[owns] = source.components[np.searchsorted(source.cells, np.array(rods.cells)[owns])]
        allowed = np.all(components[segments] == source.components[:, None, None], axis=2)
        closest = np.argmin(np.where(allowed, d, np.inf), axis=1)
        rows = np.arange(len(points))
        cell_extent = source_cell_radius(engine.scene)
        reference_margin = 1.3 * cell_extent
        weight = np.clip(1 - (d[rows, closest] - radii[rows, closest] - reference_margin) / reference_margin, 0, 1)
        influenced = np.flatnonzero(allowed[rows, closest] & (weight > 0))
        walls.append(Wall(
            source, engine.scene.shape, points, triangles, cKDTree(centers),
            float(np.linalg.norm(triangles - centers[:, None], axis=2).max()), cKDTree(points),
            cell_extent, influenced,
            near[rows, closest][influenced], fraction[rows, closest][influenced], weight[influenced],
            points[influenced] - foot[rows, closest][influenced],
        ))
    cached = tuple(walls)
    engine._vessel_wall_definitions = cached
    return cached


def _offsets(wall, rods, position, radius, live_bonds):
    offsets = np.zeros_like(wall.positions)
    links = np.array([(rods.bonds[k][0], rods.bonds[k][1]) for k in wall.links], dtype=int).reshape(-1, 2)
    # Edited faces belong to their nearest surviving side. Native walls fall
    # back to individual source-cell surfaces at a local ownership edit.
    fraction = np.where(np.isin(wall.links, live_bonds), wall.fractions, np.rint(wall.fractions))
    delta = position - rods.rest
    scale = 1. - radius / rods.radius
    blend = (1 - fraction)[:, None] * delta[links[:, 0]] + fraction[:, None] * delta[links[:, 1]]
    radial = (1 - fraction) * scale[links[:, 0]] + fraction * scale[links[:, 1]]
    offsets[wall.influenced] = wall.weights[:, None] * (blend - radial[:, None] * wall.radial_offsets)
    return offsets, links, fraction


def _candidates(tree, a, b, radius, extent):
    length = np.linalg.norm(b - a)
    # Cover the query segment with spheres. The hypotenuse accounts for
    # every point between consecutive sample centers.
    n = max(1, int(np.ceil(length / max(.5, radius + extent))))
    if n == 1 and length <= 1e-12:
        return np.array(tree.query_ball_point(a, radius + extent + 1e-7), dtype=int)
    samples = a + np.linspace(0., 1., n + 1)[:, None] * (b - a)
    hits = tree.query_ball_point(samples, np.hypot(radius + extent, length / (2 * n)) + 1e-7)
    return np.array(sorted({i for row in hits for i in row}), dtype=int)


def _triangle_distances(a, b, triangles, *, with_points=False):
    """Closest triangle-to-segment vectors in millimeters."""
    first, second, third = triangles.transpose((1, 0, 2))
    u, v = second - first, third - first
    normal = np.cross(u, v)
    normal2 = np.sum(normal * normal, axis=1)
    uv = np.sum(u * v, axis=1)
    uu, vv = np.sum(u * u, axis=1), np.sum(v * v, axis=1)
    det = np.maximum(uu * vv - uv * uv, 1e-30)
    axis = b - a

    def inside(points):
        rel = points - first
        pu, pv = np.sum(rel * u, axis=1), np.sum(rel * v, axis=1)
        s = (pu * vv - pv * uv) / det
        t = (pv * uu - pu * uv) / det
        return (s >= -1e-10) & (t >= -1e-10) & (s + t <= 1 + 1e-10) & (normal2 > 1e-30)

    differences, valid, closest = [], [], []
    for point in (a, b):
        distance = np.sum((point - first) * normal, axis=1) / np.maximum(normal2, 1e-30)
        delta = -distance[:, None] * normal
        differences.append(delta)
        valid.append(inside(point + delta))
        if with_points:
            closest.append(point + delta)
    denominator = normal @ axis
    fraction = np.sum((first - a) * normal, axis=1) / np.where(np.abs(denominator) > 1e-20, denominator, 1.)
    differences.append(np.zeros_like(first))
    valid.append((np.abs(denominator) > 1e-20) & (fraction >= 0) & (fraction <= 1)
                 & inside(a + fraction[:, None] * axis))
    if with_points:
        closest.append(a + fraction[:, None] * axis)

    for start, end in ((first, second), (second, third), (third, first)):
        # Endpoint projections and the unconstrained line-to-line solution.
        from_a, from_b = _projection(a, start, end)[0], _projection(b, start, end)[0]
        differences.extend((from_a - a, from_b - b,
                            start - _projection(start, a, b)[0], end - _projection(end, a, b)[0]))
        valid.extend([np.ones(len(first), dtype=bool)] * 4)
        edge = end - start
        edge2, axis2 = np.sum(edge * edge, axis=1), axis @ axis
        dot = edge @ axis
        det = edge2 * axis2 - dot ** 2
        from_edge = start - a
        edge_from = np.sum(edge * from_edge, axis=1)
        axis_from = from_edge @ axis
        safe = np.maximum(det, 1e-30)
        s = (dot * axis_from - axis2 * edge_from) / safe
        t = (edge2 * axis_from - dot * edge_from) / safe
        differences.append(from_edge + s[:, None] * edge - t[:, None] * axis)
        valid.append((det > 1e-20) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1))
        if with_points:
            closest.extend((from_a, from_b, start, end, start + s[:, None] * edge))
    differences = np.asarray(differences)
    distances = np.where(valid, np.linalg.norm(differences, axis=2), np.inf)
    rows = np.arange(len(first))
    k = distances.argmin(axis=0)
    if with_points:
        return distances[k, rows], differences[k, rows], np.asarray(closest)[k, rows]
    return distances[k, rows], differences[k, rows]


def _cut_patch(vertices, triangles, center, axis, reach, half_gap):
    signed = (vertices - center) @ axis
    radial = np.linalg.norm(vertices - center - signed[:, None] * axis, axis=1)
    band = np.flatnonzero((signed[triangles].max(axis=1) >= -half_gap)
                          & (signed[triangles].min(axis=1) <= half_gap)
                          & (radial[triangles].min(axis=1) <= reach))
    if not len(band):
        return
    near = {}
    for i in band:
        for cell in triangles[i]:
            near.setdefault(int(cell), []).append(int(i))
    unseen = set(map(int, band))
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component, queue = [seed], [seed]
        while queue:
            i = queue.pop()
            for vertex in triangles[i]:
                for neighbor in near[vertex]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
        yield sorted(component), signed


def _concave_cap(points, ring, radial, axis):
    """Triangulate a closed cut ring that surrounds the mapped vessel axis."""
    following = np.roll(radial, -1, axis=0)
    turn = np.cross(radial, following) @ axis
    total_turn = np.arctan2(turn, np.einsum('ij,ij->i', radial, following)).sum()
    if not np.isclose(abs(total_turn), 2. * np.pi, atol=1e-6):
        return None
    # Reverse the boundary so cap edges oppose the exterior face winding.
    remaining = ring[::-1]
    winding = -np.sign(total_turn)
    points = np.asarray(points)
    cap = []
    while len(remaining) > 3:
        current = points[remaining]
        before = np.roll(current, 1, axis=0)
        after = np.roll(current, -1, axis=0)
        area = winding * (np.cross(current - before, after - before) @ axis)
        for at in np.argsort(-area, kind='stable'):
            if area[at] <= 1e-12:
                return None
            a, b, c = before[at], current[at], after[at]
            other = np.ones(len(current), dtype=bool)
            other[[(at - 1) % len(current), at, (at + 1) % len(current)]] = False
            p = current[other]
            if np.any(((winding * (np.cross(b - a, p - a) @ axis)) >= -1e-12)
                      & ((winding * (np.cross(c - b, p - b) @ axis)) >= -1e-12)
                      & ((winding * (np.cross(a - c, p - c) @ axis)) >= -1e-12)):
                continue
            cap.append((remaining[at - 1], remaining[at], remaining[(at + 1) % len(remaining)]))
            del remaining[at]
            break
        else:
            return None
    a, b, c = points[remaining]
    if winding * (np.cross(b - a, c - a) @ axis) <= 1e-12:
        return None
    return [*cap, tuple(remaining)]


def _closed_cut(vertices, owners, triangles, center, axis, first, second, half_gap, reach, *, concave=False):
    """Open a narrow planar gap only when both level contours close around the trace."""
    for patch, distances in _cut_patch(vertices, triangles, center, axis, reach, half_gap):
        points = vertices.tolist()
        bindings = owners.tolist()
        replacement = []
        valid = True
        for side, stump in ((-1, first), (1, second)):
            cache = {}
            border = []
            plane = center + side * half_gap * axis
            signed = side * distances - half_gap
            if np.any(np.abs(signed[triangles[patch]]) < 1e-9):
                valid = False
                break
            for row in triangles[patch]:
                polygon = []
                seam = set()
                for start, end in zip(row, np.roll(row, -1)):
                    if signed[start] > 0:
                        polygon.append(int(start))
                    if (signed[start] > 0) != (signed[end] > 0):
                        edge = tuple(sorted((int(start), int(end))))
                        vertex = cache.get(edge)
                        if vertex is None:
                            fraction = signed[start] / (signed[start] - signed[end])
                            vertex = len(points)
                            points.append((vertices[start] + fraction * (vertices[end] - vertices[start])).tolist())
                            bindings.append(int(owners[start if signed[start] > 0 else end]))
                            cache[edge] = vertex
                        polygon.append(vertex)
                        seam.add(vertex)
                for i in range(1, len(polygon) - 1):
                    replacement.append((polygon[0], polygon[i], polygon[i + 1]))
                border.extend((a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1]) if a in seam and b in seam)
            links = dict(border)
            if len(border) < 3 or len(links) != len(border) or set(links) != set(links.values()):
                valid = False
                break
            ring = [min(links)]
            while links[ring[-1]] != ring[0] and len(ring) < len(links):
                ring.append(links[ring[-1]])
            if len(set(ring)) != len(links) or links[ring[-1]] != ring[0]:
                valid = False
                break
            radial = np.asarray(points)[ring] - plane
            winding = np.cross(radial, np.roll(radial, -1, axis=0)) @ axis
            if not (np.all(winding > 1e-10) or np.all(winding < -1e-10)):
                cap_faces = _concave_cap(points, ring, radial, axis) if concave else None
                if cap_faces is None:
                    valid = False
                    break
                replacement.extend(cap_faces)
                continue
            cap = len(points)
            points.append(plane.tolist())
            bindings.append(int(stump))
            replacement.extend((b, a, cap) for a, b in border)
        if valid:
            surviving = triangles[~np.isin(np.arange(len(triangles)), patch)]
            faces = np.vstack((surviving, replacement))
            used, inverse = np.unique(faces, return_inverse=True)
            return np.asarray(points)[used], inverse.reshape(-1, 3), np.asarray(bindings)[used]
    return None


def _edit_wall(engine, rods, wall, removed, cuts):
    scene = engine.scene
    source = wall.source
    label = source.label
    owned = scene.labels.ravel()
    retired = tuple(sorted(c for c in removed if owned[c] == label))
    faces_to_cut = tuple(sorted({tuple(sorted(pair)) for pair in cuts
                                if all(owned[c] == label and c not in removed for c in pair)}))
    if not retired and not faces_to_cut:
        return None
    key = label, retired, faces_to_cut
    cache = getattr(engine, '_vessel_edited_walls', {})
    if key in cache:
        return cache[key]
    vertices, faces, owners = source.vertices, source.triangles, source.owners
    if retired:
        mask = np.zeros(scene.shape, dtype=bool)
        mask.flat[source.cells] = True
        native = mask.copy() if source.geometry_policy == ISOLATED_SOURCE_GEOMETRY else None
        mask.flat[list(retired)] = False
        if mask.any():
            geometry = _wall_geometry(source.scores, mask, scene.affine, source.geometry_policy, native_mask=native)
            if geometry is None:
                raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: retained score-wall contour is unavailable')
            ijk, faces, _, source_cells = geometry
            vertices = ijk @ scene.affine[:3, :3].T + scene.affine[:3, 3]
            owners = np.searchsorted(source.cells, source_cells)
        else:
            vertices, faces, owners = np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty(0, dtype=int)
    bonds = {tuple(sorted((rods.cells[i], rods.cells[j]))): (i, j) for i, j, _, _ in rods.bonds}
    fallback = []
    for face in faces_to_cut:
        if face not in bonds:
            fallback.extend(face)
            continue
        i, j = bonds[face]
        points = rods.rest[[i, j]]
        axis = points[1] - points[0]
        half_gap = min(.15 * min(scene.spacing), .2 * np.linalg.norm(axis))
        axis /= np.linalg.norm(axis)
        reach = 1.65 * np.max(rods.radius[[i, j]]) + wall.cell_extent
        row_a, row_b = np.searchsorted(source.cells, [rods.cells[i], rods.cells[j]])
        conditioned = source.geometry_policy in CONDITIONED_GEOMETRIES
        # A conditioned crossing can graze a plane at the original midpoint.
        # Move the clipping bands within the kerf so the declared point of
        # transection remains inside the gap and both rings close.
        offsets = (0., -.75, .75) if conditioned else (0.,)
        for offset in offsets:
            center = points.mean(axis=0) + offset * half_gap * axis
            cut = _closed_cut(vertices, owners, faces, center, axis, row_a, row_b, half_gap,
                              reach, concave=conditioned)
            if cut is not None:
                break
        if cut is None:
            fallback.extend(face)
        else:
            vertices, faces, owners = cut
    cuboids = np.empty(0, dtype=int)
    if fallback:
        groups = set(source.components[np.searchsorted(source.cells, fallback)])
        triangle_groups = source.components[owners[faces]]
        # A scalar isosurface can span adjacent foreground components. Keep
        # every joined component together so fallback cannot leave open seams.
        while True:
            affected = np.isin(triangle_groups, list(groups)).any(axis=1)
            expanded = groups.union(triangle_groups[affected].flat)
            if expanded == groups:
                break
            groups = expanded
        faces = faces[~affected]
        used, inverse = np.unique(faces, return_inverse=True)
        vertices, faces, owners = vertices[used], inverse.reshape(-1, 3), owners[used]
        cuboids = np.flatnonzero(np.isin(source.components, list(groups)) & ~np.isin(source.cells, retired))

    # Use this float32 IJK packet for both owners. The conversion is done
    # before contact geometry is built so rest vertices agree end to end.
    ijk = (vertices @ scene.inverse[:3, :3].T + scene.inverse[:3, 3]).astype(np.float32)
    world = ijk @ scene.affine[:3, :3].T + scene.affine[:3, 3]
    world_faces = world[faces]
    centers = world_faces.mean(axis=1)
    edited = EditedWall(semantic_digest([scene.identity, key]), ijk, faces, owners, world_faces,
                        cKDTree(centers), float(np.linalg.norm(world_faces - centers[:, None], axis=2).max(initial=0.)),
                        cuboids)
    engine._vessel_edited_walls = {**{old: entry for old, entry in cache.items() if old[0] != label}, key: edited}
    return edited


def _vertex_field(engine, wall, rods, edited, live_bonds):
    label = wall.source.label
    key = (edited.identity if edited is not None else None, live_bonds)
    cache = getattr(engine, '_vessel_vertex_fields', {})
    current = cache.get(label)
    if current is None or current[0] != key:
        if edited is None:
            points, faces, owners = wall.source.vertices, wall.source.triangles, wall.source.owners
            ijk = wall.source.positions_ijk
            identity = wall.source.geometry_binding
        else:
            ijk = edited.positions_ijk
            points = ijk @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
            faces, owners, identity = edited.triangles, edited.owners, edited.identity
        if not identity or ijk is None:
            raise ValueError('VESSEL_VERTEX_TRANSPORT_REQUIRES_BOUND_SOURCE_GEOMETRY')
        # The applied transport at acquisition boundaries and newly clipped
        # cut vertices keeps its own stencil rather than borrowing nearby loads.
        protected = np.any((ijk < 1.) | (ijk > np.asarray(wall.shape) - 2.), axis=1)
        if edited is not None:
            if wall.source.positions_ijk is None:
                raise ValueError('VESSEL_VERTEX_TRANSPORT_REQUIRES_BOUND_SOURCE_GEOMETRY')
            row_bytes = np.dtype('V12')
            intact = np.ascontiguousarray(wall.source.positions_ijk, dtype='<f4').view(row_bytes).ravel()
            edited_rows = np.ascontiguousarray(ijk, dtype='<f4').view(row_bytes).ravel()
            protected |= ~np.isin(edited_rows, intact)
        field = vertex_kinematics.bind(wall, rods, points, faces, owners, live_bonds,
                                      identity, float(min(engine.scene.spacing)), protected)
        current = key, field
        engine._vessel_vertex_fields = {**cache, label: current}
    return current[1]


def frames(engine, rods, position, radius, live_bonds):
    removed = set(engine.state['removed'])
    blocked = [pair for name in ('cutEdges', 'transectedEdges') for pair in engine.state.get(name, ())]
    blocked.extend(row['cutFace'] for row in engine.state.get('vascularGraphV1', {}).get('edges', {}).values()
                   if row['cutFace'])
    vertex_transport = engine.scene.properties.get('gliomaMaterialsV1', {}).get(vertex_kinematics.PROPERTY) == vertex_kinematics.POLICY
    for wall in definitions(engine, rods):
        offsets, links, fraction = _offsets(wall, rods, position, radius, live_bonds)
        extent = float(np.linalg.norm(offsets, axis=1).max())
        edited = _edit_wall(engine, rods, wall, removed, blocked) if removed or blocked else None
        if vertex_transport:
            field = _vertex_field(engine, wall, rods, edited, live_bonds)
            shifts = field.offsets(position - rods.rest, 1. - radius / rods.radius)
            wall = replace(wall, vertex_motion=vertex_kinematics.VertexMotion(field, shifts))
            extent = max(extent, float(np.linalg.norm(shifts, axis=1).max(initial=0.)))
        yield wall, offsets, links, fraction, extent, edited


def query(engine, frame, a, b, radius):
    return contact_features(engine, frame, a, b, radius)[:3]


def contact_features(engine, frame, a, b, radius, *, with_points=False):
    wall, offsets, _, _, extent, edited = frame
    box_owners = np.empty((0, 3), dtype=int)
    box_distances = np.empty(0)
    box_differences = np.empty((0, 3))
    if edited is not None and len(edited.cuboids):
        cells = _candidates(wall.cell_tree, a, b, radius, wall.cell_extent + extent)
        cells = cells[np.isin(cells, edited.cuboids)]
        moved = wall.positions[cells] + offsets[cells]
        native = moved @ engine.scene.inverse[:3, :3].T + engine.scene.inverse[:3, 3]
        distances = _segment_cell_distances(engine.grid, native, a, b)
        contact = distances <= radius + 1e-7
        # The same owner array shape also carries triangle feature hits.
        box_owners = np.repeat(cells[contact, None], 3, axis=1)
        box_distances = distances[contact]
        box_differences = moved[contact] - _projection(moved[contact], a, b)[0]

    tree = edited.tree if edited is not None else wall.tree
    tri_extent = edited.extent if edited is not None else wall.triangle_extent
    elements = edited.triangles if edited is not None else wall.source.triangles
    bindings = edited.owners if edited is not None else wall.source.owners
    source_triangles = edited.world_triangles if edited is not None else wall.triangles
    motion_bindings = None if wall.vertex_motion is not None else bindings
    motion = wall.vertex_motion.offsets if wall.vertex_motion is not None else offsets
    hits = _candidates(tree, a, b, radius, tri_extent + extent)
    corners = elements[hits]
    triangles = source_triangles[hits] + motion[corners if motion_bindings is None else motion_bindings[corners]]
    distances, difference, *closest = _triangle_distances(a, b, triangles, with_points=with_points)
    accepted = distances <= radius + 1e-7
    box_corners = np.full(box_owners.shape, -1, dtype=int)
    if not np.any(accepted) and wall.cell_tree.query(a)[0] <= wall.cell_extent + extent + 1e-7:
        hit = _interior_contact(tree, tri_extent, elements, source_triangles, motion_bindings, motion, extent, a)
        if hit is not None:
            vertices, delta = hit
            if with_points:
                rest = wall.source.vertices[vertices] if edited is None else (
                    edited.positions_ijk[vertices] @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3])
                current = rest + motion[vertices if motion_bindings is None else bindings[vertices]]
                return bindings[vertices][None], np.zeros(1), delta[None], vertices[None], (a - delta)[None], current[None]
            return (np.vstack((bindings[vertices], box_owners)), np.r_[0., box_distances],
                    np.vstack((delta, box_differences)), np.vstack((vertices, box_corners)))
    vertices = corners[accepted]
    if with_points:
        return bindings[vertices], distances[accepted], difference[accepted], vertices, closest[0][accepted], triangles[accepted]
    return (np.vstack((bindings[vertices], box_owners)), np.r_[distances[accepted], box_distances],
            np.vstack((difference[accepted], box_differences)), np.vstack((vertices, box_corners)))


def _interior_contact(tree, triangle_extent, elements, source_triangles, motion_bindings, offsets, extent, point):
    # A capsule completely inside a closed native component crosses no wall.
    # A non-axis-aligned ray tests that case and retains the nearest exit owner.
    axis = np.array([1., .072934, .039701])
    reach = max(0., tree.maxes[0] - point[0]) + triangle_extent + extent + 1.
    hits = _candidates(tree, point, point + reach * axis, 0., triangle_extent + extent)
    corners = elements[hits]
    motion_indices = corners if motion_bindings is None else motion_bindings[corners]
    first, second, third = (source_triangles[hits] + offsets[motion_indices]).transpose(1, 0, 2)
    u, v = second - first, third - first
    pvec = np.cross(axis, v)
    determinant = np.sum(u * pvec, axis=1)
    valid = np.abs(determinant) > 1e-15
    reciprocal = 1. / np.where(valid, determinant, 1.)
    rel = point - first
    s = np.sum(rel * pvec, axis=1) * reciprocal
    qvec = np.cross(rel, u)
    t = qvec @ axis * reciprocal
    distance = np.sum(v * qvec, axis=1) * reciprocal
    intersected = np.flatnonzero(valid & (s >= -1e-10) & (t >= -1e-10) & (s + t <= 1. + 1e-10)
                                 & (distance > 1e-10) & (distance <= reach))
    if len(np.unique(np.round(distance[intersected], 8))) % 2 == 0:
        return None
    nearest = intersected[np.argmin(distance[intersected])]
    return elements[hits[nearest]], -axis * distance[nearest]


def sites_for_owners(frame, owners):
    wall, _, links, fractions, _, _ = frame
    flat = owners.ravel()
    rows = np.searchsorted(wall.influenced, flat)
    covered = rows < len(wall.influenced)
    covered[covered] = wall.influenced[rows[covered]] == flat[covered]
    links_for_owners = np.full((len(flat), 2), -1, dtype=int)
    weights = np.zeros((len(flat), 2))
    rr = rows[covered]
    links_for_owners[covered] = links[rr]
    weights[covered] = wall.weights[rr, None] * np.column_stack((1 - fractions[rr], fractions[rr]))
    return links_for_owners.reshape((*owners.shape, 2)), weights.reshape((*owners.shape, 2))


def contact_sites(frame, owners, vertices):
    if frame[0].vertex_motion is None:
        return sites_for_owners(frame, owners)
    flat = vertices.ravel()
    native = np.flatnonzero(flat >= 0)
    boxes = np.flatnonzero(flat < 0)
    matrix = frame[0].vertex_motion.binding.translation[flat[native]]
    counts = np.diff(matrix.indptr)
    width = max(int(counts.max(initial=0)), 2 if len(boxes) else 0, 1)
    links = np.full((len(flat), width), -1, dtype=int)
    weight = np.zeros((len(flat), width))
    rows = np.repeat(native, counts)
    columns = np.arange(matrix.nnz) - np.repeat(matrix.indptr[:-1], counts)
    links[rows, columns], weight[rows, columns] = matrix.indices, matrix.data
    if len(boxes):
        links[boxes, :2], weight[boxes, :2] = sites_for_owners(frame, owners.ravel()[boxes])
    return links.reshape((*owners.shape, width)), weight.reshape((*owners.shape, width))


def sweep_mesh(frame):
    wall, offsets, _, _, _, edited = frame
    faces = edited.triangles if edited is not None else wall.source.triangles
    rest = edited.world_triangles if edited is not None else wall.triangles
    if wall.vertex_motion is not None:
        return rest, faces, faces, wall.vertex_motion.offsets
    bindings = edited.owners if edited is not None else wall.source.owners
    return rest, faces, bindings[faces], offsets


def sweep_offsets(frame):
    motion = frame[0].vertex_motion
    return frame[1] if motion is None else motion.offsets


def displacements(frames):
    result = {}
    for wall, offsets, _, _, _, _ in frames:
        moved = np.flatnonzero(np.max(np.abs(offsets), axis=1) > 1e-10)
        result.update((str(wall.source.cells[i]), offsets[i].tolist()) for i in moved)
    return result


def geometry_metadata(frames):
    modes = {}
    edits = {}
    motions = {}
    for wall, _, _, _, _, edited in frames:
        label = str(wall.source.label)
        if edited is not None:
            modes[label] = 'edited-native-wall'
            edits[label] = {'identity': edited.identity, 'positionsIJK': edited.positions_ijk.tolist(),
                            'triangles': edited.triangles.tolist(),
                            'sourceCells': wall.source.cells[edited.owners].tolist(),
                            **({'cuboidSourceCells': wall.source.cells[edited.cuboids].tolist()} if len(edited.cuboids) else {})}
        else:
            modes[label] = 'native-probability-wall'
        if wall.vertex_motion is not None:
            motion = wall.vertex_motion
            indices = np.flatnonzero(np.any(motion.offsets != 0., axis=1))
            motions[label] = {'geometryBinding': motion.binding.identity, 'vertexCount': len(motion.offsets),
                              'indices': indices.tolist(), 'offsetsMm': motion.offsets[indices].tolist()}
    return {'geometryModeByLabel': modes, **({'editedWallsByLabel': edits} if edits else {}),
            **({'vertexDisplacementsByLabel': motions} if motions else {})}
