"""Prepare a bounded native angiography wall with selected patient tissue."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy import ndimage

from .anatomy_evidence import inference_bounds
from .anatomy_geometry import conditioned_vascular_probability_mesh, read_source_branch_graph, source_branch_cache_path
from .anatomy_model_registry import ANGIOGRAPHY, source_volume
from .core import array_digest, semantic_digest
from .surgical_glioma_materials import research_profile
from .surgical_vessel_rods import MODEL, GEOMETRY, SECTION_MECHANICS, section_profile
from .surgical_vessel_surface import CONDITIONED_GEOMETRIES, GEOMETRY_PROPERTY, ISOLATED_SOURCE_GEOMETRY
from . import surgical_vessel_vertex_motion as vertex_motion, surgical_wall_sweep as wall_sweep
from .vascular_source_branches import build_source_branches


SCHEMA = 'resectionlab.patient-native-wall.v1'
HALF_WIDTH_MM = 12.
WORKING_RADIUS_MM = 7.


@dataclass(frozen=True)
class NativeWallGrid:
    affine: np.ndarray
    mask: np.ndarray
    scores: np.ndarray
    start: np.ndarray
    center: np.ndarray
    source_shape: tuple[int, int, int]
    source_sha256: str
    source_branches: list | None


def native_wall_grid(collection, row, registration, center):
    if (not isinstance(center, (list, tuple)) or len(center) != 3
            or any(type(value) not in (float, int) or not np.isfinite(value) for value in center)):
        raise ValueError('Select a finite patient RAS cursor position for local native-wall rehearsal')
    center = np.asarray(center, dtype=float)
    scan = collection.series(row['source_series'])
    native_affine = np.asarray(row['affine'])
    shape = tuple(scan['data']['shape'])
    if (scan['modality'] not in ANGIOGRAPHY or not source_volume(scan) or not row.get('probability')
            or row.get('edits') or row.get('revision') != 0 or row.get('publisher_surface')
            or tuple(row['data']['shape']) != shape or tuple(row['probability']['shape']) != shape
            or not np.allclose(native_affine, scan['affine'], atol=1e-5, rtol=0)):
        raise ValueError('Local native-wall rehearsal requires an intact native angiography score layer')
    patient_affine = registration @ native_affine
    spacing = np.linalg.norm(patient_affine[:3, :3], axis=0)
    axes = patient_affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Local native-wall rehearsal requires an orthogonal angiography lattice')
    native_center = np.linalg.solve(patient_affine[:3, :3], center - patient_affine[:3, 3])
    if np.any(native_center < 0) or np.any(native_center >= shape):
        raise ValueError('The patient cursor is outside the native angiography grid')
    anchor = np.rint(native_center).astype(np.int64)
    half = np.ceil(HALF_WIDTH_MM / spacing).astype(np.int64)
    start, stop = anchor - half, anchor + half + 1
    if np.any(start < 0) or np.any(stop > shape) or np.prod(stop - start) > 2_000_000:
        raise ValueError('The 24 mm local wall domain requires complete native scan coverage within the working budget')
    bounds = inference_bounds(row)
    supported_slices = row.get('coverage_fields', {}).get('supported_axial_indices')
    if (bounds is not None and (np.any(start < bounds[0]) or np.any(stop - 1 > bounds[1]))
            or supported_slices is not None and (not isinstance(supported_slices, list)
                or any(type(index) is not int or not 0 <= index < shape[2] for index in supported_slices)
                or not set(range(start[2], stop[2])).issubset(supported_slices))):
        raise ValueError('The local wall domain extends outside scored vascular coverage')
    crop = tuple(slice(lo, hi) for lo, hi in zip(start, stop))
    grid = patient_affine.copy()
    grid[:3, 3] += grid[:3, :3] @ start
    mask = collection.array(row['data'])[crop] == row['label_value']
    scores = np.asarray(collection.array(row['probability'])[crop], dtype=np.float32)
    if not mask.any() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)) or np.any(scores[mask] < .5):
        raise ValueError('The local vessel annotation is empty or disagrees with the retained score crossings')
    branch_path = source_branch_cache_path(collection, row)
    branches = read_source_branch_graph(collection, row, branch_path)['branches'] if branch_path.is_file() else None
    return NativeWallGrid(grid, mask, scores, start, center, shape, scan['original']['sha256'], branches)


def _face_path(points, mask, radius):
    """Resolve diagonal steps only through existing foreground source cells."""
    if np.any(~mask[tuple(points.T)]) or np.any(np.abs(np.diff(points, axis=0)) > 1):
        return None
    path = [tuple(points[0])]
    reserved = set(map(tuple, points[1:]))
    if len(reserved) != len(points) - 1 or path[0] in reserved:
        return None
    for target in points[1:]:
        reserved.remove(tuple(target))
        origin = np.array(path[-1])
        delta = target - origin
        options = []
        for order in permutations(np.flatnonzero(delta)):
            steps = np.zeros((len(order), 3), int)
            steps[np.arange(len(order)), order] = delta[list(order)]
            cells = list(map(tuple, origin + steps.cumsum(axis=0)))
            if all(mask[cell] and cell not in path and cell not in reserved for cell in cells):
                options.append(cells)
        if not options:
            return None
        path.extend(max(options, key=lambda cells: (sum(radius[cell] for cell in cells), cells)))
    return np.asarray(path)


def _local_chains(grid, radius):
    mask, affine = grid.mask, grid.affine
    original = grid.source_branches is not None
    branches = grid.source_branches if original else build_source_branches(
        mask, affine, source_scan_sha256=grid.source_sha256)['branches']
    candidates = []
    for branch in branches:
        cells = branch['source_cell_ids']
        if branch['closed'] or len(cells) < 3:
            continue
        points = np.column_stack(np.unravel_index(cells, grid.source_shape if original else mask.shape))
        if original:
            points -= grid.start
        nearest = np.linalg.norm(points @ affine[:3, :3].T + affine[:3, 3] - grid.center, axis=1)
        local = (nearest <= WORKING_RADIUS_MM) & ((points >= 1) & (points < np.asarray(mask.shape) - 1)).all(axis=1)
        supported = np.asarray(branch['edge_local_face_supported']) & local[:-1] & local[1:]
        for part in np.split(np.arange(len(cells)), np.flatnonzero(~supported) + 1):
            if len(part) < 3 or nearest[part].min() > 2.:
                continue
            path = _face_path(points[part], mask, radius)
            if path is None or not 3 <= len(path) <= 256:
                continue
            length = np.linalg.norm(np.diff(path, axis=0) @ affine[:3, :3].T, axis=1).sum()
            if length >= 3.:
                candidates.append((float(nearest[part].min()), cells[part[0]], path, branch['id']))
    chains, occupied_faces = {}, set()
    for _, _, path, source_id in sorted(candidates, key=lambda row: row[:2])[:8]:
        cells = np.ravel_multi_index(path.T, mask.shape).tolist()
        faces = {tuple(sorted(pair)) for pair in zip(cells, cells[1:])}
        if faces & occupied_faces:
            continue
        source_cells = np.ravel_multi_index((path + grid.start).T, grid.source_shape).tolist()
        branch_id = (source_id if original else
                     'local-' + semantic_digest([grid.source_sha256, source_cells]).split(':')[1][:20])
        if branch_id in chains:
            continue
        occupied_faces.update(faces)
        chains[branch_id] = (cells, source_cells)
    if not chains:
        raise ValueError('Place the patient cursor within 2 mm of a continuous, locally supported vascular trace')
    return chains


def qualifying_properties(grid, masks, row, patient_id, *, cancelled=lambda: False):
    mask, affine, scores = grid.mask, grid.affine, grid.scores
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    radius = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    chains = _local_chains(grid, radius)
    sites = sorted({cell for cells, _ in chains.values() for cell in cells})
    tissue = masks['brain'].copy()
    if 'tumor' in masks:
        tissue |= masks['tumor']
    if 'bone' in masks:
        tissue &= ~masks['bone']
    tissue &= ~mask
    if not tissue.any():
        raise ValueError('Local native-wall rehearsal requires surrounding patient tissue samples')
    distance, nearest = ndimage.distance_transform_edt(~tissue, sampling=spacing, return_indices=True)
    attachments = []
    rod_sites = {}
    for cell in sites:
        index = np.unravel_index(cell, mask.shape)
        point = affine[:3, :3] @ index + affine[:3, 3]
        caliber = float(np.clip(radius[index] - .5 * min(spacing), .03, 8.))
        if distance[index] <= caliber + np.linalg.norm(spacing):
            attachments.append([cell, int(np.ravel_multi_index(nearest[(slice(None), *index)], mask.shape))])
        rod_sites[str(cell)] = [*point.tolist(), caliber]
    if not attachments or len(attachments) > 512:
        raise ValueError('Selected anatomy does not supply a bounded set of nearby patient tissue supports')

    assumptions = ('Native label-derived local vascular traces and assumed compliant rods. The closest occupied patient tissue '
                   'cells supply assumed springs. Nearby exposed soft-cell faces receive normal wall loads and assumed sliding friction. '
                   'The solver reserves lattice capacity for loaded cells and bounds discarded numerical loads. '
                   'Source masks do not measure physical attachment, lumen radius or material properties.')
    profile = research_profile()
    profile.update(vesselMotionModel=MODEL, assignment=assumptions, vesselAttachments=attachments)
    profile[GEOMETRY] = {'source': 'Native source-cell centers with physical-distance caliber estimates from the vascular annotation.',
                         'sites': rod_sites}
    profile[wall_sweep.PROPERTY] = wall_sweep.CURRENT_POLICY
    profile[vertex_motion.PROPERTY] = vertex_motion.POLICY
    profile[SECTION_MECHANICS] = section_profile(bounded_tissue=True)
    profile['axialStiffnessN']['uncertain'] = .04
    profile['effectiveLumenAreaMm2']['uncertain'] = 1.

    edges = []
    nodes = set()
    for branch_id, (cells, _) in chains.items():
        first, last = f'source:{cells[0]}', f'source:{cells[-1]}'
        nodes.update((first, last))
        edges.append({'id': branch_id, 'kind': 'uncertain', 'from': first, 'to': last,
                      'sourceCells': cells, 'sourceFace': cells[:2], 'conductanceMm3PerPaS': 0.,
                      'woundConductanceMm3PerPaS': [0., 0.], 'source': assumptions})
    vascular = {
        'schema': 'resectionlab.vascular-graph.v1', 'fieldPressurePa': 0.,
        'coverage': {'status': 'partial-acquired', 'sourceCaseId': patient_id,
                     'missing': ['Pressure, circulation and actual tissue attachment', 'Vessel geometry outside the local domain']},
        'provenance': {'kind': 'assumed-research', 'description': assumptions},
        'nodes': [{'id': node, 'complianceMm3PerPa': .002, 'initialStorageMm3': 0., 'source': assumptions}
                  for node in sorted(nodes)],
        'edges': edges, 'boundaries': [], 'regions': [], 'compressionForceN': .04,
        'coagulationDose': 2., 'clotRatePerSecond': 0.,
    }
    geometry = conditioned_vascular_probability_mesh(scores, mask, affine, retain_isolated_components=True,
                                                     cancelled=cancelled)
    if geometry is None or not len(geometry.indices):
        raise ValueError('The local window has no usable native score surface')
    receipt = {
        'schema': SCHEMA, 'sourceSeries': row['source_series'], 'layerId': row['id'],
        'sourceShape': list(grid.source_shape), 'sourceStartIJK': grid.start.tolist(),
        'centerRasMm': grid.center.tolist(), 'sourceImageSha256': grid.source_sha256,
        'sourceMaskSha256': row['data']['sha256'], 'sourceScoresSha256': row['probability']['sha256'],
        'localMaskHash': array_digest(mask), 'localScoreHash': array_digest(scores),
        'geometryPolicy': ISOLATED_SOURCE_GEOMETRY, wall_sweep.PROPERTY: wall_sweep.CURRENT_POLICY,
        'geometryBinding': vertex_motion.geometry_binding(geometry.positions, geometry.indices, geometry.source_cells),
        'sourceCrossingBinding': vertex_motion.geometry_binding(geometry.source, geometry.indices, geometry.source_cells),
        'sourceRetainedIsolatedComponents': geometry.diagnostics.get('sourceRetainedIsolatedComponents', 0),
        'branches': {branch_id: source_cells for branch_id, (_, source_cells) in chains.items()},
        'branchIdentityBasis': 'source study branch graph' if grid.source_branches is not None else 'local trace with original native source cell indices',
        'assumptions': assumptions, 'scope': 'cursor-local mechanical rehearsal',
    }
    return {'gliomaMaterialsV1': profile, 'vascularGraphV1': vascular}, receipt


def validate_saved_wall(scene, case, receipt):
    """Recheck the accepted wall and owner mapping reconstructed on reopen."""
    walls = getattr(scene, '_vessel_source_walls', ())
    profile = scene.properties.get('gliomaMaterialsV1', {})
    if (not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA or len(walls) != 1
            or 'vessels' not in case.compartments
            or receipt.get('geometryPolicy') not in CONDITIONED_GEOMETRIES
            or receipt.get('geometryPolicy') != case.metadata.get(GEOMETRY_PROPERTY)
            or receipt.get('geometryPolicy') != walls[0].geometry_policy
            or receipt.get('geometryBinding') != walls[0].geometry_binding
            or receipt.get('localMaskHash') != array_digest(case.compartments['vessels'])
            or receipt.get('localScoreHash') != array_digest(walls[0].scores)
            or profile.get(wall_sweep.PROPERTY) not in wall_sweep.POLICIES
            or receipt.get(wall_sweep.PROPERTY, wall_sweep.POLICY) != profile.get(wall_sweep.PROPERTY)
            or profile.get(vertex_motion.PROPERTY) != vertex_motion.POLICY):
        raise ValueError('Saved patient native wall differs from accepted source geometry')
    start, shape = receipt.get('sourceStartIJK'), receipt.get('sourceShape')
    if (not isinstance(start, list) or not isinstance(shape, list) or len(start) != 3 or len(shape) != 3
            or any(type(lo) is not int or type(size) is not int or lo < 0 or lo + extent > size
                   for lo, size, extent in zip(start, shape, scene.shape))):
        raise ValueError('Saved patient native wall has an invalid source crop')
    edges = scene.properties['vascularGraphV1']['edges']
    branches = {edge['id']: np.ravel_multi_index(
        np.array(np.unravel_index(edge['sourceCells'], scene.shape)) + np.asarray(start)[:, None], shape).tolist()
        for edge in edges}
    if branches != receipt.get('branches') or not any(
            ref.source_id == receipt.get('sourceSeries') and ref.sha256 == receipt.get('sourceImageSha256')
            for ref in case.source_refs):
        raise ValueError('Saved patient native wall differs from its source acquisition or branch mapping')
