#!/usr/bin/env python3
"""Prepared Case4 mask-to-tet10 utility; import never reads anatomy or loads Gmsh.

One separately released run uses existing marching cubes, Gmsh and VTK. There is
no equilibrium solver, landmark reader, smoothing, dilation or anatomy repair.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifests/experiments/resect-case4-patient-mesh-v1.json'
VERSION = 'resect-case4-patient-mesh-v1'
LEVEL_IDS = ('coarse', 'medium', 'fine')
EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != Path(path).resolve():
        raise ValueError('Wrong module origin')
    return module


def unchanged(bindings):
    result = {}
    for path, expected in bindings.items():
        try:
            result[path] = sha(path) == expected
        except (OSError, ValueError):
            result[path] = False
    return result


def skyline_storage(node_count):
    equations = 3 * int(node_count)
    values = equations * (equations + 1) // 2
    return {'unreduced_displacement_dofs': equations,
            'worst_case_symmetric_values_bytes': 8 * values,
            'three_value_arrays_bytes': 24 * values,
            'scope': 'Conservative dense-profile values only; other allocations and runtime remain unmeasured.'}


def validate_surface(vertices, faces):
    """Closed connected oriented 2-manifold, including one fan at each vertex."""
    X = np.asarray(vertices, float); F = np.asarray(faces)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError('Finite surface vertices required')
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in 'iu' or len(F) < 4:
        raise ValueError('Integer triangle array required')
    if F.min() < 0 or F.max() >= len(X) or np.any(np.diff(np.sort(F, axis=1), axis=1) == 0):
        raise ValueError('Invalid triangle indices')
    if len(np.unique(np.sort(F, axis=1), axis=0)) != len(F):
        raise ValueError('Duplicate triangles')
    used = np.unique(F)
    if len(used) != len(X):
        raise ValueError('Unused surface vertices')
    triangle = X[F]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-16):
        raise ValueError('Degenerate surface triangles')
    start = F.reshape(-1); finish = np.roll(F, -1, axis=1).reshape(-1)
    directed = np.column_stack((start, finish))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError('Surface is open or edge-nonmanifold')
    pairs = np.argsort(inverse, kind='stable').reshape(-1, 2)
    if not np.array_equal(directed[pairs[:, 0]], directed[pairs[:, 1], ::-1]):
        raise ValueError('Inconsistent surface orientation')
    twin = np.empty(len(start), int)
    twin[pairs[:, 0]] = pairs[:, 1]; twin[pairs[:, 1]] = pairs[:, 0]
    previous = np.roll(np.arange(len(start)).reshape(-1, 3), 1, axis=1).reshape(-1)
    rotation = twin[previous]
    graph = csr_matrix((np.ones(len(start), np.uint8), (np.arange(len(start)), rotation)), shape=(len(start), len(start)))
    _, fans = connected_components(graph, directed=True, connection='strong')
    _, first = np.unique(fans, return_index=True)
    if np.any(np.bincount(start[first], minlength=len(X)) != 1):
        raise ValueError('Surface has a nonmanifold vertex fan')
    adjacency = csr_matrix((np.ones(len(start), np.uint8), (start, finish)), shape=(len(X), len(X)))
    components, _ = connected_components(adjacency, directed=False)
    if components != 1:
        raise ValueError('One connected surface required; no automatic component selection')
    origin = X.mean(axis=0)
    signed_volume = np.einsum('ij,ij->i', triangle[:, 0] - origin, cross).sum() / 6
    if signed_volume <= 0:
        raise ValueError('Outward surface with positive enclosed volume required')
    euler = int(len(X) - len(counts) + len(F))
    if euler > 2 or euler % 2:
        raise ValueError('Invalid orientable closed-surface Euler characteristic')
    return {'vertices': len(X), 'triangles': len(F), 'components': components,
            'euler_characteristic': euler, 'genus': (2 - euler) // 2,
            'enclosed_volume_m3': float(signed_volume), 'area_m2': float(np.linalg.norm(cross, axis=1).sum() / 2)}


def extract_native_surface(mask, affine_ras_mm, marching_cubes):
    mask = np.asarray(mask); affine = np.asarray(affine_ras_mm, float)
    if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('Unmodified nonempty binary native mask required')
    if any(np.any(np.take(mask, side, axis=axis)) for axis in range(3) for side in (0, -1)):
        raise ValueError('Mask touches image boundary; padding/closure repair prohibited')
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or np.linalg.det(affine[:3, :3]) <= 0:
        raise ValueError('Expected positive-handed native affine')
    vertices, faces, _, _ = marching_cubes(mask.astype(np.float32), level=.5, spacing=(1., 1., 1.),
        gradient_direction='descent', step_size=1, allow_degenerate=False, method='lewiner')
    physical = (vertices @ affine[:3, :3].T + affine[:3, 3]) / 1000
    # A single orientation reversal changes indexing, never geometry/domain.
    t = physical[faces]; origin = physical.mean(axis=0)
    flipped = np.einsum('ij,ij->i', t[:, 0] - origin, np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum() < 0
    if flipped:
        faces = faces[:, ::-1]
    report = validate_surface(physical, faces)
    report.update(global_orientation_reversed=bool(flipped), voxel_mask_volume_m3=float(mask.sum() * np.linalg.det(affine[:3, :3]) / 1e9))
    return physical, faces.astype(np.int64), report


def triangle_samples(vertices, faces, cover_m, maximum_points, chunk_triangles=256):
    """Barycentric lattice with full triangle cover radius <= max_edge/n.

    Each face is partitioned into n² similar subtriangles. Every point belongs
    to one such triangle and lies within its diameter of a lattice vertex.
    This covers interiors and edges; it is not a vertex-only distance test.
    """
    X = np.asarray(vertices); F = np.asarray(faces)
    tri = X[F]
    longest = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    divisions = np.maximum(1, np.ceil(longest / cover_m).astype(int))
    count = int(np.sum((divisions + 1) * (divisions + 2) // 2))
    if count > maximum_points:
        raise ValueError('Surface-distance sample cap exceeded before allocation')
    for n in np.unique(divisions):
        weights = np.array([(i / n, j / n, 1 - (i + j) / n) for i in range(n + 1) for j in range(n + 1 - i)])
        group = np.flatnonzero(divisions == n)
        for offset in range(0, len(group), chunk_triangles):
            chosen = group[offset:offset + chunk_triangles]
            yield np.einsum('pi,tij->tpj', weights, tri[chosen]).reshape(-1, 3), float(np.max(longest[chosen] / n))


def directed_surface_bound(vertices, faces, distance, config):
    maximum = 0.; total = 0; bound = 0.
    for points, cover in triangle_samples(vertices, faces, config['cover_radius_m'], config['maximum_samples_per_direction']):
        values = np.asarray(distance(points))
        if values.shape != (len(points),) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid nearest-triangle distance output')
        largest = float(values.max()); maximum = max(maximum, largest)
        bound = max(bound, largest + cover); total += len(points)
    return {'sample_count': total, 'maximum_sample_distance_m': maximum,
            'full_surface_upper_bound_m': bound, 'cover_radius_cap_m': config['cover_radius_m']}


def vtk_distance_function(vertices, faces):
    """Existing VTK triangle locator; unsigned distances in physical meters."""
    from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    points = vtkPoints(); points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=np.float64), deep=True))
    cells = vtkCellArray(); cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).reshape(-1), deep=True))
    poly = vtkPolyData(); poly.SetPoints(points); poly.SetPolys(cells)
    locator = vtkImplicitPolyDataDistance(); locator.SetInput(poly)
    def distance(query):
        result = vtkDoubleArray(); result.SetNumberOfComponents(1); result.SetNumberOfTuples(len(query))
        locator.EvaluateFunction(numpy_to_vtk(np.asarray(query, dtype=np.float64), deep=True), result)
        return np.abs(vtk_to_numpy(result).copy())
    return distance


def tet10_permutation(local_coordinates):
    local = np.asarray(local_coordinates).reshape(10, 3)
    desired = np.vstack((local[:4], local[EDGES].mean(axis=1)))
    distance = np.linalg.norm(desired[:, None] - local[None, :], axis=2)
    permutation = distance.argmin(axis=1)
    if len(set(permutation)) != 10 or np.max(distance[np.arange(10), permutation]) > 1e-12:
        raise ValueError('Unknown Gmsh tet10 reference-node layout')
    return permutation


def reject_tetrahedral_overlap(nodes, corners, *, maximum_pairs, tolerance_m=1e-10):
    """Independent convex-tetrahedron separating-axis control, no mechanics."""
    t = np.asarray(nodes)[corners]; centers = t.mean(axis=1)
    radius = np.max(np.linalg.norm(t - centers[:, None], axis=2), axis=1)
    tree = cKDTree(centers)
    checked = 0
    for i in range(len(t)):
        neighbors = np.asarray(tree.query_ball_point(centers[i], radius[i] + radius.max()), int)
        neighbors = neighbors[neighbors > i]
        neighbors = neighbors[np.all(t[neighbors].min(axis=1) < t[i].max(axis=0) - tolerance_m, axis=1) & np.all(t[neighbors].max(axis=1) > t[i].min(axis=0) + tolerance_m, axis=1)]
        checked += len(neighbors)
        if checked > maximum_pairs:
            raise ValueError('Overlap candidate-pair cap exceeded')
        for first in range(0, len(neighbors), 256):
            b = t[neighbors[first:first + 256]]
            if not len(b):
                continue
            ea = t[i, EDGES[:, 1]] - t[i, EDGES[:, 0]]
            eb = b[:, EDGES[:, 1]] - b[:, EDGES[:, 0]]
            fa = t[i, FACES]; fb = b[:, FACES]
            na = np.cross(fa[:, 1] - fa[:, 0], fa[:, 2] - fa[:, 0])
            nb = np.cross(fb[:, :, 1] - fb[:, :, 0], fb[:, :, 2] - fb[:, :, 0])
            cross = np.cross(ea[None, :, None], eb[:, None]).reshape(len(b), 36, 3)
            axes = np.concatenate((np.broadcast_to(na, (len(b), 4, 3)), nb, cross), axis=1)
            lengths = np.linalg.norm(axes, axis=2); valid = lengths > 1e-18
            axes /= np.where(valid, lengths, 1)[:, :, None]
            pa = np.einsum('paj,vj->pav', axes, t[i]); pb = np.einsum('paj,pvj->pav', axes, b)
            penetration = np.minimum(pa.max(axis=2), pb.max(axis=2)) - np.maximum(pa.min(axis=2), pb.min(axis=2))
            separated = np.any(valid & (penetration <= tolerance_m), axis=1)
            if not np.all(separated):
                raise ValueError('Positive-volume tetrahedron overlap detected')
    return {'aabb_candidate_pairs_checked': checked, 'contact_tolerance_m': tolerance_m}


def validate_tet10(nodes, cells, level, protocol):
    X = np.asarray(nodes, float); E = np.asarray(cells)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all() or len(X) > level['maximum_nodes']:
        raise ValueError('Node geometry/count gate failed')
    if E.ndim != 2 or E.shape[1] != 10 or E.dtype.kind not in 'iu' or not 0 < len(E) <= level['maximum_elements']:
        raise ValueError('Only bounded tet10 volumes accepted')
    if E.min() < 0 or E.max() >= len(X) or np.any(np.diff(np.sort(E, axis=1), axis=1) == 0):
        raise ValueError('Invalid tet10 indices')
    if len(np.unique(E)) != len(X) or len(np.unique(np.sort(E[:, :4], axis=1), axis=0)) != len(E):
        raise ValueError('Unused nodes or duplicate tetrahedra')
    corners = X[E[:, :4]]
    determinant = np.linalg.det(np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2))
    if np.any(determinant <= 0):
        raise ValueError('Nonpositive reference Jacobian')
    midpoint_error = np.max(np.linalg.norm(X[E[:, 4:]] - corners[:, EDGES].mean(axis=2), axis=2))
    if midpoint_error > protocol['quality']['midpoint_tolerance_m']:
        raise ValueError('Curved tet10 reference geometry unsupported')
    # Check actual returned quadratic geometry, including its tiny floating
    # midpoint discrepancies, at the established G8 and extra witness points.
    a, b, c, d = .0158359099, .3280546970, .6791431780, .1069522740
    sites = np.vstack((np.array([[a,b,b],[b,a,b],[b,b,a],[b,b,b],
        [c,d,d],[d,c,d],[d,d,c],[d,d,d]]), np.eye(4)[:, 1:], [[.25]*3],
        np.eye(4)[EDGES].mean(axis=1)[:, 1:]))
    dl = np.array([[-1.,-1.,-1.],[1,0,0],[0,1,0],[0,0,1]])
    jacobians = []
    for point in sites:
        l = np.r_[1-point.sum(), point]
        grad = np.vstack(((4*l[:, None]-1)*dl,
            4*(l[EDGES[:, 0], None]*dl[EDGES[:, 1]] + l[EDGES[:, 1], None]*dl[EDGES[:, 0]])))
        jacobians.append(np.linalg.det(np.einsum('eni,nj->eij', X[E], grad)))
    actual_jacobians = np.asarray(jacobians)
    if actual_jacobians.min() <= 0:
        raise ValueError('Nonpositive sampled actual tet10 Jacobian')
    edge_ids = np.sort(E[:, :4][:, EDGES], axis=2).reshape(-1, 2)
    _, inv = np.unique(edge_ids, axis=0, return_inverse=True)
    middle = E[:, 4:].reshape(-1)
    order = np.argsort(inv, kind='stable')
    if np.any((np.diff(inv[order]) == 0) & (np.diff(middle[order]) != 0)):
        raise ValueError('Nonconforming shared midside node')
    edge_lengths2 = np.sum((corners[:, EDGES[:, 1]] - corners[:, EDGES[:, 0]]) ** 2, axis=2)
    quality = 12 * (determinant / 2) ** (2 / 3) / edge_lengths2.sum(axis=1)
    if quality.min() < protocol['quality']['minimum_mean_ratio']:
        raise ValueError('Tetrahedral mean-ratio quality gate failed')
    faces = E[:, :4][:, FACES].reshape(-1, 3)
    _, invf, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold volume face')
    boundary = faces[counts[invf] == 1]
    used = np.unique(boundary); mapping = np.full(len(X), -1); mapping[used] = np.arange(len(used))
    surface = validate_surface(X[used], mapping[boundary])
    volume = float(determinant.sum() / 6)
    if not np.isclose(volume, surface['enclosed_volume_m3'], rtol=1e-10, atol=1e-15):
        raise ValueError('Volume/boundary integral mismatch')
    overlap = reject_tetrahedral_overlap(X, E[:, :4], maximum_pairs=protocol['quality']['maximum_overlap_candidate_pairs'])
    return {'nodes': len(X), 'elements': len(E), 'minimum_reference_determinant_m3': float(determinant.min()),
            'minimum_actual_sampled_reference_determinant_m3': float(actual_jacobians.min()), 'jacobian_witnesses_per_element': len(sites),
            'maximum_midpoint_error_m': float(midpoint_error), 'minimum_mean_ratio': float(quality.min()),
            'volume_m3': volume, 'boundary': surface, 'overlap': overlap,
            'skyline_storage': skyline_storage(len(X))}, X[used], mapping[boundary]


def generate_gmsh_level(gmsh, vertices, faces, level, protocol, before_generate=lambda: None):
    """Pinned tutorial-13 discrete-surface workflow; injected native API only."""
    gmsh.clear(); gmsh.model.add(level['id'])
    for key, value in protocol['gmsh_options'].items():
        gmsh.option.setNumber(key, value)
    gmsh.option.setNumber('Mesh.MeshSizeMin', level['target_edge_m'])
    gmsh.option.setNumber('Mesh.MeshSizeMax', level['target_edge_m'])
    surface = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, surface, np.arange(1, len(vertices) + 1).tolist(), np.asarray(vertices).reshape(-1).tolist())
    gmsh.model.mesh.addElementsByType(surface, 2, [], (np.asarray(faces) + 1).reshape(-1).tolist())
    gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    entities = gmsh.model.getEntities(2)
    if not 0 < len(entities) <= protocol['caps']['maximum_discrete_patches']:
        raise ValueError('Discrete parameterization patch cap exceeded')
    loop = gmsh.model.geo.addSurfaceLoop([int(tag) for _, tag in entities])
    gmsh.model.geo.addVolume([loop]); gmsh.model.geo.synchronize()
    before_generate()
    gmsh.model.mesh.generate(3)
    kinds, tags, connectivity = gmsh.model.mesh.getElements(3)
    if list(kinds) != [11]:
        raise ValueError('Expected Gmsh type11 ten-node tetrahedra only')
    _, dimension, order, count, local, primary = gmsh.model.mesh.getElementProperties(11)
    if (dimension, order, count, primary) != (3, 2, 10, 4):
        raise ValueError('Unexpected tet10 properties')
    permutation = tet10_permutation(local)
    ids, xyz, _ = gmsh.model.mesh.getNodes()
    cells = np.asarray(connectivity[0], np.int64).reshape(-1, 10)[:, permutation]
    used = np.unique(cells); order_nodes = np.argsort(ids)
    sorted_ids = np.asarray(ids)[order_nodes]
    position = np.searchsorted(sorted_ids, used)
    if not np.array_equal(sorted_ids[position], used):
        raise ValueError('Missing generated node')
    nodes = np.asarray(xyz).reshape(-1, 3)[order_nodes[position]]
    zero_based = np.searchsorted(used, cells)
    return nodes, zero_based, {'gmsh_node_ids': used.tolist(), 'gmsh_element_ids': np.asarray(tags[0]).tolist(),
                              'gmsh_to_febio_permutation': permutation.tolist(), 'discrete_patches': len(entities)}


def prepare_release(release_path, output):
    """Only source/receipt bytes; no anatomy array or Gmsh import here."""
    release_path = Path(release_path).resolve(); release = json.loads(release_path.read_text())
    if release.get('schema') != VERSION + '-release' or release.get('authorized') is not True:
        raise ValueError('Separate root release required')
    if Path(release['source_directory']).resolve() != ROOT or Path(release['attempt_directory']).resolve() != Path(output).resolve():
        raise ValueError('Released source/attempt directory mismatch')
    commit = release.get('source_commit', '')
    repository = Path(release['repository_directory']).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit) or repository == ROOT:
        raise ValueError('Exact committed source archive required')
    if release.get('mask_qc_accepted_for_estimated_domain') is not True or release.get('baseline_diagnostic_accepted_for_provisional_research_frame') is not True:
        raise ValueError('Independent mask QC and provisional baseline diagnostic remain prerequisites')
    if release.get('clinical_validation') is not False or release.get('anatomical_registration_accepted') is not False:
        raise ValueError('Research release cannot declare clinical or anatomical registration acceptance')
    required = {'scripts/mechanics_patient_mesh.py', 'scripts/febio_runtime.py', MANIFEST,
                'artifacts/febio-runtime-investigation-v1/prospective-runtime.json'}
    if set(release['source_sha256']) != required:
        raise ValueError('Exact source closure required')
    bound = {str(release_path): sha(release_path)}
    for name, expected in release['source_sha256'].items():
        raw = subprocess.run(['git', '-C', str(repository), 'show', f'{commit}:{name}'],
                             capture_output=True, check=True, timeout=5).stdout
        if sha(ROOT / name) != expected or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Released source changed')
        bound[str(ROOT / name)] = expected
    for item in release['prerequisite_receipts']:
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('QC/alignment receipt changed')
        bound[str(path.resolve())] = item['sha256']
    if {item['role'] for item in release['prerequisite_receipts']} != {'mask_qc', 'baseline_diagnostic'}:
        raise ValueError('Both fixed prerequisite receipts required')
    protocol = json.loads((ROOT / MANIFEST).read_text())
    for item in protocol['input_bindings']:
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('Declared input changed')
        bound[str(path.resolve())] = item['sha256']
    return protocol, bound


def worker(release_path, output):
    output = Path(output); result = {'status': 'running',
        'levels': [{'level': name, 'status': 'not_executed'} for name in LEVEL_IDS],
        'meshing_attempts': 0, 'native_generation_calls': 0, 'solver_calls': 0}
    started = time.monotonic(); gmsh = None; bound = {}; active = None
    if (output / 'worker.json').exists():
        raise FileExistsError('Existing worker attempt preserved')
    try:
        protocol, bound = prepare_release(release_path, output)
        if tuple(level['id'] for level in protocol['levels']) != LEVEL_IDS:
            raise ValueError('Declared mesh sequence changed')
        result.update(input_hashes=bound, protocol_sha256=sha(ROOT / MANIFEST),
            coordinate_frame=protocol['coordinate_frame'], mask_evidence=protocol['mask']['source_status'],
            clinical_validation=False, anatomical_registration_accepted=False)
        write(output / 'worker.json', result)
        import importlib.metadata
        for package, expected in protocol['package_versions'].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError('Declared package version changed: ' + package)
        import nibabel as nib
        from skimage.measure import marching_cubes
        image = nib.load(protocol['mask']['path'])
        if list(image.shape) != protocol['mask']['shape'] or image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Mask native header changed')
        if not np.allclose(image.affine, protocol['mask']['affine_ras_mm'], atol=1e-8, rtol=0):
            raise ValueError('Mask affine changed')
        mask = np.asarray(image.dataobj)
        vertices, faces, surface_report = extract_native_surface(mask, image.affine, marching_cubes)
        if len(faces) > protocol['caps']['maximum_native_triangles']:
            raise ValueError('Native surface face cap exceeded; no simplification fallback')
        result['native_surface'] = surface_report
        np.savez_compressed(output / 'native-surface.npz', vertices_m=vertices, triangles=faces)
        runtime = protocol['gmsh_runtime']
        if 'gmsh' in sys.modules:
            raise ValueError('Ambient Gmsh import prohibited')
        gmsh = import_file(runtime['module_path'], 'gmsh')
        if Path(gmsh.lib._name).resolve() != Path(runtime['library_path']).resolve() or gmsh.__version__ != runtime['version']:
            raise ValueError('Wrong Gmsh native runtime')
        gmsh.initialize([], readConfigFiles=False, run=False)
        result['gmsh_runtime'] = {'module_path': gmsh.__file__, 'library_path': str(gmsh.lib._name),
                                  'version': gmsh.__version__}
        from vtkmodules.vtkCommonCore import vtkSMPTools
        if not vtkSMPTools.SetBackend('Sequential') or vtkSMPTools.GetBackend() != 'Sequential':
            raise ValueError('Serial VTK backend required')
        native_distance = vtk_distance_function(vertices, faces)
        for row, level in zip(result['levels'], protocol['levels']):
            active = row; row['status'] = 'running'
            result['meshing_attempts'] += 1; write(output / 'worker.json', result)
            def charge_generation():
                result['native_generation_calls'] += 1
                write(output / 'worker.json', result)
            nodes, cells, origin = generate_gmsh_level(gmsh, vertices, faces, level, protocol, charge_generation)
            row.update(returned_nodes=len(nodes), returned_elements=len(cells), origin=origin)
            if len(nodes) > level['maximum_nodes'] or len(cells) > level['maximum_elements']:
                raise ValueError('Returned mesh exceeds declared count cap; counts retained')
            # Preserve bounded raw native output before any quality rejection.
            np.savez_compressed(output / (level['id'] + '.npz'), nodes_m=nodes, tet10_indices=cells)
            gmsh.write(str(output / (level['id'] + '.msh')))
            quality, boundary_nodes, boundary_faces = validate_tet10(nodes, cells, level, protocol)
            row['quality'] = quality
            write(output / 'worker.json', result)
            if quality['boundary']['euler_characteristic'] != surface_report['euler_characteristic']:
                raise ValueError('Remeshing changed boundary topology')
            forward = directed_surface_bound(vertices, faces, vtk_distance_function(boundary_nodes, boundary_faces), protocol['surface_fidelity'])
            row['source_to_mesh'] = forward
            write(output / 'worker.json', result)
            reverse = directed_surface_bound(boundary_nodes, boundary_faces, native_distance, protocol['surface_fidelity'])
            volume_error = abs(quality['volume_m3'] / surface_report['enclosed_volume_m3'] - 1)
            row.update(mesh_to_source=reverse, relative_volume_error=volume_error)
            write(output / 'worker.json', result)
            if max(forward['full_surface_upper_bound_m'], reverse['full_surface_upper_bound_m']) > protocol['surface_fidelity']['maximum_distance_m']:
                raise ValueError('Declared physical surface approximation bound exceeded')
            if volume_error > protocol['surface_fidelity']['maximum_relative_volume_error']:
                raise ValueError('Surface-volume fidelity gate failed')
            np.savez_compressed(output / (level['id'] + '-boundary.npz'),
                boundary_vertices_m=boundary_nodes, boundary_triangles=boundary_faces)
            row.update(status='passed', quality=quality, source_to_mesh=forward, mesh_to_source=reverse,
                       relative_volume_error=volume_error, origin=origin)
            write(output / 'worker.json', result)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        if active is not None and active['status'] == 'running':
            active['status'] = 'failed'
    finally:
        if gmsh is not None:
            try:
                gmsh.finalize()
            except BaseException as error:
                result.update(status='failed_or_incomplete', finalize_error=str(error))
        result['elapsed_seconds'] = time.monotonic() - started
        result['inputs_after'] = unchanged(bound)
        if not bound or not all(result['inputs_after'].values()):
            result['status'] = 'failed_or_incomplete'
        result['output_sha256'] = {path.name: sha(path) for path in output.iterdir()
                                  if path.is_file() and path.name != 'worker.json'}
        write(output / 'worker.json', result)
    return 0 if result['status'] == 'completed' else 1


def launch(release_path, output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    acceptance = {'status': 'failed_or_incomplete', 'worker_started': False,
                  'solver_calls': 0, 'clinical_or_anatomical_validation': False}
    bound = {}
    try:
        protocol, bound = prepare_release(release_path, output)
        runtime = import_file(ROOT / 'scripts/febio_runtime.py', 'patient_mesh_supervisor')
        environment = runtime.private_environment(runtime.declaration())
        environment.update(VTK_SMP_MAX_THREADS='1', VTK_SMP_IMPLEMENTATION_TYPE='Sequential')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--release', str(Path(release_path).resolve()), '--output', str(output)]
        acceptance.update(worker_started=True, command=command,
                          thread_environment={name: environment[name] for name in ['OMP_NUM_THREADS', 'OMP_DYNAMIC', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VTK_SMP_MAX_THREADS', 'VTK_SMP_IMPLEMENTATION_TYPE']})
        receipt = runtime.supervise(command, output / 'supervision', cwd=ROOT, environment=environment,
            seconds=protocol['caps']['aggregate_seconds'], rss_bytes=protocol['caps']['process_group_rss_bytes'])
        acceptance['supervision'] = receipt
        record = json.loads((output / 'worker.json').read_text()) if (output / 'worker.json').is_file() else {}
        rows = record.get('levels', [])
        output_hashes = record.get('output_sha256', {})
        output_checks = {name: Path(name).name == name and unchanged({str(output / name): expected}).get(str(output / name), False)
                         for name, expected in output_hashes.items()} if isinstance(output_hashes, dict) else {}
        acceptance['outputs_after'] = output_checks
        if (receipt['status'] == 'completed' and record.get('status') == 'completed'
                and record.get('native_generation_calls') == 3 and record.get('solver_calls') == 0
                and isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
                and [row.get('level') for row in rows] == list(LEVEL_IDS)
                and all(row.get('status') == 'passed' for row in rows)
                and bool(output_checks) and all(output_checks.values())):
            acceptance['status'] = 'completed'
    except BaseException as error:
        acceptance['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        acceptance['inputs_after'] = unchanged(bound)
        if not bound or not all(acceptance['inputs_after'].values()):
            acceptance['status'] = 'failed_or_incomplete'
        write(output / 'acceptance.json', acceptance)
    return 0 if acceptance['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return worker(args.release, args.output) if args.mode == 'worker' else launch(args.release, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
