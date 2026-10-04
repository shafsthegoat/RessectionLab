#!/usr/bin/env python3
"""One prospective graded mesh and bounded diagnostics; no image reader or CLI.

Only a separately released caller may provide anatomy and an initialized Gmsh
API. Importing this preparation loads exact reviewed validators, never anatomy,
Gmsh, VTK, landmarks, or a solver. Original v1 code and records stay unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / 'manifests/experiments/resect-case4-patient-mesh-graded-v2.json'
BASE_SHA = 'd8aae815c66ae017e925c1b4ba16f1e6a64c1d2ee314153ef9889c165d64b74d'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reviewed_base():
    path = ROOT / 'scripts/mechanics_patient_mesh.py'
    if sha(path) != BASE_SHA:
        raise ValueError('Reviewed v1 geometry helpers changed')
    spec = importlib.util.spec_from_file_location('reviewed_mesh_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = reviewed_base()


def declaration():
    return json.loads(DECLARATION.read_text())


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def array_header(array):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        'descr': np.lib.format.dtype_to_descr(array.dtype),
        'fortran_order': False, 'shape': array.shape})
    return stream.getvalue()


def preserve_diagnostic(destination, nodes, cells, origin, context, limits):
    """Save complete bounded arrays, or omit all of them with an explicit reason.

    A partial directory survives write failure; it never has a complete marker.
    Rename the whole directory only after exact sizes/hashes/manifest fit the
    declared budget. Completeness means diagnostic completeness, not eligibility.
    """
    destination = Path(destination)
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError('Existing diagnostic preserved')
    X = np.asarray(nodes); E = np.asarray(cells)
    counts = {'nodes': len(X), 'tet10_elements': len(E)}
    result = {'status': 'omitted', 'complete': False, 'candidate_accepted': False,
              'counts': counts, 'limits': limits}
    if len(X) > limits['maximum_nodes'] or len(E) > limits['maximum_elements']:
        return dict(result, reason='diagnostic_count_cap')
    if (X.ndim != 2 or X.shape[1] != 3 or X.dtype != np.dtype('float64')
            or E.ndim != 2 or E.shape[1] != 10 or E.dtype != np.dtype('int64')):
        return dict(result, reason='unsupported_array_representation')
    if len(origin['gmsh_node_ids']) != len(X) or len(origin['gmsh_element_ids']) != len(E):
        return dict(result, reason='incomplete_native_ids')
    permutation = origin['gmsh_to_febio_permutation']
    if sorted(permutation) != list(range(10)):
        return dict(result, reason='invalid_febio_permutation')
    arrays = {'nodes_m.npy': X, 'tet10_indices.npy': E,
              'gmsh_node_ids.npy': np.asarray(origin['gmsh_node_ids'], dtype=np.int64),
              'gmsh_element_ids.npy': np.asarray(origin['gmsh_element_ids'], dtype=np.int64)}
    arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    sizes = {name: len(array_header(value)) + value.nbytes for name, value in arrays.items()}
    allocation = sum(sizes.values()) + limits['maximum_metadata_bytes']
    result['reserved_total_bytes'] = allocation
    if allocation > limits['maximum_total_bytes']:
        return dict(result, reason='diagnostic_byte_cap')
    manifest = {'schema': 'complete-rejected-mesh-diagnostic-v1',
        'diagnostic_complete': True, 'candidate_accepted': False,
        'interpretation': 'Full available diagnostic arrays only; no eligibility, geometry, fidelity or solver approval.',
        'coordinate_frame': 'native Case4 T1 RAS', 'coordinate_units': 'm',
        'connectivity': 'zero-based FEBio tet10; corner nodes followed by edges12,23,31,14,24,34',
        'gmsh_to_febio_permutation': permutation, 'context': context,
        'counts': counts, 'files': {name: {'bytes': size, 'sha256': '0' * 64,
            'dtype': str(arrays[name].dtype), 'shape': list(arrays[name].shape)} for name, size in sizes.items()}}
    if len(encoded(manifest)) > limits['maximum_metadata_bytes']:
        return dict(result, reason='diagnostic_metadata_cap')
    partial.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in arrays.items():
            expected = hashlib.sha256(array_header(value))
            expected.update(memoryview(value).cast('B'))
            with (partial / name).open('xb') as stream:
                np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
            if (partial / name).stat().st_size != sizes[name]:
                raise ValueError('Unexpected diagnostic array size')
            actual = sha(partial / name)
            if actual != expected.hexdigest():
                raise ValueError('Diagnostic bytes differ from complete original array')
            manifest['files'][name]['sha256'] = actual
        metadata = encoded(manifest)
        actual_bytes = sum(sizes.values()) + len(metadata)
        if len(metadata) > limits['maximum_metadata_bytes'] or actual_bytes > limits['maximum_total_bytes']:
            raise ValueError('Diagnostic completeness exceeds byte budget')
        # Do not leave a complete marker when rename/write fails.
        with (partial / 'manifest.pending').open('xb') as stream:
            stream.write(metadata)
        if (partial / 'manifest.pending').stat().st_size != len(metadata) or sha(partial / 'manifest.pending') != hashlib.sha256(metadata).hexdigest():
            raise ValueError('Incomplete diagnostic metadata write')
        manifest_sha = hashlib.sha256(metadata).hexdigest()
        partial.rename(destination)
        try:
            (destination / 'manifest.pending').rename(destination / 'manifest.json')
        except BaseException:
            destination.rename(partial)
            raise
        return dict(result, status='complete', complete=True,
                    directory=str(destination), total_bytes=actual_bytes,
                    manifest_sha256=manifest_sha)
    except BaseException as error:
        return dict(result, status='write_failed', reason=type(error).__name__ + ': ' + str(error),
                    partial_directory=str(partial))


def configure_profile(gmsh, surface_tags, profile):
    """Gmsh size requests, not geometric-error or node-count guarantees."""
    distance = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(distance, 'SurfacesList', surface_tags)
    gmsh.model.mesh.field.setNumber(distance, 'Sampling', profile['sampling_per_dimension'])
    threshold = gmsh.model.mesh.field.add('Threshold')
    for name, value in {'InField': distance, 'SizeMin': profile['boundary_size_m'],
        'SizeMax': profile['interior_size_m'], 'DistMin': profile['distance_min_m'],
        'DistMax': profile['distance_max_m'], 'Sigmoid': 0, 'StopAtDistMax': 0}.items():
        gmsh.model.mesh.field.setNumber(threshold, name, value)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber('Mesh.MeshSizeMin', profile['boundary_size_m'])
    gmsh.option.setNumber('Mesh.MeshSizeMax', profile['interior_size_m'])


def generate_one(gmsh, vertices, faces, config, charge):
    """The reviewed discrete-surface workflow with one graded size-field change."""
    gmsh.clear(); gmsh.model.add(config['candidate_id'])
    for name, value in config['gmsh_options'].items():
        gmsh.option.setNumber(name, value)
    surface = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, surface, np.arange(1, len(vertices)+1).tolist(), np.asarray(vertices).reshape(-1).tolist())
    gmsh.model.mesh.addElementsByType(surface, 2, [], (np.asarray(faces)+1).reshape(-1).tolist())
    gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    tags = [int(tag) for _, tag in gmsh.model.getEntities(2)]
    if not 0 < len(tags) <= config['caps']['maximum_discrete_patches']:
        raise ValueError('Discrete patch cap exceeded')
    loop = gmsh.model.geo.addSurfaceLoop(tags)
    gmsh.model.geo.addVolume([loop]); gmsh.model.geo.synchronize()
    configure_profile(gmsh, tags, config['size_profile'])
    charge(); gmsh.model.mesh.generate(3)
    kinds, element_ids, connectivity = gmsh.model.mesh.getElements(3)
    if list(kinds) != [11]:
        raise ValueError('Only native type11 tet10 accepted')
    _, dimension, order, count, local, primary = gmsh.model.mesh.getElementProperties(11)
    if (dimension, order, count, primary) != (3, 2, 10, 4):
        raise ValueError('Unexpected native tet10 properties')
    permutation = BASE.tet10_permutation(local)
    ids, coordinates, _ = gmsh.model.mesh.getNodes()
    cells = np.asarray(connectivity[0], dtype=np.int64).reshape(-1, 10)[:, permutation]
    used = np.unique(cells); order_nodes = np.argsort(ids); sorted_ids = np.asarray(ids)[order_nodes]
    positions = np.searchsorted(sorted_ids, used)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], used):
        raise ValueError('Missing generated node')
    nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[order_nodes[positions]]
    return nodes, np.searchsorted(used, cells).astype(np.int64), {
        'gmsh_node_ids': used.tolist(), 'gmsh_element_ids': np.asarray(element_ids[0]).tolist(),
        'gmsh_to_febio_permutation': permutation.tolist(), 'discrete_patches': len(tags)}


def assess_candidate(gmsh, vertices, faces, output, *, distance_factory, config=None):
    """One candidate only. Caller supplies released arrays/API and supervision.

    This helper is deliberately not a standalone execution launcher. It neither
    reads patient files nor authorizes a solve; a source-bound caller is required.
    """
    config = declaration() if config is None else config
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'candidate_id': config['candidate_id'],
              'native_generation_calls': 0, 'solver_calls': 0, 'retries': 0,
              'clinical_validation': False, 'solver_admitted': False}
    def save(): BASE.write(output / 'result.json', result)
    def charge():
        result['native_generation_calls'] += 1
        if result['native_generation_calls'] != 1:
            raise ValueError('Only one native generation permitted')
        save()
    save()
    try:
        source = BASE.validate_surface(vertices, faces)
        if len(faces) > config['caps']['maximum_native_triangles']:
            raise ValueError('Source face cap exceeded')
        result['source_surface'] = source; save()
        nodes, cells, origin = generate_one(gmsh, vertices, faces, config, charge)
        result.update(returned_nodes=len(nodes), returned_elements=len(cells),
                      discrete_patches=origin['discrete_patches'])
        context = {'candidate_id': config['candidate_id'],
            'declaration_sha256': hashlib.sha256(encoded(config)).hexdigest(),
            'source_surface_binding': config['source_surface_binding'],
            'source_binding_scope': 'Declared file identity must be verified by released caller; actual provided array digests follow.',
            'source_arrays': {name: {'shape': list(np.asarray(value).shape), 'dtype': str(np.asarray(value).dtype),
                'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                for name, value in [('vertices_m', vertices), ('triangles', faces)]},
            'source_case': config['source_case'], 'helper_v1_sha256': BASE_SHA}
        result['diagnostic'] = preserve_diagnostic(output / 'diagnostic', nodes, cells, origin,
                                                   context, config['diagnostic_limits'])
        save()
        if len(nodes) > config['eligibility']['maximum_nodes'] or len(cells) > config['eligibility']['maximum_elements']:
            raise ValueError('Candidate count eligibility rejected; diagnostic remains separate')
        if not result['diagnostic']['complete']:
            raise ValueError('Complete diagnostic unavailable; candidate cannot continue')
        quality, boundary_nodes, boundary_faces = BASE.validate_tet10(nodes, cells, config['eligibility'], config)
        result['quality'] = quality; save()
        if quality['boundary']['euler_characteristic'] != source['euler_characteristic']:
            raise ValueError('Boundary topology changed')
        result['source_to_mesh'] = BASE.directed_surface_bound(vertices, faces,
            distance_factory(boundary_nodes, boundary_faces), config['surface_fidelity']); save()
        result['mesh_to_source'] = BASE.directed_surface_bound(boundary_nodes, boundary_faces,
            distance_factory(vertices, faces), config['surface_fidelity']); save()
        result['relative_volume_error'] = abs(quality['volume_m3']/source['enclosed_volume_m3']-1); save()
        if max(result[key]['full_surface_upper_bound_m'] for key in ['source_to_mesh', 'mesh_to_source']) > config['surface_fidelity']['maximum_distance_m']:
            raise ValueError('Independent surface fidelity rejected')
        if result['relative_volume_error'] > config['surface_fidelity']['maximum_relative_volume_error']:
            raise ValueError('Independent volume fidelity rejected')
        result['status'] = 'geometry_candidate_passed_no_solver_authorization'
    except BaseException as error:
        result.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
    save()
    return result
