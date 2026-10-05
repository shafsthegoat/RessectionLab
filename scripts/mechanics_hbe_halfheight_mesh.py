"""Pure lower-half axial specimen extraction/decks; no I/O, mesher or solver.

Caller binds the original full mesh, protocol and separate equivalence release.
Only whole saved cells and their unchanged coordinates are retained. Reflections
verify the original full topology; they never replace source coordinates.
"""
from collections import defaultdict
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from scripts import mechanics_hbe_mesh as BASE

COUNTS = {8: (1045, 768), 12: (3199, 2592)}
DECLARATION_SHA256 = '0df5587ab7a70eb1ac092919ff067b4d85ce6c7457af9726875beb58af19ad1e'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _extract_verified(full_mesh, protocol):
    """Topology extraction core; public wrapper additionally verifies specimen gates."""
    X = np.asarray(full_mesh['rest_nodes_m'], dtype=float)
    cells = np.asarray(full_mesh['elements_hex8'])
    determinants, scaled = BASE.rest_jacobians(X, cells)
    if np.any(determinants <= 0) or np.any(scaled <= protocol['mesher']['mesh_quality']['minimum_rest_scaled_jacobian_exclusive']):
        raise ValueError('Original full reference Jacobians fail')
    if full_mesh['node_ids'] != list(range(1, len(X) + 1)) or full_mesh['element_ids'] != list(range(1, len(cells) + 1)):
        raise ValueError('Explicit contiguous original IDs required')
    BASE.boundary_topology(cells, full_mesh['boundaries'])
    H = protocol['geometry']['height_m']
    tolerance = 1e-10 * max(H, protocol['geometry']['radius_m'])
    z = X[cells - 1, 2]
    straddling = (z.min(axis=1) < H / 2 - tolerance) & (z.max(axis=1) > H / 2 + tolerance)
    if straddling.any():
        raise ValueError('Saved cell straddles the fixed half-height plane')
    selected = np.flatnonzero(z.max(axis=1) <= H / 2 + tolerance)
    if len(selected) * 2 != len(cells):
        raise ValueError('Whole-cell lower half must contain exactly half the original elements')
    used = np.unique(cells[selected])
    inverse = np.full(len(X) + 1, -1, dtype=int)
    inverse[used] = np.arange(1, len(used) + 1)
    half_cells = inverse[cells[selected]]
    half_nodes = X[used - 1].copy()
    face_owners = defaultdict(list)
    for cell in half_cells:
        for face in cell[BASE.HEX_FACES]:
            face_owners[tuple(sorted(map(int, face)))].append(face.tolist())
    original_side = {tuple(sorted(face)) for face in full_mesh['boundaries']['side']['faces_quad4']}
    rows = {'bottom': [], 'top': [], 'side': []}
    for owners in face_owners.values():
        if len(owners) != 1:
            continue
        face = owners[0]
        coordinates = half_nodes[np.asarray(face) - 1]
        if np.all(np.abs(coordinates[:, 2]) <= tolerance):
            name = 'bottom'
        elif np.all(np.abs(coordinates[:, 2] - H / 2) <= tolerance):
            name = 'top'
        elif tuple(sorted(used[np.asarray(face) - 1])) in original_side:
            name = 'side'
        else:
            raise ValueError('Unexpected extracted exterior face')
        rows[name].append(face)
    boundaries = {name: {'node_ids': sorted({i for face in faces for i in face}),
                         'faces_quad4': sorted(faces, key=lambda face: tuple(sorted(face)))}
                  for name, faces in rows.items()}
    topology = BASE.boundary_topology(half_cells, boundaries)
    for name in boundaries:
        boundaries[name]['faces_quad4'] = topology['outward_faces'][name]
    half = {'schema_version': 'hbe-lower-halfheight-hex8-v1', 'indexing': 'one_based',
            'mesh_N': full_mesh['mesh_N'], 'full_height_m': H,
            'full_geometry': copy.deepcopy(full_mesh['geometry']),
            'geometry': {**copy.deepcopy(full_mesh['geometry']), 'height_m': H / 2},
            'rest_nodes_m': half_nodes.tolist(), 'node_ids': list(range(1, len(used) + 1)),
            'element_ids': list(range(1, len(selected) + 1)), 'elements_hex8': half_cells.tolist(),
            'gmsh_node_ids': [full_mesh['gmsh_node_ids'][i - 1] for i in used],
            'gmsh_element_ids': [full_mesh['gmsh_element_ids'][i] for i in selected],
            'boundaries': boundaries,
            'boundary_roles': {'bottom': 'physical_bonded_plate', 'top': 'artificial_midplane',
                               'side': 'traction_free_outer'}}
    half_det, half_scaled = BASE.rest_jacobians(half_nodes, half_cells)
    if np.any(half_det <= 0) or np.any(half_scaled <= protocol['mesher']['mesh_quality']['minimum_rest_scaled_jacobian_exclusive']):
        raise ValueError('Extracted reference Jacobians fail')
    tree = cKDTree(X)
    full_cell_lookup = {tuple(sorted(row)): i for i, row in enumerate(cells)}
    if len(full_cell_lookup) != len(cells):
        raise ValueError('Duplicate full cell incidence')
    reflections = []
    for name, signs, offset in [('lower', [1, 1, 1], [0., 0., 0.]),
                                 ('upper', [1, 1, -1], [0., 0., H])]:
        query = half_nodes * signs + offset
        distances, neighbors = tree.query(query, k=2, workers=1)
        if np.any(distances[:, 0] > tolerance) or np.any(distances[:, 1] <= tolerance):
            raise ValueError('Reflected coordinates lack unique original-tolerance counterparts')
        node_ids = neighbors[:, 0] + 1
        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError('Reflection node map must be injective')
        element_ids, permutations = [], []
        for cell in half_cells:
            reflected_ids = node_ids[cell - 1]
            full_index = full_cell_lookup.get(tuple(sorted(reflected_ids)))
            if full_index is None:
                raise ValueError('Reflected whole-cell incidence missing from original mesh')
            full_cell = cells[full_index].tolist()
            permutation = [full_cell.index(int(i)) for i in reflected_ids]
            target = BASE.HEX_SIGNS[permutation]
            local_transform = BASE.HEX_SIGNS.T @ target / 8
            if (not np.array_equal(BASE.HEX_SIGNS @ local_transform, target)
                    or not np.array_equal(local_transform @ local_transform.T, np.eye(3))
                    or round(np.linalg.det(local_transform)) != int(np.prod(signs))):
                raise ValueError('Reflected local corner incidence is not the required cube symmetry')
            element_ids.append(full_index + 1)
            permutations.append(permutation)
        reflections.append({'name': name, 'signs': signs, 'rest_offset_m': offset,
                            'node_full_ids': node_ids.tolist(), 'element_full_ids': element_ids,
                            'element_half_to_full_local': permutations,
                            'maximum_coordinate_mismatch_m': float(distances[:, 0].max())})
    if reflections[0]['node_full_ids'] != used.tolist() or reflections[0]['element_full_ids'] != (selected + 1).tolist():
        raise ValueError('Identity reflection must retain the selected original IDs')
    element_coverage = np.bincount(np.concatenate([r['element_full_ids'] for r in reflections]), minlength=len(cells) + 1)[1:]
    node_coverage = np.bincount(np.concatenate([r['node_full_ids'] for r in reflections]), minlength=len(X) + 1)[1:]
    expected_node_coverage = np.where(np.abs(X[:, 2] - H / 2) <= tolerance, 2, 1)
    if not np.all(element_coverage == 1) or not np.array_equal(node_coverage, expected_node_coverage):
        raise ValueError('Reflections do not cover the full original topology exactly')
    mapping = {'schema': 'hbe-halfheight-mapping-v1', 'full_mesh_fingerprint': fingerprint(full_mesh),
               'half_mesh_fingerprint': fingerprint(half), 'half_to_full_node_ids': used.tolist(),
               'half_to_full_element_ids': (selected + 1).tolist(), 'reflections': reflections}
    verification = {'full_mesh_N': full_mesh['mesh_N'], 'full_height_m': H, 'retained_height_m': H / 2,
                    'coordinate_tolerance_m': tolerance, 'straddling_cells': 0,
                    'coordinates_unchanged': np.array_equal(half_nodes, X[used - 1]),
                    'full_nodes': len(X), 'full_elements': len(cells), 'half_nodes': len(used),
                    'half_elements': len(selected), 'midplane_nodes': int((expected_node_coverage == 2).sum()),
                    'full_node_and_element_coverage_verified': True,
                    'reflection_local_cube_symmetries_verified': True,
                    'maximum_coordinate_mismatch_m': max(r['maximum_coordinate_mismatch_m'] for r in reflections),
                    'minimum_half_rest_determinant_m3': float(half_det.min()),
                    'minimum_half_scaled_jacobian': float(half_scaled.min()),
                    'full_volume_m3': float(determinants[:, :8].sum()),
                    'half_volume_m3': float(half_det[:, :8].sum()),
                    'geometry_only_no_solver_or_equivalence_acceptance': True}
    return {'mesh': half, 'mapping': mapping, 'verification': verification}


def extract_halfheight(full_mesh, protocol):
    """Extract only the independently bound N8/N12 axial equivalence geometries."""
    N = full_mesh.get('mesh_N')
    if type(N) is not int or N not in COUNTS:
        raise ValueError('Only the declared full N8/N12 specimens are eligible')
    if (len(full_mesh['rest_nodes_m']), len(full_mesh['elements_hex8'])) != COUNTS[N]:
        raise ValueError('Original full specimen counts differ')
    if full_mesh['geometry'] != protocol['geometry'] or (protocol['geometry']['radius_m'], protocol['geometry']['height_m']) != (.004, .00489159):
        raise ValueError('Original full specimen geometry differs')
    if not BASE.mesh_quality(full_mesh, protocol)['passed']:
        raise ValueError('Original full specimen geometry gates failed')
    return _extract_verified(full_mesh, protocol)


def halfheight_deck(extraction, branch, protocol):
    """Original full-load curve, fixed physical bottom, uz=d/2/free ux,uy midplane."""
    mesh, mapping = extraction['mesh'], extraction['mapping']
    H = protocol['geometry']['height_m']
    if (branch not in ('compression', 'tension') or mesh['mesh_N'] not in COUNTS
            or protocol['loading']['base_equal_pseudotime_steps'] != 60
            or protocol['material']['reference_mu_Pa'] != 1000
            or mesh['full_height_m'] != H or mesh['geometry']['height_m'] != H / 2
            or fingerprint(mesh) != mapping['half_mesh_fingerprint']):
        raise ValueError('Exact half-height axial reference extraction required')
    xml, loading = BASE.specimen_deck(mesh, branch, 60, 1000., protocol)
    root = ET.fromstring(xml)
    boundaries = root.find('Boundary')
    retained = 0
    for bc in list(boundaries):
        if bc.attrib['node_set'].startswith('top_node_'):
            if bc.findtext('dof') in ('x', 'y'):
                boundaries.remove(bc)
            elif bc.findtext('dof') == 'z':
                bc.find('value').text = '0.5'
                retained += 1
            else:
                raise ValueError('Unexpected artificial-midplane DOF')
    if retained != len(mesh['boundaries']['top']['node_ids']):
        raise ValueError('Incomplete artificial-midplane normal constraint')
    ET.indent(root, space='  ')
    xml = ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n'
    loading.update(symmetry_model='lower_half_height_axial', full_height_m=H, retained_height_m=H / 2,
                   halfheight_equivalence_declaration_sha256=DECLARATION_SHA256,
                   prescribed_dofs={'bottom': 'xyz', 'midplane': 'z'},
                   artificial_midplane_tangential_dofs='free', full_reaction_scale=1., full_energy_scale=2.,
                   full_mesh_fingerprint=mapping['full_mesh_fingerprint'], half_mesh_fingerprint=fingerprint(mesh),
                   reconstruction_mapping_fingerprint=fingerprint(mapping),
                   deck_sha256=hashlib.sha256(xml.encode()).hexdigest())
    return xml, loading
