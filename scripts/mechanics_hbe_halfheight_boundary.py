"""Fixed-polygon N24 slab subdivision; geometry algebra, never a mesher/solver.

Original specimen coordinates and unaffected cell slots are retained. Generated
entities carry source-edge/parent ancestry, never invented Gmsh source IDs.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from scripts import mechanics_hbe_halfheight_spatial as spatial

original, access, backend, BASE = spatial.original, spatial.access, spatial.backend, spatial.BASE
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-halfheight-boundary-v1.json'
DECLARATION_SHA256 = '5a0f355b73085f0ae0ffc677a16b9054bc8b40ee2f8a4f2da9db8800acf3a5bc'
DECLARATION_FINGERPRINT = 'abdd04e02ab1fad4993cc9ed87fe592757945e7a1d5d7626359e67736f19b112'
ORDERED_VARIANTS = ('P2', 'I2', 'P4')
ORDERED_RUNS = tuple(f'compression:N24:{v}:S60:reference' for v in ORDERED_VARIANTS)


def require_study(study):
    if original.fingerprint(study) != DECLARATION_FINGERPRINT:
        raise ValueError('Exact frozen boundary diagnostic required')


def declaration(root, binding):
    if binding != {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256}:
        raise ValueError('Exact separately frozen boundary declaration required')
    study = access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    require_study(study)
    return study


def variant_of(key):
    if key not in ORDERED_RUNS:
        raise ValueError('Undeclared boundary case')
    return key.split(':')[2]


def _boundaries(X, cells, H, R, tolerance):
    owners = defaultdict(list)
    for cell in cells:
        for face in cell[BASE.HEX_FACES]:
            owners[tuple(sorted(map(int, face)))].append(face.tolist())
    rows = {name: [] for name in ('bottom', 'top', 'side')}
    for faces in owners.values():
        if len(faces) == 2:
            continue
        if len(faces) != 1:
            raise ValueError('Nonmanifold subdivision interface')
        face = faces[0]
        Q = X[np.asarray(face)-1]
        if np.all(np.abs(Q[:, 2]) <= tolerance):
            name = 'bottom'
        elif np.all(np.abs(Q[:, 2]-H) <= tolerance):
            name = 'top'
        elif np.all(np.abs(np.linalg.norm(Q[:, :2], axis=1)-R) <= tolerance):
            name = 'side'
        else:
            raise ValueError('Unshared internal face or changed outer polygon')
        rows[name].append(face)
    boundaries = {name: {'node_ids': sorted({i for face in faces for i in face}),
                         'faces_quad4': sorted(faces, key=lambda f: tuple(sorted(f)))}
                  for name, faces in rows.items()}
    topology = BASE.boundary_topology(cells, boundaries)
    for name in boundaries:
        boundaries[name]['faces_quad4'] = topology['outward_faces'][name]
    return boundaries


def subdivide(full, protocol, spatial_study, study, variant):
    """Authenticate original geometry before creating a declared nested variant."""
    require_study(study)
    spatial.require_protocol(protocol)
    if variant not in ORDERED_VARIANTS:
        raise ValueError('Only P2/I2/P4 subdivision is declared')
    if original.fingerprint(full) != study['full_mesh_fingerprint']:
        raise ValueError('Exact saved N24 geometry required')
    spatial.extract_halfheight(full, protocol, spatial_study)
    spec = study['variants'][variant]
    H, R = protocol['geometry']['height_m'], protocol['geometry']['radius_m']
    tolerance = study['transformation']['coordinate_tolerance_m']
    X = np.asarray(full['rest_nodes_m'], float)
    C = np.asarray(full['elements_hex8'], int)
    z = X[C-1, 2]
    lo, hi = z.min(axis=1), z.max(axis=1)
    a, b = spec['slab_index_from_bottom']/12, (spec['slab_index_from_bottom']+1)/12
    selected = ((np.abs(lo-a*H) <= tolerance) & (np.abs(hi-b*H) <= tolerance)) | (
        (np.abs(lo-(1-b)*H) <= tolerance) & (np.abs(hi-(1-a)*H) <= tolerance))
    if int(selected.sum()) != 3456:
        raise ValueError('Exactly two reflected 1728-cell slabs required')
    nodes, cells = X.tolist(), C.tolist()
    node_ancestry, cache = [], {}
    parent_ids = list(range(1, len(C)+1))
    source_elements = list(full['gmsh_element_ids'])
    q = spec['subdivisions']
    for parent in np.flatnonzero(selected):
        cell = C[parent]
        coordinates = X[cell-1]
        lower = np.flatnonzero(np.abs(coordinates[:, 2]-lo[parent]) <= tolerance)
        upper = np.flatnonzero(np.abs(coordinates[:, 2]-hi[parent]) <= tolerance)
        if len(lower) != 4 or len(upper) != 4:
            raise ValueError('Saved cell is not a two-plane extrusion')
        edges = {}
        for i in lower:
            matches = [j for j in upper if np.array_equal(coordinates[i, :2], coordinates[j, :2])]
            if len(matches) != 1:
                raise ValueError('Vertical source edge must retain exact x/y')
            j = matches[0]
            edges[i] = edges[j] = (int(cell[i]), int(cell[j]))

        def at(edge, numerator):
            if numerator == 0:
                return edge[0]
            if numerator == q:
                return edge[1]
            key = (*edge, numerator, q)
            if key not in cache:
                A, B = X[np.asarray(edge)-1]
                point = [float(A[0]), float(A[1]), float(A[2]+(B[2]-A[2])*numerator/q)]
                nodes.append(point)
                cache[key] = len(nodes)
                node_ancestry.append({'node_id': len(nodes), 'source_edge_node_ids': list(edge),
                                      'fraction_numerator': numerator, 'fraction_denominator': q})
            return cache[key]

        for child in range(q):
            row = [at(edges[i], child if i in lower else child+1) for i in range(8)]
            if child == 0:
                cells[parent] = row
                source_elements[parent] = None
            else:
                cells.append(row)
                parent_ids.append(int(parent)+1)
                source_elements.append(None)
    Y, D = np.asarray(nodes), np.asarray(cells)
    if (len(Y), len(D)) != (spec['full_nodes'], spec['full_elements']):
        raise ValueError('Exact declared expanded full geometry counts differ')
    if not np.array_equal(Y[:len(X)], X) or not np.array_equal(D[:len(C)][~selected], C[~selected]):
        raise ValueError('Original nodes or unaffected cell slots changed')
    if len(np.unique(np.rint(Y/(R*1e-12)).astype(np.int64), axis=0)) != len(Y):
        raise ValueError('Coincident nodes conceal unshared subdivision faces')
    determinants, scaled = BASE.rest_jacobians(Y, D)
    parent_det, _ = BASE.rest_jacobians(X, C)
    volumes, parent_volumes = determinants[:, :8].sum(axis=1), parent_det[:, :8].sum(axis=1)
    reconstructed = np.bincount(np.asarray(parent_ids), weights=volumes, minlength=len(C)+1)[1:]
    volume_limits = 1e-24 + study['transformation']['relative_parent_volume_tolerance']*parent_volumes
    if np.any(np.abs(reconstructed-parent_volumes) > volume_limits):
        raise ValueError('Parent-child quadrature volume is not conserved')
    if np.any(determinants <= 0) or np.min(scaled) <= protocol['mesher']['mesh_quality']['minimum_rest_scaled_jacobian_exclusive']:
        raise ValueError('Original positive/scaled Jacobian gates failed')
    boundaries = _boundaries(Y, D, H, R, tolerance)
    for name in ('bottom', 'top'):
        old_boundary = full['boundaries'][name]
        if (boundaries[name]['node_ids'] != old_boundary['node_ids'] or
                {tuple(sorted(f)) for f in boundaries[name]['faces_quad4']} !=
                {tuple(sorted(f)) for f in old_boundary['faces_quad4']}):
            raise ValueError('Physical end-face connectivity changed')
        # Preserve source ordering as well as incidence. The topology check
        # below independently verifies its outward cyclic orientation.
        boundaries[name] = deepcopy(old_boundary)
    BASE.boundary_topology(D, boundaries)
    result = deepcopy(full)
    result.update(schema_version='hbe-fixed-N24-slab-subdivision-v1', rest_nodes_m=nodes,
        node_ids=list(range(1, len(Y)+1)), elements_hex8=cells, element_ids=list(range(1, len(D)+1)),
        gmsh_node_ids=full['gmsh_node_ids']+[None]*len(node_ancestry), gmsh_element_ids=source_elements,
        boundaries=boundaries,
        subdivision={'variant': variant, 'declaration_sha256': DECLARATION_SHA256,
            'source_full_mesh': study['full_mesh'], 'generated_node_ancestry': node_ancestry,
            'cell_parent_source_ids': parent_ids, 'replaced_source_cell_ids': (np.flatnonzero(selected)+1).tolist(),
            'source_id_note': 'null means algebraically subdivided, not generated by Gmsh'})
    extraction = original._extract_verified(result, protocol)
    if any(extraction['verification'][name] != spec[name] for name in
           ('full_nodes', 'full_elements', 'half_nodes', 'half_elements', 'midplane_nodes')):
        raise ValueError('Exact half/reconstruction variant counts differ')
    proof = {'variant': variant, 'original_nodes_unchanged': True, 'unaffected_cell_slots_unchanged': True,
             'unchanged_original_xy_and_z_planes': True, 'unchanged_polygon_and_end_faces': True,
             'shared_face_conformity': True, 'parent_child_volume_conservation': True,
             'maximum_parent_volume_error_ratio': float(np.max(np.abs(reconstructed-parent_volumes)/volume_limits)),
             'source_volume_m3': float(parent_volumes.sum()), 'variant_volume_m3': float(volumes.sum()),
             'minimum_scaled_jacobian': float(scaled.min()), 'generated_nodes': len(node_ancestry),
             'generated_cells_net': len(D)-len(C), 'gmsh_generation_calls': 0,
             'numerical_geometry_only': True}
    return {'full_mesh': result, **extraction, 'subdivision_verification': proof}


def load_variant(root, study, variant):
    require_study(study)
    prior = spatial.declaration(root, study['original_spatial_declaration'])
    full = access.verify_binding(root, study['full_mesh'], maximum_bytes=16*1024**2, read_json=True)
    protocol = access.verify_binding(root, study['original_protocol'], maximum_bytes=1024**2, read_json=True)
    return subdivide(full, protocol, prior, study, variant)


def halfheight_deck(extraction, protocol, study, variant):
    require_study(study)
    spatial.require_protocol(protocol)
    if variant not in ORDERED_VARIANTS or extraction['subdivision_verification']['variant'] != variant:
        raise ValueError('Declared prepared variant required')
    # Reuse the accepted deck mathematics; this private adapter only changes
    # artificial-midplane constraints exactly as both earlier studies did.
    xml, loading = BASE.specimen_deck(extraction['mesh'], 'compression', 60, 1000., protocol)
    root = ET.fromstring(xml)
    boundaries, retained = root.find('Boundary'), 0
    for bc in list(boundaries):
        if bc.attrib['node_set'].startswith('top_node_'):
            if bc.findtext('dof') in ('x', 'y'):
                boundaries.remove(bc)
            elif bc.findtext('dof') == 'z':
                bc.find('value').text = '0.5'
                retained += 1
            else:
                raise ValueError('Unexpected artificial-midplane DOF')
    if retained != study['variants'][variant]['midplane_nodes']:
        raise ValueError('Incomplete artificial-midplane normal constraint')
    ET.indent(root, space='  ')
    xml = ET.tostring(root, encoding='unicode', xml_declaration=True)+'\n'
    loading.update(symmetry_model='lower_half_height_axial', variant=variant,
        full_height_m=protocol['geometry']['height_m'], retained_height_m=protocol['geometry']['height_m']/2,
        halfheight_boundary_declaration_sha256=DECLARATION_SHA256,
        prescribed_dofs={'bottom': 'xyz', 'midplane': 'z'}, artificial_midplane_tangential_dofs='free',
        full_reaction_scale=1., full_energy_scale=2.,
        full_mesh_fingerprint=original.fingerprint(extraction['full_mesh']),
        half_mesh_fingerprint=original.fingerprint(extraction['mesh']),
        reconstruction_mapping_fingerprint=original.fingerprint(extraction['mapping']),
        deck_sha256=hashlib.sha256(xml.encode()).hexdigest())
    return xml, loading


def reconstruction(study_binding, sources, extraction, level, variant):
    return {'schema': 'hbe-halfheight-boundary-reconstruction-v1', 'variant': variant,
            'mapping': extraction['mapping'], 'verification': extraction['verification'],
            'subdivision_verification': extraction['subdivision_verification'],
            'full_mesh': level['full_mesh'], 'half_mesh': level['mesh'], 'declaration': study_binding,
            'mesh_source': sources['halfheight_boundary'], 'extraction_source': sources['halfheight_mesh']}


def case_contents(study, protocol, extraction, variant, level):
    xml, loading = halfheight_deck(extraction, protocol, study, variant)
    adapted = backend.transform_deck(xml.encode())
    loading.update(protocol_sha256=study['original_protocol']['sha256'], mesh_sha256=level['mesh']['sha256'],
        full_mesh_sha256=level['full_mesh']['sha256'], reconstruction_sha256=level['reconstruction']['sha256'],
        deck_sha256=hashlib.sha256(adapted.encode()).hexdigest())
    return xml, adapted, loading


def verify_prepared_case(root, study_binding, primitives, reconstruction_binding, variant):
    study = declaration(root, study_binding)
    if variant not in ORDERED_VARIANTS:
        raise ValueError('Undeclared variant')
    wrapper = access.verify_binding(root, reconstruction_binding, maximum_bytes=32*1024**2, read_json=True)
    if (set(wrapper) != {'schema', 'variant', 'mapping', 'verification', 'subdivision_verification',
                        'full_mesh', 'half_mesh', 'declaration', 'mesh_source', 'extraction_source'}
            or wrapper['schema'] != 'hbe-halfheight-boundary-reconstruction-v1'
            or wrapper['variant'] != variant or wrapper['declaration'] != study_binding
            or wrapper['half_mesh'] != primitives['mesh']
            or wrapper['extraction_source']['sha256'] != study['inherited_source_sha256']['halfheight_mesh']
            or access.local_path(root, wrapper['mesh_source']['path']) != Path(__file__).resolve()
            or access.local_path(root, wrapper['extraction_source']['path']) != Path(original.__file__).resolve()):
        raise ValueError('Prepared subdivision source/provenance differs')
    for name in ('mesh_source', 'extraction_source'):
        access.verify_binding(root, wrapper[name], maximum_bytes=1024**2)
    actual = load_variant(root, study, variant)
    level = {'full_mesh': wrapper['full_mesh'], 'mesh': primitives['mesh'], 'reconstruction': reconstruction_binding}
    expected = reconstruction(study_binding, {'halfheight_boundary': wrapper['mesh_source'],
        'halfheight_mesh': wrapper['extraction_source']}, actual, level, variant)
    if access.canonical_json(wrapper) != access.canonical_json(expected):
        raise ValueError('Prepared mapping differs from authenticated subdivision')
    for name, key in (('full_mesh', 'full_mesh'), ('mesh', 'mesh')):
        saved = access.verify_binding(root, level[name], maximum_bytes=32*1024**2, read_json=True)
        if saved != actual[key]:
            raise ValueError('Prepared geometry differs from exact subdivision')
    protocol = access.verify_binding(root, study['original_protocol'], maximum_bytes=1024**2, read_json=True)
    _, deck, loading = case_contents(study, protocol, actual, variant, level)
    access.verify_binding(root, primitives['deck'], maximum_bytes=32*1024**2)
    if access.local_path(root, primitives['deck']['path']).read_bytes() != deck.encode():
        raise ValueError('Prepared deck changed physics or boundary conditions')
    saved_loading = access.verify_binding(root, primitives['loading'], maximum_bytes=1024**2, read_json=True)
    if access.canonical_json(saved_loading) != access.canonical_json(loading):
        raise ValueError('Prepared loading identity differs')
    return study, wrapper, actual['mesh'], actual['full_mesh']
