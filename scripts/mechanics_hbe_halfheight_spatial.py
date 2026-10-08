"""Frozen finer axial geometry/decks; pure algebra, no mesher or solver.

The public N8/N12 halfheight API is unchanged. This separate entry point only
accepts the two saved, hash-pinned N16/N24 geometries and three declared cases.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_backend as backend
from scripts import mechanics_hbe_halfheight_mesh as original

BASE = original.BASE
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-halfheight-spatial-v1.json'
DECLARATION_SHA256 = '00d38fd5fca1472b6f01543c6817cea8537f340e8c56a4466752914571c9230c'
DECLARATION_FINGERPRINT = '91fc0f0cdb1ca1f3e2d3388e0b79bf2c9604511a2e811bdae1b49735f7275e0b'
PROTOCOL_FINGERPRINT = 'b8e9db404985c1e57a5f134bd9a30da1107f09081aba347e2d4a2643498b49ff'
ORDERED_RUNS = ('compression:N24:S60:reference', 'tension:N16:S60:reference',
                'tension:N24:S60:reference')
COUNTS = {16: (7209, 6144, 4005, 3072, 801),
          24: (23101, 20736, 12439, 10368, 1777)}


def require_study(study):
    if original.fingerprint(study) != DECLARATION_FINGERPRINT:
        raise ValueError('Exact frozen spatial declaration body required')


def declaration(root, binding):
    if binding != {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256}:
        raise ValueError('Exact separately frozen spatial declaration required')
    study = access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    require_study(study)
    return study


def require_protocol(protocol):
    if original.fingerprint(protocol) != PROTOCOL_FINGERPRINT:
        raise ValueError('Original specimen protocol changed')


def extract_halfheight(full_mesh, protocol, study):
    """Unchanged whole-cell extraction after exact saved-mesh authentication."""
    require_study(study)
    require_protocol(protocol)
    N = full_mesh.get('mesh_N')
    if type(N) is not int or N not in COUNTS:
        raise ValueError('Only the saved N16/N24 geometries are eligible')
    declared = study['levels'][str(N)]
    if (original.fingerprint(full_mesh) != declared['full_mesh_fingerprint']
            or full_mesh['geometry'] != protocol['geometry']
            or (len(full_mesh['rest_nodes_m']), len(full_mesh['elements_hex8'])) != COUNTS[N][:2]):
        raise ValueError('Saved full specimen identity differs')
    # Reuse original quality mathematics, selecting this one finer level so
    # both saved geometries must satisfy the original finest volume/sag gates.
    quality_protocol = deepcopy(protocol)
    quality_protocol['mesher']['levels'] = [
        {'N': N, 'nominal_hex8_cells': COUNTS[N][1], 'expected_nodes': COUNTS[N][0]}]
    quality = BASE.mesh_quality(full_mesh, quality_protocol)
    if (quality['passed'] is not True or quality['node_count'] != COUNTS[N][0]
            or quality['connected_cell_count'] != COUNTS[N][1]):
        raise ValueError('Original full-geometry quality gates failed')
    extraction = original._extract_verified(full_mesh, protocol)
    if any(extraction['verification'][name] != value
           for name, value in declared['expected_counts'].items()):
        raise ValueError('Exact finer half-height counts differ')
    return extraction


def load_extraction(root, study, N):
    require_study(study)
    if type(N) is not int or N not in COUNTS:
        raise ValueError('Undeclared spatial level')
    full = access.verify_binding(root, study['levels'][str(N)]['full_mesh'],
                                 maximum_bytes=16*1024**2, read_json=True)
    protocol = access.verify_binding(root, study['original_protocol'],
                                     maximum_bytes=1024**2, read_json=True)
    return extract_halfheight(full, protocol, study)


def halfheight_deck(extraction, branch, protocol, study):
    """Apply the accepted d/2/free-tangential cut to the original deck builder."""
    require_study(study)
    require_protocol(protocol)
    mesh, mapping = extraction['mesh'], extraction['mapping']
    N = mesh['mesh_N']
    key = f'{branch}:N{N}:S60:reference'
    if (type(N) is not int or N not in COUNTS or key not in ORDERED_RUNS
            or mapping['full_mesh_fingerprint'] != study['levels'][str(N)]['full_mesh_fingerprint']
            or original.fingerprint(mesh) != mapping['half_mesh_fingerprint']
            or mesh['full_geometry'] != protocol['geometry']
            or mesh['full_height_m'] != protocol['geometry']['height_m']
            or mesh['geometry'] != dict(protocol['geometry'], height_m=protocol['geometry']['height_m']/2)):
        raise ValueError('Exact declared finer half-height case required')
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
    loading.update(symmetry_model='lower_half_height_axial',
        full_height_m=protocol['geometry']['height_m'], retained_height_m=mesh['geometry']['height_m'],
        halfheight_spatial_declaration_sha256=DECLARATION_SHA256,
        prescribed_dofs={'bottom': 'xyz', 'midplane': 'z'}, artificial_midplane_tangential_dofs='free',
        full_reaction_scale=1., full_energy_scale=2., full_mesh_fingerprint=mapping['full_mesh_fingerprint'],
        half_mesh_fingerprint=original.fingerprint(mesh),
        reconstruction_mapping_fingerprint=original.fingerprint(mapping),
        deck_sha256=hashlib.sha256(xml.encode()).hexdigest())
    return xml, loading


def reconstruction(study_binding, source_bindings, extraction, full_binding, half_binding):
    return {'schema': 'hbe-halfheight-spatial-reconstruction-v1',
            'mapping': extraction['mapping'], 'verification': extraction['verification'],
            'full_mesh': full_binding, 'half_mesh': half_binding, 'declaration': study_binding,
            'mesh_source': source_bindings['halfheight_spatial'],
            'extraction_source': source_bindings['halfheight_mesh']}


def case_contents(study, protocol, extraction, branch, full_binding, level):
    xml, loading = halfheight_deck(extraction, branch, protocol, study)
    adapted = backend.transform_deck(xml.encode())
    loading.update(protocol_sha256=study['original_protocol']['sha256'],
        mesh_sha256=level['mesh']['sha256'], full_mesh_sha256=full_binding['sha256'],
        reconstruction_sha256=level['reconstruction']['sha256'],
        deck_sha256=hashlib.sha256(adapted.encode()).hexdigest())
    return xml, adapted, loading


def verify_prepared_case(root, study_binding, half_bindings, reconstruction_binding, branch, N):
    """Rebuild exact geometry, mapping, deck and loading; no response access."""
    study = declaration(root, study_binding)
    if f'{branch}:N{N}:S60:reference' not in ORDERED_RUNS or type(N) is not int:
        raise ValueError('Undeclared spatial case')
    full_binding = study['levels'][str(N)]['full_mesh']
    wrapper = access.verify_binding(root, reconstruction_binding, maximum_bytes=16*1024**2, read_json=True)
    if (set(wrapper) != {'schema', 'mapping', 'verification', 'full_mesh', 'half_mesh', 'declaration',
                        'mesh_source', 'extraction_source'}
            or wrapper['full_mesh'] != full_binding or wrapper['half_mesh'] != half_bindings['mesh']
            or wrapper['declaration'] != study_binding
            or wrapper['schema'] != 'hbe-halfheight-spatial-reconstruction-v1'
            or wrapper['extraction_source']['sha256'] != study['inherited_source_sha256']['halfheight_mesh']
            or access.local_path(root, wrapper['mesh_source']['path']) != Path(__file__).resolve()):
        raise ValueError('Prepared finer reconstruction provenance differs')
    for binding in (wrapper['mesh_source'], wrapper['extraction_source']):
        access.verify_binding(root, binding, maximum_bytes=1024**2)
    if access.local_path(root, wrapper['extraction_source']['path']) != Path(original.__file__).resolve():
        raise ValueError('Imported extraction source differs')
    extraction = load_extraction(root, study, N)
    expected = reconstruction(study_binding,
        {'halfheight_spatial': wrapper['mesh_source'], 'halfheight_mesh': wrapper['extraction_source']},
        extraction, full_binding, half_bindings['mesh'])
    half = access.verify_binding(root, half_bindings['mesh'], maximum_bytes=16*1024**2, read_json=True)
    if access.canonical_json(wrapper) != access.canonical_json(expected) or half != extraction['mesh']:
        raise ValueError('Prepared geometry or mapping differs from exact saved-cell extraction')
    protocol = access.verify_binding(root, study['original_protocol'], maximum_bytes=1024**2, read_json=True)
    level = {'mesh': half_bindings['mesh'], 'reconstruction': reconstruction_binding}
    _, adapted, loading = case_contents(study, protocol, extraction, branch, full_binding, level)
    access.verify_binding(root, half_bindings['deck'], maximum_bytes=16*1024**2)
    if access.local_path(root, half_bindings['deck']['path']).read_bytes() != adapted.encode():
        raise ValueError('Deck differs from the exact original mechanics and accepted half-height cut')
    actual = access.verify_binding(root, half_bindings['loading'], maximum_bytes=1024**2, read_json=True)
    if access.canonical_json(actual) != access.canonical_json(loading):
        raise ValueError('Prepared loading identity differs')
    return study, wrapper, half
