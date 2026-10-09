"""Read-only HBE v5 source/mesh and fixture-topology gate.

This module cannot release or run FEBio. It checks old-domain source bytes only;
the v5 schedule adapter and all later native/physical qualifications stay separate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import mechanics_hbe_branch_calibration_v5 as v5


BINDING_PATH = 'manifests/experiments/hbe-01-03-v5-source-deck-bindings-v1.json'
BINDING_SHA256 = '65ab3d9b2056e43e3a193b6d0288078f3617159483e271d0ab3cd0d624fb02e0'
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_MESH_BYTES = 32 * 1024 * 1024
EXPECTED_MESH_KEYS = {'full:N8', 'full:N12', 'full:N16', 'half:N16',
                      'half:N24', 'half:N32', 'half:N36'}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bound(root: Path, binding: dict, *, maximum: int) -> bytes:
    if set(binding) != {'path', 'sha256'}:
        raise ValueError('Exact path/hash binding required')
    relative = Path(binding['path'])
    base = root.resolve()
    candidate = base / relative
    path = candidate.resolve()
    if (relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(base)
            or any(part.is_symlink() for part in (candidate, *candidate.parents)
                   if part != base and part.is_relative_to(base))
            or not path.is_file() or path.stat().st_size > maximum):
        raise ValueError('Bound source missing, unsafe, or exceeds reader limit')
    data = path.read_bytes()
    if _sha(data) != binding['sha256']:
        raise ValueError('Bound source SHA256 differs')
    return data


def _unique(root: ET.Element, path: str) -> ET.Element:
    found = root.findall(path)
    if len(found) != 1:
        raise ValueError('Exactly one FEBio element required: ' + path)
    return found[0]


def _ids(text: str | None) -> list[int]:
    if not text or not text.strip():
        raise ValueError('Empty node/element reference list')
    try:
        values = [int(part.strip()) for part in text.split(',')]
    except ValueError as exc:
        raise ValueError('Noninteger node/element reference') from exc
    if any(value < 1 for value in values):
        raise ValueError('Positive one-based node/element IDs required')
    return values


def _boundary_nodes(mesh: dict, name: str, node_ids: set[int]) -> list[int]:
    boundary = mesh['boundaries'][name]
    ids = boundary['node_ids']
    faces = boundary['faces_quad4']
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or not isinstance(faces, list) or not faces
            or any(len(face) != 4 or len(set(face)) != 4 for face in faces)
            or any(type(node) is not int or node not in node_ids for node in ids)
            or any(type(node) is not int for face in faces for node in face)
            or {node for face in faces for node in face} != set(ids)):
        raise ValueError('Mesh boundary face/node membership differs')
    return ids


def validate_topology_bc(mesh: dict, source_xml: bytes, *, native_domain: str) -> dict:
    """Pure check of complete FEBio mesh geometry and axial fixture BCs.

    The check compares exact hash-bound mesh arrays to deck XML. A half-domain
    top is an artificial symmetry midplane: z only at half displacement; its
    tangential DOFs remain free. A full-domain top is bonded in x/y/z.
    """
    if native_domain not in {'full_native', 'lower_half_reconstructed'}:
        raise ValueError('Unknown native-domain representation')
    if (not isinstance(source_xml, bytes) or len(source_xml) > MAX_SOURCE_BYTES
            or b'<!DOCTYPE' in source_xml or b'<!ENTITY' in source_xml):
        raise ValueError('Bounded plain FEBio XML required')
    expected_schema = ('hbe-specimen-hex8-v1' if native_domain == 'full_native'
                       else 'hbe-lower-halfheight-hex8-v1')
    if (mesh.get('schema_version') != expected_schema or mesh.get('indexing') != 'one_based'
            or set(mesh.get('boundaries', {})) != {'bottom', 'top', 'side'}):
        raise ValueError('Mesh representation or boundary inventory differs')
    if native_domain == 'lower_half_reconstructed':
        if mesh.get('boundary_roles') != {'bottom': 'physical_bonded_plate',
                                          'top': 'artificial_midplane',
                                          'side': 'traction_free_outer'}:
            raise ValueError('Artificial midplane role differs')
    elif mesh.get('boundary_roles') not in (None, {'bottom': 'physical_bonded_plate',
                                                  'top': 'physical_bonded_plate',
                                                  'side': 'traction_free_outer'}):
        raise ValueError('Full physical plate roles differ')
    node_ids, coordinates = mesh['node_ids'], mesh['rest_nodes_m']
    element_ids, elements = mesh['element_ids'], mesh['elements_hex8']
    if (not node_ids or not element_ids or len(node_ids) != len(coordinates)
            or len(element_ids) != len(elements) or len(node_ids) != len(set(node_ids))
            or len(element_ids) != len(set(element_ids))
            or any(type(i) is not int or i < 1 for i in node_ids + element_ids)
            or any(len(point) != 3 or not all(math.isfinite(float(v)) for v in point)
                   for point in coordinates)):
        raise ValueError('Mesh node/element identities invalid')
    nodes = set(node_ids)
    if any(len(cell) != 8 or len(set(cell)) != 8 or not set(cell) <= nodes for cell in elements):
        raise ValueError('Hex8 connectivity invalid')
    boundaries = {name: _boundary_nodes(mesh, name, nodes) for name in ('bottom', 'top', 'side')}
    if set(boundaries['bottom']) & set(boundaries['top']):
        raise ValueError('Top and bottom plates overlap')
    tree = ET.fromstring(source_xml)
    if tree.tag != 'febio_spec' or len(tree.findall('Mesh')) != 1:
        raise ValueError('FEBio mesh root differs')
    xml_nodes = _unique(tree, 'Mesh/Nodes')
    xml_elements = _unique(tree, 'Mesh/Elements')
    if (xml_nodes.attrib != {'name': 'specimen_nodes'}
            or xml_elements.attrib != {'type': 'hex8', 'name': 'specimen'}):
        raise ValueError('Hex8 FEBio elements required')
    if ([int(x.attrib['id']) for x in xml_nodes] != node_ids
            or any(x.tag != 'node' or set(x.attrib) != {'id'} or len(x) for x in xml_nodes)
            or [[float(part) for part in x.text.split(',')] for x in xml_nodes] != coordinates
            or [int(x.attrib['id']) for x in xml_elements] != element_ids
            or any(x.tag != 'elem' or set(x.attrib) != {'id'} or len(x) for x in xml_elements)
            or [_ids(x.text) for x in xml_elements] != elements):
        raise ValueError('FEBio nodes or hex8 elements differ from mesh')
    named_sets = tree.findall('Mesh/NodeSet')
    names = [row.attrib.get('name') for row in named_sets]
    top_singletons = {f'top_node_{node}' for node in boundaries['top']}
    if (len(names) != len(set(names)) or set(names) != {'bottom', 'top', 'side'} | top_singletons
            or any(set(row.attrib) != {'name'} or len(row) for row in named_sets)):
        raise ValueError('Complete unique boundary and top-singleton NodeSets required')
    by_name = dict(zip(names, named_sets))
    for name, ids in boundaries.items():
        if _ids(by_name[name].text) != ids:
            raise ValueError('FEBio boundary NodeSet differs from mesh')
    for node in boundaries['top']:
        if _ids(by_name[f'top_node_{node}'].text) != [node]:
            raise ValueError('Top singleton NodeSet differs from mesh')
    surfaces = tree.findall('Mesh/Surface')
    if (len(surfaces) != 3 or {s.attrib.get('name') for s in surfaces} != set(boundaries)
            or any(set(s.attrib) != {'name'} for s in surfaces)):
        raise ValueError('Complete unique boundary surfaces required')
    for surface in surfaces:
        expected = mesh['boundaries'][surface.attrib['name']]['faces_quad4']
        if (len(surface) != len(expected) or any(face.tag != 'quad4' or set(face.attrib) != {'id'}
                                                or len(face) for face in surface)
                or [int(face.attrib['id']) for face in surface] != list(range(1, len(expected)+1))
                or [_ids(face.text) for face in surface] != expected):
            raise ValueError('FEBio boundary faces differ from mesh')
    controllers = tree.findall('LoadData/load_controller')
    if len(controllers) != 2 or len(tree.findall('.//load_controller')) != 2:
        raise ValueError('Exactly two load controllers required')
    zero, axial = [], []
    for controller in controllers:
        if controller.attrib.get('type') != 'loadcurve':
            raise ValueError('Load controller type differs')
        values = [float(pt.text.split(',')[1]) for pt in _unique(controller, 'points')]
        if len(values) != 61 or not all(math.isfinite(value) for value in values):
            raise ValueError('Old-domain controller values differ')
        (zero if all(value == 0 for value in values) else axial).append(controller.attrib.get('id'))
    if len(zero) != 1 or len(axial) != 1 or not zero[0] or not axial[0] or zero[0] == axial[0]:
        raise ValueError('Distinct zero/axial controllers required')
    expected_bcs = {('bottom', axis, zero[0], 1.0) for axis in 'xyz'}
    factor = 0.5 if native_domain == 'lower_half_reconstructed' else 1.0
    for node in boundaries['top']:
        for axis in ('z',) if factor == 0.5 else 'xyz':
            expected_bcs.add((f'top_node_{node}', axis, axial[0] if axis == 'z' else zero[0],
                              factor if axis == 'z' else 1.0))
    found_bcs = []
    for bc in tree.findall('Boundary/bc'):
        value = _unique(bc, 'value')
        if (set(bc.attrib) != {'type', 'node_set'}
                or bc.attrib.get('type') != 'prescribed displacement'
                or [child.tag for child in bc] != ['dof', 'value', 'relative']
                or set(value.attrib) != {'lc'} or bc.findtext('relative') != '0'):
            raise ValueError('Only absolute prescribed displacement BCs allowed')
        found_bcs.append((bc.attrib.get('node_set'), bc.findtext('dof'), value.attrib.get('lc'),
                          float(value.text)))
    if len(found_bcs) != len(set(found_bcs)) or set(found_bcs) != expected_bcs:
        raise ValueError('Incomplete, duplicate, or extra fixture BCs')
    if len(tree.findall('.//bc')) != len(found_bcs):
        raise ValueError('BC outside expected FEBio Boundary block')
    return {'node_count': len(node_ids), 'hex8_count': len(element_ids),
            'top_node_count': len(boundaries['top']), 'bottom_node_count': len(boundaries['bottom']),
            'bc_count': len(found_bcs), 'native_domain': native_domain}


def validate_binding_manifest(root: Path, *, inspect_sources: bool = True) -> dict:
    """Verify frozen manifest and, optionally, all ten actual source/mesh pairs.

    No output is written. This is source preparation, never execution release.
    """
    root = Path(root)
    raw = _read_bound(root, {'path': BINDING_PATH, 'sha256': BINDING_SHA256}, maximum=64*1024)
    binding = json.loads(raw)
    study, _ = v5.validate_preparation(root)
    expected_sources = {f'{branch}:N{n}' for branch, n, _, _ in v5.RUNS}
    expected_rows = {row['run_id']: f"{row['branch']}:N{row['N']}"
                     for row in study['ordered_reference_runs']}
    if (binding.get('schema') != 'hbe-v5-source-deck-bindings-v1'
            or binding.get('status') != 'non_executable_source_and_mesh_identity_preparation'
            or binding.get('specimen') != 'HBE_01_03'
            or binding.get('v5_declaration') != {'path': v5.DECLARATION_PATH,
                                                  'sha256': v5.DECLARATION_SHA256}
            or binding.get('phase_gates') != dict.fromkeys(
                ('native_execution', 'automatic_reference_release', 'fit', 'held_out_torque_access'), False)
            or set(binding.get('meshes', {})) != EXPECTED_MESH_KEYS
            or set(binding.get('source_decks', {})) != expected_sources
            or binding.get('run_source_keys') != expected_rows):
        raise ValueError('Frozen non-executable source binding inventory differs')
    for mesh_key, item in binding['meshes'].items():
        n = int(mesh_key.split('N')[1])
        form = 'full' if mesh_key.startswith('full:') else 'half'
        expected_sha = study['mesh_identities'][str(n)][f'{form}_mesh_sha256']
        if set(item) != {'path', 'sha256'} or item['sha256'] != expected_sha:
            raise ValueError('Representation-specific mesh hash differs from v5')
    for source_key, item in binding['source_decks'].items():
        if set(item) != {'path', 'sha256', 'mesh'} or item['mesh'] not in EXPECTED_MESH_KEYS:
            raise ValueError('Complete source/mesh pair required')
        branch, size = source_key.split(':N')
        native = next(row['native_domain'] for row in study['ordered_reference_runs']
                      if row['branch'] == branch and row['N'] == int(size))
        expected_key = ('full' if native == 'full_native' else 'half') + ':N' + size
        if item['mesh'] != expected_key:
            raise ValueError('Source bound to wrong native mesh representation')
    if not inspect_sources:
        return {'status': 'non_executable_manifest_checked', 'source_decks': 10,
                'meshes': 7, 'rows': 12}
    mesh_bytes = {key: _read_bound(root, item, maximum=MAX_MESH_BYTES)
                  for key, item in binding['meshes'].items()}
    checks = {}
    for source_key, item in binding['source_decks'].items():
        raw_source = _read_bound(root, {'path': item['path'], 'sha256': item['sha256']},
                                 maximum=MAX_SOURCE_BYTES)
        domain = 'full_native' if item['mesh'].startswith('full:') else 'lower_half_reconstructed'
        checks[source_key] = validate_topology_bc(json.loads(mesh_bytes[item['mesh']]),
                                                   raw_source, native_domain=domain)
    return {'status': 'non_executable_source_topology_checked', 'source_decks': len(checks),
            'meshes': len(mesh_bytes), 'rows': len(expected_rows), 'checks': checks}


def require_execution_ready(*args, **kwargs):
    raise ValueError('Source/mesh/BC consistency is not a native execution release')
