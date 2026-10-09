"""Synthetic topology controls; no measured data or native solver is used."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

from scripts import mechanics_hbe_v5_source_bindings as gate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_native_execution(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('No native execution'))


def fixture(native_domain='full_native'):
    half = native_domain == 'lower_half_reconstructed'
    points = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
              [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]]
    faces = {'bottom': [[1, 2, 3, 4]], 'top': [[5, 6, 7, 8]],
             'side': [[1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8]]}
    mesh = {'schema_version': ('hbe-lower-halfheight-hex8-v1' if half
                               else 'hbe-specimen-hex8-v1'), 'indexing': 'one_based',
            'node_ids': list(range(1, 9)), 'rest_nodes_m': points,
            'element_ids': [1], 'elements_hex8': [list(range(1, 9))],
            'boundaries': {name: {'node_ids': sorted({node for face in rows for node in face}),
                                  'faces_quad4': rows} for name, rows in faces.items()}}
    if half:
        mesh['boundary_roles'] = {'bottom': 'physical_bonded_plate',
                                  'top': 'artificial_midplane', 'side': 'traction_free_outer'}
    root = ET.Element('febio_spec', version='4.0')
    xmlmesh = ET.SubElement(root, 'Mesh')
    nodes = ET.SubElement(xmlmesh, 'Nodes', name='specimen_nodes')
    for node, point in zip(mesh['node_ids'], points):
        ET.SubElement(nodes, 'node', id=str(node)).text = ','.join(map(str, point))
    elems = ET.SubElement(xmlmesh, 'Elements', type='hex8', name='specimen')
    ET.SubElement(elems, 'elem', id='1').text = '1,2,3,4,5,6,7,8'
    for name, data in mesh['boundaries'].items():
        ET.SubElement(xmlmesh, 'NodeSet', name=name).text = ','.join(map(str, data['node_ids']))
        surface = ET.SubElement(xmlmesh, 'Surface', name=name)
        for i, face in enumerate(data['faces_quad4'], 1):
            ET.SubElement(surface, 'quad4', id=str(i)).text = ','.join(map(str, face))
    for node in mesh['boundaries']['top']['node_ids']:
        ET.SubElement(xmlmesh, 'NodeSet', name=f'top_node_{node}').text = str(node)
    bcs = ET.SubElement(root, 'Boundary')

    def bc(name, axis, lc, value):
        item = ET.SubElement(bcs, 'bc', type='prescribed displacement', node_set=name)
        ET.SubElement(item, 'dof').text = axis
        ET.SubElement(item, 'value', lc=lc).text = value
        ET.SubElement(item, 'relative').text = '0'

    for axis in 'xyz':
        bc('bottom', axis, '1', '1')
    for node in mesh['boundaries']['top']['node_ids']:
        for axis in ('z',) if half else 'xyz':
            bc(f'top_node_{node}', axis, '2' if axis == 'z' else '1',
               '0.5' if half and axis == 'z' else '1')
    data = ET.SubElement(root, 'LoadData')
    for ident, multiplier in [('1', 0), ('2', 1)]:
        curve = ET.SubElement(data, 'load_controller', id=ident, type='loadcurve')
        values = ET.SubElement(curve, 'points')
        for step in range(61):
            ET.SubElement(values, 'pt').text = f'{step / 60},{multiplier * step / 60}'
    return mesh, root


def check(mesh, root, native_domain):
    return gate.validate_topology_bc(mesh, ET.tostring(root), native_domain=native_domain)


@pytest.mark.parametrize('domain', ['full_native', 'lower_half_reconstructed'])
def test_complete_synthetic_fixture_passes(domain):
    mesh, root = fixture(domain)
    result = check(mesh, root, domain)
    assert (result['node_count'], result['hex8_count'], result['top_node_count']) == (8, 1, 4)
    assert result['bc_count'] == (7 if domain == 'lower_half_reconstructed' else 15)
    with pytest.raises(ValueError, match='not a native execution release'):
        gate.require_execution_ready()


@pytest.mark.parametrize('mutation', ['missing_top_bc', 'extra_top_bc', 'side_bc', 'duplicate_bc',
                                      'singleton_swap', 'top_faces', 'node_position',
                                      'element_connectivity', 'missing_top_node_set'])
def test_synthetic_topology_and_fixture_defects_fail_closed(mutation):
    domain = 'lower_half_reconstructed'
    mesh, root = fixture(domain)
    bcs = root.find('Boundary')
    if mutation == 'missing_top_bc':
        bcs.remove(bcs[-1])
    elif mutation == 'extra_top_bc':
        ET.SubElement(bcs[-1], 'unused').text = 'ignored'  # BC payload itself must be exact.
        bcs[-1].find('dof').text = 'x'
    elif mutation == 'side_bc':
        bcs[-1].set('node_set', 'side')
    elif mutation == 'duplicate_bc':
        bcs.append(deepcopy(bcs[-1]))
    elif mutation == 'singleton_swap':
        root.find("Mesh/NodeSet[@name='top_node_5']").text = '6'
    elif mutation == 'top_faces':
        root.find("Mesh/Surface[@name='top']/quad4").text = '5,6,8,7'
    elif mutation == 'node_position':
        root.find('Mesh/Nodes/node').text = '0.001,0,0'
    elif mutation == 'element_connectivity':
        root.find('Mesh/Elements/elem').text = '1,2,3,4,5,6,8,7'
    else:
        root.find('Mesh').remove(root.find("Mesh/NodeSet[@name='top_node_5']"))
    with pytest.raises(ValueError):
        check(mesh, root, domain)


def test_full_plate_requires_every_tangential_bc_and_half_frees_them():
    mesh, root = fixture('full_native')
    top_x = next(row for row in root.findall('Boundary/bc')
                 if row.attrib['node_set'] == 'top_node_5' and row.findtext('dof') == 'x')
    root.find('Boundary').remove(top_x)
    with pytest.raises(ValueError, match='fixture BCs'):
        check(mesh, root, 'full_native')
    mesh, root = fixture('lower_half_reconstructed')
    top = next(row for row in root.findall('Boundary/bc') if row.attrib['node_set'] == 'top_node_5')
    top.find('dof').text = 'x'
    with pytest.raises(ValueError, match='fixture BCs'):
        check(mesh, root, 'lower_half_reconstructed')


def test_frozen_binding_inventory_and_no_execution_gate():
    assert gate.validate_binding_manifest(ROOT, inspect_sources=False) == {
        'status': 'non_executable_manifest_checked', 'source_decks': 10, 'meshes': 7, 'rows': 12}
    manifest = json.loads((ROOT / gate.BINDING_PATH).read_text())
    assert manifest['run_source_keys']['compression:N36:S120:reference'] == 'compression:N36'
    assert manifest['run_source_keys']['tension:N24:S120:reference'] == 'tension:N24'
    assert set(manifest['phase_gates'].values()) == {False}


def test_hash_and_path_binding_fail_closed(tmp_path):
    source = tmp_path / 'source.feb'
    source.write_bytes(b'old source')
    binding = {'path': 'source.feb', 'sha256': gate._sha(b'old source')}
    assert gate._read_bound(tmp_path, binding, maximum=64) == b'old source'
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='SHA256'):
        gate._read_bound(tmp_path, binding, maximum=64)
    with pytest.raises(ValueError, match='unsafe'):
        gate._read_bound(tmp_path, {'path': '../source.feb', 'sha256': binding['sha256']},
                         maximum=64)


def test_local_hash_bound_sources_when_present():
    # This inventory lives in ignored local outputs and is absent from clean CI.
    manifest = json.loads((ROOT / gate.BINDING_PATH).read_text())
    if not all((ROOT / row['path']).is_file() for row in manifest['source_decks'].values()):
        pytest.skip('Local ignored old-domain source decks unavailable')
    result = gate.validate_binding_manifest(ROOT)
    assert (result['source_decks'], result['meshes'], result['rows']) == (10, 7, 12)
    assert result['status'] == 'non_executable_source_topology_checked'
