"""Analytical arrays/XML only; no saved specimens, meshing or solver."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('halfheight_mesh_controls', ROOT / 'scripts/mechanics_hbe_halfheight_mesh.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


@pytest.fixture
def protocol():
    return M.BASE.read_protocol(ROOT / 'manifests/experiments/hbe-01-03-mechanics-poc-v1.json')


def box(protocol):
    """Eight analytical box cells, deliberately not a verified HBE specimen."""
    H, R = protocol['geometry']['height_m'], protocol['geometry']['radius_m']
    X = np.array([[x, y, z] for z in [0., H / 2, H] for y in [-R/2, 0., R/2] for x in [-R/2, 0., R/2]])
    corners = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]])
    cells = []
    for z, y, x in itertools.product(range(2), repeat=3):
        points = corners + [x, y, z]
        cells.append((points[:, 2]*9 + points[:, 1]*3 + points[:, 0] + 1).tolist())
    owners = {}
    for cell in np.array(cells):
        for face in cell[M.BASE.HEX_FACES]:
            owners.setdefault(tuple(sorted(face)), []).append(face.tolist())
    boundaries = {name: {'node_ids': [], 'faces_quad4': []} for name in ('bottom', 'top', 'side')}
    for faces in owners.values():
        if len(faces) != 1:
            continue
        face = faces[0]
        z = X[np.array(face) - 1, 2]
        name = 'bottom' if (z == 0).all() else 'top' if (z == H).all() else 'side'
        boundaries[name]['faces_quad4'].append(face)
    for row in boundaries.values():
        row['node_ids'] = sorted({i for face in row['faces_quad4'] for i in face})
    return {'schema_version': 'analytic_unit_box_only', 'indexing': 'one_based', 'mesh_N': 8,
            'geometry': copy.deepcopy(protocol['geometry']), 'rest_nodes_m': X.tolist(),
            'elements_hex8': cells, 'node_ids': list(range(1, 28)), 'element_ids': list(range(1, 9)),
            'gmsh_node_ids': list(range(101, 128)), 'gmsh_element_ids': list(range(201, 209)),
            'boundaries': boundaries}


def test_public_entry_refuses_analytical_box_as_a_verified_specimen(protocol):
    with pytest.raises(ValueError, match='counts differ'):
        M.extract_halfheight(box(protocol), protocol)


def test_half_extraction_preserves_coordinates_ids_and_both_reflection_incidence_maps(protocol):
    original = box(protocol)
    before = copy.deepcopy(original)
    result = M._extract_verified(original, protocol)
    assert original == before
    half, mapping, proof = result['mesh'], result['mapping'], result['verification']
    assert half['geometry']['height_m'] == original['geometry']['height_m']/2
    assert half['full_geometry'] == original['geometry']
    assert len(half['node_ids']) == 18 and len(half['element_ids']) == 4
    assert mapping['half_to_full_node_ids'] == list(range(1, 19))
    assert mapping['half_to_full_element_ids'] == [1, 2, 3, 4]
    assert half['rest_nodes_m'] == original['rest_nodes_m'][:18]
    assert proof['coordinates_unchanged'] and proof['straddling_cells'] == 0
    assert proof['half_volume_m3'] == pytest.approx(proof['full_volume_m3']/2)
    assert proof['midplane_nodes'] == 9
    for reflection in mapping['reflections']:
        assert len(set(reflection['node_full_ids'])) == 18
        assert len(set(reflection['element_full_ids'])) == 4
        for local_half, full_id, permutation in zip(half['elements_hex8'], reflection['element_full_ids'], reflection['element_half_to_full_local']):
            reflected_ids = np.array(reflection['node_full_ids'])[np.array(local_half)-1]
            full_ids = np.array(original['elements_hex8'][full_id-1])[permutation]
            assert np.array_equal(reflected_ids, full_ids)
    assert mapping['reflections'][1]['signs'] == [1,1,-1]
    assert mapping['reflections'][1]['rest_offset_m'] == [0.,0.,protocol['geometry']['height_m']]
    assert set(mapping['reflections'][0]['element_full_ids'] + mapping['reflections'][1]['element_full_ids']) == set(range(1,9))
    assert M._extract_verified(original, protocol) == result


def test_boundary_tags_preserve_physical_plate_and_free_outer_faces(protocol):
    result = M._extract_verified(box(protocol), protocol)
    mesh = result['mesh']; X = np.asarray(mesh['rest_nodes_m'])
    for name, z in [('bottom', 0.), ('top', protocol['geometry']['height_m']/2)]:
        assert (X[np.asarray(mesh['boundaries'][name]['node_ids'])-1,2] == z).all()
    assert mesh['boundary_roles'] == {'bottom':'physical_bonded_plate', 'top':'artificial_midplane', 'side':'traction_free_outer'}
    assert M.BASE.boundary_topology(mesh['elements_hex8'], mesh['boundaries'])['connected_cell_count'] == 4


def test_sorted_json_round_trip_preserves_mapping_and_deck_bytes(protocol):
    original = box(protocol)
    extracted = M._extract_verified(original, protocol)
    reloaded_source = json.loads(json.dumps(original, sort_keys=True, allow_nan=False))
    reloaded_extraction = json.loads(json.dumps(extracted, sort_keys=True, allow_nan=False))
    assert M._extract_verified(reloaded_source, protocol) == extracted
    for branch in ('compression', 'tension'):
        assert M.halfheight_deck(reloaded_extraction, branch, protocol) == M.halfheight_deck(extracted, branch, protocol)


@pytest.mark.parametrize('fault', ['straddler', 'reflection', 'inverted', 'boundary'])
def test_ineligible_saved_topology_is_rejected_without_repair(protocol, fault):
    original = box(protocol)
    if fault == 'straddler':
        for point in original['rest_nodes_m'][9:18]:
            point[2] *= .8
    elif fault == 'reflection':
        original['rest_nodes_m'][-1][0] += 1e-5
    elif fault == 'inverted':
        original['elements_hex8'][0] = [original['elements_hex8'][0][i] for i in [1,0,3,2,5,4,7,6]]
    else:
        original['boundaries']['side']['faces_quad4'].pop()
    before = copy.deepcopy(original)
    with pytest.raises(ValueError):
        M._extract_verified(original, protocol)
    assert original == before


@pytest.mark.parametrize('branch,sign', [('compression', -1), ('tension', 1)])
def test_deck_changes_only_midplane_constraints_and_preserves_full_load_curve(protocol, branch, sign):
    result = M._extract_verified(box(protocol), protocol)
    reference, _ = M.BASE.specimen_deck(result['mesh'], branch, 60, 1000., protocol)
    xml, loading = M.halfheight_deck(result, branch, protocol)
    root, old = ET.fromstring(xml), ET.fromstring(reference)
    for section in ('Control', 'Material', 'Mesh', 'MeshDomains', 'LoadData', 'Output'):
        assert ET.tostring(root.find(section)) == ET.tostring(old.find(section))
    physical = [bc for bc in root.find('Boundary') if bc.attrib['node_set'] == 'bottom']
    assert [bc.findtext('dof') for bc in physical] == ['x','y','z']
    old_physical = [bc for bc in old.find('Boundary') if bc.attrib['node_set'] == 'bottom']
    assert [ET.tostring(bc) for bc in physical] == [ET.tostring(bc) for bc in old_physical]
    top = [bc for bc in root.find('Boundary') if bc.attrib['node_set'].startswith('top_node_')]
    assert len(top) == len(result['mesh']['boundaries']['top']['node_ids'])
    for bc in top:
        assert bc.findtext('dof') == 'z' and float(bc.findtext('value')) == .5
        curve = root.find("LoadData/load_controller[@id='"+bc.find('value').attrib['lc']+"']")
        points = [tuple(map(float, pt.text.split(','))) for pt in curve.find('points')]
        assert len(points) == 61 and points[0] == (0.,0.)
        assert points[-1] == pytest.approx((1., sign*.15*protocol['geometry']['height_m']))
        assert float(bc.findtext('value'))*points[-1][1] == pytest.approx(sign*.15*result['mesh']['geometry']['height_m'])
    assert loading['prescribed_dofs'] == {'bottom':'xyz','midplane':'z'}
    assert loading['artificial_midplane_tangential_dofs'] == 'free'
    assert loading['full_reaction_scale'] == 1 and loading['full_energy_scale'] == 2
    assert loading['load_coordinate'][-1] == pytest.approx(sign*.15*protocol['geometry']['height_m'])
    assert loading['halfheight_equivalence_declaration_sha256'] == M.DECLARATION_SHA256
    assert M.halfheight_deck(result, branch, protocol) == (xml,loading)


@pytest.mark.parametrize('fault', ['torsion', 'changed_coordinates', 'wrong_height'])
def test_deck_refuses_torsion_or_changed_extraction(protocol, fault):
    result = M._extract_verified(box(protocol), protocol)
    branch = 'compression'
    if fault == 'torsion':
        branch = 'torsion_neg'
    elif fault == 'changed_coordinates':
        result['mesh']['rest_nodes_m'][0][0] += 1e-6
    else:
        result['mesh']['full_height_m'] *= 2
    with pytest.raises(ValueError):
        M.halfheight_deck(result, branch, protocol)
