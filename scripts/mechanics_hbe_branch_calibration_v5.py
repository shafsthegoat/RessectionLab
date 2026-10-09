"""Pure same-endpoint FEBio deck adapter; no native execution or measured reads.

The supplied source is an already authenticated old-domain Skyline axial deck.
This module changes only its time controls and axial load points, then applies
the established Accelerate solver transformation. A future release must bind
actual source bytes and independently validate all new-domain primitives.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from scripts import mechanics_hbe_backend as backend
from scripts import mechanics_hbe_branch_calibration_v4 as v4

DECLARATION_PATH = 'manifests/experiments/hbe-01-03-branch-calibration-v5.json'
DECLARATION_SHA256 = '50c5dbc8c45279245a8dcd9b92d16e9499bad6bc61c6dfbd5b5e307af49a71ce'
RUNS = (
    ('compression', 8, 60, 'full_native'),
    ('compression', 12, 60, 'full_native'),
    ('compression', 16, 60, 'full_native'),
    ('compression', 24, 60, 'lower_half_reconstructed'),
    ('compression', 32, 60, 'lower_half_reconstructed'),
    ('compression', 36, 60, 'lower_half_reconstructed'),
    ('compression', 36, 120, 'lower_half_reconstructed'),
    ('tension', 8, 60, 'full_native'),
    ('tension', 12, 60, 'full_native'),
    ('tension', 16, 60, 'lower_half_reconstructed'),
    ('tension', 24, 60, 'lower_half_reconstructed'),
    ('tension', 24, 120, 'lower_half_reconstructed'),
)
CONTROLS = ('Control/time_steps', 'Control/step_size',
            'Control/time_stepper/dtmin', 'Control/time_stepper/dtmax')
PERMITTED_DECK_CHANGES = [*CONTROLS, 'LoadData/zero_load_controller/points',
                          'LoadData/axial_load_controller/points']


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _bound_bytes(root, binding, maximum=1024 * 1024):
    path = (Path(root) / binding['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError('Portable evidence path missing or outside repository')
    if path.stat().st_size > maximum:
        raise ValueError('Portable evidence exceeds bounded reader')
    data = path.read_bytes()
    if _sha(data) != binding['sha256']:
        raise ValueError('Portable evidence hash differs')
    return data


def validate_preparation(root):
    """Bind committed metadata only; do not open meshes, curves or native logs."""
    study = json.loads(_bound_bytes(root, {'path': DECLARATION_PATH,
                                          'sha256': DECLARATION_SHA256}))
    if study.get('previous_v4') != {'path': v4.DECLARATION_PATH,
                                   'sha256': v4.DECLARATION_SHA256}:
        raise ValueError('v4 ancestry binding differs')
    prior = v4.validate_preparation(root)
    if (study.get('schema') != 'hbe-branch-calibration-v5-deck-preparation'
            or study.get('status') != 'non_executable_pure_adapter_preparation'
            or study.get('specimen') != 'HBE_01_03'
            or study.get('phase_gates') != dict.fromkeys(
                ('native_execution', 'fit', 'held_out_torque_access', 'mesher',
                 'automatic_reference_release'), False)
            or len(study.get('ordered_reference_runs', [])) != 12):
        raise ValueError('Non-executable 12-run preparation contract differs')
    contract = study.get('deck_adapter_contract', {})
    if (contract.get('only_changes') != PERMITTED_DECK_CHANGES
            or contract.get('only_changes_scope') !=
            'old_Skyline_to_new_Skyline_before_existing_Accelerate_backend_transform'):
        raise ValueError('Declared deck changes must include both regenerated controllers')
    for branch in v4.AXIAL:
        v4.frozen_coordinates(prior, branch)
        item, old = study['coordinates'][branch], prior['coordinates'][branch]
        if item != {'endpoint_decimal_literal_m': old['endpoint_decimal_literal_m'],
                    'endpoint_float64_hex': old['endpoint_float64_hex'],
                    'coordinate_sha256': old['coordinate_sha256'],
                    'row_count': 30, 'member_sha256': old['member']['sha256']}:
            raise ValueError('Retrospectively exposed coordinate identity differs')
    for n in (8, 12):
        item = study['mesh_identities'][str(n)]
        receipt = json.loads(_bound_bytes(root, item['origin_receipt']))
        if (receipt.get('mesh_N') != n or receipt.get('status') != 'prepared_not_solved'
                or receipt.get('mesh_sha256') != item['full_mesh_sha256']
                or receipt.get('quality', {}).get('passed') is not True
                or receipt['quality'].get('node_count') != item['full_nodes']
                or receipt['quality'].get('hex8_cell_count') != item['full_elements']
                or item['native_representation'] != 'full_native'):
            raise ValueError('N8/N12 full-native mesh origin receipt differs')
    for n in (16, 24, 32, 36):
        item, old = study['mesh_identities'][str(n)], prior['meshes'][str(n)]
        for key, expected in (('full_mesh_sha256', old['full_mesh']['sha256']),
                              ('half_mesh_sha256', old['half_mesh']['sha256']),
                              ('reconstruction_sha256', old['reconstruction']['sha256']),
                              ('half_nodes', old['half_nodes']),
                              ('half_elements', old['half_elements'])):
            if item.get(key) != expected:
                raise ValueError('Inherited mesh identity differs')
    if [(r['branch'], r['N'], r['steps'], r['native_domain'])
            for r in study['ordered_reference_runs']] != list(RUNS):
        raise ValueError('Exact 12-run branch/mesh/schedule order required')
    for row in study['ordered_reference_runs']:
        run_spec(study, prior, row['run_id'])
    return study, prior


def run_spec(study, prior, run_id):
    matches = [row for row in study['ordered_reference_runs'] if row['run_id'] == run_id]
    if len(matches) != 1:
        raise ValueError('Unique declared reference run required')
    row = matches[0]
    if (row['branch'], row['N'], row['steps'], row['native_domain']) not in RUNS:
        raise ValueError('Undeclared mesh representation or schedule')
    b, n, s = row['branch'], row['N'], row['steps']
    v4.frozen_coordinates(prior, b)
    expected = prior['coordinates'][b]['endpoint_decimal_literal_m']
    if (run_id != f'{b}:N{n}:S{s}:reference' or type(n) is not int or type(s) is not int
            or row['frame_count'] != s + 1 or row['reference_mu_Pa'] != 1000.0
            or row['endpoint_decimal_literal_m'] != expected
            or row['endpoint_float64_hex'] != float(expected).hex()
            or row['start_from_rest'] is not True
            or row['source_deck_sha256'] is not None
            or row['source_deck_binding_status'] != 'requires_separate_preexecution_release'):
        raise ValueError('Run source, coordinate or fail-closed state differs')
    return deepcopy(row)


def schedule(study, prior, run_id):
    row = run_spec(study, prior, run_id)
    endpoint = float(Decimal(row['endpoint_decimal_literal_m']))
    s = row['steps']
    times = [i / s for i in range(s + 1)]
    full = [endpoint * t for t in times]
    full[-1] = endpoint
    factor = 0.5 if row['native_domain'] == 'lower_half_reconstructed' else 1.0
    return {'times': times, 'full_coordinates_m': full,
            'native_boundary_coordinates_m': [factor * value for value in full],
            'native_boundary_factor': factor, 'endpoint_decimal_literal_m': row['endpoint_decimal_literal_m'],
            'endpoint_float64_hex': endpoint.hex()}


def _xml(value):
    data = value.encode() if isinstance(value, str) else value
    if not isinstance(data, bytes) or len(data) > 32 * 1024**2 or b'<!DOCTYPE' in data or b'<!ENTITY' in data:
        raise ValueError('Bounded plain FEBio XML required')
    return ET.fromstring(data)


def _unique(root, path):
    nodes = root.findall(path)
    if len(nodes) != 1:
        raise ValueError('Exact FEBio control subtree required: ' + path)
    return nodes[0]


def _curves(root, *, steps, endpoint, factor, legacy_height=None):
    controllers = root.findall('LoadData/load_controller')
    if len(controllers) != 2 or len(root.findall('.//load_controller')) != 2:
        raise ValueError('Exactly zero and axial load controllers required')
    found = {}
    for controller in controllers:
        if controller.attrib.get('type') != 'loadcurve' or controller.findtext('interpolate') != 'LINEAR':
            raise ValueError('Unchanged linear load controllers required')
        points = _unique(controller, 'points')
        if len(points) != steps + 1 or any(pt.tag != 'pt' or not pt.text for pt in points):
            raise ValueError('Complete point grid required')
        parsed = []
        for pt in points:
            fields = pt.text.split(',')
            if len(fields) != 2:
                raise ValueError('Two finite controller values required')
            pair = tuple(float(x) for x in fields)
            if not all(math.isfinite(x) for x in pair):
                raise ValueError('Finite controller values required')
            parsed.append(pair)
        expected_time = ([float(x) for x in np.linspace(0., 1., steps + 1)]
                         if legacy_height is not None else [i / steps for i in range(steps + 1)])
        if [p[0] for p in parsed] != expected_time:
            raise ValueError('Exact equal pseudotime grid required')
        values = [p[1] for p in parsed]
        kind = 'zero' if all(value == 0 for value in values) else 'axial'
        if kind in found:
            raise ValueError('Duplicate zero or axial controller')
        if kind == 'axial':
            # Historical specimen_deck multiplied numpy pseudotimes by strain
            # and then height. Reassociating those floats changes some ULPs.
            expected = ([float(x) for x in np.linspace(0., 1., steps + 1)
                         * math.copysign(.15, endpoint) * legacy_height]
                        if legacy_height is not None else [endpoint * t for t in expected_time])
            expected[-1] = endpoint
            if values != expected:
                raise ValueError('Axial controller endpoint, sign or grid differs')
        found[kind] = (controller, points)
    if set(found) != {'zero', 'axial'}:
        raise ValueError('Both zero and axial controllers required')
    zero_id, axial_id = (found[k][0].attrib.get('id') for k in ('zero', 'axial'))
    if not zero_id or not axial_id or zero_id == axial_id:
        raise ValueError('Distinct load-controller IDs required')
    bcs = root.findall('Boundary/bc')
    if not bcs:
        raise ValueError('Complete bottom/top boundary conditions required')
    bottom, top = set(), set()
    for bc in bcs:
        if bc.attrib.get('type') != 'prescribed displacement':
            raise ValueError('Unexpected boundary condition type')
        dof = bc.findtext('dof')
        value = _unique(bc, 'value')
        node_set = bc.attrib.get('node_set', '')
        if bc.findtext('relative') != '0' or dof not in ('x', 'y', 'z'):
            raise ValueError('Unexpected relative or boundary DOF')
        if node_set == 'bottom':
            if value.attrib.get('lc') != zero_id or value.text != '1':
                raise ValueError('Bottom boundary factor or controller differs')
            bottom.add(dof)
        elif node_set.startswith('top_node_'):
            if factor == 0.5 and dof != 'z':
                raise ValueError('Half model must leave midplane tangential DOFs free')
            if value.attrib.get('lc') != (axial_id if dof == 'z' else zero_id):
                raise ValueError('Top controller assignment differs')
            if float(value.text) != (factor if dof == 'z' else 1.0):
                raise ValueError('Full/half boundary load factor differs')
            top.add((node_set, dof))
        else:
            raise ValueError('Undeclared boundary node set')
    if bottom != {'x', 'y', 'z'} or not top or len(bottom) + len(top) != len(bcs):
        raise ValueError('Complete unique bottom/top boundary conditions required')
    top_nodes = {node for node, _ in top}
    if top != {(node, dof) for node in top_nodes for dof in (('z',) if factor == 0.5 else ('x', 'y', 'z'))}:
        raise ValueError('Incomplete top-node DOF pattern')
    return found


def _canonical(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
            tuple(_canonical(child) for child in node))


def verify_deck_delta(source, changed, old_endpoint, new_endpoint, new_steps, factor, legacy_height):
    """Independently compare every XML element outside the permitted five changes."""
    old, new = _xml(source), _xml(changed)
    for path in CONTROLS:
        a, b = _unique(old, path), _unique(new, path)
        old_value = '60' if path.endswith('time_steps') else format(1 / 60, '.17g')
        new_value = str(new_steps) if path.endswith('time_steps') else format(1 / new_steps, '.17g')
        if a.text != old_value or b.text != new_value:
            raise ValueError('Unexpected time-control value')
        b.text = a.text
    old_curves = _curves(old, steps=60, endpoint=old_endpoint, factor=factor,
                         legacy_height=legacy_height)
    new_curves = _curves(new, steps=new_steps, endpoint=new_endpoint, factor=factor)
    for key in ('zero', 'axial'):
        a_controller, a_points = old_curves[key]
        b_controller, b_points = new_curves[key]
        if a_controller.attrib != b_controller.attrib:
            raise ValueError('Load-controller identity changed')
        b_controller.remove(b_points)
        b_controller.append(deepcopy(a_points))
    if _canonical(old) != _canonical(new):
        raise ValueError('FEBio physics, geometry, material, solver or output changed')


def adapt_deck(study, prior, run_id, source_xml):
    """Transform supplied old-domain Skyline XML and return an unsigned receipt."""
    row = run_spec(study, prior, run_id)
    plan = schedule(study, prior, run_id)
    raw = source_xml.encode() if isinstance(source_xml, str) else source_xml
    if not isinstance(raw, bytes):
        raise ValueError('Source FEBio bytes required')
    old_endpoint = (-1 if row['branch'] == 'compression' else 1) * .15 * prior['scientific_invariants']['geometry']['height_m']
    tree = _xml(raw)
    for path in CONTROLS:
        node = _unique(tree, path)
        expected = '60' if path.endswith('time_steps') else format(1 / 60, '.17g')
        if node.text != expected:
            raise ValueError('Exact original S60 controls required')
    height = prior['scientific_invariants']['geometry']['height_m']
    _curves(tree, steps=60, endpoint=old_endpoint, factor=plan['native_boundary_factor'],
            legacy_height=height)
    # The existing backend accepts only the one canonical Skyline solver subtree.
    backend.transform_deck(raw)
    target = deepcopy(tree)
    for path in CONTROLS:
        node = _unique(target, path)
        node.text = str(row['steps']) if path.endswith('time_steps') else format(1 / row['steps'], '.17g')
    target_curves = _curves(target, steps=60, endpoint=old_endpoint,
                            factor=plan['native_boundary_factor'], legacy_height=height)
    for kind in ('zero', 'axial'):
        points = target_curves[kind][1]
        for child in list(points):
            points.remove(child)
        values = ([0.0] * (row['steps'] + 1) if kind == 'zero'
                  else plan['full_coordinates_m'])
        for t, value in zip(plan['times'], values):
            ET.SubElement(points, 'pt').text = f'{t:.17g},{value:.17g}'
    ET.indent(target, space='  ')
    skyline = ET.tostring(target, encoding='unicode', xml_declaration=True) + '\n'
    verify_deck_delta(raw, skyline, old_endpoint, float(row['endpoint_decimal_literal_m']),
                      row['steps'], plan['native_boundary_factor'], height)
    adapted = backend.transform_deck(skyline)
    backend.verify_deck(skyline, adapted)
    receipt = {'run_id': run_id, 'status': 'pure_deck_preparation_no_native_execution',
               'source_deck_sha256': _sha(raw), 'skyline_sha256': _sha(skyline.encode()),
               'adapted_deck_sha256': _sha(adapted.encode()),
               'reference_mu_Pa': 1000.0, 'branch': row['branch'], 'N': row['N'],
               'steps': row['steps'], 'frame_count': row['frame_count'],
               'native_domain': row['native_domain'], 'times': plan['times'],
               'full_load_coordinates_m': plan['full_coordinates_m'],
               'native_boundary_factor': plan['native_boundary_factor'],
               'native_boundary_coordinates_m': plan['native_boundary_coordinates_m'],
               'endpoint_decimal_literal_m': row['endpoint_decimal_literal_m'],
               'endpoint_float64_hex': row['endpoint_float64_hex'],
               'source_deck_identity_approved_for_execution': False}
    return skyline, adapted, receipt


def require_execution_ready(*args, **kwargs):
    raise ValueError('v5 pure deck preparation has no native primitive checker or execution release')


def require_fit_eligible(*args, **kwargs):
    raise ValueError('v5 requires independently accepted twelve-reference qualification before fit')


def require_torque_access(*args, **kwargs):
    raise ValueError('v5 has no calibrated paired solves or durable held-out prediction freeze')
