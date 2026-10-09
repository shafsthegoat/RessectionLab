"""Generated XML fixtures exercise controls only; no mechanics are observed."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

import pytest

from scripts import mechanics_hbe_branch_calibration_v5 as core

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / core.DECLARATION_PATH).read_text())
PRIOR = json.loads((ROOT / STUDY['previous_v4']['path']).read_text())


@pytest.fixture(autouse=True)
def no_measured_or_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Deck preparation cannot open measured/native data or execute solvers')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    original = io.open

    def restricted(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if resolved.is_relative_to(ROOT / 'data/mechanics') or resolved.is_relative_to(ROOT / 'outputs/mechanics'):
                forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(io, 'open', restricted)


def _add(parent, tag, text=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = str(text)
    return child


def fixture_source(branch, native_domain):
    """Tiny old-domain deck with the exact relevant FEBio structure."""
    factor = 0.5 if native_domain == 'lower_half_reconstructed' else 1.0
    endpoint = (-1 if branch == 'compression' else 1) * .15 * PRIOR['scientific_invariants']['geometry']['height_m']
    root = ET.Element('febio_spec', version='4.0')
    control = _add(root, 'Control')
    for path, value in [('time_steps', '60'), ('step_size', format(1/60, '.17g'))]:
        _add(control, path, value)
    stepper = _add(control, 'time_stepper', type='default')
    for key in ('dtmin', 'dtmax'):
        _add(stepper, key, format(1/60, '.17g'))
    solver = _add(control, 'solver', type='solid')
    _add(solver, 'linear_solver', type='skyline')
    material = _add(_add(root, 'Material'), 'material', type='Ogden')
    _add(material, 'c1', '2000')
    boundaries = _add(root, 'Boundary')
    def bc(node, dof, lc, value):
        row = _add(boundaries, 'bc', type='prescribed displacement', node_set=node)
        _add(row, 'dof', dof)
        _add(row, 'value', value, lc=lc)
        _add(row, 'relative', '0')
    for dof in 'xyz':
        bc('bottom', dof, '1', '1')
    for node in ('top_node_7', 'top_node_8'):
        for dof in ('z',) if factor == .5 else 'xyz':
            bc(node, dof, '2' if dof == 'z' else '1', '0.5' if dof == 'z' and factor == .5 else '1')
    data = _add(root, 'LoadData')
    times = np.linspace(0., 1., 61)
    for number, values in [('1', [0.0] * 61),
                           ('2', times * (-.15 if branch == 'compression' else .15)
                            * PRIOR['scientific_invariants']['geometry']['height_m'])]:
        curve = _add(data, 'load_controller', id=number, type='loadcurve')
        _add(curve, 'interpolate', 'LINEAR')
        points = _add(curve, 'points')
        for time, value in zip(times, values):
            _add(points, 'pt', f'{time:.17g},{value:.17g}')
    output = _add(root, 'Output')
    _add(output, 'node_data', 'x;y;z', file='nodes.log')
    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n'


def test_portable_metadata_and_no_execution_release():
    study, prior = core.validate_preparation(ROOT)
    assert study == STUDY and prior == PRIOR
    assert len(study['ordered_reference_runs']) == 12
    assert study['deck_adapter_contract']['only_changes'] == core.PERMITTED_DECK_CHANGES
    assert study['phase_gates'] == dict.fromkeys(study['phase_gates'], False)
    for gate in (core.require_execution_ready, core.require_fit_eligible, core.require_torque_access):
        with pytest.raises(ValueError):
            gate()


@pytest.mark.parametrize('run_id', [row['run_id'] for row in STUDY['ordered_reference_runs']])
def test_all_twelve_schedules_and_pure_febio_adapters(run_id):
    row = core.run_spec(STUDY, PRIOR, run_id)
    source = fixture_source(row['branch'], row['native_domain'])
    skyline, adapted, receipt = core.adapt_deck(STUDY, PRIOR, run_id, source)
    plan = core.schedule(STUDY, PRIOR, run_id)
    assert receipt['status'] == 'pure_deck_preparation_no_native_execution'
    assert receipt['source_deck_identity_approved_for_execution'] is False
    assert receipt['adapted_deck_sha256'] == core._sha(adapted.encode())
    assert receipt['skyline_sha256'] == core._sha(skyline.encode())
    assert receipt['full_load_coordinates_m'] == plan['full_coordinates_m']
    assert receipt['native_boundary_coordinates_m'] == plan['native_boundary_coordinates_m']
    assert receipt['full_load_coordinates_m'][-1] == float(row['endpoint_decimal_literal_m'])
    assert receipt['endpoint_float64_hex'] == row['endpoint_float64_hex']
    assert receipt['native_boundary_coordinates_m'][-1] == (
        .5 if row['native_domain'] == 'lower_half_reconstructed' else 1.0
    ) * float(row['endpoint_decimal_literal_m'])
    assert len(receipt['times']) == row['frame_count']
    assert core._xml(adapted).find('Control/solver/linear_solver').attrib['type'] == 'accelerate'
    core.verify_deck_delta(source, skyline,
        (-1 if row['branch'] == 'compression' else 1) * .15 * PRIOR['scientific_invariants']['geometry']['height_m'],
        float(row['endpoint_decimal_literal_m']), row['steps'], receipt['native_boundary_factor'],
        PRIOR['scientific_invariants']['geometry']['height_m'])


@pytest.mark.parametrize('branch', ('compression', 'tension'))
def test_s120_even_states_are_exactly_s60_for_same_branch(branch):
    n = 36 if branch == 'compression' else 24
    coarse = core.schedule(STUDY, PRIOR, f'{branch}:N{n}:S60:reference')
    fine = core.schedule(STUDY, PRIOR, f'{branch}:N{n}:S120:reference')
    assert fine['times'][::2] == coarse['times']
    assert fine['full_coordinates_m'][::2] == coarse['full_coordinates_m']
    assert fine['native_boundary_coordinates_m'][::2] == coarse['native_boundary_coordinates_m']


@pytest.mark.parametrize('mutation', ('endpoint_inward', 'old_endpoint_wrong', 'legacy_float_order',
                                      'time_grid', 'time_control',
                                      'half_factor', 'top_dof', 'extra_controller', 'solver',
                                      'material', 'output', 'duplicate_top'))
def test_adversarial_old_source_or_new_delta_refused(mutation):
    run_id = 'compression:N24:S60:reference'
    source = fixture_source('compression', 'lower_half_reconstructed')
    tree = core._xml(source)
    if mutation == 'old_endpoint_wrong':
        points = tree.findall('LoadData/load_controller')[1].find('points')
        old = float(points[-1].text.split(',')[1])
        points[-1].text = f'1,{math.nextafter(old, 0.0):.17g}'
    elif mutation == 'legacy_float_order':
        points = tree.findall('LoadData/load_controller')[1].find('points')
        old = -.15 * PRIOR['scientific_invariants']['geometry']['height_m']
        points[1].text = f'{1 / 60:.17g},{old * (1 / 60):.17g}'
    elif mutation == 'time_grid':
        points = tree.findall('LoadData/load_controller')[1].find('points')
        points[30].text = '0.5000001,' + points[30].text.split(',')[1]
    elif mutation == 'time_control':
        tree.find('Control/time_stepper/dtmax').text = '0.02'
    elif mutation == 'half_factor':
        tree.findall('Boundary/bc')[-1].find('value').text = '1'
    elif mutation == 'top_dof':
        tree.findall('Boundary/bc')[-1].find('dof').text = 'x'
    elif mutation == 'extra_controller':
        tree.find('LoadData').append(deepcopy(tree.findall('LoadData/load_controller')[1]))
    elif mutation == 'solver':
        tree.find('Control/solver/linear_solver').set('type', 'accelerate')
    elif mutation == 'duplicate_top':
        tree.find('Boundary').append(deepcopy(tree.findall('Boundary/bc')[-1]))
    if mutation not in ('endpoint_inward', 'material', 'output'):
        with pytest.raises(ValueError):
            core.adapt_deck(STUDY, PRIOR, run_id, ET.tostring(tree, encoding='unicode'))
        return
    skyline, _, receipt = core.adapt_deck(STUDY, PRIOR, run_id, source)
    changed = core._xml(skyline)
    if mutation == 'endpoint_inward':
        points = changed.findall('LoadData/load_controller')[1].find('points')
        endpoint = float(STUDY['coordinates']['compression']['endpoint_decimal_literal_m'])
        points[-1].text = f'1,{math.nextafter(endpoint, 0.0):.17g}'
    elif mutation == 'material':
        changed.find('Material/material/c1').text = '3000'
    else:
        changed.find('Output/node_data').set('file', 'different.log')
    old_endpoint = -.15 * PRIOR['scientific_invariants']['geometry']['height_m']
    with pytest.raises(ValueError):
        core.verify_deck_delta(source, ET.tostring(changed, encoding='unicode'), old_endpoint,
                               float(STUDY['coordinates']['compression']['endpoint_decimal_literal_m']),
                               60, receipt['native_boundary_factor'],
                               PRIOR['scientific_invariants']['geometry']['height_m'])


def test_mutated_manifest_and_mesh_representation_fail_closed(tmp_path):
    changed = deepcopy(STUDY)
    changed['ordered_reference_runs'][0]['native_domain'] = 'lower_half_reconstructed'
    target = tmp_path / core.DECLARATION_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='hash differs'):
        core.validate_preparation(tmp_path)
    changed = deepcopy(STUDY)
    changed['ordered_reference_runs'][0]['native_domain'] = 'lower_half_reconstructed'
    with pytest.raises(ValueError, match='mesh representation'):
        core.run_spec(changed, PRIOR, 'compression:N8:S60:reference')
