"""Generated analytical frames only; no HBE response, solver or patient reads."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_frame as frame
from scripts.mechanics_hbe_outputs import _iter_data_records
from test_mechanics_hbe_branch_calibration_v5 import fixture_source
from test_mechanics_hbe_halfheight_readout import geometry

ROOT = Path(__file__).resolve().parents[1]
STUDY_BYTES = (ROOT / v5.DECLARATION_PATH).read_bytes()
STUDY = json.loads(STUDY_BYTES)
PRIOR_BYTES = (ROOT / STUDY['previous_v4']['path']).read_bytes()


@pytest.fixture(autouse=True)
def no_measured_native_or_patient_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Generated frame evaluation cannot open observed or native data')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    original = io.open

    def restricted(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if any(resolved.is_relative_to(ROOT / section) for section in
                   ('data/mechanics', 'outputs/mechanics', 'data/patients')):
                forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(io, 'open', restricted)


def contract(branch='tension', domain='full_native'):
    N = 8 if domain == 'full_native' else 24
    run_id = f'{branch}:N{N}:S60:reference'
    source = fixture_source(branch, domain)
    _, adapted, receipt = v5.adapt_deck(STUDY, json.loads(PRIOR_BYTES), run_id, source)
    return frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES, run_id, source,
                                   adapted, receipt), source, adapted, receipt


def generated(branch='tension', domain='full_native', step=60):
    spec = contract(branch, domain)[0]
    full, half, mapping = geometry()
    mesh = full if domain == 'full_native' else half
    X = np.asarray(mesh['rest_nodes_m'], dtype=float)
    d = spec['full_coordinates_m'][step]
    current = X.copy()
    current[:, 2] += X[:, 2] * d / full['geometry']['height_m']
    reaction = np.zeros_like(current)
    force = math.copysign(.002, d) if d else 0.
    bottom, top = mesh['boundaries']['bottom']['node_ids'], mesh['boundaries']['top']['node_ids']
    reaction[np.asarray(bottom)-1, 2] = force / len(bottom)
    reaction[np.asarray(top)-1, 2] = -force / len(top)
    values = np.column_stack((current, current-X, reaction))
    stretch = 1 + d / full['geometry']['height_m']
    logged = np.tile([10., 11., 12., 13., 14., 15., stretch, 16.],
                     (len(mesh['element_ids']), 1))
    time = spec['times'][step]
    nodes = {'step': step, 'time': float(format(time, '.9g')),
             'declared_time': time, 'name': 'mechanics_nodes_si', 'values': values}
    elements = {'step': step, 'time': float(format(time, '.9g')),
                'declared_time': time, 'name': 'mechanics_elements_si', 'values': logged}
    reconstruction = (full, mapping) if domain != 'full_native' else None
    return spec, mesh, nodes, elements, reconstruction


@pytest.mark.parametrize('branch', ('tension', 'compression'))
@pytest.mark.parametrize('domain', ('full_native', 'lower_half_reconstructed'))
@pytest.mark.parametrize('step', (0, 30, 60))
def test_extended_endpoint_generated_frame(branch, domain, step):
    args = generated(branch, domain, step)
    result = frame.evaluate_generated_frame(*args[:4], reconstruction=args[4])
    assert result['fixture_passed']
    assert result['frame'] == step
    assert result['input_coordinate_full_m'] == args[0]['full_coordinates_m'][step]
    assert result['minimum_sampled_J'] > 0
    assert result['minimum_logged_J'] > 0
    assert len(result['probe_displacements_m']) == 75
    assert np.asarray(result['probe_displacements_m']).shape == (75, 3)
    assert result['applied_force_N'] == pytest.approx(0 if step == 0 else math.copysign(.002, result['input_coordinate_full_m']))
    assert result['provenance']['physical_validation_pass'] is None
    assert not result['provenance']['native_output_observed']
    assert not result['logged_sed_used_for_energy_gate']
    if domain == 'lower_half_reconstructed':
        assert result['representation'] == 'reconstructed_full'
        assert result['energy_J'] == pytest.approx(2 * result['native_energy_J'], rel=1e-12)
        assert result['reflected_stress_component_min_Pa'][4:] == [-14., -15.]


def test_old_domain_overlap_agrees_with_pinned_readout_on_affine_fixture():
    spec, mesh_data, nodes, elements, _ = generated('tension', 'full_native', 30)
    new = frame.evaluate_generated_frame(spec, mesh_data, nodes, elements)
    from scripts.mechanics_hbe_physics import HexMesh
    old_d = .15 * mesh_data['geometry']['height_m']
    old_fraction = spec['full_coordinates_m'][30] / old_d
    old = HexMesh.from_manifest(mesh_data).read_frame(nodes['values'][:, :3], nodes['values'][:, 6:9],
                                                       mu_Pa=1000., branch='tension', fraction=old_fraction)
    assert old['input_coordinate'] == pytest.approx(new['input_coordinate_full_m'])
    assert old['applied_force_N'] == new['applied_force_N']
    assert old['energy_J'] == new['energy_J']
    assert old['checks']['prescribed_motion']


@pytest.mark.parametrize('mutation', ('wrong_half_factor', 'missing_top_bc', 'force_sign',
                                      'nonpositive_deformed_J', 'nonpositive_logged_J',
                                      'nonfinite_stress', 'time', 'step', 'primitive_motion'))
def test_adversarial_frame_controls(mutation):
    spec, mesh, nodes, elements, reconstruction = generated('tension', 'lower_half_reconstructed')
    spec, mesh, nodes, elements = deepcopy((spec, mesh, nodes, elements))
    if mutation == 'wrong_half_factor':
        spec['native_coordinates_m'] = tuple(spec['full_coordinates_m'])
    elif mutation == 'missing_top_bc':
        nodes['values'][4, 2] -= 1e-4
        nodes['values'][4, 5] -= 1e-4
    elif mutation == 'force_sign':
        nodes['values'][:, 6:9] *= -1
    elif mutation == 'nonpositive_deformed_J':
        nodes['values'][4:, 2] = nodes['values'][:4, 2] - .001
        nodes['values'][:, 3:6] = nodes['values'][:, :3] - np.asarray(mesh['rest_nodes_m'])
    elif mutation == 'nonpositive_logged_J':
        elements['values'][0, 6] = 0
    elif mutation == 'nonfinite_stress':
        elements['values'][0, 0] = math.nan
    elif mutation == 'time':
        elements['time'] += 1e-5
    elif mutation == 'step':
        elements['step'] = 59
    else:
        nodes['values'][0, 3] = 1e-5
    if mutation in {'missing_top_bc', 'nonpositive_deformed_J', 'nonpositive_logged_J', 'nonfinite_stress', 'time', 'step'}:
        with pytest.raises(ValueError):
            frame.evaluate_generated_frame(spec, mesh, nodes, elements, reconstruction=reconstruction)
    else:
        assert not frame.evaluate_generated_frame(spec, mesh, nodes, elements,
                                                  reconstruction=reconstruction)['fixture_passed']


def test_deck_hash_endpoint_and_ancestry_corruption_refused():
    spec, source, adapted, receipt = contract()
    assert spec['source_deck_sha256'] == receipt['source_deck_sha256']
    for changed in (adapted.replace('2000', '3000'), adapted.replace('accelerate', 'skyline')):
        with pytest.raises(ValueError):
            frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES, spec['run_id'], source, changed, receipt)
    bad = bytearray(STUDY_BYTES)
    bad[4] = ord('X')
    with pytest.raises(ValueError, match='SHA256'):
        frame.verified_schedule(bytes(bad), PRIOR_BYTES, spec['run_id'], source, adapted, receipt)
    with pytest.raises(ValueError, match='ancestry'):
        frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES + b' ', spec['run_id'], source, adapted, receipt)
    bad_receipt = deepcopy(receipt)
    bad_receipt['native_boundary_factor'] = .5
    with pytest.raises(ValueError, match='receipt'):
        frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES, spec['run_id'], source, adapted, bad_receipt)
    changed = ET.fromstring(adapted)
    points = changed.findall('LoadData/load_controller')[1].find('points')
    endpoint = spec['full_coordinates_m'][-1]
    points[-1].text = f'1,{math.nextafter(endpoint, 0.0):.17g}'
    with pytest.raises(ValueError, match='adapted deck'):
        frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES, spec['run_id'], source,
                                ET.tostring(changed), receipt)


@pytest.mark.parametrize('branch,N', [('compression', 36), ('tension', 24)])
def test_s120_exact_even_states_and_closed_release(branch, N):
    run_id = f'{branch}:N{N}:S120:reference'
    source = fixture_source(branch, 'lower_half_reconstructed')
    _, adapted, receipt = v5.adapt_deck(STUDY, json.loads(PRIOR_BYTES), run_id, source)
    fine = frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES, run_id, source, adapted, receipt)
    base = contract(branch, 'lower_half_reconstructed')[0]
    assert len(fine['times']) == 121
    assert fine['full_coordinates_m'][::2] == base['full_coordinates_m']
    assert fine['native_coordinates_m'][::2] == base['native_coordinates_m']
    assert fine['source_binding_checked'] is False
    assert fine['native_execution_released'] is False


@pytest.mark.parametrize('bad', ('duplicate_id', 'missing_id', 'wrong_time'))
def test_strict_parser_refuses_mismatched_generated_ids_or_time(bad):
    times = [i / 60 for i in range(61)]
    lines = []
    for step, t in enumerate(times):
        lines += [f'*Step = {step}\n', f'*Time = {t:.9g}\n', '*Data = mechanics_nodes_si\n']
        ids = [1, 2]
        if step == 5 and bad == 'duplicate_id':
            ids = [1, 1]
        if step == 5 and bad == 'missing_id':
            ids = [1]
        if step == 5 and bad == 'wrong_time':
            lines[-2] = '*Time = 0.12345\n'
        lines += [f'{node},0,0,0,0,0,0,0,0,0\n' for node in ids]
    with pytest.raises(ValueError):
        list(_iter_data_records(lines, expected_times=times, item_count=2, field_count=9,
                                record_name='mechanics_nodes_si', maximum_bytes=1024**2,
                                maximum_items=2))
