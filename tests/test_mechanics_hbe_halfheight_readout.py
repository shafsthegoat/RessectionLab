"""Analytical two-cell controls only; no HBE values or native solver calls."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts import mechanics_hbe_halfheight_readout as h


def geometry():
    R, H = h.RADIUS_M, h.FULL_HEIGHT_M
    square = np.array([[-R, -R], [R, -R], [R, R], [-R, R]])
    X = np.array([[x, y, z] for z in (0, H / 2, H) for x, y in square])
    full = {
        'rest_nodes_m': X.tolist(), 'elements_hex8': [list(range(1, 9)), list(range(5, 13))],
        'node_ids': list(range(1, 13)), 'element_ids': [1, 2], 'indexing': 'one_based', 'mesh_N': 8,
        'boundaries': {'bottom': {'node_ids': [1, 2, 3, 4]}, 'top': {'node_ids': [9, 10, 11, 12]}},
        'geometry': {'radius_m': R, 'height_m': H},
    }
    half = deepcopy(full)
    half.update(rest_nodes_m=X[:8].tolist(), elements_hex8=[list(range(1, 9))],
                node_ids=list(range(1, 9)), element_ids=[1], full_height_m=H,
                full_geometry=deepcopy(full['geometry']),
                boundary_roles={'top': 'artificial_midplane', 'bottom': 'physical_bonded_plate', 'side': 'traction_free_outer'})
    half['geometry']['height_m'] = H / 2
    half['boundaries']['top']['node_ids'] = [5, 6, 7, 8]
    mapping = {
        'schema': 'hbe-halfheight-mapping-v1',
        'full_mesh_fingerprint': h._fingerprint(full), 'half_mesh_fingerprint': h._fingerprint(half),
        'half_to_full_node_ids': list(range(1, 9)), 'half_to_full_element_ids': [1],
        'reflections': [
            {'signs': [1, 1, 1], 'rest_offset_m': [0., 0., 0.],
             'node_full_ids': list(range(1, 9)), 'element_full_ids': [1],
             'element_half_to_full_local': [list(range(8))]},
            {'signs': [1, 1, -1], 'rest_offset_m': [0., 0., H],
             'node_full_ids': [9, 10, 11, 12, 5, 6, 7, 8], 'element_full_ids': [2],
             'element_half_to_full_local': [[4, 5, 6, 7, 0, 1, 2, 3]]},
        ],
    }
    return full, half, mapping


def model():
    return h.HalfHeightReconstruction(*geometry())


def test_lift_merges_motion_sums_reactions_and_reflects_tensor():
    m = model()
    d = .15 * m.height
    current = m.Xh.copy()
    current[:, 2] *= 1.15
    current[4:, 0] += 1e-4  # Midplane tangential motion is free, not clamped.
    raw = np.zeros_like(current)
    raw[:4] = [1., 2., 3.]
    raw[4:] = [4., 5., -6.]
    elements = np.array([[1., 2., 3., 4., 5., 6., 1.15, 8.]])
    lifted = m.lift(current, raw, elements, d)
    np.testing.assert_allclose(lifted['current_nodes_m'][4:8, 0] - m.X[4:8, 0], 1e-4)
    np.testing.assert_allclose(lifted['current_nodes_m'][8:, 2] - m.X[8:, 2], d)
    np.testing.assert_array_equal(lifted['raw_reactions_N'][4:8], np.tile([8., 10., 0.], (4, 1)))
    np.testing.assert_array_equal(lifted['raw_reactions_N'][8:], np.tile([1., 2., -3.], (4, 1)))
    np.testing.assert_array_equal(lifted['logged_elements'][1], [1, 2, 3, 4, -5, -6, 1.15, 8])


def test_energy_doubles_force_stays_one_and_half_cut_tangents_are_free():
    m = model()
    d = .15 * m.height
    current = m.Xh.copy()
    current[:, 2] *= 1.15
    current[4:, 0] += 2e-5
    raw = np.zeros_like(current)
    raw[:4, 2] = 2.
    raw[4:, 2] = -2.
    half = m.half_frame(current, raw, full_displacement_m=d)
    lifted = m.lift(current, raw, np.ones((1, 8)), d)
    full = m.full.read_frame(lifted['current_nodes_m'], lifted['raw_reactions_N'],
                             mu_Pa=1000., branch='tension', fraction=1.)
    assert half['ratios']['prescribed_motion'] < 1e-5
    assert half['ratios']['free_dof_reaction'] == 0
    assert full['applied_force_N'] == half['force_from_bottom_N'] == half['force_from_midplane_N'] == 8.
    assert full['energy_J'] == pytest.approx(2 * half['energy_J'], rel=1e-12)
    raw[4, 0] = 1e-4
    assert m.half_frame(current, raw, full_displacement_m=d)['ratios']['free_dof_reaction'] > 1


def test_inconsistent_midplane_is_rejected_without_averaging():
    m = model()
    current = m.Xh.copy()
    current[4:, 2] += .15 * m.height / 2
    current[4, 2] += 2 * m.tolerance
    with pytest.raises(ValueError, match='disagree'):
        m.lift(current, np.zeros_like(current), np.ones((1, 8)), .15 * m.height)


@pytest.mark.parametrize('corruption', ['duplicate_node', 'wrong_cell', 'local_permutation', 'wrong_plane', 'modified_original'])
def test_map_corruption_is_rejected(corruption):
    full, half, mapping = geometry()
    if corruption == 'duplicate_node':
        mapping['reflections'][1]['node_full_ids'][0] = 10
    elif corruption == 'wrong_cell':
        mapping['reflections'][1]['element_full_ids'] = [1]
    elif corruption == 'local_permutation':
        mapping['reflections'][1]['element_half_to_full_local'] = [list(range(8))]
    elif corruption == 'wrong_plane':
        mapping['reflections'][1]['rest_offset_m'][2] /= 2
    else:
        half['rest_nodes_m'][0][0] += 1e-14
        mapping['half_mesh_fingerprint'] = h._fingerprint(half)
    with pytest.raises(ValueError):
        h.HalfHeightReconstruction(full, half, mapping)


def analytic_files(root, monkeypatch, *, branch='tension', wrong_half_sign=False, missing_final=False):
    """Bind constructed logs to a visibly analytical test declaration.

    Only the prospective file pins/counts are replaced for this tiny fixture.
    The real parser, mesh physics, full readout and comparison execute unchanged.
    """
    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    full_mesh, half_mesh, mapping = geometry()
    full = {'mesh': save('full/mesh.json', full_mesh), 'deck': save('full/deck.feb', 'analytical-no-native-deck')}
    half = {'mesh': save('half/mesh.json', half_mesh), 'deck': save('half/deck.feb', 'analytical-no-native-deck')}
    protocol = save('protocol.json', {'purpose': 'analytical test, no specimen measurements'})
    monkeypatch.setattr(h, 'ORIGINAL_PROTOCOL_SHA256', protocol['sha256'])
    sign = -1 if branch == 'compression' else 1
    coordinate = h.TIMES * sign * .15 * h.FULL_HEIGHT_M
    loading_base = {
        'branch': branch, 'steps': 60, 'mu_Pa': 1000., 'K_Pa': 149000. / 3,
        'times': h.TIMES.tolist(), 'load_coordinate': coordinate.tolist(),
        'load_coordinate_units': 'm', 'node_fields': 'x;y;z;ux;uy;uz;Rx;Ry;Rz',
        'element_fields': 'sx;sy;sz;sxy;syz;sxz;J;sed', 'raw_reaction_convention': 'body_on_constraint',
        'min_residual_N2': (1e-10 * h.MU_PA * h.RADIUS_M ** 2) ** 2,
        'protocol_sha256': protocol['sha256'],
    }
    full['loading'] = save('full/loading.json', {**loading_base, 'prescribed_dofs': {'top': 'xyz', 'bottom': 'xyz'},
                                              'mesh_sha256': full['mesh']['sha256'], 'deck_sha256': full['deck']['sha256']})
    for role, mesh, bindings in (('full', full_mesh, full), ('half', half_mesh, half)):
        X = np.asarray(mesh['rest_nodes_m'])
        bottom = np.array(mesh['boundaries']['bottom']['node_ids']) - 1
        top = np.array(mesh['boundaries']['top']['node_ids']) - 1
        nodes, elements, solver = [], [], []
        for step, t in enumerate(h.TIMES):
            if role == 'half' and missing_final and step == 60:
                # Keep a complete solver-status stream so this control reaches
                # the separate missing-primitive-state check.
                solver.extend(['Nonlinear solution status: time= 1', ' residual 1e-6 1e-18 1e-14'])
                continue
            stretch = 1 + sign * .15 * t
            current = X.copy()
            current[:, 2] *= stretch
            mu, K = 1000., 149000. / 3
            dW = mu / 2 * (2 * stretch * stretch ** (-2 / 3) - (2 / 3) * (2 + stretch ** 2) * stretch ** (-5 / 3)) + K / 2 * (stretch - 1 / stretch)
            force = dW * (2 * h.RADIUS_M) ** 2
            raw = np.zeros_like(X)
            raw[bottom, 2] = force / 4
            raw[top, 2] = -force / 4
            if role == 'half' and wrong_half_sign:
                raw *= -1
            values = np.column_stack((current, current - X, raw))
            nodes.extend([f'*Step = {step}', f'*Time = {t:.9g}', '*Data = mechanics_nodes_si'])
            nodes.extend(str(i) + ',' + ','.join(f'{v:.12g}' for v in row) for i, row in enumerate(values, 1))
            W = mu / 2 * ((2 + stretch ** 2) * stretch ** (-2 / 3) - 3) + K / 4 * (stretch ** 2 - 1 - 2 * np.log(stretch))
            elements.extend([f'*Step = {step}', f'*Time = {t:.9g}', '*Data = mechanics_elements_si'])
            elements.extend(f'{i},0,0,0,0,0,0,{stretch:.12g},{W:.12g}' for i in mesh['element_ids'])
            if step:
                solver.extend([f'Nonlinear solution status: time= {t:.6g}', ' residual 1e-6 1e-18 1e-14'])
        solver.append('N O R M A L T E R M I N A T I O N')
        for key, rows in (('nodes', nodes), ('elements', elements), ('solver', solver)):
            bindings[key] = save(f'{role}/{key}.log', '\n'.join(rows))
    declaration = {
        'schema': 'hbe-halfheight-equivalence-v1', 'original_protocol': protocol,
        'full_references': {f'{branch}:N8:S60:reference': {'primitive_bindings': full,
                            'readout': save('old-readout.json', {'analytical': True}),
                            'execution': save('old-execution.json', {'native_execution': False})}},
        'extraction': {'expected_counts': {'8': {'full_elements': 2, 'full_nodes': 12, 'half_elements': 1, 'half_nodes': 8, 'midplane_nodes': 4}}},
    }
    declaration_binding = save('declaration.json', declaration)
    monkeypatch.setattr(h, 'DECLARATION_SHA256', declaration_binding['sha256'])
    wrapper = {'schema': 'hbe-halfheight-reconstruction-v1', 'mapping': mapping, 'verification': {},
               'full_mesh': full['mesh'], 'half_mesh': half['mesh'], 'declaration': declaration_binding,
               'mesh_source': save('analytic-source.txt', 'constructed two-cell algebra fixture')}
    reconstruction_binding = save('reconstruction.json', wrapper)
    half['loading'] = save('half/loading.json', {
        **loading_base, 'prescribed_dofs': {'bottom': 'xyz', 'midplane': 'z'},
        'full_height_m': h.FULL_HEIGHT_M, 'retained_height_m': h.FULL_HEIGHT_M / 2,
        'mesh_sha256': half['mesh']['sha256'], 'deck_sha256': half['deck']['sha256'],
        'full_mesh_sha256': full['mesh']['sha256'], 'reconstruction_sha256': reconstruction_binding['sha256'],
        'halfheight_equivalence_declaration_sha256': declaration_binding['sha256'],
    })
    return dict(half_bindings=half, full_bindings=full, reconstruction_binding=reconstruction_binding,
                declaration_binding=declaration_binding, expected_branch=branch, expected_mesh_N=8)


@pytest.mark.parametrize('branch', ['compression', 'tension'])
def test_complete_analytic_stream_checks_force_one_energy_two_and_provenance(tmp_path, monkeypatch, branch):
    args = analytic_files(tmp_path, monkeypatch, branch=branch)
    result = h.read_halfheight_equivalence(tmp_path, **args)
    assert result['passed']
    assert result['frame_count'] == 61
    assert result['full_native']['passed'] and result['half_native']['passed']
    assert result['reconstructed_full']['native_full_solver_claim'] is False
    assert np.asarray(result['reconstructed_full']['probe_displacements_m']).shape == (61, 75, 3)
    assert result['equivalence']['criteria']['bottom_axial_force_scale_one']['actual'] < .001
    assert result['equivalence']['criteria']['half_energy_scale_two']['actual'] < .001


def test_signed_reaction_corruption_is_a_negative_result(tmp_path, monkeypatch):
    result = h.read_halfheight_equivalence(tmp_path, **analytic_files(tmp_path, monkeypatch, wrong_half_sign=True))
    assert not result['passed']
    assert result['half_native']['criteria']['work_energy']['actual'] > 1
    assert result['equivalence']['criteria']['raw_reaction']['actual'] > 1


def test_missing_final_state_never_gets_reflected_or_filled(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='Missing final'):
        h.read_halfheight_equivalence(tmp_path, **analytic_files(tmp_path, monkeypatch, missing_final=True))


def test_wrong_reference_and_binding_drift_rejected_before_reports(tmp_path, monkeypatch):
    args = analytic_files(tmp_path, monkeypatch)
    changed = deepcopy(args)
    changed['full_bindings']['nodes']['sha256'] = '1' * 64
    with pytest.raises(ValueError, match='prospectively selected'):
        h.read_halfheight_equivalence(tmp_path, **changed)
    (tmp_path / args['half_bindings']['nodes']['path']).write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        h.read_halfheight_equivalence(tmp_path, **args)


def test_halfheight_cannot_enter_original_full_readout(tmp_path, monkeypatch):
    args = analytic_files(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='geometry'):
        h.read_run(tmp_path, args['half_bindings'], protocol_sha256=h.ORIGINAL_PROTOCOL_SHA256,
                   expected_branch='tension', expected_mesh_N=8, expected_steps=60, expected_mu_Pa=1000.)


@pytest.mark.parametrize('field,value', [('prescribed_dofs', {'bottom': 'xyz', 'midplane': 'xyz'}),
                                        ('full_height_m', h.FULL_HEIGHT_M / 2),
                                        ('reconstruction_sha256', '0' * 64),
                                        ('halfheight_equivalence_declaration_sha256', '0' * 64)])
def test_rehashed_half_loading_cannot_change_fixture_or_reference(tmp_path, monkeypatch, field, value):
    args = analytic_files(tmp_path, monkeypatch)
    binding = args['half_bindings']['loading']
    path = tmp_path / binding['path']
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    binding['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Half loading'):
        h.read_halfheight_equivalence(tmp_path, **args)
