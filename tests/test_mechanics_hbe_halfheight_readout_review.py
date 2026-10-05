"""Independent analytical review; no native solver or saved specimen arrays.

End-to-end text fixtures reuse only the owner's labelled two-cell log factory.
The adversaries/expectations below are independent; no physics/parser is mocked.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import mechanics_hbe_halfheight_readout as h

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('halfheight_analytic_factory', ROOT/'tests/test_mechanics_hbe_halfheight_readout.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def rewrite(root, binding, transform):
    path = root/binding['path']
    value = json.loads(path.read_text())
    transform(value)
    path.write_text(json.dumps(value))
    binding['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_loading_reconstruction(root, args):
    rewrite(root, args['half_bindings']['loading'],
            lambda value: value.update(reconstruction_sha256=args['reconstruction_binding']['sha256']))


@pytest.mark.parametrize('branch,sign', [('compression', -1), ('tension', 1)])
def test_signed_work_uses_half_coordinate_and_full_original_height(tmp_path, monkeypatch, branch, sign):
    args = fixture.analytic_files(tmp_path, monkeypatch, branch=branch)
    result = h.read_halfheight_equivalence(tmp_path, **args)
    assert result['passed']
    half, full = result['half_native'], result['reconstructed_full']
    d = np.asarray(full['load_coordinate_m'])
    assert d[-1] == pytest.approx(sign*.15*h.FULL_HEIGHT_M)
    np.testing.assert_allclose(half['load_coordinate_m'], d/2, rtol=0, atol=0)
    force = np.asarray(full['applied_force_N'])
    assert sign*force[-1] > 0
    np.testing.assert_allclose(half['force_from_bottom_N'], force, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(half['force_from_midplane_N'], force, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(2*np.asarray(half['energy_work']['work_J']),
                               full['energy_work']['work_J'], rtol=1e-12, atol=1e-16)
    assert full['energy_work']['work_J'][-1] > 0
    assert full['native_full_solver_claim'] is False


def test_reflected_moment_uses_current_origin_translation_not_polar_torque():
    m = h.HalfHeightReconstruction(*fixture.geometry())
    d = .15*m.height
    current = m.Xh.copy()
    current[:, 2] *= 1.15
    raw = np.zeros_like(current)
    raw[0, 0] = 2.  # Deliberately unbalanced force: an algebra control, not equilibrium.
    lifted = m.lift(current, raw, np.ones((1, 8)), d)
    frame = m.full.read_frame(lifted['current_nodes_m'], lifted['raw_reactions_N'],
                              mu_Pa=1000., branch='tension', fraction=1.)
    # Lower/upper forces at (-R,-R,0) and (-R,-R,H+d) point in +x.
    np.testing.assert_allclose(frame['net_moment_Nm'], [0., 2*(m.height+d), 4*m.radius], rtol=1e-14, atol=1e-18)
    assert frame['checks']['moment_balance'] is False


def test_nonzero_cut_shear_reactions_are_summed_then_rejected():
    m = h.HalfHeightReconstruction(*fixture.geometry())
    raw = np.zeros_like(m.Xh)
    raw[4:, :] = [1e-4, -2e-4, 3e-4]
    lifted = m.lift(m.Xh, raw, np.ones((1, 8)), 0.)
    np.testing.assert_array_equal(lifted['raw_reactions_N'][4:8], np.tile([2e-4, -4e-4, 0.], (4, 1)))
    half = m.half_frame(m.Xh, raw, full_displacement_m=0.)
    assert half['ratios']['free_dof_reaction'] > 1
    full = m.full.read_frame(lifted['current_nodes_m'], lifted['raw_reactions_N'],
                             mu_Pa=1000., branch='tension', fraction=0.)
    assert full['checks']['free_node_reactions'] is False


@pytest.mark.parametrize('corruption', ['float_node_ids', 'boolean_node_ids', 'noncube_permutation', 'stale_mesh_fingerprint'])
def test_malformed_mapping_is_rejected(corruption):
    full, half, mapping = fixture.geometry()
    if corruption == 'float_node_ids':
        mapping['reflections'][1]['node_full_ids'] = [float(x) for x in mapping['reflections'][1]['node_full_ids']]
    elif corruption == 'boolean_node_ids':
        mapping['reflections'][1]['node_full_ids'] = [True]*8
    elif corruption == 'noncube_permutation':
        mapping['reflections'][1]['element_half_to_full_local'] = [[5, 4, 6, 7, 0, 1, 2, 3]]
    else:
        mapping['full_mesh_fingerprint'] = '0'*64
    with pytest.raises(ValueError):
        h.HalfHeightReconstruction(full, half, mapping)


def test_external_mapping_and_geometry_edits_do_not_mutate_built_snapshot():
    full, half, mapping = fixture.geometry()
    m = h.HalfHeightReconstruction(full, half, mapping)
    before = m.lift(m.Xh, np.zeros_like(m.Xh), np.ones((1, 8)), 0.)
    half['rest_nodes_m'][0][0] = 123.
    mapping['reflections'][1]['node_full_ids'][0] = 1
    full['elements_hex8'][0][0] = 9
    after = m.lift(m.Xh, np.zeros_like(m.Xh), np.ones((1, 8)), 0.)
    np.testing.assert_array_equal(after['current_nodes_m'], before['current_nodes_m'])
    np.testing.assert_array_equal(after['raw_reactions_N'], before['raw_reactions_N'])


@pytest.mark.parametrize('field', ['declaration', 'full_mesh'])
def test_resealed_wrapper_cannot_switch_bound_reference(tmp_path, monkeypatch, field):
    args = fixture.analytic_files(tmp_path, monkeypatch)
    rewrite(tmp_path, args['reconstruction_binding'],
            lambda value: value[field].update(sha256='0'*64))
    refresh_loading_reconstruction(tmp_path, args)
    with pytest.raises(ValueError, match='provenance'):
        h.read_halfheight_equivalence(tmp_path, **args)


def test_resealed_half_loading_cannot_use_half_height_for_full_d(tmp_path, monkeypatch):
    args = fixture.analytic_files(tmp_path, monkeypatch)
    rewrite(tmp_path, args['half_bindings']['loading'],
            lambda value: value.update(load_coordinate=(np.asarray(value['load_coordinate'])/2).tolist()))
    with pytest.raises(ValueError, match='full physical loading grid'):
        h.read_halfheight_equivalence(tmp_path, **args)


@pytest.mark.parametrize('corruption', ['missing_middle', 'duplicate_final', 'zero_logged_J'])
def test_complete_states_and_positive_logged_j_remain_required(tmp_path, monkeypatch, corruption):
    args = fixture.analytic_files(tmp_path, monkeypatch)
    binding = args['half_bindings']['elements']
    path = tmp_path/binding['path']
    blocks = path.read_text().split('*Step = ')[1:]
    if corruption == 'missing_middle':
        del blocks[30]
    elif corruption == 'duplicate_final':
        blocks.append(blocks[-1])
    else:
        lines = blocks[30].splitlines()
        fields = lines[3].split(',')
        fields[7] = '0'  # one-based element ID, six stress fields, then J.
        lines[3] = ','.join(fields)
        blocks[30] = '\n'.join(lines)+'\n'
    path.write_text(''.join('*Step = '+block for block in blocks))
    binding['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        h.read_halfheight_equivalence(tmp_path, **args)


def test_logged_sed_does_not_replace_independent_three_field_energy(tmp_path, monkeypatch):
    args = fixture.analytic_files(tmp_path, monkeypatch)
    binding = args['half_bindings']['elements']
    path = tmp_path/binding['path']
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith('1,'):
            fields = line.split(','); fields[-1] = '1e12'; lines[index] = ','.join(fields)
    path.write_text('\n'.join(lines))
    binding['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = h.read_halfheight_equivalence(tmp_path, **args)
    assert result['passed'] and result['logged_sed_used_for_energy_gate'] is False
    assert result['equivalence']['diagnostics']['logged_sed_Pa'] > 1e11
    assert max(result['half_native']['energy_J']) < 1.


def test_input_changed_after_initial_binding_cannot_return_passed_receipt(tmp_path, monkeypatch):
    args = fixture.analytic_files(tmp_path, monkeypatch)
    original = h.read_run
    path = tmp_path/args['half_bindings']['nodes']['path']
    def change_then_return(*pos, **kw):
        result = original(*pos, **kw)
        path.write_text(path.read_text()+'\n')  # Numerically identical; byte identity still matters.
        return result
    monkeypatch.setattr(h, 'read_run', change_then_return)
    with pytest.raises(ValueError, match='hash'):
        h.read_halfheight_equivalence(tmp_path, **args)
