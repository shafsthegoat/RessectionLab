"""Tiny generated controls only. No realistic-size evaluation in this suite."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import os
import sys

import numpy as np
import pytest

PROHIBITED = []
def guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'socket.connect'):
        PROHIBITED.append(event)
        raise AssertionError('No processes/network in generated kernel tests')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower()
        if name.endswith(('.nii', '.nii.gz', '.mat', '.tar', '.pt', '.ckpt', '.bin',
                          '.npz', '.npy', '.pkl', '.safetensors', '.dcm', '.h5')):
            PROHIBITED.append(name)
            raise AssertionError('No patient/model payloads')
sys.addaudithook(guard)

loader = importlib.util.spec_from_file_location('streaming_candidate', Path(__file__).with_name('streaming_contact.py'))
s = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = s
loader.loader.exec_module(s)
from resectionlab.evaluation import segment_box_distance_sq


def fixture(angle=0., coverage='partial_x_half', pattern='lattice'):
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([.7, 1.1, 1.6])
    affine[:3, 3] = [3., -5., 1.]
    grid = s.Grid((9, 8, 7), affine)
    def cap(action, part, a, b, radius):
        return s.Capsule(action, part, rotation @ a + affine[:3, 3],
                         rotation @ b + affine[:3, 3], radius)
    capsules = (cap('first', 'shaft', np.array([2., 3., -2.]), np.array([4., 4., 8.]), .6),
                cap('first', 'tip', np.array([2.4, 3.2, 1.]), np.array([4., 4., 9.]), .9),
                cap('second', 'shaft', np.array([2., 3., -2.]), np.array([4., 4., 8.]), .6),
                cap('second', 'tip', np.array([3., 4., 2.]), np.array([5., 6., 8.]), .7))
    return s.GeneratedReference(grid, pattern, coverage), capsules


def oracle(reference, capsules):
    affine = np.asarray(reference.grid.affine_ras_mm)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    rotation = affine[:3, :3] / spacing
    parts = {'shaft': set(), 'tip': set(), 'whole_tool': set()}
    actions = {c.action_id: set() for c in capsules}
    outside = {}
    for c in capsules:
        a, b = [rotation.T @ (np.asarray(v) - affine[:3, 3]) for v in (c.start_ras_mm, c.end_ras_mm)]
        hits = set()
        for cell in np.ndindex(reference.grid.shape):
            centre = np.asarray(cell)*spacing
            if segment_box_distance_sq(a, b, centre-spacing/2, centre+spacing/2) <= c.radius_mm**2 + 1e-10:
                hits.add(cell)
        parts[c.part] |= hits
        parts['whole_tool'] |= hits
        actions[c.action_id] |= hits
        out = bool(np.any(np.minimum(a, b)-c.radius_mm < -.5*spacing)
                   or np.any(np.maximum(a, b)+c.radius_mm > (np.asarray(reference.grid.shape)-.5)*spacing))
        outside[c.part] = outside.get(c.part, False) or out
        outside[c.action_id] = outside.get(c.action_id, False) or out
    outside['whole_tool'] = any(outside.values())
    volume = abs(np.linalg.det(affine[:3, :3]))
    def counts(cells, out):
        indices = np.array(sorted(cells), dtype=np.int64).reshape(-1, 3)
        positive, known = reference.sample(indices)
        return s._count_record((len(cells), int(positive.sum()), int((~known).sum())), out, volume)
    return {name: counts(cells, outside.get(name, False)) for name, cells in parts.items()}, {
        name: counts(cells, outside.get(name, False)) for name, cells in actions.items()}


@pytest.mark.parametrize('angle', [0., .37, np.pi/2])
@pytest.mark.parametrize('edge', [3, 4, 16])
def test_scalar_full_grid_equivalence_rotated_anisotropic_tiles(angle, edge):
    ref, caps = fixture(angle)
    expected, actions = oracle(ref, caps)
    result = s.evaluate_generated_contacts(ref, caps, budget=s.Budget(tile_edge=edge, coarse_batch=7))
    assert {key: result[key] for key in expected} == expected
    assert result['per_action'] == actions
    assert result['work']['maximum_tile_cells'] <= edge**3
    assert result['work']['tiles_scanned'] == result['work']['tiles_pruned'] + result['work']['tiles_evaluated']


@pytest.mark.parametrize('coverage', ['full', 'partial_x_half', 'none'])
def test_repeated_sweeps_and_actions_do_not_double_count(coverage):
    ref, caps = fixture(coverage=coverage, pattern='all_covered')
    a = s.evaluate_generated_contacts(ref, caps[:2])
    b = s.evaluate_generated_contacts(ref, caps[:2] + caps[:2])
    c = s.evaluate_generated_contacts(ref, caps[:2] + tuple(replace(x, action_id='repeat') for x in caps[:2]))
    for key in ('shaft', 'tip', 'whole_tool'):
        assert a[key] == b[key] == c[key]
    assert a['per_action'] == b['per_action']
    assert c['per_action']['first'] == c['per_action']['repeat'] == a['whole_tool']


@pytest.mark.parametrize('offset', [0., .5e-5, 1e-5, 1.1e-5])
def test_zero_radius_tolerance_tangency_across_tile_face(offset):
    ref = s.GeneratedReference(s.Grid((8, 8, 8), np.eye(4)), 'all_covered', 'full')
    cap = s.Capsule('tangent', 'shaft', (3.5+offset, 2., 2.), (3.5+offset, 6., 6.), 0.)
    expected, actions = oracle(ref, (cap,))
    result = s.evaluate_generated_contacts(ref, (cap,), budget=s.Budget(tile_edge=4))
    assert result['shaft'] == expected['shaft']
    assert result['per_action'] == actions


def test_partial_and_outside_no_hit_stays_unknown():
    ref, caps = fixture(pattern='empty')
    result = s.evaluate_generated_contacts(ref, caps)
    assert result['whole_tool']['annotated_positive_encounter'] is None
    assert result['whole_tool']['outside_reference_fov'] is True
    assert result['whole_tool']['unknown_reference_cells'] > 0
    assert result['whole_tool']['clinical_injury_probability'] is None
    assert result['removed_overlap']['outcomes'] is None


@pytest.mark.parametrize('name', ['max_tile_capsule_pairs', 'max_cell_capsule_pairs', 'max_sampled_cells'])
def test_budget_refuses_without_completed_or_partial_outcomes(name):
    ref, caps = fixture()
    with pytest.raises(s.BudgetExceeded) as error:
        s.evaluate_generated_contacts(ref, caps, budget=replace(s.Budget(), **{name: 0}))
    assert error.value.completed is False and error.value.outcomes is None
    assert error.value.work[name.removeprefix('max_').replace('sampled_cells', 'reference_sampled_cells')] == 0
    assert error.value.rejected_charge['limit'] == 0
    assert error.value.rejected_charge['requested'] > 0
    assert error.value.rejected_charge['projected'] == error.value.rejected_charge['requested']
    assert error.value.budget[name] == 0


def test_cell_budget_refuses_before_voxel_allocation(monkeypatch):
    ref, caps = fixture()
    monkeypatch.setattr(s, '_indices', lambda *a: pytest.fail('refused work allocated a voxel tile'))
    with pytest.raises(s.BudgetExceeded, match='cell_capsule_pairs'):
        s.evaluate_generated_contacts(ref, caps, budget=s.Budget(max_cell_capsule_pairs=0))


def test_cancel_preserves_work_only():
    ref, caps = fixture()
    calls = []
    def cancel():
        calls.append(1)
        return len(calls) > 10
    with pytest.raises(s.BudgetExceeded, match='cancelled') as error:
        s.evaluate_generated_contacts(ref, caps, cancelled=cancel)
    assert error.value.outcomes is None
    assert 'positive_reference_cells' not in error.value.work


def test_stop_empty_history_does_not_scan_reference():
    ref, _ = fixture()
    result = s.evaluate_generated_contacts(ref, ())
    assert all(value == 0 for value in result['work'].values())
    assert result['whole_tool']['touched_reference_cells'] == 0
    assert result['per_action'] == {}


def test_all_pruned_run_still_records_coarse_geometry_work():
    ref, _ = fixture()
    cap = s.Capsule('outside', 'shaft', (-100., -100., -100.), (-90., -90., -90.), 1.)
    result = s.evaluate_generated_contacts(ref, (cap,), budget=s.Budget(tile_edge=4, coarse_batch=7))
    assert result['work']['maximum_coarse_geometry_batch'] == 7
    assert result['work']['maximum_cell_geometry_batch'] == 0
    assert result['work']['maximum_geometry_batch'] == 7
    assert result['work']['tiles_pruned'] == result['work']['tiles_scanned']
    assert result['budget']['coarse_batch'] == 7


def test_grid_size_validation_alone_does_not_evaluate_large_grid():
    grid = s.Grid((256, 256, 192), np.diag([.7, .7, .7, 1.]))
    assert grid.shape == (256, 256, 192)
    with pytest.raises(ValueError):
        s.Grid((1024, 1024, 1024), np.eye(4))
    shear = np.eye(4); shear[0, 1] = .1
    with pytest.raises(ValueError):
        s.Grid((8, 8, 8), shear)


def test_reference_swap_changes_outcomes_without_changing_geometry_work():
    ref, caps = fixture()
    a = s.evaluate_generated_contacts(replace(ref, pattern='empty'), caps)
    b = s.evaluate_generated_contacts(replace(ref, pattern='all_covered'), caps)
    assert a['capsules_identity'] == b['capsules_identity']
    assert a['work'] == b['work']
    assert a['whole_tool']['positive_reference_cells'] == 0
    assert b['whole_tool']['positive_reference_cells'] > 0


def test_only_fixed_generated_provider_is_admitted():
    ref, caps = fixture()
    with pytest.raises(ValueError, match='generated_scope_only'):
        s.evaluate_generated_contacts(lambda *a: (True, True), caps)
    class Derived(s.GeneratedReference):
        pass
    with pytest.raises(ValueError, match='generated_scope_only'):
        s.evaluate_generated_contacts(Derived(ref.grid), caps)


def test_no_prohibited_actions():
    assert PROHIBITED == []
