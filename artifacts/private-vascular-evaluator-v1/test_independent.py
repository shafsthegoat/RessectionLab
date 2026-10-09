"""Independent generated controls; no native solver or patient/model payloads."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
path = ROOT/'tests/test_private_vascular_evaluation.py'
loader = importlib.util.spec_from_file_location('vascular_tracked_author_controls_for_review', path)
author = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = author
loader.loader.exec_module(author)
v = author.v
assert Path(v.__file__).resolve() == ROOT/'src/resectionlab/private_vascular_evaluation.py'
setup = author.setup
from resectionlab.evaluation import segment_box_distance_sq

PROHIBITED_ACTIONS = []
def generated_only_guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'socket.connect'):
        PROHIBITED_ACTIONS.append(event)
        raise AssertionError('Native processes/network forbidden in generated evaluator review')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower()
        if name.endswith(('.nii', '.nii.gz', '.mat', '.tar', '.pt', '.ckpt', '.bin',
                          '.npz', '.npy', '.pkl', '.pickle', '.safetensors', '.dcm',
                          '.h5', '.hdf5', '.mha', '.mhd', '.nrrd')):
            PROHIBITED_ACTIONS.append(name)
            raise AssertionError('Patient/model array payload forbidden in generated evaluator review')
sys.addaudithook(generated_only_guard)


@pytest.mark.parametrize('angle', [0., .37, np.pi/2])
def test_contact_cells_match_scalar_box_oracle_under_rotated_anisotropic_grid(angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    spacing = np.array([.7, 1.1, 1.6])
    affine = np.eye(4)
    affine[:3,:3] = rotation @ np.diag(spacing)
    affine[:3,3] = [3.2, -1.7, .4]
    local_start, local_end = np.array([-.8, 1.1, 2.]), np.array([3., 2.3, 4.])
    radius = .43
    start, end = [rotation@point + affine[:3,3] for point in (local_start, local_end)]
    actual, outside = v._capsule_cells(start, end, radius, (4,4,4), affine, lambda: False)
    expected = set()
    for cell in np.ndindex((4,4,4)):
        centre = np.array(cell)*spacing
        distance = segment_box_distance_sq(local_start, local_end, centre-spacing/2, centre+spacing/2)
        if distance <= radius**2 + 1e-10:
            expected.add(cell)
    assert actual == expected
    assert outside is True


def test_forward_transform_and_signed_permutation_map_removed_roi_cell_exactly():
    planning_affine = np.eye(4)
    planning_affine[:3,3] = [2.,3.,4.]
    transform = np.eye(4)
    transform[:3,:3] = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    transform[:3,3] = [5.,0.,0.]
    context = {'history':[{'removed_indices_native': [[0,0,0],[1,0,0],[0,0,0]]}],
               'planning_affine':planning_affine}
    mask = np.zeros((8,8,8), bool)
    mask[2,2,4] = True
    coverage = np.ones_like(mask)
    coverage[2,3,4] = False
    ref = SimpleNamespace(mask=mask, coverage=coverage, affine_ras_mm=np.eye(4))
    result = v._removed_overlap(context, ref, transform)
    assert result['status'] == 'computed'
    assert result['removed_planning_cells'] == 2
    assert result['positive_overlap_cells'] == 1
    assert result['positive_overlap_mm3'] == 1.
    assert result['unknown_removed_cells'] == 1
    assert result['outside_reference_fov_removed_cells'] == 0


def test_removed_overlap_requires_congruence_even_when_centres_round_to_hit():
    context = {'history':[{'removed_indices_native':[[0,0,0]]}], 'planning_affine':np.eye(4)}
    context['planning_affine'][0,3] = .25
    ref = SimpleNamespace(mask=np.ones((2,2,2), bool), coverage=np.ones((2,2,2), bool),
                          affine_ras_mm=np.eye(4))
    result = v._removed_overlap(context, ref, np.eye(4))
    assert result['status'] == 'unsupported_noncongruent_grids'
    assert result['positive_overlap_mm3'] is None
    assert result['unknown_removed_cells'] == 1


def test_congruent_anisotropic_cell_overlap_uses_physical_cubic_millimetres():
    affine = np.diag([2.,3.,4.,1.])
    context = {'history':[{'removed_indices_native':[[0,0,0],[1,1,1]]}], 'planning_affine':affine}
    mask = np.zeros((2,2,2), bool)
    mask[1,1,1] = True
    ref = SimpleNamespace(mask=mask, coverage=np.ones_like(mask), affine_ras_mm=affine)
    result = v._removed_overlap(context, ref, np.eye(4))
    assert result['positive_overlap_cells'] == 1
    assert result['positive_overlap_mm3'] == pytest.approx(24.)


def test_build_parent_traversal_is_rejected_before_io():
    with pytest.raises(ValueError):
        v._safe_path(ROOT/'build'/'..'/'review-should-not-write.json')


def test_nonregular_fifo_refused_without_blocking_open(tmp_path, monkeypatch):
    path = tmp_path/'not-a-seal.fifo'
    os.mkfifo(path)
    actual_open = os.open
    def guarded_open(path_arg, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK, 'Regular-file validation must not block opening a FIFO'
        return actual_open(path_arg, flags, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(v.os, 'open', guarded_open)
        with pytest.raises(ValueError, match='seal_missing_or_oversized'):
            v._read_bound(path, 'a'*64)


def test_symlink_parent_refused(tmp_path):
    actual = tmp_path/'actual'
    actual.mkdir()
    link = tmp_path/'link'
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink_path_forbidden'):
        v._safe_path(link/'seal.json')


def test_seal_mutation_during_private_loader_saves_failure(setup, tmp_path):
    ref = author.reference(setup)
    calls = []
    original = setup[3].read_bytes()
    def mutate():
        calls.append(True)
        setup[3].write_bytes(original + b' ')
        return ref
    result = author.evaluate(setup, ref, tmp_path/'seal-mutated', mutate)
    assert calls == [True]
    assert result['status'] == 'evaluation_failed' and result['outcomes'] is None
    assert json.loads((tmp_path/'seal-mutated/report.json').read_bytes()) == result


def test_reference_mask_snapshot_detaches_caller_arrays(setup):
    ref = author.reference(setup)
    caller_mask, caller_coverage = ref.mask.copy(), ref.coverage.copy()
    copied = v.VascularReference(ref.binding, caller_mask, caller_coverage, ref.affine_ras_mm.copy())
    caller_mask[:] = False
    caller_coverage[:] = False
    copied.assert_intact()
    assert copied.mask.sum() == 2 and copied.coverage.all()
    with pytest.raises(ValueError):
        copied.mask.setflags(write=True)


def test_expected_transform_manifest_change_rejected_after_private_load(setup, tmp_path):
    ref = author.reference(setup)
    transform = np.array(ref.binding.planning_to_reference_ras_mm)
    transform[0,3] += 1
    foreign = v.VascularReference(replace(ref.binding, planning_to_reference_ras_mm=tuple(map(tuple,transform))),
                                  ref.mask, ref.coverage, ref.affine_ras_mm)
    result = author.evaluate(setup, ref, tmp_path/'foreign-transform', lambda: foreign)
    assert result['status'] == 'evaluation_failed' and result['outcomes'] is None
    assert result['private_loader_calls'] == 1


def test_generated_stop_has_no_tool_or_removal_exposure(tmp_path):
    spec = author.helpers.fixture_spec()
    public = author.identity(spec)
    plan = author.helpers.plan(spec)
    seal = tmp_path/'stop-seal.json'
    sha = v.write_strategy_seal(seal, plan=plan, spec=spec, planning_identity=public)
    facts = (spec, plan, public, seal, sha)
    ref = author.reference(facts)
    result = author.evaluate(facts, ref, tmp_path/'stop')
    assert result['status'] == 'evaluated_generated_vascular_reference'
    assert result['action_count'] == 1 and result['nonstop_sweeps'] == result['microstep_count'] == 0
    assert result['whole_tool']['touched_reference_cells'] == 0
    assert result['removed_overlap']['removed_planning_cells'] == 0
    assert result['removed_overlap']['positive_overlap_mm3'] == 0.


def test_no_prohibited_runtime_actions_or_payload_opens():
    assert PROHIBITED_ACTIONS == []
    assert author.PAYLOAD_OPENS == []
    assert 'torch' not in sys.modules
