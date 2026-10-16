"""Physical coordinates and admission controls for the single real-host pilot."""
import importlib.util
import itertools
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("ixi_virtual_pilot", Path(__file__).resolve().parents[1]/"scripts/poc_ixi265_virtual_glioma.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_oblique_reflected_grid_covers_all_source_brain_cell_corners():
    angle = .43
    transform = np.eye(4)
    transform[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]) @ np.diag([-1.7, 2.1, 2.8])
    transform[:3, 3] = [31., -28., 9.]
    brain = np.zeros((20, 19, 17), bool)
    brain[3:17, 4:16, 2:15] = True
    shape, grid = pilot.physical_grid(brain, transform)
    for index in itertools.product((2.5, 16.5), (3.5, 15.5), (1.5, 14.5)):
        world = transform @ np.r_[index, 1.]
        sampled = (np.linalg.inv(grid) @ world)[:3]
        assert np.all(sampled >= 1.) and np.all(sampled <= np.array(shape)-2.)
    assert np.array_equal(grid[:3, :3], np.eye(3)*3.)


def test_native_threshold_transfer_has_explicit_world_coordinates_and_intersection():
    coarse = np.zeros((6, 5, 4), bool)
    coarse[2:4, 1:4, 1:3] = True
    coarse_affine = np.diag([3., 3., 3., 1.]); coarse_affine[:3, 3] = [12., -9., 6.]
    source_affine = np.diag([-1.5, 1.5, 1.5, 1.]); source_affine[:3, 3] = [24., -9., 6.]
    brain = np.ones((8, 9, 7), bool); brain[:, 3, :] = False
    expected = np.zeros(brain.shape, bool); total_before = 0
    for index in np.ndindex(brain.shape):
        # Independent coordinate construction, no production interpolation helper.
        world = source_affine[:3, :3] @ index + source_affine[:3, 3]
        q = (world-coarse_affine[:3, 3])/3.
        inside = np.all(q >= 0) and np.all(q <= np.array(coarse.shape)-1)
        value = inside and coarse[tuple(np.floor(q+.5).astype(int))]
        total_before += int(value)
        expected[index] = value and brain[index]
    actual, receipt = pilot.native_target(coarse, coarse_affine, brain, source_affine)
    assert np.array_equal(actual, expected)
    assert receipt["transferred_cells_before_intersection"] == total_before
    assert receipt["cells_outside_source_brain_excluded_by_display_definition"] == total_before-int(expected.sum())
    assert not receipt["pde_arrays_modified"]


def test_parameter_seed_is_inside_envelope_without_claiming_anatomical_location():
    domain = np.zeros((21, 22, 23), bool); domain[2:-2, 2:-2, 2:-2] = True
    affine = np.diag([3., 3., 3., 1.]); affine[:3, 3] = [-30., -33., -36.]
    first, receipt = pilot.choose_seed(domain, affine)
    second, _ = pilot.choose_seed(domain, affine)
    index = np.rint((np.array(first)-affine[:3, 3])/3.).astype(int)
    assert domain[tuple(index)] and first == second
    assert receipt["lobe"] == "UNKNOWN_UNREVIEWED" and not receipt["clinical_placement_review"]


def test_wrong_host_refuses_before_any_bundle_decoding(tmp_path):
    invalid = tmp_path/"not-admitted.ressectionlab"; invalid.write_bytes(b"not an admitted patient")
    with patch.object(pilot, "load_workspace", side_effect=AssertionError("Must not decode")):
        with pytest.raises(ValueError, match="frozen source-qualified"):
            pilot.run(invalid, tmp_path/"never-created")
    assert not (tmp_path/"never-created").exists()
