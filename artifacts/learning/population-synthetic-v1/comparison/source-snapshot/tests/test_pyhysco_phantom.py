"""Physical-coordinate and failure checks; no GPL dependency required in CI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experimental_pyhysco_phantom.py"
SPEC = importlib.util.spec_from_file_location("experimental_pyhysco_phantom", SCRIPT)
phantom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phantom)


def test_oblique_geometry_has_physical_spacing_and_fixed_center():
    shape = (24, 28, 20)
    first = phantom.make_affine(shape)
    rotated = phantom.make_affine(shape, .8)
    assert np.allclose(np.linalg.norm(first[:3, :3], axis=0), 2.5)
    assert np.linalg.det(first[:3, :3]) < 0
    center = np.r_[(np.asarray(shape) - 1) / 2, 1]
    assert np.allclose(first @ center, rotated @ center)
    assert np.allclose((first @ center)[:3], [12, -18, 32])


def test_both_native_affine_and_regridded_pe_mismatch_are_rejected():
    shape = (24, 28, 20)
    first, second = phantom.make_affine(shape), phantom.make_affine(shape, .8)
    with pytest.raises(phantom.GeometryError, match="second-image affine"):
        phantom.require_supported_geometry(shape, first, [0, 1, 0], shape, second, [0, -1, 0])
    transported = phantom.transport_direction([0, -1, 0], second, first)
    assert transported[0] == pytest.approx(np.sin(np.deg2rad(.8)))
    assert transported[1] == pytest.approx(-np.cos(np.deg2rad(.8)))
    with pytest.raises(phantom.GeometryError, match="exactly opposite"):
        phantom.require_supported_geometry(shape, first, [0, 1, 0], shape, first, transported)
    phantom.require_supported_geometry(shape, first, [0, 1, 0], shape, first, [0, -1, 0])


def test_vector_transport_preserves_world_direction_and_round_trip():
    a, b = phantom.make_affine(), phantom.make_affine(relative_degrees=.8)
    original = np.array([0., 1., 0.])
    transported = phantom.transport_direction(original, b, a)
    assert np.allclose(phantom.unit_basis(a) @ transported, phantom.unit_basis(b) @ original)
    assert np.allclose(phantom.transport_direction(transported, a, b), original)
    # Coordinate resampling alone does not create a measured motion rotation.
    assert not np.allclose(transported, original)


def test_shear_is_rejected_and_direction_must_be_unit():
    affine = phantom.make_affine()
    affine[:3, 0] += .1 * affine[:3, 1]
    with pytest.raises(phantom.GeometryError, match="Sheared"):
        phantom.require_supported_geometry((4, 4, 4), affine, [0, 1, 0], (4, 4, 4), affine, [0, -1, 0])
    with pytest.raises(phantom.GeometryError, match="unit"):
        phantom.transport_direction([0, 2, 0], affine, affine)


def test_regridding_respects_full_world_affines():
    shape = (32, 34, 28)
    a, b = phantom.make_affine(shape), phantom.make_affine(shape, .8)
    world_b = phantom.grid_world(shape, b)
    # Physical linear ramp is an independent exact spatial landmark test.
    ramp_b = world_b @ np.array([.2, -.4, .7])
    resampled = phantom.regrid(ramp_b, b, shape, a)
    expected = phantom.grid_world(shape, a) @ np.array([.2, -.4, .7])
    interior = (slice(6, -6),) * 3
    assert np.max(np.abs(resampled[interior] - expected[interior])) < .001
    assert np.max(np.abs(ramp_b[interior] - expected[interior])) > .1


def test_analytic_field_gradient_matches_independent_world_difference():
    basis = phantom.unit_basis(phantom.make_affine())
    positions = np.array([[12., -18., 32.], [21., -45., 50.], [-36., 10., 13.]])
    _, analytic = phantom.analytic_field(positions, basis, 4.)
    numeric = []
    for axis in np.eye(3):
        plus, _ = phantom.analytic_field(positions + axis * .001, basis, 4.)
        minus, _ = phantom.analytic_field(positions - axis * .001, basis, 4.)
        numeric.append((plus - minus) / .002)
    assert np.allclose(analytic, np.stack(numeric, axis=-1), atol=1e-9)


def test_forward_inverse_map_and_mass_modulation_are_nontrivial():
    shape = (20, 22, 18)
    affine = phantom.make_affine(shape)
    basis = phantom.unit_basis(affine)
    zero, zero_qc = phantom.distorted_image(shape, affine, basis, basis[:, 1], 0.)
    plus, plus_qc = phantom.distorted_image(shape, affine, basis, basis[:, 1], 4.)
    minus, minus_qc = phantom.distorted_image(shape, affine, basis, -basis[:, 1], 4.)
    truth = phantom.analytic_density(phantom.grid_world(shape, affine), basis)
    assert np.allclose(zero, truth, atol=1e-7)
    assert zero_qc["min_forward_jacobian"] == 1
    assert np.linalg.norm(plus - minus) > 1
    assert plus_qc["min_forward_jacobian"] > .8
    assert minus_qc["min_forward_jacobian"] > .8
    assert max(plus_qc["max_inverse_residual_mm"], minus_qc["max_inverse_residual_mm"]) < 1e-12


def fake_outputs(tmp_path, *, shape=(8, 10, 6)):
    affine = phantom.make_affine(shape)
    reference = tmp_path / "reference.nii.gz"
    phantom.save_image(reference, np.ones(shape), affine)
    prefix = tmp_path / "raw"
    for number in (1, 2):
        phantom.save_image(str(prefix) + f"-im{number}Corrected.nii.gz", np.ones(shape) * 256, np.eye(4))
    nodes = np.broadcast_to(np.arange(shape[1] + 1)[None, :, None] * .25,
                            (shape[0], shape[1] + 1, shape[2]))
    phantom.save_image(str(prefix) + "-EstFieldMap.nii.gz", nodes, np.eye(4))
    return reference, prefix, nodes


def test_output_adapter_preserves_image_and_half_voxel_node_coordinates(tmp_path):
    reference, prefix, nodes = fake_outputs(tmp_path)
    images, centers, derivative, meta = phantom.adapt_outputs(prefix, reference, tmp_path / "adapted", np.float32(256))
    json.dumps(meta)  # Regression: NumPy scaling must not break persisted audit records.
    assert all(np.array_equal(image, np.ones((8, 10, 6))) for image in images)
    assert np.allclose(derivative, .1)
    assert np.array_equal(centers, (nodes[:, :-1] + nodes[:, 1:]) / 2)
    corrected = nib.load(tmp_path / "adapted/corrected-1.nii.gz")
    ref = nib.load(reference)
    field_nodes = nib.load(tmp_path / "adapted/displacement-nodes-mm.nii.gz")
    assert corrected.header.get_xyzt_units()[0] == "mm"
    assert np.array_equal(corrected.affine, ref.affine)
    assert meta["raw_corrected_affines_are_identity"] == [True, True]
    # First two node positions bracket image center; averaging their physical
    # locations must land on the reference voxel center, including obliquity.
    first = field_nodes.affine @ [3, 4, 2, 1]
    second = field_nodes.affine @ [3, 5, 2, 1]
    center = ref.affine @ [3, 4, 2, 1]
    assert np.allclose((first + second) / 2, center, atol=1e-5)


def test_nonzero_intensity_floor_requires_jacobian_modulation(tmp_path):
    reference, prefix, _ = fake_outputs(tmp_path)
    images, _, _, meta = phantom.adapt_outputs(prefix, reference, tmp_path / "adapted", 256, -.2)
    assert np.allclose(images[0], 1 - .2 * 1.1)
    assert np.allclose(images[1], 1 - .2 * .9)
    assert meta["input_intensity_floor_restored_with_jacobian"] == -.2


@pytest.mark.parametrize("kind", ["image_shape", "node_shape", "nan"])
def test_output_adapter_rejects_corrupt_or_unexpected_outputs(tmp_path, kind):
    reference, prefix, _ = fake_outputs(tmp_path)
    if kind == "image_shape":
        phantom.save_image(str(prefix) + "-im1Corrected.nii.gz", np.ones((4, 4, 4)), np.eye(4))
    elif kind == "node_shape":
        phantom.save_image(str(prefix) + "-EstFieldMap.nii.gz", np.ones((8, 10, 6)), np.eye(4))
    else:
        phantom.save_image(str(prefix) + "-im2Corrected.nii.gz", np.full((8, 10, 6), np.nan), np.eye(4))
    with pytest.raises(ValueError):
        phantom.adapt_outputs(prefix, reference, tmp_path / "adapted", 256)


def test_wrong_wheel_rejected_without_importing_optional_tool(tmp_path):
    wheel = tmp_path / "wrong.whl"
    wheel.write_bytes(b"not the reviewed release")
    with pytest.raises(ValueError, match="checksum"):
        phantom.verify_runtime(tmp_path, wheel)
    assert "EPI_MRI" not in sys.modules


def test_cpu_timeout_is_enforced(tmp_path):
    result = phantom.bounded_run([sys.executable, "-c", "import time; time.sleep(20)"],
                                  tmp_path / "timeout.log", {}, .1, 8 * 1024**3)
    assert result["stopped"] == "timeout"
    assert result["returncode"] != 0
    assert result["wall_seconds"] < 3
