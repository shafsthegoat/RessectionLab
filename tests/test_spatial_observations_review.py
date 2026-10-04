"""Independent tiny-array physical-coordinate and permitted-observation checks."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from resectionlab.spatial_policy import sample_ray_features, world_to_sample_grid
from resectionlab.spatial_policy import SpatialPolicy
from resectionlab.spatial_observations import (CHANNEL_NAMES, ObservedChannel, ObservedProcedureState,
    SpatialAction, SpatialInputs, build_spatial_observation)
from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS


def encoded_phantom(shape=(4, 5, 6)):
    x, y, z = np.indices(shape, dtype=np.float64)
    return np.stack((x + 10 * y + 100 * z, 2 * x - y + .5 * z))


def physical_affines():
    angle, tilt = .43, -.27
    rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    result = []
    for orientation in (np.eye(3), rz @ ry, rz @ ry @ np.diag([-1., 1., 1.])):
        affine = np.eye(4)
        affine[:3, :3] = orientation @ np.diag([1.5, 2.75, 4.])
        affine[:3, 3] = [17., -31., 8.]
        result.append(affine)
    return result


@pytest.mark.parametrize("affine", physical_affines(), ids=["anisotropic", "oblique", "reflected"])
def test_noncubic_affine_sampling_preserves_xyz_to_torch_zyx_order(affine):
    shape = (4, 5, 6)
    # Distinct coefficients and unequal dimensions detect an otherwise plausible
    # X/Z transpose. The expected answer is analytic, not another grid sampler.
    voxels = np.array([[1., 2., 3.], [1.5, 2.25, 3.5], [.25, 1.5, 4.25]])
    world = voxels @ affine[:3, :3].T + affine[:3, 3]
    grid = world_to_sample_grid(world[None], affine, shape)
    expected_grid = (2 * voxels / (np.asarray(shape) - 1) - 1)[:, ::-1]
    np.testing.assert_allclose(grid[0], expected_grid, rtol=0, atol=2e-14)
    sampled = sample_ray_features(torch.tensor(encoded_phantom()), torch.tensor(grid))
    expected = np.stack((voxels[:, 0] + 10 * voxels[:, 1] + 100 * voxels[:, 2],
                         2 * voxels[:, 0] - voxels[:, 1] + .5 * voxels[:, 2]), axis=-1)
    np.testing.assert_allclose(sampled.numpy()[0], expected, rtol=0, atol=1e-11)


def test_five_ray_points_keep_entry_to_tip_order_and_have_nonzero_image_gradient():
    entry, tip = np.array([.25, .5, .75]), np.array([2.75, 3.5, 4.25])
    voxels = entry + np.linspace(0, 1, 5)[:, None] * (tip - entry)
    grid = torch.tensor(world_to_sample_grid(voxels[None], np.eye(4), (4, 5, 6)))
    encoded = torch.tensor(encoded_phantom(), requires_grad=True)
    sampled = sample_ray_features(encoded, grid)
    assert sampled.shape == (1, 5, 2)
    np.testing.assert_allclose(sampled.detach().numpy()[0, :, 0], voxels @ [1., 10., 100.], atol=1e-12)
    sampled.sum().backward()
    assert torch.isfinite(encoded.grad).all() and float(encoded.grad.abs().sum()) > 0.


@pytest.mark.parametrize("affine", physical_affines(), ids=["anisotropic", "oblique", "reflected"])
def test_exact_physical_grid_corners_are_not_lost_to_inverse_roundoff(affine):
    corners = np.asarray(np.meshgrid([0., 3.], [0., 4.], [0., 5.], indexing="ij")).reshape(3, -1).T
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    grid = world_to_sample_grid(world[None], affine, (4, 5, 6))
    sampled = sample_ray_features(torch.ones((1, 4, 5, 6), dtype=torch.float64), torch.tensor(grid))
    np.testing.assert_array_equal(sampled.numpy(), np.ones((1, 8, 1)))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_outside_center_domain_has_zero_features_not_partial_border_tissue(axis, side):
    shape = np.array([4, 5, 6])
    points = np.array([[1., 2., 3.], [1., 2., 3.]])
    boundary = 0. if side == "lower" else shape[axis] - 1.
    points[0, axis] = boundary
    points[1, axis] = boundary + (-.25 if side == "lower" else .25)
    grid = world_to_sample_grid(points[None], np.eye(4), shape)
    sampled = sample_ray_features(torch.tensor(encoded_phantom()), torch.tensor(grid)).numpy()[0]
    np.testing.assert_allclose(sampled[0, 0], points[0] @ [1., 10., 100.], atol=1e-12)
    np.testing.assert_array_equal(sampled[1], [0., 0.])
    assert (np.abs(grid[0, 0]) <= 1).all() and not (np.abs(grid[0, 1]) <= 1).all()


@pytest.mark.parametrize("points", [np.array([[np.nan, 0., 0.]]), np.array([[0., np.inf, 0.]]),
    np.array([[1 + 1j, 0., 0.]]), np.array([[1., 2.]]), np.array([[True, False, True]])])
def test_invalid_physical_points_fail_before_interpolation(points):
    with pytest.raises(ValueError):
        world_to_sample_grid(points, np.eye(4), (4, 5, 6))


@pytest.mark.parametrize("shape", [(1, 5, 6), (4, 0, 6), (4., 5., 6.), (4, 5)])
def test_singleton_or_ambiguous_shape_is_refused(shape):
    with pytest.raises(ValueError):
        world_to_sample_grid(np.zeros((1, 3)), np.eye(4), shape)


@pytest.mark.parametrize("bad", ["singular", "nonfinite", "nonhomogeneous"])
def test_invalid_source_frame_is_refused(bad):
    affine = np.eye(4)
    if bad == "singular":
        affine[0, 0] = 0.
    elif bad == "nonfinite":
        affine[0, 3] = np.inf
    else:
        affine[3, 0] = .1
    with pytest.raises(ValueError):
        world_to_sample_grid(np.zeros((1, 3)), affine, (4, 5, 6))


def test_nonfinite_tensor_grid_cannot_be_interpreted_as_padding():
    grid = torch.zeros((1, 5, 3), dtype=torch.float64)
    grid[0, 2, 1] = float("nan")
    with pytest.raises(ValueError):
        sample_ray_features(torch.tensor(encoded_phantom()), grid)


def observation_fixture(*, extra_channels=None, track="synthetic_scan", affine=None, source_id="independent-analytic"):
    affine = np.eye(4) if affine is None else affine
    image = encoded_phantom()[0].astype(np.float32)
    source_kind = "observed_scan" if track == "inference_only" else "synthetic_scan"
    channels = {"structural_intensity": ObservedChannel(image, source_kind=source_kind),
        "observed_cavity": ObservedChannel(np.zeros(image.shape, bool), source_kind="observed_procedure_state")}
    channels.update(extra_channels or {})
    inputs = SpatialInputs(channels, affine, track, source_id)
    to_world = lambda point: tuple(affine[:3, :3] @ point + affine[:3, 3])
    access = AccessWindow(to_world([1., 2., .5]), affine[:3, 2], 6.)
    actions = (SpatialAction("STOP"), SpatialAction("ray", to_world([1., 2., .5]),
        to_world([1., 2., 4.5]), NATIVE_GENERIC_TOOLS[0]))
    return inputs, build_spatial_observation(inputs, actions, ObservedProcedureState(access, 0, 5))


def test_missing_target_is_distinct_from_observed_zero_in_actual_actor_tensors():
    _, absent = observation_fixture()
    _, known_zero = observation_fixture(extra_channels={"nominal_target": ObservedChannel(
        np.zeros((4, 5, 6)), source_kind="derived_from_scan", derivation="analytic zero-valued estimator",
        derived_from=("structural_intensity",))})
    target = CHANNEL_NAMES.index("nominal_target")
    assert not absent.channel_available[target] and not absent.coverage[target].any()
    assert known_zero.channel_available[target] and known_zero.coverage[target].all()
    np.testing.assert_array_equal(absent.image_channels[target], known_zero.image_channels[target])
    policy = SpatialPolicy()
    first, second = policy._inputs(absent)[0], policy._inputs(known_zero)[0]
    assert torch.equal(first[target], second[target])
    assert not torch.equal(first[target + 6], second[target + 6])
    assert not torch.equal(first[target + 12], second[target + 12])


def test_coverage_hides_values_and_is_separate_from_modality_availability():
    original = encoded_phantom()[0].astype(np.float32)
    coverage = np.ones(original.shape, bool)
    coverage[0, :, :] = False
    channel = ObservedChannel(original, coverage, source_kind="synthetic_scan")
    original[:] = 999.
    assert not channel.data[0].any() and channel.coverage[1].all()
    assert float(channel.data[1, 2, 3]) == 321.
    with pytest.raises(ValueError):
        channel.data.setflags(write=True)
    with pytest.raises(ValueError):
        channel.data.shape = (120,)


@pytest.mark.parametrize("name", ["nominal_target", "nominal_motor", "nominal_language"])
@pytest.mark.parametrize("track", ["synthetic_scan", "inference_only"])
def test_primary_tracks_refuse_reference_annotation_channels(name, track):
    with pytest.raises(ValueError, match="(?i)(annotation|reference|scan)"):
        observation_fixture(track=track, extra_channels={name: ObservedChannel(
            np.ones((4, 5, 6)), source_kind="supplied_annotation")})


def test_direct_dto_replace_cannot_move_reference_annotations_into_primary_track():
    _, assisted = observation_fixture(track="annotation_assisted", extra_channels={
        "nominal_target": ObservedChannel(np.ones((4, 5, 6)), source_kind="supplied_annotation")})
    with pytest.raises(ValueError):
        replace(assisted, track="synthetic_scan")


@pytest.mark.parametrize("change", ["unavailable_payload", "unavailable_coverage", "false_spacing", "source_provenance"])
def test_direct_dto_cannot_forge_missingness_frame_or_source_contract(change):
    _, observation = observation_fixture()
    target = CHANNEL_NAMES.index("nominal_target")
    if change == "unavailable_payload":
        values = observation.image_channels.copy()
        values[target] = 1.
        changes = {"image_channels": values}
    elif change == "unavailable_coverage":
        values = observation.coverage.copy()
        values[target] = True
        changes = {"coverage": values}
    elif change == "false_spacing":
        changes = {"spacing_mm": np.array([9., 9., 9.])}
    else:
        provenance = {name: dict(row) for name, row in observation.channel_provenance.items()}
        provenance["structural_intensity"]["source_kind"] = "supplied_annotation"
        changes = {"channel_provenance": provenance}
    with pytest.raises(ValueError):
        replace(observation, **changes)


def test_adapter_preserves_case_values_without_unannounced_population_normalization():
    inputs, observation = observation_fixture(affine=physical_affines()[1])
    np.testing.assert_array_equal(observation.image_channels[0], inputs.channels["structural_intensity"].data)
    np.testing.assert_allclose(observation.spacing_mm, [1.5, 2.75, 4.], rtol=0, atol=1e-14)
    assert not observation.channel_available[[1, 2, 4, 5]].any()


def test_policy_rejects_tensor_replacement_after_dto_validation():
    _, observation = observation_fixture()
    object.__setattr__(observation, "image_channels", np.asarray(observation.image_channels).copy())
    with pytest.raises(ValueError):
        SpatialPolicy()._inputs(observation)


@pytest.mark.parametrize("affine", physical_affines(), ids=["anisotropic", "oblique", "reflected"])
def test_exact_physical_boundary_ray_survives_dto_to_policy_conversion(affine):
    inputs, _ = observation_fixture(affine=affine)
    world = lambda point: tuple(affine[:3, :3] @ point + affine[:3, 3])
    entry, tip = world([3., 2., .5]), world([3., 2., 4.5])
    actions = (SpatialAction("STOP"), SpatialAction("boundary", entry, tip, NATIVE_GENERIC_TOOLS[0]))
    observation = build_spatial_observation(inputs, actions,
        ObservedProcedureState(AccessWindow(entry, affine[:3, 2], 6.), 0, 5))
    _, grid, inside, *_ = SpatialPolicy()._inputs(observation)
    np.testing.assert_array_equal(inside.numpy()[1], np.ones(5))
    np.testing.assert_array_equal(grid.numpy()[1, :, 2], np.ones(5))


def test_precast_outside_flag_withholds_ray_features_after_float32_rounding():
    inputs, _ = observation_fixture()
    # This is well outside the declared 64-float64-epsilon correction band,
    # but would round to the boundary when normalized coordinates become float32.
    entry, tip = (3. + 1e-8, 2., .5), (3. + 1e-8, 2., 4.5)
    observation = build_spatial_observation(inputs,
        (SpatialAction("STOP"), SpatialAction("outside", entry, tip, NATIVE_GENERIC_TOOLS[0])),
        ObservedProcedureState(AccessWindow((3., 2., .5), (0., 0., 1.), 6.), 0, 5))
    policy = SpatialPolicy()
    with torch.no_grad():
        for layer in (policy.encoder[0], policy.encoder[2]):
            layer.weight.zero_()
            layer.bias.zero_()
        policy.encoder[2].bias.fill_(1.)
    captured = []
    handle = policy.actor.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        policy(observation)
    finally:
        handle.remove()
    width = policy.config.encoder_channels[-1] + 1
    rays = captured[0][1, -policy.config.ray_samples * width:].reshape(policy.config.ray_samples, width)
    np.testing.assert_array_equal(rays.numpy(), np.zeros((policy.config.ray_samples, width)))
