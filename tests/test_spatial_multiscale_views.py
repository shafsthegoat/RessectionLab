"""Analytic image preparation checks only; no patient data, training or geometry runs."""
from dataclasses import FrozenInstanceError, replace
from itertools import product

import numpy as np
import pytest

from resectionlab import spatial_multiscale_views as mv


def source(shape=(7, 9, 5), *, target=None, affine=None, coverage=None, source_id="permitted_scan"):
    x, y, z = np.indices(shape)
    intensity = x + 3 * y - 2 * z
    nominal = np.zeros(shape)
    nominal[shape[0] // 2, shape[1] // 2, shape[2] // 2] = .125
    nominal = nominal if target is None else target
    cavity = ((x + 2 * y + z) % 7 == 0)
    channels = {
        "structural_intensity": mv.PermittedSourceChannel(intensity, coverage, "observed_scan"),
        "nominal_tissue": mv.PermittedSourceChannel(np.ones(shape), coverage, "supplied_annotation"),
        "nominal_target": mv.PermittedSourceChannel(nominal, coverage, "supplied_annotation"),
        "observed_cavity": mv.PermittedSourceChannel(cavity, coverage, "observed_procedure_state"),
    }
    return mv.PermittedVolumeSource(channels, np.eye(4) if affine is None else affine,
                                    "annotation_assisted", source_id)


def oblique_affine():
    a = .37
    affine = np.eye(4)
    affine[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, -1.]]) @ np.diag([1.7, 2.3, .9])
    affine[:3, 3] = [17., -42., 3.]
    return affine


def brute_average(data, output_shape):
    result = np.zeros(output_shape)
    size = np.array(data.shape) / output_shape
    for q in np.ndindex(output_shape):
        low, high = np.array(q) * size, (np.array(q) + 1) * size
        for p in np.ndindex(data.shape):
            overlap = np.maximum(0, np.minimum(high, np.array(p) + 1) - np.maximum(low, p))
            result[q] += data[p] * np.prod(overlap) / np.prod(size)
    return result


def test_coarse_exact_overlap_matches_independent_geometric_integral():
    coverage = np.ones((7, 9, 5), bool)
    coverage[1:3, 2:5, 0:3] = False
    original = source(coverage=coverage, affine=oblique_affine())
    views = mv.prepare_multiscale_views(original, local_shape=(4, 5, 3), coarse_shape=(3, 4, 2))
    for i, name in enumerate(mv.CHANNEL_NAMES[:4]):
        channel = original.channels[name]
        np.testing.assert_allclose(views.whole_source.image_channels[i], brute_average(channel.data, (3, 4, 2)), atol=3e-14, rtol=1e-14)
        np.testing.assert_allclose(views.whole_source.coverage_fraction[i], brute_average(channel.coverage, (3, 4, 2)), atol=3e-15)
        mass = views.report["channel_integrals"][name]
        assert mass["source_value_integral_mm3"] == pytest.approx(mass["coarse_value_integral_mm3"], rel=2e-14, abs=2e-13)
        assert mass["source_covered_volume_mm3"] == pytest.approx(mass["coarse_covered_volume_mm3"], rel=2e-14)
    assert views.report["coarse_nominal_target_mass_relative_error"] < 1e-12
    assert np.any((views.whole_source.image_channels[3] > 0) & (views.whole_source.image_channels[3] < 1))
    assert views.whole_source.preparation["native_geometry_eligible"] is False


@pytest.mark.parametrize("affine", [np.eye(4), oblique_affine(),
    np.array([[1., .2, 0, 3], [0, 2, .1, 4], [0, 0, 3, 5], [0, 0, 0, 1]])])
def test_original_cell_extent_and_roundtrip_match_in_each_physical_frame(affine):
    original = source(affine=affine)
    views = mv.prepare_multiscale_views(original, local_shape=(4, 5, 3), coarse_shape=(3, 4, 2))
    full_corners = np.array(list(product(*[(-.5, n - .5) for n in original.shape])))
    coarse_corners = np.array(list(product(*[(-.5, n - .5) for n in views.whole_source.shape])))
    physical = full_corners @ affine[:3, :3].T + affine[:3, 3]
    physical_coarse = coarse_corners @ views.whole_source.affine_ras_mm[:3, :3].T + views.whole_source.affine_ras_mm[:3, 3]
    np.testing.assert_allclose(physical, physical_coarse, atol=2e-14)
    inverse = np.linalg.inv(views.whole_source.affine_ras_mm)
    back = physical_coarse @ inverse[:3, :3].T + inverse[:3, 3]
    np.testing.assert_allclose(back, coarse_corners, atol=2e-14)
    assert views.report["maximum_extent_corner_error_mm"] < 1e-8
    start = np.array(views.report["local_start_ijk"])
    np.testing.assert_array_equal(views.target_local.affine_ras_mm[:3, :3], affine[:3, :3])
    np.testing.assert_allclose(views.target_local.affine_ras_mm[:3, 3], affine[:3, :3] @ start + affine[:3, 3])


def test_center_is_bbox_midpoint_not_mass_or_reference_center_and_lower_tie_is_fixed():
    target = np.zeros((11, 9, 7))
    target[1, 2, 1] = .001
    target[9, 6, 5] = .9
    original = source(target.shape, target=target)
    views = mv.prepare_multiscale_views(original, local_shape=(4, 4, 4), coarse_shape=(3, 3, 3))
    assert views.report["local_start_ijk"] == (3, 2, 1)
    assert views.report["target_bbox_min_ijk"] == (1, 2, 1)
    assert views.report["target_bbox_max_ijk"] == (9, 6, 5)
    assert views.report["target_local_clipped_fraction"] == 1.0
    assert views.report["nominal_target_total_mass_mm3"] == pytest.approx(.901)
    assert views.whole_source.image_channels[2].sum() > 0
    altered = target.copy()
    altered[1, 2, 1], altered[9, 6, 5] = .9, .001
    other = mv.prepare_multiscale_views(source(target.shape, target=altered), local_shape=(4, 4, 4), coarse_shape=(3, 3, 3))
    assert other.report["local_start_ijk"] == views.report["local_start_ijk"]


def test_boundary_target_clamps_native_crop_without_resizing_and_missing_function_stays_unknown():
    target = np.zeros((8, 9, 10))
    target[0, 8, 9] = .001
    views = mv.prepare_multiscale_views(source(target.shape, target=target), local_shape=(4, 4, 4), coarse_shape=(3, 3, 3))
    assert views.report["local_start_ijk"] == (0, 5, 6)
    assert views.report["target_local_clipped_fraction"] == 0
    for view in (views.target_local, views.whole_source):
        for i in (4, 5):
            assert not view.channel_available[i]
            assert not view.coverage_fraction[i].any() and not view.image_channels[i].any()
            assert view.channel_provenance[mv.CHANNEL_NAMES[i]]["source_kind"] == "unavailable"
    assert views.report["channel_integrals"]["nominal_motor"]["source_value_integral_mm3"] is None
    assert views.report["clinical_deficit_probability"] is None


def test_uncovered_data_are_zero_before_hash_or_centering_and_fractional_coverage_is_not_known_zero():
    shape = (7, 9, 5)
    coverage = np.ones(shape, bool)
    coverage[0:2] = False
    original = source(coverage=coverage)
    channels = dict(original.channels)
    changed = np.array(channels["nominal_target"].data)
    changed[~coverage] = 1
    channels["nominal_target"] = mv.PermittedSourceChannel(changed, coverage, "supplied_annotation")
    other = mv.PermittedVolumeSource(channels, original.affine_ras_mm, original.track, "different mixed-file audit hash")
    assert other.permitted_hash == original.permitted_hash
    a = mv.prepare_multiscale_views(original, coarse_shape=(3, 4, 2))
    b = mv.prepare_multiscale_views(other, coarse_shape=(3, 4, 2))
    assert a.whole_source.fingerprint == b.whole_source.fingerprint
    assert a.target_local.fingerprint == b.target_local.fingerprint
    assert np.any((a.whole_source.coverage_fraction[2] > 0) & (a.whole_source.coverage_fraction[2] < 1))
    known_zero = np.array(a.whole_source.image_channels[2]) == 0
    assert np.any(known_zero & (a.whole_source.coverage_fraction[2] < 1))


def test_missing_empty_and_uncovered_target_reject_local_view():
    original = source()
    for target in [mv.PermittedSourceChannel(),
        mv.PermittedSourceChannel(np.zeros(original.shape), source_kind="supplied_annotation"),
        mv.PermittedSourceChannel(np.ones(original.shape), np.zeros(original.shape, bool), "supplied_annotation")]:
        channels = dict(original.channels)
        channels["nominal_target"] = target
        with pytest.raises(mv.MultiscaleInputError, match="TARGET_LOCAL_UNAVAILABLE"):
            mv.prepare_multiscale_views(replace(original, channels=channels))


def test_native_small_source_is_not_upsampled_or_padded():
    original = source((3, 4, 2))
    views = mv.prepare_multiscale_views(original)
    assert views.target_local.shape == views.whole_source.shape == (3, 4, 2)
    np.testing.assert_array_equal(views.target_local.affine_ras_mm, original.affine_ras_mm)
    np.testing.assert_array_equal(views.whole_source.image_channels[2], original.channels["nominal_target"].data)


@pytest.mark.parametrize("field", ["reference_target", "reward", "future_cavity", "postoperative_mri"])
def test_hidden_and_future_fields_have_no_input_slot(field):
    original = source()
    channels = dict(original.channels)
    channels[field] = channels["nominal_target"]
    with pytest.raises(mv.MultiscaleInputError, match="six permitted"):
        replace(original, channels=channels)
    with pytest.raises(TypeError):
        mv.prepare_multiscale_views(original, **{field: np.zeros(original.shape)})


def test_inference_track_rejects_annotation_and_permits_explicit_scan_estimate():
    original = source()
    with pytest.raises(mv.MultiscaleInputError, match="annotation track"):
        replace(original, track="inference_only")
    channels = dict(original.channels)
    for name in ("nominal_target", "nominal_tissue"):
        channels[name] = mv.PermittedSourceChannel(channels[name].data, source_kind="derived_from_scan",
            derivation="Fixed permitted scan estimator v1", derived_from=("structural_intensity",))
    inferred = replace(original, channels=channels, track="inference_only")
    assert mv.prepare_multiscale_views(inferred).target_local.track == "inference_only"


def test_source_and_view_arrays_metadata_and_reports_are_immutable():
    original = source()
    views = mv.prepare_multiscale_views(original)
    for array in (original.channels["nominal_target"].data, original.affine_ras_mm,
                  views.whole_source.image_channels, views.target_local.coverage_fraction):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.strides = array.strides[::-1]
    with pytest.raises(FrozenInstanceError):
        original.track = "inference_only"
    with pytest.raises(TypeError):
        views.report["channel_integrals"]["nominal_target"]["source_value_integral_mm3"] = 0
    object.__setattr__(views.whole_source, "preprocessing_version", "forged")
    with pytest.raises(mv.MultiscaleInputError):
        views.assert_intact()


def test_tampered_source_replaced_arrays_and_bad_direct_view_metadata_fail():
    original = source()
    views = mv.prepare_multiscale_views(original)
    with pytest.raises(mv.MultiscaleInputError, match="preprocessing fields"):
        replace(views.whole_source, preparation={**views.whole_source.preparation, "postoperative_score": 9})
    with pytest.raises(mv.MultiscaleInputError, match="preprocessing version"):
        replace(views.whole_source, preprocessing_version="changed")
    object.__setattr__(original.channels["nominal_target"], "data", np.zeros(original.shape))
    with pytest.raises(mv.MultiscaleInputError):
        mv.prepare_multiscale_views(original)


def test_public_bundle_rejects_mixed_source_reports_preparation_and_frames():
    a = mv.prepare_multiscale_views(source())
    b = mv.prepare_multiscale_views(source(affine=oblique_affine()))
    with pytest.raises(mv.MultiscaleInputError, match="identity"):
        replace(a, report=b.report)
    with pytest.raises(mv.MultiscaleInputError, match="same permitted source"):
        replace(a, whole_source=b.whole_source)
    with pytest.raises(mv.MultiscaleInputError, match="exact schema"):
        replace(a, report={**a.report, "future_outcome": 1})
    with pytest.raises(mv.MultiscaleInputError, match="actual view affines"):
        replace(a, report={**a.report, "source_affine_ras_mm": oblique_affine().tolist()})
    altered = replace(a.whole_source, affine_ras_mm=oblique_affine())
    assert altered.preprocessing_hash != a.whole_source.preprocessing_hash


@pytest.mark.parametrize("value", [np.ones((2, 2, 2), complex), np.full((2, 2, 2), np.inf),
    np.zeros((1, 1, 1, 1)), np.broadcast_to(0., (32_000_001, 1, 1))])
def test_malformed_and_oversized_sources_fail_before_copy(value):
    with pytest.raises(mv.MultiscaleInputError):
        mv.PermittedSourceChannel(value, source_kind="observed_scan")


@pytest.mark.parametrize("shape", [(65, 32, 32), (0, 2, 3), (True, 2, 3), (3., 4, 5), (3, 4)])
def test_view_shape_bounds(shape):
    with pytest.raises(mv.MultiscaleInputError):
        mv.prepare_multiscale_views(source(), coarse_shape=shape)


def test_nonfractional_native_cavity_and_nominal_ranges_are_required():
    original = source()
    for name, values, kind in [("nominal_target", np.full(original.shape, 1.1), "supplied_annotation"),
        ("nominal_tissue", np.full(original.shape, -.1), "supplied_annotation"),
        ("observed_cavity", np.full(original.shape, .5), "observed_procedure_state")]:
        channels = dict(original.channels)
        channels[name] = mv.PermittedSourceChannel(values, source_kind=kind)
        with pytest.raises(mv.MultiscaleInputError):
            replace(original, channels=channels)


def test_no_in_place_normalization_or_source_change_across_two_view_choices():
    original = source()
    before = original.permitted_hash
    native_intensity = np.array(original.channels["structural_intensity"].data)
    a = mv.prepare_multiscale_views(original, local_shape=(3, 3, 3), coarse_shape=(3, 3, 3))
    b = mv.prepare_multiscale_views(original, local_shape=(5, 5, 5), coarse_shape=(2, 2, 2))
    assert original.permitted_hash == before == a.whole_source.source_hash == b.whole_source.source_hash
    np.testing.assert_array_equal(original.channels["structural_intensity"].data, native_intensity)
    assert a.whole_source.preprocessing_hash != b.whole_source.preprocessing_hash
    assert a.report["proposal_effect"] == "none_no_task_or_action_inventory_input"
