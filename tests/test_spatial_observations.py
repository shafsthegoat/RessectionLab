"""Permitted spatial inputs stay separate from unobserved evaluator truth."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.spatial_observations import (
    ACTION_GEOMETRY_NAMES, CHANNEL_NAMES, STATE_FEATURE_NAMES,
    ObservedChannel, ObservedProcedureState, SpatialAction, SpatialInputError,
    SpatialInputs, build_spatial_observation,
)


def inputs(*, track="synthetic_scan", **updates):
    image = np.arange(60, dtype=np.float32).reshape(3, 4, 5) / 60
    channels = {
        "structural_intensity": ObservedChannel(image, source_kind="synthetic_scan"),
        "observed_cavity": ObservedChannel(np.zeros_like(image), source_kind="observed_procedure_state"),
        "nominal_tissue": ObservedChannel(image > .1, source_kind="derived_from_scan",
            derived_from=("structural_intensity",), derivation="fixed scan threshold > .1"),
    }
    channels.update(updates)
    return SpatialInputs(channels, np.diag([1., 2., 3., 1.]), track, "permitted-scan-1")


def observation(source=None, *, steps=0):
    actions = (SpatialAction("STOP"), SpatialAction("stroke-1", (0., 0., 0.), (0., 0., 3.),
        ToolGeometry("declared-tool", .2, .3, 20., tip_length_mm=1.)))
    state = ObservedProcedureState(AccessWindow((0, 0, 0), (0, 0, 1), 2), steps, 2)
    return build_spatial_observation(source or inputs(), actions, state)


def test_actor_schema_contains_scan_geometry_and_observed_state_only():
    result = observation()
    assert result.image_channels.shape == (6, 3, 4, 5)
    assert result.action_geometry.shape == (2, 16)
    assert result.state_features.shape == (10,)
    assert result.image_channels.dtype == np.float32
    assert result.action_geometry.dtype == result.state_features.dtype == np.float64
    assert len(CHANNEL_NAMES) == 6 and len(ACTION_GEOMETRY_NAMES) == 16 and len(STATE_FEATURE_NAMES) == 10
    assert result.channel_available.tolist() == [True, True, False, True, False, False]
    np.testing.assert_array_equal(result.action_geometry[1, 7:10], [0, 0, 1])
    np.testing.assert_array_equal(result.spacing_mm, [1, 2, 3])
    assert result.action_mask.tolist() == [True, True]
    assert observation(steps=2).action_mask.tolist() == [True, False]


def test_observed_zero_is_distinct_from_missing_and_uncovered_values_are_erased():
    missing = observation()
    known = observation(inputs(nominal_motor=ObservedChannel(np.zeros((3, 4, 5)),
        source_kind="derived_from_scan", derived_from=("structural_intensity",), derivation="test-only scan estimator")))
    assert np.array_equal(missing.image_channels, known.image_channels)
    assert missing.fingerprint != known.fingerprint
    assert known.channel_available[4] and known.coverage[4].all()
    image = np.ones((3, 4, 5), np.float32)
    coverage = np.ones(image.shape, bool)
    coverage[0] = False
    image[0] = 12345
    result = observation(inputs(structural_intensity=ObservedChannel(image, coverage, "synthetic_scan")))
    assert not result.coverage[0, 0].any() and not result.image_channels[0, 0].any()


@pytest.mark.parametrize("name", ["structural_intensity", "observed_cavity"])
def test_missing_essential_evidence_is_an_explicit_gate(name):
    with pytest.raises(SpatialInputError, match="ESSENTIAL_EVIDENCE_MISSING"):
        inputs(**{name: ObservedChannel()})


@pytest.mark.parametrize("name", ["nominal_target", "nominal_motor", "nominal_language"])
def test_scan_track_forbids_reference_annotations_but_annotation_track_is_explicit(name):
    supplied = ObservedChannel(np.ones((3, 4, 5)), source_kind="supplied_annotation")
    with pytest.raises(SpatialInputError, match="reference annotations"):
        inputs(**{name: supplied})
    assert observation(inputs(track="annotation_assisted", **{name: supplied})).track == "annotation_assisted"


def test_raw_image_track_requires_observed_image_not_synthetic_or_annotation():
    with pytest.raises(SpatialInputError, match="Structural intensity"):
        inputs(track="inference_only")
    result = inputs(track="inference_only", structural_intensity=ObservedChannel(
        np.ones((3, 4, 5)), source_kind="observed_scan"))
    assert observation(result).track == "inference_only"


def test_physical_oblique_reflected_affine_preserves_spacing_without_target_crop():
    theta = .4
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, -1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([.8, 1.6, 2.4])
    affine[:3, 3] = [9, -3, 21]
    result = observation(replace(inputs(), affine_ras_mm=affine))
    np.testing.assert_array_equal(result.affine_ras_mm, affine)
    np.testing.assert_allclose(result.spacing_mm, [.8, 1.6, 2.4])


@pytest.mark.parametrize("bad", [np.eye(4).astype(complex), np.full((4, 4), np.nan), np.zeros((4, 4))])
def test_rejects_complex_nonfinite_or_singular_frames(bad):
    with pytest.raises(SpatialInputError):
        replace(inputs(), affine_ras_mm=bad)


def test_rejects_complex_action_points_and_nonempty_unavailable_payload():
    with pytest.raises(SpatialInputError):
        SpatialAction("x", np.zeros(3, complex), (0, 0, 1), ToolGeometry("x", .2, .2, 10))
    with pytest.raises(SpatialInputError, match="Unavailable"):
        ObservedChannel(np.zeros((3, 4, 5)))


def test_direct_dto_cannot_bypass_source_provenance_or_introduce_oracle_masks():
    value = observation()
    provenance = {key: dict(record) for key, record in value.channel_provenance.items()}
    provenance["nominal_tissue"]["source_kind"] = "supplied_annotation"
    with pytest.raises(SpatialInputError, match="reference annotations"):
        replace(value, channel_provenance=provenance)
    with pytest.raises(SpatialInputError, match="masking"):
        replace(value, action_mask=np.array([True, False]))
    with pytest.raises(SpatialInputError, match="spacing"):
        replace(value, spacing_mm=np.ones(3))
    with pytest.raises(SpatialInputError, match="six-channel"):
        replace(value, image_channels=value.image_channels[:3])


def test_direct_dto_cannot_hide_payload_behind_missingness_or_coverage():
    value = observation()
    images = value.image_channels.copy()
    images[4] = 1
    with pytest.raises(SpatialInputError, match="Uncovered"):
        replace(value, image_channels=images)
    provenance = {key: dict(record) for key, record in value.channel_provenance.items()}
    provenance["nominal_motor"]["oracle_target_sum"] = 55
    with pytest.raises(SpatialInputError, match="unknown or missing"):
        replace(value, channel_provenance=provenance)


def test_source_and_actor_snapshots_are_immutable_in_values_and_interpretation():
    raw = np.ones((3, 4, 5), np.float32)
    source = inputs(structural_intensity=ObservedChannel(raw, source_kind="synthetic_scan"))
    value = observation(source)
    fingerprint = value.fingerprint
    raw[:] = 99
    assert value.image_channels[0].max() == 1
    for array in (source.affine_ras_mm, source.channels["structural_intensity"].data, value.image_channels):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.strides = array.strides[::-1]
    assert value.fingerprint == fingerprint
    object.__setattr__(value, "track", "annotation_assisted")
    with pytest.raises(SpatialInputError, match="reinterpreted"):
        value.assert_intact()


def test_forced_source_provenance_replacement_is_detected_before_build():
    source = inputs()
    object.__setattr__(source.channels["nominal_tissue"], "source_kind", "supplied_annotation")
    with pytest.raises(SpatialInputError, match="reinterpreted"):
        observation(source)


def test_reference_truth_is_not_an_input_and_source_display_id_is_not_a_feature():
    permitted = inputs()
    before = observation(permitted)
    private_reference = np.zeros((3, 4, 5), bool)
    private_reference[:] = True
    after = observation(permitted)
    assert before.fingerprint == after.fingerprint
    assert before.fingerprint == observation(replace(permitted, source_id="different-display-label")).fingerprint
    with pytest.raises(TypeError):
        build_spatial_observation(permitted, (), None, reference_target=private_reference)
