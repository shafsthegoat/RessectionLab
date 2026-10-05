"""Independent analytical controls; no patient arrays, policies, or geometry runs."""
from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from resectionlab.spatial_multiscale_views import (
    MultiscaleInputError, MultiscaleViews, PermittedSourceChannel,
    PermittedVolumeSource, prepare_multiscale_views,
)
from resectionlab.spatial_observations import CHANNEL_NAMES, ObservedChannel, SpatialInputs, SpatialInputError


def source(*, shape=(5, 3, 4), source_id="audit-reference-A", unknown_payload=17., target=None):
    ijk = np.indices(shape)
    coverage = (ijk.sum(axis=0) % 3) != 0
    intensity = (np.arange(np.prod(shape)).reshape(shape) - 12.) / 7
    intensity[~coverage] = unknown_payload
    if target is None:
        target = np.zeros(shape)
        target[0, 0, 0], target[-1, -1, -1] = .75, .25
    channels = {
        "structural_intensity": PermittedSourceChannel(intensity, coverage, "observed_scan"),
        "nominal_tissue": PermittedSourceChannel(np.ones(shape), source_kind="supplied_annotation"),
        "nominal_target": PermittedSourceChannel(target, source_kind="supplied_annotation"),
        "observed_cavity": PermittedSourceChannel((ijk.sum(axis=0) % 2 == 0), source_kind="observed_procedure_state"),
    }
    affine = np.array([[0., -3., 0., 17.], [-2., 0., 0., -21.], [0., 0., 4., 6.], [0., 0., 0., 1.]])
    return PermittedVolumeSource(channels, affine, "annotation_assisted", source_id)


def overlap_reference(values, output_shape):
    """Direct rational 3D cell intersections, independent of separable averaging."""
    result = np.zeros(output_shape)
    source_shape = values.shape
    cell_volume = np.prod([Fraction(n, m) for n, m in zip(source_shape, output_shape)])
    for dest in np.ndindex(output_shape):
        total = 0.
        for cell in np.ndindex(source_shape):
            overlap = Fraction(1)
            for i in range(3):
                a, b = Fraction(dest[i]*source_shape[i], output_shape[i]), Fraction((dest[i]+1)*source_shape[i], output_shape[i])
                overlap *= max(Fraction(0), min(b, cell[i]+1)-max(a, cell[i]))
            total += float(overlap/cell_volume)*float(values[cell])
        result[dest] = total
    return result


def test_exact_odd_cell_overlap_preserves_mass_fractional_unknowns_and_reflected_extent():
    original = source()
    views = prepare_multiscale_views(original, local_shape=(3, 2, 2), coarse_shape=(2, 2, 3))
    coarse = views.whole_source
    det = abs(np.linalg.det(original.affine_ras_mm[:3, :3]))
    coarse_det = abs(np.linalg.det(coarse.affine_ras_mm[:3, :3]))
    for index, name in enumerate(CHANNEL_NAMES):
        channel = original.channels[name]
        if channel.data is None:
            assert not coarse.channel_available[index]
            assert not coarse.image_channels[index].any() and not coarse.coverage_fraction[index].any()
            assert views.report["channel_integrals"][name]["coarse_value_integral_mm3"] is None
            continue
        np.testing.assert_allclose(coarse.image_channels[index], overlap_reference(channel.data, coarse.shape), rtol=0, atol=1e-12)
        np.testing.assert_allclose(coarse.coverage_fraction[index], overlap_reference(channel.coverage, coarse.shape), rtol=0, atol=1e-12)
        assert abs(coarse.image_channels[index].sum()*coarse_det-channel.data.sum()*det) <= 1e-8
        assert abs(coarse.coverage_fraction[index].sum()*coarse_det-channel.coverage.sum()*det) <= 1e-8
    cavity = coarse.image_channels[CHANNEL_NAMES.index("observed_cavity")]
    known = coarse.coverage_fraction[CHANNEL_NAMES.index("structural_intensity")]
    assert np.any((cavity > 0) & (cavity < 1))
    assert np.any((known > 0) & (known < 1))
    for bits in product((0, 1), repeat=3):
        source_corner = np.array([original.shape[i]-.5 if b else -.5 for i,b in enumerate(bits)]+[1.])
        coarse_corner = np.array([coarse.shape[i]-.5 if b else -.5 for i,b in enumerate(bits)]+[1.])
        np.testing.assert_allclose(original.affine_ras_mm @ source_corner, coarse.affine_ras_mm @ coarse_corner, rtol=0, atol=1e-8)
    point = np.array([.2, .8, 1.1, 1.])
    np.testing.assert_allclose(np.linalg.inv(coarse.affine_ras_mm) @ (coarse.affine_ras_mm @ point), point, rtol=0, atol=1e-8)


def test_disconnected_oversized_target_does_not_fabricate_local_coverage():
    original = source()
    views = prepare_multiscale_views(original, local_shape=(3, 2, 2), coarse_shape=(2, 2, 3))
    assert tuple(views.report["local_start_ijk"]) == (1, 0, 1)
    assert views.report["nominal_target_total_mass_mm3"] == pytest.approx(24.)
    assert views.report["target_local_retained_mass_mm3"] == 0
    assert views.report["target_local_clipped_fraction"] == 1
    assert not views.target_local.image_channels[CHANNEL_NAMES.index("nominal_target")].any()
    target = np.zeros(original.shape); target[-1, -1, -1] = .25
    shifted = prepare_multiscale_views(source(target=target), local_shape=(3, 2, 2))
    assert tuple(shifted.report["local_start_ijk"]) == (2, 1, 2)
    assert shifted.report["target_local_clipped_fraction"] == 0
    np.testing.assert_allclose(shifted.target_local.affine_ras_mm[:3,3], (original.affine_ras_mm @ [2,1,2,1])[:3], rtol=0, atol=1e-12)


def test_unknown_payload_and_mixed_audit_id_cannot_change_view_identity():
    left, right = source(), source(source_id="hidden-outcome-reference-B", unknown_payload=-900.)
    assert left.permitted_hash == right.permitted_hash
    a, b = prepare_multiscale_views(left), prepare_multiscale_views(right)
    assert a.target_local.fingerprint == b.target_local.fingerprint
    assert a.whole_source.fingerprint == b.whole_source.fingerprint
    assert dict(a.report) == dict(b.report)


@pytest.mark.parametrize("forbidden", ["reference_target", "postoperative_outcome", "reward_map", "future_cavity"])
def test_forbidden_channel_keys_rejected(forbidden):
    original = source()
    with pytest.raises(MultiscaleInputError):
        replace(original, channels={**original.channels, forbidden:original.channels["nominal_target"]})


def test_coarse_fractional_cavity_cannot_be_relabelled_as_legacy_binary_observation():
    original = source()
    view = prepare_multiscale_views(original, coarse_shape=(2,2,3)).whole_source
    channels = {}
    for i,name in enumerate(CHANNEL_NAMES):
        if view.channel_available[i]:
            p = view.channel_provenance[name]
            channels[name] = ObservedChannel(view.image_channels[i], source_kind=p["source_kind"],
                                             derivation=p["derivation"], derived_from=tuple(p["derived_from"]))
    with pytest.raises(SpatialInputError, match="binary"):
        SpatialInputs(channels, view.affine_ras_mm, original.track, original.permitted_hash)
    with pytest.raises(MultiscaleInputError):
        replace(view, kind="target_local_native")


def test_unavailable_target_never_becomes_an_implicit_label_or_zero_harm():
    original = source()
    for target in (PermittedSourceChannel(), PermittedSourceChannel(np.ones(original.shape), np.zeros(original.shape,bool), "supplied_annotation")):
        changed = replace(original, channels={**original.channels,"nominal_target":target})
        with pytest.raises(MultiscaleInputError, match="TARGET_LOCAL_UNAVAILABLE"):
            prepare_multiscale_views(changed)


def test_source_array_and_interpretation_tampering_fail_closed():
    original = source()
    with pytest.raises((ValueError, MultiscaleInputError)):
        original.channels["nominal_target"].data.setflags(write=True)
    object.__setattr__(original.channels["nominal_target"], "derivation", "altered evidence")
    with pytest.raises(MultiscaleInputError):
        prepare_multiscale_views(original)
    bundle = prepare_multiscale_views(source())
    object.__setattr__(bundle.whole_source, "preprocessing_version", "unknown-v2")
    with pytest.raises(MultiscaleInputError):
        bundle.assert_intact()


def test_view_pair_rejects_cross_source_report_or_preparation_identity():
    a = prepare_multiscale_views(source(), local_shape=(3,2,2), coarse_shape=(2,2,3))
    target = np.ones((5,3,4))*.5
    b = prepare_multiscale_views(source(target=target), local_shape=(3,2,2), coarse_shape=(2,2,3))
    with pytest.raises(MultiscaleInputError):
        MultiscaleViews(a.target_local, a.whole_source, b.report)
    with pytest.raises(MultiscaleInputError):
        MultiscaleViews(a.target_local, b.whole_source, a.report)
    c = prepare_multiscale_views(source(), local_shape=(2,2,2), coarse_shape=(2,2,3))
    with pytest.raises(MultiscaleInputError):
        MultiscaleViews(a.target_local, c.whole_source, a.report)
