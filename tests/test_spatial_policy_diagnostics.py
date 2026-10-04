"""Analytic coverage and transparent instrumentation; no training or patient reads."""
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.spatial_observations import (
    ObservedChannel, ObservedProcedureState, SpatialAction, SpatialInputs, build_spatial_observation,
)
from resectionlab.spatial_policy_diagnostics import (
    NativePreviewProfiler, nominal_depth_coverage, segment_fraction_inside_grid, spatial_coverage,
)


@pytest.mark.parametrize("first,last,expected", [
    ((-2, 0, 0), (2, 0, 0), .5), ((0, 0, 0), (1, 1, 1), 1.),
    ((-2, 2, 0), (2, 2, 0), 0.), ((0, 0, 0), (0, 0, 0), 1.),
    ((2, 0, 0), (2, 0, 0), 0.),
])
def test_segment_visibility_uses_continuous_length_not_sample_count(first, last, expected):
    assert segment_fraction_inside_grid(first, last) == pytest.approx(expected)


def test_crop_coverage_and_tool_ray_report_only_visible_nominal_annotation():
    shape = (5, 7, 9)
    full = np.zeros((9, 11, 13), np.float32)
    full[2, 2, 2], full[8, 10, 12] = 1., 1.
    affine = np.eye(4)
    affine[:3, 3] = 2.
    access = AccessWindow((4., 5., 1.), (0, 0, 1), 2.)
    tool = ToolGeometry("probe", .2, .2, 20., 35., 1.)
    inputs = SpatialInputs({
        "structural_intensity": ObservedChannel(np.ones(shape), source_kind="observed_scan"),
        "nominal_target": ObservedChannel(full[2:7, 2:9, 2:11], source_kind="supplied_annotation"),
        "observed_cavity": ObservedChannel(np.zeros(shape), source_kind="observed_procedure_state"),
    }, affine, "annotation_assisted", "analytic-coverage")
    obs = build_spatial_observation(inputs, (SpatialAction("STOP"),
        SpatialAction("ACTION", access.center_mm, (4., 5., 6.), tool)), ObservedProcedureState(access, 0, 3))
    receipt = spatial_coverage(obs, source_shape=full.shape, source_affine=np.eye(4), nominal_target=full)
    assert receipt["crop_origin_source_voxels"] == [2., 2., 2.]
    assert receipt["nominal_target_positive_voxels_in_crop"] == 1
    assert receipt["nominal_target_positive_voxels_total"] == 2
    assert receipt["nominal_target_mass_fraction_visible"] == .5
    action = receipt["actions"][0]
    assert not action["entry_inside_center_bounds"] and action["tip_inside_center_bounds"]
    assert action["ray_samples_inside"] == 4
    assert action["straight_segment_fraction_inside"] == pytest.approx(.8)


def test_depth_upper_bound_does_not_call_an_out_of_scope_target_reachable():
    target = np.zeros((5, 5, 5))
    target[2, 2, 4] = 1.
    case = SimpleNamespace(affine_ras_mm=np.eye(4), access=AccessWindow((2, 2, -.5), (0, 0, 1), 2),
        nominal_target=target, tools=(ToolGeometry("probe", .2, .2, 20., 35., 1.),),
        _candidate_voxels=((2, 2, 0), (2, 2, 2)),
        _candidate_scope="fixed_access_grid_3columns_8depths_within_actor_crop")
    result = nominal_depth_coverage(case)
    assert result["retained_depth_offsets_voxels"] == [0, 2]
    assert result["distal_tip_capsule_axial_upper_bound_mm"] == pytest.approx(2.7)
    assert result["nominal_target_centers_beyond_axial_upper_bound"] == 1
    assert "ignores" in result["interpretation"]


def test_preview_profiler_preserves_outputs_exceptions_and_restores_engine_method():
    result = SimpleNamespace(feasible=False, reason="obstruction", microsteps=(1, 2))
    error = ValueError("original error")

    class Engine:
        def preview_stroke(self, token, *, entry_mm=None):
            assert entry_mm is result
            if token == "raise":
                raise error
            return result

    original = Engine.preview_stroke
    with NativePreviewProfiler(Engine) as profiler:
        with profiler.phase("constructor_and_inventory"):
            assert Engine().preview_stroke("return", entry_mm=result) is result
            with pytest.raises(ValueError) as caught:
                Engine().preview_stroke("raise", entry_mm=result)
            assert caught.value is error
        row = profiler.snapshot()["phases"]["constructor_and_inventory"]
        assert (row["started"], row["returned"], row["raised"], row["rejected"], row["returned_microsteps"]) == (2, 1, 1, 1, 2)
    assert Engine.preview_stroke is original
