"""Analytic and adversarial geometry contracts, separate from patient claims."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from resectionlab.geometry import (
    AccessWindow,
    GENERIC_TOOLS,
    GeometryScene,
    SphereObstacle,
    ToolGeometry,
    ToolPose,
    _segment_cell_distances,
    capsule_voxel_indices,
    check_motion,
    check_pose,
    check_route,
    interpolate_pose,
    point_segment_distances,
    segment_segment_distance,
)


def empty_scene(*, spheres=(), exposure=False, affine=None):
    return GeometryScene(np.zeros((40, 40, 40), dtype=bool), np.eye(4) if affine is None else affine,
                         np.ones((40, 40, 40), bool) if exposure else None,
                         sphere_obstacles=spheres)


def compact_tool(**kwargs):
    fields = dict(tool_id="analytic", tip_radius_mm=0.2, shaft_radius_mm=0.4,
                  working_length_mm=12, tip_length_mm=2, max_access_angle_deg=60)
    fields.update(kwargs)
    return ToolGeometry(**fields)


def reasons(result):
    return {failure.reason for failure in result.failures}


def test_sphere_clearance_has_known_physical_answer():
    tool = compact_tool()
    scene = empty_scene(spheres=(SphereObstacle([12, 10, 15], 0.5),))
    result = check_pose(tool, ToolPose([10, 10, 22], [0, 0, 1]), scene)
    assert result.feasible
    assert result.clearance_mm == pytest.approx(2 - 0.5 - 0.4)


def test_tip_fits_but_whole_shaft_hits_sphere():
    tool = compact_tool(shaft_radius_mm=1)
    pose = ToolPose([10, 10, 22], [0, 0, 1])
    sphere = SphereObstacle([10.9, 10, 14], 0.1)
    assert np.linalg.norm(pose.tip_mm - sphere.center_mm) > tool.tip_radius_mm + sphere.radius_mm
    result = check_pose(tool, pose, empty_scene(spheres=(sphere,)))
    assert not result.feasible
    assert "SPHERE_COLLISION" in reasons(result)


def test_tip_radius_differs_from_proximal_shaft_radius():
    scene = empty_scene(spheres=(SphereObstacle([10.5, 10, 21.8], 0.1),))
    result = check_pose(compact_tool(shaft_radius_mm=1), ToolPose([10, 10, 22], [0, 0, 1]), scene)
    assert result.feasible


def test_aperture_uses_shaft_radius_and_oblique_cross_section():
    scene = empty_scene()
    access = AccessWindow([10, 10, 10], [0, 0, 1], 0.7)
    pose = ToolPose([10, 10, 15], [0, 0, 1])
    assert check_pose(compact_tool(), pose, scene, access).feasible
    assert "ACCESS_APERTURE" in reasons(check_pose(compact_tool(shaft_radius_mm=0.8), pose, scene, access))
    axis = np.array([np.sqrt(3) / 2, 0, 0.5])
    angled_pose = ToolPose(access.center_mm + 5 * axis, axis)
    assert "ACCESS_APERTURE" in reasons(check_pose(compact_tool(), angled_pose, scene, access))


def test_reach_and_orientation_are_separate_constraints():
    tool = compact_tool(max_access_angle_deg=25)
    scene = empty_scene()
    window = AccessWindow([10, 10, 10], [0, 0, 1], 20)
    assert "WORKING_REACH" in reasons(check_pose(tool, ToolPose([10, 10, 23], [0, 0, 1]), scene, window))
    angle = np.deg2rad(40)
    axis = [np.sin(angle), 0, np.cos(angle)]
    assert "ACCESS_ANGLE" in reasons(check_pose(tool, ToolPose(window.center_mm + 5 * np.array(axis), axis), scene, window))


def test_exact_tangency_is_not_certified_as_clearance():
    tool = compact_tool()
    scene = empty_scene(spheres=(SphereObstacle([10.9, 10, 15], 0.5),))
    assert not check_pose(tool, ToolPose([10, 10, 20], [0, 0, 1]), scene).feasible


def test_crossed_tools_collide_when_both_tips_are_clear():
    tool = compact_tool()
    first = ToolPose([10, 10, 20], [0, 0, 1])
    second = ToolPose([16, 10, 14], [1, 0, 0])
    assert np.linalg.norm(first.tip_mm - second.tip_mm) > 5
    result = check_pose(tool, first, empty_scene(), other_tools=((tool, second),))
    assert not result.feasible and "TOOL_COLLISION" in reasons(result)


def test_lateral_sweep_collides_between_clear_endpoints_even_with_large_step():
    tool = compact_tool(shaft_radius_mm=0.2)
    scene = empty_scene(spheres=(SphereObstacle([15, 15, 16], 0.02),))
    start, end = ToolPose([10, 15, 22], [0, 0, 1]), ToolPose([20, 15, 22], [0, 0, 1])
    assert check_pose(tool, start, scene).feasible
    assert check_pose(tool, end, scene).feasible
    for step in (0.1, 3.0, 100.0):
        assert not check_motion(tool, start, end, scene, max_surface_step_mm=step).feasible


def test_rotating_shaft_sweep_collides_with_fixed_tip_and_clear_endpoints():
    tool = compact_tool(shaft_radius_mm=0.15)
    tip = [20, 20, 20]
    scene = empty_scene(spheres=(SphereObstacle([20, 20, 11], 0.1),))
    start = ToolPose(tip, [-0.5, 0, np.sqrt(3) / 2])
    end = ToolPose(tip, [0.5, 0, np.sqrt(3) / 2])
    assert check_pose(tool, start, scene).feasible
    assert check_pose(tool, end, scene).feasible
    assert not check_motion(tool, start, end, scene, max_surface_step_mm=2).feasible


def test_axial_sweep_is_exact_and_step_independent():
    tool = compact_tool()
    scene = empty_scene(spheres=(SphereObstacle([10, 10, 23], 0.1),))
    start, end = ToolPose([10, 10, 10], [0, 0, 1]), ToolPose([10, 10, 30], [0, 0, 1])
    # Obstacle misses both endpoint tips but is crossed during insertion.
    results = [check_motion(tool, start, end, scene, max_surface_step_mm=step) for step in (0.1, 100)]
    assert all(not result.feasible for result in results)
    np.testing.assert_array_equal(results[0].swept_voxel_indices, results[1].swept_voxel_indices)


def test_oblique_insertion_from_aperture_plane_is_valid():
    tool = compact_tool()
    scene = empty_scene()
    access = AccessWindow([10, 10, 10], [0, 0, 1], 4)
    axis = np.array([0.3, 0.4, np.sqrt(0.75)])
    result = check_motion(tool, ToolPose(access.center_mm, axis),
                          ToolPose(access.center_mm + 8 * axis, axis), scene, access)
    assert result.feasible, result.failures


def test_voxel_cell_faces_do_not_vanish_at_coarse_spacing():
    affine = np.diag([4, 2, 1, 1])
    mask = np.zeros((40, 40, 40), bool)
    mask[5, 10, 10] = True
    scene = GeometryScene(mask, affine)
    # Obstacle center x=20; cell spans x=18..22. Tool is just inside its face.
    result = check_pose(compact_tool(), ToolPose([22.1, 20, 15], [0, 0, 1]), scene)
    assert not result.feasible


def test_exact_cell_geometry_does_not_invent_neighbor_contacts():
    scene = empty_scene()
    cells = capsule_voxel_indices(scene, [10, 10, 10], [10, 10, 10], 0.2)
    np.testing.assert_array_equal(cells, [[10, 10, 10]])
    cells = capsule_voxel_indices(scene, [10, 10, 10], [10, 10, 14], 0.2)
    np.testing.assert_array_equal(cells, [[10, 10, z] for z in range(10, 15)])


def test_piecewise_quadratic_box_distance_matches_scalar_optimizer():
    rng = np.random.default_rng(8371)
    scene = empty_scene(affine=np.diag([2.3, 0.7, 1.2, 1]))
    for _ in range(40):
        start, end = rng.uniform(-3, 3, (2, 3))
        half = np.array([2.3, 0.7, 1.2]) / 2
        objective = lambda t: float(np.sum(np.maximum(np.abs(start + t * (end - start)) - half, 0)**2))
        numerical = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-13})
        reference = np.sqrt(min(numerical.fun, objective(0), objective(1)))
        actual = _segment_cell_distances(scene, np.array([[0, 0, 0]]), start, end)[0]
        assert actual == pytest.approx(reference, abs=1e-7)


def test_sheared_cells_are_conservatively_covered():
    affine = np.array([[1, 0.6, 0, 0], [0, 1, 0.3, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    scene = empty_scene(affine=affine)
    corner = scene.voxel_to_world([10.49, 10.49, 10.49])
    cells = capsule_voxel_indices(scene, corner, corner, 0.01)
    assert (cells == [10, 10, 10]).all(axis=1).any()


def test_exposure_is_unique_physical_volume_and_never_removal():
    scene = empty_scene(exposure=True, affine=np.diag([2, 1, 1, 1]))
    tool = compact_tool()
    pose = ToolPose([20, 10, 20], [0, 0, 1])
    result = check_route(tool, [pose, pose, pose], scene)
    assert result.exposure_volume_mm3 == len(result.swept_voxel_indices) * 2
    assert "not removal" in result.exposure_definition
    assert not hasattr(result, "removed_volume_mm3")


def test_missing_exposure_is_unknown_not_zero():
    result = check_pose(compact_tool(), ToolPose([10, 10, 20], [0, 0, 1]), empty_scene())
    assert result.exposure_volume_mm3 is None
    assert "normal_tissue_exposure_unassessed" in result.unknowns


def test_distal_tip_outside_volume_rejected_proximal_unknown_flagged():
    scene = empty_scene()
    result = check_pose(compact_tool(), ToolPose([10, 10, -0.6], [0, 0, 1]), scene)
    assert "TIP_OUTSIDE_IMAGE" in reasons(result)
    inside = check_pose(compact_tool(), ToolPose([10, 10, 4], [0, 0, 1]), scene)
    assert inside.feasible
    assert "tool_geometry_outside_image_unassessed" in inside.unknowns


def test_affine_snapshot_and_pose_cannot_be_mutated_or_unfrozen():
    affine = np.eye(4)
    scene = empty_scene(affine=affine)
    affine[0, 0] = 7
    assert scene.affine[0, 0] == 1
    pose = ToolPose([10, 10, 20], [0, 0, 1])
    result = check_pose(compact_tool(), pose, scene)
    for array in (scene.affine, scene.forbidden_mask, scene._inverse, pose.tip_mm,
                  pose.axis_unit, result.swept_voxel_indices):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("kwargs", [{"shaft_radius_mm": -1}, {"tip_radius_mm": np.nan},
                                     {"working_length_mm": np.inf}, {"max_access_angle_deg": 90},
                                     {"tip_length_mm": 15}])
def test_invalid_tool_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        compact_tool(**kwargs)


def test_invalid_poses_affines_and_ambiguous_motion_rejected():
    with pytest.raises(ValueError):
        ToolPose([1, 2, np.nan], [0, 0, 1])
    with pytest.raises(ValueError):
        ToolPose([1, 2, 3], [0, 0, 0])
    with pytest.raises(ValueError):
        empty_scene(affine=np.zeros((4, 4)))
    with pytest.raises(ValueError):
        interpolate_pose(ToolPose([1, 2, 3], [0, 0, 1]), ToolPose([1, 2, 3], [0, 0, -1]), 0.5)
    with pytest.raises(ValueError):
        check_route(compact_tool(), [[0, 0, 0], [1, 2, 3]], empty_scene())


def test_degenerate_and_parallel_segment_distance():
    assert segment_segment_distance([0, 0, 0], [0, 0, 0], [2, -2, 0], [2, 2, 0]) == 2
    assert segment_segment_distance([0, 0, 0], [0, 0, 5], [2, 0, 0], [2, 0, 5]) == 2
    assert segment_segment_distance([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]) == 0


def test_two_generic_tools_have_distinct_geometry_and_declared_source():
    assert len(GENERIC_TOOLS) >= 2
    assert GENERIC_TOOLS[0].shaft_radius_mm != GENERIC_TOOLS[1].shaft_radius_mm
    assert all("generic" in tool.parameter_source for tool in GENERIC_TOOLS)
