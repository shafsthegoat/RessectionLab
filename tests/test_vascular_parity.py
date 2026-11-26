import numpy as np
import pytest
from scipy import ndimage as ndi

from resectionlab.vascular_parity import (
    agreement, betti, betti_error, centerline, cl_dice, dice, floating_components, image_sharpness,
    patch_betti_error, source_deviation, surface_quality,
)


def ball(shape, center, radius):
    axes = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return sum((axis - c) ** 2 for axis, c in zip(axes, center)) <= radius ** 2


def ring(shape=(32, 32, 9), radius=9, tube=2):
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    planar = np.hypot(x - shape[0] // 2, y - shape[1] // 2) - radius
    return planar ** 2 + (z - shape[2] // 2) ** 2 <= tube ** 2


def test_overlap_requires_one_grid_and_reports_identity_and_disjoint_masks():
    first = ball((20, 20, 20), (10, 10, 10), 5)
    assert dice(first, first) == 1
    assert dice(first, ball((20, 20, 20), (3, 3, 3), 2)) == 0
    assert dice(np.zeros_like(first), np.zeros_like(first)) is None
    with pytest.raises(ValueError):
        dice(first, first[:10])


def test_centerline_dice_penalises_a_missing_thin_branch_more_than_overlap_does():
    trunk = np.zeros((40, 40, 12), bool)
    trunk[4:36, 4:9, 4:9] = True
    branch = np.zeros_like(trunk)
    branch[20, 9:38, 6] = True
    full = trunk | branch
    assert cl_dice(full, full) == pytest.approx(1)
    assert dice(trunk, full) > 0.97
    assert cl_dice(trunk, full) < 0.8
    assert cl_dice(np.zeros_like(full), full) is None


def test_centerline_keeps_a_voxel_in_components_that_thinning_erases():
    even = np.zeros((40, 14, 14), bool)
    even[4:36, 4:8, 4:8] = True
    speck = np.zeros_like(even)
    speck[38, 12, 12] = True
    line = centerline(even | speck)
    assert line[38, 12, 12] and (line & even).any()
    assert not (line & ~(even | speck)).any()


def test_betti_numbers_count_components_loops_and_cavities():
    shape = (24, 24, 24)
    solid = ball(shape, (12, 12, 12), 6)
    assert betti(solid) == {'b0': 1, 'b1': 0, 'b2': 0}
    assert betti(solid & ~ball(shape, (12, 12, 12), 3)) == {'b0': 1, 'b1': 0, 'b2': 1}
    assert betti(ring()) == {'b0': 1, 'b1': 1, 'b2': 0}
    pair = ball(shape, (6, 6, 6), 3) | ball(shape, (17, 17, 17), 3)
    assert betti(pair)['b0'] == 2
    assert betti(np.zeros(shape, bool)) == {'b0': 0, 'b1': 0, 'b2': 0}
    # Voxels that touch only at a corner are one 26-connected component.
    corner = np.zeros((4, 4, 4), bool)
    corner[1, 1, 1] = corner[2, 2, 2] = True
    assert betti(corner) == {'b0': 1, 'b1': 0, 'b2': 0}


def test_betti_error_reports_an_opened_loop_as_a_cycle_difference():
    closed = ring()
    opened = closed.copy()
    opened[:, :4 + 6, :][14:19] = False
    assert betti(opened) == {'b0': 1, 'b1': 0, 'b2': 0}
    error = betti_error(opened, closed)
    assert (error['b0_error'], error['b1_error']) == (0, 1)


def test_floating_components_are_counted_against_roots_and_never_removed():
    shape = (30, 30, 30)
    tree = np.zeros(shape, bool)
    tree[5:25, 14:16, 14:16] = True
    speck = ball(shape, (26, 4, 4), 1)
    mask = tree | speck
    before = mask.copy()
    report = floating_components(mask, spacing=(0.5, 0.5, 0.5))
    assert (report['components'], report['floating']) == (2, 1)
    assert report['floating_voxels'] == int(speck.sum())
    assert report['floating_mm3'] == pytest.approx(speck.sum() / 8)
    np.testing.assert_array_equal(mask, before)
    assert floating_components(mask, roots=speck)['floating_voxels'] == int(tree.sum())
    assert floating_components(mask, roots=mask)['floating'] == 0
    assert floating_components(np.zeros(shape, bool))['floating'] == 0
    with pytest.raises(ValueError, match='No root'):
        floating_components(mask, roots=np.roll(speck, 5, axis=1) & ~mask)


def test_surface_quality_measures_cube_edges_and_open_boundaries():
    points = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    closed = surface_quality(points, faces)
    assert closed['open_edges'] == 0 and closed['non_manifold_edges'] == 0
    assert closed['normal_turn_degrees_p95'] == pytest.approx(90)
    assert closed['normal_turn_degrees_p50'] in (pytest.approx(0), pytest.approx(90))
    assert surface_quality(points, faces[:-2])['open_edges'] == 4
    assert surface_quality(points, np.zeros((0, 3), int))['normal_turn_degrees_p95'] is None


def edge_volume(shape=(48, 48, 48), plane=20.0):
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    return np.broadcast_to(1000 / (1 + np.exp((x - plane) / 0.4)), shape).astype(np.float32)


def wall(plane, frame, normal=(1.0, 0, 0)):
    grid = np.stack(np.meshgrid(np.arange(10, 38, 3.0), np.arange(10, 38, 3.0), indexing='ij'), -1).reshape(-1, 2)
    points = np.column_stack([np.full(len(grid), plane), grid]) @ frame[:3, :3].T + frame[:3, 3]
    return points, np.tile(normal, (len(grid), 1))


def test_source_deviation_tracks_offset_in_millimetres_and_voxels():
    frame = np.diag([0.5, 0.5, 0.5, 1.0])
    volume = edge_volume()
    on_edge = source_deviation(*wall(20.0, frame), volume, frame)
    assert on_edge['resolved_fraction'] == 1 and on_edge['voxels_p95'] <= 0.11
    offset = source_deviation(*wall(21.0, frame), volume, frame)
    assert offset['mm_p50'] == pytest.approx(0.5, abs=0.06)
    assert offset['voxels_p50'] == pytest.approx(1.0, abs=0.11)


def test_source_deviation_counts_walls_that_left_the_scan_as_full_reach_misses():
    frame = np.diag([0.5, 0.5, 0.5, 1.0])
    volume = edge_volume()
    far = source_deviation(*wall(30.0, frame), volume, frame)
    assert far['resolved_fraction'] == 0 and far['voxels_p95'] == pytest.approx(2.0)
    near, normals = wall(20.0, frame)
    mixed = source_deviation(np.vstack([near, wall(30.0, frame)[0]]), np.vstack([normals, normals]), volume, frame)
    assert mixed['resolved_fraction'] == pytest.approx(0.5)
    assert mixed['voxels_p95'] == pytest.approx(2.0)
    flat = source_deviation(near, normals, np.zeros_like(volume), frame, min_drop=1)
    assert flat['resolved_fraction'] == 0
    assert source_deviation(np.zeros((0, 3)), np.zeros((0, 3)), volume, frame)['mm_p95'] is None


def test_source_deviation_measures_physical_distance_on_anisotropic_grids():
    frame = np.diag([1.0, 0.5, 0.5, 1.0])
    volume = edge_volume()
    # Half a voxel along the coarse axis is 0.5 mm, one fine voxel.
    offset = source_deviation(*wall(20.5, frame), volume, frame)
    assert offset['voxel_mm'] == 0.5
    assert offset['mm_p50'] == pytest.approx(0.5, abs=0.06)


def test_source_deviation_uses_the_near_wall_of_a_thin_vessel():
    shape = (40, 40, 40)
    x, y, _ = np.ogrid[:shape[0], :shape[1], :shape[2]]
    radius = np.hypot(x - 20, y - 20)
    volume = np.broadcast_to(1000 / (1 + np.exp((radius - 0.8) / 0.25)), shape).astype(np.float32)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    points = np.column_stack([20 + 0.8 * normals[:, 0], 20 + 0.8 * normals[:, 1], np.full(len(angles), 20.0)])
    thin = source_deviation(points, normals, volume, np.eye(4))
    assert thin['resolved_fraction'] == 1 and thin['voxels_p95'] <= 0.25


def test_image_sharpness_orders_a_blurred_frame_behind_its_source():
    frame = np.zeros((160, 160), np.float32)
    frame[40:120, 60:100] = 1
    frame[20:30] = 0.6
    soft = ndi.gaussian_filter(frame, 2.5)
    sharp, blurred = image_sharpness(frame), image_sharpness(soft)
    assert sharp['edges'] and blurred['edges']
    assert sharp['rise_px_p50'] < 1.5 < blurred['rise_px_p50']
    # The 10 to 90 percent rise of a Gaussian edge is 2.563 sigma.
    block = np.zeros((200, 200), np.float32)
    block[50:150, 50:150] = 1
    for sigma in (2.5, 5.0):
        assert image_sharpness(ndi.gaussian_filter(block, sigma))['rise_px_p50'] == pytest.approx(2.563 * sigma, rel=0.12)
    assert sharp['edge_energy'] > blurred['edge_energy']
    colour = np.repeat((frame * 255)[..., None], 3, axis=2)
    assert image_sharpness(colour)['rise_px_p50'] == pytest.approx(sharp['rise_px_p50'])
    assert image_sharpness(np.zeros((160, 160)))['rise_px_p50'] is None


def test_patch_betti_error_does_not_let_noise_loops_cancel_missing_loops():
    shape = (40, 20, 9)
    reference = np.zeros(shape, bool)
    reference[:20] = ring((20, 20, 9), radius=6, tube=1)
    predicted = np.zeros(shape, bool)
    predicted[20:] = ring((20, 20, 9), radius=6, tube=1)
    assert betti_error(predicted, reference)['b1_error'] == 0
    patch = patch_betti_error(predicted, reference, (1.0, 1.0, 1.0), cube_mm=20.0, bounds=((0, 0, 0), shape))
    assert patch['cubes'] == 2 and patch['cube_voxels'] == [20, 20, 20]
    assert patch['b1_sum'] == 2 and patch['b1_mean'] == 1 and patch['b0_sum'] == 2
    same = patch_betti_error(reference, reference, (1.0, 1.0, 1.0), cube_mm=20.0, bounds=((0, 0, 0), shape))
    assert same['b0_sum'] == same['b1_sum'] == 0
    assert agreement(reference, reference, (1.0, 1.0, 1.0))['patch_betti']['b1_sum'] == 0
