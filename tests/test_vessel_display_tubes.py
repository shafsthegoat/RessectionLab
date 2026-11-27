import numpy as np
import pytest
from scipy import ndimage as ndi

from resectionlab.vessel_display_tubes import (WALL_LEVEL, TubeDisplay, display_tags, pack, softened_mask, surface_field,
                                                thick_mask, thin_tubes, unpack, without_lone_samples)

SPACING = (.5, .5, .5)
SHAPE = (48, 48, 48)


def staircase(start, end, shape=SHAPE):
    """A one-voxel line between two points, as a segmentation draws it: lattice steps."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    mask = np.zeros(shape, bool)
    for t in np.linspace(0, 1, 4 * int(np.abs(end - start).max()) + 1):
        mask[tuple(np.rint(start + t * (end - start)).astype(int))] = True
    return mask


def along(field, start, end, samples=400):
    """Trilinear values of a field along a straight line in voxel coordinates."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    points = start[:, None] + np.linspace(0, 1, samples)[None] * (end - start)[:, None]
    return ndi.map_coordinates(np.asarray(field, np.float32), points, order=1)


def widths(field, start, end, level, samples=40):
    """Width of the region at or above a level, across a line in three directions, at points along it (voxels)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = (end - start) / np.linalg.norm(end - start)
    first = np.cross(axis, [.3, .5, .81])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    reach = np.linspace(-4, 4, 321)
    found = []
    for t in np.linspace(.15, .85, samples):
        centre = start + t * (end - start)
        for across in (first, second, (first + second) / np.sqrt(2)):
            values = ndi.map_coordinates(np.asarray(field, np.float32), centre[:, None] + reach[None] * across[:, None], order=1)
            inside = reach[values >= level]
            found.append(inside.max() - inside.min() if len(inside) else 0.)
    return np.array(found)


@pytest.mark.parametrize('start, end', [((6, 8, 10), (41, 30, 22)), ((5, 5, 5), (42, 40, 38)), ((6, 10, 24), (42, 17, 24))])
def test_a_stepped_one_voxel_line_becomes_an_unbroken_tube_of_even_width(start, end):
    line = staircase(start, end)
    packed, receipt = thin_tubes(line, SPACING)
    field, replaced = unpack(packed)
    assert np.array_equal(replaced, line), 'every voxel of the thin line is redrawn'
    inner_start, inner_end = np.add(start, np.subtract(end, start) * .08), np.add(start, np.subtract(end, start) * .92)
    assert along(field, inner_start, inner_end).min() >= .5, 'the tube never breaks along its axis'
    tube = widths(field, start, end, WALL_LEVEL)
    assert tube.min() > 1.2 and tube.std() / tube.mean() < .1
    # What was drawn before, the smoothed mask, breaks or varies about three times as much.
    mask = widths(ndi.gaussian_filter(line.astype(np.float32), .6), start, end, .3)
    assert mask.min() == 0 or mask.std() / mask.mean() > 2.5 * tube.std() / tube.mean()
    # The surface contour is taken at the same wall (the rescaled field saturates, so not to the last tenth).
    contour = widths(surface_field(field), start, end, .5)
    assert np.abs(contour - tube).max() <= .15 and abs(contour.mean() - tube.mean()) < .06
    half_diagonal = np.linalg.norm(SPACING) / 2
    assert receipt['minimumDrawnRadiusMm'] == pytest.approx(.9 * half_diagonal)
    assert receipt['drawnRadiusMm']['min'] >= receipt['minimumDrawnRadiusMm'] - 1e-9
    assert receipt['replacedVoxels'] == int(line.sum()) and receipt['tubeLengthMm'] > 15
    # No centerline voxel leaves the half-voxel box around the voxel it came from.
    assert receipt['maxCenterlineShiftMm'] <= np.linalg.norm(SPACING) / 2 + 1e-9


def test_a_bend_is_followed_not_cut():
    angle = np.linspace(0, np.pi / 2, 400)
    points = np.stack([np.full_like(angle, 10), 12 + 6 * np.cos(angle), 12 + 6 * np.sin(angle)], axis=1)
    bend = np.zeros(SHAPE, bool)
    bend[tuple(np.rint(points).astype(int).T)] = True
    packed, _ = thin_tubes(bend, SPACING)
    field = unpack(packed)[0]
    assert ndi.map_coordinates(field, points[40:-40].T, order=1).min() >= .5
    # Not a chord and not a fat tube: two voxels inside and outside the arc the field is below the wall.
    for radius in (4, 8):
        off = np.stack([np.full_like(angle, 10), 12 + radius * np.cos(angle), 12 + radius * np.sin(angle)], axis=1)
        assert ndi.map_coordinates(field, off[40:-40].T, order=1).max() < WALL_LEVEL


def test_a_thick_vessel_is_left_alone():
    grid = np.indices(SHAPE).astype(float)
    thick = (grid[1] - 24) ** 2 + (grid[2] - 24) ** 2 <= 4.2 ** 2
    thick[:4] = thick[-4:] = False
    packed, receipt = thin_tubes(thick, SPACING)
    assert packed is None and receipt['tubeCenterlineVoxels'] == 0 and receipt['replacedVoxels'] == 0
    assert receipt['centerlineVoxels'] > 20


def trunk_and_branch():
    grid = np.indices(SHAPE).astype(float)
    trunk = (grid[1] - 14) ** 2 + (grid[2] - 24) ** 2 <= 4.2 ** 2
    trunk[:4] = trunk[-4:] = False
    branch = np.zeros(SHAPE, bool)
    branch[24, 14:42, 24] = True
    return trunk, branch & ~trunk


def test_a_thin_branch_on_a_thick_trunk_is_redrawn_and_runs_into_the_trunk():
    trunk, branch = trunk_and_branch()
    packed, receipt = thin_tubes(trunk | branch, SPACING)
    field, replaced = unpack(packed)
    assert replaced[branch][6:].all(), 'the branch beyond the junction is redrawn'
    assert replaced[trunk].mean() < .02, 'the trunk keeps its voxels'
    # Whatever is redrawn at the junction, the trunk stays covered: no notch where the branch leaves.
    covered = np.maximum(field, ndi.gaussian_filter(thick_mask(trunk | branch, packed).astype(np.float32), .6))
    core = ndi.binary_erosion(trunk)
    assert (covered[core] >= .5).all()
    assert np.array_equal(thick_mask(trunk | branch, packed), (trunk | branch) & ~replaced)
    # From the branch tip to the trunk axis the drawing never opens: tube first, trunk voxels after.
    joined = np.maximum(field, thick_mask(trunk | branch, packed))
    assert along(joined, (24, 40, 24), (24, 14, 24)).min() >= .5
    assert receipt['drawnRadiusMm']['max'] <= receipt['thinRadiusMm'] + 1e-9


def test_a_branch_extending_another_layer_ends_inside_that_layer_and_never_redraws_it():
    trunk, branch = trunk_and_branch()
    alone, _ = thin_tubes(branch, SPACING)
    packed, _ = thin_tubes(branch, SPACING, context=trunk)
    field, replaced = unpack(packed)
    assert not replaced[trunk].any() and replaced[branch].all()
    # With the tree beside it the tube runs on through the trunk wall (at 18.2) instead of ending at it.
    assert along(field, (24, 19, 24), (24, 17, 24)).min() >= .5
    assert along(unpack(alone)[0], (24, 19, 24), (24, 17, 24)).min() < .1
    with pytest.raises(ValueError, match='grids'):
        thin_tubes(branch, SPACING, context=trunk[:-1])


def test_a_short_twig_on_a_thin_branch_is_not_drawn_but_a_real_side_branch_is():
    line = np.zeros(SHAPE, bool)
    line[6:42, 24, 24] = True
    twig, side = line.copy(), line.copy()
    twig[20, 25:27, 24] = True   # Two voxels: a lump on the mask.
    side[20, 25:34, 24] = True   # Nine voxels: a branch.
    plain, _ = thin_tubes(line, SPACING)
    with_twig, twig_receipt = thin_tubes(twig, SPACING)
    with_side, side_receipt = thin_tubes(side, SPACING)
    assert twig_receipt['prunedTwigVoxels'] >= 1 and side_receipt['prunedTwigVoxels'] == 0
    assert unpack(with_twig)[1][20, 25:27, 24].all(), 'the lump is not drawn as voxels either'
    # Beside the main tube the twig leaves nothing; the side branch is a tube all along.
    assert unpack(with_twig)[0][20, 27, 24] <= unpack(plain)[0][20, 27, 24] + .08
    assert along(unpack(with_side)[0], (20, 25, 24), (20, 32, 24)).min() >= .5
    # A short piece on its own has two free ends and no fork: it is a vessel, drawn whole.
    short = staircase((10, 10, 10), (12, 10, 10))
    packed, receipt = thin_tubes(short, SPACING)
    assert receipt['prunedTwigVoxels'] == 0 and unpack(packed)[1][short].all() and (unpack(packed)[0][short] >= .5).all()
    # Two voxels that do not touch: two centerline voxels without an edge, each drawn as a dot.
    apart = np.zeros(SHAPE, bool)
    apart[10, 10, 10] = apart[10, 10, 14] = True
    packed, receipt = thin_tubes(apart, SPACING)
    assert receipt['centerlineVoxels'] == 2 and (unpack(packed)[0][apart] >= .5).all() and unpack(packed)[0][10, 10, 12] < .1


def test_a_tree_whose_only_thin_parts_are_twigs_draws_no_tube_and_no_twig_voxels():
    grid = np.indices(SHAPE).astype(float)
    thick = (grid[1] - 24) ** 2 + (grid[2] - 24) ** 2 <= 4.2 ** 2
    thick[:4] = thick[-4:] = False
    lump = thick.copy()
    lump[24, 29:31, 24] = True  # Two voxels standing on the wall.
    packed, receipt = thin_tubes(lump, SPACING)
    if packed is None:
        # The thinning kept no side twig for the lump: nothing is thin and nothing changes.
        assert receipt['tubeCenterlineVoxels'] == 0
    else:
        field, replaced = unpack(packed)
        assert field.max() == 0 and receipt['prunedTwigVoxels'] >= 1 and not replaced[thick].any()


def test_voxels_left_alone_between_tubes_are_not_drawn_as_voxels():
    # A thin line with a lump of five voxels on it: the lump's centerline voxel reads thick, so the
    # thinning alone would leave a few of its voxels standing between the tubes on either side.
    line = np.zeros(SHAPE, bool)
    line[6:42, 24, 24] = True
    line[23:25, 24:26, 24:26] = True
    kept, receipt = thin_tubes(line, SPACING)
    loose, loose_receipt = thin_tubes(line, SPACING, rule=TubeDisplay(speck_voxels=1))
    remaining, standing = thick_mask(line, kept), thick_mask(line, loose)
    assert loose_receipt['leftoverVoxels'] == 0 and receipt['leftoverVoxels'] == int(standing.sum()) - int(remaining.sum())
    sizes = np.bincount(ndi.label(remaining)[0].ravel())[1:]
    assert (sizes >= 8).all(), 'no face-connected group under eight voxels is drawn as voxels'
    # A thick vessel is far above the limit and keeps every voxel.
    trunk, branch = trunk_and_branch()
    packed, trunk_receipt = thin_tubes(trunk | branch, SPACING)
    assert ndi.label(thick_mask(trunk | branch, packed))[1] == 1 and trunk_receipt['leftoverVoxels'] <= 7


def test_the_smallest_tube_follows_the_voxel_diagonal_on_an_anisotropic_grid():
    spacing = (.3, .3, .45)
    line = staircase((6, 8, 10), (41, 30, 22))
    packed, receipt = thin_tubes(line, spacing)
    assert receipt['minimumDrawnRadiusMm'] == pytest.approx(.9 * np.linalg.norm(spacing) / 2)
    assert receipt['thinRadiusMm'] == pytest.approx(1.5 * np.prod(spacing) ** (1 / 3))
    assert along(unpack(packed)[0], (9, 10, 11), (38, 28, 21)).min() >= .5


def test_packing_keeps_the_field_to_seven_bits_and_the_flag_exactly():
    rng = np.random.default_rng(3)
    field, flag = rng.random((5, 6, 7)).astype(np.float32), rng.random((5, 6, 7)) < .3
    packed = pack(field, flag)
    assert packed.dtype == np.uint8
    restored, restored_flag = unpack(packed)
    assert np.array_equal(restored_flag, flag) and np.abs(restored - field).max() <= .5 / 127 + 1e-6


def test_volume_tags_are_the_smoothed_thick_voxels_joined_to_the_tubes():
    trunk, branch = trunk_and_branch()
    own = trunk | branch
    packed, _ = thin_tubes(own, SPACING)
    field, replaced = unpack(packed)
    tags = display_tags(own, packed)
    assert tags.dtype == np.uint8 and tags.shape == own.shape
    # The thick part is smoothed as the desktop smooths a mask: five taps of a 0.6 voxel Gaussian per
    # axis, zero outside, rounded to a byte after every axis.
    taps = np.exp(-np.arange(-2, 3) ** 2 / (2 * .6 ** 2))
    taps /= taps.sum()
    smoothed = (own & ~replaced).astype(np.float64) * 255
    for axis in range(3):
        smoothed = np.rint(ndi.convolve1d(smoothed, taps, axis=axis, mode='constant'))
    assert np.abs(tags.astype(float) - np.maximum(smoothed, np.rint(field * 255))).max() <= 2
    assert tags[30, 14, 24] == 255, 'the trunk core is a full tag'
    assert (tags[24, 22:40, 24] >= 128).all(), 'the branch is tagged all along'
    # Away from the vessel nothing is tagged.
    assert tags[4:10, 40:, 40:].max() == 0


def test_a_softened_mask_keeps_every_voxel_inside_its_surface_and_every_other_outside():
    # Two blocks joined by a three-voxel line one voxel wide: plain smoothing drops the line below one half.
    ribbon = np.zeros((20, 20, 20), bool)
    ribbon[4:8, 9:11, 9:11] = True
    ribbon[8:11, 9, 9] = True
    ribbon[11:15, 9:11, 9:11] = True
    plain = ndi.gaussian_filter(ribbon.astype(np.float32), .6)
    assert plain[9, 9, 9] < .5
    soft = softened_mask(ribbon)
    assert (soft[ribbon] > .5).all() and (soft[~ribbon] < .5).all()
    assert ndi.label(soft >= .5)[1] == 1 and ndi.label(plain >= .5)[1] == 2
    # Away from the rim it is the smoothed mask.
    assert np.allclose(soft[6, 9:11, 9:11], plain[6, 9:11, 9:11])


def test_a_single_sample_touching_the_level_is_not_contoured_unless_it_is_a_kept_voxel():
    values = np.zeros((8, 8, 8), np.float32)
    values[2, 2, 2] = .51            # Alone at the level: a speck.
    values[5, 5, 5] = values[5, 5, 6] = .6   # Two neighbours: a piece of vessel.
    values[2, 6, 6] = .9             # Alone, but one of the layer's own voxels.
    values[0, 0, 0] = .7             # Alone on the grid corner.
    keep = np.zeros(values.shape, bool)
    keep[2, 6, 6] = True
    cleaned = without_lone_samples(values, keep)
    assert cleaned[2, 2, 2] < .5 and cleaned[0, 0, 0] < .5
    assert cleaned[5, 5, 5] == np.float32(.6) and cleaned[5, 5, 6] == np.float32(.6) and cleaned[2, 6, 6] == np.float32(.9)
    assert values[2, 2, 2] == np.float32(.51), 'the input is not written'
    assert without_lone_samples(cleaned, keep) is cleaned


def test_nothing_is_drawn_for_an_empty_layer_and_bad_inputs_are_refused():
    packed, receipt = thin_tubes(np.zeros(SHAPE, bool), SPACING)
    assert packed is None and receipt['ownVoxels'] == 0
    with pytest.raises(ValueError, match='spacings'):
        thin_tubes(np.zeros(SHAPE, bool), (.5, 0, .5))
    with pytest.raises(ValueError):
        TubeDisplay(min_radius_diagonals=.5)
    with pytest.raises(ValueError):
        TubeDisplay(thin_radius_voxels=0)


@pytest.mark.parametrize('allowed', [0, 1, 2, 3, 4])
def test_a_cancelled_request_stops_the_work_at_every_stage(allowed):
    calls = []

    def check():
        calls.append(1)
        if len(calls) > allowed:
            raise InterruptedError('cancelled')

    with pytest.raises(InterruptedError):
        thin_tubes(staircase((6, 8, 10), (41, 30, 22)), SPACING, check=check)
    assert len(calls) == allowed + 1
