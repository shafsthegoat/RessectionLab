import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from resectionlab.vascular_rooting import (
    ARTERY, SCHEMES, UNCLASSIFIED, VEIN, RootingPolicy, image_modality, root_vessels,
)

SPACING = (0.25, 0.25, 0.25)
SHAPE = (120, 64, 64)
AXIS = (8.0, 8.0)


def tube(start, end, radius, shape=SHAPE, spacing=SPACING):
    """Voxels whose centres lie within radius (mm) of a segment (mm)."""
    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing='ij'), -1)
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    length2 = float(axis @ axis)
    t = np.clip(((grid - start) @ axis) / length2, 0, 1) if length2 else np.zeros(shape)
    return np.linalg.norm(grid - (start + t[..., None] * axis), axis=-1) <= radius


def along(x0, x1, radius, y=AXIS[0], z=AXIS[1]):
    return tube((x0, y, z), (x1, y, z), radius)


def scan(mask, gap=None, gap_value=300.0, seed=0):
    rng = np.random.default_rng(seed)
    signal = 50 + rng.normal(0, 5, mask.shape).astype(np.float32)
    signal[mask] = 400
    if gap is not None:
        signal[gap] = gap_value
    return signal


def partition_holds(result, mask):
    assert np.array_equal(result.shown | result.review | result.lod_hidden, mask)
    assert not (result.shown & result.review).any()
    assert not (result.shown & result.lod_hidden).any()
    assert not (result.review & result.lod_hidden).any()
    assert not (result.bridge & mask).any()
    assert result.receipt['partitionExact'] is True
    labels, count = ndi.label(result.display, np.ones((3, 3, 3)))
    assert result.receipt['display']['floatingComponents'] == 0
    return labels, count


def test_largest_component_fallback_partitions_exactly_and_says_so():
    trunk = along(1, 20, 1.0)
    loose = along(26, 29, 0.5)
    mask = trunk | loose
    result = root_vessels(mask, spacing=SPACING)
    _, count = partition_holds(result, mask)
    assert count == 1
    assert result.shown[trunk].all() and result.review[loose].all()
    assert result.receipt['roots']['source'] == 'largest-component'
    assert 'largest component' in result.receipt['roots']['reason']
    assert result.receipt['review']['components'] == 1
    assert result.receipt['bridged']['support'] == 'none'
    json.dumps(result.receipt)


def test_named_roots_choose_the_tree_even_when_smaller():
    root = along(1, 6, 1.0)
    unnamed = along(10, 29, 1.0)
    mask = root | unnamed
    named = np.where(root, 4, np.where(unnamed, 17, 0)).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-ct', modality='CTA', layer_kind='artery')
    partition_holds(result, mask)
    assert result.shown[root].all() and result.review[unnamed].all()
    assert result.receipt['roots']['source'] == 'named'
    assert result.receipt['roots']['labels'] == [{'label': 4, 'name': 'R-ICA'}]
    rows = {row['name']: row for row in result.receipt['namedLabels']}
    assert rows['R-ICA']['status'] == 'shown'
    assert rows['R-M2']['status'] == 'review' and rows['R-M2']['reviewComponents'] == 1


def test_named_labels_without_a_root_fall_back_and_report_it():
    mask = along(1, 20, 1.0)
    named = np.where(mask, 17, 0).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-ct', modality='CT')
    assert result.receipt['roots']['source'] == 'largest-component'
    assert 'No root label' in result.receipt['roots']['reason']
    assert result.shown[mask].all()


def bridge_case(gap_length=1.0):
    """Two collinear 1.5 mm tubes whose rounded ends are gap_length mm apart."""
    tree = along(1, 12, 0.75)
    fragment = along(13.5 + gap_length, 25, 0.75)
    gap = along(12, 13.5 + gap_length, 0.75) & ~tree & ~fragment
    return tree, fragment, gap


def test_signal_supported_gap_is_bridged_with_provenance():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    named = np.where(tree, 6, np.where(fragment, 19, 0)).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap), named=named,
                          scheme='topbrain-v1-ct', modality='CTA', layer_kind='artery')
    _, count = partition_holds(result, mask)
    assert count == 1
    assert result.shown[fragment].all() and not result.review.any()
    assert result.bridge.any() and (result.bridge <= gap).all()
    receipt = result.receipt['bridged']
    assert receipt['components'] == 1 and receipt['support'] == 'signal'
    record = receipt['bridges'][0]
    assert record['gapMm'] <= 3.0 and record['minimumContrastFraction'] >= 0.5 and record['namedLabels'] == ['L-M2']
    rows = {row['name']: row for row in result.receipt['namedLabels']}
    assert rows['L-M2']['status'] == 'bridged' and rows['L-M2']['bridgedComponents'] == 1
    assert (result.classes[result.bridge] == ARTERY).all()


@pytest.mark.parametrize('gap_value,reason', [(60.0, 'no-supported-path'), (2000.0, 'no-supported-path')])
def test_unsupported_or_bone_bright_gap_is_refused(gap_value, reason):
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap, gap_value))
    partition_holds(result, mask)
    assert result.review[fragment].all() and not result.bridge.any()
    assert result.receipt['review']['refused'] == {reason: 1}


def test_long_gap_is_refused_even_when_supported():
    tree, fragment, gap = bridge_case(gap_length=4.0)
    mask = tree | fragment
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap))
    partition_holds(result, mask)
    assert result.review[fragment].all() and not result.bridge.any()
    assert result.receipt['review']['refused'] == {'too-far': 1}


def test_probability_alone_can_support_a_bridge():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    for value, bridged in ((0.4, True), (0.1, False)):
        probability = np.where(mask, 0.9, np.where(gap, value, 0.0)).astype(np.float32)
        result = root_vessels(mask, spacing=SPACING, probability=probability)
        partition_holds(result, mask)
        assert bool(result.shown[fragment].all()) is bridged
        assert result.receipt['bridged']['support'] == 'probability'


def test_small_specks_are_never_bridged():
    tree = along(1, 12, 0.75)
    speck = np.zeros(SHAPE, bool)
    speck[57, 32, 32] = True
    mask = tree | speck
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, along(12, 14.5, 0.3) & ~mask))
    partition_holds(result, mask)
    assert result.review[speck].all()
    assert result.receipt['review']['refused'] == {'too-small': 1}


def test_chains_attach_one_bounded_gap_per_round():
    tree = along(1, 8, 0.75)
    first = along(10, 14, 0.75)
    second = along(16, 22, 0.75)
    mask = tree | first | second
    gap = (along(8, 10, 0.75) | along(14, 16, 0.75)) & ~mask
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap))
    _, count = partition_holds(result, mask)
    assert count == 1 and result.receipt['bridged']['components'] == 2
    assert result.receipt['bridged']['rounds'] == 2


def lod_case():
    trunk = along(1, 10, 1.0)
    connector = along(10, 16, 0.25)
    distal = along(16, 26, 1.0)
    branch = tube((5, 8, 8), (5, 15, 8), 0.25)
    return trunk, connector, distal, branch


def test_level_of_detail_prunes_thin_leaves_but_never_disconnects():
    trunk, connector, distal, branch = lod_case()
    mask = trunk | connector | distal | branch
    result = root_vessels(mask, spacing=SPACING)
    _, count = partition_holds(result, mask)
    assert count == 1
    assert result.shown[trunk | connector | distal].all()
    # Calibre is the widest within 1.5 mm, so the branch base beside the trunk reads wide.
    tip = branch & tube((5, 12, 8), (5, 15, 8), 0.3)
    assert result.lod_hidden[tip].all()
    assert result.receipt['lod']['hiddenComponents'] >= 1 and not result.receipt['lod']['reverted']


def test_named_vessels_are_exempt_from_level_of_detail():
    trunk, connector, distal, branch = lod_case()
    mask = trunk | connector | distal | branch
    named = np.where(branch & ~trunk, 31, np.where(trunk, 4, 0)).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-mr', modality='TOF')
    partition_holds(result, mask)
    assert not result.lod_hidden.any()
    rows = {row['name']: row for row in result.receipt['namedLabels']}
    assert rows['R-AChA']['status'] == 'shown'


def test_local_tier_shows_mid_calibre_branches_only_near_tumour_or_corridor():
    trunk = along(1, 26, 1.0)
    branch = tube((14, 8, 8), (14, 15, 8), 0.375)
    mask = trunk | branch
    tip = branch & tube((14, 12.5, 8), (14, 15, 8), 0.4)
    policy = RootingPolicy(local_margin_mm=2.0)
    far = root_vessels(mask, spacing=SPACING, policy=policy)
    assert far.lod_hidden[tip].all()
    tumour = tube((14, 15.2, 8), (14, 15.2, 8), 0.5)
    near = root_vessels(mask, spacing=SPACING, policy=policy, tumour=tumour)
    partition_holds(near, mask)
    assert near.shown[tip & tube((14, 13.5, 8), (14, 15, 8), 0.4)].all()
    affine = np.diag([*SPACING, 1.0])
    corridor = {'entry': [14, 30, 8], 'target': [14, 16, 8], 'radiusMm': 1.0}
    through = root_vessels(mask, affine=affine, policy=policy, corridors=[corridor])
    assert through.shown[tip & tube((14, 13.5, 8), (14, 15, 8), 0.4)].all()
    assert through.receipt['lod']['corridors'] == 1
    # Thinner than the local tier is hidden even beside the tumour.
    thin = trunk | tube((14, 8, 8), (14, 15, 8), 0.25)
    beside = root_vessels(thin, spacing=SPACING, policy=policy, tumour=tumour)
    assert beside.lod_hidden[tube((14, 13.5, 8), (14, 15, 8), 0.3) & thin].all()


def test_ct_labels_are_never_applied_to_mr_and_the_reverse():
    mask = along(1, 10, 1.0)
    named = np.where(mask, 4, 0).astype(np.uint8)
    with pytest.raises(ValueError, match='cannot be applied'):
        root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-ct', modality='TOF')
    with pytest.raises(ValueError, match='cannot be applied'):
        root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-mr', modality='CTA')
    with pytest.raises(ValueError, match='outside topbrain-v1-ct'):
        root_vessels(mask, spacing=SPACING, named=np.where(mask, 41, 0).astype(np.uint8),
                     scheme='topbrain-v1-ct', modality='CT')
    with pytest.raises(ValueError, match='label scheme'):
        root_vessels(mask, spacing=SPACING, named=named)
    assert image_modality('CTA') == 'ct' and image_modality('TOF') == 'mr' and image_modality('MRA') == 'mr'


def test_root_sets_match_the_topbrain_maps():
    ct, mr = SCHEMES['topbrain-v1-ct'], SCHEMES['topbrain-v1-mr']
    assert [ct.names[v] for v in ct.roots] == ['BA', 'R-ICA', 'L-ICA', 'R-VA', 'L-VA', 'VoG', 'StS', 'SSS']
    assert [mr.names[v] for v in mr.roots] == ['BA', 'R-ICA', 'L-ICA', 'R-VA', 'L-VA']
    assert ct.veins == (35, 36, 37, 38, 39, 40) and mr.veins == ()
    assert mr.names[42] == 'L-MMA' and 41 not in ct.names


def test_classes_follow_named_labels_then_layer_kind_and_bridges_their_parent():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    named = np.where(tree, 36, 0).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap), named=named,
                          scheme='topbrain-v1-ct', modality='CT', layer_kind='vessels')
    assert (result.classes[tree] == VEIN).all()
    assert (result.classes[fragment] == UNCLASSIFIED).all()
    assert (result.classes[result.bridge] == VEIN).all()
    assert 'artery colour' in result.receipt['classes']['note']
    assert not result.classes[~result.display].any()


def test_random_masks_partition_exactly():
    rng = np.random.default_rng(3)
    field = ndi.gaussian_filter(rng.normal(size=(48, 48, 48)), 1.5)
    mask = field > np.percentile(field, 88)
    signal = np.where(mask, 300, 40).astype(np.float32) + rng.normal(0, 20, mask.shape).astype(np.float32)
    result = root_vessels(mask, spacing=(0.4, 0.4, 0.6), signal=signal)
    partition_holds(result, mask)
    assert result.receipt['input']['components'] > 1


def test_empty_mask_returns_empty_outputs():
    result = root_vessels(np.zeros((4, 4, 4), bool), spacing=(1, 1, 1))
    assert not result.display.any() and result.receipt['display']['floatingComponents'] == 0


def test_memory_mapped_support_fields_are_read_whole_and_cropped_exactly(tmp_path):
    from resectionlab.vascular_rooting import _load_crop
    values = np.random.default_rng(0).random((30, 20, 10))
    np.save(tmp_path / 'signal.npy', values)
    mapped = np.load(tmp_path / 'signal.npy', mmap_mode='r')
    crop = (slice(3, 27), slice(2, 19), slice(1, 9))
    loaded = _load_crop(mapped, crop, block_bytes=4000)
    assert loaded.dtype == np.float16 and np.array_equal(loaded, values[crop].astype(np.float16))
    resident = values.astype(np.float32)
    assert np.shares_memory(_load_crop(resident, crop), resident)


# Review fixes ---------------------------------------------------------------

def test_bridges_never_join_a_named_vein_to_an_artery():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    signal = scan(mask, gap)
    for label, bridged in ((37, False), (17, True), (0, True)):
        named = np.where(tree, 4, np.where(fragment, label, 0)).astype(np.uint8)
        result = root_vessels(mask, spacing=SPACING, signal=signal, named=named,
                              scheme='topbrain-v1-ct', modality='CTA', layer_kind='vessels')
        partition_holds(result, mask)
        assert bool(result.shown[fragment].all()) is bridged, label
        if not bridged:
            assert result.receipt['review']['refused'] == {'no-same-class-tree': 1}
            rows = {row['name']: row for row in result.receipt['namedLabels']}
            assert rows['ICVs']['status'] == 'review'
        else:
            assert (result.classes[result.bridge] == ARTERY).all()


def calibre_case(spacing, diameter, seed):
    """A 2.5 mm trunk and a terminal branch whose centreline sits off the lattice."""
    rng = np.random.default_rng(seed)
    shape = tuple(int(np.ceil(n / spacing)) for n in (24, 16, 12))
    offset = rng.uniform(0, spacing, 3)
    trunk = tube(np.array((1, 4, 6)) + offset, np.array((23, 4, 6)) + offset, 1.25, shape, (spacing,) * 3)
    tip = np.array((14.0, 14.0, 6.7)) + offset
    branch = tube(np.array((11, 4, 6)) + offset, tip, diameter / 2, shape, (spacing,) * 3)
    return trunk, branch & ~trunk, tip


@pytest.mark.parametrize('spacing', [0.3, 0.45, 0.6])
@pytest.mark.parametrize('diameter', [1.0, 1.1, 1.3])
def test_vessels_of_one_millimetre_and_up_are_always_shown_off_lattice(spacing, diameter):
    for seed in range(3):
        trunk, branch, _ = calibre_case(spacing, diameter, seed)
        mask = trunk | branch
        result = root_vessels(mask, spacing=(spacing,) * 3)
        partition_holds(result, mask)
        assert result.shown[branch].all(), (spacing, diameter, seed)


@pytest.mark.parametrize('spacing', [0.3, 0.45])
def test_six_tenths_of_a_millimetre_is_still_thin(spacing):
    trunk, branch, tip = calibre_case(spacing, 0.6, 0)
    mask = trunk | branch
    result = root_vessels(mask, spacing=(spacing,) * 3)
    partition_holds(result, mask)
    near_tip = branch & tube(tip - (0, 2.5, 0), tip, 0.6, mask.shape, (spacing,) * 3)
    assert near_tip.any() and result.lod_hidden[near_tip].all()


def test_bridges_are_widened_to_the_smaller_end_radius():
    tree, fragment, gap = bridge_case(gap_length=1.5)
    mask = tree | fragment
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap))
    partition_holds(result, mask)
    record = result.receipt['bridged']['bridges'][0]
    assert 0.4 <= record['radiusMm'] <= 1.0
    assert result.bridge.sum() > 3 * record['gapMm'] / SPACING[0]
    assert (result.bridge <= gap).all()
    assert ndi.label(result.bridge, np.ones((3, 3, 3)))[1] == 1


def test_probability_never_carries_a_bridge_through_bone():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    probability = np.where(mask | gap, 0.9, 0.0).astype(np.float32)
    bright = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap, 2000.0), probability=probability)
    assert bright.review[fragment].all() and not bright.bridge.any()
    plain = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap), probability=probability)
    assert plain.shown[fragment].all()
    boned = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap), probability=probability, bone=gap)
    assert boned.review[fragment].all() and not boned.bridge.any()


def test_ct_ceiling_in_hu_applies_to_ct_only():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    signal = scan(mask, gap, 1000.0)
    signal[mask] = 1200.0
    assert root_vessels(mask, spacing=SPACING, signal=signal, modality='TOF').shown[fragment].all()
    ct = root_vessels(mask, spacing=SPACING, signal=signal, modality='CTA')
    assert ct.review[fragment].all() and ct.receipt['review']['refused'] == {'no-supported-path': 1}


def test_bridge_windows_cover_only_the_part_of_a_fragment_near_the_tree():
    tree = along(1, 6, 0.75)
    fragment = along(8.5, 28, 0.75)
    gap = along(6, 8.5, 0.75) & ~tree & ~fragment
    mask = tree | fragment
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap))
    partition_holds(result, mask)
    record = result.receipt['bridged']['bridges'][0]
    span = ndi.find_objects(fragment.astype(int))[0][0]
    assert record['windowShape'][0] < (span.stop - span.start)


def entry_case():
    """Two vessel trees entering from the inferior face (z index 0) and two other pieces."""
    shape = (64, 64, 80)
    left = tube((4, 4, 0), (4, 4, 18), 1.0, shape) | tube((4, 4, 18), (12, 4, 18), 1.0, shape)
    right = tube((12, 12, 0), (12, 12, 12), 1.1, shape)
    thin = tube((8, 14, 0), (8, 14, 12), 0.2, shape) | tube((14, 7, 0), (14, 7, 12), 0.8, shape)
    floating = tube((4, 12, 10), (4, 12, 16), 0.8, shape)
    return shape, left, right, thin, floating


def test_without_named_labels_vessels_entering_from_below_are_roots():
    shape, left, right, thin, floating = entry_case()
    mask = left | right | thin | floating
    frame = np.diag([*SPACING, 1.0])
    result = root_vessels(mask, affine=frame)
    partition_holds(result, mask)
    assert result.shown[left].all() and result.shown[right].all()
    assert result.review[thin].all() and result.review[floating].all()
    roots = result.receipt['roots']
    accepted = roots['acceptedRoots']
    assert [row['reason'] for row in accepted] == ['largest component', 'enters through the inferior face']
    assert accepted[1]['calibreMm'] >= 2.0 and accepted[1]['extentMm'] >= 10
    # Narrower components entering from below (a 1.6 mm and a 0.4 mm branch) stay in review.
    assert roots['refusedEntryCount'] == 2 and {row['reason'] for row in roots['refusedEntries']} == {'too narrow'}
    assert all(row['calibreMm'] < 2.0 for row in roots['refusedEntries'])
    # Superior is the low z index when the frame flips z: the right tree no longer enters.
    flipped = root_vessels(mask, affine=np.diag([SPACING[0], SPACING[1], -SPACING[2], 1.0]))
    assert flipped.review[right].all()
    unframed = root_vessels(mask, spacing=SPACING)
    assert unframed.review[right].all()
    assert 'inferior face is unknown' in unframed.receipt['roots']['acceptedRoots'][0]['note']


def test_mr_external_carotid_branches_stay_in_review_and_are_listed():
    shape, left, right, thin, floating = entry_case()
    mask = left | floating
    named = np.where(left, 4, np.where(floating, 35, 0)).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-mr', modality='TOF')
    assert result.review[floating].all()
    assert result.receipt['roots']['externalCarotidInReview'] == ['R-ECA']


def test_memory_mapped_fields_passed_to_root_vessels_stay_memory_maps(tmp_path, monkeypatch):
    import resectionlab.vascular_rooting as rooting
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    np.save(tmp_path / 'signal.npy', scan(mask, gap))
    mapped = np.load(tmp_path / 'signal.npy', mmap_mode='r')
    seen = []
    original = rooting._load_crop
    monkeypatch.setattr(rooting, '_load_crop', lambda array, crop, **kw: seen.append(type(array)) or original(array, crop, **kw))
    result = root_vessels(mask, spacing=SPACING, signal=mapped)
    assert result.shown[fragment].all() and seen and all(kind is np.memmap for kind in seen)


@pytest.mark.parametrize('gap_value', [300.0, 60.0])
def test_bridge_decisions_and_contrast_do_not_depend_on_the_search_window(gap_value, monkeypatch):
    import resectionlab.vascular_rooting as rooting
    tree = along(1, 6, 0.75) | tube((3, 8, 8), (3, 14, 8), 0.5)
    fragment = along(8.5, 28, 0.75)
    gap = along(6, 8.5, 0.75) & ~tree & ~fragment
    mask = tree | fragment
    signal = scan(mask, gap, gap_value)
    signal[tube((20, 4, 8), (28, 4, 8), 1.0) & ~mask] = 250.0  # Bright context the larger window would add.
    receipts = []
    for extra in (0, 24):
        monkeypatch.setattr(rooting, '_WINDOW_EXTRA', extra)
        result = root_vessels(mask, spacing=SPACING, signal=signal)
        partition_holds(result, mask)
        receipts.append(result.receipt)
    small, large = receipts
    assert small['review']['refused'] == large['review']['refused']
    assert small['bridged']['components'] == large['bridged']['components'] == (1 if gap_value == 300.0 else 0)
    fields = ('gapMm', 'lumen', 'background', 'minimumContrastFraction', 'statisticsCentreIjk', 'bridgeVoxels')
    for first, second in zip(small['bridged']['bridges'], large['bridged']['bridges']):
        assert first['windowShape'] != second['windowShape']
        assert {k: first[k] for k in fields} == {k: second[k] for k in fields}


def corner_case():
    """A trunk and a segment that touch only at one voxel corner (one 26-connected component)."""
    shape = (40, 24, 24)
    trunk = np.zeros(shape, bool); trunk[2:20, 10:14, 10:14] = True
    segment = np.zeros(shape, bool); segment[20:32, 14:18, 14:18] = True
    return trunk, segment


def test_corner_only_joins_get_a_declared_fill_so_the_drawn_surface_is_one_piece():
    trunk, segment = corner_case()
    mask = trunk | segment
    signal = scan(mask)
    result = root_vessels(mask, spacing=(.5, .5, .5), signal=signal)
    partition_holds(result, mask)
    assert result.shown[segment].all()
    fill = result.receipt['connectivity']
    assert fill['joins'] == 1 and fill['fillVoxels'] == 2 and result.bridge.sum() == 2
    assert ndi.label(result.display)[1] == 1  # Face-connected.
    assert result.receipt['display']['connectivity'] == 'face (6-connected)'


def test_a_vetoed_fill_sends_the_far_side_to_review():
    trunk, segment = corner_case()
    mask = trunk | segment
    bone = np.zeros(mask.shape, bool); bone[18:22, 12:16, 12:16] = True
    bone &= ~mask
    result = root_vessels(mask, spacing=(.5, .5, .5), signal=scan(mask), bone=bone)
    partition_holds(result, mask)
    assert result.review[segment].all() and result.shown[trunk].all() and not result.bridge.any()
    fill = result.receipt['connectivity']
    assert fill['droppedComponents'] == 1 and fill['vetoedCandidates'] >= 1 and fill['droppedSourceVoxels'] == int(segment.sum())


def test_bridges_are_face_connected_paths():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    result = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap))
    assert result.shown[fragment].all() and ndi.label(result.display)[1] == 1


def test_ct_voxels_in_air_are_review_and_root_join_or_bridge_nothing():
    # A named trunk in tissue, a blob outside the head that a model gave the same root label, and a piece
    # joined to the trunk only through voxels in air.
    trunk, blob, far = along(1, 10, 1.0), along(22, 27, 1.0), along(13, 18, 1.0)
    link = along(10, 13, 0.6) & ~trunk & ~far
    mask = trunk | blob | far | link
    signal = scan(mask)
    signal[blob] = -950
    signal[link] = -700
    named = np.where(trunk | blob, 36, 0).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, signal=signal, named=named, scheme='topbrain-v1-ct', modality='CT')
    partition_holds(result, mask)
    assert result.shown[trunk].all() and not result.display[blob | link | far].any()
    assert result.review[blob | link | far].all()
    receipt = result.receipt
    assert receipt['air'] == {'floorHu': -300.0, 'voxels': int((blob | link).sum()),
                              'volumeMm3': round(float((blob | link).sum()) * 0.25 ** 3, 4), 'rule': receipt['air']['rule']}
    assert receipt['input']['voxels'] == int(mask.sum()) and receipt['partitionExact']
    assert receipt['review']['voxels'] == int((blob | link | far).sum()) and receipt['review']['components'] == 1
    assert receipt['display']['floatingComponents'] == 0 and np.array_equal(result.air, blob | link)
    row, = receipt['namedLabels']
    assert (row['voxels'], row['shown'], row['review'], row['inAir'], row['status']) == (
        int((trunk | blob).sum()), int(trunk.sum()), int(blob.sum()), int(blob.sum()), 'partly-review')
    # Without the acquired CT, or on MR, nothing is known about air and the blob stays a root.
    for options in ({'modality': 'CT'}, {'modality': 'CT', 'probability': np.where(mask, 0.9, 0).astype(np.float32)}):
        kept = root_vessels(mask, spacing=SPACING, named=named, scheme='topbrain-v1-ct', **options)
        assert kept.shown[blob].all() and 'air' not in kept.receipt
    mr = root_vessels(mask, spacing=SPACING, signal=signal, modality='TOF')
    assert 'air' not in mr.receipt and mr.shown[mask].sum() > trunk.sum()


def test_a_ct_bridge_is_not_carried_through_air_by_probability():
    tree, fragment, gap = bridge_case()
    mask = tree | fragment
    probability = np.where(mask | gap, 0.9, 0).astype(np.float32)
    joined = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap), probability=probability, modality='CT')
    assert joined.shown[fragment].all() and joined.receipt['air']['voxels'] == 0
    refused = root_vessels(mask, spacing=SPACING, signal=scan(mask, gap, gap_value=-900.0), probability=probability,
                           modality='CT')
    assert refused.review[fragment].all() and not refused.bridge.any()
    partition_holds(refused, mask)


def test_a_ct_mask_wholly_in_air_shows_nothing_and_a_label_wholly_in_air_is_listed():
    mask = along(1, 10, 1.0)
    signal = np.full(mask.shape, -1000.0, np.float32)
    named = np.where(mask, 40, 0).astype(np.uint8)
    result = root_vessels(mask, spacing=SPACING, signal=signal, named=named, scheme='topbrain-v1-ct', modality='CT')
    assert not result.display.any() and np.array_equal(result.review, mask)
    assert result.receipt['review']['voxels'] == int(mask.sum()) == result.receipt['input']['voxels']
    assert 'lies in air' in result.receipt['review']['reason'] and 'lies in air' in result.receipt['roots']['reason']
    assert [(row['name'], row['status'], row['inAir']) for row in result.receipt['namedLabels']] == [
        ('SSS', 'review', int(mask.sum()))]


def test_a_ct_corner_join_is_not_filled_through_air():
    trunk, segment = corner_case()
    mask = trunk | segment
    signal = scan(mask)
    between = np.zeros(mask.shape, bool); between[18:22, 12:16, 12:16] = True
    signal[between & ~mask] = -900
    filled = root_vessels(mask, spacing=(.5, .5, .5), signal=signal)
    assert filled.shown[segment].all() and filled.bridge.sum() == 2
    result = root_vessels(mask, spacing=(.5, .5, .5), signal=signal, modality='CT')
    partition_holds(result, mask)
    assert result.review[segment].all() and result.shown[trunk].all() and not result.bridge.any()
    assert result.receipt['connectivity']['droppedComponents'] == 1 and result.receipt['air']['voxels'] == 0



@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int16])
def test_the_air_floor_is_one_voxel_set_whatever_the_scan_dtype(dtype, tmp_path):
    # Voxels a hair below the floor in float64 are exactly on it in float32: they must be either air or
    # bridgeable, never both (that broke the partition).
    tree, fragment, gap = bridge_case()
    mask = tree | fragment | gap
    signal = scan(mask).astype(np.float64)
    signal[gap] = -300 - 1e-9 if dtype is np.float64 else -301
    signal = signal.astype(dtype)
    probability = np.where(mask, 0.9, 0).astype(np.float32)
    result = root_vessels(mask, spacing=SPACING, signal=signal, probability=probability, modality='CT')
    partition_holds(result, mask)
    assert not (result.bridge & mask).any()
    in_air = signal.astype(np.float32) < -300
    assert result.receipt['air']['voxels'] == int((mask & in_air).sum())
    # A memory-mapped scan gives the same answer as the array in memory.
    path = tmp_path / 'scan.dat'
    mapped = np.memmap(path, dtype=signal.dtype, mode='w+', shape=signal.shape)
    mapped[:] = signal
    mapped.flush()
    again = root_vessels(mask, spacing=SPACING, signal=np.memmap(path, dtype=signal.dtype, mode='r', shape=signal.shape),
                         probability=probability, modality='CT')
    assert np.array_equal(again.shown, result.shown) and np.array_equal(again.review, result.review)


def test_a_model_addition_in_air_does_not_extend_its_tree():
    tree = along(1, 10, 1.0)
    branch = along(10, 16, 0.75) & ~tree          # touches the tree, in tissue
    airy = tube((5, 8, 9), (5, 8, 14), 0.75) & ~tree   # touches the tree, in air
    mask = tree | branch | airy
    signal = scan(mask)
    signal[airy] = -900
    result = root_vessels(mask, spacing=SPACING, signal=signal, root_mask=tree, modality='CT')
    partition_holds(result, mask)
    assert result.shown[branch].all() and result.review[airy].all() and np.array_equal(result.air, airy)
    assert result.receipt['review']['components'] == 0 and result.receipt['review']['voxels'] == int(airy.sum())
