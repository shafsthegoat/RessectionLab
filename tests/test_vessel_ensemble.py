import numpy as np
import pytest

from resectionlab.vascular_parity import betti

from resectionlab.vessel_ensemble import (
    ConnectedAddition, SoftSegmentation, average, combine, named_label_map, require_same_grid, vessel_groups,
    vessel_probability,
)

AFFINE = np.diag([0.5, 0.5, 0.8, 1.0])
SHAPE = (24, 20, 16)


def softmax_member(modality='ct', artery=0.9, vein=0.8, labels=(0, 1, 35), affine=AFFINE):
    probabilities = np.zeros((3, *SHAPE), np.float32)
    probabilities[0] = 1
    probabilities[:, 4:20, 10, 8] = np.array([[1 - artery], [artery], [0]])
    probabilities[:, 12, 2:8, 8] = np.array([[1 - vein], [0], [vein]])
    return SoftSegmentation(modality, labels, probabilities, affine)


def tree():
    mask = np.zeros(SHAPE, bool)
    mask[4:20, 10, 8] = True
    return mask


def test_average_keeps_soft_probabilities_and_derives_groups_and_names():
    mean = average([softmax_member(artery=0.9), softmax_member(artery=0.3)])
    assert mean.probabilities[1, 6, 10, 8] == pytest.approx(0.6)
    assert mean.probabilities.sum(axis=0) == pytest.approx(np.ones(SHAPE))
    groups = vessel_groups(mean, vein_labels=[35])
    assert groups['vessel'][6, 10, 8] == pytest.approx(0.6)
    assert groups['artery'][6, 10, 8] == pytest.approx(0.6) and groups['vein'][6, 10, 8] == 0
    assert groups['vein'][12, 3, 8] == pytest.approx(0.8) and groups['artery'][12, 3, 8] == pytest.approx(0)
    named = named_label_map(mean)
    assert named[6, 10, 8] == 1 and named[12, 3, 8] == 35 and named[0, 0, 0] == 0
    assert named_label_map(average([softmax_member(artery=0.3)]))[6, 10, 8] == 0


def test_average_streams_channels_into_a_memory_mapped_output(tmp_path):
    members = [softmax_member(artery=0.9), softmax_member(artery=0.5)]
    out = np.lib.format.open_memmap(tmp_path / 'mean.npy', 'w+', np.float16, members[0].probabilities.shape)
    mean = average(members, out=out)
    assert mean.probabilities is out
    assert float(np.load(tmp_path / 'mean.npy', mmap_mode='r')[1, 6, 10, 8]) == pytest.approx(0.7, abs=1e-3)
    with pytest.raises(ValueError, match='output'):
        average(members, out=np.zeros((2, *SHAPE), np.float32))


class ChannelReader:
    """Array-like that hands out one channel at a time, like a volume streamed from disk."""

    def __init__(self, values):
        self.values, self.shape, self.dtype, self.ndim, self.reads = values, values.shape, values.dtype, 4, 0

    def __getitem__(self, channel):
        self.reads += 1
        return self.values[channel]

    def __iter__(self):
        return (self[channel] for channel in range(self.shape[0]))


class ChannelWriter:
    def __init__(self, shape):
        self.shape, self.values, self.next = shape, np.zeros(shape, np.float16), 0

    def __setitem__(self, channel, values):
        assert channel == self.next
        self.values[channel], self.next = values, self.next + 1

    def close(self):
        return self.values


def test_average_reads_and_writes_one_channel_at_a_time():
    members = [SoftSegmentation('mr', (0, 1, 35), ChannelReader(softmax_member(artery=a).probabilities), AFFINE)
               for a in (0.9, 0.3)]
    mean = average(members, out=ChannelWriter(members[0].probabilities.shape))
    assert isinstance(mean.probabilities, np.ndarray)
    assert float(mean.probabilities[1, 6, 10, 8]) == pytest.approx(0.6, abs=1e-3)
    assert named_label_map(mean)[6, 10, 8] == 1
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        SoftSegmentation('mr', (0, 1, 35), ChannelReader(softmax_member().probabilities * 3), AFFINE)


def test_members_must_share_modality_labels_and_grid():
    with pytest.raises(ValueError, match='Modality mismatch'):
        average([softmax_member('ct'), softmax_member('mr')])
    with pytest.raises(ValueError, match='label'):
        average([softmax_member(), softmax_member(labels=(0, 2, 35))])
    with pytest.raises(ValueError, match='affine'):
        average([softmax_member(), softmax_member(affine=np.diag([0.6, 0.5, 0.8, 1.0]))])
    with pytest.raises(ValueError):
        SoftSegmentation('pet', (0, 1, 35), softmax_member().probabilities, AFFINE)
    with pytest.raises(ValueError):
        SoftSegmentation('ct', (1, 0, 35), softmax_member().probabilities, AFFINE)
    with pytest.raises(ValueError, match='Vein labels'):
        vessel_groups(softmax_member(), vein_labels=[40])
    with pytest.raises(ValueError, match='shape'):
        require_same_grid(SHAPE, AFFINE, SHAPE[:2] + (5,), AFFINE)


def test_connected_additions_are_kept_with_their_scores_and_islands_rejected():
    score = np.zeros(SHAPE, np.float32)
    score[4:20, 10, 8] = 0.95
    score[12, 11:18, 8] = 0.7
    score[12, 18, 8] = 0.55
    score[2:5, 2:5, 2:5] = 0.99
    added, mask = ConnectedAddition(low=0.5, high=0.6).apply(tree(), 'ct', AFFINE, score, 'ct', AFFINE)
    assert mask[12, 11:19, 8].all()
    assert added[12, 14, 8] == pytest.approx(0.7) and added[12, 18, 8] == pytest.approx(0.55)
    assert not mask[2:5, 2:5, 2:5].any() and not mask[tree()].any()
    assert betti(tree() | mask)['b0'] == 1


def test_only_touching_components_are_added_so_no_new_component_appears():
    score = np.zeros(SHAPE, np.float32)
    score[12, 12:18, 8] = 0.8
    score[12, 11, 7] = 0.8
    touching = ConnectedAddition().apply(tree(), 'mr', AFFINE, score, 'mr', AFFINE)[1]
    assert touching[12, 12:18, 8].all() and betti(tree() | touching)['b0'] == 1
    score[12, 11, 7] = 0
    assert not ConnectedAddition().apply(tree(), 'mr', AFFINE, score, 'mr', AFFINE)[1].any()


def test_high_threshold_bounds_what_reaches_the_tree():
    weak = np.zeros(SHAPE, np.float32)
    weak[12, 11:15, 8] = 0.55
    assert not ConnectedAddition(low=0.5, high=0.9).apply(tree(), 'mr', AFFINE, weak, 'mr', AFFINE)[1].any()


def test_additions_refuse_other_modalities_grids_and_invalid_scores():
    score = np.zeros(SHAPE, np.float32)
    rule = ConnectedAddition()
    with pytest.raises(ValueError, match='Modality mismatch'):
        rule.apply(tree(), 'mr', AFFINE, score, 'ct', AFFINE)
    with pytest.raises(ValueError, match='shape'):
        rule.apply(tree(), 'ct', AFFINE, score[:, :, :8], 'ct', AFFINE)
    with pytest.raises(ValueError, match='affine'):
        rule.apply(tree(), 'ct', AFFINE, score, 'ct', np.diag([0.5, 0.5, 0.5, 1.0]))
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        rule.apply(tree(), 'ct', AFFINE, score + 2, 'ct', AFFINE)
    with pytest.raises(ValueError, match='floating'):
        rule.apply(tree(), 'ct', AFFINE, score.astype(np.uint8), 'ct', AFFINE)
    with pytest.raises(ValueError):
        ConnectedAddition(low=0.7, high=0.5)
    with pytest.raises(TypeError):
        ConnectedAddition(gap_voxels=1)


def three_members():
    """Three CT members: all agree on the artery, one alone sees an extra branch, two see a vein segment."""
    members = []
    for index in range(3):
        probabilities = np.zeros((3, *SHAPE), np.float32)
        probabilities[0] = 1
        probabilities[:, 4:20, 10, 8] = np.array([[0.1], [0.9], [0]])
        if index == 0:
            probabilities[:, 12, 11:16, 8] = np.array([[0.2], [0.8], [0]])
            probabilities[:, 2, 2, 2] = np.array([0.1, 0.9, 0])
        if index < 2:
            probabilities[:, 12, 2:8, 8] = np.array([[0.2], [0], [0.8]])
        members.append(SoftSegmentation('ct', (0, 1, 35), probabilities, AFFINE))
    return members


def test_combiners_differ_only_in_how_vessel_probability_is_pooled():
    members = three_members()
    mean = vessel_probability(members, 'mean')
    assert mean[12, 13, 8] == pytest.approx(0.8 / 3) and mean[12, 4, 8] == pytest.approx(1.6 / 3)
    assert vessel_probability(members, 'maximum')[12, 13, 8] == pytest.approx(0.8)
    median = vessel_probability(members, 'median')
    assert median[12, 13, 8] == 0 and median[12, 4, 8] == pytest.approx(0.8)
    weighted = vessel_probability(members, 'weighted', weights=[2, 1, 1])
    assert weighted[12, 13, 8] == pytest.approx(0.4)
    base = vessel_probability(members, 'base', base=2)
    assert base[12, 13, 8] == pytest.approx(0.8) and base[12, 4, 8] == 0
    assert base[2, 2, 2] == 0 and vessel_probability(members, 'maximum')[2, 2, 2] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        vessel_probability(members, 'weighted', weights=[1, 0, 1])
    with pytest.raises(ValueError):
        vessel_probability(members, 'base', base=3)
    with pytest.raises(ValueError):
        vessel_probability(members, 'vote')


def test_combined_class_probabilities_sum_to_one_and_keep_the_supporting_members_names():
    members = three_members()
    for method, options in (('mean', {}), ('maximum', {}), ('median', {}), ('weighted', {'weights': [2, 1, 1]}),
                            ('base', {'base': 2})):
        combined = combine(members, method, **options)
        assert combined.probabilities.sum(axis=0) == pytest.approx(np.ones(SHAPE), abs=1e-5)
        named = named_label_map(combined)
        assert named[6, 10, 8] == 1
        assert named[12, 4, 8] == (0 if method == 'base' else 35)
    assert named_label_map(combine(members, 'maximum'))[12, 13, 8] == 1
    assert named_label_map(combine(members, 'mean'))[12, 13, 8] == 0
    assert combine(members, 'mean').probabilities == pytest.approx(average(members).probabilities, abs=1e-6)
    members[2].probabilities[:, 20, 15, 10] = [0.7, 0.3, 0]
    for method, options in (('mean', {}), ('maximum', {}), ('median', {}), ('weighted', {'weights': [2, 1, 1]}),
                            ('base', {'base': 2})):
        combined = combine(members, method, **options).probabilities
        assert combined.sum(axis=0) == pytest.approx(np.ones(SHAPE), abs=1e-5), method
    with pytest.raises(ValueError, match='Modality mismatch'):
        combine([*members[:2], softmax_member('mr')], 'maximum')
