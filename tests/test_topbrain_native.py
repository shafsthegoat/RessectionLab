import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'topbrain_native.py'
spec = importlib.util.spec_from_file_location('topbrain_native', SCRIPT)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


@pytest.mark.parametrize('codes', ['LAS', 'RAS', 'ASL', 'LPS'])
def test_lps_view_round_trips_to_the_source_voxel_order(codes):
    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    lps = nib.Nifti1Image(values, np.diag([-0.5, -0.6, 0.7, 1.0]))
    ornt = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt('LPS'), nib.orientations.axcodes2ornt(codes))
    image = lps.as_reoriented(ornt)
    assert ''.join(nib.aff2axcodes(image.affine)) == codes
    zyx, spacing, back = native.lps_view(image)
    assert np.array_equal(zyx, values.transpose(2, 1, 0))
    assert spacing == pytest.approx([0.7, 0.6, 0.5])
    assert np.array_equal(native.to_source(zyx, back), np.asanyarray(image.dataobj))


def test_channel_files_round_trip(tmp_path):
    values = np.random.default_rng(0).random((3, 4, 5, 6)).astype(np.float16)
    writer = native.NpyWriter(tmp_path / 'x.npy', values.shape)
    for channel in range(3):
        writer[channel] = values[channel]
    reader = writer.close()
    assert [np.array_equal(reader[c], values[c]) for c in (2, 0, 1)] == [True] * 3
    with pytest.raises(ValueError, match='order'):
        bad = native.NpyWriter(tmp_path / 'y.npy', values.shape)
        bad[1] = values[1]
    with (tmp_path / 'x.npz').open('wb') as stream:
        np.savez_compressed(stream, probabilities=values)
    packed = native.NpzChannels(tmp_path / 'x.npz')
    assert packed.shape == values.shape and packed.dtype == values.dtype
    assert [np.array_equal(packed[c], values[c]) for c in (1, 2, 0, 2)] == [True] * 4


def softmax(rng, shape, channels):
    logits = rng.normal(size=(channels, *shape))
    return (np.exp(logits) / np.exp(logits).sum(axis=0)).astype(np.float32)


def test_team_compositions_sum_to_one(tmp_path):
    rng = np.random.default_rng(1)
    shape = (5, 6, 7)
    regions = [(softmax(rng, shape, 3), [0, 1, 2], {1: 4, 2: 6}), (softmax(rng, shape, 2), [0, 1], {1: 1})]
    composed = native.compose_regions(regions, tuple(range(8)), tmp_path / 'regions.npy')
    assert np.stack(list(composed)).sum(axis=0) == pytest.approx(np.ones(shape), abs=2e-3)
    stage = softmax(rng, shape, 49)
    mapped = native.map_channels(stage, list(range(49)), native.ARG_SOFT_MAPPING['ct'], native.TOPBRAIN_LABELS['ct'],
                                 tmp_path / 'arg.npy')
    total = np.stack(list(mapped)).astype(np.float32)
    assert total.sum(axis=0) == pytest.approx(np.ones(shape), abs=2e-3)
    assert total[0] == pytest.approx(stage[0] + stage[35:43].sum(axis=0), abs=2e-3)
    assert total[35] == pytest.approx(stage[43], abs=2e-3)
    masked = np.stack(list(native.binary_masked(softmax(rng, shape, 2), softmax(rng, shape, 43))))
    assert masked.sum(axis=0) == pytest.approx(np.ones(shape), abs=1e-5)


def test_arg_maps_keep_the_image_behaviour_and_drop_external_carotids_only_in_the_soft_ct_path():
    official, soft = native.ARG_MAPPING, native.ARG_SOFT_MAPPING
    assert set(official['ct']) == set(soft['ct']) == set(range(49)) == set(official['mr'])
    assert official['ct'][35] == 35 and official['ct'][43] == 35 and official['ct'][41] == 0
    assert soft['ct'][35] == soft['ct'][42] == 0 and soft['ct'][48] == 40 and soft['ct'][34] == 34
    assert official['mr'][42] == 42 and official['mr'][43] == 0 and soft['mr'] == official['mr']


def test_component_filters():
    hard = np.zeros((20, 20, 20), np.uint8)
    hard[1:8, 1:8, 1:8] = 3
    hard[15, 15, 15] = 3
    hard[12:14, 2:4, 2:4] = 5
    small = native.drop_small_per_label(hard.copy(), 13)
    assert small[15, 15, 15] == 0 and small[2, 2, 2] == 3 and (small == 5).sum() == 0
    assert native.keep_large_components(hard)[15, 15, 15] == 3
    wide = np.zeros((40, 40, 40), np.uint8)
    wide[:30, :30, :30] = 7
    wide[35, 35, 35] = 7
    wide[33:36, 2:6, 2:4] = 9
    large = native.keep_large_components(wide)
    assert large[35, 35, 35] == 0 and large[34, 3, 3] == 9 and large[5, 5, 5] == 7
    mask = hard > 0
    assert native.filter_components(mask, 'largest').sum() == 343
    assert native.filter_components(mask, 5).sum() == 351 and native.filter_components(mask, 0) is mask


def test_combiner_arguments_follow_team_names_not_positions():
    combiner = {'method': 'base', 'teams': ['UZH', 'ARG', 'KDH'], 'base': {'ct': 'UZH', 'mr': 'ARG'},
                'weights': {'ct': {'UZH': 0.9, 'ARG': 0.8, 'KDH': 0.7}}}
    assert native.combiner_arguments(combiner, ['KDH', 'ARG', 'UZH'], 'mr') == (None, 1)
    assert native.combiner_arguments(combiner, ['KDH', 'UZH', 'ARG'], 'ct') == (None, 1)
    weighted = {**combiner, 'method': 'weighted'}
    assert native.combiner_arguments(weighted, ['ARG', 'KDH', 'UZH'], 'ct') == ([0.8, 0.7, 0.9], None)
    with pytest.raises(ValueError, match='chosen for'):
        native.combiner_arguments(combiner, ['UZH', 'ARG'], 'ct')
    assert native.combined_name(combiner, ['UZH', 'ARG', 'KDH']) == 'combined-base-UZH-ARG-KDH'


def test_objective_and_held_out_log_only_grow(tmp_path, monkeypatch):
    baseline = {'dice': 0.9, 'cl_dice': 0.9, 'betti': {'reference': {'b0': 7, 'b1': 3}, 'b0_error': 2, 'b1_error': 2}}
    better = {'dice': 0.92, 'cl_dice': 0.91, 'betti': {'b0_error': 1, 'b1_error': 2}}
    assert native.objective_change(better, baseline) == pytest.approx(0.02 + 0.01 + 0.1)
    monkeypatch.setattr(native, 'HELDOUT_LOG', 'log.json')
    native.append_heldout(tmp_path, {'case': 'TopCoW-018-ct', 'pass': 1})
    native.append_heldout(tmp_path, {'case': 'TopCoW-018-ct', 'pass': 2})
    entries = json.loads((tmp_path / 'log.json').read_text())
    assert [entry['pass'] for entry in entries] == [1, 2] and all('when' in entry for entry in entries)


def test_predict_logits_weights_tiles_and_keeps_constant_inputs_constant():
    torch = pytest.importorskip('torch')
    pytest.importorskip('nnunetv2')
    from types import SimpleNamespace

    class Constant(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(float(value)))

        def forward(self, x):
            return torch.cat([x * 0 + self.value, x * 0 - self.value], 1)

    network = Constant(0)
    predictor = SimpleNamespace(
        use_mirroring=False, network=network, device=torch.device('cpu'),
        list_of_parameters=[{'value': torch.tensor(1.0)}, {'value': torch.tensor(3.0)}],
        configuration_manager=SimpleNamespace(patch_size=[4, 4, 4]),
        label_manager=SimpleNamespace(num_segmentation_heads=2))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor._internal_get_sliding_window_slicers = lambda size: nnUNetPredictor._internal_get_sliding_window_slicers(
        SimpleNamespace(configuration_manager=predictor.configuration_manager, tile_step_size=0.5, verbose=False), size)
    logits = native.predict_logits(predictor, torch.ones(1, 7, 9, 5))
    assert tuple(logits.shape) == (2, 7, 9, 5) and predictor.list_of_parameters == []
    assert torch.allclose(logits[0].float(), torch.full((7, 9, 5), 2.0), atol=1e-2)
    assert torch.allclose(logits[1].float(), torch.full((7, 9, 5), -2.0), atol=1e-2)
