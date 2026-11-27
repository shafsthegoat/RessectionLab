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


def varying_predictor(patch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('nnunetv2')
    from types import SimpleNamespace
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.manual_seed(7)
    network = torch.nn.Conv3d(1, 5, 3, padding=1)
    folds = [{key: value + shift for key, value in network.state_dict().items()} for shift in (0.0, 0.25)]
    predictor = SimpleNamespace(
        use_mirroring=False, network=network, device=torch.device('cpu'), list_of_parameters=folds,
        configuration_manager=SimpleNamespace(patch_size=list(patch)),
        label_manager=SimpleNamespace(num_segmentation_heads=5))
    predictor._internal_get_sliding_window_slicers = lambda size: nnUNetPredictor._internal_get_sliding_window_slicers(
        SimpleNamespace(configuration_manager=predictor.configuration_manager, tile_step_size=0.5, verbose=False), size)
    return torch, predictor


@pytest.mark.parametrize('shape', [(19, 13, 11), (8, 13, 6), (5, 9, 9), (19, 4, 5), (9, 6, 3)])
def test_a_scan_too_large_for_memory_gets_the_values_of_the_in_memory_path(tmp_path, shape):
    torch, predictor = varying_predictor((8, 6, 6))
    folds = list(predictor.list_of_parameters)
    data = torch.randn(1, *shape) * 3
    memory = native.network_probabilities(predictor, data, tmp_path / 'unused', limit=1 << 40)
    predictor.list_of_parameters = list(folds)
    banded = native.predict_banded(predictor, data, tmp_path / 'slices')
    assert (memory.buffer, banded.buffer) == ('memory', 'drive') and not (tmp_path / 'unused').exists()
    assert memory.heads == banded.heads == 5 and banded.shape == shape
    for channel in range(5):
        expected = memory.channel(channel)
        assert expected.shape == shape and expected.std() > 0
        assert np.array_equal(banded.channel(channel), expected)
    banded.close()
    assert not (tmp_path / 'slices').exists()


def test_a_failed_banded_prediction_leaves_no_scratch_file(tmp_path):
    torch, predictor = varying_predictor((8, 6, 6))
    tiles = predictor._internal_get_sliding_window_slicers
    predictor._internal_get_sliding_window_slicers = lambda size: tiles(size)[::-1]
    with pytest.raises(RuntimeError, match='not ordered'):
        native.predict_banded(predictor, torch.randn(1, 19, 13, 11), tmp_path / 'slices')
    assert not (tmp_path / 'slices').exists()


def test_the_buffer_is_chosen_by_its_size_and_a_band_that_cannot_fit_is_refused(tmp_path, monkeypatch):
    torch, predictor = varying_predictor((8, 6, 6))
    folds = list(predictor.list_of_parameters)
    data = torch.randn(1, 19, 13, 11)
    whole, band = 2 * 5 * 19 * 13 * 11, 2 * 5 * 8 * 13 * 11
    assert native.network_probabilities(predictor, data, tmp_path / 'slices', limit=whole).buffer == 'memory'
    predictor.list_of_parameters = list(folds)
    banded = native.network_probabilities(predictor, data, tmp_path / 'slices', limit=whole - 1)
    assert banded.buffer == 'drive'
    banded.close()
    predictor.list_of_parameters = list(folds)
    assert native.network_probabilities(predictor, data, tmp_path / 'slices', limit=band).buffer == 'drive'
    predictor.list_of_parameters = list(folds)
    with pytest.raises(SystemExit, match='one band of logits'):
        native.network_probabilities(predictor, data, tmp_path / 'other', limit=band - 1)
    assert not (tmp_path / 'other').exists()
    # A smaller scan than the patch is padded to it, and that is the size that counts.
    predictor.list_of_parameters = list(folds)
    small = torch.randn(1, 3, 4, 5)
    assert native.network_probabilities(predictor, small, tmp_path / 'small', limit=2 * 5 * 8 * 6 * 6).buffer == 'memory'
    monkeypatch.setattr(native.os, 'sysconf', lambda name: {'SC_PAGE_SIZE': 4096, 'SC_PHYS_PAGES': 1000}[name])
    assert native.memory_buffer_limit() == int(4096 * 1000 * 0.4)


@pytest.mark.parametrize('limit', [1 << 40, 0])
def test_a_stage_writes_the_same_file_from_either_buffer_and_leaves_no_scratch(tmp_path, monkeypatch, limit):
    from types import SimpleNamespace
    torch, predictor = varying_predictor((8, 6, 6))
    folds = list(predictor.list_of_parameters)
    shape = (19, 13, 11)
    predictor.plans_manager = SimpleNamespace(transpose_forward=[0, 1, 2], transpose_backward=[0, 1, 2])
    predictor.configuration_manager.spacing = [1., 1., 1.]
    properties = {'spacing': [1., 1., 1.], 'shape_before_cropping': shape, 'shape_after_cropping_and_before_resampling': shape,
                  'bbox_used_for_cropping': [[0, size] for size in shape]}
    back = np.array([[0, 1], [1, 1], [2, 1]])
    data = torch.randn(1, *shape, generator=torch.Generator().manual_seed(3)) * 3
    monkeypatch.setattr(native, 'memory_buffer_limit', lambda: limit)
    real = native.network_probabilities
    monkeypatch.setattr(native, 'network_probabilities', lambda *a, **k: real(*a, band_limit=1 << 40, **k))
    written, kept, timing = native.run_stage(predictor, data, properties, 'trilinear', back, tmp_path / 'stage.npy', keep=1)
    assert timing['logit_buffer'] == ('memory' if limit else 'drive') and sorted(item.name for item in tmp_path.iterdir()) == ['stage.npy']
    assert written.shape == (5, 11, 13, 19) and np.array_equal(written[1], kept.transpose(2, 1, 0).astype(np.float16))
    predictor.list_of_parameters = list(folds)
    monkeypatch.setattr(native, 'memory_buffer_limit', lambda: 1 << 40)
    reference, _, _ = native.run_stage(predictor, data, properties, 'trilinear', back, tmp_path / 'reference.npy')
    assert all(np.array_equal(written[channel], reference[channel]) for channel in range(5))
    # A stage that fails while writing leaves no scratch file behind either.
    predictor.list_of_parameters = list(folds)
    monkeypatch.setattr(native, 'memory_buffer_limit', lambda: limit)
    monkeypatch.setattr(native.NpyWriter, 'write', lambda self, channel: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        native.run_stage(predictor, data, properties, 'trilinear', back, tmp_path / 'failed.npy')
    assert not list(tmp_path.glob('*' + native.SCRATCH))


def test_slice_probabilities_refuse_a_short_or_misshapen_file(tmp_path):
    store = native.SliceProbabilities(tmp_path / 'slices', 2, (3, 4, 5))
    with pytest.raises(ValueError, match='shape'):
        store.write(np.zeros((2, 4, 4), np.float16))
    store.write(np.zeros((2, 4, 5), np.float16))
    with pytest.raises(ValueError, match='wrote 1 of 3'):
        store.finish()
    store.close()
    assert not (tmp_path / 'slices').exists()


def drive(tmp_path, *, series=None, run_sha=None, run_path=None, blend='gaussian'):
    """A marked data root with one prepared study (a T1 and a CTA by default) and one retained vesselFM run."""
    import hashlib
    root = tmp_path / 'drive'
    study = root / 'prepared' / 'collection' / 'Head CTA 013 (v1)'
    (study / 'originals' / 'abc').mkdir(parents=True)
    (root / '.ressectionlab-data-root.json').write_text('{}')
    image = nib.Nifti1Image(np.zeros((6, 7, 8), np.float32), np.diag([.5, .5, .6, 1.]))
    scan = study / 'originals' / 'abc' / 'Head CTA native.nii.gz'
    nib.save(image, scan)
    sha = hashlib.sha256(scan.read_bytes()).hexdigest()
    rows = series or [{'id': 't1', 'modality': 'T1', 'original': {'path': 'originals/abc/t1.nii.gz', 'sha256': 'x'}},
                      {'id': 'cta', 'modality': 'CTA', 'original': {'path': 'originals/abc/Head CTA native.nii.gz', 'sha256': sha}}]
    for row in rows:
        if row['original']['sha256'] == 'same':
            row['original'] = {'path': 'originals/abc/Head CTA native.nii.gz', 'sha256': sha}
    (study / 'patient-anatomy.json').write_text(json.dumps({'series': rows, 'layers': []}))
    run = root / 'cache' / 'poc-model-results' / 'anatomy-1234' / 'vesselfm-base' / 'run1'
    run.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((6, 7, 8), np.float32), image.affine), run / 'vessels_probability.nii.gz')
    (run / 'inference.json').write_text(json.dumps({
        'status': 'succeeded', 'input': {'path': run_path or str(scan), 'sha256': run_sha or sha, 'modality': 'CTA'},
        'probability': {'path': str(run / 'vessels_probability.nii.gz')}, 'model': {'parameters': {'blend': blend}}}))
    return root, study / 'patient-anatomy.json', sha


@pytest.fixture
def case_table(monkeypatch):
    monkeypatch.setattr(native, 'CASES', dict(native.CASES))
    monkeypatch.setattr(native, 'VESSELFM', dict(native.VESSELFM))


def test_any_prepared_study_registers_by_its_one_angiographic_series(tmp_path, case_table):
    root, manifest, sha = drive(tmp_path, run_path='/somewhere/else/the-same-file.nii.gz')
    case = native.register_study(root, manifest.relative_to(root))
    assert case == 'Head-CTA-013-v1-ct'
    entry = native.CASES[case]
    assert entry['modality'] == 'ct' and entry['sha256'] == sha and entry['series'] == 'cta' and entry['reference'] is None
    assert entry['image'] == 'prepared/collection/Head CTA 013 (v1)/originals/abc/Head CTA native.nii.gz'
    assert entry['flags'] == ['--study', 'prepared/collection/Head CTA 013 (v1)/patient-anatomy.json']
    # The vesselFM run is matched by the file's content, so a study that moved since still finds it.
    assert set(native.vesselfm_runs(root, case)) == {'gaussian'}
    # Its results live apart from the parity evidence; the parity cases stay where they were.
    assert native.kept_folder(root, case) == root / 'cache/vessel-ensembles/cases' / case
    assert native.work_folder(root, case) == root / 'cache/vessel-ensembles/work' / case
    assert native.case_folder(root, case, 'UZH') == root / 'cache/vessel-ensembles/cases' / case / 'UZH'
    assert native.kept_folder(root, 'TopCoW-014-ct') == root / native.PHASE / 'cases' / 'TopCoW-014-ct'
    assert native.work_folder(root, 'TopCoW-014-ct') == root / native.WORK / 'TopCoW-014-ct'
    # Registering it again, by absolute path, is the same case.
    assert native.register_study(root, manifest) == case


def test_time_of_flight_and_mra_register_as_mr_and_a_named_series_settles_a_choice(tmp_path, case_table):
    both = [{'id': 'tof', 'modality': 'TOF', 'original': {'path': '', 'sha256': 'same'}},
            {'id': 'cta', 'modality': 'CTA', 'original': {'path': '', 'sha256': 'same'}}]
    root, manifest, _ = drive(tmp_path, series=both)
    with pytest.raises(SystemExit, match='exactly one'):
        native.register_study(root, manifest.relative_to(root))
    document = json.loads(manifest.read_text())
    # A series named on the command line is part of the case name: two series of one study never share results.
    assert native.study_case(document, 'A study', 'tof')[::2] == ('A-study-tof-mr', 'mr')
    assert native.study_case(document, 'A study', 'cta')[::2] == ('A-study-cta-ct', 'ct')
    assert native.study_case({'series': [both[0]]}, 'A study')[::2] == ('A-study-mr', 'mr')
    with pytest.raises(SystemExit, match='exactly one'):
        native.study_case(document, 'A study', 'missing')
    with pytest.raises(SystemExit, match='exactly one'):
        native.study_case({'series': [{'id': 't1', 'modality': 'T1', 'original': {}}]}, 'A study')
    with pytest.raises(SystemExit, match='usable case name'):
        native.study_case({'series': [both[0]]}, '()')
    assert native.register_study(root, manifest.relative_to(root), 'cta') == 'Head-CTA-013-v1-cta-ct'
    assert native.CASES['Head-CTA-013-v1-cta-ct']['flags'][-2:] == ['--series', 'cta']


@pytest.mark.parametrize('fault, message', [('sha', 'no retained vesselFM run'), ('blend', 'not a gaussian-blend run'),
                                            ('missing', 'No study document'), ('changed', 'not the file the study document')])
def test_a_study_without_what_the_ensemble_needs_is_refused_with_the_reason(tmp_path, case_table, fault, message):
    root, manifest, _ = drive(tmp_path, run_sha='0' * 64 if fault == 'sha' else None,
                              blend='constant' if fault == 'blend' else 'gaussian')
    target = manifest.relative_to(root) if fault != 'missing' else 'prepared/collection/nothing/patient-anatomy.json'
    if fault == 'changed':
        scan = manifest.parent / 'originals' / 'abc' / 'Head CTA native.nii.gz'
        scan.write_bytes(scan.read_bytes() + b'0')
    before = set(native.CASES)
    with pytest.raises(SystemExit, match=message):
        native.register_study(root, target)
    assert fault == 'blend' or set(native.CASES) == before


def test_the_committed_rule_runs_a_study_and_must_agree_with_the_drive(tmp_path):
    rule = native.committed_rule()
    assert rule['teams'] == list(native.TEAM_ORDER) and set(rule['combiner']) >= {'method', 'teams', 'base'}
    assert set(rule['combiner']['base']) == {'ct', 'mr'} and set(rule['connectedAddition']) == {'low', 'high'}
    assert native.combined_name(rule['combiner'], rule['teams']) == 'combined-base-UZH-ARG-KDH'
    # The parity commands keep their meaning: without the drive's selection files nothing is frozen for them.
    assert native.frozen_combiner(tmp_path) is None
    with pytest.raises(SystemExit, match='run select'):
        native.frozen_addition(tmp_path)
    # A study runs with the committed copy, which must equal the drive's selections where they exist.
    assert native.study_rule(tmp_path) == rule
    folder = tmp_path / native.PHASE
    folder.mkdir(parents=True)
    (folder / 'addition-selection.json').write_text(json.dumps({'selected': {**rule['connectedAddition'], 'other': 1}}))
    (folder / 'combiner-selection.json').write_text(json.dumps({'selected': rule['combiner']}))
    assert native.study_rule(tmp_path) == rule
    (folder / 'combiner-selection.json').write_text(json.dumps({'selected': {**rule['combiner'], 'method': 'mean'}}))
    with pytest.raises(SystemExit, match='combiner-selection.json differs'):
        native.study_rule(tmp_path)
    (folder / 'combiner-selection.json').unlink()
    (folder / 'addition-selection.json').write_text(json.dumps({'selected': {'low': 0.5, 'high': 0.9}}))
    with pytest.raises(SystemExit, match='addition-selection.json differs'):
        native.study_rule(tmp_path)


def test_kept_and_unfinished_results_must_come_from_the_case_file(tmp_path, case_table, monkeypatch):
    root, manifest, sha = drive(tmp_path)
    case = native.register_study(root, manifest.relative_to(root))
    image = native.CASES[case]['image']
    assert native.made_from(case) == {'image': image, 'sha256': sha, 'series': 'cta'}
    assert native.made_from('TopCoW-014-ct') == {'image': native.CASES['TopCoW-014-ct']['image']}
    record = tmp_path / 'predict.json'

    def refused(document, match):
        record.write_text(json.dumps(document))
        with pytest.raises(SystemExit, match=match):
            native.require_made_from(record, case)

    record.write_text(json.dumps({'image': image, 'sha256': sha, 'series': 'cta'}))
    assert native.require_made_from(record, case)['sha256'] == sha
    # The same file after the study moved is still the same file; a new file at the old path is not.
    record.write_text(json.dumps({'image': 'elsewhere/scan.nii.gz', 'sha256': sha, 'series': 'cta'}))
    native.require_made_from(record, case)
    refused({'image': image, 'sha256': '0' * 64, 'series': 'cta'}, 'another file')
    refused({'image': image, 'sha256': sha, 'series': 'tof'}, 'another file')
    # A record from before hashes were recorded is held to the path.
    record.write_text(json.dumps({'image': image}))
    native.require_made_from(record, case)
    refused({'image': 'prepared/other/Head CTA 013 (v1)/scan.nii.gz'}, 'another file')
    # An ensemble that does not name its source is held to the team predictions beside it; a study's must have some.
    kept = native.kept_folder(root, case)
    ensemble, team = kept / 'combined' / 'ensemble.json', kept / 'UZH' / 'predict.json'
    ensemble.parent.mkdir(parents=True)
    team.parent.mkdir(parents=True)
    ensemble.write_text(json.dumps({'teams': ['UZH', 'ARG']}))
    with pytest.raises(SystemExit, match='does not say which file'):
        native.require_made_from(ensemble, case)
    team.write_text(json.dumps({'image': image}))
    native.require_made_from(ensemble, case)
    team.write_text(json.dumps({'image': 'prepared/other/scan.nii.gz'}))
    with pytest.raises(SystemExit, match='another file'):
        native.require_made_from(ensemble, case)
    # A team record with a hash is held to the hash, and a prediction of another file is never combined.
    team.write_text(json.dumps({'image': image, 'sha256': '0' * 64, 'series': 'cta'}))
    with pytest.raises(SystemExit, match='another file'):
        native.require_made_from(ensemble, case)
    with pytest.raises(SystemExit, match='another file'):
        native.team_probabilities(root, case, 'UZH')
    # The run loop checks before it reuses or predicts anything.
    name = native.combined_name(native.committed_rule()['combiner'], native.committed_rule()['teams'])
    (kept / name).mkdir()
    (kept / name / 'ensemble.json').write_text(json.dumps({'image': image, 'sha256': '0' * 64, 'series': 'cta'}))
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('predicted before checking'))
    with pytest.raises(SystemExit, match='another file'):
        native.run_cases(root, [case])
    (kept / name / 'ensemble.json').write_text(json.dumps({'image': image, 'sha256': sha, 'series': 'cta'}))
    assert native.run_cases(root, [case]) == [case]
    (kept / name / 'ensemble.json').unlink()
    unfinished = native.work_folder(root, case) / 'UZH' / 'predict.json'
    unfinished.parent.mkdir(parents=True)
    unfinished.write_text(json.dumps({'image': 'prepared/collection/another/scan.nii.gz'}))
    with pytest.raises(SystemExit, match='another file'):
        native.run_cases(root, [case])
    with pytest.raises(SystemExit, match='runs the teams'):
        native.run_cases(root, [case], teams=('UZH', 'ARG'))



def test_keeping_a_case_copies_the_record_that_marks_it_kept_last(tmp_path, case_table, monkeypatch):
    root, manifest, _ = drive(tmp_path)
    case = native.register_study(root, manifest.relative_to(root))
    work = native.work_folder(root, case) / 'combined'
    work.mkdir(parents=True)
    np.save(work / 'probabilities.npy', np.zeros((2, 2, 2, 2), np.float16))
    (work / 'ensemble.json').write_text(json.dumps({'labels': [0, 1]}))
    for name in ('named_labels.nii.gz', 'display_additions.json', 'vessel_probability.nii.gz'):
        (work / name).write_bytes(b'x')
    copied = []
    copy = native.shutil.copyfile
    monkeypatch.setattr(native.shutil, 'copyfile', lambda source, target: (copied.append(Path(source).name), copy(source, target))[1])
    kept = native.retain(root, case)
    assert copied[-1] == 'ensemble.json' and set(copied) == {path.name for path in (kept / 'combined').iterdir()} - {'probabilities.npz'}
    assert not work.exists()
