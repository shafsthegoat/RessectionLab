"""Generated detached features only: no patient, pixel, checkpoint or native run.

The synthetic sealed corpus below is a test DTO, not acquired reconstruction
provenance. Production cache approvals must come from the original owned run.
"""
import copy
from dataclasses import replace
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from resectionlab.core import immutable_array, freeze_json, thaw_json, semantic_digest, array_digest
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
from resectionlab.spatial_continuation_value import (ContinuationCorpus, _FrozenPolicy, TRAIN_SUBJECTS,
    VERSION, TARGET, WEIGHTING, recorded_action_targets, fit_continuation_value, ContinuationFitConfig)
from resectionlab import spatial_continuation_cache as cache


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        policy = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2), hidden_features=4, ray_samples=2)).eval()
    guard = _FrozenPolicy(policy); manifest = []; bindings = []; worlds = {}
    targets = []; weights = []; stops = []; rl_ordinal = 0
    for subject, teacher_length in zip(TRAIN_SUBJECTS, (1, 14, 1, 13)):
        world = {'subject': subject, 'source_hash': semantic_digest(['generated_source', subject]),
            'decision_model_hash': semantic_digest(['generated_model', subject]),
            'public_observation_binding': semantic_digest(['generated_observation_binding', subject]),
            'observation_track': 'annotation_assisted', 'max_steps': 24}
        worlds[subject] = world
        for ordinal in range(9):
            kind = 'teacher' if ordinal == 0 else 'RL'
            length = teacher_length if ordinal == 0 else 15 if rl_ordinal < 20 else 14
            if ordinal:
                rl_ordinal += 1
            key = subject + '/' + kind + str(ordinal)
            decisions = [{'observation_hash': semantic_digest(['generated_state', key, i]),
                'action_id': 'STOP' if i == length-1 else 'generated_cut',
                'reward': 0. if i == length-1 else -.1 if i == 0 and kind == 'RL' else .2,
                'terminated': i == length-1} for i in range(length)]
            values = recorded_action_targets(d['reward'] for d in decisions)
            manifest.append({'key': key, 'kind': kind, 'world': world, 'decisions': decisions,
                'history_hash': semantic_digest(['generated_history', key]),
                'behavior_parameter_hash': None if ordinal == 0 else semantic_digest(['generated_behavior', ordinal])})
            bindings.append({'key': key, 'world': world, 'rows': length,
                'context_hash': semantic_digest(['generated_context', subject]),
                'trace_seal': semantic_digest(['generated_trace', key])})
            targets.extend(values); weights.extend([1. / (4 * 9 * length)] * length)
            stops.extend(d['action_id'] == 'STOP' for d in decisions)
    width = policy.actor[0].in_features
    arrays = {'features': immutable_array(np.arange(497*width).reshape(497, width)/(497*width), np.float32),
        'targets': immutable_array(targets, np.float32), 'weights': immutable_array(weights, np.float32),
        'stop_rows': immutable_array(stops, np.bool_)}
    metadata = freeze_json({'version': VERSION, 'target': TARGET, 'weighting': WEIGHTING,
        'manifest': manifest, 'manifest_hash': semantic_digest(manifest), 'trace_bindings': bindings,
        'base_parameter_hash': guard.parameters, 'architecture_hash': guard.architecture,
        'worlds': worlds, 'states': 497, 'feature_bytes': arrays['features'].nbytes,
        'STOP_rows': sum(stops), 'STOP_prediction': 'fixed_zero_no_head_gradient',
        'negative_immediate_positive_recorded_continuation': sum(
            d['reward'] < 0 and q > 0 for row in manifest for d, q in zip(row['decisions'],
                recorded_action_targets(d['reward'] for d in row['decisions']))),
        'costs': {'policy_forwards_started': 497, 'policy_forwards_completed': 497,
            'feature_seconds': 123., 'geometry_queries': 0, 'observations_constructed': 0,
            'freeze_checks': 36, 'freeze_check_seconds': 1., 'elapsed_seconds': 400.}})
    seal = semantic_digest({'metadata': metadata, **{k: array_digest(v) for k, v in arrays.items()}})
    corpus = ContinuationCorpus(*(arrays[k] for k in cache.ARRAYS), metadata, seal, guard)
    origin = {k: hashlib.sha256(('generated:'+k).encode()).hexdigest() for k in
        ('source_index_sha256', 'encoder_checkpoint_sha256', 'reconstruction_evidence_sha256')}
    origin['feature_source_sha256'] = cache.current_feature_source_hashes()
    origin['original_reconstruction_costs'] = {'complete_wall_seconds': 400., 'native_preview_entries': 1000,
        'native_transition_calls': 497, 'frozen_policy_forwards': 497}
    return policy, corpus, origin


def saved(tmp_path):
    policy, corpus, origin = fixture(); path = tmp_path/'features.traincache'
    receipt = cache.save_continuation_cache(corpus, policy, path, origin=origin)
    return policy, corpus, path, receipt


def load(path, policy, receipt):
    return cache.load_continuation_cache(path, policy, expected_cache_sha256=receipt['sha256'],
        expected_binding=receipt['binding'])


def rewrite(path, mutate):
    with zipfile.ZipFile(path) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(values['manifest.json'])
    mutate(values, metadata)
    values['manifest.json'] = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, value in values.items(): archive.writestr(name, value)
    path.write_bytes(stream.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change_array(values, metadata, key, array):
    stream = io.BytesIO(); np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    data = stream.getvalue(); entry = 'arrays/' + key + '.npy'; values[entry] = data
    row = next(r for r in metadata['arrays'] if r['name'] == key)
    row['sha256'] = hashlib.sha256(data).hexdigest(); row['array_digest'] = array_digest(array)


def test_roundtrip_exact_corpus_and_new_guard_no_forward_original_costs_preserved(tmp_path, monkeypatch):
    policy, original, path, receipt = saved(tmp_path)
    fresh = copy.deepcopy(policy)
    monkeypatch.setattr(fresh, 'forward', lambda *_: (_ for _ in ()).throw(AssertionError('Cache must not infer')))
    decoded, costs = load(path, fresh, receipt)
    assert decoded.seal == original.seal and decoded.metadata == original.metadata
    assert decoded._guard.policy is fresh and decoded._guard is not original._guard
    assert decoded.require(fresh) is decoded
    with pytest.raises(ValueError, match='different frozen policy object'): decoded.require(policy)
    for key in cache.ARRAYS:
        np.testing.assert_array_equal(getattr(decoded, key), getattr(original, key))
        with pytest.raises(ValueError): getattr(decoded, key).setflags(write=True)
    assert costs['cache_policy_forwards'] == costs['cache_native_previews'] == costs['cache_observations_constructed'] == 0
    assert costs['original_reconstruction_costs'] == receipt['original_reconstruction_costs']
    assert costs['original_corpus_costs']['policy_forwards_completed'] == 497
    assert costs['original_costs_replaced_or_amortized'] is False
    assert parameter_hash(fresh) == parameter_hash(policy)
    second = tmp_path/'again.traincache'
    duplicate = cache.save_continuation_cache(original, policy, second, origin=receipt['binding']['origin'])
    assert duplicate['sha256'] == receipt['sha256']
    with pytest.raises(FileExistsError): cache.save_continuation_cache(original, policy, second, origin=receipt['binding']['origin'])


def test_cached_head_fit_matches_original_exactly_without_feature_recomputation(tmp_path):
    policy, original, path, receipt = saved(tmp_path)
    other = copy.deepcopy(policy); decoded, _ = load(path, other, receipt)
    before = parameter_hash(policy)
    first = fit_continuation_value(policy, original, config=ContinuationFitConfig(updates=2))
    second = fit_continuation_value(other, decoded, config=ContinuationFitConfig(updates=2))
    assert first.report['fitted_head_hash'] == second.report['fitted_head_hash']
    assert first.report['updates'] == second.report['updates']
    assert first.report['final_weighted_mse'] == second.report['final_weighted_mse']
    assert parameter_hash(policy) == parameter_hash(other) == before


@pytest.mark.parametrize('key', ['source_index_sha256', 'encoder_checkpoint_sha256', 'reconstruction_evidence_sha256'])
def test_wrong_external_origin_refused_without_guard(tmp_path, monkeypatch, key):
    policy, _, path, receipt = saved(tmp_path)
    changed = copy.deepcopy(receipt); changed['binding']['origin'][key] = '0'*64
    monkeypatch.setattr(cache, '_FrozenPolicy', lambda *_: (_ for _ in ()).throw(AssertionError('Premature guard')))
    with pytest.raises(ValueError, match='approval binding'): load(path, policy, changed)


@pytest.mark.parametrize('kind', ['parameter', 'architecture'])
def test_different_encoder_or_architecture_refused(tmp_path, kind):
    policy, _, path, receipt = saved(tmp_path)
    if kind == 'parameter':
        with torch.no_grad(): next(policy.encoder.parameters()).add_(.01)
    else:
        policy = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(3, 3), hidden_features=4, ray_samples=2)).eval()
    with pytest.raises(ValueError, match='Different frozen'): load(path, policy, receipt)


@pytest.mark.parametrize('corruption', ['file_hash', 'features', 'targets', 'weights', 'wrong_rows', 'object_dtype', 'extra_member'])
def test_mutation_and_array_bounds_refused_before_guard(tmp_path, monkeypatch, corruption):
    policy, _, path, receipt = saved(tmp_path)
    def mutate(values, metadata):
        if corruption == 'extra_member':
            values['undeclared.txt'] = b'not admitted'; return
        key = corruption if corruption in ('targets', 'weights') else 'features'
        reader = io.BytesIO(values['arrays/' + key + '.npy'])
        array = np.load(reader, allow_pickle=False).copy()
        if corruption == 'wrong_rows': array = array[:-1]
        elif corruption == 'object_dtype':
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(stream, {'descr': '|O', 'fortran_order': False, 'shape': array.shape})
            data = stream.getvalue() + b'not-a-pickle'
            entry = 'arrays/features.npy'; values[entry] = data
            next(r for r in metadata['arrays'] if r['name'] == 'features')['sha256'] = hashlib.sha256(data).hexdigest()
            return
        else: array.flat[0] += .25
        change_array(values, metadata, key, array)
    digest = rewrite(path, mutate)
    # Even approving the complete altered file cannot override original corpus
    # provenance or structural/target checks. file_hash leaves old approval.
    if corruption != 'file_hash': receipt['sha256'] = digest
    monkeypatch.setattr(cache, '_FrozenPolicy', lambda *_: (_ for _ in ()).throw(AssertionError('Premature guard')))
    with pytest.raises(ValueError): load(path, policy, receipt)


def test_unknown_role_cannot_become_authority_even_with_self_consistent_hashes(tmp_path, monkeypatch):
    policy, _, path, receipt = saved(tmp_path)
    def mutate(values, metadata):
        corpus_meta = metadata['corpus_metadata']
        corpus_meta['manifest'][0]['world']['role'] = 'SELECT'
        corpus_meta['worlds'][TRAIN_SUBJECTS[0]]['role'] = 'SELECT'
        corpus_meta['manifest_hash'] = semantic_digest(corpus_meta['manifest'])
        binding = metadata['binding']
        binding['metadata_hash'] = semantic_digest(corpus_meta)
        binding['trace_manifest_hash'] = corpus_meta['manifest_hash']
        arrays = {key: np.load(io.BytesIO(values['arrays/' + key + '.npy']), allow_pickle=False) for key in cache.ARRAYS}
        binding['corpus_seal'] = semantic_digest({'metadata': corpus_meta, **{k: array_digest(v) for k, v in arrays.items()}})
        receipt['binding'] = copy.deepcopy(binding)
    receipt['sha256'] = rewrite(path, mutate)
    monkeypatch.setattr(cache, '_FrozenPolicy', lambda *_: (_ for _ in ()).throw(AssertionError('Premature guard')))
    with pytest.raises(ValueError, match='world'): load(path, policy, receipt)


def test_oversize_and_dirty_model_refusal_then_loaded_guard_mutation(tmp_path, monkeypatch):
    policy, _, path, receipt = saved(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(cache, 'MAX_BYTES', path.stat().st_size - 1)
        with pytest.raises(ValueError, match='Bounded regular'): load(path, policy, receipt)
    parameter = next(policy.parameters()); parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError, match='clean gradient'): load(path, policy, receipt)
    parameter.grad = None
    decoded, _ = load(path, policy, receipt)
    with torch.no_grad(): parameter.add_(.001)
    with pytest.raises(ValueError, match='Frozen encoder'): decoded.require(policy)


def test_same_parameters_do_not_authorize_changed_feature_source(tmp_path, monkeypatch):
    policy, _, path, receipt = saved(tmp_path)
    changed = cache.current_feature_source_hashes(); changed['resectionlab.spatial_policy'] = '0'*64
    monkeypatch.setattr(cache, 'current_feature_source_hashes', lambda: changed)
    monkeypatch.setattr(cache, '_FrozenPolicy', lambda *_: (_ for _ in ()).throw(AssertionError('Premature guard')))
    with pytest.raises(ValueError, match='feature construction source changed'): load(path, policy, receipt)
