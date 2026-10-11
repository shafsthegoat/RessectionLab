"""Exact-frozen-encoder TRAIN feature cache; no source or checkpoint decoding.

A loader needs TWO out-of-band approvals: the cache's complete file SHA256 and
an expected binding from the original authenticated corpus/evidence receipt.
Never construct that expectation from the archive being loaded. This is owned
runner provenance, like the parameter-only loader, not a hostile-Python sandbox.
It creates no patient context, trace capability, observation or replay verdict.

Cached detached action features are simulator-derived TRAIN intermediates, not
pixels, fresh evidence or zero-cost collection. Original metadata/costs remain
byte-semantic identical; cache I/O costs are returned separately.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import time
import zipfile

import numpy as np

from .core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .spatial_policy import SpatialPolicy, parameter_hash
from .spatial_continuation_value import (ContinuationCorpus, TRAIN_SUBJECTS, VERSION,
    TARGET, WEIGHTING, _FrozenPolicy, recorded_action_targets)

CACHE_VERSION = 'exact-frozen-spatial-TRAIN-feature-cache-v1'
MAX_BYTES = 8 * 1024**2
MAX_METADATA_BYTES = 1024**2
ARRAYS = ('features', 'targets', 'weights', 'stop_rows')
ORIGIN_KEYS = {'source_index_sha256', 'encoder_checkpoint_sha256',
               'reconstruction_evidence_sha256', 'original_reconstruction_costs', 'feature_source_sha256'}
BINDING_KEYS = {'corpus_seal', 'metadata_hash', 'trace_manifest_hash',
                'base_parameter_hash', 'architecture_hash', 'origin'}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _hash(value, *, semantic=False):
    prefix = 'sha256:' if semantic else ''
    _need(type(value) is str and value.startswith(prefix) and len(value) == len(prefix) + 64
          and all(c in '0123456789abcdef' for c in value[len(prefix):]), 'Exact SHA256 identity required')
    return value


def current_feature_source_hashes():
    """Source-only encoder/DTO/feature-extractor identity, never a pixel read.

The owned runner also freezes its complete closure. These exact relevant bytes
prevent unchanged tensors/architecture from silently admitting changed feature
construction. Expected hashes must originate in the original approved closure.
"""
    from . import spatial_policy, spatial_observations, public_target_context, spatial_continuation_value
    modules = (spatial_policy, spatial_observations, public_target_context, spatial_continuation_value)
    result = {}
    for module in modules:
        path = Path(module.__file__)
        _need(path.suffix == '.py' and not path.is_symlink() and 0 < path.stat().st_size <= 1024**2,
              'Bounded actual feature source bytes required')
        result[module.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _origin(origin):
    origin = freeze_json(origin)
    _need(set(origin) == ORIGIN_KEYS, 'Exact original evidence/cost fields required')
    for key in ORIGIN_KEYS - {'original_reconstruction_costs', 'feature_source_sha256'}:
        _hash(origin[key])
    sources = origin['feature_source_sha256']
    _need(set(sources) == {'resectionlab.spatial_policy', 'resectionlab.spatial_observations',
        'resectionlab.public_target_context', 'resectionlab.spatial_continuation_value'},
        'Exact original feature-source pins required')
    for digest in sources.values(): _hash(digest)
    costs = origin['original_reconstruction_costs']
    _need(set(costs) == {'complete_wall_seconds', 'native_preview_entries', 'native_transition_calls',
                         'frozen_policy_forwards'}, 'Explicit original paid reconstruction costs required')
    _need(type(costs['complete_wall_seconds']) in (int, float) and costs['complete_wall_seconds'] >= 0,
          'Finite nonnegative original time required')
    for key in ('native_preview_entries', 'native_transition_calls', 'frozen_policy_forwards'):
        _need(type(costs[key]) is int and costs[key] >= 0, 'Integer original cost required')
    _need(costs['frozen_policy_forwards'] == 497, 'Original497 extraction forwards remain charged')
    return origin


def _binding(binding):
    binding = freeze_json(binding)
    _need(set(binding) == BINDING_KEYS, 'Exact independently approved cache binding required')
    for key in BINDING_KEYS - {'origin'}:
        _hash(binding[key], semantic=True)
    _origin(binding['origin'])
    return binding


def _seal(metadata, arrays):
    return semantic_digest({'metadata': metadata, **{k: array_digest(arrays[k]) for k in ARRAYS}})


def _validate_payload(metadata, arrays, policy, binding):
    """No new guard/authority until all byte and semantic joins have passed."""
    _need(type(policy) is SpatialPolicy, 'Existing exact frozen spatial policy required')
    _need(freeze_json(current_feature_source_hashes()) == binding['origin']['feature_source_sha256'],
          'Frozen feature construction source changed')
    _need(policy.architecture_hash == binding['architecture_hash'] and
          parameter_hash(policy) == binding['base_parameter_hash'], 'Different frozen encoder/actor identity')
    _need(semantic_digest(metadata) == binding['metadata_hash'], 'Original corpus metadata changed')
    _need(metadata['version'] == VERSION and metadata['target'] == TARGET and metadata['weighting'] == WEIGHTING,
          'Original logged-action target/weighting declaration required')
    _need(metadata['base_parameter_hash'] == binding['base_parameter_hash'] and
          metadata['architecture_hash'] == binding['architecture_hash'], 'Corpus policy identity differs')
    manifest = metadata['manifest']; worlds = metadata['worlds']; traces = metadata['trace_bindings']
    _need(len(manifest) == len(traces) == 36 and metadata['states'] == 497,
          'Exact36 complete traces and497 detached rows required')
    _need(semantic_digest(manifest) == metadata['manifest_hash'] == binding['trace_manifest_hash'],
          'Original trace manifest identity changed')
    _need(set(worlds) == set(TRAIN_SUBJECTS), 'Only fixed original four TRAIN worlds')
    counts = Counter(); keys = set(); target = []; weights = []; stops = []; negative_preparation = 0
    world_keys = {'subject', 'source_hash', 'decision_model_hash', 'public_observation_binding',
                  'observation_track', 'max_steps'}
    for row, trace in zip(manifest, traces):
        _need(set(row) == {'key', 'kind', 'world', 'decisions', 'history_hash', 'behavior_parameter_hash'},
              'Exact original trace descriptor fields required')
        world = row['world']; subject = world.get('subject'); kind = row['kind']; decisions = row['decisions']
        _need(set(world) == world_keys and subject in TRAIN_SUBJECTS and worlds[subject] == world and
              world['max_steps'] == 24 and world['observation_track'] == 'annotation_assisted',
              'Unknown/foreign TRAIN subject, world, track or horizon')
        for key in ('source_hash', 'decision_model_hash', 'public_observation_binding'):
            _hash(world[key], semantic=True)
        _need(type(row['key']) is str and row['key'] not in keys and kind in ('teacher', 'RL'),
              'Unique declared teacher/RL key required')
        keys.add(row['key']); counts[(subject, kind)] += 1
        _need((kind == 'teacher') == (row['behavior_parameter_hash'] is None), 'Historical behavior provenance differs')
        if kind == 'RL':
            _hash(row['behavior_parameter_hash'], semantic=True)
        _hash(row['history_hash'], semantic=True)
        _need(1 <= len(decisions) <= 24, 'Bounded complete trajectory required')
        for i, d in enumerate(decisions):
            _need(set(d) == {'observation_hash', 'action_id', 'reward', 'terminated'} and
                  type(d['terminated']) is bool and d['terminated'] is (i == len(decisions)-1),
                  'Exact complete decision fields required')
            _hash(d['observation_hash'], semantic=True)
            _need(type(d['action_id']) is str and type(d['reward']) in (float, int), 'Recorded scalar decision required')
            _need(d['action_id'] != 'STOP' or (i == len(decisions)-1 and d['reward'] == 0.), 'STOP0 only at end')
        _need(decisions[-1]['action_id'] == 'STOP' or len(decisions) == 24, 'Genuine STOP or horizon terminal required')
        values = recorded_action_targets(d['reward'] for d in decisions)
        target.extend(values); weights.extend([1. / (4 * 9 * len(decisions))] * len(decisions))
        stops.extend(d['action_id'] == 'STOP' for d in decisions)
        negative_preparation += sum(d['reward'] < 0 and q > 0 for d, q in zip(decisions, values))
        _need(set(trace) == {'key', 'context_hash', 'trace_seal', 'world', 'rows'} and
              trace['key'] == row['key'] and trace['world'] == world and trace['rows'] == len(decisions),
              'Original reconstructed trace bindings differ')
        _hash(trace['context_hash'], semantic=True); _hash(trace['trace_seal'], semantic=True)
    _need(all(counts[(s, 'teacher')] == 1 and counts[(s, 'RL')] == 8 for s in TRAIN_SUBJECTS),
          'Four teachers and32 completed RL trajectories required')
    _need(sum(len(r['decisions']) for r in manifest if r['kind'] == 'teacher') == 29 and len(target) == 497,
          'No zero-state or corpus denominator omissions')
    width = policy.actor[0].in_features
    wanted = {'features': ((497, width), np.dtype('<f4')), 'targets': ((497,), np.dtype('<f4')),
              'weights': ((497,), np.dtype('<f4')), 'stop_rows': ((497,), np.dtype('bool'))}
    for key, (shape, dtype) in wanted.items():
        array = arrays[key]
        _need(array.shape == shape and array.dtype == dtype and np.isfinite(array).all(),
              'Exact finite compact array shape/dtype required: ' + key)
    _need(arrays['features'].nbytes <= 4 * 1024**2 and metadata['feature_bytes'] == arrays['features'].nbytes,
          'Original bounded feature allocation differs')
    for key, expected in (('targets', np.asarray(target, np.float32)), ('weights', np.asarray(weights, np.float32)),
                          ('stop_rows', np.asarray(stops, np.bool_))):
        _need(np.array_equal(arrays[key], expected), 'Recomputed original ' + key + ' differs')
    _need(metadata['STOP_rows'] == sum(stops) and metadata['STOP_prediction'] == 'fixed_zero_no_head_gradient'
          and metadata['negative_immediate_positive_recorded_continuation'] == negative_preparation,
          'Original target/STOP summary differs')
    _need(metadata['costs']['policy_forwards_started'] == metadata['costs']['policy_forwards_completed'] == 497,
          'Original feature reconstruction cost must remain visible')
    _need(_seal(metadata, arrays) == binding['corpus_seal'], 'Original array/corpus seal changed')


def cache_binding(corpus, policy, *, origin):
    """Call while the genuine admitted corpus still exists; persist with evidence.

This descriptor is NOT self-authenticating. A later loader caller must obtain it
from root-approved original evidence, alongside an independently pinned ZIP SHA.
"""
    _need(type(corpus) is ContinuationCorpus, 'Exact original detached corpus required')
    corpus.require(policy)
    binding = freeze_json({'corpus_seal': corpus.seal, 'metadata_hash': semantic_digest(corpus.metadata),
        'trace_manifest_hash': corpus.metadata['manifest_hash'], 'base_parameter_hash': corpus.metadata['base_parameter_hash'],
        'architecture_hash': corpus.metadata['architecture_hash'], 'origin': _origin(origin)})
    _validate_payload(corpus.metadata, {k: getattr(corpus, k) for k in ARRAYS}, policy, binding)
    return binding


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'Duplicate cache JSON key')
        result[key] = value
    return result


def save_continuation_cache(corpus, policy, path, *, origin):
    """Fresh bounded parameter-style ZIP, no images, full policy or pickle."""
    started = time.perf_counter(); binding = cache_binding(corpus, policy, origin=origin)
    entries = []; payloads = {}
    for key in ARRAYS:
        stream = io.BytesIO()
        np.lib.format.write_array(stream, getattr(corpus, key), version=(1, 0), allow_pickle=False)
        raw = stream.getvalue(); entry = 'arrays/' + key + '.npy'; payloads[entry] = raw
        entries.append({'name': key, 'entry': entry, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'array_digest': array_digest(getattr(corpus, key))})
    metadata = {'schema': CACHE_VERSION, 'classification': 'simulator_derived_TRAIN_intermediate_no_pixels',
        'binding': binding, 'corpus_metadata': corpus.metadata, 'arrays': entries}
    raw_metadata = json.dumps(thaw_json(metadata), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _need(len(raw_metadata) <= MAX_METADATA_BYTES, 'Cache metadata byte cap')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        # Fixed member timestamps make identical cache payloads deterministic.
        for name, raw in [('manifest.json', raw_metadata), *payloads.items()]:
            archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), raw)
    payload = stream.getvalue()
    _need(0 < len(payload) <= MAX_BYTES, 'Cache file byte cap')
    corpus.require(policy)
    with Path(path).open('xb') as target:
        target.write(payload)
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
        'binding': thaw_json(binding), 'cache_write_seconds': time.perf_counter()-started,
        'cache_policy_forwards': 0, 'cache_native_previews': 0,
        'original_reconstruction_costs': thaw_json(binding['origin']['original_reconstruction_costs'])}


def load_continuation_cache(path, policy, *, expected_cache_sha256, expected_binding):
    """Return (exact original corpus bound to current frozen model, load costs).

There is no default for either expectation. Archive metadata cannot supply its
own approval. New objectives still need their own root-authorized head protocol;
loading this TRAIN intermediate never opens SELECT/EVAL or replays a patient.
"""
    started = time.perf_counter(); _hash(expected_cache_sha256); binding = _binding(expected_binding)
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BYTES, 'Bounded regular cache file required')
        with os.fdopen(fd, 'rb', closefd=False) as source:
            payload = source.read(MAX_BYTES + 1)
        after = os.fstat(fd)
        _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
              (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
              'Cache changed during bounded read')
    finally:
        os.close(fd)
    _need(len(payload) == before.st_size and hashlib.sha256(payload).hexdigest() == expected_cache_sha256,
          'Exact root-approved complete cache file hash required')
    _need(type(policy) is SpatialPolicy and policy.architecture_hash == binding['architecture_hash'] and
          parameter_hash(policy) == binding['base_parameter_hash'], 'Different frozen encoder/architecture')
    shape = {'features': (497, policy.actor[0].in_features), **{k: (497,) for k in ARRAYS[1:]}}
    dtype = {k: np.dtype('bool' if k == 'stop_rows' else '<f4') for k in ARRAYS}
    values = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist(); names = [i.filename for i in infos]
        _need(len(names) == 5 and set(names) == {'manifest.json', *('arrays/' + k + '.npy' for k in ARRAYS)}
              and all(i.compress_type == zipfile.ZIP_STORED and not i.flag_bits & 1 for i in infos)
              and sum(i.file_size for i in infos) <= MAX_BYTES
              and archive.getinfo('manifest.json').file_size <= MAX_METADATA_BYTES,
              'Exact bounded uncompressed cache inventory required')
        metadata = json.loads(archive.read('manifest.json'), object_pairs_hook=_json_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite cache JSON')))
        _need(set(metadata) == {'schema', 'classification', 'binding', 'corpus_metadata', 'arrays'} and
              metadata['schema'] == CACHE_VERSION and
              metadata['classification'] == 'simulator_derived_TRAIN_intermediate_no_pixels' and
              freeze_json(metadata['binding']) == binding, 'Original approval binding differs')
        entries = metadata['arrays']
        _need(len(entries) == 4 and [r['name'] for r in entries] == list(ARRAYS), 'Exact array order/count required')
        for row in entries:
            key = row['name']; entry = 'arrays/' + key + '.npy'
            _need(set(row) == {'name', 'entry', 'sha256', 'array_digest'} and row['entry'] == entry,
                  'Fixed array member required')
            data = archive.read(entry)
            _need(hashlib.sha256(data).hexdigest() == row['sha256'], 'Serialized array digest differs')
            reader = io.BytesIO(data)
            _need(np.lib.format.read_magic(reader) == (1, 0), 'Only NPY v1 accepted')
            observed_shape, fortran, observed_dtype = np.lib.format.read_array_header_1_0(reader, max_header_size=1024)
            _need(observed_shape == shape[key] and observed_dtype == dtype[key] and not fortran and
                  len(data)-reader.tell() == int(np.prod(shape[key])) * dtype[key].itemsize,
                  'Array shape/dtype/bytes differ before allocation')
            array = np.load(io.BytesIO(data), allow_pickle=False, max_header_size=1024)
            _need(np.isfinite(array).all() and array_digest(array) == row['array_digest'], 'Finite original array digest required')
            values[key] = immutable_array(array)
    original_metadata = freeze_json(metadata['corpus_metadata'])
    _validate_payload(original_metadata, values, policy, binding)
    # Only after every approved identity and payload join: a NEW in-memory guard.
    guard = _FrozenPolicy(policy)
    _need(guard.parameters == binding['base_parameter_hash'] and guard.architecture == binding['architecture_hash'],
          'Policy changed while verifying cache')
    corpus = ContinuationCorpus(*(values[k] for k in ARRAYS), original_metadata, binding['corpus_seal'], guard)
    corpus.require(policy)
    return corpus, {'cache_sha256': expected_cache_sha256, 'bytes_read': len(payload),
        'cache_load_seconds': time.perf_counter()-started, 'cache_policy_forwards': 0,
        'cache_native_previews': 0, 'cache_observations_constructed': 0,
        'original_reconstruction_costs': thaw_json(binding['origin']['original_reconstruction_costs']),
        'original_corpus_costs': thaw_json(original_metadata['costs']),
        'original_costs_replaced_or_amortized': False}
