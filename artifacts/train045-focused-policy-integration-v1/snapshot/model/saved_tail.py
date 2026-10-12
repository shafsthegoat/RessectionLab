"""Bounded parameter-only reload; external receipts, never archive self-admission.

The caller authenticates the original report/index and regular checkpoint bytes.
No corpus, source arrays, optimizer, training or deployment admission lives here.
"""
import copy
import hashlib
import io
import json
import zipfile

import numpy as np
import torch

from resectionlab.core import semantic_digest
from resectionlab.spatial_policy import parameter_hash
from resectionlab.spatial_continuation_value import _FrozenPolicy, TRAIN_SUBJECTS
from resectionlab.spatial_continuation_tail import FittedContinuationTail, TARGET, PREDICTION_SEMANTICS


def _json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate checkpoint JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite checkpoint metadata: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def load_saved_tail(policy, raw_checkpoint_bytes, *, expected_sha256, expected_manifest,
        tail_report, expected_report_hash, worlds, expected_worlds_hash,
        max_scores=24, seconds=600., cancelled=None):
    """Recreate exactly the old trained-world scorer from externally pinned data.

    expected_manifest/report/world hashes must come from the owning runner's
    authenticated immutable old result/index, NOT be derived from this archive.
    Focused deployment needs a separate capability in focused_policy.py.
    """
    raw = raw_checkpoint_bytes
    if (type(raw) is not bytes or not 0 < len(raw) <= 128 * 1024
            or hashlib.sha256(raw).hexdigest() != expected_sha256
            or semantic_digest(tail_report) != expected_report_hash
            or semantic_digest(worlds) != expected_worlds_hash
            or set(worlds) != set(TRAIN_SUBJECTS)
            or type(max_scores) is not int or not 1 <= max_scores <= 96
            or type(seconds) not in (int, float) or not 0 < seconds <= 3600):
        raise ValueError('External frozen checkpoint/report/world pins or caps differ')
    guard = _FrozenPolicy(policy)
    if (tail_report['base_parameter_hash'] != guard.parameters
            or tail_report['architecture_hash'] != guard.architecture):
        raise ValueError('Original encoder/architecture differs')
    head = copy.deepcopy(policy.actor)
    expected_shapes = {k: tuple(v.shape) for k, v in sorted(head.state_dict().items())}
    if set(expected_shapes) != {'0.bias', '0.weight', '2.bias', '2.weight'}:
        raise ValueError('Exact original actor MLP required')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = ['manifest.json', *('head/%03d.npy' % i for i in range(4))]
        if (len(infos) != 5 or [r.filename for r in infos] != names
                or any(r.compress_type != zipfile.ZIP_STORED or r.flag_bits & 1
                    or r.file_size != r.compress_size or r.file_size > 64 * 1024 for r in infos)
                or sum(r.file_size for r in infos) > 128 * 1024):
            raise ValueError('Exact bounded stored-only head archive required')
        manifest = _json(archive.read('manifest.json'))
        if (semantic_digest({k:v for k,v in manifest.items() if k != 'entries'}) != semantic_digest(expected_manifest)
                or set(manifest) != {'schema', 'method', 'objective', 'prediction_semantics',
                    'base', 'head_hash', 'corpus_seal', 'entries'}
                or manifest['schema'] != 'logged-continuation-head-only-v2'
                or manifest['method'] != 'TAIL' or manifest['objective'] != TARGET
                or manifest['prediction_semantics'] != PREDICTION_SEMANTICS
                or manifest['head_hash'] != tail_report['fitted_head_hash']
                or manifest['corpus_seal'] != tail_report['corpus_seal']
                or len(manifest['entries']) != 4):
            raise ValueError('Externally authenticated original tail declaration differs')
        state = {}
        for i, (key, shape) in enumerate(expected_shapes.items()):
            row = manifest['entries'][i]
            if (set(row) != {'name', 'member', 'sha256', 'shape'} or row['name'] != key
                    or row['member'] != names[i + 1] or tuple(row['shape']) != shape):
                raise ValueError('Exact tensor name/member/shape required')
            payload = archive.read(row['member'])
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Tensor bytes differ')
            stream = io.BytesIO(payload)
            if np.lib.format.read_magic(stream) != (1, 0):
                raise ValueError('Bounded NPY v1 required')
            got_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
            if (got_shape != shape or fortran or dtype.str != '<f4'
                    or len(payload) - stream.tell() != int(np.prod(shape)) * 4):
                raise ValueError('Finite exact float32 tensor layout required')
            array = np.frombuffer(payload, dtype=dtype, offset=stream.tell()).reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite saved head')
            state[key] = torch.from_numpy(array.copy())
    head.load_state_dict(state, strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False); parameter.grad = None
    if (parameter_hash(head) != tail_report['fitted_head_hash']
            or parameter_hash(head[0]) != tail_report['frozen_hidden_hash']):
        raise ValueError('Reloaded head/hidden bytes differ')
    guard.require()
    result = FittedContinuationTail(guard, head, worlds, tail_report,
        max_scores=max_scores, seconds=seconds, cancelled=cancelled)
    result.require()
    return result
