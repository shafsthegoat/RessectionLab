"""Bounded non-pickle goal/mode checkpoints, with explicit experiment lineage.

ZIP_STORED JSON + fixed-shape float32 NPY entries; never torch.load/pickle.
File integrity is not clinical or independently verified training validity.
"""
from __future__ import annotations
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import stat
import zipfile
import numpy as np
import torch

from .contact_learning_contract import (ContactExperiment, PROTOCOL, VERSION,
    expected_initial_parameter_hash, policy_config)
from .contact_learning import ContactLearningSession
from .core import freeze_json, semantic_digest, thaw_json
from .goal_mode_spatial_policy import CHECKPOINT_VERSION, GoalModeSpatialPolicy
from .spatial_policy import parameter_hash

MAX_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 512 * 1024
_LOADED = object()


@dataclass(frozen=True)
class VerifiedContactCheckpoint(Mapping):
    metadata: Mapping
    file_sha256: str
    _capability: object = field(repr=False)
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        if self._capability is not _LOADED:
            raise ValueError('Verified checkpoint metadata must come from bounded decoding')
        object.__setattr__(self, 'metadata', freeze_json(self.metadata))
        object.__setattr__(self, '_identity', semantic_digest({'metadata': self.metadata, 'file': self.file_sha256}))

    def __getitem__(self, key): return self.metadata[key]
    def __iter__(self): return iter(self.metadata)
    def __len__(self): return len(self.metadata)

    def require(self, experiment, *, kind):
        if (self._capability is not _LOADED or semantic_digest({'metadata': self.metadata, 'file': self.file_sha256}) != self._identity
                or self['architecture_hash'] != experiment.record()['architecture_hash']):
            raise ValueError('Verified checkpoint identity changed')
        _validate_lineage(self['lineage'], experiment, kind=kind)


def initial_lineage(experiment):
    if type(experiment) is not ContactExperiment: raise TypeError('Frozen contact experiment required')
    return {'kind': 'initial', 'method': 'COMMON_INITIALIZATION', 'experiment_hash': experiment.fingerprint,
        'family_hash': experiment.manifest['family_hash'], 'learning_contract_version': experiment.record()['version'],
        'optimizer_updates': 0, 'initial_parameter_hash': expected_initial_parameter_hash(experiment.policy_variant),
        'parameter_hash': expected_initial_parameter_hash(experiment.policy_variant), 'training_bindings': [],
        'training_status': 'not_started', 'real_patient_count': 0}


def final_lineage(session):
    if type(session) is not ContactLearningSession: raise TypeError('Exact learning session required')
    session.assert_intact()
    if (session.updates != session.experiment.protocol['updates']
            or session.updates != session._completed_updates
            or parameter_hash(session.policy) != session._expected_parameter_hash
            or not session._used_bindings):
        raise ValueError('Final checkpoint requires exactly the fixed completed admitted updates')
    session.experiment.assert_intact()
    return {'kind': 'final', 'method': session.method, 'experiment_hash': session.experiment.fingerprint,
        'family_hash': session.experiment.manifest['family_hash'], 'learning_contract_version': session.experiment.record()['version'],
        'optimizer_updates': session.updates, 'initial_parameter_hash': session.initial_parameter_hash,
        'parameter_hash': parameter_hash(session.policy),
        'training_bindings': list(session._used_bindings.values()),
        'training_status': 'completed_fixed_endpoint', 'real_patient_count': 0}


def partial_lineage(session):
    if type(session) is not ContactLearningSession: raise TypeError('Exact learning session required')
    session.assert_intact()
    if (session.updates != session._completed_updates
            or not 0 <= session.updates <= session.experiment.protocol['updates']
            or parameter_hash(session.policy) != session._expected_parameter_hash):
        raise ValueError('Partial checkpoint requires the actual unchanged admitted session state')
    return {'kind': 'partial', 'method': session.method, 'experiment_hash': session.experiment.fingerprint,
        'family_hash': session.experiment.manifest['family_hash'], 'learning_contract_version': session.experiment.record()['version'],
        'optimizer_updates': session.updates, 'initial_parameter_hash': session.initial_parameter_hash,
        'parameter_hash': parameter_hash(session.policy), 'training_bindings': list(session._used_bindings.values()),
        'training_status': 'incomplete_at_fixed_cap', 'real_patient_count': 0}


def _validate_lineage(lineage, experiment, *, kind):
    if type(experiment) is not ContactExperiment: raise TypeError('Exact frozen experiment required')
    experiment.assert_intact()
    expected_keys = {'kind', 'method', 'experiment_hash', 'family_hash', 'learning_contract_version',
        'optimizer_updates', 'initial_parameter_hash', 'parameter_hash', 'training_bindings',
        'training_status', 'real_patient_count'}
    if (set(lineage) != expected_keys or lineage['kind'] != kind or kind not in ('initial', 'partial', 'final')
            or lineage['experiment_hash'] != experiment.fingerprint
            or lineage['family_hash'] != experiment.manifest['family_hash']
            or lineage['learning_contract_version'] != experiment.record()['version'] or type(lineage['real_patient_count']) is not int
            or lineage['real_patient_count'] != 0 or type(lineage['optimizer_updates']) is not int
            or lineage['initial_parameter_hash'] != expected_initial_parameter_hash(experiment.policy_variant)):
        raise ValueError('Checkpoint training lineage differs from frozen generated experiment')
    if kind == 'initial':
        if semantic_digest(lineage) != semantic_digest(initial_lineage(experiment)):
            raise ValueError('Initial checkpoint is not the common scratch initialization')
    else:
        if kind == 'partial':
            if (lineage['method'] not in experiment.protocol['methods'] or not 0 <= lineage['optimizer_updates'] <= experiment.protocol['updates']
                    or lineage['training_status'] != 'incomplete_at_fixed_cap'
                    or (lineage['optimizer_updates'] > 0 and not lineage['training_bindings'])):
                raise ValueError('Partial checkpoint lacks admitted completed-update lineage')
        elif (lineage['method'] not in experiment.protocol['methods'] or lineage['optimizer_updates'] != experiment.protocol['updates']
                or lineage['training_status'] != 'completed_fixed_endpoint' or not lineage['training_bindings']):
            raise ValueError('Final checkpoint is incomplete or has no TRAIN lineage')
        from .contact_learning_contract import ContactTrainBinding
        from .public_surface_contact import SurfaceContactDevelopmentContext
        seen = set()
        for record in lineage['training_bindings']:
            context = SurfaceContactDevelopmentContext(**dict(record['context']))
            binding = ContactTrainBinding(experiment, record['layout_id'], record['goal_id'], context)
            if semantic_digest(record) != semantic_digest(binding.record()) or binding.fingerprint in seen:
                raise ValueError('Checkpoint contains substituted or duplicate training provenance')
            seen.add(binding.fingerprint)


def encode_contact_checkpoint(policy, experiment, lineage):
    if type(policy) is not experiment.policy_type or policy.architecture_hash != experiment.record()['architecture_hash']:
        raise ValueError('Only the new exact goal/mode architecture is a contact checkpoint')
    lineage = freeze_json(lineage); _validate_lineage(lineage, experiment, kind=lineage['kind'])
    if lineage['parameter_hash'] != parameter_hash(policy): raise ValueError('Lineage parameter hash changed')
    entries, payloads = [], {}
    for i, (name, tensor) in enumerate(sorted(policy.state_dict().items())):
        array = tensor.detach().cpu().contiguous().numpy()
        if array.dtype != np.float32 or not np.isfinite(array).all(): raise ValueError('Finite float32 model tensors required')
        stream = io.BytesIO(); np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
        payload = stream.getvalue(); filename = f'tensors/{i:04d}.npy'
        payloads[filename] = payload
        entries.append({'name': name, 'entry': filename, 'shape': list(array.shape), 'dtype': '<f4',
                        'sha256': hashlib.sha256(payload).hexdigest()})
    metadata = {'version': policy.checkpoint_identity()['version'], 'architecture': policy.architecture_record(),
        'architecture_hash': policy.architecture_hash, 'parameter_hash': parameter_hash(policy),
        'lineage': lineage, 'tensors': entries, 'format': 'bounded_zip_stored_json_float32_npy_v1'}
    metadata_bytes = json.dumps(thaw_json(freeze_json(metadata)), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(metadata_bytes) > MAX_JSON_BYTES: raise ValueError('Checkpoint metadata exceeds bound')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json', metadata_bytes)
        for name, payload in payloads.items(): archive.writestr(name, payload)
    payload = stream.getvalue()
    if len(payload) > MAX_BYTES: raise ValueError('Checkpoint exceeds fixed read bound')
    return payload


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate checkpoint JSON key')
        result[key] = value
    return result


def decode_contact_checkpoint(payload, *, expected_sha256, experiment, kind):
    if (type(payload) is not bytes or not 0 < len(payload) <= MAX_BYTES
            or hashlib.sha256(payload).hexdigest() != expected_sha256):
        raise ValueError('Checkpoint bytes/hash differ from expected bounded artifact')
    with torch.random.fork_rng(devices=[]): model = experiment.policy_type(policy_config())
    expected = model.state_dict()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist(); names = [info.filename for info in infos]
        if (len(names) != len(set(names)) or len(names) != len(expected)+1
                or any(i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1 for i in infos)
                or sum(i.file_size for i in infos) > MAX_BYTES or 'manifest.json' not in names
                or archive.getinfo('manifest.json').file_size > MAX_JSON_BYTES):
            raise ValueError('Invalid bounded checkpoint archive inventory')
        metadata = json.loads(archive.read('manifest.json'), object_pairs_hook=_unique_json)
        if (set(metadata) != {'version', 'architecture', 'architecture_hash', 'parameter_hash', 'lineage', 'tensors', 'format'}
                or metadata['version'] != model.checkpoint_identity()['version']
                or metadata['format'] != 'bounded_zip_stored_json_float32_npy_v1'
                or metadata['architecture_hash'] != model.architecture_hash
                or semantic_digest(metadata['architecture']) != semantic_digest(model.architecture_record())):
            raise ValueError('Unsupported checkpoint version/architecture; legacy aspiration weights refused')
        _validate_lineage(metadata['lineage'], experiment, kind=kind)
        rows = metadata['tensors']
        if (len(rows) != len(expected) or [r['name'] for r in rows] != sorted(expected)
                or [r['entry'] for r in rows] != [f'tensors/{i:04d}.npy' for i in range(len(expected))]
                or set(names) != {'manifest.json', *(r['entry'] for r in rows)}):
            raise ValueError('Missing, duplicate or unexpected checkpoint tensor')
        values = {}
        for row in rows:
            if set(row) != {'name', 'entry', 'shape', 'dtype', 'sha256'}: raise ValueError('Unexpected tensor metadata')
            data = archive.read(row['entry'])
            if hashlib.sha256(data).hexdigest() != row['sha256']: raise ValueError('Tensor digest changed')
            reader = io.BytesIO(data)
            if np.lib.format.read_magic(reader) != (1, 0): raise ValueError('Only bounded NPY v1 headers accepted')
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(reader, max_header_size=1024)
            wanted = tuple(expected[row['name']].shape)
            if (shape != wanted or row['shape'] != list(wanted) or dtype != np.dtype('<f4')
                    or row['dtype'] != '<f4' or fortran
                    or len(data)-reader.tell() != int(np.prod(wanted))*4):
                raise ValueError('Tensor header/shape/dtype differs before allocation')
            array = np.frombuffer(data, dtype='<f4', offset=reader.tell()).reshape(wanted).copy()
            if not np.isfinite(array).all(): raise ValueError('Nonfinite checkpoint tensor')
            values[row['name']] = torch.from_numpy(array)
    model.load_state_dict(values, strict=True)
    if parameter_hash(model) != metadata['parameter_hash'] or metadata['parameter_hash'] != metadata['lineage']['parameter_hash']:
        raise ValueError('Decoded parameter identity differs from checkpoint/lineage')
    model.eval()
    return model, VerifiedContactCheckpoint(metadata, expected_sha256, _LOADED)


def save_contact_checkpoint(path, policy, experiment, lineage):
    payload = encode_contact_checkpoint(policy, experiment, lineage)
    path = Path(path)
    # A fresh owned output path; no overwrite/retry of previous evidence.
    with path.open('xb') as target: target.write(payload); target.flush(); os.fsync(target.fileno())
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(),
            'bytes': len(payload), 'parameter_hash': parameter_hash(policy), 'lineage': thaw_json(freeze_json(lineage))}


def load_contact_checkpoint(path, *, expected_sha256, experiment, kind):
    # Refuse a FIFO/device after opening without waiting for a peer writer.
    # Nonblocking mode has no effect on the bounded regular-file read below.
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            raise ValueError('Checkpoint must be a bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as source: payload = source.read(MAX_BYTES+1)
        after = os.fstat(fd)
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Checkpoint changed during bounded read')
    finally: os.close(fd)
    return decode_contact_checkpoint(payload, expected_sha256=expected_sha256, experiment=experiment, kind=kind)
