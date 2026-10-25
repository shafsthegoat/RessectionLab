"""Disposable authenticated disk packages of read-only reconstructed actions.

This namespace has no canonical/archive or editable replay authority. Only the
private prepared-prefix publisher may store complete compressed receipts and
states. The store never decodes state blobs or executes/initializes an engine.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import MappingProxyType

from .surgical_verified_cache import VerifiedSurgicalCache, _read
from .surgical_flow_display import validate_flow_display, trim_frame_flows

SCHEMA = 'resectionlab.prepared-prefix-action.v1'
BINDING_SCHEMA = 'resectionlab.prepared-prefix-binding.v1'
DIRECTORY_ENV = 'RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY'
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / 'testcases/outputs/.prepared-prefix-cache'
MAX_DISK_BYTES = 512 * 1024**2
MAX_ACTION_BYTES = 64 * 1024**2
MAX_MANIFEST_BYTES = 1024**2
MAX_METADATA_NODES = 65536
MAX_TREE_ITEMS = 16384
_BINDING_KEYS = frozenset({'schema', 'sceneId', 'originSnapshotHash', 'actionLogHash',
    'protectionContractHash', 'implementation', 'prefixImplementation'})
_PAYLOAD_KEYS = frozenset({'actionHash', 'beforeStateId', 'afterStateId', 'receipt', 'checkpoint', 'frames'})
_PUBLISH_AUTHORITY = object()
_DOMAIN = b'resectionlab.prepared-prefix-action.v1\0'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return type(value) is str and len(value.removeprefix('sha256:')) == 64 and all(
        char in '0123456789abcdef' for char in value.removeprefix('sha256:'))


def _binding(value, index):
    _need(type(index) is int and 0 <= index < 2**63, 'Invalid prepared action ordinal')
    _need(type(value) is dict and all(type(key) is str for key in value) and set(value) == _BINDING_KEYS, 'Invalid prepared binding keys')
    _need(type(value['schema']) is str and value['schema'] == BINDING_SCHEMA, 'Invalid prepared binding schema')
    _need(all(_digest(value[key]) for key in _BINDING_KEYS - {'schema'}), 'Invalid prepared binding identity')
    return dict(value)


def _json(value, *, _budget=None):
    """Detach strict small metadata with a conservative ASCII allocation cap."""
    active = set()
    budget = [MAX_MANIFEST_BYTES, 0] if _budget is None else _budget
    def charge(amount):
        _need(amount <= budget[0], 'Prepared metadata exceeds manifest budget')
        budget[0] -= amount
    def copy(item, depth=0):
        budget[1] += 1
        _need(budget[1] <= MAX_METADATA_NODES and depth <= 64, 'Prepared metadata structure budget exceeded')
        kind = type(item)
        if kind is str:
            charge(12 * len(item) + 2); return item
        if item is None or kind is bool:
            charge(5); return item
        if kind is int:
            _need(-2**63 <= item < 2**63, 'Prepared metadata integer exceeds int64')
            charge(21); return item
        if kind is float:
            _need(math.isfinite(item), 'Nonfinite prepared metadata')
            charge(32); return item
        _need(kind in (dict, list, tuple, MappingProxyType), 'Unsupported prepared metadata type')
        _need(id(item) not in active, 'Cyclic prepared metadata')
        active.add(id(item)); charge(2 + (2 if kind in (dict, MappingProxyType) else 1) * len(item))
        try:
            if kind in (dict, MappingProxyType):
                _need(all(type(key) is str for key in item), 'Prepared metadata keys must be exact strings')
                return {copy(key, depth + 1): copy(child, depth + 1) for key, child in item.items()}
            return [copy(child, depth + 1) for child in item]
        finally:
            active.remove(id(item))
    return copy(value)


def _pack(payload, index, max_action_bytes, scene_id=None):
    _need(type(payload) is dict and all(type(key) is str for key in payload) and set(payload) == _PAYLOAD_KEYS, 'Invalid prepared payload keys')
    _need(all(_digest(payload[name]) for name in ('actionHash', 'beforeStateId', 'afterStateId')),
          'Invalid prepared action/state identity')
    chunks, descriptors, by_hash = [], [], {}
    total = 0
    def blob(value):
        nonlocal total
        _need(type(value) is bytes, 'Prepared binary fields must be exact bytes')
        digest = hashlib.sha256(value).hexdigest()
        if digest in by_hash:
            number = by_hash[digest]
            _need(chunks[number] == value, 'Prepared content digest collision')
            return number
        _need(total + len(value) <= max_action_bytes, 'Prepared binary action budget exceeded')
        number = len(chunks); by_hash[digest] = number
        descriptors.append({'offset': total, 'bytes': len(value), 'sha256': digest})
        chunks.append(value); total += len(value)
        return number
    receipt = payload['receipt']
    _need(type(receipt) is tuple and len(receipt) == 3 and type(receipt[1]) is int
          and 0 <= receipt[1] < 2**63, 'Invalid complete compressed prepared receipt')
    receipt_ref = [blob(receipt[0]), receipt[1], blob(receipt[2])]
    checkpoint = payload['checkpoint']
    _need(type(checkpoint) is dict and all(type(key) is str for key in checkpoint)
          and set(checkpoint) == {'position', 'stateId', 'state'}
          and type(checkpoint['position']) is int and checkpoint['position'] == index + 1
          and _digest(checkpoint['stateId']) and checkpoint['stateId'] == payload['afterStateId'], 'Prepared checkpoint boundary differs')
    checkpoint_ref = {'position': checkpoint['position'], 'stateId': checkpoint['stateId'],
                      'state': blob(checkpoint['state'])}
    frames = payload['frames']
    _need(type(frames) is list and 1 <= len(frames) <= MAX_METADATA_NODES, 'Invalid prepared frames')
    # Validate every optional block before any metadata-budget fallback can
    # discard samples. Generic disk admission has no grid; the prefix owner
    # validates native bounds/adjacency against its admitted scene on load.
    for ordinal, frame in enumerate(frames):
        _need(type(frame) is dict, 'Invalid prepared frame descriptor')
        if 'flowDisplay' in frame:
            flow = frame['flowDisplay']
            _need(type(flow) is dict, 'Invalid prepared flow display')
            previous = frames[max(0, ordinal - 1)]
            validate_flow_display(flow, scene_id=scene_id if scene_id is not None else flow.get('sceneId'),
                action_index=index, frame_index=ordinal, before_state_id=previous['stateId'],
                state_id=frame['stateId'], start_seconds=previous['observedTimeSeconds'],
                end_seconds=frame['observedTimeSeconds'])
    frame_refs = []
    metadata_budget = [MAX_MANIFEST_BYTES, 0]
    for frame in frames:
        _need(type(frame) is dict and all(type(key) is str for key in frame)
              and {'state', 'stateId', 'fieldTokens', 'effectiveActivation'} <= frame.keys(),
              'Invalid prepared frame descriptor')
        _need(_digest(frame['stateId']) and type(frame['effectiveActivation']) is dict
              and type(frame['fieldTokens']) in (dict, MappingProxyType), 'Invalid prepared frame metadata')
        row = _json({key: value for key, value in frame.items() if key != 'state'}, _budget=metadata_budget)
        row['state'] = blob(frame['state']); frame_refs.append(row)
    _need(frames[0]['stateId'] == payload['beforeStateId'] and frames[-1]['stateId'] == payload['afterStateId'],
          'Prepared frame endpoint ancestry differs')
    metadata = {name: payload[name] for name in ('actionHash', 'beforeStateId', 'afterStateId')}
    metadata.update(receipt=receipt_ref, checkpoint=checkpoint_ref, frames=frame_refs)
    return metadata, descriptors, chunks, total


def _parents(path, *, create=False):
    """Never resolve through a symlink, including above the namespace root."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for name in path.parts[1:]:
        current = current / name
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            current.mkdir(mode=0o700)
            info = current.lstat()
        _need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
              'Prepared cache ancestor must be a real directory')
    return path


def _regular(path, *, mode=None):
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Prepared cache leaf must be a single regular file')
    if mode is not None:
        _need(stat.S_IMODE(info.st_mode) == mode, 'Prepared cache leaf permission differs')
        _need(not hasattr(os, 'getuid') or info.st_uid == os.getuid(), 'Prepared cache leaf owner differs')
    return info


class _StoreLock:
    """Plain provenance-sealed methods; optional cache contention never waits."""
    def __init__(self, store):
        self.store = store
        self._fd = None

    def __enter__(self):
        import fcntl
        key = self.store._ready()
        path = self.store._namespace / 'lock'
        fd = os.open(path, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        try:
            info = os.fstat(fd)
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o600
                  and (not hasattr(os, 'getuid') or info.st_uid == os.getuid()), 'Invalid prepared lock descriptor')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = _regular(path, mode=0o600)
            _need((current.st_dev, current.st_ino) == (info.st_dev, info.st_ino), 'Prepared lock identity changed')
            _parents(self.store._namespace)
            self.store._scan()
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return key

    def __exit__(self, kind, error, traceback):
        import fcntl
        fd = self._fd
        try:
            if kind is None:
                _parents(self.store._namespace)
                self.store._scan()
        finally:
            self._fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return False


class PreparedPrefixStore:
    """Separate optional packages; directory is the base, not the namespace.

    A load may coexist with a64MiB package plus at most64MiB detached slices and
    <=1MiB manifest text and an O(1MiB) bounded JSON parse graph; accepted
    metadata has<=65536 nodes. There is no decoded
    state graph or retained in-memory package cache. These transient allowances
    are separate from the512MiB disk cap and the caller's whole-process budget.
    """
    def __init__(self, directory=None, *, max_disk_bytes=MAX_DISK_BYTES, max_action_bytes=MAX_ACTION_BYTES):
        if type(max_disk_bytes) is not int or not 1 <= max_disk_bytes <= MAX_DISK_BYTES:
            raise ValueError('Prepared disk budget must be1..512MiB')
        if type(max_action_bytes) is not int or not 1 <= max_action_bytes <= MAX_ACTION_BYTES:
            raise ValueError('Prepared action budget must be1..64MiB')
        selected = (os.environ.get(DIRECTORY_ENV) or DEFAULT_DIRECTORY) if directory is None else directory
        self.directory = Path(os.path.abspath(selected))
        self._namespace = self.directory / 'prepared-prefix-v1'
        self._owner = VerifiedSurgicalCache(self._namespace, max_disk_bytes=max_disk_bytes)
        self.max_disk_bytes, self.max_action_bytes = max_disk_bytes, max_action_bytes
        self.hits = self.misses = self.publications = 0
        self.lastSkip = None

    def _scan(self):
        _parents(self._namespace)
        count, total = 0, 0
        pending = [(self._namespace, 0)]
        while pending:
            path, depth = pending.pop()
            _need(depth <= 3, 'Prepared namespace directory depth exceeded')
            with os.scandir(path) as entries:
                for entry in entries:
                    count += 1; _need(count <= MAX_TREE_ITEMS, 'Prepared namespace item budget exceeded')
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append((Path(entry.path), depth + 1))
                    else:
                        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                              'Prepared namespace contains redirected or nonregular remnants')
                        total += info.st_size
        return total

    def _ready(self):
        _need(type(self.max_disk_bytes) is int and 1 <= self.max_disk_bytes <= MAX_DISK_BYTES
              and self._owner.max_disk_bytes == self.max_disk_bytes
              and type(self.max_action_bytes) is int and 1 <= self.max_action_bytes <= MAX_ACTION_BYTES,
              'Prepared cache budget changed outside hard bounds')
        _parents(self._namespace, create=True)
        self._scan()
        key = self._owner._key()
        _regular(self._namespace / 'key', mode=0o600)
        lock = self._namespace / 'lock'
        if not lock.exists() and not lock.is_symlink():
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            os.close(fd)
        _regular(lock, mode=0o600)
        return key

    def _locked(self):
        return _StoreLock(self)

    def _entry(self, binding, index):
        raw = json.dumps({'schema': SCHEMA, 'binding': binding, 'index': index},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
        return self._namespace / ('entry-' + hashlib.sha256(_DOMAIN + raw).hexdigest())

    @property
    def statistics(self):
        try:
            _parents(self._namespace)
            disk_bytes = self._scan()
        except FileNotFoundError:
            disk_bytes = 0
        except Exception as error:
            disk_bytes = self.max_disk_bytes + 1
            self.lastSkip = str(error)
        return {'diskBytes': disk_bytes, 'maxDiskBytes': self.max_disk_bytes,
                'maxActionBytes': self.max_action_bytes, 'hits': self.hits, 'misses': self.misses,
                'publications': self.publications, 'lastSkip': self.lastSkip,
                'directory': str(self.directory), 'namespaceDirectory': str(self._namespace),
                'maxTransientBinaryBytes': 2*self.max_action_bytes,
                'maxManifestBytes': MAX_MANIFEST_BYTES, 'maxMetadataNodes': MAX_METADATA_NODES}

    def _manifest(self, entry, binding, index, key):
        _parents(entry)
        for name in ('manifest.json', 'manifest.mac', 'payload.bin'):
            _regular(entry / name)
        raw = _read(entry / 'manifest.json', MAX_MANIFEST_BYTES)
        signature = _read(entry / 'manifest.mac', 64)
        _need(len(raw) <= MAX_MANIFEST_BYTES and len(signature) == 64 and hmac.compare_digest(
            signature, hmac.digest(key, _DOMAIN + raw, 'sha256').hex().encode('ascii')),
            'Prepared manifest authentication differs')
        manifest = json.loads(raw)
        _need(type(manifest) is dict and set(manifest) == {'schema', 'binding', 'index', 'payload',
            'blobs', 'binaryBytes', 'binarySha256'}, 'Prepared manifest structure differs')
        _need(manifest['schema'] == SCHEMA and manifest['binding'] == binding
              and type(manifest['index']) is int and manifest['index'] == index,
              'Prepared manifest binding or ordinal differs')
        _json(manifest)
        size = manifest['binaryBytes']
        _need(type(size) is int and 0 <= size <= self.max_action_bytes and _digest(manifest['binarySha256']),
              'Prepared binary length or digest exceeds budget')
        _need(_regular(entry / 'payload.bin').st_size == size, 'Prepared binary file length differs')
        _need(type(manifest['blobs']) is list, 'Prepared binary descriptors differ')
        offset = 0
        for item in manifest['blobs']:
            _need(type(item) is dict and set(item) == {'offset','bytes','sha256'}
                  and type(item['offset']) is int and type(item['bytes']) is int
                  and item['offset'] == offset and 0 <= item['bytes'] <= size-offset
                  and _digest(item['sha256']), 'Prepared binary descriptor exceeds bounds')
            offset += item['bytes']
        _need(offset == size, 'Prepared binary has unbound trailing bytes')
        # Validate reference types and complete receipt/endpoint shape without
        # decoding or allocating any state bytes. Empty placeholders are used
        # only for structural admission; content is verified separately below.
        def placeholder(number):
            _need(type(number) is int and 0 <= number < len(manifest['blobs']),
                  'Prepared part reference exceeds bounds')
            return b''
        _pack(self._unpack(manifest['payload'], placeholder), index, self.max_action_bytes, binding['sceneId'])
        return raw, manifest

    @staticmethod
    def _unpack(stored, part):
        _need(type(stored) is dict and set(stored) == _PAYLOAD_KEYS, 'Prepared stored payload differs')
        receipt = stored['receipt']
        _need(type(receipt) is list and len(receipt) == 3, 'Prepared stored receipt differs')
        result = {name: stored[name] for name in ('actionHash','beforeStateId','afterStateId')}
        result['receipt'] = (part(receipt[0]), receipt[1], part(receipt[2]))
        checkpoint = stored['checkpoint']
        _need(type(checkpoint) is dict and set(checkpoint) == {'position','stateId','state'},
              'Prepared stored checkpoint differs')
        result['checkpoint'] = {**checkpoint, 'state': part(checkpoint['state'])}
        _need(type(stored['frames']) is list, 'Prepared stored frames differ')
        result['frames'] = [{**frame, 'state': part(frame['state'])} for frame in stored['frames']]
        return result

    def _existing(self, entry, binding, index, key):
        """Full bounded authentication, with only64KiB binary read scratch."""
        raw, manifest = self._manifest(entry, binding, index, key)
        fd = os.open(entry / 'payload.bin', os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                     | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            expected = _regular(entry / 'payload.bin')
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                  and (info.st_dev, info.st_ino, info.st_size) ==
                      (expected.st_dev, expected.st_ino, manifest['binaryBytes']),
                  'Prepared binary descriptor identity differs')
            complete = hashlib.sha256()
            for item in manifest['blobs']:
                digest = hashlib.sha256(); remaining = item['bytes']
                while remaining:
                    block = stream.read(min(65536, remaining))
                    _need(bool(block), 'Prepared binary ended early')
                    digest.update(block); complete.update(block); remaining -= len(block)
                _need(digest.hexdigest() == item['sha256'], 'Prepared part digest differs')
            _need(not stream.read(1) and complete.hexdigest() == manifest['binarySha256'],
                  'Prepared complete binary digest differs')
        return raw

    def publish_action(self, binding, index, payload, *, _authority):
        pending = None
        try:
            _need(_authority is _PUBLISH_AUTHORITY, 'Prepared publication requires private prefix authority')
            binding = _binding(binding, index)
            candidate = payload
            for mode in (None, False, True):
                try:
                    metadata, blobs, chunks, binary_bytes = _pack(candidate, index, self.max_action_bytes, binding['sceneId'])
                    binary_hash = hashlib.sha256()
                    for chunk in chunks: binary_hash.update(chunk)
                    manifest = _json({'schema': SCHEMA, 'binding': binding, 'index': index,
                        'payload': metadata, 'blobs': blobs, 'binaryBytes': binary_bytes,
                        'binarySha256': binary_hash.hexdigest()})
                    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
                    _need(len(raw) <= MAX_MANIFEST_BYTES, 'Prepared manifest byte budget exceeded')
                    break
                except ValueError as error:
                    if mode is True or 'budget' not in str(error).lower(): raise
                    candidate = {**payload, 'frames': [dict(frame) for frame in payload['frames']]}
                    trim_frame_flows(candidate['frames'], omit=mode is False)
            incoming = binary_bytes + len(raw) + 64
            with self._locked() as key:
                entry = self._entry(binding, index)
                existing = entry.exists() or entry.is_symlink()
                if existing:
                    try:
                        previous = self._existing(entry, binding, index, key)
                    except (ValueError, FileNotFoundError):
                        previous = None
                    if previous is not None:
                        _need(previous == raw, 'Prepared authenticated determinism conflict')
                        self.lastSkip = None
                        return True
                _need(self._owner._room(incoming, entry), 'Prepared disk quota exhausted (including remnants)')
                self._scan()
                pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=self._namespace))
                _need(pending.parent == self._namespace and pending.name.startswith('.pending-'),
                      'Prepared staging directory differs')
                _parents(pending)
                pending.chmod(0o700)
                for filename, pieces in (('payload.bin', chunks), ('manifest.json', (raw,)),
                                         ('manifest.mac', (hmac.digest(key, _DOMAIN + raw, 'sha256').hex().encode('ascii'),))):
                    _parents(pending)
                    fd = os.open(pending / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                 getattr(os, 'O_NOFOLLOW', 0), 0o600)
                    with os.fdopen(fd, 'wb') as stream:
                        for piece in pieces: stream.write(piece)
                        stream.flush(); os.fsync(stream.fileno())
                _need(self._scan() <= self.max_disk_bytes, 'Prepared staged quota exceeded')
                _parents(pending); _parents(self._namespace)
                if existing:
                    # Only a fully inspected corrupt disposable package reaches
                    # this point. Preserve it until fresh bytes are fsynced and
                    # quota has counted BOTH packages. A later failure is a miss.
                    _parents(entry); self._scan()
                    shutil.rmtree(entry)
                os.replace(pending, entry); pending = None
            self.publications += 1; self.lastSkip = None
            return True
        except Exception as error:
            self.lastSkip = str(error)
            return False
        finally:
            if pending is not None:
                try:
                    _parents(pending)
                    shutil.rmtree(pending)
                except Exception:
                    pass

    def load_action(self, binding, index):
        try:
            binding = _binding(binding, index)
            with self._locked() as key:
                _need(self._scan() <= self.max_disk_bytes, 'Prepared disk quota exceeded')
                entry = self._entry(binding, index)
                raw, manifest = self._manifest(entry, binding, index, key)
                size = manifest['binaryBytes']
                binary = _read(entry / 'payload.bin', self.max_action_bytes)
                _need(len(binary) == size and hashlib.sha256(binary).hexdigest() == manifest['binarySha256'],
                      'Prepared binary digest/length differs')
                values, offset = [], 0
                _need(type(manifest['blobs']) is list, 'Prepared binary descriptors differ')
                for item in manifest['blobs']:
                    _need(type(item) is dict and set(item) == {'offset','bytes','sha256'}
                          and type(item['offset']) is int and type(item['bytes']) is int
                          and item['offset'] == offset and 0 <= item['bytes'] <= size-offset
                          and _digest(item['sha256']), 'Prepared binary descriptor exceeds bounds')
                    value = binary[offset:offset+item['bytes']]
                    _need(hashlib.sha256(value).hexdigest() == item['sha256'], 'Prepared part digest differs')
                    values.append(value); offset += item['bytes']
                _need(offset == size, 'Prepared binary has unbound trailing bytes')
                def part(number):
                    _need(type(number) is int and 0 <= number < len(values), 'Prepared part reference exceeds bounds')
                    return values[number]
                result = self._unpack(manifest['payload'], part)
                # Validate the same complete public package shape without state
                # decompression, then return detached descriptor containers.
                _pack(result, index, self.max_action_bytes, binding['sceneId'])
            self.hits += 1; self.lastSkip = None
            return result
        except Exception as error:
            self.misses += 1; self.lastSkip = str(error)
            return None
