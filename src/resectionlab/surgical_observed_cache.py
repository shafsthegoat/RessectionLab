"""Private, disposable field-deduplicated cache of verified observed frames.

Only a public-action replay whose receipt and endpoint match a canonical
admission can publish. The native source/archive is never changed or trusted
because of this cache. Unavailable, stale or corrupt cache data is a miss.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import hashlib
import hmac
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .core import freeze_json, thaw_json, semantic_digest
from .surgical_verified_cache import (VerifiedSurgicalCache, STATE_SCHEMA, MAX_JSON_BYTES,
    _encode, _encode_sized, _decode, _restore_state, _state_overhead, _read,
    implementation_fingerprint, _MODULES, _code_signature)

SCHEMA = 'resectionlab.observed-surgical-frames.v1'
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / 'testcases/outputs/.observed-surgical-frames'
_AUTHORITY = object()
RECEIPT_FIELDS = ('status', 'reason', 'reasonCode', 'completedFraction', 'events', 'deltas')


class ImplementationChanged(ValueError):
    pass


def receipt_hash(value):
    return semantic_digest({field: value.get(field) for field in RECEIPT_FIELDS})


def _default_identity(value):
    if inspect.isfunction(value):
        return ['function', value.__module__, value.__qualname__, _code_signature(value.__code__)]
    if isinstance(value, dict): return {str(k): _default_identity(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_default_identity(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return ['type', type(value).__module__, type(value).__qualname__, repr(value)]


def observed_implementation_fingerprint():
    base = implementation_fingerprint()
    if base is None: return None
    digest = hashlib.sha256(base.encode())
    for name in (*_MODULES, 'surgical_frame_replay', 'surgical_observed_cache', 'surgical_telemetry'):
        module = importlib.import_module('resectionlab.' + name)
        for key, value in sorted(vars(module).items()):
            if key.isupper() and isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                try: digest.update(semantic_digest([name, key, value]).encode())
                except (TypeError, ValueError): pass
            functions = [(key, value)] if inspect.isfunction(value) and value.__module__ == module.__name__ else []
            if inspect.isclass(value) and value.__module__ == module.__name__:
                for member, method in sorted(vars(value).items()):
                    if isinstance(method, (staticmethod, classmethod)): method = method.__func__
                    elif isinstance(method, property): method = method.fget
                    if inspect.isfunction(method): functions.append((key + '.' + member, method))
            for member, function in functions:
                digest.update((name + ':' + member).encode())
                digest.update(json.dumps(_code_signature(function.__code__), separators=(',', ':')).encode())
                digest.update(semantic_digest(_default_identity([function.__defaults__, function.__kwdefaults__])).encode())
    return digest.hexdigest()


def light_implementation_seal():
    """Check source bytes, loaded code/defaults/constants and binary identities.

    The expensive numerical-build byte fingerprint is checked at admission and
    publication. This seal still detects ordinary edits/monkeypatches between
    frames; no archive, history or numerical binary is re-encoded per frame.
    """
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    for name in (*_MODULES, 'surgical_frame_replay', 'surgical_observed_cache', 'surgical_telemetry'):
        module = importlib.import_module('resectionlab.' + name)
        functions = []
        for key, value in sorted(vars(module).items()):
            if inspect.isfunction(value): functions.append((key, value))
            elif inspect.isclass(value) and value.__module__ == module.__name__:
                for member, method in sorted(vars(value).items()):
                    if isinstance(method, (staticmethod, classmethod)): method = method.__func__
                    elif isinstance(method, property): method = method.fget
                    if inspect.isfunction(method): functions.append((key + '.' + member, method))
            if key.isupper() and isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                digest.update(repr((name, key, value)).encode())
        for key, function in functions:
            digest.update(repr((name, key, id(function), id(function.__code__),
                function.__defaults__, function.__kwdefaults__)).encode())
    for name in ('numpy._core._multiarray_umath', 'scipy.spatial._ckdtree',
            'scipy.spatial.transform._rotation', 'scipy.integrate._odepack',
            'scipy.ndimage._nd_image', 'scipy.linalg._fblas', 'scipy.linalg._flapack'):
        path = Path(importlib.import_module(name).__file__)
        info = path.stat()
        digest.update(repr((str(path), info.st_dev, info.st_ino, info.st_size,
                            info.st_mtime_ns, info.st_ctime_ns)).encode())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class FrameAdmission:
    """Internal capability owning recursively immutable canonical replay data."""
    scene: object
    shell: object
    snapshot: object
    history: object
    snapshot_hash: str
    history_hash: str
    fingerprint: str | None
    light_seal: str
    receipt_hashes: tuple
    action_hashes: tuple
    _authority: object

    @classmethod
    def canonical(cls, current, snapshot, history):
        from .surgical_frame_replay import SurgicalFrameCache
        current.scene.require_cache_source()
        SurgicalFrameCache._ancestry(snapshot, history)
        if current.scene.identity != snapshot.get('sceneId'):
            raise ValueError('Observed frame admission source differs')
        from .surgical_replay_cache import SurgicalReplayCache
        return cls(current.scene, SurgicalReplayCache._shell(current), freeze_json(snapshot), freeze_json(history),
            semantic_digest(snapshot), semantic_digest(history), observed_implementation_fingerprint(),
            light_implementation_seal(), tuple(receipt_hash(row) for row in history),
            tuple(semantic_digest(action) for action in snapshot['actions']), _AUTHORITY)

    def check(self, current=None, snapshot=None, history=None):
        current = self.shell if current is None else current
        current.scene.require_cache_source()
        if (self._authority is not _AUTHORITY or current.scene is not self.scene
                or snapshot is not None and snapshot is not self.snapshot
                or history is not None and history is not self.history):
            raise ValueError('Observed frames require the exact private immutable admission')
        if light_implementation_seal() != self.light_seal:
            raise ImplementationChanged('Observed frame implementation changed; refresh canonical admission')

    def boundary_id(self, position):
        return self.history[position - 1]['stateAfter'] if position else (
            self.history[0]['stateBefore'] if self.history else self.snapshot['stateId'])


class ObservedFrameStore:
    def __init__(self, directory=None, *, max_disk_bytes=512 * 1024**2):
        self.owner = VerifiedSurgicalCache(directory or DEFAULT_DIRECTORY, max_disk_bytes=max_disk_bytes)
        self.hits = self.misses = self.publications = 0

    @property
    def statistics(self):
        return {**self.owner.statistics, 'hits': self.hits, 'misses': self.misses,
                'publications': self.publications}

    def entry(self, admission):
        return self.owner.directory / self.owner._entry_name(admission.snapshot_hash, admission.fingerprint or 'unavailable')

    def begin(self, admission, position):
        if admission.fingerprint is None: return None
        self.owner._key()
        path = Path(tempfile.mkdtemp(prefix='.pending-', dir=self.owner.directory))
        path.chmod(0o700)
        # Reserve once per action, not one recursive disk scan per state field.
        # The bounded reservation includes descriptors and rejects unusually
        # large actions rather than allowing unbounded temporary disk growth.
        reserve = min(64 * 1024**2, max(1, self.owner.max_disk_bytes - 4096))
        with self.owner._locked():
            if not self.owner._room(reserve, path):
                path.rmdir(); return None
            with (path / '.reservation').open('xb') as stream: stream.truncate(reserve)
        return ObservedFrameWriter(self, admission, position, path, reserve)

    def load(self, admission, position):
        """Read only authenticated descriptors. Field bytes are checked on use."""
        try:
            if admission.fingerprint is None: raise ValueError('Unavailable implementation')
            key = self.owner._key(); entry = self.entry(admission)
            if entry.is_symlink(): raise ValueError('Observed entry cannot be redirected')
            raw = _read(entry / f'action-{position}.json.gz', self.owner.max_disk_bytes)
            signature = _read(entry / f'action-{position}.mac', 64)
            if not hmac.compare_digest(signature, hmac.digest(key, raw, 'sha256').hex().encode()):
                raise ValueError('Observed manifest authentication differs')
            manifest = _decode(raw)
            if (manifest.get('schema') != SCHEMA or manifest.get('sceneId') != admission.scene.identity
                    or manifest.get('snapshotHash') != admission.snapshot_hash
                    or manifest.get('historyHash') != admission.history_hash
                    or manifest.get('implementation') != admission.fingerprint
                    or manifest.get('position') != position
                    or manifest.get('receiptHash') != admission.receipt_hashes[position]
                    or manifest.get('actionHash') != admission.action_hashes[position]):
                raise ValueError('Observed manifest canonical binding differs')
            frames = manifest['frames']
            if not isinstance(frames, list) or len(frames) < 2:
                raise ValueError('Observed frames missing boundaries')
            times = [frame['observedTimeSeconds'] for frame in frames]
            if times != sorted(times) or frames[0]['stateId'] != admission.boundary_id(position) or frames[-1]['stateId'] != admission.boundary_id(position + 1):
                raise ValueError('Observed frame endpoint ancestry differs')
            self.hits += 1
            return {'entry': entry, 'frames': frames}
        except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError):
            self.misses += 1
            return None

    def state(self, item, frame_index):
        return _restore_state(item['entry'], item['frames'][frame_index]['fields'], self.owner.max_disk_bytes)


class ObservedFrameWriter:
    def __init__(self, store, admission, position, directory, reserve):
        self.store, self.admission, self.position, self.directory = store, admission, position, directory
        self.frames = []; self.closed = False
        self.reserve = reserve; self.written_bytes = 0

    def abort(self):
        if not self.closed: shutil.rmtree(self.directory, ignore_errors=True); self.closed = True

    def _write(self, name, data):
        destination = self.directory / name
        if destination.exists(): return
        if self.written_bytes + len(data) > self.reserve:
            raise ValueError('Observed action exceeds reserved disk budget')
        # Logical reservation bytes participate in the shared disk budget;
        # replace them with actual staged bytes, including across processes.
        with (self.directory / '.reservation').open('r+b') as stream:
            stream.truncate(self.reserve - self.written_bytes - len(data))
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream: stream.write(data)
        self.written_bytes += len(data)

    def capture(self, engine, kind, activation):
        if self.closed: return
        fields = {}; total = _state_overhead(engine.state)
        for name, value in engine.state.items():
            data, size = _encode_sized(value); total += size
            if total > MAX_JSON_BYTES: raise ValueError('Observed state exceeds transient decode budget')
            digest = hashlib.sha256(data).hexdigest(); filename = 'part-' + digest + '.json.gz'
            # Deduplicate stable large topology fields within and across actions.
            existing = self.store.entry(self.admission) / filename
            try:
                valid_existing = hashlib.sha256(_read(existing, self.store.owner.max_disk_bytes)).hexdigest() == digest
            except (OSError, ValueError): valid_existing = False
            if not valid_existing: self._write(filename, data)
            fields[name] = {'file': filename, 'sha256': digest, 'uncompressedBytes': size}
        self.frames.append({'observedTimeSeconds': float(engine.state['elapsedSeconds']), 'kind': kind,
            'effectiveActivation': dict(activation), 'stateId': engine.state_id,
            'fields': {'schema': STATE_SCHEMA, 'fields': fields}})

    def publish(self, engine, outcome):
        admission = self.admission; position = self.position
        if (engine.state_id != admission.boundary_id(position + 1)
                or receipt_hash(outcome) != admission.receipt_hashes[position]):
            self.abort(); raise ValueError('Observed action outcome differs from canonical admission')
        if observed_implementation_fingerprint() != admission.fingerprint:
            self.abort(); return False
        admission.check(engine)
        manifest = {'schema': SCHEMA, 'sceneId': engine.scene.identity,
            'snapshotHash': admission.snapshot_hash, 'historyHash': admission.history_hash,
            'implementation': admission.fingerprint, 'position': position,
            'receiptHash': admission.receipt_hashes[position], 'actionHash': admission.action_hashes[position],
            'frames': self.frames}
        data = _encode(manifest); key = self.store.owner._key()
        self._write(f'action-{position}.json.gz', data)
        self._write(f'action-{position}.mac', hmac.digest(key, data, 'sha256').hex().encode())
        entry = self.store.entry(admission)
        try:
            with self.store.owner._locked():
                entry.mkdir(mode=0o700, exist_ok=True)
                if entry.is_symlink(): raise ValueError('Observed entry cannot be redirected')
                (self.directory / '.reservation').unlink()
                for path in self.directory.iterdir(): os.replace(path, entry / path.name)
                self.directory.rmdir()
            self.closed = True; self.store.publications += 1
            return True
        except (OSError, ValueError):
            self.abort(); return False
