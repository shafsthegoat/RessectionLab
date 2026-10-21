"""Disposable local acceleration for independently replay-verified archives.

Native checkpoints are never trusted by this module on their own. Only the
engine's successful canonical restore path can publish a signed cache entry.
The private local key authenticates cached history and every state blob; source,
complete native snapshot and loaded implementation identities bind each entry.
No pickle or executable serialization is used. Missing/invalid entries fall back
to ordinary public-action replay. No effect state is retained in memory here.
"""
from __future__ import annotations

from contextlib import contextmanager
import ast
import copy
import gzip
import hashlib
import hmac
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import stat
import sys
import tempfile
import types

from .core import semantic_digest

SCHEMA = 'resectionlab.verified-surgical-cache.v2'
STATE_SCHEMA = 'resectionlab.verified-surgical-state-fields.v1'
MAX_DISK_BYTES = 512 * 1024**2
MAX_DISK_ALLOWANCE_BYTES = 8 * 1024**3
DISK_BUDGET_ENV = 'RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES'
MAX_JSON_BYTES = 128 * 1024**2
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / 'testcases/outputs/.verified-surgical-cache'
_MODULES = ('core', 'geometry', 'surgical_scene', 'surgical_engine', 'surgical_tools',
            'surgical_motion', 'surgical_effects', 'surgical_physics', 'surgical_physics_v2',
            'surgical_drilling_v2', 'surgical_vessels_v2', 'surgical_deformation_v2',
            'surgical_occupancy', 'surgical_rl_protection', 'surgical_geometry_cache',
            'surgical_verified_cache')


def _code_signature(code):
    """Stable public bytecode/constant identity, independent of marshal interns.

    Python 3.14 can change marshal reference flags after a generator executes;
    those serialization flags are not an implementation change.
    """
    def constant(value):
        if isinstance(value, types.CodeType):
            return ['code', _code_signature(value)]
        if isinstance(value, (tuple, frozenset)):
            items = [constant(item) for item in value]
            if isinstance(value, frozenset):
                items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return [type(value).__name__, items]
        return [type(value).__name__, repr(value)]
    return [code.co_code.hex(), [constant(value) for value in code.co_consts],
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_flags, code.co_exceptiontable.hex(), code.co_firstlineno]


def _fingerprint():
    """Bind disk sources AND loaded Python code, constants and dependency builds.

    File hashes alone would misidentify an old running process after a source
    edit. Loaded code hashes distinguish that process from a fresh import.
    Foreign runtime monkeypatches disable admission rather than hiding a change.
    """
    import numpy
    import scipy
    digest = hashlib.sha256()
    components = {}
    def add(name, data):
        digest.update(data)
        components[name] = hashlib.sha256(data).hexdigest()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob('*.py')):
        add('source:' + path.name, path.name.encode() + path.read_bytes())
    for name in _MODULES:
        module = importlib.import_module('resectionlab.' + name)
        declared = {node.name for node in ast.parse(Path(module.__file__).read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        functions = []
        for key, value in sorted(vars(module).items()):
            if inspect.isfunction(value):
                if not value.__module__.startswith('resectionlab.') and key not in declared:
                    continue  # An ordinary imported standard-library helper.
                functions.append((key, value))
            elif inspect.isclass(value) and value.__module__ == module.__name__:
                for member, method in sorted(vars(value).items()):
                    if isinstance(method, (staticmethod, classmethod)):
                        method = method.__func__
                    elif isinstance(method, property):
                        method = method.fget
                    if inspect.isfunction(method):
                        functions.append((key + '.' + member, method))
            if key.isupper() and isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                try:
                    add('constant:' + module.__name__ + ':' + key, semantic_digest(value).encode())
                except (ValueError, TypeError):
                    pass  # Non-JSON declarations are covered by source/code.
        for key, function in functions:
            if not function.__module__.startswith('resectionlab.') and not (
                    function.__module__ == 'dataclasses' and key.endswith(('.__getstate__', '.__setstate__'))):
                return None
            add('code:' + module.__name__ + ':' + key, (module.__name__ + ':' + key).encode()
                + json.dumps(_code_signature(function.__code__), separators=(',', ':')).encode())
    add('runtime', repr((sys.version, sys.implementation.cache_tag, platform.platform(),
                        numpy.__version__, scipy.__version__)).encode())
    # Version strings do not distinguish numerical-library build variants.
    for name in ('numpy._core._multiarray_umath', 'scipy.spatial._ckdtree',
                 'scipy.spatial.transform._rotation', 'scipy.integrate._odepack',
                 'scipy.ndimage._nd_image', 'scipy.linalg._fblas', 'scipy.linalg._flapack'):
        dependency = importlib.import_module(name)
        add('dependency:' + name, name.encode() + Path(dependency.__file__).read_bytes())
    globals()['_last_components'] = components
    return digest.hexdigest()


def implementation_fingerprint():
    """Return None during an unavailable/changing implementation, never guess."""
    try:
        return _fingerprint()
    except (OSError, SyntaxError, ValueError, TypeError, ImportError, AttributeError):
        return None


def _check(check):
    if check is not None:
        check()


def _encode(value):
    return _encode_sized(value)[0]


def _encode_sized(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('Verified cache JSON exceeds transient decode budget')
    return gzip.compress(raw, compresslevel=1, mtime=0), len(raw)


def _decode(data):
    return _decode_sized(data)[0]


def _decode_sized(data):
    # Bound decompression before parsing even an authenticated obsolete entry.
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('Verified cache JSON exceeds transient decode budget')
    return json.loads(raw), len(raw)


def _state_overhead(names):
    return 2 + max(0, len(names) - 1) + sum(
        len(json.dumps(name, ensure_ascii=True).encode()) + 1 for name in names)


def _restore_state(entry, descriptor, limit):
    """Reassemble only SHA-bound fields from a signed canonical descriptor."""
    if descriptor.get('schema') != STATE_SCHEMA or set(descriptor) != {'schema', 'fields'}:
        raise ValueError('Verified cache state encoding differs')
    fields = descriptor['fields']
    if not isinstance(fields, dict):
        raise ValueError('Verified cache state fields differ')
    result = {}
    total = _state_overhead(fields)
    for name, blob in fields.items():
        digest = blob['sha256']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or blob['file'] != 'part-' + digest + '.json.gz'
                or type(blob['uncompressedBytes']) is not int
                or not 0 <= blob['uncompressedBytes'] <= MAX_JSON_BYTES):
            raise ValueError('Verified cache state field identity differs')
        total += blob['uncompressedBytes']
        if total > MAX_JSON_BYTES:
            raise ValueError('Verified cache state exceeds transient decode budget')
        compressed = _read(entry / blob['file'], limit)
        if hashlib.sha256(compressed).hexdigest() != digest:
            raise ValueError('Verified cache state field authentication failed')
        value, size = _decode_sized(compressed)
        if size != blob['uncompressedBytes']:
            raise ValueError('Verified cache state field length differs')
        result[name] = value
    return result


def _read(path, limit):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('Invalid verified cache file')
        return stream.read(limit + 1)


class VerifiedSurgicalCache:
    _disk_budget_limit = MAX_DISK_ALLOWANCE_BYTES

    def __init__(self, directory=None, *, max_disk_bytes=None):
        if max_disk_bytes is None:
            configured = os.environ.get(DISK_BUDGET_ENV)
            if configured is None:
                max_disk_bytes = MAX_DISK_BYTES
            else:
                try:
                    max_disk_bytes = int(configured)
                except ValueError as error:
                    raise ValueError(f'{DISK_BUDGET_ENV} must be integer bytes in 1..{self._disk_budget_limit}') from error
                if not 1 <= max_disk_bytes <= self._disk_budget_limit:
                    raise ValueError(f'{DISK_BUDGET_ENV} must be integer bytes in 1..{self._disk_budget_limit}')
        if type(max_disk_bytes) is not int or not 1 <= max_disk_bytes <= self._disk_budget_limit:
            raise ValueError(f'Verified cache disk budget must be 1..{self._disk_budget_limit // 1024**2} MiB')
        self.directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
        self.max_disk_bytes = max_disk_bytes
        self.hits = self.misses = self.publications = 0
        self.last_skip = None

    def _key(self):
        if self.max_disk_bytes < 32:
            raise ValueError('Verified cache budget cannot hold its private key')
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError('Verified cache directory cannot be a symlink')
        self.directory.chmod(0o700)
        path = self.directory / 'key'
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(secrets.token_bytes(32))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError('Verified cache key must be a regular 0600 file')
        if hasattr(os, 'getuid') and info.st_uid != os.getuid():
            raise ValueError('Verified cache key belongs to another user')
        key = _read(path, 32)
        if len(key) != 32:
            raise ValueError('Invalid verified cache key')
        return key

    @contextmanager
    def _locked(self):
        # This optional local optimization is unavailable on systems without a
        # process file lock; callers then retain full replay semantics.
        import fcntl
        fd = os.open(self.directory / 'lock', os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _bytes(self):
        return sum(path.lstat().st_size for path in self.directory.rglob('*') if path.is_file() and not path.is_symlink())

    @property
    def statistics(self):
        return {'diskBytes': self._bytes() if self.directory.exists() else 0,
                'maxDiskBytes': self.max_disk_bytes, 'maxFileBytes': self.max_file_bytes, 'ownedMemoryCacheBytes': 0,
                'maxMemoryCacheBytes': 128 * 1024**2, 'maxTransientJsonBytes': MAX_JSON_BYTES,
                'hits': self.hits, 'misses': self.misses, 'publications': self.publications}

    @property
    def max_file_bytes(self):
        # Aggregate disk policy must not widen the existing compressed read
        # allowance or the separate MAX_JSON_BYTES decompression limit.
        return min(self.max_disk_bytes, MAX_DISK_BYTES)

    def _room(self, incoming, keep):
        entries = sorted((path for path in self.directory.iterdir()
                          if path.is_dir() and not path.is_symlink() and path.name.startswith('entry-')
                          and path != keep), key=lambda path: path.stat().st_mtime)
        while self._bytes() + incoming > self.max_disk_bytes and entries:
            shutil.rmtree(entries.pop(0))
        return self._bytes() + incoming <= self.max_disk_bytes

    @staticmethod
    def _entry_name(snapshot_hash, implementation):
        return 'entry-' + hashlib.sha256((snapshot_hash + ':' + implementation).encode()).hexdigest()

    def restore(self, cls, scene, record, *, position=None, check=None, include_history=False):
        """Read a sealed canonical replay, or return None for ordinary replay."""
        _check(check)
        scene.require_cache_source()
        from .surgical_engine import SurgicalEngine
        if cls is not SurgicalEngine:
            return None  # A subclass's transitions are a different implementation.
        try:
            fingerprint = implementation_fingerprint()
            if fingerprint is None:
                self.misses += 1
                return None
            snapshot_hash = semantic_digest(record)
            key = self._key()
            entry = self.directory / self._entry_name(snapshot_hash, fingerprint)
            compressed = _read(entry / 'manifest.json.gz', self.max_file_bytes)
            signature = _read(entry / 'manifest.mac', 64)
            if not hmac.compare_digest(signature, hmac.digest(key, compressed, 'sha256').hex().encode()):
                raise ValueError('Verified cache authentication failed')
            manifest = _decode(compressed)
            if (manifest['schema'] != SCHEMA or manifest['snapshotHash'] != snapshot_hash
                    or manifest['sceneId'] != scene.identity or manifest['implementation'] != fingerprint
                    or len(manifest['states']) != len(record['actions']) + 1
                    or len(manifest['history']) != len(record['actions'])):
                raise ValueError('Verified cache ancestry differs')
            count = len(record['actions']) if position is None else position
            if type(count) is not int or not 0 <= count <= len(record['actions']):
                raise ValueError('Invalid verified history position')
            blob = manifest['states'][count]
            if blob['file'] != f'{count}.json.gz':
                raise ValueError('Verified cache state position differs')
            compressed_state = _read(entry / blob['file'], self.max_file_bytes)
            if hashlib.sha256(compressed_state).hexdigest() != blob['sha256']:
                raise ValueError('Verified cache state authentication failed')
            state = _restore_state(entry, _decode(compressed_state), self.max_file_bytes)
            options = {}
            if 'protectionContract' in record:
                from .surgical_rl_protection import restore_protection_contract
                protection, config = restore_protection_contract(scene, record['protectionContract'])
                options = {'protection': protection, 'guard_config': config}
            result = cls(scene, **options)
            result.state = state
            result.actions = json.loads(json.dumps(record['actions'][:count], allow_nan=False))
            result.history = (copy.deepcopy(manifest['history'][:count]) if include_history
                              else manifest['history'][:count])
            expected = manifest['states'][count]['stateId']
            if result.state_id != expected:
                raise ValueError('Verified cache state identity differs')
            if count == len(record['actions']) and expected != record['stateId']:
                raise ValueError('Verified cache endpoint differs')
            if semantic_digest(record) != snapshot_hash or implementation_fingerprint() != fingerprint:
                raise ValueError('Verified cache inputs changed while loading')
            scene.require_cache_source()
        except (OSError, ValueError, KeyError, TypeError, ImportError, EOFError):
            self.misses += 1
            return None
        _check(check)
        try:
            os.utime(entry, None)
        except OSError:
            pass
        self.hits += 1
        return (result, manifest['history']) if include_history else result

    def _begin_replay(self, engine, record, check=None):
        """Internal staging only. No signature exists before restore validates."""
        from .surgical_engine import SurgicalEngine
        if type(engine) is not SurgicalEngine:
            return None
        pending = None
        try:
            engine.scene.require_cache_source()
            fingerprint = implementation_fingerprint()
            if fingerprint is None:
                return None
            key = self._key()
            pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=self.directory))
            pending.chmod(0o700)
            writer = _ReplayWriter(self, pending, key, fingerprint, semantic_digest(record))
            return writer
        except (OSError, ValueError, ImportError):
            if pending is not None:
                shutil.rmtree(pending, ignore_errors=True)
            return None


class _ReplayWriter:
    def __init__(self, owner, directory, key, fingerprint, snapshot_hash):
        self.owner, self.directory, self.key = owner, directory, key
        self.fingerprint, self.snapshot_hash = fingerprint, snapshot_hash
        self.components = dict(globals().get('_last_components', {}))
        self.states = []
        self.closed = False

    def abort(self):
        if not self.closed:
            shutil.rmtree(self.directory, ignore_errors=True)
            self.closed = True

    def _write(self, name, data):
        with self.owner._locked():
            if not self.owner._room(len(data), self.directory):
                raise ValueError('Verified cache disk budget exhausted')
            fd = os.open(self.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)

    def capture(self, engine, check=None):
        _check(check)
        if self.closed:
            return
        try:
            fields = {}
            total = _state_overhead(engine.state)
            for field, value in engine.state.items():
                compressed, size = _encode_sized(value)
                total += size
                if total > MAX_JSON_BYTES:
                    raise ValueError('Verified cache state exceeds transient decode budget')
                digest = hashlib.sha256(compressed).hexdigest()
                part_name = 'part-' + digest + '.json.gz'
                # Content addressing stays inside this private, unpublished
                # entry. Repeated large topology fields consume disk once.
                if not (self.directory / part_name).exists():
                    self._write(part_name, compressed)
                fields[field] = {'file': part_name, 'sha256': digest, 'uncompressedBytes': size}
            data = _encode({'schema': STATE_SCHEMA, 'fields': fields})
            name = str(len(self.states)) + '.json.gz'
            self._write(name, data)
            self.states.append({'file': name, 'sha256': hashlib.sha256(data).hexdigest(),
                                'stateId': engine.state_id})
        except (OSError, ValueError, ImportError):
            self.abort()
        _check(check)

    def publish(self, engine, record, check=None):
        _check(check)
        if self.closed:
            return
        try:
            engine.scene.require_cache_source()
            current_fingerprint = implementation_fingerprint()
            if (current_fingerprint != self.fingerprint or semantic_digest(record) != self.snapshot_hash
                    or len(self.states) != len(engine.actions) + 1):
                self.owner.last_components = [k for k,v in self.components.items() if globals().get('_last_components', {}).get(k) != v]
                self.owner.last_skip = ('implementation changed' if current_fingerprint != self.fingerprint else
                                        'snapshot changed' if semantic_digest(record) != self.snapshot_hash else 'state count differs')
                self.abort()
                return
            data = _encode({'schema': SCHEMA, 'sceneId': engine.scene.identity,
                            'implementation': self.fingerprint, 'snapshotHash': self.snapshot_hash,
                            'states': self.states, 'history': engine.history})
            self._write('manifest.json.gz', data)
            self._write('manifest.mac', hmac.digest(self.key, data, 'sha256').hex().encode())
        except (OSError, ValueError, ImportError):
            self.abort()
            return
        _check(check)
        destination = self.owner.directory / self.owner._entry_name(self.snapshot_hash, self.fingerprint)
        try:
            with self.owner._locked():
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(self.directory, destination)
            self.closed = True
            self.owner.publications += 1
        except OSError:
            self.abort()

