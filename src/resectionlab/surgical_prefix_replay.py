"""Isolated read-only reconstruction of demanded imported action prefixes.

This owner is not a surgical session, canonical archive admission, or disk
cache. Imported checkpoint state never seeds reconstruction. Only saved public
actions execute, once; evicted boundaries/frames are unavailable rather than
silently executing their actions again. Disposable compressed checkpoints and
observations have separate retained-byte budgets. Source anatomy, the bounded
immutable input, and transient engine/view allocations are reported separately.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import secrets
import sys
from types import MappingProxyType
import zlib

from .core import freeze_json, semantic_digest, thaw_json
from .surgical_engine import MAX_STEPS, SCHEMA, SurgicalEngine

_FROZEN_INPUT_AUTHORITY = object()


class PrefixBudgetExceeded(RuntimeError):
    """The next prefix action cannot fit this owner's explicit limits."""


class PrefixInvalidated(ValueError):
    """The exact source or loaded implementation no longer matches admission."""


class _Interrupted(RuntimeError):
    def __init__(self, cause):
        self.cause = cause
        super().__init__(str(cause))


def _fingerprint():
    # Lazy import: fingerprinting imports engine dependencies, including replay
    # helpers. Do not create a module-import cycle or instantiate a cache.
    from .surgical_verified_cache import implementation_fingerprint
    return implementation_fingerprint()


def _light_seal():
    from .surgical_observed_cache import light_implementation_seal
    return light_implementation_seal()


def _prefix_seal():
    """Bind this module's loaded code/defaults, absent from legacy _MODULES."""
    from .surgical_verified_cache import _code_signature
    functions = []
    for name, value in sorted(globals().items()):
        if inspect.isfunction(value) and value.__module__ == __name__:
            functions.append((name, value))
        elif inspect.isclass(value) and value.__module__ == __name__:
            for key, method in sorted(vars(value).items()):
                if isinstance(method, (classmethod, staticmethod)):
                    method = method.__func__
                elif isinstance(method, property):
                    method = method.fget
                if inspect.isfunction(method):
                    functions.append((name + '.' + key, method))
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for name, function in functions:
        digest.update(json.dumps([name, _code_signature(function.__code__),
            repr(function.__defaults__), repr(function.__kwdefaults__)], separators=(',', ':')).encode())
    return digest.hexdigest()


def _owned_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        # MappingProxyType owns a private frozen dictionary as well as its shell.
        size += sys.getsizeof(dict(value)) if not isinstance(value, dict) else 0
        size += sum(_owned_bytes(k, seen) + _owned_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_owned_bytes(item, seen) for item in value)
    return size


def _require_frozen(value):
    if isinstance(value, MappingProxyType):
        if any(type(key) is not str for key in value):
            raise ValueError('Borrowed frozen snapshot requires JSON string keys')
        for item in value.values():
            _require_frozen(item)
    elif type(value) is tuple:
        for item in value:
            _require_frozen(item)
    elif value is None or type(value) in (str, bool, int):
        return
    elif type(value) is float and math.isfinite(value):
        return
    else:
        raise ValueError('Borrowed input must be exact private recursively frozen JSON')


def _json_string_chunks(value):
    """Default ASCII escaping, with at most 1,024 source codepoints at once.

    One non-BMP codepoint may need twelve ASCII bytes (two surrogate escapes).
    Built-in string operations preserve the encoder's treatment of subclasses
    without invoking their overridden slicing/length/string methods.
    """
    length = str.__len__(value)
    if length <= 1024:
        yield json.encoder.encode_basestring_ascii(value)
        return
    yield '"'
    for start in range(0, length, 1024):
        part = str.__getitem__(value, slice(start, start + 1024))
        yield json.encoder.encode_basestring_ascii(part)[1:-1]
    yield '"'


def _c_json_bound(value, remaining=16384, active=None):
    """Conservative encoded-byte admission for small exact native JSON trees.

    Refuse custom objects, nonfinite floats, large integers/strings and more
    than 512 entries before C serialization. A non-BMP codepoint needs up to
    twelve ASCII bytes. The returned bound includes every delimiter.
    """
    kind = type(value)
    if kind is str:
        length = len(value)
        size = 12 * length + 2 if length <= 1024 else remaining + 1
    elif kind is int:
        size = 21 if value.bit_length() <= 64 else remaining + 1
    elif kind is float:
        size = 25 if math.isfinite(value) else remaining + 1
    elif value is None:
        size = 4
    elif kind is bool:
        size = 5
    elif kind in (dict, list, tuple):
        if len(value) > 512:
            return None
        size = 2 + max(0, len(value) - 1)
        if size > remaining:
            return None
        active = set() if active is None else active
        marker = id(value)
        if marker in active:
            return None
        active.add(marker)
        try:
            if kind is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        return None
                    key_size = _c_json_bound(key, remaining - size - 1, active)
                    if key_size is None:
                        return None
                    size += key_size + 1
                    item_size = _c_json_bound(item, remaining - size, active)
                    if item_size is None:
                        return None
                    size += item_size
            else:
                for item in value:
                    item_size = _c_json_bound(item, remaining - size, active)
                    if item_size is None:
                        return None
                    size += item_size
        finally:
            active.remove(marker)
    else:
        return None
    return size if size <= remaining else None


def _json_chunks(value, raw_limit):
    """Compact JSON with bounded C batches for exact native JSON subtrees.

    Container traversal follows JSONEncoder's insertion order, basic key
    coercions, circular checks and subclass iteration. Only exact lists/tuples
    are batched speculatively. A batch has at most 512 entries and a conservative
    full encoded-byte bound of 16 KiB, including escaping and delimiters.
    Other scalars retain the standard encoder's conversion/error behavior.
    """
    encoder = json.JSONEncoder(separators=(',', ':'), allow_nan=False)
    active = set()

    def scalar(item):
        if isinstance(item, int) and int.bit_length(item) * 3 // 10 > raw_limit:
            # A conservative decimal-length lower bound avoids converting a
            # giant integer before discovering it cannot fit the raw budget.
            raise PrefixBudgetExceeded('Decoded JSON budget exceeded')
        yield from encoder.iterencode(item)

    def walk(item):
        if _c_json_bound(item) is not None:
            yield json.dumps(item, separators=(',', ':'), allow_nan=False)
        elif isinstance(item, str):
            yield from _json_string_chunks(item)
        elif isinstance(item, (list, tuple, dict)):
            dictionary = isinstance(item, dict)
            if not item:
                yield '{}' if dictionary else '[]'
                return
            marker = id(item)
            if marker in active:
                raise ValueError('Circular reference detected')
            active.add(marker)
            try:
                yield '{' if dictionary else '['
                first = True
                if dictionary:
                    batch = {}
                    batch_size = 2
                    for key, child in item.items():
                        key_size = _c_json_bound(key) if type(item) is dict and type(key) is str else None
                        child_size = _c_json_bound(child) if key_size is not None else None
                        entry_size = (key_size + 1 + child_size) if child_size is not None else None
                        if batch and (entry_size is None or len(batch) == 512
                                      or batch_size + 1 + entry_size > 16384):
                            if not first:
                                yield ','
                            first = False
                            yield json.dumps(batch, separators=(',', ':'), allow_nan=False)[1:-1]
                            batch = {}
                            batch_size = 2
                        if entry_size is not None and entry_size + 2 <= 16384:
                            batch_size += entry_size + int(bool(batch))
                            batch[key] = child
                            continue
                        if not isinstance(key, str):
                            if key is None or isinstance(key, (int, float)):
                                key = ''.join(scalar(key))
                            else:
                                raise TypeError('keys must be str, int, float, bool or None, '
                                                f'not {key.__class__.__name__}')
                        if not first:
                            yield ','
                        first = False
                        yield from _json_string_chunks(key)
                        yield ':'
                        yield from walk(child)
                    if batch:
                        if not first:
                            yield ','
                        yield json.dumps(batch, separators=(',', ':'), allow_nan=False)[1:-1]
                elif type(item) in (list, tuple):
                    batch = []
                    batch_size = 2
                    for child in item:
                        child_size = _c_json_bound(child)
                        if batch and (child_size is None or len(batch) == 512
                                      or batch_size + 1 + child_size > 16384):
                            if not first:
                                yield ','
                            first = False
                            yield json.dumps(batch, separators=(',', ':'), allow_nan=False)[1:-1]
                            batch = []
                            batch_size = 2
                        if child_size is not None and child_size + 2 <= 16384:
                            batch_size += child_size + int(bool(batch))
                            batch.append(child)
                            continue
                        if not first:
                            yield ','
                        first = False
                        yield from walk(child)
                    if batch:
                        if not first:
                            yield ','
                        yield json.dumps(batch, separators=(',', ':'), allow_nan=False)[1:-1]
                else:
                    # Respect list/tuple subclass iterators instead of slicing
                    # or speculatively encoding custom container contents.
                    for child in item:
                        if not first:
                            yield ','
                        first = False
                        yield from walk(child)
                yield '}' if dictionary else ']'
            finally:
                active.remove(marker)
        else:
            yield from scalar(item)
    yield from walk(value)


def _encode(value, *, compressed_limit, raw_limit, on_growth=None):
    """Bound compressed growth while streaming; never buffer a whole action."""
    compressor = zlib.compressobj(1)
    output = bytearray()
    pending = bytearray()
    raw_bytes = 0
    def append(chunk):
        if len(output) + len(chunk) > compressed_limit:
            raise PrefixBudgetExceeded('Compressed state budget exceeded')
        if on_growth is not None:
            on_growth(len(output) + len(chunk))
        output.extend(chunk)
    for text in _json_chunks(value, raw_limit):
        raw = text.encode()
        raw_bytes += len(raw)
        if raw_bytes > raw_limit:
            raise PrefixBudgetExceeded('Decoded JSON budget exceeded')
        for offset in range(0, len(raw), 16384):
            part = raw[offset:offset + 16384]
            if len(pending) + len(part) > 16384:
                append(compressor.compress(pending))
                pending.clear()
            pending.extend(part)
    if pending:
        append(compressor.compress(pending))
    append(compressor.flush())
    return bytes(output)


def _decode(blob, limit):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(blob, limit + 1)
    if len(raw) > limit or not decoder.eof or decoder.unused_data:
        raise ValueError('Prefix compressed state exceeds decode budget or is incomplete')
    return json.loads(raw)


class _FrameStage:
    def __init__(self, owner):
        self.owner = owner
        self.frames = []
        self.next_time = None
        self.activation = None

    def capture(self, engine, kind, activation):
        # Encoding itself stops at the action's remaining cap. Old committed
        # frames may be evicted, but never held alongside an unbounded action.
        row = {'observedTimeSeconds': float(engine.state['elapsedSeconds']),
               'kind': kind, 'effectiveActivation': activation,
               'stateId': engine.state_id, 'state': b'',
               'fieldTokens': MappingProxyType({'flapComponents': '0' * 64})
                   if 'flapComponents' in engine.state else MappingProxyType({})}
        # Include metadata before encoding and reserve owning-map overhead.
        # Growth callbacks evict old actions before appending compressed bytes.
        base = _owned_bytes(self.frames + [row]) + 512
        blob = _encode(engine.state, compressed_limit=max(0, self.owner._max_frame_bytes - base
                       - self.owner._stream_metadata_bytes),
                       raw_limit=self.owner._max_json_bytes,
                       on_growth=lambda size: self.owner._frame_room(base + size))
        row['state'] = blob
        if 'flapComponents' in engine.state:
            # The complete state's streaming encode has already enforced the
            # decoded JSON ceiling, which also bounds this field's temporary
            # canonical encoding. Tokens describe actual observed full values.
            value = json.dumps(engine.state['flapComponents'], sort_keys=True,
                               separators=(',', ':'), allow_nan=False).encode()
            row['fieldTokens'] = MappingProxyType({'flapComponents': hashlib.sha256(value).hexdigest()})
        prospective = self.frames + [row]
        self.owner._frame_room(_owned_bytes(prospective))
        self.frames.append(row)

    def begin(self, engine):
        self.next_time = float(engine.state['elapsedSeconds']) + .2
        self.capture(engine, 'action_start', {key: bool(slot['active']) for key, slot in engine.state['slots'].items()})

    def observe(self, engine, key, active):
        self.activation = {hand: bool(active if hand == key else slot['active'])
                           for hand, slot in engine.state['slots'].items()}
        if engine.state['elapsedSeconds'] + 1e-10 >= self.next_time:
            self.capture(engine, 'microstep', self.activation)
            while self.next_time <= engine.state['elapsedSeconds'] + 1e-10:
                self.next_time += .2

    def end(self, engine):
        self.capture(engine, 'action_boundary', self.activation or
                     {key: bool(slot['active']) for key, slot in engine.state['slots'].items()})


class _PrefixEngine(SurgicalEngine):
    def _substep(self, key, op, p, q, active, dt):
        events = super()._substep(key, op, p, q, active, dt)
        self._stage.observe(self, key, active)
        return events


class SurgicalPrefixReplay:
    """Bounded read-only JSON views of one immutable imported action log.

    Defaults: four new actions per request; 16 MiB compressed checkpoints,
    16 MiB compressed frames, 4 MiB computed receipts, 64 MiB immutable input
    and per-state decoded JSON. Overrides have hard ceilings. No frame lookup
    or old-boundary lookup reexecutes an already computed action; eviction is an
    explicit unavailable result. A new owner is required to reconstruct it.
    """
    def __init__(self, scene, snapshot, *, max_new_actions=4,
                 max_checkpoint_bytes=16 * 1024**2, max_frame_bytes=16 * 1024**2,
                 max_history_bytes=4 * 1024**2, max_input_bytes=64 * 1024**2,
                 max_json_bytes=64 * 1024**2, _frozen_input_authority=None,
                 _display_row_projection=None):
        if (_frozen_input_authority is not None
                and _frozen_input_authority is not _FROZEN_INPUT_AUTHORITY):
            raise ValueError('Private frozen-input authority differs')
        if _display_row_projection is not None and not callable(_display_row_projection):
            raise ValueError('Private prefix display row projection must be callable')
        limits = [(max_new_actions, 32), (max_checkpoint_bytes, 64 * 1024**2),
                  (max_frame_bytes, 64 * 1024**2), (max_history_bytes, 16 * 1024**2),
                  (max_input_bytes, 128 * 1024**2), (max_json_bytes, 128 * 1024**2)]
        if any(type(value) is not int or not 1 <= value <= ceiling for value, ceiling in limits):
            raise ValueError('Prefix replay limits must be positive integers within hard ceilings')
        self._stream_nonce = secrets.token_hex(16)
        if (max_frame_bytes < _owned_bytes(OrderedDict()) + _owned_bytes(self._stream_nonce)
                or max_history_bytes < _owned_bytes([])):
            raise PrefixBudgetExceeded('Empty frame/history owner exceeds retention budget')
        scene.require_cache_source()
        if _owned_bytes(snapshot) > max_input_bytes:
            raise PrefixBudgetExceeded('Immutable input budget exceeded')
        self._borrowed_input = _frozen_input_authority is _FROZEN_INPUT_AUTHORITY
        if self._borrowed_input:
            _require_frozen(snapshot)
        record = snapshot if self._borrowed_input else freeze_json(snapshot)
        self._input_bytes = _owned_bytes(record)
        if self._input_bytes > max_input_bytes:
            raise PrefixBudgetExceeded('Immutable input budget exceeded')
        if (not isinstance(record, Mapping) or record.get('schema') != SCHEMA or record.get('sceneId') != scene.identity
                or not isinstance(record.get('actions'), tuple)
                or len(record['actions']) > MAX_STEPS
                or any(not isinstance(action, Mapping) for action in record['actions'])
                or not isinstance(record.get('stateId'), str)):
            raise ValueError('Imported source, schema or action log differs')
        self._scene = self.__source_object = scene
        self._snapshot = record
        self._origin_hash = semantic_digest(record)
        self._action_hash = semantic_digest(record['actions'])
        self._protection_hash = semantic_digest(record.get('protectionContract'))
        self._implementation = _fingerprint()
        if self._implementation is None:
            raise ValueError('Prefix replay implementation fingerprint unavailable')
        self._implementation_light = _light_seal()
        self._prefix_implementation = _prefix_seal()
        self._max_new_actions = max_new_actions
        self._max_checkpoint_bytes = max_checkpoint_bytes
        self._max_frame_bytes = max_frame_bytes
        self._max_history_bytes = max_history_bytes
        self._max_json_bytes = max_json_bytes
        # Only this private execution shell sees metric-only ancestry. Complete
        # canonical receipts remain losslessly available through _receipt().
        self._history = []
        self._receipts = []
        self._display_receipts = []
        self._metric_rows = {}
        self._display_row_projection = _display_row_projection
        self._history_store = (self._history, self._receipts, self._display_receipts, self._metric_rows)
        self._retained_history_bytes = self._history_store_bytes(self._history_store)
        if self._retained_history_bytes > max_history_bytes:
            raise PrefixBudgetExceeded('Empty history retention budget exceeded')
        self._checkpoints = OrderedDict()
        self._frames = OrderedDict()
        self._retained_frame_bytes = _owned_bytes(self._frames)
        self._stream_metadata_bytes = _owned_bytes(self._stream_nonce)
        self._failure = None
        self._blocked = False
        self._endpoint_verified = False
        self._endpoint_error = None
        self._invalidated = None
        engine = self._fresh()
        self._initial_id = engine.state_id
        self._put_checkpoint(0, engine)
        self._validate_endpoint(engine)
        # Share only the private empty shell's pure geometry/protection runtime.
        # State/ancestry are reconstructed into a new shell for every read.
        self._template = copy.copy(engine)
        self._template.state = {}
        self._template.actions = []
        self._template.history = []
        self._template._occupancy_cache = None
        self._check()

    @classmethod
    def _borrow_frozen_snapshot(cls, scene, snapshot, **limits):
        """Bridge-private capability: borrow its exact privately frozen branch.

        The caller must own a core.freeze_json result with no exposed mutable
        backing dictionaries; an arbitrary user-supplied MappingProxyType is
        not such an admission. This avoids another full imported archive copy.
        It grants only this isolated read-only API, never session/edit/cache
        authority. The normal constructor still defensively freezes all input.
        Input limits remain explicit and unchanged, including the 128 MiB cap.
        """
        return cls(scene, snapshot, _frozen_input_authority=_FROZEN_INPUT_AUTHORITY, **limits)

    def _fresh(self):
        options = {}
        if 'protectionContract' in self._snapshot:
            from .surgical_rl_protection import restore_protection_contract
            protection, config = restore_protection_contract(self._scene, thaw_json(self._snapshot['protectionContract']))
            options = {'protection': protection, 'guard_config': config}
        return _PrefixEngine(self._scene, **options)

    def _check(self, check=None, *, full=False):
        if check is not None:
            check()
        if self._invalidated is not None:
            raise PrefixInvalidated(self._invalidated)
        if self._scene is not self.__source_object:
            self._invalidated = 'Prefix replay source object changed'
            raise PrefixInvalidated(self._invalidated)
        try:
            self._scene.require_cache_source()
        except ValueError as error:
            self._invalidated = str(error)
            raise PrefixInvalidated(self._invalidated) from None
        try:
            light = _light_seal()
        except ValueError as error:
            self._invalidated = 'Prefix replay implementation changed; create a new owner'
            raise PrefixInvalidated(self._invalidated) from error
        if (light != self._implementation_light or _prefix_seal() != self._prefix_implementation):
            self._invalidated = 'Prefix replay implementation changed; create a new owner'
            raise PrefixInvalidated(self._invalidated)
        if full:
            current = _fingerprint()
            if current is None or current != self._implementation:
                self._invalidated = 'Prefix replay implementation changed; create a new owner'
                raise PrefixInvalidated(self._invalidated)

    def _checkpoint(self, position, engine):
        return {'position': position, 'stateId': engine.state_id,
                'state': _encode(engine.state, compressed_limit=self._max_checkpoint_bytes,
                                 raw_limit=self._max_json_bytes)}

    def _put_checkpoint(self, position, engine, item=None):
        self._checkpoints = self._propose_checkpoint(position, engine, item)

    def _propose_checkpoint(self, position, engine, item=None):
        item = self._checkpoint(position, engine) if item is None else item
        if _owned_bytes({position: item}) > self._max_checkpoint_bytes:
            raise PrefixBudgetExceeded('Checkpoint retention budget exceeded')
        proposed = OrderedDict(self._checkpoints)
        proposed[position] = item
        while _owned_bytes(proposed) > self._max_checkpoint_bytes:
            if len(proposed) == 1:
                raise PrefixBudgetExceeded('Checkpoint retention budget exceeded')
            proposed.popitem(last=False)
            # Rebuild after eviction so dictionary capacity left over from the
            # larger map cannot consume a tight one-checkpoint allowance.
            proposed = OrderedDict(proposed)
        return proposed

    def _engine_at(self, position):
        item = self._checkpoints.get(position)
        if item is None:
            raise ValueError('Prefix boundary evicted; create a new read-only owner')
        engine = copy.copy(self._template)
        engine.state = _decode(item['state'], self._max_json_bytes)
        engine._occupancy_cache = None
        # Private execution needs prior status/reason only for exact collision
        # metrics. Complete receipts remain in the bounded compressed store.
        engine.actions = list(self._snapshot['actions'][:position])
        engine.history = self._history[:position]
        if engine.state_id != item['stateId']:
            raise ValueError('Private reconstructed checkpoint identity differs')
        engine._prefix_verified_state_id = item['stateId']
        return engine

    def _frame_bytes(self):
        # Committed frame graphs are private and never mutated after publish.
        # Recount exactly at mapping mutations, not on every compressed chunk.
        return self._retained_frame_bytes + self._stream_metadata_bytes

    def _frame_room(self, stage_bytes):
        while self._frames and self._frame_bytes() + stage_bytes > self._max_frame_bytes:
            self._frames.popitem(last=False)
            self._frames = OrderedDict(self._frames)
            self._retained_frame_bytes = _owned_bytes(self._frames)
        if self._frame_bytes() + stage_bytes > self._max_frame_bytes:
            raise PrefixBudgetExceeded('Observed frame retention budget exceeded during capture')

    def _publish_frames(self, index, frames):
        self._frames, self._retained_frame_bytes = self._propose_frames(index, frames)

    def _propose_frames(self, index, frames):
        # Exact whole-map accounting includes shared keys and OrderedDict table
        # growth. An additive per-action estimate would miss those changes.
        proposed = OrderedDict(self._frames)
        proposed[index] = frames
        retained = _owned_bytes(proposed)
        while retained + self._stream_metadata_bytes > self._max_frame_bytes:
            if len(proposed) == 1:
                raise PrefixBudgetExceeded('Observed frame publication exceeds retention budget')
            proposed.popitem(last=False)
            proposed = OrderedDict(proposed)
            retained = _owned_bytes(proposed)
        return proposed, retained

    def _validate_endpoint(self, engine):
        self._endpoint_verified, self._endpoint_error = self._propose_endpoint(engine)

    def _propose_endpoint(self, engine):
        if len(engine.actions) != len(self._snapshot['actions']):
            return self._endpoint_verified, self._endpoint_error
        checkpoint = self._snapshot.get('checkpoint')
        exact = engine.state_id == self._snapshot['stateId']
        if checkpoint is not None:
            exact = exact and isinstance(checkpoint, Mapping) and (checkpoint.get('position') == len(engine.actions)
                    and checkpoint.get('stateId') == engine.state_id
                    and semantic_digest(checkpoint.get('state')) == semantic_digest(engine.state))
        return bool(exact), None if exact else 'Imported endpoint/checkpoint differs from reconstructed actions'

    @staticmethod
    def _history_store_bytes(store):
        # Include all indexes, metric interning metadata and the retained-byte
        # counter itself. The store is replaced transactionally, never grown
        # behind this exact count. No complete action is stored here.
        size = _owned_bytes(store)
        return size + sys.getsizeof(size)

    def _receipt_digest(self, raw, index, display):
        context = json.dumps([self._origin_hash, self._action_hash, self._protection_hash,
            self._implementation, self._prefix_implementation, index, display], separators=(',', ':')).encode()
        return hashlib.sha256(context + b'\0' + raw).digest()

    def _pack_receipt(self, value, *, index=None, display=False):
        blob = _encode(value, compressed_limit=self._max_history_bytes, raw_limit=self._max_json_bytes)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(blob, self._max_json_bytes + 1)
        if len(raw) > self._max_json_bytes or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise PrefixBudgetExceeded('Decoded receipt JSON budget exceeded')
        return (blob, len(raw), self._receipt_digest(raw, index, display))

    def _unpack_receipt(self, entry, *, index=None, display=False):
        if (type(entry) is not tuple or len(entry) != 3 or type(entry[0]) is not bytes
                or len(entry[0]) > self._max_history_bytes or type(entry[1]) is not int
                or not 0 <= entry[1] <= self._max_json_bytes
                or type(entry[2]) is not bytes or len(entry[2]) != 32):
            raise ValueError('Invalid compressed prefix receipt descriptor')
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(entry[0], entry[1] + 1)
        except zlib.error as error:
            raise ValueError('Corrupt compressed prefix receipt') from error
        if (len(raw) != entry[1] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or self._receipt_digest(raw, index, display) != entry[2]):
            raise ValueError('Compressed prefix receipt identity differs')
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError('Compressed prefix receipt must be a mapping')
        return value

    def _receipt(self, index):
        """One complete immutable receipt; full actions borrow bounded input."""
        self._check()
        if type(index) is not int or not 0 <= index < len(self._receipts):
            raise ValueError('Invalid computed prefix receipt index')
        result = self._complete_receipt(index, self._unpack_receipt(self._receipts[index], index=index))
        self._check()
        return result

    def _complete_receipt(self, index, body):
        if ('action' in body or body.get('status') != self._history[index]['status']
                or body.get('reason') != self._history[index]['reason']):
            raise ValueError('Computed prefix receipt ancestry differs')
        return MappingProxyType({'action': self._snapshot['actions'][index], **freeze_json(body)})

    def _materialize_history(self, *, display=False):
        self._check()
        entries = self._display_receipts if display else self._receipts
        if sum(entry[1] for entry in entries) > self._max_json_bytes:
            raise PrefixBudgetExceeded('Transient complete history JSON budget exceeded')
        rows, seen = [], set()
        retained = sys.getsizeof(rows)
        for index, entry in enumerate(entries):
            body = self._unpack_receipt(entry, index=index, display=display)
            row = freeze_json(body) if display else self._complete_receipt(index, body)
            before = sys.getsizeof(rows)
            rows.append(row)
            retained += _owned_bytes(row, seen) + sys.getsizeof(rows) - before
            # The temporary list and the eventual immutable tuple coexist.
            if retained + sys.getsizeof(()) + len(rows) * 8 > self._max_json_bytes:
                raise PrefixBudgetExceeded('Transient complete history ownership budget exceeded')
        return tuple(rows)

    def _propose_receipt(self, receipt, action, borrowed_action):
        body = dict(receipt)
        if body.pop('action') != action:
            raise ValueError('Reconstructed receipt action differs from imported action')
        index = len(self._history)
        entry = self._pack_receipt(body, index=index)
        metric_key = (body['status'], body.get('reason'))
        metrics = dict(self._metric_rows)
        metric = metrics.get(metric_key)
        if metric is None:
            metric = freeze_json({'status': metric_key[0], 'reason': metric_key[1]})
            metrics[metric_key] = metric
        displayed = self._display_receipts[:]
        if self._display_row_projection is not None:
            complete = MappingProxyType({'action': borrowed_action, **freeze_json(body)})
            projected = thaw_json(freeze_json(self._display_row_projection(complete)))
            if type(projected) is not dict:
                raise ValueError('Prefix display row projection must return a mapping')
            displayed.append(self._pack_receipt(projected, index=index, display=True))
        store = (self._history + [metric], self._receipts + [entry], displayed, metrics)
        size = self._history_store_bytes(store)
        if size > self._max_history_bytes:
            raise PrefixBudgetExceeded('Computed history retention budget exceeded')
        return store, size

    @property
    def statistics(self):
        return {'verifiedThrough': len(self._history), 'totalActions': len(self._snapshot['actions']),
                'checkpointBytes': _owned_bytes(self._checkpoints), 'maxCheckpointBytes': self._max_checkpoint_bytes,
                'frameBytes': self._frame_bytes(), 'maxFrameBytes': self._max_frame_bytes,
                'streamMetadataBytes': self._stream_metadata_bytes,
                'historyBytes': self._retained_history_bytes, 'maxHistoryBytes': self._max_history_bytes,
                'historyActionsSharedWithBoundedInput': True,
                'historyReceiptsCompressed': True, 'displayRowsCompressed': self._display_row_projection is not None,
                'maxMaterializedHistoryGraphBytes': self._max_json_bytes,
                'materializedHistoryLimitMeaning': 'Per-graph limit; decoded, frozen and thawed output graphs may coexist; whole-process RSS is separate',
                'immutableInputBytes': self._input_bytes,
                'ownedImmutableInputBytes': 0 if self._borrowed_input else self._input_bytes,
                'sharedImmutableInputBytes': self._input_bytes if self._borrowed_input else 0,
                'immutableInputBorrowed': self._borrowed_input,
                'checkpointPositions': list(self._checkpoints), 'frameActions': list(self._frames),
                'memoryMeaning': 'Checkpoint/frame/receipt ownership; immutable input reported separately; source anatomy, one private pure geometry/protection runtime and transient engine/view allocations excluded'}

    def _result(self, engine, position, requested, *, budget_stopped=False, timeline_projection=None,
                compact=False, omit_fields=()):
        verification = {'status': 'reconstructed-prefix', 'readOnly': True,
            'verifiedThrough': len(self._history), 'totalActions': len(self._snapshot['actions']),
            'originSnapshotHash': self._origin_hash, 'actionLogHash': self._action_hash,
            'protectionContractHash': self._protection_hash, 'implementation': self._implementation,
            'prefixImplementation': self._prefix_implementation,
            'archiveEndpointVerified': self._endpoint_verified,
            'endpointValidationError': self._endpoint_error,
            'failure': thaw_json(freeze_json(self._failure))}
        timeline = ()
        if not compact and timeline_projection is not None:
            # Private display-only callback sees immutable complete receipts;
            # its result is detached JSON. It never receives an engine/state.
            projected = timeline_projection(self._materialize_history())
            if _owned_bytes(projected) > self._max_json_bytes:
                raise PrefixBudgetExceeded('Projected history ownership budget exceeded')
            timeline = freeze_json(projected)
        elif not compact:
            timeline = self._materialize_history(display=self._display_row_projection is not None)
        view = engine.view(_state_id=engine._prefix_verified_state_id,
                           include_history=False, _omit_fields=omit_fields)
        if compact:
            for name in ('scene', 'catalog', 'protection'):
                view.pop(name, None)
        return {'view': view, **({} if compact else {'timeline': thaw_json(timeline)}), 'position': position,
                'requestedPosition': requested, 'budgetStopped': budget_stopped,
                'readOnly': True, 'verification': verification}

    def _ensure_prefix(self, position, check):
        self._check(check)
        if type(position) is not int or not 0 <= position <= len(self._snapshot['actions']):
            raise ValueError('Invalid prefix boundary')
        target = min(position, len(self._history) + self._max_new_actions)
        def action_check():
            if check is not None:
                try:
                    check()
                except BaseException as error:
                    raise _Interrupted(error) from None
        while len(self._history) < target and not self._blocked:
            index = len(self._history)
            engine = self._engine_at(index)
            stage = _FrameStage(self)
            try:
                self._check(action_check)
                before = engine.state_id
                stage.begin(engine)
                engine._stage = stage
                action = thaw_json(self._snapshot['actions'][index])
                outcome = engine.step(action, check=action_check)
                if outcome['status'] not in ('accepted', 'partial'):
                    raise ValueError('Saved action rejected: ' + str(outcome.get('reason')))
                if (len(engine.actions) != index + 1 or engine.actions[-1] != action
                        or engine.history[-1]['stateBefore'] != before
                        or engine.history[-1]['stateAfter'] != engine.state_id
                        or outcome['stateId'] != engine.state_id):
                    raise ValueError('Reconstructed public action ancestry differs')
                stage.end(engine)
                if stage.frames[0]['stateId'] != before or stage.frames[-1]['stateId'] != engine.state_id:
                    raise ValueError('Observed start/final identity differs from public action')
                store, history_bytes = self._propose_receipt(engine.history[-1], action,
                                                           self._snapshot['actions'][index])
                checkpoint = self._checkpoint(index + 1, engine)
                if _owned_bytes(OrderedDict([(index + 1, checkpoint)])) > self._max_checkpoint_bytes:
                    raise PrefixBudgetExceeded('Checkpoint retention budget exceeded')
                # Account the eventual owning mapping as well as staged frames.
                self._frame_room(_owned_bytes(OrderedDict([(index, stage.frames)])))
                checkpoints = self._propose_checkpoint(index + 1, engine, checkpoint)
                frames, frame_bytes = self._propose_frames(index, stage.frames)
                endpoint_verified, endpoint_error = self._propose_endpoint(engine)
                self._check(action_check, full=True)
                # Every allocation/validation above precedes publication. A
                # failed proposal cannot evict the committed boundary cursor.
                self._checkpoints = checkpoints
                self._frames, self._retained_frame_bytes = frames, frame_bytes
                self._history_store = store
                self._history, self._receipts, self._display_receipts, self._metric_rows = store
                self._retained_history_bytes = history_bytes
                self._failure = None
                self._endpoint_verified, self._endpoint_error = endpoint_verified, endpoint_error
            except BaseException as error:
                cause = error.cause if isinstance(error, _Interrupted) else error
                self._failure = {'actionIndex': index, 'type': type(cause).__name__, 'reason': str(cause)}
                if isinstance(error, _Interrupted):
                    raise cause from None
                if isinstance(error, PrefixInvalidated):
                    raise
                if isinstance(error, (PrefixBudgetExceeded, ValueError)):
                    self._blocked = True
                    break
                raise
        actual = min(position, len(self._history))
        self._check(check)
        return actual

    def seek(self, position, *, check=None, _timeline_projection=None):
        actual = self._ensure_prefix(position, check)
        result = self._result(self._engine_at(actual), actual, position,
                              budget_stopped=actual < position, timeline_projection=_timeline_projection)
        self._check(check)
        return result

    def _timeline_token(self):
        # This prefix-only domain can never acknowledge canonical replay data.
        # One owner nonce binds even equal archives in independent sessions.
        return 'prefix-replay.v1/' + semantic_digest({
            'nonce': self._stream_nonce, 'scene': self._scene.identity,
            'origin': self._origin_hash, 'actions': self._action_hash,
            'protection': self._protection_hash, 'implementation': self._implementation,
            'prefixImplementation': self._prefix_implementation,
            'verifiedThrough': len(self._history)})[7:]

    def seek_frame(self, action_index, frame_index, *, check=None, _timeline_projection=None,
                   prefix_token=None, prefix_field_tokens=None):
        """Return a full frame unless exact prefix-domain acknowledgements match.

        Tokens suppress redundant display fields only. They never change saved
        actions, reconstruction limits, verification, progress or authority.
        Missing/stale tokens receive full data; malformed acknowledgements fail.
        """
        if prefix_token is not None and (not isinstance(prefix_token, str) or not 1 <= len(prefix_token) <= 1024):
            raise ValueError('Invalid prefix timeline acknowledgement')
        known_fields = {} if prefix_field_tokens is None else prefix_field_tokens
        if (not isinstance(known_fields, dict) or set(known_fields) - {'flapComponents'}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in '0123456789abcdef' for char in value)
                       for value in known_fields.values())):
            raise ValueError('Invalid prefix view field acknowledgement')
        if type(action_index) is not int or not 0 <= action_index < len(self._snapshot['actions']):
            raise ValueError('Invalid observed action index')
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError('Invalid observed frame index')
        self._check(check)
        if action_index >= len(self._history):
            self._ensure_prefix(action_index + 1, check)
        item = self._frames.get(action_index)
        if item is None or frame_index >= len(item):
            raise ValueError('Observed frame unavailable: action not reconstructed or frames evicted')
        frame = item[frame_index]
        engine = copy.copy(self._template)
        engine.state = _decode(frame['state'], self._max_json_bytes)
        engine._occupancy_cache = None
        final = frame_index == len(item) - 1
        position = action_index + int(final)
        engine.actions = list(self._snapshot['actions'][:position])
        engine.history = self._history[:position]
        if engine.state_id != frame['stateId']:
            raise ValueError('Private observed frame identity differs')
        engine._prefix_verified_state_id = frame['stateId']
        timeline_token = self._timeline_token()
        compact = prefix_token == timeline_token
        field_tokens = dict(frame['fieldTokens'])
        omitted = [name for name, digest in field_tokens.items()
                   if compact and known_fields.get(name) == digest]
        result = self._result(engine, position, action_index + 1, timeline_projection=_timeline_projection,
                              compact=compact, omit_fields=omitted)
        result.update(timelineToken=timeline_token, timelineOmitted=compact,
                      viewFieldTokens=field_tokens, viewFieldsOmitted=omitted)
        result['playback'] = {'readOnly': True, 'position': action_index, 'actionIndex': action_index,
            'frameIndex': frame_index, 'frameCount': len(item),
            'frames': [{key: (dict(value) if key == 'effectiveActivation' else value)
                        for key, value in row.items() if key not in ('state', 'fieldTokens')} | {'frameIndex': index}
                       for index, row in enumerate(item)],
            'observedTimeSeconds': frame['observedTimeSeconds'],
            'actionElapsedSeconds': frame['observedTimeSeconds'] - item[0]['observedTimeSeconds'],
            'actionDurationSeconds': item[-1]['observedTimeSeconds'] - item[0]['observedTimeSeconds'],
            'isActionBoundary': frame_index in (0, len(item) - 1),
            'effectiveActivation': dict(frame['effectiveActivation']), 'frameStateId': frame['stateId']}
        self._check(check)
        # The view and timeline are already detached. Only each descriptor's
        # nested activation map required a fresh copy above.
        return result
