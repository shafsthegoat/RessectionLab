"""Invocation-local, bounded copies of fully checked public action prefixes.

No engines, scenes, callbacks or observations are stored here. The caller owns
one qualified public model and binds this cache to its complete immutable input.
This is disposable exact computation reuse, never persisted validation authority.
"""
from __future__ import annotations

import copy
import sys
from collections import OrderedDict

from .core import semantic_digest

MAX_ENTRIES = 64
MAX_BYTES = 16 * 1024**2


def _json_size(value):
    """Conservative owned-object accounting; shared leaves count per entry.

    Unknown objects bypass caching rather than invoking caller deepcopy hooks.
    Public engine checkpoints contain only these built-in JSON data structures.
    """
    seen = set(); pending = [value]; size = 0
    while pending:
        item = pending.pop()
        if id(item) in seen: continue
        seen.add(id(item)); size += sys.getsizeof(item)
        if type(item) is dict:
            pending.extend(item.keys()); pending.extend(item.values())
        elif type(item) in (list, tuple): pending.extend(item)
        elif type(item) not in (str, int, float, bool, type(None)): return None
    return size


class ProgramPrefixCache:
    def __init__(self, *, max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES):
        if any(type(v) is not int or v < 0 for v in (max_entries, max_bytes)):
            raise ValueError('Public prefix cache limits must be nonnegative integers')
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict(); self._bytes = self._peak_bytes = 0
        self._hits = self._misses = self._stores = self._evictions = self._bypasses = 0
        self._closed = False

    def get(self, actions, *, check):
        check()
        if self._closed: raise ValueError('Public prefix cache is closed')
        key = semantic_digest(actions)
        found = self._entries.get(key)
        if found is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key); self._hits += 1
        value = copy.deepcopy(found[0])
        check()
        return value

    def put(self, actions, payload, *, check):
        check()
        if self._closed: raise ValueError('Public prefix cache is closed')
        key = semantic_digest(actions)
        size = _json_size(payload)
        if size is None or not self.max_entries or not self.max_bytes or size+sys.getsizeof(key) > self.max_bytes:
            self._bypasses += 1
            return
        value = copy.deepcopy(payload)
        # Count copied container capacity, not just the caller's allocation.
        size = _json_size(value)+sys.getsizeof(key)
        check()
        if size > self.max_bytes:
            self._bypasses += 1
            return
        previous = self._entries.pop(key, None)
        if previous is not None: self._bytes -= previous[1]
        while self._entries and (len(self._entries) >= self.max_entries or self._bytes+size > self.max_bytes):
            _, old = self._entries.popitem(last=False)
            self._bytes -= old[1]; self._evictions += 1
        self._entries[key] = value, size
        self._bytes += size; self._peak_bytes = max(self._bytes, self._peak_bytes); self._stores += 1

    def stats(self):
        return {'hits': self._hits, 'misses': self._misses, 'stores': self._stores,
            'evictions': self._evictions, 'bypasses': self._bypasses,
            'entries': len(self._entries), 'maxEntries': self.max_entries,
            'accountedPayloadBytes': self._bytes, 'peakAccountedPayloadBytes': self._peak_bytes,
            'maxPayloadBytes': self.max_bytes, 'closed': self._closed}

    def close(self):
        self._entries.clear(); self._bytes = 0; self._closed = True
