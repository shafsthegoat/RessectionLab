"""Sparse material identity lookup over immutable source cells.

No patient-sized occupancy array is allocated during transitions. Dense conversion
is explicitly available for inspection/tests, never required by the engine.
"""
from itertools import repeat
from operator import is_

import numpy as np
from .core import immutable_array
from . import native_backend

_native_fallback = object()


_MAX_SORTED_CLEARED_BYTES = 8 * 1024**2
# A cover at most this size, or small beside the sparse maps, is resolved cell
# by cell against the live mappings. Larger covers use the array paths, whose
# per-call cost is proportional to the moved and membership maps.
_SMALL_QUERY_CELLS = 64


def _exact_integer_set(cells):
    """True when every member is an exact int in int64 range (as the original generator test)."""
    if not cells:
        return True
    return (all(map(is_, map(type, cells), repeat(int)))
            and min(cells) >= -(1 << 63) and max(cells) < (1 << 63))


class MaterialOccupancy:
    def __init__(self, labels, cleared, moved, memberships=None):
        self.memberships = memberships or {}
        self.labels = labels.ravel()
        self.cleared = frozenset(cleared)
        self._integer_cleared = self.cleared if _exact_integer_set(self.cleared) else None
        self._sorted_integer_cleared = None
        self.moved = dict(moved)

    def _small_query(self, ids, sparse):
        """Admit a one-dimensional cover small enough for exact per-cell lookups."""
        return (ids.ndim == 1 and self.cleared is self._integer_cleared and type(self.moved) is dict
                and (ids.size <= _SMALL_QUERY_CELLS or ids.size * 8 <= sparse))

    def _cell_values(self, ids):
        """Per-cell occupancy of a 1-D int64 cover: moved owner, else cleared/empty 0, else cell + 1.

        This is the rule of the native and array paths, read from the live
        public mappings, so a caller's edit between queries is still honoured.
        """
        present = (self.labels[ids] != 0).tolist()
        moved, cleared = self.moved, self.cleared
        values = []
        for cell, occupied in zip(ids.tolist(), present):
            owner = moved.get(cell)
            values.append((0 if not occupied or cell in cleared else cell + 1) if owner is None else owner)
        return values

    def __getitem__(self, indices):
        if np.isscalar(indices):
            i = int(indices)
            return self.moved.get(i, 0 if i in self.cleared or not self.labels[i] else i + 1)
        ids = np.asarray(indices, dtype=np.int64)
        if self._small_query(ids, len(self.moved)):
            return np.array(self._cell_values(ids), dtype=np.int64).reshape(ids.shape)
        if self.cleared is self._integer_cleared:
            native = native_backend.occupancy_values(self.labels, ids, self.cleared, self.moved, _native_fallback)
            if native is not _native_fallback:
                return native
        values = np.where(self.labels[ids] != 0, ids + 1, 0)
        if self.cleared:
            if self.cleared is self._integer_cleared and ids.size <= len(self.cleared):
                # Collision covers are small while the cleared ledger grows
                # throughout an episode. Query its immutable hash set directly
                # instead of rebuilding/sorting the complete ledger each time.
                cleared = np.fromiter((cell in self.cleared for cell in ids.flat),
                                      dtype=np.bool_, count=ids.size).reshape(ids.shape)
            elif self.cleared is self._integer_cleared and len(self.cleared) <= _MAX_SORTED_CLEARED_BYTES // 8:
                # The owner already holds an immutable exact cleared set. Large
                # capsule covers need no repeated unique/sort of that ledger.
                # Search accepts arbitrary query ordering, shape and duplicates.
                if self._sorted_integer_cleared is None:
                    self._sorted_integer_cleared = immutable_array(sorted(self.cleared), np.int64)
                ledger = self._sorted_integer_cleared
                positions = np.minimum(np.searchsorted(ledger, ids), len(ledger) - 1)
                cleared = ledger[positions] == ids
            else:
                cleared = np.isin(ids, np.fromiter(self.cleared, dtype=np.int64))
            values[cleared] = 0
        if self.moved:
            # Moved ownership takes precedence over a cleared source location.
            # Rebuild these small sparse arrays so the public mapping remains
            # authoritative even if a caller updates it between queries.
            keys = np.array(sorted(self.moved), dtype=np.int64)
            positions = np.searchsorted(keys, ids)
            positions = np.minimum(positions, len(keys) - 1)
            matched = keys[positions] == ids
            owners = np.array([self.moved[int(k)] for k in keys], dtype=np.int64)
            values[matched] = owners[positions[matched]]
        return values

    def material_ids(self, indices):
        ids = np.asarray(indices, dtype=np.int64)
        memberships = self.memberships
        if (type(memberships) is dict and type(self).__getitem__ is MaterialOccupancy.__getitem__
                and self._small_query(ids, len(self.moved) + 4 * len(memberships))):
            # Same set, filled in the same order: retained single owners in
            # cover order, then every shared cell's sources in ascending cell order.
            result = set()
            shared = []
            for cell, value in zip(ids.tolist(), self._cell_values(ids)):
                if cell in memberships:
                    shared.append(cell)
                elif value != 0:
                    result.add(int(value) - 1)
            for cell in sorted(set(shared)):
                result.update(memberships[cell])
            return sorted(result)
        values = self[ids]
        if self.memberships:
            shared = np.isin(ids, np.fromiter(self.memberships, dtype=np.int64))
            result = set((values[(values != 0) & ~shared] - 1).tolist())
            for cell in np.unique(ids[shared]):
                result.update(self.memberships[int(cell)])
        else:
            result = set((values[values != 0] - 1).tolist())
        return sorted(result)

    def __array__(self, dtype=None, copy=None):
        result = np.where(self.labels != 0, np.arange(len(self.labels), dtype=np.int32) + 1, 0)
        if self.cleared: result[list(self.cleared)] = 0
        if self.moved:
            result[list(self.moved)] = list(self.moved.values())
        return result.astype(dtype, copy=False) if dtype is not None else result
