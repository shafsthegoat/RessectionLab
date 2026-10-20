"""Sparse material identity lookup over immutable source cells.

No patient-sized occupancy array is allocated during transitions. Dense conversion
is explicitly available for inspection/tests, never required by the engine.
"""
import numpy as np


class MaterialOccupancy:
    def __init__(self, labels, cleared, moved, memberships=None):
        self.memberships = memberships or {}
        self.labels = labels.ravel()
        self.cleared = frozenset(cleared)
        self.moved = dict(moved)

    def __getitem__(self, indices):
        if np.isscalar(indices):
            i = int(indices)
            return self.moved.get(i, 0 if i in self.cleared or not self.labels[i] else i + 1)
        ids = np.asarray(indices, dtype=np.int64)
        values = np.where(self.labels[ids] != 0, ids + 1, 0)
        if self.cleared:
            values[np.isin(ids, np.fromiter(self.cleared, dtype=np.int64))] = 0
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
