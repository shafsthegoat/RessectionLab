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
        for at, i in np.ndenumerate(ids):
            if int(i) in self.moved: values[at] = self.moved[int(i)]
            elif int(i) in self.cleared: values[at] = 0
        return values

    def material_ids(self, indices):
        result = set()
        for cell, value in zip(indices, self[indices]):
            if int(cell) in self.memberships: result.update(self.memberships[int(cell)])
            elif value: result.add(int(value) - 1)
        return sorted(result)

    def __array__(self, dtype=None, copy=None):
        result = np.where(self.labels != 0, np.arange(len(self.labels), dtype=np.int32) + 1, 0)
        if self.cleared: result[list(self.cleared)] = 0
        if self.moved:
            result[list(self.moved)] = list(self.moved.values())
        return result.astype(dtype, copy=False) if dtype is not None else result
