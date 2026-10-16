"""Bounded draft edits. Only explicit commit creates an anatomy revision."""

import numpy as np


class AnatomyDraft:
    def __init__(self, collection, layer_id):
        self.snapshot_id = collection.identity()
        self.layer_id = layer_id
        self.original = collection.mask(collection.layer(layer_id))
        self.mask = self.original.copy()
        self.undo = []
        self.redo = []
        self.operations = []

    def apply(self, collection, operation, **kwargs):
        if collection.identity() != self.snapshot_id:
            raise ValueError("Draft belongs to a stale anatomy snapshot")
        after = collection.edit(self.layer_id, operation, mask=self.mask, **kwargs)
        indices = np.flatnonzero(after.ravel() != self.mask.ravel()).astype(np.uint32)
        if indices.nbytes > 256 * 1024**2:
            raise ValueError("Edit exceeds undo memory budget")
        self.undo.append((indices, self.mask.ravel()[indices].copy(), operation))
        while sum(x[0].nbytes + x[1].nbytes for x in self.undo) > 256 * 1024**2:
            self.undo.pop(0)
        self.redo.clear()
        self.mask = after
        self.operations.append(
            {
                "operation": operation,
                **{
                    k: (np.asarray(v).tolist() if k == "points" else v)
                    for k, v in kwargs.items()
                },
            }
        )

    def step(self, direction):
        source, destination = (
            (self.undo, self.redo) if direction == "undo" else (self.redo, self.undo)
        )
        if not source:
            return
        indices, values, op = source.pop()
        destination.append((indices, self.mask.ravel()[indices].copy(), op))
        self.mask.ravel()[indices] = values
        self.operations.append({"operation": direction})

    def replace(self, after, operation):
        indices = np.flatnonzero(after.ravel() != self.mask.ravel()).astype(np.uint32)
        if indices.nbytes > 256 * 1024**2:
            raise ValueError("Edit exceeds undo memory budget")
        self.undo.append((indices, self.mask.ravel()[indices].copy(), operation['operation']))
        while sum(x[0].nbytes + x[1].nbytes for x in self.undo) > 256 * 1024**2:
            self.undo.pop(0)
        self.redo.clear()
        self.mask = after
        self.operations.append(operation)
