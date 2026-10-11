"""Bounded certificates for an explicit idealized cumulative cutting model.

A certified subcell represents an assumed cut AND evacuated volume under an
executed active aspiration sweep. This is not inferred from contact, accessibility
or anatomy, and is not validated tissue mechanics. Native cells remain occupied
until their entire fixed affine partition is certified and the native connected
surface rule permits removal. Unknown/uncovered leaves never become free space.
"""
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
import math

import numpy as np

from .core import semantic_digest
from .geometry import point_segment_distances

MODE = "cumulative_affine_subcell_cut_union_v1"
MARGIN_MM = 1e-10
MAX_WITNESS_ROWS_PER_PREVIEW = 8192
ASSUMPTION = "idealized_exposed_active_subcell_cut_and_evacuation_not_validated_tissue_mechanics"


@dataclass(frozen=True)
class CutUnionSpec:
    max_depth: int = 3
    max_tracked_cells: int = 4096
    max_leaf_tests_per_preview: int = 4194304

    def __post_init__(self):
        for name, upper in (("max_depth", 3), ("max_tracked_cells", 4096),
                            ("max_leaf_tests_per_preview", 4194304)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("Invalid bounded cumulative cutting field: " + name)

    def to_record(self):
        return {"mode": MODE, "max_depth": self.max_depth,
                "max_tracked_cells": self.max_tracked_cells,
                "max_leaf_tests_per_preview": self.max_leaf_tests_per_preview}

    @property
    def leaf_count(self):
        return 8 ** self.max_depth

    @property
    def full_mask(self):
        return (1 << self.leaf_count)-1


class CutUnionLimit(RuntimeError):
    """Reject the whole preview; never commit a partially updated certificate."""


def ledger_digest(spec, ledger):
    if type(spec) is not CutUnionSpec or len(ledger) > spec.max_tracked_cells:
        raise ValueError("Invalid cumulative cutting ledger or declaration")
    rows = []
    for cell, mask in sorted(ledger.items()):
        if (type(cell) is not tuple or len(cell) != 3 or any(type(v) is not int or v < 0 for v in cell)
                or type(mask) is not int or not 0 < mask <= spec.full_mask):
            raise ValueError("Invalid canonical cumulative cutting ledger entry")
        rows.append({"index": list(cell), "covered_bits_hex": format(mask, "x")})
    return semantic_digest({"mode": MODE, "spec": spec.to_record(), "cells": rows})


def _exposed(cell, free):
    for dimension in range(3):
        for sign in (-1, 1):
            neighbor = list(cell); neighbor[dimension] += sign
            if neighbor[dimension] < 0 or neighbor[dimension] >= free.shape[dimension]:
                return True
            if free[tuple(neighbor)]: return True
    return False


class PreviewCutCoverage:
    """Private transactional ledger; the engine installs it only after commit."""
    def __init__(self, spec, ledger, affine):
        if type(spec) is not CutUnionSpec:
            raise TypeError("An explicit typed cumulative cutting declaration is required")
        ledger_digest(spec, ledger)
        self.spec, self.cells = spec, dict(ledger)
        self.affine = np.asarray(affine, float)
        if self.affine.shape != (4, 4) or not np.isfinite(self.affine).all():
            raise ValueError("Finite affine cell partition required")
        self.new_leaf_count = self.leaf_tests = self.witness_rows = 0
        self.peak_tracked_cells = len(self.cells)
        n = 2 ** spec.max_depth
        self._leaf_centers = np.asarray(list(product(range(n), repeat=3)), float)/n + .5/n - .5
        self._corners = np.asarray(list(product((-.5/n, .5/n), repeat=3))) @ self.affine[:3, :3].T

    @property
    def digest(self):
        return ledger_digest(self.spec, self.cells)

    def cover(self, occupied, direct_whole_cells, connected_free, start, end, radius):
        """Certify only NEW leaves in this one capsule, after its shaft passed.

        direct_whole_cells is the existing contained-and-connected whole-cell
        set, so the fast path preserves that same-step connected-component rule.
        Partial cells must be face-exposed before this interval's removals.
        """
        start, end = np.asarray(start, float), np.asarray(end, float)
        if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all()
                or not np.isfinite(end).all() or not math.isfinite(radius) or radius <= MARGIN_MM):
            raise ValueError("Finite positive executed active capsule required")
        before = self.digest; before_leaves = self.new_leaf_count; before_tests = self.leaf_tests
        direct = {tuple(int(v) for v in cell) for cell in direct_whole_cells}
        rows = []
        for value in occupied:
            cell = tuple(int(v) for v in value)
            old = self.cells.get(cell, 0)
            if old == self.spec.full_mask: continue
            if cell in direct:
                added = self.spec.full_mask ^ old; proof = "whole_cell_capsule"
            elif _exposed(cell, connected_free):
                pending = [i for i in range(self.spec.leaf_count) if not (old >> i) & 1]
                if self.leaf_tests+len(pending) > self.spec.max_leaf_tests_per_preview:
                    raise CutUnionLimit("Leaf proof work ceiling")
                self.leaf_tests += len(pending)
                centers = (self._leaf_centers[pending]+value) @ self.affine[:3, :3].T+self.affine[:3, 3]
                corners = centers[:, None, :]+self._corners
                contained = np.all(point_segment_distances(corners, start, end) <= radius-MARGIN_MM, axis=1)
                added = sum(1 << pending[i] for i in np.flatnonzero(contained))
                proof = "leaf_corners"
            else:
                continue
            if not added: continue
            if cell not in self.cells and len(self.cells) >= self.spec.max_tracked_cells:
                raise CutUnionLimit("Tracked cell ceiling")
            if self.witness_rows >= MAX_WITNESS_ROWS_PER_PREVIEW:
                raise CutUnionLimit("Sparse witness row ceiling")
            self.cells[cell] = old | added
            self.peak_tracked_cells = max(self.peak_tracked_cells, len(self.cells))
            self.new_leaf_count += added.bit_count(); self.witness_rows += 1
            rows.append({"index": list(cell), "added_bits_hex": format(added, "x"), "proof": proof})
        return {"before_digest": before, "new_leaf_count": self.new_leaf_count-before_leaves,
                "leaf_tests": self.leaf_tests-before_tests, "cells": rows}

    def complete(self, occupied):
        return np.asarray([cell for cell in occupied
            if self.cells.get(tuple(int(v) for v in cell), 0) == self.spec.full_mask], dtype=np.int64).reshape(-1, 3)

    def finish_step(self, record, removed):
        for value in removed:
            cell = tuple(int(v) for v in value)
            if self.cells.get(cell) != self.spec.full_mask:
                raise ValueError("Uncertified cell cannot become removed")
            del self.cells[cell]
        return {**record, "after_digest": self.digest}

    def empty_step(self):
        return {"before_digest": self.digest, "after_digest": self.digest,
                "new_leaf_count": 0, "leaf_tests": 0, "cells": []}

    def freeze(self):
        ledger_digest(self.spec, self.cells)
        return MappingProxyType(dict(self.cells))
