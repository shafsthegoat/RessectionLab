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

from .core import semantic_digest, array_digest, immutable_array
from .geometry import point_segment_distances

MODE = "cumulative_affine_subcell_cut_union_v1"
PARENT_CLOSURE_EXPOSURE = "same_step_parent_closure"
PARENT_CLOSURE_MODE = "cumulative_affine_parent_closure_cut_union_v3"
PARENT_CLOSURE_EVACUATED_FOOTPRINT = "cumulative_affine_parent_closure_evacuation_v3"
TERMINAL_CLOSURE_SCHEDULE = "terminal_microstep_only"
TERMINAL_CLOSURE_MODE = "cumulative_affine_terminal_parent_closure_cut_union_v4"
TERMINAL_CLOSURE_EVACUATED_FOOTPRINT = "cumulative_affine_terminal_parent_closure_evacuation_v4"
MARGIN_MM = 1e-10
MAX_WITNESS_ROWS_PER_PREVIEW = 8192
ASSUMPTION = "idealized_exposed_active_subcell_cut_and_evacuation_not_validated_tissue_mechanics"
EVACUATED_VOLUME_VERSION = "certified_evacuated_subcell_volume_v1"
EVACUATED_FOOTPRINT = "cumulative_affine_subcell_evacuation_v2"
EVACUATED_VOLUME_SCOPE = ("certified_subset_of_represented_source_support_under_idealized_evacuation;"
                         "whole_parent_shaft_envelope_and_reward_unchanged")


@dataclass(frozen=True)
class EvacuatedVolumeSpec:
    """Explicit unlocated parent target mass, independent of reward labels.

    Equal endpoints declare exact represented parent mass; unequal endpoints
    retain its uncertainty. Neither array identifies target inside a subcell.
    There is deliberately no fallback to labels or evaluator anatomy.
    """
    target_fraction_lower: np.ndarray
    target_fraction_upper: np.ndarray
    provenance: str

    def __post_init__(self):
        values = []
        for name in ("target_fraction_lower", "target_fraction_upper"):
            value = np.asarray(getattr(self, name))
            if (value.ndim != 3 or min(value.shape) < 1 or value.dtype.kind not in "biuf"
                    or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)):
                raise ValueError("Target mass bounds require finite source-shaped fractions in [0,1]")
            values.append(immutable_array(value, np.float64))
        if values[0].shape != values[1].shape or np.any(values[0] > values[1]):
            raise ValueError("Target fraction lower/upper bounds must have equal shape and be ordered")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Explicit target mass provenance is required")
        object.__setattr__(self, "target_fraction_lower", values[0])
        object.__setattr__(self, "target_fraction_upper", values[1])

    def to_record(self):
        return {"version": EVACUATED_VOLUME_VERSION,
                "shape": list(self.target_fraction_lower.shape),
                "target_fraction_lower_hash": array_digest(self.target_fraction_lower),
                "target_fraction_upper_hash": array_digest(self.target_fraction_upper),
                "provenance": self.provenance,
                "target_semantics": "unlocated_parent_target_mass_bounds"}


@dataclass(frozen=True)
class CutUnionSpec:
    max_depth: int = 3
    max_tracked_cells: int = 4096
    max_leaf_tests_per_preview: int = 4194304
    exposure_mode: str | None = None
    closure_schedule: str | None = None

    def __post_init__(self):
        if self.exposure_mode is not None and (type(self.exposure_mode) is not str
                or self.exposure_mode != PARENT_CLOSURE_EXPOSURE):
            raise ValueError("Unsupported cumulative cutting exposure mode")
        if self.closure_schedule is not None and (type(self.closure_schedule) is not str
                or self.closure_schedule != TERMINAL_CLOSURE_SCHEDULE
                or self.exposure_mode != PARENT_CLOSURE_EXPOSURE):
            raise ValueError("Terminal closure schedule requires explicit parent-closure exposure")
        for name, upper in (("max_depth", 3), ("max_tracked_cells", 4096),
                            ("max_leaf_tests_per_preview", 4194304)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("Invalid bounded cumulative cutting field: " + name)

    def to_record(self):
        return {"mode": (TERMINAL_CLOSURE_MODE if self.closure_schedule is not None else
                         MODE if self.exposure_mode is None else PARENT_CLOSURE_MODE), "max_depth": self.max_depth,
                "max_tracked_cells": self.max_tracked_cells,
                "max_leaf_tests_per_preview": self.max_leaf_tests_per_preview,
                **({"exposure_mode": self.exposure_mode} if self.exposure_mode is not None else {}),
                **({"closure_schedule": self.closure_schedule} if self.closure_schedule is not None else {})}

    @property
    def leaf_count(self):
        return 8 ** self.max_depth

    @property
    def full_mask(self):
        return (1 << self.leaf_count)-1


class CutUnionLimit(RuntimeError):
    """Reject the whole preview; never commit a partially updated certificate."""


def _validated_ledger_items(spec, ledger):
    if type(spec) is not CutUnionSpec or len(ledger) > spec.max_tracked_cells:
        raise ValueError("Invalid cumulative cutting ledger or declaration")
    items = tuple(sorted(ledger.items()))
    for cell, mask in items:
        if (type(cell) is not tuple or len(cell) != 3 or any(type(v) is not int or v < 0 for v in cell)
                or type(mask) is not int or not 0 < mask <= spec.full_mask):
            raise ValueError("Invalid canonical cumulative cutting ledger entry")
    return items


def _digest_ledger_items(spec, items):
    rows = [{"index": list(cell), "covered_bits_hex": format(mask, "x")} for cell, mask in items]
    return semantic_digest({"mode": spec.to_record()["mode"], "spec": spec.to_record(), "cells": rows})


def ledger_digest(spec, ledger):
    return _digest_ledger_items(spec, _validated_ledger_items(spec, ledger))


def evacuated_volume_snapshot(volume_spec, cut_spec, ledger, removed_mask, affine):
    """Read the certified subset without changing the conservative obstacle.

    Retired full-parent bitsets are represented by removed_mask. Partial bits
    remain separate, so a parent that clears later never receives volume twice.
    Target intervals are cumulative allocations of each parent's mass; interval
    endpoints must not be differenced or summed as exact action target rewards.
    """
    if type(volume_spec) is not EvacuatedVolumeSpec:
        raise TypeError("Explicit evacuated volume declaration required")
    items = _validated_ledger_items(cut_spec, ledger)
    removed = np.asarray(removed_mask)
    matrix = np.asarray(affine, np.float64)
    if (removed.dtype != np.dtype(bool) or removed.shape != volume_spec.target_fraction_lower.shape
            or matrix.shape != (4,4) or not np.isfinite(matrix).all()):
        raise ValueError("Evacuation requires a matching whole-parent mask and finite affine")
    voxel = abs(float(np.linalg.det(matrix[:3, :3])))
    if not math.isfinite(voxel) or voxel <= 0:
        raise ValueError("Positive finite represented source-cell volume required")
    full_indices = np.argwhere(removed)
    lower, upper = [], []
    for value in full_indices:
        cell = tuple(value)
        lower.append(float(volume_spec.target_fraction_lower[cell])*voxel)
        upper.append(float(volume_spec.target_fraction_upper[cell])*voxel)
    partial_leaves = 0
    for cell, mask in items:
        if any(i >= size for i, size in zip(cell, removed.shape)) or removed[cell]:
            raise ValueError("Partial certificate overlaps a removed or out-of-grid parent")
        count = mask.bit_count()
        if count == cut_spec.leaf_count:
            raise ValueError("A fully certified parent must clear in its current microstep")
        partial_leaves += count
        evacuated = count*(voxel/cut_spec.leaf_count)
        target_lower = float(volume_spec.target_fraction_lower[cell])*voxel
        target_upper = float(volume_spec.target_fraction_upper[cell])*voxel
        lower.append(max(0., evacuated-(voxel-target_lower)))
        upper.append(min(evacuated, target_upper))
    total_leaves = len(full_indices)*cut_spec.leaf_count+partial_leaves
    descriptor = {"version": EVACUATED_VOLUME_VERSION, "cut_union_spec": cut_spec.to_record(),
                  "evacuated_volume_spec": volume_spec.to_record(), "affine_hash": array_digest(matrix),
                  "whole_removed_mask_hash": array_digest(removed),
                  "partial_coverage_digest": _digest_ledger_items(cut_spec, items)}
    return {"state_digest": semantic_digest(descriptor),
            "whole_parent_clearance_count": len(full_indices),
            "partial_certified_leaf_count": partial_leaves, "certified_leaf_count": total_leaves,
            "evacuated_source_volume_mm3": total_leaves*(voxel/cut_spec.leaf_count),
            "evacuated_target_source_bounds_mm3": [math.fsum(lower), math.fsum(upper)]}


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
        # A content key also detects out-of-band edits to the public working
        # dictionary. Never trust a dirty flag or mutable-object identity here.
        self._cached_digest_key = self._cached_digest_value = None
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
        items = _validated_ledger_items(self.spec, self.cells)
        # Spec types are included so bool/int equality cannot hide a changed
        # declaration. All cell/index/mask types were checked above, on hits too.
        spec_key = tuple((key, type(value), value) for key, value in self.spec.to_record().items())
        key = (spec_key, items)
        if key != self._cached_digest_key:
            self._cached_digest_value = _digest_ledger_items(self.spec, items)
            self._cached_digest_key = key
        return self._cached_digest_value

    def cover(self, occupied, direct_whole_cells, connected_free, start, end, radius,
              *, terminal_microstep=False):
        """Certify only NEW leaves in this one capsule, after its shaft passed.

        direct_whole_cells is the existing contained-and-connected whole-cell
        set, so the fast path preserves that same-step connected-component rule.
        Legacy partial cells must be face-exposed before this interval's removals.
        The explicit parent-closure mode also admits face exposure through fully
        cleared parents from this same capsule, never through partial channels.
        An optional terminal schedule uses that closure only when the native
        caller identifies the actual final microstep of the prescribed stroke.
        """
        start, end = np.asarray(start, float), np.asarray(end, float)
        if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all()
                or not np.isfinite(end).all() or not math.isfinite(radius) or radius <= MARGIN_MM):
            raise ValueError("Finite positive executed active capsule required")
        if type(terminal_microstep) is not bool:
            raise ValueError("Native terminal microstep context must be an explicit bool")
        if (self.spec.exposure_mode == PARENT_CLOSURE_EXPOSURE
                and (self.spec.closure_schedule is None or terminal_microstep)):
            return self._cover_parent_closure(occupied, direct_whole_cells, connected_free, start, end, radius)
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

    def _cover_parent_closure(self, occupied, direct_whole_cells, connected_free, start, end, radius):
        """One capsule's whole-parent exposure fixed point, never a leaf channel.

        Existing contained-and-connected full parents may expose adjacent partial
        parents in this same executed interval. A partial parent can propagate
        exposure only if all its leaves become certified. The native caller still
        independently applies the original whole-parent removal BFS and checks
        the shaft against the prior cavity. No newly opened empty pocket is
        flooded here, and no second capsule or future endpoint enters the proof.
        """
        before, before_leaves, before_tests = self.digest, self.new_leaf_count, self.leaf_tests
        parents = {tuple(int(v) for v in value) for value in occupied}
        direct = {tuple(int(v) for v in value) for value in direct_whole_cells}
        if not direct <= parents:
            raise ValueError("Direct whole-cell proof outside occupied candidate set")
        records, completed, processed = [], set(), set()

        def add(cell, bits, proof):
            if not bits: return
            if cell not in self.cells and len(self.cells) >= self.spec.max_tracked_cells:
                raise CutUnionLimit("Tracked cell ceiling")
            if self.witness_rows >= MAX_WITNESS_ROWS_PER_PREVIEW:
                raise CutUnionLimit("Sparse witness row ceiling")
            self.cells[cell] = self.cells.get(cell, 0) | bits
            self.new_leaf_count += bits.bit_count(); self.witness_rows += 1
            self.peak_tracked_cells = max(self.peak_tracked_cells, len(self.cells))
            records.append({"index": list(cell), "added_bits_hex": format(bits, "x"), "proof": proof})

        # This set already has the canonical whole-parent connected-component
        # certificate. Install its union before testing any adjacent partial cell.
        for cell in sorted(direct):
            add(cell, self.spec.full_mask ^ self.cells.get(cell, 0), "whole_cell_capsule")
            completed.add(cell); processed.add(cell)

        def neighbors(cell):
            for d in range(3):
                for sign in (-1, 1):
                    value = list(cell); value[d] += sign
                    yield tuple(value)

        queue = sorted(cell for cell in parents-direct
                       if _exposed(cell, connected_free) or any(n in completed for n in neighbors(cell)))
        queued = set(queue); position = 0
        while position < len(queue):
            cell = queue[position]; position += 1
            if cell in processed: continue
            processed.add(cell)
            old = self.cells.get(cell, 0)
            pending = [i for i in range(self.spec.leaf_count) if not (old >> i) & 1]
            if self.leaf_tests+len(pending) > self.spec.max_leaf_tests_per_preview:
                raise CutUnionLimit("Leaf proof work ceiling")
            self.leaf_tests += len(pending)
            centers = (self._leaf_centers[pending]+np.asarray(cell)) @ self.affine[:3, :3].T+self.affine[:3, 3]
            corners = centers[:, None, :]+self._corners
            contained = np.all(point_segment_distances(corners, start, end) <= radius-MARGIN_MM, axis=1)
            bits = sum(1 << pending[i] for i in np.flatnonzero(contained))
            add(cell, bits, "leaf_corners")
            if self.cells.get(cell, 0) == self.spec.full_mask:
                completed.add(cell)
                for near in neighbors(cell):
                    if near in parents and near not in queued and near not in processed:
                        queue.append(near); queued.add(near)
        return {"before_digest": before, "new_leaf_count": self.new_leaf_count-before_leaves,
                "leaf_tests": self.leaf_tests-before_tests, "cells": records}

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
