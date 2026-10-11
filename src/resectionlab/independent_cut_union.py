"""Independent replay of opt-in idealized cutting coverage witnesses.

This verifier uses recorded *executed* aspiration capsules and its own geometry
arithmetic. It does not import the producing coverage implementation. A covered
subcell is a geometric assumption about cutting/evacuation, not measured tissue
response. Entire native cells remain occupied until every subcell is certified.
"""
from __future__ import annotations

from itertools import product
import math
import re

import numpy as np

from .core import array_digest, semantic_digest


MODE = "cumulative_affine_subcell_cut_union_v1"
PARENT_CLOSURE_MODE = "cumulative_affine_parent_closure_cut_union_v3"
PARENT_CLOSURE_EXPOSURE = "same_step_parent_closure"
PARENT_CLOSURE_EVACUATION_FOOTPRINT = "cumulative_affine_parent_closure_evacuation_v3"
TERMINAL_PARENT_CLOSURE_MODE = "cumulative_affine_terminal_parent_closure_cut_union_v4"
TERMINAL_PARENT_CLOSURE_SCHEDULE = "terminal_microstep_only"
TERMINAL_PARENT_CLOSURE_EVACUATION_FOOTPRINT = "cumulative_affine_terminal_parent_closure_evacuation_v4"
EVACUATION_VERSION = "certified_evacuated_subcell_volume_v1"
EVACUATION_FOOTPRINT = "cumulative_affine_subcell_evacuation_v2"
EVACUATION_SCOPE = ("certified_subset_of_represented_source_support_under_idealized_evacuation;"
                    "whole_parent_shaft_envelope_and_reward_unchanged")
_HEX = re.compile(r"[1-9a-f][0-9a-f]*\Z")


class IndependentEvacuatedVolume:
    """Reconstruct volume from audited bits, never from producer totals.

    Target mass is unlocated within each source parent. Intersections therefore
    yield cumulative allocation bounds, not a target label for each leaf. The
    arrays are an explicit caller declaration; no evaluator anatomy is consulted.
    """

    def __init__(self, coverage, spec, lower, upper, tissue):
        fields = {"version", "shape", "target_fraction_lower_hash",
                  "target_fraction_upper_hash", "provenance", "target_semantics"}
        if (not isinstance(spec, dict) or set(spec) != fields
                or spec["version"] != EVACUATION_VERSION
                or spec["target_semantics"] != "unlocated_parent_target_mass_bounds"
                or not isinstance(spec["provenance"], str) or not spec["provenance"].strip()
                or not isinstance(spec["shape"], list)
                or any(type(x) is not int for x in spec["shape"])
                or tuple(spec["shape"]) != coverage.shape):
            raise ValueError("Explicit source-bound evacuation specification required")
        tissue = np.asarray(tissue)
        if tissue.dtype != np.dtype(bool) or tissue.shape != coverage.shape:
            raise ValueError("Evacuation needs the declared original tissue support")
        arrays = []
        for value, name in ((lower, "lower"), (upper, "upper")):
            raw = np.asarray(value)
            if (raw.shape != coverage.shape or raw.dtype.kind not in "biuf"
                    or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1)):
                raise ValueError("Explicit finite source target fraction bounds required")
            array = np.array(raw, dtype=np.float64, copy=True)
            if (np.any(array[~tissue] != 0)
                    or array_digest(array) != spec[f"target_fraction_{name}_hash"]):
                raise ValueError("Target bound hashes or original support mismatch")
            array.flags.writeable = False
            arrays.append(array)
        if np.any(arrays[0] > arrays[1]):
            raise ValueError("Target fraction bounds are not ordered")
        matrix = coverage.matrix
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Evacuation needs a finite source affine")
        # Independent signed determinant expansion; rotations/reflections do not
        # change physical volume. This does not import the producing metric.
        a, b, c = matrix[:3, :3]
        self.parent_volume = abs(float(a[0]*(b[1]*c[2]-b[2]*c[1])
            - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])))
        if not math.isfinite(self.parent_volume) or self.parent_volume <= 0:
            raise ValueError("Evacuation needs positive source-cell volume")
        self.coverage = coverage
        self.spec = {**spec, "shape": list(spec["shape"])}
        self.lower, self.upper = arrays
        self.original = np.array(tissue, dtype=bool, copy=True)
        self.original.flags.writeable = False
        self.removed = np.zeros(coverage.shape, dtype=bool)
        self.certified_count = 0
        self.maximum_count = int(np.count_nonzero(tissue))*coverage.leaf_count
        self.before = None

    def note_step(self, removed, added):
        for key in removed:
            if not self.original[key] or self.removed[key]:
                raise ValueError("Evacuation repeats a cleared source parent")
        for key in removed:
            self.removed[key] = True
        self.certified_count += added
        if not 0 <= self.certified_count <= self.maximum_count:
            raise ValueError("Evacuation exceeds represented tissue volume")

    def snapshot(self):
        c = self.coverage
        whole = np.argwhere(self.removed)
        lower, upper, partial = [], [], 0
        for row in whole:
            key = tuple(row)
            lower.append(float(self.lower[key])*self.parent_volume)
            upper.append(float(self.upper[key])*self.parent_volume)
        for key, bits in sorted(c.ledger.items()):
            if not self.original[key] or self.removed[key] or not 0 < bits < c.full:
                raise ValueError("Partial evacuation overlaps cleared or absent tissue")
            count = bits.bit_count()
            partial += count
            volume = count*(self.parent_volume/c.leaf_count)
            # Sharp bounds on |evacuated subset intersect target| with only the
            # parent target mass known. Recomputed cumulatively, never added as
            # independent action intervals or assumed uniformly mixed target.
            lower.append(max(0., volume-(self.parent_volume-float(self.lower[key])*self.parent_volume)))
            upper.append(min(volume, float(self.upper[key])*self.parent_volume))
        total = len(whole)*c.leaf_count+partial
        if total != self.certified_count:
            raise ValueError("Evacuated leaf union disagrees with executed additions")
        state = {"version": EVACUATION_VERSION, "cut_union_spec": c.spec,
                 "evacuated_volume_spec": self.spec, "affine_hash": array_digest(c.matrix),
                 "whole_removed_mask_hash": array_digest(self.removed),
                 "partial_coverage_digest": c.digest()}
        return {"state_digest": semantic_digest(state),
                "whole_parent_clearance_count": len(whole),
                "partial_certified_leaf_count": partial, "certified_leaf_count": total,
                "evacuated_source_volume_mm3": total*(self.parent_volume/c.leaf_count),
                "evacuated_target_source_bounds_mm3": [math.fsum(lower), math.fsum(upper)]}

    @staticmethod
    def _number(actual, expected):
        return (type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9))

    def _check_snapshot(self, actual, expected):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Malformed evacuated volume snapshot")
        for key in ("whole_parent_clearance_count", "partial_certified_leaf_count", "certified_leaf_count"):
            if type(actual[key]) is not int or actual[key] != expected[key]:
                raise ValueError("Evacuated leaf count mismatch")
        bounds = actual["evacuated_target_source_bounds_mm3"]
        if (actual["state_digest"] != expected["state_digest"]
                or not self._number(actual["evacuated_source_volume_mm3"], expected["evacuated_source_volume_mm3"])
                or actual["evacuated_source_volume_mm3"] < 0
                or not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(not self._number(x, y) for x, y in zip(bounds, expected["evacuated_target_source_bounds_mm3"]))
                or bounds[0] < 0 or bounds[1] < bounds[0]):
            raise ValueError("Evacuated cumulative volume or target bounds mismatch")

    def _record(self, record):
        if (not isinstance(record, dict) or set(record) != {"spec", "before", "after",
                "delta_certified_leaf_count", "delta_evacuated_source_volume_mm3", "scope"}
                or record["spec"] != self.spec or record["scope"] != EVACUATION_SCOPE
                or semantic_digest(record["spec"]) != semantic_digest(self.spec)):
            raise ValueError("Missing or changed declared evacuation accounting")

    def start(self, record):
        self._record(record)
        self.before = self.snapshot()
        self._check_snapshot(record["before"], self.before)

    def finish(self, record):
        self._record(record)
        if self.before is None:
            raise ValueError("Evacuation action has no checked starting state")
        self._check_snapshot(record["before"], self.before)
        after = self.snapshot()
        self._check_snapshot(record["after"], after)
        delta = after["certified_leaf_count"]-self.before["certified_leaf_count"]
        if (type(record["delta_certified_leaf_count"]) is not int
                or record["delta_certified_leaf_count"] != delta
                or delta != self.coverage.preview_new
                or not self._number(record["delta_evacuated_source_volume_mm3"],
                                    delta*(self.parent_volume/self.coverage.leaf_count))
                or record["delta_evacuated_source_volume_mm3"] < 0):
            raise ValueError("Evacuated action delta double counts or invents volume")
        self.before = None

    def geometry(self):
        """Compact in-memory checked union for an external analytic evaluator."""
        snapshot = self.snapshot()
        value = {"version": EVACUATION_VERSION, "source_shape": list(self.coverage.shape),
                 "native_affine": self.coverage.matrix.tolist(),
                 "cut_union_spec": self.coverage.spec, "evacuated_volume_spec": self.spec,
                 "state_digest": snapshot["state_digest"],
                 "whole_removed_indices": np.argwhere(self.removed).tolist(),
                 "partial_cells": [{"index": list(key), "covered_bits_hex": format(bits, "x")}
                     for key, bits in sorted(self.coverage.ledger.items())]}
        return {**value, "geometry_digest": semantic_digest(value)}


class IndependentCutCoverage:
    """Bounded witness checker; source connectivity is supplied by the auditor."""

    def __init__(self, spec, matrix, shape, *, evacuated_volume_spec=None,
                 target_fraction_lower=None, target_fraction_upper=None, tissue_mask=None):
        fields = {"mode", "max_depth", "max_tracked_cells", "max_leaf_tests_per_preview"}
        legacy = isinstance(spec, dict) and set(spec) == fields and spec.get("mode") == MODE
        closure = (isinstance(spec, dict) and set(spec) == fields | {"exposure_mode"}
            and spec.get("mode") == PARENT_CLOSURE_MODE
            and spec.get("exposure_mode") == PARENT_CLOSURE_EXPOSURE)
        terminal = (isinstance(spec, dict) and set(spec) == fields | {"exposure_mode", "closure_schedule"}
            and spec.get("mode") == TERMINAL_PARENT_CLOSURE_MODE
            and spec.get("exposure_mode") == PARENT_CLOSURE_EXPOSURE
            and spec.get("closure_schedule") == TERMINAL_PARENT_CLOSURE_SCHEDULE)
        if not legacy and not closure and not terminal:
            raise ValueError("Unsupported independent cut-union specification")
        for name, cap in (("max_depth", 3), ("max_tracked_cells", 4096),
                          ("max_leaf_tests_per_preview", 4194304)):
            if type(spec[name]) is not int or not 1 <= spec[name] <= cap:
                raise ValueError("Invalid independent cut-union resource bound")
        self.spec = dict(spec)
        self.same_step_parent_closure = closure or terminal
        self.terminal_parent_closure = terminal
        self.matrix = np.array(matrix, dtype=float, copy=True)
        self.shape = tuple(shape)
        self.n = 2 ** spec["max_depth"]
        self.leaf_count = self.n ** 3
        self.full = (1 << self.leaf_count) - 1
        self.ledger = {}
        self.corner_bits = np.array(list(product((0., 1.), repeat=3)))
        self.preview_tests = 0
        self.preview_new = 0
        self.preview_rows = 0
        self.evacuation = None
        if evacuated_volume_spec is not None:
            self.evacuation = IndependentEvacuatedVolume(self, evacuated_volume_spec,
                target_fraction_lower, target_fraction_upper, tissue_mask)
        elif target_fraction_lower is not None or target_fraction_upper is not None:
            raise ValueError("Target fraction arrays require an explicit evacuation declaration")

    def digest(self):
        return semantic_digest({"mode": self.spec["mode"], "spec": self.spec, "cells": [
            {"index": list(key), "covered_bits_hex": format(value, "x")}
            for key, value in sorted(self.ledger.items())]})

    def start(self, record, evacuated_volume_record=None):
        if not isinstance(record, dict) or record.get("spec") != self.spec:
            raise ValueError("Cut-union history differs from declared specification")
        if record.get("before_digest") != self.digest():
            raise ValueError("Cut-union action starts from a different coverage state")
        self.preview_tests = self.preview_new = self.preview_rows = 0
        if self.evacuation is not None:
            self.evacuation.start(evacuated_volume_record)

    def _inside(self, vertices, start, end, radius):
        # Independent projection formula, with an inward tolerance identical in
        # physical units to the declared conservative model, not a fitted epsilon.
        segment = end - start
        denominator = float(segment @ segment)
        if denominator == 0:
            nearest = start
        else:
            weight = np.clip((vertices - start) @ segment / denominator, 0., 1.)
            nearest = start + weight[..., None] * segment
        return np.all(np.sum((vertices - nearest) ** 2, axis=-1)
                      <= max(0., radius - 1e-10) ** 2)

    def _exposed(self, key, free, removed, allow_same_step_removal):
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(key)
                neighbor[axis] += sign
                if any(x < 0 or x >= size for x, size in zip(neighbor, self.shape)):
                    return True
                if free[tuple(neighbor)] or (allow_same_step_removal and tuple(neighbor) in removed):
                    return True
        return False

    def step(self, record, start, end, radius, mode, remaining, free, removed, *, terminal_microstep=False):
        """Verify additions, then removal completeness; caller checks shaft first.

        This private replay state can be discarded on any failed audit. No producer
        masks, candidate previews, contact flags, or future capsules add coverage.
        The optional closure rule also accepts a face adjacent to a same-step
        removed whole parent. The caller first verifies that removed parents
        form components rooted in PRIOR connected free space. All their claimed
        leaf additions and full coverage are checked here before state mutation;
        partial parents, empty pockets and later steps cannot seed exposure.
        For the terminal-only declaration, the auditor supplies terminal status
        from the concrete macro history index, never a claimed witness flag.
        """
        if self.terminal_parent_closure and type(terminal_microstep) is not bool:
            raise ValueError("Terminal closure needs auditor-owned microstep position")
        allow_same_step_removal = (self.same_step_parent_closure
            and (not self.terminal_parent_closure or terminal_microstep))
        if not isinstance(record, dict) or record.get("before_digest") != self.digest():
            raise ValueError("Missing or discontinuous cut-union microstep")
        cells = record.get("cells")
        tests = record.get("leaf_tests")
        count = record.get("new_leaf_count")
        if (not isinstance(cells, list) or len(cells) > self.spec["max_tracked_cells"]
                or type(tests) is not int or tests < 0 or type(count) is not int or count < 0):
            raise ValueError("Invalid cut-union witness accounting")
        self.preview_tests += tests
        self.preview_rows += len(cells)
        if self.preview_rows > 8192:
            raise ValueError("Cut-union action exceeds witness row cap")
        if self.preview_tests > self.spec["max_leaf_tests_per_preview"]:
            raise ValueError("Cut-union action exceeds declared work cap")
        if mode != "aspirate" and (cells or tests or count or removed):
            raise ValueError("Non-aspiration interaction cannot create cutting coverage")
        seen, added_total, individually_checked = set(), 0, 0
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != {"index", "added_bits_hex", "proof"}:
                raise ValueError("Malformed cut-union cell witness")
            index = cell["index"]
            if (not isinstance(index, (list, tuple)) or len(index) != 3
                    or any(type(x) is not int or x < 0 or x >= size
                           for x, size in zip(index, self.shape))):
                raise ValueError("Cut-union witness outside source grid")
            key = tuple(index)
            if key in seen or not remaining[key]:
                raise ValueError("Repeated or already removed coverage cell")
            if self.evacuation is not None and (self.evacuation.removed[key]
                    or not self.evacuation.original[key]):
                raise ValueError("Evacuation witness repeats cleared or absent source tissue")
            seen.add(key)
            encoded = cell["added_bits_hex"]
            if (not isinstance(encoded, str) or len(encoded) > (self.leaf_count + 3) // 4
                    or _HEX.fullmatch(encoded) is None):
                raise ValueError("Cut-union mask must be bounded canonical nonzero hex")
            bits = int(encoded, 16)
            old = self.ledger.get(key, 0)
            if bits > self.full or bits & old:
                raise ValueError("Coverage witness repeats or invents subcells")
            base = np.asarray(key, dtype=float) - .5
            proof = cell["proof"]
            if proof == "whole_cell_capsule":
                if key not in removed:
                    raise ValueError("Whole-cell proof must remove a connected cell now")
                vertices = (base + self.corner_bits) @ self.matrix[:3, :3].T + self.matrix[:3, 3]
                if not self._inside(vertices, start, end, radius) or bits != self.full ^ old:
                    raise ValueError("Whole-cell capsule witness is unsupported")
            elif proof == "leaf_corners":
                if not self._exposed(key, free, removed, allow_same_step_removal):
                    raise ValueError("Partial cutting coverage was banked in buried tissue")
                leaf_ids = [i for i in range(self.leaf_count) if (bits >> i) & 1]
                individually_checked += len(leaf_ids)
                # A leaf must lie in one capsule. Checking different original
                # corners against different capsules would permit an interior gap.
                for offset in range(0, len(leaf_ids), 64):
                    ids = np.asarray(leaf_ids[offset:offset+64])
                    coordinates = np.stack((ids // self.n ** 2,
                        (ids // self.n) % self.n, ids % self.n), axis=1)
                    vertices = base + (coordinates[:, None, :] + self.corner_bits) / self.n
                    vertices = vertices @ self.matrix[:3, :3].T + self.matrix[:3, 3]
                    if not self._inside(vertices, start, end, radius):
                        raise ValueError("Subcell capsule witness is unsupported")
            else:
                raise ValueError("Unknown cutting coverage proof")
            if key not in self.ledger and len(self.ledger) >= self.spec["max_tracked_cells"]:
                raise ValueError("Cut-union coverage exceeds declared resident cell cap")
            self.ledger[key] = old | bits
            added_total += bits.bit_count()
        if len(self.ledger) > self.spec["max_tracked_cells"]:
            raise ValueError("Cut-union coverage exceeds declared resident cell cap")
        # Failed producer containment tests leave no witness. Their cost counter
        # is bounded and lower-bound checked, not independently reconstructed.
        if count != added_total or tests < individually_checked:
            raise ValueError("Cut-union leaf accounting is unsupported")
        if any(mask == self.full and key not in removed for key, mask in self.ledger.items()):
            raise ValueError("Completed exposed cutting coverage must be removed immediately")
        for key in removed:
            if self.ledger.get(key, 0) != self.full:
                raise ValueError("Whole-cell removal has incomplete cutting coverage")
        for key in removed:
            del self.ledger[key]
        self.preview_new += count
        if record.get("after_digest") != self.digest():
            raise ValueError("Cut-union microstep coverage digest mismatch")
        if self.evacuation is not None:
            self.evacuation.note_step(removed, added_total)

    def finish(self, record, removed_count, mode, evacuated_volume_record=None):
        if self.evacuation is not None and (type(record.get("new_leaf_count")) is not int
                or type(record.get("leaf_tests")) is not int):
            raise ValueError("Evacuation needs integer action coverage counters")
        expected_progress = removed_count == 0 and self.preview_new > 0 and mode == "aspirate"
        if (record.get("after_digest") != self.digest()
                or record.get("new_leaf_count") != self.preview_new
                or record.get("leaf_tests") != self.preview_tests
                or type(record.get("coverage_only_progress")) is not bool
                or record["coverage_only_progress"] != expected_progress):
            raise ValueError("Cut-union action coverage accounting mismatch")
        if mode == "aspirate" and not removed_count and not self.preview_new:
            raise ValueError("Empty aspiration cannot be committed as cutting progress")
        if self.evacuation is not None:
            self.evacuation.finish(evacuated_volume_record)
