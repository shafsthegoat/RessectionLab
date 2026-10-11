"""Independent replay of opt-in idealized cutting coverage witnesses.

This verifier uses recorded *executed* aspiration capsules and its own geometry
arithmetic. It does not import the producing coverage implementation. A covered
subcell is a geometric assumption about cutting/evacuation, not measured tissue
response. Entire native cells remain occupied until every subcell is certified.
"""
from __future__ import annotations

from itertools import product
import re

import numpy as np

from .core import semantic_digest


MODE = "cumulative_affine_subcell_cut_union_v1"
_HEX = re.compile(r"[1-9a-f][0-9a-f]*\Z")


class IndependentCutCoverage:
    """Bounded witness checker; source connectivity is supplied by the auditor."""

    def __init__(self, spec, matrix, shape):
        if not isinstance(spec, dict) or set(spec) != {
            "mode", "max_depth", "max_tracked_cells", "max_leaf_tests_per_preview"
        } or spec["mode"] != MODE:
            raise ValueError("Unsupported independent cut-union specification")
        for name, cap in (("max_depth", 3), ("max_tracked_cells", 4096),
                          ("max_leaf_tests_per_preview", 4194304)):
            if type(spec[name]) is not int or not 1 <= spec[name] <= cap:
                raise ValueError("Invalid independent cut-union resource bound")
        self.spec = dict(spec)
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

    def digest(self):
        return semantic_digest({"mode": MODE, "spec": self.spec, "cells": [
            {"index": list(key), "covered_bits_hex": format(value, "x")}
            for key, value in sorted(self.ledger.items())]})

    def start(self, record):
        if not isinstance(record, dict) or record.get("spec") != self.spec:
            raise ValueError("Cut-union history differs from declared specification")
        if record.get("before_digest") != self.digest():
            raise ValueError("Cut-union action starts from a different coverage state")
        self.preview_tests = self.preview_new = self.preview_rows = 0

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

    def _exposed(self, key, free):
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(key)
                neighbor[axis] += sign
                if any(x < 0 or x >= size for x, size in zip(neighbor, self.shape)):
                    return True
                if free[tuple(neighbor)]:
                    return True
        return False

    def step(self, record, start, end, radius, mode, remaining, free, removed):
        """Verify additions, then removal completeness; caller checks shaft first.

        This private replay state can be discarded on any failed audit. No producer
        masks, candidate previews, contact flags, or future capsules add coverage.
        """
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
                if not self._exposed(key, free):
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

    def finish(self, record, removed_count, mode):
        expected_progress = removed_count == 0 and self.preview_new > 0 and mode == "aspirate"
        if (record.get("after_digest") != self.digest()
                or record.get("new_leaf_count") != self.preview_new
                or record.get("leaf_tests") != self.preview_tests
                or type(record.get("coverage_only_progress")) is not bool
                or record["coverage_only_progress"] != expected_progress):
            raise ValueError("Cut-union action coverage accounting mismatch")
        if mode == "aspirate" and not removed_count and not self.preview_new:
            raise ValueError("Empty aspiration cannot be committed as cutting progress")
