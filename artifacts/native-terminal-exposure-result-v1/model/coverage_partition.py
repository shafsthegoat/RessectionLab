"""Read-only shadow of executed cutting in public nominal-positive parents.

No eligibility or native state is modified. Capsule containment without exposure
is a diagnostic counterfactual, never an executable removal certificate.
"""
from itertools import product
import time
import numpy as np
from resectionlab.geometry import point_segment_distances
from resectionlab.native_cut_union import MARGIN_MM


def combine(destination, source):
    for cell, bits in source.items(): destination[cell] = destination.get(cell, 0) | bits


def subtract(left, right):
    return {c: v & ~right.get(c, 0) for c, v in left.items() if v & ~right.get(c, 0)}


def subset(left, right):
    return not subtract(left, right)


def rows(value):
    return [{'index': list(c), 'covered_bits_hex': format(v, 'x')} for c, v in sorted(value.items())]


class CoveragePartition:
    def __init__(self, *, deadline, max_leaf_tests=4000000):
        self.deadline, self.maximum = deadline, max_leaf_tests
        self.leaf_tests = self.cover_calls = self.target_cover_calls = 0
        self.shadow_seconds = 0.
        self.geometry, self.exposed, self.whole_sweep = {}, {}, {}
        self.actions = []; self.active = None; self.config = None

    def guard(self):
        if time.perf_counter() >= self.deadline: raise TimeoutError('Diagnostic cooperative deadline')

    def begin(self, config):
        if self.active is not None: raise ValueError('Nested native action')
        self.config = config
        self.active = {'geometry': {}, 'exposed': {}, 'parents': set(), 'first_start': None,
                       'last_end': None, 'radius': None, 'microsteps': [], 'tests_before': self.leaf_tests}

    def contained(self, coverage, cell, start, end, radius):
        self.guard(); n = coverage.spec.leaf_count
        if self.leaf_tests+n > self.maximum: raise TimeoutError('Shadow leaf-distance ceiling')
        self.leaf_tests += n
        centers = (coverage._leaf_centers + np.asarray(cell)) @ coverage.affine[:3, :3].T + coverage.affine[:3, 3]
        corners = centers[:, None, :] + coverage._corners
        inside = np.all(point_segment_distances(corners, start, end) <= radius-MARGIN_MM, axis=1)
        return sum(1 << int(i) for i in np.flatnonzero(inside))

    @staticmethod
    def parent_exposed(cell, free):
        for d in range(3):
            for sign in (-1, 1):
                near = list(cell); near[d] += sign
                if near[d] < 0 or near[d] >= free.shape[d] or free[tuple(near)]: return True
        return False

    def cover(self, original, coverage, occupied, direct, free, start, end, radius):
        if self.active is None: raise ValueError('Unscoped native coverage call')
        stamp = time.perf_counter(); self.cover_calls += 1; a = self.active
        if a['first_start'] is None: a['first_start'] = np.asarray(start).copy()
        a['last_end'] = np.asarray(end).copy(); a['radius'] = float(radius); a['coverage'] = coverage
        direct_set = {tuple(int(x) for x in c) for c in direct}
        expected = {}; geometry = {}; eligible = {}; blocked = {}
        for raw in occupied:
            cell = tuple(int(x) for x in raw)
            # Existing generated public nominal target membership, never the
            # evaluator plane or subcell target. This limits attribution scope.
            if self.config.target_labels[cell] <= 0: continue
            a['parents'].add(cell)
            bits = self.contained(coverage, cell, start, end, radius)
            if bits: geometry[cell] = bits
            if cell in direct_set or self.parent_exposed(cell, free):
                if bits: eligible[cell] = bits
                new = bits & ~coverage.cells.get(cell, 0)
                if new: expected[cell] = new
            elif bits: blocked[cell] = bits
        self.shadow_seconds += time.perf_counter()-stamp
        record = original(coverage, occupied, direct, free, start, end, radius)
        stamp = time.perf_counter()
        actual = {tuple(r['index']): int(r['added_bits_hex'], 16) for r in record['cells']
                  if self.config.target_labels[tuple(r['index'])] > 0}
        if actual != expected: raise AssertionError('Shadow containment/exposure differs from actual new bits')
        combine(a['geometry'], geometry); combine(a['exposed'], eligible)
        if geometry:
            self.target_cover_calls += 1
            a['microsteps'].append({'active_start_mm': np.asarray(start).tolist(), 'active_end_mm': np.asarray(end).tolist(),
                'geometrically_contained_leaves': sum(v.bit_count() for v in geometry.values()),
                'prior_parent_exposure_eligible_leaves': sum(v.bit_count() for v in eligible.values()),
                'newly_certified_leaves': sum(v.bit_count() for v in actual.values()),
                'geometry_but_parent_unexposed': rows(blocked)})
        self.shadow_seconds += time.perf_counter()-stamp
        return record

    def finish(self, result):
        a = self.active
        if a is None: raise ValueError('No active native action')
        stamp = time.perf_counter(); whole = {}
        if result.feasible:
            combine(self.geometry, a['geometry']); combine(self.exposed, a['exposed'])
            for cell in sorted(a['parents']):
                bits = self.contained(a['coverage'], cell, a['first_start'], a['last_end'], a['radius'])
                if bits: whole[cell] = bits
            combine(self.whole_sweep, whole)
        self.actions.append({'action_index': len(self.actions)+1, 'accepted': bool(result.feasible),
            'reason': result.reason, 'diagnostic_leaf_tests': self.leaf_tests-a['tests_before'],
            'shadow_sets_committed_for_diagnosis': bool(result.feasible), 'microsteps': a['microsteps']})
        self.active = None; self.shadow_seconds += time.perf_counter()-stamp

    def final(self, engine):
        if self.active is not None: raise ValueError('Unfinished native action')
        certified = {tuple(int(x) for x in c): engine.config.cut_union.full_mask for c in np.argwhere(engine.removed_mask)
                     if engine.config.target_labels[tuple(c)] > 0}
        for cell, bits in engine._cut_union_ledger.items():
            if engine.config.target_labels[cell] > 0: certified[cell] = bits
        if certified != self.exposed: raise AssertionError('Exposed geometric leaf union differs from actual certification')
        if not subset(certified, self.geometry) or not subset(self.geometry, self.whole_sweep):
            raise AssertionError('Expected containment nesting failed; no numerical normalization permitted')
        return {'certified': certified, 'executed_capsule_geometry': self.geometry,
            'whole_straight_sweep_geometry': self.whole_sweep,
            'parent_exposure_deficit': subtract(self.geometry, certified),
            'microstep_capsule_deficit': subtract(self.whole_sweep, self.geometry)}
