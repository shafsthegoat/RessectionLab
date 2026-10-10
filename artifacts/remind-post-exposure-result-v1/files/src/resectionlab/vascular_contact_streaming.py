"""Bounded tiled full-grid contact counts using the independent geometry oracle.

This geometry primitive grants no patient/source/strategy admission. Its caller
must own and authenticate the reference sampler and supply complete checked tool
capsules. Existing evaluation admission, sealing, frame and lineage gates remain
outside this module. A sampler is called synchronously with at most one bounded
tile's contacted voxel indices; arbitrary callback time/memory needs external
supervision. No image reader, planner, model or removal calculation is supplied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import time

import numpy as np

from resectionlab.independent_geometry_batch import segment_box_contact_indices

VERSION = 'tiled-vascular-contact-v1'
CONTACT_TOLERANCE_SQ_MM2 = 1e-10
MAX_CAPSULES = 128
MAX_TILE_EDGE = 16
MAX_COARSE_BATCH = 256


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Grid:
    shape: tuple
    affine_ras_mm: tuple

    def __post_init__(self):
        shape = tuple(self.shape)
        require(len(shape) == 3 and all(type(n) is int and 1 <= n <= 1024 for n in shape)
                and math.prod(shape) <= 512**3, 'grid_dimensions')
        a = np.asarray(self.affine_ras_mm, dtype=np.float64)
        require(a.shape == (4, 4) and np.isfinite(a).all()
                and np.array_equal(a[3], [0, 0, 0, 1]), 'grid_affine')
        spacing = np.linalg.norm(a[:3, :3], axis=0)
        require(np.all((spacing >= .01) & (spacing <= 100))
                and np.allclose((a[:3, :3] / spacing).T @ (a[:3, :3] / spacing),
                                np.eye(3), atol=1e-9, rtol=0), 'orthogonal_reference_grid_required')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'affine_ras_mm', tuple(tuple(float(v) for v in row) for row in a))

    @property
    def fingerprint(self):
        return identity(asdict(self))


@dataclass(frozen=True)
class Capsule:
    action_id: str
    part: str
    start_ras_mm: tuple
    end_ras_mm: tuple
    radius_mm: float

    def __post_init__(self):
        require(isinstance(self.action_id, str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', self.action_id), 'action_id')
        require(self.part in ('shaft', 'tip'), 'tool_part')
        for name in ('start_ras_mm', 'end_ras_mm'):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            require(value.shape == (3,) and np.isfinite(value).all(), 'capsule_coordinates')
            object.__setattr__(self, name, tuple(float(v) for v in value))
        require(type(self.radius_mm) in (int, float) and math.isfinite(self.radius_mm)
                and 0 <= self.radius_mm, 'capsule_radius')
        object.__setattr__(self, 'radius_mm', float(self.radius_mm))


@dataclass(frozen=True)
class Budget:
    tile_edge: int = 16
    coarse_batch: int = 256
    max_tile_capsule_pairs: int = 2_000_000
    max_cell_capsule_pairs: int = 4_000_000
    max_sampled_cells: int = 2_000_000
    wall_seconds: float = 20.

    def __post_init__(self):
        require(type(self.tile_edge) is int and 1 <= self.tile_edge <= MAX_TILE_EDGE, 'tile_edge')
        require(type(self.coarse_batch) is int and 1 <= self.coarse_batch <= MAX_COARSE_BATCH, 'coarse_batch')
        for name in ('max_tile_capsule_pairs', 'max_cell_capsule_pairs', 'max_sampled_cells'):
            require(type(getattr(self, name)) is int and 0 <= getattr(self, name) <= 8_000_000, 'work_budget')
        require(type(self.wall_seconds) in (int, float) and math.isfinite(self.wall_seconds)
                and 0 < self.wall_seconds <= 30, 'wall_budget')


class BudgetExceeded(RuntimeError):
    def __init__(self, reason, work, budget, rejected_charge=None):
        super().__init__(reason)
        self.reason = reason
        self.work = dict(work)
        self.budget = asdict(budget)
        self.rejected_charge = rejected_charge
        self.completed = False
        self.outcomes = None


def _indices(start, stop):
    """Only one bounded tile; never materialize the complete grid/bounding box."""
    local = np.indices(tuple(stop - start), dtype=np.int64).reshape(3, -1).T
    return local + start


def _count_record(counts, outside, volume):
    touched, positive, unknown = map(int, counts)
    return {'touched_reference_cells': touched, 'positive_reference_cells': positive,
            'unknown_reference_cells': unknown, 'outside_reference_fov': bool(outside),
            'annotated_positive_encounter': True if positive else None if unknown or outside else False,
            'annotation_coverage_complete_for_sweep': not unknown and not outside,
            'positive_cell_volume_upper_bound_mm3': positive*volume,
            'unknown_in_grid_cell_volume_mm3': unknown*volume,
            'biological_vessel_free': None, 'clinical_injury_probability': None}


def evaluate_contacts(grid, capsules, *, sample_reference, budget=Budget(), cancelled=None):
    """Stream disjoint tiles; union whole-strategy, parts and each action locally.

    The caller supplies authenticated coverage/positive labels and complete checked
    capsules; this primitive proves no admission or history completeness. Budget
    failures expose work counters without partial outcomes. Callback and wall
    limits are cooperative; an external process supervisor remains necessary.
    """
    require(type(grid) is Grid and type(budget) is Budget and callable(sample_reference), 'typed_grid_budget_sampler')
    require(type(capsules) is tuple and len(capsules) <= MAX_CAPSULES
            and all(type(c) is Capsule for c in capsules), 'bounded_capsules')
    require(cancelled is None or callable(cancelled), 'cancelled_callback')
    started = time.monotonic()
    work = {'tile_capsule_pairs': 0, 'cell_capsule_pairs': 0, 'tiles_scanned': 0,
            'tiles_pruned': 0, 'tiles_evaluated': 0, 'reference_sample_calls': 0,
            'reference_sampled_cells': 0, 'maximum_tile_cells': 0,
            'maximum_geometry_batch': 0, 'maximum_coarse_geometry_batch': 0,
            'maximum_cell_geometry_batch': 0, 'maximum_action_masks': 0}

    def check():
        if cancelled is not None and cancelled():
            raise BudgetExceeded('cancelled', work, budget)
        if time.monotonic() - started >= budget.wall_seconds:
            raise BudgetExceeded('cooperative_wall_budget', work, budget)
        return False

    def charge(name, amount, limit):
        check()
        if work[name] + amount > limit:
            raise BudgetExceeded(name, work, budget, {'counter': name, 'current': work[name],
                'requested': amount, 'projected': work[name] + amount, 'limit': limit})
        work[name] += amount

    affine = np.asarray(grid.affine_ras_mm)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    rotation = affine[:3, :3] / spacing
    local = [(rotation.T @ (np.asarray(c.start_ras_mm) - affine[:3, 3]),
              rotation.T @ (np.asarray(c.end_ras_mm) - affine[:3, 3]), c.radius_mm) for c in capsules]
    outside = [bool(np.any(np.minimum(a, b) - radius < -.5*spacing)
                    or np.any(np.maximum(a, b) + radius > (np.asarray(grid.shape) - .5)*spacing))
               for a, b, radius in local]
    actions = tuple(dict.fromkeys(c.action_id for c in capsules))
    counts = {key: np.zeros(3, dtype=np.int64) for key in ('shaft', 'tip', 'whole')}
    action_counts = {key: np.zeros(3, dtype=np.int64) for key in actions}
    tiles_shape = tuple((n + budget.tile_edge - 1) // budget.tile_edge for n in grid.shape)
    total_tiles = math.prod(tiles_shape)
    # Broad-phase padding only makes admission more inclusive. Cell contact
    # always uses exactly the established 1e-10 mm² tolerance, without padding.
    coordinate_scale = max(1., float(np.max(np.asarray(grid.shape)*spacing)),
                           *(float(np.max(np.abs(v))) for row in local for v in row[:2]))
    pruning_padding_mm = 1e-9*coordinate_scale
    for offset in range(0, total_tiles if capsules else 0, budget.coarse_batch):
        check()
        flat = np.arange(offset, min(offset + budget.coarse_batch, total_tiles), dtype=np.int64)
        tile_ids = np.column_stack(np.unravel_index(flat, tiles_shape))
        starts = tile_ids*budget.tile_edge
        stops = np.minimum(starts + budget.tile_edge, grid.shape)
        lower, upper = (starts - .5)*spacing, (stops - .5)*spacing
        admitted = np.zeros((len(flat), len(capsules)), bool)
        for index, (a, b, radius) in enumerate(local):
            charge('tile_capsule_pairs', len(flat), budget.max_tile_capsule_pairs)
            work['maximum_coarse_geometry_batch'] = max(work['maximum_coarse_geometry_batch'], len(flat))
            work['maximum_geometry_batch'] = max(work['maximum_geometry_batch'], len(flat))
            rows = segment_box_contact_indices(a, b, lower - pruning_padding_mm, upper + pruning_padding_mm,
                radius, tolerance_sq=CONTACT_TOLERANCE_SQ_MM2, batch_size=256, cancelled=check)
            admitted[rows, index] = True
        work['tiles_scanned'] += len(flat)
        for tile, (start, stop) in enumerate(zip(starts, stops)):
            check()
            active = np.flatnonzero(admitted[tile])
            if not len(active):
                work['tiles_pruned'] += 1
                continue
            size = math.prod(int(v) for v in stop - start)
            # Admit all work for this tile before its voxel arrays are allocated.
            charge('cell_capsule_pairs', size*len(active), budget.max_cell_capsule_pairs)
            indices = _indices(start, stop)
            centres = indices*spacing
            lo, hi = centres - spacing/2, centres + spacing/2
            parts = {'shaft': np.zeros(size, bool), 'tip': np.zeros(size, bool)}
            action_masks = {}
            work['maximum_tile_cells'] = max(work['maximum_tile_cells'], size)
            work['maximum_geometry_batch'] = max(work['maximum_geometry_batch'], min(size, 256))
            work['maximum_cell_geometry_batch'] = max(work['maximum_cell_geometry_batch'], min(size, 256))
            for index in active:
                check()
                a, b, radius = local[index]
                rows = segment_box_contact_indices(a, b, lo, hi, radius,
                    tolerance_sq=CONTACT_TOLERANCE_SQ_MM2, batch_size=256, cancelled=check)
                c = capsules[index]
                parts[c.part][rows] = True
                if c.action_id not in action_masks:
                    action_masks[c.action_id] = np.zeros(size, bool)
                action_masks[c.action_id][rows] = True
            work['maximum_action_masks'] = max(work['maximum_action_masks'], len(action_masks))
            whole = parts['shaft'] | parts['tip']
            work['tiles_evaluated'] += 1
            if not whole.any():
                continue
            selected = indices[whole]
            charge('reference_sampled_cells', len(selected), budget.max_sampled_cells)
            work['reference_sample_calls'] += 1
            pair = sample_reference(selected)
            check()
            require(type(pair) is tuple and len(pair) == 2
                    and all(type(value) is np.ndarray and value.dtype == np.bool_
                            and value.shape == (len(selected),) for value in pair), 'bounded_boolean_sample')
            positive, known = (np.array(value, copy=True) for value in pair)
            require(not np.any(positive & ~known), 'positive_outside_annotation_domain')
            for name, flags in (*parts.items(), ('whole', whole)):
                chosen = flags[whole]
                counts[name] += (int(chosen.sum()), int((chosen & positive).sum()), int((chosen & ~known).sum()))
            for name, flags in action_masks.items():
                chosen = flags[whole]
                action_counts[name] += (int(chosen.sum()), int((chosen & positive).sum()), int((chosen & ~known).sum()))
        check()
    volume = float(abs(np.linalg.det(affine[:3, :3])))
    result = {'schema': VERSION, 'status': 'complete_tiled_contact_evaluation', 'patient_admission': False,
              'strategy_replay_or_admission_performed': False,
              'reference_grid_identity': grid.fingerprint, 'capsules_identity': identity([asdict(c) for c in capsules]),
              'grid_shape': list(grid.shape), 'grid_voxels': math.prod(grid.shape),
              'capsule_count': len(capsules), 'action_count': len(actions), 'work': dict(work), 'budget': asdict(budget),
              'contact_tolerance_squared_mm2': CONTACT_TOLERANCE_SQ_MM2,
              'tile_pruning_padding_mm': pruning_padding_mm,
              'per_action': {name: _count_record(value, any(outside[i] for i, c in enumerate(capsules) if c.action_id == name), volume)
                             for name, value in action_counts.items()},
              'removed_overlap': {'status': 'not_evaluated_by_contact_kernel', 'outcomes': None,
                                  'existing_congruence_requirement_unchanged': True},
              'reference_scope': 'Supplied annotation domain only; absent labels do not certify absence of biological vessels.',
              'geometry_scope': 'Discrete reference-cell contacts; volume is a cell-volume surrogate, not continuous injury.',
              'budget_scope': 'Cooperative checkpoints and fixed tile arrays; no hard wall/RSS containment.',
              'elapsed_seconds': time.monotonic() - started}
    for name in ('shaft', 'tip', 'whole'):
        result['whole_tool' if name == 'whole' else name] = _count_record(counts[name],
            any(outside[i] for i, c in enumerate(capsules) if name == 'whole' or c.part == name), volume)
    check()
    return result
