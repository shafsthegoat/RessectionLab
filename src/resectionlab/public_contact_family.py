"""Fixed generated sources for the shared public retained-surface-contact task.

Recipe generation and source inspection are allowed for every split. Execution
of MEASUREMENT_EVAL is deliberately unavailable in this preparation module.
TRAIN/SELECT factory access is not permission to train or tune on SELECT.
No patient, estimator, privileged label, model, or new simulator is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations, product
from types import MappingProxyType

import numpy as np

from .core import array_digest, freeze_json, semantic_digest
from .development_episode import TOOLS
from .geometry import AccessWindow
from .native_spatial_task import NativeSpatialCase

FAMILY_VERSION = 'generated-public-contact-family-v1'
SPLIT_VERSION = 'fixed-stratified-geometry-split-v1'
GOAL_IDS = ('surface', 'deep')
ROLES = ('TRAIN', 'SELECT', 'MEASUREMENT_EVAL')
SHAPE = (15, 15, 14)


@dataclass(frozen=True)
class _Recipe:
    layout_id: str
    center_xy: tuple[int, int]
    front_depths: tuple[int, int, int, int, int]
    access_x_offset: float
    access_radius_mm: float
    surface_goal_x_offset: int

    def record(self):
        return {**asdict(self), 'shape': SHAPE, 'affine_ras_mm': np.eye(4).tolist(),
            'occupied_y_offsets': (-2, 2), 'occupied_final_depth': 10,
            'access_z_mm': 1.5, 'normal_inward': (0., 0., 1.),
            'deep_goal_z': 7, 'goal_ids': GOAL_IDS}

    @property
    def digest(self):
        return semantic_digest(self.record())


_RECIPES = tuple(_Recipe(f'pcf-{i:02d}', (5+i % 5, 5+i // 5),
    (2+(i//3) % 3, 2+i//9, 2+i % 3, 2+(i//3) % 3, 2+(i//3+1) % 3),
    .25*((i % 3)-1), (1.5, 2., 2.6)[i % 3], -1 if i % 2 else 1)
    for i in range(24))
_BY_ID = MappingProxyType({r.layout_id: r for r in _RECIPES})


def _split_roles():
    # Prospective hash ordering; no actions, objectives, or outcomes are scored.
    roles = {}
    for center_depth, select_count in ((2, 2), (3, 1), (4, 1)):
        group = sorted((r for r in _RECIPES if r.front_depths[2] == center_depth),
            key=lambda r: semantic_digest({'split': SPLIT_VERSION, 'recipe': r.record()}))
        for index, recipe in enumerate(group):
            roles[recipe.layout_id] = ('TRAIN' if index < 4 else
                'SELECT' if index < 4+select_count else 'MEASUREMENT_EVAL')
    return MappingProxyType(roles)


_ROLE_BY_ID = _split_roles()


def _recipe(layout_id):
    if type(layout_id) is not str or layout_id not in _BY_ID:
        raise ValueError('Unknown fixed generated layout ID')
    return _BY_ID[layout_id]


def _goal_index(recipe, goal_id):
    if type(goal_id) is not str or goal_id not in GOAL_IDS:
        raise ValueError('Choose a prospectively fixed surface or deep public goal')
    x, y = recipe.center_xy
    if goal_id == 'deep':
        return (x, y, 7)
    dx = recipe.surface_goal_x_offset
    return (x+dx, y, recipe.front_depths[dx+2]+1)


def family_record():
    """Split metadata is deliberately separate from actor-side source fields."""
    return {'version': FAMILY_VERSION, 'split_version': SPLIT_VERSION,
        'layouts': [{'recipe': r.record(), 'recipe_hash': r.digest,
            'role': _ROLE_BY_ID[r.layout_id]} for r in _RECIPES],
        'tools': [asdict(t) for t in TOOLS],
        'role_counts': {'TRAIN': 12, 'SELECT': 4, 'MEASUREMENT_EVAL': 8},
        'split_unit': 'whole_layout_including_both_goals_and_all_state_descendants',
        'scope': 'generated_software_task_no_real_patient_inference',
        'task_factory_roles': ['TRAIN', 'SELECT'], 'training_admission': False,
        'held_out_execution_admitted': False}


def family_digest():
    return semantic_digest(family_record())


def layout_metadata(layout_id):
    recipe = _recipe(layout_id)
    return {'layout_id': recipe.layout_id, 'role': _ROLE_BY_ID[layout_id],
        'recipe_hash': recipe.digest, 'family_hash': family_digest(),
        'goals': {g: {'native_index': _goal_index(recipe, g)} for g in GOAL_IDS},
        'task_execution_available': _ROLE_BY_ID[layout_id] in ('TRAIN', 'SELECT'),
        'training_admission': False}


def build_family_source(layout_id):
    """Build public generated arrays only, without engine preview or transition.

    Zero nominal/reference fields are unused compatibility fields, not hidden
    targets. All tissue is visible through the analytic signal/support rule.
    The aperture is hypothetical; it does not carve any initial tissue away.
    """
    recipe = _recipe(layout_id)
    cx, cy = recipe.center_xy
    support = np.zeros(SHAPE, bool)
    for dx, front in zip(range(-2, 3), recipe.front_depths):
        support[cx+dx, cy-2:cy+3, front:11] = True
    image = np.where(support, .2, 0.).astype(np.float32)
    zero = np.zeros(SHAPE, np.float32)
    return NativeSpatialCase(image, support, zero, np.eye(4),
        AccessWindow((cx+recipe.access_x_offset, float(cy), 1.5), (0., 0., 1.),
            recipe.access_radius_mm, 'generated-family-access'), TOOLS,
        track='synthetic_scan', support_source_kind='derived_from_scan',
        support_derivation='analytic generated signal is nonzero exactly on the stepped occupied support',
        nominal_target=zero, target_source_kind='derived_from_scan',
        target_derivation='constant zero compatibility field; unused by public surface-contact objective',
        proposal_mode='fixed_lattice')


def topology_digest(support):
    """Translation/axis-permutation/reflection invariant occupied-grid digest.

    A finite orthogonal-grid duplicate check, not a continuum shape metric.
    No task, legal inventory, geometry preview, or reward is computed.
    """
    support = np.asarray(support)
    if support.ndim != 3 or support.dtype != np.bool_ or not support.any():
        raise ValueError('A nonempty binary 3D support is required')
    occupied = np.argwhere(support)
    crop = support[tuple(slice(int(lo), int(hi)+1)
        for lo, hi in zip(occupied.min(0), occupied.max(0)))]
    return min(array_digest(np.flip(np.transpose(crop, order),
        tuple(axis for axis, flip in enumerate(flips) if flip)))
        for order in permutations(range(3)) for flips in product((False, True), repeat=3))


def family_manifest():
    """Source-only identities; decision-model identities await released execution."""
    from .public_surface_contact import PublicSurfaceGoal, OBJECTIVE_VERSION
    rows = []
    for recipe in _RECIPES:
        case = build_family_source(recipe.layout_id)
        origin = np.asarray(case._crop_origin)
        crop_affine = case._native_affine_ras_mm.copy()
        crop_affine[:3, 3] += crop_affine[:3, :3] @ origin
        goals = {}
        for goal_id in GOAL_IDS:
            index = _goal_index(recipe, goal_id)
            grid = np.zeros(case._crop_shape, bool)
            grid[tuple(np.subtract(index, origin))] = True
            goals[goal_id] = {'native_index': index,
                'objective_hash': PublicSurfaceGoal(case.source_hash, index).fingerprint,
                'goal_grid_hash': array_digest(grid)}
        rows.append({**layout_metadata(recipe.layout_id), 'goals': goals,
            'source_hash': case.source_hash, 'support_hash': array_digest(case.observed_support),
            'topology_hash': topology_digest(case.observed_support),
            'structural_intensity_hash': array_digest(case.structural_intensity),
            'nominal_target_hash': array_digest(case.nominal_target),
            'affine_hash': array_digest(case.affine_ras_mm),
            'crop_origin_native': case._crop_origin, 'crop_shape': case._crop_shape,
            'crop_affine_hash': array_digest(crop_affine),
            'support_cell_count': int(case.observed_support.sum()),
            'candidate_descriptor_count': len(case._candidate_voxels)*len(case.tools),
            'candidate_scope': case._candidate_scope, 'decision_model_hash': None,
            'execution_binding_status': 'not_materialized_source_only',
            'actions_executed': 0, 'geometry_previews': 0})
    return {**family_record(), 'family_hash': family_digest(),
        'objective_version': OBJECTIVE_VERSION, 'source_bindings': rows,
        'held_out_task_executions': 0, 'training_performed': False}


def make_family_task(layout_id, goal_id, *, cancelled=None):
    """Build an existing task for TRAIN/SELECT forward use, never authorize loss.

    This preparation API has no override for held-out execution. A reviewed
    experiment freeze must add a separately bound held-out evaluation path.
    """
    recipe = _recipe(layout_id)
    if _ROLE_BY_ID[layout_id] == 'MEASUREMENT_EVAL':
        raise ValueError('HELD_OUT_EXECUTION_CLOSED: experiment and weights are not frozen')
    point = _goal_index(recipe, goal_id)
    from .public_surface_contact import PublicSurfaceGoal, SurfaceContactTask
    case = build_family_source(layout_id)
    return SurfaceContactTask(case, objective=PublicSurfaceGoal(case.source_hash, point), cancelled=cancelled)


def bind_family_context(task, *, layout_id, goal_id, experiment_hash):
    """Return (existing exact context, immutable split-bound declaration).

    This is forward/plan provenance only. Future learning admission must require
    TRAIN and enumerate these identities; SELECT factory permission is not it.
    """
    from .public_surface_contact import PublicSurfaceGoal, SurfaceContactTask, HASH
    recipe = _recipe(layout_id)
    if _ROLE_BY_ID[layout_id] == 'MEASUREMENT_EVAL':
        raise ValueError('HELD_OUT_EXECUTION_CLOSED: cannot bind a held-out task')
    if not isinstance(experiment_hash, str) or not HASH.fullmatch(experiment_hash):
        raise ValueError('A frozen future experiment declaration digest is required')
    if type(task) is not SurfaceContactTask:
        raise TypeError('Use the exact shared public surface-contact task')
    case = build_family_source(layout_id)
    goal = PublicSurfaceGoal(case.source_hash, _goal_index(recipe, goal_id))
    if task.case.source_hash != case.source_hash or task.objective.fingerprint != goal.fingerprint:
        raise ValueError('Task differs from the canonical layout/goal recipe')
    declaration = freeze_json({'version': FAMILY_VERSION, 'experiment_hash': experiment_hash,
        'layout_id': layout_id, 'goal_id': goal_id, 'role': _ROLE_BY_ID[layout_id],
        'family_hash': family_digest(), 'recipe_hash': recipe.digest,
        'source_hash': case.source_hash, 'objective_hash': goal.fingerprint,
        'decision_model_hash': task.decision_model_hash,
        'training_admission': False, 'scope': 'generated_forward_context_only'})
    context = task.development_context(semantic_digest(declaration))
    context.require_task(task)
    return context, declaration
