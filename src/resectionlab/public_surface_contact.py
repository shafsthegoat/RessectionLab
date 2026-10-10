"""Generated public geometric goal over NativeSpatialTask; no new simulator.

The goal means contact a retained native cell with the existing tangential
probe. It is not anatomy, a force measurement, or clinical task success.
No learning API, checkpoint loader, file reader, or patient admission is added.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import copy
import re
import numpy as np

from .core import array_digest, freeze_json, semantic_digest, thaw_json
from .development_episode import make_development_task, replay_frames, MODES
from .native_spatial_task import NativeSpatialTask
from .sequential_spatial_observation import SequentialSpatialObservation
from .simulation import RewardSpec

OBJECTIVE_VERSION = 'public-retained-surface-contact-objective-v1'
OBSERVATION_VERSION = 'public-goal-sequential-spatial-observation-v2'
CONTEXT_VERSION = 'generated-public-contact-context-v2'
HASH = re.compile(r'sha256:[0-9a-f]{64}\Z')
# Existing effort weights, with ALL removal charged and no tumor reward.
COSTS = RewardSpec(target_per_mm3=0., normal_per_mm3=.2, motor_per_mm3=0.,
    language_per_mm3=0., action_cost=.03, motion_per_mm=.001,
    tool_change_cost=.03, graph_edge_cost=0.)


@dataclass(frozen=True)
class PublicSurfaceGoal:
    source_hash: str
    native_index: tuple[int, int, int]

    def __post_init__(self):
        object.__setattr__(self, 'native_index', tuple(self.native_index))
        self.record()

    def record(self):
        if (not isinstance(self.source_hash, str) or not HASH.fullmatch(self.source_hash)
                or len(self.native_index) != 3 or any(type(v) is not int or v < 0 for v in self.native_index)):
            raise ValueError('Public goal requires a bound source and three native integer indices')
        return {'version': OBJECTIVE_VERSION, 'source_hash': self.source_hash,
            'native_index': self.native_index, 'completion_value': 1., 'costs': asdict(COSTS),
            'meaning': 'committed_probe_contact_AND_currently_retained_cell',
            'clinical_or_sensor_claim': False}

    @property
    def fingerprint(self):
        return semantic_digest(self.record())


@dataclass(frozen=True)
class SurfaceContactObservation:
    base: SequentialSpatialObservation
    public_goal_grid: np.ndarray
    objective: PublicSurfaceGoal
    crop_origin_native: tuple[int, int, int]
    decision_model_hash: str
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.base) is not SequentialSpatialObservation:
            raise TypeError('Goal observation requires the exact shared mode/contact DTO')
        self.base.assert_intact()
        goal = np.asarray(self.public_goal_grid)
        origin = tuple(self.crop_origin_native)
        if (type(self.objective) is not PublicSurfaceGoal
                or self.objective.source_hash != self.base.source_id
                or len(origin) != 3 or any(type(v) is not int or v < 0 for v in origin)):
            raise ValueError('Goal provenance requires the bound public source and native crop origin')
        self.objective.record()
        point = np.subtract(self.objective.native_index, origin)
        if (goal.dtype != np.bool_ or goal.shape != self.base.observed_probe_contact_grid.shape
                or int(goal.sum()) != 1 or np.any(goal & ~self.base.base.coverage[0])
                or np.any(point < 0) or np.any(point >= goal.shape) or not goal[tuple(point)]
                or any(not isinstance(h, str) or not HASH.fullmatch(h)
                       for h in (self.objective_hash, self.decision_model_hash))):
            raise ValueError('Goal grid differs from the declared native goal and observed crop')
        object.__setattr__(self, 'crop_origin_native', origin)
        object.__setattr__(self, 'public_goal_grid', np.frombuffer(goal.tobytes(), bool).reshape(goal.shape))
        object.__setattr__(self, '_identity', self._digest())

    def _digest(self):
        return semantic_digest({'version': OBSERVATION_VERSION, 'base': self.base.fingerprint,
            'goal': array_digest(self.public_goal_grid), 'objective': self.objective_hash,
            'crop_origin_native': self.crop_origin_native, 'decision_model': self.decision_model_hash})

    def assert_intact(self):
        self.base.assert_intact()
        if self.public_goal_grid.flags.writeable or self._digest() != self._identity:
            raise ValueError('Public goal observation changed after construction')

    @property
    def objective_hash(self): return self.objective.fingerprint
    @property
    def fingerprint(self):
        self.assert_intact()
        return self._identity

    @property
    def action_ids(self): return self.base.action_ids
    @property
    def action_mask(self): return self.base.action_mask
    @property
    def action_tool_ids(self): return self.base.action_tool_ids
    @property
    def action_modes(self): return self.base.action_modes
    @property
    def source_id(self): return self.base.source_id
    @property
    def track(self): return self.base.track


@dataclass(frozen=True)
class SurfaceContactDevelopmentContext:
    """Bind detached goal grids to one task/crop; no training API admission."""
    declaration_hash: str
    source_hash: str
    decision_model_hash: str
    objective_hash: str
    goal_grid_hash: str
    crop_origin_native: tuple[int, int, int]
    crop_affine_hash: str
    version: str = CONTEXT_VERSION
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'crop_origin_native', tuple(self.crop_origin_native))
        self._validate()
        object.__setattr__(self, '_identity', semantic_digest(self._record()))

    def _record(self):
        return {key: getattr(self, key) for key in ('declaration_hash', 'source_hash',
            'decision_model_hash', 'objective_hash', 'goal_grid_hash',
            'crop_origin_native', 'crop_affine_hash', 'version')}

    def _validate(self):
        if (self.version != CONTEXT_VERSION or len(self.crop_origin_native) != 3
                or any(type(v) is not int or v < 0 for v in self.crop_origin_native)
                or any(not isinstance(v, str) or not HASH.fullmatch(v)
                for v in (self.declaration_hash, self.source_hash, self.decision_model_hash,
                          self.objective_hash, self.goal_grid_hash, self.crop_affine_hash))):
            raise ValueError('Explicit generated goal/grid contract required')

    def assert_intact(self):
        self._validate()
        if semantic_digest(self._record()) != self._identity:
            raise ValueError('Generated goal/grid context changed after construction')

    def require_task(self, task):
        self.assert_intact()
        if type(task) is not SurfaceContactTask:
            raise TypeError('Context requires the exact generated contact adapter')
        task._assert_frozen()
        if (task.case.track != 'synthetic_scan' or task.max_steps != 2
                or task.case.source_hash != self.source_hash
                or task.decision_model_hash != self.decision_model_hash
                or task.objective.fingerprint != self.objective_hash):
            raise ValueError('Task differs from declared generated source/objective')
        self.require_observation(task.observation())

    def require_observation(self, observation):
        self.assert_intact()
        if type(observation) is not SurfaceContactObservation:
            raise TypeError('Explicit goal/mode observation required')
        observation.assert_intact()
        if (observation.track != 'synthetic_scan' or observation.source_id != self.source_hash
                or observation.decision_model_hash != self.decision_model_hash
                or observation.objective_hash != self.objective_hash
                or array_digest(observation.public_goal_grid) != self.goal_grid_hash
                or observation.crop_origin_native != self.crop_origin_native
                or array_digest(observation.base.base.affine_ras_mm) != self.crop_affine_hash
                or observation.base.base.state_features[1] != 2):
            raise ValueError('Observation differs from declared source/objective')


class SurfaceContactTask(NativeSpatialTask):
    """Only observation, public scoring and identity differ; transitions inherited."""
    def __init__(self, case, *, objective: PublicSurfaceGoal, cancelled=None, _planning=False):
        if type(objective) is not PublicSurfaceGoal or objective.source_hash != case.source_hash:
            raise ValueError('Goal must bind this exact public source')
        point = np.asarray(objective.native_index)
        origin, shape = np.asarray(case._crop_origin), np.asarray(case._crop_shape)
        if (case.track != 'synthetic_scan' or np.any(point >= case.observed_support.shape)
                or not case.observed_support[objective.native_index]
                or np.any(point < origin) or np.any(point >= origin + shape)):
            raise ValueError('Generated goal must be an observed occupied cell inside actor crop')
        self.objective = objective
        super().__init__(case, max_steps=2, reward=COSTS, cancelled=cancelled,
                         _planning=_planning, tool_modes=MODES)

    def _contract_record(self):
        return {**super()._contract_record(), 'public_surface_objective': self.objective.record(),
            'target_model': 'unused_by_public_surface_contact_objective',
            'observation_contract': OBSERVATION_VERSION}

    def observation(self):
        base = super().observation()
        grid = np.zeros(base.observed_probe_contact_grid.shape, bool)
        index = tuple(np.asarray(self.objective.native_index) - self.case._crop_origin)
        grid[index] = True
        return SurfaceContactObservation(base, grid, self.objective, self.case._crop_origin,
                                         self.decision_model_hash)

    def development_context(self, declaration_hash):
        """Capture expected public goal/grid/frame before a detached consumer."""
        obs = self.observation()
        return SurfaceContactDevelopmentContext(declaration_hash, self.case.source_hash,
            self.decision_model_hash, self.objective.fingerprint,
            array_digest(obs.public_goal_grid), obs.crop_origin_native,
            array_digest(obs.base.base.affine_ras_mm))

    def _potential(self):
        point = self.objective.native_index
        return int(self._engine.probe_contact_mask[point] and self._engine.remaining_mask[point])

    def _score_record(self, geometry, prior_tool):
        before = self._potential()
        stop = geometry.get('action_id') == 'STOP'
        removed = {tuple(v) for v in geometry.get('removed_indices_native', ())}
        contact = {tuple(v) for v in geometry.get('contact_indices_native', ())}
        point = self.objective.native_index
        remaining = bool(self._engine.remaining_mask[point] and point not in removed)
        observed_contact = bool(self._engine.probe_contact_mask[point] or
            (geometry.get('interaction_mode') == 'probe' and point in contact))
        after = int(remaining and observed_contact)
        distance = 0. if stop else float(np.linalg.norm(np.subtract(geometry['tip_mm'], geometry['entry_mm'])))
        volume = len(removed) * float(self._config.voxel_volume_mm3)
        change = int(not stop and prior_tool is not None and prior_tool != geometry['tool_id'])
        cost = COSTS.normal_per_mm3 * volume + COSTS.action_cost * (not stop) + 2*COSTS.motion_per_mm*distance + COSTS.tool_change_cost*change
        return {**geometry, 'reward': float(after-before-cost),
            'goal_potential_before': before, 'goal_potential_after': after,
            'public_objective_hash': self.objective.fingerprint, 'removal_cost_volume_mm3': volume,
            'insertion_distance_mm': distance, 'complete_tool_path_length_mm': 2*distance,
            'tool_change_count': change, 'effort_and_removal_cost': float(cost),
            'outcome_scope': 'public_geometric_retained_surface_contact',
            'clinical_deficit_probability': None}

    def planning_clone(self):
        # Public scoring is already independent of private target. Replace that
        # unused field as well, so the search branch carries no private target.
        result = self.clone()
        result.case = copy.copy(self.case)
        public_zero = np.frombuffer(np.zeros(self.case.observed_support.shape, np.float32).tobytes(), np.float32).reshape(self.case.observed_support.shape)
        object.__setattr__(result.case, 'reference_target', public_zero)
        object.__setattr__(result.case, '_identity', result.case._identity_record())
        object.__setattr__(result.case, '_reference_hash', semantic_digest({'source': self._source_hash, 'target': array_digest(public_zero)}))
        result._reference_hash, result._planning = result.case.reference_hash, True
        # Keep authoritative prefix rewards. Do not rescore using final contact.
        result._seal()
        return result

    def fresh(self):
        self._assert_frozen()
        # Base constructor's planning guard assumes a nominal removal target;
        # contact planning is instead established by our planning_clone above.
        result = type(self)(self.case, objective=self.objective, cancelled=self._cancelled)
        return result.planning_clone() if self._planning else result

    def metrics(self):
        self._assert_frozen()
        point = self.objective.native_index
        return {'task_version': OBJECTIVE_VERSION, 'source_hash': self._source_hash,
            'decision_model_hash': self.decision_model_hash, 'steps': self._steps,
            'terminated': self._terminated, 'total_reward': self._total_reward,
            'planning_estimator_only': self._planning, 'history': copy.deepcopy(self._history),
            'observation_track': self.case.track, 'objective': self.objective.record(),
            'goal_retained': bool(self._engine.remaining_mask[point]),
            'goal_contacted_and_retained': bool(self._potential()),
            'removed_volume_mm3': int(self._engine.removed_mask.sum())*float(self._config.voxel_volume_mm3),
            'outcome_scope': 'public_geometric_goal_no_private_target_scoring',
            'clinical_validation': False}


def make_contact_task(*, goal=(6, 6, 3), reference_target=None, cancelled=None):
    case = make_development_task(reference_target=reference_target, cancelled=cancelled).case
    return SurfaceContactTask(case, objective=PublicSurfaceGoal(case.source_hash, goal), cancelled=cancelled)


def seal_complete_strategy(task, actions):
    """Reuse development_episode's complete strategy fields and replay frames."""
    worker = task.fresh()
    for action in actions:
        worker.step(action)
    if not worker.terminated:
        raise ValueError('Only an explicitly stopped or horizon-complete strategy may be sealed')
    strategy = thaw_json(freeze_json({'decision_model_hash': task.decision_model_hash,
        'source_hash': task.case.source_hash, 'actions': list(actions),
        'history': worker.metrics()['history'], 'observation_contract': OBSERVATION_VERSION,
        'max_steps': task.max_steps}))
    audit = worker.independent_geometry_check().to_dict()
    if not audit['feasible']:
        raise ValueError('Independent full-tool geometry check failed')
    return {'strategy': strategy, 'strategySeal': semantic_digest(strategy),
            'geometryAudit': audit, 'replayFrames': replay_frames(worker, strategy['history']),
            'metrics': worker.metrics()}


def verify_complete_strategy(task, package):
    strategy = package['strategy']
    if semantic_digest(strategy) != package['strategySeal']:
        raise ValueError('Strategy seal changed')
    rebuilt = seal_complete_strategy(task, strategy['actions'])
    # Canonical JSON comparison also rejects bool/int and int/float changes;
    # ordinary Python container equality would accept those substitutions.
    if semantic_digest(rebuilt) != semantic_digest(package):
        raise ValueError('Authoritative native replay differs from sealed strategy')
    return True
