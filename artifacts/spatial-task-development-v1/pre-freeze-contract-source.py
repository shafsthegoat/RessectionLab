"""Small scan-conditioned sequential task, using the existing rigid cell primitive.

An action inserts a complete rigid instrument from one declared access window,
removes exposed tip-intersected cells, and withdraws on the same line. These
analytic whole-cell removals differ from the native engine's contained-cell rule.
This
is a synthetic learning task, not incision, retraction or deformable surgery.
The geometry engine has no target or functional labels. Private labels score
committed removals only; planning clones use a declared scan-only estimator.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .core import array_digest, freeze_json
from .geometry import AccessWindow, ToolGeometry
from .simulation import InvalidActionError, RewardSpec, SequentialSimulator, SimulationConfig
from .spatial_observations import (
    ObservedChannel, ObservedProcedureState, SpatialAction, SpatialInputs,
    build_spatial_observation,
)
from .worlds import content_hash

SPATIAL_TASK_VERSION = "synthetic-scan-opening-v1"
GEOMETRIC_OBJECTIVE_VERSION = "scan-removal-normal-effort-v1"
TISSUE_THRESHOLD = .05
NORMAL_SIGNAL = .2
MINIMUM_TARGET_CONTRAST = .30
TARGET_THRESHOLD = (NORMAL_SIGNAL + (NORMAL_SIGNAL + MINIMUM_TARGET_CONTRAST)) / 2
TARGET_ESTIMATOR = "synthetic_forward_model_midpoint_v1"
DEFAULT_REWARD = RewardSpec(target_per_mm3=1., normal_per_mm3=.2,
    motor_per_mm3=0., language_per_mm3=0., action_cost=.03,
    motion_per_mm=.001, tool_change_cost=.03, graph_edge_cost=0.)


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class SpatialTaskCase:
    """Generated source and private references; never pass this object to an actor."""
    structural_intensity: np.ndarray
    reference_target: np.ndarray
    reference_motor: np.ndarray
    reference_language: np.ndarray
    affine_ras_mm: np.ndarray
    access: AccessWindow
    tools: tuple[ToolGeometry, ...]
    recipe: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = _readonly(self.structural_intensity, np.float32)
        if image.ndim != 3 or not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError("A finite nonnegative synthetic source volume is required")
        if image.size > 4096 or min(image.shape) < 3:
            raise ValueError("This experimental task is bounded to small three-dimensional grids")
        object.__setattr__(self, "structural_intensity", image)
        for name, dtype in (("reference_target", bool), ("reference_motor", np.float32),
                            ("reference_language", np.float32)):
            value = _readonly(getattr(self, name), dtype)
            if value.shape != image.shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("Private reference fields must align with the source volume")
            if np.any(value[image <= TISSUE_THRESHOLD] != 0):
                raise ValueError("Private references must lie in observed support")
            object.__setattr__(self, name, value)
        affine = _readonly(self.affine_ras_mm, np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("A finite RAS affine is required")
        object.__setattr__(self, "affine_ras_mm", affine)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "recipe", freeze_json(self.recipe))

    @property
    def source_hash(self) -> str:
        return content_hash({"version": SPATIAL_TASK_VERSION,
            "scan": array_digest(self.structural_intensity), "affine": array_digest(self.affine_ras_mm)})

    @property
    def anatomy_hash(self) -> str:
        """Actual shape identity, excluding scan noise and world-realization seeds."""
        return content_hash({"support": array_digest(self.structural_intensity > TISSUE_THRESHOLD),
            "target": array_digest(self.reference_target), "motor_support": array_digest(self.reference_motor > 0),
            "language_support": array_digest(self.reference_language > 0), "affine": array_digest(self.affine_ras_mm)})


def make_spatial_case(anatomy_seed: int = 0, *, morphology: str = "paired_lobes",
                      tool_regime: str = "reference") -> SpatialTaskCase:
    """Generate a declared scan and distinct 3-D branch/support shapes on a 7×7×6 grid.

    Group assignment belongs to the experiment manifest, not this generator.
    Compare anatomy_hash values across groups: different seed names alone do not
    establish held-out anatomy. Functional references are deliberately unobserved.
    """
    if type(anatomy_seed) is not int or anatomy_seed < 0:
        raise ValueError("Anatomy seed must be a nonnegative integer")
    if morphology not in {"paired_lobes", "extended_lobes"} or tool_regime not in {"reference", "thick_shaft"}:
        raise ValueError("Choose a declared morphology and tool regime")
    rng = np.random.default_rng(anatomy_seed)
    shape = (7, 7, 6)
    support = np.zeros(shape, bool)
    support[1:6, 1:6, :] = True
    corner = int(rng.integers(4))
    depth_cut = int(rng.integers(3, 6))
    corner_x, corner_y = ((1, 1), (1, 5), (5, 1), (5, 5))[corner]
    support[corner_x, corner_y, depth_cut:] = False
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    chosen = rng.choice(4, size=2, replace=False)
    depths = tuple(int(value) for value in rng.integers(
        2 if morphology == "paired_lobes" else 3, 5 if morphology == "paired_lobes" else 6, size=2))
    target = np.zeros(shape, bool)
    for branch, depth in zip(chosen, depths):
        dx, dy = directions[int(branch)]
        target[3 + dx, 3 + dy, 1:depth] = True
        # A depth-dependent lateral wing makes geometry vary beyond rotations.
        if rng.random() < .75 or morphology == "extended_lobes":
            side = -1 if rng.random() < .5 else 1
            target[3 + dx + (side if dy else 0), 3 + dy + (side if dx else 0), 2:depth + 1] = True
    target &= support
    motor, language = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    dx, dy = directions[int(chosen[0])]
    motor[3 + dx, 3 + dy, 1:depths[0]] = float(rng.uniform(.2, .8))
    dx, dy = directions[int(chosen[1])]
    language[3 + dx, 3 + dy, 1:depths[1]] = float(rng.uniform(.1, .7))
    contrast = float(rng.uniform(.30, .65))
    noise_std = float(rng.uniform(.01, .05))
    bias_strength = float(rng.uniform(-.04, .04))
    coordinates = np.indices(shape)
    intensity = NORMAL_SIGNAL + contrast * target + bias_strength * (coordinates[0] - 3) / 3
    intensity += rng.normal(0., noise_std, shape)
    intensity = np.where(support, np.clip(intensity, .1, 1.), 0.).astype(np.float32)
    # Dimensions are given to the actor, and enter the existing swept geometry.
    narrow_radius = float(rng.uniform(.10, .16) if tool_regime == "reference" else rng.uniform(.18, .24))
    broad_radius = float(rng.uniform(.23, .34) if tool_regime == "reference" else rng.uniform(.34, .42))
    tools = (
        ToolGeometry("spatial-fine", narrow_radius, narrow_radius, 12., 70., .15),
        ToolGeometry("spatial-broad", broad_radius, broad_radius, 12., 70., .15),
    )
    return SpatialTaskCase(intensity, target, motor, language, np.eye(4),
        AccessWindow((3., 3., -.5), (0., 0., 1.), 1.6, "synthetic-fixed-access"), tools,
        {"version": SPATIAL_TASK_VERSION, "anatomy_seed": anatomy_seed, "shape": shape,
         "morphology": morphology, "tool_regime": tool_regime,
         "branches": [int(value) for value in chosen], "depths": depths,
         "support_corner": corner, "support_depth_cut": depth_cut,
         "forward_model": "clip(.2 + contrast*reference_target + linear_bias + Gaussian_noise,.1,1) inside support; zero outside",
         "contrast": contrast, "noise_std": noise_std, "bias_strength": bias_strength,
         "tissue_estimator": "structural_intensity > .05",
         "planning_target_estimator": TARGET_ESTIMATOR,
         "planning_target_threshold": TARGET_THRESHOLD,
         "planning_target_formula": "(.2 + (.2 + .30)) / 2; public synthetic class bounds, not an MRI threshold",
         "functional_measurements": "unavailable; reference fields only score simulated outcomes"})


@dataclass(frozen=True)
class SpatialStep:
    observation: Any
    reward: float
    terminated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated


class SpatialTask:
    """Evaluator environment; policies receive observation(), never this object.

    planning_clone() removes all private reference fields from its case and
    scores only the fixed intensity estimator. Its rewards are an observed-data
    search objective, not predictions of unmeasured functional outcomes.
    """
    def __init__(self, case: SpatialTaskCase, *, world_seed: int = 0, max_steps: int = 6,
                 reward: RewardSpec = DEFAULT_REWARD, _planning: bool = False):
        if type(max_steps) is not int or not 1 <= max_steps <= 12:
            raise ValueError("Transition horizon must be an integer from one to twelve")
        if reward.motor_per_mm3 != 0 or reward.language_per_mm3 != 0 or reward.graph_edge_cost != 0:
            raise ValueError("This geometric screen requires zero functional and graph reward weights; function remains unassessed")
        self.case, self.max_steps, self.reward_spec = case, max_steps, reward
        self._planning = bool(_planning)
        support = case.structural_intensity > TISSUE_THRESHOLD
        # No target ordering, surrogate hazard field or hidden obstacle enters
        # the shared candidate generator or complete-tool occupancy checks.
        config = SimulationConfig(support, np.zeros(support.shape, np.int16), case.affine_ras_mm,
            case.access, case.tools, max_steps=max_steps, max_actions=1 + support.size * len(case.tools),
            proposal_scan_limit=support.size, reward=RewardSpec(), evidence_available=(False, False),
            source_hash=case.source_hash, case_id=case.source_hash,
            derivation={"task": SPATIAL_TASK_VERSION, "proposal_basis": "observed_support_and_cavity_only"})
        self._geometry = SequentialSimulator(config)
        self._frozen_source_hash = case.source_hash
        self._frozen_geometry_hash = config.decision_model_hash
        self._frozen_reward = asdict(reward)
        self._decision_model_hash = content_hash({"geometry": self._frozen_geometry_hash,
            "reward": asdict(reward), "task": SPATIAL_TASK_VERSION,
            "objective_version": GEOMETRIC_OBJECTIVE_VERSION,
            "target_estimator": TARGET_ESTIMATOR,
            "target_estimator_threshold": TARGET_THRESHOLD, "outcome_mode": "planning" if _planning else "private_evaluation"})
        self.reset(world_seed)

    @property
    def decision_model_hash(self) -> str:
        return self._decision_model_hash

    @property
    def terminated(self) -> bool:
        return self._terminated

    def reset(self, world_seed: int = 0):
        if type(world_seed) is not int or world_seed < 0:
            raise ValueError("World seed must be a nonnegative integer")
        self._geometry.reset(0)
        self._world_seed = world_seed
        self._steps, self._total_reward, self._terminated = 0, 0., False
        self._current_tool = None
        self._history: list[dict[str, Any]] = []
        self._target = _readonly(self.case.reference_target, bool)
        # World seeds vary unobserved functional magnitude; they never change
        # source pixels, geometry, proposal count, ordering or legal masks.
        rng = np.random.default_rng(world_seed)
        scale = np.ones(2) if world_seed == 0 or self._planning else rng.uniform(.7, 1.3, 2)
        self._motor = _readonly(np.asarray(self.case.reference_motor) * scale[0], np.float64)
        self._language = _readonly(np.asarray(self.case.reference_language) * scale[1], np.float64)
        self._world_hash = content_hash({"target": array_digest(self._target),
            "motor": array_digest(self._motor), "language": array_digest(self._language)})
        return self.observation()

    def _assert_frozen(self) -> None:
        if (self.case.source_hash != self._frozen_source_hash
                or self._geometry.config.decision_model_hash != self._frozen_geometry_hash
                or asdict(self.reward_spec) != self._frozen_reward):
            raise RuntimeError("Observed task source, geometry or reward changed during an episode")

    def observation(self):
        self._assert_frozen()
        channels = {
            "structural_intensity": ObservedChannel(self.case.structural_intensity,
                source_kind="synthetic_scan", derivation="declared synthetic intensity forward model"),
            "nominal_tissue": ObservedChannel((self.case.structural_intensity > TISSUE_THRESHOLD).astype(np.float32),
                source_kind="derived_from_scan", derivation="structural_intensity > .05",
                derived_from=("structural_intensity",)),
            "observed_cavity": ObservedChannel(self._geometry.removed_mask.astype(np.float32),
                source_kind="observed_procedure_state", derivation="committed discrete removal only"),
        }
        inputs = SpatialInputs(channels, self.case.affine_ras_mm,
                               track="synthetic_scan", source_id=self.case.source_hash)
        actions = [SpatialAction("STOP")]
        if not self._terminated:
            tools = {tool.tool_id: tool for tool in self.case.tools}
            actions.extend(SpatialAction(action.action_id, self.case.access.center_mm,
                action.tip_mm, tools[action.tool_id]) for action in self._geometry.proposed_actions()[1:])
        state = ObservedProcedureState(self.case.access, self._steps, self.max_steps, self._current_tool)
        return build_spatial_observation(inputs, tuple(actions), state)

    def step(self, action: str | int) -> SpatialStep:
        if self._terminated:
            raise InvalidActionError("The spatial task has terminated")
        observation = self.observation()
        ids = observation.action_ids
        if type(action) is int or isinstance(action, np.integer):
            if action < 0 or action >= len(ids):
                raise InvalidActionError("Action index is outside the current inventory")
            action = ids[int(action)]
        if not isinstance(action, str) or action not in ids:
            raise InvalidActionError("Unknown or inaccessible spatial action")
        prior_tool = self._current_tool
        result = self._geometry.step(action)
        self._steps += 1
        target = normal = motor = language = distance = 0.
        if action != "STOP":
            indices = np.asarray(result.info["removed_indices"], dtype=int)
            coordinates = tuple(indices.T)
            volume = self._geometry.voxel_volume_mm3
            target = float(self._target[coordinates].sum() * volume)
            normal = float((~self._target[coordinates]).sum() * volume)
            motor = float(self._motor[coordinates].sum() * volume)
            language = float(self._language[coordinates].sum() * volume)
            self._current_tool = action.split(":")[1]
            distance = float(np.linalg.norm(np.asarray(result.info["tip_mm"]) - self.case.access.center_mm))
            w = self.reward_spec
            value = (w.target_per_mm3 * target - w.normal_per_mm3 * normal
                - w.motor_per_mm3 * motor - w.language_per_mm3 * language - w.action_cost
                - 2 * w.motion_per_mm * distance
                - w.tool_change_cost * (prior_tool is not None and prior_tool != self._current_tool))
        else:
            value = 0.
        self._terminated = action == "STOP" or self._steps >= self.max_steps
        self._total_reward += value
        record = {"action_id": action, "reward": float(value), "target_removed_mm3": target,
            "normal_removed_mm3": normal,
            "motor_surrogate": None if self._planning else motor,
            "language_surrogate": None if self._planning else language,
            "function_unassessed_removed_volume_mm3": target + normal,
            "insertion_distance_mm": distance, "withdrawal_distance_mm": distance,
            "complete_tool_path_length_mm": 2 * distance,
            "removed_indices": result.info.get("removed_indices", []),
            "outcome_scope": "observed_scan_estimator" if self._planning else "private_synthetic_reference"}
        self._history.append(copy.deepcopy(record))
        return SpatialStep(self.observation(), float(value), self._terminated, record)

    def clone(self) -> SpatialTask:
        """Evaluator-only state copy. Search must use planning_clone()."""
        result = copy.copy(self)
        result._geometry = self._geometry.clone()
        # Latent fields are immutable and fixed per episode; mutable geometry
        # and evaluator history are copied independently above and below.
        result._history = copy.deepcopy(self._history)
        return result

    def planning_clone(self) -> SpatialTask:
        """Same observed state/actions, with private labels removed completely."""
        target = self.case.structural_intensity >= TARGET_THRESHOLD
        zeros = np.zeros(target.shape, np.float32)
        public_case = SpatialTaskCase(self.case.structural_intensity, target, zeros, zeros,
            self.case.affine_ras_mm, self.case.access, self.case.tools,
            {"version": SPATIAL_TASK_VERSION, "planning_objective": TARGET_ESTIMATOR,
             "target_threshold": TARGET_THRESHOLD,
             "missing_functional_evidence": "unassessed; omitted from search objective"})
        result = SpatialTask(public_case, max_steps=self.max_steps, reward=self.reward_spec, _planning=True)
        # Replay preserves geometry and observed memory without copying any
        # private reward, history, world seed or outcome summary into search.
        for record in self._history:
            result.step(record["action_id"])
        return result

    def fresh(self, *, world_seed: int = 0) -> SpatialTask:
        return SpatialTask(self.case, world_seed=world_seed, max_steps=self.max_steps,
                           reward=self.reward_spec, _planning=self._planning)

    def metrics(self) -> dict[str, Any]:
        """Evaluator receipt; not an actor input or search observation."""
        return {"task_version": SPATIAL_TASK_VERSION, "source_hash": self.case.source_hash,
            "objective_version": GEOMETRIC_OBJECTIVE_VERSION, "reward_weights": asdict(self.reward_spec),
            "planning_target_estimator": TARGET_ESTIMATOR, "planning_target_threshold": TARGET_THRESHOLD,
            "anatomy_hash": self.case.anatomy_hash, "decision_model_hash": self.decision_model_hash,
            "world_hash": self._world_hash, "world_seed": self._world_seed,
            "steps": self._steps, "terminated": self._terminated, "total_reward": self._total_reward,
            "target_removed_mm3": sum(item["target_removed_mm3"] for item in self._history),
            "normal_removed_mm3": sum(item["normal_removed_mm3"] for item in self._history),
            "function_unassessed_removed_volume_mm3": sum(item["function_unassessed_removed_volume_mm3"] for item in self._history),
            "motor_surrogate_exposure": None if self._planning else sum(item["motor_surrogate"] for item in self._history),
            "language_surrogate_exposure": None if self._planning else sum(item["language_surrogate"] for item in self._history),
            "insertion_distance_mm": sum(item["insertion_distance_mm"] for item in self._history),
            "withdrawal_distance_mm": sum(item["withdrawal_distance_mm"] for item in self._history),
            "complete_tool_path_length_mm": sum(item["complete_tool_path_length_mm"] for item in self._history),
            "history": copy.deepcopy(self._history), "planning_estimator_only": self._planning,
            "candidate_inventory": self.candidate_inventory(),
            "clinical_deficit_probability": None,
            "assumptions": ["rigid_tip_intersected_cell_suction_not_native_contained_cell_removal",
                "no_force_or_deformation_model", "geometric_objective_does_not_optimize_functional_outcomes",
                "functional_measurements_unavailable", "not_a_patient"]}

    def candidate_inventory(self) -> dict[str, Any]:
        """Full observed-frontier denominator; no geometry or action cap omission."""
        frontier = self._geometry.frontier_mask() if not self._terminated else np.zeros_like(self._geometry.remaining_mask)
        cells = np.argwhere(frontier)
        accepted = [] if self._terminated else [action.action_id for action in self._geometry.proposed_actions()[1:]]
        slots = len(cells) * len(self.case.tools)
        return {"basis": "every_observed_frontier_cell_times_every_declared_tool",
            "frontier_cells": int(len(cells)), "tool_count": len(self.case.tools),
            "slot_count": int(slots), "accepted_action_ids": accepted,
            "accepted_count": len(accepted), "rejected_geometry_or_cavity_count": int(slots - len(accepted)),
            "omitted_count": 0, "complete": True,
            "scope": "complete_local_primitive_inventory_not_global_surgical_paths"}

    def independent_geometry_check(self):
        from .evaluation import independent_check_sequence
        config = self._geometry.config
        return independent_check_sequence(lambda: SequentialSimulator(config),
                                           [record["action_id"] for record in self._history])


def make_spatial_task(anatomy_seed: int = 0, *, world_seed: int = 0, max_steps: int = 6,
                      morphology: str = "paired_lobes", tool_regime: str = "reference") -> SpatialTask:
    return SpatialTask(make_spatial_case(anatomy_seed, morphology=morphology, tool_regime=tool_regime),
                       world_seed=world_seed, max_steps=max_steps)


def make_opening_task() -> SpatialTask:
    """One tiny development fixture with an exactly enumerable three-step optimum."""
    target = np.zeros((3, 3, 3), bool)
    target[0, 1, 1] = True
    scan = np.full(target.shape, .2, np.float32)
    scan[target] = .8
    zeros = np.zeros_like(scan)
    case = SpatialTaskCase(scan, target, zeros, zeros, np.eye(4),
        AccessWindow((1., 1., -.5), (0., 0., 1.), 1.6, "analytic-opening"),
        (ToolGeometry("opening-tool", .12, .12, 12., 70., .15),),
        {"version": SPATIAL_TASK_VERSION, "purpose": "development_exact_short_horizon_reference",
         "forward_model": ".2 background tissue plus .6 target contrast; no noise"})
    return SpatialTask(case, max_steps=3)
