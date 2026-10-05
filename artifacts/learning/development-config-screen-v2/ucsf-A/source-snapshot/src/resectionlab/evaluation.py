"""Independent candidate validation and model-conditioned event accounting.

Final evaluation consumes an already frozen candidate set. Its results are not a
checkpoint-selection signal. The geometry routines intentionally do not import
the planning collision implementation: orthogonal voxel cells are checked using
an exact piecewise-quadratic segment/box distance, with conservative continuous
motion enclosures. These are geometry certificates, never surgical clearance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .worlds import (FrozenDecisionModel, LatentWorld, WorldGenerator,
                     WorldPartitionManifest, WorldRole, content_hash,
                     validate_training_partitions)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Binomial Monte Carlo interval, not uncertainty about the clinical model."""
    if isinstance(successes, bool) or isinstance(trials, bool) or not isinstance(successes, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise ValueError("Event counts must be integers")
    if successes < 0 or trials < successes or not 0 < confidence < 1:
        raise ValueError("Invalid event counts or interval confidence")
    if trials == 0:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def upper_tail_cvar(costs: Sequence[float], alpha: float = 0.95) -> float:
    """Mean of the worst (1-alpha) empirical mass, with fractional boundary mass.

    Using ``cost >= quantile`` would incorrectly include an entire tied mass.
    Costs remain in the caller's declared surrogate units.
    """
    values = np.asarray(costs, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 <= alpha < 1:
        raise ValueError("CVaR requires finite costs and 0 <= alpha < 1")
    descending = np.sort(values)[::-1]
    tail_mass = (1 - alpha) * len(values)
    full = min(int(math.floor(tail_mass)), len(values))
    remainder = tail_mass - full
    total = float(descending[:full].sum())
    if remainder > 0 and full < len(values):
        total += remainder * float(descending[full])
    return total / tail_mass


@dataclass(frozen=True)
class EventSummary:
    event: str
    definition: str
    numerator: int
    denominator: int
    assessed_worlds: int
    unknown_worlds: int
    model_conditioned_frequency: float | None
    monte_carlo_interval: tuple[float, float] | None
    world_model_version: str
    world_partition_hash: str
    interpretation: str
    clinical_deficit_probability: None = None

    def __post_init__(self) -> None:
        if self.clinical_deficit_probability is not None:
            raise ValueError("A modeled event frequency is not a clinical deficit probability")


def summarize_event(event: str, definition: str, outcomes: Sequence[bool | None],
                    partition: WorldPartitionManifest) -> EventSummary:
    if not event or not definition or len(outcomes) != len(partition.seeds):
        raise ValueError("A named event, definition and one outcome per world are required")
    if any(value is not None and not isinstance(value, (bool, np.bool_)) for value in outcomes):
        raise ValueError("World event outcomes must be boolean or unknown")
    successes = sum(bool(value) for value in outcomes if value is not None)
    assessed = sum(value is not None for value in outcomes)
    total = len(outcomes)
    unknown = total - assessed
    frequency = successes / total if unknown == 0 else None
    deterministic = partition.generator.deterministic
    interval = wilson_interval(successes, total) if unknown == 0 and not deterministic else None
    interpretation = ("unknown_due_to_incomplete_anatomical_coverage" if unknown else
                      "deterministic_replay_no_uncertainty_validation" if deterministic else
                      "model_conditioned_event_frequency; interval_is_finite_Monte_Carlo_error_only")
    return EventSummary(event, definition, successes, total, assessed, unknown, frequency,
                        interval, partition.generator.version, partition.partition_hash, interpretation)


def _candidate_identity(candidate: Any) -> tuple[str, str]:
    if not hasattr(candidate, "plan_id") or not hasattr(candidate, "semantic_hash"):
        raise TypeError("Candidates require plan_id and semantic_hash")
    return str(candidate.plan_id), str(candidate.semantic_hash)


@dataclass(frozen=True)
class CandidateFreezeManifest:
    case_hash: str
    decision_model_fingerprint: str
    candidate_hashes: tuple[tuple[str, str], ...]
    optimization_partition_hash: str
    selection_partition_hash: str
    development_seeds: tuple[int, ...]
    generator_fingerprint: str
    generator_family: str
    selection_rule: str
    frozen_at: str

    def __post_init__(self) -> None:
        pairs = tuple(tuple(pair) for pair in self.candidate_hashes)
        if not pairs or any(len(pair) != 2 or any(not isinstance(v, str) or not v for v in pair) for pair in pairs) or len({p[0] for p in pairs}) != len(pairs):
            raise ValueError("Frozen candidates need unique identifiers and hashes")
        for name in ("case_hash", "decision_model_fingerprint", "optimization_partition_hash",
                     "selection_partition_hash", "generator_fingerprint", "generator_family",
                     "selection_rule", "frozen_at"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Frozen candidate manifest requires nonempty {name}")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.development_seeds):
            raise ValueError("Frozen development seeds must be nonnegative integers")
        object.__setattr__(self, "candidate_hashes", pairs)
        object.__setattr__(self, "development_seeds", tuple(self.development_seeds))

    @property
    def fingerprint(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fingerprint": self.fingerprint}


def freeze_candidates(candidates: Sequence[Any], decision_model: FrozenDecisionModel,
                      selection_manifest: WorldPartitionManifest, selection_rule: str, *,
                      optimization_manifest: WorldPartitionManifest) -> CandidateFreezeManifest:
    validate_training_partitions(optimization_manifest, selection_manifest, decision_model)
    if any(candidate.case_hash != decision_model.case_hash for candidate in candidates):
        raise ValueError("Candidate belongs to an obsolete or different case")
    return CandidateFreezeManifest(
        decision_model.case_hash, decision_model.fingerprint,
        tuple(sorted(_candidate_identity(c) for c in candidates)),
        optimization_manifest.partition_hash, selection_manifest.partition_hash,
        optimization_manifest.seeds + selection_manifest.seeds,
        selection_manifest.generator.fingerprint, selection_manifest.generator.family,
        selection_rule, datetime.now(timezone.utc).isoformat())


@dataclass
class EvaluationLedger:
    """Persist ``to_dict`` with runs; reuse of revealed worlds cannot hide tuning."""

    partition_candidates: dict[str, str] = field(default_factory=dict)

    def claim(self, partition: WorldPartitionManifest, freeze: CandidateFreezeManifest) -> None:
        # Seed overlap must be detected even after reorder, subset, superset or
        # role relabeling. A deterministic generator has one realized world.
        identities = [content_hash({"case": partition.planning_hash or partition.case_hash,
                       "generator": partition.generator.fingerprint,
                       "seed": None if partition.generator.deterministic else seed})
                      for seed in partition.seeds]
        for identity in identities:
            previous = self.partition_candidates.get(identity)
            if previous is not None and previous != freeze.fingerprint:
                raise ValueError("EVALUATION_WORLDS_REVEALED: new candidates require untouched evaluation worlds")
        for identity in identities:
            self.partition_candidates[identity] = freeze.fingerprint

    def to_dict(self) -> dict[str, str]:
        return dict(self.partition_candidates)


@dataclass(frozen=True)
class WorldOutcome:
    events: Mapping[str, bool | None]
    costs: Mapping[str, float | None] = field(default_factory=dict)
    accessible_target_volume_mm3: float | None = None
    simulated_removed_target_volume_mm3: float | None = None
    unknowns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        events = dict(self.events)
        costs = dict(self.costs)
        if any(not isinstance(name, str) or not name or
               (value is not None and not isinstance(value, (bool, np.bool_)))
               for name, value in events.items()):
            raise ValueError("World events require named boolean or unknown outcomes")
        if any(not isinstance(name, str) or not name or
               (value is not None and not np.isfinite(value)) for name, value in costs.items()):
            raise ValueError("World costs require named finite values or unknowns")
        object.__setattr__(self, "events", MappingProxyType({k: None if v is None else bool(v) for k, v in events.items()}))
        object.__setattr__(self, "costs", MappingProxyType({k: None if v is None else float(v) for k, v in costs.items()}))
        object.__setattr__(self, "unknowns", tuple(self.unknowns))

    def to_dict(self) -> dict[str, Any]:
        return {"events": dict(self.events), "costs": dict(self.costs),
                "accessible_target_volume_mm3": self.accessible_target_volume_mm3,
                "simulated_removed_target_volume_mm3": self.simulated_removed_target_volume_mm3,
                "unknowns": list(self.unknowns)}


@dataclass(frozen=True)
class IndependentGeometryResult:
    feasible: bool
    failures: tuple[str, ...] = ()
    collision_position_mm: tuple[float, float, float] | None = None
    checker_version: str = "independent-cell-capsule-v1"
    conservative: bool = True
    unknowns: tuple[str, ...] = ()


def evaluate_frozen_candidates(
    candidates: Sequence[Any], freeze: CandidateFreezeManifest,
    decision_model: FrozenDecisionModel, partition: WorldPartitionManifest, *,
    event_definitions: Mapping[str, str], cost_units: Mapping[str, str],
    world_evaluator: Callable[[Any, LatentWorld], WorldOutcome],
    geometry_checker: Callable[[Any], IndependentGeometryResult],
    ledger: EvaluationLedger, cvar_alpha: float = 0.95,
) -> dict[str, Any]:
    """Evaluate immutable candidates; there is no training or selection callback.

    The application must persist the ledger across runs. Unknown anatomical
    coverage suppresses the corresponding event frequency rather than counting
    unobserved anatomy as a non-event. All candidates, including failures, remain.
    """
    identities = tuple(sorted(_candidate_identity(c) for c in candidates))
    if identities != freeze.candidate_hashes:
        raise ValueError("Candidate set changed after selection/freeze")
    if any(c.case_hash != freeze.case_hash for c in candidates):
        raise ValueError("Frozen candidate case mismatch")
    if decision_model.fingerprint != freeze.decision_model_fingerprint:
        raise ValueError("Decision model changed after candidate freeze")
    if partition.case_hash != freeze.case_hash:
        raise ValueError("Evaluation case differs from frozen case")
    if partition.role not in {WorldRole.FINAL_EVALUATION, WorldRole.STRESS}:
        raise ValueError("Final evaluator accepts final-evaluation or withheld stress worlds only")
    if set(partition.seeds) & set(freeze.development_seeds):
        raise ValueError("Final worlds overlap optimization or checkpoint-selection worlds")
    if partition.role is WorldRole.FINAL_EVALUATION and partition.generator.fingerprint != freeze.generator_fingerprint:
        raise ValueError("Final evaluation must use the frozen world generator")
    if partition.role is WorldRole.STRESS and partition.generator.family == freeze.generator_family:
        raise ValueError("Stress evaluation requires a withheld model family")
    if not 0 <= cvar_alpha < 1 or any(not key or not text for key, text in (*event_definitions.items(), *cost_units.items())):
        raise ValueError("Evaluation definitions or CVaR alpha are invalid")
    ledger.claim(partition, freeze)
    generator = WorldGenerator(partition.generator)
    records = []
    for candidate in candidates:
        certificate = geometry_checker(candidate)
        if not isinstance(certificate, IndependentGeometryResult):
            raise TypeError("Independent geometry certificate required")
        outcomes = [world_evaluator(candidate, generator.sample(partition, i))
                    for i in range(len(partition.seeds))] if certificate.feasible else []
        for outcome in outcomes:
            if set(outcome.events) - set(event_definitions) or set(outcome.costs) - set(cost_units):
                raise ValueError("Evaluator produced an undefined event or cost")
            if getattr(candidate, "plan_type", "route_only") == "route_only" and outcome.simulated_removed_target_volume_mm3 is not None:
                raise ValueError("Route accessibility cannot be reported as simulated removal")
            for volume in (outcome.accessible_target_volume_mm3, outcome.simulated_removed_target_volume_mm3):
                if volume is not None and (not np.isfinite(volume) or volume < 0):
                    raise ValueError("Volumes must be finite nonnegative millimeters cubed")
        events = [asdict(summarize_event(name, definition,
                  [outcome.events.get(name) for outcome in outcomes], partition))
                  for name, definition in event_definitions.items()] if outcomes else []
        costs = {}
        for name, units in cost_units.items():
            values = [outcome.costs.get(name) for outcome in outcomes]
            assessed = [float(v) for v in values if v is not None]
            if not np.isfinite(assessed).all():
                raise ValueError("Evaluator cost contains nonfinite values")
            complete = bool(values) and len(assessed) == len(values)
            costs[name] = {"units": units, "mean": float(np.mean(assessed)) if complete else None,
                           "upper_tail_cvar": upper_tail_cvar(assessed, cvar_alpha) if complete else None,
                           "cvar_alpha": cvar_alpha, "assessed_worlds": len(assessed),
                           "unknown_worlds": len(values) - len(assessed)}
        records.append({"plan_id": candidate.plan_id, "geometry": asdict(certificate),
                        "status": "evaluated" if certificate.feasible else "rejected_geometry",
                        "model_events": events, "surrogate_costs": costs,
                        "world_outcomes": [outcome.to_dict() for outcome in outcomes],
                        "unknowns": sorted(set(certificate.unknowns).union(
                            *(set(outcome.unknowns) for outcome in outcomes))),
                        "clinical_deficit_probability": None,
                        "clinical_risk_reason": "no_validated_clinical_outcome_model"})
    if tuple(sorted(_candidate_identity(c) for c in candidates)) != identities:
        raise ValueError("Candidate mutated during independent evaluation")
    return {"evaluator_version": "frozen-independent-v1", "clinical_use_status": "research_only",
            "unit_of_independent_patient_inference": "one_patient",
            "candidate_freeze_hash": freeze.fingerprint,
            "decision_model_fingerprint": decision_model.fingerprint,
            "world_partition": partition.to_dict(), "world_partition_hash": partition.partition_hash,
            "candidates": records}


def segment_box_distance_sq(start: np.ndarray, end: np.ndarray,
                             lower: np.ndarray, upper: np.ndarray) -> float:
    """Exact squared distance between a finite segment and an axis-aligned box.

    On each parameter interval bounded by coordinate/face crossings, squared
    distance is a quadratic. Checking its stationary point and endpoints is exact
    up to floating-point arithmetic and catches voxel corners missed by ray tests.
    """
    start, end, lower, upper = (np.asarray(v, float) for v in (start, end, lower, upper))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (start, end, lower, upper)) or np.any(lower > upper):
        raise ValueError("Segment and box require finite ordered xyz coordinates")
    direction = end - start
    breaks = [0.0, 1.0]
    for axis in range(3):
        if abs(direction[axis]) > 1e-15:
            for face in (lower[axis], upper[axis]):
                t = (face - start[axis]) / direction[axis]
                if 0 < t < 1:
                    breaks.append(float(t))
    breaks = sorted(set(breaks))

    def distance(t: float) -> float:
        point = start + t * direction
        delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        return float(delta @ delta)

    result = min(distance(t) for t in breaks)
    for left, right in zip(breaks[:-1], breaks[1:]):
        mid = start + ((left + right) / 2) * direction
        active = (mid < lower) | (mid > upper)
        bound = np.where(mid < lower, lower, upper)
        a = float(direction[active] @ direction[active])
        if a > 0:
            b = float(direction[active] @ (start[active] - bound[active]))
            optimum = np.clip(-b / a, left, right)
            result = min(result, distance(float(optimum)))
    return result


def _pose_values(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    tip = np.asarray(pose.tip_mm, float)
    axis = np.asarray(pose.axis_unit, float)
    if tip.shape != (3,) or axis.shape != (3,) or not np.isfinite(tip).all() or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6):
        raise ValueError("Tool pose requires finite tip and unit axis")
    return tip, axis / np.linalg.norm(axis)


def _tool_dimensions(tool: Any) -> tuple[float, float, float, float]:
    values = tuple(float(getattr(tool, name)) for name in
                   ("working_length_mm", "tip_length_mm", "shaft_radius_mm", "tip_radius_mm"))
    length, tip_length, shaft_radius, tip_radius = values
    if not np.isfinite(values).all() or min(values) <= 0 or tip_length > length:
        raise ValueError("Tool dimensions must be positive, with tip length <= shaft reach")
    return length, tip_length, shaft_radius, tip_radius


def _cell_collision(scene: Any, a: np.ndarray, b: np.ndarray,
                    radius: float) -> tuple[float, float, float] | None:
    mask = np.asarray(scene.forbidden_mask, bool)
    affine = np.asarray(scene.affine, float)
    if mask.ndim != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Invalid independent geometry scene")
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0) or not np.allclose(basis.T @ basis, np.diag(spacing**2), atol=1e-7):
        raise ValueError("INDEPENDENT_GEOMETRY_UNSUPPORTED_SHEAR")
    rotation = basis / spacing
    a_local = rotation.T @ (a - affine[:3, 3])
    b_local = rotation.T @ (b - affine[:3, 3])
    lo = np.maximum(np.floor((np.minimum(a_local, b_local) - radius) / spacing - 0.5).astype(int), 0)
    hi = np.minimum(np.ceil((np.maximum(a_local, b_local) + radius) / spacing + 0.5).astype(int), np.array(mask.shape) - 1)
    if np.any(hi < lo):
        return None
    slices = tuple(slice(int(l), int(h) + 1) for l, h in zip(lo, hi))
    for index in np.argwhere(mask[slices]) + lo:
        center = index * spacing
        if segment_box_distance_sq(a_local, b_local, center - spacing / 2, center + spacing / 2) <= radius * radius + 1e-10:
            point = affine[:3, :3] @ index + affine[:3, 3]
            return tuple(float(v) for v in point)
    return None


def _sphere_collision(scene: Any, a: np.ndarray, b: np.ndarray,
                      radius: float) -> tuple[float, float, float] | None:
    for obstacle in getattr(scene, "sphere_obstacles", ()):
        # Geometry's public sphere contract uses center_mm/radius_mm. A tuple
        # representation is also useful for analytic independent fixtures.
        if hasattr(obstacle, "center_mm"):
            center, obstacle_radius = np.asarray(obstacle.center_mm, float), float(obstacle.radius_mm)
        else:
            center, obstacle_radius = np.asarray(obstacle[0], float), float(obstacle[1])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(obstacle_radius) or obstacle_radius < 0:
            raise ValueError("Invalid sphere obstacle")
        direction = b - a
        t = np.clip(np.dot(center - a, direction) / np.dot(direction, direction), 0, 1) if np.dot(direction, direction) else 0
        if np.linalg.norm(center - (a + t * direction)) <= radius + obstacle_radius + 1e-9:
            return tuple(float(v) for v in center)
    return None


def _check_envelope(tool: Any, tip: np.ndarray, axis: np.ndarray, scene: Any,
                     inflation_mm: float = 0) -> IndependentGeometryResult:
    length, tip_length, shaft_radius, tip_radius = _tool_dimensions(tool)
    for component, a, b, radius in (
        ("shaft", tip - length * axis, tip - tip_length * axis, shaft_radius),
        ("tip", tip - tip_length * axis, tip, tip_radius),
    ):
        location = _cell_collision(scene, a, b, radius + inflation_mm)
        if location is None:
            location = _sphere_collision(scene, a, b, radius + inflation_mm)
        if location is not None:
            return IndependentGeometryResult(False, (f"{component}_envelope_collision",), location)
    unknowns = ["geometry_checks_only_supplied_anatomical_obstacles"]
    inverse = np.linalg.inv(np.asarray(scene.affine, float))
    proximal_index = inverse[:3, :3] @ (tip - length * axis) + inverse[:3, 3]
    if np.any(proximal_index < -0.5) or np.any(proximal_index > np.asarray(scene.forbidden_mask.shape) - 0.5):
        unknowns.append("proximal_tool_outside_image_coverage_unassessed")
    return IndependentGeometryResult(True, unknowns=tuple(unknowns))


def _access_failure(tool: Any, tip: np.ndarray, axis: np.ndarray, access: Any) -> str | None:
    center = np.asarray(access.center_mm, float)
    normal = np.asarray(access.normal_inward, float)
    if center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("Invalid access window")
    normal = normal / np.linalg.norm(normal)
    cosine = float(axis @ normal)
    if cosine <= 0 or cosine < math.cos(math.radians(float(tool.max_access_angle_deg))) - 1e-9:
        return "access_angle_exceeded"
    depth = float((tip - center) @ normal) / cosine
    if depth < -1e-9 or depth > float(tool.working_length_mm) + 1e-9:
        return "working_distance_exceeded"
    crossing = tip - depth * axis
    # Conservative plane footprint of the widest capsule. Radius/normal cosine
    # encloses its ellipse and is deliberately not an exact oblique aperture test.
    if np.linalg.norm(crossing - center) + max(float(tool.shaft_radius_mm), float(tool.tip_radius_mm)) / cosine > float(access.radius_mm) + 1e-9:
        return "full_tool_does_not_fit_access_window"
    return None


def independent_check_pose(tool: Any, pose: Any, scene: Any,
                           access: Any | None = None) -> IndependentGeometryResult:
    tip, axis = _pose_values(pose)
    _tool_dimensions(tool)
    inverse = np.linalg.inv(np.asarray(scene.affine, float))
    index = inverse[:3, :3] @ tip + inverse[:3, 3]
    if getattr(scene, "enforce_tip_in_bounds", True) and (np.any(index < -0.5) or np.any(index > np.asarray(scene.forbidden_mask.shape) - 0.5)):
        return IndependentGeometryResult(False, ("tip_outside_image_coverage",), tuple(tip))
    if access is not None:
        failure = _access_failure(tool, tip, axis, access)
        if failure:
            return IndependentGeometryResult(False, (failure,), tuple(tip))
    return _check_envelope(tool, tip, axis, scene)


def _interpolate_axis(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    cosine = float(np.clip(start @ end, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return start.copy()
    if math.pi - angle < 1e-7:
        raise ValueError("Antipodal tool rotation is ambiguous; provide an intermediate pose")
    # Independently implemented great-circle interpolation, no planning helpers.
    axis = (math.sin((1 - t) * angle) * start + math.sin(t * angle) * end) / math.sin(angle)
    return axis / np.linalg.norm(axis)


def independent_check_motion(tool: Any, start: Any, end: Any, scene: Any,
                             access: Any | None = None, *,
                             max_surface_step_mm: float = 0.25) -> IndependentGeometryResult:
    """Continuous swept-tool enclosure, including collisions between poses.

    Every intermediate capsule is enclosed by its interval midpoint capsule plus
    a translation/rotation displacement bound. This may conservatively reject
    close routes; it cannot certify a known collision away through sparse sampling.
    Aperture constraints throughout a rotating motion are currently unsupported;
    an explicit failure is returned rather than testing endpoints alone.
    """
    if not np.isfinite(max_surface_step_mm) or max_surface_step_mm <= 0:
        raise ValueError("Motion resolution must be a positive millimeter step")
    p0, a0 = _pose_values(start)
    p1, a1 = _pose_values(end)
    length, _, _, _ = _tool_dimensions(tool)
    endpoint_unknowns = set()
    for pose in (start, end):
        checked = independent_check_pose(tool, pose, scene, access)
        if not checked.feasible:
            return checked
        endpoint_unknowns.update(checked.unknowns)
    angle = math.acos(float(np.clip(a0 @ a1, -1.0, 1.0)))
    if math.pi - angle < 1e-7:
        return IndependentGeometryResult(False, ("ambiguous_antipodal_rotation",))
    if access is not None and angle > 1e-10:
        return IndependentGeometryResult(False, ("continuous_rotating_access_window_check_unsupported",))
    translation = float(np.linalg.norm(p1 - p0))
    advance = float((p1 - p0) @ a0)
    if angle < 1e-10 and np.linalg.norm((p1 - p0) - advance * a0) < 1e-9:
        # Axial insertion/retraction has an exact swept union: each component
        # capsule is simply extended along its unchanged axis. This removes
        # hundreds of redundant pose checks for the route-first workflow.
        low, high = (p0, p1) if advance >= 0 else (p1, p0)
        _, tip_length, shaft_radius, tip_radius = _tool_dimensions(tool)
        for name, a, b, radius in (
            ("shaft", low - length * a0, high - tip_length * a0, shaft_radius),
            ("tip", low - tip_length * a0, high, tip_radius),
        ):
            location = _cell_collision(scene, a, b, radius)
            if location is None:
                location = _sphere_collision(scene, a, b, radius)
            if location is not None:
                return IndependentGeometryResult(False, (f"swept_{name}_envelope_collision",), location)
        return IndependentGeometryResult(True, unknowns=tuple(sorted(endpoint_unknowns)))
    intervals = max(1, int(math.ceil((translation + length * angle) / max_surface_step_mm)))
    inflation = translation / (2 * intervals) + 2 * length * math.sin(angle / (4 * intervals))
    for i in range(intervals):
        t = (i + 0.5) / intervals
        checked = _check_envelope(tool, p0 + t * (p1 - p0), _interpolate_axis(a0, a1, t), scene, inflation)
        if not checked.feasible:
            return IndependentGeometryResult(False, tuple("swept_" + failure for failure in checked.failures),
                                             checked.collision_position_mm)
    return IndependentGeometryResult(True, unknowns=tuple(sorted(endpoint_unknowns)))


def independent_check_route(candidate: Any, scene: Any) -> IndependentGeometryResult:
    """Check a search route without reusing its stored planning certificate."""
    from types import SimpleNamespace

    entry = np.asarray(candidate.entry_mm, float)
    target = np.asarray(candidate.target_mm, float)
    direction = target - entry
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-10:
        return IndependentGeometryResult(False, ("zero_length_route",))
    axis = direction / norm
    start = SimpleNamespace(tip_mm=entry, axis_unit=axis)
    end = SimpleNamespace(tip_mm=target, axis_unit=axis)
    checked = independent_check_motion(candidate.tool, start, end, scene, candidate.window)
    return IndependentGeometryResult(checked.feasible, checked.failures, checked.collision_position_mm,
                                     unknowns=tuple(sorted(set(checked.unknowns) | set(candidate.unknowns))))


def independent_check_sequence(factory: Callable[[], Any], actions: Sequence[str], *,
                                seed: int = 0,
                                cancelled: Callable[[], bool] | None = None) -> IndependentGeometryResult:
    """Independently inspect a fixed sequence on its declared simulation grid.

    The simulator supplies macro-action proposals, while separate direct tests
    check exposed removal, swept geometry and exact occupancy accounting. This
    verifies the discrete model, not the realism of an indivisible coarse-cell
    suction footprint. Native-resolution removal remains explicitly unverified.
    The actor/checkpoint is never updated, including on a failed certificate.
    """
    from .geometry import GeometryScene, ToolPose

    simulator = factory()
    simulator.reset(seed)
    unknowns = {"discrete_suction_footprint_is_model_assumption",
                "geometry_checked_on_declared_simulation_grid",
                "native_resolution_removal_not_independently_verified",
                "vascular_anatomy_and_tissue_mechanics_unassessed"}
    for position, action_id in enumerate(actions):
        if cancelled is not None and cancelled():
            return IndependentGeometryResult(False, ("independent_validation_cancelled",))
        if simulator.terminated:
            return IndependentGeometryResult(False, ("action_after_termination",))
        if action_id == "STOP":
            if position != len(actions) - 1:
                return IndependentGeometryResult(False, ("action_after_stop",))
            simulator.step("STOP")
            continue
        action = next((a for a in simulator.proposed_actions() if a.action_id == action_id), None)
        if action is None:
            return IndependentGeometryResult(False, ("action_not_legal_in_current_cavity",))
        before = simulator.remaining_mask.copy()
        footprint = np.asarray(action.removal_indices)
        if footprint.ndim != 2 or footprint.shape[1] != 3 or footprint.dtype.kind not in "iu" or np.any(footprint < 0) or np.any(footprint >= np.array(before.shape)):
            return IndependentGeometryResult(False, ("invalid_removal_footprint",))
        if len(np.unique(footprint, axis=0)) != len(footprint):
            return IndependentGeometryResult(False, ("duplicate_removal_voxels",))
        occupied = footprint[before[tuple(footprint.T)]]
        if not len(occupied):
            return IndependentGeometryResult(False, ("repeated_or_empty_removal",))
        for index in occupied:
            exposed = False
            for axis in range(3):
                for delta in (-1, 1):
                    neighbor = index.copy()
                    neighbor[axis] += delta
                    if np.any(neighbor < 0) or np.any(neighbor >= before.shape) or not before[tuple(neighbor)]:
                        exposed = True
            if not exposed:
                return IndependentGeometryResult(False, ("enclosed_nonfrontier_removal",))
        forbidden = before.copy()
        forbidden[tuple(footprint.T)] = False
        forbidden |= simulator.config.hard_exclusion
        tool = next(tool for tool in simulator.config.tools if tool.tool_id == action.tool_id)
        checked = independent_check_motion(
            tool, ToolPose(simulator.config.access.center_mm, action.axis_unit),
            ToolPose(action.tip_mm, action.axis_unit),
            GeometryScene(forbidden, simulator.config.affine, enforce_tip_in_bounds=False),
            simulator.config.access)
        if not checked.feasible:
            return checked
        unknowns.update(checked.unknowns)
        simulator.step(action_id)
        expected = before.copy()
        expected[tuple(occupied.T)] = False
        if not np.array_equal(expected, simulator.remaining_mask):
            return IndependentGeometryResult(False, ("removal_accounting_mismatch",))
    if not simulator.terminated:
        return IndependentGeometryResult(False, ("unterminated_sequence",))
    return IndependentGeometryResult(True, checker_version="independent-sequence-cell-capsule-v1",
                                     unknowns=tuple(sorted(unknowns)))
