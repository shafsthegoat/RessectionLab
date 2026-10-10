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
                    radius: float, *, _batch_size: int | None = None,
                    _cancelled: Callable[[], bool] | None = None) -> tuple[float, float, float] | None:
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
    cell_indices = np.argwhere(mask[slices]) + lo
    if _batch_size is not None:
        for hits in _batch_cell_contacts(a_local, b_local, cell_indices, spacing, radius,
                                        1e-10, _batch_size, _cancelled, first_only=True):
            point = affine[:3, :3] @ hits[0] + affine[:3, 3]
            return tuple(float(v) for v in point)
        return None
    for index in cell_indices:
        center = index * spacing
        if segment_box_distance_sq(a_local, b_local, center - spacing / 2, center + spacing / 2) <= radius * radius + 1e-10:
            point = affine[:3, :3] @ index + affine[:3, 3]
            return tuple(float(v) for v in point)
    return None


def _batch_cell_contacts(start: np.ndarray, end: np.ndarray, cell_indices: np.ndarray,
                         spacing: np.ndarray, radius: float, tolerance_sq: float,
                         batch_size: int, cancelled: Callable[[], bool] | None, *,
                         first_only: bool = False):
    """Bound box construction, retaining the caller's current occupancy/order.

    Voxel enumeration is unchanged from the scalar checker. No distance, scene,
    contact, or feasibility result is retained between calls or microsteps.
    """
    from .independent_geometry_batch import (
        IndependentBatchCancelled, first_segment_box_contact, segment_box_contact_indices,
    )

    if cancelled is not None and cancelled():
        raise IndependentBatchCancelled("Independent geometry batch cancelled")
    for offset in range(0, len(cell_indices), batch_size):
        cells = cell_indices[offset:offset + batch_size]
        centers = cells * spacing
        lower, upper = centers - spacing / 2, centers + spacing / 2
        if first_only:
            row = first_segment_box_contact(start, end, lower, upper, radius,
                tolerance_sq=tolerance_sq, batch_size=batch_size, cancelled=cancelled)
            if row is not None:
                yield cells[row:row + 1]
                return
        else:
            rows = segment_box_contact_indices(start, end, lower, upper, radius,
                tolerance_sq=tolerance_sq, batch_size=batch_size, cancelled=cancelled)
            if len(rows):
                yield cells[rows]


def _native_active_contacts(remaining: np.ndarray, rotation: np.ndarray,
                            spacing: np.ndarray, origin: np.ndarray,
                            start: np.ndarray, end: np.ndarray, radius: float, *,
                            distance_backend: str, distance_batch_size: int,
                            cancelled: Callable[[], bool] | None) -> set[tuple[int, int, int]]:
    """Exact active-contact set against the current pre-removal source mask."""
    start_local = rotation.T @ (start - origin)
    end_local = rotation.T @ (end - origin)
    low = np.maximum(np.floor((np.minimum(start_local, end_local) - radius) / spacing - .5).astype(int), 0)
    high = np.minimum(np.ceil((np.maximum(start_local, end_local) + radius) / spacing + .5).astype(int), np.asarray(remaining.shape) - 1)
    result = set()
    if np.any(high < low):
        return result
    region = tuple(slice(int(a), int(b) + 1) for a, b in zip(low, high))
    cell_indices = np.argwhere(remaining[region]) + low
    if distance_backend == "batch":
        for hits in _batch_cell_contacts(start_local, end_local, cell_indices, spacing,
                                        radius, 1e-9, distance_batch_size, cancelled):
            result.update(tuple(int(v) for v in index) for index in hits)
        return result
    for index in cell_indices:
        center = index * spacing
        if segment_box_distance_sq(start_local, end_local, center - spacing / 2, center + spacing / 2) <= radius**2 + 1e-9:
            result.add(tuple(int(v) for v in index))
    return result


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
        if action_id == "STOP":
            if position != len(actions) - 1:
                return IndependentGeometryResult(False, ("action_after_stop",))
            simulator.step("STOP")
            continue
        if simulator.terminated:
            return IndependentGeometryResult(False, ("action_after_termination",))
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


@dataclass(frozen=True)
class NativeRemovalAudit:
    feasible: bool
    failures: tuple[str, ...]
    first_failed_action: str | None
    first_unsupported_source_voxel: tuple[int, int, int] | None
    first_unsupported_position_mm: tuple[float, float, float] | None
    claimed_source_tissue_volume_mm3: float
    contained_source_tissue_volume_mm3: float
    unsupported_source_tissue_volume_mm3: float
    source_case_hash: str
    source_voxel_volume_mm3: float
    action_count: int
    checker_version: str = "independent-native-footprint-v1"
    interpretation: str = "All eight source-cell corners must lie within the modeled active-tip capsule; tissue mechanics unvalidated"
    complete_tool_checked: bool = False
    frontier_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def independent_native_removal_check(case: Any, config: Any,
                                     history: Sequence[Mapping[str, Any]]) -> NativeRemovalAudit:
    """Audit source-space tissue allegedly erased by coarse terminal contact.

    A coarse voxel touching a tip does not establish that its underlying source
    cells were removed. This check maps the exact declared block derivation back
    to original source cells and requires full-cell containment in the active
    capsule, a sufficient condition under the declared geometric removal model.
    Negative results preserve the first failing source location and union volumes.
    This gate does not validate tissue forces or turn a passing footprint into a
    clinical procedure; native complete-tool replay remains a separate check.
    """
    from itertools import product
    from scipy.ndimage import binary_fill_holes

    if config.source_hash != case.semantic_hash:
        raise ValueError("Native removal audit source differs from simulation provenance")
    derivation = config.derivation
    block = derivation.get("block_size_native_voxels")
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        raise ValueError("Native audit requires an explicit integer source-block derivation")
    if tuple(derivation.get("native_shape", ())) != tuple(case.mri.shape):
        raise ValueError("Native audit source grid does not match recorded derivation")
    expected = np.array(case.affine, float)
    expected[:3, 3] += expected[:3, :3] @ np.full(3, (block - 1) / 2)
    expected[:3, :3] *= block
    if not np.allclose(expected, config.affine, atol=1e-7, rtol=0):
        raise ValueError("Coarse-to-source affine mapping is not the declared block derivation")
    targets = np.zeros(case.mri.shape, bool)
    for mask in case.compartments.values():
        targets |= mask
    if case.brain_mask is not None:
        source_tissue = np.asarray(case.brain_mask, bool) | targets
    elif derivation.get("tissue_envelope_source") == "hole_filled_nonzero_MRI_support_unreviewed_skull_strip_assumption":
        source_tissue = binary_fill_holes(np.asarray(case.mri) != 0) | targets
    else:
        raise ValueError("Native audit cannot reconstruct the declared tissue envelope")
    corner_offsets_mm = np.array(list(product((-.5, .5), repeat=3))) @ np.asarray(case.affine)[:3, :3].T
    claimed: set[tuple[int, int, int]] = set()
    unsupported: set[tuple[int, int, int]] = set()
    first_action = None
    first_voxel = None
    first_position = None
    failure_reasons = set()
    action_count = 0
    for record in history:
        if record.get("action_id") == "STOP":
            continue
        action_count += 1
        action_id = str(record.get("action_id", ""))
        tool_id = record.get("tool_id") or (action_id.split(":", 2)[1] if action_id.startswith("REMOVE:") else None)
        tool = next((tool for tool in config.tools if tool.tool_id == tool_id), None)
        if tool is None:
            raise ValueError("Native removal record does not identify a configured tool")
        tip = np.asarray(record.get("tip_mm"), float)
        axis = np.asarray(record.get("axis_unit"), float)
        if tip.shape != (3,) or axis.shape != (3,) or not np.isfinite(tip).all() or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1, atol=1e-7):
            raise ValueError("Native removal record needs a finite physical tip and unit axis")
        footprint = np.asarray(record.get("removed_indices", ()))
        if footprint.ndim != 2 or footprint.shape[1] != 3 or footprint.dtype.kind not in "iu" or np.any(footprint < 0) or np.any(footprint >= np.array(config.tissue_mask.shape)):
            raise ValueError("Native removal record has invalid coarse cell indices")
        for coarse_index in footprint:
            low = coarse_index * block
            high = np.minimum(low + block, source_tissue.shape)
            indices = np.stack(np.meshgrid(*(np.arange(lo, hi) for lo, hi in zip(low, high)), indexing="ij"), axis=-1).reshape(-1, 3)
            indices = indices[source_tissue[tuple(indices.T)]]
            if not len(indices):
                continue
            centers = indices @ np.asarray(case.affine)[:3, :3].T + np.asarray(case.affine)[:3, 3]
            corners = centers[:, None, :] + corner_offsets_mm[None, :, :]
            proximal_tip = tip - float(tool.tip_length_mm) * axis
            segment = tip - proximal_tip
            fractions = np.clip(np.sum((corners - proximal_tip) * segment, axis=-1) / np.dot(segment, segment), 0, 1)
            nearest = proximal_tip + fractions[..., None] * segment
            contained = np.all(np.sum((corners - nearest)**2, axis=-1) <= float(tool.tip_radius_mm)**2 + 1e-10, axis=1)
            for index, center, supported in zip(indices, centers, contained):
                key = tuple(int(v) for v in index)
                if key in claimed:
                    failure_reasons.add("repeated_source_tissue_removal")
                claimed.add(key)
                if not supported:
                    unsupported.add(key)
                    failure_reasons.add("unsupported_coarse_removal_footprint")
                    if first_voxel is None:
                        first_action, first_voxel = action_id, key
                        first_position = tuple(float(v) for v in center)
    volume = float(abs(np.linalg.det(np.asarray(case.affine)[:3, :3])))
    return NativeRemovalAudit(not failure_reasons, tuple(sorted(failure_reasons)), first_action,
                              first_voxel, first_position, len(claimed) * volume,
                              (len(claimed) - len(unsupported)) * volume, len(unsupported) * volume,
                              case.semantic_hash, volume, action_count)


def _extend_independent_free_space(remaining: np.ndarray, connected: np.ndarray,
                                   newly_removed: set[tuple[int, int, int]], *,
                                   full_flood_threshold: int = 4096,
                                   interaction_domain: np.ndarray | None = None) -> None:
    """Update the established exterior component after certified connected cuts.

    Only a removed cell and a previously sealed air cavity it opens can newly join
    the exterior component. Explore those cells, leaving the large unchanged air
    volume untouched. A large opened cavity falls back to the same full six-face
    propagation used for initialization. This changes work, not connectivity.
    The caller has already independently verified that the removed cluster is
    connected to the existing exterior/cavity and has applied the removal.
    """
    from collections import deque
    from scipy.ndimage import binary_propagation, generate_binary_structure

    if not newly_removed:
        return
    queue = deque(sorted(newly_removed))
    for key in newly_removed:
        if interaction_domain is not None and not interaction_domain[key]:
            raise ValueError("Unknown domain cannot become connected free space")
        if remaining[key]:
            raise ValueError("Connected-free update precedes committed removal")
        connected[key] = True
    expanded = 0
    while queue:
        key = queue.popleft()
        expanded += 1
        if expanded > full_flood_threshold:
            free_mask = ~remaining if interaction_domain is None else ~remaining & interaction_domain
            connected[:] = binary_propagation(connected, structure=generate_binary_structure(3, 1),
                                               mask=free_mask)
            return
        for axis in range(3):
            for direction in (-1, 1):
                coordinate = key[axis] + direction
                if coordinate < 0 or coordinate >= remaining.shape[axis]:
                    continue
                neighbor = list(key)
                neighbor[axis] = coordinate
                neighbor = tuple(neighbor)
                if (not remaining[neighbor] and not connected[neighbor]
                        and (interaction_domain is None or interaction_domain[neighbor])):
                    connected[neighbor] = True
                    queue.append(neighbor)


def independent_check_native_history(case: Any, tools: Sequence[Any],
                                     history: Sequence[Mapping[str, Any]], *,
                                     tissue_mask: np.ndarray, access: Any,
                                     hard_exclusion: np.ndarray | None = None,
                                     interaction_domain: np.ndarray | None = None,
                                     geometry_frame: str = "RAS+",
                                     cancelled: Callable[[], bool] | None = None,
                                     distance_backend: str = "scalar",
                                     distance_batch_size: int = 256,
                                     tool_modes: Mapping[str, str] | None = None) -> NativeRemovalAudit:
    """Independent source-cell audit of the native contained-cell cutting model.

    All tissue removed in a microstep must be fully inside its active capsule and
    connected to exterior/cavity by six-face adjacency. The swept shaft must avoid
    tissue remaining before each microstep, preventing it from borrowing removal
    that occurs only at the step endpoint. The whole tool avoids hard exclusions. Active
    contact with partially contained cells is the declared cutting abstraction:
    those cells must be recorded as contacted and remain occupied. This distinction
    is explicit and does not assert quantitatively validated tissue mechanics.
    Tool poses and access use ``geometry_frame`` (RAS+ by default); source cells
    are never resampled. Each recorded entry is checked on the same access plane
    and the complete tool must fit the aperture throughout its stroke.

    ``distance_backend='batch'`` explicitly opts into bounded distance batches
    for active contacts and the pre-removal shaft scan only. Scalar remains the
    default and near-threshold oracle; hard-exclusion and general motion checks
    remain scalar. The backend and batch size belong in experiment metadata;
    scientific certificate fields are unchanged. No occupancy is cached.
    """
    from collections import deque
    from itertools import product
    from scipy.ndimage import binary_propagation, generate_binary_structure
    from types import SimpleNamespace
    from .geometry import ToolPose
    from .independent_geometry_batch import IndependentBatchCancelled, MAX_BATCH_SIZE

    if hasattr(case, "critical_evidence"):
        from .critical_evidence import canonical_hard_exclusion
        hard_exclusion, _ = canonical_hard_exclusion(case, hard_exclusion)

    if not isinstance(distance_backend, str) or distance_backend not in {"scalar", "batch"}:
        raise ValueError("Native distance backend must be 'scalar' or 'batch'")
    if isinstance(distance_batch_size, (bool, np.bool_)) or not isinstance(distance_batch_size, (int, np.integer)) or not 1 <= distance_batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"Native distance_batch_size must be an integer in [1, {MAX_BATCH_SIZE}]")
    distance_batch_size = int(distance_batch_size)

    original = np.asarray(tissue_mask)
    if original.shape != case.mri.shape or original.dtype != np.bool_:
        raise ValueError("Native tissue support must be an explicit boolean mask on the source grid")
    hard = np.zeros_like(original) if hard_exclusion is None else np.asarray(hard_exclusion)
    if hard.shape != original.shape or hard.dtype != np.bool_:
        raise ValueError("Native hard exclusions must be boolean and source-aligned")
    if interaction_domain is not None:
        interaction_domain = np.asarray(interaction_domain)
        if (interaction_domain.shape != original.shape or interaction_domain.dtype != np.bool_
                or np.any(original & ~interaction_domain)):
            raise ValueError("Native interaction domain must contain all modeled tissue on the source grid")
        hard = hard | ~interaction_domain
    remaining = original.copy()
    if geometry_frame not in {"RAS+", "LPS+"} or case.frame not in {"RAS+", "LPS+"}:
        raise ValueError("Independent native geometry needs an explicit RAS+ or LPS+ frame")
    matrix = np.asarray(case.affine, float)
    if case.frame != geometry_frame:
        matrix = np.diag([-1., -1., 1., 1.]) @ matrix
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.diag(spacing**2), atol=1e-7):
        raise ValueError("Independent native tool check does not support sheared source cells")
    rotation = matrix[:3, :3] / spacing
    corners = np.array(list(product((-.5, .5), repeat=3))) @ matrix[:3, :3].T
    catalog = {tool.tool_id: tool for tool in tools}
    if len(catalog) != len(tools):
        raise ValueError("Native tool IDs must be unique")
    if tool_modes is not None:
        tool_modes = dict(tool_modes)
        if set(tool_modes) != set(catalog) or any(mode not in {"aspirate", "probe"} for mode in tool_modes.values()):
            raise ValueError("Independent interaction registry must bind every frozen tool")
    connectivity = generate_binary_structure(3, 1)
    hard_scene = SimpleNamespace(forbidden_mask=hard, affine=matrix,
                                 sphere_obstacles=(), enforce_tip_in_bounds=False)
    border = np.zeros_like(remaining)
    for axis in range(3):
        selector = [slice(None)] * 3
        selector[axis] = 0
        border[tuple(selector)] = True
        selector[axis] = -1
        border[tuple(selector)] = True
    free_mask = ~remaining if interaction_domain is None else ~remaining & interaction_domain
    free = binary_propagation(border & free_mask, structure=connectivity, mask=free_mask)
    declared: set[tuple[int, int, int]] = set()
    accepted: set[tuple[int, int, int]] = set()
    volume = float(abs(np.linalg.det(matrix[:3, :3])))
    count = 0

    def report(reason: str | None = None, action: str | None = None,
               voxel: tuple[int, int, int] | None = None) -> NativeRemovalAudit:
        physical = None if voxel is None else tuple(float(v) for v in matrix[:3, :3] @ voxel + matrix[:3, 3])
        return NativeRemovalAudit(reason is None, () if reason is None else (reason,), action, voxel, physical,
            len(declared) * volume, len(accepted) * volume, len(declared - accepted) * volume,
            case.semantic_hash, volume, count, checker_version="independent-native-sequence-v2",
            interpretation=f"Source-grid prefix audit in {geometry_frame}: explicit entries within one aperture, contained connected cell removal, prior-tissue swept-shaft clearance, full-tool hard-exclusion clearance, and recorded partial active contact; tissue mechanics unvalidated",
            complete_tool_checked=reason is None, frontier_checked=reason is None)

    def indices(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.size == 0:
            return np.empty((0, 3), dtype=int)
        if array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array >= remaining.shape):
            raise ValueError("Invalid source-cell indices in native microstep")
        if len(np.unique(array, axis=0)) != len(array):
            raise ValueError("Duplicate source-cell indices in native microstep")
        return array

    for record in history:
        if record.get("action_id") == "STOP":
            continue
        count += 1
        action_id = str(record.get("action_id", f"native-action-{count}"))
        if record.get("source_hash") != case.semantic_hash or tuple(record.get("source_shape", ())) != remaining.shape or not np.allclose(record.get("native_affine"), matrix):
            raise ValueError("Native history source identity or affine mismatch")
        if record.get("native_footprint") != "fully_contained_connected_cells_v1":
            return report("unsupported_native_footprint_model", action_id)
        tool = catalog.get(record.get("tool_id"))
        if tool is None:
            return report("unknown_native_tool_configuration", action_id)
        length, tip_length, shaft_radius, tip_radius = _tool_dimensions(tool)
        axis = np.asarray(record.get("axis_unit"), float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1, atol=1e-7):
            raise ValueError("Native history needs a unit tool axis")
        previous = np.asarray(record.get("entry_mm", access.center_mm), float)
        if previous.shape != (3,) or not np.isfinite(previous).all():
            raise ValueError("Native stroke entry must be a finite physical three-vector")
        normal = np.asarray(access.normal_inward, float)
        normal = normal / np.linalg.norm(normal)
        if abs(float((previous - access.center_mm) @ normal)) > 1e-7:
            return report("native_entry_outside_access_plane", action_id)
        mode = record.get("interaction_mode", "aspirate")
        if mode not in {"aspirate", "probe"}:
            return report("unsupported_native_interaction_mode", action_id)
        if tool_modes is not None and mode != tool_modes[tool.tool_id]:
            return report("native_interaction_differs_from_frozen_tool_registry", action_id)
        macro_contacts, probe_contacts = set(), set()
        macro_removed: set[tuple[int, int, int]] = set()
        microsteps = record.get("microsteps", ())
        if not microsteps:
            return report("native_action_missing_microsteps", action_id)
        for micro in microsteps:
            if cancelled is not None and cancelled():
                return report("independent_validation_cancelled", action_id)
            tip_start = np.asarray(micro.get("tip_start_mm"), float)
            tip_end = np.asarray(micro.get("tip_end_mm"), float)
            start = np.asarray(micro.get("active_stroke_start_mm"), float)
            end = np.asarray(micro.get("active_stroke_end_mm"), float)
            if any(v.shape != (3,) or not np.isfinite(v).all() for v in (tip_start, tip_end, start, end)):
                raise ValueError("Native microsteps require finite physical coordinates")
            displacement = tip_end - tip_start
            if not np.allclose(previous, tip_start, atol=1e-7) or np.linalg.norm(displacement - (displacement @ axis) * axis) > 1e-7 or displacement @ axis < -1e-7:
                return report("noncontiguous_or_nonaxial_native_stroke", action_id)
            if not np.allclose(start, tip_start - tip_length * axis, atol=1e-7) or not np.allclose(end, tip_end, atol=1e-7) or not np.isclose(micro.get("active_radius_mm", -1), tip_radius, atol=1e-9):
                return report("native_active_envelope_differs_from_frozen_tool", action_id)
            removed = indices(micro.get("removed_indices_native", ()))
            keys = {tuple(int(v) for v in index) for index in removed}
            declared.update(keys)
            if mode == "probe" and keys:
                return report("probe_must_not_remove_native_tissue", action_id)
            if any(not remaining[key] for key in keys):
                return report("repeated_or_non_tissue_native_removal", action_id, next(key for key in keys if not remaining[key]))
            if len(removed):
                vertices = removed @ matrix[:3, :3].T + matrix[:3, 3]
                vertices = vertices[:, None, :] + corners[None, :, :]
                segment = end - start
                fractions = np.clip(np.sum((vertices - start) * segment, axis=-1) / np.dot(segment, segment), 0, 1)
                contained = np.all(np.sum((vertices - (start + fractions[..., None] * segment))**2, axis=-1) <= tip_radius**2 + 1e-9, axis=1)
                if not contained.all():
                    return report("native_removed_cell_not_fully_contained", action_id, tuple(int(v) for v in removed[np.flatnonzero(~contained)[0]]))
                # Reachability is checked by an independent explicit graph walk.
                queue = deque()
                reached = set()
                for key in keys:
                    for dimension in range(3):
                        for sign in (-1, 1):
                            neighbor = list(key)
                            neighbor[dimension] += sign
                            if any(v < 0 or v >= n for v, n in zip(neighbor, remaining.shape)) or free[tuple(neighbor)]:
                                reached.add(key)
                                queue.append(key)
                                break
                        if key in reached:
                            break
                while queue:
                    key = queue.popleft()
                    for dimension in range(3):
                        for sign in (-1, 1):
                            neighbor = list(key)
                            neighbor[dimension] += sign
                            neighbor = tuple(neighbor)
                            if neighbor in keys and neighbor not in reached:
                                reached.add(neighbor)
                                queue.append(neighbor)
                if reached != keys:
                    return report("disconnected_native_removal", action_id, next(iter(keys - reached)))
            contacts = {tuple(int(v) for v in index) for index in indices(micro.get("contact_indices_native", ()))}
            macro_contacts.update(contacts)
            if mode == "probe":
                expected_contacts = _native_active_contacts(original, rotation, spacing, matrix[:3, 3],
                    start, end, tip_radius, distance_backend=distance_backend,
                    distance_batch_size=distance_batch_size, cancelled=cancelled)
                if contacts != expected_contacts:
                    return report("probe_contact_record_differs_from_independent_geometry", action_id)
                occupied_contacts = {key for key in contacts if remaining[key]}
                probe_contacts.update(occupied_contacts)
                occupied_scene = SimpleNamespace(forbidden_mask=remaining, affine=matrix,
                    sphere_obstacles=(), enforce_tip_in_bounds=False)
                if _cell_collision(occupied_scene, start, end, max(0., tip_radius - 1e-8)) is not None:
                    return report("probe_active_region_penetrates_remaining_tissue", action_id)
                for key in occupied_contacts:
                    exposed = False
                    for dimension in range(3):
                        for sign in (-1, 1):
                            neighbor = list(key)
                            neighbor[dimension] += sign
                            if any(v < 0 or v >= n for v, n in zip(neighbor, remaining.shape)) or free[tuple(neighbor)]:
                                exposed = True
                    if not exposed:
                        return report("probe_contact_not_exposed", action_id, key)
            try:
                omitted = _native_active_contacts(remaining, rotation, spacing, matrix[:3, 3],
                    start, end, tip_radius, distance_backend=distance_backend,
                    distance_batch_size=distance_batch_size, cancelled=cancelled) - contacts - keys
            except IndependentBatchCancelled:
                return report("independent_validation_cancelled", action_id)
            if omitted:
                return report("unrecorded_partial_active_tissue_contact", action_id, next(iter(omitted)))
            certificate = independent_check_motion(tool, ToolPose(tip_start, axis), ToolPose(tip_end, axis),
                hard_scene, access)
            if not certificate.feasible:
                return report("native_full_tool_hard_constraint_failure", action_id)
            # The shaft cannot use tissue clearance produced only at this
            # microstep's endpoint. Testing ``after`` would permit temporal
            # borrowing, which an optimizer could exploit by using long steps.
            tissue_scene = SimpleNamespace(forbidden_mask=remaining, affine=matrix,
                                           sphere_obstacles=(), enforce_tip_in_bounds=False)
            try:
                if distance_backend == "batch":
                    shaft_collision = _cell_collision(tissue_scene,
                        tip_start - length * axis, tip_end - tip_length * axis, shaft_radius,
                        _batch_size=distance_batch_size, _cancelled=cancelled)
                else:
                    shaft_collision = _cell_collision(tissue_scene,
                        tip_start - length * axis, tip_end - tip_length * axis, shaft_radius)
            except IndependentBatchCancelled:
                return report("independent_validation_cancelled", action_id)
            if shaft_collision is not None:
                return report("native_shaft_collides_with_remaining_tissue", action_id)
            if keys:
                remaining[tuple(removed.T)] = False
            accepted.update(keys)
            macro_removed.update(keys)
            _extend_independent_free_space(remaining, free, keys, interaction_domain=interaction_domain)
            previous = tip_end
        if mode == "probe":
            if not probe_contacts:
                return report("probe_missing_exposed_contact", action_id)
            declared_contacts = {tuple(int(v) for v in index) for index in indices(record.get("contact_indices_native", ()))}
            if declared_contacts != macro_contacts:
                return report("probe_macro_contact_accounting_mismatch", action_id)
            if "probe_contact_indices_native" in record:
                declared_probe = {tuple(int(v) for v in index) for index in indices(record["probe_contact_indices_native"])}
                if declared_probe != probe_contacts:
                    return report("probe_committed_contact_accounting_mismatch", action_id)
        declared_macro = {tuple(int(v) for v in index) for index in indices(record.get("removed_indices_native", ()))}
        if declared_macro != macro_removed:
            return report("native_macro_removal_accounting_mismatch", action_id)
        if "tip_mm" in record and not np.allclose(record["tip_mm"], previous, atol=1e-7, rtol=0):
            return report("native_macro_tip_mismatch", action_id)
        if "removed_volume_mm3" in record and not np.isclose(record["removed_volume_mm3"], len(macro_removed) * volume, atol=1e-7, rtol=0):
            return report("native_macro_volume_accounting_mismatch", action_id)
    return report()
