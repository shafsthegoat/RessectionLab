"""Independent whole-tool events conditional on supplied functional evidence.

The tool sweep is intersected with physical source cells independently of the
planner. A coherent rigid world then resamples the supplied map at those cells'
centres. This is an explicit piecewise-constant, source-grid sensitivity model,
not continuous tract geometry, tissue mechanics or a clinical outcome model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from .evaluation import (CandidateFreezeManifest, EvaluationLedger,
                         IndependentGeometryResult, WorldOutcome,
                         evaluate_frozen_candidates, segment_box_distance_sq)
from .worlds import (FrozenDecisionModel, LatentWorld, WorldGenerator,
                     WorldPartitionManifest, content_hash)


EVENT_EVALUATOR_VERSION = "independent-functional-sweep-events-v1"
_COMPONENTS = ("motor", "language")


def _frozen(array: Any, dtype: Any) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class FunctionalEventConfig:
    """Thresholds describe released-map support, never clinical tolerances."""

    motor_threshold: float = .5
    language_threshold: float = .5
    cvar_alpha: float = .95
    version: str = "thresholded-supplied-map-contact-v1"

    def __post_init__(self) -> None:
        for name in ("motor_threshold", "language_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Map-support thresholds must lie in (0, 1]")
        if not np.isfinite(self.cvar_alpha) or not 0 <= self.cvar_alpha < 1:
            raise ValueError("CVaR alpha must lie in [0, 1)")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Event definition requires a version")

    def to_dict(self) -> dict[str, Any]:
        return {"motor_threshold": self.motor_threshold,
                "language_threshold": self.language_threshold,
                "cvar_alpha": self.cvar_alpha, "version": self.version}


@dataclass(frozen=True)
class AxialToolSweep:
    """Insertion/retraction union for one unchanged, complete rigid tool."""

    tool: Any
    tip_start_mm: tuple[float, float, float]
    tip_end_mm: tuple[float, float, float]
    axis_unit: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("tip_start_mm", "tip_end_mm", "axis_unit"):
            value = np.asarray(getattr(self, name), float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Tool sweep needs finite physical three-vectors")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        axis = np.asarray(self.axis_unit)
        delta = np.asarray(self.tip_end_mm) - self.tip_start_mm
        if not np.isclose(np.linalg.norm(axis), 1, atol=1e-9, rtol=0):
            raise ValueError("Tool sweep axis must be a unit vector")
        if np.linalg.norm(delta - (delta @ axis) * axis) > 1e-8:
            raise ValueError("Only axial unchanged-tool sweeps are supported")
        values = [getattr(self.tool, name) for name in
                  ("working_length_mm", "tip_length_mm", "shaft_radius_mm", "tip_radius_mm")]
        if not np.isfinite(values).all() or min(values) <= 0 or values[0] < values[1]:
            raise ValueError("Invalid full-tool dimensions")

    def capsules(self) -> tuple[tuple[np.ndarray, np.ndarray, float], ...]:
        axis = np.asarray(self.axis_unit)
        start, end = np.asarray(self.tip_start_mm), np.asarray(self.tip_end_mm)
        low, high = (start, end) if (end - start) @ axis >= 0 else (end, start)
        return ((low - self.tool.working_length_mm * axis,
                 high - self.tool.tip_length_mm * axis, float(self.tool.shaft_radius_mm)),
                (low - self.tool.tip_length_mm * axis, high, float(self.tool.tip_radius_mm)))


def sweeps_from_route(candidate: Any) -> tuple[AxialToolSweep, ...]:
    entry, target = np.asarray(candidate.entry_mm, float), np.asarray(candidate.target_mm, float)
    delta = target - entry
    if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) <= 1e-10:
        raise ValueError("Route must have distinct finite entry and target")
    return (AxialToolSweep(candidate.tool, tuple(entry), tuple(target),
                          tuple(delta / np.linalg.norm(delta))),)


def sweeps_from_native_history(history: Sequence[Mapping[str, Any]],
                               tools: Sequence[Any]) -> tuple[AxialToolSweep, ...]:
    """Decode poses only; callers must independently certify the full history.

    Reverse withdrawal follows the same envelope. Tool changes occur outside the
    anatomy. No removal is inferred from a route or from an exposure footprint.
    """
    catalog = {tool.tool_id: tool for tool in tools}
    if len(catalog) != len(tools):
        raise ValueError("Tool catalog has duplicate IDs")
    sweeps = []
    for record in history:
        if record.get("action_id") == "STOP":
            continue
        tool = catalog.get(record.get("tool_id"))
        if tool is None or not record.get("microsteps"):
            raise ValueError("Functional sequence requires native poses and a known tool")
        entry = np.asarray(record.get("entry_mm"), float)
        previous = entry
        for micro in record["microsteps"]:
            start, end = np.asarray(micro["tip_start_mm"]), np.asarray(micro["tip_end_mm"])
            if previous.shape != (3,) or not np.allclose(previous, start, atol=1e-8, rtol=0):
                raise ValueError("Native functional sequence has a discontinuous stroke")
            AxialToolSweep(tool, tuple(start), tuple(end), tuple(record["axis_unit"]))
            if (end - start) @ np.asarray(record["axis_unit"]) < -1e-8:
                raise ValueError("Native insertion reverses before the declared withdrawal")
            previous = end
        if not np.allclose(previous, record.get("tip_mm"), atol=1e-8, rtol=0):
            raise ValueError("Native macro endpoint differs from the microstep sequence")
        # The contiguous unchanged-axis microstep union is exactly the macro
        # sweep. Collapse it before regional cell queries; do not rasterize the
        # same shaft thousands of times.
        sweeps.append(AxialToolSweep(tool, tuple(entry), tuple(previous), tuple(record["axis_unit"])))
    return tuple(sweeps)


@dataclass(frozen=True)
class FunctionalExposureFootprint:
    """Unique source tissue cells touched anywhere by the whole tool sequence."""

    indices: np.ndarray
    source_shape: tuple[int, int, int]
    affine_ras_mm: np.ndarray
    tissue_support_hash: str
    sweep_count: int
    candidate_hash: str | None = None
    case_hash: str | None = None
    sweep_geometry_hash: str | None = None
    _array_state: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        shape = tuple(self.source_shape)
        if (len(shape) != 3 or min(shape) < 1 or indices.ndim != 2 or indices.shape[1] != 3
                or indices.dtype.kind not in "iu" or np.any(indices < 0)
                or np.any(indices >= shape) or len(np.unique(indices, axis=0)) != len(indices)):
            raise ValueError("Functional footprint must contain unique source cell indices")
        object.__setattr__(self, "indices", _frozen(indices, np.int64))
        object.__setattr__(self, "affine_ras_mm", _frozen(_grid(self.affine_ras_mm), float))
        object.__setattr__(self, "source_shape", shape)
        if not isinstance(self.sweep_count, int) or isinstance(self.sweep_count, bool) or self.sweep_count < 0:
            raise ValueError("Sweep count must be a nonnegative integer")
        object.__setattr__(self, "_array_state", self._layout())

    def _layout(self) -> tuple:
        return tuple((a.shape, a.dtype.str, a.strides, a.__array_interface__["data"][0])
                     for a in (self.indices, self.affine_ras_mm))

    def assert_intact(self) -> None:
        if self._layout() != self._array_state:
            raise ValueError("Functional footprint array metadata changed after construction")

    @property
    def fingerprint(self) -> str:
        self.assert_intact()
        return content_hash({"indices": self.indices, "source_shape": self.source_shape,
            "affine_ras_mm": self.affine_ras_mm, "tissue_support_hash": self.tissue_support_hash,
            "sweep_count": self.sweep_count, "sweep_geometry_hash": self.sweep_geometry_hash,
            "candidate_hash": self.candidate_hash, "case_hash": self.case_hash})

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine_ras_mm[:3, :3])))


def _grid(affine: Any) -> np.ndarray:
    matrix = np.asarray(affine, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Functional events require a finite physical affine")
    basis = matrix[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0) or not np.allclose(basis.T @ basis, np.diag(spacing**2), atol=1e-7):
        raise ValueError("Independent functional events require orthogonal source cells")
    return matrix


def prepare_functional_exposure(sweeps: Sequence[AxialToolSweep], *, tissue_mask: np.ndarray,
                                affine_ras_mm: np.ndarray,
                                candidate_hash: str | None = None, case_hash: str | None = None,
                                cancelled: Callable[[], bool] | None = None) -> FunctionalExposureFootprint:
    """Exact segment/box distance for each full shaft and active capsule.

    Intersection counts the complete touched cell volume as an exposure
    surrogate. It does not estimate the continuous intersection volume, nor does
    it claim the cell is removed. External shaft in declared air is excluded.
    """
    tissue = np.asarray(tissue_mask)
    if tissue.dtype != np.bool_ or tissue.ndim != 3 or min(tissue.shape) < 1:
        raise ValueError("Functional assessment requires explicit boolean tissue support")
    matrix = _grid(affine_ras_mm)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    rotation = matrix[:3, :3] / spacing
    touched: set[tuple[int, int, int]] = set()
    for sweep in sweeps:
        for start, end, radius in sweep.capsules():
            local_start = rotation.T @ (start - matrix[:3, 3])
            local_end = rotation.T @ (end - matrix[:3, 3])
            lower = np.maximum(np.floor((np.minimum(local_start, local_end) - radius) / spacing - .5).astype(int), 0)
            upper = np.minimum(np.ceil((np.maximum(local_start, local_end) + radius) / spacing + .5).astype(int), np.asarray(tissue.shape) - 1)
            if np.any(upper < lower):
                continue
            region = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper))
            for count, index in enumerate(np.argwhere(tissue[region]) + lower):
                if count % 256 == 0 and cancelled is not None and cancelled():
                    raise InterruptedError("Functional exposure evaluation cancelled")
                key = tuple(int(v) for v in index)
                if key in touched:
                    continue
                centre = index * spacing
                if segment_box_distance_sq(local_start, local_end, centre - spacing / 2,
                                           centre + spacing / 2) <= radius**2 + 1e-10:
                    touched.add(key)
    import hashlib
    support_hash = "sha256:" + hashlib.sha256(tissue.tobytes()).hexdigest()
    sweep_hash = content_hash([{ "start": sweep.tip_start_mm, "end": sweep.tip_end_mm,
        "axis": sweep.axis_unit, "tool": {name: getattr(sweep.tool, name) for name in
        ("tool_id", "working_length_mm", "tip_length_mm", "shaft_radius_mm", "tip_radius_mm")}}
        for sweep in sweeps])
    return FunctionalExposureFootprint(np.asarray(sorted(touched), dtype=np.int64).reshape(-1, 3),
                                       tissue.shape, matrix, support_hash, len(sweeps), candidate_hash, case_hash, sweep_hash)


def _sample_covered(values: np.ndarray, coverage: np.ndarray,
                    coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear value sampling with every nonzero interpolation contributor known."""
    shape = np.asarray(values.shape)
    coordinates = np.array(coordinates, dtype=np.float64, copy=True)
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(shape, 1)
    coordinates = np.where(np.abs(coordinates) <= tolerance, 0, coordinates)
    coordinates = np.where(np.abs(coordinates - (shape - 1)) <= tolerance, shape - 1, coordinates)
    inside = np.all((coordinates >= 0) & (coordinates <= shape - 1), axis=1)
    clipped = np.clip(coordinates, 0, shape - 1)
    base = np.floor(clipped).astype(int)
    fraction = clipped - base
    known = inside.copy()
    for corner in product((0, 1), repeat=3):
        corner = np.asarray(corner)
        index = np.minimum(base + corner, shape - 1)
        weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        known &= (weight == 0) | coverage[tuple(index.T)]
    sampled = map_coordinates(values, coordinates.T, order=1, mode="constant", cval=0., prefilter=False)
    sampled[~known] = np.nan
    return sampled, known


class FunctionalScenarioEvaluator:
    """Shared-world event evaluator; raw missing evidence never becomes zero."""

    def __init__(self, evidence: Any, config: FunctionalEventConfig = FunctionalEventConfig()):
        self.evidence = evidence
        self.config = config
        self.evidence_hash = evidence.fingerprint
        self.affine = _grid(evidence.affine_ras_mm)
        self.inverse_affine = np.linalg.inv(self.affine)
        for component in _COMPONENTS:
            values, coverage = getattr(evidence, component), getattr(evidence, component + "_coverage")
            if (values is None) != (coverage is None):
                raise ValueError("Functional field and its coverage must be jointly present")
            if values is not None and (values.ndim != 3 or values.shape != coverage.shape
                    or coverage.dtype != np.bool_ or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("Functional values need aligned coverage and finite unitless [0,1] support")

    @property
    def event_definitions(self) -> dict[str, str]:
        return {component + "_supplied_map_contact":
                (f"Any native tissue cell touched by the complete shaft or active tool sweep has "
                 f"covered, coherently resampled supplied {component} map support >= "
                 f"{getattr(self.config, component + '_threshold'):g}. Known hit is true; no hit "
                 "with incomplete contacted-cell coverage is unknown. This is a supplied-map "
                 "encounter, not individual function or postoperative deficit.")
                for component in _COMPONENTS}

    @property
    def cost_units(self) -> dict[str, str]:
        return {component + suffix: units for component in _COMPONENTS for suffix, units in (
            ("_contact_surrogate", "unitless supplied-map support × contacted-cell mm3; not removed volume"),
            ("_known_contact_lower_bound", "known supplied-map support × contacted-cell mm3; incomplete-model lower bound"),
            ("_unassessed_contact_volume", "mm3 of touched source cells lacking supplied-map coverage"))}

    def world_outcome(self, footprint: FunctionalExposureFootprint, world: LatentWorld) -> WorldOutcome:
        footprint.assert_intact()
        self.evidence.assert_intact()
        points = footprint.indices @ footprint.affine_ras_mm[:3, :3].T + footprint.affine_ras_mm[:3, 3]
        transform = self.inverse_affine @ np.linalg.inv(world.anatomy_transform_mm)
        coordinates = points @ transform[:3, :3].T + transform[:3, 3]
        events, costs, unknowns = {}, {}, []
        for component in _COMPONENTS:
            values = getattr(self.evidence, component)
            if values is None:
                events[component + "_supplied_map_contact"] = None if len(points) else False
                costs[component + "_contact_surrogate"] = None if len(points) else 0.
                costs[component + "_known_contact_lower_bound"] = None if len(points) else 0.
                costs[component + "_unassessed_contact_volume"] = len(points) * footprint.voxel_volume_mm3
                unknowns.append(component + "_evidence_unavailable")
                continue
            sampled, known = _sample_covered(values, getattr(self.evidence, component + "_coverage"), coordinates)
            hit = bool(np.any(sampled[known] >= getattr(self.config, component + "_threshold")))
            events[component + "_supplied_map_contact"] = True if hit else (False if known.all() else None)
            lower = float(sampled[known].sum() * footprint.voxel_volume_mm3)
            costs[component + "_contact_surrogate"] = lower if known.all() else None
            costs[component + "_known_contact_lower_bound"] = lower
            costs[component + "_unassessed_contact_volume"] = int((~known).sum()) * footprint.voxel_volume_mm3
            if not known.all():
                unknowns.append(component + "_contacted_map_coverage_incomplete")
        unknowns.append("patient_specific_function_and_clinical_outcomes_unassessed")
        return WorldOutcome(events, costs, unknowns=tuple(unknowns))


def evaluate_functional_candidates(
    candidates: Sequence[Any], freeze: CandidateFreezeManifest,
    decision_model: FrozenDecisionModel, partition: WorldPartitionManifest, *,
    evidence: Any, footprints: Mapping[str, FunctionalExposureFootprint],
    geometry_checker: Callable[[Any], IndependentGeometryResult],
    ledger: EvaluationLedger, config: FunctionalEventConfig = FunctionalEventConfig(),
) -> dict[str, Any]:
    """Add concrete evidence-bound events to the existing frozen evaluator.

    The decision model's geometry must bind ``functional_evidence_hash`` and
    ``functional_event_config`` before candidate freeze. Footprint construction
    and independent geometry checking are separate; exposure is never removal.
    """
    import json
    # Callbacks must not redirect evaluation to a different footprint after the
    # frozen-content check. Values are immutable and guard array interpretation.
    footprints = dict(footprints)
    geometry = json.loads(decision_model.configuration_json)["geometry"]
    if (not isinstance(geometry, dict) or geometry.get("functional_evidence_hash") != evidence.fingerprint
            or geometry.get("functional_event_config") != config.to_dict()):
        raise ValueError("Frozen decision model must bind exact functional evidence and event definitions")
    if set(footprints) != {candidate.plan_id for candidate in candidates}:
        raise ValueError("Every frozen candidate requires exactly one exposure footprint")
    frozen_generator = json.loads(decision_model.configuration_json)["world_generator"]
    if content_hash(frozen_generator) != content_hash(evidence.uncertainty.to_dict()):
        raise ValueError("Frozen generator differs from evidence uncertainty")
    for candidate in candidates:
        footprint = footprints[candidate.plan_id]
        if geometry.get("functional_footprint_hashes", {}).get(candidate.plan_id) != footprint.fingerprint:
            raise ValueError("Functional footprint content differs from frozen geometry")
        if footprint.candidate_hash != candidate.semantic_hash or footprint.case_hash != candidate.case_hash:
            raise ValueError("Exposure footprint belongs to another candidate or case")
        if geometry.get("tissue_support_hash") != footprint.tissue_support_hash:
            raise ValueError("Exposure tissue support differs from frozen anatomy")
        if (not np.allclose(footprint.affine_ras_mm, evidence.affine_ras_mm, atol=1e-8, rtol=0)
                or any(a is not None and a.shape != footprint.source_shape for a in (evidence.motor, evidence.language))):
            raise ValueError("Exposure and functional evidence must share the declared source grid")
    evaluator = FunctionalScenarioEvaluator(evidence, config)
    report = evaluate_frozen_candidates(candidates, freeze, decision_model, partition,
        event_definitions=evaluator.event_definitions, cost_units=evaluator.cost_units,
        world_evaluator=lambda candidate, world: evaluator.world_outcome(footprints[candidate.plan_id], world),
        geometry_checker=geometry_checker, ledger=ledger, cvar_alpha=config.cvar_alpha)
    generator = WorldGenerator(partition.generator)
    transforms = [generator.sample(partition, i).anatomy_transform_mm for i in range(len(partition.seeds))]
    # Numerically identical +/-0 matrices are the same anatomy. Byte hashing
    # would mislabel different zero-perturbation seeds as distinct scenarios.
    distinct = len({tuple(matrix.ravel()) for matrix in transforms})
    report.update(functional_evaluator_version=EVENT_EVALUATOR_VERSION,
                  functional_evidence=evidence.to_manifest(), event_configuration=config.to_dict(),
                  distinct_anatomy_transforms=distinct,
                  nonidentical_worlds_verified=distinct > 1,
                  robustness_interpretation=("nonidentical_uncalibrated_registration_sensitivity"
                    if distinct > 1 else "identical_anatomy_replays_do_not_test_robustness"),
                  sampling_model="linear map values at native tissue cell centres assigned to complete source cells",
                  exposure_model="union of source tissue cells intersecting complete active and shaft swept capsules",
                  exposure_volume_interpretation="whole touched-cell surrogate, not exact intersection or removed tissue",
                  patient_function_assessed=False,
                  clinical_deficit_probability=None)
    for row in report["candidates"]:
        footprint = footprints[row["plan_id"]]
        row["exposure"] = {"touched_source_cell_count": len(footprint.indices),
            "contacted_cell_volume_mm3": len(footprint.indices) * footprint.voxel_volume_mm3,
            "sweep_count": footprint.sweep_count, "tissue_support_hash": footprint.tissue_support_hash,
            "source_shape": list(footprint.source_shape), "affine_ras_mm": footprint.affine_ras_mm.tolist(),
            "footprint_hash": footprint.fingerprint, "sweep_geometry_hash": footprint.sweep_geometry_hash}
    if evidence.fingerprint != evaluator.evidence_hash:
        raise ValueError("Functional evidence changed during independent evaluation")
    return report
