"""Deterministic straight-access search, with explicit geometric limitations.

The candidate set is an enumeration of fixed rigid-tool approaches. It is not
an execution of resection: accessible volumes count target voxel centres in
the swept active-tip tube, conditional on establishing the stated access.
Normal-tissue exposure is retained and no tissue array is cleared or removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from .geometry import (
    GENERIC_TOOLS,
    GEOMETRY_VERSION,
    AccessWindow,
    GeometryScene,
    ToolGeometry,
    ToolPose,
    check_motion,
)

if TYPE_CHECKING:
    from .core import CaseData, Plan


SEARCH_VERSION = "straight-access-search-v1"
CRITICAL_EVIDENCE = ("motor", "language", "vessels")


@dataclass(frozen=True)
class SearchConfig:
    """A bounded, reproducible action-proposal budget shared with comparators."""

    targets_per_compartment: int = 3
    max_target_pool: int = 12000
    max_windows: int = 3
    window_radius_mm: float = 4.0
    max_surface_step_mm: float = 0.5
    seed: int = 20261004
    hard_exclusions: tuple[str, ...] = ("vessels",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_exclusions", tuple(self.hard_exclusions))
        for name in ("targets_per_compartment", "max_target_pool", "max_windows"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_windows > 3:
            raise ValueError("At most three automatic hypothetical windows are supported")
        if self.max_target_pool < self.targets_per_compartment:
            raise ValueError("max_target_pool must cover targets_per_compartment")
        for name in ("window_radius_mm", "max_surface_step_mm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    case_hash: str
    entry_mm: tuple[float, float, float]
    target_mm: tuple[float, float, float]
    target_compartment: str
    tool_id: str
    window_id: str
    tool: ToolGeometry
    window: AccessWindow
    geometry: object
    route_length_mm: float
    accessible_target_volume_mm3_by_compartment: Mapping[str, float]
    normal_tissue_exposure_mm3: float | None
    structure_contact_volume_mm3: Mapping[str, float | None]
    unknowns: tuple[str, ...]
    assumptions: tuple[str, ...]
    category: str = "unclassified"
    dominated_by: tuple[str, ...] = ()
    assessment: str = "incomplete"
    planning_model_hash: str = ""
    accessible_union_volume_mm3: float = 0.0
    simulated_removed_target_volume_mm3: None = None
    clinical_deficit_probability: None = None
    metric_definitions: Mapping[str, str] = field(default_factory=lambda: {
        "accessible_target_volume_mm3": "Target voxel centres inside the active-tip swept tube; conditional static accessibility, not removal.",
        "normal_tissue_exposure_mm3": "Conservative unique-voxel full-tool swept-envelope overlap with the supplied non-target brain mask; not removed tissue.",
        "structure_contact_volume_mm3": "Conservative unique-voxel full-tool swept-envelope overlap with the named supplied structure; not neurological injury.",
        "pareto": "Sampled non-dominance on available geometric volumes and length. Unknown anatomy is excluded from numerical objectives and remains unassessed.",
    })

    def __post_init__(self) -> None:
        if self.simulated_removed_target_volume_mm3 is not None:
            raise ValueError("Route accessibility cannot be reported as simulated removal")
        if self.clinical_deficit_probability is not None:
            raise ValueError("No validated clinical deficit probability is available")
        for name in ("route_length_mm", "accessible_union_volume_mm3", "normal_tissue_exposure_mm3"):
            value = getattr(self, name)
            if value is None and name == "normal_tissue_exposure_mm3":
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("entry_mm", "target_mm"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be three finite physical coordinates")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        for name in ("accessible_target_volume_mm3_by_compartment", "structure_contact_volume_mm3"):
            values = dict(getattr(self, name))
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("Metric names must be nonempty strings")
                if value is None and name == "structure_contact_volume_mm3":
                    continue
                if value is None or not np.isfinite(value) or value < 0:
                    raise ValueError("Geometric volume metrics must be finite and nonnegative")
            object.__setattr__(self, name, MappingProxyType(values))
        object.__setattr__(self, "metric_definitions", MappingProxyType(dict(self.metric_definitions)))
        for name in ("unknowns", "assumptions", "dominated_by"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def accessible_target_volume_mm3(self) -> float:
        return self.accessible_union_volume_mm3

    @property
    def feasible(self) -> bool:
        return bool(self.geometry.feasible)

    @property
    def route_points_mm(self) -> tuple[tuple[float, float, float], ...]:
        return (self.entry_mm, self.target_mm)

    def to_plan(self) -> Plan:
        """Convert the evaluator output to the portable, stale-checked core plan."""
        from .core import Plan

        return Plan(
            plan_id=self.route_id,
            case_hash=self.case_hash,
            route_points_mm=self.route_points_mm,
            tool_id=self.tool_id,
            accessible_target_volume_mm3=self.accessible_target_volume_mm3,
            unknowns=self.unknowns,
            metadata=self.to_dict(),
        )

    def to_dict(self) -> dict:
        # The portable record keeps the measured certificate and replay poses;
        # re-serializing every full-tool envelope voxel balloons saved cases.
        from dataclasses import fields

        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "geometry"}
        result["geometry"] = self.geometry.to_dict()
        result["tool"] = asdict(self.tool)
        result["window"] = asdict(self.window)
        result["accessible_target_volume_mm3"] = self.accessible_target_volume_mm3
        return _json_value(result)


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[RouteCandidate, ...]
    case_hash: str
    planning_model_hash: str
    elapsed_seconds: float
    cancelled: bool
    requested_candidates: int
    assumptions: tuple[str, ...]
    access_support: Mapping[str, str] = field(default_factory=dict)
    optimizer_mode: str = "SEARCH"
    optimizer_version: str = SEARCH_VERSION
    gradient_steps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "access_support", MappingProxyType(dict(self.access_support)))

    @property
    def pareto_candidates(self) -> tuple[RouteCandidate, ...]:
        return tuple(c for c in self.candidates if c.category == "pareto")

    def to_dict(self) -> dict:
        from dataclasses import fields

        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "candidates"}
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return _json_value(result)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ affine[:3, :3].T + affine[:3, 3]


def sample_target_points(
    case: CaseData, config: SearchConfig | None = None
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Stratified deterministic farthest-point proposals, including each compartment.

    Sampling occurs in physical space. The bounded pool affects proposal diversity,
    never the resolution of the supplied geometry or accessibility measurements.
    """
    config = config or SearchConfig()
    rng = np.random.default_rng(config.seed)
    proposals = []
    for name in sorted(case.compartments):
        voxels = np.argwhere(case.compartments[name])
        if len(voxels) == 0:
            continue
        centroid = _world(voxels.mean(axis=0), case.affine)
        if len(voxels) > config.max_target_pool:
            voxels = voxels[np.sort(rng.choice(len(voxels), config.max_target_pool, replace=False))]
        points = _world(voxels, case.affine)
        first = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
        chosen = [first]
        distance = np.sum((points - points[first]) ** 2, axis=1)
        while len(chosen) < min(config.targets_per_compartment, len(points)):
            next_index = int(np.argmax(distance))
            if distance[next_index] <= 0:
                break
            chosen.append(next_index)
            distance = np.minimum(distance, np.sum((points - points[next_index]) ** 2, axis=1))
        proposals.extend((name, tuple(float(x) for x in points[i])) for i in chosen)
    # Overlapping radiological labels can nominate the identical physical action.
    # Keep a single route while retaining every compartment's measured volume.
    unique = {}
    for name, point in proposals:
        unique.setdefault(point, (name, point))
    return tuple(unique.values())


def generate_hypothetical_windows(
    case: CaseData, config: SearchConfig | None = None,
    *, support_mask: np.ndarray | None = None,
    support_provenance: Mapping[str, Any] | None = None,
) -> tuple[AccessWindow, ...]:
    """Propose three explicitly hypothetical brain-envelope access windows.

    Requires a supplied/derived brain mask or an explicitly passed support mask
    with provenance. Estimated MRI support is never treated as verified tissue.
    The outer mask boundary is a hypothetical tissue envelope; it does not
    identify the cortex, sulci, vessels, scalp or skull.
    User-reviewed windows should be passed to ``generate_candidate_routes``.
    """
    config = config or SearchConfig()
    support, _ = _access_support(case, support_mask, support_provenance)
    if support is None or not np.any(support):
        raise ValueError("BRAIN_ENVELOPE_REQUIRED: provide a brain mask or explicit hypothetical access windows")
    target = np.zeros(case.mri.shape, dtype=bool)
    for mask in case.compartments.values():
        target |= mask
    voxels = np.argwhere(target)
    if not len(voxels):
        raise ValueError("EMPTY_TARGET: no radiological target voxels")
    centre = _world(voxels.mean(axis=0), case.affine)
    brain_centre = _world((np.asarray(case.mri.shape) - 1) / 2, case.affine)
    side = 1.0 if centre[0] >= brain_centre[0] else -1.0
    directions = np.array(((0, 0, 1), (side, 0, 0), (0, 1, 0)), dtype=float)
    if case.frame == "LPS+":
        directions[2, 1] *= -1
    # Rays are in patient RAS millimetres; affine inversion handles obliquity.
    inverse = np.linalg.inv(case.affine)
    step = float(np.linalg.svd(case.affine[:3, :3], compute_uv=False).min()) / 3.0
    maximum = float(np.linalg.norm(np.asarray(case.mri.shape) @ np.abs(case.affine[:3, :3]).T))
    distances = np.arange(0, maximum + step, step)
    windows = []
    for index, direction in enumerate(directions[: config.max_windows]):
        points = centre + distances[:, None] * direction
        indices = np.floor(_world(points, inverse) + 0.5).astype(int)
        inside = np.all((indices >= 0) & (indices < np.asarray(case.mri.shape)), axis=1)
        ray_support = np.zeros(len(points), dtype=bool)
        ray_support[inside] = support[tuple(indices[inside].T)]
        occupied = np.flatnonzero(ray_support)
        if not len(occupied):
            continue
        # Use the outermost support, avoiding an internal ventricle as an entrance.
        edge = int(occupied[-1])
        position = centre + (distances[edge] + step / 2) * direction
        windows.append(AccessWindow(
            center_mm=tuple(float(x) for x in position),
            normal_inward=tuple(float(x) for x in -direction),
            radius_mm=config.window_radius_mm,
            window_id=f"hypothetical-envelope-{index + 1}",
        ))
    if not windows:
        raise ValueError("ACCESS_WINDOW_UNRESOLVED: no brain-envelope intersection")
    return tuple(windows)


def _access_support(case, support_mask, support_provenance):
    if support_mask is None:
        if support_provenance:
            raise ValueError("support_provenance requires a support_mask")
        from .structural_evidence import planning_brain_support
        return planning_brain_support(case)
    support = np.asarray(support_mask)
    if case.brain_mask is not None:
        from .structural_evidence import planning_brain_support
        planning_brain_support(case)
    if support.shape != case.mri.shape or support.dtype != np.bool_ or not np.any(support):
        raise ValueError("support_mask must be a nonempty boolean mask in the case grid")
    record = dict(support_provenance or {})
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in ("source", "method", "evidence_type")):
        raise ValueError("Explicit support requires source, method and evidence_type provenance")
    if record["evidence_type"] not in {"estimated", "observed", "simulated"}:
        raise ValueError("Support evidence_type must be estimated, observed or simulated")
    for proposal in getattr(case, "structural_evidence", {}).values():
        proposal._assert_mask_layout()
        if np.array_equal(support, proposal.mask) and proposal.review_status != "accepted":
            raise ValueError("BRAIN_MASK_REVIEW_REQUIRED: an extraction proposal cannot define working access support")
    if (case.metadata.get("allow_nonzero_mri_access_support") is False or case.metadata.get("structural_coverage") == "full_head") and (
        "nonzero" in record["method"].lower().replace("-", "")
        or np.array_equal(support, case.mri != 0)
    ):
        raise ValueError("FULL_HEAD_SUPPORT_NOT_CORTEX: nonzero whole-head MRI cannot define an intracranial access window; use a reviewed brain mask or explicit hypothetical window")
    from .structural_evidence import validate_explicit_support
    record = validate_explicit_support(case, support, record)
    record["mask_sha256"] = sha256(np.ascontiguousarray(support).tobytes()).hexdigest()
    return support, record


def _accessible_volumes(
    case: CaseData, entry: np.ndarray, target: np.ndarray, tool: ToolGeometry,
    compartment_points: Mapping[str, np.ndarray],
) -> dict[str, float]:
    # The active region is a distal capsule, not a point. Its complete insertion
    # sweep starts one active-tip length behind the initial tip centre.
    direction = (target - entry) / np.linalg.norm(target - entry)
    entry = entry - tool.tip_length_mm * direction
    axis = target - entry
    length_squared = float(axis @ axis)
    volumes = {}
    for name, points in compartment_points.items():
        along = np.clip((points - entry) @ axis / length_squared, 0.0, 1.0)
        distance_squared = np.sum((points - entry - along[:, None] * axis) ** 2, axis=1)
        volumes[name] = float(np.count_nonzero(distance_squared <= tool.tip_radius_mm**2 + 1e-10) * case.voxel_volume_mm3)
    return volumes


def _objective(candidate: RouteCandidate) -> np.ndarray:
    # Each radiological compartment is a separate objective; no genotype or
    # invented biological weights can change tool/hazard geometry here.
    values = [-float(v) for _, v in sorted(candidate.accessible_target_volume_mm3_by_compartment.items())]
    values.append(candidate.route_length_mm)
    if candidate.normal_tissue_exposure_mm3 is not None:
        values.append(candidate.normal_tissue_exposure_mm3)
    values.extend(float(v) for _, v in sorted(candidate.structure_contact_volume_mm3.items()) if v is not None)
    return np.asarray(values, dtype=float)


def classify_candidates(candidates: Sequence[RouteCandidate]) -> tuple[RouteCandidate, ...]:
    """Keep rejected candidates and finite sampled geometric Pareto alternatives.

    Comparability requires the same case/model and observed metric keys; missing
    evidence cannot silently become a zero-cost objective in a pooled frontier.
    """
    if not candidates:
        return ()
    if len({candidate.route_id for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate route IDs must be unique")
    case_models = {(c.case_hash, c.planning_model_hash) for c in candidates}
    if len(case_models) != 1:
        raise ValueError("Cannot rank routes from different frozen cases or models")
    signatures = {
        (tuple(sorted(c.accessible_target_volume_mm3_by_compartment)),
         c.normal_tissue_exposure_mm3 is not None,
         tuple(sorted(k for k, v in c.structure_contact_volume_mm3.items() if v is not None)))
        for c in candidates if c.feasible
    }
    if len(signatures) > 1:
        raise ValueError("Cannot rank candidates with different evidence coverage")
    feasible = [c for c in candidates if c.feasible]
    objectives = {c.route_id: _objective(c) for c in feasible}
    classified = []
    for candidate in candidates:
        if not candidate.feasible:
            classified.append(replace(candidate, category="rejected", dominated_by=()))
            continue
        current = objectives[candidate.route_id]
        dominators = tuple(
            other.route_id for other in feasible if other.route_id != candidate.route_id
            and np.all(objectives[other.route_id] <= current + 1e-9)
            and np.any(objectives[other.route_id] < current - 1e-9)
        )
        classified.append(replace(candidate, category="dominated" if dominators else "pareto", dominated_by=dominators))
    return tuple(classified)


def generate_candidate_routes(
    case: CaseData,
    *,
    windows: Sequence[AccessWindow] | None = None,
    tools: Sequence[ToolGeometry] | None = None,
    critical_masks: Mapping[str, np.ndarray | None] | None = None,
    support_mask: np.ndarray | None = None,
    support_provenance: Mapping[str, Any] | None = None,
    explicit_targets: Sequence[tuple[str, Sequence[float]]] | None = None,
    config: SearchConfig | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SearchResult:
    """Search fixed straight insertions under an explicit, immutable case model.

    Cancellation returns the evaluated partial set; no completion is fabricated.
    All supplied masks share the case grid. Missing motor/language/vessels remain
    unknown even when every tested geometric constraint passes. Explicit targets
    replace sampling and must be native voxel centres in the named compartment;
    their exact physical coordinates are part of the frozen search model.
    """
    started = perf_counter()
    config = config or SearchConfig()
    _, support_record = _access_support(case, support_mask, support_provenance)
    if windows is not None and support_mask is not None:
        raise ValueError("Supply explicit windows or support for automatic windows, not both")
    windows = tuple(generate_hypothetical_windows(case, config, support_mask=support_mask, support_provenance=support_provenance) if windows is None else windows)
    tools = tuple(GENERIC_TOOLS if tools is None else tools)
    if not windows or not tools:
        raise ValueError("At least one access window and tool are required")
    if len({w.window_id for w in windows}) != len(windows):
        raise ValueError("Access window IDs must be unique")
    if len({t.tool_id for t in tools}) != len(tools):
        raise ValueError("Tool IDs must be unique")
    masks: dict[str, np.ndarray | None] = {name: None for name in CRITICAL_EVIDENCE}
    for name, mask in (critical_masks or {}).items():
        if mask is None:
            masks[name] = None
            continue
        array = np.asarray(mask)
        if array.shape != case.mri.shape or array.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean mask in the case grid")
        array = np.array(array, dtype=bool, copy=True)
        array.setflags(write=False)
        masks[name] = array
    if explicit_targets is None:
        proposals = sample_target_points(case, config)
    else:
        validated = []
        for name, point in explicit_targets:
            target = np.asarray(point, dtype=float)
            if name not in case.compartments or target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError("Explicit targets require a known compartment and three finite physical coordinates")
            index_float = _world(target, np.linalg.inv(case.affine))
            index = np.rint(index_float).astype(int)
            if (not np.allclose(index_float, index, rtol=0, atol=1e-7)
                    or np.any(index < 0) or np.any(index >= np.asarray(case.mri.shape))
                    or not case.compartments[name][tuple(index)]):
                raise ValueError("Explicit targets must be native voxel centres inside their named compartment")
            validated.append((name, tuple(float(value) for value in target)))
        if len({point for _, point in validated}) != len(validated):
            raise ValueError("Explicit target coordinates must be unique")
        proposals = tuple(validated)
    if not proposals:
        raise ValueError("EMPTY_TARGET: no radiological target voxels")
    compartment_points = {name: _world(np.argwhere(mask), case.affine) for name, mask in sorted(case.compartments.items())}
    union = np.zeros(case.mri.shape, dtype=bool)
    for mask in case.compartments.values():
        union |= mask
    union_points = {"union": _world(np.argwhere(union), case.affine)}
    normal = None if case.brain_mask is None else case.brain_mask & ~union
    forbidden = np.zeros(case.mri.shape, dtype=bool)
    for name in config.hard_exclusions:
        if masks.get(name) is not None:
            forbidden |= masks[name]
    scene = GeometryScene(forbidden_mask=forbidden, affine=case.affine, exposure_mask=normal)
    case_hash = case.semantic_hash
    if callable(case_hash):
        case_hash = case_hash()
    mask_hashes = {
        name: None if mask is None else sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        for name, mask in sorted(masks.items())
    }
    frozen = {
        "case_hash": case_hash, "config": asdict(config), "geometry_version": GEOMETRY_VERSION,
        "search_version": SEARCH_VERSION, "windows": [asdict(w) for w in windows],
        "tools": [asdict(t) for t in tools], "critical_masks": mask_hashes,
        "access_support": support_record,
    }
    if explicit_targets is not None:
        # The default search receipt and original route IDs remain unchanged.
        frozen["explicit_targets"] = proposals
    model_hash = sha256(json.dumps(_json_value(frozen), sort_keys=True, allow_nan=False).encode()).hexdigest()
    assumptions = (
        "Annotation-assisted static route enumeration; no tissue removal is simulated.",
        "Access windows are explicitly hypothetical intracranial openings; cortical surface, skull and scalp suitability are not certified.",
        "Accessibility is conditional on establishing the straight access; brain-envelope overlap is recorded, not silently cleared.",
        "Only supplied anatomical constraints are checked; absent functional and vascular anatomy remains unassessed.",
        "Sampled geometric Pareto set within the declared candidate budget; no global or clinical optimum is claimed.",
    )
    if support_mask is not None:
        assumptions += (f"Hypothetical window support: {support_record['evidence_type']}; {support_record['method']}; source {support_record['source']}. This support does not label intervening tissue.",)
    unknowns = tuple(dict.fromkeys(
        tuple(case.unknowns) + tuple(f"{name}_anatomy_unassessed" for name in sorted(set(CRITICAL_EVIDENCE + config.hard_exclusions)) if masks.get(name) is None)
        + (("normal_tissue_exposure_unassessed",) if normal is None else ())
        + ("skull_and_scalp_unassessed", "tissue_forces_unmodeled", "clinical_outcomes_unavailable")
    ))
    requested = len(proposals) * len(windows) * len(tools)
    candidates = []
    cancelled = False
    if progress:
        progress(0, requested)
    for window in windows:
        for compartment, target_tuple in proposals:
            for tool in tools:
                if cancel and cancel():
                    cancelled = True
                    break
                entry = np.asarray(window.center_mm, dtype=float)
                target = np.asarray(target_tuple, dtype=float)
                length = float(np.linalg.norm(target - entry))
                if length <= 1e-8:
                    raise ValueError("DEGENERATE_ROUTE: target coincides with access centre")
                axis = tuple(float(x) for x in (target - entry) / length)
                certificate = check_motion(
                    tool, ToolPose(tuple(entry), axis), ToolPose(target_tuple, axis), scene,
                    access=window, max_surface_step_mm=config.max_surface_step_mm,
                )
                indices = np.asarray(certificate.swept_voxel_indices, dtype=int).reshape(-1, 3)
                contacts = {
                    name: None if mask is None else float(np.count_nonzero(mask[tuple(indices.T)]) * case.voxel_volume_mm3)
                    for name, mask in sorted(masks.items())
                }
                accessible = _accessible_volumes(case, entry, target, tool, compartment_points) if certificate.feasible else {name: 0.0 for name in case.compartments}
                accessible_union = _accessible_volumes(case, entry, target, tool, union_points)["union"] if certificate.feasible else 0.0
                route_id = "search-" + sha256(json.dumps([model_hash, window.window_id, tool.tool_id, target_tuple], sort_keys=True).encode()).hexdigest()[:16]
                candidates.append(RouteCandidate(
                    route_id=route_id, case_hash=case_hash, entry_mm=tuple(float(x) for x in entry),
                    target_mm=target_tuple, target_compartment=compartment, tool_id=tool.tool_id,
                    window_id=window.window_id, tool=tool, window=window, geometry=certificate,
                    route_length_mm=length, accessible_target_volume_mm3_by_compartment=accessible,
                    normal_tissue_exposure_mm3=certificate.exposure_volume_mm3,
                    structure_contact_volume_mm3=contacts,
                    unknowns=tuple(dict.fromkeys(unknowns + tuple(certificate.unknowns))), assumptions=assumptions,
                    assessment="incomplete" if any(masks[name] is None for name in CRITICAL_EVIDENCE) else "supplied_anatomy_only",
                    planning_model_hash=model_hash,
                    accessible_union_volume_mm3=accessible_union,
                ))
                if progress:
                    progress(len(candidates), requested)
            if cancelled:
                break
        if cancelled:
            break
    return SearchResult(classify_candidates(candidates), case_hash, model_hash, perf_counter() - started, cancelled, requested, assumptions, support_record)


def replay_route(candidate: RouteCandidate, *, step_mm: float = 0.5) -> tuple[ToolPose, ...]:
    """Reconstruct the proposed fixed-axis motion, including rejected proposals.

    This is route-pose replay only. A rejected route's certificate retains the
    first failed constraint; replay does not authorize executing that route.
    """
    if not np.isfinite(step_mm) or step_mm <= 0:
        raise ValueError("step_mm must be finite and positive")
    entry, target = np.asarray(candidate.entry_mm), np.asarray(candidate.target_mm)
    axis = (target - entry) / candidate.route_length_mm
    fractions = np.linspace(0, 1, max(2, int(np.ceil(candidate.route_length_mm / step_mm)) + 1))
    return tuple(ToolPose(tuple(entry + fraction * (target - entry)), tuple(axis)) for fraction in fractions)
