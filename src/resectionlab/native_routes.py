"""Explicit native-grid research approaches, kept separate from the default search.

These two named profiles have active tips wider than their shafts. Their axial
source-grid approaches avoid an avoidable voxel-boundary staircase, without
relaxing any native-cell containment, prior-cavity or complete-tool constraint.
Static route feasibility still makes no claim that an action can remove tissue.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_fill_holes

from .geometry import AccessWindow, ToolGeometry
from .native_resection import NATIVE_GENERIC_TOOLS
from .planning import SearchConfig, SearchResult, generate_candidate_routes
from .structural_evidence import declared_mri_support_allowed, planning_brain_support


NATIVE_AXIS_PROPOSAL_VERSION = "explicit-native-source-axis-research-v1"


@dataclass(frozen=True)
class NativeAxisProposal:
    """One exact ray in the source case's declared physical coordinate frame."""

    window: AccessWindow
    target_mm: tuple[float, float, float]
    target_compartment: str
    source_voxel: tuple[int, int, int]
    support_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_provenance", MappingProxyType(dict(self.support_provenance)))


def propose_native_axis_access(case: Any, *, radius_mm: float = 6.) -> NativeAxisProposal:
    """Find the nearest outer support boundary along a source voxel-grid axis.

    The target is the rounded union centroid when that cell is annotated;
    otherwise it is the closest annotated native cell in physical distance.
    An assumed or reviewed brain envelope remains hypothetical access. An
    unknown or explicitly full-head source never acquires a cerebral surface.
    """
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Native research access radius must be finite and positive")
    affine = np.asarray(case.affine)
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    gram = basis.T @ basis
    if not np.allclose(gram, np.diag(spacing ** 2), rtol=1e-8, atol=1e-8):
        raise ValueError("Native source-axis proposals require an orthogonal source affine")
    union = np.zeros(case.mri.shape, dtype=bool)
    for mask in case.compartments.values():
        if np.any(union & mask):
            raise ValueError("Native source-axis routes require nonoverlapping source target compartments")
        union |= mask
    indices = np.argwhere(union)
    if len(indices) == 0:
        raise ValueError("EMPTY_TARGET: no native target cell is available")
    centre = indices.mean(axis=0)
    target_index = np.rint(centre).astype(int)
    if not union[tuple(target_index)]:
        distances = np.sum(((indices - centre) @ basis.T) ** 2, axis=1)
        target_index = indices[int(np.argmin(distances))]
    if case.brain_mask is None:
        if not declared_mri_support_allowed(case):
            raise ValueError("BRAIN_ENVELOPE_REQUIRED: native research access needs reviewed support or an explicit skull-stripped source declaration")
        tissue = binary_fill_holes(np.asarray(case.mri) != 0) | union
        support = {
            "source": case.semantic_hash,
            "method": "hole_filled_nonzero_declared_skull_stripped_MRI_union_target_v1",
            "evidence_type": "estimated",
            "review_status": "declared_skull_strip_assumption",
            "cortical_access_permitted": False,
        }
    else:
        tissue, support = planning_brain_support(case)
        if np.any(union & ~tissue):
            raise ValueError("Target annotation lies outside the supplied brain mask; review conflicting anatomy before planning")
        support = dict(support)
    support.update(
        mask_sha256=sha256(np.ascontiguousarray(tissue).tobytes()).hexdigest(),
        proposal_version=NATIVE_AXIS_PROPOSAL_VERSION,
        coordinate_frame=case.frame,
        cortical_access_permitted=False,
    )
    options = []
    for axis in range(3):
        selector = list(target_index)
        selector[axis] = slice(None)
        occupied = np.flatnonzero(tissue[tuple(selector)])
        for side in (-1, 1):
            position = target_index.astype(float)
            position[axis] = occupied[0] - .5 if side == -1 else occupied[-1] + .5
            depth = abs(position[axis] - target_index[axis]) * spacing[axis]
            entry = basis @ position + affine[:3, 3]
            inward = basis[:, axis] * (-side) / spacing[axis]
            options.append((depth, axis, side, entry, inward))
    _, _, _, entry, inward = min(options, key=lambda row: row[:3])
    target = basis @ target_index + affine[:3, 3]
    compartment = next(name for name in sorted(case.compartments)
                       if case.compartments[name][tuple(target_index)])
    window = AccessWindow(entry, inward, float(radius_mm),
                          "native_hypothetical_nearest_envelope_axis")
    return NativeAxisProposal(window, tuple(float(value) for value in target), compartment,
                              tuple(int(value) for value in target_index), support)


def generate_native_axis_routes(
    case: Any, *, tools: Sequence[ToolGeometry] = NATIVE_GENERIC_TOOLS,
    critical_masks: Mapping[str, np.ndarray | None] | None = None,
    config: SearchConfig | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SearchResult:
    """Generate a separately hashed, explicit alternative research scenario.

    Callers may append these candidates to the original search for inspection,
    retaining both model receipts and their separately computed Pareto cohorts.
    Only the unchanged named native profiles are accepted here; selecting one
    never changes the dimensions of a generic suction or aspirator route.
    """
    started = perf_counter()
    tools = tuple(tools)
    if not tools or any(tool not in NATIVE_GENERIC_TOOLS for tool in tools):
        raise ValueError("Native-axis routes require the explicitly named, unchanged native research profiles")
    config = config or SearchConfig(targets_per_compartment=1, max_windows=1, window_radius_mm=6.)
    proposal = propose_native_axis_access(case, radius_mm=config.window_radius_mm)
    result = generate_candidate_routes(
        case, windows=(proposal.window,), tools=tools, critical_masks=critical_masks,
        explicit_targets=((proposal.target_compartment, proposal.target_mm),),
        config=config, cancel=cancel, progress=progress,
    )
    support = dict(proposal.support_provenance)
    frozen = {"route_model_hash": result.planning_model_hash,
              "proposal_version": NATIVE_AXIS_PROPOSAL_VERSION,
              "native_access_support": support}
    model_hash = sha256(json.dumps(frozen, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assumptions = result.assumptions + (
        "Separately named native-cell aspiration research profiles; original generic instrument dimensions are unchanged.",
        "Explicit source-grid-axis hypothetical access to an annotated native target centre; this alternative is annotation-assisted and has its own frozen model.",
        f"Native access support: {support['evidence_type']}; {support['method']}; source {support['source']}. The cortical surface and opening remain unverified.",
        "Static access remains conditional; source-cell removal and prior-cavity shaft legality require a separate native simulation and independent audit.",
        "Pareto classification applies only within this native-axis research scenario; it is not pooled with other search models.",
    )
    id_map = {route.route_id: "native-axis-" + sha256(json.dumps(
        [model_hash, route.window_id, route.tool_id, route.entry_mm, route.target_mm],
        sort_keys=True).encode()).hexdigest()[:16] for route in result.candidates}
    candidates = tuple(replace(route, route_id=id_map[route.route_id], planning_model_hash=model_hash,
                               assumptions=assumptions,
                               dominated_by=tuple(id_map[other] for other in route.dominated_by))
                       for route in result.candidates)
    return replace(result, candidates=candidates, planning_model_hash=model_hash,
                   elapsed_seconds=perf_counter() - started, assumptions=assumptions,
                   access_support=support, optimizer_version=NATIVE_AXIS_PROPOSAL_VERSION)
