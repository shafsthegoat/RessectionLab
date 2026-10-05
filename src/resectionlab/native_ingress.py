"""Opt-in initial ingress screening; never a complete-stroke certificate.

The caller supplies the six existing axis exits and their authenticated nominal
proposers. This module neither derives new accesses nor reads evaluator labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import numpy as np

from .core import array_digest, freeze_json, semantic_digest, thaw_json
from .geometry import GeometryScene, ToolPose, capsule_voxel_indices, check_pose
from .native_proposals import NOMINAL_CAVITY_PROPOSAL_VERSION, PreparedNominalCavityProposer
from .native_resection import NativeResectionEngine

INGRESS_VERSION = "six-axis-any-entry-native-ingress-v1"
_PROVIDER_SCHEMA = "permitted-nominal-cavity-columns-v1"
MAX_EXITS = 6
MAX_POSES_PER_EXIT = 78
MAX_SCREENS = MAX_EXITS * MAX_POSES_PER_EXIT


@dataclass(frozen=True)
class IngressCandidate:
    """Existing exit metadata plus a source-bound, unmodified initial engine."""
    axis: int
    outward_sign: int
    distance_mm: float
    engine: NativeResectionEngine
    proposer: PreparedNominalCavityProposer

    def __post_init__(self):
        if type(self.axis) is not int or self.axis not in (0, 1, 2):
            raise ValueError("Exit axis must be 0,1,2")
        if type(self.outward_sign) is not int or self.outward_sign not in (-1, 1):
            raise ValueError("Outward sign must be -1 or +1")
        if (isinstance(self.distance_mm, (bool, np.bool_))
                or not isinstance(self.distance_mm, (int, float, np.integer, np.floating))
                or not np.isfinite(self.distance_mm) or self.distance_mm <= 0):
            raise ValueError("Exit distance must be a positive finite physical length")
        if not isinstance(self.engine, NativeResectionEngine) or not isinstance(self.proposer, PreparedNominalCavityProposer):
            raise TypeError("A native engine and authenticated nominal proposer are required")
        object.__setattr__(self, "distance_mm", float(self.distance_mm))

    @property
    def key(self):
        return self.axis, self.outward_sign


@dataclass(frozen=True)
class IngressScreeningResult:
    """Immutable diagnostic receipt. No serialized result authorizes removal."""
    status: str
    selected_exit: tuple[int, int] | None
    exit_records: tuple[Mapping, ...]
    screen_count: int
    error: str | None = None

    @property
    def complete(self):
        return self.status in {"selected", "no_ingress_admissible_access"}

    def to_dict(self):
        return {"version": INGRESS_VERSION, "status": self.status, "complete": self.complete,
            "selected_exit": None if self.selected_exit is None else list(self.selected_exit),
            "exits": [thaw_json(row) for row in self.exit_records], "screen_count": self.screen_count,
            "max_screens": MAX_SCREENS, "error": self.error, "complete_stroke_certified": False,
            "removal_authorized": False, "selection": "any_entry_then_distance_axis_outward_sign",
            "scope": "static initial pose; outside-image geometry remains unassessed"}

    @property
    def fingerprint(self):
        return semantic_digest(self.to_dict())


def _initial(candidate):
    engine = candidate.engine
    if engine.revision != 0 or engine.history:
        raise ValueError("Ingress screening requires the unchanged initial cavity")


def _shared_identity(config, batch, proposer):
    # Access-dependent source hashes may differ. Physical source masks, tools,
    # nominal evidence and proposal rule must be the same across all six exits.
    return semantic_digest({"arrays": {name: array_digest(getattr(config, name)) for name in
        ("affine", "tissue_mask", "target_labels", "hard_exclusion")},
        "tools": [asdict(tool) for tool in config.tools], "case_id": config.case_id,
        "support": config.tissue_support_provenance, "step": config.max_tip_step_mm,
        "max_microsteps": config.max_microsteps, "nominal": batch.nominal_target_hash,
        "rule": proposer.rule_hash, "original_index_affine": array_digest(proposer._index_affine),
        "aperture_radius_mm": config.access.radius_mm})


def _ray_pose(config, ray):
    tool = next((tool for tool in config.tools if tool.tool_id == ray.tool_id), None)
    if tool is None:
        raise ValueError("Proposal names an undeclared tool")
    entry, tip = np.asarray(ray.entry_mm, dtype=float), np.asarray(ray.tip_mm, dtype=float)
    if entry.shape != (3,) or tip.shape != (3,) or not np.isfinite([entry, tip]).all():
        raise ValueError("Proposal coordinates must be finite three-vectors")
    if abs(float((entry-config.access.center_mm) @ config.access.normal_inward)) > 1e-8:
        raise ValueError("Native entry must lie on its declared aperture plane")
    displacement = tip-entry
    length = float(np.linalg.norm(displacement))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Native stroke needs a nonzero finite direction")
    axis = displacement / length  # Exact expression used by preview_stroke.
    key = (tool.tool_id, entry.tobytes(), axis.tobytes())
    return tool, entry, axis, key


def screen_axis_accesses(candidates, *, cancelled: Callable[[], bool] | None = None) -> IngressScreeningResult:
    """Screen all six exits and select any-entry-eligible shortest local exit.

    Cancellation or any incomplete/invalid batch returns no selection, retaining
    completed rows. The unchanged native preview must still check every stroke.
    No preview/commit, learning, crop or reward API is called here.
    """
    records, checked, screens = [], [], 0
    status, selected, error = "failed", None, None

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError("Ingress screening interrupted")

    try:
        candidates = tuple(candidates)
        if NOMINAL_CAVITY_PROPOSAL_VERSION != _PROVIDER_SCHEMA:
            raise ValueError("Ingress requires the audited nominal provider schema")
        if (len(candidates) != MAX_EXITS or any(not isinstance(c, IngressCandidate) for c in candidates)
                or {c.key for c in candidates} != {(a, s) for a in range(3) for s in (-1, 1)}):
            raise ValueError("Exactly the six distinct existing source-axis exits are required")
        candidates = tuple(sorted(candidates, key=lambda c: c.key))
        metadata = tuple((c.key, c.distance_mm, id(c.engine), id(c.proposer)) for c in candidates)
        shared = None
        for candidate in candidates:
            row = {"axis": candidate.axis, "outward_sign": candidate.outward_sign,
                "distance_mm": candidate.distance_mm, "status": "not_screened", "eligible": False,
                "poses": [], "proposal_dispositions": None}
            records.append(row)
        for candidate, row in zip(candidates, records):
            check_cancelled()
            _initial(candidate)
            batch = candidate.proposer.propose(candidate.engine, cancelled=cancelled)
            config = candidate.engine.config
            # propose() authenticates this discrete source-axis descriptor. The
            # original source axis may differ slightly from the native grid after
            # declared reconciliation; do not invent a new floating tolerance.
            if (candidate.axis, -candidate.outward_sign) != (candidate.proposer._axis, candidate.proposer._sign):
                raise ValueError("Exit axis/sign metadata differs from the authenticated source-axis access")
            identity = _shared_identity(config, batch, candidate.proposer)
            if shared is not None and identity != shared:
                raise ValueError("Six exits must share the same physical source, tools, nominal target and proposal rule")
            shared = identity
            row.update(status="screening", engine_model_hash=config.fingerprint,
                source_hash=config.source_hash, access={"center_mm": config.access.center_mm.tolist(),
                    "normal_inward": config.access.normal_inward.tolist(), "radius_mm": config.access.radius_mm,
                    "window_id": config.access.window_id},
                provider_model_hash=batch.model_hash, rule_hash=candidate.proposer.rule_hash,
                cavity_state_hash=batch.cavity_state_hash, proposal_dispositions=batch.to_dict())
            if (len(config.tools) > 2 or len(batch.proposals) > MAX_POSES_PER_EXIT
                    or len(batch.ledger) > MAX_POSES_PER_EXIT
                    or any(slot.reason == "CANDIDATE_CAP" for slot in batch.ledger)):
                raise ValueError("Incomplete or oversized initial proposal inventory")
            hard_scene = GeometryScene(config.hard_exclusion, config.affine)
            cell_scene = GeometryScene(np.zeros(config.tissue_mask.shape, bool), config.affine)
            seen = {}
            for ray in batch.proposals:
                check_cancelled()
                tool, entry, axis, key = _ray_pose(config, ray)
                result = {"proposal_id": ray.proposal_id, "tool_id": tool.tool_id,
                    "entry_mm": entry.tolist(), "tip_mm": list(ray.tip_mm), "axis_unit": axis.tolist()}
                if key in seen:
                    prior = seen[key]
                    result.update(duplicate_of=prior["proposal_id"], screen_index=prior["screen_index"],
                        admissible=prior["admissible"], geometry=prior["geometry"],
                        blocked_cell_count=prior["blocked_cell_count"], first_blocked_cell=prior["first_blocked_cell"],
                        first_blocked_cell_mm=prior["first_blocked_cell_mm"], reasons=prior["reasons"])
                else:
                    if screens >= MAX_SCREENS:
                        raise ValueError("Ingress screen budget exceeded")
                    screens += 1
                    geometry = check_pose(tool, ToolPose(entry, axis), hard_scene, config.access)
                    # Use the raw normalized axis exactly as native step zero does;
                    # ToolPose itself normalizes again for the separate hard check.
                    shaft = capsule_voxel_indices(cell_scene, entry-tool.working_length_mm*axis,
                                                   entry-tool.tip_length_mm*axis, tool.shaft_radius_mm)
                    blocked = shaft[config.tissue_mask[tuple(shaft.T)]]
                    first = None if not len(blocked) else min(map(tuple, blocked.tolist()))
                    reasons = ["HARD_GEOMETRY:" + failure.reason for failure in geometry.failures]
                    if first is not None:
                        reasons.append("SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE")
                    result.update(duplicate_of=None, screen_index=screens-1, admissible=not reasons,
                        geometry=geometry.to_dict(), reasons=reasons, blocked_cell_count=len(blocked),
                        first_blocked_cell=first, first_blocked_cell_mm=None if first is None else
                            (config.affine[:3,:3] @ first + config.affine[:3,3]).tolist())
                    seen[key] = result
                row["poses"].append(result)
            row.update(status="screened", eligible=any(pose["admissible"] for pose in row["poses"]))
            checked.append((candidate, batch))
        # A final callback cannot mutate an earlier input and leave a selection.
        check_cancelled()
        if metadata != tuple((c.key, c.distance_mm, id(c.engine), id(c.proposer)) for c in candidates):
            raise RuntimeError("Exit metadata changed during screening")
        for candidate, batch in checked:
            _initial(candidate)
            candidate.proposer.validate_batch(batch, candidate.engine)
        eligible = [row for row in records if row["eligible"]]
        if eligible:
            winner = min(eligible, key=lambda row: (row["distance_mm"], row["axis"], row["outward_sign"]))
            selected = winner["axis"], winner["outward_sign"]
            status = "selected"
        else:
            status = "no_ingress_admissible_access"
    except InterruptedError as exc:
        status, error = "interrupted", str(exc)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    return IngressScreeningResult(status, selected, tuple(freeze_json(row) for row in records), screens, error)
