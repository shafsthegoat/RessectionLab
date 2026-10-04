"""Read-only inspection of the experimental native axis inventory.

This facade is intentionally separate from selected-route refinement. It creates
one fresh source-bound RAW model, previews its complete initial inventory and
returns a detached record. It never executes an action or grants replay authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Callable

import numpy as np

from .core import CaseData, array_digest, semantic_digest
from .geometry import AccessWindow, ToolGeometry
from .native_axis_simulation import AXIS_ADAPTER_VERSION, AxisColumnNativeSimulator
from .native_proposals import AxisColumnProposalConfig
from .native_resection import NATIVE_RESECTION_VERSION, native_config_from_case
from .native_simulation import NATIVE_ACTION_FEATURE_NAMES
from .simulation import RewardSpec
from .structural_evidence import declared_mri_support_allowed, planning_brain_support
from .worlds import WorldGeneratorConfig, content_hash

AXIS_INSPECTION_VERSION = "native-axis-inspection-v1"
_LPS_DIRECTION_ROUNDOFF_ATOL = 4 * np.finfo(np.float64).eps


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"),
                      default=lambda item: np.asarray(item).tolist())


@dataclass(frozen=True)
class AxisPlanningInspection:
    """Immutable JSON snapshot; each export is an independent mutable copy."""

    payload_json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def binding_hash(self) -> str:
        return self.to_dict()["binding"]["binding_hash"]


def _support_record(case: CaseData, acknowledged: bool) -> dict[str, Any]:
    if type(acknowledged) is not bool:
        raise ValueError("Estimated-support acknowledgement must be a boolean")
    if case.brain_mask is None:
        if not declared_mri_support_allowed(case):
            raise ValueError("Reviewed brain support is required for full-head or unknown source imaging")
        record = {"method": "hole_filled_nonzero_MRI_with_source_targets_retained",
                  "evidence_type": "estimated", "review_status": "declared_skull_strip_assumption",
                  "cortical_access_permitted": False}
    else:
        _, record = planning_brain_support(case)
    if record.get("evidence_type") == "estimated" and record.get("review_status") != "accepted" and not acknowledged:
        raise ValueError("Explicit estimated-support acknowledgement is required separately from neighboring columns")
    return {**record, "estimated_support_acknowledged": acknowledged}


def _access_in_source_frame(access_ras: AccessWindow, frame: str) -> AccessWindow:
    # The public boundary always receives RAS+. The direct native builder expects
    # case.frame and performs its own conversion back into canonical RAS+.
    if frame == "RAS+":
        return access_ras
    signs = np.array([-1., -1., 1.])
    return AccessWindow(signs * access_ras.center_mm, signs * access_ras.normal_inward,
                        access_ras.radius_mm, access_ras.window_id)


def _source_identity(case: CaseData) -> tuple[str, str]:
    # CaseData caches its hashes for normal immutable use. At this external
    # callback boundary, also reject illicit scalar/mapping changes that bypass
    # its frozen dataclass. Perform this full-source read only at entry and exit.
    manifest = case.to_manifest(include_hash=False)
    return semantic_digest(manifest), semantic_digest(case._planning_manifest(manifest))


def inspect_axis_planning(
    case: CaseData, *, access_ras: AccessWindow, tools: tuple[ToolGeometry, ...],
    acknowledge_neighboring_columns: bool, expected_case_hash: str,
    expected_planning_hash: str, reward: RewardSpec, world_generator: WorldGeneratorConfig,
    acknowledge_estimated_support: bool = False,
    proposal_config: AxisColumnProposalConfig | None = None,
    max_steps: int = 3, partial_contact_weight: float = .05, max_tip_step_mm: float = .25,
    hard_exclusion: np.ndarray | None = None, hard_exclusion_provenance: str | None = None,
    expected_binding_hash: str | None = None, cancelled: Callable[[], bool] | None = None,
) -> AxisPlanningInspection:
    """Inspect all initial actions, with no policy, transition or clinical claim.

    Unsupported access, stale inputs, preview exceptions and cancellation raise;
    none returns an incomplete inventory or a synthetic successful STOP result.
    A supported model whose complete inventory has no legal cut is reported as
    ``no_actionable_moves``. Model identity is separate from timed observations.
    """
    started = time.perf_counter()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Native axis inspection cancelled; no inventory published")
        if case.semantic_hash != expected_case_hash or case.planning_hash != expected_planning_hash:
            raise ValueError("Stale source case or planning identity")

    if not isinstance(case, CaseData):
        raise TypeError("A source-bound CaseData is required")
    expected_source = (expected_case_hash, expected_planning_hash)
    if acknowledge_neighboring_columns is not True:
        raise ValueError("Explicit neighboring-column acknowledgement is required")
    if not isinstance(access_ras, AccessWindow) or not isinstance(reward, RewardSpec) or not isinstance(world_generator, WorldGeneratorConfig):
        raise TypeError("Explicit RAS+ access, reward and world-generator configurations are required")
    if not tools or any(not isinstance(tool, ToolGeometry) for tool in tools):
        raise TypeError("An explicit complete tool catalog is required")
    if expected_binding_hash is not None and (not isinstance(expected_binding_hash, str) or not expected_binding_hash):
        raise ValueError("Expected binding hash must be a nonempty string")
    check()
    if _source_identity(case) != expected_source:
        raise ValueError("Stale source case or planning identity")
    support = _support_record(case, acknowledge_estimated_support)
    evidence = case.functional_evidence
    motor = language = motor_coverage = language_coverage = evidence_record = None
    if evidence is not None:
        evidence.assert_matches(case)
        if world_generator.fingerprint != evidence.uncertainty.fingerprint:
            raise ValueError("Uncertainty differs from the frozen functional evidence")
        motor, language = evidence.planning_arrays()
        motor_coverage, language_coverage = evidence.motor_coverage, evidence.language_coverage
        evidence_record = evidence.to_manifest()
    if (hard_exclusion is None) != (hard_exclusion_provenance is None):
        raise ValueError("A supplied hard-exclusion mask requires its own provenance")
    if hard_exclusion_provenance is not None and (not isinstance(hard_exclusion_provenance, str) or not hard_exclusion_provenance.strip()):
        raise ValueError("Hard-exclusion provenance must be a nonempty string")
    requested_access = json.loads(_json(asdict(access_ras)))
    native = native_config_from_case(case, access=_access_in_source_frame(access_ras, case.frame),
        tools=tuple(tools), hard_exclusion=hard_exclusion, max_tip_step_mm=max_tip_step_mm)
    actual_access = json.loads(_json(asdict(native.access)))
    direction_delta = float(np.max(np.abs(
        np.asarray(actual_access["normal_inward"]) - requested_access["normal_inward"])))
    direction_atol = _LPS_DIRECTION_ROUNDOFF_ATOL if case.frame == "LPS+" else 0.
    if (any(actual_access[key] != requested_access[key] for key in
            ("center_mm", "radius_mm", "window_id")) or direction_delta > direction_atol):
        raise ValueError("Native configuration did not preserve canonical RAS+ access within its declared conversion tolerance")
    check()
    source_preparation_seconds = time.perf_counter() - started
    simulator = AxisColumnNativeSimulator(native, proposal_config=proposal_config,
        nominal_motor=motor, nominal_language=language,
        nominal_motor_coverage=motor_coverage, nominal_language_coverage=language_coverage,
        functional_evidence_record=evidence_record,
        reward=reward, world_generator=world_generator, max_steps=max_steps,
        partial_contact_weight=partial_contact_weight,
        compartment_names={index: name for index, name in enumerate(sorted(case.compartments), 1)},
        cancelled=cancelled)
    check()
    actions = simulator.proposed_actions()
    metrics = simulator.metrics()
    receipts = metrics["inventory_receipts"]
    if len(receipts) != 1 or receipts[0]["status"] != "complete":
        raise RuntimeError("Inspection requires exactly one complete initial inventory")
    inventory = receipts[0]
    batch = inventory["batch"]
    if batch["unsupported_reason"] is not None:
        raise ValueError(batch["unsupported_reason"])
    if (simulator.engine.revision != 0 or simulator.engine.history or metrics["environment_steps"] != 0
            or simulator.removed_mask.any() or simulator.exposed_mask.any() or metrics["total_reward"] != 0.):
        raise RuntimeError("Read-only inspection unexpectedly changed native tissue or episode state")
    if [action.action_id for action in actions] != ["STOP", *inventory["certified_action_ids"]]:
        raise RuntimeError("Exported actions differ from the complete certified inventory")

    binding = {
        "version": AXIS_INSPECTION_VERSION, "mode": "experimental_axis_columns",
        "case_hash": expected_case_hash, "planning_hash": expected_planning_hash,
        "case_id": case.case_id, "source_frame": case.frame, "geometry_frame": "RAS+",
        "source_shape": list(case.mri.shape), "native_affine_ras_mm": native.affine,
        "planning_as_of": None if case.context is None else case.context.planning_as_of.isoformat(),
        "neighboring_columns_acknowledged": True, "access": asdict(native.access),
        "requested_access_ras": requested_access,
        "access_conversion": {"source_frame": case.frame, "normal_absolute_tolerance": direction_atol,
                              "normal_maximum_absolute_difference": direction_delta,
                              "policy": "LPS unit-direction renormalization roundoff only; center/radius/id exact"},
        "access_status": "hypothetical_unverified_cortical_access",
        "source_support": {**support, "native_provenance": native.tissue_support_provenance,
                           "mask_hash": array_digest(native.tissue_mask)},
        "tools": [asdict(tool) for tool in native.tools],
        "source_grid_hashes": {name: array_digest(getattr(native, name)) for name in
                               ("affine", "tissue_mask", "target_labels", "hard_exclusion")},
        "compartment_labels": dict(simulator.config.compartment_names),
        "hard_exclusion": {"provided": hard_exclusion is not None,
                           "provenance": hard_exclusion_provenance,
                           "mask_hash": array_digest(native.hard_exclusion)},
        "native_engine_version": NATIVE_RESECTION_VERSION, "native_config_hash": native.fingerprint,
        "adapter_version": AXIS_ADAPTER_VERSION, "decision_model_hash": simulator.decision_model_hash,
        "proposal_model_hash": batch["proposal_model_hash"], "proposal_rule_hash": batch["rule_hash"],
        "proposal_rule": asdict(simulator.proposal_config), "max_steps": simulator.config.max_steps,
        "max_actions": simulator.config.max_actions, "max_tip_step_mm": native.max_tip_step_mm,
        "max_microsteps": native.max_microsteps, "input_profile": "RAW",
        "action_features": NATIVE_ACTION_FEATURE_NAMES, "state_feature_count": 6,
        "reward": asdict(simulator.config.reward), "partial_contact_weight": simulator.partial_contact_weight,
        "world_generator": simulator.config.world_generator.to_dict(),
        "world_generator_hash": simulator.world_generator_fingerprint,
        "world_role": None, "world_partitions_created": False,
        "functional_evidence_available": {"motor": motor is not None, "language": language is not None},
        "vascular_evidence_status": "unassessed", "population_priors_used": evidence is not None,
        "partial_contact_policy": "retained_tissue_exposure_not_removed",
        "fallback_policy": "only_after_primary_preview_rejection",
        "ordering": "STOP_then_provider_column_tool_order",
    }
    if evidence_record is not None:
        binding["functional_evidence"] = evidence_record
    binding = json.loads(_json(binding))
    binding["binding_hash"] = content_hash(binding)
    if expected_binding_hash is not None and binding["binding_hash"] != expected_binding_hash:
        raise ValueError("Stale axis inspection binding")
    accepted_attempts = [attempt for attempt in inventory["attempts"] if attempt["feasible"]]
    if len(accepted_attempts) != len(actions) - 1:
        raise RuntimeError("Preview ledger and accepted action counts disagree")
    exported_actions = [{"action_id": "STOP", "kind": "stop", "geometry": None,
                         "native_preview": None}]
    geometry_unknowns: set[str] = set()
    for action, attempt in zip(actions[1:], accepted_attempts, strict=True):
        if action.tool_id != attempt["tool_id"] or tuple(action.tip_mm) != tuple(attempt["tip_mm"]):
            raise RuntimeError("Native action does not match its preview ledger geometry")
        # The adapter has just authenticated this retained inventory. Read its
        # preview for uncertainty metadata; never issue another geometry query
        # or serialize the engine-local commit certificate as public authority.
        preview = simulator._previews[action.action_id]
        geometry_unknowns.update(preview.geometry_unknowns)
        exported_actions.append({
            "action_id": action.action_id, "kind": "native_stroke",
            "geometry": {"frame": "RAS+", "tool_id": action.tool_id,
                         "entry_mm": attempt["entry_mm"], "tip_mm": action.tip_mm,
                         "axis_unit": action.axis_unit, "insertion_distance_mm": action.insertion_distance_mm},
            "native_preview": {"scope": "native_engine_preview_only", "independent_history_checked": False,
                               "proposal_id": attempt["proposal_id"], "phase": attempt["phase"],
                               "reason": attempt["reason"], "source_hash": batch["source_hash"],
                               "source_state_hash": batch["cavity_state_hash"],
                               "native_config_hash": native.fingerprint,
                               "geometry_unknowns": preview.geometry_unknowns,
                               "native_footprint": preview.native_footprint,
                               "unexecuted_contained_cell_count": len(action.removal_indices),
                               "unexecuted_contact_cell_count": len(action.swept_indices)},
        })
    check()
    simulator.assert_model_frozen()
    report = {
        "version": AXIS_INSPECTION_VERSION, "role": "inspection", "binding": binding,
        "status": "ready" if len(actions) > 1 else "no_actionable_moves",
        "inventory_complete": True, "initial_cavity_state_hash": batch["cavity_state_hash"],
        "inventory": inventory, "actions": exported_actions,
        "legal_non_stop_actions": len(actions) - 1,
        "accounting": {"gradient_steps": 0, "executed_transitions": 0, "native_commits": 0,
                       "simulated_removed_volume_mm3": 0.},
        "proposal_accounting": metrics["proposal_accounting"],
        "timing": {"source_preparation_seconds": source_preparation_seconds,
                   "simulator_initialization": metrics["initialization_timing"],
                   "through_snapshot_preparation_seconds": time.perf_counter() - started,
                   "scope": "nested local measurements; excludes final serialization/checks; not a latency guarantee"},
        "candidate_eligible": False, "removal_authorized": False,
        "clinical_deficit_probability": None,
        "unknowns": sorted(set(case.unknowns) | geometry_unknowns | {"motor_function_unassessed", "language_function_unassessed",
                            "vascular_anatomy_unassessed", "cortical_access_unverified",
                            "rigid_geometry_and_active_brush_are_research_assumptions"}),
    }
    report["inspection_hash"] = content_hash(report)
    result = AxisPlanningInspection(_json(report))
    check()  # Cancellation during serialization still withholds the entire result.
    simulator.assert_model_frozen()  # The callback itself is also outside our trust boundary.
    if _source_identity(case) != expected_source:
        raise ValueError("Stale source case or planning identity after inspection")
    return result
