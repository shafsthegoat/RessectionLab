"""Unadmitted research prototype: same estimated world for SEARCH, IL and RL.

This intentionally does not call or replace production ``require_admitted_model``.
Only generated-array controls may execute here until source, split and asset
receipts have independent review. There is no clinical or patient admission.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
import re

import numpy as np

from resectionlab.core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from resectionlab.geometry import AccessWindow, GeometryScene, ToolGeometry, capsule_voxel_indices
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask
from resectionlab.simulation import RewardSpec


SCOPE = "development-estimate-route-research-prototype-v0"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OUTCOME_FIELDS = {"reward", "target_removed_mm3", "normal_removed_mm3",
                   "function_unassessed_removed_volume_mm3", "outcome_scope"}


def _hash(value: str, name: str) -> str:
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact SHA-256 digest")
    return value


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Availability must be a timezone-aware timestamp")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Availability must be a timezone-aware timestamp")
    return moment


def _frame(shape: tuple[int, ...], affine: np.ndarray) -> str:
    return semantic_digest({"shape": list(shape), "affine_ras_mm": affine.tolist(), "frame": "RAS+"})


def _access(access: AccessWindow) -> dict:
    return {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
            "radius_mm": access.radius_mm, "window_id": access.window_id}


@dataclass(frozen=True, slots=True)
class ScanSource:
    modality: str
    image: np.ndarray
    affine_ras_mm: np.ndarray
    source_file_sha256: str
    acquired_at: str | None
    array_sha256: str
    frame_sha256: str
    availability_basis: str = "timestamped"
    preoperative_evidence_sha256: str | None = None

    def __post_init__(self):
        image, affine = np.asarray(self.image), np.asarray(self.affine_ras_mm)
        if (not isinstance(self.modality, str) or not self.modality.strip() or len(self.modality) > 32
                or image.ndim != 3 or min(image.shape) < 3 or image.size > 32_000_000
                or image.dtype.kind not in "iuf" or not np.isfinite(image).all()
                or affine.shape != (4, 4) or not np.isfinite(affine).all()):
            raise ValueError("A finite, same-grid RAS+ structural scan is required")
        image = immutable_array(image, np.float32)
        affine = immutable_array(affine, np.float64)
        if not np.isfinite(image).all():
            raise ValueError("Structural scan exceeds finite float32 range")
        if self.availability_basis == "timestamped":
            _time(self.acquired_at)
        elif self.availability_basis == "source_attested_preoperative":
            if self.acquired_at is not None:
                raise ValueError("An unknown acquisition time must remain unknown")
            _hash(self.preoperative_evidence_sha256, "preoperative source evidence")
        else:
            raise ValueError("Scan availability needs an exact time or source-backed preoperative attestation")
        _hash(self.source_file_sha256, "source file")
        if _hash(self.array_sha256, "scan array") != array_digest(image):
            raise ValueError("Scan array bytes changed")
        if _hash(self.frame_sha256, "scan frame") != _frame(image.shape, affine):
            raise ValueError("Scan frame changed")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "affine_ras_mm", affine)

    def identity(self) -> dict:
        if array_digest(self.image) != self.array_sha256 or _frame(self.image.shape, self.affine_ras_mm) != self.frame_sha256:
            raise ValueError("Scan source changed after construction")
        return {"modality": self.modality, "source_file_sha256": self.source_file_sha256,
                "array_sha256": self.array_sha256, "frame_sha256": self.frame_sha256,
                "acquired_at": self.acquired_at, "availability_basis": self.availability_basis,
                "preoperative_evidence_sha256": self.preoperative_evidence_sha256}


def source_set_hash(sources: tuple[ScanSource, ...]) -> str:
    return semantic_digest([source.identity() for source in sources])


@dataclass(frozen=True, slots=True)
class ScanEstimate:
    kind: str  # brain_envelope_candidate or whole_tumor_candidate
    mask: np.ndarray
    coverage: np.ndarray
    model_sha256: str
    run_sha256: str
    asset_sha256: str
    source_set_sha256: str
    frame_sha256: str
    output_sha256: str
    coverage_sha256: str
    available_at: str
    label_semantics: str
    qc_status: str  # pass or abstain
    training_lineage_status: str  # unknown is retained, not silently approved
    training_overlap_status: str

    def __post_init__(self):
        if self.kind not in {"brain_envelope_candidate", "whole_tumor_candidate"}:
            raise ValueError("Support and target estimate kinds must remain distinct")
        mask, coverage = np.asarray(self.mask), np.asarray(self.coverage)
        if (mask.ndim != 3 or mask.shape != coverage.shape or mask.dtype.kind not in "biuf"
                or coverage.dtype.kind not in "biuf" or not np.isfinite(mask).all()
                or not np.isfinite(coverage).all() or not np.isin(mask, (0, 1)).all()
                or not np.isin(coverage, (0, 1)).all()):
            raise ValueError("Estimate and coverage must be finite binary same-grid arrays")
        mask, coverage = immutable_array(mask, bool), immutable_array(coverage, bool)
        for name in ("model_sha256", "run_sha256", "asset_sha256", "source_set_sha256",
                     "frame_sha256", "output_sha256", "coverage_sha256"):
            _hash(getattr(self, name), name)
        if self.output_sha256 != array_digest(mask) or self.coverage_sha256 != array_digest(coverage):
            raise ValueError("Estimate output or coverage changed")
        if not isinstance(self.label_semantics, str) or not self.label_semantics.strip() or len(self.label_semantics) > 256:
            raise ValueError("Explicit bounded label semantics are required")
        if self.qc_status not in {"pass", "abstain"}:
            raise ValueError("QC must be pass or abstain")
        if self.training_lineage_status not in {"unknown", "audited"} or self.training_overlap_status not in {"unknown", "audited_no_overlap", "known_overlap"}:
            raise ValueError("Training lineage and overlap status must be explicit")
        _time(self.available_at)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "coverage", coverage)

    def identity(self) -> dict:
        if array_digest(self.mask) != self.output_sha256 or array_digest(self.coverage) != self.coverage_sha256:
            raise ValueError("Estimate changed after construction")
        return {name: getattr(self, name) for name in (
            "kind", "model_sha256", "run_sha256", "asset_sha256", "source_set_sha256",
            "frame_sha256", "output_sha256", "coverage_sha256", "available_at",
            "label_semantics", "qc_status", "training_lineage_status", "training_overlap_status")}


@dataclass(frozen=True, slots=True)
class DevelopmentUseDeclaration:
    """Caller declaration bound to exact inputs, not an authenticated release."""
    scope: str
    patient_group: str
    split_role: str
    source_set_sha256: str
    support_identity_sha256: str
    target_identity_sha256: str
    operation: str
    clinical_use_permitted: bool = False
    independent_evaluation_permitted: bool = False

    def __post_init__(self):
        if (self.scope != SCOPE or self.split_role != "DEVELOPMENT"
                or not isinstance(self.patient_group, str) or not self.patient_group.strip()
                or len(self.patient_group) > 128 or self.operation != "nominal_route_comparison"
                or self.clinical_use_permitted is not False
                or self.independent_evaluation_permitted is not False):
            raise ValueError("Only a declared, nonclinical DEVELOPMENT comparison is supported")
        for name in ("source_set_sha256", "support_identity_sha256", "target_identity_sha256"):
            _hash(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ResearchEstimatePlanningSpec:
    sources: tuple[ScanSource, ...]
    actor_modality: str
    support: ScanEstimate
    target: ScanEstimate
    declaration: DevelopmentUseDeclaration
    access: AccessWindow
    tools: tuple[ToolGeometry, ...]
    reward: RewardSpec
    horizon: int
    decision_cutoff: str
    roi_start: tuple[int, int, int] | None = None
    roi_stop: tuple[int, int, int] | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self):
        sources = tuple(self.sources)
        if (not sources or any(type(s) is not ScanSource for s in sources)
                or len({s.modality for s in sources}) != len(sources)
                or self.actor_modality not in {s.modality for s in sources}
                or type(self.support) is not ScanEstimate or type(self.target) is not ScanEstimate
                or type(self.declaration) is not DevelopmentUseDeclaration
                or type(self.access) is not AccessWindow or type(self.reward) is not RewardSpec
                or type(self.horizon) is not int or not 1 <= self.horizon <= 4
                or not 1 <= len(self.tools) <= 4 or any(type(t) is not ToolGeometry for t in self.tools)):
            raise ValueError("Exact preoperative research input types and bounded horizon are required")
        first = sources[0]
        if any(s.image.shape != first.image.shape or s.frame_sha256 != first.frame_sha256 for s in sources):
            raise ValueError("All source modalities need one verified physical grid; no implicit registration")
        cutoff = _time(self.decision_cutoff)
        if any(s.acquired_at is not None and _time(s.acquired_at) > cutoff for s in sources) or any(
                _time(e.available_at) > cutoff for e in (self.support, self.target)):
            raise ValueError("Only estimates available before the planning decision are permitted")
        if any(s.acquired_at is not None and _time(e.available_at) < _time(s.acquired_at)
               for s in sources for e in (self.support, self.target)):
            raise ValueError("An estimate cannot precede an exactly dated source scan")
        if (self.support.kind != "brain_envelope_candidate" or self.target.kind != "whole_tumor_candidate"
                or any(e.mask.shape != first.image.shape or e.source_set_sha256 != source_set_hash(sources)
                       or e.frame_sha256 != first.frame_sha256 for e in (self.support, self.target))):
            raise ValueError("Distinct target/support outputs must bind every preoperative source and native frame")
        if (self.declaration.source_set_sha256 != source_set_hash(sources)
                or self.declaration.support_identity_sha256 != semantic_digest(self.support.identity())
                or self.declaration.target_identity_sha256 != semantic_digest(self.target.identity())):
            raise ValueError("Scoped research declaration does not bind these exact estimates")
        start = (0, 0, 0) if self.roi_start is None else tuple(self.roi_start)
        stop = first.image.shape if self.roi_stop is None else tuple(self.roi_stop)
        if (len(start) != 3 or len(stop) != 3
                or any(type(a) is not int or type(b) is not int or a < 0 or b > n or b - a < 3
                       for a, b, n in zip(start, stop, first.image.shape))):
            raise ValueError("Qualified ROI must be a bounded native-grid box at least three cells wide")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "roi_start", start)
        object.__setattr__(self, "roi_stop", stop)
        object.__setattr__(self, "fingerprint", semantic_digest(self._identity()))

    def _identity(self) -> dict:
        return {"scope": SCOPE, "sources": [s.identity() for s in self.sources],
                "actor_modality": self.actor_modality, "support": self.support.identity(),
                "target": self.target.identity(), "declaration": asdict(self.declaration),
                "access": _access(self.access), "tools": [asdict(t) for t in self.tools],
                "reward": asdict(self.reward), "horizon": self.horizon,
                "decision_cutoff": self.decision_cutoff,
                "roi_start": list(self.roi_start), "roi_stop": list(self.roi_stop)}

    def assert_intact(self) -> None:
        if semantic_digest(self._identity()) != self.fingerprint:
            raise ValueError("Research planning inputs changed after construction")


@dataclass(frozen=True, slots=True)
class Abstention:
    reason: str
    input_hash: str
    clinical_use_permitted: bool = False


@dataclass(frozen=True, slots=True)
class FrozenResearchPlan:
    input_hash: str
    method: str
    configuration_hash: str
    initial_observation_hash: str
    decision_model_hash: str
    action_ids: tuple[str, ...]
    physical_history_hash: str
    horizon: int
    method_accounting: Mapping
    coverage_scope: str = field(init=False, default="native_grid_in_image_tool_supercover_only")
    outside_source_fov_tool_feasibility_assessed: bool = field(init=False, default=False)
    clinical_use_permitted: bool = field(init=False, default=False)
    seal_hash: str = field(init=False)

    def __post_init__(self):
        if self.method not in {"SEARCH", "IL", "RL", "HYBRID"} or not 1 <= len(self.action_ids) <= self.horizon:
            raise ValueError("Only complete named SEARCH, IL, RL or HYBRID plans can seal")
        for name in ("input_hash", "configuration_hash", "initial_observation_hash",
                     "decision_model_hash", "physical_history_hash"):
            _hash(getattr(self, name), name)
        if (("STOP" in self.action_ids and self.action_ids[-1] != "STOP")
                or ("STOP" not in self.action_ids and len(self.action_ids) != self.horizon)
                or self.action_ids.count("STOP") > 1):
            raise ValueError("A full STOP-or-horizon route is required")
        accounting = dict(self.method_accounting)
        if (set(accounting) != {"model_transition_calls", "actor_forward_calls"}
                or any(type(value) is not int or value < 0 for value in accounting.values())):
            raise ValueError("Explicit nonnegative method accounting is required")
        object.__setattr__(self, "method_accounting", freeze_json(accounting))
        object.__setattr__(self, "seal_hash", semantic_digest(self._payload()))

    def _payload(self) -> dict:
        return {"input_hash": self.input_hash, "method": self.method,
                "configuration_hash": self.configuration_hash,
                "initial_observation_hash": self.initial_observation_hash,
                "decision_model_hash": self.decision_model_hash,
                "action_ids": list(self.action_ids), "physical_history_hash": self.physical_history_hash,
                "horizon": self.horizon, "method_accounting": dict(self.method_accounting),
                "coverage_scope": self.coverage_scope,
                "outside_source_fov_tool_feasibility_assessed": self.outside_source_fov_tool_feasibility_assessed,
                "clinical_use_permitted": self.clinical_use_permitted}

    def assert_intact(self) -> None:
        if semantic_digest(self._payload()) != self.seal_hash:
            raise ValueError("Frozen research plan changed")


def _estimate_case(spec: ResearchEstimatePlanningSpec, reference: np.ndarray) -> NativeSpatialCase:
    spec.assert_intact()
    region = tuple(slice(a, b) for a, b in zip(spec.roi_start, spec.roi_stop))
    image = next(s.image for s in spec.sources if s.modality == spec.actor_modality)[region]
    transform = np.eye(4)
    transform[:3, 3] = spec.roi_start
    affine = spec.sources[0].affine_ras_mm @ transform
    return NativeSpatialCase(image, spec.support.mask[region], reference[region],
        affine, spec.access, spec.tools, track="inference_only",
        support_source_kind="derived_from_scan", support_derivation=(
            "provisional model support " + spec.support.model_sha256 + "; nonclinical; no cortical approval"),
        nominal_target=spec.target.mask[region], target_source_kind="derived_from_scan",
        target_derivation="provisional target " + spec.target.model_sha256 + "; no independent accuracy claim",
        proposal_mode="fixed_lattice")


def _nominal_task(spec: ResearchEstimatePlanningSpec) -> NativeSpatialTask:
    # Never instantiate any rich task with evaluator truth, even transiently.
    return NativeSpatialTask(_estimate_case(spec, spec.target.mask),
                             max_steps=spec.horizon, reward=spec.reward).planning_clone()


def _physical_hash(history: list[dict]) -> str:
    return semantic_digest([{k: v for k, v in row.items() if k not in _OUTCOME_FIELDS} for row in history])


def _qualified_mask(spec: ResearchEstimatePlanningSpec) -> np.ndarray:
    region = tuple(slice(a, b) for a, b in zip(spec.roi_start, spec.roi_stop))
    qualified = np.zeros(spec.support.mask.shape, bool)
    qualified[region] = spec.support.coverage[region] & spec.target.coverage[region]
    return qualified


def _audit_candidate_footprints(spec: ResearchEstimatePlanningSpec, task: NativeSpatialTask) -> bool:
    """Conservative full-tool in-image supercover for every declared fixed ray.

    The shaft/tip union is enclosed by one larger capsule from proximal shaft
    end to distal tip. Audit all declared slots, including presently infeasible
    rays that could become legal after opening a cavity. No reference truth is
    consulted. External approach outside the source grid remains unassessed.
    """
    qualified = _qualified_mask(spec)
    scene = GeometryScene(np.zeros(qualified.shape, bool), spec.sources[0].affine_ras_mm)
    tools = {tool.tool_id: tool for tool in spec.tools}
    for row in task.candidate_inventory()["ledger"]:
        if row.get("entry_mm") is None or row.get("tip_mm") is None:
            continue
        entry, tip = np.asarray(row["entry_mm"], float), np.asarray(row["tip_mm"], float)
        displacement = tip - entry
        length = float(np.linalg.norm(displacement))
        if length <= 0:
            continue
        tool = tools[row["tool_id"]]
        proximal = entry - tool.working_length_mm * displacement / length
        touched = capsule_voxel_indices(scene, proximal, tip,
            max(tool.tip_radius_mm, tool.shaft_radius_mm))
        if len(touched) and not qualified[tuple(touched.T)].all():
            return False
    return True


def _preflight_nominal_task(spec: ResearchEstimatePlanningSpec) -> NativeSpatialTask | Abstention:
    """One permitted-input gate shared by plan creation and private evaluation."""
    spec.assert_intact()
    if any(e.qc_status != "pass" for e in (spec.support, spec.target)):
        return Abstention("ESTIMATE_QC_INCOMPLETE", spec.fingerprint)
    qualified = _qualified_mask(spec)
    region = tuple(slice(a, b) for a, b in zip(spec.roi_start, spec.roi_stop))
    if not qualified[region].all():
        return Abstention("PARTIAL_SPATIAL_COVERAGE", spec.fingerprint)
    if np.any(spec.target.mask & ~qualified):
        return Abstention("TARGET_OUTSIDE_QUALIFIED_ROI", spec.fingerprint)
    if (not spec.support.mask[region].any() or not spec.target.mask[region].any()
            or np.any(spec.target.mask[region] & ~spec.support.mask[region])):
        return Abstention("TARGET_SUPPORT_INCONSISTENT", spec.fingerprint)
    task = _nominal_task(spec)
    if not _audit_candidate_footprints(spec, task):
        return Abstention("TOOL_FOOTPRINT_OUTSIDE_COVERAGE", spec.fingerprint)
    return task


def research_planning_from_estimates(spec: ResearchEstimatePlanningSpec, *, method: str,
        configuration_hash: str, planner: Callable[[NativeSpatialTask], tuple[tuple[str, ...], Mapping]]) -> FrozenResearchPlan | Abstention:
    """One method gets one fresh nominal-only task, or an explicit abstention.

    The declaration is content binding, not authentication of split/source or
    model ancestry. Callers must supply an independently reviewed receipt before
    using patient inputs in a future promoted implementation.
    """
    if type(spec) is not ResearchEstimatePlanningSpec or not callable(planner):
        raise ValueError("Exact estimate spec and planner callback are required")
    spec.assert_intact()
    _hash(configuration_hash, "method configuration")
    if method not in {"SEARCH", "IL", "RL", "HYBRID"}:
        raise ValueError("Only SEARCH, IL, RL and HYBRID are comparable methods")
    initial = _preflight_nominal_task(spec)
    if type(initial) is Abstention:
        return initial
    observation_hash = initial.observation().fingerprint
    action_ids, accounting = planner(initial)
    spec.assert_intact()
    if not isinstance(action_ids, (list, tuple)) or any(type(a) is not str for a in action_ids):
        raise ValueError("Planner must return explicit ordered action IDs")
    action_ids = tuple(action_ids)
    replay = _nominal_task(spec)
    for action in action_ids:
        replay.advance_planning(action)
    if not replay.terminated:
        raise ValueError("Only a complete nominal route can seal")
    return FrozenResearchPlan(spec.fingerprint, method, configuration_hash, observation_hash,
        replay.decision_model_hash, action_ids, _physical_hash(replay.metrics()["history"]),
        spec.horizon, accounting)


def evaluate_sealed_research_plan(plan: FrozenResearchPlan, spec: ResearchEstimatePlanningSpec,
        *, load_reference: Callable[[], np.ndarray]) -> Mapping:
    """Target-only generated diagnostic; load withheld truth after nominal replay.

    This prototype has no authentic patient reference binding or process/file
    isolation, so it cannot certify a patient evaluation.
    """
    if type(plan) is not FrozenResearchPlan or type(spec) is not ResearchEstimatePlanningSpec:
        raise ValueError("Exact sealed plan and research input spec are required")
    plan.assert_intact()
    spec.assert_intact()
    nominal = _preflight_nominal_task(spec)
    if type(nominal) is Abstention:
        raise ValueError("Research estimate abstains before private reference load: " + nominal.reason)
    if (plan.input_hash != spec.fingerprint or plan.initial_observation_hash != nominal.observation().fingerprint
            or plan.decision_model_hash != nominal.decision_model_hash or plan.horizon != spec.horizon):
        raise ValueError("Plan/source/observation binding mismatch before private reference load")
    for action in plan.action_ids:
        nominal.advance_planning(action)
    if not nominal.terminated or _physical_hash(nominal.metrics()["history"]) != plan.physical_history_hash:
        raise ValueError("Nominal physical replay mismatch before private reference load")
    if not callable(load_reference):
        raise ValueError("Evaluator-owned zero-argument loader required")
    reference = np.asarray(load_reference())
    if (reference.shape != spec.target.mask.shape or reference.dtype.kind not in "biuf"
            or not np.isfinite(reference).all() or not np.isin(reference, (0, 1)).all()):
        return freeze_json({"status": "reference_rejected", "plan_seal_hash": plan.seal_hash,
            "coverage_scope": plan.coverage_scope,
            "outside_source_fov_tool_feasibility_assessed": False,
            "clinical_use_permitted": False})
    evaluator = NativeSpatialTask(_estimate_case(spec, reference), max_steps=spec.horizon, reward=spec.reward)
    for action in plan.action_ids:
        evaluator.step(action)
    if _physical_hash(evaluator.metrics()["history"]) != plan.physical_history_hash:
        raise ValueError("Private reference changed physical route geometry")
    return freeze_json({"status": "generated_target_only_diagnostic", "plan_seal_hash": plan.seal_hash,
        "reference_hash": array_digest(reference), "physical_history_hash": plan.physical_history_hash,
        "target_removed_mm3": evaluator.metrics()["target_removed_mm3"],
        "coverage_scope": plan.coverage_scope,
        "outside_source_fov_tool_feasibility_assessed": False,
        "clinical_use_permitted": False,
        "clinical_deficit_probability": None, "patient_generalization": None})


_STRATEGY_RECORD_VERSION = "research-estimate-strategy-record-v1"


def _replay_strategy_nominal(plan: FrozenResearchPlan,
        spec: ResearchEstimatePlanningSpec) -> NativeSpatialTask:
    """Recover the existing sealed route without any reference loader."""
    if type(plan) is not FrozenResearchPlan or type(spec) is not ResearchEstimatePlanningSpec:
        raise ValueError("Exact sealed plan and research input spec are required")
    if type(plan.horizon) is not int:
        raise ValueError("Strategy replay requires an exact integer horizon")
    plan.assert_intact()
    spec.assert_intact()
    task = _preflight_nominal_task(spec)
    if type(task) is Abstention:
        raise ValueError("Research estimate abstains before strategy replay: " + task.reason)
    if (plan.input_hash != spec.fingerprint or plan.initial_observation_hash != task.observation().fingerprint
            or plan.decision_model_hash != task.decision_model_hash or plan.horizon != spec.horizon):
        raise ValueError("Plan/source/observation binding mismatch before strategy replay")
    for action in plan.action_ids:
        task.advance_planning(action)
    if not task.terminated or _physical_hash(task.metrics()["history"]) != plan.physical_history_hash:
        raise ValueError("Nominal physical replay mismatch")
    return task


def _strategy_record(plan: FrozenResearchPlan, task: NativeSpatialTask) -> dict:
    """Sparse state changes plus exact bound support reconstruct the modeled state.

    These are consequences of a nominal simulation, not observed operative
    cavity, retained tissue, or measured instrument contact. Native indices use
    the declared planning ROI grid; each stroke also carries its physical affine.
    No evaluator target, outcome score, future observation, or private loader is
    read or exported. The original plan seal binds every physical history row.
    """
    history = [{key: value for key, value in row.items() if key not in _OUTCOME_FIELDS}
               for row in task.metrics()["history"]]
    removed = {tuple(cell) for row in history for cell in row.get("removed_indices_native", ())}
    contact = {tuple(cell) for row in history for cell in row.get("contact_indices_native", ())}
    current_tool = next((row["tool_id"] for row in reversed(history) if "tool_id" in row), None)
    record = {"version": _STRATEGY_RECORD_VERSION,
        "plan": {**plan._payload(), "seal_hash": plan.seal_hash},
        "physical_history": history,
        "terminal_state": {
            "lineage": "nominal_simulation_replay_not_observed_patient_state",
            "steps_taken": len(history), "remaining_steps": plan.horizon - len(history),
            "terminated": True, "terminal_reason": "STOP" if plan.action_ids[-1] == "STOP" else "HORIZON",
            "current_tool_id": current_tool,
            "source_shape": list(task.case.structural_intensity.shape),
            "affine_ras_mm": task.case.affine_ras_mm.tolist(),
            "removed_indices_native": sorted(removed),
            "contact_indices_native": sorted(contact),
            "retained_contact_indices_native": sorted(contact - removed)},
        "recording_accounting": {"nominal_replay_transition_calls": len(history)}}
    return thaw_json(freeze_json(record))


def research_strategy_to_record(plan: FrozenResearchPlan,
        spec: ResearchEstimatePlanningSpec) -> dict:
    """Export the complete sealed nominal strategy as detached JSON data.

    One extra nominal replay is needed to recover full microsteps and state
    changes from the existing plan hash. Its transitions are recording overhead,
    separately reported from caller-declared online method accounting.
    """
    return _strategy_record(plan, _replay_strategy_nominal(plan, spec))


def research_strategy_from_record(record: Mapping,
        spec: ResearchEstimatePlanningSpec) -> FrozenResearchPlan:
    """Round-trip the existing plan after checking its full recorded replay.

    Import executes one fresh nominal replay and returns the original plan type.
    It neither accepts nor invokes an evaluator loader. Exact content checks are
    accidental-corruption protection, not authentication or clinical admission.
    """
    expected = {"version", "plan", "physical_history", "terminal_state", "recording_accounting"}
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError("Exact research strategy record fields are required")
    if record["version"] != _STRATEGY_RECORD_VERSION or not isinstance(record["plan"], Mapping):
        raise ValueError("Unsupported research strategy record")
    payload = dict(record["plan"])
    init_names = {item.name for item in fields(FrozenResearchPlan) if item.init}
    fixed_names = {item.name for item in fields(FrozenResearchPlan) if not item.init}
    if set(payload) != init_names | fixed_names:
        raise ValueError("Exact frozen research plan fields are required")
    arguments = {name: payload[name] for name in init_names}
    arguments["action_ids"] = tuple(arguments["action_ids"])
    plan = FrozenResearchPlan(**arguments)
    if semantic_digest(payload) != semantic_digest({**plan._payload(), "seal_hash": plan.seal_hash}):
        raise ValueError("Frozen research plan record changed")
    expected_record = research_strategy_to_record(plan, spec)
    if semantic_digest(record) != semantic_digest(expected_record):
        raise ValueError("Recorded physical strategy or resulting state differs from nominal replay")
    return plan
