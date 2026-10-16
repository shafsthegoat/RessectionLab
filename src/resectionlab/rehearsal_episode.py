"""One source-bound, post-exposure research episode for desktop and planning.

This is deliberately a voxel-cell removal/field-management model, not calibrated
surgical physics. The initial exposure is an explicit hypothesis, separately
accounted from episode removal. Source arrays are never changed. No patient
outcome, hidden evaluator label, or generated vessel is a planning input.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import math
import re
from collections.abc import Mapping

import numpy as np

from .core import CaseData, array_digest, semantic_digest, thaw_json, freeze_json, immutable_array
from .critical_evidence import resolve_critical_evidence
from .evaluation import independent_check_motion
from .geometry import (GeometryScene, ToolGeometry, ToolPose, capsule_voxel_indices,
                       check_motion, check_pose, interpolate_pose, tool_capsules,
                       segment_segment_distance)
from .structural_evidence import structural_frame_hash

SCHEMA = "resectionlab.source-bound-rehearsal.v2"
AUTHORITY = "cached_estimates_research_only"
TOOLS = (ToolGeometry("aspirator", 1.8, .65, 100., 60., 2.),
         ToolGeometry("field_manager", .8, .55, 100., 60., 1.5))
MODES = {"aspirator": "aspiration", "field_manager": "field_management"}
LIMITS = {"maxActionSeconds": 8., "maxEpisodeSeconds": 120., "maxSteps": 128,
          "maxSpeedMmPerSecond": 4., "maxMotionMm": 12., "microstepMm": .5,
          "microstepSeconds": .5}
UNKNOWN = ("unobserved_vessels_remain_unknown", "motor_function_unknown",
           "language_function_unknown", "exposure_assumed_not_operatively_demonstrated",
           "uncalibrated_removal_and_field_management", "no_brain_shift_or_tissue_forces")


def _json(value):
    return thaw_json(freeze_json(value))


def _pose(pose):
    return {"tipRasMm": pose.tip_mm.tolist(), "axis": pose.axis_unit.tolist()}


def _flat(indices, shape):
    a = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    return set() if not len(a) else set(map(int, np.ravel_multi_index(a.T, shape)))


def _cells(ids, shape):
    return [] if not ids else np.column_stack(np.unravel_index(sorted(ids), shape)).tolist()


def _scalar(value, name, low=0., high=math.inf):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _synchronized_collision(tools, starts, ends, intervals):
    """Enclose both moving full tools at synchronized interval midpoints.

    For fixed-axis translation the half-interval tip displacement bounds every
    centerline point. Inflating BOTH tool radii encloses every simultaneous pose;
    endpoint-only checks or treating the second moving tool as stationary do not.
    """
    ids = sorted(tools)
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            a, b = tools[first], tools[second]
            da = float(np.linalg.norm(ends[first].tip_mm - starts[first].tip_mm)) / (2 * intervals)
            db = float(np.linalg.norm(ends[second].tip_mm - starts[second].tip_mm)) / (2 * intervals)
            for n in range(intervals):
                t = (n + .5) / intervals
                pa, pb = interpolate_pose(starts[first], ends[first], t), interpolate_pose(starts[second], ends[second], t)
                for a0, a1, ar, _ in tool_capsules(a, pa, da):
                    for b0, b1, br, _ in tool_capsules(b, pb, db):
                        if segment_segment_distance(a0, a1, b0, b1) <= ar + br + 1e-9:
                            return f"TOOL_COLLISION: swept full tools {first} and {second}"
    return None


def _require_development_source(case):
    """Metadata-only exclusion before geometry or source-mask consumption.

    Role records can be nested in split/cohort/source bindings. In particular,
    BTC's outer role "development" does not supersede its locked inner
    "checkpoint_selection_development" role. Prose reasons and eligibility
    booleans are not role declarations. No image/path is opened by this check.
    """
    role_keys = {"role", "roles", "split_role", "development_role", "outer_role",
                 "benchmark_role", "source_role", "cohort_role", "data_role",
                 "usage_role", "partition", "split"}
    identity_keys = {"case_id", "patient_id", "patient_group", "participant_id",
                     "subject", "canonical_person"}
    roles, identities = [], [case.case_id]
    restricted = False

    def visit(value, path=(), *, role_payload=False, identity_payload=False):
        nonlocal restricted
        if isinstance(value, Mapping):
            for key, item in value.items():
                key = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(key)).lower()
                if key in {"restricted", "is_restricted", "is_holdout"} and item is True:
                    restricted = True
                split_value = bool(path) and path[-1] in {"split", "partition"} and key in {"name", "value", "code", "label"}
                declared_role = key in role_keys or split_value or (key == "purpose" and "split" in path)
                if (role_payload or "split" in path) and item is True:
                    roles.append(key)
                # A split object contains comments/counts as well as its role;
                # inspect named role fields, not every word in that object.
                if key in {"split", "partition"} and isinstance(item, Mapping):
                    declared_role = False
                structured_value = role_payload and key not in {"reason", "rationale", "description", "note", "notes"}
                visit(item, (*path, key), role_payload=declared_role or structured_value,
                      identity_payload=key in identity_keys or identity_payload)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, path, role_payload=role_payload, identity_payload=identity_payload)
        elif isinstance(value, str):
            if role_payload:
                roles.append(value)
            if identity_payload:
                identities.append(value)

    visit(case.metadata)
    collection = case.metadata.get("source_collection", {})
    identities.append(str(collection.get("name", "")) if isinstance(collection, Mapping) else str(collection))
    tokens = {token for role in roles for token in re.findall(r"[a-z0-9]+", role.casefold())}
    forbidden_roles = {"select", "selection", "eval", "evaluation", "test", "testing", "val", "validation",
                       "holdout", "heldout", "final", "stress", "restricted", "unopened", "transfer"}
    identity = " ".join(identities).casefold()
    # Preserve the already fixed SELECT people even if an old bundle omits a
    # role or a caller relabels it TRAIN. These canonical IDs are established in
    # btc-spatial-development-cohort-v1 and remind-component-cohort-v1 metadata.
    locked_select = (re.search(r"\bremind[-_: ]0*(?:13|37)(?![0-9])", identity) is not None
                     or ("btc" in identity and re.search(r"\b(?:sub[-_: ]?)?pat0*(?:26|27)(?![0-9])", identity) is not None))
    if (restricted or "upenn" in identity or locked_select or tokens & forbidden_roles
            or {"held", "out"}.issubset(tokens)):
        raise ValueError("Protected cohort/split/source is not admitted to development rehearsal")


def _source_inputs(case, target_names, support_evidence_id, authority):
    if type(case) is not CaseData or authority != AUTHORITY:
        raise ValueError("Exact CaseData and explicit cached_estimates_research_only authority required")
    _require_development_source(case)
    if case.frame != "RAS+":
        raise ValueError("Rehearsal requires a source-bound RAS+ case; no implicit frame conversion")
    spacing = np.linalg.norm(case.affine[:3, :3], axis=0)
    if not np.allclose(case.affine[:3, :3].T @ case.affine[:3, :3], np.diag(spacing ** 2), atol=1e-8):
        raise ValueError("This bounded independent checker requires an orthogonal source grid")
    if support_evidence_id is None:
        support = case.brain_mask
        support_record = _json(case.metadata.get("anatomy_layers", {}).get("brain", {"provenance": "supplied"}))
    else:
        evidence = case.structural_evidence[support_evidence_id]
        evidence.assert_matches(case)
        support, support_record = evidence.mask, evidence.to_manifest()
    if support is None or not support.any():
        raise ValueError("A supplied or explicitly authorized estimated brain support is required")
    layers = case.metadata.get("anatomy_layers", {})
    if target_names is None:
        if "monai_tumor_core" in case.compartments:
            target_names = ("monai_tumor_core",)
        elif "enhancing_target" in case.compartments:
            target_names = ("enhancing_target",)
        else:
            target_names = tuple(k for k in case.compartments if layers.get(k, {}).get("kind") == "tumor")
    target_names = tuple(target_names)
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("Select a nonempty unique list of visible research target layers")
    for name in target_names:
        if (name not in case.compartments or any(x in name.lower() for x in ("privileged", "evaluator", "ground_truth", "future", "reference"))
                or layers.get(name, {}).get("kind", "tumor") != "tumor"):
            raise ValueError("Only named visible target compartments are admitted; privileged inputs are forbidden")
    target = np.logical_or.reduce([case.compartments[k] for k in target_names])
    if not np.any(target & support):
        raise ValueError("Selected source target has no overlap with brain support")
    critical = resolve_critical_evidence(case)
    forbidden = np.zeros(support.shape, bool)
    constraint_records = {}
    for name, mask in critical.masks.items():
        if mask is not None:
            forbidden |= mask
            constraint_records[name] = {"maskHash": array_digest(mask), "authority": "existing_critical_evidence_admission"}
    # Explicit research authority is narrow: cached masks must be attached to
    # this immutable case, with their own source and inference provenance.
    for name, layer in layers.items():
        if name in case.compartments and layer.get("kind") in {"skull", "vessel", "vessels"}:
            if not layer.get("source_sha256") or not layer.get("source_role"):
                raise ValueError("Research exclusion masks require source-bound layer provenance")
            forbidden |= case.compartments[name]
            constraint_records[name] = {"maskHash": array_digest(case.compartments[name]), "authority": AUTHORITY,
                                        "provenance": _json(layer)}
    permitted = {"imageHash": array_digest(case.mri), "frameHash": structural_frame_hash(case),
                 "supportHash": array_digest(support), "targetHash": array_digest(target),
                 "targetNames": list(target_names), "forbiddenHash": array_digest(forbidden),
                 "authority": authority}
    binding = {**permitted, "caseHash": case.semantic_hash, "caseId": case.case_id,
               "shape": list(case.mri.shape), "affine": case.affine.tolist(), "affineRasMm": case.affine.tolist(), "frame": case.frame,
               "sourceRefs": [r.to_dict() for r in case.source_refs],
               "supportEvidence": support_record, "targetEvidence": {k: _json(layers.get(k, {"provenance": "supplied_visible_annotation"})) for k in target_names},
               "constraints": constraint_records, "planningInputHash": semantic_digest(permitted),
               "privateReferencePublished": False}
    return support, target, forbidden, binding, target_names


def _initial_geometry(target, scene, tools):
    """Deterministic visible-target column; no assessment of surgical access.

    The two instruments begin held in an assumed exposed field. This function
    does not simulate gaining exposure or label an image-derived entry safe.
    """
    candidates = []
    spacing = np.linalg.norm(scene.affine[:3, :3], axis=0)
    for axis in range(3):
        summed = target.sum(axis=axis)
        column = np.unravel_index(int(np.argmax(summed)), summed.shape)
        selector = list(column)
        selector.insert(axis, slice(None))
        active = np.flatnonzero(target[tuple(selector)])
        groups = np.split(active, np.flatnonzero(np.diff(active) != 1) + 1)
        run = max(groups, key=len)
        point = list(column)
        point.insert(axis, int(run[0]))
        candidates.append((len(run) * spacing[axis], axis, np.asarray(point, float)))
    _, axis, index = max(candidates, key=lambda x: (x[0], -x[1]))
    direction = scene.affine[:3, axis] / spacing[axis]
    lateral = scene.affine[:3, (axis + 1) % 3] / spacing[(axis + 1) % 3]
    start = scene.voxel_to_world(index) - direction * (tools["aspirator"].tip_radius_mm + 1.5 * max(spacing))
    poses = {"aspirator": ToolPose(start, direction), "field_manager": ToolPose(start + 6. * lateral, direction)}
    # A bounded cylindrical research exposure up to a flat plane. It is not a
    # craniotomy or actual prior resection, and never mutates source anatomy.
    indices = capsule_voxel_indices(scene, start - 100. * direction, start, 9.)
    centers = scene.voxel_to_world(indices)
    exposure = indices[(centers - start) @ direction <= tools["aspirator"].tip_radius_mm + .5 * max(spacing) + 1e-8]
    return poses, _flat(exposure, scene.shape), direction, lateral


class RehearsalEpisode:
    """Mutable authoritative sparse state, immutable source, atomic actions."""

    def __init__(self, case, *, research_authority, target_names=None, support_evidence_id=None):
        support, target, forbidden, binding, names = _source_inputs(case, target_names, support_evidence_id, research_authority)
        self.case, self.support, self.target, self.source_binding = case, support, immutable_array(target), freeze_json(binding)
        self.tools = {tool.tool_id: tool for tool in TOOLS}
        self.scene = GeometryScene(forbidden, case.affine, unknowns=UNKNOWN)
        self.config = freeze_json({"research_authority": research_authority, "target_names": list(names), "support_evidence_id": support_evidence_id})
        self.poses, assumed, self.direction, self.lateral = _initial_geometry(target & support, self.scene, self.tools)
        self.assumed_exposure = frozenset(i for i in assumed if support.ravel()[i])
        self.initial_poses = dict(self.poses)
        for key, tool in self.tools.items():
            checked = check_pose(tool, self.poses[key], self.scene,
                                 other_tools=[(v, self.poses[k]) for k, v in self.tools.items() if k != key])
            if not checked.feasible:
                raise ValueError("Initial post-exposure hypothesis conflicts with observed anatomy: " + checked.failures[0].reason)
        self.removed, self.contact, self.field_managed = set(), {}, {}
        self.active = {key: False for key in self.tools}
        self.elapsed = 0.
        self.field_burden = 0.
        self.pending_events = []
        self.last_inspection = None
        self.terminated, self.stop_reason = False, None
        self.actions, self.history, self.frames = [], [], []
        self._support_count = int(support.sum())
        self._target_count = int(target.sum())
        self._target_outside_support_count = int(np.count_nonzero(target & ~support))
        self._assumed_target_count = sum(bool(target.ravel()[i]) for i in self.assumed_exposure)
        self._initial_state_id = self.state_id
        self.frames.append(self._frame("initial", -1, self.state_id))

    @property
    def state_id(self):
        return semantic_digest({"schema": SCHEMA, "planningInputHash": self.source_binding["planningInputHash"],
            "config": self.config, "exposure": sorted(self.assumed_exposure), "elapsed": self.elapsed,
            "tools": self._tool_records(), "removed": sorted(self.removed), "contact": sorted(self.contact.items()),
            "field_managed": sorted(self.field_managed.items()), "fieldBurden": self.field_burden,
            "pendingEvents": self.pending_events, "lastInspection": self.last_inspection,
            "terminated": self.terminated, "stopReason": self.stop_reason, "actionCount": len(self.actions)})

    def _tool_records(self):
        return [{"toolId": key, "mode": MODES[key], **_pose(self.poses[key]), "active": self.active[key],
                 "geometry": asdict(tool), "workingLengthMm": tool.working_length_mm,
                 "tipLengthMm": tool.tip_length_mm, "shaftRadiusMm": tool.shaft_radius_mm,
                 "tipRadiusMm": tool.tip_radius_mm} for key, tool in self.tools.items()]

    def metrics(self):
        target_removed = sum(bool(self.target.ravel()[i]) for i in self.removed)
        voxel = float(abs(np.linalg.det(self.case.affine[:3, :3])))
        return {"targetRemovedMm3": target_removed * voxel, "nonTargetRemovedMm3": (len(self.removed) - target_removed) * voxel,
                "removedMm3": len(self.removed) * voxel,
                "residualTargetMm3": (self._target_count - self._assumed_target_count - target_removed) * voxel,
                "sourceTargetMm3": self._target_count * voxel, "targetOutsideSupportMm3": self._target_outside_support_count * voxel, "assumedExposureTargetMm3": self._assumed_target_count * voxel,
                "assumedExposureTissueMm3": len(self.assumed_exposure) * voxel,
                "remainingTissueMm3": (self._support_count - len(self.assumed_exposure) - len(self.removed)) * voxel,
                "sourceTissueMm3": self._support_count * voxel, "contactCellSeconds": sum(self.contact.values()),
                "fieldManagementCellSeconds": sum(self.field_managed.values()), "fieldBurden": self.field_burden,
                "elapsedSeconds": self.elapsed, "vascularEvents": None, "motorRisk": None, "languageRisk": None,
                "anatomicalConflicts": 0, "stopReason": self.stop_reason, "terminated": self.terminated}

    def _frame(self, kind, action_index, before, removed=(), touched=(), field_managed=()):
        return {"frameIndex": len(self.frames), "actionIndex": action_index, "kind": kind,
                "elapsedSeconds": self.elapsed, "tools": self._tool_records(),
                "removedIndicesNative": _cells(removed, self.scene.shape),
                "contactIndicesNative": _cells(touched, self.scene.shape),
                "fieldContactIndicesNative": _cells(field_managed, self.scene.shape),
                "stateBefore": before, "stateAfter": self.state_id, "metrics": self.metrics(),
                "pendingEvents": _json(self.pending_events)}

    def fork(self):
        # Immutable image/masks and geometry are shared; mutable episode state is not.
        result = copy.copy(self)
        for key in ("tools", "poses", "active", "contact", "field_managed"):
            setattr(result, key, dict(getattr(self, key)))
        result.removed = set(self.removed)
        for key in ("pending_events", "actions", "history", "frames"):
            setattr(result, key, copy.deepcopy(getattr(self, key)))
        return result

    def planning_observation(self):
        """Whitelisted visible state; no raw CaseData/context/reference source."""
        return {"schema": SCHEMA, "planningInputHash": self.source_binding["planningInputHash"],
                "stateId": self.state_id, "tools": self._tool_records(), "metrics": self.metrics(),
                "elapsedSeconds": self.elapsed, "pendingEvents": _json(self.pending_events),
                "lastInspectionSeconds": self.last_inspection, "unknowns": list(UNKNOWN),
                "targetNames": list(self.config["target_names"]), "suggestedActions": self.suggested_actions()}

    def suggested_actions(self, tool_id=None):
        if self.terminated:
            return []
        if tool_id is not None and tool_id not in self.tools:
            raise ValueError("Unknown persistent tool")
        records = [{"id": kind, "label": label, "action": {"kind": kind, "durationSeconds": duration}}
                   for kind, label, duration in (("inspect", "Inspect current field", 1.), ("reassess", "Reassess approach", 1.), ("STOP", "STOP episode", 0.))]
        for key in ([tool_id] if tool_id else self.tools):
            current = self.poses[key]
            for offset, label in ((1., "Move inward 1 mm"), (-1., "Withdraw 1 mm")):
                records.append({"id": f"{key}:move:{offset}", "label": label,
                    "action": {"kind": "move", "toolId": key, "durationSeconds": .5,
                               "tipRasMm": (current.tip_mm + offset * current.axis_unit).tolist()}})
            action = {"kind": "aspirate" if key == "aspirator" else "clear_field", "toolId": key, "durationSeconds": 1.}
            action["tipRasMm"] = (current.tip_mm + (1. if key == "aspirator" else .5) * current.axis_unit).tolist()
            records.append({"id": key + ":activate", "label": "Aspirate inward 1 mm" if key == "aspirator" else "Clear field / manage field 1 s", "action": action})
            records.append({"id": key + ":exchange", "label": "Exchange withdrawn instrument", "action": {"kind": "exchange", "toolId": key, "durationSeconds": 1.}})
        return records

    def _validate(self, action):
        if not isinstance(action, Mapping):
            raise ValueError("Action must be a JSON object")
        allowed = {"kind", "toolId", "durationSeconds", "tipRasMm", "axis", "motions", "reason"}
        if set(action) - allowed:
            raise ValueError("Unknown action fields; private/reference inputs are forbidden")
        kind = action.get("kind")
        if kind not in {"move", "aspirate", "clear_field", "inspect", "reassess", "exchange", "STOP"}:
            raise ValueError("Unknown bounded action")
        if self.terminated:
            raise ValueError("Episode has stopped")
        if len(self.actions) >= LIMITS["maxSteps"] and kind != "STOP":
            raise ValueError("Episode action budget exhausted; STOP required")
        duration = _scalar(action.get("durationSeconds", 0. if kind == "STOP" else 1.), "durationSeconds", 0., LIMITS["maxActionSeconds"])
        if (duration == 0 and kind != "STOP") or self.elapsed + duration > LIMITS["maxEpisodeSeconds"]:
            raise ValueError("Positive action duration and bounded shared clock required")
        ends = dict(self.poses)
        key = action.get("toolId")
        if kind in {"move", "aspirate", "clear_field", "exchange"} and key not in self.tools and "motions" not in action:
            raise ValueError("Select one of the two persistent tools")
        if kind == "aspirate" and key != "aspirator" or kind == "clear_field" and key != "field_manager":
            raise ValueError("Instrument does not support the requested activation")
        if "motions" in action:
            if kind != "move" or key is not None or "tipRasMm" in action or "axis" in action:
                raise ValueError("Synchronized motions are inactive moves with one shared duration")
            motions = action["motions"]
            if (not isinstance(motions, list) or not 1 <= len(motions) <= 2
                    or any(not isinstance(x, Mapping) for x in motions)
                    or len({x.get("toolId") for x in motions}) != len(motions)):
                raise ValueError("One or two distinct synchronized tool motions required")
        elif "tipRasMm" in action or "axis" in action:
            if kind not in {"move", "aspirate", "clear_field"}:
                raise ValueError("Information/exchange/STOP actions cannot move instruments")
            motions = [{k: action[k] for k in ("toolId", "tipRasMm", "axis") if k in action}]
        else:
            motions = []
        for motion in motions:
            if not isinstance(motion, Mapping) or set(motion) - {"toolId", "tipRasMm", "axis"} or motion.get("toolId") not in self.tools:
                raise ValueError("Invalid motion fields")
            who = motion["toolId"]
            start = self.poses[who]
            end = ToolPose(motion.get("tipRasMm", start.tip_mm), motion.get("axis", start.axis_unit))
            if not np.allclose(end.axis_unit, start.axis_unit, atol=1e-12, rtol=0):
                raise ValueError("Bounded episode supports continuous translation with fixed instrument axes only")
            distance = float(np.linalg.norm(end.tip_mm - start.tip_mm))
            if distance > LIMITS["maxMotionMm"] or distance > LIMITS["maxSpeedMmPerSecond"] * duration + 1e-9:
                raise ValueError("Motion exceeds declared distance or speed bound; no teleportation")
            if kind == "aspirate":
                displacement = end.tip_mm - start.tip_mm
                advance = float(displacement @ start.axis_unit)
                if advance < -1e-9 or np.linalg.norm(displacement - advance * start.axis_unit) > 1e-9:
                    raise ValueError("Activated aspiration supports forward axial motion only")
            ends[who] = end
        if kind == "exchange" and np.linalg.norm(self.poses[key].tip_mm - self.initial_poses[key].tip_mm) > 1e-8:
            raise ValueError("Exchange is permitted only at the initial withdrawn pose; no teleportation")
        if "reason" in action and (not isinstance(action["reason"], str) or not 1 <= len(action["reason"]) <= 512):
            raise ValueError("Stop reason must be bounded text")
        return kind, duration, key, ends

    def step(self, action):
        """Check/execute on a transaction clone; a rejected action changes nothing."""
        before = self.state_id
        try:
            if self.case.semantic_hash != self.source_binding["caseHash"]:
                raise ValueError("Source case identity changed")
            kind, duration, key, ends = self._validate(action)
            trial = self.fork()
            first_frame = len(trial.frames)
            trial._execute(_json(action), kind, duration, key, ends)
        except (ValueError, TypeError, KeyError) as error:
            return {"accepted": False, "reason": str(error), "stateId": before, "frames": []}
        self.__dict__.update(trial.__dict__)
        return {"accepted": True, "stateId": self.state_id, "frames": copy.deepcopy(self.frames[first_frame:]),
                "frame": copy.deepcopy(self.frames[-1])}

    def _execute(self, action, kind, duration, key, ends):
        starts = dict(self.poses)
        action_index = len(self.actions)
        action_before = self.state_id
        action_start_seconds = self.elapsed
        max_distance = max(np.linalg.norm(ends[k].tip_mm - starts[k].tip_mm) for k in self.tools)
        count = max(1, math.ceil(max_distance / LIMITS["microstepMm"]), math.ceil(duration / LIMITS["microstepSeconds"]))
        conflict = _synchronized_collision(self.tools, starts, ends, count)
        if conflict:
            raise ValueError(conflict)
        # Both implementations certify the entire path against source-bound hard
        # exclusions. Ordinary tissue is then checked per microstep as it changes.
        audits = []
        for who, tool in self.tools.items():
            check = check_motion(tool, starts[who], ends[who], self.scene)
            independent = independent_check_motion(tool, starts[who], ends[who], self.scene)
            if not check.feasible or not independent.feasible:
                raise ValueError("ANATOMICAL_COLLISION: " + (check.failures[0].reason if check.failures else str(independent.failures)))
            audits.append({"toolId": who, "primaryFeasible": True, "independentFeasible": True})
        self.actions.append(action)
        if kind == "STOP":
            self.terminated, self.stop_reason = True, action.get("reason", "operator_STOP_under_incomplete_anatomy")
        for n in range(count):
            before = action_before if n == 0 else self.state_id
            old = dict(self.poses)
            new = {who: interpolate_pose(starts[who], ends[who], (n + 1) / count) for who in self.tools}
            dt = duration / count
            removed, touched, field_managed = set(), set(), set()
            self.active = {who: who == key and kind in {"aspirate", "clear_field"} for who in self.tools}
            for who, tool in self.tools.items():
                p0, p1 = old[who], new[who]
                active_tip = tool_capsules(tool, p1)[1]
                if who == key and kind == "aspirate":
                    # Axial active capsule union is exact; whole intersected cells
                    # are removed. This is voxel research physics, not tissue mass.
                    indices = capsule_voxel_indices(self.scene, p0.tip_mm - tool.tip_length_mm * p0.axis_unit,
                                                    p1.tip_mm, tool.tip_radius_mm)
                    cells = _flat(indices, self.scene.shape)
                    removed |= {i for i in cells if self.support.ravel()[i]} - self.assumed_exposure - self.removed
                    touched |= removed
                elif who == key and kind == "clear_field":
                    # Local field-management footprint/exposure proxy. This
                    # represents suction/irrigation-style clearing without tissue
                    # removal; no coagulation, vessel sealing, or fluid mass claim.
                    indices = capsule_voxel_indices(self.scene, active_tip[0], active_tip[1], 2.5)
                    field_managed |= {i for i in _flat(indices, self.scene.shape) if self.support.ravel()[i]} - self.assumed_exposure - self.removed
                    touched |= field_managed
            self.removed.update(removed)
            for who, tool in self.tools.items():
                p0, p1 = old[who], new[who]
                # Full passive tool must lie in known empty/assumed-exposed cells;
                # active aspirator's shaft may only follow its just-cleared tip.
                checked_tool = tool
                if who == key and kind == "aspirate":
                    checked_tool = replace(tool, tip_radius_mm=tool.shaft_radius_mm)
                swept = check_motion(checked_tool, p0, p1, self.scene).swept_voxel_indices
                blocked = {i for i in _flat(swept, self.scene.shape) if self.support.ravel()[i]} - self.assumed_exposure - self.removed
                if blocked:
                    raise ValueError("RETAINED_TISSUE_COLLISION: full shaft/tip cannot move through uncleared tissue")
            for i in touched:
                self.contact[i] = self.contact.get(i, 0.) + dt
            for i in field_managed:
                self.field_managed[i] = self.field_managed.get(i, 0.) + dt
            # Dimensionless debris/visibility proxy; it is explicitly not blood.
            self.field_burden = min(1., self.field_burden + len(removed) * .002)
            if kind == "clear_field" and field_managed:
                self.field_burden = max(0., self.field_burden - .2 * dt)
            self.poses = new
            self.elapsed = round(action_start_seconds + (n + 1) * duration / count, 12)
            if kind in {"inspect", "reassess"}:
                self.last_inspection = self.elapsed
            self.pending_events = ([{"kind": "modeled_field_debris", "severity": self.field_burden,
                                     "meaning": "generated visibility burden; not observed bleeding"}] if self.field_burden > 0 else [])
            if n == count - 1:
                self.active = {who: False for who in self.tools}
            self.frames.append(self._frame(kind, action_index, before, removed, touched, field_managed))
        self.history.append({"actionIndex": action_index, "action": action, "stateBefore": action_before,
                             "stateAfter": self.state_id, "startSeconds": self.elapsed - duration,
                             "endSeconds": self.elapsed, "geometryAudit": audits, "metrics": self.metrics()})

    def serialize(self):
        record = {"schema": SCHEMA, "sourceBinding": _json(self.source_binding), "config": _json(self.config),
                  "actions": _json(self.actions), "initialStateId": self._initial_state_id, "finalStateId": self.state_id}
        record["recordHash"] = semantic_digest(record)
        return record

    def export(self):
        return {"schema": SCHEMA, "episodeId": self.state_id, "caseHash": self.case.semantic_hash,
                "sourceBinding": _json(self.source_binding), "initialStateId": self._initial_state_id,
                "finalStateId": self.state_id, "frame": "RAS+", "physicalUnits": "mm", "shape": list(self.scene.shape),
                "affine": self.case.affine.tolist(), "tools": self._tool_records(),
                "replayFrames": copy.deepcopy(self.frames), "history": copy.deepcopy(self.history),
                "finalRemovedIndicesNative": _cells(self.removed, self.scene.shape),
                "assumedExposureIndicesNative": _cells(self.assumed_exposure, self.scene.shape),
                "exposure": {"label": "Research post-exposure hypothesis", "observed": False,
                             "meaning": "Assumed pre-existing exposed field; separate from episode removal; no validated surgical access"},
                "metrics": self.metrics(), "unknowns": sorted(set(UNKNOWN) | set(self.case.unknowns)), "limits": dict(LIMITS),
                "physics": "uncalibrated voxel-cell aspiration; nonremoving local field-management exposure and field-debris proxy",
                "clinicalValidation": False, "vascularInjuryModel": "unavailable", "anatomyAuthority": AUTHORITY,
                "suggestedActions": self.suggested_actions(), "savedEpisode": self.serialize()}


def create_episode(case: CaseData, *, research_authority=AUTHORITY, target_names=None, support_evidence_id=None):
    return RehearsalEpisode(case, research_authority=research_authority,
                            target_names=target_names, support_evidence_id=support_evidence_id)


def restore_episode(case: CaseData, record):
    if not isinstance(record, Mapping) or set(record) != {"schema", "sourceBinding", "config", "actions", "initialStateId", "finalStateId", "recordHash"}:
        raise ValueError("Invalid saved episode schema")
    body = dict(record)
    seal = body.pop("recordHash")
    if record["schema"] != SCHEMA or semantic_digest(body) != seal:
        raise ValueError("Saved episode integrity check failed")
    result = create_episode(case, **record["config"])
    if _json(result.source_binding) != _json(record["sourceBinding"]) or result.state_id != record["initialStateId"]:
        raise ValueError("Saved episode source identity, frame, or initial state differs")
    if (not isinstance(record["actions"], list) or len(record["actions"]) > LIMITS["maxSteps"] + 1
            or (len(record["actions"]) == LIMITS["maxSteps"] + 1 and record["actions"][-1].get("kind") != "STOP")):
        raise ValueError("Invalid saved action budget")
    for action in record["actions"]:
        accepted = result.step(action)
        if not accepted["accepted"]:
            raise ValueError("Saved action no longer executes: " + accepted["reason"])
    if result.state_id != record["finalStateId"]:
        raise ValueError("Saved episode replay does not reproduce its exact final state")
    return result


def make_alternative_strategies(episode):
    """Two bounded action programs computed only from current visible state.

    Both are executable by the same runtime. Failed continuation is followed by
    explicit reassessment/STOP, retaining all accepted prefix state and reason.
    """
    if episode.terminated:
        raise ValueError("Branch from a saved state before STOP")
    pose = episode.poses["aspirator"]
    common = [{"kind": "inspect", "durationSeconds": 1.}]
    short = [*common, {"kind": "aspirate", "toolId": "aspirator", "durationSeconds": 2.,
                      "tipRasMm": (pose.tip_mm + 2. * pose.axis_unit).tolist()},
             {"kind": "clear_field", "toolId": "field_manager", "durationSeconds": 2.,
              "tipRasMm": (episode.poses["field_manager"].tip_mm + .5 * episode.poses["field_manager"].axis_unit).tolist()},
             {"kind": "reassess", "durationSeconds": 1.},
             {"kind": "STOP", "durationSeconds": 0., "reason": "bounded_short_pass_then_field_management; vessels_and_function_unknown"}]
    long = [*common, {"kind": "aspirate", "toolId": "aspirator", "durationSeconds": 4.,
                     "tipRasMm": (pose.tip_mm + 6. * pose.axis_unit).tolist()},
            {"kind": "reassess", "durationSeconds": 1.},
            {"kind": "STOP", "durationSeconds": 0., "reason": "bounded_deeper_pass_limit; vessels_and_function_unknown"}]
    return [{"id": "short_pass_field_management", "label": "Short pass + field management", "actions": short},
            {"id": "deeper_pass_reassess", "label": "Deeper pass + reassess", "actions": long}]


def execute_alternatives(episode):
    snapshot = episode.serialize()
    initial = episode.state_id
    branches = []
    for strategy in make_alternative_strategies(episode):
        branch = episode.fork()
        diagnostics = []
        for action in strategy["actions"]:
            result = branch.step(action)
            if not result["accepted"]:
                diagnostics.append({"action": action, "reason": result["reason"], "stateId": result["stateId"]})
                stopped = branch.step({"kind": "STOP", "durationSeconds": 0., "reason": "anatomical_or_motion_constraint: " + result["reason"][:400]})
                if not stopped["accepted"]:
                    raise ValueError("Failed branch could not STOP: " + stopped["reason"])
                break
        branches.append({"id": strategy["id"], "label": strategy["label"], "branchPointStateId": initial,
                         "branchPointFrameIndex": len(episode.frames) - 1, "strategy": strategy,
                         "attemptDiagnostics": diagnostics, "episode": branch.export()})
    if episode.state_id != initial:
        raise RuntimeError("Alternative execution mutated the common saved snapshot")
    return {"schema": SCHEMA, "snapshot": snapshot, "branchPointStateId": initial,
            "identicalSnapshot": True, "branches": branches}
