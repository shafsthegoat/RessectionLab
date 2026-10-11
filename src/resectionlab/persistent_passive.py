"""Opt-in generated passive tool continuity on the shared NativeSpatialTask.

This is a bounded geometry/software control, not a surgical or sensor model.
The inherited native strokes remain atomic insert/interact/withdraw actions and
are deliberately unavailable while this passive instrument is retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
import json
import math

import numpy as np

from .core import freeze_json, semantic_digest
from .evaluation import independent_check_motion
from .geometry import AccessWindow, GeometryScene, ToolGeometry, ToolPose, check_motion, check_pose
from .native_spatial_task import NativeSpatialCase, NativeSpatialStep, NativeSpatialTask
from .sequential_spatial_observation import SequentialSpatialObservation


VERSION = "generated-persistent-passive-v1"
OBSERVATION_VERSION = "generated-persistent-passive-observation-v1"
DECISION_VERSION = "generated-persistent-passive-decision-v2"
DECISION_OBSERVATION_VERSION = "generated-persistent-passive-decision-observation-v2"
MOVE = "PASSIVE_MOVE"
INSPECT = "ACQUIRE_GENERATED_STATUS"
ADVANCE = "PASSIVE_ADVANCE"
WITHDRAW = "PASSIVE_WITHDRAW"
WITHDRAW_STATIONARY = "PASSIVE_WITHDRAW_STATIONARY"
STOP = "STOP"


def _pose_record(pose: ToolPose | None):
    return None if pose is None else {"tip_mm": [float(x) for x in pose.tip_mm],
                                     "axis_unit": [float(x) for x in pose.axis_unit]}


@dataclass(frozen=True)
class PassiveProtocol:
    moving_tool_id: str
    start_pose: ToolPose
    hold_pose: ToolPose
    stationary_tool_id: str | None = None
    stationary_pose: ToolPose | None = None
    stationary_exit_pose: ToolPose | None = None
    move_seconds: float = 1.0
    inspect_seconds: float = 1.0
    withdraw_seconds: float = 1.0
    withdraw_stationary_seconds: float = 1.0
    stop_seconds: float = 0.0

    def record(self):
        if (len({self.stationary_tool_id is None, self.stationary_pose is None,
                 self.stationary_exit_pose is None}) != 1
                or self.stationary_tool_id == self.moving_tool_id):
            raise ValueError("A stationary tool needs a distinct ID, retained pose and exit pose")
        durations = (self.move_seconds, self.inspect_seconds,
                     self.withdraw_seconds, self.withdraw_stationary_seconds, self.stop_seconds)
        if (any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in durations)
                or sum(durations) > 60.):
            raise ValueError("Generated schedule must be finite, nonnegative and at most 60 seconds")
        if not all(type(x) is ToolPose for x in (self.start_pose, self.hold_pose)):
            raise TypeError("Passive endpoints require exact ToolPose objects")
        if self.stationary_pose is not None and type(self.stationary_pose) is not ToolPose:
            raise TypeError("Stationary instrument requires an exact ToolPose")
        if self.stationary_exit_pose is not None and type(self.stationary_exit_pose) is not ToolPose:
            raise TypeError("Stationary exit requires an exact ToolPose")
        if np.array_equal(self.start_pose.tip_mm, self.hold_pose.tip_mm) and np.array_equal(
                self.start_pose.axis_unit, self.hold_pose.axis_unit):
            raise ValueError("The passive segment must actually move")
        displacement = np.asarray(self.hold_pose.tip_mm) - np.asarray(self.start_pose.tip_mm)
        axis = np.asarray(self.start_pose.axis_unit)
        if (not np.array_equal(axis, self.hold_pose.axis_unit)
                or float(displacement @ axis) <= 0
                or np.linalg.norm(displacement - float(displacement @ axis) * axis) > 1e-9):
            raise ValueError("The first passive version accepts only inward fixed-axis motion")
        if self.stationary_pose is not None:
            displacement = np.asarray(self.stationary_exit_pose.tip_mm) - np.asarray(self.stationary_pose.tip_mm)
            axis = np.asarray(self.stationary_pose.axis_unit)
            if (not np.array_equal(axis, self.stationary_exit_pose.axis_unit)
                    or float(displacement @ axis) >= 0
                    or np.linalg.norm(displacement - float(displacement @ axis) * axis) > 1e-9):
                raise ValueError("Stationary withdrawal must be outward fixed-axis motion")
        return {"version": VERSION, "moving_tool_id": self.moving_tool_id,
                "start_pose": _pose_record(self.start_pose), "hold_pose": _pose_record(self.hold_pose),
                "stationary_tool_id": self.stationary_tool_id,
                "stationary_pose": _pose_record(self.stationary_pose),
                "stationary_exit_pose": _pose_record(self.stationary_exit_pose),
                "generated_seconds": list(durations),
                "time_meaning": "declared generated schedule; not measured surgical duration",
                "movement_scope": "one passive rigid tool, other tool stationary; no active cutting"}


@dataclass(frozen=True)
class GeneratedInspectionInput:
    source_hash: str
    status: str

    def __post_init__(self):
        if self.status not in {"verified", "unresolved"} or not self.source_hash.startswith("sha256:"):
            raise ValueError("Inspection is an explicit generated status, not a sensor or diagnosis")

    @property
    def fingerprint(self):
        return semantic_digest({"version": "generated-registration-status-v1",
                                "source_hash": self.source_hash, "status": self.status})


@dataclass(frozen=True)
class PassiveDecisionProtocol:
    """Public navigation waypoint and generated registration decision rule."""
    target_nav_mm: tuple[float, float, float]
    max_error_bound_mm: float = .25
    advance_seconds: float = .5

    def record(self):
        waypoint = tuple(self.target_nav_mm)
        if (len(waypoint) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in waypoint)
                or type(self.max_error_bound_mm) not in (int, float)
                or not math.isfinite(self.max_error_bound_mm) or not 0 <= self.max_error_bound_mm <= 1.
                or type(self.advance_seconds) not in (int, float)
                or not math.isfinite(self.advance_seconds) or not 0 <= self.advance_seconds <= 10.):
            raise ValueError("Finite bounded generated navigation decision protocol required")
        return {"version": DECISION_VERSION, "target_navigation_ras_mm": [float(x) for x in waypoint],
                "max_declared_registration_error_mm": float(self.max_error_bound_mm),
                "advance_generated_seconds": float(self.advance_seconds),
                "mapping": "source_RAS=target_navigation_RAS+observed_nav_to_source_translation_RAS",
                "uncertainty_model": "known_exact_held_start; nominal_axial_command_with_fixed_orientation; generated_endpoint_translation_error_within_declared_Euclidean_bound; interpolated_passive_segment_may_deviate_laterally; no_rotation_or_deformation_error",
                "rule": "after_reveal_only; nominal_axial_forward_with_forward_margin_greater_than_error; error<=public_cap; inflate_moving_tool_radii_by_error_before_full_motion_check; separately_bound_tip_image_domain_and_access_depth_reach",
                "scope": "generated registration/known geometry only; unknown anatomy remains unknown"}


@dataclass(frozen=True)
class GeneratedRegistrationMeasurement:
    """Private until inspection; no patient sensor or anatomical warp claim."""
    source_hash: str
    translation_ras_mm: tuple[float, float, float] | None
    worst_case_error_bound_mm: float | None

    def record(self):
        available = self.translation_ras_mm is not None
        if not isinstance(self.source_hash, str) or not self.source_hash.startswith("sha256:"):
            raise ValueError("Generated registration must bind its source")
        if available != (self.worst_case_error_bound_mm is not None):
            raise ValueError("Unavailable registration must retain unknown translation and error")
        if available:
            values = tuple(self.translation_ras_mm)
            bound = self.worst_case_error_bound_mm
            if (len(values) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 10.
                                        for x in values)
                    or type(bound) not in (int, float) or not math.isfinite(bound) or not 0 <= bound <= 10.):
                raise ValueError("Finite bounded generated RAS registration required")
        return {"version": "generated-navigation-registration-measurement-v1",
                "sourceHash": self.source_hash, "available": available,
                "translationNavToSourceRasMm": None if not available else [float(x) for x in values],
                "worstCaseErrorBoundMm": None if not available else float(bound),
                "frame": "RAS+", "units": "mm", "coverage": "generated_input_only_no_anatomical_coverage_claim",
                "evidence": "generated_software_measurement_not_patient_sensor",
                "anatomicalWarpSupported": False}

    @property
    def fingerprint(self):
        return semantic_digest(self.record())


@dataclass(frozen=True)
class PassiveObservation:
    base: SequentialSpatialObservation
    phase: str
    elapsed_generated_seconds: float
    inspection_status: str
    moving_tool_pose: dict
    stationary_tool_pose: dict | None
    action_ids: tuple[str, ...]
    action_mask: tuple[bool, ...]
    action_tool_ids: tuple[str | None, ...]
    observation_version: str = OBSERVATION_VERSION
    registration_observation: dict | None = None
    candidate_decision: dict | None = None
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "moving_tool_pose", freeze_json(self.moving_tool_pose))
        object.__setattr__(self, "stationary_tool_pose", freeze_json(self.stationary_tool_pose))
        object.__setattr__(self, "registration_observation", freeze_json(self.registration_observation))
        object.__setattr__(self, "candidate_decision", freeze_json(self.candidate_decision))
        object.__setattr__(self, "_identity", self._digest())

    def _digest(self):
        body = {"version": self.observation_version, "base": self.base.fingerprint,
            "phase": self.phase, "elapsed_generated_seconds": self.elapsed_generated_seconds,
            "inspection_status": self.inspection_status,
            "moving_tool_pose": self.moving_tool_pose,
            "stationary_tool_pose": self.stationary_tool_pose,
            "action_ids": self.action_ids, "action_mask": self.action_mask,
            "action_tool_ids": self.action_tool_ids}
        if self.observation_version != OBSERVATION_VERSION:
            body.update(registration_observation=self.registration_observation,
                        candidate_decision=self.candidate_decision)
        return semantic_digest(body)

    def assert_intact(self):
        self.base.assert_intact()
        if (not self.action_ids or self.action_ids[0] != STOP
                or len(self.action_ids) != len(self.action_mask)
                or len(self.action_ids) != len(self.action_tool_ids)
                or self._digest() != self._identity):
            raise ValueError("Passive observation action inventory changed")

    @property
    def fingerprint(self):
        self.assert_intact()
        return self._identity


class PersistentPassiveTask(NativeSpatialTask):
    """One checked retained passive segment in the canonical task state seal.

    This adapter intentionally offers no SEARCH/planning-clone or trained-policy
    admission: their observation/hidden-status contracts need separate review.
    """

    def __init__(self, case: NativeSpatialCase, protocol: PassiveProtocol,
                 inspection: GeneratedInspectionInput, *, max_steps=None, cancelled=None,
                 decision_protocol: PassiveDecisionProtocol | None = None,
                 registration_measurement: GeneratedRegistrationMeasurement | None = None):
        decision = type(decision_protocol) is PassiveDecisionProtocol
        expected_steps = (5 if type(protocol) is PassiveProtocol and protocol.stationary_pose is not None else 4) + int(decision)
        if (type(case) is not NativeSpatialCase or case.track != "synthetic_scan"
                or type(protocol) is not PassiveProtocol or type(inspection) is not GeneratedInspectionInput
                or inspection.source_hash != case.source_hash
                or (decision_protocol is not None and not decision)
                or (registration_measurement is None) != (decision_protocol is None)
                or (decision and (type(registration_measurement) is not GeneratedRegistrationMeasurement
                                  or registration_measurement.source_hash != case.source_hash
                                  or inspection.status != ("verified" if registration_measurement.translation_ras_mm is not None
                                                           else "unresolved")))
                or max_steps not in (None, expected_steps)):
            raise ValueError("The passive episode has a fixed bounded generated source")
        tool_ids = {tool.tool_id for tool in case.tools}
        if (len(tool_ids) != 2 or protocol.moving_tool_id not in tool_ids
                or (protocol.stationary_tool_id is not None and protocol.stationary_tool_id not in tool_ids)):
            raise ValueError("The protocol must bind the two declared native tools")
        protocol.record()
        if decision:
            decision_protocol.record()
            registration_measurement.record()
            if sum(protocol.record()["generated_seconds"]) + decision_protocol.advance_seconds > 60.:
                raise ValueError("Generated decision schedule exceeds 60 seconds")
        self._protocol = protocol
        self._inspection_input = inspection
        self._decision_protocol = decision_protocol
        self._registration_input = registration_measurement
        self._visible_registration = None
        self._advance_pose = None
        self._phase = "ready"
        self._elapsed = 0.0
        self._visible_status = "not_acquired"
        super().__init__(case, max_steps=expected_steps, cancelled=cancelled,
            tool_modes={tool.tool_id: ("probe" if tool.tool_id == protocol.moving_tool_id
                                       else "aspirate") for tool in case.tools})
        self._check_start()

    def _contract_record(self):
        body = {**super()._contract_record(), "persistent_passive": self._protocol.record()}
        if self._decision_protocol is not None:
            body["passive_decision"] = self._decision_protocol.record()
        return body

    def _operative_record(self):
        pose = (self._advance_pose if self._phase == "advanced" else
                self._protocol.hold_pose if self._phase in {"held", "inspected"} else self._protocol.start_pose)
        other_pose = (self._protocol.stationary_exit_pose if self._phase in {"both_withdrawn", "stopped"}
                      else self._protocol.stationary_pose)
        moving_held = self._phase in {"held", "inspected", "advanced"}
        other_held = self._protocol.stationary_pose is not None and self._phase not in {"both_withdrawn", "stopped"}
        body = {"phase": self._phase, "elapsed_generated_seconds": self._elapsed,
                "moving_tool": {"tool_id": self._protocol.moving_tool_id,
                    "pose": _pose_record(pose), "activation": "passive_only" if moving_held else "off",
                    "retained_at_depth": moving_held,
                    "pose_status": "held_in_declared_void" if moving_held else "access_pose_recorded",
                    "physical_absence_claim": False},
                "other_tool": {"tool_id": self._protocol.stationary_tool_id,
                    "pose": _pose_record(other_pose),
                    "activation": "off", "retained_at_depth": other_held,
                    "pose_status": ("held_in_declared_void" if other_held else "access_pose_recorded")
                                   if other_pose is not None else "not_present_in_this_protocol",
                    "physical_absence_claim": False},
                "inspection_status": self._visible_status}
        if self._decision_protocol is not None:
            body["registration_observation"] = copy.deepcopy(self._visible_registration)
        return body

    def _state_record(self):
        body = {**super()._state_record(), "operative": self._operative_record(),
                "private_generated_inspection_binding": self._inspection_input.fingerprint}
        if self._decision_protocol is not None:
            body["private_generated_registration_binding"] = self._registration_input.fingerprint
            body["committed_advance_pose"] = _pose_record(self._advance_pose)
        return body

    def reset(self, seed=0):
        if type(seed) is not int or seed != 0:
            raise ValueError("Only the declared generated scenario is supported")
        if hasattr(self, "_state_seal"):
            self._assert_frozen()
        self._check_cancelled()
        self._engine.reset()
        self._steps, self._total_reward, self._current_tool = 0, 0.0, None
        self._terminated, self._history = False, []
        self._inventory, self._ledger, self._proposal_batch = None, (), None
        self._phase, self._elapsed, self._visible_status = "ready", 0.0, "not_acquired"
        self._visible_registration, self._advance_pose = None, None
        self._seal()
        return self.observation()

    def _scene(self):
        config = self._config
        forbidden = self._engine.remaining_mask | config.hard_exclusion
        if config.interaction_domain is not None:
            forbidden = forbidden | ~config.interaction_domain
        return GeometryScene(forbidden, config.affine, exposure_mask=self._engine.remaining_mask)

    def _other_tools(self):
        if self._protocol.stationary_pose is None or self._phase in {"both_withdrawn", "stopped"}:
            return ()
        tool = next(tool for tool in self.case.tools if tool.tool_id == self._protocol.stationary_tool_id)
        return ((tool, self._protocol.stationary_pose),)

    def _moving_tool(self):
        return next(tool for tool in self.case.tools if tool.tool_id == self._protocol.moving_tool_id)

    def _check_start(self):
        scene = self._scene()
        start = self._protocol.start_pose
        result = check_pose(self._moving_tool(), start, scene, self.case.access, self._other_tools())
        if not result.feasible:
            raise ValueError("Initial retained tool pose is not certified: " + result.failures[0].reason)
        if self._other_tools():
            other_tool, other_pose = self._other_tools()[0]
            result = check_pose(other_tool, other_pose, scene, self.case.access,
                                ((self._moving_tool(), start),))
            if not result.feasible:
                raise ValueError("Initial stationary tool pose is not certified: " + result.failures[0].reason)

    def _advance_preview(self):
        """Only the already-revealed generated registration may form a candidate."""
        if self._decision_protocol is None or self._visible_registration is None:
            return None, None
        measurement = self._visible_registration
        public = self._decision_protocol.record()
        base = {"version": DECISION_VERSION, "measurementAvailable": measurement["available"],
                "candidateSourceRasMm": None, "declaredErrorBoundMm": measurement["worstCaseErrorBoundMm"],
                "publicMaxErrorBoundMm": public["max_declared_registration_error_mm"],
                "inflatedToolRadiiMm": None, "knownGeometryClearanceMm": None,
                "geometryUnknowns": [], "eligible": False,
                "uncertaintyScope": "generated_endpoint_translation_bound_from_known_held_start_nominal_axial_command_no_angular_or_deformation_error"}
        if not measurement["available"]:
            return {**base, "reason": "registration_measurement_unavailable"}, None
        candidate = np.asarray(public["target_navigation_ras_mm"], float) + np.asarray(
            measurement["translationNavToSourceRasMm"], float)
        base["candidateSourceRasMm"] = candidate.tolist()
        bound = measurement["worstCaseErrorBoundMm"]
        if bound > public["max_declared_registration_error_mm"]:
            return {**base, "reason": "declared_error_bound_exceeds_public_limit"}, None
        delta = candidate - self._protocol.hold_pose.tip_mm
        axis = self._protocol.hold_pose.axis_unit
        distance = float(delta @ axis)
        if distance <= bound or np.linalg.norm(delta - distance * axis) > 1e-9:
            return {**base, "reason": "mapped_waypoint_not_forward_on_fixed_axis"}, None
        end = ToolPose(candidate, axis)
        tool = self._moving_tool()
        access = self.case.access
        alignment = float(axis @ access.normal_inward)
        depth = float((candidate - access.center_mm) @ access.normal_inward)
        if (alignment <= 0 or depth - bound < 0
                or (depth + bound) / alignment > tool.working_length_mm):
            return {**base, "reason": "bounded_translation_violates_access_direction_or_reach"}, None
        scene = self._scene()
        voxel = scene.world_to_voxel(candidate)
        voxel_bound = np.linalg.norm(np.linalg.inv(scene.affine)[:3, :3], axis=1) * bound
        if (np.any(voxel - voxel_bound < -.5)
                or np.any(voxel + voxel_bound > np.asarray(scene.shape) - .5)):
            return {**base, "reason": "bounded_translation_leaves_source_image_tip_domain"}, None
        inflated = replace(tool, tip_radius_mm=tool.tip_radius_mm + bound,
                           shaft_radius_mm=tool.shaft_radius_mm + bound)
        base["inflatedToolRadiiMm"] = {"tip": inflated.tip_radius_mm, "shaft": inflated.shaft_radius_mm}
        geometry = check_motion(inflated, self._protocol.hold_pose, end,
                                scene, access, other_tools=self._other_tools())
        base["geometryUnknowns"] = list(geometry.unknowns)
        base["knownGeometryClearanceMm"] = geometry.clearance_mm
        if not geometry.feasible:
            return {**base, "reason": "native_full_tool_geometry_refused",
                    "nativeFailure": geometry.failures[0].reason}, None
        return {**base, "eligible": True, "reason": "generated_registration_rule_and_known_geometry_passed"}, end

    def observation(self):
        self._assert_frozen()
        base = super().observation()
        if type(base) is not SequentialSpatialObservation:
            raise RuntimeError("The passive protocol lost its canonical mixed-tool observation")
        ids = (STOP, MOVE if self._phase == "ready" else
                     INSPECT if self._phase == "held" else
                     WITHDRAW if self._phase in {"inspected", "advanced"} else
                     WITHDRAW_STATIONARY if self._phase == "withdrawn" and self._protocol.stationary_pose is not None
                     else STOP)
        if ids[1] == STOP:
            ids = (STOP,)
        candidate = None
        if self._decision_protocol is not None and self._phase == "inspected":
            candidate, _ = self._advance_preview()
            if candidate["eligible"]:
                ids += (ADVANCE,)
        # STOP follows explicit withdrawal of every retained instrument.
        mask = (self._phase == ("both_withdrawn" if self._protocol.stationary_pose is not None else "withdrawn"),) + (
            (True,) * (len(ids)-1))
        operative = self._operative_record()
        tool_ids = (None,) + tuple(self._protocol.stationary_tool_id if action == WITHDRAW_STATIONARY
                                   else self._protocol.moving_tool_id for action in ids[1:])
        return PassiveObservation(base, self._phase, self._elapsed, self._visible_status,
            operative["moving_tool"]["pose"], operative["other_tool"]["pose"],
            ids, mask, tool_ids,
            observation_version=DECISION_OBSERVATION_VERSION if self._decision_protocol is not None else OBSERVATION_VERSION,
            registration_observation=self._visible_registration,
            candidate_decision=candidate)

    def step(self, action: str):
        self._assert_frozen()
        if self._terminated:
            raise ValueError("The passive episode has terminated")
        observed = self.observation()
        if type(action) is not str or action not in observed.action_ids or not observed.action_mask[observed.action_ids.index(action)]:
            raise ValueError("Unknown, stale or unsupported passive action")
        self._check_cancelled()
        prior_phase = self._phase
        before = semantic_digest(self._operative_record())
        source_state = self._engine.state_hash
        geometry = None
        if action in {MOVE, WITHDRAW, WITHDRAW_STATIONARY, ADVANCE}:
            if action == WITHDRAW_STATIONARY:
                start, end = self._protocol.stationary_pose, self._protocol.stationary_exit_pose
                tool = next(tool for tool in self.case.tools if tool.tool_id == self._protocol.stationary_tool_id)
                # The moving tool has returned to its recorded access pose.
                # Conservatively retain that full envelope for this pair check.
                other_tools = ((self._moving_tool(), self._protocol.start_pose),)
            elif action == ADVANCE:
                candidate, end = self._advance_preview()
                if candidate is None or not candidate["eligible"] or end is None:
                    raise ValueError("Unrevealed or unsupported registration cannot authorize passive advance")
                start = self._protocol.hold_pose
                tool, other_tools = self._moving_tool(), self._other_tools()
            else:
                start, end = ((self._protocol.start_pose, self._protocol.hold_pose) if action == MOVE
                              else (self._advance_pose if self._phase == "advanced" else self._protocol.hold_pose,
                                    self._protocol.start_pose))
                tool, other_tools = self._moving_tool(), self._other_tools()
            geometry = check_motion(tool, start, end, self._scene(),
                                    self.case.access, other_tools=other_tools)
            if not geometry.feasible:
                raise ValueError("Passive motion refused: " + geometry.failures[0].reason)
            # This independent checker covers tissue/hard exclusion and access,
            # but not the stationary second tool. Pair clearance above is still
            # solely the primary check_motion certificate.
            independent = independent_check_motion(tool, start, end,
                                                   self._scene(), self.case.access)
            if not independent.feasible:
                raise ValueError("Independent passive tissue motion refused: " + independent.failures[0])
            self._check_cancelled()  # No operative mutation has happened yet.
            self._phase = ("held" if action == MOVE else "advanced" if action == ADVANCE else
                           "both_withdrawn" if action == WITHDRAW_STATIONARY else "withdrawn")
            if action == ADVANCE:
                self._advance_pose = end
            duration = (self._protocol.move_seconds if action == MOVE else
                        self._decision_protocol.advance_seconds if action == ADVANCE else
                        self._protocol.withdraw_stationary_seconds if action == WITHDRAW_STATIONARY
                        else self._protocol.withdraw_seconds)
        elif action == INSPECT:
            self._phase = "inspected"
            self._visible_status = self._inspection_input.status
            if self._decision_protocol is not None:
                self._visible_registration = self._registration_input.record()
            duration = self._protocol.inspect_seconds
        else:
            self._phase = "stopped"
            self._terminated = True
            duration = self._protocol.stop_seconds
        observed_candidate = next((row["information_event"]["candidate_decision"]
                                   for row in self._history if row["action_id"] == INSPECT
                                   and row["information_event"] is not None
                                   and row["information_event"].get("candidate_decision") is not None), None)
        committed_advance = any(row["action_id"] == ADVANCE for row in self._history)
        self._elapsed += duration
        record = {"action_id": action, "interaction_mode": "passive_motion" if geometry is not None else
                  "generated_information" if action == INSPECT else "stop",
            "tool_id": None if action == STOP else (self._protocol.stationary_tool_id
                       if action == WITHDRAW_STATIONARY else self._protocol.moving_tool_id),
            "pose_before": _pose_record(start if geometry is not None else
                          self._advance_pose if prior_phase == "advanced" else
                          self._protocol.hold_pose if prior_phase in {"held", "inspected"}
                          else self._protocol.start_pose),
            "pose_after": _pose_record(end if geometry is not None else
                         self._advance_pose if self._phase == "advanced" else
                         self._protocol.hold_pose if self._phase in {"held", "inspected"}
                         else self._protocol.start_pose),
            "duration_generated_seconds": duration, "elapsed_generated_seconds": self._elapsed,
            "time_source": "fixed_generated_schedule_not_measured_surgical_time",
            "source_state_hash": source_state, "result_state_hash": self._engine.state_hash,
            "operative_before": before, "operative_after": semantic_digest(self._operative_record()),
            "geometry": None if geometry is None else geometry.to_dict(),
            "independent_tissue_geometry": None if geometry is None else {
                "feasible": independent.feasible, "checker_version": independent.checker_version,
                "unknowns": list(independent.unknowns),
                "stationary_tool_pair_independently_checked": False},
            "information_event": None if action != INSPECT else (
                {"version": "generated-navigation-registration-observation-v2",
                 "measurement": copy.deepcopy(self._visible_registration),
                 "measurement_binding": self._registration_input.fingerprint,
                 "candidate_decision": self._advance_preview()[0],
                 "meaning": "generated registration estimate only; no patient sensor, anatomical warp or clinical clearance"}
                if self._decision_protocol is not None else {
                "version": "generated-registration-status-v1", "status": self._visible_status,
                "input_fingerprint": self._inspection_input.fingerprint,
                "meaning": "generated protocol status; no anatomy, sensor or clinical inference"}),
            "stop_reason": (("declared_advance_completed_without_injury_claim" if committed_advance
                              else "declined_eligible_generated_advance" if observed_candidate is not None
                              and observed_candidate["eligible"] else
                              "registration_measurement_unavailable_or_unqualified")
                             if self._decision_protocol is not None else
                             "generated_registration_unresolved" if self._visible_status == "unresolved"
                             else "declared_stop_without_injury_claim") if action == STOP else None,
            "removed_indices_native": [], "contact_indices_native": [], "reward": 0.0,
            "clinical_deficit_probability": None, "physiological_response": None}
        self._history.append(copy.deepcopy(record))
        self._steps += 1
        self._inventory, self._ledger, self._proposal_batch = None, (), None
        self._seal()
        try:
            # As in NativeSpatialTask, cancellation after sealing reports the
            # committed transition rather than an ordinary precommit abort.
            successor = None if self._terminated else self.observation()
            if self._terminated:
                self._check_cancelled()
        except InterruptedError as error:
            from .native_axis_simulation import CommittedTransitionInterrupted
            raise CommittedTransitionInterrupted(record, 0.) from error
        return NativeSpatialStep(successor, 0.0, self._terminated, copy.deepcopy(record))

    def planning_clone(self):
        raise ValueError("No search admission until hidden generated information has a belief contract")

    def clone(self):
        """Isolate revealed measurement state across native branch copies."""
        result = super().clone()
        result._visible_registration = copy.deepcopy(self._visible_registration)
        result._advance_pose = copy.deepcopy(self._advance_pose)
        return result

    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case, self._protocol, self._inspection_input,
                          max_steps=self.max_steps, cancelled=self._cancelled,
                          decision_protocol=self._decision_protocol,
                          registration_measurement=self._registration_input)


def verify_passive_replay(task: PersistentPassiveTask, actions: tuple[str, ...], records: tuple[dict, ...]):
    """Fresh-task replay of the same source-bound action sequence."""
    worker = task.fresh()
    observed = []
    for action in actions:
        observed.append(worker.step(action).info)
    if (not worker.terminated or observed != list(records)
            or worker._engine.state_hash != task._engine.state_hash
            or worker._state_seal != task._state_seal):
        raise ValueError("Passive operative history differs from a fresh native replay")
    return True


def export_passive_episode(task: PersistentPassiveTask):
    """Export a bounded generated replay; no old stroke frame is reused."""
    if type(task) is not PersistentPassiveTask or not task.terminated:
        raise ValueError("Only a completed generated passive episode can be exported")
    task._assert_frozen()
    records = copy.deepcopy(task._history)
    actions = tuple(row["action_id"] for row in records)
    verify_passive_replay(task, actions, tuple(records))
    replay = task.fresh()
    frames = [{"frameIndex": 0, "operative": copy.deepcopy(replay._operative_record())}]
    for index, action in enumerate(actions, start=1):
        replay.step(action)
        frames.append({"frameIndex": index, "operative": copy.deepcopy(replay._operative_record())})
    decision = task._decision_protocol is not None
    body = {"schema": ("resectionlab.generated-persistent-passive-decision-episode.v2" if decision
                       else "resectionlab.generated-persistent-passive-episode.v1"),
            "sourceHash": task.case.source_hash, "decisionModelHash": task.decision_model_hash,
            "protocol": task._protocol.record(), "actions": list(actions),
            "history": records,
            "physicalHistoryCanonicalJson": json.dumps(records, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False),
            "physicalHistoryHash": semantic_digest(records),
            "frames": frames,
            "frameMeaning": "operative state after each committed passive action; not legacy native stroke frames",
            "finalStateHash": task._state_seal,
            "scope": ("generated registration decision and passive geometry only; no patient sensor,"
                      " anatomical warp, surgical or clinical validation" if decision else
                      "generated passive geometry and declared time only; no surgical or sensor validation")}
    if decision:
        body["decisionProtocol"] = task._decision_protocol.record()
        body["revealedRegistration"] = copy.deepcopy(task._visible_registration)
        body["informationEventIndex"] = actions.index(INSPECT)
        body["registrationInputBinding"] = task._registration_input.fingerprint
    return {**body, "episodeId": semantic_digest(body)}


def verify_passive_episode(task: PersistentPassiveTask, episode: dict):
    """Backend-owned exact rehydration check against a new native task replay."""
    if type(episode) is not dict or episode != export_passive_episode(task):
        raise ValueError("Passive exported history or frame state differs from the native task")
    return True


def make_passive_development_task(*, inspection_status="unresolved", paired=False, cancelled=None):
    """Fixed analytic free tunnel through generated tissue, not patient anatomy.

    The 3-by-3 tunnel spans the generated support shell from z=-0.5 through
    z=7.5. At hold depth z=6.5 the complete 6 mm shaft and tip lie inside
    the declared void, without any episode-time material removal.
    No source material is removed to manufacture the void during the episode.
    """
    if type(paired) is not bool or inspection_status not in {"verified", "unresolved"}:
        raise ValueError("Only the declared generated passive fixture is available")
    support = np.zeros((13, 13, 12), dtype=bool)
    support[3:10, 3:10, 0:10] = True
    support[5:8, 5:8, 0:8] = False  # Explicit pre-existing free tunnel.
    nominal = np.zeros_like(support)
    nominal[4, 6, 6] = True
    image = np.where(nominal, .8, np.where(support, .2, 0.)).astype(np.float32)
    other = ToolGeometry("generated-narrow-parked-tool", .15, .1, 6., 30., .4)
    moving = ToolGeometry("generated-passive-probe", .15, .1, 6., 30., .4)
    case = NativeSpatialCase(image, support, nominal, np.eye(4),
        AccessWindow((6., 6., .5), (0., 0., 1.), 2., "generated-free-tunnel-access"),
        (other, moving), track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="analytic 3x3 free tunnel through a generated support shell; not observed anatomy",
        nominal_target=nominal, target_source_kind="derived_from_scan",
        target_derivation="one analytic shell cell for source identity, not an MRI estimate")
    x = 6.5 if paired else 6.
    start, hold = ToolPose((x, 6., .5), (0., 0., 1.)), ToolPose((x, 6., 6.5), (0., 0., 1.))
    protocol = PassiveProtocol(moving.tool_id, start, hold,
        other.tool_id if paired else None,
        ToolPose((5.5, 6., 6.5), (0., 0., 1.)) if paired else None,
        ToolPose((5.5, 6., .5), (0., 0., 1.)) if paired else None)
    return PersistentPassiveTask(case, protocol,
        GeneratedInspectionInput(case.source_hash, inspection_status), cancelled=cancelled)


def run_passive_development_episode(*, inspection_status="unresolved", paired=False, cancelled=None):
    """One fixed generated source execution and fresh replay-checked export."""
    task = make_passive_development_task(inspection_status=inspection_status,
                                          paired=paired, cancelled=cancelled)
    actions = (MOVE, INSPECT, WITHDRAW) + ((WITHDRAW_STATIONARY,) if paired else ()) + (STOP,)
    for action in actions:
        task.step(action)
    return export_passive_episode(task)


def make_passive_decision_task(*, translation_ras_mm=(0., 0., 0.),
                               worst_case_error_bound_mm=.1, paired=True, cancelled=None):
    """Fixed same-source generated measurement alternatives for a passive decision.

    The navigation registration is a declared synthetic input. It is private
    until INSPECT, while the physical source, access, void and tool set stay
    identical across measurement alternatives.
    """
    if type(paired) is not bool:
        raise ValueError("The generated paired-tool choice must be explicit")
    base = make_passive_development_task(paired=paired, cancelled=cancelled)
    old = base._protocol
    x = float(old.hold_pose.tip_mm[0])
    protocol = replace(old, hold_pose=ToolPose((x, 6., 5.5), (0., 0., 1.)))
    measurement = GeneratedRegistrationMeasurement(base.case.source_hash, translation_ras_mm,
                                                     worst_case_error_bound_mm)
    measurement.record()
    inspection = GeneratedInspectionInput(base.case.source_hash,
                                          "verified" if translation_ras_mm is not None else "unresolved")
    return PersistentPassiveTask(base.case, protocol, inspection, cancelled=cancelled,
        decision_protocol=PassiveDecisionProtocol((x, 6., 6.2)),
        registration_measurement=measurement)


def run_passive_decision_episode(*, translation_ras_mm=(0., 0., 0.),
                                 worst_case_error_bound_mm=.1, paired=True,
                                 choose_advance=True, cancelled=None):
    """Observed-only generated policy, with optional offered-advance refusal."""
    if type(choose_advance) is not bool:
        raise ValueError("The generated selector must be explicit")
    task = make_passive_decision_task(translation_ras_mm=translation_ras_mm,
        worst_case_error_bound_mm=worst_case_error_bound_mm, paired=paired, cancelled=cancelled)
    task.step(MOVE)
    task.step(INSPECT)
    observed = task.observation()
    if (choose_advance and ADVANCE in observed.action_ids
            and observed.action_mask[observed.action_ids.index(ADVANCE)]):
        task.step(ADVANCE)
    task.step(WITHDRAW)
    if paired:
        task.step(WITHDRAW_STATIONARY)
    task.step(STOP)
    return export_passive_episode(task)
