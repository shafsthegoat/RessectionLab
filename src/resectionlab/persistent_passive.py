"""Opt-in generated passive tool continuity on the shared NativeSpatialTask.

This is a bounded geometry/software control, not a surgical or sensor model.
The inherited native strokes remain atomic insert/interact/withdraw actions and
are deliberately unavailable while this passive instrument is retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
MOVE = "PASSIVE_MOVE"
INSPECT = "ACQUIRE_GENERATED_STATUS"
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
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "moving_tool_pose", freeze_json(self.moving_tool_pose))
        object.__setattr__(self, "stationary_tool_pose", freeze_json(self.stationary_tool_pose))
        object.__setattr__(self, "_identity", self._digest())

    def _digest(self):
        return semantic_digest({"version": OBSERVATION_VERSION, "base": self.base.fingerprint,
            "phase": self.phase, "elapsed_generated_seconds": self.elapsed_generated_seconds,
            "inspection_status": self.inspection_status,
            "moving_tool_pose": self.moving_tool_pose,
            "stationary_tool_pose": self.stationary_tool_pose,
            "action_ids": self.action_ids, "action_mask": self.action_mask,
            "action_tool_ids": self.action_tool_ids})

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
                 inspection: GeneratedInspectionInput, *, max_steps=None, cancelled=None):
        expected_steps = 5 if type(protocol) is PassiveProtocol and protocol.stationary_pose is not None else 4
        if (type(case) is not NativeSpatialCase or case.track != "synthetic_scan"
                or type(protocol) is not PassiveProtocol or type(inspection) is not GeneratedInspectionInput
                or inspection.source_hash != case.source_hash
                or max_steps not in (None, expected_steps)):
            raise ValueError("The passive episode has a fixed four- or five-action generated source")
        tool_ids = {tool.tool_id for tool in case.tools}
        if (len(tool_ids) != 2 or protocol.moving_tool_id not in tool_ids
                or (protocol.stationary_tool_id is not None and protocol.stationary_tool_id not in tool_ids)):
            raise ValueError("The protocol must bind the two declared native tools")
        protocol.record()
        self._protocol = protocol
        self._inspection_input = inspection
        self._phase = "ready"
        self._elapsed = 0.0
        self._visible_status = "not_acquired"
        super().__init__(case, max_steps=expected_steps, cancelled=cancelled,
            tool_modes={tool.tool_id: ("probe" if tool.tool_id == protocol.moving_tool_id
                                       else "aspirate") for tool in case.tools})
        self._check_start()

    def _contract_record(self):
        return {**super()._contract_record(), "persistent_passive": self._protocol.record()}

    def _operative_record(self):
        pose = self._protocol.hold_pose if self._phase in {"held", "inspected"} else self._protocol.start_pose
        other_pose = (self._protocol.stationary_exit_pose if self._phase in {"both_withdrawn", "stopped"}
                      else self._protocol.stationary_pose)
        moving_held = self._phase in {"held", "inspected"}
        other_held = self._protocol.stationary_pose is not None and self._phase not in {"both_withdrawn", "stopped"}
        return {"phase": self._phase, "elapsed_generated_seconds": self._elapsed,
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

    def _state_record(self):
        return {**super()._state_record(), "operative": self._operative_record(),
                "private_generated_inspection_binding": self._inspection_input.fingerprint}

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

    def observation(self):
        self._assert_frozen()
        base = super().observation()
        if type(base) is not SequentialSpatialObservation:
            raise RuntimeError("The passive protocol lost its canonical mixed-tool observation")
        ids = (STOP, MOVE if self._phase == "ready" else
                     INSPECT if self._phase == "held" else
                     WITHDRAW if self._phase == "inspected" else
                     WITHDRAW_STATIONARY if self._phase == "withdrawn" and self._protocol.stationary_pose is not None
                     else STOP)
        if ids[1] == STOP:
            ids = (STOP,)
        # STOP follows explicit withdrawal of every retained instrument.
        mask = (self._phase == ("both_withdrawn" if self._protocol.stationary_pose is not None else "withdrawn"),) + (
            (True,) if len(ids) == 2 else ())
        operative = self._operative_record()
        return PassiveObservation(base, self._phase, self._elapsed, self._visible_status,
            operative["moving_tool"]["pose"], operative["other_tool"]["pose"],
            ids, mask, (None,) + (((self._protocol.stationary_tool_id if ids[1] == WITHDRAW_STATIONARY
                                   else self._protocol.moving_tool_id),) if len(ids) == 2 else ()))

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
        if action in {MOVE, WITHDRAW, WITHDRAW_STATIONARY}:
            if action == WITHDRAW_STATIONARY:
                start, end = self._protocol.stationary_pose, self._protocol.stationary_exit_pose
                tool = next(tool for tool in self.case.tools if tool.tool_id == self._protocol.stationary_tool_id)
                # The moving tool has returned to its recorded access pose.
                # Conservatively retain that full envelope for this pair check.
                other_tools = ((self._moving_tool(), self._protocol.start_pose),)
            else:
                start, end = ((self._protocol.start_pose, self._protocol.hold_pose) if action == MOVE
                              else (self._protocol.hold_pose, self._protocol.start_pose))
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
            self._phase = ("held" if action == MOVE else "both_withdrawn" if action == WITHDRAW_STATIONARY
                           else "withdrawn")
            duration = (self._protocol.move_seconds if action == MOVE else
                        self._protocol.withdraw_stationary_seconds if action == WITHDRAW_STATIONARY
                        else self._protocol.withdraw_seconds)
        elif action == INSPECT:
            self._phase = "inspected"
            self._visible_status = self._inspection_input.status
            duration = self._protocol.inspect_seconds
        else:
            self._phase = "stopped"
            self._terminated = True
            duration = self._protocol.stop_seconds
        self._elapsed += duration
        record = {"action_id": action, "interaction_mode": "passive_motion" if geometry is not None else
                  "generated_information" if action == INSPECT else "stop",
            "tool_id": None if action == STOP else (self._protocol.stationary_tool_id
                       if action == WITHDRAW_STATIONARY else self._protocol.moving_tool_id),
            "pose_before": _pose_record(start if geometry is not None else
                          self._protocol.hold_pose if prior_phase in {"held", "inspected"}
                          else self._protocol.start_pose),
            "pose_after": _pose_record(end if geometry is not None else
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
            "information_event": None if action != INSPECT else {
                "version": "generated-registration-status-v1", "status": self._visible_status,
                "input_fingerprint": self._inspection_input.fingerprint,
                "meaning": "generated protocol status; no anatomy, sensor or clinical inference"},
            "stop_reason": ("generated_registration_unresolved" if self._visible_status == "unresolved"
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

    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case, self._protocol, self._inspection_input,
                          max_steps=self.max_steps, cancelled=self._cancelled)


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
    body = {"schema": "resectionlab.generated-persistent-passive-episode.v1",
            "sourceHash": task.case.source_hash, "decisionModelHash": task.decision_model_hash,
            "protocol": task._protocol.record(), "actions": list(actions),
            "history": records,
            "physicalHistoryCanonicalJson": json.dumps(records, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False),
            "physicalHistoryHash": semantic_digest(records),
            "frames": frames,
            "frameMeaning": "operative state after each committed passive action; not legacy native stroke frames",
            "finalStateHash": task._state_seal,
            "scope": "generated passive geometry and declared time only; no surgical or sensor validation"}
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
