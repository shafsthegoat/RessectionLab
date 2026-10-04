"""Fail-closed transition accounting for experimental RAW axis training.

This wrapper does not recover a failed episode, alter learner counters, or publish
candidates. Its separate receipt records native commits even when preparing the
next observation (or exporting this receipt) fails. Instances are single-worker.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import weakref

import numpy as np

from .native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from .worlds import WorldPartitionManifest, WorldRole

ACCOUNTING_VERSION = "experimental-native-axis-training-accounting-v1"


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _observation_array_identity(value: Any) -> tuple:
    array = np.asarray(value)
    return array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_receipt(path: Path, value: dict[str, Any]) -> None:
    """Replace a complete JSON receipt; a failed export never truncates its predecessor."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class AccountingReceiptError(RuntimeError):
    """A real transition may exist despite failed receipt export; inspect receipt."""

    def __init__(self, receipt: dict[str, Any]):
        super().__init__("Axis accounting export failed; in-memory receipt retained; run cannot continue")
        self.receipt = copy.deepcopy(receipt)


class AxisTrainingAccounting:
    """Wrap only the exact experimental axis class and declared optimization/selection seeds.

    ``recorded_factory`` is passed unchanged to the generic RAW trainer. Its
    constructor observation is unassigned; only an explicit reset binds a role.
    After any exception this session is closed. Use a new directory for a new run.
    """

    def __init__(self, factory: Callable[[], AxisColumnNativeSimulator],
                 optimization_manifest: WorldPartitionManifest,
                 selection_manifest: WorldPartitionManifest, *,
                 receipt_path: str | Path, expected_model_hash: str):
        manifests = (optimization_manifest, selection_manifest)
        if any(type(item) is not WorldPartitionManifest for item in manifests):
            raise TypeError("Typed immutable world manifests are required")
        if tuple(item.role for item in manifests) != (WorldRole.OPTIMIZATION, WorldRole.SELECTION):
            raise ValueError("Only optimization and selection roles are permitted")
        if (optimization_manifest.case_hash != selection_manifest.case_hash
                or optimization_manifest.generator != selection_manifest.generator
                or optimization_manifest.planning_hash != selection_manifest.planning_hash):
            raise ValueError("Partitions must bind the same source, planning identity and world generator")
        if set(optimization_manifest.seeds) & set(selection_manifest.seeds):
            raise ValueError("Optimization and selection seeds must be disjoint")
        if not isinstance(expected_model_hash, str) or not expected_model_hash:
            raise ValueError("An explicit frozen decision model identity is required")
        self._factory = factory
        self.optimization_manifest, self.selection_manifest = manifests
        self._roles = {seed: item.role.value for item in manifests for seed in item.seeds}
        self.expected_model_hash = expected_model_hash
        self.receipt_path = Path(receipt_path).resolve()
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if self.receipt_path.exists():
            raise FileExistsError("Accounting receipts cannot be resumed or overwritten")
        # Learner episodes own their source-volume copies. Receipts retain scalar
        # histories, never every completed episode's full patient simulator.
        self._live_raw_instances: weakref.WeakSet[AxisColumnNativeSimulator] = weakref.WeakSet()
        self._next_instance = 0
        self._events: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._status = "running"
        self._failure: dict[str, Any] | None = None
        self._export_error: dict[str, Any] | None = None
        self._learner_result: dict[str, Any] | None = None
        self._record_decisions = False
        self._next_decision = 0
        self._active_decision_instance: weakref.ReferenceType | None = None
        self._persist()

    def _require_running(self) -> None:
        if self._status != "running":
            raise RuntimeError("Axis accounting session is closed; implicit continuation is forbidden")

    def _bind(self, simulator: AxisColumnNativeSimulator) -> None:
        if type(simulator) is not AxisColumnNativeSimulator:
            raise TypeError("Accounting requires the exact experimental AxisColumnNativeSimulator")
        simulator.assert_model_frozen()
        simulator._assert_episode()
        if (simulator.decision_model_hash != self.expected_model_hash
                or simulator.case_hash != self.optimization_manifest.case_hash
                or simulator.world_generator_fingerprint != self.optimization_manifest.generator.fingerprint):
            raise ValueError("Axis source, sampler or decision model differs from the declared run")

    def recorded_factory(self) -> _RecordedAxisSimulator:
        self._require_running()
        try:
            raw = self._factory()
            self._bind(raw)
            if raw in self._live_raw_instances:
                raise ValueError("Factory reused a live simulator")
            if raw.engine.revision or raw.engine.history or raw._actions or raw.total_reward or raw.terminated:
                raise ValueError("Factory must return a fresh unexecuted axis episode")
            instance = _RecordedAxisSimulator(self, raw, self._next_instance)
            self._next_instance += 1
            self._live_raw_instances.add(raw)
            self._events.append({"kind": "factory", "instance": instance._instance,
                                 "role": None, "constructor_seed": int(raw.seed), "executed_transition": False})
            self._persist()
            return instance
        except Exception as error:
            self._fail(error, "factory")
            raise

    def enable_decision_recording(self) -> None:
        """Enable before factories run; no simulator read or policy call is made."""
        self._require_running()
        if self._events:
            raise ValueError("Decision recording must be declared before constructing learner episodes")
        self._record_decisions = True
        self._persist()

    def observe_decision(self, record: Any) -> None:
        """Persist the learner's detached same-forward evidence before execution."""
        self._require_running()
        try:
            if not self._record_decisions:
                raise RuntimeError("Decision recording was not declared for this run")
            instance = None if self._active_decision_instance is None else self._active_decision_instance()
            if instance is None or instance._episode is None or instance._episode["status"] != "active":
                raise RuntimeError("Decision has no active declared accounting episode")
            payload = record.to_dict()
            episode = instance._episode
            for name, expected in instance._observation_bindings.items():
                value = getattr(record.inputs, name)
                if name in ("action_features", "state_features", "actor_action_features") and not record.forward_evaluated:
                    if value is not None:
                        raise ValueError("Unevaluated decision unexpectedly contains policy inputs")
                elif value is None or _observation_array_identity(value) != expected:
                    raise ValueError("Decision features or mask differ from the served native observation")
            if (instance._pending_decision is not None or payload["role"] != episode["role"]
                    or payload["seed"] != episode["seed"] or payload["step"] != instance._decision_count
                    or tuple(payload["inputs"]["action_ids"]) != instance._action_ids):
                raise ValueError("Decision does not bind the current declared episode and action inventory")
            selected = payload["selected_index"]
            if (type(selected) is not int or not 0 <= selected < len(instance._action_ids)
                    or instance._action_ids[selected] != payload["selected_action_id"]
                    or not payload["inputs"]["action_mask"]["values"][selected]):
                raise ValueError("Decision selected an unavailable action")
            event = {"kind": "decision", "decision_id": f"axis-decision-{self._next_decision:06d}",
                     "instance": instance._instance, "episode": episode["episode"],
                     "role": episode["role"], "seed": episode["seed"],
                     "executed_transition": False, "status": "recorded_pre_step", "payload": payload}
            self._next_decision += 1
            instance._decision_count += 1
            instance._pending_decision = event
            self._events.append(event)
            self._persist()
        except Exception as error:
            self._fail(error, "decision_observer")
            raise

    def snapshot(self) -> dict[str, Any]:
        """Return a detached receipt, including truthful context after export failure."""
        totals = {}
        for role in ("optimization", "selection"):
            steps = [event for event in self._events
                     if event.get("role") == role and event.get("executed_transition")]
            totals[role] = {
                "counts_complete": not any(event.get("counts_complete") is False
                    and event.get("role") in (None, role) for event in self._events),
                "executed_transitions": len(steps),
                "returned_transitions": sum(event["returned_to_learner"] for event in steps),
                "native_commits": sum(event["native_commit"] for event in steps),
                "committed_but_unreturned": sum(event["native_commit"] and not event["returned_to_learner"] for event in steps),
                "executed_reward": sum(event["reward"] for event in steps),
            }
        receipt = {"version": ACCOUNTING_VERSION, "status": self._status,
                   "case_hash": self.optimization_manifest.case_hash,
                   "decision_model_hash": self.expected_model_hash, "input_profile": "RAW",
                   "partitions": {item.role.value: item.to_dict()
                                  for item in (self.optimization_manifest, self.selection_manifest)},
                   "totals": totals, "events": self._events, "episodes": self._episodes,
                   "counts_complete": all(item["counts_complete"] for item in totals.values()),
                   "count_interpretation": "Verified transitions only; lower bounds whenever counts_complete is false.",
                   "failure": self._failure, "receipt_export_failure": self._export_error,
                   "learner_result": self._learner_result,
                   "decision_recording_enabled": self._record_decisions,
                   "decision_observer_version": "learner-decision-observer-v1" if self._record_decisions else None,
                   "candidate_eligible": False, "resume_supported": False,
                   "independent_geometry_evaluation": False,
                   "consumer_contract": "Read this receipt together with learner result/failures; generic counters remain unchanged. Accounting alone never authorizes a candidate."}
        receipt = copy.deepcopy(receipt)
        receipt["receipt_hash"] = _hash(receipt)
        return receipt

    def _persist(self) -> None:
        try:
            _atomic_receipt(self.receipt_path, self.snapshot())
        except Exception as error:
            self._status = "failed"
            self._export_error = {"exception": type(error).__name__, "reason": str(error)}
            raise AccountingReceiptError(self.snapshot()) from error

    def _fail(self, error: Exception, stage: str, *, event: dict[str, Any] | None = None) -> None:
        """Preserve the original exception even if the separate journal is unwritable."""
        self._status = "failed"
        if event is not None:
            event["returned_to_learner"] = False
        if self._failure is None:
            self._failure = {"stage": stage, "exception": type(error).__name__, "reason": str(error)}
        try:
            self._persist()
        except AccountingReceiptError as export_error:
            error.add_note(str(export_error))
        # Read this even when the last durable receipt predates the failed step.
        error.axis_accounting_receipt = self.snapshot()
        if isinstance(error, AccountingReceiptError):
            error.receipt = self.snapshot()

    def finish(self, result: Any) -> None:
        self._require_running()
        self._learner_result = asdict(result)
        self._status = "learner_returned"
        for episode in self._episodes:
            if episode["status"] == "active":
                episode["status"] = "unfinished_at_learner_return"
        try:
            self._persist()
        except Exception as error:
            self._fail(error, "final_export")
            raise


class _RecordedAxisSimulator:
    """Narrow Simulator protocol; intentionally fails existing non-RAW class gates."""

    def __init__(self, owner: AxisTrainingAccounting, raw: AxisColumnNativeSimulator, instance: int):
        self._owner, self._raw, self._instance = owner, raw, instance
        self._episode: dict[str, Any] | None = None
        self._action_ids: tuple[str, ...] = ()
        self._last_state = self._state()
        self._pending_decision: dict[str, Any] | None = None
        self._decision_count = 0
        self._observation_bindings: dict[str, tuple] = {}

    def _remember_observation(self, observation: Any) -> None:
        """Bind the returned object, without another simulator observation call."""
        actions, state = np.asarray(observation.action_features), np.asarray(observation.state_features)
        actual_actions = np.asarray(actions, dtype=np.float32)
        self._observation_bindings = {
            "source_action_features": _observation_array_identity(actions),
            "source_state_features": _observation_array_identity(state),
            "action_mask": _observation_array_identity(np.asarray(observation.action_mask, dtype=bool)),
            "action_features": _observation_array_identity(actual_actions),
            "state_features": _observation_array_identity(np.asarray(state, dtype=np.float32).flatten()),
            "actor_action_features": _observation_array_identity(actual_actions),
        }

    @property
    def decision_model_hash(self):
        return self._raw.decision_model_hash

    @property
    def case_hash(self):
        return self._raw.case_hash

    @property
    def world_generator_fingerprint(self):
        return self._raw.world_generator_fingerprint

    def assert_model_frozen(self) -> None:
        self._owner._require_running()
        try:
            self._owner._bind(self._raw)
            self._guard_state()
        except Exception as error:
            self._owner._fail(error, "model_or_state_binding")
            raise

    def _guard_state(self) -> None:
        current = self._state()
        if current != self._last_state:
            owner = self._owner
            error = RuntimeError("Axis state changed outside its accounting wrapper")
            owner._events.append({"kind": "unrecorded_state_change", "instance": self._instance,
                "role": self._episode["role"] if self._episode is not None else None,
                "executed_transition": False, "counts_complete": False,
                "before": self._last_state, "observed_after": current})
            owner._fail(error, "state_binding")
            raise error

    def reset(self, seed: int = 0):
        owner = self._owner
        owner._require_running()
        attempt = None
        try:
            self.assert_model_frozen()
            if self._pending_decision is not None:
                raise RuntimeError("Cannot reset after a recorded decision without its transition")
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) not in owner._roles:
                raise ValueError("Reset seed does not belong to a declared optimization or selection partition")
            if self._episode is not None and self._episode["status"] == "active":
                self._episode["status"] = "abandoned_on_explicit_reset"
            self._episode = {"episode": len(owner._episodes), "instance": self._instance,
                             "seed": int(seed), "role": owner._roles[int(seed)], "status": "resetting"}
            attempt = self._episode
            owner._episodes.append(self._episode)
            observation = self._raw.reset(int(seed))
            self._action_ids = tuple(observation.action_ids)
            if owner._record_decisions:
                self._remember_observation(observation)
            self._episode["status"] = "active"
            self._last_state = self._state()
            self._decision_count = 0
            owner._active_decision_instance = weakref.ref(self)
            owner._persist()
            return observation
        except Exception as error:
            if attempt is not None:
                attempt["status"] = "reset_failed"
            owner._events.append({"kind": "reset_failure", "instance": self._instance,
                "role": attempt["role"] if attempt is not None else None,
                "executed_transition": False, "requested_seed": repr(seed),
                "exception": type(error).__name__})
            owner._fail(error, "reset")
            raise

    def _state(self) -> dict[str, Any]:
        raw = self._raw
        return {"revision": raw.engine.revision, "state_hash": raw.engine.state_hash,
                "seed": int(raw.seed), "world_hash": raw._hidden_world_hash,
                "native_history": copy.deepcopy(raw.engine.history), "history": copy.deepcopy(raw._history),
                "actions": list(raw._actions), "total_reward": float(raw.total_reward),
                "removed_cells": int(np.count_nonzero(raw.engine.removed_mask)),
                "terminated": bool(raw.terminated), "termination_reason": raw.termination_reason}

    def _authenticate(self, before: dict[str, Any], selected: str | None,
                      *, returned: bool) -> dict[str, Any] | None:
        raw = self._raw
        self._owner._bind(raw)
        # Integrity validation does not generate previews or call cancellation.
        # This additional full native-history/cavity scan is deliberately counted.
        raw._verified_batch()
        after = self._state()
        if after["seed"] != before["seed"] or after["world_hash"] != before["world_hash"]:
            raise RuntimeError("Step changed the frozen episode world")
        delta = after["revision"] - before["revision"]
        if delta == 1:
            if (after["native_history"][:-1] != before["native_history"]
                    or after["history"][:-1] != before["history"]
                    or after["actions"] != before["actions"] + [selected]
                    or selected in (None, "STOP")
                    or after["state_hash"] == before["state_hash"]):
                raise RuntimeError("Transition does not bind exactly one requested native commit")
            actual_info = after["history"][-1]
            if any(actual_info.get(key) != value for key, value in after["native_history"][-1].items()):
                raise RuntimeError("Adapter and native committed histories disagree")
            removed = len(after["native_history"][-1]["removed_indices_native"])
            if after["removed_cells"] - before["removed_cells"] != removed or removed <= 0:
                raise RuntimeError("Committed removal accounting disagrees with native history")
            actual_reward = after["total_reward"] - before["total_reward"]
            native_commit = True
        elif delta == 0:
            native_keys = ("revision", "state_hash", "native_history", "history", "actions", "total_reward", "removed_cells")
            if any(after[key] != before[key] for key in native_keys):
                raise RuntimeError("Failed step changed state without one authenticated commit")
            if selected != "STOP" or before["terminated"] or not after["terminated"]:
                if returned:
                    raise RuntimeError("Step returned without an executed transition")
                return None
            actual_info = {"action_id": "STOP", "termination_reason": after["termination_reason"]}
            actual_reward, native_commit = 0., False
        else:
            raise RuntimeError("Step executed an invalid number of native commits")
        if not math.isfinite(actual_reward):
            raise RuntimeError("Committed reward is nonfinite")
        return {"kind": "transition", "instance": self._instance,
                "episode": self._episode["episode"], "role": self._episode["role"],
                "seed": self._episode["seed"], "executed_transition": True,
                "decision_id": None if self._pending_decision is None else self._pending_decision["decision_id"],
                "returned_to_learner": returned, "native_commit": native_commit,
                "action_id": selected, "reward": actual_reward, "info": actual_info,
                "before_revision": before["revision"], "after_revision": after["revision"],
                "before_state_hash": before["state_hash"], "after_state_hash": after["state_hash"],
                "removed_cells": after["removed_cells"] - before["removed_cells"],
                "terminated": after["terminated"], "next_observation_complete": returned,
                "proposal_accounting_after": {"integrity_calls": raw._integrity_calls,
                    "integrity_seconds": raw._integrity_seconds, "preview_calls": raw._preview_calls,
                    "preview_seconds": raw._preview_seconds}}

    @staticmethod
    def _check_report(event: dict[str, Any], info: Any, reward: Any) -> None:
        """False reported metadata cannot erase an already authenticated cut."""
        mismatch = None
        try:
            history_matches = _hash(info) == _hash(event["info"])
        except (ValueError, TypeError, OverflowError):
            history_matches = False
        if not history_matches:
            mismatch = "Reported transition history differs from committed state"
        else:
            try:
                matches = math.isfinite(float(reward)) and math.isclose(
                    float(reward), event["reward"], rel_tol=1e-12, abs_tol=1e-12)
            except (ValueError, TypeError, OverflowError):
                matches = False
            if not matches:
                mismatch = "Reported reward differs from committed state"
        if mismatch is not None:
            event["protocol_mismatch"] = mismatch
            raise RuntimeError(mismatch)

    def step(self, action: str | int):
        owner = self._owner
        owner._require_running()
        if self._episode is None or self._episode["status"] != "active":
            raise RuntimeError("A successful declared reset is required before each episode")
        self.assert_model_frozen()
        before = self._state()
        selected = (self._action_ids[int(action)] if isinstance(action, (int, np.integer))
                    and not isinstance(action, (bool, np.bool_)) and 0 <= action < len(self._action_ids)
                    else action if isinstance(action, str) and action in self._action_ids else None)
        if owner._record_decisions and (self._pending_decision is None
                or selected != self._pending_decision["payload"]["selected_action_id"]):
            error = RuntimeError("Step has no matching persisted learner decision")
            owner._fail(error, "decision_step_binding")
            raise error
        event = None
        try:
            result = self._raw.step(action)
        except Exception as error:
            try:
                event = self._authenticate(before, selected, returned=False)
                if isinstance(error, CommittedTransitionInterrupted) and (event is None or not event["native_commit"]):
                    raise RuntimeError("Claimed interrupted commit has no corresponding native transition")
            except Exception as verification_error:
                owner._events.append({"kind": "unverified_step_failure", "instance": self._instance,
                    "episode": self._episode["episode"], "role": self._episode["role"],
                    "decision_id": None if self._pending_decision is None else self._pending_decision["decision_id"],
                    "executed_transition": False, "counts_complete": False,
                    "error": str(verification_error), "before": before, "observed_after": self._state()})
                error.add_note("Transition accounting could not authenticate state: " + str(verification_error))
            if event is not None:
                if isinstance(error, CommittedTransitionInterrupted):
                    try:
                        self._check_report(event, error.info, error.reward)
                    except RuntimeError as mismatch:
                        error.add_note(str(mismatch))
                event["exception"] = type(error).__name__
                owner._events.append(event)
            else:
                owner._events.append({"kind": "step_failure", "instance": self._instance,
                    "episode": self._episode["episode"], "role": self._episode["role"],
                    "decision_id": None if self._pending_decision is None else self._pending_decision["decision_id"],
                    "executed_transition": False, "exception": type(error).__name__})
            self._episode["status"] = "failed"
            if self._pending_decision is not None:
                self._pending_decision["status"] = "executed_unreturned" if event is not None else "step_failed_unexecuted_or_unverified"
            owner._fail(error, "step", event=event)
            raise
        try:
            event = self._authenticate(before, selected, returned=True)
            owner._events.append(event)
            self._check_report(event, result.info, result.reward)
            if bool(result.terminated) != bool(self._raw.terminated):
                event["protocol_mismatch"] = "Reported termination differs from authenticated state"
                raise RuntimeError("Reported termination differs from authenticated state")
            self._last_state = self._state()
            self._action_ids = tuple(result.observation.action_ids)
            if owner._record_decisions:
                self._remember_observation(result.observation)
            if result.terminated:
                self._episode["status"] = "complete"
            if self._pending_decision is not None:
                self._pending_decision["status"] = "step_returned"
            owner._persist()
            self._pending_decision = None
            return result
        except Exception as error:
            self._episode["status"] = "failed"
            if self._pending_decision is not None:
                self._pending_decision["status"] = "executed_unreturned" if event is not None else "step_failed_unexecuted_or_unverified"
            if event is None:
                owner._events.append({"kind": "unverified_returned_step", "instance": self._instance,
                    "episode": self._episode["episode"], "role": self._episode["role"],
                    "decision_id": None if self._pending_decision is None else self._pending_decision["decision_id"],
                    "executed_transition": False, "counts_complete": False,
                    "before": before, "observed_after": self._state(), "error": str(error)})
            owner._fail(error, "step_verification_or_export", event=event)
            raise

    def metrics(self):
        self.assert_model_frozen()
        try:
            return self._raw.metrics()
        except Exception as error:
            self._owner._fail(error, "metrics")
            raise


def train_axis_policy(accounting: AxisTrainingAccounting, *, config: Any,
                      output_dir: str | Path, cancelled: Callable[[], bool] | None = None,
                      progress: Callable[[dict[str, Any]], None] | None = None,
                      record_decisions: bool = False):
    """Scratch RAW training only; return the unchanged result or re-raise failure.

    Consumers must read both the generic learner artifacts and ``accounting``'s
    separate receipt. This helper offers no resume, transfer, evaluation or
    candidate-publication path, including after failed final export.
    """
    from .learning import train_patient_policy

    accounting._require_running()
    try:
        if record_decisions:
            accounting.enable_decision_recording()
        directory = Path(output_dir).resolve()
        reserved = {"checkpoint.pt", "initial.pt", "result.json", "contract.json", "failures.jsonl"}
        if accounting.receipt_path.parent == directory and accounting.receipt_path.name in reserved:
            raise ValueError("The separate accounting receipt cannot replace a learner artifact")
        result = train_patient_policy(accounting.recorded_factory, accounting.optimization_manifest,
            accounting.selection_manifest, config=config, output_dir=output_dir,
            cancelled=cancelled, progress=progress, input_profile="RAW", resume=False,
            decision_observer=accounting.observe_decision if record_decisions else None)
        accounting.finish(result)
        return result
    except Exception as error:
        accounting._fail(error, "learner")
        raise
