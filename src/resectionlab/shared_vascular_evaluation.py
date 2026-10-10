"""Generated-only vessel-annotation contact sidecar for the native desktop episode.

The strategy and independent native geometry are checked before the private
reference callback is invoked. No annotation enters the task, actor, search,
reward, or replay frames. Contact is neither removal nor clinical injury.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from resectionlab.core import array_digest, immutable_array, semantic_digest, thaw_json, freeze_json
from resectionlab.development_episode import (SCHEMA as EPISODE_SCHEMA, make_development_task,
    public_display_case, replay_frames)
from resectionlab.functional_events import AxialToolSweep, sweeps_from_native_history
from resectionlab.structural_evidence import structural_frame_hash
from resectionlab.vascular_contact_streaming import (Grid, Capsule, Budget, evaluate_contacts)


SCHEMA = "generated-shared-vascular-encounter-v1"
SHAPE = (24, 24, 48)
AFFINE = ((1., 0., 0., -6.), (0., 1., 0., -6.),
          (0., 0., 1., -32.), (0., 0., 0., 1.))
IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.),
            (0., 0., 1., 0.), (0., 0., 0., 1.))
# Hashes of the fixed generated fixture, computed with core.array_digest.
# Reference arrays are created only by _load_generated_reference after preflight.
MASK_HASH = "sha256:47a48d657b90bd4185e94ee51f000e80a3e7e7d05eabdd3dbaa31b5df672ad37"
COVERAGE_HASH = "sha256:38eaefdf1c3f1ad2bd533584de6dc3e99d0fb0c04c608720fd49479d1d82afa0"
FRAME_HASH = "sha256:074e863bd5276934cfa2c27e5fff8d474f1ebfafa7f3f2855d98e2c1fb923bb4"


def _canonical(value) -> str:
    return json.dumps(thaw_json(freeze_json(value)), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _output_directory(path) -> Path:
    """A fresh caller-owned directory; desktop owns ephemeral parents itself."""
    path = Path(path).absolute()
    _need(".." not in path.parts and path.parent.is_dir(), "output_parent_required")
    _need(not any(part.is_symlink() for part in (path, *path.parents)),
          "output_symlink_forbidden")
    path.mkdir(parents=False, exist_ok=False)
    _sync_directory(path.parent)
    return path


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, value: dict) -> None:
    raw = (_canonical(value) + "\n").encode()
    _need(len(raw) <= 2 * 1024**2, "evaluation_document_budget")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


@dataclass(frozen=True, slots=True)
class GeneratedReferenceBinding:
    episode_id: str
    source_hash: str
    decision_model_hash: str
    strategy_seal: str
    physical_history_hash: str
    reference_frame_hash: str
    mask_hash: str
    coverage_hash: str
    planning_to_reference_ras_mm: tuple[tuple[float, ...], ...]
    source_kind: str = "generated_private_annotation_fixture"
    coverage_meaning: str = "annotation_domain_not_vessel_completeness"
    transform_provenance: str = "generated_declared_rigid_RAS_mm_to_RAS_mm"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        values = (self.episode_id, self.source_hash, self.decision_model_hash,
                  self.strategy_seal, self.physical_history_hash,
                  self.reference_frame_hash, self.mask_hash, self.coverage_hash)
        _need(all(isinstance(v, str) and len(v) == 71 and v.startswith("sha256:")
                  and all(c in "0123456789abcdef" for c in v[7:]) for v in values),
              "reference_binding_hashes")
        _need(self.source_kind == "generated_private_annotation_fixture" and
              self.coverage_meaning == "annotation_domain_not_vessel_completeness" and
              self.transform_provenance == "generated_declared_rigid_RAS_mm_to_RAS_mm",
              "reference_scope")
        transform = np.asarray(self.planning_to_reference_ras_mm, float)
        _need(transform.shape == (4, 4) and np.array_equal(transform[3], [0, 0, 0, 1])
              and np.all(np.isfinite(transform)) and
              np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-9, rtol=0)
              and np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-9, rtol=0),
              "reference_transform")
        object.__setattr__(self, "planning_to_reference_ras_mm",
                           tuple(tuple(float(v) for v in row) for row in transform))
        object.__setattr__(self, "fingerprint", semantic_digest(self.record()))

    def record(self) -> dict:
        return {name: thaw_json(freeze_json(getattr(self, name)))
                for name in self.__dataclass_fields__ if name != "fingerprint"}

    def assert_intact(self) -> None:
        _need(self.fingerprint == semantic_digest(self.record()), "reference_binding_changed")


@dataclass(frozen=True, slots=True)
class GeneratedReference:
    binding: GeneratedReferenceBinding
    mask: np.ndarray
    coverage: np.ndarray
    affine_ras_mm: np.ndarray

    def __post_init__(self) -> None:
        _need(type(self.binding) is GeneratedReferenceBinding, "reference_binding_type")
        mask, coverage = np.asarray(self.mask), np.asarray(self.coverage)
        _need(mask.dtype == coverage.dtype == np.bool_ and mask.ndim == 3
              and mask.shape == coverage.shape and not np.any(mask & ~coverage),
              "reference_arrays_or_domain")
        frame = Grid(tuple(int(v) for v in mask.shape), tuple(map(tuple, self.affine_ras_mm)))
        _need(semantic_digest({"shape": list(frame.shape), "affine_ras_mm": frame.affine_ras_mm})
              == self.binding.reference_frame_hash, "reference_frame_mismatch")
        object.__setattr__(self, "mask", immutable_array(mask, bool))
        object.__setattr__(self, "coverage", immutable_array(coverage, bool))
        object.__setattr__(self, "affine_ras_mm", immutable_array(self.affine_ras_mm, np.float64))
        self.assert_intact()

    def assert_intact(self) -> None:
        self.binding.assert_intact()
        _need(array_digest(self.mask) == self.binding.mask_hash and
              array_digest(self.coverage) == self.binding.coverage_hash and
              semantic_digest({"shape": list(self.mask.shape),
                               "affine_ras_mm": self.affine_ras_mm.tolist()}) ==
                  self.binding.reference_frame_hash,
              "reference_content_changed")


def _preflight(episode: dict, cancelled: Callable[[], bool] | None) -> tuple:
    if cancelled is not None and cancelled():
        raise InterruptedError("vascular_preflight_cancelled")
    _need(type(episode) is dict and episode.get("schema") == EPISODE_SCHEMA
          and episode.get("evidenceKind") == "generated_software_fixture"
          and episode.get("patientAdmission") is False
          and episode.get("clinicalValidation") is False, "generated_episode_only")
    _need(len(_canonical(episode)) <= 2 * 1024**2, "episode_document_budget")
    body = {key: value for key, value in episode.items() if key != "episodeId"}
    _need(semantic_digest(body) == episode.get("episodeId"), "episode_content_changed")
    _need(episode.get("selector") in ("scripted", "SEARCH")
          and episode.get("frame") == "RAS+" and episode.get("physicalUnits") == "mm",
          "episode_scope_or_frame")
    task = make_development_task(cancelled=cancelled)
    display = public_display_case(task)
    _need(episode.get("caseHash") == display.semantic_hash
          and episode.get("sourceHash") == task.case.source_hash
          and episode.get("decisionModelHash") == task.decision_model_hash
          and episode.get("shape") == list(task.case.observed_support.shape)
          and episode.get("affine") == task.case.affine_ras_mm.tolist(),
          "episode_source_or_environment_mismatch")
    source_binding = {"display_case_hash": display.semantic_hash,
        "native_source_hash": task.case.source_hash,
        "structural_intensity_hash": array_digest(display.mri),
        "affine_hash": array_digest(display.affine),
        "source_frame_hash": structural_frame_hash(display),
        "support_hash": array_digest(display.brain_mask),
        "nominal_target_mask_hash": array_digest(display.compartments["generated_nominal_target"]),
        "private_reference_published": False}
    _need(episode.get("sourceBinding") == source_binding,
          "public_source_binding_changed")
    _need(episode.get("tools") == [{**asdict(tool), "interactionMode": task.tool_modes[tool.tool_id]}
                                   for tool in task.case.tools]
          and episode.get("access") == {"center_mm": task.case.access.center_mm.tolist(),
              "normal_inward": task.case.access.normal_inward.tolist(),
              "radius_mm": task.case.access.radius_mm,
              "window_id": task.case.access.window_id},
          "episode_tool_or_access_changed")
    planning = episode.get("planning")
    _need(type(planning) is dict and planning.get("sealedBeforeReferenceScoring") is True
          and planning.get("learnedPolicyExecuted") is False
          and semantic_digest(planning.get("strategy")) == planning.get("strategySeal"),
          "unsealed_generated_strategy")
    strategy = planning["strategy"]
    history = episode.get("history")
    _need(type(history) is list and 1 <= len(history) <= 6
          and sum(len(row.get("microsteps", ())) for row in history) <= 256
          and strategy.get("source_hash") == task.case.source_hash
          and strategy.get("decision_model_hash") == task.decision_model_hash
          and strategy.get("actions") == [row.get("action_id") for row in history],
          "strategy_history_or_budget")
    geometry_keys = ("action_id", "interaction_mode", "source_state_hash",
        "result_state_hash", "removed_indices_native", "contact_indices_native", "microsteps")
    _need(strategy.get("max_steps") == task.max_steps
          and strategy.get("observation_contract") == "sequential-spatial-observation-v1"
          and type(strategy.get("history")) is list
          and len(strategy["history"]) == len(history)
          and all(all(thaw_json(freeze_json(planned.get(key))) == executed.get(key)
                      for key in geometry_keys)
                  for planned, executed in zip(strategy["history"], history)),
          "sealed_nominal_geometry_changed")
    for action in strategy.get("actions", ()):
        if cancelled is not None and cancelled():
            raise InterruptedError("vascular_preflight_cancelled")
        task.step(action)
    _need(task.terminated and thaw_json(freeze_json(task.metrics()["history"])) == history
          and task._engine.state_hash == episode.get("nativeEngineFinalStateId")
          and episode.get("replayFrames") == replay_frames(task, history)
          and episode.get("metrics") == thaw_json(freeze_json(task.metrics())),
          "authoritative_replay_changed")
    _need(episode.get("initialStateId") == episode["replayFrames"][0]["stateAfter"]
          and episode.get("finalStateId") == episode["replayFrames"][-1]["stateAfter"]
          and episode.get("finalRemovedIndicesNative") == np.argwhere(task._engine.removed_mask).tolist(),
          "display_replay_or_removed_cells_changed")
    audit = task.independent_geometry_check().to_dict()
    _need(audit["feasible"] and thaw_json(freeze_json(audit)) == episode.get("geometryAudit"),
          "independent_native_geometry_failed")
    history_json = _canonical(history)
    history_hash = "sha256:" + sha256(history_json.encode()).hexdigest()
    _need(history_hash == semantic_digest(history), "physical_history_canonical_mismatch")
    return task, history, history_json, history_hash


def _binding(episode: dict, history_hash: str) -> GeneratedReferenceBinding:
    return GeneratedReferenceBinding(episode["episodeId"], episode["sourceHash"],
        episode["decisionModelHash"], episode["planning"]["strategySeal"], history_hash,
        FRAME_HASH, MASK_HASH, COVERAGE_HASH, IDENTITY)


def _load_generated_reference(binding: GeneratedReferenceBinding) -> GeneratedReference:
    """Private generated fixture is materialized only after episode preflight."""
    mask = np.zeros(SHAPE, bool)
    mask[12, 12, 34] = mask[12, 12, 5] = True
    coverage = np.ones(SHAPE, bool)
    coverage[12, 12, 36] = False
    return GeneratedReference(binding, mask, coverage, np.asarray(AFFINE))


def _counts_only(row: dict) -> dict:
    return {key: row[key] for key in ("touched_reference_cells", "positive_reference_cells",
        "unknown_reference_cells", "outside_reference_fov", "annotated_positive_encounter",
        "annotation_coverage_complete_for_sweep", "positive_cell_volume_upper_bound_mm3",
        "unknown_in_grid_cell_volume_mm3", "biological_vessel_free", "clinical_injury_probability")}


def _score(task, episode, history, history_json, history_hash, reference,
           cancelled: Callable[[], bool] | None) -> dict:
    transform = np.asarray(reference.binding.planning_to_reference_ras_mm, float)
    sweeps = iter(sweeps_from_native_history(history, task.case.tools))
    per_action = []
    capsules = []
    for index, row in enumerate(history):
        if row["action_id"] == "STOP":
            per_action.append({"actionIndex": index, "actionId": "STOP",
                "interactionMode": "stop", "sweepCount": 0,
                "shaft": None, "tip": None, "wholeTool": None})
            continue
        sweep = next(sweeps)
        mapped = AxialToolSweep(sweep.tool,
            tuple(transform[:3, :3] @ sweep.tip_start_mm + transform[:3, 3]),
            tuple(transform[:3, :3] @ sweep.tip_end_mm + transform[:3, 3]),
            tuple(transform[:3, :3] @ sweep.axis_unit))
        pair = tuple(Capsule(f"step_{index}", part, tuple(start), tuple(end), radius)
            for part, (start, end, radius) in zip(("shaft", "tip"), mapped.capsules()))
        capsules.extend(pair)
        per_action.append({"actionIndex": index, "actionId": row["action_id"],
            "interactionMode": row["interaction_mode"], "sweepCount": 1,
            "_capsules": pair})
    _need(next(sweeps, None) is None and len(capsules) <= 12,
          "native_sweep_action_alignment")
    grid = Grid(tuple(int(v) for v in reference.mask.shape),
                tuple(map(tuple, reference.affine_ras_mm)))
    def sample(indices):
        if cancelled is not None and cancelled():
            raise InterruptedError("vascular_evaluation_cancelled")
        return reference.mask[tuple(indices.T)], reference.coverage[tuple(indices.T)]
    budget = Budget(wall_seconds=10.)
    full = evaluate_contacts(grid, tuple(capsules), sample_reference=sample,
                             budget=budget, cancelled=cancelled)
    for row in per_action:
        if row["sweepCount"]:
            local = evaluate_contacts(grid, row.pop("_capsules"), sample_reference=sample,
                                      budget=budget, cancelled=cancelled)
            row.update(shaft=_counts_only(local["shaft"]), tip=_counts_only(local["tip"]),
                       wholeTool=_counts_only(local["whole_tool"]))
    result = {"schema": SCHEMA, "status": "evaluated_generated_vascular_reference",
        "episodeId": episode["episodeId"], "caseHash": episode["caseHash"],
        "sourceHash": episode["sourceHash"], "decisionModelHash": episode["decisionModelHash"],
        "strategySeal": episode["planning"]["strategySeal"],
        "physicalHistoryHash": history_hash, "physicalHistoryCanonicalJson": history_json,
        "actionIds": [row["action_id"] for row in history],
        "referenceBindingHash": reference.binding.fingerprint,
        "perAction": per_action, "shaft": _counts_only(full["shaft"]),
        "tip": _counts_only(full["tip"]), "wholeTool": _counts_only(full["whole_tool"]),
        "removedOverlap": {"status": "not_evaluated_by_contact_kernel", "outcomes": None},
        "clinicalInjuryProbability": None, "patientAdmission": False,
        "scope": "generated_geometry_annotation_contact_only"}
    result["evaluationId"] = semantic_digest(result)
    return thaw_json(freeze_json(result))


def _evaluate_development_episode_vascular(*, episode: dict, output_directory,
        reference_binding: GeneratedReferenceBinding,
        load_reference: Callable[[], GeneratedReference], cancelled=None) -> dict:
    """Injection seam for generated controls; never accepts a patient task."""
    # The callback must not be able to edit poses after replay but before contact
    # conversion. Keep a detached canonical copy and also detect caller changes.
    snapshot = thaw_json(freeze_json(episode))
    task, history, history_json, history_hash = _preflight(snapshot, cancelled)
    _need(type(reference_binding) is GeneratedReferenceBinding and callable(load_reference),
          "typed_reference_loader")
    reference_binding.assert_intact()
    expected_binding_record = reference_binding.record()
    expected_binding_hash = reference_binding.fingerprint
    _need((reference_binding.episode_id, reference_binding.source_hash,
           reference_binding.decision_model_hash, reference_binding.strategy_seal,
           reference_binding.physical_history_hash) ==
          (snapshot["episodeId"], snapshot["sourceHash"],
           snapshot["decisionModelHash"], snapshot["planning"]["strategySeal"],
           history_hash),
          "evaluator_reference_binding_mismatch")
    _need(_canonical(episode) == _canonical(snapshot), "episode_changed_before_private_load")
    output = _output_directory(output_directory)
    _write_new(output / "attempt.json", {"schema": SCHEMA,
        "status": "reserved_before_private_load", "episodeId": snapshot["episodeId"],
        "physicalHistoryHash": history_hash,
        "referenceBindingHash": expected_binding_hash,
        "privateLoaderCalls": 0})
    try:
        reference = load_reference()
        _need(type(reference) is GeneratedReference, "private_reference_type")
        reference.assert_intact()
        reference_binding.assert_intact()
        _need(_canonical(episode) == _canonical(snapshot),
              "episode_changed_during_private_load")
        _need(reference_binding.record() == expected_binding_record
              and reference_binding.fingerprint == expected_binding_hash
              and reference.binding.record() == expected_binding_record,
              "loaded_private_reference_binding_mismatch")
        result = _score(task, snapshot, history, history_json, history_hash,
                        reference, cancelled)
        reference.assert_intact()
        _need(_canonical(episode) == _canonical(snapshot)
              and reference_binding.record() == expected_binding_record
              and reference_binding.fingerprint == expected_binding_hash
              and reference.binding.record() == expected_binding_record
              and reference.binding.fingerprint == expected_binding_hash,
              "episode_or_binding_changed_during_private_scoring")
    except Exception as error:
        result = {"schema": SCHEMA, "status": "evaluation_failed",
            "episodeId": snapshot["episodeId"], "caseHash": snapshot["caseHash"],
            "sourceHash": snapshot["sourceHash"],
            "decisionModelHash": snapshot["decisionModelHash"],
            "strategySeal": snapshot["planning"]["strategySeal"],
            "physicalHistoryHash": history_hash,
            "physicalHistoryCanonicalJson": history_json,
            "actionIds": [row["action_id"] for row in history],
            "referenceBindingHash": expected_binding_hash,
            "perAction": None, "shaft": None, "tip": None, "wholeTool": None,
            "removedOverlap": {"status": "not_evaluated", "outcomes": None},
            "clinicalInjuryProbability": None, "patientAdmission": False,
            "scope": "generated_geometry_annotation_contact_only",
            "errorType": type(error).__name__,
            "reason": "private_reference_or_contact_evaluation_failed"}
        result["evaluationId"] = semantic_digest(result)
    _write_new(output / "report.json", result)
    return result


def evaluate_development_episode_vascular(*, episode: dict, output_directory,
                                          cancelled=None) -> dict:
    """Evaluate the fixed generated desktop episode without exposing labels to UI."""
    _need(type(episode) is dict, "episode_required")
    binding = _binding(episode, semantic_digest(episode.get("history")))
    return _evaluate_development_episode_vascular(episode=episode,
        output_directory=output_directory, reference_binding=binding,
        load_reference=lambda: _load_generated_reference(binding), cancelled=cancelled)
