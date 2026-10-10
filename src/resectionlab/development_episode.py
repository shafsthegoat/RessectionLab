"""Generated vertical slice through the existing native simulator and desktop case.

No patient/model loading lives here. The fixture's signal/target rule is an
analytic software construct, never an MRI estimator. Probe records geometric
tangency, not force, material diagnosis, or acquired sensor measurements.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .core import CaseData, SourceRef, array_digest, semantic_digest, freeze_json, thaw_json
from .geometry import AccessWindow, ToolGeometry
from .native_spatial_task import NativeSpatialCase, NativeSpatialTask
from .observed_search import observed_beam_search
from .structural_evidence import structural_frame_hash

SCHEMA = "resectionlab.shared-native-development-episode.v1"
FIXTURE = "generated-sequential-v1"
TOOLS = (ToolGeometry("development-aspirator", 1.25, .35, 12., 30., 1.5),
         ToolGeometry("development-probe", .5, .2, 12., 30., .4))
MODES = {TOOLS[0].tool_id: "aspirate", TOOLS[1].tool_id: "probe"}


def make_development_task(*, cancelled=None, reference_target=None):
    """175 tissue cells in a 13x13x12 source; source-derived fixed action lattice."""
    shape = (13, 13, 12)
    support = np.zeros(shape, bool)
    support[4:9, 4:9, 2:9] = True
    nominal = np.zeros(shape, bool)
    nominal[5:8, 5:8, 5:9] = True
    image = np.where(nominal, .8, np.where(support, .2, 0.)).astype(np.float32)
    case = NativeSpatialCase(image, support, nominal if reference_target is None else reference_target,
        np.eye(4), AccessWindow((6., 6., 1.5), (0., 0., 1.), 2., "generated-access"), TOOLS,
        track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="analytic fixture forward signal is nonzero exactly on supplied tissue support",
        nominal_target=image >= .5, target_source_kind="derived_from_scan",
        target_derivation="analytic .2/.8 fixture signal threshold .5, not an MRI estimator")
    return NativeSpatialTask(case, max_steps=6, tool_modes=MODES, cancelled=cancelled)


def public_display_case(task):
    """Publish only the same permitted images/support/nominal target as the actor."""
    source = task.case
    return CaseData(FIXTURE, source.structural_intensity,
        {"generated_nominal_target": source.nominal_target > 0}, source.affine_ras_mm,
        (SourceRef(FIXTURE, "generated://" + FIXTURE, native_frame="RAS+", provenance="simulated"),),
        frame="RAS+", brain_mask=source.observed_support,
        metadata={"is_synthetic": True, "evidence_kind": "generated_software_fixture", "patient_admission": False,
                  "source_task_hash": source.source_hash, "scope": "analytic instrument interaction development"})


def _select(task, tool_id, voxel):
    rows = task.candidate_inventory()["ledger"]
    matches = [row for row in rows if row["tool_id"] == tool_id and row["voxel"] == list(voxel)]
    if len(matches) != 1 or not matches[0]["feasible"]:
        raise RuntimeError("Generated scripted action is not in the shared legal inventory")
    return matches[0]["action_id"]


def plan_development_episode(task, selector):
    """Seal a full nominal rollout before the generated reference is scored."""
    if selector not in {"scripted", "SEARCH"}:
        raise ValueError("Choose scripted or SEARCH; no learned result is supplied by this fixture")
    nominal = task.planning_clone()
    if selector == "scripted":
        actions = []
        for tool, depth in ((TOOLS[0], 2), (TOOLS[1], 2), (TOOLS[0], 4), (TOOLS[1], 4), (TOOLS[0], 6)):
            action = _select(nominal, tool.tool_id, (6, 6, depth))
            actions.append(action)
            nominal.step(action)
        actions.append("STOP")
        nominal.step("STOP")
        accounting = {"selector": "scripted_interface_demonstration", "model_transition_calls": 6,
                      "actor_forward_calls": 0, "optimizer_updates": 0}
    else:
        actions, accounting = observed_beam_search(nominal, max_calls=24, beam_width=2, seconds=10.,
            objective_source="permitted_generated_nominal_target_only", transition_mode="lazy_planning")
        actions = list(actions)
        for action in actions:
            nominal.step(action)
        accounting = {**accounting, "selector": "existing_observed_beam_search", "optimizer_updates": 0}
    if not nominal.terminated:
        raise RuntimeError("A development plan must terminate explicitly or exhaust its declared horizon")
    record = {"decision_model_hash": task.decision_model_hash, "source_hash": task.case.source_hash,
              "actions": actions, "history": nominal.metrics()["history"],
              "observation_contract": "sequential-spatial-observation-v1", "max_steps": task.max_steps}
    record = thaw_json(freeze_json(record))
    return freeze_json(record), semantic_digest(record), accounting


def replay_frames(task, history):
    """Exact native microstep deltas and reversed poses; no time interpolation.

    Frame states describe certified prefix snapshots, not partially committed
    engine transactions. Engine macro ancestry remains in history separately.
    """
    shape = task.case.observed_support.shape
    removed, contact, probe = (np.zeros(shape, bool) for _ in range(3))
    frames = []
    previous = "initial"

    def append(action_index, phase, record, tip, remove=(), touch=(), probed=()):
        nonlocal previous
        for array, cells in ((removed, remove), (contact, touch), (probe, probed)):
            cells = np.asarray(cells, dtype=int).reshape(-1, 3)
            if len(cells):
                array[tuple(cells.T)] = True
        state = {"frameIndex": len(frames), "actionIndex": action_index, "phase": phase,
            "toolId": None if record is None else record.get("tool_id"),
            "mode": None if record is None else record["interaction_mode"],
            "tipRasMm": None if tip is None else list(tip),
            "axis": None if record is None else record.get("axis_unit"),
            "cavityHash": array_digest(removed),
            "remainingHash": array_digest(task.case.observed_support & ~removed),
            "contactHash": array_digest(contact), "probeContactHash": array_digest(probe)}
        identity = semantic_digest(state)
        frames.append({**state, "removedIndicesNative": list(remove), "contactIndicesNative": list(touch),
                       "probeContactIndicesNative": list(probed),
                       "stateBefore": identity if not frames else previous, "stateAfter": identity})
        previous = identity

    append(-1, "initial", None, None)
    for action_index, record in enumerate(history):
        if record["action_id"] == "STOP":
            append(action_index, "stop", record, None)
            continue
        for micro in record["microsteps"]:
            touched = micro["contact_indices_native"]
            probed = ([] if record["interaction_mode"] != "probe" else
                      [cell for cell in touched if not removed[tuple(cell)]])
            append(action_index, "insertion", record, micro["tip_end_mm"],
                   micro["removed_indices_native"], touched, probed)
        # The final cavity contains the entire insertion path. Exact axial
        # reversal introduces no new removed/contact cells in this model.
        for micro in reversed(record["microsteps"]):
            append(action_index, "withdrawal", record, micro["tip_start_mm"])
    if (not np.array_equal(removed, task._engine.removed_mask)
            or not np.array_equal(contact, task._engine.contact_mask)
            or not np.array_equal(probe, task._engine.probe_contact_mask)):
        raise RuntimeError("Replay delta prefixes differ from authoritative native state")
    return frames


def execute_development_episode(*, selector="scripted", cancelled=None):
    """Execute one small generated episode and return the existing CaseData + JSON.

    The desktop bridge owns installation/transfers. Cancellation/errors raise;
    a failed computation is not converted into a completed STOP strategy.
    """
    task = make_development_task(cancelled=cancelled)
    initial = task._engine.state_hash
    rejected = task._engine.preview_stroke(TOOLS[1].tool_id, (6., 6., 2.),
        entry_mm=(6., 6., 1.5), interaction_mode="probe")
    if rejected.feasible or task._engine.state_hash != initial:
        raise RuntimeError("Expected pre-opening probe refusal did not preserve state")
    plan, seal, accounting = plan_development_episode(task, selector)
    if semantic_digest(plan) != seal:
        raise RuntimeError("Development strategy changed before execution")
    for action in plan["actions"]:
        task.step(action)
    history = thaw_json(freeze_json(task.metrics()["history"]))
    # Reference reward may differ; executed physical history must match the seal.
    geometry_keys = ("action_id", "interaction_mode", "source_state_hash", "result_state_hash",
                     "removed_indices_native", "contact_indices_native", "microsteps")
    for planned, executed in zip(plan["history"], history):
        if any(thaw_json(planned[key]) != executed[key] for key in geometry_keys):
            raise RuntimeError("Executed native geometry differs from sealed nominal history")
    audit = task.independent_geometry_check().to_dict()
    if not audit["feasible"]:
        raise RuntimeError("Generated episode failed independent native geometry validation")
    display = public_display_case(task)
    frames = replay_frames(task, history)
    source_binding = {"display_case_hash": display.semantic_hash, "native_source_hash": task.case.source_hash,
        "structural_intensity_hash": array_digest(display.mri), "affine_hash": array_digest(display.affine),
        "source_frame_hash": structural_frame_hash(display),
        "support_hash": array_digest(display.brain_mask),
        "nominal_target_mask_hash": array_digest(display.compartments["generated_nominal_target"]),
        "private_reference_published": False}
    episode = {"schema": SCHEMA, "caseHash": display.semantic_hash, "sourceHash": task.case.source_hash,
        "decisionModelHash": task.decision_model_hash, "selector": selector,
        "evidenceKind": "generated_software_fixture",
        "fidelity": "native_grid_connected_exposed_tip_aspiration_and_nonremoving_geometric_probe",
        "backendStatus": "generated_executed", "patientAdmission": False, "clinicalValidation": False,
        "frame": "RAS+", "physicalUnits": "mm", "shape": list(display.mri.shape), "affine": display.affine.tolist(),
        "sourceBinding": source_binding,
        "tools": [{**asdict(tool), "interactionMode": MODES[tool.tool_id]} for tool in TOOLS],
        "access": {"center_mm": task.case.access.center_mm.tolist(),
                   "normal_inward": task.case.access.normal_inward.tolist(),
                   "radius_mm": task.case.access.radius_mm, "window_id": task.case.access.window_id},
        "history": history, "replayFrames": frames, "initialStateId": frames[0]["stateAfter"],
        "finalStateId": frames[-1]["stateAfter"], "nativeEngineFinalStateId": task._engine.state_hash,
        "finalRemovedIndicesNative": np.argwhere(task._engine.removed_mask).tolist(),
        "geometryAudit": audit, "metrics": task.metrics(),
        "planning": {**accounting, "strategySeal": seal, "strategy": thaw_json(plan),
                     "sealedBeforeReferenceScoring": True, "learnedPolicyExecuted": False},
        "attemptDiagnostics": [{"status": "rejected", "interactionMode": "probe", "reason": rejected.reason,
            "toolId": TOOLS[1].tool_id, "tipRasMm": [6., 6., 2.],
            "stateBefore": initial, "stateAfter": initial, "removedIndicesNative": []}],
        "sequentialEffect": {"preOpeningProbeFeasible": False, "preOpeningProbeReason": rejected.reason,
            "executedProbeCount": sum(row["interaction_mode"] == "probe" for row in history),
            "explanation": "Aspiration changes the shared cavity. Only then can the same probe reach and tangentially contact deeper retained surfaces."},
        "hashEncoding": "core.array_digest: sha256(UTF8 sorted Python JSON default separators of {dtype:'|b1',shape:[X,Y,Z]} followed by C-order boolean bytes)",
        "replayStateInterpretation": "Hash-bound certified microstep prefixes and reversed poses; engine actions commit atomically and keep separate native state hashes.",
        "unsupported": ["tissue_forces", "deformation", "actual_sensor_measurement", "vascular_injury", "clinical_outcomes"],
        "interpretation": "Generated software-development interaction sequence; probe is geometric contact, not new anatomical knowledge. SEARCH may omit its cost under the unchanged removal objective."}
    episode = thaw_json(freeze_json(episode))
    episode["episodeId"] = semantic_digest(episode)
    return display, episode
