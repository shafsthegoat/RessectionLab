"""Score sealed native plans against a separately loaded source annotation.

This adapter grants no patient or anatomical admission. The experiment runner
owns that qualification and seals all methods before calling it. The existing
independent full-tool contact primitive counts annotation encounters, not injury.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .core import array_digest, semantic_digest
from .functional_events import sweeps_from_native_history
from .geometry import ToolGeometry
from .vascular_contact_streaming import Budget, Capsule, Grid, evaluate_contacts


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def evaluate_sealed_annotation_plans(path, *, expected_sha256, load_reference):
    """Load the private reference only after every method's seal passes.

    ``load_reference`` returns a qualified same-patient RAS+ reference with mask,
    coverage, affine, source identity and explicit label/annotation provenance.
    Different voxel grids are permitted only with an upstream checked common
    physical frame. No registration is estimated using the private label here.
    File/process isolation remains the runner's responsibility.
    """
    raw = Path(path).read_bytes()
    digest = sha256(raw).hexdigest()
    _need(digest == expected_sha256.removeprefix("sha256:"), "sealed_file_changed")
    record = json.loads(raw)
    _need(record.get("version") == "patient-native-preflight-sealed-plans-v1"
          and record.get("status") == "complete", "complete_plan_bundle_required")
    methods = record.get("expected_methods")
    plans = record.get("plans", [])
    _need(methods == ["SEARCH", "IL", "RL"]
          and [row.get("method") for row in plans] == methods, "all_matched_methods_required")
    tools = tuple(ToolGeometry(**item) for item in record["tools"])
    prepared = []
    for row in plans:
        plan, audit = row["plan"], row["independent_geometry"]
        nominal_history = plan["history"]
        history = row["replayed_history"]
        _need(row["plan_seal"] == semantic_digest(plan), "plan_seal_changed")
        _need(plan["source_hash"] == record["source_hash"]
              and plan["decision_model_hash"] == record["decision_model_hash"]
              and plan["initial_observation_hash"] == record["initial_observation_hash"]
              and plan["context_hash"] == semantic_digest(record["patient_context"]),
              "unmatched_planning_inputs")
        # The real public-world replay uses a different outcome-scope label.
        # Every other field, including full-tool poses and tissue effects, must match.
        _need(len(history) == len(nominal_history)
              and all(((a["action_id"] == "STOP" and "outcome_scope" not in a and "outcome_scope" not in b)
                       or (a["action_id"] != "STOP"
                           and a.get("outcome_scope") == "permitted_nominal_model"
                           and b.get("outcome_scope") == "separate_evaluator_reference"))
                      and {k: v for k, v in a.items() if k != "outcome_scope"}
                          == {k: v for k, v in b.items() if k != "outcome_scope"}
                      for a, b in zip(nominal_history, history)),
              "nominal_and_replayed_history_differ")
        _need(audit.get("accepted") is True and audit.get("complete_episode") is True
              and audit.get("geometry", {}).get("feasible") is True
              and audit.get("geometry", {}).get("complete_tool_checked") is True
              and audit.get("committed_history_hash") == semantic_digest(history)
              and audit.get("source_hash") == plan["source_hash"]
              and audit.get("decision_model_hash") == plan["decision_model_hash"],
              "accepted_complete_native_geometry_required")
        _need(history and len(history) <= plan["max_steps"]
              and plan["actions"] == [step["action_id"] for step in history],
              "complete_ordered_actions_required")
        stops = [i for i, step in enumerate(history) if step["action_id"] == "STOP"]
        _need(stops == [len(history)-1] if stops else len(history) == plan["max_steps"],
              "incomplete_or_early_STOP_history")
        capsules = []
        for index, sweep in enumerate(sweeps_from_native_history(history, tools)):
            for part, (start, end, radius) in zip(("shaft", "tip"), sweep.capsules()):
                capsules.append(Capsule(str(index), part, tuple(start), tuple(end), radius))
        prepared.append((row["method"], row["plan_seal"], tuple(capsules)))

    # All planning metadata and geometry have passed before the only private read.
    reference = load_reference()
    _need(sha256(Path(path).read_bytes()).hexdigest() == digest, "sealed_file_changed_during_reference_load")
    _need(reference["patient_group"] == record["patient_context"]["patient_group"]
          and reference["public_source_hash"] == record["source_hash"]
          and reference["frame_correspondence"] == "qualified_common_RAS_mm_frame",
          "reference_patient_or_frame_mismatch")
    _need(reference["source_kind"] in {"source_automatic_annotation", "source_manual_annotation",
                                        "generated_control"}
          and isinstance(reference["label_name"], str) and bool(reference["label_name"]),
          "annotation_provenance_required")
    mask = np.array(reference["mask"], copy=True)
    coverage = np.array(reference["coverage"], copy=True)
    _need(mask.dtype == coverage.dtype == np.bool_ and mask.ndim == 3
          and mask.shape == coverage.shape and not np.any(mask & ~coverage),
          "binary_annotation_and_explicit_coverage_required")
    grid = Grid(tuple(mask.shape), tuple(map(tuple, reference["affine_ras_mm"])))
    source_digest = reference["source_sha256"].removeprefix("sha256:")
    _need(len(source_digest) == 64 and all(c in "0123456789abcdef" for c in source_digest),
          "reference_source_identity_required")
    results = []
    for method, seal, capsules in prepared:
        contacts = evaluate_contacts(grid, capsules,
            sample_reference=lambda indices: (mask[tuple(indices.T)], coverage[tuple(indices.T)]),
            budget=Budget(wall_seconds=30.))
        results.append({"method": method, "plan_seal": seal, "contacts": contacts})
    return {"version": "sealed-patient-annotation-encounters-v1", "status": "complete",
        "sealed_file_sha256": digest, "patient_group": reference["patient_group"],
        "label_name": reference["label_name"], "source_kind": reference["source_kind"],
        "reference_source_sha256": source_digest, "reference_mask_hash": array_digest(mask),
        "reference_coverage_hash": array_digest(coverage), "reference_grid_hash": grid.fingerprint,
        "private_load_calls": 1, "planning_after_reference_load": False,
        "scope": "full_tool_geometric_annotation_encounters_not_injury_or_removal",
        "clinical_injury_probability": None, "annotation_accuracy_established": False,
        "file_access_isolation": False, "methods": results}
