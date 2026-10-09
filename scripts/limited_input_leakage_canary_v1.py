#!/usr/bin/env python3
"""Generated-only action-inventory side-channel diagnostic; no training or patient IO.

This is a bounded admission canary, not a scan estimator or clinical planner.
Its negative control deliberately supplies generated annotations to the
nominal/cavity proposer while hiding only the actor's target image plane.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resectionlab.core import array_digest
from resectionlab.limited_observation_boundary import (
    LimitedPlanningSpec, VERSION as LIMITED_VERSION, SCOPE as LIMITED_SCOPE,
)
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.observed_search import observed_beam_search
from resectionlab.spatial_observations import CHANNEL_NAMES


VERSION = "limited-input-leakage-canary-v1"
MANIFEST = ROOT / "manifests/experiments/limited-input-leakage-canary-v1.json"
SOURCE_FILES = (
    "scripts/limited_input_leakage_canary_v1.py",
    "src/resectionlab/native_spatial_task.py",
    "src/resectionlab/native_proposals.py",
    "src/resectionlab/native_resection.py",
    "src/resectionlab/observed_search.py",
    "src/resectionlab/spatial_observations.py",
    "src/resectionlab/limited_observation_boundary.py",
)
CONFIG = {
    "fixture": "make_native_opening_task; six generated cells; no patient anatomy",
    "reference_a_xyz": [[4, 4, 4], [4, 4, 5]],
    "reference_b_xyz": [[4, 4, 3], [4, 4, 4]],
    "nominal_fixed": "original generated signal-derived target",
    "proposal_mode": "nominal_cavity_v1",
    "column_offsets_source_voxels": [[0, 0], [1, 1]],
    "candidate_cap": 96,
    "max_steps": 2,
    "search": {"max_calls": 6, "beam_width": 2, "seconds": 20.0,
               "transition_mode": "lazy_planning"},
    "negative_control": "generated annotations become nominal target while only actor target image is zeroed",
}
EXPECT = {
    "fixed_private_swap": "identical observation bytes, fingerprint, candidate inventory and bounded search",
    "negative_control": "equal masked image bytes but different annotation-driven action IDs and emitted endpoints",
    "missing_estimate": "nominal/cavity and SEARCH reject; generic fixed-lattice actor availability reported separately",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError("CANARY_INVARIANT_FAILED: " + message)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def frozen_manifest(path=MANIFEST):
    declaration = json.loads(Path(path).read_text())
    if (set(declaration) != {"version", "scope", "source_sha256", "config", "expected"}
            or declaration["version"] != VERSION or declaration["scope"] != "generated_software_diagnostic"
            or declaration["config"] != CONFIG or declaration["expected"] != EXPECT
            or declaration["source_sha256"] != source_hashes()):
        raise ValueError("Canary declaration or frozen source hash changed before execution")
    return declaration


def _layout(shape, cells):
    target = np.zeros(shape, np.float32)
    for cell in cells:
        target[tuple(cell)] = 1.
    return target


def _case(base, reference, *, nominal, annotation):
    return replace(base, reference_target=reference, nominal_target=nominal,
        track="annotation_assisted" if annotation else "inference_only",
        target_source_kind="supplied_annotation" if annotation else "derived_from_scan",
        target_derivation=("generated analytic annotation; negative control only" if annotation else
                           "held-fixed generated signal-derived nominal estimate; no MRI estimator"),
        proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(
            tuple(tuple(pair) for pair in CONFIG["column_offsets_source_voxels"]), CONFIG["candidate_cap"]))


def _task(base, reference, *, nominal, annotation=False):
    return NativeSpatialTask(_case(base, reference, nominal=nominal, annotation=annotation),
                             max_steps=CONFIG["max_steps"])


def _array_bytes(observation):
    names = ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm",
             "action_geometry", "action_mask", "state_features")
    return {name: (np.asarray(getattr(observation, name)).dtype.str,
                   np.asarray(getattr(observation, name)).shape,
                   np.asarray(getattr(observation, name)).tobytes()) for name in names}


def _array_hashes(observation):
    return {name: array_digest(getattr(observation, name)) for name in _array_bytes(observation)}


def _masked_target_image(observation):
    images = np.array(observation.image_channels, copy=True)
    images[CHANNEL_NAMES.index("nominal_target")] = 0.
    # Reconstructs and revalidates the whole public DTO; no source mutation.
    return replace(observation, image_channels=images)


def _endpoint_rows(inventory):
    return [{key: row[key] for key in ("family", "tool_id", "voxel", "entry_mm", "tip_mm", "feasible")}
            for row in inventory["emitted"]]


def _stable_search_cost(cost):
    return {key: cost[key] for key in ("model_transition_calls", "completed_layers",
        "call_cap_reached", "estimated_incremental_return", "root_legal_nonstop_actions",
        "evaluated_transition_prefixes", "transition_mode", "negative_prefixes_evaluated")}


def run(path=MANIFEST):
    declaration = frozen_manifest(path)
    base = make_native_opening_task().case
    original = np.array(base.nominal_target, copy=True)
    reference_a = _layout(original.shape, CONFIG["reference_a_xyz"])
    reference_b = _layout(original.shape, CONFIG["reference_b_xyz"])
    fixed = [_task(base, target, nominal=original) for target in (reference_a, reference_b)]
    inventories = [task.candidate_inventory() for task in fixed]
    observations = [task.observation() for task in fixed]
    sequences = [observed_beam_search(task, **CONFIG["search"]) for task in fixed]
    _require(fixed[0].case.reference_hash != fixed[1].case.reference_hash,
             "private reference swap did not change reference identity")
    _require(fixed[0].case.source_hash == fixed[1].case.source_hash,
             "private reference swap changed permitted source identity")
    _require(inventories[0] == inventories[1], "private swap changed candidate inventory")
    _require(_array_bytes(observations[0]) == _array_bytes(observations[1]),
             "private swap changed actor array bytes")
    _require(observations[0].fingerprint == observations[1].fingerprint,
             "private swap changed observation fingerprint")
    _require(observations[0].action_ids == observations[1].action_ids,
             "private swap changed action IDs")
    _require(sequences[0][0] == sequences[1][0], "private swap changed bounded search choice")
    _require(_stable_search_cost(sequences[0][1]) == _stable_search_cost(sequences[1][1]),
             "private swap changed stable search accounting")
    # Same committed physical actions may have different private evaluator scores.
    for task in fixed:
        for action in sequences[0][0]:
            task.step(action)
    private_scores = [task.metrics()["total_reward"] for task in fixed]
    _require(private_scores[0] != private_scores[1],
             "private reference swap did not change evaluator outcome")

    # This is deliberately annotation-assisted, even though the scan stays fixed.
    annotated = [_task(base, target, nominal=target, annotation=True)
                 for target in (reference_a, reference_b)]
    negative_inventories = [task.candidate_inventory() for task in annotated]
    negative_observations = [_masked_target_image(task.observation()) for task in annotated]
    negative_sequences = [observed_beam_search(task, **CONFIG["search"]) for task in annotated]
    _require(np.array_equal(negative_observations[0].image_channels,
                            negative_observations[1].image_channels),
             "negative-control target masking did not equalize actor images")
    _require(negative_observations[0].action_ids != negative_observations[1].action_ids,
             "negative control did not change root action IDs")
    _require(_endpoint_rows(negative_inventories[0]) != _endpoint_rows(negative_inventories[1]),
             "negative control did not change proposed endpoints")
    # Commit the same *physical* opening in both worlds. Distal strokes become
    # legal here, so actor-visible action geometry, not just hashed IDs or
    # rejected proposal ledgers, tests the side channel.
    openings = []
    for task, inventory in zip(annotated, negative_inventories):
        matching = [row for row in inventory["emitted"] if row["family"] == "exposed_opening"
            and row["voxel"] == [4, 4, 1] and row["tool_id"] == task.case.tools[0].tool_id
            and row["feasible"]]
        _require(len(matching) == 1, "controlled opening was not unique and feasible")
        openings.append(matching[0])
    _require(openings[0]["entry_mm"] == openings[1]["entry_mm"],
             "negative-control opening entries differ")
    _require(openings[0]["tip_mm"] == openings[1]["tip_mm"],
             "negative-control opening tips differ")
    for task, opening in zip(annotated, openings):
        task.step(opening["action_id"])
    after_inventories = [task.candidate_inventory() for task in annotated]
    after_observations = [_masked_target_image(task.observation()) for task in annotated]
    _require(np.array_equal(after_observations[0].image_channels,
                            after_observations[1].image_channels),
             "controlled opening changed masked actor images")
    _require(after_observations[0].action_geometry.shape != after_observations[1].action_geometry.shape,
             "annotation side channel did not change legal actor geometry")
    _require(after_observations[0].action_ids != after_observations[1].action_ids,
             "annotation side channel did not change post-opening action IDs")
    after_search = [observed_beam_search(task, **CONFIG["search"]) for task in annotated]

    # Report the lower-level actor gap separately from the route entry points.
    missing = {}
    try:
        _task(base, reference_a, nominal=None)
    except ValueError as error:
        missing["nominal_cavity_constructor"] = str(error)
    else:
        raise AssertionError("Nominal/cavity proposer accepted a missing estimate")
    generic = NativeSpatialTask(replace(base, track="inference_only", reference_target=reference_a,
        nominal_target=None, target_derivation="", proposal_mode="fixed_lattice", proposal_config=None),
        max_steps=CONFIG["max_steps"])
    missing["generic_fixed_lattice_actor_nonstop_actions"] = len(generic.observation().action_ids) - 1
    try:
        observed_beam_search(generic, **CONFIG["search"])
    except ValueError as error:
        missing["generic_fixed_lattice_search"] = str(error)
    else:
        raise AssertionError("SEARCH emitted a plan without a nominal estimate")
    try:
        LimitedPlanningSpec(LIMITED_VERSION, LIMITED_SCOPE, base.structural_intensity,
            base.affine_ras_mm, base.observed_support, None,
            {"source_kind": base.support_source_kind, "derivation": base.support_derivation},
            {"source_kind": base.target_source_kind, "derivation": base.target_derivation},
            base.access, base.tools, base.crop_shape, base.intensity_normalization,
            base.proposal_mode, CONFIG["max_steps"], generic.reward_spec)
    except ValueError as error:
        missing["sealed_limited_planning_spec"] = str(error)
    else:
        raise AssertionError("Sealed limited planning spec accepted a missing estimate")
    _require(missing["generic_fixed_lattice_actor_nonstop_actions"] > 0,
             "generic fixed-lattice actor no longer exhibits the recorded availability gap")

    result = {
        "version": VERSION, "status": "diagnostic_complete_boundary_hold",
        "scope": "generated_software_diagnostic", "manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "source_sha256": declaration["source_sha256"], "patient_count": 0,
        "training_updates": 0, "clinical_evidence": False,
        "fixed_private_swap": {"reference_hashes": [task.case.reference_hash for task in fixed],
            "source_hash": fixed[0].case.source_hash, "observation_fingerprints":
            [obs.fingerprint for obs in observations], "observation_array_hashes":
            [_array_hashes(obs) for obs in observations], "candidate_ids":
            [list(obs.action_ids) for obs in observations], "endpoints": _endpoint_rows(inventories[0]),
            "legal_masks": [obs.action_mask.tolist() for obs in observations],
            "search_decisions": [list(seq[0]) for seq in sequences],
            "search_cost_stable": [_stable_search_cost(seq[1]) for seq in sequences],
            "private_evaluator_returns": private_scores, "invariants_passed": True},
        "annotation_mask_only_negative": {"masked_image_hashes":
            [array_digest(obs.image_channels) for obs in negative_observations],
            "observation_fingerprints": [obs.fingerprint for obs in negative_observations],
            "candidate_ids": [list(obs.action_ids) for obs in negative_observations],
            "endpoints": [_endpoint_rows(inv) for inv in negative_inventories],
            "legal_masks": [obs.action_mask.tolist() for obs in negative_observations],
            "search_decisions": [list(seq[0]) for seq in negative_sequences],
            "search_cost_stable": [_stable_search_cost(seq[1]) for seq in negative_sequences],
            "same_physical_opening_then_masked_actor": {
                "opening_entry_mm": openings[0]["entry_mm"], "opening_tip_mm": openings[0]["tip_mm"],
                "masked_image_hashes": [array_digest(obs.image_channels) for obs in after_observations],
                "observation_fingerprints": [obs.fingerprint for obs in after_observations],
                "candidate_ids": [list(obs.action_ids) for obs in after_observations],
                "action_geometry_hashes": [array_digest(obs.action_geometry) for obs in after_observations],
                "legal_masks": [obs.action_mask.tolist() for obs in after_observations],
                "endpoints": [_endpoint_rows(inv) for inv in after_inventories],
                "search_decisions": [list(seq[0]) for seq in after_search]},
            "side_channel_exposed": True,
            "qualification": "Masked actor image is a counterfactual projection; SEARCH still scores the supplied annotation."},
        "missing_estimate": missing,
        "admission_limit": "No scan-derived patient estimator or shared actor admission gate is established. "
            "The generic fixed-lattice task still exposes legal actor actions without a nominal objective; "
            "nominal/cavity construction, sealed limited planning spec and observed SEARCH fail closed. "
            "No deployment plan was emitted here.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.manifest)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
