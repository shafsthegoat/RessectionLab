"""Bounded independent generated-only v2 controls; no acquired data/model I/O."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
          ROOT / "build/hybrid-observation-decision-v1/stage/resectionlab/persistent_passive.py")
spec = importlib.util.spec_from_file_location("resectionlab.independent_passive_review", SOURCE)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
from resectionlab.geometry import AccessWindow, ToolGeometry, ToolPose, check_motion
from resectionlab.native_spatial_task import NativeSpatialCase

START = time.monotonic()
signal.signal(signal.SIGALRM, lambda *_: sys.exit("28 s generated control cap"))
signal.alarm(28)
RESULTS = []


def checked(name, function):
    before = time.monotonic()
    try:
        detail = function()
        RESULTS.append({"name": name, "passed": True, "seconds": time.monotonic() - before,
                        "detail": detail})
    except Exception as error:
        RESULTS.append({"name": name, "passed": False, "seconds": time.monotonic() - before,
                        "error": type(error).__name__ + ": " + str(error)})


def inspect(task):
    task.step(p.MOVE)
    task.step(p.INSPECT)
    return task.observation()


def fingerprint(task):
    observed = task.observation()
    return (observed.fingerprint, observed.action_ids, observed.action_mask,
            observed.base.fingerprint, task._ledger, task._history,
            task.decision_model_hash, task.case.source_hash)


def finish(task, advance):
    before = task._engine.committed_mask_digests()
    if advance:
        task.step(p.ADVANCE)
    task.step(p.WITHDRAW)
    if task._protocol.stationary_pose is not None:
        assert task.observation().action_mask[0] is False
        task.step(p.WITHDRAW_STATIONARY)
    assert task.observation().action_mask[0] is True
    task.step(p.STOP)
    assert task._engine.committed_mask_digests() == before
    for row in task._history:
        assert row["clinical_deficit_probability"] is None
        assert row["physiological_response"] is None
        assert row["removed_indices_native"] == row["contact_indices_native"] == []
    return task._history[-1]["stop_reason"]


def generated_custom(*, start=(6., 6., 1.), hold=(6., 6., 11.), target=(6., 6., 11.4),
                     axis=(0., 0., 1.), length=20., bound=.2, affine=None):
    support = np.zeros((13, 13, 12), bool)
    support[10, 10, 10] = True
    image = support.astype(np.float32)
    tools = (ToolGeometry("review-other", .15, .1, length, 35., .4),
             ToolGeometry("review-moving", .15, .1, length, 35., .4))
    case = NativeSpatialCase(image, support, support, np.eye(4) if affine is None else affine,
        AccessWindow(start, (0., 0., 1.), 2., "review-generated-access"), tools,
        track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="one generated remote cell; no acquired source", nominal_target=support,
        target_source_kind="derived_from_scan", target_derivation="same generated remote cell")
    protocol = p.PassiveProtocol(tools[1].tool_id, ToolPose(start, axis), ToolPose(hold, axis))
    return p.PersistentPassiveTask(case, protocol, p.GeneratedInspectionInput(case.source_hash, "verified"),
        decision_protocol=p.PassiveDecisionProtocol(tuple(float(x) for x in target)),
        registration_measurement=p.GeneratedRegistrationMeasurement(case.source_hash, (0., 0., 0.), bound))


def prereveal_and_branches():
    variants = [p.make_passive_decision_task(translation_ras_mm=t, worst_case_error_bound_mm=b)
                for t, b in [((0., 0., 0.), .1), ((0., 0., .3), .1),
                             ((0., 0., 0.), .3), (None, None)]]
    assert len({task._state_seal for task in variants}) == 4  # private binding differs
    ready = [fingerprint(task) for task in variants]
    assert all(row == ready[0] for row in ready)
    for task in variants:
        task.step(p.MOVE)
    held = [fingerprint(task) for task in variants]
    assert all(row == held[0] for row in held)
    clones = [task.clone() for task in variants]
    clone_held = [fingerprint(task) for task in clones]
    assert all(row == clone_held[0] for row in clone_held)
    for task in variants:
        try:
            task.planning_clone()
        except ValueError:
            pass
        else:
            raise AssertionError("Unrevealed planning clone was admitted")
        task.step(p.INSPECT)
    assert [task.observation().candidate_decision["eligible"] for task in variants] == [True, False, False, False]
    assert all(task._visible_registration is None and task._phase == "held" for task in clones)
    assert all(fingerprint(task) == clone_held[0] for task in clones)
    for task in clones:
        task.step(p.INSPECT)
    assert [fingerprint(task) for task in clones] == [fingerprint(task) for task in variants]
    reasons = [finish(task, index == 0) for index, task in enumerate(variants)]
    return {"private_variants": 4, "public_ready_held_and_cached_clone_states_identical": True,
            "observed_eligibility": [True, False, False, False], "stop_reasons": reasons}


def changed_unchanged_and_declined():
    alternatives = [p.make_passive_decision_task(translation_ras_mm=t, worst_case_error_bound_mm=b)
                    for t, b in [((0., 0., 0.), .1), ((0., 0., 0.), .2), ((0., 0., .2), .1)]]
    observations = [inspect(task) for task in alternatives]
    assert all(obs.candidate_decision["eligible"] for obs in observations)
    endpoints = [list(obs.candidate_decision["candidateSourceRasMm"]) for obs in observations]
    assert endpoints[0] == endpoints[1] and endpoints[0] != endpoints[2]
    alternatives[0].step(p.ADVANCE)
    assert alternatives[0]._operative_record()["moving_tool"]["pose"]["tip_mm"] == endpoints[0]
    assert alternatives[0]._operative_record()["other_tool"]["pose"]["tip_mm"] == [5.5, 6., 6.5]
    finish(alternatives[0], False)
    finish(alternatives[1], True)
    assert finish(alternatives[2], False) == "declined_eligible_generated_advance"
    assert alternatives[0]._history[2]["pose_after"] == alternatives[0]._history[3]["pose_before"]
    return {"same_action_and_endpoint_despite_different_sufficient_bounds": endpoints[:2],
            "changed_offered_endpoint": endpoints[2], "voluntary_refusal_truthful": True}


def support_and_reach_bounds():
    domain = generated_custom()
    nominal = ToolPose((6., 6., 11.4), (0., 0., 1.))
    assert check_motion(domain._moving_tool(), domain._protocol.hold_pose, nominal,
                        domain._scene(), domain.case.access).feasible
    decision = inspect(domain).candidate_decision
    assert not decision["eligible"] and decision["reason"] == "bounded_translation_leaves_source_image_tip_domain"
    reach = p.make_passive_decision_task(translation_ras_mm=(0., 0., .3), worst_case_error_bound_mm=.1)
    nominal = ToolPose((6.5, 6., 6.5), (0., 0., 1.))
    assert check_motion(reach._moving_tool(), reach._protocol.hold_pose, nominal,
                        reach._scene(), reach.case.access, other_tools=reach._other_tools()).feasible
    decision = inspect(reach).candidate_decision
    assert not decision["eligible"] and decision["reason"] == "bounded_translation_violates_access_direction_or_reach"
    return {"nominal_geometry_passes_but_uncertain_tip_domain_and_reach_refused": True}


def oblique_lower_depth_bound():
    axis = np.array((.5, 0., np.sqrt(.75)))
    start = np.array((6., 6., 1.))
    task = generated_custom(start=start, hold=start + .001 * axis, target=start + .111 * axis,
                            axis=axis, length=6., bound=.1)
    decision = inspect(task).candidate_decision
    assert not decision["eligible"] and decision["reason"] == "bounded_translation_violates_access_direction_or_reach"
    return {"nominal_forward_step_exceeds_bound_but_uncertain_access_depth_is_negative": True}


def pair_uncertainty_refusal():
    source = p.make_passive_decision_task()
    protocol = replace(source._protocol,
        stationary_pose=ToolPose((6.15, 6., 6.5), (0., 0., 1.)),
        stationary_exit_pose=ToolPose((6.15, 6., .5), (0., 0., 1.)))
    task = p.PersistentPassiveTask(source.case, protocol, source._inspection_input,
        decision_protocol=source._decision_protocol, registration_measurement=source._registration_input)
    nominal = ToolPose((6.5, 6., 6.2), (0., 0., 1.))
    assert check_motion(task._moving_tool(), protocol.hold_pose, nominal, task._scene(),
                        task.case.access, other_tools=task._other_tools()).feasible
    decision = inspect(task).candidate_decision
    assert not decision["eligible"] and decision["nativeFailure"] == "TOOL_COLLISION"
    before = task._state_seal
    try:
        task.step(p.ADVANCE)
    except ValueError:
        pass
    else:
        raise AssertionError("Uncertain tool-pair collision advance committed")
    assert task._state_seal == before
    return {"nominal_pair_clear_but_uncertain_envelope_refused": True,
            "failed_attempt_keeps_state": True}


def legacy_unchanged():
    episode = p.run_passive_development_episode(paired=True)
    assert episode["episodeId"] == "sha256:33cbb8bdea5873acece2461fbf1c992791ff7cae18ba2099c4bc1724cd76717a"
    assert episode["schema"] == "resectionlab.generated-persistent-passive-episode.v1"
    return {"canonical_v1_paired_episode_id": episode["episodeId"]}


for name, test in [
    ("pre_reveal_cache_and_clone_invariance", prereveal_and_branches),
    ("changed_unchanged_and_voluntarily_declined_decisions", changed_unchanged_and_declined),
    ("uncertain_tip_domain_and_axial_reach", support_and_reach_bounds),
    ("oblique_uncertain_access_depth", oblique_lower_depth_bound),
    ("uncertain_stationary_pair_collision", pair_uncertainty_refusal),
    ("legacy_v1_exact_episode_identity", legacy_unchanged),
]:
    checked(name, test)
    print(json.dumps(RESULTS[-1]), flush=True)

signal.alarm(0)
receipt = {"review": "independent-generated-registration-decision-controls-v1",
           "source": str(SOURCE.relative_to(ROOT)),
           "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           "elapsed_seconds": time.monotonic() - START,
           "bounded_process_seconds": 28, "acquired_data_reads": False,
           "model_reads_or_fitting": False, "results": RESULTS,
           "passed": all(row["passed"] for row in RESULTS)}
output = Path(__file__).with_name("receipt.json")
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": str(output), "passed": receipt["passed"],
                  "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
raise SystemExit(0 if receipt["passed"] else 1)
