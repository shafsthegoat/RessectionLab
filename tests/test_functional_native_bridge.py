"""Analytic propagation controls, not patient or clinical outcome benchmarks."""
from dataclasses import replace
import json

import numpy as np
import pytest

from resectionlab import desktop_bridge as bridge
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.functional_evidence import FunctionalEvidence, anatomy_identity
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import load_case, save_case
from resectionlab.native_axis_refinement import inspect_axis_planning
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, native_config_from_case
from resectionlab.simulation import RewardSpec
from resectionlab.worlds import WorldGeneratorConfig


def case_with_evidence():
    support = np.ones((7, 7, 8), bool)
    target = np.zeros_like(support)
    target[1:6, 1:6, 2:7] = True
    case = CaseData("analytic-functional-propagation", support.astype(np.float32),
        {"target": target}, np.eye(4),
        (SourceRef("analytic", "simulated:functional-propagation", provenance="simulated"),),
        brain_mask=support)
    coverage = np.ones_like(support)
    coverage[0] = False
    values = (np.indices(support.shape)[1] / 6).astype(np.float32)
    values[~coverage] = 0
    evidence = FunctionalEvidence("analytic-sensitivity", values, None, coverage, None, np.eye(4),
        array_digest(case.mri), anatomy_identity(case),
        {"analytic-map": {"source": SourceRef("analytic-prior", "simulated:map-control",
            array_digest(values), provenance="prior").to_dict(),
            "mni_ras_to_patient_ras_mm": np.eye(4).tolist()}},
        WorldGeneratorConfig(translation_scale_mm=(.4, .5, .2)))
    return case.revised(functional_evidence=evidence)


def axis_options(case, **updates):
    return dict(access_ras=AccessWindow((3., 3., -.5), (0., 0., 1.), 4.),
        tools=(NATIVE_GENERIC_TOOLS[0],), acknowledge_neighboring_columns=True,
        expected_case_hash=case.semantic_hash, expected_planning_hash=case.planning_hash,
        reward=RewardSpec(), world_generator=case.functional_evidence.uncertainty,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 1), max_steps=1) | updates


def test_axis_actual_worlds_and_fresh_preserve_evidence_coverage_and_change_costs():
    case = case_with_evidence()
    evidence = case.functional_evidence
    native = native_config_from_case(case, access=axis_options(case)["access_ras"],
                                    tools=(NATIVE_GENERIC_TOOLS[0],))
    motor, language = evidence.planning_arrays()
    sim = AxisColumnNativeSimulator(native, proposal_config=AxisColumnProposalConfig(((0, 0),), 1),
        nominal_motor=motor, nominal_language=language, nominal_motor_coverage=evidence.motor_coverage,
        nominal_language_coverage=evidence.language_coverage, functional_evidence_record=evidence.to_manifest(),
        world_generator=evidence.uncertainty, max_steps=1)
    fresh = sim.fresh()
    assert fresh.decision_model_hash == sim.decision_model_hash
    assert fresh.config.derivation["functional_evidence"]["evidence_hash"] == evidence.fingerprint
    np.testing.assert_array_equal(fresh.config.nominal_motor_coverage, evidence.motor_coverage)
    assert np.all(fresh.config.nominal_motor[~evidence.motor_coverage] == 1.)
    with pytest.raises(ValueError, match="World generator"):
        sim.fresh(world_generator=WorldGeneratorConfig())
    fields, costs, inventories = [], [], []
    for seed in (11, 23):
        fresh.reset(seed)
        fields.append(fresh._hidden_motor.copy())
        actions = fresh.proposed_actions()
        inventories.append(tuple(action.action_id for action in actions))
        assert len(actions) > 1
        fresh.step(actions[1].action_id)
        metrics = fresh.metrics()
        costs.append(metrics["motor_surrogate"])
        assert metrics["language_surrogate"] is None
        assert metrics["clinical_deficit_probability"] is None
    assert inventories[0] == inventories[1]
    assert not np.array_equal(*fields), "Different seed labels alone are insufficient"
    assert costs[0] != costs[1], "Same actual geometry must respond to changed anatomical scenarios"


def test_axis_binding_carries_population_evidence_and_rejects_world_override():
    case = case_with_evidence()
    report = inspect_axis_planning(case, **axis_options(case)).to_dict()
    binding = report["binding"]
    assert binding["functional_evidence"]["evidence_hash"] == case.functional_evidence.fingerprint
    assert binding["world_generator_hash"] == case.functional_evidence.uncertainty.fingerprint
    assert binding["functional_evidence_available"] == {"motor": True, "language": False}
    assert binding["population_priors_used"] is True
    assert binding["vascular_evidence_status"] == "unassessed"
    assert report["accounting"]["executed_transitions"] == 0
    with pytest.raises(ValueError, match="Uncertainty"):
        inspect_axis_planning(case, **axis_options(case, world_generator=WorldGeneratorConfig()))


def test_explicit_hard_exclusion_blocks_actual_axis_inventory_with_functional_evidence():
    case = case_with_evidence()
    open_report = inspect_axis_planning(case, **axis_options(case)).to_dict()
    hard = np.ones(case.mri.shape, bool)
    blocked = inspect_axis_planning(case, **axis_options(case, hard_exclusion=hard,
        hard_exclusion_provenance="analytic fully blocked control; no vascular source")).to_dict()
    assert open_report["legal_non_stop_actions"] > 0
    assert blocked["legal_non_stop_actions"] == 0
    assert blocked["binding"]["hard_exclusion"]["mask_hash"] == array_digest(hard)
    assert blocked["binding"]["functional_evidence"] == open_report["binding"]["functional_evidence"]


def test_saved_case_and_desktop_frozen_run_options_keep_exact_evidence(tmp_path):
    case = case_with_evidence()
    path = save_case(case, tmp_path / "control.rslab")
    restored = load_case(path)
    assert restored.semantic_hash == case.semantic_hash
    config = bridge.BridgeSession._route_config(bridge._CaseEntry(case, {}, {}), None)
    persisted = json.loads(json.dumps(config))
    options = bridge.BridgeSession._run_options(persisted, restored)
    assert options["world_generator"].fingerprint == case.functional_evidence.uncertainty.fingerprint
    persisted["worldGenerator"]["translation_scale_mm"][0] = 0.
    with pytest.raises(bridge.BridgeError, match="evidence or uncertainty"):
        bridge.BridgeSession._run_options(persisted, restored)


def test_desktop_inspection_uses_case_evidence_and_accounts_for_array_memory(tmp_path):
    case = case_with_evidence()
    bare = replace(case, functional_evidence=None)
    evidence = case.functional_evidence
    expected_bytes = sum(array.nbytes for array in (evidence.motor, evidence.motor_coverage, evidence.affine_ras_mm))
    assert bridge.BridgeSession._case_array_bytes(case) - bridge.BridgeSession._case_array_bytes(bare) == expected_bytes
    events = []
    runtime = bridge.BridgeRuntime(tmp_path / "transfers", events.append)
    def send(identity, operation, args):
        runtime.submit({"id": identity, "op": operation, "args": args})
        assert runtime.wait_idle(15)
        rows = [row for row in events if row.get("id") == identity and row["event"] in {"result", "error"}]
        assert len(rows) == 1 and rows[0]["event"] == "result", rows
        return rows[0]["result"]
    try:
        descriptor = runtime.session._install_case(case, {}, bridge._Request("fixture"))
        assert descriptor["functionalEvidence"]["evidence_hash"] == evidence.fingerprint
        routes = send("routes", "generateNativeRoutes", {"caseHash": case.semantic_hash})
        route = next(row for row in routes["candidates"] if row["geometry"]["feasible"])
        report = send("axis", "inspectAxisPlanning", {"caseHash": case.semantic_hash,
            "planningHash": case.planning_hash, "routeId": route["route_id"],
            "routePlanningModelHash": route["planning_model_hash"],
            "toolIds": [NATIVE_GENERIC_TOOLS[0].tool_id], "acknowledgeNeighboringColumns": True,
            "acknowledgeEstimatedSupport": False})["inspection"]
        assert report["binding"]["functional_evidence"]["evidence_hash"] == evidence.fingerprint
        assert report["binding"]["world_generator_hash"] == evidence.uncertainty.fingerprint
        ready = send("refinement", "inspectRefinement", {"caseHash": case.semantic_hash,
            "routeId": route["route_id"]})
        assert ready["route_binding"]["evidence_and_constraints"]["functional_evidence"]["evidence_hash"] == evidence.fingerprint
    finally:
        runtime.close()


def test_actual_desktop_checkpoint_disk_recheck_and_export_retain_same_evidence(tmp_path, monkeypatch):
    """A bounded analytic integration control, not a learning benchmark."""
    import torch
    torch.set_num_threads(1)
    case = case_with_evidence()
    evidence_hash = case.functional_evidence.fingerprint
    world_hash = case.functional_evidence.uncertainty.fingerprint
    events = []
    runtime = bridge.BridgeRuntime(tmp_path / "transfers", events.append, run_dir=tmp_path / "runs")
    def send(identity, operation, args):
        runtime.submit({"id": identity, "op": operation, "args": args})
        assert runtime.wait_idle(30)
        rows = [row for row in events if row.get("id") == identity and row["event"] in {"result", "error"}]
        assert len(rows) == 1 and rows[0]["event"] == "result", rows
        return rows[0]["result"]
    try:
        runtime.session._install_case(case, {}, bridge._Request("fixture"))
        routes = send("routes", "generateNativeRoutes", {"caseHash": case.semantic_hash})
        route = next(row for row in routes["candidates"] if row["geometry"]["feasible"])
        trained = send("train", "trainPatient", {"caseHash": case.semantic_hash,
            "routeId": route["route_id"], "budgetSeconds": 5, "seed": 11})
        assert trained["training"]["gradient_steps"] > 0
        assert trained["training"]["replay_status"] == "accepted_independent_geometry"
        assert trained["config"]["functionalEvidenceHash"] == evidence_hash
        run = {"caseHash": case.semantic_hash, "runId": trained["runId"]}
        from resectionlab import native_functional_evaluation as functional
        original_evaluate = functional._evaluate
        def interrupt_evaluation(*args, **kwargs):
            runtime.submit({"id": "cancel-evaluation", "op": "cancel", "args": {
                "requestId": "evaluate-interrupted"}})
            raise InterruptedError("analytic cancellation after the durable seal")
        monkeypatch.setattr(functional, "_evaluate", interrupt_evaluation)
        runtime.submit({"id": "evaluate-interrupted", "op": "evaluateCandidate", "args": run})
        assert runtime.wait_idle(10)
        assert [row["event"] for row in events if row.get("id") == "evaluate-interrupted"
                and row["event"] in {"result", "error", "cancelled"}] == ["cancelled"]
        _, sealed_manifest = runtime.session._read_run(case, trained["runId"])
        assert sealed_manifest["evaluationSealed"] is True
        assert sealed_manifest["functionalEvaluationStatus"] == "sealed_interrupted"
        monkeypatch.setattr(functional, "_evaluate", original_evaluate)
        evaluated = send("evaluate", "evaluateCandidate", run)
        assert evaluated["evaluationSealed"] is True
        assessment = evaluated["functionalAssessment"]
        assert assessment["functional_evidence_hash"] == evidence_hash
        assert assessment["event_report"]["distinct_anatomy_transforms"] == 3
        assert assessment["event_report"]["world_partition"]["role"] == "final_evaluation"
        repeated = send("evaluate-again", "evaluateCandidate", run)
        assert repeated["functionalAssessment"] == assessment
        runtime.submit({"id": "cannot-train", "op": "trainPatient", "args": {
            "caseHash": case.semantic_hash, "resumeRunId": trained["runId"]}})
        assert runtime.wait_idle(5)
        rejected = next(row for row in events if row.get("id") == "cannot-train" and row["event"] == "error")
        assert rejected["error"]["code"] == "RUN_EVALUATION_SEALED"
        runtime.session.run_reports.clear()  # Force actual persisted checkpoint/history/event checking.
        replayed = send("replay", "replayTraining", run)
        assert replayed["clinicalDeficitProbability"] is None
        assert replayed["functionalAssessment"] == assessment
        assert replayed["functionalAssessmentScope"] == "entire_frozen_sequence_not_current_replay_prefix"
        path = tmp_path / "candidate.json"
        send("export", "exportCandidate", {**run, "path": str(path)})
        exported = json.loads(path.read_text())["candidate"]
        assert exported["functional_assessment"] == assessment
        binding = exported["evidence_and_constraints"]
        assert binding["functional_evidence"]["evidence_hash"] == evidence_hash
        assert binding["world_generator_hash"] == world_hash
        assert exported["route_binding"]["evidence_and_constraints"] == binding
        assert exported["metrics"]["world_generator_fingerprint"] == world_hash
        assert exported["metrics"]["functional_evidence_available"] == {"motor": True, "language": False}
        assert exported["metrics"]["language_surrogate"] is None
        assert exported["metrics"]["clinical_deficit_probability"] is None
    finally:
        runtime.close()
