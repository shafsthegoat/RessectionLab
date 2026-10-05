"""Analytical integrity/unknownness controls; no patient or training claims."""
from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.functional_evidence import FunctionalEvidence, anatomy_identity, population_prior_sensitivity
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import ImagingError, load_case, save_case
from resectionlab.native_refinement import inspect_native_refinement, run_native_refinement, recheck_native_replay
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS
from resectionlab.native_simulation import make_native_patient_simulator
from resectionlab.worlds import WorldGeneratorConfig


def analytic_case():
    tissue = np.ones((7, 7, 6), bool)
    target = np.zeros_like(tissue)
    target[1:6, 1:6, 3:] = True
    return CaseData("analytic-functional-control", tissue.astype(float), {"target": target}, np.eye(4),
                    (SourceRef("fixture", "simulated:analytical-integrity-control", provenance="simulated"),),
                    brain_mask=tissue)


def evidence_for(case, *, shift=.3, motor=True, language=True):
    values = np.zeros(case.mri.shape, np.float32)
    values[2:4, :, :] = .8
    coverage = np.ones(values.shape, bool)
    coverage[0] = False
    return FunctionalEvidence("analytic-prior-control", values if motor else None, values if language else None,
        coverage if motor else None, coverage if language else None, np.eye(4), array_digest(case.mri),
        anatomy_identity(case), {"analytic_released_map": {"source": SourceRef("fixture-prior",
            "simulated:analytical-map", "a" * 64, provenance="prior").to_dict(),
            "mni_ras_to_patient_ras_mm": np.eye(4).tolist()}},
        WorldGeneratorConfig(translation_scale_mm=(shift, shift, shift)))


def test_explicit_attachment_versions_planning_but_preserves_registration_parent():
    case = analytic_case()
    evidence = evidence_for(case)
    attached = case.revised(functional_evidence=evidence)
    assert attached.planning_hash != case.planning_hash
    assert attached.prior_registration_input_hash == case.planning_hash
    assert attached.semantic_hash != case.semantic_hash
    assert attached.functional_evidence.review_status == "alignment_review_required"
    assert evidence.to_manifest()["expert_approval"] is False
    assert evidence.to_manifest()["clinical_deficit_probability"] is None


def test_source_bound_roundtrip_preserves_raw_values_coverage_and_uncertainty(tmp_path):
    case = analytic_case()
    case = case.revised(functional_evidence=evidence_for(case))
    path = save_case(case, tmp_path / "case.ressectionlab")
    restored = load_case(path)
    assert restored.semantic_hash == case.semantic_hash
    assert restored.planning_hash == case.planning_hash
    assert restored.functional_evidence.to_manifest() == case.functional_evidence.to_manifest()
    np.testing.assert_array_equal(restored.functional_evidence.motor, case.functional_evidence.motor)
    np.testing.assert_array_equal(restored.functional_evidence.motor_coverage, case.functional_evidence.motor_coverage)
    assert not restored.functional_evidence.uncertainty.deterministic


@pytest.mark.parametrize("edit", ["image", "labels", "affine"])
def test_image_or_anatomy_edit_rejects_old_evidence(edit):
    case = analytic_case()
    attached = case.revised(functional_evidence=evidence_for(case))
    if edit == "image":
        values = case.mri.copy(); values[0, 0, 0] = 2
        changes = {"mri": values}
    elif edit == "labels":
        values = case.compartments["target"].copy(); values[0, 0, 0] = True
        changes = {"compartments": {"target": values}}
    else:
        values = case.affine.copy(); values[0, 3] = 1
        changes = {"affine": values}
    with pytest.raises(ValueError, match="different image, anatomy or physical frame"):
        attached.revised(**changes)


def test_raw_unknown_stays_unknown_and_missing_language_stays_absent():
    evidence = evidence_for(analytic_case(), language=False)
    assert np.all(evidence.motor[0] == 0)
    motor, language = evidence.planning_arrays()
    assert np.all(motor[0] == 1) and language is None
    assert evidence.language_coverage is None
    with pytest.raises(ValueError, match="expert approval"):
        replace(evidence, review_status="expert_reviewed")
    with pytest.raises(ValueError):
        evidence.motor.setflags(write=True)


def test_constructor_rejects_uncovered_signal_and_reflection():
    evidence = evidence_for(analytic_case())
    values = evidence.motor.copy(); values[0] = .5
    with pytest.raises(ValueError, match="Uncovered"):
        replace(evidence, motor=values)
    records = evidence.to_manifest()["source_records"]
    records["analytic_released_map"]["mni_ras_to_patient_ras_mm"][0][0] = -1
    with pytest.raises(ValueError, match="non-reflecting"):
        replace(evidence, source_records=records)


def native_options():
    return {"access": AccessWindow((3, 3, -.5), (0, 0, 1), 3.), "tools": (NATIVE_GENERIC_TOOLS[0],),
            "selected_entry_mm": (3, 3, -.5), "selected_target_mm": (3, 3, 4), "max_steps": 3}


def test_native_factory_carries_evidence_coverage_and_hidden_world_without_actor_leakage():
    case = analytic_case()
    case = case.revised(functional_evidence=evidence_for(case, shift=.6))
    sim = make_native_patient_simulator(case, **native_options())
    assert sim.config.evidence_available == (True, True)
    assert sim.config.world_generator == case.functional_evidence.uncertainty
    assert sim.config.derivation["functional_evidence"]["evidence_hash"] == case.functional_evidence.fingerprint
    assert np.all(sim.config.nominal_motor[0] == 1)
    np.testing.assert_array_equal(sim.config.nominal_motor_coverage, case.functional_evidence.motor_coverage)
    first = sim.reset(10)
    hidden = sim._hidden_motor.copy()
    second = sim.reset(11)
    assert not np.array_equal(hidden, sim._hidden_motor)
    np.testing.assert_array_equal(first.action_features, second.action_features)
    assert sim.fresh().decision_model_hash == sim.decision_model_hash
    with pytest.raises(ValueError, match="Uncertainty differs"):
        sim.fresh(world_generator=WorldGeneratorConfig())
    with pytest.raises(ValueError, match="Uncertainty differs"):
        make_native_patient_simulator(case, world_generator=WorldGeneratorConfig(), **native_options())


def test_facade_hard_exclusion_blocks_identical_complete_tool_route():
    case = analytic_case()
    clear = inspect_native_refinement(case, **native_options())
    assert clear["legalNonStopActions"] > 0
    forbidden = np.ones(case.mri.shape, bool)
    blocked = inspect_native_refinement(case, hard_exclusion=forbidden,
        hard_exclusion_provenance="analytic_known_obstacle", **native_options())
    assert blocked["legalNonStopActions"] == 0
    assert blocked["reasons"]
    assert blocked["route_binding"]["evidence_and_constraints"]["hard_exclusion"]["mask_hash"] == array_digest(forbidden)
    with pytest.raises(ValueError, match="both"):
        inspect_native_refinement(case, hard_exclusion=forbidden, **native_options())


def test_legacy_structural_case_has_no_new_manifest_fields(tmp_path):
    case = analytic_case()
    assert "functional_evidence" not in case.to_manifest()
    legacy = load_case(save_case(case, tmp_path / "legacy.ressectionlab"))
    assert legacy.semantic_hash == case.semantic_hash
    assert legacy.functional_evidence is None
    sim = make_native_patient_simulator(case, **native_options())
    assert sim.config.evidence_available == (False, False)
    assert sim.metrics()["motor_surrogate"] is None
    assert "functional_evidence" not in sim.config.derivation


def test_corrupted_evidence_manifest_rejected_even_with_valid_outer_bundle(tmp_path):
    case = analytic_case()
    case = case.revised(functional_evidence=evidence_for(case))
    path = save_case(case, tmp_path / "case.ressectionlab")
    with ZipFile(path) as archive:
        manifest, payload = json.loads(archive.read("manifest.json")), archive.read("arrays.npz")
    manifest["functional_evidence"]["manifest"]["review_status"] = "engineering_qc_only"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest)); archive.writestr("arrays.npz", payload)
    with pytest.raises(ImagingError, match="differs"):
        load_case(path)


def test_functional_checkpoint_replay_and_resume_bind_frozen_uncertainty(tmp_path):
    case = analytic_case()
    case = case.revised(functional_evidence=evidence_for(case))
    options = {**native_options(), "max_steps": 1}
    report = run_native_refinement(case, tmp_path, budget_seconds=1., seed=11, **options)
    assert report["gradient_steps"] > 0
    replay = report["replay"]
    assert replay["evidence_and_constraints"]["functional_evidence"]["evidence_hash"] == case.functional_evidence.fingerprint
    assert replay["metrics"]["functional_evidence_available"] == {"motor": True, "language": True}
    assert recheck_native_replay(case, replay, **options)["route_binding"] == replay["route_binding"]
    same = run_native_refinement(case, tmp_path, budget_seconds=1., seed=11, resume=True, **options)
    assert same["route_binding"] == report["route_binding"]
    changed = case.revised(functional_evidence=replace(case.functional_evidence,
        uncertainty=WorldGeneratorConfig(translation_scale_mm=(1., 1., 1.))))
    with pytest.raises(ValueError, match="Resume route or settings changed"):
        run_native_refinement(changed, tmp_path, budget_seconds=1., seed=11, resume=True, **options)


def test_zero_perturbation_coverage_does_not_become_known_empty():
    case = analytic_case()
    case = case.revised(functional_evidence=evidence_for(case, shift=0))
    sim = make_native_patient_simulator(case, **native_options())
    assert not sim._hidden_known_coverage[0].any()
    assert np.all(sim._hidden_motor[0] == 1)


# Reuse the real importer with an analytical source fixture; this is not a
# population-training episode or a patient-source substitute.
from test_prior_proposals import registered_fixture
from resectionlab.prior_proposals import import_registered_prior_proposals


def test_selecting_prior_preserves_unreviewed_original_and_exact_parent_binding(registered_fixture, tmp_path):
    case, kwargs, _ = registered_fixture
    proposals = import_registered_prior_proposals(case, **kwargs)
    evidence = population_prior_sensitivity(proposals,
        uncertainty=WorldGeneratorConfig(translation_scale_mm=(.5, .5, .5)))
    attached = proposals.revised(functional_evidence=evidence)
    restored = load_case(save_case(attached, tmp_path / "functional.ressectionlab"))
    assert restored.planning_hash != proposals.planning_hash
    assert restored.prior_registration_input_hash == proposals.planning_hash
    prior = next(iter(restored.prior_proposals.values()))
    assert prior.view_only and not prior.planning_eligible
    assert prior.review_status == restored.functional_evidence.review_status == "alignment_review_required"
    assert restored.functional_evidence.language is None
