"""Generated arrays only; no patient inference, training, or clinical claim."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest
from resectionlab.data_policy import DataPolicyError, require_admitted_model
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.observed_search import observed_beam_search

from resectionlab.research_estimate_planning import (
    SCOPE, Abstention, DevelopmentUseDeclaration, FrozenResearchPlan,
    ResearchEstimatePlanningSpec, ScanEstimate, ScanSource, _estimate_case, _frame, _nominal_task,
    evaluate_sealed_research_plan, research_planning_from_estimates, source_set_hash,
)


def fixture_spec(*, coverage=None, target_mask=None, qc_status="pass", extra_modality=False,
        roi_start=None, roi_stop=None, source_time_unknown=False):
    generated = make_native_opening_task()
    native = generated.case
    image, affine = native.structural_intensity, native.affine_ras_mm
    source = ScanSource("T1", image, affine, "sha256:" + "a" * 64,
        None if source_time_unknown else "2026-10-01T00:00:00+00:00",
        array_digest(image), _frame(image.shape, affine),
        "source_attested_preoperative" if source_time_unknown else "timestamped",
        "sha256:" + "c" * 64 if source_time_unknown else None)
    sources = (source,)
    if extra_modality:
        second = image * 2
        sources += (ScanSource("FLAIR", second, affine, "sha256:" + "b" * 64,
            "2026-10-01T00:00:00+00:00", array_digest(second), source.frame_sha256),)
    source_hash = source_set_hash(sources)
    coverage = np.ones(image.shape, bool) if coverage is None else coverage
    target_mask = native.nominal_target if target_mask is None else target_mask
    support = ScanEstimate("brain_envelope_candidate", native.observed_support, coverage,
        "sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64,
        source_hash, source.frame_sha256, array_digest(native.observed_support), array_digest(coverage),
        "2026-10-01T01:00:00+00:00", "candidate full-brain envelope; no cortical approval",
        qc_status, "unknown", "unknown")
    target = ScanEstimate("whole_tumor_candidate", target_mask, coverage,
        "sha256:" + "4" * 64, "sha256:" + "5" * 64, "sha256:" + "6" * 64,
        source_hash, source.frame_sha256, array_digest(np.asarray(target_mask, bool)), array_digest(coverage),
        "2026-10-01T02:00:00+00:00", "whole-tumor imaging hypothesis; not prescribed removal",
        qc_status, "unknown", "unknown")
    declaration = DevelopmentUseDeclaration(SCOPE, "GENERATED:unit-fixture", "DEVELOPMENT",
        source_hash, semantic_digest(support.identity()), semantic_digest(target.identity()),
        "nominal_route_comparison")
    return ResearchEstimatePlanningSpec(sources, "T1", support, target, declaration,
        native.access, native.tools, generated.reward_spec, 2, "2026-10-02T00:00:00+00:00",
        roi_start, roi_stop)


def stop_planner(task):
    assert task.metrics()["planning_estimator_only"] is True
    return ("STOP",), {"model_transition_calls": 1, "actor_forward_calls": 0}


def plan(spec, method="SEARCH", planner=stop_planner):
    return research_planning_from_estimates(spec, method=method,
        configuration_hash=semantic_digest({"method": method, "fixture": "generated"}), planner=planner)


def target_reaching_planner(task):
    sequence, costs = observed_beam_search(task, max_calls=128, beam_width=16,
        seconds=5., transition_mode="lazy_planning")
    return sequence, {"model_transition_calls": costs["model_transition_calls"], "actor_forward_calls": 0}


def test_four_methods_get_identical_permitted_observation_and_keep_model_hashes():
    spec = fixture_spec(extra_modality=True)
    seen = []
    def planner(task):
        observation = task.observation()
        seen.append((observation.fingerprint, observation.action_ids,
            observation.action_geometry.tobytes(), observation.action_mask.tobytes(),
            task.case.source_hash))
        return stop_planner(task)
    plans = [plan(spec, method, planner) for method in ("SEARCH", "IL", "RL", "HYBRID")]
    assert all(type(row) is FrozenResearchPlan for row in plans)
    assert seen[0] == seen[1] == seen[2] == seen[3]
    assert len({row.input_hash for row in plans}) == 1
    assert spec.target.model_sha256 and spec.support.model_sha256
    assert spec.target.training_overlap_status == spec.support.training_lineage_status == "unknown"
    with pytest.raises(DataPolicyError, match="WEIGHT_LINEAGE_UNVERIFIED"):
        require_admitted_model(spec.support.model_sha256, "production_support")


def test_private_target_swap_changes_score_only_after_full_route_seal():
    spec = fixture_spec()
    sealed = plan(spec, planner=target_reaching_planner)
    assert len(sealed.action_ids) == 2 and "STOP" not in sealed.action_ids
    observed = []
    def loader(reference):
        def load():
            sealed.assert_intact()
            observed.append(sealed.seal_hash)
            return reference
        return load
    first = evaluate_sealed_research_plan(sealed, spec, load_reference=loader(spec.target.mask))
    second = evaluate_sealed_research_plan(sealed, spec, load_reference=loader(np.zeros_like(spec.target.mask)))
    assert observed == [sealed.seal_hash, sealed.seal_hash]
    assert first["status"] == second["status"] == "generated_target_only_diagnostic"
    assert first["target_removed_mm3"] > second["target_removed_mm3"] == 0
    assert first["physical_history_hash"] == second["physical_history_hash"] == sealed.physical_history_hash
    assert first["reference_hash"] != second["reference_hash"]
    assert first["clinical_deficit_probability"] is second["patient_generalization"] is None


@pytest.mark.parametrize("mutation", ["model", "output", "source_array", "source_file", "frame", "late", "role", "clinical"])
def test_bound_identity_and_development_role_fail_closed(mutation):
    spec = fixture_spec()
    if mutation == "model":
        changed = replace(spec.support, model_sha256="sha256:" + "f" * 64)
        update = {"support": changed}
    elif mutation == "output":
        with pytest.raises(ValueError, match="output or coverage changed"):
            replace(spec.target, output_sha256="sha256:" + "f" * 64)
        return
    elif mutation == "source_array":
        with pytest.raises(ValueError, match="array bytes changed"):
            replace(spec.sources[0], image=spec.sources[0].image + 1)
        return
    elif mutation == "source_file":
        changed = replace(spec.sources[0], source_file_sha256="sha256:" + "f" * 64)
        update = {"sources": (changed,)}
    elif mutation == "frame":
        affine = spec.sources[0].affine_ras_mm.copy()
        affine[0, 3] += 1
        with pytest.raises(ValueError, match="frame changed"):
            replace(spec.sources[0], affine_ras_mm=affine)
        return
    elif mutation == "late":
        changed = replace(spec.target, available_at="2026-10-03T00:00:00+00:00")
        update = {"target": changed, "declaration": replace(spec.declaration,
            target_identity_sha256=semantic_digest(changed.identity()))}
        with pytest.raises(ValueError, match="available before"):
            replace(spec, **update)
        return
    elif mutation == "role":
        with pytest.raises(ValueError, match="DEVELOPMENT"):
            replace(spec.declaration, split_role="MEASUREMENT_EVAL")
        return
    else:
        with pytest.raises(ValueError, match="DEVELOPMENT"):
            replace(spec.declaration, clinical_use_permitted=True)
        return
    with pytest.raises(ValueError, match="bind every preoperative source|declaration does not bind"):
        replace(spec, **update)


@pytest.mark.parametrize("change,reason", [
    ("partial_coverage", "PARTIAL_SPATIAL_COVERAGE"),
    ("qc", "ESTIMATE_QC_INCOMPLETE"),
    ("target_outside_support", "TARGET_SUPPORT_INCONSISTENT"),
])
def test_uncertain_or_inconsistent_region_abstains_before_planner(change, reason):
    if change == "partial_coverage":
        coverage = np.ones((9, 9, 7), bool)
        coverage[0, 0, 0] = False
        spec = fixture_spec(coverage=coverage)
    elif change == "qc":
        spec = fixture_spec(qc_status="abstain")
    else:
        target = np.zeros((9, 9, 7), bool)
        target[0, 0, 0] = True
        spec = fixture_spec(target_mask=target)
    def forbidden(_):
        pytest.fail("No action inventory or planner may run after abstention")
    result = plan(spec, planner=forbidden)
    assert result == Abstention(reason, spec.fingerprint)


def test_remote_unknown_region_can_be_excluded_with_exact_physical_roi():
    coverage = np.ones((9, 9, 7), bool)
    coverage[0, 0, 0] = False  # Remote unknown, outside declared physical workspace.
    spec = fixture_spec(coverage=coverage, roi_start=(2, 2, 0), roi_stop=(8, 8, 7))
    native = _estimate_case(spec, spec.target.mask)
    np.testing.assert_allclose(native.affine_ras_mm[:3, 3],
        spec.sources[0].affine_ras_mm[:3, :3] @ np.asarray(spec.roi_start)
        + spec.sources[0].affine_ras_mm[:3, 3])
    result = plan(spec)
    assert type(result) is FrozenResearchPlan


def test_cropped_physical_roi_seals_search_before_private_target_swap():
    coverage = np.ones((9, 9, 7), bool)
    coverage[0, 0, 0] = False
    spec = fixture_spec(coverage=coverage, roi_start=(2, 2, 0), roi_stop=(8, 8, 7))
    sealed = plan(spec, planner=target_reaching_planner)
    first = evaluate_sealed_research_plan(sealed, spec, load_reference=lambda: spec.target.mask)
    second = evaluate_sealed_research_plan(sealed, spec,
        load_reference=lambda: np.zeros_like(spec.target.mask))
    assert first["target_removed_mm3"] > second["target_removed_mm3"] == 0
    assert first["physical_history_hash"] == second["physical_history_hash"] == sealed.physical_history_hash


def test_outside_fov_shaft_remains_machine_readable_unknown_not_safety_certificate():
    spec = fixture_spec()
    task = _nominal_task(spec)
    affine = spec.sources[0].affine_ras_mm
    tools = {tool.tool_id: tool for tool in spec.tools}
    outside = False
    for row in task.candidate_inventory()["ledger"]:
        entry, tip = np.asarray(row["entry_mm"], float), np.asarray(row["tip_mm"], float)
        direction = tip - entry
        length = float(np.linalg.norm(direction))
        if length <= 0:
            continue
        proximal = entry - tools[row["tool_id"]].working_length_mm * direction / length
        proximal_voxel = np.linalg.solve(affine[:3, :3], proximal - affine[:3, 3])
        outside |= bool(np.any(proximal_voxel < -0.5)
                        or np.any(proximal_voxel > np.asarray(spec.support.mask.shape) - 0.5))
    assert outside  # The generated fixture actually exercises this limit.
    sealed = plan(spec)
    assert sealed.coverage_scope == "native_grid_in_image_tool_supercover_only"
    assert sealed.outside_source_fov_tool_feasibility_assessed is False
    assert sealed.clinical_use_permitted is False
    report = evaluate_sealed_research_plan(sealed, spec, load_reference=lambda: spec.target.mask)
    assert report["coverage_scope"] == sealed.coverage_scope
    assert report["outside_source_fov_tool_feasibility_assessed"] is False
    assert report["clinical_use_permitted"] is False


def test_tool_footprint_crossing_unqualified_roi_abstains_before_planner():
    coverage = np.ones((9, 9, 7), bool)
    coverage[:4, :, :] = False
    spec = fixture_spec(coverage=coverage, roi_start=(4, 4, 0), roi_stop=(7, 7, 7))
    def forbidden(_):
        pytest.fail("Tool crossing an unknown boundary must not reach a planner")
    assert plan(spec, planner=forbidden) == Abstention("TOOL_FOOTPRINT_OUTSIDE_COVERAGE", spec.fingerprint)


def test_target_outside_qualified_roi_abstains_instead_of_clipping_objective():
    spec = fixture_spec(roi_start=(0, 0, 0), roi_stop=(9, 9, 4))
    def forbidden(_):
        pytest.fail("An incomplete target objective must not reach the planner")
    assert plan(spec, planner=forbidden) == Abstention("TARGET_OUTSIDE_QUALIFIED_ROI", spec.fingerprint)


@pytest.mark.parametrize("scenario,reason", [
    ("partial", "PARTIAL_SPATIAL_COVERAGE"),
    ("qc", "ESTIMATE_QC_INCOMPLETE"),
    ("footprint", "TOOL_FOOTPRINT_OUTSIDE_COVERAGE"),
    ("target", "TARGET_OUTSIDE_QUALIFIED_ROI"),
])
def test_forged_stop_seal_cannot_reach_private_loader_after_abstention(scenario, reason):
    if scenario == "partial":
        coverage = np.ones((9, 9, 7), bool)
        coverage[0, 0, 0] = False
        spec = fixture_spec(coverage=coverage)
    elif scenario == "qc":
        spec = fixture_spec(qc_status="abstain")
    elif scenario == "footprint":
        coverage = np.ones((9, 9, 7), bool)
        coverage[:4, :, :] = False
        spec = fixture_spec(coverage=coverage, roi_start=(4, 4, 0), roi_stop=(7, 7, 7))
    else:
        spec = fixture_spec(roi_start=(0, 0, 0), roi_stop=(9, 9, 4))
    # Caller can construct a content-valid plan object, but not bypass the
    # evaluator's own permitted-input preflight before private reference IO.
    digest = semantic_digest({"forged": "nominal metadata only"})
    forged = FrozenResearchPlan(spec.fingerprint, "SEARCH", digest, digest, digest,
        ("STOP",), digest, spec.horizon,
        {"model_transition_calls": 0, "actor_forward_calls": 0})
    def forbidden():
        pytest.fail("A rejected research input must never invoke private reference IO")
    with pytest.raises(ValueError, match=reason):
        evaluate_sealed_research_plan(forged, spec, load_reference=forbidden)


def test_unknown_exact_acquisition_time_remains_unknown_with_source_attestation():
    spec = fixture_spec(source_time_unknown=True)
    assert spec.sources[0].acquired_at is None
    assert spec.sources[0].availability_basis == "source_attested_preoperative"
    assert spec.sources[0].preoperative_evidence_sha256 is not None
    assert type(plan(spec)) is FrozenResearchPlan
    with pytest.raises(ValueError, match="preoperative source evidence"):
        replace(spec.sources[0], preoperative_evidence_sha256=None)


def test_private_loader_is_not_called_for_changed_seal_or_spec():
    spec = fixture_spec()
    sealed = plan(spec)
    def forbidden():
        pytest.fail("Private reference loader must follow plan preflight")
    object.__setattr__(sealed, "physical_history_hash", "sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="changed"):
        evaluate_sealed_research_plan(sealed, spec, load_reference=forbidden)
    fresh = plan(spec)
    changed = fixture_spec(extra_modality=True)
    with pytest.raises(ValueError, match="binding mismatch"):
        evaluate_sealed_research_plan(fresh, changed, load_reference=forbidden)


def test_reference_shape_rejected_after_seal_without_any_clinical_score():
    spec = fixture_spec()
    sealed = plan(spec)
    report = evaluate_sealed_research_plan(sealed, spec, load_reference=lambda: np.zeros((2, 2, 2)))
    assert report == {"status": "reference_rejected", "plan_seal_hash": sealed.seal_hash,
        "coverage_scope": "native_grid_in_image_tool_supercover_only",
        "outside_source_fov_tool_feasibility_assessed": False,
        "clinical_use_permitted": False}
