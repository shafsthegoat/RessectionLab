"""Small physical-grid checks for read-only expanded planning inspection."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_refinement import inspect_axis_planning
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionEngine
from resectionlab.simulation import RewardSpec
from resectionlab.worlds import WorldGeneratorConfig, content_hash


def fixture(*, affine=None, frame="RAS+"):
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros(tissue.shape, bool)
    target[1:6, 1:6, 2:7] = True
    return CaseData("axis-inspection-analytic", tissue.astype(np.float32), {"target": target},
        np.eye(4) if affine is None else affine,
        (SourceRef("analytic", "simulated:axis-inspection", provenance="simulated"),),
        brain_mask=tissue, frame=frame)


def options(case, **changes):
    return dict(access_ras=AccessWindow((3, 3, -.5), (0, 0, 1), 4), tools=NATIVE_GENERIC_TOOLS,
        acknowledge_neighboring_columns=True, expected_case_hash=case.semantic_hash,
        expected_planning_hash=case.planning_hash, reward=RewardSpec(),
        world_generator=WorldGeneratorConfig(),
        proposal_config=AxisColumnProposalConfig(((0, 0), (1, 0)), 4), max_steps=2) | changes


def test_complete_initial_inventory_preserves_sources_and_never_commits(monkeypatch):
    case = fixture()
    arrays = [case.mri, case.affine, case.brain_mask, case.compartments["target"]]
    before = [array_digest(array) for array in arrays]
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not execute transitions")
    monkeypatch.setattr(NativeResectionEngine, "commit_preview", forbidden)
    monkeypatch.setattr(AxisColumnNativeSimulator, "step", forbidden)
    inspection = inspect_axis_planning(case, **options(case))
    report = inspection.to_dict()
    assert report["status"] == "ready" and report["inventory_complete"]
    assert report["legal_non_stop_actions"] == 4
    assert report["inventory"]["batch"]["slot_count"] == 4
    assert len(report["inventory"]["attempts"]) == 4
    assert report["actions"][0]["action_id"] == "STOP"
    assert [item["action_id"] for item in report["actions"]][1:] == report["inventory"]["certified_action_ids"]
    assert report["accounting"] == dict(gradient_steps=0, executed_transitions=0,
                                         native_commits=0, simulated_removed_volume_mm3=0.)
    assert not report["candidate_eligible"] and not report["removal_authorized"]
    assert report["clinical_deficit_probability"] is None
    assert report["binding"]["functional_evidence_available"] == {"motor": False, "language": False}
    assert report["binding"]["world_role"] is None
    assert "tool_geometry_outside_image_unassessed" in report["unknowns"]
    assert all("tool_geometry_outside_image_unassessed" in action["native_preview"]["geometry_unknowns"]
               for action in report["actions"][1:])
    assert [array_digest(array) for array in arrays] == before
    assert content_hash({k: v for k, v in report.items() if k != "inspection_hash"}) == report["inspection_hash"]
    assert content_hash({k: v for k, v in report["binding"].items() if k != "binding_hash"}) == inspection.binding_hash
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    report["actions"].clear()
    assert inspection.to_dict()["actions"]
    with pytest.raises(FrozenInstanceError):
        inspection.payload_json = "{}"


def test_canonical_ras_access_converts_once_for_lps_source_and_oblique_grid():
    angle = .31
    affine = np.eye(4)
    affine[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    affine[:3, 3] = [10, -20, 7]
    access = AccessWindow(affine[:3, :3] @ [3, 3, -.5] + affine[:3, 3], affine[:3, 2], 4)
    ras = fixture(affine=affine)
    lps = fixture(affine=np.diag([-1., -1., 1., 1.]) @ affine, frame="LPS+")
    one = inspect_axis_planning(ras, **options(ras, access_ras=access)).to_dict()
    two = inspect_axis_planning(lps, **options(lps, access_ras=access)).to_dict()
    assert one["binding"]["native_affine_ras_mm"] == two["binding"]["native_affine_ras_mm"]
    assert one["binding"]["access"] == two["binding"]["access"]
    assert one["binding"]["source_frame"] == "RAS+" and two["binding"]["source_frame"] == "LPS+"
    assert [a["geometry"] for a in one["actions"]] == [a["geometry"] for a in two["actions"]]
    # Physical actions agree; source provenance and model identities remain distinct.
    assert one["binding"]["decision_model_hash"] != two["binding"]["decision_model_hash"]


@pytest.mark.parametrize("normal,center,reason", [
    ((.2, 0, 1), (3, 3, -.5), "UNSUPPORTED_NONAXIAL_ACCESS"),
    ((0, 0, 1), (3.2, 3, -.5), "UNSUPPORTED_FRACTIONAL_TRANSVERSE_ORIGIN"),
])
def test_unsupported_access_is_refused_not_published_as_stop_only(normal, center, reason):
    case = fixture()
    with pytest.raises(ValueError, match=reason):
        inspect_axis_planning(case, **options(case, access_ras=AccessWindow(center, normal, 4)))


def test_estimated_support_and_neighboring_column_acknowledgements_are_independent():
    case = replace(fixture(), brain_mask=None, metadata={"skull_stripped": True})
    with pytest.raises(ValueError, match="estimated-support acknowledgement"):
        inspect_axis_planning(case, **options(case))
    with pytest.raises(ValueError, match="neighboring-column acknowledgement"):
        inspect_axis_planning(case, **options(case, acknowledge_estimated_support=True,
                                             acknowledge_neighboring_columns=False))
    report = inspect_axis_planning(case, **options(case, acknowledge_estimated_support=True)).to_dict()
    support = report["binding"]["source_support"]
    assert support["evidence_type"] == "estimated" and support["estimated_support_acknowledged"]
    assert not support["cortical_access_permitted"] and case.brain_mask is None


@pytest.mark.parametrize("metadata", [{"structural_coverage": "full_head", "skull_stripped": True},
                                       {"skull_stripped": True, "allow_nonzero_mri_access_support": False}, {}])
def test_support_prohibition_survives_both_acknowledgements(metadata):
    case = replace(fixture(), brain_mask=None, metadata=metadata)
    with pytest.raises(ValueError, match="Reviewed brain support"):
        inspect_axis_planning(case, **options(case, acknowledge_estimated_support=True))


def test_real_unreviewed_brain_is_not_promoted_by_acknowledgements():
    case = replace(fixture(), source_refs=(SourceRef("unreviewed", "local:unreviewed"),))
    with pytest.raises(ValueError, match="BRAIN_MASK_REVIEW_REQUIRED"):
        inspect_axis_planning(case, **options(case, acknowledge_estimated_support=True))


def test_fallback_ledger_retains_both_attempts_and_exact_accepted_geometry():
    case = fixture()
    barrier = np.zeros(case.mri.shape, bool)
    barrier[3, 3, 5] = True
    report = inspect_axis_planning(case, **options(case,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 2),
        hard_exclusion=barrier, hard_exclusion_provenance="explicit analytic distal obstacle")).to_dict()
    attempts = report["inventory"]["attempts"]
    assert [a["phase"] for a in attempts] == ["primary", "fallback", "primary", "fallback"]
    assert all(not a["feasible"] for a in attempts if a["phase"] == "primary")
    assert report["proposal_accounting"]["preview_calls"] == 4
    for action in report["actions"][1:]:
        assert action["native_preview"]["phase"] == "fallback"
        assert action["native_preview"]["unexecuted_contained_cell_count"] > 0
        assert action["geometry"]["entry_mm"] == [3., 3., -.5]
        assert action["geometry"]["tip_mm"] == [3., 3., 2.]
        assert not action["native_preview"]["independent_history_checked"]
    assert report["binding"]["hard_exclusion"]["mask_hash"] == array_digest(barrier)


def test_complete_all_rejected_inventory_is_not_unsupported_success():
    case = fixture()
    report = inspect_axis_planning(case, **options(case, hard_exclusion=np.ones(case.mri.shape, bool),
        hard_exclusion_provenance="analytic full barrier")).to_dict()
    assert report["status"] == "no_actionable_moves" and report["inventory_complete"]
    assert report["legal_non_stop_actions"] == 0
    assert report["inventory"]["attempts"] and all(not a["feasible"] for a in report["inventory"]["attempts"])


def test_stale_source_refuses_before_any_native_construction(monkeypatch):
    case = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Stale inputs must be rejected before numerical work")
    monkeypatch.setattr("resectionlab.native_axis_refinement.native_config_from_case", forbidden)
    with pytest.raises(ValueError, match="Stale source"):
        inspect_axis_planning(case, **options(case, expected_case_hash="sha256:stale"))
    with pytest.raises(ValueError, match="Stale source"):
        inspect_axis_planning(case, **options(case, expected_planning_hash="sha256:stale"))


def test_changed_reward_tool_world_and_rule_cannot_reuse_inspection_binding():
    case = fixture()
    original = inspect_axis_planning(case, **options(case))
    replay = inspect_axis_planning(case, **options(case, expected_binding_hash=original.binding_hash))
    assert replay.binding_hash == original.binding_hash
    changes = [dict(reward=RewardSpec(action_cost=.5)),
               dict(tools=(replace(NATIVE_GENERIC_TOOLS[0], shaft_radius_mm=.5), NATIVE_GENERIC_TOOLS[1])),
               dict(world_generator=WorldGeneratorConfig(translation_scale_mm=(.1, 0, 0))),
               dict(proposal_config=AxisColumnProposalConfig(((0, 0),), 2))]
    for changed in changes:
        with pytest.raises(ValueError, match="Stale axis inspection binding"):
            inspect_axis_planning(case, **options(case, expected_binding_hash=original.binding_hash, **changed))


def test_cancellation_during_preview_publishes_nothing_and_does_not_commit(monkeypatch):
    case = fixture()
    flag = {"cancelled": False, "previews": 0}
    original = NativeResectionEngine.preview_stroke
    def preview(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        flag["previews"] += 1
        flag["cancelled"] = True
        assert engine.revision == 0 and not engine.removed_mask.any()
        return result
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", preview)
    with pytest.raises(InterruptedError):
        inspect_axis_planning(case, **options(case, cancelled=lambda: flag["cancelled"]))
    assert flag["previews"] == 1


def test_hard_exclusion_requires_its_own_provenance():
    case = fixture()
    with pytest.raises(ValueError, match="own provenance"):
        inspect_axis_planning(case, **options(case, hard_exclusion=np.zeros(case.mri.shape, bool)))


def test_existing_selected_route_readiness_remains_exact_after_expanded_inspection():
    from resectionlab.native_refinement import inspect_native_refinement
    case = fixture()
    selected = dict(access=options(case)["access_ras"], tools=(NATIVE_GENERIC_TOOLS[0],),
                    selected_entry_mm=(3., 3., -.5), selected_target_mm=(3., 3., 6.),
                    max_steps=2)
    before = inspect_native_refinement(case, **selected)
    expanded = inspect_axis_planning(case, **options(case)).to_dict()
    after = inspect_native_refinement(case, **selected)
    assert after["route_binding"] == before["route_binding"]
    assert after["action_ids"] == before["action_ids"]
    assert after["route_binding"]["mode"] == "exact_selected_route"
    assert after["route_binding"]["candidate_targets_mm"] == [[3., 3., 6.]]
    assert expanded["legal_non_stop_actions"] > after["legalNonStopActions"]
    assert expanded["binding"]["decision_model_hash"] != after["decision_model_hash"]
