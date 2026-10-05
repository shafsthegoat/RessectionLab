"""Independent tiny-fixture review of the read-only axis inspection boundary."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from resectionlab import native_axis_refinement as facade
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.geometry import AccessWindow
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionEngine
from resectionlab.simulation import RewardSpec
from resectionlab.structural_evidence import StructuralEvidence, structural_frame_hash
from resectionlab.worlds import WorldGeneratorConfig, content_hash


def case_fixture():
    tissue = np.zeros((9, 9, 10), bool)
    tissue[1:8, 1:8, 1:9] = True
    target = np.zeros_like(tissue)
    target[2:7, 2:7, 5:8] = True
    return CaseData("axis-inspection-independent", tissue.astype(np.float32), {"target": target}, np.eye(4),
        (SourceRef("analytic", "synthetic://axis-inspection-review", provenance="simulated"),), brain_mask=tissue)


def arguments(case):
    return {"access_ras": AccessWindow((4, 4, .5), (0, 0, 1), 6), "tools": (NATIVE_GENERIC_TOOLS[0],),
        "acknowledge_neighboring_columns": True, "expected_case_hash": case.semantic_hash,
        "expected_planning_hash": case.planning_hash, "reward": RewardSpec(), "world_generator": WorldGeneratorConfig(),
        "proposal_config": AxisColumnProposalConfig(offsets_source_voxels=((0, 0),), max_primary_rays=1)}


def inspect(case, **changes):
    return facade.inspect_axis_planning(case, **(arguments(case) | changes))


def test_lps_source_preserves_exact_canonical_ras_geometry_without_resampling():
    ras = case_fixture()
    lps = replace(ras, frame="LPS+", affine=np.diag([-1., -1., 1., 1.]))
    first, second = inspect(ras).to_dict(), inspect(lps).to_dict()
    assert first["status"] == second["status"] == "ready"
    assert first["binding"]["access"] == second["binding"]["access"]
    assert first["binding"]["native_affine_ras_mm"] == second["binding"]["native_affine_ras_mm"] == np.eye(4).tolist()
    assert [row["geometry"] for row in first["actions"]] == [row["geometry"] for row in second["actions"]]
    assert first["binding"]["source_grid_hashes"] == second["binding"]["source_grid_hashes"]
    assert first["binding"]["binding_hash"] != second["binding"]["binding_hash"]


@pytest.mark.parametrize("angle", [.01, .17, .29, .61, .93, 1.13, 1.57, 2.03])
def test_oblique_lps_inspection_survives_unit_normal_machine_roundoff(angle):
    c, s = np.cos(angle), np.sin(angle)
    cz, sz = np.cos(.7 * angle), np.sin(.7 * angle)
    cx, sx = np.cos(.2), np.sin(.2)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    affine = np.eye(4)
    affine[:3, :3] = rz @ ry @ rx
    affine[:3, 3] = [17., -26., 4.]
    ras = replace(case_fixture(), affine=affine)
    lps = replace(ras, affine=np.diag([-1., -1., 1., 1.]) @ affine, frame="LPS+")
    access = AccessWindow(affine[:3, :3] @ [4, 4, .5] + affine[:3, 3], affine[:3, 2], 6)
    first, second = (inspect(case, access_ras=access).to_dict() for case in (ras, lps))
    assert first["status"] == second["status"] == "ready"
    np.testing.assert_allclose(second["binding"]["access"]["normal_inward"], access.normal_inward, rtol=0, atol=2e-15)
    assert first["binding"]["native_affine_ras_mm"] == second["binding"]["native_affine_ras_mm"]
    assert [row["geometry"] for row in first["actions"]] == [row["geometry"] for row in second["actions"]]


def test_known_oblique_lps_roundtrip_does_not_reject_one_ulp_normal_change():
    raw = np.array([-1.168650074969459, .3986177272005879, .32715561200889304])
    normal = raw / np.linalg.norm(raw)
    first_axis = np.cross([0., 0., 1.], normal)
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(normal, first_axis)
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack((first_axis, second_axis, normal))
    affine[:3, 3] = [17., -26., 4.]
    center = affine[:3, :3] @ [4, 4, .5] + affine[:3, 3]
    access = AccessWindow(center, raw, 6)
    ras = replace(case_fixture(), affine=affine)
    lps = replace(ras, affine=np.diag([-1., -1., 1., 1.]) @ affine, frame="LPS+")
    first, second = (inspect(case, access_ras=access).to_dict() for case in (ras, lps))
    assert first["status"] == second["status"] == "ready"
    np.testing.assert_array_equal(second["binding"]["access"]["center_mm"], access.center_mm)
    np.testing.assert_allclose(second["binding"]["access"]["normal_inward"], access.normal_inward, rtol=0, atol=2e-15)
    assert [row["geometry"] for row in first["actions"]] == [row["geometry"] for row in second["actions"]]


@pytest.mark.parametrize("frame", ["RAS+", "LPS+"])
@pytest.mark.parametrize("change", ["normal", "center", "radius", "window_id"])
def test_factory_cannot_use_normalization_tolerance_to_change_requested_access(monkeypatch, frame, change):
    case = case_fixture()
    if frame == "LPS+":
        case = replace(case, frame=frame, affine=np.diag([-1., -1., 1., 1.]))
    original = facade.native_config_from_case
    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        access = result.access
        changes = {"normal": {"normal_inward": (1e-10, 0., 1.)},
            "center": {"center_mm": (access.center_mm[0] + 1e-12, *access.center_mm[1:])},
            "radius": {"radius_mm": access.radius_mm + 1e-12},
            "window_id": {"window_id": "different-window"}}[change]
        return replace(result, access=replace(access, **changes))
    monkeypatch.setattr(facade, "native_config_from_case", altered)
    with pytest.raises(ValueError, match="(?i)(access|preserv|canonical)"):
        inspect(case)


@pytest.mark.parametrize("prohibition", [{"structural_coverage": "full_head"},
    {"allow_nonzero_mri_access_support": False}, {"skull_stripped": False}])
def test_explicit_source_prohibition_wins_over_both_acknowledgments(prohibition):
    case = replace(case_fixture(), brain_mask=None,
        source_refs=(SourceRef("observed", "fixture://review-not-real-patient"),),
        metadata={"source_collection": {"name": "UCSF-PDGM"}, **prohibition})
    with pytest.raises(ValueError, match="(?i)(review|support|full.head)"):
        inspect(case, acknowledge_estimated_support=True)


def test_neighboring_acknowledgment_never_substitutes_for_estimated_support():
    case = replace(case_fixture(), brain_mask=None, metadata={"skull_stripped": True})
    with pytest.raises(ValueError, match="estimated-support"):
        inspect(case)
    result = inspect(case, acknowledge_estimated_support=True).to_dict()
    assert result["binding"]["source_support"]["estimated_support_acknowledged"] is True
    assert result["binding"]["access_status"] == "hypothetical_unverified_cortical_access"
    assert result["candidate_eligible"] is result["removal_authorized"] is False


def test_pending_structural_proposal_cannot_be_promoted_by_support_acknowledgment():
    case = replace(case_fixture(), source_refs=(SourceRef("observed", "fixture://observed-analytic"),))
    item = StructuralEvidence("pending", case.brain_mask, array_digest(case.mri), structural_frame_hash(case),
        None, "b" * 64, "c" * 64, "synthetic model proposal")
    case = replace(case, structural_evidence={item.evidence_id: item})
    with pytest.raises(ValueError, match="BRAIN_MASK_REVIEW_REQUIRED"):
        inspect(case, acknowledge_estimated_support=True)


@pytest.mark.parametrize("access", [AccessWindow((4, 4, .5), (.001, 0, 1), 6),
    AccessWindow((4.0005, 4, .5), (0, 0, 1), 6)])
def test_unsupported_access_raises_instead_of_publishing_stop(access):
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        inspect(case_fixture(), access_ras=access)


def test_complete_blocked_inventory_preserves_denominator_and_no_authority():
    case = case_fixture()
    blocked = np.ones(case.mri.shape, bool)
    result = inspect(case, hard_exclusion=blocked, hard_exclusion_provenance="explicit analytic barrier").to_dict()
    assert result["status"] == "no_actionable_moves" and result["inventory_complete"]
    assert result["actions"] == [{"action_id": "STOP", "kind": "stop", "geometry": None, "native_preview": None}]
    assert result["inventory"]["attempts"] and all(not row["feasible"] for row in result["inventory"]["attempts"])
    assert result["inventory"]["batch"]["slot_count"] == 1
    assert result["accounting"] == {"gradient_steps": 0, "executed_transitions": 0, "native_commits": 0, "simulated_removed_volume_mm3": 0.}


def test_snapshot_is_detached_and_binding_excludes_timing():
    case = case_fixture()
    first, second = inspect(case), inspect(case)
    assert first.binding_hash == second.binding_hash
    exported = first.to_dict()
    exported["binding"]["tools"][0]["tip_radius_mm"] = 999
    assert first.to_dict()["binding"]["tools"][0]["tip_radius_mm"] != 999
    with pytest.raises(FrozenInstanceError):
        first.payload_json = "{}"
    body = first.to_dict()["binding"]
    assert body.pop("binding_hash") == content_hash(body)
    with pytest.raises(ValueError, match="Stale axis inspection binding"):
        inspect(case, reward=RewardSpec(normal_per_mm3=.7), expected_binding_hash=first.binding_hash)


def test_cancellation_after_one_preview_never_returns_partial_inventory(monkeypatch):
    case = case_fixture()
    original = NativeResectionEngine.preview_stroke
    state = {"cancelled": False, "calls": 0}
    def preview(*args, **kwargs):
        result = original(*args, **kwargs)
        state["calls"] += 1
        state["cancelled"] = True
        return result
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", preview)
    with pytest.raises(InterruptedError):
        inspect(case, cancelled=lambda: state["cancelled"])
    assert state["calls"] == 1


@pytest.mark.parametrize("target", ["reward", "tool", "world"])
def test_final_callback_mutation_cannot_publish_an_old_model_binding(monkeypatch, target):
    case = case_fixture()
    kwargs = arguments(case)
    # Do not mutate the shared module-level tool catalog in an adversarial test.
    kwargs["tools"] = (replace(NATIVE_GENERIC_TOOLS[0]),)
    original = facade._json
    state = {"serialized": False, "changed": False}
    def serialize(value):
        result = original(value)
        if isinstance(value, dict) and value.get("role") == "inspection":
            state["serialized"] = True
        return result
    def cancelled():
        if state["serialized"] and not state["changed"]:
            obj, name, value = {"reward": (kwargs["reward"], "normal_per_mm3", .75),
                "tool": (kwargs["tools"][0], "shaft_radius_mm", .7),
                "world": (kwargs["world_generator"], "translation_scale_mm", (1., 0., 0.))}[target]
            object.__setattr__(obj, name, value)
            state["changed"] = True
        return False
    monkeypatch.setattr(facade, "_json", serialize)
    with pytest.raises((RuntimeError, ValueError), match="(?i)(changed|stale|frozen)"):
        facade.inspect_axis_planning(case, **kwargs, cancelled=cancelled)
    assert state["changed"]


def test_final_callback_source_metadata_mutation_cannot_reuse_cached_case_identity(monkeypatch):
    case = case_fixture()
    kwargs = arguments(case)
    original = facade._json
    state = {"serialized": False, "changed": False}
    def serialize(value):
        result = original(value)
        if isinstance(value, dict) and value.get("role") == "inspection":
            state["serialized"] = True
        return result
    def cancelled():
        if state["serialized"] and not state["changed"]:
            object.__setattr__(case, "case_id", "changed-during-final-callback")
            state["changed"] = True
        return False
    monkeypatch.setattr(facade, "_json", serialize)
    with pytest.raises((RuntimeError, ValueError), match="(?i)(changed|stale|source|frozen)"):
        facade.inspect_axis_planning(case, **kwargs, cancelled=cancelled)
    assert state["changed"]


def test_preexisting_source_metadata_replacement_cannot_reuse_cached_case_identity(monkeypatch):
    case = case_fixture()
    kwargs = arguments(case)
    object.__setattr__(case, "case_id", "changed-before-inspection")
    def forbidden(*args, **kwargs):
        raise AssertionError("Stale source must fail before native construction")
    monkeypatch.setattr(facade, "native_config_from_case", forbidden)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(source|stale|identity|changed)"):
        facade.inspect_axis_planning(case, **kwargs)
