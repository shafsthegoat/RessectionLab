"""Independent bridge-boundary review; small analytic arrays only."""
from dataclasses import replace
import json
import threading

import numpy as np
import pytest

from resectionlab import desktop_bridge as bridge
from resectionlab import native_axis_refinement as facade
from resectionlab.core import CaseData, SourceRef, array_digest, freeze_json, thaw_json
from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS
from resectionlab.planning import SearchConfig, generate_candidate_routes
from resectionlab.structural_evidence import StructuralEvidence, structural_frame_hash
from resectionlab.worlds import content_hash


def analytic_case(**changes):
    support = np.zeros((9, 9, 10), bool)
    support[1:8, 1:8, 1:9] = True
    target = np.zeros_like(support)
    target[2:7, 2:7, 5:8] = True
    case = CaseData("independent-bridge-review", support.astype(np.float32), {"target": target},
        np.eye(4), (SourceRef("analytic", "simulated:bridge-review", provenance="simulated"),),
        brain_mask=support)
    return replace(case, **changes)


@pytest.fixture
def runtime(tmp_path):
    events = []
    instance = bridge.BridgeRuntime(tmp_path / "transfers", events.append)
    yield instance, events
    instance.close()


def install(runtime, case):
    instance, _ = runtime
    instance.session._install_case(case, {}, bridge._Request("setup"))
    access = AccessWindow(case.voxel_to_world((4., 4., .5)), case.affine[:3, 2], 6., "explicit-analytic-window")
    # The cache contains an actual freshly evaluated route, never imported metrics.
    result = generate_candidate_routes(case, windows=(access,), tools=(NATIVE_GENERIC_TOOLS[0],),
        explicit_targets=(("target", case.voxel_to_world((4., 4., 7.))),),
        config=SearchConfig(max_windows=1, targets_per_compartment=1)).to_dict()
    instance.session.cases[case.semantic_hash].routes = freeze_json(result)
    route = next(row for row in result["candidates"] if row["geometry"]["feasible"])
    return {"caseHash": case.semantic_hash, "planningHash": case.planning_hash,
        "routeId": route["route_id"], "routePlanningModelHash": route["planning_model_hash"],
        "toolIds": [NATIVE_GENERIC_TOOLS[0].tool_id], "acknowledgeNeighboringColumns": True,
        "acknowledgeEstimatedSupport": False}


def invoke(runtime, args, identity="inspect", **message):
    instance, events = runtime
    instance.submit({"id": identity, "op": "inspectAxisPlanning", "args": args, **message})
    assert instance.wait_idle(5)
    terminal = [row for row in events if row.get("id") == identity and row["event"] in {"result", "error", "cancelled"}]
    assert len(terminal) == 1
    return terminal[0]


def reseal(report):
    binding = report["binding"]
    binding["binding_hash"] = content_hash({key: value for key, value in binding.items() if key != "binding_hash"})
    report["inspection_hash"] = content_hash({key: value for key, value in report.items() if key != "inspection_hash"})
    return facade.AxisPlanningInspection(json.dumps(report))


@pytest.mark.parametrize("change", ["requested_center", "native_access", "source_frame", "source_shape", "native_affine",
                                    "native_direction", "conversion_tolerance"])
def test_rehashed_facade_binding_cannot_disagree_with_cached_source_window(runtime, monkeypatch, change):
    case = analytic_case()
    args = install(runtime, case)
    original = facade.inspect_axis_planning
    def malformed(*positional, **keywords):
        report = original(*positional, **keywords).to_dict()
        binding = report["binding"]
        if change == "requested_center":
            binding["requested_access_ras"]["center_mm"][0] += 1.
        elif change == "native_access":
            binding["access"]["radius_mm"] += 1.
        elif change == "source_frame":
            binding["source_frame"] = "LPS+"
        elif change == "source_shape":
            binding["source_shape"][0] += 1
        elif change == "native_affine":
            binding["native_affine_ras_mm"][0][3] += 1.
        elif change == "native_direction":
            binding["access"]["normal_inward"][0] = 1e-10
        else:
            binding["access_conversion"]["normal_absolute_tolerance"] = 1.
        return reseal(report)
    monkeypatch.setattr(facade, "inspect_axis_planning", malformed)
    result = invoke(runtime, args)
    assert result["event"] == "error", "A self-consistent hash does not authenticate the requested source/window"
    assert "result" not in result


def test_translated_anisotropic_lps_matches_ras_action_geometry(runtime):
    affine = np.array([[0., -2., 0., 17.], [1.5, 0., 0., -26.], [0., 0., 1.25, 4.], [0., 0., 0., 1.]])
    ras = analytic_case(affine=affine)
    lps = replace(ras, frame="LPS+", affine=np.diag([-1., -1., 1., 1.]) @ affine)
    first = invoke(runtime, install(runtime, ras), "ras")["result"]
    second = invoke(runtime, install(runtime, lps), "lps")["result"]
    assert first["anchorWindowRas"] == second["anchorWindowRas"]
    assert first["inspection"]["binding"]["native_affine_ras_mm"] == affine.tolist()
    assert second["inspection"]["binding"]["native_affine_ras_mm"] == affine.tolist()
    assert [row["geometry"] for row in first["inspection"]["actions"]] == [row["geometry"] for row in second["inspection"]["actions"]]
    assert first["inspection"]["binding"]["source_grid_hashes"] == second["inspection"]["binding"]["source_grid_hashes"]


def test_returned_binding_must_match_explicit_previous_binding(runtime, monkeypatch):
    args = install(runtime, analytic_case())
    first = invoke(runtime, args, "first")["result"]["inspection"]["binding"]["binding_hash"]
    original = facade.inspect_axis_planning
    def changed(*positional, **keywords):
        report = original(*positional, **keywords).to_dict()
        report["binding"]["max_steps"] += 1
        return reseal(report)
    monkeypatch.setattr(facade, "inspect_axis_planning", changed)
    result = invoke(runtime, args | {"expectedBindingHash": first}, "expected-binding")
    assert result["event"] == "error" and "result" not in result


@pytest.mark.parametrize("field,value", [("max_steps", 4), ("input_profile", "FEATURE_UNITS"),
    ("max_tip_step_mm", .5), ("partial_contact_weight", .5), ("neighboring_columns_acknowledged", False)])
def test_returned_fixed_preset_must_match_server_even_without_previous_binding(runtime, monkeypatch, field, value):
    args = install(runtime, analytic_case())
    original = facade.inspect_axis_planning
    def changed(*positional, **keywords):
        report = original(*positional, **keywords).to_dict()
        report["binding"][field] = value
        return reseal(report)
    monkeypatch.setattr(facade, "inspect_axis_planning", changed)
    result = invoke(runtime, args)
    assert result["event"] == "error" and "result" not in result


def test_acknowledgments_cannot_override_full_head_support_prohibition(runtime):
    case = analytic_case(brain_mask=None, source_refs=(SourceRef("observed", "fixture:analytic-observed"),),
        metadata={"structural_coverage": "full_head", "allow_nonzero_mri_access_support": False})
    args = install(runtime, case)
    result = invoke(runtime, args | {"acknowledgeEstimatedSupport": True})
    assert result["event"] == "error" and result["error"]["code"] == "AXIS_SUPPORT_UNAVAILABLE"
    assert case.brain_mask is None


def test_pending_extraction_cannot_be_promoted_using_acknowledgments(runtime):
    case = analytic_case(brain_mask=None, source_refs=(SourceRef("observed", "fixture:analytic-observed"),),
        metadata={"structural_coverage": "full_head", "allow_nonzero_mri_access_support": False})
    proposal = StructuralEvidence("pending", analytic_case().brain_mask, array_digest(case.mri),
        structural_frame_hash(case), None, "a" * 64, "b" * 64, "analytic test proposal")
    case = replace(case, structural_evidence={proposal.evidence_id: proposal})
    args = install(runtime, case)
    # Deliberately attempt an illicit working-mask promotion after route generation.
    promoted = replace(case, brain_mask=proposal.mask)
    entry = runtime[0].session.cases.pop(case.semantic_hash)
    entry.case = promoted
    routes = thaw_json(entry.routes)
    for route in routes["candidates"]:
        route["case_hash"] = promoted.semantic_hash
    entry.routes = freeze_json(routes)
    runtime[0].session.cases[promoted.semantic_hash] = entry
    result = invoke(runtime, args | {"caseHash": promoted.semantic_hash, "planningHash": promoted.planning_hash,
        "acknowledgeEstimatedSupport": True})
    assert result["event"] == "error" and result["error"]["code"] == "AXIS_SUPPORT_UNAVAILABLE"
    assert proposal.review_status == "review_required" and proposal.cortical_access_permitted is False


def test_case_byte_admission_limit_precedes_facade(runtime, monkeypatch):
    args = install(runtime, analytic_case())
    monkeypatch.setattr(bridge, "MAX_CASE_BYTES", 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical inspection started despite case-byte limit")
    monkeypatch.setattr(facade, "inspect_axis_planning", forbidden)
    assert invoke(runtime, args)["error"]["code"] == "AXIS_INPUT_SIZE_LIMIT"


def test_changed_cached_window_after_complete_inventory_is_withheld(runtime, monkeypatch):
    case = analytic_case()
    args = install(runtime, case)
    original = facade.inspect_axis_planning
    def changed(*positional, **keywords):
        result = original(*positional, **keywords)
        entry = runtime[0].session.cases[case.semantic_hash]
        rows = thaw_json(entry.routes)
        rows["candidates"][0]["window"]["radius_mm"] += .5
        entry.routes = freeze_json(rows)
        return result
    monkeypatch.setattr(facade, "inspect_axis_planning", changed)
    result = invoke(runtime, args)
    assert result["event"] == "error" and result["error"]["code"] == "ROUTE_VERSION_MISMATCH"


def test_timeout_after_complete_inventory_withholds_late_result(runtime, monkeypatch):
    instance, events = runtime
    args = install(runtime, analytic_case())
    original = facade.inspect_axis_planning
    completed, release, timed_out = threading.Event(), threading.Event(), threading.Event()
    def held(*positional, **keywords):
        result = original(*positional, **keywords)
        completed.set()
        assert release.wait(3)
        return result
    original_emit = instance.emit
    def emitted(row):
        original_emit(row)
        if row.get("id") == "timeout" and row.get("error", {}).get("code") == "TIMEOUT":
            timed_out.set()
    instance.emit = emitted
    monkeypatch.setattr(facade, "inspect_axis_planning", held)
    files_before = sorted(instance.session.transfers.root.iterdir())
    instance.submit({"id": "timeout", "op": "inspectAxisPlanning", "args": args, "timeoutMs": 500})
    try:
        assert completed.wait(3)
        assert timed_out.wait(3)
    finally:
        release.set()
    assert instance.wait_idle(3)
    terminal = [row for row in events if row.get("id") == "timeout" and row["event"] in {"result", "error", "cancelled"}]
    assert len(terminal) == 1 and terminal[0]["error"]["code"] == "TIMEOUT"
    assert sorted(instance.session.transfers.root.iterdir()) == files_before
    assert not instance.session.run_reports and not instance.session.replay_transfers


def test_cancel_at_final_progress_checkpoint_publishes_no_inventory(runtime):
    instance, events = runtime
    args = install(runtime, analytic_case())
    original_emit = instance.emit
    def emitted(row):
        original_emit(row)
        if row.get("id") == "last-boundary" and row.get("progress", {}).get("fraction") == 1.:
            instance.submit({"id": "cancel", "op": "cancel", "args": {"requestId": "last-boundary"}})
    instance.emit = emitted
    result = invoke(runtime, args, "last-boundary")
    assert result["event"] == "cancelled"
    assert all("inspection" not in row.get("progress", {}) for row in events)
    assert not instance.session.run_reports and not instance.session.replay_transfers
