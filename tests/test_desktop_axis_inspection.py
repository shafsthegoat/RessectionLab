"""Read-only expanded-inventory protocol checks on tiny native source grids."""
from dataclasses import replace
import json
import threading

import numpy as np
import pytest

from resectionlab import desktop_bridge as bridge
from resectionlab.core import CaseData, SourceRef, array_digest, freeze_json, thaw_json
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_refinement import AxisPlanningInspection
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionEngine
from resectionlab.planning import SearchConfig, generate_candidate_routes
from resectionlab.worlds import content_hash


def tiny_case(**changes):
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros_like(tissue)
    target[1:6, 1:6, 2:7] = True
    case = CaseData("axis-bridge-analytic", tissue.astype(np.float32), {"target": target}, np.eye(4),
        (SourceRef("analytic", "simulated:axis-bridge", provenance="simulated"),), brain_mask=tissue)
    return replace(case, **changes) if changes else case


@pytest.fixture
def runtime(tmp_path):
    events = []
    instance = bridge.BridgeRuntime(tmp_path / "transfers", events.append)
    yield instance, events
    instance.close()


def send(runtime, identity, operation, args):
    instance, events = runtime
    instance.submit({"id": identity, "op": operation, "args": args})
    assert instance.wait_idle(10)
    rows = [row for row in events if row.get("id") == identity and row["event"] in {"result", "error", "cancelled"}]
    assert len(rows) == 1
    return rows[0]


def install(runtime, case=None, identity="routes"):
    instance, _ = runtime
    case = tiny_case() if case is None else case
    instance.session._install_case(case, {}, bridge._Request("fixture-install"))
    generated = send(runtime, identity, "generateNativeRoutes", {"caseHash": case.semantic_hash})
    assert generated["event"] == "result", generated
    route = next(row for row in generated["result"]["candidates"] if row["geometry"]["feasible"])
    return case, route


def arguments(case, route, **changes):
    return {"caseHash": case.semantic_hash, "planningHash": case.planning_hash,
            "routeId": route["route_id"], "routePlanningModelHash": route["planning_model_hash"],
            "toolIds": [tool.tool_id for tool in NATIVE_GENERIC_TOOLS],
            "acknowledgeNeighboringColumns": True, "acknowledgeEstimatedSupport": False} | changes


def test_complete_read_only_wire_inventory_does_not_install_routes_or_files(runtime, monkeypatch):
    instance, events = runtime
    case, route = install(runtime)
    entry = instance.session.cases[case.semantic_hash]
    before_routes = thaw_json(entry.routes)
    before_arrays = [array_digest(array) for array in (case.mri, case.brain_mask, case.compartments["target"])]
    before_files = sorted(instance.session.transfers.root.iterdir())
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection cannot commit a native preview")
    monkeypatch.setattr(NativeResectionEngine, "commit_preview", forbidden)
    result = send(runtime, "inspect", "inspectAxisPlanning", arguments(case, route,
        toolIds=list(reversed(arguments(case, route)["toolIds"]))))
    assert result["event"] == "result", result
    value, report = result["result"], result["result"]["inspection"]
    assert value["schemaVersion"] == 1 and value["accessSource"] == "selected_route_window_only"
    assert value["anchorWindowRas"] == route["window"]
    assert value["requestedToolIds"] == [tool.tool_id for tool in NATIVE_GENERIC_TOOLS]
    assert report["inventory_complete"] and report["inventory"]["batch"]["slot_count"] == 26
    assert report["legal_non_stop_actions"] > 1
    assert report["actions"][0]["action_id"] == "STOP"
    assert report["accounting"] == dict(gradient_steps=0, executed_transitions=0,
                                        native_commits=0, simulated_removed_volume_mm3=0.)
    assert report["candidate_eligible"] is report["removal_authorized"] is False
    assert report["clinical_deficit_probability"] is None
    assert report["binding"]["input_profile"] == "RAW" and report["binding"]["max_steps"] == 3
    assert len(json.dumps(value).encode()) < bridge.MAX_AXIS_INSPECTION_RESULT_BYTES
    assert thaw_json(entry.routes) == before_routes
    assert before_arrays == [array_digest(array) for array in (case.mri, case.brain_mask, case.compartments["target"])]
    assert sorted(instance.session.transfers.root.iterdir()) == before_files
    assert not instance.session.run_reports and not instance.session.replay_transfers
    assert send(runtime, "runs", "listRuns", {"caseHash": case.semantic_hash})["result"]["runs"] == []
    action_id = report["actions"][1]["action_id"]
    refused = send(runtime, "cannot-refine-preview", "inspectRefinement", {"caseHash": case.semantic_hash, "routeId": action_id})
    assert refused["error"]["code"] == "ROUTE_UNAVAILABLE"
    assert [row["event"] for row in events if row.get("id") == "inspect"] == ["started", "progress", "progress", "result"]


@pytest.mark.parametrize("extra", [{"access": {}}, {"reward": {}}, {"worldGenerator": {}},
    {"inputProfile": "FEATURE_UNITS"}, {"execute": True}, {"resumeRunId": "forged"}, {"inspection": {}}])
def test_renderer_cannot_supply_geometry_model_execution_or_evaluator_results(runtime, extra):
    case, route = install(runtime)
    result = send(runtime, "unknown-field", "inspectAxisPlanning", arguments(case, route, **extra))
    assert result["error"]["code"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize("changes,code", [
    ({"planningHash": "sha256:" + "0" * 64}, "CASE_VERSION_MISMATCH"),
    ({"routePlanningModelHash": "0" * 64}, "ROUTE_VERSION_MISMATCH"),
    ({"caseHash": "sha256:" + "0" * 64}, "CASE_VERSION_UNAVAILABLE"),
    ({"toolIds": []}, "INVALID_ARGUMENT"),
    ({"toolIds": ["generic_suction"]}, "INVALID_ARGUMENT"),
    ({"toolIds": ["native-fine-aspiration"] * 2}, "INVALID_ARGUMENT"),
    ({"toolIds": [{}]}, "INVALID_ARGUMENT"),
    ({"acknowledgeNeighboringColumns": False}, "AXIS_ACKNOWLEDGEMENT_REQUIRED"),
    ({"acknowledgeNeighboringColumns": 1}, "AXIS_ACKNOWLEDGEMENT_REQUIRED"),
    ({"acknowledgeEstimatedSupport": "false"}, "INVALID_ARGUMENT"),
    ({"expectedBindingHash": None}, "INVALID_ARGUMENT"),
    ({"expectedBindingHash": "stale"}, "INVALID_ARGUMENT"),
])
def test_stale_and_malformed_requests_refuse_before_facade(runtime, monkeypatch, changes, code):
    case, route = install(runtime)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid request reached numerical inspection")
    monkeypatch.setattr("resectionlab.native_axis_refinement.inspect_axis_planning", forbidden)
    result = send(runtime, "invalid", "inspectAxisPlanning", arguments(case, route, **changes))
    assert result["error"]["code"] == code, result


def test_estimated_support_needs_separate_acknowledgment(runtime):
    case, route = install(runtime, tiny_case(brain_mask=None, metadata={"skull_stripped": True}))
    refused = send(runtime, "no-support-ack", "inspectAxisPlanning", arguments(case, route))
    assert refused["error"]["code"] == "AXIS_SUPPORT_ACKNOWLEDGEMENT_REQUIRED"
    result = send(runtime, "support-ack", "inspectAxisPlanning", arguments(case, route, acknowledgeEstimatedSupport=True))
    assert result["event"] == "result", result
    assert result["result"]["inspection"]["binding"]["source_support"]["evidence_type"] == "estimated"
    assert case.brain_mask is None


def test_fractional_window_is_not_snapped_into_supported_axis_access(runtime):
    instance, _ = runtime
    case, _ = install(runtime)
    source = generate_candidate_routes(case,
        windows=(AccessWindow((3.2, 3, -.5), (0, 0, 1), 6),), tools=NATIVE_GENERIC_TOOLS,
        explicit_targets=(("target", (3., 3., 6.)),), config=SearchConfig(max_windows=1, targets_per_compartment=1))
    instance.session.cases[case.semantic_hash].routes = freeze_json(source.to_dict())
    route = next(row for row in source.to_dict()["candidates"] if row["geometry"]["feasible"])
    result = send(runtime, "fractional", "inspectAxisPlanning", arguments(case, route))
    assert result["error"]["code"] == "AXIS_ACCESS_UNSUPPORTED", result
    assert "FRACTIONAL" in result["error"]["message"]


def test_lps_cached_route_window_is_converted_once_before_inspection(runtime):
    case = tiny_case(frame="LPS+", affine=np.diag([-1., -1., 1., 1.]))
    case, route = install(runtime, case)
    result = send(runtime, "lps", "inspectAxisPlanning", arguments(case, route))
    assert result["event"] == "result", result
    value = result["result"]
    np.testing.assert_array_equal(value["anchorWindowRas"]["center_mm"], np.asarray(route["window"]["center_mm"]) * [-1, -1, 1])
    assert value["inspection"]["binding"]["source_frame"] == "LPS+"
    assert value["inspection"]["binding"]["native_affine_ras_mm"] == np.eye(4).tolist()


@pytest.mark.parametrize("limit_name,code", [("MAX_AXIS_INSPECTION_VOXELS", "AXIS_INPUT_SIZE_LIMIT"),
    ("MAX_AXIS_INSPECTION_METADATA_BYTES", "AXIS_METADATA_SIZE_LIMIT")])
def test_admission_limits_run_before_native_constructor(runtime, monkeypatch, limit_name, code):
    case, route = install(runtime)
    monkeypatch.setattr(bridge, limit_name, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("Over-limit source reached inspection")
    monkeypatch.setattr("resectionlab.native_axis_refinement.inspect_axis_planning", forbidden)
    result = send(runtime, "input-cap", "inspectAxisPlanning", arguments(case, route))
    assert result["error"]["code"] == code


def test_complete_response_limit_does_not_publish_or_retain_partial_payload(runtime, monkeypatch):
    instance, _ = runtime
    case, route = install(runtime)
    routes = thaw_json(instance.session.cases[case.semantic_hash].routes)
    files = sorted(instance.session.transfers.root.iterdir())
    monkeypatch.setattr(bridge, "MAX_AXIS_INSPECTION_RESULT_BYTES", 1)
    result = send(runtime, "output-cap", "inspectAxisPlanning", arguments(case, route))
    assert result["error"]["code"] == "AXIS_RESULT_SIZE_LIMIT"
    assert thaw_json(instance.session.cases[case.semantic_hash].routes) == routes
    assert sorted(instance.session.transfers.root.iterdir()) == files


@pytest.mark.parametrize("field,value", [("inventory_complete", False), ("candidate_eligible", True),
                                         ("clinical_deficit_probability", .7)])
def test_incomplete_or_promoted_facade_result_is_withheld(runtime, monkeypatch, field, value):
    from resectionlab import native_axis_refinement as facade
    case, route = install(runtime)
    original = facade.inspect_axis_planning
    def changed(*args, **kwargs):
        result = original(*args, **kwargs).to_dict()
        result[field] = value
        result["inspection_hash"] = content_hash({k: v for k, v in result.items() if k != "inspection_hash"})
        return AxisPlanningInspection(json.dumps(result))
    monkeypatch.setattr(facade, "inspect_axis_planning", changed)
    result = send(runtime, "bad-result", "inspectAxisPlanning", arguments(case, route))
    assert result["error"]["code"] == "AXIS_INCOMPLETE_INSPECTION"


def test_stale_expected_binding_never_returns_an_alternate_model(runtime):
    case, route = install(runtime)
    first = send(runtime, "first-model", "inspectAxisPlanning", arguments(case, route))["result"]
    changed = send(runtime, "changed-model", "inspectAxisPlanning", arguments(case, route,
        toolIds=[NATIVE_GENERIC_TOOLS[0].tool_id], expectedBindingHash=first["inspection"]["binding"]["binding_hash"]))
    assert changed["error"]["code"] == "AXIS_BINDING_CHANGED"


def test_returned_expected_binding_is_checked_even_with_unchanged_geometry(runtime, monkeypatch):
    from resectionlab import native_axis_refinement as facade
    case, route = install(runtime)
    first = send(runtime, "original-binding", "inspectAxisPlanning", arguments(case, route))["result"]
    expected = first["inspection"]["binding"]["binding_hash"]
    original = facade.inspect_axis_planning
    def changed(*args, **kwargs):
        report = original(*args, **kwargs).to_dict()
        # Self-consistent hashes alone do not satisfy an exact prior-binding request.
        report["binding"]["unexpected_annotation"] = "changed after facade validation"
        report["binding"]["binding_hash"] = content_hash({
            key: value for key, value in report["binding"].items() if key != "binding_hash"})
        report["inspection_hash"] = content_hash({
            key: value for key, value in report.items() if key != "inspection_hash"})
        return AxisPlanningInspection(json.dumps(report))
    monkeypatch.setattr(facade, "inspect_axis_planning", changed)
    result = send(runtime, "returned-binding", "inspectAxisPlanning",
                  arguments(case, route, expectedBindingHash=expected))
    assert result["error"]["code"] == "AXIS_BINDING_CHANGED"


def test_cancel_after_preview_emits_one_terminal_and_no_late_inventory(runtime, monkeypatch):
    instance, events = runtime
    case, route = install(runtime)
    entered, release = threading.Event(), threading.Event()
    original = NativeResectionEngine.preview_stroke
    def slow(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        assert engine.revision == 0 and not engine.removed_mask.any()
        entered.set()
        assert release.wait(3)
        return result
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", slow)
    instance.submit({"id": "cancelled-inspection", "op": "inspectAxisPlanning", "args": arguments(case, route)})
    assert entered.wait(3)
    try:
        instance.submit({"id": "cancel", "op": "cancel", "args": {"requestId": "cancelled-inspection"}})
        instance.submit({"id": "ping", "op": "ping", "args": {}})
        assert any(row.get("id") == "ping" and row["event"] == "result" for row in events)
    finally:
        release.set()
    assert instance.wait_idle(3)
    terminal = [row for row in events if row.get("id") == "cancelled-inspection" and row["event"] in {"result", "error", "cancelled"}]
    assert [row["event"] for row in terminal] == ["cancelled"]
    assert not instance.session.run_reports and not instance.session.replay_transfers
