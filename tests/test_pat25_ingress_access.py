"""Offline analytical and mocked release controls; no patient data or execution."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import diagnose_pat25_ingress_access as d
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask
from test_pat25_access_diagnostic import fixture_case


def test_declaration_is_metadata_only_and_explicitly_unreleased(monkeypatch):
    monkeypatch.setattr(d.prep,"_decode_case",lambda *a:pytest.fail("No patient decoding in declaration"))
    record=d.declaration()
    assert record["subject"]=="sub-PAT25" and not record["authorized_execution"]
    assert record["settings"]["max_total_previews"]==156 and record["settings"]["max_static_screens"]==468
    assert "scripts/diagnose_pat25_ingress_access.py" in record["source_sha256"]
    assert record["legacy_input"]["common_task"]==d.prep.common_task_definition()


@pytest.mark.parametrize("subject",["sub-PAT05","sub-PAT22","sub-PAT26","sub-PAT29",None])
def test_other_roles_refused_before_any_metadata_or_decode(monkeypatch,subject):
    monkeypatch.setattr(d,"declaration",lambda:pytest.fail("Role gate must precede metadata"))
    with pytest.raises(ValueError,match="PAT25"):
        d.validate({"version":d.VERSION,"subject":subject,"role":"TRAIN","settings":d.SETTINGS})


def test_six_static_sources_only_replace_access_never_create_task(monkeypatch):
    source,exits,_,_=fixture_case()
    monkeypatch.setattr(NativeSpatialTask,"__init__",lambda *a,**k:pytest.fail("Static candidates cannot construct full inventories"))
    candidates,sources=d.prepare_candidates(source,exits,lambda:None)
    assert len(candidates)==6
    for candidate,exit_record in zip(candidates,exits,strict=True):
        assert candidate.axis==exit_record["axis"] and candidate.outward_sign==exit_record["outward_sign"]
        assert candidate.distance_mm==exit_record["distance_mm"]
        created=sources[candidate.key]
        np.testing.assert_array_equal(created.observed_support,source.observed_support)
        np.testing.assert_array_equal(created.nominal_target,source.nominal_target)
        np.testing.assert_array_equal(created._native_affine_ras_mm,source._native_affine_ras_mm)
        assert created.tools==source.tools and not candidate.engine.config.target_labels.any()
        assert not candidate.engine.history and not candidate.engine.removed_mask.any()


def test_existing_derivation_authenticates_distance_and_axis_sign():
    source,_,derivation,original=fixture_case()
    derivation["six_axis_exit_distances_mm"][0]["distance_mm"]+=1.
    with pytest.raises(ValueError,match="axis walk"):
        d.legacy.six_accesses(derivation,source.affine_ras_mm,original)


def test_guard_refuses_preview_outside_phase_before_native_execution():
    source,_,_,_=fixture_case(); engine=NativeResectionEngine(source._native_config)
    with pytest.raises(RuntimeError,match="outside"):
        with d.InitialOnlyTrace():
            engine.preview_stroke(source.tools[0].tool_id,[5,5,5])
    assert not engine.history and sys.getprofile() is None


def test_guard_refuses_phase_retry_and_overall_budget_before_preview():
    source,_,_,_=fixture_case(); engine=NativeResectionEngine(source._native_config)
    trace=d.InitialOnlyTrace(); trace.begin("original"); trace.end()
    with pytest.raises(ValueError):trace.begin("original")
    with pytest.raises(RuntimeError,match="budget"):
        with trace:
            trace.begin("screened_selected"); trace.total=156
            engine.preview_stroke(source.tools[0].tool_id,[5,5,5])
    assert sys.getprofile() is None and not engine.history


def test_guard_forbids_committed_transition():
    source,_,_,_=fixture_case(); task=NativeSpatialTask(source)
    with pytest.raises(RuntimeError,match="prohibited"):
        with d.InitialOnlyTrace():task.step("STOP")
    assert not task._terminated and sys.getprofile() is None


def test_initial_inventory_reporting_is_pure_and_retains_shaft_cavity_visibility():
    source,exits,_,_=fixture_case()
    trace=d.InitialOnlyTrace()
    with trace:
        task,record=d.inspect_initial(lambda:NativeSpatialTask(source),exits[0],"original",trace,lambda:None)
    assert record["initial_inventory"]["declared_slots"]==78
    assert trace.total==record["preview_calls"]<=78
    assert record["actor_visibility"]["cavity_source_cells"]==record["actor_visibility"]["cavity_visible_cells"]==0
    assert set(record["actor_visibility"]["actions"][0]["segments"])=={"entry_to_tip","approach_shaft_centerline","deepest_shaft_centerline"}
    assert not task._engine.removed_mask.any() and not task._engine.history


def test_partial_preview_failure_survives_and_trace_restores():
    source,exits,_,_=fixture_case();calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==5:raise TimeoutError("bounded analytic interruption")
    trace=d.InitialOnlyTrace(stop)
    with pytest.raises(TimeoutError) as caught:
        with trace:d.inspect_initial(lambda:NativeSpatialTask(source),exits[0],"original",trace,stop)
    assert caught.value.diagnostic["status"]=="unassessed"
    assert caught.value.diagnostic["started_preview_calls"]>0 and sys.getprofile() is None


def setup_worker(monkeypatch,tmp_path):
    source,exits,_,_=fixture_case(); baseline=NativeSpatialTask(source)
    saved={"binding":{"member":{},"decision_model_hash":baseline.decision_model_hash},
        "initial_inventory":baseline.candidate_inventory()}
    common={"objective":asdict(baseline.reward_spec),"max_steps":baseline.max_steps}
    record={"legacy_input":{"common_task":common}}
    manifest=tmp_path/"manifest.json";release=tmp_path/"release.json"
    manifest.write_text("{}");release.write_text("{}")
    bound={str(manifest):d.sha256(manifest),str(release):d.sha256(release)}
    monkeypatch.setattr(d,"preflight",lambda *args,**kwargs:(record,saved,bound,{"source_commit":"a"*40}))
    monkeypatch.setattr(d.legacy,"load_bound_case",lambda *args:(None,saved,exits))
    monkeypatch.setattr(d.prep,"_construct_task",lambda *args:NativeSpatialTask(source))
    output=tmp_path/"attempt";output.mkdir()
    return manifest,release,output,source,exits


def test_mocked_worker_two_inventories_no_extra_preview_search_or_cut(monkeypatch,tmp_path):
    manifest,release,output,_,_=setup_worker(monkeypatch,tmp_path)
    record=d.worker(manifest,release,output, expected_manifest_sha256=d.sha256(manifest), expected_release_sha256=d.sha256(release))
    assert record["status"]=="complete",record.get("error")
    assert record["original"]["historical_inventory_exactly_reproduced"]
    assert record["screened_selected"]["status"]=="complete"
    assert record["total_preview_calls"]==record["original"]["preview_calls"]+record["screened_selected"]["preview_calls"]<=156
    assert record["screening"]["screen_count"]<=468
    assert not record["executed_transitions"] and not record["optimizer_updates"]


def test_selected_no_ingress_preserves_baseline_and_does_not_invent_zero_outcome(monkeypatch,tmp_path):
    manifest,release,output,_,_=setup_worker(monkeypatch,tmp_path)
    import resectionlab.native_ingress as ingress
    result=ingress.IngressScreeningResult("no_ingress_admissible_access",None,(),0)
    monkeypatch.setattr(ingress,"screen_axis_accesses",lambda *a,**k:result)
    record=d.worker(manifest,release,output, expected_manifest_sha256=d.sha256(manifest), expected_release_sha256=d.sha256(release))
    assert record["status"]=="complete"
    assert record["screened_selected"]=={"status":"not_executed_no_ingress_admissible_access"}
    assert record["total_preview_calls"]==record["original"]["preview_calls"]


def test_source_change_at_end_makes_complete_geometry_incomplete(monkeypatch,tmp_path):
    manifest,release,output,_,_=setup_worker(monkeypatch,tmp_path)
    monkeypatch.setattr(d,"unchanged",lambda bound:False)
    record=d.worker(manifest,release,output, expected_manifest_sha256=d.sha256(manifest), expected_release_sha256=d.sha256(release))
    assert record["status"]=="incomplete" and not record["inputs_unchanged"]
    assert record["screened_selected"]["status"]=="complete"


def test_existing_attempt_worker_receipt_cannot_be_overwritten(tmp_path):
    output=tmp_path/"attempt";output.mkdir();path=output/"receipt.json";path.write_text("original")
    with pytest.raises(ValueError,match="restarted"):d.worker(None,None,output)
    assert path.read_text()=="original"


def test_unreleased_preflight_refuses_before_metadata_or_bundle(monkeypatch,tmp_path):
    manifest=tmp_path/"manifest.json";release=tmp_path/"release.json"
    manifest.write_text(json.dumps({"subject":d.SUBJECT,"role":"TRAIN"}));release.write_text("{}")
    monkeypatch.setattr(d,"validate",lambda *a:pytest.fail("No validation after absent release"))
    with pytest.raises(ValueError,match="authorized"):d.preflight(manifest,release,tmp_path/"attempt")


def test_parent_failed_supervision_cannot_promote_completed_worker(monkeypatch,tmp_path):
    output=tmp_path/"attempt"
    monkeypatch.setattr(d,"preflight",lambda *a,**kwargs:({}, {}, {str(tmp_path/"manifest"):"b"*64, str(tmp_path/"release"):"b"*64}, {"source_commit":"a"*40}))
    monkeypatch.setattr(d,"sha256",lambda p:"b"*64)
    def supervisor(command,destination,settings,digest):
        assert 0<settings["max_wall_seconds"]<=174 and settings["max_rss_bytes"]==6*1024**3 and "--worker" in command
        d.write_json(destination/"receipt.json",{"version":d.VERSION,"subject":d.SUBJECT,"status":"complete","inputs_unchanged":True})
        return {"status":"failed","timed_out":True}
    monkeypatch.setattr(d,"supervise_worker",supervisor)
    record=d.run(tmp_path/"manifest",tmp_path/"release",output)
    assert record["status"]=="failed" and (output/"acceptance.json").is_file()


def test_final_input_recheck_cannot_leave_acceptance_complete(monkeypatch,tmp_path):
    manifest,release=tmp_path/"manifest",tmp_path/"release"
    bound={str(manifest):"b"*64,str(release):"b"*64}
    monkeypatch.setattr(d,"preflight",lambda *a,**k:({}, {}, bound, {"source_commit":"a"*40}))
    monkeypatch.setattr(d,"sha256",lambda p:"b"*64)
    checks=iter([True,False])
    monkeypatch.setattr(d,"unchanged",lambda bound:next(checks))
    def supervisor(command,destination,settings,digest):
        d.write_json(destination/"receipt.json",{"version":d.VERSION,"subject":d.SUBJECT,"status":"complete","inputs_unchanged":True})
        return {"status":"complete"}
    monkeypatch.setattr(d,"supervise_worker",supervisor)
    result=d.run(manifest,release,tmp_path/"attempt")
    assert result["status"]=="failed" and result["inputs_unchanged"] is False


def test_removed_observer_cannot_claim_phase_completion():
    trace=d.InitialOnlyTrace()
    with pytest.raises(RuntimeError,match="observer"):
        with trace:
            trace.begin("original")
            sys.setprofile(None)
            trace.assert_healthy()
    assert trace.violation is not None and sys.getprofile() is None
