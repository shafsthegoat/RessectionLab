"""Tiny generated geometry only; no patient, model or policy observation paths."""
from dataclasses import asdict, replace, FrozenInstanceError
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab import native_resection as native
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry


def single_cell(*, step=.5, immutable=False, affine=None):
    tissue = np.zeros((7, 7, 10), bool); tissue[3, 3, 5] = True
    affine = np.eye(4) if affine is None else np.asarray(affine, float)
    tool = ToolGeometry("short-active", .9, .1, 10., tip_length_mm=.1)
    entry = affine[:3, :3] @ [3, 3, 4.49] + affine[:3, 3]
    axis = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    config = native.NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), affine,
        AccessWindow(entry, axis, 2.), (tool,), "generated-obstruction-source",
        "generated single cell; no anatomical claim", max_tip_step_mm=step)
    tip = affine[:3, :3] @ [3, 3, 5.49] + affine[:3, 3]
    return native.NativeResectionEngine(config, immutable_state=immutable), tool.tool_id, entry, tip


def opening(*, immutable=False):
    tissue = np.zeros((9, 9, 7), bool)
    tissue[4, 4, 1:6] = True; tissue[5, 5, 1] = True
    tools = (ToolGeometry("opener", 2.25, .45, 2.2, 35., .75),
        ToolGeometry("cutter", .9, 1.1, 12., 35., 3.))
    config = native.NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), np.eye(4),
        AccessWindow([3.9, 4., .5], [0, 0, 1], 2.4), tools,
        "generated-preparation-source", "generated off-axis shaft blocker")
    return native.NativeResectionEngine(config, immutable_state=immutable)


def normalized_result(result):
    # Sidecar is intentionally absent from existing dataclass serialization.
    return json.loads(json.dumps(asdict(result), default=lambda x: x.tolist()))


def checked_record(result):
    record = thaw_json(result.obstruction_diagnostic)
    assert record is not None
    fingerprint = record.pop("fingerprint")
    assert fingerprint == semantic_digest(record)
    assert record["source_hash"] == result.source_hash
    assert record["source_state_hash"] == result.source_state_hash
    assert record["decision_model_hash"] == result.decision_model_hash
    assert record["requested_tip_mm"] == list(result.tip_mm)
    assert record["entry_mm"] == list(result.entry_mm)
    assert record["axis_unit"] == list(result.axis_unit)
    assert record["failure_tip_mm"] == list(result.failure_tip_mm)
    assert record["native_affine_hash"] == array_digest(result.native_affine)
    assert record["failure_interval_index"] == len(result.microsteps)
    assert record["prior_temporary_removed_count"] == sum(len(s.removed_indices_native) for s in result.microsteps)
    assert record["prior_temporary_removals_hash"] == semantic_digest([
        native._hash_array(s.removed_indices_native) for s in result.microsteps])
    return record


@pytest.mark.parametrize("step", [.5, .25, .1])
def test_exact_blockers_use_only_prior_completed_microstep_cavity(step):
    engine, tool_id, entry, tip = single_cell(step=step)
    before, state = engine.remaining_mask.copy(), engine.state_hash
    result = engine.preview_stroke(tool_id, tip, entry_mm=entry, obstruction_diagnostics=True)
    assert not result.feasible and result.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
    record = checked_record(result)
    remaining = before.copy()
    for microstep in result.microsteps:
        remaining[tuple(microstep.removed_indices_native.T)] = False
    shaft = native.capsule_voxel_indices(engine._cell_scene,
        np.asarray(record["shaft_sweep_start_mm"]), np.asarray(record["shaft_sweep_end_mm"]), .1)
    expected = shaft[remaining[tuple(shaft.T)]]
    assert record["blocked_indices_native"] == expected.tolist() == [[3, 3, 5]]
    assert record["blocked_indices_hash"] == array_digest(expected)
    assert record["complete_first_failure_set"] and not record["truncated"]
    assert record["prior_temporary_removals_committed"] is False
    np.testing.assert_array_equal(engine.remaining_mask, before)
    assert engine.state_hash == state and engine.history == [] and not engine.removed_mask.any()
    assert len(result.removed_indices_native) == 0
    if step == .5:
        # This cell would qualify at the CURRENT tip, but may not be borrowed
        # to excuse the shaft sweep before that endpoint's cut is credited.
        active_start = np.asarray(record["previous_tip_mm"]) - .1*np.asarray(result.axis_unit)
        active_end = np.asarray(record["failure_tip_mm"])
        inside = native.contained_capsule_cells(engine._cell_scene, expected, active_start, active_end, .9)
        assert inside.tolist() == [[3, 3, 5]]


@pytest.mark.parametrize("immutable", [False, True])
def test_diagnostic_does_not_change_result_geometry_or_commit_identity(immutable, monkeypatch):
    a, b = opening(immutable=immutable), opening(immutable=immutable)
    queried = []
    original = native.capsule_voxel_indices
    def counted(scene, start, end, radius):
        queried.append((tuple(start), tuple(end), radius))
        return original(scene, start, end, radius)
    monkeypatch.setattr(native, "capsule_voxel_indices", counted)
    first = a.preview_stroke("cutter", [4, 4, 5], entry_mm=[4, 4, .5])
    old_queries = list(queried); queried.clear()
    second = b.preview_stroke("cutter", [4, 4, 5], entry_mm=[4, 4, .5], obstruction_diagnostics=True)
    assert queried == old_queries  # Zero extra geometric queries or previews.
    assert normalized_result(first) == normalized_result(second)
    assert first.obstruction_diagnostic is None
    record = checked_record(second)
    assert record["prior_temporary_removed_count"] > 0
    temporary = {tuple(row) for step in second.microsteps for row in step.removed_indices_native.tolist()}
    assert not temporary.intersection(map(tuple, record["blocked_indices_native"]))
    assert [5, 5, 1] in record["blocked_indices_native"]
    assert a.state_hash == b.state_hash and a.history == b.history == []
    for engine, enabled in ((a, False), (b, True)):
        result = engine.preview_stroke("opener", [4, 4, 1], entry_mm=[4, 4, .5], obstruction_diagnostics=enabled)
        assert result.feasible and result.obstruction_diagnostic is None
        engine.commit_preview(result)
    assert a.history == b.history and a.state_hash == b.state_hash
    assert a._result_digest(a.execute_stroke("cutter", [4, 4, 5], entry_mm=[4, 4, .5])) == \
        b._result_digest(b.execute_stroke("cutter", [4, 4, 5], entry_mm=[4, 4, .5]))
    assert a.history == b.history and a.state_hash == b.state_hash
    assert record["source_state_hash"] != b.state_hash


def test_diagnostic_cap_is_explicit_full_digest_stable_and_payload_immutable():
    engine, _, _, _ = single_cell()
    tissue = np.zeros(engine.config.tissue_mask.shape, bool); tissue[1:6, 1:6, 5] = True
    tool = ToolGeometry("wide-shaft", .9, 1.6, 10., tip_length_mm=.1)
    config = replace(engine.config, tissue_mask=tissue, tools=(tool,),
        access=AccessWindow([3, 3, 4.49], [0, 0, 1], 3.))
    engine = native.NativeResectionEngine(config)
    full = engine.preview_stroke(tool.tool_id, [3, 3, 5.49], obstruction_diagnostics=True)
    small = engine.preview_stroke(tool.tool_id, [3, 3, 5.49], obstruction_diagnostics=True, obstruction_cell_limit=1)
    a, b = checked_record(full), checked_record(small)
    assert a["blocked_cell_count"] > 1
    assert a["blocked_cell_count"] == b["blocked_cell_count"]
    assert a["blocked_indices_hash"] == b["blocked_indices_hash"]
    assert b["blocked_indices_native"] == a["blocked_indices_native"][:1]
    assert b["retained_cell_count"] == b["cell_limit"] == 1
    assert b["truncated"] and not b["complete_first_failure_set"]
    with pytest.raises(TypeError): small.obstruction_diagnostic["blocked_cell_count"] = 0
    with pytest.raises(TypeError): small.obstruction_diagnostic["blocked_indices_native"][0][0] = 99
    with pytest.raises(FrozenInstanceError): small.obstruction_diagnostic = {}
    exported = thaw_json(small.obstruction_diagnostic); exported["blocked_indices_native"][0][0] = 99
    assert checked_record(small) == b
    with pytest.raises(ValueError): engine.commit_preview(small)


def test_binding_follows_reflected_anisotropic_frame_and_exact_tool_ray():
    affine = np.diag([.8, 1.2, -1., 1.]); affine[:3, 3] = [10, -8, 20]
    engine, tool, entry, tip = single_cell(step=.25, affine=affine)
    result = engine.preview_stroke(tool, tip, entry_mm=entry, obstruction_diagnostics=True)
    record = checked_record(result)
    assert record["tool"] == asdict(engine._tools[tool])
    assert record["blocked_indices_native"] == [[3, 3, 5]]
    np.testing.assert_array_equal(record["axis_unit"], [0, 0, -1])
    assert record["source_shape"] == list(engine.config.tissue_mask.shape)


@pytest.mark.parametrize("kind", ["budget", "hard_geometry"])
def test_no_shaft_evidence_is_invented_for_other_failures(kind):
    engine, tool, entry, tip = single_cell()
    if kind == "budget": config = replace(engine.config, max_microsteps=1)
    else:
        hard = engine.config.hard_exclusion.copy(); hard[3, 3, 5] = True
        config = replace(engine.config, hard_exclusion=hard)
    engine = native.NativeResectionEngine(config)
    result = engine.preview_stroke(tool, tip, entry_mm=entry, obstruction_diagnostics=True)
    assert not result.feasible and "SHAFT_BLOCKED" not in result.reason
    assert result.obstruction_diagnostic is None


@pytest.mark.parametrize("kwargs", [{"obstruction_diagnostics": 1}, {"obstruction_diagnostics": "yes"},
    {"obstruction_diagnostics": True, "obstruction_cell_limit": True},
    {"obstruction_diagnostics": True, "obstruction_cell_limit": 0},
    {"obstruction_diagnostics": True, "obstruction_cell_limit": 4097}, {"obstruction_cell_limit": 1}])
def test_malformed_diagnostic_option_refused_without_geometry(kwargs, monkeypatch):
    engine, tool, entry, tip = single_cell()
    def forbidden(*a, **k): raise AssertionError("geometry must not execute")
    monkeypatch.setattr(native, "check_motion", forbidden)
    with pytest.raises((TypeError, ValueError)):
        engine.preview_stroke(tool, tip, entry_mm=entry, **kwargs)


def test_historical_default_result_and_commit_parity(monkeypatch):
    path = os.environ.get("OBSTRUCTION_BASELINE_SOURCE")
    if not path: pytest.skip("historical-file control requires its pinned prechange source")
    name = "resectionlab._obstruction_baseline"
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    current = opening()
    old_config = module.NativeResectionConfig(**{f.name: getattr(current.config, f.name)
        for f in __import__("dataclasses").fields(current.config) if f.init})
    old = module.NativeResectionEngine(old_config)
    assert old.decision_model_hash == current.decision_model_hash
    for tool, tip in (("cutter", [4, 4, 5]), ("opener", [4, 4, 1]), ("cutter", [4, 4, 5])):
        a = old.preview_stroke(tool, tip, entry_mm=[4, 4, .5])
        b = current.preview_stroke(tool, tip, entry_mm=[4, 4, .5])
        assert normalized_result(a) == normalized_result(b)
        if a.feasible:
            assert old._result_digest(a) == current._result_digest(b)
            old.commit_preview(a); current.commit_preview(b)
        assert old.history == current.history and old.state_hash == current.state_hash
