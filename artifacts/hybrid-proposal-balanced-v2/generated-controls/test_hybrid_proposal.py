"""Generated seam controls only; no patient inputs or historical weights."""
from dataclasses import replace
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_proposal import (Costs, FEATURES, GeneratedSparseTask, PublicRows,
                             RidgePriority, generated_training_rows, public_rows)
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask


def fixture(**changes):
    support = np.zeros((13, 13, 10), bool)
    support[4:9, 4:9, 2:4] = True
    target = support.copy()
    case = NativeSpatialCase(np.where(support, .8, 0.).astype(np.float32), support, target, np.eye(4),
        AccessWindow((6., 6., 1.5), (0., 0., 1.), 4., "generated-shortlist"),
        (ToolGeometry("generated-small", .9, .5, 8., 45., .8),
         ToolGeometry("generated-large", 1.3, .7, 8., 45., 1.5)),
        track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="analytic generated slab from nonzero scan",
        nominal_target=target, target_source_kind="derived_from_scan",
        target_derivation="analytic generated positive slab; no real estimate",
        proposal_mode="nominal_cavity_v1", proposal_config=NominalCavityProposalConfig(
            offsets_source_voxels=tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)),
            max_candidates=120, intermediate_opening_mm=1., tool_footprint_opening=True))
    return replace(case, **changes) if changes else case


def raw(case):
    engine = NativeResectionEngine(case._native_config, immutable_state=True)
    batch = case._nominal_proposer.propose(engine)
    return engine, batch


def test_raw_features_and_ranking_have_no_preview_or_task_query(monkeypatch):
    case = fixture(); engine, batch = raw(case)
    def forbidden(*a, **k): raise AssertionError("free native/task query")
    for name in ("observation", "candidate_inventory", "step", "_score_record"):
        monkeypatch.setattr(NativeSpatialTask, name, forbidden)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    rows = public_rows(case, engine.removed_mask, batch.proposals, steps=0, horizon=6, current_tool=None)
    ranker = RidgePriority(np.zeros(FEATURES), np.ones(FEATURES), np.ones(FEATURES), 0.)
    assert ranker.score(rows).shape == (len(batch.proposals),)
    assert len(rows.ids) > 8 and not rows.features.flags.writeable
    assert not hasattr(rows, "legal_mask") and not hasattr(rows, "reward")


@pytest.mark.parametrize("mode", ["sampled_target_top8", "ridge_top8", "exhaustive"])
def test_constructor_and_cached_observation_charge_exact_queries(mode, monkeypatch):
    case = fixture(); calls = []
    original = NativeResectionEngine.preview_stroke
    def record(self, tool, tip, **kw):
        calls.append((tool, tuple(tip), tuple(kw["entry_mm"])))
        return original(self, tool, tip, **kw)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", record)
    ranker = RidgePriority(np.zeros(FEATURES), np.ones(FEATURES), np.zeros(FEATURES), 0.) if mode == "ridge_top8" else None
    task = GeneratedSparseTask(case, mode=mode, ranker=ranker, max_steps=6)
    record = task.candidate_inventory()["shortlist"]
    expected = record["raw_count"] if mode == "exhaustive" else 8
    assert len(calls) == task.costs.counts["native_previews"] == expected
    for _ in range(3): task.observation(); task.candidate_inventory()
    assert len(calls) == expected
    assert record["not_queried_count"] == record["raw_count"]-expected
    if mode == "ridge_top8":
        assert record["selected_ids"] == list(task._raw_rows.ids[:8])
    if mode != "exhaustive":
        skipped = [r for r in task._ledger if r["reason"] == "PROPOSED_UNCERTIFIED" and r["query_disposition"] == "NOT_QUERIED"]
        assert skipped and all("feasible" not in r for r in skipped)


def test_exhaustive_adapter_matches_canonical_positive_greedy_and_full_replay():
    case = fixture()
    canonical = NativeSpatialTask(case, max_steps=6)
    adapter = GeneratedSparseTask(case, mode="exhaustive", max_steps=6)
    a, ac = canonical.observed_greedy_search(seconds=10)
    b, bc = adapter.observed_greedy_search(seconds=10)
    assert a == b and any(v != "STOP" for v in a)
    assert ac["estimated_incremental_return"] == bc["estimated_incremental_return"] > 0
    assert ac["evaluated_nonstop_actions"] == bc["evaluated_nonstop_actions"]
    first, second = canonical.fresh(), adapter.fresh()
    for action in a:
        assert first.step(action).info == second.step(action).info
    assert first.terminated and second.terminated
    assert first._engine.state_hash == second._engine.state_hash
    assert first.independent_geometry_check().feasible and second.independent_geometry_check().feasible


def test_reference_changes_neither_raw_input_selection_queries_nor_nominal_search():
    first = GeneratedSparseTask(fixture(), mode="sampled_target_top8", max_steps=6)
    second = GeneratedSparseTask(replace(first.case, reference_target=np.zeros(first.case.reference_target.shape)),
                                 mode="sampled_target_top8", max_steps=6)
    np.testing.assert_array_equal(first._raw_rows.features, second._raw_rows.features)
    assert first._raw_rows.public_binding == second._raw_rows.public_binding
    assert first.candidate_inventory() == second.candidate_inventory()
    a, ac = first.observed_greedy_search(seconds=10)
    b, bc = second.observed_greedy_search(seconds=10)
    assert a == b and ac["decisions"] == bc["decisions"]
    assert ac["estimated_incremental_return"] == bc["estimated_incremental_return"]
    assert first.costs.counts == second.costs.counts


def test_generated_ridge_fit_and_three_sealed_complete_routes():
    case = fixture(); training = GeneratedSparseTask(case, mode="exhaustive", max_steps=6)
    teacher, _ = training.observed_greedy_search(seconds=10)
    collector = training.planning_clone()
    xs, ys, states = [], [], []
    for step, action in enumerate(teacher):
        rows, labels = generated_training_rows(collector)
        if len(rows.ids):
            xs.append(rows.features); ys.append(labels); states.extend([step]*len(labels))
        collector.advance_planning(action)
    ranker = RidgePriority.fit_generated(np.concatenate(xs), np.concatenate(ys), states)
    frozen = ranker.fingerprint
    # All strategies seal complete actions before any authoritative reward read.
    tasks, plans = {}, {}
    for mode in ("ridge_top8", "sampled_target_top8", "exhaustive"):
        task = GeneratedSparseTask(case, mode=mode, ranker=ranker if mode == "ridge_top8" else None, max_steps=6)
        plans[mode], accounting = task.observed_greedy_search(seconds=10)
        assert accounting["complete"]
        assert all("all_current_legal_actions_scored" not in row for row in accounting["decisions"])
        tasks[mode] = task
    for mode, task in tasks.items():
        replay = task.fresh()
        for action in plans[mode]: replay.step(action)
        assert replay.terminated and replay.independent_geometry_check().feasible
        assert all(row["reason"] != "PROPOSED_UNCERTIFIED" or row["query_disposition"] in {
            "NOT_QUERIED", "CERTIFIED_LEGAL", "CERTIFIED_REJECTED"} for row in replay._ledger)
    assert ranker.fingerprint == frozen
    with pytest.raises(ValueError, match="nominal planning"):
        generated_training_rows(training)


def test_clone_and_stale_ranker_guard_no_native_requery():
    ranker = RidgePriority(np.zeros(FEATURES), np.ones(FEATURES), np.zeros(FEATURES), 0.)
    task = GeneratedSparseTask(fixture(), mode="ridge_top8", ranker=ranker, max_steps=6)
    before = task.costs.counts.copy()
    clone = task.planning_clone()
    clone.observation()
    assert task.costs.counts == before and clone._engine is not task._engine
    object.__setattr__(ranker, "intercept", 1.)
    with pytest.raises(RuntimeError, match="changed outside"):
        task.observation()


def test_unknown_stencil_samples_are_flagged_not_clipped():
    case = fixture(); engine, batch = raw(case)
    ray = replace(batch.proposals[0], entry_mm=(-20., -20., -20.), tip_mm=(-10., -10., -10.))
    rows = public_rows(case, engine.removed_mask, [ray], steps=0, horizon=6, current_tool=None)
    stencil = rows.features[0, :648].reshape(81, 8)
    assert np.all(stencil == 0) and rows.sampled_target_proxy[0] == 0


def test_refuse_obstruction_patient_condition_and_bad_ranker():
    case = fixture()
    with pytest.raises(ValueError, match="Generated"):
        GeneratedSparseTask(replace(case, proposal_config=replace(case.proposal_config, obstruction_opening=True)), mode="exhaustive")
    with pytest.raises(ValueError, match="Generated"):
        GeneratedSparseTask(replace(case, track="annotation_assisted"), mode="exhaustive")
    with pytest.raises(ValueError, match="Generated"):
        GeneratedSparseTask(case, mode="ridge_top8", ranker=None)


def test_time_limit_preserves_sparse_partial_accounting(monkeypatch):
    from resectionlab.observed_search import ObservedSearchLimit
    task = GeneratedSparseTask(fixture(), mode="sampled_target_top8", max_steps=6)
    def limited(self, **kw):
        raise ObservedSearchLimit("time", accounting={"complete": False, "method": "observed_greedy",
            "decisions": [{"all_current_legal_actions_scored": True}]}, best_sequence=("STOP",))
    monkeypatch.setattr(NativeSpatialTask, "_observed_greedy_search", limited)
    with pytest.raises(ObservedSearchLimit) as caught:
        task.observed_greedy_search(seconds=1.)
    assert not caught.value.accounting["complete"]
    assert caught.value.accounting["method"] == "sampled_target_top8"
    assert caught.value.best_sequence == ("STOP",)
    assert "all_current_legal_actions_scored" not in caught.value.accounting["decisions"][0]
