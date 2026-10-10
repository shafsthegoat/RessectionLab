"""Finite invented-tree/physical-cell controls; no patient, model or training."""
from dataclasses import replace
import os

import numpy as np
import pytest

from test_observed_search_opening_lane import Tree, clean, load, opening_tree

search = load(os.environ.get("OBSERVED_SEARCH_CANDIDATE", "src/resectionlab/observed_search.py"),
              "opening_volume_candidate")
baseline = (load(os.environ["OBSERVED_SEARCH_BASELINE"], "opening_volume_baseline")
            if "OBSERVED_SEARCH_BASELINE" in os.environ else None)
MODE = "return_plus_opening_depth_volume_v1"
OLD = "return_plus_opening_depth_v1"


def widening_tree():
    return Tree({(): {"seed": (-1., ((1, 1, 3),), False)},
        ("seed",): {"cheap": (-.05, (), False), "penny": (-.1, ((1, 1, 2),), False),
                    "wide": (-2., ((2, 1, 2), (3, 1, 2), (4, 1, 2)), False)},
        ("seed", "cheap"): {"end": (-.1, (), True)},
        ("seed", "penny"): {"end": (-.1, (), True)},
        ("seed", "wide"): {"reach": (5., ((2, 1, 4),), True)}}, max_steps=3)


@pytest.mark.parametrize("transition", ["eager", "lazy_planning"])
def test_equal_depth_widening_retained_without_changing_terminal_reward_or_calls(transition):
    old, task = widening_tree(), widening_tree()
    options = dict(max_calls=30, beam_width=2, seconds=10, transition_mode=transition,
                   retained_prefix_diagnostics=True)
    old_path, old_report = search.observed_beam_search(old, **options, retention_mode=OLD)
    path, report = search.observed_beam_search(task, **options, retention_mode=MODE)
    assert old_path == ("STOP",) and path == ("seed", "wide", "reach")
    assert report["estimated_incremental_return"] == 2. and report["stop_baseline_return"] == 0.
    assert report["model_transition_calls"] == old_report["model_transition_calls"] == 6
    assert task.counters == old.counters
    rows = report["layers"][1]["opening_depth_retention"]
    assert [(row["actions"], row["reason"], row["opening_depth_mm"], row["opening_removed_volume_mm3"])
            for row in rows] == [(["seed", "cheap"], "best_return", 3., 1.),
                                (["seed", "wide"], "opening", 3., 4.)]
    assert report["peak_next_layer_states"] <= 4
    assert report["opening_depth_retention"]["extra_observations_or_previews"] == 0


def test_negative_wide_path_still_returns_stop():
    task = widening_tree()
    task.tree[("seed", "wide")]["reach"] = (-1., ((2, 1, 4),), True)
    path, report = search.observed_beam_search(task, max_calls=30, beam_width=2, retention_mode=MODE)
    assert path == ("STOP",) and report["estimated_incremental_return"] == 0.


def test_depth_remains_primary_even_against_much_larger_volume():
    task = Tree({(): {"cheap": (-.1, ((1, 1, 1),), False),
                      "wide": (-2., ((2, 1, 2), (3, 1, 2), (4, 1, 2)), False),
                      "deeper": (-3., ((1, 1, 3),), False)}})
    _, report = search.observed_beam_search(task, max_calls=3, beam_width=2, retention_mode=MODE)
    assert report["layers"][0]["opening_depth_retention"][1]["actions"] == ["deeper"]


def test_volume_counts_native_mm3_not_tip_motion_or_reported_reward_components():
    task = opening_tree(); observed = task.observation(); result = task.planning_clone().step("open")
    affine = np.diag([2., 3., 4., 1.]); affine[:3, 3] = [10., -7., 5.]
    observed.affine_ras_mm = affine
    observed.state_features[3:6] = affine[:3, 3]
    result.info.update(native_affine=affine.tolist(), target_removed_mm3=1e12,
                       normal_removed_mm3=1e12, insertion_distance_mm=1e12)
    depth, volume = search._public_removed_opening_depth(observed, result, "open", with_volume=True)
    assert depth == 12. and volume == pytest.approx(24.)
    transformed = np.array([[0., 0., 1., 20.], [1., 0., 0., -2.], [0., 1., 0., 9.], [0., 0., 0., 1.]])
    observed.affine_ras_mm = transformed @ affine
    observed.state_features[3:6] = observed.affine_ras_mm[:3, 3]
    observed.state_features[6:9] = transformed[:3, :3] @ np.array([0., 0., 1.])
    result.info["native_affine"] = observed.affine_ras_mm.tolist()
    assert search._public_removed_opening_depth(observed, result, "open", with_volume=True) == pytest.approx((depth, volume))
    reflection = np.diag([-1., 1., 1., 1.])
    observed.affine_ras_mm = reflection @ observed.affine_ras_mm
    observed.state_features[3:6] = observed.affine_ras_mm[:3, 3]
    observed.state_features[6:9] = reflection[:3, :3] @ observed.state_features[6:9]
    result.info["native_affine"] = observed.affine_ras_mm.tolist()
    assert search._public_removed_opening_depth(observed, result, "open", with_volume=True) == pytest.approx((depth, volume))


@pytest.mark.parametrize("condition", ["no_op", "uncovered", "outside", "no_support", "behind_plane"])
def test_accessible_unknown_or_uncommitted_volume_earns_no_credit(condition):
    task = opening_tree(); obs = task.observation(); result = task.planning_clone().step("open")
    if condition == "no_op": result.info["removed_indices_native"] = []
    if condition == "uncovered": obs.coverage[1, 1, 1, 3] = False
    if condition == "outside": obs.affine_ras_mm = np.eye(4); obs.affine_ras_mm[0, 3] = 20.
    if condition == "no_support": obs.image_channels[1, 1, 1, 3] = 0.
    if condition == "behind_plane": obs.state_features[3:6] = [0., 0., 4.]
    assert search._public_removed_opening_depth(obs, result, "open", with_volume=True) == (None, 0.)


@pytest.mark.parametrize("condition", ["duplicate", "prior_cavity", "private_record", "singular_frame"])
def test_malformed_or_previously_removed_volume_is_refused(condition):
    task = opening_tree(); obs = task.observation(); result = task.planning_clone().step("open")
    if condition == "duplicate": result.info["removed_indices_native"] = [[1, 1, 3], [1, 1, 3]]
    if condition == "prior_cavity": obs.image_channels[3, 1, 1, 3] = 1.
    if condition == "private_record": result.info["outcome_scope"] = "separate_evaluator_reference"
    if condition == "singular_frame": result.info["native_affine"][0][0] = 0.
    with pytest.raises(ValueError):
        search._public_removed_opening_depth(obs, result, "open", with_volume=True)


def test_cumulative_volume_cannot_credit_repeated_committed_cell():
    task = widening_tree()
    task.tree[("seed",)]["wide"] = (-2., ((1, 1, 3),), False)
    with pytest.raises(ValueError, match="previously observed cavity"):
        search.observed_beam_search(task, max_calls=30, beam_width=2, retention_mode=MODE)


def test_cap_remains_partial_and_transition_failure_keeps_accounting():
    _, report = search.observed_beam_search(widening_tree(), max_calls=4, beam_width=2, retention_mode=MODE)
    assert report["call_cap_reached"] and report["model_transition_calls"] == 4
    assert report["completed_layers"] == 2 and report["estimated_incremental_return"] == 0.
    task = widening_tree(); task.failure = True
    with pytest.raises(InterruptedError) as caught:
        search.observed_beam_search(task, max_calls=30, beam_width=2, retention_mode=MODE)
    assert caught.value.accounting["model_transition_calls"] == 1
    assert caught.value.accounting["completed_layers"] == 0


def test_native_committed_cells_and_complete_replay_support_new_mode():
    from resectionlab.native_spatial_task import make_native_opening_task
    task = make_native_opening_task(max_steps=2)
    path, report = search.observed_beam_search(task, max_calls=32, beam_width=2, seconds=10.,
        transition_mode="lazy_planning", retention_mode=MODE)
    assert len(path) == 2 and "STOP" not in path and not report["call_cap_reached"]
    model = task.planning_clone()
    observed = model.observation()
    first = model.step(path[0])
    depth, volume = search._public_removed_opening_depth(observed,
        first, path[0], with_volume=True)
    assert depth > 0 and volume == len(first.info["removed_indices_native"])
    for action in path:
        task.step(action)
    assert task.terminated and task.metrics()["total_reward"] == pytest.approx(report["estimated_incremental_return"])
    assert task.independent_geometry_check().feasible

@pytest.mark.parametrize("mode", ["return_only", OLD])
@pytest.mark.parametrize("transition", ["eager", "lazy_planning"])
@pytest.mark.parametrize("cap", [0, 1, 4, 30])
@pytest.mark.skipif(baseline is None, reason="historical source supplied during staged review")
def test_old_modes_exact_paths_counters_and_accounting(mode, transition, cap):
    a, b = widening_tree(), widening_tree()
    options = dict(max_calls=cap, beam_width=2, seconds=10, transition_mode=transition,
                   retained_prefix_diagnostics=True, retention_mode=mode)
    pa, ra = baseline.observed_beam_search(a, **options)
    pb, rb = search.observed_beam_search(b, **options)
    assert pa == pb and clean(ra) == clean(rb) and a.counters == b.counters
