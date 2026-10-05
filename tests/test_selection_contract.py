"""A saved initial actor is not evidence that checkpoint selection completed."""
import copy

import pytest

from resectionlab.native_refinement import require_completed_selection_replay
from resectionlab.selection_contract import has_completed_selection


@pytest.fixture
def selected():
    # A completed initial STOP panel is a valid zero-update negative result.
    report = {"status": "wall_time_budget", "gradient_steps": 0,
        "selected_selection_return": 0., "selected_checkpoint_hash": "initial",
        "selection_history": [{"world_count": 3, "mean_return": 0., "checkpoint_hash": "initial"}],
        "replay": {"checkpoint_hash": "initial", "selection_partition_hash": "frozen-worlds"}}
    selection = {"role": "selection", "seeds": [11, 22, 33], "partition_hash": "frozen-worlds"}
    return report, {"partitions": {"selection": selection}}


def test_initial_zero_score_is_complete_on_the_actual_prescribed_panel(selected):
    report, contract = selected
    assert has_completed_selection(report, contract["partitions"]["selection"])
    require_completed_selection_replay(report, contract)


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -float("inf"), True, "0"])
def test_missing_or_nonfinite_selection_cannot_release_initial_checkpoint(selected, score):
    report, contract = selected
    report["selected_selection_return"] = score
    assert not has_completed_selection(report, contract["partitions"]["selection"])
    with pytest.raises(ValueError, match="completed selection"):
        require_completed_selection_replay(report, contract)


@pytest.mark.parametrize("mutate", [
    lambda r, c: r.update(selection_history=[]),
    lambda r, c: r["selection_history"][0].update(world_count=2),
    lambda r, c: r["selection_history"][0].update(checkpoint_hash="unselected"),
    lambda r, c: r["selection_history"][0].update(mean_return=1.),
    lambda r, c: c["partitions"]["selection"]["seeds"].append(44),
    lambda r, c: r["replay"].update(checkpoint_hash="unselected"),
    lambda r, c: r["replay"].update(selection_partition_hash="other-worlds"),
    lambda r, c: c["partitions"]["selection"].pop("partition_hash"),
    lambda r, c: r.update(status="cancelled"),
    lambda r, c: r.update(status="failed"),
])
def test_replay_needs_its_exact_completed_selection_evidence(selected, mutate):
    report, contract = copy.deepcopy(selected)
    mutate(report, contract)
    with pytest.raises(ValueError, match="completed selection"):
        require_completed_selection_replay(report, contract)
