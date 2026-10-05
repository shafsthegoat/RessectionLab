"""Selection completion evidence shared by experiment and desktop consumers."""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any


def _finite_return(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def has_completed_selection(report: Mapping[str, Any], selection: Mapping[str, Any]) -> bool:
    """Require the selected checkpoint's complete frozen-world panel, including 0.

    A checkpoint file exists before the first selection panel finishes. Neither
    its existence, its hash, nor a budget-exhaustion status proves selection.
    Trainers append history only after every prescribed world finishes; partial
    rollout resource counts remain in the raw report, outside this evidence.
    Cancellation/failure eligibility is checked separately by each consumer.
    """
    if not isinstance(report, Mapping) or not isinstance(selection, Mapping):
        return False
    seeds = selection.get("seeds")
    if (selection.get("role") != "selection" or not isinstance(seeds, (list, tuple))
            or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        return False
    score, checkpoint = report.get("selected_selection_return"), report.get("selected_checkpoint_hash")
    history = report.get("selection_history")
    if not _finite_return(score) or not isinstance(checkpoint, str) or not checkpoint:
        return False
    if not isinstance(history, (list, tuple)):
        return False
    return any(isinstance(panel, Mapping)
        and type(panel.get("world_count")) is int and panel["world_count"] == len(seeds)
        and panel.get("checkpoint_hash") == checkpoint
        and _finite_return(panel.get("mean_return")) and panel["mean_return"] == score
        for panel in history)
