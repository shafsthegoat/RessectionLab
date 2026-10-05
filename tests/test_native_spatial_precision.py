"""Source-cell arithmetic regressions; no patient arrays or policy execution."""
from dataclasses import replace
import math

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode, _same_number
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task


def scaled_fractional_episode(memberships, scale):
    original = make_native_opening_task()
    case = original.case
    reference = np.zeros(case.reference_target.shape, np.float32)
    reference[case.observed_support] = np.asarray(memberships, dtype=np.float32)
    tools = tuple(replace(tool, tip_radius_mm=tool.tip_radius_mm * scale,
                          shaft_radius_mm=tool.shaft_radius_mm * scale,
                          working_length_mm=tool.working_length_mm * scale,
                          tip_length_mm=tool.tip_length_mm * scale) for tool in case.tools)
    source = replace(case, reference_target=reference,
                     affine_ras_mm=np.diag([scale, scale, scale, 1.]), tools=tools,
                     access=AccessWindow(case.access.center_mm * scale, case.access.normal_inward,
                                         case.access.radius_mm * scale, case.access.window_id))
    task = NativeSpatialTask(source, max_steps=2)
    for tool, voxel in zip(tools, ([4, 4, 1], [4, 4, 5])):
        action = next(row["action_id"] for row in task.candidate_inventory()["ledger"]
                      if row["feasible"] and row["tool_id"] == tool.tool_id and row["voxel"] == voxel)
        task.step(action)
    return task


@pytest.mark.parametrize("scale", [1.0000011580409386, 1.23456789])
@pytest.mark.parametrize("memberships", [
    [1., 0., 1., 1., 0., 1.],
    [.1, .2, .3, .4, .6, .7],
    [np.nextafter(np.float32(1.), np.float32(0.))] * 6,
])
def test_split_native_removals_match_float64_membership_mass_and_union(memberships, scale):
    task = scaled_fractional_episode(memberships, scale)
    metrics = task.metrics()
    volume = float(abs(np.linalg.det(task.case._native_affine_ras_mm[:3, :3])))
    union = set()
    target_by_action, normal_by_action = [], []
    for record in metrics["history"]:
        cells = [tuple(cell) for cell in record["removed_indices_native"]]
        assert not union.intersection(cells)
        union.update(cells)
        target = math.fsum(float(task.case.reference_target[cell]) for cell in cells) * volume
        normal = len(cells) * volume - target
        assert record["target_removed_mm3"] == pytest.approx(target, abs=1e-12, rel=1e-14)
        assert record["normal_removed_mm3"] == pytest.approx(normal, abs=1e-12, rel=1e-14)
        target_by_action.append(target)
        normal_by_action.append(normal)
    assert len(union) == 6  # Preserve the existing physical opening fixture.
    target_union = math.fsum(float(task.case.reference_target[cell]) for cell in union) * volume
    assert metrics["target_removed_mm3"] == pytest.approx(target_union, abs=1e-12, rel=1e-14)
    assert metrics["normal_removed_mm3"] == pytest.approx(6 * volume - target_union, abs=1e-12, rel=1e-14)
    assert metrics["target_removed_mm3"] == pytest.approx(math.fsum(target_by_action), abs=1e-12, rel=1e-14)
    assert metrics["normal_removed_mm3"] == pytest.approx(math.fsum(normal_by_action), abs=1e-12, rel=1e-14)
    result = evaluate_native_spatial_episode(task)
    assert result["accepted"] and result["geometry"]["complete_tool_checked"]
    assert result["outcomes"]["total_reference_target_mm3"] == pytest.approx(target_union, abs=1e-12, rel=1e-14)
    assert result["outcomes"]["target_removed_mm3"] == pytest.approx(target_union, abs=1e-12, rel=1e-14)
    assert result["outcomes"]["normal_removed_mm3"] == pytest.approx(6 * volume - target_union, abs=1e-12, rel=1e-14)


def test_support_excluded_reference_mass_also_uses_float64():
    original = make_native_opening_task()
    reference = np.array(original.case.reference_target)
    reference[0, 0, :6] = [.1, .2, .3, .4, .6, .7]
    task = NativeSpatialTask(replace(original.case, reference_target=reference), max_steps=2)
    expected = math.fsum(float(value) for value in reference[0, 0, :6])
    assert task.metrics()["reference_target_outside_observed_support_mm3"] == pytest.approx(expected, abs=1e-12, rel=1e-14)


def test_accounting_acceptance_tolerance_is_not_relaxed():
    _same_number(1., 1. + .9e-7, "control")
    with pytest.raises(ValueError, match="accounting mismatch"):
        _same_number(1., 1. + 1.1e-7, "control")
    with pytest.raises(ValueError, match="accounting mismatch"):
        _same_number(0., 1.1e-7, "near-zero normal volume")
