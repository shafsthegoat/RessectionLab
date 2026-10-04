"""Independent closed-form and printed-text controls; no measured curves/FEM runs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e = load("mechanics_hbe_evaluation")
o = load("mechanics_hbe_outputs")


def test_one_signed_scale_equal_modes_not_rows_and_no_zero_offset_repair():
    reference = {
        "compression": e.curve("compression", [0, -1, -4], [-1, -1, -1]),
        "tension": e.curve("tension", [0, 1], [1, 1]),
    }
    observed = {
        "compression": e.curve("compression", [0, -1, -4], [-2, -2, -2]),
        "tension": e.curve("tension", np.linspace(0, 1, 101), [4] * 101),
    }
    result = e.fit_scale(observed, reference)
    assert result["scale"] == pytest.approx(3)
    assert result["mu_Pa"] == pytest.approx(3000)
    assert result["numerator"] == pytest.approx(3)
    assert result["denominator"] == pytest.approx(1)
    np.testing.assert_allclose(result["modes"]["compression"]["weights"], [1 / 16, 1 / 4, 3 / 16])
    assert result["modes"]["compression"]["observed"][0] == -2  # retained zero-input response
    reversed_response = {key: e.curve(key, value.coordinate, -np.array(value.response))
                         for key, value in observed.items()}
    with pytest.raises(ValueError, match="Positive"):
        e.fit_scale(reversed_response, reference)
    with pytest.raises(ValueError, match="extrapolation"):
        e.interpolate(reference["tension"], [-1e-12])


def test_torque_branch_balance_units_endpoint_signs_and_odd_symmetry():
    observed = {
        "torsion_neg": e.curve("torsion_neg", [0, -.1, -.4], [-2] * 3),
        "torsion_pos": e.curve("torsion_pos", np.linspace(0, .4, 81), [4] * 81),
    }
    prediction = {
        "torsion_neg": e.curve("torsion_neg", [0, -.4], [-3, -3]),
        "torsion_pos": e.curve("torsion_pos", [0, .4], [7, 7]),
    }
    result = e.paired_metrics(observed, prediction, characteristic_response_scale=1)
    assert result["equal_branch_RMSE"] == pytest.approx(np.sqrt(5))
    assert result["equal_branch_normalized_RMSE"] == pytest.approx(np.sqrt(13 / 32))
    assert result["branches"]["torsion_neg"]["signed_endpoint_bias"] == -1
    assert result["branches"]["torsion_pos"]["signed_endpoint_bias"] == 3
    assert {r["response_unit"] for r in result["branches"].values()} == {"Nm"}
    assert result["odd_symmetry"]["observed_RMSE_Nm"] == pytest.approx(2)
    assert result["odd_symmetry"]["predicted_RMSE_Nm"] == pytest.approx(4)
    assert len(result["odd_symmetry"]["angles_rad"]) == 21
    assert result["physical_validation_pass"] is None
    assert result["empirical_tolerance"] is None


def test_near_zero_denominator_is_null_at_declared_floor():
    floor = 64 * np.finfo(float).eps
    observed = e.curve("torsion_pos", [0, .1], [floor, floor])
    predicted = e.curve("torsion_pos", [0, .1], [1, 1])
    result = e.branch_metrics(observed, predicted, characteristic_response_scale=1)
    assert result["observed_RMS_denominator"] == floor
    assert result["normalized_RMSE"] is None
    assert result["normalized_error_reason"]


def test_paired_metric_mapping_cannot_relabel_two_compression_curves_as_two_modes():
    value = e.curve("compression", [0, -1], [0, -1])
    wrong = {"compression": value, "tension": value}
    with pytest.raises(ValueError):
        e.paired_metrics(wrong, wrong, characteristic_response_scale=1)


def primitives(steps):
    for step in range(steps + 1):
        yield f"*Step = {step}\n"
        # DataRecord.cpp headers use %.9lg; the prior fixture mistakenly used
        # the separate numeric-row precision (12 significant digits) here.
        yield f"*Time = {step / steps:.9g}\n"
        yield "*Data = nodes\n"
        yield f"2,{step},2,3\n"
        yield f"1,{step},4,5\n"


@pytest.mark.parametrize("steps", [60, 120])
def test_actual_declared_primitive_counts_include_zero_and_canonical_ids(steps):
    records = list(o.iter_data_records(primitives(steps), expected_times=np.linspace(0, 1, steps + 1),
        item_count=2, field_count=3, record_name="nodes"))
    assert len(records) == steps + 1
    assert records[0]["time"] == 0 and records[-1]["time"] == 1
    np.testing.assert_array_equal(records[-1]["values"], [[steps, 4, 5], [steps, 2, 3]])


def solver_log(times, *, actual="1.000000e-15", required="1.000000e-14"):
    for time in times:
        # Pinned FESolidSolver2.cpp uses %lg for time and %15le for residuals.
        yield f" Nonlinear solution status: time= {time:.6g}\n"
        yield f" residual 1.000000e-06 {actual} {required}\n"
    yield "N O R M A L T E R M I N A T I O N\n"


@pytest.mark.parametrize("steps", [60, 120])
def test_real_source_printf_precision_accepts_all_declared_solver_states(steps):
    times = np.linspace(0, 1, steps + 1)
    result = o.check_solver_records(solver_log(times[1:]), expected_times=times, residual_floor_N2=1e-20)
    assert result["passed"]
    assert len(result["states"]) == steps


def test_same_time_iterations_allowed_but_time_backtracking_rejected():
    args = dict(expected_times=[0, .5, 1], residual_floor_N2=1e-20)
    assert o.check_solver_records(solver_log([.5, .5, 1, 1]), **args)["passed"]
    with pytest.raises(ValueError):
        o.check_solver_records(solver_log([.5, 1, .5, 1]), **args)


def test_solver_log_cannot_relabel_loose_reported_tolerance_as_frozen_residual_gate():
    with pytest.raises(ValueError):
        o.check_solver_records(solver_log([.5, 1], actual="1e-5", required="1e-4"),
            expected_times=[0, .5, 1], residual_floor_N2=1e-20)


def test_final_iteration_missing_residual_cannot_reuse_previous_iteration():
    lines = list(solver_log([.5, 1]))[:-1]
    lines += ["Nonlinear solution status: time= 1\n", "NORMAL TERMINATION\n"]
    with pytest.raises(ValueError):
        o.check_solver_records(lines, expected_times=[0, .5, 1], residual_floor_N2=1e-20)


def test_solver_log_requires_nonempty_complete_increasing_declaration():
    with pytest.raises(ValueError):
        o.check_solver_records(["NORMAL TERMINATION"], expected_times=[0], residual_floor_N2=1e-20)
    with pytest.raises(ValueError):
        o.check_solver_records(solver_log([1, .5]), expected_times=[0, 1, .5], residual_floor_N2=1e-20)


def test_primitive_stream_rejects_wrong_header_token_even_if_record_name_matches():
    lines = [line.replace("*Data =", "*DataUnexpected =") for line in primitives(60)]
    with pytest.raises(ValueError):
        list(o.iter_data_records(lines, expected_times=np.linspace(0, 1, 61),
            item_count=2, field_count=3, record_name="nodes"))
