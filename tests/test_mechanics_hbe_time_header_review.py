"""Independent pinned-format controls; no solver or measured response access."""
import math

import numpy as np
import pytest

from scripts.mechanics_hbe_outputs import iter_data_records


def records(times):
    return [f"*Step = {i}\n*Time = {t:.9g}\n*Data = nodes\n1,1.23456789012\n"
            for i, t in enumerate(times)]


def parse(blocks, times):
    return list(iter_data_records("".join(blocks).splitlines(keepends=True),
        expected_times=times, item_count=1, field_count=1, record_name="nodes"))


@pytest.mark.parametrize("steps", [60, 120])
def test_header_print_precision_is_independent_of_numeric_rows(steps):
    times = np.linspace(0, 1, steps + 1)
    output = parse(records(times), times)
    assert len(output) == steps + 1
    assert output[1]["time"] == float(format(times[1], ".9g"))
    assert output[1]["declared_time"] == times[1]
    assert output[1]["time"] != output[1]["declared_time"]
    assert all(row["values"][0, 0] == 1.23456789012 for row in output)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "backtrack", "duplicate_time"])
def test_format_repair_does_not_relax_record_inventory(corruption):
    times = np.linspace(0, 1, 61)
    blocks = records(times)
    if corruption == "missing":
        del blocks[1]
    elif corruption == "duplicate":
        blocks.insert(1, blocks[1])
    elif corruption == "backtrack":
        blocks[1], blocks[2] = blocks[2], blocks[1]
    else:
        blocks[1] = blocks[1].replace("*Data", "*Time = 0.0166666667\n*Data")
    with pytest.raises(ValueError):
        parse(blocks, times)


def test_off_grid_one_float_increment_is_not_a_tolerance_match():
    times = np.linspace(0, 1, 61)
    blocks = records(times)
    printed = float(format(times[1], ".9g"))
    altered = math.nextafter(printed, math.inf)
    assert altered != printed and abs(altered - printed) < 5e-12
    blocks[1] = blocks[1].replace(f"{times[1]:.9g}", repr(altered))
    with pytest.raises(ValueError, match="time"):
        parse(blocks, times)


def test_aliased_declared_grid_rejects_before_reading_any_rows():
    times = [0.0, 0.12345678901, 0.12345678902, 1.0]
    assert times[1] != times[2]
    assert format(times[1], ".9g") == format(times[2], ".9g")
    def forbidden():
        raise AssertionError("Aliased declaration must reject before primitive reads")
        yield ""
    with pytest.raises(ValueError, match="alias"):
        list(iter_data_records(forbidden(), expected_times=times,
            item_count=1, field_count=1, record_name="nodes"))
