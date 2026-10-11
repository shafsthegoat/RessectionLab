"""Geometric counterexamples for the independent cumulative-cut auditor."""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.independent_cut_union import IndependentCutCoverage, MODE


SPEC = {"mode": MODE, "max_depth": 1, "max_tracked_cells": 8,
        "max_leaf_tests_per_preview": 1000}
KEY = (1, 1, 1)


def digest(cells):
    return semantic_digest({"mode": MODE, "spec": SPEC, "cells": [
        {"index": list(key), "covered_bits_hex": format(bits, "x")}
        for key, bits in sorted(cells.items())]})


def setup():
    affine = np.eye(4)
    affine[:3, 3] = -1
    checker = IndependentCutCoverage(SPEC, affine, (3, 3, 3))
    remaining = np.zeros((3, 3, 3), bool)
    remaining[KEY] = True
    return checker, remaining, ~remaining


def witness(before, after, bits, *, proof="leaf_corners", tests=4):
    return {"before_digest": digest(before), "after_digest": digest(after),
        "new_leaf_count": bits.bit_count(), "leaf_tests": tests, "cells": [
            {"index": list(KEY), "added_bits_hex": format(bits, "x"), "proof": proof}]}


def run(checker, record, remaining, free, *, x=-.32, radius=.66, mode="aspirate", removed=()):
    checker.step(record, np.array([x, 0., -1.]), np.array([x, 0., 1.]),
                 radius, mode, remaining, free, set(removed))


def test_complementary_executed_half_cells_prove_whole_cell():
    c, remaining, free = setup()
    c.start({"spec": SPEC, "before_digest": digest({})})
    run(c, witness({}, {KEY: 15}, 15), remaining, free)
    assert c.ledger == {KEY: 15}
    # Whole-cell occupancy remains; only the second complementary proof completes it.
    assert remaining[KEY]
    run(c, witness({KEY: 15}, {}, 240), remaining, free, x=.32, removed=[KEY])
    c.finish({"after_digest": digest({}), "new_leaf_count": 8, "leaf_tests": 8,
              "coverage_only_progress": False}, 1, "aspirate")
    assert not c.ledger


def test_outer_corners_covered_by_different_capsules_does_not_prove_union():
    c, remaining, free = setup()
    # At radius .56 the original outer corners fit left/right capsules, but
    # the intervening y=+/- .5 points at x=0 lie outside BOTH capsules.
    assert np.hypot(.5-.32, .5) < .56 < np.hypot(.32, .5)
    with pytest.raises(ValueError, match="Subcell capsule"):
        run(c, witness({}, {KEY: 15}, 15), remaining, free, radius=.56)


def test_partial_coverage_cannot_claim_whole_cell_removed():
    c, remaining, free = setup()
    with pytest.raises(ValueError, match="incomplete"):
        run(c, witness({}, {}, 15), remaining, free, removed=[KEY])


def test_buried_cell_cannot_bank_partial_cutting():
    c, remaining, free = setup()
    with pytest.raises(ValueError, match="buried"):
        run(c, witness({}, {KEY: 15}, 15), remaining, np.zeros_like(free))


def test_probe_cannot_add_coverage():
    c, remaining, free = setup()
    with pytest.raises(ValueError, match="Non-aspiration"):
        run(c, witness({}, {KEY: 15}, 15), remaining, free, mode="probe")


def test_old_bits_cannot_be_counted_as_new_progress():
    c, remaining, free = setup()
    run(c, witness({}, {KEY: 15}, 15), remaining, free)
    with pytest.raises(ValueError, match="repeats"):
        run(c, witness({KEY: 15}, {KEY: 15}, 15), remaining, free)


@pytest.mark.parametrize("value", ["00f", "0xf", "F", "0", "100", "-1", "1"*1000])
def test_noncanonical_or_out_of_range_masks_rejected(value):
    c, remaining, free = setup()
    w = witness({}, {KEY: 15}, 15)
    w["cells"][0]["added_bits_hex"] = value
    with pytest.raises(ValueError):
        run(c, w, remaining, free)


@pytest.mark.parametrize("field,value", [("max_depth", True), ("max_depth", 4),
    ("max_tracked_cells", 0), ("max_leaf_tests_per_preview", 4194305), ("mode", "other")])
def test_invalid_or_unbounded_specs_rejected(field, value):
    spec = {**SPEC, field: value}
    with pytest.raises(ValueError):
        IndependentCutCoverage(spec, np.eye(4), (3, 3, 3))


def test_full_capsule_shortcut_has_independent_corner_check():
    c, remaining, free = setup()
    with pytest.raises(ValueError, match="Whole-cell capsule"):
        run(c, witness({}, {}, 255, proof="whole_cell_capsule", tests=0),
            remaining, free, removed=[KEY])
    c, remaining, free = setup()
    run(c, witness({}, {}, 255, proof="whole_cell_capsule", tests=0),
        remaining, free, x=0., radius=.8, removed=[KEY])
    assert not c.ledger


def test_unsupported_whole_cell_bank_rejected():
    c, remaining, free = setup()
    with pytest.raises(ValueError, match="must remove"):
        run(c, witness({}, {KEY: 255}, 255, proof="whole_cell_capsule", tests=0),
            remaining, free, x=0., radius=.8)


def test_full_leaf_bank_cannot_defer_evacuation_to_a_remote_action():
    c, remaining, free = setup()
    # Otherwise a later unrelated capsule could claim the old fully covered cell
    # as its own removal, moving removal timing and reward without new cutting.
    with pytest.raises(ValueError, match="removed immediately"):
        run(c, witness({}, {KEY: 255}, 255, tests=8),
            remaining, free, x=0., radius=.8)


@pytest.mark.parametrize("field,value", [("new_leaf_count", 3), ("leaf_tests", 3),
                                       ("after_digest", "forged")])
def test_claimed_accounting_cannot_replace_independent_proof(field, value):
    c, remaining, free = setup()
    w = witness({}, {KEY: 15}, 15)
    w[field] = value
    with pytest.raises(ValueError):
        run(c, w, remaining, free)


def test_source_continuity_and_duplicate_witnesses_rejected():
    c, remaining, free = setup()
    with pytest.raises(ValueError, match="different coverage"):
        c.start({"spec": SPEC, "before_digest": digest({KEY: 15})})
    w = witness({}, {KEY: 15}, 15)
    w["cells"].append(copy.deepcopy(w["cells"][0]))
    with pytest.raises(ValueError, match="Repeated"):
        run(c, w, remaining, free)


def test_coverage_only_progress_is_explicit_and_zero_removal():
    c, remaining, free = setup()
    c.start({"spec": SPEC, "before_digest": digest({})})
    run(c, witness({}, {KEY: 15}, 15), remaining, free)
    summary = {"after_digest": digest({KEY: 15}), "new_leaf_count": 4,
               "leaf_tests": 4, "coverage_only_progress": True}
    c.finish(summary, 0, "aspirate")
    with pytest.raises(ValueError, match="accounting"):
        c.finish({**summary, "coverage_only_progress": False}, 0, "aspirate")


def test_work_budget_is_cumulative_across_microsteps():
    c, remaining, free = setup()
    w = witness({}, {KEY: 15}, 15, tests=700)
    run(c, w, remaining, free)
    with pytest.raises(ValueError, match="work cap"):
        run(c, witness({KEY: 15}, {}, 240, tests=301), remaining, free, x=.32, removed=[KEY])


@pytest.mark.parametrize("angle", [.37, 1.2])
def test_partition_geometry_respects_oblique_affine_and_translation(angle):
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)],
                         [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    shift = np.array([12., -9., 31.])
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = shift - rotation @ np.ones(3)
    c = IndependentCutCoverage(SPEC, affine, (3, 3, 3))
    remaining = np.zeros((3, 3, 3), bool); remaining[KEY] = True
    for x, before, after, bits, removed in [
        (-.32, {}, {KEY: 15}, 15, set()), (.32, {KEY: 15}, {}, 240, {KEY})]:
        c.step(witness(before, after, bits), rotation @ np.array([x, 0., -1.]) + shift,
            rotation @ np.array([x, 0., 1.]) + shift, .66, "aspirate",
            remaining, ~remaining, removed)
    assert not c.ledger
