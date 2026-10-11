"""Independent same-capsule parent exposure closure (no leaf-topology change)."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.evaluation import independent_check_native_history
from resectionlab.independent_cut_union import (
    IndependentCutCoverage, MODE, PARENT_CLOSURE_MODE, PARENT_CLOSURE_EXPOSURE)


A, B = (2, 2, 2), (2, 2, 3)
BASE = {"mode": MODE, "max_depth": 1, "max_tracked_cells": 8,
        "max_leaf_tests_per_preview": 1000}
DECLARED = {**BASE, "mode": PARENT_CLOSURE_MODE, "exposure_mode": PARENT_CLOSURE_EXPOSURE}


def digest(spec, cells):
    return semantic_digest({"mode": spec['mode'], "spec": spec, "cells": [
        {"index": list(key), "covered_bits_hex": format(bits, 'x')}
        for key, bits in sorted(cells.items())]})


def proof(spec, *, reverse=False):
    rows = [{"index": list(A), "added_bits_hex": 'ff', "proof": 'whole_cell_capsule'},
            {"index": list(B), "added_bits_hex": '55', "proof": 'leaf_corners'}]
    return {"before_digest": digest(spec, {}), "after_digest": digest(spec, {B: 85}),
            "new_leaf_count": 12, "leaf_tests": 4, "cells": rows[::-1] if reverse else rows}


def low_level(spec=DECLARED):
    c = IndependentCutCoverage(spec, np.eye(4), (5, 5, 6))
    material = np.zeros(c.shape, bool); material[A] = material[B] = True
    free = np.zeros(c.shape, bool); free[2, 2, 1] = True
    return c, material, free


def apply(c, material, free, record, *, removed=(A,), radius=.8):
    c.step(record, np.array([2., 2., 0.]), np.array([2., 2., 2.9]), radius,
           'aspirate', material, free, set(removed))


@pytest.mark.parametrize('reverse', [False, True])
def test_verified_same_step_parent_exposes_partial_neighbor_independent_of_row_order(reverse):
    c, material, free = low_level()
    apply(c, material, free, proof(DECLARED, reverse=reverse))
    assert c.ledger == {B: 85}
    assert material[B]  # Partial certification cannot clear its shaft envelope.
    assert c.digest() == digest(DECLARED, {B: 85})
    old, material, free = low_level(BASE)
    with pytest.raises(ValueError, match='buried'):
        apply(old, material, free, proof(BASE, reverse=reverse))


def test_same_step_removed_parent_must_itself_have_complete_geometric_proof():
    c, material, free = low_level()
    forged = proof(DECLARED); forged['cells'] = forged['cells'][1:]
    forged['new_leaf_count'] = 4
    with pytest.raises(ValueError, match='incomplete'):
        apply(c, material, free, forged)
    c, material, free = low_level()
    with pytest.raises(ValueError, match='Subcell capsule'):
        apply(c, material, free, proof(DECLARED), radius=.71)


@pytest.mark.parametrize('bridge', ['partial', 'diagonal', 'future', 'pocket'])
def test_partial_diagonal_future_and_unconnected_empty_pockets_cannot_seed_exposure(bridge):
    c, material, free = low_level()
    before, removed = {}, set()
    if bridge == 'partial':
        # Even genuinely prior-certified bits do not change this PARENT rule.
        before = {A: 15}; c.ledger = dict(before)
    if bridge == 'diagonal': removed = {(1, 2, 2)}
    if bridge == 'pocket':
        material[A] = False  # Empty, but absent from prior connected-free mask.
    record = {"before_digest": digest(DECLARED, before),
        "after_digest": digest(DECLARED, {**before, B: 85}),
        "new_leaf_count": 4, "leaf_tests": 4,
        "cells": [{"index": list(B), "added_bits_hex": '55', "proof": 'leaf_corners'}]}
    with pytest.raises(ValueError, match='buried'):
        apply(c, material, free, record, removed=removed)


@pytest.mark.parametrize('change', [
    {'mode': MODE}, {'exposure_mode': None}, {'exposure_mode': 'leaf_faces'},
    {'max_depth': 4}, {'max_leaf_tests_per_preview': 4194305}])
def test_new_mode_is_explicit_and_cannot_raise_existing_limits(change):
    with pytest.raises(ValueError):
        IndependentCutCoverage({**DECLARED, **change}, np.eye(4), (5, 5, 6))
    assert IndependentCutCoverage(BASE, np.eye(4), (5, 5, 6)).digest() == digest(BASE, {})


@pytest.fixture(scope='module', params=[False, True])
def histories(request):
    from resectionlab.geometry import AccessWindow, ToolGeometry
    from resectionlab.native_cut_union import CutUnionSpec, EvacuatedVolumeSpec
    from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
    material = np.zeros((5, 5, 6), bool); material[:, :, 2:] = True
    volume = (EvacuatedVolumeSpec(material.astype(float)*.25, material.astype(float)*.25,
                                 'generated unlocated parent mass') if request.param else None)
    results = []
    for exposure in (None, PARENT_CLOSURE_EXPOSURE):
        cfg = NativeResectionConfig(material, material.astype(np.int16), np.eye(4),
            AccessWindow([2., 2., .5], [0., 0., 1.], 3.),
            (ToolGeometry('generated-closure', .8, .05, 5., 30., 2.),),
            'generated-parent-closure', 'generated plane support',
            cut_union=CutUnionSpec(exposure_mode=exposure), evacuated_volume=volume)
        engine = NativeResectionEngine(cfg)
        result = engine.execute_stroke('generated-closure', [2., 2., 2.25], entry_mm=[2., 2., .5])
        assert result.feasible
        case = SimpleNamespace(mri=np.zeros(material.shape), affine=cfg.affine,
                               frame='RAS+', semantic_hash=cfg.source_hash)
        results.append((cfg, case, copy.deepcopy(engine.history), dict(engine._cut_union_ledger)))
    return results


def audit(cfg, case, history, **override):
    arguments = dict(tissue_mask=cfg.tissue_mask, access=cfg.access,
                     cut_union_spec=cfg.cut_union.to_record())
    if cfg.evacuated_volume is not None:
        arguments.update(evacuated_volume_spec=cfg.evacuated_volume.to_record(),
            evacuated_target_fraction_lower=cfg.evacuated_volume.target_fraction_lower,
            evacuated_target_fraction_upper=cfg.evacuated_volume.target_fraction_upper)
    arguments.update(override)
    return independent_check_native_history(case, cfg.tools, history, **arguments)


def test_actual_last_microstep_closure_has_independent_geometry_and_declared_volume(histories):
    old, new = histories
    assert not old[3].get(B, 0) and new[3].get(B, 0)
    for cfg, case, history, _ in histories:
        checked = audit(cfg, case, history)
        assert checked.feasible
        assert checked.contained_source_tissue_volume_mm3 == 1.
        assert checked.complete_tool_checked and checked.frontier_checked
    cfg, case, history, _ = new
    wrong = audit(cfg, case, history, cut_union_spec=old[0].cut_union.to_record())
    assert not wrong.feasible and wrong.failures == ('unsupported_native_footprint_model',)
    assert audit(cfg, case, []).feasible
    if cfg.evacuated_volume is not None:
        before = audit(old[0], old[1], old[2]).to_dict()['evacuated_volume']
        after = audit(cfg, case, history).to_dict()['evacuated_volume']
        assert after['evacuated_source_volume_mm3'] > before['evacuated_source_volume_mm3']


def test_existing_parent_connectivity_gate_still_rejects_a_buried_removal_component(histories):
    cfg, case, saved, _ = histories[1]
    history = copy.deepcopy(saved)
    # Isolate the adversarial removal interval. With no prior connected free
    # parent, its alleged clearance must fail the original parent graph walk,
    # before any same-step partial exposure or remaining/free update is possible.
    row = history[0]
    terminal = next(m for m in reversed(row['microsteps']) if m['removed_indices_native'])
    terminal['tip_start_mm'] = row['entry_mm']
    terminal['active_stroke_start_mm'] = [2., 2., -1.5]
    row['microsteps'] = [terminal]
    result = audit(cfg, case, history, tissue_mask=np.ones(cfg.tissue_mask.shape, bool))
    assert not result.feasible and result.failures == ('disconnected_native_removal',)
    assert result.certified_evacuated_geometry is None
