"""Adversarial independent evacuation accounting; no patient/reference anatomy."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.evaluation import independent_check_native_history
from resectionlab.independent_cut_union import (
    IndependentCutCoverage, MODE, EVACUATION_VERSION, EVACUATION_SCOPE)


SPEC = {"mode": MODE, "max_depth": 1, "max_tracked_cells": 8,
        "max_leaf_tests_per_preview": 1000}
KEY = (1, 1, 1)


def declaration(lower, upper):
    return {"version": EVACUATION_VERSION, "shape": list(lower.shape),
            "target_fraction_lower_hash": array_digest(lower),
            "target_fraction_upper_hash": array_digest(upper),
            "provenance": "generated unlocated source mass",
            "target_semantics": "unlocated_parent_target_mass_bounds"}


def setup(lower=.25, upper=.25, affine=None):
    material = np.zeros((3, 3, 3), bool); material[KEY] = True
    lo = material.astype(np.float64)*lower
    hi = material.astype(np.float64)*upper
    spec = declaration(lo, hi)
    if affine is None:
        affine = np.eye(4); affine[:3, 3] = -1
    c = IndependentCutCoverage(SPEC, affine, material.shape,
        evacuated_volume_spec=spec, target_fraction_lower=lo,
        target_fraction_upper=hi, tissue_mask=material)
    return c, material, lo, hi


def ledger_digest(cells):
    return semantic_digest({"mode": MODE, "spec": SPEC, "cells": [
        {"index": list(key), "covered_bits_hex": format(bits, "x")}
        for key, bits in sorted(cells.items())]})


def snapshot(c, *, partial=0, whole=False):
    """Hand-described one-parent state; expected volumes are rational halves."""
    removed = np.zeros(c.shape, bool); removed[KEY] = whole
    cells = {} if not partial else {KEY: partial}
    descriptor = {"version": EVACUATION_VERSION, "cut_union_spec": SPEC,
        "evacuated_volume_spec": c.evacuation.spec, "affine_hash": array_digest(c.matrix),
        "whole_removed_mask_hash": array_digest(removed),
        "partial_coverage_digest": ledger_digest(cells)}
    count = 8 if whole else partial.bit_count()
    v = c.evacuation.parent_volume
    r = count*v/8
    bounds = [max(0., r-(v-float(c.evacuation.lower[KEY])*v)),
              min(r, float(c.evacuation.upper[KEY])*v)]
    return {"state_digest": semantic_digest(descriptor),
        "whole_parent_clearance_count": int(whole),
        "partial_certified_leaf_count": partial.bit_count(),
        "certified_leaf_count": count, "evacuated_source_volume_mm3": r,
        "evacuated_target_source_bounds_mm3": bounds}


def half_action(c, material, *, second=False, mutate=None):
    before, after = ({KEY: 15}, {}) if second else ({}, {KEY: 15})
    bits = 240 if second else 15
    start = snapshot(c, partial=15 if second else 0)
    finish = snapshot(c, whole=True) if second else snapshot(c, partial=15)
    evacuation = {"spec": c.evacuation.spec, "before": start, "after": finish,
        "delta_certified_leaf_count": 4,
        "delta_evacuated_source_volume_mm3": c.evacuation.parent_volume/2,
        "scope": EVACUATION_SCOPE}
    if mutate is not None:
        mutate(evacuation)
    macro = {"spec": SPEC, "before_digest": ledger_digest(before),
        "after_digest": ledger_digest(after), "new_leaf_count": 4,
        "leaf_tests": 4, "coverage_only_progress": not second}
    c.start(macro, evacuation)
    micro = {"before_digest": ledger_digest(before), "after_digest": ledger_digest(after),
        "new_leaf_count": 4, "leaf_tests": 4,
        "cells": [{"index": list(KEY), "added_bits_hex": format(bits, 'x'), "proof": "leaf_corners"}]}
    x = .32 if second else -.32
    c.step(micro, np.array([x, 0., -1.]), np.array([x, 0., 1.]), .66,
           "aspirate", material, ~material, {KEY} if second else set())
    c.finish(macro, int(second), "aspirate", evacuation)
    return evacuation


@pytest.mark.parametrize("lo,hi,partial,complete", [
    (.25, .25, [0., .25], [.25, .25]),
    (.2, .4, [0., .4], [.2, .4]),
    (1., 1., [.5, .5], [1., 1.]),
    (0., 0., [0., 0.], [0., 0.])])
def test_executed_complements_have_unique_volume_and_coherent_target_interval(lo, hi, partial, complete):
    c, material, _, _ = setup(lo, hi)
    a = half_action(c, material)
    assert material[KEY]  # partial volume is not shaft clearance
    assert c.evacuation.snapshot()['evacuated_source_volume_mm3'] == .5
    assert c.evacuation.snapshot()['evacuated_target_source_bounds_mm3'] == partial
    b = half_action(c, material, second=True)
    assert c.evacuation.snapshot()['evacuated_source_volume_mm3'] == 1.
    assert c.evacuation.snapshot()['evacuated_target_source_bounds_mm3'] == pytest.approx(complete)
    assert a['delta_evacuated_source_volume_mm3']+b['delta_evacuated_source_volume_mm3'] == 1.
    assert c.evacuation.geometry()['whole_removed_indices'] == [list(KEY)]
    assert not c.evacuation.geometry()['partial_cells']
    # Ledger returns to empty after clearance; the whole mask must still reject
    # a fabricated second removal even if a direct caller lies about remaining.
    with pytest.raises(ValueError):
        half_action(c, material)
    fresh, material, _, _ = setup(lo, hi)
    assert fresh.evacuation.snapshot()['certified_leaf_count'] == 0
    assert half_action(fresh, material) == a


@pytest.mark.parametrize("mutate", [
    lambda r: r['after'].__setitem__('evacuated_source_volume_mm3', 1.),
    lambda r: r['after'].__setitem__('evacuated_target_source_bounds_mm3', [.125, .125]),
    lambda r: r['after'].__setitem__('evacuated_target_source_bounds_mm3', [float('nan'), .25]),
    lambda r: r['after'].__setitem__('certified_leaf_count', True),
    lambda r: r['after'].__setitem__('state_digest', 'forged'),
    lambda r: r.__setitem__('delta_certified_leaf_count', 8),
    lambda r: r.__setitem__('delta_evacuated_source_volume_mm3', 1.),
    lambda r: r.__setitem__('delta_evacuated_source_volume_mm3', True),
    lambda r: r.__setitem__('delta_evacuated_target_source_bounds_mm3', [0., .25]),
    lambda r: r['before'].__setitem__('whole_parent_clearance_count', 1),
])
def test_forged_totals_uniform_target_allocation_and_delta_credit_are_rejected(mutate):
    c, material, _, _ = setup()
    with pytest.raises(ValueError):
        half_action(c, material, mutate=mutate)


@pytest.mark.parametrize('bad', ['missing', 'hash', 'outside', 'unordered', 'nan', 'shape'])
def test_bounds_are_explicit_source_bound_and_copied(bad):
    c, tissue, lower, upper = setup()
    spec = declaration(lower, upper)
    if bad == 'missing': lower = None
    if bad == 'hash': lower[KEY] = .1
    if bad == 'outside':
        lower[0, 0, 0] = .1; upper[0, 0, 0] = .1; spec = declaration(lower, upper)
    if bad == 'unordered':
        lower[KEY] = .5; spec = declaration(lower, upper)
    if bad == 'nan':
        upper[KEY] = np.nan; spec = declaration(lower, upper)
    if bad == 'shape': spec['shape'] = [True, 3, 3]
    with pytest.raises(ValueError):
        IndependentCutCoverage(SPEC, c.matrix, c.shape, evacuated_volume_spec=spec,
            target_fraction_lower=lower, target_fraction_upper=upper, tissue_mask=tissue)
    assert c.evacuation.lower[KEY] == .25


def test_reflected_anisotropic_affine_has_physical_volume_and_bound_union():
    matrix = np.eye(4); matrix[:3, :3] = [[0., -2., 0.], [1., 0., 0.], [0., 0., -3.]]
    matrix[:3, 3] = [7., -8., 9.]
    c, tissue, _, _ = setup(.25, .5, matrix)
    proof = {"before_digest": ledger_digest({}), "after_digest": ledger_digest({}),
        "new_leaf_count": 8, "leaf_tests": 0, "cells": [
            {"index": list(KEY), "added_bits_hex": 'ff', "proof": 'whole_cell_capsule'}]}
    # The actual independent corner test runs in the reflected affine. A sphere
    # of radius 2 contains this 1x2x3 parent (corner distance sqrt(3.5)).
    center = matrix[:3, :3] @ KEY + matrix[:3, 3]
    c.step(proof, center, center, 2., 'aspirate', tissue, ~tissue, {KEY})
    result = c.evacuation.snapshot()
    assert result['evacuated_source_volume_mm3'] == 6.
    assert result['evacuated_target_source_bounds_mm3'] == [1.5, 3.]


@pytest.fixture(scope='module')
def executed_history():
    # The producer supplies a history, never the expected independent answers.
    from resectionlab.geometry import AccessWindow, ToolGeometry
    from resectionlab.native_cut_union import CutUnionSpec, EvacuatedVolumeSpec
    from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
    tissue = np.zeros((5, 5, 7), bool); tissue[2, 2, 3] = True
    bounds = EvacuatedVolumeSpec(tissue.astype(float)*.25, tissue.astype(float)*.25,
                                'generated parent target mass')
    cfg = NativeResectionConfig(tissue, tissue.astype(np.int16), np.eye(4),
        AccessWindow([2., 2., 1.5], [0., 0., 1.], 3.),
        (ToolGeometry('cut', .65, .05, 5., 30., 2.),), 'generated-independent-evac',
        'generated support', cut_union=CutUnionSpec(max_depth=1), evacuated_volume=bounds)
    engine = NativeResectionEngine(cfg)
    for x in (1.65, 2.35):
        assert engine.execute_stroke('cut', [x, 2., 3.6], entry_mm=[x, 2., 1.5]).feasible
    class NoPrivateTarget(SimpleNamespace):
        @property
        def reference_target(self):
            raise AssertionError('Independent checker must not inspect hidden target geometry')
    case = NoPrivateTarget(mri=np.zeros(tissue.shape), affine=cfg.affine,
                           frame='RAS+', semantic_hash=cfg.source_hash)
    return cfg, case, copy.deepcopy(engine.history)


def audit(cfg, case, history, **override):
    args = dict(tissue_mask=cfg.tissue_mask, access=cfg.access,
        cut_union_spec=cfg.cut_union.to_record(), evacuated_volume_spec=cfg.evacuated_volume.to_record(),
        evacuated_target_fraction_lower=cfg.evacuated_volume.target_fraction_lower,
        evacuated_target_fraction_upper=cfg.evacuated_volume.target_fraction_upper)
    args.update(override)
    return independent_check_native_history(case, cfg.tools, history, **args)


def test_full_native_history_exports_only_independently_checked_immutable_union(executed_history):
    cfg, case, history = executed_history
    first = audit(cfg, case, history[:1]); final = audit(cfg, case, history)
    assert first.feasible and final.feasible
    assert first.claimed_source_tissue_volume_mm3 == 0
    assert final.claimed_source_tissue_volume_mm3 == 1
    assert first.to_dict()['evacuated_volume']['evacuated_source_volume_mm3'] == .5
    assert first.to_dict()['evacuated_volume']['evacuated_target_source_bounds_mm3'] == [0., .25]
    assert final.to_dict()['evacuated_volume']['evacuated_target_source_bounds_mm3'] == [.25, .25]
    geometry = final.certified_evacuated_geometry
    assert geometry['whole_removed_indices'] == ((2, 2, 3),)
    assert geometry['partial_cells'] == ()
    with pytest.raises(TypeError): geometry['state_digest'] = 'forged'
    with pytest.raises(TypeError): geometry['whole_removed_indices'][0][0] = 0
    wire = thaw_json(geometry); seal = wire.pop('geometry_digest')
    assert seal == semantic_digest(wire) == final.to_dict()['evacuated_volume']['geometry_digest']
    assert 'whole_removed_indices' not in str(final.to_dict())
    assert audit(cfg, case, []).to_dict()['evacuated_volume']['certified_leaf_count'] == 0


@pytest.mark.parametrize('tamper', ['target', 'delta', 'missing', 'geometry', 'continuity'])
def test_failed_native_history_never_exports_certified_partial_geometry(executed_history, tamper):
    cfg, case, saved = executed_history
    history = copy.deepcopy(saved)
    if tamper == 'target': history[0]['evacuated_volume']['after']['evacuated_target_source_bounds_mm3'] = [.125, .125]
    if tamper == 'delta': history[1]['evacuated_volume']['delta_certified_leaf_count'] = 8
    if tamper == 'missing': del history[0]['evacuated_volume']
    if tamper == 'geometry':
        row = next(m for m in history[0]['microsteps'] if m['cut_union']['cells'])
        row['cut_union']['cells'][0]['added_bits_hex'] = 'ff'
    if tamper == 'continuity': history = history[1:]
    result = audit(cfg, case, history)
    assert not result.feasible
    assert result.certified_evacuated_geometry is None
    assert 'evacuated_volume' not in result.to_dict()


def test_default_and_v1_wire_remain_unchanged_and_v2_requires_declaration(executed_history):
    cfg, case, history = executed_history
    with pytest.raises(ValueError): audit(cfg, case, history, evacuated_target_fraction_lower=None)
    with pytest.raises(ValueError): audit(cfg, case, history, cut_union_spec=None)
    with pytest.raises(ValueError): audit(cfg, case, history, evacuated_volume_spec=None)
    undeclared = independent_check_native_history(case, cfg.tools, history,
        tissue_mask=cfg.tissue_mask, access=cfg.access, cut_union_spec=cfg.cut_union.to_record())
    assert not undeclared.feasible and undeclared.failures == ('unsupported_native_footprint_model',)
    v1 = copy.deepcopy(history)
    for row in v1:
        del row['evacuated_volume']
        row['native_footprint'] = MODE
    legacy = independent_check_native_history(case, cfg.tools, v1,
        tissue_mask=cfg.tissue_mask, access=cfg.access, cut_union_spec=cfg.cut_union.to_record())
    default = independent_check_native_history(case, cfg.tools, [], tissue_mask=cfg.tissue_mask, access=cfg.access)
    for result in (legacy, default):
        assert result.feasible and result.certified_evacuated_geometry is None
        assert result.to_dict() == asdict(result)
        assert 'evacuated_volume' not in result.to_dict()
