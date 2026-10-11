"""Generated producer checks for explicitly declared partial evacuation readout.

The target arrays declare unlocated mass in each parent, not subcell labels.
No analytic physical target or patient source enters these tests.
"""
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib
import subprocess
import sys

import numpy as np
import pytest

from resectionlab.core import freeze_json, thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_cut_union import (CutUnionSpec, EvacuatedVolumeSpec,
    EVACUATED_FOOTPRINT, evacuated_volume_snapshot)
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask, DEFAULT_NATIVE_SPATIAL_REWARD


def config(*, lower=.5, upper=None, enabled=True, union=True, affine=None, cut_spec=None):
    tissue = np.zeros((5, 5, 7), bool); tissue[2, 2, 3] = True
    matrix = np.eye(4) if affine is None else affine
    point = lambda p: matrix[:3, :3] @ p + matrix[:3, 3]
    axis = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    bounds = EvacuatedVolumeSpec(tissue.astype(float)*lower,
        tissue.astype(float)*(lower if upper is None else upper), 'generated parent mass only')
    return NativeResectionConfig(tissue, tissue.astype(np.int16), matrix,
        AccessWindow(point([2, 2, 1.5]), axis, 3.),
        (ToolGeometry('generated-cut', .65, .05, 5., 30., 2.),),
        'generated-evacuation', 'one-cell generated represented support',
        cut_union=(cut_spec or CutUnionSpec()) if union else None,
        evacuated_volume=bounds if enabled else None)


def stroke(engine, x=1.65, z=3.6, *, mode='aspirate', preview=False):
    matrix = engine.config.affine
    point = lambda p: matrix[:3, :3] @ p + matrix[:3, 3]
    call = engine.preview_stroke if preview else engine.execute_stroke
    return call('generated-cut', point([x, 2, z]), entry_mm=point([x, 2, 1.5]),
                interaction_mode=mode)


@pytest.mark.parametrize('immutable', [False, True])
def test_partial_then_full_volume_is_unique_and_target_is_an_interval(immutable):
    engine = NativeResectionEngine(config(), immutable_state=immutable)
    initial = thaw_json(engine.evacuated_volume)
    assert initial['certified_leaf_count'] == 0
    first = stroke(engine, preview=True)
    assert first.feasible and first.removed_volume_mm3 == 0
    assert thaw_json(engine.evacuated_volume) == initial
    proof = thaw_json(first.evacuated_volume)
    assert proof['before'] == initial
    r = proof['after']['evacuated_source_volume_mm3']
    assert 0 < r < 1
    bounds = proof['after']['evacuated_target_source_bounds_mm3']
    assert bounds == [max(0., r-.5), min(r, .5)] and bounds[0] < bounds[1]
    assert bounds != [.5*r, .5*r]
    engine.commit_preview(first)
    assert thaw_json(engine.evacuated_volume) == proof['after']
    assert engine.remaining_mask[2, 2, 3] and not engine.removed_mask.any()
    second = stroke(engine, 2.35, preview=True)
    p2 = thaw_json(second.evacuated_volume)
    assert second.feasible and second.removed_volume_mm3 == 1
    assert p2['before'] == proof['after']
    assert p2['delta_certified_leaf_count'] == 512-proof['delta_certified_leaf_count']
    assert p2['delta_evacuated_source_volume_mm3'] == 1-r
    engine.commit_preview(second)
    assert thaw_json(engine.evacuated_volume) == p2['after']
    assert p2['after']['whole_parent_clearance_count'] == 1
    assert p2['after']['partial_certified_leaf_count'] == 0
    assert p2['after']['evacuated_target_source_bounds_mm3'] == [.5, .5]
    assert not engine._cut_union_ledger
    assert sum(row['evacuated_volume']['delta_evacuated_source_volume_mm3']
               for row in engine.history) == 1
    assert all(row['native_footprint'] == EVACUATED_FOOTPRINT for row in engine.history)
    assert all('delta_evacuated_target_source_bounds_mm3' not in row['evacuated_volume']
               for row in engine.history)


@pytest.mark.parametrize('lower,upper', [(0., 0.), (1., 1.), (.2, .8)])
def test_binary_and_uncertain_parent_target_mass_is_not_a_point_assignment(lower, upper):
    e = NativeResectionEngine(config(lower=lower, upper=upper))
    first = stroke(e); r = first.evacuated_volume['after']['evacuated_source_volume_mm3']
    assert first.evacuated_volume['after']['evacuated_target_source_bounds_mm3'] == (
        max(0., r-(1-lower)), min(r, upper))
    stroke(e, 2.35)
    assert e.evacuated_volume['evacuated_target_source_bounds_mm3'] == (lower, upper)


def test_fraction_declaration_is_immutable_explicit_validated_and_fingerprinted():
    a = np.zeros((5, 5, 7)); a[2, 2, 3] = .5
    spec = EvacuatedVolumeSpec(a, a, 'declared')
    seal = spec.to_record(); a[2, 2, 3] = 0
    assert spec.to_record() == seal
    with pytest.raises(ValueError): spec.target_fraction_lower.setflags(write=True)
    with pytest.raises(ValueError): spec.target_fraction_upper[2, 2, 3] = 0
    cfg = config()
    assert cfg.fingerprint != replace(cfg, evacuated_volume=None).fingerprint
    assert cfg.fingerprint != config(lower=.25).fingerprint
    for bad in (np.ones((2, 2)), np.full((2, 2, 2), np.nan),
                np.full((2, 2, 2), 1.1), np.full((2, 2, 2), -.1)):
        with pytest.raises(ValueError): EvacuatedVolumeSpec(bad, bad, 'declared')
    with pytest.raises(ValueError): EvacuatedVolumeSpec(np.ones((2, 2, 2)), np.zeros((2, 2, 2)), 'declared')
    with pytest.raises(ValueError): EvacuatedVolumeSpec(a, a, ' ')
    with pytest.raises(ValueError): EvacuatedVolumeSpec(a, np.zeros((2, 2, 2)), 'declared')
    with pytest.raises(TypeError): replace(cfg, evacuated_volume={})
    with pytest.raises(TypeError): replace(cfg, cut_union=None)
    outside = np.zeros(cfg.tissue_mask.shape); outside[0, 0, 0] = 1
    with pytest.raises(ValueError): replace(cfg, evacuated_volume=EvacuatedVolumeSpec(outside, outside, 'bad support'))
    small = np.zeros((2, 2, 2))
    with pytest.raises(ValueError): replace(cfg, evacuated_volume=EvacuatedVolumeSpec(small, small, 'wrong shape'))


def test_probe_clone_reset_and_forged_preview_cannot_change_volume():
    e = NativeResectionEngine(config(), immutable_state=True)
    initial = e.state_hash; initial_volume = e.evacuated_volume
    first = stroke(e, preview=True); clone = e.clone()
    clone.commit_preview(first)
    assert e.evacuated_volume == initial_volume and clone.evacuated_volume == first.evacuated_volume['after']
    e.commit_preview(first); before = e.evacuated_volume
    probe = stroke(e, 2., 1.85, mode='probe')
    assert probe.feasible and probe.evacuated_volume['delta_certified_leaf_count'] == 0
    assert probe.evacuated_volume['delta_evacuated_source_volume_mm3'] == 0
    assert e.evacuated_volume == before == probe.evacuated_volume['before'] == probe.evacuated_volume['after']
    child = e.clone(); second = stroke(child, 2.35, preview=True)
    forged = thaw_json(second.evacuated_volume)
    forged['after']['evacuated_source_volume_mm3'] += 1
    object.__setattr__(second, '_evacuated_volume_record', freeze_json(forged))
    state = child.state_hash
    with pytest.raises(ValueError, match='changed'): child.commit_preview(second)
    assert child.state_hash == state and child.evacuated_volume == before
    with pytest.raises(ValueError): e.commit_preview(first)
    clone.reset()
    assert clone.state_hash == initial and clone.evacuated_volume == initial_volume
    assert e.evacuated_volume == before
    replay = NativeResectionEngine(config(), immutable_state=True)
    stroke(replay); stroke(replay, 2., 1.85, mode='probe')
    assert replay.history == e.history and replay.evacuated_volume == e.evacuated_volume


@pytest.mark.parametrize('immutable', [False, True])
def test_late_failure_and_leaf_budget_refusal_roll_back_partial_volume(immutable):
    for cfg, expected in ((config(), 'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'),
            (config(cut_spec=CutUnionSpec(max_leaf_tests_per_preview=1)), 'CUMULATIVE_CUT_COVERAGE_BUDGET')):
        e = NativeResectionEngine(cfg, immutable_state=immutable)
        before = e.state_hash; volume = e.evacuated_volume
        rejected = stroke(e, z=5. if expected.startswith('SHAFT') else 3.6)
        assert not rejected.feasible and rejected.reason == expected
        if expected.startswith('SHAFT'):
            assert rejected.cut_union_attempt['new_leaf_count'] > 0
            assert any(m.to_dict()['cut_union']['new_leaf_count'] for m in rejected.microsteps)
        assert rejected.evacuated_volume is None
        assert e.state_hash == before and e.evacuated_volume == volume
        assert not e.history and not e._cut_union_ledger and not e.removed_mask.any()


def test_v2_reporting_preserves_actual_geometry_whole_cell_reward_and_occupancy():
    old = NativeResectionEngine(config(enabled=False), immutable_state=True)
    new = NativeResectionEngine(config(), immutable_state=True)
    score_task = SimpleNamespace(_config=new.config, _planning=True,
        case=SimpleNamespace(reference_target=new.config.tissue_mask.astype(float)*.25),
        reward_spec=DEFAULT_NATIVE_SPATIAL_REWARD)
    prior = None
    for x in (1.65, 2.35):
        a, b = stroke(old, x), stroke(new, x)
        assert a.feasible == b.feasible and a.reason == b.reason
        ra, rb = a.to_history_record(), b.to_history_record()
        for name in ('microsteps', 'removed_indices_native', 'contact_indices_native', 'cut_union'):
            assert ra[name] == rb[name]
        for name in ('remaining_mask', 'removed_mask', 'connected_free_mask', 'contact_mask'):
            assert np.array_equal(getattr(old, name), getattr(new, name))
        sa = NativeSpatialTask._score_record(score_task, {**ra, 'action_id': 'same'}, prior)
        sb = NativeSpatialTask._score_record(score_task, {**rb, 'action_id': 'same'}, prior)
        for name in ('reward', 'target_removed_mm3', 'normal_removed_mm3', 'complete_tool_path_length_mm'):
            assert sa[name] == sb[name]
        if prior is None: assert sb['target_removed_mm3'] == 0 and sb['reward'] < 0
        prior = a.tool_id


def test_affine_parent_volume_and_invalid_ledger_readout():
    cut = CutUnionSpec(); material = np.zeros((2, 2, 2), bool)
    fractions = np.full(material.shape, .5)
    declaration = EvacuatedVolumeSpec(fractions, fractions, 'generated mass')
    matrix = np.eye(4); matrix[:3, :3] = [[0, -2, 0], [1, 0, .3], [0, 0, -3]]
    partial = evacuated_volume_snapshot(declaration, cut, {(0, 0, 0): 255}, material, matrix)
    assert partial['evacuated_source_volume_mm3'] == pytest.approx(8*6/512)
    assert partial['evacuated_target_source_bounds_mm3'] == pytest.approx([0, 8*6/512])
    with pytest.raises(ValueError): evacuated_volume_snapshot(declaration, cut, {(0, 0, 0): cut.full_mask}, material, matrix)
    material[0, 0, 0] = True
    with pytest.raises(ValueError): evacuated_volume_snapshot(declaration, cut, {(0, 0, 0): 1}, material, matrix)
    with pytest.raises(ValueError): evacuated_volume_snapshot(declaration, cut, {(3, 0, 0): 1}, material, matrix)


@pytest.mark.parametrize('union', [False, True])
@pytest.mark.parametrize('immutable', [False, True])
def test_absent_readout_matches_required_committed_predecessor_exactly(union, immutable):
    # Read a pinned committed source in memory: no ignored build dependency or
    # duplicate baseline file. Missing repository history is an explicit failure.
    root = Path(__file__).resolve().parents[1]
    revision = '9fc4a50d66c562b5df065175686c3df3e6685ccb'
    source = subprocess.run(['git', 'show', revision+':src/resectionlab/native_resection.py'],
        cwd=root, check=True, capture_output=True).stdout
    assert hashlib.sha256(source).hexdigest() == '871f747103358329ad4953f7ad58b8d6952f1860af5832bc5d340c6bba538aee'
    old = ModuleType('resectionlab._pre_evacuated_volume'); old.__package__ = 'resectionlab'
    sys.modules[old.__name__] = old
    exec(compile(source, '<committed pre-evacuation native engine>', 'exec'), old.__dict__)
    cfg = config(enabled=False, union=union)
    legacy = old.NativeResectionConfig(cfg.tissue_mask, cfg.target_labels, cfg.affine,
        cfg.access, cfg.tools, cfg.source_hash, cfg.tissue_support_provenance, cut_union=cfg.cut_union)
    assert cfg.fingerprint == legacy.fingerprint
    a, b = NativeResectionEngine(cfg, immutable_state=immutable), old.NativeResectionEngine(legacy, immutable_state=immutable)
    for x in (1.65, 2.35):
        aa, bb = stroke(a, x), stroke(b, x)
        assert aa.feasible == bb.feasible and aa.reason == bb.reason
        assert aa.cut_union_attempt == bb.cut_union_attempt
        assert a.history == b.history and a.state_hash == b.state_hash and a.metrics() == b.metrics()
        assert a.evacuated_volume is None and aa.evacuated_volume is None
