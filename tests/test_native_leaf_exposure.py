"""Same-capsule PARENT exposure closure; no subcell channel or shaft relaxation."""
from dataclasses import replace
from pathlib import Path
from types import ModuleType
import hashlib
import subprocess
import sys

import numpy as np
import pytest

from resectionlab.core import thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry, point_segment_distances
from resectionlab.native_cut_union import (CutUnionSpec, EvacuatedVolumeSpec, PreviewCutCoverage,
    MODE, PARENT_CLOSURE_EXPOSURE, PARENT_CLOSURE_MODE, PARENT_CLOSURE_EVACUATED_FOOTPRINT)
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine


def config(enabled=True, *, leaf_cap=4194304):
    tissue = np.zeros((7, 7, 8), bool); tissue[:, :, 3:] = True
    spec = CutUnionSpec(max_leaf_tests_per_preview=leaf_cap,
        exposure_mode=PARENT_CLOSURE_EXPOSURE if enabled else None)
    return NativeResectionConfig(tissue, tissue.astype(np.int16), np.eye(4),
        AccessWindow([3, 3, 1.5], [0, 0, 1], 3.),
        (ToolGeometry('generated-parent-closure', 1., .05, 5., 30., 3.),),
        'generated-parent-closure', 'declared generated solid slab', max_tip_step_mm=.5,
        cut_union=spec, evacuated_volume=EvacuatedVolumeSpec(tissue*.5, tissue*.5, 'unlocated generated mass'))


def cut(engine, z=3.1, preview=False):
    call = engine.preview_stroke if preview else engine.execute_stroke
    return call('generated-parent-closure', [3, 3, z], entry_mm=[3, 3, 1.5])


def test_native_same_capsule_new_whole_parent_exposes_partial_neighbor():
    old = NativeResectionEngine(config(False), immutable_state=True)
    new = NativeResectionEngine(config(), immutable_state=True)
    a, b = cut(old), cut(new)
    assert a.feasible and b.feasible
    assert a.removed_indices_native.tolist() == b.removed_indices_native.tolist() == [[3, 3, 3]]
    assert (3, 3, 4) not in old._cut_union_ledger
    assert 0 < new._cut_union_ledger[(3, 3, 4)] < new.config.cut_union.full_mask
    assert new.remaining_mask[3, 3, 4] and not new.connected_free_mask[3, 3, 4]
    assert b.evacuated_volume['after']['evacuated_source_volume_mm3'] > a.evacuated_volume['after']['evacuated_source_volume_mm3']
    assert b.native_footprint == PARENT_CLOSURE_EVACUATED_FOOTPRINT
    assert new.evacuated_volume == b.evacuated_volume['after']
    # New coverage appears in the same executed interval that clears the front
    # parent, not through a future capsule or an additional model transition.
    step = b.to_history_record()['microsteps'][-1]
    assert [3, 3, 3] in step['removed_indices_native']
    assert any(row['index'] == [3, 3, 4] for row in step['cut_union']['cells'])
    assert (3, 3, 5) not in new._cut_union_ledger  # a partial parent is not a channel
    replay = NativeResectionEngine(config(), immutable_state=True)
    cut(replay)
    assert replay.history == new.history and replay.state_hash == new.state_hash


def test_closure_is_order_independent_and_same_capsule_cells_are_unique():
    cfg = config(); free = np.zeros(cfg.tissue_mask.shape, bool); free[:, :, :3] = True
    occupied = np.asarray([[3, 3, 3], [3, 3, 4], [3, 3, 5]], dtype=np.int64)
    results = []
    for order in (occupied, occupied[::-1]):
        coverage = PreviewCutCoverage(cfg.cut_union, {}, cfg.affine)
        record = coverage.cover(order, occupied[:1], free, np.array([3, 3, -1.5]), np.array([3, 3, 3.1]), 1.)
        assert len({tuple(r['index']) for r in record['cells']}) == len(record['cells'])
        assert record['leaf_tests'] <= 2*512
        assert (3, 3, 5) not in coverage.cells
        results.append((record, dict(coverage.cells)))
    assert results[0] == results[1]


def test_partial_neighbor_and_diagonal_free_space_do_not_seed_exposure():
    spec = CutUnionSpec(exposure_mode=PARENT_CLOSURE_EXPOSURE)
    free = np.zeros((7, 7, 7), bool); free[2, 2, 2] = True
    cell = np.asarray([[3, 3, 3]], dtype=np.int64)
    # Even a geometrically full capsule cannot enter a buried component merely
    # because a diagonal empty cell or adjacent prior partial bits exist.
    partial = {(3, 3, 2): 255}
    c = PreviewCutCoverage(spec, partial, np.eye(4))
    record = c.cover(cell, np.empty((0, 3), int), free, np.array([3, 3, 2]), np.array([3, 3, 4]), 2.)
    assert not record['cells'] and c.cells == partial
    assert record['leaf_tests'] == 0


def test_current_capsule_completes_prior_partial_parent_then_exposes_next_parent():
    spec = CutUnionSpec(exposure_mode=PARENT_CLOSURE_EXPOSURE)
    free = np.zeros((7, 7, 7), bool); free[3, 3, 2] = free[4, 3, 2] = True
    start, end = np.array([3, 3, 2]), np.array([3, 3, 4])
    a, b, c = (3, 3, 3), (4, 3, 3), (4, 4, 3)
    geometry = PreviewCutCoverage(spec, {}, np.eye(4))
    corners = (geometry._leaf_centers+np.array(b))[:, None, :]+geometry._corners
    contained = np.all(point_segment_distances(corners, start, end) <= 1.-1e-10, axis=1)
    new = sum(1 << int(i) for i in np.flatnonzero(contained))
    prior = spec.full_mask ^ new
    assert 0 < new < spec.full_mask and 0 < prior < spec.full_mask
    coverage = PreviewCutCoverage(spec, {b: prior}, np.eye(4))
    occupied = np.asarray([a, b, c], dtype=np.int64)
    record = coverage.cover(occupied, np.asarray([a]), free, start, end, 1.)
    assert coverage.cells[b] == spec.full_mask
    assert 0 < coverage.cells[c] < spec.full_mask
    assert {tuple(row) for row in coverage.complete(occupied)} == {a, b}
    final = coverage.finish_step(record, np.asarray([a, b]))
    assert set(coverage.cells) == {c} and len(final['cells']) == 3


def test_prior_whole_parent_shaft_still_refuses_and_preview_is_atomic():
    cfg = config()
    cfg = replace(cfg, tools=(replace(cfg.tools[0], shaft_radius_mm=.8),))
    engine = NativeResectionEngine(cfg, immutable_state=True)
    initial = engine.state_hash; empty = engine.evacuated_volume
    preview = cut(engine, preview=True)
    assert preview.feasible and engine.state_hash == initial and engine.evacuated_volume == empty
    clone = engine.clone(); clone.commit_preview(preview)
    assert engine.state_hash == initial and clone.evacuated_volume == preview.evacuated_volume['after']
    engine.commit_preview(preview); before = engine.state_hash; volume = engine.evacuated_volume
    rejected = cut(engine, 6.)
    assert not rejected.feasible and rejected.reason == 'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
    assert engine.state_hash == before and engine.evacuated_volume == volume
    clone.reset(); assert clone.state_hash == initial and clone.evacuated_volume == empty
    with pytest.raises(ValueError): engine.commit_preview(preview)


def test_shared_leaf_budget_refuses_after_temporary_partial_cut_without_commit():
    complete = cut(NativeResectionEngine(config()), preview=True)
    assert complete.feasible
    budget = 0
    for index, step in enumerate(complete.microsteps):
        proof = step.to_dict()['cut_union']
        budget += proof['leaf_tests']
        if proof['new_leaf_count']:
            break
    assert index < len(complete.microsteps)-1 and 0 < budget < complete.cut_union_attempt['leaf_tests']
    engine = NativeResectionEngine(config(leaf_cap=budget), immutable_state=True)
    state, volume = engine.state_hash, engine.evacuated_volume
    result = cut(engine)
    assert not result.feasible and result.reason == 'CUMULATIVE_CUT_COVERAGE_BUDGET'
    assert result.cut_union_attempt['new_leaf_count'] > 0
    assert engine.state_hash == state and engine.evacuated_volume == volume and not engine.history
    assert not engine._cut_union_ledger and not engine.removed_mask.any()


def test_explicit_mode_declaration_does_not_modify_legacy_serialization():
    assert CutUnionSpec().to_record() == {'mode': MODE, 'max_depth': 3,
        'max_tracked_cells': 4096, 'max_leaf_tests_per_preview': 4194304}
    declared = CutUnionSpec(exposure_mode=PARENT_CLOSURE_EXPOSURE).to_record()
    assert declared['mode'] == PARENT_CLOSURE_MODE and declared['exposure_mode'] == PARENT_CLOSURE_EXPOSURE
    assert config().fingerprint != config(False).fingerprint
    for invalid in (True, '', 'leaf_connectivity', 1):
        with pytest.raises(ValueError): CutUnionSpec(exposure_mode=invalid)


@pytest.mark.parametrize('immutable', [False, True])
def test_disabled_mode_matches_committed_partial_volume_predecessor(immutable):
    root = Path(__file__).resolve().parents[1]
    revision = 'b16ffece399a9bd86851e8661cb1a6f05a85ec20'
    source = subprocess.run(['git', 'show', revision+':src/resectionlab/native_resection.py'],
        cwd=root, check=True, capture_output=True).stdout
    assert hashlib.sha256(source).hexdigest() == 'ddbb5571e401a5da8aa3f68a878f7c97d24342a47d2180441374ad8ebade9e9b'
    old = ModuleType('resectionlab._before_parent_closure'); old.__package__ = 'resectionlab'
    sys.modules[old.__name__] = old
    exec(compile(source, '<committed partial-volume engine>', 'exec'), old.__dict__)
    cfg = config(False)
    previous = old.NativeResectionConfig(cfg.tissue_mask, cfg.target_labels, cfg.affine,
        cfg.access, cfg.tools, cfg.source_hash, cfg.tissue_support_provenance,
        max_tip_step_mm=cfg.max_tip_step_mm, cut_union=cfg.cut_union, evacuated_volume=cfg.evacuated_volume)
    assert cfg.fingerprint == previous.fingerprint
    a, b = NativeResectionEngine(cfg, immutable_state=immutable), old.NativeResectionEngine(previous, immutable_state=immutable)
    for z in (3.1, 3.35, 7.):
        aa, bb = cut(a, z), cut(b, z)
        assert aa.feasible == bb.feasible and aa.reason == bb.reason
        assert aa.cut_union_attempt == bb.cut_union_attempt
        assert a.history == b.history and a.metrics() == b.metrics() and a.state_hash == b.state_hash
