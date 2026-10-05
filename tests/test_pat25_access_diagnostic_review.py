"""Independent analytical controls; no patient arrays or execution."""
from pathlib import Path
import copy
import gc
import json
import sys
import types
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnose_pat25_access as diagnostic
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import make_native_opening_task


def test_preview_limit_refuses_next_body_before_it_executes(monkeypatch):
    task = make_native_opening_task()
    reference = next(iter(task._inventory.values()))
    original = NativeResectionEngine.preview_stroke
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NativeResectionEngine, 'preview_stroke', counted)
    monkeypatch.setattr(diagnostic, 'SETTINGS', {**diagnostic.SETTINGS, 'max_declared_slots_per_exit': 1})
    with diagnostic.PreviewTrace() as trace:
        task._engine.preview_stroke(reference.tool_id, reference.tip_mm, entry_mm=reference.entry_mm)
        with pytest.raises(RuntimeError, match='preview budget'):
            task._engine.preview_stroke(reference.tool_id, reference.tip_mm, entry_mm=reference.entry_mm)
    assert len(calls) == 1, 'The refused preview body must never execute'
    assert len(trace.rows) == 1
    assert sys.getprofile() is None


def test_trace_copies_result_diagnostics_without_retaining_engine_frames_or_arrays():
    task = make_native_opening_task()
    reference = next(iter(task._inventory.values()))
    before = json.dumps(reference.to_history_record(), sort_keys=True)
    engine_reference = weakref.ref(task._engine)
    state_hash = task._engine.state_hash
    with diagnostic.PreviewTrace() as trace:
        returned = task._engine.preview_stroke(reference.tool_id, reference.tip_mm, entry_mm=reference.entry_mm)
    assert json.dumps(returned.to_history_record(), sort_keys=True) == before
    assert task._engine.state_hash == state_hash and not task._engine.history

    def check_plain(value):
        assert not isinstance(value, (types.FrameType, types.TracebackType, np.ndarray, NativeResectionEngine))
        if isinstance(value, dict):
            for item in value.values():
                check_plain(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_plain(item)

    check_plain(trace.rows)
    del task, reference, returned
    gc.collect()
    assert engine_reference() is None
    assert sys.getprofile() is None


def test_observer_exception_restores_profile_and_does_not_commit():
    task = make_native_opening_task()
    reference = next(iter(task._inventory.values()))
    initial = task._engine.state_hash

    def stop():
        raise TimeoutError('analytical observer cancellation')

    with pytest.raises(TimeoutError, match='observer cancellation'):
        with diagnostic.PreviewTrace(stop) as trace:
            task._engine.preview_stroke(reference.tool_id, reference.tip_mm, entry_mm=reference.entry_mm)
    assert len(trace.rows) == 1 and trace.preview_calls == 1
    assert sys.getprofile() is None
    assert task._engine.state_hash == initial and task._engine.revision == 0
    assert not task._engine.removed_mask.any() and not task._engine.contact_mask.any()


def test_existing_profile_is_neither_replaced_nor_cleared():
    def existing(frame, event, result):
        return None

    sys.setprofile(existing)
    try:
        with pytest.raises(RuntimeError, match='exclusive interpreter trace'):
            with diagnostic.PreviewTrace():
                pytest.fail('Must refuse before context entry')
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(None)


def test_reflected_oblique_exit_coordinates_keep_source_walk_and_inputs():
    support = np.zeros((5, 5, 5), dtype=bool)
    support[1:4, 1:4, 1:4] = True
    target = np.zeros_like(support)
    target[2, 2, 2] = True
    affine = np.array([[0., -2., 0., 11.], [-1., 0., 0., 13.], [0., 0., 3., 17.], [0., 0., 0., 1.]])
    access, derivation = diagnostic.prep.derive_access(target, support, affine, 'sub-PAT25')
    saved_access, saved_derivation = copy.deepcopy(access), copy.deepcopy(derivation)
    rows = diagnostic.six_accesses(derivation, affine, access)
    assert access == saved_access and derivation == saved_derivation
    assert len(rows) == 6 and rows[0]['selected_original']
    assert rows[0]['access'] == access
    assert rows[0]['access']['center_mm'] == [7., 12.5, 23.]
    assert rows[0]['outside_neighbor_voxel'] == [0, 2, 2]
    center = affine[:3, :3] @ np.array([2., 2., 2.]) + affine[:3, 3]
    for row in rows:
        boundary = affine[:3, :3] @ row['boundary_voxel'] + affine[:3, 3]
        np.testing.assert_array_equal(row['access']['center_mm'], boundary)
        assert np.dot(center - boundary, row['access']['normal_inward']) == row['distance_mm']
