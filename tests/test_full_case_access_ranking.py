"""Residual route bookkeeping on explicit engineering source grids only."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.full_case_geometry import access_candidates, rank_remaining_accesses
from resectionlab.surgical_scene import SurgicalScene


def _engine(labels, affine=None):
    scene = SurgicalScene(labels, np.eye(4) if affine is None else affine, [
        {'id': 1, 'material': 'soft_tissue'},
        {'id': 7, 'material': 'tumor', 'target': True},
        {'id': 8, 'material': 'vessel'},
        {'id': 9, 'material': 'bone'},
        {'id': 42, 'material': 'tumor'},
    ], {'kind': 'explicit-route-ranking-fixture'}, {})
    return SimpleNamespace(scene=scene, state={'removed': [], 'deformation': {}, 'holds': {}},
                           state_id='fixture-state-0')


def _access(engine, name, index, radius=1., *, indexed=True):
    point = engine.scene.affine[:3, :3] @ index + engine.scene.affine[:3, 3]
    row = {'id': name, 'entryPose': {'position': point.tolist(), 'quaternion': [1., 0., 0., 0.]},
           'outsidePose': {'position': [*point[:2], 50.], 'quaternion': [1., 0., 0., 0.]},
           'drillRadiusMm': radius, 'customEvidence': {'retain': [1, 2]}}
    if indexed:
        row['targetEntryIndex'] = list(index)
    return row


def test_rank_preserves_prefix_ties_geometry_inputs_and_source():
    labels = np.zeros((16, 16, 10), dtype=np.uint16)
    labels[3, 3, 1:7] = 7
    labels[3, 3, 4] = 1  # Only two original target cells above this barrier.
    labels[10, 3, 1:5] = labels[10, 10, 1:5] = 7
    labels[10, 3, 6] = 9  # Bone is allowed in a proposed source column.
    labels[4, 3, 8] = 42  # Non-target tumor ownership must never count.
    engine = _engine(labels)
    engine.state['removed'] = [np.ravel_multi_index((3, 3, 4), labels.shape),
                               np.ravel_multi_index((3, 3, 6), labels.shape)]
    rows = [_access(engine, name, index) for name, index in [
        ('visited', [0, 0, 0]), ('low', [3, 3, 6]),
        ('tie-first', [10, 10, 4]), ('tie-second', [10, 3, 4])]]
    original_rows, original_state = copy.deepcopy(rows), copy.deepcopy(engine.state)
    source = engine.scene.labels.copy()
    result = rank_remaining_accesses(engine, rows, start_index=1)
    assert [row['id'] for row in result] == ['visited', 'tie-first', 'tie-second', 'low']
    assert result[0] == rows[0] and 'capacityStateId' not in result[0]
    assert [row['remainingGeometricCapacityCells'] for row in result[1:]] == [4, 4, 1]
    assert all(row['capacityStateId'] == engine.state_id for row in result[1:])
    by_id = {row['id']: row for row in rows}
    for row in result[1:]:
        assert {k: v for k, v in row.items() if k not in
                ('capacityStateId', 'remainingGeometricCapacityCells')} == by_id[row['id']]
    result[0]['customEvidence']['retain'].append(3)
    assert rows == original_rows and engine.state == original_state
    np.testing.assert_array_equal(engine.scene.labels, source)


def test_rerank_tracks_new_removed_identities_and_updates_only_tail_annotations():
    labels = np.zeros((10, 10, 10), dtype=np.uint16)
    labels[2, 2, 1:5] = 7
    labels[7, 7, 1:4] = 7
    engine = _engine(labels)
    rows = [_access(engine, 'initial-best', [2, 2, 4]), _access(engine, 'retained', [7, 7, 3])]
    initial = rank_remaining_accesses(engine, rows)
    engine.state['removed'] = np.ravel_multi_index(([2] * 4, [2] * 4, [1, 2, 3, 4]), labels.shape).tolist()
    engine.state_id = 'fixture-state-1'
    updated = rank_remaining_accesses(engine, initial)
    assert [row['id'] for row in updated] == ['retained', 'initial-best']
    assert [row['remainingGeometricCapacityCells'] for row in updated] == [3, 0]
    assert all(row['capacityStateId'] == 'fixture-state-1' for row in updated)
    assert all(row['capacityStateId'] == 'fixture-state-0' for row in initial)


def test_manual_world_pose_affine_spacing_and_outside_zero_capacity():
    labels = np.zeros((12, 12, 10), dtype=np.uint16)
    labels[5, 5, 2] = labels[6, 5, 2] = labels[5, 6, 2] = 7
    labels[5, 7, 2] = 7
    affine = np.diag([-2., .5, 1., 1.]); affine[:3, 3] = [30., -10., -50.]
    engine = _engine(labels, affine)
    rows = [_access(engine, 'outside', [-1, 5, 2], indexed=False),
            _access(engine, 'empty', [0, 0, 2], indexed=False),
            _access(engine, 'manual', [5, 5, 2], indexed=False),
            _access(engine, 'indexed', [5, 5, 2])]
    result = rank_remaining_accesses(engine, rows)
    assert [row['id'] for row in result] == ['manual', 'indexed', 'outside', 'empty']
    # The one-millimeter disk includes two Y neighbors, excludes the 2mm X neighbor.
    assert [row['remainingGeometricCapacityCells'] for row in result] == [3, 3, 0, 0]
    assert 'targetEntryIndex' not in result[0]


def test_intact_ranking_matches_original_candidate_capacity_convention():
    labels = np.zeros((24, 24, 12), dtype=np.uint16)
    labels[3:21, 3:21, 2:7] = 7
    labels[7:12, 7:12, 4] = 8
    labels[3:21, 3:21, 9] = 9
    engine = _engine(labels)
    rows = access_candidates(engine, limit=12, max_radius_mm=3.)
    assert len(rows) > 1
    result = rank_remaining_accesses(engine, rows)
    assert all(row['remainingGeometricCapacityCells'] == row['geometricTargetCapacityCells']
               for row in result)


def test_empty_or_fully_visited_and_invalid_boundaries():
    engine = _engine(np.zeros((4, 4, 4), dtype=np.uint16))
    assert rank_remaining_accesses(engine, []) == []
    rows = [_access(engine, 'visited', [1, 1, 1])]
    assert rank_remaining_accesses(engine, rows, 1) == rows
    assert rank_remaining_accesses(engine, rows)[0]['remainingGeometricCapacityCells'] == 0
    for boundary in (-1, 2):
        with pytest.raises(ValueError, match='start_index'):
            rank_remaining_accesses(engine, rows, boundary)
