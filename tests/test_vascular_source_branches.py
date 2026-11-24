"""Stable native branch identities and separate source-imaged gap proposals."""

import numpy as np
import pytest

from resectionlab import vascular_source_branches as source


def test_source_loop_survives_and_ids_follow_source_coordinates():
    mask = np.zeros((24, 24, 16), bool)
    mask[4:19, 4, 7] = mask[4:19, 18, 7] = True
    mask[4, 4:19, 7] = mask[18, 4:19, 7] = True
    frame = np.array([
        [0, -1.4, 0, 13], [0.7, 0, 0, -8], [0, 0, -1.2, 32], [0, 0, 0, 1],
    ])
    graph = source.build_source_branches(mask, frame, source_scan_sha256="a" * 64)
    assert graph == source.build_source_branches(
        np.asfortranarray(mask), frame, source_scan_sha256="A" * 64,
    )
    assert graph["coverage"]["status"] == "complete"
    assert graph["components"][0]["source_topology"]["cycle_rank26"] == 1
    branch, = graph["branches"]
    ids = branch["source_cell_ids"]
    assert branch["closed"] and ids[0] == ids[-1] == min(ids)
    assert mask.ravel()[ids].all()
    coordinates = np.column_stack(np.unravel_index(ids, mask.shape))
    np.testing.assert_allclose(branch["points_ras_mm"], coordinates @ frame[:3, :3].T + frame[:3, 3])
    changed, = source.build_source_branches(mask, frame, source_scan_sha256="b" * 64)["branches"]
    assert changed["source_cell_ids"] == ids and changed["id"] != branch["id"]


def test_biconnected_homology_reduction_keeps_all_bridge_edges(monkeypatch):
    points = np.array([
        [1, 1, 1], [1, 2, 1], [2, 1, 1], [3, 1, 1],
        [4, 1, 1], [5, 1, 1], [6, 1, 1], [6, 2, 1],
    ])
    skeleton = np.zeros((8, 4, 3), bool)
    skeleton[tuple(points.T)] = True
    raw = source._sparse_edges(skeleton, points)
    expected, _, _, _ = source._bounded_representative(points, raw, 1)
    monkeypatch.setattr(source, "_MAX_REPRESENTATIVE_CHORDS", 1)
    edges, components, cycles, scope = source._bounded_representative(points, raw, 1)
    assert scope == "biconnected-blocks" and components == 1 and cycles == 0
    np.testing.assert_array_equal(edges, expected)
    assert {(2, 3), (3, 4), (4, 5)} <= set(map(tuple, edges))


def test_a_source_cavity_is_reported_separately_from_matched_h0_and_h1():
    mask = np.zeros((12, 12, 12), bool)
    mask[3:8, 3:8, 3:8] = True
    mask[5, 5, 5] = False
    graph = source.build_source_branches(mask, np.eye(4), source_scan_sha256="6" * 64)
    component, = graph["components"]
    assert component["source_topology"] == component["skeleton_topology"]
    assert component["source_topology"]["cavities6"] == 1
    assert component["representative_cycle_rank"] == 0
    assert component["status"] == "source-h0-h1-matched-cavities-unrepresented"
    assert graph["coverage"]["components_without_source_h0_h1_match"] == 0
    assert graph["coverage"]["components_with_unrepresented_cavities6"] == 1
    assert graph["coverage"]["status"] == "partial"


def test_geometry_compatible_gap_requires_independent_tof_evidence():
    prediction = np.zeros((32, 25, 25), bool)
    prediction[3:14, 12, 12] = prediction[16:28, 12, 12] = True
    signal = np.zeros(prediction.shape, np.float32)
    signal[3:28, 12, 12] = 10
    frame = np.diag([0.8, 0.8, 0.8, 1.0])
    graph = source.build_source_branches(
        prediction, frame, source_scan_sha256="7" * 64, signal=signal,
    )
    original = prediction.copy()
    supported = source.evaluate_source_gaps(graph, prediction, signal)
    weak = source.evaluate_source_gaps(graph, prediction, signal * prediction)
    bright, = supported["candidates"]
    dark, = weak["candidates"]
    assert bright["id"] == dark["id"]
    assert bright["state"] == "uncertain-image-supported"
    assert dark["state"] == "weak-source-image-support"
    assert bright["points_ras_mm"] == dark["points_ras_mm"]
    assert bright["supported_gap_station_count"] == bright["gap_station_count"]
    assert supported["predicted_voxels_added"] == weak["predicted_voxels_added"] == 0
    np.testing.assert_array_equal(prediction, original)
    changed = np.ones_like(prediction)
    with pytest.raises(ValueError, match="source prediction"):
        source.evaluate_source_gaps(graph, changed, signal)


def test_workspace_abstention_records_unrepresented_source_cells(monkeypatch):
    mask = np.ones((4, 5, 6), bool)
    monkeypatch.setattr(source, "_MAX_COMPONENT_CELLS", 10)
    graph = source.build_source_branches(mask, np.eye(4), source_scan_sha256="3" * 64)
    assert graph["coverage"]["status"] == "partial"
    assert graph["coverage"]["unrepresented_foreground_voxels"] == mask.size
    assert graph["components"][0]["status"] == "component-workspace-abstained"
    assert not graph["branches"]
