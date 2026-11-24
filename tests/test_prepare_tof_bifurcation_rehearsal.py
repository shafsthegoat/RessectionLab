import numpy as np
import pytest

from scripts.prepare_tof_bifurcation_rehearsal import graph_fork_view, source_routes


def _native_fork():
    shape = (24, 24, 24)
    junction = np.array([12, 12, 12])
    start = junction - 7
    vessel = np.zeros((15, 15, 15), dtype=bool)
    branches = []
    for axis in range(3):
        points = junction + np.arange(9)[:, None] * np.eye(3, dtype=int)[axis]
        tail = axis == 1
        if tail:
            points = points[::-1]
        covered = points[np.all(points < start + vessel.shape, axis=1)] - start
        vessel[tuple(covered.T)] = True
        flags = [True] * 7 + [False]
        branches.append({
            'id': str(axis),
            'source_cell_ids': np.ravel_multi_index(points.T, shape).tolist(),
            'edge_local_face_supported': flags[::-1] if tail else flags,
            'endpoint_degrees': [1, 3] if tail else [3, 1],
            'closed': False,
        })
    return {'source_shape': shape, 'branches': branches}, vessel, start, junction


def test_graph_fork_uses_local_source_prefixes_and_retains_branch_ids():
    graph, vessel, start, junction = _native_fork()
    paths = source_routes(graph, vessel, start, np.ones(3), junction, source_graph_fork=True)

    assert list(paths) == ['0', '1', '2']
    for axis, path in enumerate(paths.values()):
        assert np.array_equal(path[0], junction - start)
        assert np.array_equal(path[-1] - path[0], 6 * np.eye(3, dtype=int)[axis])
        assert vessel[tuple(path.T)].all()
        assert np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1)


def test_graph_fork_rejects_a_gap_before_the_minimum_source_length():
    graph, vessel, start, junction = _native_fork()
    vessel[tuple(junction - start + [2, 0, 0])] = False

    with pytest.raises(ValueError, match='continuous source-supported prefix'):
        source_routes(graph, vessel, start, np.ones(3), junction, source_graph_fork=True)


def test_graph_fork_microscope_faces_the_exposed_side_on_an_oblique_reflected_grid():
    affine = np.array([[1, 0, 0, 10], [0, 1, 0, -7], [.5, 0, -1, 4], [0, 0, 0, 1]])
    junction = np.array([3, 5, 7])
    state = graph_fork_view(junction, affine)['states']['primary']
    microscope = state['viewer']['microscope']
    outward = np.array(microscope['outward'])

    assert np.allclose(state['cursor'], (affine @ [*junction, 1])[:3])
    assert microscope['target'] == state['cursor']
    assert np.allclose(outward @ affine[:3, :2], [0, 0])
    assert np.isclose(np.linalg.norm(outward), 1)
    assert outward @ affine[:3, 2] > 0
