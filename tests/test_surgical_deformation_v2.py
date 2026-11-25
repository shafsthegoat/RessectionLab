"""Mechanics invariants, independent of instrument reachability."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from scipy import sparse
from resectionlab.surgical_deformation_v2 import begin_hold, set_target, release, tick, profile


def engine():
    shape = (13, 13, 13)
    scene = SimpleNamespace(shape=shape, affine=np.eye(4), inverse=np.eye(4),
        labels=np.ones(shape, dtype=np.uint8), properties={'tissuePhysicsV2': {'brain': profile()}})
    return SimpleNamespace(scene=scene, by_label={1: {'material': 'soft_tissue'}, 2: {'material': 'tumor'},
        3: {'material': 'bone'}, 4: {'material': 'vessel'}},
        state={'removed': [], 'holds': {}, 'cutEdges': [], 'transectedEdges': []})


def cell(e, p): return int(np.ravel_multi_index(p, e.scene.shape))
def displacement(e, c): return np.array(e.state['deformation'].get(str(c), [0., 0., 0.]))


def held():
    e = engine(); c = cell(e, (6, 6, 6))
    begin_hold(e, 'left', [c], [6, 6, 6]); set_target(e, 'left', [6.4, 6, 6])
    return e, c


def test_batched_lattice_retains_scalar_sparse_entry_order(monkeypatch):
    from resectionlab import surgical_deformation_v2 as brain

    e = engine()
    e.scene.affine[:3, :3] = [[.65, -.21, .04], [.3, 1.1, 0.], [0., .13, 1.37]]
    coordinates = [(6, 6, 6), (6, 6, 7), (6, 7, 6), (7, 6, 6)]
    ids = [cell(e, p) for p in coordinates]
    e.scene.labels[coordinates[2]] = 2
    settings = profile()
    original_coo = sparse.coo_matrix
    captured = []

    def capture_entries(entries, *args, **kwargs):
        if len(captured) < 2:
            values, (rows, cols) = entries
            captured.append((np.array(rows), np.array(cols), np.array(values)))
        return original_coo(entries, *args, **kwargs)

    monkeypatch.setattr(brain.sparse, 'coo_matrix', capture_entries)
    _, _, _, edges, _ = brain._system(e, settings, ids, [ids[1]], set(ids), {(ids[0], ids[1])})

    rest = np.array(coordinates) @ e.scene.affine[:3, :3].T
    stiffness = np.array([settings['brainStiffnessNPerMm']] * 4)
    stiffness[2] = settings['tumorStiffnessNPerMm']
    expected_springs, expected_divergence = [], []
    expected_edges = []
    for a, b in [(0, 2), (0, 3)]:
        delta = rest[b] - rest[a]
        length = np.linalg.norm(delta)
        direction = delta / length
        k = 2 * stiffness[a] * stiffness[b] / (stiffness[a] + stiffness[b])
        expected_edges.append((a, b, length))
        for axis in range(3):
            i, j = 3 * a + axis, 3 * b + axis
            expected_springs.extend(((i, i, k), (i, j, -k), (j, i, -k), (j, j, k)))
            for node in (a, b):
                expected_divergence.extend(((node, i, -direction[axis] / (2 * length)),
                                            (node, j, direction[axis] / (2 * length))))
    assert edges == expected_edges
    assert len(captured) == 2
    for (rows, cols, values), expected in zip(captured, (expected_springs, expected_divergence)):
        row, col, value = zip(*expected)
        np.testing.assert_array_equal(rows, row)
        np.testing.assert_array_equal(cols, col)
        np.testing.assert_array_equal(values.view(np.uint64), np.asarray(value).view(np.uint64))


@pytest.mark.parametrize('reverse', [False, True])
def test_lattice_repeated_bonds_keep_exact_matrices_and_solve_on_irregular_masks(monkeypatch, reverse):
    from resectionlab import surgical_deformation_v2 as brain

    e = engine()
    e.scene.affine[:3, :3] = [[.65, -.21, .04], [.3, 1.1, 0.], [0., .13, 1.37]]
    coords = [(i, j, k) for i in range(2, 10) for j in range(3, 10) for k in range(1, 7)
              if (i + 2*j + k) % 11 and not (i == 4 and j >= 6)]
    ids = [cell(e, point) for point in coords] + [cell(e, (12, 12, 12))]
    if reverse:
        ids.reverse()
    anchors = [*ids[::4], ids[0], cell(e, (0, 0, 0))]
    cuts = {(cell(e, (2, 3, 1)), cell(e, (2, 3, 2)))}
    e.scene.labels.ravel()[ids[1::3]] = 2
    e.state['holds'] = {'left': {'cells': [ids[5], ids[-2]], 'target': [0., 0., 0.]}}
    factorized = brain.factorized
    matrices = []

    def capture(matrix):
        matrices.append(matrix.copy())
        return factorized(matrix)

    def reference_lengths(vectors):
        distinct, groups = np.unique(vectors, axis=0, return_inverse=True)
        return np.asarray([np.linalg.norm(vector) for vector in distinct])[groups.reshape(-1)]

    monkeypatch.setattr(brain, 'factorized', capture)
    arguments = (e, profile(), ids, anchors, set(ids), cuts, {ids[1]: .003, ids[2]: .007})
    solve, free, index, edges, stiffness = brain._system(*arguments)
    del e._deformation_system_v2
    monkeypatch.setattr(brain, '_bond_lengths', reference_lengths)
    reference_solve, reference_free, reference_index, reference_edges, reference_stiffness = brain._system(*arguments)

    assert index == reference_index and edges == reference_edges
    np.testing.assert_array_equal(free, np.repeat([c not in anchors for c in ids], 3))
    np.testing.assert_array_equal(free, reference_free)
    for actual, reference in ((stiffness, reference_stiffness), matrices):
        for field in ('indptr', 'indices', 'data'):
            assert getattr(actual, field).tobytes() == getattr(reference, field).tobytes()
    rhs = np.linspace(-.01, .01, np.count_nonzero(free))
    assert solve(rhs).tobytes() == reference_solve(rhs).tobytes()


def test_neighbors_deform_smoothly_and_recover_over_time():
    e, c = held(); tick(e, .25)
    near = cell(e, (6, 7, 6)); far = cell(e, (6, 8, 6))
    assert 0 < np.linalg.norm(displacement(e, far)) < np.linalg.norm(displacement(e, near)) < np.linalg.norm(displacement(e, c))
    before = displacement(e, c).copy(); release(e, 'left')
    assert np.array_equal(displacement(e, c), before)
    tick(e, 1 / 120)
    assert 0 < np.linalg.norm(displacement(e, c)) < np.linalg.norm(before)
    tick(e, 2.)
    assert np.linalg.norm(displacement(e, c)) < .1 * np.linalg.norm(before)


def test_nonuniform_partition_and_replay_are_identical():
    a, c = held(); b = copy.deepcopy(a); replay = copy.deepcopy(a)
    tick(a, .3)
    for dt in [.013, .077, .041, .169]: tick(b, dt)
    tick(replay, .3)
    assert a.state == b.state == replay.state


def test_strain_screen_matches_full_scalar_projection_during_contact_and_recovery():
    from resectionlab.surgical_deformation_v2 import diagnostic_scope
    fast, _ = held()
    exact, _ = held()
    for e in (fast, exact):
        e.scene.properties['tissuePhysicsV2']['brain']['maxEdgeStrain'] = .05
    projected = 0
    for target in ([6.001, 6., 6.], [7., 6.5, 6.2], None):
        for e in (fast, exact):
            if target is None:
                release(e, 'left')
            else:
                set_target(e, 'left', target)
        with diagnostic_scope() as verified:
            tick(exact, .2)
        projected += verified['strainProjectionSteps']
        tick(fast, .2)
        assert fast.state == exact.state
    assert projected > 0


def test_cached_bonds_match_fresh_assembly_across_hold_cut_and_removal(monkeypatch):
    import resectionlab.surgical_deformation_v2 as brain
    warm, center = held()
    cold, _ = held()
    adjacent = cell(warm, (6, 7, 6))
    outside = [cell(warm, (0, 0, 0)), cell(warm, (0, 0, 1))]
    adjacent_cut = sorted((center, adjacent))
    original_neighbors = brain._neighbor_ids
    calls = []

    def neighbors(shape, c):
        calls.append(c)
        return original_neighbors(shape, c)

    monkeypatch.setattr(brain, '_neighbor_ids', neighbors)

    def compare(expect_assembly):
        calls.clear()
        cold.__dict__.pop('_deformation_system_v2', None)
        tick(warm, 1 / 120)
        assert bool(calls) == expect_assembly
        tick(cold, 1 / 120)
        assert warm.state == cold.state

    compare(True)
    compare(False)
    for e in (warm, cold):
        e.state['cutEdges'].append(outside)
    compare(False)
    for e in (warm, cold):
        e.state['transectedEdges'].append(adjacent_cut)
    compare(True)
    for e in (warm, cold):
        release(e, 'left')
    compare(True)
    for e in (warm, cold):
        begin_hold(e, 'right', [adjacent], [6., 7., 6.])
    compare(True)
    for e in (warm, cold):
        e.state['removed'].append(center)
    compare(True)


def test_two_holds_and_removal_never_regenerate():
    e, c = held(); d = cell(e, (8, 6, 6))
    begin_hold(e, 'right', [d], [8, 6, 6]); set_target(e, 'right', [7.7, 6, 6]); tick(e, .2)
    assert displacement(e, c)[0] > 0 and displacement(e, d)[0] < 0
    e.state['removed'] = [c]; tick(e, .1); release(e, 'left'); tick(e, .1)
    assert str(c) not in e.state['deformation'] and c not in e.state['deformationModel']['active']
    assert e.state['removed'] == [c]


def test_cut_bonds_do_not_transmit_to_isolated_node():
    e, c = held(); adjacent = cell(e, (6, 7, 6))
    from resectionlab.surgical_deformation_v2 import _neighbors
    e.state['cutEdges'] = [sorted((adjacent, n)) for n, _ in _neighbors(e, adjacent)]
    tick(e, .5)
    assert np.linalg.norm(displacement(e, c)) > 0
    assert np.linalg.norm(displacement(e, adjacent)) == 0


def test_stiffness_caps_and_fixed_material():
    brain, c = held(); tumor, _ = held(); tumor.scene.labels[:] = 2
    tick(brain, .2); tick(tumor, .2)
    assert np.linalg.norm(displacement(tumor, c)) < np.linalg.norm(displacement(brain, c))
    bone = cell(brain, (7, 6, 6)); brain.scene.labels.flat[bone] = 3
    # Fresh patch must never acquire a fixed-material node.
    fresh = engine(); fresh.scene.labels.flat[bone] = 3
    begin_hold(fresh, 'left', [c, bone], [6, 6, 6]); set_target(fresh, 'left', [100, 6, 6]); tick(fresh, .5)
    assert bone not in fresh.state['deformationModel']['active']
    assert np.linalg.norm(displacement(fresh, c)) <= .5
    field = np.array(list(fresh.state['deformation'].values()))
    assert np.isfinite(field).all() and np.max(np.linalg.norm(field, axis=1)) <= 3
    assert np.isfinite(fresh.state['deformationModel']['elasticEnergyNmm'])


def test_invalid_time_and_target_rejected():
    e, _ = held()
    with pytest.raises(ValueError): tick(e, float('nan'))
    with pytest.raises(ValueError): set_target(e, 'left', [float('inf'), 0, 0])


def test_rigid_hold_is_not_a_viscoelastic_constraint():
    a, c = held(); b = copy.deepcopy(a)
    other = cell(a, (6, 7, 6))
    b.state['holds']['right'] = {'cells': [other], 'origin': [6, 7, 6], 'offset': [0, 1, 0]}
    tick(a, .2); tick(b, .2)
    assert a.state['deformation'] == b.state['deformation']


@pytest.mark.parametrize('key,value', [('fixedStepSeconds', 1e-15), ('fixedStepSeconds', 1.), ('radiusMm', 1e20)])
def test_profile_numerical_domain_rejects_unbounded_work(key, value):
    e, _ = held(); e.scene.properties['tissuePhysicsV2']['brain'][key] = value
    with pytest.raises(ValueError): tick(e, .1)


def test_obstacle_stop_preserves_neighbor_strain_bound():
    e = engine(); c = cell(e, (6, 6, 6)); bone = cell(e, (7, 6, 6))
    e.scene.labels.flat[bone] = 3
    begin_hold(e, 'left', [c], [6, 6, 6]); set_target(e, 'left', [9, 6, 6]); tick(e, .5)
    assert e.state['deformationModel']['maxNeighborDisplacementRatio'] <= profile()['maxEdgeStrain'] + 1e-12


def _scalar_obstacle_collision(e, vox, inside, removed, blocking_labels):
    # Preserved per-node decision, independent of the vectorized implementation.
    removed = set(map(int, removed))
    for point, valid in zip(vox, inside):
        if not valid: return True
        c = int(np.ravel_multi_index(tuple(point), e.scene.shape))
        material = e.by_label.get(int(e.scene.labels.flat[c]), {}).get('material')
        if c not in removed and material in ('bone', 'vessel', 'unknown'): return True
    return False


def test_vectorized_obstacle_decisions_match_scalar_with_removed_and_outside_cells():
    from resectionlab.surgical_deformation_v2 import _obstacle_collision
    e = engine(); e.by_label[5] = {'material': 'unknown'}
    e.scene.labels[0, 0, :5] = [1, 2, 3, 4, 5]
    samples = np.array([[0, 0, i] for i in range(6)])
    blocking = np.array([3, 4, 5], dtype=e.scene.labels.dtype)
    for removed in ([], [2, 3, 4], [0, 1, 2, 3, 4, 5]):
        for points in (samples, samples + [0, 0, -1], samples + [0, 0, 12]):
            inside = np.all((points >= 0) & (points < e.scene.shape), axis=1)
            assert _obstacle_collision(e, points, inside, removed, blocking) == _scalar_obstacle_collision(e, points, inside, removed, blocking)


def test_solver_state_is_bit_identical_to_scalar_obstacle_reference(monkeypatch):
    import resectionlab.surgical_deformation_v2 as brain
    a, c = held(); a.scene.labels.flat[cell(a, (7, 6, 6))] = 3
    b = copy.deepcopy(a)
    for dt in (.008, .009, .014): tick(a, dt)
    release(a, 'left'); tick(a, .01)
    monkeypatch.setattr(brain, '_obstacle_collision', _scalar_obstacle_collision)
    for dt in (.008, .009, .014): tick(b, dt)
    release(b, 'left'); tick(b, .01)
    assert a.state == b.state
