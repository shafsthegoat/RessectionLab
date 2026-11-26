from dataclasses import replace

import numpy as np
import pytest

from resectionlab import vessel_tubes as vt
from resectionlab.vascular_parity import betti
from resectionlab.vascular_rooting import SCHEMES


def field(capsules, shape, spacing, points=None):
    """Hard union of capsules (mm) sampled at voxel centres (or at ``points``)."""
    grid = points if points is not None else np.stack(
        np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing='ij'), -1).reshape(-1, 3)
    d = np.full(len(grid), np.inf)
    for a, b, r in capsules:
        d = np.minimum(d, vt._round_cone(grid, np.asarray(a, float)[None], np.asarray(b, float)[None],
                                         np.array([r]), np.array([r])))
    return d if points is not None else d.reshape(shape)


def case(capsules, shape, spacing=(.5, .5, .5)):
    """Partial-volume probability, its half-level mask and the affine."""
    probability = np.clip(.5 - field(capsules, shape, spacing) / min(spacing), 0, 1).astype(np.float32)
    return probability >= .5, probability, np.diag([*spacing, 1.])


def build(capsules, shape, spacing=(.5, .5, .5), **kwargs):
    mask, probability, affine = case(capsules, shape, spacing)
    model = vt.extract(mask, affine, probability=probability, **kwargs)
    return model, vt.mesh(model, block=16), mask, probability, affine


def build_field(distance, shape, spacing=(.5, .5, .5)):
    """Model and mesh of the zero level of ``distance`` (mm points -> mm), with partial-volume probability."""
    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing='ij'), -1).reshape(-1, 3)
    probability = np.clip(.5 - distance(grid).reshape(shape) / min(spacing), 0, 1).astype(np.float32)
    affine = np.diag([*spacing, 1.])
    model = vt.extract(probability >= .5, affine, probability=probability)
    return model, vt.mesh(model, block=16), probability >= .5, probability, affine


def elliptical_tube(p, a=1.3, b=.7):
    """Elliptical tube along x with flat-cut ends."""
    return np.maximum((np.hypot((p[:, 1] - 8) / a, (p[:, 2] - 8) / b) - 1) * min(a, b), np.abs(p[:, 0] - 10) - 7)


def flat_cylinder(p, radius=1.1):
    """Round cylinder with flat-cut ends (exact distance)."""
    along, across = np.abs(p[:, 0] - 10) - 6, np.hypot(p[:, 1] - 8, p[:, 2] - 8) - radius
    return np.minimum(np.maximum(along, across), 0) + np.hypot(np.maximum(along, 0), np.maximum(across, 0))


def assert_closed(mesh, components=1, b1=0):
    report = vt.audit(mesh.positions, mesh.faces)
    for key in ('open_edges', 'non_manifold_edges', 'inconsistent_edges', 'non_manifold_vertices',
                'degenerate_faces', 'self_intersections'):
        assert report[key] == 0, (key, report)
    assert report['volume_mm3'] > 0
    assert (report['components'], report['b1']) == (components, b1)
    return report


def assert_deviation(mesh, probability, affine, limit=.5):
    report = vt.deviation(mesh, probability, affine)
    for side in ('tube_to_source', 'source_to_tube'):
        assert report[side]['all']['voxels_p95'] <= limit, report
        assert report[side]['all']['voxels_mean'] <= limit / 2, report
    return report


def test_round_cone_is_exact_for_capsules_balls_and_tapers():
    rng = np.random.default_rng(1)
    p = rng.uniform(-3, 3, (500, 3))
    a, b = np.array([[0., 0., -1.]]), np.array([[0., 0., 1.]])
    capsule = vt._round_cone(p, a, b, np.array([.5]), np.array([.5]))
    t = np.clip(p[:, 2], -1, 1)
    assert np.allclose(capsule, np.linalg.norm(p - np.c_[0 * t, 0 * t, t], axis=1) - .5)
    nested = vt._round_cone(p, a, b, np.array([3.]), np.array([.5]))
    assert np.allclose(nested, np.linalg.norm(p - a, axis=1) - 3.)
    # A tapered cone: points on the tangent line at the wall have zero distance.
    ra, rb = 1., .4
    sin = (ra - rb) / 2.
    cos = np.sqrt(1 - sin * sin)
    wall = a[0] + ra * np.array([cos, 0., sin])
    along = (b[0] + rb * np.array([cos, 0., sin])) - wall
    points = wall + np.linspace(0, 1, 9)[:, None] * along
    assert np.allclose(vt._round_cone(points, a, b, np.array([ra]), np.array([rb])), 0., atol=1e-12)


def test_oblique_tube_recovers_sub_voxel_radius_and_axis():
    start, end, radius = np.array([4., 5., 6.]), np.array([22., 18., 14.]), .685
    model, mesh, mask, probability, affine = build([(start, end, radius)], (54, 46, 40))
    assert len(model.edges) == 1 and not len(model.junctions)
    interior = np.concatenate(model.edge_radii)[3:-3]
    assert abs(np.median(interior) - radius) < .1 * .5
    axis = (end - start) / np.linalg.norm(end - start)
    offset = model.points - start
    lateral = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)
    assert np.percentile(lateral, 95) < .15 * .5
    assert_closed(mesh)
    assert_deviation(mesh, probability, affine)
    # Without a probability the mask's own half level is the reference boundary.
    from_mask = vt.mesh(vt.extract(mask, affine), block=16)
    assert_closed(from_mask)
    assert_deviation(from_mask, mask.astype(np.float32), affine)


@pytest.mark.parametrize('capsules', [
    [((5, 12, 12), (15, 12, 12), 1.6), ((15, 12, 12), (24, 17, 13), 1.1), ((15, 12, 12), (24, 7, 11), 1.)],
    [((3, 12, 12), (27, 12, 12), 1.5), ((15, 12, 12), (15, 22, 12), .8)],
], ids=['Y', 'T'])
def test_junctions_are_closed_and_within_half_a_voxel(capsules):
    model, mesh, _, probability, affine = build(capsules, (60, 50, 50))
    assert len(model.junctions) == 1 and len(model.junction_edges[0]) == 3
    assert_closed(mesh)
    assert_deviation(mesh, probability, affine)
    # Without a fitted blend the field is the hard union of the section cones and the node ball.
    near = model.points[model.junctions[0]] + np.random.default_rng(2).normal(0, 1., (400, 3))
    value, _ = model.evaluate(near)
    seg = model.segments
    rows = np.repeat(np.arange(len(near)), len(seg['a']))
    hard = model._segment_field(near[rows], np.tile(np.arange(len(seg['a'])), len(near))).reshape(len(near), -1).min(axis=1)
    assert np.allclose(value, np.minimum(hard, model.spacing.max() * 4))


def test_blended_junction_never_exceeds_the_hard_union_and_stays_a_distance_bound():
    capsules = [((5, 12, 12), (15, 12, 12), 1.6), ((15, 12, 12), (24, 17, 13), 1.1), ((15, 12, 12), (24, 7, 11), 1.)]
    model, *_ = build(capsules, (60, 50, 50))
    blended = replace(model, junction_blend=np.array([.5]))
    near = model.points[model.junctions[0]] + np.random.default_rng(2).normal(0, 1., (400, 3))
    soft, hard = blended.evaluate(near)[0], model.evaluate(near)[0]
    assert np.all(soft <= hard + 1e-9) and np.any(soft < hard - 1e-3)
    wall = vt.mesh(blended, block=16)
    assert_closed(wall)
    assert_distance_bound(blended, wall, near)


def test_block_seams_do_not_change_the_wall():
    capsules = [((5, 12, 12), (15, 12, 12), 1.6), ((15, 12, 12), (24, 17, 13), 1.1), ((15, 12, 12), (24, 7, 11), 1.)]
    model, mesh, *_ = build(capsules, (60, 50, 50))
    whole = vt.mesh(model, block=512)
    assert (len(whole.positions), len(whole.faces)) == (len(mesh.positions), len(mesh.faces))
    from scipy.spatial import cKDTree
    gap, match = cKDTree(whole.positions).query(mesh.positions)
    assert gap.max() < 1e-5 and len(np.unique(match)) == len(match)


def test_parallel_thin_vessels_closer_than_the_blend_stay_separate():
    # Walls 0.6 mm apart (1.2 voxels) with a junction blend radius of about 0.35 mm nearby.
    capsules = [((4, 10, 10), (24, 10, 10), .7), ((4, 12, 10), (24, 12, 10), .7),
                ((14, 10, 10), (14, 4, 10), .7)]
    model, mesh, mask, probability, affine = build(capsules, (58, 34, 40))
    assert betti(mask)['b0'] == 2
    assert_closed(mesh, components=2)
    assert_deviation(mesh, probability, affine)
    value, _ = model.evaluate(np.array([[9., 11., 10.], [19., 11., 10.]]))
    assert np.all(value > .2)


def test_loop_keeps_its_cycle_and_anisotropic_spacing_is_honoured():
    t = np.linspace(0, 2 * np.pi, 25)
    ring = np.c_[10 + 5 * np.cos(t), 10 + 5 * np.sin(t), np.full_like(t, 6.)]
    capsules = [(a, b, .9) for a, b in zip(ring[:-1], ring[1:])]
    spacing = (.4, .4, .7)
    model, mesh, mask, probability, affine = build(capsules, (52, 52, 18), spacing)
    assert betti(mask)['b1'] == 1
    assert_closed(mesh, b1=1)
    assert_deviation(mesh, probability, affine)
    assert np.isclose(model.receipt['radiusFloorMm'], .2)


def test_tiny_stub_is_kept_as_its_own_component():
    capsules = [((3, 8, 8), (20, 8, 8), 1.), ((12, 15, 8), (12.4, 15, 8), .5)]
    model, mesh, mask, probability, affine = build(capsules, (48, 36, 34))
    assert betti(mask)['b0'] == 2
    assert_closed(mesh, components=2)
    assert_deviation(mesh, probability, affine)


def assert_distance_bound(model, mesh, points):
    """Outside the wall the field never exceeds the distance to the meshed wall plus the mesh tolerance."""
    value, _ = model.evaluate(points)
    outside = (value > 0) & (value < 4 * model.spacing.max())
    distance, _ = vt.point_mesh_distance(points[outside], mesh.positions, mesh.faces)
    assert outside.sum() > 100
    assert np.all(value[outside] <= distance + model.wall_tolerance)
    # The gradient-normalised field is 1-Lipschitz outside the wall.
    seg = model.segments
    rows = np.flatnonzero(outside)[:300]
    pairs = np.repeat(rows, len(seg['a'])), np.tile(np.arange(len(seg['a'])), len(rows))
    eps = 1e-5
    field = lambda q: model._segment_field(q, pairs[1], safe=False).reshape(len(rows), -1)
    base = field(points[pairs[0]])
    for e in np.eye(3) * eps:
        slope = np.abs(field(points[pairs[0]] + e) - base) / eps
        assert np.all(slope[base > eps] <= 1 + 1e-4)


def test_collision_queries_use_the_implicit_wall():
    start, end, radius = np.array([4., 8., 8.]), np.array([20., 8., 8.]), 1.2
    model, mesh, *_ = build([(start, end, radius)], (48, 34, 34))
    points = np.array([[12., 8., 8.], [12., 8. + 2.5, 8.], [12., 8., 8. - 1.]])
    distance, normal, closest, owner = model.query(points)
    assert np.allclose(distance, [-radius, 2.5 - radius, 1. - radius], atol=.06)
    assert np.allclose(normal[1], [0, 1, 0], atol=1e-3) and np.allclose(normal[2], [0, 0, -1], atol=1e-3)
    assert np.allclose(np.linalg.norm(closest[1:] - [12., 8., 8.], axis=1), radius, atol=.06)
    assert np.all(owner == 0)
    # Mesh vertices lie on the zero level that collision uses.
    value, _ = model.evaluate(mesh.positions)
    assert np.abs(value).max() < .02
    assert np.all(np.einsum('ij,ij->i', mesh.normals, model.query(mesh.positions)[1]) > .98)
    clearance, _, _, _ = model.capsule_clearance([12., 12., 8.], [12., 9.6, 8.], .3)
    exact = 9.6 - 8. - radius - .3
    assert exact - model.wall_tolerance - .1 <= clearance <= exact


@pytest.mark.parametrize('shape', ['ellipse', 'flat'])
def test_capsule_clearance_is_a_lower_bound_on_the_meshed_wall(shape):
    """A capsule reported clear never meets the mesh: clearance <= mesh distance of the axis less the radius."""
    model, mesh, *_ = build_field(elliptical_tube if shape == 'ellipse' else flat_cylinder, (40, 32, 32))
    assert_distance_bound(model, mesh, np.random.default_rng(4).uniform(1, 15, (3000, 3)))
    rng = np.random.default_rng(5)
    measured = max(vt.audit(mesh.positions, mesh.faces, self_intersections=False, model=model)['wall_outside_max_mm'], 0.)
    assert measured <= model.wall_tolerance
    reported, audited, exact = [], [], []
    for k in rng.choice(len(mesh.positions), 60, replace=False):
        # Capsules beside the wall (and some through it), lying along it.
        radius = rng.uniform(.05, .5)
        centre = mesh.positions[k] + mesh.normals[k] * (rng.uniform(-.3, 1.) + radius)
        direction = np.cross(mesh.normals[k], rng.normal(size=3))
        a, b = centre - direction, centre + direction
        reported.append(model.capsule_clearance(a, b, radius)[0])
        audited.append(model.capsule_clearance(a, b, radius, tolerance=measured)[0])
        axis = a + np.linspace(0, 1, 400)[:, None] * (b - a)
        distance, _ = vt.point_mesh_distance(axis, mesh.positions, mesh.faces)
        value, _ = model.evaluate(axis)
        exact.append(np.where(value > 0, distance, -distance).min() - radius)
    reported, audited, exact = np.array(reported), np.array(audited), np.array(exact)
    clear = reported > 0
    assert clear.sum() >= 10 and np.any(~clear)
    assert np.all(reported[clear] <= exact[clear])
    # With the audited mesh tolerance the bound still holds and is tighter.
    assert np.all(audited[audited > 0] <= exact[audited > 0]) and np.all(audited >= reported)
    # The bound is conservative by the gradient bound of the sections (about 1.35 beside this ellipse).
    near = (audited > 0) & (exact < .5)
    assert near.sum() >= 5 and np.median(exact[near] - audited[near]) < .25


def test_tags_rods_classes_and_round_trip_share_the_model():
    capsules = [((5, 12, 12), (15, 12, 12), 1.6), ((15, 12, 12), (24, 17, 13), 1.1), ((15, 12, 12), (24, 7, 11), 1.)]
    mask, probability, affine = case(capsules, (60, 50, 50))
    named = np.where(mask, 4, 0).astype(np.uint8)
    named[(np.arange(60) >= 34)[:, None, None] & (np.arange(50) >= 25)[None, :, None] & mask] = 35
    model = vt.extract(mask, affine, probability=probability, named=named, scheme=SCHEMES['topbrain-v1-ct'])
    assert set(model.edge_label) == {4, 35}
    assert {model.edge_class[model.edge_label.index(4)], model.edge_class[model.edge_label.index(35)]} == {'artery', 'vein'}

    origin, tags = vt.tag_volume(model)
    full = np.zeros(mask.shape, np.float32)
    full[tuple(slice(o, o + n) for o, n in zip(origin, tags.shape))] = tags
    inside = full >= .5
    assert 2 * np.sum(inside & mask) / (inside.sum() + mask.sum()) > .97
    assert np.all((full >= 0) & (full <= 1)) and np.any((full > 0) & (full < 1))

    cells = np.flatnonzero(mask)[::97]
    sites = vt.rod_sites(model, cells)
    ijk = np.column_stack(np.unravel_index(cells, mask.shape))
    points = np.array([sites[int(c)][:3] for c in cells]) / .5
    assert np.abs(points - ijk).max() <= .5 + 1e-9
    assert all(.03 <= site[3] <= 1.6 + .1 for site in sites.values())

    again = vt.TubeModel.from_dict(model.to_dict())
    probe = np.random.default_rng(3).uniform(2, 12, (200, 3))
    assert np.array_equal(again.evaluate(probe)[0], model.evaluate(probe)[0])
    with pytest.raises(ValueError, match='CENTERLINE_TUBES_V2'):
        vt.TubeModel.from_dict({**model.to_dict(), 'model': 'other'})


def test_elliptical_section_is_fitted_by_harmonics():
    model, mesh, _, probability, affine = build_field(elliptical_tube, (40, 32, 32))
    assert len(model.edges) == 1
    assert_closed(mesh)
    report = assert_deviation(mesh, probability, affine, limit=.3)
    # The second harmonic carries the ellipse: (a - b) / 2 = 0.3 mm in the interior.
    interior = model.sections[5:-5]
    assert np.median(np.hypot(interior[:, 2], interior[:, 3])) > .2
    assert report['source_to_tube']['all']['fraction_within_half_voxel'] > .99


def test_fused_pair_is_one_tube_with_a_waist():
    capsules = [((4, 10, 10), (20, 10, 10), .8), ((4, 11.3, 10), (20, 11.3, 10), .8)]
    model, mesh, mask, probability, affine = build(capsules, (48, 42, 40))
    assert betti(mask)['b0'] == 1
    assert_closed(mesh)
    assert_deviation(mesh, probability, affine, limit=.3)


def test_flared_junction_is_followed():
    def flared(p):
        arms = field([((5, 12, 12), (15, 12, 12), 1.), ((15, 12, 12), (24, 17, 13), .8), ((15, 12, 12), (24, 7, 11), .8)],
                     None, None, points=p)
        cone = vt._round_cone(p, np.array([[12.5, 12, 12]]), np.array([[16.5, 12, 12]]), np.array([1.]), np.array([2.]))
        return np.minimum(arms, cone)
    model, mesh, _, probability, affine = build_field(flared, (60, 50, 50))
    assert len(model.junctions) >= 1
    assert_closed(mesh)
    assert_deviation(mesh, probability, affine)


def test_flat_cut_end_is_followed():
    model, mesh, _, probability, affine = build_field(flat_cylinder, (40, 32, 32))
    assert_closed(mesh)
    assert_deviation(mesh, probability, affine, limit=.3)
    # The end cap terms flatten the hemisphere: the axial offset at the tips is negative.
    assert np.all(model.sections[[0, -1], -2:].sum(axis=1) < 0)


def test_audit_reports_holes_and_crossings():
    model, mesh, *_ = build([((4, 8, 8), (20, 8, 8), 1.)], (48, 34, 34))
    holed = vt.audit(mesh.positions, mesh.faces[1:])
    assert holed['open_edges'] == 3
    shifted = mesh.positions + [0., 1., 0.]
    both = vt.audit(np.vstack((mesh.positions, shifted)), np.vstack((mesh.faces, mesh.faces + len(mesh.positions))))
    assert both['self_intersections'] > 0 and both['open_edges'] == 0
    glued = np.vstack((mesh.faces, mesh.faces + len(mesh.positions)))
    glued[glued == len(mesh.positions)] = 0
    assert vt.audit(np.vstack((mesh.positions, mesh.positions)), glued, self_intersections=False)['non_manifold_vertices'] >= 1


def test_tubes_of_separate_components_are_kept_apart():
    capsules = [((4, 10, 10), (24, 10, 10), .7), ((4, 12.4, 10), (24, 12.4, 10), .7)]
    model, *_ = build(capsules, (58, 40, 40))
    inflated = replace(model, edge_radii=tuple(r + .6 for r in model.edge_radii))
    separated = vt._separate(inflated, np.array([1, 2]), gap=.25, floor=.25)
    assert separated.receipt['separatedSegments'] > 0
    touching = vt.mesh(inflated, block=16)
    assert vt.audit(touching.positions, touching.faces, self_intersections=False)['components'] == 1
    mesh = vt.mesh(separated, block=16)
    assert_closed(mesh, components=2)


def test_query_projects_to_the_wall_and_reports_nothing_beyond_reach():
    start, end, radius = np.array([4., 8., 8.]), np.array([20., 8., 8.]), 1.2
    model, *_ = build([(start, end, radius)], (48, 34, 34))
    points = np.array([[12., 9.7, 8.3], [12., 8.4, 8.], [12., 30., 8.]])
    distance, normal, closest, owner = model.query(points, reach=4.)
    raw, _ = model.evaluate(closest[:2], raw=True)
    assert np.abs(raw).max() < 1e-4
    assert np.allclose(closest[:2, 0], 12., atol=1e-3)
    assert np.allclose(np.linalg.norm(closest[:2, 1:] - [8., 8.], axis=1), radius, atol=.06)
    assert np.isinf(distance[2]) and np.isnan(normal[2]).all() and np.isnan(closest[2]).all() and owner[2] == -1
    assert np.all(owner[:2] == 0)


def _tetrahedron(base, apex):
    points = np.vstack((base, apex))
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    if vt._signed_volume(points, faces) < 0:
        faces = faces[:, ::-1]
    return points, faces


def test_audit_finds_coplanar_overlapping_faces():
    first = _tetrahedron(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), [0., 0., 1.])
    second = _tetrahedron(np.array([[.2, .2, 0.], [1.2, .2, 0.], [.2, 1.2, 0.]]), [.2, .2, -1.])
    points = np.vstack((first[0], second[0]))
    faces = np.vstack((first[1], second[1] + 4))
    assert vt.audit(points, faces)['self_intersections'] > 0
    apart = np.vstack((first[0], second[0] + [3., 0., 0.]))
    assert vt.audit(apart, faces)['self_intersections'] == 0
    a, b = np.array([[.1, .1, 0.], [2., 2., 0.]]), np.array([[.3, .2, 0.], [3., 2., 0.]])
    triangle = np.repeat(first[0][None, :3], 2, axis=0)
    assert vt._coplanar_overlaps(a, b, triangle).tolist() == [True, False]


def test_deviation_counts_a_missed_vessel_against_its_source_class():
    capsules = [((4, 10, 10), (24, 10, 10), 1.), ((4, 22, 10), (24, 22, 10), 1.)]
    mask, probability, affine = case(capsules, (58, 64, 40))
    only_first = mask & (np.arange(64)[None, :, None] < 32)
    model = vt.extract(only_first, affine, probability=np.where(only_first, probability, 0.))
    classes = np.where(mask, 1, 0)
    report = vt.deviation(vt.mesh(model, block=16), probability, affine, model=model, source_classes=classes)
    side = report['source_to_tube']
    assert report['classes_from'] == 'source' and side['unowned'] > 100
    assert side['artery']['points'] == side['walls']['points'] and side['artery']['voxels_p95'] > 5
