"""Independent continuum controls: units, rigid modes, energy and source topology."""
import copy
import json

import numpy as np
import pytest

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab import surgical_continuum as continuum
from resectionlab import surgical_deformation_v2 as brain
from resectionlab.surgical_engine import SurgicalEngine


def material(mu=1000., nu=.3, branches=()):
    return {'shearPa': mu, 'poissonRatio': nu, 'relaxationShearPa': list(branches),
        'source': 'Generated numerical-control coefficient; not a patient measurement'}


def block(n=1, affine=None, *, branches=(), times=()):
    labels = np.ones((n, n, n), np.uint16)
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
        [{'id': 1, 'name': 'Generated block', 'material': 'soft_tissue'}],
        {'kind': 'analytical engineering coupon'}, {'tissuePhysicsV2': default_profile(),
            continuum.PROPERTY: continuum.research_profile({'1': material(branches=branches)}, relaxation_seconds=times)})


def kernel(scene):
    mesh = continuum.voxel_mesh(scene, range(scene.labels.size))
    matrices, gradients = continuum.assemble(scene, mesh, scene.properties[continuum.PROPERTY]['materials'])
    return mesh, matrices[0], gradients


def test_hex8_has_exactly_six_rigid_modes_and_no_hourglass_modes():
    mesh, matrix, _ = kernel(block())
    eigenvalues = np.linalg.eigvalsh(matrix.toarray())
    assert np.count_nonzero(np.abs(eigenvalues) < eigenvalues[-1] * 1e-10) == 6
    assert eigenvalues[6] > eigenvalues[-1] * 1e-3
    assert eigenvalues.min() > -1e-14


def test_small_gradient_guard_includes_corner_extrema_missed_by_gauss_points():
    mesh, _, gradients = kernel(block())
    u = np.zeros_like(mesh.nodes); u[:, 0] = .4 * mesh.nodes[:, 0] * mesh.nodes[:, 1]
    gradient = np.einsum('ai,gaj->gij', u, gradients)
    assert np.linalg.norm(gradient[:8], axis=(1, 2)).max() < .2
    assert np.linalg.norm(gradient, axis=(1, 2)).max() == pytest.approx(np.sqrt(.08))


AFFINES = [np.diag([1., 2., .7, 1.]),
    np.array([[1., .2, 0., 4.], [0., 1.1, .3, -2.], [.1, 0., .8, 7.], [0., 0., 0., 1.]]),
    np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
    np.diag([-1., 1., 1., 1.])]


@pytest.mark.parametrize('affine', AFFINES)
def test_rigid_motion_and_general_affine_energy_in_physical_units(affine):
    source = block(2, affine); mesh, matrix, _ = kernel(source)
    skew = np.array([[0., -.01, .02], [.01, 0., -.03], [-.02, .03, 0.]])
    rigid = (mesh.nodes @ skew.T + [3., -2., 4.]).ravel()
    assert np.linalg.norm(matrix @ rigid) < 1e-13
    g = np.array([[.01, .004, -.002], [0., -.003, .005], [.001, .002, .006]])
    u = (mesh.nodes @ g.T).ravel()
    eps = (g + g.T) / 2
    mu = .001; nu = .3; lam = 2 * mu * nu / (1 - 2 * nu)
    density = mu * np.sum(eps**2) + .5 * lam * np.trace(eps)**2
    expected = density * mesh.volume_mm3 * len(mesh.cells)
    assert .5 * u @ (matrix @ u) == pytest.approx(expected, rel=1e-10, abs=1e-16)


def test_same_physical_cube_has_grid_invariant_energy_and_resultant_reaction():
    energies = []; loads = []
    for spacing in (1., .5, .25):
        source = block(round(2 / spacing), np.diag([spacing]*3 + [1.]))
        mesh, matrix, _ = kernel(source)
        u = np.zeros_like(mesh.nodes); u[:, 0] = .01 * mesh.nodes[:, 0]
        force = (matrix @ u.ravel()).reshape((-1, 3))
        right = np.isclose(mesh.nodes[:, 0], mesh.nodes[:, 0].max())
        energies.append(.5 * u.ravel() @ (matrix @ u.ravel()))
        loads.append(force[right, 0].sum())
    np.testing.assert_allclose(energies, [energies[0]] * 3, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(loads, [loads[0]] * 3, rtol=1e-10, atol=1e-16)


def test_cut_face_splits_all_corner_connections_and_removal_is_exact():
    labels = np.ones((2, 1, 1), np.uint16)
    source = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'block', 'material': 'soft_tissue'}], {}, {})
    intact = continuum.voxel_mesh(source, [0, 1])
    cut = continuum.voxel_mesh(source, [0, 1], [(0, 1)])
    assert len(intact.nodes) == 12 and len(cut.nodes) == 16
    assert len(cut.components) == 2
    assert not set(cut.element_nodes[0]) & set(cut.element_nodes[1])
    matrix = continuum.assemble(source, cut, {'1': material()})[0][0]
    assert not matrix[cut.element_dofs[0]][:, cut.element_dofs[1]].nnz
    retained = continuum.voxel_mesh(source, [1])
    assert retained.cells == (1,) and retained.volume_mm3 * len(retained.cells) == 1.
    np.testing.assert_array_equal(source.labels, labels)


def test_edge_or_corner_touching_voxels_are_not_glued_without_a_face_path():
    source = block(2)
    a = int(np.ravel_multi_index((0, 0, 0), source.shape))
    for point in ((1, 1, 0), (1, 1, 1)):
        b = int(np.ravel_multi_index(point, source.shape))
        mesh = continuum.voxel_mesh(source, [a, b])
        assert len(mesh.nodes) == 16 and len(mesh.components) == 2


def test_unsupported_component_and_dof_cap_fail_explicitly():
    mesh, _, _ = kernel(block())
    with pytest.raises(ValueError, match='UNSUPPORTED_RIGID_COMPONENT'):
        continuum.check_support(mesh, set(), {0})
    with pytest.raises(ValueError, match='DOF_LIMIT'):
        continuum.voxel_mesh(block(2), range(8), max_dofs=24)


def held_engine(*, branches=(500.,), times=(.5,)):
    source = block(13, branches=branches, times=times)
    e = SurgicalEngine(source)
    c = int(np.ravel_multi_index((6, 6, 6), source.shape))
    brain.begin_hold(e, 'left', [c], [6., 6., 6.])
    brain.set_target(e, 'left', [6.1, 6., 6.])
    return e, c


def test_history_relaxes_reaction_and_release_recovers_with_energy_balance():
    e, c = held_engine()
    brain.tick(e, .1)
    initial = np.linalg.norm(e.state[continuum.PROPERTY]['handleReactionsN']['left'])
    brain.tick(e, 2.)
    later = np.linalg.norm(e.state[continuum.PROPERTY]['handleReactionsN']['left'])
    assert 0 < later < initial
    assert e.state[continuum.PROPERTY]['dissipatedEnergyNmm'] > 0
    held_displacement = np.linalg.norm(e.state['deformation'][str(c)])
    brain.release(e, 'left'); brain.tick(e, .1)
    after = np.linalg.norm(e.state['deformation'][str(c)])
    assert 0 < after < held_displacement
    brain.tick(e, 3.)
    assert np.linalg.norm(e.state['deformation'][str(c)]) < after
    assert e.state[continuum.PROPERTY]['maxFreeResidualN'] < 1e-8
    assert e.state[continuum.PROPERTY]['maxEnergyBalanceResidualNmm'] < 1e-8


@pytest.mark.parametrize('strengths,times', [((500.,), (.5,)), ((300., 800.), (.2, 2.)), ((0., 800.), (.2, 2.))])
def test_imposed_shear_relaxation_matches_independent_two_time_constant_oracle(strengths, times):
    source = block(branches=strengths, times=times)
    mesh = continuum.voxel_mesh(source, [0])
    matrices, _ = continuum.assemble(source, mesh, source.properties[continuum.PROPERTY]['materials'])
    u = np.zeros_like(mesh.nodes); u[:, 0] = .01 * mesh.nodes[:, 2]
    u = u.ravel(); memories = [np.zeros_like(u) for _ in times]
    upper = np.isclose(mesh.nodes[:, 2], .5)
    h = .01
    for n in range(1, 31):
        memories = continuum.update_memory(memories, u, times, h)
        force = matrices[0] @ u
        for matrix, z in zip(matrices[1:], memories): force += matrix @ (u - z)
        observed = force.reshape((-1, 3))[upper, 0].sum()
        # Stress on a unit-area, homogeneously sheared voxel, derived from the
        # declared Maxwell law, not solver-provided reaction/error summaries.
        expected = (1000. + sum(mu * (t / (t + h))**n for mu, t in zip(strengths, times))) * 1e-6 * .01
        assert observed == pytest.approx(expected, rel=1e-10, abs=1e-15)


def test_volume_time_exposure_matches_independent_homogeneous_shear_integrals():
    source = block(); mesh, _, gradients = kernel(source)
    u = np.zeros_like(mesh.nodes); u[:, 0] = .01 * mesh.nodes[:, 2]
    dt = .2; mu_pa = 1000.
    values = continuum.exposure_increment(source, mesh, gradients, u.ravel(), [], source.properties[continuum.PROPERTY]['materials'], dt)['1']
    assert values['strainSquaredVolumeTimeMm3Seconds'] == pytest.approx(.01**2 / 2 * dt, rel=1e-12)
    assert values['stressSquaredVolumeTimePa2Mm3Seconds'] == pytest.approx(2 * (mu_pa * .01)**2 * dt, rel=1e-12)


def test_source_removal_exports_its_stored_energy_instead_of_hiding_it():
    e, c = held_engine(); brain.tick(e, .1)
    state = e.state[continuum.PROPERTY]
    field = np.asarray(state['cornerDisplacementMm'][str(c)])
    branch_fields = [field - np.asarray(z[str(c)]) for z in state['memoryDisplacementMm']]
    exported = continuum._cell_energy(e.scene, e.scene.properties[continuum.PROPERTY], c, field, branch_fields)
    assert exported > 0
    e.state['removed'] = [c]; brain.tick(e, .1)
    assert e.state[continuum.PROPERTY]['removedStoredEnergyNmm'] == pytest.approx(exported, rel=1e-12)
    assert e.state[continuum.PROPERTY]['removedElementCount'] == 1
    assert str(c) not in e.state[continuum.PROPERTY]['cornerDisplacementMm']
    assert str(c) not in e.state['deformation']
    brain.tick(e, .1)
    assert e.state[continuum.PROPERTY]['removedStoredEnergyNmm'] == pytest.approx(exported, rel=1e-12)


def translated_patch():
    labels = np.ones((2, 2, 1), np.uint16); labels[1, 1, 0] = 0
    source = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'free translated coupon', 'material': 'soft_tissue'}], {},
        {'tissuePhysicsV2': default_profile(), continuum.PROPERTY: continuum.research_profile({'1': material()})})
    e = SurgicalEngine(source); cells = list(map(int, np.flatnonzero(labels)))
    brain.begin_hold(e, 'left', cells, [0., 0., 0.]); brain.set_target(e, 'left', [.1, 0., 0.]); brain.tick(e, .1)
    assert e.state['deformationModel']['elasticEnergyNmm'] < 1e-14
    assert max(np.linalg.norm(v) for v in e.state['deformation'].values()) > .09
    return e, cells


def test_zero_energy_rigid_displacement_cannot_disappear_as_inactive_history():
    e, _ = translated_patch(); e.state['deformationModel']['active'] = []
    with pytest.raises(ValueError, match='UNSUPPORTED_HISTORY_TRANSFER'):
        brain.tick(e, .1)


def test_fully_removed_patch_has_empty_active_summary_but_retains_export_ledger():
    e, cells = translated_patch(); e.state['removed'] = cells
    brain.tick(e, .1)
    state = e.state[continuum.PROPERTY]
    assert state['activeElements'] == state['activeDofs'] == 0
    assert state['handleReactionsN'] == state['cornerDisplacementMm'] == {}
    assert state['removedElementCount'] == len(cells)


def test_fixed_step_partition_and_json_history_roundtrip_are_exact():
    a, _ = held_engine(); b = copy.deepcopy(a)
    brain.tick(a, .3)
    for duration in (.013, .077, .041, .169): brain.tick(b, duration)
    assert a.state == b.state
    assert json.loads(json.dumps(a.state)) == a.state


def test_cached_operator_refuses_unversioned_source_array_exchange():
    e, _ = held_engine(); brain.tick(e, .1)
    replacement = e.scene.labels.copy(); replacement.setflags(write=False)
    e.scene.labels = replacement
    with pytest.raises(ValueError, match='Source scene changed'):
        brain.tick(e, .1)


def test_cached_neighbor_diagnostic_keeps_order_and_rebuilds_after_removal():
    e, c = held_engine(); brain.tick(e, .1)

    def expected(system):
        cells = system['mesh'].cells; index = {c: i for i, c in enumerate(cells)}
        blocked = brain._blocked(e)
        return tuple((i, index[other], length) for i, cell in enumerate(cells)
            for other, length in brain._neighbors(e, cell)
            if other in index and cell < other and (cell, other) not in blocked)

    system = e._continuum_system_v1[1]
    assert system['neighbor_pairs'] == expected(system)
    brain.tick(e, .1)
    assert e._continuum_system_v1[1] is system
    e.state['removed'] = [c]; brain.tick(e, .1)
    changed = e._continuum_system_v1[1]
    assert changed is not system and changed['neighbor_pairs'] == expected(changed)
    assert c not in changed['mesh'].cells


@pytest.mark.parametrize('change', [
    lambda p: p.update(maxElementGradientNorm=.9),
    lambda p: p['materials']['1'].update(poissonRatio=.49),
    lambda p: p['materials']['1'].update(shearPa=-1.),
    lambda p: p.update(materials={}),
    lambda p: p['materials']['1'].update(relaxationShearPa=[1., 2.]),
])
def test_incomplete_or_unqualified_material_domains_are_refused(change):
    source = block(); change(source.properties[continuum.PROPERTY])
    with pytest.raises(ValueError): continuum.validate(source)


def test_large_gradient_is_rejected_instead_of_projecting_a_solution():
    e, _ = held_engine()
    brain.set_target(e, 'left', [8., 6., 6.])
    with pytest.raises(ValueError, match='SMALL_GRADIENT_DOMAIN'): brain.tick(e, .1)
