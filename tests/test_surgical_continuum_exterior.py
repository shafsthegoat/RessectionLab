"""Dense full equations independently check condensed fields and full energy."""
import numpy as np
import pytest
from scipy import sparse

from resectionlab.surgical_continuum_exterior import ExteriorOperator, ExteriorHistorySolve, source_stiffness


def problem(seed=812, branches=2, contacts=True):
    rng = np.random.default_rng(seed); a = rng.normal(size=(14, 14))
    k = a.T@a+np.eye(14); core = np.array([8, 0, 5, 11, 3])
    c = np.zeros((3 if contacts else 0, 14))
    if contacts:
        c[0, core[:2]] = [.3, .7]; c[1, core[2:4]] = [.5, .5]; c[2, core[-1]] = 1.
    d = np.array([.2, .7, 1.3]) if contacts else np.empty(0)
    alpha = np.array([.4, .2][:branches]); times = np.array([.2, .9][:branches])
    solve = ExteriorHistorySolve(ExteriorOperator(sparse.csr_matrix(k), core), c, d, alpha, times)
    return rng, k, core, c, d, alpha, times, solve


@pytest.mark.parametrize('branches', [0, 1, 2])
@pytest.mark.parametrize('contacts', [False, True])
def test_distributed_exterior_history_body_changes_and_energy_match_complete_equations(branches, contacts):
    rng, k, core, c, d, alpha, times, solve = problem(branches=branches, contacts=contacts)
    u = rng.normal(size=14)*.002; z = rng.normal(size=(branches, 14))*.003
    old_target = np.zeros(len(d))
    for index in range(16):
        h = [.01, .07, .2][index % 3]; target = rng.normal(size=len(d))*.01
        body = rng.normal(size=14)*.002 if index < 12 else np.zeros(14)
        before = u.copy(); old_z = z.copy(); ratios = times/(times+h)
        matrix = (1+ratios@alpha)*k+c.T@np.diag(d)@c
        force = body+c.T@(d*target)
        expected = np.linalg.solve(matrix, force+sum((a*r*(k@v) for a,r,v in zip(alpha,ratios,z)), np.zeros(14)))
        expected_z = (times[:, None]*z+h*expected)/(times[:, None]+h)
        result = solve.step(u, z, body, target, h, previous_target=old_target)
        np.testing.assert_allclose(result['displacement'], expected, atol=1e-13, rtol=1e-11)
        np.testing.assert_allclose(result['memories'], expected_z, atol=1e-13, rtol=1e-11)
        energy = lambda x, zs, t: .5*x@k@x+.5*(c@x-t)@(d*(c@x-t))+sum(.5*a*(x-v)@k@(x-v) for a,v in zip(alpha,zs))
        assert abs(result['energyNmm']-energy(expected, expected_z, target)) < 1e-13
        assert abs(result['oldEnergyNmm']-energy(before, old_z, old_target)) < 1e-13
        actuator = float((d*(target-c@expected))@(target-old_target))
        assert abs(result['targetWorkNmm']-actuator) < 1e-13
        assert abs(result['energyBalanceResidualNmm']) < 1e-13
        np.testing.assert_array_equal(u, before); np.testing.assert_array_equal(z, old_z)
        u, z = result['displacement'], result['memories']
        old_target = target.copy()


def test_batched_and_matrix_free_operator_match_independent_block_elimination():
    _, k, core, _, _, _, _, solve = problem()
    op = solve.operator; exterior = np.setdiff1d(np.arange(14), core)
    expected = k[np.ix_(core, core)]-k[np.ix_(core, exterior)]@np.linalg.solve(k[np.ix_(exterior, exterior)], k[np.ix_(exterior, core)])
    np.testing.assert_allclose(op.dense_schur(batch_columns=1), expected, atol=1e-13)
    np.testing.assert_allclose(op.apply(np.arange(5)*.01), expected@(np.arange(5)*.01), atol=1e-13)
    for columns in [[0], [3, 1, 4]]:
        np.testing.assert_allclose(op.columns(columns)[0], expected[:, columns], atol=1e-13)


@pytest.mark.parametrize('k,core,error', [
    (np.zeros((3,3)), [0], 'SYMMETRY'),
    (np.diag([1., -1., 2.]), [0], 'UNSUPPORTED_EXTERIOR'),
    (np.diag([-1., 1., 2.]), [0], 'UNSUPPORTED_CORE'),
    (np.ones((3,3)), [0], 'UNSUPPORTED_EXTERIOR'),
    (np.eye(3), [0, 0], 'CORE_PARTITION'),
    (np.eye(3), [0., 1.], 'CORE_PARTITION'),
])
def test_unsupported_tissue_cannot_be_hidden_by_contacts(k, core, error):
    with pytest.raises(ValueError, match=error):
        op = ExteriorOperator(k, core)
        ExteriorHistorySolve(op, np.array([[1., 0., 0.]]), [100.], [], [])


def test_outside_core_contacts_and_nonfinite_fields_refuse():
    _, k, core, c, d, alpha, times, solve = problem()
    bad = c.copy(); bad[0, solve.operator.exterior[0]] = 1.
    with pytest.raises(ValueError, match='CONTACT_OUTSIDE'):
        ExteriorHistorySolve(solve.operator, bad, d, alpha, times)
    for field in [np.full(14, np.nan), np.zeros(13)]:
        with pytest.raises(ValueError, match='DISPLACEMENT'):
            solve.step(field, np.zeros((2,14)), np.zeros(14), np.zeros(3), .1, previous_target=np.zeros(3))
    with pytest.raises(ValueError, match='BATCH_COLUMNS'):
        solve.operator.columns(np.zeros(65, dtype=int))


def test_oblique_heterogeneous_source_coupon_and_own_affine_admission():
    from resectionlab import surgical_continuum as c
    from resectionlab.surgical_scene import SurgicalScene
    labels = np.ones((5,5,3), np.uint16); labels[2:] = 2
    affine = np.array([[-1.,.2,0.,4.],[0.,1.1,.1,-3.],[0.,0.,.9,2.],[0.,0.,0.,1.]])
    materials = {'1': {'shearPa': 1000., 'poissonRatio': .3, 'relaxationShearPa': [500.]},
                 '2': {'shearPa': 2000., 'poissonRatio': .4, 'relaxationShearPa': [1000.]}}
    scene = SurgicalScene(labels, affine,
        [{'id': i, 'name': 'Generated region', 'material': 'soft_tissue'} for i in [1,2]],
        {'kind': 'generated oblique boundary control'}, {})
    mesh = c.voxel_mesh(scene, range(labels.size))
    grid = (mesh.nodes-affine[:3,3])@np.linalg.inv(affine[:3,:3]).T+.5
    fixed = np.any(np.isclose(grid, 0.) | np.isclose(grid, np.asarray(scene.shape)), axis=1)
    free = np.repeat(~fixed, 3); indices = np.flatnonzero(free)
    mu = np.array([materials[str(labels.flat[i])]['shearPa'] for i in mesh.cells])
    nu = np.array([materials[str(labels.flat[i])]['poissonRatio'] for i in mesh.cells])
    geometry = dict(cells=np.asarray(mesh.cells), shape=np.asarray(scene.shape), nodes=mesh.nodes,
                    elementNodes=mesh.element_nodes, affine=affine, freeDofs=free, muPa=mu, nu=nu)
    full = source_stiffness(geometry)
    canonical = c.assemble(scene, mesh, materials)[0][0]
    assert np.max(np.abs((full-canonical).data), initial=0.) < 1e-14
    cell = int(np.ravel_multi_index((2,2,1), scene.shape)); nodes = mesh.element_nodes[cell]
    contact = np.zeros((3, len(free)))
    for axis in range(3): contact[axis, 3*nodes+axis] = 1/8.
    core = np.searchsorted(indices, (3*nodes[:,None]+np.arange(3)).ravel())
    k = full[free][:,free].toarray(); contact = contact[:,free]; spring = np.full(3, .02)
    solve = ExteriorHistorySolve(ExteriorOperator(k, core), contact, spring, [.5], [.4])
    rng = np.random.default_rng(78); u = rng.normal(size=len(k))*.001; z = rng.normal(size=(1,len(k)))*.002
    target = np.array([.02, -.01, .003]); body = rng.normal(size=len(k))*1e-6; h = .025
    expected = np.linalg.solve((1+.5*.4/(.4+h))*k+contact.T@np.diag(spring)@contact,
                               body+contact.T@(spring*target)+.5*.4/(.4+h)*(k@z[0]))
    np.testing.assert_allclose(solve.step(u,z,body,target,h,previous_target=np.zeros(3))['displacement'], expected, atol=1e-12)
    # Moving both exports consistently would pass cross-input equality, but
    # violates each export's own declared affine and must refuse mechanics.
    geometry['nodes'] = mesh.nodes+np.array([.001,0.,0.])
    with pytest.raises(ValueError, match='SOURCE_AFFINE_CORNERS'):
        source_stiffness(geometry)


def test_release_preserves_distributed_history_with_separate_binding_energy_jump():
    rng, k, core, c, d, alpha, times, held = problem()
    u = rng.normal(size=14)*.002; z = rng.normal(size=(2,14))*.003
    target = np.array([.02, -.01, .03])
    result = held.step(u,z,np.zeros(14),target,.1,previous_target=np.zeros(3))
    u, z = result['displacement'], result['memories']
    released = ExteriorHistorySolve(held.operator, np.empty((0,14)), [], alpha, times)
    jump = released.contact_energy(u, [])-held.contact_energy(u, target)
    assert jump < 0
    after = released.step(u,z,np.zeros(14),[],.1,previous_target=[])
    assert abs(after['oldEnergyNmm']-result['energyNmm']-jump) < 1e-13
    assert abs(after['energyBalanceResidualNmm']) < 1e-13
    assert np.max(np.abs(after['memories'])) > 0


def test_sparse_support_certificate_spans_multiple_column_blocks():
    rng = np.random.default_rng(103); a = rng.normal(size=(149,149))
    k = a.T@a+2*np.eye(149); core = np.array([8, 71, 0, 140])
    calls = []
    op = ExteriorOperator(k, core, check=lambda: calls.append(True))
    exterior = np.setdiff1d(np.arange(149), core)
    expected = k[np.ix_(core,core)]-k[np.ix_(core,exterior)]@np.linalg.solve(k[np.ix_(exterior,exterior)],k[np.ix_(exterior,core)])
    np.testing.assert_allclose(op.dense_schur(), expected, atol=1e-11)
    assert len(calls) >= 5


def test_contact_rebind_reuses_only_fixed_tissue_and_keeps_distributed_history(monkeypatch):
    rng,k,core,c,d,alpha,times,held = problem()
    def forbidden(**kwargs): raise AssertionError('Contact change must not rebuild Schur')
    monkeypatch.setattr(held.operator,'dense_schur',forbidden)
    released = held.rebind(np.empty((0,14)),[])
    changed = held.rebind(c[::-1],d[::-1])
    assert released.prepared is changed.prepared is held.prepared
    u = rng.normal(size=14)*.001; z = rng.normal(size=(2,14))*.002
    body = rng.normal(size=14)*.001
    for binding in [held,released,changed,released]:
        target = rng.normal(size=len(binding.spring))*.01; h=.07; r=times/(times+h)
        a=(1+r@alpha)*k+binding.c.T@sparse.diags(binding.spring)@binding.c
        rhs=body+binding.c.T@(binding.spring*target)
        for aj,rj,zj in zip(alpha,r,z):rhs+=aj*rj*(k@zj)
        expected=np.linalg.solve(a,rhs)
        result=binding.step(u,z,body,target,h,previous_target=np.zeros(len(target)))
        np.testing.assert_allclose(result['displacement'],expected,atol=1e-13)
        u,z=result['displacement'],result['memories']
        body[1]+=.0001  # Shared derived-load cache must notice public in-place edits.
    other=ExteriorOperator(k,core)
    with pytest.raises(ValueError,match='ANOTHER_OPERATOR'):
        ExteriorHistorySolve(other,c,d,alpha,times,prepared=held.prepared)
