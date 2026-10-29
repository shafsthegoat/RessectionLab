"""Analytical homogeneous physical-gradient and actual spring-work exports."""
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from resectionlab.surgical_continuum import voxel_mesh
from resectionlab.surgical_scene import SurgicalScene

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    api = runpy.run_path(str(scripts/'qualify_continuum_exterior.py'))
finally:
    sys.path.remove(str(scripts))


@pytest.mark.parametrize('affine', [np.eye(4), np.array([[-1.,.2,0.,4.],[0.,1.1,.1,-3.],[0.,0.,.9,2.],[0.,0.,0.,1.]])])
def test_probe_quadrature_matches_known_affine_gradient_and_stress(affine):
    scene = SurgicalScene(np.ones((2,1,1),np.uint16), affine,
        [{'id':1,'name':'Generated coupon','material':'soft_tissue'}], {'kind':'generated field control'}, {})
    mesh = voxel_mesh(scene, [0,1]); matrix = np.array([[.01,.002,0.],[0.,-.003,.001],[0.,0.,.004]])
    u = (mesh.nodes@matrix.T).ravel(); z = np.asarray([.25*u])
    geometry = dict(nodes=mesh.nodes,elementNodes=mesh.element_nodes,affine=affine,
                    muPa=np.full(2,1000.),nu=np.full(2,.3))
    rates, gradient, displacement = api['field_metrics'](geometry,np.arange(len(u)),u,z,np.array([0,1]),[.5])
    eps = (matrix+matrix.T)/2; lam = 2*1000.*.3/(1-2*.3)
    stress = 1.375*(2*1000.*eps+lam*np.trace(eps)*np.eye(3))
    volume = 2*abs(np.linalg.det(affine[:3,:3]))
    np.testing.assert_allclose(rates,[np.sum(eps**2)*volume,np.sum(stress**2)*volume],atol=1e-10,rtol=1e-12)
    assert abs(gradient-np.linalg.norm(matrix))<1e-12 and displacement>0


def test_complete_accounting_has_true_moving_spring_energy_and_work():
    # One equilibrium spring k=2, contact d=3, target0 -> .1, no Maxwell:
    # u_new=.06, contact stretch=-.04. Actuator work=.012; stored=.006;
    # BE numerical loss=.006. This distinguishes force-gauge accounting.
    from scipy import sparse
    r = api['full_accounting'](sparse.csr_matrix([[2.]]),sparse.csr_matrix([[1.]]),np.array([3.]),
        [],[],np.array([0.]),np.empty((0,1)),np.array([.06]),np.empty((0,1)),np.array([.1]),np.array([0.]),.1)
    assert abs(r['energyNmm']-.006)<1e-15
    assert abs(r['workNmm']-.012)<1e-15
    assert abs(r['numericalDissipationNmm']-.006)<1e-15
    assert abs(r['energyBalanceResidualNmm'])<1e-15


@pytest.mark.parametrize('branches', [0, 1, 2])
def test_nonaffine_oblique_fields_keep_corner_guard_and_all_maxwell_stresses(branches):
    # Direct element/point sums give a separate contraction oracle. A curved
    # Q1 field has different Gauss/corner values, so counting only Gauss points
    # for the gradient guard, or corners as integration volume, both fail.
    from resectionlab.surgical_continuum import element_operators
    affine = np.array([[-1.,.2,0.,4.],[0.,1.1,.1,-3.],[0.,0.,.9,2.],[0.,0.,0.,1.]])
    scene = SurgicalScene(np.ones((2,1,1), np.uint16), affine,
        [{'id':1,'name':'Generated coupon','material':'soft_tissue'}], {'kind':'generated nonaffine control'}, {})
    mesh = voxel_mesh(scene, [0,1]); rng = np.random.default_rng(182)
    u = rng.normal(size=mesh.nodes.size)*.01
    z = rng.normal(size=(branches,len(u)))*.02; alpha = np.array([.5,.2])[:branches]
    mu = np.array([900., 1500.]); nu = np.array([.2,.4]); gradients = element_operators(affine)[2]
    geometry = dict(nodes=mesh.nodes,elementNodes=mesh.element_nodes,affine=affine,muPa=mu,nu=nu)
    rates, guard, center = api['field_metrics'](geometry,np.arange(len(u)),u,z,np.array([0,1]),alpha)
    expected = np.zeros(2); maximum = 0.; gauss_max = 0.
    weight = abs(np.linalg.det(affine[:3,:3]))/8
    for row,nodes in enumerate(mesh.element_nodes):
        points = []
        for field in [u,*[u-memory for memory in z]]:
            values = field.reshape((-1,3))[nodes]
            points.append(np.array([sum((np.outer(values[a],g[a]) for a in range(8)),np.zeros((3,3))) for g in gradients]))
        maximum = max(maximum,float(np.linalg.norm(points[0],axis=(1,2)).max()))
        gauss_max = max(gauss_max,float(np.linalg.norm(points[0][:8],axis=(1,2)).max()))
        for point in range(8):
            strain = (points[0][point]+points[0][point].T)/2
            stress = np.zeros((3,3)); lame = 2*mu[row]*nu[row]/(1-2*nu[row])
            for coefficient,values in zip([1.,*alpha],points):
                eps = (values[point]+values[point].T)/2
                stress += coefficient*(2*mu[row]*eps+lame*np.trace(eps)*np.eye(3))
            expected += [np.sum(strain**2)*weight,np.sum(stress**2)*weight]
    np.testing.assert_allclose(rates,expected,atol=1e-10,rtol=1e-12)
    assert guard == pytest.approx(maximum,abs=1e-14) and maximum > gauss_max
    assert center == pytest.approx(np.linalg.norm(u.reshape((-1,3))[mesh.element_nodes].mean(axis=1),axis=1).max())
