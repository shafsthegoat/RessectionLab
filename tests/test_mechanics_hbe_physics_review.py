"""Independent polynomial-hex controls, without real curves or a solver."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "hbe_physics_review_target", Path(__file__).resolve().parents[1] / "scripts/mechanics_hbe_physics.py"
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)

RADIUS, HEIGHT = .004, .006
NATURAL_NODES = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])


def polynomial_fields(q):
    u, v, w = q
    r, h = RADIUS, HEIGHT
    rest = np.array([r*u*(1+.18*w), r*v*(1+.12*u), h*(1+w)/2])
    displacement = np.array([r*.08*u*v*w, r*.05*u*w, h*.02*v*w])
    rest_derivative = np.array([[r*(1+.18*w), 0, r*.18*u],
                                [r*.12*v, r*(1+.12*u), 0], [0, 0, h/2]])
    displacement_derivative = np.array([[r*.08*v*w, r*.08*u*w, r*.08*u*v],
                                        [r*.05*w, 0, r*.05*u], [0, h*.02*w, h*.02*v]])
    return rest, displacement, rest_derivative, displacement_derivative


def warped_fixture():
    fields = [polynomial_fields(q) for q in NATURAL_NODES]
    X = np.array([row[0] for row in fields])
    current = X + np.array([row[1] for row in fields])
    mesh = p.HexMesh(X, [[1,2,3,4,5,6,7,8]], [5,6,7,8], [1,2,3,4],
                     radius_m=RADIUS, height_m=HEIGHT)
    return X, current, mesh


def test_warped_rest_cell_uses_physical_quadrature_and_averaged_volume_energy():
    _, current, mesh = warped_fixture()
    sites, _ = np.polynomial.legendre.leggauss(2)
    weights, ratios, deviatoric = [], [], []
    mu, bulk = 1000., 149000/3
    for site in itertools.product(sites, repeat=3):
        _, _, A, B = polynomial_fields(site)
        F = (A+B) @ np.linalg.inv(A)
        stretches = np.linalg.svd(F, compute_uv=False)
        J = np.prod(stretches)
        weights.append(np.linalg.det(A))
        ratios.append(J)
        deviatoric.append(mu/2*(np.sum((stretches/np.cbrt(J))**2)-3))
    weights, ratios, deviatoric = map(np.asarray, (weights, ratios, deviatoric))
    volume = weights.sum()
    weighted_J = np.dot(weights, ratios)/volume
    U = lambda J: bulk/4*(J*J-1-2*np.log(J))
    expected = np.dot(weights, deviatoric) + volume*U(weighted_J)
    output = mesh.deformation(current, mu)
    assert np.ptp(weights) > 1e-9  # This fixture exercises nonuniform rest weights.
    assert abs(weighted_J-ratios.mean()) > 1e-6
    assert output['energy_J'] == pytest.approx(expected, rel=2e-11, abs=1e-16)
    assert output['cell_volume_ratios'][0] == pytest.approx(weighted_J, abs=1e-14)
    assert mesh.rest_volume_m3 == pytest.approx(volume, abs=1e-20)
    wrong_pointwise = np.dot(weights, deviatoric+U(ratios))
    wrong_equal_weights = volume*(deviatoric.mean()+U(ratios.mean()))
    assert abs(expected-wrong_pointwise) > 1e-9
    assert abs(expected-wrong_equal_weights) > 1e-9


def test_warped_inverse_map_recovers_trilinear_displacement_and_rejects_bbox_only_point():
    _, current, mesh = warped_fixture()
    natural = [[-.6,.4,-.3],[.7,-.2,.8],[0,0,0]]
    fields = [polynomial_fields(q) for q in natural]
    points = np.array([row[0] for row in fields])
    expected = np.array([row[1] for row in fields])
    mapping = mesh.probe_map(points)
    np.testing.assert_allclose(mesh.interpolate_displacement(current, mapping), expected,
                               atol=2e-13, rtol=0.)
    translation = np.array([.0002,-.0001,.0003])
    np.testing.assert_allclose(mesh.interpolate_displacement(mesh.rest_nodes_m+translation, mapping),
                               np.broadcast_to(translation, expected.shape), atol=1e-17, rtol=0.)
    # Inside the node bounding box, outside the narrowing bottom cross-section.
    outside = [RADIUS*1.05, 0., HEIGHT*.05]
    with pytest.raises(ValueError, match='outside'):
        mesh.probe_map([outside])


@pytest.mark.parametrize('bad_scale', [float('inf'), float('nan'), -1., 0.])
def test_work_and_refinement_cannot_accept_invalid_physical_scale(bad_scale):
    with pytest.raises(ValueError):
        p.energy_work_check([0.,1.],[0.,-100.],[0.,1.],mu_Pa=bad_scale,
                            radius_m=RADIUS,height_m=HEIGHT)
    records = [{'response':np.array([0.,value]), 'probe_displacements_m':np.zeros((2,75,3))}
               for value in (100.,10.,1.)]
    with pytest.raises(ValueError):
        p.compare_refinement(*records,response_scale=bad_scale,radius_m=RADIUS)
    X, _, mesh = warped_fixture()
    with pytest.raises(ValueError):
        p.compare_scale(mesh,X,X,np.zeros_like(X),np.ones_like(X),mu_Pa=bad_scale)


def test_overflowed_work_scale_cannot_convert_a_failed_energy_check_to_pass():
    with pytest.raises(ValueError):
        p.energy_work_check([0.,1.],[0.,-100.],[0.,1.],mu_Pa=1e300,
                            radius_m=1e100,height_m=1.)
