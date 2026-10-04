"""Independent analytic controls; no patient, HBE curves, mesher or solver."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('hbe_physics', Path(__file__).resolve().parents[1]/'scripts/mechanics_hbe_physics.py')
p = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)


def fixture():
    X = np.array([[-.004,-.004,0],[.004,-.004,0],[.004,.004,0],[-.004,.004,0],
                  [-.004,-.004,.006],[.004,-.004,.006],[.004,.004,.006],[-.004,.004,.006]])
    mesh = p.HexMesh(X, [[1,2,3,4,5,6,7,8]], [5,6,7,8], [1,2,3,4], radius_m=.004, height_m=.006)
    return X, mesh


def test_rigid_translation_rotation_zero_energy_and_source_snapshot():
    X, mesh = fixture()
    angle = .3
    R = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    state = mesh.deformation(X@R.T+[.02,-.04,.03], 1000)
    assert state['minimum_sampled_J'] == pytest.approx(1)
    assert state['energy_J'] == pytest.approx(0, abs=1e-17)
    X[:] = 99
    assert mesh.rest_nodes_m.max() == .006
    with pytest.raises(AttributeError): mesh.radius_m = 3


def test_affine_shear_exact_energy_volume_and_uniform_dilatation():
    X, mesh = fixture()
    moved = X.copy(); moved[:,0] += .1*X[:,1]
    volume = .008**2*.006
    assert mesh.rest_volume_m3 == pytest.approx(volume)
    assert mesh.deformation(moved,1000)['energy_J'] == pytest.approx(1000*.1**2/2*volume)
    scale=1.02; J=scale**3; K=149000/3
    expected=K/4*(J**2-1-2*np.log(J))*volume
    assert mesh.deformation(scale*X,1000)['energy_J'] == pytest.approx(expected, rel=1e-10)


def test_nonuniform_j_uses_cell_volume_average_not_mean_point_energy():
    X, mesh = fixture()
    moved=X.copy(); moved[6] += [.0007,.0002,.0001]
    state=mesh.deformation(moved,1000)
    # Independent finite-difference derivatives verify the readout's integration.
    gauss=p.GAUSS; dev=0.; point_vol=0.; weights=[]; Js=[]
    for q in gauss:
        eps=1e-5; derivatives=[]
        for axis in range(3):
            delta=np.eye(3)[axis]*eps
            derivatives.append((p.shape_functions(q+delta)-p.shape_functions(q-delta))/(2*eps))
        D=np.array(derivatives).T
        A=X.T@D; F=moved.T@D@np.linalg.inv(A); w=np.linalg.det(A); J=np.linalg.det(F)
        dev+=1000/2*(np.sum(F*F)*J**(-2/3)-3)*w
        point_vol+=p.K_OVER_MU*1000/4*(J**2-1-2*np.log(J))*w
        weights.append(w); Js.append(J)
    Jbar=np.dot(weights,Js)/sum(weights)
    expected=dev+sum(weights)*p.K_OVER_MU*1000/4*(Jbar**2-1-2*np.log(Jbar))
    assert state['energy_J'] == pytest.approx(expected, rel=1e-8)
    assert abs(state['energy_J']-(dev+point_vol)) > 1e-9


def test_probe_interpolation_affine_field_and_outside_rejected():
    X,mesh=fixture(); probes=p.fixed_probes(.004,.006)
    assert probes.shape==(75,3)
    mapping=mesh.probe_map(probes)
    A=np.array([[.01,.03,0],[0,-.02,.04],[0,0,.015]])
    moved=X+X@A.T+[.0001,-.0002,.0003]
    np.testing.assert_allclose(mesh.interpolate_displacement(moved,mapping),probes@A.T+[.0001,-.0002,.0003],atol=1e-15)
    with pytest.raises(ValueError,match='outside'): mesh.probe_map([[1,1,1]])


def test_inversion_and_corner_warp_rejected_even_if_logged_J_could_be_positive():
    X,mesh=fixture()
    bad=X.copy();bad[:,0]*=-1
    with pytest.raises(ValueError,match='Jacobian'):mesh.deformation(bad,1000)
    bad=X.copy();bad[6]=X[0]
    with pytest.raises(ValueError,match='Jacobian'):mesh.deformation(bad,1000)


def test_pure_couple_uses_current_lever_arm_and_negative_raw_sign():
    X,mesh=fixture(); moved=X.copy(); moved[4:,0]*=2
    raw=np.zeros_like(X);raw[4,1]=1;raw[5,1]=-1
    raw[0,1]=-2;raw[1,1]=2
    frame=mesh.read_frame(moved,raw,mu_Pa=1000,branch='torsion_pos',fraction=0)
    assert frame['applied_torque_Nm']==pytest.approx(.016)
    assert frame['net_moment_Nm']==pytest.approx([0,0,0],abs=1e-18)
    assert not frame['checks']['prescribed_motion']
    assert frame['applied_torque_Nm'] != -np.cross(X[4:],raw[4:])[:,2].sum()


def test_reaction_corruption_and_exact_rotation_motion():
    X,mesh=fixture();theta=.15*.006/.004
    moved=X.copy();c,s=np.cos(theta),np.sin(theta)
    moved[4:,:2]=X[4:,:2]@np.array([[c,s],[-s,c]])
    zero=np.zeros_like(X)
    assert mesh.read_frame(moved,zero,mu_Pa=1000,branch='torsion_pos',fraction=1)['checks']['prescribed_motion']
    zero[4,2]=.1
    result=mesh.read_frame(moved,zero,mu_Pa=1000,branch='torsion_pos',fraction=1)
    assert not result['passed']
    assert result['applied_force_N']==-.1


def test_work_energy_signed_compression_and_sign_corruption():
    x=np.linspace(0,-.001,61);y=10*x;energy=5*x*x
    kwargs=dict(mu_Pa=1000,radius_m=.004,height_m=.006)
    assert p.energy_work_check(x,y,energy,**kwargs)['passed']
    assert not p.energy_work_check(x,-y,energy,**kwargs)['passed']
    with pytest.raises(ValueError):p.energy_work_check(x[::-1],y[::-1],energy[::-1],**kwargs)


def test_refinement_and_scale_checks_preserve_failures():
    zero=np.zeros((61,75,3))
    def run(offset):return dict(response=np.linspace(0,.01,61)+offset,probe_displacements_m=zero)
    assert p.compare_refinement(run(.0003),run(.0001),run(0),response_scale=.016,radius_m=.004)['passed']
    assert not p.compare_refinement(run(0),run(.001),run(.0021),response_scale=.016,radius_m=.004)['passed']
    X,mesh=fixture();r=np.zeros_like(X);r[4,2]=.01;r[0,2]=-.01
    assert p.compare_scale(mesh,X,X,r,2*r,mu_Pa=1000)['passed']
    assert not p.compare_scale(mesh,X,X,r,1.9*r,mu_Pa=1000)['passed']


@pytest.mark.parametrize('change',[lambda c:c.__setitem__((0,0),2),lambda c:c.__setitem__((0,0),0)])
def test_invalid_or_duplicate_connectivity(change):
    X,_=fixture();cells=np.array([[1,2,3,4,5,6,7,8]])
    change(cells)
    with pytest.raises(ValueError):p.HexMesh(X,cells,[5,6,7,8],[1,2,3,4],radius_m=.004,height_m=.006)
