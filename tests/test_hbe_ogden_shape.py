"""Generated arithmetic/XML only; no measured inputs, solves or fitting.

HBE_OGDEN_SOURCE_ROOT selects this ignored stage for its first controlled run.
HBE_OGDEN_BASELINE_ROOT selects the exact preserved sources for byte regression.
Ordinary canonical tests use the tracked, independently pinned baseline source
without requiring ignored files, measured inputs or environment configuration.
"""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('HBE_OGDEN_SOURCE_ROOT',ROOT))


def load(name,path,expected=None):
    raw=path.read_bytes()
    if expected is not None:assert hashlib.sha256(raw).hexdigest()==expected
    spec=importlib.util.spec_from_loader(name,loader=None,origin=str(path))
    module=importlib.util.module_from_spec(spec);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


p=load('ogden_shape_physics',SOURCE/'scripts/mechanics_hbe_physics.py')
d=load('ogden_shape_deck',SOURCE/'scripts/mechanics_hbe_mesh.py')


@pytest.fixture(autouse=True)
def no_process_or_network(monkeypatch):
    import socket
    def forbidden(*args,**kwargs):raise AssertionError('Generated-only arithmetic/XML')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    for name in ('connect','connect_ex','bind','listen','accept','sendto'):
        monkeypatch.setattr(socket.socket,name,forbidden)
    monkeypatch.setattr(socket,'create_connection',forbidden)


def hex_fixture(module=p):
    X=np.array([[-.004,-.004,0],[.004,-.004,0],[.004,.004,0],[-.004,.004,0],
                [-.004,-.004,.006],[.004,-.004,.006],[.004,.004,.006],[-.004,.004,.006]])
    return X,module.HexMesh(X,[[1,2,3,4,5,6,7,8]],[5,6,7,8],[1,2,3,4],radius_m=.004,height_m=.006)


def deck_fixture():
    # Analytic rectangular cell; not an actual HBE specimen mesh or input file.
    X,_=hex_fixture();cells=np.arange(1,9)[None,:]
    faces=cells[0,d.HEX_FACES];boundaries={}
    for name,rows in (('bottom',faces[:1]),('top',faces[1:2]),('side',faces[2:])):
        boundaries[name]={'node_ids':sorted(set(rows.flat)),'faces_quad4':rows.tolist()}
    mesh={'rest_nodes_m':X.tolist(),'elements_hex8':cells.tolist(),'boundaries':boundaries}
    protocol={'geometry':{'radius_m':.004,'height_m':.006},
        'loading':{'base_equal_pseudotime_steps':4,'fine_repeat_equal_pseudotime_steps':8,'branches':[
            {'id':'compression','mode':'axial','strain_endpoint':-.15},
            {'id':'tension','mode':'axial','strain_endpoint':.15}]},
        'solver':{'symmetric_stiffness':1,'max_refs':15,'dtol':1e-8,'etol':1e-8,'rtol':1e-8}}
    return mesh,protocol


def analytic_piola(F,mu,alpha):
    # Independent analytic stress via B eigenvectors; production energy uses SVD.
    J=np.linalg.det(F);eigen,Q=np.linalg.eigh(F@F.T)
    powers=(np.sqrt(eigen)*J**(-1/3))**alpha
    kirchhoff=(2*mu/alpha)*(Q@np.diag(powers-np.mean(powers))@Q.T)
    return kirchhoff@np.linalg.inv(F).T


@pytest.mark.parametrize('alpha',[-32.,-8.,-.5,.5,2.,8.,32.])
@pytest.mark.parametrize('F',[
    np.eye(3),np.eye(3)*1.03,
    np.diag([1.15,.93,1.02]),
    np.array([[1.,.15,.04],[0.,1.,-.03],[0.,0.,1.]])])
def test_energy_derivative_matches_independent_piola(alpha,F):
    mu=1234.;expected=analytic_piola(F,mu,alpha);finite=np.zeros((3,3));step=1e-6
    for i in range(3):
        for j in range(3):
            delta=np.zeros((3,3));delta[i,j]=step
            finite[i,j]=(p.ogden_isochoric_density(F+delta,mu,ogden_alpha=alpha)
                         -p.ogden_isochoric_density(F-delta,mu,ogden_alpha=alpha))/(2*step)
    np.testing.assert_allclose(finite,expected,rtol=2e-6,atol=2e-6*mu)


@pytest.mark.parametrize('alpha',[-32.,-8.,-.5,.5,2.,8.,32.])
def test_rigid_zero_repeated_stretches_and_small_strain_shear(alpha):
    angle=.37;Q=np.array([[math.cos(angle),-math.sin(angle),0],
        [math.sin(angle),math.cos(angle),0],[0,0,1.]])
    assert p.ogden_isochoric_density(Q,1000.,ogden_alpha=alpha)==pytest.approx(0,abs=1e-11)
    assert p.ogden_isochoric_density(1.07*Q,1000.,ogden_alpha=alpha)==pytest.approx(0,abs=1e-11)
    gamma=1e-4;F=np.eye(3);F[0,1]=gamma
    energy=p.ogden_isochoric_density(F,1000.,ogden_alpha=alpha)
    assert 2*energy/gamma**2==pytest.approx(1000.,rel=3e-6)
    np.testing.assert_allclose(p.ogden_isochoric_density(Q@F,1000.,ogden_alpha=alpha),energy,rtol=3e-6,atol=1e-11)


@pytest.mark.parametrize('alpha',[-8.,-.5,.5,2.,8.])
def test_homogeneous_uniaxial_analytic_force_and_simple_shear(alpha):
    mu=1000.;step=1e-6
    for stretch in (.85,1.15):
        def energy(lam):
            return p.ogden_isochoric_density(np.diag([lam,lam**-.5,lam**-.5]),mu,ogden_alpha=alpha)
        derivative=(energy(stretch+step)-energy(stretch-step))/(2*step)
        expected=2*mu/alpha*(stretch**(alpha-1)-stretch**(-alpha/2-1))
        assert derivative==pytest.approx(expected,rel=2e-6,abs=.002)
    gamma=.13;F=np.eye(3);F[0,1]=gamma
    q=math.asinh(gamma/2)
    expected=8*mu/alpha**2*math.sinh(alpha*q/2)**2
    assert p.ogden_isochoric_density(F,mu,ogden_alpha=alpha)==pytest.approx(expected,rel=2e-10)


@pytest.mark.parametrize('alpha',[-8.,.5,8.])
def test_nonuniform_cell_J_bulk_term_and_mu_scaling(alpha):
    X,mesh=hex_fixture();moved=X.copy();moved[6]+=[.0007,.0002,.0001]
    result=mesh.deformation(moved,1000.,ogden_alpha=alpha)
    dev=0.;point_vol=0.;weights=[];Js=[]
    for q in p.GAUSS:
        step=1e-5
        D=np.array([(p.shape_functions(q+np.eye(3)[i]*step)
                     -p.shape_functions(q-np.eye(3)[i]*step))/(2*step) for i in range(3)]).T
        A=X.T@D;F=moved.T@D@np.linalg.inv(A);w=np.linalg.det(A);J=np.linalg.det(F)
        eig=np.linalg.eigvalsh(F.T@F);stretches=np.sqrt(eig)*J**(-1/3)
        dev+=2*1000/alpha**2*(sum(stretches**alpha)-3)*w
        point_vol+=p.K_OVER_MU*1000/4*(J*J-1-2*math.log(J))*w
        weights.append(w);Js.append(J)
    Jbar=sum(w*j for w,j in zip(weights,Js))/sum(weights)
    expected=dev+sum(weights)*p.K_OVER_MU*1000/4*(Jbar*Jbar-1-2*math.log(Jbar))
    assert result['energy_J']==pytest.approx(expected,rel=2e-8,abs=1e-15)
    assert abs(result['energy_J']-(dev+point_vol))>1e-9
    assert result['ogden_alpha']==alpha
    assert mesh.deformation(moved,2000.,ogden_alpha=alpha)['energy_J']==pytest.approx(2*result['energy_J'],rel=1e-14)


@pytest.mark.parametrize('alpha',[0.,True,False,float('nan'),float('inf'),'2',[2.]])
def test_invalid_exponent_refuses_energy_and_deck(alpha):
    with pytest.raises(ValueError,match='exponent'):
        p.ogden_isochoric_density(np.eye(3),1000.,ogden_alpha=alpha)
    mesh,protocol=deck_fixture()
    with pytest.raises(ValueError,match='exponent'):
        d.specimen_deck(mesh,'compression',4,1000.,protocol,ogden_alpha=alpha)


def test_invalid_gradient_and_unrepresentable_energy_refuse():
    for F in (np.diag([-1.,1.,1.]),np.diag([0.,1.,1.]),np.ones((2,3)),np.eye(3)*float('nan')):
        with pytest.raises(ValueError):p.ogden_isochoric_density(F,1000.,ogden_alpha=-8.)
    with pytest.raises(ValueError,match='Unrepresentable'):
        p.ogden_isochoric_density(np.diag([2.,.5,1.]),1000.,ogden_alpha=1e6)


@pytest.mark.parametrize('alpha',[-8.,.5,8.])
def test_only_first_exponent_and_receipt_change_in_deck(alpha):
    mesh,protocol=deck_fixture()
    xml,old=d.specimen_deck(mesh,'compression',4,1000.,protocol)
    changed,new=d.specimen_deck(mesh,'compression',4,1000.,protocol,ogden_alpha=alpha)
    first,second=ET.fromstring(xml),ET.fromstring(changed)
    assert float(second.findtext('Material/material/m1'))==alpha
    assert second.findtext('Material/material/c1')=='2000'
    assert second.findtext('Material/material/k')==first.findtext('Material/material/k')
    second.find('Material/material/m1').text='2'
    assert ET.tostring(first)==ET.tostring(second)
    for key in old:
        if key!='deck_sha256':assert new[key]==old[key]
    assert new['ogden_alpha']==alpha and new['native_execution_released'] is False
    assert new['baseline_protocol_amended'] is False


def test_alpha2_default_xml_metadata_and_energy_exact_historical_compatibility():
    baseline=os.environ.get('HBE_OGDEN_BASELINE_ROOT',
        str(ROOT/'artifacts/mechanics/hbe-ogden-shape-source-v1/baseline'))
    path=Path(baseline)
    oldp=load('old_physics',path/'mechanics_hbe_physics.py',
        '00b8b5c87696609178a673ed98bbbe706ec524b4a8a09d0f0a55a85f53fb345d')
    oldd=load('old_deck',path/'mechanics_hbe_mesh.py',
        'b00afbb1f5f203d6918c52472bb257b6be4b2a6c114cac9f264a938119e4a414')
    X,newmesh=hex_fixture();_,oldmesh=hex_fixture(oldp)
    for F in (np.eye(3),np.diag([1.1,.93,1.02]),np.array([[1.,.1,0],[0,1.,0],[0,0,1.]])):
        moved=X@F.T
        expected=oldmesh.deformation(moved,1000.)
        for actual in (newmesh.deformation(moved,1000.),newmesh.deformation(moved,1000.,ogden_alpha=2.)):
            assert set(actual)==set(expected)
            for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    mesh,protocol=deck_fixture()
    for branch in ('compression','tension'):
        expected=oldd.specimen_deck(mesh,branch,4,1000.,protocol)
        assert d.specimen_deck(mesh,branch,4,1000.,protocol)==expected
        assert d.specimen_deck(mesh,branch,4,1000.,protocol,ogden_alpha=2.)==expected
