"""Pure numerical/XML/output controls; no FEBio process or patient data."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mpc_fixture',ROOT/'scripts/mechanics_patient_constraints.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def primitive_logs(case, corrupt=None):
    """Constructed oracle text only; never a measured solver receipt."""
    X,E,_=v.fixture_mesh();node_text=[];element_text=[];solver=['N O R M A L   T E R M I N A T I O N']
    for step,t in enumerate(v.TIMES):
        if case=='tet10_affine': current=X@v.affine_F(t).T; reactions=v.affine_reactions(t)
        elif case=='mpc_translation': current=X+t*v.targets(case)[0];reactions=np.zeros_like(X)
        else:
            H,p,_,_=v.observation_operator();current=X.copy();current[p]+=np.linalg.solve(H[:,p],t*v.targets(case));reactions=np.zeros_like(X)
        nodes=np.column_stack([current,current-X,reactions]);elements=v.state_mechanics(current)['primitive']
        if corrupt:corrupt(nodes,elements,t)
        for text,values,name in ((node_text,nodes,'mechanics_nodes_si'),(element_text,elements,'mechanics_elements_si')):
            text.extend([f'*Step = {step}',f'*Time = {t}',f'*Data = {name}'])
            text.extend(str(i)+','+','.join(format(float(x),'.12g') for x in row) for i,row in enumerate(values,1))
        if t:solver.extend([f'Nonlinear solution status: time= {t}',' residual 1e-8 1e-26 1e-18'])
    return '\n'.join(node_text),'\n'.join(element_text),'\n'.join(solver)


def test_mesh_is_six_positive_tet10_with_one_genuinely_free_interior_node():
    X,E,b=v.fixture_mesh()
    assert X.shape==(27,3) and E.shape==(6,10) and len(b)==26
    inner=[i for i in range(27) if i not in b]
    np.testing.assert_array_equal(X[inner],[[.005,.005,.005]])
    for e in E:
        assert np.linalg.det((X[e[1:4]]-X[e[0]]).T)>0
        for k,(i,j) in enumerate(v.EDGES):np.testing.assert_array_equal(X[e[k+4]],(X[e[i]]+X[e[j]])/2)
    assert v.state_mechanics(X)['minimum_sampled_J']==pytest.approx(1,abs=1e-14)


def test_shape_partition_nodal_cardinality_and_affine_reproduction():
    points=np.r_[np.zeros((1,3)),np.eye(3)]
    points=np.vstack([points,*[(points[i]+points[j])/2 for i,j in v.EDGES]])
    for i,p in enumerate(points):np.testing.assert_array_equal(v.shape(p)[0],np.eye(10)[i])
    for p in ([.1,.2,.3],[.25,.25,.25],[0,0,0]):
        N,dN=v.shape(p)
        np.testing.assert_allclose(N@points,p,rtol=0,atol=2e-16)
        np.testing.assert_allclose(points.T@dN,np.eye(3),rtol=0,atol=5e-16)
        np.testing.assert_allclose(dN.sum(axis=0),0,rtol=0,atol=5e-16)


def test_mixed_affine_law_matches_independently_expressed_principal_energy():
    F=v.affine_F(1);J,W,sigma=v.response(F);s=np.linalg.svd(F,compute_uv=False)
    assert J!=pytest.approx(1)
    assert abs(sigma[0,1])>0 and abs(sigma[1,2])>0
    expected=v.MU_PA/2*(sum((s/J**(1/3))**2)-3)+v.K_PA/4*(J*J-1-2*np.log(J))
    assert W==pytest.approx(expected,abs=1e-12)
    h=1e-6;P=J*sigma@np.linalg.inv(F).T
    for i in range(3):
        for j in range(3):
            D=np.zeros((3,3));D[i,j]=1
            derivative=(v.response(F+h*D)[1]-v.response(F-h*D)[1])/(2*h)
            assert derivative==pytest.approx(P[i,j],rel=2e-7,abs=5e-6)


def test_affine_surface_reactions_have_correct_virtual_work_and_sign():
    X,_,_=v.fixture_mesh();F=v.affine_F(1);J,_,sigma=v.response(F);P=J*sigma@np.linalg.inv(F).T
    delta=np.array([[.02,.01,0],[-.01,.03,.01],[0,0,-.01]])
    raw=v.affine_reactions(1)
    assert -np.sum(raw*(X@delta.T))==pytest.approx(np.sum(P*delta)*v.SIDE_M**3,rel=1e-12,abs=1e-18)
    np.testing.assert_allclose(raw.sum(axis=0),0,rtol=0,atol=1e-16)


@pytest.mark.parametrize('case',v.CASES)
def test_decks_bind_element_material_domain_steps_and_no_extra_mpc_anchor(case):
    xml=ET.fromstring(v.deck_xml(case));domain=xml.find('MeshDomains/SolidDomain')
    assert domain.attrib['elem_type']=='TET10G8' and domain.attrib['type']=='elastic-solid'
    assert xml.findtext('Material/material/k')==format(v.K_PA,'.17g')
    assert xml.findtext('Control/time_stepper/max_retries')=='0'
    assert xml.findtext('Control/plot_level')=='PLOT_NEVER'
    assert xml.findtext('Control/time_steps')=='4'
    bc=xml.findall('Boundary/bc')
    if case=='tet10_affine':assert len(bc)==78 and all(b.attrib['type']=='prescribed displacement' for b in bc)
    else:
        assert len(bc)==9 and all(b.attrib['type']=='linear constraint' for b in bc)
        parents={(b.findtext('node'),b.findtext('dof')) for b in bc}
        children={(c.findtext('node'),c.findtext('dof')) for b in bc for c in b.findall('child_dof')}
        assert len(parents)==9 and not parents.intersection(children)
        assert not xml.findall('.//bc[@type="zero displacement"]')


def test_h_rigid_mode_rank_and_all_feasible_basis_directions():
    X,_,_=v.fixture_mesh();H,_,_,T=v.observation_operator()
    np.testing.assert_array_equal(H@T,np.zeros((3,24)))
    assert np.linalg.matrix_rank(H)==3
    rows=[]
    for x,y,z in (H@X)/v.SIDE_M:
        skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        rows.append(np.column_stack([np.eye(3),-skew]))
    assert np.linalg.matrix_rank(np.vstack(rows))==6
    audit=v.feasible_virtual_work(X+v.targets('mpc_translation')[0])
    assert audit['passed'] and audit['direction_count']==72


def test_compatible_but_unrelaxed_displacement_fails_virtual_work():
    X,_,_=v.fixture_mesh();H,p,_,_=v.observation_operator();current=X.copy()
    current[p]+=np.linalg.solve(H[:,p],v.targets('mpc_nonrigid'))
    np.testing.assert_allclose(H@(current-X),v.targets('mpc_nonrigid'),rtol=0,atol=1e-18)
    audit=v.feasible_virtual_work(current)
    assert not audit['passed'] and audit['max_abs_finest_N']>100*v.TOL['virtual_work_N']


@pytest.mark.parametrize('case',['tet10_affine','mpc_translation'])
def test_constructed_complete_logs_accept_only_analytical_expected_fields(case):
    checked=v.check_outputs(case,*primitive_logs(case));assert checked['passed']
    assert len(checked['states'])==5


def test_raw_parent_zero_reactions_cannot_make_nonstationary_mpc_pass():
    checked=v.check_outputs('mpc_nonrigid',*primitive_logs('mpc_nonrigid'))
    assert not checked['passed']
    assert all(s['checks']['original_Hu_equals_d'] for s in checked['states'])
    assert not checked['states'][-1]['checks']['independent_feasible_virtual_work']
    assert not np.any(checked['states'][-1]['raw_parent_reactions_N'])


@pytest.mark.parametrize('kind',['stress','energy','constraint','reaction'])
def test_corrupted_actual_fields_are_rejected(kind):
    case='tet10_affine' if kind=='reaction' else 'mpc_translation'
    def corrupt(n,e,t):
        if not t:return
        if kind=='stress':e[0,0]+=.1
        elif kind=='energy':e[0,7]+=.1
        elif kind=='reaction':n[0,6]+=.001
        else:n[0,3]+=.001
    assert not v.check_outputs(case,*primitive_logs(case,corrupt))['passed']


def test_initial_state_and_fixed_relative_residual_evidence_required():
    texts=primitive_logs('tet10_affine')
    altered=texts[2].replace('1e-18','1e-10')
    assert not v.check_outputs('tet10_affine',texts[0],texts[1],altered)['passed']
    no_initial=texts[0][texts[0].index('*Step = 1'):]
    with pytest.raises(ValueError,match='initial'):v.check_outputs('tet10_affine',no_initial,texts[1],texts[2])


def test_inverted_actual_geometry_is_rejected():
    X,_,_=v.fixture_mesh();X[:,0]*=-1
    with pytest.raises(ValueError,match='Jacobian'):v.state_mechanics(X)
