"""Closed-form constitutive kinematics and mock log contracts, never solver data."""
import hashlib
import json

import numpy as np
import pytest

from scripts.mechanics_hbe_readout import read_run


def fixture(root,*,reaction_sign=1,corrupt_displacement=False,mu=1000.):
    R=.004;H=.00489159;K=149*mu/3;steps=60;times=np.linspace(0,1,steps+1)
    X=np.array([[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0],[-R,-R,H],[R,-R,H],[R,R,H],[-R,R,H]])
    bindings={}
    def save(key,data):
        path=root/(key+'.txt');path.write_text(data)
        bindings[key]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    mesh={'rest_nodes_m':X.tolist(),'elements_hex8':[[1,2,3,4,5,6,7,8]],'node_ids':list(range(1,9)),
          'element_ids':[1],'boundaries':{'top':{'node_ids':[5,6,7,8]},'bottom':{'node_ids':[1,2,3,4]}},
          'geometry':{'radius_m':R,'height_m':H},'mesh_N':4,'indexing':'one_based'}
    save('mesh',json.dumps(mesh));save('deck','analytical-log-contract-no-solver-deck')
    loading={'branch':'tension','steps':steps,'mu_Pa':mu,'K_Pa':K,'times':times.tolist(),'load_coordinate':(times*.15*H).tolist(),
             'load_coordinate_units':'m','prescribed_dofs':{'bottom':'xyz','top':'xyz'},'node_fields':'x;y;z;ux;uy;uz;Rx;Ry;Rz',
             'element_fields':'sx;sy;sz;sxy;syz;sxz;J;sed','raw_reaction_convention':'body_on_constraint',
             'min_residual_N2':(1e-10*mu*R**2)**2,'protocol_sha256':'1'*64,
             'mesh_sha256':bindings['mesh']['sha256'],'deck_sha256':bindings['deck']['sha256']}
    save('loading',json.dumps(loading));nodes=[];elements=[];solver=[]
    for step,t in enumerate(times):
        stretch=1+.15*t;moved=X.copy();moved[:,2]*=stretch;u=moved-X
        dW=mu/2*(2*stretch*stretch**(-2/3)-(2/3)*(2+stretch**2)*stretch**(-5/3))+K/2*(stretch-1/stretch)
        force=dW*(2*R)**2;raw=np.zeros_like(X);raw[:4,2]=force/4;raw[4:,2]=-force/4;raw*=reaction_sign
        if corrupt_displacement and step==30:u[6,0]+=.001
        values=np.column_stack((moved,u,raw))
        nodes.extend([f'*Step = {step}',f'*Time = {t:.12g}','*Data = mechanics_nodes_si'])
        nodes.extend(str(i)+','+','.join(f'{v:.12g}' for v in row) for i,row in enumerate(values,1))
        W=mu/2*((2+stretch**2)*stretch**(-2/3)-3)+K/4*(stretch**2-1-2*np.log(stretch))
        elements.extend([f'*Step = {step}',f'*Time = {t:.12g}','*Data = mechanics_elements_si',
                         f'1,0,0,0,0,0,0,{stretch:.12g},{W:.12g}'])
        if step:solver.extend([f'Nonlinear solution status: time= {t:.6g}',' residual 1e-6 1e-18 1e-14'])
    solver.append('N O R M A L T E R M I N A T I O N')
    for key,lines in [('nodes',nodes),('elements',elements),('solver',solver)]:save(key,'\n'.join(lines))
    return bindings


def run(root,bindings):
    return read_run(root,bindings,protocol_sha256='1'*64,expected_branch='tension',expected_mesh_N=4,
                    expected_steps=60,expected_mu_Pa=1000,retain_scale_primitives=True)


def test_complete_readout_independent_energy_reactions_and_initial_state(tmp_path):
    receipt,cache=run(tmp_path,fixture(tmp_path))
    assert receipt['passed']
    assert receipt['frame_count']==61
    assert receipt['applied_force_N'][-1]>0
    assert receipt['energy_J'][-1]>0
    assert receipt['logged_sed_used_for_energy_gate'] is False
    assert np.asarray(receipt['probe_displacements_m']).shape==(61,75,3)
    assert cache['current_nodes_m'].shape==(61,8,3)


def test_sign_flip_and_displacement_column_corruption_fail_numerical_gate(tmp_path):
    receipt,_=run(tmp_path,fixture(tmp_path,reaction_sign=-1))
    assert not receipt['passed'] and receipt['criteria']['work_energy']['actual']>1
    receipt,_=run(tmp_path,fixture(tmp_path,corrupt_displacement=True))
    assert not receipt['passed'] and receipt['criteria']['primitive_consistency']['actual']>1


def test_source_drift_and_loading_identity_rejected(tmp_path):
    bindings=fixture(tmp_path);(tmp_path/'nodes.txt').write_text('changed')
    with pytest.raises(ValueError,match='hash'):run(tmp_path,bindings)
    bindings=fixture(tmp_path)
    with pytest.raises(ValueError,match='identity'):
        read_run(tmp_path,bindings,protocol_sha256='1'*64,expected_branch='compression',expected_mesh_N=4,expected_steps=60,expected_mu_Pa=1000)


def test_arbitrary_positive_fitted_scale_uses_declared_residual_arithmetic(tmp_path):
    # This scale exposes the one-ulp difference between mu*R**2 and mu*R*R.
    # Both solver deck and readout must use the already-declared first ordering.
    mu = 123.456
    bindings = fixture(tmp_path, mu=mu)
    receipt, _ = read_run(
        tmp_path, bindings, protocol_sha256='1'*64, expected_branch='tension',
        expected_mesh_N=4, expected_steps=60, expected_mu_Pa=mu,
    )
    assert receipt['passed']
