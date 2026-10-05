#!/usr/bin/env python3
"""Three fixed tet10/MPC software fixtures and output audit; never solve FEBio.

The only volume mesh is the explicit analytic cube below. Nodal averages here
isolate MPC implementation; they are NOT the proposed patient tent operator.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

_PATCH_PATH = Path(__file__).resolve().with_name('mechanics_febio_verification.py')
_spec = importlib.util.spec_from_file_location('_frozen_patch_primitives', _PATCH_PATH)
patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch)
assert Path(patch.__file__).resolve() == _PATCH_PATH

VERSION = 'mechanics-patient-constraints-runtime-v1'
FEBIO_COMMIT = '32ae206ff4881dfb54f62296cd1558e58ed9fcc6'
SIDE_M = .01
MU_PA = 1000.0  # Arbitrary numerical scale; never a fitted tissue property.
K_PA = MU_PA * 29 / 3
TIMES = (0., .25, .5, .75, 1.)
CASES = ('tet10_affine', 'mpc_translation', 'mpc_nonrigid')
CAPS = {'aggregate_seconds': 60, 'process_group_rss_bytes': 3 * 1024**3,
        'numerical_threads': 1, 'maximum_cases': 3, 'time_steps': 4,
        'retries': 0}
TOL = {'position_m': 1e-10, 'constraint_m': 1e-10, 'force_N': 1e-8,
       'stress_Pa': 1e-5, 'density_Pa': 1e-7, 'jacobian': 1e-7,
       'relative': 2e-6, 'minimum_sampled_J': .2,
       'virtual_work_N': 1e-7, 'fd_last_change_N': 2e-8,
       'fd_convergence_roundoff_N': 1e-9}
FD_STEPS_M = (2e-7, 1e-7, 5e-8)
RESIDUAL_RTOL = 1e-10
RESIDUAL_FLOOR_N2 = 1e-20
# Pinned FETet10 order: four vertices then edges12,23,31,14,24,34.
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def fixture_mesh():
    vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                         [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float) * SIDE_M
    linear = ((0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6))
    nodes = list(vertices); mids = {}; elements = []
    for tet in linear:
        element = list(tet)
        for i, j in EDGES:
            edge = tuple(sorted((tet[i], tet[j])))
            if edge not in mids:
                mids[edge] = len(nodes)
                nodes.append((vertices[edge[0]] + vertices[edge[1]]) / 2)
            element.append(mids[edge])
        elements.append(element)
    X = np.asarray(nodes)
    boundary = np.where(np.any((X == 0) | (X == SIDE_M), axis=1))[0]
    return X, np.asarray(elements, int), boundary


def shape(rst):
    l = np.r_[1 - sum(rst), rst]
    dl = np.array([[-1.,-1.,-1.],[1,0,0],[0,1,0],[0,0,1]])
    N = np.r_[l * (2*l - 1), [4*l[i]*l[j] for i,j in EDGES]]
    dN = np.vstack(((4*l[:,None]-1)*dl,
                   [4*(l[i]*dl[j]+l[j]*dl[i]) for i,j in EDGES]))
    return N, dN


def gauss_rule():
    # Exact constants used by pinned FEBio v4.13 FETet10G8, not a new rule.
    a, b, c, d = .0158359099, .3280546970, .6791431780, .1069522740
    points = np.array([[a,b,b],[b,a,b],[b,b,a],[b,b,b],
                       [c,d,d],[d,c,d],[d,d,c],[d,d,d]])
    weights = np.r_[np.full(4,.138527967/6),np.full(4,.111472033/6)]
    return points, weights


def observation_operator():
    X, _, _ = fixture_mesh()
    center = int(np.flatnonzero(np.all(X == SIDE_M/2, axis=1))[0])
    parents = np.array([0, 1, 3], int)
    H = np.zeros((3, len(X)))
    H[np.arange(3), parents] = .5
    H[:, center] = .5
    free = np.array([i for i in range(len(X)) if i not in parents])
    A = -np.linalg.solve(H[:, parents], H[:, free])
    T = np.zeros((len(X), len(free))); T[free] = np.eye(len(free)); T[parents] = A
    return H, parents, free, T


def targets(case):
    if case == 'mpc_translation':
        return np.tile([.0003,-.0002,.0001], (3,1))
    if case == 'mpc_nonrigid':
        return np.array([[0.,0.,0.],[.00012,.00002,0.],[-.00001,-.00006,.00003]])
    raise ValueError('Not a declared MPC case')


def affine_F(time):
    final = np.array([[1.03,.02,0.],[0.,.99,.01],[0.,0.,1.01]])
    return np.eye(3) + time * (final - np.eye(3))


def response(F):
    F = np.asarray(F, float)
    if F.shape[-2:] != (3,3) or not np.isfinite(F).all():
        raise ValueError('Finite deformation gradient required')
    J = np.linalg.det(F)
    if np.any(J <= 0): raise ValueError('Nonpositive actual Jacobian')
    B = F @ np.swapaxes(F,-1,-2); tr = np.trace(B,axis1=-2,axis2=-1)
    W = MU_PA/2*(J**(-2/3)*tr-3) + K_PA/4*(J*J-1-2*np.log(J))
    sigma = MU_PA*J[...,None,None]**(-5/3)*(B-tr[...,None,None]/3*np.eye(3))
    sigma += (K_PA/2*(J-1/J))[...,None,None]*np.eye(3)
    return J, W, sigma


def integration_geometry():
    X, E, _ = fixture_mesh(); points, weights = gauss_rule()
    derivatives = np.stack([shape(q)[1] for q in points])
    jac0 = np.einsum('eni,qnj->eqij',X[E],derivatives)
    det0 = np.linalg.det(jac0)
    if np.any(det0 <= 0): raise ValueError('Invalid fixed reference mesh')
    grad = np.einsum('qni,eqij->eqnj',derivatives,np.linalg.inv(jac0))
    return E, grad, det0*weights


def state_mechanics(current):
    X, E, _ = fixture_mesh()
    current = np.asarray(current,float)
    if current.shape != X.shape or not np.isfinite(current).all(): raise ValueError('Complete actual node positions required')
    _, grad, weights = integration_geometry()
    F = np.einsum('eni,eqnj->eqij',current[E],grad)
    J, W, sigma = response(F)
    # Actual nodal/corner/centroid Jacobians add witnesses beyond solver points.
    extra = [q for q in np.eye(4)[:,1:]] + [[.25,.25,.25]]
    extra += [(.5*np.eye(4)[i]+.5*np.eye(4)[j])[1:] for i,j in EDGES]
    extra_derivatives = np.stack([shape(q)[1] for q in extra])
    jac0 = np.einsum('eni,qnj->eqij',X[E],extra_derivatives)
    jac = np.einsum('eni,qnj->eqij',current[E],extra_derivatives)
    extra_J = np.linalg.det(jac)/np.linalg.det(jac0)
    if np.any(extra_J <= 0): raise ValueError('Nonpositive extra-point Jacobian')
    s = sigma.mean(axis=1)  # FEBio element primitive is unweighted IP average.
    primitive = np.column_stack([s[:,0,0],s[:,1,1],s[:,2,2],s[:,0,1],s[:,1,2],s[:,0,2],J.mean(axis=1),W.mean(axis=1)])
    return {'F':F,'primitive':primitive,'energy_J':float(np.sum(W*weights)),
            'minimum_sampled_J':float(min(J.min(),extra_J.min()))}


def energy(current, prepared=None):
    E, grad, weights = prepared if prepared is not None else integration_geometry()
    F = np.einsum('eni,eqnj->eqij',np.asarray(current)[E],grad)
    J = np.linalg.det(F)
    if np.any(J <= 0): raise ValueError('Nonpositive perturbed Jacobian')
    tr = np.sum(F*F,axis=(-2,-1))
    W = MU_PA/2*(J**(-2/3)*tr-3) + K_PA/4*(J*J-1-2*np.log(J))
    return float(np.sum(W*weights))


def feasible_virtual_work(current):
    """Energy-only audit in ALL 72 feasible basis directions, no equilibrium solve."""
    _, _, _, T = observation_operator(); prepared = integration_geometry()
    derivatives = []
    for column, axis in itertools.product(range(T.shape[1]), range(3)):
        direction = np.zeros_like(current); direction[:,axis] = T[:,column]
        direction /= np.max(np.linalg.norm(direction,axis=1))
        values = [(energy(current+h*direction,prepared)-energy(current-h*direction,prepared))/(2*h)
                  for h in FD_STEPS_M]
        derivatives.append(values)
    values = np.asarray(derivatives)
    first_change = np.abs(values[:,1]-values[:,0]); last_change = np.abs(values[:,2]-values[:,1])
    checks = {'all_finest_directions_stationary':bool(np.max(np.abs(values[:,-1])) <= TOL['virtual_work_N']),
              'last_step_change_bounded':bool(last_change.max() <= TOL['fd_last_change_N']),
              'step_convergence_with_roundoff':bool(np.all(last_change <= first_change+TOL['fd_convergence_roundoff_N']))}
    return {'passed':all(checks.values()),'checks':checks,'direction_count':len(values),
            'steps_m':list(FD_STEPS_M),'directional_derivatives_N':values.tolist(),
            'max_abs_finest_N':float(np.abs(values[:,-1]).max()),'max_last_change_N':float(last_change.max())}


def affine_reactions(time):
    X, E, _ = fixture_mesh(); F = affine_F(time); J, _, sigma = response(F)
    P = J * sigma @ np.linalg.inv(F).T
    faces = {}
    for element in E:
        for local in itertools.combinations(range(4),3):
            vertices = tuple(int(element[i]) for i in local)
            faces.setdefault(tuple(sorted(vertices)),[]).append((element,vertices))
    result = np.zeros_like(X)
    for entries in faces.values():
        if len(entries) != 1: continue
        element, vertices = entries[0]; coords = X[list(vertices)]
        area_vector = np.cross(coords[1]-coords[0],coords[2]-coords[0])/2
        if area_vector @ (coords.mean(axis=0)-X[element[:4]].mean(axis=0)) < 0: area_vector *= -1
        # Quadratic triangular vertex shape integrals are zero; each edge is A/3.
        for a,b in itertools.combinations(vertices,2):
            midpoint = (X[a]+X[b])/2
            node = int(np.flatnonzero(np.all(X==midpoint,axis=1))[0])
            result[node] -= P @ area_vector / 3
    return result


def deck_xml(case):
    if case not in CASES: raise ValueError('Only three declared numerical fixtures exist')
    child = patch._child; X,E,boundary=fixture_mesh()
    root=ET.Element('febio_spec',version='4.0')
    root.append(ET.Comment('Analytical software fixture only; nodal averages are NOT patient tent kernels; SI units'))
    child(root,'Module',type='solid'); control=child(root,'Control')
    for k,v in {'analysis':'STATIC','time_steps':4,'step_size':.25,'plot_level':'PLOT_NEVER','output_level':'OUTPUT_MAJOR_ITRS','output_stride':1}.items(): child(control,k,v)
    timestep=child(control,'time_stepper',type='default')
    for k,v in {'max_retries':0,'dtmin':.25,'dtmax':.25}.items():child(timestep,k,v)
    solver=child(control,'solver',type='solid')
    for k,v in {'symmetric_stiffness':1,'max_refs':20,'dtol':1e-10,'etol':1e-10,'rtol':RESIDUAL_RTOL,'min_residual':RESIDUAL_FLOOR_N2}.items():child(solver,k,v)
    child(solver,'qn_method',type='BFGS');child(solver,'linear_solver',type='skyline')
    mat=child(child(root,'Material'),'material',id='1',name='numerical_ogden',type='Ogden')
    for i in range(1,7):child(mat,f'c{i}',2*MU_PA if i==1 else 0);child(mat,f'm{i}',2)
    child(mat,'k',format(K_PA,'.17g'));child(mat,'pressure_model',1)
    mesh=child(root,'Mesh');nodes=child(mesh,'Nodes',name='fixture_nodes')
    for i,x in enumerate(X,1):child(nodes,'node',','.join(format(v,'.17g') for v in x),id=str(i))
    group=child(mesh,'Elements',type='tet10',name='fixture')
    for i,e in enumerate(E,1):child(group,'elem',','.join(str(n+1) for n in e),id=str(i))
    if case=='tet10_affine':
        for i in boundary:child(mesh,'NodeSet',i+1,name=f'node_{i+1}')
    child(child(root,'MeshDomains'),'SolidDomain',name='fixture',mat='numerical_ogden',elem_type='TET10G8',type='elastic-solid')
    bc_root=child(root,'Boundary')
    if case=='tet10_affine':
        displacement=X@affine_F(1).T-X
        for i in boundary:
            for axis,label in enumerate('xyz'):
                bc=child(bc_root,'bc',type='prescribed displacement',node_set=f'node_{i+1}')
                child(bc,'dof',label);child(bc,'value',format(displacement[i,axis],'.17g'),lc='1');child(bc,'relative',0)
    else:
        H,parents,free,T=observation_operator();offset=np.linalg.solve(H[:,parents],targets(case))
        for row,parent in enumerate(parents):
            for axis,label in enumerate('xyz'):
                bc=child(bc_root,'bc',type='linear constraint');child(bc,'node',parent+1);child(bc,'dof',label)
                child(bc,'offset',format(offset[row,axis],'.17g'),lc='1')
                for col,node in enumerate(free):
                    if T[parent,col] != 0:
                        term=child(bc,'child_dof');child(term,'node',node+1);child(term,'dof',label);child(term,'value',format(T[parent,col],'.17g'))
    curve=child(child(root,'LoadData'),'load_controller',id='1',type='loadcurve');child(curve,'interpolate','LINEAR')
    pts=child(curve,'points');child(pts,'pt','0,0');child(pts,'pt','1,1')
    log=child(child(root,'Output'),'logfile',file=f'{case}.log')
    child(log,'node_data',data=patch.NODE_FIELDS,name='mechanics_nodes_si',file=f'{case}.nodes.log',delim=',')
    child(log,'element_data',data=patch.ELEMENT_FIELDS,name='mechanics_elements_si',file=f'{case}.elements.log',delim=',')
    ET.indent(root,space='  ');return ET.tostring(root,encoding='unicode',xml_declaration=True)+'\n'


def check_outputs(case,node_text,element_text,solver_text):
    if case not in CASES: raise ValueError('Undeclared case')
    X,E,_=fixture_mesh()
    nodes=patch.parse_data_log(node_text,field_count=9,record_name='mechanics_nodes_si',item_count=len(X))
    elements=patch.parse_data_log(element_text,field_count=8,record_name='mechanics_elements_si',item_count=len(E))
    if [s['time'] for s in nodes]!=list(TIMES) or [s['time'] for s in elements]!=list(TIMES):
        raise ValueError('Actual initial and four states required')
    solver=patch.check_solver_log(solver_text)
    required_consistent=all(np.isclose(v[2],v[0]*RESIDUAL_RTOL,rtol=2e-5,atol=1e-30)
                            for v in solver['last_squared_force_residual_norms_N2'].values())
    rows=[]
    for ns,es in zip(nodes,elements):
        t=ns['time']; values=ns['values']; current=values[:,:3]; displacement=values[:,3:6]
        mechanics=state_mechanics(current); primitive=es['values']
        checks={'displacement_position_consistency':bool(np.allclose(current-X,displacement,rtol=0,atol=TOL['position_m'])),
                'actual_jacobians':mechanics['minimum_sampled_J']>=TOL['minimum_sampled_J'],
                'stress_from_actual_deformation':bool(np.allclose(primitive[:,:6],mechanics['primitive'][:,:6],rtol=TOL['relative'],atol=TOL['stress_Pa'])),
                'energy_from_actual_deformation':bool(np.allclose(primitive[:,7],mechanics['primitive'][:,7],rtol=TOL['relative'],atol=TOL['density_Pa'])),
                'J_from_actual_deformation':bool(np.allclose(primitive[:,6],mechanics['primitive'][:,6],rtol=TOL['relative'],atol=TOL['jacobian']))}
        row={'time':t,'checks':checks,'minimum_actual_sampled_J':mechanics['minimum_sampled_J'],'reconstructed_energy_J':mechanics['energy_J']}
        if case=='tet10_affine':
            checks['affine_all_nodes_including_free_center']=bool(np.allclose(current,X@affine_F(t).T,rtol=0,atol=TOL['position_m']))
            checks['signed_reactions']=bool(np.allclose(values[:,6:],affine_reactions(t),rtol=TOL['relative'],atol=TOL['force_N']))
            row['expected_energy_J']=float(response(affine_F(t))[1]*SIDE_M**3)
            checks['homogeneous_energy']=bool(np.isclose(mechanics['energy_J'],row['expected_energy_J'],rtol=TOL['relative'],atol=1e-13))
        else:
            H,parents,_,_=observation_operator(); residual=H@displacement-t*targets(case)
            row['maximum_constraint_error_m']=float(np.abs(residual).max());checks['original_Hu_equals_d']=row['maximum_constraint_error_m']<=TOL['constraint_m']
            row['raw_parent_reactions_N']=values[parents,6:].tolist()
            row['raw_parent_reactions_scope']='retained only; eliminated DOF reactions are unavailable here and are not a force-balance gate'
            row['virtual_work']=feasible_virtual_work(current);checks['independent_feasible_virtual_work']=row['virtual_work']['passed']
            if case=='mpc_translation':
                checks['whole_field_translation']=bool(np.allclose(displacement,t*targets(case)[0],rtol=0,atol=TOL['position_m']))
                checks['zero_energy']=abs(mechanics['energy_J'])<=1e-13
        rows.append(row)
    return {'case':case,'passed':bool(solver['passed'] and required_consistent and all(all(r['checks'].values()) for r in rows)),
            'solver':solver,'reported_required_matches_fixed_rtol':bool(required_consistent),'states':rows,
            'scope':'Analytical software fixture only; no patient tent operator, mesh convergence, tissue fit or clinical validation'}


def write_bundle(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    X,E,boundary=fixture_mesh();H,parents,free,T=observation_operator()
    files={}
    for case in CASES:
        f=output/f'{case}.feb';f.write_text(deck_xml(case));files[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    manifest={'version':VERSION,'status':'prepared_not_executed','FEBio_commit':FEBIO_COMMIT,'caps':CAPS,'cases':list(CASES),
              'case_order':'Stop at first solver/checker failure; retain unexecuted statuses; never retry.',
              'units':'m,N,Pa,J','mu_Pa':MU_PA,'K_Pa':K_PA,'nu_initial':.45,'pressure_model':1,
              'node_count':len(X),'element_count':len(E),'free_affine_node_ids':[i+1 for i in range(len(X)) if i not in boundary],
              'mesh':'Explicit analytic cube split into6tet10, not anatomy and not an automatic mesher',
              'H_nodal_averages':H.tolist(),'parents_1based':(parents+1).tolist(),'free_1based':(free+1).tolist(),
              'observation_scope':'Explicit overlapping nodal averages only; NOT fixed5mm patient volume-tent kernels',
              'targets_m':{c:targets(c).tolist() for c in CASES if c!='tet10_affine'},
              'F_affine_final':affine_F(1).tolist(),'time_fractions':list(TIMES),'tolerances':TOL,
              'finite_difference_steps_m':list(FD_STEPS_M),'feasible_direction_count':3*T.shape[1],
              'residual_rtol':RESIDUAL_RTOL,'residual_floor_N2':RESIDUAL_FLOOR_N2,
              'check_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'shared_parser_source_sha256':hashlib.sha256(_PATCH_PATH.read_bytes()).hexdigest(),'decks':files,
              'numerical_method':'Pinned FEBio solid/BFGS/symmetric Skyline; ordinary elastic-solid TET10G8; no new solver',
              'runtime_wrapper':'Reuse committed scripts/febio_runtime.py supervise; worker checks source/deck/runtime inputs before/after; root must release execution separately'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('check');p.add_argument('--directory',required=True);p.add_argument('--case',choices=CASES,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':write_bundle(args.output);return
    directory=Path(args.directory)
    if (directory/f'{args.case}.feb').read_text()!=deck_xml(args.case):raise ValueError('Deck differs from frozen fixture')
    texts=[patch.read_bounded_text(directory/f'{args.case}.{suffix}') for suffix in ('nodes.log','elements.log','log')]
    result=check_outputs(args.case,*texts);print(json.dumps(result,indent=2,allow_nan=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
