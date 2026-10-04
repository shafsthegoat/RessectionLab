"""Analytic software controls only: no patient arrays, destinations or solver."""
import dataclasses
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/mechanics_patient_observation.py'
spec=importlib.util.spec_from_file_location('patient_observation_owner',PATH)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

@pytest.fixture(scope='module')
def mesh():return m.tet.fixture_mesh()[:2]

@pytest.fixture(scope='module')
def complete(mesh):return m.integrate_tent_observations(*mesh,[[.005,.005,.005]])

@pytest.fixture(scope='module')
def half(mesh):return m.integrate_tent_observations(*mesh,[[0,.005,.005]])

@pytest.fixture(scope='module')
def small():
 X,E,_=m.tet.fixture_mesh();X=X/5
 return X,E,m.integrate_tent_observations(X,E,[[.0004,.0004,.0004],[.0016,.0005,.0006],[.0006,.0016,.0008]])

def test_cubature_polynomial_integrals_and_red_partition():
 for points,weights,degree in [(m._QUAD,m._WEIGHTS,2),(m._CONFIRM_QUAD,m._CONFIRM_WEIGHTS,5)]:
  assert np.all(points>=0) and np.all(weights>0)
  for powers in itertools.product(range(degree+1),repeat=4):
   if sum(powers)>degree:continue
   expected=6*math.prod(math.factorial(v) for v in powers)/math.factorial(3+sum(powers))
   assert weights@np.prod(points**powers,axis=1)==pytest.approx(expected,abs=8e-16)
 for cell in m._SUBCELLS:
  assert abs(np.linalg.det((cell[1:,1:]-cell[0,1:]).T))==pytest.approx(1/8)
 # Each child centroid belongs to exactly that child's interior; volumes sum1.
 for i,p in enumerate(m._SUBCELLS.mean(axis=1)):
  containing=[]
  for j,cell in enumerate(m._SUBCELLS):
   bary=np.linalg.solve((cell[1:,1:]-cell[0,1:]).T,p[1:]-cell[0,1:])
   if np.all(bary>0) and bary.sum()<1:containing.append(j)
  assert containing==[i]

def test_shape_matches_frozen_order():
 bary=np.random.default_rng(6).dirichlet(np.ones(4),30)
 bary[:,0]=1-bary[:,1:].sum(axis=1) # Same reference-coordinate representation.
 expected=np.array([m.tet.shape(b[1:])[0] for b in bary])
 assert np.allclose(m._shape_values(bary),expected,rtol=0,atol=4e-16)

def test_full_sphere_analytic_mass_centroid_and_support(complete):
 r=complete.reports[0]
 assert r['weighted_volume_m3']==pytest.approx(math.pi*.005**3/3,rel=1e-3)
 assert complete.centroids_m[0]==pytest.approx([.005]*3,abs=1e-8)
 assert r['geometric_support_lower_m3']<=4*math.pi*.005**3/3<=r['geometric_support_upper_m3']
 assert r['quadrature_points']<=262144
 assert r['refinement_ledger'][-1]['h_refinement']['passes']
 assert r['refinement_ledger'][-1]['higher_order_cross_check']['passes']
 assert r['weighted_support_fraction']>1 # Tiny excess preserved, never clamped to probability.

def test_half_sphere_analytic_mass_and_shift(half):
 r=half.reports[0]
 assert r['weighted_volume_m3']==pytest.approx(math.pi*.005**3/6,rel=1e-3)
 assert half.centroids_m[0]==pytest.approx([3*.005/10,.005,.005],abs=5e-6)
 assert r['geometric_support_lower_m3']<=2*math.pi*.005**3/3<=r['geometric_support_upper_m3']
 assert r['truncated_support_estimate']
 assert np.linalg.norm(r['kernel_first_moment_about_center_m4'])>0

def test_constant_affine_and_negative_quadratic_coefficients(mesh,complete,half):
 X,E=mesh;A=np.array([[2.,-.7,.2],[-3.,.6,.1],[.5,2.,4.]])
 for op in (complete,half):
  assert op.apply(np.ones(len(X)))==pytest.approx([1],abs=2e-14)
  assert op.apply(X@A.T+.7)==pytest.approx(op.centroids_m@A.T+.7,abs=2e-14)
  assert op.weights.min()<-.01

def test_independent_tensor_cube_integration(mesh,half):
 # Tensor Gauss integration in Cartesian cube, independent of mesh/tet rules.
 X,_=mesh;g,w=np.polynomial.legendre.leggauss(60);g=(g+1)*.005;w=w*.005
 xyz=np.stack(np.meshgrid(g,g,g,indexing='ij'),axis=-1).reshape(-1,3)
 weight=np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),axis=-1),axis=-1).ravel()
 k=np.maximum(1-np.linalg.norm(xyz-[0,.005,.005],axis=1)/.005,0)
 wk=weight*k;f=xyz[:,0]**2+xyz[:,1]*xyz[:,2]
 expected=wk@f/wk.sum();actual=half.apply(X[:,0]**2+X[:,1]*X[:,2])[0]
 assert actual==pytest.approx(expected,rel=m.RULE['mass_relative_error'])
 assert half.reports[0]['weighted_volume_m3']==pytest.approx(wk.sum(),rel=1e-3)

def test_source_only_elimination_and_no_parent_chains(small):
 X,E,op=small;result=m.prepare_elimination(op)
 assert not set(result.parent_node_indices)&set(result.child_node_indices)
 u=np.zeros((len(X),3));u[result.child_node_indices]=np.random.default_rng(20).normal(size=(len(result.child_node_indices),3))
 u[result.parent_node_indices]=result.child_coefficients@u[result.child_node_indices]
 assert np.max(np.abs(op.apply(u)))<1e-12
 assert result.diagnostics['pivot_condition']<1e8
 assert np.array_equal(result.parent_node_indices,m.prepare_elimination(op).parent_node_indices)

def test_row_rank_failure(mesh):
 X,E=mesh;X=X/5
 op=m.integrate_tent_observations(X,E,[[.001,.001,.001]]*3)
 with pytest.raises(m.ObservationRefusal,match='dependent_observation_rows'):m.prepare_elimination(op)

def test_rigid_rank_failure(complete):
 with pytest.raises(m.ObservationRefusal,match='rigid_mode_rank_failure'):m.prepare_elimination(complete)

@pytest.mark.parametrize('center',[[.04,.04,.04],[-.005,.005,.005]])
def test_disjoint_or_zero_volume_tangent(mesh,center):
 with pytest.raises(m.ObservationRefusal,match='no_positive_kernel_support'):m.integrate_tent_observations(*mesh,[center])

def test_tiny_support_is_not_rescaled_to_valid_mass(mesh):
 X,E=mesh;X=X*.001
 with pytest.raises(m.ObservationRefusal,match='insufficient_weighted_support'):
  m.integrate_tent_observations(X,E,[[.000005]*3])

def test_missed_small_support_refines_then_abstains(mesh):
 X,E=mesh;X=X*20
 with pytest.raises(m.ObservationRefusal,match='refinement_depth_cap') as error:
  m.integrate_tent_observations(X,E,[[.011,.007,.004]],limits=m.Limits(depth=1))
 assert error.value.details['refinement_ledger'][0]['weighted_volume_m3']==0
 assert len(error.value.details['refinement_ledger'])==2

@pytest.mark.parametrize('limits,reason',[(m.Limits(quadrature_points_per_row=8),'quadrature_point_cap'),(m.Limits(candidate_elements_per_row=1),'broadphase_candidate_cap'),(m.Limits(active_nodes=10),'active_node_cap')])
def test_caps_no_partial_operator(mesh,limits,reason):
 with pytest.raises(m.ObservationRefusal,match=reason):m.integrate_tent_observations(*mesh,[[.005]*3],limits=limits)

def test_cancellation_and_limits(mesh):
 with pytest.raises(m.ObservationRefusal,match='cancelled'):m.integrate_tent_observations(*mesh,[[.005]*3],cancelled=lambda:True)
 for value in [0,False,262145]:
  with pytest.raises(ValueError):m.Limits(quadrature_points_per_row=value)

def test_input_copy_and_descriptor_tampering(small):
 X,E,op=small
 with pytest.raises(ValueError):op.weights.setflags(write=True)
 bad=dataclasses.replace(op,weights=op.weights.copy())
 bad.weights[0,0]+=.01
 with pytest.raises(m.ObservationRefusal,match='operator_content_changed'):m.prepare_elimination(bad)
 with pytest.raises(m.ObservationRefusal,match='operator_content_changed'):bad.apply(np.ones(len(X)))
 with pytest.raises(m.ObservationRefusal,match='operator_content_changed'):m.prepare_elimination(dataclasses.replace(op,source_node_count=len(X)+1))

@pytest.mark.parametrize('kind',['curved','inverted','float_indices','complex','nan','unused','disconnected'])
def test_invalid_reference_domains(mesh,kind):
 X,E=mesh;X=X.copy();E=E.copy()
 if kind=='curved':X[E[0,4],0]+=1e-6
 elif kind=='inverted':X[:,0]*=-1
 elif kind=='float_indices':E=E.astype(float)
 elif kind=='complex':X=X.astype(complex)+1j
 elif kind=='nan':X[0,0]=np.nan
 elif kind=='unused':X=np.vstack([X,[1,1,1]])
 elif kind=='disconnected':E=np.vstack([E,E+len(X)]);X=np.vstack([X,X+.1])
 with pytest.raises(m.ObservationRefusal):m.integrate_tent_observations(X,E,[[.005]*3])

def test_rigid_reexpression_and_source_binding(mesh,half):
 X,E=mesh;theta=.37;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
 t=np.array([.03,-.05,.08]);centers=np.array([[0,.005,.005]])@R.T+t
 op=m.integrate_tent_observations(X@R.T+t,E,centers)
 assert op.weights==pytest.approx(half.weights,abs=2e-5)
 assert op.centroids_m==pytest.approx(half.centroids_m@R.T+t,abs=1e-7)
 assert op.source_hash!=half.source_hash and op.operator_hash!=half.operator_hash
 assert op.rule_hash==half.rule_hash


def test_child_term_cap(small):
 X,E,op=small
 bounded=m.integrate_tent_observations(X,E,op.centers_m,limits=m.Limits(scalar_child_terms=1))
 with pytest.raises(m.ObservationRefusal,match='elimination_term_cap'):m.prepare_elimination(bounded)


def test_cancel_during_integration_and_copied_inputs(mesh):
 X,E=mesh;X=X.copy();E=E.copy();calls=0
 def cancelled():
  nonlocal calls
  calls+=1
  return calls>12
 with pytest.raises(m.ObservationRefusal,match='cancelled') as exc:
  m.integrate_tent_observations(X,E,[[.005]*3],cancelled=cancelled)
 assert exc.value.details['points_used']>0


def test_exact_distances_avoid_aabb_false_support():
 V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
 assert m._tet_distance(np.array([.8,.8,.8]),V)==pytest.approx(1.4/np.sqrt(3))
 assert m._tet_distance(np.array([.1,.1,.1]),V)==0
 assert m._tet_distance(np.array([-.1,0,0]),V)==pytest.approx(.1)
