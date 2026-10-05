"""Bounded reference-volume tent averages on linear-reference tet10 meshes.

No destinations, patient paths, boundary assumptions or mechanics solver. The
refinement discrepancy is a numerical convergence estimate, not a rigorous
integral error bound. Separate geometric support-volume bounds are conservative.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from types import MappingProxyType
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).with_name('mechanics_patient_constraints.py')
_spec = importlib.util.spec_from_file_location('_verified_tet10_fixture', _SOURCE)
tet = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tet)

VERSION = 'reference-volume-tent-tet10-v1'
RADIUS_M = .005
FULL_MASS_M3 = math.pi * RADIUS_M**3 / 3
FULL_VOLUME_M3 = 4 * math.pi * RADIUS_M**3 / 3
RULE = MappingProxyType({'radius_m':RADIUS_M, 'mass_relative_error':1e-3,
        'coefficient_l1_error':2e-3, 'centroid_error_m':5e-6,
        'minimum_weighted_support_fraction':1e-4,
        'linear_midpoint_atol_m':1e-12, 'row_sum_atol':1e-10,
        'rank_relative_threshold':1e-8, 'maximum_pivot_condition':1e8,
        'elimination_residual_atol':1e-10, 'rotation_scale_m':RADIUS_M,'required_consecutive_refinements':2})


@dataclass(frozen=True)
class Limits:
    nodes: int = 200000
    elements: int = 250000
    observations: int = 6
    candidate_elements_per_row: int = 2048
    quadrature_points_per_row: int = 262144
    depth: int = 8
    active_nodes: int = 4096
    scalar_child_terms: int = 20000

    def __post_init__(self):
        maxima=(200000,250000,6,2048,262144,8,4096,20000)
        for (name,value),maximum in zip(vars(self).items(),maxima):
            if type(value) is not int or not 1<=value<=maximum:
                raise ValueError('Only positive, bounded numerical limits allowed: '+name)


class ObservationRefusal(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason=reason;self.details=details or {}


def _cancel(cancelled):
    if cancelled is not None and cancelled():raise ObservationRefusal('cancelled')


def _immutable(value,dtype=None):
    array=np.ascontiguousarray(value,dtype=dtype)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def _digest(*arrays,metadata):
    h=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for array in arrays:
        a=np.ascontiguousarray(array)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ObservationOperator:
    weights: np.ndarray
    active_node_indices: np.ndarray
    centers_m: np.ndarray
    centroids_m: np.ndarray
    source_hash: str
    rule_hash: str
    operator_hash: str
    reports: tuple
    limits: Limits
    source_node_count: int

    def apply(self, nodal_field):
        _validate_operator(self)
        values=np.asarray(nodal_field)
        if values.ndim not in (1,2) or values.shape[0]!=self.source_node_count or not np.isfinite(values).all():
            raise ValueError('Complete finite source-nodal field required')
        return self.weights @ values[self.active_node_indices]


@dataclass(frozen=True)
class ConstraintElimination:
    parent_node_indices: np.ndarray
    child_node_indices: np.ndarray
    child_coefficients: np.ndarray
    pivot_block: np.ndarray
    diagnostics: dict
    operator_hash: str
    # No displacement destinations accepted; caller must later solve pivot_block
    # against authorized input displacements using a separately bound workflow.


def _triangle_distance(p,a,b,c):
    """Closest-point region classification on one nondegenerate triangle."""
    ab=b-a;ac=c-a;ap=p-a;d1=ab@ap;d2=ac@ap
    if d1<=0 and d2<=0:return float(np.linalg.norm(ap))
    bp=p-b;d3=ab@bp;d4=ac@bp
    if d3>=0 and d4<=d3:return float(np.linalg.norm(bp))
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:return float(np.linalg.norm(p-(a+d1/(d1-d3)*ab)))
    cp=p-c;d5=ab@cp;d6=ac@cp
    if d6>=0 and d5<=d6:return float(np.linalg.norm(cp))
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:return float(np.linalg.norm(p-(a+d2/(d2-d6)*ac)))
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        return float(np.linalg.norm(p-(b+(d4-d3)/((d4-d3)+(d5-d6))*(c-b))))
    return abs(float(ap@np.cross(ab,ac)))/float(np.linalg.norm(np.cross(ab,ac)))


def _tet_distance(center,vertices):
    A=(vertices[1:]-vertices[0]).T
    bary=np.linalg.solve(A,center-vertices[0])
    if np.min(bary)>=0 and bary.sum()<=1:return 0.
    return min(_triangle_distance(center,*vertices[list(face)]) for face in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)))


def _shape_values(bary):
    return np.column_stack([bary*(2*bary-1),*[4*bary[:,i]*bary[:,j] for i,j in tet.EDGES]])


# Positive degree-five tetrahedron cubature (15 points), independently checked
# against all barycentric monomial integrals through total degree five.
from itertools import combinations
_QUAD=np.vstack([np.full((1,4),.25),
    np.full((4,4),1/3)-np.eye(4)/3,
    np.full((4,4),1/11)+np.eye(4)*7/11,
    np.array([[.4334498464263357 if i in pair else .0665501535736643
               for i in range(4)] for pair in combinations(range(4),2)])])
_WEIGHTS=np.r_[.1817020685825351,np.full(4,.0361607142857143),
               np.full(4,.0698714945161738),np.full(6,.0656948493683187)]
_POINTS=len(_WEIGHTS)
_SUBNODES=np.vstack([np.eye(4),*[(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in tet.EDGES]])
# Midpoint indices4=01,5=12,6=20,7=03,8=13,9=23.
_SUBTETS=np.array([[0,4,6,7],[4,1,5,8],[6,5,2,9],[7,8,9,3],
                  [4,6,7,9],[4,6,5,9],[4,5,8,9],[4,7,8,9]])
_SUBCELLS=_SUBNODES[_SUBTETS]


def _mesh(nodes_m,elements,limits,cancelled):
    X=np.asarray(nodes_m)
    E=np.asarray(elements)
    if X.ndim!=2 or X.shape[1]!=3 or not 4<=len(X)<=limits.nodes or X.dtype.kind not in 'fiu' or not np.isfinite(X).all():
        raise ObservationRefusal('invalid_or_oversized_nodes')
    if E.ndim!=2 or E.shape[1]!=10 or not 1<=len(E)<=limits.elements or not np.issubdtype(E.dtype,np.integer):
        raise ObservationRefusal('invalid_or_oversized_tet10_indices')
    X=np.array(X,dtype=np.float64,copy=True);E=np.array(E,dtype=np.int64,copy=True)
    if np.any(E<0) or np.any(E>=len(X)) or np.any(np.diff(np.sort(E,axis=1),axis=1)==0):
        raise ObservationRefusal('invalid_connectivity')
    if len(np.unique(E))!=len(X):raise ObservationRefusal('unused_source_nodes')
    _cancel(cancelled)
    vertices=X[E[:,:4]]; jac=np.swapaxes(vertices[:,1:]-vertices[:,0,None],1,2)
    determinants=np.linalg.det(jac)
    if np.any(determinants<=0):raise ObservationRefusal('nonpositive_reference_jacobian')
    expected=np.stack([(vertices[:,i]+vertices[:,j])/2 for i,j in tet.EDGES],axis=1)
    deviation=float(np.max(np.abs(X[E[:,4:]]-expected)))
    if deviation>RULE['linear_midpoint_atol_m']:raise ObservationRefusal('curved_reference_tet10_unsupported',{'maximum_midpoint_error_m':deviation})
    # Topology/QC belongs to the mesh producer. Refuse disconnected node graphs
    # here because a global rigid-mode check could otherwise miss a free piece.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a=np.repeat(E[:,0],9);b=E[:,1:].ravel()
    graph=coo_matrix((np.ones(len(a),dtype=np.uint8),(a,b)),shape=(len(X),len(X)))
    count,_=connected_components(graph,directed=False)
    if count!=1:raise ObservationRefusal('disconnected_reference_domain',{'components':int(count)})
    return X,E,vertices,determinants/6,deviation


def _integrate_row(vertices,volumes,indices,center,limits,cancelled):
    # Only intersecting subcells are retained. Complete local levels prevent an
    # accidentally small single adaptive update from being called convergence.
    cells=[(e,np.eye(4),V) for e,V in enumerate(volumes)]
    points_used=0;discarded=0;previous=None;consecutive=0;ledger=[]
    try:
        for depth in range(limits.depth+1):
            totals=np.zeros((len(indices),15));retained=[]
            support_lower=0.;support_upper=0.
            for e,B,V in cells:
                _cancel(cancelled)
                xyz=B@vertices[e];dmin=_tet_distance(center,xyz)
                if dmin>=RADIUS_M:discarded+=1;continue
                retained.append((e,B,V));dmax=float(np.linalg.norm(xyz-center,axis=1).max())
                support_lower+=V if dmax<RADIUS_M else 0.;support_upper+=V
                if points_used+_POINTS>limits.quadrature_points_per_row:
                    raise ObservationRefusal('quadrature_point_cap')
                points_used+=_POINTS
                bary=_QUAD@B;qxyz=bary@vertices[e];distance=np.linalg.norm(qxyz-center,axis=1)
                k=np.maximum(1-distance/RADIUS_M,0);wk=k*_WEIGHTS
                totals[e]+=V*np.r_[wk.sum(),wk@_shape_values(bary),
                                  wk@((qxyz-center)/RADIUS_M),_WEIGHTS@(distance<RADIUS_M)]
            total=totals.sum(axis=0);mass=float(total[0]);h_error=center_error=mass_error=float('inf')
            if previous is not None and mass>0 and previous[:,0].sum()>0:
                prior_mass=previous[:,0].sum()
                mass_error=abs(mass-prior_mass)
                # Per-original-element normalized coefficients prevent cancellation
                # between unrelated global nodes while allowing local signed error.
                h_error=float(np.abs(totals[:,1:11]/mass-previous[:,1:11]/prior_mass).sum())
                center_error=float(RADIUS_M*np.linalg.norm(total[11:14]/mass-previous[:,11:14].sum(axis=0)/prior_mass))
                passed=(mass_error<=RULE['mass_relative_error']*mass and h_error<=RULE['coefficient_l1_error'] and center_error<=RULE['centroid_error_m'])
                consecutive=consecutive+1 if passed else 0
            else:consecutive=0
            ledger.append({'depth':depth,'retained_cells':len(retained),'weighted_volume_m3':mass,
                           'mass_interlevel_difference_m3':mass_error if math.isfinite(mass_error) else None,
                           'coefficient_l1_interlevel_difference':h_error if math.isfinite(h_error) else None,
                           'centroid_interlevel_difference_m':center_error if math.isfinite(center_error) else None,
                           'consecutive_passes':consecutive,'quadrature_points_cumulative':points_used})
            if consecutive>=2:
                if mass<FULL_MASS_M3*RULE['minimum_weighted_support_fraction']:
                    raise ObservationRefusal('insufficient_weighted_support')
                report={'weighted_volume_m3':mass,'full_tent_weighted_volume_m3':FULL_MASS_M3,
                    'weighted_support_fraction':mass/FULL_MASS_M3,
                    'geometric_support_volume_estimate_m3':float(total[14]),
                    'geometric_support_lower_m3':float(max(0,support_lower)),
                    'geometric_support_upper_m3':float(min(FULL_VOLUME_M3,support_upper)),
                    'full_sphere_volume_m3':FULL_VOLUME_M3,
                    'kernel_first_moment_about_center_m4':(total[11:14]*RADIUS_M).tolist(),
                    'centroid_offset_m':(total[11:14]*RADIUS_M/mass).tolist(),
                    'refinement_mass_error_estimate_m3':mass_error,
                    'refinement_coefficient_l1_error_estimate':h_error,
                    'refinement_centroid_error_estimate_m':center_error,
                    'quadrature_points':points_used,'final_leaf_count':len(retained),'max_depth':depth,
                    'discarded_disjoint_cells':discarded,'candidate_elements':len(indices),
                    'refinement_ledger':ledger,
                    'error_scope':'Two successive complete-level interlevel differences; empirical convergence, not a rigorous quadrature error bound.',
                    'support_scope':'Unweighted support estimate plus convex-cell inside/possible-volume bounds; mesh topology must be independently validated.'}
                return totals[:,1:11]/mass,np.asarray(report['centroid_offset_m'])+center,report
            previous=totals
            cells=[(e,child,V/8) for e,B,V in retained for child in _SUBCELLS@B]
        raise ObservationRefusal('refinement_depth_cap')
    except ObservationRefusal as error:
        error.details.update(points_used=points_used,max_depth=depth,refinement_ledger=ledger)
        raise


def integrate_tent_observations(nodes_m,tet10_indices,centers_m,*,limits=Limits(),cancelled=None):
    """Integrate fixed-radius observations; accepts source locations only."""
    if not isinstance(limits,Limits):raise TypeError('Frozen Limits required')
    _cancel(cancelled)
    centers=np.asarray(centers_m)
    if centers.ndim!=2 or centers.shape[1]!=3 or not 1<=len(centers)<=limits.observations or centers.dtype.kind not in 'fiu' or not np.isfinite(centers).all():
        raise ObservationRefusal('invalid_or_oversized_centers')
    centers=np.array(centers,dtype=np.float64,copy=True)
    X,E,V,volumes,midpoint_error=_mesh(nodes_m,tet10_indices,limits,cancelled)
    source_hash=_digest(X,E,metadata={'coordinate_unit':'m','element_order':'FEBio tet10 linear-reference'})
    rule_hash=_digest(metadata={'version':VERSION,'rule':dict(RULE), 'quadrature':'positive degree-five tetrahedron cubature15, red-eight local refinement','limits':vars(limits),'shape_source_sha256':hashlib.sha256(_SOURCE.read_bytes()).hexdigest()})
    low=V.min(axis=1);high=V.max(axis=1);rows=[];reports=[];centroids=[];all_nodes=set()
    for row,center in enumerate(centers):
        _cancel(cancelled)
        closest=np.maximum(low-center,0)+np.minimum(high-center,0)
        candidates=np.flatnonzero(np.linalg.norm(closest,axis=1)<RADIUS_M)
        if len(candidates)>limits.candidate_elements_per_row:raise ObservationRefusal('broadphase_candidate_cap',{'row':row,'count':len(candidates)})
        retained=[]
        for i in candidates:
            _cancel(cancelled)
            if _tet_distance(center,V[i])<RADIUS_M:retained.append(i)
        candidates=np.array(retained,dtype=int)
        if not len(candidates):raise ObservationRefusal('no_positive_kernel_support',{'row':row})
        if len(candidates)>limits.candidate_elements_per_row:raise ObservationRefusal('candidate_element_cap',{'row':row,'count':len(candidates)})
        all_nodes.update(E[candidates].ravel().tolist())
        if len(all_nodes)>limits.active_nodes:raise ObservationRefusal('active_node_cap',{'count':len(all_nodes)})
        try:coefficients,centroid,report=_integrate_row(V[candidates],volumes[candidates],candidates,center,limits,cancelled)
        except ObservationRefusal as error:error.details['row']=row;raise
        nodes=np.unique(E[candidates]);weights=np.zeros(len(nodes))
        np.add.at(weights,np.searchsorted(nodes,E[candidates].ravel()),coefficients.ravel())
        if abs(weights.sum()-1)>RULE['row_sum_atol']:raise ObservationRefusal('constant_reproduction_failure',{'row':row})
        first=(weights[:,None]*(X[nodes]-center)).sum(axis=0)
        if not np.allclose(first,centroid-center,rtol=0,atol=1e-10):raise ObservationRefusal('affine_reproduction_failure',{'row':row})
        report.update(row=row,minimum_nodal_weight=float(weights.min()),maximum_nodal_weight=float(weights.max()),
                      row_sum=float(weights.sum()),active_nodes=len(nodes),maximum_linear_midpoint_error_m=midpoint_error,
                      truncated_support_estimate=bool(report['weighted_support_fraction']<1-RULE['mass_relative_error']),
                      centroid_m=centroid.tolist())
        rows.append((nodes,weights));reports.append(report);centroids.append(centroid)
    active=np.array(sorted(all_nodes),dtype=np.int64);H=np.zeros((len(rows),len(active)))
    for i,(nodes,weights) in enumerate(rows):H[i,np.searchsorted(active,nodes)]=weights
    operator_hash=_digest(H,active,centers,np.array(centroids),metadata={'source':source_hash,'rule':rule_hash,'source_nodes':len(X),'limits':vars(limits)})
    return ObservationOperator(_immutable(H),_immutable(active),_immutable(centers),_immutable(centroids),source_hash,rule_hash,operator_hash,tuple(reports),limits,len(X))


def _validate_operator(operator):
    if not isinstance(operator,ObservationOperator):raise TypeError('ObservationOperator required')
    actual=_digest(operator.weights,operator.active_node_indices,operator.centers_m,operator.centroids_m,
                   metadata={'source':operator.source_hash,'rule':operator.rule_hash,
                             'source_nodes':operator.source_node_count,'limits':vars(operator.limits)})
    if actual!=operator.operator_hash:raise ObservationRefusal('operator_content_changed')


def prepare_elimination(operator):
    """Source-only deterministic pivots; no observed destinations accepted."""
    if not isinstance(operator,ObservationOperator):raise TypeError('ObservationOperator required')
    _validate_operator(operator)
    H=np.array(operator.weights,copy=True);m,n=H.shape
    singular=np.linalg.svd(H,compute_uv=False)
    if m>n or singular[-1]<=RULE['rank_relative_threshold']*singular[0]:raise ObservationRefusal('dependent_observation_rows',{'singular_values':singular.tolist()})
    centered=(operator.centroids_m-operator.centroids_m.mean(axis=0))/RULE['rotation_scale_m']
    rigid=[]
    for x,y,z in centered:
        skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        rigid.append(np.column_stack([np.eye(3),-skew]))
    rs=np.linalg.svd(np.vstack(rigid),compute_uv=False)
    if len(rs)<6 or rs[-1]<=RULE['rank_relative_threshold']*rs[0]:raise ObservationRefusal('rigid_mode_rank_failure',{'singular_values':rs.tolist()})
    selected=[]
    for _ in range(m):
        residual=H.copy()
        if selected:
            Q,_=np.linalg.qr(H[:,selected],mode='reduced')
            # Repeat projection to limit loss of orthogonality; exact ties favor
            # the lower global node index because active nodes are sorted.
            residual-=Q@(Q.T@residual);residual-=Q@(Q.T@residual)
        norm=np.sum(residual*residual,axis=0);norm[selected]=-1
        selected.append(int(np.argmax(norm)))
    free=np.array([i for i in range(n) if i not in selected]);P=H[:,selected]
    condition=float(np.linalg.cond(P))
    if condition>RULE['maximum_pivot_condition']:raise ObservationRefusal('pivot_condition_failure',{'condition':condition})
    A=-np.linalg.solve(P,H[:,free]);terms=int(np.count_nonzero(A))
    if terms>operator.limits.scalar_child_terms:raise ObservationRefusal('elimination_term_cap',{'scalar_terms':terms,'vector_terms':3*terms})
    residual=float(np.max(np.abs(P@A+H[:,free]))) if len(free) else 0.
    if residual>RULE['elimination_residual_atol']:raise ObservationRefusal('elimination_residual_failure')
    return ConstraintElimination(_immutable(operator.active_node_indices[selected]),_immutable(operator.active_node_indices[free]),
        _immutable(A),_immutable(P),{'row_singular_values':singular.tolist(),'rigid_mode_singular_values':rs.tolist(),
         'pivot_condition':condition,'scalar_child_terms':terms,'vector_child_terms':3*terms,'maximum_original_row_residual':residual,
         'pivot_method':'Greedy twice-projected QR column norm; exact ties lower global source node index.'},operator.operator_hash)
