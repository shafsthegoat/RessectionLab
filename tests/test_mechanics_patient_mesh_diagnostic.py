"""Analytical geometry/index controls only; no saved anatomy or native APIs."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
D=module('saved_mesh_diagnostic','scripts/mechanics_patient_mesh_diagnostic.py')
BASE=module('original_mesh_lattice','scripts/mechanics_patient_mesh.py')
X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
F=D.FACES.copy()


@pytest.mark.parametrize('scale',[.001,.004,1.])
def test_grouped_sample_order_and_cover_exactly_matches_original(scale):
    nodes=np.vstack([X*scale,X*scale*2+[5*scale,0,0]])
    faces=np.vstack([F,F+4])
    actual=list(D.indexed_samples(nodes,faces,.9*scale,2000,chunk_triangles=2))
    expected=list(BASE.triangle_samples(nodes,faces,.9*scale,2000,chunk_triangles=2))
    assert len(actual)==len(expected)
    seen=[]
    for (points,ids,n,cover),(old,old_cover) in zip(actual,expected):
        assert np.array_equal(points,old) and cover==old_cover
        # Independent plane membership and barycentric partition for each row.
        for row,face in enumerate(ids):
            tri=nodes[faces[face]];group=points[row*n:(row+1)*n]
            coeff=np.linalg.lstsq((tri[1:]-tri[0]).T,(group-tri[0]).T,rcond=None)[0].T
            assert np.all(coeff>=-1e-12) and np.all(coeff.sum(axis=1)<=1+1e-12)
            assert np.allclose(group,tri[0]+coeff@(tri[1:]-tri[0]),rtol=0,atol=1e-12)
        seen.extend(ids)
    assert sorted(seen)==list(range(len(faces)))


def test_lattice_cap_rejects_before_distance_or_large_allocation():
    with pytest.raises(ValueError,match='Sample cap'):next(D.indexed_samples(X,F,.01,10))


@pytest.mark.parametrize('cover',[0.,-1.,float('nan')])
def test_invalid_physical_cover(cover):
    with pytest.raises(ValueError):next(D.indexed_samples(X,F,cover,100))


def test_planar_distance_maximum_and_witness_triangle_ids():
    nodes=X*.003
    result,maxima=D.directed(nodes,F,lambda p:p[:,2],D.Budget(),cover=.001)
    assert maxima.tolist()==pytest.approx([.003,.003,.003,0])
    assert result['maximum_sample_distance_m']==.003
    assert len(result['worst_10'])==10
    for row in result['worst_10']:
        assert row['distance_m']==row['point_m'][2]
        assert row['triangle']!=3
    assert result['worst_10']==sorted(result['worst_10'],key=lambda r:(-r['distance_m'],r['ordinal']))


def test_total_query_cap_prevents_callback(monkeypatch):
    budget=D.Budget();budget.queries=D.CAPS['maximum_distance_queries']
    def forbidden(points):pytest.fail('Distance invoked after exhausted cap')
    with pytest.raises(ValueError,match='query cap'):D.directed(X*.001,F,forbidden,budget)


def test_cooperative_deadline_before_callback(monkeypatch):
    budget=D.Budget();budget.started-=111
    with pytest.raises(TimeoutError):D.directed(X*.001,F,lambda p:np.zeros(len(p)),budget)


def test_boundary_extracts_oriented_corners_and_actual_midside_indices():
    nodes=np.vstack([X,X[D.EDGES].mean(axis=1)])
    y,g,used,middle=D.boundary(nodes,np.arange(10)[None,:])
    assert np.array_equal(y,X) and np.array_equal(g,F)
    assert np.array_equal(used,np.arange(4))
    assert sorted(middle)==list(range(4,10))
    assert np.sum(np.einsum('ij,ij->i',y[g][:,0],np.cross(y[g][:,1],y[g][:,2])))/6==pytest.approx(1/6)
    nodes[4,0]+=.001
    with pytest.raises(ValueError,match='straight reference'):D.boundary(nodes,np.arange(10)[None,:])


def test_shared_interior_face_is_removed_without_renumbering_midsides():
    corners=np.vstack([X,[0,0,-1.]])
    cells4=np.array([[0,1,2,3],[0,2,1,4]])
    nodes=list(corners); edge_map={}; cells=[]
    for cell in cells4:
        mids=[]
        for a,b in D.EDGES:
            edge=tuple(sorted([int(cell[a]),int(cell[b])]))
            if edge not in edge_map:edge_map[edge]=len(nodes);nodes.append(corners[list(edge)].mean(axis=0))
            mids.append(edge_map[edge])
        cells.append([*cell,*mids])
    y,g,used,middle=D.boundary(np.array(nodes),np.array(cells))
    assert len(g)==6 and len(middle)==9
    assert not any(set(row)=={0,1,2} for row in used[g])


def test_area_weighting_clusters_and_index_mapping():
    vertices=np.vstack([X*.001,X*.002+[.01,0,0]])
    faces=np.vstack([F,F+4]);maxima=np.array([.003,.003,0,0,.004,.004,0,0])
    report,angle,labels=D.summaries(vertices,faces,maxima)
    assert report['connected_witness_components']==2
    a,b=report['largest_10_components']
    assert a['first_triangle']==4 and b['first_triangle']==0
    assert a['area_m2']==pytest.approx(4*b['area_m2'])
    assert labels[0]==labels[1] and labels[4]==labels[5] and labels[0]!=labels[4]
    assert np.array_equal(labels[[2,3,6,7]],[-1]*4)
    assert report['faces_with_sample_over_2mm']==4
    assert report['area_weighted_face_maximum_quantiles_m']['1.0']==.004
    assert np.all((angle>=0)&(angle<=180))


def test_known_right_angle_and_strict_threshold():
    _,areas,pairs,angles=D.topology(X,F)
    assert areas.tolist()==pytest.approx([np.sqrt(3)/2,.5,.5,.5])
    assert len(pairs)==6
    assert angles[3]>=90
    report,_,labels=D.summaries(X,F,np.full(4,.002))
    assert report['connected_witness_components']==0 and np.all(labels==-1)
    assert report['edge_normal_change_association']['30']['witness_face_area_fraction'] is None


def test_open_surface_refused():
    with pytest.raises(ValueError,match='Closed'):D.summaries(X,F[:3],np.zeros(3))


@pytest.mark.parametrize('values',[lambda p:np.full(len(p),np.nan),lambda p:-np.ones(len(p)),lambda p:np.zeros((len(p),1))])
def test_invalid_distance_outputs_rejected(values):
    with pytest.raises(ValueError,match='Invalid distance'):D.directed(X*.001,F,values,D.Budget())
