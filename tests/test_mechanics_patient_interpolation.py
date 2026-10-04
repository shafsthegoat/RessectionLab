"""Analytic reference tetrahedra and temporary field archives, never anatomy."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import mechanics_patient_interpolation as m
from scripts import mechanics_patient_comparison as comparison


def expand(vertices, corners):
    nodes=list(np.asarray(vertices,dtype=float)); mids={}; result=[]
    for raw in corners:
        cell=list(raw)
        if np.linalg.det((np.asarray(nodes)[cell[1:]]-nodes[cell[0]]).T)<0:
            cell[1],cell[2]=cell[2],cell[1]
        ids=cell.copy()
        for first,second in m.EDGES:
            edge=tuple(sorted((cell[first],cell[second])))
            if edge not in mids:
                mids[edge]=len(nodes); nodes.append((nodes[edge[0]]+nodes[edge[1]])/2)
            ids.append(mids[edge])
        result.append(ids)
    return np.asarray(nodes),np.asarray(result,dtype=np.int64)


def mesh(two=False):
    corners=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[0,0,-.01]])
    return expand(corners if two else corners[:4],[[0,1,2,3],[0,2,1,4]] if two else [[0,1,2,3]])


def transform(rotation=None, translation=None):
    a=np.eye(4); a[:3,:3]=1000*(np.eye(3) if rotation is None else rotation)
    a[:3,3]=np.zeros(3) if translation is None else translation
    return a


def motions(x,kind):
    if kind=='constant':return np.tile([.0001,-.0003,.0002],(len(x),1))
    if kind=='affine':return x@np.array([[.1,.2,.3],[-.2,.3,.4],[.1,-.4,.2]]).T+[.0003,0,-.0002]
    return np.column_stack((x[:,0]**2+2*x[:,1]*x[:,2],x[:,1]**2-x[:,0]*x[:,2],x[:,2]**2+x[:,0]*x[:,1]))


@pytest.mark.parametrize('kind',['constant','affine','quadratic'])
def test_constant_affine_quadratic_patch_and_all_nodes(kind):
    x,e=mesh(); field=m.FrozenTet10Field(x,e,motions(x,kind),transform())
    query=np.vstack(([[.001,.002,.003],[.0025]*3],x))
    result=m.sample_field(field,query*1000)
    assert result['counts']=={'supported':len(query)}
    np.testing.assert_allclose([r['displacement_ras_mm'] for r in result['rows']],motions(query,kind)*1000,atol=3e-15)


def test_conforming_shared_face_edge_and_vertex_check_all_containing_cells():
    x,e=mesh(two=True); u=np.random.default_rng(13).normal(0,.001,x.shape)
    field=m.FrozenTet10Field(x,e,u,transform())
    report=m.sample_field(field,[[2,3,0],[2,0,0],[0,0,0]])
    assert report['counts']=={'supported':3}
    assert all(len(row['element_ids'])==2 for row in report['rows'])
    assert all(row['maximum_containing_field_disagreement_m']<1e-16 for row in report['rows'])
    reversed_field=m.FrozenTet10Field(x,e[::-1],u,transform())
    np.testing.assert_allclose([r['displacement_ras_mm'] for r in report['rows']],
                              [r['displacement_ras_mm'] for r in m.sample_field(reversed_field,[[2,3,0],[2,0,0],[0,0,0]])['rows']],atol=1e-14)


def test_near_outside_remains_uncertain_without_snapping_and_clear_outside_is_null():
    x,e=mesh();field=m.FrozenTet10Field(x,e,motions(x,'affine'),transform())
    query=np.array([[2,3,-.5e-9],[2,3,-2e-9],[2,3,0],[20,30,40.]])
    original=query.copy(); report=m.sample_field(field,query)
    assert [r['status'] for r in report['rows']]==['uncertain_reference_boundary','outside_reference_domain','supported','outside_reference_domain']
    assert report['rows'][0]['displacement_ras_mm'] is None
    assert report['query_snapping'] is report['extrapolation'] is False
    np.testing.assert_array_equal(query,original)


def test_common_frame_rotation_units_translation_and_inverse_point_mapping():
    x,e=mesh();q=np.array([[.001,.002,.003],[.002,.001,.004]])
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    a=transform(rotation,[-7,12,36]);u=motions(x,'quadratic')
    field=m.FrozenTet10Field(x,e,u,a)
    common=q@a[:3,:3].T+a[:3,3]
    result=m.sample_field(field,common)
    np.testing.assert_allclose([r['displacement_ras_mm'] for r in result['rows']],motions(q,'quadratic')@a[:3,:3].T,atol=1e-15)
    moved=m.FrozenTet10Field(x@rotation.T,e,u@rotation.T,transform())
    actual=m.sample_field(moved,q@rotation.T*1000)
    np.testing.assert_allclose([r['displacement_ras_mm'] for r in actual['rows']],motions(q,'quadratic')@rotation.T*1000,atol=1e-15)


@pytest.mark.parametrize('bad',['meters','reflection','shear','homogeneous'])
def test_transform_refuses_implicit_units_reflection_shear_and_nonhomogeneous(bad):
    x,e=mesh();a=transform()
    if bad=='meters':a[:3,:3]/=1000
    if bad=='reflection':a[0,0]*=-1
    if bad=='shear':a[0,1]=1
    if bad=='homogeneous':a[3,0]=1e-15
    with pytest.raises(ValueError):m.FrozenTet10Field(x,e,np.zeros_like(x),a)


def test_overlapping_interiors_are_ambiguous_not_first_hit():
    x,e=mesh();v=np.vstack((x[:4],x[:4]+[.001,.001,.001]))
    x,e=expand(v,[[0,1,2,3],[4,5,6,7]])
    field=m.FrozenTet10Field(x,e,motions(x,'constant'),transform())
    row=m.sample_field(field,[[3,3,2]])['rows'][0]
    assert row['status']=='ambiguous_reference_location' and row['displacement_ras_mm'] is None


def test_disconnected_duplicate_shared_midpoint_is_nonconforming():
    x,e=mesh(two=True);duplicate=e[1,4];x=np.vstack((x,x[duplicate]));e[1,4]=len(x)-1
    field=m.FrozenTet10Field(x,e,np.zeros_like(x),transform())
    assert field.geometry_status=='nonconforming_reference_mesh'
    assert m.sample_field(field,[[2,3,0]])['rows'][0]['displacement_ras_mm'] is None


@pytest.mark.parametrize('bad',['curved','inverted','degenerate'])
def test_unsupported_reference_geometry_is_not_silently_linearized(bad):
    x,e=mesh()
    if bad=='curved':x[4,0]+=1e-6
    if bad=='inverted':e[0,[1,2]]=e[0,[2,1]]
    if bad=='degenerate':x[3]=x[0]
    field=m.FrozenTet10Field(x,e,np.zeros_like(x),transform())
    result=m.sample_field(field,[[1,2,3]])
    assert result['counts']=={'unsupported_reference_geometry':1}
    assert result['rows'][0]['displacement_ras_mm'] is None


def test_immutable_copies_and_metadata_guard():
    x,e=mesh();u=np.zeros_like(x);field=m.FrozenTet10Field(x,e,u,transform())
    x[:]=99;u[:]=99;e[:]=0
    assert m.sample_field(field,[[1,2,3]])['counts']=={'supported':1}
    with pytest.raises(ValueError):field.nodes_m[0,0]=1
    with pytest.warns(DeprecationWarning):
        field.nodes_m.shape=(field.nodes_m.size,)
    with pytest.raises(ValueError,match='changed'):m.sample_field(field,[[1,2,3]])


def test_absolute_midpoint_tolerance_cannot_hide_relatively_curved_tiny_cells():
    x,e=mesh();x*=1e-10;x[4,0]+=5e-13
    field=m.FrozenTet10Field(x,e,np.zeros_like(x),transform())
    assert field.geometry_status=='unsupported_reference_geometry'
    assert field.geometry_reason=='curved_reference_tet10_unsupported'


def artifact(root,name,payload):
    (root/name).write_bytes(payload)
    return {'path':name,'sha256':hashlib.sha256(payload).hexdigest()}


def bundle(root):
    x,e=mesh();mesh_bytes=io.BytesIO();np.savez(mesh_bytes,nodes_m=x,tet10_indices=e)
    motion_bytes=io.BytesIO();np.savez(motion_bytes,displacement_m=motions(x,'affine'))
    frame={'schema':'native-rest-to-common-ras-v1','native_reference_frame':'MRI_RAS+','native_length_units':'m',
           'common_frame':'RAS+','common_length_units':'mm','native_m_to_common_mm':transform().tolist(),
           'baseline_alignment_sha256':'1'*64,'source_image_sha256':'2'*64,'alignment_status':'provisional_baseline_diagnostic'}
    manifest={'schema':'frozen-tet10-displacement-field-v1','frame':'RAS+','units':'mm',
              'mesh_length_units':'m','nodal_displacement_units':'m','tet10_indexing':'zero_based',
              'tet10_order':m.ORDER,'direction':'reference_to_displaced','forward_hash':'3'*64,
              'protocol_sha256':'4'*64,'interpolator_entrypoint':'sample_frozen_field','artifacts':{
                  'reference_mesh':artifact(root,'mesh.npz',mesh_bytes.getvalue()),
                  'nodal_displacements':artifact(root,'motion.npz',motion_bytes.getvalue()),
                  'geometry_frame':artifact(root,'frame.json',json.dumps(frame).encode()),
                  'numerical_evidence':artifact(root,'numerical.json',b'{"fixture_only":true}'),
                  'interpolator_source':artifact(root,'interpolator.py',Path(m.__file__).read_bytes())}}
    return manifest


def test_portable_complete_field_adapter_preserves_queries_and_null_counts(tmp_path):
    manifest=bundle(tmp_path)
    query=comparison.FieldQueries((1,2,3),((1.,2.,3.),(20.,30.,40.),(2.,3.,-.5e-9)))
    result=m.sample_frozen_field(tmp_path,manifest,query)
    assert result.query_sha256==query.sha256
    assert result.status==('supported','outside_reference_domain','uncertain_reference_boundary')
    assert result.displacement_ras_mm[1:] == (None,None)
    np.testing.assert_allclose(result.displacement_ras_mm[0],motions(np.array([[.001,.002,.003]]),'affine')[0]*1000)


def test_forward_identity_accepts_existing_prefixed_hash_convention(tmp_path):
    manifest=bundle(tmp_path);manifest['forward_hash']='sha256:'+manifest['forward_hash']
    assert m.load_frozen_field(tmp_path,manifest).geometry_status=='supported'


@pytest.mark.parametrize('member',['mesh.npz','motion.npz','frame.json','numerical.json','interpolator.py'])
def test_each_complete_field_artifact_must_keep_its_frozen_hash(tmp_path,member):
    manifest=bundle(tmp_path);(tmp_path/member).write_bytes(b'changed')
    with pytest.raises(ValueError):m.load_frozen_field(tmp_path,manifest)


def test_header_allocation_bomb_rejected_before_numpy_load(monkeypatch):
    header=io.BytesIO();np.lib.format.write_array_header_1_0(header,{'descr':'<f8','fortran_order':False,'shape':(1000000000,3)})
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z:z.writestr('nodes_m.npy',header.getvalue())
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('allocated before header bound'))
    with pytest.raises(ValueError,match='before allocation'):
        m._npz(archive.getvalue(),{'nodes_m':(3,m.MAX_NODES,'float64')})


def test_provisional_frame_and_no_query_options_in_frozen_manifest(tmp_path):
    manifest=bundle(tmp_path);manifest['query_tolerance']=1
    with pytest.raises(ValueError):m.load_frozen_field(tmp_path,manifest)
    manifest=bundle(tmp_path);frame=json.loads((tmp_path/'frame.json').read_text());frame['alignment_status']='clinically_accepted'
    manifest['artifacts']['geometry_frame']=artifact(tmp_path,'frame.json',json.dumps(frame).encode())
    with pytest.raises(ValueError):m.load_frozen_field(tmp_path,manifest)
