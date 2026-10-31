"""Source identity, topology losses and image-boundary uncertainty of derived graphs."""
import numpy as np
import pytest

from resectionlab import anatomy_geometry as geometry


def test_source_identity_exact_world_coordinates_and_mask_preservation():
    mask = np.zeros((9,10,11),bool); mask[2:7,4,5] = True; mask[6,4:8,5] = True
    a = np.array([[0.,-.7,0.,13.],[.4,0.,0.,-8.],[0.,0.,-1.2,32.],[0.,0.,0.,1.]])
    before = mask.copy(); result = geometry.vessel_graph(mask,a)
    ids = result['source_cell_ids']; ijk = np.column_stack(np.unravel_index(ids,mask.shape))
    np.testing.assert_array_equal(mask,before)
    assert mask.ravel()[ids].all() and len(set(ids)) == len(ids)
    np.testing.assert_allclose(result['points_ras_mm'],ijk@a[:3,:3].T+a[:3,3],atol=1e-14)
    assert result['points_ras_mm'].dtype == np.float64
    assert result['source_components6'] == result['components'] == 1
    assert result['unrepresented_source_components26'] == 0
    assert result['voxel_skeleton_betti_preserved26']
    assert result['edge_local_face_supported'].all()


def test_diagonal_only_annotation_is_not_promoted_to_face_connectivity(monkeypatch):
    mask = np.zeros((5,5,5),bool); mask[1,1,2] = True; mask[2,2,2] = True
    # Freeze an admissible skeleton containing the actual two source voxels.
    monkeypatch.setattr(geometry,'skeletonize',lambda value,**kwargs:value.copy())
    result = geometry.vessel_graph(mask,np.eye(4))
    assert result['components'] == 1 and result['source_components6'] == 2
    assert result['edges'].shape == (1,2)
    assert not result['edge_face_adjacent'].any()
    assert not result['edge_local_face_supported'].any()


def test_diagonal_centerline_with_local_positive_face_path_is_distinguished(monkeypatch):
    mask = np.zeros((5,5,5),bool); mask[1,1,2] = mask[2,1,2] = mask[2,2,2] = True
    def skeleton(value,**kwargs):
        result = value.copy(); indices=np.argwhere(result); result[tuple(indices[1])] = False; return result
    monkeypatch.setattr(geometry,'skeletonize',skeleton)
    result = geometry.vessel_graph(mask,np.eye(4))
    assert result['edges'].shape == (1,2) and not result['edge_face_adjacent'].any()
    assert result['edge_local_face_supported'].all()


def test_omitted_component_is_retained_as_failure_instead_of_empty_anatomy(monkeypatch):
    mask = np.zeros((6,6,6),bool); mask[1,1,1] = mask[4,4,4] = True
    monkeypatch.setattr(geometry,'skeletonize',lambda value,**kwargs:np.zeros_like(value))
    result = geometry.vessel_graph(mask,np.eye(4))
    assert result['components'] == 2 and result['skeleton_components26'] == 0
    assert result['unrepresented_source_components26'] == result['unrepresented_source26_voxels'] == 2
    assert not result['voxel_skeleton_betti_preserved26']


def test_unknown_image_exterior_limits_annotation_distance():
    mask = np.ones((5,5,5),bool)
    result = geometry.vessel_graph(mask,np.eye(4))
    assert result['source_boundary_foreground_voxels'] == 125-27
    assert result['radius_limited_by_image_extent'].all()
    assert 'not measured lumen radius' in result['meaning']


def test_empty_annotation_returns_empty_graph_and_explicit_source_grid():
    result = geometry.vessel_graph(np.zeros((3,4,5),bool),np.eye(4))
    assert result['points_ras_mm'].shape == (0,3) and result['edges'].shape == (0,2)
    assert result['components'] == result['source_foreground_voxels'] == 0
    np.testing.assert_array_equal(result['source_shape'],[3,4,5])


@pytest.mark.parametrize('bad',[np.ones((2,2)),np.full((3,3,3),2,np.uint8),np.full((3,3,3),float('nan'))])
def test_ambiguous_source_labels_or_grid_refused(bad):
    with pytest.raises(ValueError,match='binary source mask'): geometry.vessel_graph(bad,np.eye(4))


def test_sheared_radius_metric_abstains_instead_of_orthogonalizing():
    a=np.eye(4); a[0,1]=.2
    with pytest.raises(ValueError,match='orthogonal source grid'):
        geometry.vessel_graph(np.ones((3,3,3),bool),a)


def _qualification_module():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/qualify_native_vascular_graph.py'
    spec=importlib.util.spec_from_file_location('native_graph_qualification',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_unknown_annotation_unit_binds_only_to_exact_named_mm_reference_without_header_edits():
    import nibabel as nib
    q=_qualification_module(); image=nib.Nifti1Image(np.zeros((3,4,5),np.uint8),np.eye(4))
    image.header.set_xyzt_units('mm'); annotation=nib.Nifti1Image(np.ones((3,4,5),np.uint8),np.eye(4))
    before=annotation.header.binaryblock
    result=q.bind_annotation_grid(image,annotation,image.shape,image.affine,association_proven=True)
    assert result['physicalUnit']=='mm' and result['annotationUnitCopiedFromReference']
    assert result['annotationHeaderUnit']=='unknown' and annotation.header.binaryblock==before


@pytest.mark.parametrize('change',['reference-unit','annotation-unit','affine','shape','association','sform'])
def test_conflicting_or_unbound_annotation_grid_is_refused(change):
    import nibabel as nib
    q=_qualification_module(); image=nib.Nifti1Image(np.zeros((3,4,5),np.uint8),np.eye(4));image.header.set_xyzt_units('mm')
    annotation=nib.Nifti1Image(np.ones((3,4,5),np.uint8),np.eye(4)); association=True
    if change=='reference-unit': image.header.set_xyzt_units('unknown')
    if change=='annotation-unit': annotation.header.set_xyzt_units('meter')
    if change=='affine':
        a=np.eye(4);a[1,3]=.2;annotation=nib.Nifti1Image(np.ones((3,4,5),np.uint8),a)
    if change=='shape': annotation=nib.Nifti1Image(np.ones((3,4,6),np.uint8),np.eye(4))
    if change=='association': association=False
    if change=='sform': annotation.header['sform_code']=0
    with pytest.raises(ValueError,match='named reference or physical grid mismatch'):
        q.bind_annotation_grid(image,annotation,image.shape,image.affine,association_proven=association)


def test_source_and_foreground_caps_refuse_before_coordinate_allocation():
    with pytest.raises(ValueError,match='grid cap'):
        geometry.vessel_graph(np.broadcast_to(False,(1,1,128_000_001)),np.eye(4))
    with pytest.raises(ValueError,match='one-million-cell cap'):
        geometry.vessel_graph(np.broadcast_to(True,(101,101,101)),np.eye(4))


def test_corner_clique_cycles_are_not_mistaken_for_voxel_holes(monkeypatch):
    mask=np.zeros((5,5,5),bool);mask[1,1,2]=mask[2,1,2]=mask[2,2,2]=True
    monkeypatch.setattr(geometry,'skeletonize',lambda value,**kwargs:value.copy())
    result=geometry.vessel_graph(mask,np.eye(4))
    assert result['voxel_skeleton_betti_preserved26']
    assert result['edge_graph_cycle_rank']==1 and result['voxel_skeleton_cycle_rank26']==0
    assert not result['edge_graph_matches_voxel_betti']


def test_source6_island_omission_is_not_hidden_by_source26_preservation(monkeypatch):
    mask=np.zeros((5,5,5),bool);mask[1,1,2]=mask[2,2,2]=True
    def thin(value,**kwargs):
        result=value.copy();result[tuple(np.argwhere(result)[0])]=False;return result
    monkeypatch.setattr(geometry,'skeletonize',thin)
    result=geometry.vessel_graph(mask,np.eye(4))
    assert result['voxel_skeleton_betti_preserved26']
    assert result['unrepresented_source26_voxels']==0 and result['unrepresented_source6_voxels']==1


def test_native_inspector_exposes_source_fov_and_preserves_source_id(tmp_path):
    class Collection:
        root=tmp_path
        def layer(self,name):return {'kind':'artery','data':{'sha256':'test-source'},'label_value':1}
        def patient_affine(self,row):return np.eye(4)
        def mask(self,row):
            value=np.zeros((5,5,5),bool);value[:4,2,2]=True;return value
        def identity(self):return 'generated-source-boundary-control'
    report=geometry.inspect_vessel_geometry(Collection(),'artery',[0.,2.,2.])
    assert report['source_boundary_foreground_voxels']==1
    assert report['nearest']['source_cell_id']==12
    assert report['radius_is_lumen_measurement'] is False and report['flow_model_admitted'] is False


def test_h1_cap_retains_raw_diagnostics_but_no_partial_representative(monkeypatch):
    from resectionlab import vascular_homology
    def refuse(*args,**kwargs):raise vascular_homology.HomologyLimit('bounded test cap')
    monkeypatch.setattr(vascular_homology,'representative_subgraph',refuse)
    mask=np.zeros((5,5,5),bool);mask[1:4,2,2]=True
    result=geometry.vessel_graph(mask,np.eye(4))
    assert str(result['h1_status'])=='cap-abstained' and 'h1_edges' not in result
    assert len(result['source_cell_ids'])>0 and not result['h1_matches_source_betti']


def test_hollow_annotation_cavity_cannot_be_certified_by_a_one_dimensional_graph(monkeypatch):
    mask=np.ones((3,3,3),bool);mask[1,1,1]=False
    monkeypatch.setattr(geometry,'skeletonize',lambda value,**kwargs:value.copy())
    result=geometry.vessel_graph(mask,np.eye(4))
    assert result['source_cavities6_background']==result['voxel_skeleton_cavities6_background']==1
    assert result['source_cycle_rank26']==result['h1_betti1']==0
    assert result['voxel_skeleton_betti_preserved26']
    assert not result['h1_matches_source_betti']
