"""Physical placement and categorical-resampling controls for EDEN v2."""
import importlib.util
from pathlib import Path
import numpy as np
import nibabel as nib
import pytest

spec=importlib.util.spec_from_file_location('eden_anatomy',Path(__file__).resolve().parents[1]/'scripts/build_eden2020_anatomy.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
planning_spec=importlib.util.spec_from_file_location('eden_planning',Path(__file__).resolve().parents[1]/'scripts/prepare_eden2020_planning.py')
planning=importlib.util.module_from_spec(planning_spec);planning_spec.loader.exec_module(planning)


def test_eight_placement_strata_follow_physical_ras_with_reflected_oblique_grid():
    grid=np.indices((112,112,112));domain=np.sum(((grid-55.5)/51.)**2,axis=0)<1
    a=np.eye(4);theta=.2;a[:3,:3]=[[-np.cos(theta),-np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]];a[:3,3]=[60,-20,-40]
    groups,distance=builder.placements(domain,a)
    assert len(groups)==8 and len({name for _,name in groups})==8
    xyz=np.argwhere(domain)@a[:3,:3].T+a[:3,3];low,high=np.percentile(xyz,[1,99],axis=0);mid=(low+high)/2
    for candidates,name in groups:
        points=candidates@a[:3,:3].T+a[:3,3]
        assert np.all(points[:,0]>mid[0]+18)if name.startswith('right')else np.all(points[:,0]<mid[0]-18)
        assert np.all(points[:,1]<mid[1]-3)if 'posterior'in name else np.all(points[:,1]>mid[1]+3)
        assert np.all(points[:,2]<mid[2]-3)if name.endswith('inferior')else np.all(points[:,2]>mid[2]+3)
        assert np.all(domain[tuple(candidates.T)]) and np.all(distance[tuple(candidates.T)]>=3)


def test_label_sampling_uses_physical_translation_and_header_conflict_stays_visible():
    data=np.zeros((12,12,12),np.uint16);data[3:5,4:6,5:7]=915
    moving=np.eye(4);moving[0,3]=2.
    result=builder.sample(data,moving,((12,12,12),np.eye(4)),0)
    assert np.array_equal(np.argwhere(result==915),np.argwhere(data==915)+[2,0,0])
    image=nib.Nifti1Image(data,moving);image.set_qform(np.eye(4),1);image.set_sform(moving,1);image.header.set_xyzt_units('mm')
    header=builder.source_header(image)
    assert header['maximum_qform_sform_corner_difference_mm']==2.
    assert header['working_basis']=='coded sform' and header['clinical_frame_adjudicated']is False
    assert builder.material(915)=='bone' and builder.material(4)=='fluid'


def test_published_tract_overlay_never_turns_its_background_mri_into_a_tract(tmp_path):
    values=np.full((12,12,12),100.,np.float32);values[3:6,4,5]=32001
    image=nib.Nifti1Image(values,np.eye(4));image.header.set_xyzt_units('mm')
    path=tmp_path/'CTRL_0001_CSTs_on_3DFLAIR.nii.gz';nib.save(image,path)
    header=nib.streamlines.TrkFile.create_empty_header();header['voxel_to_rasmm']=np.eye(4);header['dimensions']=[12,12,12];header['voxel_sizes']=[1,1,1];header['voxel_order']=b'RAS'
    tractogram=nib.streamlines.Tractogram([np.array([[3,4,5],[4,4,5],[5,4,5]],np.float32)],affine_to_rasmm=np.eye(4))
    nib.streamlines.save(nib.streamlines.TrkFile(tractogram,header),tmp_path/'CTRL_0001_CSTs_to_3DFLAIR.trk')
    mask,_,receipt=builder.tract_overlay_mask(path,image)
    assert mask.sum()==3 and receipt['tract_points_inside_overlay_fraction']==1.
    # Some source overlays include31000-band values as well as32000-band values.
    # The same physical TRK/gap checks, not a single code band, govern them.
    values[3,4,5]=31001;nib.save(nib.Nifti1Image(values,np.eye(4)),path)
    both,_,receipt=builder.tract_overlay_mask(path,image)
    assert np.array_equal(both,mask) and receipt['foreground_min']==31001
    values[2,2,2]=20000;nib.save(nib.Nifti1Image(values,np.eye(4)),path)
    with pytest.raises(ValueError,match='intensity encoding'):builder.tract_overlay_mask(path,image)


def test_native_planning_tract_overlay_crosscheck_uses_physical_coordinates(tmp_path):
    a=np.array([[0,-2,0,31],[1.5,0,0,-17],[0,0,3,9],[0,0,0,1]],float)
    voxels=np.array([[3,4,5],[4,4,5],[5,4,5]],np.float32)
    points=nib.affines.apply_affine(a,voxels).astype(np.float32)
    values=np.full((12,12,12),100,np.float32);values[tuple(voxels.astype(int).T)]=32001
    image=nib.Nifti1Image(values,a);image.header.set_xyzt_units('mm')
    overlay=tmp_path/'bundle_on_3DFLAIR.nii.gz';nib.save(image,overlay)
    header=nib.streamlines.TrkFile.create_empty_header();header['voxel_to_rasmm']=a;header['dimensions']=[12]*3;header['voxel_sizes']=nib.affines.voxel_sizes(a)
    tractogram=nib.streamlines.Tractogram([points],affine_to_rasmm=np.eye(4))
    trk=tmp_path/'bundle_to_3DFLAIR.trk';nib.streamlines.TrkFile(tractogram,header).save(trk)
    proof=planning.published_tract_alignment(points,trk,overlay,image)
    assert proof['source_frame_verified'] and proof['points_on_supplied_overlay_fraction']==1
    assert proof['registration_accuracy_measured'] is False
    with pytest.raises(ValueError,match='cross-check'):
        planning.published_tract_alignment(points+[0,0,3],trk,overlay,image)
    different=nib.Nifti1Image(values,a@np.diag([1,1,2,1]))
    with pytest.raises(ValueError,match='matching FLAIR grid'):
        planning.published_tract_alignment(points,trk,overlay,different)


def test_native_planning_entry_stays_on_local_superior_outer_envelope():
    voxels=np.indices((32,32,32));head=np.sum(((voxels-15.5)/13)**2,axis=0)<=1
    a=np.array([[-1,0,0,40],[0,1,0,-20],[0,0,1,-5],[0,0,0,1]],float)
    target=nib.affines.apply_affine(a,[13,16,13])
    entry=planning.superior_entry(head,a,target)
    assert entry is not None and entry[2]>target[2]+8
    assert np.linalg.norm(np.asarray(entry)[:2]-target[:2])<=4
    index=np.rint(nib.affines.apply_affine(np.linalg.inv(a),entry)).astype(int)
    assert head[tuple(index)] and not head[tuple(index+[0,0,1])]
    assert planning.superior_entry(np.zeros_like(head),a,target) is None
