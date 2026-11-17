"""Physical placement and categorical-resampling controls for EDEN v2."""
import importlib.util
from pathlib import Path
import numpy as np
import nibabel as nib
import pytest

spec=importlib.util.spec_from_file_location('eden_anatomy',Path(__file__).resolve().parents[1]/'scripts/build_eden2020_anatomy.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


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
    # Some source overlays encode one side in the31000 band and the other in
    #32000. The same physical TRK/gap checks, not a single code band, govern it.
    values[3,4,5]=31001;nib.save(nib.Nifti1Image(values,np.eye(4)),path)
    both,_,receipt=builder.tract_overlay_mask(path,image)
    assert np.array_equal(both,mask) and receipt['foreground_min']==31001
    values[2,2,2]=20000;nib.save(nib.Nifti1Image(values,np.eye(4)),path)
    with pytest.raises(ValueError,match='intensity encoding'):builder.tract_overlay_mask(path,image)
