"""Physical placement and categorical-resampling controls for EDEN v2."""
import importlib.util
from pathlib import Path
import numpy as np
import nibabel as nib

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
