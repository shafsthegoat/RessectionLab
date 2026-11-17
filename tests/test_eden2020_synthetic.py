"""Generated source controls for the new seeded placement option."""
import numpy as np
import pytest
from resectionlab.data_synthesis import geometry, appearance


def test_seeded_random_lesions_respect_oblique_physical_support_and_preserve_baseline():
    grid=np.indices((80,80,80));brain=np.sum(((grid-39.5)/37.)**2,axis=0)<1
    angle=.4
    affine=np.eye(4);affine[:3,:3]=np.array([[-np.cos(angle),-np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
    affine[:3,3]=[70.,-50.,10.]
    support=brain.copy();support[:10]=False
    baseline=(100+grid[0]+grid[1]*.4+grid[2]*.2).astype(np.float32);unchanged=baseline.copy()
    centers=[]
    for seed in (7,11,19):
        masks,recipe=geometry(brain,affine,seed=seed,variant=seed%3,placement_support=support,random_placement=True)
        again,repeat=geometry(brain,affine,seed=seed,variant=seed%3,placement_support=support,random_placement=True)
        assert recipe==repeat and all(np.array_equal(masks[k],again[k]) for k in masks)
        assert np.allclose(recipe['center_ras_mm'],affine[:3,:3]@recipe['center_voxel']+affine[:3,3])
        total=sum(m.astype(np.uint8) for m in masks.values())
        assert total.max()==1 and not np.any((total>0)&~support)
        image,_=appearance(baseline,brain,masks,seed=seed)
        assert np.array_equal(baseline,unchanged)
        assert np.isfinite(image).all() and not np.array_equal(image,baseline)
        assert np.array_equal(image[total==0],baseline[total==0])
        centers.append(tuple(recipe['center_voxel']))
    assert len(set(centers))==3


def test_random_placement_does_not_relax_minimum_support_or_change_default():
    brain=np.zeros((40,40,40),bool);brain[15:25,15:25,15:25]=True
    with pytest.raises(ValueError,match='minimum engineering lesion'):
        geometry(brain,np.eye(4),seed=7,random_placement=True)
    brain=np.zeros((80,80,80),bool);brain[3:-3,3:-3,3:-3]=True
    first,r1=geometry(brain,np.eye(4),seed=8)
    second,r2=geometry(brain,np.eye(4),seed=8,random_placement=False)
    assert r1==r2 and all(np.array_equal(first[k],second[k]) for k in first)


def test_conflicting_auxiliary_header_is_retained_and_withheld_before_display_resampling(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import nibabel as nib
    import nibabel.processing
    spec=importlib.util.spec_from_file_location('eden_builder',Path(__file__).resolve().parents[1]/'scripts/build_eden2020_synthetic.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    subject='CTRL_0001';directory=tmp_path/'input';directory.mkdir()
    values=np.arange(12**3,dtype=np.float32).reshape((12,)*3)+1
    for suffix in ['T1_3D_PROSET_Sag','3D_FLAIR_Tra','s3DI_MC_HR','SWIp_axial']:
        image=nib.Nifti1Image(values,np.eye(4));image.header.set_xyzt_units('mm')
        image.set_qform(np.eye(4),1);transform=np.eye(4)
        if suffix=='3D_FLAIR_Tra':transform[0,3]=.2
        image.set_sform(transform,1);nib.save(image,directory/(subject+'_'+suffix+'.nii.gz'))
    brain=directory/'brain.nii.gz';mask=np.zeros((12,)*3,np.uint8);mask[2:-2,2:-2,2:-2]=1
    image=nib.Nifti1Image(mask,np.eye(4));image.header.set_xyzt_units('mm');image.set_qform(np.eye(4),1);image.set_sform(np.eye(4),1);nib.save(image,brain)
    provenance=tmp_path/'sources/eden2020-healthy-3994749/provenance';provenance.mkdir(parents=True)
    for name in ['zenodo-record.json','development-plan.json']:(provenance/name).write_text(json.dumps({'source':'generated test'}))
    monkeypatch.setenv('RESSECTIONLAB_DATA_ROOT','test-reset-after-fixture')
    monkeypatch.setenv('RESSECTIONLAB_MODEL_ROOT','test-reset-after-fixture')
    monkeypatch.setattr(builder,'unpack',lambda *args:(directory,{'archive_sha256':'1'*64}))
    monkeypatch.setattr(builder,'infer',lambda *args,**kwargs:{'outputs':[{'name':'brain_mask.nii.gz','path':str(brain)}]})
    monkeypatch.setattr(builder,'MAX_IMAGE_BYTES',1)
    original_resample=nibabel.processing.resample_to_output;resampled=[]
    def resample(image,**kwargs):
        resampled.append(image.get_filename());return original_resample(image,**kwargs)
    monkeypatch.setattr(nibabel.processing,'resample_to_output',resample)
    captured={}
    class BaselineReady(Exception):pass
    def stop_after_baseline(root,case,sources,originals,*args):
        captured.update(case=case,sources=sources,originals=originals);raise BaselineReady
    monkeypatch.setattr(builder,'package',stop_after_baseline)
    with pytest.raises(BaselineReady):builder.build_subject(tmp_path,subject)
    withheld=captured['case'].metadata['data_case']['withheld_reference_channels']
    assert len(withheld)==1 and withheld[0]['modality']=='FLAIR'
    assert 'QFORM_SFORM_DISAGREEMENT' in withheld[0]['reason']
    assert all('FLAIR' not in path for path in resampled) and len(resampled)==2
    assert [s.modality for s in captured['sources']]==['MRA','SWI']
    retained=next(r for r in captured['originals'] if 'FLAIR' in r.name)
    assert retained.data==(directory/(subject+'_3D_FLAIR_Tra.nii.gz')).read_bytes()
