import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('integrated_anatomy', Path(__file__).parents[1]/'scripts/data_integrated_anatomy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_native_enclosure_excludes_dense_vessel_and_unseen_open_slices():
    x,y=np.ogrid[:65,:65];radius=np.sqrt((x-32)**2+(y-32)**2)
    ct=np.full((65,65,20),-1000.,np.float32)
    ct[radius<27,:]=35
    ct[(radius>=24)&(radius<=27),:]=1000
    vessels=np.zeros_like(ct,np.uint8);vessels[31:34,31:34,:]=1;ct[vessels>0]=800
    # A truly open lower slice cannot be closed by the 2 mm kernel.
    ct[32:,:10,0:3]=-1000
    result=module.estimate(ct,vessels,(1.,1.,1.))
    assert result['brain_estimate'][20,32,10]
    assert not result['observed_skull'][vessels>0].any()
    assert not result['tumor_placement_support'][vessels>0].any()
    assert np.all(result['tumor_placement_support'] <= result['brain_estimate'])
    assert not result['enclosed_coverage'][:,:,0:3].any()
    assert result['observed_skull'][8,32,10]
    assert all(mask.shape==ct.shape for mask in result.values())


def test_physical_bounds_use_voxel_faces_and_real_affine():
    affine=np.diag([-2.,3.,4.,1.]);affine[:3,3]=[10,20,30]
    bounds=module.physical_bounds((5,6,7),affine)
    assert bounds['minimum_ras_mm']==[1.,18.5,28.]
    assert bounds['maximum_ras_mm']==[11.,36.5,56.]


def test_no_enclosed_anatomy_fails_without_fabrication():
    with pytest.raises(ValueError,match='No enclosed'):
        module.estimate(np.zeros((10,10,10)),np.zeros((10,10,10)),(1,1,1))


def test_existing_default_version_refuses_before_source_decode_or_estimation(monkeypatch,tmp_path):
    destination=tmp_path/'integrated-v1/anatomy/TopCoW-006/enclosure-v1';destination.mkdir(parents=True)
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:tmp_path)
    monkeypatch.setattr(module,'estimate',lambda *a,**k:pytest.fail('Existing immutable output must not be estimated again'))
    with pytest.raises(FileExistsError,match='Immutable output exists'):
        module.prepare(tmp_path,'006')


@pytest.mark.parametrize('version',['../escape','bad/name','has space',''])
def test_unsafe_version_is_rejected_before_external_root_access(monkeypatch,version):
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:pytest.fail('Invalid version must not access data root'))
    with pytest.raises(ValueError,match='safe immutable version'):
        module.prepare('/unused','006',version=version)


def test_additive_union_preserves_old_tissue_and_bone_and_physical_support():
    shape=(25,25,25)
    old=np.zeros(shape,bool);old[2:16,2:23,2:23]=True
    model=np.zeros(shape,bool);model[10:23,2:23,2:23]=True
    skull=np.zeros(shape,bool);skull[1,12,12]=True
    ct=np.full(shape,35.);vessels=np.zeros(shape,np.uint8);vessels[12,12,12]=1
    ct[7,13,12]=151.;before=[x.copy()for x in(old,model,skull)]
    result=module.additive_brain_proposal(ct,vessels,(1,1,1),old,skull,model)
    assert result['brain_estimate'][4,12,12]  # Old tissue excluded by model remains.
    assert result['brain_estimate'][20,12,12]  # New model extent is added.
    assert np.array_equal(result['observed_skull'],skull)
    assert not result['tumor_placement_support'][9,12,12]  # Exactly3mm from supplied vessel.
    assert result['tumor_placement_support'][8,12,12]  #4mm from vessel,7mm from brain edge.
    assert not result['tumor_placement_support'][4,12,12]  #3mm from exterior.
    assert result['tumor_placement_support'][5,12,12]  # Prior includes exactly4mm brain depth.
    assert not result['tumor_placement_support'][7,13,12]  # Original measured intensity unsupported.
    assert all(np.array_equal(a,b)for a,b in zip((old,model,skull),before))


@pytest.fixture
def additive_sources(tmp_path,monkeypatch):
    import nibabel as nib
    import json
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:tmp_path)
    shape=(15,17,19);affine=np.diag([-.7,.8,1.2,1.]);affine[:3,3]=[35.,-80.,120.]
    prior=tmp_path/'integrated-v1/anatomy/TopCoW-006/enclosure-v1';prior.mkdir(parents=True)
    pilot=prior.parent/'model-v1';pilot.mkdir()
    def save(path,data,aff=affine):
        header=nib.Nifti1Header();header.set_xyzt_units('mm');nib.save(nib.Nifti1Image(data,aff,header),path)
        return {'path':str(path.relative_to(tmp_path)),'sha256':module.sha256_file(path)}
    ct=np.full(shape,35.,np.float32);vessels=np.zeros(shape,np.uint8);vessels[7,8,9]=1
    image=save(tmp_path/'original-ct.nii.gz',ct);image['source_member']='release/imagesTr_topbrain/topcow_ct_006_0000.nii.gz'
    annotation=save(tmp_path/'original-label.nii.gz',vessels);annotation['source_member']='release/labelsTr_topbrain_v1_ct/topcow_ct_006.nii.gz'
    receipt={'family':{'family_id':'TopCoW:006','role':'TRAIN'},'files':[image,annotation]}
    catalog=tmp_path/'catalog/topbrain';catalog.mkdir(parents=True);(catalog/'006-selective-extraction.json').write_text(json.dumps(receipt))
    old=np.zeros(shape,np.uint8);old[2:9,2:15,2:17]=1
    model=np.zeros(shape,np.uint8);model[6:13,2:15,2:17]=1
    bone=np.zeros(shape,np.uint8);bone[1,8,9]=1
    outputs={name:save(prior/(name+'.nii.gz'),mask)for name,mask in {'brain_estimate':old,'observed_skull':bone,'enclosed_coverage':old,'tumor_placement_support':np.zeros(shape,np.uint8)}.items()}
    (prior/'provenance.json').write_text(json.dumps({'family_id':'TopCoW:006','sources':[image,annotation],'outputs':outputs}))
    weights=tmp_path/'cached-weight';weights.write_bytes(b'already cached model')
    method={'subject':'006','role':'TRAIN','source':image,'weights':[{'path':'cached-weight','sha256':module.sha256_file(weights)}]}
    (pilot/'prospective-method.json').write_text(json.dumps(method))
    # Deliberately overlapping model skull must never become mechanical bone.
    model_outputs=[save(pilot/'brain_mask.nii.gz',model),save(pilot/'skull_mask.nii.gz',np.ones(shape,np.uint8))]
    result={'status':'succeeded_pending_independent_review','versions':{'TotalSegmentator':'fixture-version'},'outputs':model_outputs}
    (pilot/'result.json').write_text(json.dumps(result))
    return {'root':tmp_path,'pilot':pilot,'prior':prior,'old':old,'model':model,'bone':bone,'affine':affine,'save':save,'result':result}


def test_additive_native_proposal_binds_exact_inputs_and_copies_prior_bytes(additive_sources):
    import nibabel as nib
    f=additive_sources;root=f['root']
    result=module.prepare_additive(root,'006',version='additive-v1',model_pilot=str(f['pilot'].relative_to(root)))
    out=root/'integrated-v1/anatomy/TopCoW-006/additive-v1'
    saved=nib.load(out/'brain_estimate.nii.gz')
    assert np.array_equal(np.asarray(saved.dataobj),f['old']|f['model'])
    assert np.array_equal(saved.affine,nib.load(f['pilot']/'brain_mask.nii.gz').affine)
    for name in('observed_skull','enclosed_coverage'):
        assert (out/(name+'.nii.gz')).read_bytes()==(f['prior']/(name+'.nii.gz')).read_bytes()
    assert result['accepted'] is False and result['model_skull_mechanical'] is False
    assert result['task_domain_changed'] is False and result['native_exported'] is False
    assert result['model_versions']=={'TotalSegmentator':'fixture-version'}
    assert result['inputs']['model_skull']['sha256']==module.sha256_file(f['pilot']/'skull_mask.nii.gz')


@pytest.mark.parametrize('failure',['shifted-grid','nonbinary','nonfinite','wrong-source-hash'])
def test_additive_refuses_estimate_misbinding_without_resampling_or_writes(additive_sources,failure):
    import json
    f=additive_sources
    if failure=='wrong-source-hash':
        path=f['pilot']/'prospective-method.json';value=json.loads(path.read_text());value['source']['sha256']='0'*64;path.write_text(json.dumps(value))
    else:
        data=f['model'].astype(float);aff=f['affine'].copy()
        if failure=='shifted-grid':aff[0,3]+=.1
        if failure=='nonbinary':data[3,3,3]=.5
        if failure=='nonfinite':data[3,3,3]=np.nan
        f['result']['outputs'][0]=f['save'](f['pilot']/'brain_mask.nii.gz',data,aff)
        (f['pilot']/'result.json').write_text(json.dumps(f['result']))
    with pytest.raises(ValueError):
        module.prepare_additive(f['root'],'006',version='refused-v1',model_pilot=str(f['pilot'].relative_to(f['root'])))
    assert not(f['prior'].parent/'refused-v1').exists()


def test_additive_existing_version_refuses_before_input_read(monkeypatch,tmp_path):
    destination=tmp_path/'integrated-v1/anatomy/TopCoW-006/existing';destination.mkdir(parents=True)
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:tmp_path)
    with pytest.raises(FileExistsError):
        module.prepare_additive(tmp_path,'006',version='existing',model_pilot='missing')


def test_additive_only_pinned_people_before_root_access(monkeypatch):
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:pytest.fail('Unpinned source must not access root'))
    with pytest.raises(ValueError,match='five pinned TRAIN'):
        module.prepare_additive('/unused','013',version='new',model_pilot='missing')
