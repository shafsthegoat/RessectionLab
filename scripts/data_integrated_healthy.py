#!/usr/bin/env python3
"""Same-person IXI MRA integration; explicit physical rigid mapping, pending QC."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import time
import urllib.request
import zlib
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from resectionlab.data_root import checked_root, atomic_json, sha256_file, safe_path
from resectionlab.prior_registration import image_from_ras, linear_transform_matrix, FRAME_FLIP

REPO=Path(__file__).resolve().parents[1]
SUBJECTS=('IXI077','IXI081')
GENERATOR_SOURCE=Path(__file__).read_bytes()
GENERATOR_SHA=hashlib.sha256(GENERATOR_SOURCE).hexdigest()
ARCHIVE_SHA='6b5e425001208cb11685ff610b5b15ee388b73d22bfede5237b0da9148276b93'
ARCHIVE_MD5='96be3dd9d1413e8890f89721829a79f7'
URL='https://zenodo.org/api/records/17393202/files/vessel_dataset.zip/content'


def sources(root):
    root=checked_root(root,required_bytes=1024**3)
    base=root/'integrated-v1/healthy';base.mkdir(parents=True,exist_ok=True)
    cohort_path=REPO/'manifests/experiments/ixi-component-person-cohort-v1.json'
    cohort=json.loads(cohort_path.read_text())
    acquisition=json.loads((root/'catalog/source-acquisition.json').read_text())
    selected=[]
    for subject in SUBJECTS:
        person='IXI:'+subject[3:]
        member=next(row for row in cohort['members'] if row['person_group']==person)
        if member['role']!='TRAIN' or not member['annotation_available']:raise ValueError('Only frozen TRAIN annotated people')
        acquired=next(row for row in acquisition['candidates'] if row['person_id']==person)
        selected.append({'subject':subject,'person_id':person,'role':'TRAIN','annotation':member['vessel_annotation'],
                         'acquired_files':acquired['acquired_files']})
    plan={'schema':'integrated-healthy-source-plan-v1','cohort_sha256':sha256_file(cohort_path),
        'selection':'first two already-acquired frozen TRAIN people with source vascular labels, selected from metadata before label decoding',
        'people':selected,'archive_sha256':ARCHIVE_SHA,'source_url':URL,'label_license':'CC BY4.0',
        'raw_license':'CC BY-SA3.0','label_evidence':'source model-assisted/manual-refined incomplete vascular reference; not clinical ground truth',
        'no_skull_CT':True,'no_mass_effect':True,'new_source_people':0}
    plan_path=base/'source-plan-v1.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen source plan changed')
    if not plan_path.exists():atomic_json(plan_path,plan)
    archive=root/'sources/ixi-vascular-zenodo17393202/vessel_dataset.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        identical=next((p for p in (root/'sources').rglob('*') if p.is_file() and not p.name.startswith('.')
            and p.stat().st_size==6580597 and sha256_file(p)==ARCHIVE_SHA),None)
        if identical:shutil.copyfile(identical,archive)
        else:
            temporary=archive.with_suffix('.zip.partial')
            if temporary.exists():raise ValueError('Preserve previous partial; inspect before retry')
            with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'RessectionLab-local-research/1.0'}),timeout=40) as response,temporary.open('xb') as output:
                if response.status!=200:raise ValueError('Expected ordinary public download')
                count=0
                while block:=response.read(1024**2):
                    count+=len(block)
                    if count>6580597:raise ValueError('Unexpected archive size')
                    output.write(block)
            if temporary.stat().st_size!=6580597 or sha256_file(temporary)!=ARCHIVE_SHA:raise ValueError('Source archive integrity mismatch')
            temporary.replace(archive)
    if sha256_file(archive)!=ARCHIVE_SHA or hashlib.md5(archive.read_bytes()).hexdigest()!=ARCHIVE_MD5:raise ValueError('Frozen archive changed')
    reports=[]
    with ZipFile(archive) as zipped:
        for row in selected:
            destination=base/row['subject']/'source';destination.mkdir(parents=True,exist_ok=True)
            member=row['annotation']['member'];info=zipped.getinfo(member)
            path=destination/'supplied-vascular-reference.nii.gz'
            if not path.exists():
                with zipped.open(info) as source,path.open('xb') as output:shutil.copyfileobj(source,output)
            if path.stat().st_size!=info.file_size or zlib.crc32(path.read_bytes())!=info.CRC:
                raise ValueError('Extracted annotation differs from frozen archive member')
            mask=nib.load(path)
            files={entry['modality']:entry for entry in row['acquired_files'] if entry['path'].endswith('.nii.gz')}
            mra=nib.load(safe_path(root,files['MRA']['path']))
            for entry in files.values():
                if sha256_file(safe_path(root,entry['path']))!=entry['sha256']:raise ValueError('Original acquisition changed')
            report={**row,'annotation_path':str(path.relative_to(root)),'annotation_sha256':sha256_file(path),
                'zip_member_crc32':info.CRC,'shape':list(mask.shape),'affine_ras_mm':mask.affine.tolist(),
                'units':mask.header.get_xyzt_units()[0],'mra_shape':list(mra.shape),'mra_affine_ras_mm':mra.affine.tolist(),
                'native_grid_equal':mask.shape==mra.shape and np.allclose(mask.affine,mra.affine,rtol=0,atol=1e-5),
                'labels':np.unique(np.asarray(mask.dataobj)).tolist(),'status':'source reference acquired; independent review required'}
            atomic_json(destination/'receipt.json',report);reports.append(report)
    atomic_json(base/'source-receipts-v1.json',{'schema':'integrated-healthy-source-receipts-v1','people':reports})
    print(json.dumps({'people':[(r['subject'],r['shape'],r['native_grid_equal']) for r in reports]}),flush=True)
    return reports


def normalized(data):
    values=data[data>0]
    lo,hi=np.percentile(values,[1,99.5])
    return np.clip((data-lo)/max(float(hi-lo),1e-6),0,1).astype(np.float32)


def save_native(path,array,reference):
    header=reference.header.copy();header.set_data_dtype(array.dtype);header.set_xyzt_units('mm')
    nib.save(nib.Nifti1Image(array,reference.affine,header),path)


def forward_ras_from_pull(transform):
    """Physical moving-RAS -> fixed-RAS inverse of SimpleITK fixed-LPS pull."""
    pull_lps=linear_transform_matrix(transform)
    return FRAME_FLIP@np.linalg.inv(pull_lps)@FRAME_FLIP


def warp_native(array, moving_affine, fixed_reference, transform, interpolation):
    """Sample native moving data at fixed physical points, never resize arrays."""
    image=image_from_ras(array,moving_affine)
    return sitk.GetArrayFromImage(sitk.Resample(image,fixed_reference,transform,interpolation,0)).transpose(2,1,0)


def register(root,subject):
    root=checked_root(root,required_bytes=2*1024**3)
    base=root/'integrated-v1/healthy'/subject
    receipt=json.loads((base/'source/receipt.json').read_text())
    files={entry['modality']:entry for entry in receipt['acquired_files'] if entry['path'].endswith('.nii.gz')}
    fixed_nifti=nib.load(safe_path(root,files['T1']['path']));moving_nifti=nib.load(safe_path(root,files['MRA']['path']))
    label=nib.load(safe_path(root,receipt['annotation_path']))
    for entry in (files['T1'],files['MRA']):
        if sha256_file(safe_path(root,entry['path']))!=entry['sha256']:raise ValueError('Acquired source hash changed')
    if sha256_file(safe_path(root,receipt['annotation_path']))!=receipt['annotation_sha256']:raise ValueError('Source vascular reference changed')
    if not receipt['native_grid_equal']:raise ValueError('Source label/MRA physical grid requires separate qualification before registration')
    output=base/'rigid-v1'
    if output.exists():raise FileExistsError('Keep previous registration immutable')
    output.mkdir();started=time.monotonic()
    fixed_array=np.asarray(fixed_nifti.dataobj,np.float32);moving_array=np.asarray(moving_nifti.dataobj,np.float32)
    control_path=root/'cases/controls'/(receipt['person_id'].replace(':','-')+'-control-native-v5.ressectionlab')
    from resectionlab.workspace_bundle import load_workspace
    control,_,_=load_workspace(control_path)
    if control.mri.shape!=fixed_array.shape or not np.array_equal(control.affine,fixed_nifti.affine):raise ValueError('Baseline native T1 grid differs')
    if not np.array_equal(control.mri,fixed_array):raise ValueError('Baseline native T1 intensities differ from acquired original')
    fixed=image_from_ras(normalized(fixed_array),fixed_nifti.affine)
    moving=image_from_ras(normalized(moving_array),moving_nifti.affine)
    brain=image_from_ras(control.brain_mask.astype(np.uint8),control.affine)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    initial=sitk.Euler3DTransform()
    centroid=control.voxel_to_world(np.argwhere(control.brain_mask).mean(0));initial.SetCenter(tuple((FRAME_FLIP@np.r_[centroid,1])[:3]))
    method=sitk.ImageRegistrationMethod();method.SetNumberOfThreads(2)
    method.SetMetricAsMattesMutualInformation(48);method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(.08,42771);method.SetMetricFixedMask(brain)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.,minStep=.01,numberOfIterations=90,relaxationFactor=.5)
    method.SetOptimizerScalesFromPhysicalShift();method.SetShrinkFactorsPerLevel([4,2,1]);method.SetSmoothingSigmasPerLevel([2,1,0]);method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial,inPlace=False)
    def timeout():
        if time.monotonic()-started>120:method.StopRegistration()
    method.AddCommand(sitk.sitkIterationEvent,timeout)
    initial_metric=float(method.MetricEvaluate(fixed,moving))
    transform=method.Execute(fixed,moving)
    pull_lps=linear_transform_matrix(transform);forward_ras=forward_ras_from_pull(transform)
    registered=warp_native(moving_array,moving_nifti.affine,fixed,transform,sitk.sitkLinear).astype(np.float32)
    vessels=warp_native((np.asarray(label.dataobj)>0).astype(np.uint8),label.affine,fixed,transform,sitk.sitkNearestNeighbor).astype(np.uint8)
    observed=warp_native(np.ones(moving_array.shape,np.uint8),moving_nifti.affine,fixed,transform,sitk.sitkNearestNeighbor).astype(np.uint8)
    for name,array in [('registered-MRA',registered),('registered-vascular-reference',vessels),('MRA-observed-coverage',observed)]:save_native(output/(name+'.nii.gz'),array,fixed_nifti)
    sitk.WriteTransform(transform,str(output/'fixed-T1-LPS-to-moving-MRA-LPS-pull.tfm'))
    result={'schema':'integrated-healthy-rigid-registration-v1','subject':subject,'person_id':receipt['person_id'],'role':'TRAIN',
        'source_receipt':str((base/'source/receipt.json').relative_to(root)),'source_receipt_sha256':sha256_file(base/'source/receipt.json'),
        'fixed_T1':files['T1'],'moving_MRA':files['MRA'],'source_label_sha256':receipt['annotation_sha256'],
        'method':'SimpleITK Euler3D rigid MattesMI48; source scanner identity start, fixed estimated-brain metric; 2threads, seed42771, multires4/2/1, 90iterations/level',
        'fixed_T1_RAS_to_moving_MRA_RAS_pull':(FRAME_FLIP@pull_lps@FRAME_FLIP).tolist(),
        'moving_MRA_RAS_to_fixed_T1_RAS_forward':forward_ras.tolist(),'transform_direction':'resampling uses fixed T1 point -> original MRA point; source vessels use SAME pull',
        'initial_MI_metric':initial_metric,'final_MI_metric':float(method.GetMetricValue()),'optimizer_stop':method.GetOptimizerStopConditionDescription(),
        'seconds':time.monotonic()-started,'clinical_alignment_validated':False,'independent_alignment_review':'pending',
        'timing':'same IXI subject and acquisition code; exact source acquisition dates unavailable; no temporal-change claim',
        'native_primary_T1_unchanged':True,'interpolation':{'image':'linear','labels':'nearest-neighbor','coverage':'nearest-neighbor'},
        'processing_dtype':{'image':'source scaled float32 -> SITKfloat32 linear -> storedfloat32','labels':'uint8 nearest-neighbor','coverage':'uint8 nearest-neighbor'},
        'generator_sha256':GENERATOR_SHA,
        'outputs':[{'path':str(p.relative_to(root)),'sha256':sha256_file(p)} for p in output.iterdir() if p.is_file() and not p.name.startswith('.')],
        'limits':['Vascular labels are incomplete model-assisted/source-refined references','Absent label or outside MRA slab is unknown','No skull CT/function/tracts','Similarity metric does not grant anatomical acceptance']}
    (output/'data_integrated_healthy.py.snapshot').write_bytes(GENERATOR_SOURCE)
    atomic_json(output/'registration.json',result);print(json.dumps({'subject':subject,'registered_vessel_voxels':int(vessels.sum()),'seconds':result['seconds'],'forward_ras':forward_ras.tolist()}),flush=True)
    return result


def canonical_array(array,affine):
    """Axis permutation/flips only for review; preserve source obliquity."""
    array=np.asarray(array)
    if array.dtype==bool:array=array.astype(np.uint8)
    image=nib.as_closest_canonical(nib.Nifti1Image(array,affine))
    return np.asarray(image.dataobj),image.affine


def registration_review(root,subject):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resectionlab.workspace_bundle import load_workspace
    root=checked_root(root,required_bytes=1024**3);base=root/'integrated-v1/healthy'/subject
    path=base/'rigid-v1/registration.json';record=json.loads(path.read_text())
    output=base/'review-v1'
    if output.exists():raise FileExistsError('Immutable review output exists')
    output.mkdir()
    fixed=nib.load(safe_path(root,record['fixed_T1']['path']))
    moving=nib.load(safe_path(root,record['moving_MRA']['path']))
    receipt=json.loads((base/'source/receipt.json').read_text());label=nib.load(safe_path(root,receipt['annotation_path']))
    arrays={}
    for name,filename in [('mra','registered-MRA'),('vessels','registered-vascular-reference'),('coverage','MRA-observed-coverage')]:
        image=nib.load(base/'rigid-v1'/(filename+'.nii.gz'))
        if image.shape!=fixed.shape or not np.array_equal(image.affine,fixed.affine):raise ValueError('Derived grid changed')
        arrays[name]=np.asarray(image.dataobj)
    control,_,_=load_workspace(root/'cases/controls'/(record['person_id'].replace(':','-')+'-control-native-v5.ressectionlab'))
    t1,affine=canonical_array(np.asarray(fixed.dataobj,np.float32),fixed.affine)
    views={name:canonical_array(array,fixed.affine)[0] for name,array in arrays.items()}
    brain=canonical_array(control.brain_mask.astype(np.uint8),fixed.affine)[0]>0
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    supported=brain&(views['coverage']>0);coordinates=np.array(np.where(supported));lo=coordinates.min(1);hi=coordinates.max(1)
    t1norm=normalized(t1);mranorm=normalized(views['mra'])
    images=[]
    for fraction in [.25,.5,.75]:
        anchor=np.rint(lo+fraction*(hi-lo)).astype(int)
        fig,axes=plt.subplots(3,3,figsize=(12,12))
        for axis,name in enumerate(['Sagittal R+','Coronal A+','Axial S+']):
            remaining=[a for a in range(3) if a!=axis];extent=[0,t1.shape[remaining[0]]*spacing[remaining[0]],0,t1.shape[remaining[1]]*spacing[remaining[1]]]
            first=np.take(t1norm,anchor[axis],axis=axis).T;second=np.take(mranorm,anchor[axis],axis=axis).T
            vessel=np.take(views['vessels']>0,anchor[axis],axis=axis).T;coverage=np.take(views['coverage']>0,anchor[axis],axis=axis).T
            for column in range(3):
                ax=axes[axis,column];ax.imshow(first if column<2 else second,cmap='gray',origin='lower',vmin=0,vmax=1,extent=extent)
                if column==1:ax.imshow(second,cmap='magma',origin='lower',vmin=0,vmax=1,alpha=.6*coverage*(second>.05),extent=extent)
                if column>0 and vessel.any():ax.contour(vessel,levels=[.5],colors=['cyan'],linewidths=.45,extent=extent)
                if coverage.any() and not coverage.all():ax.contour(coverage,levels=[.5],colors=['yellow'],linewidths=.5,extent=extent)
                ax.set_title(f'{name} index={anchor[axis]} | '+['T1','T1 + registered MRA','registered MRA + supplied vessels'][column],fontsize=8)
                ax.set_xlabel('native RAS-oriented axis distance mm');ax.set_ylabel('mm')
        fig.suptitle(f'{subject}: distributed physical rigid registration review, support fraction {fraction}\nCyan partial source vascular reference; yellow observed MRA slab edge. Axis flips/permutation only; obliquity retained. No acceptance from MI.')
        fig.tight_layout();destination=output/f'fixed-overlay-{int(fraction*100):02d}.png';fig.savefig(destination,dpi=115);plt.close(fig);images.append(str(destination.relative_to(root)))
    original,original_affine=canonical_array(np.asarray(moving.dataobj,np.float32),moving.affine)
    native_label=canonical_array(np.asarray(label.dataobj),label.affine)[0]>0
    coordinates=np.array(np.where(native_label));lo=coordinates.min(1);hi=coordinates.max(1);native_spacing=np.linalg.norm(original_affine[:3,:3],axis=0)
    fig,axes=plt.subplots(3,3,figsize=(12,12));original_norm=normalized(original)
    for axis,name in enumerate(['Sagittal R+','Coronal A+','Axial S+']):
        remaining=[a for a in range(3) if a!=axis];extent=[0,original.shape[remaining[0]]*native_spacing[remaining[0]],0,original.shape[remaining[1]]*native_spacing[remaining[1]]]
        for column,fraction in enumerate([.25,.5,.75]):
            index=int(round(lo[axis]+fraction*(hi[axis]-lo[axis])));section=np.take(native_label,index,axis=axis).T
            ax=axes[axis,column];ax.imshow(np.take(original_norm,index,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=1,extent=extent)
            if section.any():ax.contour(section,levels=[.5],colors=['cyan'],linewidths=.5,extent=extent)
            ax.set_title(f'{name} source index={index}');ax.set_xlabel('source native axis mm');ax.set_ylabel('mm')
    fig.suptitle(f'{subject}: unchanged original native MRA + original supplied partial vascular reference\nOriginal source affine/obliquity retained; no T1 resampling in this panel')
    fig.tight_layout();destination=output/'native-MRA-source.png';fig.savefig(destination,dpi=115);plt.close(fig);images.append(str(destination.relative_to(root)))
    forward=np.asarray(record['moving_MRA_RAS_to_fixed_T1_RAS_forward']);pull=np.asarray(record['fixed_T1_RAS_to_moving_MRA_RAS_pull'])
    review={'schema':'integrated-healthy-registration-review-v1','subject':subject,'registration_sha256':sha256_file(path),'registration':record,
            'primary_T1_unchanged':True,'source_images':{'T1':record['fixed_T1'],'MRA':record['moving_MRA']},'source_label_sha256':receipt['annotation_sha256'],
            'forward_determinant':float(np.linalg.det(forward[:3,:3])),'pull_forward_inverse_max_error':float(abs(pull@forward-np.eye(4)).max()),
            'T1_estimated_brain_voxels':int(control.brain_mask.sum()),'brain_observed_MRA_fraction':float((control.brain_mask&(arrays['coverage']>0)).sum()/control.brain_mask.sum()),
            'registered_vessel_voxels':int((arrays['vessels']>0).sum()),'vessel_voxels_outside_coverage':int(((arrays['vessels']>0)&~(arrays['coverage']>0)).sum()),
            'images':[{'path':name,'sha256':sha256_file(safe_path(root,name))} for name in images],'independent_review':'pending; distributed anatomical landmark comparison required',
            'display':'Only axis permutation/flips for image panels; original obliquity retained; original MRA source panel independent native grid',
            'unknown_coverage':'Outside transformed native MRA acquisition slab is unknown; absence of source label is not vessel absence','script_sha256':GENERATOR_SHA}
    (output/'data_integrated_healthy.py.snapshot').write_bytes(GENERATOR_SOURCE)
    atomic_json(output/'review.json',review);return review


def target_proposal(root,subject):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.core import array_digest
    from data_integrated_cases import ellipsoid
    root=checked_root(root,required_bytes=1024**3);base=root/'integrated-v1/healthy'/subject
    registration=base/'rigid-v1/registration.json';record=json.loads(registration.read_text())
    source=root/'cases/controls'/(record['person_id'].replace(':','-')+'-control-native-v5.ressectionlab')
    host,_,_=load_workspace(source);brain=host.brain_mask
    coverage=np.asarray(nib.load(base/'rigid-v1/MRA-observed-coverage.nii.gz').dataobj)>0
    vessels=np.asarray(nib.load(base/'rigid-v1/registered-vascular-reference.nii.gz').dataobj)>0
    spacing=np.linalg.norm(host.affine[:3,:3],axis=0)
    median=ndi.median_filter(host.mri,size=3);threshold=float(np.percentile(host.mri[brain&(host.mri>0)],40))
    vessel_distance=ndi.distance_transform_edt(~vessels,sampling=spacing)
    support=(brain&coverage&(median>=threshold)&(ndi.distance_transform_edt(brain,sampling=spacing)>=4.)
             &(ndi.distance_transform_edt(coverage,sampling=spacing)>=4.)&(vessel_distance>=5.))
    distance=ndi.distance_transform_edt(support,sampling=spacing);center_ijk=np.unravel_index(distance.argmax(),distance.shape)
    radius=min(4.,float(distance[center_ijk])-1.)
    if radius<2.5:raise ValueError('No adequate high-T1 observed-slab target support')
    center=host.voxel_to_world(center_ijk);target=ellipsoid(host.mri.shape,host.affine,center,[radius]*3)
    if not np.all(support[target]):raise ValueError('Physical target escaped supported tissue')
    output=base/'target-proposal-v2'
    if output.exists():raise FileExistsError('Immutable target proposal exists')
    checked_root(root);output.mkdir()
    reference=nib.load(safe_path(root,record['fixed_T1']['path']))
    for name,array in [('target',target.astype(np.uint8)),('tissue-placement-support',support.astype(np.uint8))]:save_native(output/(name+'.nii.gz'),array,reference)
    arrays={name:canonical_array(array,host.affine)[0] for name,array in [('T1',host.mri),('target',target),('brain',brain),('vessels',vessels),('coverage',coverage)]}
    _,affine=canonical_array(host.mri,host.affine);voxel=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),center)).astype(int);view_spacing=np.linalg.norm(affine[:3,:3],axis=0)
    image=normalized(arrays['T1']);fig,axes=plt.subplots(3,3,figsize=(12,12))
    for axis,name in enumerate(['Sagittal R+','Coronal A+','Axial S+']):
        remaining=[a for a in range(3) if a!=axis];extent=[0,image.shape[remaining[0]]*view_spacing[remaining[0]],0,image.shape[remaining[1]]*view_spacing[remaining[1]]]
        for column in range(3):
            ax=axes[axis,column];ax.imshow(np.take(image,voxel[axis],axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=1,extent=extent)
            if column!=1:
                for layer,color in [('brain','lime'),('target','yellow'),('vessels','red'),('coverage','orange')]:
                    section=np.take(arrays[layer],voxel[axis],axis=axis).T
                    if section.any() and not section.all():ax.contour(section,levels=[.5],colors=[color],linewidths=.6,extent=extent)
            if column>0:
                xy=voxel[remaining]*view_spacing[remaining];ax.set_xlim(xy[0]-18,xy[0]+18);ax.set_ylim(xy[1]-18,xy[1]+18)
            ax.set_title(f'{name} | '+['whole source context','unchanged source zoom','source + synthetic geometry zoom'][column],fontsize=9);ax.set_xlabel('native RAS-oriented axis mm');ax.set_ylabel('mm')
    fig.suptitle(f'{subject}: target proposal in original T1 estimated brain and actual registered MRA observed slab\nYellow synthetic geometry, green estimated brain, red incomplete source vessels, orange slab edge. Skull missing; zero mass effect.')
    fig.tight_layout();fig.savefig(output/'target-three-plane.png',dpi=120);plt.close(fig)
    result={'schema':'integrated-healthy-target-proposal-v1','subject':subject,'person_id':record['person_id'],'role':'TRAIN','accepted':False,
            'source_native_path':str(source.relative_to(root)),'source_native_sha256':sha256_file(source),'primary_T1_sha256':record['fixed_T1']['sha256'],
            'registration_sha256':sha256_file(registration),'registered_vessel_sha256':sha256_file(base/'rigid-v1/registered-vascular-reference.nii.gz'),
            'coverage_sha256':sha256_file(base/'rigid-v1/MRA-observed-coverage.nii.gz'),'center_ras_mm':center.tolist(),'center_native_ijk':list(map(int,center_ijk)),
            'radii_mm':[radius]*3,'target_voxels':int(target.sum()),'target_array_hash':array_digest(target),'target_sha256':sha256_file(output/'target.nii.gz'),
            'support_sha256':sha256_file(output/'tissue-placement-support.nii.gz'),'target_support_fraction':float(support[target].mean()),
            'target_original_brain_estimate_fraction':float(brain[target].mean()),'target_actual_MRA_coverage_fraction':float(coverage[target].mean()),
            'minimum_supplied_vessel_distance_mm':float(vessel_distance[target].min()),'high_T1_tissue_threshold':threshold,
            'minimum_target_median_T1':float(median[target].min()),'method':'3x3x3median T1 >= original estimated brain40thpercentile; >=4mm original brain/slab boundary; >=5mm supplied vascular distance; RAS physical sphere max4mm radius with1mm support margin',
            'primary_source_intensities_unchanged':True,'appearance':'geometry only; no observed tumor or synthesized MRI intensity','mass_effect':False,'skull':'missing; no MRI-derived bone estimate',
            'review_image':str((output/'target-three-plane.png').relative_to(root)),'review_image_sha256':sha256_file(output/'target-three-plane.png'),
            'limitations':['Source brain is model estimate; high T1 proxy is not measured tissue class','Registered labels incomplete source model/manual-refined vascular reference; clearance is not vessel-free truth','Target and registration require independent anatomical review before accepted','No skull anatomy, craniotomy mechanics, tractography or function'],
            'script_sha256':GENERATOR_SHA}
    (output/'data_integrated_healthy.py.snapshot').write_bytes(GENERATOR_SOURCE)
    atomic_json(output/'proposal.json',result);return result


def build_native(root,subject,*,registration_decision=None,target_decision=None,version='geometry-v1'):
    """Export a review-bound zero-mass geometry scenario on unchanged native T1."""
    from dataclasses import replace
    from resectionlab.core import SourceRef,array_digest,thaw_json
    from resectionlab.workspace_bundle import load_workspace,save_workspace,view_state,OriginalFile
    from resectionlab.workspace_imaging import DisplaySource,source_registration
    from resectionlab.imaging import load_nifti_case
    from resectionlab.surgical_scene import from_case
    root=checked_root(root,required_bytes=2*1024**3);base=root/'integrated-v1/healthy'/subject
    if not re.fullmatch(r'[A-Za-z0-9_-]+',version):raise ValueError('Plain immutable version token required')
    registration_path=base/'rigid-v1/registration.json';registration=json.loads(registration_path.read_text())
    proposal_path=base/'target-proposal-v2/proposal.json';proposal=json.loads(proposal_path.read_text())
    if sha256_file(registration_path)!=proposal['registration_sha256']:raise ValueError('Target belongs to another registration')
    for entry in registration['outputs']:
        if sha256_file(safe_path(root,entry['path']))!=entry['sha256']:raise ValueError('Registration output changed')
    source=safe_path(root,proposal['source_native_path'])
    if sha256_file(source)!=proposal['source_native_sha256']:raise ValueError('Original native host changed')
    host,_,workspace=load_workspace(source)
    acquired_T1=nib.load(safe_path(root,registration['fixed_T1']['path']))
    if not np.array_equal(host.mri,np.asarray(acquired_T1.dataobj,np.float32)) or not np.array_equal(host.affine,acquired_T1.affine):raise ValueError('Native primary differs from original acquired T1')
    if workspace is None:raise ValueError('Source original-file bundle missing')
    for entry in [registration['fixed_T1'],registration['moving_MRA']]:
        if sha256_file(safe_path(root,entry['path']))!=entry['sha256']:raise ValueError('Acquired source changed')
    arrays={}
    for name,path,expected in [('synthetic_target',base/'target-proposal-v2/target.nii.gz',proposal['target_sha256']),
                               ('registered_partial_vessels',base/'rigid-v1/registered-vascular-reference.nii.gz',proposal['registered_vessel_sha256']),
                               ('MRA-observed-coverage',base/'rigid-v1/MRA-observed-coverage.nii.gz',proposal['coverage_sha256'])]:
        if sha256_file(path)!=expected:raise ValueError('Derived mask checksum differs')
        image=nib.load(path)
        if image.shape!=host.mri.shape or not np.array_equal(image.affine,host.affine):raise ValueError('Derived mask not exact native T1 grid')
        arrays[name]=np.asarray(image.dataobj)>0
    target=arrays['synthetic_target'];vessels=arrays['registered_partial_vessels'];coverage=arrays['MRA-observed-coverage']
    if np.any(target&~host.brain_mask) or np.any(target&~coverage) or np.any(target&vessels):raise ValueError('Invalid target support')
    review_bindings={}
    for name,decision_path in [('registration',registration_decision),('target',target_decision)]:
        if decision_path is None:continue
        path=Path(decision_path).resolve()
        if not path.is_relative_to(root):raise ValueError('Independent decision must remain under data root')
        decision=json.loads(path.read_text())
        if not decision.get('reviewer'):raise ValueError('Independent reviewer missing')
        expected=sha256_file(registration_path if name=='registration' else proposal_path)
        if name=='registration':
            if decision.get('decision') not in {'accepted_for_bounded_engineering_alignment','accepted_for_registered_overlay_inspection_with_alignment_uncertainty'} or decision.get('registration_json_sha256',decision.get('registration_sha256'))!=expected:
                raise ValueError('Independent registration decision missing exact method binding')
        else:
            if decision.get('decision')!='accepted_for_geometry_target' or decision.get('proposal_sha256')!=expected:
                raise ValueError('Independent target decision missing exact proposal binding')
        review_bindings[name]={'path':str(path.relative_to(root)),'sha256':sha256_file(path),'decision':decision}
    case_id=proposal['person_id'].replace(':','-')+'-T1-MRA-INTEGRATED-'+version
    directory=root/'integrated-v1/cases'/case_id
    if directory.exists():raise FileExistsError('Immutable healthy native version exists')
    registered_path=base/'rigid-v1/registered-MRA.nii.gz'
    metadata=thaw_json(host.metadata);original_data=metadata['data_case'];restrictions=proposal['limitations']+registration['limits']+[
        'Primary T1 remains acquired baseline, synthetic target is geometry only; no tumor-present acquisition.',
        'Skull is missing; no bone substitute, craniotomy mechanics or whole-head access claim.',
        'Only native MRA slab is observed; outside coverage and unannotated vascular background remain unknown.',
        'Same person acquisition code; exact temporal compatibility is unknown. No mass effect or clinical function claim.',
        'Registered overlays have unresolved patient alignment uncertainty; no numeric patient error bound, precise navigation, corridor or clinical vascular-clearance claim.']
    layers={'brain':{'kind':'brain','label':'Brain · original SynthStrip estimate','provenance':'estimated','color':'#96b7d4'},
            'synthetic_target':{'kind':'tumor','label':'Synthetic geometry target · baseline T1','provenance':'simulated','color':'#f78db3'},
            'registered_partial_vessels':{'kind':'vessels','label':'Registered partial vessels · alignment uncertain','provenance':'estimated','color':'#e16c64','coverage':'Incomplete source model-assisted/manual-refined reference, native MRA observed slab only'}}
    data={**original_data,'case_id':case_id,'family_id':proposal['person_id'],'person_id':proposal['person_id'],'role':'TRAIN',
          'classification':'synthetic_scenario_on_acquired_healthy_host','case_type':'synthetic','accepted':False,'status':'pending_integrated_review',
          'components':{'T1':'unchanged_acquired_primary','MRA':'acquired_image_rigidly_derived_to_T1_plus_original_native','brain':'original_SynthStrip_estimate',
                        'vessels':'registered_model-assisted_source_annotation_with_unresolved_alignment_uncertainty','arteries':'not_typed_partial_source_annotation','veins':'unestablished_not_typed','skull':'missing','tumor':'synthetic_geometry_only','function':'missing','tracts':'missing'},
          'missing_anatomy':['measured_skull','complete_vasculature','functional_mapping','validated_tractography'],
          'capabilities':['native_T1_inspection','approximate_registered_partial_vascular_overlay_inspection','synthetic_geometry_fixture_exploration_with_registration_uncertainty'],
          'allowed_tasks':['approximate registered overlay inspection','geometry fixtures in observed reference grid with explicit alignment uncertainty'],
          'withheld_tasks':['precise_navigation_or_corridor','clinical_vascular_clearance','skull_drilling','functional_or_tract_safety','full_integrated_rehearsal'],
          'simulation_admission':'restricted_approximate_overlay_and_geometry_inspection_only_unresolved_alignment_uncertainty',
          'allowed_planner_observations':['structural_T1','estimated_brain','registered_partial_vascular_reference','explicit_synthetic_target'],
          'observations':{'T1':'acquired_unmodified_baseline','MRA':'original_native_and_rigidly_derived_on_T1_grid','tumor':'synthetic_geometry_only'},
          'recipe':proposal,'registration':registration,'independent_reviews':review_bindings,'limitations':restrictions,
          'mass_effect':False,'appearance':None,'deformation':None,'clinical_validation':False,'new_real_patient_count':0}
    metadata.update(data_case=data,anatomy_layers=layers,is_synthetic=True,selected_modality='T1',
                    display_label='Baseline healthy T1 + synthetic geometry target + partial registered vessels',
                    annotation_review='independent registration/target reviewed; integrated runtime pending' if len(review_bindings)==2 else 'pending',
                    benchmark_track='synthetic_engineering_development',surgical_adapter={'ownershipPrecedence':['brain','registered_partial_vessels','synthetic_target'],
                    'sourceCaseHash':host.semantic_hash,'sourcePackageSha256':proposal['source_native_sha256'],'coverageLimitations':restrictions,'MRAObservedCoverageSha256':proposal['coverage_sha256']})
    derived_data={**data,'case_type':'real','classification':'derived_acquired_MRA_display','components':{'MRA':'rigidly_resampled_acquired_image'},'recipe':None,'capabilities':['registered_MRA_source_inspection']}
    shared=None
    if 'registration' in review_bindings:
        binding=review_bindings['registration'];review=binding['decision']
        shared={'kind':'independently-reviewed-derived-registration-v1','patient_id':case_id,'source_person_id':proposal['person_id'],
                'group_id':proposal['person_id']+'-T1-MRA-rigid-v1','source_sha256_by_modality':{'T1':registration['fixed_T1']['sha256'],'MRA':sha256_file(registered_path)},
                'evidence_sha256':binding['sha256'],'evidence_path':binding['path'],'reviewer':review['reviewer'],
                'review_decision':review['decision'],'method_report_sha256':sha256_file(registration_path),
                'original_moving_image_sha256':registration['moving_MRA']['sha256'],'original_annotation_sha256':registration['source_label_sha256'],
                'original_moving_to_reference_ras_mm':registration['moving_MRA_RAS_to_fixed_T1_RAS_forward'],
                'reference_to_original_moving_ras_mm':registration['fixed_T1_RAS_to_moving_MRA_RAS_pull'],'registered_grid':'reference-native-grid'}
        metadata['workspace_registration']=shared
    compartments={'brain':host.brain_mask,'synthetic_target':target,'registered_partial_vessels':vessels}
    refs=tuple(host.source_refs)+(SourceRef('registered-partial-vessels',(base/'rigid-v1/registered-vascular-reference.nii.gz').as_uri(),proposal['registered_vessel_sha256'],license='CC BY4.0',native_frame='RAS+',provenance='estimated'),
          SourceRef('synthetic-target','urn:ressectionlab:'+case_id,array_digest(target),native_frame='RAS+',provenance='simulated'))
    candidate=replace(host,case_id=case_id,compartments=compartments,source_compartments=compartments,brain_mask=None,metadata=metadata,source_refs=refs)
    auxiliary_metadata={'data_case':derived_data,'is_synthetic':False,'selected_modality':'MRA','display_label':'Registered acquired MRA · reviewed rigid derivation' if shared else 'Registered acquired MRA · unreviewed derivation'}
    if shared:auxiliary_metadata['workspace_registration']=shared
    auxiliary=load_nifti_case(registered_path,case_id=case_id,metadata=auxiliary_metadata)
    original_aux=load_nifti_case(safe_path(root,registration['moving_MRA']['path']),case_id=case_id,
        metadata={'data_case':{**derived_data,'classification':'original_acquired_MRA_native_display','registration':'unregistered original native source'},'is_synthetic':False,'selected_modality':'MRA','display_label':'Original acquired MRA · native separate grid'})
    sources=[DisplaySource(auxiliary,'MRA','none'),DisplaySource(original_aux,'MRA','none')]
    originals=list(workspace['originals']);receipt=json.loads((base/'source/receipt.json').read_text())
    for path,role in [(safe_path(root,receipt['annotation_path']),'annotation'),(registered_path,'imaging'),(base/'rigid-v1/registered-vascular-reference.nii.gz','annotation'),(base/'rigid-v1/MRA-observed-coverage.nii.gz','annotation'),(registration_path,'metadata'),(proposal_path,'metadata')]:
        originals.append(OriginalFile.from_path(path,role=role))
    checked_root(root,required_bytes=1024**3);directory.mkdir(parents=True)
    destination=directory/(case_id+'.ressectionlab');state=view_state(candidate,sources);state['states']['primary']['cursor']=proposal['center_ras_mm']
    save_workspace(candidate,destination,sources=sources,imaging=state,episode=None,selection=None,artifacts={},originals=originals)
    reopened,_,restored=load_workspace(destination)
    if reopened.semantic_hash!=candidate.semantic_hash or not np.array_equal(reopened.mri,host.mri) or not np.array_equal(reopened.affine,host.affine):raise ValueError('Native roundtrip changed primary T1')
    if [(x.name,x.sha256,x.role) for x in restored['originals']]!=[(x.name,x.sha256,x.role) for x in originals]:raise ValueError('Original bytes changed')
    if sha256_file(source)!=proposal['source_native_sha256']:raise ValueError('Original native host changed')
    association,admission=source_registration(reopened,sources[0].case)
    if shared and not admission['overlayPermitted']:raise ValueError('Reviewed derived registration did not pass production admission')
    scene=from_case(reopened)
    region_ids={row['name']:row['id'] for row in scene.regions}
    if not np.array_equal(scene.labels==region_ids['synthetic_target'],target) or not np.array_equal(scene.labels==region_ids['registered_partial_vessels'],vessels):raise ValueError('Saved material precedence lost target or vascular ownership')
    report={'schema':'integrated-healthy-native-case-v1','case_id':case_id,'person_id':proposal['person_id'],'role':'TRAIN','accepted':False,
            'path':str(destination.relative_to(root)),'sha256':sha256_file(destination),'case_hash':reopened.semantic_hash,'primary_T1_array_unchanged':True,
            'native_primary_geometry_unchanged':True,'original_files_preserved':len(originals),'auxiliary_sources':len(sources),'registered_aux_is_synthetic':False,
            'derived_overlay_admission':admission,'original_MRA_overlay_permitted':source_registration(reopened,sources[1].case)[1]['overlayPermitted'],
            'independent_reviews':review_bindings,'target_proposal_sha256':sha256_file(proposal_path),'registration_sha256':sha256_file(registration_path),
            'production_save_reopen':'passed','simulator_regions':scene.regions,'target_and_vessel_material_ownership_preserved':True,'skull':'missing','mass_effect':False,
            'restrictions':restrictions,'generator_sha256':GENERATOR_SHA}
    (directory/'data_integrated_healthy.py.snapshot').write_bytes(GENERATOR_SOURCE)
    atomic_json(directory/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['sources','register','review','propose','build']);parser.add_argument('--subject',choices=SUBJECTS);parser.add_argument('--registration-decision');parser.add_argument('--target-decision');parser.add_argument('--version',default='geometry-v1');args=parser.parse_args()
    root=checked_root()
    if args.stage=='sources':sources(root)
    else:
        if args.subject is None:parser.error('--subject is required for register')
        build_native(root,args.subject,registration_decision=args.registration_decision,target_decision=args.target_decision,version=args.version) if args.stage=='build' else {'register':register,'review':registration_review,'propose':target_proposal}[args.stage](root,args.subject)
