#!/usr/bin/env python3
"""One declared, label-free multimodal rigid diagnostic on native ReMIND-001 MRI."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_on_qform(path):
    source = nib.load(path)
    data = source.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite MRI')
    q, code = source.get_qform(coded=True)
    if not code:
        raise ValueError('Explicit qform required for orthogonal ITK grid')
    corners=np.asarray(np.meshgrid(*[(0,n-1) for n in source.shape],indexing='ij')).reshape(3,-1).T
    error=float(np.linalg.norm((np.c_[corners,np.ones(8)]@(q-source.affine).T)[:,:3],axis=1).max())
    if error>.001:
        raise ValueError('Orthogonal qform differs materially from native sform')
    lower,upper=np.percentile(data,(1.,99.5))
    if not upper>lower:
        raise ValueError('Degenerate image intensity range')
    scaled=np.clip((data-lower)/(upper-lower),0,1).astype(np.float32)
    image=sitk.GetImageFromArray(scaled.transpose(2,1,0))
    ras_to_lps=np.diag([-1.,-1.,1.]); basis=ras_to_lps@q[:3,:3]
    spacing=np.linalg.norm(basis,axis=0)
    image.SetSpacing(tuple(spacing));image.SetOrigin(tuple(ras_to_lps@q[:3,3]));image.SetDirection(tuple((basis/spacing).ravel()))
    size=[int(np.ceil((n-1)*s/3.))+1 for n,s in zip(image.GetSize(),image.GetSpacing())]
    coarse=sitk.Resample(image,size,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkLinear,image.GetOrigin(),(3.,)*3,image.GetDirection(),0.,sitk.sitkFloat32)
    support=sitk.GetImageFromArray(np.ones(scaled.shape[::-1],np.uint8));support.CopyInformation(image)
    coverage=sitk.Resample(support,coarse,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return coarse,coverage,{'source_sha256':sha(path),'native_shape':list(source.shape),'qform_sform_max_corner_mm':error,
        'normalization_percentiles':[1.,99.5],'normalization_values':[float(lower),float(upper)],
        'normalized_using_this_image_only':True,'coarse_size_xyz':list(coarse.GetSize()),'coarse_spacing_mm':[3.,3.,3.]}


def matrix(transform):
    origin=np.asarray(transform.TransformPoint((0.,0.,0.)))
    basis=np.column_stack([np.asarray(transform.TransformPoint(tuple(np.eye(3)[i])))-origin for i in range(3)])
    lps=np.eye(4);lps[:3,:3]=basis;lps[:3,3]=origin
    flip=np.diag([-1.,-1.,1.,1.])
    return flip@lps@flip


def scalar_metrics(a,b):
    histogram=np.histogram2d(a,b,bins=32,range=((0,1),(0,1)))[0]
    p=histogram/histogram.sum();px=p.sum(1);py=p.sum(0)
    entropy=lambda values:float(-(values[values>0]*np.log(values[values>0])).sum())
    hx,hy,hxy=entropy(px),entropy(py),entropy(p)
    return {'mutual_information_nats':hx+hy-hxy,'normalized_mutual_information':(hx+hy)/hxy,'sample_count':int(len(a))}


def save_image(image,path):
    affine=np.eye(4);flip=np.diag([-1.,-1.,1.]);affine[:3,:3]=flip@np.asarray(image.GetDirection()).reshape(3,3)@np.diag(image.GetSpacing());affine[:3,3]=flip@image.GetOrigin()
    result=nib.Nifti1Image(sitk.GetArrayFromImage(image).transpose(2,1,0),affine)
    result.header.set_xyzt_units('mm');result.set_qform(affine,1);result.set_sform(affine,1);nib.save(result,path)


def run(output):
    started=time.perf_counter();output=Path(output)
    declaration_path=output/'declaration.json';d=json.loads(declaration_path.read_text())
    if (output/'result.json').exists() or (output/'child-attempt.json').exists():
        raise FileExistsError('Refusing a repeated registration attempt')
    if sha(__file__)!=d['script_sha256']:
        raise ValueError('Declared registration script changed')
    for item in d['images'].values():
        if sha(item['path'])!=item['sha256']:
            raise ValueError('Declared MRI source changed')
    (output/'child-attempt.json').write_text(json.dumps({'declaration_sha256':sha(declaration_path),'started_unix':time.time()},indent=2)+'\n')
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed,fixed_coverage,fr=image_on_qform(d['images']['fixed_T1']['path'])
    moving,moving_coverage,mr=image_on_qform(d['images']['moving_T2']['path'])
    shape=sitk.GetArrayFromImage(fixed).shape
    parity=np.indices(shape).sum(0)%2
    train=(parity==0)&(sitk.GetArrayFromImage(fixed_coverage)>0)
    metric_mask=sitk.GetImageFromArray(train.astype(np.uint8));metric_mask.CopyInformation(fixed)
    initial=sitk.Euler3DTransform()
    initial.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(tuple((n-1)/2 for n in fixed.GetSize())))
    registration=sitk.ImageRegistrationMethod();registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricFixedMask(metric_mask)
    registration.SetMetricSamplingStrategy(registration.RANDOM);registration.SetMetricSamplingPercentage(.2,2718)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=2.,minStep=.001,numberOfIterations=100,relaxationFactor=.5,gradientMagnitudeTolerance=1e-6)
    registration.SetOptimizerScalesFromPhysicalShift();registration.SetShrinkFactorsPerLevel([2,1]);registration.SetSmoothingSigmasPerLevel([1.,0.]);registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial,inPlace=False)
    trace=[]
    registration.AddCommand(sitk.sitkIterationEvent,lambda:trace.append({'level':int(registration.GetCurrentLevel()),'iteration':int(registration.GetOptimizerIteration()),'metric':float(registration.GetMetricValue())}))
    fitting=time.perf_counter();fitted=registration.Execute(fixed,moving);fit_seconds=time.perf_counter()-fitting
    forward=matrix(fitted);inverse=np.linalg.inv(forward)
    if (not np.isfinite(forward).all() or not np.allclose(forward[:3,:3].T@forward[:3,:3],np.eye(3),rtol=0,atol=1e-9)
            or not np.isclose(np.linalg.det(forward[:3,:3]),1.,atol=1e-9)):
        raise ValueError('Optimizer returned an invalid rigid transformation')
    before=sitk.Resample(moving,fixed,initial,sitk.sitkLinear,0.,sitk.sitkFloat32)
    after=sitk.Resample(moving,fixed,fitted,sitk.sitkLinear,0.,sitk.sitkFloat32)
    cover_before=sitk.Resample(moving_coverage,fixed,initial,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    cover_after=sitk.Resample(moving_coverage,fixed,fitted,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    common=(parity==1)&(sitk.GetArrayFromImage(fixed_coverage)>0)&(sitk.GetArrayFromImage(cover_before)>0)&(sitk.GetArrayFromImage(cover_after)>0)
    fixed_values=sitk.GetArrayFromImage(fixed);before_values=sitk.GetArrayFromImage(before);after_values=sitk.GetArrayFromImage(after)
    if common.sum()<1000:
        raise ValueError('Insufficient common coverage for diagnostic comparison')
    before_metrics=scalar_metrics(fixed_values[common],before_values[common]);after_metrics=scalar_metrics(fixed_values[common],after_values[common])
    for name,img in [('fixed_T1_3mm.nii.gz',fixed),('T2_identity_control_3mm.nii.gz',before),('T2_rigid_proposal_3mm.nii.gz',after),('T2_rigid_coverage_3mm.nii.gz',cover_after)]:
        save_image(img,output/name)
    sitk.WriteTransform(fitted,str(output/'T1_LPS_to_T2_LPS.tfm'))
    result={'schema_version':1,'status':'rigid_proposal_generated_requires_review','declaration_sha256':sha(declaration_path),
        'source_images':{'fixed_T1':fr,'moving_T2':mr},'transform_sampling_direction':'Fixed T1 physical points map into moving T2; inverse maps T2 anatomy into T1.',
        'T1_RAS_mm_to_T2_RAS_mm':forward.tolist(),'T2_RAS_mm_to_T1_RAS_mm':inverse.tolist(),
        'identity_assumed_as_correspondence':False,'identity_used_as_initialization_only':True,
        'optimizer_stop':registration.GetOptimizerStopConditionDescription(),'optimization_trace':trace,'fit_seconds':fit_seconds,
        'diagnostic_metrics_before':before_metrics,'diagnostic_metrics_after':after_metrics,
        'metric_comparison_scope':'Alternate fixed-grid parity voxels excluded by fixed optimization mask; common before/after moving FOV. Spatially correlated samples, not an independent clinical validation cohort.',
        'heldout_parity':1,'training_mask_parity':0,'final_moving_coverage_fraction':float((sitk.GetArrayFromImage(cover_after)>0).mean()),
        'landmark_error_mm':None,'expert_review':False,'source_segmentation_used_for_optimization':False,
        'registration_accepted':False,'cortical_access_permitted':False,'functional_or_vascular_evidence_created':False,
        'artifacts':{p.name:sha(p) for p in output.iterdir() if p.suffix in ('.gz','.tfm')},
        'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        'runtime':{'SimpleITK':sitk.Version_VersionString(),'numpy':np.__version__,'python':sys.version,'threads':1}}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'fit_seconds':fit_seconds,'wall_seconds':result['wall_seconds'],'before':before_metrics,'after':after_metrics,'peak_mib':result['peak_rss_bytes']/1024**2}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
