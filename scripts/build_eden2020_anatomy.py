#!/usr/bin/env python3
"""Prepare private, segmented EDEN2020 engineering scenarios on the data drive.

Originals and v1 projects are immutable. Model labels, registrations and generated
lesions are explicitly unreviewed estimates; this does not grant clinical or RL
admission. Heavy model inference and project construction run serially.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import gc
import hashlib
import itertools
import json
import os
import time
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.processing import resample_from_to,resample_to_output
from resectionlab.core import CaseData,SourceRef,array_digest,thaw_json
from resectionlab.data_root import atomic_json,checked_root,sha256_file
from resectionlab.data_synthesis import appearance
from resectionlab.head_segmentation import segment_head
from resectionlab.imaging import inspect_nifti,load_nifti_case
from resectionlab.organic_tumors import organic_geometry,signed_domain,shape_metrics
from resectionlab.prior_registration import image_from_ras,linear_transform_matrix
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import OriginalFile,save_workspace,load_workspace,view_state
from resectionlab.workspace_imaging import DisplaySource

LICENSE='CC-BY-NC-ND-4.0; private noncommercial adaptations; no derivative redistribution'
URL='https://zenodo.org/records/3994749'
FLIP=np.diag([-1.,-1.,1.,1.])
BRAIN_IDS={2,3,7,8,10,11,12,13,16,17,18,26,41,42,46,47,49,50,51,52,53,54,58,77,85,99,819,820,821,822,843,844,865,866,869,870}
CSF_IDS={4,5,14,15,24,43,44}
LIMITS=['Engineering rehearsal with synthetic disease, not clinically reviewed anatomy or an operative plan.',
 'All full-head labels are SuperSynth estimates from T1; skull is not CT-measured bone.',
 'MRA vesselFM output is an uncalibrated model estimate with partial scan coverage; absent labels do not prove absent vessels.',
 'Published tract reconstructions are registered baseline references, not functional mapping or complete tracts.',
 'Rigid registrations, tissue labels and synthetic locations require independent anatomical review.',
 'No mass effect, biological growth, displacement of vessels/tracts, or calibrated tissue mechanics is simulated.',
 'No dura label or direct cortical/subcortical functional mapping is available; no safe corridor or clinical readiness is established.',
 'Private noncommercial adaptations only; no derivative redistribution under the source license.']


def source_file(directory,subject,suffix):
    paths=list(directory.rglob(subject+'_'+suffix+'.nii.gz'))
    if len(paths)!=1:raise ValueError('Expected one exact source member: '+subject+' '+suffix)
    return paths[0]


def save_nii(path,data,affine):
    image=nib.Nifti1Image(data,affine);image.header.set_xyzt_units('mm');image.set_qform(affine,1);image.set_sform(affine,1)
    nib.save(image,path)


def label_names(path):
    result={}
    for line in path.read_text().splitlines():
        fields=line.split()
        if len(fields)>5 and fields[0].isdigit():result[int(fields[0])]=(fields[1],'#'+''.join(f'{int(v):02x}'for v in fields[2:5]))
    return result


def material(code):
    if code in CSF_IDS:return 'fluid'
    if code in {915,916}:return 'bone'
    if code in {902,914}:return 'vessel'
    if code in {901,99}:return 'unknown'
    return 'soft_tissue'


def source_header(image):
    q,qc=image.get_qform(coded=True);s,sc=image.get_sform(coded=True)
    corners=np.array(list(itertools.product(*[(0,n-1)for n in image.shape])))
    error=0. if not(qc and sc)else float(np.linalg.norm((corners@q[:3,:3].T+q[:3,3])-(corners@s[:3,:3].T+s[:3,3]),axis=1).max())
    return {'qform_code':int(qc),'sform_code':int(sc),'qform':None if q is None else q.tolist(),'sform':None if s is None else s.tolist(),
        'maximum_qform_sform_corner_difference_mm':error,'working_basis':'coded sform'if sc else 'coded qform',
        'units':image.header.get_xyzt_units()[0],'clinical_frame_adjudicated':False}


def sample(array,affine,grid,order=0):
    return np.asarray(resample_from_to(nib.Nifti1Image(array,affine),grid,order=order).dataobj)


def normalize(data):
    lo,hi=np.percentile(data[data>0],[1,99.5]);return np.clip((data-lo)/max(float(hi-lo),1e-6),0,1).astype(np.float32)


def tract_overlay_mask(path,flair_image):
    """Decode the publisher's high-value fiber overlay, never threshold the MRI.

    Readme describes *_on_* as fibers superimposed on MRI. Require a clean
    intensity gap and independently verify against the matching tract-only TRK
    in physical RAS mm before accepting the overlay's fiber voxels.
    """
    image=nib.load(path);values=np.asarray(image.dataobj)
    if image.shape!=flair_image.shape or not np.allclose(image.affine,flair_image.affine,rtol=0,atol=1e-4):raise ValueError('Tract does not match claimed source FLAIR grid')
    if not np.isfinite(values).all()or values.min()<0 or np.any((values>=10000)&(values<30000)):raise ValueError('Unresolved tract overlay intensity encoding')
    mask=values>=30000
    if not mask.any()or values[mask].min()<32000:raise ValueError('Expected separated publisher fiber overlay values')
    trk=path.with_name(path.name.replace('_on_','_to_').removesuffix('.nii.gz')+'.trk')
    fibers=nib.streamlines.load(trk)
    if not np.allclose(fibers.header['voxel_to_rasmm'],image.affine,rtol=0,atol=1e-4):raise ValueError('TRK and overlay coordinate frames differ')
    points=fibers.tractogram.to_world().streamlines.get_data();inverse=np.linalg.inv(image.affine)
    voxels=np.rint(points@inverse[:3,:3].T+inverse[:3,3]).astype(int);inside=np.all((voxels>=0)&(voxels<image.shape),axis=1)
    agreement=float(mask[tuple(voxels[inside].T)].mean())if inside.any()else 0.
    if not inside.all()or agreement<.99:raise ValueError('Published tract-only coordinates fail overlay cross-check')
    return mask,image,{'foreground_rule':'publisher high-value fiber overlay >=30000; all foreground >=32000 and no values in [10000,30000)',
        'background_max':float(values[~mask].max()),'foreground_min':float(values[mask].min()),'foreground_max':float(values[mask].max()),
        'tract_only_source':str(trk),'tract_only_sha256':sha256_file(trk),'tract_points':len(points),'tract_points_inside_overlay_fraction':agreement}


def registration(path,fixed,brain,out):
    """Retain original header ambiguities; explicitly estimate a working rigid map."""
    out.mkdir(parents=True,exist_ok=True);receipt=out/'registration.json';source_sha=sha256_file(path)
    if receipt.exists():
        row=json.loads(receipt.read_text())
        if row['source_sha256']!=source_sha:raise ValueError('Registration source changed')
        for name,digest in row['output_sha256'].items():
            if sha256_file(out/name)!=digest:raise ValueError('Registration output changed')
        return row
    moving_nifti=nib.load(path);header=source_header(moving_nifti)
    if header['units']!='mm' or not(header['sform_code']or header['qform_code']):raise ValueError('Unresolved source frame')
    # A conflicted q/s pair is never silently accepted by the general importer.
    # This derivative records both and estimates registration using the coded
    # sform as a working coordinate basis. Original bytes are preserved.
    moving_small=resample_to_output(moving_nifti,voxel_sizes=(1.2,)*3,order=1)
    fixed_array=np.asarray(fixed.dataobj,np.float32);moving=np.asarray(moving_small.dataobj,np.float32)
    fixed_sitk=image_from_ras(normalize(fixed_array),fixed.affine);moving_sitk=image_from_ras(normalize(moving),moving_small.affine)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    initial=sitk.Euler3DTransform();centroid=fixed.affine[:3,:3]@np.argwhere(brain).mean(0)+fixed.affine[:3,3]
    initial.SetCenter(tuple((FLIP@np.r_[centroid,1])[:3]))
    method=sitk.ImageRegistrationMethod();method.SetNumberOfThreads(2);method.SetMetricAsMattesMutualInformation(48)
    method.SetMetricSamplingStrategy(method.RANDOM);method.SetMetricSamplingPercentage(.08,42771)
    method.SetMetricFixedMask(image_from_ras(brain.astype(np.uint8),fixed.affine));method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(1.,.01,90,.5);method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4,2,1]);method.SetSmoothingSigmasPerLevel([2,1,0]);method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial,inPlace=False);started=time.monotonic()
    method.AddCommand(sitk.sitkIterationEvent,lambda:method.StopRegistration()if time.monotonic()-started>120 else None)
    transform=method.Execute(fixed_sitk,moving_sitk)
    pull=FLIP@linear_transform_matrix(transform)@FLIP;forward=np.linalg.inv(pull)
    # Source coordinate start and bounded rigid estimation; no anatomical
    # acceptance is inferred from a mutual-information improvement.
    if abs(np.linalg.det(pull[:3,:3])-1)>1e-5:raise ValueError('Non-rigid transform')
    native=np.asarray(moving_nifti.dataobj,np.float32)
    grid=(fixed.shape,fixed.affine)
    registered=sample(native,forward@moving_nifti.affine,grid,1).astype(np.float32)
    coverage=sample(np.ones(moving_nifti.shape,np.uint8),forward@moving_nifti.affine,grid,0)>0
    save_nii(out/'registered.nii.gz',registered,fixed.affine);save_nii(out/'coverage.nii.gz',coverage.astype(np.uint8),fixed.affine)
    row={'source':str(path),'source_sha256':source_sha,'source_header':header,'fixed_grid_shape':list(fixed.shape),'fixed_affine':fixed.affine.tolist(),
        'moving_RAS_to_fixed_RAS_forward':forward.tolist(),'fixed_RAS_to_moving_RAS_pull':pull.tolist(),
        'method':'Euler3D, MattesMI48, random8%seed42771, 2threads, multires4/2/1, 90iterations/level, scanner-frame identity start',
        'optimizer_stop':method.GetOptimizerStopConditionDescription(),'metric':float(method.GetMetricValue()),'seconds':time.monotonic()-started,
        'brain_scan_coverage_fraction':float(coverage[brain].mean()),'clinical_alignment_validated':False,
        'source_header_conflict_retained':header['maximum_qform_sform_corner_difference_mm']>.01,
        'output_sha256':{p.name:sha256_file(p)for p in out.glob('*.nii.gz')}}
    atomic_json(receipt,row);return row


def vessels(root,source,subject,out):
    from resectionlab.poc_models import run
    import psutil
    out.mkdir(parents=True,exist_ok=True);input=out/'MRA-0p6mm.nii.gz';receipt=out/'input.json'
    inspect_nifti(source)
    if not input.exists():
        image=resample_to_output(nib.load(source),voxel_sizes=(.6,)*3,order=1);image.set_data_dtype(np.float32);image.header.set_xyzt_units('mm');nib.save(image,input)
        atomic_json(receipt,{'source':str(source),'source_sha256':sha256_file(source),'output_sha256':sha256_file(input),
            'operation':'0.6mm RAS linear resample for bounded vesselFM inference; no registration','clinical_validation':False})
    row=json.loads(receipt.read_text())
    if row['source_sha256']!=sha256_file(source)or row['output_sha256']!=sha256_file(input):raise ValueError('Vascular input changed')
    result=run(input,'EDEN2020-'+subject+'-segmented-v2','vesselfm-base','MRA',timeout=900,output_root=root/'cache/inference',blend='gaussian',
        cancelled=lambda:psutil.virtual_memory().available<1.25*1024**3)
    atomic_json(out/'inference.json',result)
    return Path(next(r['path']for r in result['outputs']if r['name'].endswith('_mask.nii.gz')))


def placements(domain,affine):
    """Eight physical octants, non-midline centers and reproducible random jitter."""
    coordinates=np.argwhere(domain);world=coordinates@affine[:3,:3].T+affine[:3,3]
    low,high=np.percentile(world,[1,99],axis=0);mid=(low+high)/2
    distance=signed_domain(domain,affine);admitted=domain&(distance>=3.)&(distance<=18.)
    candidates=np.argwhere(admitted);points=candidates@affine[:3,:3].T+affine[:3,3]
    result=[]
    for posterior,inferior,right in itertools.product([False,True],repeat=3):
        side=points[:,0]>mid[0]+18 if right else points[:,0]<mid[0]-18
        y=points[:,1]<mid[1]-3 if posterior else points[:,1]>mid[1]+3
        z=points[:,2]<mid[2]-3 if inferior else points[:,2]>mid[2]+3
        group=candidates[side&y&z]
        if len(group)<100:raise ValueError('Insufficient noncentral tissue in requested placement stratum')
        result.append((group,('right'if right else'left')+'-'+('posterior'if posterior else'anterior')+'-'+('inferior'if inferior else'superior')))
    return result,distance


def qc_image(case,healthy,path,recipe=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes
    tumor=(case.compartments.get('synthetic_core',np.zeros(case.mri.shape,bool))|case.compartments.get('synthetic_rim',np.zeros(case.mri.shape,bool)))
    center=np.asarray(recipe['center_voxel'])if recipe else np.rint(np.argwhere(case.brain_mask).mean(0)).astype(int)
    lo,hi=np.percentile(healthy[case.brain_mask],[2,98]);fig=plt.figure(figsize=(14,8))
    for axis in range(3):
        for row,image in enumerate([healthy,case.mri]):
            ax=fig.add_subplot(2,4,row*4+axis+1);ax.imshow(np.take(image,int(center[axis]),axis=axis).T,origin='lower',cmap='gray',vmin=lo,vmax=hi)
            for name,color in [('synthetic_core','#ff5a9e'),('synthetic_rim','#ffd16c'),('synthetic_edema','#55c9ff'),('head_915','#f3e2ae'),('mra_vessels','#fc4545')]:
                if name not in case.compartments:continue
                plane=np.take(case.compartments[name],int(center[axis]),axis=axis).T
                if plane.any():ax.contour(plane,[.5],colors=[color],linewidths=.5)
            ax.set_title(('Healthy 'if row==0 else'Scenario ')+['sagittal','coronal','axial'][axis]);ax.axis('off')
    if tumor.any():
        occupied=np.argwhere(tumor);low=occupied.min(0);high=occupied.max(0)+1;crop=np.pad(tumor[tuple(slice(a,b)for a,b in zip(low,high))],2)
        from scipy.ndimage import gaussian_filter
        vertices,faces,_,_=marching_cubes(gaussian_filter(crop.astype(np.float32),.6),.5,spacing=case.spacing_mm)
        ax=fig.add_subplot(1,4,4,projection='3d');ax.plot_trisurf(vertices[:,0],vertices[:,1],faces,vertices[:,2],color='#e45c8a',linewidth=0,antialiased=True)
        ax.set_box_aspect(np.maximum(vertices.ptp(0)if hasattr(vertices,'ptp')else np.ptp(vertices,axis=0),1));ax.set_title('Generated lobulated target');ax.set_axis_off()
    fig.suptitle(case.case_id+'\nModel-estimated anatomy · synthetic target · no mass effect · unreviewed registration',fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)


def package(root,out,case,sources,originals,healthy,recipe):
    receipt=out/'projects'/(case.case_id+'.json');destination=root/'cases'/('synthetic'if recipe else'controls')/(case.case_id+'.ressectionlab')
    if receipt.exists():
        row=json.loads(receipt.read_text())
        if sha256_file(destination)!=row['sha256']or row['case_hash']!=case.semantic_hash:raise ValueError('Completed project changed')
        return row
    if destination.exists():raise ValueError('Preserve an unreceipted earlier project')
    state=view_state(case,sources)
    for s in state['states'].values():
        s['visibleLayers']={name:False for name in s['visibleLayers']}
    primary=state['states']['primary'];primary['visibleLayers']['brain']=True
    for name in ['synthetic_core','synthetic_rim','synthetic_edema','mra_vessels']:
        if name in primary['visibleLayers']:primary['visibleLayers'][name]=True
    if recipe:primary['cursor']=recipe['center_ras_mm']
    primary['viewer']={'smoothSurfaces':True,'layerOpacity':{'brain':.12,'synthetic_core':1.,'synthetic_rim':.85,'synthetic_edema':.15,'mra_vessels':.8},'sliceOverlay':'contour'}
    primary['viewer']['layerOpacity']={k:v for k,v in primary['viewer']['layerOpacity'].items()if k in primary['visibleLayers']}
    started=time.monotonic();session=save_workspace(case,destination,sources=sources,imaging=state,episode=None,selection=None,artifacts={},originals=originals)
    reopened,_,saved=load_workspace(destination)
    if reopened.semantic_hash!=case.semantic_hash or saved['sessionHash']!=session:raise ValueError('Production native roundtrip failed')
    if [r.sha256 for r in saved['originals']]!=[r.sha256 for r in originals]:raise ValueError('Original archive changed')
    if len(reopened.compartments)!=len(case.compartments):raise ValueError('Anatomical labels lost')
    scene=from_case(reopened)
    if recipe:
        target=reopened.compartments['synthetic_core']|reopened.compartments['synthetic_rim']
        union=target|reopened.compartments['synthetic_edema']
        if np.any(union&~reopened.brain_mask)or not np.array_equal(reopened.mri[~union],healthy[~union]):raise ValueError('Synthetic background/containment failed')
        metrics=shape_metrics(target,reopened.affine)
        if metrics['is_cuboid']or metrics['surface_solidity']>.98 or metrics['components']!=1:raise ValueError('Reopened organic shape failed')
        ids={r['id']for r in scene.regions if r['target']}
        if not np.array_equal(np.isin(scene.labels,list(ids)),target):raise ValueError('Tumor material ownership changed')
    sheet=out/'qc'/(case.case_id+'.png');qc_image(case,healthy,sheet,recipe)
    row={'case_id':case.case_id,'person_id':case.metadata['patient_group'],'path':str(destination.relative_to(root)),'sha256':sha256_file(destination),
        'bytes':destination.stat().st_size,'case_hash':case.semantic_hash,'session_hash':session,'synthetic':recipe is not None,'recipe':recipe,
        'role':'DEVELOPMENT','clinical_validation':False,'training_admitted':False,'production_save_reopen':'passed','source_originals_preserved':True,
        'scene_initialization':'passed','scene_region_count':len(scene.regions),'contact_sheet':str(sheet.relative_to(root)),
        'shape':list(case.mri.shape),'spacing_mm':list(case.spacing_mm),'seconds':time.monotonic()-started,
        'compartments':{name:{'voxels':int(mask.sum()),'volume_mm3':float(mask.sum()*case.voxel_volume_mm3)}for name,mask in case.compartments.items()}}
    atomic_json(receipt,row);print(json.dumps({'project':case.case_id,'verified':True,'layers':len(case.compartments),'seconds':row['seconds']}),flush=True)
    del reopened,saved,scene;gc.collect();return row


def build_subject(root,subject,head_attempt='head-model-v2b'):
    source=root/'sources/eden2020-healthy-3994749';directory=source/'extracted'/subject;out=root/'prepared/eden2020-anatomy-v2'
    for name in ['projects','qc','labels','subjects','logs']:(out/name).mkdir(parents=True,exist_ok=True)
    working=out/'subjects'/subject;working.mkdir(exist_ok=True);done=working/'complete.json'
    if done.exists():
        row=json.loads(done.read_text())
        for r in row['projects']:
            if sha256_file(root/r['path'])!=r['sha256']:raise ValueError('Completed subject bytes changed')
        return row['projects']
    t1=source_file(directory,subject,'T1_3D_PROSET_Sag');inspect_nifti(t1)
    head=working/head_attempt;segment_head(t1,head,root/'cache/models/supersynth-source')
    head_path=head/'labels_1mm.nii.gz';native=nib.load(t1);labels=nib.load(head_path)
    tracts=sorted(p for p in directory.rglob('*_on_3DFLAIR.nii.gz')if p.name.startswith(subject+'_'))
    # Full FOV and all model classes, with adaptive declared spacing bounded by
    # the existing 1GiB expanded workspace cap. No array alias or limit bypass.
    spacing=1.2
    while True:
        fixed=resample_to_output(native,voxel_sizes=(spacing,)*3,order=1)
        if np.prod(fixed.shape)*(5+2*(55+len(tracts)+4))<880*1024**2:break
        spacing=round(spacing+.05,2)
    fixed=nib.Nifti1Image(np.asarray(fixed.dataobj,np.float32),fixed.affine);fixed.header.set_xyzt_units('mm')
    grid=(fixed.shape,fixed.affine);head_labels=sample(np.asarray(labels.dataobj,np.uint16),labels.affine,grid,0).astype(np.uint16)
    brain=np.isin(head_labels,list(BRAIN_IDS));cerebral=np.isin(head_labels,[2,3,41,42])
    if brain.sum()<200000 or cerebral.sum()<150000:raise ValueError('Insufficient head-model cerebral coverage')
    names=label_names(root/'cache/models/supersynth-source/FreeSurferColorLUT.txt');compartments={};layers={};materials={'brain':'soft_tissue'}
    present=list(map(int,np.unique(np.asarray(labels.dataobj))))
    lost=[]
    for code in present:
        if code==0:continue
        name='head_'+str(code);mask=head_labels==code
        if not mask.any():lost.append(code);continue
        compartments[name]=mask;mat=material(code);materials[name]=mat
        layers[name]={'kind':'skull'if mat=='bone'else'vessels'if mat=='vessel'else'brain'if code in BRAIN_IDS else'context',
            'label':names[code][0]+' · T1 MODEL ESTIMATE','color':names[code][1],'provenance':'estimated','review_status':'unreviewed',
            'protected':code in {16,902,914,930},'model_label_id':code,'material_basis':'explicit engineering assignment; uncalibrated'}
    mra=source_file(directory,subject,'s3DI_MC_HR');flair=source_file(directory,subject,'3D_FLAIR_Tra');swi=source_file(directory,subject,'SWIp_axial')
    os.environ['RESSECTIONLAB_DATA_ROOT']=str(root);os.environ['RESSECTIONLAB_MODEL_ROOT']=str(root/'cache/models')
    vessel_path=vessels(root,mra,subject,working/'vesselfm-0p6mm')
    registrations={mod:registration(path,fixed,brain,working/('rigid-'+mod))for mod,path in [('MRA',mra),('FLAIR',flair),('SWI',swi)]}
    v=nib.load(vessel_path);forward=np.array(registrations['MRA']['moving_RAS_to_fixed_RAS_forward'])
    vascular=sample((np.asarray(v.dataobj)>0).astype(np.uint8),forward@v.affine,grid,0)>0
    if not vascular.any():raise ValueError('Empty registered vascular estimate')
    compartments['mra_vessels']=vascular;materials['mra_vessels']='vessel'
    layers['mra_vessels']={'kind':'vessels','label':'MRA vessels · vesselFM ESTIMATE','color':'#ef5454','provenance':'estimated','protected':True,'review_status':'unreviewed'}
    tract_rows=[];flair_image=nib.load(flair);forward=np.array(registrations['FLAIR']['moving_RAS_to_fixed_RAS_forward'])
    for p in tracts:
        native_tract,image,interpretation=tract_overlay_mask(p,flair_image)
        name='tract_'+p.name.split(subject+'_',1)[1].split('_on_')[0]
        mask=sample(native_tract.astype(np.uint8),forward@image.affine,grid,0)>0
        if not mask.any():raise ValueError('Published tract lost on registered rehearsal grid')
        compartments[name]=mask;materials[name]='functional_region'
        layers[name]={'kind':'tract','label':name.removeprefix('tract_')+' · PUBLISHED BASELINE TRACT','color':'#70c6b1','provenance':'estimated',
            'protected':True,'review_status':'unreviewed_registration','functional_validation':False}
        tract_rows.append({'layer':name,'source':str(p),'source_sha256':sha256_file(p),'native_header':source_header(image),
            'units_inherited_from_exact_source_FLAIR_grid':image.header.get_xyzt_units()[0]=='unknown',**interpretation,
            'registered_voxels':int(mask.sum()),'clinical_functional_mapping':False})
    layers['brain']={'kind':'brain','label':'Brain overview · MODEL ESTIMATE','color':'#9eb8cd','provenance':'estimated'}
    metadata={'patient_group':'EDEN2020:'+subject,'patient_id':subject,'split':{'role':'DEVELOPMENT'},'training_admitted':False,
        'is_synthetic':False,'selected_modality':'T1 (ACQUIRED HEALTHY BASELINE, RESAMPLED REHEARSAL GRID)',
        'display_label':'Segmented healthy baseline · estimated anatomy','anatomy_layers':layers,
        'surgical_adapter':{'ownershipPrecedence':['brain',*compartments],'materialByRegion':materials},
        'data_case':{'source_url':URL,'source_subject':subject,'source_license':LICENSE,'role':'DEVELOPMENT','clinical_validation':False,
            'classification':'real_healthy_control_with_estimated_anatomy','limitations':LIMITS,'original_T1_sha256':sha256_file(t1),
            'head_model_receipt':str((head/'inference.json').relative_to(root)),'head_model_receipt_sha256':sha256_file(head/'inference.json'),
            'head_model_labels_present_1mm':present,'model_labels_absent_on_rehearsal_grid':lost,'rehearsal_grid':{'spacing_mm':spacing,'shape':list(fixed.shape),
                'image_interpolation':'linear','label_interpolation':'nearest-neighbor','full_source_FOV':True,'original_native_T1_retained':True},
            'registrations':registrations,'published_tracts':tract_rows,'missing_anatomy':['dura','measured_CT_bone','functional_mapping','reviewed_complete_vessels'],
            'components':{'brain_and_head':'SuperSynth estimates','vessels':'T1 SuperSynth and MRA vesselFM estimates','tracts':'published reconstructions with estimated rigid registration','tumor':'absent_healthy_baseline'}}}
    refs=(SourceRef('structural',t1.as_uri(),sha256_file(t1),LICENSE,'RAS+','observed'),
        SourceRef('head_model',head_path.as_uri(),sha256_file(head_path),'FreeSurfer model; research estimates','RAS+','estimated'),
        SourceRef('mra_vessels',vessel_path.as_uri(),sha256_file(vessel_path),'OpenRAIL++M, research use','RAS+','estimated'))
    healthy=np.asarray(fixed.dataobj,np.float32);base_id='EDEN2020-'+subject+'-BASELINE-SEGMENTED-v2'
    host=CaseData(base_id,healthy,compartments,fixed.affine,refs,brain_mask=brain,metadata=metadata,
        unknowns=('functional_mapping_missing','segmentation_unreviewed','registration_unreviewed','MRI_bone_estimate_not_CT','vascular_completeness_unknown','mass_effect_not_simulated'))
    source_paths=[t1,head_path,mra,flair,swi,*tracts,*[Path(r['tract_only_source'])for r in tract_rows],head/'inference.json',working/'vesselfm-0p6mm/inference.json',source/'provenance/zenodo-record.json']
    for mod in registrations:source_paths.append(working/('rigid-'+mod)/'registration.json')
    originals=[OriginalFile.from_path(p,role='imaging'if p==t1 or p in [mra,flair,swi]else'model-result'if p.suffix=='.gz'else'metadata')for p in source_paths]
    acquired=load_nifti_case(t1,case_id=subject+'-native-healthy',license=LICENSE,source_url=URL,
        metadata={'display_label':'Original acquired healthy T1 · unchanged native resolution','selected_modality':'T1','is_synthetic':False})
    sources=[DisplaySource(acquired,'T1','none')]
    for mod in registrations:
        p=working/('rigid-'+mod)/'registered.nii.gz'
        display=load_nifti_case(p,case_id=subject+'-registered-'+mod,license=LICENSE,source_url=URL,
            metadata={'display_label':mod+' · acquired baseline registered to rehearsal grid; UNREVIEWED','selected_modality':mod,'is_synthetic':False,
                'registration_receipt':registrations[mod],'partial_source_coverage':True,'observation_role':'healthy_baseline_reference'})
        sources.append(DisplaySource(display,mod,'none'))
    records=[package(root,out,host,sources,originals,healthy,None)]
    groups,distance=placements(cerebral,fixed.affine);previous=[]
    for variant,(candidates,stratum)in enumerate(groups):
        seed=int.from_bytes(hashlib.sha256(f'EDEN2020-organic-v2:{subject}:{variant}'.encode()).digest()[:4],'big');rng=np.random.default_rng(seed)
        for attempt in range(100):
            center=candidates[rng.integers(len(candidates))];point=fixed.affine[:3,:3]@center+fixed.affine[:3,3]
            if any(np.linalg.norm(point-p)<18 for p in previous):continue
            radii=(rng.uniform(8,13),rng.uniform(10,16),rng.uniform(11,18))
            try:masks,recipe=organic_geometry(cerebral,fixed.affine,center_voxel=center,seed=seed+attempt,radii_mm=radii,signed=distance)
            except ValueError:continue
            if recipe['metrics']['volume_mm3']<1500:continue
            break
        else:raise ValueError('Could not prepare distinct organic target for '+stratum)
        previous.append(point);recipe.update(placement_stratum=stratum,placement_domain='estimated cerebral GM/WM only, outside ventricles and extracerebral tissues',
            clinical_location_reviewed=False,base_seed=seed,rejection_draw=attempt,no_mass_effect=True,min_center_separation_mm=18.)
        image,signal=appearance(healthy,brain,masks,seed=seed);recipe['appearance']=signal
        target=masks['synthetic_core']|masks['synthetic_rim'];active={k:(m&~target if k.startswith('head_')and int(k[5:])in BRAIN_IDS else m)for k,m in compartments.items()}
        active.update(masks);source_masks={**compartments,**masks};details=thaw_json(host.metadata);details.update(is_synthetic=True,selected_modality='T1 (SYNTHETIC ORGANIC TUMOR)',display_label='Segmented synthetic tumor · '+stratum)
        details['data_case'].update(classification='synthetic_scenario_on_acquired_host',host_case_id=host.case_id,recipe=recipe,new_real_patient_count=0)
        details['data_case']['components']['tumor']='generated_lobulated_core_rim_edema'
        for name,color in zip(masks,['#f568a1','#ffcf6a','#66cbf0']):
            details['anatomy_layers'][name]={'kind':'edema'if name.endswith('edema')else'tumor','label':name.replace('_',' ').title()+' · GENERATED','color':color,'provenance':'simulated'}
        details['surgical_adapter']['ownershipPrecedence']=['brain',*compartments,*masks]
        details['surgical_adapter']['materialByRegion'].update({k:'soft_tissue'if k.endswith('edema')else'tumor'for k in masks})
        case_id='EDEN2020-'+subject+f'-SYNTHETIC-{variant+1:02d}-SEGMENTED-v2'
        synthetic=CaseData(case_id,image,active,fixed.affine,(*refs,SourceRef('synthetic_scenario','synthetic://'+case_id,array_digest(image),LICENSE,'RAS+','simulated')),
            brain_mask=brain,source_compartments=source_masks,metadata=details,unknowns=host.unknowns)
        label_path=out/'labels'/(case_id+'.nii.gz');label_array=np.zeros(fixed.shape,np.uint8)
        for i,m in enumerate(masks.values(),1):label_array[m]=i
        if not label_path.exists():save_nii(label_path,label_array,fixed.affine)
        records.append(package(root,out,synthetic,sources,[*originals,OriginalFile.from_path(label_path,role='annotation')],healthy,recipe))
        del synthetic,active,source_masks,image,masks;gc.collect()
    row={'subject':subject,'complete':True,'projects':records,'head_segmentation':str(head.relative_to(root)),'tract_count':len(tracts),'lost_labels_on_rehearsal_grid':lost}
    atomic_json(done,row);return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path);parser.add_argument('--subject');parser.add_argument('--head-attempt',default='head-model-v2b');args=parser.parse_args()
    root=checked_root(args.root,required_bytes=80*1024**3);base=root/'prepared/eden2020-anatomy-v2'
    subjects=json.loads((root/'sources/eden2020-healthy-3994749/provenance/development-plan.json').read_text())['subjects']
    if args.subject and args.subject not in subjects:raise ValueError('Unknown subject')
    for subject in ([args.subject]if args.subject else subjects):
        print('Preparing '+subject,flush=True);build_subject(root,subject,args.head_attempt);gc.collect()
        records=[r for p in sorted((base/'subjects').glob('*/complete.json'))for r in json.loads(p.read_text())['projects']]
        atomic_json(base/'collection.json',{'complete':len(records)==135,'volunteers':len(records)//9,'baseline_projects':sum(not r['synthetic']for r in records),
            'synthetic_projects':sum(r['synthetic']for r in records),'role':'DEVELOPMENT','clinical_validation':False,'training_admitted':False,'projects':records,'limitations':LIMITS})

if __name__=='__main__':main()
