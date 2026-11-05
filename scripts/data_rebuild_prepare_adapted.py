#!/usr/bin/env python3
"""Typed source-sidecar and explicit display-LOD adapter for the frozen v5 recipe.

The scientific generator/export implementation remains the reviewed v5 file.
This adapter records its own lineage and never changes native primary geometry.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np

import data_rebuild_prepare as shared
from resectionlab.core import thaw_json
from resectionlab.data_root import atomic_json,checked_root,safe_path,sha256_file
from resectionlab.workspace_bundle import OriginalFile
from resectionlab.workspace_imaging import MAX_IMAGE_BYTES


def typed_files(acquisition):
    images=[row for row in acquisition['acquired_files'] if row['path'].endswith(('.nii','.nii.gz')) and not Path(row['path']).name.startswith('.')]
    sidecars=[row for row in acquisition['acquired_files'] if row['path'].endswith('.json') and not Path(row['path']).name.startswith('.')]
    if len({row['modality'] for row in images})!=len(images):raise ValueError('Ambiguous duplicate source image modalities')
    return images,sidecars


def run(subject,*,cpu=False,mass_effect=False):
    root=checked_root();adapter_hash=sha256_file(Path(__file__))
    snapshot=root/'prepared/recipes'/(adapter_hash+'-'+Path(__file__).name)
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    original_host,original_display,original_save,original_infer=shared.host,shared.DisplaySource,shared.save_workspace,shared.infer
    adapter={'schema':'typed-source-and-display-adapter-v1','sha256':adapter_hash,'source':str(snapshot.relative_to(root)),
        'sidecar_policy':'JSON sidecars remain metadata; only NIfTI observations select imaging modalities',
        'display_lod':'images over128MiB use explicit1mm linear display derivative; native primary/source originals preserved',
        'brain_extraction_device':'cpu' if cpu else 'mps'}
    def display(case,modality,annotation):
        if case.mri.nbytes<=MAX_IMAGE_BYTES:return original_display(case,modality,annotation)
        if case.compartments or case.brain_mask is not None:raise ValueError('Only unlabelled baseline display observations may use this LOD')
        image=resample_to_output(nib.Nifti1Image(case.mri,case.affine),voxel_sizes=1.,order=1)
        array=np.asarray(image.dataobj,np.float32)
        record={'operation':'linear physical resampling for display only','spacing_mm':[1.,1.,1.],
            'native_shape':list(case.mri.shape),'native_affine_ras_mm':case.affine.tolist(),
            'display_shape':list(array.shape),'display_affine_ras_mm':image.affine.tolist(),
            'display_voxel_to_native_voxel':(np.linalg.inv(case.affine)@image.affine).tolist(),
            'native_original_bytes_preserved':True,'anatomical_registration_performed':False,'planning_input':False}
        metadata={**thaw_json(case.metadata),'display_resampling':record,
            'display_label':str(case.metadata.get('display_label',modality))+' ·1mm display derivative; native preserved',
            'observation_role':'baseline_reference_not_tumor_present'}
        derived=replace(case,mri=array,affine=image.affine,metadata=metadata)
        return original_display(derived,modality,annotation)
    def host(root,acquisition):
        images,sidecars=typed_files(acquisition)
        case,sources,originals=original_host(root,{**acquisition,'acquired_files':images})
        for row in sidecars:
            path=safe_path(root,row['path'])
            if sha256_file(path)!=row['sha256']:raise ValueError('Source metadata sidecar changed')
            originals.append(OriginalFile.from_path(path,role='metadata',expected_sha256=row['sha256']))
        metadata=thaw_json(case.metadata);metadata['data_case'].update(ingestion_adapter=adapter,
            source_files=[{**row,'file_role':'imaging' if row in images else 'metadata'} for row in [*images,*sidecars]])
        metadata['data_case']['limitations'].append('Baseline display derivatives over128MiB are explicitly resampled to1mm; native primary and originals remain available')
        return replace(case,metadata=metadata),sources,originals
    def save(primary,path,**kwargs):
        imaging=kwargs['imaging']
        volumes=[('primary',primary),*[(source.identity(primary),source.case) for source in kwargs['sources']]]
        for key,case in volumes:
            if case.mri.nbytes>MAX_IMAGE_BYTES or case.metadata.get('display_resampling'):
                state=imaging['states'][key];state.setdefault('viewer',{})['volumeRendering']={'enabled':False}
        return original_save(primary,path,**kwargs)
    shared.host=host;shared.DisplaySource=display;shared.save_workspace=save
    if cpu:
        shared.infer=lambda *args,**kwargs:original_infer(*args,**kwargs,device='cpu')
    try:
        acquisition=json.loads((root/'catalog/source-acquisition.json').read_text())
        row=next(row for row in acquisition['candidates'] if row['subject']==subject)
        image=next(row for row in typed_files(row)[0] if row['modality'] in {'T1','T1w'})
        voxel_count=int(np.prod(nib.load(safe_path(root,image['path'])).shape))
        if mass_effect and voxel_count*200>4*1024**3:
            admission=root/'qc/mass-effect-resource-admission';admission.mkdir(exist_ok=True)
            atomic_json(admission/(subject+'.json'),{'schema':'bounded-mass-effect-resource-admission-v1','subject':subject,
                'requested':True,'admitted':False,'estimated_additional_array_bytes':voxel_count*200,'budget_bytes':4*1024**3,
                'basis':'native grid field,coordinates,inverse andJacobian arrays plus images; conservative200bytes/voxel estimate',
                'action':'retain native geometry/appearance cases; withhold large-grid deformation instead of silently resampling',
                'native_voxel_count':voxel_count,'clinical_validation':False})
            mass_effect=False
        shared.run(root,subject=subject,variants=4,mass_effect=mass_effect)
    finally:
        shared.host=original_host;shared.DisplaySource=original_display;shared.save_workspace=original_save;shared.infer=original_infer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--subject',required=True)
    parser.add_argument('--cpu',action='store_true');parser.add_argument('--mass-effect',action='store_true');args=parser.parse_args()
    run(args.subject,cpu=args.cpu,mass_effect=args.mass_effect)
