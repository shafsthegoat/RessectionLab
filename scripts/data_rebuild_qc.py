#!/usr/bin/env python3
"""Incremental raw-array QC of v5 healthy exports, separate from generation.

This automated engineering check never grants independent visual acceptance.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from resectionlab.data_root import checked_root,atomic_json,safe_path,sha256_file


def primary(archive):
    manifest=json.loads(archive.read('manifest.json'));payload=archive.read('arrays.npz')
    if hashlib.sha256(payload).hexdigest()!=manifest['array_sha256']:raise ValueError('Primary array payload checksum differs')
    with np.load(BytesIO(payload),allow_pickle=False) as arrays:
        data={name:arrays[name] for name in arrays.files}
    return manifest,data


def inspect(root,receipt):
    path=safe_path(root,receipt['path'])
    if sha256_file(path)!=receipt['sha256']:raise ValueError('Native package differs from preparation receipt')
    with ZipFile(path) as archive:
        manifest,arrays=primary(archive)
        workspace=json.loads(archive.read('workspace.json'))
        contract=manifest['metadata']['data_case']
        image=arrays['mri'];affine=arrays['affine'];brain=arrays['brain_mask']
        masks={name:arrays[key].astype(bool) for name,key in manifest['array_index']['compartments'].items()}
        if contract['role']!='TRAIN' or contract['person_id']!=receipt['person_id']:raise ValueError('Person/role mismatch')
        if (image.ndim!=3 or not np.isfinite(image).all() or affine.shape!=(4,4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3],[0,0,0,1],rtol=0,atol=1e-8) or abs(np.linalg.det(affine[:3,:3]))<1e-9):raise ValueError('Invalid physical image grid')
        if brain.shape!=image.shape or not np.isin(brain,[0,1]).all() or not brain.any():raise ValueError('Invalid brain estimate')
        result={'case_id':receipt['case_id'],'path':receipt['path'],'sha256':receipt['sha256'],'family_id':contract['person_id'],
            'role':contract['role'],'shape':list(image.shape),'spacing_mm':np.linalg.norm(affine[:3,:3],axis=0).tolist(),
            'voxel_volume_mm3':float(abs(np.linalg.det(affine[:3,:3]))),'source_originals':len(workspace.get('originalFiles',[])),
            'mask_checks':{},'classification':contract['classification'],'independent_visual_review_required':True}
        if not masks:
            result['automated_decision']='pass_for_control_source_and_estimate_only';return result
        if set(masks)!={'synthetic_core','synthetic_rim','synthetic_edema'}:raise ValueError('Unknown synthetic label dictionary')
        combined=np.zeros(image.shape,np.uint8)
        for name,mask in masks.items():
            if mask.shape!=image.shape or not mask.any() or np.any(mask & ~brain.astype(bool)):raise ValueError('Empty/escaped compartment')
            if any(np.take(mask,[0,n-1],axis=a).any() for a,n in enumerate(mask.shape)):raise ValueError('Compartment touches FOV edge')
            combined+=mask
            result['mask_checks'][name]={'voxels':int(mask.sum()),'outside_brain':0,
                'array_sha256':hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()}
        if combined.max()>1:raise ValueError('Synthetic compartment overlap without a declared priority')
        recipe=contract['recipe']
        if recipe.get('donors'):raise ValueError('Unexpected donor provenance outside frozen donor-free recipe')
        support_record=recipe['placement_support'];support_path=safe_path(root,support_record['mask_path'])
        if sha256_file(support_path)!=support_record['mask_sha256']:raise ValueError('Placement support changed')
        support=np.asarray(nib.load(support_path).dataobj)>0
        deformation=contract.get('deformation')
        if deformation:
            field_path=safe_path(root,deformation['field_path'])
            if sha256_file(field_path)!=deformation['field_sha256']:raise ValueError('Deformation field changed')
            with np.load(field_path,allow_pickle=False) as stored:
                field=stored['displacement_ras_mm'];field_affine=stored['affine_ras_mm']
            if field.shape!=(3,*image.shape) or not np.array_equal(field_affine,affine) or not np.isfinite(field).all():raise ValueError('Deformation grid mismatch')
            # Independently sample only occupied lesion centers, avoiding a
            # second full-volume coordinate field during parallel preparation.
            points=np.argwhere(combined>0)
            backward=field[:,points[:,0],points[:,1],points[:,2]]
            pull=points.T+np.linalg.inv(affine[:3,:3])@backward
            inside=map_coordinates(support.astype(np.uint8),pull,order=0,mode='constant',cval=0,prefilter=False)>0
            if not inside.all():raise ValueError('Warped lesion exits its shared transformed placement support')
            result['deformation']={'field_path':deformation['field_path'],'field_sha256':deformation['field_sha256'],
                'jacobian_min_builder':deformation['jacobian_min'],'jacobian_max_builder':deformation['jacobian_max'],
                'all_jacobians_independently_recomputed_here':False,'independent_reviewer_full_field_check_required':True}
        elif np.any((combined>0)&~support):raise ValueError('Lesion exits placement support')
        baseline_row=workspace['images'][0]
        payload=archive.read(baseline_row['member'])
        if hashlib.sha256(payload).hexdigest()!=baseline_row['sha256']:raise ValueError('Baseline source package changed')
        with ZipFile(BytesIO(payload)) as nested:
            _,baseline_arrays=primary(nested)
        baseline=baseline_arrays['mri']
        if baseline.shape!=image.shape or not np.array_equal(baseline_arrays['affine'],affine):raise ValueError('Baseline/source grid mismatch')
        difference=int(np.count_nonzero(image!=baseline))
        if receipt['stage']=='geometry' and difference:raise ValueError('Geometry-only case altered the acquired background')
        if receipt['stage']!='geometry' and not difference:raise ValueError('Generated observation equals baseline')
        result.update(stage=receipt['stage'],changed_image_voxels=difference,donors=[],outside_placement_support=0,
            mask_set_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(masks[name]).tobytes() for name in sorted(masks))).hexdigest(),
            automated_decision='pass_for_stated_engineering_checks_only')
        return result


def run(root=None,watch=False):
    root=checked_root(root)
    directory=root/'qc/automated-v5';directory.mkdir(exist_ok=True)
    index=directory/'index.json'
    while True:
        rows=[]
        for receipt_path in sorted((root/'catalog/preparation').glob('*-v5.json')):
            if receipt_path.name.startswith('.'):continue
            receipt=json.loads(receipt_path.read_text());destination=directory/(receipt['case_id']+'.json')
            if destination.exists():
                result=json.loads(destination.read_text())
                if result.get('sha256')!=receipt['sha256']:raise ValueError('QC receipt binding changed')
            else:
                checked_root(root,required_bytes=512*1024**2)
                try:result=inspect(root,receipt)
                except Exception as error:
                    result={'case_id':receipt['case_id'],'path':receipt['path'],'sha256':receipt['sha256'],
                        'automated_decision':'quarantined_pending_repair','error':type(error).__name__+': '+str(error)}
                atomic_json(destination,result)
            rows.append({'case_id':result['case_id'],'sha256':result['sha256'],'automated_decision':result['automated_decision'],
                'report':str(destination.relative_to(root)),'report_sha256':sha256_file(destination)})
        atomic_json(index,{'schema':'ressectionlab-v5-automated-qc-index-v1','independent_visual_acceptance':False,'cases':rows})
        print(json.dumps({'automated_cases':len(rows),'quarantined':sum(row['automated_decision']=='quarantined_pending_repair' for row in rows)}),flush=True)
        if not watch:break
        state=json.loads((root/'catalog/healthy-batch/state-v5.json').read_text())
        if len(state['subjects'])>=50 and all(row['status']!='running' for row in state['subjects'].values()):break
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch-batch',action='store_true');args=parser.parse_args()
    run(watch=args.watch_batch)
