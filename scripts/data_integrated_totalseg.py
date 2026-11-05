#!/usr/bin/env python3
"""Cached CT anatomy estimates through nnU-Net's official single-array API.

Avoids multiprocessing tensor export on ExFAT. No model training/download,
source editing, scene generation, tissue reclassification or action admission.
All model outputs remain estimates pending independent review.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src')]
PREPARATION_SUBJECTS=('004','006','012','014','018')


def predict_one_file(predictor, folder, output, *, save_probabilities=False,
                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0, **unused):
    """Exact installed API's XYZ↔ZYX convention; same input grid on return."""
    import nibabel as nib
    import numpy as np
    if save_probabilities or folder_with_segs_from_prev_stage is not None or num_parts!=1 or part_id!=0:
        raise ValueError('Single-image noncascade segmentation only')
    source=Path(folder)/'s01_0000.nii.gz'
    image=nib.load(source)
    if len(image.shape)!=3:raise ValueError('Expected one 3D model input')
    data=np.asarray(image.dataobj,dtype=np.float32).transpose(2,1,0)[None,...]
    spacing=tuple(float(x) for x in image.header.get_zooms()[::-1])
    result=predictor.predict_single_npy_array(data,{'spacing':spacing},None,None,False)
    result=np.asarray(result)
    if (result.shape!=data.shape[1:] or not np.isfinite(result).all()
            or np.any(result<0) or np.any(result!=np.floor(result)) or result.max()>255):
        raise ValueError('Single-array result did not restore its exact input grid/integer ontology')
    header=image.header.copy();header.set_data_dtype(np.uint8)
    destination=Path(output)/'s01.nii.gz'
    destination.parent.mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(result.transpose(2,1,0).astype(np.uint8),image.affine,header),destination)
    restored=nib.load(destination)
    if restored.shape!=image.shape or not np.array_equal(restored.affine,image.affine):
        raise ValueError('Model intermediate grid changed')


def run(root,subject,version):
    import re
    from resectionlab.data_root import checked_root,safe_path,sha256_file,atomic_json
    root=checked_root(root,required_bytes=4*1024**3)
    if subject not in PREPARATION_SUBJECTS or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}',version):
        raise ValueError('Only the five reviewed TRAIN preparation subjects and one safe version token are permitted')
    receipt=json.loads((root/f'catalog/topbrain/{subject}-selective-extraction.json').read_text())
    if receipt['family']['family_id']!=f'TopCoW:{subject}' or receipt['family']['role']!='TRAIN':
        raise ValueError('Frozen TRAIN source identity differs')
    source_row=next(row for row in receipt['files'] if '/imagesTr_topbrain/topcow_ct_' in row['source_member'])
    source=safe_path(root,source_row['path'])
    if sha256_file(source)!=source_row['sha256']:raise ValueError('Original source changed')
    output=root/f'integrated-v1/anatomy/TopCoW-{subject}'/version
    if output.exists():raise FileExistsError('Preserve previous anatomy attempts')
    output.mkdir();temporary=output/'temporary';temporary.mkdir()
    model_root=root/'cache/models';weights=model_root/'poc-totalsegmentator/weights'
    checkpoints=list(weights.glob('**/checkpoint_final.pth'))
    if len(checkpoints)!=1:raise ValueError('Exactly one already cached fast model is required; no download')
    os.environ.update(RESSECTIONLAB_MODEL_ROOT=str(model_root),TOTALSEG_HOME_DIR=str(model_root/'poc-totalsegmentator/config'),
        TOTALSEG_WEIGHTS_PATH=str(weights),nnUNet_raw=str(temporary/'raw'),nnUNet_preprocessed=str(temporary/'preprocessed'),
        nnUNet_results=str(weights),TMPDIR=str(temporary),MPLCONFIGDIR=str(root/'cache/matplotlib'))
    import tempfile
    tempfile.tempdir=str(temporary)
    started=time.monotonic()
    plan={'schema':'single-array-cached-CT-pilot-v1','subject':subject,'role':'TRAIN','source':source_row,
        'adapter_sha256':sha256_file(__file__),'worker_sha256':sha256_file(REPO/'src/resectionlab/poc_models.py'),
        'weights':[{'path':str(p.relative_to(root)),'sha256':sha256_file(p)} for p in checkpoints],
        'method':'Installed TotalSegmentator total/fast with only file prediction replaced by official nnUNetPredictor.predict_single_npy_array; original native restoration retained',
        'new_model_training':False,'downloads_allowed':False,'scene_or_source_edits':False,
        'limits':['Estimated anatomy only; not clinical parenchyma/CSF truth','Source CT scaled intensities are HU-like; original DICOM calibration unavailable, model applicability remains uncertain','Excluded envelope voxels remain unresolved, not automatically air or free space','No tool or native-case promotion before separate independent source/estimate review'],
        'prior_attempts':'integrated-v1/anatomy/006-model-pilot-attempts.json'}
    atomic_json(output/'prospective-method.json',plan)
    (output/'data_integrated_totalseg.py.snapshot').write_bytes(Path(__file__).read_bytes())
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        nnUNetPredictor.predict_from_files=predict_one_file
        import totalsegmentator.python_api as api
        def cached_only(task_id,*args,**kwargs):
            if int(task_id)!=297 or not checkpoints[0].is_file():raise ValueError('Only already cached Dataset297 allowed; downloads forbidden')
        api.download_pretrained_weights=cached_only
        from totalsegmentator.map_to_binary import class_map
        if class_map['total'].get(90)!='brain' or class_map['total'].get(91)!='skull':
            raise ValueError('Installed ontology differs from declared brain/skull labels')
        from resectionlab.poc_models import worker
        worker({'image':str(source),'output':str(output),'device':'mps','model_id':'totalsegmentator-skull'})
        import nibabel as nib
        import numpy as np
        reference=nib.load(source);outputs=[]
        for name in ['brain_mask.nii.gz','skull_mask.nii.gz']:
            p=output/name;image=nib.load(p);values=np.asanyarray(image.dataobj)
            if image.shape!=reference.shape or not np.array_equal(image.affine,reference.affine) or not np.isin(values,[0,1]).all():
                raise ValueError('Final anatomy did not preserve source physical grid/binary semantics')
            outputs.append({'path':str(p.relative_to(root)),'sha256':sha256_file(p),'voxels':int(values.sum())})
        if sha256_file(source)!=source_row['sha256']:raise ValueError('Source changed during pilot')
        result={'status':'succeeded_pending_independent_review','seconds':time.monotonic()-started,'outputs':outputs,
            'versions':{p:importlib.metadata.version(p) for p in ['torch','TotalSegmentator','nnunetv2','nibabel']},'clinical_validation':False}
    except Exception as error:
        result={'status':'failed_without_promotion','seconds':time.monotonic()-started,'error':repr(error),'clinical_validation':False}
        atomic_json(output/'result.json',result)
        raise
    atomic_json(output/'result.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--subject',default='006');parser.add_argument('--version',required=True);args=parser.parse_args()
    print(json.dumps(run(args.root,args.subject,args.version),indent=2))
