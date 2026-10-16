"""One bounded official SynthSeg1.0 CPU compatibility trial on admitted IXI265.

No source vessel annotation is read. The legacy model runs in its own environment;
outputs are estimates, with native-grid remapping recorded separately from the
upstream 1mm output. Neither tissue labels nor healthy-cohort membership prove
individual anatomy or tumor absence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PIN='2a2aa3bbfccb83f8253a51ca8b329b9938a2646d'
INPUT_SHA='317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950'
WEIGHT_SHA='7d2e32d298fe38dc51ea38e6a8e8fc5c665d56b63cddb409e8b199535f8b5298'
GROUPS={'white_matter_estimated':(2,41,7,46),
        'gray_matter_estimated':(3,42,8,47,10,49,11,50,12,51,13,52,17,53,18,54,26,58),
        'ventricles_estimated':(4,43,5,44,14,15)}

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write_json(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def brain_roi(image, mask, margin_mm=5.):
    """Preserve every estimated brain cell with a physical source-grid margin."""
    if mask.shape!=image.shape or mask.dtype!=np.bool_ or not mask.any():
        raise ValueError('Nonempty source-grid binary brain mask required')
    spacing=np.linalg.norm(image.affine[:3,:3],axis=0)
    margin=np.ceil(margin_mm/spacing).astype(int)
    bounds=[]
    for axis in range(3):
        occupied=np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a!=axis)))
        bounds.append((max(0,int(occupied[0])-int(margin[axis])),min(mask.shape[axis],int(occupied[-1])+1+int(margin[axis]))))
    slices=tuple(slice(lo,hi) for lo,hi in bounds)
    translation=np.eye(4);translation[:3,3]=[lo for lo,_ in bounds]
    affine=image.affine@translation
    header=image.header.copy()
    cropped=nib.Nifti1Image(np.array(image.dataobj[slices]),affine,header)
    cropped.set_qform(affine,int(image.header['qform_code']));cropped.set_sform(affine,int(image.header['sform_code']))
    coverage=np.zeros(image.shape,np.uint8);coverage[slices]=1
    if int(mask[slices].sum())!=int(mask.sum()):raise ValueError('Estimated brain was cropped')
    return cropped,coverage,{'bounds_native_exclusive':bounds,'margin_mm':margin_mm,
        'estimated_brain_cells_preserved':int(mask.sum()),'prediction_coverage_cells':int(coverage.sum()),
        'source_cells':int(np.prod(image.shape)),
        'limit':'Brain extraction is an estimate; preserving its cells does not prove complete anatomical coverage. Cropped inference may differ from full-FOV predictions.'}

def run(source,model_dir,output,timeout=300,memory_gib=4,brain_record=None,fast=False):
    import psutil  # Optional anatomy runtime; ROI-only controls need no supervisor.
    # Fixed role and byte checks precede all image decoding.
    cohort_path=ROOT/'manifests/experiments/ixi-component-person-cohort-v1.json'
    cohort=json.loads(cohort_path.read_text())
    role=next(row for row in cohort['members'] if row['person_group']=='IXI:265')
    if role['role']!='TRAIN' or role['training_admitted'] is not False:
        raise ValueError('Preserve the frozen IXI265 TRAIN, not-training-admitted role')
    if digest(source)!=INPUT_SHA:raise ValueError('Only the admitted IXI265 T1 bytes are supported')
    receipt=json.loads((model_dir/'source-receipt.json').read_text())
    if receipt['revision']!=PIN:raise ValueError('Pinned official inference source required')
    for item in receipt['files']:
        if digest(model_dir/item['path'])!=item['sha256']:raise ValueError('Changed pinned model source')
    weights=model_dir/'models/synthseg_1.0.h5'
    if digest(weights)!=WEIGHT_SHA:raise ValueError('Pinned official checkpoint required')
    output.mkdir(parents=True,exist_ok=False)
    image=nib.load(source)
    if len(image.shape)!=3 or not np.isfinite(image.affine).all():raise ValueError('Finite3D source required')
    q,qcode=image.get_qform(coded=True);s,scode=image.get_sform(coded=True)
    if qcode and scode and not np.allclose(q,s,atol=1e-4,rtol=0):raise ValueError('Source qform/sform conflict')
    inference_input=source
    crop_record=None
    if brain_record is not None:
        brain_record_bytes=brain_record.read_bytes()
        brain_doc=json.loads(brain_record_bytes)
        if brain_doc.get('status')!='succeeded' or brain_doc.get('model_id')!='synthstrip-main' or brain_doc.get('input',{}).get('sha256')!=INPUT_SHA:
            raise ValueError('Brain extraction must bind the exact admitted T1 source')
        masks=[item for item in brain_doc['outputs'] if Path(item['path']).name=='brain_mask.nii.gz']
        if len(masks)!=1 or digest(Path(masks[0]['path']))!=masks[0]['sha256']:
            raise ValueError('Verified brain mask required')
        brain_image=nib.load(masks[0]['path'])
        if brain_image.shape!=image.shape or not np.allclose(brain_image.affine,image.affine,atol=1e-5,rtol=0):
            raise ValueError('Brain extraction grid mismatch')
        values=np.asarray(brain_image.dataobj)
        if not np.all((values==0)|(values==1)):raise ValueError('Binary brain mask required')
        cropped,coverage,crop_record=brain_roi(image,values.astype(bool))
        (output/'brain-extraction-input-receipt.json').write_bytes(brain_record_bytes)
        crop_record.update(brain_record_sha256=hashlib.sha256(brain_record_bytes).hexdigest(),brain_mask_sha256=masks[0]['sha256'])
        inference_input=output/'model_input_roi.nii.gz';nib.save(cropped,inference_input)
        header=image.header.copy();header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(coverage,image.affine,header),output/'prediction_coverage_native.nii.gz')
    command=[str(model_dir/'cpu-env/bin/python'),str(model_dir/'scripts/commands/SynthSeg_predict.py'),
             '--i',str(inference_input),'--o',str(output/'labels_1mm.nii.gz'),
             '--resample',str(output/'source_resampled_1mm.nii.gz'),
             '--vol',str(output/'volumes.csv'),'--v1','--cpu','--threads','2']
    if fast:command.append('--fast')
    before={'sha256':digest(source),'mtime_ns':source.stat().st_mtime_ns}
    record={'schema':'resectionlab-poc-synthseg-v1','case':'IXI265-Guys-0845','person_group':'IXI:265',
            'role':'TRAIN','training_admitted':False,'model':'SynthSeg1.0','model_id':'synthseg-1.0-cpu',
            'revision':PIN,'license':'Apache-2.0','weights_sha256':WEIGHT_SHA,
            'runner_sha256':digest(Path(__file__)),
            'source_receipt_sha256':digest(model_dir/'source-receipt.json'),
            'environment_requirements_sha256':digest(ROOT/'manifests/synthseg-pilot-runtime-requirements.txt'),
            'input':{'path':str(source),'sha256':INPUT_SHA,'shape':list(image.shape),'affine_ras_mm':image.affine.tolist()},
            'parameters':{'upstream_mode':'--v1 --cpu --threads2'+(' --fast' if fast else ''),'crop':None,'fast':fast,'parcellation':False,'automated_qc':False,
                          'fast_mode_difference':'Official flag disables flip averaging and topology postprocessing; output equivalence/accuracy benefit is not asserted' if fast else None},
            'command':command,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
            'model_input':{'path':str(inference_input),'sha256':digest(inference_input)},'roi':crop_record,
            'memory_limit_bytes':int(memory_gib*1024**3),'timeout_seconds':timeout,
            'minimum_available_memory_bytes':int(.75*1024**3),'memory_sampling_seconds':.2,
            'memory_caveat':'Sampled worker process-family RSS; allocator/native-driver peaks and between-sample peaks may differ. No unified-memory sum is inferred.',
            'limits':['Compatibility trial on Apple Silicon: Python3.9 TensorFlow2.13.1/Keras2.13.1, rather than historical upstream Python3.8/TensorFlow2.2 dependencies',
                      'Retained official v1 checkpoint; no v2 parcellation or automated-QC weights',
                      'Estimated anatomy; no manual reference or clinical accuracy claim',
                      'Source-described healthy cohort does not certify individual tumor absence',
                      'Binary tissue-group membership is not measured histological fraction']}
    write_json(output/'inference.json',record)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                     TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='1',PYTHONHASHSEED='0')
    started=time.monotonic();peak=0;reason=None
    with (output/'worker.log').open('w') as log:
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            try:
                family=[psutil.Process(process.pid),*psutil.Process(process.pid).children(recursive=True)]
                rss=sum(p.memory_info().rss for p in family if p.is_running());peak=max(peak,rss)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if peak>record['memory_limit_bytes']:reason='memory limit'
            if psutil.virtual_memory().available<record['minimum_available_memory_bytes']:reason='system available-memory floor'
            if time.monotonic()-started>timeout:reason='time limit'
            if reason:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(.2)
    record.update(elapsed_seconds=time.monotonic()-started,exit_code=process.returncode,
                  peak_process_family_rss_bytes_sampled=peak,
                  input_unchanged=before=={'sha256':digest(source),'mtime_ns':source.stat().st_mtime_ns})
    if process.returncode or reason:
        record.update(status='failed',failure=reason or 'See retained worker log')
        write_json(output/'inference.json',record)
        raise RuntimeError(record['failure'])
    try:
        label_image=nib.load(output/'labels_1mm.nii.gz')
        labels=np.asarray(label_image.dataobj)
        if not np.isfinite(labels).all() or not np.array_equal(labels,np.rint(labels)) or np.count_nonzero(np.bincount(labels.astype(np.int16).ravel()))<10:
            raise ValueError('Missing or invalid multiclass prediction')
        native_image=resample_from_to(label_image,(image.shape,image.affine),order=0)
        native=np.asarray(native_image.dataobj).astype(np.int16)
        header=image.header.copy();header.set_data_dtype(np.int16)
        native_out=nib.Nifti1Image(native,image.affine,header)
        native_out.set_qform(image.get_qform(),int(image.header['qform_code']));native_out.set_sform(image.get_sform(),int(image.header['sform_code']))
        nib.save(native_out,output/'labels_native.nii.gz')
        outputs=[]
        for name,codes in GROUPS.items():
            mask=np.isin(native,codes).astype(np.uint8)
            if not mask.any():raise ValueError('Missing expected group: '+name)
            header=image.header.copy();header.set_data_dtype(np.uint8)
            out=nib.Nifti1Image(mask,image.affine,header)
            out.set_qform(image.get_qform(),int(image.header['qform_code']));out.set_sform(image.get_sform(),int(image.header['sform_code']))
            path=output/(name+'.nii.gz');nib.save(out,path)
            outputs.append({'path':str(path),'name':name,'sha256':digest(path),'label_values':list(codes),
                            'foreground_voxels':int(mask.sum()),'shape':list(mask.shape),'affine_ras_mm':image.affine.tolist(),'provenance':'estimated'})
        record.update(status='succeeded',outputs=outputs,unique_labels=np.flatnonzero(np.bincount(native.ravel())).tolist(),
                      upstream_1mm_grid={'shape':list(labels.shape),'affine_ras_mm':label_image.affine.tolist()},
                      native_remapping='Nearest-neighbor world-coordinate resampling of official1mm labels; acquired T1 unchanged',
                      labels_native_sha256=digest(output/'labels_native.nii.gz'))
        if not record['input_unchanged']:raise ValueError('Source changed during inference')
    except Exception as error:
        record.update(status='failed',failure=str(error));write_json(output/'inference.json',record);raise
    write_json(output/'inference.json',record)
    return {key:record[key] for key in ('status','elapsed_seconds','peak_process_family_rss_bytes_sampled','input_unchanged','unique_labels')}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'.tools/poc-synthseg-2a2aa3b')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--brain-record',type=Path,help='Verified exact-source SynthStrip receipt for a brain-preserving ROI')
    parser.add_argument('--fast',action='store_true',help='Explicit official lower-cost alternative, not equivalent to the full method')
    args=parser.parse_args()
    print(json.dumps(run(args.input.resolve(),args.model_dir.resolve(),args.output.resolve(),brain_record=args.brain_record,fast=args.fast)))
