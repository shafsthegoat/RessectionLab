#!/usr/bin/env python3
"""Independent post-inference saved-array QC; no network, variant, or access selection."""
from pathlib import Path
from datetime import datetime,timezone
import gc,hashlib,importlib.util,itertools,json,resource,sys,time
import nibabel as nib
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
HELPER=ROOT/'scripts/audit_brain_extraction.py'
spec=importlib.util.spec_from_file_location('independent_extraction_helpers',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha=helper.digest
DECLARATION=ROOT/'manifests/experiments/brain-extraction-btc-spatial-main-v1.json'
BATCH=ROOT/'artifacts/brain-extraction/BTC-spatial-main-v1-batch/inference-batch.json'
EXPECTED=('sub-PAT22','sub-PAT25','sub-PAT26','sub-PAT27')


def verify(path,expected):
    if sha(path)!=expected:raise ValueError(f'Hash mismatch: {path}')


def centroid(mask):
    return np.array([np.dot(np.arange(mask.shape[a]),mask.sum(axis=tuple(j for j in range(3) if j!=a),dtype=np.float64))/mask.sum() for a in range(3)])


def native_displacement(path,reference):
    image=nib.load(path);corners=np.array(list(itertools.product(*[(-.5,n-.5) for n in reference.shape])))
    discrepancy=float(np.linalg.norm((np.c_[corners,np.ones(8)]@(image.affine-reference.affine).T)[:,:3],axis=1).max())
    if discrepancy>.01:raise ValueError('Artifact changed native physical cell corners')
    return {'maximum_full_cell_corner_displacement_mm':discrepancy,'affine_exactly_equal':bool(np.array_equal(image.affine,reference.affine))}


def render(subject,role,image,mask,target,measurements):
    source=image.get_fdata(dtype=np.float32);outside=target&~mask
    centers=[('Envelope centroid',centroid(mask)),('Source annotation centroid',centroid(target))]
    if outside.any():centers.append(('Excluded source-annotation detail',centroid(outside)))
    fig,axes=plt.subplots(len(centers),3,figsize=(13,4.4*len(centers)),squeeze=False)
    low,high=np.percentile(source,(1,99.5));plane_records=[]
    for row,(title,center) in enumerate(centers):
        for axis,ax in enumerate(axes[row]):
            k=int(np.rint(center[axis]));other=[j for j in range(3) if j!=axis]
            values=np.take(source,k,axis=axis);envelope=np.take(mask,k,axis=axis);annotation=np.take(target,k,axis=axis);omission=np.take(outside,k,axis=axis)
            ax.imshow(values.T,origin='lower',cmap='gray',vmin=low,vmax=high)
            if envelope.any() and not envelope.all():ax.contour(envelope.T,[.5],colors=['#39dcbd'],linewidths=.8)
            if annotation.any() and not annotation.all():ax.contour(annotation.T,[.5],colors=['#ffb33b'],linewidths=.9)
            if omission.any():
                overlay=np.ma.masked_where(~omission.T,omission.T)
                ax.imshow(overlay,origin='lower',cmap=matplotlib.colors.ListedColormap(['#f047ba']),alpha=.85,vmin=0,vmax=1)
            if row==2:
                ax.set_xlim(max(0,center[other[0]]-30),min(source.shape[other[0]]-1,center[other[0]]+30))
                ax.set_ylim(max(0,center[other[1]]-30),min(source.shape[other[1]]-1,center[other[1]]+30))
            ax.set_title(f'{title}\nNative axis {axis}, index {k}',fontsize=10)
            ax.set_xlabel(f'voxel axis {other[0]}');ax.set_ylabel(f'voxel axis {other[1]}')
            plane_records.append({'row':title,'axis':axis,'index':k,'source_annotation_voxels':int(annotation.sum()),'excluded_annotation_voxels':int(omission.sum())})
    title_role='TRAIN' if role=='population_training' else 'SELECT'
    fig.suptitle(f'{subject} · {title_role} · Fixed main model estimate · Review required\nGreen: estimated envelope; orange: source annotation threshold0.5; magenta: excluded annotation\n{measurements["source_annotation_outside_voxels"]}/{measurements["source_annotation_voxels"]} annotation voxels outside; no anatomical or cortical approval',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.925));path=OUT/f'{subject}-native-qc.png';fig.savefig(path,dpi=170);plt.close(fig)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'planes':plane_records,'expert_anatomical_review':False}


def main():
    started=time.perf_counter()
    if (OUT/'independent-qc.json').exists():raise FileExistsError('Retain the existing audit')
    verify(DECLARATION,'f6560e7d35a3c85d3e56f2abfbaa487337f22405a4be20dc28ddeb368f78450f')
    verify(BATCH,'57b41c12d937358165ca4aa8e46d0085b9577ee1619034a375e2c90d84bbc975')
    declaration=json.loads(DECLARATION.read_text());batch=json.loads(BATCH.read_text())
    if tuple(declaration['allowed_subjects'])!=EXPECTED or tuple(a['subject'] for a in batch['attempts'])!=EXPECTED:raise ValueError('Unexpected subject/role inventory')
    if batch['status']!='all_four_inferences_completed_before_overlap_QC' or batch['failures']:raise ValueError('All outputs must already be fixed')
    if datetime.fromisoformat(batch['finished_at'])>=datetime.now(timezone.utc):raise ValueError('Batch completion does not precede this audit')
    for name,record in declaration['model']['files'].items():verify(ROOT/declaration['model_cache']/name,record['sha256'])
    results=[]
    for item,attempt in zip(declaration['subjects'],batch['attempts']):
        subject=item['subject'];directory=ROOT/item['output_directory']
        if attempt['subject']!=subject or attempt['development_role']!=item['development_role']:raise ValueError('Role mismatch')
        for name,pinned in attempt['artifact_files'].items():verify(directory/name,pinned['sha256'])
        verify(ROOT/item['source_t1'],item['source_t1_sha256']);verify(ROOT/item['source_manifest'],item['source_manifest_sha256']);verify(ROOT/item['source_case'],item['source_case_sha256'])
        report=json.loads((directory/'brain_extraction_report.json').read_text());run=report['variants']['main']['inference']
        if list(report['variants'])!=['main'] or report['source_annotation'] is not None:raise ValueError('Inference report variant/annotation boundary differs')
        if report['source_t1_sha256']!=item['source_t1_sha256'] or report['development_role']!=item['development_role']:raise ValueError('Run bound to wrong source/role')
        if report['brain_reviewed'] or report['cortical_access_permitted'] or run['brain_reviewed'] or run['cortical_access_permitted']:raise ValueError('Unexpected anatomical promotion')
        if run['failure'] is not None or run['exit_code']!=0 or run['model']!=declaration['model']:raise ValueError('Model or successful execution differs')
        for key in ('device','cpu_threads','border_mm','timeout_seconds','maximum_rss_bytes'):
            if run['configuration'][key]!=declaration['frozen_configuration'][key]:raise ValueError(f'Setting drift: {key}')
        if run['executed_runner_sha256']!=declaration['expected_executed_MPS_runner_sha256']:raise ValueError('Runner mismatch')
        verify(directory/'implementation_snapshot.py',declaration['wrapper']['sha256'])
        source_manifest=json.loads((ROOT/item['source_manifest']).read_text())
        annotation_relative=f'derivatives/tumor_masks/{subject}/anat/{subject}_space_T1_label-tumor.nii'
        annotation_record=next(r for r in source_manifest['files'] if r['path']==annotation_relative)
        annotation_path=ROOT/'data/diffusion_source/ds001226-v5.0.1'/annotation_relative;verify(annotation_path,annotation_record['sha256'])
        reference=nib.load(ROOT/item['source_t1'])
        target=helper.source_annotation(annotation_path,reference,threshold=.5)
        mask,mg=helper.read_native(directory/'main_mask.nii.gz',reference,binary=True)
        distance,dg=helper.read_native(directory/'main_distance_mm.nii.gz',reference,binary=False)
        measurements=helper.measure(mask,distance,target,reference.affine,border_mm=1.)
        mg.update(native_displacement(directory/'main_mask.nii.gz',reference));dg.update(native_displacement(directory/'main_distance_mm.nii.gz',reference))
        visual=render(subject,item['development_role'],reference,mask,target,measurements)
        results.append({'subject':subject,'development_role':item['development_role'],'source_t1_sha256':item['source_t1_sha256'],
            'source_annotation_sha256':annotation_record['sha256'],'source_annotation_threshold':.5,'source_manifest_sha256':item['source_manifest_sha256'],
            'original_report_sha256':sha(directory/'brain_extraction_report.json'),'mask_sha256':sha(directory/'main_mask.nii.gz'),'distance_sha256':sha(directory/'main_distance_mm.nii.gz'),
            'source_model_wrapper_runner_and_artifacts_verified':True,'mask_geometry':mg,'distance_geometry':dg,'measurements':measurements,
            'visual':visual,'review_status':'review_required','brain_reviewed':False,'cortex_localized':False,'cortical_access_permitted':False,
            'target_omission_used_to_select_variant_or_access':False,'working_brain_mask_selected':False})
        print(json.dumps({'subject':subject,'volume_ml':measurements['volume_ml'],'omission_voxels':measurements['source_annotation_outside_voxels'],'target_voxels':measurements['source_annotation_voxels'],'reconstruction_mismatch':measurements['reconstructed_mask_mismatch_voxels'],'face_contact':measurements['input_face_contact_voxels']}),flush=True)
        del reference,mask,target,distance;gc.collect()
    for item,attempt in zip(declaration['subjects'],batch['attempts']):
        verify(ROOT/item['source_t1'],item['source_t1_sha256']);verify(ROOT/item['source_case'],item['source_case_sha256'])
        for name,pinned in attempt['artifact_files'].items():verify(ROOT/item['output_directory']/name,pinned['sha256'])
    receipt={'schema_version':1,'status':'independent_array_and_binding_checks_complete_visual_read_pending','declaration_sha256':sha(DECLARATION),'inference_batch_sha256':sha(BATCH),
        'auditor_sha256':sha(Path(__file__)),'independent_helper_sha256':sha(HELPER),'completed_utc':datetime.now(timezone.utc).isoformat(),'subjects':results,
        'all_four_outputs_fixed_before_annotation_QC':True,'original_sources_and_inference_artifacts_unchanged':True,'model_runs_performed':0,'clinical_deficit_probability':None,
        'expert_anatomical_review':False,'review_required':True,'forbidden_subjects_opened':False,'external_final_cohorts_opened':False,
        'limitations':['Source annotation inclusion is a discrepancy check, not segmentation accuracy.','Predicted SDT, including upstream exterior100mmfill, is not an independently measured surgical clearance field.','Estimated whole-brain support does not localize cortex or approve an access window.','Exact pretrained-model cohort overlap remains unverified; no disjointness claim.','SELECT cases retain their frozen role; no target-driven variant, threshold, rescue, access or retry selection.'],
        'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'nibabel':nib.__version__,'matplotlib':matplotlib.__version__}}
    (OUT/'independent-qc.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'seconds':receipt['elapsed_seconds'],'peak_mib':receipt['peak_rss_bytes']/1024**2,'receipt_sha256':sha(OUT/'independent-qc.json')}))

if __name__=='__main__':main()
