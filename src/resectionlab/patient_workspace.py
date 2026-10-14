"""Prepare one source-identified, cached imaging workspace without inference.

The POC case uses the downloaded UCSF-PDGM mirror subset. Its source-declared
preprocessed registration is not scanner-native geometry or a clinical review.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from .core import SourceRef, semantic_digest, thaw_json
from .imaging import _aligned_mask, file_sha256, inspect_nifti, load_nifti_case
from .workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
from .workspace_imaging import DisplaySource, source_registration

UCSF_MANIFEST_SHA256='b05552355f2099058e9d25bc93faf9f2981fe1c00ffb4f24443edfabeda4c6f6'
UCSF_CASE='UCSF-PDGM-0004'
COLORS={'nonenhancing_or_necrotic_core':'#d49aff','FLAIR_abnormality':'#ffbf69','enhancing_target':'#ff557e','brain':'#65bfff'}
NAMES={'nonenhancing_or_necrotic_core':'Nonenhancing / necrotic tumor core','FLAIR_abnormality':'Peritumoral FLAIR abnormality','enhancing_target':'Enhancing tumor','brain':'Brain extraction (estimated)'}


def _verified_path(root,item):
    path=(root/item['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Acquisition source is missing or escapes its data root')
    if path.stat().st_size!=item['size_bytes'] or file_sha256(path)!=item['sha256']:
        raise ValueError(f'Acquisition bytes changed: {path.name}')
    return path


def prepare_ucsf_workspace(manifest_path,data_root,output,*,brain_mask=None,brain_inference=None,tumor_inference=None):
    """Verify pinned public files, preserve original bytes, and save/reopen POC A.

    No network or segmentation calls occur here. Optional inference output is
    imported only with its successful, hash-bound run manifest.
    """
    started=time.perf_counter();manifest_path=Path(manifest_path);root=Path(data_root).resolve();output=Path(output)
    if file_sha256(manifest_path)!=UCSF_MANIFEST_SHA256:
        raise ValueError('POC import requires the pinned UCSF-PDGM-0004 acquisition manifest')
    manifest=json.loads(manifest_path.read_text())
    if manifest['case_id']!=UCSF_CASE or manifest['patient_cluster_id']!=UCSF_CASE or manifest['license']['identifier']!='CC-BY-4.0':
        raise ValueError('Source patient or license differs from reviewed acquisition')
    originals=[OriginalFile.from_path(manifest_path,role='metadata',expected_sha256=UCSF_MANIFEST_SHA256)]
    paths={};source_files={}
    for item in manifest['files']:
        path=_verified_path(root,item);paths[item['modality']]=path;source_files[item['modality']]=item
        if not path.name.startswith(UCSF_CASE+'_') or item['source_remote_path'].split('/')[0]!=UCSF_CASE:
            raise ValueError('Every image must belong to the same declared source patient')
        originals.append(OriginalFile.from_path(path,role='annotation' if item['modality']=='tumor_segmentation' else 'imaging',expected_sha256=item['sha256']))
    if set(paths)!={'T1','T1c','T2','FLAIR','tumor_segmentation'}:
        raise ValueError('The source must contain its complete four-modality same-patient release')
    metadata_paths={}
    for item in manifest['metadata_files']:
        path=_verified_path(root,item);metadata_paths[path.name]=path
        originals.append(OriginalFile.from_path(path,role='metadata',expected_sha256=item['sha256']))
    evidence=metadata_paths['mirror_README.md']
    # This pinned source text explicitly asserts co-registration and subject
    # grouping. Header agreement alone never enables multimodal fusion.
    registration={'kind':'source-provided-coregistration','patient_id':UCSF_CASE,
        'group_id':f"MedOtter/UCSF-PDGM/{manifest['mirror']['revision']}/{UCSF_CASE}/baseline_preoperative",
        'source_sha256_by_modality':{('T1CE' if k=='T1c' else k):v['sha256'] for k,v in source_files.items() if k!='tumor_segmentation'},
        'evidence_sha256':file_sha256(evidence),
        'evidence_url':f"{manifest['mirror']['url']}/blob/{manifest['mirror']['revision']}/README.md"}
    label_map={int(label):info['name'] for label,info in manifest['label_dictionary'].items() if int(label)>0}
    layers={name:{'name':name,'label':NAMES.get(name,name),'kind':'tumor','provenance':'source-provided',
        'source_role':'imported_source_annotation','source_sha256':source_files['tumor_segmentation']['sha256'],
        'source_uri':paths['tumor_segmentation'].as_uri(),'label_value':value,'color':COLORS.get(name,'#65bfff'),
        'model':None,'cache_key':semantic_digest({'mask_sha256':source_files['tumor_segmentation']['sha256'],'label':value}),
        'review_status':'source_expert_corrected; local anatomical review pending',
        'coverage':'Source labels; unlabelled voxels are not certified normal anatomy'} for value,name in label_map.items()}
    common={'patient_id':UCSF_CASE,'patient_name':UCSF_CASE,'session_id':'baseline_preoperative','anatomy_workbench_only':True,
        'acquisition_datetime':None,'acquisition_datetime_status':'not_released_in_source',
        'source_collection':manifest['source_collection'],'source_distribution':manifest['mirror'],
        'source_citation':manifest['license']['citation'],'source_license':manifest['license'],
        'source_files':manifest['files'],'acquisition_manifest_sha256':UCSF_MANIFEST_SHA256,
        'workspace_registration':registration,'anatomy_layers':layers,'split':manifest['split'],
        'source_frame_declaration':manifest['spatial_frame']['meaning'],
        'primary_source_equivalence':'unverified_public_mirror',
        'original_source_status':'Exact downloaded preprocessed NIfTI bytes preserved; raw DICOM and acquired scanner-native geometry unavailable',
        'missing_modalities':['CT','CTA','MRA','TOF-MRA','DICOM'],
        'missing_anatomy':['skull','vessels','ventricles'],
        'segmentation_cache_policy':'saved_masks_only; no automatic inference on open or layer changes'}
    if (brain_mask is None)!=(brain_inference is None):raise ValueError('A cached brain mask requires its inference manifest')
    if brain_mask is not None:
        brain_mask=Path(brain_mask);brain_inference=Path(brain_inference)
        inferred=_brain_record(brain_mask,brain_inference,registration['source_sha256_by_modality'])
        layers['brain']=inferred
        originals.extend([OriginalFile.from_path(brain_mask,role='model-result'),OriginalFile.from_path(brain_inference,role='metadata')])
    primary=load_nifti_case(paths['T1c'],paths['tumor_segmentation'],case_id=UCSF_CASE,label_map=label_map,
        brain_mask_path=brain_mask,source_url=manifest['source_collection']['official_url'],license=manifest['license']['identifier'],
        metadata={**common,'selected_modality':'T1CE'})
    if brain_mask is not None:
        primary=primary.revised(source_refs=tuple(replace(ref,provenance='estimated') if ref.source_id=='supplied_brain_mask' else ref for ref in primary.source_refs))
    if tumor_inference is not None:
        tumor_masks,tumor_layers,tumor_originals,tumor_refs=_tumor_results(Path(tumor_inference),primary,registration['source_sha256_by_modality'])
        layers.update(tumor_layers);originals.extend(tumor_originals)
        primary=replace(primary,compartments={**primary.compartments,**tumor_masks},
            source_compartments={**primary.source_compartments,**tumor_masks},source_refs=(*primary.source_refs,*tumor_refs),
            metadata={**thaw_json(primary.metadata),'anatomy_layers':layers})
    sources=[]
    for modality in ('T1','T2','FLAIR'):
        case=load_nifti_case(paths[modality],paths['tumor_segmentation'],case_id=UCSF_CASE,label_map=label_map,
            source_url=manifest['source_collection']['official_url'],license=manifest['license']['identifier'],
            metadata={**common,'selected_modality':modality,'anatomy_layers':{k:v for k,v in layers.items() if k in label_map.values()}})
        if not source_registration(primary,case)[1]['overlayPermitted']:
            raise ValueError(f'{modality} has not passed the source-registered geometry checks')
        sources.append(DisplaySource(case,modality,'source-provided'))
    state=view_state(primary,sources)
    # Start at the source tumor's centroid while retaining native RAS positions.
    mask=np.logical_or.reduce(list(primary.compartments.values()))
    center=(primary.affine@np.r_[np.argwhere(mask).mean(axis=0),1.])[:3].tolist()
    for value in state['states'].values():
        value['cursor']=center
        names=list(value['visibleLayers'])
        value['viewer']={'layerOpacity':{name:.72 for name in names},
                         'layerColor':{name:layers[name]['color'] for name in names}}
        for name in names:
            if name.startswith('monai_'):value['visibleLayers'][name]=False
    if brain_mask is not None:state['states']['primary']['viewer']['layerOpacity']['brain']=.16
    output.parent.mkdir(parents=True,exist_ok=True)
    session_hash=save_workspace(primary,output,sources=sources,imaging=state,episode=None,selection=None,artifacts={},originals=originals)
    reopened,_,restored=load_workspace(output)
    if reopened.semantic_hash!=primary.semantic_hash or restored['sessionHash']!=session_hash or restored['imagingState']!=state:
        raise RuntimeError('Canonical workspace changed across save/reopen')
    if [item.sha256 for item in originals]!=[item.sha256 for item in restored['originals']]:
        raise RuntimeError('Archived original source bytes changed')
    report={'schema':'poc-patient-workspace-v1','case_id':UCSF_CASE,'case_hash':primary.semantic_hash,'session_hash':session_hash,
        'bundle':str(output.resolve()),'source_manifest_sha256':UCSF_MANIFEST_SHA256,'modalities':['T1CE','T1','T2','FLAIR'],
        'layers':list(layers),'original_files':len(originals),'original_bytes':sum(len(item.data) for item in originals),
        'same_patient':True,'registration':'source-provided-coregistration; matching grids verified',
        'original_dicom_available':False,'scanner_native_geometry_available':False,'acquisition_datetime':None,
        'source_equivalence':'pinned public mirror; official TCIA byte equivalence unverified',
        'reopened_identical':True,'reinference_performed':False,'elapsed_seconds':time.perf_counter()-started}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    _catalog(output,report)
    return report


def _tumor_results(report_path,primary,images):
    report=json.loads(report_path.read_text())
    if (report.get('schema')!='resectionlab-poc-inference-v1' or report.get('status')!='succeeded' or
            report.get('case')!=primary.case_id):
        raise ValueError('Tumor inference must be a successful result for this same patient')
    inputs=report.get('inputs',[])
    bound={('T1CE' if row.get('modality')=='T1c' else row.get('modality')):row.get('sha256') for row in inputs}
    if bound!=images or len(inputs)!=4:raise ValueError('Tumor model requires all four exact registered MRI inputs')
    model=_checked_model(report);qc=thaw_json(primary.metadata['imaging_qc']['structural'])
    originals=[OriginalFile.from_path(report_path,role='metadata')];masks={};layers={};refs=[]
    labels={'monai_whole_tumor':'MONAI whole tumor (estimated)','monai_tumor_core':'MONAI tumor core (estimated)',
            'monai_enhancing_tumor':'MONAI enhancing tumor (estimated)'}
    colors={'monai_whole_tumor':'#ffd866','monai_tumor_core':'#b59aff','monai_enhancing_tumor':'#ff7999'}
    for row in report['outputs']:
        path=Path(row['path']);name=path.name.removesuffix('_mask.nii.gz')
        if name not in labels:continue
        if file_sha256(path)!=row['sha256']:raise ValueError('Saved tumor mask bytes changed')
        values,_=_aligned_mask(path,qc)
        if set(np.unique(values))-{0,1} or not values.any():raise ValueError('Expected nonempty binary tumor model mask')
        if name in masks:raise ValueError('Duplicate tumor model output')
        masks[name]=values.astype(bool)
        layers[name]={'name':name,'label':labels[name],'kind':'tumor','provenance':'estimated',
            'source_role':'cached_model_inference','source_sha256':row['sha256'],'source_uri':path.resolve().as_uri(),
            'input_sha256_by_modality':images,'color':colors[name],'model':model,'cache_key':report['cache_key'],
            'inference_manifest_sha256':file_sha256(report_path),'runtime_seconds':report['elapsed_seconds'],
            'review_status':'model output; local anatomical review pending','coverage':'Model estimate; imported source labels remain separate',
            'benchmark_caveat':'Known BraTS training cohort overlap; not independent model validation'}
        originals.append(OriginalFile.from_path(path,role='model-result',expected_sha256=row['sha256']))
        refs.append(SourceRef('model_'+name,path.resolve().as_uri(),sha256=row['sha256'],license=model['license'],native_frame='RAS+',provenance='estimated'))
    if set(masks)!=set(labels):raise ValueError('Tumor inference is missing a required saved mask')
    return masks,layers,originals,refs


def _brain_record(mask,report_path,images):
    """Accept the explicit POC inference record shared by the model runner."""
    report=json.loads(report_path.read_text())
    if report.get('schema')!='resectionlab-poc-inference-v1' or report.get('status')!='succeeded':
        raise ValueError('Only a successful saved inference result can be imported')
    source_hash=report.get('input',{}).get('sha256')
    outputs=[row for row in report.get('outputs',[]) if Path(row.get('path','')).resolve()==mask.resolve()]
    if source_hash not in images.values() or len(outputs)!=1 or outputs[0].get('sha256')!=file_sha256(mask):
        raise ValueError('Inference mask/input does not belong to this patient acquisition')
    model=_checked_model(report)
    return {'name':'brain','label':NAMES['brain'],'kind':'brain','provenance':'estimated',
        'source_role':'cached_model_inference','source_sha256':file_sha256(mask),'source_uri':mask.resolve().as_uri(),
        'input_sha256':source_hash,'color':COLORS['brain'],'model':model,
        'cache_key':report['cache_key'],
        'inference_manifest_sha256':file_sha256(report_path),'runtime_seconds':report.get('elapsed_seconds'),
        'review_status':'model output; local anatomical review pending','coverage':'Brain extraction envelope; not skull, vessels or parcellation'}


def _checked_model(report):
    model=report.get('model');weights=report.get('weights')
    if (not isinstance(model,dict) or not all(model.get(k) for k in ('name','version','license')) or
            not isinstance(weights,list) or not weights or not report.get('cache_key')):
        raise ValueError('Inference requires model version, license, weights and cache identity')
    for row in weights:
        path=Path(row['path'])
        if not path.is_file() or path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']:
            raise ValueError('Inference model weights do not match its manifest')
    return {**model,'weights':weights,'weights_path':weights[0]['path'],'weights_sha256':weights[0]['sha256'],
            'device':report.get('device'),'peak_memory_bytes':report.get('peak_process_family_rss_bytes_sampled')}


def prepare_model_workspace(report_path,output,*,source_metadata):
    """Save a separate named real CT/MRA case with cached binary model layers.

    A caller must supply attributable acquisition metadata; model identity does
    not establish where imaging came from or connect two patient identities.
    """
    report_path=Path(report_path);output=Path(output)
    report=json.loads(report_path.read_text())
    if report.get('schema')!='resectionlab-poc-inference-v1' or report.get('status')!='succeeded':
        raise ValueError('Only successful POC inference can become a named case')
    if not all(source_metadata.get(k) for k in ('case_id','source_url','source_license','source_name','source_image_sha256')):
        raise ValueError('An explicit patient acquisition identity and license are required')
    if source_metadata['case_id']!=report['case']:raise ValueError('Model output belongs to another patient')
    model=_checked_model(report);source=report['input'];image=Path(source['path'])
    if source_metadata['source_image_sha256']!=source['sha256']:raise ValueError('Source acquisition hash differs from model input')
    if file_sha256(image)!=source['sha256']:raise ValueError('Model source image changed')
    qc=inspect_nifti(image)
    if qc['shape']!=source['shape'] or not np.allclose(qc['affine_ras_mm'],source['affine_ras_mm'],rtol=0,atol=1e-6):
        raise ValueError('Inference input frame differs from acquired image')
    originals=[OriginalFile.from_path(image,role='imaging',expected_sha256=source['sha256']),
               OriginalFile.from_path(report_path,role='metadata')]
    masks={};layers={};refs=[]
    colors={'skull':'#d8e7f8','brain':'#65bfff','vessels':'#ff597c'}
    for row in report['outputs']:
        mask_path=Path(row['path']);name=row['name'].removesuffix('.nii.gz').removesuffix('.nii').removesuffix('_mask')
        if name not in {'skull','brain','vessels','vessel'}:continue
        if file_sha256(mask_path)!=row['sha256']:raise ValueError('Saved inference output checksum changed')
        mask,mask_qc=_aligned_mask(mask_path,qc)
        if set(np.unique(mask))-{0,1} or not mask.any():raise ValueError('Expected nonempty binary anatomy mask')
        if name=='vessel':name='vessels'
        if name in masks:raise ValueError('Duplicate anatomy layer')
        masks[name]=mask.astype(bool)
        layers[name]={'name':name,'label':name.title()+' (estimated)','kind':name,'provenance':'estimated',
            'source_role':'cached_model_inference','source_sha256':row['sha256'],'source_uri':mask_path.resolve().as_uri(),
            'input_sha256':source['sha256'],'color':colors.get(name,'#65bfff'),'model':model,'cache_key':report['cache_key'],
            'inference_manifest_sha256':file_sha256(report_path),'runtime_seconds':report.get('elapsed_seconds'),
            'review_status':'model output; local anatomical review pending','coverage':'Model estimate; unlabelled voxels are unknown'}
        originals.append(OriginalFile.from_path(mask_path,role='model-result',expected_sha256=row['sha256']))
        refs.append(SourceRef('model_'+name,mask_path.resolve().as_uri(),sha256=row['sha256'],license=model['license'],native_frame='RAS+',provenance='estimated'))
    if not masks:raise ValueError('No supported binary model anatomy outputs')
    for path in source_metadata.get('metadata_paths',[]):originals.append(OriginalFile.from_path(path,role='metadata'))
    metadata={**{k:v for k,v in source_metadata.items() if k!='metadata_paths'},'patient_id':report['case'],'anatomy_workbench_only':True,
        'selected_modality':source['modality'],'acquisition_datetime':source_metadata.get('acquisition_datetime'),
        'anatomy_layers':layers,'model_identity':model,'segmentation_cache_policy':'saved_masks_only; no automatic inference',
        'poc_separate_patient':True,'annotation_review':'pending'}
    case=load_nifti_case(image,case_id=report['case'],source_url=source_metadata['source_url'],
        license=source_metadata['source_license'],metadata=metadata)
    case=replace(case,compartments=masks,source_compartments=masks,source_refs=(*case.source_refs,*refs))
    state=view_state(case,[])
    state['states']['primary']['viewer']={'layerColor':{name:colors.get(name,'#65bfff') for name in masks},
        'layerOpacity':{name:.18 if name=='brain' else .65 for name in masks}}
    output.parent.mkdir(parents=True,exist_ok=True)
    identity=save_workspace(case,output,sources=[],imaging=state,episode=None,selection=None,artifacts={},originals=originals)
    reopened,_,saved=load_workspace(output)
    if reopened.semantic_hash!=case.semantic_hash or saved['sessionHash']!=identity:
        raise RuntimeError('Model case changed across save/reopen')
    result={'case_id':case.case_id,'case_hash':case.semantic_hash,'session_hash':identity,'bundle':str(output.resolve()),
        'modalities':[source['modality']],'layers':list(layers),'original_files':len(originals),'reopened_identical':True,
        'reinference_performed':False,'source':source_metadata}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    catalog_path=output.parent/'catalog.json';catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else {'cases':[]}
    item={'id':case.case_id,'title':case.case_id+' · '+source['modality'],'path':str(output.resolve()),
        'modalities':[source['modality']],'layers':list(layers),'source':source_metadata['source_name'],
        'limitation':source_metadata.get('limitation','Separate source patient; estimated anatomy has not been clinically reviewed')}
    catalog['cases']=[row for row in catalog.get('cases',[]) if row.get('id')!=case.case_id]+[item]
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
    return result


def _catalog(output,report):
    catalog_path=output.parent/'catalog.json'
    catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else {'cases':[]}
    item={'id':UCSF_CASE,'title':f'{UCSF_CASE} · Glioma MRI','path':str(output.resolve()),
        'modalities':report['modalities'],'layers':report['layers'],'source':'UCSF-PDGM / pinned MedOtter structural mirror · CC BY 4.0',
        'limitation':'Preprocessed MRI. No CT, angiography, raw DICOM or acquisition timestamp; source tumor labels are imported.'}
    catalog['cases']=[row for row in catalog.get('cases',[]) if row.get('id')!=UCSF_CASE]+[item]
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
