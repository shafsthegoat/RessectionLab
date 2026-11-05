#!/usr/bin/env python3
"""Prepare reviewed-source healthy candidates and deterministic native scenarios.

Candidates remain in staging until a separate independent release decision.
The native format is reused unchanged, with versioned data_case metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import shutil
import struct
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from resectionlab.core import SourceRef, array_digest, thaw_json
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.data_synthesis import appearance, geometry, mild_deformation
from resectionlab.imaging import ImagingError, inspect_nifti, load_nifti_case
from resectionlab.poc_models import run as infer
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource

REPO = Path(__file__).resolve().parents[1]
VERSION = 'v5'


def precision_header_derivative(root, source):
    """Source-specific NIfTI1 quaternion-rounding repair, never registration."""
    from itertools import product
    image=nib.load(source);q,qcode=image.get_qform(coded=True);s,scode=image.get_sform(coded=True)
    if not isinstance(image,nib.Nifti1Image) or not qcode or qcode!=scode:
        raise ValueError('Header repair requires matching active NIfTI1 frame codes')
    reproduction=image.header.copy();reproduction.set_qform(s,int(qcode))
    fields=['quatern_b','quatern_c','quatern_d','qoffset_x','qoffset_y','qoffset_z']
    if not all(image.header[key].tobytes()==reproduction[key].tobytes() for key in fields):
        raise ValueError('Source qform is not exactly the float32 encoding of its own sform')
    corners=np.array(list(product(*[(0,n-1) for n in image.shape])))
    residual=float(np.linalg.norm(corners@(q[:3,:3]-s[:3,:3]).T+q[:3,3]-s[:3,3],axis=1).max())
    spacing=np.linalg.norm(s[:3,:3],axis=0)
    if residual>.05 or residual>.1*spacing.min():
        raise ValueError('Source-specific precision repair exceeds prospective subvoxel bound')
    directory=root/'prepared/header-precision';directory.mkdir(exist_ok=True)
    destination=directory/source.name
    receipt=destination.with_suffix('.header-repair.json')
    if destination.exists() and not receipt.exists():
        quarantine=root/'quarantine/header-precision-failed-roundtrip';quarantine.mkdir(exist_ok=True)
        retired=quarantine/(sha256_file(destination)+'-'+source.name)
        if retired.exists():
            if sha256_file(retired)!=sha256_file(destination):raise ValueError('Quarantine collision')
        else:shutil.copyfile(destination,retired)
        atomic_json(retired.with_suffix('.reason.json'),{'reason':'Unreceipted header derivative from failed intensity roundtrip; never accepted','sha256':sha256_file(retired)})
        destination.unlink()
    if not destination.exists():
        # Preserve exact on-disk scaling, extension blocks and voxel bytes.
        # Re-encoding scaled arrays through nibabel can quantize observed values.
        temporary=destination.with_name(destination.name+'.partial')
        with gzip.open(source,'rb') as original,gzip.open(temporary,'wb') as output:
            header=bytearray(original.read(352))
            if len(header)!=352 or struct.unpack(image.header.endianness+'i',header[:4])[0]!=348:raise ValueError('Expected NIfTI1 header')
            header[252:254]=struct.pack(image.header.endianness+'h',0)
            output.write(header);shutil.copyfileobj(original,output,1024*1024)
        os.replace(temporary,destination)
        reopened=nib.load(destination)
        if not np.array_equal(np.asanyarray(reopened.dataobj),np.asanyarray(image.dataobj)) or not np.array_equal(reopened.get_sform(),s):
            raise ValueError('Header-only derivative altered observed voxel values or source sform')
        atomic_json(receipt,{'schema':'same-source-quaternion-precision-repair-v1','source_path':str(source.relative_to(root)),
            'source_sha256':sha256_file(source),'derivative_path':str(destination.relative_to(root)),
            'derivative_sha256':sha256_file(destination),'qform_ras_mm':q.tolist(),'sform_ras_mm':s.tolist(),
            'max_corner_disagreement_mm':residual,'exact_float32_quaternion_reproduction':True,
            'operation':'disable qform code; retain same-source sform and exact observed voxel values',
            'registration_performed':False,'source_modified':False,'independent_review_required':True,
            'recipe_review':'qc/independent/IXI016-qform-sform-review-v1.json'})
    else:
        record=json.loads(receipt.read_text())
        if record['source_sha256']!=sha256_file(source) or record['derivative_sha256']!=sha256_file(destination):raise ValueError('Header derivative binding changed')
    return destination,receipt


def save_image(path, data, reference):
    image = nib.Nifti1Image(data, reference.affine, reference.header.copy())
    image.set_data_dtype(data.dtype)
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(image, path)


def contact_sheet(case, baseline, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Canonical reorientation is used for display only; package geometry stays
    # exactly native. Labels follow the identical array-axis operation.
    current = nib.as_closest_canonical(nib.Nifti1Image(case.mri, case.affine))
    source = nib.as_closest_canonical(nib.Nifti1Image(baseline, case.affine))
    image = np.asarray(current.dataobj); old = np.asarray(source.dataobj)
    masks = {name: np.asarray(nib.as_closest_canonical(nib.Nifti1Image(mask.astype(np.uint8), case.affine)).dataobj)
             for name, mask in case.compartments.items()}
    union = np.logical_or.reduce(list(masks.values())) if masks else np.asarray(nib.as_closest_canonical(nib.Nifti1Image(case.brain_mask.astype(np.uint8), case.affine)).dataobj)>0
    center = np.rint(np.argwhere(union).mean(0)).astype(int)
    values = old[old != 0]; low, high = np.percentile(values, [2,98])
    colors = ['#f78db3','#ffce70','#91d4fb']
    fig, axes = plt.subplots(2,3,figsize=(12,8),facecolor='white')
    spacing = current.header.get_zooms()[:3]
    for axis, name in enumerate(['sagittal RAS x','coronal RAS y','axial RAS z']):
        kept=[a for a in range(3) if a != axis]
        extent=[0,image.shape[kept[0]]*spacing[kept[0]],0,image.shape[kept[1]]*spacing[kept[1]]]
        for row, values in enumerate([old,image]):
            ax=axes[row,axis];ax.imshow(np.take(values,center[axis],axis=axis).T,origin='lower',cmap='gray',vmin=low,vmax=high,extent=extent)
            if row:
                for (label,mask),color in zip(masks.items(),colors):
                    plane=np.take(mask,center[axis],axis=axis).T
                    if plane.any():ax.contour(plane,levels=[.5],colors=[color],linewidths=.7,extent=extent)
            ax.set_title(('Baseline reference / ' if row==0 else 'Candidate / ')+name,fontsize=9)
            ax.set_xlabel('mm along positive RAS axis');ax.set_ylabel('mm')
    fig.suptitle(case.case_id+'\nSynthetic contours where present; no clinical validation',fontsize=11)
    fig.tight_layout();fig.savefig(destination,dpi=105);plt.close(fig)


def package(root, case, sources, originals, baseline, category, receipt):
    checked_root(root,required_bytes=512*1024**2)
    directory=root/'staging/candidates'/category;directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(case.case_id+'.ressectionlab')
    report_path=root/'catalog/preparation'/(case.case_id+'.json');report_path.parent.mkdir(exist_ok=True)
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if sha256_file(destination)!=report['sha256'] or report['case_hash']!=case.semantic_hash:raise ValueError('Previously prepared package or recipe changed; choose a new version')
        return report
    if destination.exists():raise ValueError('Unreceipted candidate exists; preserve and inspect before resuming')
    started=time.perf_counter()
    state=view_state(case,sources)
    if case.compartments:
        union=np.logical_or.reduce(list(case.compartments.values()))
        state['states']['primary']['cursor']=case.voxel_to_world(np.argwhere(union).mean(0)).tolist()
    # Keep expensive whole-head context surfaces off initially; MRI slices and
    # target remain available. No anatomical downsampling occurs.
    for item in state['states'].values():
        if 'brain' in item['visibleLayers']:item['visibleLayers']['brain']=False
    session=save_workspace(case,destination,sources=sources,imaging=state,episode=None,selection=None,artifacts={},originals=originals)
    load_started=time.perf_counter();reopened,_,saved=load_workspace(destination)
    load_seconds=time.perf_counter()-load_started
    if reopened.semantic_hash!=case.semantic_hash or saved['sessionHash']!=session:raise ValueError('Production reopen identity failed')
    if [x.sha256 for x in originals]!=[x.sha256 for x in saved['originals']]:raise ValueError('Original bytes changed')
    scene=from_case(reopened)
    sheet=root/'qc/contact-sheets'/(case.case_id+'.png');sheet.parent.mkdir(exist_ok=True)
    contact_sheet(case,baseline,sheet)
    report={**receipt,'case_id':case.case_id,'path':str(destination.relative_to(root)),'sha256':sha256_file(destination),
        'bytes':destination.stat().st_size,'case_hash':case.semantic_hash,'session_hash':session,'native_shape':list(case.mri.shape),
        'affine_ras_mm':case.affine.tolist(),'spacing_mm':list(case.spacing_mm),'source_images_immutable':True,
        'production_load_workspace':'passed','save_reopen_originals':'passed','production_load_seconds':load_seconds,
        'simulator_scene_initialization':'passed','simulator_regions':len(scene.regions),
        'contact_sheet':str(sheet.relative_to(root)),'prepare_seconds':time.perf_counter()-started,
        'review_status':'awaiting_independent_review','compartments':{name:{'voxels':int(mask.sum()),'volume_mm3':float(mask.sum()*case.voxel_volume_mm3)} for name,mask in case.compartments.items()}}
    atomic_json(report_path,report);print(json.dumps({'case_id':case.case_id,'path':str(destination),'status':'candidate_ready'}),flush=True)
    return report


def host(root, acquisition):
    if acquisition['role']!='TRAIN':raise ValueError('This preparation recipe is TRAIN development only')
    # Pin existing frozen role declaration again at use, before any decoding.
    cohort=REPO/acquisition['cohort_manifest']
    if sha256_file(cohort)!=acquisition['cohort_sha256']:raise ValueError('Frozen person roles changed')
    files=acquisition['acquired_files']
    paths={row['modality']:safe_path(root,row['path']) for row in files}
    for row in files:
        if sha256_file(safe_path(root,row['path']))!=row['sha256']:raise ValueError('Original source bytes changed')
    structural=paths.get('T1',paths.get('T1w'))
    if structural is None:raise ValueError('A structural T1 source is required')
    structural_derivatives=[]
    try:inspect_nifti(structural)
    except ImagingError as error:
        if error.code!='QFORM_SFORM_DISAGREEMENT':raise
        structural,repair_receipt=precision_header_derivative(root,structural)
        structural_derivatives=[OriginalFile.from_path(structural,role='imaging'),OriginalFile.from_path(repair_receipt,role='metadata')]
    name=acquisition['person_id'].replace(':','-')
    os.environ['RESSECTIONLAB_DATA_ROOT']=str(root)
    os.environ['RESSECTIONLAB_MODEL_ROOT']=str(root/'cache/models')
    record=infer(structural,name+'-data-rebuild','synthstrip-main','T1',output_root=root/'cache/inference',timeout=600)
    brain=Path(next(row['path'] for row in record['outputs'] if row['name']=='brain_mask.nii.gz'))
    provenance={'schema':'ressectionlab-data-case-v1','case_id':name+'-control-native-'+VERSION,'person_id':acquisition['person_id'],
        'source_subject':acquisition['subject'],'classification':'real_control','diagnosis':acquisition['diagnosis'],'role':'TRAIN',
        'exposure':'development source and anatomy review; not untouched evaluation','source':acquisition['dataset'],
        'source_url':acquisition['source_url'],'license':acquisition['license'],'source_version':acquisition.get('source_version',acquisition.get('cohort_sha256')),
        'source_files':files,'pairing':'same source subject and acquisition/session identifiers; temporal compatibility otherwise unknown',
        'components':{'T1':'acquired_image','MRA':'acquired_image_separate_native_grid','brain':'estimated_structure_SynthStrip',
                      'skull':'missing','arteries':'missing_annotation','veins':'missing','tumor':'missing_no_observed_target','tracts':'missing','function':'missing'},
        'observations':{'T1':'acquired_native','MRA':'acquired_separate_native_grid; no reviewed fusion'},
        'capabilities':['native_structural_and_angiographic_source_inspection','estimated_brain_geometry'],
        'planner_mode':'richer_anatomy_annotation_assisted; no limited-input performance claim',
        'allowed_planner_observations':['structural_T1','estimated_brain_mask'], 'private_reference_fields':[],
        'missing_anatomy':['measured_skull','reviewed_vessel_labels','tumor','tracts','functional_mapping'],
        'limitations':['No CT; no measured skull','MRA registration unreviewed; no vascular avoidance','Brain is model estimate; tissue classes and CSF boundaries unavailable'],
        'review_status':'awaiting_independent_review','clinical_validation':False}
    case=load_nifti_case(structural,brain_mask_path=brain,case_id=provenance['case_id'],license=acquisition['license'],source_url=acquisition['source_url'],
        metadata={'data_case':provenance,'patient_group':acquisition['person_id'],'patient_id':acquisition['subject'],
          'source_license':acquisition['license'],'split':{'role':'TRAIN'},'training_admitted':False,'selected_modality':'T1 (acquired healthy host)',
          'anatomy_layers':{'brain':{'kind':'brain','label':'Brain · SynthStrip estimate','provenance':'estimated','color':'#96b7d4'}},
          'anatomy_workbench_only':False})
    case=replace(case,source_refs=tuple(replace(ref,provenance='estimated') if ref.source_id=='supplied_brain_mask' else ref for ref in case.source_refs))
    sources=[];derivative_originals=list(structural_derivatives)
    for modality,path in paths.items():
        if modality in {'T1','T1w'}:continue
        try:
            display=load_nifti_case(path,case_id=acquisition['subject'],license=acquisition['license'],metadata={'observation_role':'acquired_baseline_native','data_case':provenance})
        except ImagingError as error:
            if error.code=='QFORM_SFORM_DISAGREEMENT':
                try:
                    derivative,receipt=precision_header_derivative(root,path)
                    display=load_nifti_case(derivative,case_id=acquisition['subject'],license=acquisition['license'],
                        metadata={'observation_role':'acquired_baseline_native','display_label':'Acquired MRA · baseline reference, header-only derivative',
                                  'data_case':provenance,'header_repair_receipt':str(receipt.relative_to(root))})
                    sources.append(DisplaySource(display,'MRA','none'))
                    derivative_originals.extend([OriginalFile.from_path(derivative,role='imaging'),OriginalFile.from_path(receipt,role='metadata')])
                    provenance.setdefault('header_only_derivatives',[]).append(json.loads(receipt.read_text()))
                    continue
                except ValueError as repair_error:
                    provenance.setdefault('header_repair_failures',[]).append(str(repair_error))
            # Do not weaken production header checks or silently copy another
            # modality's affine. Preserve the acquired original for review.
            provenance.setdefault('withheld_observations',[]).append({'modality':modality,'reason':str(error),'original_path':str(path.relative_to(root))})
            provenance['capabilities']=['native_structural_source_inspection','estimated_brain_geometry']
            provenance['observations']['MRA']='original retained; application display withheld for source-header review'
            continue
        sources.append(DisplaySource(display,'MRA','none'))
    case=replace(case,metadata={**thaw_json(case.metadata),'data_case':provenance})
    originals=[OriginalFile.from_path(safe_path(root,row['path']),role='imaging',expected_sha256=row['sha256']) for row in files]
    originals += [OriginalFile.from_path(brain,role='model-result'),*derivative_originals]
    return case,sources,originals


def run(root, *, subject=None, variants=4, mass_effect=False, max_hosts=None):
    root=checked_root(root)
    recipes=root/'prepared/recipes';recipes.mkdir(exist_ok=True)
    for code in [Path(__file__),REPO/'src/resectionlab/data_synthesis.py']:
        snapshot=recipes/(sha256_file(code)+'-'+code.name)
        if not snapshot.exists():snapshot.write_bytes(code.read_bytes())
    acquisition=json.loads((root/'catalog/source-acquisition.json').read_text())
    completed=[]
    for row in acquisition['candidates']:
        if row.get('status')!='bytes_and_header_validated_awaiting_independent_review':continue
        if subject and row['subject']!=subject:continue
        if max_hosts is not None and len(completed)>=max_hosts:break
        control,sources,originals=host(root,row)
        if max(control.spacing_mm)>2.5:
            details=thaw_json(control.metadata)
            details['data_case']['limitations'].append('Coarse anisotropic structural source: stress/source inspection only; default synthetic insertion withheld')
            control=replace(control,metadata=details)
        package(root,control,sources,originals,control.mri,'controls',{'classification':'real_control','person_id':row['person_id']})
        if max(control.spacing_mm)>2.5:
            completed.append(row['subject'])
            continue
        cutoff=float(np.percentile(control.mri[control.brain_mask],20))
        smooth=gaussian_filter(np.asarray(control.mri,np.float32),sigma=1./np.asarray(control.spacing_mm))
        placement_support=control.brain_mask & (smooth>cutoff)
        support_dir=root/'prepared/placement-support';support_dir.mkdir(exist_ok=True)
        support_path=support_dir/(row['subject']+'-high-T1-support-v1.nii.gz')
        if not support_path.exists():save_image(support_path,placement_support.astype(np.uint8),nib.Nifti1Image(control.mri,control.affine))
        selection={'method':'source-brain p20 intensity cutoff after 1mm Gaussian smoothing; conservative high-T1-signal proxy',
            'cutoff_source_intensity':cutoff,'smoothing_sigma_mm':1.,'mask_path':str(support_path.relative_to(root)),
            'mask_sha256':sha256_file(support_path),'source_image_array_hash':array_digest(control.mri),
            'anatomical_tissue_classes_validated':False,'reason':'independent v3 ventricular/CSF-like overlap rejection'}
        originals=[*originals,OriginalFile.from_path(support_path,role='model-result')]
        for variant in range(variants):
            seed=int.from_bytes(__import__('hashlib').sha256(f"data-rebuild-v1:{row['person_id']}:{variant}".encode()).digest()[:4],'big')
            masks,recipe=geometry(control.brain_mask,control.affine,seed=seed,variant=variant,placement_support=placement_support)
            recipe['placement_support']=selection
            recipe['software']={'builder_sha256':sha256_file(Path(__file__)),
                'synthesis_sha256':sha256_file(REPO/'src/resectionlab/data_synthesis.py'),
                'numpy':np.__version__,'nibabel':nib.__version__}
            stage='geometry' if variant%2==0 else 'appearance'
            image=control.mri;brain=control.brain_mask
            imaging=None;deformation=None
            if stage=='appearance':image,imaging=appearance(image,brain,masks,seed=seed)
            if mass_effect and variant==variants-1:
                image,brain,masks,field,deformation=mild_deformation(image,brain,control.affine,masks,center_ras_mm=recipe['center_ras_mm'])
                stage='mass-effect'
                field_dir=root/'prepared/synthetic-fields';field_dir.mkdir(exist_ok=True)
                field_path=field_dir/(row['subject']+'-'+VERSION+'-mild-backward-field.npz')
                if not field_path.exists():np.savez_compressed(field_path,displacement_ras_mm=field,affine_ras_mm=control.affine)
                else:
                    with np.load(field_path,allow_pickle=False) as saved_field:
                        if not np.array_equal(saved_field['displacement_ras_mm'],field):raise ValueError('Prior displacement differs; use a new recipe version')
                deformation['field_path']=str(field_path.relative_to(root));deformation['field_sha256']=sha256_file(field_path)
            case_id=row['person_id'].replace(':','-')+f'-SYNTHETIC-{stage}-{variant:02d}-'+VERSION
            details=thaw_json(control.metadata);data_case=details['data_case']
            data_case.update(case_id=case_id,classification='synthetic_scenario_on_acquired_host',host_case_id=control.case_id,
                host_case_hash=control.semantic_hash,recipe=recipe,appearance=imaging,deformation=deformation,
                new_real_patient_count=0,synthetic_variant_count=1,
                observations={'primary_T1':'baseline_acquired_background_geometry_only' if stage=='geometry' else 'synthetic_appearance_and_deformation' if stage=='mass-effect' else 'synthetic_appearance',
                              'auxiliary_T1_MRA':'baseline_reference_only; not contemporaneous tumor-present acquisitions'},
                capabilities=['synthetic_target_spatial_planning','bounded_instrument_geometry']+([] if imaging is None else ['synthetic_T1_appearance'])+([] if deformation is None else ['bounded_engineering_deformation']),
                allowed_planner_observations=['primary_T1','estimated_or_deformed_brain_mask','synthetic_target_labels'],
                limitations=data_case['limitations']+['Synthetic lesion; not an additional real patient','No calibrated growth, tissue response, bleeding, neurological outcome or clinical realism'])
            data_case['components'].update(tumor='synthetic_structure',edema='synthetic_structure',T1='acquired_baseline_background' if stage=='geometry' else 'synthetic_image')
            if deformation:data_case['components']['brain']='synthetically_displaced_model_estimate'
            details['selected_modality']='T1 (BASELINE + SYNTHETIC TARGET)' if stage=='geometry' else 'T1 (SYNTHETIC '+stage.upper()+')'
            details['is_synthetic']=True
            details['anatomy_layers'].update({name:{'kind':'edema' if name.endswith('edema') else 'tumor','label':name.replace('_',' ').title()+' · GENERATED',
                'provenance':'simulated','color':color} for name,color in zip(masks,['#f78db3','#ffce70','#91d4fb'])})
            details['surgical_adapter']={'ownershipPrecedence':['brain','synthetic_edema','synthetic_rim','synthetic_core']}
            refs=list(control.source_refs)
            refs.append(SourceRef(source_id='synthetic_scenario',uri='urn:ressectionlab:'+case_id,sha256=array_digest(image),license=row['license'],native_frame='RAS+',provenance='simulated'))
            candidate=replace(control,case_id=case_id,mri=image,brain_mask=brain,compartments=masks,source_compartments=masks,metadata=details,source_refs=tuple(refs))
            baseline=replace(control,brain_mask=None,source_refs=tuple(ref for ref in control.source_refs if ref.source_id!='supplied_brain_mask'),
                metadata={**thaw_json(control.metadata),'observation_role':'baseline_reference_not_tumor_present','display_label':'Baseline reference T1 · before synthetic insertion','selected_modality':'T1 (BASELINE REFERENCE)'} )
            # Baseline channels are retained as references; they are absent from
            # the primary surgical scene and cannot become stationary vessels.
            reference_sources=[DisplaySource(baseline,'T1','none'),*sources]
            package(root,candidate,reference_sources,originals,control.mri,'synthetic',
                {'classification':data_case['classification'],'person_id':row['person_id'],'stage':stage,'recipe':recipe,'appearance':imaging,'deformation':deformation})
        completed.append(row['subject'])
    atomic_json(root/'catalog/preparation/run.json',{'schema':'data-rebuild-preparation-run-v1','completed_hosts':completed,'variants_per_host':variants,'mass_effect_requested':mass_effect})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=None);parser.add_argument('--subject');parser.add_argument('--variants',type=int,default=4)
    parser.add_argument('--mass-effect',action='store_true');parser.add_argument('--max-hosts',type=int)
    args=parser.parse_args()
    if not 0<=args.variants<=16:parser.error('Use 0–16 variants per host')
    run(args.root,subject=args.subject,variants=args.variants,mass_effect=args.mass_effect,max_hosts=args.max_hosts)
