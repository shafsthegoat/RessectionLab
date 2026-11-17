#!/usr/bin/env python3
"""Build private EDEN2020 baseline and explicitly synthetic native projects.

Run from the primary checkout. Bulk data, inference, projects and QA stay on
its configured external data volume. This is engineering scenario generation;
no clinical distribution, tumor biology, growth, or mass effect is asserted.
"""
from dataclasses import replace
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zipfile import ZipFile
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from resectionlab.core import SourceRef, array_digest, thaw_json
from resectionlab.data_root import checked_root, atomic_json, sha256_file
from resectionlab.data_synthesis import geometry, appearance
from resectionlab.imaging import load_nifti_case, inspect_nifti, ImagingError
from resectionlab.poc_models import run as infer
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import OriginalFile, save_workspace, load_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource, MAX_IMAGE_BYTES

SOURCE='EDEN2020 healthy volunteers, Zenodo 3994749'
URL='https://zenodo.org/records/3994749'
LICENSE='CC-BY-NC-ND-4.0; private noncommercial adaptations; no derivative redistribution'
LIMITS=['Synthetic tumor and edema, not observed disease or clinically validated anatomy.',
    'Brain extraction and high-T1 placement support are algorithmic estimates, not reviewed tissue classes.',
    'Random intracerebral placement is not a clinical incidence distribution or an operability judgment.',
    'No simulated mass effect or displacement; surrounding acquired anatomy remains baseline.',
    'No skull CT, reviewed vessel masks or functional mapping; no safe surgical corridor established.',
    'Other sequences, diffusion and supplied tractography remain baseline references in their source grids.',
    'Private noncommercial use only; retain source attribution and no-derivative-sharing restriction.']


def unpack(base, subject):
    archive=base/'archives'/(subject+'.zip')
    receipt=json.loads((base/'provenance'/(subject+'.zip.json')).read_text())
    if sha256_file(archive)!=receipt['sha256']:raise ValueError('Source archive changed')
    destination=base/'extracted'/subject
    inventory=base/'provenance'/(subject+'-extraction.json')
    if inventory.exists():
        record=json.loads(inventory.read_text())
        for row in record['files']:
            p=destination/row['path']
            if not p.is_file() or p.stat().st_size!=row['bytes']:raise ValueError('Extracted source missing or changed')
        return destination,record
    destination.mkdir(parents=True,exist_ok=True)
    files=[]
    with ZipFile(archive) as z:
        for member in z.infolist():
            relative=Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts or (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unexpected unsafe ZIP member')
            if member.is_dir():continue
            p=destination/relative;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                # Resume a complete member only after independently comparing
                # its streamed digest with the checksummed ZIP's member bytes.
                with z.open(member) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if p.stat().st_size!=member.file_size or sha256_file(p)!=digest:raise ValueError('Preserve conflicting extracted source')
            else:
                temp=p.with_name(p.name+'.partial')
                with z.open(member) as stream,temp.open('wb') as output:shutil.copyfileobj(stream,output,1024**2)
                temp.replace(p);digest=sha256_file(p)
            files.append({'path':str(relative),'bytes':member.file_size,'sha256':digest,'zip_crc32':member.CRC})
    record={'archive_sha256':receipt['sha256'],'files':files,'source_modified':False}
    atomic_json(inventory,record)
    return destination,record


def source_path(directory, subject, suffix):
    matches=list(directory.rglob(subject+'_'+suffix+'.nii.gz'))
    if len(matches)!=1:raise ValueError(f'Expected one {subject} {suffix}, found {len(matches)}')
    return matches[0]


def support_for(case):
    brain=case.brain_mask
    cutoff=float(np.percentile(case.mri[brain],20))
    smooth=gaussian_filter(np.asarray(case.mri,np.float32),sigma=1./np.asarray(case.spacing_mm))
    support=brain & (smooth>cutoff)
    # Broadcast coordinates, retaining the native oblique/reflected source frame.
    axes=np.ogrid[tuple(slice(n) for n in brain.shape)]
    x=sum(case.affine[0,i]*axes[i] for i in range(3))+case.affine[0,3]
    z=sum(case.affine[2,i]*axes[i] for i in range(3))+case.affine[2,3]
    coordinates=np.argwhere(brain)@case.affine[:3,:3].T+case.affine[:3,3]
    low,high=coordinates.min(0),coordinates.max(0);mid=(low[0]+high[0])/2
    support &= z > low[2]+.38*(high[2]-low[2])
    del coordinates,smooth
    hemispheres=[support & (x<mid-5.),support & (x>mid+5.)]
    return hemispheres,{'method':'SynthStrip envelope intersected with source T1 p20 cutoff after 1mm smoothing, upper 62% of brain RAS-z extent and 5mm midline exclusion',
        'intensity_cutoff':cutoff,'brain_ras_min_mm':low.tolist(),'brain_ras_max_mm':high.tolist(),
        'midline_proxy_ras_x_mm':float(mid),'clinical_tissue_segmentation':False,
        'named_lobe_assignment':'unreviewed; use physical coordinates and source views'}


def save_masks(path,masks,affine):
    labels=np.zeros(next(iter(masks.values())).shape,np.uint8)
    for index,mask in enumerate(masks.values(),1):labels[mask]=index
    image=nib.Nifti1Image(labels,affine);image.header.set_xyzt_units('mm')
    image.set_qform(affine,1);image.set_sform(affine,1);nib.save(image,path)


def contact_sheet(case,baseline,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=nib.as_closest_canonical(nib.Nifti1Image(case.mri,case.affine))
    acquired=np.asarray(nib.as_closest_canonical(nib.Nifti1Image(baseline,case.affine)).dataobj)
    masks=[np.asarray(nib.as_closest_canonical(nib.Nifti1Image(m.astype(np.uint8),case.affine)).dataobj)>0 for m in case.compartments.values()]
    union=np.logical_or.reduce(masks) if masks else np.asarray(nib.as_closest_canonical(nib.Nifti1Image(case.brain_mask.astype(np.uint8),case.affine)).dataobj)>0
    center=np.rint(np.argwhere(union).mean(0)).astype(int)
    low,high=np.percentile(acquired[acquired>0],[2,98]);spacing=primary.header.get_zooms()[:3]
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for axis in range(3):
        kept=[a for a in range(3) if a!=axis];extent=[0,acquired.shape[kept[0]]*spacing[kept[0]],0,acquired.shape[kept[1]]*spacing[kept[1]]]
        for row,values in enumerate([acquired,np.asarray(primary.dataobj)]):
            ax=axes[row,axis];ax.imshow(np.take(values,center[axis],axis=axis).T,origin='lower',cmap='gray',vmin=low,vmax=high,extent=extent)
            if row:
                for mask,color in zip(masks,['#ff7eb6','#ffc65b','#6ed5ff']):
                    plane=np.take(mask,center[axis],axis=axis).T
                    if plane.any():ax.contour(plane,levels=[.5],colors=[color],linewidths=.7,extent=extent)
            ax.set_title(('Acquired baseline / ' if row==0 else 'Synthetic scenario / ')+['sagittal','coronal','axial'][axis]);ax.set_xlabel('mm along RAS axis')
    fig.suptitle(case.case_id+'\nPink core / gold rim / blue modeled edema; private engineering simulation; no mass effect',fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=115);plt.close(fig)


def package(root,case,sources,originals,baseline,output,recipe=None):
    record_path=output/'projects'/(case.case_id+'.json')
    category='synthetic' if case.compartments else 'controls'
    destination=root/'cases'/category/(case.case_id+'.ressectionlab')
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        if sha256_file(destination)!=previous['sha256'] or previous['case_hash']!=case.semantic_hash:
            raise ValueError('Existing project differs; choose a new version')
        return previous
    if destination.exists():raise ValueError('Unreceipted project exists; preserve it for review')
    state=view_state(case,sources)
    if case.compartments:
        center=np.asarray(recipe['center_ras_mm']);state['states']['primary']['cursor']=center.tolist()
    for view in state['states'].values():
        if 'brain' in view.get('visibleLayers',{}):view['visibleLayers']['brain']=False
    started=time.monotonic()
    session=save_workspace(case,destination,sources=sources,imaging=state,episode=None,selection=None,artifacts={},originals=originals)
    reopened,_,saved=load_workspace(destination)
    if reopened.semantic_hash!=case.semantic_hash or session!=saved['sessionHash']:raise ValueError('Production save/reopen failed')
    if [r.sha256 for r in saved['originals']]!=[r.sha256 for r in originals]:raise ValueError('Original bytes not preserved')
    if not np.array_equal(reopened.affine,case.affine):raise ValueError('Source geometry changed')
    if not np.array_equal(reopened.mri,case.mri):raise ValueError('Image roundtrip changed')
    scene=from_case(reopened)
    if case.compartments:
        union=np.logical_or.reduce(list(case.compartments.values()))
        if np.any(union&~case.brain_mask):raise ValueError('Tumor escapes brain estimate')
        if not np.array_equal(case.mri[~union],baseline[~union]):raise ValueError('Background changed outside lesion')
        if not reopened.metadata['is_synthetic']:raise ValueError('Synthetic flag lost')
        save_masks(output/'labels'/(case.case_id+'.nii.gz'),case.compartments,case.affine)
    sheet=output/'qc'/(case.case_id+'.png');contact_sheet(case,baseline,sheet)
    record={'case_id':case.case_id,'person_id':case.metadata['patient_group'],'path':str(destination.relative_to(root)),
        'sha256':sha256_file(destination),'bytes':destination.stat().st_size,'case_hash':case.semantic_hash,'session_hash':session,
        'synthetic':bool(case.compartments),'role':'DEVELOPMENT','clinical_validation':False,'review_status':'automated_engineering_checks_only',
        'production_save_reopen':'passed','source_originals_preserved':True,'native_affine_preserved':True,
        'scene_initialization':'passed','scene_region_count':len(scene.regions),'seconds':time.monotonic()-started,
        'contact_sheet':str(sheet.relative_to(root)),'recipe':recipe,
        'compartments':{name:{'voxels':int(mask.sum()),'volume_mm3':float(mask.sum()*case.voxel_volume_mm3)} for name,mask in case.compartments.items()}}
    atomic_json(record_path,record)
    print(json.dumps({'project':case.case_id,'status':'verified_native_project','seconds':record['seconds']}),flush=True)
    del reopened,saved,scene;gc.collect()
    return record


def build_subject(root,subject):
    base=root/'sources/eden2020-healthy-3994749';out=root/'prepared/eden2020-synthetic-v1'
    for name in ['projects','qc','labels','hosts','logs']:(out/name).mkdir(parents=True,exist_ok=True)
    directory,inventory=unpack(base,subject)
    t1=source_path(directory,subject,'T1_3D_PROSET_Sag');inspect_nifti(t1)
    os.environ['RESSECTIONLAB_DATA_ROOT']=str(root);os.environ['RESSECTIONLAB_MODEL_ROOT']=str(root/'cache/models')
    extraction=infer(t1,'EDEN2020-'+subject,'synthstrip-main','T1',timeout=1200,output_root=root/'cache/inference')
    brain=Path(next(row['path'] for row in extraction['outputs'] if row['name']=='brain_mask.nii.gz'))
    data={'schema':'ressectionlab-data-case-v1','case_id':'EDEN2020-'+subject+'-BASELINE-v1','person_id':'EDEN2020:'+subject,
        'source_subject':subject,'classification':'real_control','diagnosis':'healthy_control','role':'DEVELOPMENT',
        'exposure':'synthetic scenario development; not reserved assessment','source':SOURCE,'source_url':URL,'license':LICENSE,
        'source_version':'10.5281/zenodo.3994749','source_archive_sha256':inventory['archive_sha256'],
        'components':{'T1':'acquired_native','brain':'estimated_SynthStrip','tumor':'absent_healthy_baseline','skull':'missing','vessels':'missing_annotation','function':'missing'},
        'missing_anatomy':['measured_skull','reviewed_vessel_labels','functional_mapping'],
        'limitations':LIMITS,'clinical_validation':False,'review_status':'automated_engineering_checks_only'}
    metadata={'data_case':data,'patient_group':'EDEN2020:'+subject,'patient_id':subject,'source_license':LICENSE,
        'split':{'role':'DEVELOPMENT'},'training_admitted':False,'selected_modality':'T1 (ACQUIRED HEALTHY BASELINE)',
        'anatomy_layers':{'brain':{'kind':'brain','label':'Brain · SynthStrip estimate','provenance':'estimated','color':'#96b7d4'}},
        'is_synthetic':False,'anatomy_workbench_only':False}
    host=load_nifti_case(t1,brain_mask_path=brain,case_id=data['case_id'],license=LICENSE,source_url=URL,metadata=metadata)
    host=replace(host,source_refs=tuple(replace(r,provenance='estimated') if r.source_id=='supplied_brain_mask' else r for r in host.source_refs))
    originals=[OriginalFile.from_path(t1,role='imaging'),OriginalFile.from_path(brain,role='model-result'),
        OriginalFile.from_path(base/'provenance/zenodo-record.json',role='metadata'),
        OriginalFile.from_path(base/'provenance/development-plan.json',role='metadata')]
    aux=[]
    # Preserve full ancillary data in the source collection; native baseline
    # projects include these useful 3D source sequences on their separate grids.
    for suffix,modality in [('3D_FLAIR_Tra','FLAIR'),('s3DI_MC_HR','MRA'),('SWIp_axial','SWI')]:
        p=source_path(directory,subject,suffix)
        try:inspect_nifti(p)
        except ImagingError as error:
            data.setdefault('withheld_reference_channels',[]).append({'modality':modality,'original_path':str(p.relative_to(root)),
                'original_sha256':sha256_file(p),'reason':str(error),'source_original_preserved':True})
            originals.append(OriginalFile.from_path(p,role='imaging'))
            continue
        display_path=p
        original_header=nib.load(p)
        if np.prod(original_header.shape)*4>MAX_IMAGE_BYTES:
            from nibabel.processing import resample_to_output
            display_path=out/'hosts'/(subject+'-'+modality+'-display-1mm.nii.gz')
            if not display_path.exists():
                derivative=resample_to_output(original_header,voxel_sizes=(1.,1.,1.),order=1)
                derivative.set_data_dtype(np.float32);derivative.header.set_xyzt_units('mm')
                nib.save(derivative,display_path)
                atomic_json(display_path.with_suffix('.json'),{'source_path':str(p),'source_sha256':sha256_file(p),
                    'derivative_sha256':sha256_file(display_path),'operation':'1mm RAS display LOD, linear interpolation',
                    'registration_performed':False,'native_original_preserved':True})
                del derivative;gc.collect()
        observation=load_nifti_case(display_path,case_id=subject+'-'+modality,license=LICENSE,source_url=URL,
            metadata={'selected_modality':modality,'display_label':'Acquired baseline '+modality+(' · 1mm display LOD, original retained' if display_path!=p else ' · native grid')+'; registration unreviewed','is_synthetic':False,'observation_role':'baseline_native_reference'})
        aux.append(DisplaySource(observation,modality,'none'))
        originals.append(OriginalFile.from_path(p,role='imaging'))
    host=replace(host,metadata={**thaw_json(host.metadata),'data_case':data})
    records=[package(root,host,aux,originals,host.mri,out)]
    hemispheres,support_recipe=support_for(host)
    prior_centers=[]; distance_cache={}
    # The same person and all eight descendants retain one family/development
    # role. Alternate physical hemispheres, then sample admitted voxels uniformly.
    for variant in range(8):
        case_id='EDEN2020-'+subject+f'-SYNTHETIC-{variant+1:02d}-v1'
        previous_path=out/'projects'/(case_id+'.json')
        if previous_path.exists():
            previous=json.loads(previous_path.read_text())
            if sha256_file(root/previous['path'])!=previous['sha256']:
                raise ValueError('Completed project bytes changed')
            if previous['recipe']['placement_support']!=support_recipe:
                raise ValueError('Previously completed placement support changed')
            prior_centers.append(np.asarray(previous['recipe']['center_ras_mm']))
            records.append(previous);continue
        seed=int.from_bytes(hashlib.sha256(f'EDEN2020-healthy-3994749-synthetic-v1:{subject}:{variant}'.encode()).digest()[:4],'big')
        side=variant%2
        if side not in distance_cache:
            distance_cache[side]=distance_transform_edt(hemispheres[side],sampling=host.spacing_mm)
        distance=distance_cache[side]
        fit=min(1.,(float(distance.max())-1.6)/(13.*1.6))
        found=False
        for geometry_variant in dict.fromkeys([variant,0,1]):
            radius=10*(1.+.15*(geometry_variant%3))*fit
            candidates=np.argwhere(distance>=radius*1.6+1.5)
            if not len(candidates):continue
            for separation in [12.,8.,6.]:
                for attempt in range(20000):
                    rng=np.random.default_rng(seed+attempt)
                    voxel=candidates[rng.integers(len(candidates))]
                    center=host.affine[:3,:3]@voxel+host.affine[:3,3]
                    if all(np.linalg.norm(center-old)>=separation for old in prior_centers):
                        found=True;break
                if found:break
            if found:break
        if not found:raise ValueError('No distinct placement available; refuse duplicate')
        masks,recipe=geometry(host.brain_mask,host.affine,seed=seed+attempt,variant=geometry_variant,
            placement_support=hemispheres[side],random_placement=True)
        recipe['generation_slot']=variant
        recipe['size_adapted_for_distinct_placement']=geometry_variant!=variant
        if not np.allclose(recipe['center_ras_mm'],center):raise ValueError('Independent seeded center calculation differs')
        prior_centers.append(center)
        recipe.update(placement_support=support_recipe,hemisphere_proxy='RAS-left' if variant%2==0 else 'RAS-right',
            distinct_center_min_separation_mm=separation,base_seed=seed,rejection_draw=attempt,
            no_mass_effect=True,clinically_validated=False)
        image,signal=appearance(host.mri,host.brain_mask,masks,seed=seed)
        recipe['appearance']=signal
        case_id='EDEN2020-'+subject+f'-SYNTHETIC-{variant+1:02d}-v1'
        details=thaw_json(host.metadata)
        details.update(is_synthetic=True,selected_modality='T1 (SYNTHETIC TUMOR)',display_label='SYNTHETIC lesion on healthy baseline · no mass effect',
            benchmark_track='synthetic_engineering_development',
            surgical_adapter={'ownershipPrecedence':['brain','synthetic_edema','synthetic_rim','synthetic_core']})
        details['data_case'].update(case_id=case_id,classification='synthetic_scenario_on_acquired_host',
            host_case_id=host.case_id,host_case_hash=host.semantic_hash,recipe=recipe,
            synthetic_variant_count=1,new_real_patient_count=0,appearance=signal,deformation=None,
            observations={'primary_T1':'synthetic_appearance','reference_T1':'unchanged_acquired_healthy_baseline','ancillary_sequences':'retained_in_baseline_project_and_source_collection'},
            capabilities=['synthetic_geometry_and_T1_appearance_inspection','annotation_assisted_engineering_scene'],
            components={**details['data_case']['components'],'tumor':'generated_core_and_rim','edema':'generated_shell','T1':'synthetic_image'})
        for name,color in zip(masks,['#f78db3','#ffce70','#91d4fb']):
            details['anatomy_layers'][name]={'kind':'edema' if name.endswith('edema') else 'tumor','label':name.replace('_',' ').title()+' · GENERATED','provenance':'simulated','color':color}
        candidate=replace(host,case_id=case_id,mri=image,compartments=masks,source_compartments=masks,metadata=details,
            source_refs=(*host.source_refs,SourceRef('synthetic_scenario','urn:ressectionlab:'+case_id,array_digest(image),license=LICENSE,native_frame='RAS+',provenance='simulated')))
        reference=replace(host,brain_mask=None,source_refs=tuple(r for r in host.source_refs if r.source_id!='supplied_brain_mask'),
            metadata={**thaw_json(host.metadata),'observation_role':'baseline_reference_not_tumor_present','display_label':'Original healthy baseline T1 · before synthetic insertion'})
        # Keep each variant compact and self-contained. All 3D baseline channels
        # remain available in that volunteer's separate baseline project.
        variant_originals=[originals[0],originals[1],originals[2],originals[3]]
        records.append(package(root,candidate,[DisplaySource(reference,'T1','none')],variant_originals,host.mri,out,recipe))
        del candidate,reference,image,masks;gc.collect()
    atomic_json(out/'hosts'/(subject+'.json'),{'subject':subject,'complete':True,'projects':records})
    return records


def run(root=None,watch=False,subject=None):
    root=checked_root(root,required_bytes=40*1024**3)
    out=root/'prepared/eden2020-synthetic-v1';base=root/'sources/eden2020-healthy-3994749'
    plan=json.loads((base/'provenance/development-plan.json').read_text())
    subjects=[subject] if subject else plan['subjects']
    if any(s not in plan['subjects'] for s in subjects):raise ValueError('Unknown subject')
    while True:
        missing=[]
        for s in subjects:
            complete=out/'hosts'/(s+'.json')
            if complete.exists():continue
            if not (base/'provenance'/(s+'.zip.json')).exists():missing.append(s);continue
            print('building '+s,flush=True)
            build_subject(root,s)
        if not missing:break
        if not watch:break
        print('waiting for verified archives: '+','.join(missing),flush=True);time.sleep(20)
    host_files=[out/'hosts'/(s+'.json') for s in plan['subjects'] if (out/'hosts'/(s+'.json')).exists()]
    projects=[p for f in host_files for p in json.loads(f.read_text())['projects']]
    atomic_json(out/'collection.json',{'source':SOURCE,'source_url':URL,'license':LICENSE,'role':'DEVELOPMENT',
        'clinical_validation':False,'independent_clinical_review':False,'volunteer_count':len(host_files),
        'synthetic_project_count':sum(p['synthetic'] for p in projects),'baseline_project_count':sum(not p['synthetic'] for p in projects),
        'projects':projects,'limitations':LIMITS,'complete':len(host_files)==15})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--watch',action='store_true');p.add_argument('--subject')
    a=p.parse_args();run(a.root,a.watch,a.subject)
