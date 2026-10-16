"""Bounded, resumable qualification of existing patient bundles.

Qualification means source/geometry/persistence engineering checks, never accepted
anatomical truth, clinical validity, training admission or a newly acquired person.
The default plan admits exactly the nine already inspected TRAIN/development
people. It reads only role/acquisition metadata before freezing that allowlist.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import csv
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
from uuid import uuid4
from zipfile import ZipFile

import numpy as np

from .core import array_digest, semantic_digest, thaw_json
from .imaging import file_sha256, inspect_nifti
from .structural_evidence import structural_frame_hash
from .workspace_bundle import _case_layout, _json, MAX_PRIMARY, MAX_ATTACHMENT_BYTES, extension_layout, load_workspace
from .workspace_imaging import source_registration

VERSION='patient-corpus-qualification-v1'
DEFAULT_RELEASE='existing-nine-v1'
IXI_RELEASE='existing-nine-plus-ixi265-v1'
RELEASES=(DEFAULT_RELEASE,IXI_RELEASE)
IXI_RELEASE_PATH='manifests/patient-corpus-releases/existing-nine-plus-ixi265-v1.json'
IXI_RELEASE_SHA256='2c426f82e160be5b11bb19d9d5c4d2ac804ae9d85011cf46a4dc83904adbea84'
IXI_PLAN_HASH='sha256:a1a2623d05270881d4291a1037d303578cb27a4f63d52cce28b5f5db251d8755'
UNKNOWN='UNKNOWN'
READABLE_ROLES={'TRAIN','DEVELOPMENT'}
ANATOMY=('brain','tumor','skull','vessels','ventricles')
MAX_BUNDLE_BYTES=512*1024**2
MAX_METADATA_BYTES=16*1024**2
BTC_TRAIN=('05','16','20','22','25','28')
ADMITTED_ROLES={**{'BTC:sub-PAT'+number:'TRAIN' for number in BTC_TRAIN},
    'UCSF-PDGM:004':'DEVELOPMENT','SynthRAD2023:1BA336':'TRAIN','Lausanne:sub-000':'TRAIN'}
ROLE_DOCUMENTS={
 'manifests/experiments/btc-spatial-development-cohort-v1.json':'962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985',
 'manifests/lausanne-component-cohort-v1.json':'891a53680edfce604eed9a1791bc790efa615db8ce54da0c92de834a6b87d49a',
 'manifests/synthrad-component-cohort-v1.json':'3bb4dd48e2e674bd921653d938ec7c7695c77afb8776c517800bbefb6a32ff30'}


class QualificationError(ValueError):
    def __init__(self,code,message):self.code=code;super().__init__(message)


def _require(condition,code,message):
    if not condition:raise QualificationError(code,message)


def canonical_patient(value):
    text=str(value)
    patterns=((r'UCSF[-:]PDGM[-:]0*(\d+)','UCSF-PDGM'),(r'BTC.*?(sub-PAT\d+)','BTC'),
        (r'Lausanne[-:](sub-\d+)','Lausanne'),(r'SynthRAD2023[-:](1B[A-Z]\d+)','SynthRAD2023'),
        (r'ReMIND[-:]0*(\d+)','ReMIND'),(r'IXI[-:]?0*(\d+)','IXI'))
    for pattern,family in patterns:
        match=re.search(pattern,text,re.I)
        if match:
            identifier=match.group(1)
            if family in {'UCSF-PDGM','ReMIND','IXI'}:identifier=f'{int(identifier):03d}'
            if family=='BTC':identifier='sub-PAT'+f"{int(re.search(r'\d+',identifier).group()):02d}"
            if family=='Lausanne':identifier=identifier.lower()
            return family+':'+identifier
    return None


def require_candidate(candidate,release=DEFAULT_RELEASE):
    """Do not stat/open a payload until its explicit participant/role passes."""
    patient=candidate.get('patient_group','');role=candidate.get('role')
    blocked=(re.search(r'upenn|utsw',patient,re.I) or patient in
        {'BTC:sub-PAT26','BTC:sub-PAT27','BTC:sub-PAT29','BTC:sub-PAT31','ReMIND:013','ReMIND:037'} or
        re.search(r'resect.*(?:case)?0*4(?:\D|$)',patient,re.I))
    _require(not blocked and role in READABLE_ROLES,'PROTECTED_ROLE','Protected or undeclared participant is not opened')
    _require(canonical_patient(patient)==patient,'PARTICIPANT_ID','A canonical source-family participant ID is required')
    _require(release in RELEASES,'RELEASE_UNKNOWN','Only explicitly frozen corpus releases are supported')
    if release==IXI_RELEASE:
        frozen,_=_frozen_release()
        admitted={row['patient_group']:row for row in [*frozen['base_candidates'],frozen['additional_candidate']]}
        _require(candidate==admitted.get(patient),'NOT_ALLOWLISTED','Candidate differs from the exact frozen release declaration')
    else:
        _require(ADMITTED_ROLES.get(patient)==role,'NOT_ALLOWLISTED','This participant and role are outside the frozen nine-person tranche')
    _require(candidate.get('existing_materialized') is True,'NEW_PARTICIPANT_NOT_ADMITTED','This tranche only reuses previously materialized people')
    allowed_licenses={'CC0','CC0-1.0','CC-BY-4.0','CC-BY-NC-4.0'}|({'CC-BY-SA-3.0'} if release==IXI_RELEASE else set())
    _require(candidate.get('rights',{}).get('license') in allowed_licenses,'RIGHTS_UNRESOLVED','Source rights must be explicitly established')
    _require(str(candidate['rights'].get('source','')).startswith('https://'),'RIGHTS_UNRESOLVED','Source rights require an attributed source')


def require_manifest(manifest,candidate,release=DEFAULT_RELEASE):
    require_candidate(candidate,release)
    _require(canonical_patient(manifest.get('case_id'))==candidate['patient_group'],'PATIENT_MISMATCH','Saved series belongs to another participant')
    metadata=manifest.get('metadata',{})
    _require(metadata.get('is_synthetic') is False,'NOT_ACQUIRED_IMAGING','Only explicitly acquired imaging is eligible')
    generated_fields={'virtual_glioma','virtual_geometry','generated_descendant'}
    _require(not any(key in metadata and metadata[key] is not False and metadata[key] is not None for key in generated_fields),
        'GENERATED_DESCENDANT','Generated descendants are not independent acquired people')
    _require(not any(str(ref.get('provenance','')).casefold() in {'simulated','generated'} for ref in manifest.get('source_refs',[])),
        'GENERATED_DESCENDANT','Simulated source artifacts are not acquired patient sources')
    for layer in metadata.get('anatomy_layers',{}).values():
        tokens=set(re.findall(r'[a-z]+',str(layer.get('provenance','')).casefold()))
        _require(not tokens&{'simulated','generated'},'GENERATED_DESCENDANT','Generated anatomy cannot be qualified as acquired patient anatomy')
    role_keys={'role','roles','source_role','split_role','development_role','outer_role','benchmark_role','cohort_role','data_role','usage_role','partition','split'}
    identity_keys={'case_id','patient_id','patient_group','person_group','participant_id','subject','canonical_person','host_patient_group'}
    forbidden={'select','selection','eval','evaluation','test','testing','val','validation','holdout','heldout','final','stress','restricted','unopened','transfer'}
    collection_keys={'source_collection','collection','dataset','source_dataset'}
    prose_keys={'reason','rationale','description','note','notes'}
    def visit(value,path=(),role_payload=False):
        if isinstance(value,Mapping):
            for key,item in value.items():
                key=re.sub(r'(?<=[a-z])(?=[A-Z])','_',str(key)).lower()
                tokens=set(re.findall(r'[a-z0-9]+',key))
                role_mapping=role_payload or bool(path and path[-1] in role_keys)
                if role_mapping and (tokens&forbidden or {'held','out'}<=tokens) and item is not False and item is not None:
                    raise QualificationError('PROTECTED_ROLE','Structured role metadata enables a protected role')
                if key in {'restricted','is_restricted','is_holdout'} and item is True:
                    raise QualificationError('PROTECTED_ROLE','Nested source metadata marks the case protected')
                collection_identity=key in collection_keys|{'collection_name','dataset_name'} or (path and path[-1] in collection_keys and key in {'name','id','identifier','title'})
                if (key in identity_keys or collection_identity) and isinstance(item,str):
                    identity=canonical_patient(item)
                    _require(identity is None or identity==candidate['patient_group'],'PATIENT_MISMATCH','Nested patient identity differs')
                    _require(not re.search(r'upenn|utsw',item,re.I),'PROTECTED_ROLE','External protected collection is excluded')
                declared=key in role_keys or (path and path[-1] in {'split','partition'} and key in {'name','value','code','label'})
                if key in {'split','partition'} and isinstance(item,Mapping):declared=False
                visit(item,(*path,key),declared or (role_payload and key not in prose_keys))
        elif isinstance(value,(list,tuple)):
            for item in value:visit(item,path,role_payload)
        elif isinstance(value,str) and role_payload:
            tokens=set(re.findall(r'[a-z0-9]+',value.casefold()))
            _require(not tokens&forbidden and not {'held','out'}<=tokens,'PROTECTED_ROLE','Nested split/role remains protected')
    visit(metadata)


def _relative(value):
    path=PurePosixPath(str(value))
    _require(not path.is_absolute() and '..' not in path.parts and str(path) not in {'','.'},'PATH_SCOPE','Paths must be repository relative')
    return str(path)


def _payload_path(root,relative,*,patient=None):
    path=root/_relative(relative)
    resolved=path.resolve(strict=False)
    text=str(resolved)
    blocked=re.search(r'upenn|utsw|sub[-_]PAT0*(?:26|27|29|31)(?!\d)|ReMIND[-_: ]0*(?:13|37)(?!\d)|RESECT.*Case0*4(?!\d)',text,re.I)
    _require(not blocked,'PROTECTED_PATH','Protected payload path is not opened')
    if patient is not None:
        _require(canonical_patient(resolved.name)==patient,'PATIENT_PATH','Bundle path does not identify the allowlisted participant')
    return path


def _metadata(root,relative,expected=None):
    relative=_relative(relative);path=root/relative
    _require(path.stat().st_size<=MAX_METADATA_BYTES,'METADATA_LIMIT','Metadata record exceeds limit')
    payload=path.read_bytes();digest=sha256(payload).hexdigest()
    _require(expected is None or expected==digest,'METADATA_HASH','Pinned metadata changed: '+relative)
    return _json(payload),{'path':relative,'sha256':digest,'bytes':len(payload)}


def default_plan(root):
    """Only metadata is touched here; no bundle, NIfTI, model or mask is opened."""
    root=Path(root);evidence=[];roles={};records={}
    for path,digest in ROLE_DOCUMENTS.items():
        document,binding=_metadata(root,path,digest);records[path]=document;evidence.append(binding)
    btc=records[next(path for path in records if 'btc-spatial' in path)]
    for row in btc['existing_development_records']:roles[row['patient_group']]='TRAIN'
    for row in btc['candidates']:roles[row['patient_group']]='TRAIN' if row['development_role']=='population_training' else 'SELECT' if 'selection' in row['development_role'] else 'UNOPENED_TRANSFER'
    for path,key in [('manifests/lausanne-component-cohort-v1.json','canonical_person'),('manifests/synthrad-component-cohort-v1.json','patient_group')]:
        for row in records[path]['members']:roles[row[key]]=row['role']
    candidates=[]
    def read(path):
        record,binding=_metadata(root,path);evidence.append(binding);return record
    for number in BTC_TRAIN:
        subject='sub-PAT'+number;patient='BTC:'+subject
        path=(f'data/diffusion_source/ds001226-v5.0.1/btc_{subject}_verified_manifest.json' if number in {'22','25'}
              else 'manifests/'+('btc_acquisition.json' if number=='28' else f'btc_pat{number}_acquisition.json'))
        acquisition=read(path)
        _require(acquisition['subject']==subject,'PARTICIPANT_ID','Acquisition subject differs from allowlist')
        sources=[{'path':'data/diffusion_source/ds001226-v5.0.1/'+row['path'],'sha256':row['sha256'],
                  'role':'annotation' if 'tumor_masks' in row['path'] else 'imaging'} for row in acquisition['files'] if '.nii' in row['path'] and ('/anat/' in row['path'])]
        participants=next(row for row in acquisition['files'] if row['path']=='participants.tsv')
        candidates.append({'patient_group':patient,'case_id':f'BTC-ds001226-{subject}-preop','role':roles[patient],
            'bundle':f'outputs/cases/BTC-{subject}-structural-evidence.ressectionlab','existing_materialized':True,
            'rights':{'license':acquisition['license'],'source':acquisition.get('primary_source',btc['source']['primary_source']),'use':'research'},
            'participant':{'path':'data/diffusion_source/ds001226-v5.0.1/participants.tsv','sha256':participants['sha256'],
                           'delimiter':'\t','key':'participant_id','value':subject},'source_files':sources,'acquisition_record':path})
    acquisition=read('manifests/data_acquisition_ucsf.json');ucsf_metadata=next(row for row in acquisition['metadata_files'] if row['path'].endswith('metadata_v5.csv'))
    candidates.append({'patient_group':'UCSF-PDGM:004','case_id':'UCSF-PDGM-0004','role':'DEVELOPMENT',
        'bundle':'outputs/poc-cases/UCSF-PDGM-0004.ressectionlab','existing_materialized':True,
        'rights':{'license':acquisition['license']['identifier'],'source':acquisition['source_collection']['official_url'],'use':'research; attribution required'},
        'participant':{'path':'data/'+ucsf_metadata['path'],'sha256':ucsf_metadata['sha256'],'delimiter':',','key':'ID','value':'UCSF-PDGM-004'},
        'source_files':[{'path':'data/'+row['path'],'sha256':row['sha256'],'role':'annotation' if row['modality']=='tumor_segmentation' else 'imaging'} for row in acquisition['files']],
        'acquisition_record':'manifests/data_acquisition_ucsf.json','known_aliases':['BraTS2021:BraTS2021_00097'],
        'overlap_caveat':'Known BraTS training overlap; not independent segmentation evaluation; mirror/official byte equivalence unverified'})
    for patient,filename,image in [('SynthRAD2023:1BA336','poc-synthrad-1ba336.json','data/qc/synthrad-task1-first-pair-v1/restricted/ct.nii.gz'),
            ('Lausanne:sub-000','poc-lausanne-sub-000.json','data/anatomy/ds003949-v1.0.1/sub-000/ses-20110101/anat/sub-000_ses-20110101_angio.nii.gz')]:
        metadata=read('manifests/'+filename)
        _require(canonical_patient(metadata['case_id'])==patient,'PARTICIPANT_ID','Named source case differs')
        candidates.append({'patient_group':patient,'case_id':metadata['case_id'],'role':roles[patient],
            'bundle':f"outputs/poc-cases/{metadata['case_id']}.ressectionlab",'existing_materialized':True,
            'rights':{'license':metadata['source_license'],'source':metadata['source_url'],'use':'noncommercial research' if 'NC' in metadata['source_license'] else 'research'},
            'source_files':[{'path':image,'sha256':metadata['source_image_sha256'],'role':'imaging'}],
            'participant':{'kind':'source_cohort_and_bound_acquisition_record','value':patient},'acquisition_record':'manifests/'+filename})
    for candidate in candidates:require_candidate(candidate)
    plan={'schema':VERSION+'.plan','scope':'existing_nine_named_people_only','metadata_evidence':evidence,
          'candidates':sorted(candidates,key=lambda row:row['patient_group']),'existing_materialized_people':9,
          'new_patient_acquisition_allowed':False,'protected_payloads_opened':False,
          'protected_roles':{'UPENN':'external_final','UTSW':'structural_validation','BTC26_27':'SELECT','BTC29_31':'UNOPENED_TRANSFER'}}
    plan['plan_hash']=semantic_digest(plan)
    return plan


def _frozen_release(root=None):
    return _metadata(Path(root) if root is not None else Path(__file__).resolve().parents[2],IXI_RELEASE_PATH,IXI_RELEASE_SHA256)


def release_plan(root,release=DEFAULT_RELEASE):
    """Fixed release selection only; no caller-supplied admission documents."""
    _require(release in RELEASES,'RELEASE_UNKNOWN','Only explicitly frozen corpus releases are supported')
    base=default_plan(root)
    if release==DEFAULT_RELEASE:return base
    frozen,release_binding=_frozen_release(root)
    _require(base['plan_hash']==frozen['base_plan_hash'],'BASE_RELEASE_CHANGED','The nine-person parent plan changed')
    expected_base=[{k:v for k,v in row.items() if k!='bundle_sha256'} for row in frozen['base_candidates']]
    _require(base['candidates']==expected_base,'BASE_RELEASE_CHANGED','The original nine declarations changed')
    for row in frozen['metadata_evidence']:_metadata(Path(root),row['path'],row['sha256'])
    cohort,_=_metadata(Path(root),frozen['additional_candidate']['participant']['cohort_path'],frozen['additional_candidate']['participant']['cohort_sha256'])
    people=[row for row in cohort['members'] if row['person_group']=='IXI:265']
    _require(len(people)==1 and people[0]['role']=='TRAIN' and people[0]['training_admitted'] is False,'PROTECTED_ROLE','IXI265 frozen role differs')
    base.pop('plan_hash')
    base.update(release_id=release,parent_plan_hash=frozen['base_plan_hash'],scope='frozen_nine_plus_existing_ixi265_only',
        candidates=sorted([*frozen['base_candidates'],frozen['additional_candidate']],key=lambda row:row['patient_group']),
        existing_materialized_people=10,newly_cataloged_existing_people=1,virtual_descendants_admitted=False)
    base['metadata_evidence'] += [release_binding,*frozen['metadata_evidence']]
    base['plan_hash']=semantic_digest(base);validate_plan(base)
    return base


def validate_plan(plan):
    _require(plan.get('schema')==VERSION+'.plan','PLAN_SCHEMA','Unknown qualification plan')
    _require(plan.get('plan_hash')==semantic_digest({k:v for k,v in plan.items() if k!='plan_hash'}),'PLAN_HASH','Qualification plan changed')
    candidates=plan.get('candidates',[])
    release=plan.get('release_id',DEFAULT_RELEASE)
    _require(release in RELEASES,'RELEASE_UNKNOWN','Only explicitly frozen corpus releases are supported')
    _require(isinstance(candidates,list) and 0<len(candidates)<=(10 if release==IXI_RELEASE else 9),'PLAN_LIMIT','Candidate count exceeds the frozen release')
    identities=[row.get('patient_group') for row in candidates]
    _require(len(identities)==len(set(identities)),'DUPLICATE_PATIENT','Multiple bundle representations cannot count as different people')
    if release==IXI_RELEASE:
        _require(plan['plan_hash']==IXI_PLAN_HASH,'RELEASE_PLAN_CHANGED','The fixed release plan cannot be edited or expanded')
        _require(set(identities)==set(ADMITTED_ROLES)|{'IXI:265'},'RELEASE_MEMBERS','The explicit release must contain exactly the frozen ten people')
        _require(plan.get('virtual_descendants_admitted') is False and plan.get('new_patient_acquisition_allowed') is False,'RELEASE_SCOPE','New acquisitions and descendants are not released')
    for row in candidates:
        require_candidate(row,release);_relative(row['bundle'])
        for source in row.get('source_files',[]):_relative(source['path'])


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.pending-',delete=False) as stream:
        temporary=Path(stream.name);json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _manifest_preflight(path,candidate,*,check_arrays=True,release=DEFAULT_RELEASE):
    """Reject wrong identity and protected nested roles before numeric decoding."""
    with ZipFile(path) as archive:
        _require(archive.getinfo('manifest.json').file_size<=MAX_METADATA_BYTES,'METADATA_LIMIT','Bundle manifest too large')
        manifest=_json(archive.read('manifest.json'));require_manifest(manifest,candidate,release)
        document=extension_layout(archive) if 'workspace.json' in archive.namelist() else None
        if document:
            for row in document['images']:
                with ZipFile(BytesIO(archive.read(row['member']))) as nested:
                    _require(nested.getinfo('manifest.json').file_size<=MAX_METADATA_BYTES,'METADATA_LIMIT','Series manifest too large')
                    require_manifest(_json(nested.read('manifest.json')),candidate,release)
                    if check_arrays:_case_layout(nested,MAX_ATTACHMENT_BYTES)
        if check_arrays:_case_layout(archive,MAX_PRIMARY,extension=document is not None)
    return manifest


def _input_binding(root,candidate,plan):
    release=plan.get('release_id',DEFAULT_RELEASE)
    require_candidate(candidate,release)
    bindings=[]
    for row in plan.get('metadata_evidence',[]):
        _,binding=_metadata(root,row['path'],row['sha256']);bindings.append(binding)
    bundle=_payload_path(root,candidate['bundle'],patient=candidate['patient_group'])
    _require(bundle.stat().st_size<=MAX_BUNDLE_BYTES,'BUNDLE_LIMIT','Bundle exceeds bounded qualification limit')
    if candidate.get('bundle_sha256'):
        _require(file_sha256(bundle)==candidate['bundle_sha256'],'BUNDLE_HASH','Bundle differs from the frozen release before decoding')
    _manifest_preflight(bundle,candidate,check_arrays=False,release=release)
    participant=candidate.get('participant',{})
    if 'path' in participant:
        path=root/_relative(participant['path']);digest=file_sha256(path)
        _require(digest==participant['sha256'],'PARTICIPANT_HASH','Participant metadata changed')
        with path.open(newline='',encoding='utf-8-sig') as stream:
            rows=[row for row in csv.DictReader(stream,delimiter=participant['delimiter']) if row.get(participant['key'])==participant['value']]
        _require(len(rows)==1,'PARTICIPANT_ID','Expected exactly one matching source participant')
        bindings.append({'path':participant['path'],'sha256':digest})
    for source in candidate['source_files']:
        path=_payload_path(root,source['path'])
        _require(path.stat().st_size<=MAX_BUNDLE_BYTES,'SOURCE_LIMIT','Original file exceeds qualification limit')
        digest=file_sha256(path)
        _require(digest==source['sha256'],'SOURCE_HASH','Original source bytes differ: '+source['path'])
        bindings.append({'path':source['path'],'sha256':digest,'bytes':path.stat().st_size,'role':source['role']})
    digest=file_sha256(bundle)
    return {'plan_hash':plan['plan_hash'],'candidate':candidate,'metadata_and_sources':bindings,
            'bundle_sha256':digest,'bundle_bytes':bundle.stat().st_size,'validation_code_sha256':_code_bindings(),
            **({'release_id':release} if release!=DEFAULT_RELEASE else {})}


def _code_bindings():
    # Resume must also invalidate when a reused validator changes.
    return {name+'.py':file_sha256(Path(__file__).with_name(name+'.py')) for name in
        ('patient_corpus','workspace_bundle','workspace_imaging','core','imaging','structural_evidence')}


def _layer_details(declared):
    model=declared.get('model')
    fields=('name','version','revision','license','weights_sha256','code_license',
            'weights_license_choice','weights_license_source','upstream_code_commit','weights_version','variant')
    return {'source_sha256':declared.get('source_sha256'),'source_role':declared.get('source_role'),
        'coverage':thaw_json(declared.get('coverage')),
        **{key:thaw_json(declared[key]) for key in ('source_license','source_lineage','registration_status','policy_use','training_lineage') if key in declared},
        'model':{key:thaw_json(model[key]) for key in fields if key in model} if isinstance(model,Mapping) else None}


def _mask_record(name,kind,mask,case,provenance,review,extra=None):
    count=int(np.count_nonzero(mask));face_count=sum(int(np.count_nonzero(np.take(mask,index,axis=axis))) for axis in range(3) for index in (0,-1))
    return {'name':name,'kind':kind,'present':count>0,'positive_voxels':count,'volume_ml':count*case.voxel_volume_mm3/1000,
            'array_hash':array_digest(mask),'frame_hash':structural_frame_hash(case),'provenance':provenance,
            'review_status':review,'accepted_anatomical_truth':False,'face_contact_voxels_sum':face_count,
            'boundary_coverage':'truncated_or_boundary_touching' if face_count else 'no_positive_boundary_voxels; completeness not established',**(extra or {})}


def _qualification(root,candidate,binding):
    release=binding.get('release_id',DEFAULT_RELEASE)
    path=root/candidate['bundle'];_manifest_preflight(path,candidate,release=release)
    case,_,workspace=load_workspace(path)
    _require(case.case_id==candidate['case_id'],'PATIENT_MISMATCH','Primary case ID differs from metadata-only declaration')
    cases=[case]+([] if workspace is None else [item.case for item in workspace['sources']])
    source_by_hash={row['sha256']:row for row in candidate['source_files']}
    imaging=[]
    import nibabel as nib
    for current in cases:
        for observed in current.source_refs:
            if observed.provenance=='observed':
                _require(observed.sha256 in source_by_hash,'SOURCE_LINEAGE','Observed source is absent from the same-person acquisition record')
                declared_license=source_by_hash[observed.sha256].get('license')
                _require(declared_license is None or observed.license==declared_license,'RIGHTS_MISMATCH','Saved source rights differ from the frozen per-file declaration')
        ref=next((r for r in current.source_refs if r.source_id=='structural'),None)
        _require(ref is not None and ref.provenance=='observed' and ref.sha256 in source_by_hash,'IMAGE_LINEAGE','Saved image is not bound to this participant acquisition')
        normalize=lambda value:'CC0-1.0' if value=='CC0' else value
        _require(normalize(ref.license)==normalize(candidate['rights']['license']),'RIGHTS_MISMATCH','Saved imaging license differs from acquisition evidence')
        source=source_by_hash[ref.sha256];source_path=root/source['path'];qc=inspect_nifti(source_path)
        ras=np.diag([-1.,-1.,1.,1.])@current.affine if current.frame=='LPS+' else current.affine
        _require(qc['shape']==list(current.mri.shape) and np.allclose(qc['affine_ras_mm'],ras,atol=1e-6,rtol=0),'IMAGE_GRID','Saved image grid differs from source')
        original=nib.load(source_path).get_fdata(dtype=np.float32)
        _require(np.array_equal(original,current.mri),'IMAGE_ARRAY','Saved imaging does not equal the acquired source voxels')
        del original
        imaging.append({'modality':current.metadata.get('selected_modality',UNKNOWN),'source_sha256':ref.sha256,
            'source_path':source['path'],'shape':list(current.mri.shape),'affine_ras_mm':ras.tolist(),
            'frame_hash':structural_frame_hash(current),'array_hash':array_digest(current.mri),'source_voxels_equal':True,
            **({'license':ref.license} if release!=DEFAULT_RELEASE else {})})
    layers=[];tumor_masks=[];brain_masks=[];metadata_layers=case.metadata.get('anatomy_layers',{})
    for name,mask in case.compartments.items():
        declared=metadata_layers.get(name,{})
        fractional='fractional_source_target_threshold_scenario'==name
        kind=declared.get('kind','tumor' if fractional else 'unknown')
        provenance=declared.get('provenance','derived_source_threshold' if fractional else 'unknown')
        if declared.get('source_role')=='imported_source_annotation' or fractional:
            tumor_masks.append(case.source_compartments[name])
        if kind=='brain':brain_masks.append((name,mask))
        layers.append(_mask_record(name,kind,mask,case,provenance,declared.get('review_status',case.metadata.get('annotation_review','unreviewed')),
            _layer_details(declared)))
    if case.brain_mask is not None:
        declared=metadata_layers.get('brain',{});brain_masks.append(('brain',case.brain_mask))
        layers.append(_mask_record('brain','brain',case.brain_mask,case,declared.get('provenance','unknown'),declared.get('review_status','unreviewed'),_layer_details(declared)))
    for identity,evidence in case.structural_evidence.items():
        evidence.assert_matches(case);brain_masks.append((identity,evidence.mask))
        layers.append(_mask_record(identity,'brain',evidence.mask,case,evidence.provenance,evidence.review_status,
            {**_layer_details(evidence.metadata),'model_sha256':evidence.model_sha256,'evidence_hash':evidence.evidence_hash,'selected_as_working_brain':False}))
    primary_present=sorted({row['kind'] for row in layers if row['present'] and row['kind'] in ANATOMY})
    if release!=DEFAULT_RELEASE and workspace:
        # Per-series masks stay on their saved native grid; no resampling/fusion.
        for source in workspace['sources']:
            for name,mask in source.case.compartments.items():
                declared=source.case.metadata.get('anatomy_layers',{}).get(name,{})
                refs=[r for r in source.case.source_refs if r.source_id=='supplied_target_annotation']
                layers.append(_mask_record(name,declared.get('kind','unknown'),mask,source.case,
                    declared.get('provenance','unknown'),declared.get('review_status',declared.get('anatomical_review','unreviewed')),
                    {**_layer_details(declared),'series_id':source.identity(case),'modality':source.modality,
                     'scope':'display-only-native-grid','accepted_constraint':False,
                     'annotation_sha256':refs[0].sha256 if refs else None,
                     'annotation_license':refs[0].license if refs else None}))
    present=sorted({row['kind'] for row in layers if row['present'] and row['kind'] in ANATOMY})
    missing=sorted(set(ANATOMY)-set(present));location={'lobe':UNKNOWN,'hemisphere':UNKNOWN,'anatomical_region':UNKNOWN,'centroid_ras_mm':None}
    conflicts=[]
    if tumor_masks:
        union=np.logical_or.reduce(tumor_masks);count=int(union.sum())
        if count:
            centroid=np.array([np.sum(union.sum(axis=tuple(i for i in range(3) if i!=axis))*np.arange(union.shape[axis]))/count for axis in range(3)])
            ras=np.diag([-1.,-1.,1.,1.])@case.affine if case.frame=='LPS+' else case.affine
            location.update(centroid_ras_mm=(ras@np.r_[centroid,1])[:3].tolist(),source_label_union_volume_ml=count*case.voxel_volume_mm3/1000,
                meaning='Source annotation union or declared fractional threshold; not histological tumor volume or accepted localization')
            for identity,mask in brain_masks:
                conflicts.append({'brain_layer':identity,'source_target_outside_support_voxels':int(np.count_nonzero(union&~mask)),
                    'support_accepted':False,'target_clipped':False})
    registrations=[]
    if workspace:
        for source in workspace['sources']:
            association,registration=source_registration(case,source.case)
            registrations.append({'source_modality':source.modality,'source_frame_hash':structural_frame_hash(source.case),
                'reference_frame_hash':structural_frame_hash(case),'same_person':association['samePersonVerified'],**registration,
                'independently_validated_registration':False,
                **({'series_id':source.identity(case),'same_person_source_identity_verified':True,
                    'saved_registration_candidate_status':source.case.metadata.get('registration_candidate',{}).get('classification'),
                    'anatomical_registration_accepted':False} if release!=DEFAULT_RELEASE else {})})
    originals=[] if workspace is None else workspace.get('originals',[])
    embedded_hashes={row.sha256 for row in originals}
    _require(file_sha256(path)==binding['bundle_sha256'],'INPUT_CHANGED','Bundle changed during qualification')
    for source in candidate['source_files']:
        _require(file_sha256(root/source['path'])==source['sha256'],'INPUT_CHANGED','Original source changed during qualification')
    return {'schema':VERSION+'.receipt','patient_group':candidate['patient_group'],'case_id':case.case_id,'role':candidate['role'],
        'status':'qualified_with_limitations','qualification_scope':'source_identity_bytes_geometry_anatomical_presence_only',
        'existing_materialized':True,'new_patient':False,'case_hash':case.semantic_hash,'bundle':candidate['bundle'],
        'bundle_sha256':binding['bundle_sha256'],'input_binding_hash':semantic_digest(binding),'input_binding':binding,'rights':candidate['rights'],
        'participant_verified':True,'imaging':imaging,'layers':layers,'anatomy_present':present,'missing_anatomy':missing,
        'registrations':registrations,'tumor_location':location,'support_conflicts':conflicts,
        'originals':{'embedded_count':len(originals),'embedded_bytes':sum(len(row.data) for row in originals),
            'embedded_hashes_verified':True if originals else None,'external_source_hashes_verified':len(candidate['source_files']),
            'portable_originals_complete':set(source_by_hash)<=embedded_hashes,
            'files':[{'name':row.name,'sha256':row.sha256,'role':row.role,'bytes':len(row.data)} for row in originals]},
        'known_aliases':candidate.get('known_aliases',[]),'overlap_caveat':candidate.get('overlap_caveat'),
        'limitations':['All model estimates remain unaccepted anatomical truth.','Lobe and hemisphere remain UNKNOWN.',
            'No new inference, registration optimization, planning, training or downloads performed.',
            'Engineering qualification does not establish clinical or independent model accuracy.']+
            ([] if originals else ['Legacy bundle references original files externally; exact original bytes are not embedded.']),
        'clinical_use_permitted':False,'training_admission_established':False,'gui_reopen_verified_by_this_pipeline':False,
        **({'release_id':release,'anatomy_present_on_primary':primary_present,
            'anatomy_presence_scope':'Saved same-person layers across separate native grids; registration, completeness and constraints are not accepted.',
            'training_admitted':False,'virtual_descendant':False,'coverage':candidate.get('coverage','Completeness not established')} if release!=DEFAULT_RELEASE else {})}


def run_qualification(root,output,plan,*,limit=None):
    root=Path(root);output=Path(output);validate_plan(plan)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path=output/'plan.json'
        if plan_path.exists():_require(_json(plan_path.read_bytes())==plan,'PLAN_CHANGED','Use a new corpus output version for a changed allowlist')
        else:_atomic_json(plan_path,plan)
        entries=[];created=reused=attempted=0
        for candidate in plan['candidates']:
            binding=None
            try:
                binding=_input_binding(root,candidate,plan);fingerprint=semantic_digest(binding).removeprefix('sha256:')
                receipt_path=output/'receipts'/f'{fingerprint}.json'
                if receipt_path.exists():
                    receipt=_json(receipt_path.read_bytes());_require(receipt.get('input_binding_hash')=='sha256:'+fingerprint,'RECEIPT_HASH','Cached receipt has incorrect binding')
                    _require(receipt.get('receipt_hash')==semantic_digest({k:v for k,v in receipt.items() if k!='receipt_hash'}),'RECEIPT_HASH','Cached receipt was modified')
                    reused+=1
                else:
                    if limit is not None and attempted>=limit:
                        entries.append({'patient_group':candidate['patient_group'],'role':candidate['role'],'status':'pending'});continue
                    attempted+=1
                    receipt=_qualification(root,candidate,binding);receipt['receipt_hash']=semantic_digest(receipt);_atomic_json(receipt_path,receipt);created+=1
                entries.append({**receipt,'receipt_path':str(receipt_path.relative_to(output))})
            except Exception as error:
                code=getattr(error,'code','INPUT_OR_BUNDLE_FAILURE')
                receipt={'schema':VERSION+'.receipt','patient_group':candidate['patient_group'],'role':candidate['role'],'existing_materialized':True,
                    'new_patient':False,'status':'rejected','failure_code':code,'reason':str(error),'bundle':candidate['bundle'],
                    'input_binding_hash':None if binding is None else semantic_digest(binding)}
                receipt['receipt_hash']=semantic_digest(receipt);receipt_path=output/'rejections'/f"{receipt['receipt_hash'].removeprefix('sha256:')}.json";_atomic_json(receipt_path,receipt)
                entries.append({**receipt,'receipt_path':str(receipt_path.relative_to(output))})
        entries=[{k:v for k,v in row.items() if k!='reused'} for row in entries]
        counts=Counter(row['status'] for row in entries)
        qualified=[row for row in entries if row['status']=='qualified_with_limitations']
        combinations={name:sum(set(required)<=set(row['anatomy_present']) for row in qualified) for name,required in
            {'brain_plus_tumor':['brain','tumor'],'brain_plus_skull':['brain','skull'],'brain_plus_vessels':['brain','vessels'],
             'brain_tumor_skull_vessels':['brain','tumor','skull','vessels']}.items()}
        catalog={'schema':VERSION+'.catalog','plan_hash':plan['plan_hash'],'entries':entries,
            'counts':{'existing_materialized_people':len(plan['candidates']),'qualified_existing_people':len(qualified),
                      'newly_acquired_people':0,'rejected':counts['rejected'],'pending':counts['pending'],
                      'anatomical_truth_accepted_people':0,'anatomical_presence_combinations':combinations},
            'cross_cohort_independence':'unresolved; namespaced participants only','protected_imaging_opened':False}
        if plan.get('release_id')==IXI_RELEASE:
            catalog.update(release_id=IXI_RELEASE,parent_plan_hash=plan['parent_plan_hash'],
                presence_combination_meaning='Same-person presence across saved grids; not accepted fusion or complete anatomy')
            catalog['counts'].update(newly_cataloged_existing_people=1,virtual_descendants_counted_as_people=0,
                people_with_portable_source_originals=sum(row['originals']['portable_originals_complete'] for row in qualified),
                verified_embedded_original_files=sum(row['originals']['embedded_count'] for row in qualified),
                same_person_presence_across_saved_grids=combinations,
                accepted_fused_anatomy_combinations={name:0 for name in combinations})
        catalog['catalog_hash']=semantic_digest(catalog)
        version=output/'catalogs'/f"{catalog['catalog_hash'].removeprefix('sha256:')}.json"
        if not version.exists():_atomic_json(version,catalog)
        _atomic_json(output/'catalog.json',catalog);_write_search_index(output/'catalog.sqlite',catalog)
        result={'schema':VERSION+'.run','created_utc':datetime.now(timezone.utc).isoformat(),'catalog_hash':catalog['catalog_hash'],
            'new_qualification_receipts':created,'reused_qualification_receipts':reused,'counts':catalog['counts'],
            'rejections':[{'patient_group':row['patient_group'],'failure_code':row['failure_code'],'reason':row['reason']} for row in entries if row['status']=='rejected']}
        _atomic_json(output/'runs'/f'{uuid4()}.json',result)
        return result


def _write_search_index(path,catalog):
    temporary=path.with_name('.catalog-'+str(uuid4())+'.sqlite')
    with sqlite3.connect(temporary) as connection:
        connection.execute('CREATE TABLE cases(patient_group TEXT PRIMARY KEY, role TEXT, status TEXT, modalities TEXT, anatomy TEXT, missing TEXT, lobe TEXT, hemisphere TEXT, payload TEXT)')
        for row in catalog['entries']:
            connection.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?)',(row['patient_group'],row['role'],row['status'],
                '|'.join(image['modality'] for image in row.get('imaging',[])),'|'.join(row.get('anatomy_present',[])),
                '|'.join(row.get('missing_anatomy',[])),UNKNOWN,UNKNOWN,json.dumps(row,sort_keys=True)))
    os.replace(temporary,path)


def query_catalog(output,*,text='',anatomy=None,modality=None,role=None,status=None):
    """Read the local searchable catalog; never opens image or bundle payloads."""
    path=(Path(output)/'catalog.sqlite').resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
        rows=[json.loads(row[0]) for row in connection.execute('SELECT payload FROM cases ORDER BY patient_group')]
    def searchable(row):
        # Shared release evidence is an audit binding, not this person's identity.
        return json.dumps({key:row.get(key) for key in ('patient_group','case_id','role','status',
            'anatomy_present','missing_anatomy','imaging','layers','tumor_location','rights','overlap_caveat')}).casefold()
    return [row for row in rows if (not text or text.casefold() in searchable(row)) and
        (not anatomy or anatomy in row.get('anatomy_present',[])) and
        (not modality or modality in {image['modality'] for image in row.get('imaging',[])}) and
        (not role or row['role']==role) and (not status or row['status']==status)]
