#!/usr/bin/env python3
"""Build review-only real-tumor candidates and a predeclared UCSF TRAIN pilot.

External immutable inputs, production workspace import/export, no model inference,
no training, no assessment sources. Run plan, then pilot; batch requires explicit
independent pilot review receipt. Source mirror equivalence remains UNKNOWN.
"""
from __future__ import annotations
import argparse,csv,dataclasses,hashlib,io,json,os,pathlib,re,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from data_rebuild_inventory import mount
from resectionlab.core import thaw_json
from resectionlab.workspace_bundle import load_workspace,save_workspace,OriginalFile
from resectionlab.workspace_imaging import DisplaySource
from resectionlab.imaging import load_nifti_case,file_sha256
from acquire_public_case import acquire_file
DATA=pathlib.Path('/Volumes/shafrir/old coding projects/RessectionLabData')
REV='e9372219cf1cd2fdd52260cd45f7514b4aa7638e';BASE=f'https://huggingface.co/datasets/MedOtter/UCSF-PDGM';SOURCE=DATA/'sources/UCSF-PDGM'/REV
OUT=DATA/'staging/candidates/real';META=DATA/'catalog/inventory/real-tumor';PLAN=META/'ucsf-train-plan.json'
LABELS={1:'nonenhancing_core_or_necrosis',2:'peritumoral_edema',4:'enhancing_tumor'}

def write(p,x):
 mount();p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def get(url):
 with urllib.request.urlopen(url,timeout=45) as r:return r.read()
def reserve():
 mount();usage=os.statvfs(DATA);free=usage.f_bavail*usage.f_frsize;total=usage.f_blocks*usage.f_frsize
 if free<max(50*1024**3,total//10)+1024**3:raise RuntimeError('External free-space reserve exhausted')
def plan():
 if PLAN.exists():return json.loads(PLAN.read_text())
 reserve();raw=get(f'https://huggingface.co/api/datasets/MedOtter/UCSF-PDGM/tree/{REV}?limit=1000');listing=json.loads(raw)
 card=get(f'{BASE}/raw/{REV}/README.md');SOURCE.mkdir(parents=True,exist_ok=True);(SOURCE/'README.md').write_bytes(card)
 if 'license: cc-by-4.0' not in card.decode() or 'Non-enhancing tumor core / necrosis' not in card.decode():raise ValueError('Pinned card license/label contract changed')
 registry=(ROOT/'manifests/cohort_registry.json').read_text();excluded={int(x) for x in re.findall(r'UCSF-PDGM[-:]0*(\d+)',registry)}|{4,5,7,8}
 # Subject identities/known aliases only; no diagnosis, outcomes, voxel counts,
 # tumor-slice index, segmentation quality or performance used in selection.
 local=ROOT/'data/provenance/ucsf-pdgm/UCSF-PDGM-metadata_v5.csv';identity_source='local official metadata' if local.is_file() else 'pinned mirror metadata; official byte equivalence unknown'
 payload=local.read_bytes() if local.is_file() else get(f'{BASE}/raw/{REV}/UCSF-PDGM-metadata_v5.csv')
 identities=[]
 for r in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))):
  identities.append({k:r.get(k) for k in ['ID','BraTS21 ID','BraTS21 Segmentation Cohort','BraTS21 MGMT Cohort']})
 write(META/'ucsf-identity-projection.json',{'source':identity_source,'raw_sha256':hashlib.sha256(payload).hexdigest(),'fields':'Identity and published cohort only; clinical/outcome fields not retained','identities':identities})
 ids=sorted([r['path'] for r in listing if r['type']=='directory' and re.fullmatch(r'UCSF-PDGM-\d{4}',r['path']) and int(r['path'].split('-')[-1]) not in excluded])[:50]
 members=[]
 for subject in ids:
  identity=next((r for r in identities if r['ID']==f'UCSF-PDGM-{int(subject.split("-")[-1]):03d}'),None)
  members.append({'patient_id':subject,'family_id':f'UCSF-PDGM:{int(subject.split("-")[-1]):03d}','role':'TRAIN','identity':identity,'role_frozen_before_image_bytes':True,'pretraining_exposure':'unknown; published BraTS overlap recorded, never independent evaluation'})
 d={'schema':'ressectionlab-data-train-plan-v1','declared_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selection':'first 50 ascending pinned MRI folder IDs excluding every UCSF identity/alias mentioned by existing registry plus development0004/0005/0007 and frozen0008; no pixel/outcome selection','excluded_numeric_ids':sorted(excluded),'registry_sha256':hashlib.sha256(registry.encode()).hexdigest(),'mirror_revision':REV,'source_card_sha256':hashlib.sha256(card).hexdigest(),'members':members,'pilot_patient_id':ids[0],'independent_evaluation_claim':False}
 write(PLAN,d);return d

def contract(subject,role,components,source_files,**extras):
 return {'schema':'ressectionlab-data-case-v1','case_id':subject,'family_id':f'UCSF-PDGM:{int(subject.split("-")[-1]):03d}' if subject.startswith('UCSF') else 'BTC:sub-PAT22','case_type':'real','population':'real_tumor','source':'UCSF-PDGM' if subject.startswith('UCSF') else 'BTC-ds001226','source_version':REV if subject.startswith('UCSF') else '5.0.1','source_license':'CC-BY-4.0' if subject.startswith('UCSF') else 'CC0-1.0','source_doi':'10.7937/tcia.bdgf-8v37' if subject.startswith('UCSF') else '10.18112/openneuro.ds001226.v5.0.1','role':role,'split':role,'exposure':'previously consulted development' if role=='DEVELOPMENT' else 'new TRAIN; no independent-evaluation claim; pretrained overlap unknown','original_patient':subject,'source_files':source_files,'components':components,'capabilities':['structural_MRI_inspection','supplied_tumor_annotation_inspection'],'missing_anatomy':['acquired_skull_CT','dedicated_angiography','functional_mapping','validated_tractography'],'coordinate_frame':'RAS+ millimeters; source-registered structural grid','clinical_use_status':'research_only','status':'candidate_pending_independent_review','accepted':False,'official_tcia_byte_equivalence':'unknown' if subject.startswith('UCSF') else 'not_applicable','source_mirror_attribution':BASE if subject.startswith('UCSF') else 'https://openneuro.org/datasets/ds001226/versions/5.0.1',**extras}

def package(case,path,dc,sources=(),originals=()):
 reserve();meta=thaw_json(case.metadata);meta['data_case']=dc;meta['is_synthetic']=False;updated=dataclasses.replace(case,metadata=meta)
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  opened,_,workspace=load_workspace(path)
  if thaw_json(opened.metadata.get('data_case',{}))!=dc:raise ValueError('Existing candidate metadata differs; preserve distinct versions')
  status='existing_reopened'
 else:
  save_workspace(updated,path,sources=sources,imaging=None,episode=None,selection=None,artifacts={},originals=originals)
  opened,_,workspace=load_workspace(path);status='production_reopen_pass'
 if opened.semantic_hash!=updated.semantic_hash or opened.planning_hash!=updated.planning_hash:raise RuntimeError('Production reopen identity mismatch')
 return {'path':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size,'semantic_hash':opened.semantic_hash,'status':status,'case_id':dc['case_id'],'family_id':dc['family_id'],'role':dc['role'],'accepted':False,'review_status':'pending_independent_review','source_grid_shape':list(opened.mri.shape),'source_label_voxels':{k:int(v.sum()) for k,v in opened.source_compartments.items()},'package_embeds_originals':len(originals),'package_embeds_auxiliary_series':len(sources)}

def existing():
 entries=[('UCSF-PDGM-0004',ROOT/'testcases/cases/UCSF-PDGM-0004-rehearsal-ready.ressectionlab','DEVELOPMENT'),('UCSF-PDGM-0005',ROOT/'testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0005/scene/patient.ressectionlab','DEVELOPMENT'),('UCSF-PDGM-0007',ROOT/'testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/scene/patient.ressectionlab','DEVELOPMENT'),('BTC-PAT22',ROOT/'testcases/downloaded-datasets/BTC-PAT22-recovery/source-case.ressectionlab','TRAIN')];receipts=[]
 for subject,p,role in entries:
  case,_,w=load_workspace(p);m=thaw_json(case.metadata);layers=m.get('anatomy_layers',{});components={k:{key:v.get(key) for key in ['kind','provenance','review_status','coverage','source_role']} for k,v in layers.items()}
  extra_sources=w['sources'] if w else [];originals=w.get('originals',[]) if w else []
  version='v1'
  if subject in {'UCSF-PDGM-0005','UCSF-PDGM-0007'}:
   version='v2';source=ROOT/'testcases/downloaded-datasets/UCSF-fresh-development-20261026'/subject
   case=dataclasses.replace(case,source_refs=tuple(dataclasses.replace(r,license='CC-BY-4.0' if r.source_id=='T1CE' else 'local-derived metadata; source imaging CC-BY-4.0; model terms retained in embedded manifest') for r in case.source_refs),metadata={**thaw_json(case.metadata),'patient_id':subject,'session_id':'preoperative'})
   originals=[OriginalFile.from_path(source/f'{subject}_{suffix}.nii.gz',role='annotation' if suffix=='tumor_segmentation' else 'imaging') for suffix in ['T1','T1c','T2','FLAIR','tumor_segmentation']]
   native=p.parent.parent/'native/patient-anatomy.json';originals.append(OriginalFile.from_path(native,role='metadata'));originals.append(OriginalFile.from_path(SOURCE/'README.md',role='metadata'))
   extra_sources=[DisplaySource(load_nifti_case(source/f'{subject}_{suffix}.nii.gz',case_id=subject,license='CC-BY-4.0',source_url=BASE,metadata={'patient_id':subject,'session_id':'preoperative'}),suffix,'none') for suffix in ['T1','T2','FLAIR']]
  if subject=='BTC-PAT22':version='v2'
  refs=[r.to_dict() for r in case.source_refs];dc=contract(subject,role,components,refs,exposure=('previously consulted TRAIN development source; not new acquisition; model-pretraining exposure unknown' if role=='TRAIN' else 'previously consulted development'),parent_package_sha256=file_sha256(p),parent_package_path=str(p),brain_segmentation='estimated' if case.brain_mask is not None or 'brain' in case.compartments else 'missing',simulation_admission='unqualified research estimates only; preserve original restrictions',exact_acquired_all_anatomy=False)
  r=package(case,OUT/(subject+'-existing-'+version+'.ressectionlab'),dc,sources=extra_sources,originals=originals);receipts.append(r)
 write(META/'existing-candidates.json',receipts);return receipts

def acquire_prepare(subject):
 d=plan()
 if file_sha256(SOURCE/'README.md')!=d['source_card_sha256']:raise ValueError('Pinned source card changed after declaration')
 member=next((x for x in d['members'] if x['patient_id']==subject),None)
 if not member or member['role']!='TRAIN':raise ValueError('Only frozen TRAIN subjects may acquire/prepare')
 reserve();from prepare_fresh_ucsf_demo import source_listing
 listing=source_listing(subject,REV);records=[];SOURCE.mkdir(parents=True,exist_ok=True)
 previous_path=SOURCE/subject/'acquisition.json';previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
 for row in listing:
  entry={'path':row['path'],'size_bytes':row['size'],'sha256':row['lfs']['oid'],'source_url':f'{BASE}/resolve/{REV}/{row["path"]}'}
  reserve()
  for attempt in range(3):
   try:disposition=acquire_file(entry,SOURCE);break
   except (OSError,TimeoutError):
    if attempt==2:raise
    time.sleep(2**attempt)
  prior=next((r for r in previous.get('files',[]) if r.get('path')==entry['path'] and r.get('sha256')==entry['sha256']),None)
  records.append(prior if prior is not None else {**entry,'disposition':disposition})
 write(SOURCE/subject/'acquisition.json',{'role':'TRAIN','plan_sha256':file_sha256(PLAN),'files':records,'official_tcia_byte_equivalence':'unknown'})
 source=SOURCE/subject;paths={s:source/f'{subject}_{s}.nii.gz' for s in ['T1','T1c','T2','FLAIR','tumor_segmentation']}
 # Production import enforces finite volumes and aligned integer supplied labels.
 dc=contract(subject,'TRAIN',{'structural_MRI':{'provenance':'observed_preprocessed','modalities':['T1','T1CE','T2','FLAIR']},'tumor_annotation':{'provenance':'source_provided_expert_corrected_according_to_mirror_card','labels':{str(k):v for k,v in LABELS.items()}},'brain':{'provenance':'missing_explicit_mask','coverage':'skullstripped imaging only; no anatomical brain segmentation inferred'}},records,aliases=member['identity'],brain_segmentation='not_provided',simulation_admission='withheld_missing_validated_brain_mask',exact_acquired_all_anatomy=False,source_registration='pinned mirror declares nonlinear coregistration to FLAIR, checked native affines; no local landmark validation')
 primary=load_nifti_case(paths['T1c'],paths['tumor_segmentation'],case_id=subject,label_map=LABELS,license='CC-BY-4.0',source_url=BASE,metadata={'selected_modality':'T1CE','patient_id':subject,'session_id':'preoperative','source_collection':'UCSF-PDGM','source_distribution':'public BraTS-format structural mirror','source_distribution_version':REV,'missing_anatomy':dc['missing_anatomy'],'source_frame_declaration':dc['source_registration']})
 sources=[]
 for suffix in ['T1','T2','FLAIR']:
  aux=load_nifti_case(paths[suffix],case_id=subject,license='CC-BY-4.0',source_url=BASE,metadata={'patient_id':subject,'session_id':'preoperative','source_frame_declaration':dc['source_registration']});
  import numpy as np
  if aux.mri.shape!=primary.mri.shape or not np.allclose(aux.affine,primary.affine,atol=1e-5,rtol=0):raise ValueError('Source structural series do not share the declared registration grid')
  sources.append(DisplaySource(aux,suffix,'none'))
 originals=[OriginalFile.from_path(p,role='annotation' if key=='tumor_segmentation' else 'imaging') for key,p in paths.items()]
 version='v1'
 if subject=='UCSF-PDGM-0040':
  import numpy as np
  zero=primary.mri==0
  for item in sources:zero&=item.case.mri==0
  target=primary.source_compartments['enhancing_tumor']
  if int(target.sum())!=23979 or int(np.count_nonzero(target&zero))!=521:raise ValueError('Pinned0040 support mismatch differs from independent source review')
  dc['source_limitations']=known_warning_0040();version='v2'
 receipt=package(primary,OUT/(subject+'-structural-'+version+'.ressectionlab'),dc,sources,originals);receipt['builder_contact_sheet']=contact_sheet(primary,sources,subject);write(META/(subject+'-candidate.json'),receipt);return receipt

def contact_sheet(case,sources,subject):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 union=np.zeros(case.mri.shape,dtype=bool)
 for mask in case.source_compartments.values():union|=mask
 coords=np.where(union)
 cuts=[int(np.median(a)) if len(a) else case.mri.shape[i]//2 for i,a in enumerate(coords)]
 fig,axes=plt.subplots(4,3,figsize=(10,12),facecolor='white')
 items=[('T1CE',case.mri)]+[(s.modality,s.case.mri) for s in sources]
 for row,(name,image) in enumerate(items):
  sample=image[::3,::3,::3];positive=sample[sample!=0];lo,hi=np.percentile(positive if positive.size else sample,[1,99])
  for axis in range(3):
   ax=axes[row,axis];ax.imshow(np.take(image,cuts[axis],axis=axis).T,origin='lower',cmap='gray',vmin=lo,vmax=hi)
   rgba=np.zeros(np.take(union,cuts[axis],axis=axis).T.shape+(4,),dtype=float)
   for key,color in [('nonenhancing_core_or_necrosis',(1,1,0,.45)),('peritumoral_edema',(0,.8,1,.35)),('enhancing_tumor',(1,0,0,.55))]:
    if key in case.source_compartments:rgba[np.take(case.source_compartments[key],cuts[axis],axis=axis).T]=color
   ax.imshow(rgba,origin='lower');ax.set_title(name+' native axis'+str(axis)+' index'+str(cuts[axis]),fontsize=9);ax.axis('off')
 fig.suptitle(subject+' | supplied labels: NCR yellow, edema cyan, ET red\nNative voxel planes; no anatomical/clinical validation; missing skull/angiography/brain mask',fontsize=11);fig.tight_layout(rect=(0,0,1,.95))
 folder=DATA/'qc/real-tumor-candidates';folder.mkdir(parents=True,exist_ok=True);path=folder/(subject+'-source-overlay.png');fig.savefig(path,dpi=120);plt.close(fig)
 return str(path)

def known_warning_0040():
 return [{'code':'SOURCE_ANNOTATION_OUTSIDE_RETAINED_MRI_SUPPORT','affected_label':'enhancing_tumor','source_label_voxels':23979,'voxels_exactly_zero_in_all_four_source_MRI':521,'fraction':521/23979,'basis':'Independent source-original review real-outlier-native-original-checks-v1.json; inherited source skullstripping/annotation support mismatch','interpretation':'source and supplied-annotation inspection only; no complete-target-image or surgical-boundary claim','source_labels_modified':False}]

def source_warning_0040():
 subject='UCSF-PDGM-0040';old=OUT/(subject+'-structural-v1.ressectionlab');case,_,w=load_workspace(old)
 dc=thaw_json(case.metadata['data_case']);dc['source_limitations']=known_warning_0040()
 receipt=package(case,OUT/(subject+'-structural-v2.ressectionlab'),dc,w['sources'],w['originals'])
 import numpy as np
 opened,_,_=load_workspace(receipt['path'])
 if not np.array_equal(opened.mri,case.mri) or any(not np.array_equal(opened.source_compartments[k],case.source_compartments[k]) for k in case.source_compartments):raise ValueError('Metadata repair changed source arrays')
 receipt['metadata_repair_only']=True;receipt['builder_contact_sheet']=str(DATA/'qc/real-tumor-candidates'/(subject+'-source-overlay.png'));write(META/(subject+'-candidate.json'),receipt)
 batch=json.loads((META/'batch-candidates.json').read_text());batch=[receipt if r['case_id']==subject else r for r in batch];write(META/'batch-candidates.json',batch)
 superseded=json.loads((META/'superseded-candidates.json').read_text());record={'superseded_path':str(old),'replacement_path':receipt['path'],'reason':'Explicit independent source-support limitation added; all original arrays preserved exactly','action':'preserve v1 as superseded staging evidence; accept only current v2 with inspection-only capability'}
 if record not in superseded:superseded.append(record)
 write(META/'superseded-candidates.json',superseded);return receipt

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['plan','existing','pilot','batch','source-warning-0040']);ap.add_argument('--count',type=int,default=50);ap.add_argument('--review-receipt',type=pathlib.Path);a=ap.parse_args();reserve()
 if a.command=='plan':result=plan()
 elif a.command=='existing':result=existing()
 elif a.command=='pilot':result=acquire_prepare(plan()['pilot_patient_id'])
 elif a.command=='source-warning-0040':result=source_warning_0040()
 else:
  if a.review_receipt is None:raise ValueError('Batch requires independent pilot review receipt')
  review=json.loads(a.review_receipt.read_text());pilot=plan()['pilot_patient_id']
  if review.get('patient_id')!=pilot or review.get('decision') not in {'accept','accepted'} or review.get('independent') is not True:raise ValueError('Pilot review receipt does not independently accept source/geometry')
  if not 1<=a.count<=50:raise ValueError('Batch count must fit frozen 1..50 declaration')
  if review.get('package_sha256')!=file_sha256(OUT/(pilot+'-structural-v1.ressectionlab')) or review.get('role_plan_sha256')!=file_sha256(PLAN):raise ValueError('Independent scale review no longer binds pilot/role plan')
  result=[]
  for member in plan()['members'][:a.count]:
   subject=member['patient_id']
   try:receipt=acquire_prepare(subject)
   except Exception as error:
    receipt={'patient_id':subject,'role':'TRAIN','status':'failed_preparation_excluded','error':str(error),'accepted':False};write(META/(subject+'-failure.json'),receipt)
   result.append(receipt);write(META/'batch-progress.json',{'completed_or_failed':len(result),'planned':a.count,'results':result});print(json.dumps({'subject':subject,'status':receipt['status']}),flush=True)
  write(META/'batch-candidates.json',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
