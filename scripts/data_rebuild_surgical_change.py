#!/usr/bin/env python3
"""Bounded RESECT Case3 TRAIN source-native MRI/US stage pilot; no registration.

Reuses frozen source identities/MD5s and production workspace readers/writers.
All bytes, partials, native exports and QC are external. No SELECT/EVAL images,
new cohort downloads, annotations with different rights, inference or RL.
"""
from __future__ import annotations
import dataclasses,hashlib,json,math,pathlib,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from data_rebuild_inventory import mount
from data_rebuild_real import reserve
from resectionlab.imaging import load_nifti_case,file_sha256,inspect_nifti
from resectionlab.workspace_bundle import load_workspace,save_workspace,OriginalFile,view_state
from resectionlab.workspace_imaging import DisplaySource,MAX_ATTACHMENT_BYTES
DATA=pathlib.Path('/Volumes/shafrir/old coding projects/RessectionLabData');SOURCE=DATA/'sources/RESECT/v1/Case3';OUT=DATA/'staging/candidates/surgical-change';QC=DATA/'qc/surgical-change/RESECT-Case3';CAT=DATA/'catalog/inventory/surgical-change';PREFIX='https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/'

def write(p,x):
 mount();p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def md5(p):
 h=hashlib.md5()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def acquire(row,deadline):
 reserve();p=SOURCE/pathlib.Path(row['source_path']).name;expected=row['bytes']
 if p.exists():
  if p.stat().st_size!=expected or md5(p)!=row['expected_md5']:raise ValueError('Existing immutable source differs')
  return p
 # Reuse identical existing local content by published size+MD5, excluding
 # protected imaging before any byte read. No broad source-image scan.
 inventory=json.loads((DATA/'catalog/inventory/files.json').read_text())
 for record in inventory:
  if record['kind']=='file' and record['bytes']==expected and not record['protected']:
   candidate=pathlib.Path(record['path'])
   if candidate.is_file() and md5(candidate)==row['expected_md5']:
    import shutil
    mount();SOURCE.mkdir(parents=True,exist_ok=True);shutil.copyfile(candidate,p)
    return p
 mount();SOURCE.mkdir(parents=True,exist_ok=True);partial=p.with_suffix(p.suffix+'.partial');offset=partial.stat().st_size if partial.exists() else 0
 for attempt in range(2):
  if time.monotonic()>deadline:raise TimeoutError('15 minute surgical-change bound reached')
  reserve();offset=partial.stat().st_size if partial.exists() else 0
  if offset>expected:raise ValueError('Partial exceeds source byte count')
  if offset==expected:break
  request=urllib.request.Request(row['source_url'],headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity','User-Agent':'RessectionLab-source-pilot/1'})
  try:
   with urllib.request.urlopen(request,timeout=30) as response:
    cr=response.headers.get('Content-Range','')
    if not ((response.status==206 and cr==f'bytes {offset}-{expected-1}/{expected}') or (response.status==200 and offset==0 and int(response.headers.get('Content-Length','-1'))==expected)):raise ValueError('Source response range/length differs from frozen file')
    with partial.open('ab') as f:
     while b:=response.read(1024*1024):
      if time.monotonic()>deadline:raise TimeoutError('15 minute source bound reached')
      offset+=len(b)
      if offset>expected:raise ValueError('Source exceeds frozen length')
      f.write(b)
   if partial.stat().st_size==expected:break
  except (OSError,TimeoutError):
   if attempt==1:raise
 if partial.stat().st_size!=expected or md5(partial)!=row['expected_md5']:raise ValueError('Downloaded size/MD5 differs from published inventory')
 mount();partial.replace(p);return p

def contact_sheet(primary,sources):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 items=[('Gd-T1 preoperative',primary.mri)]+[(s.case.metadata['display_label'],s.case.mri) for s in sources]
 fig,axes=plt.subplots(5,3,figsize=(10,14),facecolor='white')
 for row,(name,image) in enumerate(items):
  sample=image[::4,::4,::4];positive=sample[sample!=0];lo,hi=np.percentile(positive if positive.size else sample,[1,99])
  for axis in range(3):
   cut=image.shape[axis]//2;ax=axes[row,axis];ax.imshow(np.take(image,cut,axis=axis).T,origin='lower',cmap='gray',vmin=lo,vmax=hi);ax.set_title(name+'\nNativeaxis'+str(axis)+' index'+str(cut),fontsize=8);ax.axis('off')
 fig.suptitle('RESECT Case3 | source-native MRI and acquired ultrasound stages\nDistinct native planes and per-volume display windows; no registration/overlay or surgical-boundary claim',fontsize=11);fig.tight_layout(rect=(0,0,1,.95));mount();QC.mkdir(parents=True,exist_ok=True);path=QC/'source-stage-native-contact.png';fig.savefig(path,dpi=120);plt.close(fig);return str(path)

def verify_display_payloads(previous,current):
 from io import BytesIO
 import zipfile
 proof=[]
 def digest_member(archive,name):
  h=hashlib.sha256();count=0
  with archive.open(name) as f:
   while b:=f.read(4*1024*1024):h.update(b);count+=len(b)
  return h.hexdigest(),count
 with zipfile.ZipFile(previous) as a,zipfile.ZipFile(current) as b:
  names=['primary']+[x for x in a.namelist() if x.startswith('workspace/series-') and x.endswith('.ressectionlab')]
  for name in names:
   aa=a if name=='primary' else zipfile.ZipFile(BytesIO(a.read(name)));bb=b if name=='primary' else zipfile.ZipFile(BytesIO(b.read(name)))
   with zipfile.ZipFile(BytesIO(aa.read('arrays.npz'))) as an,zipfile.ZipFile(BytesIO(bb.read('arrays.npz'))) as bn:
    if an.namelist()!=bn.namelist():raise ValueError('Displaypolicy changed source array member names')
    for key in an.namelist():
     left,size=digest_member(an,key);right,other=digest_member(bn,key)
     if left!=right or size!=other:raise ValueError('Displaypolicy changed NPY bytes '+name+'/'+key)
     proof.append({'source':name,'npy_member':key,'bytes':size,'sha256':left,'before_after_equal':True})
   if name!='primary':aa.close();bb.close()
 result={'previous_package':str(previous),'previous_sha256':file_sha256(previous),'current_package':str(current),'current_sha256':file_sha256(current),'anatomical_resampling':False,'all_npy_payloads_exact':True,'arrays':proof};write(QC/'display-v3-exact-npy-proof.json',result);return str(QC/'display-v3-exact-npy-proof.json')

def main():
 start=time.monotonic();deadline=start+900;reserve();cohort=ROOT/'manifests/resect-component-cohort-v1.json';c=json.loads(cohort.read_text());member=next(r for r in c['members'] if r['patient_group']=='RESECT:Case3')
 if member['role']!='TRAIN':raise ValueError('Frozen Case3 role is not TRAIN; no source bytes permitted')
 manifest=ROOT/'manifests/resect-train-originals-v1.json';d=json.loads(manifest.read_text());rows=[r for r in d['sources'] if r.get('patient_group')=='RESECT:Case3' and r['kind'] in {'original_mri','original_ultrasound'}]
 for r in d['excluded_existing_originals']:
  if '/Case3/' in r['source_path']:rows.append({**r,'source_url':PREFIX+r['source_path'],'kind':'original_ultrasound'})
 if len(rows)!=5 or sum(r['bytes'] for r in rows)>60*1024*1024:raise ValueError('Unexpected frozen five-volume Case3 scope')
 acquisition=[];headers=[];paths={}
 for row in rows:
  p=acquire(row,deadline);name=p.name.removesuffix('.nii.gz').removeprefix('Case3-');paths[name]=p
  acquisition.append({'name':name,'path':str(p),'source_url':row['source_url'],'bytes':p.stat().st_size,'md5_verified':row['expected_md5'],'sha256':file_sha256(p),'role':'TRAIN','license':'CC-BY-4.0'})
  header=inspect_nifti(p);headers.append({'name':name,**header});write(CAT/'RESECT-Case3-acquisition.json',{'patient_id':'RESECT:Case3','role':'TRAIN','cohort_sha256':file_sha256(cohort),'source_manifest_sha256':file_sha256(manifest),'files':acquisition,'status':'in_progress' if len(acquisition)<5 else 'bytes_verified','array_values_decoded':False})
 write(QC/'source-native-headers.json',headers)
 # Header budget precedes source-array loading. No downsampling or arbitrary
 # grid resizing is introduced merely to fit the native app attachment budget.
 aux_names=['FLAIR','US-before','US-during','US-after'];by_name={r['name']:r for r in headers};expanded=sum(math.prod(by_name[n]['shape'])*4 for n in aux_names)
 if expanded>MAX_ATTACHMENT_BYTES:raise ValueError(f'Original native display attachments exceed app budget ({expanded}>{MAX_ATTACHMENT_BYTES}); preserve sources and headers, no resizing')
 captions={'FLAIR':'FLAIR preoperative · acquired native','US-before':'US before resection · acquired native','US-during':'US during resection · acquired native','US-after':'US after resection · acquired native'}
 metadata={'display_label':'Gd-T1 preoperative · acquired native','patient_id':'RESECT-Case3','source_collection':'RESECT','source_record_url':'https://doi.org/10.11582/2017.00004','selected_modality':'T1CE','session_id':'preoperative','is_synthetic':False,'source_frame_declaration':'Source native NIfTI physical grids; MRI/US cross-timepoint registration unreviewed; no guessed transform or direct array overlay'}
 primary=load_nifti_case(paths['T1'],case_id='RESECT-Case3',license='CC-BY-4.0',source_url=metadata['source_record_url'],metadata=metadata);sources=[]
 for name in aux_names:
  modality='FLAIR' if name=='FLAIR' else 'other-3D-scalar';stage='preoperative' if name=='FLAIR' else 'intraoperative-'+name.removeprefix('US-')
  aux=load_nifti_case(paths[name],case_id='RESECT-Case3',license='CC-BY-4.0',source_url=metadata['source_record_url'],metadata={**metadata,'selected_modality':modality,'display_label':captions[name],'native_modality':'MRI' if name=='FLAIR' else 'US','acquisition_stage':stage,'session_id':stage});sources.append(DisplaySource(aux,modality,'none'))
 dc={'schema':'ressectionlab-data-case-v1','case_id':'RESECT-Case3','family_id':'RESECT:Case3','case_type':'real','population':'publisher-described low-grade glioma surgical cohort','source':'RESECT','source_version':'v1/2017; frozen NIRD original inventory','source_doi':'10.11582/2017.00004','source_license':'CC-BY-4.0','role':'TRAIN','split':'TRAIN','exposure':'Existing prospectively assigned TRAIN pilot; source/registration method previously investigated; not independent evaluation','components':{'preoperative_MRI':{'provenance':'observed','modalities':['Gd-T1w','FLAIR']},'intraoperative_US':{'provenance':'observed','stages':['before','during','after']},'brain_mask':{'provenance':'missing'},'tumor_or_cavity_labels':{'provenance':'not acquired in this image-only pilot'}},'capabilities':['source_native_MRI_inspection','source_native_3D_ultrasound_stage_inspection'],'missing_anatomy':['skull_CT','dedicated_angiography','brain_segmentation','tumor_segmentation','cavity_segmentation','functional_mapping','validated_tractography'],'source_files':acquisition,'coordinate_frame':'NIfTI source-native RAS+ physical grids, distinct MRI and US sampling','registration':'unreviewed; no local registration/overlay/brain-shift claim','application_consumption_limit':'US displayed via existing other-3D-scalar source type; no calibrated ultrasound renderer, time-aligned deformation, instrument data or action/force outcome modeling','simulation_admission':'withheld; source visual inspection only','accepted':False,'status':'candidate_pending_independent_review','is_synthetic':False,'recorded_rl_transitions':0}
 primary=dataclasses.replace(primary,metadata={**metadata,**dict(primary.metadata),'data_case':dc});originals=[OriginalFile.from_path(p,role='imaging') for p in paths.values()];mount();OUT.mkdir(parents=True,exist_ok=True);package=OUT/'RESECT-Case3-source-stages-v3.ressectionlab'
 initial=view_state(primary,sources)
 for item in sources:
  if item.case.metadata.get('native_modality')=='US':initial['states'][item.identity(primary)]['viewer']={'volumeRendering':{'enabled':False,'preset':'mri'}}
 if not package.exists():save_workspace(primary,package,sources=sources,imaging=initial,episode=None,selection=None,artifacts={},originals=originals)
 opened,_,workspace=load_workspace(package)
 if opened.semantic_hash!=primary.semantic_hash:raise ValueError('Production reopen semantic identity mismatch')
 for item in workspace['sources']:
  if item.case.metadata.get('native_modality')=='US' and workspace['imagingState']['states'][item.identity(opened)].get('viewer',{}).get('volumeRendering',{}).get('enabled') is not False:raise ValueError('US display default lost safe MPR policy')
 receipt={'path':str(package),'sha256':file_sha256(package),'bytes':package.stat().st_size,'case_id':'RESECT-Case3','family_id':'RESECT:Case3','role':'TRAIN','prior_project_source_exposure':'Already-exposed frozen TRAIN Case3 pilot recovery; not a never-seen new patient','new_never_seen_people':0,'recovered_exposed_family_count':1,'accepted':False,'production_reopen':'pass','semantic_hash':opened.semantic_hash,'embedded_originals':len(workspace['originals']),'aux_series':len(workspace['sources']),'attachment_expanded_bytes':expanded,'elapsed_seconds':time.monotonic()-start,'source_array_grids':[{'name':r['name'],'shape':r['shape'],'spacing_mm':r.get('spacing_mm')} for r in headers]};receipt['display_policy']='US native MPR retained, initial US3D raycast disabled; primaryMRI normal display retained';previous=OUT/'RESECT-Case3-source-stages-v2.ressectionlab'
 if previous.exists():receipt['exact_npy_payload_proof']=verify_display_payloads(previous,package)
 receipt['builder_contact_sheet']=contact_sheet(opened,workspace['sources']);receipt['display_labels']=[str(s.case.metadata['display_label']) for s in workspace['sources']];write(CAT/'RESECT-Case3-candidate.json',receipt);write(CAT/'superseded-candidates.json',[{'path':str(OUT/f'RESECT-Case3-source-stages-v{v}.ressectionlab'),'replacement':str(package),'reason':'v3 retains stagecaptions and disables initialUS3Draycast only; v1/v2 remain byte-preserved, superseded displaypolicy'} for v in (1,2)]);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
