"""Metadata/stat-only exact TRAIN050 public source binding; no DICOM opens."""
from pathlib import Path
import hashlib
import json
import re
import stat
import difflib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PERSON='ReMIND-050'
KINDS={'3D_AX_T1_postcontrast':'structural_t1ce','tumor seg - MR ref: 3D_AX_T1_postcontrast':'whole_tumor','cerebrum seg - MR ref: 3D_AX_T1_postcontrast':'cerebrum'}
PINS={
 'manifests/experiments/remind-component-cohort-v1.json':'326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05',
 'build/remind-cohort-expansion-preparation-v1/idc-remind-series-metadata.json':'8301d9890e97316c8905448c9ec78e851d43ed18b383455ae138f4b13a8023be',
 'build/remind-train-public-expansion-candidate-v1/candidate.json':'d436dacfc1ed54a2061cd21bb9c309d0403fdf7b98fc637c3a66c00e546084e6',
 'data/acquisition/remind-train-originals-v1/source/train-exact-object-manifest.json':'5b1e51bb923825b96f53f789f2cf464fccb64fc2515b9b47d70c37cc693ae0f5',
 'data/acquisition/remind-train-originals-v1/completion.json':'32e319a3aafb571019ee16da1072c834015f9fdcdb97563fca4c19418f64a61d',
 'data/acquisition/remind-train-seg-v1/source/train-exact-object-manifest.json':'87eb3b3fc1b78276f1d58ae873665d89b62e21f65244084cb77a7ee34b736525',
 'data/acquisition/remind-train-seg-v1/completion.json':'2b3a5605be4d9f8376e92dc5cdae318077153f8352a8a699d78c8a906a9abd4f'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def binding(path):return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)}
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def array_rows(path,key):
 decoder=json.JSONDecoder()
 with path.open() as f:
  buffer=f.read(65536);m=re.search('"'+key+'"'+r'\s*:\s*\[',buffer);assert m;buffer=buffer[m.end():]
  while True:
   buffer=buffer.lstrip()
   if buffer.startswith(','):buffer=buffer[1:].lstrip()
   if buffer.startswith(']'):return
   try:row,end=decoder.raw_decode(buffer)
   except json.JSONDecodeError:
    more=f.read(65536);assert more and len(buffer)<1024**2;buffer+=more;continue
   assert isinstance(row,dict);yield row;buffer=buffer[end:]
def main():
 for p,h in PINS.items():assert sha(ROOT/p)==h,p
 cohort_path=ROOT/'manifests/experiments/remind-component-cohort-v1.json'
 member=next(m for m in json.loads(cohort_path.read_text())['members'] if m['subject']==PERSON)
 assert member['role']=='TRAIN' and member['patient_group']=='ReMIND:050' and member['diagnosis_eligibility_stratum']=='strict_three_label_glioma'
 candidate_path=ROOT/'build/remind-train-public-expansion-candidate-v1/candidate.json'
 candidate=json.loads(candidate_path.read_text())
 used={'ReMIND-002','ReMIND-008','ReMIND-010','ReMIND-015','ReMIND-018','ReMIND-020','ReMIND-025','ReMIND-045'}
 eligible=sorted(r['patient_id'] for r in candidate['all_63_strict_glioma_TRAIN_dispositions'] if r['public_triple_metadata_complete'] and r['patient_id'] not in used)
 assert len(eligible)==17 and eligible[0]==PERSON
 catalog_path=ROOT/'build/remind-cohort-expansion-preparation-v1/idc-remind-series-metadata.json'
 series=[s for s in json.loads(catalog_path.read_text())['series'] if s['PatientID']==PERSON and s['StudyDescription']=='Preop' and s['SeriesDescription'] in KINDS]
 assert len(series)==3 and len({s['StudyInstanceUID'] for s in series})==1 and {s['SeriesDescription'] for s in series}==set(KINDS)
 assert {s['crdc_series_uuid'] for s in series}=={'080373f0-7522-4002-a511-16f30793b709','468104cb-e59b-4ded-9355-3125bad79f90','74529c89-23d3-41a4-b7ec-f6beb90761ec'}
 assert all(s['license_short_name']=='CC BY 4.0' and s['source_DOI']=='10.7937/3rag-d070' for s in series)
 series_map={s['crdc_series_uuid']:s for s in series};objects={};receipts={};bases={}
 for family in ['remind-train-originals-v1','remind-train-seg-v1']:
  base=ROOT/'data/acquisition'/family
  for o in array_rows(base/'source/train-exact-object-manifest.json','objects'):
   if o['series_uuid'] not in series_map:continue
   s=series_map[o['series_uuid']]
   assert o['patient_id']==PERSON and o['role']=='TRAIN' and o['study_description']=='Preop' and o['series_description']==s['SeriesDescription'] and o['series_instance_uid']==s['SeriesInstanceUID']
   assert o['etag']==o['etag_md5'] and re.fullmatch('[a-f0-9]{32}',o['etag_md5']) and o['url']=='https://idc-open-data.s3.amazonaws.com/'+o['key']
   p=PERSON+'/'+o['key'];assert p not in objects;objects[p]=o;bases[p]=base
  for r in array_rows(base/'completion.json','files'):
   if r['path'] not in objects:continue
   assert r['path'] not in receipts;receipts[r['path']]=r
 assert len(objects)==len(receipts)==194
 parent={'cohort':binding(cohort_path),'series_metadata':binding(catalog_path),'source_document':binding(ROOT/'data/acquisition/remind-train-seg-v1/source/tcia-collection.html'),'expansion_candidate':binding(candidate_path)}
 dependencies=[binding(ROOT/p) for p in PINS if '/source/train-exact-object-manifest.json' in p or p.endswith('/completion.json')]
 prov={'structural_t1ce':'Acquired source-described preoperative contrast-enhanced T1 MRI; timing and physical frame await header qualification.','whole_tumor':'Source manual preoperative whole-tumor annotation, not a prescribed resection target; retain complete source positives and support mismatch.','cerebrum':'Source automatic Brainlab preoperative cerebrum, supplied estimated support rather than independent manual truth or physical occupancy. Preserve source zeros, holes and unknown outside-source domain.'}
 flat=[];nested=[];stats=[]
 for s in sorted(series,key=lambda s:list(KINDS).index(s['SeriesDescription'])):
  group=[];uid=s['crdc_series_uuid']
  for p,o in sorted(objects.items()):
   if o['series_uuid']!=uid:continue
   r=receipts[p]
   assert r['status'] in ['byte_verified','existing_byte_verified'] and r['patient_id']==PERSON and o['bytes']==r['bytes'] and r['source_checksum_type']=='MD5' and o['etag_md5']==r['source_checksum'] and re.fullmatch('[a-f0-9]{64}',r['sha256'])
   path=bases[p]/'verified'/p; st=path.lstat()
   assert stat.S_ISREG(st.st_mode) and path.resolve()==path and st.st_size==o['bytes']
   item={'path':str(path),'bytes':o['bytes'],'sha256':r['sha256'],'expected_md5':o['etag_md5'],'series_uuid':uid}
   flat.append(item);group.append({k:v for k,v in item.items() if k!='series_uuid'})
   stats.append({'path':str(path.relative_to(ROOT)),'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns})
  assert len(group)==s['instanceCount']
  kind=KINDS[s['SeriesDescription']]
  nested.append({'kind':kind,'Modality':s['Modality'],'SeriesInstanceUID':s['SeriesInstanceUID'],'StudyInstanceUID':s['StudyInstanceUID'],'series_uuid':uid,'source_description':s['SeriesDescription'],'annotation_provenance':prov[kind],'objects':group})
 assert len(flat)==194 and sum(o['bytes'] for o in flat)==27142222
 source_path=HERE/'bindings/ReMIND-050-source-binding.json'
 save(source_path,{'patient_id':PERSON,'patient_group':member['patient_group'],'role':'TRAIN',**parent,'source_series':series,'objects':flat,'total_objects':194,'total_bytes':27142222,'acquisition_dependencies':dependencies,'stage':'byte_verified_unreviewed','training_admitted':False,'header_or_array_reads_in_preparation':0,'public_only':True,'private_series_included':False,'selection_rule':'Lowest original patient ID among 17 complete public triples not previously prepared in eight TRAIN cases; fixed before headers, no outcome selection.','full_target_and_support_mismatch_policy':'Keep complete S/T/source domains and all failed/unsupported outcomes; no mask clipping, filling or patient replacement.'})
 case_path=HERE/'bindings/ReMIND-050-case.json'
 save(case_path,{'patient_id':PERSON,'patient_group':member['patient_group'],'role':'TRAIN','parent_bindings':{'source_binding':binding(source_path),**parent,'acquisition_dependencies':dependencies},'series':nested,'stage':'byte_verified_unreviewed','case_uncertainties':['Source describes Preop phase; exact timestamps, physical frames and source-SOP linkage await header qualification.','Manual whole tumor is supplied target evidence, not a prescribed resection target.','Automatic Brainlab cerebrum is supplied estimated support, not anatomical truth or physical occupancy.','Native/placed full target counts, any crop losses and all target/support conflicts must remain visible.','Cross-collection identity and pretrained exposure are unknown.','Only public three-series metadata bound here; no headers, pixels or arrays opened in this binding step.']})
 save(HERE/'source-stat-before.json',{'objects':stats,'source_payload_bytes_read':0})
 save(HERE/'public-train-index.json',{'schema':'remind-public-TRAIN-050-header-index-v1','status':'header_preparation_only','subjects':[PERSON],'planned_denominator':1,'cases':[{'patient_id':PERSON,'patient_group':member['patient_group'],'role':'TRAIN','case':binding(case_path),'source_binding':binding(source_path),'objects':194,'bytes':27142222,'public_input_manifest':None,'training_admitted':False}],'old_roles_and_pilots_unchanged':True,'negative_retained_without_replacement':True,'crop_released':False})
 module=ROOT/'src/resectionlab/remind_planning_qc.py';cli=ROOT/'scripts/prepare_remind_planning.py'
 raw=module.read_text();old='"ReMIND-025", "ReMIND-045"),';new='"ReMIND-025", "ReMIND-045", "ReMIND-050"),'
 assert raw.count(old)==1
 staged=HERE/'stage/src/resectionlab/remind_planning_qc.py';staged.parent.mkdir(parents=True,exist_ok=True)
 with staged.open('x') as f:f.write(raw.replace(old,new))
 with (staged.parent/'__init__.py').open('xb') as f:f.write((ROOT/'src/resectionlab/__init__.py').read_bytes())
 target=HERE/'stage/scripts/prepare_remind_planning.py';target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(cli.read_bytes())
 with (HERE/'reader.patch').open('x') as f:f.writelines(difflib.unified_diff(raw.splitlines(True),staged.read_text().splitlines(True),fromfile=str(module.relative_to(ROOT)),tofile=str(staged.relative_to(ROOT))))
 save(HERE/'reader-delta.json',{'canonical_source':binding(module),'staged_source':binding(staged),'only_delta':'Add fixed TRAIN050 to public scope allowlist. Geometry/parser/roles unchanged. Header-only runner; crop-mr-domains gate and planning/admission remain closed for050.','canonical_cli':binding(cli),'staged_cli':binding(target),'canonical_files_modified':False})
 print(json.dumps({'case':binding(case_path),'source_binding':binding(source_path),'public_index':binding(HERE/'public-train-index.json'),'reader':binding(staged)},indent=2))
if __name__=='__main__':main()
