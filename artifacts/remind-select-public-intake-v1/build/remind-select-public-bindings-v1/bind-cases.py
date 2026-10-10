"""Join byte receipts and frozen SELECT/source metadata; stat only, no DICOM opens."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
BASE=ROOT/'data/acquisition/remind-select-public-v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(p,absolute=False): return {'path':str(p if absolute else p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
def save(p,value):
 with p.open('x') as f: json.dump(value,f,indent=2); f.write('\n')
def main():
 assert sha(BASE/'declaration.json')=='4618ccc76d9b28e382fa9205392aadf0bac88b20cc5fe4da9e10afcbbbf66a43'
 D=json.loads((BASE/'declaration.json').read_text())
 for p,h in D['metadata_pins'].items(): assert sha(ROOT/p)==h
 assert sha(BASE/'source/exact-object-manifest.json')==D['manifest_sha256']
 M=json.loads((BASE/'source/exact-object-manifest.json').read_text()); C=json.loads((BASE/'completion.json').read_text())
 assert C['status']=='all_bytes_verified' and C['verified_files']==372 and C['verified_bytes']==61657062 and C['unresolved_files']==0
 assert C['declaration_sha256']==sha(BASE/'declaration.json') and C['manifest_sha256']==D['manifest_sha256']
 lookup={r['path']:r for r in C['files']}; assert len(lookup)==372 and set(lookup)=={e['path'] for e in M['objects']}
 parent={k:binding(ROOT/p) for k,p in {'cohort':'manifests/experiments/remind-component-cohort-v1.json','pilot':'manifests/experiments/remind-planning-pilot-v1.json','series_metadata':'build/remind-cohort-expansion-preparation-v1/idc-remind-series-metadata.json','source_document':'data/acquisition/remind-train-seg-v1/source/tcia-collection.html'}.items()}
 dependencies=[binding(BASE/'source/exact-object-manifest.json'),binding(BASE/'completion.json'),binding(BASE/'declaration.json')]
 provenance={
  'structural_t1ce':'Acquired source-described preoperative contrast-enhanced T1 MRI. Case-specific DICOM frame/timing and anatomy remain unreviewed.',
  'whole_tumor':'Source manual preoperative whole-tumor annotation created during planning; not an explicit prescribed resection target. Case-specific segment algorithm and source SOP linkage remain unreviewed.',
  'cerebrum':'Source automatic Brainlab preoperative annotation created during planning; supplied estimated support pending intended-use QC, not independent manual truth or reviewed pial/intact-normal-tissue geometry.'}
 kinds={'3D_AX_T1_postcontrast':'structural_t1ce','tumor seg - MR ref: 3D_AX_T1_postcontrast':'whole_tumor','cerebrum seg - MR ref: 3D_AX_T1_postcontrast':'cerebrum'}
 summary=[]
 for person in ['ReMIND-013','ReMIND-037']:
  assert next(m for m in M['members'] if m['subject']==person)['role']=='SELECT'
  series=[s for s in M['source_series'] if s['PatientID']==person]; objects=[]; nested=[]
  assert len(series)==3 and {s['SeriesDescription'] for s in series}==set(kinds) and len({s['StudyInstanceUID'] for s in series})==1
  for s in sorted(series,key=lambda s:list(kinds).index(s['SeriesDescription'])):
   group=[]; es=[e for e in M['objects'] if e['patient_id']==person and e['series_uuid']==s['crdc_series_uuid']]
   assert len(es)==s['instanceCount']
   for e in es:
    r=lookup[e['path']]; p=BASE/'verified'/e['path']; st=p.stat()
    assert p.is_file() and not p.is_symlink() and r['status'] in {'byte_verified','existing_byte_verified'}
    assert r['role']=='SELECT' and r['patient_id']==person and r['bytes']==e['bytes']==st.st_size and r['source_checksum']==e['expected_md5']
    assert r['publication_stat']=={'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    item={'path':str(p),'bytes':e['bytes'],'sha256':r['sha256'],'expected_md5':e['expected_md5'],'series_uuid':e['series_uuid']}
    objects.append(item); group.append({k:v for k,v in item.items() if k!='series_uuid'})
   kind=kinds[s['SeriesDescription']]
   nested.append({'kind':kind,'Modality':s['Modality'],'SeriesInstanceUID':s['SeriesInstanceUID'],'StudyInstanceUID':s['StudyInstanceUID'],'series_uuid':s['crdc_series_uuid'],'source_description':s['SeriesDescription'],'annotation_provenance':provenance[kind],'objects':group})
  flat={'patient_id':person,'patient_group':'ReMIND:'+person.split('-')[1],'role':'SELECT',**parent,'source_series':series,'acquisition_dependencies':dependencies,'objects':objects,'total_objects':len(objects),'total_bytes':sum(o['bytes'] for o in objects),'stage':'byte_verified_unreviewed','current_action':'Metadata-only preparation for separately authorized SELECT public three-series QC; no private/evaluation reader release implied','anatomical_admission':False,'training_admitted':False,'source_prior_cavity_label_present':False,'private_annotation_geometry_accessed':False,'exact_preoperative_timestamps':None,'whole_tumor_is_not_prescribed_resection_target':True,'headers_or_arrays_opened':False,'missing_private_reference_intentional':True,'support_may_encode_ventricular_voids':True,'cross_dataset_or_pretrained_overlap':'unknown; no independent external-transfer claim'}
  flatp=HERE/(person+'-source-binding.json'); save(flatp,flat)
  case={'patient_id':person,'patient_group':flat['patient_group'],'role':'SELECT','parent_bindings':{'source_binding':binding(flatp,True),**parent,'acquisition_dependencies':dependencies},'series':nested,'stage':'byte_verified_unreviewed','case_uncertainties':['Only public MRI/manual whole-tumor/automatic cerebrum are in this case; ventricular/private annotations and EVAL067 remain unopened and unacquired by this intake.','Metadata source says preoperative; exact timestamps, DICOM source linkage/frames, geometry/coverage and anatomy remain unreviewed.','Supplied whole tumor is not a prescribed surgical resection target.','Automatic Brainlab cerebrum is supplied estimated support; it may encode ventricular void information and is not independent manual truth.','No prior-cavity label is listed in source metadata; this is not proof of intact normal tissue.','All derivatives preserve SELECT; no policy training or final evaluation admission follows from byte verification.','Cross-dataset aliases and pretrained exposure remain unknown.']}
  casep=HERE/(person+'-case.json'); save(casep,case)
  summary.append({'patient_id':person,'role':'SELECT','case':binding(casep),'source_binding':binding(flatp),'objects':len(objects),'bytes':flat['total_bytes'],'series':[{'kind':s['kind'],'series_uuid':s['series_uuid'],'objects':len(s['objects']),'bytes':sum(o['bytes'] for o in s['objects'])} for s in nested]})
 report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'all_public_SELECT_bytes_verified_no_QC','cases':summary,'completion':binding(BASE/'completion.json'),'declaration':binding(BASE/'declaration.json'),'manifest':binding(BASE/'source/exact-object-manifest.json'),'total_objects':372,'total_bytes':61657062,'unresolved_files':0,'header_or_array_reads':0,'private_or_eval_payload_requests':0,'training_admitted':False,'metadata_preparation_note':'First official listing included an exact zero-byte S3 directory marker; initial preparation assertion stopped before payload. Saved TLS response was reused with explicit exact-marker exclusion. No payload attempt/history reset.'}
 save(HERE/'completion-summary.json',report); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
