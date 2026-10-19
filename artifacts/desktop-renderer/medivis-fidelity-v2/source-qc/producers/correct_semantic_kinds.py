"""Source-class metadata correction only; never changes scan, annotation, or binary arrays."""
from pathlib import Path
import json,copy,hashlib,shutil
from resectionlab.patient_anatomy import PatientImagingCollection,file_hash
R=Path(__file__).resolve().parent;Q=R.parents[1]/'artifacts/desktop-renderer/medivis-fidelity-v2/source-qc';C=R/'native-cta'
c=PatientImagingCollection(C)
before=copy.deepcopy(c.document);before_hash=file_hash(C/'patient-anatomy.json');before_snapshot=c.identity()
def binary_hashes():
 return {str(p.relative_to(C)):file_hash(p) for parent in ['arrays','originals'] for p in sorted((C/parent).rglob('*')) if p.is_file()}
protected_before=binary_hashes()
assert c.layer('vessels')['kind']=='vessels' and c.layer('venous-vessels')['kind']=='vessels'
c.layer('vessels')['kind']='artery';c.layer('venous-vessels')['kind']='vein'
for lid,values in [('vessels','1–34'),('venous-vessels','35–40')]:
 c.layer(lid)['provenance']['semantic_kind_basis']=f'Kind taken from publisher categorical class identities (supplied values {values}); metadata correction only, not anatomy repair or local model inference.'
c.save('Correct grouped source-label semantic kinds to artery/vein from publisher classes; arrays, masks and geometry unchanged')
protected_after=binary_hashes();assert protected_before==protected_after
for category in ['series','layers']:
 for old,new in zip(before[category],c.document[category],strict=True):
  assert old['id']==new['id'] and old['data']==new['data'] and old['affine']==new['affine']
  if category=='series':assert old['original']==new['original']
for row in c.document['series']+c.document['layers']:c.array(row['data'])
# Keep the one-case producer faithful to the final metadata without re-importing.
p=R/'package_native.py';s=p.read_text();s=s.replace("source_series='cta',kind='vessels',role='critical'","source_series='cta',kind='artery' if layer_id=='vessels' else 'vein',role='critical'")
s=s.replace("'voxels':int(mask.sum()),'components26':int(n)","'kind':row['kind'],'voxels':int(mask.sum()),'components26':int(n)")
p.write_text(s)
# Update the current source QC; earlier desktop captures remain evidence of their earlier attempt.
p=R/'native-cta-qc.json';qc=json.loads(p.read_text());qc['native_manifest_sha256']=file_hash(C/'patient-anatomy.json');qc['snapshot_id']=c.identity()
for row in c.document['layers']:qc['layers'][row['id']]['kind']=row['kind']
qc['metadata_correction']={'change':'vessels.kind artery; venous-vessels.kind vein','basis':'Publisher categorical class definitions; no inferred anatomical classification','source_and_binary_arrays_unchanged':True,'prior_manifest_sha256':before_hash,'prior_snapshot_id':before_snapshot}
p.write_text(json.dumps(qc,indent=2)+'\n');shutil.copyfile(p,Q/'topbrain-cta001-native-qc.json')
p=Q/'TOPBRAIN-SOURCE-NOTICE.md';s=p.read_text().replace('This pilot is restricted to local noncommercial research/display evaluation.','This pilot uses the data for noncommercial research/display evaluation; the license does not impose a local-only or no-publication restriction.')
s=s.replace('All components are retained; no interpolation, gap filling, model inference, registration, or cross-patient fusion was performed.','The grouped layer semantic kinds are `artery` and `vein`, respectively, matching the publisher categorical identities. This is a metadata correction, not anatomy repair or our inference. All components are retained; no interpolation, gap filling, model inference, registration, or cross-patient fusion was performed.')
p.write_text(s)
record={'schema':'topbrain-cta001-semantic-metadata-correction-v1','previous_manifest_sha256':before_hash,'manifest_sha256':file_hash(C/'patient-anatomy.json'),'previous_snapshot_id':before_snapshot,'snapshot_id':c.identity(),'native_manifest':str(C/'patient-anatomy.json'),'changed_fields':{'vessels.kind':['vessels','artery'],'venous-vessels.kind':['vessels','vein']},'source_and_binary_array_hashes_before':protected_before,'source_and_binary_array_hashes_after':protected_after,'all_protected_hashes_unchanged':True,'native_affines_unchanged':True,'source_label_values_and_grouped_masks_unchanged':True,'new_inference':False,'new_registration':False,'meaning':'Semantic metadata from publisher annotation class definitions; not anatomy repair.'}
(Q/'topbrain-cta001-semantic-metadata-correction.json').write_text(json.dumps(record,indent=2)+'\n')
p=Q/'topbrain-cta001-reproducibility.json';rp=json.loads(p.read_text())
for row in rp['scripts']:row['sha256']=file_hash(row['path'])
rp['scripts'].append({'path':str(R/'correct_semantic_kinds.py'),'sha256':file_hash(R/'correct_semantic_kinds.py')})
rp['current_manifest']={'path':str(C/'patient-anatomy.json'),'sha256':record['manifest_sha256'],'snapshot_id':record['snapshot_id']}
rp['metadata_correction_receipt']={'path':str(Q/'topbrain-cta001-semantic-metadata-correction.json'),'sha256':file_hash(Q/'topbrain-cta001-semantic-metadata-correction.json')}
p.write_text(json.dumps(rp,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['native_manifest','manifest_sha256','snapshot_id','all_protected_hashes_unchanged']},indent=2))
