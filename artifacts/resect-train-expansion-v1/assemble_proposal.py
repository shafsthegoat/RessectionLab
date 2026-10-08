"""Assemble metadata-only TRAIN proposal; no scientific file reads."""
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def evidence(p):
 p=ROOT/p
 return {'path':str(p.relative_to(ROOT)),'sha256':digest(p)}
cohort_path=Path('manifests/resect-component-cohort-v1.json'); cohort=json.loads((ROOT/cohort_path).read_text())
assert digest(ROOT/cohort_path)=='4fc51f01e42253d395660d850cda2296a3e7f23e5a5ade80bbd5a3ed411b6a77'
train=[m for m in cohort['members'] if m['role']=='TRAIN']; assert len(train)==14
originals=json.loads((OUT/'originals-metadata.json').read_text())
records={}; complete_folders={}
for path in sorted(OUT.glob('request-*-metadata.json')):
 b=json.loads(path.read_text()); assert b['links']['next'] is None
 for d in b['data']:
  a=d['attributes']; materialized=a.get('materialized_path','')
  if a['kind']=='file' and materialized.startswith('/RESECT-Segmentation/Case'):
   case=materialized.split('/')[2]; complete_folders[case]=evidence(path.relative_to(ROOT))
  if a['name'].endswith('-resection.nii.gz'):
   assert a['name'] not in records
   records[a['name']]=(d,path)
assert len(records)==25
pairs=[]; missing=[]
for member in train:
 case=member['case_id']; assert case in complete_folders
 for phase in ('during','after'):
  name=f'{case}-US-{phase}-resection.nii.gz'
  expected=member[f'cavity_{phase}_label_reported']
  assert (name in records)==expected,(case,phase)
  if not expected:
   missing.append({'patient_group':member['patient_group'],'role':'TRAIN','phase':phase,'status':'deferred_missing_source_annotation','evidence':complete_folders[case],'negative_mask_allowed':False,'acquire_original_for_this_pair':False})
   continue
  d,path=records[name]; a=d['attributes']; hashes=a['extra']['hashes']
  assert a['current_version']==2 and a['kind']=='file' and a['size']>0
  assert re.fullmatch('[0-9a-f]{32}',hashes['md5']) and re.fullmatch('[0-9a-f]{64}',hashes['sha256'])
  original=[x for x in originals if x['person']==member['patient_group'] and x['phase']==phase]; assert len(original)==1
  original=original[0]
  assert re.fullmatch('[0-9a-f]{32}',original['published_md5'])
  original['kind']='original_ultrasound_'+phase+'_resection'
  original['suggested_relative_cache_path']='originals/'+Path(original['path']).name
  latest_url=d['links']['download']; assert latest_url.startswith('https://osf.io/download/') and '?' not in latest_url
  mask={'kind':'human_reviewed_visible_cavity_annotation','name':name,'source_path':a['materialized_path'],'suggested_relative_cache_path':'originals/'+name,'osf_file_id':d['id'],'bytes':a['size'],'published_md5':hashes['md5'],'published_sha256':hashes['sha256'],'file_revision':2,'revision_evidence':'attributes.current_version=2 and hashes/size from the same complete OSF folder catalog response','source_url':latest_url+'?revision=2','source_url_basis':'published OSF download link plus explicit revision selector, following the existing frozen and acquired Case3 revision-2 route; no new payload-route request in this qualification','published_current_download_url':latest_url,'file_metadata_url':d['links']['self'],'version_metadata_url':d['relationships']['versions']['links']['related']['href']+'2/','version_metadata_endpoint_requested_in_this_audit':False,'date_modified':a['date_modified'],'license':'CC-BY-NC-SA-4.0','metadata_evidence':evidence(path.relative_to(ROOT))}
  pilot=case=='Case3' and phase=='during'
  pairs.append({'patient_group':member['patient_group'],'role':'TRAIN','phase':phase,'metadata_status':'qualified','pairing_basis':'Annotation README identifies original RESECT DOI; creator case directory and exact phase filename match original NIFTI inventory. Spatial/anatomical correspondence is not established by filenames.','scientific_status':'existing_pilot_structural_qc_passed_anatomical_review_pending' if pilot else 'not_acquired_or_decoded_in_this_qualification','already_acquired_pilot_pair':pilot,'files':[original,mask],'scientific_payload_bytes':original['bytes']+mask['bytes']})
rights=evidence(Path('artifacts/resect-component-admission-v1/rights-review-02/README.txt'))
assert rights['sha256']=='0a75fef344487ee2649a831919388940f2000bf4cb936979e0da2f73a9aa81a9'
receipts=[json.loads(p.read_text()) for p in sorted(OUT.glob('request-*-receipt.json'))]
state=json.loads((OUT/'request-budget.json').read_text())
summary={'frozen_train_people':14,'people_with_at_least_one_cavity_label':len({p['patient_group'] for p in pairs}),'qualified_pairs':len(pairs),'qualified_original_files':len(pairs),'qualified_mask_files':len(pairs),'during_pairs':sum(p['phase']=='during' for p in pairs),'after_pairs':sum(p['phase']=='after' for p in pairs),'existing_pilot_pairs':sum(p['already_acquired_pilot_pair'] for p in pairs),'remaining_pairs':sum(not p['already_acquired_pilot_pair'] for p in pairs),'remaining_scientific_files':sum(2 for p in pairs if not p['already_acquired_pilot_pair']),'new_people_whose_scientific_payloads_would_be_opened':len({p['patient_group'] for p in pairs if p['patient_group']!='RESECT:Case3'}),'all_original_bytes':sum(p['files'][0]['bytes'] for p in pairs),'all_mask_bytes':sum(p['files'][1]['bytes'] for p in pairs),'all_pair_bytes':sum(p['scientific_payload_bytes'] for p in pairs),'remaining_scientific_bytes':sum(p['scientific_payload_bytes'] for p in pairs if not p['already_acquired_pilot_pair']),'largest_original_bytes':max(p['files'][0]['bytes'] for p in pairs),'largest_mask_bytes':max(p['files'][1]['bytes'] for p in pairs),'missing_label_slots':len(missing),'failed_metadata_requests':sum(r['outcome']!='metadata_retained' for r in receipts),'metadata_response_bytes':state['response_bytes'],'metadata_requests':state['requests'],'metadata_request_elapsed_seconds':round(sum(r['elapsed_seconds'] for r in receipts),6),'metadata_sweep_wall_seconds':round(max(r['started_unix']+r['elapsed_seconds'] for r in receipts)-state['started_unix'],6),'scientific_body_bytes_requested_or_read':0,'training_updates':0,'recorded_rl_transitions':0}
proposal={'schema':'resect-train-expansion-qualification-proposal-v1','created_at':datetime.now(timezone.utc).isoformat(),'status':'metadata_qualified_proposal_only_not_a_payload_or_training_admission','existing_cohort':evidence(cohort_path),'existing_pilot':evidence(Path('manifests/resect-case3-cavity-pilot-v1.json')),'selection':'All reported cavity-label phases for existing frozen TRAIN people, no outcome-based selection, reassignment, or substitutes','original_release':{'doi':cohort['image_source']['doi'],'version':1,'rights':'CC-BY-4.0','provider_metadata':evidence(Path('artifacts/mechanics/resect-case4-acquisition-metadata-v1/provider-dataset-metadata.json')),'inventory':evidence(Path(cohort['image_source']['inventory'])),'uncompressed_inventory_sha256':cohort['image_source']['uncompressed_inventory_sha256'],'published_checksum_type':'MD5 inventory fixity, never ETag; original published SHA256 unavailable'},'annotation_release':{'doi':cohort['annotation_source']['doi'],'version':cohort['annotation_source']['release'],'rights':'CC-BY-NC-SA-4.0','rights_evidence':rights,'intended_use':'noncommercial component research only; no commercial clearance','rights_obligations':'Attribution and citation of original and annotation publications, noncommercial use, share-alike terms for adapted annotation material; README explicitly prohibits financial benefits from annotation dataset distribution','annotation_chain':'manual contours + morphological interpolation + expert refinement/review; not every voxel hand-drawn; no learned teacher reported in documented protocol','semantic_limit':'Visible dark ultrasound cavity can omit blood-filled regions; not total removed tissue, physical force, clinical outcome, or surgical reward'},'summary':summary,'qualified_pairs':pairs,'deferred_missing_annotations':missing,'failed':[],'protected_members_not_descended_into':[{'patient_group':m['patient_group'],'role':m['role']} for m in cohort['members'] if m['role']!='TRAIN'],'metadata_scope':{'requests':'Two public root/family folder catalogs, then only 14 frozen TRAIN case folder catalogs; pagination complete for every folder','public_root_case_names_visible':True,'scientific_payload_requests':False,'scientific_payload_local_reads':False,'signed_query_credentials_retained':False,'automatic_retries':0,'redirects_followed':0,'limits':{'requests':30,'wall_seconds':300,'response_bytes':4194304},'request_journal':evidence(Path('build/resect-train-expansion-qualification-v1/request-budget.json'))},'pilot_qc_evidence':evidence(Path('build/resect-case3-pair-qc-independent-review-v1/verification.json')),'first_implementation_step':'Freeze a separate TRAIN-only manifest from these 25 exact pairs and three explicit missing-label slots, then add a manifest-bound batch intake using the reviewed native NIRD trust transport and OSF hash-bound route checks. Keep the Case3 pilot and cohort unchanged. Verify/reuse the existing pilot pair; per-file attempt receipts, byte/deadline caps, stop on failure, exclusive partial/publication and no silent retry/overwrite. Acquire only the 48 remaining declared files (548605591 bytes), then separately perform structural and anatomical QC before any component fitting.','implementation_blocker':'Current native NIRD function accepts only SOURCES[1] (Case3 during); an explicit TRAIN manifest whitelist is needed before any expansion transfer. Do not loosen host-only URL authorization.','admission':{'scientific_expansion_acquired':False,'spatially_qualified_expansion':False,'anatomically_reviewed_expansion':False,'training_admitted':False,'clinical_validation':False}}
(OUT/'manifest-proposal.json').write_text(json.dumps(proposal,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2)); print('proposal_sha256',digest(OUT/'manifest-proposal.json'))
