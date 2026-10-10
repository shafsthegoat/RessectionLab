"""Final-manifest metadata follow-up; never opens referenced array/image paths."""
import hashlib,json,resource,signal,time
from pathlib import Path
import audit_saved as a
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s metadata follow-up')));signal.alarm(30)
P=a.P;R=a.R;need=a.need;read=a.read
m=read(P/'public-inputs-v1/ReMIND-013-public-inputs.json','841261035a7d7be5c55984e717b8d65b712065a708e9260231ebcd49cc4d700e')
i=read(P/'public-inputs-v1/public-select-index.json','a5a7ed8b504b3f768666cdec4c77908d680ff242eb157afb0f377b95105dc389')
c=read(R/'ReMIND-013-crop-mr-domains/conversion-result.json','bf9174ee066f2ce744a4b3ace1ae048954733981c8126e461c15a75d7c89a057')
q=read(R/'ReMIND-013-saved-domain-review/saved-domain-review.json','6479dd6df8a9d81a05438e481533b2083a7d95cb5889f93b3c4dea038f6d6a7d')
v=read(P/'root-visual-review.json','1471e4e32b4fa09d7565664f279489d4b60a8eab149471e82475ee6539a58a86')
prior=read(Path(i['original_select_ledger']['path']),i['original_select_ledger']['sha256'])
cohort=read(a.ROOT/'manifests/experiments/remind-component-cohort-v1.json',m['source_bindings']['cohort_sha256'])
read(P/'prepare_manifest.py',source=True)
need(m['schema']=='remind-fixed-SELECT-partial-domain-public-inputs-v1','explicit partial-domain schema')
need(m['patient_id']=='ReMIND-013' and m['patient_group']=='ReMIND:013' and m['role']=='SELECT','person/role')
need(next(x for x in cohort['members'] if x['subject']=='ReMIND-013')['role']=='SELECT','unchanged cohort role')
need(m['source_domain_condition']=='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain' and m['task_condition']=='PARTIAL_TARGET_PROGRESS','explicit simulated condition')
need(m['public_only'] and not any(m[k] for k in ('private_evaluation_files_included','training_admitted','actor_inputs_admitted','anatomical_accuracy_reviewed','clinical_suitability','whole_tumor_is_prescribed_resection_target','public_support_domain_fully_covered')),'scope flags')
files={'image':('MR_native_crop.npy','float32'),'supplied_support':('cerebrum_source_label.npy','uint8'),'supplied_whole_tumor':('whole_tumor_source_label.npy','uint8'),'whole_tumor_domain':('whole_tumor_source_grid_domain.npy','uint8'),'supplied_support_domain':('cerebrum_source_grid_domain.npy','uint8')}
need(set(m['input_files'])==set(files),'exact five public inputs')
for key,(name,dtype) in files.items():
 expected={'path':str(R/'ReMIND-013-crop-mr-domains'/name),'dtype':dtype,**c['artifacts'][name]}
 need(m['input_files'][key]==expected,'array path/dtype/bytes/hash join '+key)
for mk,ck in [('shape_xyz','shape_xyz'),('affine_ras_mm','planning_derived_affine_ras_mm'),('source_MR_crop_affine_ras_mm','MR_native_crop_affine_ras_mm'),('reindex_policy','reindex_policy')]:need(m[mk]==c[ck],'frame exact '+mk)
for mk,qk in [('public_label_resampling','label_summaries'),('source_centre_coverage','source_centre_coverage'),('expanded_preparation_provenance','expanded_preparation_provenance')]:need(m[mk]==q[qk],'saved review join '+mk)
need(m['public_source_domain_counts']=={k:q['saved_counts'][k] for k in ('support_domain','target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain')},'Ds and full target partition')
need(m['public_target_support_consistency']=={'whole_tumor_positive_voxels':35260,'whole_tumor_positive_outside_supplied_support':33681},'full T unchanged')
for name in ('conversion','saved_review','root_visual_review'):
 b=m['source_bindings'][name];p=Path(b['path']);read(p,b['sha256']);need(a.seen[str(p.relative_to(a.ROOT))]['bytes']==b['bytes'],'binding length '+name)
need(m['source_bindings']['case_sha256']==q['case_sha256']==c['case_sha256'] and m['source_bindings']['headers_sha256']==q['header_snapshot_sha256']==c['header_snapshot_sha256'],'case/header ancestry')
need(m['source_bindings']['saved_array_review_sha256']==m['source_bindings']['saved_review']['sha256'],'saved array review link')
need(m['root_visual_observation']==v and len(v['image_files'])==3 and v['patient_id']=='ReMIND-013' and v['role']=='SELECT','limited root visual note exact')
for x in v['image_files']:need(x['viewed'] and c['artifacts'][Path(x['path']).name]['sha256']==x['sha256'],'recorded PNG pins; no pixels opened')
for k in ('establishes_segmentation_accuracy','establishes_all_volume_alignment','establishes_physical_occupancy','establishes_clinical_validity','planning_admission','private_data_viewed'):need(v[k] is False,'visual scope '+k)
for k in ('cerebrum','whole_tumor'):
 ca=c['annotations'][k];need(m['source_annotation_ancestry'][k]=={'segments':ca['ancestry']['segments'],'source_description':ca['correspondence']['source_description'],'correspondence':ca['correspondence']['correspondence'],'explicit_source_SOP_links_present':ca['correspondence']['explicit_SOP_refs_present']},'annotation ancestry '+k)
need(m['source_availability']=={'basis':'retrospective_annotation_assisted_source_preop','acquired_at':None,'annotation_available_at':None},'no prospective availability inference')
need(i['planned_case_denominator']==2 and len(i['cases'])==2 and i['max_optimizer_updates']==0 and i['actual_policy_runs']==0 and not i['private_reference_loaded'] and not i['split_changes'],'two case denominator/no use')
need(i['cases'][1]==prior['cases'][1] and i['cases'][1]['patient_id']=='ReMIND-037' and i['cases'][1]['status']=='HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT' and i['cases'][1]['path'] is None,'037 unchanged complete object')
r=i['cases'][0];need(r['patient_id']=='ReMIND-013' and r['sha256']==a.seen[str((P/'public-inputs-v1/ReMIND-013-public-inputs.json').relative_to(a.ROOT))]['sha256'] and r['full_target_denominator_voxels']==35260 and r['supported_target_positive_voxels']==1579 and not r['anatomical_accuracy_pass'],'013 manifest ledger link')
need(a.ROOT/r['path']==P/'public-inputs-v1/ReMIND-013-public-inputs.json','ledger path')
for p,b in list(a.seen.items()):read(a.ROOT/p,b['sha256'],source=True)
result={'status':'PASS_final_expanded_manifest_saved_metadata','checks':a.checks,'files_checked':len(a.seen),'manifest_sha256':r['sha256'],'select_index_sha256':'a5a7ed8b504b3f768666cdec4c77908d680ff242eb157afb0f377b95105dc389','root_visual_sha256':'1471e4e32b4fa09d7565664f279489d4b60a8eab149471e82475ee6539a58a86','full_target_voxels':35260,'unknown_support_domain_voxels':176294,'SELECT037_HOLD_object_unchanged':True,'source_prepared_only':True,'audit_seconds':time.monotonic()-start,'audit_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Saved JSON/source byte comparisons only; no arrays/images/checkpoints/DICOM/models/native calls. Root visual observation retained without independent pixel inspection.'}
a.save('manifest-audit-result.json',result);a.save('manifest-input-index.json',a.seen);print(json.dumps(result,sort_keys=True))
