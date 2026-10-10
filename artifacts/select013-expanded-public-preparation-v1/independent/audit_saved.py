"""Small saved-JSON/source audit. Does not open arrays, images or source DICOM."""
import hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
P=ROOT/'build/select013-expanded-public-preparation-v1'; R=P/'actual-domain-qc-v1'
seen={}; checks=0
def need(ok,msg):
 global checks
 checks+=1
 if not ok:raise AssertionError(msg)
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>256*1024**2:raise MemoryError('metadata audit cap')
def read(path,pin=None,source=False):
 path=Path(path);need(path.is_relative_to(ROOT) and '..' not in path.parts and path.suffix in ('.json','.py'),'metadata/source only')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  info=os.fstat(fd);need(stat.S_ISREG(info.st_mode) and 0<info.st_size<4*1024**2,'regular bounded file')
  with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(4*1024**2+1)
 finally:os.close(fd)
 need(len(data)==info.st_size,'bounded read');sha=hashlib.sha256(data).hexdigest();key=str(path.relative_to(ROOT))
 need(pin is None or pin==sha,'exact evidence pin');need(key not in seen or seen[key]['sha256']==sha,'stable repeated input')
 seen[key]={'sha256':sha,'bytes':len(data)}
 return data if source else json.loads(data)
def near(a,b,msg):need(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),msg)
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def main():
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved audit')));signal.alarm(30)
 receipt=read(R/'receipt.json','5ce0cfb3af6ff49289de9d2283b03f7e00e4517cf5bb92524a674785db6e8f6a')
 declaration=read(R/'declaration.json');release=read(Path(declaration['release_path']),declaration['release_sha256'])
 prepared=read(P/'prepared-release.json','80580dee055d86e10a6d59e525385e069f16b701b4efac3a1508f7fa3859ed80')
 need(release['execution_released'] is True and {**release,'execution_released':False}==prepared,'only released Boolean delta')
 need(release['source_index']==declaration['source_index'] and declaration['caps']==receipt['caps']==release['caps'],'release/cap/source joins')
 index=read(ROOT/release['source_index']['path'],release['source_index']['sha256'])
 for row in index['files']:
  data=read(ROOT/row['path'],row['sha256'],source=True);need(len(data)==row['bytes'],'source/metadata extent')
 need(receipt['fatal'] is None and receipt['planned_subjects']==['ReMIND-013'] and len(receipt['outcomes'])==2,'one case/two phases complete')
 need(receipt['elapsed_seconds']<330 and receipt['training_admitted'] is False and receipt['roles_changed'] is False,'no training/role changes')
 expected=['crop-mr-domains','saved-domain-review']
 for row,phase in zip(receipt['outcomes'],expected):
  need(row['phase']==phase and row['subject']=='ReMIND-013' and row['role']=='SELECT' and row['phase_complete'] and row['exit_code']==0,'phase terminal')
  need(row['reaped'] and not row['cleanup_errors'] and not row['remaining_owned_pids'] and row['result_guard_error'] is None and row['stop_reason'] is None,'clean owned phase')
  need(row['case_elapsed_seconds']<300 and row['sampled_peak_rss_bytes']<3*1024**3 and row['case_output_bytes']<256*1024**2 and row['final_phase_log_bytes']<4*1024**2,'phase resources')
 c=read(R/'ReMIND-013-crop-mr-domains/conversion-result.json','bf9174ee066f2ce744a4b3ace1ae048954733981c8126e461c15a75d7c89a057')
 a=read(R/'ReMIND-013-saved-domain-review/saved-domain-review.json','6479dd6df8a9d81a05438e481533b2083a7d95cb5889f93b3c4dea038f6d6a7d')
 need(a['conversion_receipt_sha256']==receipt['outcomes'][0]['result_sha256'] and receipt['outcomes'][1]['result_sha256']==seen[str((R/'ReMIND-013-saved-domain-review/saved-domain-review.json').relative_to(ROOT))]['sha256'],'phase result links')
 need(c['patient_id']==a['patient_id']=='ReMIND-013' and c['role']==a['role']=='SELECT','fixed person/role')
 need(c['executing_script_sha256']=='565843dac7efeef29640bb1690a7f1f167d38310d8b7e771babad5302571d796' and a['executing_script_sha256']=='961488206f4edfea05914f90cdda44b3c690a9da7208e5ac81e461100b9b00e1','reviewed source identity')
 for value in (c,a):need(value['public_only'] and value['private_reference_loaded'] is False and value['training_admitted'] is False and value['optimizer_updates_performed']==0,'public/no training boundary')
 need(c['objects_verified']==119 and c['MR_pixel_source_objects']==117 and c['source_file_returned_bytes']==23053728,'117 MRI+2 public SEG accounting')
 need(c['public_crop_start_MR']==[61,70,23] and c['shape_xyz']==a['shape_xyz']==[131,154,117],'expanded native sampling grid')
 need(c['selected_MR_pixels_match_independent_raw_bytes'] and c['reindex_policy']['image_interpolation']=='none','saved source-byte MRI comparison')
 old=read(ROOT/'build/remind-select-public-preparation-v1/public-select-inputs-v1/ReMIND-013-public-inputs-v1.json','563963e7cf6ae7c1dff9f163db3208bf1af89b5611bc20e76a84c442a2b407a8')
 diag=read(ROOT/'build/select013-source-domain-diagnostic-v1/attempt-03/result.json','531e60692158457077d1a11b951bb5b217b8333abd269968a651eca474828445')
 p=a['expanded_preparation_provenance'];counts=a['saved_counts'];n=math.prod(a['shape_xyz'])
 need(p['old_overlap_file_sha256']=={k:v['sha256'] for k,v in old['input_files'].items()},'all four old array overlap pins')
 need(p['fixed_opening_from_bound_diagnostic']==diag['unchanged_rule'] and p['preentry_point_bounds_from_bound_diagnostic']==diag['preentry_distal_point_bounds'],'exact diagnostic localization')
 need(p['old_sources_or_failed_attempt_modified'] is False and p['new_access_or_route_admitted'] is False,'original failure preserved')
 need(counts['support']==745592 and counts['target']==35260 and counts['support_domain']==2184064 and n==2360358,'full counts')
 need(counts['support']-p['old_support_placed_voxels']==p['restored_support_placed_voxels']==69,'69 restored placed support cells')
 need(n-counts['support_domain']==a['support_unknown_domain_voxels']==c['public_support_source_domain_unknown_voxels']==176294,'unknown domain count')
 need(counts['target']==sum(counts[k] for k in ('target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain')),'three-way target partition')
 need((counts['target_in_known_support_positive'],counts['target_in_known_support_zero'],counts['target_in_unknown_support_domain'])==(1579,33681,0),'target partition values')
 need(counts['target_outside_support']==counts['target_in_known_support_zero']+counts['target_in_unknown_support_domain'],'target outside support')
 for kind,key in [('cerebrum','support'),('whole_tumor','target')]:
  annotation=c['annotations'][kind];placement=annotation['placement'];summary=a['label_summaries'][kind]
  need(placement['source_positive_centres_outside_target_grid']==summary['source_positive_centres_cropped_from_bound_execution']==0,'zero source-positive crop loss')
  need(annotation['source_samples_equal'] and annotation['placed_positive_voxels']==placement['resampled_positive_voxels']==summary['saved_positive_voxels']==counts[key],'source/placed count join')
  need(summary['source_positive_voxels_from_bound_execution']==placement['source_positive_voxels'],'native count retained')
  near(summary['volume_change_percent'],100*(summary['saved_positive_volume_mm3']/summary['source_positive_volume_mm3_from_bound_execution']-1),'volume change arithmetic')
 need(a['full_coverage_factory_compatible'] is False and a['public_support_domain_fully_covered'] is False and a['public_input_manifest'] is None and not a['actor_inputs_admitted'],'no full-coverage or actor admission')
 need(a['all_saved_artifact_hashes_verified'] and a['all_source_domain_voxels_world_checked'],'saved reviewer checks recorded')
 need(len(a['visual_review']['views'])==3 and a['visual_review']['expert_anatomy_review'] is False,'root visual scope not invented')
 for path,record in list(seen.items()):read(ROOT/path,record['sha256'],source=True)
 summary={'status':'PASS_saved_expanded_SELECT013_metadata_audit','checks':checks,'files_checked':len(seen),
  'receipt_sha256':'5ce0cfb3af6ff49289de9d2283b03f7e00e4517cf5bb92524a674785db6e8f6a',
  'release_sha256':declaration['release_sha256'],'source_index_sha256':release['source_index']['sha256'],
  'conversion_sha256':receipt['outcomes'][0]['result_sha256'],'saved_review_sha256':receipt['outcomes'][1]['result_sha256'],
  'elapsed_seconds':receipt['elapsed_seconds'],'phase_sampled_peak_RSS_bytes':[x['sampled_peak_rss_bytes'] for x in receipt['outcomes']],
  'saved_counts':counts,'unknown_support_domain_voxels':176294,'restored_placed_support_voxels':69,'full_target_voxels':35260,
  'old_overlap_hashes':p['old_overlap_file_sha256'],'unchanged_aperture_diagnostic':p['fixed_opening_from_bound_diagnostic'],
  'scope':'Metadata/source hashes and saved arithmetic only. Recorded array/source-sample checks are authenticated evidence, not independently rerun. No arrays/DICOM/images/native/model/checkpoint reads. No anatomy, route, clinical or inference admission; original failed attempt preserved.',
  'audit_seconds':time.monotonic()-start,'audit_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 save('audit-result.json',summary);save('input-index.json',seen);print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
