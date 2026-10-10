"""Build prospective fixed-TRAIN public manifests from saved JSON only; no array access."""
from pathlib import Path
import argparse,hashlib,json
SUBJECTS=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
CONDITION='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
COHORT='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
FILES={'image':('MR_native_crop.npy','float32'),'supplied_support':('cerebrum_source_label.npy','uint8'),
 'supplied_whole_tumor':('whole_tumor_source_label.npy','uint8'),'whole_tumor_domain':('whole_tumor_source_grid_domain.npy','uint8'),
 'supplied_support_domain':('cerebrum_source_grid_domain.npy','uint8')}
def need(ok,reason):
 if not ok:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(record):
 p=Path(record['path']);need(p.suffix=='.json','JSON metadata only');raw=p.read_bytes()
 need(len(raw)<=2*1024**2 and sha(raw)==record['sha256'],'Bound metadata changed')
 return json.loads(raw)
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def prepare(index,output):
 cohort=read(index['cohort']);need(index['cohort']['sha256']==COHORT,'Original cohort required')
 visual=read(index['visual']);need(visual['planning_admission'] is False,'Visual review cannot imply planning admission')
 need(tuple(x['patient_id'] for x in index['cases'])==SUBJECTS,'Exact fixed four-case order required')
 output.mkdir(exist_ok=False);rows=[]
 for entry in index['cases']:
  subject=entry['patient_id'];c=read(entry['conversion']);q=read(entry['saved_review'])
  member=[m for m in cohort['members'] if m['subject']==subject]
  need(len(member)==1 and member[0]['role']=='TRAIN' and member[0]['diagnosis_eligibility_stratum']=='strict_three_label_glioma','Frozen eligible TRAIN role')
  need(c['patient_id']==q['patient_id']==subject and c['role']==q['role']=='TRAIN','Saved case role join')
  need(c['case_sha256']==q['case_sha256'] and c['header_snapshot_sha256']==q['header_snapshot_sha256']
   and q['conversion_receipt_sha256']==entry['conversion']['sha256'],'Saved input provenance join')
  need(c['public_only'] is q['public_only'] is True and c['private_reference_loaded'] is q['private_reference_loaded'] is False
   and c['training_admitted'] is q['training_admitted'] is False,'Public-only untrained source')
  need(q['all_saved_artifact_hashes_verified'] is True and q['all_source_domain_voxels_world_checked'] is True
   and q['public_support_domain_fully_covered'] is False and c['selected_MR_pixels_match_independent_raw_bytes'] is True,'Prior source/domain/sample audit required')
  need(c['reindex_policy']['source_MR_samples_preserved'] is True and c['reindex_policy']['original_MR_affine_overwritten'] is False,'Original MRI samples/frame retained')
  need(c['shape_xyz']==q['shape_xyz'],'Saved shape join')
  for key in ('cerebrum','whole_tumor'):
   need(q['label_summaries'][key]['source_positive_centres_cropped_from_bound_execution']==0,'No source-positive crop loss')
  views=[v for v in visual['image_files'] if subject+'-crop-mr-domains/' in v['path']]
  need(len(views)==3 and all(v['viewed'] is True and c['artifacts'][Path(v['path']).name]['sha256']==v['sha256'] for v in views),'Three exact root-viewed overlays required')
  directory=Path(entry['conversion']['path']).resolve().parent
  inputs={key:{'path':str(directory/name),'dtype':dtype,**c['artifacts'][name]} for key,(name,dtype) in FILES.items()}
  counts=q['saved_counts'];need(sum(counts[k] for k in ('target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain'))==counts['target'],'Full placed T partition')
  manifest={'schema':'remind-fixed-TRAIN-partial-domain-public-inputs-v1','patient_id':subject,'patient_group':member[0]['patient_group'],
   'role':'TRAIN','source_domain_condition':CONDITION,'task_condition':'PARTIAL_TARGET_PROGRESS','public_only':True,
   'private_evaluation_files_included':False,'training_admitted':False,'actor_inputs_admitted':False,
   'intended_use_status':'prospective source-bound fixed-access search qualification only; no planning-task or model execution by this declaration',
   'anatomical_accuracy_reviewed':False,'clinical_suitability':False,'whole_tumor_is_prescribed_resection_target':False,
   'input_files':inputs,'shape_xyz':c['shape_xyz'],'affine_ras_mm':c['planning_derived_affine_ras_mm'],
   'source_MR_crop_affine_ras_mm':c['MR_native_crop_affine_ras_mm'],'reindex_policy':c['reindex_policy'],
   'public_support_domain_fully_covered':False,'public_source_domain_counts':{k:counts[k] for k in ('support_domain','target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain')},
   'public_target_support_consistency':{'whole_tumor_positive_voxels':counts['target'],'whole_tumor_positive_outside_supplied_support':counts['target_outside_support']},
   'public_label_resampling':q['label_summaries'],'source_centre_coverage':q['source_centre_coverage'],
   'source_bindings':{'cohort_sha256':COHORT,'case_sha256':c['case_sha256'],'headers_sha256':c['header_snapshot_sha256'],
    'conversion':entry['conversion'],'saved_review':entry['saved_review'],'saved_array_review_sha256':entry['saved_review']['sha256'],'root_visual_review':index['visual']},
   'source_availability':{'basis':'retrospective_annotation_assisted_source_preop','acquired_at':None,'annotation_available_at':None},
   'source_annotation_ancestry':{k:{'segments':c['annotations'][k]['ancestry']['segments'],'source_description':c['annotations'][k]['correspondence']['source_description'],
    'correspondence':c['annotations'][k]['correspondence']['correspondence'],'explicit_source_SOP_links_present':c['annotations'][k]['correspondence']['explicit_SOP_refs_present']} for k in ('cerebrum','whole_tumor')},
   'original_source_inventory_uncertainties':c['case_uncertainties'],
   'source_uncertainties':[
    'Source describes Preop phase; exact acquisition and annotation-availability timestamps remain unverified. This is a retrospective annotation-assisted condition, not proof of prospective availability.',
    'Bound header/frame and source-grid geometry, actual public crop conversion, saved array/domain/count checks and source-positive centre containment have completed. The recorded source-link correspondence limitations remain; absent explicit source-SOP links are not inferred.',
    'Root viewed three bound orthogonal overlays per case. This limited visual review and geometric qualification are not expert anatomical accuracy or clinical suitability validation.',
    'Manual whole tumor is not a prescribed resection target. Native source counts and nearest-neighbour placed counts/volumes remain distinct; full placed target is retained.',
    'Automatic Brainlab cerebrum is supplied estimated support, not physical occupancy. Source-known label zero does not establish physical empty space, and unknown source coverage remains unknown.',
    'No mask repair or source-domain extension is permitted. Any later artifact or fixed-access failure retains the person in the fixed four-person denominator without replacement.',
    'Cross-collection identity and pretrained exposure remain unknown; no independent external-transfer claim. No private annotation is included.'],
   'root_visual_observation':visual['observations'][subject],
   'assumptions':{'observed_support_knownness':'unchanged Ds','simulated_material':'S OR T positive, not a repaired anatomical mask',
    'simulated_interaction_domain':'Ds OR T positive; within-grid complement U unavailable to full tool and free-space/exposure flood',
    'label_zero':'not proof of physical empty anatomy','outside_image_tool_extent':'unchanged unassessed limitation under hypothetical aperture/workspace',
    'target_denominator':'entire saved placed T; native source counts and NN differences separately retained',
    'normalization':'existing support percentiles refer to explicit assumed O=S OR T world',
    'model_or_cross_dataset_overlap':'unknown; no independent external-generalization claim'},
   'limits':q['limits']+[visual['scope'],'No optimizer/policy/model calls; fixed access cannot be replaced after failure.']}
  path=output/(subject+'-public-inputs.json');write(path,manifest)
  rows.append({'patient_id':subject,'role':'TRAIN','path':str(path.resolve()),'sha256':sha(path.read_bytes())})
 result={'schema':'remind-partial-domain-fixed-TRAIN-public-index-v1','cases':rows,'planned_denominator':4,
  'cohort_sha256':COHORT,'condition':CONDITION,'models_or_updates':0,'execution_released':False,'failed_cases_replaced':False}
 write(output/'public-index.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence-index',type=Path,required=True);p.add_argument('--evidence-index-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 prepare(read({'path':str(a.evidence_index),'sha256':a.evidence_index_sha256}),a.output)
