"""Bind saved TRAIN050 public preparation; metadata only, no use admission."""
import hashlib
import json
import math
from pathlib import Path
import stat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COHORT='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
CONDITION='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
FILES={'image':('MR_native_crop.npy','float32'),'supplied_support':('cerebrum_source_label.npy','uint8'),
 'supplied_whole_tumor':('whole_tumor_source_label.npy','uint8'),'whole_tumor_domain':('whole_tumor_source_grid_domain.npy','uint8'),
 'supplied_support_domain':('cerebrum_source_grid_domain.npy','uint8')}
def sha(path):
 assert path.suffix!='.npy','This metadata writer never opens arrays'
 return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path,expected):
 assert path.suffix=='.json' and path.stat().st_size<2*1024**2 and sha(path)==expected
 return json.loads(path.read_bytes())
def binding(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
 directory=HERE/'crop-attempt-01/ReMIND-050-crop-mr-domains'
 cp=directory/'conversion-result.json';qp=HERE/'crop-attempt-01/ReMIND-050-saved-domain-review/saved-domain-review.json'
 c=read(cp,'d188d189c538883538194b690db49f0414e04dc9c1dbc3033fd9c56fa7634776')
 q=read(qp,'e8f3f31bfc7420b2f80517e568c4f7c56923c5f07b5ba0f4544630e5f20e67de')
 owned=HERE/'crop-attempt-01/receipt.json';receipt=read(owned,'d0715fe192ba2723b713f137309266325d6e6d281aa017c2ba1f6c1e29b77c78')
 cohort=read(ROOT/'manifests/experiments/remind-component-cohort-v1.json',COHORT)
 member=next(m for m in cohort['members'] if m['subject']=='ReMIND-050');assert member['role']=='TRAIN' and member['patient_group']=='ReMIND:050'
 assert receipt['fatal'] is None and len(receipt['outcomes'])==2 and all(r['phase_complete'] and r['exit_code']==0 and r['reaped'] and not r['remaining_owned_pids'] for r in receipt['outcomes'])
 assert c['patient_id']==q['patient_id']=='ReMIND-050' and c['role']==q['role']=='TRAIN'
 assert c['public_only'] and q['public_only'] and not c['private_reference_loaded'] and not q['original_source_reopened']
 assert c['shape_xyz']==q['shape_xyz']==[134,168,137] and c['case_sha256']==q['case_sha256']
 assert q['conversion_receipt_sha256']==sha(cp) and q['all_saved_artifact_hashes_verified'] and q['all_source_domain_voxels_world_checked']
 assert c['selected_MR_pixels_match_independent_raw_bytes'] and not q['public_support_domain_fully_covered']
 assert all(q['label_summaries'][k]['source_positive_centres_cropped_from_bound_execution']==0 for k in ('cerebrum','whole_tumor'))
 views=[]
 for v in c['public_overlays']['images']:
  item=c['artifacts'][v['file']]
  views.append({'path':str((directory/v['file']).relative_to(ROOT)),'sha256':item['sha256'],'bytes':item['bytes'],'axis':v['axis'],'crop_index':v['index'],'viewed':True})
 assert len(views)==3 and [(v['axis'],v['crop_index']) for v in views]==[(0,107),(1,62),(2,83)]
 visual={'reviewer':'root assistant, actual visual inspection reported in task message','patient_id':'ReMIND-050','role':'TRAIN','image_files':views,
  'observation':'Automatic green outline roughly follows brain contours; red supplied whole-tumor overlaps the visible lesion; no gross displacement, reversal or clipping apparent in the selected planes.',
  'scope':'Limited gross-alignment review of these three low-resolution views only; not expert truth, full-volume alignment, functional or physical occupancy, registration validation or clinical validity.',
  'establishes_segmentation_accuracy':False,'establishes_all_volume_alignment':False,'establishes_physical_occupancy':False,'establishes_functional_anatomy':False,'establishes_registration_accuracy':False,'establishes_clinical_validity':False,
  'planning_admission':False,'model_admission':False,'training_admission':False,'private_data_viewed':False,
  'record_provenance':'Transcription of root task instruction; prepared for root acceptance, no images reinterpreted by metadata writer.',
  'coordinates':'Crop-array planes 107/62/83 correspond to original MR axis indices 149/122/111 through crop start 42/60/28. Oblique physical orientation is defined by retained RAS affine, not a clinical radiological convention.'}
 vp=HERE/'root-visual-review.json';write(vp,visual)
 inputs={}
 for k,(name,dtype) in FILES.items():
  p=directory/name;st=p.lstat();expected=c['artifacts'][name]
  assert stat.S_ISREG(st.st_mode) and p.resolve()==p and st.st_size==expected['bytes']
  inputs[k]={'path':str(p),'dtype':dtype,**expected}
 count=q['saved_counts'];assert count=={'support':963346,'support_domain':2900340,'target':19557,'target_domain':125195,'target_in_known_support_positive':9791,'target_in_known_support_zero':9766,'target_in_unknown_support_domain':0,'target_outside_support':9766}
 assert q['support_unknown_domain_voxels']==183804 and math.prod(q['shape_xyz'])==count['support_domain']+183804
 independent=ROOT/'build/hybrid-train050-independent-v1/saved-crop-metadata-review.json'
 read(independent,'3750f5a08151cb322a32f5ebfeb570330b2ae7d62b1324b35dda351180b296f1')
 manifest={'schema':'remind-fixed-TRAIN-partial-domain-public-inputs-v1','patient_id':'ReMIND-050','patient_group':'ReMIND:050','role':'TRAIN',
  'source_domain_condition':CONDITION,'task_condition':'PARTIAL_TARGET_PROGRESS','public_only':True,'private_evaluation_files_included':False,
  'training_admitted':False,'actor_inputs_admitted':False,'planning_admitted':False,'model_admitted':False,
  'intended_use_status':'Source sample/domain preparation and limited root gross-view review recorded; separate exact 050 zero-update frozen-model source capability and runtime release required.',
  'anatomical_accuracy_reviewed':False,'clinical_suitability':False,'whole_tumor_is_prescribed_resection_target':False,
  'input_files':inputs,'shape_xyz':c['shape_xyz'],'affine_ras_mm':c['planning_derived_affine_ras_mm'],
  'source_MR_crop_affine_ras_mm':c['MR_native_crop_affine_ras_mm'],'reindex_policy':c['reindex_policy'],
  'public_support_domain_fully_covered':False,'public_support_source_domain_unknown_voxels':183804,
  'public_source_domain_counts':{k:count[k] for k in ('support_domain','target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain')},
  'public_target_support_consistency':{'whole_tumor_positive_voxels':19557,'whole_tumor_positive_outside_supplied_support':9766},
  'public_label_resampling':q['label_summaries'],'source_centre_coverage':q['source_centre_coverage'],
  'source_bindings':{'cohort_sha256':COHORT,'case_sha256':c['case_sha256'],'headers_sha256':c['header_snapshot_sha256'],
    'conversion':binding(cp),'saved_review':binding(qp),'saved_array_review_sha256':sha(qp),'root_visual_review':binding(vp),
    'intended_use_review_sha256':sha(vp),'owned_preparation_receipt':binding(owned),'independent_saved_metadata_review':binding(independent)},
  'source_preparation_provenance':{'source_MR_shape':[216,256,192],'crop_start_MR':c['public_crop_start_MR'],'crop_selection':c['source_cropping_map'],
    'source_MR_samples_preserved':True,'source_masks_repaired':False,'positive_centres_lost_to_crop':{'cerebrum':0,'whole_tumor':0},'array_hash_authority':'Exact saved conversion artifacts, independently rehashed by the completed saved-domain audit. This metadata writer stat-checked arrays without opening them.'},
  'source_availability':{'basis':'retrospective_annotation_assisted_source_preop','acquired_at':None,'annotation_available_at':None},
  'source_annotation_ancestry':{k:{'segments':c['annotations'][k]['ancestry']['segments'],'source_description':c['annotations'][k]['correspondence']['source_description'],
    'correspondence':c['annotations'][k]['correspondence']['correspondence'],'explicit_source_SOP_links_present':c['annotations'][k]['correspondence']['explicit_SOP_refs_present']} for k in ('cerebrum','whole_tumor')},
  'original_source_inventory_uncertainties':c['case_uncertainties'],
  'source_uncertainties':['Source describes Preop. Exact acquisition and annotation-availability timestamps remain unverified; this is retrospective annotation-assisted evidence.',
    'Shared-frame header geometry, source-sample conversion and saved domain/count checks completed. Both SEG lack explicit MR SOP references; anatomical alignment is not proved by shared frame.',
    'Root inspected only three saved planes for gross displacement/reversal/clipping; no expert segmentation, full-volume, functional, physical occupancy or registration accuracy claim.',
    'Full native and placed target denominator is 19,557. Whole tumor is not prescribed resection. 9,766 target cells remain outside automatic support.',
    'Automatic BrainLab cerebrum is estimated support; known label zero is not measured air. 183,804 cells outside its source domain remain unknown and unfilled.',
    'Cross-collection identity and pretrained exposure remain unknown. Original TRAIN role is unchanged; this is not an independently reserved external test.'],
  'root_visual_observation':visual,
  'assumptions':{'observed_support_knownness':'unaltered source Ds','prospective_simulated_material':'S OR T only under separately declared simulation condition; not anatomical repair',
    'prospective_simulated_interaction_domain':'Ds OR T; source evidence knownness stays Ds and inward unknown remains unavailable',
    'prospective_external_workspace':'Separate post-exposure assumption, never inferred physical air or source tissue deletion',
    'target_denominator':'Entire 19,557 placed T; native counts and NN limitations separately retained','aperture_or_route_feasibility':'unmeasured; no access or patient substitution permitted from this preparation'},
  'limits':q['limits']+['Source preparation alone grants no native route, model call, fitting or optimizer update.','Previous four-person learning/teacher cohorts and all SELECT/EVAL assignments remain unchanged.']}
 output=HERE/'public-inputs-v1';output.mkdir(exist_ok=False);mp=output/'ReMIND-050-public-inputs.json';write(mp,manifest)
 index={'schema':'remind-public-TRAIN050-preparation-index-v1','status':'source_metadata_not_execution_release','role':'TRAIN','planned_case_denominator':1,'max_optimizer_updates':0,
  'cases':[{'patient_id':'ReMIND-050','patient_group':'ReMIND:050','role':'TRAIN','status':'SOURCE_PREPARED_PENDING_SEPARATE_ZERO_UPDATE_CAPABILITY','path':str(mp.relative_to(ROOT)),'sha256':sha(mp),'full_target_denominator_voxels':19557,'unknown_support_voxels':183804,'anatomical_accuracy_pass':False}],
  'cohort_sha256':COHORT,'original_cohorts_modified':False,'split_changes':False,'actual_policy_runs':0,'private_reference_loaded':False}
 write(output/'public-train050-index.json',index)
 print(json.dumps({'manifest':binding(mp),'index':binding(output/'public-train050-index.json'),'root_visual_record':binding(vp)},indent=2))
if __name__=='__main__':main()
