"""Create SELECT013 source metadata only; no arrays or planning admission."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
COHORT='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
CONDITION='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
FILES={'image':('MR_native_crop.npy','float32'),'supplied_support':('cerebrum_source_label.npy','uint8'),
 'supplied_whole_tumor':('whole_tumor_source_label.npy','uint8'),'whole_tumor_domain':('whole_tumor_source_grid_domain.npy','uint8'),
 'supplied_support_domain':('cerebrum_source_grid_domain.npy','uint8')}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path,expected):
    assert path.suffix=='.json' and path.stat().st_size<2*1024**2 and sha(path)==expected
    return json.loads(path.read_bytes())
def binding(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root-visual',type=Path,required=True);p.add_argument('--root-visual-sha256',required=True)
    args=p.parse_args();visual=read(args.root_visual,args.root_visual_sha256)
    directory=HERE/'actual-domain-qc-v1/ReMIND-013-crop-mr-domains'
    cp=directory/'conversion-result.json'
    qp=HERE/'actual-domain-qc-v1/ReMIND-013-saved-domain-review/saved-domain-review.json'
    c=read(cp,'bf9174ee066f2ce744a4b3ace1ae048954733981c8126e461c15a75d7c89a057')
    q=read(qp,'6479dd6df8a9d81a05438e481533b2083a7d95cb5889f93b3c4dea038f6d6a7d')
    cohort_path=ROOT/'manifests/experiments/remind-component-cohort-v1.json';cohort=read(cohort_path,COHORT)
    member=next(m for m in cohort['members'] if m['subject']=='ReMIND-013');assert member['role']=='SELECT'
    assert c['patient_id']==q['patient_id']=='ReMIND-013' and c['role']==q['role']=='SELECT'
    assert c['public_only'] and q['public_only'] and not c['private_reference_loaded'] and not q['original_source_reopened']
    assert c['shape_xyz']==q['shape_xyz']==[131,154,117]
    assert q['conversion_receipt_sha256']==sha(cp) and c['case_sha256']==q['case_sha256']
    assert q['all_saved_artifact_hashes_verified'] and q['all_source_domain_voxels_world_checked']
    assert c['selected_MR_pixels_match_independent_raw_bytes'] and not q['public_support_domain_fully_covered']
    assert all(q['label_summaries'][k]['source_positive_centres_cropped_from_bound_execution']==0 for k in ('cerebrum','whole_tumor'))
    assert visual['patient_id']=='ReMIND-013' and visual['role']=='SELECT' and visual['planning_admission'] is False
    assert len(visual['image_files'])==3 and all(v['viewed'] is True and
        c['artifacts'][Path(v['path']).name]['sha256']==v['sha256'] for v in visual['image_files'])
    # Root's separate, hash-bound note is retained verbatim as limited visual
    # evidence. This metadata writer does not inspect images or grant use.
    inputs={k:{'path':str(directory/name),'dtype':dtype,**c['artifacts'][name]} for k,(name,dtype) in FILES.items()}
    count=q['saved_counts'];assert count['target']==35260 and count['support']==745592
    manifest={'schema':'remind-fixed-SELECT-partial-domain-public-inputs-v1','patient_id':'ReMIND-013','patient_group':'ReMIND:013','role':'SELECT',
      'source_domain_condition':CONDITION,'task_condition':'PARTIAL_TARGET_PROGRESS','public_only':True,
      'private_evaluation_files_included':False,'training_admitted':False,'actor_inputs_admitted':False,
      'intended_use_status':'source preparation and saved geometry complete; separate fixed four-arm SELECT inference release required',
      'anatomical_accuracy_reviewed':False,'clinical_suitability':False,'whole_tumor_is_prescribed_resection_target':False,
      'input_files':inputs,'shape_xyz':c['shape_xyz'],'affine_ras_mm':c['planning_derived_affine_ras_mm'],
      'source_MR_crop_affine_ras_mm':c['MR_native_crop_affine_ras_mm'],'reindex_policy':c['reindex_policy'],
      'public_support_domain_fully_covered':False,'public_source_domain_counts':{k:count[k] for k in ('support_domain','target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain')},
      'public_target_support_consistency':{'whole_tumor_positive_voxels':count['target'],'whole_tumor_positive_outside_supplied_support':count['target_outside_support']},
      'public_label_resampling':q['label_summaries'],'source_centre_coverage':q['source_centre_coverage'],
      'source_bindings':{'cohort_sha256':COHORT,'case_sha256':c['case_sha256'],'headers_sha256':c['header_snapshot_sha256'],
        'conversion':binding(cp),'saved_review':binding(qp),'saved_array_review_sha256':sha(qp),'root_visual_review':binding(args.root_visual)},
      'expanded_preparation_provenance':q['expanded_preparation_provenance'],
      'source_availability':{'basis':'retrospective_annotation_assisted_source_preop','acquired_at':None,'annotation_available_at':None},
      'source_annotation_ancestry':{k:{'segments':c['annotations'][k]['ancestry']['segments'],'source_description':c['annotations'][k]['correspondence']['source_description'],
        'correspondence':c['annotations'][k]['correspondence']['correspondence'],'explicit_source_SOP_links_present':c['annotations'][k]['correspondence']['explicit_SOP_refs_present']} for k in ('cerebrum','whole_tumor')},
      'original_source_inventory_uncertainties':c['case_uncertainties'],
      'source_uncertainties':[
        'Source describes Preop; exact acquisition and annotation availability timestamps remain unverified. Retrospective annotation-assisted input is not established prospective availability.',
        'Same-frame header geometry, source-sample conversion, saved domain/counts and original overlap have been checked. Absent explicit source SOP links remain a correspondence limitation.',
        'Limited root overlay review and geometry checks do not establish expert anatomy or clinical accuracy.',
        'Supplied whole tumor is not a prescribed resection target. Full native and NN-placed target denominators remain explicit.',
        'Automatic Brainlab cerebrum is estimated support, not physical occupancy. Known label zero is not measured air. Outside Ds remains unknown.',
        'Cross-collection identity and pretrained exposure remain unknown. No private annotation or external independence claim.'],
      'root_visual_observation':visual,
      'assumptions':{'observed_support_knownness':'unchanged source Ds','simulated_material':'S OR T; explicitly assumed, no anatomical repair',
        'simulated_interaction_domain':'Ds OR T; observation knownness remains Ds',
        'external_workspace':'separate post-exposure assumption; never observed air or tissue deletion',
        'outside_image_tool_extent':'unassessed, with the existing distal-point image guard retained',
        'target_denominator':'entire35260 placed T; original native count and NN sampling separately reported',
        'aperture':'same physical side/transverse and occupancy-aware boundary as bound diagnostic; no outcome-based replacement'},
      'limits':q['limits']+['Preparation alone does not admit a native route, model call or optimizer update.']}
    output=HERE/'public-inputs-v1';output.mkdir(exist_ok=False)
    mp=output/'ReMIND-013-public-inputs.json';write(mp,manifest)
    prior_path=ROOT/'build/remind-select-public-preparation-v1/public-select-index-v2.json'
    prior=read(prior_path,'adc7232f0fd05300f90078385bc2576150eccc882712e31bfc97534f68680d5f')
    assert len(prior['cases'])==2 and {r['patient_id'] for r in prior['cases']}=={'ReMIND-013','ReMIND-037'}
    held=next(r for r in prior['cases'] if r['patient_id']=='ReMIND-037');assert held['path'] is None and held['status']=='HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT' and held['role']=='SELECT'
    index={'schema':'remind-public-SELECT-expanded-input-index-v1','status':'source_metadata_not_execution_release',
      'role':'SELECT','max_optimizer_updates':0,'planned_case_denominator':2,'original_select_ledger':binding(prior_path),
      'cases':[{'patient_id':'ReMIND-013','patient_group':'ReMIND:013','role':'SELECT',
        'status':'EXPANDED_SOURCE_PREPARED_PENDING_SEPARATE_INFERENCE_RELEASE','path':str(mp.relative_to(ROOT)),'sha256':sha(mp),
        'supported_target_positive_voxels':count['target_in_known_support_positive'],'full_target_denominator_voxels':count['target'],'anatomical_accuracy_pass':False},held],
      'private_reference_loaded':False,'split_changes':False,'actual_policy_runs':0}
    write(output/'public-select-index.json',index)
    print(json.dumps({'manifest':binding(mp),'index':binding(output/'public-select-index.json')},indent=2))
if __name__=='__main__':main()
