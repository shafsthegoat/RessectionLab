"""Receipt/count arithmetic only. No image, array, original header, or geometry reads."""
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'build/hybrid-train050-preparation-v1'
RUN=PREP/'crop-attempt-01'
CROP=RUN/'ReMIND-050-crop-mr-domains'
REVIEW=RUN/'ReMIND-050-saved-domain-review'
def load(p):return json.loads(p.read_bytes())
def pin(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
expected={RUN/'receipt.json':'d0715fe192ba2723b713f137309266325d6e6d281aa017c2ba1f6c1e29b77c78', CROP/'conversion-result.json':'d188d189c538883538194b690db49f0414e04dc9c1dbc3033fd9c56fa7634776', REVIEW/'saved-domain-review.json':'e8f3f31bfc7420b2f80517e568c4f7c56923c5f07b5ba0f4544630e5f20e67de', PREP/'crop-root-release.json':'728676c207535198693edf549f723d8b2ff146f20fb3df79ca6e08b1a1af5706', PREP/'crop-source-index.json':'1615cc9a7237f3bb0bbcbf537d14ab895fd60e7a69ecd79184988754caa6f789'}
for path,digest in expected.items():assert pin(path)['sha256']==digest
receipt=load(RUN/'receipt.json'); conversion=load(CROP/'conversion-result.json'); saved=load(REVIEW/'saved-domain-review.json')
release=load(PREP/'crop-root-release.json');prepared=load(PREP/'crop-prepared-release.json');declaration=load(RUN/'declaration.json')
assert release['execution_released'] is True and {**release,'execution_released':False}==prepared
assert declaration['release_sha256']==pin(PREP/'crop-root-release.json')['sha256']
assert declaration['source_index']==pin(PREP/'crop-source-index.json')
assert declaration['parent_sha256']==pin(PREP/'run_crop_owned.py')['sha256']
source=load(PREP/'crop-source-index.json')
for row in source['files']:assert pin(ROOT/row['path'])==row
case_path=PREP/'bindings/ReMIND-050-case.json';case=load(case_path)
header_path=PREP/'header-attempt-01/ReMIND-050-headers/result.json';headers=load(header_path)
for result in (conversion,saved):
    assert result['patient_id']=='ReMIND-050' and result['role']=='TRAIN'
    assert result['case_sha256']==pin(case_path)['sha256'] and result['header_snapshot_sha256']==pin(header_path)['sha256']
    assert result['public_only'] is True and result['private_reference_loaded'] is False
    assert result['training_admitted'] is result['actor_inputs_admitted'] is False
    assert result['optimizer_updates_performed']==0 and result['full_coverage_factory_compatible'] is False
assert conversion['executing_script_sha256']==pin(PREP/'crop-stage/src/resectionlab/remind_planning_qc.py')['sha256']
assert saved['executing_script_sha256']==saved['audit_script_sha256']==pin(PREP/'audit_saved_domains.py')['sha256']
assert saved['conversion_receipt_sha256']==pin(CROP/'conversion-result.json')['sha256']
assert saved['original_source_reopened'] is False and saved['public_input_manifest'] is None
assert saved['all_saved_artifact_hashes_verified'] is saved['all_source_domain_voxels_world_checked'] is True
assert receipt['fatal'] is None and receipt['planned_subjects']==['ReMIND-050']
assert receipt['training_admitted'] is receipt['roles_changed'] is receipt['failed_cases_replaced'] is False
assert [r['phase'] for r in receipt['outcomes']]==['crop-mr-domains','saved-domain-review']
for phase in receipt['outcomes']:
    assert phase==load(RUN/('ReMIND-050-'+phase['phase']+'-supervisor.json'))
    assert phase['phase_complete'] is phase['reaped'] is True and phase['exit_code']==0
    assert phase['cleanup_errors']==phase['remaining_owned_pids']==[] and phase['stop_reason'] is phase['result_guard_error'] is None
    assert phase['sampled_peak_rss_bytes']<receipt['caps']['sampled_rss_bytes']
    assert phase['case_elapsed_seconds']<receipt['caps']['seconds_per_case']
    assert phase['case_output_bytes']<receipt['caps']['output_bytes_per_case']
    assert phase['final_phase_log_bytes']<receipt['caps']['log_bytes_per_phase']
assert receipt['outcomes'][0]['result_sha256']==pin(CROP/'conversion-result.json')['sha256']
assert receipt['outcomes'][1]['result_sha256']==pin(REVIEW/'saved-domain-review.json')['sha256']
mr=next(s for s in case['series'] if s['kind']=='structural_t1ce')
mr_header=next(s for s in headers['series'] if s['kind']=='structural_t1ce')
start=conversion['public_crop_start_MR'];shape=conversion['shape_xyz'];nvox=math.prod(shape)
selected=mr_header['geometry']['sorted_source_indices'][start[2]:start[2]+shape[2]]
selected_objects=[mr['objects'][i] for i in selected]+[o for s in case['series'] if s['kind']!='structural_t1ce' for o in s['objects']]
assert len(selected)==conversion['MR_pixel_source_objects']==137
assert len(mr['objects'])-len(selected)==conversion['MR_header_only_objects_outside_selected_z']==55
assert len(selected_objects)==conversion['objects_verified']==139
assert sum(o['bytes'] for o in selected_objects)==conversion['source_file_returned_bytes']==20937654
assert saved['shape_xyz']==shape==[134,168,137]
counts=saved['saved_counts'];relation=conversion['public_target_support_domain_relation']
assert counts['target']==relation['full_placed_target_voxels']==19557
parts=['target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain']
assert all(counts[k]==relation[k] for k in parts)
assert sum(counts[k] for k in parts)==counts['target']
assert counts['target_outside_support']==counts['target_in_known_support_zero']+counts['target_in_unknown_support_domain']==9766
assert counts['target_in_unknown_support_domain']==0
assert relation['known_label_zero_is_physical_empty'] is relation['unknown_label_zero_is_observed_empty'] is False
assert conversion['public_target_support_relation']['outside_fraction']==counts['target_outside_support']/counts['target']
assert nvox-counts['support_domain']==saved['support_unknown_domain_voxels']==conversion['public_support_source_domain_unknown_voxels']==183804
assert conversion['public_support_source_domain_complete'] is saved['public_support_domain_fully_covered'] is False
assert conversion['support_source_domain_required'] is True
assert conversion['selected_MR_pixels_match_independent_raw_bytes'] is True
assert conversion['reindex_policy']['source_MR_samples_preserved'] is True and conversion['reindex_policy']['original_MR_affine_overwritten'] is False
assert conversion['reindex_policy']['maximum_crop_world_difference_mm']<=conversion['reindex_policy']['maximum_bound_mm']==.001
assert conversion['reindex_policy']['image_interpolation']=='none'
for kind,ck,dk in [('cerebrum','support','support_domain'),('whole_tumor','target','target_domain')]:
    ann=conversion['annotations'][kind];placement=ann['placement'];summary=saved['label_summaries'][kind]
    assert ann['placed_positive_voxels']==placement['resampled_positive_voxels']==summary['saved_positive_voxels']==counts[ck]
    assert ann['placed_source_domain_voxels']==placement['resampled_source_domain_voxels']==summary['saved_domain_voxels']==counts[dk]
    assert ann['source_positive_voxels']==placement['source_positive_voxels']==summary['source_positive_voxels_from_bound_execution']
    assert placement['source_positive_centres_outside_target_grid']==summary['source_positive_centres_cropped_from_bound_execution']==0
    assert ann['source_samples_equal'] is True and placement['outside_source_domain']=='unknown'
    assert summary['source_positive_volume_mm3_from_bound_execution']==placement['source_positive_volume_mm3']
    assert summary['saved_positive_volume_mm3']==placement['resampled_positive_volume_mm3']
    change=100*(summary['saved_positive_volume_mm3']/summary['source_positive_volume_mm3_from_bound_execution']-1)
    assert math.isclose(change,summary['volume_change_percent'],abs_tol=1e-10)
    assert saved['source_centre_coverage'][kind]['all_source_grid_centres_inside_crop'] is True
assert conversion['annotations']['whole_tumor']['source_positive_voxels']==counts['target']
artifact_bytes=sum(r['bytes'] for r in conversion['artifacts'].values())
assert artifact_bytes==conversion['output_bytes_before_receipt']
assert artifact_bytes+pin(CROP/'conversion-result.json')['bytes']==receipt['outcomes'][0]['case_output_bytes']
assert receipt['outcomes'][0]['case_output_bytes']+pin(REVIEW/'saved-domain-review.json')['bytes']+pin(REVIEW/'saved-review-index.json')['bytes']==receipt['outcomes'][1]['case_output_bytes']
pending=load(RUN/'pending-qualification-index.json')
assert pending['planned_denominator']==1 and pending['cases'][0]['public_input_manifest'] is None and pending['cases'][0]['training_admitted'] is False
result={'status':'PASS_SAVED_TRAIN050_CROP_METADATA_AND_ACCOUNTING_ANATOMY_UNREVIEWED','bindings':[pin(p) for p in expected], 'public_case':pin(case_path),'header':pin(header_path),'source_files_unchanged':len(source['files']), 'source_objects_reread_for_conversion':139,'source_bytes_returned':20937654,'MR_planes_selected':137,'MR_planes_not_reread':55, 'shape_xyz':shape,'crop_voxels':nvox,'counts':counts,'unknown_support_voxels':183804,'unknown_support_fraction':183804/nvox,'target_outside_support_fraction':9766/19557,'full_source_and_placed_target_voxels':19557,'source_positive_centres_cropped':{'cerebrum':0,'whole_tumor':0},'volume_change_percent':{k:v['volume_change_percent'] for k,v in saved['label_summaries'].items()},'declared_MRI_sample_equality':'bound conversion worker raw-byte audit; not independently reread here','declared_domain_and_artifact_checks':'bound saved-domain worker hash/slab/world checks; this independent lane reconciles their receipts/count arithmetic only','max_reported_MRI_reindex_world_difference_mm':conversion['reindex_policy']['maximum_crop_world_difference_mm'],'supervised_seconds':receipt['elapsed_seconds'],'max_sampled_worker_RSS_bytes':max(o['sampled_peak_rss_bytes'] for o in receipt['outcomes']),'conversion_self_peak_RSS_bytes':conversion['peak_RSS_bytes'],'all_workers_reaped_without_cleanup_errors':True,'output_artifact_declared_pins':conversion['artifacts'],'public_overlay_paths':[str((CROP/v['file']).relative_to(ROOT)) for v in conversion['public_overlays']['images']], 'intended_use_prerequisites':['Root reviews three saved overlays/source-artifact appearance; this lane made no visual or expert-anatomy claim.', 'Prepare exact five-array public manifest MRI/S/T/Ds/Dt with original/derived affines and full19557 target denominator,963346 support positives,2900340 support-domain and183804 unknown cells; no full-domain factory declaration.', 'Any planning use must explicitly state supplied cerebrum is not physical occupancy. Raw support permits at most9791 of19557 target cells to be support-contained; a modeled S OR T world adds9766 known-label-zero target cells as a declared assumption, not acquisition repair.', 'Canonical partial-domain admission currently permits only002/015/018/045; a separate exact050 source capability is needed. Existing fixed-four sparse-ranker fitting capability cannot simply be broadened; bind frozen TRAIN fit lineage and zero fit/update transfer behavior.', 'Keep fixed geometry/tools/reward/access protocol and declare any modeled exposure, unknown exterior and missing anatomy. Retain050 and failures without outcome-based replacement.'], 'training_or_planning_admission':False,'reviewer_original_payload_array_image_geometry_or_model_reads':0}
output=HERE/'saved-crop-metadata-review.json'
with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'output':pin(output),'target_outside_support_percent':100*9766/19557,'unknown_support_percent':100*183804/nvox,'crop_voxels':nvox}))
