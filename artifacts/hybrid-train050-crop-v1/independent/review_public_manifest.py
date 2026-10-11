"""Independent saved-JSON-only joins; never opens arrays, images, or models."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'build/hybrid-train050-preparation-v1'
READS = {}

def read(path, expected=None):
    path = Path(path)
    assert path.suffix == '.json' and path.stat().st_size < 2*1024**2
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert expected is None or digest == expected
    READS[str(path.relative_to(ROOT))] = {'sha256': digest, 'bytes': len(raw)}
    return json.loads(raw)

def main():
    mp = PREP/'public-inputs-v1/ReMIND-050-public-inputs.json'
    m = read(mp, '588757e8df0e420b3fcae2ac24a75a5276fd33e1722a49c681836938d16dda1d')
    index = read(PREP/'public-inputs-v1/public-train050-index.json', '7eec2ee858bc2e7989133cc41b80435ff59beddbf087cc46259dcbdded0bc536')
    c = read(PREP/'crop-attempt-01/ReMIND-050-crop-mr-domains/conversion-result.json', 'd188d189c538883538194b690db49f0414e04dc9c1dbc3033fd9c56fa7634776')
    q = read(PREP/'crop-attempt-01/ReMIND-050-saved-domain-review/saved-domain-review.json', 'e8f3f31bfc7420b2f80517e568c4f7c56923c5f07b5ba0f4544630e5f20e67de')
    visual = read(PREP/'root-visual-review.json', 'bbc15185fa35157a8cd5da6e97d19b6024f0a1f4e659f795c7979670e96b7ec4')
    cohort = read(ROOT/'manifests/experiments/remind-component-cohort-v1.json', '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05')
    member = [x for x in cohort['members'] if x['subject'] == 'ReMIND-050']
    assert len(member) == 1 and member[0]['role'] == m['role'] == 'TRAIN'
    assert member[0]['patient_group'] == m['patient_group'] == 'ReMIND:050'
    assert m['patient_id'] == c['patient_id'] == q['patient_id'] == 'ReMIND-050'
    assert m['schema'] == 'remind-fixed-TRAIN-partial-domain-public-inputs-v1'
    assert m['source_domain_condition'] == 'cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
    assert m['task_condition'] == 'PARTIAL_TARGET_PROGRESS' and m['public_only'] is True
    for key in ('private_evaluation_files_included', 'training_admitted', 'actor_inputs_admitted', 'planning_admitted', 'model_admitted', 'anatomical_accuracy_reviewed', 'clinical_suitability', 'whole_tumor_is_prescribed_resection_target'):
        assert m[key] is False
    files = {'image': ('MR_native_crop.npy', 'float32'),
        'supplied_support': ('cerebrum_source_label.npy', 'uint8'),
        'supplied_whole_tumor': ('whole_tumor_source_label.npy', 'uint8'),
        'whole_tumor_domain': ('whole_tumor_source_grid_domain.npy', 'uint8'),
        'supplied_support_domain': ('cerebrum_source_grid_domain.npy', 'uint8')}
    assert set(m['input_files']) == set(files)
    for key, (name, dtype) in files.items():
        record = m['input_files'][key]
        assert record == {'path': str(PREP/'crop-attempt-01/ReMIND-050-crop-mr-domains'/name), 'dtype': dtype, **c['artifacts'][name]}
    assert m['shape_xyz'] == c['shape_xyz'] == q['shape_xyz'] == [134, 168, 137]
    assert m['affine_ras_mm'] == c['planning_derived_affine_ras_mm']
    assert m['source_MR_crop_affine_ras_mm'] == c['MR_native_crop_affine_ras_mm']
    assert m['reindex_policy'] == c['reindex_policy']
    assert m['reindex_policy']['source_MR_samples_preserved'] is True
    assert m['reindex_policy']['original_MR_affine_overwritten'] is False
    assert m['reindex_policy']['image_interpolation'] == 'none'
    assert m['reindex_policy']['maximum_crop_world_difference_mm'] <= .001
    counts = q['saved_counts']
    assert m['public_target_support_consistency'] == {'whole_tumor_positive_voxels': counts['target'], 'whole_tumor_positive_outside_supplied_support': counts['target_outside_support']}
    assert m['public_source_domain_counts'] == {k: counts[k] for k in ('support_domain', 'target_in_known_support_positive', 'target_in_known_support_zero', 'target_in_unknown_support_domain')}
    assert counts['target'] == 19557 == counts['target_in_known_support_positive'] + counts['target_in_known_support_zero'] + counts['target_in_unknown_support_domain']
    assert counts['target_in_unknown_support_domain'] == 0
    assert counts['support_domain'] + m['public_support_source_domain_unknown_voxels'] == math.prod(m['shape_xyz'])
    assert m['public_support_source_domain_unknown_voxels'] == q['support_unknown_domain_voxels'] == 183804
    assert m['public_support_domain_fully_covered'] is False
    assert m['public_label_resampling'] == q['label_summaries'] and m['source_centre_coverage'] == q['source_centre_coverage']
    assert all(x['source_positive_centres_cropped_from_bound_execution'] == 0 for x in q['label_summaries'].values())
    for key in ('conversion', 'saved_review', 'root_visual_review', 'owned_preparation_receipt', 'independent_saved_metadata_review'):
        d = m['source_bindings'][key]
        read(d['path'], d['sha256'])
        assert READS[str(Path(d['path']).relative_to(ROOT))]['bytes'] == d['bytes']
    assert m['source_bindings']['headers_sha256'] == c['header_snapshot_sha256']
    assert m['source_bindings']['case_sha256'] == c['case_sha256'] == q['case_sha256']
    assert m['source_bindings']['saved_array_review_sha256'] == m['source_bindings']['saved_review']['sha256']
    assert m['source_bindings']['intended_use_review_sha256'] == m['source_bindings']['root_visual_review']['sha256']
    assert m['root_visual_observation'] == visual
    for row in visual['image_files']:
        assert {'bytes': row['bytes'], 'sha256': row['sha256']} == c['artifacts'][Path(row['path']).name]
    assert [(v['axis'],v['crop_index']) for v in visual['image_files']] == [(0,107),(1,62),(2,83)]
    for k in ('establishes_segmentation_accuracy','establishes_all_volume_alignment','establishes_physical_occupancy','establishes_clinical_validity','planning_admission','model_admission','training_admission'):
        assert visual[k] is False
    for k in ('cerebrum', 'whole_tumor'):
        assert m['source_annotation_ancestry'][k]['segments'] == c['annotations'][k]['ancestry']['segments']
        assert m['source_annotation_ancestry'][k]['explicit_source_SOP_links_present'] is False
    assert m['source_availability'] == {'basis':'retrospective_annotation_assisted_source_preop','acquired_at':None,'annotation_available_at':None}
    assert index['planned_case_denominator'] == 1 and index['original_cohorts_modified'] is False and index['actual_policy_runs'] == 0
    assert index['cases'][0]['sha256'] == READS[str(mp.relative_to(ROOT))]['sha256'] and index['cases'][0]['full_target_denominator_voxels'] == 19557
    report = {'status': 'PASS_SAVED_PUBLIC_MANIFEST_METADATA_NO_RUNTIME_ADMISSION', 'inputs': READS,
        'checks': ['original fixed TRAIN050 role', 'exact five public array descriptors joined without array opens', 'native and nominal affines retained', 'full T19557 and Ds unknown183804 preserved', 'known-label-zero target9766 distinct from unknown0', 'zero source-positive crop losses bound to completed saved audit', 'limited root visual record bound without image reinterpretation', 'default full-coverage incompatibility explicit', 'no planning/model/training admission'],
        'review_boundary': {'arrays_opened':False,'originals_opened':False,'images_opened':False,'model_parameters_opened':False,'source_functions_imported':False},
        'remaining_scope':'Separate fixed050 no-fit frozen-ridge capability and root runtime release; no clinical or held-out SELECT claim.'}
    out = HERE/'public-manifest-review.json'
    with out.open('x') as f: json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'path':str(out.relative_to(ROOT)), 'sha256':hashlib.sha256(out.read_bytes()).hexdigest(), 'status':report['status']}))

if __name__ == '__main__': main()
