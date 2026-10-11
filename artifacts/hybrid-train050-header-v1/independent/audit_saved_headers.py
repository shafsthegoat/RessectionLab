"""Saved JSON only; no DICOM/arrays, refit, native geometry, or model imports."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = ROOT/'build/hybrid-train050-preparation-v1'
RUN = PREP/'header-attempt-01'

def load(path): return json.loads(path.read_bytes())
def pin(path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}

header_path=RUN/'ReMIND-050-headers/result.json'
receipt_path=RUN/'receipt.json'
assert pin(header_path)['sha256']=='9a9f91bc38173df42e765b8bb886290d22b1aaea48970449a4f0f914a9c1b9ba'
assert pin(receipt_path)['sha256']=='8a80c471666b60487ef63c16c5a5ae7ca65cac6ecd0aec759beabead38cf0b43'
header, receipt=load(header_path),load(receipt_path)
declaration=load(RUN/'declaration.json')
released=load(Path(declaration['release_path']))
prepared=load(PREP/'prepared-release.json')
assert released['execution_released'] is True
assert {**released,'execution_released':False}==prepared
assert pin(Path(declaration['release_path']))['sha256']==declaration['release_sha256']
assert declaration['parent_sha256']==pin(PREP/'run_headers_owned.py')['sha256']
assert declaration['source_index']==pin(PREP/'source-index.json')
case_path=PREP/'bindings/ReMIND-050-case.json'
case=load(case_path)
assert header['case_sha256']==pin(case_path)['sha256']
assert header['executing_script_sha256']==pin(PREP/'stage/src/resectionlab/remind_planning_qc.py')['sha256']
assert header['reused_converter_sha256']==pin(ROOT/'scripts/convert_remind_development.py')['sha256']
assert (header['patient_id'],header['patient_group'],header['role'])==('ReMIND-050','ReMIND:050','TRAIN')
assert header['status']=='header_geometry_and_ancestry_projected'
assert header['objects_verified']==194 and header['source_file_returned_bytes']==27142222
assert header['pixel_decode_calls']==0 and header['native_conversion_performed'] is False
assert header['public_only'] is True and header['private_reference_loaded'] is False
assert header['training_admitted'] is header['planner_inputs_admitted'] is header['actor_inputs_admitted'] is False
assert header['optimizer_updates_performed']==0
assert header['anatomy_and_coverage_review']=='unreviewed'
assert receipt['fatal'] is None and receipt['training_admitted'] is False and receipt['failed_cases_replaced'] is False
assert receipt['planned_subjects']==['ReMIND-050'] and len(receipt['outcomes'])==1
outcome,=receipt['outcomes']
assert outcome['exit_code']==0 and outcome['phase_complete'] is outcome['reaped'] is True
assert outcome['remaining_owned_pids']==outcome['cleanup_errors']==[]
assert outcome['stop_reason'] is outcome['result_guard_error'] is None
assert outcome['result_sha256']==pin(header_path)['sha256']
assert outcome==load(RUN/'ReMIND-050-headers-supervisor.json')
assert outcome['sampled_peak_rss_bytes']<receipt['caps']['sampled_rss_bytes']
assert outcome['case_elapsed_seconds']<receipt['caps']['seconds_per_case']
assert outcome['case_output_bytes']==pin(header_path)['bytes']<receipt['caps']['output_bytes_per_case']
assert outcome['final_phase_log_bytes']==(RUN/'ReMIND-050-headers.log').stat().st_size<receipt['caps']['log_bytes_per_phase']
series={s['kind']:s for s in header['series']}
assert len(series)==len(header['series'])==3 and set(series)=={'structural_t1ce','whole_tumor','cerebrum'}
for item in case['series']:
    saved=series[item['kind']]
    assert saved['source_objects']==item['objects']
    assert saved['series_instance_uid']==item['SeriesInstanceUID'] and saved['modality']==item['Modality']
    assert len(saved['sop_instance_uids'])==len(item['objects'])==len(set(saved['sop_instance_uids']))
frames={s['geometry']['frame_of_reference_uid'] for s in series.values()}
assert len(frames)==1
for kind in ('whole_tumor','cerebrum'):
    align=series[kind]['alignment']
    assert align['frame_uid_matches'] is True and align['correspondence']=='shared_frame_plus_source_description'
    assert align['explicit_SOP_refs_present'] is False and align['unmatched_explicit_SOP_refs']==[]
    assert align['registration_performed'] is align['identity_cross_frame_assumed'] is align['expert_alignment_review'] is False
    assert series[kind]['ancestry']['referenced_sop_instance_uids']==[]
tumor,=series['whole_tumor']['ancestry']['segments']
support,=series['cerebrum']['ancestry']['segments']
assert (tumor['algorithm_type'],tumor['label'])==('MANUAL','Glioma')
assert (support['algorithm_type'],support['algorithm_name'],support['label'])==('AUTOMATIC','BrainLab','Brain')
pending=load(RUN/'pending-qualification-index.json')
assert pending['planned_denominator']==1 and pending['cases'][0]['public_input_manifest'] is None
assert pending['cases'][0]['training_admitted'] is False
result={'status':'PASS_SAVED_TRAIN050_HEADER_SOURCE_BINDINGS_ONLY', 'header':pin(header_path), 'receipt':pin(receipt_path), 'release':pin(Path(declaration['release_path'])), 'source_index':pin(PREP/'source-index.json'), 'case':pin(case_path), 'objects':194, 'source_bytes':27142222, 'decode_calls':0, 'supervised_seconds':receipt['elapsed_seconds'], 'worker_seconds':header['elapsed_seconds'], 'sampled_peak_rss_bytes':outcome['sampled_peak_rss_bytes'], 'worker_self_peak_rss_bytes':header['peak_RSS_bytes'], 'reported_shapes':{k:s['geometry']['shape_xyz'] for k,s in series.items()}, 'reported_common_frame_uid':next(iter(frames)), 'correspondence':'shared_frame_plus_source_description; no explicit source SOP references', 'tumor_ancestry':tumor, 'support_ancestry':support, 'reported_geometry_independently_refit':False, 'source_objects_match_exact_case':True, 'all_owned_workers_reaped':True, 'training_or_planning_admission':False, 'limitations':['No actual image/label pixels inspected; positive counts, target/support overlap, artifact status and coverage remain unknown.', 'Matching frame/text references are not expert anatomical alignment or direct SOP ancestry.', 'Saved deidentified timestamps and Preop source label do not establish annotation availability at a prospective decision time.', 'The automatic BrainLab label remains supplied cerebrum evidence, not validated physical occupancy.', 'MR/source-label grids differ. Any later conversion must retain source affines, native MRI samples, explicit source-domain masks and full target denominator; no missing coverage is inferred here.'], 'reviewer_original_header_pixel_or_array_opens':0}
output=HERE/'saved-header-review.json'
with output.open('x') as f:
    json.dump(result,f,indent=2)
    f.write('\n')
print(json.dumps(pin(output)))
