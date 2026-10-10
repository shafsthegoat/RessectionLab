"""Read small saved source/receipt JSON only; never open images or array bodies."""
import hashlib
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/remind-train-public-qc-preparation-v1'
ACTUAL = BASE / 'actual-public-qc-v1'
OUT = BASE / 'metadata-preservation-v1'
SUBJECTS = ['ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045']
READS = {}

def metadata(path):
    p = Path(path)
    assert p.suffix in {'.json', '.py'}, p
    data = p.read_bytes()
    desc = {'path': str(p.relative_to(ROOT)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    READS[desc['path']] = desc
    return desc, data

def read(path):
    desc, data = metadata(path)
    return json.loads(data), desc

def write(name, value):
    p = OUT / name
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return metadata(p)[0]

receipt, receipt_pin = read(ACTUAL / 'receipt.json')
declaration, declaration_pin = read(ACTUAL / 'declaration.json')
release, release_pin = read(BASE / 'root-release.json')
prepared, prepared_pin = read(BASE / 'prepared-release.json')
assert prepared['execution_released'] is False and release['execution_released'] is True
assert {**prepared, 'execution_released': True} == release
assert declaration['release_sha256'] == release_pin['sha256']
index, index_pin = read(BASE / 'source-index.json')
assert release['source_index'] == index_pin == declaration['source_index']
for expected in index['files']:
    actual, _ = metadata(ROOT / expected['path'])
    assert actual == expected, expected['path']
public, public_pin = read(BASE / 'public-train-index.json')
assert release['public_index'] == public_pin == declaration['public_index']
assert public['subjects'] == release['subjects'] == receipt['planned_subjects'] == SUBJECTS
assert receipt['fatal'] is None and receipt['caps'] == release['caps']
assert receipt['elapsed_seconds'] < receipt['caps']['total_seconds']
assert receipt['public_only'] and not receipt['roles_changed'] and not receipt['training_admitted']
assert not receipt['failed_cases_replaced'] and not receipt['automatic_source_artifact_clearance']
pending, pending_pin = read(ACTUAL / 'pending-qualification-index.json')
assert pending['planned_denominator'] == 4
assert [v['patient_id'] for v in pending['cases']] == SUBJECTS
assert all(v['public_input_manifest'] is None and not v['training_admitted'] and v['role'] == 'TRAIN' for v in pending['cases'])
expected_pairs = [(s, phase) for s in SUBJECTS for phase in ['headers', 'crop-mr']]
assert [(o['subject'], o['phase']) for o in receipt['outcomes']] == expected_pairs
rows, phase_projection, artifacts = [], [], []
reader_sha = next(v['sha256'] for v in index['files'] if v['path'].endswith('/src/resectionlab/remind_planning_qc.py'))
for subject in SUBJECTS:
    case, case_pin = read(BASE / 'bindings' / (subject + '-case.json'))
    reports = {}
    for phase in ['headers', 'crop-mr']:
        outcome = next(v for v in receipt['outcomes'] if (v['subject'], v['phase']) == (subject, phase))
        supervisor, supervisor_pin = read(ACTUAL / (subject + '-' + phase + '-supervisor.json'))
        assert supervisor == outcome
        assert outcome['exit_code'] == 0 and outcome['phase_complete'] and outcome['reaped']
        assert not outcome['remaining_owned_pids'] and not outcome['cleanup_errors'] and not outcome['cleanup_actions']
        assert outcome['stop_reason'] is None and outcome['result_guard_error'] is None
        assert outcome['sampled_peak_rss_bytes'] < receipt['caps']['sampled_rss_bytes']
        assert outcome['case_elapsed_seconds'] < receipt['caps']['seconds_per_case']
        assert outcome['case_output_bytes'] < receipt['caps']['output_bytes_per_case']
        assert outcome['final_phase_log_bytes'] < receipt['caps']['log_bytes_per_phase']
        name = 'result.json' if phase == 'headers' else 'conversion-result.json'
        report, pin = read(ACTUAL / (subject + '-' + phase) / name)
        assert pin['sha256'] == outcome['result_sha256'] and report['status'] == outcome['result_status']
        assert report['case_sha256'] == case_pin['sha256'] and report['executing_script_sha256'] == reader_sha
        assert report['patient_id'] == subject and report['role'] == case['role'] == 'TRAIN'
        assert report['immutable_role_binding']['cohort_sha256'] == public['cohort']['sha256']
        assert report['public_only'] and not report['private_reference_loaded'] and not report['training_admitted']
        assert report['optimizer_updates_performed'] == 0 and not report['actor_inputs_admitted']
        reports[phase] = (report, pin)
        phase_projection.append({k: outcome[k] for k in outcome if k != 'argv'} | {'result': pin, 'supervisor': supervisor_pin})
    h, hp = reports['headers']; c, cp = reports['crop-mr']
    assert h['pixel_decode_calls'] == 0 and c['header_snapshot_sha256'] == hp['sha256']
    assert {v['kind'] for v in h['series']} == {'structural_t1ce', 'whole_tumor', 'cerebrum'}
    assert c['selected_MR_pixels_match_independent_raw_bytes'] and not c['registration_performed']
    rel = c['public_target_support_relation']; target = c['annotations']['whole_tumor']; support = c['annotations']['cerebrum']
    assert not rel['support_filled_or_modified'] and not rel['target_clipped']
    assert rel['whole_tumor_positive_voxels'] == target['placed_positive_voxels']
    assert abs(rel['outside_fraction'] - rel['whole_tumor_positive_outside_public_support'] / rel['whole_tumor_positive_voxels']) < 1e-14
    row = {'patient_id': subject, 'role': 'TRAIN', 'status': 'SOURCE_RESULT_UNREVIEWED_NO_TRAINING_ADMISSION', 'shape_xyz': c['shape_xyz'], 'header_result': hp, 'conversion_result': cp,
           'placed_target_positive_voxels': rel['whole_tumor_positive_voxels'], 'placed_target_outside_support_voxels': rel['whole_tumor_positive_outside_public_support'], 'outside_support_fraction': rel['outside_fraction'],
           'placed_target_inside_support_voxels': rel['whole_tumor_positive_voxels'] - rel['whole_tumor_positive_outside_public_support'], 'annotations': {}}
    for kind, a in [('whole_tumor', target), ('cerebrum', support)]:
        p = a['placement']; match = a['correspondence']
        row['annotations'][kind] = {k: p[k] for k in ['source_positive_voxels', 'source_positive_centres_outside_target_grid', 'source_positive_volume_mm3', 'resampled_positive_voxels', 'resampled_positive_volume_mm3', 'resampling_can_omit_subvoxel_labels']}
        row['annotations'][kind].update({k: match[k] for k in ['correspondence', 'frame_uid_matches', 'explicit_SOP_refs_present', 'expert_alignment_review', 'registration_performed']})
        row['annotations'][kind]['source_samples_equal_recorded'] = a['source_samples_equal']
    rows.append(row)
    for name, a in c['artifacts'].items():
        p = ACTUAL / (subject + '-crop-mr') / name
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode) and s.st_size == a['bytes']
        artifacts.append({'patient_id': subject, 'path': str(p.relative_to(ROOT)), 'bytes': s.st_size, 'recorded_sha256': a['sha256'], 'fresh_body_hash_performed': False, 'stat_only': {'device': s.st_dev, 'inode': s.st_ino, 'mtime_ns': s.st_mtime_ns}})
OUT.mkdir(exist_ok=True)
summary = {'schema': 'remind-next-four-saved-metadata-audit-v1', 'audit_scope': 'Saved JSON and source hashes; output array/PNG stat only. No DICOM, array or image body opened.', 'planned_denominator': 4, 'converted': 4, 'reviewed_anatomy': 0, 'training_admitted': 0, 'failed_cases_replaced': False, 'elapsed_seconds': receipt['elapsed_seconds'], 'sampled_peak_rss_bytes': max(v['sampled_peak_rss_bytes'] for v in receipt['outcomes']), 'all_eight_phases_pass': True, 'all_owned_children_reaped_recorded': True, 'source_index_files_rehashed': len(index['files']), 'root_release_delta_only_execution_released': True, 'rows': rows, 'phases': phase_projection}
write('summary.json', summary)
write('deferred-saved-review.json', {'execution_released': False, 'patient_ids': SUBJECTS, 'input_scope': 'Only these already-converted public arrays and saved overlays, no source DICOM or private annotations.', 'artifacts': artifacts, 'reference_reviewer': metadata(ROOT / 'build/remind-select-public-preparation-v1/audit_saved_public.py')[0], 'required_adaptation': ['Replace hardcoded SELECT IDs and visual claims with fixed four TRAIN identities.', 'Verify saved artifact hashes then plane-stream finite/binary values, geometry/domain coverage and full target/support counts.', 'Inspect existing public overlays for alignment/source artifacts without silently clearing uncertainty.', 'Do not generate runnable public manifests or admit training automatically.', 'Retain all four outcomes and full native/placed target/crop/support denominators, including any held source artifacts.'], 'semantic_limit': 'Brainlab cerebrum is automatic supplied estimated support, not proven healthy/removable tissue. Manual whole tumor is not a prescribed resection target; missing source-domain labels are unknown. Separate source-artifact/coverage judgment and root dispatch remain required.'})
write('preservation-index.json', {'schema': 'remind-next-four-compact-preservation-index-v1', 'metadata_and_source_rehashed': sorted(READS.values(), key=lambda x: x['path']), 'array_and_overlay_bodies_not_opened': True, 'duplicate_raw_header_uid_lists_or_arrays_in_package': False, 'receipt': receipt_pin, 'declaration': declaration_pin, 'root_release': release_pin, 'prepared_release_preserved': prepared_pin, 'pending_qualification': pending_pin})
print(json.dumps({'out': str(OUT.relative_to(ROOT)), 'phases': len(phase_projection), 'rows': [{k:r[k] for k in ['patient_id','placed_target_positive_voxels','placed_target_outside_support_voxels','outside_support_fraction']} for r in rows]}, indent=2))
