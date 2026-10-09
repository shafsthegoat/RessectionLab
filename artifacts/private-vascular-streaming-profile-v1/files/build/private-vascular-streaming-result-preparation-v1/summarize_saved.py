"""Saved JSON/source hashes only. Never imports or reruns geometry/models."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREP = ROOT/'build/private-vascular-streaming-preparation-v1'
RUN = ROOT/'build/private-vascular-streaming-profile-v1'
FORBIDDEN = []

def guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'socket.connect'):
        FORBIDDEN.append(event); raise RuntimeError('No process/network in saved-only summary')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower()
        if name.endswith(('.mat', '.nii', '.nii.gz', '.npy', '.npz', '.tar', '.pt', '.ckpt', '.dcm', '.bin', '.h5')):
            FORBIDDEN.append(name); raise RuntimeError('No patient/model/array payload in saved-only summary')
sys.addaudithook(guard)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

receipt = read(RUN/'receipt.json')
profile = read(RUN/'worker/profile.json')
release = read(PREP/'root-supervisor-release.json')
disabled = read(PREP/'supervisor-disabled-release.json')
preflight = read(PREP/'root-preflight.json')
assert release == {**disabled, 'execution_released': True}
assert sha(PREP/'root-supervisor-release.json') == receipt['root_release_sha256'] == preflight['release_sha256']
assert sha(ROOT/'build/private-vascular-streaming-launch-independent-v1/REPORT.md') == preflight['review_sha256']
assert receipt['source_bindings'] == release['source_bindings']
for name, expected in receipt['source_bindings'].items(): assert sha(ROOT/name) == expected
repository = read(PREP/'repository-source-pins.json')
for name, expected in repository.items(): assert sha(ROOT/name) == expected
assert len(repository) == profile['repository_sources_verified'] == 77
assert {str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()} == set(receipt['output_bindings']) | {'receipt.json'}
assert not any(p.is_symlink() for p in RUN.rglob('*'))
for name, binding in receipt['output_bindings'].items():
    path = RUN/name
    assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
assert sha(RUN/'worker/profile.json') == receipt['profile_sha256']
assert receipt['status'] == 'generated_streaming_profile_complete'
stage, cleanup = receipt['readout_stage'], receipt['cleanup']
assert stage['exit_code'] == 0 and stage['kill_reason'] is None and stage['status'] == 'completed_within_caps'
assert cleanup['contained'] and cleanup['direct_child_reaped'] and not cleanup['fallback_used']
assert cleanup['errors'] == cleanup['remaining_members'] == [] and receipt['readout_calls_attempted'] == 1
assert not (RUN/'worker/unused-pycache').exists() and not (PREP/'launcher-unused-pycache').exists()
result, work = profile['result'], profile['result']['work']
assert profile['source_bytes_unchanged'] and profile['prohibited_actions'] == []
assert result['per_action']['first'] == result['per_action']['repeated']
assert result['whole_tool']['touched_reference_cells'] == work['reference_sampled_cells'] == 21314
assert result['whole_tool']['positive_reference_cells'] == 60 and result['whole_tool']['unknown_reference_cells'] == 9956
assert result['whole_tool']['outside_reference_fov'] and not result['whole_tool']['annotation_coverage_complete_for_sweep']
assert work['tiles_pruned'] + work['tiles_evaluated'] == work['tiles_scanned'] == 3072
for record in [result['shaft'], result['tip'], result['whole_tool'], *result['per_action'].values()]:
    assert record['positive_reference_cells'] + record['unknown_reference_cells'] <= record['touched_reference_cells']
    assert record['biological_vessel_free'] is None and record['clinical_injury_probability'] is None
    assert math.isclose(record['positive_cell_volume_upper_bound_mm3'], record['positive_reference_cells']*.7**3, rel_tol=1e-12)
assert result['removed_overlap']['outcomes'] is None and result['strategy_replay_or_admission_performed'] is False
total = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
assert total == 15401 and total <= receipt['caps']['aggregate_output_bytes']
summary = {
    'status': 'saved_generated_profile_verified',
    'receipt_sha256': sha(RUN/'receipt.json'), 'profile_sha256': sha(RUN/'worker/profile.json'),
    'root_release_sha256': sha(PREP/'root-supervisor-release.json'), 'root_preflight_sha256': sha(PREP/'root-preflight.json'),
    'root_source_head': preflight['source_head'], 'repository_sources_verified': len(repository),
    'grid_voxels': result['grid_voxels'], 'grid_shape': result['grid_shape'], 'capsules': 6, 'actions': 3,
    'kernel_profile_seconds_with_tracing': profile['profile_seconds'],
    'whole_child_seconds': stage['elapsed_seconds'], 'worker_cleanup_finalization_seconds': receipt['lifecycle_seconds'],
    'peak_traced_kernel_bytes': profile['tracemalloc_peak_bytes'],
    'process_peak_rss_bytes': profile['process_peak_rss_bytes'],
    'peak_sampled_group_rss_bytes': stage['peak_sampled_process_group_rss_bytes'],
    'saved_output_bytes': total, 'work': work,
    'tiles_pruned_percent': 100*work['tiles_pruned']/work['tiles_scanned'],
    'grid_cells_sampled_percent': 100*work['reference_sampled_cells']/result['grid_voxels'],
    'mask_contacts': {name: result[name] for name in ('shaft', 'tip', 'whole_tool')},
    'per_action': result['per_action'], 'repeated_action_counts_equal': True,
    'hypothetical_legacy_largest_bounding_box_cells': profile['hypothetical_legacy_largest_bounding_box_cells'],
    'hypothetical_legacy_four_coordinate_arrays_bytes': profile['hypothetical_legacy_four_coordinate_arrays_bytes'],
    'legacy_comparison_scope': 'Arithmetic allocation estimate only; legacy method was not run. No empirical speedup or measured memory-ratio claim.',
    'cleanup': cleanup, 'original_data_or_model_access': False, 'extra_grid_runs': 0,
    'claim_scope': 'Generated procedural labels and explicit capsule geometry only; no patient admission, image I/O performance, full strategy admission, removed overlap or biological injury validation.',
    'prohibited_actions_during_saved_summary': FORBIDDEN}
with (OUT/'summary.json').open('x') as stream:
    json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
print(json.dumps({'summary_sha256': sha(OUT/'summary.json'), 'receipt_sha256': summary['receipt_sha256'],
                  'profile_sha256': summary['profile_sha256'], 'tiles_pruned_percent': summary['tiles_pruned_percent'],
                  'traced_peak_MiB': summary['peak_traced_kernel_bytes']/1024**2,
                  'process_peak_MiB': summary['process_peak_rss_bytes']/1024**2}, sort_keys=True))
