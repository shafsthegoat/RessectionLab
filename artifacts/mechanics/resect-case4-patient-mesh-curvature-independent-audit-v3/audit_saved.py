"""Saved-only independent audit. Standard library; no array or native imports.

Hashes anatomy containers as opaque bytes, never decodes them. No experiment,
launcher, supervisor, Gmsh, VTK, image, landmark, or solver API is invoked.
"""
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT/'artifacts/mechanics/resect-case4-patient-mesh-curvature-v3'
OUT = Path(__file__).resolve().parent
INDEX_SHA = 'd3c8c47a3e02a3fc76c29a61043a2bc85fe606f1852b5dbc68d72fbbf5dd839a'
RELEASE_SHA = 'ac226fb7ff677ff9d06e204be7f565a0e9a6b30240e9506444ec2a0f1d54ffe9'
COMMIT = 'b8de4416e777b11d31f1b75adbf164522a92eaa3'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((OWNER/name).read_text())


def verify_files(base, files):
    for name, entry in files.items():
        path = base/name
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == entry['bytes'], name
        assert sha(path) == entry['sha256'], name


def audit():
    assert sha(OWNER/'artifact-index.json') == INDEX_SHA
    index = read('artifact-index.json')
    verify_files(OWNER, index['files'])
    assert len(index['files']) == index['file_count'] == 12
    assert sum(x['bytes'] for x in index['files'].values()) == index['total_bytes'] == 49812
    release = read('release.json'); assert sha(OWNER/'release.json') == RELEASE_SHA
    assert release['authorized'] is True and release['source_commit'] == COMMIT
    assert all(release[x] is False for x in ['clinical_validation', 'anatomical_registration_accepted', 'solver_authorized'])
    assert release['reuse_saved_native_surface_only'] is True
    archive = Path(release['source_directory']); raw = Path(release['attempt_directory'])
    assert len(release['source_sha256']) == 10
    assert {str(p.relative_to(archive)) for p in archive.rglob('*') if p.is_file()} == set(release['source_sha256'])
    for name, expected in release['source_sha256'].items():
        assert sha(archive/name) == expected, name
        # Read immutable Git bytes, without executing any archived code.
        contents = subprocess.run(['git', '-C', str(ROOT), 'show', f'{COMMIT}:{name}'],
                                  capture_output=True, check=True, timeout=5).stdout
        assert hashlib.sha256(contents).hexdigest() == expected, name
    baseline = read('execution-baseline.json'); post = read('post-execution-verification.json')
    acceptance = read('saved-records/acceptance.json')
    worker = read('saved-records/worker.json'); candidate = read('saved-records/candidate/result.json')
    supervisor = read('saved-records/supervision/supervision.json'); summary = read('summary.json')
    preflight = read('preflight.json')
    bindings = baseline['inputs_sha256']
    assert len(bindings) == baseline['input_count'] == preflight['bound_file_count'] == 27
    assert bindings == worker['input_sha256']
    assert set(bindings) == set(post['input_checks']) == set(acceptance['inputs_after'])
    assert all(post['input_checks'].values()) and all(acceptance['inputs_after'].values())
    for path, expected in bindings.items(): assert sha(path) == expected, path
    assert baseline['existing_attempt_absent'] is True and preflight['attempt_directory_exists'] is False
    assert datetime.fromisoformat(release['root_authorization']['utc']) < datetime.fromisoformat(baseline['verified_at_utc']) < datetime.fromisoformat(post['checked_at_utc'])
    raw_index = read('raw-output-index.json')
    assert Path(raw_index['raw_directory']) == raw
    assert raw_index['files'] == post['raw_files']
    verify_files(raw, raw_index['files']); verify_files(OWNER/'saved-records', raw_index['files'])
    assert {str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()} == set(raw_index['files'])
    assert len(raw_index['files']) == raw_index['file_count'] == 5
    assert sum(x['bytes'] for x in raw_index['files'].values()) == raw_index['total_bytes'] == 22993
    for entry in post['prior_attempts'].values():
        previous_path = ROOT/entry['index_path']
        assert sha(previous_path) == entry['index_sha256']
        previous = json.loads(previous_path.read_text())
        # Older index uses output_directory; current index uses raw_directory.
        previous_root = Path(previous.get('raw_directory', previous.get('output_directory', '')))
        assert previous_root.is_absolute()
        verify_files(previous_root, previous['files'])
        assert set(entry['checks']) == set(previous['files']) and all(entry['checks'].values())
    config = json.loads((archive/'manifests/experiments/resect-case4-patient-mesh-curvature-v3.json').read_text())
    first = json.loads((archive/'manifests/experiments/resect-case4-patient-mesh-v1.json').read_text())
    assert config['quality'] == first['quality'] and config['surface_fidelity'] == first['surface_fidelity']
    assert config['surface_fidelity']['maximum_distance_m'] == .002
    assert config['surface_fidelity']['maximum_relative_volume_error'] == .03
    assert config['gmsh_options']['Mesh.MeshSizeFromCurvature'] == 24
    assert config['global_size_profile']['minimum_m'] == .003
    assert config['gmsh_options']['Mesh.SecondOrderLinear'] == 1
    assert config['caps'] == baseline['caps']
    assert supervisor == acceptance['supervision']
    assert supervisor['command'] == acceptance['command']
    assert supervisor['command'][3] == 'worker' and baseline['command'][3] == 'run'
    assert supervisor['wall_cap_seconds'] == 180 and supervisor['rss_cap_bytes'] == 3*1024**3
    assert supervisor['exit_code'] == -9 and supervisor['kill_reason'] == 'supervision_exception'
    assert supervisor['error']['type'] == 'TimeoutExpired' and '/bin/ps' in supervisor['error']['message']
    observer_timeout = float(re.search(r'timed out after ([0-9.]+) seconds', supervisor['error']['message']).group(1))
    assert 0 < observer_timeout < .02
    assert supervisor['elapsed_seconds'] == summary['supervised_elapsed_seconds'] == 180.0281012909254
    assert supervisor['sampled_peak_process_group_rss_bytes'] == summary['sampled_peak_process_group_rss_bytes'] == 590708736
    assert supervisor['cleanup_error'] is None and post['worker_process_group_survivors'] == []
    assert supervisor['no_retry'] is True
    assert summary['launcher_exit_code'] == 1
    assert acceptance['status'] == supervisor['status'] == summary['status'] == 'failed_or_incomplete'
    assert worker['status'] == candidate['status'] == 'running'
    assert worker['native_generation_calls'] is None and candidate['native_generation_calls'] == 1
    assert worker['source_arrays_reused'] is True and worker['MRI_reextracted'] is False and worker['B_or_V_access'] is False
    assert worker['solver_calls'] == candidate['solver_calls'] == candidate['retries'] == 0
    assert candidate['solver_admitted'] is False
    absent = ['returned_nodes', 'returned_elements', 'diagnostic', 'quality', 'source_to_mesh', 'mesh_to_source', 'relative_volume_error']
    assert not any(name in candidate for name in absent)
    assert not (raw/'candidate/diagnostic').exists() and not (raw/'candidate/diagnostic.partial').exists()
    assert summary['returned_nodes'] is None and summary['returned_tet10_elements'] is None
    assert summary['quality_and_fidelity_status'] == 'not_assessed_no_returned_mesh'
    assert not summary['complete_diagnostic_available'] and not summary['candidate_accepted']
    assert all(value == ('FALSE' if key == 'OMP_DYNAMIC' else 'Sequential' if key == 'VTK_SMP_IMPLEMENTATION_TYPE' else '1')
               for key, value in acceptance['thread_environment'].items())
    guard = acceptance['output_guard']
    assert guard['aggregate_bytes_cap'] == 32*1024**2 and guard['per_file_bytes_cap'] == 8*1024**2 and guard['maximum_files'] == 32
    assert guard['error'] is None and guard['observations'] == 1395
    assert worker['per_file_hard_limit_bytes'] == 8*1024**2
    assert raw_index['total_bytes'] < guard['aggregate_bytes_cap']
    assert max(x['bytes'] for x in raw_index['files'].values()) < guard['per_file_bytes_cap']
    assert guard['final_usage']['bytes'] + raw_index['files']['acceptance.json']['bytes'] == raw_index['total_bytes']
    log = (OWNER/'saved-records/supervision/combined.log').read_text()
    assert log.count('Info    : Meshing 1D...') == 1 and log.count('Info    : Meshing 2D...') == 1
    assert 'Meshing 3D' not in log and 'Done meshing 2D' not in log
    stages = {name: {'wall_seconds': float(wall), 'cpu_seconds': float(cpu)}
              for name, wall, cpu in re.findall(r'Done (.*?) \(Wall ([0-9.eE+-]+)s, CPU ([0-9.eE+-]+)s\)', log)}
    for name, value in summary['native_logged_stage_times'].items(): assert stages[name] == value, name
    assert stages['meshing 1D']['wall_seconds'] == 170.079
    assert log.rstrip().endswith('Meshing surface 2 (Discrete surface, Frontal-Delaunay)')
    # Full index reread at end guards accidental alteration during review.
    verify_files(OWNER, index['files']); assert sha(OWNER/'artifact-index.json') == INDEX_SHA
    return {'schema': 'case4-curvature-saved-independent-audit-v3',
        'status': 'passed_saved_record_audit_of_failed_attempt', 'reviewed_commit': COMMIT,
        'owner_index_sha256': INDEX_SHA, 'release_sha256': RELEASE_SHA,
        'verified': {'compact_records': 12, 'raw_records_and_lossless_copies': 5,
            'raw_bytes': 22993, 'exact_git_archive_files': 10, 'input_bindings_rehashed': 27,
            'prior_attempt_indices_and_opaque_raw_outputs_unchanged': 2,
            'physical_gates_unchanged': True, 'single_generation_charged_before_entry': True,
            'no_volume_or_diagnostic_returned': True, 'no_retry_evidence': True},
        'terminal_result': {'status': 'failed_or_incomplete', 'worker_exit_code': -9,
            'kill_reason': 'supervision_exception', 'error_type': 'TimeoutExpired',
            'observer_timeout_seconds': observer_timeout,
            'elapsed_seconds': supervisor['elapsed_seconds'],
            'elapsed_over_180s_cap_seconds': supervisor['elapsed_seconds']-180,
            'sampled_peak_group_rss_bytes': supervisor['sampled_peak_process_group_rss_bytes'],
            'last_completed_stage': 'meshing 1D', 'stage_wall_seconds': 170.079,
            'last_logged_stage': '2D surface2', 'returned_nodes': None,
            'returned_elements': None, 'quality': 'unassessed', 'fidelity': 'unassessed'},
        'interpretation': 'Time-budget-adjacent supervision exception; no tested geometry rejection or acceptance. Source-bound checkpoints accurately preserve interrupted running/null state.',
        'scope': {'array_decodes': 0, 'image_or_landmark_inspection': 0,
            'native_calls': 0, 'distance_queries': 0, 'mesh_or_solver_replays': 0,
            'anatomy_container_access': 'Opaque SHA256 only; no arrays, images or metadata decoded.',
            'process_cleanup': 'Saved cleanup and postflight survivor receipts checked; no live process or performance experiment repeated.'},
        'limitations': ['RSS peaks between samples may be missed; the180s cap includes28.1ms observed supervision/cleanup overhead.',
            'Stage timings are one attempted run, not a controlled performance comparison.',
            'No mesh, count, quality, topology, overlap, volume or surface-fidelity result exists.',
            'No solver admission, retry or new experiment is authorized by this audit.'],
        'remaining_reporting_blockers': []}


if __name__ == '__main__':
    result = audit()
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'verification_sha256': sha(OUT/'verification.json')}))
