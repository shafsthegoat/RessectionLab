"""Saved-record and packet-framing audit; no image, geometry API or solver reads."""
from pathlib import Path
import hashlib
import json
import os
import stat
import subprocess
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT / 'artifacts/mechanics/resect-case4-patient-mesh-graded-v2'
OUT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    start = time.monotonic()
    index_hash = sha(OWNER / 'artifact-index.json')
    assert index_hash == '8c3298d014cbf638487fca30f1d46475dadd6c2f440a1550134d77b9a5d36043'
    index = read(OWNER / 'artifact-index.json')
    for name, item in index['files'].items():
        path = OWNER / name
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    assert sum(i['bytes'] for i in index['files'].values()) == index['total_bytes']
    raw_index = read(OWNER / 'raw-output-index.json')
    raw = Path(raw_index['raw_directory'])
    paths = {str(p.relative_to(raw)): p for p in raw.rglob('*') if p.is_file()}
    assert set(paths) == set(raw_index['files'])
    for name, item in raw_index['files'].items():
        assert not paths[name].is_symlink()
        assert paths[name].stat().st_size == item['bytes'] and sha(paths[name]) == item['sha256']
    assert sum(p.stat().st_size for p in paths.values()) == raw_index['total_bytes'] == 446610
    for name in index['files']:
        if name.startswith('saved-records/'):
            assert sha(OWNER / name) == sha(raw / name.removeprefix('saved-records/'))

    release = read(OWNER / 'release.json')
    archive = Path(release['source_directory'])
    assert release['source_commit'] == '715f0bf99fdd76eb984e5522aa2c7fa538498f63'
    assert release['authorized'] is True and release['reuse_saved_native_surface_only'] is True
    assert all(release[k] is False for k in ('clinical_validation', 'anatomical_registration_accepted', 'solver_authorized'))
    archive_paths = {str(p.relative_to(archive)): p for p in archive.rglob('*') if p.is_file()}
    assert set(archive_paths) == set(release['source_sha256']) and len(archive_paths) == 7
    for name, expected in release['source_sha256'].items():
        committed = subprocess.run(['git', '-C', str(ROOT), 'show', release['source_commit'] + ':' + name], capture_output=True, check=True, timeout=5).stdout
        assert hashlib.sha256(committed).hexdigest() == sha(archive / name) == sha(ROOT / name) == expected
        assert stat.S_IMODE((archive / name).stat().st_mode) == 0o444
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o555 for p in [archive, *[p for p in archive.rglob('*') if p.is_dir()]])

    baseline = read(OWNER / 'execution-baseline.json')
    after = read(OWNER / 'post-execution-verification.json')
    worker = read(raw / 'worker.json')
    accepted = read(raw / 'acceptance.json')
    summary = read(OWNER / 'summary.json')
    result = read(raw / 'candidate/result.json')
    supervision = read(raw / 'supervision/supervision.json')
    assert baseline['attempt_existed'] is False and baseline['native_generations_so_far'] == 0
    assert len(baseline['file_sha256']) == len(after['input_bindings']) == 28
    # Deliberately do not read the original mask or original native-surface array archive.
    excluded = [p for p in baseline['file_sha256'] if p.endswith('.nii.gz') or p.endswith('/native-surface.npz')]
    assert len(excluded) == 2
    for name, expected in baseline['file_sha256'].items():
        record = after['input_bindings'][name]
        assert record == {'expected': expected, 'actual': expected, 'unchanged': True}
        if name not in excluded:
            assert sha(name) == expected
    assert len(worker['input_sha256']) == len(worker['inputs_after']) == len(accepted['inputs_after']) == 19
    for name, expected in worker['input_sha256'].items():
        assert baseline['file_sha256'][name] == expected
        assert worker['inputs_after'][name] is True and accepted['inputs_after'][name] is True
    assert worker['source_arrays_reused'] and not worker['MRI_reextracted'] and not worker['B_or_V_access']
    assert worker['native_generation_calls'] == result['native_generation_calls'] == summary['native_generation_calls'] == 1
    assert worker['solver_calls'] == result['solver_calls'] == 0 and result['retries'] == 0
    assert worker['status'] == accepted['status'] == result['status'] == 'failed_or_incomplete'
    assert result['error']['message'] == 'Independent surface fidelity rejected'

    packet = raw / 'candidate/diagnostic'
    manifest = read(packet / 'manifest.json')
    assert manifest['diagnostic_complete'] is True and manifest['candidate_accepted'] is False
    assert manifest['coordinate_frame'] == 'native Case4 T1 RAS' and manifest['coordinate_units'] == 'm'
    assert manifest['gmsh_to_febio_permutation'] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
    assert set(manifest['files']) == {'nodes_m.npy', 'tet10_indices.npy', 'gmsh_node_ids.npy', 'gmsh_element_ids.npy'}
    arrays = {}
    framing = {}
    for name, item in manifest['files'].items():
        path = packet / name
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        with path.open('rb') as stream:
            assert np.lib.format.read_magic(stream) == (1, 0)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            header_bytes = stream.tell()
            payload = stream.read()
        value = np.load(path, allow_pickle=False)
        assert not fortran and tuple(shape) == value.shape == tuple(item['shape'])
        assert str(dtype) == str(value.dtype) == item['dtype']
        assert header_bytes + value.nbytes == item['bytes'] and payload == value.tobytes(order='C')
        assert np.isfinite(value).all()
        arrays[name] = value
        framing[name] = {'shape': list(shape), 'dtype': str(dtype), 'header_bytes': header_bytes, 'payload_bytes': value.nbytes, 'sha256': item['sha256']}
    nodes, cells = arrays['nodes_m.npy'], arrays['tet10_indices.npy']
    assert nodes.shape == (5223, 3) and cells.shape == (2761, 10)
    assert cells.min() >= 0 and cells.max() < len(nodes)
    assert all(len(set(row)) == 10 for row in cells)
    for name, count in [('gmsh_node_ids.npy', 5223), ('gmsh_element_ids.npy', 2761)]:
        ids = arrays[name]
        assert ids.shape == (count,) and (ids > 0).all() and len(np.unique(ids)) == count
    assert sum(p.stat().st_size for p in packet.iterdir()) == result['diagnostic']['total_bytes'] == 413290 < 2097152
    assert manifest['counts'] == result['diagnostic']['counts'] == {'nodes': 5223, 'tet10_elements': 2761}
    assert sha(packet / 'manifest.json') == result['diagnostic']['manifest_sha256'] == summary['diagnostic_manifest_sha256']
    for name, expected in worker['candidate_output_sha256'].items():
        assert sha(raw / name) == expected and accepted['candidate_output_checks'][name] is True

    config = read(archive / 'manifests/experiments/resect-case4-patient-mesh-graded-v2.json')
    assert result['returned_nodes'] <= config['eligibility']['maximum_nodes'] == 6000
    assert result['returned_elements'] <= config['eligibility']['maximum_elements'] == 6000
    assert result['quality'] == summary['quality']
    for name in ('source_to_mesh', 'mesh_to_source'):
        metric = result[name]
        assert metric == summary[name]
        assert metric['maximum_sample_distance_m'] > config['surface_fidelity']['maximum_distance_m'] == 0.002
        assert metric['full_surface_upper_bound_m'] >= metric['maximum_sample_distance_m']
        assert metric['sample_count'] <= config['surface_fidelity']['maximum_samples_per_direction']
    assert result['relative_volume_error'] == summary['relative_volume_error_computed'] < 0.03
    assert result['solver_admitted'] is False and summary['status'] == 'rejected_independent_surface_fidelity'
    assert supervision == accepted['supervision']
    assert supervision['exit_code'] == 1 and supervision['kill_reason'] is None and supervision['error'] is None
    assert supervision['elapsed_seconds'] < supervision['wall_cap_seconds'] == 180
    assert supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 3221225472
    assert supervision['no_retry'] is True and accepted['output_guard']['error'] is None
    assert all(accepted['thread_environment'][k] == '1' for k in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VTK_SMP_MAX_THREADS'))
    assert accepted['thread_environment']['VTK_SMP_IMPLEMENTATION_TYPE'] == 'Sequential'
    assert len(paths) == 10 <= 32 and max(p.stat().st_size for p in paths.values()) <= 4194304
    assert raw_index['total_bytes'] < 8388608
    assert sha(OWNER / 'artifact-index.json') == index_hash
    report = {
        'schema': 'graded-case4-independent-saved-audit-v2', 'status': 'passed_audit_of_rejected_candidate',
        'owner_index_sha256': index_hash, 'owner_index_files_checked': len(index['files']),
        'raw_files_checked': len(paths), 'raw_bytes': raw_index['total_bytes'],
        'archive_git_and_current_source_files_checked': len(archive_paths),
        'before_after_bindings_consistent': 28, 'current_nonimage_nonsurface_bindings_rehashed': 26,
        'saved_worker_and_launcher_binding_count': 19,
        'excluded_original_inputs': [{'path': p, 'basis': 'Saved before/after hashes only; not reread in this audit.'} for p in excluded],
        'diagnostic': {'status': 'complete_untruncated_saved_packet', 'bytes': 413290, 'framing': framing,
                       'nodes': 5223, 'elements': 2761, 'indices_in_range': True, 'unique_positive_native_ids': True},
        'scientific_result': {'status': summary['status'], 'source_to_mesh': result['source_to_mesh'],
                              'mesh_to_source': result['mesh_to_source'], 'relative_volume_error': result['relative_volume_error'],
                              'quality_scope': 'Saved quality/topology/J/non-overlap records are consistent. No independent geometry recomputation in this audit.'},
        'resource_observations': {'seconds': supervision['elapsed_seconds'], 'sampled_group_peak_rss_bytes': supervision['sampled_peak_process_group_rss_bytes'], 'exit_code': 1, 'kill_reason': None},
        'limitations': ['No accepted patient mesh or solver admission.', 'Sampled distances already exceed 2 mm; cover slack alone cannot explain rejection.', 'Distance records contain aggregate maxima, not locations or per-triangle witnesses.', 'Volume agreement does not establish local surface accuracy.', 'Native surface/mask were not reread; their preservation is supported by saved execution records.', 'Resource observations are sampled and not an isolated performance benchmark.'],
        'activity': {'returned_mesh_arrays_loaded': 4, 'image_or_source_surface_arrays_loaded': 0, 'landmark_reads': 0, 'native_mesher_imports_or_calls': 0, 'geometry_recomputations': 0, 'solver_calls': 0, 'training_updates': 0},
        'audit_seconds': time.monotonic() - start,
    }
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'nodes': len(nodes), 'elements': len(cells), 'packet_bytes': 413290, 'seconds': report['audit_seconds']}))


if __name__ == '__main__':
    main()
