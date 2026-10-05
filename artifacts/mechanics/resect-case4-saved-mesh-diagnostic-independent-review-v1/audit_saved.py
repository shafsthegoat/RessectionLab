"""Audit saved diagnostic fields and receipts without recomputing distances."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT / 'artifacts/mechanics/resect-case4-saved-mesh-diagnostic-v1'
OUT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    start = time.monotonic()
    index_hash = sha(OWNER / 'artifact-index.json')
    assert index_hash == '8dce43edbadc4548153a656ba2b1967d83d23df03b18b41fec1caf44b722badf'
    index = read(OWNER / 'artifact-index.json')
    for name, item in index['files'].items():
        assert sha(OWNER / name) == item['sha256'] and (OWNER / name).stat().st_size == item['bytes']
    assert sum(row['bytes'] for row in index['files'].values()) == index['total_bytes']
    release, binding = read(OWNER / 'release.json'), read(OWNER / 'archive-binding.json')
    raw, archive = Path(release['attempt_directory']), Path(release['source_directory'])
    assert release['authorized'] and release['source_commit'] == binding['source_commit'] == 'afd0226be1b2619780cc9f8b6f5668ef0ba2b6fa'
    files = {str(p.relative_to(archive)): p for p in archive.rglob('*') if p.is_file()}
    assert set(files) == set(binding['source_files']) and len(files) == 6
    for name, expected in binding['source_files'].items():
        committed = subprocess.run(['git', '-C', str(ROOT), 'show', release['source_commit'] + ':' + name], check=True, capture_output=True, timeout=5).stdout
        assert hashlib.sha256(committed).hexdigest() == sha(archive / name) == sha(ROOT / name) == expected
        assert not (files[name].stat().st_mode & 0o222)
    raw_index = read(OWNER / 'raw-output-index.json')
    actual = {str(p.relative_to(raw)): p for p in raw.rglob('*') if p.is_file()}
    assert set(actual) == set(raw_index['files']) and len(actual) == 5
    for name, item in raw_index['files'].items():
        assert sha(actual[name]) == item['sha256'] and actual[name].stat().st_size == item['bytes']
    assert sum(p.stat().st_size for p in actual.values()) == 998481
    for name in index['files']:
        if name.startswith('saved-records/'):
            assert sha(OWNER / name) == sha(raw / name.removeprefix('saved-records/'))

    report, after = read(raw / 'report.json'), read(OWNER / 'post-execution-verification.json')
    assert len(report['input_hashes']) == len(report['inputs_unchanged']) == len(after['inputs']) == 17
    for name, expected in report['input_hashes'].items():
        assert report['inputs_unchanged'][name] is True and sha(name) == expected
        assert after['inputs'][name] == {'sha256': expected, 'expected_sha256': expected, 'unchanged': True}
    # Hash old geometry bytes without decoding them or constructing any geometry object.
    previous = read(ROOT / 'artifacts/mechanics/resect-case4-patient-mesh-graded-v2/raw-output-index.json')
    assert len(previous['files']) == 10
    for name, item in previous['files'].items():
        assert sha(Path(previous['raw_directory']) / name) == item['sha256']
        assert after['all_10_original_graded_attempt_files_unchanged'][name] is True
    original = read(ROOT / 'artifacts/mechanics/resect-case4-patient-mesh-graded-v2/saved-records/candidate/result.json')
    summary = read(OWNER / 'summary.json')
    accepted, supervision = read(raw / 'acceptance.json'), read(raw / 'supervision/supervision.json')
    assert report['status'] == 'completed_saved_geometry_diagnostic_only' and accepted['status'] == 'completed'
    assert not report['candidate_accepted'] and not report['seam_attribution_available']
    assert report['mesher_calls'] == report['solver_calls'] == report['image_or_landmark_reads'] == 0
    assert supervision == accepted['supervision'] and supervision['exit_code'] == 0
    assert supervision['kill_reason'] is None and supervision['error'] is None and supervision['no_retry']
    assert report['elapsed_seconds'] < 110 and supervision['elapsed_seconds'] < supervision['wall_cap_seconds'] == 120
    assert supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 2 * 1024**3
    assert accepted['output_guard']['error'] is None and 998481 < accepted['output_guard']['aggregate_bytes_cap'] == 8 * 1024**2

    directions = ('source_to_mesh', 'mesh_to_source')
    with np.load(raw / 'per-triangle-diagnostics.npz', allow_pickle=False) as saved:
        expected_keys = {name + suffix for name in directions for suffix in ('_maximum_m', '_normal_change_degrees', '_component')}
        assert set(saved.files) == expected_keys
        payload = {key: saved[key] for key in saved.files}
    details = {}
    for name, count, target_count in [('source_to_mesh', 183902, 1464), ('mesh_to_source', 1464, 183902)]:
        direction = report[name]
        for key in ('sample_count', 'maximum_sample_distance_m', 'full_surface_upper_bound_m'):
            assert direction[key] == original[name][key] == summary[name][key]
        maximum, angle, labels = (payload[name + suffix] for suffix in ('_maximum_m', '_normal_change_degrees', '_component'))
        assert maximum.shape == angle.shape == labels.shape == (count,)
        assert np.isfinite(maximum).all() and np.isfinite(angle).all()
        assert (maximum >= 0).all() and ((angle >= 0) & (angle <= 180)).all()
        assert labels.dtype.kind in 'iu' and ((labels >= -1)).all()
        selected = maximum > .002
        assert np.array_equal(labels >= 0, selected)
        stat = direction['summary']
        assert float(maximum.max()) == direction['maximum_sample_distance_m']
        assert int(selected.sum()) == stat['faces_with_sample_over_2mm'] == summary[name]['faces_with_sample_over_2mm']
        assert len(np.unique(labels[selected])) == stat['connected_witness_components'] == summary[name]['connected_witness_components']
        assert np.isclose(stat['area_of_faces_with_sample_over_2mm_m2'] / stat['area_m2'], stat['area_fraction_of_faces_with_sample_over_2mm'], rtol=1e-14)
        assert stat['area_fraction_of_faces_with_sample_over_2mm'] == summary[name]['area_fraction_of_faces_with_sample_over_2mm']
        groups = stat['largest_10_components']
        assert groups == sorted(groups, key=lambda row: (-row['area_m2'], row['first_triangle']))
        group_ids = []
        for group in groups:
            label = labels[group['first_triangle']]
            ids = np.flatnonzero(labels == label)
            assert label >= 0 and ids.min() == group['first_triangle'] and len(ids) == group['triangles']
            assert float(maximum[ids].max()) == group['maximum_sample_distance_m']
            bounds, centroid = np.asarray(group['bounds_m']), np.asarray(group['centroid_m'])
            assert ((centroid >= bounds[0]) & (centroid <= bounds[1])).all()
            group_ids.append(int(label))
        assert len(set(group_ids)) == len(groups) == 10
        assert sum(group['area_m2'] for group in groups) <= stat['area_of_faces_with_sample_over_2mm_m2']
        quantiles = list(stat['area_weighted_face_maximum_quantiles_m'].values())
        assert quantiles == sorted(quantiles) and quantiles[-1] == float(maximum.max())
        assert all(np.any(maximum == value) for value in quantiles)
        for cut in ('30', '60'):
            assert stat['edge_normal_change_association'][cut] == summary[name]['edge_normal_change_association'][cut]
            assert all(0 <= fraction <= 1 for fraction in stat['edge_normal_change_association'][cut].values())
        worst = direction['worst_10']
        assert len(worst) == 10 and worst == sorted(worst, key=lambda row: (-row['distance_m'], row['ordinal']))
        assert len({row['ordinal'] for row in worst}) == 10
        for row in worst:
            assert 0 <= row['ordinal'] < direction['sample_count']
            assert 0 <= row['triangle'] < count and 0 <= row['nearest_triangle'] < target_count
            assert row['distance_m'] <= maximum[row['triangle']] + 1e-12
            separation = np.linalg.norm(np.asarray(row['point_m']) - row['nearest_point_m'])
            assert abs(separation - row['distance_m']) < 1e-10
            assert abs(row['nearest_distance_m'] - row['distance_m']) < 1e-10
        assert worst[0]['distance_m'] == float(maximum.max())
        details[name] = {'sample_count': direction['sample_count'], 'maximum_m': float(maximum.max()),
                         'selected_faces': int(selected.sum()), 'components': len(np.unique(labels[selected])),
                         'worst_sample_entries': 10, 'unique_exact_point_tuples': len({tuple(row['point_m']) for row in worst}),
                         'saved_witness_face_area_fraction': stat['area_fraction_of_faces_with_sample_over_2mm']}
    probes = report['mesh_to_source_probe_classes']
    assert probes == summary['mesh_to_source_probe_classes']
    assert [(probes[name]['count'], probes[name]['over_2mm']) for name in ('boundary_vertices', 'boundary_midsides', 'boundary_face_centroids')] == [(732, 0), (2196, 32), (1464, 25)]
    assert probes['boundary_vertices']['maximum_m'] < 2e-17
    for probe in probes.values():
        assert probe['quantiles_m'] == sorted(probe['quantiles_m']) and probe['quantiles_m'][-1] == probe['maximum_m']
    charged = sum(report[name]['sample_count'] + len(report[name]['worst_10']) for name in directions) + sum(p['count'] for p in probes.values())
    assert charged == report['query_count'] == summary['query_count'] == 1152670 < 1200000
    assert sha(OWNER / 'artifact-index.json') == index_hash
    record = {'schema': 'case4-saved-diagnostic-independent-review-v1', 'status': 'passed_saved_evidence_audit',
              'owner_index_sha256': index_hash, 'owner_records_verified': len(index['files']), 'raw_files_verified': 5,
              'raw_bytes': 998481, 'exact_source_files_verified': 6, 'current_input_hashes_verified': 17,
              'old_graded_raw_files_unchanged': 10, 'distance_replay_exact': True, 'directional_links': details,
              'probe_summary': probes, 'charged_queries': charged,
              'audit_scope': 'Loaded only six derived per-triangle fields. Checked labels, maxima, witness endpoint arithmetic and saved ratio/order consistency. No geometry construction, distance queries or area/normal/connectivity recomputation.',
              'limits': ['Whole-face witness area is not exact excursion area; triangle areas were not independently reintegrated.', 'Component memberships match saved labels; shared-edge connectedness was established by reviewed producer code, not recomputed here.', 'Top ten entries can repeat physical points; they are not ten distinct defect sites.', 'Normal-change associations and vertex/chord observations do not establish seam causality or anatomy.', 'Candidate remains rejected; no new mesh, solver admission or settings change follows.'],
              'activity': {'derived_fields_loaded': 6, 'original_geometry_arrays_decoded': 0, 'distance_queries': 0, 'native_mesher_calls': 0, 'solver_calls': 0, 'image_or_landmark_reads': 0},
              'audit_seconds': time.monotonic() - start}
    (OUT / 'verification.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'seconds': record['audit_seconds'], 'directional_links': details}))


if __name__ == '__main__':
    main()
