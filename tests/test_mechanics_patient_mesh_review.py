"""Independent analytic/mocked review; no native libraries or patient reads."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_patient_mesh', ROOT / 'scripts/mechanics_patient_mesh.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def test_full_triangle_cover_catches_an_interior_miss_despite_zero_vertex_distances():
    # Distance to three observed vertices is 1-Lipschitz and zero at all vertices.
    # Vertex-only QA would therefore miss the nonzero triangle interior entirely.
    X = np.array([[0., 0, 0], [.009, 0, 0], [0, .009, 0]])
    F = np.array([[0, 1, 2]])
    def distances(points):
        return np.linalg.norm(points[:, None] - X[None], axis=2).min(axis=1)
    result = v.directed_surface_bound(X, F, distances,
        {'cover_radius_m': .001, 'maximum_samples_per_direction': 1000})
    assert distances(X).max() == 0
    assert result['maximum_sample_distance_m'] > .004
    assert result['full_surface_upper_bound_m'] >= np.sqrt(2) * .0045
    assert result['sample_count'] == 105


def test_contact_tolerance_and_rigid_coordinate_frame_preserve_overlap_decision():
    X = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]])
    theta = .37
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    def transform(x): return x @ rotation.T + [.13, -.21, .081]
    # The second tetrahedron shares a whole face; its remaining vertex is below.
    touching = np.vstack((X, [[0., 0, -.01]]))
    assert v.reject_tetrahedral_overlap(transform(touching), np.array([[0,1,2,3], [0,2,1,4]]), maximum_pairs=10)['aabb_candidate_pairs_checked'] <= 1
    # Containment must fail even though none of the inner vertices touches a face.
    contained = np.vstack((X, X * .1 + .001))
    with pytest.raises(ValueError, match='Positive-volume'):
        v.reject_tetrahedral_overlap(transform(contained), np.arange(8).reshape(2,4), maximum_pairs=10)


def test_duplicate_geometric_corner_ids_cannot_hide_overlapping_tets():
    X = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]])
    with pytest.raises(ValueError, match='Positive-volume'):
        v.reject_tetrahedral_overlap(np.vstack((X, X)), np.arange(8).reshape(2,4), maximum_pairs=10)


def mock_release(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'; repository = tmp_path / 'repository'; output = tmp_path / 'attempt'
    archive.mkdir(); repository.mkdir()
    monkeypatch.setattr(v, 'ROOT', archive)
    names = ['scripts/mechanics_patient_mesh.py', 'scripts/febio_runtime.py', v.MANIFEST,
             'artifacts/febio-runtime-investigation-v1/prospective-runtime.json']
    for name in names:
        path = archive / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'input_bindings': []}) if name == v.MANIFEST else name)
    receipts = []
    for role in ['mask_qc', 'baseline_diagnostic']:
        p = tmp_path / (role + '.json'); p.write_text(json.dumps({'role': role, 'fixture_only': True}))
        receipts.append({'role': role, 'path': str(p), 'sha256': v.sha(p)})
    release = {'schema': v.VERSION + '-release', 'authorized': True,
        'source_commit': 'a' * 40, 'source_directory': str(archive),
        'repository_directory': str(repository), 'attempt_directory': str(output),
        'mask_qc_accepted_for_estimated_domain': True,
        'baseline_diagnostic_accepted_for_provisional_research_frame': True,
        'clinical_validation': False, 'anatomical_registration_accepted': False,
        'source_sha256': {name: v.sha(archive / name) for name in names},
        'prerequisite_receipts': receipts}
    path = tmp_path / 'release.json'
    def mocked_git(command, **_):
        assert command[:4] == ['git', '-C', str(repository), 'show']
        return SimpleNamespace(stdout=(archive / command[-1].split(':', 1)[1]).read_bytes())
    monkeypatch.setattr(v.subprocess, 'run', mocked_git)
    return archive, output, path, release


@pytest.mark.parametrize('fault', ['none', 'source', 'receipt', 'commit', 'clinical', 'mask_unreviewed', 'missing_baseline', 'attempt'])
def test_release_closure_and_prerequisite_fail_closed(tmp_path, monkeypatch, fault):
    archive, output, path, release = mock_release(tmp_path, monkeypatch)
    if fault == 'source': (archive / 'scripts/mechanics_patient_mesh.py').write_text('changed')
    elif fault == 'receipt': Path(release['prerequisite_receipts'][0]['path']).write_text('changed')
    elif fault == 'commit': release['source_commit'] = 'main'
    elif fault == 'clinical': release['clinical_validation'] = True
    elif fault == 'mask_unreviewed': release['mask_qc_accepted_for_estimated_domain'] = False
    elif fault == 'missing_baseline': release['prerequisite_receipts'].pop()
    elif fault == 'attempt': release['attempt_directory'] += '-other'
    path.write_text(json.dumps(release))
    if fault == 'none':
        protocol, bound = v.prepare_release(path, output)
        assert protocol == {'input_bindings': []} and len(bound) == 7
    else:
        with pytest.raises(ValueError): v.prepare_release(path, output)


def test_launch_preserves_an_existing_attempt_without_import_or_overwrite(tmp_path, monkeypatch):
    output = tmp_path / 'attempt'; output.mkdir()
    marker = output / 'partial.npz'; marker.write_bytes(b'preserved')
    monkeypatch.setattr(v, 'prepare_release', lambda *_: pytest.fail('No release execution on existing attempt'))
    with pytest.raises(FileExistsError): v.launch('unused', output)
    assert marker.read_bytes() == b'preserved' and list(output.iterdir()) == [marker]
