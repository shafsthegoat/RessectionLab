"""Audit existing analytical receipts. Never start FEBio or access patient data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'artifacts/mechanics-patient-constraints-run-v1'
ATTEMPT = RUN / 'attempt-01'
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def primitive_records(text, count, columns, label):
    """Small independent parser requiring exact headers, states and row IDs."""
    records = []
    for block in re.split(r'(?=^\*Step\s*=)', text, flags=re.M):
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        step = int(re.fullmatch(r'\*Step\s*=\s*(\d+)', lines[0])[1])
        time = float(re.fullmatch(r'\*Time\s*=\s*(\S+)', lines[1])[1])
        assert re.fullmatch(r'\*Data\s*=\s*(\S+)', lines[2])[1] == label
        rows = [line.split(',') for line in lines[3:]]
        assert len(rows) == count and all(len(row) == columns + 1 for row in rows)
        assert [int(row[0]) for row in rows] == list(range(1, count + 1))
        values = np.array([[float(cell) for cell in row[1:]] for row in rows])
        assert np.isfinite(values).all()
        records.append((step, time, values))
    assert [(step, time) for step, time, _ in records] == list(enumerate((0., .25, .5, .75, 1.)))
    return records


def main():
    before = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
    index = read(RUN / 'artifact-index.json')
    for name, expected in index['files'].items():
        assert before[name] == expected['sha256']
        assert (RUN / name).stat().st_size == expected['bytes']
    binding = read(RUN / 'archive-binding.json')
    release = read(RUN / 'release.json')
    base = read(ATTEMPT / 'execution-baseline.json')
    result = read(ATTEMPT / 'results.json')
    execution = read(ATTEMPT / 'execution.json')
    supervision = read(ATTEMPT / 'supervision/supervision.json')
    archive = Path(binding['source_directory'])
    assert binding['source_commit'] == base['source_commit'] == release['source_commit'] == 'ddaa7e352282d159f454562cfa091cdad0a284d3'
    assert sha(binding['archive']) == binding['archive_sha256']
    assert sha(RUN / 'release.json') == binding['release_sha256']
    assert Path(release['attempt_directory']) == ATTEMPT
    assert len(base['input_hashes']) == 37
    for path, expected in base['input_hashes'].items():
        assert sha(path) == expected, path
    for name, expected in binding['exact_nine_file_closure'].items():
        assert sha(archive / name) == expected
    for record in [result['inputs_after'], execution['inputs_after'], execution['final_inputs_after']]:
        assert set(record) == set(base['input_hashes'])
        assert all(row['unchanged'] for row in record.values())
    assert execution['supervision'] == supervision
    assert execution['status'] == result['status'] == supervision['status'] == 'completed'
    assert supervision['exit_code'] == 0 and supervision['kill_reason'] is None
    assert supervision['error'] is None and supervision['cleanup_error'] is None
    assert supervision['elapsed_seconds'] < supervision['wall_cap_seconds'] == 60
    assert supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 3221225472
    assert execution['thread_environment'] == {'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE',
        'VECLIB_MAXIMUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    assert result['solver_invocations'] == 3 and result['no_retry'] is True
    assert sorted(p.name for p in RUN.glob('attempt-*') if p.is_dir()) == ['attempt-01']
    spec = importlib.util.spec_from_file_location('saved_output_frozen_checker', archive / 'scripts/mechanics_patient_constraints.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert result['checker_import_path'] == checker.__file__
    # Reuse the independent force oracle reviewed before any solver output.
    review_source = ROOT / 'tests/test_mechanics_patient_constraints_review.py'
    prior = read(ROOT / 'artifacts/mechanics-patient-constraints-review-v1/final-review.json')
    assert sha(review_source) == prior['reviewed_sha256']['tests/test_mechanics_patient_constraints_review.py']
    assert sha(ROOT / 'scripts/mechanics_patient_constraints.py') == sha(archive / 'scripts/mechanics_patient_constraints.py')
    oracle = runpy.run_path(str(review_source))
    X, E, _ = checker.fixture_mesh()
    H, parents, _, T = checker.observation_operator()
    cases = []
    assert [row['case'] for row in result['cases']] == list(checker.CASES)
    for outcome in result['cases']:
        name = outcome['case']; directory = ATTEMPT / name
        assert outcome['status'] == 'passed' and outcome['solver_exit_code'] == 0
        assert outcome['command'] == [base['executable'], '-noconfig', '-no_title', '-i', name + '.feb', '-o', name + '.log']
        for relative, expected in outcome['output_hashes'].items():
            assert sha(directory / relative) == expected
        assert (directory / (name + '.feb')).read_text() == checker.deck_xml(name)
        texts = [(directory / (name + '.' + suffix)).read_text() for suffix in ('nodes.log', 'elements.log', 'log')]
        saved = read(directory / 'checked.json')
        recomputed = checker.check_outputs(name, *texts)
        assert recomputed == saved and recomputed['passed']
        nodes = primitive_records(texts[0], 27, 9, 'mechanics_nodes_si')
        elements = primitive_records(texts[1], 6, 8, 'mechanics_elements_si')
        assert re.search(r'Number of nodes\s*\.+\s*:\s*27\b', texts[2])
        assert re.search(r'Number of solid elements\s*\.+\s*:\s*6\b', texts[2])
        assert 'version 4.13.0' in texts[2]
        assert 'Selecting linear solver skyline' in (directory / 'console.txt').read_text()
        assert [float(t) for t in re.findall(r'converged at time\s*:\s*(\S+)', texts[2])] == [.25, .5, .75, 1.]
        states = []
        for (_, time, values), (_, _, logged), checked in zip(nodes, elements, saved['states']):
            current = values[:, :3]
            internal, total_energy, _, Js, Ws, stress = oracle['independent_assembly'](current)
            means = stress.mean(axis=1)
            expected = np.column_stack((means[:, 0, 0], means[:, 1, 1], means[:, 2, 2],
                means[:, 0, 1], means[:, 1, 2], means[:, 0, 2], Js.mean(axis=1), Ws.mean(axis=1)))
            np.testing.assert_allclose(logged[:, :6], expected[:, :6], rtol=2e-6, atol=1e-5)
            np.testing.assert_allclose(logged[:, 6], expected[:, 6], rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(logged[:, 7], expected[:, 7], rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(total_energy, checked['reconstructed_energy_J'], rtol=2e-6, atol=1e-13)
            row = {'time': time, 'independent_energy_J': float(total_energy), 'minimum_gauss_J': float(Js.min()),
                'minimum_all_declared_sample_J': checked['minimum_actual_sampled_J'],
                'maximum_stress_log_difference_Pa': float(np.max(np.abs(logged[:, :6] - expected[:, :6]))),
                'maximum_J_log_difference': float(np.max(np.abs(logged[:, 6] - expected[:, 6]))),
                'maximum_density_log_difference_Pa': float(np.max(np.abs(logged[:, 7] - expected[:, 7])))}
            if name == 'tet10_affine':
                expected_position = X @ checker.affine_F(time).T
                np.testing.assert_allclose(current, expected_position, rtol=0, atol=1e-10)
                np.testing.assert_allclose(values[:, 6:], -internal, rtol=2e-6, atol=1e-8)
                row.update(free_center_error_m=float(np.max(np.abs(current[11] - expected_position[11]))),
                    all_node_position_error_m=float(np.max(np.abs(current - expected_position))),
                    maximum_signed_reaction_difference_N=float(np.max(np.abs(values[:, 6:] + internal))))
            else:
                errors = H @ values[:, 3:6] - time * checker.targets(name)
                assert np.max(np.abs(errors)) <= 1e-10
                force = (T.T @ internal).reshape(-1)
                derivatives = np.array(checked['virtual_work']['directional_derivatives_N'])
                assert derivatives.shape == (72, 3)
                assert checked['virtual_work']['steps_m'] == [2e-7, 1e-7, 5e-8]
                assert all(checked['virtual_work']['checks'].values())
                for step in range(3):
                    np.testing.assert_allclose(derivatives[:, step], force, atol=2e-9, rtol=1e-6)
                assert np.max(np.abs(force)) <= 1e-7
                row.update(maximum_Hu_error_m=float(np.max(np.abs(errors))),
                    independently_assembled_reduced_force_N=force.tolist(),
                    maximum_independent_reduced_force_N=float(np.max(np.abs(force))),
                    maximum_finest_energy_derivative_N=float(np.max(np.abs(derivatives[:, -1]))),
                    maximum_force_derivative_difference_N_by_step=np.max(np.abs(derivatives - force[:, None]), axis=0).tolist(),
                    parent_reactions_used_as_equilibrium_evidence=False)
            states.append(row)
        cases.append({'case': name, 'states': states, 'node_count': 27, 'element_count': 6,
            'frozen_checker_recomputed_exactly': True, 'logged_final_residuals_N2': saved['solver']['last_squared_force_residual_norms_N2']})
    after = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
    assert before == after
    for path, expected in base['input_hashes'].items():
        assert sha(path) == expected
    receipt = {'status': 'saved_output_independent_numerical_review_passed', 'source_commit': binding['source_commit'],
        'script_sha256': sha(Path(__file__)), 'oracle_sha256': sha(review_source),
        'reviewed_run_files_sha256': before, 'bound_input_count_rehashed_before_and_after': 37,
        'resource_receipt': supervision, 'release_prepared_utc': release['prepared_utc'],
        'actual_time_scope': 'All three records contain exact initial plus 0.25,0.5,0.75,1 load fractions and four converged states. Timing is the recorded single supervised attempt, not a performance benchmark.',
        'observed_attempt_directories': ['attempt-01'], 'recorded_solver_invocations': 3,
        'concurrent_workload': release['concurrent_workload_context'], 'cases': cases,
        'limitations': ['Three analytic fixtures only; no mesh convergence, patient material fit or clinical validation.',
            'Nodal averages are not the proposed 5mm volume-tent operator.',
            'Jacobian positivity is sampled, not a global curved-element proof.',
            'RSS is sampled; brief peaks between samples may be missed.',
            'One-attempt claim is supported by release binding and saved artifacts; this does not inspect unrelated system activity.'],
        'solver_rerun': False, 'patient_data_accessed': False, 'original_receipts_modified': False}
    with (OUT / 'numerical-verification.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'cases': [{
        'case': case['case'], 'final_energy_J': case['states'][-1]['independent_energy_J'],
        'maximum_independent_reduced_force_N': max((s.get('maximum_independent_reduced_force_N', 0) for s in case['states']))
    } for case in cases]}, indent=2))


if __name__ == '__main__':
    main()
