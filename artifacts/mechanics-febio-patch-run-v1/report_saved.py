"""Summarize this completed patch attempt without invoking FEBio or its checker."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name: str):
    return json.loads((ROOT / name).read_text())

def write_new(name: str, data: str) -> None:
    with (ROOT / name).open('x') as f:
        f.write(data)

def main() -> None:
    results = read('results.json')
    execution = read('execution.json')
    baseline = read('execution-baseline.json')
    archive = read('archive-binding.json')
    assert digest(Path(archive['archive'])) == archive['archive_sha256']
    assert digest(ROOT / 'run_once.py') == baseline['wrapper_sha256']
    assert all(digest(Path(p)) == h for p, h in baseline['input_hashes'].items())
    assert len(baseline['input_hashes']) == 36
    assert all(results['inputs_unchanged_after'].values())
    assert execution['status'] == results['status'] == 'completed'
    assert execution['all_cases_passed'] and results['scaling_check']['passed']
    assert execution['solver_invocations'] == results['solver_invocations'] == 5
    assert execution['supervision']['wall_cap_seconds'] == baseline['applied_patch_cap_seconds'] == 56.49
    rows = []
    for case in results['cases']:
        name = case['case']
        for path, sha in case['output_hashes'].items():
            assert digest(ROOT / name / path) == sha, (name, path)
        checked = read(f'{name}/checked.json')
        assert case['status'] == 'passed' and checked['passed']
        states = checked['states']
        assert [s['time'] for s in states] == [0, .25, .5, .75, 1]
        assert all(all(s['checks'].values()) for s in states)
        assert len(case['skyline_console_lines']) == 2
        final = states[-1]
        residuals = checked['solver']['last_squared_force_residual_norms_N2']
        rows.append({
            'case': name, 'passed': True, 'recorded_states': 5,
            'solver_seconds': case['solver_seconds'],
            'minimum_actual_sampled_J_all_states': min(s['minimum_actual_sampled_J'] for s in states),
            'minimum_actual_sampled_J_final': final['minimum_actual_sampled_J'],
            'final_logged_energy_J': final['total_logged_energy_J'],
            'final_expected_energy_J': final['expected_energy_J'],
            'maximum_absolute_energy_error_J': max(abs(s['total_logged_energy_J'] - s['expected_energy_J']) for s in states),
            'maximum_force_imbalance_l2_N': max(math.hypot(*s['net_force_N']) for s in states),
            'maximum_current_moment_imbalance_l2_Nm': max(math.hypot(*s['net_current_moment_Nm']) for s in states),
            'maximum_final_iteration_squared_force_residual_N2': max(v[1] for v in residuals.values()),
            'final_free_center_displacement_m': final['center_displacement_m'],
            'skyline_console_lines': case['skyline_console_lines'],
        })
    report = {
        'status': 'passed_all_five_fixed_patch_controls',
        'source_commit': archive['commit'],
        'runtime_identity_sha256': baseline['runtime_identity_sha256'],
        'method': 'Read saved checker receipts and verify original input/output hashes; no solver or checker replay.',
        'source_scope': archive['scope'],
        'input_files_rehashed': 36,
        'output_files_rehashed': sum(len(c['output_hashes']) for c in results['cases']),
        'solver_invocations': 5,
        'rows': rows,
        'stiffness_scaling': results['scaling_check'],
        'original_allowance_seconds': baseline['aggregate_original_allowance_seconds'],
        'informational_version_seconds_counted': baseline['version_probe_seconds_counted'],
        'applied_patch_cap_seconds': baseline['applied_patch_cap_seconds'],
        'supervisor_seconds': execution['supervision']['elapsed_seconds'],
        'outer_execution_seconds': execution['elapsed_seconds'],
        'worker_seconds': results['worker_seconds'],
        'sum_solver_seconds': sum(c['solver_seconds'] for c in results['cases']),
        'sampled_peak_process_group_rss_bytes': execution['supervision']['sampled_peak_process_group_rss_bytes'],
        'rss_sampling_note': execution['supervision']['sampling_note'],
        'numerical_threads': 1,
        'retries': 0,
        'original_receipt_note': 'results.json retains original 60 s nominal cap; execution-baseline and supervisor record actual tighter 56.49 s enforcement.',
        'negative_energy_roundoff_note': 'Tiny negative zero/translation energy values are preserved, not clamped; maximum magnitude is below 3e-19 J.',
        'limits': [
            'Eight-element homogeneous affine patch only, with three free center-node degrees of freedom.',
            'Jacobian positivity checked at Gauss points, element corners and centers; no claim for arbitrary warped meshes.',
            'No locking, mesh convergence, specimen material fit, tissue damage, cutting, contact or patient mechanics validation.',
            '1000 Pa is an arbitrary numerical reference modulus, not a fitted human tissue property.',
            'One fixed attempt; timings are observations, not a performance comparison.',
            'Actual default xplt files are retained unchanged alongside primitive text output.',
        ],
    }
    write_new('summary.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    table = '\n'.join(f"| {r['case']} | {r['minimum_actual_sampled_J_final']:.12g} | {r['final_logged_energy_J']:.12g} | {r['maximum_force_imbalance_l2_N']:.3g} |" for r in rows)
    text = f'''# FEBio numerical patch verification — one actual attempt

All five frozen cases passed, including signed reactions, free-center motion, stress, energy, reconstructed Jacobians, force/moment balance and logged convergence residuals. Each case contains the initial state and four prescribed load states. Doubling stiffness preserved motion/J and doubled reactions, stress and energy within the predeclared tolerances.

| Case | Final minimum sampled J | Final logged energy (J) | Maximum force imbalance (N) |
| --- | ---: | ---: | ---: |
{table}

The zero and translation cases retain tiny negative energy roundoff (magnitude below 3e-19 J); values were not clamped. The finite stretch has expected final J=1.045. The simple-shear cases (shear strain 0.1) have expected J=1 and final energy 5e-6 J and 1e-5 J. All five solver consoles explicitly select Skyline.

The single attempt used archived commit `{archive['commit']}` and the accepted local FEBio 4.13 runtime. All 36 bound inputs and 35 per-case outputs were rehashed successfully. The archive includes only the committed checker, process supervisor, decks and manifests; Python imported the checker from that archive. Original source/decks, raw logs, xplt files, checker receipts and runtime identity remain unchanged.

The enforced patch cap was **56.49 s**, after counting the 3.502 s informational version probe inside the original 60 s allowance. The supervisor observed **{report['supervisor_seconds']:.6f} s**, the worker {report['worker_seconds']:.6f} s and the five solver calls together {report['sum_solver_seconds']:.6f} s. Sampled peak process-group RSS was {report['sampled_peak_process_group_rss_bytes']:,} bytes; brief between-sample peaks may be missed. One numerical thread, a 3 GiB cap and zero retries were used. The original result's nominal 60 s field is retained; the baseline and supervision receipt document the tighter enforced cap.

This verifies an eight-element homogeneous patch with a free center node and an arbitrary 1000 Pa reference modulus. It does not validate a human tissue property, locking or mesh convergence, nonuniform specimen mechanics, damage, contact, cutting or patient mechanics. For these affine controls, logged element energy and reconstructed energy agree; that does not establish an energy rule for nonuniform three-field specimens.

[Saved measurements](summary.json), [original result](results.json), [execution receipt](execution.json), [prospective baseline](execution-baseline.json), [archive binding](archive-binding.json). The independent saved-output audit is recorded separately; no solver or checker was rerun to write this report.
'''
    write_new('RESULT.md', text)
    files = {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'artifact-index.json'}
    write_new('artifact-index.json', json.dumps({'schema': 'mechanics-febio-patch-run-artifacts-v1', 'files': files,
                                              'file_count': len(files), 'total_bytes': sum(v['bytes'] for v in files.values())}, indent=2) + '\n')
    print(json.dumps({'files_indexed': len(files), 'bytes_indexed': sum(v['bytes'] for v in files.values()),
                      'index_sha256': digest(ROOT / 'artifact-index.json'), 'result_sha256': digest(ROOT / 'RESULT.md')}, indent=2))

if __name__ == '__main__':
    main()
