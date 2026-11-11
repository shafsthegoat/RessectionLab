#!/usr/bin/env python3
"""Build and verify the complete data-free native delivery, sequentially.

Run with the project's Python 3.12+ development environment, Node and installed
desktop dependencies. Performance comparisons use benchmark_cpp_acceleration.py.
Evidence and failed logs are retained; this command never uses patient assets.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', type=Path, default=shutil.which('node'))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'testcases'):
        parser.error('Keep workflow evidence under this checkout\'s ignored testcases directory')
    if output.exists():
        parser.error('Preserve previous evidence; choose a new output directory')
    if not args.node or not args.node.is_file():
        parser.error('Provide --node /absolute/path/to/node or add Node to PATH')
    node = args.node.absolute()
    desktop = ROOT / 'desktop'
    electron = desktop / 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron'
    if not electron.is_file():
        parser.error('Install the locked desktop dependencies before verification')
    output.mkdir(parents=True)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONNOUSERSITE='1',
                       PATH=str(node.parent) + os.pathsep + os.environ.get('PATH', ''))
    environment.update({name: '1' for name in THREADS})
    clean_environment = {k: v for k, v in environment.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    report = {'schema': 'resectionlab-cpp-complete-verification-v1', 'passed': False, 'steps': []}

    def save():
        (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(name, command, *, cwd=ROOT, extra=None, clean=False):
        started = time.perf_counter()
        row = {'name': name, 'command': list(map(str, command)), 'cwd': str(cwd),
               'log': str(output / (name + '.log'))}
        report['steps'].append(row)
        save()
        print(name, flush=True)
        with Path(row['log']).open('w') as log:
            completed = subprocess.run(command, cwd=cwd,
                env={**(clean_environment if clean else environment), **(extra or {})},
                stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        row.update(exitCode=completed.returncode, elapsedSeconds=time.perf_counter() - started)
        save()
        if completed.returncode:
            raise RuntimeError(f'{name} failed; see {row["log"]}')

    python = sys.executable
    try:
        run('build-tools', [python, '-m', 'pip', 'install', 'setuptools>=77', 'pybind11>=3,<4'], clean=True)
        run('source-extension', [python, 'setup.py', 'build_ext', '--inplace'])
        focused = ['tests/test_generated_cpp_fixture.py', 'tests/test_cpp_geometry.py',
            'tests/test_cpp_state.py', 'tests/test_cpp_motion.py', 'tests/test_cpp_independent_acceptance.py',
            'tests/test_cpp_engine_copy.py', 'tests/test_generated_cpp_supplement.py',
            'tests/test_surgical_geometry_batch.py', 'tests/test_geometry.py',
            'tests/test_surgical_geometry_cache.py', 'tests/test_surgical_replay_cache.py',
            'tests/test_surgical_cancellation.py', 'tests/test_surgical_rl_contract.py',
            'tests/test_surgical_idle_native.py', 'tests/test_surgical_prefix_json_native.py',
            'tests/test_surgical_clone.py', 'tests/test_surgical_rollback_copy.py',
            'tests/test_surgical_continuum.py', 'tests/test_surgical_continuum_topology.py',
            'tests/test_surgical_continuum_solve.py', 'tests/test_surgical_continuum_vector_kernels.py',
            'tests/test_surgical_cranial_planning.py', 'tests/test_build_electron_sidecar.py',
            'tests/test_frozen_surgical_cache_paths.py',
            'tests/test_surgical_archive_codec.py', 'tests/test_surgical_verified_cache.py',
            'tests/test_surgical_frame_replay.py', 'tests/test_surgical_prefix_cache.py',
            'tests/test_imaging.py', 'tests/test_workspace_persistence.py',
            'tests/test_surgical_strategy_proof_authority.py',
            'tests/test_surgical_cusa_v3.py::test_well_formed_legacy_cusa_extraction_reaches_native_refusal_without_mutation',
            'tests/test_desktop_bridge.py::test_cancel_and_ping_are_immediate_while_worker_busy']
        run('source-tests-cpp', [python, '-m', 'pytest', '-q', *focused],
            extra={'RESSECTIONLAB_NUMERICAL_BACKEND': 'cpp'})
        run('source-tests-reference', [python, '-m', 'pytest', '-q',
            'tests/test_geometry.py', 'tests/test_surgical_geometry_cache.py',
            'tests/test_surgical_cancellation.py', 'tests/test_surgical_rl_contract.py',
            'tests/test_cpp_engine_copy.py', 'tests/test_generated_cpp_supplement.py',
            'tests/test_surgical_cusa_v3.py::test_well_formed_legacy_cusa_extraction_reaches_native_refusal_without_mutation',
            'tests/test_generated_cpp_fixture.py'], extra={'RESSECTIONLAB_NUMERICAL_BACKEND': 'reference'})
        probe = [python, 'scripts/verify_cpp_runtime.py']
        run('source-runtime', [*probe, '--runtime', 'source', '--python', python,
            '--package-root', str(ROOT / 'src'), '--output', str(output / 'source-runtime.json')])

        wheels = output / 'wheels'
        wheels.mkdir()
        run('wheel-build', [python, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(wheels), '.'], clean=True)
        wheel, = wheels.glob('resectionlab-*.whl')
        installed = Path(tempfile.mkdtemp(prefix='resectionlab-cpp-wheel-'))
        report['cleanWheelEnvironment'] = str(installed)
        venv.EnvBuilder(with_pip=True).create(installed)
        wheel_python = installed / 'bin/python'
        # Match the qualified dependency versions; install no checkout/editable paths.
        dependencies = [name + '==' + importlib.metadata.version(name) for name in
                        ('numpy', 'scipy', 'nibabel', 'scikit-image', 'torch', 'gymnasium')]
        run('wheel-install', [str(wheel_python), '-m', 'pip', 'install', str(wheel), *dependencies],
            cwd=installed, clean=True)
        site = subprocess.check_output([str(wheel_python), '-I', '-c',
            'import sysconfig;print(sysconfig.get_paths()["purelib"])'], cwd=installed,
            env=clean_environment, text=True).strip()
        run('wheel-runtime', [*probe, '--runtime', 'wheel', '--python', str(wheel_python),
            '--package-root', site, '--output', str(output / 'wheel-runtime.json')], clean=True)

        run('frozen-build', [python, 'scripts/build_electron_sidecar.py'])
        engine = desktop / 'sidecar/ressectionlab-engine/ressectionlab-engine'
        run('frozen-runtime', [*probe, '--runtime', 'frozen', '--executable', str(engine),
            '--output', str(output / 'frozen-runtime.json')], clean=True)
        run('desktop-types', [str(node), 'node_modules/typescript/bin/tsc', '--noEmit'], cwd=desktop)
        run('desktop-build', [str(node), 'node_modules/vite/bin/vite.js', 'build'], cwd=desktop)
        for name, pattern in (('main', 'electron/*.test.cjs'), ('renderer', 'tests/*.test.mjs'),
                              ('viewer', 'src/viewer/*.test.mjs')):
            run('desktop-tests-' + name, [str(node), '--experimental-strip-types', '--test',
                *map(str, sorted(desktop.glob(pattern)))], cwd=desktop)
        run('app-package', [str(node), 'electron/package-macos.cjs'], cwd=desktop)
        build = json.loads((desktop / 'release/build-report.json').read_text())
        packaged_engine = Path(build['appPath']) / 'Contents/Resources/research-engine/ressectionlab-engine'
        run('packaged-runtime', [*probe, '--runtime', 'frozen', '--executable', str(packaged_engine),
            '--output', str(output / 'packaged-runtime.json')], clean=True)

        case = output / 'fresh-generated.ressectionlab'
        run('fixture-export', [python, 'scripts/benchmark_cpp_acceleration.py',
            '--output', str(output), '--export-case', str(case)],
            extra={'RESSECTIONLAB_NUMERICAL_BACKEND': 'reference'})
        for backend in ('cpp', 'reference'):
            warm = output / (backend + '-warm')
            common = {'WORKFLOW_CRANIAL_GENERATE': '1', 'WORKFLOW_ENGINE': str(packaged_engine),
                'WORKFLOW_APP': build['appPath'],
                'WORKFLOW_EXPECT_BACKEND': backend, 'RESSECTIONLAB_NUMERICAL_BACKEND': backend}
            command = [str(node), 'scripts/verify_packaged_workflow.cjs']
            run('workflow-' + backend + '-warm', command,
                extra={**common, 'WORKFLOW_CASE': str(case), 'WORKFLOW_OUTPUT': str(warm)})
            assert json.loads((warm / 'report.json').read_text())['passed']
            run('workflow-' + backend + '-cold', command, extra={**common,
                'WORKFLOW_CASE': str(warm / 'workflow-saved.ressectionlab'),
                'WORKFLOW_OUTPUT': str(output / (backend + '-cold')), 'WORKFLOW_REOPEN': '1',
                'WORKFLOW_EXPECT_REPORT': str(warm / 'report.json')})
            assert json.loads((output / (backend + '-cold') / 'report.json').read_text())['passed']
        run('app-signature-after-workflows', ['/usr/bin/codesign', '--verify', '--deep', '--strict', build['appPath']])
        report['passed'] = True
        save()
        print(output / 'verification.json')
    except BaseException as error:
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
