#!/usr/bin/env python3
"""One released Case4 native-T1 estimate through the unchanged SynthStrip wrapper."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

MANIFEST = 'manifests/experiments/resect-case4-brain-envelope-v1.json'
MANIFEST_SHA = '2a6fadfa1eed72399580b2a6eda2aa3f5d97063c973f5b531da0d3ff4472dabe'
INPUT = 'data/mechanics/resect-case4-v1/RESECT/NIFTI/Case4/MRI/Case4-T1.nii.gz'
OUTPUT = 'outputs/mechanics/resect-case4-brain-envelope-v1'
SCOPE = 'single_case4_native_t1_main_mps_estimate'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_file(root, item):
    path = root / item['path']
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError('Pinned file changed: ' + item['path'])


def check_spec(spec):
    if (spec['input']['path'] != INPUT or spec['output_directory'] != OUTPUT
            or spec['patient_group'] != 'RESECT:Case4' or spec['role'] != 'development'):
        raise ValueError('Only the declared Case4 native T1 is permitted')
    pipeline = spec['pipeline']
    if (pipeline['model_variant'], pipeline['device'], pipeline['allow_download'],
            pipeline['border_mm'], pipeline['cpu_threads']) != ('main', 'mps', False, 1, 2):
        raise ValueError('Frozen main/MPS configuration changed')
    if spec['interpretation']['working_brain_mask'] is not None:
        raise ValueError('Estimated proposal cannot become working support')


def contract(root, release_path, release_sha):
    if sha(release_path) != release_sha:
        raise ValueError('Root release changed')
    release = json.loads(release_path.read_text())
    if release['scope'] != SCOPE or release['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Wrong release scope or declaration')
    snapshot = Path(release['snapshot_root']).resolve()
    if not snapshot.is_relative_to(root / 'build') or snapshot == root:
        raise ValueError('An immutable build snapshot is required')
    if Path(__file__).resolve() != snapshot / 'scripts/run_resect_brain_envelope.py':
        raise ValueError('Do not execute this adapter from the mutable checkout')
    if sha(snapshot / MANIFEST) != MANIFEST_SHA:
        raise ValueError('Frozen declaration changed')
    for relative, expected in release['snapshot_files_sha256'].items():
        if sha(snapshot / relative) != expected:
            raise ValueError('Source snapshot changed: ' + relative)
    for required in [MANIFEST, 'scripts/run_resect_brain_envelope.py',
                     'scripts/febio_runtime.py', 'src/resectionlab/brain_extraction.py']:
        if required not in release['snapshot_files_sha256']:
            raise ValueError('Incomplete snapshot binding')
    for source_dir in ('src', 'scripts'):
        for path in (snapshot / source_dir).rglob('*.py'):
            if str(path.relative_to(snapshot)) not in release['snapshot_files_sha256']:
                raise ValueError('Unbound source file in snapshot')
    spec = json.loads((snapshot / MANIFEST).read_text())
    check_spec(spec)
    return release, snapshot, spec


def verify_inputs(root, snapshot, spec):
    verify_file(root, spec['input'])
    for item in spec['source_receipts'] + spec['runtime']['selected_file_bindings']:
        verify_file(root, item)
    for key in ('wrapper', 'supervisor'):
        verify_file(snapshot, spec['pipeline'][key])
    for name, item in spec['pipeline']['model']['files'].items():
        verify_file(root, {**item, 'path': spec['pipeline']['model_cache'] + '/' + name})


def invoke(extraction, root, spec):
    """The sole model invocation; no mask repair, variant search or other inputs."""
    check_spec(spec)
    if (root / OUTPUT).exists():
        raise FileExistsError('Preserve existing output; never retry implicitly')
    return extraction.run_synthstrip(
        root / INPUT, root / spec['pipeline']['model_cache'], root / OUTPUT,
        model='main', device='mps', allow_download=False,
        timeout_seconds=300, maximum_rss_bytes=6 * 1024**3,
        inference_python=root / spec['runtime']['inference_interpreter'])


RUNTIME_PROBE = """
import json,sys,importlib.metadata,numpy,torch,surfa,scipy,nibabel,xxhash
packages=[numpy,torch,surfa,scipy,nibabel,xxhash]
print(json.dumps({'versions':{'python':sys.version.split()[0], **{p.__name__:importlib.metadata.version(p.__name__) for p in packages}},'origins':{p.__name__:p.__file__ for p in packages},'mps_available':torch.backends.mps.is_available(),'sys_path':sys.path}))
"""


def worker(root, release_path, release_sha, evidence):
    started = time.monotonic()
    release, snapshot, spec = contract(root, release_path, release_sha)
    state = {'status': 'failed_or_incomplete', 'source_declaration_sha256': MANIFEST_SHA,
             'root_release_sha256': release_sha, 'execution_commit': release['execution_commit'],
             'model_inferences_requested': 0, 'input_scope': 'original_native_T1_only',
             'brain_reviewed': False, 'working_brain_mask': None,
             'world_transform_accepted': False, 'motion_or_annotation_access': False}
    try:
        verify_inputs(root, snapshot, spec)
        if (root / OUTPUT).exists():
            raise FileExistsError('Preserve existing output; never retry implicitly')
        probe = subprocess.run([str(root / spec['runtime']['inference_interpreter']), '-I', '-B',
                                '-c', RUNTIME_PROBE], check=True, capture_output=True,
                               text=True, timeout=30)
        runtime = json.loads(probe.stdout)
        save(evidence / 'runtime-preflight.json', runtime)
        if runtime['versions'] != spec['runtime']['expected_versions'] or not runtime['mps_available']:
            raise ValueError('Compatible runtime or MPS availability changed')
        origins = {str((root / p['path']).resolve()): p['sha256']
                   for p in spec['runtime']['selected_file_bindings']}
        for name, path in runtime['origins'].items():
            if name != 'xxhash' and (str(Path(path).resolve()) not in origins
                                    or sha(path) != origins[str(Path(path).resolve())]):
                raise ValueError('Unexpected runtime origin: ' + name)
        sys.path[:0] = [str(snapshot / 'src')]
        sys.path.append(str(root / spec['runtime']['parent_site_packages']))
        from resectionlab import brain_extraction as extraction
        if Path(extraction.__file__).resolve() != snapshot / spec['pipeline']['wrapper']['path']:
            raise ValueError('Mutable wrapper imported')
        state['model_inferences_requested'] = 1
        child = invoke(extraction, root, spec)
        if child['executed_runner_sha256'] != spec['pipeline']['expected_executed_runner_sha256']:
            raise ValueError('Marked MPS runner changed')
        if any(child['runtime_versions'][key] != spec['runtime']['expected_versions'][key]
               for key in child['runtime_versions']):
            raise ValueError('Inference runtime changed after preflight')
        state['inference'] = child
        output = root / OUTPUT
        (output / 'implementation_snapshot.py').write_bytes(Path(extraction.__file__).read_bytes())
        state['output_files'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                 for p in sorted(output.iterdir()) if p.is_file()}
        state['project_module_origins'] = {
            name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
            if name.startswith('resectionlab') and getattr(module, '__file__', None)}
        if any(not Path(path).is_relative_to(snapshot / 'src')
               for path in state['project_module_origins'].values()):
            raise ValueError('Mutable project dependency imported')
        verify_inputs(root, snapshot, spec)
        for relative, expected in release['snapshot_files_sha256'].items():
            if sha(snapshot / relative) != expected:
                raise ValueError('Source snapshot changed during execution')
        state.update(status='completed_unreviewed_estimate_pending_independent_QC',
                     source_and_runtime_pins_unchanged=True)
    except BaseException as error:
        state['failure'] = {'type': type(error).__name__, 'message': str(error),
                            'traceback': traceback.format_exc()}
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        save(evidence / 'worker-record.json', state)
    return 0 if state['status'].startswith('completed_') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        return worker(root, args.release, args.release_sha256, args.worker)
    release, snapshot, spec = contract(root, args.release, args.release_sha256)
    helper_path = snapshot / spec['pipeline']['supervisor']['path']
    verify_file(snapshot, spec['pipeline']['supervisor'])
    helper_spec = importlib.util.spec_from_file_location('frozen_brain_supervisor', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    evidence = root / spec['receipt_directory'] / 'attempt-01'
    environment = helper.private_environment({'caps': {'thread_environment': {
        key: '2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}}})
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       PYTORCH_ENABLE_MPS_FALLBACK='0')
    command = [str(root / spec['runtime']['parent_interpreter']), '-I', '-S', '-B',
               str(Path(__file__).resolve()), '--root', str(root), '--release', str(args.release),
               '--release-sha256', args.release_sha256, '--worker', str(evidence)]
    result = helper.supervise(command, evidence, cwd=snapshot, environment=environment,
                              seconds=420, rss_bytes=6 * 1024**3)
    record = evidence / 'worker-record.json'
    accepted = (result['status'] == 'completed' and record.exists()
                and json.loads(record.read_text())['status'].startswith('completed_'))
    save(evidence / 'acceptance.json', {
        'status': 'completed_unreviewed_pending_independent_QC' if accepted else 'failed_or_incomplete',
        'supervision_sha256': sha(evidence / 'supervision.json'),
        'worker_record_sha256': sha(record) if record.exists() else None,
        'root_release_sha256': args.release_sha256, 'manifest_sha256': MANIFEST_SHA,
        'brain_reviewed': False, 'working_brain_mask': None, 'world_transform_accepted': False})
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
