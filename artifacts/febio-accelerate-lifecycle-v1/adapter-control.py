"""Separately released repaired-only SDK initialization and mocked lifecycle controls."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECLARATION = HERE/'control-declaration.json'
BUILD = ROOT/'build/febio-accelerate-lifecycle-v1'
OUTPUT = HERE/'control-01'
HELPER = ROOT/'scripts/febio_runtime.py'
HELPER_SHA = '679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e'
EXPECTED_LOGS = {
    'initialization': ''.join(f'PASS matrix-init profile={i} rows={2+i//2} symmetric={int(i%2 == 0)}\n' for i in range(4)),
    'lifecycle': ''.join(f'PASS lifecycle scenario={i}\n' for i in range(14)),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, record):
    with path.open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def declaration(expected):
    if sha(DECLARATION) != expected:
        raise ValueError('Exact root-released adapter control declaration required')
    value = json.loads(DECLARATION.read_bytes())
    for item in value['pins'].values():
        path = ROOT/item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Adapter input changed: '+str(path))
    return value


def worker(expected):
    spec = declaration(expected)
    BUILD.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    records = []
    result = {'status': 'failed_no_retry', 'declaration_sha256': expected,
              'patched_source_sha256': spec['pins']['combined_source']['sha256'],
              'sdk_factorization_executed': False, 'lifecycle_api_is_mock': True,
              'initialization_fragment_sha256': spec['pins']['initialization_fragment']['sha256'],
              'original_code_executed': False, 'factorization_or_model_executed': False}
    compiler = ROOT/spec['pins']['compiler']['path']
    try:
        for name in ('initialization', 'lifecycle'):
            fixture = ROOT/spec['pins'][name+'_fixture']['path']
            binary = BUILD/(name+'-check')
            command = [str(compiler), '-isysroot', spec['sdk'], '-std=c++17', '-O1', '-g',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(fixture), '-o', str(binary)]
            for phase, argv, seconds in (('compile', command, 40), ('cases', [str(binary)], 5)):
                started = time.monotonic()
                stem = name+'-'+phase
                with (OUTPUT/(stem+'.stdout')).open('x') as out, (OUTPUT/(stem+'.stderr')).open('x') as err:
                    completed = subprocess.run(argv, cwd=BUILD, stdout=out, stderr=err,
                                               timeout=seconds, check=False)
                records.append({'name': stem, 'argv': argv, 'exit_code': completed.returncode,
                    'elapsed_seconds': time.monotonic()-started,
                    'stdout_sha256': sha(OUTPUT/(stem+'.stdout')),
                    'stderr_sha256': sha(OUTPUT/(stem+'.stderr'))})
                if completed.returncode:
                    raise RuntimeError(stem+' failed; no retry')
            if (OUTPUT/(name+'-cases.stdout')).read_text() != EXPECTED_LOGS[name]:
                raise ValueError('Unexpected '+name+' control output')
        declaration(expected)
        result.update(status='adapter_initialization_lifecycle_controls_passed', all_pins_unchanged=True,
                      binaries_sha256={name:sha(BUILD/(name+'-check')) for name in EXPECTED_LOGS})
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    result['rows'] = records
    save(OUTPUT/'result.json', result)
    return 0 if result['status'] == 'adapter_initialization_lifecycle_controls_passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args.declaration_sha256)
    spec = declaration(args.declaration_sha256)
    save(HERE/'control-attempt.json', {'declaration_sha256': args.declaration_sha256})
    if sha(HELPER) != HELPER_SHA:
        raise ValueError('Reviewed supervisor changed')
    module_spec = importlib.util.spec_from_file_location('checked_supervisor', HELPER)
    helper = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helper)
    environment = helper.private_environment({'caps': {'thread_environment': spec['thread_environment']}})
    for key in list(environment):
        if key.startswith('PYTHON') or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', '__PYVENV_LAUNCHER__'):
            environment.pop(key)
    environment.update(ASAN_OPTIONS='detect_leaks=0:symbolize=0',
                       UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=0')
    receipt = helper.supervise([sys.executable, '-I', '-S', '-B', str(Path(__file__)),
        '--declaration-sha256', args.declaration_sha256, '--worker'], HERE/'supervision-01',
        cwd=ROOT, environment=environment, seconds=100, rss_bytes=1024**3)
    accepted = False
    try:
        declaration(args.declaration_sha256)
        result = json.loads((OUTPUT/'result.json').read_bytes())
        accepted = (receipt['status'] == 'completed' and receipt['exit_code'] == 0
            and receipt['kill_reason'] is None and receipt['cleanup_error'] is None
            and receipt['wall_cap_seconds'] == 100 and receipt['rss_cap_bytes'] == 1024**3
            and result['status'] == 'adapter_initialization_lifecycle_controls_passed'
            and result['declaration_sha256'] == args.declaration_sha256
            and result['all_pins_unchanged'] is True)
    except (ValueError, OSError, KeyError):
        pass
    save(HERE/'control-acceptance.json', {'status': 'completed' if accepted else 'failed_no_retry',
        'declaration_sha256': args.declaration_sha256,
        'result_sha256': sha(OUTPUT/'result.json') if (OUTPUT/'result.json').exists() else None,
        'supervision_sha256': sha(HERE/'supervision-01/supervision.json'),
        'original_code_executed': False, 'factorization_or_model_executed': False})
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
