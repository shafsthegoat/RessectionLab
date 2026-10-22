"""Explicit local build of the optional CPython idle-cooling kernel.

Run ``python -m resectionlab.build_surgical_idle_native --output-dir ...``.
Nothing compiles or installs when the simulator imports this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import time

SCHEMA = 'surgical-idle-native.v1'
FLAGS = ('-bundle', '-undefined', 'dynamic_lookup', '-O2', '-std=c11',
         '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra')
KERNEL_NAME = 'resectionlab._surgical_idle_native_kernel'
MAX_SOURCE_BYTES = 1024 * 1024
MAX_BINARY_BYTES = 4 * 1024 * 1024


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def runtime_identity():
    executable = Path(sys.executable).resolve()
    return {'implementation': sys.implementation.name, 'version': sys.version,
            'versionTuple': list(sys.version_info[:3]), 'platform': sys.platform,
            'machine': platform.machine(), 'pointerBits': struct.calcsize('P') * 8,
            'byteorder': sys.byteorder, 'soabi': sysconfig.get_config_var('SOABI'),
            'extensionSuffix': sysconfig.get_config_var('EXT_SUFFIX'),
            'executable': str(executable), 'executableSha256': file_digest(executable)}


def qualified_runtime(runtime):
    return (runtime['implementation'] == 'cpython' and runtime['versionTuple'][:2] == [3, 12]
            and runtime['platform'] == 'darwin' and runtime['machine'] == 'arm64'
            and runtime['pointerBits'] == 64 and runtime['byteorder'] == 'little')


def header_identity(include):
    include = Path(include)
    definitions = {
        'internal/pycore_dict_state.h': ('#define DICT_MAX_WATCHERS 8',),
        'internal/pycore_dict.h': ('#define DICT_VERSION_INCREMENT (1 << DICT_MAX_WATCHERS)',
                                 '#define DICT_VERSION_MASK (DICT_VERSION_INCREMENT - 1)'),
    }
    for name, expected in definitions.items():
        text = (include / name).read_text()
        if any(line not in text for line in expected):
            raise ValueError('UNSUPPORTED_IDLE_NATIVE_PRIVATE_DICT_ABI')
    headers = {str(path.relative_to(include)): file_digest(path) for path in sorted(include.rglob('*.h'))}
    if 'Python.h' not in headers or 'cpython/dictobject.h' not in headers:
        raise ValueError('INCOMPLETE_IDLE_NATIVE_HEADERS')
    return {'directory': str(include.resolve()), 'sha256': headers,
            'treeSha256': hashlib.sha256(canonical(headers).encode('ascii')).hexdigest(),
            'dictWatcherBits': 8, 'dictWatcherMask': 255}


def compile_command(metadata, source, binary):
    embedded = canonical(metadata)
    return [metadata['compiler']['path'], *FLAGS, '-I' + metadata['headers']['directory'],
            '-DIDLE_THERMAL_BUILD_IDENTITY=' + json.dumps(embedded, ensure_ascii=True),
            str(source), '-o', str(binary)]


def build(output_dir, *, max_wall_seconds=20.):
    """Build once into a new directory, preserving compiler output on failure."""
    start = time.monotonic()
    runtime = runtime_identity()
    if not qualified_runtime(runtime):
        raise ValueError('IDLE_NATIVE_REQUIRES_CPYTHON_312_MACOS_ARM64')
    if not 0 < max_wall_seconds <= 60:
        raise ValueError('INVALID_IDLE_NATIVE_BUILD_DEADLINE')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('surgical_idle_native.c').resolve()
    recipe = Path(__file__).resolve()
    if any(not 0 < path.stat().st_size <= MAX_SOURCE_BYTES for path in (source, recipe)):
        raise ValueError('IDLE_NATIVE_BUILD_SOURCE_SIZE')
    compiler = shutil.which('clang')
    if compiler is None:
        raise ValueError('LOCAL_CLANG_NOT_FOUND')
    compiler = Path(compiler).resolve()
    def remaining():
        value = max_wall_seconds - (time.monotonic() - start)
        if value <= 0:
            raise TimeoutError('IDLE_NATIVE_BUILD_DEADLINE')
        return value
    try:
        version = subprocess.run([str(compiler), '--version'], capture_output=True, text=True,
                                 check=True, timeout=remaining())
        metadata = {'schema': SCHEMA, 'sourceSha256': file_digest(source),
                    'recipeSha256': file_digest(recipe), 'compileFlags': list(FLAGS),
                    'compiler': {'path': str(compiler), 'sha256': file_digest(compiler),
                                 'version': version.stdout},
                    'headers': header_identity(Path(sysconfig.get_path('include')).resolve()),
                    'runtime': runtime, 'moduleName': KERNEL_NAME,
                    'arithmetic': 'IEEE-binary64-separated-operations-sequential-leftfold',
                    'privateAbi': 'CPython3.12-PyDictObject-low8-watcher-bits'}
        binary = output / ('_surgical_idle_native_kernel' + runtime['extensionSuffix'])
        command = compile_command(metadata, source, binary)
        (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        begin = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=remaining())
        compile_seconds = time.monotonic() - begin
        (output / 'stdout.txt').write_text(result.stdout)
        (output / 'stderr.txt').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError('IDLE_NATIVE_COMPILATION_FAILED')
        if not 0 < binary.stat().st_size <= MAX_BINARY_BYTES:
            raise ValueError('IDLE_NATIVE_BUILD_BINARY_SIZE')
        if metadata['sourceSha256'] != file_digest(source) or metadata['recipeSha256'] != file_digest(recipe):
            raise ValueError('IDLE_NATIVE_BUILD_INPUT_CHANGED')
        if runtime_identity() != runtime:
            raise ValueError('IDLE_NATIVE_BUILD_RUNTIME_CHANGED')
        remaining()
        receipt = {'schema': SCHEMA, 'status': 'compiled-not-loaded', 'metadata': metadata,
                   'embeddedMetadataSha256': hashlib.sha256(canonical(metadata).encode('ascii')).hexdigest(),
                   'binary': str(binary), 'binarySha256': file_digest(binary),
                   'binaryBytes': binary.stat().st_size, 'source': str(source), 'recipe': str(recipe),
                   'command': command, 'compileSeconds': compile_seconds,
                   'wallSeconds': time.monotonic() - start, 'maxWallSeconds': max_wall_seconds,
                   'stdoutSha256': file_digest(output / 'stdout.txt'),
                   'stderrSha256': file_digest(output / 'stderr.txt'),
                   'compileInvocations': 1, 'nativeImports': 0}
        receipt_path = output / 'build-receipt.json'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt_path
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps({'error': repr(exc),
             'wallSeconds': time.monotonic() - start, 'nativeImports': 0}, indent=2) + '\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path.cwd() / 'testcases' / 'outputs' / ('idle-native-' + time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--output-dir', type=Path, default=default)
    parser.add_argument('--max-wall-seconds', type=float, default=20.)
    args = parser.parse_args(argv)
    print(build(args.output_dir, max_wall_seconds=args.max_wall_seconds))


if __name__ == '__main__':
    main()
