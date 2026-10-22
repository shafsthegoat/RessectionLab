"""Explicit optional build of the bounded prefix JSON snapshot kernel.

``python -m resectionlab.build_surgical_prefix_json_native --output-dir ...``
compiles once into a fresh local directory. Importing this module never builds,
installs, imports the kernel, constructs an engine, or opens a patient case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time
from types import BuiltinFunctionType, ModuleType

from . import build_surgical_idle_native as _utilities

SCHEMA = 'surgical-prefix-json-native.v1'
KERNEL_NAME = 'resectionlab._surgical_prefix_json_kernel'
FLAGS = _utilities.FLAGS
MAX_SOURCE_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_BINARY_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_RAW_BYTES = 16 * 1024 * 1024
LOGICAL_SCRATCH_BYTES = 48 * 1024 * 1024


def qualified_runtime(runtime):
    return _utilities.qualified_runtime(runtime) and runtime['versionTuple'] == [3, 12, 12]


def header_identity(include):
    include = Path(include).resolve()
    headers = {str(path.relative_to(include)): _utilities.file_digest(path)
               for path in sorted(include.rglob('*.h'))}
    if not {'Python.h', 'cpython/pystate.h'} <= headers.keys():
        raise ValueError('PREFIX_JSON_INCOMPLETE_HEADERS')
    return {'directory': str(include), 'sha256': headers,
            'treeSha256': hashlib.sha256(_utilities.canonical(headers).encode('ascii')).hexdigest()}


def scalar_images():
    """Bind actual standard C scalar helpers; their offsets are checked on load."""
    import _json
    if (type(_json) is not ModuleType or type(_json.encode_basestring_ascii) is not BuiltinFunctionType
            or _json.encode_basestring_ascii.__self__ is not _json
            or _json.encode_basestring_ascii.__name__ != 'encode_basestring_ascii'):
        raise ValueError('PREFIX_JSON_STOCK_ASCII_REQUIRED')
    framework_name = sysconfig.get_config_var('PYTHONFRAMEWORK')
    if type(framework_name) is not str or not framework_name:
        raise ValueError('PREFIX_JSON_FRAMEWORK_REQUIRED')
    paths = {'json': Path(_json.__file__).resolve(strict=True),
             'python': (Path(sys.base_prefix) / framework_name).resolve(strict=True)}
    images = {}
    for name, path in paths.items():
        size = path.stat().st_size
        if not 0 < size <= MAX_IMAGE_BYTES:
            raise ValueError('PREFIX_JSON_SCALAR_IMAGE_SIZE')
        images[name] = {'path': str(path), 'bytes': size, 'sha256': _utilities.file_digest(path)}
    return images


def compile_command(metadata, source, binary):
    embedded = _utilities.canonical(metadata)
    return [metadata['compiler']['path'], *FLAGS, '-I' + metadata['headers']['directory'],
            '-DPREFIX_JSON_BUILD_IDENTITY=' + json.dumps(embedded, ensure_ascii=True),
            str(source), '-o', str(binary)]


def build(output_dir, *, max_wall_seconds=20.):
    start = time.monotonic()
    runtime = _utilities.runtime_identity()
    if not qualified_runtime(runtime):
        raise ValueError('PREFIX_JSON_REQUIRES_CPYTHON_31212_MACOS_ARM64')
    if type(max_wall_seconds) not in (int, float) or not 0 < max_wall_seconds <= 60:
        raise ValueError('PREFIX_JSON_BUILD_DEADLINE')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('surgical_prefix_json_native.c').resolve()
    recipe = Path(__file__).resolve()
    utilities = Path(_utilities.__file__).resolve()
    def remaining():
        seconds = max_wall_seconds - (time.monotonic() - start)
        if seconds <= 0:
            raise TimeoutError('PREFIX_JSON_BUILD_DEADLINE')
        return seconds
    try:
        if any(not 0 < path.stat().st_size <= MAX_SOURCE_BYTES for path in (source, recipe, utilities)):
            raise ValueError('PREFIX_JSON_BUILD_SOURCE_SIZE')
        compiler_name = shutil.which('clang')
        if compiler_name is None:
            raise ValueError('PREFIX_JSON_LOCAL_CLANG_NOT_FOUND')
        compiler = Path(compiler_name).resolve()
        version = subprocess.run([str(compiler), '--version'], capture_output=True, text=True,
                                 check=True, timeout=remaining())
        metadata = {'schema': SCHEMA, 'moduleName': KERNEL_NAME,
                    'sourceSha256': _utilities.file_digest(source),
                    'recipeSha256': _utilities.file_digest(recipe),
                    'utilitiesSha256': _utilities.file_digest(utilities),
                    'compiler': {'path': str(compiler), 'sha256': _utilities.file_digest(compiler),
                                 'version': version.stdout},
                    'compileFlags': list(FLAGS), 'runtime': runtime,
                    'headers': header_identity(sysconfig.get_path('include')),
                    'scalarImages': scalar_images(),
                    'snapshot': {'asciiCap': MAX_RAW_BYTES, 'depthCap': 64, 'nodeCap': 1000000,
                                 'logicalScratchBytes': LOGICAL_SCRATCH_BYTES,
                                 'integer': 'signed-int64-unsigned-decimal',
                                 'strings': 'stock-raw-METH_O-ASCII', 'floats': 'stock-raw-float-tp_repr'},
                    'privateAbi': 'CPython3.12.12-PyThreadState.py_recursion_remaining>=256'}
        binary = output / ('_surgical_prefix_json_kernel' + runtime['extensionSuffix'])
        command = compile_command(metadata, source, binary)
        (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        begin = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=remaining())
        except subprocess.TimeoutExpired as error:
            (output / 'stdout.txt').write_bytes(error.stdout or b'')
            (output / 'stderr.txt').write_bytes(error.stderr or b'')
            raise
        compile_seconds = time.monotonic() - begin
        (output / 'stdout.txt').write_text(result.stdout)
        (output / 'stderr.txt').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError('PREFIX_JSON_COMPILATION_FAILED')
        if not 0 < binary.stat().st_size <= MAX_BINARY_BYTES:
            raise ValueError('PREFIX_JSON_BINARY_SIZE')
        if any(_utilities.file_digest(path) != metadata[key] for path, key in
               ((source, 'sourceSha256'), (recipe, 'recipeSha256'), (utilities, 'utilitiesSha256'))):
            raise ValueError('PREFIX_JSON_BUILD_SOURCE_CHANGED')
        if _utilities.runtime_identity() != runtime or scalar_images() != metadata['scalarImages']:
            raise ValueError('PREFIX_JSON_BUILD_RUNTIME_CHANGED')
        remaining()
        receipt = {'schema': SCHEMA, 'status': 'compiled-not-loaded', 'metadata': metadata,
                   'embeddedMetadataSha256': hashlib.sha256(_utilities.canonical(metadata).encode('ascii')).hexdigest(),
                   'binary': str(binary), 'binaryBytes': binary.stat().st_size,
                   'binarySha256': _utilities.file_digest(binary), 'source': str(source),
                   'recipe': str(recipe), 'command': command,
                   'compileSeconds': compile_seconds, 'wallSeconds': time.monotonic() - start,
                   'maxWallSeconds': max_wall_seconds, 'compileInvocations': 1, 'nativeImports': 0,
                   'stdoutSha256': _utilities.file_digest(output / 'stdout.txt'),
                   'stderrSha256': _utilities.file_digest(output / 'stderr.txt')}
        data = json.dumps(receipt, indent=2) + '\n'
        if len(data.encode('ascii')) > MAX_RECEIPT_BYTES:
            raise ValueError('PREFIX_JSON_BUILD_RECEIPT_SIZE')
        receipt_path = output / 'build-receipt.json'
        receipt_path.write_text(data)
        return receipt_path
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps({'error': repr(error),
            'wallSeconds': time.monotonic() - start, 'nativeImports': 0}, indent=2) + '\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path.cwd() / 'testcases' / 'outputs' / ('prefix-json-native-' + time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--output-dir', type=Path, default=default)
    parser.add_argument('--max-wall-seconds', type=float, default=20.)
    args = parser.parse_args(argv)
    print(build(args.output_dir, max_wall_seconds=args.max_wall_seconds))


if __name__ == '__main__':
    main()
