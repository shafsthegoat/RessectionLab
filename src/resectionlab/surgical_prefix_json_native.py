"""Fixed optional selection for bounded private prefix JSON snapshots.

Set RESSECTIONLAB_PREFIX_JSON_BUILD to an explicit build receipt before import.
Unconfigured processes remain on the Python path. Invalid configured builds or
selected implementation drift raise ValueError; compilation is never implicit.
Full and light identity checks return the same detached identity. Full checks
also rehash the admitted standard scalar-helper images; light checks retain
those admitted hashes and reject image path/inode/size/timestamp changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
from types import BuiltinFunctionType, ModuleType

from . import build_surgical_prefix_json_native as _recipe
from . import build_surgical_idle_native as _utilities

_ENVIRONMENT = 'RESSECTIONLAB_PREFIX_JSON_BUILD'
_SELECTED_ENVIRONMENT = os.environ.get(_ENVIRONMENT)
_RECIPE_NAMES = ('qualified_runtime', 'header_identity', 'scalar_images', 'compile_command')
_UTILITY_NAMES = ('file_digest', 'canonical', 'runtime_identity', 'qualified_runtime')
_BINDINGS = tuple((module, name, getattr(module, name), getattr(module, name).__code__,
                   getattr(module, name).__defaults__, getattr(module, name).__kwdefaults__)
                  for module, names in ((_recipe, _RECIPE_NAMES), (_utilities, _UTILITY_NAMES)) for name in names)
_CONSTANTS = (_recipe.SCHEMA, _recipe.KERNEL_NAME, _recipe.FLAGS, _recipe.MAX_SOURCE_BYTES,
              _recipe.MAX_RECEIPT_BYTES, _recipe.MAX_BINARY_BYTES, _recipe.MAX_IMAGE_BYTES,
              _recipe.MAX_RAW_BYTES, _recipe.LOGICAL_SCRATCH_BYTES)


@dataclass(frozen=True, repr=False)
class _Native:
    module: object
    encode: object
    raw_identity: object
    embedded: str
    binary: str
    files: tuple
    images: tuple
    helper_json: str
    runtime_json: str
    json_module: object
    ascii_function: object


def _need(condition, reason):
    if not condition:
        raise ValueError('PREFIX_JSON_IDENTITY:' + reason)


def _stat_identity(path):
    path = Path(path)
    value = path.stat()
    return (str(path.resolve(strict=True)), value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _check_bindings():
    try:
        current = (_recipe.SCHEMA, _recipe.KERNEL_NAME, _recipe.FLAGS, _recipe.MAX_SOURCE_BYTES,
                   _recipe.MAX_RECEIPT_BYTES, _recipe.MAX_BINARY_BYTES, _recipe.MAX_IMAGE_BYTES,
                   _recipe.MAX_RAW_BYTES, _recipe.LOGICAL_SCRATCH_BYTES)
        _need(current == _CONSTANTS, 'RECIPE_CONSTANTS')
        for module, name, function, code, defaults, kwdefaults in _BINDINGS:
            _need(getattr(module, name) is function and function.__code__ is code
                  and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults,
                  'RECIPE_FUNCTION_CHANGED')
    except (AttributeError, TypeError) as error:
        raise ValueError('PREFIX_JSON_IDENTITY:RECIPE_FUNCTION_UNAVAILABLE') from error


def _check_images(images, *, full):
    for path, digest, admitted_stat in images:
        _need(_stat_identity(path) == admitted_stat, 'SCALAR_IMAGE_STAT_CHANGED')
        if full:
            _need(_utilities.file_digest(path) == digest, 'SCALAR_IMAGE_BYTES_CHANGED')
            _need(_stat_identity(path) == admitted_stat, 'SCALAR_IMAGE_DURING_HASH')


def _select():
    _check_bindings()
    if _SELECTED_ENVIRONMENT is None:
        return None, _utilities.canonical({'schema': _recipe.SCHEMA, 'mode': 'python',
                                           'selection': 'unconfigured-fixed-fallback'})
    try:
        _need(bool(_SELECTED_ENVIRONMENT), 'EMPTY_CONFIGURATION')
        receipt_path = Path(_SELECTED_ENVIRONMENT).expanduser().resolve(strict=True)
        with receipt_path.open('rb') as stream:
            raw = stream.read(_recipe.MAX_RECEIPT_BYTES + 1)
        _need(len(raw) <= _recipe.MAX_RECEIPT_BYTES, 'RECEIPT_SIZE')
        receipt = json.loads(raw)
        _need(receipt['schema'] == _recipe.SCHEMA and receipt['status'] == 'compiled-not-loaded', 'RECEIPT_SCHEMA')
        metadata = receipt['metadata']
        runtime = _utilities.runtime_identity()
        _need(_recipe.qualified_runtime(runtime), 'UNSUPPORTED_RUNTIME')
        _need(metadata['runtime'] == runtime, 'RUNTIME_BUILD')
        _need(metadata['schema'] == _recipe.SCHEMA and metadata['moduleName'] == _recipe.KERNEL_NAME, 'METADATA_SCHEMA')
        _need(metadata['compileFlags'] == list(_recipe.FLAGS), 'BUILD_FLAGS')
        paths = ((Path(__file__).with_suffix('.c').resolve(strict=True), 'sourceSha256'),
                 (Path(_recipe.__file__).resolve(strict=True), 'recipeSha256'),
                 (Path(_utilities.__file__).resolve(strict=True), 'utilitiesSha256'))
        for path, field in paths:
            _need(0 < path.stat().st_size <= _recipe.MAX_SOURCE_BYTES, 'SOURCE_SIZE')
            _need(_utilities.file_digest(path) == metadata[field], 'BUILD_SOURCE')
        include = Path(sysconfig.get_path('include')).resolve()
        _need(Path(metadata['headers']['directory']).resolve() == include, 'HEADER_DIRECTORY')
        _need(_recipe.header_identity(include) == metadata['headers'], 'HEADERS')
        _need(_utilities.file_digest(metadata['compiler']['path']) == metadata['compiler']['sha256'], 'COMPILER')
        _need(_recipe.scalar_images() == metadata['scalarImages'], 'SCALAR_IMAGE_BUILD')
        images = tuple((Path(item['path']), item['sha256'], _stat_identity(item['path']))
                       for item in metadata['scalarImages'].values())
        _check_images(images, full=True)
        embedded = _utilities.canonical(metadata)
        _need(hashlib.sha256(embedded.encode('ascii')).hexdigest() == receipt['embeddedMetadataSha256'], 'METADATA_DIGEST')
        binary = Path(receipt['binary']).resolve(strict=True)
        _need(type(receipt['binaryBytes']) is int and
              0 < binary.stat().st_size == receipt['binaryBytes'] <= _recipe.MAX_BINARY_BYTES, 'BINARY_SIZE')
        _need(binary.name == '_surgical_prefix_json_kernel' + runtime['extensionSuffix'], 'BINARY_ABI_NAME')
        _need(receipt['command'] == _recipe.compile_command(metadata, Path(receipt['source']), Path(receipt['binary'])), 'BUILD_COMMAND')
        _need(_utilities.file_digest(binary) == receipt['binarySha256'], 'BINARY_BEFORE_LOAD')
        _need(_recipe.KERNEL_NAME not in sys.modules, 'PREEXISTING_MODULE')
        specification = importlib.util.spec_from_file_location(_recipe.KERNEL_NAME, binary)
        _need(specification is not None and specification.loader is not None, 'EXTENSION_SPECIFICATION')
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        _need(_utilities.file_digest(binary) == receipt['binarySha256'], 'BINARY_DURING_LOAD')
        _check_images(images, full=True)
        _need(module.BUILD_IDENTITY == embedded, 'LOADED_BUILD_METADATA')
        encode, raw_identity = module.try_encode, module.raw_identity
        _need(all(type(function) is BuiltinFunctionType and function.__self__ is module
                  for function in (encode, raw_identity)), 'LOADED_ENTRY_POINTS')
        helpers = raw_identity()
        expected_json = Path(metadata['scalarImages']['json']['path'])
        expected_python = Path(metadata['scalarImages']['python']['path'])
        _need(Path(helpers['asciiLibrary']).resolve() == expected_json, 'LOADED_ASCII_IMAGE')
        _need(Path(helpers['floatLibrary']).resolve() == expected_python and
              Path(helpers['conversionLibrary']).resolve() == expected_python, 'LOADED_FLOAT_IMAGE')
        import _json
        _need(type(_json) is ModuleType and type(_json.encode_basestring_ascii) is BuiltinFunctionType,
              'STOCK_JSON_MODULE')
        for path, field in paths:
            _need(0 < path.stat().st_size <= _recipe.MAX_SOURCE_BYTES
                  and _utilities.file_digest(path) == metadata[field], 'SOURCE_DURING_LOAD')
        with receipt_path.open('rb') as stream:
            _need(stream.read(_recipe.MAX_RECEIPT_BYTES + 1) == raw, 'RECEIPT_DURING_LOAD')
        _need(_utilities.runtime_identity() == runtime, 'RUNTIME_DURING_LOAD')
        sys.modules[_recipe.KERNEL_NAME] = module
        files = tuple((path, metadata[field], path.stat().st_size) for path, field in paths)
        files += ((receipt_path, hashlib.sha256(raw).hexdigest(), len(raw)),
                  (binary, receipt['binarySha256'], receipt['binaryBytes']))
        selected = _Native(module, encode, raw_identity, module.BUILD_IDENTITY, str(binary),
                           files, images, _utilities.canonical(helpers), _utilities.canonical(runtime),
                           _json, _json.encode_basestring_ascii)
        identity = {'schema': _recipe.SCHEMA, 'mode': 'native', 'sourceSha256': metadata['sourceSha256'],
                    'recipeSha256': metadata['recipeSha256'], 'utilitiesSha256': metadata['utilitiesSha256'],
                    'binarySha256': receipt['binarySha256'], 'buildReceiptSha256': hashlib.sha256(raw).hexdigest(),
                    'embeddedMetadataSha256': receipt['embeddedMetadataSha256'],
                    'compilerSha256': metadata['compiler']['sha256'], 'headerTreeSha256': metadata['headers']['treeSha256'],
                    'runtime': runtime, 'scalarImages': metadata['scalarImages'], 'rawHelpers': helpers,
                    'snapshot': metadata['snapshot'], 'privateAbi': metadata['privateAbi']}
        return selected, _utilities.canonical(identity)
    except (OSError, KeyError, TypeError, AttributeError, ImportError, json.JSONDecodeError) as error:
        raise ValueError('PREFIX_JSON_IDENTITY:INVALID_CONFIGURED_BUILD') from error


_SELECTION, _IDENTITY_JSON = _select()


def _check_live(*, full=False):
    _need(os.environ.get(_ENVIRONMENT) == _SELECTED_ENVIRONMENT, 'ENVIRONMENT_CHANGED')
    _check_bindings()
    selected = _SELECTION
    if selected is None:
        return
    try:
        _need(sys.modules.get(_recipe.KERNEL_NAME) is selected.module and type(selected.module) is ModuleType,
              'MODULE_REPLACED')
        _need(selected.module.try_encode is selected.encode and selected.module.raw_identity is selected.raw_identity
              and selected.module.BUILD_IDENTITY is selected.embedded, 'ENTRY_POINT_REPLACED')
        _need(selected.module.__file__ == selected.binary and selected.module.__spec__.origin == selected.binary,
              'MODULE_ORIGIN')
        _need(sys.modules.get('_json') is selected.json_module and type(selected.json_module) is ModuleType
              and selected.json_module.encode_basestring_ascii is selected.ascii_function, 'SCALAR_MODULE_REPLACED')
        _need(_utilities.canonical(selected.raw_identity()) == selected.helper_json, 'RAW_HELPER_IDENTITY')
        _need(_utilities.canonical(_utilities.runtime_identity()) == selected.runtime_json, 'RUNTIME_CHANGED')
        for path, digest, size in selected.files:
            _need(path.stat().st_size == size and _utilities.file_digest(path) == digest, 'LOADED_FILE_CHANGED:' + path.name)
        _check_images(selected.images, full=full)
    except (OSError, AttributeError, TypeError) as error:
        raise ValueError('PREFIX_JSON_IDENTITY:SELECTED_IMPLEMENTATION_UNAVAILABLE') from error


def strict_live_identity(full=False):
    """Detached stable JSON; full checks additionally verify scalar-image bytes."""
    if type(full) is not bool:
        raise ValueError('PREFIX_JSON_IDENTITY:INVALID_CHECK_MODE')
    _check_live(full=full)
    return json.loads(_IDENTITY_JSON)


def try_encode(value, raw_limit):
    """Immutable ASCII snapshot, or None for the unchanged streaming fallback."""
    _check_live()
    if _SELECTION is None or type(raw_limit) is not int:
        return None
    return _SELECTION.encode(value, min(raw_limit, _recipe.MAX_RAW_BYTES))
