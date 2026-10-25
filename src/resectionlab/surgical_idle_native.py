"""Optional exact idle-cooling kernel with a fixed, checked process selection.

Unconfigured processes keep the Python implementation even if a native build
appears later. Set RESSECTIONLAB_IDLE_THERMAL_BUILD to an explicit build receipt
before importing the simulator to select native execution. Invalid configured
builds or changes to a selected implementation raise ValueError; there is no
silent implementation switch. Compilation is always an explicit separate step.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
from types import BuiltinFunctionType, ModuleType

from . import build_surgical_idle_native as _recipe

_ENVIRONMENT = 'RESSECTIONLAB_IDLE_THERMAL_BUILD'
_EXACT_INT_LIST_CAP = 1048576
_SELECTED_ENVIRONMENT = os.environ.get(_ENVIRONMENT)
_RECIPE_FUNCTIONS = tuple((function, function.__code__, function.__defaults__)
    for function in (_recipe.file_digest, _recipe.canonical, _recipe.runtime_identity,
                     _recipe.qualified_runtime, _recipe.header_identity, _recipe.compile_command))
_RECIPE_NAMES = ('file_digest', 'canonical', 'runtime_identity', 'qualified_runtime',
                 'header_identity', 'compile_command')
_RECIPE_CONSTANTS = (_recipe.SCHEMA, _recipe.FLAGS, _recipe.KERNEL_NAME,
                     _recipe.MAX_SOURCE_BYTES, _recipe.MAX_BINARY_BYTES)


def _need(condition, reason):
    if not condition:
        raise ValueError('IDLE_NATIVE_IDENTITY:' + reason)


def _digest(path):
    return _recipe.file_digest(path)


def _load_selection():
    if _SELECTED_ENVIRONMENT is None:
        return None, _recipe.canonical({'schema': _recipe.SCHEMA, 'mode': 'python',
                                       'selection': 'unconfigured-fixed-fallback'})
    try:
        _need(bool(_SELECTED_ENVIRONMENT), 'EMPTY_CONFIGURATION')
        receipt_path = Path(_SELECTED_ENVIRONMENT).expanduser().resolve(strict=True)
        with receipt_path.open('rb') as stream:
            receipt_bytes = stream.read(1024 * 1024 + 1)
        _need(len(receipt_bytes) <= 1024 * 1024, 'OVERSIZED_RECEIPT')
        receipt = json.loads(receipt_bytes)
        _need(receipt['schema'] == _recipe.SCHEMA and receipt['status'] == 'compiled-not-loaded', 'RECEIPT_SCHEMA')
        metadata = receipt['metadata']
        runtime = _recipe.runtime_identity()
        _need(_recipe.qualified_runtime(runtime), 'UNSUPPORTED_RUNTIME')
        _need(metadata['runtime'] == runtime, 'RUNTIME_BUILD')
        _need(metadata['schema'] == _recipe.SCHEMA and metadata['moduleName'] == _recipe.KERNEL_NAME, 'METADATA_SCHEMA')
        _need(metadata['compileFlags'] == list(_recipe.FLAGS), 'BUILD_FLAGS')
        source = Path(__file__).with_suffix('.c').resolve(strict=True)
        recipe = Path(_recipe.__file__).resolve(strict=True)
        _need(all(0 < path.stat().st_size <= _recipe.MAX_SOURCE_BYTES for path in (source, recipe)), 'SOURCE_SIZE')
        _need(_digest(source) == metadata['sourceSha256'], 'C_SOURCE')
        _need(_digest(recipe) == metadata['recipeSha256'], 'BUILD_RECIPE')
        include = Path(metadata['headers']['directory'])
        _need(include.resolve() == Path(sysconfig.get_path('include')).resolve(), 'HEADER_DIRECTORY')
        _need(_recipe.header_identity(include) == metadata['headers'], 'PRIVATE_ABI_HEADERS')
        _need(_digest(metadata['compiler']['path']) == metadata['compiler']['sha256'], 'COMPILER')
        embedded = _recipe.canonical(metadata)
        _need(hashlib.sha256(embedded.encode('ascii')).hexdigest() == receipt['embeddedMetadataSha256'], 'EMBEDDED_METADATA')
        binary = Path(receipt['binary']).resolve(strict=True)
        _need(type(receipt['binaryBytes']) is int and
              0 < binary.stat().st_size == receipt['binaryBytes'] <= _recipe.MAX_BINARY_BYTES, 'BINARY_SIZE')
        _need(binary.name == '_surgical_idle_native_kernel' + runtime['extensionSuffix'], 'BINARY_ABI_NAME')
        _need(receipt['command'] == _recipe.compile_command(metadata, Path(receipt['source']), Path(receipt['binary'])), 'BUILD_COMMAND')
        binary_digest = _digest(binary)
        _need(binary_digest == receipt['binarySha256'], 'BINARY_BEFORE_LOAD')
        _need(_recipe.KERNEL_NAME not in sys.modules, 'PREEXISTING_MODULE')
        specification = importlib.util.spec_from_file_location(_recipe.KERNEL_NAME, binary)
        _need(specification is not None and specification.loader is not None, 'EXTENSION_SPECIFICATION')
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        _need(_digest(binary) == binary_digest, 'BINARY_DURING_LOAD')
        _need(module.BUILD_IDENTITY == embedded, 'LOADED_BUILD_METADATA')
        callable_ = module.try_idle
        _need(type(callable_) is BuiltinFunctionType and callable_.__self__ is module, 'LOADED_ENTRY_POINT')
        predicate = module.try_exact_int_list
        cap = module.MAX_EXACT_INT_LIST_ITEMS
        _need(type(predicate) is BuiltinFunctionType and predicate.__self__ is module and
              predicate.__name__ == 'try_exact_int_list', 'LOADED_PREDICATE_ENTRY_POINT')
        _need(type(cap) is int and cap == _EXACT_INT_LIST_CAP, 'LOADED_PREDICATE_CAP')
        sys.modules[_recipe.KERNEL_NAME] = module
        files = {source: metadata['sourceSha256'], recipe: metadata['recipeSha256'],
                 receipt_path: hashlib.sha256(receipt_bytes).hexdigest(), binary: binary_digest}
        sizes = {path: path.stat().st_size for path in files}
        # Entire header inventory is verified once above and bound by this
        # embedded digest. Hot reads copy only the compact immutable identity.
        identity = {'schema': _recipe.SCHEMA, 'mode': 'native', 'sourceSha256': metadata['sourceSha256'],
                    'recipeSha256': metadata['recipeSha256'], 'binarySha256': binary_digest,
                    'buildReceiptSha256': files[receipt_path],
                    'embeddedMetadataSha256': receipt['embeddedMetadataSha256'],
                    'compilerSha256': metadata['compiler']['sha256'],
                    'headerTreeSha256': metadata['headers']['treeSha256'],
                    'runtime': runtime, 'privateAbi': metadata['privateAbi'],
                    'exactIntListPredicate': 'try_exact_int_list:True-or-None', 'exactIntListMaxItems': cap}
        selected = (module, callable_, module.BUILD_IDENTITY, str(binary), files,
                    _recipe.canonical(runtime), sizes, predicate, cap)
        return selected, _recipe.canonical(identity)
    except (OSError, KeyError, TypeError, AttributeError, ImportError, json.JSONDecodeError) as exc:
        raise ValueError('IDLE_NATIVE_IDENTITY:INVALID_CONFIGURED_BUILD') from exc


_SELECTION, _IDENTITY_JSON = _load_selection()


def _check_live():
    _need(os.environ.get(_ENVIRONMENT) == _SELECTED_ENVIRONMENT, 'ENVIRONMENT_CHANGED')
    try:
        _need((_recipe.SCHEMA, _recipe.FLAGS, _recipe.KERNEL_NAME,
               _recipe.MAX_SOURCE_BYTES, _recipe.MAX_BINARY_BYTES) == _RECIPE_CONSTANTS, 'RECIPE_CONSTANTS')
        for name, (function, code, defaults) in zip(_RECIPE_NAMES, _RECIPE_FUNCTIONS):
            _need(getattr(_recipe, name) is function and function.__code__ is code
                  and function.__defaults__ is defaults, 'RECIPE_FUNCTION_REPLACED')
    except (AttributeError, TypeError) as exc:
        raise ValueError('IDLE_NATIVE_IDENTITY:RECIPE_FUNCTION_UNAVAILABLE') from exc
    if _SELECTION is None:
        return
    module, callable_, embedded, binary, files, runtime, sizes, predicate, cap = _SELECTION
    _need(sys.modules.get(_recipe.KERNEL_NAME) is module, 'MODULE_REPLACED')
    try:
        _need(type(module) is ModuleType, 'MODULE_TYPE')
        _need(module.try_idle is callable_ and module.BUILD_IDENTITY is embedded, 'ENTRY_POINT_REPLACED')
        _need(module.try_exact_int_list is predicate, 'PREDICATE_ENTRY_POINT_REPLACED')
        _need(module.MAX_EXACT_INT_LIST_ITEMS is cap, 'PREDICATE_CAP_REPLACED')
        _need(module.__file__ == binary and module.__spec__.origin == binary, 'MODULE_ORIGIN')
        _need(_recipe.canonical(_recipe.runtime_identity()) == runtime, 'RUNTIME_CHANGED')
        for path, expected in files.items():
            _need(path.stat().st_size == sizes[path], 'LOADED_FILE_SIZE_CHANGED:' + path.name)
            _need(_digest(path) == expected, 'LOADED_FILE_CHANGED:' + path.name)
    except (OSError, AttributeError, TypeError) as exc:
        raise ValueError('IDLE_NATIVE_IDENTITY:LOADED_FILE_UNAVAILABLE') from exc


def strict_live_identity():
    """Return detached compact JSON, or reject drift of the fixed selection."""
    _check_live()
    return json.loads(_IDENTITY_JSON)


def try_idle(*args):
    """Return False for Python/admission fallback; selected native drift raises."""
    _check_live()
    if _SELECTION is None:
        return False
    return _SELECTION[1](*args)


def try_exact_int_list(value):
    """True for admitted exact list/ints; None retains the caller's predicate.

    Explicit calls always check the fixed selection, including small inputs.
    Engine callers skip small-ledger native access before reaching this API.
    """
    _check_live()
    if _SELECTION is None:
        return None
    return _SELECTION[7](value)
