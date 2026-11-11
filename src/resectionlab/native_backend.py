"""Fixed numerical backend selection; compilation happens only during installation.

Auto falls back when the extension is absent or the geometry dispatcher identifies
an unsupported configuration. A present but invalid extension is always an error.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import sysconfig
import threading
import numpy as np
from types import BuiltinFunctionType

_ENVIRONMENT = 'RESSECTIONLAB_NUMERICAL_BACKEND'
_MODULE_NAME = 'resectionlab._geometry_cpp'
MAX_PRIMITIVES = 32768
MAX_BATCH_ITEMS = 64
MAX_ENGINE_COPY_ITEMS = 262144
MAX_ENGINE_COPY_DEPTH = 64


class NativeUnavailableError(RuntimeError):
    pass


class NativeIntegrityError(RuntimeError):
    pass


class NativeUnsupportedError(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


_configured_environment = os.environ.get(_ENVIRONMENT)
_requested = 'auto' if _configured_environment is None else _configured_environment
if _requested not in ('reference', 'cpp', 'auto'):
    raise ValueError(f'{_ENVIRONMENT} must be reference, cpp, or auto')
_module = None
_entry_points = ()
_embedded_identity = None
_binary_path = None
_files = {}
_identity = {'schema': 'resectionlab-numerical-backend-v1', 'requested': _requested,
             'selected': 'reference', 'fallbackReason': None, 'nativeThreads': 1}


def _load():
    global _module, _entry_points, _embedded_identity, _binary_path
    if _requested == 'reference':
        return
    if importlib.util.find_spec(_MODULE_NAME) is None:
        if _requested == 'cpp':
            raise NativeUnavailableError('C++ numerical backend unavailable; install a compiled resectionlab wheel')
        _identity['fallbackReason'] = 'extension-unavailable'
        return
    try:
        module = importlib.import_module(_MODULE_NAME)
        binary = Path(module.__file__).resolve(strict=True)
        manifest_path = binary.with_name('_geometry_cpp.build.json')
        if manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError('oversized build manifest')
        manifest = json.loads(manifest_path.read_text())
        build = manifest['buildIdentity']
        if (manifest['schema'] != 'resectionlab-geometry-cpp-build-v1'
                or manifest['binaryName'] != binary.name
                or module.BUILD_IDENTITY != _canonical(build)
                or build['schema'] != 'resectionlab-geometry-cpp-v1'
                or build['flags'][-3:] != ['-O3', '-fno-fast-math', '-ffp-contract=off']
                or build['cppStandard'] != 17
                or build['python']['abi'] != sysconfig.get_config_var('SOABI')
                or build['platform'] != {'system': platform.system(), 'machine': platform.machine()}
                or module.MAX_PRIMITIVES != MAX_PRIMITIVES or module.MAX_BATCH_ITEMS != MAX_BATCH_ITEMS
                or module.MAX_ENGINE_COPY_ITEMS != MAX_ENGINE_COPY_ITEMS
                or module.MAX_ENGINE_COPY_DEPTH != MAX_ENGINE_COPY_DEPTH):
            raise ValueError('build/runtime identity mismatch')
        binary_digest = _digest(binary)
        if binary_digest != manifest['binarySha256']:
            raise ValueError('binary digest differs from build manifest')
        for name, expected in build['sources'].items():
            if Path(name).name != name:
                raise ValueError('invalid source name')
            path = binary.parent / name
            if _digest(path) != expected:
                raise ValueError('native source differs from build')
            _files[path] = (_stat(path), expected)
        for path in (binary, manifest_path):
            _files[path] = (_stat(path), _digest(path))
        _entry_points = tuple((name, getattr(module, name)) for name in
                              ('capsule_chunk', 'copy_builtin_json', 'segment_predicate', 'copy_engine_graph',
                               'segment_predicate_batch', 'run_trajectory', 'physical_substep', 'owned_digest',
                               'function_identities', 'occupancy_values'))
        if (sys.modules.get(_MODULE_NAME) is not module
                or Path(module.__spec__.origin).resolve() != binary
                or any(type(function) is not BuiltinFunctionType
                       or function.__name__ != name or function.__module__ != _MODULE_NAME
                       for name, function in _entry_points)):
            raise ValueError('invalid native module origin or entrypoint')
        _embedded_identity = module.BUILD_IDENTITY
        _binary_path = str(binary)
        _module = module
        _identity.update(selected='cpp', buildIdentity=build, binaryPath=str(binary),
                         binarySha256=binary_digest, manifestPath=str(manifest_path))
    except (ImportError, OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        raise NativeIntegrityError('C++ numerical backend integrity failure: ' + str(exc)) from exc


_load()
_identity_json = _canonical(_identity)
_counters = {}
_execution = threading.local()


def reset_counters():
    _counters.clear()
    _counters.update(bindingCalls=0, geometryBindingCalls=0, segmentPredicateCalls=0,
                     segmentPredicateRefinements=0, segmentBatchCalls=0, segmentBatchItems=0,
                     segmentBatchExamined=0, batchCalls=0, batchItems=0, maxBatchItems=0,
                     primitiveCandidates=0, maxPrimitiveCandidates=0, ambiguousCandidates=0,
                     referenceCalls=0, unsupportedCalls=0, jsonCopyCalls=0, jsonCopyFallbackCalls=0,
                     engineCopyCalls=0, engineCopyFallbackCalls=0, trajectoryCalls=0,
                     physicalSubstepCalls=0, ownedDigestCalls=0, cacheIdentityCalls=0,
                     occupancyCalls=0, batchSizes={})


reset_counters()


def _check_live():
    if os.environ.get(_ENVIRONMENT) != _configured_environment:
        raise NativeIntegrityError('Numerical backend environment changed after selection')
    if _module is None:
        return
    try:
        changed = (sys.modules.get(_MODULE_NAME) is not _module
            or _module.BUILD_IDENTITY is not _embedded_identity
            or _module.__file__ != _binary_path
            or _module.__spec__.origin != _binary_path
            or any(getattr(_module, name, None) is not function for name, function in _entry_points))
    except (AttributeError, TypeError, ValueError) as exc:
        raise NativeIntegrityError('Loaded numerical backend metadata unavailable') from exc
    if changed:
        raise NativeIntegrityError('Loaded numerical backend implementation changed')
    # The bounded executor verifies disk identity before entry and before exit.
    # Inner bindings still validate environment, module and entrypoints above.
    if getattr(_execution, 'depth', 0):
        return
    try:
        for path, (previous_stat, expected) in tuple(_files.items()):
            current_stat = _stat(path)
            if current_stat != previous_stat:
                if _digest(path) != expected:
                    raise NativeIntegrityError('Loaded numerical backend file changed: ' + path.name)
                _files[path] = (current_stat, expected)
    except OSError as exc:
        raise NativeIntegrityError('Loaded numerical backend file unavailable') from exc


def strict_live_identity():
    """Stable detached identity, checking selected code and disk drift."""
    _check_live()
    return json.loads(_identity_json)


def implementation_identity():
    _check_live()
    return _identity_json, _module, _entry_points


def backend_identity():
    result = strict_live_identity()
    result['counters'] = json.loads(_canonical(_counters))
    return result


def cpp_enabled():
    _check_live()
    return _module is not None


def reference_call(count=1, *, unsupported=False):
    if unsupported and _requested == 'cpp':
        raise NativeUnsupportedError('C++ geometry configuration is unsupported; select reference explicitly')
    _counters['referenceCalls'] += count
    _counters['unsupportedCalls'] += count if unsupported else 0


def capsule_chunk(queries, spans, affine, spacing, cell_radius, eps):
    _check_live()
    if _module is None:
        raise NativeUnavailableError('C++ numerical backend unavailable')
    items, primitives = len(queries), int(spans[:, 7].sum())
    _counters['bindingCalls'] += 1
    _counters['geometryBindingCalls'] += 1
    result = _module.capsule_chunk(queries, spans, affine, spacing, cell_radius, eps)
    _counters['batchCalls'] += 1
    _counters['batchItems'] += items
    _counters['maxBatchItems'] = max(_counters['maxBatchItems'], items)
    _counters['primitiveCandidates'] += primitives
    _counters['maxPrimitiveCandidates'] = max(_counters['maxPrimitiveCandidates'], primitives)
    _counters['ambiguousCandidates'] += sum(len(pair[1]) for pair in result)
    histogram = _counters['batchSizes']
    histogram[str(items)] = histogram.get(str(items), 0) + 1
    return result


def native_json_copy(value, fallback_marker, depth=0):
    _check_live()
    if _module is None:
        return fallback_marker
    _counters['bindingCalls'] += 1
    _counters['jsonCopyCalls'] += 1
    result = _module.copy_builtin_json(value, fallback_marker, depth)
    if result is fallback_marker:
        _counters['jsonCopyFallbackCalls'] += 1
    return result


def native_engine_copy(value, memo, fallback_marker):
    """Copy a bounded exact-builtin graph; unsupported inputs leave memo intact."""
    _check_live()
    if _module is None:
        return fallback_marker
    _counters['bindingCalls'] += 1
    _counters['engineCopyCalls'] += 1
    result = _module.copy_engine_graph(value, memo, fallback_marker)
    if result is fallback_marker:
        _counters['engineCopyFallbackCalls'] += 1
    return result


def try_segment_predicate(a0, a1, b0, b1, threshold, eps):
    _check_live()
    if _module is None:
        return None
    endpoints = np.array((a0, a1, b0, b1), dtype=np.float64)
    _counters['bindingCalls'] += 1
    _counters['geometryBindingCalls'] += 1
    _counters['segmentPredicateCalls'] += 1
    result = _module.segment_predicate(endpoints, threshold, eps)
    if result < 0:
        _counters['segmentPredicateRefinements'] += 1
    return result


def try_segment_batch(endpoints, thresholds, eps):
    """First HIT/AMBIGUOUS in at most 64 rows; unsupported EPS retains scalar dispatch."""
    _check_live()
    if _module is None or type(eps) is not float or not math.isfinite(eps) or eps <= 0.:
        return None
    submitted = len(endpoints)
    _counters['bindingCalls'] += 1
    _counters['geometryBindingCalls'] += 1
    _counters['segmentBatchCalls'] += 1
    _counters['segmentBatchItems'] += submitted
    result = _module.segment_predicate_batch(endpoints, thresholds, eps)
    _counters['segmentBatchExamined'] += result[2]
    _check_live()
    return result


def run_trajectory(*args):
    _check_live()
    _counters['bindingCalls'] += 1
    _counters['trajectoryCalls'] += 1
    return _module.run_trajectory(*args)


def physical_substep(*args):
    _check_live()
    _counters['bindingCalls'] += 1
    _counters['physicalSubstepCalls'] += 1
    _execution.depth = getattr(_execution, 'depth', 0) + 1
    try:
        return _module.physical_substep(*args)
    finally:
        _execution.depth -= 1
        _check_live()


def owned_digest(value, dumps, sha256, fallback):
    if _module is None:
        return fallback
    _check_live()
    _counters['bindingCalls'] += 1
    _counters['ownedDigestCalls'] += 1
    return _module.owned_digest(value, dumps, sha256, fallback)


def function_identities(functions, atoms):
    # Called immediately after implementation_identity validated this binding.
    _counters['bindingCalls'] += 1
    _counters['cacheIdentityCalls'] += 1
    return _module.function_identities(functions, atoms)


def occupancy_values(labels, ids, cleared, moved, fallback):
    if _module is None:
        return fallback
    _check_live()
    _counters['bindingCalls'] += 1
    _counters['occupancyCalls'] += 1
    return _module.occupancy_values(labels, ids, cleared, moved, fallback)
