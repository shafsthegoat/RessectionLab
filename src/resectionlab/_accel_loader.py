"""Optional loader for the opt-in ``_accel_cpp`` extension.

The extension is separate from ``_geometry_cpp`` so the qualified simulator
binary, its build manifest and its live integrity checks stay untouched.
Every caller keeps a NumPy/Python reference path for when it is absent.
"""
from __future__ import annotations

import importlib

_module = None
_error = None
_loaded = False


def module():
    """The compiled extension, or None when it was not built."""
    global _module, _error, _loaded
    if not _loaded:
        _loaded = True
        try:
            _module = importlib.import_module('resectionlab._accel_cpp')
        except ImportError as exc:  # Not built: Python reference paths apply.
            _module, _error = None, f'{type(exc).__name__}: {exc}'
    return _module


def native_identity():
    loaded = module()
    if loaded is None:
        return {'available': False, 'reason': _error}
    return {'available': True, 'version': getattr(loaded, 'VERSION', None),
            'file': getattr(loaded, '__file__', None)}
