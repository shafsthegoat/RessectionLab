"""Opt-in performance paths; the reference implementations stay the default.

Nothing here runs unless it is named in the ``RESECTIONLAB_ACCEL`` environment
variable (``all`` or a comma-separated list of feature names) when this module
is first imported, or enabled in-process with :func:`override`.

Each feature declares how it relates to the reference path:

``exact``
    Bit-identical outputs, checked by ``tests/test_acceleration*.py``.
``reordered``
    Same equations and tolerances, different floating-point operation order.
    Results can differ in the last bits, so recorded digests change.
``solver``
    A different numerical method for the same certified problem. The caller's
    own residual/KKT certificates still decide acceptance.

Receipts produced with any feature enabled must record :func:`identity` and
must not be compared bit-for-bit with reference runs. This module adds no
tissue law, guard, tolerance or clinical claim.
"""
from __future__ import annotations

from contextlib import contextmanager
import os

ENVIRONMENT = 'RESECTIONLAB_ACCEL'
SCHEMA = 'resectionlab-acceleration-v1'

FEATURES = {
    'continuum': 'exact',
    'sparse-ordering': 'reordered',
    'stroke': 'exact',
    'rl-ledger': 'exact',
    'surface-audit': 'exact',
    'contact': 'solver',
    'transaction-checks': 'exact',
    'deformation-scan': 'exact',
}


def _parse(value):
    names = {part.strip() for part in (value or '').split(',') if part.strip()}
    if 'all' in names:
        names = (names - {'all'}) | set(FEATURES)
    unknown = sorted(names - set(FEATURES))
    if unknown:
        raise ValueError(f'{ENVIRONMENT} names unknown acceleration features: {", ".join(unknown)}')
    return frozenset(names)


_active = _parse(os.environ.get(ENVIRONMENT, ''))


def enabled(name):
    """True when an opt-in feature is active (unknown names are a bug)."""
    if name not in FEATURES:
        raise KeyError(name)
    return name in _active


def active():
    return tuple(sorted(_active))


@contextmanager
def override(*names, replace=True):
    """Enable exactly ``names`` (or add them) for the duration of a block."""
    global _active
    previous = _active
    requested = _parse(','.join(names))
    _active = requested if replace else previous | requested
    try:
        yield
    finally:
        _active = previous


def identity():
    """Detached description of the active opt-in paths for receipts."""
    from . import _accel_loader
    return {'schema': SCHEMA, 'enabled': list(active()),
            'relations': {name: FEATURES[name] for name in active()},
            'native': _accel_loader.native_identity()}
