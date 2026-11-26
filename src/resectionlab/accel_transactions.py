"""Opt-in ``transaction-checks`` (exact): one disk-identity scan per stroke preview.

The native backend re-reads its binary and source file identities at every
binding call; in stroke previews that scan is about a fifth of the time.
Under ``transaction-checks`` a whole ``preview_stroke`` runs inside
``native_backend._substep_file_guard`` (the scope the operative substep
executor already uses): all files are checked on entry and on exit, and every
inner binding call still verifies the loaded module, its build identity and its
entry points. A file that changes during a preview raises on exit, so its
result is never returned. No arithmetic changes.
"""
from __future__ import annotations

import functools

from . import acceleration, native_backend


def guarded(method):
    """Wrap an engine method in one file-identity transaction when enabled."""
    @functools.wraps(method)
    def call(*args, **kwargs):
        if not acceleration.enabled('transaction-checks'):
            return method(*args, **kwargs)
        with native_backend._substep_file_guard(resident_sources=True):
            return method(*args, **kwargs)
    return call
