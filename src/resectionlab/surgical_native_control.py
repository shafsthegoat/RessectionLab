"""Explicit cancellation/deadline transport for known planner checks.

Only production checks whose effects are cancellation and budget enforcement
may be wrapped. An arbitrary callable remains on the reference execution path.
"""
from __future__ import annotations

import math
from time import monotonic


class NativeCheck:
    """Keep the outer check callable while sharing a native stop token."""

    __slots__ = ('_check', '_token', '_deadline')

    def __init__(self, check, token, *, deadline=None):
        if not callable(check):
            raise TypeError('Native outer check must be callable')
        if deadline is not None and (type(deadline) not in (int, float)
                or not math.isfinite(deadline)):
            raise ValueError('Native deadline must be finite monotonic time')
        self._check, self._token, self._deadline = check, token, deadline

    def __call__(self):
        return self._check()


def derive_native_check(parent, check, *, deadline=None):
    """Explicitly bind a known guard; never inspect arbitrary callbacks."""
    if type(parent) is not NativeCheck:
        return check
    if parent._deadline is not None:
        deadline = parent._deadline if deadline is None else min(parent._deadline, deadline)
    return NativeCheck(check, parent._token, deadline=deadline)


def combine_native_checks(first, second, combined):
    """Mark the meter's existing composition without replacing its closure."""
    if (type(first) is NativeCheck and type(second) is NativeCheck
            and first._token is second._token):
        return derive_native_check(first, combined, deadline=second._deadline)
    return combined


def native_stop_token(check):
    """Prepare the shared token immediately before one native action."""
    if type(check) is not NativeCheck:
        return None
    remaining = None if check._deadline is None else max(0., check._deadline - monotonic())
    check._token.set_timeout(remaining)
    return check._token


def translate_native_stop(check, error):
    """Recover the existing outer cancellation/deadline exception and reason."""
    if (type(check) is NativeCheck and type(error) is RuntimeError
            and len(error.args) == 1 and type(error.args[0]) is str
            and error.args[0] in ('SIMULATION_CANCELLED', 'SIMULATION_DEADLINE_EXCEEDED')):
        check()
    raise error
