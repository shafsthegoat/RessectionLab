"""Row-neutral control seam for future HBE v5 continuation wrappers.

Candidate tracked location: launchers/hbe_v5_later_continuation_core_v1.py.
No row is released by importing this module. The immutable row spec, exact
source/release checks and authenticated extension ledger belong to each bound
entry point. This core retains the reviewed old validation/native/readout path.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time


def scoped_old_execute(remaining, hint, allowed_directories: list[Path],
                       old_execute, *, expected_hint_opens: int,
                       expected_hint_bytes: int, preflight_check,
                       native_stage_check, stage_cleanup_callback,
                       owner_factory, observer) -> tuple[dict, dict]:
    """One old validator and one old execute, with restored nested hash hints.

    This is the ordinal-9 reviewed lifecycle expressed with explicit row
    parameters. The dedicated launcher temporarily replaces selected frozen
    module call seams and restores them on every exit, including a rejected
    preflight or failed child cleanup. No source file or row-specific policy
    constant is changed.
    """
    original_chain = remaining.validate_prior_chain
    original_validate = remaining.validate_release
    original_hash = remaining.io.file_hash
    original_supervise = remaining.supervise_stage
    audit = {"eligible_bytes": 0, "eligible_files": 0,
             "hinted_file_opens": 0, "hinted_file_paths": []}
    depth = 0
    validation_count = 0

    def hinted_chain(*args, **kwargs):
        nonlocal depth
        if depth:
            return original_chain(*args, **kwargs)

        def selected_hash(path, *, maximum=128 * 1024**2):
            return hint.hinted_hash(
                path, maximum=maximum, original_hash=original_hash,
                allowed_directories=allowed_directories, audit=audit)

        previous_hash = remaining.io.file_hash
        depth += 1
        remaining.io.file_hash = selected_hash
        try:
            return original_chain(*args, **kwargs)
        finally:
            remaining.io.file_hash = previous_hash
            depth -= 1

    def checked_validate(*args, **kwargs):
        nonlocal validation_count
        try:
            context = original_validate(*args, **kwargs)
            validation_count += 1
            if (audit["eligible_files"] != expected_hint_opens
                    or audit["hinted_file_opens"] != expected_hint_opens
                    or audit["eligible_bytes"] != expected_hint_bytes
                    or remaining.io.file_hash is not original_hash):
                raise ValueError("Complete declared predecessor hint coverage required")
            preflight_check(context, copy.deepcopy(audit))
            return context
        finally:
            remaining.validate_release = original_validate

    def checked_supervise(stage, *args, **kwargs):
        if stage == "native":
            native_stage_check()
        if kwargs.get("popen") is not None:
            raise ValueError("Old stage cannot override bound owned-child launcher")
        owner = owner_factory()
        outcome = None
        stage_error = None
        try:
            outcome = original_supervise(stage, *args, **dict(kwargs, popen=owner))
        except BaseException as error:
            stage_error = error
        try:
            cleanup = owner.cleanup(observer, time.monotonic() + 3)
        except BaseException as error:
            cleanup = {"contained": False, "direct_child_reaped": False,
                       "fallback_used": True, "remaining_members": [],
                       "errors": ["cleanup_exception:" + type(error).__name__]}
        stage_cleanup_callback(stage, cleanup)
        if (not cleanup["contained"] or not cleanup["direct_child_reaped"]
                or cleanup["fallback_used"] or cleanup["errors"]):
            raise RuntimeError("Owned " + stage + " child cleanup incomplete")
        if stage_error is not None:
            raise stage_error
        return outcome

    remaining.validate_prior_chain = hinted_chain
    remaining.validate_release = checked_validate
    remaining.supervise_stage = checked_supervise
    try:
        result = old_execute()
        if validation_count != 1 or remaining.io.file_hash is not original_hash:
            raise ValueError("One fully restored old validation required")
        return result, audit
    finally:
        remaining.validate_prior_chain = original_chain
        remaining.validate_release = original_validate
        remaining.io.file_hash = original_hash
        remaining.supervise_stage = original_supervise
