"""Identity-checked, fail-closed finalization for one supervised macOS child.

No bare process-group signal is used. A kernel start-identity check followed by
PID signaling still has a narrow check-to-signal race on Darwin; receipts must
state that limitation and never infer accepted resources from saved logits.
"""
import datetime
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json_once_durable(path, payload):
    """Serialize first, then fsync and atomically link a no-clobber receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode('utf-8')
    temporary = path.parent / ('.%s.%d.%s.tmp' % (path.name, os.getpid(), secrets.token_hex(8)))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    linked = False
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication; EEXIST preserves history.
        linked = True
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if linked:
            try:
                path.unlink()  # Never leave a possibly undurable positive receipt.
            except OSError:
                pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _observe_post_liveness(fast, child, pgid, detached_seen, report):
    """Never accept a run with an unverified residual worker or descendant."""
    try:
        live = fast.group(pgid, child.pid)['pids']
    except Exception as error:
        report['cleanup_errors'].append('post_group_inventory_%s: %s' % (type(error).__name__, error))
        report['residual_possible'] = True
        live = []
    if live:
        report['remaining_group_pids'] = sorted(set(live))
        report['cleanup_errors'].append('group_still_live_after_bounded_wait')
        report['residual_possible'] = True
    for pid, identity in sorted(detached_seen.items()):
        try:
            observed = fast.process_start_identity(int(pid))
            if observed == tuple(identity):
                report['remaining_detached_pids'].append(int(pid))
                report['residual_possible'] = True
        except Exception as error:
            report['cleanup_errors'].append('detached_liveness_pid_%s_%s: %s' %
                                            (pid, type(error).__name__, error))
            report['residual_possible'] = True


def finalize_child(child, fast, pgid, launch_identity, terminate_requested,
                   provisional_path, finalization_path, provisional_payload,
                   previously_seen_detached=None, wait_seconds=5.0):
    """Persist a negative provisional trace, safely stop if needed, and reap.

    Storage errors never skip cleanup. Unbound or unverifiable processes are
    not signaled; the result is negative with explicit manual attention.
    """
    if wait_seconds <= 0:
        raise ValueError('wait_seconds must be positive')
    if not isinstance(launch_identity, (tuple, list)) or len(launch_identity) != 2:
        raise ValueError('child start identity must be bound at launch')
    if int(pgid) != int(child.pid):
        raise ValueError('expected a new-session child whose PGID equals PID')
    detached_seen = previously_seen_detached or {}
    provisional = dict(provisional_payload)
    provisional.update({'accepted_for_feasibility': False,
                        'status': 'provisional_unaccepted', 'no_retry': True})
    report = {
        'scope': 'Identity-checked child cleanup with provisional trace before signaling',
        'pid': int(child.pid), 'pgid': int(pgid),
        'launch_start_identity': list(launch_identity),
        'terminate_requested': bool(terminate_requested),
        'signal_results': [], 'cleanup_errors': [],
        'provisional_receipt_write_error': None,
        'finalization_receipt_write_error': None,
        'exit_code': None, 'residual_possible': False,
        'remaining_group_pids': [], 'remaining_detached_pids': [],
        'manual_attention_required': False,
        'completed_at_utc': None,
    }
    try:
        write_json_once_durable(provisional_path, provisional)
    except Exception as error:
        report['provisional_receipt_write_error'] = '%s: %s' % (type(error).__name__, error)
        report['cleanup_errors'].append('provisional_write_failed_before_cleanup')
        report['residual_possible'] = True
        terminate_requested = True  # No durable monitor trace: fail closed and stop if still live.
        report['terminate_requested'] = True
    if terminate_requested and child.poll() is None:
        root_valid = False
        try:
            root_valid = (fast.process_start_identity(child.pid) == tuple(launch_identity)
                          and os.getpgid(child.pid) == pgid)
        except (OSError, RuntimeError) as error:
            report['cleanup_errors'].append('root_identity_check_%s: %s' % (type(error).__name__, error))
        if not root_valid:
            report['cleanup_errors'].append('root_start_identity_or_pgid_unverified_no_group_signal')
            report['residual_possible'] = True
        else:
            try:
                group = fast.group(pgid, child.pid)
                extras = sorted(set(group['pids']) - {child.pid})
                if extras:
                    report['cleanup_errors'].append('unbound_group_members_not_signaled: %s' % extras)
                    report['residual_possible'] = True
            except Exception as error:
                report['cleanup_errors'].append('group_inventory_%s: %s' % (type(error).__name__, error))
                report['residual_possible'] = True
            # The child is bound to its launch identity. The helper rechecks
            # that identity just before a PID signal, and never calls killpg.
            try:
                outcome = fast.signal_if_same_process(child.pid, tuple(launch_identity), signal.SIGKILL)
                report['signal_results'].append(outcome)
                if not outcome.get('signaled'):
                    report['cleanup_errors'].append('root_signal_not_confirmed: %s' % outcome)
                    report['residual_possible'] = True
            except Exception as error:
                report['cleanup_errors'].append('root_signal_%s: %s' % (type(error).__name__, error))
                report['residual_possible'] = True
    if terminate_requested:
        for pid, identity in sorted(detached_seen.items()):
            try:
                outcome = fast.signal_if_same_process(int(pid), tuple(identity), signal.SIGKILL)
                report['signal_results'].append(outcome)
                if not outcome.get('signaled') and fast.process_start_identity(int(pid)) == tuple(identity):
                    report['cleanup_errors'].append('detached_signal_pid_%s_not_confirmed: %s' % (pid, outcome))
                    report['residual_possible'] = True
            except Exception as error:
                report['cleanup_errors'].append('detached_signal_pid_%s_%s: %s' %
                                                (pid, type(error).__name__, error))
                report['residual_possible'] = True
    try:
        report['exit_code'] = child.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        report['cleanup_errors'].append('child_wait_timeout')
        report['residual_possible'] = True
    except Exception as error:
        report['cleanup_errors'].append('child_wait_%s: %s' % (type(error).__name__, error))
        report['residual_possible'] = True
    _observe_post_liveness(fast, child, pgid, detached_seen, report)
    report['manual_attention_required'] = bool(report['residual_possible'] or report['cleanup_errors'])
    report['completed_at_utc'] = utc_now()
    try:
        write_json_once_durable(finalization_path, report)
    except Exception as error:
        report['finalization_receipt_write_error'] = '%s: %s' % (type(error).__name__, error)
        report['cleanup_errors'].append('finalization_receipt_write_failed')
        report['manual_attention_required'] = True
    return report


def write_supervision_fail_closed(path, provisional, finalization, post_fast,
                                  post_slow, post_error, post_guard_reason,
                                  slow_errors, slow_close_error, result_exists):
    """Write final supervision even when post-sampling or slow close fails."""
    accepted = bool(
        finalization.get('exit_code') == 0 and
        provisional.get('watchdog_reason') is None and
        provisional.get('monitor_error') is None and
        not finalization.get('cleanup_errors') and
        not finalization.get('residual_possible') and
        finalization.get('provisional_receipt_write_error') is None and
        finalization.get('finalization_receipt_write_error') is None and
        not slow_errors and slow_close_error is None and
        post_error is None and post_guard_reason is None and
        post_fast is not None and post_slow is not None and
        result_exists)
    report = dict(provisional)
    report.update({
        'exit_code': finalization.get('exit_code'),
        'finalization': finalization,
        'post_fast_host': post_fast,
        'post_slow_inventory': post_slow,
        'post_error': post_error,
        'post_guard_reason': post_guard_reason,
        'slow_inventory_errors': slow_errors,
        'slow_close_error': slow_close_error,
        'result_exists': bool(result_exists),
        'accepted_for_feasibility': accepted,
        'completed_at_utc': utc_now(),
        'no_retry': True,
        'manual_attention_required': bool(finalization.get('manual_attention_required') or
                                          slow_close_error or post_error),
    })
    try:
        write_json_once_durable(path, report)
    except Exception as error:
        report['accepted_for_feasibility'] = False
        report['manual_attention_required'] = True
        report['supervision_receipt_write_error'] = '%s: %s' % (type(error).__name__, error)
        fallback = Path(path).with_name('supervision-write-failure.json')
        try:
            write_json_once_durable(fallback, {
                'accepted_for_feasibility': False, 'no_retry': True,
                'error': report['supervision_receipt_write_error'],
                'provisional_path': str(Path(path).with_name('provisional-supervision.json')),
                'finalization_path': str(Path(path).with_name('finalization.json')),
                'manual_attention_required': True,
            })
        except Exception:
            pass  # Earlier provisional/finalization receipts are still the evidence.
    return report


def closeout_supervision(child, fast, slow, pgid, launch_identity, outdir,
                         provisional_payload, baseline, contract,
                         post_collect, post_guard, post_overlap,
                         acquisition_allowlist=None, wait_seconds=5.0,
                         detached_seen=None):
    """Versioned supervisor closeout: never lose a trace on cleanup/post errors.

    `post_collect`, `post_guard`, and `post_overlap` are the unchanged host and
    overlap policy callables from the reviewed monitor. This function handles
    only finalization and receipts; it does not loosen any stop threshold.
    """
    outdir = Path(outdir)
    provisional_payload = dict(provisional_payload)
    live_at_entry = child.poll() is None
    if (live_at_entry and provisional_payload.get('watchdog_reason') is None and
            provisional_payload.get('monitor_error') is None):
        provisional_payload['watchdog_reason'] = 'child_live_after_monitor_exit'
    terminate = (provisional_payload.get('watchdog_reason') is not None or
                 provisional_payload.get('monitor_error') is not None)
    finalization = finalize_child(
        child, fast, pgid, launch_identity, terminate,
        outdir/'provisional-supervision.json', outdir/'finalization.json',
        provisional_payload, previously_seen_detached=detached_seen,
        wait_seconds=wait_seconds)
    slow_close_error = None
    try:
        slow.close()
    except Exception as error:
        slow_close_error = '%s: %s' % (type(error).__name__, error)
    try:
        slow_errors = slow.status()[2]
    except Exception as error:
        slow_errors = [{'type': type(error).__name__, 'error': str(error)}]
    post_fast = post_slow = None
    post_error = None
    post_guard_reason = None
    try:
        post_fast = fast.host()
        post_slow = post_collect(pgid, acquisition_allowlist)
        post_guard_reason = post_guard(post_fast, baseline, contract)
        if post_overlap(post_slow, pgid):
            post_guard_reason = 'post_project_compute_overlap'
    except Exception as error:
        post_error = '%s: %s' % (type(error).__name__, error)
    if post_fast is None or post_slow is None:
        post_guard_reason = post_guard_reason or 'post_host_or_inventory_unavailable'
    report = write_supervision_fail_closed(
        outdir/'supervision.json', provisional_payload, finalization,
        post_fast, post_slow, post_error, post_guard_reason,
        slow_errors, slow_close_error, (outdir/'result.json').exists())
    return report
