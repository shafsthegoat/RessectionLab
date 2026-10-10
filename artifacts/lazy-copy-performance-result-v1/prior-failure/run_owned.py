"""One owned single-arm TRAIN opening-proposal search; no retry, model or held-out work."""
import argparse
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from pilot_contract import (OUTPUT, TRAIN, CLOSED, WORKER_SECONDS, PARENT_SECONDS,
    MEMORY_BYTES, OUTPUT_BYTES, SUPERVISION_BYTES, source_guard, sha, complete_result,
    endpoint_control, canonical)


def tree_bytes(root):
    """Metadata-only accounting, refusing output links; no patient array reads."""
    if not root.exists(): return 0
    total = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory)/name
            try: info = path.lstat()
            except FileNotFoundError: continue  # Atomic progress replacement.
            if stat.S_ISLNK(info.st_mode): raise ValueError('Output symlink refused: '+str(path))
            if stat.S_ISREG(info.st_mode): total += info.st_size
            elif not stat.S_ISDIR(info.st_mode): raise ValueError('Nonregular output refused: '+str(path))
    return total


def small_json(path):
    if not path.is_file() or path.stat().st_size > 2*1024**2:
        raise ValueError('Bounded terminal JSON required: '+str(path))
    value = json.loads(path.read_text())
    if type(value) is not dict: raise ValueError('Terminal JSON object required')
    return value


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release_path = args.release.resolve()
    release = small_json(release_path)
    index = source_guard(release, release_path, args.release_sha256)
    # Import the reused owned-process helpers only after their source guard.
    sys.path.insert(0, str(ROOT/'build/goal-conditioned-policy-v1'))
    from run_contact_owned import cleanup_owned, FastDarwinSampler, _receipt
    output = ROOT/OUTPUT
    supervision = output.with_name(output.name+'.supervision')
    if output.exists() or output.is_symlink() or supervision.exists() or supervision.is_symlink():
        raise FileExistsError('Attempt or reservation exists; no retry or policy switching')
    supervision.mkdir(exist_ok=False)
    caps = {'hard_total_wall_seconds': PARENT_SECONDS, 'worker_deadline_seconds': WORKER_SECONDS,
        'sampled_owned_tree_rss_bytes': MEMORY_BYTES, 'output_bytes': OUTPUT_BYTES,
        'supervision_bytes': SUPERVISION_BYTES, 'worker_log_bytes': 8*1024**2,
        'threads': 1, 'sample_interval_seconds': .2, 'output_sample_interval_seconds': 1.,
        'attempts': 1, 'automatic_retry': False}
    _receipt(supervision/'declaration.json', {'version': 'owned-paired-TRAIN-obstruction-opening-search-v1',
        'head': release['expected_head'], 'release_path': str(release_path),
        'release_sha256': args.release_sha256, 'source_index': release['source_index'],
        'source_files': index['source_files'], 'caps': caps, 'condition_admission': release['condition_admission'],
        'TRAIN': list(TRAIN), 'planned_arms': release['planned_arms'],
        'closed_roles': CLOSED, 'SELECT_EVAL_execution': False, 'output': str(output),
        'completion_scope': 'One fixed TRAIN025 route; zero models/updates/checkpoints; complete search/seal/own-world replay/independent geometry or explicit failures',
        'recipe':release['recipe'],'input_index':release['input_index'],
        'execution_limits':release['execution_limits'],
        'preview_bound_scope':'Per-child10000 previews and72 transitions, including24 greedy and48 rollout/replay'})
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    command = [sys.executable, '-I', '-B', '-X', 'pycache_prefix='+str(supervision/'fresh-pycache'),
        str(HERE/'search_worker.py'), '--release', str(release_path),
        '--release-sha256', args.release_sha256]
    started = time.monotonic(); peak = samples = 0; reason = None
    errors = []; actions = []; detached = {}; remaining = []; process = None
    lease_read = lease_write = None; sampler = None; output_size = supervision_size = 0
    result = {}; result_digest = None; worker_final = {}; control_digest = None
    try:
        sampler = FastDarwinSampler()
        lease_read, lease_write = os.pipe()
        env['RESECTIONLAB_PARENT_LEASE_FD'] = str(lease_read)
        with (supervision/'worker.log').open('x') as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(lease_read,), start_new_session=True)
            finally:
                os.close(lease_read); lease_read = None
            next_output_check = 0.
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= PARENT_SECONDS: reason = 'hard_total_wall_cap'; break
                if os.fstat(log.fileno()).st_size > caps['worker_log_bytes']:
                    reason = 'worker_log_cap'; break
                measured = sampler.group(process.pid, process.pid)
                detached.update(measured['detached_descendant_start_identities'])
                samples += 1; peak = max(peak, measured['process_group_resident_bytes'])
                if measured['process_group_resident_bytes'] > MEMORY_BYTES:
                    reason = 'sampled_owned_tree_rss_cap'; break
                if elapsed >= next_output_check:
                    output_size, supervision_size = tree_bytes(output), tree_bytes(supervision)
                    if output_size > OUTPUT_BYTES: reason = 'output_cap'; break
                    # Leave 64 KiB for the durable terminal record, including failures.
                    if supervision_size > SUPERVISION_BYTES-65536:
                        reason = 'supervision_output_cap'; break
                    next_output_check = elapsed+1.
                _receipt(supervision/'progress.json', {'elapsed_seconds': elapsed,
                    'worker_pid': process.pid, 'sampled_peak_rss_bytes': peak, 'samples': samples,
                    'owned_process_measurement': measured, 'output_bytes_at_last_stat': output_size,
                    'supervision_bytes_at_last_stat': supervision_size})
                time.sleep(.2)
    except BaseException as error:
        reason = reason or type(error).__name__+':'+str(error)
    finally:
        if process is not None:
            try:
                remaining = cleanup_owned(process, sampler, detached, actions, errors)
            except BaseException as error:
                errors.append('owned_cleanup_error:'+type(error).__name__+':'+str(error))
                # The retained Popen is the authority for this fallback. Never
                # signal a global name or silently turn a cleanup error into success.
                try:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        actions.append({'target': 'owned_process_group_fallback', 'signal': int(signal.SIGKILL)})
                except BaseException as failure:
                    errors.append('owned_group_fallback_error:'+type(failure).__name__+':'+str(failure))
                try:
                    if process.poll() is None:
                        process.kill()
                        actions.append({'target': 'retained_direct_child_fallback', 'signal': int(signal.SIGKILL)})
                except BaseException as failure:
                    errors.append('direct_child_kill_error:'+type(failure).__name__+':'+str(failure))
                try: process.wait(timeout=2)
                except BaseException as failure:
                    errors.append('direct_child_wait_error:'+type(failure).__name__+':'+str(failure))
                for pid, identity in detached.items():
                    try: actions.append(sampler.signal_if_same_process(int(pid), identity, signal.SIGKILL))
                    except BaseException as failure: errors.append('descendant_cleanup_error:'+str(failure))
                try: remaining = sampler.group(process.pid, process.pid)['pids']
                except BaseException as failure: errors.append('final_owned_sample_error:'+str(failure))
        for fd in (lease_read, lease_write):
            if fd is not None:
                try: os.close(fd)
                except OSError as error: errors.append('lease_close_error:'+str(error))
        code = None if process is None else process.poll()
        try:
            output_size, supervision_size = tree_bytes(output), tree_bytes(supervision)
            if output_size > OUTPUT_BYTES: raise ValueError('Final output cap exceeded')
            if supervision_size > SUPERVISION_BYTES-65536: raise ValueError('Final supervision reserve exceeded')
            if (supervision/'worker.log').stat().st_size > caps['worker_log_bytes']:
                raise ValueError('Final worker log cap exceeded')
            source_guard(release, release_path, args.release_sha256)
            result = small_json(output/'result.json'); result_digest = sha(output/'result.json')
            worker_final = small_json(supervision/'worker-final.json')
            if not complete_result(result): raise ValueError('Incomplete fixed TRAIN025 endpoint')
            if (worker_final.get('status') != 'complete_owned_paired_TRAIN_obstruction'
                    or worker_final.get('result_sha256') != result_digest
                    or worker_final.get('canonical_result_sha256') != result_digest):
                raise ValueError('Worker final record does not bind completed canonical result')
            control_path=supervision/'endpoint-control.json'
            control=small_json(control_path);control_digest=sha(control_path)
            if (canonical(control)!=canonical(endpoint_control(output,release))
                    or worker_final.get('endpoint_control_sha256')!=control_digest):
                raise ValueError('Endpoint comparison is not exactly bound by worker completion')
        except BaseException as error:
            reason = reason or 'final_guard:'+type(error).__name__+':'+str(error)
        elapsed = time.monotonic()-started
        if elapsed >= PARENT_SECONDS: reason = reason or 'hard_total_wall_cap'
        complete = reason is None and code == 0 and samples > 0 and not errors and not remaining
        record = {'status': 'complete' if complete else 'failed_or_unresolved', 'stop_reason': reason,
            'exit_code': code, 'worker_termination_confirmed': code is not None,
            'elapsed_seconds': elapsed, 'sampled_peak_rss_bytes': peak, 'samples': samples,
            'cleanup_actions': actions, 'cleanup_errors': errors, 'final_owned_pids': remaining,
            'output_bytes': output_size, 'supervision_bytes_before_receipt': supervision_size,
            'canonical_result_status': result.get('status'), 'result_sha256': result_digest,
            'endpoint_control_sha256': control_digest,
            'worker_final_sha256': sha(supervision/'worker-final.json') if (supervision/'worker-final.json').is_file() else None,
            'checkpoint_loads':0, 'caps': caps, 'release_sha256': args.release_sha256,
            'source_index': release['source_index'], 'automatic_retry': False,
            'SELECT_EVAL_execution': False, 'private_reference_execution': False,
            'sampling_limit': 'transient peaks or descendants between samples may be missed',
            'worker_process_scope': 'owned process group plus discovered identity-matched descendants'}
        _receipt(supervision/'receipt.json', record)
        if tree_bytes(supervision) > SUPERVISION_BYTES:
            complete = False; record.update(status='failed_or_unresolved', stop_reason='closed_supervision_output_cap')
            _receipt(supervision/'receipt.json', record)
    if not complete: raise RuntimeError('Owned fixed TRAIN025 attempt failed or unresolved: '+str(reason or code))


def main():
    def terminated(signum, frame):
        raise InterruptedError('Owned paired parent received signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,terminated)
    try:return _main()
    finally:signal.signal(signal.SIGTERM,previous)

if __name__ == '__main__': main()
