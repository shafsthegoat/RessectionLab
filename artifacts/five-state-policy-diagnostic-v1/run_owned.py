"""Existing owned process-tree monitor for one fixed readout/baseline attempt."""
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
sys.path.insert(0, str(HERE))
from diagnostic_worker import ROOT, OUTPUT, CAPS, source_guard, sha, read, complete_result


def size(root):
    total = 0
    if not root.exists(): return total
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories+files:
            try: info = (Path(directory)/name).lstat()
            except FileNotFoundError: continue
            if stat.S_ISLNK(info.st_mode): raise ValueError('Output symlink refused')
            if stat.S_ISREG(info.st_mode): total += info.st_size
            elif not stat.S_ISDIR(info.st_mode): raise ValueError('Nonregular output refused')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); release_path = args.release.resolve(); release = read(release_path)
    index = source_guard(release, release_path, args.release_sha256)
    sys.path.insert(0, str(ROOT/'build/goal-conditioned-policy-v1'))
    from run_contact_owned import FastDarwinSampler, cleanup_owned, _receipt
    output = ROOT/OUTPUT; supervision = output.with_name(output.name+'.supervision')
    if output.exists() or output.is_symlink() or supervision.exists() or supervision.is_symlink():
        raise FileExistsError('One-attempt destination already exists')
    supervision.mkdir(exist_ok=False)
    _receipt(supervision/'declaration.json', dict(release=release, release_sha256=args.release_sha256,
        source_index=release['source_index'], source_files=index['source_files'], caps=CAPS,
        scope='fixed_five_teacher_states_and_existing_native_greedy_zero_updates'))
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
        'VECLIB_MAXIMUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1'}
    command = [sys.executable, '-I', '-B', '-X', 'pycache_prefix='+str(supervision/'fresh-pycache'),
        str(HERE/'diagnostic_worker.py'), '--release', str(release_path), '--release-sha256', args.release_sha256]
    started = time.monotonic(); peak = samples = 0; process = sampler = None
    reason = None; cleanup_actions = []; errors = []; remaining = []; detached = {}
    lease_read = lease_write = None; result = {}; result_digest = None
    try:
        sampler = FastDarwinSampler(); lease_read, lease_write = os.pipe()
        env['RESECTIONLAB_PARENT_LEASE_FD'] = str(lease_read)
        with (supervision/'worker.log').open('x') as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(lease_read,), start_new_session=True)
            finally: os.close(lease_read); lease_read = None
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= CAPS['parent_seconds']: reason = 'hard_total_wall_cap'; break
                measured = sampler.group(process.pid, process.pid)
                detached.update(measured['detached_descendant_start_identities'])
                samples += 1; peak = max(peak, measured['process_group_resident_bytes'])
                if peak > CAPS['memory_bytes']: reason = 'sampled_owned_tree_rss_cap'; break
                if size(output) > CAPS['output_bytes']: reason = 'output_cap'; break
                if size(supervision) > CAPS['supervision_bytes']-65536: reason = 'supervision_cap'; break
                _receipt(supervision/'progress.json', dict(elapsed_seconds=elapsed,
                    sampled_peak_rss_bytes=peak, samples=samples, owned_process_measurement=measured))
                time.sleep(.2)
    except BaseException as error:
        reason = reason or type(error).__name__+':'+str(error)
    finally:
        if process is not None:
            try: remaining = cleanup_owned(process, sampler, detached, cleanup_actions, errors)
            except BaseException as error:
                errors.append('cleanup:'+str(error))
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    try: process.wait(timeout=2)
                    except BaseException as failure: errors.append('wait:'+str(failure))
                for pid, identity in detached.items():
                    try: cleanup_actions.append(sampler.signal_if_same_process(int(pid),identity,signal.SIGKILL))
                    except BaseException as failure: errors.append('descendant:'+str(failure))
                try: remaining = sampler.group(process.pid,process.pid)['pids']
                except BaseException as failure: errors.append('final_group:'+str(failure))
        for fd in (lease_read, lease_write):
            if fd is not None:
                try: os.close(fd)
                except OSError as error: errors.append(str(error))
        code = None if process is None else process.poll()
        try:
            source_guard(release, release_path, args.release_sha256)
            path = output/'result.json'
            if not path.is_file() or path.stat().st_size > 2*1024**2: raise ValueError('Bounded terminal result absent')
            parsed = read(path)
            if not isinstance(parsed,dict): raise ValueError('Terminal JSON object required')
            result = parsed; result_digest = sha(path)
            if not complete_result(result): raise ValueError('Incomplete diagnostic result')
            if size(output)>CAPS['output_bytes'] or size(supervision)>CAPS['supervision_bytes']-65536:
                raise ValueError('Final output bound exceeded')
        except BaseException as error: reason = reason or 'final_guard:'+type(error).__name__+':'+str(error)
        elapsed = time.monotonic()-started
        if elapsed >= CAPS['parent_seconds']: reason = reason or 'hard_total_wall_cap'
        complete = reason is None and code == 0 and samples > 0 and not errors and not remaining
        _receipt(supervision/'receipt.json', dict(status='complete' if complete else 'failed_or_unresolved',
            stop_reason=reason, exit_code=code, elapsed_seconds=elapsed, sampled_peak_rss_bytes=peak,
            samples=samples, worker_termination_confirmed=code is not None, final_owned_pids=remaining,
            cleanup_actions=cleanup_actions, cleanup_errors=errors, result_sha256=result_digest,
            result_status=result.get('status'), caps=CAPS, release_sha256=args.release_sha256,
            source_index=release['source_index'], automatic_retry=False,
            sampling_limit='transient peaks or descendants between samples may be missed'))
    if not complete: raise RuntimeError('Diagnostic failed or unresolved: '+str(reason or code))


if __name__ == '__main__': main()
