"""One bounded child; reuse exact root-tested Darwin exit lifecycle bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(HERE))
from common import HELPER, HELPER_SHA, SAMPLER, SAMPLER_SHA, guard, sha, tree_bytes, write_json
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def exact_module(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest: raise ValueError('Reviewed helper changed')
    namespace = {'__file__': str(path), '__name__': path.stem}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-index-sha256', required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    index_path = HERE/'source-index.json'
    if sha(index_path) != args.source_index_sha256: raise ValueError('Exact frozen index required')
    index = json.loads(index_path.read_bytes()); guard(ROOT, index)
    config = json.loads((HERE/'config.json').read_bytes()); caps = config['caps']
    if not args.execute:
        print(json.dumps({'status': 'source_only_checked', 'native_previews': 0})); return 0
    if not index['execution_released']: raise ValueError('Root has not released the single attempt')
    if not sys.dont_write_bytecode or not sys.pycache_prefix or Path(sys.pycache_prefix).exists():
        raise ValueError('Parent requires -I -B and a fresh absent pycache prefix')
    output = ROOT/config['output_directory']
    if output.exists() or output.is_symlink(): raise FileExistsError('Fresh exact one-attempt output required')
    build = HERE.parent
    if tree_bytes(build)+caps['output_bytes'] > caps['whole_build_bytes']: raise ValueError('Whole-build reservation cap')
    if shutil.disk_usage(ROOT).free < 50*1024**3+caps['output_bytes']: raise ValueError('50GiB free floor plus output reserve')
    OwnedChildExit = exact_module(ROOT/HELPER, HELPER_SHA)['OwnedChildExit']
    Sampler = exact_module(ROOT/SAMPLER, SAMPLER_SHA)['FastDarwinSampler']
    sampler = Sampler(); output.mkdir(); started = time.monotonic()
    receipt = {'schema': 'owned-four-TRAIN-cache-readout-recovery-v1', 'status': 'running', 'source_index_sha256': args.source_index_sha256,
               'caps': caps, 'automatic_retry': False, 'sampled_peak_group_rss_bytes': 0, 'rss_samples': 0,
               'helper_sha256': HELPER_SHA, 'sampler_sha256': SAMPLER_SHA, 'command': None}
    write_json(output/'intent.json', receipt, 65536)
    process = owned = None; reason = None; lease_read = lease_write = None; previous = {}; requested = False
    def interrupted(signum, frame):
        nonlocal requested
        requested = True; raise KeyboardInterrupt('termination requested')
    for sig in (signal.SIGINT, signal.SIGTERM): previous[sig] = signal.signal(sig, interrupted)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'DYLD_', 'LD_'))}
    temporary = output/'tmp'; temporary.mkdir()
    env.update(THREADS, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(temporary))
    try:
        cache = output/'absent-worker-cache'
        if cache.exists() or cache.is_symlink(): raise ValueError('Fresh child source-cache prefix required')
        command = [sys.executable, '-I', '-B', '-X', 'pycache_prefix='+str(cache), str(HERE/'worker.py'),
                   '--source-index-sha256', args.source_index_sha256, '--output', str(output)]
        receipt['command'] = command
        lease_read, lease_write = os.pipe(); env['RESECTIONLAB_PARENT_LEASE_FD'] = str(lease_read)
        with (output/'worker.log').open('x') as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lease_read,), start_new_session=True)
            finally:
                os.close(lease_read); lease_read = None
            owned = OwnedChildExit(process, sampler)
            while process.poll() is None:
                if time.monotonic()-started >= caps['work_seconds']:
                    reason = 'work_deadline_with_30s_cleanup_reserve'; break
                observation = owned.sample()
                if observation['state'] == 'exited': break
                measured = observation['measurement']; receipt['rss_samples'] += 1
                receipt['sampled_peak_group_rss_bytes'] = max(receipt['sampled_peak_group_rss_bytes'], measured['process_group_resident_bytes'])
                if measured['process_group_resident_bytes'] > caps['worker_peak_rss_bytes']: reason = 'rss_cap'; break
                if os.fstat(log.fileno()).st_size > caps['log_bytes']: reason = 'log_cap'; break
                if tree_bytes(output, live=True) > caps['output_bytes']-65536: reason = 'output_cap'; break
                if tree_bytes(build, live=True, live_output=output) > caps['whole_build_bytes']-65536: reason = 'whole_build_cap'; break
                if shutil.disk_usage(ROOT).free < 50*1024**3: reason = 'free_disk_floor'; break
                time.sleep(.1)
    except BaseException as error:
        reason = reason or type(error).__name__+': '+str(error)
    finally:
        accepted = False; cleanup = None
        try:
            if owned is not None:
                cleanup = owned.cleanup(deadline=started+caps['parent_wall_seconds']-1.)
            elif process is not None:
                # Failed lifecycle initialization is never a successful run.
                if process.poll() is None: process.kill()
                process.wait(timeout=max(.01, min(2., started+caps['parent_wall_seconds']-time.monotonic())))
                reason = reason or 'owned_lifecycle_not_initialized'
        except BaseException as error:
            receipt['cleanup_exception'] = {'phase': 'outer_cleanup', 'pid': None if process is None else process.pid,
                'exception': type(error).__name__, 'errno': getattr(error, 'errno', None), 'message': str(error)}
            reason = reason or 'cleanup_failure'
            if process is not None:
                try:
                    if process.poll() is None: process.kill()
                    process.wait(timeout=max(.01, min(.5, started+caps['parent_wall_seconds']-time.monotonic())))
                except BaseException as final:
                    receipt['direct_child_fallback_failure'] = {'exception': type(final).__name__, 'errno': getattr(final, 'errno', None), 'message': str(final)}
        finally:
            for fd in (lease_read, lease_write):
                if fd is not None:
                    try: os.close(fd)
                    except OSError as error:
                        receipt['lease_close_error'] = str(error); reason = reason or 'lease_close_failure'
            receipt.update(owned_lifecycle=cleanup, stop_reason=reason,
                           worker_exit_code=None if process is None else process.poll())
            try:
                guard(ROOT, index)
                report_path = output/'report.json'
                report = json.loads(report_path.read_bytes())
                receipt.update(report_sha256=sha(report_path), report_status=report.get('status'),
                               useful_criterion=report.get('useful_criterion'))
                if (output/'progress.json').exists():
                    receipt['last_saved_costs'] = json.loads((output/'progress.json').read_bytes())
                accepted = (reason is None and not requested and cleanup is not None and cleanup['verified']
                    and cleanup['child_reaped'] and cleanup['exit_code'] == 0 and not cleanup['remaining_owned_pids']
                    and not cleanup['errors'] and report['status'] == 'completed' and report['source_unchanged']
                    and report['useful_criterion']['comparison_admissible'] and len(report['cases']) == 4
                    and len(report['reconstructed_traces']) == 36 and report['head_optimizer_updates'] == 64
                    and report['baseline_reproduced'] is True and report['calibration_calls_completed'] == 1
                    and report['calibration_calls_attempted'] == 1 and report['cache']['bytes'] <= 2*1024**2
                    and report['reconstruction_executed_this_attempt'] is False and report['cache_copied'] is False
                    and report['logged_prediction_readouts_completed'] == 2
                    and report['total_frozen_policy_forwards'] <= caps['frozen_policy_forwards']
                    and report['native_budget']['native_preview_entries'] <= caps['native_previews']
                    and report['self_peak_rss_bytes'] <= caps['worker_peak_rss_bytes']
                    and receipt['rss_samples'] > 0 and time.monotonic()-started < caps['parent_wall_seconds']
                    and tree_bytes(output) <= caps['output_bytes']-65536 and tree_bytes(build) <= caps['whole_build_bytes']-65536)
            except BaseException as error:
                accepted = False
                receipt['finalization_error'] = {'exception': type(error).__name__, 'message': str(error)}
            receipt['status'] = 'completed' if accepted else 'incomplete_or_failed'
            receipt['elapsed_seconds'] = time.monotonic()-started
            try:
                write_json(output/'launcher.json', receipt, 65536)
            finally:
                for sig, handler in previous.items(): signal.signal(sig, handler)
    print(json.dumps({'status': receipt['status'], 'elapsed_seconds': receipt['elapsed_seconds'], 'stop_reason': reason}))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
