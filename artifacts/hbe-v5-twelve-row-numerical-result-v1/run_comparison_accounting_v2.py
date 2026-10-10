"""Root-owned one-attempt supervisor for the reviewed saved-output comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_v5_remaining_one_shot as stages
from scripts import mechanics_hbe_v5_n8_one_shot as io
from scripts import febio_runtime
from launchers.hbe_v5_ordinal9_continuation_v1 import OwnedStage
from launchers.hbe_v5_nocache_host_v1 import FastDarwinSampler

BASE = ROOT / 'build/hbe-v5-native-manifest-preparation-v1'
OUTPUT = ROOT / 'build/hbe-v5-native-twelve-row-comparison-v1/attempt-02'
BINDINGS = {
    'compare_worker.py': '94e2e93cb439cf4602f95fa475c430f77f14428c37e2f4f07ae03d5953e2dacf',
    'actual-input-manifest-accounting-v2.json': '4ba15123c7d0482a38576528b64a95d0b44bd7cccbd9851e8701fca1834fe69c',
}


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Explicit one-attempt execution required')
    for name, expected in BINDINGS.items():
        path = BASE / name
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    host = FastDarwinSampler().host()
    assert host['kernel_pressure_mask'] == 1 and host['available_percent'] >= 54
    OUTPUT.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'hbe-v5-root-saved-comparison-supervision-v1',
              'status': 'started', 'bindings': BINDINGS, 'host_before': host,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'native_calls': 0, 'saved_stream_replays': 0,
              'wall_cap_seconds': 600, 'rss_cap_bytes': 3 * 1024**3,
              'output_cap_bytes': 16 * 1024**2, 'threads': 1}
    io.durable_json(OUTPUT / 'receipt.json', record)
    command = [sys.executable, '-I', '-B', '-X',
               'pycache_prefix=' + str(OUTPUT / 'unused-worker-pycache'),
               str(BASE / 'compare_worker.py'), '--root', str(ROOT),
               '--manifest', str((BASE / 'actual-input-manifest-accounting-v2.json').relative_to(ROOT)),
               '--manifest-sha256', BINDINGS['actual-input-manifest-accounting-v2.json'],
               '--output-directory', str(OUTPUT), '--execute']
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    owner = OwnedStage()
    try:
        try:
            stage = stages.supervise_stage('readout', command, OUTPUT, record,
                cwd=ROOT, environment=env, wall_cap=600, rss_cap=3 * 1024**3,
                output_cap=16 * 1024**2, rss_observer=febio_runtime.process_group_rss, popen=owner)
        finally:
            record['cleanup'] = owner.cleanup(febio_runtime.process_group_rss, time.monotonic() + 10)
        cleanup = record['cleanup']
        assert cleanup['contained'] and cleanup['direct_child_reaped']
        assert not cleanup['fallback_used'] and not cleanup['errors'] and not cleanup['remaining_members']
        assert stage['status'] == 'completed_within_caps' and stage['exit_code'] == 0
        result_file = OUTPUT / 'comparison.json'
        result = json.loads(result_file.read_bytes())
        assert result['native_output_admitted'] and result['physical_validation_pass'] is None
        for name, expected in BINDINGS.items():
            assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected
        record.update(status='completed_numerical_comparison',
                      numerical_qualification_passed=result['numerical_qualification_passed'],
                      comparison_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest(),
                      physical_validation_pass=None)
    except BaseException as error:
        record.update(status='failed_comparison_attempt', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['host_after'] = FastDarwinSampler().host()
        io.durable_json(OUTPUT / 'receipt.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'numerical_qualification_passed', 'physical_validation_pass')}))


if __name__ == '__main__':
    main()
