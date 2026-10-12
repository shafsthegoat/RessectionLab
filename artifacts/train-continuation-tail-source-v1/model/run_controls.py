"""Single bounded generated suite using the existing owned exit/RSS utilities."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FILES = ('src/resectionlab/spatial_continuation_tail.py',
         'src/resectionlab/spatial_continuation_planner.py',
         'tests/test_spatial_continuation_tail.py',
         'tests/test_spatial_continuation_planner.py',
         'src/resectionlab/spatial_continuation_value.py',
         'src/resectionlab/spatial_continuation_calibration.py')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def exact(path, digest):
    path = ROOT/path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest: raise ValueError('Utility pin changed')
    namespace = {'__file__': str(path), '__name__': path.stem}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace
def size(): return sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())

def main():
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError('Require isolated -I -B parent')
    receipt, log = HERE/'controls-01.json', HERE/'controls-01.log'
    if receipt.exists() or log.exists(): raise FileExistsError('Preserve prior control attempts')
    for name in FILES: ast.parse((ROOT/name).read_text())
    before = {name: sha(ROOT/name) for name in FILES}
    Owned = exact('build/native-cut-residual-diagnostic-v1/monitor-exit-repair-v1/owned_child_exit.py',
        'aff4b197ea033aef8b160f43eabdd5a03394ae4024b6fc2f59f5241f2fcbfe14')['OwnedChildExit']
    Sampler = exact('build/goal-conditioned-policy-v1/darwin_fast_sampler.py',
        '906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c')['FastDarwinSampler']
    env = {k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','DYLD_','LD_'))}
    env.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    temporary = HERE/'controls-01-tmp'; temporary.mkdir()
    env['TMPDIR'] = str(temporary); env['PYTHONDONTWRITEBYTECODE'] = '1'
    cache = HERE/'absent-controls-01-cache'
    if cache.exists(): raise FileExistsError('Fresh absent cache required')
    command = [sys.executable,'-I','-B','-X','pycache_prefix='+str(cache),str(HERE/'control_bootstrap.py')]
    started = time.monotonic(); peak = 0; reason = None
    sampler = Sampler()
    with log.open('x') as output:
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        owned = Owned(child,sampler)
        try:
            while child.poll() is None:
                if time.monotonic()-started >= 57: reason='work deadline'; break
                row = owned.sample()
                if row['state'] == 'exited': break
                peak=max(peak,row['measurement']['process_group_resident_bytes'])
                if peak>1024**3: reason='1GiB group RSS cap'; break
                if size()>8*1024**2-65536: reason='8MiB output cap'; break
                time.sleep(.05)
        except BaseException as error:
            reason=type(error).__name__+': '+str(error)
        finally:
            cleanup=owned.cleanup(deadline=started+60)
    after={name:sha(ROOT/name) for name in FILES}
    accepted=(not reason and child.returncode==0 and cleanup['verified'] and before==after)
    record={'accepted':accepted,'returncode':child.returncode,'failure':reason,'cleanup':cleanup,
        'command':command,'wall_seconds':time.monotonic()-started,'peak_group_rss_bytes':peak,
        'caps':{'seconds':60,'group_rss_bytes':1024**3,'output_bytes':8*1024**2,'threads':1},
        'source_before':before,'source_after':after,'source_unchanged':before==after,
        'log_sha256':sha(log),'log_bytes':log.stat().st_size,
        'scope':'generated-only, no acquired payload or saved weights'}
    receipt.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(log.read_text()); print(json.dumps({k:v for k,v in record.items() if k not in ('cleanup','command','source_before','source_after')},sort_keys=True))
    return 0 if accepted else 1

if __name__ == '__main__': raise SystemExit(main())
