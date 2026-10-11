"""One owned generated test child; preserve failures and exact source pins."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
attempt=sys.argv[1];log=HERE/('controls-'+attempt+'.log');receipt=HERE/('controls-'+attempt+'.json')
files=[HERE/'initial_goal_burden.py',HERE/'tests/test_initial_goal_burden.py',
       ROOT/'src/resectionlab/native_resection.py',ROOT/'src/resectionlab/native_proposals.py',
       ROOT/'src/resectionlab/geometry.py',ROOT/'src/resectionlab/post_exposure.py']
def pins():return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
before=pins();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
cmd=[str(ROOT/'.venv/bin/python'),'-B','-m','pytest','-q','-p','no:cacheprovider',str(HERE/'tests/test_initial_goal_burden.py')]
started=time.monotonic();timeout=False
with log.open('x') as stream:
 child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=child.wait(timeout=45)
 except subprocess.TimeoutExpired:timeout=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
after=pins();record={'schema':'bounded-generated-goal-burden-controls-v1','command':cmd,'returncode':code,
 'timed_out':timeout,'wall_seconds':time.monotonic()-started,'hard_seconds':45,'threads':1,
 'child_reaped':child.poll() is not None,'source_before':before,'source_after':after,
 'source_unchanged':before==after,'generated_only':True,'acquired_or_model_inputs':False,
 'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record));print(log.read_text());sys.exit(code or int(before!=after))
