"""One bounded generated producer/checker integration process; retain failures."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
attempt=sys.argv[1];log=HERE/('producer-controls-'+attempt+'.log');receipt=HERE/('producer-controls-'+attempt+'.json')
test_files=[HERE/'tests/test_native_cut_union.py',ROOT/'tests/test_independent_cut_union.py',
            ROOT/'tests/test_native_resection.py',ROOT/'tests/test_native_spatial_evaluation.py',
            ROOT/'tests/test_native_immutable_state.py',ROOT/'tests/test_native_spatial_task.py']
files=[HERE/'stage/resectionlab/native_resection.py',HERE/'stage/resectionlab/native_cut_union.py',
       HERE/'tests/test_native_cut_union.py',ROOT/'src/resectionlab/evaluation.py',
       ROOT/'src/resectionlab/independent_cut_union.py',Path(__file__).resolve(),*test_files[1:]]
def pins():return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
before=pins();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
cmd=[str(ROOT/'.venv/bin/python'),'-B','-m','pytest','-q','-p','no:cacheprovider',*map(str,test_files)]
started=time.monotonic();timeout=False
with log.open('x') as stream:
 child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=child.wait(timeout=45)
 except subprocess.TimeoutExpired:timeout=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
after=pins();record={'schema':'bounded-generated-native-cut-union-producer-v1','command':cmd,'returncode':code,
 'timed_out':timeout,'wall_seconds':time.monotonic()-started,'hard_seconds':45,'threads':1,
 'child_reaped':child.poll() is not None,'source_before':before,'source_after':after,
 'source_unchanged':before==after,'generated_only':True,'acquired_or_model_inputs':False,
 'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record));print(log.read_text());sys.exit(code or int(before!=after))
