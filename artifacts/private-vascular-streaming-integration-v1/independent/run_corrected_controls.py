"""Rerun only six geometry cases after fixing reviewer scalar type."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
paths=list(json.loads((HERE/'source-before.json').read_bytes()))+[str(Path(__file__).relative_to(ROOT))]
def snapshot():return {name:sha(ROOT/name) for name in sorted(paths)}
before=snapshot();(HERE/'corrected-source-before.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
command=[sys.executable,'-B','-X','pycache_prefix='+str(HERE/'corrected-unused-pycache'),'-m','pytest','-q','-p','no:cacheprovider',
         str(HERE/'test_independent.py'),'-k','test_full_grid_parts_sweeps_and_duplicates_match_scalar_oracle',
         '--basetemp='+str(HERE/'corrected-six-cases-v1')]
env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
started=time.monotonic()
with (HERE/'corrected-pytest-output.txt').open('xb') as stream:
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
after=snapshot();(HERE/'corrected-source-after.json').write_text(json.dumps(after,sort_keys=True,indent=2)+'\n')
record={'schema':'independent-reviewer-fixture-correction-v1','command':command,'exit_code':result.returncode,
        'elapsed_seconds':time.monotonic()-started,'source_file_count':len(before),'snapshots_equal':before==after,
        'changed_files':[n for n in paths if before[n]!=after[n]],'reviewer_correction':'float(numpy scalar radius) matches declared builtin float contract',
        'candidate_changed':False,'initial_negative_preserved':True,'initial_test_source_sha256':sha(HERE/'test_independent-initial.py'),
        'corrected_test_source_sha256':sha(HERE/'test_independent.py'),'output_sha256':sha(HERE/'corrected-pytest-output.txt'),
        'before_sha256':sha(HERE/'corrected-source-before.json'),'after_sha256':sha(HERE/'corrected-source-after.json')}
(HERE/'corrected-run-receipt.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print(json.dumps(record,indent=2));raise SystemExit(result.returncode if before==after else 2)
