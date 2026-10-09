"""Short parent bound and immutable input snapshots for generated review."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PREP=ROOT/'build/private-vascular-streaming-integration-preparation-v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot():
    paths=list((ROOT/'src/resectionlab').rglob('*.py'))+list((PREP/'stage').rglob('*.py'))
    paths += [PREP/name for name in ('prepare_candidate.py','source-delta.json','run_controls.py')]
    paths += [ROOT/'tests'/name for name in ('test_private_vascular_evaluation.py','test_matched_private_vascular.py','test_research_estimate_planning.py')]
    paths += [HERE/name for name in ('test_independent.py','review_worker.py','run_review.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
before=snapshot()
(HERE/'source-before.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
command=[sys.executable,'-B','-X','pycache_prefix='+str(HERE/'unused-pycache'),str(HERE/'review_worker.py')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='1',
         OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
started=time.monotonic()
with (HERE/'pytest-output.txt').open('xb') as stream:
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    error=None
    try: code=process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        error='generated_review_60_second_outer_timeout';os.killpg(process.pid,signal.SIGKILL);code=process.wait(timeout=3)
after=snapshot()
(HERE/'source-after.json').write_text(json.dumps(after,sort_keys=True,indent=2)+'\n')
receipt={'schema':'independent-streaming-integration-review-v1','command':command,'exit_code':code,'error':error,
         'elapsed_seconds':time.monotonic()-started,'source_file_count':len(before),'snapshots_equal':before==after,
         'changed_files':[name for name in sorted(set(before)|set(after)) if before.get(name)!=after.get(name)],
         'before_sha256':sha(HERE/'source-before.json'),'after_sha256':sha(HERE/'source-after.json'),
         'pytest_output_sha256':sha(HERE/'pytest-output.txt'),'model_or_native_or_patient_access':False,'full_profile_rerun':False}
(HERE/'run-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(receipt,indent=2));raise SystemExit(code if before==after else 2)
