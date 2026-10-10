"""Manual root-released generated test attempt; not a patient/model experiment."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent


def main():
    output=WORK/'generated-attempt-01';output.mkdir()
    overlay=output/'overlay';(overlay/'src').mkdir(parents=True)
    shutil.copytree(ROOT/'src/resectionlab',overlay/'src/resectionlab',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(WORK/'stage/resectionlab',overlay/'src/resectionlab',dirs_exist_ok=True)
    (overlay/'tests').mkdir()
    for name in ('test_patient_cohort_sequential.py','test_obstruction_opening_learning_admission.py',
                 'test_paired_train_occupancy.py','test_partial_domain_factory.py',
                 'test_patient_teacher_trace_cache.py','test_obstruction_opening_proposals.py'):
        shutil.copy2(ROOT/'tests'/name,overlay/'tests'/name)
    shutil.copytree(WORK/'stage/tests',overlay/'tests',dirs_exist_ok=True)
    manifest=Path('manifests/experiments/remind-component-cohort-v1.json')
    (overlay/manifest).parent.mkdir(parents=True);shutil.copy2(ROOT/manifest,overlay/manifest)
    env=dict(os.environ,PYTHONPATH=str(overlay/'src')+os.pathsep+str(overlay/'tests'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        POST_EXPOSURE_LEARNING_BASELINE=str(WORK/'baseline/resectionlab'))
    paths=[*sorted((WORK/'stage').rglob('*.py')),
        *[ROOT/'src/resectionlab'/p.name for p in (WORK/'stage/resectionlab').glob('*.py')]]
    def pins():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before=pins();started=time.monotonic();status=None
    command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q','-c','/dev/null','--rootdir='+str(overlay),
        'tests/test_post_exposure_learning.py','tests/test_partial_domain_factory.py',
        'tests/test_obstruction_opening_learning_admission.py',
        'tests/test_patient_cohort_sequential.py::test_pinned_weight_roundtrip_preserves_old_and_new_protocols',
        'tests/test_patient_teacher_trace_cache.py::test_two_updates_exact_gradients_parameters_and_Adam_state','--junitxml='+str(output/'tests.xml')]
    with (output/'tests.log').open('xb') as log:
        try:
            result=subprocess.run(command,cwd=overlay,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            status=result.returncode
        except subprocess.TimeoutExpired:status='timeout'
    after=pins();elapsed=time.monotonic()-started
    receipt={'status':'pass' if status==0 and before==after and elapsed<30 else 'failed',
        'returncode':status,'wall_seconds':elapsed,'wall_limit_seconds':30,'threads':1,
        'source_hashes_before':before,'source_hashes_after':after,
        'scope':'generated DTO/format/loss/optimizer and tiny synthetic admission controls only',
        'acquired_array_reads':0,'released_checkpoint_reads':0,
        'canonical_writes':False,'command':command}
    (output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','returncode','wall_seconds')}))
    raise SystemExit(0 if receipt['status']=='pass' else 1)


if __name__=='__main__':main()
