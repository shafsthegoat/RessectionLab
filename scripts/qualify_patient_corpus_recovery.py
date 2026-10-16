#!/usr/bin/env python3
"""Fixed twelve-source recovery: fresh sequential workers, no new admission."""
from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resectionlab import patient_corpus as pc
from resectionlab.core import semantic_digest

ROOT=Path(__file__).resolve().parents[1]
DECLARATION_PATH='manifests/patient-corpus-releases/existing-twelve-recovery-v1.json'
DECLARATION_SHA256='654ba518110924cad832b185e2fe1dbcb8c329271ff1477c7ee6f6b684e65493'
OUTPUT='outputs/patient-corpus-twelve-recovery-v1'
SCRIPT=Path(__file__).resolve()


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def read_record(path):
    path=Path(path)
    pc._require(not path.is_symlink() and path.stat().st_size<=pc.MAX_METADATA_BYTES,'METADATA_LIMIT','Recovery metadata is redirected or oversized')
    return json.loads(path.read_text())


def output_path(root):
    root=Path(root);path=root/OUTPUT
    pc._require(path.resolve()==root.resolve()/OUTPUT,'OUTPUT_SCOPE','Recovery output cannot redirect to another release')
    return path


def declaration(root):
    """Only metadata/code is read; existing source hashes are not remeasured here."""
    root=Path(root);document,_=pc._metadata(root,DECLARATION_PATH,DECLARATION_SHA256)
    plan=pc.execution_plan(root)
    pc._require(plan['plan_hash']==document['qualification_plan_hash'],'PLAN_HASH','Recovery qualification plan changed')
    for name,expected in document['frozen_qualification_code_sha256'].items():
        pc._require(digest(Path(pc.__file__).with_name(name))==expected,'CODE_CHANGED','Frozen qualification validator changed: '+name)
    failure=document['prior_failure']
    for path,expected in [(failure['path'],failure['sha256']),(failure['supervision_path'],failure['supervision_sha256'])]:
        _,binding=pc._metadata(root,path,expected)
    pc._require(document['bounds']==plan['resource_bounds'],'RESOURCE_SCOPE','Recovery keeps the exact declared resource limits')
    for row in document['frozen_receipts']:
        receipt,_=pc._metadata(root,row['path'],row['file_sha256'])
        pc._require(receipt['receipt_hash']==row['receipt_hash'] and receipt['patient_group']==row['patient_group'],
            'RECEIPT_HASH','Frozen source receipt changed')
    return document,plan


def prepare(root=ROOT):
    root=Path(root);document,plan=declaration(root);output=output_path(root)
    record={'schema':'patient-corpus-exact-recovery-preparation-v1','declaration_sha256':DECLARATION_SHA256,
        'execution_plan_hash':plan['plan_hash'],'dispatcher_sha256':digest(SCRIPT),'bounds':document['bounds'],
        'stages':document['stages'],'actual_launch_authorized':False,'source_payloads_opened':False}
    record['preparation_hash']=semantic_digest(record)
    path=output/'prepared-recovery.json';pc._require(not path.is_symlink(),'OUTPUT_SCOPE','Prepared recovery cannot redirect')
    if path.exists():pc._require(read_record(path)==record,'PREPARATION_CHANGED','Keep prior recovery preparation for review')
    else:pc._atomic_json(path,record)
    return record


def expected_launch(prepared):
    return {'schema':'patient-corpus-exact-recovery-launch-v1','declaration_sha256':DECLARATION_SHA256,
        'preparation_hash':prepared['preparation_hash'],'dispatcher_sha256':prepared['dispatcher_sha256'],
        'execution_plan_hash':prepared['execution_plan_hash'],'source_decoding_authorized':True,
        'authority':'Later explicit root release of the frozen fresh-worker recovery; not the earlier failed-attempt release.',
        'automatic_retries':0,'stages':3}


def require_launch(root):
    root=Path(root);document,plan=declaration(root);output=output_path(root)
    path=output/'prepared-recovery.json'
    pc._require(path.exists(),'RECOVERY_NOT_RELEASED','Recovery preparation must be frozen before launch')
    prepared=read_record(path)
    pc._require(prepared.get('preparation_hash')==semantic_digest({k:v for k,v in prepared.items() if k!='preparation_hash'})
        and prepared['dispatcher_sha256']==digest(SCRIPT) and prepared['declaration_sha256']==DECLARATION_SHA256
        and prepared['execution_plan_hash']==plan['plan_hash'] and prepared['bounds']==document['bounds']
        and prepared['stages']==document['stages'],'PREPARATION_CHANGED','Frozen recovery preparation differs')
    launch=output/'launch-admission.json'
    pc._require(launch.exists(),'RECOVERY_NOT_RELEASED','Recovery awaits separate explicit root launch release')
    pc._require(read_record(launch)==expected_launch(prepared),'RECOVERY_NOT_RELEASED','Recovery launch receipt differs from this exact prepared protocol')
    pc._require_execution_launch(root,plan)
    return document,plan,prepared


def receipt_index(root,document):
    """Hash only receipt metadata; numeric source validation remains inside worker."""
    output=Path(root)/document['qualification_output'];rows={}
    for path in sorted((output/'receipts').glob('*.json')):
        record=read_record(path)
        pc._require(record.get('receipt_hash')==semantic_digest({k:v for k,v in record.items() if k!='receipt_hash'}),
            'RECEIPT_HASH','A qualification receipt was modified')
        patient=record['patient_group']
        pc._require(patient not in rows,'RECEIPT_SCOPE','Multiple receipts for one person require review')
        pc._require(record.get('status')=='qualified_with_limitations' and record['input_binding']['plan_hash']==document['qualification_plan_hash'],
            'RECEIPT_SCOPE','Unexpected receipt status or source release')
        rows[patient]={'path':str(path.relative_to(root)),'file_sha256':digest(path),'record':record}
    pc._require(not list((output/'rejections').glob('*.json')),'SOURCE_REJECTION','A source rejection stops recovery')
    return rows


def expected_before(document,index):
    return {r['patient_group'] for r in document['frozen_receipts']}|{
        stage['new_patient'] for stage in document['stages'][:index] if stage['new_patient'] is not None}


def check_stage_metadata(root,document,index):
    rows=receipt_index(root,document)
    pc._require(set(rows)==expected_before(document,index),'STAGE_SCOPE','Receipt identities do not match the fixed recovery stage')
    if index:
        previous=read_record(output_path(root)/f'stage-{index-1}.json')
        pc._require(previous['status']=='completed' and previous['stage']==index-1,'STAGE_SCOPE','Previous recovery stage has not completed')
        for row in previous['receipts']:
            pc._require(rows[row['patient_group']]['file_sha256']==row['file_sha256'],'RECEIPT_HASH','Receipt changed between fresh workers')
    return rows


def run_worker(root,index):
    root=Path(root).resolve();document,plan,_=require_launch(root)
    pc._require(type(index) is int and 0<=index<len(document['stages']),'STAGE_SCOPE','Only the three fixed recovery stages exist')
    pc._require(os.environ.get('RESSECTIONLAB_CORPUS_SUPERVISOR_PID')==str(os.getppid()),'SUPERVISOR_REQUIRED','Recovery requires its supervisor')
    pc._require(all(os.environ.get(name)=='2' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')),'THREAD_BOUND','Recovery worker requires two numerical threads')
    rows=check_stage_metadata(root,document,index);stage=document['stages'][index]
    # Validate all cached bindings first. An invalid prior cache cannot consume the
    # one permitted new decode or silently turn the cached-only pass into inference.
    for candidate in plan['candidates']:
        if candidate['patient_group'] in rows:
            binding=pc._input_binding(root,candidate,plan)
            pc._require(semantic_digest(binding)==rows[candidate['patient_group']]['record']['input_binding_hash'],
                'CACHED_BINDING_CHANGED','Existing source binding changed; stop for review before new decoding')
    result=pc._run_qualification(root,root/document['qualification_output'],plan,limit=stage['maximum_new_decodes'])
    pc._require(not result['rejections'] and result['counts']['rejected']==0,'SOURCE_REJECTION','New source rejection stops recovery')
    pc._require(result['counts']['qualified_existing_people']==stage['expected_qualified']
        and result['counts']['pending']==12-stage['expected_qualified']
        and result['new_qualification_receipts']==stage['maximum_new_decodes']
        and result['reused_qualification_receipts']==stage['expected_reused'],
        'STAGE_RESULT','Recovery did not produce the exact declared new/cached receipt counts')
    after=receipt_index(root,document)
    expected=set(rows)|({stage['new_patient']} if stage['new_patient'] else set())
    pc._require(set(after)==expected,'STAGE_SCOPE','Unexpected person qualified during recovery')
    for patient,row in rows.items():pc._require(after[patient]['file_sha256']==row['file_sha256'],'RECEIPT_HASH','Cached receipt changed')
    receipt={'schema':'patient-corpus-exact-recovery-stage-v1','stage':index,'stage_id':stage['id'],'status':'completed',
        'complete_twelve_person_catalog':stage['expected_qualified']==12,'final_cached_verification':index==2,'result':result,
        'receipts':[{'patient_group':patient,'path':row['path'],'file_sha256':row['file_sha256']} for patient,row in sorted(after.items())]}
    pc._atomic_json(output_path(root)/f'stage-{index}.json',receipt)


def sample_resources(psutil,root,document,started,pid=None):
    output=output_path(root)
    sample=pc._execution_sample(psutil,output,started,pid)
    sample['output_bytes']+=pc._execution_directory_bytes(Path(root)/document['qualification_output'])
    return sample


def run(root=ROOT):
    root=Path(root).resolve();document,plan,prepared=require_launch(root);output=output_path(root)
    import psutil
    bounds=document['bounds'];started=time.monotonic();process=None;peak=0;minimum=None;completed=[]
    with (output/'.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=output/'started.json'
        pc._require(not marker.exists(),'RETRY_NOT_RELEASED','A prior recovery attempt exists; no automatic retry or restart')
        pc._atomic_json(marker,{'schema':'patient-corpus-exact-recovery-start-v1','preparation_hash':prepared['preparation_hash']})
        record={'schema':'patient-corpus-exact-recovery-supervision-v1','status':'started','automatic_retries':0,
            'declaration_sha256':DECLARATION_SHA256,'preparation_hash':prepared['preparation_hash'],'bounds':bounds,'completed_stages':completed}
        try:
            for index,stage in enumerate(document['stages']):
                check_stage_metadata(root,document,index)
                sample=sample_resources(psutil,root,document,started)
                peak=max(peak,sample['rss']);minimum=sample['available'] if minimum is None else min(minimum,sample['available'])
                failure=pc._execution_resource_violation(bounds,**sample)
                pc._require(failure is None,failure or 'RESOURCE_LIMIT','Recovery preflight resource limit')
                environment=dict(os.environ,PYTHONPATH=str(root/'src'),RESSECTIONLAB_CORPUS_SUPERVISOR_PID=str(os.getpid()))
                for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):environment[name]='2'
                with (output/f'worker-{index}.log').open('wb') as log:
                    process=subprocess.Popen([sys.executable,str(SCRIPT),'--worker-stage',str(index)],cwd=root,
                        env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while True:
                        sample=sample_resources(psutil,root,document,started,process.pid)
                        peak=max(peak,sample['rss']);minimum=min(minimum,sample['available'])
                        failure=pc._execution_resource_violation(bounds,**sample)
                        pc._require(failure is None,failure or 'RESOURCE_LIMIT','Recovery resource budget exceeded')
                        if process.poll() is not None:break
                        time.sleep(bounds['monitor_interval_seconds'])
                pc._require(process.returncode==0,'WORKER_FAILURE','Recovery worker failed; no subsequent stage or retry')
                result=read_record(output/f'stage-{index}.json')
                pc._require(result['status']=='completed' and result['stage']==index,'STAGE_RESULT','Worker completion receipt missing')
                completed.append({'stage':index,'stage_id':stage['id'],'receipt_sha256':digest(output/f'stage-{index}.json')})
                process=None  # Popen.poll reaped the exited worker before the next launch.
            record.update(status='completed',complete_twelve_person_catalog=True,final_cached_verification=True)
            return record
        except BaseException as error:
            record.update(status='failed',complete_twelve_person_catalog=False,final_cached_verification=False,
                failure_code=getattr(error,'code','INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'SUPERVISOR_FAILURE'),
                reason=str(error) or type(error).__name__)
            raise
        finally:
            if process is not None:pc._stop_execution_worker(process)
            record.update(elapsed_seconds=time.monotonic()-started,peak_family_plus_supervisor_rss_bytes_sampled=peak,
                minimum_available_bytes_sampled=minimum,last_worker_exit_code=None if process is None else process.returncode)
            pc._atomic_json(output/'supervision.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker-stage',type=int,choices=(0,1,2),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.prepare:result=prepare()
    elif args.run:result=run()
    else:run_worker(ROOT,args.worker_stage);return
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
