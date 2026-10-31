#!/usr/bin/env python3
"""Serial, resumable healthy-host preparation after independent recipe review."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

from resectionlab.data_root import atomic_json,checked_root,sha256_file

REPO=Path(__file__).resolve().parents[1]


def run(root=None,limit=None):
    root=checked_root(root,required_bytes=20*1024**3)
    gate_path=root/'qc/independent/IXI016-synthetic-scale-review-v5.json'
    gate=json.loads(gate_path.read_text())
    if gate.get('decision')!='accept' or gate.get('independent') is not True:
        raise ValueError('Independent v5 synthesis recipe review is required before scale-up')
    builder=REPO/'scripts/data_rebuild_prepare.py';synthesis=REPO/'src/resectionlab/data_synthesis.py'
    bindings={'builder_sha256':sha256_file(builder),'synthesis_sha256':sha256_file(synthesis)}
    for reviewed in gate['packages']:
        package=Path(reviewed['path'])
        if sha256_file(package)!=reviewed['sha256']:raise ValueError('Reviewed package changed')
        with ZipFile(package) as archive:
            software=json.loads(archive.read('manifest.json'))['metadata']['data_case']['recipe']['software']
        if any(software.get(k)!=v for k,v in bindings.items()):raise ValueError('Reviewed recipe code changed')
    directory=root/'catalog/healthy-batch';directory.mkdir(exist_ok=True)
    plan_path=directory/'plan-v5.json';state_path=directory/'state-v5.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:
        source=json.loads((root/'catalog/source-acquisition.json').read_text())
        rows=[row for row in source['candidates'] if row.get('status')=='bytes_and_header_validated_awaiting_independent_review']
        ixi=[row for row in rows if row['dataset']=='ixi']
        lausanne=[row for row in rows if row['dataset']=='lausanne' and row['subject'] in {'sub-091','sub-197'}]
        if len(ixi)<48 or len(lausanne)!=2:raise ValueError('Wait for 48 IXI TRAIN pairs and both fine Lausanne controls; do not freeze an incomplete batch')
        selected=ixi[:48]+lausanne
        mass={row['subject'] for row in ixi[:3]+lausanne}
        plan={'schema':'ressectionlab-healthy-preparation-plan-v5','created_utc':datetime.now(timezone.utc).isoformat(),
            **bindings,'independent_gate_sha256':sha256_file(gate_path),'subjects':[{'subject':row['subject'],'person_id':row['person_id'],
            'role':row['role'],'variants':4,'mass_effect':row['subject'] in mass} for row in selected],
            'expected_hosts':50,'expected_synthetic_variants':200,'no_training':True,'no_donors':True}
        atomic_json(plan_path,plan)
    if any(plan[k]!=v for k,v in bindings.items()):raise ValueError('Frozen recipe differs; preserve prior batch and use an explicit new version')
    state=json.loads(state_path.read_text()) if state_path.exists() else {'schema':'ressectionlab-healthy-preparation-state-v5','plan_sha256':sha256_file(plan_path),'subjects':{}}
    if state['plan_sha256']!=sha256_file(plan_path):raise ValueError('Batch plan changed')
    attempted=0
    for row in plan['subjects']:
        name=row['subject']
        if state['subjects'].get(name,{}).get('status')=='prepared_awaiting_independent_review':continue
        if limit is not None and attempted>=limit:break
        checked_root(root,required_bytes=2*1024**3)
        log_dir=root/'staging/preparation-logs';log_dir.mkdir(exist_ok=True)
        attempt=1+state['subjects'].get(name,{}).get('attempt',0)
        log=log_dir/f'{name}-v5-attempt-{attempt}.log'
        command=[str(REPO/'.venv/bin/python'),str(builder),'--subject',name,'--variants','4']
        if row['mass_effect']:command.append('--mass-effect')
        record={'status':'running','attempt':attempt,'command':command,'log':str(log.relative_to(root)),
            'started_utc':datetime.now(timezone.utc).isoformat()}
        state['subjects'][name]=record;atomic_json(state_path,state)
        started=time.monotonic()
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=REPO,env={**os.environ,'RESSECTIONLAB_DATA_ROOT':str(root)},stdout=stream,stderr=subprocess.STDOUT)
            returncode=child.wait()
        record.update(status='prepared_awaiting_independent_review' if returncode==0 else 'preparation_failed_preserved',
            returncode=returncode,seconds=time.monotonic()-started,log_sha256=sha256_file(log))
        atomic_json(state_path,state);attempted+=1
        print(json.dumps({'subject':name,**record}),flush=True)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--limit',type=int);args=parser.parse_args()
    run(limit=args.limit)
