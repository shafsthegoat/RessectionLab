#!/usr/bin/env python3
"""Resumable, checksummed public EDEN2020 healthy-source intake on the data drive."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from resectionlab.data_root import checked_root, atomic_json, sha256_file


def acquire(root=None, workers=18):
    root = checked_root(root, required_bytes=25*1024**3)
    base = root/'sources/eden2020-healthy-3994749'
    provenance = base/'provenance'; provenance.mkdir(parents=True, exist_ok=True)
    record = provenance/'zenodo-record.json'
    if not record.exists():
        subprocess.run(['curl','-fsSL','--retry','5','https://zenodo.org/api/records/3994749','-o',str(record)],check=True)
    document = json.loads(record.read_text())
    rows = sorted(document['files'], key=lambda row:row['key'])
    if len([r for r in rows if r['key'].endswith('.zip')]) != 15:
        raise ValueError('Expected pinned fifteen-volunteer release')
    plan = provenance/'development-plan.json'
    if not plan.exists():
        atomic_json(plan, {'created_utc':datetime.now(timezone.utc).isoformat(),
            'subjects':[r['key'][:-4] for r in rows if r['key'].endswith('.zip')],
            'role':'DEVELOPMENT','training_performed':False,'variants_per_subject':8,
            'seed_namespace':'EDEN2020-healthy-3994749-synthetic-v1',
            'source_license':'CC-BY-NC-ND-4.0','local_private_adaptations_only':True,'clinical_validation':False})
    archives=base/'archives';archives.mkdir(exist_ok=True)
    chunks=base/'transfer-chunks';chunks.mkdir(exist_ok=True)
    # Keep completed ZIPs, and reuse any verified or partial prior transfer.
    for row in rows:
        path=archives/row['key']
        if path.exists():
            with path.open('rb') as stream: digest=hashlib.file_digest(stream,'md5').hexdigest()
            if path.stat().st_size!=row['size'] or digest!=row['checksum'][4:]:
                raise ValueError('Existing source differs: '+str(path))
            atomic_json(provenance/(row['key']+'.json'),{**row,'sha256':sha256_file(path),
                'verified_utc':datetime.now(timezone.utc).isoformat(),'published_md5_verified':True,'path':str(path)})
            continue
        partial=path.with_name(path.name+'.part')
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>row['size']:raise ValueError('Oversized partial')
        block=4*1024**2
        jobs=[(start,min(start+block,row['size'])) for start in range(offset,row['size'],block)]
        print('acquiring '+row['key']+'; retained prefix '+str(offset),flush=True)
        def transfer(bounds):
            start,stop=bounds; p=chunks/f'{row["key"]}.{start}-{stop}'
            if p.exists() and p.stat().st_size==stop-start:return p
            temp=p.with_name(p.name+'.partial')
            url=f'https://zenodo.org/records/3994749/files/{row["key"]}?download=1'
            subprocess.run(['curl','-fsSL','--retry','8','--retry-all-errors','--retry-delay','4',
                '--connect-timeout','30','--max-time','600','--range',f'{start}-{stop-1}',url,'-o',str(temp)],check=True)
            if temp.stat().st_size!=stop-start:raise ValueError('Server did not honor exact byte range')
            temp.replace(p)
            return p
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index,part in enumerate(executor.map(transfer,jobs)):
                print(row['key']+f' chunk {index+1}/{len(jobs)}',flush=True)
        # Append only after every requested range exists. ZIP MD5 authenticates
        # ranges and retained prefix together before source publication.
        with partial.open('ab') as output:
            for start,stop in jobs:
                with (chunks/f'{row["key"]}.{start}-{stop}').open('rb') as stream:
                    shutil.copyfileobj(stream,output,1024**2)
        with partial.open('rb') as stream:digest=hashlib.file_digest(stream,'md5').hexdigest()
        if digest!=row['checksum'][4:] or partial.stat().st_size!=row['size']:
            raise ValueError('Published source checksum failed: '+row['key'])
        partial.replace(path)
        for start,stop in jobs:(chunks/f'{row["key"]}.{start}-{stop}').unlink()
        atomic_json(provenance/(row['key']+'.json'),{**row,'sha256':sha256_file(path),
            'verified_utc':datetime.now(timezone.utc).isoformat(),'published_md5_verified':True,'path':str(path)})
        print('verified '+row['key'],flush=True)
    atomic_json(provenance/'acquisition.json',{'complete':True,'files':[
        {**r,'sha256':sha256_file(archives/r['key'])} for r in rows]})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--workers',type=int,default=18)
    args=parser.parse_args()
    if not 1<=args.workers<=24:parser.error('Use 1–24 transfer workers')
    acquire(args.root,args.workers)
