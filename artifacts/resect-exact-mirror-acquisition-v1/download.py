#!/usr/bin/env python3
"""One bounded, isolated acquisition of 24 checksum-identical TRAIN masks."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import ssl
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
REVISION = 'e86fb37dd93f7a9c64e48952f71410af59b04b9b'
BINDINGS = {
 'build/resect-osf-bottleneck-investigation-v1/mirror-metadata-comparison.json':'031a97a9588c2dd36054d8c7b19b3916a034dfa1fc4db9432a7259bbe71f00d7',
 'manifests/resect-train-cavity-acquisition-v1.json':'09a15ffe52af966c27d53fba4924a7c67fba5590b12e53cb48aa022ef60f04f1',
 'artifacts/resect-component-admission-v1/rights-review-02/README.txt':'0a75fef344487ee2649a831919388940f2000bf4cb936979e0da2f73a9aa81a9',
 'build/resect-osf-bottleneck-investigation-v1/hf-tree-metadata.json':'6f3a9c90b3848e7ff367acf72d8764a95660cf9b3466efd8bf6d582e09719776',
 'build/resect-osf-bottleneck-investigation-v1/hf-README.md':'f28e590b6b35f781f57f3be333eba75a4826542326054872c507de1b45104023'}
CDN_HOSTS = {'cas-bridge.xethub.hf.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.hf.co',
             'cdn-lfs-eu-1.hf.co', 'us.aws.cdn.hf.co', 'us.gcp.cdn.hf.co'}
CAP_BYTES, CAP_SECONDS, CAP_REQUESTS = 10*1024**2, 900, 144
CLAIMS = {'header_qc':'not_run','geometry_qc':'not_run','anatomy_qc':'not_run',
          'training_admitted':False,'spatial_planning_admitted':False,'decoded_array_bytes':0}

class Refusal(ValueError): pass
class Deadline(TimeoutError): pass
class Deferred(Exception):
    def __init__(self, until, status): self.until, self.status = until, status
class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None

def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(value): return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def now(): return datetime.now(timezone.utc).isoformat()
def safe(path):
    path = path.absolute()
    if not path.is_relative_to(ROOT) or '..' in path.parts: raise Refusal('path_scope')
    current=ROOT
    for part in path.relative_to(ROOT).parts:
        current/=part
        if current.is_symlink(): raise Refusal('symlink_refused')
    return path
def small(path):
    with safe(path).open('rb') as stream: data=stream.read(2*1024**2+1)
    if len(data)>2*1024**2: raise Refusal('metadata_cap')
    return data
def preserve(path, raw):
    path=safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
def save(path,value): preserve(path,enc(value))
def binding(row): return sha(enc(row))

def prepare():
    retained={}
    for name,expected in BINDINGS.items():
        raw=small(ROOT/name)
        if sha(raw)!=expected:raise Refusal('reference_binding_changed')
        retained[name]=raw
    comparison=json.loads(retained[next(iter(BINDINGS))])
    manifest=json.loads(retained['manifests/resect-train-cavity-acquisition-v1.json'])
    tree={r['path']:r for r in json.loads(retained['build/resect-osf-bottleneck-investigation-v1/hf-tree-metadata.json'])}
    authority={s['id']:s for s in manifest['sources']}
    pairs={p['id']:p for p in manifest['pairs']}
    rows=[]
    for candidate in comparison['rows']:
        if not candidate['currently_missing']:continue
        original=authority[candidate['source_id']];pair=pairs[original['pair_id']]
        mirrored=tree[candidate['mirror_path']]
        expected_url=f'https://huggingface.co/datasets/MedOtter/RESECT-SEG/resolve/{REVISION}/{candidate["mirror_path"]}'
        if (original['kind']!='cavity_annotation' or original['file_revision']!=2 or pair['role']!='TRAIN'
            or candidate['candidate_url']!=expected_url or candidate['published_sha256']!=original['sha256']
            or candidate['published_md5']!=original['expected_md5'] or candidate['expected_bytes']!=original['bytes']
            or mirrored['lfs']['oid']!=original['sha256'] or mirrored['size']!=original['bytes']
            or mirrored['lfs']['size']!=original['bytes'] or not re.fullmatch('[a-f0-9]{64}',mirrored['xetHash'])):
            raise Refusal('exact_source_scope')
        rows.append({'source_id':original['id'],'source_authority':original,'patient_group':pair['patient_group'],
            'role':'TRAIN','source_release':manifest['release_by_kind']['cavity_annotation'],
            'mirror_commit':REVISION,'mirror_path':candidate['mirror_path'],'mirror_url':expected_url,
            'mirror_xet_hash':mirrored['xetHash'],'expected_bytes':original['bytes'],
            'sha256':original['sha256'],'md5':original['expected_md5']})
    if len(rows)!=24 or len({r['source_id'] for r in rows})!=24 or sum(r['expected_bytes'] for r in rows)!=1561749:
        raise Refusal('frozen_24_file_denominator')
    return rows,retained,manifest['rights_and_semantics']

def route(row,url,initial=False):
    if not isinstance(url,str) or len(url)>16384 or any(not 33<=ord(c)<=126 for c in url):raise Refusal('url_encoding')
    p=urlsplit(url)
    if p.scheme!='https' or p.username or p.password or p.fragment or p.port not in (None,443) or p.netloc!=p.hostname:
        raise Refusal('https_authority_required')
    if initial or p.hostname=='huggingface.co':
        exact=row['mirror_url']
        cached=f'https://huggingface.co/api/resolve-cache/datasets/MedOtter/RESECT-SEG/{REVISION}/{row["mirror_path"]}'
        if url!=exact and (initial or p._replace(query='').geturl()!=cached):raise Refusal('exact_pinned_hf_path_required')
    elif p.hostname not in CDN_HOSTS or p.path.rstrip('/').split('/')[-1] not in {row['sha256'],row['mirror_xet_hash']}:
        raise Refusal('selected_object_redirect_required')
    return {'host':p.hostname,'path':p.path,'full_url_sha256':sha(url.encode()),
            'query_field_names':sorted({k for k,v in parse_qsl(p.query,keep_blank_values=True)})}

def after(value):
    if not value or len(value)>128:return None
    if re.fullmatch('[0-9]{1,12}',value.strip()):return time.time()+int(value)
    try:
        dt=parsedate_to_datetime(value)
        if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        return max(time.time(),dt.timestamp())
    except (ValueError,TypeError,OverflowError):return None

def check(state):
    if time.monotonic()>=state['deadline']:raise Deadline('aggregate_deadline')
    if state['bytes_read']>CAP_BYTES or state['requests']>=CAP_REQUESTS:raise Refusal('aggregate_resource_cap')

def verify(path,row,state=None):
    if state:check(state)
    a,b=hashlib.sha256(),hashlib.md5();n=0
    with safe(path).open('rb') as stream:
        before=os.fstat(stream.fileno())
        if before.st_size!=row['expected_bytes']:raise Refusal('size_mismatch')
        while n<=row['expected_bytes']:
            if state:check(state)
            chunk=stream.read(min(65536,row['expected_bytes']+1-n))
            if not chunk:break
            n+=len(chunk);a.update(chunk);b.update(chunk)
        end=os.fstat(stream.fileno())
    if (n!=row['expected_bytes'] or (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns)
        or a.hexdigest()!=row['sha256'] or b.hexdigest()!=row['md5']):raise Refusal('full_body_fixity_mismatch')
    return {'bytes':n,'sha256':a.hexdigest(),'md5':b.hexdigest()}

def transfer(row,trial,state,opener):
    directory=safe(BASE/'objects'/row['source_id']);directory.mkdir(parents=True,exist_ok=True)
    partial=safe(directory/'body.partial');final=safe(directory/Path(row['mirror_path']).name)
    offset=partial.stat().st_size if partial.exists() else 0
    if final.exists():return {'status':'existing_verified',**verify(final,row,state),'path':str(final.relative_to(ROOT))}
    if offset>row['expected_bytes']:raise Refusal('partial_over_limit')
    if offset==row['expected_bytes']:
        result=verify(partial,row,state);os.link(partial,final);partial.unlink()
        return {'status':'recovered_verified',**result,'path':str(final.relative_to(ROOT))}
    url=row['mirror_url']
    for hop in range(1,5):
        check(state);provenance=route(row,url,initial=hop==1)
        state['requests']+=1
        headers={'Accept-Encoding':'identity','User-Agent':'RessectionLab-exact-research-mirror/1'}
        if offset:headers['Range']=f'bytes={offset}-'
        event={'request':state['requests'],'source_binding':binding(row),'range_offset':offset,**provenance}
        save(trial/f'request-{hop:02}.json',event)
        try:
            response=opener.open(Request(url,headers=headers),timeout=min(30,max(.001,state['deadline']-time.monotonic())))
        except HTTPError as error:
            event.update(status=error.code,retry_after_utc_epoch=after(error.headers.get('Retry-After')))
            destination=error.headers.get('Location');error.close()
            if destination:
                target=urljoin(url,destination)
                event['redirect_destination']=route(row,target)
            save(trial/f'response-{hop:02}.json',event)
            if event['retry_after_utc_epoch'] is not None or error.code==429:
                raise Deferred(max(event['retry_after_utc_epoch'] or 0,time.time()+60),error.code) from None
            if error.code in (301,302,303,307,308) and destination:
                url=target;continue
            if error.code in (408,500,502,503,504):raise Deferred(time.time()+30,error.code) from None
            raise Refusal('terminal_http_failure') from None
        with response:
            event.update(status=response.status,retry_after_utc_epoch=after(response.headers.get('Retry-After')))
            save(trial/f'response-{hop:02}.json',event)
            if response.geturl()!=url or response.headers.get('Content-Encoding','identity')!='identity':raise Refusal('response_contract')
            if offset:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{row["expected_bytes"]-1}/{row["expected_bytes"]}':raise Refusal('range_refused_partial_preserved')
            elif response.status!=200:raise Refusal('response_status')
            lengths=response.headers.get_all('Content-Length',[])
            if lengths and lengths!=[str(row['expected_bytes']-offset)]:raise Refusal('content_length_mismatch')
            with partial.open('ab' if offset else 'xb') as output:
                count=offset
                while count<=row['expected_bytes']:
                    check(state)
                    chunk=response.read(min(65536,row['expected_bytes']+1-count,CAP_BYTES-state['bytes_read']+1))
                    if not chunk:break
                    state['bytes_read']+=len(chunk);count+=len(chunk)
                    check(state)
                    if count>row['expected_bytes']:raise Refusal('source_byte_cap')
                    output.write(chunk);output.flush();os.fsync(output.fileno())
            verified=verify(partial,row,state)
            check(state);os.link(partial,final)
            fd=os.open(final.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            partial.unlink()
            if event['retry_after_utc_epoch'] is not None:state['not_before']=max(state['not_before'],event['retry_after_utc_epoch'])
            return {'status':'downloaded_verified',**verified,'path':str(final.relative_to(ROOT))}
    raise Refusal('redirect_hop_cap')

def execute():
    rows,retained,rights=prepare()
    source=small(Path(__file__))
    declaration={'schema':'resect-exact-mirror-acquisition-v1','created_utc':now(),'source_sha256':sha(source),
        'bindings':BINDINGS,'rows':rows,'mirror_is_third_party_transport_only':True,'rights':rights,
        'caps':{'bytes':CAP_BYTES,'seconds':CAP_SECONDS,'requests':CAP_REQUESTS,'workers':1,'per_file_attempts':3},
        'url_provenance':'host/path/full-url-SHA256/query-key-names; transient signed query values are never retained',
        'tls':{'verification':True,'custom_ca':False,'openssl':ssl.OPENSSL_VERSION},**CLAIMS}
    save(BASE/'declaration.json',declaration)
    preserve(BASE/'source-snapshot.py',source)
    for name,raw in retained.items():preserve(BASE/'references'/name,raw)
    state={'deadline':time.monotonic()+CAP_SECONDS,'bytes_read':0,'requests':0,'not_before':0}
    started=time.monotonic();counts={r['source_id']:0 for r in rows};outcomes={};pending=list(rows)
    opener=build_opener(ProxyHandler({}),NoRedirect(),HTTPSHandler(context=ssl.create_default_context()))
    def stop(signum,frame):raise Deadline('aggregate_deadline_or_stop')
    old_alarm=signal.signal(signal.SIGALRM,stop);old_term=signal.signal(signal.SIGTERM,stop);signal.alarm(CAP_SECONDS)
    summary={'status':'incomplete','files':24,'expected_bytes':1561749,'declaration_sha256':sha(enc(declaration)),**CLAIMS}
    try:
        while pending:
            check(state)
            if state['not_before']>time.time():
                time.sleep(min(1.,state['not_before']-time.time()));continue
            row=pending.pop(0);key=row['source_id'];counts[key]+=1
            trial=safe(BASE/'objects'/key/f'attempt-{counts[key]:02}');trial.mkdir(parents=True,exist_ok=False)
            save(trial/'intent.json',{'source_binding':binding(row),'declaration_sha256':summary['declaration_sha256'],'started_utc':now(),'attempt':counts[key]})
            receipt={'source_id':key,'source_binding':binding(row),'status':'failed','source':row,
                     'declaration_sha256':summary['declaration_sha256'],'started_utc':now(),**CLAIMS}
            try:
                receipt.update(transfer(row,trial,state,opener))
            except Deferred as error:
                receipt.update(status='rate_or_transport_deferred',http_status=error.status,not_before_utc_epoch=error.until)
                state['not_before']=max(state['not_before'],error.until)
                if counts[key]<3:pending.append(row)
            except (TimeoutError,URLError,ConnectionError) as error:
                reason=getattr(error,'reason',error)
                if isinstance(reason,ssl.SSLError):receipt['status']='tls_failed'
                else:
                    receipt['status']='transport_deferred';state['not_before']=time.time()+30
                    if counts[key]<3:pending.append(row)
                receipt['error_type']=type(error).__name__
            except Refusal as error:receipt.update(status='refused',error_code=str(error))
            finally:
                partial=BASE/'objects'/key/'body.partial'
                receipt.update(finished_utc=now(),retained_partial_bytes=partial.stat().st_size if partial.exists() else 0)
                save(trial/'receipt.json',receipt);outcomes[key]={'receipt':str((trial/'receipt.json').relative_to(ROOT)),'receipt_sha256':sha(enc(receipt)),'status':receipt['status']}
                print(json.dumps({'source_id':key,'status':receipt['status'],'verified_files':sum(x['status'].endswith('_verified') for x in outcomes.values()),'bytes_read':state['bytes_read']}),flush=True)
        summary['status']='all_24_bodies_verified' if len(outcomes)==24 and all(v['status'].endswith('_verified') for v in outcomes.values()) else 'completed_with_unresolved_files'
    except (Exception,KeyboardInterrupt) as error:summary.update(status='stopped_incomplete',error_type=type(error).__name__)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
        summary.update(finished_utc=now(),elapsed_seconds=time.monotonic()-started,bytes_read=state['bytes_read'],requests=state['requests'],outcomes=outcomes,pending=[r['source_id'] for r in pending])
        if summary['elapsed_seconds']>=CAP_SECONDS:summary['status']='stopped_deadline'
        save(BASE/'summary.json',summary);print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='all_24_bodies_verified' else 2

if __name__=='__main__':
    try:raise SystemExit(execute())
    except Exception as error:
        print(json.dumps({'status':'refused','error_type':type(error).__name__,'code':str(error) if isinstance(error,Refusal) else 'setup_failed'}),flush=True)
        raise SystemExit(2)
