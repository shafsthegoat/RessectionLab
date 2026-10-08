"""Bounded TRAIN metadata qualification. Never GETs a scientific S3 object."""
from __future__ import annotations
import concurrent.futures, datetime, hashlib, json, os, re, signal, subprocess, sys
import threading, time, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INPUT = ROOT/'build/lausanne-annotation-expansion-audit-v1/train-candidate-inventory.json'
INPUT_SHA = '146c92904409c9a0dfdbe452352b4fd83cef366b40ea9e14d942cfe07523591d'
COHORT = ROOT/'manifests/lausanne-component-cohort-v1.json'
COHORT_SHA = '891a53680edfce604eed9a1791bc790efa615db8ce54da0c92de834a6b87d49a'
ORIGINALS = ROOT/'manifests/lausanne-train-originals-v1.json'
ORIGINALS_SHA = '6f1fc7812af0d66550076aa08d37d7f36f08d764fdab701bbbc9ca0608629e66'
COMMIT = '896b8846d899acee68c0246cc987ca96e77267d4'
RAW = 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds003949/'+COMMIT+'/'
S3 = 'https://s3.amazonaws.com/openneuro.org/'
LIMIT_REQUESTS, LIMIT_BODY, LIMIT_SECONDS = 300, 8*1024*1024, 600

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def encoded(x): return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def atomic(path, b):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: f.write(b)
    os.replace(tmp,path)
def record(path, value): atomic(path,encoded(value))
def checked(path, expected):
    b=path.read_bytes()
    if sha(b)!=expected: raise ValueError('Source input hash changed: '+str(path))
    return json.loads(b)

def inputs():
    declaration=checked(INPUT,INPUT_SHA)
    members={r['subject']:r for r in checked(COHORT,COHORT_SHA)['members']}
    originals={(r['subject'],r['session']):r for r in checked(ORIGINALS,ORIGINALS_SHA)['sessions']}
    rows=declaration['candidates']
    assert len(rows)==148 and len({r['path'] for r in rows})==148
    for row in rows:
        m=members[row['subject']]
        assert row['role']==m['role']=='TRAIN' and row['session'][4:] in m['sessions']
        assert (row['subject'],row['session']) in originals
        path=row['path']
        expected='derivatives/manual_masks/{s}/{t}/anat/{s}_{t}_desc-Lesion_'.format(s=row['subject'],t=row['session'])
        assert path.startswith(expected) and re.fullmatch(r'[0-9]+_mask\.nii\.gz',path[len(expected):])
        assert row['pointer_url']==RAW+path
        assert row['sidecar_url']==RAW+path.removesuffix('.nii.gz')+'.json'
        assert row['s3_current_url']==S3+'ds003949/'+path
        assert re.fullmatch('[0-9a-f]{40}',row['git_pointer_blob_sha1'])
    return rows,originals

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

class Runner:
    def __init__(self, rows, originals):
        self.rows,self.originals=rows,originals
        self.started=time.monotonic();self.lock=threading.Lock()
        self.requests=0;self.body_bytes=0;self.reserved_body=0
        self.get_urls={r[k] for r in rows for k in ('pointer_url','sidecar_url')}
        self.head_urls={r['s3_current_url'] for r in rows}
        self.records=[];self.versions={};self.listing_receipts=[]
        self.stop_reason=None
        (OUT/'requests').mkdir();(OUT/'files').mkdir()
    def remaining(self): return 580-(time.monotonic()-self.started)
    def fetch(self,method,url,limit,kind):
        parsed=urllib.parse.urlsplit(url)
        listing=(method=='GET' and parsed.scheme=='https' and parsed.netloc=='s3.amazonaws.com'
                 and parsed.path=='/openneuro.org' and 'versions' in urllib.parse.parse_qs(parsed.query,keep_blank_values=True))
        assert (method=='GET' and url in self.get_urls) or (method=='HEAD' and url in self.head_urls) or listing
        if listing:
            query=urllib.parse.parse_qs(parsed.query,keep_blank_values=True)
            assert query.get('prefix')==['ds003949/derivatives/manual_masks/']
            assert set(query)<={'versions','prefix','max-keys','key-marker','version-id-marker'}
        reserve=limit+1 if method=='GET' else 0
        with self.lock:
            if self.requests>=LIMIT_REQUESTS or self.body_bytes+self.reserved_body+reserve>LIMIT_BODY or self.remaining()<2:
                raise RuntimeError('Declared request/body/time budget reached')
            self.requests+=1;number=self.requests;self.reserved_body+=reserve
        tag='request-%03d'%number
        receipt={'number':number,'method':method,'url':url,'kind':kind,'started_at':now(),
                 'body_limit':limit,'scientific_payload_request':False,'automatic_retries':0}
        record(OUT/'requests'/(tag+'-started.json'),receipt)
        b=b'';chunks=[];started=time.monotonic();response=None
        try:
            opener=urllib.request.build_opener(NoRedirect())
            request=urllib.request.Request(url,method=method,headers={'User-Agent':'RessectionLab-metadata-qualification/1'})
            try:
                response=opener.open(request,timeout=min(15,max(1,self.remaining())))
            except urllib.error.HTTPError as error:
                response=error
            receipt.update(status=response.status,final_url=response.url,
                           response_headers=dict(response.headers))
            assert response.url==url, 'Redirect or alternate endpoint not allowed'
            if method=='GET':
                read=0
                while read<limit+1:
                    if self.remaining()<=0: raise TimeoutError('Aggregate metadata deadline')
                    try:
                        chunk=response.read(min(65536,limit+1-read))
                    except Exception as error:
                        partial=getattr(error,'partial',b'')
                        if isinstance(partial,bytes): chunks.append(partial)
                        raise
                    if not chunk: break
                    chunks.append(chunk);read+=len(chunk)
                b=b''.join(chunks)
                if len(b)>limit: raise ValueError('Response exceeds metadata body bound')
            if response.status!=200: raise ValueError('HTTP status '+str(response.status))
            receipt['ok']=True
        except Exception as error:
            receipt.update(ok=False,error={'type':type(error).__name__,'message':str(error)})
        finally:
            if method=='GET': b=b''.join(chunks)
            if response is not None: response.close()
            with self.lock:
                self.reserved_body-=reserve;self.body_bytes+=len(b)
            body_path=OUT/'requests'/(tag+'-body.bin')
            atomic(body_path,b)
            receipt.update(completed_at=now(),elapsed_seconds=time.monotonic()-started,
                           body={'path':str(body_path),'bytes':len(b),'sha256':sha(b)})
            record(OUT/'requests'/(tag+'.json'),receipt)
        return b,receipt
    def list_versions(self):
        params={'versions':'','prefix':'ds003949/derivatives/manual_masks/','max-keys':'1000'}
        wanted={'ds003949/'+r['path'] for r in self.rows}
        complete=False
        for page in range(4):
            url='https://s3.amazonaws.com/openneuro.org?'+urllib.parse.urlencode(params)
            b,receipt=self.fetch('GET',url,2*1024*1024,'official_S3_version_inventory')
            self.listing_receipts.append(receipt)
            if not receipt['ok']: break
            try:
                root=ET.fromstring(b);ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
                for node in root.findall('s:Version',ns):
                    item={x.tag.split('}')[-1]:x.text for x in node}
                    if item.get('Key') not in wanted: continue
                    item['listing_request']=receipt['number']
                    self.versions.setdefault(item['Key'],[]).append(item)
                truncated=root.findtext('s:IsTruncated',namespaces=ns)
                if truncated=='false': complete=True;break
                if truncated!='true': raise ValueError('Missing pagination status')
                key=root.findtext('s:NextKeyMarker',namespaces=ns)
                version=root.findtext('s:NextVersionIdMarker',namespaces=ns)
                if not key: raise ValueError('Missing next key marker')
                params.update({'key-marker':key})
                if version: params['version-id-marker']=version
            except Exception as error:
                record(OUT/('version-list-page-%d-error.json'%page),{'error':str(error),'request':receipt['number']})
                break
        record(OUT/'version-inventory.json',{'complete':complete,'scope':'Only existing148 candidate paths are interpreted',
               'records':self.versions,'request_numbers':[r['number'] for r in self.listing_receipts],
               'fallback':'HEAD current exact TRAIN object only where a pinned pointer has no matching listed version'})
        return complete
    def qualify(self,row,pool):
        result={'path':row['path'],'subject':row['subject'],'session':row['session'],'role':'TRAIN',
                'status':'metadata_failed','annotation_subtype_status':row['annotation_subtype_status'],
                'source_crosswalk_entry':row['source_crosswalk_entry'],'annotation_available_at':None,
                'review_available_at':None,'scientific_payload_accessed':False,
                'scanner_frame_admitted':False,'spatial_planning_admitted':False,'training_admitted':False,
                'optimizer_updates':0,'requests':[]}
        futures=[pool.submit(self.fetch,'GET',row[k],2048,kind) for k,kind in
                 [('pointer_url','pinned_Git_annex_pointer'),('sidecar_url','pinned_same_session_sidecar')]]
        responses=[]
        for future in futures:
            try: responses.append(future.result())
            except Exception as error:
                responses.append((b'',{'ok':False,'error':{'type':type(error).__name__,'message':str(error)}}))
        result['requests']=[receipt for _,receipt in responses]
        try:
            if not all(receipt['ok'] for _,receipt in responses): raise ValueError('Metadata request failed; no retry')
            pointer,sidecar=(r[0] for r in responses)
            blob=hashlib.sha1(('blob '+str(len(pointer))+'\0').encode()+pointer).hexdigest()
            if blob!=row['git_pointer_blob_sha1']: raise ValueError('Pinned Git pointer blob SHA1 mismatch')
            target=pointer.decode('ascii')
            match=re.fullmatch(r'MD5E-s([0-9]+)--([0-9a-f]{32})\.nii\.gz',target.split('/')[-1])
            if not match: raise ValueError('Unrecognized published MD5E pointer')
            size,md5=int(match[1]),match[2]
            if size<=0: raise ValueError('Invalid published size')
            source_name=row['subject']+'_'+row['session']+'_angio.nii.gz'
            metadata=json.loads(sidecar)
            if not isinstance(metadata,dict) or any(metadata.get(k)!=v for k,v in {'Type':'Lesion','RawSources':source_name,'Space':'orig'}.items()):
                raise ValueError('Sidecar does not bind exact same-session original TOF')
            originals=self.originals[(row['subject'],row['session'])]['files']
            source=[x for x in originals if x['path'].split('/')[-1]==source_name]
            if len(source)!=1: raise ValueError('Exact original reference is not unique')
            matches=[v for v in self.versions.get('ds003949/'+row['path'],[]) if
                     int(v.get('Size','-1'))==size and v.get('ETag','').strip('"')==md5 and v.get('VersionId') not in (None,'','null')]
            if matches:
                selected=sorted(matches,key=lambda x:(x.get('LastModified',''),x['VersionId']))[-1]
                version=selected['VersionId'];version_proof={'kind':'official_S3_version_listing','selected':selected,'matching_versions':len(matches)}
            else:
                _,head=self.fetch('HEAD',row['s3_current_url'],0,'official_exact_TRAIN_object_HEAD')
                result['requests'].append(head)
                if not head['ok']: raise ValueError('Immutable version request failed; no retry')
                headers={k.lower():v for k,v in head['response_headers'].items()}
                version=headers.get('x-amz-version-id')
                if (int(headers.get('content-length','-1'))!=size or headers.get('etag','').strip('"')!=md5 or version in (None,'','null')):
                    raise ValueError('Current S3 object does not match pinned source size/MD5 and immutable version')
                version_proof={'kind':'official_S3_HEAD','request':head['number'],'version_id':version}
            result.update(status='metadata_qualified',file={'path':row['path'],'bytes':size,'expected_md5':md5,'sha256':None,
                          'source_url':row['s3_current_url']+'?versionId='+urllib.parse.quote(version,safe='')},
                          pointer_git_blob_sha1=blob,pointer_sha256=sha(pointer),sidecar_sha256=sha(sidecar),
                          sidecar=metadata,original_reference=source[0],version_proof=version_proof,
                          reference_grid_status='uninspected',annotation_content_status='uninspected',
                          semantic_scope='source_manual_lesion_region; contour status preserved separately; no negative vascular domain')
        except Exception as error:
            result['error']={'type':type(error).__name__,'message':str(error)}
        return result
    def checkpoint(self):
        done={r['path'] for r in self.records}
        qualified=[r for r in self.records if r['status']=='metadata_qualified']
        record(OUT/'qualification-index.json',{'schema':'lausanne-train-annotation-metadata-qualification-v1',
               'created_at':now(),'input_sha256':INPUT_SHA,'git_commit':COMMIT,'cohort_sha256':COHORT_SHA,
               'original_index_sha256':ORIGINALS_SHA,'candidate_count':148,'records':self.records,
               'deferred':[{'path':r['path'],'reason':self.stop_reason or 'not_yet_requested'} for r in self.rows if r['path'] not in done],
               'statistics':{'qualified_masks':len(qualified),'qualified_people':len({r['subject'] for r in qualified}),
                 'qualified_sessions':len({(r['subject'],r['session']) for r in qualified}),
                 'qualified_mask_bytes':sum(r['file']['bytes'] for r in qualified),
                 'failed_masks':len(self.records)-len(qualified),'deferred_masks':148-len(self.records),
                 'http_requests_started':self.requests,'response_body_bytes':self.body_bytes,
                 'elapsed_seconds':time.monotonic()-self.started,'scientific_gets':0,'optimizer_updates':0},
               'stop_reason':self.stop_reason,'payload_admission':False,'training_admission':False,
               'unresolved':['Per-file binary/grid QC','Review and annotation availability','Contour/date crosswalk for unresolved cases',
                             'Task-specific annotation/background domain','Brain support and scanner/planning admission'],
               'model_or_atlas_used':False})
    def run(self):
        complete=self.list_versions()
        record(OUT/'feasibility.json',{'version_listing_complete':complete,'listed_candidate_paths':len(self.versions),
               'maximum_total_requests':LIMIT_REQUESTS,'maximum_response_body_bytes':LIMIT_BODY,
               'outer_hard_seconds':LIMIT_SECONDS,'concurrency':2,'requests_before_files':self.requests,
               'estimated_requests_if_one_version_listing_covers_all':297,
               'head_fallback_note':'If version inventory is unavailable, budget admits only a bounded prefix; remaining candidates are deferred.',
               'automatic_retries':0,'scientific_payload_requests':0})
        self.checkpoint();consecutive_raw_failures=0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for row in self.rows:
                needs_head=not self.versions.get('ds003949/'+row['path'])
                if self.remaining()<20 or self.requests+2+int(needs_head)>LIMIT_REQUESTS or self.body_bytes+4098>LIMIT_BODY:
                    self.stop_reason='declared_time_request_or_body_budget';break
                result=self.qualify(row,pool)
                self.records.append(result)
                record(OUT/'files'/(row['path'].split('/')[-1]+'.metadata.json'),result)
                raw=[r for r in result['requests'] if r.get('kind') in {'pinned_Git_annex_pointer','pinned_same_session_sidecar'}]
                raw_failed=any(not r.get('ok',False) for r in raw)
                consecutive_raw_failures=consecutive_raw_failures+1 if raw_failed else 0
                if any(r.get('status') in (403,429) for r in raw): self.stop_reason='raw_service_access_or_rate_limit'
                elif consecutive_raw_failures>=3: self.stop_reason='three_consecutive_raw_metadata_file_failures'
                self.checkpoint()
                if len(self.records)%10==0: print(json.dumps({'completed_file_attempts':len(self.records),'requests':self.requests,'body_bytes':self.body_bytes,'elapsed_seconds':time.monotonic()-self.started}),flush=True)
                if self.stop_reason: break
        if self.stop_reason is None: self.stop_reason='all_candidate_metadata_attempted'
        self.checkpoint()

def main():
    rows,originals=inputs()
    if '--preflight' in sys.argv:
        print(json.dumps({'scope_valid':True,'candidates':len(rows),'source_hashes_verified':True,'network_requests':0}));return
    if '--worker' in sys.argv:
        Runner(rows,originals).run();return
    marker=OUT/'run-started.json'
    with marker.open('xb') as f: f.write(encoded({'started_at':now(),'requests_limit':300,'body_limit':LIMIT_BODY,'hard_seconds':600,'automatic_retries':0}))
    script=Path(__file__).read_bytes()
    record(OUT/'execution-provenance.json',{'script':str(Path(__file__).resolve()),'script_bytes':len(script),'script_sha256':sha(script),
           'python':sys.version,'source_inputs':[{'path':str(p),'sha256':s} for p,s in [(INPUT,INPUT_SHA),(COHORT,COHORT_SHA),(ORIGINALS,ORIGINALS_SHA)]],
           'request_policy':'GET pinned148 TRAIN Git pointer/sidecar URLs; GET official metadata version listing; HEAD exact148 TRAIN S3 objects; all redirects refused',
           'network_max_concurrency':2,'no_scientific_payload_gets':True})
    started=time.monotonic();status='completed';returncode=None
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],start_new_session=True)
    try: returncode=child.wait(timeout=596)
    except subprocess.TimeoutExpired:
        status='outer_hard_watchdog_terminated';os.killpg(child.pid,signal.SIGKILL);returncode=child.wait(timeout=1)
    index_path=OUT/'qualification-index.json'
    if index_path.exists():
        index=json.loads(index_path.read_bytes())
        if status!='completed' or returncode!=0:
            index['stop_reason']=status if status!='completed' else 'worker_failed'
            for row in index['deferred']: row['reason']=index['stop_reason']
            index['incomplete_worker_state']=True;record(index_path,index)
    record(OUT/'closure.json',{'status':status,'worker_returncode':returncode,'elapsed_seconds':time.monotonic()-started,
           'closed_at':now(),'script_sha256_unchanged':sha(Path(__file__).read_bytes())==sha(script),
           'scientific_payloads_acquired_or_decoded':0,'recorded_rl_transitions':0})
    print(json.dumps({'status':status,'returncode':returncode,'output':str(OUT),'statistics':index.get('statistics') if index_path.exists() else None}),flush=True)

if __name__=='__main__': main()
