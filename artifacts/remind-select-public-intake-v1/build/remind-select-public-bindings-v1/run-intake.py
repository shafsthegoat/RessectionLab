"""One-worker byte-only SELECT public intake using the existing unchanged transport."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import build_opener, HTTPSHandler
from urllib.error import HTTPError, URLError
import argparse, errno, fcntl, hashlib, http.client, json, os, re, shutil, socket, ssl, sys, time, traceback, uuid

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/acquisition/remind-select-public-v1'
PEOPLE={'ReMIND-013','ReMIND-037'}
SERIES={'79cb63bb-6c98-4d94-bd4b-c8f736e71ed8','30b3b7f5-a542-4fb2-9744-f7e91359d9bb','6aecb7da-0b67-4765-9811-213b3c983b22','d5fd818e-ba97-4115-8dfb-eca2aa919b89','71000b63-56e3-42c3-977a-4389fb87046a','88911f75-9082-427c-8cd6-9784aecf46c0'}
def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1024**2): h.update(block)
 return h.hexdigest()
def emit(value):
 try: print(json.dumps(value),flush=True)
 except BrokenPipeError: pass
def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--declaration-sha256',required=True); parser.add_argument('--check-only',action='store_true'); args=parser.parse_args()
 assert sha(BASE/'declaration.json')==args.declaration_sha256
 D=json.loads((BASE/'declaration.json').read_text()); assert D['runner_sha256']==sha(Path(__file__)) and D['execution_released'] is True
 assert D['max_workers']==1 and D['max_transport_attempts_per_object_across_restarts']==3
 assert D['free_space_reserve_bytes']==100*1024**3 and D['active_outputs_allowance_bytes']==64*1024**3 and D['max_payload_bytes']==100*1024**2
 for p,h in D['metadata_pins'].items(): assert sha(ROOT/p)==h
 for name,h in D['helper_sha256'].items(): assert sha(BASE/'source'/name)==h
 assert D['manifest_path']=='data/acquisition/remind-select-public-v1/source/exact-object-manifest.json'
 assert sha(ROOT/D['manifest_path'])==D['manifest_sha256']
 M=json.loads((ROOT/D['manifest_path']).read_text()); entries=M['objects']
 assert M['status']=='all_six_public_series_listed' and len(entries)==D['objects']==372
 assert {m['subject'] for m in M['members']}==PEOPLE and all(m['role']=='SELECT' for m in M['members'])
 assert {e['series_uuid'] for e in entries}==SERIES and {e['patient_id'] for e in entries}==PEOPLE
 for e in entries:
  assert e['role']=='SELECT' and e['kind'] in {'structural_t1ce','whole_tumor','cerebrum'}
  assert re.fullmatch(re.escape(e['series_uuid'])+r'/[0-9a-f-]{36}\.dcm',e['key'])
  assert e['path']==e['patient_id']+'/'+e['key'] and e['source_url']=='https://idc-open-data.s3.amazonaws.com/'+e['key']
  assert re.fullmatch('[a-f0-9]{32}',e['expected_md5']) and e['etag_opaque']=='"'+e['expected_md5']+'"'
  assert e['size_bytes']==e['bytes'] and e['bytes']>0 and e['sha256'] is None
 assert len({e['path'] for e in entries})==372 and sum(e['bytes'] for e in entries)==M['exact_total_bytes']==D['exact_source_bytes']<D['max_payload_bytes']
 for proof in M['listing_proofs']: assert sha(ROOT/proof['path'])==proof['sha256'] and proof['tls_verified']
 if args.check_only:
  emit({'status':'contract_valid','objects':372,'bytes':D['exact_source_bytes'],'network_requests':0,'payload_reads':0}); return
 if os.getpgrp()!=os.getpid(): os.setsid()
 lock=(BASE/'queue.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 reserve=D['free_space_reserve_bytes']+D['active_outputs_allowance_bytes']; assert shutil.disk_usage(BASE).free-D['exact_source_bytes']>=reserve
 sys.dont_write_bytecode=True; sys.path.insert(0,str(BASE/'source'))
 import acquire_public_case as transport
 import acquire_btc_case as guards
 import real_intake_io as io
 for module in [transport,guards,io]: assert Path(module.__file__).resolve().parent==BASE/'source'
 transport.verify_file=guards.verify_file
 def save(p,value): io.atomic_preserve(p,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())
 ctx=ssl.create_default_context(); assert ctx.check_hostname and ctx.verify_mode==ssl.CERT_REQUIRED
 runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]; run=BASE/'runs'/runid; run.mkdir(parents=True)
 out=BASE/'verified'; out.mkdir(exist_ok=True)
 def verified(p,e):
  digest=io.verify_source_file(p,e); st=p.stat()
  return {'path':e['path'],'patient_id':e['patient_id'],'role':'SELECT','series_uuid':e['series_uuid'],'kind':e['kind'],'bytes':st.st_size,'sha256':digest,'source_checksum_type':'single_part_S3_ETag_MD5','source_checksum':e['expected_md5'],'publication_stat':{'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns}}
 def process(e):
  if shutil.disk_usage(BASE).free-D['exact_source_bytes']<reserve: return {'path':e['path'],'status':'storage_reserve_refusal'}
  p=transport.checked_path(out,e['path']); partial=transport.checked_path(out,e['path']+'.partial')
  keyhash=hashlib.sha256((e['source_url']+'|'+str(e['bytes'])+'|'+e['expected_md5']).encode()).hexdigest(); journal=BASE/'attempts'/keyhash; journal.mkdir(parents=True,exist_ok=True)
  old=sorted(journal.glob('*-intent.json')); prior=[json.loads(x.read_text()) for x in journal.glob('*-result.json')]
  if len(old)!=len(prior): return {'path':e['path'],'status':'unclosed_attempt_refusal'}
  if p.exists():
   try: return {'status':'existing_byte_verified','http_requests':0,**verified(p,e)}
   except Exception as err: return {'path':e['path'],'status':'existing_integrity_refusal','error':str(err),'traceback':traceback.format_exc()}
  if any(r['status']=='integrity_or_scope_refusal' for r in prior): return {'path':e['path'],'status':'prior_integrity_or_scope_refusal'}
  if len(old)>=3: return {'path':e['path'],'status':'transport_exhausted','attempts_before_run':len(old)}
  for attempt in range(len(old)+1,4):
   aid=f'{attempt:02d}-{runid}'; offset=partial.stat().st_size if partial.exists() else 0; events=[]
   save(journal/(aid+'-intent.json'),{'entry':e,'started_utc':utc(),'attempt':attempt,'resume_offset':offset,'declaration_sha256':args.declaration_sha256,'tls_verified':True})
   opener=build_opener(guards.RejectRedirects(),HTTPSHandler(context=ctx))
   def observed(request,*,timeout):
    assert request.full_url==e['source_url']; request.add_header('If-Match',e['etag_opaque'])
    event={'requested_utc':utc(),'url':request.full_url,'range':request.get_header('Range'),'tls_verified':True}
    try: response=opener.open(request,timeout=timeout)
    except Exception as err:
     event.update(error_type=type(err).__name__,error=str(err));
     if isinstance(err,HTTPError): event['status']=err.code
     events.append(event); save(journal/(aid+'-http.json'),event); raise
    h=response.headers; event.update(status=response.status,effective_url=response.geturl(),headers={k:h.get(k) for k in ['Content-Length','Content-Range','ETag','Content-Encoding']}); events.append(event); save(journal/(aid+'-http.json'),event)
    valid=response.geturl()==e['source_url'] and response.status==(206 if offset else 200) and h.get('ETag')==e['etag_opaque'] and int(h.get('Content-Length','-1'))==e['bytes']-offset and h.get('Content-Encoding','identity')=='identity'
    if offset: valid=valid and h.get('Content-Range')==f'bytes {offset}-{e["bytes"]-1}/{e["bytes"]}'
    if not valid: response.close(); raise transport.AcquisitionError('source_identity_or_resume_contract_failed')
    return response
   try:
    state=transport.acquire_file(e,out,opener=observed)
    result={'status':'byte_verified','transport_status':state,'attempt':attempt,'resume_offset':offset,'http_requests':len(events),'finished_utc':utc(),**verified(p,e)}
    save(journal/(aid+'-result.json'),result); return result
   except Exception as err:
    network_reason=isinstance(err,URLError) and isinstance(err.reason,OSError) and not isinstance(err.reason,ssl.SSLError) and (isinstance(err.reason,(TimeoutError,ConnectionResetError,ConnectionRefusedError,socket.gaierror)) or err.reason.errno in {errno.ENETUNREACH,errno.EHOSTUNREACH,errno.ETIMEDOUT})
    transient=(isinstance(err,HTTPError) and err.code in {429,500,502,503,504}) or isinstance(err,(TimeoutError,ConnectionResetError,http.client.IncompleteRead)) or network_reason
    truncated=isinstance(err,transport.AcquisitionError) and str(err).startswith('Pending source size/checksum contract mismatch:') and partial.exists() and partial.stat().st_size<e['bytes'] and bool(events) and events[-1].get('status') in {200,206}
    transient=(transient or truncated) and not isinstance(err,ssl.SSLError)
    result={'path':e['path'],'patient_id':e['patient_id'],'status':('transport_deferred' if attempt<3 else 'transport_exhausted') if transient else 'integrity_or_scope_refusal','attempt':attempt,'error_type':type(err).__name__,'error':str(err),'traceback':traceback.format_exc(),'phase':'transport_or_atomic_verification','partial_bytes':partial.stat().st_size if partial.exists() else 0,'http_requests':len(events),'finished_utc':utc()}
    save(journal/(aid+'-result.json'),result)
    if not transient or attempt==3: return result
    time.sleep(5 if attempt==1 else 15)
 started={'run_id':runid,'started_utc':utc(),'pid':os.getpid(),'pgid':os.getpgrp(),'declaration_sha256':args.declaration_sha256,'runner_sha256':sha(Path(__file__)),'objects':372,'bytes':D['exact_source_bytes'],'max_workers':1,'role':'SELECT','scope':'three public series each for013/037; no private/EVAL','storage_before':shutil.disk_usage(BASE)._asdict(),'decoded_array_bytes':0}
 save(run/'started.json',started); emit({'event':'started',**started}); results=[]
 with io.termination_cleanup():
  for e in entries:
   result=process(e); results.append(result)
   if len(results)%25==0 or result['status'] not in {'byte_verified','existing_byte_verified'}:
    progress={'event':'progress','finished':len(results),'total':372,'verified':sum(r['status'] in {'byte_verified','existing_byte_verified'} for r in results),'verified_bytes':sum(r.get('bytes',0) for r in results),'last_status':result['status'],'utc':utc()}; save(run/f'progress-{len(results):04d}.json',progress); emit(progress)
 good=[r for r in results if r['status'] in {'byte_verified','existing_byte_verified'}]
 completion={'status':'all_bytes_verified' if len(good)==372 else 'completed_with_unresolved_files','run_id':runid,'finished_utc':utc(),'declaration_sha256':args.declaration_sha256,'manifest_sha256':D['manifest_sha256'],'verified_files':len(good),'verified_bytes':sum(r['bytes'] for r in good),'unresolved_files':372-len(good),'role':'SELECT','patients':sorted(PEOPLE),'all_payloads_unreviewed':True,'header_qc':'not_run','anatomy_qc':'not_run','training_admitted':False,'private_or_eval_access':False,'decoded_array_bytes':0,'storage_after':shutil.disk_usage(BASE)._asdict(),'files':results}
 save(run/'completion.json',completion)
 if len(good)==372: save(BASE/'completion.json',completion)
 emit({'event':'terminal','status':completion['status'],'verified_files':len(good),'verified_bytes':completion['verified_bytes'],'receipt':str(run/'completion.json')})
if __name__=='__main__': main()
