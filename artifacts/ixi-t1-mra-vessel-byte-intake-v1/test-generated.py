"""Offline generated controls for the exact IXI adapter and preserved helpers."""
from pathlib import Path
from collections import namedtuple
import contextlib,hashlib,io,json,runpy,shutil,ssl,sys,tempfile
R=Path.cwd();P=R/'build/ixi-paired-intake-preparation-v1';OUT=P/'generated-controls';OUT.mkdir(exist_ok=True)
RUNNER=P/'run-intake.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();D=json.loads((P/'prepared-declaration.json').read_text());BODY=b'control!';results=[]
class Response(io.BytesIO):
 def __init__(self,body,status,headers,url):super().__init__(body);self.status=status;self.headers=headers;self.url=url
 def geturl(self):return self.url
cases=['fresh','label_no_etag','strict_resume','label_resume_no_etag','ignored_range','wrong_md5','wrong_etag','wrong_length','wrong_content_range','wrong_encoding','wrong_effective_url','three_attempt_exhaustion','tls_failure','orphaned_attempt','truncated_then_resume','existing_immutable','fully_received_partial']
for case in cases:
 root=Path(tempfile.mkdtemp(prefix=case+'-',dir=OUT));base=root/'data/acquisition/ixi-t1-mra-vessel-v1';(base/'source').mkdir(parents=True)
 for name in ['acquire_public_case.py','acquire_btc_case.py','real_intake_io.py']:shutil.copyfile(R/'data/acquisition/ixi-t1-mra-vessel-v1/source'/name,base/'source'/name)
 entry={'path':'generated.tar','size_bytes':8,'bytes':8,'sha256':None,'source_url':'https://example.invalid/generated','expected_md5':hashlib.md5(BODY).hexdigest(),'etag_opaque':None if 'no_etag' in case else '"opaque-publisher-id"','checksum_authority':'generated-only'}
 concurrent=root/'concurrent.json';concurrent.write_text('{"objects":[]}');d={**D,'scope_commit':'generated-only','concurrent_Tracto':{'manifest_path':'concurrent.json','manifest_sha256':sha(concurrent),'base_path':'data/concurrent'}}
 path=base/'verified'/entry['path'];partial=path.with_name(path.name+'.partial')
 if case in ['strict_resume','label_resume_no_etag','ignored_range','fully_received_partial','existing_immutable','wrong_content_range']:
  path.parent.mkdir(parents=True);(path if case=='existing_immutable' else partial).write_bytes(BODY if case in ['fully_received_partial','existing_immutable'] else BODY[:3])
 if case=='orphaned_attempt':
  kh=hashlib.sha256((entry['source_url']+'|8|'+entry['expected_md5']).encode()).hexdigest();j=base/'attempts'/kh[:2]/kh;j.mkdir(parents=True);(j/'01-old-intent.json').write_text('{}\n')
 calls=[]
 class Opener:
  def open(self,request,timeout):
   assert request.full_url==entry['source_url'] and request.get_header('If-match')==entry['etag_opaque'];calls.append(request.get_header('Range'))
   if case=='three_attempt_exhaustion':raise TimeoutError('generated timeout')
   if case=='tls_failure':raise ssl.SSLCertVerificationError('generated invalid certificate')
   offset=int(request.get_header('Range').split('=')[1].split('-')[0]) if request.get_header('Range') else 0
   if case=='ignored_range':offset=0
   body=BODY[offset:];headers={'Content-Length':str(len(body)),'Content-Encoding':'identity'}
   if entry['etag_opaque']:headers['ETag']=entry['etag_opaque']
   if offset:headers['Content-Range']=f'bytes {offset}-7/8'
   if case=='wrong_md5':body=b'badbytes'
   if case=='wrong_etag':headers['ETag']='"wrong"'
   if case=='wrong_length':headers['Content-Length']='9'
   if case=='wrong_content_range':headers['Content-Range']='bytes 2-7/8'
   if case=='wrong_encoding':headers['Content-Encoding']='gzip'
   if case=='truncated_then_resume' and len(calls)==1:body=BODY[:3]
   return Response(body,206 if offset else 200,headers,entry['source_url'] if case!='wrong_effective_url' else 'https://wrong.invalid/')
 def once():
  for name in ['acquire_public_case','acquire_btc_case','real_intake_io']:sys.modules.pop(name,None)
  ns=runpy.run_path(str(RUNNER),run_name='generated_fixture');g=ns['execute'].__globals__;g['build_opener']=lambda *a,**kw:Opener();sleep=g['time'].sleep;g['time'].sleep=lambda v:None
  try:
   with contextlib.redirect_stdout(io.StringIO()):return ns['execute'](root,base,d,[entry],'generated-declaration')
  finally:g['time'].sleep=sleep
 result=once();r=result['files'][0]
 if case in ['fresh','label_no_etag','strict_resume','label_resume_no_etag','truncated_then_resume','existing_immutable','fully_received_partial']:
  assert result['status']=='all_bytes_verified' and path.read_bytes()==BODY and not partial.exists()
  assert len(calls)==(2 if case=='truncated_then_resume' else 0 if case in ['existing_immutable','fully_received_partial'] else 1)
  if case in ['strict_resume','label_resume_no_etag']:assert calls==['bytes=3-'] and r['resume_offset']==3
  if case=='truncated_then_resume':assert calls==[None,'bytes=3-']
  old=(base/'completion.json').read_bytes();again=once();assert again['status']=='all_bytes_verified' and (base/'completion.json').read_bytes()==old
 elif case=='orphaned_attempt':assert r['status']=='orphaned_attempt_requires_reconciliation_no_network' and not calls
 elif case=='three_attempt_exhaustion':
  assert len(calls)==3 and r['status']=='transport_exhausted' and r['opener_calls_this_run']==3;assert sum(json.loads(p.read_text())['http_requests'] for p in (base/'attempts').rglob('*-result.json'))==3;assert len(list((base/'attempts').rglob('*-http.json')))==3;history={str(p.relative_to(base)):p.read_bytes() for p in (base/'attempts').rglob('*.json')};again=once();assert len(calls)==3 and again['files'][0]['status']=='transport_exhausted';assert history=={str(p.relative_to(base)):p.read_bytes() for p in (base/'attempts').rglob('*.json')}
 else:
  assert r['status']=='integrity_or_scope_refusal' and len(calls)==1 and not path.exists()
  assert r['http_requests']==r['opener_calls_this_run']==1
  if case=='tls_failure':
   events=[json.loads(p.read_text()) for p in (base/'attempts').rglob('*-http.json')];assert len(events)==1 and events[0]['error_type']=='SSLCertVerificationError' and 'status' not in events[0]
  if case in ['ignored_range','wrong_content_range']:assert partial.read_bytes()==BODY[:3]
  before=len(calls);again=once();assert len(calls)==before and again['files'][0]['status']=='prior_integrity_or_scope_refusal_no_network'
 results.append({'case':case,'status':'pass','fake_open_calls':len(calls),'actual_network_requests':0})
# Simultaneous acquisition reservations include bytes still absent from Tracto.
ns=runpy.run_path(str(RUNNER),run_name='budget_fixture');g=ns['RemainingBudget'].__init__.__globals__;Disk=namedtuple('Disk','total used free');original=g['shutil'].disk_usage
root=Path(tempfile.mkdtemp(prefix='budget-',dir=OUT));free=[500*1024**3];other=[250*1024**3];g['shutil'].disk_usage=lambda p:Disk(10**12,0,free[0]);entries=[{'path':'one','bytes':15*1024**3},{'path':'two','bytes':5*1024**3}]
try:
 b=ns['RemainingBudget'](root,root,entries,lambda o,p:o/p,100*1024**3,64*1024**3,lambda:other[0]);assert b.check()
 free[0]-=150*1024**3;other[0]-=150*1024**3;assert b.check()
 free[0]-=15*1024**3;b.terminal(entries[0]);assert b.remaining==5*1024**3 and b.check()
 free[0]=268*1024**3;assert not b.check()
 results.append({'case':'both_queues_plus_output_allowance_budget','status':'pass','actual_network_requests':0})
finally:g['shutil'].disk_usage=original
root=Path(tempfile.mkdtemp(prefix='stat-',dir=OUT));out=root/'data/concurrent/verified';out.mkdir(parents=True);m=root/'manifest.json';m.write_text(json.dumps({'objects':[{'path':'one','bytes':8},{'path':'two','bytes':10}]}));binding={'manifest_path':'manifest.json','manifest_sha256':sha(m),'base_path':'data/concurrent'}
assert ns['concurrent_remaining'](root,binding)==18;(out/'one.partial').write_bytes(b'abc');assert ns['concurrent_remaining'](root,binding)==15;(out/'one.partial').rename(out/'one');assert ns['concurrent_remaining'](root,binding)==15;(out/'two').write_bytes(b'0123456789');assert ns['concurrent_remaining'](root,binding)==5;(out/'two').unlink();assert ns['concurrent_remaining'](root,binding)==15
results.append({'case':'concurrent_source_stat_accounting_partial_complete_deleted','status':'pass','actual_network_requests':0})
receipt={'schema':'ixi-generated-transport-controls-v1','runner_sha256':sha(RUNNER),'prepared_declaration_sha256':sha(P/'prepared-declaration.json'),'status':'pass','controls':results,'boundary':'Generated eight-byte files, fake opener and isolated ignored roots. Exact execute function and unchanged helpers; no source payloads, decoding, model or network.'}
(P/'generated-controls.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'pass','controls':len(results),'receipt':str(P/'generated-controls.json')}))
