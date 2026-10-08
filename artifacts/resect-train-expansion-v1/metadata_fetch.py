"""One-shot, metadata-only audit helper; never downloads scientific payloads."""
import hashlib,json,os,re,signal,sys,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STATE=ROOT/'request-budget.json'
def sanitize(x):
 if isinstance(x,dict): return {k:sanitize(v) for k,v in x.items() if k.lower() not in {'authorization','cookie','set-cookie'}}
 if isinstance(x,list): return [sanitize(v) for v in x]
 if isinstance(x,str) and x.startswith(('https://','http://')):
  p=urllib.parse.urlsplit(x); q=urllib.parse.parse_qsl(p.query,keep_blank_values=True)
  keep=[(k,v) for k,v in q if k in {'revision','page[size]','page','page[number]','format'}]
  return urllib.parse.urlunsplit((p.scheme,p.netloc,p.path,urllib.parse.urlencode(keep),''))
 return x
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k): return None
class Deadline(Exception): pass
def alarm(*args): raise Deadline()
def fetch(url):
 p=urllib.parse.urlsplit(url)
 if p.scheme!='https' or p.netloc!='api.osf.io' or not p.path.startswith('/v2/') or 'download' in p.path or any(k not in {'page[size]','page','page[number]','format'} for k,v in urllib.parse.parse_qsl(p.query)):
  raise ValueError('metadata endpoint outside scope')
 state=json.loads(STATE.read_text()) if STATE.exists() else {'scope':'public OSF JSON metadata only; no scientific bodies; no retry','started_unix':time.time(),'maximum_requests':30,'maximum_wall_seconds':300,'maximum_response_bytes':4194304,'requests':0,'response_bytes':0,'queried':[]}
 if url in state['queried']: raise ValueError('no retry')
 left=300-(time.time()-state['started_unix']); cap=4194304-state['response_bytes']
 if state['requests']>=30 or left<=0 or cap<=0: raise ValueError('budget exhausted')
 state['requests']+=1; state['queried'].append(url); STATE.write_text(json.dumps(state,indent=2)+'\n')
 n=state['requests']; receipt={'request':n,'url':sanitize(url),'started_unix':time.time(),'http_status':None,'response_bytes':0,'outcome':'failed'}
 raw=bytearray(); body=None
 try:
  signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,min(20,left))
  request=urllib.request.Request(url,headers={'Accept':'application/json','Accept-Encoding':'identity','User-Agent':'RessectionLab-metadata-audit/1'})
  with urllib.request.build_opener(NoRedirect).open(request,timeout=min(15,left)) as response:
   receipt['http_status']=response.status
   if response.headers.get_content_type() not in {'application/json','application/vnd.api+json'}: raise ValueError('non JSON response refused')
   length=response.headers.get('Content-Length')
   if length and int(length)>cap: raise ValueError('response exceeds remaining byte budget')
   while len(raw)<cap:
    part=response.read(min(16384,cap-len(raw)))
    if not part: break
    raw.extend(part)
   else: raise ValueError('response reached remaining byte budget')
  body=json.loads(raw); receipt['outcome']='metadata_retained'; receipt['response_sha256']=hashlib.sha256(raw).hexdigest()
 except Exception as error:
  receipt['failure_class']=type(error).__name__
  if isinstance(error,urllib.error.HTTPError): receipt['http_status']=error.code
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  receipt['elapsed_seconds']=round(time.time()-receipt['started_unix'],6); receipt['response_bytes']=len(raw)
  state['response_bytes']+=len(raw); STATE.write_text(json.dumps(state,indent=2)+'\n')
  if body is not None:
   bp=ROOT/f'request-{n:02d}-metadata.json'; bp.write_text(json.dumps(sanitize(body),indent=2)+'\n'); receipt['sanitized_metadata_file']=bp.name; receipt['sanitized_sha256']=hashlib.sha256(bp.read_bytes()).hexdigest()
  (ROOT/f'request-{n:02d}-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
 return body
if __name__=='__main__':
 body=fetch(sys.argv[1])
 if body:
  data=body.get('data',[]); data=data if isinstance(data,list) else [data]
  for item in data:
   a=item.get('attributes',{}); print(json.dumps({'id':item.get('id'),'name':a.get('name'),'kind':a.get('kind'),'size':a.get('size'),'extra':a.get('extra'),'relationships':item.get('relationships'),'links':sanitize(item.get('links',{}))}))
  print('pagination',body.get('links'))
