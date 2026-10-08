"""Fetch only the three already-inspected official HTML metadata endpoints; no archives."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.parse import urlparse,urljoin
import hashlib,json,time
OUT=Path(__file__).resolve().parent
urls=[('project','https://www.nitrc.org/projects/ibsr/'),('filelist','https://www.nitrc.org/frs/?group_id=48&rel='),('readme_access','https://www.nitrc.org/frs/download.php/5720/README.txt')]
class Redirects(HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  u=urlparse(newurl)
  if u.scheme!='https' or u.netloc!='www.nitrc.org' or not (u.path.startswith('/account/login.php') or newurl in [x[1] for x in urls]):raise ValueError('Unexpected metadata redirect')
  return super().redirect_request(req,fp,code,msg,headers,newurl)
class Links(HTMLParser):
 def __init__(self):super().__init__();self.href=None;self.text='';self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':self.href=dict(attrs).get('href');self.text=''
 def handle_data(self,data):
  if self.href is not None:self.text+=data
 def handle_endtag(self,tag):
  if tag=='a' and self.href is not None:
   self.links.append({'text':self.text.strip(),'url':urljoin('https://www.nitrc.org/',self.href)});self.href=None
opener=build_opener(Redirects());receipts=[]
for name,url in urls:
 start=time.perf_counter();r={'name':name,'url':url,'started_at':datetime.now(timezone.utc).isoformat(),'method':'GET','scope':'official textual metadata only','byte_cap':524288,'automatic_retries':0}
 try:
  with opener.open(Request(url,headers={'User-Agent':'RessectionLab-source-metadata-audit/1.0','Accept':'text/html,text/plain'}),timeout=25) as response:
   r['status']=response.status;r['final_url']=response.url;r['headers']={k:v for k,v in response.headers.items() if k.lower() in ['content-type','content-length','last-modified','etag','date']}
   b=response.read(524289)
   if len(b)>524288:raise ValueError('Metadata response over cap')
   if 'html' not in response.headers.get('Content-Type','') and 'text/plain' not in response.headers.get('Content-Type',''):raise ValueError('Not allowed textual metadata')
   p=OUT/(name+'.html');p.write_bytes(b);r['body']={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
   if name=='filelist':
    parser=Links();parser.feed(b.decode('utf-8',errors='strict'))
    r['published_download_links_not_requested']=[x for x in parser.links if x['text'] in ['IBSR_V2.0_nifti_stripped.tgz','IBSR_01_ANALYZE.tgz','IBSR_01parc_ANALYZE.tgz','README.txt','20Normals.README.txt']]
   r['ok']=True
 except Exception as e:r.update(ok=False,error={'type':type(e).__name__,'message':str(e)})
 r['elapsed_seconds']=time.perf_counter()-start;receipts.append(r)
(OUT/'metadata-requests.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k in ['name','status','final_url','body','ok','error','published_download_links_not_requested']} for r in receipts],indent=2))
