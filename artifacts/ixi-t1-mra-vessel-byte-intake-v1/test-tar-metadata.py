from pathlib import Path
import importlib.util,io,tarfile,tempfile,json,hashlib
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ixi_meta',P/'walk-tar-metadata.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def generated(body=b'X'*777,kind=tarfile.REGTYPE,name='IXI057-Guys-0000-T1.nii.gz'):
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w',format=tarfile.USTAR_FORMAT) as t:
  i=tarfile.TarInfo(name);i.size=len(body) if kind==tarfile.REGTYPE else 0;i.type=kind
  t.addfile(i,io.BytesIO(body) if i.size else None)
 return b.getvalue()
passed=[]
a=generated();seen=[]
r=m.walk(lambda o:(seen.append(o),a[o:o+512])[1],len(a));assert r[0]['size']==777 and all(o==0 or o>=1536 for o in seen);passed.append('header_offsets_skip_body_and_padding')
b=generated(b'Y'*777);assert m.walk(lambda o:a[o:o+512],len(a))==m.walk(lambda o:b[o:o+512],len(b));passed.append('body_value_invariance')
for name,buf,total in [('bad_checksum',b'Z'+a[1:],len(a)),('no_end_records',a[:1536],1536),('nonzero_trailer',a[:2560]+b'Z'+a[2561:],len(a)),('link',generated(kind=tarfile.SYMTYPE),10240),('pax',generated(kind=tarfile.XHDTYPE),10240),('traversal',generated(name='../escape'),10240)]:
 try:m.walk(lambda o:buf[o:o+512],total)
 except Exception:passed.append(name+'_refused')
 else:raise AssertionError(name)
class R:
 def __init__(self,url,headers,body,status=206):self.url=url;self.headers=headers;self.body=body;self.status=status;self.reads=0
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def geturl(self):return self.url
 def read(self,n):assert n==512;self.reads+=1;return self.body[:n]
class O:
 def __init__(self,change=None):self.change=change;self.responses=[]
 def open(self,req,timeout):
  assert timeout==20
  start=int(req.headers['Range'].split('=')[1].split('-')[0]);assert req.headers['If-match']=='"exact"'
  h={'Content-Range':f'bytes {start}-{start+511}/{len(a)}','Content-Length':'512','ETag':'"exact"'}
  r=R(req.full_url,h,a[start:start+512])
  if self.change:self.change(r)
  self.responses.append(r);return r
p={'schema':m.SCHEMA,'execution_released':True,'archives':[{'filename':f'IXI-{x}.tar','url':f'https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-{x}.tar','bytes':len(a),'etag_opaque':'"exact"'} for x in ['T1','MRA']],'max_headers_per_archive':1024,'max_response_body_bytes':1100000,'max_requests':2100,'deadline_seconds':1800}
with tempfile.TemporaryDirectory(prefix='ixi-generated-') as d:
 d=Path(d);o=O();result=m.execute(p,d/'fresh',o);assert result['status']=='complete_metadata_only' and result['archive_member_body_bytes_read']==0;passed.append('strict_exact_range_two_archives')
 for name,change in [('ignored_range',lambda r:setattr(r,'status',200)),('wrong_etag',lambda r:r.headers.__setitem__('ETag','"changed"')),('wrong_length',lambda r:r.headers.__setitem__('Content-Length','513')),('wrong_contentrange',lambda r:r.headers.__setitem__('Content-Range','bytes 1-512/10240')),('encoded_body',lambda r:r.headers.__setitem__('Content-Encoding','gzip'))]:
  o=O(change)
  try:m.execute(p,d/name,o)
  except Exception:assert len(o.responses)==1 and o.responses[0].reads==0;passed.append(name+'_refused_before_body_read')
  else:raise AssertionError(name)
 p2=dict(p,execution_released=False);o=O()
 try:m.execute(p2,d/'not_released',o)
 except Exception:assert not o.responses and not (d/'not_released').exists();passed.append('false_release_zero_network')
 else:raise AssertionError('false release')
result={'status':'passed','tests':passed,'count':len(passed),'generated_only':True,'network_requests':0,'actual_archive_bytes_read':0,'reader_sha256':hashlib.sha256((P/'walk-tar-metadata.py').read_bytes()).hexdigest()}
(P/'generated-header-controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
