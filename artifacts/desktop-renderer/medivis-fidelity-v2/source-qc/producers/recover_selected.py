"""Recover only one fixed CTA case plus official labels and attribution from a public ZIP."""
import hashlib,json,struct,urllib.request,zlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
index=json.loads((ROOT/'zip-members.json').read_text())
receipt={'schema':'topbrain-one-case-selected-range-recovery-v1','source_page':'https://zenodo.org/records/16623496','source_doi':'10.5281/zenodo.16623496','source_archive_url':index['url'],'source_archive_bytes':index['size'],'source_archive_published_md5':'34d684cc349d57d453935d3de3baeb00','whole_archive_retrieved_or_verified':False,'case':'topcow_ct_001','selection':'First numerically identified CTA case from fixed Batch1 release; no multi-case search or model inference.','license':'Open use with attribution; noncommercial use permitted; commercial use requires prior permission from data owner.','max_payload_bytes':250000000,'members':[],'directory_tail_bytes_received':131072,'commercial_use_permitted_without_permission':False}
selected=['License.txt','README.txt','labelmap_topbrain_ct.txt','topcow_ct_001_0000.nii.gz','topcow_ct_001.nii.gz']
for basename in selected:
 row=next(x for x in index['rows'] if x['name'].endswith('/'+basename));start=row['offset'];stop=start+30+len(row['name'].encode())+row['compressed']+1023
 if row['compressed']>100000000:raise ValueError('Member exceeds one-case budget')
 dest=ROOT/'originals'/basename
 if dest.exists():raise ValueError(f'Existing member {dest}')
 request=urllib.request.Request(index['url'],headers={'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity'})
 with urllib.request.urlopen(request,timeout=60) as r:
  if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{stop}/{index["size"]}':raise ValueError('Range not honored')
  body=r.read(stop-start+2)
  if len(body)!=stop-start+1:raise ValueError('Wrong range length')
  hdr=struct.unpack_from('<4s5H3I2H',body,0);n,e=hdr[-2:];name=body[30:30+n].decode()
  if hdr[0]!=b'PK\x03\x04' or name!=row['name'] or hdr[3]!=row['method']:raise ValueError('Selected ZIP header differs')
  payload=body[30+n+e:30+n+e+row['compressed']]
  data=zlib.decompress(payload,-15) if row['method']==8 else payload
  if len(data)!=row['size'] or zlib.crc32(data)!=row['crc']:raise ValueError('Selected member CRC/size differs')
  dest.write_bytes(data)
  rec={**row,'path':str(dest),'sha256':hashlib.sha256(data).hexdigest(),'range':f'bytes={start}-{stop}','bytes_received':len(body),'content_range':r.headers.get('Content-Range'),'last_modified':r.headers.get('Last-Modified'),'verification':'Central directory and local-header name/method; exact bounded HTTP206 range; decompressed size and ZIP CRC32. Member SHA256 recorded for future integrity. Whole archive MD5 not independently verified.'}
  receipt['members'].append(rec);receipt['total_bytes_received']=sum(x['bytes_received'] for x in receipt['members'])+131072
  (ROOT/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(basename,len(data),rec['sha256'],flush=True)
receipt['status']='verified_selected_members'
(ROOT/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Total transfer:',receipt['total_bytes_received'],flush=True)
