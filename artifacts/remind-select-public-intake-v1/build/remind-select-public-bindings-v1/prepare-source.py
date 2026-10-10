"""Bind only the six authorized SELECT public series from existing IDC metadata."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import build_opener, HTTPSHandler, HTTPRedirectHandler, Request
from urllib.parse import urlencode
import hashlib, json, re, shutil, ssl, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'data/acquisition/remind-select-public-v1'
PINS = {
 'manifests/experiments/remind-component-cohort-v1.json': '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05',
 'manifests/experiments/remind-planning-pilot-v1.json': 'ff092655049c1f6d5eb324f9610a4c1f5b8ec50c1a29cfaf92a54f74c951395f',
 'build/remind-cohort-expansion-preparation-v1/idc-remind-series-metadata.json': '8301d9890e97316c8905448c9ec78e851d43ed18b383455ae138f4b13a8023be',
 'data/acquisition/remind-train-seg-v1/source/tcia-collection.html': 'cb172ef0f1fbdc79880dd7eccd932694175ffb9a6e0d0ba8fe757e9ed87813b9',
}
HELPERS = {
 'acquire_public_case.py': '9871654b5e1dc0e4d51fa4982a869d79dc5890c33a2386709da62b48dfdc2bcb',
 'acquire_btc_case.py': '2978b5c7756f46aac9dc8aa9f308fdf6f84a054326a48bd704fe3addc5bca0bf',
 'real_intake_io.py': '496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8',
}
NAMES = {'3D_AX_T1_postcontrast': 'structural_t1ce', 'tumor seg - MR ref: 3D_AX_T1_postcontrast': 'whole_tumor', 'cerebrum seg - MR ref: 3D_AX_T1_postcontrast': 'cerebrum'}
PEOPLE = {'ReMIND-013', 'ReMIND-037'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
 p.parent.mkdir(parents=True, exist_ok=True)
 with p.open('x') as f: json.dump(obj, f, indent=2, sort_keys=True); f.write('\n')
def main():
 for p,h in PINS.items(): assert sha(ROOT/p) == h, p
 cohort=json.loads((ROOT/next(iter(PINS))).read_text())
 members=[m for m in cohort['members'] if m['subject'] in PEOPLE]
 assert len(members)==2 and all(m['role']=='SELECT' for m in members)
 pilot=json.loads((ROOT/'manifests/experiments/remind-planning-pilot-v1.json').read_text())
 assert set(pilot['pilot']['SELECT'])==PEOPLE
 index=json.loads((ROOT/'build/remind-cohort-expansion-preparation-v1/idc-remind-series-metadata.json').read_text())
 series=[s for s in index['series'] if s['PatientID'] in PEOPLE and s['StudyDescription']=='Preop' and s['SeriesDescription'] in NAMES]
 assert len(series)==6 and sum(s['instanceCount'] for s in series)==372
 for person in PEOPLE:
  sub=[s for s in series if s['PatientID']==person]
  assert len(sub)==3 and {s['SeriesDescription'] for s in sub}==set(NAMES) and len({s['StudyInstanceUID'] for s in sub})==1
 assert all(s['license_short_name']=='CC BY 4.0' and s['source_DOI']=='10.7937/3rag-d070' and s['aws_bucket']=='idc-open-data' for s in series)
 assert shutil.disk_usage(ROOT).free > (100+64)*1024**3+100*1024**2
 assert not (BASE/'declaration.json').exists() and not (BASE/'verified').exists()
 (BASE/'source').mkdir(parents=True, exist_ok=True)
 for name,h in HELPERS.items():
  source=ROOT/'data/acquisition/remind-train-seg-v1/source'/name
  assert sha(source)==h
  target=BASE/'source'/name
  if target.exists(): assert sha(target)==h
  else: shutil.copyfile(source,target)
 class NoRedirect(HTTPRedirectHandler):
  def redirect_request(self,*args,**kwargs): raise RuntimeError('metadata redirect refused')
 ctx=ssl.create_default_context(); assert ctx.check_hostname and ctx.verify_mode==ssl.CERT_REQUIRED
 opener=build_opener(NoRedirect(), HTTPSHandler(context=ctx))
 ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}; objects=[]; proofs=[]
 for row in sorted(series,key=lambda s:(s['PatientID'],s['crdc_series_uuid'])):
  uid=row['crdc_series_uuid']; assert re.fullmatch('[0-9a-f-]{36}',uid)
  url='https://idc-open-data.s3.amazonaws.com/?'+urlencode({'list-type':'2','prefix':uid+'/','max-keys':'1000'})
  p=BASE/'source'/('listing-'+uid+'.xml')
  if p.exists():
   raw=p.read_bytes(); headers={'saved_response_reused':'Prior verified-TLS HTTP200 response saved before exact zero-byte directory marker handling was added; no payload request.'}
  else:
   with opener.open(Request(url,headers={'Accept-Encoding':'identity'}),timeout=30) as response:
    assert response.status==200 and response.geturl()==url and response.headers.get('Content-Encoding','identity')=='identity'
    raw=response.read(1024*1024+1); assert len(raw)<=1024*1024
    headers={k:response.headers.get(k) for k in ['Date','ETag','Content-Length','Content-Type']}
   with p.open('xb') as f: f.write(raw)
  tree=ET.fromstring(raw); assert tree.findtext('s:IsTruncated',namespaces=ns)=='false'
  assert tree.findtext('s:Name',namespaces=ns)=='idc-open-data' and tree.findtext('s:Prefix',namespaces=ns)==uid+'/'
  rows=[]
  for item in tree.findall('s:Contents',ns):
   key=item.findtext('s:Key',namespaces=ns); size=int(item.findtext('s:Size',namespaces=ns)); etag=item.findtext('s:ETag',namespaces=ns).strip('"')
   if key==uid+'/':
    assert size==0 and etag=='d41d8cd98f00b204e9800998ecf8427e'; continue
   assert re.fullmatch(re.escape(uid)+r'/[0-9a-f-]{36}\.dcm',key) and size>0
   assert re.fullmatch('[a-f0-9]{32}',etag), 'single-part source checksum required; multipart not whole-file MD5'
   rows.append({'patient_id':row['PatientID'],'patient_group':'ReMIND:'+row['PatientID'].split('-')[1],'role':'SELECT','kind':NAMES[row['SeriesDescription']], 'series_uuid':uid,'series_instance_uid':row['SeriesInstanceUID'],'study_instance_uid':row['StudyInstanceUID'],'modality':row['Modality'],'source_description':row['SeriesDescription'],'key':key,'path':row['PatientID']+'/'+key,'bytes':size,'size_bytes':size,'source_url':'https://idc-open-data.s3.amazonaws.com/'+key,'expected_md5':etag,'etag_opaque':'"'+etag+'"','sha256':None})
  assert len(rows)==row['instanceCount']
  objects.extend(rows); proofs.append({'series_uuid':uid,'url':url,'path':str(p.relative_to(ROOT)),'bytes':len(raw),'sha256':sha(p),'tls_verified':True,'headers':headers,'objects':len(rows),'exact_bytes':sum(o['bytes'] for o in rows)})
 assert len(objects)==372 and len({o['path'] for o in objects})==372
 total=sum(o['bytes'] for o in objects); assert total < 100*1024**2
 manifest={'schema':'remind-select-public-exact-objects-v1','created_utc':datetime.now(timezone.utc).isoformat(),'status':'all_six_public_series_listed','members':members,'source_series':series,'metadata_bindings':[{'path':p,'sha256':h,'bytes':(ROOT/p).stat().st_size} for p,h in PINS.items()], 'listing_proofs':proofs,'objects':sorted(objects,key=lambda x:x['path']),'total_objects':len(objects),'exact_total_bytes':total,'index_estimated_bytes':sum(round(s['series_size_MB']*1000000) for s in series),'source_checksum_authority':'IDC public S3 single-part ETag MD5, verified against each payload; multipart unsupported','prior_acquisition':'The two TRAIN-only raw/SEG manifests and completion receipts contain none of these six series; both SELECT patient verified directories absent.','private_series_allowed':False,'eval_allowed':False,'all_payloads_unreviewed':True,'header_qc':'not_run','anatomy_qc':'not_run','training_admitted':False,'stage':'byte_intake_only','task':'Approach to supplied whole-tumor region; automatic cerebrum is supplied estimated support; no independent manual anatomy or clinical injury claim.'}
 save(BASE/'source/exact-object-manifest.json',manifest)
 declaration={'schema':'remind-select-public-byte-release-v1','created_utc':datetime.now(timezone.utc).isoformat(),'authorization':'Root explicit continuous exact six-public-series SELECT013/037 intake; no headers/pixels/arrays, private ventricular series or EVAL067; same frozen roles.','execution_released':True,'manifest_path':str((BASE/'source/exact-object-manifest.json').relative_to(ROOT)),'manifest_sha256':sha(BASE/'source/exact-object-manifest.json'),'runner_path':str((HERE/'run-intake.py').relative_to(ROOT)),'runner_sha256':sha(HERE/'run-intake.py'),'helper_sha256':HELPERS,'metadata_pins':PINS,'free_space_reserve_bytes':100*1024**3,'active_outputs_allowance_bytes':64*1024**3,'max_workers':1,'max_transport_attempts_per_object_across_restarts':3,'max_payload_bytes':100*1024**2,'exact_source_bytes':total,'objects':372,'roles':['SELECT'],'all_payloads_unreviewed':True,'headers_or_arrays_allowed':False,'training_admitted':False}
 save(BASE/'declaration.json',declaration)
 save(HERE/'source-summary.json',{'status':manifest['status'],'manifest_path':declaration['manifest_path'],'manifest_sha256':declaration['manifest_sha256'],'declaration_path':str((BASE/'declaration.json').relative_to(ROOT)),'declaration_sha256':sha(BASE/'declaration.json'),'objects':372,'bytes':total,'metadata_requests':6,'payload_requests':0,'series':proofs})
 print(json.dumps({'status':manifest['status'],'objects':372,'bytes':total,'declaration_sha256':sha(BASE/'declaration.json')}),flush=True)
if __name__=='__main__': main()
