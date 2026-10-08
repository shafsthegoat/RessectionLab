"""Bounded independent QC of the frozen 144 real masks; no acquisition."""
import base64
from collections import Counter
from datetime import datetime,timezone
import gzip,hashlib,json,math,os,resource,signal,struct,subprocess,sys,time,zlib
from itertools import product
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import xml.etree.ElementTree as ET
START=time.monotonic();RSS_LIMIT=128*1024**2
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
DATA=ROOT/'data/anatomy/ds003949-v1.0.1';CACHE=DATA/'train-annotation-intake-v1';RUN=CACHE/'attempts/full-train-annotations-01'
COMMIT='229990cf152cb53ada1c52c3f9072c87b1b06c31'
class ResourceStop(RuntimeError):pass
def bound():
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 if rss>=RSS_LIMIT or time.monotonic()-START>=600:raise ResourceStop('Declared review resource boundary reached')
 return rss
def timeout(*_):raise ResourceStop('Declared review wall deadline reached')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):
 bound();assert p.stat().st_size<=2*1024**2
 return json.loads(p.read_bytes())
manifest_path=ROOT/'manifests/lausanne-train-annotations-v1.json';m=load(manifest_path)
assert sha(manifest_path.read_bytes())=='f779066f5cb784f623446f12565eb1405994ac9e3b41c1fef8210546598de624'
rows=[r for r in m['records'] if r['status']=='metadata_qualified'];assert len(rows)==144
allowed={DATA/r['path'] for r in rows}|{DATA/r['original_reference']['path'] for r in rows}
def audit_hook(event,args):
 if event=='socket.connect':raise AssertionError('No audit network access permitted')
 if event=='open' and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0]))
  if str(p).endswith(('.nii','.nii.gz','.nii.gz.partial')):
   assert p in allowed,str(p)
   assert not args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
sys.addaudithook(audit_hook)
import numpy as np
import nibabel as nib
bound()

def verify_file(p,entry,digest):
 before=p.stat();assert not p.is_symlink() and before.st_size==entry['bytes']
 h,md=hashlib.sha256(),hashlib.md5()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b);md.update(b);bound()
 assert h.hexdigest()==digest and md.hexdigest()==entry['expected_md5']
 assert before==p.stat()
 return {'bytes':before.st_size,'sha256':h.hexdigest(),'md5':md.hexdigest()}

def extensions(region,offset,endian):
 assert len(region)==offset-348 and 352<=offset<=65536
 assert not any(region[1:4]);records=[];cursor=4
 if region[0]:
  assert cursor<len(region)
  while cursor<len(region):
   assert len(region)-cursor>=8
   size,code=struct.unpack_from(endian+'ii',region,cursor)
   assert size>=16 and size%16==0 and code>=0 and cursor+size<=len(region)
   records.append({'offset':348+cursor,'end_exclusive':348+cursor+size,'esize':size,'ecode':code,
    'block_sha256':sha(region[cursor:cursor+size]),'payload_bytes':size-8,'payload_sha256':sha(region[cursor+8:cursor+size])})
   cursor+=size
 else:assert not any(region[4:])
 return {'status':'framed_uninterpreted' if records else 'absent_zero_padding','offset':348,'bytes':len(region),'sha256':sha(region),
  'indicator':list(region[:4]),'extension_count':len(records),'records':records,'payloads_interpreted':False,'used_as_annotation_or_coordinate_evidence':False}

def header(raw,record,digest):
 assert len(raw)==348 and raw==base64.b64decode(record['header_base64'],validate=True)
 assert sha(raw)==record['header_sha256'] and record['source_file_sha256']=='sha256:'+digest
 h=nib.Nifti1Header(binaryblock=raw,check=False)
 assert int(h['sizeof_hdr'])==348 and bytes(h['magic'])==b'n+1\0'
 summary={'shape':list(h.get_data_shape()),'spatial_units':h.get_xyzt_units()[0],'xyzt_units_code':int(h['xyzt_units']),
  'pixdim':h['pixdim'].tolist(),'qform_code':int(h['qform_code']),'sform_code':int(h['sform_code']),
  'qform_numeric_including_inactive':h.get_qform().tolist(),'sform_numeric':h.get_sform().tolist()}
 assert summary==record['raw_grid']
 return h

def step_info(a,b):
 x,y=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
 if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.signbit(x)!=np.signbit(y)):return {'valid':False,'reason':'nonfinite_or_sign_mismatch'}
 steps=np.abs(x.view(np.int32).astype(np.int64)-y.view(np.int32).astype(np.int64));pos=np.unravel_index(int(steps.argmax()),steps.shape)
 return {'valid':True,'maximum':int(steps.max()),'maximum_index':[int(i) for i in pos],'maximum_absolute_coefficient_difference':float(np.abs(np.asarray(a)-np.asarray(b)).max())}

def grid_check(a,b):
 diag={};eligible=True
 for tag,h in [('annotation',a),('reference',b)]:
  s=h.get_sform();q=h.get_qform();qc=int(h['qform_code']);sc=int(h['sform_code'])
  info=step_info(q,s) if qc else {'valid':True,'maximum':0};diag[tag]={'qform_code':qc,'sform_code':sc,'active_qform_sform_steps':info}
  eligible &= (len(h.get_data_shape())==3 and all(n>0 for n in h.get_data_shape()) and sc in (1,2)
   and np.isfinite(s).all() and np.array_equal(s[3],[0,0,0,1]) and abs(np.linalg.det(s[:3,:3]))>=1e-12
   and info['valid'] and info.get('maximum',5)<=4)
 same_shape=a.get_data_shape()==b.get_data_shape();units=b.get_xyzt_units()[0]=='mm' and a.get_xyzt_units()[0] in ('mm','unknown')
 A,B=a.get_sform(),b.get_sform();between=step_info(A,B);diag['between_sforms']=between
 if eligible and same_shape and units and between['valid'] and between['maximum']<=4:
  shape=a.get_data_shape();corners=np.array([(*p,1.) for p in product(*[(-.5,n-.5) for n in shape])])
  displacement=((A-B)@corners.T)[:3];inv=np.linalg.inv(B)[:3,:3];vd=inv@displacement
  ulp=np.maximum(np.abs(np.spacing(A.astype(np.float32))).astype(float),np.abs(np.spacing(B.astype(np.float32))).astype(float))
  wb=(4*ulp[:3])@np.array([*(n-.5 for n in shape),1.]);vb=np.abs(inv)@wb
  if np.all(np.max(np.abs(displacement),axis=1)<=wb) and np.max(vb)<.5 and np.max(np.abs(vd))<.5:
   return {'rule':'source_reference_serialization_equivalence','metrics':{'rule':'nifti1_float32_four_ulp_and_unchanged_indices_v1','maximum_allowed_float32_steps':4,
    'maximum_observed_float32_steps':between['maximum'],'maximum_support_displacement_reference_mm':float(np.linalg.norm(displacement,axis=0).max()),
    'maximum_support_displacement_voxels':float(np.linalg.norm(vd,axis=0).max()),'maximum_support_component_voxels':float(np.abs(vd).max()),
    'serialization_component_bound_reference_mm':wb.tolist(),'serialization_component_bound_voxels':vb.tolist(),'all_voxel_centres_keep_reference_index':True},
    'array_operation':'unchanged source voxel indices'},diag
 affines=[];exact=True
 for h in (a,b):
  q,qc=h.get_qform(coded=True);s,sc=h.get_sform(coded=True)
  valid=(qc or sc) and (not qc or qc in (1,2)) and (not sc or sc in (1,2)) and (not(qc and sc) or np.array_equal(q,s))
  if not valid:exact=False;break
  A=s if sc else q;exact &= np.isfinite(A).all() and np.array_equal(A[3],[0,0,0,1]) and abs(np.linalg.det(A[:3,:3]))>=1e-12;affines.append(A)
 if exact and same_shape and units and np.array_equal(affines[0],affines[1]):
  return {'rule':'exact_coded_native_voxel_grid_v1','all_voxel_centres_keep_reference_index':True,'reference_mm_inherited':a.get_xyzt_units()[0]=='unknown','array_operation':'unchanged source voxel indices'},diag
 return None,diag

def compare_nested(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[compare_nested(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[compare_nested(x,y) for x,y in zip(a,b)]
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15),(a,b)
 else:assert a==b,(a,b)

batch=load(RUN/'batch.json');source=load(RUN/'source.json');decl=load(RUN/'declaration.json')
assert sha((RUN/'batch.json').read_bytes())=='82cca3ac73b5b3e1eedac425c2c5910d901ea6ad47c64e5e71885a26f0024c46'
assert len(batch['outcomes'])==len(decl['outcomes'])==len(m['records'])==148
assert [o['path'] for o in batch['outcomes']]==[r['path'] for r in m['records']]
assert batch['outcome_counts']=={'completed':144,'metadata_failed_excluded':4}
assert 0<batch['elapsed_seconds']<batch['max_seconds']==600
assert batch['attempted_source_bytes']==batch['max_source_bytes']==13977015 and batch['automatic_retries']==0
source_sha=sha((RUN/'source.json').read_bytes());assert batch['execution_source_sha256']==decl['execution_source_sha256']==source_sha
for name,digest in source['files'].items():
 assert sha((RUN/'source-snapshot'/name).read_bytes())==digest
 assert sha(subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT))==digest
for e in (m['cohort'],m['original_index'],*m['metadata_sources'].values()):
 b=(CACHE/'metadata'/(e['sha256']+'.bin')).read_bytes();assert len(b)==e['bytes'] and sha(b)==e['sha256']
cohort=load(ROOT/m['cohort']['path']);people={p['subject']:p for p in cohort['members']}
assert load(ROOT/m['metadata_sources']['rights']['path'])['License']=='CC0'
ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'};xml=ET.fromstring((ROOT/m['metadata_sources']['versions']['path']).read_bytes())
assert xml.findtext('s:IsTruncated',namespaces=ns)=='false'
versions={(v.findtext('s:Key',namespaces=ns),v.findtext('s:VersionId',namespaces=ns)):{k:v.findtext('s:'+k,namespaces=ns) for k in ('Size','ETag','LastModified')} for v in xml.findall('s:Version',ns)}
previous={}
for run in CACHE.joinpath('attempts').iterdir():
 if run==RUN or not run.is_dir():continue
 for p in [*run.glob('batch.json'),*run.glob('*/receipt.json')]:previous[str(p.relative_to(ROOT))]=sha(p.read_bytes())
assert previous['data/anatomy/ds003949-v1.0.1/train-annotation-intake-v1/attempts/pilot-sub022-transfer-01/sub-022_ses-20101011_desc-Lesion_1_mask.nii.gz/receipt.json']=='1a8ffdf58847b6468f884905e5101313d6799f2addeeaa0f68350dce705f1c09'
report={'schema':'lausanne-annotation-full-intake-independent-review-v1','started_at':datetime.now(timezone.utc).isoformat(),'source_commit':COMMIT,'source_sha256':source_sha,
 'batch_sha256':sha((RUN/'batch.json').read_bytes()),'manifest_sha256':sha(manifest_path.read_bytes()),'review_outcomes':[{'path':r['path'],'subject':r['subject'],'session':r['session'],'status':'not_reviewed' if r['status']=='metadata_qualified' else 'metadata_failed_excluded','metadata_failure':r.get('metadata_failure')} for r in m['records']],
 'historical_batch_counts':batch['outcome_counts'],'previous_attempt_metadata_hashes':previous,'resource_limits':{'seconds':600,'sampled_RSS_bytes':RSS_LIMIT,'RSS_is_diagnostic_not_hard_guarantee':True,'CPU_threads':1},
 'download_coordination':'Separate QC phase; this audit does not gate continuous acquisition.','training_admitted':False,'scanner_frame_admitted':False,'spatial_planning_admitted':False}
results={r['path']:r for r in report['review_outcomes']};outcomes={r['path']:r for r in batch['outcomes']};ref_cache={};source_cache=set();mask_total=0;tof_bytes=0;request_total=0
try:
 for row in m['records']:
  bound();outcome=outcomes[row['path']];result=results[row['path']]
  if row['status']=='metadata_failed':assert outcome['status']=='metadata_failed_excluded' and outcome['metadata_failure']==row['metadata_failure'];continue
  try:
   person=people[row['subject']];assert person['role']==row['role']=='TRAIN' and person['group']=='patient' and row['session'][4:] in person['sessions']
   trial=ROOT/outcome['attempt'];assert trial.parent==RUN and trial.name==Path(row['path']).name
   receipt=load(trial/'receipt.json');receipt_sha=sha((trial/'receipt.json').read_bytes());assert receipt_sha==outcome['receipt_sha256']
   assert receipt['path']==row['path'] and receipt['role']=='TRAIN' and receipt['subject']==row['subject'] and receipt['session']==row['session']
   assert receipt['manifest_sha256']==report['manifest_sha256'] and receipt['execution_source_sha256']==source_sha
   assert receipt['annotation_subtype_status']==row['annotation_subtype_status'] and receipt['semantics']==m['semantics']
   for f in ('training_admitted','scanner_frame_admitted','spatial_planning_admitted'):assert receipt[f] is False
   assert receipt['optimizer_updates']==receipt['recorded_rl_transitions']==0 and receipt['annotation_available_at'] is receipt['review_available_at'] is None
   assert receipt['content_qc']['status']=='passed'
   assert outcome['status']=='completed' and outcome['exit_code']==0 and receipt['status']==outcome['receipt_status']
   started=load(trial/'started.json');claim=load(trial/'worker-claim.json');assert started['worker_pid']==outcome['worker_pid']
   assert 0<receipt['elapsed_seconds']<started['max_seconds']<=120
   assert claim=={'mask':row['path'],'manifest_sha256':report['manifest_sha256'],'max_seconds':started['max_seconds']}
   for method in (os.kill,os.killpg):
    try:method(started['worker_pid'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('PID/process group still exists')
   file=row['file'];v=parse_qs(urlsplit(file['source_url']).query)['versionId'][0]
   assert file['source_url'].split('?')[0]=='https://s3.amazonaws.com/openneuro.org/ds003949/'+row['path']
   official=versions[('ds003949/'+row['path'],v)];assert int(official['Size'])==file['bytes'] and official['ETag'].strip('"')==file['expected_md5']
   pointer=base64.b64decode(row['pointer_base64'],validate=True);assert sha(pointer)==row['pointer_sha256']
   assert hashlib.sha1(b'blob '+str(len(pointer)).encode()+b'\0'+pointer).hexdigest()==row['pointer_git_blob_sha1']
   sidecar=base64.b64decode(row['sidecar_base64'],validate=True);assert sha(sidecar)==row['sidecar_sha256'] and json.loads(sidecar)==row['sidecar']=={'Type':'Lesion','RawSources':Path(row['original_reference']['path']).name,'Space':'orig'}
   requests=receipt['requests'];assert len(requests)<=1;request_total+=len(requests)
   for req in requests:
    rh={k.lower():v for k,v in req['response_headers'].items()};assert req['method']=='GET' and req['url']==req['final_url']==file['source_url'] and req['status'] in (200,206)
    assert rh['x-amz-version-id']==v and rh.get('content-encoding','identity')=='identity' and rh['etag'].strip('"')==file['expected_md5']
    assert req['status']==200 and int(rh['content-length'])==file['bytes']
   digest=receipt['acquisition']['sha256'];fixity=verify_file(DATA/file['path'],file,digest);mask_total+=file['bytes']
   with gzip.open(DATA/file['path'],'rb') as stream:
    raw=stream.read(348);h=header(raw,receipt['content_qc']['raw_grid'],digest)
    assert h.get_data_dtype()==np.dtype('uint8') and int(h['bitpix'])==8
    assert h.get_slope_inter()==(1.,0.) and receipt['content_qc']['source_scaling']==[1.,0.] and receipt['content_qc']['dtype']=='|u1'
    offset=float(h['vox_offset']);assert offset.is_integer() and 352<=offset<=65536
    ext=extensions(stream.read(int(offset)-348),int(offset),h.endianness);assert ext==receipt['content_qc']['extensions']
    total=math.prod(h.get_data_shape());assert 0<total<=m['bounds']['max_mask_voxels'] and total+offset<=m['bounds']['max_uncompressed_mask_bytes']
    remaining=total;zeros=ones=0
    while remaining:
     bound();n=min(65536,remaining);block=stream.read(n);assert len(block)==n
     z=block.count(b'\0');o=block.count(b'\1');assert z+o==n;zeros+=z;ones+=o;remaining-=n
    assert stream.read(1)==b''
   assert 0<ones<total and ones==receipt['content_qc']['positive_voxels'] and zeros==receipt['content_qc']['background_voxels'] and receipt['content_qc']['background_semantics']=='unknown'
   budget=receipt['content_qc']['decoding_budget'];assert budget=={'voxels':total,'native_payload_bytes':total,'logical_float64_bytes':total*8,'chunk_voxels':262144,'chunk_working_bytes_bound':3145728}
   ref=receipt['reference_qc'];grid=receipt['grid_qc'];reference_result={}
   if ref['status']=='deferred_missing_original_receipt':
    assert grid['status']=='deferred_reference_not_passed' and receipt['status']=='content_passed_reference_or_grid_pending'
    path=ROOT/ref['receipt_path'];assert path==DATA/'train-intake-v1/acquired'/(row['subject']+'_'+row['session']+'.json')
    reference_result={'batch_status':ref['status'],'current_original_receipt_exists':path.exists(),'current_TOF_complete_path_exists':(DATA/row['original_reference']['path']).exists()}
   else:
    assert ref['status']=='passed' and ref['full_pair_current_fixity_checked'] is False
    snapshot=ROOT/ref['original_receipt_snapshot'];assert snapshot==trial/'original-receipt.json' and sha(snapshot.read_bytes())==ref['original_receipt_sha256']
    original=load(snapshot);assert original['role']=='TRAIN' and original['subject']==row['subject'] and original['session']==row['session']
    sp=ROOT/original['execution_source_record']
    if str(sp) not in source_cache:
     sd=load(sp);assert sha(sp.read_bytes())==original['execution_source_sha256']
     for name,hashvalue in sd['files'].items():assert sha((sp.parent/'source-snapshot'/name).read_bytes())==hashvalue
     source_cache.add(str(sp))
    rf=row['original_reference'];key=rf['path'];rsha=ref['original_tof_sha256']
    if key not in ref_cache:
     rf_fixity=verify_file(DATA/key,rf,rsha);tof_bytes+=rf['bytes'];decompress=zlib.decompressobj(31);rraw=b''
     with (DATA/key).open('rb') as stream:
      while len(rraw)<348:
       block=stream.read(1024);assert block;rraw+=decompress.decompress(block,348-len(rraw))
     rh=header(rraw,ref['raw_grid'],rsha);ref_cache[key]=(rh,rf_fixity,sha(rraw))
    rh,rf_fixity,rheaderhash=ref_cache[key];assert ref['raw_grid']['header_sha256']==rheaderhash and rf_fixity['sha256']==rsha
    proof,diagnostic=grid_check(h,rh)
    if grid['status']=='passed':assert proof is not None;compare_nested(proof,grid['proof']);assert receipt['status']=='qc_complete'
    else:assert grid['status']=='unresolved' and proof is None and receipt['status']=='content_passed_reference_or_grid_pending'
    reference_result={'batch_status':'passed','TOF_sha256':rsha,'TOF_header_sha256':rheaderhash,'grid_status':grid['status'],'proof_rule':proof['rule'] if proof else None}
    if proof is None:reference_result.update(reason=grid['reason'],independent_diagnostic=diagnostic)
   result.update(status='review_passed',receipt_status=receipt['status'],receipt_sha256=receipt_sha,role='TRAIN',annotation_subtype_status=row['annotation_subtype_status'],
    fixity=fixity,source_version=v,GETs=len(requests),shape=list(h.get_data_shape()),voxels=total,positives=ones,zeros=zeros,extension_count=ext['extension_count'],
    extension_codes=[r['ecode'] for r in ext['records']],extension_sha256=ext['sha256'],raw_header_sha256=sha(raw),gzip_integrity_verified=True,reference=reference_result)
  except ResourceStop:raise
  except BaseException as error:
   result.update(status='review_failed',error={'type':type(error).__name__,'message':str(error)})
  reviewed=sum(o['status'] in ('review_passed','review_failed') for o in results.values())
  if reviewed%12==0:print(json.dumps({'reviewed':reviewed,'elapsed_seconds':round(time.monotonic()-START,3),'peak_RSS_bytes':bound()}),flush=True)
except BaseException as error:
 report['stop_reason']={'type':type(error).__name__,'message':str(error)}
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
 report['previous_attempts_unchanged']=all(sha((ROOT/p).read_bytes())==h for p,h in previous.items())
 report['review_counts']=dict(Counter(o['status'] for o in results.values()))
 passed=[o for o in results.values() if o['status']=='review_passed']
 report['verified_receipt_status_counts']=dict(Counter(o['receipt_status'] for o in passed))
 report['verified_reference_states']=dict(Counter(o['reference']['batch_status'] for o in passed))
 report['verified_grid_rules']=dict(Counter(o['reference'].get('proof_rule') for o in passed))
 report['verified_subtypes']=dict(Counter(o['annotation_subtype_status'] for o in passed))
 report['verified_positive_voxels']=sum(o['positives'] for o in passed)
 report['verified_source_voxels']=sum(o['voxels'] for o in passed)
 report['extension_summary']={'masks_with_extensions':sum(o['extension_count']>0 for o in passed),'codes':dict(Counter(code for o in passed for code in o['extension_codes']))}
 report['source_bytes_verified']={'masks':mask_total,'distinct_TOF_references':len(ref_cache),'TOF_compressed_bytes':tof_bytes,'reference_header_bytes_decompressed':348*len(ref_cache)}
 report['batch_resources']={'elapsed_seconds':batch['elapsed_seconds'],'limit_seconds':600,'workers_completed':144,'sampled_RSS_not_recorded_by_batch':True,'GETs_in_verified_receipts':request_total}
 report['review_resources']={'elapsed_seconds':time.monotonic()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'threads_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
 report['unreviewed_paths']=[o['path'] for o in results.values() if o['status']=='not_reviewed']
 report['pending_references']=[{'path':o['path'],'subject':o['subject'],'session':o['session'],**o['reference']} for o in passed if o['reference']['batch_status']!='passed']
 report['failed_grids']=[{'path':o['path'],'subject':o['subject'],'session':o['session'],**o['reference']} for o in passed if o['reference'].get('grid_status')=='unresolved']
 report['conclusion']='complete_independent_QC_audit' if report['review_counts']=={'review_passed':144,'metadata_failed_excluded':4} and report['previous_attempts_unchanged'] else 'incomplete_or_discrepant_review_preserved'
 report['scope_limits']=['QC is separate from continuous downloading and does not gate acquisitions.','Completed workers are not equivalent to qc_complete; pending references and failed grids remain pending.','No label repair, other image scalar decode, registration, annotation negative coverage, anatomical review or training/planning admission follows.','All historical roles, availability unknowns, manual-region subtype distinctions and earlier attempts remain unchanged.','No network calls or original writes in this audit; mask scalars were checked using bounded 64KiB chunks, no full image array.','128MiB RSS is an observed diagnostic; this is not a hard memory guarantee.']
 report['audit_script_sha256']=sha(Path(__file__).read_bytes())
 (OUT/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'conclusion':report['conclusion'],'counts':report['review_counts'],'receipt_status':report['verified_receipt_status_counts'],'elapsed_seconds':report['review_resources']['elapsed_seconds'],'peak_RSS_bytes':report['review_resources']['peak_RSS_bytes'],'verification_sha256':sha((OUT/'verification.json').read_bytes())}),flush=True)
