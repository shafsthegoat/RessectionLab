"""Saved metadata/header audit only. No source payload opens or network."""
import ast,base64,collections,hashlib,json,math,os,pathlib,subprocess,sys,time
from itertools import product
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=pathlib.Path(__file__).resolve().parent
BASE=ROOT/'build/lausanne-deferred-qc-execution-v1'
def guard(event,args):
 if event=='socket.connect':raise AssertionError('Network forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).endswith(('.nii','.nii.gz','.nii.gz.partial')):
  raise AssertionError('Scientific payload opens forbidden')
sys.addaudithook(guard)
sys.path.insert(0,str(ROOT/'scripts'));import lausanne_deferred_qc as qc
import numpy as np,nibabel as nib
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
 p=pathlib.Path(p)
 if not p.is_absolute():p=ROOT/p
 assert p.stat().st_size<=2*1024**2 and not p.is_symlink()
 return p.read_bytes()
def load(p):return json.loads(read(p))
def proof(p):
 raw=read(p['path']);assert len(raw)==p['bytes'] and sha(raw)==p['sha256']
 return json.loads(raw)
def header(record,digest):
 raw=base64.b64decode(record['header_base64'],validate=True)
 assert len(raw)==348 and sha(raw)==record['header_sha256'] and record['source_file_sha256']=='sha256:'+digest
 h=nib.Nifti1Header(binaryblock=raw,check=False)
 actual={'shape':list(h.get_data_shape()),'spatial_units':h.get_xyzt_units()[0],'xyzt_units_code':int(h['xyzt_units']),'pixdim':h['pixdim'].tolist(),'qform_code':int(h['qform_code']),'sform_code':int(h['sform_code']),'qform_numeric_including_inactive':h.get_qform().tolist(),'sform_numeric':h.get_sform().tolist()}
 assert record['raw_grid']==actual
 return h
# Reuse only the three independent small-matrix functions from our earlier audit.
source=read('build/lausanne-annotation-full-intake-review-v1/audit_full.py')
assert sha(source)=='eeaa3bd4e6cb8112dbe9af48d28a80d0e85281a7c1f93cd26e7d85a3c9b352cb'
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('step_info','grid_check','compare_nested')]
assert len(nodes)==3
exec(compile(ast.Module(body=nodes,type_ignores=[]),'retained_independent_grid_functions','exec'),globals())
report={'schema':'lausanne-deferred-qc-output-independent-review-v1','status':'in_progress','payload_files_opened':0,'scalar_arrays_decoded':0,'network_requests':0,'tracked_edits':0}
try:
 summary=load(BASE/'summary.json');idx=load(BASE/'receipt-index.json')
 assert sha(read(BASE/'summary.json'))=='af9e4c67b64d48135a0e792145ee1746b9a97f872e1a6f1c9933122610129ae3'
 assert sha(read(BASE/'receipt-index.json'))=='5568bf96755df0aed141ef3552bba9eda5e2cabe1df15e6d9e0051cddc8c86b4'
 m,am,sessions=qc.preflight()
 commit=subprocess.check_output(['git','rev-parse','e2fe6d4'],cwd=ROOT,text=True).strip();report['source_commit']=commit
 cohort=proof(m['cohort']);people={p['subject']:p for p in cohort['members']}
 originals={r['path']:r for r in m['originals']};followups={r['path']:r for r in m['reference_followups']}
 annotations={r['path']:r for r in am['records']}
 source_sets=set();all_receipts={};record_results=[];pids=[];runs={}
 for run_id,saved in idx.items():
  declaration=proof(saved['declaration']);batch=proof(saved['batch']);run=ROOT/pathlib.Path(saved['declaration']['path']).parent
  assert declaration['run_id']==batch['run_id']==run_id and declaration['manifest_sha256']==batch['manifest_sha256']==qc.MANIFEST_SHA
  assert qc.claims_match(declaration,qc.CLAIMS) and qc.claims_match(batch,qc.CLAIMS)
  assert declaration['workers']==1 and declaration['automatic_retries']==0
  assert 0<batch['elapsed_seconds']<declaration['max_seconds']==1800
  assert batch['status']=='bounded_attempts_finished' and batch['preserved_failures']==m['preserved_failures']
  rows=originals if declaration['stage']=='originals' else followups
  assert declaration['selected_paths']==list(rows) and [r['path'] for r in batch['outcomes']]==list(rows)
  assert [r['path'] for r in saved['records']]==list(rows)
  assert batch['outcome_counts']=={'completed':len(rows)}==saved['outcome_counts']
  qc.validate_execution(run,declaration,m,current=True)
  for path,digest in declaration['execution']['files'].items():
   if (path,digest) not in source_sets:
    assert sha(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))==digest
    source_sets.add((path,digest))
  outcomes={r['path']:r for r in batch['outcomes']}
  for item in saved['records']:
   row=rows[item['path']];receipt=proof(item['receipt']);outcome=proof(item['outcome']);intent=proof(item['intent'])
   trial=ROOT/pathlib.Path(item['receipt']['path']).parent
   for key in ('receipt','outcome','intent'):
    assert read(BASE/'receipts'/run_id/trial.name/(key+'.json'))==read(item[key]['path'])
   assert outcome==outcomes[item['path']] and outcome['status']==outcome['supervision_status']=='completed' and outcome['exit_code']==0
   assert outcome['receipt_sha256']==item['receipt']['sha256'] and outcome['intent_sha256']==item['intent']['sha256']
   assert outcome['review_status']==receipt['status'] and outcome['attempt']==str(trial.relative_to(ROOT))
   assert intent['path']==row['path'] and intent['stage']==declaration['stage'] and intent['declaration_sha256']==saved['declaration']['sha256']
   assert 0<receipt['elapsed_seconds']<intent['max_seconds']<=m['bounds']['max_worker_seconds']
   assert receipt['stage']==declaration['stage']
   qc.review_receipt_contract(receipt,row,item['intent']['sha256'],saved['declaration']['sha256'])
   assert people[row['subject']]['role']==row['role']=='TRAIN' and row['session'][4:] in people[row['subject']]['sessions']
   claim=load(trial/'worker-claim.json');assert claim=={'path':row['path'],'pid':outcome['worker_pid']}
   pids.append(outcome['worker_pid'])
   for kill in (os.kill,os.killpg):
    try:kill(outcome['worker_pid'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('Recorded worker or process group remains present')
   all_receipts[row['path']]=(receipt,trial,item)
   record_results.append({'path':row['path'],'subject':row['subject'],'session':row['session'],'stage':declaration['stage'],'role':'TRAIN','parent_status':outcome['status'],'review_status':receipt['status'],'receipt_sha256':item['receipt']['sha256'],'elapsed_seconds':receipt['elapsed_seconds']})
  runs[run_id]={'stage':declaration['stage'],'outcomes':len(rows),'elapsed_seconds':batch['elapsed_seconds'],'limit_seconds':declaration['max_seconds'],'declaration_sha256':saved['declaration']['sha256'],'batch_sha256':saved['batch']['sha256']}
 # Authenticate all26 original results and recompute saved header/scalar arithmetic.
 for path,row in originals.items():
  r,trial,item=all_receipts[path];validated,raw=qc.completed_original_review(ROOT/item['receipt']['path'],row,m)
  assert validated==r and raw==read(item['receipt']['path'])
  h=header(r['raw_grid'],r['sha256']);dtype=h.get_data_dtype()
  assert r['header_qc']['dtype']==dtype.str and dtype.kind in 'iuf'
  slope,intercept=h.get_slope_inter();scaling=[1.,0.] if slope is None else [float(slope),float(intercept)]
  assert scaling==r['header_qc']['source_scaling']
  assert r['decoding_budget']==qc.scalar_budget([int(v) for v in h.get_data_shape()],dtype.itemsize,m['bounds'])
  assert r['scalar_qc']['voxels']==math.prod(h.get_data_shape()) and r['scalar_qc']['array_retained'] is False
  g=r['geometry_qc']['header'];assert g['shape']==list(h.get_data_shape()) and g['physical_units']=='mm'
  assert g['qform_native']==(h.get_qform().tolist() if int(h['qform_code']) else None)
  assert g['sform_native']==(h.get_sform().tolist() if int(h['sform_code']) else None)
 # Authenticate baseline mask review; unchanged bytes are the prior scalar authority.
 baseline=load('build/lausanne-annotation-full-intake-review-v1/verification.json')
 assert sha(read('build/lausanne-annotation-full-intake-review-v1/verification.json'))=='756a12b35e2634efa5d53629903d25798fb434b2c82d397a6d20cf958aacfcd9'
 oldbatch=load('data/anatomy/ds003949-v1.0.1/train-annotation-intake-v1/attempts/full-train-annotations-01/batch.json')
 assert sha(read('data/anatomy/ds003949-v1.0.1/train-annotation-intake-v1/attempts/full-train-annotations-01/batch.json'))==baseline['batch_sha256']
 oldpaths={r['path']:r for r in oldbatch['outcomes'] if r['status']=='completed'}
 oldreceipts={}
 for old in baseline['review_outcomes']:
  if old['status']!='review_passed':continue
  raw=read(pathlib.Path(oldpaths[old['path']]['attempt'])/'receipt.json');assert sha(raw)==old['receipt_sha256']
  oldreceipts[old['path']]=json.loads(raw)
 references=[];legacy_sources=set()
 for path,row in followups.items():
  r,trial,item=all_receipts[path];a=annotations[path];prior=qc.prior_mask_contract(row,a)
  assert read(trial/'prior-mask-receipt.json')==qc.proof_bytes(row['receipt'])
  assert prior==oldreceipts[path] and r['prior_mask_receipt']==row['receipt']
  assert r['mask_sha256']==prior['acquisition']['sha256'] and r['mask_scalars_decoded'] is False
  assert r['content_qc']=={'status':'reused_source_bound_prior_pass','receipt_sha256':row['receipt']['sha256']}
  assert r['annotation_subtype_status']==a['annotation_subtype_status'] and r['background_semantics']=='unknown'
  ref=r['reference_qc'];assert ref['status']=='passed' and ref['referenced_tof_passed'] is True and ref['current_tof_fixity_checked'] is True and ref['full_pair_current_fixity_checked'] is False
  if row['reference_mode']=='separate_original_review':
   original,_,original_item=all_receipts[row['reference_path']]
   assert ref['review_receipt_path']==original_item['receipt']['path'] and ref['review_receipt_sha256']==original_item['receipt']['sha256']
   assert read(ref['review_receipt_snapshot'])==read(original_item['receipt']['path'])
   assert ref['original_tof_sha256']==original['sha256'] and ref['raw_grid']==original['raw_grid']
   assert ref['referenced_tof_qc']=={k:original[k] for k in ('header_qc','scalar_qc','geometry_qc')}
   assert ref['other_modality_qc']==[] and ref['source_receipt_pair_qc']=='not_assessed'
  else:
   original=proof(row['legacy_receipt']);assert read(ref['original_receipt_snapshot'])==qc.proof_bytes(row['legacy_receipt'])
   assert ref['original_receipt_sha256']==row['legacy_receipt']['sha256']
   c=qc.annotations.original_receipt_contract(sessions[(row['subject'],row['session'])],original,m['original_index']['sha256'],a['original_reference'])
   assert all(ref[k]==v for k,v in c.items())
   f=next(f for f in original['files'] if f['path']==row['reference_path']);assert f['sha256']==ref['original_tof_sha256']
   if original['execution_source_sha256'] not in legacy_sources:
    qc.annotations.originals.validate_retained_source(original);legacy_sources.add(original['execution_source_sha256'])
  mh=header(prior['content_qc']['raw_grid'],r['mask_sha256']);rh=header(ref['raw_grid'],ref['original_tof_sha256'])
  result,diagnostic=grid_check(mh,rh)
  if result is None:assert r['grid_qc']['status']=='unresolved' and r['status']=='review_failed'
  else:assert r['grid_qc']['status']=='passed' and r['status']=='review_passed';compare_nested(result,r['grid_qc']['proof'])
  references.append({'path':path,'subject':row['subject'],'session':row['session'],'reference_mode':row['reference_mode'],'reference_status':'passed','grid_status':r['grid_qc']['status'],'rule':result['rule'] if result else None,'annotation_subtype_status':a['annotation_subtype_status']})
 # Reconcile all420 originals using authoritative legacy or new separate receipts.
 reconciled=load(BASE/'original-reconciliation.json');assert len(reconciled)==420 and len({r['path'] for r in reconciled})==420
 actual_originals={};legacy_receipts={}
 for row in reconciled:
  p=row['path']
  if p in originals:
   r=all_receipts[p][0];passed=r['status']=='review_passed';reason=None
  else:
   r=proof(row['receipt'])
   key=(row['subject'],row['session']);session=sessions[key]
   if key not in legacy_receipts:
    tof=next(e for e in session['files'] if e['path'].endswith('_angio.nii.gz'))
    qc.annotations.original_receipt_contract(session,r,m['original_index']['sha256'],tof)
    if r['execution_source_sha256'] not in legacy_sources:
     qc.annotations.originals.validate_retained_source(r);legacy_sources.add(r['execution_source_sha256'])
    legacy_receipts[key]=r
   assert r==legacy_receipts[key]
   q=next(q for q in r['integrity_qc'] if q['path']==p);passed=q['status']=='header_and_scalar_checks_passed';reason=None if passed else q['reason']
  assert row['passed']==passed and row['failure_reason']==reason
  assert people[row['subject']]['role']=='TRAIN'
  actual_originals[p]=passed
 expected_images={e['path'] for s in sessions.values() for e in s['files'] if e['path'].endswith('.nii.gz')}
 assert set(actual_originals)==expected_images
 assert len(legacy_receipts)==197
 assert sum(actual_originals.values())==414 and sum(v for p,v in actual_originals.items() if p.endswith('_angio.nii.gz'))==210
 failed_originals=[r for r in reconciled if not r['passed']]
 assert {r['subject'] for r in failed_originals}=={'sub-253','sub-410','sub-411','sub-418','sub-424','sub-473'}
 masks=load(BASE/'mask-reconciliation.json');assert len(masks)==144 and {r['path'] for r in masks}==set(oldreceipts)
 for row in masks:
  p=row['path'];r=all_receipts[p][0] if p in followups else oldreceipts[p]
  assert row['new_followup']==(p in followups) and row['reference_qc_status']==r['reference_qc']['status']=='passed'
  assert row['grid_qc_status']==r['grid_qc']['status'] and row['content_qc_status']=='passed'
  assert row['annotation_subtype_status']==annotations[p]['annotation_subtype_status']
  assert row['background_semantics']=='unknown' and row['annotation_available_at'] is row['review_available_at'] is None
 finalmaskcounts=dict(collections.Counter(r['grid_qc_status'] for r in masks));assert finalmaskcounts=={'passed':136,'unresolved':8}
 # Recompute retained old sub454 rejection as well; no new scalar access.
 old454=next(r for r in oldreceipts.values() if r['subject']=='sub-454')
 assert grid_check(header(old454['content_qc']['raw_grid'],old454['acquisition']['sha256']),header(old454['reference_qc']['raw_grid'],old454['reference_qc']['original_tof_sha256']))[0] is None
 assert summary['original_reconciliation']['image_integrity_geometry_passed']==414
 assert summary['mask_reconciliation']['final_grid_passed']==136 and summary['mask_reconciliation']['final_grid_unresolved']==8
 assert [r['path'] for r in summary['mask_reconciliation']['new_unresolved']]==[r['path'] for r in references if r['grid_status']=='unresolved']
 assert qc.claims_match(summary['admission'],qc.CLAIMS)
 report.update(status='saved_results_verified',runs=runs,source_snapshot_files_verified=len(source_sets),copied_receipt_outcome_intent_files_verified=159,parent_completed_outcomes_verified=53,worker_PIDs_and_process_groups_absent=len(pids),
  original_findings={'new_originals_passed':26,'new_people':len({r['subject'] for r in originals.values()}),'new_sessions':13,'source_compressed_bytes_receipted':sum(r['source']['entry']['bytes'] for r in originals.values()),'legacy_sessions':197,'combined_sessions':210,'combined_people':199,'combined_image_qc_passed':414,'combined_image_qc_failed':6,'combined_TOF_passed':210,'combined_T1_passed':204,'retained_failures':failed_originals,'scalar_passes_authenticated_from_saved_receipts_not_redecoded':True},
  reference_findings={'new_reference_passed':27,'new_grid_counts':dict(collections.Counter(r['grid_status'] for r in references)),'combined_grid_counts':finalmaskcounts,'combined_content_passed':144,'combined_reference_passed':144,'metadata_exclusions':4,'independent_saved_header_recalculations':28,'reference_rows':references,'remaining_unresolved_subjects':[r['subject'] for r in masks if r['grid_qc_status']=='unresolved']},
  history={'original_mask_checkpoint_and144receipt_hashes_unchanged':True,'prior_known_T1_failures_retained':6,'prior_mask_grid_failure_retained':'sub-454','separate_review_namespace_preserved':True},outcomes=record_results,
  resources={'maximum_saved_worker_elapsed_seconds':max(r['elapsed_seconds'] for r in record_results),'maximum_saved_scalar_workspace_bytes':max(all_receipts[p][0]['decoding_budget']['scalar_workspace_bytes_bound'] for p in originals),'scientific_execution_RSS':'not measured; no claim made'},
  limitations=['This independent audit read saved receipts, source snapshots and348-byte headers already embedded in metadata; it did not open original or mask payload files.','New scalar checks are authenticated execution results; scalar values were not independently recounted in this review.','An image geometry pass does not establish annotation-to-reference grid equivalence. Seven new grid refusals and the prior sub454 refusal remain unresolved.','No anatomy or annotation-semantics review, timestamp availability, negative annotation coverage, scanner/planning/training admission or tolerance change follows.','Continuous downloads remain independent and were not interrupted.'])
except BaseException as e:
 report.update(status='review_failed',error={'type':type(e).__name__,'message':str(e)})
 import traceback;report['traceback']=traceback.format_exc()
finally:
 report['elapsed_seconds']=time.monotonic()-START
 report['input_summary_sha256']=sha(read(BASE/'summary.json'));report['input_receipt_index_sha256']=sha(read(BASE/'receipt-index.json'))
 report['audit_script_sha256']=sha(pathlib.Path(__file__).read_bytes())
 path=OUT/'verification.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'error':report.get('error'),'elapsed_seconds':report['elapsed_seconds'],'verification_sha256':sha(read(path))}))
