from pathlib import Path
import ast,copy,hashlib,json,runpy
R=Path.cwd();P=R/'build/tractoinferno-recovery-preparation-v1';classify=runpy.run_path(str(P/'classify.py'))['classify'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e={'bytes':100,'source_url':'https://example.invalid/object?versionId=version','etag_opaque':'"opaque"','s3_version_id':'version'}
h={'intent':{'resume_offset':0},'result':{'status':'integrity_or_scope_refusal','error_type':'BrokenPipeError','error':'[Errno 32] Broken pipe'},'http':{'status':200,'effective_url':e['source_url'],'headers':{'Content-Length':'100','Content-Range':None,'ETag':'"opaque"','x-amz-version-id':'version','Content-Encoding':None}}};cases=[]
def check(name,row,expected,orphan=False,final=False,regular=True,size=20,extra=None):
 history=([extra] if extra else [])+[row];actual=classify(e,history,['active'] if orphan else [],final,regular,size);assert actual==expected,(name,actual);cases.append({'case':name,'status':'pass'})
check('worker_broken_pipe_with_exact_HTTP_and_short_partial',h,'proposed_worker_broken_pipe_transport_reclassification')
for field,bad in [('ETag','"changed"'),('x-amz-version-id','changed'),('Content-Length','101'),('Content-Range','bytes 0-99/100'),('Content-Encoding','gzip')]:
 row=copy.deepcopy(h);row['http']['headers'][field]=bad;check('changed_'+field,row,'exclude_broken_pipe_http_identity_mismatch')
check('active_intent_excluded',h,'exclude_in_flight_or_unreconciled_intent',orphan=True)
check('existing_final_excluded',h,'exclude_local_final_requires_verification',final=True)
check('nonregular_partial_excluded',h,'exclude_partial_type_or_size',regular=False)
check('oversized_partial_excluded',h,'exclude_partial_type_or_size',size=101)
check('full_partial_requires_offline_verification',h,'exclude_full_partial_requires_local_verification',size=100)
row=copy.deepcopy(h);row['http']=None;check('missing_HTTP_stage_excluded',row,'exclude_broken_pipe_without_http_stage_evidence')
for kind,error in [('AcquisitionError','Source annex MD5 mismatch: generated'),('SSLCertVerificationError','certificate failure'),('URLError','<urlopen error certificate failure>')]:
 prior=copy.deepcopy(h);prior['result'].update(error_type=kind,error=error);check('prior_'+kind+'_stays_excluded',h,'exclude_other_error_or_integrity_evidence',extra=prior)
for kind,error in [('URLError','<urlopen error [Errno 8] nodename nor servname provided, or not known>'),('URLError','<urlopen error [Errno 54] Connection reset by peer>'),('TimeoutError','The read operation timed out')]:
 row=copy.deepcopy(h);row['result'].update(status='transport_exhausted',error_type=kind,error=error);check('explicit_transport_'+error,row,'proposed_exhausted_transport_budget_extension')
# Static control for attribution: direct print lives in the collector, not worker.
runner=R/'build/tractoinferno-train-preparation-v1/run-intake.py';tree=ast.parse(runner.read_text());worker=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='process_inner');assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='print' for n in ast.walk(worker));cases.append({'case':'main_progress_logger_not_in_worker_exception_handler','status':'pass'})
report={'schema':'tractoinferno-recovery-classification-controls-v1','status':'pass','classification_source_sha256':sha(P/'classify.py'),'original_runner_sha256':sha(runner),'cases':cases,'actual_network_requests':0,'source_payload_bytes_read':0,'boundary':'Synthetic metadata dictionaries and AST only. No new transfer budget or release follows from these tests.'};(P/'classification-controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'pass','controls':len(cases),'receipt_sha256':sha(P/'classification-controls.json')}))
