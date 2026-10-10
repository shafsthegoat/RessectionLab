"""Independent source-bound generated controls; no actual queue or network access."""
import contextlib,hashlib,io,json,os,runpy,sys,time
from pathlib import Path
ROOT=Path.cwd();OUT=ROOT/'build/tractoinferno-recovery-independent-v1'
CAND=ROOT/'build/tractoinferno-recovery-preparation-v1/adapter-v1'
RUNNER=CAND/'run-recovery.py';TEST=CAND/'test-generated.py'
EXPECTED='2444c11e608dfa84691a9485f1e62e9e4b3454a8f6faf5d79d9d2236a3731a65'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(RUNNER)==EXPECTED
sources=[RUNNER,TEST,CAND/'prepared-declaration.json',ROOT/'build/tractoinferno-recovery-preparation-v1/classify.py',ROOT/'build/tractoinferno-recovery-preparation-v1/proposed-recovery.json',ROOT/'build/tractoinferno-train-preparation-v1/run-intake.py']
helperdir=ROOT/'data/acquisition/tractoinferno-train-v1/source'
helpers={helperdir/n for n in ['acquire_public_case.py','acquire_btc_case.py','real_intake_io.py']}
before={str(p.relative_to(ROOT)):sha(p) for p in sources+sorted(helpers)}
blocked=[]
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if ROOT/'data' in p.parents and p not in helpers:
            blocked.append(str(p));raise PermissionError('Review may read helper source only, no original dataset/journal')
    if event in ('socket.connect','socket.connect_ex','socket.bind','os.kill','os.killpg'):
        blocked.append(event);raise PermissionError('Review has no network or process-signal authority')
sys.addaudithook(guard)
# Run unchanged authored test bodies, redirecting only their output root and exact runner location.
text=TEST.read_text().replace("PREP = ROOT / 'build/tractoinferno-recovery-preparation-v1/adapter-v1'",'PREP = Path('+repr(str(OUT))+')').replace("RUNNER = PREP / 'run-recovery.py'",'RUNNER = Path('+repr(str(RUNNER))+')')
start=time.monotonic()
with (OUT/'generated-tests.log').open('w') as log,contextlib.redirect_stdout(log):
    exec(compile(text,str(TEST),'exec'),{'__name__':'__main__','__file__':str(TEST)})
result=json.loads((OUT/'generated-controls.json').read_text())
assert len(result['controls'])==32 and result['status']=='pass'
# Independent controls below exercise the actual lock guard and full history gate.
import fcntl,tempfile,subprocess,ssl
from unittest.mock import patch
ns=runpy.run_path(str(RUNNER),run_name='independent')
ind=[]
def expect_refusal(f,needle):
    try:f()
    except (RuntimeError,BlockingIOError) as e:
        if needle:assert needle in str(e),str(e)
    else:raise AssertionError('Expected refusal')
root=Path(tempfile.mkdtemp(prefix='independent-',dir=OUT))
for relative in [ns['ORIGINAL']/'queue.lock',Path('predecessor.lock')]:
    p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
proposal={'original_pid':2147483647};old={'predecessor_queue_locks':['predecessor.lock']}
def enter():
    with ns['original_gate'](root,proposal,old):return True
with patch.object(subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','')):
    with ns['original_gate'](root,proposal,old):
        for rel in [ns['ORIGINAL']/'queue.lock',Path('predecessor.lock')]:
            with (root/rel).open('r+') as f:
                expect_refusal(lambda:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB),'')
    # Gate releases own locks only after body exits.
    with (root/ns['ORIGINAL']/'queue.lock').open('r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
ind.append('original_and_predecessor_locks_held_during_body_then_released')
for response in [subprocess.CompletedProcess([],0,'92434\n',''),subprocess.CompletedProcess([],2,'','error')]:
    with patch.object(subprocess,'run',return_value=response):expect_refusal(enter,'original_pid_still_live' if response.returncode==0 else 'process_query_failed')
ind.append('live_pid_and_process_query_error_refuse_without_signal')
# Existing original history hashes are required even if replacement has same byte length.
prep=root/ns['PREP'];prep.mkdir(parents=True)
(prep/'classify.py').write_bytes((ROOT/ns['PREP']/'classify.py').read_bytes())
e={'source_url':'https://example.invalid/frozen','bytes':8,'expected_md5':'a'*32,'expected_git_blob_sha1':None,'path':'x'}
j=root/ns['ORIGINAL']/'attempts'/ns['key'](e)[:2]/ns['key'](e);j.mkdir(parents=True)
intent=j/'01-intent.json';receipt=j/'01-result.json'
intent.write_text('{"resume_offset":0}')
receipt.write_text(json.dumps({'status':'transport_exhausted','error_type':'TimeoutError','error':'The read operation timed out'}))
item={'entry':e,'history_records':[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in [intent,receipt]],'original_attempts_used':1,'partial_stat':None,'classification':'proposed_exhausted_transport_budget_extension'}
ns['check_history'](root,item)
receipt.write_text(receipt.read_text().replace('timed','Timed'))
expect_refusal(lambda:ns['check_history'](root,item),'original_history_changed')
ind.append('same_size_historical_receipt_mutation_refused')
# A TLS error nested in a cause cannot masquerade as a read timeout.
wrapped=TimeoutError('outer');wrapped.__cause__=ssl.SSLCertVerificationError('inner')
assert not ns['transient_error'](wrapped,'http_read')
assert not ns['transient_error'](BrokenPipeError(),'filesystem_flush')
assert not ns['transient_error'](BrokenPipeError(),'filesystem_publish')
ind.append('nested_TLS_and_filesystem_flush_publish_remain_nontransient')
after={str(p.relative_to(ROOT)):sha(p) for p in sources+sorted(helpers)}
assert before==after and not blocked
review={'status':'pass','candidate_sha256':EXPECTED,'author_controls':32,'independent_controls':ind,'elapsed_seconds':time.monotonic()-start,'source_hashes_before':before,'source_hashes_unchanged':before==after,'blocked_accesses':blocked,'actual_network_requests':0,'actual_dataset_or_original_journal_reads':0,'original_process_signals':0,'limits':'Generated response streams and fake queue files only. Runtime terminal identity, exact final selection, disk space and endpoint availability remain execution-time requirements.'}
(OUT/'review.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':review['status'],'author_controls':32,'independent_controls':len(ind),'elapsed_seconds':review['elapsed_seconds'],'review_sha256':sha(OUT/'review.json')}))
