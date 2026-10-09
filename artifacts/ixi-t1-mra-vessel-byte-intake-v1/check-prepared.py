from pathlib import Path
import builtins,contextlib,hashlib,io,json,runpy,sys
R=Path.cwd();P=R/'build/ixi-paired-intake-preparation-v1';B=R/'data/acquisition/ixi-t1-mra-vessel-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runner=P/'run-intake.py';d=P/'prepared-declaration.json';expected=sha(d)
def files():return sorted(str(p.relative_to(B)) for p in B.rglob('*'))
before=files();original_import=builtins.__import__;blocked=[]
def guard(name,*a,**kw):
 if name in ['acquire_public_case','acquire_btc_case','real_intake_io']:
  blocked.append(name);raise AssertionError('transport import forbidden in unreleased/check-only control')
 return original_import(name,*a,**kw)
args=sys.argv;builtins.__import__=guard
try:
 out=io.StringIO();sys.argv=[str(runner),'--declaration',str(d),'--declaration-sha256',expected,'--check-only']
 with contextlib.redirect_stdout(out):runpy.run_path(str(runner),run_name='__main__')
 result=json.loads(out.getvalue());assert result['status']=='prepared_contract_valid' and result['execution_released'] is False
 sys.argv=sys.argv[:-1]
 try:runpy.run_path(str(runner),run_name='__main__')
 except AssertionError as e:assert not str(e),repr(e)
 else:raise AssertionError('false release accepted')
 assert before==files() and not blocked and not (B/'verified').exists() and not (B/'queue.lock').exists()
finally:builtins.__import__=original_import;sys.argv=args
receipt={'schema':'ixi-prepared-release-boundary-controls-v1','status':'pass','runner_sha256':sha(runner),'prepared_declaration_sha256':expected,'check_only':result,'false_release':'refused before transport import, queue creation, payload directory or network','actual_network_requests':0,'source_payload_bytes_read':0,'existing_paths_unchanged':True}
(P/'prepared-controls.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
