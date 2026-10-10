import hashlib,json,os,resource,signal,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
files=[here/name for name in ('comparison_contract.py','select_worker.py','run_owned.py','freeze-runtime.py','test_worker_flow.py')]
def pins():return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=pins();start=time.monotonic();log=here/'controls-01.log'
code="""import sys,torch,resectionlab,pytest
from pathlib import Path
torch.set_num_threads(1)
root=Path.cwd()
sys.path.insert(0,str(root/'tests'))
raise SystemExit(pytest.main(['-q','build/obstruction-opening-select013-comparison-v1/test_worker_flow.py']))
"""
env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with log.open('x') as stream:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-c',code],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=30)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
receipt={'exit_code':code,'elapsed_seconds':time.monotonic()-start,'peak_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'threads':1,'timeout_seconds':30,'source_before':before,'source_after':pins(),'source_unchanged':before==pins(),'scope':'one glue-only fabricated world/checkpoint test; no scientific model, optimizer, native preview or acquired input/checkpoint payload'}
(here/'controls-01.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if not k.startswith('source_')}));print(log.read_text()[-5000:])
raise SystemExit(code)
