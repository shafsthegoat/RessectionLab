"""One bounded generated suite; subprocess pattern reused from prior pilot controls."""
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
name=sys.argv[1];out=P/name;out.mkdir(exist_ok=False)
files=[*P.glob('*.py'),*P.glob('stage/resectionlab/*.py'),*P.glob('tests/*.py')]
pins={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
child=out/'generated_child.py'
child.write_text('''import os,sys\nfrom pathlib import Path\nR=Path(sys.argv[1]);scratch=Path(sys.argv[2]).resolve()\nsys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'build/hybrid-train050-transfer-v1'))\ndef guard(event,args):\n if event=='open' and args and isinstance(args[0],(str,bytes)):\n  p=Path(os.fsdecode(args[0])).resolve()\n  if p.is_relative_to(R/'data') or p.is_relative_to(R/'sources') or p.suffix.lower() in ('.dcm','.nii','.gz','.pt','.pth','.ckpt','.psckpt') or (p.suffix=='.npy' and not p.is_relative_to(scratch)) or p==R/'build/hybrid-proposal-balanced-v2/attempt-01/balanced_ridge_top8.json':\n   raise PermissionError('Generated-only control blocked acquired payload: '+str(p))\n if event in ('subprocess.Popen','os.system','os.fork','socket.connect'):raise PermissionError('No child process/network in generated controls')\nsys.addaudithook(guard)\nimport pytest\nraise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(scratch),str(R/'build/hybrid-train050-transfer-v1/tests/test_transfer_source.py')]))\n''')
env=dict(os.environ)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';env['PYTHONDONTWRITEBYTECODE']='1'
cmd=[str(R/'.venv/bin/python'),'-I','-B','-X','pycache_prefix='+str(out/'fresh-pycache'),str(child),str(R),str(out/'generated-tmp')]
started=time.monotonic();timeout=False
with (out/'stdout.log').open('xb') as log:
 try:completed=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30);code=completed.returncode
 except subprocess.TimeoutExpired:timeout=True;code=None
receipt={'command':cmd,'exit_code':code,'elapsed_seconds':time.monotonic()-started,'timeout':timeout,'hard_seconds':30,'threads':1,
 'generated_only':True,'acquired_arrays_or_frozen_model_allowed':False,'files':pins,
 'source_unchanged':all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in pins.items()),
 'observed_child_maxrss':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'log_sha256':hashlib.sha256((out/'stdout.log').read_bytes()).hexdigest()}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}));print((out/'stdout.log').read_text())
raise SystemExit(0 if code==0 and not timeout and receipt['source_unchanged'] else 1)
