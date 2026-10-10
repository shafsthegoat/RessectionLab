import hashlib,json,os,pathlib,resource,signal,subprocess,time
root=pathlib.Path.cwd();base=root/'build/obstruction-opening-proposals-v1'
paths=list((base/'stage').rglob('*.py'))+list((base/'before').glob('*.py'))
pins={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
tests=['-q',str(base/'stage/tests/test_obstruction_opening_proposals.py'),'tests/test_nominal_cavity_proposals.py','tests/test_nominal_cavity_proposals_review.py','tests/test_tool_footprint_opening_proposals.py','tests/test_tool_footprint_candidate_cap.py']
code='import resectionlab; resectionlab.__path__.insert(0, '+repr(str(base/'stage/resectionlab'))+'); import pytest; raise SystemExit(pytest.main('+repr(tests)+'))'
env={**os.environ,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
start=time.monotonic();timeout=False
with (base/'controls-01.log').open('x') as log:
 child=subprocess.Popen([str(root/'.venv/bin/python'),'-c',code],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=child.wait(timeout=30)
 except subprocess.TimeoutExpired:
  timeout=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
record={'exit_code':code,'timed_out':timeout,'elapsed_seconds':time.monotonic()-start,'peak_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'threads':1,'source_pins':pins,'pins_unchanged':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in pins.items()),'scope':'generated native/source controls and pinned historical protocol JSON only; no patient arrays/models/optimizer'}
(base/'controls-01.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));print((base/'controls-01.log').read_text()[-7000:]);raise SystemExit(code)
