import pathlib,sys,os,subprocess,json
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
archive=(pathlib.Path(sys.base_prefix)/'lib'/('python%d%d.zip'%sys.version_info[:2])).resolve()
def no_payload(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
        path=pathlib.Path(os.path.abspath(os.fsdecode(values[0])))
        if path==archive:return
        if path.suffix in ('.npy','.npz','.psckpt','.qckpt','.mat','.zip','.nii','.gz'):
            raise PermissionError('Generated repair control cannot load acquired/learned payloads')
sys.addaudithook(no_payload)
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(HERE/'pytest-temp'),str(HERE.parent/'test_frontier.py')+'::test_stale_or_forged_frontier_and_packet_fail_before_native_query'])
if code:raise SystemExit(code)
output=HERE/'bootstrap';cache=HERE/'absent-bootstrap-cache'
result=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+str(cache),str(HERE.parent/'bootstrap_smoke.py'),'--output',str(output)],timeout=28)
if result.returncode:raise SystemExit(result.returncode)
record=json.loads((output/'receipt.json').read_text())
assert record['status']=='passed' and record['payload_open_attempts']==[]
assert record['native_calls']==record['policy_forwards']==record['optimizer_steps']==0
print('Actual isolated worker bootstrap passed; no scientific payload opens.')
