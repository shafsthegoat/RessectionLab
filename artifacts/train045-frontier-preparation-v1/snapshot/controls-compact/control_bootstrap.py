import pathlib,sys,os,subprocess,json
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
archive=(pathlib.Path(sys.base_prefix)/'lib'/('python%d%d.zip'%sys.version_info[:2])).resolve()
def no_payload(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
        path=pathlib.Path(os.path.abspath(os.fsdecode(values[0])))
        mode,flags=values[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if path==archive and not writing:return
        if path.suffix in ('.npy','.npz','.psckpt','.qckpt','.mat','.zip','.nii','.gz'):
            raise PermissionError('Generated stack control cannot load acquired/learned payloads')
sys.addaudithook(no_payload)
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(HERE/'pytest-temp'),str(HERE.parent/'recovery-compact/test_compact_output.py')])
if code:raise SystemExit(code)
child=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+str(HERE/'absent-bootstrap-cache'),str(HERE.parent/'recovery-compact/bootstrap_smoke.py'),'--output',str(HERE/'bootstrap')],check=True)
r=json.loads((HERE/'bootstrap/receipt.json').read_text())
assert r['status']=='passed' and r['payload_open_attempts']==[]
assert r['native_calls']==r['policy_forwards']==r['optimizer_steps']==0
print('Lossless24-action transport and isolated worker imports passed; no acquired payload opens.')
