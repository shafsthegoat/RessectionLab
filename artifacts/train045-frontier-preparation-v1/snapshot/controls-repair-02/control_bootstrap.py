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
print('Repaired stale/forged frontier control passed.')
