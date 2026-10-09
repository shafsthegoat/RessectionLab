"""Stage routing only; all evaluation fixtures are generated and small."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
import resectionlab
resectionlab.__path__.insert(0,str(HERE/'stage/src/resectionlab'))
from resectionlab import private_vascular_evaluation as candidate
candidate.ROOT = ROOT  # Staging-only path relocation; not included in source patch.
assert Path(candidate.__file__).resolve() == HERE/'stage/src/resectionlab/private_vascular_evaluation.py'
PROHIBITED=[]
def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        PROHIBITED.append(event);raise AssertionError('No process/network in tiny integration controls')
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).lower()
        if name.endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin','.npy','.npz','.dcm','.h5')):
            PROHIBITED.append(name);raise AssertionError('No patient/model/array payloads')
sys.addaudithook(guard)
import pytest
result=pytest.main(['-q','-p','no:cacheprovider',
    str(ROOT/'tests/test_private_vascular_evaluation.py'),str(ROOT/'tests/test_matched_private_vascular.py'),
    str(HERE/'stage/tests/test_vascular_contact_streaming.py'),str(HERE/'stage/tests/test_private_vascular_streaming.py'),
    '--basetemp='+str(HERE/'generated-controls-v1')])
assert PROHIBITED == []
raise SystemExit(result)
