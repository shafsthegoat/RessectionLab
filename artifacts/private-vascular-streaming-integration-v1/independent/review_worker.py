"""Route a generated regression set to the staged candidate, with no payloads."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'build/private-vascular-streaming-integration-preparation-v1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        raise AssertionError('No child/native/model/network operation allowed')
    if event=='open' and isinstance(args[0],(str,bytes)):
        assert not os.fsdecode(args[0]).lower().endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin','.npz','.npy','.pkl','.safetensors','.dcm','.h5'))
sys.addaudithook(guard)
import resectionlab
resectionlab.__path__.insert(0,str(PREP/'stage/src/resectionlab'))
from resectionlab import private_vascular_evaluation as candidate
from resectionlab import vascular_contact_streaming as kernel
assert Path(candidate.__file__).resolve()==PREP/'stage/src/resectionlab/private_vascular_evaluation.py'
assert Path(kernel.__file__).resolve()==PREP/'stage/src/resectionlab/vascular_contact_streaming.py'
candidate.ROOT=ROOT  # Sole staging-only relocation for real build output boundary.
import pytest
code=pytest.main(['-q','-p','no:cacheprovider',
    str(ROOT/'tests/test_private_vascular_evaluation.py'),str(ROOT/'tests/test_matched_private_vascular.py'),
    str(PREP/'stage/tests/test_vascular_contact_streaming.py'),str(PREP/'stage/tests/test_private_vascular_streaming.py'),
    str(HERE/'test_independent.py'),'--basetemp='+str(HERE/'tiny-generated-controls-v1')])
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
raise SystemExit(code)
