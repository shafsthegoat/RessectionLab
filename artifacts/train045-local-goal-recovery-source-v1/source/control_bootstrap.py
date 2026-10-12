"""Generated-only controls; no patient payloads or native transitions."""
import os,sys,json,resource,signal,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(HERE))
import paths
def boundary(event,args):
    if event.startswith(('socket.','subprocess.')) or event in ('os.fork','os.posix_spawn','os.system'):
        raise PermissionError('Generated controls cannot acquire data or start children')
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).lower()
        if name.endswith(('.npy','.npz','.nii','.nii.gz','.psckpt','.qckpt','.zip','.pth','.pt','.h5')):
            raise PermissionError('Generated controls cannot open saved payloads')
sys.addaudithook(boundary)
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
signal.alarm(55)
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask
def forbidden(*a,**k):raise AssertionError('No native calls in generated recovery controls')
NativeResectionEngine.preview_stroke=forbidden;NativeSpatialTask._transition=forbidden
import pytest
started=time.perf_counter();code=pytest.main(['-q','-p','no:cacheprovider',str(HERE/'test_recovery.py')])
print(json.dumps({'status':'pass' if code==0 else 'failed','exit_code':int(code),
    'elapsed_seconds':time.perf_counter()-started,'peak_self_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'native_calls':0,'acquired_payloads':False,'threads':2}))
sys.exit(code)
