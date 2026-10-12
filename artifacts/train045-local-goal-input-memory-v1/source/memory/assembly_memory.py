"""Fresh-process generated 64-cube input assembly only, extracted from source."""
import ast
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
signal.alarm(50)
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
mode=sys.argv[1]
if mode not in ('old','new'):raise ValueError('Fixed arm required')
path=(ROOT/'artifacts/train045-local-goal-curriculum-interruption-v1/original-policy.py'
      if mode=='old' else HERE.parent/'model/policy.py')
source=path.read_text();tree=ast.parse(source)
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='logits_values')
body=next(n for n in method.body if isinstance(n,ast.Try)).body
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='visible' for t in n.targets))
end=next(i for i,n in enumerate(body) if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='freeze_encoder')
selected=ast.Module(body=body[start:end],type_ignores=[]);ast.fix_missing_locations(selected)
image=torch.full((4,6,64,64,64),.125,dtype=torch.float32)
coverage=torch.ones_like(image,dtype=torch.bool);coverage[:,:,::3]=False
available=torch.tensor([[1,0,1,1,0,1]]*4,dtype=torch.bool)
baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
namespace={'torch':torch,'image':image,'coverage':coverage,'available':available,'batch':4}
started=time.perf_counter();exec(compile(selected,str(path),'exec'),namespace)
elapsed=time.perf_counter()-started
volume=namespace['volume'];peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert volume.shape==(4,18,64,64,64) and volume.dtype==torch.float32 and volume.is_contiguous()
digest=hashlib.sha256(memoryview(volume.numpy()).cast('B')).hexdigest()
print(json.dumps({'mode':mode,'scope':'generated assembly only; no encoder, policy, native task or acquired inputs',
    'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(volume.shape),'volume_sha256':digest,
    'before_peak_self_rss_bytes':baseline,'after_peak_self_rss_bytes':peak,
    'assembly_seconds':elapsed,'threads':torch.get_num_threads()},sort_keys=True))
