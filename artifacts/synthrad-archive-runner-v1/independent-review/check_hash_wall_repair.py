"""Independent hash-boundary wall-expiry control; opaque local bytes only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
root=Path.cwd()
sys.path.insert(0,str(root/'scripts'))
import acquire_synthrad_task1_archive as a
out=root/'build/synthrad-archive-runner-independent-review-v1'
qroot=out/'hash-wall-repair-control'
qroot.mkdir()
m=copy.deepcopy(a.load_manifest())
s=m['sources'][0]
raw=b'opaque-byte-clock-control\0'*(100000)
s.update(bytes=len(raw),expected_md5=hashlib.md5(raw).hexdigest())
q=a.Queue(qroot,m)
target=q.target(s)
target.parent.mkdir(parents=True)
target.write_bytes(raw)
original_sha=hashlib.sha256(raw).hexdigest()
wall=[time.time()]
start_mono=time.monotonic()
deadline=a.phase_deadline(q,start_mono+86400,wall[0]+86400)
assert deadline.wall_end<wall[0]+1801
original_hash=hashlib.sha256
hashed=[]
class JumpHash:
    def __init__(self,*args,**kwargs): self.h=original_hash(*args,**kwargs)
    def update(self,data):
        hashed.append(len(data))
        self.h.update(data)
        wall[0]=deadline.wall_end+1
    def hexdigest(self):return self.h.hexdigest()
refused=False
with patch.object(a.time,'time',lambda:wall[0]),patch.object(a.transfer.hashlib,'sha256',JumpHash):
    try:a.transfer.recover(q,s,deadline)
    except a.io.IntakeDeadline:refused=True
assert refused and hashed==[1048576]
assert time.monotonic()-start_mono<2
assert hashlib.sha256(target.read_bytes()).hexdigest()==original_sha
assert not (q.object_dir(s)/'completion.json').exists()
evidence={'wall_hash_expiry_refused':refused,'sha_bytes_processed_before_stop':sum(hashed),'block_updates':len(hashed),'source_unchanged':True,'completion_published':False,'phase_wall_cap_seconds':1800,'monotonic_elapsed_seconds':time.monotonic()-start_mono,'network_requests':0,'patient_payload_reads':0,'method':'Real unchanged recover/verify/file_hash; independent SHA update wrapper advances only wall clock after first1MiB, before helper next deadline check.'}
(out/'hash-wall-repair-reproduction.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence,indent=2))
